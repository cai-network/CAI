# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from collections.abc import AsyncGenerator, Iterable, Mapping, Sequence
import ipaddress

import anyio
import httpx
from anyio import create_task_group
from loguru import logger

from cai.shared.topology import Topology
from cai.shared.types.common import NodeId
from cai.shared.types.profiling import NodeIdentity, NodeNetworkInfo
from cai.utils.channels import Sender, channel

REACHABILITY_ATTEMPTS = 3


def _is_probeable_remote_ip(candidate: str) -> bool:
    normalized = str(candidate or "").strip()
    if not normalized:
        return False
    try:
        parsed = ipaddress.ip_address(normalized)
    except ValueError:
        # Hostnames are still acceptable for advertised endpoints.
        return True
    return not (
        parsed.is_loopback
        or parsed.is_unspecified
        or parsed.is_multicast
    )


async def check_reachability(
    target_ip: str,
    target_port: int,
    expected_node_id: NodeId,
    out: dict[NodeId, set[str]],
    client: httpx.AsyncClient,
) -> None:
    """Check if a node is reachable at the given IP and verify its identity."""
    if ":" in target_ip:
        # TODO: use real IpAddress types
        url = f"http://[{target_ip}]:{target_port}/node_id"
    else:
        url = f"http://{target_ip}:{target_port}/node_id"

    remote_node_id = None
    last_error = None

    for _ in range(REACHABILITY_ATTEMPTS):
        try:
            r = await client.get(url)
            if r.status_code != 200:
                await anyio.sleep(1)
                continue

            body = r.text.strip().strip('"')
            if not body:
                await anyio.sleep(1)
                continue

            remote_node_id = NodeId(body)
            break

        # expected failure cases
        except (
            httpx.TimeoutException,
            httpx.NetworkError,
        ):
            await anyio.sleep(1)

        # other failures should be logged on last attempt
        except httpx.HTTPError as e:
            last_error = e
            await anyio.sleep(1)

    if last_error is not None:
        logger.warning(
            f"connect error {type(last_error).__name__} from {target_ip} after {REACHABILITY_ATTEMPTS} attempts; treating as down"
        )

    if remote_node_id is None:
        return

    if remote_node_id != expected_node_id:
        logger.debug(
            f"Discovered node with unexpected node_id; "
            f"ip={target_ip}, expected_node_id={expected_node_id}, "
            f"remote_node_id={remote_node_id}"
        )
        return

    if remote_node_id not in out:
        out[remote_node_id] = set()
    out[remote_node_id].add(f"{target_ip}:{target_port}")


def candidate_targets(
    node_id: NodeId,
    node_network: Mapping[NodeId, NodeNetworkInfo],
    node_identities: Mapping[NodeId, NodeIdentity],
    *,
    default_api_port: int,
) -> Iterable[tuple[str, int]]:
    seen: set[tuple[str, int]] = set()
    identity = node_identities.get(node_id)
    candidate_port = (
        identity.api_port
        if identity is not None and identity.api_port is not None
        else default_api_port
    )
    if identity is not None and identity.api_host:
        if _is_probeable_remote_ip(identity.api_host):
            target = (identity.api_host, candidate_port)
            seen.add(target)
            yield target

    for iface in node_network.get(node_id, NodeNetworkInfo()).interfaces:
        if not _is_probeable_remote_ip(iface.ip_address):
            continue
        target = (iface.ip_address, candidate_port)
        if target in seen:
            continue
        seen.add(target)
        yield target


def _direct_overlay_peer_ids(
    self_node_id: NodeId,
    overlay_peers: Mapping[NodeId, Sequence[NodeId]],
) -> set[NodeId]:
    direct_peers = set(overlay_peers.get(self_node_id, ()))
    for source_node_id, peers in overlay_peers.items():
        if self_node_id in peers:
            direct_peers.add(source_node_id)
    direct_peers.discard(self_node_id)
    return direct_peers


def _overlay_trusted_target(
    node_id: NodeId,
    *,
    direct_overlay_peers: set[NodeId],
    node_identities: Mapping[NodeId, NodeIdentity],
    default_api_port: int,
) -> tuple[str, int] | None:
    if node_id not in direct_overlay_peers:
        return None

    identity = node_identities.get(node_id)
    if identity is None or not identity.api_host:
        return None

    port = (
        identity.api_port
        if identity.api_port is not None and identity.api_port > 0
        else default_api_port
    )
    return (identity.api_host, port)


async def check_reachable(
    topology: Topology,
    self_node_id: NodeId,
    node_network: Mapping[NodeId, NodeNetworkInfo],
    node_identities: Mapping[NodeId, NodeIdentity],
    api_port: int,
    overlay_peers: Mapping[NodeId, Sequence[NodeId]] | None = None,
    *,
    include_overlay_fallback: bool = False,
) -> AsyncGenerator[tuple[str, int, NodeId], None]:
    """Yield (ip, port, node_id) pairs as reachability probes complete.

    When ``include_overlay_fallback`` is enabled, peers with a direct overlay
    relationship may yield their advertised API endpoint even if direct probing
    did not succeed. This is useful as a soft liveness/reachability hint, but
    strict direct topology maintenance should leave it disabled.
    """

    send, recv = channel[tuple[str, int, NodeId]]()
    expected_node_ids: set[NodeId] = set()
    direct_overlay_peers = (
        _direct_overlay_peer_ids(self_node_id, overlay_peers or {})
        if include_overlay_fallback
        else set()
    )

    # these are intentionally httpx's defaults so we can tune them later
    timeout = httpx.Timeout(timeout=5.0)
    limits = httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=5,
    )

    async def _probe(
        target_ip: str,
        target_port: int,
        expected_node_id: NodeId,
        client: httpx.AsyncClient,
        send: Sender[tuple[str, int, NodeId]],
    ) -> None:
        async with send:
            out: defaultdict[NodeId, set[str]] = defaultdict(set)
            await check_reachability(
                target_ip, target_port, expected_node_id, out, client
            )
            if expected_node_id in out:
                await send.send((target_ip, target_port, expected_node_id))

    async with (
        httpx.AsyncClient(timeout=timeout, limits=limits, verify=False) as client,
        create_task_group() as tg,
    ):
        for node_id in topology.list_nodes():
            if node_id == self_node_id:
                continue
            expected_node_ids.add(node_id)
            for target_ip, target_port in candidate_targets(
                node_id,
                node_network,
                node_identities,
                default_api_port=api_port,
            ):
                tg.start_soon(
                    _probe, target_ip, target_port, node_id, client, send.clone()
                )
        send.close()

        discovered_targets: defaultdict[NodeId, set[tuple[str, int]]] = defaultdict(set)
        with recv:
            async for item in recv:
                target_ip, target_port, node_id = item
                discovered_targets[node_id].add((target_ip, target_port))
                yield item

        if not include_overlay_fallback:
            return

        for node_id in expected_node_ids:
            if node_id in discovered_targets:
                continue
            trusted_target = _overlay_trusted_target(
                node_id,
                direct_overlay_peers=direct_overlay_peers,
                node_identities=node_identities,
                default_api_port=api_port,
            )
            if trusted_target is None:
                continue
            yield (*trusted_target, node_id)


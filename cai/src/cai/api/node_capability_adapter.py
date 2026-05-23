# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast
from urllib.parse import urlparse

from cai.shared.types.common import NodeId
from cai.shared.types.profiling import (
    AdvertisedTransportEndpoint,
    MemoryUsage,
    NodeIdentity,
)
from cai_compute_chain.node_capabilities import NodeCapabilityRecord


def worker_identity_state(identity: Any) -> tuple[bool | None, str | None]:
    if isinstance(identity, Mapping):
        worker_enabled = identity.get("workerEnabled")
        reward_address = identity.get("workerRewardAddress")
    else:
        worker_enabled = getattr(identity, "worker_enabled", None)
        reward_address = getattr(identity, "worker_reward_address", None)

    normalized_enabled: bool | None
    if worker_enabled is None:
        normalized_enabled = None
    else:
        normalized_enabled = bool(worker_enabled)

    normalized_reward_address = str(reward_address or "").strip() or None
    return normalized_enabled, normalized_reward_address


def capability_record_node_memory(
    record: NodeCapabilityRecord,
) -> MemoryUsage | None:
    summary = record.resource_summary or {}
    ram_total = resource_summary_int(
        summary,
        "ramBytes",
        "ramTotalBytes",
        "ram_total_bytes",
        "ramTotal",
        "ram_total",
    )
    ram_available = resource_summary_int(
        summary,
        "ramAvailableBytes",
        "availableRamBytes",
        "ram_available_bytes",
        "ramAvailable",
        "ram_available",
    )
    if ram_total is None and ram_available is not None:
        ram_total = ram_available
    if ram_available is None and ram_total is not None:
        ram_available = ram_total
    if ram_total is None or ram_available is None or ram_total <= 0:
        return None
    swap_total = resource_summary_int(
        summary,
        "swapBytes",
        "swapTotalBytes",
        "swap_total_bytes",
        "swapTotal",
        "swap_total",
    ) or 0
    swap_available = resource_summary_int(
        summary,
        "swapAvailableBytes",
        "swap_available_bytes",
        "swapAvailable",
        "swap_available",
    )
    if swap_available is None:
        swap_available = swap_total
    return MemoryUsage.from_bytes(
        ram_total=ram_total,
        ram_available=max(0, ram_available),
        swap_total=max(0, swap_total),
        swap_available=max(0, swap_available),
    )


def resource_summary_int(summary: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key not in summary:
            continue
        value = summary[key]
        if isinstance(value, Mapping):
            for nested_key in ("inBytes", "in_bytes", "bytes"):
                if nested_key not in value:
                    continue
                value = value[nested_key]
                break
            else:
                continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def capability_record_node_identity(
    record: NodeCapabilityRecord,
) -> NodeIdentity:
    api_host: str | None = None
    api_port: int | None = None
    endpoints = capability_record_transport_endpoints(record)
    for url in record.api_urls:
        endpoint = api_endpoint_from_url(url)
        if endpoint is None:
            continue
        endpoints.append(endpoint)
        if api_host is None and endpoint.host not in {"127.0.0.1", "::1", "localhost"}:
            api_host = endpoint.host
            api_port = endpoint.port

    data_endpoint = next(
        (
            endpoint
            for endpoint in endpoints
            if endpoint.purpose == "data" and endpoint.host not in {"0.0.0.0", "::"}
        ),
        None,
    )
    resource_summary = record.resource_summary or {}
    total_vram_bytes = resource_summary_int(
        resource_summary,
        "vramBytes",
        "totalVramBytes",
        "total_vram_bytes",
        "vram_bytes",
    )
    cpu_physical_cores = resource_summary_int(
        resource_summary,
        "cpuCores",
        "cpu_cores",
        "cpuPhysicalCores",
        "cpu_physical_cores",
    )
    cpu_logical_cores = resource_summary_int(
        resource_summary,
        "cpuLogicalCores",
        "cpu_logical_cores",
    )
    return NodeIdentity(
        friendly_name=record.friendly_name or record.node_id,
        api_host=api_host,
        api_port=api_port,
        data_host=data_endpoint.host if data_endpoint is not None else None,
        data_port=data_endpoint.port if data_endpoint is not None else None,
        transport_endpoints=tuple(dedupe_transport_endpoints(endpoints)),
        cpu_physical_cores=cpu_physical_cores,
        cpu_logical_cores=cpu_logical_cores,
        total_vram_bytes=total_vram_bytes,
        worker_enabled=record.worker_enabled,
        relay_enabled=record.relay_enabled,
        worker_reward_address=record.worker_reward_address,
        node_public_key_b64=record.node_public_key_b64,
        node_public_key_address=record.node_public_key_address,
        readiness=dict(record.readiness or {}),
    )


def capability_record_transport_endpoints(
    record: NodeCapabilityRecord,
) -> list[AdvertisedTransportEndpoint]:
    endpoints: list[AdvertisedTransportEndpoint] = []
    for raw in record.data_endpoints or []:
        if not isinstance(raw, Mapping):
            continue
        purpose = str(raw.get("purpose") or "").strip().lower()
        route_type = str(
            raw.get("routeType") or raw.get("route_type") or ""
        ).strip().lower()
        host = str(raw.get("host") or "").strip()
        if (
            purpose not in {"api", "data"}
            or route_type not in {"direct", "overlay", "relay"}
            or not host
            or host in {"0.0.0.0", "::"}
        ):
            continue
        port = raw.get("port")
        try:
            normalized_port = int(port) if port is not None else None
        except (TypeError, ValueError):
            normalized_port = None
        source = str(raw.get("source") or "").strip().lower()
        if source not in {"explicit", "auto", "interface_scan"}:
            source = ""
        endpoints.append(
            AdvertisedTransportEndpoint(
                purpose=cast(Literal["api", "data"], purpose),
                route_type=cast(Literal["direct", "overlay", "relay"], route_type),
                host=host,
                port=normalized_port,
                source=cast(
                    Literal["explicit", "auto", "interface_scan"] | None,
                    source or None,
                ),
                interface_name=raw.get("interfaceName")
                or raw.get("interface_name")
                or None,
            )
        )
    return endpoints


def api_endpoint_from_url(url: str) -> AdvertisedTransportEndpoint | None:
    parsed = urlparse(str(url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    try:
        port = int(parsed.port or (443 if parsed.scheme == "https" else 80))
    except (TypeError, ValueError):
        return None
    host = str(parsed.hostname).strip()
    if not host or host in {"0.0.0.0", "::", "127.0.0.1", "::1", "localhost"}:
        return None
    return AdvertisedTransportEndpoint(
        purpose="api",
        route_type="direct",
        host=host,
        port=port,
        source="auto",
    )


def dedupe_transport_endpoints(
    endpoints: Sequence[AdvertisedTransportEndpoint],
) -> list[AdvertisedTransportEndpoint]:
    seen: set[tuple[str, str, str, int | None]] = set()
    deduped: list[AdvertisedTransportEndpoint] = []
    for endpoint in endpoints:
        key = (endpoint.purpose, endpoint.route_type, endpoint.host, endpoint.port)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(endpoint)
    return deduped


def capability_record_route_peers(record: NodeCapabilityRecord) -> set[NodeId]:
    route_hints = record.route_hints or {}
    peers: set[NodeId] = set()
    for key in ("overlayPeerIds", "overlay_peer_ids", "directPeerIds", "direct_peer_ids"):
        raw = route_hints.get(key)
        if isinstance(raw, Mapping):
            iterable = raw.keys()
        elif isinstance(raw, (list, tuple, set)):
            iterable = raw
        else:
            continue
        for item in iterable:
            peer_id = str(item or "").strip()
            if peer_id:
                peers.add(NodeId(peer_id))
    return peers

# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import re
from collections.abc import Mapping, Sequence
from typing import Any

from cai.shared.types.common import NodeId
from cai.shared.types.profiling import NodeIdentity
from cai_compute_chain.model import CaiNetworkConfig


def api_base_url_from_multiaddr(peer: str, api_port: int) -> str | None:
    ip4_match = re.match(r"^/ip4/([^/]+)", peer)
    if ip4_match:
        return f"http://{ip4_match.group(1)}:{api_port}"

    ip6_match = re.match(r"^/ip6/([^/]+)", peer)
    if ip6_match:
        return f"http://[{ip6_match.group(1)}]:{api_port}"

    dns_match = re.match(r"^/dns(?:4|6)?/([^/]+)", peer)
    if dns_match:
        return f"http://{dns_match.group(1)}:{api_port}"

    return None


def bootstrap_api_base_url_for_node(
    node_id: str,
    *,
    config: Any | None = None,
) -> str | None:
    target_node_id = str(node_id).strip()
    if not target_node_id:
        return None

    network_config = config or CaiNetworkConfig()
    fallback_bootstrap_url: str | None = None
    for peer in network_config.bootstrap_peers:
        peer_node_match = re.search(r"/p2p/([^/]+)$", peer)
        if not peer_node_match:
            if fallback_bootstrap_url is None:
                fallback_bootstrap_url = api_base_url_from_multiaddr(
                    peer,
                    network_config.default_api_port,
                )
            continue
        if peer_node_match.group(1).strip() != target_node_id:
            continue
        return api_base_url_from_multiaddr(peer, network_config.default_api_port)
    return fallback_bootstrap_url


def resolve_local_node_id_from_identities(
    node_identities: dict[NodeId, Any],
    *,
    target_port: int,
) -> str | None:
    matches: list[str] = []
    for node_id, identity in node_identities.items():
        if isinstance(identity, Mapping):
            raw_port = identity.get("apiPort", -1)
        else:
            raw_port = getattr(identity, "api_port", -1)
        try:
            api_port = int(raw_port)
        except (TypeError, ValueError):
            continue
        if api_port == int(target_port):
            matches.append(str(node_id).strip())

    if len(matches) == 1:
        return matches[0]
    return None


def format_host_for_url(host: str) -> str:
    normalized = str(host or "").strip()
    if ":" in normalized and not normalized.startswith("["):
        return f"[{normalized}]"
    return normalized


def build_http_url(host: str, port: int, path: str = "") -> str:
    normalized_path = str(path or "")
    if normalized_path and not normalized_path.startswith("/"):
        normalized_path = f"/{normalized_path}"
    return f"http://{format_host_for_url(host)}:{int(port)}{normalized_path}"


def mapping_transport_http_urls(
    identity: Mapping[str, Any],
    *,
    endpoint_path: str = "",
) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    raw_endpoints = identity.get("transportEndpoints")
    if raw_endpoints is None:
        raw_endpoints = identity.get("transport_endpoints")
    if isinstance(raw_endpoints, Sequence) and not isinstance(
        raw_endpoints, (str, bytes, bytearray)
    ):
        route_priority = {"direct": 0, "overlay": 1, "relay": 2}
        source_priority = {"explicit": 0, "auto": 1, "interface_scan": 2}
        normalized_endpoints: list[tuple[int, int, str, int]] = []
        for raw in raw_endpoints:
            if not isinstance(raw, Mapping):
                continue
            if str(raw.get("purpose") or "").strip().lower() != "api":
                continue
            host = str(raw.get("host") or "").strip()
            route_type = str(
                raw.get("routeType") or raw.get("route_type") or ""
            ).strip().lower()
            source = str(raw.get("source") or "").strip().lower() or "interface_scan"
            try:
                port = int(raw.get("port"))
            except (TypeError, ValueError):
                continue
            if not host or host in {"0.0.0.0", "::"} or port <= 0:
                continue
            normalized_endpoints.append(
                (
                    route_priority.get(route_type, 99),
                    source_priority.get(source, 99),
                    host,
                    port,
                )
            )
        for _route_rank, _source_rank, host, port in sorted(normalized_endpoints):
            url = build_http_url(host, port, endpoint_path)
            if url in seen:
                continue
            seen.add(url)
            urls.append(url)

    host = str(identity.get("apiHost") or "").strip()
    try:
        port = int(identity.get("apiPort"))
    except (TypeError, ValueError):
        port = -1
    if host and host not in {"0.0.0.0", "::"} and port > 0:
        fallback_url = build_http_url(host, port, endpoint_path)
        if fallback_url not in seen:
            urls.append(fallback_url)
    return urls


def candidate_http_urls_from_identity(
    identity: Any,
    *,
    endpoint_path: str = "",
) -> list[str]:
    if isinstance(identity, NodeIdentity):
        urls: list[str] = []
        seen: set[str] = set()
        for endpoint in identity.transport_endpoints_for(
            purpose="api",
            require_port=True,
        ):
            assert endpoint.port is not None
            url = build_http_url(endpoint.host, endpoint.port, endpoint_path)
            if url in seen:
                continue
            seen.add(url)
            urls.append(url)
        host = str(identity.api_host or "").strip()
        if host and host not in {"0.0.0.0", "::"} and identity.api_port is not None:
            fallback_url = build_http_url(host, identity.api_port, endpoint_path)
            if fallback_url not in seen:
                urls.append(fallback_url)
        return urls

    if isinstance(identity, Mapping):
        return mapping_transport_http_urls(identity, endpoint_path=endpoint_path)
    return []


def cai_summary_urls_by_node_id(
    *,
    node_identities: dict[NodeId, Any],
    local_port: int,
    local_node_id: str | None = None,
) -> dict[str, str]:
    urls: dict[str, str] = {}
    resolved_local_node_id = (
        (local_node_id or "").strip()
        or resolve_local_node_id_from_identities(
            node_identities,
            target_port=local_port,
        )
    )
    if resolved_local_node_id:
        urls[resolved_local_node_id] = f"http://127.0.0.1:{local_port}/v1/cai/summary"

    for node_id, identity in node_identities.items():
        normalized_node_id = str(node_id).strip()
        if not normalized_node_id or normalized_node_id == resolved_local_node_id:
            continue
        candidates = candidate_http_urls_from_identity(
            identity,
            endpoint_path="/v1/cai/summary",
        )
        if candidates:
            urls[normalized_node_id] = candidates[0]
    return urls

# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Callable
from typing import Any
from urllib.parse import urlencode, urlsplit

from .transport_endpoints import (
    build_http_url,
    candidate_identity_http_urls,
    identity_transport_endpoints,
)


def resolve_local_node_id_from_state_payload(
    state_payload: dict[str, Any],
    cai_url: str,
) -> str | None:
    identities = state_payload.get("nodeIdentities") or {}
    if not isinstance(identities, dict):
        return None

    target_port = urlsplit(cai_url.rstrip("/")).port
    loopback_matches = [
        str(node_id).strip()
        for node_id, info in identities.items()
        if isinstance(info, dict)
        and (
            info.get("apiHost") is None
            or str(info.get("apiHost") or "").strip().lower()
            in {"", "127.0.0.1", "localhost", "::1"}
        )
    ]
    if len(loopback_matches) == 1:
        return loopback_matches[0]

    if target_port is not None:
        matches: list[str] = []
        for node_id, info in identities.items():
            if not isinstance(info, dict):
                continue
            try:
                api_port = int(info.get("apiPort", -1))
            except (TypeError, ValueError):
                continue
            if api_port == int(target_port):
                matches.append(str(node_id).strip())
        if len(matches) == 1:
            return matches[0]

    if len(identities) == 1:
        return str(next(iter(identities.keys()))).strip() or None
    return None


def cai_summary_urls_by_node_id(
    cai_url: str,
    state_payload: dict[str, Any],
) -> dict[str, str]:
    identities = state_payload.get("nodeIdentities") or {}
    if not isinstance(identities, dict):
        return {}

    urls: dict[str, str] = {}
    local_node_id = resolve_local_node_id_from_state_payload(state_payload, cai_url)
    if local_node_id:
        urls[local_node_id] = cai_url.rstrip("/") + "/v1/cai/summary"

    for node_id, identity in identities.items():
        normalized_node_id = str(node_id).strip()
        if not normalized_node_id or normalized_node_id == local_node_id:
            continue
        if not isinstance(identity, dict):
            continue
        candidates = candidate_identity_http_urls(
            identity,
            endpoint_path="/v1/cai/summary",
        )
        if candidates:
            urls[normalized_node_id] = candidates[0]
    return urls


def cai_api_urls_by_node_id(
    cai_url: str,
    state_payload: dict[str, Any],
    *,
    list_node_capabilities_func: Callable[[], list[Any]],
) -> dict[str, list[str]]:
    identities = state_payload.get("nodeIdentities") or {}
    if not isinstance(identities, dict):
        return {}

    urls: dict[str, list[str]] = {}
    local_node_id = resolve_local_node_id_from_state_payload(state_payload, cai_url)
    if local_node_id:
        urls[local_node_id] = [cai_url.rstrip("/")]

    for node_id, identity in identities.items():
        normalized_node_id = str(node_id).strip()
        if not normalized_node_id:
            continue
        candidates = []
        if isinstance(identity, dict):
            candidates = candidate_identity_http_urls(identity)
            candidates.extend(
                cai_owned_overlay_peer_urls_for_target(
                    normalized_node_id,
                    identities,
                    state_payload,
                )
            )
        if normalized_node_id == local_node_id:
            urls[normalized_node_id] = list(
                dict.fromkeys([*(urls.get(normalized_node_id) or []), *candidates])
            )
        elif candidates:
            urls[normalized_node_id] = list(dict.fromkeys(candidates))
    for record in list_node_capabilities_func():
        normalized_node_id = str(record.node_id or "").strip()
        if not normalized_node_id:
            continue
        capability_urls = [
            str(url or "").strip().rstrip("/")
            for url in (record.api_urls or [])
            if str(url or "").strip()
        ]
        if not capability_urls:
            continue
        urls[normalized_node_id] = list(
            dict.fromkeys([*(urls.get(normalized_node_id) or []), *capability_urls])
        )
    return urls


def candidate_cai_chat_base_urls(
    cai_url: str,
    *,
    load_cai_state_payload_func: Callable[[str], dict[str, Any] | None],
) -> list[str]:
    normalized_root = cai_url.rstrip("/")
    candidates = [normalized_root]
    state_payload = load_cai_state_payload_func(normalized_root) or {}
    node_identities = state_payload.get("nodeIdentities") or {}
    if not isinstance(node_identities, dict):
        return candidates

    for identity in node_identities.values():
        if not isinstance(identity, dict):
            continue
        for candidate in candidate_identity_http_urls(identity):
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def cai_owned_overlay_peer_urls_for_target(
    target_node_id: str,
    identities: dict[str, Any],
    state_payload: dict[str, Any],
) -> list[str]:
    target = str(target_node_id or "").strip()
    if not target:
        return []

    urls: list[str] = []
    for relay_node_id, relay_identity in identities.items():
        relay_id = str(relay_node_id or "").strip()
        if not relay_id or relay_id == target:
            continue
        if not isinstance(relay_identity, dict):
            continue
        if not identity_bool(relay_identity, "relayEnabled", "relay_enabled"):
            continue
        if not relay_has_overlay_path_to_target(relay_id, target, state_payload):
            continue
        relay_role = cai_owned_overlay_relay_role(relay_identity)
        for relay_url in direct_cai_api_urls_for_overlay_relay(relay_identity):
            query = urlencode({"targetNodeId": target, "relayRole": relay_role})
            urls.append(f"cai-overlay:{relay_url}?{query}")
    return list(dict.fromkeys(urls))


def direct_cai_api_urls_for_overlay_relay(identity: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for endpoint in identity_transport_endpoints(
        identity,
        purpose="api",
        route_types=("direct",),
        require_port=True,
    ):
        host = str(endpoint.get("host") or "").strip()
        port = endpoint.get("port")
        if not host or host in {"0.0.0.0", "::"} or port is None or int(port) <= 0:
            continue
        urls.append(build_http_url(host, int(port)))
    return list(dict.fromkeys(urls))


def relay_has_overlay_path_to_target(
    relay_node_id: str,
    target_node_id: str,
    state_payload: dict[str, Any],
) -> bool:
    overlay_peers = state_payload.get("overlayPeers")
    if overlay_peers is None:
        overlay_peers = state_payload.get("overlay_peers")
    if not isinstance(overlay_peers, dict) or not overlay_peers:
        return True

    relay = str(relay_node_id or "").strip()
    target = str(target_node_id or "").strip()
    if not relay or not target:
        return False
    relay_peers = overlay_peer_set(overlay_peers.get(relay))
    target_peers = overlay_peer_set(overlay_peers.get(target))
    return target in relay_peers or relay in target_peers


def overlay_peer_set(value: Any) -> set[str]:
    if isinstance(value, (str, bytes, bytearray)):
        return {str(value).strip()} if str(value).strip() else set()
    if not isinstance(value, (list, tuple, set)):
        return set()
    return {str(item or "").strip() for item in value if str(item or "").strip()}


def cai_owned_overlay_relay_role(identity: dict[str, Any]) -> str:
    if identity_bool(identity, "workerEnabled", "worker_enabled"):
        return "ordinary"
    return "bootstrap"


def identity_bool(identity: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        if key not in identity:
            continue
        raw = identity.get(key)
        if isinstance(raw, str):
            return raw.strip().lower() not in {"", "0", "false", "no", "off"}
        return bool(raw)
    return False

# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from .transport_endpoints import identity_transport_endpoints

_SEGMENT_PRIORITY = {"direct": 0, "overlay": 1}


def _normalized_node_ids(node_ids: Sequence[str] | set[str]) -> list[str]:
    return sorted(
        {
            str(node_id).strip()
            for node_id in node_ids
            if str(node_id).strip()
        }
    )


def _state_topology(state_payload: dict[str, Any]) -> dict[str, Any]:
    topology = state_payload.get("topology")
    return topology if isinstance(topology, Mapping) else {}


def _state_node_identities(state_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = state_payload.get("nodeIdentities")
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(node_id): identity
        for node_id, identity in raw.items()
        if str(node_id).strip() and isinstance(identity, Mapping)
    }


def _edge_has_socket_connection(edge_payload: Any) -> bool:
    if not isinstance(edge_payload, list) or not edge_payload:
        return False
    return any(
        isinstance(edge_item, Mapping)
        and isinstance(edge_item.get("sinkMultiaddr"), Mapping)
        for edge_item in edge_payload
    )


def socket_adjacency(
    state_payload: dict[str, Any],
    visible_node_ids: Sequence[str] | set[str],
) -> dict[str, set[str]]:
    normalized_node_ids = _normalized_node_ids(visible_node_ids)
    adjacency = {node_id: set() for node_id in normalized_node_ids}
    node_id_set = set(normalized_node_ids)
    connections = _state_topology(state_payload).get("connections")
    if not isinstance(connections, Mapping):
        return adjacency

    for source_node_id, sink_payload in connections.items():
        normalized_source_node_id = str(source_node_id or "").strip()
        if normalized_source_node_id not in node_id_set:
            continue
        if not isinstance(sink_payload, Mapping):
            continue
        for sink_node_id, edge_payload in sink_payload.items():
            normalized_sink_node_id = str(sink_node_id or "").strip()
            if (
                normalized_sink_node_id not in node_id_set
                or normalized_sink_node_id == normalized_source_node_id
            ):
                continue
            if _edge_has_socket_connection(edge_payload):
                adjacency[normalized_source_node_id].add(normalized_sink_node_id)
    return adjacency


def overlay_adjacency(
    state_payload: dict[str, Any],
    visible_node_ids: Sequence[str] | set[str],
) -> dict[str, set[str]]:
    normalized_node_ids = _normalized_node_ids(visible_node_ids)
    adjacency = {node_id: set() for node_id in normalized_node_ids}
    node_id_set = set(normalized_node_ids)
    overlay_peers = state_payload.get("overlayPeers")
    if not isinstance(overlay_peers, Mapping):
        return adjacency

    for source_node_id, peer_payload in overlay_peers.items():
        normalized_source_node_id = str(source_node_id or "").strip()
        if normalized_source_node_id not in node_id_set:
            continue
        if isinstance(peer_payload, Mapping):
            peer_iterable = peer_payload.keys()
        elif isinstance(peer_payload, (list, tuple, set)):
            peer_iterable = peer_payload
        else:
            continue
        for peer_node_id in peer_iterable:
            normalized_peer_node_id = str(peer_node_id or "").strip()
            if (
                normalized_peer_node_id not in node_id_set
                or normalized_peer_node_id == normalized_source_node_id
            ):
                continue
            adjacency[normalized_source_node_id].add(normalized_peer_node_id)
    return adjacency


def relay_capable_node_ids(state_payload: dict[str, Any]) -> list[str]:
    relay_node_ids: list[str] = []
    for node_id, identity in _state_node_identities(state_payload).items():
        relay_enabled = identity.get("relayEnabled")
        if relay_enabled is None:
            relay_enabled = identity.get("relay_enabled")
        relay_endpoints = identity_transport_endpoints(
            identity,
            route_types=["relay"],
        )
        if bool(relay_enabled) or bool(relay_endpoints):
            relay_node_ids.append(node_id)
    return sorted(dict.fromkeys(relay_node_ids))


def _best_segment_type(
    source_node_id: str,
    sink_node_id: str,
    direct_adjacency: Mapping[str, set[str]],
    overlay_links: Mapping[str, set[str]],
) -> str | None:
    if sink_node_id in direct_adjacency.get(source_node_id, set()):
        return "direct"
    if sink_node_id in overlay_links.get(source_node_id, set()):
        return "overlay"
    return None


def _relay_tie_break_score(
    source_node_id: str,
    sink_node_id: str,
    relay_node_id: str,
) -> int:
    digest = hashlib.sha256(
        f"{source_node_id}\0{sink_node_id}\0{relay_node_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _relay_routes_for_pair(
    *,
    source_node_id: str,
    sink_node_id: str,
    participant_node_ids: set[str],
    relay_node_ids: Sequence[str],
    direct_adjacency: Mapping[str, set[str]],
    overlay_links: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    for relay_node_id in relay_node_ids:
        if relay_node_id in {source_node_id, sink_node_id}:
            continue

        source_segment_type = _best_segment_type(
            source_node_id,
            relay_node_id,
            direct_adjacency,
            overlay_links,
        )
        if source_segment_type is None:
            continue
        sink_segment_type = _best_segment_type(
            relay_node_id,
            sink_node_id,
            direct_adjacency,
            overlay_links,
        )
        if sink_segment_type is None:
            continue

        routes.append(
            {
                "sourceNodeId": source_node_id,
                "transitNodeId": relay_node_id,
                "sinkNodeId": sink_node_id,
                "pathNodeIds": [source_node_id, relay_node_id, sink_node_id],
                "sourceSegmentType": source_segment_type,
                "sinkSegmentType": sink_segment_type,
                "transitParticipates": relay_node_id in participant_node_ids,
                "candidateOnly": True,
            }
        )

    routes.sort(
        key=lambda item: _relay_route_base_score(
            item,
            source_node_id=source_node_id,
            sink_node_id=sink_node_id,
        )
    )
    return routes


def _relay_route_base_score(
    route: Mapping[str, Any],
    *,
    source_node_id: str,
    sink_node_id: str,
) -> tuple[int, int, int, int, str]:
    transit_node_id = str(route.get("transitNodeId") or "").strip()
    return (
        _SEGMENT_PRIORITY.get(str(route.get("sourceSegmentType") or ""), 99),
        _SEGMENT_PRIORITY.get(str(route.get("sinkSegmentType") or ""), 99),
        1 if bool(route.get("transitParticipates")) else 0,
        _relay_tie_break_score(source_node_id, sink_node_id, transit_node_id),
        transit_node_id,
    )


def _best_relay_route(
    *,
    source_node_id: str,
    sink_node_id: str,
    participant_node_ids: set[str],
    relay_node_ids: Sequence[str],
    direct_adjacency: Mapping[str, set[str]],
    overlay_links: Mapping[str, set[str]],
) -> dict[str, Any] | None:
    routes = _relay_routes_for_pair(
        source_node_id=source_node_id,
        sink_node_id=sink_node_id,
        participant_node_ids=participant_node_ids,
        relay_node_ids=relay_node_ids,
        direct_adjacency=direct_adjacency,
        overlay_links=overlay_links,
    )
    return routes[0] if routes else None


def llama_cpp_reachable_targets_by_source(
    state_payload: dict[str, Any],
    visible_node_ids: Sequence[str] | set[str],
) -> dict[str, set[str]]:
    normalized_node_ids = _normalized_node_ids(visible_node_ids)
    visible_node_id_set = set(normalized_node_ids)
    direct_adjacency = socket_adjacency(state_payload, normalized_node_ids)
    overlay_links = overlay_adjacency(state_payload, normalized_node_ids)
    relay_node_ids = relay_capable_node_ids(state_payload)
    reachable_targets = {
        node_id: set(direct_adjacency.get(node_id, set()))
        for node_id in normalized_node_ids
    }

    for source_node_id in normalized_node_ids:
        for sink_node_id in normalized_node_ids:
            if (
                sink_node_id == source_node_id
                or sink_node_id in reachable_targets[source_node_id]
            ):
                continue
            relay_route = _best_relay_route(
                source_node_id=source_node_id,
                sink_node_id=sink_node_id,
                participant_node_ids=visible_node_id_set,
                relay_node_ids=relay_node_ids,
                direct_adjacency=direct_adjacency,
                overlay_links=overlay_links,
            )
            if relay_route is not None:
                reachable_targets[source_node_id].add(sink_node_id)

    return reachable_targets


def relay_route_candidates(
    state_payload: dict[str, Any],
    participant_node_ids: Sequence[str] | set[str],
    *,
    include_alternatives: bool = False,
) -> list[dict[str, Any]]:
    normalized_participant_node_ids = _normalized_node_ids(participant_node_ids)
    participant_node_id_set = set(normalized_participant_node_ids)
    if len(normalized_participant_node_ids) <= 1:
        return []

    visible_node_ids = set(normalized_participant_node_ids)
    visible_node_ids.update(_state_node_identities(state_payload).keys())
    topology_nodes = _state_topology(state_payload).get("nodes")
    if isinstance(topology_nodes, list):
        visible_node_ids.update(
            str(node_id.get("id") if isinstance(node_id, Mapping) else node_id).strip()
            for node_id in topology_nodes
            if str(
                node_id.get("id") if isinstance(node_id, Mapping) else node_id
            ).strip()
        )
    overlay_peers = state_payload.get("overlayPeers")
    if isinstance(overlay_peers, Mapping):
        visible_node_ids.update(str(node_id).strip() for node_id in overlay_peers.keys())
        for peer_payload in overlay_peers.values():
            if isinstance(peer_payload, Mapping):
                peer_iterable = peer_payload.keys()
            elif isinstance(peer_payload, (list, tuple, set)):
                peer_iterable = peer_payload
            else:
                continue
            visible_node_ids.update(
                str(peer_node_id).strip()
                for peer_node_id in peer_iterable
                if str(peer_node_id).strip()
            )

    normalized_visible_node_ids = _normalized_node_ids(visible_node_ids)
    direct_adjacency = socket_adjacency(state_payload, normalized_visible_node_ids)
    participant_direct_adjacency = socket_adjacency(
        state_payload, normalized_participant_node_ids
    )
    overlay_links = overlay_adjacency(state_payload, normalized_visible_node_ids)
    relay_node_ids = relay_capable_node_ids(state_payload)

    relay_routes: list[dict[str, Any]] = []
    transit_route_counts: dict[str, int] = {}
    for source_node_id in normalized_participant_node_ids:
        for sink_node_id in normalized_participant_node_ids:
            if (
                sink_node_id == source_node_id
                or sink_node_id in participant_direct_adjacency.get(source_node_id, set())
            ):
                continue
            pair_routes = _relay_routes_for_pair(
                source_node_id=source_node_id,
                sink_node_id=sink_node_id,
                participant_node_ids=participant_node_id_set,
                relay_node_ids=relay_node_ids,
                direct_adjacency=direct_adjacency,
                overlay_links=overlay_links,
            )
            if include_alternatives:
                relay_routes.extend(pair_routes)
                continue
            relay_route = _least_loaded_relay_route(
                pair_routes,
                source_node_id=source_node_id,
                sink_node_id=sink_node_id,
                transit_route_counts=transit_route_counts,
            )
            if relay_route is None:
                continue
            transit_node_id = str(relay_route.get("transitNodeId") or "").strip()
            transit_route_counts[transit_node_id] = (
                transit_route_counts.get(transit_node_id, 0) + 1
            )
            relay_routes.append(relay_route)

    relay_routes.sort(
        key=lambda item: (
            str(item.get("sourceNodeId") or ""),
            str(item.get("sinkNodeId") or ""),
            str(item.get("transitNodeId") or ""),
        )
    )
    return relay_routes


def _least_loaded_relay_route(
    routes: Sequence[Mapping[str, Any]],
    *,
    source_node_id: str,
    sink_node_id: str,
    transit_route_counts: Mapping[str, int],
) -> dict[str, Any] | None:
    if not routes:
        return None
    selected = min(
        routes,
        key=lambda item: (
            _SEGMENT_PRIORITY.get(str(item.get("sourceSegmentType") or ""), 99),
            _SEGMENT_PRIORITY.get(str(item.get("sinkSegmentType") or ""), 99),
            1 if bool(item.get("transitParticipates")) else 0,
            int(transit_route_counts.get(str(item.get("transitNodeId") or ""), 0)),
            _relay_tie_break_score(
                source_node_id,
                sink_node_id,
                str(item.get("transitNodeId") or "").strip(),
            ),
            str(item.get("transitNodeId") or "").strip(),
        ),
    )
    return dict(selected)


def relay_coordinator_candidate_node_ids(
    state_payload: dict[str, Any],
    participant_node_ids: Sequence[str] | set[str],
) -> list[str]:
    normalized_participant_node_ids = _normalized_node_ids(participant_node_ids)
    participant_node_id_set = set(normalized_participant_node_ids)
    if len(normalized_participant_node_ids) <= 1:
        return normalized_participant_node_ids

    direct_adjacency = socket_adjacency(
        state_payload,
        normalized_participant_node_ids,
    )
    relay_routes = relay_route_candidates(
        state_payload,
        normalized_participant_node_ids,
    )
    relay_reachable_targets: dict[str, set[str]] = {
        node_id: set() for node_id in normalized_participant_node_ids
    }
    for route in relay_routes:
        source_node_id = str(route.get("sourceNodeId") or "").strip()
        sink_node_id = str(route.get("sinkNodeId") or "").strip()
        if (
            source_node_id in participant_node_id_set
            and sink_node_id in participant_node_id_set
        ):
            relay_reachable_targets[source_node_id].add(sink_node_id)

    return [
        node_id
        for node_id in normalized_participant_node_ids
        if relay_reachable_targets.get(node_id)
        if all(
            other_node_id == node_id
            or other_node_id in direct_adjacency.get(node_id, set())
            or other_node_id in relay_reachable_targets.get(node_id, set())
            for other_node_id in participant_node_id_set
        )
    ]

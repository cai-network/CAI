# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import hashlib
from collections.abc import Mapping, Sequence

from cai.shared.topology import Topology
from cai.shared.types.common import NodeId
from cai.shared.types.profiling import NodeIdentity
from cai.shared.types.topology import SocketConnection

_SEGMENT_PRIORITY = {"direct": 0, "overlay": 1}


def socket_adjacency(
    topology: Topology,
    visible_nodes: set[NodeId],
) -> dict[NodeId, set[NodeId]]:
    adjacency: dict[NodeId, set[NodeId]] = {
        node_id: set() for node_id in visible_nodes
    }
    for connection in topology.list_connections():
        if (
            connection.source not in visible_nodes
            or connection.sink not in visible_nodes
            or connection.source == connection.sink
            or not isinstance(connection.edge, SocketConnection)
        ):
            continue
        adjacency[connection.source].add(connection.sink)
    return adjacency


def overlay_adjacency(
    visible_nodes: set[NodeId],
    overlay_peers: Mapping[NodeId, Sequence[NodeId]] | None,
) -> dict[NodeId, set[NodeId]]:
    adjacency: dict[NodeId, set[NodeId]] = {
        node_id: set() for node_id in visible_nodes
    }
    if not overlay_peers:
        return adjacency

    for source_node_id, peer_node_ids in overlay_peers.items():
        if source_node_id not in visible_nodes:
            continue
        for peer_node_id in peer_node_ids:
            if peer_node_id not in visible_nodes or peer_node_id == source_node_id:
                continue
            adjacency[source_node_id].add(peer_node_id)
    return adjacency


def relay_capable_node_ids(
    node_identities: Mapping[NodeId, NodeIdentity] | None,
) -> set[NodeId]:
    if not node_identities:
        return set()

    relay_node_ids: set[NodeId] = set()
    for node_id, identity in node_identities.items():
        relay_endpoints = identity.transport_endpoints_for(
            route_types=["relay"]
        )
        if bool(getattr(identity, "relay_enabled", False)) or relay_endpoints:
            relay_node_ids.add(node_id)
    return relay_node_ids


def _best_segment_type(
    source_node_id: NodeId,
    sink_node_id: NodeId,
    direct_adjacency: Mapping[NodeId, set[NodeId]],
    overlay_links: Mapping[NodeId, set[NodeId]],
) -> str | None:
    if sink_node_id in direct_adjacency.get(source_node_id, set()):
        return "direct"
    if sink_node_id in overlay_links.get(source_node_id, set()):
        return "overlay"
    return None


def _relay_tie_break_score(
    source_node_id: NodeId,
    sink_node_id: NodeId,
    relay_node_id: NodeId,
) -> int:
    digest = hashlib.sha256(
        f"{source_node_id}\0{sink_node_id}\0{relay_node_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _relay_routes_for_pair(
    *,
    source_node_id: NodeId,
    sink_node_id: NodeId,
    participant_node_ids: set[NodeId],
    relay_node_ids: set[NodeId],
    direct_adjacency: Mapping[NodeId, set[NodeId]],
    overlay_links: Mapping[NodeId, set[NodeId]],
) -> list[dict[str, object]]:
    routes: list[dict[str, object]] = []
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
                "sourceNodeId": str(source_node_id),
                "transitNodeId": str(relay_node_id),
                "sinkNodeId": str(sink_node_id),
                "pathNodeIds": [
                    str(source_node_id),
                    str(relay_node_id),
                    str(sink_node_id),
                ],
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
    route: Mapping[str, object],
    *,
    source_node_id: NodeId,
    sink_node_id: NodeId,
) -> tuple[int, int, int, int, str]:
    transit_node_id = NodeId(str(route.get("transitNodeId") or "").strip())
    return (
        _SEGMENT_PRIORITY.get(str(route.get("sourceSegmentType") or ""), 99),
        _SEGMENT_PRIORITY.get(str(route.get("sinkSegmentType") or ""), 99),
        1 if bool(route.get("transitParticipates")) else 0,
        _relay_tie_break_score(source_node_id, sink_node_id, transit_node_id),
        str(transit_node_id),
    )


def _best_relay_route(
    *,
    source_node_id: NodeId,
    sink_node_id: NodeId,
    participant_node_ids: set[NodeId],
    relay_node_ids: set[NodeId],
    direct_adjacency: Mapping[NodeId, set[NodeId]],
    overlay_links: Mapping[NodeId, set[NodeId]],
) -> dict[str, object] | None:
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
    topology: Topology,
    overlay_peers: Mapping[NodeId, Sequence[NodeId]] | None,
    node_identities: Mapping[NodeId, NodeIdentity] | None,
    visible_nodes: set[NodeId],
) -> dict[NodeId, set[NodeId]]:
    route_nodes = set(visible_nodes)
    if overlay_peers:
        route_nodes.update(overlay_peers.keys())
        for peer_node_ids in overlay_peers.values():
            route_nodes.update(peer_node_ids)
    if node_identities:
        route_nodes.update(node_identities.keys())

    direct_adjacency = socket_adjacency(topology, route_nodes)
    overlay_links = overlay_adjacency(route_nodes, overlay_peers)
    relay_node_ids = relay_capable_node_ids(node_identities)
    reachable_targets: dict[NodeId, set[NodeId]] = {
        node_id: set(direct_adjacency.get(node_id, set())).intersection(visible_nodes)
        for node_id in visible_nodes
    }

    for source_node_id in sorted(visible_nodes, key=str):
        for sink_node_id in sorted(visible_nodes, key=str):
            if (
                sink_node_id == source_node_id
                or sink_node_id in reachable_targets[source_node_id]
            ):
                continue
            relay_route = _best_relay_route(
                source_node_id=source_node_id,
                sink_node_id=sink_node_id,
                participant_node_ids=visible_nodes,
                relay_node_ids=relay_node_ids,
                direct_adjacency=direct_adjacency,
                overlay_links=overlay_links,
            )
            if relay_route is not None:
                reachable_targets[source_node_id].add(sink_node_id)

    return reachable_targets


def relay_route_candidates(
    topology: Topology,
    overlay_peers: Mapping[NodeId, Sequence[NodeId]] | None,
    node_identities: Mapping[NodeId, NodeIdentity] | None,
    participant_node_ids: Sequence[NodeId],
    *,
    include_alternatives: bool = False,
) -> list[dict[str, object]]:
    participant_nodes = {
        node_id for node_id in participant_node_ids if str(node_id).strip()
    }
    if len(participant_nodes) <= 1:
        return []

    direct_adjacency = socket_adjacency(topology, participant_nodes)
    visible_nodes = set(participant_nodes)
    visible_nodes.update(node_id for node_id in topology.list_nodes())
    if overlay_peers:
        visible_nodes.update(overlay_peers.keys())
        for peer_node_ids in overlay_peers.values():
            visible_nodes.update(peer_node_ids)
    if node_identities:
        visible_nodes.update(node_identities.keys())
    visible_direct_adjacency = socket_adjacency(topology, visible_nodes)
    overlay_links = overlay_adjacency(visible_nodes, overlay_peers)
    relay_node_ids = relay_capable_node_ids(node_identities)

    relay_routes: list[dict[str, object]] = []
    transit_route_counts: dict[str, int] = {}
    for source_node_id in sorted(participant_nodes, key=str):
        for sink_node_id in sorted(participant_nodes, key=str):
            if (
                sink_node_id == source_node_id
                or sink_node_id in direct_adjacency.get(source_node_id, set())
            ):
                continue
            pair_routes = _relay_routes_for_pair(
                source_node_id=source_node_id,
                sink_node_id=sink_node_id,
                participant_node_ids=participant_nodes,
                relay_node_ids=relay_node_ids,
                direct_adjacency=visible_direct_adjacency,
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
    routes: Sequence[Mapping[str, object]],
    *,
    source_node_id: NodeId,
    sink_node_id: NodeId,
    transit_route_counts: Mapping[str, int],
) -> dict[str, object] | None:
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
                NodeId(str(item.get("transitNodeId") or "").strip()),
            ),
            str(item.get("transitNodeId") or "").strip(),
        ),
    )
    return dict(selected)


def relay_coordinator_candidate_node_ids(
    topology: Topology,
    overlay_peers: Mapping[NodeId, Sequence[NodeId]] | None,
    node_identities: Mapping[NodeId, NodeIdentity] | None,
    participant_node_ids: Sequence[NodeId],
) -> list[NodeId]:
    participant_nodes = [
        node_id for node_id in participant_node_ids if str(node_id).strip()
    ]
    participant_node_id_set = set(participant_nodes)
    participant_by_str = {
        str(node_id): node_id for node_id in participant_node_id_set if str(node_id)
    }
    if len(participant_node_id_set) <= 1:
        return participant_nodes

    direct_adjacency = socket_adjacency(topology, participant_node_id_set)
    relay_routes = relay_route_candidates(
        topology,
        overlay_peers,
        node_identities,
        participant_nodes,
    )
    relay_reachable_targets: dict[NodeId, set[NodeId]] = {
        node_id: set() for node_id in participant_node_id_set
    }
    for route in relay_routes:
        source_node_id = participant_by_str.get(
            str(route.get("sourceNodeId") or "").strip()
        )
        sink_node_id = participant_by_str.get(
            str(route.get("sinkNodeId") or "").strip()
        )
        if source_node_id is not None and sink_node_id is not None:
            relay_reachable_targets[source_node_id].add(sink_node_id)

    return [
        node_id
        for node_id in sorted(participant_node_id_set, key=str)
        if relay_reachable_targets.get(node_id)
        if all(
            other_node_id == node_id
            or other_node_id in direct_adjacency.get(node_id, set())
            or other_node_id in relay_reachable_targets.get(node_id, set())
            for other_node_id in participant_node_id_set
        )
    ]


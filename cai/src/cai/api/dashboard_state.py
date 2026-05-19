# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from cai.shared.network_routes import (
    relay_coordinator_candidate_node_ids,
    relay_route_candidates,
)
from cai.shared.types.common import NodeId
from cai.shared.types.topology import SocketConnection
from cai.shared.types.state import State


def _dashboard_stale_node_hide_seconds() -> int:
    raw = str(os.getenv("CAI_DASHBOARD_STALE_NODE_HIDE_SECONDS", "90") or "90").strip()
    try:
        return max(int(raw), 0)
    except ValueError:
        return 90


def _empty_topology_snapshot() -> dict[str, Any]:
    return {"nodes": [], "connections": {}}


def _normalize_selected_node_ids(
    node_ids: set[NodeId] | set[str] | None,
) -> set[str] | None:
    if node_ids is None:
        return None
    return {str(node_id).strip() for node_id in node_ids if str(node_id).strip()}


def _discover_visible_node_ids(payload: dict[str, Any]) -> set[str]:
    visible_node_ids: set[str] = set()

    topology = payload.get("topology")
    if isinstance(topology, Mapping):
        nodes = topology.get("nodes")
        if isinstance(nodes, list):
            visible_node_ids.update(
                str(node_id).strip() for node_id in nodes if str(node_id).strip()
            )
        connections = topology.get("connections")
        if isinstance(connections, Mapping):
            for source_node_id, connection_payload in connections.items():
                normalized_source_node_id = str(source_node_id).strip()
                if normalized_source_node_id:
                    visible_node_ids.add(normalized_source_node_id)
                if not isinstance(connection_payload, Mapping):
                    continue
                visible_node_ids.update(
                    str(target_node_id).strip()
                    for target_node_id in connection_payload.keys()
                    if str(target_node_id).strip()
                )

    for field_name in (
        "lastSeen",
        "overlayPeers",
        "overlayAdvertisedPeers",
        "nodeIdentities",
        "nodeMemory",
        "nodeDisk",
        "nodeSystem",
        "nodeNetwork",
        "nodeThunderbolt",
        "nodeThunderboltBridge",
        "nodeRdmaCtl",
        "downloads",
    ):
        field_payload = payload.get(field_name)
        if isinstance(field_payload, Mapping):
            visible_node_ids.update(
                str(node_id).strip()
                for node_id in field_payload.keys()
                if str(node_id).strip()
            )

    return visible_node_ids


def _discover_topology_node_ids(payload: dict[str, Any]) -> set[str]:
    topology_node_ids: set[str] = set()

    topology = payload.get("topology")
    if isinstance(topology, Mapping):
        nodes = topology.get("nodes")
        if isinstance(nodes, list):
            topology_node_ids.update(
                str(node_id).strip() for node_id in nodes if str(node_id).strip()
            )
        connections = topology.get("connections")
        if isinstance(connections, Mapping):
            for source_node_id, connection_payload in connections.items():
                normalized_source_node_id = str(source_node_id).strip()
                if normalized_source_node_id:
                    topology_node_ids.add(normalized_source_node_id)
                if not isinstance(connection_payload, Mapping):
                    continue
                topology_node_ids.update(
                    str(target_node_id).strip()
                    for target_node_id in connection_payload.keys()
                    if str(target_node_id).strip()
                )

    overlay_peers = payload.get("overlayPeers")
    if isinstance(overlay_peers, Mapping):
        for source_node_id, peer_payload in overlay_peers.items():
            normalized_source_node_id = str(source_node_id).strip()
            if normalized_source_node_id:
                topology_node_ids.add(normalized_source_node_id)
            if isinstance(peer_payload, Mapping):
                topology_node_ids.update(
                    str(target_node_id).strip()
                    for target_node_id in peer_payload.keys()
                    if str(target_node_id).strip()
                )
            elif isinstance(peer_payload, Sequence) and not isinstance(
                peer_payload, (str, bytes)
            ):
                topology_node_ids.update(
                    str(target_node_id).strip()
                    for target_node_id in peer_payload
                    if str(target_node_id).strip()
                )

    return topology_node_ids


def _parse_dashboard_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _prune_stale_visible_node_ids(
    payload: dict[str, Any],
    *,
    visible_node_ids: set[str],
    local_node_id: str,
) -> set[str]:
    if not visible_node_ids:
        return set()

    last_seen = payload.get("lastSeen")
    if not isinstance(last_seen, Mapping):
        return set(visible_node_ids)

    now = datetime.now(tz=timezone.utc)
    stale_after = timedelta(seconds=_dashboard_stale_node_hide_seconds())
    fresh_node_ids: set[str] = set()
    for node_id in visible_node_ids:
        if node_id == local_node_id:
            fresh_node_ids.add(node_id)
            continue
        parsed = _parse_dashboard_timestamp(last_seen.get(node_id))
        if parsed is None or now - parsed <= stale_after:
            fresh_node_ids.add(node_id)
    return fresh_node_ids


def _filter_mapping_by_keys(
    payload: Any, visible_keys: set[str]
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    return {
        str(key): value for key, value in payload.items() if str(key) in visible_keys
    }


def _count_worker_overlay_links(
    overlay_peers: Mapping[NodeId, Sequence[NodeId]],
    worker_node_ids: set[str],
) -> int:
    unique_links: set[tuple[str, str]] = set()
    for source_node_id, peers in overlay_peers.items():
        normalized_source_node_id = str(source_node_id).strip()
        if normalized_source_node_id not in worker_node_ids:
            continue
        for peer_node_id in peers:
            normalized_peer_node_id = str(peer_node_id).strip()
            if (
                normalized_peer_node_id not in worker_node_ids
                or normalized_peer_node_id == normalized_source_node_id
            ):
                continue
            unique_links.add(
                tuple(
                    sorted(
                        (
                            normalized_source_node_id,
                            normalized_peer_node_id,
                        )
                    )
                )
            )
    return len(unique_links)


def _worker_socket_topology_metrics(
    state: State,
    worker_node_ids: set[str],
) -> tuple[int, int, int]:
    unique_socket_links: set[tuple[str, str]] = set()
    directed_socket_links = 0
    worker_socket_topology = state.topology.__class__()

    for node_id in state.topology.list_nodes():
        if str(node_id).strip() in worker_node_ids:
            worker_socket_topology.add_node(node_id)

    for conn in state.topology.list_connections():
        normalized_source_node_id = str(conn.source).strip()
        normalized_sink_node_id = str(conn.sink).strip()
        if (
            normalized_source_node_id not in worker_node_ids
            or normalized_sink_node_id not in worker_node_ids
            or normalized_source_node_id == normalized_sink_node_id
            or not isinstance(conn.edge, SocketConnection)
        ):
            continue
        directed_socket_links += 1
        unique_socket_links.add(
            tuple(
                sorted(
                    (
                        normalized_source_node_id,
                        normalized_sink_node_id,
                    )
                )
            )
        )
        worker_socket_topology.add_connection(conn)

    largest_direct_cycle = max(
        (
            len(cycle)
            for cycle in worker_socket_topology.get_cycles()
            if len(cycle) >= 2
        ),
        default=0,
    )
    return directed_socket_links, len(unique_socket_links), largest_direct_cycle


def _llama_cpp_direct_coordinator_candidate_count(
    state: State,
    worker_node_ids: set[str],
) -> int:
    if len(worker_node_ids) <= 1:
        return len(worker_node_ids)

    adjacency: dict[str, set[str]] = {node_id: set() for node_id in worker_node_ids}
    for conn in state.topology.list_connections():
        normalized_source_node_id = str(conn.source).strip()
        normalized_sink_node_id = str(conn.sink).strip()
        if (
            normalized_source_node_id not in worker_node_ids
            or normalized_sink_node_id not in worker_node_ids
            or normalized_source_node_id == normalized_sink_node_id
            or not isinstance(conn.edge, SocketConnection)
        ):
            continue
        adjacency[normalized_source_node_id].add(normalized_sink_node_id)

    return sum(
        1
        for node_id in worker_node_ids
        if len(adjacency.get(node_id, set())) >= len(worker_node_ids) - 1
    )


def _relay_transit_candidate_count(
    state: State,
    worker_node_ids: set[str],
) -> int:
    return sum(
        1
        for node_id, identity in state.node_identities.items()
        if str(node_id).strip() not in worker_node_ids
        and bool(getattr(identity, "relay_enabled", False))
    )


def _worker_enabled_node_ids_from_state(state: State) -> set[str]:
    return {
        str(node_id).strip()
        for node_id, identity in state.node_identities.items()
        if str(node_id).strip()
        and bool(getattr(identity, "worker_enabled", False))
    }


def _llama_cpp_distributed_status(
    *,
    worker_count: int,
    coordinator_candidate_count: int,
    relay_coordinator_candidate_count: int,
    relay_route_candidate_count: int,
    directed_socket_links: int,
    worker_overlay_links: int,
    largest_direct_cycle: int,
    relay_transit_candidate_count: int,
) -> tuple[bool, str | None]:
    if worker_count < 2:
        return False, "Need at least 2 worker-enabled nodes for distributed llama.cpp."
    if coordinator_candidate_count > 0:
        return True, None
    if relay_coordinator_candidate_count > 0:
        return (
            True,
            None,
        )
    if relay_route_candidate_count > 0:
        return (
            False,
            "Workers expose partial coordinator-to-relay-to-worker routes, but no "
            "single coordinator can yet reach every worker through direct or relay "
            "transport.",
        )
    if worker_overlay_links > 0 and directed_socket_links == 0:
        if relay_transit_candidate_count > 0:
            return (
                False,
                "Workers are visible through overlay and relay-capable transit nodes "
                "are present, but no executable coordinator-to-worker relay fanout "
                "was found yet.",
            )
        return (
            False,
            "Workers are visible through overlay, but distributed llama.cpp still "
            "needs an executable direct or relay coordinator-to-worker path.",
        )
    if directed_socket_links > 0:
        return (
            False,
            "Workers have partial direct links, but no single coordinator can "
            "reach every worker for distributed llama.cpp.",
        )
    return (
        False,
        "Workers do not expose a direct coordinator-to-worker path yet for distributed llama.cpp.",
    )


def _cai_owned_transport_network_readiness(
    state: State,
    worker_node_ids: set[str],
) -> dict[str, Any]:
    worker_count = len(worker_node_ids)
    runtime_ready_worker_ids: list[str] = []
    implemented_worker_ids: list[str] = []
    contract_ready_worker_ids: list[str] = []
    production_ready_worker_ids: list[str] = []
    backend_health_failed_worker_ids: list[str] = []
    failed_worker_ids: list[str] = []
    disabled_worker_ids: list[str] = []
    missing_worker_ids: list[str] = []
    observed_statuses: list[str] = []
    protocol: str | None = None

    identities_by_node_id = {
        str(node_id).strip(): identity
        for node_id, identity in state.node_identities.items()
        if str(node_id).strip()
    }
    for node_id in sorted(worker_node_ids, key=str):
        identity = identities_by_node_id.get(node_id)
        readiness = getattr(identity, "readiness", {}) if identity is not None else {}
        transport_readiness = (
            readiness.get("caiOwnedTransport")
            if isinstance(readiness, Mapping)
            else None
        )
        if not isinstance(transport_readiness, Mapping):
            missing_worker_ids.append(node_id)
            observed_statuses.append("planned")
            continue

        if protocol is None and transport_readiness.get("protocol"):
            protocol = str(transport_readiness.get("protocol"))
        status = str(transport_readiness.get("status") or "").strip().lower()
        if not status:
            status = (
                "ready" if bool(transport_readiness.get("runtimeReady")) else "planned"
            )
        if bool(transport_readiness.get("implemented")):
            implemented_worker_ids.append(node_id)
        self_test = transport_readiness.get("llmShardSelfTest")
        backend_health_failed = (
            isinstance(self_test, Mapping)
            and self_test.get("backendHealthReady") is False
        )
        production_self_test_ready = (
            isinstance(self_test, Mapping)
            and bool(self_test.get("productionReady"))
            and self_test.get("generationProbeReady") is True
            and not backend_health_failed
        )
        if backend_health_failed and status in {
            "ready",
            "test_adapter_ready",
            "planned",
        }:
            status = "failed"
        runtime_ready_claim = bool(transport_readiness.get("runtimeReady"))
        runtime_ready_proof = transport_readiness.get("runtimeReadyProof")
        runtime_ready_proof_verified = (
            isinstance(runtime_ready_proof, Mapping)
            and runtime_ready_proof.get("verified") is True
        )
        if (
            runtime_ready_claim
            and not (runtime_ready_proof_verified and production_self_test_ready)
            and status == "ready"
        ):
            status = "test_adapter_ready"
        observed_statuses.append(status)

        if (
            runtime_ready_claim
            and runtime_ready_proof_verified
            and production_self_test_ready
            and not backend_health_failed
        ):
            runtime_ready_worker_ids.append(node_id)
        if isinstance(self_test, Mapping):
            if bool(self_test.get("contractReady")) and not backend_health_failed:
                contract_ready_worker_ids.append(node_id)
            if production_self_test_ready:
                production_ready_worker_ids.append(node_id)
        if backend_health_failed:
            backend_health_failed_worker_ids.append(node_id)
        if status in {"failed", "error", "unreachable", "route_failed"}:
            failed_worker_ids.append(node_id)
        if status in {"disabled", "off"}:
            disabled_worker_ids.append(node_id)

    if worker_count <= 0:
        status = "planned"
        reason = "No worker-enabled nodes advertise CAI-owned transport readiness yet."
    elif failed_worker_ids:
        status = "failed"
        reason = (
            "At least one worker reports degraded CAI-owned LLM shard backend health."
            if backend_health_failed_worker_ids
            else "At least one worker reports failed CAI-owned transport readiness."
        )
    elif (
        len(runtime_ready_worker_ids) >= 2
        and len(runtime_ready_worker_ids) == worker_count
    ):
        status = "ready"
        reason = None
    elif runtime_ready_worker_ids and len(runtime_ready_worker_ids) < 2:
        status = "test_adapter_ready"
        reason = (
            "At least 2 runtime-ready worker nodes are required before "
            "CAI-owned transport is network-ready."
        )
    elif implemented_worker_ids:
        status = "test_adapter_ready"
        if contract_ready_worker_ids:
            reason = (
                "CAI-owned LLM shard adapter contract self-test is present, but "
                "productionReady proof is not enabled for all workers."
            )
        else:
            reason = (
                "CAI-owned deterministic shard adapter is present, but production "
                "runtimeReady proof is not enabled for all workers."
            )
    elif len(disabled_worker_ids) == worker_count:
        status = "disabled"
        reason = "CAI-owned transport is disabled on worker-enabled nodes."
    else:
        status = "planned"
        reason = "Worker nodes have not advertised CAI-owned transport readiness yet."

    return {
        "protocol": protocol,
        "status": status,
        "ready": status == "ready",
        "runtimeReady": status == "ready",
        "reason": reason,
        "workerCount": worker_count,
        "runtimeReadyWorkerCount": len(runtime_ready_worker_ids),
        "implementedWorkerCount": len(implemented_worker_ids),
        "contractReadyWorkerCount": len(contract_ready_worker_ids),
        "productionReadyWorkerCount": len(production_ready_worker_ids),
        "backendHealthFailedWorkerCount": len(backend_health_failed_worker_ids),
        "failedWorkerCount": len(failed_worker_ids),
        "missingWorkerCount": len(missing_worker_ids),
        "observedStatuses": sorted(set(observed_statuses)),
        "runtimeReadyWorkerIds": runtime_ready_worker_ids,
        "implementedWorkerIds": implemented_worker_ids,
        "contractReadyWorkerIds": contract_ready_worker_ids,
        "productionReadyWorkerIds": production_ready_worker_ids,
        "backendHealthFailedWorkerIds": backend_health_failed_worker_ids,
        "failedWorkerIds": failed_worker_ids,
        "missingWorkerIds": missing_worker_ids,
    }


def _sanitize_topology_snapshot(
    payload: dict[str, Any],
    *,
    visible_node_ids: set[str],
    topology_node_ids: set[str],
    local_node_id: str,
) -> dict[str, Any]:
    topology = payload.get("topology")
    filtered_nodes: list[str] = []
    seen_nodes: set[str] = set()
    filtered_connections: dict[str, Any] = {}

    if isinstance(topology, Mapping):
        nodes = topology.get("nodes")
        if isinstance(nodes, list):
            for node_id in nodes:
                normalized_node_id = str(node_id).strip()
                if (
                    normalized_node_id
                    and normalized_node_id in visible_node_ids
                    and normalized_node_id not in seen_nodes
                ):
                    filtered_nodes.append(normalized_node_id)
                    seen_nodes.add(normalized_node_id)

        connections_payload = topology.get("connections")
        if isinstance(connections_payload, Mapping):
            for source_node_id, connection_payload in connections_payload.items():
                normalized_source_node_id = str(source_node_id).strip()
                if normalized_source_node_id not in visible_node_ids:
                    continue
                if not isinstance(connection_payload, Mapping):
                    continue
                visible_connections = {
                    str(target_node_id): connection_data
                    for target_node_id, connection_data in connection_payload.items()
                    if str(target_node_id).strip() in visible_node_ids
                }
                if visible_connections:
                    filtered_connections[normalized_source_node_id] = visible_connections

    extra_nodes = [node_id for node_id in topology_node_ids if node_id not in seen_nodes]
    extra_nodes.sort(key=lambda node_id: (0 if node_id == local_node_id else 1, node_id))
    filtered_nodes.extend(extra_nodes)

    if not filtered_nodes:
        return _empty_topology_snapshot()

    return {"nodes": filtered_nodes, "connections": filtered_connections}


def _filter_instance_wrapper(
    wrapper: Any, visible_node_ids: set[str]
) -> tuple[dict[str, Any] | None, set[str]]:
    if not isinstance(wrapper, Mapping) or not wrapper:
        return None, set()

    instance_tag, instance_payload = next(iter(wrapper.items()))
    if not isinstance(instance_payload, Mapping):
        return {str(instance_tag): dict(wrapper[str(instance_tag)])}, set()

    payload = dict(instance_payload)
    shard_assignments = payload.get("shardAssignments")
    if not isinstance(shard_assignments, Mapping):
        return {str(instance_tag): payload}, set()

    shard_payload = dict(shard_assignments)
    node_to_runner = shard_payload.get("nodeToRunner")
    if not isinstance(node_to_runner, Mapping):
        payload["shardAssignments"] = shard_payload
        return {str(instance_tag): payload}, set()

    filtered_node_to_runner = {
        str(node_id): runner_id
        for node_id, runner_id in node_to_runner.items()
        if str(node_id) in visible_node_ids
    }
    if not filtered_node_to_runner:
        return None, set()

    visible_runner_ids = {str(runner_id) for runner_id in filtered_node_to_runner.values()}

    runner_to_shard = shard_payload.get("runnerToShard")
    if isinstance(runner_to_shard, Mapping):
        shard_payload["runnerToShard"] = {
            str(runner_id): shard
            for runner_id, shard in runner_to_shard.items()
            if str(runner_id) in visible_runner_ids
        }
    shard_payload["nodeToRunner"] = filtered_node_to_runner
    payload["shardAssignments"] = shard_payload

    hosts_by_node = payload.get("hostsByNode")
    if isinstance(hosts_by_node, Mapping):
        payload["hostsByNode"] = {
            str(node_id): hosts
            for node_id, hosts in hosts_by_node.items()
            if str(node_id) in visible_node_ids
        }
    relay_routes_by_node = payload.get("relayRoutesByNode")
    if isinstance(relay_routes_by_node, Mapping):
        payload["relayRoutesByNode"] = {
            str(node_id): routes
            for node_id, routes in relay_routes_by_node.items()
            if str(node_id) in visible_node_ids
        }

    jaccl_coordinators = payload.get("jacclCoordinators")
    if isinstance(jaccl_coordinators, Mapping):
        payload["jacclCoordinators"] = {
            str(node_id): coordinator
            for node_id, coordinator in jaccl_coordinators.items()
            if str(node_id) in visible_node_ids
        }

    return {str(instance_tag): payload}, visible_runner_ids


def sanitize_dashboard_state_payload(
    payload: dict[str, Any],
    *,
    local_node_id: str,
    network_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    visible_node_ids = _discover_visible_node_ids(payload)
    visible_node_ids = _prune_stale_visible_node_ids(
        payload,
        visible_node_ids=visible_node_ids,
        local_node_id=local_node_id,
    )
    topology_node_ids = _discover_topology_node_ids(payload)
    topology_node_ids = _prune_stale_visible_node_ids(
        payload,
        visible_node_ids=topology_node_ids,
        local_node_id=local_node_id,
    )
    sanitized = dict(payload)
    sanitized["topology"] = _sanitize_topology_snapshot(
        payload,
        visible_node_ids=visible_node_ids,
        topology_node_ids=topology_node_ids,
        local_node_id=local_node_id,
    )

    for field_name in (
        "lastSeen",
        "overlayPeers",
        "overlayAdvertisedPeers",
        "nodeIdentities",
        "nodeMemory",
        "nodeDisk",
        "nodeSystem",
        "nodeNetwork",
        "nodeThunderbolt",
        "nodeThunderboltBridge",
        "nodeRdmaCtl",
        "downloads",
    ):
        sanitized[field_name] = _filter_mapping_by_keys(
            payload.get(field_name), visible_node_ids
        )

    thunderbolt_cycles = payload.get("thunderboltBridgeCycles")
    if isinstance(thunderbolt_cycles, list):
        sanitized["thunderboltBridgeCycles"] = [
            cycle
            for cycle in thunderbolt_cycles
            if isinstance(cycle, list)
            and cycle
            and all(str(node_id) in visible_node_ids for node_id in cycle)
        ]
    else:
        sanitized["thunderboltBridgeCycles"] = []

    visible_runner_ids: set[str] = set()
    instances = payload.get("instances")
    if isinstance(instances, Mapping):
        filtered_instances: dict[str, Any] = {}
        for instance_id, wrapper in instances.items():
            filtered_wrapper, runner_ids = _filter_instance_wrapper(
                wrapper, visible_node_ids
            )
            if filtered_wrapper is None:
                continue
            filtered_instances[str(instance_id)] = filtered_wrapper
            visible_runner_ids.update(runner_ids)
        sanitized["instances"] = filtered_instances
    else:
        sanitized["instances"] = {}

    runners = payload.get("runners")
    if isinstance(runners, Mapping):
        sanitized["runners"] = {
            str(runner_id): runner_state
            for runner_id, runner_state in runners.items()
            if str(runner_id) in visible_runner_ids
        }
    else:
        sanitized["runners"] = {}

    if network_summary is not None:
        sanitized["networkSummary"] = network_summary
    sanitized["currentNodeId"] = local_node_id

    return sanitized


def build_dashboard_state(
    state: State,
    local_node_id: NodeId,
    *,
    worker_node_ids: set[NodeId] | set[str] | None = None,
) -> dict[str, Any]:
    payload = state.model_dump(by_alias=True)
    visible_node_ids = _discover_visible_node_ids(payload)
    visible_node_ids = _prune_stale_visible_node_ids(
        payload,
        visible_node_ids=visible_node_ids,
        local_node_id=str(local_node_id),
    )
    total_nodes = (
        len(visible_node_ids)
        if visible_node_ids
        else len(list(state.topology.list_nodes()))
    )
    total_connections = len(list(state.topology.list_connections()))
    local_overlay_peers = len(
        [
            peer_id
            for peer_id in state.overlay_peers.get(local_node_id, ())
            if str(peer_id).strip() in visible_node_ids
        ]
    )
    total_ram_bytes = 0
    total_available_ram_bytes = 0
    total_vram_bytes = 0
    total_cpu_cores = 0
    normalized_worker_node_ids = _normalize_selected_node_ids(worker_node_ids)
    state_worker_node_ids = _worker_enabled_node_ids_from_state(state)
    if normalized_worker_node_ids is None:
        normalized_worker_node_ids = state_worker_node_ids
    elif not normalized_worker_node_ids and state_worker_node_ids:
        # An empty eligibility set can happen when capability verification or
        # summary probing fails. The dashboard should still surface visible
        # worker-enabled nodes and their readiness errors instead of hiding them.
        normalized_worker_node_ids = state_worker_node_ids
    normalized_worker_node_ids = {
        node_id
        for node_id in normalized_worker_node_ids
        if str(node_id).strip() in visible_node_ids
    }
    worker_total_ram_bytes = 0
    worker_total_available_ram_bytes = 0
    worker_total_vram_bytes = 0
    worker_total_cpu_cores = 0
    relay_count = 0
    worker_count = len(normalized_worker_node_ids)
    selected_worker_node_ids = normalized_worker_node_ids
    for node_id, memory in state.node_memory.items():
        if str(node_id).strip() not in visible_node_ids:
            continue
        ram_total_bytes = int(getattr(getattr(memory, "ram_total", None), "in_bytes", 0))
        ram_available_bytes = int(
            getattr(getattr(memory, "ram_available", None), "in_bytes", 0)
        )
        total_ram_bytes += ram_total_bytes
        total_available_ram_bytes += int(
            ram_available_bytes
        )
        if str(node_id).strip() in normalized_worker_node_ids:
            worker_total_ram_bytes += ram_total_bytes
            worker_total_available_ram_bytes += ram_available_bytes
    for node_id, identity in state.node_identities.items():
        if str(node_id).strip() not in visible_node_ids:
            continue
        identity_vram_bytes = int(getattr(identity, "total_vram_bytes", 0) or 0)
        identity_cpu_cores = int(
            getattr(identity, "cpu_physical_cores", None)
            or getattr(identity, "cpu_logical_cores", 0)
            or 0
        )
        if bool(getattr(identity, "relay_enabled", False)):
            relay_count += 1
        total_vram_bytes += identity_vram_bytes
        total_cpu_cores += identity_cpu_cores
        if str(node_id).strip() in normalized_worker_node_ids:
            worker_total_vram_bytes += identity_vram_bytes
            worker_total_cpu_cores += identity_cpu_cores
    directed_worker_socket_links, worker_direct_socket_links, llama_cpp_largest_direct_cycle = (
        _worker_socket_topology_metrics(state, selected_worker_node_ids)
    )
    worker_overlay_links = _count_worker_overlay_links(
        state.overlay_peers,
        selected_worker_node_ids,
    )
    coordinator_candidate_count = _llama_cpp_direct_coordinator_candidate_count(
        state,
        selected_worker_node_ids,
    )
    relay_route_candidate_items = relay_route_candidates(
        state.topology,
        state.overlay_peers,
        state.node_identities,
        sorted(selected_worker_node_ids, key=str),
    )
    relay_coordinator_candidates = relay_coordinator_candidate_node_ids(
        state.topology,
        state.overlay_peers,
        state.node_identities,
        sorted(selected_worker_node_ids, key=str),
    )
    relay_transit_candidate_count = _relay_transit_candidate_count(
        state,
        selected_worker_node_ids,
    )
    llama_cpp_distributed_ready, llama_cpp_distributed_reason = (
        _llama_cpp_distributed_status(
            worker_count=worker_count,
            coordinator_candidate_count=coordinator_candidate_count,
            relay_coordinator_candidate_count=len(relay_coordinator_candidates),
            relay_route_candidate_count=len(relay_route_candidate_items),
            directed_socket_links=directed_worker_socket_links,
            worker_overlay_links=worker_overlay_links,
            largest_direct_cycle=llama_cpp_largest_direct_cycle,
            relay_transit_candidate_count=relay_transit_candidate_count,
        )
    )
    cai_owned_transport_readiness = _cai_owned_transport_network_readiness(
        state,
        selected_worker_node_ids,
    )
    network_summary = {
        "knownNodes": total_nodes,
        "knownWorkers": worker_count,
        "knownRelays": relay_count,
        "knownConnections": total_connections,
        "localOverlayPeers": local_overlay_peers,
        "totalRamBytes": total_ram_bytes,
        "totalAvailableRamBytes": total_available_ram_bytes,
        "totalVramBytes": total_vram_bytes,
        "totalCpuCores": total_cpu_cores,
        "workerTotalRamBytes": worker_total_ram_bytes,
        "workerTotalAvailableRamBytes": worker_total_available_ram_bytes,
        "workerTotalVramBytes": worker_total_vram_bytes,
        "workerTotalCpuCores": worker_total_cpu_cores,
        "workerDirectSocketLinks": worker_direct_socket_links,
        "workerOverlayLinks": worker_overlay_links,
        "llamaCppLargestDirectWorkerCycle": llama_cpp_largest_direct_cycle,
        "llamaCppRelayCoordinatorCandidateCount": len(relay_coordinator_candidates),
        "llamaCppRelayRouteCandidateCount": len(relay_route_candidate_items),
        "llamaCppDistributedReady": llama_cpp_distributed_ready,
        "llamaCppDistributedReason": llama_cpp_distributed_reason,
        "caiOwnedTransportReadiness": cai_owned_transport_readiness,
    }
    return sanitize_dashboard_state_payload(
        payload,
        local_node_id=str(local_node_id),
        network_summary=network_summary,
    )


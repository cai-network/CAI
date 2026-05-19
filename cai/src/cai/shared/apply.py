# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import copy
from collections.abc import Mapping, Sequence
from datetime import datetime

from loguru import logger

from cai.shared.types.common import NodeId
from cai.shared.types.events import (
    ChunkGenerated,
    CustomModelCardAdded,
    CustomModelCardDeleted,
    Event,
    IndexedEvent,
    InputChunkReceived,
    InstanceCreated,
    InstanceDeleted,
    NodeDownloadProgress,
    NodeGatheredInfo,
    OverlayBootstrapPeersAdvertised,
    OverlayPeerConnected,
    OverlayPeerDisconnected,
    NodeTimedOut,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskCreated,
    TaskDeleted,
    TaskFailed,
    TaskStatusUpdated,
    TestEvent,
    TopologyEdgeCreated,
    TopologyEdgeDeleted,
    TracesCollected,
    TracesMerged,
)
from cai.shared.types.profiling import (
    AdvertisedTransportEndpoint,
    NodeIdentity,
    NodeNetworkInfo,
    NodeRdmaCtlStatus,
    NodeThunderboltInfo,
    ThunderboltBridgeStatus,
)
from cai.shared.types.state import State
from cai.shared.types.tasks import Task, TaskId, TaskStatus
from cai.shared.types.topology import Connection, RDMAConnection
from cai.shared.types.worker.downloads import DownloadProgress
from cai.shared.types.worker.instances import Instance, InstanceId
from cai.shared.types.worker.runners import RunnerId, RunnerShutdown, RunnerStatus
from cai.utils.info_gatherer.info_gatherer import (
    ApiEndpointInfo,
    MacmonMetrics,
    MacThunderboltConnections,
    MacThunderboltIdentifiers,
    MemoryUsage,
    MiscData,
    NodeConfig,
    NodeDiskUsage,
    NodeNetworkInterfaces,
    PsutilSystemMetrics,
    RdmaCtlStatus,
    StaticNodeInformation,
    ThunderboltBridgeInfo,
    WorkerStateInfo,
)


def _dedupe_transport_endpoints(
    endpoints: Sequence[AdvertisedTransportEndpoint],
) -> list[AdvertisedTransportEndpoint]:
    deduped: list[AdvertisedTransportEndpoint] = []
    seen: set[tuple[object, ...]] = set()
    for endpoint in endpoints:
        key = (
            endpoint.purpose,
            endpoint.route_type,
            endpoint.host,
            endpoint.port,
            endpoint.source,
            endpoint.interface_name,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(endpoint)
    return deduped


def _normalized_transport_endpoints(
    current_identity: NodeIdentity,
    *,
    endpoints: Sequence[AdvertisedTransportEndpoint] | None = None,
    relay_enabled: bool | None = None,
) -> list[AdvertisedTransportEndpoint]:
    base_endpoints = [
        endpoint
        for endpoint in (endpoints or current_identity.transport_endpoints)
        if endpoint.route_type != "relay"
    ]
    effective_relay_enabled = (
        bool(current_identity.relay_enabled)
        if relay_enabled is None
        else bool(relay_enabled)
    )
    if not effective_relay_enabled:
        return _dedupe_transport_endpoints(base_endpoints)

    relay_endpoints = [
        endpoint.model_copy(update={"route_type": "relay"})
        for endpoint in base_endpoints
    ]
    return _dedupe_transport_endpoints([*base_endpoints, *relay_endpoints])


def event_apply(event: Event, state: State) -> State:
    """Apply an event to state."""
    match event:
        case (
            TestEvent()
            | ChunkGenerated()
            | TaskAcknowledged()
            | InputChunkReceived()
            | TracesCollected()
            | TracesMerged()
            | CustomModelCardAdded()
            | CustomModelCardDeleted()
        ):  # Pass-through events that don't modify state
            return state
        case InstanceCreated():
            return apply_instance_created(event, state)
        case InstanceDeleted():
            return apply_instance_deleted(event, state)
        case NodeTimedOut():
            return apply_node_timed_out(event, state)
        case NodeDownloadProgress():
            return apply_node_download_progress(event, state)
        case NodeGatheredInfo():
            return apply_node_gathered_info(event, state)
        case RunnerStatusUpdated():
            return apply_runner_status_updated(event, state)
        case TaskCreated():
            return apply_task_created(event, state)
        case TaskDeleted():
            return apply_task_deleted(event, state)
        case TaskFailed():
            return apply_task_failed(event, state)
        case TaskStatusUpdated():
            return apply_task_status_updated(event, state)
        case TopologyEdgeCreated():
            return apply_topology_edge_created(event, state)
        case TopologyEdgeDeleted():
            return apply_topology_edge_deleted(event, state)
        case OverlayPeerConnected():
            return apply_overlay_peer_connected(event, state)
        case OverlayPeerDisconnected():
            return apply_overlay_peer_disconnected(event, state)
        case OverlayBootstrapPeersAdvertised():
            return apply_overlay_bootstrap_peers_advertised(event, state)


def apply(state: State, event: IndexedEvent) -> State:
    # Just to test that events are only applied in correct order
    if state.last_event_applied_idx != event.idx - 1:
        logger.warning(
            f"Expected event {state.last_event_applied_idx + 1} but received {event.idx}"
        )
    assert state.last_event_applied_idx == event.idx - 1
    new_state: State = event_apply(event.event, state)
    return new_state.model_copy(update={"last_event_applied_idx": event.idx})


def apply_node_download_progress(event: NodeDownloadProgress, state: State) -> State:
    """
    Update or add a node download progress to state.
    """
    dp = event.download_progress
    node_id = dp.node_id

    current = list(state.downloads.get(node_id, ()))

    replaced = False
    for i, existing_dp in enumerate(current):
        # TODO(ciaran): deduplicate by model_id for now. Will need to use
        # shard_metadata again when pipeline and tensor downloads differ.
        # For now this is fine
        if (
            existing_dp.shard_metadata.model_card.model_id
            == dp.shard_metadata.model_card.model_id
        ):
            current[i] = dp
            replaced = True
            break

    if not replaced:
        current.append(dp)

    new_downloads: Mapping[NodeId, Sequence[DownloadProgress]] = {
        **state.downloads,
        node_id: current,
    }
    return state.model_copy(update={"downloads": new_downloads})


def apply_task_created(event: TaskCreated, state: State) -> State:
    new_tasks: Mapping[TaskId, Task] = {**state.tasks, event.task_id: event.task}
    return state.model_copy(update={"tasks": new_tasks})


def apply_task_deleted(event: TaskDeleted, state: State) -> State:
    new_tasks: Mapping[TaskId, Task] = {
        tid: task for tid, task in state.tasks.items() if tid != event.task_id
    }
    return state.model_copy(update={"tasks": new_tasks})


def apply_task_status_updated(event: TaskStatusUpdated, state: State) -> State:
    if event.task_id not in state.tasks:
        # maybe should raise
        return state

    update: dict[str, TaskStatus | None] = {
        "task_status": event.task_status,
    }
    if event.task_status != TaskStatus.Failed:
        update["error_type"] = None
        update["error_message"] = None

    updated_task = state.tasks[event.task_id].model_copy(update=update)
    new_tasks: Mapping[TaskId, Task] = {**state.tasks, event.task_id: updated_task}
    return state.model_copy(update={"tasks": new_tasks})


def apply_task_failed(event: TaskFailed, state: State) -> State:
    if event.task_id not in state.tasks:
        # maybe should raise
        return state

    updated_task = state.tasks[event.task_id].model_copy(
        update={"error_type": event.error_type, "error_message": event.error_message}
    )
    new_tasks: Mapping[TaskId, Task] = {**state.tasks, event.task_id: updated_task}
    return state.model_copy(update={"tasks": new_tasks})


def apply_instance_created(event: InstanceCreated, state: State) -> State:
    instance = event.instance
    new_instances: Mapping[InstanceId, Instance] = {
        **state.instances,
        instance.instance_id: instance,
    }
    return state.model_copy(update={"instances": new_instances})


def apply_instance_deleted(event: InstanceDeleted, state: State) -> State:
    new_instances: Mapping[InstanceId, Instance] = {
        iid: inst for iid, inst in state.instances.items() if iid != event.instance_id
    }
    return state.model_copy(update={"instances": new_instances})


def apply_runner_status_updated(event: RunnerStatusUpdated, state: State) -> State:
    if isinstance(event.runner_status, RunnerShutdown):
        new_runners: Mapping[RunnerId, RunnerStatus] = {
            rid: rs for rid, rs in state.runners.items() if rid != event.runner_id
        }
        return state.model_copy(update={"runners": new_runners})
    new_runners = {
        **state.runners,
        event.runner_id: event.runner_status,
    }
    return state.model_copy(update={"runners": new_runners})


def apply_node_timed_out(event: NodeTimedOut, state: State) -> State:
    topology = copy.deepcopy(state.topology)
    topology.remove_node(event.node_id)
    last_seen = {
        key: value for key, value in state.last_seen.items() if key != event.node_id
    }
    downloads = {
        key: value for key, value in state.downloads.items() if key != event.node_id
    }
    # Clean up all granular node mappings
    node_memory = {
        key: value for key, value in state.node_memory.items() if key != event.node_id
    }
    node_identities = {
        key: value
        for key, value in state.node_identities.items()
        if key != event.node_id
    }
    node_disk = {
        key: value for key, value in state.node_disk.items() if key != event.node_id
    }
    node_system = {
        key: value for key, value in state.node_system.items() if key != event.node_id
    }
    node_network = {
        key: value for key, value in state.node_network.items() if key != event.node_id
    }
    node_thunderbolt = {
        key: value
        for key, value in state.node_thunderbolt.items()
        if key != event.node_id
    }
    node_thunderbolt_bridge = {
        key: value
        for key, value in state.node_thunderbolt_bridge.items()
        if key != event.node_id
    }
    node_rdma_ctl = {
        key: value for key, value in state.node_rdma_ctl.items() if key != event.node_id
    }
    overlay_peers = {
        key: [peer for peer in value if peer != event.node_id]
        for key, value in state.overlay_peers.items()
        if key != event.node_id and any(peer != event.node_id for peer in value)
    }
    overlay_advertised_peers = {
        key: value
        for key, value in state.overlay_advertised_peers.items()
        if key != event.node_id
    }
    # Only recompute cycles if the leaving node had TB bridge enabled
    leaving_node_status = state.node_thunderbolt_bridge.get(event.node_id)
    leaving_node_had_tb_enabled = (
        leaving_node_status is not None and leaving_node_status.enabled
    )
    thunderbolt_bridge_cycles = (
        topology.get_thunderbolt_bridge_cycles(node_thunderbolt_bridge, node_network)
        if leaving_node_had_tb_enabled
        else [list(cycle) for cycle in state.thunderbolt_bridge_cycles]
    )
    return state.model_copy(
        update={
            "downloads": downloads,
            "topology": topology,
            "last_seen": last_seen,
            "node_identities": node_identities,
            "node_memory": node_memory,
            "node_disk": node_disk,
            "node_system": node_system,
            "node_network": node_network,
            "node_thunderbolt": node_thunderbolt,
            "node_thunderbolt_bridge": node_thunderbolt_bridge,
            "node_rdma_ctl": node_rdma_ctl,
            "overlay_peers": overlay_peers,
            "overlay_advertised_peers": overlay_advertised_peers,
            "thunderbolt_bridge_cycles": thunderbolt_bridge_cycles,
        }
    )


def apply_node_gathered_info(event: NodeGatheredInfo, state: State) -> State:
    topology = copy.deepcopy(state.topology)
    topology.add_node(event.node_id)
    info = event.info

    # Build update dict with only the mappings that change
    update: dict[str, object] = {
        "last_seen": {
            **state.last_seen,
            event.node_id: datetime.fromisoformat(event.when),
        },
        "topology": topology,
    }

    match info:
        case MacmonMetrics():
            update["node_system"] = {
                **state.node_system,
                event.node_id: info.system_profile,
            }
            update["node_memory"] = {**state.node_memory, event.node_id: info.memory}
        case PsutilSystemMetrics():
            update["node_system"] = {
                **state.node_system,
                event.node_id: info.system_profile,
            }
        case MemoryUsage():
            update["node_memory"] = {**state.node_memory, event.node_id: info}
        case NodeDiskUsage():
            update["node_disk"] = {**state.node_disk, event.node_id: info.disk_usage}
        case NodeConfig():
            pass
        case MiscData():
            current_identity = state.node_identities.get(event.node_id, NodeIdentity())
            new_identity = current_identity.model_copy(
                update={"friendly_name": info.friendly_name}
            )
            update["node_identities"] = {
                **state.node_identities,
                event.node_id: new_identity,
            }
        case ApiEndpointInfo():
            current_identity = state.node_identities.get(event.node_id, NodeIdentity())
            new_identity = current_identity.model_copy(
                update={
                    "api_host": info.host,
                    "api_port": info.port,
                    "data_host": info.data_host,
                    "data_port": info.data_port,
                    "transport_endpoints": _normalized_transport_endpoints(
                        current_identity,
                        endpoints=list(info.transport_endpoints),
                    ),
                }
            )
            update["node_identities"] = {
                **state.node_identities,
                event.node_id: new_identity,
            }
        case StaticNodeInformation():
            current_identity = state.node_identities.get(event.node_id, NodeIdentity())
            new_identity = current_identity.model_copy(
                update={
                    "model_id": info.model,
                    "chip_id": info.chip,
                    "os_version": info.os_version,
                    "os_build_version": info.os_build_version,
                    "cpu_physical_cores": info.cpu_physical_cores,
                    "cpu_logical_cores": info.cpu_logical_cores,
                    "total_vram_bytes": info.total_vram_bytes,
                }
            )
            update["node_identities"] = {
                **state.node_identities,
                event.node_id: new_identity,
            }
        case WorkerStateInfo():
            current_identity = state.node_identities.get(event.node_id, NodeIdentity())
            new_identity = current_identity.model_copy(
                update={
                    "worker_enabled": info.worker_enabled,
                    "relay_enabled": info.relay_enabled,
                    "worker_reward_address": info.worker_reward_address,
                    "node_public_key_b64": info.node_public_key_b64,
                    "node_public_key_address": info.node_public_key_address,
                    "readiness": dict(info.readiness or {}),
                    "transport_endpoints": _normalized_transport_endpoints(
                        current_identity,
                        relay_enabled=info.relay_enabled,
                    ),
                }
            )
            update["node_identities"] = {
                **state.node_identities,
                event.node_id: new_identity,
            }
        case NodeNetworkInterfaces():
            update["node_network"] = {
                **state.node_network,
                event.node_id: NodeNetworkInfo(interfaces=info.ifaces),
            }
        case MacThunderboltIdentifiers():
            update["node_thunderbolt"] = {
                **state.node_thunderbolt,
                event.node_id: NodeThunderboltInfo(interfaces=info.idents),
            }
        case MacThunderboltConnections():
            conn_map = {
                tb_ident.domain_uuid: (nid, tb_ident.rdma_interface)
                for nid in state.node_thunderbolt
                for tb_ident in state.node_thunderbolt[nid].interfaces
            }
            as_rdma_conns = [
                Connection(
                    source=event.node_id,
                    sink=conn_map[tb_conn.sink_uuid][0],
                    edge=RDMAConnection(
                        source_rdma_iface=conn_map[tb_conn.source_uuid][1],
                        sink_rdma_iface=conn_map[tb_conn.sink_uuid][1],
                    ),
                )
                for tb_conn in info.conns
                if tb_conn.source_uuid in conn_map
                if tb_conn.sink_uuid in conn_map
            ]
            topology.replace_all_out_rdma_connections(event.node_id, as_rdma_conns)
        case ThunderboltBridgeInfo():
            new_tb_bridge: dict[NodeId, ThunderboltBridgeStatus] = {
                **state.node_thunderbolt_bridge,
                event.node_id: info.status,
            }
            update["node_thunderbolt_bridge"] = new_tb_bridge
            # Only recompute cycles if the enabled status changed
            old_status = state.node_thunderbolt_bridge.get(event.node_id)
            old_enabled = old_status.enabled if old_status else False
            new_enabled = info.status.enabled
            if old_enabled != new_enabled:
                update["thunderbolt_bridge_cycles"] = (
                    topology.get_thunderbolt_bridge_cycles(
                        new_tb_bridge, state.node_network
                    )
                )
        case RdmaCtlStatus():
            update["node_rdma_ctl"] = {
                **state.node_rdma_ctl,
                event.node_id: NodeRdmaCtlStatus(enabled=info.enabled),
            }

    return state.model_copy(update=update)


def apply_topology_edge_created(event: TopologyEdgeCreated, state: State) -> State:
    topology = copy.deepcopy(state.topology)
    topology.add_connection(event.conn)
    return state.model_copy(update={"topology": topology})


def apply_topology_edge_deleted(event: TopologyEdgeDeleted, state: State) -> State:
    topology = copy.deepcopy(state.topology)
    topology.remove_connection(event.conn)
    # TODO: Clean up removing the reverse connection
    return state.model_copy(update={"topology": topology})


def apply_overlay_peer_connected(event: OverlayPeerConnected, state: State) -> State:
    overlay_peers = {
        node_id: list(peers) for node_id, peers in state.overlay_peers.items()
    }

    for source, sink in (
        (event.local_node_id, event.remote_node_id),
        (event.remote_node_id, event.local_node_id),
    ):
        current = list(overlay_peers.get(source, ()))
        if sink not in current:
            current.append(sink)
            current.sort()
        overlay_peers[source] = current

    return state.model_copy(update={"overlay_peers": overlay_peers})


def apply_overlay_peer_disconnected(
    event: OverlayPeerDisconnected, state: State
) -> State:
    overlay_peers = {
        node_id: list(peers) for node_id, peers in state.overlay_peers.items()
    }

    for source, sink in (
        (event.local_node_id, event.remote_node_id),
        (event.remote_node_id, event.local_node_id),
    ):
        current = [peer for peer in overlay_peers.get(source, ()) if peer != sink]
        if current:
            overlay_peers[source] = current
        elif source in overlay_peers:
            overlay_peers.pop(source)

    return state.model_copy(update={"overlay_peers": overlay_peers})


def apply_overlay_bootstrap_peers_advertised(
    event: OverlayBootstrapPeersAdvertised, state: State
) -> State:
    overlay_advertised_peers = {
        **state.overlay_advertised_peers,
        event.node_id: list(event.peers),
    }
    return state.model_copy(
        update={"overlay_advertised_peers": overlay_advertised_peers}
    )


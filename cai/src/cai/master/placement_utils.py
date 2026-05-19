# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import os
from collections.abc import Generator, Mapping, Sequence
from datetime import UTC, datetime

from loguru import logger
from typing import Any
from urllib.parse import urlencode, urlparse

from cai.shared.models.model_cards import ModelCard
from cai.shared.network_model_policy import (
    private_network_model_min_pipeline_layers_per_node,
)
from cai.shared.network_routes import (
    llama_cpp_reachable_targets_by_source,
    relay_route_candidates,
)
from cai.shared.topology import Topology
from cai.shared.types.common import Host, NodeId
from cai.shared.types.memory import Memory
from cai.shared.types.profiling import MemoryUsage, NodeIdentity, NodeNetworkInfo
from cai.shared.types.topology import Cycle, RDMAConnection, SocketConnection
from cai.shared.types.worker.runners import RunnerId, ShardAssignments
from cai.shared.types.worker.instances import LlamaCppRelayRoute
from cai.shared.types.worker.shards import (
    CfgShardMetadata,
    PipelineShardMetadata,
    Sharding,
    ShardMetadata,
    TensorShardMetadata,
)

_LLAMA_CPP_RPC_ROUTE_TYPES = {
    "llama_cpp_rpc",
    "llama_cpp_rpc_direct",
    "llama_cpp_rpc_relay",
}
_DEFAULT_LLAMA_CPP_RPC_FAILURE_BACKOFF_SECONDS = 1800


def filter_cycles_by_memory(
    cycles: list[Cycle],
    node_memory: Mapping[NodeId, MemoryUsage],
    required_memory: Memory,
) -> list[Cycle]:
    filtered_cycles: list[Cycle] = []
    for cycle in cycles:
        if not all(node in node_memory for node in cycle):
            continue

        total_mem = sum(
            (node_memory[node_id].ram_available for node_id in cycle.node_ids),
            start=Memory(),
        )
        if total_mem >= required_memory:
            filtered_cycles.append(cycle)
    return filtered_cycles


def get_smallest_cycles(
    cycles: list[Cycle],
) -> list[Cycle]:
    min_nodes = min(len(cycle) for cycle in cycles)
    return [cycle for cycle in cycles if len(cycle) == min_nodes]


def _socket_adjacency(
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


def get_llama_cpp_direct_candidate_cycles(
    topology: Topology,
    node_memory: Mapping[NodeId, MemoryUsage],
    min_nodes: int,
    *,
    required_nodes: set[NodeId] | None = None,
    route_health_records: Sequence[Any] | None = None,
) -> list[Cycle]:
    """Return coordinator-first node selections for distributed llama.cpp.

    The current llama.cpp runtime only requires rank 0 to reach each remote RPC
    worker directly. The remote workers do not need to form a full cycle.
    """
    visible_nodes = {
        node_id for node_id in topology.list_nodes() if node_id in node_memory
    }
    if required_nodes and not required_nodes.issubset(visible_nodes):
        missing_nodes = sorted(map(str, required_nodes.difference(visible_nodes)))
        raise ValueError(
            f"Missing required llama.cpp placement nodes: {missing_nodes}"
        )

    adjacency = _socket_adjacency(topology, visible_nodes)
    for source_node_id, sink_node_ids in _route_health_direct_adjacency(
        visible_nodes,
        route_health_records,
    ).items():
        adjacency.setdefault(source_node_id, set()).update(sink_node_ids)
    candidate_cycles: list[Cycle] = []
    seen_node_orders: set[tuple[str, ...]] = set()
    min_candidate_size = max(min_nodes, len(required_nodes or ()))

    for coordinator in sorted(visible_nodes, key=str):
        reachable = set(adjacency.get(coordinator, set()))
        if required_nodes:
            required_targets = set(required_nodes)
            required_targets.discard(coordinator)
            if not required_targets.issubset(reachable):
                continue

        if 1 + len(reachable) < min_candidate_size:
            continue

        ordered_targets = sorted(
            reachable,
            key=lambda node_id: (
                0 if required_nodes and node_id in required_nodes else 1,
                -node_memory[node_id].ram_available.in_bytes,
                str(node_id),
            ),
        )
        for size in range(min_candidate_size, 1 + len(ordered_targets) + 1):
            ordered_nodes = [coordinator, *ordered_targets[: size - 1]]
            if required_nodes and not required_nodes.issubset(ordered_nodes):
                continue
            dedupe_key = tuple(str(node_id) for node_id in ordered_nodes)
            if dedupe_key in seen_node_orders:
                continue
            seen_node_orders.add(dedupe_key)
            candidate_cycles.append(Cycle(node_ids=ordered_nodes))

    return candidate_cycles


def get_llama_cpp_relay_candidate_cycles(
    topology: Topology,
    node_memory: Mapping[NodeId, MemoryUsage],
    overlay_peers: Mapping[NodeId, Sequence[NodeId]] | None,
    node_identities: Mapping[NodeId, NodeIdentity] | None,
    min_nodes: int,
    *,
    required_nodes: set[NodeId] | None = None,
) -> list[Cycle]:
    visible_nodes = {
        node_id for node_id in topology.list_nodes() if node_id in node_memory
    }
    if required_nodes and not required_nodes.issubset(visible_nodes):
        missing_nodes = sorted(map(str, required_nodes.difference(visible_nodes)))
        raise ValueError(
            f"Missing required relay placement nodes: {missing_nodes}"
        )

    reachable_targets = llama_cpp_reachable_targets_by_source(
        topology,
        overlay_peers,
        node_identities,
        visible_nodes,
    )
    candidate_cycles: list[Cycle] = []
    seen_node_orders: set[tuple[str, ...]] = set()
    min_candidate_size = max(min_nodes, len(required_nodes or ()))

    for coordinator in sorted(visible_nodes, key=str):
        reachable = set(reachable_targets.get(coordinator, set()))
        if required_nodes:
            required_targets = set(required_nodes)
            required_targets.discard(coordinator)
            if not required_targets.issubset(reachable):
                continue

        if 1 + len(reachable) < min_candidate_size:
            continue

        ordered_targets = sorted(
            reachable,
            key=lambda node_id: (
                0 if required_nodes and node_id in required_nodes else 1,
                -node_memory[node_id].ram_available.in_bytes,
                str(node_id),
            ),
        )
        for size in range(min_candidate_size, 1 + len(ordered_targets) + 1):
            ordered_nodes = [coordinator, *ordered_targets[: size - 1]]
            if required_nodes and not required_nodes.issubset(ordered_nodes):
                continue
            dedupe_key = tuple(str(node_id) for node_id in ordered_nodes)
            if dedupe_key in seen_node_orders:
                continue
            seen_node_orders.add(dedupe_key)
            candidate_cycles.append(Cycle(node_ids=ordered_nodes))

    return candidate_cycles


def allocate_layers_proportionally(
    total_layers: int,
    memory_fractions: list[float],
    *,
    minimum_layers_per_node: int = 1,
) -> list[int]:
    n = len(memory_fractions)
    if n == 0:
        raise ValueError("Cannot allocate layers to an empty node list")
    minimum_layers_per_node = max(1, int(minimum_layers_per_node))
    required_minimum_layers = n * minimum_layers_per_node
    if total_layers < required_minimum_layers:
        requirement = (
            "1 layer per node"
            if minimum_layers_per_node == 1
            else f"{minimum_layers_per_node} layers per node"
        )
        raise ValueError(
            f"Cannot distribute {total_layers} layers across {n} nodes "
            f"(need at least {requirement})"
        )

    # Largest remainder: floor each, then distribute remainder by fractional part
    raw = [f * total_layers for f in memory_fractions]
    result = [int(r) for r in raw]
    by_remainder = sorted(range(n), key=lambda i: raw[i] - result[i], reverse=True)
    for i in range(total_layers - sum(result)):
        result[by_remainder[i]] += 1

    # Ensure the requested floor per node by taking from the largest donors.
    for i in range(n):
        while result[i] < minimum_layers_per_node:
            donors = [
                j
                for j in range(n)
                if j != i and result[j] > minimum_layers_per_node
            ]
            if not donors:
                raise ValueError(
                    "Cannot satisfy minimum layer allocation for every node"
                )
            max_idx = max(
                donors,
                key=lambda j: (
                    result[j] - minimum_layers_per_node,
                    result[j],
                    memory_fractions[j],
                ),
            )
            result[max_idx] -= 1
            result[i] += 1

    return result


def _validate_cycle(cycle: Cycle) -> None:
    if not cycle.node_ids:
        raise ValueError("Cannot create shard assignments for empty node cycle")


def _compute_total_memory(
    node_ids: list[NodeId],
    node_memory: Mapping[NodeId, MemoryUsage],
) -> Memory:
    total_memory = sum(
        (node_memory[node_id].ram_available for node_id in node_ids),
        start=Memory(),
    )
    if total_memory.in_bytes == 0:
        raise ValueError("Cannot create shard assignments: total available memory is 0")
    return total_memory


def _allocate_and_validate_layers(
    node_ids: list[NodeId],
    node_memory: Mapping[NodeId, MemoryUsage],
    total_memory: Memory,
    model_card: ModelCard,
) -> list[int]:
    minimum_layers_per_node = private_network_model_min_pipeline_layers_per_node(
        model_card.model_id,
        model_card=model_card,
    )
    layer_allocations = allocate_layers_proportionally(
        total_layers=model_card.n_layers,
        memory_fractions=[
            node_memory[node_id].ram_available / total_memory for node_id in node_ids
        ],
        minimum_layers_per_node=minimum_layers_per_node,
    )

    total_storage = model_card.storage_size
    total_layers = model_card.n_layers
    for i, node_id in enumerate(node_ids):
        node_layers = layer_allocations[i]
        required_memory = (total_storage * node_layers) // total_layers
        available_memory = node_memory[node_id].ram_available
        if required_memory > available_memory:
            raise ValueError(
                f"Node {i} ({node_id}) has insufficient memory: "
                f"requires {required_memory.in_gb:.2f} GB for {node_layers} layers, "
                f"but only has {available_memory.in_gb:.2f} GB available"
            )

    return layer_allocations


def get_shard_assignments_for_pipeline_parallel(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> ShardAssignments:
    """Create shard assignments for pipeline parallel execution."""
    world_size = len(cycle)
    use_cfg_parallel = model_card.uses_cfg and world_size >= 2 and world_size % 2 == 0

    if use_cfg_parallel:
        return _get_shard_assignments_for_cfg_parallel(model_card, cycle, node_memory)
    else:
        return _get_shard_assignments_for_pure_pipeline(model_card, cycle, node_memory)


def _get_shard_assignments_for_cfg_parallel(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> ShardAssignments:
    """Create shard assignments for CFG parallel execution.

    CFG parallel runs two independent pipelines. Group 0 processes the positive
    prompt, group 1 processes the negative prompt. The ring topology places
    group 1's ranks in reverse order so both "last stages" are neighbors for
    efficient CFG exchange.
    """
    _validate_cycle(cycle)

    world_size = len(cycle)
    cfg_world_size = 2
    pipeline_world_size = world_size // cfg_world_size

    # Allocate layers for one pipeline group (both groups run the same layers)
    pipeline_node_ids = cycle.node_ids[:pipeline_world_size]
    pipeline_memory = _compute_total_memory(pipeline_node_ids, node_memory)
    layer_allocations = _allocate_and_validate_layers(
        pipeline_node_ids, node_memory, pipeline_memory, model_card
    )

    # Ring topology: group 0 ascending [0,1,2,...], group 1 descending [...,2,1,0]
    # This places both last stages as neighbors for CFG exchange.
    position_to_cfg_pipeline = [(0, r) for r in range(pipeline_world_size)] + [
        (1, r) for r in reversed(range(pipeline_world_size))
    ]

    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for device_rank, node_id in enumerate(cycle.node_ids):
        cfg_rank, pipeline_rank = position_to_cfg_pipeline[device_rank]
        layers_before = sum(layer_allocations[:pipeline_rank])
        node_layers = layer_allocations[pipeline_rank]

        shard = CfgShardMetadata(
            model_card=model_card,
            device_rank=device_rank,
            world_size=world_size,
            start_layer=layers_before,
            end_layer=layers_before + node_layers,
            n_layers=model_card.n_layers,
            cfg_rank=cfg_rank,
            cfg_world_size=cfg_world_size,
            pipeline_rank=pipeline_rank,
            pipeline_world_size=pipeline_world_size,
        )

        runner_id = RunnerId()
        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    return ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )


def _get_shard_assignments_for_pure_pipeline(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> ShardAssignments:
    """Create shard assignments for pure pipeline execution."""
    _validate_cycle(cycle)
    total_memory = _compute_total_memory(cycle.node_ids, node_memory)

    layer_allocations = _allocate_and_validate_layers(
        cycle.node_ids, node_memory, total_memory, model_card
    )

    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for pipeline_rank, node_id in enumerate(cycle.node_ids):
        layers_before = sum(layer_allocations[:pipeline_rank])
        node_layers = layer_allocations[pipeline_rank]

        shard = PipelineShardMetadata(
            model_card=model_card,
            device_rank=pipeline_rank,
            world_size=len(cycle),
            start_layer=layers_before,
            end_layer=layers_before + node_layers,
            n_layers=model_card.n_layers,
        )

        runner_id = RunnerId()
        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    return ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )


def get_shard_assignments_for_tensor_parallel(
    model_card: ModelCard,
    cycle: Cycle,
):
    total_layers = model_card.n_layers
    world_size = len(cycle)
    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for i, node_id in enumerate(cycle):
        shard = TensorShardMetadata(
            model_card=model_card,
            device_rank=i,
            world_size=world_size,
            start_layer=0,
            end_layer=total_layers,
            n_layers=total_layers,
        )

        runner_id = RunnerId()

        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    shard_assignments = ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )

    return shard_assignments


def get_shard_assignments(
    model_card: ModelCard,
    cycle: Cycle,
    sharding: Sharding,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> ShardAssignments:
    match sharding:
        case Sharding.Pipeline:
            return get_shard_assignments_for_pipeline_parallel(
                model_card=model_card,
                cycle=cycle,
                node_memory=node_memory,
            )
        case Sharding.Tensor:
            return get_shard_assignments_for_tensor_parallel(
                model_card=model_card,
                cycle=cycle,
            )


def get_mlx_jaccl_devices_matrix(
    selected_cycle: list[NodeId],
    cycle_digraph: Topology,
) -> list[list[str | None]]:
    """Build connectivity matrix mapping device i to device j via RDMA interface names.

    The matrix element [i][j] contains the interface name on device i that connects
    to device j, or None if no connection exists or no interface name is found.
    Diagonal elements are always None.
    """
    num_nodes = len(selected_cycle)
    matrix: list[list[str | None]] = [
        [None for _ in range(num_nodes)] for _ in range(num_nodes)
    ]

    for i, node_i in enumerate(selected_cycle):
        for j, node_j in enumerate(selected_cycle):
            if i == j:
                continue

            for conn in cycle_digraph.get_all_connections_between(node_i, node_j):
                if isinstance(conn, RDMAConnection):
                    matrix[i][j] = conn.source_rdma_iface
                    break
            else:
                raise ValueError(
                    "Current jaccl backend requires all-to-all RDMA connections"
                )

    return matrix


def _find_connection_ip(
    node_i: NodeId,
    node_j: NodeId,
    cycle_digraph: Topology,
) -> Generator[str, None, None]:
    """Find all IP addresses that connect node i to node j."""
    for connection in cycle_digraph.get_all_connections_between(node_i, node_j):
        if isinstance(connection, SocketConnection):
            yield connection.sink_multiaddr.ip_address


def _find_ip_prioritised(
    node_id: NodeId,
    other_node_id: NodeId,
    cycle_digraph: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
    ring: bool,
) -> str | None:
    """Find an IP address between nodes with prioritization.

    Priority: ethernet > wifi > unknown > thunderbolt
    """
    ips = list(_find_connection_ip(node_id, other_node_id, cycle_digraph))
    if not ips:
        return None
    other_network = node_network.get(other_node_id, NodeNetworkInfo())
    ip_to_type = {
        iface.ip_address: iface.interface_type for iface in other_network.interfaces
    }

    # Ring should prioritise fastest connection. As a best-effort, we prioritise TB.
    # TODO: Profile and get actual connection speeds.
    if ring:
        priority = {
            "thunderbolt": 0,
            "maybe_ethernet": 1,
            "ethernet": 2,
            "wifi": 3,
            "unknown": 4,
        }

    # RDMA prefers ethernet coordinator
    else:
        priority = {
            "ethernet": 0,
            "wifi": 1,
            "unknown": 2,
            "maybe_ethernet": 3,
            "thunderbolt": 4,
        }
    return min(ips, key=lambda ip: priority.get(ip_to_type.get(ip, "unknown"), 2))


def _preferred_advertised_host(identity: NodeIdentity) -> str | None:
    preferred_data_endpoint = identity.preferred_transport_endpoint(
        purpose="data"
    )
    if preferred_data_endpoint is not None:
        advertised_host = str(preferred_data_endpoint.host or "").strip()
        if advertised_host:
            return advertised_host

    preferred_api_endpoint = identity.preferred_transport_endpoint(
        purpose="api"
    )
    if preferred_api_endpoint is not None:
        advertised_host = str(preferred_api_endpoint.host or "").strip()
        if advertised_host:
            return advertised_host

    advertised_host = str(identity.data_host or identity.api_host or "").strip()
    return advertised_host or None


def _preferred_advertised_data_port(identity: NodeIdentity) -> int | None:
    preferred_data_endpoint = identity.preferred_transport_endpoint(
        purpose="data",
        require_port=True,
    )
    if preferred_data_endpoint is not None:
        return preferred_data_endpoint.port
    return identity.data_port


def _preferred_route_endpoint(
    identity: NodeIdentity,
    *,
    purpose: str,
    route_type: str,
    require_port: bool = True,
) -> Host | None:
    route_specific_endpoint = identity.preferred_transport_endpoint(
        purpose=purpose,
        route_types=[route_type],
        require_port=require_port,
    )
    if route_specific_endpoint is not None:
        return Host(
            ip=str(route_specific_endpoint.host).strip(),
            port=int(route_specific_endpoint.port or 0),
        )

    preferred_endpoint = identity.preferred_transport_endpoint(
        purpose=purpose,
        require_port=require_port,
    )
    if preferred_endpoint is not None:
        return Host(
            ip=str(preferred_endpoint.host).strip(),
            port=int(preferred_endpoint.port or 0),
        )

    if purpose == "data":
        host = str(identity.data_host or "").strip()
        port = identity.data_port
    else:
        host = str(identity.api_host or "").strip()
        port = identity.api_port
    if host and port is not None and port > 0:
        return Host(ip=host, port=port)
    return None


def _build_http_url(host: str, port: int) -> str:
    normalized_host = str(host or "").strip()
    if ":" in normalized_host and not normalized_host.startswith("["):
        normalized_host = f"[{normalized_host}]"
    return f"http://{normalized_host}:{int(port)}"


def _preferred_route_host(
    identity: NodeIdentity,
    *,
    purpose: str,
    route_type: str,
) -> str | None:
    route_specific_endpoint = identity.preferred_transport_endpoint(
        purpose=purpose,
        route_types=[route_type],
        require_port=False,
    )
    if route_specific_endpoint is not None:
        host = str(route_specific_endpoint.host or "").strip()
        if host:
            return host

    preferred_endpoint = identity.preferred_transport_endpoint(
        purpose=purpose,
        require_port=False,
    )
    if preferred_endpoint is not None:
        host = str(preferred_endpoint.host or "").strip()
        if host:
            return host

    host = identity.data_host if purpose == "data" else identity.api_host
    normalized_host = str(host or "").strip()
    return normalized_host or None


def _preferred_runtime_data_endpoint(
    identity: NodeIdentity,
    *,
    route_type: str,
    ephemeral_port: int,
    target_ip: str | None = None,
) -> Host | None:
    data_endpoint = _preferred_route_endpoint(
        identity,
        purpose="data",
        route_type=route_type,
    )
    if data_endpoint is not None:
        return data_endpoint

    target_port = _preferred_advertised_data_port(identity) or ephemeral_port
    if target_port <= 0:
        return None

    target_host = (
        str(target_ip or "").strip()
        or _preferred_route_host(identity, purpose="data", route_type=route_type)
        or _preferred_route_host(identity, purpose="api", route_type=route_type)
    )
    if not target_host:
        return None

    return Host(ip=target_host, port=target_port)


def get_llama_cpp_relay_routes_by_node(
    selected_cycle: Cycle,
    cycle_digraph: Topology,
    ephemeral_port: int,
    node_network: Mapping[NodeId, NodeNetworkInfo],
    node_identities: Mapping[NodeId, NodeIdentity],
    overlay_peers: Mapping[NodeId, Sequence[NodeId]] | None,
    route_health_records: Sequence[Any] | None = None,
) -> dict[NodeId, list[LlamaCppRelayRoute]]:
    selected_route_candidates = relay_route_candidates(
        cycle_digraph,
        overlay_peers,
        node_identities,
        selected_cycle.node_ids,
    )
    alternative_route_candidates = relay_route_candidates(
        cycle_digraph,
        overlay_peers,
        node_identities,
        selected_cycle.node_ids,
        include_alternatives=True,
    )
    route_candidates = _dedupe_relay_candidate_dicts(
        [*selected_route_candidates, *alternative_route_candidates]
    )
    if not route_candidates:
        return {}

    candidates_by_pair: dict[tuple[str, str], list[dict[str, object]]] = {}
    for route in route_candidates:
        pair = (
            str(route.get("sourceNodeId") or "").strip(),
            str(route.get("sinkNodeId") or "").strip(),
        )
        if not all(pair):
            continue
        candidates_by_pair.setdefault(pair, []).append(route)

    routes_by_node: dict[NodeId, list[LlamaCppRelayRoute]] = {}
    for source_node_id in selected_cycle.node_ids:
        source_routes: list[LlamaCppRelayRoute] = []
        for sink_node_id in selected_cycle.node_ids:
            if sink_node_id == source_node_id:
                continue
            if _route_health_directly_reachable(
                source_node_id,
                sink_node_id,
                route_health_records,
            ):
                continue
            pair_routes = candidates_by_pair.get((str(source_node_id), str(sink_node_id)))
            if not pair_routes:
                continue

            for route in pair_routes:
                transit_node_id = NodeId(str(route.get("transitNodeId") or "").strip())
                source_segment_type = str(route.get("sourceSegmentType") or "direct")
                sink_segment_type = str(route.get("sinkSegmentType") or "direct")
                relay_identity = node_identities.get(transit_node_id)
                sink_identity = node_identities.get(sink_node_id)
                if relay_identity is None or sink_identity is None:
                    continue

                relay_api = _preferred_route_endpoint(
                    relay_identity,
                    purpose="api",
                    route_type=source_segment_type,
                )
                if relay_api is None and source_segment_type == "direct":
                    relay_api_ip = _find_ip_prioritised(
                        source_node_id,
                        transit_node_id,
                        cycle_digraph,
                        node_network,
                        ring=False,
                    )
                    if relay_api_ip and relay_identity.api_port:
                        relay_api = Host(ip=relay_api_ip, port=relay_identity.api_port)
                if relay_api is None:
                    continue

                target_ip = None
                if sink_segment_type == "direct":
                    target_ip = _find_ip_prioritised(
                        transit_node_id,
                        sink_node_id,
                        cycle_digraph,
                        node_network,
                        ring=False,
                    )
                target = _preferred_runtime_data_endpoint(
                    sink_identity,
                    route_type=sink_segment_type,
                    ephemeral_port=ephemeral_port,
                    target_ip=target_ip,
                )
                if target is None:
                    continue

                source_routes.append(
                    LlamaCppRelayRoute(
                        source_node_id=source_node_id,
                        transit_node_id=transit_node_id,
                        sink_node_id=sink_node_id,
                        relay_api_host=relay_api.ip,
                        relay_api_port=relay_api.port,
                        target_host=target.ip,
                        target_port=target.port,
                        source_segment_type=source_segment_type,
                        sink_segment_type=sink_segment_type,
                    )
                )

        if source_routes:
            routes_by_node[source_node_id] = source_routes

    return routes_by_node


def _dedupe_relay_candidate_dicts(
    routes: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    deduped: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for route in routes:
        key = (
            str(route.get("sourceNodeId") or "").strip(),
            str(route.get("sinkNodeId") or "").strip(),
            str(route.get("transitNodeId") or "").strip(),
        )
        if not all(key) or key in seen:
            continue
        seen.add(key)
        deduped.append(dict(route))
    return deduped


def get_cai_api_urls_by_node(
    selected_cycle: Cycle,
    node_identities: Mapping[NodeId, NodeIdentity],
    relay_routes_by_node: Mapping[NodeId, Sequence[LlamaCppRelayRoute]] | None = None,
) -> dict[NodeId, list[str]]:
    api_urls_by_node: dict[NodeId, list[str]] = {}
    selected_node_ids = set(selected_cycle.node_ids)
    for node_id in selected_cycle:
        identity = node_identities.get(node_id, NodeIdentity())
        urls: list[str] = []
        seen: set[str] = set()
        for endpoint in identity.transport_endpoints_for(
            purpose="api",
            require_port=True,
        ):
            host = str(endpoint.host or "").strip()
            port = int(endpoint.port or 0)
            if not host or host in {"0.0.0.0", "::"} or port <= 0:
                continue
            url = _build_http_url(host, port)
            if url in seen:
                continue
            seen.add(url)
            urls.append(url)

        host = str(identity.api_host or "").strip()
        port = int(identity.api_port or 0)
        if host and host not in {"0.0.0.0", "::"} and port > 0:
            url = _build_http_url(host, port)
            if url not in seen:
                seen.add(url)
                urls.append(url)

        for url in _cai_owned_overlay_urls_for_node(
            node_id,
            selected_node_ids=selected_node_ids,
            node_identities=node_identities,
            relay_routes_by_node=relay_routes_by_node,
        ):
            if url in seen:
                continue
            seen.add(url)
            urls.append(url)

        api_urls_by_node[node_id] = urls
    return api_urls_by_node


def _cai_owned_overlay_urls_for_node(
    target_node_id: NodeId,
    *,
    selected_node_ids: set[NodeId],
    node_identities: Mapping[NodeId, NodeIdentity],
    relay_routes_by_node: Mapping[NodeId, Sequence[LlamaCppRelayRoute]] | None,
) -> list[str]:
    if not relay_routes_by_node:
        return []

    urls: list[str] = []
    seen: set[str] = set()
    for routes in relay_routes_by_node.values():
        for route in routes:
            if route.sink_node_id != target_node_id:
                continue
            relay_host = str(route.relay_api_host or "").strip()
            try:
                relay_port = int(route.relay_api_port or 0)
            except (TypeError, ValueError):
                continue
            if not relay_host or relay_host in {"0.0.0.0", "::"} or relay_port <= 0:
                continue
            relay_url = _build_http_url(relay_host, relay_port)
            query = urlencode(
                {
                    "targetNodeId": str(target_node_id),
                    "relayRole": _cai_owned_overlay_relay_role(
                        route.transit_node_id,
                        selected_node_ids=selected_node_ids,
                        node_identities=node_identities,
                    ),
                    "transitNodeId": str(route.transit_node_id),
                }
            )
            url = f"cai-overlay:{relay_url}?{query}"
            if url in seen:
                continue
            seen.add(url)
            urls.append(url)
    return urls


def _cai_owned_overlay_relay_role(
    transit_node_id: NodeId,
    *,
    selected_node_ids: set[NodeId],
    node_identities: Mapping[NodeId, NodeIdentity],
) -> str:
    if transit_node_id in selected_node_ids:
        return "ordinary"
    identity = node_identities.get(transit_node_id)
    if identity is not None and bool(getattr(identity, "worker_enabled", False)):
        return "ordinary"
    return "bootstrap"


def get_mlx_ring_hosts_by_node(
    selected_cycle: Cycle,
    cycle_digraph: Topology,
    ephemeral_port: int,
    node_network: Mapping[NodeId, NodeNetworkInfo],
    node_identities: Mapping[NodeId, NodeIdentity],
) -> dict[NodeId, list[Host]]:
    """Generate per-node host lists for MLX ring backend.

    Each node gets a list where:
    - Self position: Host(ip="0.0.0.0", port=ephemeral_port)
    - Left/right neighbors: actual connection IPs
    - Non-neighbors: Host(ip="198.51.100.1", port=0) placeholder (RFC 5737 TEST-NET-2)
    """
    world_size = len(selected_cycle)
    if world_size == 0:
        return {}

    hosts_by_node: dict[NodeId, list[Host]] = {}

    for rank, node_id in enumerate(selected_cycle):
        left_rank = (rank - 1) % world_size
        right_rank = (rank + 1) % world_size

        hosts_for_node: list[Host] = []
        current_identity = node_identities.get(node_id, NodeIdentity())
        current_data_port = _preferred_advertised_data_port(current_identity)
        local_bind_port = (
            current_data_port if current_data_port is not None else ephemeral_port
        )

        for idx, other_node_id in enumerate(selected_cycle):
            if idx == rank:
                hosts_for_node.append(Host(ip="0.0.0.0", port=local_bind_port))
                continue

            if idx not in {left_rank, right_rank}:
                # Placeholder IP from RFC 5737 TEST-NET-2
                hosts_for_node.append(Host(ip="198.51.100.1", port=0))
                continue

            connection_ip = _find_ip_prioritised(
                node_id, other_node_id, cycle_digraph, node_network, ring=True
            )
            other_identity = node_identities.get(other_node_id, NodeIdentity())
            target_ip = _preferred_advertised_host(other_identity) or connection_ip
            other_data_port = _preferred_advertised_data_port(other_identity)
            target_port = (
                other_data_port if other_data_port is not None else ephemeral_port
            )
            if target_ip is None:
                raise ValueError(
                    "MLX ring backend requires connectivity between neighbouring nodes"
                )

            hosts_for_node.append(Host(ip=target_ip, port=target_port))

        hosts_by_node[node_id] = hosts_for_node

    return hosts_by_node


def get_llama_cpp_hosts_by_node(
    selected_cycle: Cycle,
    cycle_digraph: Topology,
    ephemeral_port: int,
    node_network: Mapping[NodeId, NodeNetworkInfo],
    node_identities: Mapping[NodeId, NodeIdentity],
    relay_routes_by_node: Mapping[NodeId, Sequence[LlamaCppRelayRoute]] | None = None,
    route_health_records: Sequence[Any] | None = None,
) -> dict[NodeId, list[Host]]:
    """Generate per-node host lists for distributed llama.cpp RPC backends."""
    world_size = len(selected_cycle)
    if world_size == 0:
        return {}

    hosts_by_node: dict[NodeId, list[Host]] = {}

    for rank, node_id in enumerate(selected_cycle):
        hosts_for_node: list[Host] = []
        relay_sink_node_ids = {
            route.sink_node_id
            for route in (relay_routes_by_node or {}).get(node_id, [])
        }
        current_identity = node_identities.get(node_id, NodeIdentity())
        current_data_port = _preferred_advertised_data_port(current_identity)
        local_bind_port = (
            current_data_port if current_data_port is not None else ephemeral_port
        )
        for idx, other_node_id in enumerate(selected_cycle):
            if idx == rank:
                hosts_for_node.append(Host(ip="0.0.0.0", port=local_bind_port))
                continue
            if other_node_id in relay_sink_node_ids:
                hosts_for_node.append(Host(ip="198.51.100.1", port=0))
                continue

            other_identity = node_identities.get(other_node_id, NodeIdentity())
            target_ip = _route_health_direct_endpoint_host(
                node_id,
                other_node_id,
                route_health_records,
            ) or _preferred_advertised_host(
                other_identity
            ) or _find_ip_prioritised(
                node_id,
                other_node_id,
                cycle_digraph,
                node_network,
                ring=False,
            )
            other_data_port = _preferred_advertised_data_port(other_identity)
            target_port = (
                other_data_port if other_data_port is not None else ephemeral_port
            )
            if target_ip is None:
                if rank != 0:
                    hosts_for_node.append(Host(ip="0.0.0.0", port=local_bind_port))
                    continue
                raise ValueError(
                    "llama.cpp RPC backend requires the coordinator to reach every participating node"
                )

            hosts_for_node.append(Host(ip=target_ip, port=target_port))

        hosts_by_node[node_id] = hosts_for_node

    return hosts_by_node


def _route_health_direct_adjacency(
    visible_nodes: set[NodeId],
    route_health_records: Sequence[Any] | None,
) -> dict[NodeId, set[NodeId]]:
    adjacency: dict[NodeId, set[NodeId]] = {
        node_id: set() for node_id in visible_nodes
    }
    if not route_health_records:
        return adjacency

    for source_node_id in visible_nodes:
        for sink_node_id in visible_nodes:
            if source_node_id == sink_node_id:
                continue
            if _route_health_directly_reachable(
                source_node_id,
                sink_node_id,
                route_health_records,
            ):
                adjacency[source_node_id].add(sink_node_id)
    return adjacency


def _route_health_directly_reachable(
    source_node_id: NodeId,
    sink_node_id: NodeId,
    route_health_records: Sequence[Any] | None,
) -> bool:
    if not route_health_records:
        return False

    source = str(source_node_id)
    sink = str(sink_node_id)
    rpc_status = _route_health_llama_cpp_rpc_status(
        source,
        sink,
        route_health_records,
        route_types={"llama_cpp_rpc", "llama_cpp_rpc_direct"},
    )
    if rpc_status is not None:
        return rpc_status

    for record in route_health_records:
        if not bool(_route_health_field(record, "reachable")):
            continue
        record_source = str(_route_health_field(record, "source_node_id") or "").strip()
        record_sink = str(_route_health_field(record, "sink_node_id") or "").strip()
        route_type = str(_route_health_field(record, "route_type") or "").strip()
        if (
            route_type in {"direct_data", "direct_socket"}
            and record_source == source
            and record_sink == sink
        ):
            return True
    return False


def _route_health_llama_cpp_rpc_status(
    source_node_id: str,
    sink_node_id: str,
    route_health_records: Sequence[Any] | None,
    *,
    route_types: set[str] | None = None,
) -> bool | None:
    if not route_health_records:
        return None

    allowed_route_types = route_types or _LLAMA_CPP_RPC_ROUTE_TYPES
    latest_checked_at = ""
    latest_key: tuple[str, int] | None = None
    latest_reachable: bool | None = None
    for index, record in enumerate(route_health_records):
        route_type = str(_route_health_field(record, "route_type") or "").strip()
        if route_type not in allowed_route_types:
            continue
        record_source = str(_route_health_field(record, "source_node_id") or "").strip()
        record_sink = str(_route_health_field(record, "sink_node_id") or "").strip()
        if record_source != source_node_id or record_sink != sink_node_id:
            continue
        checked_at = str(_route_health_field(record, "checked_at") or "")
        key = (checked_at, index)
        if latest_key is None or key > latest_key:
            latest_checked_at = checked_at
            latest_key = key
            latest_reachable = bool(_route_health_field(record, "reachable"))
    if latest_reachable is False and _llama_cpp_rpc_failure_backoff_expired(
        latest_checked_at
    ):
        return None
    return latest_reachable


def _llama_cpp_rpc_failure_backoff_expired(checked_at: str | None) -> bool:
    if not checked_at:
        return False
    try:
        parsed = datetime.fromisoformat(str(checked_at).replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    try:
        backoff_seconds = int(
            os.environ.get(
                "CAI_LLAMA_CPP_RPC_FAILURE_BACKOFF_SECONDS",
                str(_DEFAULT_LLAMA_CPP_RPC_FAILURE_BACKOFF_SECONDS),
            )
            or _DEFAULT_LLAMA_CPP_RPC_FAILURE_BACKOFF_SECONDS
        )
    except ValueError:
        backoff_seconds = _DEFAULT_LLAMA_CPP_RPC_FAILURE_BACKOFF_SECONDS
    if backoff_seconds <= 0:
        return False
    return (datetime.now(tz=UTC) - parsed.astimezone(UTC)).total_seconds() > backoff_seconds


def _route_health_direct_endpoint_host(
    source_node_id: NodeId,
    sink_node_id: NodeId,
    route_health_records: Sequence[Any] | None,
) -> str | None:
    if not route_health_records:
        return None

    source = str(source_node_id)
    sink = str(sink_node_id)
    for record in route_health_records:
        if not bool(_route_health_field(record, "reachable")):
            continue
        record_source = str(_route_health_field(record, "source_node_id") or "").strip()
        record_sink = str(_route_health_field(record, "sink_node_id") or "").strip()
        route_type = str(_route_health_field(record, "route_type") or "").strip()
        if route_type not in {
            "direct_data",
            "direct_socket",
            "llama_cpp_rpc_direct",
        }:
            continue
        if record_source != source or record_sink != sink:
            continue
        endpoint_url = str(_route_health_field(record, "endpoint_url") or "").strip()
        if not endpoint_url:
            continue
        parsed = urlparse(endpoint_url)
        if parsed.hostname:
            return parsed.hostname
    return None


def _route_health_field(record: Any, field_name: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(field_name)
    return getattr(record, field_name, None)


def get_mlx_jaccl_coordinators(
    coordinator: NodeId,
    coordinator_port: int,
    cycle_digraph: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> dict[NodeId, str]:
    """Get the coordinator addresses for MLX JACCL (rank 0 device).

    Select an IP address that each node can reach for the rank 0 node. Returns
    address in format "X.X.X.X:PORT" per node.
    """
    logger.debug(f"Selecting coordinator: {coordinator}")

    def get_ip_for_node(n: NodeId) -> str:
        if n == coordinator:
            return "0.0.0.0"

        ip = _find_ip_prioritised(
            n, coordinator, cycle_digraph, node_network, ring=False
        )
        if ip is not None:
            return ip

        raise ValueError(
            "Current jaccl backend requires all participating devices to be able to communicate"
        )

    return {
        n: f"{get_ip_for_node(n)}:{coordinator_port}"
        for n in cycle_digraph.list_nodes()
    }


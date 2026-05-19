# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import importlib
import os
import random
from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
from typing import Any, Sequence

from cai.master.placement_utils import (
    Cycle,
    filter_cycles_by_memory,
    get_cai_api_urls_by_node,
    get_llama_cpp_direct_candidate_cycles,
    get_llama_cpp_hosts_by_node,
    get_llama_cpp_relay_routes_by_node,
    get_llama_cpp_relay_candidate_cycles,
    get_mlx_jaccl_coordinators,
    get_mlx_jaccl_devices_matrix,
    get_mlx_ring_hosts_by_node,
    get_shard_assignments,
    get_smallest_cycles,
)
from cai.shared.network_model_policy import (
    enforce_private_network_model_request,
    is_private_network_model,
    private_network_model_min_pipeline_layers_per_node,
    private_network_model_ram_headroom,
)
from cai.shared.models.model_cards import InferenceBackend, ModelId
from cai.shared.topology import Topology
from cai.shared.types.commands import (
    CancelDownload,
    CreateInstance,
    DeleteInstance,
    DownloadCommand,
    PlaceInstance,
)
from cai.shared.types.common import NodeId
from cai.shared.types.events import (
    Event,
    InstanceCreated,
    InstanceDeleted,
    TaskStatusUpdated,
)
from cai.shared.types.memory import Memory
from cai.shared.types.profiling import MemoryUsage, NodeIdentity, NodeNetworkInfo
from cai.shared.types.tasks import Task, TaskId, TaskStatus
from cai.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
)
from cai.shared.types.worker.instances import (
    Instance,
    InstanceId,
    InstanceMeta,
    MlxJacclInstance,
    MlxRingInstance,
)
from cai.shared.types.worker.shards import PipelineShardMetadata, Sharding


@lru_cache(maxsize=1)
def _get_cai_model_distribution_module():
    try:
        return importlib.import_module("cai_compute_chain.model_distribution")
    except Exception:
        return None


@lru_cache(maxsize=1)
def _get_cai_route_health_module():
    try:
        return importlib.import_module("cai_compute_chain.route_health")
    except Exception:
        return None


@lru_cache(maxsize=1)
def _get_cai_decentralized_compute_module():
    try:
        return importlib.import_module("cai_compute_chain.decentralized_compute")
    except Exception:
        return None


def _cai_owned_transport_generation_enabled() -> bool:
    return (
        os.environ.get("CAI_OWNED_TRANSPORT_GENERATION_ENABLED", "").strip().lower()
        in {"1", "true", "yes", "on"}
    )


def _cai_owned_transport_generation_require_runtime_ready() -> bool:
    return (
        os.environ.get(
            "CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_RUNTIME_READY",
            "",
        )
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )


def _cai_owned_transport_minimum_relay_quorum() -> int:
    try:
        return max(
            0,
            int(os.environ.get("CAI_OWNED_TRANSPORT_GENERATION_MIN_RELAY_QUORUM", "0")),
        )
    except ValueError:
        return 0


def _effective_node_memory_for_model(
    model_id: ModelId,
    node_memory: Mapping[NodeId, MemoryUsage],
    *,
    model_card=None,
) -> dict[NodeId, MemoryUsage]:
    headroom = private_network_model_ram_headroom(
        model_id,
        model_card=model_card,
    )
    if headroom.in_bytes <= 0:
        return dict(node_memory)

    effective: dict[NodeId, MemoryUsage] = {}
    for node_id, usage in node_memory.items():
        adjusted_available = usage.ram_available - headroom
        if adjusted_available.in_bytes < 0:
            adjusted_available = Memory()
        effective[node_id] = usage.model_copy(update={"ram_available": adjusted_available})
    return effective


def random_ephemeral_port() -> int:
    port = random.randint(49153, 65535)
    return port - 1 if port <= 52415 else port


def add_instance_to_placements(
    command: CreateInstance,
    topology: Topology,
    current_instances: Mapping[InstanceId, Instance],
) -> Mapping[InstanceId, Instance]:
    # TODO: validate against topology

    return {**current_instances, command.instance.instance_id: command.instance}


def _get_node_download_fraction(
    node_id: NodeId,
    model_id: ModelId,
    download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> float:
    """Return the download fraction (0.0–1.0) for a model on a given node."""
    for progress in download_status.get(node_id, []):
        if progress.shard_metadata.model_card.model_id != model_id:
            continue
        match progress:
            case DownloadCompleted():
                return 1.0
            case DownloadOngoing():
                total = progress.download_progress.total.in_bytes
                return (
                    progress.download_progress.downloaded.in_bytes / total
                    if total > 0
                    else 0.0
                )
            case DownloadPending():
                total = progress.total.in_bytes
                return progress.downloaded.in_bytes / total if total > 0 else 0.0
            case DownloadFailed():
                return 0.0
    return 0.0


def _cycle_download_score(
    cycle: Cycle,
    model_id: ModelId,
    download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> float:
    """Sum of download fractions across all nodes in a cycle."""
    return sum(
        _get_node_download_fraction(node_id, model_id, download_status)
        for node_id in cycle
    )


def _cycle_chunk_coverage_score(
    cycle: Cycle,
    command: PlaceInstance,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> tuple[int, float]:
    if command.model_card.inference_backend != InferenceBackend.LlamaCpp:
        return (0, 0.0)

    model_distribution = _get_cai_model_distribution_module()
    if model_distribution is None:
        return (0, 0.0)

    try:
        manifest = model_distribution.select_model_package_manifest_for_model(
            str(command.model_card.model_id)
        )
    except Exception:
        return (0, 0.0)
    if manifest is None:
        return (0, 0.0)

    try:
        peer_inventory = model_distribution.build_chunk_inventory_index(
            manifest.catalog_id,
            manifest.version,
            source_kind=model_distribution.ChunkInventorySourceKind.PEER_CACHE,
        )
        shard_assignments = get_shard_assignments(
            command.model_card,
            cycle,
            command.sharding,
            node_memory,
        )
    except Exception:
        return (0, 0.0)

    ready_nodes = 0
    total_required_bytes = 0
    total_present_bytes = 0
    for node_id, runner_id in shard_assignments.node_to_runner.items():
        shard = shard_assignments.runner_to_shard.get(runner_id)
        if not isinstance(shard, PipelineShardMetadata):
            continue
        try:
            coverage = manifest.compute_chunk_coverage(
                peer_inventory.get(str(node_id), set()),
                start_layer=shard.start_layer,
                end_layer=shard.end_layer,
            )
        except Exception:
            continue
        total_required_bytes += int(getattr(coverage, "required_bytes", 0))
        total_present_bytes += int(getattr(coverage, "present_bytes", 0))
        if bool(getattr(coverage, "ready", False)):
            ready_nodes += 1

    coverage_ratio = (
        total_present_bytes / total_required_bytes
        if total_required_bytes > 0
        else 0.0
    )
    return (ready_nodes, min(coverage_ratio, 1.0))


def _cycle_route_health_score(
    cycle: Cycle,
    route_health_records: Sequence[Any] | None,
) -> tuple[int, int, int]:
    if len(cycle.node_ids) <= 1 or not route_health_records:
        return (0, 0, 0)

    route_health = _get_cai_route_health_module()
    if route_health is None:
        return (0, 0, 0)

    coordinator = cycle.node_ids[0]
    try:
        score = route_health.route_health_score_for_path(
            str(coordinator),
            [str(sink_node_id) for sink_node_id in cycle.node_ids[1:]],
            list(route_health_records),
        )
    except Exception:
        return (0, 0, 0)
    return tuple(score)


def _cycle_has_failed_llama_cpp_rpc_route(
    cycle: Cycle,
    route_health_records: Sequence[Any] | None,
) -> bool:
    if len(cycle.node_ids) <= 1 or not route_health_records:
        return False

    route_health = _get_cai_route_health_module()
    if route_health is None:
        return False

    coordinator = cycle.node_ids[0]
    for sink_node_id in cycle.node_ids[1:]:
        try:
            status = route_health.llama_cpp_rpc_status_for_pair(
                str(coordinator),
                str(sink_node_id),
                list(route_health_records),
            )
        except Exception:
            continue
        if status is False:
            return True
    return False


def _cycle_has_unproven_llama_cpp_rpc_route(
    cycle: Cycle,
    route_health_records: Sequence[Any] | None,
) -> bool:
    if len(cycle.node_ids) <= 1:
        return False

    route_health = _get_cai_route_health_module()
    if route_health is None or not route_health_records:
        return True

    coordinator = cycle.node_ids[0]
    for sink_node_id in cycle.node_ids[1:]:
        try:
            status = route_health.llama_cpp_rpc_status_for_pair(
                str(coordinator),
                str(sink_node_id),
                list(route_health_records),
            )
        except Exception:
            return True
        if status is not True:
            return True
    return False


def _cycle_compute_cell_score(
    cycle: Cycle,
    route_health_records: Sequence[Any] | None,
) -> tuple[int, int]:
    profile = _cycle_compute_cell_profile(cycle, route_health_records)
    if profile is None:
        return (0, 0)

    profile_name = str(profile.get("profile") or "")
    profile_rank = {
        "low_latency_sharded_cell": 4,
        "proven_unknown_latency_sharded_cell": 3,
        "unproven_sharded_cell": 2,
        "wan_risky_sharded_cell": 1,
        "failed_sharded_cell": 0,
    }.get(profile_name, 0)
    max_latency = profile.get("maxLatencyMs")
    latency_rank = 0
    if max_latency is not None:
        try:
            latency_rank = -int(float(max_latency) * 1000)
        except (TypeError, ValueError):
            latency_rank = 0
    return (profile_rank, latency_rank)


def _cycle_compute_cell_profile(
    cycle: Cycle,
    route_health_records: Sequence[Any] | None,
) -> dict[str, Any] | None:
    if len(cycle.node_ids) <= 1 or not route_health_records:
        return None

    route_health = _get_cai_route_health_module()
    if route_health is None:
        return None

    coordinator = cycle.node_ids[0]
    try:
        profile = route_health.llama_cpp_compute_cell_profile_for_path(
            str(coordinator),
            [str(sink_node_id) for sink_node_id in cycle.node_ids[1:]],
            list(route_health_records),
        )
    except Exception:
        return None
    return profile if isinstance(profile, dict) else None


def _cycle_has_wan_risky_llama_cpp_compute_cell(
    cycle: Cycle,
    route_health_records: Sequence[Any] | None,
) -> bool:
    profile = _cycle_compute_cell_profile(cycle, route_health_records)
    if profile is None:
        return False
    return str(profile.get("profile") or "") in {
        "wan_risky_sharded_cell",
        "failed_sharded_cell",
    }


def _cycle_cai_owned_transport_route_readiness(
    cycle: Cycle,
    route_health_records: Sequence[Any] | None,
) -> dict[str, Any] | None:
    if len(cycle.node_ids) <= 1:
        return {"status": "single_node", "ready": True}
    decentralized_compute = _get_cai_decentralized_compute_module()
    if decentralized_compute is None:
        return None
    coordinator = cycle.node_ids[0]
    try:
        readiness = decentralized_compute.cai_owned_transport_route_health_readiness(
            source_node_id=str(coordinator),
            sink_node_ids=[str(sink_node_id) for sink_node_id in cycle.node_ids[1:]],
            route_health_records=list(route_health_records or []),
            route_policy={
                "avoidSingleTransitBottleneck": True,
                "minimumRelayQuorum": _cai_owned_transport_minimum_relay_quorum(),
            },
        )
    except Exception:
        return None
    return readiness if isinstance(readiness, dict) else None


def _cycle_cai_owned_worker_readiness(
    cycle: Cycle,
    node_identities: Mapping[NodeId, NodeIdentity] | None,
) -> dict[str, Any]:
    require_runtime_ready = _cai_owned_transport_generation_require_runtime_ready()
    node_audits: list[dict[str, Any]] = []
    fatal_reasons: list[str] = []
    for node_id in cycle.node_ids:
        identity = (node_identities or {}).get(node_id)
        audit = _node_cai_owned_worker_readiness(
            node_id,
            identity,
            require_runtime_ready=require_runtime_ready,
        )
        node_audits.append(audit)
        if not bool(audit.get("ready")):
            fatal_reasons.append(
                str(audit.get("error") or audit.get("reason") or node_id)
            )
    return {
        "ready": not fatal_reasons,
        "requireRuntimeReady": require_runtime_ready,
        "fatalReasons": fatal_reasons,
        "nodeAudits": node_audits,
    }


def _node_cai_owned_worker_readiness(
    node_id: NodeId,
    identity: NodeIdentity | None,
    *,
    require_runtime_ready: bool,
) -> dict[str, Any]:
    if identity is None:
        return {
            "nodeId": str(node_id),
            "ready": False,
            "reason": "identity_missing",
            "error": "Worker identity is missing.",
        }
    if getattr(identity, "worker_enabled", None) is not True:
        return {
            "nodeId": str(node_id),
            "ready": False,
            "reason": "worker_disabled",
            "error": "Node is not advertising worker mode.",
        }
    readiness = getattr(identity, "readiness", {}) or {}
    cai_readiness = (
        readiness.get("caiOwnedTransport")
        if isinstance(readiness, Mapping)
        else None
    )
    if not isinstance(cai_readiness, Mapping):
        return {
            "nodeId": str(node_id),
            "ready": False,
            "reason": "cai_owned_readiness_missing",
            "error": "Node does not advertise CAI-owned transport readiness.",
        }
    status = str(cai_readiness.get("status") or "").strip().lower()
    implemented = bool(cai_readiness.get("implemented"))
    runtime_ready = bool(cai_readiness.get("runtimeReady"))
    runtime_ready_proof = cai_readiness.get("runtimeReadyProof")
    runtime_ready_proof_verified = (
        isinstance(runtime_ready_proof, Mapping)
        and runtime_ready_proof.get("verified") is True
    )
    self_test = cai_readiness.get("llmShardSelfTest")
    production_self_test_ready = (
        isinstance(self_test, Mapping)
        and bool(self_test.get("productionReady"))
        and self_test.get("generationProbeReady") is True
        and self_test.get("backendHealthReady") is not False
    )
    if status in {"failed", "error", "disabled", "off"}:
        return {
            "nodeId": str(node_id),
            "ready": False,
            "reason": f"cai_owned_{status}",
            "error": f"CAI-owned transport readiness is {status}.",
            "caiOwnedTransport": dict(cai_readiness),
        }
    if not implemented:
        return {
            "nodeId": str(node_id),
            "ready": False,
            "reason": "cai_owned_not_implemented",
            "error": "CAI-owned transport adapter is not implemented on this worker.",
            "caiOwnedTransport": dict(cai_readiness),
        }
    if require_runtime_ready and not (
        (runtime_ready or status == "ready")
        and runtime_ready_proof_verified
        and production_self_test_ready
    ):
        return {
            "nodeId": str(node_id),
            "ready": False,
            "reason": "cai_owned_runtime_not_ready",
            "error": (
                "CAI-owned transport runtime is not ready on this worker "
                "with a verified live proof and production LLM self-test."
            ),
            "caiOwnedTransport": dict(cai_readiness),
        }
    return {
        "nodeId": str(node_id),
        "ready": True,
        "reason": "ready",
        "runtimeReady": runtime_ready,
        "caiOwnedTransport": dict(cai_readiness),
    }


def _has_failed_llama_cpp_rpc_record(
    route_health_records: Sequence[Any] | None,
) -> bool:
    if not route_health_records:
        return False

    route_types = {"llama_cpp_rpc", "llama_cpp_rpc_direct", "llama_cpp_rpc_relay"}
    route_health = _get_cai_route_health_module()
    for record in route_health_records:
        route_type = str(getattr(record, "route_type", "") or "").strip()
        if isinstance(record, Mapping):
            route_type = str(record.get("route_type") or "").strip()
            reachable = bool(record.get("reachable"))
            source_node_id = str(record.get("source_node_id") or "").strip()
            sink_node_id = str(record.get("sink_node_id") or "").strip()
        else:
            reachable = bool(getattr(record, "reachable", False))
            source_node_id = str(getattr(record, "source_node_id", "") or "").strip()
            sink_node_id = str(getattr(record, "sink_node_id", "") or "").strip()
        if route_type in route_types and not reachable:
            if route_health is not None and source_node_id and sink_node_id:
                try:
                    status = route_health.llama_cpp_rpc_status_for_pair(
                        source_node_id,
                        sink_node_id,
                        list(route_health_records),
                    )
                    if status is False:
                        return True
                    continue
                except Exception:
                    pass
            return True
    return False


def _requires_llama_cpp_rpc_proof(command: PlaceInstance) -> bool:
    if command.model_card.inference_backend != InferenceBackend.LlamaCpp:
        return False
    if command.min_nodes <= 1:
        return False
    allow_bootstrap = os.environ.get(
        "CAI_ALLOW_LLAMA_CPP_RPC_BOOTSTRAP_WITHOUT_PROOF",
        "",
    )
    if allow_bootstrap.strip().lower() in {"1", "true", "yes", "on"}:
        return False
    raw = os.environ.get("CAI_REQUIRE_LLAMA_CPP_RPC_PROOF")
    if raw is None:
        return is_private_network_model(
            command.model_card.model_id,
            model_card=command.model_card,
        )
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _canonical_cycle_for_model(
    cycle: Cycle,
    command: PlaceInstance,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> Cycle:
    if command.model_card.inference_backend != InferenceBackend.LlamaCpp:
        return cycle
    if len(cycle.node_ids) <= 1:
        return cycle
    coordinator = cycle.node_ids[0]
    return Cycle(
        node_ids=[
            coordinator,
            *sorted(
                cycle.node_ids[1:],
                key=lambda node_id: (
                    -node_memory[node_id].ram_available.in_bytes,
                    str(node_id),
                ),
            ),
        ]
    )


def _overlay_connected_candidate_cycles(
    topology: Topology,
    node_memory: Mapping[NodeId, MemoryUsage],
    overlay_peers: Mapping[NodeId, Sequence[NodeId]] | None,
    min_nodes: int,
    *,
    required_nodes: set[NodeId] | None = None,
) -> list[Cycle]:
    if not overlay_peers:
        return []

    visible_nodes = {
        node_id for node_id in topology.list_nodes() if node_id in node_memory
    }
    if required_nodes and not required_nodes.issubset(visible_nodes):
        missing_nodes = sorted(map(str, required_nodes.difference(visible_nodes)))
        raise ValueError(
            f"Missing required overlay placement nodes: {missing_nodes}"
        )

    adjacency: dict[NodeId, set[NodeId]] = {
        node_id: set() for node_id in visible_nodes
    }
    for source, peers in overlay_peers.items():
        if source not in visible_nodes:
            continue
        for peer in peers:
            if peer not in visible_nodes:
                continue
            adjacency[source].add(peer)
            adjacency[peer].add(source)

    candidate_cycles: list[Cycle] = []
    visited: set[NodeId] = set()
    ordered_visible_nodes = sorted(visible_nodes, key=str)
    min_component_size = max(min_nodes, len(required_nodes or ()))

    for start_node in ordered_visible_nodes:
        if start_node in visited:
            continue

        component: set[NodeId] = set()
        stack = [start_node]
        visited.add(start_node)
        while stack:
            node_id = stack.pop()
            component.add(node_id)
            for peer in adjacency[node_id]:
                if peer in visited:
                    continue
                visited.add(peer)
                stack.append(peer)

        if len(component) < min_component_size:
            continue
        if required_nodes and not required_nodes.issubset(component):
            continue

        ordered_nodes = sorted(
            component,
            key=lambda node_id: (
                0 if required_nodes and node_id in required_nodes else 1,
                -node_memory[node_id].ram_available.in_bytes,
                str(node_id),
            ),
        )
        for size in range(min_component_size, len(ordered_nodes) + 1):
            candidate_cycles.append(Cycle(node_ids=ordered_nodes[:size]))

    return candidate_cycles


def _visible_relay_count(
    node_identities: Mapping[NodeId, NodeIdentity] | None,
) -> int:
    if not node_identities:
        return 0
    return sum(
        1
        for identity in node_identities.values()
        if bool(getattr(identity, "relay_enabled", False))
    )


def _llama_cpp_unavailable_message(
    *,
    model_id: ModelId,
    min_nodes: int,
    available_nodes: int,
    connection_count: int,
    relay_count: int,
    relay_plannable_nodes: int = 0,
) -> str:
    message = (
        f"Distributed llama.cpp model {model_id} requires at least {min_nodes} "
        f"worker node(s) where one coordinator can reach every other participant "
        f"directly, but topology currently exposes {available_nodes} node(s) and "
        f"{connection_count} direct connection(s)."
    )
    if relay_plannable_nodes >= min_nodes:
        message += (
            f" A relay-plannable route graph is visible for up to "
            f"{relay_plannable_nodes} node(s), but no executable relay "
            "transport endpoints were available for the selected worker set."
        )
        return message
    if relay_count > 0:
        message += (
            " Relay-capable nodes are visible, but overlay and non-worker relay "
            "hops are not yet used for the llama.cpp data plane."
        )
    return message


def place_instance(
    command: PlaceInstance,
    topology: Topology,
    current_instances: Mapping[InstanceId, Instance],
    node_memory: Mapping[NodeId, MemoryUsage],
    node_network: Mapping[NodeId, NodeNetworkInfo],
    node_identities: Mapping[NodeId, NodeIdentity] | None = None,
    overlay_peers: Mapping[NodeId, Sequence[NodeId]] | None = None,
    required_nodes: set[NodeId] | None = None,
    download_status: Mapping[NodeId, Sequence[DownloadProgress]] | None = None,
    route_health_records: Sequence[Any] | None = None,
) -> dict[InstanceId, Instance]:
    effective_node_memory = _effective_node_memory_for_model(
        command.model_card.model_id,
        node_memory,
        model_card=command.model_card,
    )
    effective_sharding, effective_min_nodes = enforce_private_network_model_request(
        command.model_card.model_id,
        command.sharding,
        command.min_nodes,
        model_card=command.model_card,
        available_nodes=len(effective_node_memory),
    )
    if (
        effective_sharding != command.sharding
        or effective_min_nodes != command.min_nodes
    ):
        command = command.model_copy(
            update={
                "sharding": effective_sharding,
                "min_nodes": effective_min_nodes,
            }
        )

    if command.model_card.inference_backend == InferenceBackend.LlamaCpp:
        if command.instance_meta != InstanceMeta.LlamaCpp:
            raise ValueError(
                "llama.cpp models require the LlamaCpp instance backend"
            )
        if command.sharding != Sharding.Pipeline:
            raise ValueError(
                "llama.cpp backend currently supports Pipeline sharding only"
            )

    cycles = topology.get_cycles()
    if (
        command.model_card.inference_backend == InferenceBackend.LlamaCpp
        and command.min_nodes > 1
    ):
        direct_candidate_cycles = get_llama_cpp_direct_candidate_cycles(
            topology,
            effective_node_memory,
            command.min_nodes,
            required_nodes=required_nodes,
            route_health_records=route_health_records,
        )
    else:
        direct_candidate_cycles = list(
            filter(lambda it: len(it) >= command.min_nodes, cycles)
        )
    candidate_cycles = direct_candidate_cycles

    if required_nodes:
        candidate_cycles = [
            cycle
            for cycle in candidate_cycles
            if required_nodes.issubset(cycle.node_ids)
        ]

    allow_overlay_candidate_cycles = not (
        command.model_card.inference_backend == InferenceBackend.LlamaCpp
        and command.min_nodes > 1
    )

    if (
        len(candidate_cycles) == 0
        and command.instance_meta != InstanceMeta.MlxJaccl
        and allow_overlay_candidate_cycles
    ):
        candidate_cycles = _overlay_connected_candidate_cycles(
            topology,
            effective_node_memory,
            overlay_peers,
            command.min_nodes,
            required_nodes=required_nodes,
        )

    if (
        len(candidate_cycles) == 0
        and required_nodes
        and len(direct_candidate_cycles) > 0
        and not (
            command.model_card.inference_backend == InferenceBackend.LlamaCpp
            and command.min_nodes > 1
        )
    ):
        raise ValueError(
            f"No cycle satisfies required nodes {sorted(map(str, required_nodes))} "
            f"for model {command.model_card.model_id}."
        )
    relay_candidate_cycles: list[Cycle] = []
    if len(candidate_cycles) == 0 and (
        command.model_card.inference_backend == InferenceBackend.LlamaCpp
        and command.min_nodes > 1
    ):
        relay_candidate_cycles = get_llama_cpp_relay_candidate_cycles(
            topology,
            effective_node_memory,
            overlay_peers,
            node_identities,
            command.min_nodes,
            required_nodes=required_nodes,
        )
        candidate_cycles = relay_candidate_cycles

    if len(candidate_cycles) == 0:
        available_nodes = len(list(topology.list_nodes()))
        connection_count = sum(1 for _ in topology.list_connections())
        if (
            command.model_card.inference_backend == InferenceBackend.LlamaCpp
            and command.min_nodes > 1
        ):
            if _has_failed_llama_cpp_rpc_record(route_health_records):
                raise ValueError(
                    "No usable decentralized llama.cpp RPC route remains for the "
                    "selected worker set. RouteHealth marks the latest distributed "
                    "RPC probe as failed, so placement is refusing to repeat a "
                    "known-broken model-parallel data path."
                )
            raise ValueError(
                _llama_cpp_unavailable_message(
                    model_id=command.model_card.model_id,
                    min_nodes=command.min_nodes,
                    available_nodes=available_nodes,
                    connection_count=connection_count,
                    relay_count=_visible_relay_count(node_identities),
                    relay_plannable_nodes=max(
                        (len(cycle.node_ids) for cycle in relay_candidate_cycles),
                        default=0,
                    ),
                )
            )
        raise ValueError(
            f"Model {command.model_card.model_id} requires at least "
            f"{command.min_nodes} routable node(s), but topology currently exposes "
            f"{available_nodes} node(s) and {connection_count} connection(s)."
        )
    cycles_with_sufficient_memory = filter_cycles_by_memory(
        candidate_cycles, effective_node_memory, command.model_card.storage_size
    )
    if len(cycles_with_sufficient_memory) == 0:
        raise ValueError("No cycles found with sufficient memory")

    if command.sharding == Sharding.Tensor:
        if not command.model_card.supports_tensor:
            raise ValueError(
                f"Requested Tensor sharding but this model does not support tensor parallelism: {command.model_card.model_id}"
            )
        # TODO: the condition here for tensor parallel is not correct, but it works good enough for now.
        kv_heads = command.model_card.num_key_value_heads
        cycles_with_sufficient_memory = [
            cycle
            for cycle in cycles_with_sufficient_memory
            if command.model_card.hidden_size % len(cycle) == 0
            and (kv_heads is None or kv_heads % len(cycle) == 0)
        ]
        if not cycles_with_sufficient_memory:
            raise ValueError(
                f"No tensor sharding found for model with "
                f"hidden_size={command.model_card.hidden_size}"
                f"{f', num_key_value_heads={kv_heads}' if kv_heads is not None else ''}"
                f" across candidate cycles"
            )
    if command.sharding == Sharding.Pipeline and command.model_card.model_id == ModelId(
        "deepseek-ai/DeepSeek-V3.1-GGUF"
    ):
        raise ValueError(
            "Pipeline parallelism is not supported for DeepSeek V3.1 (8-bit)"
        )
    if (
        command.sharding == Sharding.Pipeline
        and command.model_card.base_model.startswith("Gemma 4")
    ):
        cycles_with_sufficient_memory = [
            cycle for cycle in cycles_with_sufficient_memory if len(cycle) == 1
        ]
        if not cycles_with_sufficient_memory:
            raise ValueError(
                "Pipeline parallelism is not supported for Gemma 4; use tensor parallelism instead."
            )

    smallest_cycles = get_smallest_cycles(cycles_with_sufficient_memory)

    smallest_rdma_cycles = [
        cycle for cycle in smallest_cycles if topology.is_rdma_cycle(cycle)
    ]

    if command.instance_meta == InstanceMeta.MlxJaccl:
        if not smallest_rdma_cycles:
            raise ValueError(
                "Requested RDMA (MlxJaccl) but no RDMA-connected cycles available"
            )
        smallest_cycles = smallest_rdma_cycles
    cycles_with_leaf_nodes: list[Cycle] = [
        cycle
        for cycle in smallest_cycles
        if any(topology.node_is_leaf(node_id) for node_id in cycle)
    ]

    resolved_download_status = download_status or {}
    candidate_cycles = (
        cycles_with_leaf_nodes if cycles_with_leaf_nodes != [] else smallest_cycles
    )
    candidate_cycles = [
        _canonical_cycle_for_model(cycle, command, effective_node_memory)
        for cycle in candidate_cycles
    ]
    if command.instance_meta == InstanceMeta.LlamaCpp:
        runnable_cycles = [
            cycle
            for cycle in candidate_cycles
            if not _cycle_has_failed_llama_cpp_rpc_route(cycle, route_health_records)
        ]
        if runnable_cycles:
            candidate_cycles = runnable_cycles
        elif candidate_cycles:
            raise ValueError(
                "No usable decentralized llama.cpp RPC route remains for the "
                "selected worker set. RouteHealth marks the latest distributed "
                "RPC probe as failed, so placement is refusing to repeat a "
                "known-broken model-parallel data path."
            )
        if _requires_llama_cpp_rpc_proof(command):
            proven_cycles = [
                cycle
                for cycle in candidate_cycles
                if not _cycle_has_unproven_llama_cpp_rpc_route(
                    cycle,
                    route_health_records,
                )
            ]
            if proven_cycles:
                candidate_cycles = proven_cycles
            elif candidate_cycles:
                raise ValueError(
                    "No proven decentralized llama.cpp RPC route remains for the "
                    "selected worker set. Strict RPC proof is enabled, so every "
                    "coordinator-to-worker pair must have a successful llama.cpp "
                "RPC readiness or inference record in RouteHealth before "
                "model-parallel placement."
            )
        low_latency_cycles = [
            cycle
            for cycle in candidate_cycles
            if not _cycle_has_wan_risky_llama_cpp_compute_cell(
                cycle,
                route_health_records,
            )
        ]
        if low_latency_cycles:
            candidate_cycles = low_latency_cycles
        elif candidate_cycles:
            if _cai_owned_transport_generation_enabled():
                cai_owned_ready_cycles: list[Cycle] = []
                cai_owned_blockers: list[str] = []
                for cycle in candidate_cycles:
                    route_readiness = (
                        _cycle_cai_owned_transport_route_readiness(
                            cycle,
                            route_health_records,
                        )
                        or {}
                    )
                    worker_readiness = _cycle_cai_owned_worker_readiness(
                        cycle,
                        node_identities,
                    )
                    if bool(route_readiness.get("ready")) and bool(
                        worker_readiness.get("ready")
                    ):
                        cai_owned_ready_cycles.append(cycle)
                        continue
                    blockers = [
                        *(
                            str(item)
                            for item in route_readiness.get("fatalReasons", [])
                        ),
                        *(
                            str(item)
                            for item in worker_readiness.get("fatalReasons", [])
                        ),
                    ]
                    if blockers:
                        cai_owned_blockers.extend(blockers)
                if cai_owned_ready_cycles:
                    candidate_cycles = cai_owned_ready_cycles
                else:
                    detail = ""
                    if cai_owned_blockers:
                        detail = " " + "; ".join(
                            list(dict.fromkeys(cai_owned_blockers))[:3]
                        )
                    raise ValueError(
                        "No CAI-owned transport route proof remains for the "
                        "selected WAN-risky llama.cpp worker set. CAI-owned "
                        "generation is enabled, but every required DAG hop and "
                        f"worker readiness check must pass before placement.{detail}"
                    )
            else:
                raise ValueError(
                    "No low-latency decentralized llama.cpp compute-cell remains for "
                    "the selected worker set. RouteHealth marks the available "
                    "coordinator-to-worker RPC path as WAN-risky for standard "
                    "model-parallel decode; use a low-latency PC<->PC cell or a "
                    "CAI-owned WAN-safe transport before placement."
                )

    selected_cycle = max(
        candidate_cycles,
        key=lambda cycle: (
            _cycle_compute_cell_score(cycle, route_health_records),
            _cycle_route_health_score(cycle, route_health_records),
            _cycle_chunk_coverage_score(cycle, command, effective_node_memory),
            _cycle_download_score(
                cycle, command.model_card.model_id, resolved_download_status
            ),
            sum(
                (effective_node_memory[node_id].ram_available for node_id in cycle),
                start=Memory(),
            ),
        ),
    )

    # Single-node: force Pipeline/Ring (Tensor and Jaccl require multi-node)
    if len(selected_cycle) == 1:
        if command.instance_meta != InstanceMeta.LlamaCpp:
            command.instance_meta = InstanceMeta.MlxRing
        command.sharding = Sharding.Pipeline

    shard_assignments = get_shard_assignments(
        command.model_card, selected_cycle, command.sharding, effective_node_memory
    )
    _validate_private_pipeline_layer_distribution(
        command.model_card.model_id,
        shard_assignments,
    )

    cycle_digraph: Topology = topology.get_subgraph_from_nodes(selected_cycle.node_ids)

    instance_id = InstanceId()
    target_instances = dict(deepcopy(current_instances))

    match command.instance_meta:
        case InstanceMeta.MlxJaccl:
            # TODO(evan): shard assignments should contain information about ranks, this is ugly
            def get_device_rank(node_id: NodeId) -> int:
                runner_id = shard_assignments.node_to_runner[node_id]
                shard_metadata = shard_assignments.runner_to_shard.get(runner_id)
                assert shard_metadata is not None
                return shard_metadata.device_rank

            zero_node_ids = [
                node_id
                for node_id in selected_cycle.node_ids
                if get_device_rank(node_id) == 0
            ]
            assert len(zero_node_ids) == 1
            coordinator_node_id = zero_node_ids[0]

            mlx_jaccl_devices = get_mlx_jaccl_devices_matrix(
                [node_id for node_id in selected_cycle],
                cycle_digraph,
            )
            mlx_jaccl_coordinators = get_mlx_jaccl_coordinators(
                coordinator=coordinator_node_id,
                coordinator_port=random_ephemeral_port(),
                cycle_digraph=cycle_digraph,
                node_network=node_network,
            )
            target_instances[instance_id] = MlxJacclInstance(
                instance_id=instance_id,
                shard_assignments=shard_assignments,
                jaccl_devices=mlx_jaccl_devices,
                jaccl_coordinators=mlx_jaccl_coordinators,
            )
        case InstanceMeta.MlxRing | InstanceMeta.LlamaCpp:
            ephemeral_port = random_ephemeral_port()
            relay_routes_by_node = {}
            if command.instance_meta == InstanceMeta.LlamaCpp:
                relay_routes_by_node = get_llama_cpp_relay_routes_by_node(
                    selected_cycle=selected_cycle,
                    cycle_digraph=cycle_digraph,
                    ephemeral_port=ephemeral_port,
                    node_network=node_network,
                    node_identities=node_identities or {},
                    overlay_peers=overlay_peers,
                    route_health_records=route_health_records,
                )
                hosts_by_node = get_llama_cpp_hosts_by_node(
                    selected_cycle=selected_cycle,
                    cycle_digraph=cycle_digraph,
                    ephemeral_port=ephemeral_port,
                    node_network=node_network,
                    node_identities=node_identities or {},
                    relay_routes_by_node=relay_routes_by_node,
                    route_health_records=route_health_records,
                )
                coordinator_node_id = selected_cycle.node_ids[0]
                planned_relay_sinks = {
                    route.sink_node_id
                    for route in relay_routes_by_node.get(coordinator_node_id, [])
                }
                coordinator_hosts = hosts_by_node.get(coordinator_node_id, [])
                for idx, other_node_id in enumerate(selected_cycle.node_ids):
                    if other_node_id == coordinator_node_id or idx >= len(coordinator_hosts):
                        continue
                    coordinator_host = coordinator_hosts[idx]
                    if (
                        coordinator_host.port <= 0
                        or coordinator_host.ip in {"0.0.0.0", "198.51.100.1"}
                    ) and other_node_id not in planned_relay_sinks:
                        raise ValueError(
                            "llama.cpp relay fallback identified a candidate worker set, "
                            "but the coordinator could not derive executable direct or "
                            f"relay transport for node {other_node_id}."
                        )
            else:
                hosts_by_node = get_mlx_ring_hosts_by_node(
                    selected_cycle=selected_cycle,
                    cycle_digraph=cycle_digraph,
                    ephemeral_port=ephemeral_port,
                    node_network=node_network,
                    node_identities=node_identities or {},
                )
            target_instances[instance_id] = MlxRingInstance(
                instance_id=instance_id,
                shard_assignments=shard_assignments,
                hosts_by_node=hosts_by_node,
                ephemeral_port=ephemeral_port,
                relay_routes_by_node=relay_routes_by_node,
                cai_api_urls_by_node=get_cai_api_urls_by_node(
                    selected_cycle,
                    node_identities or {},
                    relay_routes_by_node=relay_routes_by_node,
                ),
            )

    return target_instances


def _validate_private_pipeline_layer_distribution(
    model_id: ModelId,
    shard_assignments,
) -> None:
    shard_cards = [
        shard.model_card
        for shard in shard_assignments.runner_to_shard.values()
        if getattr(shard, "model_card", None) is not None
    ]
    model_card = shard_cards[0] if shard_cards else None
    if not is_private_network_model(model_id, model_card=model_card):
        return

    minimum_layers = private_network_model_min_pipeline_layers_per_node(
        model_id,
        model_card=model_card,
    )
    if minimum_layers <= 1:
        return

    too_small: list[tuple[str, int]] = []
    for node_id, runner_id in shard_assignments.node_to_runner.items():
        shard = shard_assignments.runner_to_shard.get(runner_id)
        if not isinstance(shard, PipelineShardMetadata):
            continue
        layer_count = max(shard.end_layer - shard.start_layer, 0)
        if layer_count < minimum_layers:
            too_small.append((str(node_id), layer_count))

    if too_small:
        details = ", ".join(
            f"{node_id}={layer_count}" for node_id, layer_count in too_small
        )
        raise ValueError(
            f"Private network model {model_id} requires at least {minimum_layers} "
            f"pipeline layer(s) per node, but current placement would assign too few: {details}."
        )


def delete_instance(
    command: DeleteInstance,
    current_instances: Mapping[InstanceId, Instance],
) -> dict[InstanceId, Instance]:
    target_instances = dict(deepcopy(current_instances))
    if command.instance_id in target_instances:
        del target_instances[command.instance_id]
        return target_instances
    raise ValueError(f"Instance {command.instance_id} not found")


def get_transition_events(
    current_instances: Mapping[InstanceId, Instance],
    target_instances: Mapping[InstanceId, Instance],
    tasks: Mapping[TaskId, Task],
) -> Sequence[Event]:
    events: list[Event] = []

    # find instances to create
    for instance_id, instance in target_instances.items():
        if instance_id not in current_instances:
            events.append(
                InstanceCreated(
                    instance=instance,
                )
            )

    # find instances to delete
    for instance_id in current_instances:
        if instance_id not in target_instances:
            for task in tasks.values():
                if task.instance_id == instance_id and task.task_status in [
                    TaskStatus.Pending,
                    TaskStatus.Running,
                ]:
                    events.append(
                        TaskStatusUpdated(
                            task_status=TaskStatus.Cancelled,
                            task_id=task.task_id,
                        )
                    )

            events.append(
                InstanceDeleted(
                    instance_id=instance_id,
                )
            )

    return events


def cancel_unnecessary_downloads(
    instances: Mapping[InstanceId, Instance],
    download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> Sequence[DownloadCommand]:
    commands: list[DownloadCommand] = []
    currently_downloading = [
        (k, v.shard_metadata.model_card.model_id)
        for k, vs in download_status.items()
        for v in vs
        if isinstance(v, (DownloadOngoing))
    ]
    active_models = set(
        (
            node_id,
            instance.shard_assignments.runner_to_shard[runner_id].model_card.model_id,
        )
        for instance in instances.values()
        for node_id, runner_id in instance.shard_assignments.node_to_runner.items()
    )
    for pair in currently_downloading:
        if pair not in active_models:
            commands.append(CancelDownload(target_node_id=pair[0], model_id=pair[1]))

    return commands


# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import Any


def execution_node_id_attempts(
    worker_node_ids: list[str] | None,
    *,
    state_payload: dict[str, Any],
    cai_url: str,
    private_network_model: bool,
    requester_node_id: str | None = None,
    excluded_node_ids: list[str] | set[str] | tuple[str, ...] | None = None,
    env_flag_func: Callable[[str, bool], bool],
    resolve_local_node_id_from_state_payload_func: Callable[[dict[str, Any], str], str],
    private_network_node_id_attempts_func: Callable[[list[str] | None], list[list[str] | None]],
    dedupe_execution_node_id_attempts_func: Callable[
        [list[list[str] | None]],
        list[list[str] | None],
    ],
) -> list[list[str] | None]:
    excluded = {
        str(node_id).strip()
        for node_id in (excluded_node_ids or [])
        if str(node_id).strip()
    }
    if excluded and worker_node_ids:
        preferred_worker_node_ids = [
            node_id
            for node_id in worker_node_ids
            if str(node_id).strip() and str(node_id).strip() not in excluded
        ]
        preferred_attempts = (
            execution_node_id_attempts(
                preferred_worker_node_ids,
                state_payload=state_payload,
                cai_url=cai_url,
                private_network_model=private_network_model,
                requester_node_id=requester_node_id,
                env_flag_func=env_flag_func,
                resolve_local_node_id_from_state_payload_func=(
                    resolve_local_node_id_from_state_payload_func
                ),
                private_network_node_id_attempts_func=(
                    private_network_node_id_attempts_func
                ),
                dedupe_execution_node_id_attempts_func=(
                    dedupe_execution_node_id_attempts_func
                ),
            )
            if preferred_worker_node_ids
            else []
        )
        fallback_attempts = execution_node_id_attempts(
            worker_node_ids,
            state_payload=state_payload,
            cai_url=cai_url,
            private_network_model=private_network_model,
            requester_node_id=requester_node_id,
            env_flag_func=env_flag_func,
            resolve_local_node_id_from_state_payload_func=(
                resolve_local_node_id_from_state_payload_func
            ),
            private_network_node_id_attempts_func=private_network_node_id_attempts_func,
            dedupe_execution_node_id_attempts_func=(
                dedupe_execution_node_id_attempts_func
            ),
        )
        return dedupe_execution_node_id_attempts_func(
            preferred_attempts + fallback_attempts
        )

    if private_network_model:
        return private_network_node_id_attempts_func(worker_node_ids)
    if not worker_node_ids:
        return [worker_node_ids]
    if not env_flag_func("CAI_PREFER_REMOTE_WORKER_EXECUTION", True):
        return [worker_node_ids]

    local_node_id = (
        str(requester_node_id or "").strip()
        or resolve_local_node_id_from_state_payload_func(state_payload, cai_url)
    )
    if not local_node_id or local_node_id not in worker_node_ids:
        return [worker_node_ids]

    remote_worker_node_ids = [
        node_id for node_id in worker_node_ids if node_id != local_node_id
    ]
    if not remote_worker_node_ids:
        return [worker_node_ids]
    return [remote_worker_node_ids, worker_node_ids]


def dedupe_execution_node_id_attempts(
    attempts: list[list[str] | None],
) -> list[list[str] | None]:
    deduped: list[list[str] | None] = []
    seen: set[tuple[str, ...] | None] = set()
    for attempt in attempts:
        key = None if attempt is None else tuple(attempt)
        if key in seen:
            continue
        deduped.append(attempt)
        seen.add(key)
    return deduped


def private_network_node_id_attempts(
    worker_node_ids: list[str] | None,
    *,
    effective_private_worker_shard_minimum_func: Callable[..., int],
    env_positive_int_func: Callable[[str, int], int],
) -> list[list[str] | None]:
    if worker_node_ids is None:
        return [None]
    ordered_node_ids = list(
        dict.fromkeys(
            str(node_id).strip()
            for node_id in worker_node_ids
            if str(node_id).strip()
        )
    )
    if not ordered_node_ids:
        return [[]]
    minimum_worker_nodes = effective_private_worker_shard_minimum_func(
        available_worker_count=len(ordered_node_ids)
    )
    if len(ordered_node_ids) <= minimum_worker_nodes:
        return [ordered_node_ids]

    attempts: list[list[str] | None] = [ordered_node_ids]
    max_attempts = env_positive_int_func("CAI_PRIVATE_PLACEMENT_NODE_ATTEMPT_LIMIT", 32)
    for subset_size in range(minimum_worker_nodes, len(ordered_node_ids)):
        for subset in combinations(ordered_node_ids, subset_size):
            candidate = list(subset)
            if candidate not in attempts:
                attempts.append(candidate)
            if len(attempts) >= max_attempts:
                return attempts
    return attempts


def select_preferred_preview(
    previews: list[dict[str, Any]],
    *,
    prefer_multi_node: bool = False,
    requester_node_id: str | None = None,
    model_id: str | None = None,
    route_health_records: list[Any] | None = None,
    performance_records: list[Any] | None = None,
    preview_preference_key_func: Callable[[dict[str, Any]], tuple[int, int]],
    single_node_preview_preference_key_func: Callable[
        [dict[str, Any]],
        tuple[int, int],
    ],
    preview_execution_preference_key_func: Callable[..., tuple],
    preview_execution_preference_penalty_key_func: Callable[..., tuple[float, ...]],
) -> dict[str, Any] | None:
    usable_previews = [
        item
        for item in previews
        if isinstance(item, dict) and not item.get("error") and item.get("instance")
    ]
    if not usable_previews:
        return None
    if prefer_multi_node:
        return max(
            usable_previews,
            key=lambda item: (
                *preview_preference_key_func(item),
                *preview_execution_preference_key_func(
                    item,
                    requester_node_id=requester_node_id,
                    model_id=model_id,
                    route_health_records=route_health_records,
                    performance_records=performance_records,
                ),
            ),
        )
    return min(
        usable_previews,
        key=lambda item: (
            *single_node_preview_preference_key_func(item),
            *preview_execution_preference_penalty_key_func(
                item,
                requester_node_id=requester_node_id,
                model_id=model_id,
                route_health_records=route_health_records,
                performance_records=performance_records,
            ),
        ),
    )


def preview_preference_key(
    item: dict[str, Any],
    *,
    preview_participant_count_func: Callable[[dict[str, Any]], int],
) -> tuple[int, int]:
    participant_count = preview_participant_count_func(item)
    memory_delta_by_node = item.get("memory_delta_by_node") or {}
    return (
        participant_count,
        len(memory_delta_by_node) if isinstance(memory_delta_by_node, dict) else 0,
    )


def single_node_preview_preference_key(
    item: dict[str, Any],
    *,
    preview_participant_count_func: Callable[[dict[str, Any]], int],
) -> tuple[int, int]:
    participant_count = preview_participant_count_func(item)
    memory_delta_by_node = item.get("memory_delta_by_node") or {}
    return (
        participant_count,
        -(
            len(memory_delta_by_node)
            if isinstance(memory_delta_by_node, dict)
            else 0
        ),
    )


def preview_participant_count(
    item: dict[str, Any],
    *,
    instance_definition_participant_count_func: Callable[[Any], int],
) -> int:
    return instance_definition_participant_count_func(item.get("instance"))


def preview_participant_node_ids(item: dict[str, Any]) -> list[str]:
    instance_definition = item.get("instance")
    if not isinstance(instance_definition, dict) or not instance_definition:
        return []
    _instance_type, instance_payload = next(iter(instance_definition.items()))
    if not isinstance(instance_payload, dict):
        return []
    shard_assignments = instance_payload.get("shardAssignments") or {}
    if not isinstance(shard_assignments, dict):
        return []
    node_to_runner = shard_assignments.get("nodeToRunner") or {}
    if not isinstance(node_to_runner, dict):
        return []
    return [
        str(node_id).strip()
        for node_id in node_to_runner.keys()
        if str(node_id).strip()
    ]


def preview_execution_preference_key(
    item: dict[str, Any],
    *,
    requester_node_id: str | None,
    model_id: str | None,
    route_health_records: list[Any] | None,
    performance_records: list[Any] | None,
    preview_participant_node_ids_func: Callable[[dict[str, Any]], list[str]],
    route_health_score_for_path_func: Callable[..., tuple],
    execution_performance_preference_key_func: Callable[..., tuple],
) -> tuple[int, int, int, int, int, int, float, float, int]:
    requester = str(requester_node_id or "").strip()
    participant_node_ids = [
        node_id
        for node_id in preview_participant_node_ids_func(item)
        if node_id and node_id != requester
    ]
    if not requester or not participant_node_ids:
        return (0, 0, 0, 0, 5000, 0, -1_000_000.0, -1_000_000.0, 0)

    route_score = route_health_score_for_path_func(
        requester,
        participant_node_ids,
        route_health_records,
    )
    performance_keys = [
        execution_performance_preference_key_func(
            model_id=model_id,
            requester_node_id=requester,
            executor_node_id=node_id,
            performance_records=performance_records,
        )
        for node_id in participant_node_ids
    ]
    min_health = min((int(item[0]) for item in performance_keys), default=0)
    success_rate_sum = sum(int(item[1]) for item in performance_keys)
    timeout_penalty_sum = sum(int(item[2]) for item in performance_keys)
    response_preference_sum = sum(float(item[3]) for item in performance_keys)
    attempt_preference_sum = sum(float(item[4]) for item in performance_keys)
    failure_penalty_sum = sum(int(item[5]) for item in performance_keys)
    return (
        int(route_score[0]),
        int(route_score[1]),
        int(route_score[2]),
        min_health,
        success_rate_sum,
        timeout_penalty_sum,
        response_preference_sum,
        attempt_preference_sum,
        failure_penalty_sum,
    )


def preview_execution_preference_penalty_key(
    item: dict[str, Any],
    *,
    requester_node_id: str | None,
    model_id: str | None,
    route_health_records: list[Any] | None,
    performance_records: list[Any] | None,
    preview_execution_preference_key_func: Callable[..., tuple],
) -> tuple[float, ...]:
    preference = preview_execution_preference_key_func(
        item,
        requester_node_id=requester_node_id,
        model_id=model_id,
        route_health_records=route_health_records,
        performance_records=performance_records,
    )
    return tuple(-float(value) for value in preference)


def instance_definition_participant_count(instance_definition: Any) -> int:
    if not isinstance(instance_definition, dict) or not instance_definition:
        return 0
    _instance_type, instance_payload = next(iter(instance_definition.items()))
    if not isinstance(instance_payload, dict):
        return 0
    shard_assignments = instance_payload.get("shardAssignments") or {}
    if not isinstance(shard_assignments, dict):
        return 0
    node_to_runner = shard_assignments.get("nodeToRunner") or {}
    if not isinstance(node_to_runner, dict):
        return 0
    return len(node_to_runner)


def instances_have_model(payload: dict[str, Any], model_id: str) -> bool:
    for item in payload.values():
        if not isinstance(item, dict):
            continue
        for instance_payload in item.values():
            if not isinstance(instance_payload, dict):
                continue
            shard_assignments = instance_payload.get("shardAssignments") or {}
            if shard_assignments.get("modelId") == model_id:
                return True
    return False


def snapshot_from_instance_state_item(
    *,
    instance_id: str,
    instance_item: dict[str, Any],
    model_id: str | None = None,
    extract_instance_participants_func: Callable[
        [dict[str, Any]],
        list[dict[str, Any]],
    ],
) -> dict[str, Any] | None:
    for instance_payload in instance_item.values():
        if not isinstance(instance_payload, dict):
            continue
        shard_assignments = instance_payload.get("shardAssignments") or {}
        if not isinstance(shard_assignments, dict):
            continue
        current_model_id = shard_assignments.get("modelId")
        if model_id is not None and current_model_id != model_id:
            continue
        return {
            "instance_id": str(instance_id),
            "snapshot_source": "state",
            "participants": extract_instance_participants_func(shard_assignments),
            "relay_routes_by_node": instance_payload.get("relayRoutesByNode")
            or instance_payload.get("relay_routes_by_node")
            or {},
        }
    return None


def snapshot_from_instance_definition(
    instance_definition: dict[str, Any] | None,
    *,
    snapshot_source: str = "planned_definition",
    extract_instance_participants_func: Callable[
        [dict[str, Any]],
        list[dict[str, Any]],
    ],
) -> dict[str, Any] | None:
    if not isinstance(instance_definition, dict) or not instance_definition:
        return None
    instance_type, instance_payload = next(iter(instance_definition.items()))
    if not isinstance(instance_payload, dict):
        return None
    shard_assignments = instance_payload.get("shardAssignments") or {}
    if not isinstance(shard_assignments, dict):
        return None
    return {
        "instance_id": str(instance_payload.get("instanceId") or f"planned:{instance_type}"),
        "snapshot_source": snapshot_source,
        "participants": extract_instance_participants_func(shard_assignments),
        "relay_routes_by_node": instance_payload.get("relayRoutesByNode")
        or instance_payload.get("relay_routes_by_node")
        or {},
    }


def require_settleable_instance_snapshot(
    instance_snapshot: dict[str, Any] | None,
    *,
    task_level_transport_job_source: str,
) -> None:
    if not isinstance(instance_snapshot, dict):
        raise RuntimeError(
            "Cannot settle execution reward: CAI did not expose the actual "
            "instance used for the completed command."
        )
    snapshot_source = str(instance_snapshot.get("snapshot_source") or "").strip()
    if snapshot_source == "planned_definition":
        raise RuntimeError(
            "Cannot settle execution reward from a planned placement snapshot. "
            "CAI must expose the actual command-bound or live instance before "
            "worker rewards can be recorded."
        )
    if snapshot_source == task_level_transport_job_source:
        metadata = instance_snapshot.get("caiOwnedTaskLevelTransport") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        participant_range_source = str(
            metadata.get("participantRangeSource") or ""
        ).strip()
        executor_count = len(
            instance_snapshot.get("caiOwnedTransportExecutorNodeIds") or []
        )
        if (
            participant_range_source == "synthetic_executor_index_fallback"
            and executor_count > 1
        ):
            raise RuntimeError(
                "Cannot settle execution reward from a synthetic task-level shard "
                "snapshot. CAI must expose proof-backed or DAG-backed layer ranges "
                "before worker rewards can be recorded."
            )


def find_model_instance(
    instances_payload: dict[str, Any],
    model_id: str,
) -> dict[str, Any] | None:
    for instance_id, item in instances_payload.items():
        if not isinstance(item, dict):
            continue
        for instance_type, instance_payload in item.items():
            if not isinstance(instance_payload, dict):
                continue
            shard_assignments = instance_payload.get("shardAssignments") or {}
            if shard_assignments.get("modelId") != model_id:
                continue
            return {
                "instance_id": str(instance_id),
                "instance_type": str(instance_type),
                "payload": instance_payload,
            }
    return None


def instance_is_ready(
    instance: dict[str, Any],
    runners_payload: dict[str, Any],
    *,
    runner_status_name_func: Callable[[Any], str | None],
) -> bool:
    payload = instance.get("payload") or {}
    shard_assignments = payload.get("shardAssignments") or {}
    runner_to_shard = shard_assignments.get("runnerToShard") or {}
    runner_ids = [str(runner_id) for runner_id in runner_to_shard.keys()]
    if not runner_ids:
        return False

    for runner_id in runner_ids:
        runner_status = runners_payload.get(runner_id)
        status_name = runner_status_name_func(runner_status)
        if status_name not in {"RunnerReady", "RunnerRunning"}:
            return False
    return True


def runner_status_name(runner_status: Any) -> str | None:
    if not isinstance(runner_status, dict) or not runner_status:
        return None
    return str(next(iter(runner_status.keys())))


def extract_instance_participants(
    shard_assignments: dict[str, Any],
    *,
    unwrap_shard_metadata_func: Callable[[Any], dict[str, Any]],
    layer_count_from_metadata_func: Callable[[dict[str, Any]], int],
) -> list[dict[str, Any]]:
    runner_to_shard = shard_assignments.get("runnerToShard") or {}
    node_to_runner = shard_assignments.get("nodeToRunner") or {}
    participants: list[dict[str, Any]] = []
    for node_id, runner_id in node_to_runner.items():
        shard_payload = runner_to_shard.get(runner_id) or {}
        metadata = unwrap_shard_metadata_func(shard_payload)
        layer_start = metadata.get("startLayer")
        layer_end = metadata.get("endLayer")
        layer_count = layer_count_from_metadata_func(metadata)
        participants.append(
            {
                "node_id": str(node_id),
                "runner_id": str(runner_id),
                "layer_start": int(layer_start) if isinstance(layer_start, int) else None,
                "layer_end": int(layer_end) if isinstance(layer_end, int) else None,
                "layer_count": layer_count,
            }
        )
    participants.sort(
        key=lambda item: (
            item["layer_start"] is None,
            item["layer_start"] if item["layer_start"] is not None else 10**9,
            item["node_id"],
        )
    )
    return participants


def cai_cluster_node_count(state_payload: dict[str, Any]) -> int:
    topology = state_payload.get("topology") or {}
    if isinstance(topology, dict):
        nodes = topology.get("nodes") or []
        if isinstance(nodes, list):
            normalized_nodes = {str(node).strip() for node in nodes if str(node).strip()}
            if normalized_nodes:
                return len(normalized_nodes)
    node_identities = state_payload.get("nodeIdentities") or {}
    if isinstance(node_identities, dict):
        normalized_nodes = {
            str(node_id).strip()
            for node_id in node_identities.keys()
            if str(node_id).strip()
        }
        if normalized_nodes:
            return len(normalized_nodes)
    return 0

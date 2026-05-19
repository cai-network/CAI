# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .gguf_shard_policy import GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED


def worker_node_ids_from_audit(audit: dict[str, Any] | None) -> list[str] | None:
    if audit is None:
        return None
    eligible_node_ids = audit.get("eligibleNodeIds")
    if isinstance(eligible_node_ids, list) and eligible_node_ids:
        return sorted(str(node_id) for node_id in eligible_node_ids)
    if int(audit.get("checkedNodeCount") or 0) > 0:
        return []
    return None


def task_level_transport_gguf_shard_compatibility(
    model_id: str,
    *,
    wallet_policy: Any | None = None,
    curated_model_for_id_func: Callable[[str], Any | None],
    select_model_package_manifest_for_model_func: Callable[[str, Any | None], Any | None],
    gguf_shard_compatibility_func: Callable[..., Any],
):
    curated_model = curated_model_for_id_func(model_id)
    filename = None
    gguf_architecture = None
    model_format = None
    allow_full_model_local = False
    if curated_model is not None:
        filename = str(curated_model.preferred_filename or "").strip() or None
        gguf_architecture = str(curated_model.gguf_architecture or "").strip() or None
        model_format = str(curated_model.model_format or "").strip().lower()
        allow_full_model_local = bool(
            not curated_model.private_network and curated_model.allow_single_node_fallback
        )
    manifest = select_model_package_manifest_for_model_func(model_id, wallet_policy)
    manifest_metadata = (
        getattr(manifest, "metadata", {}) if manifest is not None else {}
    ) or {}
    if not isinstance(manifest_metadata, dict):
        manifest_metadata = {}
    if filename is None and manifest is not None:
        filename = str(getattr(manifest, "preferred_filename", "") or "").strip() or None
    if gguf_architecture is None:
        gguf_architecture = str(
            manifest_metadata.get("gguf_architecture")
            or manifest_metadata.get("ggufArchitecture")
            or getattr(manifest, "family", "")
            or ""
        ).strip() or None
    if model_format is None:
        model_format = str(
            manifest_metadata.get("model_format")
            or manifest_metadata.get("modelFormat")
            or ""
        ).strip().lower() or None
    if manifest is not None and curated_model is None:
        package_kind = str(getattr(manifest, "package_kind", "") or "").strip().lower()
        allow_full_model_local = package_kind == "public_shared"
    normalized_model_id = str(model_id or "").strip()
    looks_like_gguf = bool(
        (model_format == "gguf")
        or (filename and filename.lower().endswith(".gguf"))
        or ("gguf" in normalized_model_id.lower())
        or gguf_architecture
    )
    if not looks_like_gguf:
        return None
    return gguf_shard_compatibility_func(
        model_id=normalized_model_id,
        gguf_architecture=gguf_architecture,
        family=gguf_architecture,
        filename=filename,
        allow_full_model_local=allow_full_model_local,
    )


def task_level_transport_effective_executor_count(
    model_id: str,
    *,
    requested_executor_count: int | None = None,
    wallet_policy: Any | None = None,
    task_level_transport_executor_count_func: Callable[[], int],
    gguf_shard_compatibility_func: Callable[..., Any],
    total_layer_count_func: Callable[..., int],
    curated_model_for_id_func: Callable[[str], Any | None],
) -> int:
    requested_count = max(
        int(
            requested_executor_count
            if requested_executor_count is not None
            else task_level_transport_executor_count_func()
        ),
        1,
    )
    if requested_count <= 1:
        return 1
    compatibility = gguf_shard_compatibility_func(
        model_id,
        wallet_policy=wallet_policy,
    )
    if (
        compatibility is None
        or compatibility.shard_compatibility != GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED
        or not compatibility.layer_range_supported
    ):
        return 1
    total_layer_count = total_layer_count_func(
        model_id,
        executor_count=requested_count,
        wallet_policy=wallet_policy,
    )
    if total_layer_count <= 1:
        return 1
    curated_model = curated_model_for_id_func(model_id)
    minimum_worker_shards = 1
    if curated_model is not None:
        minimum_worker_shards = max(int(curated_model.minimum_worker_shards or 0), 1)
    effective_count = max(requested_count, minimum_worker_shards)
    effective_count = min(effective_count, int(total_layer_count))
    if effective_count < minimum_worker_shards:
        return 1
    return max(effective_count, 1)


def task_level_transport_total_layer_count(
    model_id: str,
    *,
    executor_count: int,
    wallet_policy: Any | None = None,
    curated_model_for_id_func: Callable[[str], Any | None],
    select_model_package_manifest_for_model_func: Callable[[str, Any | None], Any | None],
    optional_int_value_func: Callable[[Any], int | None],
    optional_int_field_value_func: Callable[..., int | None],
) -> int:
    curated_model = curated_model_for_id_func(model_id)
    if curated_model is not None and optional_int_value_func(curated_model.total_layers):
        return max(int(curated_model.total_layers or 0), 1)
    manifest = select_model_package_manifest_for_model_func(model_id, wallet_policy)
    if manifest is not None:
        total_layers = optional_int_field_value_func(
            getattr(manifest, "metadata", {}) or {},
            "total_layers",
            "totalLayers",
            "nLayers",
            "block_count",
        )
        if total_layers is not None and total_layers > 0:
            return max(int(total_layers), 1)
    return max(int(executor_count or 0), 1)


def task_level_transport_llm_runtime_metadata(
    model_id: str,
    *,
    total_layer_count: int,
    wallet_policy: Any | None = None,
    curated_model_for_id_func: Callable[[str], Any | None],
    select_model_package_manifest_for_model_func: Callable[[str, Any | None], Any | None],
    optional_int_value_func: Callable[[Any], int | None],
    optional_int_field_value_func: Callable[..., int | None],
) -> dict[str, Any] | None:
    curated_model = curated_model_for_id_func(model_id)
    total_layers = max(int(total_layer_count or 0), 1)
    if curated_model is not None and curated_model.layer_range_supported:
        hidden_size = optional_int_value_func(curated_model.hidden_size)
        if hidden_size is not None and hidden_size > 0:
            metadata: dict[str, Any] = {
                "metadataSource": "curated_model_policy",
                "modelId": str(curated_model.execution_model_id or model_id).strip(),
                "totalLayerCount": total_layers,
                "totalLayers": total_layers,
                "blockCount": total_layers,
                "hiddenSize": hidden_size,
                "nEmbd": hidden_size,
                "ggufArchitecture": str(curated_model.gguf_architecture or "").strip(),
                "shardCompatibility": str(curated_model.shard_compatibility or "").strip(),
                "layerRangeSupported": bool(curated_model.layer_range_supported),
                "stateFormat": str(curated_model.state_format or "").strip(),
                "activationStateFormat": str(
                    curated_model.activation_state_format or ""
                ).strip(),
                "decodeStateFormat": str(curated_model.decode_state_format or "").strip(),
                "preferredFilename": str(curated_model.preferred_filename or "").strip(),
                "layerRangeProbeAbi": str(curated_model.layer_range_probe_abi or "").strip(),
                "layerRangeProbeReport": str(
                    curated_model.layer_range_probe_report or ""
                ).strip(),
                "layerRangeEquivalenceProbeReport": str(
                    curated_model.layer_range_equivalence_probe_report or ""
                ).strip(),
            }
            return {
                key: value for key, value in metadata.items() if value not in ("", None)
            }

    manifest = select_model_package_manifest_for_model_func(model_id, wallet_policy)
    if manifest is None:
        return None
    manifest_metadata = getattr(manifest, "metadata", {}) or {}
    if not isinstance(manifest_metadata, dict):
        return None
    layer_range_supported = manifest_metadata.get("layer_range_supported")
    if layer_range_supported is None:
        layer_range_supported = manifest_metadata.get("layerRangeSupported")
    shard_compatibility = str(
        manifest_metadata.get("shard_compatibility")
        or manifest_metadata.get("shardCompatibility")
        or ""
    ).strip()
    if layer_range_supported is not True:
        return None
    if (
        shard_compatibility
        and shard_compatibility != GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED
    ):
        return None
    hidden_size = optional_int_field_value_func(
        manifest_metadata,
        "hidden_size",
        "hiddenSize",
        "embedding_length",
        "embeddingLength",
        "n_embd",
        "nEmbd",
        "nEmbedding",
    )
    if hidden_size is None or hidden_size <= 0:
        return None
    metadata = {
        "metadataSource": "model_package_manifest",
        "modelId": str(getattr(manifest, "model_id", None) or model_id).strip(),
        "totalLayerCount": total_layers,
        "totalLayers": total_layers,
        "blockCount": total_layers,
        "hiddenSize": hidden_size,
        "nEmbd": hidden_size,
        "ggufArchitecture": str(
            manifest_metadata.get("gguf_architecture")
            or manifest_metadata.get("ggufArchitecture")
            or ""
        ).strip(),
        "shardCompatibility": shard_compatibility,
        "layerRangeSupported": True,
        "stateFormat": str(
            manifest_metadata.get("state_format")
            or manifest_metadata.get("stateFormat")
            or ""
        ).strip(),
        "activationStateFormat": str(
            manifest_metadata.get("activation_state_format")
            or manifest_metadata.get("activationStateFormat")
            or ""
        ).strip(),
        "decodeStateFormat": str(
            manifest_metadata.get("decode_state_format")
            or manifest_metadata.get("decodeStateFormat")
            or ""
        ).strip(),
        "preferredFilename": str(
            getattr(manifest, "preferred_filename", None)
            or manifest_metadata.get("preferred_filename")
            or manifest_metadata.get("preferredFilename")
            or ""
        ).strip(),
        "layerRangeProbeAbi": str(
            manifest_metadata.get("layer_range_probe_abi")
            or manifest_metadata.get("layerRangeProbeAbi")
            or ""
        ).strip(),
        "layerRangeProbeReport": str(
            manifest_metadata.get("layer_range_probe_report")
            or manifest_metadata.get("layerRangeProbeReport")
            or ""
        ).strip(),
        "layerRangeEquivalenceProbeReport": str(
            manifest_metadata.get("layer_range_equivalence_probe_report")
            or manifest_metadata.get("layerRangeEquivalenceProbeReport")
            or ""
        ).strip(),
    }
    return {key: value for key, value in metadata.items() if value not in ("", None)}


def task_level_transport_planned_shard_ranges(
    cai_url: str,
    model_id: str,
    *,
    executor_node_ids: list[str],
    total_layer_count: int,
    resolve_cai_instance_create_payload_for_nodes_func: Callable[..., dict[str, Any]],
    snapshot_from_instance_definition_func: Callable[..., dict[str, Any] | None],
    optional_int_field_value_func: Callable[..., int | None],
    is_private_curated_model_id_func: Callable[[str], bool],
) -> tuple[list[str], list[dict[str, Any]] | None]:
    executors = [
        str(node_id or "").strip()
        for node_id in executor_node_ids
        if str(node_id or "").strip()
    ]
    if len(executors) <= 1:
        return executors, None
    try:
        total_layers = int(total_layer_count)
    except (TypeError, ValueError):
        return executors, None
    if total_layers <= 0:
        return executors, None
    try:
        placement_payload = resolve_cai_instance_create_payload_for_nodes_func(
            cai_url,
            model_id,
            node_ids=executors,
            private_network_model=is_private_curated_model_id_func(model_id),
            cluster_node_count=len(executors),
            prefer_multi_node=True,
        )
    except ValueError:
        return executors, None
    planned_snapshot = snapshot_from_instance_definition_func(
        placement_payload.get("instance")
        if isinstance(placement_payload, dict)
        else None,
        snapshot_source="task_level_placement_preview",
    )
    participants = (
        planned_snapshot.get("participants")
        if isinstance(planned_snapshot, dict)
        else None
    )
    if not isinstance(participants, list) or not participants:
        return executors, None
    planned_ranges: list[dict[str, Any]] = []
    for item in participants:
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("node_id") or "").strip()
        if node_id not in executors:
            continue
        layer_start = optional_int_field_value_func(item, "layer_start", "layerStart")
        layer_end = optional_int_field_value_func(item, "layer_end", "layerEnd")
        if layer_start is None or layer_end is None or layer_end <= layer_start:
            return executors, None
        planned_ranges.append(
            {
                "nodeId": node_id,
                "layerStart": layer_start,
                "layerEnd": layer_end,
                "layerCount": layer_end - layer_start,
            }
        )
    if len(planned_ranges) != len(executors):
        return executors, None
    if {item["nodeId"] for item in planned_ranges} != set(executors):
        return executors, None
    expected_start = 0
    for item in planned_ranges:
        if int(item["layerStart"]) != expected_start:
            return executors, None
        expected_start = int(item["layerEnd"])
    if expected_start != total_layers:
        return executors, None
    planned_executor_ids = [str(item["nodeId"]) for item in planned_ranges]
    return planned_executor_ids, planned_ranges


def task_level_transport_instance_snapshot(
    *,
    instance_id: str,
    execution_model_id: str,
    requester_node_id: str,
    executor_node_ids: list[str],
    proof: dict[str, Any],
    dispatch_result: dict[str, Any],
    snapshot_source: str,
    protocol_version: int,
    participants_from_proof_func: Callable[..., list[dict[str, Any]]],
    participants_from_dag_func: Callable[..., list[dict[str, Any]]],
    total_layer_count_func: Callable[..., int],
    total_layer_count_from_dag_func: Callable[[Any], int | None],
    total_layer_count_from_participants_func: Callable[..., int],
) -> dict[str, Any]:
    participant_range_source = "shard_receipts"
    participants = participants_from_proof_func(
        proof,
        executor_node_ids=executor_node_ids,
    )
    if not participants:
        participant_range_source = "dispatch_dag"
        participants = participants_from_dag_func(
            dispatch_result.get("dag"),
            executor_node_ids=executor_node_ids,
        )
    if not participants:
        participants = []
        total_layer_count = total_layer_count_func(
            execution_model_id,
            executor_count=max(len(executor_node_ids), 1),
        )
        participant_range_source = "synthetic_executor_index_fallback"
        for index, node_id in enumerate(executor_node_ids):
            layer_start = index
            layer_end = index + 1
            if len(executor_node_ids) == 1:
                participant_range_source = "single_executor_transport_fallback"
                layer_start = 0
                layer_end = max(total_layer_count, 1)
            participants.append(
                {
                    "node_id": node_id,
                    "runner_id": f"cai-task-http:{node_id}",
                    "layer_start": layer_start,
                    "layer_end": layer_end,
                    "layer_count": max(layer_end - layer_start, 1),
                }
            )
    dag_total_layer_count = total_layer_count_from_dag_func(
        dispatch_result.get("dag"),
    )
    total_layers = total_layer_count_from_participants_func(
        participants,
        fallback=dag_total_layer_count or max(len(executor_node_ids), 1),
    )
    transport_participants = [
        str(node_id or "").strip()
        for node_id in (
            dispatch_result.get("participantNodeIds")
            if isinstance(dispatch_result.get("participantNodeIds"), list)
            else [requester_node_id, *executor_node_ids]
        )
        if str(node_id or "").strip()
    ]
    return {
        "instance_id": instance_id,
        "snapshot_source": snapshot_source,
        "model_id": execution_model_id,
        "participants": participants,
        "relay_routes_by_node": {},
        "caiOwnedTransportProof": proof,
        "caiOwnedTransportParticipantNodeIds": list(
            dict.fromkeys(transport_participants)
        ),
        "caiOwnedTransportExecutorNodeIds": list(dict.fromkeys(executor_node_ids)),
        "caiOwnedTaskLevelTransport": {
            "schemaVersion": protocol_version,
            "mode": "task_level_http_inference",
            "requesterNodeId": requester_node_id,
            "executorNodeIds": list(executor_node_ids),
            "totalLayerCount": total_layers,
            "participantRangeSource": participant_range_source,
        },
    }


def task_level_transport_participants_from_proof(
    proof: dict[str, Any] | None,
    *,
    executor_node_ids: list[str],
    optional_int_field_value_func: Callable[..., int | None],
    optional_int_value_func: Callable[[Any], int | None],
) -> list[dict[str, Any]]:
    if not isinstance(proof, dict):
        return []
    shard_receipts = proof.get("shardReceipts")
    if not isinstance(shard_receipts, list) or not shard_receipts:
        return []

    participants_by_node: dict[str, dict[str, Any]] = {}
    executor_node_set = {
        str(node_id or "").strip()
        for node_id in executor_node_ids
        if str(node_id or "").strip()
    }
    for receipt in shard_receipts:
        if not isinstance(receipt, dict):
            continue
        if str(receipt.get("status") or "").strip() != "completed":
            continue
        node_id = str(receipt.get("nodeId") or "").strip()
        if not node_id or (executor_node_set and node_id not in executor_node_set):
            continue
        layer_start = optional_int_field_value_func(receipt, "layerStart", "layer_start")
        layer_end = optional_int_field_value_func(receipt, "layerEnd", "layer_end")
        participant = participants_by_node.get(node_id)
        if participant is None:
            participant = {
                "node_id": node_id,
                "runner_id": f"cai-task-http:{node_id}",
                "layer_start": layer_start,
                "layer_end": layer_end,
                "layer_count": 1,
            }
            participants_by_node[node_id] = participant
        else:
            existing_start = optional_int_value_func(participant.get("layer_start"))
            existing_end = optional_int_value_func(participant.get("layer_end"))
            if layer_start is not None and (
                existing_start is None or layer_start < existing_start
            ):
                participant["layer_start"] = layer_start
            if layer_end is not None and (existing_end is None or layer_end > existing_end):
                participant["layer_end"] = layer_end

    ordered: list[dict[str, Any]] = []
    for node_id in executor_node_ids:
        participant = participants_by_node.get(str(node_id or "").strip())
        if participant is not None:
            ordered.append(participant)
    for node_id, participant in participants_by_node.items():
        if node_id not in {
            str(item.get("node_id") or "").strip()
            for item in ordered
            if isinstance(item, dict)
        }:
            ordered.append(participant)

    for participant in ordered:
        layer_start = optional_int_value_func(participant.get("layer_start"))
        layer_end = optional_int_value_func(participant.get("layer_end"))
        if layer_start is not None and layer_end is not None and layer_end > layer_start:
            participant["layer_count"] = layer_end - layer_start
    return ordered


def task_level_transport_participants_from_dag(
    dag: Any,
    *,
    executor_node_ids: list[str],
    optional_int_field_value_func: Callable[..., int | None],
    optional_int_value_func: Callable[[Any], int | None],
) -> list[dict[str, Any]]:
    if not isinstance(dag, dict):
        return []
    stages = dag.get("stages")
    if not isinstance(stages, list) or not stages:
        return []
    participants_by_node: dict[str, dict[str, Any]] = {}
    executor_node_set = {
        str(node_id or "").strip()
        for node_id in executor_node_ids
        if str(node_id or "").strip()
    }
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        node_id = str(
            stage.get("executorNodeId") or stage.get("sinkNodeId") or ""
        ).strip()
        if not node_id or (executor_node_set and node_id not in executor_node_set):
            continue
        layer_start = optional_int_field_value_func(stage, "layerStart", "layer_start")
        layer_end = optional_int_field_value_func(stage, "layerEnd", "layer_end")
        participant = participants_by_node.get(node_id)
        if participant is None:
            participant = {
                "node_id": node_id,
                "runner_id": f"cai-task-http:{node_id}",
                "layer_start": layer_start,
                "layer_end": layer_end,
                "layer_count": 1,
            }
            participants_by_node[node_id] = participant
        else:
            existing_start = optional_int_value_func(participant.get("layer_start"))
            existing_end = optional_int_value_func(participant.get("layer_end"))
            if layer_start is not None and (
                existing_start is None or layer_start < existing_start
            ):
                participant["layer_start"] = layer_start
            if layer_end is not None and (existing_end is None or layer_end > existing_end):
                participant["layer_end"] = layer_end

    ordered: list[dict[str, Any]] = []
    seen_node_ids: set[str] = set()
    for node_id in executor_node_ids:
        participant = participants_by_node.get(str(node_id or "").strip())
        if participant is not None:
            ordered.append(participant)
            seen_node_ids.add(str(node_id or "").strip())
    for node_id, participant in participants_by_node.items():
        if node_id not in seen_node_ids:
            ordered.append(participant)
    for participant in ordered:
        layer_start = optional_int_value_func(participant.get("layer_start"))
        layer_end = optional_int_value_func(participant.get("layer_end"))
        if layer_start is not None and layer_end is not None and layer_end > layer_start:
            participant["layer_count"] = layer_end - layer_start
    return ordered


def task_level_transport_total_layer_count_from_dag(
    dag: Any,
    *,
    optional_int_field_value_func: Callable[..., int | None],
) -> int | None:
    if not isinstance(dag, dict):
        return None
    total_layers = optional_int_field_value_func(
        dag,
        "totalLayerCount",
        "total_layer_count",
        "totalLayers",
        "nLayers",
        "block_count",
    )
    if total_layers is None or total_layers <= 0:
        return None
    return total_layers


def task_level_transport_total_layer_count_from_participants(
    participants: list[dict[str, Any]],
    *,
    fallback: int,
    optional_int_value_func: Callable[[Any], int | None],
) -> int:
    layer_end_values = [
        int(layer_end)
        for layer_end in (
            optional_int_value_func(item.get("layer_end"))
            for item in participants
            if isinstance(item, dict)
        )
        if layer_end is not None and int(layer_end) > 0
    ]
    if layer_end_values:
        return max(layer_end_values)
    return max(int(fallback or 1), 1)


def task_level_transport_response(
    final_output: Any,
    *,
    model_id: str,
    session_id: str,
    proof: dict[str, Any],
    protocol_version: int,
    current_time_func: Callable[[], float],
    final_output_text_func: Callable[[Any], str],
    usage_from_proof_func: Callable[[dict[str, Any]], dict[str, int] | None],
) -> dict[str, Any]:
    text = final_output_text_func(final_output)
    response: dict[str, Any] = {
        "id": f"caiot_{session_id}",
        "object": "chat.completion",
        "created": int(current_time_func()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "caiOwnedTransport": {
            "schemaVersion": protocol_version,
            "mode": "task_level_http_inference",
            "sessionId": session_id,
            "proofVerified": True,
        },
    }
    usage = usage_from_proof_func(proof)
    if usage is not None:
        response["usage"] = dict(usage)
        response["caiOwnedTransport"]["usage"] = usage
    return response


def task_level_transport_final_output_text(
    final_output: Any,
    *,
    base64_decode_func: Callable[[str], bytes],
    log_best_effort_failure_func: Callable[[str, Exception], None],
) -> str:
    if not isinstance(final_output, dict):
        return ""
    payload = final_output.get("payload")
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace").strip()
    payload_base64 = final_output.get("payloadBase64")
    if isinstance(payload_base64, str) and payload_base64.strip():
        try:
            return (
                base64_decode_func(payload_base64)
                .decode("utf-8", errors="replace")
                .strip()
            )
        except Exception as exc:
            log_best_effort_failure_func(
                "task-level transport final output base64 decode",
                exc,
            )
            return ""
    payload_text = final_output.get("payloadText") or final_output.get("text")
    return str(payload_text or "").strip()


def task_level_transport_usage_from_proof(
    proof: dict[str, Any],
    *,
    extract_cai_owned_transport_token_usage_func: Callable[[dict[str, Any]], Any],
) -> dict[str, int] | None:
    audit_payload = {
        "caiOwnedTransportExecuted": True,
        "caiOwnedTransportExecutionProof": proof,
    }
    usage = extract_cai_owned_transport_token_usage_func(audit_payload)
    usage_values = usage.get("usage") if isinstance(usage, dict) else None
    if not isinstance(usage_values, dict):
        return None
    return {
        "prompt_tokens": int(usage_values.get("prompt_tokens") or 0),
        "completion_tokens": int(usage_values.get("completion_tokens") or 0),
        "total_tokens": int(usage_values.get("total_tokens") or 0),
    }


def clean_task_level_peer_cai_urls(peer_cai_urls: Sequence[str]) -> list[str]:
    return list(
        dict.fromkeys(
            str(item or "").strip().rstrip("/")
            for item in peer_cai_urls
            if str(item or "").strip()
        )
    )


def task_level_transport_executor_fallback_attempts(
    executor_node_ids: list[str],
    *,
    requester_node_id: str,
) -> list[list[str]]:
    primary = [
        str(node_id or "").strip()
        for node_id in executor_node_ids
        if str(node_id or "").strip()
    ]
    attempts: list[list[str]] = []

    def add(candidate: list[str]) -> None:
        clean = list(
            dict.fromkeys(
                str(node_id or "").strip()
                for node_id in candidate
                if str(node_id or "").strip()
            )
        )
        if clean and clean not in attempts:
            attempts.append(clean)

    add(primary)
    requester = str(requester_node_id or "").strip()
    if requester and requester in primary and len(primary) > 1:
        add([requester])
    for node_id in primary:
        add([node_id])
    return attempts


def select_task_level_transport_executor_node_ids(
    node_id_attempts: list[list[str] | None],
    *,
    peer_cai_urls_by_node: dict[str, list[str]],
    requester_node_id: str,
    executor_count: int | None = None,
    route_health_records: list[Any] | None = None,
    model_id: str | None = None,
    performance_records: list[Any] | None = None,
    task_level_transport_executor_count_func: Callable[[], int],
    sort_executor_candidates_by_route_health_func: Callable[..., list[str]],
) -> list[str]:
    max_count = max(
        int(
            executor_count
            if executor_count is not None
            else task_level_transport_executor_count_func()
        ),
        1,
    )
    best_partial: list[str] = []
    for attempt in node_id_attempts:
        candidates = [
            str(node_id or "").strip()
            for node_id in (attempt or [])
            if str(node_id or "").strip()
        ]
        routable_candidates = [
            node_id
            for node_id in candidates
            if node_id in peer_cai_urls_by_node
            and node_id != requester_node_id
        ]
        routable_candidates = sort_executor_candidates_by_route_health_func(
            routable_candidates,
            requester_node_id=requester_node_id,
            route_health_records=route_health_records,
            model_id=model_id,
            performance_records=performance_records,
        )
        if len(routable_candidates) >= max_count:
            return routable_candidates[:max_count]
        if len(routable_candidates) > len(best_partial):
            best_partial = routable_candidates
        local_candidates = [
            node_id
            for node_id in candidates
            if node_id in peer_cai_urls_by_node
        ]
        local_candidates = sort_executor_candidates_by_route_health_func(
            local_candidates,
            requester_node_id=requester_node_id,
            route_health_records=route_health_records,
            model_id=model_id,
            performance_records=performance_records,
        )
        if len(local_candidates) >= max_count:
            return local_candidates[:max_count]
        if len(local_candidates) > len(best_partial):
            best_partial = local_candidates
    if best_partial:
        return best_partial[:max_count]
    return []


def sort_executor_candidates_by_route_health(
    candidates: list[str],
    *,
    requester_node_id: str,
    route_health_records: list[Any] | None,
    model_id: str | None = None,
    performance_records: list[Any] | None = None,
    executor_candidate_route_preference_key_func: Callable[..., tuple],
) -> list[str]:
    if (
        not candidates
        or not requester_node_id
        or (not route_health_records and not performance_records)
    ):
        return candidates
    indexed = list(enumerate(candidates))
    indexed.sort(
        key=lambda item: executor_candidate_route_preference_key_func(
            item[1],
            requester_node_id=requester_node_id,
            route_health_records=route_health_records,
            model_id=model_id,
            performance_records=performance_records,
            original_index=item[0],
        ),
        reverse=True,
    )
    return [node_id for _index, node_id in indexed]


def executor_candidate_route_preference_key(
    node_id: str,
    *,
    requester_node_id: str,
    route_health_records: list[Any] | None,
    model_id: str | None,
    performance_records: list[Any] | None,
    original_index: int,
    route_health_score_for_path_func: Callable[..., tuple],
    execution_performance_preference_key_func: Callable[..., tuple],
    latest_known_route_latency_ms_func: Callable[..., float | None],
) -> tuple[int, int, int, int, int, int, float, float, int, int, float, int]:
    score = route_health_score_for_path_func(
        requester_node_id,
        [node_id],
        route_health_records,
    )
    performance_key = execution_performance_preference_key_func(
        model_id=model_id,
        requester_node_id=requester_node_id,
        executor_node_id=node_id,
        performance_records=performance_records,
    )
    latency_ms = latest_known_route_latency_ms_func(
        requester_node_id,
        node_id,
        route_health_records,
    )
    known_latency = 1 if latency_ms is not None else 0
    latency_preference = -(latency_ms if latency_ms is not None else 1_000_000.0)
    return (
        int(score[0]),
        int(score[1]),
        int(score[2]),
        int(performance_key[0]),
        int(performance_key[1]),
        int(performance_key[2]),
        float(performance_key[3]),
        float(performance_key[4]),
        int(performance_key[5]),
        known_latency,
        latency_preference,
        -int(original_index),
    )


def latest_known_route_latency_ms(
    source_node_id: str,
    sink_node_id: str,
    route_health_records: list[Any] | None,
    *,
    route_health_record_field_func: Callable[[Any, str], Any],
) -> float | None:
    source = str(source_node_id or "").strip()
    sink = str(sink_node_id or "").strip()
    if not source or not sink or source == sink or not route_health_records:
        return None

    latest_key: tuple[str, int] | None = None
    latest_latency: float | None = None
    for index, record in enumerate(route_health_records):
        record_source = str(
            route_health_record_field_func(record, "source_node_id") or ""
        ).strip()
        record_sink = str(
            route_health_record_field_func(record, "sink_node_id") or ""
        ).strip()
        if record_source != source or record_sink != sink:
            continue
        if not bool(route_health_record_field_func(record, "reachable")):
            continue
        latency_value = route_health_record_field_func(record, "latency_ms")
        try:
            latency_ms = float(latency_value)
        except (TypeError, ValueError):
            continue
        checked_at = str(route_health_record_field_func(record, "checked_at") or "")
        key = (checked_at, index)
        if latest_key is None or key > latest_key:
            latest_key = key
            latest_latency = latency_ms
    return latest_latency


def estimated_prompt_token_count(prompt: str) -> int:
    return max(1, len(str(prompt or "").split()))


def format_worker_node_rejection_summary(audit: dict[str, Any] | None) -> str:
    if not isinstance(audit, dict):
        return ""
    nodes = audit.get("nodes") if isinstance(audit.get("nodes"), list) else []
    rejected = []
    for node in nodes:
        if not isinstance(node, dict) or node.get("eligible"):
            continue
        reasons = node.get("reasons") if isinstance(node.get("reasons"), list) else []
        if not reasons:
            continue
        rejected.append(f"{node.get('nodeId')}: {', '.join(str(item) for item in reasons)}")
    if not rejected:
        return ""
    preview = "; ".join(rejected[:5])
    if len(rejected) > 5:
        preview = f"{preview}; ... {len(rejected) - 5} more"
    return f" Worker candidate rejection reasons: {preview}."

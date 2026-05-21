# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .cai_owned_transport_common import clean_node_ids as _clean_node_ids
from .cai_owned_transport_protocol import (
    EXECUTION_MODE_CAI_OWNED_TRANSPORT_REQUIRED,
    EXECUTION_MODE_LLAMA_CPP_RPC_LOW_LATENCY,
    EXECUTION_MODE_LLAMA_CPP_RPC_PROVEN_UNKNOWN_LATENCY,
    EXECUTION_MODE_SINGLE_NODE,
)


def execution_mode_for_compute_cell(profile: dict[str, Any]) -> str:
    profile_name = str(profile.get("profile") or "").strip()
    rpc_ready = bool(profile.get("readyForLlamaCppRpc"))
    if profile_name == "single_node":
        return EXECUTION_MODE_SINGLE_NODE
    if profile_name == "low_latency_sharded_cell" and rpc_ready:
        return EXECUTION_MODE_LLAMA_CPP_RPC_LOW_LATENCY
    if profile_name == "proven_unknown_latency_sharded_cell" and rpc_ready:
        return EXECUTION_MODE_LLAMA_CPP_RPC_PROVEN_UNKNOWN_LATENCY
    return EXECUTION_MODE_CAI_OWNED_TRANSPORT_REQUIRED


def execution_reason(execution_mode: str, profile: dict[str, Any]) -> str:
    if execution_mode == EXECUTION_MODE_SINGLE_NODE:
        return "No remote shard participant is required for this execution."
    if execution_mode == EXECUTION_MODE_LLAMA_CPP_RPC_LOW_LATENCY:
        return "Standard llama.cpp RPC is allowed because the compute-cell is low-latency."
    if execution_mode == EXECUTION_MODE_LLAMA_CPP_RPC_PROVEN_UNKNOWN_LATENCY:
        return (
            "Standard llama.cpp RPC is provisionally allowed, but CAI should still "
            "measure latency before production placement."
        )
    profile_reason = str(profile.get("reason") or "").strip()
    if profile_reason:
        return (
            f"{profile_reason} CAI-owned transport is required for WAN-safe "
            "distributed execution."
        )
    return "CAI-owned transport is required for WAN-safe distributed execution."


def clean_sink_node_ids(source_node_id: str, sink_node_ids: Sequence[str]) -> list[str]:
    source = str(source_node_id or "").strip()
    sinks: list[str] = []
    seen: set[str] = set()
    for node_id in sink_node_ids:
        clean = str(node_id or "").strip()
        if not clean or clean == source or clean in seen:
            continue
        seen.add(clean)
        sinks.append(clean)
    return sinks


def cai_owned_transport_layer_ranges(
    executor_node_ids: Sequence[str],
    total_layer_count: int,
) -> list[dict[str, Any]]:
    executors = _clean_node_ids(executor_node_ids)
    total_layers = int(total_layer_count)
    if total_layers <= 0 or not executors:
        return []
    base_layer_count = total_layers // len(executors)
    extra_layers = total_layers % len(executors)
    ranges: list[dict[str, Any]] = []
    layer_start = 0
    for index, node_id in enumerate(executors):
        layer_count = base_layer_count + (1 if index < extra_layers else 0)
        layer_end = layer_start + layer_count
        ranges.append(
            {
                "nodeId": node_id,
                "layerStart": layer_start,
                "layerEnd": layer_end,
                "layerCount": layer_count,
            }
        )
        layer_start = layer_end
    return ranges


def normalize_cai_owned_transport_shard_ranges(
    executor_node_ids: Sequence[str],
    total_layer_count: int,
    *,
    shard_ranges: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    executors = _clean_node_ids(executor_node_ids)
    total_layers = int(total_layer_count)
    if total_layers <= 0 or not executors:
        return []
    if shard_ranges is None:
        return cai_owned_transport_layer_ranges(executors, total_layers)
    if isinstance(shard_ranges, (str, bytes)) or not isinstance(shard_ranges, Sequence):
        raise ValueError("CAI-owned transport execution DAG shard ranges are invalid.")
    if len(shard_ranges) != len(executors):
        raise ValueError("CAI-owned transport execution DAG shard ranges are invalid.")
    normalized: list[dict[str, Any]] = []
    expected_start = 0
    for expected_node_id, item in zip(executors, shard_ranges):
        if not isinstance(item, Mapping):
            raise ValueError(
                "CAI-owned transport execution DAG shard ranges are invalid."
            )
        node_id = str(item.get("nodeId") or item.get("node_id") or "").strip()
        if node_id != expected_node_id:
            raise ValueError(
                "CAI-owned transport execution DAG shard ranges do not match executor order."
            )
        layer_start = _optional_int(
            item.get("layerStart")
            if "layerStart" in item
            else item.get("layer_start")
            if "layer_start" in item
            else item.get("start")
        )
        layer_end = _optional_int(
            item.get("layerEnd")
            if "layerEnd" in item
            else item.get("layer_end")
            if "layer_end" in item
            else item.get("end")
        )
        if layer_start is None or layer_end is None or layer_end <= layer_start:
            raise ValueError(
                "CAI-owned transport execution DAG shard ranges are invalid."
            )
        if layer_start != expected_start:
            raise ValueError(
                "CAI-owned transport execution DAG shard ranges must be contiguous."
            )
        layer_count = _optional_int(
            item.get("layerCount")
            if "layerCount" in item
            else item.get("layer_count")
        )
        computed_layer_count = layer_end - layer_start
        if layer_count is not None and layer_count != computed_layer_count:
            raise ValueError(
                "CAI-owned transport execution DAG shard range layer count is invalid."
            )
        normalized.append(
            {
                "nodeId": node_id,
                "layerStart": int(layer_start),
                "layerEnd": int(layer_end),
                "layerCount": int(computed_layer_count),
            }
        )
        expected_start = layer_end
    if expected_start != total_layers:
        raise ValueError(
            "CAI-owned transport execution DAG shard ranges do not cover total layer count."
        )
    return normalized


def cai_owned_transport_output_route_plan_from_dag(
    dag: Mapping[str, Any],
    *,
    requester_node_id: str,
) -> list[dict[str, Any]]:
    stages = [
        dict(stage)
        for stage in dag.get("stages") or []
        if isinstance(stage, dict)
    ]
    requester = str(requester_node_id or "").strip()
    if not stages or not requester:
        return []
    plan: list[dict[str, Any]] = []
    for stage in stages[1:]:
        plan.append(
            {
                "sinkNodeId": str(stage.get("sinkNodeId") or "").strip(),
                "phase": str(stage.get("phase") or "").strip(),
                "sequence": int(stage.get("sequence") or 0),
                "stageId": stage.get("stageId"),
                "executorNodeId": stage.get("executorNodeId"),
                "layerStart": stage.get("layerStart"),
                "layerEnd": stage.get("layerEnd"),
            }
        )
    final_stage = stages[-1]
    plan.append(
        {
            "sinkNodeId": requester,
            "phase": str(final_stage.get("phase") or "").strip(),
            "sequence": int(final_stage.get("sequence") or 0) + 1,
            "stageId": "final_result",
            "finalOutput": True,
        }
    )
    return [item for item in plan if item["sinkNodeId"] and item["phase"]]


def cai_owned_transport_frame_kind_for_phase(phase: str) -> str:
    if phase == "decode_activation_batches":
        return "decode"
    if phase == "prefill_activation_batches":
        return "activation"
    return "activation"


def cai_owned_transport_template_token_start(phase: str, token_count: int) -> int:
    if phase == "decode_activation_batches":
        return max(0, int(token_count or 0))
    return 0


def cai_owned_transport_template_token_end(phase: str, token_count: int) -> int:
    if phase == "decode_activation_batches":
        return max(0, int(token_count or 0)) + 1
    return max(0, int(token_count or 0))


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

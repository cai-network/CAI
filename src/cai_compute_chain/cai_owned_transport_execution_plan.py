# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .cai_owned_transport_common import clean_node_ids as _clean_node_ids


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


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

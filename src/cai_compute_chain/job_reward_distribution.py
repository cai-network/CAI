# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Any


def unwrap_shard_metadata(shard_payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(shard_payload, dict):
        return {}
    if len(shard_payload) == 1:
        value = next(iter(shard_payload.values()))
        if isinstance(value, dict):
            return value
    return shard_payload


def layer_count_from_metadata(metadata: dict[str, Any]) -> int:
    start = metadata.get("startLayer")
    end = metadata.get("endLayer")
    if isinstance(start, int) and isinstance(end, int) and end > start:
        return max(end - start, 1)
    n_layers = metadata.get("nLayers")
    if isinstance(n_layers, int) and n_layers > 0:
        return n_layers
    return 1


def optional_int_value(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def optional_int_field_value(payload: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key in payload:
            value = optional_int_value(payload.get(key))
            if value is not None:
                return value
    return None


def settlement_participants_for_reward(
    participants: list[dict[str, Any]],
    *,
    network_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    if not bool(network_audit.get("caiOwnedTransportExecuted")):
        return participants

    proof = network_audit.get("caiOwnedTransportExecutionProof")
    if not isinstance(proof, dict):
        return participants

    shard_receipts = proof.get("shardReceipts")
    if not isinstance(shard_receipts, list) or not shard_receipts:
        network_audit["rewardPayoutSource"] = "cai_owned_transport_shard_receipts"
        network_audit["rewardPayoutNodeIds"] = []
        return []

    participant_by_node = {
        str(item.get("node_id") or item.get("nodeId") or "").strip(): item
        for item in participants
        if isinstance(item, dict)
        and str(item.get("node_id") or item.get("nodeId") or "").strip()
    }
    receipt_backed_participants: list[dict[str, Any]] = []
    seen_node_ids: set[str] = set()
    for receipt in shard_receipts:
        if not isinstance(receipt, dict):
            continue
        if str(receipt.get("status") or "").strip() != "completed":
            continue
        node_id = str(receipt.get("nodeId") or "").strip()
        if not node_id or node_id in seen_node_ids:
            continue
        seen_node_ids.add(node_id)

        base = participant_by_node.get(node_id) or {}
        layer_start = optional_int_field_value(receipt, "layerStart", "layer_start")
        layer_end = optional_int_field_value(receipt, "layerEnd", "layer_end")
        if layer_start is None:
            layer_start = optional_int_field_value(base, "layer_start", "layerStart")
        if layer_end is None:
            layer_end = optional_int_field_value(base, "layer_end", "layerEnd")
        layer_count = None
        if layer_start is not None and layer_end is not None and layer_end > layer_start:
            layer_count = layer_end - layer_start
        if layer_count is None:
            layer_count = optional_int_field_value(base, "layer_count", "layerCount")
        if layer_count is None:
            layer_count = max(
                1,
                optional_int_value(receipt.get("activationBatchCount")) or 0,
                optional_int_value(receipt.get("decodeBatchCount")) or 0,
            )

        receipt_backed_participants.append(
            {
                "node_id": node_id,
                "runner_id": (
                    base.get("runner_id")
                    or base.get("runnerId")
                    or f"cai-owned-transport:{node_id}"
                ),
                "layer_start": layer_start,
                "layer_end": layer_end,
                "layer_count": max(int(layer_count), 1),
                "reward_proof_source": "cai_owned_transport_shard_receipt",
            }
        )

    network_audit["rewardPayoutSource"] = "cai_owned_transport_shard_receipts"
    network_audit["rewardPayoutNodeIds"] = [
        item["node_id"] for item in receipt_backed_participants
    ]
    skipped_node_ids = [
        str(item.get("node_id") or item.get("nodeId") or "").strip()
        for item in participants
        if isinstance(item, dict)
        and str(item.get("node_id") or item.get("nodeId") or "").strip()
        and str(item.get("node_id") or item.get("nodeId") or "").strip()
        not in seen_node_ids
    ]
    if skipped_node_ids:
        network_audit["rewardSkippedNodeIdsWithoutShardReceipt"] = sorted(
            dict.fromkeys(skipped_node_ids)
        )
    return receipt_backed_participants


def distribute_worker_reward(
    total_reward_atomic: int, participants: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if total_reward_atomic <= 0 or not participants:
        return []

    total_weight = sum(max(int(item.get("layer_count", 1)), 1) for item in participants)
    if total_weight <= 0:
        total_weight = len(participants)

    raw_parts: list[tuple[int, int, dict[str, Any]]] = []
    allocated = 0
    for item in participants:
        weight = max(int(item.get("layer_count", 1)), 1)
        raw_value = total_reward_atomic * weight
        reward_atomic = raw_value // total_weight
        allocated += reward_atomic
        raw_parts.append((reward_atomic, raw_value % total_weight, item))

    remainder = total_reward_atomic - allocated
    raw_parts.sort(key=lambda entry: (-entry[1], str(entry[2].get("node_id"))))
    for index in range(remainder):
        current = raw_parts[index % len(raw_parts)]
        raw_parts[index % len(raw_parts)] = (current[0] + 1, current[1], current[2])

    results: list[dict[str, Any]] = []
    for reward_atomic, _, item in raw_parts:
        weight = max(int(item.get("layer_count", 1)), 1)
        share_bps = (weight * 10_000) // total_weight
        results.append(
            {
                "node_id": item["node_id"],
                "runner_id": item.get("runner_id"),
                "layer_start": item.get("layer_start"),
                "layer_end": item.get("layer_end"),
                "layer_count": weight,
                "share_bps": share_bps,
                "reward_atomic": reward_atomic,
                "note": "Distributed by pipeline layer share.",
            }
        )
    results.sort(
        key=lambda item: (
            item["layer_start"] is None,
            item["layer_start"] if item["layer_start"] is not None else 10**9,
            item["node_id"],
        )
    )
    return results

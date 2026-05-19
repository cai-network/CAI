# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Any

from .economics import resolve_reserved_output_tokens
from .model import MoneyPolicy
from .wallet import coins_to_atomic


def pricing_floor_atomic(money_policy: MoneyPolicy) -> int:
    return coins_to_atomic(money_policy.automatic_price_floor_coins, money_policy)


def pricing_cap_atomic(money_policy: MoneyPolicy) -> int:
    return coins_to_atomic(money_policy.automatic_price_cap_coins, money_policy)


def extract_reserved_output_tokens(
    request_payload_preview: dict[str, Any] | None,
    money_policy: MoneyPolicy,
) -> int:
    if isinstance(request_payload_preview, dict):
        for key in ("max_tokens", "max_output_tokens"):
            raw_value = request_payload_preview.get(key)
            try:
                if raw_value is not None and int(raw_value) > 0:
                    return resolve_reserved_output_tokens(
                        max_output_tokens=int(raw_value),
                        money_policy=money_policy,
                    )
            except (TypeError, ValueError):
                continue
    return resolve_reserved_output_tokens(
        max_output_tokens=None,
        money_policy=money_policy,
    )


def extract_llm_token_usage(payload: dict[str, Any]) -> dict[str, int] | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt_keys = ("prompt_tokens", "input_tokens")
    completion_keys = ("completion_tokens", "output_tokens")
    prompt_tokens = first_int_value(usage, prompt_keys)
    completion_tokens = first_int_value(usage, completion_keys)
    if prompt_tokens is None and completion_tokens is None:
        return None
    normalized_prompt_tokens = max(prompt_tokens or 0, 0)
    normalized_completion_tokens = max(completion_tokens or 0, 0)
    total_tokens = first_int_value(usage, ("total_tokens",))
    if total_tokens is None:
        total_tokens = normalized_prompt_tokens + normalized_completion_tokens
    return {
        "prompt_tokens": normalized_prompt_tokens,
        "completion_tokens": normalized_completion_tokens,
        "total_tokens": max(total_tokens, 0),
    }


def extract_cai_owned_transport_token_usage(
    network_audit: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(network_audit, dict) or not bool(
        network_audit.get("caiOwnedTransportExecuted")
    ):
        return None
    proof = network_audit.get("caiOwnedTransportExecutionProof")
    if not isinstance(proof, dict):
        return None
    shard_receipts = proof.get("shardReceipts")
    if not isinstance(shard_receipts, list) or not shard_receipts:
        return None

    per_executor: list[dict[str, Any]] = []
    prompt_counts: list[int] = []
    completion_counts: list[int] = []
    input_counts: list[int] = []
    output_counts: list[int] = []
    total_counts: list[int] = []
    aggregate_prompt = 0
    aggregate_completion = 0
    aggregate_input = 0
    aggregate_output = 0
    aggregate_total = 0
    for receipt in shard_receipts:
        if not isinstance(receipt, dict):
            continue
        metrics = receipt.get("metrics")
        if not isinstance(metrics, dict):
            continue
        prompt_count = first_int_value(
            metrics,
            ("promptTokenCount", "promptTokens", "prompt_tokens"),
        )
        completion_count = first_int_value(
            metrics,
            ("completionTokenCount", "completionTokens", "completion_tokens"),
        )
        input_count = first_int_value(
            metrics,
            ("inputTokenCount", "inputTokens", "input_tokens"),
        )
        output_count = first_int_value(
            metrics,
            ("outputTokenCount", "outputTokens", "output_tokens"),
        )
        total_count = first_int_value(
            metrics,
            ("tokenCount", "totalTokenCount", "totalTokens", "total_tokens", "tokens"),
        )
        if (
            prompt_count is None
            and completion_count is None
            and input_count is None
            and output_count is None
            and total_count is None
        ):
            continue
        normalized_prompt = max(0, int(prompt_count or 0))
        normalized_completion = max(0, int(completion_count or 0))
        normalized_input = max(0, int(input_count or 0))
        normalized_output = max(0, int(output_count or 0))
        normalized_total = max(0, int(total_count or 0))
        if (
            normalized_prompt <= 0
            and normalized_completion <= 0
            and normalized_input <= 0
            and normalized_output <= 0
            and normalized_total <= 0
        ):
            continue
        if prompt_count is not None:
            prompt_counts.append(normalized_prompt)
            aggregate_prompt += normalized_prompt
        if completion_count is not None:
            completion_counts.append(normalized_completion)
            aggregate_completion += normalized_completion
        input_counts.append(normalized_input)
        output_counts.append(normalized_output)
        total_counts.append(normalized_total)
        aggregate_input += normalized_input
        aggregate_output += normalized_output
        aggregate_total += normalized_total
        per_executor.append(
            {
                "node_id": str(receipt.get("nodeId") or "").strip() or None,
                "prompt_token_count": (
                    normalized_prompt if prompt_count is not None else None
                ),
                "completion_token_count": (
                    normalized_completion if completion_count is not None else None
                ),
                "input_token_count": normalized_input,
                "output_token_count": normalized_output,
                "token_count": normalized_total,
                "activation_batch_count": max(
                    0,
                    int(receipt.get("activationBatchCount") or 0),
                ),
                "decode_batch_count": max(
                    0,
                    int(receipt.get("decodeBatchCount") or 0),
                ),
                "batch_ids": list(receipt.get("batchIds") or []),
                "stage_ids": list(receipt.get("stageIds") or []),
            }
        )

    if not per_executor:
        return None

    logical_prompt = max(prompt_counts) if prompt_counts else None
    logical_completion = max(completion_counts) if completion_counts else None
    logical_input = max(input_counts) if input_counts else 0
    logical_output = max(output_counts) if output_counts else 0
    logical_total_metric = max(total_counts) if total_counts else 0
    if logical_prompt is not None:
        prompt_tokens = logical_prompt
        completion_tokens = (
            logical_completion if logical_completion is not None else logical_output
        )
        total_tokens = prompt_tokens + completion_tokens
    elif logical_input > 0:
        prompt_tokens = (
            max(0, logical_input - logical_output)
            if logical_output > 0 and logical_input >= logical_output
            else logical_input
        )
        completion_tokens = logical_output
        total_tokens = prompt_tokens + completion_tokens
    elif logical_total_metric > 0:
        completion_tokens = logical_output
        prompt_tokens = max(0, logical_total_metric - completion_tokens)
        total_tokens = prompt_tokens + completion_tokens
    elif logical_output > 0:
        prompt_tokens = 0
        completion_tokens = logical_output
        total_tokens = logical_output
    else:
        return None

    usage = {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
    }
    return {
        "usage": usage,
        "audit": {
            "schema_version": 1,
            "source": "cai_owned_transport_proof",
            "session_id": str(proof.get("sessionId") or "").strip() or None,
            "instance_id": str(proof.get("instanceId") or "").strip() or None,
            "model_id": str(proof.get("modelId") or "").strip() or None,
            "executor_count": len(per_executor),
            "logical_prompt_token_count": (
                int(logical_prompt) if logical_prompt is not None else None
            ),
            "logical_completion_token_count": (
                int(logical_completion) if logical_completion is not None else None
            ),
            "logical_input_token_count": int(logical_input),
            "logical_output_token_count": int(logical_output),
            "logical_total_token_count": int(total_tokens),
            "aggregate_prompt_token_count": int(aggregate_prompt),
            "aggregate_completion_token_count": int(aggregate_completion),
            "aggregate_input_token_count": int(aggregate_input),
            "aggregate_output_token_count": int(aggregate_output),
            "aggregate_total_token_count": int(aggregate_total),
            "pricing_uses_logical_token_stream": True,
            "per_executor": per_executor,
        },
    }


def merge_llm_token_usage_audit(
    *,
    response_usage: dict[str, int] | None,
    proof_usage: dict[str, Any] | None,
) -> tuple[dict[str, int] | None, str | None, dict[str, Any] | None]:
    proof_audit = proof_usage.get("audit") if isinstance(proof_usage, dict) else None
    proof_usage_values = (
        proof_usage.get("usage") if isinstance(proof_usage, dict) else None
    )
    if isinstance(proof_usage_values, dict) and isinstance(proof_audit, dict):
        normalized_proof_usage = {
            "prompt_tokens": int(proof_usage_values.get("prompt_tokens") or 0),
            "completion_tokens": int(
                proof_usage_values.get("completion_tokens") or 0
            ),
            "total_tokens": int(proof_usage_values.get("total_tokens") or 0),
        }
        audit: dict[str, Any] = {
            "schema_version": 1,
            "source": "cai_owned_transport_proof",
            "cai_owned_transport_proof": proof_audit,
            "proof_usage": dict(proof_usage_values),
        }
        if response_usage is not None:
            audit["response_usage"] = dict(response_usage)
            audit["proof_matches_response_usage"] = (
                int(response_usage.get("prompt_tokens") or 0)
                == normalized_proof_usage["prompt_tokens"]
                and int(response_usage.get("completion_tokens") or 0)
                == normalized_proof_usage["completion_tokens"]
                and int(response_usage.get("total_tokens") or 0)
                == normalized_proof_usage["total_tokens"]
            )
        return normalized_proof_usage, "cai_owned_transport_proof", audit
    if response_usage is not None:
        audit: dict[str, Any] = {
            "schema_version": 1,
            "source": "response_usage",
            "response_usage": dict(response_usage),
        }
        return response_usage, "response_usage", audit
    return None, None, None


def first_int_value(payload: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        raw_value = payload.get(key)
        try:
            if raw_value is not None:
                return int(raw_value)
        except (TypeError, ValueError):
            continue
    return None

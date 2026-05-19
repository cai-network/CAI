# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any


_ERROR_RULES: tuple[tuple[str, tuple[str, ...], str, bool], ...] = (
    (
        "hash_mismatch",
        ("hash does not match", "hash mismatch", "payload hash", "output hash"),
        "Reject this batch, keep the payload for audit, then redispatch from the last verified batch.",
        False,
    ),
    (
        "proof_invalid",
        ("proof", "execution audit", "receipt", "participant set", "duplicates batch"),
        "Do not settle rewards; request fresh receipts/proof from executors or rerun verification.",
        False,
    ),
    (
        "timeout",
        ("timeout", "timed out", "expired", "lease expired", "not claimed"),
        "Reconcile session timeouts and retry/reclaim the batch if attempts remain.",
        True,
    ),
    (
        "runtime_busy",
        ("already processing", "runtime busy", "capacity", "max concurrent"),
        "Wait for the current runtime lease to finish or reclaim after the lease timeout.",
        True,
    ),
    (
        "no_route",
        ("no route", "route", "overlay relay is not available", "unreachable"),
        "Refresh route health and node capabilities, then retry direct/relay route selection.",
        True,
    ),
    (
        "no_model_shard",
        ("model shard", "shard is missing", "missing shard", "model is missing"),
        "Sync/download the required model shard on the selected executor before retrying.",
        True,
    ),
    (
        "auth_failed",
        ("auth token", "unauthorized", "not trusted", "signature is invalid", "signature is missing"),
        "Check peer auth/signing configuration and trusted node identity before retrying.",
        False,
    ),
    (
        "wrong_node",
        ("not assigned to local node", "does not include local node", "not a participant"),
        "Route this request to the assigned participant or create a new session for this node.",
        False,
    ),
    (
        "replay_detected",
        ("replay detected", "already used", "replay payload"),
        "Ignore the duplicate payload unless the stored hash differs, then investigate tampering.",
        False,
    ),
    (
        "not_found",
        ("not found", "missing", "is missing"),
        "Refresh local transport state or fetch the missing payload/session from peers.",
        True,
    ),
)


def build_cai_transport_error_detail(
    exc: BaseException | str,
    *,
    operation: str | None = None,
    status_code: int | None = None,
) -> dict[str, Any]:
    message = str(exc)
    code, action, retryable = classify_cai_transport_error(message)
    return {
        "type": "cai_owned_transport_error",
        "code": code,
        "message": message,
        "operation": operation,
        "action": action,
        "retryable": retryable,
        "statusCode": status_code,
    }


def classify_cai_transport_error(message: str) -> tuple[str, str, bool]:
    normalized = str(message or "").strip()
    lowered = normalized.lower()
    for code, patterns, action, retryable in _ERROR_RULES:
        if any(pattern in lowered for pattern in patterns):
            return code, action, retryable
    return (
        "invalid_request",
        "Check the transport request payload and local session state, then retry.",
        False,
    )

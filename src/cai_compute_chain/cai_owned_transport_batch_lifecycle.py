# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

from .cai_owned_transport_common import (
    parse_cai_owned_transport_datetime as _parse_cai_owned_transport_datetime,
)
from .cai_owned_transport_protocol import (
    CAI_OWNED_TRANSPORT_DEFAULT_BATCH_CLAIM_TIMEOUT_SECONDS,
    CAI_OWNED_TRANSPORT_DEFAULT_BATCH_LEASE_SECONDS,
    CAI_OWNED_TRANSPORT_DEFAULT_MAX_BATCH_ATTEMPTS,
)


def apply_cai_owned_transport_batch_lease(
    batch: dict[str, Any],
    now: datetime,
    lease_seconds: float | int | None,
) -> None:
    seconds = coerce_cai_owned_transport_batch_lease_seconds(lease_seconds)
    batch["heartbeatAt"] = now.isoformat()
    batch["leaseSeconds"] = seconds
    batch["leaseExpiresAt"] = (now + timedelta(seconds=seconds)).isoformat()


def cai_owned_transport_batch_lease_expired(
    batch: dict[str, Any],
    now: datetime,
) -> bool:
    lease_expires_at = _parse_cai_owned_transport_datetime(
        batch.get("leaseExpiresAt")
    )
    if lease_expires_at is None:
        return True
    return lease_expires_at <= now


def coerce_cai_owned_transport_batch_lease_seconds(
    value: float | int | None,
) -> float:
    if value is None:
        return CAI_OWNED_TRANSPORT_DEFAULT_BATCH_LEASE_SECONDS
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return CAI_OWNED_TRANSPORT_DEFAULT_BATCH_LEASE_SECONDS


def coerce_cai_owned_transport_batch_claim_timeout_seconds(
    value: float | int | None,
) -> float:
    if value is None:
        return CAI_OWNED_TRANSPORT_DEFAULT_BATCH_CLAIM_TIMEOUT_SECONDS
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return CAI_OWNED_TRANSPORT_DEFAULT_BATCH_CLAIM_TIMEOUT_SECONDS


def coerce_cai_owned_transport_max_attempts(value: int | None) -> int:
    if value is None:
        return CAI_OWNED_TRANSPORT_DEFAULT_MAX_BATCH_ATTEMPTS
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return CAI_OWNED_TRANSPORT_DEFAULT_MAX_BATCH_ATTEMPTS


def cai_owned_transport_batch_attempt_count(batch: Mapping[str, Any]) -> int:
    try:
        return max(0, int(batch.get("attemptCount") or 0))
    except (TypeError, ValueError):
        return 0


def cai_owned_transport_batch_claim_expired(
    batch: Mapping[str, Any],
    now: datetime,
    timeout_seconds: float,
) -> bool:
    reference = (
        _parse_cai_owned_transport_datetime(batch.get("updatedAt"))
        or _parse_cai_owned_transport_datetime(batch.get("createdAt"))
    )
    if reference is None:
        return True
    return reference + timedelta(seconds=timeout_seconds) <= now


def mark_cai_owned_transport_batch_timed_out(
    batch: dict[str, Any],
    now: datetime,
    *,
    error: str,
    reason: str,
) -> None:
    now_iso = now.isoformat()
    batch["status"] = "timed_out"
    batch["updatedAt"] = now_iso
    batch["timedOutAt"] = now_iso
    batch["timeoutReason"] = reason
    batch["lastError"] = error
    batch["error"] = error
    batch["retryable"] = False


def clear_cai_owned_transport_batch_runtime_claim(batch: dict[str, Any]) -> None:
    runtime_id = str(batch.get("runtimeId") or "").strip()
    if runtime_id:
        batch["previousRuntimeId"] = runtime_id
    for key in (
        "runtimeId",
        "heartbeatAt",
        "leaseExpiresAt",
        "leaseSeconds",
        "claimedByNodeId",
    ):
        batch.pop(key, None)

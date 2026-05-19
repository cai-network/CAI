# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any


def execution_attempt_record(
    *,
    attempt: int,
    status: str,
    started_at: str,
    completed_at: str | None,
    participant_node_ids: list[str],
    excluded_node_ids: list[str],
    now_iso_func: Callable[[], str],
    instance_id: str | None = None,
    error: Exception | None = None,
    retry_scheduled: bool = False,
    timeout_sec: int | None = None,
    phase: str | None = None,
    phase_started_at: str | None = None,
    phase_message: str | None = None,
    attempt_duration_ms: int | None = None,
    readiness_duration_ms: int | None = None,
    response_duration_ms: int | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "attempt": attempt,
        "status": status,
        "startedAt": started_at,
        "completedAt": completed_at,
        "participantNodeIds": participant_node_ids,
        "excludedNodeIds": excluded_node_ids,
        "retryScheduled": retry_scheduled,
    }
    if instance_id:
        record["instanceId"] = instance_id
    if phase:
        record["phase"] = phase
        record["phaseStartedAt"] = phase_started_at or now_iso_func()
    if phase_message:
        record["phaseMessage"] = phase_message
    if timeout_sec is not None:
        record["timeoutSec"] = timeout_sec
    if attempt_duration_ms is not None:
        record["attemptDurationMs"] = attempt_duration_ms
    if readiness_duration_ms is not None:
        record["readinessDurationMs"] = readiness_duration_ms
    if response_duration_ms is not None:
        record["responseDurationMs"] = response_duration_ms
    if error is not None:
        record["errorType"] = type(error).__name__
        record["message"] = str(error)
    return record


def update_latest_execution_attempt_phase(
    job: Any,
    execution_attempts: list[dict[str, Any]],
    *,
    phase: str,
    wallet_policy: Any,
    now_iso_func: Callable[[], str],
    update_job_intent_func: Callable[[Any, Any], None],
    phase_message: str | None = None,
    participant_node_ids: list[str] | None = None,
    instance_id: str | None = None,
    timeout_sec: int | None = None,
) -> None:
    if not execution_attempts:
        return
    current = dict(execution_attempts[-1])
    current["phase"] = phase
    current["phaseStartedAt"] = now_iso_func()
    if phase_message:
        current["phaseMessage"] = phase_message
    if participant_node_ids is not None:
        current["participantNodeIds"] = list(participant_node_ids)
    if instance_id:
        current["instanceId"] = instance_id
    if timeout_sec is not None:
        current["timeoutSec"] = timeout_sec
    execution_attempts[-1] = current
    job.execution_attempts = execution_attempts
    update_job_intent_func(job, wallet_policy)


def elapsed_ms(started_monotonic: float | None) -> int | None:
    if started_monotonic is None:
        return None
    return max(0, int(round((time.monotonic() - started_monotonic) * 1000.0)))


def record_execution_attempt_performance_best_effort(
    *,
    model_id: str,
    requester_node_id: str | None,
    executor_node_ids: list[str] | tuple[str, ...] | set[str] | None,
    status: str,
    record_execution_attempt_performance_func: Callable[..., Any],
    log_best_effort_failure_func: Callable[[str, Exception], None],
    attempt_duration_ms: int | None = None,
    readiness_duration_ms: int | None = None,
    response_duration_ms: int | None = None,
    timeout_sec: int | float | None = None,
    error_type: str | None = None,
    wallet_policy: Any = None,
) -> None:
    try:
        record_execution_attempt_performance_func(
            model_id=model_id,
            requester_node_id=requester_node_id,
            executor_node_ids=executor_node_ids,
            status=status,
            attempt_duration_ms=attempt_duration_ms,
            readiness_duration_ms=readiness_duration_ms,
            response_duration_ms=response_duration_ms,
            timeout_sec=timeout_sec,
            error_type=error_type,
            policy=wallet_policy,
        )
    except Exception as exc:
        log_best_effort_failure_func("execution attempt performance record", exc)


def best_effort_attempt_participant_node_ids(
    *,
    instance_snapshot: dict[str, Any] | None,
    cai_url: str,
    model_id: str,
    participant_node_ids_func: Callable[[dict[str, Any] | None], list[str]],
    resolve_cai_instance_snapshot_func: Callable[[str, str], dict[str, Any] | None],
    log_best_effort_failure_func: Callable[[str, Exception], None],
) -> list[str]:
    participant_node_ids = participant_node_ids_func(instance_snapshot)
    if participant_node_ids:
        return participant_node_ids
    try:
        return participant_node_ids_func(
            resolve_cai_instance_snapshot_func(cai_url, model_id)
        )
    except Exception as exc:
        log_best_effort_failure_func("execution participant snapshot fallback", exc)
        return []


def should_retry_job_execution_error(
    exc: Exception,
    *,
    should_retry_cai_startup_error_func: Callable[[Exception], bool],
) -> bool:
    message = str(exc).lower()
    non_retryable_fragments = (
        "active wallet must",
        "wallet balance is insufficient",
        "cannot settle execution reward",
        "worker participant eligibility failed",
        "did not report any worker participants",
        "final token-priced settlement could not be funded",
    )
    if any(fragment in message for fragment in non_retryable_fragments):
        return False
    if should_retry_cai_startup_error_func(exc):
        return True
    retryable_fragments = (
        "timed out",
        "timeout",
        "temporarily unavailable",
        "service unavailable",
        "http error 400",
        "http error 408",
        "http error 409",
        "http error 425",
        "http error 429",
        "http error 500",
        "http error 502",
        "http error 503",
        "http error 504",
        "connection reset",
        "connection refused",
        "connection aborted",
        "remote end closed",
        "incomplete read",
        "closed the response",
        "empty response body",
        "invalid json",
        "no output chunks",
        "runner",
        "runtime",
        "route",
        "lease expired",
        "winerror 10048",
        "winerror 10054",
        "winerror 10060",
        "winerror 10061",
    )
    return any(fragment in message for fragment in retryable_fragments)


def should_retry_cai_startup_error(exc: Exception) -> bool:
    message = str(exc).lower()
    retryable_fragments = (
        "no usable cai placement preview found",
        "timed out waiting for cai instance",
        "no usable cai placement preview found",
        "timed out waiting for cai instance",
        "no cycles found with sufficient memory",
        "winerror 10048",
        "only one usage of each socket address",
        "no output chunks were received from the runner",
        "returned an empty response body",
        "returned invalid json",
        "closed the response before returning a complete json body",
        "out of memory",
        "bad allocation",
        "failed to allocate compute pp buffers",
        "memory allocation failure",
    )
    return any(fragment in message for fragment in retryable_fragments)

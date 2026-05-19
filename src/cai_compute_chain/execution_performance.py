# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .local_json_store import atomic_write_json_array_file, read_json_array_file
from .model import WalletPolicy
from .wallet import data_root


PERFORMANCE_HISTORY_FILE_NAME = "execution-performance.json"


@dataclass
class ExecutionPerformanceRecord:
    record_id: str
    model_id: str
    requester_node_id: str
    executor_node_id: str
    sample_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    timeout_count: int = 0
    attempt_duration_sample_count: int = 0
    readiness_duration_sample_count: int = 0
    response_duration_sample_count: int = 0
    avg_attempt_duration_ms: float | None = None
    avg_readiness_duration_ms: float | None = None
    avg_response_duration_ms: float | None = None
    min_response_duration_ms: int | None = None
    last_attempt_duration_ms: int | None = None
    last_readiness_duration_ms: int | None = None
    last_response_duration_ms: int | None = None
    last_status: str | None = None
    last_error_type: str | None = None
    updated_at: str = ""


def execution_performance_file_path(policy: WalletPolicy | None = None) -> Path:
    return data_root(policy) / PERFORMANCE_HISTORY_FILE_NAME


def list_execution_performance_records(
    policy: WalletPolicy | None = None,
) -> list[ExecutionPerformanceRecord]:
    raw = read_json_array_file(
        execution_performance_file_path(policy),
        heal_corrupt=True,
    )
    allowed_fields = {item.name for item in fields(ExecutionPerformanceRecord)}
    records: list[ExecutionPerformanceRecord] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        payload = {key: value for key, value in item.items() if key in allowed_fields}
        record_id = str(payload.get("record_id") or "").strip()
        model_id = str(payload.get("model_id") or "").strip()
        requester_node_id = str(payload.get("requester_node_id") or "").strip()
        executor_node_id = str(payload.get("executor_node_id") or "").strip()
        if not record_id or not model_id or not requester_node_id or not executor_node_id:
            continue
        payload["record_id"] = record_id
        payload["model_id"] = model_id
        payload["requester_node_id"] = requester_node_id
        payload["executor_node_id"] = executor_node_id
        records.append(ExecutionPerformanceRecord(**payload))
    records.sort(
        key=lambda item: (
            item.model_id,
            item.requester_node_id,
            item.executor_node_id,
        )
    )
    return records


def save_execution_performance_records(
    records: list[ExecutionPerformanceRecord],
    policy: WalletPolicy | None = None,
) -> None:
    atomic_write_json_array_file(
        execution_performance_file_path(policy),
        [asdict(item) for item in records],
    )


def record_execution_attempt_performance(
    *,
    model_id: str,
    requester_node_id: str | None,
    executor_node_ids: list[str] | tuple[str, ...] | set[str] | None,
    status: str,
    attempt_duration_ms: int | None = None,
    readiness_duration_ms: int | None = None,
    response_duration_ms: int | None = None,
    timeout_sec: int | float | None = None,
    error_type: str | None = None,
    policy: WalletPolicy | None = None,
) -> list[ExecutionPerformanceRecord]:
    clean_model_id = str(model_id or "").strip()
    clean_requester = str(requester_node_id or "").strip()
    if not clean_model_id or not clean_requester:
        return []

    executor_ids = _normalized_executor_node_ids(executor_node_ids, clean_requester)
    if not executor_ids:
        return []

    records = list_execution_performance_records(policy)
    by_id = {item.record_id: item for item in records}
    updated_records: list[ExecutionPerformanceRecord] = []
    now = _now_iso()
    for executor_node_id in executor_ids:
        record_id = execution_performance_record_id(
            model_id=clean_model_id,
            requester_node_id=clean_requester,
            executor_node_id=executor_node_id,
        )
        record = by_id.get(record_id)
        if record is None:
            record = ExecutionPerformanceRecord(
                record_id=record_id,
                model_id=clean_model_id,
                requester_node_id=clean_requester,
                executor_node_id=executor_node_id,
            )
            records.append(record)
        _update_performance_record(
            record,
            status=status,
            attempt_duration_ms=attempt_duration_ms,
            readiness_duration_ms=readiness_duration_ms,
            response_duration_ms=response_duration_ms,
            timeout_sec=timeout_sec,
            error_type=error_type,
            updated_at=now,
        )
        updated_records.append(record)

    save_execution_performance_records(records, policy)
    return updated_records


def execution_performance_preference_key(
    *,
    model_id: str | None,
    requester_node_id: str,
    executor_node_id: str,
    performance_records: list[Any] | None,
) -> tuple[int, int, int, float, float, int]:
    record = _find_performance_record(
        model_id=model_id,
        requester_node_id=requester_node_id,
        executor_node_id=executor_node_id,
        performance_records=performance_records,
    )
    if record is None:
        return (0, 5000, 0, -1_000_000.0, -1_000_000.0, 0)

    sample_count = max(1, _int_field(record, "sample_count"))
    success_count = max(0, _int_field(record, "success_count"))
    failure_count = max(0, _int_field(record, "failure_count"))
    timeout_count = max(0, _int_field(record, "timeout_count"))
    success_rate_bps = int(round(success_count * 10_000 / sample_count))
    timeout_rate_bps = int(round(timeout_count * 10_000 / sample_count))
    health = 0
    if success_count > 0 and success_rate_bps >= 5000:
        health = 1
    elif failure_count > 0 and success_rate_bps < 5000:
        health = -1

    avg_response_ms = _float_field(record, "avg_response_duration_ms")
    avg_attempt_ms = _float_field(record, "avg_attempt_duration_ms")
    response_preference = -(
        avg_response_ms if avg_response_ms is not None else 1_000_000.0
    )
    attempt_preference = -(
        avg_attempt_ms if avg_attempt_ms is not None else 1_000_000.0
    )
    return (
        health,
        success_rate_bps,
        -timeout_rate_bps,
        response_preference,
        attempt_preference,
        -failure_count,
    )


def execution_performance_record_id(
    *,
    model_id: str,
    requester_node_id: str,
    executor_node_id: str,
) -> str:
    payload = json.dumps(
        {
            "modelId": str(model_id or "").strip(),
            "requesterNodeId": str(requester_node_id or "").strip(),
            "executorNodeId": str(executor_node_id or "").strip(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _update_performance_record(
    record: ExecutionPerformanceRecord,
    *,
    status: str,
    attempt_duration_ms: int | None,
    readiness_duration_ms: int | None,
    response_duration_ms: int | None,
    timeout_sec: int | float | None,
    error_type: str | None,
    updated_at: str,
) -> None:
    normalized_status = str(status or "").strip().lower()
    normalized_error = str(error_type or "").strip() or None
    record.sample_count = max(0, int(record.sample_count or 0)) + 1
    if normalized_status == "completed":
        record.success_count = max(0, int(record.success_count or 0)) + 1
    else:
        record.failure_count = max(0, int(record.failure_count or 0)) + 1
    if _is_timeout_sample(
        status=normalized_status,
        error_type=normalized_error,
        attempt_duration_ms=attempt_duration_ms,
        timeout_sec=timeout_sec,
    ):
        record.timeout_count = max(0, int(record.timeout_count or 0)) + 1

    (
        record.avg_attempt_duration_ms,
        record.attempt_duration_sample_count,
    ) = _updated_average(
        record.avg_attempt_duration_ms,
        record.attempt_duration_sample_count,
        attempt_duration_ms,
    )
    (
        record.avg_readiness_duration_ms,
        record.readiness_duration_sample_count,
    ) = _updated_average(
        record.avg_readiness_duration_ms,
        record.readiness_duration_sample_count,
        readiness_duration_ms,
    )
    (
        record.avg_response_duration_ms,
        record.response_duration_sample_count,
    ) = _updated_average(
        record.avg_response_duration_ms,
        record.response_duration_sample_count,
        response_duration_ms,
    )
    if response_duration_ms is not None:
        record.min_response_duration_ms = (
            int(response_duration_ms)
            if record.min_response_duration_ms is None
            else min(int(record.min_response_duration_ms), int(response_duration_ms))
        )

    record.last_attempt_duration_ms = attempt_duration_ms
    record.last_readiness_duration_ms = readiness_duration_ms
    record.last_response_duration_ms = response_duration_ms
    record.last_status = normalized_status or None
    record.last_error_type = normalized_error
    record.updated_at = updated_at


def _updated_average(
    current_average: float | None,
    sample_count: int,
    value: int | float | None,
) -> tuple[float | None, int]:
    if value is None:
        return current_average, max(0, int(sample_count or 0))
    count = max(0, int(sample_count or 0))
    numeric_value = float(value)
    if count <= 0 or current_average is None:
        return numeric_value, 1
    return ((float(current_average) * count) + numeric_value) / (count + 1), count + 1


def _normalized_executor_node_ids(
    executor_node_ids: list[str] | tuple[str, ...] | set[str] | None,
    requester_node_id: str,
) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for node_id in executor_node_ids or []:
        normalized = str(node_id or "").strip()
        if not normalized or normalized == requester_node_id or normalized in seen:
            continue
        result.append(normalized)
        seen.add(normalized)
    return result


def _find_performance_record(
    *,
    model_id: str | None,
    requester_node_id: str,
    executor_node_id: str,
    performance_records: list[Any] | None,
) -> Any | None:
    clean_model_id = str(model_id or "").strip()
    clean_requester = str(requester_node_id or "").strip()
    clean_executor = str(executor_node_id or "").strip()
    if not clean_model_id or not clean_requester or not clean_executor:
        return None
    for record in performance_records or []:
        if (
            str(_field(record, "model_id") or "").strip() == clean_model_id
            and str(_field(record, "requester_node_id") or "").strip()
            == clean_requester
            and str(_field(record, "executor_node_id") or "").strip()
            == clean_executor
        ):
            return record
    return None


def _is_timeout_sample(
    *,
    status: str,
    error_type: str | None,
    attempt_duration_ms: int | None,
    timeout_sec: int | float | None,
) -> bool:
    if "timeout" in status or "timed out" in status:
        return True
    if error_type and "timeout" in error_type.lower():
        return True
    if attempt_duration_ms is None or timeout_sec is None:
        return False
    try:
        return float(attempt_duration_ms) >= float(timeout_sec) * 1000.0
    except (TypeError, ValueError):
        return False


def _int_field(record: Any, field_name: str) -> int:
    value = _field(record, field_name)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_field(record: Any, field_name: str) -> float | None:
    value = _field(record, field_name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _field(record: Any, field_name: str) -> Any:
    if isinstance(record, dict):
        return record.get(field_name)
    return getattr(record, field_name, None)


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()

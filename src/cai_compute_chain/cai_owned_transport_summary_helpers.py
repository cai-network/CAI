# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def summary_resource_payload(
    summary: Mapping[str, Any],
    worker: Mapping[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for payload in (summary, worker):
        for field_name in ("resources", "resourceSummary", "resource_summary"):
            value = payload.get(field_name)
            if isinstance(value, Mapping):
                merged.update(value)
        merged.update(
            {
                key: value
                for key, value in payload.items()
                if key
                in {
                    "ramBytes",
                    "ramTotalBytes",
                    "ramAvailableBytes",
                    "availableRamBytes",
                    "ram_bytes",
                    "ram_total_bytes",
                    "ram_available_bytes",
                    "available_ram_bytes",
                    "ramTotal",
                    "ramAvailable",
                    "ram_total",
                    "ram_available",
                    "vramBytes",
                    "vramTotalBytes",
                    "vramAvailableBytes",
                    "availableVramBytes",
                    "vram_bytes",
                    "vram_total_bytes",
                    "vram_available_bytes",
                    "available_vram_bytes",
                    "vramTotal",
                    "vramAvailable",
                    "vram_total",
                    "vram_available",
                    "cpuCores",
                    "cpu_cores",
                }
            }
        )
    return merged


def summary_resource_bytes(
    payload: Mapping[str, Any],
    *field_names: str,
) -> int | None:
    for field_name in field_names:
        if field_name not in payload:
            continue
        parsed = coerce_byte_count(payload.get(field_name))
        if parsed is not None:
            return parsed
    return None


def coerce_byte_count(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in ("inBytes", "in_bytes", "bytes", "value"):
            if key in value:
                return coerce_byte_count(value.get(key))
        return None
    if isinstance(value, bool):
        return None
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def non_negative_int_or_default(value: int | None, default: int) -> int:
    if value is None:
        return max(0, int(default or 0))
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, int(default or 0))


def summary_bool(payload: Mapping[str, Any], *field_names: str) -> bool | None:
    for field_name in field_names:
        if field_name not in payload:
            continue
        value = payload.get(field_name)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on", "enabled", "ready"}:
                return True
            if lowered in {"0", "false", "no", "off", "disabled"}:
                return False
    return None


def summary_string_list(payload: Mapping[str, Any], *field_names: str) -> list[str]:
    for field_name in field_names:
        value = payload.get(field_name)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [
                str(item or "").strip()
                for item in value
                if str(item or "").strip()
            ]
    return []


def summary_payload_model_matches(payload: Mapping[str, Any], model_id: str) -> bool:
    for field_name in ("modelId", "model_id", "id", "name"):
        if summary_model_id_matches(str(payload.get(field_name) or ""), model_id):
            return True
    return False


def summary_model_id_in_list(value: Any, model_id: str) -> bool:
    return any(
        summary_model_id_matches(str(item or ""), model_id)
        for item in summary_string_values(value)
    )


def summary_string_values(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [str(item) for item in value.keys()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    return []


def summary_model_id_matches(value: str, expected: str) -> bool:
    candidate = str(value or "").strip()
    requested = str(expected or "").strip()
    return bool(candidate and requested) and candidate.lower() == requested.lower()

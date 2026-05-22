# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from .cai_owned_transport_common import optional_int as _optional_int


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


def summary_resource_headroom_audit(
    summary: Mapping[str, Any],
    worker: Mapping[str, Any],
    *,
    minimum_ram_headroom_bytes: int,
    minimum_vram_headroom_bytes: int,
) -> dict[str, Any]:
    required_ram = max(0, int(minimum_ram_headroom_bytes or 0))
    required_vram = max(0, int(minimum_vram_headroom_bytes or 0))
    resources = summary_resource_payload(summary, worker)
    ram_available = summary_resource_bytes(
        resources,
        "ramAvailableBytes",
        "availableRamBytes",
        "ram_available_bytes",
        "available_ram_bytes",
        "ramAvailable",
        "ram_available",
    )
    ram_total = summary_resource_bytes(
        resources,
        "ramBytes",
        "ramTotalBytes",
        "ram_total_bytes",
        "ramTotal",
        "ram_total",
    )
    vram_available = summary_resource_bytes(
        resources,
        "vramAvailableBytes",
        "availableVramBytes",
        "vram_available_bytes",
        "available_vram_bytes",
        "vramAvailable",
        "vram_available",
    )
    vram_total = summary_resource_bytes(
        resources,
        "vramBytes",
        "vramTotalBytes",
        "vram_total_bytes",
        "vramTotal",
        "vram_total",
    )
    audit = {
        "ready": True,
        "reason": "resource_headroom_not_required",
        "minimumRamHeadroomBytes": required_ram,
        "minimumVramHeadroomBytes": required_vram,
        "ramAvailableBytes": ram_available,
        "ramBytes": ram_total,
        "vramAvailableBytes": vram_available,
        "vramBytes": vram_total,
    }
    if required_ram <= 0 and required_vram <= 0:
        return audit

    missing: list[str] = []
    insufficient: list[str] = []
    if required_ram > 0:
        if ram_available is not None:
            if ram_available < required_ram:
                insufficient.append("ram")
        elif ram_total is not None and ram_total < required_ram:
            insufficient.append("ram")
        else:
            missing.append("ram")
    if required_vram > 0:
        if vram_available is not None:
            if vram_available < required_vram:
                insufficient.append("vram")
        elif vram_total is not None and vram_total < required_vram:
            insufficient.append("vram")
        else:
            missing.append("vram")

    if insufficient:
        audit.update(
            {
                "ready": False,
                "reason": "resource_headroom_insufficient",
                "error": (
                    "Executor does not have enough "
                    + "/".join(item.upper() for item in insufficient)
                    + " headroom."
                ),
                "insufficientResources": insufficient,
                "missingResources": missing,
            }
        )
        return audit
    if missing:
        audit.update(
            {
                "reason": "resource_headroom_unknown",
                "missingResources": missing,
            }
        )
        return audit
    audit["reason"] = "resource_headroom_ok"
    return audit


def summary_cai_owned_transport_readiness(
    summary: Mapping[str, Any],
    worker: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    candidates = [
        worker.get("caiOwnedTransport"),
        worker.get("cai_owned_transport"),
    ]
    readiness = worker.get("readiness")
    if isinstance(readiness, Mapping):
        candidates.append(readiness.get("caiOwnedTransport"))
        candidates.append(readiness.get("cai_owned_transport"))
    summary_readiness = summary.get("readiness")
    if isinstance(summary_readiness, Mapping):
        candidates.append(summary_readiness.get("caiOwnedTransport"))
        candidates.append(summary_readiness.get("cai_owned_transport"))
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            return candidate
    return None


def summary_available_ranges_from_candidate(
    candidate: Mapping[str, Any],
) -> list[dict[str, int]]:
    ranges: list[dict[str, int]] = []
    ranges.extend(summary_layer_ranges([candidate], assume_ready=False))
    for field_name in (
        "shards",
        "layerRanges",
        "layer_ranges",
        "availableRanges",
        "available_ranges",
        "cachedRanges",
        "cached_ranges",
        "loadedRanges",
        "loaded_ranges",
        "downloadableRanges",
        "downloadable_ranges",
    ):
        value = candidate.get(field_name)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            ranges.extend(summary_layer_ranges(value, assume_ready=False))
    return summary_merge_layer_ranges(ranges)


def summary_layer_ranges(
    values: Sequence[Any],
    *,
    assume_ready: bool,
) -> list[dict[str, int]]:
    ranges: list[dict[str, int]] = []
    for item in values:
        if not isinstance(item, Mapping):
            continue
        if not assume_ready and not summary_layer_range_item_ready(item):
            continue
        start = _optional_int(
            item.get("layerStart")
            if "layerStart" in item
            else item.get("layer_start")
            if "layer_start" in item
            else item.get("start")
        )
        end = _optional_int(
            item.get("layerEnd")
            if "layerEnd" in item
            else item.get("layer_end")
            if "layer_end" in item
            else item.get("end")
        )
        if start is None or end is None or end <= start:
            continue
        ranges.append({"layerStart": start, "layerEnd": end})
    return summary_merge_layer_ranges(ranges)


def summary_layer_range_item_ready(item: Mapping[str, Any]) -> bool:
    if summary_layer_range_item_block_reason(item):
        return False
    ready = summary_bool(
        item,
        "ready",
        "loaded",
        "materialized",
        "downloaded",
        "cached",
        "available",
    )
    if ready is not None:
        return ready
    status = str(item.get("status") or "").strip().lower()
    if status in {"ready", "loaded", "materialized", "inference_ready"}:
        return True
    if status in {"cached", "downloaded", "available"}:
        return summary_cached_shard_integrity_satisfied(item)
    return summary_can_load_before_deadline(item)


def summary_blocked_ranges_from_candidate(
    candidate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    blocked: list[dict[str, Any]] = []
    for item in summary_candidate_range_items(candidate):
        if not isinstance(item, Mapping):
            continue
        reason = summary_layer_range_item_block_reason(item)
        if not reason:
            continue
        ranges = summary_layer_ranges([item], assume_ready=True)
        for layer_range in ranges:
            blocked.append(
                {
                    **layer_range,
                    "reason": reason,
                    "error": summary_layer_range_item_block_error(reason),
                }
            )
    return blocked


def summary_candidate_range_items(
    candidate: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    items: list[Mapping[str, Any]] = [candidate]
    for field_name in (
        "shards",
        "layerRanges",
        "layer_ranges",
        "availableRanges",
        "available_ranges",
        "cachedRanges",
        "cached_ranges",
        "loadedRanges",
        "loaded_ranges",
        "downloadableRanges",
        "downloadable_ranges",
    ):
        value = candidate.get(field_name)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            items.extend(item for item in value if isinstance(item, Mapping))
    return items


def summary_layer_range_item_block_reason(item: Mapping[str, Any]) -> str | None:
    for field_name, reason in (
        ("chunkManifestVerified", "chunk_manifest_not_verified"),
        ("chunk_manifest_verified", "chunk_manifest_not_verified"),
        ("verifiedChunkManifest", "chunk_manifest_not_verified"),
        ("verified_chunk_manifest", "chunk_manifest_not_verified"),
        ("manifestVerified", "chunk_manifest_not_verified"),
        ("manifest_verified", "chunk_manifest_not_verified"),
        ("integrityVerified", "chunk_integrity_not_verified"),
        ("integrity_verified", "chunk_integrity_not_verified"),
        ("hashVerified", "chunk_integrity_not_verified"),
        ("hash_verified", "chunk_integrity_not_verified"),
        ("cacheVerified", "cache_not_verified"),
        ("cache_verified", "cache_not_verified"),
    ):
        if summary_bool(item, field_name) is False:
            return reason

    encrypted = summary_bool(
        item,
        "encryptedAtRest",
        "encrypted_at_rest",
        "encryptedCache",
        "encrypted_cache",
    )
    if encrypted is True and not summary_encrypted_cache_accessible(item):
        return "encrypted_cache_key_missing"

    if summary_deadline_expired(item) and not summary_item_already_materialized(item):
        return "download_deadline_expired"

    return None


def summary_layer_range_item_block_error(reason: str) -> str:
    errors = {
        "chunk_manifest_not_verified": "Executor shard chunk manifest is not verified.",
        "chunk_integrity_not_verified": "Executor shard cache integrity is not verified.",
        "cache_not_verified": "Executor shard cache is not verified.",
        "encrypted_cache_key_missing": (
            "Executor shard cache is encrypted and no usable shard key/materialized "
            "copy is advertised."
        ),
        "download_deadline_expired": "Executor cannot load the shard before the deadline.",
    }
    return errors.get(reason, "Executor model shard is not execution-ready.")


def summary_cached_shard_integrity_satisfied(item: Mapping[str, Any]) -> bool:
    return summary_layer_range_item_block_reason(item) is None


def summary_encrypted_cache_accessible(item: Mapping[str, Any]) -> bool:
    if summary_item_already_materialized(item):
        return True
    accessible = summary_bool(
        item,
        "decryptionKeyAvailable",
        "decryption_key_available",
        "shardKeyAvailable",
        "shard_key_available",
        "materialized",
        "decrypted",
        "loaded",
    )
    return accessible is True


def summary_item_already_materialized(item: Mapping[str, Any]) -> bool:
    materialized = summary_bool(
        item,
        "loaded",
        "materialized",
        "decrypted",
        "inferenceReady",
        "inference_ready",
    )
    if materialized is not None:
        return materialized
    status = str(item.get("status") or "").strip().lower()
    return status in {"ready", "loaded", "materialized", "inference_ready"}


def summary_can_load_before_deadline(item: Mapping[str, Any]) -> bool:
    can_load = summary_bool(
        item,
        "canLoadBeforeDeadline",
        "can_load_before_deadline",
        "canDownloadBeforeDeadline",
        "can_download_before_deadline",
    )
    return can_load is True and not summary_deadline_expired(item)


def summary_deadline_expired(item: Mapping[str, Any]) -> bool:
    for field_name in (
        "downloadDeadlineAt",
        "download_deadline_at",
        "loadDeadlineAt",
        "load_deadline_at",
        "deadlineAt",
        "deadline_at",
    ):
        value = str(item.get(field_name) or "").strip()
        if not value:
            continue
        try:
            normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
            deadline = datetime.fromisoformat(normalized)
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=UTC)
        except ValueError:
            continue
        return datetime.now(tz=UTC) > deadline
    return False


def summary_merge_layer_ranges(
    ranges: Sequence[Mapping[str, int]],
) -> list[dict[str, int]]:
    normalized = sorted(
        (
            {
                "layerStart": int(item["layerStart"]),
                "layerEnd": int(item["layerEnd"]),
            }
            for item in ranges
            if int(item.get("layerEnd", 0)) > int(item.get("layerStart", 0))
        ),
        key=lambda item: (item["layerStart"], item["layerEnd"]),
    )
    merged: list[dict[str, int]] = []
    for item in normalized:
        if not merged or item["layerStart"] > merged[-1]["layerEnd"]:
            merged.append(dict(item))
            continue
        merged[-1]["layerEnd"] = max(merged[-1]["layerEnd"], item["layerEnd"])
    return merged


def summary_missing_required_ranges(
    required_ranges: Sequence[Mapping[str, int]],
    available_ranges: Sequence[Mapping[str, int]],
) -> list[dict[str, int]]:
    missing: list[dict[str, int]] = []
    for required in required_ranges:
        required_start = int(required["layerStart"])
        required_end = int(required["layerEnd"])
        covered = any(
            int(available["layerStart"]) <= required_start
            and int(available["layerEnd"]) >= required_end
            for available in available_ranges
        )
        if not covered:
            missing.append(
                {"layerStart": required_start, "layerEnd": required_end}
            )
    return missing

# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .cai_owned_transport_common import (
    is_safe_transport_file_id as _is_safe_transport_file_id,
    require_safe_transport_file_id as _require_safe_transport_file_id,
)


def cai_owned_transport_proof_batch_ids(
    proof: dict[str, Any] | None,
) -> tuple[set[str], list[str]]:
    if not isinstance(proof, dict):
        return set(), ["CAI-owned transport proof is missing."]
    shard_receipts = proof.get("shardReceipts")
    if not isinstance(shard_receipts, list):
        return set(), ["CAI-owned transport proof shard receipts are missing."]
    batch_ids: set[str] = set()
    errors: list[str] = []
    for receipt in shard_receipts:
        if not isinstance(receipt, dict):
            continue
        raw_batch_ids = receipt.get("batchIds") or []
        if isinstance(raw_batch_ids, (str, bytes)) or not isinstance(
            raw_batch_ids,
            Sequence,
        ):
            errors.append(
                "CAI-owned transport proof shard receipt batch ids are invalid."
            )
            continue
        for raw_batch_id in raw_batch_ids:
            batch_id = str(raw_batch_id or "").strip()
            if not batch_id:
                continue
            if not _is_safe_transport_file_id(batch_id, prefix="caibatch_"):
                errors.append(
                    "CAI-owned transport proof shard receipt batch id is invalid."
                )
                continue
            if batch_id in batch_ids:
                errors.append(
                    f"CAI-owned transport proof duplicates batch id '{batch_id}'."
                )
            batch_ids.add(batch_id)
    return batch_ids, errors


def cai_owned_transport_shard_receipt_batch_ids(
    shard_receipts: Sequence[dict[str, Any]] | None,
) -> tuple[set[str], list[str]]:
    batch_ids: set[str] = set()
    errors: list[str] = []
    for receipt in shard_receipts or []:
        if not isinstance(receipt, dict):
            continue
        for raw_batch_id in receipt.get("batchIds") or []:
            batch_id = str(raw_batch_id or "").strip()
            if not batch_id:
                continue
            if not _is_safe_transport_file_id(batch_id, prefix="caibatch_"):
                errors.append("CAI-owned transport shard receipt batch id is invalid.")
                continue
            batch_ids.add(batch_id)
    return batch_ids, errors


def clean_cai_owned_transport_receipt_batch_ids(
    values: Sequence[str] | None,
) -> list[str]:
    cleaned: list[str] = []
    for value in cai_owned_transport_receipt_values(values):
        batch_id = _require_safe_transport_file_id(value, prefix="caibatch_")
        if batch_id not in cleaned:
            cleaned.append(batch_id)
    return cleaned


def clean_cai_owned_transport_receipt_stage_ids(
    values: Sequence[str] | None,
) -> list[str]:
    cleaned: list[str] = []
    for value in cai_owned_transport_receipt_values(values):
        stage_id = str(value or "").strip()
        if not stage_id:
            continue
        if not _is_safe_transport_file_id(stage_id, prefix="caistage_"):
            raise ValueError("CAI-owned transport shard receipt stage id is invalid.")
        if stage_id not in cleaned:
            cleaned.append(stage_id)
    return cleaned


def clean_cai_owned_transport_receipt_sequences(
    values: Sequence[int] | None,
) -> list[int]:
    cleaned: list[int] = []
    for value in cai_owned_transport_receipt_values(values):
        try:
            sequence = max(0, int(value))
        except (TypeError, ValueError):
            continue
        if sequence not in cleaned:
            cleaned.append(sequence)
    return cleaned


def clean_cai_owned_transport_receipt_hashes(
    values: Sequence[str] | None,
    *,
    field_name: str,
) -> list[str]:
    cleaned: list[str] = []
    for value in cai_owned_transport_receipt_values(values):
        if not str(value or "").strip():
            continue
        normalized = _normalize_sha256_hex(value, field_name=field_name)
        if normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned


def clean_cai_owned_transport_receipt_audits(
    values: Sequence[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    if isinstance(values, dict):
        return [dict(values)]
    for item in cai_owned_transport_receipt_values(values):
        if isinstance(item, dict):
            cleaned.append(dict(item))
    return cleaned


def cai_owned_transport_receipt_values(values: object) -> list[object]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        return [values]
    if isinstance(values, Mapping):
        return []
    try:
        return list(values)  # type: ignore[arg-type]
    except TypeError:
        return []


def max_receipt_count(
    receipts: Sequence[dict[str, Any]],
    field_name: str,
) -> int:
    values: list[int] = []
    for item in receipts:
        if not isinstance(item, dict):
            continue
        try:
            values.append(max(0, int(item.get(field_name) or 0)))
        except (TypeError, ValueError):
            continue
    return max(values) if values else 0


def _normalize_sha256_hex(
    value: object,
    *,
    field_name: str,
) -> str:
    clean = str(value or "").strip().lower()
    if len(clean) != 64 or any(ch not in "0123456789abcdef" for ch in clean):
        raise ValueError(f"CAI-owned transport {field_name} must be sha256 hex.")
    return clean

# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from .model import MoneyPolicy, WalletPolicy


def clean_node_ids(node_ids: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for node_id in node_ids:
        clean = str(node_id or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        cleaned.append(clean)
    return cleaned


def cai_owned_transport_chain_id(
    policy: WalletPolicy | None = None,
    chain_id: str | None = None,
) -> str:
    clean_chain_id = str(chain_id or "").strip().lower()
    if clean_chain_id:
        return clean_chain_id
    if policy is not None:
        return policy.chain_network.value
    return MoneyPolicy().chain_network.value


def parse_cai_owned_transport_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def is_safe_transport_file_id(value: object, *, prefix: str) -> bool:
    clean = str(value or "").strip()
    return (
        bool(clean)
        and clean.startswith(prefix)
        and clean.isascii()
        and all(ch.isalnum() or ch in {"_", "-"} for ch in clean)
    )


def require_safe_transport_file_id(value: object, *, prefix: str) -> str:
    clean = str(value or "").strip()
    if not is_safe_transport_file_id(clean, prefix=prefix):
        raise ValueError("CAI-owned transport storage id is invalid.")
    return clean


def cai_owned_transport_payload_chain_id(payload: dict[str, Any]) -> str | None:
    chain_id = str(
        payload.get("chainId") or payload.get("chain_id") or ""
    ).strip().lower()
    network = str(payload.get("network") or "").strip().lower()
    return chain_id or network or None


def validate_cai_owned_transport_chain_id(
    payload: dict[str, Any],
    *,
    expected_chain_id: str,
    payload_name: str,
) -> tuple[bool, str | None, str | None]:
    chain_id = str(
        payload.get("chainId") or payload.get("chain_id") or ""
    ).strip().lower()
    network = str(payload.get("network") or "").strip().lower()
    if chain_id and network and chain_id != network:
        return (
            False,
            (
                f"CAI-owned transport {payload_name} has mismatched chain id "
                f"'{chain_id}' and network '{network}'."
            ),
            None,
        )
    incoming = chain_id or network
    if not incoming:
        return (
            False,
            f"CAI-owned transport {payload_name} chain id is missing.",
            None,
        )
    expected = str(expected_chain_id or "").strip().lower()
    if expected and incoming != expected:
        return (
            False,
            (
                f"CAI-owned transport {payload_name} is for chain '{incoming}', "
                f"expected '{expected}'."
            ),
            incoming,
        )
    return True, None, incoming


def jsonable_dict(value: dict[str, Any], *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"CAI-owned transport {field_name} must be an object.")
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"CAI-owned transport {field_name} must be JSON serializable."
        ) from exc


def normalize_sha256_hex(
    value: object,
    *,
    field_name: str,
) -> str:
    clean = str(value or "").strip().lower()
    if len(clean) != 64 or any(ch not in "0123456789abcdef" for ch in clean):
        raise ValueError(f"CAI-owned transport {field_name} must be sha256 hex.")
    return clean


def optional_sha256_hex(
    value: object,
    *,
    field_name: str,
) -> str | None:
    if value is None or str(value or "").strip() == "":
        return None
    return normalize_sha256_hex(value, field_name=field_name)


def optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

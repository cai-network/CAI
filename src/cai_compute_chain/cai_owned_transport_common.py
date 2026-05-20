# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

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

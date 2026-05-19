# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SENSITIVE_AUDIT_KEY_PARTS = (
    "authorization",
    "cookie",
    "password",
    "seed",
    "secret",
    "token",
)


def safe_audit_event(
    event: str,
    *,
    method: str,
    path: str,
    client_host: str,
    status: str,
    detail: str | None = None,
) -> dict[str, str]:
    payload = {
        "event": _safe_text(event),
        "method": _safe_text(method).upper(),
        "path": _safe_path(path),
        "client": _safe_text(client_host),
        "status": _safe_text(status),
    }
    if detail:
        payload["detail"] = _safe_text(detail)
    return payload


def redact_sensitive_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in payload.items():
        normalized_key = str(key).strip().lower()
        if any(part in normalized_key for part in SENSITIVE_AUDIT_KEY_PARTS):
            redacted[str(key)] = "<redacted>"
        elif isinstance(value, Mapping):
            redacted[str(key)] = redact_sensitive_mapping(value)
        else:
            redacted[str(key)] = value
    return redacted


def _safe_path(path: str) -> str:
    return _safe_text(path).split("?", 1)[0]


def _safe_text(value: object) -> str:
    return str(value or "").replace("\r", "").replace("\n", "")[:256]

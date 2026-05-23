# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import json
import os
from typing import NoReturn
from urllib.error import HTTPError
from urllib.request import urlopen

from fastapi import HTTPException

from cai.api.cai_transport_errors import build_cai_transport_error_detail


def load_json_url(url: str, *, timeout: int = 5) -> dict[str, object]:
    with urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def http_error_detail(exc: HTTPError) -> str | None:
    try:
        raw = exc.read()
    except Exception:
        return None
    if not raw:
        return None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return raw.decode("utf-8", errors="replace").strip() or None
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if detail is not None:
            return str(detail)
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message") is not None:
            return str(error["message"])
    return str(payload)


def api_command_send_timeout_seconds() -> float:
    raw = os.getenv("CAI_API_COMMAND_SEND_TIMEOUT_SECONDS", "30")
    try:
        timeout = float(str(raw).strip() or "30")
    except ValueError:
        timeout = 30.0
    return max(0.1, timeout)


def raise_cai_transport_http_error(
    exc: BaseException,
    *,
    status_code: int = 400,
    operation: str | None = None,
) -> NoReturn:
    raise HTTPException(
        status_code=status_code,
        detail=build_cai_transport_error_detail(
            exc,
            operation=operation,
            status_code=status_code,
        ),
    ) from exc


def execution_cai_base_url(local_port: int) -> str:
    configured = str(os.getenv("CAI_EXECUTION_CAI_URL") or "").strip().rstrip("/")
    if configured:
        return configured
    return f"http://127.0.0.1:{local_port}"

# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def decode_json_http_payload(content: bytes, *, context: str) -> dict[str, Any]:
    if not content or not content.strip():
        raise RuntimeError(f"{context} returned an empty response body.")
    try:
        payload = json.loads(content.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{context} returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} returned a non-object JSON payload.")
    return payload


def get_json(url: str, *, timeout: int = 30) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:
        return decode_json_http_payload(response.read(), context=f"HTTP GET {url}")


def post_json(url: str, payload: dict[str, Any], *, timeout: int) -> dict[str, Any]:
    request = Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return decode_json_http_payload(response.read(), context=f"HTTP POST {url}")


def delete_json(url: str, *, timeout: int) -> dict[str, Any]:
    request = Request(url=url, headers={"Content-Type": "application/json"}, method="DELETE")
    with urlopen(request, timeout=timeout) as response:
        content = response.read()
        if not content:
            return {}
        return decode_json_http_payload(content, context=f"HTTP DELETE {url}")


def extract_output_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts).strip()
    if content is None:
        return ""
    return str(content).strip()


def extract_finish_reason(payload: dict[str, Any]) -> str | None:
    choices = payload.get("choices") or []
    if not choices:
        return None
    finish_reason = choices[0].get("finish_reason")
    return str(finish_reason) if finish_reason is not None else None


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
        if isinstance(error, dict):
            message = error.get("message")
            if message is not None:
                return str(message)
    return str(payload)

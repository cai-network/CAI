# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def build_text_job_request_payload(
    model_id: str,
    prompt: str,
    *,
    request_payload_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_payload = dict(request_payload_override or {})
    if not request_payload:
        request_payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            # Network jobs should default to concise non-thinking execution unless
            # a higher-level product surface explicitly asks for reasoning mode.
            "enable_thinking": False,
            "reasoning_effort": "none",
        }
    request_payload["model"] = model_id
    request_payload["stream"] = False
    return request_payload


def task_level_transport_initial_prompt_text(
    request_payload: Mapping[str, Any],
    *,
    fallback_prompt: str,
) -> str:
    messages = request_payload.get("messages")
    user_text = latest_user_message_text(messages)
    if user_text:
        return user_text
    prompt_text = request_payload_prompt_text(request_payload)
    if prompt_text:
        return prompt_text
    return str(fallback_prompt or "")


def latest_user_message_text(messages: Any) -> str:
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return ""
    for item in reversed(list(messages)):
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role and role != "user":
            continue
        text = message_content_text(item.get("content")).strip()
        if text:
            return text
    return ""


def message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts: list[str] = []
        for segment in content:
            if isinstance(segment, Mapping):
                text = segment.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
            elif isinstance(segment, str) and segment.strip():
                parts.append(segment)
        return "\n".join(parts)
    return ""


def request_payload_prompt_text(request_payload: Mapping[str, Any]) -> str:
    for field_name in ("prompt", "input", "text", "content"):
        value = request_payload.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""

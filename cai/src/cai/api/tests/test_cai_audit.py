# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from cai.api.audit import redact_sensitive_mapping, safe_audit_event


def test_safe_audit_event_drops_query_and_newlines() -> None:
    event = safe_audit_event(
        "bearer_auth_failed\n",
        method="post",
        path="/v1/cai/chat/completions?api_key=secret",
        client_host="198.51.100.10\r\nspoof",
        status="invalid",
    )

    assert event == {
        "event": "bearer_auth_failed",
        "method": "POST",
        "path": "/v1/cai/chat/completions",
        "client": "198.51.100.10spoof",
        "status": "invalid",
    }


def test_redact_sensitive_mapping_redacts_nested_secret_fields() -> None:
    redacted = redact_sensitive_mapping(
        {
            "authorization": "Bearer secret",
            "wallet": {
                "password": "pw",
                "seed_phrase": "seed words",
                "address": "abcd",
            },
            "safe": "value",
        }
    )

    assert redacted == {
        "authorization": "<redacted>",
        "wallet": {
            "password": "<redacted>",
            "seed_phrase": "<redacted>",
            "address": "abcd",
        },
        "safe": "value",
    }

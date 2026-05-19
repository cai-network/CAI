# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from cai.api.cai_transport_errors import build_cai_transport_error_detail


def test_cai_transport_error_detail_classifies_actionable_failures() -> None:
    cases = {
        "CAI-owned transport payload hash does not match.": (
            "hash_mismatch",
            False,
        ),
        "CAI-owned transport proof does not cover every executor.": (
            "proof_invalid",
            False,
        ),
        "CAI-owned transport batch lease expired; retry scheduled.": (
            "timeout",
            True,
        ),
        "CAI-owned transport batch is already processing.": (
            "runtime_busy",
            True,
        ),
        "CAI-owned transport overlay relay is not available.": (
            "no_route",
            True,
        ),
        "Required model shard is missing on executor.": (
            "no_model_shard",
            True,
        ),
        "CAI-owned transport batch envelope payload signature is invalid": (
            "auth_failed",
            False,
        ),
        "CAI-owned transport batch is not assigned to local node.": (
            "wrong_node",
            False,
        ),
        "CAI-owned transport batch envelope payload signature replay detected": (
            "replay_detected",
            False,
        ),
    }

    for message, (code, retryable) in cases.items():
        detail = build_cai_transport_error_detail(
            ValueError(message),
            operation="test_operation",
            status_code=400,
        )
        assert detail["type"] == "cai_owned_transport_error"
        assert detail["code"] == code
        assert detail["message"] == message
        assert detail["operation"] == "test_operation"
        assert detail["retryable"] is retryable
        assert detail["statusCode"] == 400
        assert detail["action"]


def test_cai_transport_error_detail_falls_back_to_invalid_request() -> None:
    detail = build_cai_transport_error_detail(
        ValueError("CAI-owned transport batch payload must be an object."),
        operation="record_transport_batch",
        status_code=400,
    )

    assert detail["code"] == "invalid_request"
    assert detail["operation"] == "record_transport_batch"
    assert detail["retryable"] is False

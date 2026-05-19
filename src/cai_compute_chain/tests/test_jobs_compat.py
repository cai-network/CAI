# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from cai_compute_chain.jobs import (
    _normalize_execution_receipt_payload,
    _normalize_job_intent_payload,
)


def test_normalize_job_intent_payload_maps_and_drops_legacy_cai_keys() -> None:
    payload = _normalize_job_intent_payload(
        {
            "job_id": "job-1",
            "CAI_url": "http://127.0.0.1:52415",
            "execution_cai_url": "http://127.0.0.1:52416",
        }
    )

    assert payload["cai_url"] == "http://127.0.0.1:52415"
    assert payload["execution_cai_url"] == "http://127.0.0.1:52416"
    assert payload["requester_node_id"] is None
    assert "CAI_url" not in payload


def test_normalize_execution_receipt_payload_maps_and_drops_legacy_cai_keys() -> None:
    payload = _normalize_execution_receipt_payload(
        {
            "receipt_id": "receipt-1",
            "CAI_url": "http://127.0.0.1:52415",
            "execution_cai_url": "http://127.0.0.1:52416",
        }
    )

    assert payload["cai_url"] == "http://127.0.0.1:52415"
    assert "CAI_url" not in payload
    assert "execution_cai_url" not in payload

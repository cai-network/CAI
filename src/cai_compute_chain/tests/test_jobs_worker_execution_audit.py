# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from unittest.mock import patch

from cai_compute_chain.jobs import _resolve_worker_execution_node_audit
from cai_compute_chain.node_capabilities import NodeCapabilityRecord


def test_worker_execution_audit_counts_verified_capability_node_missing_from_state() -> None:
    record = NodeCapabilityRecord(
        node_id="node-worker",
        source="peer",
        source_url="http://198.51.100.20:52415/v1/cai/node-capabilities",
        last_seen_at="2999-01-01T00:00:00+00:00",
        updated_at="2999-01-01T00:00:00+00:00",
        worker_enabled=True,
        worker_reward_address="worker-address",
        worker_verified=True,
        worker_verification_reason="signed worker capability",
    )

    with (
        patch(
            "cai_compute_chain.jobs._get_json",
            return_value={
                "node-validator": {
                    "workerEnabled": False,
                    "workerRewardAddress": None,
                }
            },
        ),
        patch(
            "cai_compute_chain.jobs.worker_capability_verification_required",
            return_value=True,
        ),
        patch(
            "cai_compute_chain.jobs.list_verified_worker_node_ids",
            return_value={"node-worker"},
        ),
        patch(
            "cai_compute_chain.jobs.list_node_capabilities",
            return_value=[record],
        ),
    ):
        audit = _resolve_worker_execution_node_audit(
            "http://127.0.0.1:52415",
            "cai-network/Qwen3-0.6B-GGUF",
        )

    assert audit is not None
    assert audit["eligibleNodeIds"] == ["node-worker"]
    worker_item = next(
        item for item in audit["nodes"] if item["nodeId"] == "node-worker"
    )
    assert worker_item["capabilityBacked"] is True
    assert worker_item["identityKnown"] is False
    assert worker_item["verifiedCapability"] is True
    assert worker_item["warnings"] == [
        "node identity is missing from live state; using validator-attested capability record"
    ]

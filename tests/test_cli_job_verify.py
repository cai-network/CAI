# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.cli import handle_job_verify  # noqa: E402
from cai_compute_chain.jobs import create_job_intent, execute_job_intent  # noqa: E402
from cai_compute_chain.model import PaymentPreference  # noqa: E402
from cai_compute_chain.route_health import RouteHealthRecord  # noqa: E402
from cai_compute_chain.wallet import coins_to_atomic, create_wallet, credit_wallet, unlock_wallet  # noqa: E402


class CliJobVerifyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo_patch = patch(
            "cai_compute_chain.wallet.repo_root",
            return_value=Path(self.tempdir.name),
        )
        self.repo_patch.start()

    def tearDown(self) -> None:
        self.repo_patch.stop()
        self.tempdir.cleanup()

    def _distributed_two_worker_state(self) -> dict[str, object]:
        return {
            "nodeIdentities": {
                "node-a": {"workerEnabled": True, "relayEnabled": False},
                "node-b": {"workerEnabled": True, "relayEnabled": False},
                "node-relay": {"workerEnabled": False, "relayEnabled": True},
            },
            "topology": {
                "connections": {
                    "node-a": {
                        "node-b": [
                            {
                                "sinkMultiaddr": {
                                    "host": "10.0.0.11",
                                    "port": 6001,
                                }
                            }
                        ]
                    },
                    "node-b": {
                        "node-a": [
                            {
                                "sinkMultiaddr": {
                                    "host": "10.0.0.10",
                                    "port": 6000,
                                }
                            }
                        ]
                    },
                }
            },
            "overlayPeers": {
                "node-a": ["node-b"],
                "node-b": ["node-a"],
            },
        }

    def test_handle_job_verify_reports_network_and_reward_consistency(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))
        job = create_job_intent(
            prompt="network audit",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
        )
        instance_snapshot = {
            "instance_id": "instance-audit",
            "participants": [
                {
                    "node_id": "node-a",
                    "runner_id": "runner-a",
                    "layer_start": 0,
                    "layer_end": 14,
                    "layer_count": 14,
                },
                {
                    "node_id": "node-b",
                    "runner_id": "runner-b",
                    "layer_start": 14,
                    "layer_end": 28,
                    "layer_count": 14,
                },
            ],
        }
        response_payload = {
            "id": "chatcmpl-audit",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "verified"},
                }
            ],
            "usage": {
                "prompt_tokens": 14,
                "completion_tokens": 9,
                "total_tokens": 23,
            },
        }

        with (
            patch(
                "cai_compute_chain.jobs.ensure_cai_instance",
                return_value=instance_snapshot,
            ),
            patch(
                "cai_compute_chain.jobs.resolve_cai_instance_snapshot",
                return_value=instance_snapshot,
            ),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=self._distributed_two_worker_state(),
            ),
            patch(
                "cai_compute_chain.jobs.list_route_health_records",
                return_value=[
                    RouteHealthRecord(
                        route_id="rpc-low-latency",
                        source_node_id="node-a",
                        sink_node_id="node-b",
                        route_type="llama_cpp_rpc_direct",
                        endpoint_url="llama-cpp-rpc://10.0.0.11:6001",
                        reachable=True,
                        checked_at="2026-05-03T00:00:00+00:00",
                        latency_ms=7.0,
                    )
                ],
            ),
            patch(
                "cai_compute_chain.jobs._submit_text_job_to_cai",
                return_value=response_payload,
            ),
            patch(
                "cai_compute_chain.jobs.apply_local_validator_attestation",
                return_value=None,
            ),
            patch(
                "cai_compute_chain.jobs.request_remote_committee_attestations",
                return_value=[],
            ),
        ):
            _, receipt = execute_job_intent(job.job_id)

        report = handle_job_verify(receipt_id=receipt.receipt_id)

        self.assertIn("Execution verification:", report)
        self.assertIn(f"- receipt_id={receipt.receipt_id}", report)
        self.assertIn("- pricing_basis=manual", report)
        self.assertIn("- prompt_tokens=14", report)
        self.assertIn("- completion_tokens=9", report)
        self.assertIn("- total_tokens=23", report)
        self.assertIn("- usage_priced=no", report)
        self.assertIn("- response_received=yes", report)
        self.assertIn("- transport_mode=multi_worker_direct", report)
        self.assertIn("- participant_eligibility_can_settle=yes", report)
        self.assertIn("- participant_eligibility_route_reachable=True", report)
        self.assertIn("- participant_eligibility_fatal_reasons=<none>", report)
        self.assertIn("- decentralized_execution=yes", report)
        self.assertIn("- llama_cpp_execution_mode=llama_cpp_rpc_low_latency", report)
        self.assertIn("- cai_owned_transport_executed=no", report)
        self.assertIn("- cai_owned_transport_proof_error=<none>", report)
        self.assertIn("- direct_paths_checked=node-a<->node-b", report)
        self.assertIn("- overlay_paths_checked=node-a~node-b", report)
        self.assertIn("- relay_transit_candidates=node-relay", report)
        self.assertIn("- reward_sum_receipt=0.98000000", report)
        self.assertIn("- reward_sum_payout_records=0.98000000", report)
        self.assertIn("- reward_accounting_consistent=yes", report)
        self.assertIn("- settlement_found=yes", report)
        self.assertIn("- settlement_status=pending", report)

    def test_handle_job_verify_reports_empty_state_when_no_receipts_exist(self) -> None:
        self.assertEqual(handle_job_verify(), "Execution verification:\n- <empty>")


if __name__ == "__main__":
    unittest.main()

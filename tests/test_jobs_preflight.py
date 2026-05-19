# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
import builtins
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cai_compute_chain.jobs import (
    _augment_route_health_records_from_worker_attestations,
    _best_effort_attempt_participant_node_ids,
    _execution_compute_cell_strategy,
    _pending_model_download_node_labels,
    _record_execution_attempt_performance_best_effort,
    _resolve_worker_execution_node_audit,
    _run_chain_push_audit,
    _run_preflight_peer_sync,
    _route_health_endpoints_from_worker_attestations,
    _sync_task_level_transport_node_capabilities_best_effort,
    _task_level_transport_final_output_text,
    _wait_for_cai_instance_ready,
    cleanup_cai_model_instances,
    cleanup_orphan_llama_cpp_processes,
)


class PreflightPeerSyncTests(unittest.TestCase):
    def test_preflight_peer_sync_records_partial_failure(self) -> None:
        with (
            patch(
                "cai_compute_chain.jobs.sync_validator_set_from_cai_peers",
                side_effect=RuntimeError("validator sync unavailable"),
            ),
            patch(
                "cai_compute_chain.jobs.sync_chain_from_cai_peers",
                return_value=None,
            ) as sync_chain,
        ):
            audit = _run_preflight_peer_sync(
                state_payload={"nodeIdentities": {}},
                cai_url="http://127.0.0.1:52415",
                wallet_policy=None,
            )

        self.assertTrue(audit["attempted"])
        self.assertTrue(audit["statePayloadAvailable"])
        self.assertEqual(audit["validatorSet"]["status"], "failed")
        self.assertEqual(audit["validatorSet"]["errorType"], "RuntimeError")
        self.assertIn("validator sync unavailable", audit["validatorSet"]["message"])
        self.assertEqual(audit["chain"]["status"], "ok")
        self.assertEqual(audit["chain"]["attemptedPeers"], 0)
        self.assertEqual(audit["chain"]["failedPeerUrls"], [])
        sync_chain.assert_called_once()

    def test_preflight_peer_sync_records_validator_peer_counts(self) -> None:
        validator_result = SimpleNamespace(
            attempted_peers=2,
            successful_peers=1,
            failed_peers=1,
            imported_records=3,
            peer_urls=[
                "http://node-a/v1/cai/validators",
                "http://node-b/v1/cai/validators",
            ],
            failed_peer_urls=["http://node-a/v1/cai/validators"],
            peer_errors=[
                {
                    "peerUrl": "http://node-a/v1/cai/validators",
                    "errorType": "OSError",
                    "message": "validator peer offline",
                }
            ],
        )
        chain_result = SimpleNamespace(
            attempted_peers=0,
            successful_peers=0,
            failed_peers=0,
            imported_blocks=0,
            imported_transactions=0,
            peer_urls=[],
            failed_peer_urls=[],
            peer_errors=[],
        )
        with (
            patch(
                "cai_compute_chain.jobs.sync_validator_set_from_cai_peers",
                return_value=validator_result,
            ),
            patch(
                "cai_compute_chain.jobs.sync_chain_from_cai_peers",
                return_value=chain_result,
            ),
        ):
            audit = _run_preflight_peer_sync(
                state_payload={"nodeIdentities": {}},
                cai_url="http://127.0.0.1:52415",
                wallet_policy=None,
            )

        self.assertEqual(audit["validatorSet"]["status"], "ok")
        self.assertEqual(audit["validatorSet"]["attemptedPeers"], 2)
        self.assertEqual(audit["validatorSet"]["successfulPeers"], 1)
        self.assertEqual(audit["validatorSet"]["failedPeers"], 1)
        self.assertEqual(audit["validatorSet"]["importedRecords"], 3)
        self.assertEqual(
            audit["validatorSet"]["failedPeerUrls"],
            validator_result.failed_peer_urls,
        )
        self.assertEqual(
            audit["validatorSet"]["peerErrors"],
            validator_result.peer_errors,
        )

    def test_preflight_peer_sync_marks_missing_state_as_skipped(self) -> None:
        with (
            patch("cai_compute_chain.jobs.sync_validator_set_from_cai_peers") as sync_validators,
            patch("cai_compute_chain.jobs.sync_chain_from_cai_peers") as sync_chain,
        ):
            audit = _run_preflight_peer_sync(
                state_payload=None,
                cai_url="http://127.0.0.1:52415",
                wallet_policy=None,
            )

        self.assertFalse(audit["attempted"])
        self.assertFalse(audit["statePayloadAvailable"])
        self.assertEqual(audit["validatorSet"]["status"], "skipped")
        self.assertEqual(audit["chain"]["status"], "skipped")
        sync_validators.assert_not_called()
        sync_chain.assert_not_called()

    def test_chain_push_audit_records_peer_counts(self) -> None:
        result = SimpleNamespace(
            attempted_peers=2,
            successful_peers=1,
            peer_urls=[
                "http://node-a/v1/cai/chain/sync",
                "http://node-b/v1/cai/chain/sync",
            ],
        )
        with patch(
            "cai_compute_chain.jobs.push_chain_to_cai_peers",
            return_value=result,
        ) as push_chain:
            audit = _run_chain_push_audit(
                state_payload={"nodeIdentities": {}},
                cai_url="http://127.0.0.1:52415",
                wallet_policy=None,
            )

        self.assertTrue(audit["attempted"])
        self.assertEqual(audit["status"], "ok")
        self.assertEqual(audit["attemptedPeers"], 2)
        self.assertEqual(audit["successfulPeers"], 1)
        self.assertEqual(audit["failedPeers"], 1)
        self.assertEqual(audit["failedPeerUrls"], [])
        self.assertEqual(audit["peerUrls"], result.peer_urls)
        push_chain.assert_called_once()

    def test_chain_push_audit_keeps_failure_visible(self) -> None:
        with patch(
            "cai_compute_chain.jobs.push_chain_to_cai_peers",
            side_effect=OSError("network unavailable"),
        ):
            audit = _run_chain_push_audit(
                state_payload={"nodeIdentities": {}},
                cai_url="http://127.0.0.1:52415",
                wallet_policy=None,
            )

        self.assertTrue(audit["attempted"])
        self.assertEqual(audit["status"], "failed")
        self.assertEqual(audit["errorType"], "OSError")
        self.assertIn("network unavailable", audit["message"])

    def test_task_level_node_capability_sync_keeps_best_effort_errors_visible(self) -> None:
        sync_result = SimpleNamespace(
            attempted_peers=2,
            successful_peers=1,
            failed_peers=1,
            imported_records=3,
            pruned_records=0,
            peer_urls=["http://node-a/v1/cai/node-capabilities"],
            failed_peer_urls=["http://node-b/v1/cai/node-capabilities"],
            peer_errors=[
                {
                    "peerUrl": "http://node-b/v1/cai/node-capabilities",
                    "errorType": "TimeoutError",
                    "message": "peer timed out",
                }
            ],
            convergence_status="partial",
            convergence_repair_recommended=True,
            convergence_repair_actions=["refresh_local_capability"],
        )
        with (
            patch(
                "cai_compute_chain.jobs.refresh_local_node_capabilities",
                side_effect=RuntimeError("local state unavailable"),
            ),
            patch(
                "cai_compute_chain.jobs.sync_node_capabilities_from_cai_peers",
                return_value=sync_result,
            ),
            self.assertLogs("cai_compute_chain.jobs", level="WARNING") as logs,
        ):
            audit = _sync_task_level_transport_node_capabilities_best_effort(
                state_payload={"nodeIdentities": {}},
                execution_cai_url="http://127.0.0.1:52415",
                requester_node_id="node-requester",
                wallet_policy=None,
            )

        self.assertTrue(audit["attempted"])
        self.assertEqual(audit["refreshLocal"]["status"], "failed")
        self.assertEqual(audit["refreshLocal"]["errorType"], "RuntimeError")
        self.assertIn("local state unavailable", audit["refreshLocal"]["message"])
        self.assertEqual(audit["peerSync"]["status"], "ok")
        self.assertEqual(audit["peerSync"]["attemptedPeers"], 2)
        self.assertEqual(audit["peerSync"]["failedPeerUrls"], sync_result.failed_peer_urls)
        self.assertEqual(audit["peerSync"]["peerErrors"], sync_result.peer_errors)
        self.assertEqual(audit["peerSync"]["convergenceStatus"], "partial")
        self.assertTrue(audit["peerSync"]["convergenceRepairRecommended"])
        self.assertIn("task-level node capability refresh failed", "\n".join(logs.output))

    def test_execution_performance_record_failure_is_logged(self) -> None:
        with (
            patch(
                "cai_compute_chain.jobs.record_execution_attempt_performance",
                side_effect=OSError("performance store locked"),
            ),
            self.assertLogs("cai_compute_chain.jobs", level="WARNING") as logs,
        ):
            _record_execution_attempt_performance_best_effort(
                model_id="Qwen/Qwen3-0.6B-GGUF",
                requester_node_id="node-requester",
                executor_node_ids=["node-worker"],
                status="failed",
                error_type="TimeoutError",
            )

        self.assertIn(
            "execution attempt performance record failed",
            "\n".join(logs.output),
        )
        self.assertIn("performance store locked", "\n".join(logs.output))

    def test_participant_snapshot_fallback_failure_is_logged(self) -> None:
        with (
            patch(
                "cai_compute_chain.jobs.resolve_cai_instance_snapshot",
                side_effect=TimeoutError("state endpoint timed out"),
            ),
            self.assertLogs("cai_compute_chain.jobs", level="WARNING") as logs,
        ):
            result = _best_effort_attempt_participant_node_ids(
                instance_snapshot=None,
                cai_url="http://127.0.0.1:52415",
                model_id="Qwen/Qwen3-0.6B-GGUF",
            )

        self.assertEqual(result, [])
        self.assertIn(
            "execution participant snapshot fallback failed",
            "\n".join(logs.output),
        )
        self.assertIn("state endpoint timed out", "\n".join(logs.output))

    def test_route_health_attestation_lookup_failure_is_logged(self) -> None:
        with (
            patch(
                "cai_compute_chain.jobs.list_worker_capability_attestations",
                side_effect=OSError("attestation store unavailable"),
            ),
            self.assertLogs("cai_compute_chain.jobs", level="WARNING") as logs,
        ):
            endpoints = _route_health_endpoints_from_worker_attestations(
                ["node-worker"],
                wallet_policy=None,
            )

        self.assertEqual(endpoints, [])
        self.assertIn(
            "route health worker capability attestation lookup failed",
            "\n".join(logs.output),
        )
        self.assertIn("attestation store unavailable", "\n".join(logs.output))

    def test_route_health_peer_sync_failure_is_logged(self) -> None:
        attestation = SimpleNamespace(
            worker_node_id="node-worker",
            accepted=True,
            expires_at=None,
            updated_at="2026-05-15T00:00:00+00:00",
            source_url="http://node-worker/v1/cai/node-capabilities",
        )
        with (
            patch(
                "cai_compute_chain.jobs.list_worker_capability_attestations",
                return_value=[attestation],
            ),
            patch(
                "cai_compute_chain.jobs._get_json",
                side_effect=TimeoutError("route health peer timed out"),
            ),
            self.assertLogs("cai_compute_chain.jobs", level="WARNING") as logs,
        ):
            records = _augment_route_health_records_from_worker_attestations(
                [],
                participant_node_ids=["node-requester", "node-worker"],
                wallet_policy=None,
            )

        self.assertEqual(records, [])
        self.assertIn(
            "route health peer sync from http://node-worker/v1/cai/route-health failed",
            "\n".join(logs.output),
        )
        self.assertIn("route health peer timed out", "\n".join(logs.output))

    def test_compute_cell_strategy_planning_failure_is_logged(self) -> None:
        with (
            patch(
                "cai_compute_chain.jobs.plan_llama_cpp_distributed_execution",
                side_effect=RuntimeError("route graph is inconsistent"),
            ),
            self.assertLogs("cai_compute_chain.jobs", level="WARNING") as logs,
        ):
            strategy = _execution_compute_cell_strategy(
                ["node-source", "node-sink"],
                route_health_records=[],
            )

        self.assertIsNone(strategy)
        self.assertIn(
            "distributed compute cell strategy planning failed",
            "\n".join(logs.output),
        )
        self.assertIn("route graph is inconsistent", "\n".join(logs.output))

    def test_task_level_final_output_base64_failure_is_logged(self) -> None:
        with self.assertLogs("cai_compute_chain.jobs", level="WARNING") as logs:
            text = _task_level_transport_final_output_text({"payloadBase64": "abc"})

        self.assertEqual(text, "")
        self.assertIn(
            "task-level transport final output base64 decode failed",
            "\n".join(logs.output),
        )
        self.assertIn("Incorrect padding", "\n".join(logs.output))

    def test_worker_execution_node_identity_audit_failure_is_logged(self) -> None:
        with (
            patch(
                "cai_compute_chain.jobs._get_json",
                side_effect=OSError("node identities endpoint offline"),
            ),
            self.assertLogs("cai_compute_chain.jobs", level="WARNING") as logs,
        ):
            audit = _resolve_worker_execution_node_audit(
                "http://127.0.0.1:52425",
                "public/test-model",
            )

        self.assertIsNone(audit)
        self.assertIn(
            "worker execution node identity audit failed",
            "\n".join(logs.output),
        )
        self.assertIn("node identities endpoint offline", "\n".join(logs.output))

    def test_worker_execution_node_summary_failure_is_logged_and_audited(self) -> None:
        def fake_get_json(url: str, *, timeout: int = 30) -> dict:
            if url.endswith("/state/nodeIdentities"):
                return {
                    "node-worker": {
                        "apiHost": "10.0.0.2",
                        "apiPort": 52425,
                    }
                }
            raise TimeoutError("worker summary timed out")

        with (
            patch("cai_compute_chain.jobs._get_json", side_effect=fake_get_json),
            patch(
                "cai_compute_chain.jobs._capability_records_by_node_id",
                return_value={},
            ),
            self.assertLogs("cai_compute_chain.jobs", level="WARNING") as logs,
        ):
            audit = _resolve_worker_execution_node_audit(
                "http://127.0.0.1:52425",
                "public/test-model",
            )

        assert audit is not None
        node = audit["nodes"][0]
        self.assertEqual(node["nodeId"], "node-worker")
        self.assertEqual(node["summaryStatus"], "failed")
        self.assertEqual(node["summaryError"]["errorType"], "TimeoutError")
        self.assertIn("worker summary timed out", node["summaryError"]["message"])
        self.assertIn(
            "worker summary fetch for node node-worker failed",
            "\n".join(logs.output),
        )
        self.assertIn("worker summary timed out", "\n".join(logs.output))

    def test_best_effort_cai_instance_cleanup_failure_is_logged(self) -> None:
        with (
            patch(
                "cai_compute_chain.jobs.list_cai_instances",
                side_effect=OSError("instances endpoint unavailable"),
            ),
            self.assertLogs("cai_compute_chain.jobs", level="WARNING") as logs,
        ):
            cleanup_cai_model_instances(
                "http://127.0.0.1:52425",
                "public/test-model",
                best_effort=True,
            )

        self.assertIn(
            "CAI model instance cleanup failed",
            "\n".join(logs.output),
        )
        self.assertIn("instances endpoint unavailable", "\n".join(logs.output))

    def test_cai_instance_readiness_polling_failure_is_logged_once(self) -> None:
        with (
            patch(
                "cai_compute_chain.jobs._get_json",
                side_effect=TimeoutError("state polling timed out"),
            ),
            patch("cai_compute_chain.jobs.time.sleep"),
            patch("cai_compute_chain.jobs.time.time", side_effect=[0.0, 0.0, 2.0]),
            self.assertLogs("cai_compute_chain.jobs", level="WARNING") as logs,
        ):
            ready = _wait_for_cai_instance_ready(
                "http://127.0.0.1:52425",
                "public/test-model",
                timeout_sec=1,
            )

        self.assertFalse(ready)
        output = "\n".join(logs.output)
        self.assertIn(
            "CAI instance readiness polling (1 failed attempt(s)) failed",
            output,
        )
        self.assertIn("state polling timed out", output)

    def test_orphan_llama_cleanup_instance_lookup_failure_is_logged(self) -> None:
        with (
            patch(
                "cai_compute_chain.jobs.list_cai_instances",
                side_effect=OSError("state instances endpoint unavailable"),
            ),
            patch("cai_compute_chain.runtime_cleanup.Path.exists", return_value=False),
            self.assertLogs("cai_compute_chain.jobs", level="WARNING") as logs,
        ):
            terminated = cleanup_orphan_llama_cpp_processes(
                cai_url="http://127.0.0.1:52425",
                model_id="public/test-model",
            )

        self.assertEqual(terminated, 0)
        self.assertIn(
            "orphan llama.cpp CAI instance lookup failed",
            "\n".join(logs.output),
        )
        self.assertIn("state instances endpoint unavailable", "\n".join(logs.output))

    def test_orphan_llama_cleanup_psutil_import_failure_is_logged(self) -> None:
        original_import = builtins.__import__

        def failing_psutil_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "psutil":
                raise ImportError("psutil unavailable")
            return original_import(name, globals, locals, fromlist, level)

        with self.assertLogs("cai_compute_chain.jobs", level="WARNING") as logs:
            with patch("builtins.__import__", side_effect=failing_psutil_import):
                terminated = cleanup_orphan_llama_cpp_processes()

        self.assertEqual(terminated, 0)
        self.assertIn(
            "orphan llama.cpp psutil import failed",
            "\n".join(logs.output),
        )
        self.assertIn("psutil unavailable", "\n".join(logs.output))

    def test_pending_download_labels_match_equivalent_runtime_model_id(self) -> None:
        labels = _pending_model_download_node_labels(
            {
                "nodeIdentities": {
                    "node-remote": {"friendlyName": "DESKTOP-REMOTE"},
                },
                "downloads": {
                    "node-remote": [
                        {
                            "DownloadPending": {
                                "shardMetadata": {
                                    "PipelineShardMetadata": {
                                        "modelCard": {
                                            "modelId": "cai-network/Qwen3-0.6B-GGUF"
                                        }
                                    }
                                }
                            }
                        }
                    ]
                },
            },
            "Qwen/Qwen3-0.6B-GGUF",
        )

        self.assertEqual(labels, ["DESKTOP-REMOTE"])


if __name__ == "__main__":
    unittest.main()

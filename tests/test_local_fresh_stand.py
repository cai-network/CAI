# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "local_fresh_stand.py"


def load_tool_module():
    spec = importlib.util.spec_from_file_location("local_fresh_stand", TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load local_fresh_stand tool module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LocalFreshStandTests(unittest.TestCase):
    def test_default_topology_is_mainnet_local_and_has_two_workers(self) -> None:
        tool = load_tool_module()

        specs = tool.default_node_specs("mainnet")
        self.assertEqual(
            [spec.name for spec in specs],
            ["validator-bootstrap", "client", "worker-a", "worker-b"],
        )
        self.assertFalse(specs[0].worker_enabled)
        self.assertTrue(specs[2].worker_enabled)
        self.assertTrue(specs[3].worker_enabled)
        self.assertTrue(all(spec.relay_enabled for spec in specs))

        bootstrap_peer = f"/ip4/127.0.0.1/tcp/{specs[0].libp2p_port}"
        self.assertEqual(specs[0].bootstrap_peers, ())
        self.assertEqual(specs[1].bootstrap_peers, (bootstrap_peer,))
        self.assertEqual(specs[2].bootstrap_peers, (bootstrap_peer,))
        self.assertEqual(specs[3].bootstrap_peers, (bootstrap_peer,))

        commands = [tool.runtime_command(spec) for spec in specs]
        self.assertNotIn("85.137.164.250", json.dumps(commands))

    def test_prepare_creates_isolated_node_dirs_configs_and_reports(self) -> None:
        tool = load_tool_module()

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "stand"
            manifest = tool.prepare_stand(
                root=root,
                network="mainnet",
                fresh=True,
                create_wallets=False,
            )

            self.assertEqual(manifest["network"], "mainnet")
            self.assertEqual(manifest["namespace"], "cai-ai-net")
            self.assertFalse(manifest["public_bootstrap_peers_used"])
            self.assertTrue((root / tool.MARKER_FILE_NAME).is_file())
            self.assertTrue((root / "stand.json").is_file())
            self.assertTrue(Path(manifest["reports"]["json"]).is_file())
            self.assertTrue(Path(manifest["reports"]["markdown"]).is_file())

            by_name = {node["name"]: node for node in manifest["nodes"]}
            self.assertFalse(by_name["client"]["node_config"]["worker_enabled"])
            self.assertTrue(by_name["worker-a"]["node_config"]["worker_enabled"])
            self.assertEqual(
                by_name["worker-a"]["node_config"]["worker_allowed_model_ids"],
                ["cai-network/Qwen3-0.6B-GGUF", "Qwen/Qwen3-0.6B-GGUF"],
            )
            self.assertNotIn(
                "qwen3-0.6b-4bit",
                json.dumps(by_name["worker-a"]["node_config"]).lower(),
            )

            manifest_text = (root / "stand.json").read_text(encoding="utf-8")
            self.assertNotIn("85.137.164.250", manifest_text)
            self.assertNotIn("seed phrase", manifest_text.lower())

    def test_safe_reset_refuses_unmarked_custom_existing_root(self) -> None:
        tool = load_tool_module()

        with tempfile.TemporaryDirectory() as tempdir:
            with self.assertRaises(ValueError):
                tool.safe_reset_stand_root(Path(tempdir))

    def test_evaluation_detects_partial_topology_visibility(self) -> None:
        tool = load_tool_module()

        manifest = {
            "nodes": [
                {
                    "name": "client",
                    "role": "client",
                    "probes": {
                        "state": {"ok": True, "topology_nodes": 2},
                        "summary": {
                            "ok": True,
                            "worker_enabled": False,
                            "relay_enabled": True,
                        },
                    },
                },
                {
                    "name": "worker-a",
                    "role": "worker",
                    "probes": {
                        "state": {"ok": True, "topology_nodes": 2},
                        "summary": {
                            "ok": True,
                            "worker_enabled": True,
                            "relay_enabled": True,
                        },
                    },
                },
                {
                    "name": "worker-b",
                    "role": "worker",
                    "probes": {
                        "state": {"ok": True, "topology_nodes": 1},
                        "summary": {
                            "ok": True,
                            "worker_enabled": True,
                            "relay_enabled": True,
                        },
                    },
                },
            ]
        }

        evaluation = tool.evaluate_smoke_manifest(manifest)

        self.assertFalse(evaluation["ok"])
        self.assertTrue(evaluation["api_ok"])
        self.assertTrue(evaluation["role_ok"])
        self.assertFalse(evaluation["topology_visibility_ok"])

    def test_node_audit_evaluation_detects_chain_tip_mismatch(self) -> None:
        tool = load_tool_module()

        manifest = {
            "create_wallets": True,
            "nodes": [
                {"name": "client", "role": "client"},
                {"name": "worker-a", "role": "worker"},
            ],
            "node_audits": {
                "client": {
                    "summary_ok": True,
                    "chain_ok": True,
                    "dashboard_ok": True,
                    "wallet": {"address": "addr-a", "unlocked": True},
                    "chain": {
                        "chain_id": "mainnet",
                        "schema_version": 1,
                        "network": "mainnet",
                        "block_count": 1,
                        "transaction_count": 2,
                        "tip_hash": "tip-a",
                        "tip_tx_root": "tx-root",
                        "tip_state_root": "state-root-a",
                    },
                    "network_summary": {
                        "knownWorkers": 1,
                        "llamaCppDistributedReady": True,
                    },
                },
                "worker-a": {
                    "summary_ok": True,
                    "chain_ok": True,
                    "dashboard_ok": True,
                    "wallet": {"address": "addr-b", "unlocked": True},
                    "chain": {
                        "chain_id": "mainnet",
                        "schema_version": 1,
                        "network": "mainnet",
                        "block_count": 1,
                        "transaction_count": 2,
                        "tip_hash": "tip-b",
                        "tip_tx_root": "tx-root",
                        "tip_state_root": "state-root-b",
                    },
                    "network_summary": {
                        "knownWorkers": 1,
                        "llamaCppDistributedReady": True,
                    },
                },
            },
        }

        evaluation = tool.evaluate_node_audits(manifest)

        self.assertFalse(evaluation["ok"])
        self.assertTrue(evaluation["wallet_preflight_ok"])
        self.assertFalse(evaluation["chain_sync_ok"])
        self.assertEqual(evaluation["chain_fingerprint_count"], 2)

    def test_node_audit_evaluation_accepts_synced_chain_and_worker_visibility(self) -> None:
        tool = load_tool_module()

        manifest = {
            "create_wallets": True,
            "nodes": [
                {"name": "client", "role": "client"},
                {"name": "worker-a", "role": "worker"},
                {"name": "worker-b", "role": "worker"},
            ],
            "node_audits": {
                name: {
                    "summary_ok": True,
                    "chain_ok": True,
                    "dashboard_ok": True,
                    "wallet": {"address": f"addr-{name}", "unlocked": True},
                    "chain": {
                        "chain_id": "mainnet",
                        "schema_version": 1,
                        "network": "mainnet",
                        "block_count": 1,
                        "transaction_count": 2,
                        "tip_hash": "same-tip",
                        "tip_tx_root": "same-tx-root",
                        "tip_state_root": "same-state-root",
                    },
                    "network_summary": {
                        "knownWorkers": 2,
                        "workerTotalRamBytes": 1024,
                        "workerTotalVramBytes": 0,
                        "workerTotalCpuCores": 8,
                        "llamaCppDistributedReady": True,
                    },
                }
                for name in ("client", "worker-a", "worker-b")
            },
        }

        evaluation = tool.evaluate_node_audits(manifest)

        self.assertTrue(evaluation["ok"])
        self.assertTrue(evaluation["chain_sync_ok"])
        self.assertTrue(evaluation["worker_visibility_ok"])

    def test_chain_payload_summary_builds_address_balance_index(self) -> None:
        tool = load_tool_module()

        summary = tool.summarize_chain_payload(
            {
                "chain": {
                    "chainId": "mainnet",
                    "schemaVersion": 1,
                    "blocks": [
                        {
                            "transactions": [
                                {"address": "ADDR-WORKER", "deltaAtomic": 125},
                                {"address": "addr-worker", "delta_atomic": -25},
                                {"address": "addr-zero", "deltaAtomic": 0},
                            ]
                        }
                    ],
                }
            }
        )

        self.assertEqual(summary["address_balances_atomic"], {"addr-worker": 100})

    def test_worker_reward_visibility_requires_synced_worker_balance_on_all_nodes(self) -> None:
        tool = load_tool_module()

        result = {
            "requester": "client",
            "summary": {"settlement_chain_recorded": True},
            "after": {
                "client": {
                    "chain_ok": True,
                    "latest_settlement": {
                        "chainTransactions": [
                            {
                                "txType": "worker_reward_credit",
                                "address": "addr-worker",
                                "deltaAtomic": 100,
                            }
                        ]
                    },
                    "chain": {
                        "address_balances_atomic": {"addr-worker": 100},
                    },
                },
                "worker-a": {
                    "chain_ok": True,
                    "chain": {
                        "address_balances_atomic": {"addr-worker": 0},
                    },
                },
            },
        }

        visibility = tool.summarize_worker_reward_visibility(result)

        self.assertFalse(visibility["ok"])
        self.assertEqual(
            visibility["expected_addresses"],
            [{"address": "addr-worker", "expected_min_balance_atomic": 100}],
        )
        by_node = {item["node"]: item for item in visibility["nodes"]}
        self.assertTrue(by_node["client"]["ok"])
        self.assertFalse(by_node["worker-a"]["ok"])

    def test_chat_result_summary_requires_answer_receipt_settlement_and_payouts(self) -> None:
        tool = load_tool_module()

        summary = tool.summarize_chat_post_result(
            {
                "ok": True,
                "json": {
                    "response": {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"role": "assistant", "content": "OK"},
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 5,
                            "completion_tokens": 1,
                            "total_tokens": 6,
                        },
                    },
                    "job": {
                        "jobId": "job-1",
                        "status": "completed",
                        "settlementId": "settlement-1",
                    },
                    "receipt": {
                        "receiptId": "receipt-1",
                        "finishReason": "stop",
                        "payoutCount": 2,
                        "workerPayoutTotalAtomic": 930,
                        "actualComputeCostAtomic": 1000,
                        "networkAudit": {
                            "transportMode": "multi_worker_direct",
                            "participantCount": 2,
                            "participantNodeIds": ["worker-a", "worker-b"],
                            "decentralizedExecution": True,
                            "checkedDirectSocketLinks": [
                                {"source": "worker-a", "target": "worker-b"}
                            ],
                            "checkedOverlayLinks": [],
                            "checkedRelayRoutes": [],
                        },
                    },
                },
            },
            min_payout_count=2,
            requester_audit={
                "latest_settlement": {
                    "settlementId": "settlement-1",
                    "computeCostAtomic": 1000,
                    "settlementFeeAtomic": 20,
                    "aiDevelopmentFeeAtomic": 50,
                    "workerRewardAtomic": 930,
                    "chainRecorded": True,
                    "chainTransactionCount": 6,
                }
            },
        )

        self.assertTrue(summary["ok"])
        self.assertEqual(summary["answer_text"], "OK")
        self.assertEqual(summary["job_id"], "job-1")
        self.assertEqual(summary["receipt_id"], "receipt-1")
        self.assertEqual(summary["settlement_id"], "settlement-1")
        self.assertEqual(summary["payout_count"], 2)
        self.assertEqual(summary["usage"]["total_tokens"], 6)
        self.assertTrue(summary["route_audit_recorded"])
        self.assertTrue(summary["decentralized_route_ok"])
        self.assertEqual(summary["route_audit"]["checked_direct_socket_link_count"], 1)

    def test_chat_result_summary_fails_without_minimum_payouts(self) -> None:
        tool = load_tool_module()

        summary = tool.summarize_chat_post_result(
            {
                "ok": True,
                "json": {
                    "response": {
                        "choices": [
                            {"message": {"role": "assistant", "content": "OK"}}
                        ],
                    },
                    "job": {
                        "jobId": "job-1",
                        "status": "completed",
                        "settlementId": "settlement-1",
                    },
                    "receipt": {
                        "receiptId": "receipt-1",
                        "payoutCount": 1,
                    },
                },
            },
            min_payout_count=2,
        )

        self.assertFalse(summary["ok"])
        self.assertTrue(summary["answer_returned"])
        self.assertFalse(summary["minimum_payout_count_met"])

    def test_chat_result_summary_uses_requester_audit_for_openai_body(self) -> None:
        tool = load_tool_module()

        summary = tool.summarize_chat_post_result(
            {
                "ok": True,
                "json": {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "OK"},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 1,
                        "total_tokens": 6,
                    },
                },
            },
            min_payout_count=2,
            requester_audit={
                "latest_job": {
                    "jobId": "job-1",
                    "status": "completed",
                    "settlementId": "settlement-1",
                },
                "latest_receipt": {
                    "receiptId": "receipt-1",
                    "finishReason": "stop",
                    "payoutCount": 2,
                    "workerPayoutTotalAtomic": 930,
                    "actualComputeCostAtomic": 1000,
                    "networkAudit": {
                        "transportMode": "multi_worker_relay",
                        "participantCount": 2,
                        "participantNodeIds": ["worker-a", "worker-b"],
                        "decentralizedExecution": True,
                        "relayHopsUsed": True,
                        "checkedDirectSocketLinks": [],
                        "checkedOverlayLinks": [],
                        "checkedRelayRoutes": [
                            {
                                "source": "worker-a",
                                "target": "worker-b",
                                "transit": "relay-a",
                            }
                        ],
                    },
                },
                "latest_settlement": {
                    "settlementId": "settlement-1",
                    "computeCostAtomic": 1000,
                    "settlementFeeAtomic": 20,
                    "aiDevelopmentFeeAtomic": 50,
                    "workerRewardAtomic": 930,
                    "chainRecorded": True,
                    "chainTransactionCount": 6,
                },
            },
        )

        self.assertTrue(summary["ok"])
        self.assertEqual(summary["answer_text"], "OK")
        self.assertEqual(summary["job_id"], "job-1")
        self.assertEqual(summary["receipt_id"], "receipt-1")
        self.assertEqual(summary["settlement_id"], "settlement-1")
        self.assertEqual(summary["transport_mode"], "multi_worker_relay")
        self.assertEqual(summary["route_audit"]["checked_relay_route_count"], 1)

    def test_chat_result_summary_fails_on_reward_accounting_mismatch(self) -> None:
        tool = load_tool_module()

        summary = tool.summarize_chat_post_result(
            {
                "ok": True,
                "json": {
                    "choices": [
                        {"message": {"role": "assistant", "content": "OK"}}
                    ],
                    "job": {
                        "jobId": "job-1",
                        "status": "completed",
                        "settlementId": "settlement-1",
                    },
                    "receipt": {
                        "receiptId": "receipt-1",
                        "payoutCount": 2,
                        "workerPayoutTotalAtomic": 900,
                        "actualComputeCostAtomic": 1000,
                        "networkAudit": {
                            "transportMode": "multi_worker_direct",
                            "participantCount": 2,
                            "decentralizedExecution": True,
                        },
                    },
                },
            },
            min_payout_count=2,
            requester_audit={
                "latest_settlement": {
                    "settlementId": "settlement-1",
                    "computeCostAtomic": 1000,
                    "settlementFeeAtomic": 20,
                    "aiDevelopmentFeeAtomic": 50,
                    "workerRewardAtomic": 930,
                    "chainRecorded": True,
                    "chainTransactionCount": 6,
                }
            },
        )

        self.assertFalse(summary["ok"])
        self.assertFalse(summary["reward_accounting_ok"])
        self.assertFalse(
            summary["reward_accounting"]["payout_matches_worker_reward"]
        )

    def test_chat_result_summary_requires_settlement_chain_transactions(self) -> None:
        tool = load_tool_module()

        summary = tool.summarize_chat_post_result(
            {
                "ok": True,
                "json": {
                    "choices": [
                        {"message": {"role": "assistant", "content": "OK"}}
                    ],
                    "job": {
                        "jobId": "job-1",
                        "status": "completed",
                        "settlementId": "settlement-1",
                    },
                    "receipt": {
                        "receiptId": "receipt-1",
                        "payoutCount": 2,
                        "workerPayoutTotalAtomic": 930,
                        "actualComputeCostAtomic": 1000,
                        "networkAudit": {
                            "transportMode": "multi_worker_direct",
                            "participantCount": 2,
                            "decentralizedExecution": True,
                        },
                    },
                },
            },
            min_payout_count=2,
            requester_audit={
                "latest_settlement": {
                    "settlementId": "settlement-1",
                    "computeCostAtomic": 1000,
                    "settlementFeeAtomic": 20,
                    "aiDevelopmentFeeAtomic": 50,
                    "workerRewardAtomic": 930,
                    "chainRecorded": False,
                    "chainTransactionCount": 0,
                }
            },
        )

        self.assertFalse(summary["ok"])
        self.assertFalse(summary["settlement_chain_recorded"])
        self.assertFalse(summary["reward_accounting_ok"])
        self.assertFalse(
            summary["reward_accounting"]["settlement_chain_recorded"]
        )

    def test_chat_result_summary_requires_route_audit_for_two_worker_request(self) -> None:
        tool = load_tool_module()

        summary = tool.summarize_chat_post_result(
            {
                "ok": True,
                "json": {
                    "choices": [
                        {"message": {"role": "assistant", "content": "OK"}}
                    ],
                    "job": {
                        "jobId": "job-1",
                        "status": "completed",
                        "settlementId": "settlement-1",
                    },
                    "receipt": {
                        "receiptId": "receipt-1",
                        "payoutCount": 2,
                        "workerPayoutTotalAtomic": 930,
                        "actualComputeCostAtomic": 1000,
                    },
                },
            },
            min_payout_count=2,
            requester_audit={
                "latest_settlement": {
                    "settlementId": "settlement-1",
                    "computeCostAtomic": 1000,
                    "settlementFeeAtomic": 20,
                    "aiDevelopmentFeeAtomic": 50,
                    "workerRewardAtomic": 930,
                    "chainRecorded": True,
                    "chainTransactionCount": 6,
                }
            },
        )

        self.assertFalse(summary["ok"])
        self.assertFalse(summary["route_audit_recorded"])
        self.assertFalse(summary["decentralized_route_ok"])

    def test_request_failure_classification_detects_decentralized_route_failure(self) -> None:
        tool = load_tool_module()

        classification = tool.classify_request_failure(
            {
                "ok": False,
                "summary": {
                    "answer_returned": True,
                    "receipt_recorded": True,
                    "settlement_recorded": True,
                    "settlement_chain_recorded": True,
                    "minimum_payout_count_met": True,
                    "route_audit_recorded": True,
                    "decentralized_route_ok": False,
                    "settlement_id": "settlement-1",
                    "route_audit": {
                        "transport_mode": "multi_worker_disconnected",
                        "participant_count": 2,
                        "decentralized_execution": False,
                    },
                },
            }
        )

        self.assertEqual(classification["phase"], "decentralized_route_failed")
        self.assertEqual(classification["settlement_id"], "settlement-1")

    def test_request_runtime_overrides_keep_server_timeout_bounded(self) -> None:
        tool = load_tool_module()

        overrides = tool.request_runtime_overrides(
            instance_ready_timeout_sec=45,
            request_timeout_sec=50,
        )

        self.assertEqual(overrides["CAI_INSTANCE_READY_TIMEOUT_SECONDS"], "45")
        self.assertEqual(overrides["CAI_PRIVATE_INSTANCE_READY_TIMEOUT_SECONDS"], "45")
        self.assertEqual(overrides["CAI_CHAT_COMPLETION_TIMEOUT_SECONDS"], "50")

    def test_request_scenario_specs_use_public_single_and_private_two_worker_models(self) -> None:
        tool = load_tool_module()

        specs = tool.request_scenario_specs("all")

        by_name = {item["name"]: item for item in specs}
        self.assertEqual(
            by_name["client_to_network"]["model_id"],
            "Qwen/Qwen3-0.6B-GGUF",
        )
        self.assertEqual(
            by_name["worker_as_client"]["model_id"],
            "Qwen/Qwen3-0.6B-GGUF",
        )
        self.assertEqual(
            by_name["client_to_two_workers"]["model_id"],
            "cai-network/Qwen3-0.6B-GGUF",
        )
        self.assertEqual(by_name["client_to_network"]["min_payout_count"], 1)
        self.assertEqual(by_name["client_to_two_workers"]["min_payout_count"], 2)

    def test_request_failure_classification_detects_not_ready_runners(self) -> None:
        tool = load_tool_module()

        classification = tool.classify_request_failure(
            {
                "ok": False,
                "requester": "client",
                "post": {"error": "TimeoutError: timed out"},
                "summary": {"error": "TimeoutError: timed out"},
                "runtime_snapshots": {
                    "client": {
                        "sections": {
                            "instances": {"summary": {"count": 1}},
                            "runners": {
                                "summary": {
                                    "status_counts": {
                                        "RunnerConnecting": 1,
                                        "RunnerReady": 1,
                                    }
                                }
                            },
                        }
                    }
                },
            }
        )

        self.assertEqual(classification["phase"], "instance_created_but_runners_not_ready")
        self.assertEqual(
            classification["non_ready_runner_status_counts"],
            {"RunnerConnecting": 1},
        )

    def test_request_failure_classification_detects_missing_rpc_proof(self) -> None:
        tool = load_tool_module()

        classification = tool.classify_request_failure(
            {
                "ok": False,
                "post": {
                    "error": {
                        "detail": (
                            "No proven decentralized llama.cpp RPC route remains. "
                            "Strict RPC proof is enabled."
                        )
                    }
                },
                "runtime_snapshots": {
                    "client": {
                        "sections": {
                            "instances": {"summary": {"count": 0}},
                            "runners": {"summary": {"status_counts": {}}},
                        }
                    }
                },
                "requester": "client",
            }
        )

        self.assertEqual(classification["phase"], "llama_cpp_rpc_proof_missing")
        self.assertIn("Strict RPC proof", classification["reason"])

    def test_request_failure_classification_detects_runtime_rpc_failure(self) -> None:
        tool = load_tool_module()

        classification = tool.classify_request_failure(
            {
                "ok": False,
                "post": {
                    "error": {
                        "detail": (
                            "Timed out waiting for remote llama.cpp rpc-server peers: "
                            "127.0.0.1:60123"
                        )
                    }
                },
                "runtime_snapshots": {
                    "client": {
                        "sections": {
                            "instances": {"summary": {"count": 1}},
                            "runners": {
                                "summary": {
                                    "status_counts": {"RunnerLoading": 1}
                                }
                            },
                            "route_health": {
                                "summary": {
                                    "failed_llama_cpp_rpc_records": [
                                        {
                                            "source_node_id": "node-a",
                                            "sink_node_id": "node-b",
                                            "route_type": "llama_cpp_rpc_relay",
                                            "endpoint_url": "relay://node-r/host:50052",
                                            "error": "connection closed during HELLO",
                                        }
                                    ]
                                }
                            },
                        }
                    }
                },
                "requester": "client",
            }
        )

        self.assertEqual(classification["phase"], "llama_cpp_rpc_runtime_failed")
        self.assertEqual(
            classification["failed_llama_cpp_rpc_records"][0]["sink_node_id"],
            "node-b",
        )

    def test_route_health_summary_extracts_llama_cpp_rpc_failures(self) -> None:
        tool = load_tool_module()

        summary = tool.summarize_route_health_payload(
            {
                "records": [
                    {
                        "source_node_id": "node-a",
                        "sink_node_id": "node-b",
                        "route_type": "llama_cpp_rpc_direct",
                        "reachable": False,
                        "endpoint_url": "llama-cpp-rpc://node-b:50052",
                        "error": "malformed response",
                    },
                    {
                        "source_node_id": "node-a",
                        "sink_node_id": "node-c",
                        "route_type": "direct_data",
                        "reachable": True,
                    },
                ]
            }
        )

        self.assertEqual(summary["failed_llama_cpp_rpc_count"], 1)
        self.assertEqual(
            summary["route_type_counts"],
            {"llama_cpp_rpc_direct": 1, "direct_data": 1},
        )

    def test_request_failure_classification_detects_missing_settlement_chain(self) -> None:
        tool = load_tool_module()

        classification = tool.classify_request_failure(
            {
                "ok": False,
                "summary": {
                    "answer_returned": True,
                    "receipt_recorded": True,
                    "settlement_recorded": True,
                    "settlement_chain_recorded": False,
                    "minimum_payout_count_met": True,
                    "settlement_id": "settlement-1",
                },
            }
        )

        self.assertEqual(classification["phase"], "settlement_chain_missing")
        self.assertEqual(classification["settlement_id"], "settlement-1")


if __name__ == "__main__":
    unittest.main()

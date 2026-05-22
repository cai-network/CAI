# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, patch
from urllib.error import HTTPError

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.jobs import (
    _accepted_worker_model_ids,
    _cai_api_urls_by_node_id,
    _cai_summary_urls_by_node_id,
    _candidate_cai_chat_base_urls,
    _build_execution_network_audit,
    _build_participant_eligibility_audit,
    _describe_pending_model_downloads,
    _distribute_worker_reward,
    _execution_node_id_attempts,
    _resolve_worker_execution_node_audit,
    _resolve_cai_instance_create_payload,
    _resolve_cai_instance_create_payload_for_nodes,
    _select_task_level_transport_executor_node_ids,
    _submit_text_job_to_cai,
    _model_selection_audit,
    _require_settleable_instance_snapshot,
    _task_level_transport_effective_executor_count,
    _task_level_transport_executor_fallback_attempts,
    _task_level_transport_initial_prompt_text,
    _task_level_transport_llm_runtime_metadata,
    _task_level_transport_planned_shard_ranges,
    _sync_worker_reward_bindings_from_cai,
    _task_level_transport_instance_snapshot,
    _task_level_transport_total_layer_count,
    _validate_request_payload_model_matches_job,
    _worker_model_allowed,
    apply_local_validator_attestation,
    cai_instance_readiness_audit,
    cleanup_cai_model_instances,
    create_job_intent,
    ensure_cai_instance,
    execute_job_intent,
    list_cai_instances,
    list_execution_receipts,
    list_job_intents,
    reconcile_stale_running_job_intents,
    repair_local_worker_reward_state,
    request_remote_committee_attestations,
    resolve_cai_command_instance_snapshot,
    update_job_intent,
)
from cai_compute_chain.decentralized_compute import (
    build_cai_owned_transport_execution_proof,
    complete_cai_owned_transport_batch_processing,
    complete_cai_owned_transport_session,
    create_cai_owned_transport_session,
    record_cai_owned_transport_batch,
)
from cai_compute_chain.execution_performance import (
    list_execution_performance_records,
    record_execution_attempt_performance,
)
from cai_compute_chain.route_health import RouteHealthRecord
from cai_compute_chain.chain import (
    append_chain_block,
    chain_balance_atomic,
    chain_settlement_history,
    compute_reserve_chain_address,
    ensure_chain_genesis,
    list_chain_blocks,
    make_chain_transaction,
)
from cai_compute_chain.model import MoneyPolicy, NetworkModelPolicy, PaymentPreference
from cai_compute_chain.node_capabilities import (
    NodeCapabilityRecord,
    save_node_capabilities,
)
from cai_compute_chain.node_config import (
    bind_worker_reward_address,
    load_or_create_node_config,
    resolve_worker_reward_address,
    set_validator_ha_mode,
    set_validator_mode,
    set_validator_static_ip_confirmation,
)
from cai_compute_chain.settlement import (
    SettlementRecord,
    list_attestations,
    list_settlements,
    list_validator_evidence,
    list_worker_payouts,
    record_funding_settlement,
    record_settlement_execution_audit,
    record_worker_payouts,
    record_validator_attestation,
    save_settlements,
    sign_settlement_envelope,
)
from cai_compute_chain.economics import plan_funding
from cai_compute_chain.wallet import (
    coins_to_atomic,
    create_wallet,
    credit_wallet,
    get_active_wallet,
    list_journal_entries,
    load_or_create_ledger,
    list_wallets,
    load_session,
    unlock_wallet,
)
from cai_compute_chain.validators import sync_validator_record


class JobIntentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self._patchers = [
            patch(
                "cai_compute_chain.wallet.repo_root",
                return_value=Path(self.tempdir.name),
            ),
            patch("cai_compute_chain.jobs.sync_validator_set_from_cai_peers", return_value=None),
            patch("cai_compute_chain.jobs.sync_chain_from_cai_peers", return_value=None),
            patch(
                "cai_compute_chain.settlement.sync_validator_set_from_cai_peers",
                return_value=None,
            ),
        ]
        if not self._testMethodName.startswith("test_resolve_cai_command_instance_snapshot"):
            self._patchers.append(
                patch(
                    "cai_compute_chain.jobs.resolve_cai_command_instance_snapshot",
                    return_value=None,
                )
            )
        for patcher in self._patchers:
            patcher.start()
        self._chain_credit_counter = 0

    def tearDown(self) -> None:
        for patcher in reversed(self._patchers):
            patcher.stop()
        self.tempdir.cleanup()

    def _credit_wallet_on_chain(self, wallet, amount_atomic: int) -> None:
        self._chain_credit_counter += 1
        money_policy = MoneyPolicy()
        ensure_chain_genesis(money_policy=money_policy)
        reserve_debit = make_chain_transaction(
            tx_type="test_compute_reserve_debit",
            address=compute_reserve_chain_address(money_policy),
            delta_atomic=-amount_atomic,
            wallet_id=wallet.wallet_id,
            note="Test chain credit reserve debit.",
            counterparty_address=wallet.address,
            nonce=(
                f"test-wallet-credit:{wallet.wallet_id}:"
                f"{self._chain_credit_counter}:reserve-debit"
            ),
            chain_id=money_policy.chain_network.value,
        )
        wallet_credit = make_chain_transaction(
            tx_type="test_wallet_credit",
            address=wallet.address,
            delta_atomic=amount_atomic,
            wallet_id=wallet.wallet_id,
            note="Test chain credit.",
            counterparty_address=compute_reserve_chain_address(money_policy),
            nonce=f"test-wallet-credit:{wallet.wallet_id}:{self._chain_credit_counter}",
            chain_id=money_policy.chain_network.value,
        )
        self.assertIsNotNone(append_chain_block([reserve_debit, wallet_credit]))

    def _create_verified_cai_owned_transport_proof(
        self,
        instance_id: str,
        *,
        metrics_by_node: dict[str, dict] | None = None,
    ) -> dict:
        session = create_cai_owned_transport_session(
            instance_id=instance_id,
            participant_node_ids=["node-a", "node-b"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
        )
        for node_id, payload_hash, layer_start, layer_end in [
            ("node-a", "a1" * 32, 0, 14),
            ("node-b", "b2" * 32, 14, 28),
        ]:
            batch_id = f"caibatch_{node_id.replace('-', '_')}_{instance_id}"
            record_cai_owned_transport_batch(
                session.session_id,
                batch_id=batch_id,
                phase="prefill_activation_batches",
                source_node_id="node-a",
                sink_node_id=node_id,
                payload_size_bytes=32,
                payload_sha256_hex=payload_hash,
                metadata={"layerStart": layer_start, "layerEnd": layer_end},
                status="received",
            )
            complete_cai_owned_transport_batch_processing(
                session.session_id,
                batch_id,
                node_id=node_id,
                metrics=(
                    dict(metrics_by_node[node_id])
                    if metrics_by_node and node_id in metrics_by_node
                    else None
                ),
                output_payload=f"{node_id}:{instance_id}".encode("utf-8"),
            )
        completed = complete_cai_owned_transport_session(session.session_id)
        self.assertIsNotNone(completed.proof)
        return dict(completed.proof or {})

    def test_create_job_intent_persists_job(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))

        job = create_job_intent(
            prompt="2+2=?",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
        )

        jobs = list_job_intents()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].job_id, job.job_id)
        self.assertEqual(jobs[0].status, "created")

    def test_request_payload_model_validation_accepts_execution_alias(self) -> None:
        policy = NetworkModelPolicy()

        _validate_request_payload_model_matches_job(
            job_model_id="cai-network/Qwen3-0.6B-GGUF",
            execution_model_id="Qwen/Qwen3-0.6B-GGUF",
            request_payload_override={"model": "Qwen/Qwen3-0.6B-GGUF"},
            network_model_policy=policy,
        )

        audit = _model_selection_audit(
            job_model_id="cai-network/Qwen3-0.6B-GGUF",
            execution_model_id="Qwen/Qwen3-0.6B-GGUF",
            request_payload_override={"model": "Qwen/Qwen3-0.6B-GGUF"},
            network_model_policy=policy,
        )

        self.assertEqual(audit["status"], "matched")
        self.assertTrue(audit["requestPayloadMatchesJob"])
        self.assertFalse(audit["requestPayloadModelOverridden"])

    def test_request_payload_model_validation_rejects_model_drift(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "does not match metered job model",
        ):
            _validate_request_payload_model_matches_job(
                job_model_id="cai-network/Qwen3-0.6B-GGUF",
                execution_model_id="Qwen/Qwen3-0.6B-GGUF",
                request_payload_override={
                    "model": "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
                },
                network_model_policy=NetworkModelPolicy(),
            )

    def test_create_job_intent_persists_requester_node_id(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))

        job = create_job_intent(
            prompt="2+2=?",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
            requester_node_id="node-local",
        )

        jobs = list_job_intents()
        self.assertEqual(job.requester_node_id, "node-local")
        self.assertEqual(jobs[0].requester_node_id, "node-local")

    def test_create_job_intent_creates_device_wallet_when_missing(self) -> None:
        job = create_job_intent(
            prompt="2+2=?",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
        )

        wallet = get_active_wallet()
        session = load_session()
        self.assertIsNotNone(wallet)
        assert wallet is not None
        self.assertEqual(wallet.name, "CAI Device Wallet")
        self.assertEqual(job.source_wallet_id, wallet.wallet_id)
        self.assertEqual(len(list_wallets()), 1)
        self.assertIsNone(session.unlocked_wallet_id)

    def test_create_job_intent_auto_prices_when_amount_missing(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))

        with patch(
            "cai_compute_chain.jobs.resolve_compute_price"
        ) as resolve_compute_price:
            resolve_compute_price.return_value.compute_cost_atomic = coins_to_atomic(
                "0.00100000"
            )
            resolve_compute_price.return_value.pricing_mode = "network_auto"
            resolve_compute_price.return_value.pricing_reason = "Auto quote"
            resolve_compute_price.return_value.automatic_quote = None

            job = create_job_intent(
                prompt="2+2=?",
                compute_amount_coins=None,
                payment_preference=PaymentPreference.AUTO,
                cai_url="http://127.0.0.1:52425",
            )

        self.assertEqual(job.pricing_mode, "network_auto")
        self.assertEqual(job.pricing_reason, "Auto quote")
        self.assertEqual(job.requested_compute_cost_atomic, coins_to_atomic("0.00100000"))

    def test_candidate_cai_chat_base_urls_formats_ipv6_remote_host(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-local": {"apiHost": "127.0.0.1", "apiPort": 52425},
                "node-remote": {"apiHost": "2001:db8::10", "apiPort": 52415},
            }
        }

        with patch(
            "cai_compute_chain.jobs._load_cai_state_payload",
            return_value=state_payload,
        ):
            candidates = _candidate_cai_chat_base_urls("http://127.0.0.1:52425")

        self.assertEqual(
            candidates,
            [
                "http://127.0.0.1:52425",
                "http://[2001:db8::10]:52415",
            ],
        )

    def test_candidate_cai_chat_base_urls_use_transport_endpoints_before_legacy_api_host(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-local": {"apiHost": "127.0.0.1", "apiPort": 52425},
                "node-remote": {
                    "apiHost": "198.51.100.99",
                    "apiPort": 52415,
                    "transportEndpoints": [
                        {
                            "purpose": "api",
                            "routeType": "overlay",
                            "host": "26.97.29.153",
                            "port": 52415,
                            "source": "interface_scan",
                        },
                        {
                            "purpose": "api",
                            "routeType": "direct",
                            "host": "85.137.164.250",
                            "port": 52415,
                            "source": "explicit",
                        },
                    ],
                },
            }
        }

        with patch(
            "cai_compute_chain.jobs._load_cai_state_payload",
            return_value=state_payload,
        ):
            candidates = _candidate_cai_chat_base_urls("http://127.0.0.1:52425")

        self.assertEqual(
            candidates,
            [
                "http://127.0.0.1:52425",
                "http://85.137.164.250:52415",
                "http://26.97.29.153:52415",
                "http://198.51.100.99:52415",
            ],
        )

    def test_cai_summary_urls_by_node_id_formats_ipv6_remote_host(self) -> None:
        urls = _cai_summary_urls_by_node_id(
            "http://127.0.0.1:52425",
            {
                "nodeIdentities": {
                    "node-local": {"apiHost": "127.0.0.1", "apiPort": 52425},
                    "node-remote": {"apiHost": "2001:db8::10", "apiPort": 52415},
                }
            },
        )

        self.assertEqual(
            urls,
            {
                "node-local": "http://127.0.0.1:52425/v1/cai/summary",
                "node-remote": "http://[2001:db8::10]:52415/v1/cai/summary",
            },
        )

    def test_cai_summary_urls_by_node_id_uses_transport_endpoints_before_legacy_api_host(self) -> None:
        urls = _cai_summary_urls_by_node_id(
            "http://127.0.0.1:52425",
            {
                "nodeIdentities": {
                    "node-local": {"apiHost": "127.0.0.1", "apiPort": 52425},
                    "node-remote": {
                        "apiHost": "198.51.100.99",
                        "apiPort": 52415,
                        "transportEndpoints": [
                            {
                                "purpose": "api",
                                "routeType": "overlay",
                                "host": "26.97.29.153",
                                "port": 52415,
                                "source": "interface_scan",
                            },
                            {
                                "purpose": "api",
                                "routeType": "direct",
                                "host": "85.137.164.250",
                                "port": 52415,
                                "source": "explicit",
                            },
                        ],
                    },
                }
            },
        )

        self.assertEqual(
            urls,
            {
                "node-local": "http://127.0.0.1:52425/v1/cai/summary",
                "node-remote": "http://85.137.164.250:52415/v1/cai/summary",
            },
        )

    def test_cai_api_urls_by_node_id_preserves_all_candidates(self) -> None:
        urls = _cai_api_urls_by_node_id(
            "http://127.0.0.1:52425",
            {
                "nodeIdentities": {
                    "node-local": {"apiHost": "127.0.0.1", "apiPort": 52425},
                    "node-remote": {
                        "apiHost": "198.51.100.99",
                        "apiPort": 52415,
                        "transportEndpoints": [
                            {
                                "purpose": "api",
                                "routeType": "overlay",
                                "host": "26.97.29.153",
                                "port": 52415,
                            },
                            {
                                "purpose": "api",
                                "routeType": "direct",
                                "host": "85.137.164.250",
                                "port": 52415,
                            },
                        ],
                    },
                }
            },
        )

        self.assertEqual(urls["node-local"], ["http://127.0.0.1:52425"])
        self.assertEqual(
            urls["node-remote"],
            [
                "http://85.137.164.250:52415",
                "http://26.97.29.153:52415",
                "http://198.51.100.99:52415",
            ],
        )

    def test_cai_api_urls_by_node_id_adds_overlay_urls_through_bootstrap_relay(
        self,
    ) -> None:
        urls = _cai_api_urls_by_node_id(
            "http://127.0.0.1:52425",
            {
                "nodeIdentities": {
                    "node-local": {
                        "apiHost": "127.0.0.1",
                        "apiPort": 52425,
                        "workerEnabled": True,
                    },
                    "node-bootstrap": {
                        "relayEnabled": True,
                        "workerEnabled": False,
                        "transportEndpoints": [
                            {
                                "purpose": "api",
                                "routeType": "direct",
                                "host": "85.137.164.250",
                                "port": 52415,
                            }
                        ],
                    },
                    "node-remote": {
                        "workerEnabled": True,
                        "relayEnabled": True,
                    },
                },
                "overlayPeers": {
                    "node-bootstrap": ["node-local", "node-remote"],
                    "node-local": ["node-bootstrap"],
                    "node-remote": ["node-bootstrap"],
                },
            },
        )

        self.assertEqual(
            urls["node-remote"],
            [
                "cai-overlay:http://85.137.164.250:52415?"
                "targetNodeId=node-remote&relayRole=bootstrap",
            ],
        )
        self.assertEqual(
            urls["node-local"],
            [
                "http://127.0.0.1:52425",
                "cai-overlay:http://85.137.164.250:52415?"
                "targetNodeId=node-local&relayRole=bootstrap",
            ],
        )

    def test_cai_api_urls_by_node_id_marks_worker_relay_as_ordinary(self) -> None:
        urls = _cai_api_urls_by_node_id(
            "http://127.0.0.1:52425",
            {
                "nodeIdentities": {
                    "node-local": {"apiHost": "127.0.0.1", "apiPort": 52425},
                    "node-relay": {
                        "relayEnabled": True,
                        "workerEnabled": True,
                        "transportEndpoints": [
                            {
                                "purpose": "api",
                                "routeType": "direct",
                                "host": "relay.example",
                                "port": 52415,
                            }
                        ],
                    },
                    "node-remote": {"workerEnabled": True},
                },
                "overlayPeers": {"node-relay": ["node-remote"]},
            },
        )

        self.assertEqual(
            urls["node-remote"],
            [
                "cai-overlay:http://relay.example:52415?"
                "targetNodeId=node-remote&relayRole=ordinary",
            ],
        )

    def test_cai_api_urls_by_node_id_skips_unconnected_overlay_relay(self) -> None:
        urls = _cai_api_urls_by_node_id(
            "http://127.0.0.1:52425",
            {
                "nodeIdentities": {
                    "node-local": {"apiHost": "127.0.0.1", "apiPort": 52425},
                    "node-bootstrap": {
                        "relayEnabled": True,
                        "transportEndpoints": [
                            {
                                "purpose": "api",
                                "routeType": "direct",
                                "host": "85.137.164.250",
                                "port": 52415,
                            }
                        ],
                    },
                    "node-remote": {"workerEnabled": True},
                },
                "overlayPeers": {"node-bootstrap": ["node-other"]},
            },
        )

        self.assertNotIn("node-remote", urls)

    def test_cai_api_urls_by_node_id_uses_cached_capability_urls_for_missing_worker(
        self,
    ) -> None:
        save_node_capabilities(
            [
                NodeCapabilityRecord(
                    node_id="node-remote",
                    source="peer",
                    source_url="http://85.137.164.250:52415",
                    last_seen_at=datetime.now(tz=UTC).isoformat(),
                    updated_at=datetime.now(tz=UTC).isoformat(),
                    api_urls=["http://26.97.29.153:52425"],
                    worker_enabled=True,
                    worker_reward_address="bbbb1234bbbb1234bbbb1234bbbb1234",
                )
            ]
        )

        urls = _cai_api_urls_by_node_id(
            "http://127.0.0.1:52425",
            {
                "nodeIdentities": {
                    "node-local": {"apiHost": "127.0.0.1", "apiPort": 52425},
                    "node-bootstrap": {
                        "apiHost": "85.137.164.250",
                        "apiPort": 52415,
                        "relayEnabled": True,
                    },
                }
            },
        )

        self.assertEqual(urls["node-remote"], ["http://26.97.29.153:52425"])

    def test_resolve_worker_execution_node_audit_uses_cached_capability_record_when_state_missing_worker(
        self,
    ) -> None:
        observed_at = datetime.now(tz=UTC).isoformat()
        save_node_capabilities(
            [
                NodeCapabilityRecord(
                    node_id="node-remote",
                    source="peer",
                    source_url="http://85.137.164.250:52415",
                    last_seen_at=observed_at,
                    updated_at=observed_at,
                    api_urls=["http://26.97.29.153:52425"],
                    worker_enabled=True,
                    worker_reward_address="bbbb1234bbbb1234bbbb1234bbbb1234",
                    worker_allowed_model_ids=["public/test-model"],
                    worker_verified=True,
                )
            ]
        )

        with patch(
            "cai_compute_chain.jobs._get_json",
            return_value={
                "node-bootstrap": {
                    "apiHost": "85.137.164.250",
                    "apiPort": 52415,
                    "workerEnabled": False,
                    "workerRewardAddress": "validator-address",
                }
            },
        ):
            audit = _resolve_worker_execution_node_audit(
                "http://127.0.0.1:52425",
                "public/test-model",
            )

        assert audit is not None
        self.assertEqual(audit["eligibleNodeIds"], ["node-remote"])
        remote_item = next(item for item in audit["nodes"] if item["nodeId"] == "node-remote")
        self.assertTrue(remote_item["eligible"])
        self.assertTrue(remote_item["capabilityBacked"])
        self.assertTrue(remote_item["verifiedCapability"])

    def test_resolve_worker_execution_node_audit_rejects_unverified_capability_only_worker(
        self,
    ) -> None:
        observed_at = datetime.now(tz=UTC).isoformat()
        save_node_capabilities(
            [
                NodeCapabilityRecord(
                    node_id="node-remote",
                    source="peer",
                    source_url="http://85.137.164.250:52415",
                    last_seen_at=observed_at,
                    updated_at=observed_at,
                    api_urls=["http://26.97.29.153:52425"],
                    worker_enabled=True,
                    worker_reward_address="bbbb1234bbbb1234bbbb1234bbbb1234",
                    worker_allowed_model_ids=["public/test-model"],
                    worker_verified=False,
                )
            ]
        )

        with patch(
            "cai_compute_chain.jobs._get_json",
            return_value={
                "node-bootstrap": {
                    "apiHost": "85.137.164.250",
                    "apiPort": 52415,
                    "workerEnabled": False,
                    "workerRewardAddress": "validator-address",
                }
            },
        ):
            audit = _resolve_worker_execution_node_audit(
                "http://127.0.0.1:52425",
                "public/test-model",
            )

        assert audit is not None
        self.assertEqual(audit["eligibleNodeIds"], [])
        remote_item = next(item for item in audit["nodes"] if item["nodeId"] == "node-remote")
        self.assertFalse(remote_item["eligible"])
        self.assertFalse(remote_item["verifiedCapability"])
        self.assertIn("worker capability is not verified", remote_item["reasons"])

    def test_task_level_transport_instance_snapshot_uses_completed_shard_receipt_ranges(
        self,
    ) -> None:
        snapshot = _task_level_transport_instance_snapshot(
            instance_id="caitask_job-1",
            execution_model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            requester_node_id="node-user",
            executor_node_ids=["node-a", "node-b"],
            proof={
                "shardReceipts": [
                    {
                        "nodeId": "node-a",
                        "status": "completed",
                        "layerStart": 0,
                        "layerEnd": 14,
                    },
                    {
                        "nodeId": "node-b",
                        "status": "completed",
                        "layerStart": 14,
                        "layerEnd": 28,
                    },
                ]
            },
            dispatch_result={
                "participantNodeIds": ["node-user", "node-a", "node-b"],
            },
        )

        self.assertEqual(
            snapshot["participants"],
            [
                {
                    "node_id": "node-a",
                    "runner_id": "cai-task-http:node-a",
                    "layer_start": 0,
                    "layer_end": 14,
                    "layer_count": 14,
                },
                {
                    "node_id": "node-b",
                    "runner_id": "cai-task-http:node-b",
                    "layer_start": 14,
                    "layer_end": 28,
                    "layer_count": 14,
                },
            ],
        )
        self.assertEqual(
            snapshot["caiOwnedTaskLevelTransport"]["totalLayerCount"],
            28,
        )
        self.assertEqual(
            snapshot["caiOwnedTaskLevelTransport"]["participantRangeSource"],
            "shard_receipts",
        )

    def test_task_level_transport_instance_snapshot_falls_back_to_dispatch_dag_ranges(
        self,
    ) -> None:
        snapshot = _task_level_transport_instance_snapshot(
            instance_id="caitask_job-2",
            execution_model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            requester_node_id="node-user",
            executor_node_ids=["node-a", "node-b"],
            proof={},
            dispatch_result={
                "participantNodeIds": ["node-user", "node-a", "node-b"],
                "dag": {
                    "totalLayerCount": 24,
                    "stages": [
                        {
                            "executorNodeId": "node-a",
                            "layerStart": 0,
                            "layerEnd": 12,
                        },
                        {
                            "executorNodeId": "node-b",
                            "layerStart": 12,
                            "layerEnd": 24,
                        },
                        {
                            "executorNodeId": "node-a",
                            "layerStart": 0,
                            "layerEnd": 12,
                        },
                        {
                            "executorNodeId": "node-b",
                            "layerStart": 12,
                            "layerEnd": 24,
                        },
                    ],
                },
            },
        )

        self.assertEqual(
            snapshot["participants"],
            [
                {
                    "node_id": "node-a",
                    "runner_id": "cai-task-http:node-a",
                    "layer_start": 0,
                    "layer_end": 12,
                    "layer_count": 12,
                },
                {
                    "node_id": "node-b",
                    "runner_id": "cai-task-http:node-b",
                    "layer_start": 12,
                    "layer_end": 24,
                    "layer_count": 12,
                },
            ],
        )
        self.assertEqual(
            snapshot["caiOwnedTaskLevelTransport"]["totalLayerCount"],
            24,
        )
        self.assertEqual(
            snapshot["caiOwnedTaskLevelTransport"]["participantRangeSource"],
            "dispatch_dag",
        )

    def test_task_level_transport_instance_snapshot_uses_full_model_range_for_single_executor(
        self,
    ) -> None:
        snapshot = _task_level_transport_instance_snapshot(
            instance_id="caitask_job-3",
            execution_model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            requester_node_id="node-user",
            executor_node_ids=["node-a"],
            proof={},
            dispatch_result={
                "participantNodeIds": ["node-user", "node-a"],
            },
        )

        self.assertEqual(
            snapshot["participants"],
            [
                {
                    "node_id": "node-a",
                    "runner_id": "cai-task-http:node-a",
                    "layer_start": 0,
                    "layer_end": 24,
                    "layer_count": 24,
                }
            ],
        )
        self.assertEqual(
            snapshot["caiOwnedTaskLevelTransport"]["totalLayerCount"],
            24,
        )
        self.assertEqual(
            snapshot["caiOwnedTaskLevelTransport"]["participantRangeSource"],
            "single_executor_transport_fallback",
        )

    def test_task_level_transport_multi_worker_synthetic_snapshot_is_not_settleable(
        self,
    ) -> None:
        snapshot = _task_level_transport_instance_snapshot(
            instance_id="caitask_job-4",
            execution_model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            requester_node_id="node-user",
            executor_node_ids=["node-a", "node-b"],
            proof={},
            dispatch_result={
                "participantNodeIds": ["node-user", "node-a", "node-b"],
            },
        )

        self.assertEqual(
            snapshot["caiOwnedTaskLevelTransport"]["participantRangeSource"],
            "synthetic_executor_index_fallback",
        )
        with self.assertRaisesRegex(RuntimeError, "synthetic task-level shard snapshot"):
            _require_settleable_instance_snapshot(snapshot)

    def test_task_level_transport_total_layer_count_prefers_curated_model_layers(
        self,
    ) -> None:
        self.assertEqual(
            _task_level_transport_total_layer_count(
                "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                executor_count=2,
            ),
            24,
        )
        self.assertEqual(
            _task_level_transport_total_layer_count(
                "Qwen/Qwen3-0.6B-GGUF",
                executor_count=2,
            ),
            28,
        )

    def test_task_level_transport_total_layer_count_falls_back_to_executor_count(
        self,
    ) -> None:
        self.assertEqual(
            _task_level_transport_total_layer_count(
                "public/test-model",
                executor_count=2,
            ),
            2,
        )

    def test_task_level_transport_total_layer_count_uses_manifest_metadata(
        self,
    ) -> None:
        with patch(
            "cai_compute_chain.jobs.select_model_package_manifest_for_model",
            return_value=SimpleNamespace(metadata={"total_layers": 22}),
        ):
            self.assertEqual(
                _task_level_transport_total_layer_count(
                    "cai-network/TinyLlama-1.1B-Chat-v1.0-GGUF",
                    executor_count=2,
                ),
                22,
            )

    def test_task_level_transport_llm_runtime_metadata_uses_curated_gguf_policy(
        self,
    ) -> None:
        metadata = _task_level_transport_llm_runtime_metadata(
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            total_layer_count=24,
        )

        self.assertEqual(metadata["metadataSource"], "curated_model_policy")
        self.assertEqual(metadata["modelId"], "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
        self.assertEqual(metadata["totalLayerCount"], 24)
        self.assertEqual(metadata["hiddenSize"], 896)
        self.assertEqual(metadata["ggufArchitecture"], "qwen2")
        self.assertEqual(metadata["shardCompatibility"], "layer_range_supported")
        self.assertTrue(metadata["layerRangeSupported"])
        self.assertEqual(
            metadata["preferredFilename"],
            "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        )

    def test_task_level_transport_llm_runtime_metadata_omits_non_curated_model(
        self,
    ) -> None:
        self.assertIsNone(
            _task_level_transport_llm_runtime_metadata(
                "public/test-model",
                total_layer_count=2,
            )
        )

    def test_task_level_transport_llm_runtime_metadata_uses_manifest_policy(
        self,
    ) -> None:
        manifest = SimpleNamespace(
            model_id="Example/TinyLlama-GGUF",
            preferred_filename="tinyllama.gguf",
            metadata={
                "total_layers": 22,
                "hidden_size": 2048,
                "gguf_architecture": "llama",
                "shard_compatibility": "layer_range_supported",
                "layer_range_supported": True,
                "state_format": "ggml-tensor-v1/layer-range-activation-v1",
                "activation_state_format": "ggml-tensor-v1/layer-range-activation-v1",
                "decode_state_format": "ggml-kv-cache-v1/token-step-kv-cache-v1",
                "layer_range_probe_abi": "cai-layer-range-v1",
                "layer_range_probe_report": "docs/reports/llama-tinyllama-production-binary-conformance-2026-05-11.json",
                "layer_range_equivalence_probe_report": "docs/reports/llama-tinyllama-layer-range-equivalence-probe-2026-05-11.json",
            },
        )
        with patch(
            "cai_compute_chain.jobs.select_model_package_manifest_for_model",
            return_value=manifest,
        ):
            metadata = _task_level_transport_llm_runtime_metadata(
                "Example/TinyLlama-GGUF",
                total_layer_count=22,
            )

        self.assertEqual(metadata["metadataSource"], "model_package_manifest")
        self.assertEqual(metadata["modelId"], "Example/TinyLlama-GGUF")
        self.assertEqual(metadata["totalLayerCount"], 22)
        self.assertEqual(metadata["hiddenSize"], 2048)
        self.assertEqual(metadata["ggufArchitecture"], "llama")
        self.assertEqual(metadata["shardCompatibility"], "layer_range_supported")
        self.assertTrue(metadata["layerRangeSupported"])
        self.assertEqual(metadata["preferredFilename"], "tinyllama.gguf")

    def test_task_level_transport_llm_runtime_metadata_rejects_manifest_without_layer_range_compatibility(
        self,
    ) -> None:
        manifest = SimpleNamespace(
            model_id="Example/TinyLlama-GGUF",
            preferred_filename="tinyllama.gguf",
            metadata={
                "total_layers": 22,
                "hidden_size": 2048,
                "gguf_architecture": "llama",
                "shard_compatibility": "full_model_local",
                "layer_range_supported": True,
            },
        )
        with patch(
            "cai_compute_chain.jobs.select_model_package_manifest_for_model",
            return_value=manifest,
        ):
            self.assertIsNone(
                _task_level_transport_llm_runtime_metadata(
                    "Example/TinyLlama-GGUF",
                    total_layer_count=22,
                )
            )

    def test_task_level_transport_llm_runtime_metadata_falls_back_from_curated_to_manifest(
        self,
    ) -> None:
        curated_model = SimpleNamespace(
            layer_range_supported=True,
            hidden_size=None,
        )
        manifest = SimpleNamespace(
            model_id="Example/TinyLlama-GGUF",
            preferred_filename="tinyllama.gguf",
            metadata={
                "total_layers": 22,
                "hidden_size": 2048,
                "gguf_architecture": "llama",
                "shard_compatibility": "layer_range_supported",
                "layer_range_supported": True,
            },
        )
        with (
            patch("cai_compute_chain.jobs.curated_model_for_id", return_value=curated_model),
            patch(
                "cai_compute_chain.jobs.select_model_package_manifest_for_model",
                return_value=manifest,
            ),
        ):
            metadata = _task_level_transport_llm_runtime_metadata(
                "Example/TinyLlama-GGUF",
                total_layer_count=22,
            )

        self.assertEqual(metadata["metadataSource"], "model_package_manifest")
        self.assertEqual(metadata["hiddenSize"], 2048)

    def test_task_level_transport_effective_executor_count_keeps_multi_worker_for_proven_gguf(
        self,
    ) -> None:
        self.assertEqual(
            _task_level_transport_effective_executor_count(
                "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                requested_executor_count=2,
            ),
            2,
        )

    def test_task_level_transport_effective_executor_count_uses_manifest_proven_gguf(
        self,
    ) -> None:
        manifest = SimpleNamespace(
            preferred_filename="tinyllama.gguf",
            family="llama",
            package_kind="public_shared",
            metadata={
                "model_format": "gguf",
                "gguf_architecture": "llama",
                "shard_compatibility": "layer_range_supported",
                "layer_range_supported": True,
            },
        )
        with patch(
            "cai_compute_chain.jobs.select_model_package_manifest_for_model",
            return_value=manifest,
        ):
            self.assertEqual(
                _task_level_transport_effective_executor_count(
                    "Example/TinyLlama-GGUF",
                    requested_executor_count=2,
                ),
                2,
            )

    def test_task_level_transport_effective_executor_count_caps_to_total_layers(
        self,
    ) -> None:
        manifest = SimpleNamespace(
            preferred_filename="tinyllama.gguf",
            family="llama",
            package_kind="public_shared",
            metadata={
                "model_format": "gguf",
                "gguf_architecture": "llama",
                "shard_compatibility": "layer_range_supported",
                "layer_range_supported": True,
                "total_layers": 2,
            },
        )
        with patch(
            "cai_compute_chain.jobs.select_model_package_manifest_for_model",
            return_value=manifest,
        ):
            self.assertEqual(
                _task_level_transport_effective_executor_count(
                    "Example/TinyLlama-GGUF",
                    requested_executor_count=4,
                ),
                2,
            )

    def test_task_level_transport_planned_shard_ranges_use_placement_preview(
        self,
    ) -> None:
        placement_instance = {
            "MlxRingInstance": {
                "instanceId": "planned-task-level",
                "shardAssignments": {
                    "modelId": "Qwen/Qwen3-0.6B-GGUF",
                    "runnerToShard": {
                        "runner-a": {
                            "PipelineShardMetadata": {
                                "startLayer": 0,
                                "endLayer": 26,
                                "nLayers": 28,
                            }
                        },
                        "runner-b": {
                            "PipelineShardMetadata": {
                                "startLayer": 26,
                                "endLayer": 28,
                                "nLayers": 28,
                            }
                        },
                    },
                    "nodeToRunner": {
                        "node-a": "runner-a",
                        "node-b": "runner-b",
                    },
                },
            }
        }
        with patch(
            "cai_compute_chain.jobs._resolve_cai_instance_create_payload_for_nodes",
            return_value={"instance": placement_instance},
        ):
            executor_node_ids, shard_ranges = _task_level_transport_planned_shard_ranges(
                "http://127.0.0.1:52425",
                "Qwen/Qwen3-0.6B-GGUF",
                executor_node_ids=["node-b", "node-a"],
                total_layer_count=28,
            )

        self.assertEqual(executor_node_ids, ["node-a", "node-b"])
        self.assertEqual(
            shard_ranges,
            [
                {
                    "nodeId": "node-a",
                    "layerStart": 0,
                    "layerEnd": 26,
                    "layerCount": 26,
                },
                {
                    "nodeId": "node-b",
                    "layerStart": 26,
                    "layerEnd": 28,
                    "layerCount": 2,
                },
            ],
        )

    def test_task_level_transport_planned_shard_ranges_falls_back_without_full_coverage(
        self,
    ) -> None:
        placement_instance = {
            "MlxRingInstance": {
                "instanceId": "planned-task-level",
                "shardAssignments": {
                    "modelId": "Qwen/Qwen3-0.6B-GGUF",
                    "runnerToShard": {
                        "runner-a": {
                            "PipelineShardMetadata": {
                                "startLayer": 0,
                                "endLayer": 14,
                                "nLayers": 28,
                            }
                        },
                    },
                    "nodeToRunner": {
                        "node-a": "runner-a",
                    },
                },
            }
        }
        with patch(
            "cai_compute_chain.jobs._resolve_cai_instance_create_payload_for_nodes",
            return_value={"instance": placement_instance},
        ):
            executor_node_ids, shard_ranges = _task_level_transport_planned_shard_ranges(
                "http://127.0.0.1:52425",
                "Qwen/Qwen3-0.6B-GGUF",
                executor_node_ids=["node-a", "node-b"],
                total_layer_count=28,
            )

        self.assertEqual(executor_node_ids, ["node-a", "node-b"])
        self.assertIsNone(shard_ranges)

    def test_task_level_transport_effective_executor_count_downgrades_unproven_gguf(
        self,
    ) -> None:
        self.assertEqual(
            _task_level_transport_effective_executor_count(
                "Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF",
                requested_executor_count=2,
            ),
            1,
        )

    def test_task_level_transport_effective_executor_count_downgrades_manifest_full_model_local(
        self,
    ) -> None:
        manifest = SimpleNamespace(
            preferred_filename="mistral.gguf",
            family="mistral",
            package_kind="public_shared",
            metadata={
                "model_format": "gguf",
                "gguf_architecture": "mistral",
                "shard_compatibility": "full_model_local",
                "layer_range_supported": False,
            },
        )
        with patch(
            "cai_compute_chain.jobs.select_model_package_manifest_for_model",
            return_value=manifest,
        ):
            self.assertEqual(
                _task_level_transport_effective_executor_count(
                    "Example/Mistral-7B-GGUF",
                    requested_executor_count=2,
                ),
                1,
            )

    def test_task_level_transport_effective_executor_count_downgrades_non_gguf_model(
        self,
    ) -> None:
        self.assertEqual(
            _task_level_transport_effective_executor_count(
                "public/test-model",
                requested_executor_count=2,
            ),
            1,
        )

    def test_execute_job_intent_records_receipt_and_settlement(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("20000.00000000"))
        self._credit_wallet_on_chain(wallet, coins_to_atomic("20000.00000000"))
        set_validator_static_ip_confirmation(True)
        set_validator_mode(
            True,
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )
        job = create_job_intent(
            prompt="2+2=?",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
        )

        payload = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "4"},
                }
            ],
        }

        with (
            patch("cai_compute_chain.jobs.ensure_cai_instance"),
            patch(
                "cai_compute_chain.jobs.resolve_cai_instance_snapshot",
                return_value={
                    "instance_id": "instance-1",
                    "participants": [
                        {
                            "node_id": "node-a",
                            "runner_id": "runner-a",
                            "layer_start": 0,
                            "layer_end": 24,
                            "layer_count": 24,
                        },
                        {
                            "node_id": "node-b",
                            "runner_id": "runner-b",
                            "layer_start": 24,
                            "layer_end": 28,
                            "layer_count": 4,
                        },
                    ],
                    "caiOwnedTransportProof": self._create_verified_cai_owned_transport_proof(
                        "instance-1"
                    ),
                },
            ),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=self._distributed_two_worker_state(),
            ),
            patch("cai_compute_chain.jobs._submit_text_job_to_cai", return_value=payload),
        ):
            updated_job, receipt = execute_job_intent(job.job_id)

        self.assertEqual(updated_job.status, "completed")
        self.assertEqual(receipt.output_text, "4")
        self.assertEqual(receipt.instance_id, "instance-1")
        self.assertEqual(len(receipt.worker_payouts), 2)
        self.assertEqual(len(list_execution_receipts()), 1)
        self.assertEqual(len(list_settlements()), 1)
        self.assertEqual(len(list_attestations()), 1)
        self.assertEqual(len(list_worker_payouts()), 2)

    def test_execute_job_intent_keeps_receipt_when_settlement_tail_warns(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("20000.00000000"))
        self._credit_wallet_on_chain(wallet, coins_to_atomic("20000.00000000"))
        set_validator_static_ip_confirmation(True)
        set_validator_mode(
            True,
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )
        job = create_job_intent(
            prompt="5+23=",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
        )

        payload = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "28"},
                }
            ],
        }

        with (
            patch("cai_compute_chain.jobs.ensure_cai_instance"),
            patch(
                "cai_compute_chain.jobs.resolve_cai_instance_snapshot",
                return_value={
                    "instance_id": "instance-1",
                    "participants": [
                        {
                            "node_id": "node-a",
                            "runner_id": "runner-a",
                            "layer_start": 0,
                            "layer_end": 28,
                            "layer_count": 28,
                        }
                    ],
                },
            ),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=self._distributed_two_worker_state(),
            ),
            patch("cai_compute_chain.jobs._submit_text_job_to_cai", return_value=payload),
            patch("cai_compute_chain.jobs.request_remote_committee_attestations", return_value=[]),
            patch(
                "cai_compute_chain.jobs._apply_settlement_after_canonical_chain_sync",
                side_effect=json.JSONDecodeError("Expecting value", "", 0),
            ),
        ):
            updated_job, receipt = execute_job_intent(job.job_id)

        self.assertEqual(updated_job.status, "completed")
        self.assertEqual(updated_job.receipt_id, receipt.receipt_id)
        self.assertTrue(updated_job.settlement_id)
        self.assertEqual(receipt.output_text, "28")
        self.assertEqual(len(list_execution_receipts()), 1)
        self.assertEqual(len(list_settlements()), 1)
        self.assertTrue(
            any(
                entry.event_type == "execution_settlement_warning"
                for entry in list_journal_entries()
            )
        )

    def test_execute_job_intent_can_use_task_level_cai_owned_transport_path(
        self,
    ) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))
        self._credit_wallet_on_chain(wallet, coins_to_atomic("2.00000000"))
        worker_reward_address = "bbbb1234bbbb1234bbbb1234bbbb1234"
        model_id = "public/test-model"
        state_payload = {
            "nodeIdentities": {
                "node-user": {
                    "apiHost": "127.0.0.1",
                    "apiPort": 52425,
                },
                "node-b": {
                    "apiHost": "198.51.100.11",
                    "apiPort": 52425,
                    "workerEnabled": True,
                    "workerRewardAddress": worker_reward_address,
                    "workerAllowedModelIds": [model_id],
                },
            },
            "topology": {"nodes": ["node-user", "node-b"], "connections": {}},
        }
        job = create_job_intent(
            prompt="network answer?",
            compute_amount_coins="0.00100000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
            model_id=model_id,
            requester_node_id="node-user",
        )
        proof = build_cai_owned_transport_execution_proof(
            session_id="session-task-http",
            instance_id=f"caitask_{job.job_id}",
            participant_node_ids=["node-user", "node-b"],
            executor_node_ids=["node-b"],
            model_id=model_id,
            task_id=job.job_id,
            activation_batch_count=1,
            decode_batch_count=1,
            shard_receipts=[
                {
                    "nodeId": "node-b",
                    "network": "mainnet",
                    "chainId": "mainnet",
                    "status": "completed",
                    "activationBatchCount": 1,
                    "decodeBatchCount": 1,
                    "metrics": {
                        "promptTokenCount": 3,
                        "completionTokenCount": 2,
                        "inputTokenCount": 3,
                        "outputTokenCount": 2,
                        "totalTokenCount": 5,
                    },
                }
            ],
        )
        proof["executionAudit"] = {
            "verified": True,
            "sessionId": "session-task-http",
            "processedBatchIds": [],
            "finalOutputBatchIds": ["caibatch_final"],
            "blockedBatchIds": [],
            "receiptBatchIds": [],
            "hashChainSha256Hexes": [],
            "executionDag": None,
            "batchRecordCount": 2,
            "processedBatchCount": 2,
            "finalOutputBatchCount": 1,
            "errorCount": 0,
            "errors": [],
            "verifiedAt": "2026-05-04T00:00:00+00:00",
        }

        def fake_dispatch(**kwargs):
            self.assertEqual(kwargs["requester_node_id"], "node-user")
            self.assertEqual(kwargs["executor_node_ids"], ["node-b"])
            self.assertEqual(kwargs["initial_payload"], b"network answer?")
            self.assertFalse(kwargs["require_cai_owned_runtime_ready"])
            self.assertFalse(kwargs["require_executor_shard_readiness"])
            self.assertIsNone(kwargs["llm_runtime_metadata"])
            self.assertTrue(kwargs["single_executor_direct_final_output"])
            return {
                "status": "dispatched",
                "sessionId": "session-task-http",
                "instanceId": f"caitask_{job.job_id}",
                "requesterNodeId": "node-user",
                "executorNodeIds": ["node-b"],
                "participantNodeIds": ["node-user", "node-b"],
                "chainId": "mainnet",
            }

        final_result = {
            "status": "completed",
            "sessionId": "session-task-http",
            "finalOutput": {"payload": b"remote worker answer"},
            "proof": proof,
            "proofVerified": True,
        }

        with (
            patch.dict(
                os.environ,
                {
                    "CAI_ENABLE_TASK_LEVEL_TRANSPORT_JOBS": "1",
                    "CAI_REQUIRE_TASK_LEVEL_TRANSPORT_JOBS": "1",
                    "CAI_TASK_LEVEL_TRANSPORT_REQUIRE_RUNTIME_READY": "0",
                    "CAI_TASK_LEVEL_TRANSPORT_EXECUTOR_COUNT": "2",
                },
                clear=False,
            ),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=state_payload,
            ),
            patch(
                "cai_compute_chain.jobs._resolve_worker_execution_node_audit",
                return_value={
                    "schemaVersion": 1,
                    "modelId": model_id,
                    "checkedNodeCount": 1,
                    "eligibleNodeIds": ["node-b"],
                    "nodes": [],
                },
            ),
            patch(
                "cai_compute_chain.jobs.dispatch_cai_owned_transport_execution_dag",
                side_effect=fake_dispatch,
            ) as dispatch,
            patch(
                "cai_compute_chain.jobs.await_cai_owned_transport_session_final_result",
                return_value=final_result,
            ) as await_final,
            patch(
                "cai_compute_chain.jobs.ensure_cai_instance",
                side_effect=AssertionError("standard instance path should not run"),
            ),
            patch(
                "cai_compute_chain.jobs._submit_text_job_to_cai",
                side_effect=AssertionError("standard chat path should not run"),
            ),
            patch("cai_compute_chain.jobs.sync_validator_set_from_cai_peers"),
            patch("cai_compute_chain.settlement.sync_validator_set_from_cai_peers"),
            patch("cai_compute_chain.jobs.sync_chain_from_cai_peers"),
            patch("cai_compute_chain.jobs.apply_local_validator_attestation", return_value=None),
            patch("cai_compute_chain.jobs.request_remote_committee_attestations", return_value=[]),
            patch("cai_compute_chain.jobs.push_chain_to_cai_peers", return_value=[]),
        ):
            updated_job, receipt = execute_job_intent(job.job_id)

        dispatch.assert_called_once()
        await_final.assert_called_once_with(
            "session-task-http",
            requester_node_id="node-user",
            timeout_sec=1800.0,
            poll_interval_sec=0.25,
            policy=None,
        )
        self.assertEqual(updated_job.status, "completed")
        self.assertEqual(receipt.output_text, "remote worker answer")
        self.assertEqual(receipt.response_id, "caiot_session-task-http")
        self.assertEqual(receipt.prompt_tokens, 3)
        self.assertEqual(receipt.completion_tokens, 2)
        self.assertEqual(receipt.total_tokens, 5)
        self.assertEqual(receipt.token_usage_source, "cai_owned_transport_proof")
        self.assertTrue(receipt.token_usage_audit["proof_matches_response_usage"])
        self.assertEqual(
            receipt.raw_response["usage"],
            {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        )
        self.assertEqual(
            receipt.raw_response["caiOwnedTransport"]["usage"],
            receipt.raw_response["usage"],
        )
        self.assertEqual(receipt.network_audit["transportMode"], "single_worker")
        self.assertTrue(receipt.network_audit["singleWorkerRemote"])
        self.assertTrue(receipt.network_audit["caiOwnedTransportExecuted"])
        self.assertEqual(receipt.network_audit["participantNodeIds"], ["node-b"])
        self.assertEqual(len(receipt.worker_payouts), 1)
        self.assertEqual(receipt.worker_payouts[0]["node_id"], "node-b")
        self.assertEqual(list_worker_payouts()[0].recipient_address, worker_reward_address)

    def test_execute_job_intent_uses_execution_state_for_task_level_routes(
        self,
    ) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))
        self._credit_wallet_on_chain(wallet, coins_to_atomic("2.00000000"))
        model_id = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
        worker_a_address = "aaaa1234aaaa1234aaaa1234aaaa1234"
        worker_b_address = "bbbb1234bbbb1234bbbb1234bbbb1234"
        requester_state = {
            "nodeIdentities": {
                "node-user": {
                    "apiHost": "127.0.0.1",
                    "apiPort": 52455,
                },
            },
            "topology": {"nodes": ["node-user"], "connections": {}},
        }
        execution_state = {
            "nodeIdentities": {
                "node-vps": {
                    "apiHost": "85.137.164.250",
                    "apiPort": 52415,
                    "relayEnabled": True,
                },
                "node-user": {
                    "apiHost": "127.0.0.1",
                    "apiPort": 52455,
                },
                "node-a": {
                    "apiHost": "127.0.0.1",
                    "apiPort": 52445,
                    "workerEnabled": True,
                    "workerRewardAddress": worker_a_address,
                    "workerAllowedModelIds": [model_id],
                },
                "node-b": {
                    "apiHost": "26.97.29.153",
                    "apiPort": 52425,
                    "workerEnabled": True,
                    "workerRewardAddress": worker_b_address,
                    "workerAllowedModelIds": [model_id],
                },
            },
            "topology": {
                "nodes": ["node-vps", "node-user", "node-a", "node-b"],
                "connections": {
                    "node-a": {
                        "node-b": [
                            {"sinkMultiaddr": {"Ip4": ["26.97.29.153", "Tcp", 52425]}}
                        ]
                    },
                    "node-b": {
                        "node-a": [
                            {"sinkMultiaddr": {"Ip4": ["127.0.0.1", "Tcp", 52445]}}
                        ]
                    },
                },
            },
        }
        requester_url = "http://127.0.0.1:52455"
        execution_url = "http://85.137.164.250:52415"
        job = create_job_intent(
            prompt="network answer?",
            compute_amount_coins="0.00100000",
            payment_preference=PaymentPreference.AUTO,
            cai_url=requester_url,
            execution_cai_url=execution_url,
            model_id=model_id,
            requester_node_id="node-user",
        )
        proof = build_cai_owned_transport_execution_proof(
            session_id="session-task-execution-state",
            instance_id=f"caitask_{job.job_id}",
            participant_node_ids=["node-user", "node-a", "node-b"],
            executor_node_ids=["node-a", "node-b"],
            model_id=model_id,
            task_id=job.job_id,
            activation_batch_count=2,
            decode_batch_count=2,
            shard_receipts=[
                {
                    "nodeId": "node-a",
                    "network": "mainnet",
                    "chainId": "mainnet",
                    "status": "completed",
                    "activationBatchCount": 1,
                    "decodeBatchCount": 1,
                    "metrics": {
                        "promptTokenCount": 4,
                        "completionTokenCount": 1,
                        "inputTokenCount": 4,
                        "outputTokenCount": 1,
                        "totalTokenCount": 5,
                    },
                },
                {
                    "nodeId": "node-b",
                    "network": "mainnet",
                    "chainId": "mainnet",
                    "status": "completed",
                    "activationBatchCount": 1,
                    "decodeBatchCount": 1,
                    "metrics": {
                        "promptTokenCount": 0,
                        "completionTokenCount": 2,
                        "inputTokenCount": 0,
                        "outputTokenCount": 2,
                        "totalTokenCount": 2,
                    },
                },
            ],
        )
        proof["executionAudit"] = {
            "verified": True,
            "sessionId": "session-task-execution-state",
            "processedBatchIds": [],
            "finalOutputBatchIds": ["caibatch_final"],
            "blockedBatchIds": [],
            "receiptBatchIds": [],
            "hashChainSha256Hexes": [],
            "executionDag": None,
            "batchRecordCount": 3,
            "processedBatchCount": 3,
            "finalOutputBatchCount": 1,
            "errorCount": 0,
            "errors": [],
            "verifiedAt": "2026-05-04T00:00:00+00:00",
        }

        def fake_state(url: str, **_kwargs) -> dict:
            if url.rstrip("/") == requester_url:
                return requester_state
            if url.rstrip("/") == execution_url:
                return execution_state
            raise AssertionError(f"unexpected CAI state URL: {url}")

        def fake_worker_audit(cai_url: str, model_id_arg: str) -> dict:
            self.assertEqual(cai_url, execution_url)
            self.assertEqual(model_id_arg, model_id)
            return {
                "schemaVersion": 1,
                "modelId": model_id,
                "checkedNodeCount": 2,
                "eligibleNodeIds": ["node-a", "node-b"],
                "nodes": [],
            }

        placement_instance = {
            "MlxRingInstance": {
                "instanceId": "planned-task-http-qwen2.5",
                "shardAssignments": {
                    "modelId": model_id,
                    "runnerToShard": {
                        "runner-a": {
                            "PipelineShardMetadata": {
                                "startLayer": 0,
                                "endLayer": 20,
                                "nLayers": 24,
                            }
                        },
                        "runner-b": {
                            "PipelineShardMetadata": {
                                "startLayer": 20,
                                "endLayer": 24,
                                "nLayers": 24,
                            }
                        },
                    },
                    "nodeToRunner": {
                        "node-a": "runner-a",
                        "node-b": "runner-b",
                    },
                },
            }
        }

        def fake_dispatch(**kwargs):
            peer_urls = kwargs["peer_cai_urls_by_node"]
            self.assertEqual(kwargs["requester_node_id"], "node-user")
            self.assertEqual(kwargs["executor_node_ids"], ["node-a", "node-b"])
            self.assertEqual(kwargs["initial_payload"], b"network answer?")
            self.assertEqual(kwargs["total_layer_count"], 24)
            self.assertEqual(
                kwargs["shard_ranges"],
                [
                    {
                        "nodeId": "node-a",
                        "layerStart": 0,
                        "layerEnd": 20,
                        "layerCount": 20,
                    },
                    {
                        "nodeId": "node-b",
                        "layerStart": 20,
                        "layerEnd": 24,
                        "layerCount": 4,
                    },
                ],
            )
            self.assertEqual(
                kwargs["llm_runtime_metadata"],
                {
                    "metadataSource": "curated_model_policy",
                    "modelId": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                    "totalLayerCount": 24,
                    "totalLayers": 24,
                    "blockCount": 24,
                    "hiddenSize": 896,
                    "nEmbd": 896,
                    "ggufArchitecture": "qwen2",
                    "shardCompatibility": "layer_range_supported",
                    "layerRangeSupported": True,
                    "stateFormat": "ggml-tensor-v1/layer-range-activation-v1",
                    "activationStateFormat": "ggml-tensor-v1/layer-range-activation-v1",
                    "decodeStateFormat": "ggml-kv-cache-v1/token-step-kv-cache-v1",
                    "preferredFilename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
                    "layerRangeProbeAbi": "cai-layer-range-v1",
                    "layerRangeProbeReport": "docs/reports/qwen2.5-production-binary-conformance-2026-05-11.json",
                    "layerRangeEquivalenceProbeReport": "docs/reports/qwen2.5-layer-range-equivalence-probe-2026-05-11.json",
                },
            )
            self.assertIn("http://127.0.0.1:52445", peer_urls["node-a"])
            self.assertIn("http://26.97.29.153:52425", peer_urls["node-b"])
            self.assertFalse(kwargs["single_executor_direct_final_output"])
            self.assertTrue(kwargs["require_executor_shard_readiness"])
            return {
                "status": "dispatched",
                "sessionId": "session-task-execution-state",
                "instanceId": f"caitask_{job.job_id}",
                "requesterNodeId": "node-user",
                "executorNodeIds": ["node-a", "node-b"],
                "participantNodeIds": ["node-user", "node-a", "node-b"],
                "chainId": "mainnet",
            }

        final_result = {
            "status": "completed",
            "sessionId": "session-task-execution-state",
            "finalOutput": {"payload": b"execution state answer"},
            "proof": proof,
            "proofVerified": True,
        }

        with (
            patch.dict(
                os.environ,
                {
                    "CAI_ENABLE_TASK_LEVEL_TRANSPORT_JOBS": "1",
                    "CAI_REQUIRE_TASK_LEVEL_TRANSPORT_JOBS": "1",
                    "CAI_TASK_LEVEL_TRANSPORT_REQUIRE_RUNTIME_READY": "0",
                    "CAI_TASK_LEVEL_TRANSPORT_EXECUTOR_COUNT": "2",
                },
                clear=False,
            ),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                side_effect=fake_state,
            ),
            patch(
                "cai_compute_chain.jobs._resolve_worker_execution_node_audit",
                side_effect=fake_worker_audit,
            ),
            patch(
                "cai_compute_chain.jobs._resolve_cai_instance_create_payload_for_nodes",
                return_value={"instance": placement_instance},
            ),
            patch(
                "cai_compute_chain.jobs.dispatch_cai_owned_transport_execution_dag",
                side_effect=fake_dispatch,
            ) as dispatch,
            patch(
                "cai_compute_chain.jobs.list_route_health_records",
                return_value=[
                    RouteHealthRecord(
                        route_id="node-a->node-b",
                        source_node_id="node-a",
                        sink_node_id="node-b",
                        route_type="llama_cpp_rpc_direct",
                        endpoint_url="llama-cpp-rpc://26.97.29.153:52435",
                        reachable=True,
                        checked_at="2026-05-03T00:00:00+00:00",
                        latency_ms=7.0,
                    )
                ],
            ),
            patch(
                "cai_compute_chain.jobs.await_cai_owned_transport_session_final_result",
                return_value=final_result,
            ),
            patch(
                "cai_compute_chain.jobs.ensure_cai_instance",
                side_effect=AssertionError("standard instance path should not run"),
            ),
            patch(
                "cai_compute_chain.jobs._submit_text_job_to_cai",
                side_effect=AssertionError("standard chat path should not run"),
            ),
            patch("cai_compute_chain.jobs.sync_validator_set_from_cai_peers"),
            patch("cai_compute_chain.settlement.sync_validator_set_from_cai_peers"),
            patch("cai_compute_chain.jobs.sync_chain_from_cai_peers"),
            patch("cai_compute_chain.jobs.apply_local_validator_attestation", return_value=None),
            patch("cai_compute_chain.jobs.request_remote_committee_attestations", return_value=[]),
            patch("cai_compute_chain.jobs.push_chain_to_cai_peers", return_value=[]),
        ):
            updated_job, receipt = execute_job_intent(job.job_id)

        dispatch.assert_called_once()
        self.assertEqual(updated_job.status, "completed")
        self.assertEqual(receipt.output_text, "execution state answer")
        self.assertEqual(receipt.network_audit["participantNodeIds"], ["node-a", "node-b"])
        self.assertEqual(len(receipt.worker_payouts), 2)
        self.assertEqual(
            {payout["node_id"] for payout in receipt.worker_payouts},
            {"node-a", "node-b"},
        )

    def test_execute_job_intent_required_task_level_transport_fails_without_executor(
        self,
    ) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))
        self._credit_wallet_on_chain(wallet, coins_to_atomic("2.00000000"))
        job = create_job_intent(
            prompt="network answer?",
            compute_amount_coins="0.00100000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
            model_id="public/test-model",
            requester_node_id="node-user",
        )

        with (
            patch.dict(
                os.environ,
                {
                    "CAI_ENABLE_TASK_LEVEL_TRANSPORT_JOBS": "1",
                    "CAI_REQUIRE_TASK_LEVEL_TRANSPORT_JOBS": "1",
                },
                clear=False,
            ),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value={
                    "nodeIdentities": {
                        "node-user": {
                            "apiHost": "127.0.0.1",
                            "apiPort": 52425,
                        }
                    }
                },
            ),
            patch(
                "cai_compute_chain.jobs._resolve_worker_execution_node_audit",
                return_value={
                    "schemaVersion": 1,
                    "modelId": "public/test-model",
                    "checkedNodeCount": 0,
                    "eligibleNodeIds": [],
                    "nodes": [],
                },
            ),
            patch(
                "cai_compute_chain.jobs.ensure_cai_instance",
                side_effect=AssertionError("standard instance path should not run"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "task-level transport"):
                execute_job_intent(job.job_id)

        persisted_job = next(item for item in list_job_intents() if item.job_id == job.job_id)
        self.assertEqual(persisted_job.status, "failed")
        self.assertEqual(list_execution_receipts(), [])
        self.assertEqual(list_worker_payouts(), [])

    def test_execute_job_intent_falls_back_to_proven_path_when_task_level_is_not_required(
        self,
    ) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))
        self._credit_wallet_on_chain(wallet, coins_to_atomic("2.00000000"))
        job = create_job_intent(
            prompt="2+2=?",
            compute_amount_coins="0.00100000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            requester_node_id="node-user",
        )
        payload = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "4"},
                }
            ],
        }

        with (
            patch.dict(
                os.environ,
                {
                    "CAI_ENABLE_TASK_LEVEL_TRANSPORT_JOBS": "1",
                    "CAI_REQUIRE_TASK_LEVEL_TRANSPORT_JOBS": "0",
                },
                clear=False,
            ),
            patch(
                "cai_compute_chain.jobs._try_execute_task_level_transport_job",
                side_effect=RuntimeError("task-level path is not ready"),
            ) as task_level,
            patch("cai_compute_chain.jobs.ensure_cai_instance"),
            patch(
                "cai_compute_chain.jobs.resolve_cai_instance_snapshot",
                return_value={
                    "instance_id": "instance-1",
                    "participants": [
                        {
                            "node_id": "node-a",
                            "runner_id": "runner-a",
                            "layer_start": 0,
                            "layer_end": 24,
                            "layer_count": 24,
                        },
                        {
                            "node_id": "node-b",
                            "runner_id": "runner-b",
                            "layer_start": 24,
                            "layer_end": 28,
                            "layer_count": 4,
                        },
                    ],
                },
            ),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=self._distributed_two_worker_state(),
            ),
            patch(
                "cai_compute_chain.jobs._submit_text_job_to_cai",
                return_value=payload,
            ) as submit_text_job,
            patch(
                "cai_compute_chain.jobs._build_participant_eligibility_audit",
                return_value={
                    "canSettle": True,
                    "fatalReasons": [],
                    "warnings": [],
                },
            ),
            patch("cai_compute_chain.jobs.sync_validator_set_from_cai_peers"),
            patch("cai_compute_chain.settlement.sync_validator_set_from_cai_peers"),
            patch("cai_compute_chain.jobs.sync_chain_from_cai_peers"),
            patch("cai_compute_chain.jobs.apply_local_validator_attestation", return_value=None),
            patch("cai_compute_chain.jobs.request_remote_committee_attestations", return_value=[]),
            patch("cai_compute_chain.jobs.push_chain_to_cai_peers", return_value=[]),
        ):
            updated_job, receipt = execute_job_intent(job.job_id)

        task_level.assert_called_once()
        submit_text_job.assert_called_once()
        self.assertEqual(updated_job.status, "completed")
        self.assertEqual(receipt.output_text, "4")
        self.assertFalse(bool(receipt.network_audit.get("caiOwnedTransportExecuted")))
        self.assertEqual(receipt.network_audit["transportMode"], "multi_worker_direct")

    def test_task_level_executor_selection_keeps_required_two_pc_count(self) -> None:
        with patch.dict(
            os.environ,
            {"CAI_TASK_LEVEL_TRANSPORT_EXECUTOR_COUNT": "2"},
            clear=False,
        ):
            selected = _select_task_level_transport_executor_node_ids(
                [["node-remote"], ["node-local", "node-remote"]],
                peer_cai_urls_by_node={
                    "node-local": ["http://127.0.0.1:52425"],
                    "node-remote": [
                        "cai-overlay:http://validator:52415?targetNodeId=node-remote"
                    ],
                },
                requester_node_id="node-local",
            )

        self.assertEqual(selected, ["node-local", "node-remote"])

    def test_task_level_executor_selection_allows_partial_two_pc_count(self) -> None:
        with patch.dict(
            os.environ,
            {"CAI_TASK_LEVEL_TRANSPORT_EXECUTOR_COUNT": "2"},
            clear=False,
        ):
            selected = _select_task_level_transport_executor_node_ids(
                [["node-remote"]],
                peer_cai_urls_by_node={
                    "node-remote": [
                        "cai-overlay:http://validator:52415?targetNodeId=node-remote"
                    ],
                },
                requester_node_id="node-local",
            )

        self.assertEqual(selected, ["node-remote"])

    def test_task_level_executor_selection_prefers_low_latency_healthy_route(self) -> None:
        route_health_records = [
            RouteHealthRecord(
                route_id="local-to-slow",
                source_node_id="node-local",
                sink_node_id="node-slow",
                route_type="llama_cpp_rpc_direct",
                endpoint_url="llama-cpp-rpc://198.51.100.21:52435",
                reachable=True,
                checked_at="2026-05-03T00:00:00+00:00",
                latency_ms=75.0,
            ),
            RouteHealthRecord(
                route_id="local-to-fast",
                source_node_id="node-local",
                sink_node_id="node-fast",
                route_type="llama_cpp_rpc_direct",
                endpoint_url="llama-cpp-rpc://198.51.100.22:52435",
                reachable=True,
                checked_at="2026-05-03T00:00:01+00:00",
                latency_ms=6.0,
            ),
        ]

        with patch.dict(
            os.environ,
            {"CAI_TASK_LEVEL_TRANSPORT_EXECUTOR_COUNT": "2"},
            clear=False,
        ):
            selected = _select_task_level_transport_executor_node_ids(
                [["node-slow", "node-unproven", "node-fast"]],
                peer_cai_urls_by_node={
                    "node-slow": [
                        "cai-overlay:http://validator:52415?targetNodeId=node-slow"
                    ],
                    "node-unproven": [
                        "cai-overlay:http://validator:52415?targetNodeId=node-unproven"
                    ],
                    "node-fast": [
                        "cai-overlay:http://validator:52415?targetNodeId=node-fast"
                    ],
                },
                requester_node_id="node-local",
                route_health_records=route_health_records,
            )

        self.assertEqual(selected, ["node-fast", "node-slow"])

    def test_task_level_executor_selection_uses_performance_history(self) -> None:
        record_execution_attempt_performance(
            model_id="Qwen/Qwen3-0.6B-GGUF",
            requester_node_id="node-local",
            executor_node_ids=["node-slow"],
            status="completed",
            attempt_duration_ms=1200,
            readiness_duration_ms=200,
            response_duration_ms=1000,
        )
        record_execution_attempt_performance(
            model_id="Qwen/Qwen3-0.6B-GGUF",
            requester_node_id="node-local",
            executor_node_ids=["node-fast"],
            status="completed",
            attempt_duration_ms=180,
            readiness_duration_ms=80,
            response_duration_ms=100,
        )

        with patch.dict(
            os.environ,
            {"CAI_TASK_LEVEL_TRANSPORT_EXECUTOR_COUNT": "1"},
            clear=False,
        ):
            selected = _select_task_level_transport_executor_node_ids(
                [["node-slow", "node-fast"]],
                peer_cai_urls_by_node={
                    "node-slow": [
                        "cai-overlay:http://validator:52415?targetNodeId=node-slow"
                    ],
                    "node-fast": [
                        "cai-overlay:http://validator:52415?targetNodeId=node-fast"
                    ],
                },
                requester_node_id="node-local",
                model_id="Qwen/Qwen3-0.6B-GGUF",
                performance_records=list_execution_performance_records(),
            )

        self.assertEqual(selected, ["node-fast"])

    def test_task_level_executor_fallback_attempts_try_local_single_executor(self) -> None:
        attempts = _task_level_transport_executor_fallback_attempts(
            ["node-remote", "node-local"],
            requester_node_id="node-local",
        )

        self.assertEqual(
            attempts,
            [["node-remote", "node-local"], ["node-local"], ["node-remote"]],
        )

    def test_task_level_transport_initial_prompt_text_prefers_latest_user_message(self) -> None:
        payload = {
            "messages": [
                {"role": "system", "content": "Answer directly."},
                {"role": "user", "content": "2+3="},
            ],
            "prompt": "raw fallback prompt",
        }

        self.assertEqual(
            _task_level_transport_initial_prompt_text(
                payload,
                fallback_prompt="job fallback",
            ),
            "2+3=",
        )

    def test_task_level_transport_initial_prompt_text_reads_content_segments(self) -> None:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "first"},
                        {"type": "text", "text": "second"},
                    ],
                },
            ],
        }

        self.assertEqual(
            _task_level_transport_initial_prompt_text(
                payload,
                fallback_prompt="job fallback",
            ),
            "first\nsecond",
        )

    def test_standard_placement_attempts_use_performance_history(self) -> None:
        record_execution_attempt_performance(
            model_id="Qwen/Qwen3-0.6B-GGUF",
            requester_node_id="node-local",
            executor_node_ids=["node-a-slow"],
            status="completed",
            attempt_duration_ms=1200,
            readiness_duration_ms=200,
            response_duration_ms=1000,
        )
        record_execution_attempt_performance(
            model_id="Qwen/Qwen3-0.6B-GGUF",
            requester_node_id="node-local",
            executor_node_ids=["node-z-fast"],
            status="completed",
            attempt_duration_ms=180,
            readiness_duration_ms=80,
            response_duration_ms=100,
        )

        with (
            patch(
                "cai_compute_chain.jobs._resolve_worker_execution_node_audit",
                return_value={
                    "eligibleNodeIds": ["node-a-slow", "node-z-fast"],
                    "checkedNodeCount": 2,
                },
            ),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value={
                    "nodeIdentities": {
                        "node-local": {"apiHost": "127.0.0.1", "apiPort": 52425}
                    }
                },
            ),
            patch(
                "cai_compute_chain.jobs._resolve_cai_instance_create_payload_for_nodes",
                return_value={"instance": {"MlxRingInstance": {}}},
            ) as resolve_for_nodes,
        ):
            payload = _resolve_cai_instance_create_payload(
                "http://127.0.0.1:52425",
                "Qwen/Qwen3-0.6B-GGUF",
            )

        self.assertEqual(payload, {"instance": {"MlxRingInstance": {}}})
        self.assertEqual(
            resolve_for_nodes.call_args.kwargs["node_ids"],
            ["node-z-fast", "node-a-slow"],
        )

    def test_standard_preview_selection_uses_performance_history(self) -> None:
        record_execution_attempt_performance(
            model_id="Qwen/Qwen3-0.6B-GGUF",
            requester_node_id="node-local",
            executor_node_ids=["node-a-slow"],
            status="completed",
            attempt_duration_ms=1400,
            readiness_duration_ms=300,
            response_duration_ms=1100,
        )
        record_execution_attempt_performance(
            model_id="Qwen/Qwen3-0.6B-GGUF",
            requester_node_id="node-local",
            executor_node_ids=["node-z-fast"],
            status="completed",
            attempt_duration_ms=220,
            readiness_duration_ms=90,
            response_duration_ms=130,
        )

        previews_payload = {
            "previews": [
                {
                    "instance": {
                        "MlxRingInstance": {
                            "shardAssignments": {
                                "nodeToRunner": {"node-a-slow": "runner-slow"}
                            }
                        }
                    }
                },
                {
                    "instance": {
                        "MlxRingInstance": {
                            "shardAssignments": {
                                "nodeToRunner": {"node-z-fast": "runner-fast"}
                            }
                        }
                    }
                },
            ]
        }

        with patch("cai_compute_chain.jobs._get_json", return_value=previews_payload):
            payload = _resolve_cai_instance_create_payload_for_nodes(
                "http://127.0.0.1:52425",
                "Qwen/Qwen3-0.6B-GGUF",
                node_ids=["node-a-slow", "node-z-fast"],
                private_network_model=False,
                cluster_node_count=2,
                prefer_multi_node=False,
                requester_node_id="node-local",
                performance_records=list_execution_performance_records(),
            )

        node_to_runner = payload["instance"]["MlxRingInstance"]["shardAssignments"][
            "nodeToRunner"
        ]
        self.assertEqual(node_to_runner, {"node-z-fast": "runner-fast"})

    def test_execute_job_intent_e2e_reward_integrity_for_verified_cai_owned_transport(
        self,
    ) -> None:
        worker_a_address = "ABCD1234ABCD1234ABCD1234ABCD1234"
        worker_b_address = "BCDE1234BCDE1234BCDE1234BCDE1234"
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("20000.00000000"))
        self._credit_wallet_on_chain(wallet, coins_to_atomic("20000.00000000"))
        set_validator_static_ip_confirmation(True)
        set_validator_mode(
            True,
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )
        job = create_job_intent(
            prompt="reward integrity",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
        )
        state_payload = self._distributed_two_worker_state()
        validator_state = self._public_validator_state()
        state_payload["nodeIdentities"].update(validator_state["nodeIdentities"])
        state_payload["nodeNetwork"] = validator_state["nodeNetwork"]
        state_payload["nodeIdentities"]["node-a"]["apiPort"] = 52426
        state_payload["nodeIdentities"]["node-b"]["apiPort"] = 52427
        state_payload["nodeIdentities"]["node-relay"]["apiPort"] = 52428
        state_payload["nodeIdentities"]["node-a"].update(
            {
                "workerEnabled": True,
                "workerRewardAddress": worker_a_address,
                "workerAllowedModelIds": ["cai-network/Qwen3-0.6B-GGUF"],
            }
        )
        state_payload["nodeIdentities"]["node-b"].update(
            {
                "workerEnabled": True,
                "workerRewardAddress": worker_b_address,
                "workerAllowedModelIds": ["cai-network/Qwen3-0.6B-GGUF"],
            }
        )
        instance_snapshot = {
            "instance_id": "instance-reward-integrity",
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
            "caiOwnedTransportProof": self._create_verified_cai_owned_transport_proof(
                "instance-reward-integrity"
            ),
        }
        payload = {
            "id": "chatcmpl-reward-integrity",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
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
                return_value=state_payload,
            ),
            patch(
                "cai_compute_chain.jobs.worker_capability_verification_required",
                return_value=False,
            ),
            patch("cai_compute_chain.jobs.sync_validator_set_from_cai_peers"),
            patch("cai_compute_chain.settlement.sync_validator_set_from_cai_peers"),
            patch("cai_compute_chain.jobs.sync_chain_from_cai_peers"),
            patch("cai_compute_chain.jobs._submit_text_job_to_cai", return_value=payload),
            patch(
                "cai_compute_chain.jobs.request_remote_committee_attestations",
                return_value=[],
            ),
            patch("cai_compute_chain.jobs.push_chain_to_cai_peers", return_value=[]),
        ):
            updated_job, receipt = execute_job_intent(job.job_id)

        settlements = list_settlements()
        payouts = sorted(list_worker_payouts(), key=lambda item: item.node_id)
        settlement = settlements[0]
        payout_reward_sum = sum(item.reward_atomic for item in payouts)
        settlement_history = chain_settlement_history(settlement.settlement_id)
        reward_history = [
            item
            for item in settlement_history
            if item["tx_type"] == "worker_reward_credit"
        ]

        self.assertEqual(updated_job.status, "completed")
        self.assertEqual(receipt.output_text, "ok")
        self.assertTrue(receipt.network_audit["caiOwnedTransportExecuted"])
        self.assertTrue(
            receipt.network_audit["caiOwnedTransportExecutionProof"]["executionAudit"][
                "verified"
            ]
        )
        self.assertEqual(
            settlement.balance_audit["execution"]["receipt_id"],
            receipt.receipt_id,
        )
        self.assertEqual(settlement.status, "applied")
        self.assertEqual(len(payouts), 2)
        self.assertEqual([item.node_id for item in payouts], ["node-a", "node-b"])
        self.assertEqual(payout_reward_sum, settlement.worker_reward_atomic)
        self.assertEqual(payouts[0].reward_atomic, payouts[1].reward_atomic)
        self.assertEqual(payouts[0].recipient_address, worker_a_address.lower())
        self.assertEqual(payouts[1].recipient_address, worker_b_address.lower())
        self.assertEqual(len(reward_history), 2)
        self.assertEqual(
            sum(int(item["delta_atomic"]) for item in reward_history),
            settlement.worker_reward_atomic,
        )
        self.assertEqual(
            chain_balance_atomic(worker_a_address),
            payouts[0].reward_atomic,
        )
        self.assertEqual(
            chain_balance_atomic(worker_b_address),
            payouts[1].reward_atomic,
        )
        payout_audit = settlement.balance_audit["payouts"]
        self.assertTrue(payout_audit["worker_reward_matches_payouts"])
        self.assertEqual(
            payout_audit["recorded_worker_reward_atomic"],
            settlement.worker_reward_atomic,
        )

    def test_execute_job_intent_rewards_only_transport_shard_receipt_nodes(
        self,
    ) -> None:
        worker_a_address = "ABCD1234ABCD1234ABCD1234ABCD1234"
        worker_b_address = "BCDE1234BCDE1234BCDE1234BCDE1234"
        relay_address = "CDEF1234CDEF1234CDEF1234CDEF1234"
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))
        self._credit_wallet_on_chain(wallet, coins_to_atomic("2.00000000"))
        model_id = "cai-network/Qwen3-0.6B-GGUF"
        job = create_job_intent(
            prompt="receipt-backed payouts",
            compute_amount_coins="0.00100000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
            model_id=model_id,
        )
        state_payload = self._distributed_two_worker_state()
        for node_id, address in {
            "node-a": worker_a_address,
            "node-b": worker_b_address,
            "node-relay": relay_address,
        }.items():
            state_payload["nodeIdentities"][node_id].update(
                {
                    "workerEnabled": True,
                    "workerRewardAddress": address,
                    "workerAllowedModelIds": [model_id],
                }
            )
        proof = build_cai_owned_transport_execution_proof(
            session_id="session-receipt-backed-payouts",
            instance_id="instance-receipt-backed-payouts",
            participant_node_ids=["node-a", "node-b", "node-relay"],
            executor_node_ids=["node-a", "node-b"],
            model_id=model_id,
            task_id=job.job_id,
            shard_receipts=[
                {
                    "nodeId": "node-a",
                    "network": "mainnet",
                    "chainId": "mainnet",
                    "status": "completed",
                    "activationBatchCount": 1,
                    "decodeBatchCount": 1,
                    "layerStart": 0,
                    "layerEnd": 14,
                },
                {
                    "nodeId": "node-b",
                    "network": "mainnet",
                    "chainId": "mainnet",
                    "status": "completed",
                    "activationBatchCount": 1,
                    "decodeBatchCount": 1,
                    "layerStart": 14,
                    "layerEnd": 28,
                },
            ],
        )
        proof["executionAudit"] = {
            "verified": True,
            "sessionId": "session-receipt-backed-payouts",
            "processedBatchIds": [],
            "finalOutputBatchIds": ["caibatch_final"],
            "blockedBatchIds": [],
            "receiptBatchIds": [],
            "hashChainSha256Hexes": [],
            "executionDag": None,
            "batchRecordCount": 2,
            "processedBatchCount": 2,
            "finalOutputBatchCount": 1,
            "errorCount": 0,
            "errors": [],
            "verifiedAt": "2026-05-10T00:00:00+00:00",
        }
        instance_snapshot = {
            "instance_id": "instance-receipt-backed-payouts",
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
                {
                    "node_id": "node-relay",
                    "runner_id": "runner-relay",
                    "layer_start": 28,
                    "layer_end": 29,
                    "layer_count": 1,
                },
            ],
            "caiOwnedTransportParticipantNodeIds": [
                "node-a",
                "node-b",
                "node-relay",
            ],
            "caiOwnedTransportExecutorNodeIds": ["node-a", "node-b"],
            "caiOwnedTransportProof": proof,
        }
        payload = {
            "id": "chatcmpl-receipt-backed-payouts",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
        }

        with (
            patch("cai_compute_chain.jobs.ensure_cai_instance", return_value=instance_snapshot),
            patch(
                "cai_compute_chain.jobs.resolve_cai_instance_snapshot",
                return_value=instance_snapshot,
            ),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=state_payload,
            ),
            patch(
                "cai_compute_chain.jobs.worker_capability_verification_required",
                return_value=False,
            ),
            patch("cai_compute_chain.jobs.sync_validator_set_from_cai_peers"),
            patch("cai_compute_chain.settlement.sync_validator_set_from_cai_peers"),
            patch("cai_compute_chain.jobs.sync_chain_from_cai_peers"),
            patch("cai_compute_chain.jobs._submit_text_job_to_cai", return_value=payload),
            patch("cai_compute_chain.jobs.apply_local_validator_attestation", return_value=None),
            patch(
                "cai_compute_chain.jobs.request_remote_committee_attestations",
                return_value=[],
            ),
            patch("cai_compute_chain.jobs.push_chain_to_cai_peers", return_value=[]),
        ):
            _, receipt = execute_job_intent(job.job_id)

        payouts = sorted(list_worker_payouts(), key=lambda item: item.node_id)
        self.assertEqual([item.node_id for item in payouts], ["node-a", "node-b"])
        self.assertEqual(
            receipt.network_audit["rewardPayoutSource"],
            "cai_owned_transport_shard_receipts",
        )
        self.assertEqual(
            receipt.network_audit["rewardPayoutNodeIds"],
            ["node-a", "node-b"],
        )
        self.assertEqual(
            receipt.network_audit["rewardSkippedNodeIdsWithoutShardReceipt"],
            ["node-relay"],
        )

    def test_execute_job_intent_prices_auto_jobs_from_actual_llm_usage(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))

        with patch("cai_compute_chain.jobs.resolve_compute_price") as resolve_compute_price:
            resolve_compute_price.return_value = SimpleNamespace(
                compute_cost_atomic=coins_to_atomic("0.00150000"),
                pricing_mode="network_auto",
                pricing_basis="llm_tokens",
                pricing_reason="Reserved token budget.",
                automatic_quote=SimpleNamespace(
                    prompt_tokens_estimate=12,
                    reserved_output_tokens=64,
                    final_multiplier_bps=10_000,
                    input_token_price_atomic=coins_to_atomic("0.00000300"),
                    output_token_price_atomic=coins_to_atomic("0.00000600"),
                ),
            )
            job = create_job_intent(
                prompt="2+2=?",
                compute_amount_coins=None,
                payment_preference=PaymentPreference.AUTO,
                cai_url="http://127.0.0.1:52425",
            )

        payload = {
            "id": "chatcmpl-usage",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "4"},
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }

        with (
            patch("cai_compute_chain.jobs.ensure_cai_instance"),
            patch(
                "cai_compute_chain.jobs.resolve_cai_instance_snapshot",
                return_value={
                    "instance_id": "instance-usage",
                    "participants": [
                        {
                            "node_id": "node-a",
                            "runner_id": "runner-a",
                            "layer_start": 0,
                            "layer_end": 28,
                            "layer_count": 28,
                        }
                    ],
                },
            ),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=self._distributed_two_worker_state(),
            ),
            patch("cai_compute_chain.jobs._submit_text_job_to_cai", return_value=payload),
            patch("cai_compute_chain.jobs.apply_local_validator_attestation", return_value=None),
            patch("cai_compute_chain.jobs.request_remote_committee_attestations", return_value=[]),
        ):
            updated_job, receipt = execute_job_intent(job.job_id)

        settlement = list_settlements()[0]
        self.assertEqual(updated_job.status, "completed")
        self.assertEqual(receipt.pricing_basis, "llm_tokens")
        self.assertEqual(receipt.prompt_tokens, 10)
        self.assertEqual(receipt.completion_tokens, 20)
        self.assertEqual(receipt.total_tokens, 30)
        self.assertTrue(receipt.usage_priced)
        self.assertEqual(
            receipt.reserved_compute_cost_atomic,
            coins_to_atomic("0.00150000"),
        )
        self.assertEqual(
            receipt.actual_compute_cost_atomic,
            coins_to_atomic("0.00100000"),
        )
        self.assertEqual(
            receipt.reservation_surplus_atomic,
            coins_to_atomic("0.00050000"),
        )
        self.assertEqual(
            settlement.compute_cost_atomic,
            coins_to_atomic("0.00100000"),
        )
        execution_audit = settlement.balance_audit["execution"]
        self.assertEqual(execution_audit["receipt_id"], receipt.receipt_id)
        self.assertEqual(execution_audit["job_id"], updated_job.job_id)
        self.assertEqual(execution_audit["pricing"]["pricing_basis"], "llm_tokens")
        self.assertTrue(execution_audit["pricing"]["usage_priced"])
        self.assertEqual(
            execution_audit["pricing"]["reserved_compute_cost_atomic"],
            coins_to_atomic("0.00150000"),
        )
        self.assertEqual(
            execution_audit["pricing"]["actual_compute_cost_atomic"],
            coins_to_atomic("0.00100000"),
        )
        self.assertEqual(execution_audit["usage"]["prompt_tokens"], 10)
        self.assertEqual(execution_audit["usage"]["completion_tokens"], 20)
        self.assertEqual(execution_audit["usage"]["total_tokens"], 30)
        self.assertEqual(execution_audit["route"]["transportMode"], "single_worker")
        self.assertTrue(
            execution_audit["route"]["participantEligibility"]["canSettle"]
        )
        self.assertEqual(execution_audit["workers"]["count"], 1)
        self.assertEqual(
            execution_audit["workers"]["participants"][0]["reward_atomic"],
            settlement.worker_reward_atomic,
        )
        signed_envelope = settlement.balance_audit["signed_envelope"]
        self.assertEqual(signed_envelope["status"], "signed")
        self.assertTrue(signed_envelope["required"])
        self.assertTrue(signed_envelope["signature_valid"])

    def test_execute_job_intent_prices_cai_owned_jobs_from_proof_token_usage(
        self,
    ) -> None:
        money_policy = MoneyPolicy(
            automatic_price_floor_coins="0.00000000",
            automatic_price_cap_coins="1.00000000",
        )
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))

        with patch("cai_compute_chain.jobs.resolve_compute_price") as resolve_compute_price:
            resolve_compute_price.return_value = SimpleNamespace(
                compute_cost_atomic=coins_to_atomic("0.00150000", money_policy),
                pricing_mode="network_auto",
                pricing_basis="llm_tokens",
                pricing_reason="Reserved token budget.",
                automatic_quote=SimpleNamespace(
                    prompt_tokens_estimate=12,
                    reserved_output_tokens=64,
                    final_multiplier_bps=10_000,
                    input_token_price_atomic=coins_to_atomic(
                        "0.00000300",
                        money_policy,
                    ),
                    output_token_price_atomic=coins_to_atomic(
                        "0.00000600",
                        money_policy,
                    ),
                ),
            )
            job = create_job_intent(
                prompt="proof priced",
                compute_amount_coins=None,
                payment_preference=PaymentPreference.AUTO,
                cai_url="http://127.0.0.1:52425",
                money_policy=money_policy,
            )

        payload = {
            "id": "chatcmpl-proof-usage",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
        }
        proof = self._create_verified_cai_owned_transport_proof(
            "instance-proof-usage",
            metrics_by_node={
                "node-a": {
                    "promptTokenCount": 4,
                    "completionTokenCount": 1,
                    "inputTokenCount": 5,
                    "outputTokenCount": 1,
                },
                "node-b": {
                    "promptTokenCount": 4,
                    "completionTokenCount": 1,
                    "inputTokenCount": 5,
                    "outputTokenCount": 1,
                },
            },
        )

        with (
            patch("cai_compute_chain.jobs.ensure_cai_instance"),
            patch(
                "cai_compute_chain.jobs.resolve_cai_instance_snapshot",
                return_value={
                    "instance_id": "instance-proof-usage",
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
                    "caiOwnedTransportProof": proof,
                },
            ),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=self._distributed_two_worker_state(),
            ),
            patch("cai_compute_chain.jobs._submit_text_job_to_cai", return_value=payload),
            patch("cai_compute_chain.jobs.apply_local_validator_attestation", return_value=None),
            patch("cai_compute_chain.jobs.request_remote_committee_attestations", return_value=[]),
        ):
            updated_job, receipt = execute_job_intent(
                job.job_id,
                money_policy=money_policy,
            )

        settlement = list_settlements()[0]
        expected_actual_cost = coins_to_atomic("0.00001800", money_policy)
        self.assertEqual(updated_job.status, "completed")
        self.assertEqual(receipt.token_usage_source, "cai_owned_transport_proof")
        self.assertEqual(receipt.prompt_tokens, 4)
        self.assertEqual(receipt.completion_tokens, 1)
        self.assertEqual(receipt.total_tokens, 5)
        self.assertTrue(receipt.usage_priced)
        self.assertEqual(receipt.actual_compute_cost_atomic, expected_actual_cost)
        self.assertEqual(settlement.compute_cost_atomic, expected_actual_cost)
        token_audit = settlement.balance_audit["execution"]["token_usage_audit"]
        proof_audit = token_audit["cai_owned_transport_proof"]
        self.assertEqual(token_audit["source"], "cai_owned_transport_proof")
        self.assertEqual(token_audit["proof_usage"]["prompt_tokens"], 4)
        self.assertEqual(proof_audit["logical_prompt_token_count"], 4)
        self.assertEqual(proof_audit["logical_completion_token_count"], 1)
        self.assertEqual(proof_audit["logical_input_token_count"], 5)
        self.assertEqual(proof_audit["logical_output_token_count"], 1)
        self.assertEqual(proof_audit["aggregate_input_token_count"], 10)
        self.assertEqual(proof_audit["aggregate_output_token_count"], 2)

    def test_execute_job_intent_records_direct_multi_worker_network_audit(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))
        job = create_job_intent(
            prompt="network check",
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
        payload = {
            "id": "chatcmpl-audit",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
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
                        endpoint_url="llama-cpp-rpc://198.51.100.11:52435",
                        reachable=True,
                        checked_at="2026-05-03T00:00:00+00:00",
                        latency_ms=7.0,
                    )
                ],
            ),
            patch("cai_compute_chain.jobs._submit_text_job_to_cai", return_value=payload),
            patch("cai_compute_chain.jobs.apply_local_validator_attestation", return_value=None),
            patch("cai_compute_chain.jobs.request_remote_committee_attestations", return_value=[]),
        ):
            _, receipt = execute_job_intent(job.job_id)

        assert receipt.network_audit is not None
        self.assertEqual(receipt.network_audit["participantCount"], 2)
        self.assertEqual(receipt.network_audit["transportMode"], "multi_worker_direct")
        self.assertTrue(receipt.network_audit["stronglyConnectedDirectGraph"])
        self.assertEqual(receipt.network_audit["directBidirectionalLinkCount"], 1)
        self.assertEqual(receipt.network_audit["overlayLinkCount"], 1)
        self.assertTrue(receipt.network_audit["decentralizedExecution"])
        self.assertFalse(receipt.network_audit["relayHopsUsed"])
        self.assertTrue(receipt.network_audit["coordinatorDirectFanout"])
        self.assertEqual(
            receipt.network_audit["coordinatorCandidateNodeIds"],
            ["node-a", "node-b"],
        )
        self.assertEqual(receipt.network_audit["relayCapableNodeCount"], 1)
        self.assertEqual(receipt.network_audit["relayCapableNodeIds"], ["node-relay"])
        self.assertEqual(receipt.network_audit["relayTransitCandidateCount"], 1)
        self.assertEqual(
            receipt.network_audit["relayTransitCandidateNodeIds"],
            ["node-relay"],
        )
        self.assertEqual(receipt.network_audit["relayCoordinatorCandidateCount"], 0)
        self.assertEqual(receipt.network_audit["relayCoordinatorCandidateNodeIds"], [])
        self.assertEqual(receipt.network_audit["relayRouteCandidateCount"], 0)
        self.assertEqual(receipt.network_audit["checkedRelayRoutes"], [])
        self.assertEqual(
            receipt.network_audit["llamaCppComputeCell"]["profile"],
            "low_latency_sharded_cell",
        )
        self.assertTrue(
            receipt.network_audit["llamaCppComputeCell"]["readyForLlamaCppRpc"]
        )
        self.assertEqual(
            receipt.network_audit["llamaCppExecutionStrategy"]["executionMode"],
            "llama_cpp_rpc_low_latency",
        )
        self.assertFalse(
            receipt.network_audit["llamaCppExecutionStrategy"][
                "requiresCaiOwnedTransport"
            ]
        )
        self.assertTrue(receipt.network_audit["participantEligibility"]["canSettle"])
        settlement = list_settlements()[0]
        execution_audit = settlement.balance_audit["execution"]
        self.assertEqual(execution_audit["route"]["participantCount"], 2)
        self.assertEqual(
            execution_audit["route"]["transportMode"],
            "multi_worker_direct",
        )
        self.assertTrue(execution_audit["route"]["decentralizedExecution"])
        self.assertEqual(execution_audit["workers"]["count"], 2)
        self.assertEqual(
            [item["node_id"] for item in execution_audit["workers"]["participants"]],
            ["node-a", "node-b"],
        )

    def test_execute_job_intent_uses_attested_worker_route_health_for_settlement(
        self,
    ) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))
        job = create_job_intent(
            prompt="attested route health",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
        )

        instance_snapshot = {
            "instance_id": "instance-attested-route-health",
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
        payload = {
            "id": "chatcmpl-attested-route-health",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
        }
        attestation = SimpleNamespace(
            worker_node_id="node-a",
            accepted=True,
            expires_at="2099-01-01T00:00:00+00:00",
            updated_at="2026-05-03T00:00:00+00:00",
            last_seen_at=None,
            created_at="2026-05-03T00:00:00+00:00",
            source_url=None,
            probe_result={
                "sourceUrl": "http://127.0.0.1:25445/v1/cai/node-capabilities",
            },
        )
        route_health_payload = {
            "records": [
                {
                    "route_id": "rpc-from-attested-worker",
                    "source_node_id": "node-a",
                    "sink_node_id": "node-b",
                    "route_type": "llama_cpp_rpc_direct",
                    "endpoint_url": "llama-cpp-rpc://198.51.100.11:52435",
                    "reachable": True,
                    "checked_at": "2026-05-03T00:00:00+00:00",
                    "latency_ms": 7.0,
                }
            ],
        }
        get_json_calls: list[tuple[str, int]] = []

        def fake_get_json(url: str, *, timeout: int = 30) -> dict:
            get_json_calls.append((url, timeout))
            if url == "http://127.0.0.1:25445/v1/cai/route-health":
                return route_health_payload
            raise AssertionError(f"unexpected URL: {url}")

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
                "cai_compute_chain.jobs.resolve_cai_command_instance_snapshot",
                return_value=None,
            ),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=self._distributed_two_worker_state(),
            ),
            patch("cai_compute_chain.jobs.sync_validator_set_from_cai_peers", return_value=None),
            patch("cai_compute_chain.settlement.sync_validator_set_from_cai_peers", return_value=None),
            patch("cai_compute_chain.jobs.sync_chain_from_cai_peers", return_value=None),
            patch("cai_compute_chain.jobs.list_route_health_records", return_value=[]),
            patch(
                "cai_compute_chain.jobs.list_worker_capability_attestations",
                return_value=[attestation],
            ),
            patch("cai_compute_chain.jobs._get_json", side_effect=fake_get_json),
            patch("cai_compute_chain.jobs._submit_text_job_to_cai", return_value=payload),
            patch("cai_compute_chain.jobs._sync_worker_reward_bindings_from_cai", return_value=None),
            patch("cai_compute_chain.jobs.apply_local_validator_attestation", return_value=None),
            patch("cai_compute_chain.jobs.request_remote_committee_attestations", return_value=[]),
            patch("cai_compute_chain.jobs._apply_settlement_after_canonical_chain_sync", return_value=None),
            patch("cai_compute_chain.jobs.push_chain_to_cai_peers", return_value=None),
        ):
            _, receipt = execute_job_intent(job.job_id)

        self.assertEqual(
            get_json_calls,
            [("http://127.0.0.1:25445/v1/cai/route-health", 5)],
        )
        self.assertEqual(
            receipt.network_audit["llamaCppComputeCell"]["profile"],
            "low_latency_sharded_cell",
        )
        self.assertEqual(
            receipt.network_audit["llamaCppExecutionStrategy"]["executionMode"],
            "llama_cpp_rpc_low_latency",
        )
        self.assertTrue(receipt.network_audit["participantEligibility"]["canSettle"])
        payouts = sorted(list_worker_payouts(), key=lambda item: item.node_id)
        self.assertEqual([item.node_id for item in payouts], ["node-a", "node-b"])

    def test_execution_network_audit_marks_single_remote_worker_decentralized(self) -> None:
        audit = _build_execution_network_audit(
            state_payload={
                "nodeIdentities": {
                    "node-local": {"apiHost": "127.0.0.1", "apiPort": 52425},
                    "node-remote": {"apiHost": "26.97.29.153", "apiPort": 52425},
                }
            },
            instance_snapshot={
                "instance_id": "instance-remote",
                "participants": [
                    {
                        "node_id": "node-remote",
                        "runner_id": "runner-remote",
                        "layer_start": 0,
                        "layer_end": 28,
                        "layer_count": 28,
                    }
                ],
            },
            requester_node_id="node-local",
        )

        self.assertEqual(audit["transportMode"], "single_worker")
        self.assertEqual(audit["requesterNodeId"], "node-local")
        self.assertTrue(audit["singleWorkerRemote"])
        self.assertFalse(audit["singleWorkerSelfExecution"])
        self.assertTrue(audit["decentralizedExecution"])

    def test_execution_network_audit_marks_single_self_worker_as_non_decentralized(
        self,
    ) -> None:
        audit = _build_execution_network_audit(
            state_payload={
                "nodeIdentities": {
                    "node-local": {"apiHost": "127.0.0.1", "apiPort": 52425},
                }
            },
            instance_snapshot={
                "instance_id": "instance-local",
                "participants": [
                    {
                        "node_id": "node-local",
                        "runner_id": "runner-local",
                        "layer_start": 0,
                        "layer_end": 28,
                        "layer_count": 28,
                    }
                ],
            },
            requester_node_id="node-local",
        )

        self.assertEqual(audit["transportMode"], "single_worker")
        self.assertFalse(audit["singleWorkerRemote"])
        self.assertTrue(audit["singleWorkerSelfExecution"])
        self.assertFalse(audit["decentralizedExecution"])

    def test_multi_worker_wan_risky_compute_cell_blocks_settlement(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))
        job = create_job_intent(
            prompt="network check",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
        )
        instance_snapshot = {
            "instance_id": "instance-wan-risky",
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
        payload = {
            "id": "chatcmpl-wan-risky",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
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
                        route_id="rpc-wan-risky",
                        source_node_id="node-a",
                        sink_node_id="node-b",
                        route_type="llama_cpp_rpc_direct",
                        endpoint_url="llama-cpp-rpc://198.51.100.11:52435",
                        reachable=True,
                        checked_at="2026-05-03T00:00:00+00:00",
                        latency_ms=48.0,
                    )
                ],
            ),
            patch("cai_compute_chain.jobs._submit_text_job_to_cai", return_value=payload),
            patch("cai_compute_chain.jobs.apply_local_validator_attestation", return_value=None),
            patch("cai_compute_chain.jobs.request_remote_committee_attestations", return_value=[]),
        ):
            with self.assertRaisesRegex(RuntimeError, "compute-cell is not settlement-safe"):
                execute_job_intent(job.job_id)

        self.assertEqual(list_worker_payouts(), [])
        self.assertEqual(list_execution_receipts(), [])

    def test_unproven_multi_worker_compute_cell_requires_transport_proof(self) -> None:
        instance_snapshot = {
            "instance_id": "instance-unproven",
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
        state_payload = self._distributed_two_worker_state()
        audit = _build_execution_network_audit(
            state_payload=state_payload,
            instance_snapshot=instance_snapshot,
            requester_node_id="node-user",
            route_health_records=[],
        )

        eligibility = _build_participant_eligibility_audit(
            state_payload=state_payload,
            instance_snapshot=instance_snapshot,
            requested_model_id="cai-network/Qwen3-0.6B-GGUF",
            execution_model_id="cai-network/Qwen3-0.6B-GGUF",
            network_audit=audit,
        )

        self.assertEqual(
            audit["llamaCppExecutionStrategy"]["executionMode"],
            "cai_owned_transport_required",
        )
        self.assertTrue(
            audit["llamaCppExecutionStrategy"]["requiresCaiOwnedTransport"]
        )
        self.assertFalse(eligibility["canSettle"])
        self.assertTrue(
            any(
                "CAI-owned WAN-safe transport" in item
                for item in eligibility["fatalReasons"]
            )
        )

    def test_private_network_worker_id_allows_internal_execution_model(self) -> None:
        accepted_model_ids = _accepted_worker_model_ids("cai-network/Qwen3-0.6B-GGUF")

        self.assertIn("cai-network/Qwen3-0.6B-GGUF", accepted_model_ids)
        self.assertTrue(
            _worker_model_allowed(
                ["cai-network/Qwen3-0.6B-GGUF"],
                accepted_model_ids,
            )
        )

    def test_public_qwen_test_model_allows_default_worker_config(self) -> None:
        accepted_model_ids = _accepted_worker_model_ids(
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
        )

        self.assertIn("Qwen/Qwen2.5-0.5B-Instruct-GGUF", accepted_model_ids)
        self.assertTrue(
            _worker_model_allowed(
                [
                    "cai-network/Qwen3-0.6B-GGUF",
                    "Qwen/Qwen3-0.6B-GGUF",
                    "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                ],
                accepted_model_ids,
            )
        )

    def test_reconcile_stale_running_job_intents_marks_old_job_failed(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))
        job = create_job_intent(
            prompt="stale",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
        )
        job.status = "running"
        job.created_at = "2026-01-01T00:00:00+00:00"
        update_job_intent(job)

        changed = reconcile_stale_running_job_intents(stale_after_seconds=60)

        persisted = next(
            item for item in list_job_intents() if item.job_id == job.job_id
        )
        self.assertEqual(changed, 1)
        self.assertEqual(persisted.status, "failed")
        self.assertIn("remained running", persisted.last_error or "")

    def test_reconcile_stale_running_job_intents_recovers_running_job_from_receipt(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("20000.00000000"))
        self._credit_wallet_on_chain(wallet, coins_to_atomic("20000.00000000"))
        set_validator_static_ip_confirmation(True)
        set_validator_mode(
            True,
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )
        job = create_job_intent(
            prompt="recover me",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
        )

        payload = {
            "id": "chatcmpl-recover",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
        }

        with (
            patch("cai_compute_chain.jobs.ensure_cai_instance"),
            patch(
                "cai_compute_chain.jobs.resolve_cai_instance_snapshot",
                return_value={
                    "instance_id": "instance-1",
                    "participants": [
                        {
                            "node_id": "node-a",
                            "runner_id": "runner-a",
                            "layer_start": 0,
                            "layer_end": 28,
                            "layer_count": 28,
                        }
                    ],
                },
            ),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=self._distributed_two_worker_state(),
            ),
            patch("cai_compute_chain.jobs._submit_text_job_to_cai", return_value=payload),
            patch("cai_compute_chain.jobs.request_remote_committee_attestations", return_value=[]),
        ):
            updated_job, receipt = execute_job_intent(job.job_id)

        expected_settlement_id = updated_job.settlement_id
        updated_job.status = "running"
        updated_job.receipt_id = None
        updated_job.settlement_id = None
        update_job_intent(updated_job)

        changed = reconcile_stale_running_job_intents(stale_after_seconds=3600)

        persisted = next(
            item for item in list_job_intents() if item.job_id == job.job_id
        )
        self.assertEqual(changed, 1)
        self.assertEqual(persisted.status, "completed")
        self.assertEqual(persisted.receipt_id, receipt.receipt_id)
        self.assertEqual(persisted.settlement_id, expected_settlement_id)

    def test_verified_cai_owned_transport_proof_allows_multi_worker_settlement(
        self,
    ) -> None:
        instance_snapshot = {
            "instance_id": "instance-proof",
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
            "caiOwnedTransportProof": self._create_verified_cai_owned_transport_proof(
                "instance-proof"
            ),
        }
        state_payload = self._distributed_two_worker_state()
        audit = _build_execution_network_audit(
            state_payload=state_payload,
            instance_snapshot=instance_snapshot,
            requester_node_id="node-user",
            route_health_records=[],
        )

        eligibility = _build_participant_eligibility_audit(
            state_payload=state_payload,
            instance_snapshot=instance_snapshot,
            requested_model_id="cai-network/Qwen3-0.6B-GGUF",
            execution_model_id="cai-network/Qwen3-0.6B-GGUF",
            network_audit=audit,
        )

        self.assertTrue(audit["caiOwnedTransportExecuted"])
        self.assertTrue(
            audit["llamaCppExecutionStrategy"]["caiOwnedTransportExecuted"]
        )
        self.assertTrue(eligibility["canSettle"])

    def test_verified_cai_owned_transport_proof_allows_overlay_only_multi_worker_settlement(
        self,
    ) -> None:
        instance_snapshot = {
            "instance_id": "instance-overlay-proof",
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
            "caiOwnedTransportProof": self._create_verified_cai_owned_transport_proof(
                "instance-overlay-proof"
            ),
        }
        state_payload = self._relay_candidate_two_worker_state()
        audit = _build_execution_network_audit(
            state_payload=state_payload,
            instance_snapshot=instance_snapshot,
            requester_node_id="node-user",
            route_health_records=[],
        )

        eligibility = _build_participant_eligibility_audit(
            state_payload=state_payload,
            instance_snapshot=instance_snapshot,
            requested_model_id="cai-network/Qwen3-0.6B-GGUF",
            execution_model_id="cai-network/Qwen3-0.6B-GGUF",
            network_audit=audit,
        )

        self.assertEqual(audit["transportMode"], "multi_worker_disconnected")
        self.assertTrue(audit["caiOwnedTransportExecuted"])
        self.assertTrue(eligibility["routeReachable"])
        self.assertTrue(eligibility["canSettle"])

    def test_strict_worker_capabilities_block_unverified_participants(
        self,
    ) -> None:
        instance_snapshot = {
            "instance_id": "instance-strict-unverified-capability",
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
            "caiOwnedTransportProof": self._create_verified_cai_owned_transport_proof(
                "instance-strict-unverified-capability"
            ),
        }
        state_payload = self._distributed_two_worker_state()
        audit = _build_execution_network_audit(
            state_payload=state_payload,
            instance_snapshot=instance_snapshot,
            requester_node_id="node-user",
            route_health_records=[],
        )

        with (
            patch(
                "cai_compute_chain.jobs.worker_capability_verification_required",
                return_value=True,
            ),
            patch(
                "cai_compute_chain.jobs.list_verified_worker_node_ids",
                return_value=set(),
            ),
        ):
            eligibility = _build_participant_eligibility_audit(
                state_payload=state_payload,
                instance_snapshot=instance_snapshot,
                requested_model_id="cai-network/Qwen3-0.6B-GGUF",
                execution_model_id="cai-network/Qwen3-0.6B-GGUF",
                network_audit=audit,
            )

        self.assertFalse(eligibility["canSettle"])
        self.assertTrue(
            any(
                "worker capability is not verified" in reason
                for reason in eligibility["fatalReasons"]
            )
        )
        self.assertEqual(
            [item["verifiedCapability"] for item in eligibility["participants"]],
            [False, False],
        )

    def test_strict_worker_capabilities_allow_verified_participants(
        self,
    ) -> None:
        instance_snapshot = {
            "instance_id": "instance-strict-verified-capability",
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
            "caiOwnedTransportProof": self._create_verified_cai_owned_transport_proof(
                "instance-strict-verified-capability"
            ),
        }
        state_payload = self._distributed_two_worker_state()
        audit = _build_execution_network_audit(
            state_payload=state_payload,
            instance_snapshot=instance_snapshot,
            requester_node_id="node-user",
            route_health_records=[],
        )

        with (
            patch(
                "cai_compute_chain.jobs.worker_capability_verification_required",
                return_value=True,
            ),
            patch(
                "cai_compute_chain.jobs.list_verified_worker_node_ids",
                return_value={"node-a", "node-b"},
            ),
        ):
            eligibility = _build_participant_eligibility_audit(
                state_payload=state_payload,
                instance_snapshot=instance_snapshot,
                requested_model_id="cai-network/Qwen3-0.6B-GGUF",
                execution_model_id="cai-network/Qwen3-0.6B-GGUF",
                network_audit=audit,
            )

        self.assertTrue(eligibility["canSettle"])
        self.assertEqual(
            [item["verifiedCapability"] for item in eligibility["participants"]],
            [True, True],
        )

    def test_duplicate_cai_owned_transport_proof_receipt_blocks_settlement(
        self,
    ) -> None:
        proof = self._create_verified_cai_owned_transport_proof(
            "instance-duplicate-proof"
        )
        proof["shardReceipts"][1]["batchIds"] = list(
            proof["shardReceipts"][0]["batchIds"]
        )
        instance_snapshot = {
            "instance_id": "instance-duplicate-proof",
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
            "caiOwnedTransportProof": proof,
        }

        audit = _build_execution_network_audit(
            state_payload=self._distributed_two_worker_state(),
            instance_snapshot=instance_snapshot,
            requester_node_id="node-user",
            route_health_records=[],
        )
        eligibility = _build_participant_eligibility_audit(
            state_payload=self._distributed_two_worker_state(),
            instance_snapshot=instance_snapshot,
            requested_model_id="cai-network/Qwen3-0.6B-GGUF",
            execution_model_id="cai-network/Qwen3-0.6B-GGUF",
            network_audit=audit,
        )

        self.assertFalse(audit["caiOwnedTransportExecuted"])
        self.assertEqual(
            audit["caiOwnedTransportProofError"],
            (
                "CAI-owned transport proof duplicates batch id "
                f"'{proof['shardReceipts'][0]['batchIds'][0]}'."
            ),
        )
        self.assertFalse(eligibility["canSettle"])

    def test_invalid_cai_owned_transport_proof_does_not_spend_reserve_limit(
        self,
    ) -> None:
        money_policy = MoneyPolicy(
            daily_user_reserve_limit_coins="1.00000000",
            default_tx_fee_coins="0.00010000",
        )
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        job = create_job_intent(
            prompt="reserve must not be spent",
            compute_amount_coins="0.99990000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
            money_policy=money_policy,
        )
        reserve_address = compute_reserve_chain_address(money_policy)
        reserve_before = chain_balance_atomic(reserve_address)
        proof = self._create_verified_cai_owned_transport_proof(
            "instance-invalid-proof-spend"
        )
        proof["shardReceipts"][1]["batchIds"] = list(
            proof["shardReceipts"][0]["batchIds"]
        )
        instance_snapshot = {
            "instance_id": "instance-invalid-proof-spend",
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
            "caiOwnedTransportProof": proof,
        }
        response = {
            "id": "chatcmpl-invalid-proof-spend",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "should-not-settle"},
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
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
                "cai_compute_chain.jobs._submit_text_job_to_cai",
                return_value=response,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "CAI-owned WAN-safe transport"):
                execute_job_intent(job.job_id, money_policy=money_policy)

        self.assertEqual(list_execution_receipts(), [])
        self.assertEqual(list_settlements(), [])
        self.assertEqual(list_worker_payouts(), [])
        self.assertEqual(
            chain_balance_atomic(reserve_address),
            reserve_before,
        )
        self.assertFalse(
            any(
                tx.tx_type == "settlement_compute_reserve_debit"
                for block in list_chain_blocks()
                for tx in block.transactions
            )
        )

        ledger = load_or_create_ledger(money_policy)
        decision = plan_funding(
            ledger=ledger,
            wallet=wallet,
            compute_cost_atomic=coins_to_atomic("0.00020000", money_policy),
            payment_preference=PaymentPreference.AUTO,
            money_policy=money_policy,
        )
        self.assertTrue(decision.can_fund)
        self.assertEqual(decision.funding_source.value, "reserve")
        self.assertEqual(decision.daily_reserve_spent_today_atomic, 0)

    def test_cai_owned_transport_proof_without_execution_audit_blocks_settlement(
        self,
    ) -> None:
        instance_snapshot = {
            "instance_id": "instance-unverified-proof",
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
            "caiOwnedTransportProof": build_cai_owned_transport_execution_proof(
                session_id="session-unverified-proof",
                instance_id="instance-unverified-proof",
                participant_node_ids=["node-a", "node-b"],
                model_id="cai-network/Qwen3-0.6B-GGUF",
                activation_batch_count=2,
                decode_batch_count=1,
            ),
        }
        audit = _build_execution_network_audit(
            state_payload=self._distributed_two_worker_state(),
            instance_snapshot=instance_snapshot,
            requester_node_id="node-user",
            route_health_records=[],
        )
        eligibility = _build_participant_eligibility_audit(
            state_payload=self._distributed_two_worker_state(),
            instance_snapshot=instance_snapshot,
            requested_model_id="cai-network/Qwen3-0.6B-GGUF",
            execution_model_id="cai-network/Qwen3-0.6B-GGUF",
            network_audit=audit,
        )

        self.assertFalse(audit["caiOwnedTransportExecuted"])
        self.assertEqual(
            audit["caiOwnedTransportProofError"],
            "CAI-owned transport execution audit is missing or not verified.",
        )
        self.assertFalse(eligibility["canSettle"])

    def test_network_audit_uses_persisted_cai_owned_transport_session_proof(
        self,
    ) -> None:
        instance_snapshot = {
            "instance_id": "instance-persisted-proof",
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
        proof = self._create_verified_cai_owned_transport_proof(
            "instance-persisted-proof"
        )

        audit = _build_execution_network_audit(
            state_payload=self._distributed_two_worker_state(),
            instance_snapshot=instance_snapshot,
            requester_node_id="node-user",
            route_health_records=[],
        )

        self.assertTrue(audit["caiOwnedTransportExecuted"])
        self.assertEqual(
            audit["caiOwnedTransportExecutionProof"]["sessionId"],
            proof["sessionId"],
        )
        self.assertTrue(
            audit["caiOwnedTransportExecutionProof"]["executionAudit"]["verified"]
        )

    def test_execute_job_intent_records_one_way_coordinator_fanout_as_direct_execution(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))
        job = create_job_intent(
            prompt="one-way fanout",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
        )

        instance_snapshot = {
            "instance_id": "instance-fanout",
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
            "caiOwnedTransportProof": self._create_verified_cai_owned_transport_proof(
                "instance-fanout"
            ),
        }
        payload = {
            "id": "chatcmpl-fanout",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
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
                return_value=self._distributed_oneway_two_worker_state(),
            ),
            patch("cai_compute_chain.jobs._submit_text_job_to_cai", return_value=payload),
            patch("cai_compute_chain.jobs.apply_local_validator_attestation", return_value=None),
            patch("cai_compute_chain.jobs.request_remote_committee_attestations", return_value=[]),
        ):
            _, receipt = execute_job_intent(job.job_id)

        assert receipt.network_audit is not None
        self.assertEqual(receipt.network_audit["transportMode"], "multi_worker_direct")
        self.assertTrue(receipt.network_audit["coordinatorDirectFanout"])
        self.assertEqual(receipt.network_audit["coordinatorCandidateNodeIds"], ["node-a"])
        self.assertFalse(receipt.network_audit["stronglyConnectedDirectGraph"])
        self.assertTrue(receipt.network_audit["decentralizedExecution"])
        self.assertEqual(receipt.network_audit["relayCoordinatorCandidateCount"], 0)
        self.assertEqual(receipt.network_audit["relayRouteCandidateCount"], 0)

    def test_execute_job_intent_records_relay_candidate_routes_without_marking_execution_decentralized(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))
        job = create_job_intent(
            prompt="relay candidate",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
        )

        instance_snapshot = {
            "instance_id": "instance-relay-candidate",
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
        payload = {
            "id": "chatcmpl-relay-candidate",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
        }
        state_payload = self._relay_candidate_two_worker_state()

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
                return_value=state_payload,
            ),
            patch("cai_compute_chain.jobs._submit_text_job_to_cai", return_value=payload),
        ):
            with self.assertRaisesRegex(RuntimeError, "worker participant eligibility"):
                execute_job_intent(job.job_id)

        persisted_job = next(item for item in list_job_intents() if item.job_id == job.job_id)
        self.assertEqual(persisted_job.status, "failed")
        self.assertIn("multi-worker route was not proven", persisted_job.last_error or "")
        self.assertEqual(list_execution_receipts(), [])
        self.assertEqual(list_settlements(), [])
        self.assertEqual(list_worker_payouts(), [])

        network_audit = _build_execution_network_audit(
            state_payload=state_payload,
            instance_snapshot=instance_snapshot,
        )
        self.assertEqual(network_audit["transportMode"], "multi_worker_disconnected")
        self.assertFalse(network_audit["decentralizedExecution"])
        self.assertFalse(network_audit["coordinatorDirectFanout"])
        self.assertEqual(network_audit["relayCoordinatorCandidateCount"], 2)
        self.assertEqual(
            network_audit["relayCoordinatorCandidateNodeIds"],
            ["node-a", "node-b"],
        )
        self.assertEqual(network_audit["relayRouteCandidateCount"], 2)
        self.assertEqual(
            network_audit["checkedRelayRoutes"],
            [
                {
                    "candidateOnly": True,
                    "pathNodeIds": ["node-a", "node-relay", "node-b"],
                    "sinkNodeId": "node-b",
                    "sinkSegmentType": "overlay",
                    "sourceNodeId": "node-a",
                    "sourceSegmentType": "overlay",
                    "transitNodeId": "node-relay",
                    "transitParticipates": False,
                },
                {
                    "candidateOnly": True,
                    "pathNodeIds": ["node-b", "node-relay", "node-a"],
                    "sinkNodeId": "node-a",
                    "sinkSegmentType": "overlay",
                    "sourceNodeId": "node-b",
                    "sourceSegmentType": "overlay",
                    "transitNodeId": "node-relay",
                    "transitParticipates": False,
                },
            ],
        )

    def test_execute_job_intent_marks_active_relay_routes_when_instance_uses_them(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))
        job = create_job_intent(
            prompt="relay active",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
        )

        instance_snapshot = {
            "instance_id": "instance-relay-active",
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
            "relay_routes_by_node": {
                "node-a": [
                    {
                        "sourceNodeId": "node-a",
                        "transitNodeId": "node-relay",
                        "sinkNodeId": "node-b",
                        "relayApiHost": "198.51.100.12",
                        "relayApiPort": 52425,
                        "targetHost": "198.51.100.11",
                        "targetPort": 52435,
                        "sourceSegmentType": "overlay",
                        "sinkSegmentType": "overlay",
                    }
                ]
            },
            "caiOwnedTransportProof": self._create_verified_cai_owned_transport_proof(
                "instance-relay-active"
            ),
        }
        payload = {
            "id": "chatcmpl-relay-active",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
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
                return_value=self._relay_candidate_two_worker_state(),
            ),
            patch("cai_compute_chain.jobs._submit_text_job_to_cai", return_value=payload),
            patch("cai_compute_chain.jobs.apply_local_validator_attestation", return_value=None),
            patch("cai_compute_chain.jobs.request_remote_committee_attestations", return_value=[]),
        ):
            _, receipt = execute_job_intent(job.job_id)

        assert receipt.network_audit is not None
        self.assertEqual(receipt.network_audit["transportMode"], "multi_worker_relay")
        self.assertTrue(receipt.network_audit["decentralizedExecution"])
        self.assertTrue(receipt.network_audit["relayHopsUsed"])
        self.assertTrue(receipt.network_audit["relayBottleneckRisk"])
        self.assertEqual(receipt.network_audit["activeRelayTransitNodeIds"], ["node-relay"])
        self.assertEqual(
            receipt.network_audit["checkedRelayRoutes"],
            [
                {
                    "candidateOnly": False,
                    "pathNodeIds": ["node-a", "node-relay", "node-b"],
                    "sinkNodeId": "node-b",
                    "sinkSegmentType": "overlay",
                    "sourceNodeId": "node-a",
                    "sourceSegmentType": "overlay",
                    "transitNodeId": "node-relay",
                    "transitParticipates": False,
                }
            ],
        )

    def test_execute_job_intent_allows_locked_wallet_when_reserve_funds_all(self) -> None:
        job = create_job_intent(
            prompt="2+2=?",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
        )

        payload = {
            "id": "chatcmpl-device-wallet",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "4"},
                }
            ],
        }
        instance_snapshot = {
            "instance_id": "instance-device-wallet",
            "participants": [
                {
                    "node_id": "node-a",
                    "runner_id": "runner-a",
                    "layer_start": 0,
                    "layer_end": 28,
                    "layer_count": 28,
                }
            ],
        }

        with (
            patch("cai_compute_chain.jobs.ensure_cai_instance", return_value=instance_snapshot),
            patch(
                "cai_compute_chain.jobs.resolve_cai_instance_snapshot",
                return_value=instance_snapshot,
            ),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=self._public_validator_state(),
            ),
            patch("cai_compute_chain.jobs._submit_text_job_to_cai", return_value=payload),
            patch("cai_compute_chain.jobs.apply_local_validator_attestation", return_value=None),
            patch("cai_compute_chain.jobs.request_remote_committee_attestations", return_value=[]),
        ):
            updated_job, receipt = execute_job_intent(job.job_id)

        self.assertEqual(updated_job.status, "completed")
        self.assertEqual(receipt.output_text, "4")
        self.assertEqual(list_settlements()[0].funding_source, "reserve")
        self.assertEqual(list_settlements()[0].status, "pending")

        from cai_compute_chain.wallet import load_or_create_ledger

        ledger = load_or_create_ledger()
        self.assertEqual(ledger.settlements_applied, 0)
        self.assertEqual(ledger.worker_distributed_atomic, 0)

    def test_execute_job_intent_uses_worker_url_for_execution_and_validator_url_for_attestation(
        self,
    ) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("20000.00000000"))
        job = create_job_intent(
            prompt="2+3=?",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52415",
            execution_cai_url="http://127.0.0.1:52425",
        )

        instance_snapshot = {
            "instance_id": "instance-split-url",
            "participants": [
                {
                    "node_id": "node-worker",
                    "runner_id": "runner-worker",
                    "layer_start": 0,
                    "layer_end": 28,
                    "layer_count": 28,
                }
            ],
        }
        payload = {
            "id": "chatcmpl-split-url",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "5"},
                }
            ],
        }

        with (
            patch(
                "cai_compute_chain.jobs.ensure_cai_instance",
                return_value=instance_snapshot,
            ) as ensure_instance_mock,
            patch(
                "cai_compute_chain.jobs.resolve_cai_instance_snapshot",
                return_value=instance_snapshot,
            ),
            patch(
                "cai_compute_chain.jobs._submit_text_job_to_cai",
                return_value=payload,
            ) as submit_mock,
            patch(
                "cai_compute_chain.jobs.resolve_cai_command_instance_snapshot",
                return_value=None,
            ),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=self._public_validator_state(),
            ) as load_state_mock,
            patch(
                "cai_compute_chain.jobs.apply_local_validator_attestation",
                return_value=None,
            ) as local_attest_mock,
            patch(
                "cai_compute_chain.jobs.request_remote_committee_attestations",
                return_value=[],
            ) as remote_attest_mock,
        ):
            updated_job, receipt = execute_job_intent(job.job_id)

        self.assertEqual(updated_job.status, "completed")
        self.assertEqual(receipt.output_text, "5")
        self.assertEqual(receipt.cai_url, "http://127.0.0.1:52425")
        self.assertEqual(ensure_instance_mock.call_args.args[0], "http://127.0.0.1:52425")
        self.assertEqual(submit_mock.call_args.args[0], "http://127.0.0.1:52425")
        self.assertEqual(load_state_mock.call_args.args[0], "http://127.0.0.1:52415")
        self.assertEqual(
            local_attest_mock.call_args.kwargs["cai_url"], "http://127.0.0.1:52415"
        )
        self.assertEqual(
            remote_attest_mock.call_args.kwargs["cai_url"], "http://127.0.0.1:52415"
        )

    def test_execute_job_intent_wallet_only_debits_wallet_and_preserves_reserve(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("20000.00000000"))
        self._credit_wallet_on_chain(wallet, coins_to_atomic("20000.00000000"))
        set_validator_static_ip_confirmation(True)
        set_validator_mode(
            True,
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )
        job = create_job_intent(
            prompt="3+4=?",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.WALLET_ONLY,
            cai_url="http://127.0.0.1:52425",
        )

        payload = {
            "id": "chatcmpl-wallet-only",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "7"},
                }
            ],
        }

        with (
            patch("cai_compute_chain.jobs.ensure_cai_instance"),
            patch(
                "cai_compute_chain.jobs.resolve_cai_instance_snapshot",
                return_value={
                    "instance_id": "instance-wallet-only",
                    "participants": [
                        {
                            "node_id": "node-a",
                            "runner_id": "runner-a",
                            "layer_start": 0,
                            "layer_end": 28,
                            "layer_count": 28,
                        }
                    ],
                },
            ),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=self._public_validator_state(),
            ),
            patch("cai_compute_chain.jobs._submit_text_job_to_cai", return_value=payload),
        ):
            updated_job, receipt = execute_job_intent(job.job_id)

        self.assertEqual(updated_job.status, "completed")
        self.assertEqual(receipt.output_text, "7")

        from cai_compute_chain.wallet import get_active_wallet, load_or_create_ledger

        active_wallet = get_active_wallet()
        ledger = load_or_create_ledger()
        self.assertIsNotNone(active_wallet)
        self.assertEqual(
            active_wallet.spendable_balance_atomic,
            coins_to_atomic("9998.99990000"),
        )
        self.assertEqual(
            ledger.compute_reserve_balance_atomic,
            coins_to_atomic("850000000.00000000"),
        )
        self.assertEqual(list_settlements()[0].funding_source, "wallet")

    def test_execute_job_intent_wallet_only_ignores_local_only_balance(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("20000.00000000"))
        job = create_job_intent(
            prompt="local cache should not fund this",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.WALLET_ONLY,
            cai_url="http://127.0.0.1:52425",
        )

        with self.assertRaisesRegex(ValueError, "Wallet balance is insufficient"):
            execute_job_intent(job.job_id)

        persisted_job = next(item for item in list_job_intents() if item.job_id == job.job_id)
        self.assertEqual(persisted_job.status, "failed")
        self.assertEqual(list_execution_receipts(), [])
        self.assertEqual(list_settlements(), [])

    def test_execute_job_intent_retries_transient_startup_failure(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("20000.00000000"))
        job = create_job_intent(
            prompt="2+2=?",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
        )

        instance_snapshot = {
            "instance_id": "instance-retry",
            "participants": [
                {
                    "node_id": "node-a",
                    "runner_id": "runner-a",
                    "layer_start": 0,
                    "layer_end": 28,
                    "layer_count": 28,
                }
            ],
        }
        payload = {
            "id": "chatcmpl-retry",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "4"},
                }
            ],
        }

        with (
            patch(
                "cai_compute_chain.jobs.ensure_cai_instance",
                side_effect=[
                    ValueError(
                        "No usable CAI placement preview found for model Qwen/Qwen3-0.6B-GGUF."
                    ),
                    instance_snapshot,
                ],
            ) as ensure_instance_mock,
            patch(
                "cai_compute_chain.jobs.resolve_cai_instance_snapshot",
                return_value=instance_snapshot,
            ),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=self._public_validator_state(),
            ),
            patch("cai_compute_chain.jobs._submit_text_job_to_cai", return_value=payload),
            patch("cai_compute_chain.jobs.cleanup_cai_model_instances") as cleanup_mock,
            patch("cai_compute_chain.jobs.time.sleep"),
        ):
            updated_job, receipt = execute_job_intent(job.job_id)

        self.assertEqual(updated_job.status, "completed")
        self.assertEqual(receipt.output_text, "4")
        self.assertEqual(ensure_instance_mock.call_count, 2)
        cleanup_mock.assert_called_once()

    def test_execute_job_intent_retries_generation_timeout_on_alternate_node(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("20000.00000000"))
        job = create_job_intent(
            prompt="2+3=?",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
        )

        first_snapshot = {
            "instance_id": "instance-node-a",
            "participants": [
                {
                    "node_id": "node-a",
                    "runner_id": "runner-a",
                    "layer_start": 0,
                    "layer_end": 28,
                    "layer_count": 28,
                }
            ],
        }
        second_snapshot = {
            "instance_id": "instance-node-b",
            "participants": [
                {
                    "node_id": "node-b",
                    "runner_id": "runner-b",
                    "layer_start": 0,
                    "layer_end": 28,
                    "layer_count": 28,
                }
            ],
        }
        payload = {
            "id": "chatcmpl-retry-generation",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "5"},
                }
            ],
        }

        with (
            patch(
                "cai_compute_chain.jobs.ensure_cai_instance",
                side_effect=[first_snapshot, second_snapshot],
            ) as ensure_instance_mock,
            patch(
                "cai_compute_chain.jobs.resolve_cai_instance_snapshot",
                return_value=None,
            ),
            patch(
                "cai_compute_chain.jobs.resolve_cai_command_instance_snapshot",
                return_value=None,
            ),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=self._distributed_two_worker_state(),
            ),
            patch(
                "cai_compute_chain.jobs._submit_text_job_to_cai",
                side_effect=[TimeoutError("timed out"), payload],
            ) as submit_mock,
            patch("cai_compute_chain.jobs.cleanup_cai_model_instances") as cleanup_mock,
            patch("cai_compute_chain.jobs.time.sleep"),
        ):
            updated_job, receipt = execute_job_intent(job.job_id)

        self.assertEqual(updated_job.status, "completed")
        self.assertEqual(receipt.output_text, "5")
        self.assertEqual(ensure_instance_mock.call_count, 2)
        self.assertNotIn("excluded_node_ids", ensure_instance_mock.call_args_list[0].kwargs)
        self.assertEqual(
            ensure_instance_mock.call_args_list[1].kwargs["excluded_node_ids"],
            ["node-a"],
        )
        self.assertEqual(submit_mock.call_count, 2)
        cleanup_mock.assert_called_once()
        self.assertEqual(
            [item["status"] for item in updated_job.execution_attempts],
            ["retrying", "completed"],
        )
        self.assertEqual(
            updated_job.execution_attempts[0]["participantNodeIds"],
            ["node-a"],
        )
        self.assertEqual(receipt.network_audit["executionAttemptCount"], 2)

    def test_execute_job_intent_persists_running_attempt_before_submit(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("20000.00000000"))
        job = create_job_intent(
            prompt="2+4=?",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
        )

        instance_snapshot = {
            "instance_id": "instance-running-attempt",
            "participants": [
                {
                    "node_id": "node-a",
                    "runner_id": "runner-a",
                    "layer_start": 0,
                    "layer_end": 28,
                    "layer_count": 28,
                }
            ],
        }
        payload = {
            "id": "chatcmpl-running-attempt",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "6"},
                }
            ],
        }
        observed_statuses: list[str] = []
        observed_phases: list[str | None] = []

        def fake_ensure(*args, **kwargs):
            persisted_job = next(
                item for item in list_job_intents() if item.job_id == job.job_id
            )
            observed_phases.append(persisted_job.execution_attempts[-1].get("phase"))
            return instance_snapshot

        def fake_submit(*args, **kwargs):
            persisted_job = next(
                item for item in list_job_intents() if item.job_id == job.job_id
            )
            observed_statuses.append(persisted_job.execution_attempts[-1]["status"])
            observed_phases.append(persisted_job.execution_attempts[-1].get("phase"))
            return payload

        with (
            patch(
                "cai_compute_chain.jobs.ensure_cai_instance",
                side_effect=fake_ensure,
            ),
            patch(
                "cai_compute_chain.jobs.resolve_cai_instance_snapshot",
                return_value=instance_snapshot,
            ),
            patch(
                "cai_compute_chain.jobs.resolve_cai_command_instance_snapshot",
                return_value=None,
            ),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=self._public_validator_state(),
            ),
            patch("cai_compute_chain.jobs._submit_text_job_to_cai", side_effect=fake_submit),
        ):
            updated_job, receipt = execute_job_intent(job.job_id)

        self.assertEqual(observed_statuses, ["running"])
        self.assertEqual(observed_phases, ["instance_readiness", "first_response_wait"])
        self.assertEqual(updated_job.execution_attempts[-1]["status"], "completed")
        self.assertEqual(updated_job.execution_attempts[-1]["phase"], "completed")
        self.assertEqual(receipt.output_text, "6")

    def test_execute_job_intent_uses_attempt_timeout_budget_for_submit(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("20000.00000000"))
        job = create_job_intent(
            prompt="2+5=?",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
        )

        instance_snapshot = {
            "instance_id": "instance-timeout-budget",
            "participants": [
                {
                    "node_id": "node-a",
                    "runner_id": "runner-a",
                    "layer_start": 0,
                    "layer_end": 28,
                    "layer_count": 28,
                }
            ],
        }
        payload = {
            "id": "chatcmpl-timeout-budget",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "7"},
                }
            ],
        }

        with (
            patch.dict(
                os.environ,
                {
                    "CAI_ENABLE_TASK_LEVEL_TRANSPORT_JOBS": "0",
                    "CAI_JOB_EXECUTION_TOTAL_TIMEOUT_SECONDS": "60",
                    "CAI_JOB_EXECUTION_ATTEMPT_TIMEOUT_SECONDS": "7",
                },
            ),
            patch(
                "cai_compute_chain.jobs.ensure_cai_instance",
                return_value=instance_snapshot,
            ) as ensure_instance_mock,
            patch(
                "cai_compute_chain.jobs.resolve_cai_instance_snapshot",
                return_value=instance_snapshot,
            ),
            patch(
                "cai_compute_chain.jobs.resolve_cai_command_instance_snapshot",
                return_value=None,
            ),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=self._public_validator_state(),
            ),
            patch(
                "cai_compute_chain.jobs._submit_text_job_to_cai",
                return_value=payload,
            ) as submit_mock,
        ):
            updated_job, receipt = execute_job_intent(job.job_id)

        self.assertEqual(updated_job.status, "completed")
        self.assertEqual(receipt.output_text, "7")
        self.assertLessEqual(
            ensure_instance_mock.call_args.kwargs["ready_timeout_sec"],
            7,
        )
        self.assertEqual(submit_mock.call_args.kwargs["timeout_sec"], 7)
        self.assertEqual(updated_job.execution_attempts[-1]["timeoutSec"], 7)
        self.assertGreaterEqual(
            updated_job.execution_attempts[-1]["attemptDurationMs"],
            0,
        )
        self.assertGreaterEqual(
            updated_job.execution_attempts[-1]["readinessDurationMs"],
            0,
        )
        self.assertGreaterEqual(
            updated_job.execution_attempts[-1]["responseDurationMs"],
            0,
        )
        performance_records = list_execution_performance_records()
        self.assertEqual(len(performance_records), 1)
        self.assertEqual(performance_records[0].executor_node_id, "node-a")
        self.assertEqual(performance_records[0].success_count, 1)
        self.assertEqual(performance_records[0].failure_count, 0)
        self.assertIsNotNone(performance_records[0].avg_response_duration_ms)

    def test_execute_job_intent_retries_after_first_response_timeout_budget(
        self,
    ) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("20000.00000000"))
        job = create_job_intent(
            prompt="2+8=?",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
        )

        snapshot_a = {
            "instance_id": "instance-first-response-a",
            "participants": [
                {
                    "node_id": "node-a",
                    "runner_id": "runner-a",
                    "layer_start": 0,
                    "layer_end": 28,
                    "layer_count": 28,
                }
            ],
        }
        snapshot_b = {
            "instance_id": "instance-first-response-b",
            "participants": [
                {
                    "node_id": "node-b",
                    "runner_id": "runner-b",
                    "layer_start": 0,
                    "layer_end": 28,
                    "layer_count": 28,
                }
            ],
        }
        payload = {
            "id": "chatcmpl-first-response-budget",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "10"},
                }
            ],
        }

        with (
            patch.dict(
                os.environ,
                {
                    "CAI_ENABLE_TASK_LEVEL_TRANSPORT_JOBS": "0",
                    "CAI_JOB_EXECUTION_TOTAL_TIMEOUT_SECONDS": "60",
                    "CAI_JOB_EXECUTION_ATTEMPT_TIMEOUT_SECONDS": "30",
                    "CAI_JOB_EXECUTION_FIRST_RESPONSE_TIMEOUT_SECONDS": "3",
                    "CAI_JOB_EXECUTION_RETRY_BACKOFF_SECONDS": "1",
                },
            ),
            patch(
                "cai_compute_chain.jobs.ensure_cai_instance",
                side_effect=[snapshot_a, snapshot_b],
            ) as ensure_instance_mock,
            patch(
                "cai_compute_chain.jobs.resolve_cai_instance_snapshot",
                side_effect=[snapshot_a, snapshot_b, snapshot_b],
            ),
            patch(
                "cai_compute_chain.jobs.resolve_cai_command_instance_snapshot",
                return_value=None,
            ),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=self._public_validator_state(),
            ),
            patch(
                "cai_compute_chain.jobs._submit_text_job_to_cai",
                side_effect=[
                    TimeoutError("timed out waiting for first response"),
                    payload,
                ],
            ) as submit_mock,
            patch("cai_compute_chain.jobs.cleanup_cai_model_instances") as cleanup_mock,
            patch("cai_compute_chain.jobs.cleanup_orphan_llama_cpp_processes"),
            patch("cai_compute_chain.jobs.time.sleep"),
        ):
            updated_job, receipt = execute_job_intent(job.job_id)

        self.assertEqual(updated_job.status, "completed")
        self.assertEqual(receipt.output_text, "10")
        self.assertEqual(ensure_instance_mock.call_count, 2)
        self.assertEqual(
            ensure_instance_mock.call_args_list[1].kwargs["excluded_node_ids"],
            ["node-a"],
        )
        self.assertEqual(
            [call.kwargs["timeout_sec"] for call in submit_mock.call_args_list],
            [3, 3],
        )
        cleanup_mock.assert_called_once()
        self.assertEqual(
            [item["status"] for item in updated_job.execution_attempts],
            ["retrying", "completed"],
        )
        self.assertEqual(
            [item["timeoutSec"] for item in updated_job.execution_attempts],
            [3, 3],
        )
        self.assertEqual(updated_job.execution_attempts[0]["phase"], "retry_scheduled")
        self.assertEqual(updated_job.execution_attempts[1]["phase"], "completed")
        self.assertEqual(receipt.network_audit["executionAttemptCount"], 2)

    def test_execute_job_intent_jails_validator_when_attestation_fails(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("20000.00000000"))
        self._credit_wallet_on_chain(wallet, coins_to_atomic("20000.00000000"))
        set_validator_static_ip_confirmation(True)
        set_validator_mode(
            True,
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )
        job = create_job_intent(
            prompt="2+2=?",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
        )

        payload = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "4"},
                }
            ],
        }

        with (
            patch("cai_compute_chain.jobs.ensure_cai_instance"),
            patch(
                "cai_compute_chain.jobs.resolve_cai_instance_snapshot",
                return_value={
                    "instance_id": "instance-1",
                    "participants": [
                        {
                            "node_id": "node-a",
                            "runner_id": "runner-a",
                            "layer_start": 0,
                            "layer_end": 28,
                            "layer_count": 28,
                        }
                    ],
                },
            ),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=self._nat_backed_state(),
            ),
            patch(
                "cai_compute_chain.jobs.sync_validator_set_from_cai_peers",
            ),
            patch(
                "cai_compute_chain.settlement.sync_validator_set_from_cai_peers",
            ),
            patch("cai_compute_chain.jobs._submit_text_job_to_cai", return_value=payload),
        ):
            updated_job, receipt = execute_job_intent(job.job_id)

        self.assertEqual(updated_job.status, "completed")
        self.assertEqual(receipt.output_text, "4")
        attestations = list_attestations()
        self.assertEqual(len(attestations), 1)
        self.assertFalse(attestations[0].accepted)
        self.assertIn("behind NAT or a relay", attestations[0].note)

        config = load_or_create_node_config()
        self.assertFalse(config.validator_enabled)
        self.assertEqual(config.validator_state, "jailed")
        self.assertIn("attestation eligibility check", config.validator_jail_reason or "")

    def test_duplicate_accepted_attestation_with_different_note_is_idempotent(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("20000.00000000"))
        self._credit_wallet_on_chain(wallet, coins_to_atomic("20000.00000000"))
        set_validator_static_ip_confirmation(True)
        set_validator_mode(
            True,
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )
        record_validator_attestation(
            settlement_id="settle-conflict",
            validator_id=wallet.address,
            accepted=True,
            note="Local bonded validator accepted settlement.",
        )

        attestation = apply_local_validator_attestation(
            settlement_id="settle-conflict",
            accepted_note="Different conflicting note",
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
            fallback_validator_address=wallet.address,
        )

        self.assertIsNotNone(attestation)
        assert attestation is not None
        self.assertTrue(attestation.accepted)
        config = load_or_create_node_config()
        self.assertEqual(config.validator_state, "bonded")
        self.assertEqual(config.validator_last_slash_atomic, 0)
        self.assertEqual(len(list_validator_evidence()), 0)
        self.assertEqual(len(list_attestations(settlement_id="settle-conflict")), 1)

    def test_conflicting_attestation_jails_validator_with_stronger_slash(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("20000.00000000"))
        self._credit_wallet_on_chain(wallet, coins_to_atomic("20000.00000000"))
        set_validator_static_ip_confirmation(True)
        set_validator_mode(
            True,
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )
        record_validator_attestation(
            settlement_id="settle-conflict",
            validator_id=wallet.address,
            accepted=False,
            note="Local bonded validator rejected settlement.",
        )

        result = apply_local_validator_attestation(
            settlement_id="settle-conflict",
            accepted_note="Local bonded validator accepted settlement.",
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
            fallback_validator_address=wallet.address,
        )

        self.assertIsNone(result)
        config = load_or_create_node_config()
        self.assertEqual(config.validator_state, "jailed")
        self.assertEqual(config.validator_last_slash_atomic, 200_000_000_000)
        evidence = list_validator_evidence()
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].evidence_type, "conflicting_attestation")

    def test_apply_local_validator_attestation_skips_validator_outside_committee(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("20000.00000000"))
        self._credit_wallet_on_chain(wallet, coins_to_atomic("20000.00000000"))
        set_validator_static_ip_confirmation(True)
        set_validator_mode(
            True,
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )

        with patch(
            "cai_compute_chain.jobs.resolve_settlement",
            return_value=SettlementRecord(
                settlement_id="settle-committee",
                created_at="2026-04-22T00:00:00+00:00",
                source_wallet_id=wallet.wallet_id,
                source_wallet_address=wallet.address,
                funding_source="reserve",
                compute_cost_atomic=coins_to_atomic("1.00000000"),
                tx_fee_atomic=0,
                settlement_fee_atomic=0,
                worker_reward_atomic=0,
                committee_selection_seed="settle-committee",
                committee_target_size=1,
                committee_selection_mode="stake_weighted_lottery",
                committee_validator_ids=["some-other-validator"],
                committee_bonded_atomic_by_validator_id={"some-other-validator": coins_to_atomic("10000.00000000")},
                committee_total_bonded_atomic=coins_to_atomic("10000.00000000"),
                committee_quorum_bond_atomic=666_666_666_667,
            ),
        ):
            result = apply_local_validator_attestation(
                settlement_id="settle-committee",
                accepted_note="should be skipped",
                state_payload=self._public_validator_state(),
                cai_url="http://127.0.0.1:52425",
                fallback_validator_address=wallet.address,
            )

        self.assertIsNone(result)
        self.assertEqual(len(list_attestations()), 0)
        self.assertEqual(len(list_validator_evidence()), 0)

    def test_apply_local_validator_attestation_skips_passive_ha_replica(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("20000.00000000"))
        self._credit_wallet_on_chain(wallet, coins_to_atomic("20000.00000000"))
        set_validator_static_ip_confirmation(True)
        set_validator_mode(
            True,
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )
        set_validator_ha_mode(
            enabled=True,
            role="passive",
            replica_id="replica-passive",
            auto_failover=False,
        )

        result = apply_local_validator_attestation(
            settlement_id="settle-passive",
            accepted_note="Passive replica must not sign.",
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
            fallback_validator_address=wallet.address,
        )

        config = load_or_create_node_config()
        self.assertIsNone(result)
        self.assertEqual(len(list_attestations()), 0)
        self.assertEqual(len(list_validator_evidence()), 0)
        self.assertTrue(config.validator_enabled)
        self.assertEqual(config.validator_state, "bonded")
        self.assertEqual(config.validator_last_slash_atomic, 0)

    def test_request_remote_committee_attestations_records_remote_acceptance(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        wallet = credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))
        sync_validator_record(
            validator_id="validator-remote",
            wallet_id=None,
            address="validator-remote",
            state="bonded",
            bonded_atomic=coins_to_atomic("10000.00000000"),
            static_ip_confirmed=True,
            current_node_id="node-remote",
            advertised_api_host="85.137.164.250",
            advertised_data_host="85.137.164.250",
            source="peer",
            source_url="http://85.137.164.250:52415/v1/cai/validators",
        )

        settlement = SettlementRecord(
            settlement_id="settle-remote",
            created_at="2026-04-22T00:00:00+00:00",
            source_wallet_id=wallet.wallet_id,
            source_wallet_address=wallet.address,
            funding_source="reserve",
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            tx_fee_atomic=0,
            settlement_fee_atomic=0,
            worker_reward_atomic=0,
            committee_selection_seed="settle-remote",
            committee_target_size=1,
            committee_selection_mode="stake_weighted_lottery",
            committee_validator_ids=["validator-remote"],
            committee_bonded_atomic_by_validator_id={"validator-remote": coins_to_atomic("10000.00000000")},
            committee_total_bonded_atomic=coins_to_atomic("10000.00000000"),
            committee_quorum_bond_atomic=666_666_666_667,
        )
        save_settlements([settlement])

        with (
            patch(
                "cai_compute_chain.jobs.resolve_settlement",
                return_value=settlement,
            ),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=self._public_validator_state(),
            ),
            patch(
                "cai_compute_chain.jobs.sync_validator_set_from_cai_peers",
            ),
            patch(
                "cai_compute_chain.jobs._post_json",
                return_value={
                    "validatorId": "validator-remote",
                    "attested": True,
                    "ignored": False,
                    "accepted": True,
                    "note": "Remote committee validator accepted settlement.",
                },
            ),
        ):
            responses = request_remote_committee_attestations(
                settlement_id="settle-remote",
                accepted_note="Remote committee validator accepted settlement.",
                cai_url="http://127.0.0.1:52425",
            )

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].validator_id, "validator-remote")
        self.assertTrue(responses[0].accepted)
        self.assertEqual(len(list_attestations()), 1)

    def test_request_remote_committee_attestations_reports_endpoint_failure(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        sync_validator_record(
            validator_id="validator-remote",
            wallet_id=None,
            address="validator-remote",
            state="bonded",
            bonded_atomic=coins_to_atomic("10000.00000000"),
            static_ip_confirmed=True,
            current_node_id="node-remote",
            advertised_api_host="85.137.164.250",
            advertised_data_host="85.137.164.250",
            source="peer",
            source_url="http://85.137.164.250:52415/v1/cai/validators",
        )
        settlement = SettlementRecord(
            settlement_id="settle-remote-failure",
            created_at="2026-04-22T00:00:00+00:00",
            source_wallet_id=wallet.wallet_id,
            source_wallet_address=wallet.address,
            funding_source="reserve",
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            tx_fee_atomic=0,
            settlement_fee_atomic=0,
            worker_reward_atomic=0,
            committee_selection_seed="settle-remote-failure",
            committee_target_size=1,
            committee_selection_mode="stake_weighted_lottery",
            committee_validator_ids=["validator-remote"],
            committee_bonded_atomic_by_validator_id={
                "validator-remote": coins_to_atomic("10000.00000000")
            },
            committee_total_bonded_atomic=coins_to_atomic("10000.00000000"),
            committee_quorum_bond_atomic=666_666_666_667,
        )
        save_settlements([settlement])
        audit: dict[str, object] = {}

        with (
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=self._public_validator_state(),
            ),
            patch("cai_compute_chain.jobs.sync_validator_set_from_cai_peers"),
            patch(
                "cai_compute_chain.jobs._post_json",
                side_effect=OSError("validator endpoint unavailable"),
            ),
            self.assertLogs("cai_compute_chain.jobs", level="WARNING") as logs,
        ):
            responses = request_remote_committee_attestations(
                settlement_id="settle-remote-failure",
                accepted_note="Remote committee validator accepted settlement.",
                cai_url="http://127.0.0.1:52425",
                audit=audit,
            )

        self.assertEqual(responses, [])
        self.assertEqual(audit["status"], "no_attestations")
        self.assertEqual(audit["acceptedResponses"], 0)
        validators = audit["validators"]
        self.assertIsInstance(validators, list)
        assert isinstance(validators, list)
        self.assertEqual(validators[0]["validatorId"], "validator-remote")
        self.assertEqual(validators[0]["status"], "failed")
        self.assertEqual(validators[0]["errorType"], "OSError")
        self.assertIn("validator endpoint unavailable", validators[0]["message"])
        self.assertIn("remote committee validator request failed", "\n".join(logs.output))

    def test_request_remote_committee_attestations_prefers_validator_advertised_host(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        wallet = credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))
        sync_validator_record(
            validator_id="validator-remote",
            wallet_id=None,
            address="validator-remote",
            state="bonded",
            bonded_atomic=coins_to_atomic("10000.00000000"),
            static_ip_confirmed=True,
            current_node_id="node-remote",
            advertised_api_host="85.137.164.252",
            advertised_data_host="85.137.164.252",
            source="peer",
            source_url="http://85.137.164.251:52415/v1/cai/validators",
        )

        settlement = SettlementRecord(
            settlement_id="settle-remote-direct",
            created_at="2026-04-22T00:00:00+00:00",
            source_wallet_id=wallet.wallet_id,
            source_wallet_address=wallet.address,
            funding_source="reserve",
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            tx_fee_atomic=0,
            settlement_fee_atomic=0,
            worker_reward_atomic=0,
            committee_selection_seed="settle-remote-direct",
            committee_target_size=1,
            committee_selection_mode="stake_weighted_lottery",
            committee_validator_ids=["validator-remote"],
            committee_bonded_atomic_by_validator_id={"validator-remote": coins_to_atomic("10000.00000000")},
            committee_total_bonded_atomic=coins_to_atomic("10000.00000000"),
            committee_quorum_bond_atomic=666_666_666_667,
        )
        save_settlements([settlement])

        def fake_post_json(url: str, payload: dict[str, object], *, timeout: int) -> dict[str, object]:
            self.assertEqual(
                url,
                "http://85.137.164.252:52415/v1/cai/settlement/attest",
            )
            self.assertEqual(payload["settlement_id"], "settle-remote-direct")
            return {
                "validatorId": "validator-remote",
                "attested": True,
                "ignored": False,
                "accepted": True,
                "note": "Remote committee validator accepted settlement.",
            }

        with (
            patch(
                "cai_compute_chain.jobs.resolve_settlement",
                return_value=settlement,
            ),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=self._public_validator_state(),
            ),
            patch(
                "cai_compute_chain.jobs.sync_validator_set_from_cai_peers",
            ),
            patch(
                "cai_compute_chain.jobs._post_json",
                side_effect=fake_post_json,
            ),
        ):
            responses = request_remote_committee_attestations(
                settlement_id="settle-remote-direct",
                accepted_note="Remote committee validator accepted settlement.",
                cai_url="http://127.0.0.1:52425",
            )

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].validator_id, "validator-remote")
        self.assertTrue(responses[0].accepted)

    def test_remote_attestation_without_chain_does_not_apply_local_reward(self) -> None:
        source_wallet = create_wallet("source", "testpass", select=True)
        unlock_wallet("testpass")
        source_wallet = credit_wallet(
            source_wallet.wallet_id,
            coins_to_atomic("2.00000000"),
        )
        worker_wallet = create_wallet("worker", "testpass", select=False)
        bind_worker_reward_address("node-worker", worker_wallet.address)
        sync_validator_record(
            validator_id="validator-remote",
            wallet_id=None,
            address="validator-remote",
            state="bonded",
            bonded_atomic=coins_to_atomic("10000.00000000"),
            static_ip_confirmed=True,
            current_node_id="node-remote",
            advertised_api_host="85.137.164.250",
            advertised_data_host="85.137.164.250",
            source="peer",
            source_url="http://85.137.164.250:52415/v1/cai/validators",
        )
        decision = plan_funding(
            ledger=load_or_create_ledger(MoneyPolicy()),
            wallet=source_wallet,
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=MoneyPolicy(),
        )
        settlement = record_funding_settlement(
            source_wallet_id=source_wallet.wallet_id,
            source_wallet_address=source_wallet.address,
            decision=decision,
            note="remote accepted without chain",
            money_policy=MoneyPolicy(),
        )
        payouts = record_worker_payouts(
            settlement_id=settlement.settlement_id,
            receipt_id="receipt-no-chain",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            participants=[
                {
                    "node_id": "node-worker",
                    "runner_id": "runner-worker",
                    "layer_start": 0,
                    "layer_end": 1,
                    "layer_count": 1,
                    "share_bps": 10000,
                    "reward_atomic": settlement.worker_reward_atomic,
                }
            ],
            money_policy=MoneyPolicy(),
        )
        sign_settlement_envelope(settlement.settlement_id, money_policy=MoneyPolicy())

        def fake_post_json(url: str, payload: dict[str, object], *, timeout: int) -> dict[str, object]:
            self.assertIn("settlement_proposal", payload)
            proposal = payload["settlement_proposal"]
            self.assertIsInstance(proposal, dict)
            assert isinstance(proposal, dict)
            self.assertEqual(
                proposal["settlement"]["settlement_id"],
                settlement.settlement_id,
            )
            self.assertEqual(
                proposal["worker_payouts"][0]["payout_id"],
                payouts[0].payout_id,
            )
            return {
                "validatorId": "validator-remote",
                "attested": True,
                "ignored": False,
                "accepted": True,
                "note": "Remote committee validator accepted settlement.",
            }

        with (
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=self._public_validator_state(),
            ),
            patch(
                "cai_compute_chain.jobs.sync_validator_set_from_cai_peers",
            ),
            patch(
                "cai_compute_chain.jobs._post_json",
                side_effect=fake_post_json,
            ),
        ):
            responses = request_remote_committee_attestations(
                settlement_id=settlement.settlement_id,
                accepted_note="Remote committee validator accepted settlement.",
                cai_url="http://127.0.0.1:52425",
            )

        self.assertEqual(len(responses), 1)
        self.assertEqual(list_settlements()[0].status, "finalized")
        self.assertEqual(chain_settlement_history(settlement.settlement_id), [])
        self.assertEqual(load_or_create_ledger().settlements_applied, 0)
        refreshed_worker = next(
            item for item in list_wallets() if item.wallet_id == worker_wallet.wallet_id
        )
        self.assertEqual(refreshed_worker.spendable_balance_atomic, 0)
        self.assertEqual(
            list_worker_payouts(settlement_id=settlement.settlement_id)[0].status,
            "pending_settlement",
        )

    def test_repair_local_worker_reward_state_waits_for_remote_canonical_chain(
        self,
    ) -> None:
        money_policy = MoneyPolicy()
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        wallet = credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))
        self._credit_wallet_on_chain(wallet, coins_to_atomic("2.00000000"))
        bind_worker_reward_address("node-local", wallet.address)

        decision = plan_funding(
            ledger=load_or_create_ledger(money_policy),
            wallet=wallet,
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=money_policy,
        )
        settlement = record_funding_settlement(
            source_wallet_id=wallet.wallet_id,
            source_wallet_address=wallet.address,
            decision=decision,
            note="repair pending worker payout",
            money_policy=money_policy,
        )
        self.assertEqual(settlement.status, "pending")
        self.assertEqual(settlement.committee_validator_ids, [])

        record_worker_payouts(
            settlement_id=settlement.settlement_id,
            receipt_id="receipt-repair-1",
            model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            participants=[
                {
                    "node_id": "node-local",
                    "runner_id": "runner-local",
                    "layer_start": 0,
                    "layer_end": 28,
                    "layer_count": 28,
                    "share_bps": 10000,
                    "reward_atomic": settlement.worker_reward_atomic,
                    "note": "Distributed by pipeline layer share.",
                }
            ],
        )
        payout_records = list_worker_payouts(settlement_id=settlement.settlement_id)
        record_settlement_execution_audit(
            settlement_id=settlement.settlement_id,
            receipt_id="receipt-repair-1",
            job_id="job-repair-1",
            model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            execution_model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            pricing_mode="network_auto",
            pricing_basis="llm_tokens",
            prompt_tokens=8,
            completion_tokens=4,
            total_tokens=12,
            reserved_prompt_tokens=4,
            reserved_completion_tokens=32,
            reserved_compute_cost_atomic=settlement.compute_cost_atomic,
            actual_compute_cost_atomic=settlement.compute_cost_atomic,
            usage_priced=True,
            token_usage_source="response_usage",
            token_usage_audit={"schema_version": 1, "source": "response_usage"},
            network_audit={"participantCount": 1},
            worker_payouts=payout_records,
        )
        self.assertEqual(list_worker_payouts()[0].status, "pending_settlement")

        def fake_sync_validator_set_from_cai_peers(**kwargs):
            sync_validator_record(
                validator_id="validator-remote",
                wallet_id=None,
                address="validator-remote",
                state="bonded",
                bonded_atomic=coins_to_atomic("10000.00000000"),
                static_ip_confirmed=True,
                current_node_id="node-remote",
                advertised_api_host="85.137.164.250",
                advertised_data_host="85.137.164.250",
                source="peer",
                source_url="http://85.137.164.250:52415/v1/cai/validators",
            )
            return None

        with (
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=self._distributed_two_worker_state(),
            ),
            patch(
                "cai_compute_chain.jobs.sync_validator_set_from_cai_peers",
                side_effect=fake_sync_validator_set_from_cai_peers,
            ),
            patch(
                "cai_compute_chain.jobs._post_json",
                return_value={
                    "validatorId": "validator-remote",
                    "attested": True,
                    "ignored": False,
                    "accepted": True,
                    "note": "Remote committee validator accepted settlement.",
                },
            ),
        ):
            result = repair_local_worker_reward_state(
                cai_url="http://127.0.0.1:52425",
                money_policy=money_policy,
                state_payload=self._public_validator_state(),
            )

        self.assertEqual(result["committeeBackfilled"], 1)
        self.assertEqual(result["remoteAttestations"], 1)

        repaired_settlement = list_settlements()[0]
        self.assertEqual(repaired_settlement.status, "finalized")
        self.assertEqual(
            repaired_settlement.committee_validator_ids,
            ["validator-remote"],
        )

        payout = list_worker_payouts()[0]
        self.assertEqual(payout.status, "pending_settlement")
        self.assertIsNone(payout.credited_wallet_id)

        refreshed_wallet = next(
            item for item in list_wallets() if item.wallet_id == wallet.wallet_id
        )
        self.assertEqual(
            refreshed_wallet.spendable_balance_atomic,
            coins_to_atomic("2.00000000"),
        )

    def test_repair_local_worker_reward_state_reports_peer_sync_errors(self) -> None:
        with (
            patch(
                "cai_compute_chain.jobs.sync_validator_set_from_cai_peers",
                side_effect=OSError("validator sync unavailable"),
            ),
            patch(
                "cai_compute_chain.jobs.sync_chain_from_cai_peers",
                side_effect=RuntimeError("chain sync unavailable"),
            ),
        ):
            result = repair_local_worker_reward_state(
                cai_url="http://127.0.0.1:52425",
                money_policy=MoneyPolicy(),
                state_payload={"nodeIdentities": {}},
            )

        peer_sync = result["peerSync"]
        self.assertTrue(peer_sync["attempted"])
        self.assertEqual(peer_sync["validatorSet"]["status"], "failed")
        self.assertEqual(peer_sync["validatorSet"]["errorType"], "OSError")
        self.assertIn(
            "validator sync unavailable",
            peer_sync["validatorSet"]["message"],
        )
        self.assertEqual(peer_sync["chain"]["status"], "failed")
        self.assertEqual(peer_sync["chain"]["errorType"], "RuntimeError")
        self.assertIn("chain sync unavailable", peer_sync["chain"]["message"])

    def test_repair_local_worker_reward_state_skips_unsigned_pending_settlement(
        self,
    ) -> None:
        money_policy = MoneyPolicy()
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        wallet = credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))
        self._credit_wallet_on_chain(wallet, coins_to_atomic("2.00000000"))
        bind_worker_reward_address("node-local", wallet.address)

        decision = plan_funding(
            ledger=load_or_create_ledger(money_policy),
            wallet=wallet,
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=money_policy,
        )
        settlement = record_funding_settlement(
            source_wallet_id=wallet.wallet_id,
            source_wallet_address=wallet.address,
            decision=decision,
            note="repair should skip unsigned settlement",
            money_policy=money_policy,
        )
        self.assertEqual(settlement.status, "pending")
        self.assertEqual(settlement.committee_validator_ids, [])

        record_worker_payouts(
            settlement_id=settlement.settlement_id,
            receipt_id="receipt-repair-skip-1",
            model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            participants=[
                {
                    "node_id": "node-local",
                    "runner_id": "runner-local",
                    "layer_start": 0,
                    "layer_end": 28,
                    "layer_count": 28,
                    "share_bps": 10000,
                    "reward_atomic": settlement.worker_reward_atomic,
                    "note": "Distributed by pipeline layer share.",
                }
            ],
        )

        def fake_sync_validator_set_from_cai_peers(**kwargs):
            sync_validator_record(
                validator_id="validator-remote",
                wallet_id=None,
                address="validator-remote",
                state="bonded",
                bonded_atomic=coins_to_atomic("10000.00000000"),
                static_ip_confirmed=True,
                current_node_id="node-remote",
                advertised_api_host="85.137.164.250",
                advertised_data_host="85.137.164.250",
                source="peer",
                source_url="http://85.137.164.250:52415/v1/cai/validators",
            )
            return None

        with (
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=self._distributed_two_worker_state(),
            ),
            patch(
                "cai_compute_chain.jobs.sync_validator_set_from_cai_peers",
                side_effect=fake_sync_validator_set_from_cai_peers,
            ),
            patch(
                "cai_compute_chain.jobs.request_remote_committee_attestations",
                return_value=[],
            ) as remote_attestations,
        ):
            result = repair_local_worker_reward_state(
                cai_url="http://127.0.0.1:52425",
                money_policy=money_policy,
                state_payload=self._public_validator_state(),
            )

        self.assertEqual(result["committeeBackfilled"], 1)
        self.assertEqual(result["remoteAttestations"], 0)
        self.assertEqual(remote_attestations.call_count, 0)

        repaired_settlement = list_settlements()[0]
        self.assertEqual(repaired_settlement.status, "pending")
        self.assertEqual(
            repaired_settlement.committee_validator_ids,
            ["validator-remote"],
        )
        self.assertEqual(
            list_attestations(settlement_id=settlement.settlement_id),
            [],
        )

    def test_repair_local_worker_reward_state_attests_with_restored_local_validator(
        self,
    ) -> None:
        money_policy = MoneyPolicy()
        wallet = create_wallet("validator", "testpass", select=True)
        unlock_wallet("testpass")
        wallet = credit_wallet(wallet.wallet_id, coins_to_atomic("10002.00000000"))
        self._credit_wallet_on_chain(wallet, coins_to_atomic("10002.00000000"))
        bind_worker_reward_address("node-local", wallet.address)
        set_validator_static_ip_confirmation(True)
        config = set_validator_mode(
            True,
            money_policy=money_policy,
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )

        decision = plan_funding(
            ledger=load_or_create_ledger(money_policy),
            wallet=next(
                item for item in list_wallets() if item.wallet_id == wallet.wallet_id
            ),
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=money_policy,
        )
        settlement = record_funding_settlement(
            source_wallet_id=wallet.wallet_id,
            source_wallet_address=wallet.address,
            decision=decision,
            note="repair local validator settlement",
            money_policy=money_policy,
        )
        self.assertEqual(settlement.status, "pending")
        self.assertEqual(settlement.committee_validator_ids, [config.validator_address])

        record_worker_payouts(
            settlement_id=settlement.settlement_id,
            receipt_id="receipt-local-repair-1",
            model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            participants=[
                {
                    "node_id": "node-local",
                    "runner_id": "runner-local",
                    "layer_start": 0,
                    "layer_end": 28,
                    "layer_count": 28,
                    "share_bps": 10000,
                    "reward_atomic": settlement.worker_reward_atomic,
                    "note": "Distributed by pipeline layer share.",
                }
            ],
        )
        payout_records = list_worker_payouts(settlement_id=settlement.settlement_id)
        record_settlement_execution_audit(
            settlement_id=settlement.settlement_id,
            receipt_id="receipt-local-repair-1",
            job_id="job-local-repair-1",
            model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            execution_model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            pricing_mode="network_auto",
            pricing_basis="llm_tokens",
            prompt_tokens=8,
            completion_tokens=4,
            total_tokens=12,
            reserved_prompt_tokens=4,
            reserved_completion_tokens=32,
            reserved_compute_cost_atomic=settlement.compute_cost_atomic,
            actual_compute_cost_atomic=settlement.compute_cost_atomic,
            usage_priced=True,
            token_usage_source="response_usage",
            token_usage_audit={"schema_version": 1, "source": "response_usage"},
            network_audit={"participantCount": 1},
            worker_payouts=payout_records,
        )
        self.assertEqual(
            list_attestations(settlement_id=settlement.settlement_id),
            [],
        )

        with (
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=self._public_validator_state(),
            ),
            patch(
                "cai_compute_chain.jobs.request_remote_committee_attestations",
                side_effect=AssertionError(
                    "local validator should settle without remote attestation"
                ),
            ),
        ):
            result = repair_local_worker_reward_state(
                cai_url="http://127.0.0.1:52425",
                money_policy=money_policy,
                state_payload=self._public_validator_state(),
            )

        self.assertEqual(result["localAttestations"], 1)
        self.assertEqual(result["remoteAttestations"], 0)

        repaired_settlement = list_settlements()[0]
        self.assertEqual(repaired_settlement.status, "applied")
        self.assertEqual(repaired_settlement.accepted_attestations, 1)

        payout = list_worker_payouts()[0]
        self.assertEqual(payout.status, "credited_local_wallet")
        self.assertEqual(payout.credited_wallet_id, wallet.wallet_id)

    def _public_validator_state(self) -> dict:
        return {
            "nodeIdentities": {
                "node-public": {
                    "apiHost": "85.137.164.250",
                    "apiPort": 52425,
                    "dataHost": "85.137.164.250",
                    "dataPort": 52435,
                }
            },
            "nodeNetwork": {
                "node-public": {
                    "interfaces": [
                        {"name": "eth0", "ipAddress": "85.137.164.250"},
                        {"name": "lo", "ipAddress": "127.0.0.1"},
                    ]
                }
            },
        }

    def _nat_backed_state(self) -> dict:
        return {
            "nodeIdentities": {
                "node-nat": {
                    "apiHost": "26.242.160.75",
                    "apiPort": 52425,
                    "dataHost": "26.242.160.75",
                    "dataPort": 52435,
                }
            },
            "nodeNetwork": {
                "node-nat": {
                    "interfaces": [
                        {"name": "eth0", "ipAddress": "172.25.27.64"},
                        {"name": "lo", "ipAddress": "127.0.0.1"},
                    ]
                }
            },
        }

    def test_execute_job_intent_resolves_network_model_to_execution_model(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))
        job = create_job_intent(
            prompt="2+2=?",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
        )
        payload = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "4"},
                }
            ],
        }

        with (
            patch("cai_compute_chain.jobs.ensure_cai_instance") as ensure_instance,
            patch(
                "cai_compute_chain.jobs.resolve_cai_instance_snapshot",
                return_value={
                    "instance_id": "instance-1",
                    "participants": [
                        {
                            "node_id": "node-worker",
                            "runner_id": "runner-worker",
                            "layer_start": 0,
                            "layer_end": 28,
                            "layer_count": 28,
                        }
                    ],
                },
            ),
            patch("cai_compute_chain.jobs._submit_text_job_to_cai", return_value=payload) as submit_job,
        ):
            _, receipt = execute_job_intent(job.job_id)

        ensure_instance.assert_called_once_with(
            "http://127.0.0.1:52425",
            "Qwen/Qwen3-0.6B-GGUF",
            ready_timeout_sec=600,
            private_network_model=True,
            requester_node_id=None,
        )
        submit_job.assert_called_once_with(
            "http://127.0.0.1:52425",
            "Qwen/Qwen3-0.6B-GGUF",
            "2+2=?",
            timeout_sec=ANY,
            request_payload_override=None,
        )
        self.assertEqual(receipt.model_id, "cai-network/Qwen3-0.6B-GGUF")
        self.assertEqual(receipt.execution_model_id, "Qwen/Qwen3-0.6B-GGUF")
        self.assertEqual(len(receipt.worker_payouts), 1)

    def test_execute_job_intent_requires_worker_participants_for_reward(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))
        job = create_job_intent(
            prompt="2+2=?",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
        )
        payload = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "4"},
                }
            ],
        }

        with (
            patch("cai_compute_chain.jobs.ensure_cai_instance"),
            patch(
                "cai_compute_chain.jobs.resolve_cai_instance_snapshot",
                return_value={"instance_id": "instance-1", "participants": []},
            ),
            patch("cai_compute_chain.jobs._submit_text_job_to_cai", return_value=payload),
        ):
            with self.assertRaisesRegex(RuntimeError, "Cannot settle execution reward"):
                execute_job_intent(job.job_id)

        persisted_job = next(item for item in list_job_intents() if item.job_id == job.job_id)
        self.assertEqual(persisted_job.status, "failed")
        self.assertIn("worker participants", persisted_job.last_error or "")
        self.assertEqual(list_execution_receipts(), [])
        self.assertEqual(list_settlements(), [])
        self.assertEqual(list_worker_payouts(), [])

        from cai_compute_chain.wallet import get_active_wallet, load_or_create_ledger

        active_wallet = get_active_wallet()
        ledger = load_or_create_ledger()
        self.assertIsNotNone(active_wallet)
        self.assertEqual(active_wallet.spendable_balance_atomic, coins_to_atomic("2.00000000"))
        self.assertEqual(ledger.worker_distributed_atomic, 0)

    def test_execute_job_intent_refuses_planned_snapshot_for_settlement(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))
        job = create_job_intent(
            prompt="2+2=?",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
        )
        planned_snapshot = {
            "instance_id": "planned:MlxRingInstance",
            "snapshot_source": "planned_definition",
            "participants": [
                {
                    "node_id": "node-remote",
                    "runner_id": "runner-remote",
                    "layer_start": 0,
                    "layer_end": 28,
                    "layer_count": 28,
                }
            ],
        }
        payload = {
            "id": "chatcmpl-planned",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "4"},
                }
            ],
        }

        with (
            patch("cai_compute_chain.jobs.ensure_cai_instance", return_value=planned_snapshot),
            patch("cai_compute_chain.jobs.resolve_cai_instance_snapshot", return_value=None),
            patch("cai_compute_chain.jobs.resolve_cai_command_instance_snapshot", return_value=None),
            patch("cai_compute_chain.jobs._submit_text_job_to_cai", return_value=payload),
            patch("cai_compute_chain.jobs.cleanup_cai_model_instances"),
        ):
            with self.assertRaisesRegex(RuntimeError, "planned placement snapshot"):
                execute_job_intent(job.job_id)

        persisted_job = next(item for item in list_job_intents() if item.job_id == job.job_id)
        self.assertEqual(persisted_job.status, "failed")
        self.assertIn("planned placement snapshot", persisted_job.last_error or "")
        self.assertEqual(list_execution_receipts(), [])
        self.assertEqual(list_settlements(), [])
        self.assertEqual(list_worker_payouts(), [])

    def test_execute_job_intent_rejects_explicitly_disabled_worker_participant(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))
        job = create_job_intent(
            prompt="disabled worker",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
        )
        payload = {
            "id": "chatcmpl-disabled-worker",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
        }
        state_payload = {
            "nodeIdentities": {
                "node-a": {
                    "workerEnabled": False,
                    "workerRewardAddress": "ABCD1234ABCD1234ABCD1234ABCD1234",
                    "workerAllowedModelIds": ["cai-network/Qwen3-0.6B-GGUF"],
                }
            },
            "topology": {"nodes": ["node-a"], "connections": {}},
        }

        with (
            patch("cai_compute_chain.jobs.ensure_cai_instance"),
            patch(
                "cai_compute_chain.jobs.resolve_cai_instance_snapshot",
                return_value={
                    "instance_id": "instance-disabled-worker",
                    "participants": [
                        {
                            "node_id": "node-a",
                            "runner_id": "runner-a",
                            "layer_start": 0,
                            "layer_end": 28,
                            "layer_count": 28,
                        }
                    ],
                },
            ),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=state_payload,
            ),
            patch("cai_compute_chain.jobs._submit_text_job_to_cai", return_value=payload),
        ):
            with self.assertRaisesRegex(RuntimeError, "worker participant eligibility"):
                execute_job_intent(job.job_id)

        persisted_job = next(item for item in list_job_intents() if item.job_id == job.job_id)
        self.assertEqual(persisted_job.status, "failed")
        self.assertIn("worker mode is explicitly disabled", persisted_job.last_error or "")
        self.assertEqual(list_execution_receipts(), [])
        self.assertEqual(list_settlements(), [])
        self.assertEqual(list_worker_payouts(), [])

    def test_execute_job_intent_rejects_worker_participant_with_disallowed_model(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))
        job = create_job_intent(
            prompt="wrong model",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
        )
        payload = {
            "id": "chatcmpl-disallowed-model",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
        }
        state_payload = {
            "nodeIdentities": {
                "node-a": {
                    "workerEnabled": True,
                    "workerRewardAddress": "ABCD1234ABCD1234ABCD1234ABCD1234",
                    "workerAllowedModelIds": ["cai-network/Other-Model-GGUF"],
                }
            },
            "topology": {"nodes": ["node-a"], "connections": {}},
        }

        with (
            patch("cai_compute_chain.jobs.ensure_cai_instance"),
            patch(
                "cai_compute_chain.jobs.resolve_cai_instance_snapshot",
                return_value={
                    "instance_id": "instance-disallowed-model",
                    "participants": [
                        {
                            "node_id": "node-a",
                            "runner_id": "runner-a",
                            "layer_start": 0,
                            "layer_end": 28,
                            "layer_count": 28,
                        }
                    ],
                },
            ),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=state_payload,
            ),
            patch("cai_compute_chain.jobs._submit_text_job_to_cai", return_value=payload),
        ):
            with self.assertRaisesRegex(RuntimeError, "worker participant eligibility"):
                execute_job_intent(job.job_id)

        persisted_job = next(item for item in list_job_intents() if item.job_id == job.job_id)
        self.assertEqual(persisted_job.status, "failed")
        self.assertIn("allowed model ids", persisted_job.last_error or "")
        self.assertEqual(list_execution_receipts(), [])
        self.assertEqual(list_settlements(), [])
        self.assertEqual(list_worker_payouts(), [])

    def test_execute_job_intent_rejects_stale_worker_identity(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))
        job = create_job_intent(
            prompt="stale worker",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
        )
        payload = {
            "id": "chatcmpl-stale-worker",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
        }
        state_payload = {
            "nodeIdentities": {
                "node-a": {
                    "workerEnabled": True,
                    "workerRewardAddress": "ABCD1234ABCD1234ABCD1234ABCD1234",
                    "workerAllowedModelIds": ["cai-network/Qwen3-0.6B-GGUF"],
                    "lastSeenAt": "2020-01-01T00:00:00+00:00",
                }
            },
            "topology": {"nodes": ["node-a"], "connections": {}},
        }

        with (
            patch("cai_compute_chain.jobs.ensure_cai_instance"),
            patch(
                "cai_compute_chain.jobs.resolve_cai_instance_snapshot",
                return_value={
                    "instance_id": "instance-stale-worker",
                    "participants": [
                        {
                            "node_id": "node-a",
                            "runner_id": "runner-a",
                            "layer_start": 0,
                            "layer_end": 28,
                            "layer_count": 28,
                        }
                    ],
                },
            ),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=state_payload,
            ),
            patch("cai_compute_chain.jobs._submit_text_job_to_cai", return_value=payload),
        ):
            with self.assertRaisesRegex(RuntimeError, "worker participant eligibility"):
                execute_job_intent(job.job_id)

        persisted_job = next(item for item in list_job_intents() if item.job_id == job.job_id)
        self.assertEqual(persisted_job.status, "failed")
        self.assertIn("worker identity is stale", persisted_job.last_error or "")
        self.assertEqual(list_execution_receipts(), [])
        self.assertEqual(list_settlements(), [])
        self.assertEqual(list_worker_payouts(), [])

    def test_sync_worker_reward_bindings_from_cai_summary(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-worker": {
                    "apiHost": "198.51.100.10",
                    "apiPort": 52415,
                }
            }
        }
        summary_payload = {
            "worker": {
                "worker_enabled": True,
                "worker_reward_address": "ABCD1234ABCD1234ABCD1234ABCD1234",
            }
        }

        with (
            patch("cai_compute_chain.jobs._load_cai_state_payload", return_value=state_payload),
            patch("cai_compute_chain.jobs._get_json", return_value=summary_payload),
        ):
            _sync_worker_reward_bindings_from_cai(
                "http://127.0.0.1:52415",
                [{"node_id": "node-worker"}],
            )

        self.assertEqual(
            resolve_worker_reward_address("node-worker"),
            "abcd1234abcd1234abcd1234abcd1234",
        )

    def test_sync_worker_reward_bindings_from_cai_logs_state_failure(self) -> None:
        with (
            patch(
                "cai_compute_chain.jobs.urlopen",
                side_effect=OSError("state endpoint unavailable"),
            ),
            self.assertLogs("cai_compute_chain.jobs", level="WARNING") as logs,
        ):
            _sync_worker_reward_bindings_from_cai(
                "http://127.0.0.1:52415",
                [{"node_id": "node-worker"}],
            )

        self.assertIn(
            "worker reward binding CAI state fetch failed",
            "\n".join(logs.output),
        )
        self.assertIn("state endpoint unavailable", "\n".join(logs.output))

    def test_sync_worker_reward_bindings_from_cai_logs_summary_failure(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-worker": {
                    "apiHost": "198.51.100.10",
                    "apiPort": 52415,
                }
            }
        }

        with (
            patch("cai_compute_chain.jobs._load_cai_state_payload", return_value=state_payload),
            patch(
                "cai_compute_chain.jobs._get_json",
                side_effect=TimeoutError("summary endpoint timed out"),
            ),
            self.assertLogs("cai_compute_chain.jobs", level="WARNING") as logs,
        ):
            _sync_worker_reward_bindings_from_cai(
                "http://127.0.0.1:52415",
                [{"node_id": "node-worker"}],
            )

        self.assertIn(
            "worker reward binding summary fetch for node node-worker failed",
            "\n".join(logs.output),
        )
        self.assertIn("summary endpoint timed out", "\n".join(logs.output))

    def test_submit_text_job_to_cai_disables_thinking_by_default(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return b'{"id":"chatcmpl-test","choices":[]}'

        captured: dict[str, object] = {}

        def fake_urlopen(request, timeout=0):
            captured["timeout"] = timeout
            captured["url"] = request.full_url
            captured["payload"] = request.data.decode("utf-8")
            return FakeResponse()

        with patch("cai_compute_chain.jobs.urlopen", side_effect=fake_urlopen):
            _submit_text_job_to_cai(
                "http://127.0.0.1:52425",
                "Qwen/Qwen3-0.6B-GGUF",
                "2+2=?",
                timeout_sec=123,
            )

        self.assertEqual(captured["timeout"], 123)
        self.assertEqual(
            captured["url"], "http://127.0.0.1:52425/v1/chat/completions"
        )
        self.assertIn('"enable_thinking": false', captured["payload"])
        self.assertIn('"reasoning_effort": "none"', captured["payload"])
        self.assertNotIn('"max_tokens"', captured["payload"])

    def test_submit_text_job_to_cai_preserves_request_payload_override(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return b'{"id":"chatcmpl-test","choices":[]}'

        captured: dict[str, object] = {}

        def fake_urlopen(request, timeout=0):
            captured["timeout"] = timeout
            captured["payload"] = request.data.decode("utf-8")
            return FakeResponse()

        with patch("cai_compute_chain.jobs.urlopen", side_effect=fake_urlopen):
            _submit_text_job_to_cai(
                "http://127.0.0.1:52425",
                "Qwen/Qwen3-0.6B-GGUF",
                "2+2=?",
                timeout_sec=45,
                request_payload_override={
                    "model": "should-be-overridden",
                    "messages": [
                        {"role": "system", "content": "Be precise."},
                        {"role": "user", "content": "2+2=?"},
                    ],
                    "stream": True,
                    "enable_thinking": True,
                    "temperature": 0.2,
                },
            )

        self.assertEqual(captured["timeout"], 45)
        self.assertIn('"model": "Qwen/Qwen3-0.6B-GGUF"', captured["payload"])
        self.assertIn('"stream": false', captured["payload"])
        self.assertIn('"enable_thinking": true', captured["payload"])
        self.assertIn('"temperature": 0.2', captured["payload"])
        self.assertIn('"role": "system"', captured["payload"])

    def test_submit_text_job_to_cai_retries_known_peer_api_hosts_after_404(self) -> None:
        class FakeResponse:
            def __init__(self, payload: dict[str, object]) -> None:
                self._payload = payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return json.dumps(self._payload).encode("utf-8")

        attempted_urls: list[str] = []

        def fake_urlopen(request, timeout=0):
            attempted_urls.append(request.full_url)
            if request.full_url == "http://127.0.0.1:52415/v1/chat/completions":
                raise HTTPError(
                    request.full_url,
                    404,
                    "",
                    hdrs=None,
                    fp=None,
                )
            return FakeResponse(
                {
                    "id": "chatcmpl-peer",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "4"},
                        }
                    ],
                }
            )

        with (
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value={
                    "nodeIdentities": {
                        "node-local": {
                            "apiHost": "127.0.0.1",
                            "apiPort": 52415,
                        },
                        "node-peer": {
                            "apiHost": "85.137.164.250",
                            "apiPort": 52415,
                        },
                    }
                },
            ),
            patch("cai_compute_chain.jobs.urlopen", side_effect=fake_urlopen),
        ):
            payload = _submit_text_job_to_cai(
                "http://127.0.0.1:52415",
                "model-x",
                "2+2=?",
            )

        self.assertEqual(payload["choices"][0]["message"]["content"], "4")
        self.assertEqual(
            attempted_urls,
            [
                "http://127.0.0.1:52415/v1/chat/completions",
                "http://85.137.164.250:52415/v1/chat/completions",
            ],
        )

    def test_submit_text_job_to_cai_raises_clear_error_for_empty_success_body(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return b""

        with patch("cai_compute_chain.jobs.urlopen", return_value=FakeResponse()):
            with self.assertRaises(RuntimeError) as exc:
                _submit_text_job_to_cai(
                    "http://127.0.0.1:52425",
                    "Qwen/Qwen3-0.6B-GGUF",
                    "5+23=",
                )

        self.assertIn("returned an empty response body", str(exc.exception))

    def test_submit_text_job_to_cai_surfaces_http_error_message_payload(self) -> None:
        def fake_urlopen(request, timeout=0):
            raise HTTPError(
                request.full_url,
                503,
                "",
                hdrs=None,
                fp=None,
            )

        with (
            patch("cai_compute_chain.jobs.urlopen", side_effect=fake_urlopen),
            patch(
                "cai_compute_chain.jobs._http_error_detail",
                return_value="No output chunks were received from the runner before the response stream ended",
            ),
        ):
            with self.assertRaises(RuntimeError) as exc:
                _submit_text_job_to_cai(
                    "http://127.0.0.1:52425",
                    "Qwen/Qwen3-0.6B-GGUF",
                    "5+23=",
                )

        self.assertIn("No output chunks were received from the runner", str(exc.exception))

    def test_resolve_cai_instance_create_payload_filters_to_worker_enabled_nodes(self) -> None:
        queried_urls: list[str] = []

        def fake_get_json(url: str, *, timeout: int = 30):
            queried_urls.append(url)
            if url == "http://127.0.0.1:52425/state/nodeIdentities":
                return state_payload["nodeIdentities"]
            if url == "http://127.0.0.1:52425/v1/cai/summary":
                return {
                    "worker": {
                        "worker_enabled": True,
                        "worker_reward_address": "ABCD1234ABCD1234ABCD1234ABCD1234",
                    }
                }
            if url == "http://85.137.164.250:52415/v1/cai/summary":
                return {"worker": {"worker_enabled": False}}
            if url.startswith("http://127.0.0.1:52425/instance/previews?"):
                self.assertIn("node_ids=node-local", url)
                self.assertNotIn("node_ids=node-validator", url)
                return {
                    "previews": [
                        {
                            "instance": {
                                "mlxRing": {
                                    "instanceId": "instance-worker-only",
                                    "shardAssignments": {
                                        "modelId": "cai-network/Qwen3-0.6B-GGUF",
                                        "nodeToRunner": {"node-local": "runner-local"},
                                    },
                                }
                            }
                        }
                    ]
                }
            raise AssertionError(f"Unexpected URL: {url}")

        state_payload = {
            "nodeIdentities": {
                "node-local": {
                    "apiHost": None,
                    "apiPort": 52425,
                },
                "node-validator": {
                    "apiHost": "85.137.164.250",
                    "apiPort": 52415,
                },
            }
        }

        with (
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=state_payload,
            ),
            patch("cai_compute_chain.jobs._get_json", side_effect=fake_get_json),
        ):
            payload = _resolve_cai_instance_create_payload(
                "http://127.0.0.1:52425",
                "cai-network/Qwen3-0.6B-GGUF",
            )

        self.assertEqual(
            payload["instance"]["mlxRing"]["instanceId"],
            "instance-worker-only",
        )
        self.assertIn("http://127.0.0.1:52425/v1/cai/summary", queried_urls)
        self.assertIn("http://85.137.164.250:52415/v1/cai/summary", queried_urls)

    def test_resolve_cai_instance_create_payload_uses_state_worker_metadata(self) -> None:
        def fake_get_json(url: str, *, timeout: int = 30):
            if url == "http://127.0.0.1:52425/state/nodeIdentities":
                return state_payload["nodeIdentities"]
            if url.startswith("http://127.0.0.1:52425/instance/previews?"):
                self.assertIn("node_ids=node-remote", url)
                self.assertNotIn("node_ids=node-local", url)
                return {
                    "previews": [
                        {
                            "instance": {
                                "mlxRing": {
                                    "instanceId": "instance-remote-worker",
                                    "shardAssignments": {
                                        "modelId": "cai-network/Qwen3-0.6B-GGUF",
                                        "nodeToRunner": {"node-remote": "runner-remote"},
                                    },
                                }
                            }
                        }
                    ]
                }
            raise AssertionError(f"Unexpected URL: {url}")

        state_payload = {
            "nodeIdentities": {
                "node-local": {
                    "apiHost": None,
                    "apiPort": 52425,
                    "workerEnabled": False,
                },
                "node-remote": {
                    "apiHost": "198.51.100.20",
                    "apiPort": 52415,
                    "workerEnabled": True,
                    "workerRewardAddress": "ABCD1234ABCD1234ABCD1234ABCD1234",
                    "workerAllowedModelIds": ["cai-network/Qwen3-0.6B-GGUF"],
                },
            }
        }

        with (
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=state_payload,
            ),
            patch("cai_compute_chain.jobs._get_json", side_effect=fake_get_json),
        ):
            payload = _resolve_cai_instance_create_payload(
                "http://127.0.0.1:52425",
                "cai-network/Qwen3-0.6B-GGUF",
            )

        self.assertEqual(
            payload["instance"]["mlxRing"]["instanceId"],
            "instance-remote-worker",
        )

    def test_resolve_cai_instance_create_payload_rejects_when_no_worker_nodes_available(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-validator": {
                    "apiHost": "85.137.164.250",
                    "apiPort": 52415,
                }
            }
        }

        def fake_get_json(url: str, *, timeout: int = 30):
            if url == "http://127.0.0.1:52425/state/nodeIdentities":
                return state_payload["nodeIdentities"]
            return {"worker": {"worker_enabled": False}}

        with (
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=state_payload,
            ),
            patch(
                "cai_compute_chain.jobs._get_json",
                side_effect=fake_get_json,
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "No worker-enabled CAI nodes are currently available",
            ):
                _resolve_cai_instance_create_payload(
                    "http://127.0.0.1:52425",
                    "Qwen/Qwen3-0.6B-GGUF",
                )

    def test_resolve_cai_instance_create_payload_allows_single_private_worker(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-worker": {
                    "apiHost": "198.51.100.20",
                    "apiPort": 52415,
                    "workerEnabled": True,
                    "workerRewardAddress": "ABCD1234ABCD1234ABCD1234ABCD1234",
                }
            }
        }

        def fake_get_json(url: str, *, timeout: int = 30):
            if url == "http://127.0.0.1:52425/state/nodeIdentities":
                return state_payload["nodeIdentities"]
            if url.startswith("http://127.0.0.1:52425/instance/previews?"):
                return {
                    "previews": [
                        {
                            "instance": {
                                "MlxRingInstance": {
                                    "shardAssignments": {
                                        "nodeToRunner": {"node-worker": "runner-1"}
                                    }
                                }
                            }
                        }
                    ]
                }
            raise AssertionError(f"Unexpected URL: {url}")

        with (
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=state_payload,
            ),
            patch(
                "cai_compute_chain.jobs._get_json",
                side_effect=fake_get_json,
            ),
        ):
            payload = _resolve_cai_instance_create_payload(
                "http://127.0.0.1:52425",
                "Qwen/Qwen3-0.6B-GGUF",
                private_network_model=True,
            )

        assert payload["instance"]["MlxRingInstance"]["shardAssignments"][
            "nodeToRunner"
        ] == {"node-worker": "runner-1"}

    def test_resolve_cai_instance_create_payload_rejects_worker_model_allow_list_mismatch(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-worker": {
                    "apiHost": "198.51.100.20",
                    "apiPort": 52415,
                    "workerEnabled": True,
                    "workerRewardAddress": "ABCD1234ABCD1234ABCD1234ABCD1234",
                    "workerAllowedModelIds": ["cai-network/Other-Model-GGUF"],
                }
            }
        }

        def fake_get_json(url: str, *, timeout: int = 30):
            if url == "http://127.0.0.1:52425/state/nodeIdentities":
                return state_payload["nodeIdentities"]
            raise AssertionError(f"Unexpected URL: {url}")

        with (
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=state_payload,
            ),
            patch(
                "cai_compute_chain.jobs._get_json",
                side_effect=fake_get_json,
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "model is not in worker allow-list",
            ):
                _resolve_cai_instance_create_payload(
                    "http://127.0.0.1:52425",
                    "cai-network/Qwen3-0.6B-GGUF",
                )

    def test_resolve_cai_instance_create_payload_rejects_stale_worker_identity(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-worker": {
                    "apiHost": "198.51.100.20",
                    "apiPort": 52415,
                    "workerEnabled": True,
                    "workerRewardAddress": "ABCD1234ABCD1234ABCD1234ABCD1234",
                    "workerAllowedModelIds": ["cai-network/Qwen3-0.6B-GGUF"],
                    "lastSeenAt": "2000-01-01T00:00:00+00:00",
                }
            }
        }

        def fake_get_json(url: str, *, timeout: int = 30):
            if url == "http://127.0.0.1:52425/state/nodeIdentities":
                return state_payload["nodeIdentities"]
            raise AssertionError(f"Unexpected URL: {url}")

        with (
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=state_payload,
            ),
            patch(
                "cai_compute_chain.jobs._get_json",
                side_effect=fake_get_json,
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "worker identity is stale",
            ):
                _resolve_cai_instance_create_payload(
                    "http://127.0.0.1:52425",
                    "Qwen/Qwen3-0.6B-GGUF",
                )

    def test_sync_worker_reward_bindings_from_cai_uses_state_metadata(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-remote": {
                    "workerEnabled": True,
                    "workerRewardAddress": "ABCD1234ABCD1234ABCD1234ABCD1234",
                }
            }
        }

        with (
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=state_payload,
            ),
            patch("cai_compute_chain.jobs._get_json", side_effect=AssertionError("HTTP fallback should not run")),
        ):
            _sync_worker_reward_bindings_from_cai(
                "http://127.0.0.1:52425",
                [{"node_id": "node-remote", "reward_atomic": 1}],
            )

        self.assertEqual(
            resolve_worker_reward_address("node-remote"),
            "abcd1234abcd1234abcd1234abcd1234",
        )

    def test_resolve_cai_instance_create_payload_rejects_worker_without_reward_address(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-validator": {
                    "apiHost": "85.137.164.250",
                    "apiPort": 52415,
                }
            }
        }

        def fake_get_json(url: str, *, timeout: int = 30):
            if url == "http://127.0.0.1:52425/state/nodeIdentities":
                return state_payload["nodeIdentities"]
            return {"worker": {"worker_enabled": True}}

        with (
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=state_payload,
            ),
            patch(
                "cai_compute_chain.jobs._get_json",
                side_effect=fake_get_json,
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "No worker-enabled CAI nodes are currently available",
            ):
                _resolve_cai_instance_create_payload(
                    "http://127.0.0.1:52425",
                    "Qwen/Qwen3-0.6B-GGUF",
                )

    def test_distribute_worker_reward_uses_layer_share(self) -> None:
        payouts = _distribute_worker_reward(
            coins_to_atomic("0.93000000"),
            [
                {
                    "node_id": "node-a",
                    "runner_id": "runner-a",
                    "layer_start": 0,
                    "layer_end": 21,
                    "layer_count": 21,
                },
                {
                    "node_id": "node-b",
                    "runner_id": "runner-b",
                    "layer_start": 21,
                    "layer_end": 28,
                    "layer_count": 7,
                },
            ],
        )

        self.assertEqual(len(payouts), 2)
        self.assertEqual(sum(item["reward_atomic"] for item in payouts), coins_to_atomic("0.93000000"))
        self.assertGreater(payouts[0]["reward_atomic"], payouts[1]["reward_atomic"])

    def test_ensure_cai_instance_creates_instance_when_missing(self) -> None:
        state_instances = [
            {},
            {},
            {
                "iid-1": {
                    "MlxRingInstance": {
                        "shardAssignments": {"modelId": "Qwen/Qwen3-0.6B-GGUF"}
                    }
                }
            },
        ]
        state_root = [
            {
                "instances": {
                    "iid-1": {
                        "MlxRingInstance": {
                            "shardAssignments": {"modelId": "Qwen/Qwen3-0.6B-GGUF", "runnerToShard": {"runner-a": {}}}
                        }
                    }
                },
                "runners": {"runner-a": {"RunnerReady": {}}},
            }
        ]
        previews = {
            "previews": [
                {
                    "instance": {"MlxRingInstance": {"shardAssignments": {"modelId": "Qwen/Qwen3-0.6B-GGUF"}}},
                    "error": None,
                }
            ]
        }

        def fake_get_json(url: str, *, timeout: int = 30):
            if "/instance/previews" in url:
                return previews
            if url.endswith("/state/instances"):
                return state_instances.pop(0) if state_instances else {
                    "iid-1": {
                        "MlxRingInstance": {
                            "shardAssignments": {"modelId": "Qwen/Qwen3-0.6B-GGUF"}
                        }
                    }
                }
            if url.endswith("/state"):
                return state_root.pop(0)
            raise AssertionError(f"Unexpected URL {url}")

        with (
            patch("cai_compute_chain.jobs._get_json", side_effect=fake_get_json),
            patch("cai_compute_chain.jobs._post_json", return_value={"message": "ok"}) as post_json,
            patch("cai_compute_chain.jobs.time.sleep"),
        ):
            ensure_cai_instance("http://127.0.0.1:52425", "Qwen/Qwen3-0.6B-GGUF")

        post_json.assert_called_once()

    def test_ensure_cai_instance_uses_private_placement_query_when_requested(self) -> None:
        state_instances = [
            {},
            {"iid-1": {"MlxRingInstance": {"shardAssignments": {"modelId": "Qwen/Qwen3-0.6B-GGUF"}}}},
        ]
        state_root = [
            {
                "instances": {
                    "iid-1": {
                        "MlxRingInstance": {
                            "shardAssignments": {
                                "modelId": "Qwen/Qwen3-0.6B-GGUF",
                                "runnerToShard": {"runner-a": {}},
                            }
                        }
                    }
                },
                "runners": {"runner-a": {"RunnerReady": {}}},
            }
        ]
        seen_urls: list[str] = []

        def fake_get_json(url: str, *, timeout: int = 30):
            seen_urls.append(url)
            if "/instance/previews" in url:
                return {
                    "previews": [
                        {
                            "instance": {
                                "MlxRingInstance": {
                                    "shardAssignments": {
                                        "modelId": "Qwen/Qwen3-0.6B-GGUF"
                                    }
                                }
                            },
                            "error": None,
                        }
                    ]
                }
            if url.endswith("/state/instances"):
                return state_instances.pop(0) if state_instances else {
                    "iid-1": {
                        "MlxRingInstance": {
                            "shardAssignments": {"modelId": "Qwen/Qwen3-0.6B-GGUF"}
                        }
                    }
                }
            if url.endswith("/state"):
                return state_root.pop(0)
            raise AssertionError(f"Unexpected URL {url}")

        with (
            patch("cai_compute_chain.jobs._get_json", side_effect=fake_get_json),
            patch("cai_compute_chain.jobs._post_json", return_value={"message": "ok"}),
            patch("cai_compute_chain.jobs.time.sleep"),
        ):
            ensure_cai_instance(
                "http://127.0.0.1:52425",
                "Qwen/Qwen3-0.6B-GGUF",
                private_network_model=True,
            )

        self.assertTrue(
            any("private_network_model=true" in url for url in seen_urls),
            seen_urls,
        )

    def test_ensure_cai_instance_prefers_single_node_preview_for_public_models(self) -> None:
        state_instances = [
            {},
            {},
            {
                "iid-1": {
                    "MlxRingInstance": {
                        "shardAssignments": {"modelId": "Qwen/Qwen3-0.6B-GGUF"}
                    }
                }
            },
        ]
        state_root = [
            {
                "instances": {
                    "iid-1": {
                        "MlxRingInstance": {
                            "shardAssignments": {
                                "modelId": "Qwen/Qwen3-0.6B-GGUF",
                                "runnerToShard": {
                                    "runner-local": {},
                                    "runner-remote": {},
                                },
                            }
                        }
                    }
                },
                "runners": {
                    "runner-local": {"RunnerReady": {}},
                    "runner-remote": {"RunnerReady": {}},
                },
            }
        ]
        single_node_instance = {
            "MlxRingInstance": {
                "shardAssignments": {
                    "modelId": "Qwen/Qwen3-0.6B-GGUF",
                    "nodeToRunner": {"node-local": "runner-local"},
                }
            }
        }
        multi_node_instance = {
            "MlxRingInstance": {
                "shardAssignments": {
                    "modelId": "Qwen/Qwen3-0.6B-GGUF",
                    "nodeToRunner": {
                        "node-local": "runner-local",
                        "node-remote": "runner-remote",
                    },
                }
            }
        }

        def fake_get_json(url: str, *, timeout: int = 30):
            if "/instance/previews" in url:
                return {
                    "previews": [
                        {"instance": single_node_instance, "error": None},
                        {"instance": multi_node_instance, "error": None},
                    ]
                }
            if url.endswith("/state/instances"):
                return state_instances.pop(0) if state_instances else {
                    "iid-1": {
                        "MlxRingInstance": {
                            "shardAssignments": {"modelId": "Qwen/Qwen3-0.6B-GGUF"}
                        }
                    }
                }
            if url.endswith("/state"):
                return state_root.pop(0)
            raise AssertionError(f"Unexpected URL {url}")

        with (
            patch("cai_compute_chain.jobs._get_json", side_effect=fake_get_json),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value={"topology": {"nodes": ["node-local", "node-remote"]}},
            ),
            patch("cai_compute_chain.jobs._post_json", return_value={"message": "ok"}) as post_json,
            patch("cai_compute_chain.jobs.time.sleep"),
        ):
            ensure_cai_instance("http://127.0.0.1:52425", "Qwen/Qwen3-0.6B-GGUF")

        post_json.assert_called_once_with(
            "http://127.0.0.1:52425/instance",
            {"instance": single_node_instance},
            timeout=180,
        )

    def test_ensure_cai_instance_prefers_remote_worker_for_public_models(self) -> None:
        state_instances = [
            {},
            {},
            {
                "iid-remote": {
                    "MlxRingInstance": {
                        "shardAssignments": {"modelId": "Qwen/Qwen3-0.6B-GGUF"}
                    }
                }
            },
        ]
        state_root = [
            {
                "instances": {
                    "iid-remote": {
                        "MlxRingInstance": {
                            "shardAssignments": {
                                "modelId": "Qwen/Qwen3-0.6B-GGUF",
                                "runnerToShard": {"runner-remote": {}},
                            }
                        }
                    }
                },
                "runners": {"runner-remote": {"RunnerReady": {}}},
            }
        ]
        state_payload = {
            "topology": {"nodes": ["node-local", "node-remote"]},
            "nodeIdentities": {
                "node-local": {
                    "apiHost": "26.242.160.75",
                    "apiPort": 52425,
                    "workerEnabled": True,
                    "workerRewardAddress": "local-reward",
                },
                "node-remote": {
                    "apiHost": "26.97.29.153",
                    "apiPort": 52425,
                    "workerEnabled": True,
                    "workerRewardAddress": "remote-reward",
                },
            },
        }
        remote_instance = {
            "MlxRingInstance": {
                "shardAssignments": {
                    "modelId": "Qwen/Qwen3-0.6B-GGUF",
                    "nodeToRunner": {"node-remote": "runner-remote"},
                }
            }
        }
        seen_preview_urls: list[str] = []

        def fake_get_json(url: str, *, timeout: int = 30):
            if url.endswith("/state/nodeIdentities"):
                return state_payload["nodeIdentities"]
            if "/instance/previews" in url:
                seen_preview_urls.append(url)
                return {"previews": [{"instance": remote_instance, "error": None}]}
            if url.endswith("/state/instances"):
                return state_instances.pop(0) if state_instances else {
                    "iid-remote": {
                        "MlxRingInstance": {
                            "shardAssignments": {"modelId": "Qwen/Qwen3-0.6B-GGUF"}
                        }
                    }
                }
            if url.endswith("/state"):
                return state_root.pop(0)
            raise AssertionError(f"Unexpected URL {url}")

        with (
            patch("cai_compute_chain.jobs._get_json", side_effect=fake_get_json),
            patch("cai_compute_chain.jobs._load_cai_state_payload", return_value=state_payload),
            patch("cai_compute_chain.jobs._post_json", return_value={"message": "ok"}) as post_json,
            patch("cai_compute_chain.jobs.time.sleep"),
        ):
            ensure_cai_instance(
                "http://127.0.0.1:52425",
                "Qwen/Qwen3-0.6B-GGUF",
                requester_node_id="node-local",
            )

        self.assertTrue(seen_preview_urls)
        self.assertTrue(all("node_ids=node-remote" in url for url in seen_preview_urls))
        self.assertTrue(all("node_ids=node-local" not in url for url in seen_preview_urls))
        post_json.assert_called_once_with(
            "http://127.0.0.1:52425/instance",
            {"instance": remote_instance},
            timeout=180,
        )

    def test_ensure_cai_instance_falls_back_when_remote_worker_preview_fails(self) -> None:
        state_instances = [
            {},
            {},
            {
                "iid-local": {
                    "MlxRingInstance": {
                        "shardAssignments": {"modelId": "Qwen/Qwen3-0.6B-GGUF"}
                    }
                }
            },
        ]
        state_root = [
            {
                "instances": {
                    "iid-local": {
                        "MlxRingInstance": {
                            "shardAssignments": {
                                "modelId": "Qwen/Qwen3-0.6B-GGUF",
                                "runnerToShard": {"runner-local": {}},
                            }
                        }
                    }
                },
                "runners": {"runner-local": {"RunnerReady": {}}},
            }
        ]
        state_payload = {
            "topology": {"nodes": ["node-local", "node-remote"]},
            "nodeIdentities": {
                "node-local": {
                    "apiHost": "127.0.0.1",
                    "apiPort": 52425,
                    "workerEnabled": True,
                    "workerRewardAddress": "local-reward",
                },
                "node-remote": {
                    "apiHost": "26.97.29.153",
                    "apiPort": 52425,
                    "workerEnabled": True,
                    "workerRewardAddress": "remote-reward",
                },
            },
        }
        local_instance = {
            "MlxRingInstance": {
                "shardAssignments": {
                    "modelId": "Qwen/Qwen3-0.6B-GGUF",
                    "nodeToRunner": {"node-local": "runner-local"},
                }
            }
        }
        preview_urls: list[str] = []

        def fake_get_json(url: str, *, timeout: int = 30):
            if url.endswith("/state/nodeIdentities"):
                return state_payload["nodeIdentities"]
            if "/instance/previews" in url:
                preview_urls.append(url)
                if "node_ids=node-remote" in url and "node_ids=node-local" not in url:
                    return {"previews": []}
                return {"previews": [{"instance": local_instance, "error": None}]}
            if "/instance/placement" in url and "node_ids=node-remote" in url:
                raise HTTPError(url, 400, "Bad Request", {}, None)
            if url.endswith("/state/instances"):
                return state_instances.pop(0) if state_instances else {
                    "iid-local": {
                        "MlxRingInstance": {
                            "shardAssignments": {"modelId": "Qwen/Qwen3-0.6B-GGUF"}
                        }
                    }
                }
            if url.endswith("/state"):
                return state_root.pop(0)
            raise AssertionError(f"Unexpected URL {url}")

        with (
            patch("cai_compute_chain.jobs._get_json", side_effect=fake_get_json),
            patch("cai_compute_chain.jobs._load_cai_state_payload", return_value=state_payload),
            patch("cai_compute_chain.jobs._post_json", return_value={"message": "ok"}) as post_json,
            patch("cai_compute_chain.jobs.time.sleep"),
        ):
            ensure_cai_instance("http://127.0.0.1:52425", "Qwen/Qwen3-0.6B-GGUF")

        self.assertGreaterEqual(len(preview_urls), 2)
        self.assertIn("node_ids=node-remote", preview_urls[0])
        self.assertIn("node_ids=node-local", preview_urls[-1])
        post_json.assert_called_once_with(
            "http://127.0.0.1:52425/instance",
            {"instance": local_instance},
            timeout=180,
        )

    def test_ensure_cai_instance_waits_for_multi_node_preview_when_private_model_requires_it(self) -> None:
        state_instances = [
            {},
            {},
            {
                "iid-1": {
                    "MlxRingInstance": {
                        "shardAssignments": {"modelId": "Qwen/Qwen3-0.6B-GGUF"}
                    }
                }
            },
        ]
        state_root = [
            {
                "topology": {"nodes": ["node-local", "node-remote"]},
                "instances": {
                    "iid-1": {
                        "MlxRingInstance": {
                            "shardAssignments": {
                                "modelId": "Qwen/Qwen3-0.6B-GGUF",
                                "runnerToShard": {
                                    "runner-local": {},
                                    "runner-remote": {},
                                },
                            }
                        }
                    }
                },
                "runners": {
                    "runner-local": {"RunnerReady": {}},
                    "runner-remote": {"RunnerReady": {}},
                },
            }
        ]
        single_node_instance = {
            "MlxRingInstance": {
                "shardAssignments": {
                    "modelId": "Qwen/Qwen3-0.6B-GGUF",
                    "nodeToRunner": {"node-local": "runner-local"},
                }
            }
        }
        multi_node_instance = {
            "MlxRingInstance": {
                "shardAssignments": {
                    "modelId": "Qwen/Qwen3-0.6B-GGUF",
                    "nodeToRunner": {
                        "node-local": "runner-local",
                        "node-remote": "runner-remote",
                    },
                }
            }
        }
        preview_payloads = [
            {"previews": [{"instance": single_node_instance, "error": None}]},
            {"previews": [{"instance": multi_node_instance, "error": None}]},
        ]

        def fake_get_json(url: str, *, timeout: int = 30):
            if "/instance/previews" in url:
                if preview_payloads:
                    return preview_payloads.pop(0)
                return {"previews": [{"instance": multi_node_instance, "error": None}]}
            if url.endswith("/state/instances"):
                return state_instances.pop(0) if state_instances else {
                    "iid-1": {
                        "MlxRingInstance": {
                            "shardAssignments": {"modelId": "Qwen/Qwen3-0.6B-GGUF"}
                        }
                    }
                }
            if url.endswith("/state"):
                return state_root.pop(0)
            raise AssertionError(f"Unexpected URL {url}")

        with (
            patch("cai_compute_chain.jobs._get_json", side_effect=fake_get_json),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value={"topology": {"nodes": ["node-local", "node-remote"]}},
            ),
            patch("cai_compute_chain.jobs._post_json", return_value={"message": "ok"}) as post_json,
            patch("cai_compute_chain.jobs.time.sleep"),
        ):
            ensure_cai_instance(
                "http://127.0.0.1:52425",
                "Qwen/Qwen3-0.6B-GGUF",
                private_network_model=True,
            )

        post_json.assert_called_once_with(
            "http://127.0.0.1:52425/instance",
            {"instance": multi_node_instance},
            timeout=180,
        )

    def test_private_instance_create_payload_falls_back_to_minimum_worker_subset(
        self,
    ) -> None:
        state_payload = {"nodeIdentities": {}}
        selected_instance = {
            "MlxRingInstance": {
                "shardAssignments": {
                    "modelId": "Qwen/Qwen3-0.6B-GGUF",
                    "nodeToRunner": {
                        "node-a": "runner-a",
                        "node-b": "runner-b",
                    },
                }
            }
        }
        preview_urls: list[str] = []

        def fake_get_json(url: str, *, timeout: int = 30):
            if "/instance/previews" in url:
                preview_urls.append(url)
                if (
                    "node_ids=node-a" in url
                    and "node_ids=node-b" in url
                    and "node_ids=node-c" not in url
                ):
                    return {"previews": [{"instance": selected_instance, "error": None}]}
                return {"previews": []}
            if "/instance/placement" in url:
                raise HTTPError(url, 400, "Bad Request", {}, None)
            raise AssertionError(f"Unexpected URL {url}")

        with (
            patch(
                "cai_compute_chain.jobs._resolve_worker_execution_node_audit",
                return_value={
                    "schemaVersion": 1,
                    "modelId": "Qwen/Qwen3-0.6B-GGUF",
                    "checkedNodeCount": 3,
                    "eligibleNodeIds": ["node-a", "node-b", "node-c"],
                    "nodes": [],
                },
            ),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=state_payload,
            ),
            patch("cai_compute_chain.jobs._get_json", side_effect=fake_get_json),
        ):
            payload = _resolve_cai_instance_create_payload(
                "http://127.0.0.1:52425",
                "Qwen/Qwen3-0.6B-GGUF",
                private_network_model=True,
            )

        self.assertEqual(payload["instance"], selected_instance)
        self.assertIn("node_ids=node-a", preview_urls[0])
        self.assertIn("node_ids=node-b", preview_urls[0])
        self.assertIn("node_ids=node-c", preview_urls[0])
        self.assertIn("node_ids=node-a", preview_urls[-1])
        self.assertIn("node_ids=node-b", preview_urls[-1])
        self.assertNotIn("node_ids=node-c", preview_urls[-1])

    def test_private_execution_node_attempts_include_minimum_worker_subsets(
        self,
    ) -> None:
        attempts = _execution_node_id_attempts(
            ["node-a", "node-b", "node-c"],
            state_payload={},
            cai_url="http://127.0.0.1:52425",
            private_network_model=True,
        )

        self.assertEqual(attempts[0], ["node-a", "node-b", "node-c"])
        self.assertEqual(attempts[1], ["node-a"])
        self.assertIn(["node-b"], attempts)
        self.assertIn(["node-c"], attempts)
        self.assertIn(["node-a", "node-b"], attempts)
        self.assertIn(["node-a", "node-c"], attempts)
        self.assertIn(["node-b", "node-c"], attempts)

    def test_execution_node_id_attempts_prefer_excluding_failed_node_first(
        self,
    ) -> None:
        attempts = _execution_node_id_attempts(
            ["node-local", "node-a", "node-b"],
            state_payload={},
            cai_url="http://127.0.0.1:52425",
            private_network_model=False,
            requester_node_id="node-local",
            excluded_node_ids=["node-a"],
        )

        self.assertEqual(attempts[0], ["node-b"])
        self.assertEqual(attempts[1], ["node-local", "node-b"])
        self.assertIn(["node-a", "node-b"], attempts)
        self.assertIn(["node-local", "node-a", "node-b"], attempts)

    def test_execution_node_id_attempts_fall_back_when_all_nodes_excluded(
        self,
    ) -> None:
        attempts = _execution_node_id_attempts(
            ["node-a"],
            state_payload={},
            cai_url="http://127.0.0.1:52425",
            private_network_model=False,
            excluded_node_ids=["node-a"],
        )

        self.assertEqual(attempts, [["node-a"]])

    def test_ensure_cai_instance_raises_timeout_when_instance_is_visible_but_not_ready(self) -> None:
        state_instances = [
            {},
            {
                "iid-1": {
                    "MlxRingInstance": {
                        "shardAssignments": {"modelId": "Qwen/Qwen3-0.6B-GGUF"}
                    }
                }
            },
        ]
        planned_instance = {
            "MlxRingInstance": {
                "instanceId": "planned-instance",
                "shardAssignments": {
                    "modelId": "Qwen/Qwen3-0.6B-GGUF",
                    "nodeToRunner": {"node-remote": "runner-remote"},
                },
            }
        }

        def fake_get_json(url: str, *, timeout: int = 30):
            if "/instance/previews" in url:
                return {"previews": [{"instance": planned_instance, "error": None}]}
            if url.endswith("/state/instances"):
                return state_instances.pop(0) if state_instances else {
                    "iid-1": {
                        "MlxRingInstance": {
                            "shardAssignments": {"modelId": "Qwen/Qwen3-0.6B-GGUF"}
                        }
                    }
                }
            raise AssertionError(f"Unexpected URL {url}")

        state_payload = {
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
        }

        with (
            patch("cai_compute_chain.jobs._get_json", side_effect=fake_get_json),
            patch(
                "cai_compute_chain.jobs._load_cai_state_payload",
                return_value=state_payload,
            ),
            patch("cai_compute_chain.jobs._wait_for_cai_instance_ready", return_value=False),
            patch("cai_compute_chain.jobs._post_json", return_value={"message": "ok"}),
            patch("cai_compute_chain.jobs.time.sleep"),
        ):
            with self.assertRaisesRegex(
                TimeoutError,
                "Model shards are still downloading on 1 node\\(s\\): DESKTOP-REMOTE",
            ) as exc:
                ensure_cai_instance("http://127.0.0.1:52425", "Qwen/Qwen3-0.6B-GGUF")
        self.assertIn("Readiness stage=download", str(exc.exception))
        self.assertIn("Next stage=materialize", str(exc.exception))

    def test_cai_instance_readiness_audit_reports_shard_loading(self) -> None:
        state_payload = {
            "downloads": {
                "node-remote": [
                    {
                        "DownloadOngoing": {
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
            "nodeIdentities": {
                "node-remote": {"friendlyName": "DESKTOP-REMOTE"},
            },
        }

        with patch(
            "cai_compute_chain.jobs._load_cai_state_payload",
            return_value=state_payload,
        ):
            audit = cai_instance_readiness_audit(
                "http://127.0.0.1:52425",
                "cai-network/Qwen3-0.6B-GGUF",
            )

        self.assertEqual(audit["status"], "shard_loading")
        self.assertEqual(audit["pendingDownloadNodes"], ["DESKTOP-REMOTE"])
        self.assertEqual(audit["currentStage"], "download")
        self.assertEqual(audit["nextStage"], "materialize")
        self.assertEqual(
            audit["readinessState"]["stageOrder"],
            ["download", "materialize", "load", "rpc_ready", "inference_ready"],
        )
        self.assertEqual(audit["readinessState"]["stages"][0]["status"], "current")

    def test_cai_instance_readiness_audit_reports_model_materializing(self) -> None:
        state_payload = {
            "downloads": {
                "node-remote": [
                    {
                        "DownloadCompleted": {
                            "shardMetadata": {
                                "PipelineShardMetadata": {
                                    "modelCard": {
                                        "modelId": "Qwen/Qwen3-0.6B-GGUF"
                                    }
                                }
                            }
                        }
                    }
                ]
            },
            "nodeIdentities": {
                "node-remote": {"friendlyName": "DESKTOP-REMOTE"},
            },
        }

        with patch(
            "cai_compute_chain.jobs._load_cai_state_payload",
            return_value=state_payload,
        ):
            audit = cai_instance_readiness_audit(
                "http://127.0.0.1:52425",
                "Qwen/Qwen3-0.6B-GGUF",
            )

        self.assertEqual(audit["status"], "model_materializing")
        self.assertFalse(audit["ready"])
        self.assertEqual(audit["currentStage"], "materialize")
        self.assertEqual(audit["nextStage"], "load")
        self.assertEqual(audit["completedDownloadNodes"], ["DESKTOP-REMOTE"])
        self.assertEqual(audit["readinessState"]["completedStages"], ["download"])

    def test_cai_instance_readiness_audit_reports_inference_ready(self) -> None:
        state_payload = {
            "instances": {
                "iid-1": {
                    "MlxRingInstance": {
                        "shardAssignments": {
                            "modelId": "Qwen/Qwen3-0.6B-GGUF",
                            "runnerToShard": {
                                "runner-a": {},
                                "runner-b": {},
                            },
                        }
                    }
                }
            },
            "runners": {
                "runner-a": {"RunnerReady": {}},
                "runner-b": {"RunnerRunning": {}},
            },
        }

        with patch(
            "cai_compute_chain.jobs._load_cai_state_payload",
            return_value=state_payload,
        ):
            audit = cai_instance_readiness_audit(
                "http://127.0.0.1:52425",
                "Qwen/Qwen3-0.6B-GGUF",
            )

        self.assertEqual(audit["status"], "inference_ready")
        self.assertTrue(audit["ready"])
        self.assertEqual(audit["currentStage"], "inference_ready")
        self.assertIsNone(audit["nextStage"])
        self.assertEqual(
            audit["readinessState"]["completedStages"],
            ["download", "materialize", "load", "rpc_ready", "inference_ready"],
        )
        self.assertEqual(
            [item["status"] for item in audit["runners"]],
            ["RunnerReady", "RunnerRunning"],
        )

    def test_cai_instance_readiness_audit_reports_cai_owned_route_blocked(self) -> None:
        shard_a = {
            "PipelineShardMetadata": {
                "modelCard": {"modelId": "Qwen/Qwen3-0.6B-GGUF"},
                "startLayer": 0,
                "endLayer": 14,
                "nLayers": 28,
            }
        }
        shard_b = {
            "PipelineShardMetadata": {
                "modelCard": {"modelId": "Qwen/Qwen3-0.6B-GGUF"},
                "startLayer": 14,
                "endLayer": 28,
                "nLayers": 28,
            }
        }
        state_payload = {
            "instances": {
                "iid-1": {
                    "MlxRingInstance": {
                        "shardAssignments": {
                            "modelId": "Qwen/Qwen3-0.6B-GGUF",
                            "nodeToRunner": {
                                "node-a": "runner-a",
                                "node-b": "runner-b",
                            },
                            "runnerToShard": {
                                "runner-a": shard_a,
                                "runner-b": shard_b,
                            },
                        }
                    }
                }
            },
            "runners": {
                "runner-a": {"RunnerStarting": {}},
                "runner-b": {"RunnerStarting": {}},
            },
            "topology": {"connections": {}},
        }

        with patch(
            "cai_compute_chain.jobs._load_cai_state_payload",
            return_value=state_payload,
        ):
            audit = cai_instance_readiness_audit(
                "http://127.0.0.1:52425",
                "Qwen/Qwen3-0.6B-GGUF",
            )

        self.assertEqual(audit["status"], "cai_owned_route_blocked")
        self.assertEqual(audit["currentStage"], "rpc_ready")
        self.assertEqual(audit["readinessState"]["blockedStages"], ["rpc_ready"])
        self.assertIn("CAI-owned transport is required", audit["reason"])
        self.assertEqual(
            audit["networkAudit"]["transportMode"],
            "multi_worker_disconnected",
        )

    def test_cai_instance_readiness_audit_reports_rpc_ready(self) -> None:
        shard_a = {
            "PipelineShardMetadata": {
                "modelCard": {"modelId": "Qwen/Qwen3-0.6B-GGUF"},
                "startLayer": 0,
                "endLayer": 14,
                "nLayers": 28,
            }
        }
        shard_b = {
            "PipelineShardMetadata": {
                "modelCard": {"modelId": "Qwen/Qwen3-0.6B-GGUF"},
                "startLayer": 14,
                "endLayer": 28,
                "nLayers": 28,
            }
        }
        state_payload = {
            "instances": {
                "iid-1": {
                    "MlxRingInstance": {
                        "shardAssignments": {
                            "modelId": "Qwen/Qwen3-0.6B-GGUF",
                            "nodeToRunner": {
                                "node-a": "runner-a",
                                "node-b": "runner-b",
                            },
                            "runnerToShard": {
                                "runner-a": shard_a,
                                "runner-b": shard_b,
                            },
                        }
                    }
                }
            },
            "runners": {
                "runner-a": {"RunnerStarting": {}},
                "runner-b": {"RunnerStarting": {}},
            },
            "topology": {"connections": {}},
        }
        route_health = [
            RouteHealthRecord(
                route_id="node-a->node-b",
                source_node_id="node-a",
                sink_node_id="node-b",
                route_type="llama_cpp_rpc_direct",
                endpoint_url="llama-cpp-rpc://198.51.100.11:52435",
                reachable=True,
                checked_at="2026-05-03T00:00:00+00:00",
                latency_ms=7.0,
            )
        ]

        with patch(
            "cai_compute_chain.jobs._load_cai_state_payload",
            return_value=state_payload,
        ), patch(
            "cai_compute_chain.jobs.list_route_health_records",
            return_value=route_health,
        ):
            audit = cai_instance_readiness_audit(
                "http://127.0.0.1:52425",
                "Qwen/Qwen3-0.6B-GGUF",
            )

        self.assertEqual(audit["status"], "rpc_ready")
        self.assertFalse(audit["ready"])
        self.assertEqual(audit["currentStage"], "rpc_ready")
        self.assertEqual(audit["nextStage"], "inference_ready")
        self.assertTrue(
            audit["networkAudit"]["llamaCppComputeCell"]["readyForLlamaCppRpc"]
        )

    def test_cai_instance_readiness_audit_reports_cai_owned_route_ready(self) -> None:
        shard_a = {
            "PipelineShardMetadata": {
                "modelCard": {"modelId": "Qwen/Qwen3-0.6B-GGUF"},
                "startLayer": 0,
                "endLayer": 14,
                "nLayers": 28,
            }
        }
        shard_b = {
            "PipelineShardMetadata": {
                "modelCard": {"modelId": "Qwen/Qwen3-0.6B-GGUF"},
                "startLayer": 14,
                "endLayer": 28,
                "nLayers": 28,
            }
        }
        state_payload = {
            "instances": {
                "iid-1": {
                    "MlxRingInstance": {
                        "shardAssignments": {
                            "modelId": "Qwen/Qwen3-0.6B-GGUF",
                            "nodeToRunner": {
                                "node-a": "runner-a",
                                "node-b": "runner-b",
                            },
                            "runnerToShard": {
                                "runner-a": shard_a,
                                "runner-b": shard_b,
                            },
                        }
                    }
                }
            },
            "runners": {
                "runner-a": {"RunnerStarting": {}},
                "runner-b": {"RunnerStarting": {}},
            },
            "topology": {"connections": {}},
        }
        route_health = [
            RouteHealthRecord(
                route_id="node-a->node-b",
                source_node_id="node-a",
                sink_node_id="node-b",
                route_type="llama_cpp_rpc_direct",
                endpoint_url="llama-cpp-rpc://198.51.100.11:52435",
                reachable=True,
                checked_at="2026-05-03T00:00:00+00:00",
                latency_ms=45.0,
            ),
            RouteHealthRecord(
                route_id="node-b->node-a",
                source_node_id="node-b",
                sink_node_id="node-a",
                route_type="llama_cpp_rpc_direct",
                endpoint_url="llama-cpp-rpc://198.51.100.12:52435",
                reachable=True,
                checked_at="2026-05-03T00:00:00+00:00",
                latency_ms=45.0,
            )
        ]

        with patch(
            "cai_compute_chain.jobs._load_cai_state_payload",
            return_value=state_payload,
        ), patch(
            "cai_compute_chain.jobs.list_route_health_records",
            return_value=route_health,
        ):
            audit = cai_instance_readiness_audit(
                "http://127.0.0.1:52425",
                "Qwen/Qwen3-0.6B-GGUF",
            )

        self.assertEqual(audit["status"], "cai_owned_route_ready")
        self.assertFalse(audit["ready"])
        self.assertEqual(audit["currentStage"], "rpc_ready")
        self.assertEqual(audit["nextStage"], "inference_ready")
        self.assertTrue(
            audit["networkAudit"]["llamaCppExecutionStrategy"][
                "caiOwnedTransport"
            ]["routeHealthReadiness"]["ready"]
        )

    def test_pending_download_description_ignores_model_with_completed_progress(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-remote": {"friendlyName": "DESKTOP-REMOTE"},
            },
            "downloads": {
                "node-remote": [
                    {
                        "DownloadPending": {
                            "shardMetadata": {
                                "PipelineShardMetadata": {
                                    "modelCard": {"modelId": "Qwen/Qwen3-0.6B-GGUF"}
                                }
                            }
                        }
                    },
                    {
                        "DownloadCompleted": {
                            "shardMetadata": {
                                "PipelineShardMetadata": {
                                    "modelCard": {"modelId": "Qwen/Qwen3-0.6B-GGUF"}
                                }
                            }
                        }
                    },
                ]
            },
        }

        with patch(
            "cai_compute_chain.jobs._load_cai_state_payload",
            return_value=state_payload,
        ):
            message = _describe_pending_model_downloads(
                "http://127.0.0.1:52425",
                "Qwen/Qwen3-0.6B-GGUF",
            )

        self.assertIsNone(message)

    def _distributed_two_worker_state(self) -> dict:
        return {
            "nodeIdentities": {
                "node-a": {
                    "apiHost": "198.51.100.10",
                    "apiPort": 52425,
                    "dataHost": "198.51.100.10",
                    "dataPort": 52435,
                    "workerRewardAddress": "aaaa1234aaaa1234aaaa1234aaaa1234",
                },
                "node-b": {
                    "apiHost": "198.51.100.11",
                    "apiPort": 52425,
                    "dataHost": "198.51.100.11",
                    "dataPort": 52435,
                    "workerRewardAddress": "bbbb1234bbbb1234bbbb1234bbbb1234",
                },
                "node-relay": {
                    "apiHost": "198.51.100.12",
                    "apiPort": 52425,
                    "dataHost": "198.51.100.12",
                    "dataPort": 52435,
                    "relayEnabled": True,
                },
            },
            "overlayPeers": {
                "node-a": ["node-b"],
                "node-b": ["node-a"],
            },
            "topology": {
                "nodes": ["node-a", "node-b"],
                "connections": {
                    "node-a": {
                        "node-b": [
                            {
                                "sinkMultiaddr": {
                                    "address": "/ip4/198.51.100.11/tcp/52435"
                                }
                            }
                        ]
                    },
                    "node-b": {
                        "node-a": [
                            {
                                "sinkMultiaddr": {
                                    "address": "/ip4/198.51.100.10/tcp/52435"
                                }
                            }
                        ]
                    },
                },
            },
        }

    def _distributed_oneway_two_worker_state(self) -> dict:
        return {
            "nodeIdentities": {
                "node-a": {
                    "apiHost": "198.51.100.10",
                    "apiPort": 52425,
                    "dataHost": "198.51.100.10",
                    "dataPort": 52435,
                },
                "node-b": {
                    "apiHost": "198.51.100.11",
                    "apiPort": 52425,
                    "dataHost": "198.51.100.11",
                    "dataPort": 52435,
                },
            },
            "overlayPeers": {
                "node-a": ["node-b"],
                "node-b": ["node-a"],
            },
            "topology": {
                "nodes": ["node-a", "node-b"],
                "connections": {
                    "node-a": {
                        "node-b": [
                            {
                                "sinkMultiaddr": {
                                    "address": "/ip4/198.51.100.11/tcp/52435"
                                }
                            }
                        ]
                    },
                },
            },
        }

    def _relay_candidate_two_worker_state(self) -> dict:
        return {
            "nodeIdentities": {
                "node-a": {
                    "apiHost": "198.51.100.10",
                    "apiPort": 52425,
                    "dataHost": "198.51.100.10",
                    "dataPort": 52435,
                },
                "node-b": {
                    "apiHost": "198.51.100.11",
                    "apiPort": 52425,
                    "dataHost": "198.51.100.11",
                    "dataPort": 52435,
                },
                "node-relay": {
                    "apiHost": "198.51.100.12",
                    "apiPort": 52425,
                    "dataHost": "198.51.100.12",
                    "dataPort": 52435,
                    "relayEnabled": True,
                    "transportEndpoints": [
                        {
                            "purpose": "api",
                            "routeType": "relay",
                            "host": "198.51.100.12",
                            "port": 52425,
                        }
                    ],
                },
            },
            "overlayPeers": {
                "node-a": ["node-relay"],
                "node-relay": ["node-a", "node-b"],
                "node-b": ["node-relay"],
            },
            "topology": {
                "nodes": ["node-a", "node-b", "node-relay"],
                "connections": {},
            },
        }

    def test_ensure_cai_instance_reuses_ready_instance(self) -> None:
        states = [
            {
                "iid-1": {
                    "MlxRingInstance": {
                        "shardAssignments": {
                            "modelId": "Qwen/Qwen3-0.6B-GGUF",
                            "runnerToShard": {"runner-a": {}, "runner-b": {}},
                        }
                    }
                }
            },
            {
                "instances": {
                    "iid-1": {
                        "MlxRingInstance": {
                            "shardAssignments": {
                                "modelId": "Qwen/Qwen3-0.6B-GGUF",
                                "runnerToShard": {"runner-a": {}, "runner-b": {}},
                            }
                        }
                    }
                },
                "runners": {
                    "runner-a": {"RunnerReady": {}},
                    "runner-b": {"RunnerRunning": {}},
                },
            },
        ]

        def fake_get_json(url: str, *, timeout: int = 30):
            if url.endswith("/state/instances"):
                return states.pop(0) if states else {
                    "iid-1": {
                        "MlxRingInstance": {
                            "shardAssignments": {
                                "modelId": "Qwen/Qwen3-0.6B-GGUF",
                                "runnerToShard": {"runner-a": {}, "runner-b": {}},
                            }
                        }
                    }
                }
            if url.endswith("/state"):
                return states.pop(0)
            raise AssertionError(f"Unexpected URL {url}")

        with (
            patch("cai_compute_chain.jobs._get_json", side_effect=fake_get_json),
            patch("cai_compute_chain.jobs._post_json") as post_json,
            patch("cai_compute_chain.jobs.time.sleep"),
        ):
            ensure_cai_instance("http://127.0.0.1:52425", "Qwen/Qwen3-0.6B-GGUF")

        post_json.assert_not_called()

    def test_ensure_cai_instance_falls_back_to_direct_placement_when_previews_empty(self) -> None:
        state_instances = [
            {},
            {},
            {
                "iid-1": {
                    "MlxRingInstance": {
                        "shardAssignments": {"modelId": "Qwen/Qwen3-0.6B-GGUF"}
                    }
                }
            },
        ]
        state_root = [
            {
                "instances": {
                    "iid-1": {
                        "MlxRingInstance": {
                            "shardAssignments": {
                                "modelId": "Qwen/Qwen3-0.6B-GGUF",
                                "runnerToShard": {"runner-a": {}, "runner-b": {}},
                            }
                        }
                    }
                },
                "runners": {
                    "runner-a": {"RunnerReady": {}},
                    "runner-b": {"RunnerRunning": {}},
                },
            }
        ]
        placement_instance = {
            "MlxRingInstance": {
                "shardAssignments": {"modelId": "Qwen/Qwen3-0.6B-GGUF"}
            }
        }

        def fake_get_json(url: str, *, timeout: int = 30):
            if "/instance/previews" in url:
                return {"previews": []}
            if "/instance/placement" in url:
                return placement_instance
            if url.endswith("/state/instances"):
                return state_instances.pop(0) if state_instances else {
                    "iid-1": {
                        "MlxRingInstance": {
                            "shardAssignments": {"modelId": "Qwen/Qwen3-0.6B-GGUF"}
                        }
                    }
                }
            if url.endswith("/state"):
                return state_root.pop(0)
            raise AssertionError(f"Unexpected URL {url}")

        with (
            patch("cai_compute_chain.jobs._get_json", side_effect=fake_get_json),
            patch("cai_compute_chain.jobs._post_json", return_value={"message": "ok"}) as post_json,
            patch("cai_compute_chain.jobs.time.sleep"),
        ):
            ensure_cai_instance("http://127.0.0.1:52425", "Qwen/Qwen3-0.6B-GGUF")

        post_json.assert_called_once_with(
            "http://127.0.0.1:52425/instance",
            {"instance": placement_instance},
            timeout=180,
        )

    def test_ensure_cai_instance_returns_planned_snapshot_when_state_visibility_flaps(self) -> None:
        state_instances = [{}] * 20
        placement_instance = {
            "MlxRingInstance": {
                "instanceId": "planned-instance",
                "shardAssignments": {
                    "modelId": "Qwen/Qwen3-0.6B-GGUF",
                    "runnerToShard": {
                        "runner-local": {
                            "PipelineShardMetadata": {
                                "startLayer": 0,
                                "endLayer": 26,
                                "nLayers": 28,
                            }
                        },
                        "runner-remote": {
                            "PipelineShardMetadata": {
                                "startLayer": 26,
                                "endLayer": 28,
                                "nLayers": 28,
                            }
                        },
                    },
                    "nodeToRunner": {
                        "node-local": "runner-local",
                        "node-remote": "runner-remote",
                    },
                },
            }
        }

        def fake_get_json(url: str, *, timeout: int = 30):
            if "/instance/previews" in url:
                return {"previews": [{"instance": placement_instance, "error": None}]}
            if url.endswith("/state/instances"):
                return state_instances.pop(0) if state_instances else {}
            raise AssertionError(f"Unexpected URL {url}")

        with (
            patch("cai_compute_chain.jobs._get_json", side_effect=fake_get_json),
            patch("cai_compute_chain.jobs._wait_for_cai_instance_ready", return_value=False),
            patch("cai_compute_chain.jobs._post_json", return_value={"message": "ok"}),
            patch("cai_compute_chain.jobs.time.sleep"),
        ):
            snapshot = ensure_cai_instance(
                "http://127.0.0.1:52425",
                "Qwen/Qwen3-0.6B-GGUF",
            )

        self.assertEqual(snapshot["instance_id"], "planned-instance")
        self.assertEqual(snapshot["snapshot_source"], "planned_definition")
        self.assertEqual(
            [item["node_id"] for item in snapshot["participants"]],
            ["node-local", "node-remote"],
        )
        self.assertEqual(
            [item["layer_count"] for item in snapshot["participants"]],
            [26, 2],
        )

    def test_list_cai_instances_filters_by_model(self) -> None:
        payload = {
            "instance-a": {
                "MlxRingInstance": {"shardAssignments": {"modelId": "model-a"}}
            },
            "instance-b": {
                "MlxRingInstance": {"shardAssignments": {"modelId": "model-b"}}
            },
        }

        with patch("cai_compute_chain.jobs._get_json", return_value=payload):
            items = list_cai_instances("http://127.0.0.1:52425", model_id="model-a")

        self.assertEqual(
            items,
            [
                {
                    "instance_id": "instance-a",
                    "instance_type": "MlxRingInstance",
                    "model_id": "model-a",
                }
            ],
        )

    def test_resolve_cai_command_instance_snapshot_uses_task_instance_id(self) -> None:
        state_payload = {
            "tasks": {
                "task-1": {
                    "TextGeneration": {
                        "taskId": "task-1",
                        "commandId": "cmd-distributed",
                        "instanceId": "instance-distributed",
                    }
                }
            },
            "instances": {
                "instance-single": {
                    "MlxRingInstance": {
                        "shardAssignments": {
                            "modelId": "Qwen/Qwen3-0.6B-GGUF",
                            "runnerToShard": {
                                "runner-local": {
                                    "PipelineShardMetadata": {
                                        "startLayer": 0,
                                        "endLayer": 28,
                                        "nLayers": 28,
                                    }
                                }
                            },
                            "nodeToRunner": {"node-local": "runner-local"},
                        }
                    }
                },
                "instance-distributed": {
                    "MlxRingInstance": {
                        "shardAssignments": {
                            "modelId": "Qwen/Qwen3-0.6B-GGUF",
                            "runnerToShard": {
                                "runner-local": {
                                    "PipelineShardMetadata": {
                                        "startLayer": 0,
                                        "endLayer": 25,
                                        "nLayers": 28,
                                    }
                                },
                                "runner-remote": {
                                    "PipelineShardMetadata": {
                                        "startLayer": 25,
                                        "endLayer": 28,
                                        "nLayers": 28,
                                    }
                                },
                            },
                            "nodeToRunner": {
                                "node-local": "runner-local",
                                "node-remote": "runner-remote",
                            },
                        }
                    }
                },
            },
        }

        with patch("cai_compute_chain.jobs._get_json", return_value=state_payload):
            snapshot = resolve_cai_command_instance_snapshot(
                "http://127.0.0.1:52425",
                "cmd-distributed",
                model_id="Qwen/Qwen3-0.6B-GGUF",
            )

        assert snapshot is not None
        self.assertEqual(snapshot["instance_id"], "instance-distributed")
        self.assertEqual(snapshot["snapshot_source"], "state")
        self.assertEqual(
            [item["node_id"] for item in snapshot["participants"]],
            ["node-local", "node-remote"],
        )
        self.assertEqual(
            [item["layer_count"] for item in snapshot["participants"]],
            [25, 3],
        )

    def test_resolve_cai_command_instance_snapshot_logs_state_failure(self) -> None:
        with (
            patch(
                "cai_compute_chain.jobs._get_json",
                side_effect=TimeoutError("state endpoint timed out"),
            ),
            self.assertLogs("cai_compute_chain.jobs", level="WARNING") as logs,
        ):
            snapshot = resolve_cai_command_instance_snapshot(
                "http://127.0.0.1:52425",
                "cmd-distributed",
                model_id="Qwen/Qwen3-0.6B-GGUF",
            )

        self.assertIsNone(snapshot)
        self.assertIn(
            "CAI command instance snapshot state fetch failed",
            "\n".join(logs.output),
        )
        self.assertIn("state endpoint timed out", "\n".join(logs.output))

    def test_cleanup_cai_model_instances_deletes_matching_instances(self) -> None:
        states = [
            {
                "instance-a": {
                    "MlxRingInstance": {"shardAssignments": {"modelId": "model-a"}}
                },
                "instance-b": {
                    "MlxRingInstance": {"shardAssignments": {"modelId": "model-a"}}
                },
                "instance-c": {
                    "MlxRingInstance": {"shardAssignments": {"modelId": "model-b"}}
                },
            },
            {
                "instance-c": {
                    "MlxRingInstance": {"shardAssignments": {"modelId": "model-b"}}
                }
            },
        ]

        with (
            patch("cai_compute_chain.jobs._get_json", side_effect=lambda *_args, **_kwargs: states.pop(0)),
            patch("cai_compute_chain.jobs._delete_json", return_value={"message": "ok"}) as delete_json,
            patch("cai_compute_chain.jobs.time.sleep"),
        ):
            cleanup_cai_model_instances("http://127.0.0.1:52425", "model-a")

        self.assertEqual(delete_json.call_count, 2)
        deleted_urls = [call.args[0] for call in delete_json.call_args_list]
        self.assertIn("http://127.0.0.1:52425/instance/instance-a", deleted_urls)
        self.assertIn("http://127.0.0.1:52425/instance/instance-b", deleted_urls)

    def test_execute_job_intent_cleans_up_model_instances_on_failure(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("2.00000000"))
        job = create_job_intent(
            prompt="2+2=?",
            compute_amount_coins="1.00000000",
            payment_preference=PaymentPreference.AUTO,
            cai_url="http://127.0.0.1:52425",
        )

        with (
            patch.dict(os.environ, {"CAI_JOB_EXECUTION_MAX_ATTEMPTS": "2"}),
            patch("cai_compute_chain.jobs.cleanup_cai_model_instances") as cleanup_instances,
            patch("cai_compute_chain.jobs.ensure_cai_instance"),
            patch(
                "cai_compute_chain.jobs.resolve_cai_instance_snapshot",
                return_value={"instance_id": "instance-1", "participants": []},
            ),
            patch("cai_compute_chain.jobs._submit_text_job_to_cai", side_effect=TimeoutError("timed out")),
            patch("cai_compute_chain.jobs.time.sleep"),
        ):
            with self.assertRaises(TimeoutError):
                execute_job_intent(job.job_id)

        self.assertEqual(cleanup_instances.call_count, 2)
        cleanup_instances.assert_any_call(
            "http://127.0.0.1:52425",
            "Qwen/Qwen3-0.6B-GGUF",
            best_effort=True,
        )
        persisted_job = next(item for item in list_job_intents() if item.job_id == job.job_id)
        self.assertEqual(persisted_job.status, "failed")
        self.assertEqual(len(persisted_job.execution_attempts), 2)
        self.assertEqual(list_execution_receipts(), [])
        self.assertEqual(list_settlements(), [])


if __name__ == "__main__":
    unittest.main()

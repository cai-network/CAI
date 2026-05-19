# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.economics import plan_funding
from cai_compute_chain.chain import (
    ensure_chain_genesis,
    make_chain_transaction,
    record_chain_transaction,
)
from cai_compute_chain.model import PaymentPreference, WalletPolicy
from cai_compute_chain.node_config import bind_worker_reward_address, set_relay_mode
from cai_compute_chain.settlement import (
    record_funding_settlement,
    record_validator_penalty_attestation,
    record_validator_evidence,
    record_validator_attestation,
    record_worker_payouts,
    list_validator_evidence_cases,
)
from cai_compute_chain.ui_state import build_interface_snapshot
from cai_compute_chain.validators import sync_validator_record
from cai_compute_chain.wallet import (
    coins_to_atomic,
    create_wallet,
    credit_wallet,
    load_or_create_ledger,
    unlock_wallet,
)


class InterfaceStateTests(unittest.TestCase):
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

    def test_snapshot_includes_wallet_quote_and_network_counts(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, 500_000_000)

        payload = {
            "overlayPeers": {"peer-a": {}, "peer-b": {}},
            "nodeSystem": {"node-a": {}, "node-b": {}},
            "topology": {
                "nodes": [{"id": "a"}, {"id": "b"}],
                "connections": {"a-b": {"from_id": "a", "to_id": "b"}},
            },
        }
        response = MagicMock()
        response.read.return_value = json.dumps(payload).encode("utf-8")
        response.__enter__.return_value = response
        response.__exit__.return_value = False

        with patch("cai_compute_chain.ui_state.urlopen", return_value=response):
            snapshot = build_interface_snapshot(
                state_url="http://127.0.0.1:52415/state",
                quote_amount_coins="1.00000000",
                payment_preference=PaymentPreference.AUTO,
            )

        self.assertTrue(snapshot.wallet.has_active_wallet)
        self.assertEqual(snapshot.wallet.wallet_name, "main")
        self.assertEqual(snapshot.wallet.balance_source, "chain")
        self.assertEqual(snapshot.wallet.balance_coins, "0.00000000")
        self.assertEqual(snapshot.wallet.local_cached_balance_coins, "5.00000000")
        self.assertTrue(snapshot.wallet.unlocked)
        self.assertEqual(snapshot.network.overlay_peers, 2)
        self.assertEqual(snapshot.network.topology_nodes, 2)
        self.assertEqual(snapshot.network.topology_connections, 1)
        self.assertEqual(snapshot.chain.tip_height, 0)
        self.assertEqual(snapshot.chain.finalized_height, 0)
        self.assertTrue(snapshot.chain.valid)
        self.assertEqual(snapshot.reward.pending_count, 0)
        self.assertEqual(snapshot.reward.finalized_count, 0)
        self.assertEqual(snapshot.reward.applied_count, 0)
        self.assertEqual(snapshot.compute.pricing_mode, "manual")
        self.assertEqual(snapshot.compute.funding_source, "reserve")
        self.assertEqual(snapshot.compute.ai_development_fee_coins, "0.00000000")
        self.assertEqual(snapshot.compute.worker_reward_coins, "0.98000000")
        self.assertTrue(snapshot.security.wallet_post_quantum_ready)
        self.assertEqual(snapshot.security.wallet_signing_scheme, wallet.signing_scheme)

    def test_compute_preview_uses_chain_balance_over_local_cache(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credit_wallet(wallet.wallet_id, coins_to_atomic("5.00000000"))
        ensure_chain_genesis()

        with patch(
            "cai_compute_chain.ui_state.urlopen",
            side_effect=OSError("connection refused"),
        ):
            snapshot = build_interface_snapshot(
                state_url="http://127.0.0.1:52415/state",
                quote_amount_coins="1.00000000",
                payment_preference=PaymentPreference.WALLET_ONLY,
            )

        self.assertEqual(snapshot.wallet.balance_source, "chain")
        self.assertFalse(snapshot.compute.quote_available)
        self.assertIn("wallet balance is insufficient", snapshot.compute.quote_reason.lower())

    def test_snapshot_uses_explicit_wallet_policy_chain_index(self) -> None:
        wallet_policy = WalletPolicy(wallet_data_dirname=".tmp-ui-state-policy")
        wallet = create_wallet(
            "policy-wallet",
            "testpass",
            select=True,
            wallet_policy=wallet_policy,
        )
        unlock_wallet("testpass", wallet_policy=wallet_policy)
        credit_wallet(
            wallet.wallet_id,
            coins_to_atomic("5.00000000"),
            wallet_policy=wallet_policy,
        )
        self.assertTrue(
            record_chain_transaction(
                make_chain_transaction(
                    tx_type="test_wallet_credit",
                    address=wallet.address,
                    delta_atomic=coins_to_atomic("1.25000000"),
                    wallet_id=wallet.wallet_id,
                    note="policy scoped chain balance",
                ),
                policy=wallet_policy,
            )
        )

        with patch(
            "cai_compute_chain.ui_state.urlopen",
            side_effect=OSError("connection refused"),
        ):
            snapshot = build_interface_snapshot(
                state_url="http://127.0.0.1:52415/state",
                wallet_policy=wallet_policy,
            )

        self.assertTrue(snapshot.wallet.has_active_wallet)
        self.assertEqual(snapshot.wallet.wallet_name, "policy-wallet")
        self.assertEqual(snapshot.wallet.balance_source, "chain")
        self.assertEqual(snapshot.wallet.balance_coins, "1.25000000")
        self.assertEqual(snapshot.wallet.local_cached_balance_coins, "5.00000000")

    def test_snapshot_includes_worker_reward_binding_and_earnings(self) -> None:
        worker_wallet = create_wallet("worker-a", "testpass", select=True)
        unlock_wallet("testpass")
        set_relay_mode(True)
        worker_wallet = credit_wallet(
            worker_wallet.wallet_id, coins_to_atomic("2.00000000")
        )
        sync_validator_record(
            validator_id="validator-a",
            wallet_id=worker_wallet.wallet_id,
            address="validator-a",
            state="bonded",
            bonded_atomic=coins_to_atomic("10000.00000000"),
            static_ip_confirmed=True,
            current_node_id="node-validator",
            advertised_api_host="85.137.164.250",
            advertised_data_host="85.137.164.250",
        )
        decision = plan_funding(
            ledger=load_or_create_ledger(),
            wallet=worker_wallet,
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.AUTO,
        )
        settlement = record_funding_settlement(
            source_wallet_id=worker_wallet.wallet_id,
            source_wallet_address=worker_wallet.address,
            decision=decision,
            note="ui worker earnings settlement",
        )
        bind_worker_reward_address("node-a", worker_wallet.address)
        worker_a_reward = (settlement.worker_reward_atomic * 6_000) // 10_000
        worker_b_reward = settlement.worker_reward_atomic - worker_a_reward
        record_worker_payouts(
            settlement_id=settlement.settlement_id,
            receipt_id="receipt-earnings",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            participants=[
                {
                    "node_id": "node-a",
                    "runner_id": "runner-a",
                    "layer_start": 0,
                    "layer_end": 12,
                    "layer_count": 12,
                    "share_bps": 6000,
                    "reward_atomic": worker_a_reward,
                    "note": "Distributed by pipeline layer share.",
                },
                {
                    "node_id": "node-b",
                    "runner_id": "runner-b",
                    "layer_start": 12,
                    "layer_end": 20,
                    "layer_count": 8,
                    "share_bps": 4000,
                    "reward_atomic": worker_b_reward,
                    "note": "Distributed by pipeline layer share.",
                },
            ],
        )
        record_validator_attestation(
            settlement_id=settlement.settlement_id,
            validator_id="validator-a",
            accepted=True,
            note="accept",
        )

        with patch(
            "cai_compute_chain.ui_state.urlopen",
            side_effect=OSError("connection refused"),
        ):
            snapshot = build_interface_snapshot(
                state_url="http://127.0.0.1:52415/state",
            )

        self.assertEqual(snapshot.worker.reward_bindings, 1)
        self.assertTrue(snapshot.worker.relay_enabled)
        self.assertEqual(snapshot.worker.local_worker_earnings_coins, "0.58800000")
        self.assertEqual(snapshot.worker.external_payout_records, 0)
        self.assertEqual(snapshot.worker.unbound_payout_records, 1)
        self.assertEqual(snapshot.reward.applied_count, 1)
        self.assertEqual(snapshot.reward.applied_coins, "0.58800000")
        self.assertEqual(snapshot.reward.unbound_count, 1)
        self.assertEqual(snapshot.reward.unbound_coins, "0.39200000")
        self.assertEqual(snapshot.reward.pending_count, 0)
        self.assertEqual(snapshot.reward.chain_recorded_count, 1)

    def test_snapshot_reports_pending_reward_before_finality(self) -> None:
        worker_wallet = create_wallet("worker-a", "testpass", select=True)
        unlock_wallet("testpass")
        worker_wallet = credit_wallet(
            worker_wallet.wallet_id, coins_to_atomic("2.00000000")
        )
        decision = plan_funding(
            ledger=load_or_create_ledger(),
            wallet=worker_wallet,
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.AUTO,
        )
        settlement = record_funding_settlement(
            source_wallet_id=worker_wallet.wallet_id,
            source_wallet_address=worker_wallet.address,
            decision=decision,
            note="pending reward settlement",
        )
        bind_worker_reward_address("node-a", worker_wallet.address)
        record_worker_payouts(
            settlement_id=settlement.settlement_id,
            receipt_id="receipt-pending",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            participants=[
                {
                    "node_id": "node-a",
                    "runner_id": "runner-a",
                    "layer_start": 0,
                    "layer_end": 20,
                    "layer_count": 20,
                    "share_bps": 10000,
                    "reward_atomic": settlement.worker_reward_atomic,
                    "note": "Distributed by pipeline layer share.",
                },
            ],
        )

        with patch(
            "cai_compute_chain.ui_state.urlopen",
            side_effect=OSError("connection refused"),
        ):
            snapshot = build_interface_snapshot(
                state_url="http://127.0.0.1:52415/state",
            )

        self.assertEqual(snapshot.reward.pending_count, 1)
        self.assertEqual(snapshot.reward.pending_coins, "0.98000000")
        self.assertEqual(snapshot.reward.finalized_count, 0)
        self.assertEqual(snapshot.reward.applied_count, 0)
        self.assertEqual(snapshot.reward.latest_status, "pending")

    def test_snapshot_reports_unreachable_state(self) -> None:
        with patch(
            "cai_compute_chain.ui_state.urlopen",
            side_effect=OSError("connection refused"),
        ):
            snapshot = build_interface_snapshot(
                state_url="http://127.0.0.1:52415/state",
            )

        self.assertFalse(snapshot.network.reachable)
        self.assertIn("connection refused", snapshot.network.error)

    def test_snapshot_includes_validator_evidence_quorum_summary(self) -> None:
        sync_validator_record(
            validator_id="validator-a",
            wallet_id="wallet-a",
            address="validator-a",
            state="bonded",
            bonded_atomic=coins_to_atomic("10000.00000000"),
            static_ip_confirmed=True,
        )
        sync_validator_record(
            validator_id="validator-b",
            wallet_id="wallet-b",
            address="validator-b",
            state="bonded",
            bonded_atomic=coins_to_atomic("10000.00000000"),
            static_ip_confirmed=True,
        )
        record_validator_evidence(
            validator_id="validator-target",
            reporter_validator_id="validator-a",
            evidence_type="conflicting_attestation",
            settlement_id="settlement-a",
            slash_atomic=coins_to_atomic("500.00000000"),
            jailed=True,
        )

        with patch(
            "cai_compute_chain.ui_state.urlopen",
            side_effect=OSError("connection refused"),
        ):
            snapshot = build_interface_snapshot(
                state_url="http://127.0.0.1:52415/state",
            )

        self.assertEqual(snapshot.validator.evidence_count, 1)
        self.assertEqual(snapshot.validator.evidence_case_count, 1)
        self.assertEqual(snapshot.validator.evidence_case_pending_quorum_count, 1)
        self.assertEqual(snapshot.validator.evidence_case_finalized_count, 0)
        self.assertEqual(snapshot.validator.evidence_case_applied_count, 0)
        self.assertEqual(snapshot.validator.penalty_case_count, 1)
        self.assertEqual(snapshot.validator.penalty_case_pending_count, 1)
        self.assertEqual(snapshot.validator.penalty_case_pending_attestation_count, 0)
        self.assertEqual(snapshot.validator.penalty_case_finalized_count, 0)
        self.assertEqual(snapshot.validator.penalty_case_applied_count, 0)

    def test_snapshot_includes_penalty_attestation_summary(self) -> None:
        sync_validator_record(
            validator_id="validator-local",
            wallet_id="wallet-local",
            address="validator-local",
            state="bonded",
            bonded_atomic=coins_to_atomic("10000.00000000"),
            static_ip_confirmed=True,
        )

        record_validator_evidence(
            validator_id="validator-target",
            reporter_validator_id="validator-local",
            evidence_type="conflicting_attestation",
            settlement_id="settlement-a",
            slash_atomic=coins_to_atomic("500.00000000"),
            jailed=True,
        )
        case_id = list_validator_evidence_cases()[0].case_id
        record_validator_penalty_attestation(
            case_id=case_id,
            validator_id="validator-local",
            accepted=True,
            note="validator confirms penalty case",
        )

        with patch(
            "cai_compute_chain.ui_state.urlopen",
            side_effect=OSError("connection refused"),
        ):
            snapshot = build_interface_snapshot(
                state_url="http://127.0.0.1:52415/state",
            )

        self.assertEqual(snapshot.validator.penalty_case_count, 1)
        self.assertEqual(snapshot.validator.penalty_case_pending_count, 0)
        self.assertEqual(snapshot.validator.penalty_case_pending_attestation_count, 0)
        self.assertEqual(snapshot.validator.penalty_case_finalized_count, 1)
        self.assertEqual(snapshot.validator.penalty_case_applied_count, 0)
        self.assertEqual(snapshot.validator.latest_penalty_case_status, "finalized")
        self.assertEqual(
            snapshot.validator.latest_penalty_case_validator_id, "validator-target"
        )


if __name__ == "__main__":
    unittest.main()

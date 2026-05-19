# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.economics import plan_funding
from cai_compute_chain.jobs import job_intent_file_path, list_job_intents
from cai_compute_chain.model import ChainNetwork, MoneyPolicy, PaymentPreference, WalletPolicy
from cai_compute_chain.node_config import bind_worker_reward_address
from cai_compute_chain.peer_payload import add_peer_payload_metadata, sign_peer_payload
from cai_compute_chain.settlement import (
    apply_finalized_settlement,
    attestation_file_path,
    ensure_settlement_committee,
    export_settlement_proposal_payload,
    export_validator_evidence_payload,
    import_settlement_proposal_payload,
    list_attestations,
    list_validator_evidence_cases,
    list_validator_penalty_attestations,
    list_validator_penalty_cases,
    list_settlements,
    list_validator_evidence,
    list_worker_payouts,
    merge_remote_validator_evidence_payload,
    reconcile_worker_payouts,
    record_funding_settlement,
    record_settlement_execution_audit,
    record_chain_entries_for_finalized_settlements,
    record_worker_payouts,
    record_validator_attestation,
    record_validator_evidence,
    reset_retryable_settlement_rejection,
    record_validator_penalty_attestation,
    request_remote_penalty_case_attestations,
    save_settlements,
    save_worker_payouts,
    SettlementRecord,
    settlement_file_path,
    sign_settlement_envelope,
    sync_validator_evidence_from_cai_peers,
    verify_settlement_envelope,
    worker_payout_file_path,
)
from cai_compute_chain.wallet_signing import (
    encode_bytes,
    generate_signing_seed,
    public_key_b64_from_seed,
)
from cai_compute_chain.chain import (
    chain_address_history,
    chain_balance_atomic,
    chain_settlement_history,
    chain_summary,
    compute_reserve_chain_address,
    list_chain_blocks,
    tx_fee_pool_chain_address,
    validator_settlement_fee_pool_chain_address,
)
from cai_compute_chain.validators import get_validator_record, sync_validator_record
from cai_compute_chain.wallet import (
    coins_to_atomic,
    create_wallet,
    credit_wallet,
    list_journal_entries,
    load_or_create_ledger,
    resolve_wallet,
    list_wallets,
    unlock_wallet,
)


class SettlementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo_patch = patch(
            "cai_compute_chain.wallet.repo_root",
            return_value=Path(self.tempdir.name),
        )
        self.repo_patch.start()
        self.money_policy = MoneyPolicy()

    def tearDown(self) -> None:
        self.repo_patch.stop()
        self.tempdir.cleanup()

    def peer_payload(self, payload: dict) -> dict:
        return add_peer_payload_metadata(payload)

    def test_record_finalized_settlements_can_skip_missing_canonical_chain_entry(self) -> None:
        settlement = SettlementRecord(
            settlement_id="settle-stale-local",
            created_at="2026-05-03T00:00:00+00:00",
            source_wallet_id="wallet-a",
            source_wallet_address="abcd1234abcd1234abcd1234abcd1234",
            funding_source="wallet",
            compute_cost_atomic=100_000,
            tx_fee_atomic=0,
            settlement_fee_atomic=0,
            worker_reward_atomic=0,
            committee_selection_seed="settle-stale-local",
            committee_target_size=1,
            committee_selection_mode="stake_weighted_lottery",
            committee_validator_ids=["validator-a"],
            committee_bonded_atomic_by_validator_id={"validator-a": 1_000},
            committee_total_bonded_atomic=1_000,
            committee_quorum_bond_atomic=667,
            source_wallet_debit_atomic=100_000,
            status="applied",
            applied_at="2026-05-03T00:00:01+00:00",
        )
        save_settlements([settlement])

        recorded = record_chain_entries_for_finalized_settlements(
            only_if_chain_recorded=True,
        )

        self.assertEqual(recorded, 0)
        self.assertEqual(chain_settlement_history(settlement.settlement_id), [])

    def test_list_worker_payouts_heals_null_filled_file(self) -> None:
        path = worker_payout_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00" * 128)

        payouts = list_worker_payouts()

        self.assertEqual(payouts, [])
        self.assertEqual(path.read_text(encoding="utf-8"), "[]")
        backups = list(path.parent.glob("worker-payouts.corrupt-*.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b"\x00" * 128)

    def test_list_attestations_heals_truncated_jsonl_file(self) -> None:
        path = attestation_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "attestation_id": "att-1",
                    "created_at": "2026-05-13T00:00:00+00:00",
                    "settlement_id": "settle-1",
                    "validator_id": "validator-1",
                    "accepted": True,
                    "note": "ok",
                },
                ensure_ascii=False,
            )
            + "\n"
            + '{"attestation_id":"broken"',
            encoding="utf-8",
        )

        attestations = list_attestations()

        self.assertEqual(len(attestations), 1)
        self.assertEqual(attestations[0].attestation_id, "att-1")
        healed_text = path.read_text(encoding="utf-8")
        self.assertEqual(
            healed_text,
            json.dumps(
                {
                    "attestation_id": "att-1",
                    "created_at": "2026-05-13T00:00:00+00:00",
                    "settlement_id": "settle-1",
                    "validator_id": "validator-1",
                    "accepted": True,
                    "note": "ok",
                },
                ensure_ascii=False,
            )
            + "\n",
        )
        backups = list(path.parent.glob("attestations.corrupt-*.jsonl"))
        self.assertEqual(len(backups), 1)

    def test_list_job_intents_heals_null_filled_file(self) -> None:
        path = job_intent_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00" * 64)

        intents = list_job_intents()

        self.assertEqual(intents, [])
        self.assertEqual(path.read_text(encoding="utf-8"), "[]")
        backups = list(path.parent.glob("job-intents.corrupt-*.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b"\x00" * 64)

    def _credit_wallet_local(self, wallet, amount_atomic: int):
        return credit_wallet(wallet.wallet_id, amount_atomic)

    def _create_single_validator_settlement(self):
        source_wallet = create_wallet("source", "testpass", select=True)
        unlock_wallet("testpass")
        source_wallet = self._credit_wallet_local(
            source_wallet,
            coins_to_atomic("2.00000000"),
        )
        sync_validator_record(
            validator_id="validator-a",
            wallet_id=source_wallet.wallet_id,
            address="validator-a",
            state="bonded",
            bonded_atomic=coins_to_atomic("10000.00000000"),
            static_ip_confirmed=True,
            current_node_id="node-validator",
            advertised_api_host="85.137.164.250",
            advertised_data_host="85.137.164.250",
        )
        decision = plan_funding(
            ledger=load_or_create_ledger(self.money_policy),
            wallet=source_wallet,
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=self.money_policy,
        )
        settlement = record_funding_settlement(
            source_wallet_id=source_wallet.wallet_id,
            source_wallet_address=source_wallet.address,
            decision=decision,
            note="single validator settlement",
        )
        return settlement

    def test_legacy_finalized_settlement_loads_as_applied(self) -> None:
        created_at = "2026-04-22T00:00:00+00:00"
        settlement_file_path().write_text(
            json.dumps(
                [
                    {
                        "settlement_id": "legacy-finalized",
                        "created_at": created_at,
                        "source_wallet_id": "wallet-a",
                        "source_wallet_address": "address-a",
                        "funding_source": "reserve",
                        "compute_cost_atomic": coins_to_atomic("1.00000000"),
                        "tx_fee_atomic": coins_to_atomic("0.00010000"),
                        "settlement_fee_atomic": coins_to_atomic("0.02000000"),
                        "worker_reward_atomic": coins_to_atomic("0.98000000"),
                        "status": "finalized",
                    }
                ]
            ),
            encoding="utf-8",
        )

        settlements = list_settlements()

        self.assertEqual(settlements[0].status, "applied")
        self.assertEqual(settlements[0].applied_at, created_at)

    def test_record_settlement_and_attestation(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        wallet = self._credit_wallet_local(
            wallet,
            coins_to_atomic("2.00000000"),
        )
        sync_validator_record(
            validator_id=wallet.address,
            wallet_id=wallet.wallet_id,
            address=wallet.address,
            state="bonded",
            bonded_atomic=coins_to_atomic("10000.00000000"),
            static_ip_confirmed=True,
            current_node_id="node-a",
            advertised_api_host="85.137.164.250",
            advertised_data_host="85.137.164.250",
        )
        ledger = load_or_create_ledger(self.money_policy)
        decision = plan_funding(
            ledger=ledger,
            wallet=wallet,
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=self.money_policy,
        )

        settlement = record_funding_settlement(
            source_wallet_id=wallet.wallet_id,
            source_wallet_address=wallet.address,
            decision=decision,
            note="test settlement",
        )
        attestation = record_validator_attestation(
            settlement_id=settlement.settlement_id,
            validator_id=wallet.address,
            accepted=True,
            note="test attestation",
        )

        settlements = list_settlements()
        attestations = list_attestations()
        self.assertEqual(len(settlements), 1)
        self.assertEqual(len(attestations), 1)
        self.assertEqual(settlements[0].settlement_id, settlement.settlement_id)
        self.assertEqual(attestations[0].settlement_id, settlement.settlement_id)
        self.assertEqual(attestation.validator_id, wallet.address)
        self.assertEqual(settlements[0].status, "applied")
        self.assertEqual(settlements[0].reward_token_code, self.money_policy.reward_token_code)
        self.assertEqual(
            settlements[0].committee_quorum_bond_atomic,
            666_666_666_667,
        )

    def test_signed_settlement_envelope_covers_execution_and_payout_audit(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        wallet = self._credit_wallet_local(
            wallet,
            coins_to_atomic("2.00000000"),
        )
        ledger = load_or_create_ledger(self.money_policy)
        decision = plan_funding(
            ledger=ledger,
            wallet=wallet,
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=self.money_policy,
        )
        settlement = record_funding_settlement(
            source_wallet_id=wallet.wallet_id,
            source_wallet_address=wallet.address,
            decision=decision,
            note="signed envelope test",
        )
        payouts = record_worker_payouts(
            settlement_id=settlement.settlement_id,
            receipt_id="receipt-1",
            model_id="cai-network/Qwen3-0.6B-GGUF",
            participants=[
                {
                    "node_id": "node-a",
                    "runner_id": "runner-a",
                    "layer_start": 0,
                    "layer_end": 1,
                    "layer_count": 1,
                    "share_bps": 10000,
                    "reward_atomic": settlement.worker_reward_atomic,
                }
            ],
        )
        record_settlement_execution_audit(
            settlement_id=settlement.settlement_id,
            receipt_id="receipt-1",
            job_id="job-1",
            model_id="cai-network/Qwen3-0.6B-GGUF",
            execution_model_id="cai-network/Qwen3-0.6B-GGUF",
            pricing_mode="network_auto",
            pricing_basis="llm_tokens",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            reserved_prompt_tokens=10,
            reserved_completion_tokens=128,
            reserved_compute_cost_atomic=settlement.compute_cost_atomic,
            actual_compute_cost_atomic=settlement.compute_cost_atomic,
            usage_priced=True,
            network_audit={"transportMode": "multi_worker_direct"},
            worker_payouts=payouts,
        )

        signed = sign_settlement_envelope(settlement.settlement_id)

        self.assertIsNotNone(signed)
        assert signed is not None
        envelope = signed.balance_audit["signed_envelope"]
        self.assertEqual(envelope["status"], "signed")
        self.assertTrue(envelope["required"])
        self.assertTrue(envelope["signature_valid"])
        self.assertEqual(envelope["source_wallet_id"], wallet.wallet_id)
        valid, error = verify_settlement_envelope(signed)
        self.assertTrue(valid, error)

        signed.compute_cost_atomic += 1
        valid, error = verify_settlement_envelope(signed)
        self.assertFalse(valid)
        self.assertIn("payload hash", str(error))

    def test_ensure_settlement_committee_resigns_envelope_after_backfill(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        decision = plan_funding(
            ledger=load_or_create_ledger(self.money_policy),
            wallet=wallet,
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=self.money_policy,
        )
        settlement = record_funding_settlement(
            source_wallet_id=wallet.wallet_id,
            source_wallet_address=wallet.address,
            decision=decision,
            money_policy=self.money_policy,
        )
        sign_settlement_envelope(
            settlement.settlement_id,
            money_policy=self.money_policy,
        )
        sync_validator_record(
            validator_id="validator-remote",
            wallet_id="wallet-remote",
            address="validator-remote",
            state="bonded",
            bonded_atomic=coins_to_atomic("10000.00000000"),
            static_ip_confirmed=True,
            current_node_id="node-remote",
            advertised_api_host="85.137.164.250",
            advertised_data_host="85.137.164.250",
        )

        backfilled = ensure_settlement_committee(
            settlement.settlement_id,
            money_policy=self.money_policy,
        )

        self.assertIsNotNone(backfilled)
        assert backfilled is not None
        self.assertEqual(backfilled.committee_validator_ids, ["validator-remote"])
        valid, error = verify_settlement_envelope(backfilled)
        self.assertTrue(valid)
        self.assertIsNone(error)

    def test_reset_retryable_settlement_rejection_resigns_and_reopens_pending_settlement(
        self,
    ) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        decision = plan_funding(
            ledger=load_or_create_ledger(self.money_policy),
            wallet=wallet,
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=self.money_policy,
        )
        settlement = record_funding_settlement(
            source_wallet_id=wallet.wallet_id,
            source_wallet_address=wallet.address,
            decision=decision,
            money_policy=self.money_policy,
        )
        sign_settlement_envelope(
            settlement.settlement_id,
            money_policy=self.money_policy,
        )
        settlement.committee_validator_ids = ["validator-remote"]
        settlement.committee_bonded_atomic_by_validator_id = {
            "validator-remote": coins_to_atomic("10000.00000000")
        }
        settlement.committee_total_bonded_atomic = coins_to_atomic("10000.00000000")
        settlement.committee_quorum_bond_atomic = 666_666_666_667
        settlements = list_settlements()
        for index, item in enumerate(settlements):
            if item.settlement_id == settlement.settlement_id:
                settlements[index] = settlement
                break
        save_settlements(settlements)
        record_validator_attestation(
            settlement_id=settlement.settlement_id,
            validator_id="validator-remote",
            accepted=False,
            note="settlement signed envelope payload hash does not match",
            apply_on_finalize=False,
        )

        repaired = reset_retryable_settlement_rejection(
            settlement.settlement_id,
            money_policy=self.money_policy,
        )

        self.assertIsNotNone(repaired)
        assert repaired is not None
        self.assertEqual(repaired.status, "pending")
        self.assertEqual(repaired.rejected_attestations, 0)
        self.assertEqual(
            list_attestations(settlement_id=settlement.settlement_id),
            [],
        )
        valid, error = verify_settlement_envelope(repaired)
        self.assertTrue(valid)
        self.assertIsNone(error)

    def test_reset_retryable_settlement_rejection_reopens_missing_envelope_rejection(
        self,
    ) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        decision = plan_funding(
            ledger=load_or_create_ledger(self.money_policy),
            wallet=wallet,
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=self.money_policy,
        )
        settlement = record_funding_settlement(
            source_wallet_id=wallet.wallet_id,
            source_wallet_address=wallet.address,
            decision=decision,
            money_policy=self.money_policy,
        )
        sign_settlement_envelope(
            settlement.settlement_id,
            money_policy=self.money_policy,
        )
        settlement.committee_validator_ids = ["validator-remote"]
        settlement.committee_bonded_atomic_by_validator_id = {
            "validator-remote": coins_to_atomic("10000.00000000")
        }
        settlement.committee_total_bonded_atomic = coins_to_atomic("10000.00000000")
        settlement.committee_quorum_bond_atomic = 666_666_666_667
        settlement.balance_audit = {
            key: value
            for key, value in dict(settlement.balance_audit or {}).items()
            if key != "signed_envelope"
        }
        settlements = list_settlements()
        for index, item in enumerate(settlements):
            if item.settlement_id == settlement.settlement_id:
                settlements[index] = settlement
                break
        save_settlements(settlements)
        record_validator_attestation(
            settlement_id=settlement.settlement_id,
            validator_id="validator-remote",
            accepted=False,
            note="settlement signed envelope is missing",
            apply_on_finalize=False,
        )

        repaired = reset_retryable_settlement_rejection(
            settlement.settlement_id,
            money_policy=self.money_policy,
        )

        self.assertIsNotNone(repaired)
        assert repaired is not None
        self.assertEqual(repaired.status, "pending")
        self.assertEqual(repaired.rejected_attestations, 0)
        self.assertEqual(
            list_attestations(settlement_id=settlement.settlement_id),
            [],
        )
        valid, error = verify_settlement_envelope(repaired)
        self.assertTrue(valid)
        self.assertIsNone(error)

    def test_validator_imports_signed_settlement_proposal_and_records_chain(self) -> None:
        source_policy = WalletPolicy(wallet_data_dirname="source-wallet")
        validator_policy = WalletPolicy(wallet_data_dirname="validator-wallet")
        wallet = create_wallet(
            "source",
            "testpass",
            select=True,
            wallet_policy=source_policy,
        )
        unlock_wallet("testpass", wallet_policy=source_policy)
        sync_validator_record(
            validator_id="validator-remote",
            wallet_id="validator-wallet",
            address="validator-remote",
            state="bonded",
            bonded_atomic=coins_to_atomic("10000.00000000"),
            static_ip_confirmed=True,
            current_node_id="node-validator",
            advertised_api_host="85.137.164.250",
            advertised_data_host="85.137.164.250",
            policy=source_policy,
        )
        bind_worker_reward_address(
            "node-worker",
            "1234567890abcdef1234567890abcdef",
            policy=source_policy,
        )
        decision = plan_funding(
            ledger=load_or_create_ledger(self.money_policy, source_policy),
            wallet=wallet,
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=self.money_policy,
            wallet_policy=source_policy,
        )
        settlement = record_funding_settlement(
            source_wallet_id=wallet.wallet_id,
            source_wallet_address=wallet.address,
            decision=decision,
            note="remote validator proposal",
            money_policy=self.money_policy,
            policy=source_policy,
        )
        payouts = record_worker_payouts(
            settlement_id=settlement.settlement_id,
            receipt_id="receipt-remote",
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
            money_policy=self.money_policy,
            policy=source_policy,
        )
        record_settlement_execution_audit(
            settlement_id=settlement.settlement_id,
            receipt_id="receipt-remote",
            job_id="job-remote",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            execution_model_id="Qwen/Qwen3-0.6B-GGUF",
            pricing_mode="network_auto",
            pricing_basis="llm_tokens",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            reserved_prompt_tokens=10,
            reserved_completion_tokens=128,
            reserved_compute_cost_atomic=settlement.compute_cost_atomic,
            actual_compute_cost_atomic=settlement.compute_cost_atomic,
            usage_priced=True,
            network_audit={"transportMode": "single_worker"},
            worker_payouts=payouts,
            policy=source_policy,
        )
        signed = sign_settlement_envelope(
            settlement.settlement_id,
            policy=source_policy,
            money_policy=self.money_policy,
        )
        self.assertIsNotNone(signed)

        proposal = export_settlement_proposal_payload(
            settlement.settlement_id,
            policy=source_policy,
        )
        imported = import_settlement_proposal_payload(
            proposal,
            policy=validator_policy,
            money_policy=self.money_policy,
        )
        self.assertEqual(imported.settlement_id, settlement.settlement_id)
        self.assertEqual(
            list_worker_payouts(policy=validator_policy)[0].payout_id,
            payouts[0].payout_id,
        )

        record_validator_attestation(
            settlement_id=settlement.settlement_id,
            validator_id="validator-remote",
            accepted=True,
            note="accept",
            policy=validator_policy,
        )

        finalized = list_settlements(policy=validator_policy)[0]
        self.assertEqual(finalized.status, "applied")
        self.assertGreater(
            len(chain_settlement_history(settlement.settlement_id, policy=validator_policy)),
            0,
        )

    def test_finalized_settlement_records_reservation_surplus_release_tx(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        wallet = self._credit_wallet_local(
            wallet,
            coins_to_atomic("3.00000000"),
        )
        sync_validator_record(
            validator_id=wallet.address,
            wallet_id=wallet.wallet_id,
            address=wallet.address,
            state="bonded",
            bonded_atomic=coins_to_atomic("10000.00000000"),
            static_ip_confirmed=True,
            current_node_id="node-a",
            advertised_api_host="85.137.164.250",
            advertised_data_host="85.137.164.250",
        )
        bind_worker_reward_address("node-a", "1234567890abcdef1234567890abcdef")
        decision = plan_funding(
            ledger=load_or_create_ledger(self.money_policy),
            wallet=wallet,
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=self.money_policy,
        )
        settlement = record_funding_settlement(
            source_wallet_id=wallet.wallet_id,
            source_wallet_address=wallet.address,
            decision=decision,
            note="reservation surplus test",
        )
        payouts = record_worker_payouts(
            settlement_id=settlement.settlement_id,
            receipt_id="receipt-1",
            model_id="cai-network/Qwen3-0.6B-GGUF",
            participants=[
                {
                    "node_id": "node-a",
                    "runner_id": "runner-a",
                    "layer_start": 0,
                    "layer_end": 1,
                    "layer_count": 1,
                    "share_bps": 10000,
                    "reward_atomic": settlement.worker_reward_atomic,
                }
            ],
        )
        record_settlement_execution_audit(
            settlement_id=settlement.settlement_id,
            receipt_id="receipt-1",
            job_id="job-1",
            model_id="cai-network/Qwen3-0.6B-GGUF",
            execution_model_id="cai-network/Qwen3-0.6B-GGUF",
            pricing_mode="network_auto",
            pricing_basis="llm_tokens",
            reserved_compute_cost_atomic=coins_to_atomic("1.50000000"),
            actual_compute_cost_atomic=settlement.compute_cost_atomic,
            reservation_surplus_atomic=coins_to_atomic("0.50000000"),
            usage_priced=True,
            worker_payouts=payouts,
        )

        record_validator_attestation(
            settlement_id=settlement.settlement_id,
            validator_id=wallet.address,
            accepted=True,
            note="accept",
        )

        settlement_history = chain_settlement_history(settlement.settlement_id)
        surplus_txs = [
            item
            for item in settlement_history
            if item["tx_type"] == "settlement_reservation_surplus_release"
        ]
        self.assertEqual(len(surplus_txs), 1)
        self.assertEqual(surplus_txs[0]["delta_atomic"], 0)
        self.assertEqual(
            surplus_txs[0]["metadata"]["reservation_surplus_atomic"],
            coins_to_atomic("0.50000000"),
        )
        applied = list_settlements()[0]
        chain_balance_audit = applied.balance_audit["chain_balances"]
        self.assertIn(
            "reservation_surplus_release",
            {item["role"] for item in chain_balance_audit["addresses"]},
        )
        surplus_audit = next(
            item
            for item in chain_balance_audit["addresses"]
            if item["role"] == "reservation_surplus_release"
        )
        self.assertEqual(surplus_audit["expected_delta_atomic"], 0)
        self.assertTrue(surplus_audit["delta_matches_expected"])

    def test_finalized_settlement_skips_ai_development_fee_when_disabled(self) -> None:
        source_wallet = create_wallet("source", "testpass", select=True)
        unlock_wallet("testpass")
        source_wallet = self._credit_wallet_local(
            source_wallet,
            coins_to_atomic("2.00000000"),
        )
        ai_wallet = create_wallet("AI Development", "devpass")
        money_policy = MoneyPolicy(
            ai_development_wallet_id=ai_wallet.wallet_id,
            ai_development_address=ai_wallet.address,
        )
        sync_validator_record(
            validator_id="validator-a",
            wallet_id=source_wallet.wallet_id,
            address="validator-a",
            state="bonded",
            bonded_atomic=coins_to_atomic("10000.00000000"),
            static_ip_confirmed=True,
        )
        decision = plan_funding(
            ledger=load_or_create_ledger(money_policy),
            wallet=source_wallet,
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=money_policy,
        )
        settlement = record_funding_settlement(
            source_wallet_id=source_wallet.wallet_id,
            source_wallet_address=source_wallet.address,
            decision=decision,
            money_policy=money_policy,
        )

        record_validator_attestation(
            settlement_id=settlement.settlement_id,
            validator_id="validator-a",
            accepted=True,
        )

        updated_ai_wallet = resolve_wallet(ai_wallet.wallet_id, list_wallets())
        applied_settlement = list_settlements()[0]
        ledger = load_or_create_ledger(money_policy)
        self.assertIsNotNone(updated_ai_wallet)
        assert updated_ai_wallet is not None
        self.assertEqual(
            updated_ai_wallet.spendable_balance_atomic,
            0,
        )
        self.assertEqual(
            ledger.ai_development_fee_pool_atomic,
            0,
        )
        self.assertEqual(applied_settlement.ai_development_fee_atomic, 0)
        self.assertIsNone(applied_settlement.ai_development_credited_wallet_id)

    def test_conflicting_attestation_is_rejected(self) -> None:
        record_validator_attestation(
            settlement_id="settle-conflict",
            validator_id="validator-a",
            accepted=True,
            note="accepted once",
        )

        with self.assertRaisesRegex(
            ValueError, "Conflicting attestation already exists"
        ):
            record_validator_attestation(
                settlement_id="settle-conflict",
                validator_id="validator-a",
                accepted=False,
                note="rejected later",
            )

    def test_duplicate_accepted_attestation_with_different_note_is_idempotent(self) -> None:
        first = record_validator_attestation(
            settlement_id="settle-duplicate",
            validator_id="validator-a",
            accepted=True,
            note="accepted once",
        )

        repeated = record_validator_attestation(
            settlement_id="settle-duplicate",
            validator_id="validator-a",
            accepted=True,
            note="accepted during repair",
        )

        self.assertEqual(repeated.attestation_id, first.attestation_id)
        self.assertEqual(len(list_attestations(settlement_id="settle-duplicate")), 1)

    def test_record_validator_evidence(self) -> None:
        evidence = record_validator_evidence(
            validator_id="validator-a",
            evidence_type="eligibility_failure",
            settlement_id="settle-ev-1",
            attestation_id="att-1",
            slash_atomic=coins_to_atomic("500.00000000"),
            jailed=True,
            note="validator lost public reachability",
        )
        items = list_validator_evidence()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].evidence_id, evidence.evidence_id)
        self.assertEqual(items[0].evidence_type, "eligibility_failure")
        self.assertTrue(items[0].jailed)

    def test_export_validator_evidence_payload(self) -> None:
        record_validator_evidence(
            validator_id="validator-a",
            evidence_type="eligibility_failure",
            settlement_id="settle-ev-1",
            slash_atomic=coins_to_atomic("500.00000000"),
            jailed=True,
            note="validator lost public reachability",
        )
        payload = export_validator_evidence_payload()
        self.assertEqual(payload["network"], "mainnet")
        self.assertEqual(payload["chain_id"], "mainnet")
        self.assertTrue(payload["genesis_hash"])
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(len(payload["evidence"]), 1)
        self.assertEqual(payload["evidence"][0]["validator_id"], "validator-a")

    def test_validator_evidence_payload_uses_wallet_policy_network(self) -> None:
        policy = WalletPolicy(chain_network=ChainNetwork.TESTNET)

        payload = export_validator_evidence_payload(policy)

        self.assertEqual(payload["network"], "testnet")
        self.assertEqual(payload["chain_id"], "testnet")
        self.assertTrue(payload["genesis_hash"])

    def test_merge_remote_validator_evidence_payload_rejects_other_network(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Refusing validator evidence payload for network 'testnet' on 'mainnet'",
        ):
            merge_remote_validator_evidence_payload(
                {
                    "network": "testnet",
                    "chain_id": "testnet",
                    "schema_version": 1,
                    "evidence": [
                        {
                            "evidence_id": "evidence-remote",
                            "validator_id": "validator-remote",
                            "evidence_type": "eligibility_failure",
                        }
                    ],
                },
                source_url="http://85.137.164.251:52415/v1/cai/validator-evidence",
            )

        self.assertEqual(list_validator_evidence(), [])

    def test_merge_remote_validator_evidence_payload_rejects_invalid_signature(
        self,
    ) -> None:
        signing_seed = generate_signing_seed()
        payload = sign_peer_payload(
            self.peer_payload({
                "evidence": [
                    {
                        "evidence_id": "evidence-remote",
                        "validator_id": "validator-remote",
                        "evidence_type": "eligibility_failure",
                    }
                ],
            }),
            public_key_b64=public_key_b64_from_seed(signing_seed),
            signing_seed_b64=encode_bytes(signing_seed),
        )
        payload["evidence"][0]["evidence_type"] = "conflicting_attestation"

        with self.assertRaisesRegex(ValueError, "signature is invalid"):
            merge_remote_validator_evidence_payload(
                payload,
                source_url="http://85.137.164.251:52415/v1/cai/validator-evidence",
            )

    def test_merge_remote_validator_evidence_payload_rejects_unsigned_in_strict_mode(
        self,
    ) -> None:
        with patch.dict(
            "os.environ",
            {"CAI_REQUIRE_SIGNED_PEER_PAYLOADS": "1"},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "signature is missing"):
                merge_remote_validator_evidence_payload(
                    self.peer_payload({
                        "evidence": [],
                    }),
                    source_url="http://85.137.164.251:52415/v1/cai/validator-evidence",
                )

    def test_merge_remote_validator_evidence_payload_imports_peer_records(self) -> None:
        sync_validator_record(
            validator_id="validator-remote",
            wallet_id="wallet-remote",
            address="validator-remote",
            state="bonded",
            bonded_atomic=10_000,
            static_ip_confirmed=True,
            current_node_id="node-remote",
            advertised_api_host="85.137.164.251",
            advertised_data_host="85.137.164.251",
            source="peer",
            source_url="http://85.137.164.251:52415/v1/cai/validators",
        )
        imported = merge_remote_validator_evidence_payload(
            self.peer_payload({
                "evidence": [
                    {
                        "evidence_id": "evidence-remote",
                        "created_at": "2026-04-22T00:00:00+00:00",
                        "validator_id": "validator-remote",
                        "evidence_type": "conflicting_attestation",
                        "settlement_id": "settle-remote",
                        "attestation_id": "att-remote",
                        "conflicting_attestation_id": "att-local",
                        "slash_atomic": 2_000,
                        "jailed": True,
                        "note": "remote conflict",
                    }
                ]
            }),
            source_url="http://85.137.164.251:52415/v1/cai/validator-evidence",
        )
        self.assertEqual(imported, (1, 1))
        items = list_validator_evidence()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source, "peer")
        self.assertEqual(
            items[0].source_url,
            "http://85.137.164.251:52415/v1/cai/validator-evidence",
        )
        self.assertEqual(items[0].evidence_id, "evidence-remote")
        record = get_validator_record("validator-remote")
        self.assertIsNotNone(record)
        self.assertEqual(record.state, "jailed")
        self.assertEqual(record.bonded_atomic, 0)
        self.assertEqual(record.last_slash_atomic, 2_000)
        self.assertEqual(record.total_slashed_atomic, 2_000)

    def test_merge_remote_validator_evidence_summary_payload_imports_peer_records(self) -> None:
        sync_validator_record(
            validator_id="validator-remote",
            wallet_id="wallet-remote",
            address="validator-remote",
            state="bonded",
            bonded_atomic=10_000,
            static_ip_confirmed=True,
            current_node_id="node-remote",
            advertised_api_host="85.137.164.251",
            advertised_data_host="85.137.164.251",
            source="peer",
            source_url="http://85.137.164.251:52415/v1/cai/validators",
        )
        imported = merge_remote_validator_evidence_payload(
            self.peer_payload({
                "evidence": [
                    {
                        "evidenceId": "evidence-remote",
                        "createdAt": "2026-04-22T00:00:00+00:00",
                        "validatorId": "validator-remote",
                        "evidenceType": "eligibility_failure",
                        "settlementId": "settle-remote",
                        "attestationId": "att-remote",
                        "slashAtomic": 500,
                        "jailed": True,
                        "note": "remote reachability failure",
                    }
                ]
            }),
            source_url="http://85.137.164.251:52415/v1/cai/validator-evidence",
        )
        self.assertEqual(imported, (1, 1))
        items = list_validator_evidence()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].validator_id, "validator-remote")
        self.assertEqual(items[0].evidence_type, "eligibility_failure")
        record = get_validator_record("validator-remote")
        self.assertIsNotNone(record)
        self.assertEqual(record.state, "jailed")
        self.assertEqual(record.total_slashed_atomic, 500)

    def test_remote_validator_evidence_does_not_override_local_record(self) -> None:
        sync_validator_record(
            validator_id="validator-local",
            wallet_id="wallet-local",
            address="validator-local",
            state="bonded",
            bonded_atomic=10_000,
            static_ip_confirmed=True,
            current_node_id="node-local",
            advertised_api_host="85.137.164.250",
            advertised_data_host="85.137.164.250",
            source="local",
        )
        imported = merge_remote_validator_evidence_payload(
            self.peer_payload({
                "evidence": [
                    {
                        "evidence_id": "evidence-remote",
                        "created_at": "2026-04-22T00:00:00+00:00",
                        "validator_id": "validator-local",
                        "evidence_type": "eligibility_failure",
                        "settlement_id": "settle-remote",
                        "slash_atomic": 500,
                        "jailed": True,
                        "note": "remote reachability failure",
                    }
                ]
            }),
            source_url="http://85.137.164.251:52415/v1/cai/validator-evidence",
        )
        self.assertEqual(imported, (1, 0))
        record = get_validator_record("validator-local")
        self.assertIsNotNone(record)
        self.assertEqual(record.source, "local")
        self.assertEqual(record.state, "bonded")
        self.assertEqual(record.bonded_atomic, 10_000)

    def test_single_peer_evidence_does_not_apply_without_quorum(self) -> None:
        for validator_id, source_url in [
            ("validator-a", "http://85.137.164.251:52415/v1/cai/validators"),
            ("validator-b", "http://85.137.164.252:52415/v1/cai/validators"),
            ("validator-c", "http://85.137.164.253:52415/v1/cai/validators"),
        ]:
            sync_validator_record(
                validator_id=validator_id,
                wallet_id=f"wallet-{validator_id}",
                address=validator_id,
                state="bonded",
                bonded_atomic=10_000,
                static_ip_confirmed=True,
                current_node_id=f"node-{validator_id}",
                advertised_api_host="85.137.164.251",
                advertised_data_host="85.137.164.251",
                source="peer",
                source_url=source_url,
        )

        imported = merge_remote_validator_evidence_payload(
            self.peer_payload({
                "evidence": [
                    {
                        "evidence_id": "evidence-1",
                        "created_at": "2026-04-22T00:00:00+00:00",
                        "validator_id": "validator-a",
                        "reporter_validator_id": "validator-b",
                        "evidence_type": "eligibility_failure",
                        "settlement_id": "settle-remote",
                        "slash_atomic": 500,
                        "jailed": True,
                        "note": "first remote report",
                    }
                ]
            }),
            source_url="http://85.137.164.251:52415/v1/cai/validator-evidence",
        )
        self.assertEqual(imported, (1, 0))
        record = get_validator_record("validator-a")
        self.assertIsNotNone(record)
        self.assertEqual(record.state, "bonded")
        self.assertEqual(record.bonded_atomic, 10_000)

    def test_second_peer_evidence_reaches_quorum_and_applies(self) -> None:
        for validator_id, source_url in [
            ("validator-a", "http://85.137.164.251:52415/v1/cai/validators"),
            ("validator-b", "http://85.137.164.252:52415/v1/cai/validators"),
            ("validator-c", "http://85.137.164.253:52415/v1/cai/validators"),
        ]:
            sync_validator_record(
                validator_id=validator_id,
                wallet_id=f"wallet-{validator_id}",
                address=validator_id,
                state="bonded",
                bonded_atomic=10_000,
                static_ip_confirmed=True,
                current_node_id=f"node-{validator_id}",
                advertised_api_host="85.137.164.251",
                advertised_data_host="85.137.164.251",
                source="peer",
                source_url=source_url,
            )

        first = merge_remote_validator_evidence_payload(
            self.peer_payload({
                "evidence": [
                    {
                        "evidence_id": "evidence-1",
                        "created_at": "2026-04-22T00:00:00+00:00",
                        "validator_id": "validator-a",
                        "reporter_validator_id": "validator-b",
                        "evidence_type": "eligibility_failure",
                        "settlement_id": "settle-remote",
                        "slash_atomic": 500,
                        "jailed": True,
                        "note": "first remote report",
                    }
                ]
            }),
            source_url="http://85.137.164.251:52415/v1/cai/validator-evidence",
        )
        second = merge_remote_validator_evidence_payload(
            self.peer_payload({
                "evidence": [
                    {
                        "evidence_id": "evidence-2",
                        "created_at": "2026-04-22T00:00:05+00:00",
                        "validator_id": "validator-a",
                        "reporter_validator_id": "validator-c",
                        "evidence_type": "eligibility_failure",
                        "settlement_id": "settle-remote",
                        "slash_atomic": 500,
                        "jailed": True,
                        "note": "second remote report",
                    }
                ]
            }),
            source_url="http://85.137.164.252:52415/v1/cai/validator-evidence",
        )
        self.assertEqual(first, (1, 0))
        self.assertEqual(second, (1, 1))
        record = get_validator_record("validator-a")
        self.assertIsNotNone(record)
        self.assertEqual(record.state, "jailed")
        self.assertEqual(record.bonded_atomic, 0)
        items = [item for item in list_validator_evidence() if item.validator_id == "validator-a"]
        self.assertEqual(len(items), 2)
        self.assertTrue(all(item.applied_to_registry for item in items))

    def test_same_reporter_from_multiple_sources_does_not_reach_quorum(self) -> None:
        for validator_id, source_url in [
            ("validator-a", "http://85.137.164.251:52415/v1/cai/validators"),
            ("validator-b", "http://85.137.164.252:52415/v1/cai/validators"),
            ("validator-c", "http://85.137.164.253:52415/v1/cai/validators"),
        ]:
            sync_validator_record(
                validator_id=validator_id,
                wallet_id=f"wallet-{validator_id}",
                address=validator_id,
                state="bonded",
                bonded_atomic=10_000,
                static_ip_confirmed=True,
                current_node_id=f"node-{validator_id}",
                advertised_api_host="85.137.164.251",
                advertised_data_host="85.137.164.251",
                source="peer",
                source_url=source_url,
            )

        merge_remote_validator_evidence_payload(
            self.peer_payload({
                "evidence": [
                    {
                        "evidence_id": "evidence-1",
                        "created_at": "2026-04-22T00:00:00+00:00",
                        "validator_id": "validator-a",
                        "reporter_validator_id": "validator-b",
                        "evidence_type": "eligibility_failure",
                        "settlement_id": "settle-remote",
                        "slash_atomic": 500,
                        "jailed": True,
                        "note": "first remote report",
                    }
                ]
            }),
            source_url="http://85.137.164.251:52415/v1/cai/validator-evidence",
        )
        merge_remote_validator_evidence_payload(
            self.peer_payload({
                "evidence": [
                    {
                        "evidence_id": "evidence-2",
                        "created_at": "2026-04-22T00:00:05+00:00",
                        "validator_id": "validator-a",
                        "reporter_validator_id": "validator-b",
                        "evidence_type": "eligibility_failure",
                        "settlement_id": "settle-remote",
                        "slash_atomic": 500,
                        "jailed": True,
                        "note": "second remote report",
                    }
                ]
            }),
            source_url="http://85.137.164.252:52415/v1/cai/validator-evidence",
        )

        record = get_validator_record("validator-a")
        self.assertIsNotNone(record)
        self.assertEqual(record.state, "bonded")
        self.assertEqual(record.bonded_atomic, 10_000)

        cases = list_validator_evidence_cases()
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].support_mode, "validator")
        self.assertEqual(cases[0].supporting_validator_count, 1)
        self.assertEqual(cases[0].supporting_sources_count, 2)
        self.assertFalse(cases[0].applied_to_registry)

    def test_settlement_committee_quorum_ignores_non_committee_reporters(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        wallet = self._credit_wallet_local(
            wallet,
            coins_to_atomic("2.00000000"),
        )

        for validator_id, source_url in [
            ("validator-b", "http://85.137.164.252:52415/v1/cai/validators"),
            ("validator-c", "http://85.137.164.253:52415/v1/cai/validators"),
        ]:
            sync_validator_record(
                validator_id=validator_id,
                wallet_id=f"wallet-{validator_id}",
                address=validator_id,
                state="bonded",
                bonded_atomic=10_000,
                static_ip_confirmed=True,
                current_node_id=f"node-{validator_id}",
                advertised_api_host="85.137.164.251",
                advertised_data_host="85.137.164.251",
                source="peer",
                source_url=source_url,
            )

        decision = plan_funding(
            ledger=load_or_create_ledger(self.money_policy),
            wallet=wallet,
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=self.money_policy,
        )
        settlement = record_funding_settlement(
            source_wallet_id=wallet.wallet_id,
            source_wallet_address=wallet.address,
            decision=decision,
            money_policy=self.money_policy,
        )
        self.assertEqual(sorted(settlement.committee_validator_ids), ["validator-b", "validator-c"])

        sync_validator_record(
            validator_id="validator-a",
            wallet_id="wallet-validator-a",
            address="validator-a",
            state="bonded",
            bonded_atomic=10_000,
            static_ip_confirmed=True,
            current_node_id="node-validator-a",
            advertised_api_host="85.137.164.251",
            advertised_data_host="85.137.164.251",
            source="peer",
            source_url="http://85.137.164.251:52415/v1/cai/validators",
        )

        first = merge_remote_validator_evidence_payload(
            self.peer_payload({
                "evidence": [
                    {
                        "evidence_id": "committee-evidence-1",
                        "created_at": "2026-04-22T00:00:00+00:00",
                        "validator_id": "validator-a",
                        "reporter_validator_id": "validator-b",
                        "evidence_type": "eligibility_failure",
                        "settlement_id": settlement.settlement_id,
                        "slash_atomic": 500,
                        "jailed": True,
                        "note": "committee report",
                    }
                ]
            }),
            source_url="http://85.137.164.252:52415/v1/cai/validator-evidence",
        )
        second = merge_remote_validator_evidence_payload(
            self.peer_payload({
                "evidence": [
                    {
                        "evidence_id": "committee-evidence-2",
                        "created_at": "2026-04-22T00:00:05+00:00",
                        "validator_id": "validator-a",
                        "reporter_validator_id": "validator-a",
                        "evidence_type": "eligibility_failure",
                        "settlement_id": settlement.settlement_id,
                        "slash_atomic": 500,
                        "jailed": True,
                        "note": "non-committee report",
                    }
                ]
            }),
            source_url="http://85.137.164.251:52415/v1/cai/validator-evidence",
        )

        self.assertEqual(first, (1, 0))
        self.assertEqual(second, (1, 0))

        record = get_validator_record("validator-a")
        self.assertIsNotNone(record)
        self.assertEqual(record.state, "bonded")
        self.assertEqual(record.bonded_atomic, 10_000)

        cases = list_validator_evidence_cases()
        matching = [
            item
            for item in cases
            if item.settlement_id == settlement.settlement_id
            and item.validator_id == "validator-a"
        ]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0].support_scope, "settlement_committee")
        self.assertEqual(matching[0].supporting_validator_ids, ["validator-b"])
        self.assertEqual(matching[0].supporting_validator_count, 1)
        self.assertFalse(matching[0].applied_to_registry)

    def test_list_validator_evidence_cases_shows_sources_and_applied_state(self) -> None:
        for validator_id, source_url in [
            ("validator-a", "http://85.137.164.251:52415/v1/cai/validators"),
            ("validator-b", "http://85.137.164.252:52415/v1/cai/validators"),
            ("validator-c", "http://85.137.164.253:52415/v1/cai/validators"),
        ]:
            sync_validator_record(
                validator_id=validator_id,
                wallet_id=f"wallet-{validator_id}",
                address=validator_id,
                state="bonded",
                bonded_atomic=10_000,
                static_ip_confirmed=True,
                current_node_id=f"node-{validator_id}",
                advertised_api_host="85.137.164.251",
                advertised_data_host="85.137.164.251",
                source="peer",
                source_url=source_url,
            )

        merge_remote_validator_evidence_payload(
            self.peer_payload({
                "evidence": [
                    {
                        "evidence_id": "evidence-1",
                        "created_at": "2026-04-22T00:00:00+00:00",
                        "validator_id": "validator-a",
                        "reporter_validator_id": "validator-b",
                        "evidence_type": "eligibility_failure",
                        "settlement_id": "settle-remote",
                        "slash_atomic": 500,
                        "jailed": True,
                        "note": "first remote report",
                    }
                ]
            }),
            source_url="http://85.137.164.251:52415/v1/cai/validator-evidence",
        )
        merge_remote_validator_evidence_payload(
            self.peer_payload({
                "evidence": [
                    {
                        "evidence_id": "evidence-2",
                        "created_at": "2026-04-22T00:00:05+00:00",
                        "validator_id": "validator-a",
                        "reporter_validator_id": "validator-c",
                        "evidence_type": "eligibility_failure",
                        "settlement_id": "settle-remote",
                        "slash_atomic": 500,
                        "jailed": True,
                        "note": "second remote report",
                    }
                ]
            }),
            source_url="http://85.137.164.252:52415/v1/cai/validator-evidence",
        )

        cases = list_validator_evidence_cases()
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].validator_id, "validator-a")
        self.assertEqual(cases[0].support_mode, "validator")
        self.assertEqual(cases[0].support_scope, "bonded_validator_set")
        self.assertEqual(cases[0].supporting_validator_count, 2)
        self.assertEqual(
            cases[0].supporting_validator_ids, ["validator-b", "validator-c"]
        )
        self.assertEqual(cases[0].supporting_sources_count, 2)
        self.assertEqual(cases[0].required_sources, 2)
        self.assertEqual(cases[0].status, "applied")
        self.assertTrue(cases[0].quorum_reached)
        self.assertTrue(cases[0].applied_to_registry)

        penalty_cases = list_validator_penalty_cases()
        self.assertEqual(len(penalty_cases), 1)
        self.assertEqual(penalty_cases[0].status, "applied")
        self.assertIsNotNone(penalty_cases[0].finalized_at)
        self.assertIsNotNone(penalty_cases[0].applied_at)

    def test_local_evidence_finalizes_case_without_remote_registry_apply(self) -> None:
        sync_validator_record(
            validator_id="validator-local",
            wallet_id="wallet-local",
            address="validator-local",
            state="bonded",
            bonded_atomic=10_000,
            static_ip_confirmed=True,
            source="local",
        )

        record_validator_evidence(
            validator_id="validator-local",
            reporter_validator_id="validator-local",
            evidence_type="conflicting_attestation",
            settlement_id="settle-local",
            slash_atomic=500,
            jailed=True,
            note="local conflict evidence",
        )

        cases = list_validator_evidence_cases()
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].status, "finalized")
        self.assertTrue(cases[0].quorum_reached)
        self.assertFalse(cases[0].applied_to_registry)

        penalty_cases = list_validator_penalty_cases()
        self.assertEqual(len(penalty_cases), 1)
        self.assertEqual(penalty_cases[0].status, "finalized")
        self.assertIsNotNone(penalty_cases[0].finalized_at)
        self.assertIsNone(penalty_cases[0].applied_at)

    def test_record_validator_penalty_attestation_is_idempotent(self) -> None:
        sync_validator_record(
            validator_id="validator-local",
            wallet_id="wallet-local",
            address="validator-local",
            state="bonded",
            bonded_atomic=10_000,
            static_ip_confirmed=True,
            source="local",
        )

        record_validator_evidence(
            validator_id="validator-local",
            reporter_validator_id="validator-local",
            evidence_type="conflicting_attestation",
            settlement_id="settle-local",
            slash_atomic=500,
            jailed=True,
            note="local conflict evidence",
        )
        case_id = list_validator_evidence_cases()[0].case_id

        first = record_validator_penalty_attestation(
            case_id=case_id,
            validator_id="validator-local",
            accepted=True,
            note="local validator confirms penalty case",
        )
        second = record_validator_penalty_attestation(
            case_id=case_id,
            validator_id="validator-local",
            accepted=True,
            note="local validator confirms penalty case",
        )

        self.assertEqual(first.penalty_attestation_id, second.penalty_attestation_id)
        self.assertEqual(len(list_validator_penalty_attestations(case_id=case_id)), 1)

    def test_record_validator_penalty_attestation_rejects_validator_outside_scope(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        wallet = self._credit_wallet_local(
            wallet,
            coins_to_atomic("2.00000000"),
        )

        for validator_id, source_url in [
            ("validator-b", "http://85.137.164.252:52415/v1/cai/validators"),
            ("validator-c", "http://85.137.164.253:52415/v1/cai/validators"),
            ("validator-x", "http://85.137.164.254:52415/v1/cai/validators"),
            ("validator-y", "http://85.137.164.255:52415/v1/cai/validators"),
        ]:
            sync_validator_record(
                validator_id=validator_id,
                wallet_id=f"wallet-{validator_id}",
                address=validator_id,
                state="bonded",
                bonded_atomic=10_000,
                static_ip_confirmed=True,
                current_node_id=f"node-{validator_id}",
                advertised_api_host="85.137.164.251",
                advertised_data_host="85.137.164.251",
                source="peer",
                source_url=source_url,
            )

        decision = plan_funding(
            ledger=load_or_create_ledger(self.money_policy),
            wallet=wallet,
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=self.money_policy,
        )
        settlement = record_funding_settlement(
            source_wallet_id=wallet.wallet_id,
            source_wallet_address=wallet.address,
            decision=decision,
            money_policy=self.money_policy,
        )

        merge_remote_validator_evidence_payload(
            self.peer_payload({
                "evidence": [
                    {
                        "evidence_id": "committee-penalty-1",
                        "created_at": "2026-04-22T00:00:00+00:00",
                        "validator_id": "validator-a",
                        "reporter_validator_id": "validator-b",
                        "evidence_type": "eligibility_failure",
                        "settlement_id": settlement.settlement_id,
                        "slash_atomic": 500,
                        "jailed": True,
                    }
                ]
            }),
            source_url="http://85.137.164.252:52415/v1/cai/validator-evidence",
        )
        case = list_validator_penalty_cases()[0]
        case_id = case.case_id
        outsider = next(
            item
            for item in ["validator-b", "validator-c", "validator-x", "validator-y"]
            if item not in set(case.eligible_validator_ids)
        )

        with self.assertRaisesRegex(
            ValueError, "Validator is not eligible to attest this penalty case."
        ):
            record_validator_penalty_attestation(
                case_id=case_id,
                validator_id=outsider,
                accepted=True,
                note="out of scope validator",
            )

    def test_settlement_penalty_requires_explicit_committee_attestation(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        wallet = self._credit_wallet_local(
            wallet,
            coins_to_atomic("2.00000000"),
        )

        sync_validator_record(
            validator_id="validator-b",
            wallet_id="wallet-validator-b",
            address="validator-b",
            state="bonded",
            bonded_atomic=10_000,
            static_ip_confirmed=True,
            current_node_id="node-validator-b",
            advertised_api_host="85.137.164.252",
            advertised_data_host="85.137.164.252",
            source="peer",
            source_url="http://85.137.164.252:52415/v1/cai/validators",
        )

        decision = plan_funding(
            ledger=load_or_create_ledger(self.money_policy),
            wallet=wallet,
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=self.money_policy,
        )
        settlement = record_funding_settlement(
            source_wallet_id=wallet.wallet_id,
            source_wallet_address=wallet.address,
            decision=decision,
            money_policy=self.money_policy,
        )
        self.assertEqual(settlement.committee_validator_ids, ["validator-b"])

        sync_validator_record(
            validator_id="validator-a",
            wallet_id="wallet-validator-a",
            address="validator-a",
            state="bonded",
            bonded_atomic=10_000,
            static_ip_confirmed=True,
            current_node_id="node-validator-a",
            advertised_api_host="85.137.164.251",
            advertised_data_host="85.137.164.251",
            source="peer",
            source_url="http://85.137.164.251:52415/v1/cai/validators",
        )

        imported = merge_remote_validator_evidence_payload(
            self.peer_payload({
                "evidence": [
                    {
                        "evidence_id": "committee-penalty-1",
                        "created_at": "2026-04-22T00:00:00+00:00",
                        "validator_id": "validator-a",
                        "reporter_validator_id": "validator-b",
                        "evidence_type": "eligibility_failure",
                        "settlement_id": settlement.settlement_id,
                        "slash_atomic": 500,
                        "jailed": True,
                    }
                ]
            }),
            source_url="http://85.137.164.252:52415/v1/cai/validator-evidence",
        )
        self.assertEqual(imported, (1, 0))

        case = list_validator_penalty_cases()[0]
        self.assertTrue(case.evidence_quorum_reached)
        self.assertEqual(case.penalty_attestation_required, 1)
        self.assertEqual(case.penalty_attestation_count, 0)
        self.assertFalse(case.quorum_reached)
        self.assertEqual(case.status, "pending")

        record = get_validator_record("validator-a")
        self.assertIsNotNone(record)
        self.assertEqual(record.state, "bonded")

        record_validator_penalty_attestation(
            case_id=case.case_id,
            validator_id="validator-b",
            accepted=True,
            note="committee confirms penalty",
        )

        refreshed_case = list_validator_penalty_cases()[0]
        self.assertEqual(refreshed_case.penalty_attestation_count, 1)
        self.assertTrue(refreshed_case.quorum_reached)
        self.assertEqual(refreshed_case.status, "applied")

        refreshed_record = get_validator_record("validator-a")
        self.assertIsNotNone(refreshed_record)
        self.assertEqual(refreshed_record.state, "jailed")
        self.assertEqual(refreshed_record.bonded_atomic, 0)

    def test_request_remote_penalty_case_attestations_records_remote_committee_vote(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        wallet = self._credit_wallet_local(
            wallet,
            coins_to_atomic("2.00000000"),
        )

        sync_validator_record(
            validator_id="validator-b",
            wallet_id="wallet-validator-b",
            address="validator-b",
            state="bonded",
            bonded_atomic=10_000,
            static_ip_confirmed=True,
            current_node_id="node-validator-b",
            advertised_api_host="85.137.164.252",
            advertised_data_host="85.137.164.252",
            source="peer",
            source_url="http://85.137.164.252:52415/v1/cai/validators",
        )

        decision = plan_funding(
            ledger=load_or_create_ledger(self.money_policy),
            wallet=wallet,
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=self.money_policy,
        )
        settlement = record_funding_settlement(
            source_wallet_id=wallet.wallet_id,
            source_wallet_address=wallet.address,
            decision=decision,
            money_policy=self.money_policy,
        )

        sync_validator_record(
            validator_id="validator-a",
            wallet_id="wallet-validator-a",
            address="validator-a",
            state="bonded",
            bonded_atomic=10_000,
            static_ip_confirmed=True,
            current_node_id="node-validator-a",
            advertised_api_host="85.137.164.251",
            advertised_data_host="85.137.164.251",
            source="peer",
            source_url="http://85.137.164.251:52415/v1/cai/validators",
        )

        merge_remote_validator_evidence_payload(
            self.peer_payload({
                "evidence": [
                    {
                        "evidence_id": "committee-penalty-remote-1",
                        "created_at": "2026-04-22T00:00:00+00:00",
                        "validator_id": "validator-a",
                        "reporter_validator_id": "validator-b",
                        "evidence_type": "eligibility_failure",
                        "settlement_id": settlement.settlement_id,
                        "slash_atomic": 500,
                        "jailed": True,
                    }
                ]
            }),
            source_url="http://85.137.164.252:52415/v1/cai/validator-evidence",
        )
        case = list_validator_penalty_cases()[0]
        self.assertEqual(case.status, "pending")

        state_payload = {
            "nodeIdentities": {
                "node-local": {"apiHost": "127.0.0.1", "apiPort": 52425},
                "node-remote": {"apiHost": "85.137.164.252", "apiPort": 52415},
            }
        }

        def fake_post_json(url: str, payload: dict, *, timeout: int) -> dict:
            self.assertEqual(
                url,
                "http://85.137.164.252:52415/v1/cai/validator-penalty/attest",
            )
            self.assertEqual(payload["case_id"], case.case_id)
            return {
                "validatorId": "validator-b",
                "attested": True,
                "accepted": True,
                "note": "Remote committee validator accepted penalty case.",
            }

        with (
            patch("cai_compute_chain.settlement._post_json", side_effect=fake_post_json),
            patch("cai_compute_chain.settlement.sync_validator_set_from_cai_peers"),
        ):
            recorded = request_remote_penalty_case_attestations(
                cai_url="http://127.0.0.1:52425",
                state_payload=state_payload,
            )

        self.assertEqual(len(recorded), 1)
        attestations = list_validator_penalty_attestations(case_id=case.case_id)
        self.assertEqual(len(attestations), 1)
        self.assertEqual(attestations[0].validator_id, "validator-b")
        refreshed_case = list_validator_penalty_cases()[0]
        self.assertEqual(refreshed_case.status, "applied")

    def test_request_remote_penalty_case_attestations_logs_sync_failure(self) -> None:
        with (
            patch(
                "cai_compute_chain.settlement.sync_validator_set_from_cai_peers",
                side_effect=RuntimeError("validator set offline"),
            ),
            patch(
                "cai_compute_chain.node_config.load_or_create_node_config",
                return_value=SimpleNamespace(validator_address="validator-local"),
            ),
            patch(
                "cai_compute_chain.settlement.list_validator_penalty_cases",
                return_value=[],
            ),
            self.assertLogs("cai_compute_chain.settlement", level="WARNING") as logs,
        ):
            recorded = request_remote_penalty_case_attestations(
                cai_url="http://127.0.0.1:52425",
                state_payload={},
            )

        self.assertEqual(recorded, [])
        self.assertTrue(
            any(
                "validator set sync before penalty attestation requests" in line
                and "validator set offline" in line
                for line in logs.output
            )
        )

    def test_request_remote_penalty_case_attestations_logs_peer_failure(self) -> None:
        case = SimpleNamespace(
            status="pending",
            evidence_quorum_reached=True,
            eligible_validator_ids=["validator-b"],
            case_id="case-1",
            validator_id="validator-a",
            evidence_type="eligibility_failure",
            settlement_id="settle-1",
            slash_atomic=500,
            jailed=True,
        )
        record = SimpleNamespace(
            source_url="http://85.137.164.252:52415/v1/cai/validators",
            advertised_api_host=None,
        )

        with (
            patch("cai_compute_chain.settlement.sync_validator_set_from_cai_peers"),
            patch(
                "cai_compute_chain.node_config.load_or_create_node_config",
                return_value=SimpleNamespace(validator_address="validator-local"),
            ),
            patch(
                "cai_compute_chain.settlement.list_validator_penalty_cases",
                return_value=[case],
            ),
            patch(
                "cai_compute_chain.settlement.list_validator_penalty_attestations",
                return_value=[],
            ),
            patch(
                "cai_compute_chain.settlement.get_validator_record",
                return_value=record,
            ),
            patch(
                "cai_compute_chain.settlement._post_json",
                side_effect=OSError("penalty peer offline"),
            ),
            self.assertLogs("cai_compute_chain.settlement", level="WARNING") as logs,
        ):
            recorded = request_remote_penalty_case_attestations(
                cai_url="http://127.0.0.1:52425",
                state_payload={},
            )

        self.assertEqual(recorded, [])
        self.assertTrue(
            any(
                "penalty attestation request to validator validator-b" in line
                and "penalty peer offline" in line
                for line in logs.output
            )
        )

    def test_sync_validator_evidence_from_cai_peers_imports_remote_records(self) -> None:
        sync_validator_record(
            validator_id="validator-remote",
            wallet_id="wallet-remote",
            address="validator-remote",
            state="bonded",
            bonded_atomic=10_000,
            static_ip_confirmed=True,
            current_node_id="node-remote",
            advertised_api_host="85.137.164.251",
            advertised_data_host="85.137.164.251",
            source="peer",
            source_url="http://85.137.164.251:52415/v1/cai/validators",
        )
        state_payload = {
            "nodeIdentities": {
                "node-local": {
                    "apiHost": "85.137.164.250",
                    "apiPort": 52425,
                },
                "node-remote": {
                    "apiHost": "85.137.164.251",
                    "apiPort": 52415,
                },
            }
        }

        class FakeResponse:
            def __init__(self, payload: dict) -> None:
                self._payload = payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return json.dumps(self._payload).encode("utf-8")

        def fake_urlopen(url: str, timeout: int = 0):
            self.assertEqual(
                url, "http://85.137.164.251:52415/v1/cai/validator-evidence"
            )
            return FakeResponse(
                self.peer_payload({
                    "evidence": [
                        {
                            "evidence_id": "evidence-remote",
                            "created_at": "2026-04-22T00:00:00+00:00",
                            "validator_id": "validator-remote",
                            "evidence_type": "eligibility_failure",
                            "settlement_id": "settle-remote",
                            "slash_atomic": 500,
                            "jailed": True,
                            "note": "remote reachability failure",
                        }
                    ]
                })
            )

        with patch("cai_compute_chain.settlement.urlopen", side_effect=fake_urlopen):
            result = sync_validator_evidence_from_cai_peers(
                state_payload=state_payload,
                cai_url="http://127.0.0.1:52425",
            )

        self.assertEqual(result.attempted_peers, 1)
        self.assertEqual(result.successful_peers, 1)
        self.assertEqual(result.imported_records, 1)
        self.assertEqual(result.applied_records, 1)
        items = list_validator_evidence()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].validator_id, "validator-remote")
        record = get_validator_record("validator-remote")
        self.assertIsNotNone(record)
        self.assertEqual(record.state, "jailed")

    def test_sync_validator_evidence_from_cai_peers_records_peer_errors(self) -> None:
        sync_validator_record(
            validator_id="validator-remote",
            wallet_id="wallet-remote",
            address="validator-remote",
            state="bonded",
            bonded_atomic=10_000,
            static_ip_confirmed=True,
            current_node_id="node-remote",
            advertised_api_host="85.137.164.252",
            advertised_data_host="85.137.164.252",
        )
        peer_urls = [
            "http://85.137.164.251:52415/v1/cai/validator-evidence",
            "http://85.137.164.252:52415/v1/cai/validator-evidence",
        ]
        response_payload = self.peer_payload({
            "evidence": [
                {
                    "evidence_id": "evidence-remote",
                    "created_at": "2026-04-22T00:00:00+00:00",
                    "validator_id": "validator-remote",
                    "evidence_type": "eligibility_failure",
                    "settlement_id": "settle-remote",
                    "slash_atomic": 500,
                    "jailed": True,
                    "note": "remote reachability failure",
                }
            ]
        })

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return json.dumps(response_payload).encode("utf-8")

        def fake_urlopen(url: str, timeout: int = 0):
            if "85.137.164.251" in url:
                raise OSError("evidence peer offline")
            return FakeResponse()

        with (
            patch(
                "cai_compute_chain.settlement.discover_peer_cai_urls",
                return_value=peer_urls,
            ),
            patch("cai_compute_chain.settlement.urlopen", side_effect=fake_urlopen),
            patch("cai_compute_chain.settlement.sync_validator_set_from_cai_peers"),
            patch(
                "cai_compute_chain.settlement.request_remote_penalty_case_attestations"
            ),
        ):
            result = sync_validator_evidence_from_cai_peers(
                state_payload={},
                cai_url="http://127.0.0.1:52425",
            )

        self.assertEqual(result.attempted_peers, 2)
        self.assertEqual(result.successful_peers, 1)
        self.assertEqual(result.failed_peers, 1)
        self.assertEqual(
            result.failed_peer_urls,
            ["http://85.137.164.251:52415/v1/cai/validator-evidence"],
        )
        self.assertEqual(result.peer_errors[0]["errorType"], "OSError")
        self.assertIn("evidence peer offline", result.peer_errors[0]["message"])
        self.assertEqual(result.imported_records, 1)

    def test_sync_validator_evidence_from_cai_peers_reports_followup_errors(
        self,
    ) -> None:
        with (
            patch(
                "cai_compute_chain.settlement.discover_peer_cai_urls",
                return_value=[],
            ),
            patch(
                "cai_compute_chain.settlement.sync_validator_set_from_cai_peers",
                side_effect=RuntimeError("validator set unavailable"),
            ),
            patch(
                "cai_compute_chain.settlement.request_remote_penalty_case_attestations",
                side_effect=OSError("penalty attestations unavailable"),
            ),
            self.assertLogs("cai_compute_chain.settlement", level="WARNING") as logs,
        ):
            result = sync_validator_evidence_from_cai_peers(
                state_payload={},
                cai_url="http://127.0.0.1:52425",
            )

        self.assertEqual(result.attempted_peers, 0)
        self.assertEqual(
            result.validator_set_sync_error,
            {
                "errorType": "RuntimeError",
                "message": "validator set unavailable",
            },
        )
        self.assertEqual(
            result.penalty_attestation_sync_error,
            {
                "errorType": "OSError",
                "message": "penalty attestations unavailable",
            },
        )
        output = "\n".join(logs.output)
        self.assertIn(
            "validator evidence follow-up validator set sync failed",
            output,
        )
        self.assertIn("validator set unavailable", output)
        self.assertIn(
            "validator evidence follow-up penalty attestation sync failed",
            output,
        )
        self.assertIn("penalty attestations unavailable", output)

    def test_settlement_finality_uses_committee_snapshot(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        wallet = self._credit_wallet_local(
            wallet,
            coins_to_atomic("2.00000000"),
        )
        sync_validator_record(
            validator_id="validator-a",
            wallet_id="wallet-a",
            address="validator-a",
            state="bonded",
            bonded_atomic=coins_to_atomic("10000.00000000"),
            static_ip_confirmed=True,
            current_node_id="node-a",
            advertised_api_host="85.137.164.250",
            advertised_data_host="85.137.164.250",
        )
        sync_validator_record(
            validator_id="validator-b",
            wallet_id="wallet-b",
            address="validator-b",
            state="bonded",
            bonded_atomic=coins_to_atomic("10000.00000000"),
            static_ip_confirmed=True,
            current_node_id="node-b",
            advertised_api_host="85.137.164.251",
            advertised_data_host="85.137.164.251",
        )
        sync_validator_record(
            validator_id="validator-c",
            wallet_id="wallet-c",
            address="validator-c",
            state="bonded",
            bonded_atomic=coins_to_atomic("10000.00000000"),
            static_ip_confirmed=True,
            current_node_id="node-c",
            advertised_api_host="85.137.164.252",
            advertised_data_host="85.137.164.252",
        )
        ledger = load_or_create_ledger(self.money_policy)
        decision = plan_funding(
            ledger=ledger,
            wallet=wallet,
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=self.money_policy,
        )
        settlement = record_funding_settlement(
            source_wallet_id=wallet.wallet_id,
            source_wallet_address=wallet.address,
            decision=decision,
            note="committee snapshot test",
        )
        self.assertEqual(
            settlement.committee_quorum_bond_atomic,
            coins_to_atomic("20000.00000000"),
        )
        record_validator_attestation(
            settlement_id=settlement.settlement_id,
            validator_id="validator-a",
            accepted=True,
            note="accept",
        )
        pending = list_settlements()[0]
        self.assertEqual(pending.status, "pending")
        self.assertEqual(pending.accepted_bond_atomic, coins_to_atomic("10000.00000000"))
        record_validator_attestation(
            settlement_id=settlement.settlement_id,
            validator_id="validator-b",
            accepted=True,
            note="accept",
        )
        finalized = list_settlements()[0]
        self.assertEqual(finalized.status, "applied")
        self.assertEqual(finalized.accepted_attestations, 2)
        audit = finalized.balance_audit
        self.assertEqual(
            audit["quote"]["wallet_before_atomic"],
            coins_to_atomic("2.00000000"),
        )
        self.assertEqual(
            audit["quote"]["wallet_after_atomic"],
            coins_to_atomic("1.99990000"),
        )
        self.assertEqual(
            audit["debits"]["reserve_debit_atomic"],
            coins_to_atomic("1.00000000"),
        )
        self.assertTrue(audit["fees"]["worker_plus_validator_plus_ai_matches_compute"])
        self.assertTrue(audit["applied"]["reserve_delta_matches_debit"])
        self.assertTrue(audit["applied"]["source_wallet_delta_matches_debit"])
        self.assertTrue(audit["applied"]["quote_reserve_after_matches_applied"])
        self.assertTrue(audit["applied"]["quote_wallet_after_matches_applied"])

    def test_record_funding_settlement_selects_bounded_committee(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        wallet = self._credit_wallet_local(
            wallet,
            coins_to_atomic("2.00000000"),
        )
        for validator_id in ("validator-a", "validator-b", "validator-c", "validator-d"):
            sync_validator_record(
                validator_id=validator_id,
                wallet_id=f"wallet-{validator_id}",
                address=validator_id,
                state="bonded",
                bonded_atomic=coins_to_atomic("10000.00000000"),
                static_ip_confirmed=True,
                current_node_id=f"node-{validator_id}",
                advertised_api_host=f"host-{validator_id}",
                advertised_data_host=f"host-{validator_id}",
            )
        ledger = load_or_create_ledger(self.money_policy)
        decision = plan_funding(
            ledger=ledger,
            wallet=wallet,
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=self.money_policy,
        )
        committee_policy = MoneyPolicy(validator_committee_target_size=2)
        settlement = record_funding_settlement(
            source_wallet_id=wallet.wallet_id,
            source_wallet_address=wallet.address,
            decision=decision,
            note="bounded committee test",
            money_policy=committee_policy,
        )
        self.assertEqual(len(settlement.committee_validator_ids), 2)
        self.assertEqual(
            settlement.committee_total_bonded_atomic,
            coins_to_atomic("20000.00000000"),
        )
        self.assertEqual(
            settlement.committee_quorum_bond_atomic,
            1_333_333_333_334,
        )

    def test_record_funding_settlement_syncs_peer_validator_committee_when_local_set_is_empty(
        self,
    ) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        wallet = self._credit_wallet_local(
            wallet,
            coins_to_atomic("2.00000000"),
        )
        ledger = load_or_create_ledger(self.money_policy)
        decision = plan_funding(
            ledger=ledger,
            wallet=wallet,
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=self.money_policy,
        )

        def fake_sync_validator_set_from_cai_peers(**kwargs):
            sync_validator_record(
                validator_id="validator-peer",
                wallet_id="wallet-peer",
                address="validator-peer",
                state="bonded",
                bonded_atomic=coins_to_atomic("10000.00000000"),
                static_ip_confirmed=True,
                current_node_id="node-peer",
                advertised_api_host="85.137.164.250",
                advertised_data_host="85.137.164.250",
                source="peer",
                source_url="http://85.137.164.250:52415/v1/cai/validators",
            )

            class SyncResult:
                attempted_peers = 1
                successful_peers = 1
                failed_peers = 0
                imported_records = 1
                peer_urls = ["http://85.137.164.250:52415/v1/cai/validators"]
                failed_peer_urls: list[str] = []
                peer_errors: list[dict[str, str]] = []

            return SyncResult()

        with patch(
            "cai_compute_chain.settlement.sync_validator_set_from_cai_peers",
            side_effect=fake_sync_validator_set_from_cai_peers,
        ):
            settlement = record_funding_settlement(
                source_wallet_id=wallet.wallet_id,
                source_wallet_address=wallet.address,
                decision=decision,
                note="peer-backed committee test",
                money_policy=self.money_policy,
                state_payload={
                    "nodeIdentities": {
                        "node-peer": {
                            "apiHost": "85.137.164.250",
                            "apiPort": 52415,
                        }
                    }
                },
                cai_url="http://127.0.0.1:52425",
            )

        self.assertEqual(settlement.committee_validator_ids, ["validator-peer"])
        self.assertEqual(
            settlement.committee_total_bonded_atomic,
            coins_to_atomic("10000.00000000"),
        )
        self.assertEqual(settlement.committee_quorum_bond_atomic, 666_666_666_667)
        self.assertEqual(
            settlement.balance_audit["pre_settlement_validator_sync"]["status"],
            "applied",
        )
        self.assertEqual(
            settlement.balance_audit["pre_settlement_validator_sync"]["importedRecords"],
            1,
        )

    def test_record_funding_settlement_audits_validator_sync_failure(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        wallet = self._credit_wallet_local(
            wallet,
            coins_to_atomic("2.00000000"),
        )
        ledger = load_or_create_ledger(self.money_policy)
        decision = plan_funding(
            ledger=ledger,
            wallet=wallet,
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=self.money_policy,
        )

        with (
            patch(
                "cai_compute_chain.settlement.sync_validator_set_from_cai_peers",
                side_effect=OSError("validator set sync unavailable"),
            ),
            self.assertLogs("cai_compute_chain.settlement", level="WARNING") as logs,
        ):
            settlement = record_funding_settlement(
                source_wallet_id=wallet.wallet_id,
                source_wallet_address=wallet.address,
                decision=decision,
                note="validator sync failure audit test",
                money_policy=self.money_policy,
                state_payload={
                    "nodeIdentities": {
                        "node-peer": {
                            "apiHost": "85.137.164.250",
                            "apiPort": 52415,
                        }
                    }
                },
                cai_url="http://127.0.0.1:52425",
            )

        audit = settlement.balance_audit["pre_settlement_validator_sync"]
        self.assertEqual(audit["status"], "failed")
        self.assertEqual(audit["errorType"], "OSError")
        self.assertIn("validator set sync unavailable", audit["message"])
        output = "\n".join(logs.output)
        self.assertIn(
            "funding settlement pre-settlement validator sync failed",
            output,
        )
        self.assertIn("validator set sync unavailable", output)

    def test_record_validator_attestation_rejects_validator_outside_committee(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        wallet = self._credit_wallet_local(
            wallet,
            coins_to_atomic("2.00000000"),
        )
        sync_validator_record(
            validator_id="validator-a",
            wallet_id="wallet-a",
            address="validator-a",
            state="bonded",
            bonded_atomic=coins_to_atomic("10000.00000000"),
            static_ip_confirmed=True,
            current_node_id="node-a",
            advertised_api_host="host-a",
            advertised_data_host="host-a",
        )
        ledger = load_or_create_ledger(self.money_policy)
        decision = plan_funding(
            ledger=ledger,
            wallet=wallet,
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=self.money_policy,
        )
        settlement = record_funding_settlement(
            source_wallet_id=wallet.wallet_id,
            source_wallet_address=wallet.address,
            decision=decision,
            note="committee membership test",
        )
        with self.assertRaisesRegex(ValueError, "not a member of this settlement committee"):
            record_validator_attestation(
                settlement_id=settlement.settlement_id,
                validator_id="validator-z",
                accepted=True,
                note="should fail",
            )

    def test_record_worker_payouts(self) -> None:
        records = record_worker_payouts(
            settlement_id="settle-1",
            receipt_id="receipt-1",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            participants=[
                {
                    "node_id": "node-a",
                    "runner_id": "runner-a",
                    "layer_start": 0,
                    "layer_end": 20,
                    "layer_count": 20,
                    "share_bps": 8000,
                    "reward_atomic": coins_to_atomic("0.80000000"),
                    "note": "Distributed by pipeline layer share.",
                },
                {
                    "node_id": "node-b",
                    "runner_id": "runner-b",
                    "layer_start": 20,
                    "layer_end": 25,
                    "layer_count": 5,
                    "share_bps": 2000,
                    "reward_atomic": coins_to_atomic("0.20000000"),
                    "note": "Distributed by pipeline layer share.",
                },
            ],
        )

        self.assertEqual(len(records), 2)
        persisted = list_worker_payouts()
        self.assertEqual(len(persisted), 2)
        self.assertTrue(
            all(item.reward_token_code == self.money_policy.reward_token_code for item in persisted)
        )
        self.assertEqual(
            sum(item.reward_atomic for item in persisted),
            coins_to_atomic("1.00000000"),
        )

    def test_record_worker_payouts_credit_bound_local_wallet(self) -> None:
        settlement = self._create_single_validator_settlement()
        wallet = create_wallet("worker-a", "testpass", select=False)
        bind_worker_reward_address("node-a", wallet.address)

        records = record_worker_payouts(
            settlement_id=settlement.settlement_id,
            receipt_id="receipt-2",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            participants=[
                {
                    "node_id": "node-a",
                    "runner_id": "runner-a",
                    "layer_start": 0,
                    "layer_end": 10,
                    "layer_count": 10,
                    "share_bps": 10000,
                    "reward_atomic": settlement.worker_reward_atomic,
                    "note": "Distributed by pipeline layer share.",
                }
            ],
        )

        self.assertEqual(records[0].status, "pending_settlement")
        self.assertEqual(records[0].recipient_address, wallet.address)
        self.assertIsNone(records[0].credited_wallet_id)
        payout_audit = list_settlements()[0].balance_audit["payouts"]
        self.assertEqual(
            payout_audit["expected_worker_reward_atomic"],
            settlement.worker_reward_atomic,
        )
        self.assertEqual(
            payout_audit["recorded_worker_reward_atomic"],
            settlement.worker_reward_atomic,
        )
        self.assertTrue(payout_audit["worker_reward_matches_payouts"])
        self.assertTrue(
            payout_audit["worker_plus_validator_plus_ai_matches_compute"]
        )
        self.assertTrue(payout_audit["deterministic_split_matches"])
        self.assertTrue(payout_audit["settlement_fee_matches_policy"])
        self.assertTrue(payout_audit["ai_development_fee_matches_policy"])

        refreshed = resolve_wallet(wallet.wallet_id, list_wallets())
        self.assertIsNotNone(refreshed)
        self.assertEqual(refreshed.spendable_balance_atomic, 0)

        record_validator_attestation(
            settlement_id=settlement.settlement_id,
            validator_id="validator-a",
            accepted=True,
            note="accept",
        )

        credited = list_worker_payouts(settlement_id=settlement.settlement_id)[0]
        self.assertEqual(credited.status, "credited_local_wallet")
        self.assertEqual(credited.credited_wallet_id, wallet.wallet_id)

        refreshed = resolve_wallet(wallet.wallet_id, list_wallets())
        self.assertIsNotNone(refreshed)
        self.assertEqual(
            refreshed.spendable_balance_atomic, settlement.worker_reward_atomic
        )

        journal = list_journal_entries(wallet_id=wallet.wallet_id, limit=10)
        self.assertTrue(any(item.event_type == "worker_reward_credit" for item in journal))

    def test_record_worker_payouts_rejects_reward_mismatch_for_known_settlement(self) -> None:
        settlement = self._create_single_validator_settlement()

        with self.assertRaisesRegex(
            ValueError,
            "Settlement payout accounting mismatch",
        ):
            record_worker_payouts(
                settlement_id=settlement.settlement_id,
                receipt_id="receipt-mismatch",
                model_id="Qwen/Qwen3-0.6B-GGUF",
                participants=[
                    {
                        "node_id": "node-a",
                        "runner_id": "runner-a",
                        "layer_start": 0,
                        "layer_end": 10,
                        "layer_count": 10,
                        "share_bps": 10000,
                        "reward_atomic": settlement.worker_reward_atomic - 1,
                        "note": "Incorrect payout should be rejected.",
                    }
                ],
            )

        self.assertEqual(list_worker_payouts(settlement_id=settlement.settlement_id), [])
        self.assertNotIn("payouts", list_settlements()[0].balance_audit)

    def test_record_worker_payouts_rejects_nondeterministic_split(self) -> None:
        settlement = self._create_single_validator_settlement()
        first_reward = settlement.worker_reward_atomic // 2
        second_reward = settlement.worker_reward_atomic - first_reward

        with self.assertRaisesRegex(
            ValueError,
            "deterministic layer-share",
        ):
            record_worker_payouts(
                settlement_id=settlement.settlement_id,
                receipt_id="receipt-split-mismatch",
                model_id="Qwen/Qwen3-0.6B-GGUF",
                participants=[
                    {
                        "node_id": "node-a",
                        "runner_id": "runner-a",
                        "layer_start": 0,
                        "layer_end": 3,
                        "layer_count": 3,
                        "share_bps": 7500,
                        "reward_atomic": first_reward,
                        "note": "Incorrect split should be rejected.",
                    },
                    {
                        "node_id": "node-b",
                        "runner_id": "runner-b",
                        "layer_start": 3,
                        "layer_end": 4,
                        "layer_count": 1,
                        "share_bps": 2500,
                        "reward_atomic": second_reward,
                        "note": "Incorrect split should be rejected.",
                    },
                ],
            )

        self.assertEqual(list_worker_payouts(settlement_id=settlement.settlement_id), [])

    def test_record_worker_payouts_rejects_ai_fee_policy_mismatch(self) -> None:
        settlement = self._create_single_validator_settlement()
        settlement.ai_development_fee_atomic += 1
        save_settlements([settlement])

        with self.assertRaisesRegex(
            ValueError,
            "AI development fee does not match policy",
        ):
            record_worker_payouts(
                settlement_id=settlement.settlement_id,
                receipt_id="receipt-ai-fee-mismatch",
                model_id="Qwen/Qwen3-0.6B-GGUF",
                participants=[
                    {
                        "node_id": "node-a",
                        "runner_id": "runner-a",
                        "layer_start": 0,
                        "layer_end": 10,
                        "layer_count": 10,
                        "share_bps": 10000,
                        "reward_atomic": settlement.worker_reward_atomic,
                        "note": "AI fee mismatch should be rejected.",
                    }
                ],
            )

        self.assertEqual(list_worker_payouts(settlement_id=settlement.settlement_id), [])

    def test_record_worker_payouts_record_external_address_when_wallet_not_local(self) -> None:
        settlement = self._create_single_validator_settlement()
        bind_worker_reward_address("node-b", "1234567890abcdef1234567890abcdef")

        records = record_worker_payouts(
            settlement_id=settlement.settlement_id,
            receipt_id="receipt-3",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            participants=[
                {
                    "node_id": "node-b",
                    "runner_id": "runner-b",
                    "layer_start": 10,
                    "layer_end": 12,
                    "layer_count": 2,
                    "share_bps": 10000,
                    "reward_atomic": settlement.worker_reward_atomic,
                    "note": "Distributed by pipeline layer share.",
                }
            ],
        )

        self.assertEqual(records[0].status, "pending_settlement")
        self.assertEqual(records[0].recipient_address, "1234567890abcdef1234567890abcdef")
        self.assertIsNone(records[0].credited_wallet_id)

        record_validator_attestation(
            settlement_id=settlement.settlement_id,
            validator_id="validator-a",
            accepted=True,
            note="accept",
        )

        finalized = list_worker_payouts(settlement_id=settlement.settlement_id)[0]
        self.assertEqual(finalized.status, "recorded_external_address")
        self.assertEqual(finalized.recipient_address, "1234567890abcdef1234567890abcdef")
        self.assertIsNone(finalized.credited_wallet_id)
        self.assertEqual(
            chain_balance_atomic("1234567890abcdef1234567890abcdef"),
            settlement.worker_reward_atomic,
        )
        self.assertEqual(
            chain_balance_atomic(tx_fee_pool_chain_address(self.money_policy)),
            0,
        )
        self.assertEqual(
            chain_balance_atomic(compute_reserve_chain_address(self.money_policy)),
            coins_to_atomic(str(self.money_policy.compute_reserve_coins))
            - settlement.reserve_debit_atomic,
        )
        self.assertEqual(
            chain_balance_atomic(
                validator_settlement_fee_pool_chain_address(self.money_policy)
            ),
            0,
        )
        self.assertEqual(
            chain_balance_atomic("validator-a"),
            settlement.settlement_fee_atomic + settlement.tx_fee_atomic,
        )
        self.assertEqual(
            chain_balance_atomic(self.money_policy.ai_development_address),
            0,
        )
        self.assertEqual(
            settlement.worker_reward_atomic
            + settlement.settlement_fee_atomic
            + settlement.ai_development_fee_atomic,
            settlement.compute_cost_atomic,
        )
        self.assertEqual(
            chain_balance_atomic(self.money_policy.developer_treasury_address),
            coins_to_atomic(str(self.money_policy.developer_treasury_coins)),
        )
        summary = chain_summary()
        self.assertEqual(summary["txFeePoolBalanceCoins"], "0.00000000")
        self.assertEqual(
            summary["validatorSettlementFeePoolBalanceCoins"],
            "0.00000000",
        )
        self.assertTrue(summary["supplyMatchesPolicy"])
        self.assertEqual(summary["supplyDeltaCoins"], "0.00000000")
        chain_balance_audit = list_settlements()[0].balance_audit["chain_balances"]
        self.assertEqual(chain_balance_audit["recorded_transactions"], 5)
        self.assertTrue(chain_balance_audit["all_expected_deltas_match"])
        audited_roles = {item["role"] for item in chain_balance_audit["addresses"]}
        self.assertIn("worker_reward", audited_roles)
        self.assertIn("validator_tx_fee", audited_roles)
        self.assertNotIn("tx_fee_pool", audited_roles)
        self.assertIn("compute_reserve", audited_roles)
        self.assertIn("validator_settlement_fee", audited_roles)
        self.assertNotIn("validator_settlement_fee_pool", audited_roles)
        self.assertNotIn("ai_development_fund", audited_roles)
        worker_balance_audit = next(
            item
            for item in chain_balance_audit["addresses"]
            if item["role"] == "worker_reward"
        )
        self.assertEqual(
            worker_balance_audit["balance_after_atomic"],
            settlement.worker_reward_atomic,
        )
        self.assertEqual(
            worker_balance_audit["expected_delta_atomic"],
            settlement.worker_reward_atomic,
        )
        validator_fee_balance_audit = next(
            item
            for item in chain_balance_audit["addresses"]
            if item["role"] == "validator_settlement_fee"
        )
        self.assertEqual(
            validator_fee_balance_audit["balance_after_atomic"],
            settlement.settlement_fee_atomic + settlement.tx_fee_atomic,
        )
        self.assertEqual(
            validator_fee_balance_audit["expected_delta_atomic"],
            settlement.settlement_fee_atomic,
        )
        ai_development_history = chain_address_history(
            self.money_policy.ai_development_address
        )
        self.assertEqual(ai_development_history, [])
        validator_fee_history = chain_address_history("validator-a")
        self.assertEqual(len(validator_fee_history), 2)
        validator_tx_fee_history = [
            item
            for item in validator_fee_history
            if item["tx_type"] == "validator_tx_fee_payout"
        ]
        self.assertEqual(len(validator_tx_fee_history), 1)
        self.assertEqual(
            validator_tx_fee_history[0]["delta_atomic"],
            settlement.tx_fee_atomic,
        )
        validator_settlement_fee_history = [
            item
            for item in validator_fee_history
            if item["tx_type"] == "validator_settlement_fee_payout"
        ]
        self.assertEqual(len(validator_settlement_fee_history), 1)
        self.assertEqual(
            validator_settlement_fee_history[0]["tx_type"],
            "validator_settlement_fee_payout",
        )
        self.assertEqual(
            validator_settlement_fee_history[0]["delta_atomic"],
            settlement.settlement_fee_atomic,
        )
        address_history = chain_address_history("1234567890abcdef1234567890abcdef")
        self.assertEqual(len(address_history), 1)
        self.assertEqual(address_history[0]["tx_type"], "worker_reward_credit")
        self.assertEqual(
            address_history[0]["balance_after_atomic"],
            settlement.worker_reward_atomic,
        )
        settlement_history = chain_settlement_history(settlement.settlement_id)
        self.assertEqual(len(settlement_history), 5)
        self.assertEqual(
            {
                item["tx_type"]
                for item in settlement_history
            },
            {
                "settlement_wallet_debit",
                "settlement_compute_reserve_debit",
                "validator_tx_fee_payout",
                "validator_settlement_fee_payout",
                "worker_reward_credit",
            },
        )
        self.assertFalse(
            any(
                item["address"] == self.money_policy.developer_treasury_address
                and item["tx_type"] != "genesis_developer_treasury_credit"
                for item in chain_address_history(
                    self.money_policy.developer_treasury_address
                )
            )
        )
        blocks = list_chain_blocks()
        self.assertGreaterEqual(len(blocks), 2)
        self.assertEqual(blocks[0].validator_id, "genesis")
        self.assertEqual(
            {tx.tx_type for tx in blocks[0].transactions},
            {
                "genesis_compute_reserve_credit",
                "genesis_developer_contribution_fund_credit",
                "genesis_developer_treasury_credit",
            },
        )
        reward_txs = [
            tx
            for block in blocks
            for tx in block.transactions
            if tx.tx_type == "worker_reward_credit"
        ]
        self.assertEqual(len(reward_txs), 1)
        self.assertEqual(
            reward_txs[0].nonce,
            f"{settlement.settlement_id}:worker-reward:{finalized.payout_id}",
        )

    def test_validator_fee_payouts_follow_committee_bond_weight(self) -> None:
        source_wallet = create_wallet("source", "testpass", select=True)
        unlock_wallet("testpass")
        source_wallet = self._credit_wallet_local(
            source_wallet,
            coins_to_atomic("2.00000000"),
        )
        for validator_id, bonded_coins in (
            ("validator-a", "10000.00000000"),
            ("validator-b", "20000.00000000"),
            ("validator-c", "70000.00000000"),
        ):
            sync_validator_record(
                validator_id=validator_id,
                wallet_id=f"wallet-{validator_id}",
                address=validator_id,
                state="bonded",
                bonded_atomic=coins_to_atomic(bonded_coins),
                static_ip_confirmed=True,
            )
        decision = plan_funding(
            ledger=load_or_create_ledger(self.money_policy),
            wallet=source_wallet,
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=self.money_policy,
        )
        settlement = record_funding_settlement(
            source_wallet_id=source_wallet.wallet_id,
            source_wallet_address=source_wallet.address,
            decision=decision,
            money_policy=self.money_policy,
            note="stake weighted validator payout",
        )

        record_validator_attestation(
            settlement_id=settlement.settlement_id,
            validator_id="validator-c",
            accepted=True,
            note="high-stake validator reaches quorum",
        )

        self.assertEqual(
            chain_balance_atomic("validator-a"),
            coins_to_atomic("0.00201000"),
        )
        self.assertEqual(
            chain_balance_atomic("validator-b"),
            coins_to_atomic("0.00402000"),
        )
        self.assertEqual(
            chain_balance_atomic("validator-c"),
            coins_to_atomic("0.01407000"),
        )
        self.assertEqual(
            chain_balance_atomic(
                validator_settlement_fee_pool_chain_address(self.money_policy)
            ),
            0,
        )
        payout_txs = [
            item
            for item in chain_settlement_history(settlement.settlement_id)
            if item["tx_type"] == "validator_settlement_fee_payout"
        ]
        self.assertEqual(len(payout_txs), 3)
        self.assertEqual(
            sum(int(item["delta_atomic"]) for item in payout_txs),
            settlement.settlement_fee_atomic,
        )
        tx_fee_payout_txs = [
            item
            for item in chain_settlement_history(settlement.settlement_id)
            if item["tx_type"] == "validator_tx_fee_payout"
        ]
        self.assertEqual(len(tx_fee_payout_txs), 3)
        self.assertEqual(
            sum(int(item["delta_atomic"]) for item in tx_fee_payout_txs),
            settlement.tx_fee_atomic,
        )

    def test_reconcile_worker_payouts_credits_existing_unbound_records(self) -> None:
        settlement = self._create_single_validator_settlement()
        wallet = create_wallet("worker-late-bind", "testpass", select=False)
        records = record_worker_payouts(
            settlement_id=settlement.settlement_id,
            receipt_id="receipt-4",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            participants=[
                {
                    "node_id": "node-c",
                    "runner_id": "runner-c",
                    "layer_start": 0,
                    "layer_end": 4,
                    "layer_count": 4,
                    "share_bps": 10000,
                    "reward_atomic": settlement.worker_reward_atomic,
                    "note": "Distributed by pipeline layer share.",
                }
            ],
        )
        self.assertEqual(records[0].status, "unbound")

        record_validator_attestation(
            settlement_id=settlement.settlement_id,
            validator_id="validator-a",
            accepted=True,
            note="accept",
        )

        bind_worker_reward_address("node-c", wallet.address)
        reconciled = reconcile_worker_payouts()

        self.assertEqual(len(reconciled), 1)
        self.assertEqual(reconciled[0].status, "credited_local_wallet")
        self.assertEqual(reconciled[0].credited_wallet_id, wallet.wallet_id)

        refreshed = resolve_wallet(wallet.wallet_id, list_wallets())
        self.assertIsNotNone(refreshed)
        self.assertEqual(refreshed.spendable_balance_atomic, settlement.worker_reward_atomic)

    def test_apply_finalized_settlement_skips_duplicate_chain_nonce_replay(self) -> None:
        settlement = self._create_single_validator_settlement()
        worker_wallet = create_wallet("worker", "testpass", select=False)
        bind_worker_reward_address("node-replay", worker_wallet.address)
        record_worker_payouts(
            settlement_id=settlement.settlement_id,
            receipt_id="receipt-replay",
            model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            participants=[
                {
                    "node_id": "node-replay",
                    "runner_id": "runner-replay",
                    "layer_start": 0,
                    "layer_end": 4,
                    "layer_count": 4,
                    "share_bps": 10000,
                    "reward_atomic": settlement.worker_reward_atomic,
                    "note": "Distributed by pipeline layer share.",
                }
            ],
        )

        record_validator_attestation(
            settlement_id=settlement.settlement_id,
            validator_id="validator-a",
            accepted=True,
            note="accept",
        )

        reward_history = [
            item
            for item in chain_settlement_history(settlement.settlement_id)
            if item["tx_type"] == "worker_reward_credit"
        ]
        self.assertEqual(len(reward_history), 1)

        payouts = list_worker_payouts(settlement_id=settlement.settlement_id)
        self.assertEqual(len(payouts), 1)
        all_payouts = list_worker_payouts()
        for item in all_payouts:
            if item.payout_id != payouts[0].payout_id:
                continue
            item.credited_wallet_id = None
            item.status = "pending_settlement"
            break
        save_worker_payouts(all_payouts)

        applied = apply_finalized_settlement(settlement_id=settlement.settlement_id)

        self.assertIsNotNone(applied)
        reward_history_after = [
            item
            for item in chain_settlement_history(settlement.settlement_id)
            if item["tx_type"] == "worker_reward_credit"
        ]
        self.assertEqual(len(reward_history_after), 1)
        self.assertEqual(
            reward_history_after[0]["nonce"],
            f"{settlement.settlement_id}:worker-reward:{payouts[0].payout_id}",
        )


if __name__ == "__main__":
    unittest.main()

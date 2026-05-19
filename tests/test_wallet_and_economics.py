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

from cai_compute_chain.economics import (
    NetworkStatePricingSnapshot,
    plan_funding,
    quote_automatic_compute_price,
    resolve_compute_price,
)
from cai_compute_chain.chain import (
    chain_balance_atomic,
    compute_reserve_chain_address,
    ensure_chain_genesis,
    list_chain_blocks,
    make_chain_transaction,
    record_chain_transaction,
    tx_fee_pool_chain_address,
    validate_chain_blocks,
)
from cai_compute_chain.model import (
    CaiNetworkConfig,
    ChainNetwork,
    MoneyPolicy,
    NetworkModelPolicy,
    PaymentPreference,
    WalletPolicy,
)
from cai_compute_chain.validators import sync_validator_record
from cai_compute_chain.wallet import (
    apply_wallet_transfer,
    atomic_to_coins,
    ai_development_password_file_path,
    ai_development_seed_file_path,
    coins_to_atomic,
    create_wallet,
    create_seed_wallet,
    credit_wallet,
    developer_treasury_password_file_path,
    developer_treasury_seed_file_path,
    derive_seed_wallet_identity,
    ensure_local_ai_development_wallet,
    ensure_local_developer_treasury_wallet,
    get_active_wallet,
    list_journal_entries,
    load_or_create_ledger,
    load_session,
    lock_wallet,
    logout_wallet,
    restore_wallet_from_seed,
    select_active_wallet,
    unlock_wallet,
)
from cai_compute_chain.seed_phrase import (
    DEFAULT_SEED_WORD_COUNT,
    LEGACY_SEED_WORD_COUNT,
    generate_seed_phrase,
)
from cai_compute_chain.wallet_signing import (
    ADDRESS_SCHEME_FIXED_WALLET_ID_SHA256,
    DEFAULT_WALLET_HYBRID_ADDRESS_SCHEME,
    SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65,
    SIGNING_SCHEME_ML_DSA_65,
)


class WalletAndEconomicsTests(unittest.TestCase):
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

    def test_wallet_create_unlock_and_credit_write_history(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        active = get_active_wallet()
        self.assertIsNotNone(active)
        self.assertEqual(active.wallet_id, wallet.wallet_id)
        self.assertFalse(wallet.address.startswith("cai_"))
        self.assertEqual(
            wallet.signing_scheme,
            SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65,
        )
        self.assertEqual(
            wallet.address_scheme,
            DEFAULT_WALLET_HYBRID_ADDRESS_SCHEME,
        )
        self.assertEqual(len(wallet.address), 64)
        self.assertEqual(wallet.pq_signing_scheme, SIGNING_SCHEME_ML_DSA_65)
        self.assertIsNotNone(wallet.pq_public_key_b64)

        unlocked = unlock_wallet("testpass")
        self.assertEqual(unlocked.wallet_id, wallet.wallet_id)

        session = load_session()
        self.assertEqual(session.active_wallet_id, wallet.wallet_id)
        self.assertEqual(session.unlocked_wallet_id, wallet.wallet_id)

        credited = credit_wallet(wallet.wallet_id, coins_to_atomic("5.00000000"))
        self.assertEqual(
            atomic_to_coins(credited.spendable_balance_atomic, self.money_policy),
            "5.00000000",
        )

        history = list_journal_entries(wallet_id=wallet.wallet_id)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].event_type, "wallet_credit")
        self.assertEqual(history[1].event_type, "wallet_created")

        lock_wallet()
        self.assertIsNone(load_session().unlocked_wallet_id)

    def test_seed_wallet_create_and_restore_is_deterministic(self) -> None:
        wallet, seed_phrase = create_seed_wallet("seeded", "testpass", select=True)
        self.assertIsNotNone(wallet.seed_fingerprint)
        self.assertEqual(wallet.seed_word_count, DEFAULT_SEED_WORD_COUNT)
        self.assertEqual(len(seed_phrase.split()), DEFAULT_SEED_WORD_COUNT)

        with self.assertRaises(ValueError):
            restore_wallet_from_seed(
                "duplicate",
                "otherpass",
                seed_phrase=seed_phrase,
            )

        other_tempdir = tempfile.TemporaryDirectory()
        repo_patch = patch(
            "cai_compute_chain.wallet.repo_root",
            return_value=Path(other_tempdir.name),
        )
        repo_patch.start()
        try:
            restored = restore_wallet_from_seed(
                "seeded-copy",
                "testpass",
                seed_phrase=seed_phrase,
                select=True,
            )
            self.assertEqual(restored.wallet_id, wallet.wallet_id)
            self.assertEqual(restored.address, wallet.address)
            self.assertEqual(restored.pq_public_key_b64, wallet.pq_public_key_b64)
        finally:
            repo_patch.stop()
            other_tempdir.cleanup()

    def test_legacy_seed_wallet_restore_still_accepts_12_words(self) -> None:
        seed_phrase = generate_seed_phrase(LEGACY_SEED_WORD_COUNT)
        wallet = restore_wallet_from_seed(
            "legacy-seeded",
            "testpass",
            seed_phrase=seed_phrase,
            select=True,
        )

        self.assertIsNotNone(wallet.seed_fingerprint)
        self.assertEqual(wallet.seed_word_count, LEGACY_SEED_WORD_COUNT)
        self.assertEqual(len(seed_phrase.split()), LEGACY_SEED_WORD_COUNT)
        self.assertEqual(wallet.address_scheme, DEFAULT_WALLET_HYBRID_ADDRESS_SCHEME)
        self.assertEqual(wallet.signing_scheme, SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65)

    def test_switching_active_wallet_clears_unlocked_session(self) -> None:
        first = create_wallet("first", "pass1", select=True)
        second = create_wallet("second", "pass2")

        unlock_wallet("pass1", selector=first.wallet_id)
        session = load_session()
        self.assertEqual(session.active_wallet_id, first.wallet_id)
        self.assertEqual(session.unlocked_wallet_id, first.wallet_id)

        select_active_wallet(second.wallet_id)
        session = load_session()
        self.assertEqual(session.active_wallet_id, second.wallet_id)
        self.assertIsNone(session.unlocked_wallet_id)

    def test_logout_wallet_clears_active_and_unlocked_session(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass", selector=wallet.wallet_id)

        session = logout_wallet()
        self.assertIsNone(session.active_wallet_id)
        self.assertIsNone(session.unlocked_wallet_id)
        self.assertIsNone(get_active_wallet())

    def test_auto_funding_prefers_reserve_then_wallet_fee(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credited = credit_wallet(wallet.wallet_id, coins_to_atomic("1.50000000"))
        ledger = load_or_create_ledger(self.money_policy)

        decision = plan_funding(
            ledger=ledger,
            wallet=credited,
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=self.money_policy,
        )

        self.assertTrue(decision.can_fund)
        self.assertEqual(decision.funding_source.value, "reserve")
        self.assertEqual(
            atomic_to_coins(decision.reserve_after_atomic, self.money_policy),
            "949999999.00000000",
        )
        self.assertEqual(
            atomic_to_coins(decision.wallet_after_atomic, self.money_policy),
            "1.49990000",
        )
        self.assertEqual(
            atomic_to_coins(decision.fee_quote.worker_reward_atomic, self.money_policy),
            "0.98000000",
        )
        self.assertEqual(
            atomic_to_coins(decision.fee_quote.ai_development_fee_atomic, self.money_policy),
            "0.00000000",
        )

    def test_wallet_only_funding_uses_wallet_for_compute_and_fee(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credited = credit_wallet(wallet.wallet_id, coins_to_atomic("1.50000000"))
        ledger = load_or_create_ledger(self.money_policy)

        decision = plan_funding(
            ledger=ledger,
            wallet=credited,
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.WALLET_ONLY,
            money_policy=self.money_policy,
        )

        self.assertTrue(decision.can_fund)
        self.assertEqual(decision.funding_source.value, "wallet")
        self.assertEqual(
            atomic_to_coins(decision.reserve_after_atomic, self.money_policy),
            "950000000.00000000",
        )
        self.assertEqual(
            atomic_to_coins(decision.wallet_after_atomic, self.money_policy),
            "0.49990000",
        )

    def test_auto_funding_falls_back_to_wallet_when_reserve_is_unavailable(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        credited = credit_wallet(wallet.wallet_id, coins_to_atomic("1.50000000"))
        ledger = load_or_create_ledger(self.money_policy)
        ledger.compute_reserve_balance_atomic = 0

        decision = plan_funding(
            ledger=ledger,
            wallet=credited,
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=self.money_policy,
        )

        self.assertTrue(decision.can_fund)
        self.assertEqual(decision.funding_source.value, "wallet")
        self.assertEqual(
            atomic_to_coins(decision.wallet_after_atomic, self.money_policy),
            "0.49990000",
        )

    def test_auto_funding_can_use_reserve_for_tx_fee_when_wallet_is_empty(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        ledger = load_or_create_ledger(self.money_policy)

        decision = plan_funding(
            ledger=ledger,
            wallet=wallet,
            compute_cost_atomic=coins_to_atomic("1.00000000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=self.money_policy,
        )

        self.assertTrue(decision.can_fund)
        self.assertEqual(decision.funding_source.value, "reserve")
        self.assertEqual(
            atomic_to_coins(decision.reserve_after_atomic, self.money_policy),
            "949999998.99990000",
        )
        self.assertEqual(
            atomic_to_coins(decision.wallet_after_atomic, self.money_policy),
            "0.00000000",
        )

    def test_auto_funding_falls_back_to_wallet_after_daily_reserve_limit_is_consumed(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        wallet = credit_wallet(wallet.wallet_id, coins_to_atomic("5.00000000"))
        limited_policy = MoneyPolicy(
            daily_user_reserve_limit_coins="1.00000000",
            default_tx_fee_coins="0.00010000",
        )
        ledger = load_or_create_ledger(limited_policy)

        first = plan_funding(
            ledger=ledger,
            wallet=wallet,
            compute_cost_atomic=coins_to_atomic("0.99990000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=limited_policy,
        )
        self.assertTrue(first.can_fund)

        from cai_compute_chain.settlement import record_funding_settlement

        record_funding_settlement(
            source_wallet_id=wallet.wallet_id,
            source_wallet_address=wallet.address,
            decision=first,
            money_policy=limited_policy,
        )

        second = plan_funding(
            ledger=ledger,
            wallet=wallet,
            compute_cost_atomic=coins_to_atomic("0.00020000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=limited_policy,
        )

        self.assertTrue(second.can_fund)
        self.assertEqual(second.funding_source.value, "wallet")
        self.assertIn("daily reserve allowance was exhausted", second.reason.lower())
        self.assertEqual(
            atomic_to_coins(second.daily_reserve_spent_today_atomic, limited_policy),
            "0.99990000",
        )
        self.assertEqual(
            atomic_to_coins(second.daily_reserve_remaining_atomic, limited_policy),
            "0.00010000",
        )

    def test_reserve_only_funding_rejects_when_daily_reserve_limit_is_consumed(self) -> None:
        wallet = create_wallet("main", "testpass", select=True)
        unlock_wallet("testpass")
        wallet = credit_wallet(wallet.wallet_id, coins_to_atomic("5.00000000"))
        limited_policy = MoneyPolicy(
            daily_user_reserve_limit_coins="1.00000000",
            default_tx_fee_coins="0.00010000",
        )
        ledger = load_or_create_ledger(limited_policy)

        from cai_compute_chain.settlement import record_funding_settlement

        first = plan_funding(
            ledger=ledger,
            wallet=wallet,
            compute_cost_atomic=coins_to_atomic("0.99990000"),
            payment_preference=PaymentPreference.RESERVE_ONLY,
            money_policy=limited_policy,
        )
        self.assertTrue(first.can_fund)
        record_funding_settlement(
            source_wallet_id=wallet.wallet_id,
            source_wallet_address=wallet.address,
            decision=first,
            money_policy=limited_policy,
        )

        second = plan_funding(
            ledger=ledger,
            wallet=wallet,
            compute_cost_atomic=coins_to_atomic("0.00020000"),
            payment_preference=PaymentPreference.RESERVE_ONLY,
            money_policy=limited_policy,
        )

        self.assertFalse(second.can_fund)
        self.assertIn("Daily reserve allowance", second.reason)
        self.assertEqual(
            atomic_to_coins(second.daily_reserve_spent_today_atomic, limited_policy),
            "0.99990000",
        )

    def test_daily_reserve_limit_counts_same_client_ip_across_wallets(self) -> None:
        limited_policy = MoneyPolicy(
            daily_user_reserve_limit_coins="10.00000000",
            daily_ip_reserve_limit_coins="1.00000000",
            default_tx_fee_coins="0.00010000",
        )
        first_wallet = create_wallet("first", "testpass1", select=True)
        first_wallet = credit_wallet(
            first_wallet.wallet_id,
            coins_to_atomic("5.00000000"),
        )
        ledger = load_or_create_ledger(limited_policy)

        first = plan_funding(
            ledger=ledger,
            wallet=first_wallet,
            compute_cost_atomic=coins_to_atomic("0.99990000"),
            payment_preference=PaymentPreference.AUTO,
            money_policy=limited_policy,
            reserve_client_ip="203.0.113.42",
        )

        self.assertTrue(first.can_fund)
        self.assertEqual(first.funding_source.value, "reserve")
        self.assertIn(
            f"wallet:{first_wallet.wallet_id}",
            first.reserve_limit_identity_keys,
        )
        self.assertTrue(
            any(
                identity_key.startswith("ip:")
                for identity_key in first.reserve_limit_identity_keys
            )
        )
        self.assertTrue(
            any(
                identity_key.startswith("ip-prefix:")
                for identity_key in first.reserve_limit_identity_keys
            )
        )
        self.assertIsNotNone(first.reserve_client_ip_hash)

        from cai_compute_chain.settlement import record_funding_settlement

        record_funding_settlement(
            source_wallet_id=first_wallet.wallet_id,
            source_wallet_address=first_wallet.address,
            decision=first,
            money_policy=limited_policy,
        )

        second_wallet = create_wallet("second", "testpass2", select=True)
        second_wallet = credit_wallet(
            second_wallet.wallet_id,
            coins_to_atomic("5.00000000"),
        )
        second = plan_funding(
            ledger=ledger,
            wallet=second_wallet,
            compute_cost_atomic=coins_to_atomic("0.00020000"),
            payment_preference=PaymentPreference.RESERVE_ONLY,
            money_policy=limited_policy,
            reserve_client_ip="203.0.113.42",
        )

        self.assertFalse(second.can_fund)
        self.assertIn("Daily reserve allowance", second.reason)
        self.assertEqual(
            atomic_to_coins(second.daily_reserve_limit_atomic, limited_policy),
            "1.00000000",
        )
        self.assertEqual(
            atomic_to_coins(second.daily_reserve_spent_today_atomic, limited_policy),
            "0.99990000",
        )
        self.assertEqual(
            atomic_to_coins(second.daily_reserve_remaining_atomic, limited_policy),
            "0.00010000",
        )

    def test_daily_reserve_limit_reads_chain_entries_for_wallet_policy(self) -> None:
        wallet_policy = WalletPolicy(wallet_data_dirname=".tmp-chain-reserve-wallet")
        wallet = create_wallet(
            "chain-reserve",
            "testpass",
            select=True,
            wallet_policy=wallet_policy,
        )
        wallet = credit_wallet(
            wallet.wallet_id,
            coins_to_atomic("5.00000000"),
            wallet_policy=wallet_policy,
        )
        limited_policy = MoneyPolicy(
            daily_user_reserve_limit_coins="1.00000000",
            default_tx_fee_coins="0.00010000",
        )
        ledger = load_or_create_ledger(limited_policy, wallet_policy)
        spent_atomic = coins_to_atomic("0.99990000", limited_policy)
        tx = make_chain_transaction(
            tx_type="settlement_compute_reserve_debit",
            address=compute_reserve_chain_address(limited_policy),
            delta_atomic=-spent_atomic,
            settlement_id="settlement-chain-spent",
            wallet_id=f"system-compute-reserve-{limited_policy.chain_network.value}",
            metadata={
                "funding_source": "reserve",
                "source_wallet_id": wallet.wallet_id,
                "source_wallet_address": wallet.address,
                "compute_cost_atomic": spent_atomic,
                "reward_token_code": limited_policy.reward_token_code,
            },
            chain_id=limited_policy.chain_network.value,
        )
        self.assertTrue(record_chain_transaction(tx, policy=wallet_policy))

        decision = plan_funding(
            ledger=ledger,
            wallet=wallet,
            compute_cost_atomic=coins_to_atomic("0.00020000", limited_policy),
            payment_preference=PaymentPreference.AUTO,
            money_policy=limited_policy,
            wallet_policy=wallet_policy,
        )

        self.assertTrue(decision.can_fund)
        self.assertEqual(decision.funding_source.value, "wallet")
        self.assertEqual(
            atomic_to_coins(decision.daily_reserve_spent_today_atomic, limited_policy),
            "0.99990000",
        )

    def test_wallet_and_network_defaults_switch_to_testnet(self) -> None:
        with patch.dict("os.environ", {"CAI_CHAIN_NETWORK": "testnet"}):
            money_policy = MoneyPolicy()
            wallet_policy = WalletPolicy()
            network_config = CaiNetworkConfig()

        self.assertEqual(money_policy.chain_network, ChainNetwork.TESTNET)
        self.assertEqual(wallet_policy.wallet_data_dirname, ".cai-local-testnet")
        self.assertEqual(network_config.namespace, "cai-ai-testnet")
        self.assertEqual(network_config.default_api_port, 52515)
        self.assertEqual(network_config.default_libp2p_port, 52518)
        self.assertEqual(network_config.bootstrap_peers, ())

    def test_bootstrap_peers_can_be_extended_without_changing_network_preset(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "CAI_BOOTSTRAP_PEERS": "",
                "EXO_BOOTSTRAP_PEERS": "",
                "CAI_EXTRA_BOOTSTRAP_PEERS": (
                    "/ip4/198.51.100.20/tcp/52416/p2p/peer-a,"
                    "/dns4/seed-b.example/tcp/52416/p2p/peer-b"
                ),
            },
        ):
            network_config = CaiNetworkConfig()

        self.assertEqual(network_config.chain_network, ChainNetwork.MAINNET)
        self.assertIn("/ip4/192.145.29.212/tcp/52416", network_config.bootstrap_peers)
        self.assertIn(
            "/ip4/198.51.100.20/tcp/52416/p2p/peer-a",
            network_config.bootstrap_peers,
        )
        self.assertIn(
            "/dns4/seed-b.example/tcp/52416/p2p/peer-b",
            network_config.bootstrap_peers,
        )

    def test_mainnet_owner_treasury_can_be_configured_from_environment(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "CAI_CHAIN_NETWORK": "mainnet",
                "CAI_DEVELOPER_TREASURY_WALLET_ID": "owner-wallet-id",
                "CAI_DEVELOPER_TREASURY_ADDRESS": "owner-address",
            },
        ):
            money_policy = MoneyPolicy()

        self.assertEqual(money_policy.developer_treasury_wallet_id, "owner-wallet-id")
        self.assertEqual(money_policy.developer_treasury_address, "owner-address")
        self.assertEqual(money_policy.compute_reserve_coins, 850_000_000)
        self.assertEqual(money_policy.developer_treasury_coins, 50_000_000)
        self.assertEqual(money_policy.developer_contribution_fund_coins, 100_000_000)
        self.assertEqual(
            (
                money_policy.compute_reserve_coins
                + money_policy.developer_treasury_coins
                + money_policy.developer_contribution_fund_coins
            ),
            money_policy.total_supply_coins,
        )

    def test_local_developer_treasury_provision_reads_secret_files_and_credits_wallet(self) -> None:
        seed_phrase = generate_seed_phrase()
        identity = derive_seed_wallet_identity(seed_phrase)
        money_policy = MoneyPolicy(
            developer_treasury_wallet_id=identity.wallet_id,
            developer_treasury_address=identity.address,
            developer_treasury_coins=500,
        )
        wallet_policy = WalletPolicy(wallet_data_dirname=".tmp-dev-treasury-wallet")
        developer_treasury_seed_file_path(wallet_policy).write_text(seed_phrase, encoding="utf-8")
        developer_treasury_password_file_path(wallet_policy).write_text(
            "dev-treasury-pass",
            encoding="utf-8",
        )

        wallet = ensure_local_developer_treasury_wallet(
            money_policy=money_policy,
            wallet_policy=wallet_policy,
        )
        ledger = load_or_create_ledger(money_policy, wallet_policy)

        self.assertEqual(wallet.wallet_id, identity.wallet_id)
        self.assertEqual(wallet.address, identity.address)
        self.assertEqual(wallet.address_scheme, DEFAULT_WALLET_HYBRID_ADDRESS_SCHEME)
        self.assertEqual(wallet.signing_scheme, SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65)
        self.assertEqual(wallet.spendable_balance_atomic, coins_to_atomic("500", money_policy))
        self.assertTrue(ledger.developer_treasury_provisioned_locally)
        self.assertEqual(ledger.developer_treasury_wallet_id, identity.wallet_id)
        self.assertEqual(ledger.developer_treasury_address, identity.address)

    def test_local_developer_treasury_preserves_explicit_legacy_fixed_address(self) -> None:
        seed_phrase = generate_seed_phrase()
        identity = derive_seed_wallet_identity(seed_phrase)
        legacy_address = "abcd1234abcd1234abcd1234abcd1234"
        money_policy = MoneyPolicy(
            developer_treasury_wallet_id=identity.wallet_id,
            developer_treasury_address=legacy_address,
            developer_treasury_coins=500,
        )
        wallet_policy = WalletPolicy(wallet_data_dirname=".tmp-dev-treasury-legacy")
        developer_treasury_seed_file_path(wallet_policy).write_text(
            seed_phrase,
            encoding="utf-8",
        )
        developer_treasury_password_file_path(wallet_policy).write_text(
            "dev-treasury-pass",
            encoding="utf-8",
        )

        wallet = ensure_local_developer_treasury_wallet(
            money_policy=money_policy,
            wallet_policy=wallet_policy,
        )

        self.assertEqual(wallet.wallet_id, identity.wallet_id)
        self.assertEqual(wallet.address, legacy_address)
        self.assertEqual(wallet.address_scheme, ADDRESS_SCHEME_FIXED_WALLET_ID_SHA256)

    def test_local_ai_development_wallet_provision_reads_secret_files(self) -> None:
        seed_phrase = generate_seed_phrase()
        identity = derive_seed_wallet_identity(seed_phrase)
        money_policy = MoneyPolicy(
            ai_development_wallet_id=identity.wallet_id,
            ai_development_address=identity.address,
        )
        wallet_policy = WalletPolicy(wallet_data_dirname=".tmp-ai-dev-wallet")
        ai_development_seed_file_path(wallet_policy).write_text(
            seed_phrase,
            encoding="utf-8",
        )
        ai_development_password_file_path(wallet_policy).write_text(
            "ai-dev-pass",
            encoding="utf-8",
        )

        wallet = ensure_local_ai_development_wallet(
            money_policy=money_policy,
            wallet_policy=wallet_policy,
        )
        ledger = load_or_create_ledger(money_policy, wallet_policy)

        self.assertEqual(wallet.wallet_id, identity.wallet_id)
        self.assertEqual(wallet.address, identity.address)
        self.assertEqual(wallet.address_scheme, DEFAULT_WALLET_HYBRID_ADDRESS_SCHEME)
        self.assertEqual(wallet.signing_scheme, SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65)
        self.assertEqual(wallet.spendable_balance_atomic, 0)
        self.assertTrue(ledger.ai_development_provisioned_locally)
        self.assertEqual(ledger.ai_development_wallet_id, identity.wallet_id)
        self.assertEqual(ledger.ai_development_address, identity.address)

    def test_local_wallet_transfer_credits_known_recipient(self) -> None:
        sender = create_wallet("sender", "pass1", select=True)
        recipient = create_wallet("recipient", "pass2")
        credit_wallet(sender.wallet_id, coins_to_atomic("5.00000000"))

        sender_after, recipient_after = apply_wallet_transfer(
            sender_wallet_id=sender.wallet_id,
            recipient_address=recipient.address,
            amount_atomic=coins_to_atomic("2.00000000"),
            tx_fee_atomic=coins_to_atomic("0.00010000"),
        )

        self.assertIsNotNone(recipient_after)
        self.assertEqual(
            atomic_to_coins(sender_after.spendable_balance_atomic, self.money_policy),
            "2.99990000",
        )
        self.assertEqual(
            atomic_to_coins(recipient_after.spendable_balance_atomic, self.money_policy),
            "2.00000000",
        )

    def test_wallet_transfer_uses_chain_balance_when_sender_has_chain_state(self) -> None:
        wallet_policy = WalletPolicy(wallet_data_dirname=".tmp-chain-transfer-wallet")
        sender = create_wallet(
            "sender",
            "pass1",
            select=True,
            wallet_policy=wallet_policy,
        )
        recipient = create_wallet(
            "recipient",
            "pass2",
            wallet_policy=wallet_policy,
        )
        unlock_wallet("pass1", selector=sender.wallet_id, wallet_policy=wallet_policy)
        initial_credit = make_chain_transaction(
            tx_type="worker_reward_credit",
            address=sender.address,
            delta_atomic=coins_to_atomic("5.00000000"),
            wallet_id=sender.wallet_id,
            payout_id="payout-chain-transfer",
            nonce="initial-credit",
        )
        self.assertTrue(record_chain_transaction(initial_credit, policy=wallet_policy))

        sender_after, recipient_after = apply_wallet_transfer(
            sender_wallet_id=sender.wallet_id,
            recipient_address=recipient.address,
            amount_atomic=coins_to_atomic("2.00000000"),
            tx_fee_atomic=coins_to_atomic("0.00010000"),
            wallet_policy=wallet_policy,
        )

        self.assertIsNotNone(recipient_after)
        self.assertEqual(
            atomic_to_coins(sender_after.spendable_balance_atomic, self.money_policy),
            "2.99990000",
        )
        self.assertEqual(
            atomic_to_coins(recipient_after.spendable_balance_atomic, self.money_policy),
            "2.00000000",
        )
        self.assertEqual(
            atomic_to_coins(
                chain_balance_atomic(sender.address, wallet_policy),
                self.money_policy,
            ),
            "2.99990000",
        )
        self.assertEqual(
            atomic_to_coins(
                chain_balance_atomic(recipient.address, wallet_policy),
                self.money_policy,
            ),
            "2.00000000",
        )
        self.assertEqual(
            atomic_to_coins(
                chain_balance_atomic(
                    tx_fee_pool_chain_address(self.money_policy),
                    wallet_policy,
                ),
                self.money_policy,
            ),
            "0.00010000",
        )
        blocks = list_chain_blocks(wallet_policy)
        debit_txs = [
            tx
            for block in blocks
            for tx in block.transactions
            if tx.tx_type == "wallet_transfer_debit"
        ]
        self.assertEqual(len(debit_txs), 1)
        self.assertIsNotNone(debit_txs[0].public_key_b64)
        self.assertIsNotNone(debit_txs[0].signature_b64)
        self.assertTrue(debit_txs[0].metadata.get("signature_required"))
        self.assertEqual(
            debit_txs[0].metadata.get("address_scheme"),
            DEFAULT_WALLET_HYBRID_ADDRESS_SCHEME,
        )
        self.assertEqual(
            debit_txs[0].metadata.get("pq_signature_scheme"),
            SIGNING_SCHEME_ML_DSA_65,
        )
        self.assertTrue(debit_txs[0].metadata.get("pq_public_key_b64"))
        self.assertTrue(debit_txs[0].metadata.get("pq_signature_b64"))
        self.assertEqual(validate_chain_blocks(blocks), [])

    def test_wallet_transfer_pays_tx_fee_to_validator_committee(self) -> None:
        wallet_policy = WalletPolicy(
            wallet_data_dirname=".tmp-chain-transfer-validator-fee"
        )
        sender = create_wallet(
            "sender",
            "pass1",
            select=True,
            wallet_policy=wallet_policy,
        )
        recipient = create_wallet(
            "recipient",
            "pass2",
            wallet_policy=wallet_policy,
        )
        validator_wallet = create_wallet(
            "validator",
            "pass3",
            wallet_policy=wallet_policy,
        )
        unlock_wallet("pass1", selector=sender.wallet_id, wallet_policy=wallet_policy)
        sync_validator_record(
            validator_id=validator_wallet.address,
            wallet_id=validator_wallet.wallet_id,
            address=validator_wallet.address,
            state="bonded",
            bonded_atomic=coins_to_atomic("10000.00000000"),
            static_ip_confirmed=True,
            policy=wallet_policy,
        )
        initial_credit = make_chain_transaction(
            tx_type="worker_reward_credit",
            address=sender.address,
            delta_atomic=coins_to_atomic("5.00000000"),
            wallet_id=sender.wallet_id,
            payout_id="payout-chain-transfer-validator-fee",
            nonce="initial-credit",
        )
        self.assertTrue(record_chain_transaction(initial_credit, policy=wallet_policy))

        apply_wallet_transfer(
            sender_wallet_id=sender.wallet_id,
            recipient_address=recipient.address,
            amount_atomic=coins_to_atomic("2.00000000"),
            tx_fee_atomic=coins_to_atomic("0.00010000"),
            wallet_policy=wallet_policy,
        )

        self.assertEqual(
            atomic_to_coins(
                chain_balance_atomic(validator_wallet.address, wallet_policy),
                self.money_policy,
            ),
            "0.00010000",
        )
        self.assertEqual(
            atomic_to_coins(
                chain_balance_atomic(
                    tx_fee_pool_chain_address(self.money_policy),
                    wallet_policy,
                ),
                self.money_policy,
            ),
            "0.00000000",
        )
        fee_txs = [
            tx
            for block in list_chain_blocks(wallet_policy)
            for tx in block.transactions
            if tx.tx_type == "wallet_transfer_validator_fee_payout"
        ]
        self.assertEqual(len(fee_txs), 1)
        self.assertEqual(fee_txs[0].address, validator_wallet.address)
        self.assertEqual(fee_txs[0].delta_atomic, coins_to_atomic("0.00010000"))
        self.assertEqual(validate_chain_blocks(list_chain_blocks(wallet_policy)), [])

    def test_wallet_transfer_rejects_local_cache_after_chain_init(self) -> None:
        wallet_policy = WalletPolicy(wallet_data_dirname=".tmp-chain-transfer-empty")
        sender = create_wallet(
            "sender",
            "pass1",
            select=True,
            wallet_policy=wallet_policy,
        )
        recipient = create_wallet(
            "recipient",
            "pass2",
            wallet_policy=wallet_policy,
        )
        credit_wallet(
            sender.wallet_id,
            coins_to_atomic("5.00000000"),
            wallet_policy=wallet_policy,
        )
        ensure_chain_genesis(policy=wallet_policy)

        with self.assertRaisesRegex(ValueError, "chain balance is insufficient"):
            apply_wallet_transfer(
                sender_wallet_id=sender.wallet_id,
                recipient_address=recipient.address,
                amount_atomic=coins_to_atomic("2.00000000"),
                tx_fee_atomic=coins_to_atomic("0.00010000"),
                wallet_policy=wallet_policy,
            )

        self.assertEqual(chain_balance_atomic(sender.address, wallet_policy), 0)

    def test_automatic_price_quote_is_bounded_and_prefers_low_load_discount(self) -> None:
        ledger = load_or_create_ledger(self.money_policy)
        snapshot = NetworkStatePricingSnapshot(
            state_url="http://127.0.0.1:52425/state",
            reachable=True,
            topology_nodes=2,
            topology_connections=2,
            node_system_entries=2,
            average_cpu_usage=0.12,
            error=None,
        )

        quote = quote_automatic_compute_price(
            prompt="2+2=?",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            ledger=ledger,
            cai_url="http://127.0.0.1:52425",
            money_policy=self.money_policy,
            network_model_policy=NetworkModelPolicy(),
            snapshot=snapshot,
        )

        self.assertEqual(
            atomic_to_coins(quote.compute_cost_atomic, self.money_policy),
            "0.00100000",
        )
        self.assertFalse(quote.was_capped)
        self.assertIn("bounded", quote.reason.lower())

    def test_automatic_price_quote_caps_large_prompt(self) -> None:
        ledger = load_or_create_ledger(self.money_policy)
        snapshot = NetworkStatePricingSnapshot(
            state_url="http://127.0.0.1:52425/state",
            reachable=True,
            topology_nodes=1,
            topology_connections=0,
            node_system_entries=1,
            average_cpu_usage=0.96,
            error=None,
        )

        quote = quote_automatic_compute_price(
            prompt="x" * 20000,
            model_id="unknown/model",
            ledger=ledger,
            cai_url="http://127.0.0.1:52425",
            money_policy=self.money_policy,
            network_model_policy=NetworkModelPolicy(),
            snapshot=snapshot,
        )

        self.assertEqual(
            atomic_to_coins(quote.compute_cost_atomic, self.money_policy),
            self.money_policy.automatic_price_cap_coins,
        )
        self.assertTrue(quote.was_capped)

    def test_resolve_compute_price_uses_auto_mode_when_amount_missing(self) -> None:
        create_wallet("main", "testpass", select=True)
        ledger = load_or_create_ledger(self.money_policy)

        with patch(
            "cai_compute_chain.economics.fetch_network_state_pricing_snapshot",
            return_value=NetworkStatePricingSnapshot(
                state_url="http://127.0.0.1:52425/state",
                reachable=True,
                topology_nodes=2,
                topology_connections=2,
                node_system_entries=2,
                average_cpu_usage=0.10,
                error=None,
            ),
        ):
            resolved = resolve_compute_price(
                compute_amount_coins=None,
                prompt="Hello",
                model_id="Qwen/Qwen3-0.6B-GGUF",
                cai_url="http://127.0.0.1:52425",
                ledger=ledger,
                money_policy=self.money_policy,
                network_model_policy=NetworkModelPolicy(),
            )

        self.assertEqual(resolved.pricing_mode, "network_auto")
        self.assertIsNotNone(resolved.automatic_quote)

    def test_automatic_price_quote_uses_llm_token_budget_when_enabled(self) -> None:
        ledger = load_or_create_ledger(self.money_policy)
        snapshot = NetworkStatePricingSnapshot(
            state_url="http://127.0.0.1:52425/state",
            reachable=True,
            topology_nodes=2,
            topology_connections=2,
            node_system_entries=2,
            average_cpu_usage=0.10,
            error=None,
        )

        quote = quote_automatic_compute_price(
            prompt="Hello",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            max_output_tokens=64,
            ledger=ledger,
            cai_url="http://127.0.0.1:52425",
            money_policy=self.money_policy,
            network_model_policy=NetworkModelPolicy(),
            snapshot=snapshot,
        )

        self.assertEqual(quote.pricing_mode, "network_auto")
        self.assertEqual(quote.pricing_basis, "llm_tokens")
        self.assertEqual(quote.prompt_tokens_estimate, 5)
        self.assertEqual(quote.reserved_output_tokens, 64)
        self.assertEqual(
            quote.input_token_price_atomic,
            coins_to_atomic(self.money_policy.automatic_price_per_input_token_coins, self.money_policy),
        )
        self.assertEqual(
            quote.output_token_price_atomic,
            coins_to_atomic(self.money_policy.automatic_price_per_output_token_coins, self.money_policy),
        )
        self.assertIn("llm token allowance", quote.reason.lower())

    def test_legacy_prefixed_recipient_address_is_accepted(self) -> None:
        sender = create_wallet("sender", "pass1", select=True)
        recipient = create_wallet("recipient", "pass2")
        credit_wallet(sender.wallet_id, coins_to_atomic("2.00000000"))

        sender_after, recipient_after = apply_wallet_transfer(
            sender_wallet_id=sender.wallet_id,
            recipient_address=f"cai_{recipient.address}",
            amount_atomic=coins_to_atomic("1.00000000"),
            tx_fee_atomic=coins_to_atomic("0.00010000"),
        )

        self.assertIsNotNone(recipient_after)
        self.assertEqual(
            atomic_to_coins(sender_after.spendable_balance_atomic, self.money_policy),
            "0.99990000",
        )
        self.assertEqual(
            atomic_to_coins(recipient_after.spendable_balance_atomic, self.money_policy),
            "1.00000000",
        )

    def test_external_wallet_transfer_only_debits_sender(self) -> None:
        sender = create_wallet("sender", "pass1", select=True)
        credit_wallet(sender.wallet_id, coins_to_atomic("3.00000000"))

        sender_after, recipient_after = apply_wallet_transfer(
            sender_wallet_id=sender.wallet_id,
            recipient_address="externaldeadbeef0000000000000001",
            amount_atomic=coins_to_atomic("1.25000000"),
            tx_fee_atomic=coins_to_atomic("0.00010000"),
        )

        self.assertIsNone(recipient_after)
        self.assertEqual(
            atomic_to_coins(sender_after.spendable_balance_atomic, self.money_policy),
            "1.74990000",
        )


if __name__ == "__main__":
    unittest.main()

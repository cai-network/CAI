# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.chain import (
    CHAIN_SCHEMA_VERSION,
    DEFAULT_CHAIN_SNAPSHOT_INTERVAL,
    ChainBlock,
    ChainTransaction,
    _chain_snapshot_interval,
    append_chain_block,
    chain_summary,
    chain_balance_atomic,
    chain_balance_index,
    chain_index_file_path,
    chain_snapshots_file_path,
    compute_block_hash,
    compute_reserve_chain_address,
    developer_contribution_fund_chain_address,
    ensure_chain_genesis,
    expected_genesis_hash,
    export_chain_payload,
    list_chain_blocks,
    load_chain_index,
    make_chain_transaction,
    make_genesis_block,
    merge_remote_chain_payload,
    push_chain_to_cai_peers,
    record_chain_transaction,
    save_chain_blocks,
    sync_chain_from_cai_peers,
    transaction_signing_payload,
    validate_chain_blocks,
    validator_bond_pool_chain_address,
    wallet_balance_source,
    wallet_chain_balance_or_local_atomic,
)
from cai_compute_chain.model import ChainNetwork, MoneyPolicy, WalletPolicy
from cai_compute_chain.peer_payload import sign_peer_payload
from cai_compute_chain.validators import sync_validator_record
from cai_compute_chain.wallet import WalletRecord, coins_to_atomic
from cai_compute_chain.wallet_signing import (
    ADDRESS_SCHEME_ED25519,
    DEFAULT_WALLET_HYBRID_ADDRESS_SCHEME,
    SIGNING_SCHEME_ML_DSA_65,
    address_from_public_key_b64,
    encode_bytes,
    generate_mldsa65_keypair_b64,
    generate_signing_seed,
    hybrid_address_from_public_keys_b64,
    public_key_b64_from_seed,
    sign_payload_mldsa65_b64,
    sign_payload_b64,
)


_record_chain_transaction = record_chain_transaction


def record_chain_transaction(
    transaction: ChainTransaction,
    *,
    validator_id: str | None = None,
    policy: WalletPolicy | None = None,
) -> bool:
    if (
        transaction.delta_atomic <= 0
        and transaction.tx_type != "wallet_transfer_debit"
    ):
        return _record_chain_transaction(
            transaction,
            validator_id=validator_id,
            policy=policy,
        )

    active_policy = policy or WalletPolicy()
    money_policy = MoneyPolicy(chain_network=active_policy.chain_network)
    ensure_chain_genesis(policy=active_policy, money_policy=money_policy)
    if transaction.tx_type == "wallet_transfer_debit":
        balancing_credit = make_chain_transaction(
            tx_type="test_wallet_transfer_credit",
            address=transaction.counterparty_address or ("f" * 64),
            delta_atomic=-int(transaction.delta_atomic),
            settlement_id=transaction.settlement_id,
            payout_id=transaction.payout_id,
            counterparty_address=transaction.address,
            nonce=f"{transaction.nonce or transaction.tx_id}:balancing-credit",
            metadata={"test_pair": "wallet_transfer_debit"},
            chain_id=money_policy.chain_network.value,
        )
        return (
            append_chain_block(
                [transaction, balancing_credit],
                validator_id=validator_id,
                policy=active_policy,
            )
            is not None
        )

    reserve_debit = make_chain_transaction(
        tx_type="test_compute_reserve_debit",
        address=compute_reserve_chain_address(money_policy),
        delta_atomic=-int(transaction.delta_atomic),
        settlement_id=transaction.settlement_id,
        payout_id=transaction.payout_id,
        counterparty_address=transaction.address,
        nonce=f"{transaction.nonce or transaction.tx_id}:reserve-debit",
        metadata={"test_pair": "worker_reward_credit"},
        chain_id=money_policy.chain_network.value,
    )
    return (
        append_chain_block(
            [reserve_debit, transaction],
            validator_id=validator_id,
            policy=active_policy,
        )
        is not None
    )


def record_test_validator_bond(
    *,
    policy: WalletPolicy,
    validator_address: str,
    bonded_atomic: int = 1_000,
) -> None:
    money_policy = MoneyPolicy(chain_network=policy.chain_network)
    ensure_chain_genesis(policy=policy, money_policy=money_policy)
    bond_id = f"test-bond-{validator_address[:12]}"
    append_chain_block(
        [
            make_chain_transaction(
                tx_type="test_validator_funding_debit",
                address=compute_reserve_chain_address(money_policy),
                delta_atomic=-bonded_atomic,
                counterparty_address=validator_address,
                nonce=f"{bond_id}:reserve-debit",
                chain_id=money_policy.chain_network.value,
            ),
            make_chain_transaction(
                tx_type="test_validator_funding_credit",
                address=validator_address,
                delta_atomic=bonded_atomic,
                counterparty_address=compute_reserve_chain_address(money_policy),
                nonce=f"{bond_id}:wallet-credit",
                chain_id=money_policy.chain_network.value,
            ),
            make_chain_transaction(
                tx_type="validator_bond_lock",
                address=validator_address,
                delta_atomic=-bonded_atomic,
                wallet_id="validator-wallet",
                nonce=f"{bond_id}:wallet-lock",
                metadata={
                    "validator_id": validator_address,
                    "validator_wallet_id": "validator-wallet",
                    "validator_address": validator_address,
                    "bond_atomic": bonded_atomic,
                    "reward_token_code": money_policy.reward_token_code,
                },
                chain_id=money_policy.chain_network.value,
            ),
            make_chain_transaction(
                tx_type="validator_bond_pool_credit",
                address=validator_bond_pool_chain_address(money_policy),
                delta_atomic=bonded_atomic,
                wallet_id=f"system-validator-bond-pool-{money_policy.chain_network.value}",
                counterparty_address=validator_address,
                nonce=f"{bond_id}:pool-credit",
                metadata={
                    "validator_id": validator_address,
                    "validator_wallet_id": "validator-wallet",
                    "validator_address": validator_address,
                    "bond_atomic": bonded_atomic,
                    "reward_token_code": money_policy.reward_token_code,
                },
                chain_id=money_policy.chain_network.value,
            ),
        ],
        validator_id=validator_address,
        policy=policy,
    )


def recompute_raw_block_hash(raw_block: dict) -> None:
    block = ChainBlock(
        block_id=str(raw_block["block_id"]),
        height=int(raw_block["height"]),
        created_at=str(raw_block["created_at"]),
        previous_hash=str(raw_block["previous_hash"]),
        validator_id=raw_block.get("validator_id"),
        transactions=[
            ChainTransaction(
                tx_id=str(tx["tx_id"]),
                created_at=str(tx["created_at"]),
                tx_type=str(tx["tx_type"]),
                address=str(tx["address"]),
                delta_atomic=int(tx["delta_atomic"]),
                settlement_id=tx.get("settlement_id"),
                payout_id=tx.get("payout_id"),
                wallet_id=tx.get("wallet_id"),
                counterparty_address=tx.get("counterparty_address"),
                note=tx.get("note"),
                nonce=tx.get("nonce"),
                chain_id=str(tx.get("chain_id") or ""),
                schema_version=int(tx.get("schema_version") or CHAIN_SCHEMA_VERSION),
                metadata=dict(tx.get("metadata") or {}),
            )
            for tx in raw_block["transactions"]
        ],
        block_hash="",
        chain_id=str(raw_block.get("chain_id") or ""),
        schema_version=int(raw_block.get("schema_version") or CHAIN_SCHEMA_VERSION),
    )
    raw_block["block_hash"] = compute_block_hash(block)


class FakeHttpResponse:
    def __init__(self, payload: bytes = b"{}") -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False

    def read(self) -> bytes:
        return self.payload


class ChainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo_patch = patch(
            "cai_compute_chain.wallet.repo_root",
            return_value=Path(self.tempdir.name),
        )
        self.repo_patch.start()
        self.env_patch = patch.dict(
            "os.environ",
            {
                "CAI_REQUIRE_HYBRID_PEER_PAYLOAD_SIGNATURES": "0",
                "CAI_REQUIRE_SIGNED_PEER_PAYLOADS": "0",
            },
            clear=False,
        )
        self.env_patch.start()

    def tearDown(self) -> None:
        self.env_patch.stop()
        self.repo_patch.stop()
        self.tempdir.cleanup()

    def test_chain_balance_syncs_from_remote_blocks_idempotently(self) -> None:
        source_policy = WalletPolicy(wallet_data_dirname="node-a")
        target_policy = WalletPolicy(wallet_data_dirname="node-b")
        address = "abcd1234abcd1234abcd1234abcd1234"
        tx = make_chain_transaction(
            tx_type="worker_reward_credit",
            address=address,
            delta_atomic=coins_to_atomic("0.25000000"),
            settlement_id="settlement-a",
            payout_id="payout-a",
        )

        self.assertTrue(record_chain_transaction(tx, policy=source_policy))
        self.assertFalse(record_chain_transaction(tx, policy=source_policy))

        payload = export_chain_payload(source_policy)
        imported_blocks, imported_transactions = merge_remote_chain_payload(
            payload,
            policy=target_policy,
        )
        duplicate_blocks, duplicate_transactions = merge_remote_chain_payload(
            payload,
            policy=target_policy,
        )

        self.assertEqual(imported_blocks, 2)
        self.assertEqual(imported_transactions, 5)
        self.assertEqual(duplicate_blocks, 0)
        self.assertEqual(duplicate_transactions, 0)
        self.assertEqual(
            chain_balance_atomic(address, target_policy),
            coins_to_atomic("0.25000000"),
        )

    def test_sync_chain_from_cai_peers_records_peer_errors(self) -> None:
        source_policy = WalletPolicy(wallet_data_dirname="node-a")
        target_policy = WalletPolicy(wallet_data_dirname="node-b")
        record_chain_transaction(
            make_chain_transaction(
                tx_type="worker_reward_credit",
                address="abcd1234abcd1234abcd1234abcd1234",
                delta_atomic=coins_to_atomic("0.12500000"),
                settlement_id="settlement-sync",
                payout_id="payout-sync",
            ),
            policy=source_policy,
        )
        remote_payload = json.dumps(export_chain_payload(source_policy)).encode("utf-8")
        peer_urls = [
            "http://node-a/v1/cai/chain",
            "http://node-b/v1/cai/chain",
        ]

        with (
            patch("cai_compute_chain.chain.discover_peer_cai_urls", return_value=peer_urls),
            patch(
                "cai_compute_chain.chain.urlopen",
                side_effect=[
                    OSError("node-a offline"),
                    FakeHttpResponse(remote_payload),
                ],
            ),
        ):
            result = sync_chain_from_cai_peers(
                state_payload={"nodeIdentities": {}},
                cai_url="http://127.0.0.1:52415",
                policy=target_policy,
            )

        self.assertEqual(result.attempted_peers, 2)
        self.assertEqual(result.successful_peers, 1)
        self.assertEqual(result.failed_peers, 1)
        self.assertEqual(result.failed_peer_urls, ["http://node-a/v1/cai/chain"])
        self.assertEqual(result.peer_errors[0]["errorType"], "OSError")
        self.assertIn("node-a offline", result.peer_errors[0]["message"])
        self.assertGreater(result.imported_blocks, 0)

    def test_push_chain_to_cai_peers_records_peer_errors(self) -> None:
        policy = WalletPolicy(wallet_data_dirname="node-a")
        ensure_chain_genesis(policy=policy)
        peer_urls = [
            "http://node-a/v1/cai/chain/sync",
            "http://node-b/v1/cai/chain/sync",
        ]

        with (
            patch("cai_compute_chain.chain.discover_peer_cai_urls", return_value=peer_urls),
            patch(
                "cai_compute_chain.chain.urlopen",
                side_effect=[
                    FakeHttpResponse(),
                    OSError("node-b refused chain sync"),
                ],
            ),
        ):
            result = push_chain_to_cai_peers(
                state_payload={"nodeIdentities": {}},
                cai_url="http://127.0.0.1:52415",
                policy=policy,
            )

        self.assertEqual(result.attempted_peers, 2)
        self.assertEqual(result.successful_peers, 1)
        self.assertEqual(result.failed_peers, 1)
        self.assertEqual(result.failed_peer_urls, ["http://node-b/v1/cai/chain/sync"])
        self.assertEqual(result.peer_errors[0]["errorType"], "OSError")
        self.assertIn("node-b refused chain sync", result.peer_errors[0]["message"])

    def test_chain_merge_ignores_unsigned_longer_remote_fork(self) -> None:
        remote_policy = WalletPolicy(wallet_data_dirname="node-a")
        local_policy = WalletPolicy(wallet_data_dirname="node-b")
        local_only_address = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        remote_address = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        remote_second_address = "cccccccccccccccccccccccccccccccc"

        record_chain_transaction(
            make_chain_transaction(
                tx_type="local_test_credit",
                address=local_only_address,
                delta_atomic=coins_to_atomic("1.00000000"),
                nonce="local-fork",
            ),
            policy=local_policy,
        )
        record_chain_transaction(
            make_chain_transaction(
                tx_type="remote_validator_credit",
                address=remote_address,
                delta_atomic=coins_to_atomic("2.00000000"),
                nonce="remote-a",
            ),
            policy=remote_policy,
        )
        record_chain_transaction(
            make_chain_transaction(
                tx_type="remote_client_credit",
                address=remote_second_address,
                delta_atomic=coins_to_atomic("3.00000000"),
                nonce="remote-b",
            ),
            policy=remote_policy,
        )

        imported_blocks, imported_transactions = merge_remote_chain_payload(
            export_chain_payload(remote_policy),
            policy=local_policy,
        )

        self.assertEqual(imported_blocks, 0)
        self.assertEqual(imported_transactions, 0)
        self.assertEqual(len(list_chain_blocks(local_policy)), 2)
        self.assertEqual(
            chain_balance_atomic(local_only_address, local_policy),
            coins_to_atomic("1.00000000"),
        )
        self.assertEqual(chain_balance_atomic(remote_address, local_policy), 0)
        self.assertEqual(chain_balance_atomic(remote_second_address, local_policy), 0)

    def test_chain_merge_adopts_bonded_validator_signed_remote_fork(self) -> None:
        remote_policy = WalletPolicy(wallet_data_dirname="node-a")
        local_policy = WalletPolicy(wallet_data_dirname="node-b")
        local_only_address = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        remote_address = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        remote_second_address = "cccccccccccccccccccccccccccccccc"
        signing_seed = generate_signing_seed()
        public_key_b64 = public_key_b64_from_seed(signing_seed)
        validator_address = address_from_public_key_b64(public_key_b64)

        sync_validator_record(
            validator_id=validator_address,
            wallet_id="validator-wallet",
            address=validator_address,
            state="bonded",
            bonded_atomic=1_000,
            static_ip_confirmed=True,
            current_node_id="validator-node",
            policy=local_policy,
        )
        record_test_validator_bond(
            policy=local_policy,
            validator_address=validator_address,
        )
        record_chain_transaction(
            make_chain_transaction(
                tx_type="local_test_credit",
                address=local_only_address,
                delta_atomic=coins_to_atomic("1.00000000"),
                nonce="local-fork",
            ),
            policy=local_policy,
        )
        record_chain_transaction(
            make_chain_transaction(
                tx_type="remote_validator_credit",
                address=remote_address,
                delta_atomic=coins_to_atomic("2.00000000"),
                nonce="remote-a",
            ),
            policy=remote_policy,
        )
        record_chain_transaction(
            make_chain_transaction(
                tx_type="remote_client_credit",
                address=remote_second_address,
                delta_atomic=coins_to_atomic("3.00000000"),
                nonce="remote-b",
            ),
            policy=remote_policy,
        )
        payload = sign_peer_payload(
            export_chain_payload(remote_policy),
            public_key_b64=public_key_b64,
            signing_seed_b64=encode_bytes(signing_seed),
            signer_address=validator_address,
        )

        imported_blocks, imported_transactions = merge_remote_chain_payload(
            payload,
            policy=local_policy,
        )

        self.assertEqual(imported_blocks, 2)
        self.assertEqual(imported_transactions, 4)
        self.assertEqual(len(list_chain_blocks(local_policy)), 3)
        self.assertEqual(chain_balance_atomic(local_only_address, local_policy), 0)
        self.assertEqual(
            chain_balance_atomic(remote_address, local_policy),
            coins_to_atomic("2.00000000"),
        )
        self.assertEqual(
            chain_balance_atomic(remote_second_address, local_policy),
            coins_to_atomic("3.00000000"),
        )

    def test_chain_merge_adopts_bonded_validator_signed_shorter_remote_fork(self) -> None:
        remote_policy = WalletPolicy(wallet_data_dirname="node-a")
        local_policy = WalletPolicy(wallet_data_dirname="node-b")
        local_address = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        local_second_address = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        remote_address = "cccccccccccccccccccccccccccccccc"
        signing_seed = generate_signing_seed()
        public_key_b64 = public_key_b64_from_seed(signing_seed)
        validator_address = address_from_public_key_b64(public_key_b64)

        sync_validator_record(
            validator_id=validator_address,
            wallet_id="validator-wallet",
            address=validator_address,
            state="bonded",
            bonded_atomic=1_000,
            static_ip_confirmed=True,
            current_node_id="validator-node",
            policy=local_policy,
        )
        record_test_validator_bond(
            policy=local_policy,
            validator_address=validator_address,
        )
        for address, nonce in (
            (local_address, "local-a"),
            (local_second_address, "local-b"),
        ):
            record_chain_transaction(
                make_chain_transaction(
                    tx_type="local_test_credit",
                    address=address,
                    delta_atomic=coins_to_atomic("1.00000000"),
                    nonce=nonce,
                ),
                policy=local_policy,
            )
        record_chain_transaction(
            make_chain_transaction(
                tx_type="remote_validator_credit",
                address=remote_address,
                delta_atomic=coins_to_atomic("2.00000000"),
                nonce="remote-a",
            ),
            policy=remote_policy,
        )
        payload = sign_peer_payload(
            export_chain_payload(remote_policy),
            public_key_b64=public_key_b64,
            signing_seed_b64=encode_bytes(signing_seed),
            signer_address=validator_address,
        )

        imported_blocks, imported_transactions = merge_remote_chain_payload(
            payload,
            policy=local_policy,
        )

        self.assertEqual(imported_blocks, 1)
        self.assertEqual(imported_transactions, 2)
        self.assertEqual(len(list_chain_blocks(local_policy)), 2)
        self.assertEqual(chain_balance_atomic(local_address, local_policy), 0)
        self.assertEqual(chain_balance_atomic(local_second_address, local_policy), 0)
        self.assertEqual(
            chain_balance_atomic(remote_address, local_policy),
            coins_to_atomic("2.00000000"),
        )

    def test_same_address_wallet_balance_matches_after_chain_sync(self) -> None:
        source_policy = WalletPolicy(wallet_data_dirname="node-a")
        target_policy = WalletPolicy(wallet_data_dirname="node-b")
        money_policy = MoneyPolicy()
        address = "abcd1234abcd1234abcd1234abcd1234"
        tx = make_chain_transaction(
            tx_type="worker_reward_credit",
            address=address,
            delta_atomic=coins_to_atomic("0.25000000", money_policy),
            settlement_id="settlement-a",
            payout_id="payout-a",
        )
        local_wallet = WalletRecord(
            wallet_id="wallet-a",
            name="Same Address",
            address=address,
            created_at="2026-01-01T00:00:00+00:00",
            password_salt_b64="",
            password_hash_b64="",
            spendable_balance_atomic=coins_to_atomic("12.00000000", money_policy),
        )

        self.assertTrue(record_chain_transaction(tx, policy=source_policy))
        merge_remote_chain_payload(export_chain_payload(source_policy), policy=target_policy)

        expected_balance = coins_to_atomic("0.25000000", money_policy)
        self.assertEqual(
            wallet_chain_balance_or_local_atomic(local_wallet, source_policy),
            expected_balance,
        )
        self.assertEqual(
            wallet_chain_balance_or_local_atomic(local_wallet, target_policy),
            expected_balance,
        )

    def test_chain_rejects_duplicate_nonce_for_same_address(self) -> None:
        policy = WalletPolicy(wallet_data_dirname="node-a")
        address = "abcd1234abcd1234abcd1234abcd1234"
        first_tx = make_chain_transaction(
            tx_type="wallet_transfer_credit",
            address=address,
            delta_atomic=coins_to_atomic("0.10000000"),
            wallet_id="wallet-a",
            nonce="nonce-1",
        )
        second_tx = make_chain_transaction(
            tx_type="wallet_transfer_credit",
            address=address,
            delta_atomic=coins_to_atomic("0.20000000"),
            wallet_id="wallet-a",
            nonce="nonce-1",
        )

        self.assertTrue(record_chain_transaction(first_tx, policy=policy))
        with self.assertRaises(ValueError):
            record_chain_transaction(second_tx, policy=policy)

    def test_chain_genesis_records_reserve_founder_and_developer_fund_once(self) -> None:
        policy = WalletPolicy(wallet_data_dirname="node-a")
        money_policy = MoneyPolicy()

        self.assertEqual(ensure_chain_genesis(policy=policy, money_policy=money_policy), 3)
        self.assertEqual(ensure_chain_genesis(policy=policy, money_policy=money_policy), 0)

        self.assertEqual(
            chain_balance_atomic(compute_reserve_chain_address(money_policy), policy),
            coins_to_atomic(str(money_policy.compute_reserve_coins), money_policy),
        )
        self.assertEqual(
            chain_balance_atomic(
                developer_contribution_fund_chain_address(money_policy),
                policy,
            ),
            coins_to_atomic(
                str(money_policy.developer_contribution_fund_coins),
                money_policy,
            ),
        )
        self.assertEqual(
            chain_balance_atomic(money_policy.developer_treasury_address, policy),
            coins_to_atomic(str(money_policy.developer_treasury_coins), money_policy),
        )

    def test_mainnet_genesis_uses_rotated_owner_treasury(self) -> None:
        money_policy = MoneyPolicy()
        genesis = make_genesis_block(money_policy)

        self.assertEqual(
            money_policy.developer_treasury_wallet_id,
            "f566089781403edca18c2d06c9c0af8a",
        )
        self.assertEqual(
            money_policy.developer_treasury_address,
            "6d9c5fc0ab4f5ad786881d1848800f778dc8f21473ebcff514181dfb50023881",
        )
        self.assertEqual(
            genesis.block_hash,
            "ab40ac7f1841e6a5fef442cca396d326225da22062941ef4f286c3d8ce5e9a3f",
        )

    def test_chain_genesis_repairs_stale_genesis_only_chain(self) -> None:
        policy = WalletPolicy(wallet_data_dirname="node-a")
        stale_money_policy = MoneyPolicy(
            developer_treasury_wallet_id="c66834b46f99f6d05e35364ef8a24552",
            developer_treasury_address="d72ed612f384e7279842f0d003add8b6",
        )
        current_money_policy = MoneyPolicy()

        self.assertEqual(
            ensure_chain_genesis(policy=policy, money_policy=stale_money_policy),
            3,
        )
        self.assertEqual(
            list_chain_blocks(policy)[0].block_hash,
            "57d4798d9c267b050cf750971fa339c96871fc280e4e39b23f9a50ae4a073882",
        )
        self.assertEqual(ensure_chain_genesis(policy=policy), 0)

        blocks = list_chain_blocks(policy)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].block_hash, make_genesis_block(current_money_policy).block_hash)
        self.assertEqual(
            chain_balance_atomic(current_money_policy.developer_treasury_address, policy),
            coins_to_atomic(str(current_money_policy.developer_treasury_coins), current_money_policy),
        )
        self.assertEqual(
            chain_balance_atomic(stale_money_policy.developer_treasury_address, policy),
            0,
        )

    def test_chain_merge_adopts_current_chain_over_stale_genesis_only(self) -> None:
        source_policy = WalletPolicy(wallet_data_dirname="node-a")
        target_policy = WalletPolicy(wallet_data_dirname="node-b")
        stale_money_policy = MoneyPolicy(
            developer_treasury_wallet_id="c66834b46f99f6d05e35364ef8a24552",
            developer_treasury_address="d72ed612f384e7279842f0d003add8b6",
        )
        address = "abcd1234abcd1234abcd1234abcd1234"

        ensure_chain_genesis(policy=target_policy, money_policy=stale_money_policy)
        self.assertTrue(
            record_chain_transaction(
                make_chain_transaction(
                    tx_type="worker_reward_credit",
                    address=address,
                    delta_atomic=coins_to_atomic("0.25000000"),
                    settlement_id="settlement-a",
                    payout_id="payout-a",
                ),
                policy=source_policy,
            )
        )

        imported_blocks, imported_transactions = merge_remote_chain_payload(
            export_chain_payload(source_policy),
            policy=target_policy,
        )

        self.assertEqual(imported_blocks, 2)
        self.assertEqual(imported_transactions, 5)
        self.assertEqual(
            chain_balance_atomic(address, target_policy),
            coins_to_atomic("0.25000000"),
        )

    def test_chain_genesis_tip_hash_is_deterministic_across_clean_nodes(self) -> None:
        policy_a = WalletPolicy(wallet_data_dirname="node-a")
        policy_b = WalletPolicy(wallet_data_dirname="node-b")
        money_policy = MoneyPolicy()

        self.assertEqual(ensure_chain_genesis(policy=policy_a, money_policy=money_policy), 3)
        self.assertEqual(ensure_chain_genesis(policy=policy_b, money_policy=money_policy), 3)

        chain_a = export_chain_payload(policy_a)["chain"]
        chain_b = export_chain_payload(policy_b)["chain"]

        self.assertEqual(chain_a["tip_hash"], chain_b["tip_hash"])
        self.assertEqual(chain_a["blocks"][0]["block_id"], chain_b["blocks"][0]["block_id"])

    def test_record_chain_transaction_creates_genesis_first(self) -> None:
        policy = WalletPolicy(wallet_data_dirname="node-a")
        address = "abcd1234abcd1234abcd1234abcd1234"
        tx = make_chain_transaction(
            tx_type="worker_reward_credit",
            address=address,
            delta_atomic=coins_to_atomic("0.25000000"),
            settlement_id="settlement-a",
            payout_id="payout-a",
        )

        self.assertTrue(record_chain_transaction(tx, policy=policy))
        chain = export_chain_payload(policy)["chain"]

        self.assertEqual(chain["block_count"], 2)
        self.assertEqual(chain["transaction_count"], 5)
        self.assertEqual(chain["blocks"][0]["validator_id"], "genesis")
        self.assertEqual(chain["blocks"][1]["previous_hash"], chain["blocks"][0]["block_hash"])

    def test_chain_merge_rejects_payload_from_other_network(self) -> None:
        source_policy = WalletPolicy(wallet_data_dirname="node-a")
        target_policy = WalletPolicy(wallet_data_dirname="node-b")

        ensure_chain_genesis(policy=source_policy)
        payload = export_chain_payload(source_policy)
        payload["chain"]["network"] = "testnet"

        with self.assertRaises(ValueError):
            merge_remote_chain_payload(payload, policy=target_policy)

    def test_chain_merge_rejects_invalid_signed_payload(self) -> None:
        source_policy = WalletPolicy(wallet_data_dirname="node-a")
        target_policy = WalletPolicy(wallet_data_dirname="node-b")
        signing_seed = generate_signing_seed()

        ensure_chain_genesis(policy=source_policy)
        payload = sign_peer_payload(
            export_chain_payload(source_policy),
            public_key_b64=public_key_b64_from_seed(signing_seed),
            signing_seed_b64=encode_bytes(signing_seed),
        )
        payload["chain"]["block_count"] = 99

        with self.assertRaisesRegex(ValueError, "signature is invalid"):
            merge_remote_chain_payload(payload, policy=target_policy)

    def test_chain_merge_rejects_unsigned_payload_in_strict_mode(self) -> None:
        source_policy = WalletPolicy(wallet_data_dirname="node-a")
        target_policy = WalletPolicy(wallet_data_dirname="node-b")

        ensure_chain_genesis(policy=source_policy)
        payload = export_chain_payload(source_policy)

        with patch.dict(
            "os.environ",
            {"CAI_REQUIRE_SIGNED_PEER_PAYLOADS": "1"},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "signature is missing"):
                merge_remote_chain_payload(payload, policy=target_policy)

    def test_chain_payload_network_uses_wallet_policy(self) -> None:
        policy = WalletPolicy(
            chain_network=ChainNetwork.TESTNET,
            wallet_data_dirname="node-testnet",
        )

        ensure_chain_genesis(policy=policy)
        payload = export_chain_payload(policy)

        self.assertEqual(payload["chain"]["network"], "testnet")

    def test_chain_export_includes_chain_id_and_schema_version(self) -> None:
        policy = WalletPolicy(wallet_data_dirname="node-a")

        ensure_chain_genesis(policy=policy)
        payload = export_chain_payload(policy)
        block = payload["chain"]["blocks"][0]
        transaction = block["transactions"][0]
        genesis_hash = expected_genesis_hash(policy=policy)

        self.assertEqual(payload["genesis_hash"], genesis_hash)
        self.assertEqual(payload["chain"]["chain_id"], "mainnet")
        self.assertEqual(payload["chain"]["genesis_hash"], genesis_hash)
        self.assertEqual(payload["chain"]["schema_version"], CHAIN_SCHEMA_VERSION)
        self.assertEqual(payload["chain"]["tip_tx_root"], block["tx_root"])
        self.assertEqual(payload["chain"]["tip_state_root"], block["state_root"])
        self.assertEqual(block["chain_id"], "mainnet")
        self.assertEqual(block["schema_version"], CHAIN_SCHEMA_VERSION)
        self.assertEqual(transaction["chain_id"], "mainnet")
        self.assertEqual(transaction["schema_version"], CHAIN_SCHEMA_VERSION)
        self.assertTrue(block["tx_root"])
        self.assertTrue(block["state_root"])

    def test_chain_merge_rejects_payload_with_wrong_genesis_hash(self) -> None:
        source_policy = WalletPolicy(wallet_data_dirname="node-a")
        target_policy = WalletPolicy(wallet_data_dirname="node-b")

        ensure_chain_genesis(policy=source_policy)
        payload = export_chain_payload(source_policy)
        payload["genesis_hash"] = "0" * 64
        payload["chain"]["genesis_hash"] = "0" * 64

        with self.assertRaisesRegex(ValueError, "genesis_hash"):
            merge_remote_chain_payload(payload, policy=target_policy)

    def test_chain_merge_logs_malformed_remote_block_payload(self) -> None:
        source_policy = WalletPolicy(wallet_data_dirname="node-a")
        target_policy = WalletPolicy(wallet_data_dirname="node-b")

        ensure_chain_genesis(policy=source_policy)
        payload = export_chain_payload(source_policy)
        payload["chain"]["blocks"].append(
            {
                "block_id": "malformed-block",
                "height": "not-an-int",
                "created_at": "2026-05-15T00:00:00+00:00",
                "previous_hash": "0" * 64,
                "transactions": [],
                "block_hash": "malformed-block-hash",
            }
        )

        with self.assertLogs("cai_compute_chain.chain", level="WARNING") as logs:
            imported_blocks, imported_transactions = merge_remote_chain_payload(
                payload,
                policy=target_policy,
            )

        self.assertEqual(imported_blocks, 1)
        self.assertEqual(imported_transactions, 3)
        output = "\n".join(logs.output)
        self.assertIn("remote chain fork validation block coercion failed", output)
        self.assertIn("remote chain block import coercion failed", output)
        self.assertIn("not-an-int", output)

    def test_chain_merge_skips_block_with_unknown_previous_hash(self) -> None:
        source_policy = WalletPolicy(wallet_data_dirname="node-a")
        target_policy = WalletPolicy(wallet_data_dirname="node-b")
        address = "abcd1234abcd1234abcd1234abcd1234"
        tx = make_chain_transaction(
            tx_type="worker_reward_credit",
            address=address,
            delta_atomic=coins_to_atomic("0.25000000"),
            settlement_id="settlement-a",
            payout_id="payout-a",
        )

        self.assertTrue(record_chain_transaction(tx, policy=source_policy))
        payload = export_chain_payload(source_policy)
        payload["chain"]["blocks"][1]["previous_hash"] = "f" * 64
        payload["chain"]["blocks"][1]["block_hash"] = ""

        imported_blocks, imported_transactions = merge_remote_chain_payload(
            payload,
            policy=target_policy,
        )

        self.assertEqual(imported_blocks, 1)
        self.assertEqual(imported_transactions, 3)
        self.assertEqual(chain_balance_atomic(address, target_policy), 0)

    def test_chain_merge_skips_block_with_invalid_tx_root(self) -> None:
        source_policy = WalletPolicy(wallet_data_dirname="node-a")
        target_policy = WalletPolicy(wallet_data_dirname="node-b")
        address = "abcd1234abcd1234abcd1234abcd1234"
        tx = make_chain_transaction(
            tx_type="worker_reward_credit",
            address=address,
            delta_atomic=coins_to_atomic("0.25000000"),
            settlement_id="settlement-a",
            payout_id="payout-a",
        )

        self.assertTrue(record_chain_transaction(tx, policy=source_policy))
        payload = export_chain_payload(source_policy)
        payload["chain"]["blocks"][1]["tx_root"] = "f" * 64

        imported_blocks, imported_transactions = merge_remote_chain_payload(
            payload,
            policy=target_policy,
        )

        self.assertEqual(imported_blocks, 1)
        self.assertEqual(imported_transactions, 3)
        self.assertEqual(chain_balance_atomic(address, target_policy), 0)

    def test_chain_merge_skips_block_with_invalid_state_root(self) -> None:
        source_policy = WalletPolicy(wallet_data_dirname="node-a")
        target_policy = WalletPolicy(wallet_data_dirname="node-b")
        address = "abcd1234abcd1234abcd1234abcd1234"
        tx = make_chain_transaction(
            tx_type="worker_reward_credit",
            address=address,
            delta_atomic=coins_to_atomic("0.25000000"),
            settlement_id="settlement-a",
            payout_id="payout-a",
        )

        self.assertTrue(record_chain_transaction(tx, policy=source_policy))
        payload = export_chain_payload(source_policy)
        payload["chain"]["blocks"][1]["state_root"] = "f" * 64

        imported_blocks, imported_transactions = merge_remote_chain_payload(
            payload,
            policy=target_policy,
        )

        self.assertEqual(imported_blocks, 1)
        self.assertEqual(imported_transactions, 3)
        self.assertEqual(chain_balance_atomic(address, target_policy), 0)

    def test_chain_merge_rejects_block_with_wrong_chain_id(self) -> None:
        source_policy = WalletPolicy(wallet_data_dirname="node-a")
        target_policy = WalletPolicy(wallet_data_dirname="node-b")
        address = "abcd1234abcd1234abcd1234abcd1234"
        tx = make_chain_transaction(
            tx_type="worker_reward_credit",
            address=address,
            delta_atomic=coins_to_atomic("0.25000000"),
            settlement_id="settlement-a",
            payout_id="payout-a",
        )

        self.assertTrue(record_chain_transaction(tx, policy=source_policy))
        payload = export_chain_payload(source_policy)
        reward_block = payload["chain"]["blocks"][1]
        reward_block["chain_id"] = "testnet"
        for item in reward_block["transactions"]:
            item["chain_id"] = "testnet"
        recompute_raw_block_hash(reward_block)

        imported_blocks, imported_transactions = merge_remote_chain_payload(
            payload,
            policy=target_policy,
        )

        self.assertEqual(imported_blocks, 1)
        self.assertEqual(imported_transactions, 3)
        self.assertEqual(chain_balance_atomic(address, target_policy), 0)

    def test_chain_merge_sorts_remote_blocks_before_import(self) -> None:
        source_policy = WalletPolicy(wallet_data_dirname="node-a")
        target_policy = WalletPolicy(wallet_data_dirname="node-b")
        address = "abcd1234abcd1234abcd1234abcd1234"
        first_tx = make_chain_transaction(
            tx_type="worker_reward_credit",
            address=address,
            delta_atomic=coins_to_atomic("0.10000000"),
            settlement_id="settlement-a",
            payout_id="payout-a",
        )
        second_tx = make_chain_transaction(
            tx_type="worker_reward_credit",
            address=address,
            delta_atomic=coins_to_atomic("0.15000000"),
            settlement_id="settlement-b",
            payout_id="payout-b",
        )

        self.assertTrue(record_chain_transaction(first_tx, policy=source_policy))
        self.assertTrue(record_chain_transaction(second_tx, policy=source_policy))
        payload = export_chain_payload(source_policy)
        payload["chain"]["blocks"] = list(reversed(payload["chain"]["blocks"]))

        imported_blocks, imported_transactions = merge_remote_chain_payload(
            payload,
            policy=target_policy,
        )

        self.assertEqual(imported_blocks, 3)
        self.assertEqual(imported_transactions, 7)
        self.assertEqual(
            chain_balance_atomic(address, target_policy),
            coins_to_atomic("0.25000000"),
        )

    def test_chain_merge_skips_block_with_replayed_nonce(self) -> None:
        source_policy = WalletPolicy(wallet_data_dirname="node-a")
        target_policy = WalletPolicy(wallet_data_dirname="node-b")
        address = "abcd1234abcd1234abcd1234abcd1234"
        first_tx = make_chain_transaction(
            tx_type="wallet_transfer_credit",
            address=address,
            delta_atomic=coins_to_atomic("0.10000000"),
            wallet_id="wallet-a",
            nonce="nonce-1",
        )
        second_tx = make_chain_transaction(
            tx_type="wallet_transfer_credit",
            address=address,
            delta_atomic=coins_to_atomic("0.20000000"),
            wallet_id="wallet-a",
            nonce="nonce-2",
        )

        self.assertTrue(record_chain_transaction(first_tx, policy=source_policy))
        self.assertTrue(record_chain_transaction(second_tx, policy=source_policy))
        payload = export_chain_payload(source_policy)
        replayed_block = payload["chain"]["blocks"][2]
        replayed_block["transactions"][1]["nonce"] = "nonce-1"
        replayed_block["tx_root"] = ""
        recompute_raw_block_hash(replayed_block)

        imported_blocks, imported_transactions = merge_remote_chain_payload(
            payload,
            policy=target_policy,
        )

        self.assertEqual(imported_blocks, 2)
        self.assertEqual(imported_transactions, 5)
        self.assertEqual(
            chain_balance_atomic(address, target_policy),
            coins_to_atomic("0.10000000"),
        )

    def test_chain_merge_rejects_competing_genesis_block(self) -> None:
        source_policy = WalletPolicy(wallet_data_dirname="node-a")
        target_policy = WalletPolicy(wallet_data_dirname="node-b")

        ensure_chain_genesis(policy=source_policy)
        ensure_chain_genesis(policy=target_policy)
        payload = export_chain_payload(source_policy)
        payload["chain"]["blocks"][0]["block_id"] = "competing-genesis"
        recompute_raw_block_hash(payload["chain"]["blocks"][0])

        with self.assertRaisesRegex(ValueError, "genesis_hash"):
            merge_remote_chain_payload(
                payload,
                policy=target_policy,
            )

    def test_chain_merge_rejects_block_with_duplicate_transaction_ids(self) -> None:
        source_policy = WalletPolicy(wallet_data_dirname="node-a")
        target_policy = WalletPolicy(wallet_data_dirname="node-b")
        address = "abcd1234abcd1234abcd1234abcd1234"
        tx = make_chain_transaction(
            tx_type="worker_reward_credit",
            address=address,
            delta_atomic=coins_to_atomic("0.25000000"),
            settlement_id="settlement-a",
            payout_id="payout-a",
        )

        self.assertTrue(record_chain_transaction(tx, policy=source_policy))
        payload = export_chain_payload(source_policy)
        reward_block = payload["chain"]["blocks"][1]
        reward_block["transactions"].append(dict(reward_block["transactions"][0]))
        recompute_raw_block_hash(reward_block)

        imported_blocks, imported_transactions = merge_remote_chain_payload(
            payload,
            policy=target_policy,
        )

        self.assertEqual(imported_blocks, 1)
        self.assertEqual(imported_transactions, 3)
        self.assertEqual(chain_balance_atomic(address, target_policy), 0)

    def test_chain_merge_rejects_competing_block_height_until_fork_choice_exists(self) -> None:
        source_policy = WalletPolicy(wallet_data_dirname="node-a")
        target_policy = WalletPolicy(wallet_data_dirname="node-b")
        address_a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        address_b = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        source_tx = make_chain_transaction(
            tx_type="worker_reward_credit",
            address=address_a,
            delta_atomic=coins_to_atomic("0.10000000"),
            settlement_id="settlement-a",
            payout_id="payout-a",
        )
        target_tx = make_chain_transaction(
            tx_type="worker_reward_credit",
            address=address_b,
            delta_atomic=coins_to_atomic("0.20000000"),
            settlement_id="settlement-b",
            payout_id="payout-b",
        )

        self.assertTrue(record_chain_transaction(source_tx, policy=source_policy))
        self.assertTrue(record_chain_transaction(target_tx, policy=target_policy))
        payload = export_chain_payload(source_policy)

        imported_blocks, imported_transactions = merge_remote_chain_payload(
            payload,
            policy=target_policy,
        )

        self.assertEqual(imported_blocks, 0)
        self.assertEqual(imported_transactions, 0)
        self.assertEqual(chain_balance_atomic(address_a, target_policy), 0)
        self.assertEqual(
            chain_balance_atomic(address_b, target_policy),
            coins_to_atomic("0.20000000"),
        )

    def test_chain_summary_reports_local_validation_errors(self) -> None:
        policy = WalletPolicy(wallet_data_dirname="node-a")
        address = "abcd1234abcd1234abcd1234abcd1234"
        tx = make_chain_transaction(
            tx_type="worker_reward_credit",
            address=address,
            delta_atomic=coins_to_atomic("0.25000000"),
            settlement_id="settlement-a",
            payout_id="payout-a",
        )

        self.assertTrue(record_chain_transaction(tx, policy=policy))
        blocks = list_chain_blocks(policy)
        blocks[1].previous_hash = "f" * 64
        save_chain_blocks(blocks, policy)

        summary = chain_summary(policy)

        self.assertFalse(summary["valid"])
        self.assertTrue(
            any("previous_hash" in error for error in summary["validationErrors"])
        )

    def test_chain_summary_reports_negative_balances(self) -> None:
        policy = WalletPolicy(wallet_data_dirname="node-a")
        address = "abcd1234abcd1234abcd1234abcd1234"
        tx = make_chain_transaction(
            tx_type="settlement_wallet_debit",
            address=address,
            delta_atomic=-coins_to_atomic("1.00000000"),
            settlement_id="settlement-negative",
            wallet_id="wallet-negative",
        )

        self.assertTrue(record_chain_transaction(tx, policy=policy))
        summary = chain_summary(policy)

        self.assertFalse(summary["valid"])
        self.assertTrue(
            any("negative balance" in error for error in summary["validationErrors"])
        )

    def test_chain_summary_reports_schema_version_mismatch(self) -> None:
        policy = WalletPolicy(wallet_data_dirname="node-a")
        address = "abcd1234abcd1234abcd1234abcd1234"
        tx = make_chain_transaction(
            tx_type="worker_reward_credit",
            address=address,
            delta_atomic=coins_to_atomic("0.25000000"),
            settlement_id="settlement-a",
            payout_id="payout-a",
        )

        self.assertTrue(record_chain_transaction(tx, policy=policy))
        blocks = list_chain_blocks(policy)
        blocks[1].schema_version = 999
        save_chain_blocks(blocks, policy)

        summary = chain_summary(policy)

        self.assertFalse(summary["valid"])
        self.assertTrue(
            any("schema_version" in error for error in summary["validationErrors"])
        )

    def test_chain_summary_reports_state_root_mismatch(self) -> None:
        policy = WalletPolicy(wallet_data_dirname="node-a")
        address = "abcd1234abcd1234abcd1234abcd1234"
        tx = make_chain_transaction(
            tx_type="worker_reward_credit",
            address=address,
            delta_atomic=coins_to_atomic("0.25000000"),
            settlement_id="settlement-a",
            payout_id="payout-a",
        )

        self.assertTrue(record_chain_transaction(tx, policy=policy))
        blocks = list_chain_blocks(policy)
        blocks[1].state_root = "f" * 64
        save_chain_blocks(blocks, policy)

        summary = chain_summary(policy)

        self.assertFalse(summary["valid"])
        self.assertTrue(
            any("state_root" in error for error in summary["validationErrors"])
        )

    def test_chain_balance_index_counts_each_transaction_once(self) -> None:
        policy = WalletPolicy(wallet_data_dirname="node-a")
        address = "abcd1234abcd1234abcd1234abcd1234"
        tx = make_chain_transaction(
            tx_type="worker_reward_credit",
            address=address,
            delta_atomic=coins_to_atomic("0.25000000"),
            settlement_id="settlement-a",
            payout_id="payout-a",
        )

        self.assertTrue(record_chain_transaction(tx, policy=policy))
        blocks = list_chain_blocks(policy)
        blocks[-1].transactions.append(blocks[-1].transactions[0])

        index = chain_balance_index(policy, blocks=blocks)

        self.assertEqual(index[address], coins_to_atomic("0.25000000"))

    def test_chain_summary_exposes_balance_index_counts_and_ai_fund(self) -> None:
        policy = WalletPolicy(wallet_data_dirname="node-a")
        money_policy = MoneyPolicy()

        ensure_chain_genesis(policy=policy, money_policy=money_policy)
        summary = chain_summary(policy)

        self.assertGreaterEqual(summary["balanceAddressCount"], 2)
        self.assertGreaterEqual(summary["nonZeroBalanceAddressCount"], 2)
        self.assertEqual(summary["expectedTotalSupplyCoins"], "1000000000.00000000")
        self.assertEqual(summary["totalBalanceCoins"], "1000000000.00000000")
        self.assertEqual(summary["supplyDeltaCoins"], "0.00000000")
        self.assertTrue(summary["supplyMatchesPolicy"])
        self.assertEqual(summary["tipHeight"], 0)
        self.assertEqual(summary["finalizedHeight"], 0)
        self.assertEqual(summary["lastSyncAt"], "2026-01-01T00:00:00+00:00")
        self.assertEqual(summary["aiDevelopmentAddress"], money_policy.ai_development_address)
        self.assertEqual(summary["aiDevelopmentBalanceCoins"], "0.00000000")

    def test_invalid_chain_snapshot_interval_is_logged(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {"CAI_CHAIN_SNAPSHOT_INTERVAL": "invalid-interval"},
                clear=False,
            ),
            self.assertLogs("cai_compute_chain.chain", level="WARNING") as logs,
        ):
            interval = _chain_snapshot_interval()

        self.assertEqual(interval, DEFAULT_CHAIN_SNAPSHOT_INTERVAL)
        output = "\n".join(logs.output)
        self.assertIn("chain snapshot interval parse failed", output)
        self.assertIn("invalid-interval", output)

    def test_chain_writes_persistent_index_and_snapshots(self) -> None:
        policy = WalletPolicy(wallet_data_dirname="node-a")
        address = "abcd1234abcd1234abcd1234abcd1234"

        with patch.dict("os.environ", {"CAI_CHAIN_SNAPSHOT_INTERVAL": "1"}, clear=False):
            self.assertTrue(
                record_chain_transaction(
                    make_chain_transaction(
                        tx_type="worker_reward_credit",
                        address=address,
                        delta_atomic=coins_to_atomic("0.25000000"),
                        settlement_id="settlement-a",
                        payout_id="payout-a",
                        nonce="reward-a",
                    ),
                    policy=policy,
                )
            )
            self.assertTrue(
                record_chain_transaction(
                    make_chain_transaction(
                        tx_type="worker_reward_credit",
                        address=address,
                        delta_atomic=coins_to_atomic("0.75000000"),
                        settlement_id="settlement-b",
                        payout_id="payout-b",
                        nonce="reward-b",
                    ),
                    policy=policy,
                )
            )

        index = load_chain_index(policy)
        snapshots = json.loads(chain_snapshots_file_path(policy).read_text(encoding="utf-8"))

        self.assertTrue(chain_index_file_path(policy).exists())
        self.assertTrue(chain_snapshots_file_path(policy).exists())
        self.assertEqual(index["blockCount"], 3)
        self.assertEqual(index["transactionCount"], 7)
        self.assertEqual(index["balancesAtomic"][address], coins_to_atomic("1.00000000"))
        self.assertEqual(index["latestSnapshotHeight"], 2)
        self.assertEqual(snapshots["latest"]["height"], 2)
        self.assertEqual(len(snapshots["snapshots"]), 3)
        self.assertEqual(chain_summary(policy)["latestSnapshotHeight"], 2)

    def test_wallet_balance_uses_valid_persistent_index_without_full_chain_scan(self) -> None:
        policy = WalletPolicy(wallet_data_dirname="node-a")
        money_policy = MoneyPolicy()
        address = "abcd1234abcd1234abcd1234abcd1234"
        wallet = WalletRecord(
            wallet_id="wallet-a",
            name="Node A",
            address=address,
            created_at="2026-01-01T00:00:00+00:00",
            password_salt_b64="",
            password_hash_b64="",
            spendable_balance_atomic=coins_to_atomic("12.00000000", money_policy),
        )
        self.assertTrue(
            record_chain_transaction(
                make_chain_transaction(
                    tx_type="worker_reward_credit",
                    address=address,
                    delta_atomic=coins_to_atomic("0.25000000"),
                    settlement_id="settlement-a",
                    payout_id="payout-a",
                    nonce="reward-a",
                ),
                policy=policy,
            )
        )

        with patch(
            "cai_compute_chain.chain.list_chain_blocks",
            side_effect=AssertionError("unexpected full chain scan"),
        ):
            self.assertEqual(
                chain_balance_atomic(address, policy),
                coins_to_atomic("0.25000000"),
            )
            self.assertEqual(
                chain_balance_index(policy)[address],
                coins_to_atomic("0.25000000"),
            )
            self.assertEqual(
                wallet_chain_balance_or_local_atomic(wallet, policy),
                coins_to_atomic("0.25000000"),
            )
            self.assertEqual(wallet_balance_source(policy), "chain")

    def test_wallet_balance_uses_chain_zero_when_chain_is_initialized(self) -> None:
        policy = WalletPolicy(wallet_data_dirname="node-a")
        money_policy = MoneyPolicy()
        ensure_chain_genesis(policy=policy, money_policy=money_policy)
        wallet = WalletRecord(
            wallet_id="wallet-a",
            name="Node A",
            address="abcd1234abcd1234abcd1234abcd1234",
            created_at="2026-01-01T00:00:00+00:00",
            password_salt_b64="",
            password_hash_b64="",
            spendable_balance_atomic=coins_to_atomic("12.00000000", money_policy),
        )

        self.assertEqual(wallet_chain_balance_or_local_atomic(wallet, policy), 0)
        self.assertEqual(wallet_balance_source(policy), "chain")

    def test_wallet_balance_source_is_local_before_chain_init(self) -> None:
        policy = WalletPolicy(wallet_data_dirname="node-a")

        self.assertEqual(wallet_balance_source(policy), "local")

    def test_signed_wallet_transfer_debit_validates_signature(self) -> None:
        policy = WalletPolicy(
            wallet_data_dirname="legacy-ed25519-chain",
            require_post_quantum_wallet_signatures=False,
        )
        signing_seed = generate_signing_seed()
        public_key_b64 = public_key_b64_from_seed(signing_seed)
        address = address_from_public_key_b64(public_key_b64)
        self.assertTrue(
            record_chain_transaction(
                make_chain_transaction(
                    tx_type="worker_reward_credit",
                    address=address,
                    delta_atomic=coins_to_atomic("2.00000000"),
                    wallet_id="wallet-signed",
                    payout_id="payout-signed",
                    nonce="signed-credit",
                ),
                policy=policy,
            )
        )
        debit = make_chain_transaction(
            tx_type="wallet_transfer_debit",
            address=address,
            delta_atomic=-coins_to_atomic("1.00010000"),
            wallet_id="wallet-signed",
            counterparty_address="abcd" * 8,
            nonce="signed-debit",
            public_key_b64=public_key_b64,
            metadata={
                "signature_required": True,
                "address_scheme": ADDRESS_SCHEME_ED25519,
            },
        )
        debit.signature_b64 = sign_payload_b64(
            signing_seed,
            transaction_signing_payload(debit),
        )

        self.assertTrue(record_chain_transaction(debit, policy=policy))
        self.assertEqual(validate_chain_blocks(list_chain_blocks(policy), policy=policy), [])

    def test_signed_wallet_transfer_debit_rejects_invalid_signature(self) -> None:
        policy = WalletPolicy(
            wallet_data_dirname="legacy-ed25519-invalid-chain",
            require_post_quantum_wallet_signatures=False,
        )
        signing_seed = generate_signing_seed()
        public_key_b64 = public_key_b64_from_seed(signing_seed)
        address = address_from_public_key_b64(public_key_b64)
        debit = make_chain_transaction(
            tx_type="wallet_transfer_debit",
            address=address,
            delta_atomic=-coins_to_atomic("1.00010000"),
            wallet_id="wallet-signed",
            counterparty_address="abcd" * 8,
            nonce="signed-invalid-debit",
            public_key_b64=public_key_b64,
            metadata={
                "signature_required": True,
                "address_scheme": ADDRESS_SCHEME_ED25519,
            },
        )
        debit.signature_b64 = sign_payload_b64(
            signing_seed,
            transaction_signing_payload(debit),
        )
        debit.delta_atomic = -coins_to_atomic("1.50000000")

        with self.assertRaisesRegex(ValueError, "invalid signature"):
            record_chain_transaction(debit, policy=policy)

    def test_wallet_transfer_debit_rejects_missing_signature_even_without_flag(self) -> None:
        policy = WalletPolicy(wallet_data_dirname="unsigned-debit-chain")
        debit = make_chain_transaction(
            tx_type="wallet_transfer_debit",
            address="abcd" * 8,
            delta_atomic=-coins_to_atomic("1.00010000"),
            wallet_id="wallet-unsigned",
            counterparty_address="1234" * 8,
            nonce="unsigned-debit",
        )

        with self.assertRaisesRegex(ValueError, "missing public_key_b64"):
            record_chain_transaction(debit, policy=policy)

    def test_mainnet_policy_rejects_legacy_ed25519_wallet_debit(self) -> None:
        policy = WalletPolicy(
            wallet_data_dirname="strict-ed25519-chain",
            require_post_quantum_wallet_signatures=True,
        )
        signing_seed = generate_signing_seed()
        public_key_b64 = public_key_b64_from_seed(signing_seed)
        address = address_from_public_key_b64(public_key_b64)
        debit = make_chain_transaction(
            tx_type="wallet_transfer_debit",
            address=address,
            delta_atomic=-coins_to_atomic("1.00010000"),
            wallet_id="wallet-ed25519",
            counterparty_address="1234" * 8,
            nonce="strict-ed25519-debit",
            public_key_b64=public_key_b64,
            metadata={
                "signature_required": True,
                "address_scheme": ADDRESS_SCHEME_ED25519,
            },
        )
        debit.signature_b64 = sign_payload_b64(
            signing_seed,
            transaction_signing_payload(debit),
        )

        with self.assertRaisesRegex(ValueError, "post-quantum signature"):
            record_chain_transaction(debit, policy=policy)

    def test_strict_policy_accepts_hybrid_wallet_debit(self) -> None:
        policy = WalletPolicy(
            wallet_data_dirname="strict-hybrid-chain",
            require_post_quantum_wallet_signatures=True,
        )
        signing_seed = generate_signing_seed()
        public_key_b64 = public_key_b64_from_seed(signing_seed)
        pq_public_key_b64, pq_private_key_b64 = generate_mldsa65_keypair_b64()
        address = hybrid_address_from_public_keys_b64(
            ed25519_public_key_b64=public_key_b64,
            pq_public_key_b64=pq_public_key_b64,
            address_scheme=DEFAULT_WALLET_HYBRID_ADDRESS_SCHEME,
        )
        self.assertTrue(
            record_chain_transaction(
                make_chain_transaction(
                    tx_type="worker_reward_credit",
                    address=address,
                    delta_atomic=coins_to_atomic("2.00000000"),
                    wallet_id="wallet-hybrid",
                    payout_id="payout-hybrid",
                    nonce="hybrid-credit",
                ),
                policy=policy,
            )
        )
        debit = make_chain_transaction(
            tx_type="wallet_transfer_debit",
            address=address,
            delta_atomic=-coins_to_atomic("1.00010000"),
            wallet_id="wallet-hybrid",
            counterparty_address="1234" * 8,
            nonce="strict-hybrid-debit",
            public_key_b64=public_key_b64,
            metadata={
                "signature_required": True,
                "address_scheme": DEFAULT_WALLET_HYBRID_ADDRESS_SCHEME,
                "pq_signature_scheme": SIGNING_SCHEME_ML_DSA_65,
                "pq_public_key_b64": pq_public_key_b64,
            },
        )
        debit.signature_b64 = sign_payload_b64(
            signing_seed,
            transaction_signing_payload(debit),
        )
        debit.metadata["pq_signature_b64"] = sign_payload_mldsa65_b64(
            pq_private_key_b64,
            transaction_signing_payload(debit),
        )

        self.assertTrue(record_chain_transaction(debit, policy=policy))
        self.assertEqual(validate_chain_blocks(list_chain_blocks(policy), policy=policy), [])


if __name__ == "__main__":
    unittest.main()

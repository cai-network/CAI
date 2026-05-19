# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
import tempfile
import unittest
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.node_config import (
    assess_validator_network_status,
    clear_validator_jail,
    complete_validator_unbond,
    default_worker_allowed_model_ids,
    get_validator_attestation_status,
    get_validator_mode_status,
    bind_worker_reward_address,
    get_validator_identity,
    jail_validator,
    load_or_create_node_config,
    node_config_file_path,
    resolve_worker_reward_address,
    set_relay_mode,
    refresh_validator_ha_lease,
    set_validator_ha_mode,
    set_validator_static_ip_confirmation,
    set_validator_mode,
    set_worker_mode,
)
from cai_compute_chain.chain import (
    append_chain_block,
    chain_balance_atomic,
    chain_summary,
    compute_reserve_chain_address,
    ensure_chain_genesis,
    make_chain_transaction,
    validator_bond_pool_chain_address,
    validator_slash_pool_chain_address,
)
from cai_compute_chain.model import MoneyPolicy
from cai_compute_chain.wallet import (
    create_wallet,
    credit_wallet,
    unlock_wallet,
    find_wallet_by_id,
    load_or_create_ledger,
)
from cai_compute_chain.validators import list_validator_records, sync_validator_record


class NodeConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo_patch = patch(
            "cai_compute_chain.wallet.repo_root",
            return_value=Path(self.tempdir.name),
        )
        self.repo_patch.start()
        self._chain_credit_counter = 0

    def tearDown(self) -> None:
        self.repo_patch.stop()
        self.tempdir.cleanup()

    def _credit_wallet_from_reserve(self, wallet, amount_atomic: int):
        money_policy = MoneyPolicy()
        ensure_chain_genesis(money_policy=money_policy)
        self._chain_credit_counter += 1
        credit_id = f"test-validator-credit-{wallet.wallet_id}-{self._chain_credit_counter}"
        append_chain_block(
            [
                make_chain_transaction(
                    tx_type="test_reserve_debit",
                    address=compute_reserve_chain_address(money_policy),
                    delta_atomic=-amount_atomic,
                    wallet_id=f"system-compute-reserve-{money_policy.chain_network.value}",
                    counterparty_address=wallet.address,
                    nonce=f"{credit_id}:reserve-debit",
                ),
                make_chain_transaction(
                    tx_type="test_wallet_credit",
                    address=wallet.address,
                    delta_atomic=amount_atomic,
                    wallet_id=wallet.wallet_id,
                    counterparty_address=compute_reserve_chain_address(money_policy),
                    nonce=f"{credit_id}:wallet-credit",
                ),
            ],
            validator_id="test-validator",
        )
        return wallet

    def test_default_node_config_uses_network_model(self) -> None:
        config = load_or_create_node_config()
        self.assertFalse(config.validator_enabled)
        self.assertEqual(config.validator_state, "unbonded")
        self.assertIsNone(config.validator_wallet_id)
        self.assertIsNone(config.validator_address)
        self.assertEqual(config.validator_bond_atomic, 0)
        self.assertFalse(config.validator_static_ip_confirmed)
        self.assertIsNone(config.validator_unbonding_started_at)
        self.assertIsNone(config.validator_unbonding_available_at)
        self.assertFalse(config.worker_enabled)
        self.assertTrue(config.relay_enabled)
        self.assertFalse(config.relay_mode_manually_configured)
        self.assertEqual(
            config.worker_allowed_model_ids,
            default_worker_allowed_model_ids(),
        )
        self.assertIn("cai-network/Qwen3-0.6B-GGUF", config.worker_allowed_model_ids)
        self.assertIn(
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            config.worker_allowed_model_ids,
        )

    def test_node_config_preserves_public_and_private_model_ids(self) -> None:
        path = node_config_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "validator_enabled": False,
                    "validator_state": "unbonded",
                    "validator_wallet_id": None,
                    "validator_address": None,
                    "validator_bond_atomic": 0,
                    "validator_static_ip_confirmed": False,
                    "validator_unbonding_started_at": None,
                    "validator_unbonding_available_at": None,
                    "validator_jailed_at": None,
                    "validator_unjail_available_at": None,
                    "validator_jail_reason": None,
                    "validator_last_slash_atomic": 0,
                    "validator_total_slashed_atomic": 0,
                    "worker_enabled": True,
                    "relay_enabled": True,
                    "worker_allowed_model_ids": [
                        "Qwen/Qwen3-0.6B-GGUF",
                        "cai-network/Qwen3-0.6B-GGUF",
                    ],
                    "worker_max_parallel_jobs": 1,
                    "worker_max_memory_mb": None,
                    "worker_reward_address_by_node_id": {},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        config = load_or_create_node_config()

        self.assertEqual(
            config.worker_allowed_model_ids,
            ["Qwen/Qwen3-0.6B-GGUF", "cai-network/Qwen3-0.6B-GGUF"],
        )

    def test_relay_mode_toggles_independently_of_worker_and_validator(self) -> None:
        config = set_relay_mode(False)
        self.assertFalse(config.relay_enabled)
        self.assertTrue(config.relay_mode_manually_configured)

        config = set_relay_mode(True)
        self.assertTrue(config.relay_enabled)
        self.assertTrue(config.relay_mode_manually_configured)
        self.assertFalse(config.worker_enabled)
        self.assertFalse(config.validator_enabled)

        config = set_worker_mode(enabled=True, allowed_model_ids=["custom/model-a"])
        self.assertTrue(config.worker_enabled)
        self.assertTrue(config.relay_enabled)

        config = set_relay_mode(False)
        self.assertFalse(config.relay_enabled)
        self.assertTrue(config.worker_enabled)

    def test_legacy_unconfigured_relay_mode_migrates_to_default_enabled(self) -> None:
        path = node_config_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "validator_enabled": False,
                    "validator_state": "unbonded",
                    "validator_wallet_id": None,
                    "validator_address": None,
                    "validator_bond_atomic": 0,
                    "validator_static_ip_confirmed": False,
                    "validator_unbonding_started_at": None,
                    "validator_unbonding_available_at": None,
                    "validator_jailed_at": None,
                    "validator_unjail_available_at": None,
                    "validator_jail_reason": None,
                    "validator_last_slash_atomic": 0,
                    "validator_total_slashed_atomic": 0,
                    "worker_enabled": False,
                    "relay_enabled": False,
                    "worker_allowed_model_ids": ["cai-network/Qwen3-0.6B-GGUF"],
                    "worker_max_parallel_jobs": 1,
                    "worker_max_memory_mb": None,
                    "worker_reward_address_by_node_id": {},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        config = load_or_create_node_config()

        self.assertTrue(config.relay_enabled)
        self.assertFalse(config.relay_mode_manually_configured)

    def test_toggle_modes_and_resource_hints(self) -> None:
        wallet = create_wallet("validator", "pass", select=True)
        self._credit_wallet_from_reserve(wallet, 2_000_000_000_000)
        unlock_wallet("pass", selector=wallet.wallet_id)
        set_validator_static_ip_confirmation(True)

        validator = set_validator_mode(
            True,
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )
        self.assertTrue(validator.validator_enabled)
        self.assertEqual(validator.validator_state, "bonded")
        self.assertEqual(validator.validator_wallet_id, wallet.wallet_id)
        self.assertEqual(validator.validator_address, wallet.address)
        self.assertEqual(validator.validator_bond_atomic, 1_000_000_000_000)
        self.assertEqual(get_validator_identity(), wallet.address)
        bonded_wallet = find_wallet_by_id(wallet.wallet_id)
        self.assertIsNotNone(bonded_wallet)
        assert bonded_wallet is not None
        self.assertEqual(bonded_wallet.spendable_balance_atomic, 1_000_000_000_000)
        self.assertEqual(bonded_wallet.validator_reserved_atomic, 1_000_000_000_000)

        with self.assertRaisesRegex(
            ValueError, "Disable validator mode and finish unbonding before enabling worker mode."
        ):
            set_worker_mode(
                enabled=True,
                allowed_model_ids=["custom/model-a"],
                max_parallel_jobs=2,
                max_memory_mb=2048,
            )

        disabled = set_validator_mode(False)
        self.assertFalse(disabled.validator_enabled)
        self.assertEqual(disabled.validator_state, "unbonding")
        self.assertEqual(disabled.validator_wallet_id, wallet.wallet_id)
        self.assertEqual(disabled.validator_address, wallet.address)
        self.assertEqual(disabled.validator_bond_atomic, 1_000_000_000_000)
        self.assertIsNotNone(disabled.validator_unbonding_started_at)
        self.assertIsNotNone(disabled.validator_unbonding_available_at)
        self.assertIsNone(get_validator_identity())
        released_wallet = find_wallet_by_id(wallet.wallet_id)
        self.assertIsNotNone(released_wallet)
        assert released_wallet is not None
        self.assertEqual(released_wallet.spendable_balance_atomic, 1_000_000_000_000)
        self.assertEqual(released_wallet.validator_reserved_atomic, 1_000_000_000_000)

        worker = set_worker_mode(
            enabled=False,
        )
        self.assertFalse(worker.worker_enabled)

        with self.assertRaisesRegex(
            ValueError, "Disable validator mode and finish unbonding before enabling worker mode."
        ):
            set_worker_mode(
                enabled=True,
                allowed_model_ids=["custom/model-a"],
                max_parallel_jobs=2,
                max_memory_mb=2048,
            )

        available_at = datetime.fromisoformat(disabled.validator_unbonding_available_at)
        completed = complete_validator_unbond(now=available_at + timedelta(seconds=1))
        self.assertFalse(completed.validator_enabled)
        self.assertEqual(completed.validator_state, "unbonded")
        self.assertIsNone(completed.validator_wallet_id)
        self.assertIsNone(completed.validator_address)
        self.assertEqual(completed.validator_bond_atomic, 0)
        self.assertIsNone(completed.validator_unbonding_started_at)
        self.assertIsNone(completed.validator_unbonding_available_at)

        released_wallet = find_wallet_by_id(wallet.wallet_id)
        self.assertIsNotNone(released_wallet)
        assert released_wallet is not None
        self.assertEqual(released_wallet.spendable_balance_atomic, 2_000_000_000_000)
        self.assertEqual(released_wallet.validator_reserved_atomic, 0)

        worker = set_worker_mode(
            enabled=True,
            allowed_model_ids=["custom/model-a"],
            max_parallel_jobs=2,
            max_memory_mb=2048,
        )
        self.assertTrue(worker.worker_enabled)
        self.assertIn("custom/model-a", worker.worker_allowed_model_ids)
        self.assertEqual(worker.worker_max_parallel_jobs, 2)
        self.assertEqual(worker.worker_max_memory_mb, 2048)

    def test_validator_mode_requires_unlocked_wallet(self) -> None:
        create_wallet("validator", "pass", select=True)
        set_validator_static_ip_confirmation(True)

        with self.assertRaises(ValueError):
            set_validator_mode(
                True,
                state_payload=self._public_validator_state(),
                cai_url="http://127.0.0.1:52425",
            )

    def test_bonded_validator_status_reuses_reserved_bond(self) -> None:
        wallet = create_wallet("validator", "pass", select=True)
        self._credit_wallet_from_reserve(wallet, 1_000_000_000_000)
        unlock_wallet("pass", selector=wallet.wallet_id)
        set_validator_static_ip_confirmation(True)
        set_validator_mode(
            True,
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )

        status = get_validator_mode_status(
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )
        self.assertTrue(status.can_enable)
        self.assertEqual(status.current_spendable_atomic, 0)
        self.assertEqual(status.bonded_atomic, 1_000_000_000_000)
        self.assertIn("already bonded", status.reason)

        config = set_validator_mode(
            True,
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )
        self.assertTrue(config.validator_enabled)

    def test_validator_ha_passive_replica_keeps_one_identity_and_does_not_attest(
        self,
    ) -> None:
        wallet = create_wallet("validator", "pass", select=True)
        self._credit_wallet_from_reserve(wallet, 1_000_000_000_000)
        unlock_wallet("pass", selector=wallet.wallet_id)
        set_validator_static_ip_confirmation(True)
        set_validator_mode(
            True,
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )

        config = set_validator_ha_mode(
            enabled=True,
            role="passive",
            replica_id="replica-passive",
            auto_failover=False,
        )
        status = get_validator_attestation_status(
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )
        records = list_validator_records()

        self.assertTrue(config.validator_ha_enabled)
        self.assertEqual(config.validator_ha_role, "passive")
        self.assertEqual(config.validator_ha_replica_id, "replica-passive")
        self.assertFalse(status.can_attest)
        self.assertTrue(status.passive_replica)
        self.assertEqual(status.validator_id, wallet.address)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].validator_id, wallet.address)
        self.assertTrue(records[0].ha_enabled)
        self.assertEqual(records[0].replica_node_ids, ["node-public"])
        self.assertEqual(records[0].current_node_id, "node-public")

        active = set_validator_ha_mode(
            enabled=True,
            role="active",
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )
        active_status = get_validator_attestation_status(
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )
        active_records = list_validator_records()

        self.assertEqual(active.validator_ha_role, "active")
        self.assertTrue(active_status.can_attest)
        self.assertEqual(active_records[0].active_replica_node_id, "node-public")

        standalone = set_validator_ha_mode(
            enabled=False,
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )
        standalone_records = list_validator_records()

        self.assertFalse(standalone.validator_ha_enabled)
        self.assertEqual(standalone.validator_ha_role, "standalone")
        self.assertIsNone(standalone.validator_ha_replica_id)
        self.assertFalse(standalone_records[0].ha_enabled)
        self.assertIsNone(standalone_records[0].active_replica_node_id)
        self.assertEqual(standalone_records[0].replica_node_ids, [])

    def test_validator_ha_passive_replica_auto_promotes_after_expired_lease(
        self,
    ) -> None:
        wallet = create_wallet("validator", "pass", select=True)
        self._credit_wallet_from_reserve(wallet, 1_000_000_000_000)
        unlock_wallet("pass", selector=wallet.wallet_id)
        set_validator_static_ip_confirmation(True)
        standby_state = self._public_validator_state(
            node_id="node-standby",
            ip_address="85.137.164.251",
        )
        set_validator_mode(
            True,
            state_payload=standby_state,
            cai_url="http://127.0.0.1:52425",
        )
        set_validator_ha_mode(
            enabled=True,
            role="passive",
            replica_id="replica-standby",
            state_payload=standby_state,
            cai_url="http://127.0.0.1:52425",
        )

        future_lease = (datetime.now(tz=UTC) + timedelta(minutes=5)).isoformat()
        sync_validator_record(
            validator_id=wallet.address,
            wallet_id=wallet.wallet_id,
            address=wallet.address,
            state="bonded",
            bonded_atomic=1_000_000_000_000,
            static_ip_confirmed=True,
            current_node_id="node-active",
            advertised_api_host="85.137.164.250",
            advertised_data_host="85.137.164.250",
            ha_enabled=True,
            ha_role="active",
            active_replica_node_id="node-active",
            active_replica_lease_until=future_lease,
            replica_node_ids=["node-active", "node-standby"],
        )

        leased_status = get_validator_attestation_status(
            state_payload=standby_state,
            cai_url="http://127.0.0.1:52425",
        )
        self.assertFalse(leased_status.can_attest)
        self.assertTrue(leased_status.passive_replica)

        expired_lease = (datetime.now(tz=UTC) - timedelta(minutes=5)).isoformat()
        sync_validator_record(
            validator_id=wallet.address,
            wallet_id=wallet.wallet_id,
            address=wallet.address,
            state="bonded",
            bonded_atomic=1_000_000_000_000,
            static_ip_confirmed=True,
            current_node_id="node-active",
            advertised_api_host="85.137.164.250",
            advertised_data_host="85.137.164.250",
            ha_enabled=True,
            ha_role="active",
            active_replica_node_id="node-active",
            active_replica_lease_until=expired_lease,
            replica_node_ids=["node-active", "node-standby"],
        )

        promoted_status = get_validator_attestation_status(
            state_payload=standby_state,
            cai_url="http://127.0.0.1:52425",
        )
        promoted_config = refresh_validator_ha_lease(
            state_payload=standby_state,
            cai_url="http://127.0.0.1:52425",
        )
        promoted_record = list_validator_records()[0]

        self.assertTrue(promoted_status.can_attest)
        self.assertEqual(promoted_config.validator_ha_role, "active")
        self.assertEqual(promoted_record.active_replica_node_id, "node-standby")
        self.assertIn("node-active", promoted_record.replica_node_ids)
        self.assertIn("node-standby", promoted_record.replica_node_ids)
        self.assertIsNotNone(promoted_record.active_replica_lease_until)

    def test_validator_mode_requires_minimum_bond(self) -> None:
        wallet = create_wallet("validator", "pass", select=True)
        self._credit_wallet_from_reserve(wallet, 50_000_000)
        unlock_wallet("pass", selector=wallet.wallet_id)
        set_validator_static_ip_confirmation(True)

        status = get_validator_mode_status(
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )
        self.assertFalse(status.can_enable)
        self.assertIn("Minimum self-bond is 10000.00000000", status.reason)

        with self.assertRaises(ValueError):
            set_validator_mode(
                True,
                state_payload=self._public_validator_state(),
                cai_url="http://127.0.0.1:52425",
            )

    def test_validator_mode_ignores_local_only_wallet_balance(self) -> None:
        wallet = create_wallet("validator", "pass", select=True)
        credit_wallet(wallet.wallet_id, 2_000_000_000_000)
        unlock_wallet("pass", selector=wallet.wallet_id)
        set_validator_static_ip_confirmation(True)

        status = get_validator_mode_status(
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )
        self.assertFalse(status.can_enable)
        self.assertEqual(status.current_spendable_atomic, 0)
        self.assertEqual(chain_balance_atomic(wallet.address), 0)
        self.assertIn("Minimum self-bond is 10000.00000000", status.reason)

        with self.assertRaisesRegex(ValueError, "Minimum self-bond"):
            set_validator_mode(
                True,
                state_payload=self._public_validator_state(),
                cai_url="http://127.0.0.1:52425",
            )

    def test_validator_mode_blocks_reenable_during_unbonding(self) -> None:
        wallet = create_wallet("validator", "pass", select=True)
        self._credit_wallet_from_reserve(wallet, 2_000_000_000_000)
        unlock_wallet("pass", selector=wallet.wallet_id)
        set_validator_static_ip_confirmation(True)
        set_validator_mode(
            True,
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )
        set_validator_mode(False)

        status = get_validator_mode_status(
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )
        self.assertFalse(status.can_enable)
        self.assertEqual(status.state, "unbonding")
        self.assertIn("currently unbonding", status.reason)

        with self.assertRaisesRegex(ValueError, "currently unbonding"):
            set_validator_mode(
                True,
                state_payload=self._public_validator_state(),
                cai_url="http://127.0.0.1:52425",
            )

    def test_validator_mode_requires_static_ip_confirmation(self) -> None:
        wallet = create_wallet("validator", "pass", select=True)
        self._credit_wallet_from_reserve(wallet, 2_000_000_000_000)
        unlock_wallet("pass", selector=wallet.wallet_id)

        status = get_validator_mode_status(
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )
        self.assertFalse(status.can_enable)
        self.assertIn("Confirm that this validator node uses a static public IP", status.reason)

    def test_validator_mode_rejects_nat_backed_node(self) -> None:
        wallet = create_wallet("validator", "pass", select=True)
        self._credit_wallet_from_reserve(wallet, 2_000_000_000_000)
        unlock_wallet("pass", selector=wallet.wallet_id)
        set_validator_static_ip_confirmation(True)

        status = get_validator_mode_status(
            state_payload=self._nat_backed_state(),
            cai_url="http://127.0.0.1:52425",
        )
        self.assertFalse(status.can_enable)
        self.assertIn("behind NAT or a relay", status.reason)

    def test_validator_network_status_prefers_local_hostname_when_ports_collide(self) -> None:
        set_validator_static_ip_confirmation(True)
        state_payload = {
            "nodeIdentities": {
                "node-validator": {
                    "friendlyName": "vps-node",
                    "osVersion": "Linux",
                    "apiHost": "85.137.164.250",
                    "apiPort": 52415,
                    "dataHost": "85.137.164.250",
                    "dataPort": 52436,
                },
                "node-worker": {
                    "friendlyName": "DESKTOP-2NPNQ21",
                    "osVersion": "Windows",
                    "apiHost": None,
                    "apiPort": 52415,
                    "dataHost": None,
                    "dataPort": None,
                },
            },
            "nodeNetwork": {
                "node-validator": {
                    "interfaces": [
                        {"name": "lo", "ipAddress": "127.0.0.1"},
                        {"name": "ens3", "ipAddress": "85.137.164.250"},
                    ]
                },
                "node-worker": {
                    "interfaces": [
                        {"name": "lo", "ipAddress": "127.0.0.1"},
                        {"name": "wifi", "ipAddress": "192.168.0.103"},
                    ]
                },
            },
        }

        with patch.dict(
            "os.environ",
            {"COMPUTERNAME": "", "HOSTNAME": "vps-node"},
            clear=False,
        ), patch("socket.gethostname", return_value="vps-node"):
            status = assess_validator_network_status(
                state_payload=state_payload,
                cai_url="http://127.0.0.1:52415",
            )

        self.assertTrue(status.can_enable)
        self.assertEqual(status.current_node_id, "node-validator")
        self.assertEqual(status.advertised_api_host, "85.137.164.250")
        self.assertEqual(status.advertised_data_host, "85.137.164.250")

    def test_stale_validator_binding_is_migrated_to_disabled(self) -> None:
        config_path = node_config_file_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(
                {
                    "validator_enabled": True,
                    "validator_state": "bonded",
                    "validator_bond_atomic": 0,
                    "worker_enabled": False,
                    "worker_allowed_model_ids": ["cai-network/qwen3-0.6b-4bit"],
                    "worker_max_parallel_jobs": 1,
                    "worker_max_memory_mb": None,
                    "worker_reward_address_by_node_id": {},
                }
            ),
            encoding="utf-8",
        )

        config = load_or_create_node_config()
        self.assertFalse(config.validator_enabled)
        self.assertEqual(config.validator_state, "unbonded")
        self.assertIsNone(config.validator_wallet_id)
        self.assertIsNone(config.validator_address)
        self.assertIsNone(get_validator_identity())

    def test_legacy_private_model_alias_is_migrated_in_worker_config(self) -> None:
        config_path = node_config_file_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(
                {
                    "validator_enabled": False,
                    "validator_state": "unbonded",
                    "worker_enabled": True,
                    "worker_allowed_model_ids": ["cai-network/qwen3-0.6b-4bit"],
                    "worker_max_parallel_jobs": 1,
                    "worker_max_memory_mb": None,
                    "worker_reward_address_by_node_id": {},
                }
            ),
            encoding="utf-8",
        )

        config = load_or_create_node_config()
        self.assertEqual(
            config.worker_allowed_model_ids,
            default_worker_allowed_model_ids(),
        )

    def test_empty_worker_model_allow_list_is_preserved(self) -> None:
        config_path = node_config_file_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(
                {
                    "validator_enabled": False,
                    "validator_state": "unbonded",
                    "worker_enabled": True,
                    "worker_allowed_model_ids": [],
                    "worker_max_parallel_jobs": 1,
                    "worker_max_memory_mb": None,
                    "worker_reward_address_by_node_id": {},
                }
            ),
            encoding="utf-8",
        )

        config = load_or_create_node_config()
        self.assertEqual(config.worker_allowed_model_ids, [])

    def test_stale_dual_role_config_disables_worker(self) -> None:
        config_path = node_config_file_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(
                {
                    "validator_enabled": True,
                    "validator_state": "bonded",
                    "validator_wallet_id": "wallet-a",
                    "validator_address": "ABCDEF1234567890ABCDEF1234567890",
                    "validator_bond_atomic": 1_000_000_000_000,
                    "validator_static_ip_confirmed": True,
                    "worker_enabled": True,
                    "worker_allowed_model_ids": ["cai-network/qwen3-0.6b-4bit"],
                    "worker_max_parallel_jobs": 1,
                    "worker_max_memory_mb": None,
                    "worker_reward_address_by_node_id": {},
                }
            ),
            encoding="utf-8",
        )

        config = load_or_create_node_config()
        self.assertTrue(config.validator_enabled)
        self.assertEqual(config.validator_state, "bonded")
        self.assertFalse(config.worker_enabled)

    def test_stale_unbonding_dual_role_config_disables_worker(self) -> None:
        config_path = node_config_file_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(
                {
                    "validator_enabled": False,
                    "validator_state": "unbonding",
                    "validator_wallet_id": "wallet-a",
                    "validator_address": "ABCDEF1234567890ABCDEF1234567890",
                    "validator_bond_atomic": 1_000_000_000_000,
                    "validator_static_ip_confirmed": True,
                    "validator_unbonding_started_at": "2026-01-01T00:00:00+00:00",
                    "validator_unbonding_available_at": "2026-01-02T00:00:00+00:00",
                    "worker_enabled": True,
                    "worker_allowed_model_ids": ["cai-network/qwen3-0.6b-4bit"],
                    "worker_max_parallel_jobs": 1,
                    "worker_max_memory_mb": None,
                    "worker_reward_address_by_node_id": {},
                }
            ),
            encoding="utf-8",
        )

        config = load_or_create_node_config()
        self.assertFalse(config.validator_enabled)
        self.assertEqual(config.validator_state, "unbonding")
        self.assertFalse(config.worker_enabled)

    def test_validator_mode_rejects_enabled_worker_mode(self) -> None:
        wallet = create_wallet("validator", "pass", select=True)
        self._credit_wallet_from_reserve(wallet, 2_000_000_000_000)
        unlock_wallet("pass", selector=wallet.wallet_id)
        set_worker_mode(enabled=True)
        set_validator_static_ip_confirmation(True)

        status = get_validator_mode_status(
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )
        self.assertFalse(status.can_enable)
        self.assertIn("Disable worker mode before enabling validator mode.", status.reason)

        with self.assertRaisesRegex(
            ValueError, "Disable worker mode before enabling validator mode."
        ):
            set_validator_mode(
                True,
                state_payload=self._public_validator_state(),
                cai_url="http://127.0.0.1:52425",
            )

    def test_jail_validator_slashes_bond_and_blocks_reenable(self) -> None:
        wallet = create_wallet("validator", "pass", select=True)
        self._credit_wallet_from_reserve(wallet, 2_000_000_000_000)
        unlock_wallet("pass", selector=wallet.wallet_id)
        set_validator_static_ip_confirmation(True)
        set_validator_mode(
            True,
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )

        jailed = jail_validator(reason="Validator lost runtime eligibility.")
        self.assertFalse(jailed.validator_enabled)
        self.assertEqual(jailed.validator_state, "jailed")
        self.assertEqual(jailed.validator_last_slash_atomic, 50_000_000_000)
        self.assertEqual(jailed.validator_total_slashed_atomic, 50_000_000_000)
        self.assertEqual(jailed.validator_bond_atomic, 0)
        self.assertEqual(jailed.validator_jail_reason, "Validator lost runtime eligibility.")
        self.assertIsNotNone(jailed.validator_jailed_at)

        jailed_wallet = find_wallet_by_id(wallet.wallet_id)
        self.assertIsNotNone(jailed_wallet)
        assert jailed_wallet is not None
        self.assertEqual(jailed_wallet.validator_reserved_atomic, 0)
        self.assertEqual(jailed_wallet.spendable_balance_atomic, 1_950_000_000_000)

        ledger = load_or_create_ledger()
        self.assertEqual(ledger.validator_slashed_atomic, 50_000_000_000)
        records = list_validator_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].state, "jailed")
        self.assertEqual(records[0].last_slash_atomic, 50_000_000_000)

        status = get_validator_mode_status(
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )
        self.assertFalse(status.can_enable)
        self.assertIn("Validator is jailed", status.reason)

    def test_clear_validator_jail_returns_validator_to_unbonded(self) -> None:
        wallet = create_wallet("validator", "pass", select=True)
        self._credit_wallet_from_reserve(wallet, 2_000_000_000_000)
        unlock_wallet("pass", selector=wallet.wallet_id)
        set_validator_static_ip_confirmation(True)
        set_validator_mode(
            True,
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )
        jailed = jail_validator(reason="Validator lost runtime eligibility.")

        with self.assertRaisesRegex(ValueError, "cooldown is still active"):
            clear_validator_jail()

        available_at = datetime.fromisoformat(jailed.validator_unjail_available_at)
        cleared = clear_validator_jail(now=available_at + timedelta(seconds=1))
        self.assertFalse(cleared.validator_enabled)
        self.assertEqual(cleared.validator_state, "unbonded")
        self.assertIsNone(cleared.validator_wallet_id)
        self.assertIsNone(cleared.validator_address)
        self.assertIsNone(cleared.validator_jailed_at)
        self.assertIsNone(cleared.validator_unjail_available_at)
        self.assertIsNone(cleared.validator_jail_reason)
        self.assertEqual(cleared.validator_last_slash_atomic, 0)
        self.assertEqual(cleared.validator_total_slashed_atomic, 50_000_000_000)
        records = list_validator_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].state, "unbonded")
        self.assertEqual(records[0].total_slashed_atomic, 50_000_000_000)

    def test_complete_validator_unbond_releases_bond_after_delay(self) -> None:
        wallet = create_wallet("validator", "pass", select=True)
        self._credit_wallet_from_reserve(wallet, 2_000_000_000_000)
        unlock_wallet("pass", selector=wallet.wallet_id)
        set_validator_static_ip_confirmation(True)
        set_validator_mode(
            True,
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
        )
        unbonding = set_validator_mode(False)
        self.assertEqual(unbonding.validator_state, "unbonding")

        with self.assertRaisesRegex(ValueError, "unbonding period is still active"):
            complete_validator_unbond()

        available_at = datetime.fromisoformat(unbonding.validator_unbonding_available_at)
        completed = complete_validator_unbond(now=available_at + timedelta(seconds=1))
        self.assertEqual(completed.validator_state, "unbonded")
        self.assertEqual(completed.validator_bond_atomic, 0)
        self.assertIsNone(completed.validator_wallet_id)
        self.assertIsNone(completed.validator_address)
        self.assertIsNone(completed.validator_unbonding_started_at)
        self.assertIsNone(completed.validator_unbonding_available_at)

        wallet_after = find_wallet_by_id(wallet.wallet_id)
        self.assertIsNotNone(wallet_after)
        assert wallet_after is not None
        self.assertEqual(wallet_after.spendable_balance_atomic, 2_000_000_000_000)
        self.assertEqual(wallet_after.validator_reserved_atomic, 0)

        records = list_validator_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].state, "unbonded")

    def test_validator_bond_uses_chain_when_wallet_has_chain_balance(self) -> None:
        money_policy = MoneyPolicy()
        wallet = create_wallet("validator", "pass", select=True)
        unlock_wallet("pass", selector=wallet.wallet_id)
        self._credit_wallet_from_reserve(wallet, 2_000_000_000_000)
        set_validator_static_ip_confirmation(True)

        bonded = set_validator_mode(
            True,
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
            money_policy=money_policy,
        )

        self.assertEqual(bonded.validator_state, "bonded")
        self.assertEqual(
            chain_balance_atomic(wallet.address),
            1_000_000_000_000,
        )
        self.assertEqual(
            chain_balance_atomic(validator_bond_pool_chain_address(money_policy)),
            1_000_000_000_000,
        )
        bonded_wallet = find_wallet_by_id(wallet.wallet_id)
        self.assertIsNotNone(bonded_wallet)
        assert bonded_wallet is not None
        self.assertEqual(bonded_wallet.spendable_balance_atomic, 1_000_000_000_000)
        self.assertEqual(bonded_wallet.validator_reserved_atomic, 1_000_000_000_000)
        self.assertEqual(
            chain_summary()["validatorBondLockedCoins"],
            "10000.00000000",
        )
        bond_summary = chain_summary()
        self.assertEqual(bond_summary["validatorBondLockedValidatorCount"], 1)
        self.assertEqual(
            bond_summary["validatorBondLockedByValidatorCoins"],
            {bonded.validator_address: "10000.00000000"},
        )
        self.assertEqual(
            bond_summary["validatorBondLockedIndexTotalCoins"],
            "10000.00000000",
        )
        self.assertTrue(bond_summary["validatorBondLockedIndexMatchesPool"])

        unbonding = set_validator_mode(False, money_policy=money_policy)
        available_at = datetime.fromisoformat(unbonding.validator_unbonding_available_at)
        completed = complete_validator_unbond(
            now=available_at + timedelta(seconds=1),
        )

        self.assertEqual(completed.validator_state, "unbonded")
        self.assertEqual(chain_balance_atomic(wallet.address), 2_000_000_000_000)
        self.assertEqual(
            chain_balance_atomic(validator_bond_pool_chain_address(money_policy)),
            0,
        )
        released_summary = chain_summary()
        self.assertEqual(released_summary["validatorBondLockedValidatorCount"], 0)
        self.assertEqual(released_summary["validatorBondLockedIndexTotalCoins"], "0.00000000")
        self.assertTrue(released_summary["validatorBondLockedIndexMatchesPool"])

    def test_chain_backed_validator_jail_slashes_bond_on_chain(self) -> None:
        money_policy = MoneyPolicy()
        wallet = create_wallet("validator", "pass", select=True)
        unlock_wallet("pass", selector=wallet.wallet_id)
        self._credit_wallet_from_reserve(wallet, 2_000_000_000_000)
        set_validator_static_ip_confirmation(True)
        set_validator_mode(
            True,
            state_payload=self._public_validator_state(),
            cai_url="http://127.0.0.1:52425",
            money_policy=money_policy,
        )

        jailed = jail_validator(
            reason="Validator lost runtime eligibility.",
            money_policy=money_policy,
        )

        self.assertEqual(jailed.validator_state, "jailed")
        self.assertEqual(jailed.validator_last_slash_atomic, 50_000_000_000)
        self.assertEqual(chain_balance_atomic(wallet.address), 1_950_000_000_000)
        self.assertEqual(
            chain_balance_atomic(validator_bond_pool_chain_address(money_policy)),
            0,
        )
        self.assertEqual(
            chain_balance_atomic(validator_slash_pool_chain_address(money_policy)),
            50_000_000_000,
        )
        self.assertEqual(
            chain_summary()["validatorSlashedCoins"],
            "500.00000000",
        )
        slash_summary = chain_summary()
        self.assertEqual(slash_summary["validatorBondLockedValidatorCount"], 0)
        self.assertEqual(slash_summary["validatorBondLockedIndexTotalCoins"], "0.00000000")
        self.assertTrue(slash_summary["validatorBondLockedIndexMatchesPool"])

    def test_bind_worker_reward_address(self) -> None:
        config = bind_worker_reward_address(
            "node-a", "ABCDEF1234567890ABCDEF1234567890"
        )
        self.assertEqual(
            config.worker_reward_address_by_node_id["node-a"],
            "abcdef1234567890abcdef1234567890",
        )
        self.assertEqual(
            resolve_worker_reward_address("node-a"),
            "abcdef1234567890abcdef1234567890",
        )

    def _public_validator_state(
        self,
        *,
        node_id: str = "node-public",
        ip_address: str = "85.137.164.250",
        api_port: int = 52425,
    ) -> dict:
        return {
            "nodeIdentities": {
                node_id: {
                    "apiHost": ip_address,
                    "apiPort": api_port,
                    "dataHost": ip_address,
                    "dataPort": 52435,
                }
            },
            "nodeNetwork": {
                node_id: {
                    "interfaces": [
                        {"name": "eth0", "ipAddress": ip_address},
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


if __name__ == "__main__":
    unittest.main()

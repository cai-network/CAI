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

from cai_compute_chain.model import ChainNetwork, WalletPolicy
from cai_compute_chain.peer_payload import add_peer_payload_metadata, sign_peer_payload
from cai_compute_chain.validators import (
    build_validator_committee_snapshot,
    discover_peer_cai_urls,
    export_validator_set_payload,
    list_validator_records,
    merge_remote_validator_set_payload,
    select_validator_committee_snapshot,
    sync_validator_record,
    sync_validator_set_from_cai_peers,
)
from cai_compute_chain.wallet_signing import (
    address_from_public_key_b64,
    encode_bytes,
    generate_signing_seed,
    public_key_b64_from_seed,
)


class ValidatorsTests(unittest.TestCase):
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

    def test_export_validator_set_payload_includes_committee(self) -> None:
        sync_validator_record(
            validator_id="validator-a",
            wallet_id="wallet-a",
            address="validator-a",
            state="bonded",
            bonded_atomic=1_000,
            static_ip_confirmed=True,
            current_node_id="node-a",
            advertised_api_host="85.137.164.250",
            advertised_data_host="85.137.164.250",
        )
        payload = export_validator_set_payload()
        self.assertEqual(payload["network"], "mainnet")
        self.assertEqual(payload["chain_id"], "mainnet")
        self.assertTrue(payload["genesis_hash"])
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(len(payload["records"]), 1)
        self.assertEqual(payload["committee"]["total_bonded_atomic"], 1_000)
        self.assertEqual(payload["committee"]["quorum_bond_atomic"], 667)

    def test_validator_set_payload_uses_wallet_policy_network(self) -> None:
        policy = WalletPolicy(chain_network=ChainNetwork.TESTNET)

        payload = export_validator_set_payload(policy)

        self.assertEqual(payload["network"], "testnet")
        self.assertEqual(payload["chain_id"], "testnet")
        self.assertTrue(payload["genesis_hash"])

    def test_merge_remote_validator_set_payload_imports_peer_records(self) -> None:
        imported = merge_remote_validator_set_payload(
            add_peer_payload_metadata({
                "records": [
                    {
                        "validator_id": "validator-remote",
                        "wallet_id": "wallet-remote",
                        "address": "validator-remote",
                        "state": "bonded",
                        "bonded_atomic": 2_000,
                        "static_ip_confirmed": True,
                        "current_node_id": "node-remote",
                        "advertised_api_host": "85.137.164.251",
                        "advertised_data_host": "85.137.164.251",
                        "updated_at": "2026-04-22T00:00:00+00:00",
                    }
                ]
            }),
            source_url="http://85.137.164.251:52415/v1/cai/validators",
        )
        self.assertEqual(imported, 1)
        records = list_validator_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source, "peer")
        self.assertEqual(records[0].source_url, "http://85.137.164.251:52415/v1/cai/validators")
        self.assertEqual(records[0].updated_at, "2026-04-22T00:00:00+00:00")

    def test_merge_remote_validator_set_payload_uses_authoritative_remote_updated_at(
        self,
    ) -> None:
        source_url = "http://85.137.164.250:52415/v1/cai/validators"
        first_payload = add_peer_payload_metadata({
            "records": [
                {
                    "validator_id": "validator-remote",
                    "wallet_id": "wallet-remote",
                    "address": "validator-remote",
                    "state": "jailed",
                    "bonded_atomic": 0,
                    "static_ip_confirmed": True,
                    "current_node_id": "node-remote",
                    "advertised_api_host": "85.137.164.250",
                    "advertised_data_host": "85.137.164.250",
                    "updated_at": "2026-05-13T02:28:18.092590+00:00",
                }
            ]
        })
        second_payload = add_peer_payload_metadata({
            "records": [
                {
                    "validator_id": "validator-remote",
                    "wallet_id": "wallet-remote",
                    "address": "validator-remote",
                    "state": "bonded",
                    "bonded_atomic": 1_000,
                    "static_ip_confirmed": True,
                    "current_node_id": "node-remote",
                    "advertised_api_host": "85.137.164.250",
                    "advertised_data_host": "85.137.164.250",
                    "updated_at": "2026-05-13T03:32:41.328847+00:00",
                }
            ]
        })

        merge_remote_validator_set_payload(first_payload, source_url=source_url)
        first = list_validator_records()[0]
        self.assertEqual(first.state, "jailed")
        self.assertEqual(first.updated_at, "2026-05-13T02:28:18.092590+00:00")

        merge_remote_validator_set_payload(second_payload, source_url=source_url)
        second = list_validator_records()[0]
        self.assertEqual(second.state, "bonded")
        self.assertEqual(second.bonded_atomic, 1_000)
        self.assertEqual(second.updated_at, "2026-05-13T03:32:41.328847+00:00")

    def test_signed_validator_payload_prunes_stale_peer_record_for_same_node(self) -> None:
        signing_seed = generate_signing_seed()
        public_key_b64 = public_key_b64_from_seed(signing_seed)
        validator_address = address_from_public_key_b64(public_key_b64)
        sync_validator_record(
            validator_id="stale-validator",
            wallet_id="wallet-stale",
            address="stale-validator",
            state="bonded",
            bonded_atomic=1_000,
            static_ip_confirmed=True,
            current_node_id="node-vps",
            advertised_api_host="85.137.164.250",
            advertised_data_host="85.137.164.250",
            source="peer",
        )
        payload = sign_peer_payload(
            add_peer_payload_metadata({
                "records": [
                    {
                        "validator_id": validator_address,
                        "wallet_id": "wallet-validator",
                        "address": validator_address,
                        "state": "bonded",
                        "bonded_atomic": 2_000,
                        "static_ip_confirmed": True,
                        "current_node_id": "node-vps",
                        "advertised_api_host": "85.137.164.250",
                        "advertised_data_host": "85.137.164.250",
                    }
                ],
            }),
            public_key_b64=public_key_b64,
            signing_seed_b64=encode_bytes(signing_seed),
            signer_address=validator_address,
        )

        imported = merge_remote_validator_set_payload(
            payload,
            source_url="http://85.137.164.250:52415/v1/cai/validators",
        )

        records = list_validator_records()
        self.assertEqual(imported, 1)
        self.assertEqual([item.validator_id for item in records], [validator_address])

    def test_signed_validator_payload_keeps_one_bonded_participant_per_attestation_source(
        self,
    ) -> None:
        signing_seed = generate_signing_seed()
        public_key_b64 = public_key_b64_from_seed(signing_seed)
        validator_address = address_from_public_key_b64(public_key_b64)
        payload = sign_peer_payload(
            add_peer_payload_metadata({
                "records": [
                    {
                        "validator_id": validator_address,
                        "wallet_id": "wallet-validator",
                        "address": validator_address,
                        "state": "bonded",
                        "bonded_atomic": 2_000,
                        "static_ip_confirmed": True,
                        "current_node_id": "node-validator-a",
                        "advertised_api_host": "85.137.164.250",
                        "advertised_data_host": "85.137.164.250",
                    },
                    {
                        "validator_id": "validator-shadow",
                        "wallet_id": "wallet-shadow",
                        "address": "validator-shadow",
                        "state": "bonded",
                        "bonded_atomic": 1_500,
                        "static_ip_confirmed": True,
                        "current_node_id": "node-validator-b",
                        "advertised_api_host": "85.137.164.250",
                        "advertised_data_host": "85.137.164.250",
                    },
                ],
            }),
            public_key_b64=public_key_b64,
            signing_seed_b64=encode_bytes(signing_seed),
            signer_address=validator_address,
        )

        imported = merge_remote_validator_set_payload(
            payload,
            source_url="http://85.137.164.250:52415/v1/cai/validators",
        )

        records = list_validator_records()
        committee = build_validator_committee_snapshot()
        self.assertEqual(imported, 2)
        self.assertEqual([item.validator_id for item in records], [validator_address])
        self.assertEqual(committee.validator_ids, [validator_address])
        self.assertEqual(committee.quorum_bond_atomic, 1_334)

    def test_merge_remote_validator_set_payload_rejects_other_network(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Refusing validator set payload for network 'testnet' on 'mainnet'",
        ):
            merge_remote_validator_set_payload(
                {
                    "network": "testnet",
                    "chain_id": "testnet",
                    "schema_version": 1,
                    "records": [
                        {
                            "validator_id": "validator-remote",
                            "wallet_id": "wallet-remote",
                            "address": "validator-remote",
                            "state": "bonded",
                            "bonded_atomic": 2_000,
                        }
                    ],
                },
                source_url="http://85.137.164.251:52415/v1/cai/validators",
            )

        self.assertEqual(list_validator_records(), [])

    def test_merge_remote_validator_set_payload_rejects_mismatched_metadata(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "mismatched chain_id 'mainnet' and network 'testnet'",
        ):
            merge_remote_validator_set_payload(
                {
                    "network": "testnet",
                    "chain_id": "mainnet",
                    "schema_version": 1,
                    "records": [],
                },
                source_url="http://85.137.164.251:52415/v1/cai/validators",
            )

    def test_merge_remote_validator_set_payload_rejects_invalid_signature(
        self,
    ) -> None:
        signing_seed = generate_signing_seed()
        payload = sign_peer_payload(
            add_peer_payload_metadata({
                "records": [
                    {
                        "validator_id": "validator-remote",
                        "wallet_id": "wallet-remote",
                        "address": "validator-remote",
                        "state": "bonded",
                        "bonded_atomic": 2_000,
                    }
                ],
            }),
            public_key_b64=public_key_b64_from_seed(signing_seed),
            signing_seed_b64=encode_bytes(signing_seed),
        )
        payload["records"][0]["bonded_atomic"] = 3_000

        with self.assertRaisesRegex(ValueError, "signature is invalid"):
            merge_remote_validator_set_payload(
                payload,
                source_url="http://85.137.164.251:52415/v1/cai/validators",
            )

    def test_merge_remote_validator_set_payload_rejects_unsigned_in_strict_mode(
        self,
    ) -> None:
        with patch.dict(
            "os.environ",
            {"CAI_REQUIRE_SIGNED_PEER_PAYLOADS": "1"},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "signature is missing"):
                merge_remote_validator_set_payload(
                    add_peer_payload_metadata({
                        "records": [],
                    }),
                    source_url="http://85.137.164.251:52415/v1/cai/validators",
                )

    def test_merge_remote_validator_summary_payload_imports_peer_records(self) -> None:
        imported = merge_remote_validator_set_payload(
            add_peer_payload_metadata({
                "validators": [
                    {
                        "validatorId": "validator-remote",
                        "address": "validator-remote",
                        "state": "bonded",
                        "bondedAtomic": 2_500,
                        "staticIpConfirmed": True,
                        "nodeId": "node-remote",
                        "apiHost": "85.137.164.251",
                        "dataHost": "85.137.164.251",
                        "updatedAt": "2026-04-22T00:00:00+00:00",
                    }
                ]
            }),
            source_url="http://85.137.164.251:52415/v1/cai/validators",
        )
        self.assertEqual(imported, 1)
        records = list_validator_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source, "peer")
        self.assertEqual(records[0].bonded_atomic, 2_500)
        self.assertEqual(records[0].current_node_id, "node-remote")

    def test_remote_merge_does_not_override_local_record(self) -> None:
        sync_validator_record(
            validator_id="validator-a",
            wallet_id="wallet-local",
            address="validator-a",
            state="bonded",
            bonded_atomic=10_000,
            static_ip_confirmed=True,
            current_node_id="node-local",
            advertised_api_host="85.137.164.250",
            advertised_data_host="85.137.164.250",
            source="local",
        )
        imported = merge_remote_validator_set_payload(
            add_peer_payload_metadata({
                "records": [
                    {
                        "validator_id": "validator-a",
                        "wallet_id": "wallet-remote",
                        "address": "validator-a",
                        "state": "unbonded",
                        "bonded_atomic": 0,
                        "static_ip_confirmed": False,
                        "updated_at": "2030-01-01T00:00:00+00:00",
                    }
                ]
            }),
            source_url="http://85.137.164.251:52415/v1/cai/validators",
        )
        self.assertEqual(imported, 0)
        records = list_validator_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source, "local")
        self.assertEqual(records[0].bonded_atomic, 10_000)
        self.assertEqual(records[0].state, "bonded")

    def test_remote_merge_updates_local_ha_active_lease(self) -> None:
        sync_validator_record(
            validator_id="validator-a",
            wallet_id="wallet-local",
            address="validator-a",
            state="bonded",
            bonded_atomic=10_000,
            static_ip_confirmed=True,
            current_node_id="node-standby",
            advertised_api_host="85.137.164.251",
            advertised_data_host="85.137.164.251",
            ha_enabled=True,
            ha_role="passive",
            replica_node_ids=["node-standby"],
            source="local",
        )

        imported = merge_remote_validator_set_payload(
            add_peer_payload_metadata({
                "records": [
                    {
                        "validator_id": "validator-a",
                        "wallet_id": "wallet-active",
                        "address": "validator-a",
                        "state": "bonded",
                        "bonded_atomic": 10_000,
                        "static_ip_confirmed": True,
                        "current_node_id": "node-active",
                        "advertised_api_host": "85.137.164.250",
                        "advertised_data_host": "85.137.164.250",
                        "ha_enabled": True,
                        "active_replica_node_id": "node-active",
                        "active_replica_lease_until": "2999-01-01T00:00:00+00:00",
                        "replica_node_ids": ["node-active"],
                        "updated_at": "2026-05-19T00:00:00+00:00",
                    }
                ]
            }),
            source_url="http://85.137.164.250:52415/v1/cai/validators",
        )

        records = list_validator_records()
        self.assertEqual(imported, 1)
        self.assertEqual(records[0].source, "local")
        self.assertEqual(records[0].wallet_id, "wallet-local")
        self.assertEqual(records[0].current_node_id, "node-active")
        self.assertEqual(records[0].active_replica_node_id, "node-active")
        self.assertEqual(
            records[0].active_replica_lease_until,
            "2999-01-01T00:00:00+00:00",
        )
        self.assertEqual(records[0].replica_node_ids, ["node-standby", "node-active"])

    def test_merge_remote_validator_set_payload_prefers_validator_advertised_api_host(self) -> None:
        imported = merge_remote_validator_set_payload(
            add_peer_payload_metadata({
                "records": [
                    {
                        "validator_id": "validator-remote",
                        "wallet_id": "wallet-remote",
                        "address": "validator-remote",
                        "state": "bonded",
                        "bonded_atomic": 2_000,
                        "static_ip_confirmed": True,
                        "current_node_id": "node-remote",
                        "advertised_api_host": "85.137.164.252",
                        "advertised_data_host": "85.137.164.252",
                        "updated_at": "2026-04-22T00:00:00+00:00",
                    }
                ]
            }),
            source_url="http://85.137.164.251:52415/v1/cai/validators",
        )

        self.assertEqual(imported, 1)
        records = list_validator_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(
            records[0].source_url,
            "http://85.137.164.252:52415/v1/cai/validators",
        )

    def test_sync_validator_set_from_cai_peers_imports_remote_records(self) -> None:
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
            self.assertEqual(url, "http://85.137.164.251:52415/v1/cai/validators")
            return FakeResponse(
                add_peer_payload_metadata({
                    "records": [
                        {
                            "validator_id": "validator-remote",
                            "wallet_id": "wallet-remote",
                            "address": "validator-remote",
                            "state": "bonded",
                            "bonded_atomic": 4000,
                            "static_ip_confirmed": True,
                            "current_node_id": "node-remote",
                            "advertised_api_host": "85.137.164.251",
                            "advertised_data_host": "85.137.164.251",
                            "updated_at": "2026-04-22T00:00:00+00:00",
                        }
                    ]
                })
            )

        with patch("cai_compute_chain.validators.urlopen", side_effect=fake_urlopen):
            result = sync_validator_set_from_cai_peers(
                state_payload=state_payload,
                cai_url="http://127.0.0.1:52425",
            )

        self.assertEqual(result.attempted_peers, 1)
        self.assertEqual(result.successful_peers, 1)
        self.assertEqual(result.imported_records, 1)
        committee = build_validator_committee_snapshot()
        self.assertEqual(committee.total_bonded_atomic, 4000)
        self.assertEqual(committee.quorum_bond_atomic, 2667)

    def test_sync_validator_set_from_cai_peers_records_peer_errors(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-a": {
                    "apiHost": "85.137.164.251",
                    "apiPort": 52415,
                },
                "node-b": {
                    "apiHost": "85.137.164.252",
                    "apiPort": 52415,
                },
            }
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return json.dumps(
                    add_peer_payload_metadata({
                        "records": [
                            {
                                "validator_id": "validator-remote",
                                "wallet_id": "wallet-remote",
                                "address": "validator-remote",
                                "state": "bonded",
                                "bonded_atomic": 4000,
                                "static_ip_confirmed": True,
                                "current_node_id": "node-b",
                                "advertised_api_host": "85.137.164.252",
                                "advertised_data_host": "85.137.164.252",
                                "updated_at": "2026-04-22T00:00:00+00:00",
                            }
                        ]
                    })
                ).encode("utf-8")

        def fake_urlopen(url: str, timeout: int = 0):
            if "85.137.164.251" in url:
                raise OSError("validator peer offline")
            return FakeResponse()

        with patch("cai_compute_chain.validators.urlopen", side_effect=fake_urlopen):
            result = sync_validator_set_from_cai_peers(
                state_payload=state_payload,
                cai_url="http://127.0.0.1:52425",
            )

        self.assertEqual(result.attempted_peers, 2)
        self.assertEqual(result.successful_peers, 1)
        self.assertEqual(result.failed_peers, 1)
        self.assertEqual(
            result.failed_peer_urls,
            ["http://85.137.164.251:52415/v1/cai/validators"],
        )
        self.assertEqual(result.peer_errors[0]["errorType"], "OSError")
        self.assertIn("validator peer offline", result.peer_errors[0]["message"])
        self.assertEqual(result.imported_records, 1)

    def test_discover_peer_cai_urls_does_not_filter_single_remote_identity(self) -> None:
        urls = discover_peer_cai_urls(
            state_payload={
                "nodeIdentities": {
                    "node-remote": {
                        "apiHost": "85.137.164.250",
                        "apiPort": 52415,
                    }
                }
            },
            cai_url="http://127.0.0.1:52425",
            endpoint_path="/v1/cai/validators",
        )

        self.assertEqual(
            urls,
            ["http://85.137.164.250:52415/v1/cai/validators"],
        )

    def test_discover_peer_cai_urls_formats_ipv6_remote_identity(self) -> None:
        urls = discover_peer_cai_urls(
            state_payload={
                "nodeIdentities": {
                    "node-local": {
                        "apiHost": "127.0.0.1",
                        "apiPort": 52425,
                    },
                    "node-remote": {
                        "apiHost": "2001:db8::10",
                        "apiPort": 52415,
                    },
                }
            },
            cai_url="http://127.0.0.1:52425",
            endpoint_path="/v1/cai/validators",
        )

        self.assertEqual(
            urls,
            ["http://[2001:db8::10]:52415/v1/cai/validators"],
        )

    def test_discover_peer_cai_urls_uses_transport_endpoints_before_legacy_api_host(self) -> None:
        urls = discover_peer_cai_urls(
            state_payload={
                "nodeIdentities": {
                    "node-local": {
                        "apiHost": "127.0.0.1",
                        "apiPort": 52425,
                    },
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
            cai_url="http://127.0.0.1:52425",
            endpoint_path="/v1/cai/validators",
        )

        self.assertEqual(
            urls,
            [
                "http://85.137.164.250:52415/v1/cai/validators",
                "http://26.97.29.153:52415/v1/cai/validators",
                "http://198.51.100.99:52415/v1/cai/validators",
            ],
        )

    def test_select_validator_committee_snapshot_is_deterministic_and_bounded(self) -> None:
        for suffix, bond in [("a", 10_000), ("b", 9_000), ("c", 8_000), ("d", 7_000)]:
            sync_validator_record(
                validator_id=f"validator-{suffix}",
                wallet_id=f"wallet-{suffix}",
                address=f"validator-{suffix}",
                state="bonded",
                bonded_atomic=bond,
                static_ip_confirmed=True,
                current_node_id=f"node-{suffix}",
                advertised_api_host=f"host-{suffix}",
                advertised_data_host=f"host-{suffix}",
            )

        first = select_validator_committee_snapshot(
            selection_seed="settlement-seed-1",
            target_size=2,
        )
        second = select_validator_committee_snapshot(
            selection_seed="settlement-seed-1",
            target_size=2,
        )
        self.assertEqual(first.validator_ids, second.validator_ids)
        self.assertEqual(len(first.validator_ids), 2)
        self.assertEqual(first.total_bonded_atomic, sum(first.bonded_atomic_by_validator_id.values()))


if __name__ == "__main__":
    unittest.main()

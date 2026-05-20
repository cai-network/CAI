# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import base64
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

from cai_compute_chain.node_capabilities import (
    NodeCapabilityRecord,
    export_node_capabilities_payload,
    list_node_capabilities,
    list_verified_worker_node_ids,
    merge_remote_node_capabilities_payload,
    node_capability_convergence_audit,
    node_capabilities_file_path,
    prune_stale_node_capabilities,
    refresh_local_node_capabilities,
    save_node_capabilities,
    sync_node_capabilities_from_cai_peers,
    verified_node_capability_records_from_payload,
    worker_capability_verification_required,
)
from cai_compute_chain.model import WalletPolicy
from cai_compute_chain.peer_payload import add_peer_payload_metadata, sign_peer_payload
from cai_compute_chain.wallet_signing import (
    address_from_public_key_b64,
    generate_signing_seed,
    public_key_b64_from_seed,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class NodeCapabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo_patch = patch(
            "cai_compute_chain.wallet.repo_root",
            return_value=Path(self.tempdir.name),
        )
        self.repo_patch.start()
        self.policy = WalletPolicy(wallet_data_dirname=".tmp-node-capabilities")

    def tearDown(self) -> None:
        self.repo_patch.stop()
        self.tempdir.cleanup()

    def peer_payload(self, payload: dict) -> dict:
        return add_peer_payload_metadata(payload, policy=self.policy)

    def test_export_node_capabilities_normalizes_state_identities(self) -> None:
        advertised_peer = "/ip4/198.51.100.11/tcp/52416/p2p/node-b"
        node_public_key_b64 = public_key_b64_from_seed(generate_signing_seed())
        payload = export_node_capabilities_payload(
            state_payload={
                "nodeIdentities": {
                    "node-a": {
                        "friendlyName": "Worker A",
                        "nodePublicKeyB64": node_public_key_b64,
                        "apiHost": "198.51.100.10",
                        "apiPort": 52415,
                        "dataHost": "198.51.100.10",
                        "dataPort": 52418,
                        "workerEnabled": True,
                        "relayEnabled": True,
                        "workerRewardAddress": "ABCD1234ABCD1234ABCD1234ABCD1234",
                        "workerAllowedModelIds": ["Qwen/Qwen3-0.6B-GGUF"],
                        "resources": {"ramBytes": 16_000, "vramBytes": 8_000},
                        "transportEndpoints": [
                            {
                                "purpose": "api",
                                "routeType": "direct",
                                "host": "198.51.100.10",
                                "port": 52415,
                            },
                            {
                                "purpose": "data",
                                "routeType": "direct",
                                "host": "198.51.100.10",
                                "port": 52418,
                            },
                        ],
                    }
                },
                "overlayPeers": {"node-a": ["node-b"]},
                "overlayAdvertisedPeers": {
                    "node-a": [{"address": advertised_peer}],
                },
                "topology": {
                    "connections": {
                        "node-a": {
                            "node-b": [{"sinkMultiaddr": {"address": "/ip4/x"}}]
                        }
                    }
                },
            },
            cai_url="http://127.0.0.1:52415",
            local_node_id="node-a",
            policy=self.policy,
        )

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["chain_network"], self.policy.chain_network.value)
        self.assertTrue(payload["genesis_hash"])
        record = payload["records"][0]
        self.assertEqual(record["node_id"], "node-a")
        self.assertEqual(record["node_public_key_b64"], node_public_key_b64)
        self.assertEqual(
            record["node_public_key_address"],
            address_from_public_key_b64(node_public_key_b64),
        )
        self.assertTrue(record["worker_enabled"])
        self.assertTrue(record["relay_enabled"])
        self.assertIn("http://127.0.0.1:52415", record["api_urls"])
        self.assertEqual(record["worker_allowed_model_ids"], ["cai-network/Qwen3-0.6B-GGUF"])
        self.assertEqual(record["resource_summary"]["ramBytes"], 16_000)
        self.assertEqual(
            record["readiness"]["caiOwnedTransport"]["protocol"],
            "cai-owned-llm-shard-transport",
        )
        self.assertFalse(record["readiness"]["caiOwnedTransport"]["runtimeReady"])
        self.assertEqual(record["route_hints"]["overlayPeerIds"], ["node-b"])
        self.assertEqual(
            record["route_hints"]["overlayAdvertisedPeers"],
            [advertised_peer],
        )
        self.assertEqual(record["route_hints"]["directPeerIds"], ["node-b"])

    def test_export_node_capabilities_backfills_local_worker_config(self) -> None:
        with patch(
            "cai_compute_chain.node_config.load_or_create_node_config",
            return_value=SimpleNamespace(
                worker_enabled=True,
                relay_enabled=True,
                worker_allowed_model_ids=["cai-network/Qwen3-0.6B-GGUF"],
                worker_reward_address_by_node_id={
                    "node-local": "abcd1234abcd1234abcd1234abcd1234"
                },
            ),
        ):
            payload = export_node_capabilities_payload(
                state_payload={
                    "nodeIdentities": {
                        "node-local": {
                            "apiHost": "127.0.0.1",
                            "apiPort": 52415,
                            "workerEnabled": True,
                        }
                    }
                },
                cai_url="http://127.0.0.1:52415",
                local_node_id="node-local",
                policy=self.policy,
            )

        record = payload["records"][0]
        self.assertEqual(
            record["worker_allowed_model_ids"],
            ["cai-network/Qwen3-0.6B-GGUF"],
        )
        self.assertEqual(
            record["worker_reward_address"],
            "abcd1234abcd1234abcd1234abcd1234",
        )

    def test_refresh_merge_and_prune_node_capabilities(self) -> None:
        records = refresh_local_node_capabilities(
            state_payload={
                "nodeIdentities": {
                    "node-local": {
                        "apiHost": "127.0.0.1",
                        "apiPort": 52415,
                        "workerEnabled": False,
                        "relayEnabled": True,
                    }
                }
            },
            cai_url="http://127.0.0.1:52415",
            local_node_id="node-local",
            policy=self.policy,
        )
        self.assertEqual(len(records), 1)
        self.assertTrue(node_capabilities_file_path(self.policy).exists())

        imported = merge_remote_node_capabilities_payload(
            self.peer_payload({
                "records": [
                    {
                        "node_id": "node-remote",
                        "updated_at": "2026-05-02T00:00:00+00:00",
                        "last_seen_at": "2020-01-01T00:00:00+00:00",
                        "worker_enabled": True,
                        "api_urls": ["http://198.51.100.10:52415"],
                    }
                ]
            }),
            source_url="http://198.51.100.10:52415/v1/cai/node-capabilities",
            policy=self.policy,
        )
        self.assertEqual(imported, 1)
        self.assertEqual(len(list_node_capabilities(self.policy)), 2)

        pruned = prune_stale_node_capabilities(
            max_age_seconds=1,
            policy=self.policy,
        )
        self.assertEqual(pruned, 1)
        self.assertEqual(
            [item.node_id for item in list_node_capabilities(self.policy)],
            ["node-local"],
        )

    def test_list_node_capabilities_recovers_empty_store(self) -> None:
        path = node_capabilities_file_path(self.policy)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

        self.assertEqual(list_node_capabilities(self.policy), [])
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), [])

    def test_signed_worker_capability_becomes_verified(self) -> None:
        signing_seed = generate_signing_seed()
        public_key_b64 = public_key_b64_from_seed(signing_seed)
        reward_address = address_from_public_key_b64(public_key_b64)
        node_public_key_b64 = public_key_b64_from_seed(generate_signing_seed())
        payload = sign_peer_payload(
            self.peer_payload({
                "records": [
                    {
                        "node_id": "node-worker",
                        "updated_at": "2026-05-02T00:00:00+00:00",
                        "last_seen_at": "2026-05-02T00:00:00+00:00",
                        "worker_enabled": True,
                        "worker_reward_address": reward_address,
                        "node_public_key_b64": node_public_key_b64,
                        "api_urls": ["http://198.51.100.10:52415"],
                    }
                ]
            }),
            public_key_b64=public_key_b64,
            signing_seed_b64=base64.b64encode(signing_seed).decode("ascii"),
            signer_address=reward_address,
        )

        imported = merge_remote_node_capabilities_payload(
            payload,
            source_url="http://198.51.100.10:52415/v1/cai/node-capabilities",
            policy=self.policy,
        )

        self.assertEqual(imported, 1)
        records = list_node_capabilities(self.policy)
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0].worker_verified)
        self.assertEqual(
            list_verified_worker_node_ids(self.policy),
            {"node-worker"},
        )

    def test_submitted_signed_worker_payload_verifies_without_store_replacement(
        self,
    ) -> None:
        signing_seed = generate_signing_seed()
        public_key_b64 = public_key_b64_from_seed(signing_seed)
        reward_address = address_from_public_key_b64(public_key_b64)
        save_node_capabilities(
            [
                NodeCapabilityRecord(
                    node_id="node-worker",
                    source="peer",
                    source_url="http://198.51.100.10:52415/v1/cai/node-capabilities",
                    last_seen_at="2026-05-03T00:00:00+00:00",
                    updated_at="2026-05-03T00:00:00+00:00",
                    worker_enabled=True,
                    worker_reward_address=reward_address,
                    worker_verified=False,
                )
            ],
            self.policy,
        )
        payload = sign_peer_payload(
            self.peer_payload({
                "records": [
                    {
                        "node_id": "node-worker",
                        "updated_at": "2026-05-02T00:00:00+00:00",
                        "last_seen_at": "2026-05-02T00:00:00+00:00",
                        "worker_enabled": True,
                        "worker_reward_address": reward_address,
                    }
                ]
            }),
            public_key_b64=public_key_b64,
            signing_seed_b64=base64.b64encode(signing_seed).decode("ascii"),
            signer_address=reward_address,
        )

        imported = merge_remote_node_capabilities_payload(
            payload,
            source_url="http://198.51.100.10:52415/v1/cai/node-capabilities",
            policy=self.policy,
        )
        records = verified_node_capability_records_from_payload(
            payload,
            source_url="http://198.51.100.10:52415/v1/cai/node-capabilities",
            policy=self.policy,
            only_node_id="node-worker",
        )

        self.assertEqual(imported, 0)
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0].worker_verified)
        self.assertEqual(
            records[0].worker_verification_reason,
            "signed worker capability",
        )

    def test_merge_node_capabilities_can_limit_import_to_requested_node(self) -> None:
        payload = self.peer_payload({
            "records": [
                {
                    "node_id": "node-a",
                    "updated_at": "2026-05-02T00:00:00+00:00",
                    "last_seen_at": "2026-05-02T00:00:00+00:00",
                    "worker_enabled": True,
                    "api_urls": ["http://198.51.100.10:52415"],
                },
                {
                    "node_id": "node-b",
                    "updated_at": "2026-05-02T00:00:00+00:00",
                    "last_seen_at": "2026-05-02T00:00:00+00:00",
                    "worker_enabled": True,
                    "api_urls": ["http://198.51.100.11:52415"],
                },
            ]
        })

        imported = merge_remote_node_capabilities_payload(
            payload,
            source_url="http://198.51.100.10:52415/v1/cai/node-capabilities",
            policy=self.policy,
            only_node_id="node-b",
        )

        self.assertEqual(imported, 1)
        records = list_node_capabilities(self.policy)
        self.assertEqual([record.node_id for record in records], ["node-b"])

    def test_unsigned_remote_worker_capability_is_not_verified(self) -> None:
        imported = merge_remote_node_capabilities_payload(
            self.peer_payload({
                "records": [
                    {
                        "node_id": "node-fake",
                        "updated_at": "2026-05-02T00:00:00+00:00",
                        "last_seen_at": "2026-05-02T00:00:00+00:00",
                        "worker_enabled": True,
                        "worker_reward_address": "abcd1234abcd1234abcd1234abcd1234",
                        "resource_summary": {"vramBytes": 99_000_000_000},
                    }
                ]
            }),
            source_url="http://198.51.100.10:52415/v1/cai/node-capabilities",
            policy=self.policy,
        )

        self.assertEqual(imported, 1)
        record = list_node_capabilities(self.policy)[0]
        self.assertTrue(record.worker_enabled)
        self.assertFalse(record.worker_verified)
        self.assertEqual(list_verified_worker_node_ids(self.policy), set())

    def test_validator_attestation_can_verify_unsigned_live_state_record(self) -> None:
        model_id = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
        imported = merge_remote_node_capabilities_payload(
            self.peer_payload({
                "records": [
                    {
                        "node_id": "node-worker",
                        "updated_at": "2026-05-02T00:00:00+00:00",
                        "last_seen_at": "2026-05-02T00:00:00+00:00",
                        "worker_enabled": True,
                        "worker_reward_address": "abcd1234abcd1234abcd1234abcd1234",
                        "worker_allowed_model_ids": [model_id],
                    }
                ]
            }),
            source_url="http://198.51.100.10:52415/v1/cai/node-capabilities",
            policy=self.policy,
        )

        self.assertEqual(imported, 1)
        self.assertFalse(list_node_capabilities(self.policy)[0].worker_verified)

        with patch(
            "cai_compute_chain.worker_capability_attestations."
            "list_validator_attested_worker_node_ids",
            return_value={"node-worker"},
        ):
            self.assertEqual(
                list_verified_worker_node_ids(
                    self.policy,
                    accepted_model_ids={model_id},
                    require_validator_attestation=True,
                ),
                {"node-worker"},
            )

    def test_strict_node_capability_merge_rejects_unsigned_payload(self) -> None:
        with patch.dict("os.environ", {"CAI_REQUIRE_SIGNED_PEER_PAYLOADS": "1"}):
            with self.assertRaises(ValueError):
                merge_remote_node_capabilities_payload(
                    self.peer_payload({
                        "records": [
                            {
                                "node_id": "node-fake",
                                "updated_at": "2026-05-02T00:00:00+00:00",
                                "worker_enabled": True,
                                "worker_reward_address": (
                                    "abcd1234abcd1234abcd1234abcd1234"
                                ),
                            }
                        ]
                    }),
                    source_url="http://198.51.100.10:52415/v1/cai/node-capabilities",
                    policy=self.policy,
                )

    def test_worker_capability_verification_defaults_to_signed_peer_strictness(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(worker_capability_verification_required())
        with patch.dict("os.environ", {"CAI_REQUIRE_SIGNED_PEER_PAYLOADS": "1"}):
            self.assertTrue(worker_capability_verification_required())

    def test_sync_node_capabilities_from_peers(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-local": {"apiHost": "127.0.0.1", "apiPort": 52415},
                "node-peer": {"apiHost": "198.51.100.20", "apiPort": 52415},
            }
        }
        peer_payload = self.peer_payload({
            "records": [
                {
                    "node_id": "node-peer",
                    "updated_at": "2026-05-02T00:00:00+00:00",
                    "last_seen_at": "2026-05-02T00:00:00+00:00",
                    "relay_enabled": True,
                    "api_urls": ["http://198.51.100.20:52415"],
                }
            ]
        })

        with patch(
            "cai_compute_chain.node_capabilities.urlopen",
            return_value=_FakeResponse(peer_payload),
        ) as urlopen_mock:
            result = sync_node_capabilities_from_cai_peers(
                state_payload=state_payload,
                cai_url="http://127.0.0.1:52415",
                local_node_id="node-local",
                policy=self.policy,
                prune_after_seconds=999_999_999,
            )

        self.assertEqual(result.attempted_peers, 1)
        self.assertEqual(result.successful_peers, 1)
        self.assertEqual(result.imported_records, 1)
        self.assertEqual(
            urlopen_mock.call_args.args[0],
            "http://198.51.100.20:52415/v1/cai/node-capabilities",
        )
        records = list_node_capabilities(self.policy)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].node_id, "node-peer")
        self.assertTrue(records[0].relay_enabled)

    def test_sync_node_capabilities_records_peer_errors(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-local": {"apiHost": "127.0.0.1", "apiPort": 52415},
                "node-a": {"apiHost": "198.51.100.20", "apiPort": 52415},
                "node-b": {"apiHost": "198.51.100.21", "apiPort": 52415},
            }
        }
        peer_payload = self.peer_payload({
            "records": [
                {
                    "node_id": "node-b",
                    "updated_at": "2026-05-02T00:00:00+00:00",
                    "last_seen_at": "2026-05-02T00:00:00+00:00",
                    "relay_enabled": True,
                    "api_urls": ["http://198.51.100.21:52415"],
                }
            ]
        })

        def fake_urlopen(url: str, timeout: int = 0):
            if "198.51.100.20" in url:
                raise OSError("node capability peer offline")
            return _FakeResponse(peer_payload)

        with patch(
            "cai_compute_chain.node_capabilities.urlopen",
            side_effect=fake_urlopen,
        ):
            result = sync_node_capabilities_from_cai_peers(
                state_payload=state_payload,
                cai_url="http://127.0.0.1:52415",
                local_node_id="node-local",
                policy=self.policy,
                prune_after_seconds=999_999_999,
            )

        self.assertEqual(result.attempted_peers, 2)
        self.assertEqual(result.successful_peers, 1)
        self.assertEqual(result.failed_peers, 1)
        self.assertEqual(
            result.failed_peer_urls,
            ["http://198.51.100.20:52415/v1/cai/node-capabilities"],
        )
        self.assertEqual(result.peer_errors[0]["errorType"], "OSError")
        self.assertIn("node capability peer offline", result.peer_errors[0]["message"])
        self.assertEqual(result.imported_records, 1)

    def test_sync_node_capabilities_exchanges_peers_after_bootstrap(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-local": {"apiHost": "127.0.0.1", "apiPort": 52415},
                "node-bootstrap": {"apiHost": "198.51.100.20", "apiPort": 52415},
            }
        }
        bootstrap_url = "http://198.51.100.20:52415/v1/cai/node-capabilities"
        hidden_url = "http://203.0.113.42:52415/v1/cai/node-capabilities"
        bootstrap_payload = self.peer_payload({
            "records": [
                {
                    "node_id": "node-bootstrap",
                    "updated_at": "2026-05-02T00:00:00+00:00",
                    "last_seen_at": "2026-05-02T00:00:00+00:00",
                    "relay_enabled": True,
                    "api_urls": ["http://198.51.100.20:52415"],
                },
                {
                    "node_id": "node-hidden",
                    "updated_at": "2026-05-02T00:00:00+00:00",
                    "last_seen_at": "2026-05-02T00:00:00+00:00",
                    "relay_enabled": False,
                    "api_urls": ["http://203.0.113.42:52415"],
                },
            ]
        })
        hidden_payload = self.peer_payload({
            "records": [
                {
                    "node_id": "node-hidden",
                    "updated_at": "2026-05-02T00:01:00+00:00",
                    "last_seen_at": "2026-05-02T00:01:00+00:00",
                    "relay_enabled": True,
                    "worker_enabled": True,
                    "api_urls": ["http://203.0.113.42:52415"],
                }
            ]
        })

        def _urlopen(url: str, timeout: int):
            self.assertEqual(timeout, 5)
            if url == bootstrap_url:
                return _FakeResponse(bootstrap_payload)
            if url == hidden_url:
                return _FakeResponse(hidden_payload)
            raise AssertionError(f"unexpected peer url: {url}")

        with patch(
            "cai_compute_chain.node_capabilities.urlopen",
            side_effect=_urlopen,
        ):
            result = sync_node_capabilities_from_cai_peers(
                state_payload=state_payload,
                cai_url="http://127.0.0.1:52415",
                local_node_id="node-local",
                policy=self.policy,
                prune_after_seconds=999_999_999,
            )

        self.assertEqual(result.attempted_peers, 2)
        self.assertEqual(result.successful_peers, 2)
        self.assertEqual(result.imported_records, 3)
        self.assertEqual(result.peer_urls, [bootstrap_url, hidden_url])
        records_by_id = {
            record.node_id: record for record in list_node_capabilities(self.policy)
        }
        self.assertEqual(set(records_by_id), {"node-bootstrap", "node-hidden"})
        self.assertTrue(records_by_id["node-hidden"].worker_enabled)
        self.assertTrue(records_by_id["node-hidden"].relay_enabled)

    def test_node_capability_convergence_audit_recommends_repair(self) -> None:
        refresh_local_node_capabilities(
            state_payload={
                "nodeIdentities": {
                    "node-local": {"apiHost": "127.0.0.1", "apiPort": 52415},
                },
                "overlayAdvertisedPeers": {
                    "node-local": ["node-hidden"],
                },
            },
            cai_url="http://127.0.0.1:52415",
            local_node_id="node-local",
            policy=self.policy,
        )

        audit = node_capability_convergence_audit(
            state_payload={
                "nodeIdentities": {
                    "node-local": {"apiHost": "127.0.0.1", "apiPort": 52415},
                },
                "overlayAdvertisedPeers": {
                    "node-local": ["node-hidden"],
                },
            },
            local_node_id="node-local",
            policy=self.policy,
        )

        self.assertEqual(audit["status"], "repair_recommended")
        self.assertTrue(audit["repairRecommended"])
        self.assertIn("node-hidden", audit["missingFromStateNodeIds"])
        self.assertIn("node-hidden", audit["missingFromCapabilitiesNodeIds"])
        self.assertIn("request_full_sync", audit["repairActions"])
        self.assertIn("sync_node_capabilities", audit["repairActions"])

    def test_node_capability_convergence_audit_normalizes_overlay_advertised_peer_dict_entries(
        self,
    ) -> None:
        remote_peer = "/ip4/203.0.113.42/tcp/52416/p2p/node-hidden"

        audit = node_capability_convergence_audit(
            state_payload={
                "nodeIdentities": {
                    "node-local": {"apiHost": "127.0.0.1", "apiPort": 52415},
                },
                "overlayAdvertisedPeers": {
                    "node-local": [{"address": remote_peer}],
                },
            },
            local_node_id="node-local",
            policy=self.policy,
        )

        self.assertIn("node-hidden", audit["overlayReferenceNodeIds"])
        self.assertNotIn("{'address':", " ".join(audit["overlayReferenceNodeIds"]))
        self.assertIn("node-hidden", audit["missingFromStateNodeIds"])
        self.assertIn("request_full_sync", audit["repairActions"])

    def test_sync_node_capabilities_reports_convergence_repair(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-local": {"apiHost": "127.0.0.1", "apiPort": 52415},
                "node-bootstrap": {"apiHost": "198.51.100.20", "apiPort": 52415},
            },
            "overlayAdvertisedPeers": {
                "node-bootstrap": ["node-hidden"],
            },
        }
        bootstrap_url = "http://198.51.100.20:52415/v1/cai/node-capabilities"
        bootstrap_payload = self.peer_payload({
            "records": [
                {
                    "node_id": "node-bootstrap",
                    "updated_at": "2026-05-02T00:00:00+00:00",
                    "last_seen_at": "2026-05-02T00:00:00+00:00",
                    "api_urls": ["http://198.51.100.20:52415"],
                }
            ]
        })

        with patch(
            "cai_compute_chain.node_capabilities.urlopen",
            return_value=_FakeResponse(bootstrap_payload),
        ):
            result = sync_node_capabilities_from_cai_peers(
                state_payload=state_payload,
                cai_url="http://127.0.0.1:52415",
                local_node_id="node-local",
                policy=self.policy,
                prune_after_seconds=999_999_999,
            )

        self.assertEqual(result.convergence_status, "repair_recommended")
        self.assertTrue(result.convergence_repair_recommended)
        self.assertIn("request_full_sync", result.convergence_repair_actions)
        self.assertIn(
            "node-hidden",
            result.convergence_audit["missingFromCapabilitiesNodeIds"],
        )


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import base64
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.model import WalletPolicy
from cai_compute_chain.node_capabilities import (
    list_node_capabilities,
    list_verified_worker_node_ids,
    merge_remote_node_capabilities_payload,
)
from cai_compute_chain.peer_payload import add_peer_payload_metadata, sign_peer_payload
from cai_compute_chain.settlement import list_validator_evidence
from cai_compute_chain.validators import sync_validator_record
from cai_compute_chain.wallet_signing import (
    address_from_public_key_b64,
    encode_bytes,
    generate_mldsa65_keypair_b64,
    generate_signing_seed,
    hybrid_address_from_public_keys_b64,
    public_key_b64_from_seed,
)
from cai_compute_chain.worker_capability_attestations import (
    create_worker_capability_challenge,
    create_worker_capability_challenge_receipt,
    create_worker_capability_attestation,
    list_validator_attested_worker_node_ids,
    merge_remote_worker_capability_attestations_payload,
    record_worker_capability_attestation,
    record_worker_capability_challenge_failure_evidence,
    verify_worker_capability_attestation,
    verify_worker_capability_challenge,
    verify_worker_capability_challenge_receipt,
)


class WorkerCapabilityAttestationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo_patch = patch(
            "cai_compute_chain.wallet.repo_root",
            return_value=Path(self.tempdir.name),
        )
        self.repo_patch.start()
        self.policy = WalletPolicy(wallet_data_dirname=".tmp-worker-attestations")

    def tearDown(self) -> None:
        self.repo_patch.stop()
        self.tempdir.cleanup()

    def peer_payload(self, payload: dict) -> dict:
        return add_peer_payload_metadata(payload, policy=self.policy)

    def test_strict_verified_worker_requires_validator_attestation(self) -> None:
        record = self._import_signed_worker_capability()

        with patch.dict("os.environ", {"CAI_REQUIRE_SIGNED_PEER_PAYLOADS": "1"}):
            self.assertEqual(list_verified_worker_node_ids(self.policy), set())

        validator_seed = generate_signing_seed()
        validator_public_key_b64 = public_key_b64_from_seed(validator_seed)
        validator_id = address_from_public_key_b64(validator_public_key_b64)
        validator_signing_seed_b64 = base64.b64encode(validator_seed).decode("ascii")
        sync_validator_record(
            validator_id=validator_id,
            wallet_id="validator-wallet",
            address=validator_id,
            state="bonded",
            bonded_atomic=10_000,
            static_ip_confirmed=True,
            current_node_id="node-validator",
            policy=self.policy,
        )
        record_worker_capability_attestation(
            record,
            validator_id=validator_id,
            validator_public_key_b64=validator_public_key_b64,
            validator_signing_seed_b64=validator_signing_seed_b64,
            probe_result=self._challenge_probe_result(
                record,
                validator_id=validator_id,
                validator_public_key_b64=validator_public_key_b64,
                validator_signing_seed_b64=validator_signing_seed_b64,
            ),
            policy=self.policy,
        )

        with patch.dict("os.environ", {"CAI_REQUIRE_SIGNED_PEER_PAYLOADS": "1"}):
            self.assertEqual(
                list_verified_worker_node_ids(self.policy),
                {"node-worker"},
            )

    def test_attestation_is_invalidated_when_worker_changes_resource_claim(self) -> None:
        record = self._import_signed_worker_capability(vram_bytes=8_000)
        validator_seed = generate_signing_seed()
        validator_public_key_b64 = public_key_b64_from_seed(validator_seed)
        validator_id = address_from_public_key_b64(validator_public_key_b64)
        validator_signing_seed_b64 = encode_bytes(validator_seed)
        sync_validator_record(
            validator_id=validator_id,
            wallet_id="validator-wallet",
            address=validator_id,
            state="bonded",
            bonded_atomic=10_000,
            static_ip_confirmed=True,
            current_node_id="node-validator",
            policy=self.policy,
        )
        attestation = record_worker_capability_attestation(
            record,
            validator_id=validator_id,
            validator_public_key_b64=validator_public_key_b64,
            validator_signing_seed_b64=validator_signing_seed_b64,
            probe_result=self._challenge_probe_result(
                record,
                validator_id=validator_id,
                validator_public_key_b64=validator_public_key_b64,
                validator_signing_seed_b64=validator_signing_seed_b64,
            ),
            policy=self.policy,
        )
        self.assertTrue(
            verify_worker_capability_attestation(
                attestation,
                record=record,
                policy=self.policy,
            )[0]
        )

        changed = self._import_signed_worker_capability(
            vram_bytes=99_000,
            updated_at="2026-05-02T00:01:00+00:00",
        )

        self.assertNotEqual(record.resource_summary, changed.resource_summary)
        self.assertFalse(
            verify_worker_capability_attestation(
                attestation,
                record=changed,
                policy=self.policy,
            )[0]
        )
        self.assertEqual(
            list_validator_attested_worker_node_ids(
                records=[changed],
                policy=self.policy,
            ),
            set(),
        )

    def test_strict_attestation_without_challenge_is_not_verified(self) -> None:
        record = self._import_signed_worker_capability()
        validator_seed = generate_signing_seed()
        validator_public_key_b64 = public_key_b64_from_seed(validator_seed)
        validator_id = address_from_public_key_b64(validator_public_key_b64)
        sync_validator_record(
            validator_id=validator_id,
            wallet_id="validator-wallet",
            address=validator_id,
            state="bonded",
            bonded_atomic=10_000,
            static_ip_confirmed=True,
            current_node_id="node-validator",
            policy=self.policy,
        )
        attestation = record_worker_capability_attestation(
            record,
            validator_id=validator_id,
            validator_public_key_b64=validator_public_key_b64,
            validator_signing_seed_b64=encode_bytes(validator_seed),
            policy=self.policy,
        )

        with patch.dict("os.environ", {"CAI_REQUIRE_SIGNED_PEER_PAYLOADS": "1"}):
            ok, error = verify_worker_capability_attestation(
                attestation,
                record=record,
                policy=self.policy,
            )
            self.assertFalse(ok)
            self.assertIn("challenge proof is missing", error or "")
            self.assertEqual(
                list_validator_attested_worker_node_ids(
                    records=[record],
                    policy=self.policy,
                ),
                set(),
            )

    def test_remote_worker_capability_attestation_payload_imports_valid_record(self) -> None:
        record = self._import_signed_worker_capability()
        validator_seed = generate_signing_seed()
        validator_public_key_b64 = public_key_b64_from_seed(validator_seed)
        validator_id = address_from_public_key_b64(validator_public_key_b64)
        validator_signing_seed_b64 = encode_bytes(validator_seed)
        sync_validator_record(
            validator_id=validator_id,
            wallet_id="validator-wallet",
            address=validator_id,
            state="bonded",
            bonded_atomic=10_000,
            static_ip_confirmed=True,
            current_node_id="node-validator",
            policy=self.policy,
        )
        attestation = create_worker_capability_attestation(
            record,
            validator_id=validator_id,
            validator_public_key_b64=validator_public_key_b64,
            validator_signing_seed_b64=validator_signing_seed_b64,
            probe_result=self._challenge_probe_result(
                record,
                validator_id=validator_id,
                validator_public_key_b64=validator_public_key_b64,
                validator_signing_seed_b64=validator_signing_seed_b64,
            ),
        )

        imported = merge_remote_worker_capability_attestations_payload(
            self.peer_payload({"records": [attestation.__dict__]}),
            source_url="http://85.137.164.250:52415/v1/cai/worker-capability-attestations",
            policy=self.policy,
        )

        self.assertEqual(imported, 1)
        with patch.dict("os.environ", {"CAI_REQUIRE_SIGNED_PEER_PAYLOADS": "1"}):
            self.assertEqual(
                list_verified_worker_node_ids(self.policy),
                {"node-worker"},
            )

    def test_hybrid_worker_capability_challenge_and_attestation_verify(self) -> None:
        record = self._import_signed_worker_capability()
        validator_seed = generate_signing_seed()
        validator_public_key_b64 = public_key_b64_from_seed(validator_seed)
        validator_pq_public_key_b64, validator_pq_private_key_b64 = (
            generate_mldsa65_keypair_b64()
        )
        validator_id = hybrid_address_from_public_keys_b64(
            ed25519_public_key_b64=validator_public_key_b64,
            pq_public_key_b64=validator_pq_public_key_b64,
        )
        worker_pq_public_key_b64, worker_pq_private_key_b64 = (
            generate_mldsa65_keypair_b64()
        )
        worker_hybrid_address = hybrid_address_from_public_keys_b64(
            ed25519_public_key_b64=self.worker_public_key_b64,
            pq_public_key_b64=worker_pq_public_key_b64,
        )
        record.worker_reward_address = worker_hybrid_address
        record.node_public_key_address = worker_hybrid_address
        record.payload_public_key_address = worker_hybrid_address
        sync_validator_record(
            validator_id=validator_id,
            wallet_id="validator-wallet",
            address=validator_id,
            state="bonded",
            bonded_atomic=10_000,
            static_ip_confirmed=True,
            current_node_id="node-validator",
            policy=self.policy,
        )
        challenge = create_worker_capability_challenge(
            record,
            validator_id=validator_id,
            validator_public_key_b64=validator_public_key_b64,
            validator_signing_seed_b64=encode_bytes(validator_seed),
            validator_pq_public_key_b64=validator_pq_public_key_b64,
            validator_pq_private_key_b64=validator_pq_private_key_b64,
        )
        receipt = create_worker_capability_challenge_receipt(
            record,
            challenge=challenge,
            worker_public_key_b64=self.worker_public_key_b64,
            worker_signing_seed_b64=self.worker_signing_seed_b64,
            worker_pq_public_key_b64=worker_pq_public_key_b64,
            worker_pq_private_key_b64=worker_pq_private_key_b64,
        )
        attestation = create_worker_capability_attestation(
            record,
            validator_id=validator_id,
            validator_public_key_b64=validator_public_key_b64,
            validator_signing_seed_b64=encode_bytes(validator_seed),
            validator_pq_public_key_b64=validator_pq_public_key_b64,
            validator_pq_private_key_b64=validator_pq_private_key_b64,
            probe_result={
                "challenge": challenge,
                "challengeReceipt": receipt,
                "challengeVerified": True,
            },
        )

        self.assertEqual(
            challenge["signature_scheme"],
            "hybrid-ed25519-ml-dsa-65-v1",
        )
        self.assertEqual(
            receipt["signature_scheme"],
            "hybrid-ed25519-ml-dsa-65-v1",
        )
        ok, error = verify_worker_capability_challenge(
            challenge,
            policy=self.policy,
        )
        self.assertTrue(ok, error)
        ok, error = verify_worker_capability_challenge_receipt(
            receipt,
            challenge=challenge,
            record=record,
        )
        self.assertTrue(ok, error)
        ok, error = verify_worker_capability_attestation(
            attestation,
            record=record,
            policy=self.policy,
        )
        self.assertTrue(ok, error)

    def test_challenge_receipt_rejects_replay_for_different_nonce(self) -> None:
        record = self._import_signed_worker_capability()
        validator_seed = generate_signing_seed()
        validator_public_key_b64 = public_key_b64_from_seed(validator_seed)
        validator_id = address_from_public_key_b64(validator_public_key_b64)
        challenge = create_worker_capability_challenge(
            record,
            validator_id=validator_id,
            validator_public_key_b64=validator_public_key_b64,
            validator_signing_seed_b64=encode_bytes(validator_seed),
        )
        receipt = create_worker_capability_challenge_receipt(
            record,
            challenge=challenge,
            worker_public_key_b64=self.worker_public_key_b64,
            worker_signing_seed_b64=self.worker_signing_seed_b64,
        )
        replayed_challenge = {**challenge, "nonce": "different-nonce"}

        ok, error = verify_worker_capability_challenge_receipt(
            receipt,
            challenge=replayed_challenge,
            record=record,
        )

        self.assertFalse(ok)
        self.assertIn("nonce mismatch", error or "")

    def test_worker_capability_challenge_failure_records_validator_evidence(self) -> None:
        record = self._import_signed_worker_capability()
        validator_seed = generate_signing_seed()
        validator_public_key_b64 = public_key_b64_from_seed(validator_seed)
        validator_id = address_from_public_key_b64(validator_public_key_b64)
        validator_signing_seed_b64 = encode_bytes(validator_seed)
        sync_validator_record(
            validator_id=validator_id,
            wallet_id="validator-wallet",
            address=validator_id,
            state="bonded",
            bonded_atomic=10_000,
            static_ip_confirmed=True,
            current_node_id="node-validator",
            policy=self.policy,
        )
        attestation = record_worker_capability_attestation(
            record,
            validator_id=validator_id,
            validator_public_key_b64=validator_public_key_b64,
            validator_signing_seed_b64=validator_signing_seed_b64,
            probe_result=self._challenge_probe_result(
                record,
                validator_id=validator_id,
                validator_public_key_b64=validator_public_key_b64,
                validator_signing_seed_b64=validator_signing_seed_b64,
            ),
            policy=self.policy,
        )

        evidence = record_worker_capability_challenge_failure_evidence(
            attestation,
            reporter_validator_id=validator_id,
            policy=self.policy,
        )

        records = list_validator_evidence(policy=self.policy)
        self.assertEqual(records, [evidence])
        self.assertEqual(
            evidence.evidence_type,
            "worker_capability_challenge_failure",
        )
        self.assertEqual(evidence.attestation_id, attestation.attestation_id)
        self.assertGreater(evidence.slash_atomic, 0)

    def _challenge_probe_result(
        self,
        record,
        *,
        validator_id: str,
        validator_public_key_b64: str,
        validator_signing_seed_b64: str,
    ) -> dict[str, object]:
        challenge = create_worker_capability_challenge(
            record,
            validator_id=validator_id,
            validator_public_key_b64=validator_public_key_b64,
            validator_signing_seed_b64=validator_signing_seed_b64,
        )
        receipt = create_worker_capability_challenge_receipt(
            record,
            challenge=challenge,
            worker_public_key_b64=self.worker_public_key_b64,
            worker_signing_seed_b64=self.worker_signing_seed_b64,
        )
        return {
            "challenge": challenge,
            "challengeReceipt": receipt,
            "challengeVerified": True,
        }

    def _import_signed_worker_capability(
        self,
        *,
        vram_bytes: int = 8_000,
        updated_at: str = "2026-05-02T00:00:00+00:00",
    ):
        worker_seed = generate_signing_seed()
        worker_public_key_b64 = public_key_b64_from_seed(worker_seed)
        worker_address = address_from_public_key_b64(worker_public_key_b64)
        self.worker_signing_seed_b64 = encode_bytes(worker_seed)
        self.worker_public_key_b64 = worker_public_key_b64
        payload = sign_peer_payload(
            self.peer_payload({
                "records": [
                    {
                        "node_id": "node-worker",
                        "updated_at": updated_at,
                        "last_seen_at": "2026-05-02T00:00:00+00:00",
                        "worker_enabled": True,
                        "worker_reward_address": worker_address,
                        "worker_allowed_model_ids": [
                            "cai-network/Qwen3-0.6B-GGUF"
                        ],
                        "node_public_key_b64": worker_public_key_b64,
                        "api_urls": ["http://198.51.100.10:52415"],
                        "resource_summary": {"vramBytes": vram_bytes},
                    }
                ]
            }),
            public_key_b64=worker_public_key_b64,
            signing_seed_b64=encode_bytes(worker_seed),
            signer_address=worker_address,
        )
        merge_remote_node_capabilities_payload(
            payload,
            source_url="http://198.51.100.10:52415/v1/cai/node-capabilities",
            policy=self.policy,
        )
        return list_node_capabilities(self.policy)[0]

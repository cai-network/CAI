# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.peer_payload import (
    add_peer_payload_metadata,
    peer_payload_hybrid_signatures_required,
    peer_payload_signatures_required,
    policy_genesis_hash,
    sign_peer_payload,
    validate_peer_payload_network,
    verify_peer_payload_signature,
)
from cai_compute_chain.wallet_signing import (
    encode_bytes,
    generate_mldsa65_keypair_b64,
    generate_signing_seed,
    public_key_b64_from_seed,
)


class PeerPayloadTests(unittest.TestCase):
    def test_peer_payload_metadata_includes_genesis_hash(self) -> None:
        payload = add_peer_payload_metadata({"records": []})

        self.assertEqual(payload["network"], "mainnet")
        self.assertEqual(payload["chain_id"], "mainnet")
        self.assertEqual(payload["genesis_hash"], policy_genesis_hash())

    def test_peer_payload_network_rejects_missing_or_wrong_genesis_hash(self) -> None:
        with self.assertRaisesRegex(ValueError, "without genesis_hash"):
            validate_peer_payload_network(
                {
                    "network": "mainnet",
                    "chain_id": "mainnet",
                    "schema_version": 1,
                    "records": [],
                },
                payload_name="validator set",
            )

        with self.assertRaisesRegex(ValueError, "for genesis_hash"):
            validate_peer_payload_network(
                {
                    "network": "mainnet",
                    "chain_id": "mainnet",
                    "genesis_hash": "0" * 64,
                    "schema_version": 1,
                    "records": [],
                },
                payload_name="validator set",
            )

    def test_signed_peer_payload_verifies(self) -> None:
        signing_seed = generate_signing_seed()
        public_key_b64 = public_key_b64_from_seed(signing_seed)
        payload = add_peer_payload_metadata({"records": [{"id": "validator-a"}]})

        signed = sign_peer_payload(
            payload,
            public_key_b64=public_key_b64,
            signing_seed_b64=encode_bytes(signing_seed),
            signer_wallet_id="wallet-a",
            signer_address="validator-a",
            signed_at="2026-05-02T00:00:00+00:00",
        )

        ok, error = verify_peer_payload_signature(signed)
        self.assertTrue(ok, error)
        self.assertIsNone(error)
        self.assertEqual(signed["signature"]["signer_wallet_id"], "wallet-a")

    def test_signed_peer_payload_rejects_tampering(self) -> None:
        signing_seed = generate_signing_seed()
        public_key_b64 = public_key_b64_from_seed(signing_seed)
        payload = add_peer_payload_metadata({"records": [{"id": "validator-a"}]})
        signed = sign_peer_payload(
            payload,
            public_key_b64=public_key_b64,
            signing_seed_b64=encode_bytes(signing_seed),
        )

        tampered = dict(signed)
        tampered["records"] = [{"id": "validator-b"}]

        ok, error = verify_peer_payload_signature(tampered)
        self.assertFalse(ok)
        self.assertEqual(error, "peer payload signature is invalid")

    def test_hybrid_signed_peer_payload_verifies_and_rejects_tampering(self) -> None:
        signing_seed = generate_signing_seed()
        public_key_b64 = public_key_b64_from_seed(signing_seed)
        pq_public_key_b64, pq_private_key_b64 = generate_mldsa65_keypair_b64()
        payload = add_peer_payload_metadata({"records": [{"id": "validator-a"}]})

        signed = sign_peer_payload(
            payload,
            public_key_b64=public_key_b64,
            signing_seed_b64=encode_bytes(signing_seed),
            pq_public_key_b64=pq_public_key_b64,
            pq_private_key_b64=pq_private_key_b64,
        )

        ok, error = verify_peer_payload_signature(signed)
        self.assertTrue(ok, error)
        self.assertIsNone(error)
        self.assertEqual(
            signed["signature"]["scheme"],
            "cai-peer-payload-hybrid-ed25519-ml-dsa-65-v1",
        )

        tampered = dict(signed)
        tampered["records"] = [{"id": "validator-b"}]
        ok, error = verify_peer_payload_signature(tampered)
        self.assertFalse(ok)
        self.assertEqual(error, "peer payload hybrid signature is invalid")

    def test_unsigned_peer_payload_can_be_allowed_or_required(self) -> None:
        payload = add_peer_payload_metadata({"records": []})

        ok, error = verify_peer_payload_signature(payload)
        self.assertTrue(ok, error)

        ok, error = verify_peer_payload_signature(payload, require_signature=True)
        self.assertFalse(ok)
        self.assertEqual(error, "peer payload signature is missing")

    def test_hybrid_signature_can_be_required(self) -> None:
        signing_seed = generate_signing_seed()
        public_key_b64 = public_key_b64_from_seed(signing_seed)
        payload = add_peer_payload_metadata({"records": [{"id": "validator-a"}]})
        signed = sign_peer_payload(
            payload,
            public_key_b64=public_key_b64,
            signing_seed_b64=encode_bytes(signing_seed),
        )

        ok, error = verify_peer_payload_signature(
            signed,
            require_hybrid_signature=True,
        )

        self.assertFalse(ok)
        self.assertEqual(error, "peer payload requires a hybrid post-quantum signature")

    def test_signature_required_flag_parses_strict_values(self) -> None:
        self.assertTrue(peer_payload_signatures_required("1"))
        self.assertTrue(peer_payload_signatures_required("strict"))
        self.assertFalse(peer_payload_signatures_required(""))
        self.assertFalse(peer_payload_signatures_required("0"))
        self.assertTrue(peer_payload_hybrid_signatures_required("required"))
        self.assertFalse(peer_payload_hybrid_signatures_required("0"))


if __name__ == "__main__":
    unittest.main()

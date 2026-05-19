# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import base64
import hashlib
import json
import secrets
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

SIGNING_SCHEME_ED25519 = "ed25519-v1"
SIGNING_SCHEME_ML_DSA_65 = "ml-dsa-65-v1"
SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65 = "hybrid-ed25519-ml-dsa-65-v1"
ADDRESS_SCHEME_ED25519 = "ed25519-pubkey-sha256-v1"
ADDRESS_SCHEME_HYBRID_ED25519_ML_DSA_65 = "hybrid-ed25519-ml-dsa-65-sha256-v1"
ADDRESS_SCHEME_HYBRID_ED25519_ML_DSA_65_256 = (
    "hybrid-ed25519-ml-dsa-65-sha256-256-v1"
)
DEFAULT_WALLET_HYBRID_ADDRESS_SCHEME = ADDRESS_SCHEME_HYBRID_ED25519_ML_DSA_65_256
HYBRID_ADDRESS_SCHEMES = frozenset(
    {
        ADDRESS_SCHEME_HYBRID_ED25519_ML_DSA_65,
        ADDRESS_SCHEME_HYBRID_ED25519_ML_DSA_65_256,
    }
)
ADDRESS_SCHEME_FIXED_WALLET_ID_SHA256 = "fixed-wallet-id-sha256-v1"

try:
    from dilithium_py.ml_dsa import ML_DSA_65 as _ML_DSA_65
except Exception:  # pragma: no cover - availability is covered through the public helper.
    _ML_DSA_65 = None


def canonical_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def encode_bytes(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def decode_bytes(payload: str) -> bytes:
    return base64.b64decode(payload.encode("ascii"))


def generate_signing_seed() -> bytes:
    return secrets.token_bytes(32)


def derive_signing_seed_from_seed_phrase(seed_phrase: str) -> bytes:
    normalized = " ".join(seed_phrase.strip().lower().split())
    return hashlib.sha256(f"cai-wallet-ed25519-v1:{normalized}".encode("utf-8")).digest()


def derive_pq_signing_seed_from_seed_phrase(seed_phrase: str) -> bytes:
    normalized = " ".join(seed_phrase.strip().lower().split())
    return hashlib.sha256(f"cai-wallet-ml-dsa-65-v1:{normalized}".encode("utf-8")).digest()


def public_key_b64_from_seed(signing_seed: bytes) -> str:
    private_key = Ed25519PrivateKey.from_private_bytes(signing_seed)
    public_key = private_key.public_key()
    return encode_bytes(public_key.public_bytes_raw())


def mldsa65_available() -> bool:
    return _ML_DSA_65 is not None


def generate_mldsa65_keypair_b64() -> tuple[str, str]:
    backend = _require_mldsa65_backend()
    public_key, private_key = backend.keygen()
    return encode_bytes(public_key), encode_bytes(private_key)


def mldsa65_keypair_b64_from_seed(signing_seed: bytes) -> tuple[str, str]:
    backend = _require_mldsa65_backend()
    public_key, private_key = backend.key_derive(signing_seed)
    return encode_bytes(public_key), encode_bytes(private_key)


def mldsa65_public_key_b64_from_private_key_b64(private_key_b64: str) -> str:
    backend = _require_mldsa65_backend()
    return encode_bytes(backend.pk_from_sk(decode_bytes(private_key_b64)))


def address_from_public_key_b64(public_key_b64: str) -> str:
    return hashlib.sha256(decode_bytes(public_key_b64)).hexdigest()[:32]


def hybrid_address_from_public_keys_b64(
    *,
    ed25519_public_key_b64: str,
    pq_public_key_b64: str,
    address_scheme: str = ADDRESS_SCHEME_HYBRID_ED25519_ML_DSA_65,
) -> str:
    normalized_scheme = str(address_scheme or "").strip()
    if normalized_scheme not in HYBRID_ADDRESS_SCHEMES:
        raise ValueError(f"Unsupported hybrid address scheme: {address_scheme}")
    payload = canonical_payload(
        {
            "address_scheme": normalized_scheme,
            "ed25519_public_key_b64": str(ed25519_public_key_b64 or "").strip(),
            "pq_public_key_b64": str(pq_public_key_b64 or "").strip(),
        }
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if normalized_scheme == ADDRESS_SCHEME_HYBRID_ED25519_ML_DSA_65:
        return digest[:32]
    return digest


def encrypt_signing_seed(
    *,
    signing_seed: bytes,
    password: str,
    wallet_id: str,
    address: str,
    kdf_rounds: int,
) -> tuple[str, str, str]:
    salt = secrets.token_bytes(16)
    nonce = secrets.token_bytes(12)
    key = _derive_encryption_key(password, salt=salt, rounds=kdf_rounds)
    associated_data = _associated_data(wallet_id=wallet_id, address=address)
    encrypted = AESGCM(key).encrypt(nonce, signing_seed, associated_data)
    return encode_bytes(encrypted), encode_bytes(salt), encode_bytes(nonce)


def decrypt_signing_seed(
    *,
    encrypted_private_key_b64: str,
    private_key_salt_b64: str,
    private_key_nonce_b64: str,
    password: str,
    wallet_id: str,
    address: str,
    kdf_rounds: int,
) -> bytes:
    key = _derive_encryption_key(
        password,
        salt=decode_bytes(private_key_salt_b64),
        rounds=kdf_rounds,
    )
    associated_data = _associated_data(wallet_id=wallet_id, address=address)
    return AESGCM(key).decrypt(
        decode_bytes(private_key_nonce_b64),
        decode_bytes(encrypted_private_key_b64),
        associated_data,
    )


def sign_payload_b64(signing_seed: bytes, payload: dict[str, Any]) -> str:
    private_key = Ed25519PrivateKey.from_private_bytes(signing_seed)
    signature = private_key.sign(canonical_payload(payload).encode("utf-8"))
    return encode_bytes(signature)


def sign_payload_mldsa65_b64(private_key_b64: str, payload: dict[str, Any]) -> str:
    backend = _require_mldsa65_backend()
    signature = backend.sign(
        decode_bytes(private_key_b64),
        canonical_payload(payload).encode("utf-8"),
        deterministic=True,
    )
    return encode_bytes(signature)


def verify_payload_signature(
    *,
    public_key_b64: str,
    signature_b64: str,
    payload: dict[str, Any],
) -> bool:
    try:
        public_key = Ed25519PublicKey.from_public_bytes(decode_bytes(public_key_b64))
        public_key.verify(
            decode_bytes(signature_b64),
            canonical_payload(payload).encode("utf-8"),
        )
        return True
    except Exception:
        return False


def verify_payload_mldsa65_signature(
    *,
    public_key_b64: str,
    signature_b64: str,
    payload: dict[str, Any],
) -> bool:
    try:
        backend = _require_mldsa65_backend()
        return bool(
            backend.verify(
                decode_bytes(public_key_b64),
                canonical_payload(payload).encode("utf-8"),
                decode_bytes(signature_b64),
            )
        )
    except Exception:
        return False


def verify_hybrid_payload_signature(
    *,
    ed25519_public_key_b64: str,
    ed25519_signature_b64: str,
    pq_public_key_b64: str,
    pq_signature_b64: str,
    payload: dict[str, Any],
) -> bool:
    return verify_payload_signature(
        public_key_b64=ed25519_public_key_b64,
        signature_b64=ed25519_signature_b64,
        payload=payload,
    ) and verify_payload_mldsa65_signature(
        public_key_b64=pq_public_key_b64,
        signature_b64=pq_signature_b64,
        payload=payload,
    )


def _derive_encryption_key(password: str, *, salt: bytes, rounds: int) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        rounds,
        dklen=32,
    )


def _associated_data(*, wallet_id: str, address: str) -> bytes:
    return canonical_payload(
        {
            "address": str(address).strip().lower(),
            "wallet_id": str(wallet_id),
        }
    ).encode("utf-8")


def _require_mldsa65_backend():
    if _ML_DSA_65 is None:
        raise RuntimeError(
            "ML-DSA-65 backend is not available. Install dilithium-py>=1.4.0."
        )
    return _ML_DSA_65

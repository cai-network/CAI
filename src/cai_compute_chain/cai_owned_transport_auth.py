# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .cai_owned_transport_common import clean_node_ids as _clean_node_ids
from .cai_owned_transport_storage import record_cai_owned_transport_payload_replay
from .model import WalletPolicy
from .peer_payload import (
    peer_payload_signatures_required,
    sign_peer_payload,
    verify_peer_payload_signature,
)
from .wallet_signing import address_from_public_key_b64


def sign_cai_owned_transport_payload(
    payload: dict[str, Any],
    *,
    public_key_b64: str,
    signing_seed_b64: str,
    signer_node_id: str,
    signer_wallet_id: str | None = None,
    signer_address: str | None = None,
    signed_at: str | None = None,
) -> dict[str, Any]:
    signed_payload = dict(payload)
    clean_node_id = str(signer_node_id or "").strip()
    if not clean_node_id:
        raise ValueError("CAI-owned transport signer node id is required.")
    signed_payload["signerNodeId"] = clean_node_id
    return sign_peer_payload(
        signed_payload,
        public_key_b64=public_key_b64,
        signing_seed_b64=signing_seed_b64,
        signer_wallet_id=signer_wallet_id,
        signer_address=signer_address,
        signed_at=signed_at,
    )


def sign_cai_owned_transport_session_offer(
    offer: dict[str, Any],
    *,
    public_key_b64: str,
    signing_seed_b64: str,
    signer_node_id: str | None = None,
    signer_wallet_id: str | None = None,
    signer_address: str | None = None,
    signed_at: str | None = None,
) -> dict[str, Any]:
    return sign_cai_owned_transport_payload(
        offer,
        public_key_b64=public_key_b64,
        signing_seed_b64=signing_seed_b64,
        signer_node_id=signer_node_id or str(offer.get("sourceNodeId") or ""),
        signer_wallet_id=signer_wallet_id,
        signer_address=signer_address,
        signed_at=signed_at,
    )


def sign_cai_owned_transport_batch_envelope(
    envelope: dict[str, Any],
    *,
    public_key_b64: str,
    signing_seed_b64: str,
    signer_node_id: str | None = None,
    signer_wallet_id: str | None = None,
    signer_address: str | None = None,
    signed_at: str | None = None,
) -> dict[str, Any]:
    return sign_cai_owned_transport_payload(
        envelope,
        public_key_b64=public_key_b64,
        signing_seed_b64=signing_seed_b64,
        signer_node_id=signer_node_id or str(envelope.get("sourceNodeId") or ""),
        signer_wallet_id=signer_wallet_id,
        signer_address=signer_address,
        signed_at=signed_at,
    )


def sign_cai_owned_transport_shard_receipt(
    receipt: dict[str, Any],
    *,
    public_key_b64: str,
    signing_seed_b64: str,
    signer_node_id: str | None = None,
    signer_wallet_id: str | None = None,
    signer_address: str | None = None,
    signed_at: str | None = None,
) -> dict[str, Any]:
    return sign_cai_owned_transport_payload(
        receipt,
        public_key_b64=public_key_b64,
        signing_seed_b64=signing_seed_b64,
        signer_node_id=signer_node_id or str(receipt.get("nodeId") or ""),
        signer_wallet_id=signer_wallet_id,
        signer_address=signer_address,
        signed_at=signed_at,
    )


def sign_cai_owned_transport_execution_proof(
    proof: dict[str, Any],
    *,
    public_key_b64: str,
    signing_seed_b64: str,
    signer_node_id: str | None = None,
    signer_wallet_id: str | None = None,
    signer_address: str | None = None,
    signed_at: str | None = None,
) -> dict[str, Any]:
    participants = _clean_node_ids(proof.get("participantNodeIds") or [])
    return sign_cai_owned_transport_payload(
        proof,
        public_key_b64=public_key_b64,
        signing_seed_b64=signing_seed_b64,
        signer_node_id=signer_node_id or (participants[0] if participants else ""),
        signer_wallet_id=signer_wallet_id,
        signer_address=signer_address,
        signed_at=signed_at,
    )


def cai_owned_transport_peer_signing_kwargs(
    signing_material: Mapping[str, Any] | None,
) -> dict[str, str | None] | None:
    if not isinstance(signing_material, Mapping):
        return None
    public_key_b64 = _trusted_identity_text(
        signing_material,
        "public_key_b64",
        "publicKeyB64",
        "node_public_key_b64",
        "nodePublicKeyB64",
    )
    signing_seed_b64 = _trusted_identity_text(
        signing_material,
        "signing_seed_b64",
        "signingSeedB64",
        "private_key_seed_b64",
        "privateKeySeedB64",
    )
    if not public_key_b64 or not signing_seed_b64:
        return None
    return {
        "public_key_b64": public_key_b64,
        "signing_seed_b64": signing_seed_b64,
        "signer_wallet_id": _trusted_identity_text(
            signing_material,
            "signer_wallet_id",
            "signerWalletId",
            "wallet_id",
            "walletId",
        ),
        "signer_address": _trusted_identity_text(
            signing_material,
            "signer_address",
            "signerAddress",
            "address",
            "walletAddress",
        ),
    }


def validate_cai_owned_transport_payload_signature(
    payload: dict[str, Any] | None,
    *,
    payload_name: str,
    expected_signer_node_id: str | None = None,
    allowed_signer_node_ids: Sequence[str] | None = None,
    require_signature: bool | None = None,
    trusted_signer_identities_by_node: Mapping[str, Any] | None = None,
    require_trusted_signer: bool = False,
    record_replay_cache: bool = False,
    replay_cache_policy: WalletPolicy | None = None,
    replay_cache_retention_seconds: float | int | None = None,
) -> tuple[bool, str | None]:
    if not isinstance(payload, dict):
        return False, f"{payload_name} payload is missing"
    required = (
        peer_payload_signatures_required(policy=replay_cache_policy)
        if require_signature is None
        else bool(require_signature)
    )
    valid, error = verify_peer_payload_signature(
        payload,
        payload_name=payload_name,
        require_signature=required or require_trusted_signer,
    )
    if not valid:
        return False, error
    signature = payload.get("signature")
    if not isinstance(signature, dict):
        if record_replay_cache:
            return (
                False,
                f"{payload_name} payload signature is required for replay cache",
            )
        return True, None
    signer_node_id = str(payload.get("signerNodeId") or "").strip()
    if not signer_node_id:
        return False, f"{payload_name} payload signer node id is missing"
    expected = str(expected_signer_node_id or "").strip()
    if expected and signer_node_id != expected:
        return False, f"{payload_name} payload signer node id does not match"
    allowed = set(_clean_node_ids(allowed_signer_node_ids or []))
    if allowed and signer_node_id not in allowed:
        return False, f"{payload_name} payload signer node id is not allowed"
    (
        trusted_found,
        trusted_public_key_b64,
        trusted_public_key_address,
        trusted_error,
    ) = _cai_owned_transport_trusted_signer_identity(
        trusted_signer_identities_by_node,
        signer_node_id,
    )
    if trusted_error:
        return False, f"{payload_name} payload {trusted_error}"
    if not trusted_found:
        if require_trusted_signer:
            return False, f"{payload_name} payload signer is not trusted for node"
    if trusted_found:
        signature_public_key_b64 = str(signature.get("public_key_b64") or "").strip()
        signature_public_key_address = str(
            signature.get("public_key_address") or ""
        ).strip().lower()
        if not signature_public_key_address and signature_public_key_b64:
            try:
                signature_public_key_address = address_from_public_key_b64(
                    signature_public_key_b64,
                )
            except Exception:
                return False, f"{payload_name} payload signature public key is invalid"
        if (
            trusted_public_key_b64
            and signature_public_key_b64
            and trusted_public_key_b64 != signature_public_key_b64
        ):
            return False, (
                f"{payload_name} payload signer public key is not trusted for node"
            )
        if (
            trusted_public_key_address
            and signature_public_key_address
            and trusted_public_key_address != signature_public_key_address
        ):
            return False, (
                f"{payload_name} payload signer public key address "
                "is not trusted for node"
            )
    if record_replay_cache:
        replay_valid, replay_error = record_cai_owned_transport_payload_replay(
            payload,
            payload_name=payload_name,
            policy=replay_cache_policy,
            retention_seconds=replay_cache_retention_seconds,
        )
        if not replay_valid:
            return False, replay_error
    return True, None


def _cai_owned_transport_trusted_signer_identity(
    trusted_signer_identities_by_node: Mapping[str, Any] | None,
    signer_node_id: str,
) -> tuple[bool, str | None, str | None, str | None]:
    if not isinstance(trusted_signer_identities_by_node, Mapping):
        return False, None, None, None
    trusted_identity = trusted_signer_identities_by_node.get(signer_node_id)
    if trusted_identity is None:
        return False, None, None, None

    public_key_b64: str | None = None
    public_key_address: str | None = None
    if isinstance(trusted_identity, str):
        normalized = trusted_identity.strip()
        if _looks_like_cai_public_key_address(normalized):
            public_key_address = normalized.lower()
        else:
            public_key_b64 = normalized or None
    elif isinstance(trusted_identity, Mapping):
        public_key_b64 = _trusted_identity_text(
            trusted_identity,
            "node_public_key_b64",
            "nodePublicKeyB64",
            "public_key_b64",
            "publicKeyB64",
            "signing_public_key_b64",
            "signingPublicKeyB64",
        )
        public_key_address = _trusted_identity_text(
            trusted_identity,
            "node_public_key_address",
            "nodePublicKeyAddress",
            "public_key_address",
            "publicKeyAddress",
            "signing_public_key_address",
            "signingPublicKeyAddress",
        )
    else:
        public_key_b64 = _trusted_identity_attr_text(
            trusted_identity,
            "node_public_key_b64",
            "public_key_b64",
            "signing_public_key_b64",
        )
        public_key_address = _trusted_identity_attr_text(
            trusted_identity,
            "node_public_key_address",
            "public_key_address",
            "signing_public_key_address",
        )

    public_key_address = str(public_key_address or "").strip().lower() or None
    if public_key_b64:
        try:
            derived_address = address_from_public_key_b64(public_key_b64)
        except Exception:
            return True, None, None, "trusted signer public key is invalid"
        if public_key_address and public_key_address != derived_address:
            return True, None, None, "trusted signer public key address mismatch"
        public_key_address = derived_address
    if not public_key_b64 and not public_key_address:
        return True, None, None, "trusted signer identity is missing public key"
    return True, public_key_b64, public_key_address, None


def _trusted_identity_text(identity: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        raw = identity.get(key)
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    return None


def _trusted_identity_attr_text(identity: Any, *attrs: str) -> str | None:
    for attr in attrs:
        raw = getattr(identity, attr, None)
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    return None


def _looks_like_cai_public_key_address(value: str) -> bool:
    normalized = str(value or "").strip().lower()
    return len(normalized) == 32 and all(
        character in "0123456789abcdef" for character in normalized
    )

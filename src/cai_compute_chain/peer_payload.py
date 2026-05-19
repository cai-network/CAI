# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from datetime import UTC, datetime
import os
from typing import Any

from .model import MoneyPolicy, WalletPolicy, resolve_active_chain_network
from .wallet_signing import (
    SIGNING_SCHEME_ML_DSA_65,
    address_from_public_key_b64,
    decode_bytes,
    hybrid_address_from_public_keys_b64,
    sign_payload_mldsa65_b64,
    sign_payload_b64,
    verify_hybrid_payload_signature,
    verify_payload_signature,
)


PEER_PAYLOAD_SCHEMA_VERSION = 1
PEER_PAYLOAD_SIGNATURE_SCHEME_ED25519 = "cai-peer-payload-ed25519-v1"
PEER_PAYLOAD_SIGNATURE_SCHEME_HYBRID = "cai-peer-payload-hybrid-ed25519-ml-dsa-65-v1"
PEER_PAYLOAD_SIGNATURE_SCHEME = PEER_PAYLOAD_SIGNATURE_SCHEME_ED25519
REQUIRE_SIGNED_PEER_PAYLOADS_ENV = "CAI_REQUIRE_SIGNED_PEER_PAYLOADS"
REQUIRE_HYBRID_PEER_PAYLOADS_ENV = "CAI_REQUIRE_HYBRID_PEER_PAYLOAD_SIGNATURES"


def policy_chain_id(policy: WalletPolicy | object | None = None) -> str:
    return resolve_active_chain_network(getattr(policy, "chain_network", None)).value


def policy_genesis_hash(policy: WalletPolicy | object | None = None) -> str:
    chain_network = resolve_active_chain_network(getattr(policy, "chain_network", None))
    from .chain import make_genesis_block

    return make_genesis_block(MoneyPolicy(chain_network=chain_network)).block_hash


def add_peer_payload_metadata(
    payload: dict[str, Any],
    *,
    policy: WalletPolicy | object | None = None,
) -> dict[str, Any]:
    chain_id = policy_chain_id(policy)
    payload_genesis_hash = _payload_genesis_hash(payload)
    return {
        **payload,
        "network": chain_id,
        "chain_id": chain_id,
        "genesis_hash": payload_genesis_hash or policy_genesis_hash(policy),
        "schema_version": PEER_PAYLOAD_SCHEMA_VERSION,
    }


def validate_peer_payload_network(
    payload: dict[str, Any],
    *,
    policy: WalletPolicy | object | None = None,
    payload_name: str = "peer",
) -> str | None:
    if not isinstance(payload, dict):
        return None

    raw_schema_version = payload.get("schema_version", payload.get("schemaVersion"))
    if raw_schema_version not in (None, ""):
        try:
            schema_version = int(raw_schema_version)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Refusing {payload_name} payload with invalid schema_version "
                f"'{raw_schema_version}'."
            ) from exc
        if schema_version != PEER_PAYLOAD_SCHEMA_VERSION:
            raise ValueError(
                f"Refusing {payload_name} payload with unsupported schema_version "
                f"'{schema_version}'. Expected {PEER_PAYLOAD_SCHEMA_VERSION}."
            )

    payload_chain_id = _metadata_value(payload, "chain_id", "chainId")
    payload_network = _metadata_value(
        payload,
        "network",
        "chain_network",
        "chainNetwork",
    )
    normalized_chain_id = payload_chain_id.lower()
    normalized_network = payload_network.lower()
    if (
        normalized_chain_id
        and normalized_network
        and normalized_chain_id != normalized_network
    ):
        raise ValueError(
            f"Refusing {payload_name} payload with mismatched chain_id "
            f"'{payload_chain_id}' and network '{payload_network}'."
        )

    local_network = policy_chain_id(policy)
    incoming_network = normalized_chain_id or normalized_network
    if not incoming_network:
        raise ValueError(f"Refusing {payload_name} payload without network metadata.")
    if incoming_network and incoming_network != local_network:
        raise ValueError(
            f"Refusing {payload_name} payload for network '{incoming_network}' "
            f"on '{local_network}'."
        )
    payload_genesis_hashes = _payload_genesis_hashes(payload)
    if not payload_genesis_hashes:
        raise ValueError(
            f"Refusing {payload_name} payload without genesis_hash for "
            f"network '{local_network}'."
        )
    if len(payload_genesis_hashes) > 1:
        raise ValueError(
            f"Refusing {payload_name} payload with conflicting genesis_hash values: "
            f"{', '.join(sorted(payload_genesis_hashes))}."
        )
    payload_genesis_hash = next(iter(payload_genesis_hashes))
    local_genesis_hash = policy_genesis_hash(policy)
    if payload_genesis_hash != local_genesis_hash:
        raise ValueError(
            f"Refusing {payload_name} payload for genesis_hash "
            f"'{payload_genesis_hash}' on '{local_genesis_hash}'."
        )
    return incoming_network or None


def sign_peer_payload(
    payload: dict[str, Any],
    *,
    public_key_b64: str,
    signing_seed_b64: str,
    pq_public_key_b64: str | None = None,
    pq_private_key_b64: str | None = None,
    signer_wallet_id: str | None = None,
    signer_address: str | None = None,
    signed_at: str | None = None,
) -> dict[str, Any]:
    signed_payload = dict(payload)
    normalized_public_key = str(public_key_b64 or "").strip()
    if not normalized_public_key:
        raise ValueError("Peer payload signer public key is required.")
    normalized_pq_public_key = str(pq_public_key_b64 or "").strip()
    normalized_pq_private_key = str(pq_private_key_b64 or "").strip()
    signing_body = peer_payload_signing_body(signed_payload)
    signature_b64 = sign_payload_b64(
        decode_bytes(str(signing_seed_b64 or "").strip()),
        signing_body,
    )
    signature_scheme = PEER_PAYLOAD_SIGNATURE_SCHEME_ED25519
    public_key_address = address_from_public_key_b64(normalized_public_key)
    pq_signature_b64 = None
    if normalized_pq_public_key or normalized_pq_private_key:
        if not normalized_pq_public_key or not normalized_pq_private_key:
            raise ValueError("Peer payload hybrid signature requires ML-DSA keypair.")
        signature_scheme = PEER_PAYLOAD_SIGNATURE_SCHEME_HYBRID
        public_key_address = hybrid_address_from_public_keys_b64(
            ed25519_public_key_b64=normalized_public_key,
            pq_public_key_b64=normalized_pq_public_key,
        )
        pq_signature_b64 = sign_payload_mldsa65_b64(
            normalized_pq_private_key,
            signing_body,
        )
    signed_payload["signature"] = {
        "scheme": signature_scheme,
        "public_key_b64": normalized_public_key,
        "public_key_address": public_key_address,
        "signature_b64": signature_b64,
        "signer_wallet_id": str(signer_wallet_id or "") or None,
        "signer_address": str(signer_address or "").strip().lower() or None,
        "signed_at": signed_at or datetime.now(tz=UTC).isoformat(),
    }
    if pq_signature_b64:
        signed_payload["signature"].update(
            {
                "pq_scheme": SIGNING_SCHEME_ML_DSA_65,
                "pq_public_key_b64": normalized_pq_public_key,
                "pq_signature_b64": pq_signature_b64,
            }
        )
    return signed_payload


def verify_peer_payload_signature(
    payload: dict[str, Any],
    *,
    payload_name: str = "peer",
    require_signature: bool = False,
    require_hybrid_signature: bool = False,
) -> tuple[bool, str | None]:
    if not isinstance(payload, dict):
        return True, None
    signature = payload.get("signature")
    if not isinstance(signature, dict):
        if require_signature:
            return False, f"{payload_name} payload signature is missing"
        return True, None

    scheme = str(signature.get("scheme") or "").strip()
    if scheme not in {
        PEER_PAYLOAD_SIGNATURE_SCHEME_ED25519,
        PEER_PAYLOAD_SIGNATURE_SCHEME_HYBRID,
    }:
        return False, (
            f"{payload_name} payload has unsupported signature scheme '{scheme}'."
        )
    if require_hybrid_signature and scheme != PEER_PAYLOAD_SIGNATURE_SCHEME_HYBRID:
        return False, f"{payload_name} payload requires a hybrid post-quantum signature"
    public_key_b64 = str(signature.get("public_key_b64") or "").strip()
    signature_b64 = str(signature.get("signature_b64") or "").strip()
    if not public_key_b64 or not signature_b64:
        return False, f"{payload_name} payload signature is incomplete"
    try:
        if scheme == PEER_PAYLOAD_SIGNATURE_SCHEME_HYBRID:
            pq_scheme = str(signature.get("pq_scheme") or "").strip()
            if pq_scheme != SIGNING_SCHEME_ML_DSA_65:
                return False, f"{payload_name} payload has unsupported PQ signature scheme"
            pq_public_key_b64 = str(signature.get("pq_public_key_b64") or "").strip()
            pq_signature_b64 = str(signature.get("pq_signature_b64") or "").strip()
            if not pq_public_key_b64 or not pq_signature_b64:
                return False, f"{payload_name} payload hybrid signature is incomplete"
            expected_public_key_address = hybrid_address_from_public_keys_b64(
                ed25519_public_key_b64=public_key_b64,
                pq_public_key_b64=pq_public_key_b64,
            )
        else:
            expected_public_key_address = address_from_public_key_b64(public_key_b64)
    except Exception:
        return False, f"{payload_name} payload signature public key is invalid"
    declared_public_key_address = str(
        signature.get("public_key_address") or ""
    ).strip().lower()
    if (
        declared_public_key_address
        and declared_public_key_address != expected_public_key_address
    ):
        return False, f"{payload_name} payload signature public key address mismatch"
    signing_body = peer_payload_signing_body(payload)
    if scheme == PEER_PAYLOAD_SIGNATURE_SCHEME_HYBRID:
        if not verify_hybrid_payload_signature(
            ed25519_public_key_b64=public_key_b64,
            ed25519_signature_b64=signature_b64,
            pq_public_key_b64=str(signature.get("pq_public_key_b64") or "").strip(),
            pq_signature_b64=str(signature.get("pq_signature_b64") or "").strip(),
            payload=signing_body,
        ):
            return False, f"{payload_name} payload hybrid signature is invalid"
        return True, None
    if not verify_payload_signature(
        public_key_b64=public_key_b64,
        signature_b64=signature_b64,
        payload=signing_body,
    ):
        return False, f"{payload_name} payload signature is invalid"
    return True, None


def peer_payload_signatures_required(
    value: str | None = None,
    *,
    policy: WalletPolicy | object | None = None,
) -> bool:
    policy_value = getattr(policy, "require_hybrid_peer_payload_signatures", None)
    if value is None and policy_value is not None and bool(policy_value):
        return True
    raw = str(
        value
        if value is not None
        else os.getenv(REQUIRE_SIGNED_PEER_PAYLOADS_ENV, "")
    ).strip().lower()
    return raw in {"1", "true", "yes", "on", "strict", "required"}


def peer_payload_hybrid_signatures_required(
    value: str | None = None,
    *,
    policy: WalletPolicy | object | None = None,
) -> bool:
    if value is not None:
        raw = str(value).strip().lower()
        return raw in {"1", "true", "yes", "on", "strict", "required"}
    policy_value = getattr(policy, "require_hybrid_peer_payload_signatures", None)
    if policy_value is not None:
        return bool(policy_value)
    raw = str(os.getenv(REQUIRE_HYBRID_PEER_PAYLOADS_ENV, "")).strip().lower()
    return raw in {"1", "true", "yes", "on", "strict", "required"}


def peer_payload_signing_body(payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body.pop("signature", None)
    return body


def _metadata_value(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return ""


def _payload_genesis_hash(payload: dict[str, Any]) -> str:
    direct = _metadata_value(
        payload,
        "genesis_hash",
        "genesisHash",
        "chain_genesis_hash",
        "chainGenesisHash",
    ).lower()
    if direct:
        return direct
    nested_chain = payload.get("chain")
    if isinstance(nested_chain, dict):
        nested = _payload_genesis_hash(nested_chain)
        if nested:
            return nested
    raw_blocks = payload.get("blocks")
    if isinstance(raw_blocks, list) and raw_blocks:
        genesis_block = _payload_genesis_block(raw_blocks)
        if genesis_block is not None:
            return str(genesis_block.get("block_hash") or "").strip().lower()
    return ""


def _payload_genesis_block(raw_blocks: list[Any]) -> dict[str, Any] | None:
    for raw in raw_blocks:
        if not isinstance(raw, dict):
            continue
        try:
            if int(raw.get("height") or 0) == 0:
                return raw
        except (TypeError, ValueError):
            continue
    first_block = raw_blocks[0]
    return first_block if isinstance(first_block, dict) else None


def _payload_genesis_hashes(payload: dict[str, Any]) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    values: set[str] = set()
    direct = _metadata_value(
        payload,
        "genesis_hash",
        "genesisHash",
        "chain_genesis_hash",
        "chainGenesisHash",
    ).lower()
    if direct:
        values.add(direct)
    nested_chain = payload.get("chain")
    if isinstance(nested_chain, dict):
        values.update(_payload_genesis_hashes(nested_chain))
    raw_blocks = payload.get("blocks")
    if isinstance(raw_blocks, list) and raw_blocks:
        genesis_block = _payload_genesis_block(raw_blocks)
        if genesis_block is not None:
            block_hash = str(genesis_block.get("block_hash") or "").strip().lower()
            if block_hash:
                values.add(block_hash)
    return values

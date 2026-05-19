# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import os
import secrets
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .model import ValidatorLifecycleState, WalletPolicy
from .peer_payload import (
    add_peer_payload_metadata,
    peer_payload_hybrid_signatures_required,
    peer_payload_signatures_required,
    validate_peer_payload_network,
    verify_peer_payload_signature,
)
from .validators import get_validator_record
from .wallet import data_root, normalize_address
from .wallet_signing import (
    SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65,
    SIGNING_SCHEME_ML_DSA_65,
    SIGNING_SCHEME_ED25519,
    address_from_public_key_b64,
    decode_bytes,
    hybrid_address_from_public_keys_b64,
    sign_payload_mldsa65_b64,
    sign_payload_b64,
    verify_hybrid_payload_signature,
    verify_payload_signature,
)


WORKER_CAPABILITY_ATTESTATION_SCHEMA_VERSION = 1
WORKER_CAPABILITY_ATTESTATION_TTL_SECONDS = 600
WORKER_CAPABILITY_CHALLENGE_PROTOCOL = "cai-worker-capability-challenge-v1"
WORKER_CAPABILITY_CHALLENGE_RECEIPT_PROTOCOL = (
    "cai-worker-capability-challenge-receipt-v1"
)
WORKER_CAPABILITY_CHALLENGE_TTL_SECONDS = 60
WORKER_CAPABILITY_CHALLENGE_DIFFICULTY = 2


@dataclass
class WorkerCapabilityAttestation:
    attestation_id: str
    created_at: str
    expires_at: str
    validator_id: str
    worker_node_id: str
    worker_reward_address: str | None
    worker_public_key_address: str | None
    capability_fingerprint: str
    accepted_model_ids: list[str] = field(default_factory=list)
    resource_summary: dict[str, Any] = field(default_factory=dict)
    readiness: dict[str, Any] = field(default_factory=dict)
    probe_result: dict[str, Any] = field(default_factory=dict)
    accepted: bool = True
    note: str | None = None
    signature_scheme: str = SIGNING_SCHEME_ED25519
    validator_public_key_b64: str | None = None
    validator_signature_b64: str | None = None
    validator_pq_public_key_b64: str | None = None
    validator_pq_signature_b64: str | None = None
    source: str = "local"
    source_url: str | None = None
    last_seen_at: str | None = None
    updated_at: str | None = None


def worker_capability_challenge_required(value: str | None = None) -> bool:
    raw = (
        value
        if value is not None
        else os.getenv("CAI_REQUIRE_WORKER_CAPABILITY_CHALLENGE")
    )
    if raw is not None and str(raw).strip():
        return str(raw).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
            "strict",
            "required",
        }
    return peer_payload_signatures_required(policy=WalletPolicy())


def worker_capability_attestation_file_path(
    policy: WalletPolicy | None = None,
) -> Path:
    active_policy = policy or WalletPolicy()
    file_name = getattr(
        active_policy,
        "worker_capability_attestation_file_name",
        "worker-capability-attestations.jsonl",
    )
    return data_root(active_policy) / file_name


def worker_capability_fingerprint_from_record(record: Any) -> str:
    payload = {
        "node_id": _record_text(record, "node_id"),
        "worker_enabled": bool(getattr(record, "worker_enabled", False)),
        "worker_reward_address": normalize_address(
            _record_text(record, "worker_reward_address")
        )
        if _record_text(record, "worker_reward_address")
        else None,
        "worker_public_key_address": _record_worker_public_key_address(record),
        "worker_allowed_model_ids": sorted(
            str(item).strip()
            for item in (getattr(record, "worker_allowed_model_ids", None) or [])
            if str(item).strip()
        ),
        "model_ids": sorted(
            str(item).strip()
            for item in (getattr(record, "model_ids", None) or [])
            if str(item).strip()
        ),
        "resource_summary": _stable_resource_summary_for_fingerprint(
            getattr(record, "resource_summary", None)
        ),
        "readiness": _safe_json_mapping(getattr(record, "readiness", None)),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_resource_summary_for_fingerprint(value: Any) -> dict[str, Any]:
    summary = _safe_json_mapping(value)
    stable: dict[str, Any] = {}
    for key, raw_value in summary.items():
        normalized_key = str(key).strip()
        lowered_key = normalized_key.replace("_", "").lower()
        if not normalized_key:
            continue
        if "available" in lowered_key or "free" in lowered_key:
            continue
        stable[normalized_key] = raw_value
    return stable


def _identity_address_from_public_keys(
    public_key_b64: str,
    pq_public_key_b64: str | None = None,
) -> str:
    normalized_public_key = str(public_key_b64 or "").strip()
    normalized_pq_public_key = str(pq_public_key_b64 or "").strip()
    if normalized_pq_public_key:
        return hybrid_address_from_public_keys_b64(
            ed25519_public_key_b64=normalized_public_key,
            pq_public_key_b64=normalized_pq_public_key,
        )
    return address_from_public_key_b64(normalized_public_key)


def _direct_signature_scheme(pq_public_key_b64: str | None, pq_private_key_b64: str | None) -> str:
    if str(pq_public_key_b64 or "").strip() or str(pq_private_key_b64 or "").strip():
        if not str(pq_public_key_b64 or "").strip() or not str(
            pq_private_key_b64 or ""
        ).strip():
            raise ValueError("Hybrid worker capability signature requires ML-DSA keypair.")
        return SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65
    return SIGNING_SCHEME_ED25519


def _verify_worker_capability_signature(
    *,
    signature_scheme: str,
    public_key_b64: str,
    signature_b64: str,
    pq_public_key_b64: str,
    pq_signature_b64: str,
    payload: dict[str, Any],
) -> bool:
    if signature_scheme == SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65:
        return verify_hybrid_payload_signature(
            ed25519_public_key_b64=public_key_b64,
            ed25519_signature_b64=signature_b64,
            pq_public_key_b64=pq_public_key_b64,
            pq_signature_b64=pq_signature_b64,
            payload=payload,
        )
    return verify_payload_signature(
        public_key_b64=public_key_b64,
        signature_b64=signature_b64,
        payload=payload,
    )


def create_worker_capability_challenge(
    record: Any,
    *,
    validator_id: str,
    validator_public_key_b64: str,
    validator_signing_seed_b64: str,
    validator_pq_public_key_b64: str | None = None,
    validator_pq_private_key_b64: str | None = None,
    ttl_seconds: int = WORKER_CAPABILITY_CHALLENGE_TTL_SECONDS,
    difficulty: int = WORKER_CAPABILITY_CHALLENGE_DIFFICULTY,
) -> dict[str, Any]:
    now = datetime.now(tz=UTC)
    normalized_validator_id = normalize_address(validator_id)
    normalized_pq_public_key = str(validator_pq_public_key_b64 or "").strip()
    normalized_pq_private_key = str(validator_pq_private_key_b64 or "").strip()
    signature_scheme = _direct_signature_scheme(
        normalized_pq_public_key,
        normalized_pq_private_key,
    )
    public_key_address = _identity_address_from_public_keys(
        validator_public_key_b64,
        normalized_pq_public_key,
    )
    if public_key_address != normalized_validator_id:
        raise ValueError("Validator public key does not match validator id.")
    challenge = {
        "schema_version": WORKER_CAPABILITY_ATTESTATION_SCHEMA_VERSION,
        "protocol": WORKER_CAPABILITY_CHALLENGE_PROTOCOL,
        "challenge_id": secrets.token_hex(12),
        "created_at": now.isoformat(),
        "expires_at": (
            now + timedelta(seconds=max(1, int(ttl_seconds)))
        ).isoformat(),
        "validator_id": normalized_validator_id,
        "worker_node_id": _record_text(record, "node_id"),
        "capability_fingerprint": worker_capability_fingerprint_from_record(record),
        "nonce": secrets.token_hex(24),
        "difficulty": _normalize_challenge_difficulty(difficulty),
        "signature_scheme": signature_scheme,
        "validator_public_key_b64": str(validator_public_key_b64 or "").strip(),
    }
    if signature_scheme == SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65:
        challenge["validator_pq_public_key_b64"] = normalized_pq_public_key
    challenge["validator_signature_b64"] = sign_payload_b64(
        decode_bytes(str(validator_signing_seed_b64 or "").strip()),
        worker_capability_challenge_signing_body(challenge),
    )
    if signature_scheme == SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65:
        challenge["validator_pq_signature_b64"] = sign_payload_mldsa65_b64(
            normalized_pq_private_key,
            worker_capability_challenge_signing_body(challenge),
        )
    return challenge


def verify_worker_capability_challenge(
    challenge: dict[str, Any],
    *,
    policy: WalletPolicy | None = None,
    require_bonded_validator: bool = True,
    enforce_expiry: bool = True,
) -> tuple[bool, str | None]:
    if not isinstance(challenge, dict):
        return False, "worker capability challenge is missing"
    try:
        schema_version = int(
            challenge.get("schema_version") or challenge.get("schemaVersion") or 0
        )
    except (TypeError, ValueError):
        schema_version = 0
    if schema_version != WORKER_CAPABILITY_ATTESTATION_SCHEMA_VERSION:
        return False, "worker capability challenge schema is unsupported"
    if str(challenge.get("protocol") or "").strip() != WORKER_CAPABILITY_CHALLENGE_PROTOCOL:
        return False, "worker capability challenge protocol is invalid"
    expires_at = _parse_iso_datetime(
        challenge.get("expires_at") or challenge.get("expiresAt")
    )
    if expires_at is None:
        return False, "worker capability challenge expiry is missing"
    if enforce_expiry and expires_at <= datetime.now(tz=UTC):
        return False, "worker capability challenge is expired"
    validator_public_key_b64 = str(
        challenge.get("validator_public_key_b64")
        or challenge.get("validatorPublicKeyB64")
        or ""
    ).strip()
    validator_signature_b64 = str(
        challenge.get("validator_signature_b64")
        or challenge.get("validatorSignatureB64")
        or ""
    ).strip()
    signature_scheme = str(
        challenge.get("signature_scheme")
        or challenge.get("signatureScheme")
        or SIGNING_SCHEME_ED25519
    ).strip()
    if signature_scheme not in {
        SIGNING_SCHEME_ED25519,
        SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65,
    }:
        return False, "worker capability challenge signature scheme is unsupported"
    validator_pq_public_key_b64 = str(
        challenge.get("validator_pq_public_key_b64")
        or challenge.get("validatorPqPublicKeyB64")
        or ""
    ).strip()
    validator_pq_signature_b64 = str(
        challenge.get("validator_pq_signature_b64")
        or challenge.get("validatorPqSignatureB64")
        or ""
    ).strip()
    if not validator_public_key_b64 or not validator_signature_b64:
        return False, "worker capability challenge signature is missing"
    if signature_scheme == SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65 and (
        not validator_pq_public_key_b64 or not validator_pq_signature_b64
    ):
        return False, "worker capability challenge PQ signature is missing"
    try:
        validator_address = _identity_address_from_public_keys(
            validator_public_key_b64,
            validator_pq_public_key_b64
            if signature_scheme == SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65
            else None,
        )
    except Exception:
        return False, "worker capability challenge validator public key is invalid"
    if validator_address != normalize_address(str(challenge.get("validator_id") or "")):
        return False, "worker capability challenge signer does not match validator id"
    if require_bonded_validator:
        validator = get_validator_record(validator_address, policy)
        if (
            validator is None
            or validator.state != ValidatorLifecycleState.BONDED
            or int(validator.bonded_atomic or 0) <= 0
        ):
            return False, "worker capability challenge signer is not a bonded validator"
    if not _verify_worker_capability_signature(
        signature_scheme=signature_scheme,
        public_key_b64=validator_public_key_b64,
        signature_b64=validator_signature_b64,
        pq_public_key_b64=validator_pq_public_key_b64,
        pq_signature_b64=validator_pq_signature_b64,
        payload=worker_capability_challenge_signing_body(challenge),
    ):
        return False, "worker capability challenge signature is invalid"
    return True, None


def create_worker_capability_challenge_receipt(
    record: Any,
    *,
    challenge: dict[str, Any],
    worker_public_key_b64: str,
    worker_signing_seed_b64: str,
    worker_pq_public_key_b64: str | None = None,
    worker_pq_private_key_b64: str | None = None,
) -> dict[str, Any]:
    challenge_id = str(
        challenge.get("challenge_id") or challenge.get("challengeId") or ""
    ).strip()
    nonce = str(challenge.get("nonce") or "").strip()
    difficulty = _normalize_challenge_difficulty(challenge.get("difficulty"))
    capability_fingerprint = worker_capability_fingerprint_from_record(record)
    counter, proof_hash = _solve_challenge_hash(
        challenge_id=challenge_id,
        nonce=nonce,
        worker_node_id=_record_text(record, "node_id"),
        capability_fingerprint=capability_fingerprint,
        difficulty=difficulty,
    )
    normalized_worker_pq_public_key = str(worker_pq_public_key_b64 or "").strip()
    normalized_worker_pq_private_key = str(worker_pq_private_key_b64 or "").strip()
    signature_scheme = _direct_signature_scheme(
        normalized_worker_pq_public_key,
        normalized_worker_pq_private_key,
    )
    worker_public_key_address = _identity_address_from_public_keys(
        worker_public_key_b64,
        normalized_worker_pq_public_key,
    )
    receipt = {
        "schema_version": WORKER_CAPABILITY_ATTESTATION_SCHEMA_VERSION,
        "protocol": WORKER_CAPABILITY_CHALLENGE_RECEIPT_PROTOCOL,
        "challenge_id": challenge_id,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "worker_node_id": _record_text(record, "node_id"),
        "worker_reward_address": (
            normalize_address(_record_text(record, "worker_reward_address"))
            if _record_text(record, "worker_reward_address")
            else None
        ),
        "worker_public_key_address": worker_public_key_address,
        "capability_fingerprint": capability_fingerprint,
        "nonce": nonce,
        "difficulty": difficulty,
        "proof_counter": counter,
        "proof_hash": proof_hash,
        "accepted_model_ids": sorted(
            str(item).strip()
            for item in (getattr(record, "worker_allowed_model_ids", None) or [])
            if str(item).strip()
        ),
        "resource_summary": _safe_json_mapping(
            getattr(record, "resource_summary", None)
        ),
        "readiness": _safe_json_mapping(getattr(record, "readiness", None)),
        "signature_scheme": signature_scheme,
        "worker_public_key_b64": str(worker_public_key_b64 or "").strip(),
    }
    if signature_scheme == SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65:
        receipt["worker_pq_public_key_b64"] = normalized_worker_pq_public_key
    receipt["worker_signature_b64"] = sign_payload_b64(
        decode_bytes(str(worker_signing_seed_b64 or "").strip()),
        worker_capability_challenge_receipt_signing_body(receipt),
    )
    if signature_scheme == SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65:
        receipt["worker_pq_signature_b64"] = sign_payload_mldsa65_b64(
            normalized_worker_pq_private_key,
            worker_capability_challenge_receipt_signing_body(receipt),
        )
    return receipt


def verify_worker_capability_challenge_receipt(
    receipt: dict[str, Any],
    *,
    challenge: dict[str, Any] | None = None,
    record: Any | None = None,
) -> tuple[bool, str | None]:
    if not isinstance(receipt, dict):
        return False, "worker capability challenge receipt is missing"
    try:
        schema_version = int(
            receipt.get("schema_version") or receipt.get("schemaVersion") or 0
        )
    except (TypeError, ValueError):
        schema_version = 0
    if schema_version != WORKER_CAPABILITY_ATTESTATION_SCHEMA_VERSION:
        return False, "worker capability challenge receipt schema is unsupported"
    if str(receipt.get("protocol") or "").strip() != WORKER_CAPABILITY_CHALLENGE_RECEIPT_PROTOCOL:
        return False, "worker capability challenge receipt protocol is invalid"
    worker_public_key_b64 = str(
        receipt.get("worker_public_key_b64")
        or receipt.get("workerPublicKeyB64")
        or ""
    ).strip()
    worker_signature_b64 = str(
        receipt.get("worker_signature_b64")
        or receipt.get("workerSignatureB64")
        or ""
    ).strip()
    signature_scheme = str(
        receipt.get("signature_scheme")
        or receipt.get("signatureScheme")
        or SIGNING_SCHEME_ED25519
    ).strip()
    if signature_scheme not in {
        SIGNING_SCHEME_ED25519,
        SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65,
    }:
        return False, "worker capability challenge receipt signature scheme is unsupported"
    worker_pq_public_key_b64 = str(
        receipt.get("worker_pq_public_key_b64")
        or receipt.get("workerPqPublicKeyB64")
        or ""
    ).strip()
    worker_pq_signature_b64 = str(
        receipt.get("worker_pq_signature_b64")
        or receipt.get("workerPqSignatureB64")
        or ""
    ).strip()
    if not worker_public_key_b64 or not worker_signature_b64:
        return False, "worker capability challenge receipt signature is missing"
    if signature_scheme == SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65 and (
        not worker_pq_public_key_b64 or not worker_pq_signature_b64
    ):
        return False, "worker capability challenge receipt PQ signature is missing"
    try:
        worker_public_key_address = _identity_address_from_public_keys(
            worker_public_key_b64,
            worker_pq_public_key_b64
            if signature_scheme == SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65
            else None,
        )
    except Exception:
        return False, "worker capability challenge receipt worker public key is invalid"
    if worker_public_key_address != normalize_address(
        str(receipt.get("worker_public_key_address") or "")
    ):
        return False, "worker capability challenge receipt worker key mismatch"
    if not _verify_worker_capability_signature(
        signature_scheme=signature_scheme,
        public_key_b64=worker_public_key_b64,
        signature_b64=worker_signature_b64,
        pq_public_key_b64=worker_pq_public_key_b64,
        pq_signature_b64=worker_pq_signature_b64,
        payload=worker_capability_challenge_receipt_signing_body(receipt),
    ):
        return False, "worker capability challenge receipt signature is invalid"

    difficulty = _normalize_challenge_difficulty(receipt.get("difficulty"))
    proof_hash = str(receipt.get("proof_hash") or receipt.get("proofHash") or "")
    try:
        proof_counter = int(
            receipt.get("proof_counter") or receipt.get("proofCounter") or 0
        )
    except (TypeError, ValueError):
        return False, "worker capability challenge proof counter is invalid"
    expected_hash = _challenge_hash(
        challenge_id=str(receipt.get("challenge_id") or ""),
        nonce=str(receipt.get("nonce") or ""),
        worker_node_id=str(receipt.get("worker_node_id") or ""),
        capability_fingerprint=str(receipt.get("capability_fingerprint") or ""),
        counter=proof_counter,
    )
    if proof_hash != expected_hash:
        return False, "worker capability challenge proof hash is invalid"
    if not proof_hash.startswith("0" * difficulty):
        return False, "worker capability challenge proof difficulty is invalid"

    if challenge is not None:
        challenge_created_at = _parse_iso_datetime(
            challenge.get("created_at") or challenge.get("createdAt")
        )
        challenge_expires_at = _parse_iso_datetime(
            challenge.get("expires_at") or challenge.get("expiresAt")
        )
        receipt_created_at = _parse_iso_datetime(
            receipt.get("created_at") or receipt.get("createdAt")
        )
        if challenge_created_at is None or challenge_expires_at is None:
            return False, "worker capability challenge timing is invalid"
        if receipt_created_at is None:
            return False, "worker capability challenge receipt timing is invalid"
        if receipt_created_at < challenge_created_at:
            return False, "worker capability challenge receipt predates challenge"
        if receipt_created_at > challenge_expires_at:
            return False, "worker capability challenge receipt is expired"
        if difficulty != _normalize_challenge_difficulty(challenge.get("difficulty")):
            return False, "worker capability challenge receipt difficulty mismatch"
        if str(receipt.get("challenge_id") or "") != str(
            challenge.get("challenge_id") or challenge.get("challengeId") or ""
        ):
            return False, "worker capability challenge receipt id mismatch"
        if str(receipt.get("nonce") or "") != str(challenge.get("nonce") or ""):
            return False, "worker capability challenge receipt nonce mismatch"
        if str(receipt.get("worker_node_id") or "") != str(
            challenge.get("worker_node_id") or challenge.get("workerNodeId") or ""
        ):
            return False, "worker capability challenge receipt node mismatch"
        if str(receipt.get("capability_fingerprint") or "") != str(
            challenge.get("capability_fingerprint")
            or challenge.get("capabilityFingerprint")
            or ""
        ):
            return False, "worker capability challenge receipt fingerprint mismatch"

    if record is not None:
        if str(receipt.get("worker_node_id") or "") != _record_text(record, "node_id"):
            return False, "worker capability challenge receipt record node mismatch"
        if normalize_address(str(receipt.get("worker_reward_address") or "")) != (
            normalize_address(_record_text(record, "worker_reward_address"))
            if _record_text(record, "worker_reward_address")
            else ""
        ):
            return False, "worker capability challenge receipt reward mismatch"
        if str(receipt.get("capability_fingerprint") or "") != (
            worker_capability_fingerprint_from_record(record)
        ):
            return False, "worker capability challenge receipt record fingerprint mismatch"
        public_key_address = _record_worker_public_key_address(record)
        if public_key_address != normalize_address(
            str(receipt.get("worker_public_key_address") or "")
        ):
            return False, "worker capability challenge receipt worker key mismatch"
    return True, None


def create_worker_capability_attestation(
    record: Any,
    *,
    validator_id: str,
    validator_public_key_b64: str,
    validator_signing_seed_b64: str,
    validator_pq_public_key_b64: str | None = None,
    validator_pq_private_key_b64: str | None = None,
    ttl_seconds: int = WORKER_CAPABILITY_ATTESTATION_TTL_SECONDS,
    accepted: bool = True,
    note: str | None = None,
    probe_result: dict[str, Any] | None = None,
) -> WorkerCapabilityAttestation:
    now = datetime.now(tz=UTC)
    normalized_validator_id = normalize_address(validator_id)
    normalized_pq_public_key = str(validator_pq_public_key_b64 or "").strip()
    normalized_pq_private_key = str(validator_pq_private_key_b64 or "").strip()
    signature_scheme = _direct_signature_scheme(
        normalized_pq_public_key,
        normalized_pq_private_key,
    )
    public_key_address = _identity_address_from_public_keys(
        validator_public_key_b64,
        normalized_pq_public_key,
    )
    if public_key_address != normalized_validator_id:
        raise ValueError("Validator public key does not match validator id.")

    attestation = WorkerCapabilityAttestation(
        attestation_id=secrets.token_hex(12),
        created_at=now.isoformat(),
        expires_at=(now + timedelta(seconds=max(1, int(ttl_seconds)))).isoformat(),
        validator_id=normalized_validator_id,
        worker_node_id=_record_text(record, "node_id"),
        worker_reward_address=(
            normalize_address(_record_text(record, "worker_reward_address"))
            if _record_text(record, "worker_reward_address")
            else None
        ),
        worker_public_key_address=_record_worker_public_key_address(record),
        capability_fingerprint=worker_capability_fingerprint_from_record(record),
        accepted_model_ids=sorted(
            str(item).strip()
            for item in (getattr(record, "worker_allowed_model_ids", None) or [])
            if str(item).strip()
        ),
        resource_summary=_safe_json_mapping(getattr(record, "resource_summary", None)),
        readiness=_safe_json_mapping(getattr(record, "readiness", None)),
        probe_result=_safe_json_mapping(probe_result),
        accepted=bool(accepted),
        note=note,
        signature_scheme=signature_scheme,
        validator_public_key_b64=str(validator_public_key_b64 or "").strip(),
        validator_pq_public_key_b64=(
            normalized_pq_public_key
            if signature_scheme == SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65
            else None
        ),
        last_seen_at=now.isoformat(),
        updated_at=now.isoformat(),
    )
    attestation.validator_signature_b64 = sign_payload_b64(
        decode_bytes(str(validator_signing_seed_b64 or "").strip()),
        worker_capability_attestation_signing_body(attestation),
    )
    if signature_scheme == SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65:
        attestation.validator_pq_signature_b64 = sign_payload_mldsa65_b64(
            normalized_pq_private_key,
            worker_capability_attestation_signing_body(attestation),
        )
    return attestation


def record_worker_capability_attestation(
    record: Any,
    *,
    validator_id: str,
    validator_public_key_b64: str,
    validator_signing_seed_b64: str,
    validator_pq_public_key_b64: str | None = None,
    validator_pq_private_key_b64: str | None = None,
    ttl_seconds: int = WORKER_CAPABILITY_ATTESTATION_TTL_SECONDS,
    accepted: bool = True,
    note: str | None = None,
    probe_result: dict[str, Any] | None = None,
    policy: WalletPolicy | None = None,
) -> WorkerCapabilityAttestation:
    attestation = create_worker_capability_attestation(
        record,
        validator_id=validator_id,
        validator_public_key_b64=validator_public_key_b64,
        validator_signing_seed_b64=validator_signing_seed_b64,
        validator_pq_public_key_b64=validator_pq_public_key_b64,
        validator_pq_private_key_b64=validator_pq_private_key_b64,
        ttl_seconds=ttl_seconds,
        accepted=accepted,
        note=note,
        probe_result=probe_result,
    )
    append_worker_capability_attestation(attestation, policy)
    return attestation


def record_worker_capability_challenge_failure_evidence(
    attestation: WorkerCapabilityAttestation,
    *,
    reporter_validator_id: str | None = None,
    note: str | None = None,
    policy: WalletPolicy | None = None,
    money_policy: Any | None = None,
) -> Any:
    from .model import MoneyPolicy
    from .settlement import record_validator_evidence

    validator = get_validator_record(normalize_address(attestation.validator_id), policy)
    active_money_policy = money_policy or MoneyPolicy()
    bonded_atomic = int(getattr(validator, "bonded_atomic", 0) or 0)
    slash_atomic = (
        bonded_atomic * int(active_money_policy.validator_jail_slash_bps)
    ) // 10_000
    return record_validator_evidence(
        validator_id=normalize_address(attestation.validator_id),
        reporter_validator_id=reporter_validator_id,
        evidence_type="worker_capability_challenge_failure",
        attestation_id=attestation.attestation_id,
        slash_atomic=max(0, slash_atomic),
        jailed=False,
        note=note
        or (
            "Validator attested a worker capability that failed an active "
            "challenge."
        ),
        policy=policy,
    )


def list_worker_capability_attestations(
    *,
    worker_node_id: str | None = None,
    validator_id: str | None = None,
    limit: int | None = None,
    policy: WalletPolicy | None = None,
) -> list[WorkerCapabilityAttestation]:
    path = worker_capability_attestation_file_path(policy)
    if not path.exists():
        return []

    normalized_worker_node_id = str(worker_node_id or "").strip()
    normalized_validator_id = (
        normalize_address(validator_id) if str(validator_id or "").strip() else None
    )
    items: list[WorkerCapabilityAttestation] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        raw.setdefault("accepted_model_ids", [])
        raw.setdefault("resource_summary", {})
        raw.setdefault("readiness", {})
        raw.setdefault("probe_result", {})
        raw.setdefault("accepted", True)
        raw.setdefault("signature_scheme", SIGNING_SCHEME_ED25519)
        raw.setdefault("validator_public_key_b64", None)
        raw.setdefault("validator_signature_b64", None)
        raw.setdefault("validator_pq_public_key_b64", None)
        raw.setdefault("validator_pq_signature_b64", None)
        raw.setdefault("source", "local")
        raw.setdefault("source_url", None)
        raw.setdefault("last_seen_at", None)
        raw.setdefault("updated_at", None)
        item = WorkerCapabilityAttestation(**raw)
        if normalized_worker_node_id and item.worker_node_id != normalized_worker_node_id:
            continue
        if normalized_validator_id and item.validator_id != normalized_validator_id:
            continue
        items.append(item)

    items.sort(key=lambda item: item.created_at, reverse=True)
    if limit is not None:
        return items[: max(0, int(limit))]
    return items


def append_worker_capability_attestation(
    attestation: WorkerCapabilityAttestation,
    policy: WalletPolicy | None = None,
) -> None:
    path = worker_capability_attestation_file_path(policy)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(attestation), ensure_ascii=False) + "\n")


def export_worker_capability_attestations_payload(
    policy: WalletPolicy | None = None,
) -> dict[str, Any]:
    return add_peer_payload_metadata(
        {
            "exported_at": datetime.now(tz=UTC).isoformat(),
            "records": [
                asdict(item)
                for item in list_worker_capability_attestations(policy=policy)
            ],
        },
        policy=policy,
    )


def merge_remote_worker_capability_attestations_payload(
    payload: dict[str, Any],
    *,
    source_url: str,
    policy: WalletPolicy | None = None,
) -> int:
    validate_peer_payload_network(
        payload,
        policy=policy,
        payload_name="worker capability attestations",
    )
    signature_ok, signature_error = verify_peer_payload_signature(
        payload,
        payload_name="worker capability attestations",
        require_signature=peer_payload_signatures_required(policy=policy),
        require_hybrid_signature=peer_payload_hybrid_signatures_required(
            policy=policy
        ),
    )
    if not signature_ok:
        raise ValueError(
            signature_error or "Invalid worker capability attestation payload."
        )
    imported = 0
    existing_ids = {
        item.attestation_id for item in list_worker_capability_attestations(policy=policy)
    }
    now = datetime.now(tz=UTC).isoformat()
    for raw in payload.get("records") or []:
        if not isinstance(raw, dict):
            continue
        item = _attestation_from_raw(
            raw,
            source="peer",
            source_url=source_url,
            observed_at=now,
        )
        if item.attestation_id in existing_ids:
            continue
        ok, _ = verify_worker_capability_attestation(item, policy=policy)
        if not ok:
            continue
        append_worker_capability_attestation(item, policy)
        existing_ids.add(item.attestation_id)
        imported += 1
    return imported


def list_validator_attested_worker_node_ids(
    *,
    records: list[Any],
    accepted_model_ids: set[str] | None = None,
    max_age_seconds: int | None = None,
    policy: WalletPolicy | None = None,
) -> set[str]:
    accepted = {
        str(model_id).strip()
        for model_id in (accepted_model_ids or set())
        if str(model_id).strip()
    }
    by_node_id = {
        _record_text(record, "node_id"): record
        for record in records
        if _record_text(record, "node_id")
    }
    result: set[str] = set()
    for attestation in list_worker_capability_attestations(policy=policy):
        record = by_node_id.get(attestation.worker_node_id)
        if record is None:
            continue
        ok, _ = verify_worker_capability_attestation(
            attestation,
            record=record,
            accepted_model_ids=accepted,
            max_age_seconds=max_age_seconds,
            policy=policy,
        )
        if ok:
            result.add(attestation.worker_node_id)
    return result


def verify_worker_capability_attestation(
    attestation: WorkerCapabilityAttestation,
    *,
    record: Any | None = None,
    accepted_model_ids: set[str] | None = None,
    max_age_seconds: int | None = None,
    policy: WalletPolicy | None = None,
) -> tuple[bool, str | None]:
    if not attestation.accepted:
        return False, "worker capability attestation rejected worker"
    if attestation.signature_scheme not in {
        SIGNING_SCHEME_ED25519,
        SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65,
    }:
        return False, "worker capability attestation signature scheme is unsupported"
    if not attestation.validator_public_key_b64 or not attestation.validator_signature_b64:
        return False, "worker capability attestation signature is missing"
    if attestation.signature_scheme == SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65 and (
        not attestation.validator_pq_public_key_b64
        or not attestation.validator_pq_signature_b64
    ):
        return False, "worker capability attestation PQ signature is missing"
    try:
        validator_address = _identity_address_from_public_keys(
            attestation.validator_public_key_b64,
            attestation.validator_pq_public_key_b64
            if attestation.signature_scheme == SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65
            else None,
        )
    except Exception:
        return False, "worker capability attestation validator public key is invalid"
    if validator_address != normalize_address(attestation.validator_id):
        return False, "worker capability attestation signer does not match validator id"
    validator = get_validator_record(normalize_address(attestation.validator_id), policy)
    if (
        validator is None
        or validator.state != ValidatorLifecycleState.BONDED
        or int(validator.bonded_atomic or 0) <= 0
    ):
        return False, "worker capability attestation signer is not a bonded validator"
    if not _verify_worker_capability_signature(
        signature_scheme=attestation.signature_scheme,
        public_key_b64=attestation.validator_public_key_b64,
        signature_b64=attestation.validator_signature_b64,
        pq_public_key_b64=str(attestation.validator_pq_public_key_b64 or ""),
        pq_signature_b64=str(attestation.validator_pq_signature_b64 or ""),
        payload=worker_capability_attestation_signing_body(attestation),
    ):
        return False, "worker capability attestation signature is invalid"

    now = datetime.now(tz=UTC)
    expires_at = _parse_iso_datetime(attestation.expires_at)
    if expires_at is None or expires_at <= now:
        return False, "worker capability attestation is expired"
    created_at = _parse_iso_datetime(attestation.created_at)
    if (
        max_age_seconds is not None
        and created_at is not None
        and (now - created_at).total_seconds() > max(0, int(max_age_seconds))
    ):
        return False, "worker capability attestation is stale"

    accepted = {
        str(model_id).strip()
        for model_id in (accepted_model_ids or set())
        if str(model_id).strip()
    }
    attested_models = {
        str(model_id).strip()
        for model_id in (attestation.accepted_model_ids or [])
        if str(model_id).strip()
    }
    if accepted and attested_models and not accepted.intersection(attested_models):
        return False, "worker capability attestation does not cover requested model"

    if record is not None:
        if _record_text(record, "node_id") != attestation.worker_node_id:
            return False, "worker capability attestation node id mismatch"
        reward_address = (
            normalize_address(_record_text(record, "worker_reward_address"))
            if _record_text(record, "worker_reward_address")
            else None
        )
        if reward_address != attestation.worker_reward_address:
            return False, "worker capability attestation reward address mismatch"
        public_key_address = _record_worker_public_key_address(record)
        if public_key_address != attestation.worker_public_key_address:
            return False, "worker capability attestation worker key mismatch"
        if (
            worker_capability_fingerprint_from_record(record)
            != attestation.capability_fingerprint
        ):
            return False, "worker capability attestation fingerprint mismatch"
    if worker_capability_challenge_required():
        return _verify_worker_capability_attestation_challenge(
            attestation,
            record=record,
            policy=policy,
        )
    return True, None


def worker_capability_attestation_signing_body(
    attestation: WorkerCapabilityAttestation,
) -> dict[str, Any]:
    body = asdict(attestation)
    body.pop("validator_signature_b64", None)
    body.pop("validator_pq_signature_b64", None)
    body.pop("source", None)
    body.pop("source_url", None)
    body.pop("last_seen_at", None)
    body.pop("updated_at", None)
    return body


def worker_capability_challenge_signing_body(
    challenge: dict[str, Any],
) -> dict[str, Any]:
    body = dict(challenge)
    body.pop("validator_signature_b64", None)
    body.pop("validatorSignatureB64", None)
    body.pop("validator_pq_signature_b64", None)
    body.pop("validatorPqSignatureB64", None)
    return body


def worker_capability_challenge_receipt_signing_body(
    receipt: dict[str, Any],
) -> dict[str, Any]:
    body = dict(receipt)
    body.pop("worker_signature_b64", None)
    body.pop("workerSignatureB64", None)
    body.pop("worker_pq_signature_b64", None)
    body.pop("workerPqSignatureB64", None)
    return body


def _verify_worker_capability_attestation_challenge(
    attestation: WorkerCapabilityAttestation,
    *,
    record: Any | None,
    policy: WalletPolicy | None,
) -> tuple[bool, str | None]:
    challenge = attestation.probe_result.get("challenge")
    receipt = (
        attestation.probe_result.get("challengeReceipt")
        or attestation.probe_result.get("challenge_receipt")
    )
    if not isinstance(challenge, dict) or not isinstance(receipt, dict):
        return False, "worker capability challenge proof is missing"
    ok, error = verify_worker_capability_challenge(
        challenge,
        policy=policy,
        require_bonded_validator=True,
        enforce_expiry=False,
    )
    if not ok:
        return False, error
    if normalize_address(str(challenge.get("validator_id") or "")) != (
        normalize_address(attestation.validator_id)
    ):
        return False, "worker capability challenge validator mismatch"
    ok, error = verify_worker_capability_challenge_receipt(
        receipt,
        challenge=challenge,
        record=record,
    )
    if not ok:
        return False, error
    if str(receipt.get("worker_node_id") or "") != attestation.worker_node_id:
        return False, "worker capability challenge worker mismatch"
    if normalize_address(str(receipt.get("worker_reward_address") or "")) != (
        normalize_address(str(attestation.worker_reward_address or ""))
    ):
        return False, "worker capability challenge reward mismatch"
    receipt_worker_key = normalize_address(
        str(receipt.get("worker_public_key_address") or "")
    )
    attestation_worker_key = normalize_address(
        str(attestation.worker_public_key_address or "")
    )
    if receipt_worker_key != attestation_worker_key:
        return False, "worker capability challenge worker key mismatch"
    if str(receipt.get("capability_fingerprint") or "") != (
        attestation.capability_fingerprint
    ):
        return False, "worker capability challenge fingerprint mismatch"
    return True, None


def _attestation_from_raw(
    raw: dict[str, Any],
    *,
    source: str,
    source_url: str | None,
    observed_at: str,
) -> WorkerCapabilityAttestation:
    return WorkerCapabilityAttestation(
        attestation_id=str(
            raw.get("attestation_id") or raw.get("attestationId") or ""
        ).strip(),
        created_at=str(raw.get("created_at") or raw.get("createdAt") or observed_at),
        expires_at=str(raw.get("expires_at") or raw.get("expiresAt") or observed_at),
        validator_id=normalize_address(raw.get("validator_id") or raw.get("validatorId")),
        worker_node_id=str(raw.get("worker_node_id") or raw.get("workerNodeId") or "").strip(),
        worker_reward_address=(
            normalize_address(
                raw.get("worker_reward_address") or raw.get("workerRewardAddress")
            )
            if raw.get("worker_reward_address") or raw.get("workerRewardAddress")
            else None
        ),
        worker_public_key_address=(
            normalize_address(
                raw.get("worker_public_key_address")
                or raw.get("workerPublicKeyAddress")
            )
            if raw.get("worker_public_key_address")
            or raw.get("workerPublicKeyAddress")
            else None
        ),
        capability_fingerprint=str(
            raw.get("capability_fingerprint") or raw.get("capabilityFingerprint") or ""
        ).strip(),
        accepted_model_ids=[
            str(item).strip()
            for item in (
                raw.get("accepted_model_ids") or raw.get("acceptedModelIds") or []
            )
            if str(item).strip()
        ],
        resource_summary=_safe_json_mapping(
            raw.get("resource_summary") or raw.get("resourceSummary")
        ),
        readiness=_safe_json_mapping(raw.get("readiness")),
        probe_result=_safe_json_mapping(raw.get("probe_result") or raw.get("probeResult")),
        accepted=bool(raw.get("accepted", True)),
        note=raw.get("note"),
        signature_scheme=str(
            raw.get("signature_scheme")
            or raw.get("signatureScheme")
            or SIGNING_SCHEME_ED25519
        ),
        validator_public_key_b64=raw.get("validator_public_key_b64")
        or raw.get("validatorPublicKeyB64"),
        validator_signature_b64=raw.get("validator_signature_b64")
        or raw.get("validatorSignatureB64"),
        validator_pq_public_key_b64=raw.get("validator_pq_public_key_b64")
        or raw.get("validatorPqPublicKeyB64"),
        validator_pq_signature_b64=raw.get("validator_pq_signature_b64")
        or raw.get("validatorPqSignatureB64"),
        source=source,
        source_url=source_url,
        last_seen_at=observed_at,
        updated_at=str(raw.get("updated_at") or raw.get("updatedAt") or observed_at),
    )


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _safe_json_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError):
        return {}


def _record_text(record: Any, field_name: str) -> str:
    return str(getattr(record, field_name, "") or "").strip()


def _record_worker_public_key_address(record: Any) -> str | None:
    for field_name in ("node_public_key_address", "payload_public_key_address"):
        value = _record_text(record, field_name)
        if value:
            return normalize_address(value)
    return None


def _normalize_challenge_difficulty(value: Any) -> int:
    try:
        difficulty = int(value)
    except (TypeError, ValueError):
        difficulty = WORKER_CAPABILITY_CHALLENGE_DIFFICULTY
    return max(0, min(6, difficulty))


def _solve_challenge_hash(
    *,
    challenge_id: str,
    nonce: str,
    worker_node_id: str,
    capability_fingerprint: str,
    difficulty: int,
) -> tuple[int, str]:
    prefix = "0" * _normalize_challenge_difficulty(difficulty)
    counter = 0
    while True:
        proof_hash = _challenge_hash(
            challenge_id=challenge_id,
            nonce=nonce,
            worker_node_id=worker_node_id,
            capability_fingerprint=capability_fingerprint,
            counter=counter,
        )
        if proof_hash.startswith(prefix):
            return counter, proof_hash
        counter += 1


def _challenge_hash(
    *,
    challenge_id: str,
    nonce: str,
    worker_node_id: str,
    capability_fingerprint: str,
    counter: int,
) -> str:
    payload = {
        "challenge_id": str(challenge_id or "").strip(),
        "nonce": str(nonce or "").strip(),
        "worker_node_id": str(worker_node_id or "").strip(),
        "capability_fingerprint": str(capability_fingerprint or "").strip(),
        "counter": max(0, int(counter)),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

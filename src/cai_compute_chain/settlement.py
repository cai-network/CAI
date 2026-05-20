# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import logging
import secrets
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from .economics import FundingDecision, chain_backed_ledger_snapshot
from .wallet_signing import (
    ADDRESS_SCHEME_FIXED_WALLET_ID_SHA256,
    ADDRESS_SCHEME_ED25519,
    HYBRID_ADDRESS_SCHEMES,
    SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65,
    SIGNING_SCHEME_ML_DSA_65,
    SIGNING_SCHEME_ED25519,
    canonical_payload,
    decode_bytes,
    sign_payload_mldsa65_b64,
    sign_payload_b64,
    verify_hybrid_payload_signature,
    verify_payload_signature,
)
from .chain import (
    append_chain_block,
    chain_balance_atomic,
    chain_settlement_history,
    compute_reserve_chain_address,
    ensure_chain_genesis,
    has_chain_activity_for_address,
    make_chain_transaction,
    tx_fee_pool_chain_address,
    validator_settlement_fee_pool_chain_address,
)
from .model import MoneyPolicy, WalletPolicy
from .local_json_store import (
    atomic_write_json_array_file,
    atomic_write_text_file,
    read_json_array_file,
    read_jsonl_object_file,
)
from .node_config import (
    resolve_worker_reward_address,
)
from .peer_payload import (
    add_peer_payload_metadata,
    peer_payload_hybrid_signatures_required,
    peer_payload_signatures_required,
    validate_peer_payload_network,
    verify_peer_payload_signature,
)
from .validators import (
    apply_remote_validator_penalty_to_registry,
    discover_peer_cai_urls,
    get_validator_record,
    list_bonded_validators,
    resolve_validator_peer_url,
    select_validator_committee_snapshot,
    split_amount_by_validator_bond,
    sync_validator_set_from_cai_peers,
)
from .wallet import (
    JournalEntry,
    append_journal_entry,
    data_root,
    find_wallet_by_address,
    find_wallet_by_id,
    load_unlocked_wallet_signing_material,
    load_or_create_ledger,
    normalize_address,
    save_ledger,
    update_wallet,
)


LOGGER = logging.getLogger(__name__)


class ConflictingAttestationError(ValueError):
    def __init__(self, existing_attestation: "ValidatorAttestation") -> None:
        self.existing_attestation = existing_attestation
        super().__init__(
            "Conflicting attestation already exists for this settlement and validator."
        )


_RETRYABLE_SETTLEMENT_ENVELOPE_REJECTION_NOTES = {
    "settlement signed envelope payload hash does not match",
    "settlement signed envelope is missing",
}


def _error_payload(exc: Exception) -> dict[str, str]:
    return {
        "errorType": type(exc).__name__,
        "message": str(exc),
    }


def _peer_error_payload(peer_url: str, exc: Exception) -> dict[str, str]:
    return {
        "peerUrl": peer_url,
        **_error_payload(exc),
    }


def _log_best_effort_failure(operation: str, exc: Exception) -> None:
    LOGGER.warning(
        "Best-effort %s failed: %s",
        operation,
        exc,
        exc_info=LOGGER.isEnabledFor(logging.DEBUG),
    )


@dataclass
class SettlementRecord:
    settlement_id: str
    created_at: str
    source_wallet_id: str
    source_wallet_address: str
    funding_source: str
    compute_cost_atomic: int
    tx_fee_atomic: int
    settlement_fee_atomic: int
    worker_reward_atomic: int
    committee_selection_seed: str | None
    committee_target_size: int
    committee_selection_mode: str
    committee_validator_ids: list[str]
    committee_bonded_atomic_by_validator_id: dict[str, int]
    committee_total_bonded_atomic: int
    committee_quorum_bond_atomic: int
    reward_token_code: str | None = None
    ai_development_fee_atomic: int = 0
    ai_development_wallet_id: str | None = None
    ai_development_address: str | None = None
    ai_development_credited_wallet_id: str | None = None
    accepted_attestations: int = 0
    rejected_attestations: int = 0
    accepted_bond_atomic: int = 0
    rejected_bond_atomic: int = 0
    status: str = "pending"
    note: str | None = None
    source_wallet_debit_atomic: int = 0
    reserve_debit_atomic: int = 0
    reserve_limit_identity_keys: list[str] = field(default_factory=list)
    reserve_client_ip_hash: str | None = None
    applied_at: str | None = None
    applied_by_validator_id: str | None = None
    balance_audit: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidatorAttestation:
    attestation_id: str
    created_at: str
    settlement_id: str
    validator_id: str
    accepted: bool
    note: str | None = None


@dataclass
class ValidatorEvidence:
    evidence_id: str
    created_at: str
    validator_id: str
    evidence_type: str
    reporter_validator_id: str | None = None
    settlement_id: str | None = None
    attestation_id: str | None = None
    conflicting_attestation_id: str | None = None
    slash_atomic: int = 0
    jailed: bool = False
    note: str | None = None
    source: str = "local"
    source_url: str | None = None
    last_seen_at: str | None = None
    updated_at: str | None = None
    applied_to_registry: bool = False


@dataclass(frozen=True)
class ValidatorEvidenceSyncResult:
    attempted_peers: int
    successful_peers: int
    imported_records: int
    applied_records: int
    peer_urls: list[str]
    failed_peers: int = 0
    failed_peer_urls: list[str] = field(default_factory=list)
    peer_errors: list[dict[str, str]] = field(default_factory=list)
    validator_set_sync_error: dict[str, str] | None = None
    penalty_attestation_sync_error: dict[str, str] | None = None


@dataclass(frozen=True)
class ValidatorEvidenceCaseSummary:
    case_id: str
    validator_id: str
    evidence_type: str
    settlement_id: str | None
    attestation_id: str | None
    conflicting_attestation_id: str | None
    slash_atomic: int
    jailed: bool
    created_at: str
    updated_at: str | None
    evidence_count: int
    supporting_sources: list[str]
    supporting_sources_count: int
    supporting_validator_ids: list[str]
    supporting_validator_count: int
    support_mode: str
    support_scope: str
    required_sources: int
    evidence_quorum_reached: bool
    penalty_attestation_count: int
    penalty_attestation_required: int
    quorum_reached: bool
    status: str
    finalized_at: str | None
    applied_at: str | None
    applied_to_registry: bool


@dataclass
class WorkerPayoutRecord:
    payout_id: str
    created_at: str
    settlement_id: str
    receipt_id: str
    model_id: str
    node_id: str
    runner_id: str | None
    layer_start: int | None
    layer_end: int | None
    layer_count: int
    share_bps: int
    reward_atomic: int
    reward_token_code: str | None = None
    recipient_address: str | None = None
    credited_wallet_id: str | None = None
    status: str = "unbound"
    note: str | None = None


@dataclass
class ValidatorPenaltyCase:
    case_id: str
    created_at: str
    updated_at: str
    validator_id: str
    evidence_type: str
    settlement_id: str | None
    attestation_id: str | None
    conflicting_attestation_id: str | None
    slash_atomic: int
    jailed: bool
    support_mode: str
    support_scope: str
    required_sources: int
    supporting_sources: list[str]
    supporting_validator_ids: list[str]
    eligible_validator_ids: list[str]
    evidence_count: int
    evidence_quorum_reached: bool
    penalty_attestation_count: int
    penalty_attestation_required: int
    quorum_reached: bool
    status: str
    finalized_at: str | None = None
    applied_at: str | None = None


@dataclass
class ValidatorPenaltyAttestation:
    penalty_attestation_id: str
    created_at: str
    case_id: str
    validator_id: str
    accepted: bool
    note: str | None = None


def settlement_file_path(policy: WalletPolicy | None = None) -> Path:
    active_policy = policy or WalletPolicy()
    return data_root(active_policy) / active_policy.settlement_file_name


def attestation_file_path(policy: WalletPolicy | None = None) -> Path:
    active_policy = policy or WalletPolicy()
    return data_root(active_policy) / active_policy.attestation_file_name


def worker_payout_file_path(policy: WalletPolicy | None = None) -> Path:
    active_policy = policy or WalletPolicy()
    return data_root(active_policy) / active_policy.worker_payout_file_name


def validator_evidence_file_path(policy: WalletPolicy | None = None) -> Path:
    active_policy = policy or WalletPolicy()
    return data_root(active_policy) / active_policy.validator_evidence_file_name


def validator_penalty_case_file_path(policy: WalletPolicy | None = None) -> Path:
    active_policy = policy or WalletPolicy()
    return data_root(active_policy) / active_policy.validator_penalty_case_file_name


def validator_penalty_attestation_file_path(policy: WalletPolicy | None = None) -> Path:
    active_policy = policy or WalletPolicy()
    return data_root(active_policy) / active_policy.validator_penalty_attestation_file_name


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _coalesce_cai_url(cai_url: str | None = None, CAI_url: str | None = None) -> str:
    resolved = str(cai_url or CAI_url or "").strip()
    if not resolved:
        raise ValueError("CAI URL is required.")
    return resolved


def list_settlements(policy: WalletPolicy | None = None) -> list[SettlementRecord]:
    active_money_policy = MoneyPolicy()
    path = settlement_file_path(policy)
    if not path.exists():
        return []
    raw = read_json_array_file(path, heal_corrupt=True)
    records: list[SettlementRecord] = []
    for item in raw:
        legacy_applied = (
            item.get("status") == "finalized"
            and "applied_at" not in item
            and "source_wallet_debit_atomic" not in item
            and "reserve_debit_atomic" not in item
        )
        item.setdefault("committee_validator_ids", [])
        item.setdefault("committee_bonded_atomic_by_validator_id", {})
        item.setdefault("committee_selection_seed", None)
        item.setdefault("committee_target_size", 0)
        item.setdefault("committee_selection_mode", "all_bonded")
        item.setdefault("committee_total_bonded_atomic", 0)
        item.setdefault("committee_quorum_bond_atomic", 0)
        item.setdefault("accepted_attestations", 0)
        item.setdefault("rejected_attestations", 0)
        item.setdefault("accepted_bond_atomic", 0)
        item.setdefault("rejected_bond_atomic", 0)
        item.setdefault("status", "pending")
        item.setdefault("source_wallet_debit_atomic", 0)
        item.setdefault("reserve_debit_atomic", 0)
        item.setdefault("reserve_limit_identity_keys", [])
        item.setdefault("reserve_client_ip_hash", None)
        item.setdefault("applied_at", None)
        item.setdefault("applied_by_validator_id", None)
        item.setdefault("balance_audit", {})
        item.setdefault("reward_token_code", active_money_policy.reward_token_code)
        item.setdefault("ai_development_fee_atomic", 0)
        item.setdefault("ai_development_wallet_id", active_money_policy.ai_development_wallet_id)
        item.setdefault("ai_development_address", active_money_policy.ai_development_address)
        item.setdefault("ai_development_credited_wallet_id", None)
        if legacy_applied:
            item["status"] = "applied"
            item["applied_at"] = item.get("created_at")
        records.append(SettlementRecord(**item))
    records.sort(key=lambda item: item.created_at, reverse=True)
    return records


def save_settlements(
    settlements: list[SettlementRecord], policy: WalletPolicy | None = None
) -> None:
    path = settlement_file_path(policy)
    atomic_write_json_array_file(
        path,
        [asdict(item) for item in settlements],
    )


def list_attestations(
    *,
    settlement_id: str | None = None,
    limit: int | None = None,
    policy: WalletPolicy | None = None,
) -> list[ValidatorAttestation]:
    path = attestation_file_path(policy)
    if not path.exists():
        return []

    items: list[ValidatorAttestation] = []
    for raw in read_jsonl_object_file(path, heal_corrupt=True):
        item = ValidatorAttestation(**raw)
        if settlement_id is not None and item.settlement_id != settlement_id:
            continue
        items.append(item)

    items.sort(key=lambda entry: entry.created_at, reverse=True)
    if limit is not None:
        return items[:limit]
    return items


def list_worker_payouts(
    *,
    settlement_id: str | None = None,
    receipt_id: str | None = None,
    limit: int | None = None,
    policy: WalletPolicy | None = None,
) -> list[WorkerPayoutRecord]:
    active_money_policy = MoneyPolicy()
    path = worker_payout_file_path(policy)
    if not path.exists():
        return []
    raw = read_json_array_file(path, heal_corrupt=True)
    for item in raw:
        item.setdefault("reward_token_code", active_money_policy.reward_token_code)
    items = [WorkerPayoutRecord(**item) for item in raw]
    if settlement_id is not None:
        items = [item for item in items if item.settlement_id == settlement_id]
    if receipt_id is not None:
        items = [item for item in items if item.receipt_id == receipt_id]
    items.sort(key=lambda item: item.created_at, reverse=True)
    if limit is not None:
        return items[:limit]
    return items


def append_attestation(
    attestation: ValidatorAttestation, policy: WalletPolicy | None = None
) -> None:
    path = attestation_file_path(policy)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(attestation), ensure_ascii=False) + "\n")


def save_attestations(
    attestations: list[ValidatorAttestation], policy: WalletPolicy | None = None
) -> None:
    path = attestation_file_path(policy)
    lines = [
        json.dumps(asdict(item), ensure_ascii=False) for item in attestations
    ]
    payload = ("\n".join(lines) + ("\n" if lines else ""))
    atomic_write_text_file(path, payload)


def append_validator_evidence(
    evidence: ValidatorEvidence, policy: WalletPolicy | None = None
) -> None:
    path = validator_evidence_file_path(policy)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(evidence), ensure_ascii=False) + "\n")


def list_validator_evidence(
    *,
    validator_id: str | None = None,
    settlement_id: str | None = None,
    limit: int | None = None,
    policy: WalletPolicy | None = None,
) -> list[ValidatorEvidence]:
    path = validator_evidence_file_path(policy)
    if not path.exists():
        return []

    items: list[ValidatorEvidence] = []
    for raw in read_jsonl_object_file(path, heal_corrupt=True):
        raw.setdefault("source", "local")
        raw.setdefault("source_url", None)
        raw.setdefault("last_seen_at", None)
        raw.setdefault("updated_at", None)
        raw.setdefault("applied_to_registry", False)
        raw.setdefault("reporter_validator_id", None)
        item = ValidatorEvidence(**raw)
        if validator_id is not None and item.validator_id != validator_id:
            continue
        if settlement_id is not None and item.settlement_id != settlement_id:
            continue
        items.append(item)

    items.sort(key=lambda entry: entry.created_at, reverse=True)
    if limit is not None:
        return items[:limit]
    return items


def list_validator_penalty_attestations(
    *,
    case_id: str | None = None,
    limit: int | None = None,
    policy: WalletPolicy | None = None,
) -> list[ValidatorPenaltyAttestation]:
    path = validator_penalty_attestation_file_path(policy)
    if not path.exists():
        return []

    items: list[ValidatorPenaltyAttestation] = []
    for raw in read_jsonl_object_file(path, heal_corrupt=True):
        item = ValidatorPenaltyAttestation(**raw)
        if case_id is not None and item.case_id != case_id:
            continue
        items.append(item)

    items.sort(key=lambda entry: entry.created_at, reverse=True)
    if limit is not None:
        return items[:limit]
    return items


def append_validator_penalty_attestation(
    attestation: ValidatorPenaltyAttestation, policy: WalletPolicy | None = None
) -> None:
    path = validator_penalty_attestation_file_path(policy)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(attestation), ensure_ascii=False) + "\n")


def list_validator_evidence_cases(
    *,
    validator_id: str | None = None,
    limit: int | None = None,
    policy: WalletPolicy | None = None,
) -> list[ValidatorEvidenceCaseSummary]:
    case_records = {
        item.case_id: item for item in list_validator_penalty_cases(policy=policy)
    }
    grouped: dict[
        tuple[str, str, str | None, str | None, str | None, int, bool],
        list[ValidatorEvidence],
    ] = {}
    for item in list_validator_evidence(policy=policy):
        if validator_id is not None and item.validator_id != validator_id:
            continue
        grouped.setdefault(_validator_evidence_case_key(item), []).append(item)

    cases: list[ValidatorEvidenceCaseSummary] = []
    for case_key, items in grouped.items():
        items.sort(key=lambda item: item.updated_at or item.created_at, reverse=True)
        latest = items[0]
        supporting_sources = sorted({item.source_url for item in items if item.source_url})
        support_scope, eligible_validator_ids, required_sources = _validator_evidence_case_support_context(
            case_key, policy
        )
        supporting_validator_ids = _validator_evidence_case_supporting_validator_ids(
            case_key, items, policy
        )
        support_mode = "validator" if supporting_validator_ids else "source_url"
        case_id = _validator_evidence_case_id(case_key)
        penalty_case = case_records.get(case_id)
        applied_to_registry = any(item.applied_to_registry for item in items)
        evidence_quorum_reached = _validator_evidence_case_has_quorum(case_key, items, policy)
        if penalty_case is not None:
            quorum_reached = penalty_case.quorum_reached
            status = penalty_case.status
            finalized_at = penalty_case.finalized_at
            applied_at = penalty_case.applied_at
            applied_to_registry = penalty_case.status == "applied"
            penalty_attestation_count = penalty_case.penalty_attestation_count
            penalty_attestation_required = penalty_case.penalty_attestation_required
        else:
            status = "pending"
            finalized_at = None
            applied_at = None
            penalty_attestation_count = len(
                _validator_penalty_attestation_validator_ids(
                    case_id, eligible_validator_ids, policy
                )
            )
            penalty_attestation_required = _validator_penalty_attestation_required(
                support_scope=support_scope,
                required_sources=required_sources,
            )
            quorum_reached = _validator_penalty_case_has_quorum(
                support_scope=support_scope,
                penalty_attestation_count=penalty_attestation_count,
                penalty_attestation_required=penalty_attestation_required,
                evidence_quorum_reached=evidence_quorum_reached,
            )
            if quorum_reached:
                finalized_at = latest.updated_at or latest.created_at
                status = "finalized"
            if applied_to_registry:
                applied_at = latest.updated_at or latest.created_at
                status = "applied"
        cases.append(
            ValidatorEvidenceCaseSummary(
                case_id=case_id,
                validator_id=latest.validator_id,
                evidence_type=latest.evidence_type,
                settlement_id=latest.settlement_id,
                attestation_id=latest.attestation_id,
                conflicting_attestation_id=latest.conflicting_attestation_id,
                slash_atomic=latest.slash_atomic,
                jailed=latest.jailed,
                created_at=min(item.created_at for item in items),
                updated_at=latest.updated_at or latest.created_at,
                evidence_count=len(items),
                supporting_sources=supporting_sources,
                supporting_sources_count=len(supporting_sources),
                supporting_validator_ids=supporting_validator_ids,
                supporting_validator_count=len(supporting_validator_ids),
                support_mode=support_mode,
                support_scope=support_scope,
                required_sources=required_sources,
                evidence_quorum_reached=evidence_quorum_reached,
                penalty_attestation_count=penalty_attestation_count,
                penalty_attestation_required=penalty_attestation_required,
                quorum_reached=quorum_reached,
                status=status,
                finalized_at=finalized_at,
                applied_at=applied_at,
                applied_to_registry=applied_to_registry,
            )
        )
    cases.sort(key=lambda item: item.updated_at or item.created_at, reverse=True)
    if limit is not None:
        return cases[:limit]
    return cases


def list_validator_penalty_cases(
    *,
    validator_id: str | None = None,
    limit: int | None = None,
    policy: WalletPolicy | None = None,
) -> list[ValidatorPenaltyCase]:
    path = validator_penalty_case_file_path(policy)
    if not path.exists():
        return []
    raw = read_json_array_file(path, heal_corrupt=True)
    items: list[ValidatorPenaltyCase] = []
    for item in raw:
        item.setdefault("supporting_sources", [])
        item.setdefault("supporting_validator_ids", [])
        item.setdefault("eligible_validator_ids", [])
        item.setdefault("evidence_count", 0)
        item.setdefault("evidence_quorum_reached", False)
        item.setdefault("penalty_attestation_count", 0)
        item.setdefault(
            "penalty_attestation_required",
            _validator_penalty_attestation_required(
                support_scope=str(item.get("support_scope") or "source_url"),
                required_sources=int(item.get("required_sources", 1) or 1),
            ),
        )
        item.setdefault("quorum_reached", False)
        item.setdefault("status", "pending")
        item.setdefault("finalized_at", None)
        item.setdefault("applied_at", None)
        record = ValidatorPenaltyCase(**item)
        if validator_id is not None and record.validator_id != validator_id:
            continue
        items.append(record)
    items.sort(key=lambda item: item.updated_at, reverse=True)
    if limit is not None:
        return items[:limit]
    return items


def save_validator_penalty_cases(
    cases: list[ValidatorPenaltyCase], policy: WalletPolicy | None = None
) -> None:
    path = validator_penalty_case_file_path(policy)
    atomic_write_json_array_file(
        path,
        [asdict(item) for item in cases],
    )


def save_validator_evidence(
    evidence_items: list[ValidatorEvidence], policy: WalletPolicy | None = None
) -> None:
    path = validator_evidence_file_path(policy)
    lines = [
        json.dumps(asdict(item), ensure_ascii=False) for item in evidence_items
    ]
    payload = ("\n".join(lines) + ("\n" if lines else ""))
    atomic_write_text_file(path, payload)


def export_validator_evidence_payload(policy: WalletPolicy | None = None) -> dict[str, Any]:
    return add_peer_payload_metadata(
        {
            "exported_at": _now_iso(),
            "evidence": [
                asdict(item) for item in list_validator_evidence(policy=policy)
            ],
        },
        policy=policy,
    )


def merge_remote_validator_evidence_payload(
    payload: dict[str, Any],
    *,
    source_url: str,
    policy: WalletPolicy | None = None,
) -> tuple[int, int]:
    validate_peer_payload_network(
        payload,
        policy=policy,
        payload_name="validator evidence",
    )
    signature_ok, signature_error = verify_peer_payload_signature(
        payload,
        payload_name="validator evidence",
        require_signature=peer_payload_signatures_required(policy=policy),
        require_hybrid_signature=peer_payload_hybrid_signatures_required(
            policy=policy
        ),
    )
    if not signature_ok:
        raise ValueError(
            signature_error or "Invalid validator evidence payload signature."
        )
    existing_items = list_validator_evidence(policy=policy)
    existing_ids = {item.evidence_id for item in existing_items}
    imported = 0
    applied = 0
    now = _now_iso()
    for raw in _iter_remote_validator_evidence(payload):
        evidence_id = str(raw.get("evidence_id") or "").strip().lower()
        if not evidence_id or evidence_id in existing_ids:
            continue
        evidence = ValidatorEvidence(
            evidence_id=evidence_id,
            created_at=str(raw.get("created_at") or now),
            validator_id=str(raw.get("validator_id") or "").strip().lower(),
            reporter_validator_id=_normalize_validator_reporter_id(
                raw.get("reporter_validator_id")
            ),
            evidence_type=str(raw.get("evidence_type") or "").strip(),
            settlement_id=raw.get("settlement_id"),
            attestation_id=raw.get("attestation_id"),
            conflicting_attestation_id=raw.get("conflicting_attestation_id"),
            slash_atomic=max(0, int(raw.get("slash_atomic") or 0)),
            jailed=bool(raw.get("jailed")),
            note=raw.get("note"),
            source="peer",
            source_url=source_url,
            last_seen_at=now,
            updated_at=raw.get("updated_at") or raw.get("created_at") or now,
            applied_to_registry=False,
        )
        existing_items.append(evidence)
        existing_ids.add(evidence_id)
        imported += 1
    changed = imported > 0
    penalty_cases, applied = refresh_validator_penalty_cases(
        evidence_items=existing_items,
        policy=policy,
    )
    if imported or changed:
        save_validator_evidence(existing_items, policy)
    if imported or penalty_cases:
        save_validator_penalty_cases(penalty_cases, policy)
    return imported, applied


def sync_validator_evidence_from_cai_peers(
    *,
    state_payload: dict[str, Any],
    cai_url: str | None = None,
    CAI_url: str | None = None,
    policy: WalletPolicy | None = None,
    local_node_id: str | None = None,
    timeout_sec: int = 5,
) -> ValidatorEvidenceSyncResult:
    resolved_cai_url = _coalesce_cai_url(cai_url, CAI_url)
    peer_urls = discover_peer_cai_urls(
        state_payload=state_payload,
        cai_url=resolved_cai_url,
        endpoint_path="/v1/cai/validator-evidence",
        local_node_id=local_node_id,
    )
    imported_records = 0
    applied_records = 0
    successful_peers = 0
    failed_peer_urls: list[str] = []
    peer_errors: list[dict[str, str]] = []
    for peer_url in peer_urls:
        try:
            with urlopen(peer_url, timeout=timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
            imported, applied = merge_remote_validator_evidence_payload(
                payload,
                source_url=peer_url,
                policy=policy,
            )
            imported_records += imported
            applied_records += applied
            successful_peers += 1
        except Exception as exc:
            failed_peer_urls.append(peer_url)
            peer_errors.append(_peer_error_payload(peer_url, exc))
            continue
    validator_set_sync_error: dict[str, str] | None = None
    try:
        sync_validator_set_from_cai_peers(
            state_payload=state_payload,
            cai_url=resolved_cai_url,
            policy=policy,
            local_node_id=local_node_id,
            timeout_sec=timeout_sec,
        )
    except Exception as exc:
        validator_set_sync_error = _error_payload(exc)
        _log_best_effort_failure(
            "validator evidence follow-up validator set sync",
            exc,
        )
    penalty_attestation_sync_error: dict[str, str] | None = None
    try:
        request_remote_penalty_case_attestations(
            cai_url=resolved_cai_url,
            state_payload=state_payload,
            policy=policy,
            timeout_sec=timeout_sec,
        )
    except Exception as exc:
        penalty_attestation_sync_error = _error_payload(exc)
        _log_best_effort_failure(
            "validator evidence follow-up penalty attestation sync",
            exc,
        )
    return ValidatorEvidenceSyncResult(
        attempted_peers=len(peer_urls),
        successful_peers=successful_peers,
        imported_records=imported_records,
        applied_records=applied_records,
        peer_urls=peer_urls,
        failed_peers=len(failed_peer_urls),
        failed_peer_urls=failed_peer_urls,
        peer_errors=peer_errors,
        validator_set_sync_error=validator_set_sync_error,
        penalty_attestation_sync_error=penalty_attestation_sync_error,
    )


def save_worker_payouts(
    payouts: list[WorkerPayoutRecord], policy: WalletPolicy | None = None
) -> None:
    path = worker_payout_file_path(policy)
    atomic_write_json_array_file(
        path,
        [asdict(item) for item in payouts],
    )


def export_settlement_proposal_payload(
    settlement_id: str,
    *,
    policy: WalletPolicy | None = None,
) -> dict[str, Any]:
    settlement = resolve_settlement(settlement_id, policy)
    if settlement is None:
        raise ValueError(f"Settlement '{settlement_id}' not found.")
    return {
        "schema_version": 1,
        "settlement": asdict(settlement),
        "worker_payouts": [
            asdict(item)
            for item in list_worker_payouts(settlement_id=settlement_id, policy=policy)
        ],
    }


def import_settlement_proposal_payload(
    payload: dict[str, Any],
    *,
    policy: WalletPolicy | None = None,
    money_policy: MoneyPolicy | None = None,
) -> SettlementRecord:
    if not isinstance(payload, dict):
        raise ValueError("Settlement proposal payload must be an object.")
    raw_settlement = payload.get("settlement")
    if not isinstance(raw_settlement, dict):
        raise ValueError("Settlement proposal is missing settlement record.")

    active_money_policy = money_policy or MoneyPolicy()
    settlement = _coerce_settlement_record(raw_settlement, active_money_policy)
    settlement.status = "pending"
    settlement.accepted_attestations = 0
    settlement.rejected_attestations = 0
    settlement.accepted_bond_atomic = 0
    settlement.rejected_bond_atomic = 0
    settlement.applied_at = None
    settlement.applied_by_validator_id = None
    settlement.ai_development_credited_wallet_id = None
    valid_envelope, envelope_error = verify_settlement_envelope(
        settlement,
        money_policy=active_money_policy,
    )
    if not valid_envelope:
        raise ValueError(envelope_error or "Invalid settlement envelope.")

    raw_payouts = payload.get("worker_payouts") or []
    if not isinstance(raw_payouts, list):
        raise ValueError("Settlement proposal worker_payouts must be a list.")
    proposed_payouts = [
        _coerce_worker_payout_record(item)
        for item in raw_payouts
        if isinstance(item, dict)
    ]
    payout_errors = validate_settlement_accounting(
        settlement,
        proposed_payouts,
        money_policy=active_money_policy,
    )
    if payout_errors:
        raise ValueError("Settlement payout accounting mismatch: " + " ".join(payout_errors))

    existing_settlements = list_settlements(policy)
    existing = next(
        (
            item
            for item in existing_settlements
            if item.settlement_id == settlement.settlement_id
        ),
        None,
    )
    if existing is not None:
        existing_hash = settlement_envelope_hash(
            settlement_envelope_payload(existing, money_policy=active_money_policy)
        )
        incoming_hash = settlement_envelope_hash(
            settlement_envelope_payload(settlement, money_policy=active_money_policy)
        )
        if existing_hash != incoming_hash:
            raise ValueError("Existing settlement envelope differs from proposal.")
        settlement = existing
    else:
        existing_settlements.append(settlement)
        save_settlements(existing_settlements, policy)

    _upsert_worker_payout_records(proposed_payouts, policy=policy)
    return resolve_settlement(settlement.settlement_id, policy) or settlement


def _coerce_settlement_record(
    raw: dict[str, Any],
    money_policy: MoneyPolicy,
) -> SettlementRecord:
    item = dict(raw)
    item.setdefault("committee_validator_ids", [])
    item.setdefault("committee_bonded_atomic_by_validator_id", {})
    item.setdefault("committee_selection_seed", None)
    item.setdefault("committee_target_size", 0)
    item.setdefault("committee_selection_mode", "all_bonded")
    item.setdefault("committee_total_bonded_atomic", 0)
    item.setdefault("committee_quorum_bond_atomic", 0)
    item.setdefault("accepted_attestations", 0)
    item.setdefault("rejected_attestations", 0)
    item.setdefault("accepted_bond_atomic", 0)
    item.setdefault("rejected_bond_atomic", 0)
    item.setdefault("status", "pending")
    item.setdefault("source_wallet_debit_atomic", 0)
    item.setdefault("reserve_debit_atomic", 0)
    item.setdefault("reserve_limit_identity_keys", [])
    item.setdefault("reserve_client_ip_hash", None)
    item.setdefault("applied_at", None)
    item.setdefault("applied_by_validator_id", None)
    item.setdefault("balance_audit", {})
    item.setdefault("reward_token_code", money_policy.reward_token_code)
    item.setdefault("ai_development_fee_atomic", 0)
    item.setdefault("ai_development_wallet_id", money_policy.ai_development_wallet_id)
    item.setdefault("ai_development_address", money_policy.ai_development_address)
    item.setdefault("ai_development_credited_wallet_id", None)
    return SettlementRecord(**item)


def _coerce_worker_payout_record(raw: dict[str, Any]) -> WorkerPayoutRecord:
    item = dict(raw)
    item.setdefault("reward_token_code", MoneyPolicy().reward_token_code)
    item.setdefault("recipient_address", None)
    item["credited_wallet_id"] = None
    item["status"] = (
        "pending_settlement"
        if str(item.get("recipient_address") or "").strip()
        else "unbound"
    )
    item.setdefault("note", None)
    return WorkerPayoutRecord(**item)


def _upsert_worker_payout_records(
    payouts: list[WorkerPayoutRecord],
    *,
    policy: WalletPolicy | None = None,
) -> int:
    if not payouts:
        return 0
    existing = list_worker_payouts(policy=policy)
    by_id = {item.payout_id: index for index, item in enumerate(existing)}
    changed = 0
    for payout in payouts:
        if payout.payout_id in by_id:
            existing[by_id[payout.payout_id]] = payout
        else:
            existing.append(payout)
        changed += 1
    save_worker_payouts(existing, policy)
    return changed


def _validator_fee_payout_split(
    settlement: SettlementRecord,
    *,
    amount_atomic: int | None = None,
) -> list[dict[str, Any]]:
    resolved_amount_atomic = (
        max(0, int(amount_atomic))
        if amount_atomic is not None
        else max(0, int(settlement.settlement_fee_atomic or 0))
    )
    shares = split_amount_by_validator_bond(
        resolved_amount_atomic,
        dict(settlement.committee_bonded_atomic_by_validator_id or {}),
        validator_ids=list(settlement.committee_validator_ids or []),
    )
    return [
        {
            "validator_id": item.validator_id,
            "bonded_atomic": int(item.bonded_atomic),
            "fee_atomic": int(item.amount_atomic),
        }
        for item in shares
    ]


def record_chain_entries_for_settlement(
    settlement: SettlementRecord,
    *,
    policy: WalletPolicy | None = None,
    money_policy: MoneyPolicy | None = None,
) -> int:
    if settlement.status != "applied" and not settlement.applied_at:
        return 0

    active_money_policy = money_policy or MoneyPolicy()
    ensure_chain_genesis(policy=policy, money_policy=active_money_policy)
    recorded = 0
    existing_history = chain_settlement_history(settlement.settlement_id, policy=policy)
    recorded_tx_ids = {
        str(item.get("tx_id") or "").strip()
        for item in existing_history
        if str(item.get("tx_id") or "").strip()
    }
    recorded_nonces = {
        (normalize_address(item.get("address")), str(item.get("nonce")))
        for item in existing_history
        if item.get("nonce") is not None and str(item.get("address") or "").strip()
    }
    balance_audit_targets: list[dict[str, Any]] = []
    pending_transactions: list[Any] = []

    def _record_if_missing(tx: Any) -> bool:
        nonlocal recorded
        tx_id = str(getattr(tx, "tx_id", "") or "").strip()
        if tx_id and tx_id in recorded_tx_ids:
            return False

        nonce_value = getattr(tx, "nonce", None)
        nonce_key: tuple[str, str] | None = None
        if nonce_value is not None:
            nonce_key = (
                normalize_address(getattr(tx, "address")),
                str(nonce_value),
            )
            if nonce_key in recorded_nonces:
                return False

        pending_transactions.append(tx)
        if tx_id:
            recorded_tx_ids.add(tx_id)
        if nonce_key is not None:
            recorded_nonces.add(nonce_key)
        recorded += 1
        return True

    source_wallet_debit = max(0, int(settlement.source_wallet_debit_atomic or 0))
    source_address = str(settlement.source_wallet_address or "").strip()
    if source_wallet_debit > 0 and source_address:
        balance_audit_targets.append(
            {
                "role": "source_wallet",
                "address": source_address,
                "wallet_id": settlement.source_wallet_id,
                "expected_delta_atomic": -source_wallet_debit,
            }
        )
        tx = make_chain_transaction(
            tx_type="settlement_wallet_debit",
            address=source_address,
            delta_atomic=-source_wallet_debit,
            settlement_id=settlement.settlement_id,
            wallet_id=settlement.source_wallet_id,
            note=f"Wallet debit for finalized settlement {settlement.settlement_id}.",
            nonce=f"{settlement.settlement_id}:wallet-debit",
            metadata={
                "reward_token_code": active_money_policy.reward_token_code,
                "funding_source": settlement.funding_source,
            },
            chain_id=active_money_policy.chain_network.value,
        )
        _record_if_missing(tx)

    reservation_surplus = _settlement_reservation_surplus_atomic(settlement)
    if reservation_surplus > 0 and source_address:
        balance_audit_targets.append(
            {
                "role": "reservation_surplus_release",
                "address": source_address,
                "wallet_id": settlement.source_wallet_id,
                "expected_delta_atomic": 0,
            }
        )
        tx = make_chain_transaction(
            tx_type="settlement_reservation_surplus_release",
            address=source_address,
            delta_atomic=0,
            settlement_id=settlement.settlement_id,
            wallet_id=settlement.source_wallet_id,
            note=(
                f"Reservation surplus released for finalized settlement "
                f"{settlement.settlement_id}."
            ),
            nonce=f"{settlement.settlement_id}:reservation-surplus-release",
            metadata={
                "reward_token_code": active_money_policy.reward_token_code,
                "funding_source": settlement.funding_source,
                "reserved_compute_cost_atomic": _settlement_reserved_compute_atomic(
                    settlement
                ),
                "actual_compute_cost_atomic": int(
                    settlement.compute_cost_atomic or 0
                ),
                "reservation_surplus_atomic": reservation_surplus,
            },
            chain_id=active_money_policy.chain_network.value,
        )
        _record_if_missing(tx)

    reserve_debit = max(0, int(settlement.reserve_debit_atomic or 0))
    if reserve_debit > 0:
        reserve_address = compute_reserve_chain_address(active_money_policy)
        balance_audit_targets.append(
            {
                "role": "compute_reserve",
                "address": reserve_address,
                "wallet_id": f"system-compute-reserve-{active_money_policy.chain_network.value}",
                "expected_delta_atomic": -reserve_debit,
            }
        )
        tx = make_chain_transaction(
            tx_type="settlement_compute_reserve_debit",
            address=reserve_address,
            delta_atomic=-reserve_debit,
            settlement_id=settlement.settlement_id,
            wallet_id=f"system-compute-reserve-{active_money_policy.chain_network.value}",
            note=f"Compute reserve debit for finalized settlement {settlement.settlement_id}.",
            nonce=f"{settlement.settlement_id}:reserve-debit",
            metadata={
                "reward_token_code": active_money_policy.reward_token_code,
                "funding_source": settlement.funding_source,
                "source_wallet_id": settlement.source_wallet_id,
                "source_wallet_address": settlement.source_wallet_address,
                "compute_cost_atomic": int(settlement.compute_cost_atomic or 0),
                "reserve_debit_atomic": reserve_debit,
                "reserve_limit_identity_keys": list(
                    settlement.reserve_limit_identity_keys or []
                ),
                "reserve_client_ip_hash": settlement.reserve_client_ip_hash,
            },
            chain_id=active_money_policy.chain_network.value,
        )
        _record_if_missing(tx)

    tx_fee = max(0, int(settlement.tx_fee_atomic or 0))
    if tx_fee > 0:
        fee_pool_address = tx_fee_pool_chain_address(active_money_policy)
        legacy_tx_fee_pool_credit_recorded = any(
            item.get("tx_type") == "settlement_tx_fee_credit"
            and normalize_address(str(item.get("address") or ""))
            == normalize_address(fee_pool_address)
            for item in existing_history
        )
        validator_tx_fee_payouts = _validator_fee_payout_split(
            settlement,
            amount_atomic=tx_fee,
        )
        if validator_tx_fee_payouts and not legacy_tx_fee_pool_credit_recorded:
            total_bonded_atomic = sum(
                int(item["bonded_atomic"]) for item in validator_tx_fee_payouts
            )
            for payout in validator_tx_fee_payouts:
                validator_id = payout["validator_id"]
                fee_atomic = int(payout["fee_atomic"])
                validator_record = get_validator_record(validator_id, policy)
                wallet_id = (
                    validator_record.wallet_id if validator_record is not None else None
                )
                balance_audit_targets.append(
                    {
                        "role": "validator_tx_fee",
                        "address": validator_id,
                        "wallet_id": wallet_id,
                        "validator_id": validator_id,
                        "expected_delta_atomic": fee_atomic,
                    }
                )
                tx = make_chain_transaction(
                    tx_type="validator_tx_fee_payout",
                    address=validator_id,
                    delta_atomic=fee_atomic,
                    settlement_id=settlement.settlement_id,
                    wallet_id=wallet_id,
                    note=f"Validator tx fee payout for {settlement.settlement_id}.",
                    nonce=f"{settlement.settlement_id}:tx-fee:{validator_id}",
                    metadata={
                        "reward_token_code": active_money_policy.reward_token_code,
                        "funding_source": settlement.funding_source,
                        "source_wallet_id": settlement.source_wallet_id,
                        "source_wallet_address": settlement.source_wallet_address,
                        "validator_id": validator_id,
                        "validator_bonded_atomic": int(payout["bonded_atomic"]),
                        "committee_total_bonded_atomic": int(total_bonded_atomic),
                        "committee_validator_ids": list(
                            settlement.committee_validator_ids or []
                        ),
                        "committee_quorum_bond_atomic": int(
                            settlement.committee_quorum_bond_atomic or 0
                        ),
                        "distribution": "committee_bond_weighted",
                    },
                    chain_id=active_money_policy.chain_network.value,
                )
                _record_if_missing(tx)
        else:
            balance_audit_targets.append(
                {
                    "role": "tx_fee_pool",
                    "address": fee_pool_address,
                    "wallet_id": (
                        f"system-tx-fee-pool-"
                        f"{active_money_policy.chain_network.value}"
                    ),
                    "expected_delta_atomic": tx_fee,
                }
            )
            tx = make_chain_transaction(
                tx_type="settlement_tx_fee_credit",
                address=fee_pool_address,
                delta_atomic=tx_fee,
                settlement_id=settlement.settlement_id,
                wallet_id=(
                    f"system-tx-fee-pool-"
                    f"{active_money_policy.chain_network.value}"
                ),
                note=f"Transaction fee from finalized settlement {settlement.settlement_id}.",
                nonce=f"{settlement.settlement_id}:tx-fee",
                metadata={
                    "reward_token_code": active_money_policy.reward_token_code,
                    "funding_source": settlement.funding_source,
                    "source_wallet_id": settlement.source_wallet_id,
                    "source_wallet_address": settlement.source_wallet_address,
                    "distribution": "legacy_pool",
                },
                chain_id=active_money_policy.chain_network.value,
            )
            _record_if_missing(tx)

    settlement_fee = max(0, int(settlement.settlement_fee_atomic or 0))
    if settlement_fee > 0:
        validator_fee_pool_address = validator_settlement_fee_pool_chain_address(
            active_money_policy
        )
        legacy_pool_credit_recorded = any(
            item.get("tx_type") == "settlement_validator_fee_credit"
            and normalize_address(str(item.get("address") or ""))
            == normalize_address(validator_fee_pool_address)
            for item in existing_history
        )
        validator_fee_payouts = _validator_fee_payout_split(settlement)
        if validator_fee_payouts and not legacy_pool_credit_recorded:
            total_bonded_atomic = sum(
                int(item["bonded_atomic"]) for item in validator_fee_payouts
            )
            for payout in validator_fee_payouts:
                validator_id = payout["validator_id"]
                fee_atomic = int(payout["fee_atomic"])
                validator_record = get_validator_record(validator_id, policy)
                wallet_id = (
                    validator_record.wallet_id if validator_record is not None else None
                )
                balance_audit_targets.append(
                    {
                        "role": "validator_settlement_fee",
                        "address": validator_id,
                        "wallet_id": wallet_id,
                        "validator_id": validator_id,
                        "expected_delta_atomic": fee_atomic,
                    }
                )
                tx = make_chain_transaction(
                    tx_type="validator_settlement_fee_payout",
                    address=validator_id,
                    delta_atomic=fee_atomic,
                    settlement_id=settlement.settlement_id,
                    wallet_id=wallet_id,
                    note=(
                        f"Validator settlement fee payout for "
                        f"{settlement.settlement_id}."
                    ),
                    nonce=(
                        f"{settlement.settlement_id}:"
                        f"validator-settlement-fee:{validator_id}"
                    ),
                    metadata={
                        "reward_token_code": active_money_policy.reward_token_code,
                        "funding_source": settlement.funding_source,
                        "source_wallet_id": settlement.source_wallet_id,
                        "source_wallet_address": settlement.source_wallet_address,
                        "validator_id": validator_id,
                        "validator_bonded_atomic": int(payout["bonded_atomic"]),
                        "committee_total_bonded_atomic": int(total_bonded_atomic),
                        "committee_validator_ids": list(
                            settlement.committee_validator_ids or []
                        ),
                        "committee_quorum_bond_atomic": int(
                            settlement.committee_quorum_bond_atomic or 0
                        ),
                        "distribution": "committee_bond_weighted",
                    },
                    chain_id=active_money_policy.chain_network.value,
                )
                _record_if_missing(tx)
        else:
            balance_audit_targets.append(
                {
                    "role": "validator_settlement_fee_pool",
                    "address": validator_fee_pool_address,
                    "wallet_id": (
                        "system-validator-settlement-fee-pool-"
                        f"{active_money_policy.chain_network.value}"
                    ),
                    "expected_delta_atomic": settlement_fee,
                }
            )
            tx = make_chain_transaction(
                tx_type="settlement_validator_fee_credit",
                address=validator_fee_pool_address,
                delta_atomic=settlement_fee,
                settlement_id=settlement.settlement_id,
                wallet_id=(
                    "system-validator-settlement-fee-pool-"
                    f"{active_money_policy.chain_network.value}"
                ),
                note=(
                    f"Validator settlement fee from finalized settlement "
                    f"{settlement.settlement_id}."
                ),
                nonce=f"{settlement.settlement_id}:validator-settlement-fee",
                metadata={
                    "reward_token_code": active_money_policy.reward_token_code,
                    "funding_source": settlement.funding_source,
                    "source_wallet_id": settlement.source_wallet_id,
                    "source_wallet_address": settlement.source_wallet_address,
                    "committee_validator_ids": list(
                        settlement.committee_validator_ids or []
                    ),
                    "committee_quorum_bond_atomic": int(
                        settlement.committee_quorum_bond_atomic or 0
                    ),
                    "distribution": "legacy_pool",
                },
                chain_id=active_money_policy.chain_network.value,
            )
            _record_if_missing(tx)

    ai_development_fee = max(0, int(settlement.ai_development_fee_atomic or 0))
    ai_address = str(settlement.ai_development_address or "").strip()
    if ai_development_fee > 0 and ai_address:
        balance_audit_targets.append(
            {
                "role": "ai_development_fund",
                "address": ai_address,
                "wallet_id": settlement.ai_development_wallet_id,
                "expected_delta_atomic": ai_development_fee,
            }
        )
        tx = make_chain_transaction(
            tx_type="ai_development_fee_credit",
            address=ai_address,
            delta_atomic=ai_development_fee,
            settlement_id=settlement.settlement_id,
            wallet_id=settlement.ai_development_wallet_id,
            note=f"AI development fee from finalized settlement {settlement.settlement_id}.",
            nonce=f"{settlement.settlement_id}:ai-development-fee",
            metadata={"reward_token_code": active_money_policy.reward_token_code},
            chain_id=active_money_policy.chain_network.value,
        )
        _record_if_missing(tx)

    payouts = list_worker_payouts(settlement_id=settlement.settlement_id, policy=policy)
    updated_payouts = list_worker_payouts(policy=policy)
    changed = False
    for payout in payouts:
        recipient_address = payout.recipient_address or resolve_worker_reward_address(
            payout.node_id,
            policy,
        )
        if not recipient_address:
            continue
        if payout.recipient_address != recipient_address:
            for index, existing in enumerate(updated_payouts):
                if existing.payout_id == payout.payout_id:
                    updated_payouts[index].recipient_address = recipient_address
                    changed = True
                    break
        balance_audit_targets.append(
            {
                "role": "worker_reward",
                "address": recipient_address,
                "wallet_id": payout.credited_wallet_id,
                "node_id": payout.node_id,
                "payout_id": payout.payout_id,
                "expected_delta_atomic": max(0, int(payout.reward_atomic)),
            }
        )
        tx = make_chain_transaction(
            tx_type="worker_reward_credit",
            address=recipient_address,
            delta_atomic=max(0, int(payout.reward_atomic)),
            settlement_id=settlement.settlement_id,
            payout_id=payout.payout_id,
            wallet_id=payout.credited_wallet_id,
            note=f"Worker reward for node {payout.node_id}.",
            nonce=f"{settlement.settlement_id}:worker-reward:{payout.payout_id}",
            metadata={
                "node_id": payout.node_id,
                "runner_id": payout.runner_id,
                "model_id": payout.model_id,
                "share_bps": payout.share_bps,
                "reward_token_code": payout.reward_token_code
                or active_money_policy.reward_token_code,
            },
            chain_id=active_money_policy.chain_network.value,
        )
        _record_if_missing(tx)
    if changed:
        save_worker_payouts(updated_payouts, policy)
    if pending_transactions:
        append_chain_block(
            pending_transactions,
            validator_id=settlement.applied_by_validator_id,
            policy=policy,
        )
    _record_settlement_chain_balance_audit(
        settlement,
        balance_audit_targets,
        recorded_transactions=recorded,
        policy=policy,
    )
    return recorded


def record_chain_entries_for_finalized_settlements(
    *,
    policy: WalletPolicy | None = None,
    money_policy: MoneyPolicy | None = None,
    only_if_chain_recorded: bool = False,
) -> int:
    recorded = 0
    for settlement in list_settlements(policy):
        if settlement.status == "applied" or settlement.applied_at:
            if only_if_chain_recorded and not chain_settlement_history(
                settlement.settlement_id,
                policy=policy,
                limit=1,
            ):
                continue
            recorded += record_chain_entries_for_settlement(
                settlement,
                policy=policy,
                money_policy=money_policy,
            )
    return recorded


def _settlement_execution_pricing_audit(settlement: SettlementRecord) -> dict[str, Any]:
    audit = dict(settlement.balance_audit or {})
    execution = audit.get("execution")
    if not isinstance(execution, dict):
        return {}
    pricing = execution.get("pricing")
    return pricing if isinstance(pricing, dict) else {}


def _settlement_reserved_compute_atomic(settlement: SettlementRecord) -> int | None:
    pricing = _settlement_execution_pricing_audit(settlement)
    return _optional_int(pricing.get("reserved_compute_cost_atomic"))


def _settlement_reservation_surplus_atomic(settlement: SettlementRecord) -> int:
    pricing = _settlement_execution_pricing_audit(settlement)
    explicit_surplus = _optional_int(pricing.get("reservation_surplus_atomic"))
    if explicit_surplus is not None:
        return max(0, int(explicit_surplus))
    reserved = _settlement_reserved_compute_atomic(settlement)
    if reserved is None:
        return 0
    return max(0, int(reserved) - int(settlement.compute_cost_atomic or 0))


def _record_settlement_chain_balance_audit(
    settlement: SettlementRecord,
    targets: list[dict[str, Any]],
    *,
    recorded_transactions: int,
    policy: WalletPolicy | None = None,
) -> SettlementRecord:
    entries: list[dict[str, Any]] = []
    for target in targets:
        address = str(target.get("address") or "").strip()
        if not address:
            continue
        expected_delta_atomic = int(target.get("expected_delta_atomic") or 0)
        balance_after_atomic = chain_balance_atomic(address, policy)
        balance_before_atomic = balance_after_atomic - expected_delta_atomic
        entries.append(
            {
                "role": target.get("role"),
                "address": address,
                "wallet_id": target.get("wallet_id"),
                "validator_id": target.get("validator_id"),
                "node_id": target.get("node_id"),
                "payout_id": target.get("payout_id"),
                "balance_before_atomic": balance_before_atomic,
                "balance_after_atomic": balance_after_atomic,
                "expected_delta_atomic": expected_delta_atomic,
                "actual_delta_atomic": balance_after_atomic - balance_before_atomic,
                "delta_matches_expected": (
                    balance_after_atomic - balance_before_atomic
                    == expected_delta_atomic
                ),
            }
        )
    if not entries:
        return settlement
    audit = dict(settlement.balance_audit or {})
    audit["chain_balances"] = {
        "schema_version": 1,
        "recorded_transactions": int(recorded_transactions),
        "addresses": entries,
        "all_expected_deltas_match": all(
            bool(item["delta_matches_expected"]) for item in entries
        ),
    }
    return _save_settlement_balance_audit(settlement, audit, policy=policy)


def _initial_settlement_balance_audit(
    *,
    source_wallet_id: str,
    source_wallet_address: str,
    decision: FundingDecision,
    money_policy: MoneyPolicy,
    validator_sync_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fee_quote = decision.fee_quote
    source_wallet_debit = max(
        0, decision.wallet_before_atomic - decision.wallet_after_atomic
    )
    reserve_debit = max(
        0, decision.reserve_before_atomic - decision.reserve_after_atomic
    )
    reward_plus_fees = (
        int(fee_quote.worker_reward_atomic)
        + int(fee_quote.settlement_fee_atomic)
        + int(fee_quote.ai_development_fee_atomic)
    )
    audit = {
        "schema_version": 1,
        "network": money_policy.chain_network.value,
        "reward_token_code": money_policy.reward_token_code,
        "source_wallet_id": source_wallet_id,
        "source_wallet_address": source_wallet_address,
        "funding_source": decision.funding_source.value if decision.funding_source else None,
        "quote": {
            "reserve_before_atomic": int(decision.reserve_before_atomic),
            "reserve_after_atomic": int(decision.reserve_after_atomic),
            "wallet_before_atomic": int(decision.wallet_before_atomic),
            "wallet_after_atomic": int(decision.wallet_after_atomic),
        },
        "debits": {
            "source_wallet_debit_atomic": int(source_wallet_debit),
            "reserve_debit_atomic": int(reserve_debit),
            "reserve_limit_identity_keys": list(
                decision.reserve_limit_identity_keys or []
            ),
            "reserve_client_ip_hash": decision.reserve_client_ip_hash,
        },
        "fees": {
            "compute_cost_atomic": int(fee_quote.compute_cost_atomic),
            "tx_fee_atomic": int(fee_quote.tx_fee_atomic),
            "settlement_fee_atomic": int(fee_quote.settlement_fee_atomic),
            "ai_development_fee_atomic": int(fee_quote.ai_development_fee_atomic),
            "worker_reward_atomic": int(fee_quote.worker_reward_atomic),
            "worker_reward_plus_fees_atomic": int(reward_plus_fees),
            "worker_plus_validator_plus_ai_matches_compute": (
                reward_plus_fees == int(fee_quote.compute_cost_atomic)
            ),
        },
    }
    if validator_sync_audit is not None:
        audit["pre_settlement_validator_sync"] = _json_safe_audit_section(
            validator_sync_audit
        )
    return audit


def _with_applied_balance_audit(
    settlement: SettlementRecord,
    *,
    reserve_before_atomic: int,
    reserve_after_atomic: int,
    source_wallet_before_atomic: int | None,
    source_wallet_after_atomic: int | None,
) -> dict[str, Any]:
    audit = dict(settlement.balance_audit or {})
    quote = dict(audit.get("quote") or {})
    debits = dict(audit.get("debits") or {})
    audit["applied"] = {
        "reserve_before_atomic": int(reserve_before_atomic),
        "reserve_after_atomic": int(reserve_after_atomic),
        "source_wallet_before_atomic": (
            int(source_wallet_before_atomic)
            if source_wallet_before_atomic is not None
            else None
        ),
        "source_wallet_after_atomic": (
            int(source_wallet_after_atomic)
            if source_wallet_after_atomic is not None
            else None
        ),
        "reserve_delta_atomic": int(reserve_after_atomic) - int(reserve_before_atomic),
        "source_wallet_delta_atomic": (
            int(source_wallet_after_atomic) - int(source_wallet_before_atomic)
            if source_wallet_before_atomic is not None
            and source_wallet_after_atomic is not None
            else None
        ),
        "reserve_delta_matches_debit": (
            int(reserve_before_atomic) - int(reserve_after_atomic)
            == int(debits.get("reserve_debit_atomic") or 0)
        ),
        "source_wallet_delta_matches_debit": (
            source_wallet_before_atomic is None
            or source_wallet_after_atomic is None
            or (
                int(source_wallet_before_atomic) - int(source_wallet_after_atomic)
                == int(debits.get("source_wallet_debit_atomic") or 0)
            )
        ),
        "quote_reserve_after_matches_applied": (
            not quote
            or int(quote.get("reserve_after_atomic") or 0)
            == int(reserve_after_atomic)
        ),
        "quote_wallet_after_matches_applied": (
            not quote
            or source_wallet_after_atomic is None
            or int(quote.get("wallet_after_atomic") or 0)
            == int(source_wallet_after_atomic)
        ),
    }
    return audit


def _payout_sort_key(item: WorkerPayoutRecord) -> tuple[bool, int, str, str]:
    return (
        item.layer_start is None,
        int(item.layer_start) if item.layer_start is not None else 10**9,
        item.node_id,
        item.runner_id or "",
    )


def _deterministic_worker_payout_split_audit(
    total_reward_atomic: int,
    payouts: list[WorkerPayoutRecord],
) -> list[dict[str, Any]]:
    if total_reward_atomic <= 0 or not payouts:
        return []

    total_weight = sum(max(int(item.layer_count or 1), 1) for item in payouts)
    if total_weight <= 0:
        total_weight = len(payouts)

    raw_parts: list[tuple[WorkerPayoutRecord, int, int]] = []
    allocated = 0
    for item in payouts:
        weight = max(int(item.layer_count or 1), 1)
        raw_value = int(total_reward_atomic) * weight
        expected_reward_atomic = raw_value // total_weight
        allocated += expected_reward_atomic
        raw_parts.append((item, expected_reward_atomic, raw_value % total_weight))

    remainder = int(total_reward_atomic) - allocated
    raw_parts.sort(key=lambda entry: (-entry[2], entry[0].node_id))
    adjusted: list[tuple[WorkerPayoutRecord, int]] = []
    for item, expected_reward_atomic, _remainder in raw_parts:
        adjusted.append((item, expected_reward_atomic))
    for index in range(remainder):
        item, expected_reward_atomic = adjusted[index % len(adjusted)]
        adjusted[index % len(adjusted)] = (item, expected_reward_atomic + 1)

    adjusted.sort(key=lambda entry: _payout_sort_key(entry[0]))
    return [
        {
            "payout_id": item.payout_id,
            "node_id": item.node_id,
            "runner_id": item.runner_id,
            "layer_start": item.layer_start,
            "layer_end": item.layer_end,
            "layer_count": int(item.layer_count),
            "share_bps": int(item.share_bps),
            "expected_reward_atomic": int(expected_reward_atomic),
            "recorded_reward_atomic": int(item.reward_atomic),
            "reward_matches_expected": (
                int(item.reward_atomic) == int(expected_reward_atomic)
            ),
        }
        for item, expected_reward_atomic in adjusted
    ]


def _settlement_worker_payout_accounting(
    settlement: SettlementRecord,
    payouts: list[WorkerPayoutRecord],
    *,
    money_policy: MoneyPolicy | None = None,
) -> dict[str, Any]:
    active_money_policy = money_policy or MoneyPolicy()
    expected_worker_reward_atomic = max(0, int(settlement.worker_reward_atomic or 0))
    recorded_worker_reward_atomic = sum(
        max(0, int(item.reward_atomic or 0)) for item in payouts
    )
    settlement_fee_atomic = max(0, int(settlement.settlement_fee_atomic or 0))
    ai_development_fee_atomic = max(0, int(settlement.ai_development_fee_atomic or 0))
    compute_cost_atomic = max(0, int(settlement.compute_cost_atomic or 0))
    expected_settlement_fee_atomic = (
        compute_cost_atomic * int(active_money_policy.validator_settlement_fee_bps)
    ) // 10_000
    expected_ai_development_fee_atomic = (
        compute_cost_atomic * int(active_money_policy.ai_development_fee_bps)
    ) // 10_000
    worker_plus_fees_atomic = (
        recorded_worker_reward_atomic
        + settlement_fee_atomic
        + ai_development_fee_atomic
    )
    deterministic_split = _deterministic_worker_payout_split_audit(
        expected_worker_reward_atomic,
        payouts,
    )
    return {
        "schema_version": 1,
        "reward_token_code": settlement.reward_token_code
        or active_money_policy.reward_token_code,
        "settlement_id": settlement.settlement_id,
        "payout_count": len(payouts),
        "payout_ids": sorted(item.payout_id for item in payouts),
        "participant_node_ids": sorted({item.node_id for item in payouts}),
        "deterministic_split": deterministic_split,
        "expected_worker_reward_atomic": expected_worker_reward_atomic,
        "recorded_worker_reward_atomic": recorded_worker_reward_atomic,
        "settlement_fee_atomic": settlement_fee_atomic,
        "ai_development_fee_atomic": ai_development_fee_atomic,
        "expected_settlement_fee_atomic": expected_settlement_fee_atomic,
        "expected_ai_development_fee_atomic": expected_ai_development_fee_atomic,
        "compute_cost_atomic": compute_cost_atomic,
        "worker_plus_validator_plus_ai_atomic": worker_plus_fees_atomic,
        "settlement_fee_matches_policy": (
            settlement_fee_atomic == expected_settlement_fee_atomic
        ),
        "ai_development_fee_matches_policy": (
            ai_development_fee_atomic == expected_ai_development_fee_atomic
        ),
        "deterministic_split_matches": all(
            bool(item["reward_matches_expected"]) for item in deterministic_split
        )
        if deterministic_split
        else expected_worker_reward_atomic == 0,
        "worker_reward_matches_payouts": (
            recorded_worker_reward_atomic == expected_worker_reward_atomic
        ),
        "worker_plus_validator_plus_ai_matches_compute": (
            worker_plus_fees_atomic == compute_cost_atomic
        ),
    }


def validate_settlement_accounting(
    settlement: SettlementRecord,
    payouts: list[WorkerPayoutRecord],
    *,
    money_policy: MoneyPolicy | None = None,
) -> list[str]:
    audit = _settlement_worker_payout_accounting(
        settlement,
        payouts,
        money_policy=money_policy,
    )
    errors: list[str] = []
    if not audit["worker_reward_matches_payouts"]:
        errors.append(
            "Worker payout total does not match settlement worker reward "
            f"({audit['recorded_worker_reward_atomic']} != "
            f"{audit['expected_worker_reward_atomic']})."
        )
    if not audit["deterministic_split_matches"]:
        errors.append(
            "Worker payout split does not match deterministic layer-share calculation."
        )
    if not audit["worker_plus_validator_plus_ai_matches_compute"]:
        errors.append(
            "Worker payouts plus validator and AI fees do not match compute cost "
            f"({audit['worker_plus_validator_plus_ai_atomic']} != "
            f"{audit['compute_cost_atomic']})."
        )
    if not audit["settlement_fee_matches_policy"]:
        errors.append(
            "Validator settlement fee does not match policy "
            f"({audit['settlement_fee_atomic']} != "
            f"{audit['expected_settlement_fee_atomic']})."
        )
    if not audit["ai_development_fee_matches_policy"]:
        errors.append(
            "AI development fee does not match policy "
            f"({audit['ai_development_fee_atomic']} != "
            f"{audit['expected_ai_development_fee_atomic']})."
        )
    return errors


def _with_worker_payout_accounting_audit(
    settlement: SettlementRecord,
    payouts: list[WorkerPayoutRecord],
    *,
    money_policy: MoneyPolicy | None = None,
) -> dict[str, Any]:
    audit = dict(settlement.balance_audit or {})
    audit["payouts"] = _settlement_worker_payout_accounting(
        settlement,
        payouts,
        money_policy=money_policy,
    )
    return audit


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _worker_payout_execution_audit_item(item: WorkerPayoutRecord) -> dict[str, Any]:
    return {
        "payout_id": item.payout_id,
        "node_id": item.node_id,
        "runner_id": item.runner_id,
        "model_id": item.model_id,
        "layer_start": item.layer_start,
        "layer_end": item.layer_end,
        "layer_count": int(item.layer_count),
        "share_bps": int(item.share_bps),
        "reward_atomic": int(item.reward_atomic),
        "reward_token_code": item.reward_token_code,
        "recipient_address": item.recipient_address,
        "credited_wallet_id": item.credited_wallet_id,
        "status": item.status,
    }


def record_settlement_execution_audit(
    *,
    settlement_id: str,
    receipt_id: str,
    job_id: str,
    model_id: str,
    execution_model_id: str,
    pricing_mode: str,
    pricing_basis: str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    reserved_prompt_tokens: int | None = None,
    reserved_completion_tokens: int | None = None,
    reserved_compute_cost_atomic: int | None = None,
    actual_compute_cost_atomic: int | None = None,
    reservation_surplus_atomic: int = 0,
    usage_priced: bool = False,
    token_usage_source: str | None = None,
    token_usage_audit: dict[str, Any] | None = None,
    network_audit: dict[str, Any] | None = None,
    worker_payouts: list[WorkerPayoutRecord] | None = None,
    policy: WalletPolicy | None = None,
) -> SettlementRecord | None:
    settlement = resolve_settlement(settlement_id, policy)
    if settlement is None:
        return None

    payouts = list(worker_payouts or [])
    audit = dict(settlement.balance_audit or {})
    audit["execution"] = {
        "schema_version": 1,
        "receipt_id": receipt_id,
        "job_id": job_id,
        "model_id": model_id,
        "execution_model_id": execution_model_id,
        "pricing": {
            "pricing_mode": pricing_mode,
            "pricing_basis": pricing_basis,
            "usage_priced": bool(usage_priced),
            "reserved_compute_cost_atomic": _optional_int(
                reserved_compute_cost_atomic
            ),
            "actual_compute_cost_atomic": _optional_int(actual_compute_cost_atomic),
            "reservation_surplus_atomic": int(reservation_surplus_atomic or 0),
        },
        "usage": {
            "prompt_tokens": _optional_int(prompt_tokens),
            "completion_tokens": _optional_int(completion_tokens),
            "total_tokens": _optional_int(total_tokens),
            "reserved_prompt_tokens": _optional_int(reserved_prompt_tokens),
            "reserved_completion_tokens": _optional_int(reserved_completion_tokens),
            "source": str(token_usage_source or "").strip() or None,
        },
        "token_usage_audit": dict(token_usage_audit or {}),
        "route": dict(network_audit or {}),
        "workers": {
            "count": len(payouts),
            "participants": [
                _worker_payout_execution_audit_item(item) for item in payouts
            ],
        },
    }
    return _save_settlement_balance_audit(settlement, audit, policy=policy)


def settlement_envelope_payload(
    settlement: SettlementRecord,
    *,
    money_policy: MoneyPolicy | None = None,
) -> dict[str, Any]:
    active_money_policy = money_policy or MoneyPolicy()
    audit = dict(settlement.balance_audit or {})
    return {
        "schema_version": 1,
        "chain_network": active_money_policy.chain_network.value,
        "reward_token_code": settlement.reward_token_code
        or active_money_policy.reward_token_code,
        "settlement": {
            "settlement_id": settlement.settlement_id,
            "created_at": settlement.created_at,
            "source_wallet_id": settlement.source_wallet_id,
            "source_wallet_address": normalize_address(
                settlement.source_wallet_address
            ),
            "funding_source": settlement.funding_source,
            "compute_cost_atomic": int(settlement.compute_cost_atomic),
            "tx_fee_atomic": int(settlement.tx_fee_atomic),
            "settlement_fee_atomic": int(settlement.settlement_fee_atomic),
            "ai_development_fee_atomic": int(
                settlement.ai_development_fee_atomic or 0
            ),
            "worker_reward_atomic": int(settlement.worker_reward_atomic),
            "source_wallet_debit_atomic": int(
                settlement.source_wallet_debit_atomic or 0
            ),
            "reserve_debit_atomic": int(settlement.reserve_debit_atomic or 0),
            "reserve_limit_identity_keys": list(
                settlement.reserve_limit_identity_keys or []
            ),
            "reserve_client_ip_hash": settlement.reserve_client_ip_hash,
            "ai_development_wallet_id": settlement.ai_development_wallet_id,
            "ai_development_address": (
                normalize_address(settlement.ai_development_address)
                if settlement.ai_development_address
                else None
            ),
        },
        "committee": {
            "selection_seed": settlement.committee_selection_seed,
            "target_size": int(settlement.committee_target_size),
            "selection_mode": settlement.committee_selection_mode,
            "validator_ids": list(settlement.committee_validator_ids or []),
            "bonded_atomic_by_validator_id": dict(
                settlement.committee_bonded_atomic_by_validator_id or {}
            ),
            "total_bonded_atomic": int(settlement.committee_total_bonded_atomic or 0),
            "quorum_bond_atomic": int(settlement.committee_quorum_bond_atomic or 0),
        },
        "quote": _json_safe_audit_section(audit.get("quote")),
        "fees": _json_safe_audit_section(audit.get("fees")),
        "debits": _json_safe_audit_section(audit.get("debits")),
        "execution": _json_safe_audit_section(audit.get("execution")),
        "payouts": _json_safe_audit_section(audit.get("payouts")),
    }


def settlement_envelope_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_payload(payload).encode("utf-8")).hexdigest()


def sign_settlement_envelope(
    settlement_id: str,
    *,
    policy: WalletPolicy | None = None,
    money_policy: MoneyPolicy | None = None,
) -> SettlementRecord | None:
    settlement = resolve_settlement(settlement_id, policy)
    if settlement is None:
        return None

    payload = settlement_envelope_payload(settlement, money_policy=money_policy)
    payload_hash = settlement_envelope_hash(payload)
    audit = dict(settlement.balance_audit or {})
    envelope: dict[str, Any] = {
        "schema_version": 1,
        "payload_hash": payload_hash,
        "source_wallet_id": settlement.source_wallet_id,
        "source_wallet_address": normalize_address(settlement.source_wallet_address),
        "required": False,
        "scheme": None,
        "address_scheme": None,
        "public_key_b64": None,
        "signature_b64": None,
        "signature_valid": False,
        "status": "unsigned",
        "signed_at": None,
    }

    source_wallet = find_wallet_by_id(settlement.source_wallet_id, policy)
    if source_wallet is None:
        envelope["status"] = "source_wallet_not_found"
        audit["signed_envelope"] = envelope
        return _save_settlement_balance_audit(settlement, audit, policy=policy)

    envelope["address_scheme"] = source_wallet.address_scheme
    envelope["scheme"] = source_wallet.signing_scheme
    wallet_uses_ed25519 = (
        source_wallet.signing_scheme == SIGNING_SCHEME_ED25519
        and source_wallet.address_scheme == ADDRESS_SCHEME_ED25519
    )
    wallet_uses_hybrid = (
        source_wallet.signing_scheme == SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65
        and source_wallet.address_scheme
        in {
            *HYBRID_ADDRESS_SCHEMES,
            ADDRESS_SCHEME_FIXED_WALLET_ID_SHA256,
        }
    )
    if not wallet_uses_ed25519 and not wallet_uses_hybrid:
        envelope["status"] = "legacy_wallet_unsigned"
        audit["signed_envelope"] = envelope
        return _save_settlement_balance_audit(settlement, audit, policy=policy)

    envelope["required"] = True
    signer = load_unlocked_wallet_signing_material(source_wallet, policy)
    if signer is None:
        envelope["status"] = "missing_unlocked_signer"
        audit["signed_envelope"] = envelope
        return _save_settlement_balance_audit(settlement, audit, policy=policy)

    public_key_b64 = str(signer.get("public_key_b64") or "")
    if public_key_b64 != source_wallet.public_key_b64:
        envelope["status"] = "public_key_mismatch"
        audit["signed_envelope"] = envelope
        return _save_settlement_balance_audit(settlement, audit, policy=policy)
    if wallet_uses_hybrid and str(signer.get("pq_public_key_b64") or "") != str(
        source_wallet.pq_public_key_b64 or ""
    ):
        envelope["status"] = "pq_public_key_mismatch"
        audit["signed_envelope"] = envelope
        return _save_settlement_balance_audit(settlement, audit, policy=policy)

    if normalize_address(source_wallet.address) != normalize_address(
        settlement.source_wallet_address
    ):
        envelope["status"] = "source_wallet_address_mismatch"
        audit["signed_envelope"] = envelope
        return _save_settlement_balance_audit(settlement, audit, policy=policy)

    signing_seed_b64 = str(signer.get("signing_seed_b64") or "")
    signature_b64 = sign_payload_b64(decode_bytes(signing_seed_b64), payload)
    envelope["public_key_b64"] = public_key_b64
    envelope["signature_b64"] = signature_b64
    if wallet_uses_hybrid:
        pq_public_key_b64 = str(signer.get("pq_public_key_b64") or "")
        pq_private_key_b64 = str(signer.get("pq_private_key_b64") or "")
        envelope["pq_scheme"] = SIGNING_SCHEME_ML_DSA_65
        envelope["pq_public_key_b64"] = pq_public_key_b64
        envelope["pq_signature_b64"] = sign_payload_mldsa65_b64(
            pq_private_key_b64,
            payload,
        )
        envelope["signature_valid"] = verify_hybrid_payload_signature(
            ed25519_public_key_b64=public_key_b64,
            ed25519_signature_b64=signature_b64,
            pq_public_key_b64=pq_public_key_b64,
            pq_signature_b64=str(envelope["pq_signature_b64"]),
            payload=payload,
        )
    else:
        envelope["signature_valid"] = verify_payload_signature(
            public_key_b64=public_key_b64,
            signature_b64=signature_b64,
            payload=payload,
        )
    envelope["status"] = "signed" if envelope["signature_valid"] else "invalid"
    envelope["signed_at"] = _now_iso()
    audit["signed_envelope"] = envelope
    return _save_settlement_balance_audit(settlement, audit, policy=policy)


def verify_settlement_envelope(
    settlement: SettlementRecord,
    *,
    money_policy: MoneyPolicy | None = None,
) -> tuple[bool, str | None]:
    audit = dict(settlement.balance_audit or {})
    envelope = audit.get("signed_envelope")
    if not isinstance(envelope, dict):
        return False, "settlement signed envelope is missing"
    if not envelope.get("required"):
        return True, None
    payload = settlement_envelope_payload(settlement, money_policy=money_policy)
    expected_hash = settlement_envelope_hash(payload)
    if envelope.get("payload_hash") != expected_hash:
        return False, "settlement signed envelope payload hash does not match"
    public_key_b64 = str(envelope.get("public_key_b64") or "")
    signature_b64 = str(envelope.get("signature_b64") or "")
    if not public_key_b64 or not signature_b64:
        return False, "settlement signed envelope signature is missing"
    if envelope.get("scheme") == SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65:
        if str(envelope.get("pq_scheme") or "") != SIGNING_SCHEME_ML_DSA_65:
            return False, "settlement signed envelope PQ signature scheme is unsupported"
        pq_public_key_b64 = str(envelope.get("pq_public_key_b64") or "")
        pq_signature_b64 = str(envelope.get("pq_signature_b64") or "")
        if not pq_public_key_b64 or not pq_signature_b64:
            return False, "settlement signed envelope PQ signature is missing"
        if not verify_hybrid_payload_signature(
            ed25519_public_key_b64=public_key_b64,
            ed25519_signature_b64=signature_b64,
            pq_public_key_b64=pq_public_key_b64,
            pq_signature_b64=pq_signature_b64,
            payload=payload,
        ):
            return False, "settlement signed envelope hybrid signature is invalid"
        return True, None
    if not verify_payload_signature(
        public_key_b64=public_key_b64,
        signature_b64=signature_b64,
        payload=payload,
    ):
        return False, "settlement signed envelope signature is invalid"
    return True, None


def _json_safe_audit_section(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError):
        return str(value) if value is not None else None


def _sync_result_int_attr(result: Any, name: str) -> int:
    value = getattr(result, name, 0)
    return value if isinstance(value, int) else 0


def _sync_result_list_attr(result: Any, name: str) -> list[Any]:
    value = getattr(result, name, [])
    return list(value) if isinstance(value, (list, tuple)) else []


def _validator_set_sync_audit(result: Any) -> dict[str, Any]:
    attempted_peers = _sync_result_int_attr(result, "attempted_peers")
    successful_peers = _sync_result_int_attr(result, "successful_peers")
    failed_peers = _sync_result_int_attr(result, "failed_peers")
    if failed_peers <= 0 and attempted_peers > successful_peers:
        failed_peers = attempted_peers - successful_peers
    if failed_peers > 0 and successful_peers > 0:
        status = "partial"
    elif failed_peers > 0:
        status = "failed"
    elif attempted_peers <= 0:
        status = "no_peers"
    else:
        status = "applied"
    return {
        "status": status,
        "attemptedPeers": attempted_peers,
        "successfulPeers": successful_peers,
        "failedPeers": failed_peers,
        "importedRecords": _sync_result_int_attr(result, "imported_records"),
        "peerUrls": _sync_result_list_attr(result, "peer_urls"),
        "failedPeerUrls": _sync_result_list_attr(result, "failed_peer_urls"),
        "peerErrors": _sync_result_list_attr(result, "peer_errors"),
    }


def _save_settlement_balance_audit(
    settlement: SettlementRecord,
    balance_audit: dict[str, Any],
    *,
    policy: WalletPolicy | None = None,
) -> SettlementRecord:
    settlements = list_settlements(policy)
    for index, item in enumerate(settlements):
        if item.settlement_id != settlement.settlement_id:
            continue
        item.balance_audit = balance_audit
        settlements[index] = item
        save_settlements(settlements, policy)
        return item
    return settlement


def record_funding_settlement(
    *,
    source_wallet_id: str,
    source_wallet_address: str,
    decision: FundingDecision,
    note: str | None = None,
    money_policy: MoneyPolicy | None = None,
    policy: WalletPolicy | None = None,
    state_payload: dict[str, Any] | None = None,
    cai_url: str | None = None,
    CAI_url: str | None = None,
) -> SettlementRecord:
    if decision.funding_source is None:
        raise ValueError("Funding decision has no funding source.")

    active_money_policy = money_policy or MoneyPolicy()
    validator_sync_audit: dict[str, Any] | None = None
    if state_payload is not None:
        try:
            sync_result = sync_validator_set_from_cai_peers(
                state_payload=state_payload,
                cai_url=cai_url,
                CAI_url=CAI_url,
                policy=policy,
            )
            validator_sync_audit = _validator_set_sync_audit(sync_result)
        except Exception as exc:
            validator_sync_audit = {
                "status": "failed",
                **_error_payload(exc),
            }
            _log_best_effort_failure(
                "funding settlement pre-settlement validator sync",
                exc,
            )
    settlements = list_settlements(policy)
    settlement_id = secrets.token_hex(12)
    committee = select_validator_committee_snapshot(
        selection_seed=settlement_id,
        money_policy=active_money_policy,
        policy=policy,
    )
    record = SettlementRecord(
        settlement_id=settlement_id,
        created_at=_now_iso(),
        source_wallet_id=source_wallet_id,
        source_wallet_address=source_wallet_address,
        funding_source=decision.funding_source.value,
        compute_cost_atomic=decision.fee_quote.compute_cost_atomic,
        tx_fee_atomic=decision.fee_quote.tx_fee_atomic,
        settlement_fee_atomic=decision.fee_quote.settlement_fee_atomic,
        worker_reward_atomic=decision.fee_quote.worker_reward_atomic,
        reward_token_code=active_money_policy.reward_token_code,
        ai_development_fee_atomic=decision.fee_quote.ai_development_fee_atomic,
        ai_development_wallet_id=active_money_policy.ai_development_wallet_id,
        ai_development_address=active_money_policy.ai_development_address,
        source_wallet_debit_atomic=max(
            0, decision.wallet_before_atomic - decision.wallet_after_atomic
        ),
        reserve_debit_atomic=max(
            0, decision.reserve_before_atomic - decision.reserve_after_atomic
        ),
        reserve_limit_identity_keys=list(decision.reserve_limit_identity_keys or []),
        reserve_client_ip_hash=decision.reserve_client_ip_hash,
        committee_selection_seed=settlement_id,
        committee_target_size=min(
            active_money_policy.validator_committee_target_size,
            len(committee.validator_ids),
        ),
        committee_selection_mode=active_money_policy.validator_committee_selection_mode,
        committee_validator_ids=list(committee.validator_ids),
        committee_bonded_atomic_by_validator_id=dict(committee.bonded_atomic_by_validator_id),
        committee_total_bonded_atomic=committee.total_bonded_atomic,
        committee_quorum_bond_atomic=committee.quorum_bond_atomic,
        balance_audit=_initial_settlement_balance_audit(
            source_wallet_id=source_wallet_id,
            source_wallet_address=source_wallet_address,
            decision=decision,
            money_policy=active_money_policy,
            validator_sync_audit=validator_sync_audit,
        ),
        note=note,
    )
    settlements.append(record)
    save_settlements(settlements, policy)
    return record


def ensure_settlement_committee(
    settlement_id: str,
    *,
    policy: WalletPolicy | None = None,
    money_policy: MoneyPolicy | None = None,
) -> SettlementRecord | None:
    active_money_policy = money_policy or MoneyPolicy()
    settlements = list_settlements(policy)
    for index, item in enumerate(settlements):
        if item.settlement_id != settlement_id:
            continue
        if item.committee_validator_ids and item.committee_quorum_bond_atomic > 0:
            return item

        selection_seed = item.committee_selection_seed or item.settlement_id
        committee = select_validator_committee_snapshot(
            selection_seed=selection_seed,
            policy=policy,
            money_policy=active_money_policy,
            target_size=(
                item.committee_target_size if int(item.committee_target_size or 0) > 0 else None
            ),
        )
        if not committee.validator_ids:
            return item

        item.committee_selection_seed = selection_seed
        item.committee_target_size = min(
            max(1, int(item.committee_target_size or 0) or len(committee.validator_ids)),
            len(committee.validator_ids),
        )
        item.committee_selection_mode = active_money_policy.validator_committee_selection_mode
        item.committee_validator_ids = list(committee.validator_ids)
        item.committee_bonded_atomic_by_validator_id = dict(
            committee.bonded_atomic_by_validator_id
        )
        item.committee_total_bonded_atomic = committee.total_bonded_atomic
        item.committee_quorum_bond_atomic = committee.quorum_bond_atomic
        settlements[index] = item
        save_settlements(settlements, policy)
        return (
            sign_settlement_envelope(
                item.settlement_id,
                policy=policy,
                money_policy=active_money_policy,
            )
            or item
        )
    return None


def reset_retryable_settlement_rejection(
    settlement_id: str,
    *,
    policy: WalletPolicy | None = None,
    money_policy: MoneyPolicy | None = None,
) -> SettlementRecord | None:
    settlement = resolve_settlement(settlement_id, policy)
    if settlement is None or settlement.status != "rejected":
        return settlement

    settlement_attestations = list_attestations(
        settlement_id=settlement_id,
        policy=policy,
    )
    if not settlement_attestations or any(item.accepted for item in settlement_attestations):
        return settlement
    if any(
        str(item.note or "").strip().lower()
        not in _RETRYABLE_SETTLEMENT_ENVELOPE_REJECTION_NOTES
        for item in settlement_attestations
    ):
        return settlement

    signed = sign_settlement_envelope(
        settlement_id,
        policy=policy,
        money_policy=money_policy,
    )
    signed_envelope = (
        dict(getattr(signed, "balance_audit", {}) or {}).get("signed_envelope")
        if signed is not None
        else None
    )
    if not isinstance(signed_envelope, dict) or signed_envelope.get("status") != "signed":
        return settlement

    remaining_attestations = [
        item
        for item in list_attestations(policy=policy)
        if item.settlement_id != settlement_id
    ]
    save_attestations(remaining_attestations, policy)

    settlements = list_settlements(policy)
    for index, item in enumerate(settlements):
        if item.settlement_id != settlement_id:
            continue
        item.accepted_attestations = 0
        item.rejected_attestations = 0
        item.accepted_bond_atomic = 0
        item.rejected_bond_atomic = 0
        item.status = "pending"
        item.applied_at = None
        item.applied_by_validator_id = None
        settlements[index] = item
        save_settlements(settlements, policy)
        return (
            sign_settlement_envelope(
                settlement_id,
                policy=policy,
                money_policy=money_policy,
            )
            or item
        )
    return settlement


def record_validator_attestation(
    *,
    settlement_id: str,
    validator_id: str,
    accepted: bool = True,
    note: str | None = None,
    policy: WalletPolicy | None = None,
    apply_on_finalize: bool = True,
) -> ValidatorAttestation:
    settlement = resolve_settlement(settlement_id, policy)
    committee_validator_ids = list(getattr(settlement, "committee_validator_ids", []) or [])
    if committee_validator_ids and validator_id not in committee_validator_ids:
        raise ValueError("Validator is not a member of this settlement committee.")

    existing = [
        item
        for item in list_attestations(settlement_id=settlement_id, policy=policy)
        if item.validator_id == validator_id
    ]
    for item in existing:
        # The attestation decision is the slash-relevant part. Replays with the
        # same accepted/rejected position must stay idempotent even if the
        # caller provides a different informational note later.
        if item.accepted == accepted:
            settlement = refresh_settlement_finality(
                settlement_id=settlement_id, policy=policy
            )
            if (
                apply_on_finalize
                and settlement is not None
                and settlement.status == "finalized"
            ):
                apply_finalized_settlement(
                    settlement_id=settlement_id,
                    validator_id=validator_id,
                    policy=policy,
                )
            return item
        raise ConflictingAttestationError(item)

    attestation = ValidatorAttestation(
        attestation_id=secrets.token_hex(12),
        created_at=_now_iso(),
        settlement_id=settlement_id,
        validator_id=validator_id,
        accepted=accepted,
        note=note,
    )
    append_attestation(attestation, policy)
    settlement = refresh_settlement_finality(settlement_id=settlement_id, policy=policy)
    if apply_on_finalize and settlement is not None and settlement.status == "finalized":
        apply_finalized_settlement(
            settlement_id=settlement_id,
            validator_id=validator_id,
            policy=policy,
        )
    return attestation


def resolve_settlement(
    settlement_id: str, policy: WalletPolicy | None = None
) -> SettlementRecord | None:
    for item in list_settlements(policy):
        if item.settlement_id == settlement_id:
            return item
    return None


def refresh_settlement_finality(
    *,
    settlement_id: str,
    policy: WalletPolicy | None = None,
) -> SettlementRecord | None:
    settlements = list_settlements(policy)
    for index, item in enumerate(settlements):
        if item.settlement_id != settlement_id:
            continue

        committee_map = {
            str(validator_id): int(bond_atomic)
            for validator_id, bond_atomic in (
                item.committee_bonded_atomic_by_validator_id or {}
            ).items()
        }
        accepted_bond = 0
        rejected_bond = 0
        accepted_count = 0
        rejected_count = 0
        seen_validators: set[str] = set()
        for attestation in list_attestations(settlement_id=settlement_id, policy=policy):
            validator_id = str(attestation.validator_id)
            if validator_id not in committee_map or validator_id in seen_validators:
                continue
            seen_validators.add(validator_id)
            bonded_atomic = committee_map[validator_id]
            if attestation.accepted:
                accepted_bond += bonded_atomic
                accepted_count += 1
            else:
                rejected_bond += bonded_atomic
                rejected_count += 1

        item.accepted_attestations = accepted_count
        item.rejected_attestations = rejected_count
        item.accepted_bond_atomic = accepted_bond
        item.rejected_bond_atomic = rejected_bond
        if item.committee_quorum_bond_atomic <= 0:
            item.status = "pending"
        elif accepted_bond >= item.committee_quorum_bond_atomic:
            item.status = "applied" if item.applied_at else "finalized"
        elif rejected_bond >= item.committee_quorum_bond_atomic:
            item.status = "rejected"
        else:
            item.status = "pending"

        settlements[index] = item
        save_settlements(settlements, policy)
        return item
    return None


def apply_finalized_settlement(
    *,
    settlement_id: str,
    validator_id: str | None = None,
    policy: WalletPolicy | None = None,
    money_policy: MoneyPolicy | None = None,
) -> SettlementRecord | None:
    active_money_policy = money_policy or MoneyPolicy()
    settlements = list_settlements(policy)
    for index, item in enumerate(settlements):
        if item.settlement_id != settlement_id:
            continue
        if item.status not in {"finalized", "applied"}:
            return item

        funding_already_applied = bool(item.applied_at)
        if not funding_already_applied:
            ledger = chain_backed_ledger_snapshot(
                load_or_create_ledger(active_money_policy, policy),
                money_policy=active_money_policy,
                wallet_policy=policy,
            )
            source_wallet_debit = max(0, int(item.source_wallet_debit_atomic or 0))
            reserve_debit = max(0, int(item.reserve_debit_atomic or 0))
            reserve_before_atomic = int(ledger.compute_reserve_balance_atomic)
            source_wallet_before_atomic: int | None = None
            source_wallet_after_atomic: int | None = None

            if reserve_debit > ledger.compute_reserve_balance_atomic:
                raise ValueError("Reserve balance is insufficient to apply settlement.")
            source_wallet = None
            if source_wallet_debit > 0:
                source_wallet = find_wallet_by_id(item.source_wallet_id, policy)
                if source_wallet is None:
                    source_wallet = find_wallet_by_address(
                        item.source_wallet_address, policy
                    )
                ensure_chain_genesis(policy=policy, money_policy=active_money_policy)
                if source_wallet is None:
                    source_wallet_before_atomic = chain_balance_atomic(
                        item.source_wallet_address,
                        policy,
                    )
                    if source_wallet_before_atomic < source_wallet_debit:
                        raise ValueError(
                            "Source wallet chain balance is insufficient to apply settlement."
                        )
                    source_wallet_after_atomic = (
                        source_wallet_before_atomic - source_wallet_debit
                    )
                elif has_chain_activity_for_address(source_wallet.address, policy):
                    source_wallet.spendable_balance_atomic = chain_balance_atomic(
                        source_wallet.address,
                        policy,
                    )
                    if source_wallet.spendable_balance_atomic < source_wallet_debit:
                        raise ValueError(
                            "Source wallet balance is insufficient to apply settlement."
                        )
                    source_wallet_before_atomic = int(source_wallet.spendable_balance_atomic)
                    source_wallet.spendable_balance_atomic -= source_wallet_debit
                    source_wallet_after_atomic = int(source_wallet.spendable_balance_atomic)
                    update_wallet(source_wallet, policy)
                    append_journal_entry(
                        JournalEntry(
                            entry_id=secrets.token_hex(12),
                            event_type="validator_settlement_debit",
                            created_at=_now_iso(),
                            wallet_id=source_wallet.wallet_id,
                            amount_atomic=source_wallet_debit,
                            note=(
                                f"Validator-applied debit for settlement {item.settlement_id}."
                            ),
                        ),
                        policy,
                    )

            ledger.compute_reserve_balance_atomic -= reserve_debit
            reserve_after_atomic = int(ledger.compute_reserve_balance_atomic)
            if not _validator_fee_payout_split(item):
                ledger.validator_fee_pool_atomic += item.settlement_fee_atomic
            ledger.ai_development_fee_pool_atomic += int(
                item.ai_development_fee_atomic or 0
            )
            if not _validator_fee_payout_split(
                item,
                amount_atomic=item.tx_fee_atomic,
            ):
                ledger.tx_fee_pool_atomic += item.tx_fee_atomic
            ledger.worker_distributed_atomic += item.worker_reward_atomic
            ledger.settlements_applied += 1
            save_ledger(ledger, policy)

            item.applied_at = _now_iso()
            item.applied_by_validator_id = validator_id
            item.status = "applied"
            item.balance_audit = _with_applied_balance_audit(
                item,
                reserve_before_atomic=reserve_before_atomic,
                reserve_after_atomic=reserve_after_atomic,
                source_wallet_before_atomic=source_wallet_before_atomic,
                source_wallet_after_atomic=source_wallet_after_atomic,
            )
            settlements[index] = item
            save_settlements(settlements, policy)
        elif item.status != "applied":
            item.status = "applied"
            settlements[index] = item
            save_settlements(settlements, policy)

        if _apply_finalized_ai_development_fee(item, policy=policy):
            settlements[index] = item
            save_settlements(settlements, policy)
        _apply_finalized_worker_payouts(item, policy=policy)
        record_chain_entries_for_settlement(
            item,
            policy=policy,
            money_policy=active_money_policy,
        )
        return item
    return None


def record_validator_evidence(
    *,
    validator_id: str,
    reporter_validator_id: str | None = None,
    evidence_type: str,
    settlement_id: str | None = None,
    attestation_id: str | None = None,
    conflicting_attestation_id: str | None = None,
    slash_atomic: int = 0,
    jailed: bool = False,
    note: str | None = None,
    policy: WalletPolicy | None = None,
) -> ValidatorEvidence:
    normalized_reporter_validator_id = _normalize_validator_reporter_id(
        reporter_validator_id
    )
    if normalized_reporter_validator_id is None:
        from .node_config import get_validator_identity

        normalized_reporter_validator_id = get_validator_identity(policy)
    evidence = ValidatorEvidence(
        evidence_id=secrets.token_hex(12),
        created_at=_now_iso(),
        validator_id=validator_id,
        reporter_validator_id=normalized_reporter_validator_id,
        evidence_type=evidence_type,
        settlement_id=settlement_id,
        attestation_id=attestation_id,
        conflicting_attestation_id=conflicting_attestation_id,
        slash_atomic=slash_atomic,
        jailed=jailed,
        note=note,
        source="local",
        source_url=None,
        last_seen_at=_now_iso(),
        updated_at=_now_iso(),
        applied_to_registry=False,
    )
    append_validator_evidence(evidence, policy)
    refresh_validator_penalty_cases(policy=policy)
    return evidence


def record_validator_penalty_attestation(
    *,
    case_id: str,
    validator_id: str,
    accepted: bool = True,
    note: str | None = None,
    policy: WalletPolicy | None = None,
) -> ValidatorPenaltyAttestation:
    case = next(
        (item for item in list_validator_penalty_cases(policy=policy) if item.case_id == case_id),
        None,
    )
    if case is None:
        refresh_validator_penalty_cases(policy=policy)
        case = next(
            (item for item in list_validator_penalty_cases(policy=policy) if item.case_id == case_id),
            None,
        )
    if case is None:
        raise ValueError(f"Penalty case '{case_id}' not found.")

    normalized_validator_id = str(validator_id).strip().lower()
    if not normalized_validator_id:
        raise ValueError("Validator id is required for penalty attestation.")
    if case.eligible_validator_ids and normalized_validator_id not in set(case.eligible_validator_ids):
        raise ValueError("Validator is not eligible to attest this penalty case.")

    existing = [
        item
        for item in list_validator_penalty_attestations(case_id=case_id, policy=policy)
        if item.validator_id == normalized_validator_id
    ]
    for item in existing:
        if item.accepted == accepted and (item.note or None) == (note or None):
            return item
        raise ValueError("Conflicting penalty attestation already exists for this case and validator.")

    attestation = ValidatorPenaltyAttestation(
        penalty_attestation_id=secrets.token_hex(12),
        created_at=_now_iso(),
        case_id=case_id,
        validator_id=normalized_validator_id,
        accepted=accepted,
        note=note,
    )
    append_validator_penalty_attestation(attestation, policy)
    refresh_validator_penalty_cases(policy=policy)
    return attestation


def _iter_remote_validator_evidence(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("evidence"):
        normalized: list[dict[str, Any]] = []
        for raw in payload.get("evidence") or []:
            evidence_id = str(raw.get("evidence_id") or raw.get("evidenceId") or "").strip().lower()
            if not evidence_id:
                continue
            normalized.append(
                {
                    "evidence_id": evidence_id,
                    "created_at": raw.get("created_at") or raw.get("createdAt"),
                    "validator_id": str(
                        raw.get("validator_id") or raw.get("validatorId") or ""
                    ).strip().lower(),
                    "reporter_validator_id": _normalize_validator_reporter_id(
                        raw.get("reporter_validator_id")
                        or raw.get("reporterValidatorId")
                    ),
                    "evidence_type": raw.get("evidence_type") or raw.get("evidenceType"),
                    "settlement_id": raw.get("settlement_id") or raw.get("settlementId"),
                    "attestation_id": raw.get("attestation_id") or raw.get("attestationId"),
                    "conflicting_attestation_id": raw.get("conflicting_attestation_id")
                    or raw.get("conflictingAttestationId"),
                    "slash_atomic": raw.get("slash_atomic") or raw.get("slashAtomic"),
                    "jailed": raw.get("jailed"),
                    "note": raw.get("note"),
                    "updated_at": raw.get("updated_at") or raw.get("updatedAt"),
                    "applied_to_registry": raw.get("applied_to_registry")
                    or raw.get("appliedToRegistry")
                    or False,
                }
            )
        return normalized
    if payload.get("records"):
        return list(payload.get("records") or [])
    return []


def _validator_evidence_case_key(item: ValidatorEvidence) -> tuple[str, str, str | None, str | None, str | None, int, bool]:
    return (
        item.validator_id,
        item.evidence_type,
        item.settlement_id,
        item.attestation_id,
        item.conflicting_attestation_id,
        item.slash_atomic,
        item.jailed,
    )


def _validator_evidence_case_id(
    case_key: tuple[str, str, str | None, str | None, str | None, int, bool],
) -> str:
    case_hash_input = "|".join(
        [
            case_key[0],
            case_key[1],
            case_key[2] or "",
            case_key[3] or "",
            case_key[4] or "",
            str(case_key[5]),
            str(case_key[6]),
        ]
    )
    return hashlib.sha256(case_hash_input.encode("utf-8")).hexdigest()[:16]


def _iter_pending_evidence_case_keys(
    evidence_items: list[ValidatorEvidence],
) -> list[tuple[str, str, str | None, str | None, str | None, int, bool]]:
    seen: set[tuple[str, str, str | None, str | None, str | None, int, bool]] = set()
    ordered: list[tuple[str, str, str | None, str | None, str | None, int, bool]] = []
    for item in evidence_items:
        case_key = _validator_evidence_case_key(item)
        if case_key in seen:
            continue
        seen.add(case_key)
        if any(
            candidate.applied_to_registry
            for candidate in evidence_items
            if _validator_evidence_case_key(candidate) == case_key
        ):
            continue
        ordered.append(case_key)
    return ordered


def _latest_evidence_for_case(
    case_key: tuple[str, str, str | None, str | None, str | None, int, bool],
    evidence_items: list[ValidatorEvidence],
) -> ValidatorEvidence | None:
    matching = [
        item for item in evidence_items if _validator_evidence_case_key(item) == case_key
    ]
    if not matching:
        return None
    matching.sort(key=lambda item: item.updated_at or item.created_at, reverse=True)
    return matching[0]


def _validator_evidence_case_has_quorum(
    case_key: tuple[str, str, str | None, str | None, str | None, int, bool],
    evidence_items: list[ValidatorEvidence],
    policy: WalletPolicy | None = None,
) -> bool:
    supporting_count = _validator_evidence_case_supporting_count(
        case_key, evidence_items, policy
    )
    if supporting_count <= 0:
        return False
    required = _validator_evidence_source_quorum(policy)
    support_scope, _, scoped_required = _validator_evidence_case_support_context(
        case_key, policy
    )
    if support_scope != "source_url":
        required = scoped_required
    return supporting_count >= required


def _validator_evidence_source_quorum(policy: WalletPolicy | None = None) -> int:
    bonded = list_bonded_validators(policy)
    if not bonded:
        return 1
    total = len(bonded)
    return max(1, ((2 * total) + 2) // 3)


def _validator_evidence_case_supporting_validator_ids(
    case_key: tuple[str, str, str | None, str | None, str | None, int, bool],
    evidence_items: list[ValidatorEvidence],
    policy: WalletPolicy | None = None,
) -> list[str]:
    _, eligible_validator_ids, _ = _validator_evidence_case_support_context(
        case_key, policy
    )
    if not eligible_validator_ids:
        eligible_validator_ids = {
            item.validator_id for item in list_bonded_validators(policy)
        }
    return sorted(
        {
            item.reporter_validator_id
            for item in evidence_items
            if _validator_evidence_case_key(item) == case_key
            and item.reporter_validator_id
            and item.reporter_validator_id in eligible_validator_ids
        }
    )


def _validator_evidence_case_supporting_count(
    case_key: tuple[str, str, str | None, str | None, str | None, int, bool],
    evidence_items: list[ValidatorEvidence],
    policy: WalletPolicy | None = None,
) -> int:
    supporting_validator_ids = _validator_evidence_case_supporting_validator_ids(
        case_key, evidence_items, policy
    )
    if supporting_validator_ids:
        return len(supporting_validator_ids)
    supporting_sources = {
        item.source_url
        for item in evidence_items
        if _validator_evidence_case_key(item) == case_key and item.source_url
    }
    return len(supporting_sources)


def _normalize_validator_reporter_id(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def _validator_evidence_case_support_context(
    case_key: tuple[str, str, str | None, str | None, str | None, int, bool],
    policy: WalletPolicy | None = None,
) -> tuple[str, set[str], int]:
    settlement_id = case_key[2]
    if settlement_id:
        settlement = resolve_settlement(settlement_id, policy)
        if settlement is not None and settlement.committee_validator_ids:
            committee_validator_ids = {
                str(item).strip().lower()
                for item in settlement.committee_validator_ids
                if str(item).strip()
            }
            required = max(1, ((2 * len(committee_validator_ids)) + 2) // 3)
            return ("settlement_committee", committee_validator_ids, required)

    bonded_validator_ids = {
        item.validator_id for item in list_bonded_validators(policy)
    }
    if bonded_validator_ids:
        required = max(1, ((2 * len(bonded_validator_ids)) + 2) // 3)
        return ("bonded_validator_set", bonded_validator_ids, required)

    return ("source_url", set(), 1)


def _validator_penalty_attestation_validator_ids(
    case_id: str,
    eligible_validator_ids: set[str],
    policy: WalletPolicy | None = None,
) -> list[str]:
    if not eligible_validator_ids:
        return []
    return sorted(
        {
            item.validator_id
            for item in list_validator_penalty_attestations(case_id=case_id, policy=policy)
            if item.accepted and item.validator_id in eligible_validator_ids
        }
    )


def _validator_penalty_attestation_required(
    *, support_scope: str, required_sources: int
) -> int:
    if support_scope == "settlement_committee":
        return max(1, int(required_sources))
    return 0


def _validator_penalty_case_has_quorum(
    *,
    support_scope: str,
    penalty_attestation_count: int,
    penalty_attestation_required: int,
    evidence_quorum_reached: bool,
) -> bool:
    if not evidence_quorum_reached:
        return False
    if support_scope != "settlement_committee":
        return True
    return penalty_attestation_count >= max(1, penalty_attestation_required)


def _validator_penalty_attestation_endpoint(source_url: str) -> str:
    normalized = source_url.rstrip("/")
    if normalized.endswith("/v1/cai/validators"):
        return normalized[: -len("/v1/cai/validators")] + "/v1/cai/validator-penalty/attest"
    return normalized.rstrip("/") + "/v1/cai/validator-penalty/attest"


def _post_json(url: str, payload: dict[str, Any], *, timeout: int) -> dict[str, Any]:
    from urllib.request import Request

    request = Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def refresh_validator_penalty_cases(
    *,
    evidence_items: list[ValidatorEvidence] | None = None,
    policy: WalletPolicy | None = None,
) -> tuple[list[ValidatorPenaltyCase], int]:
    active_items = evidence_items if evidence_items is not None else list_validator_evidence(policy=policy)
    existing_case_map = {
        item.case_id: item for item in list_validator_penalty_cases(policy=policy)
    }
    seen_case_ids: set[str] = set()
    cases: list[ValidatorPenaltyCase] = []
    applied_records = 0
    evidence_changed = False

    grouped: dict[
        tuple[str, str, str | None, str | None, str | None, int, bool],
        list[ValidatorEvidence],
    ] = {}
    for item in active_items:
        grouped.setdefault(_validator_evidence_case_key(item), []).append(item)

    for case_key, items in grouped.items():
        items.sort(key=lambda item: item.updated_at or item.created_at, reverse=True)
        latest = items[0]
        case_id = _validator_evidence_case_id(case_key)
        seen_case_ids.add(case_id)
        support_scope, eligible_validator_ids, required_sources = (
            _validator_evidence_case_support_context(case_key, policy)
        )
        evidence_supporting_validator_ids = _validator_evidence_case_supporting_validator_ids(
            case_key, active_items, policy
        )
        supporting_sources = sorted(
            {item.source_url for item in items if item.source_url}
        )
        support_mode = "validator" if evidence_supporting_validator_ids else "source_url"
        evidence_quorum_reached = _validator_evidence_case_has_quorum(
            case_key, active_items, policy
        )
        penalty_attester_validator_ids = _validator_penalty_attestation_validator_ids(
            case_id, eligible_validator_ids, policy
        )
        combined_supporting_validator_ids = sorted(
            set(evidence_supporting_validator_ids).union(penalty_attester_validator_ids)
        )
        penalty_attestation_required = _validator_penalty_attestation_required(
            support_scope=support_scope,
            required_sources=required_sources,
        )
        penalty_attestation_count = len(penalty_attester_validator_ids)
        quorum_reached = _validator_penalty_case_has_quorum(
            support_scope=support_scope,
            penalty_attestation_count=penalty_attestation_count,
            penalty_attestation_required=penalty_attestation_required,
            evidence_quorum_reached=evidence_quorum_reached,
        )
        existing_case = existing_case_map.get(case_id)
        finalized_at = existing_case.finalized_at if existing_case else None
        applied_at = existing_case.applied_at if existing_case else None
        status = "pending"
        if quorum_reached:
            finalized_at = finalized_at or _now_iso()
            status = "finalized"
            if applied_at is None:
                if apply_remote_validator_penalty_to_registry(
                    validator_id=latest.validator_id,
                    slash_atomic=latest.slash_atomic,
                    jailed=latest.jailed,
                    observed_at=latest.created_at,
                    source_url=latest.source_url,
                    policy=policy,
                ):
                    applied_at = _now_iso()
                    applied_records += 1
                    for index, item in enumerate(active_items):
                        if (
                            _validator_evidence_case_key(item) != case_key
                            or item.applied_to_registry
                        ):
                            continue
                        active_items[index] = ValidatorEvidence(
                            **{**asdict(item), "applied_to_registry": True}
                        )
                        evidence_changed = True
            if applied_at is not None:
                status = "applied"

        cases.append(
            ValidatorPenaltyCase(
                case_id=case_id,
                created_at=min(item.created_at for item in items),
                updated_at=latest.updated_at or latest.created_at,
                validator_id=latest.validator_id,
                evidence_type=latest.evidence_type,
                settlement_id=latest.settlement_id,
                attestation_id=latest.attestation_id,
                conflicting_attestation_id=latest.conflicting_attestation_id,
                slash_atomic=latest.slash_atomic,
                jailed=latest.jailed,
                support_mode=support_mode,
                support_scope=support_scope,
                required_sources=required_sources,
                supporting_sources=supporting_sources,
                supporting_validator_ids=combined_supporting_validator_ids,
                eligible_validator_ids=sorted(eligible_validator_ids),
                evidence_count=len(items),
                evidence_quorum_reached=evidence_quorum_reached,
                penalty_attestation_count=penalty_attestation_count,
                penalty_attestation_required=penalty_attestation_required,
                quorum_reached=quorum_reached,
                status=status,
                finalized_at=finalized_at,
                applied_at=applied_at,
            )
        )

    for case_id, existing_case in existing_case_map.items():
        if case_id in seen_case_ids:
            continue
        cases.append(existing_case)

    cases.sort(key=lambda item: item.updated_at, reverse=True)
    if evidence_changed:
        save_validator_evidence(active_items, policy)
    save_validator_penalty_cases(cases, policy)
    return cases, applied_records


def request_remote_penalty_case_attestations(
    *,
    cai_url: str,
    state_payload: dict[str, Any] | None = None,
    policy: WalletPolicy | None = None,
    timeout_sec: int = 10,
) -> list[ValidatorPenaltyAttestation]:
    active_state_payload = state_payload
    if active_state_payload is None:
        state_url = cai_url.rstrip("/") + "/state"
        with urlopen(state_url, timeout=timeout_sec) as response:
            active_state_payload = json.loads(response.read().decode("utf-8"))

    try:
        sync_validator_set_from_cai_peers(
            state_payload=active_state_payload,
            cai_url=cai_url,
            policy=policy,
            timeout_sec=timeout_sec,
        )
    except Exception as exc:
        _log_best_effort_failure(
            "validator set sync before penalty attestation requests",
            exc,
        )

    from .node_config import load_or_create_node_config

    node_config = load_or_create_node_config(policy)
    local_validator_id = (
        str(node_config.validator_address).strip().lower()
        if getattr(node_config, "validator_address", None)
        else None
    )

    recorded: list[ValidatorPenaltyAttestation] = []
    for case in list_validator_penalty_cases(policy=policy):
        if case.status != "pending" or not case.evidence_quorum_reached:
            continue
        eligible_ids = [str(item).strip().lower() for item in case.eligible_validator_ids if str(item).strip()]
        if not eligible_ids:
            continue
        existing_attesters = {
            item.validator_id
            for item in list_validator_penalty_attestations(case_id=case.case_id, policy=policy)
            if item.accepted
        }
        for validator_id in eligible_ids:
            if validator_id in existing_attesters:
                continue
            if local_validator_id is not None and validator_id == local_validator_id:
                continue
            record = get_validator_record(validator_id, policy)
            if record is None:
                continue
            record_source_url = resolve_validator_peer_url(
                source_url=getattr(record, "source_url", None),
                advertised_api_host=getattr(record, "advertised_api_host", None),
            )
            if not record_source_url:
                continue
            endpoint = _validator_penalty_attestation_endpoint(record_source_url)
            try:
                payload = _post_json(
                    endpoint,
                    {
                        "case_id": case.case_id,
                        "validator_id": case.validator_id,
                        "evidence_type": case.evidence_type,
                        "settlement_id": case.settlement_id,
                        "slash_atomic": case.slash_atomic,
                        "jailed": case.jailed,
                        "eligible_validator_ids": eligible_ids,
                    },
                    timeout=timeout_sec,
                )
            except Exception as exc:
                _log_best_effort_failure(
                    f"penalty attestation request to validator {validator_id}",
                    exc,
                )
                continue
            if not payload.get("attested"):
                continue
            try:
                attestation = record_validator_penalty_attestation(
                    case_id=case.case_id,
                    validator_id=str(
                        payload.get("validatorId") or validator_id
                    ).strip().lower(),
                    accepted=bool(payload.get("accepted")),
                    note=str(
                        payload.get("note")
                        or (
                            "Remote validator accepted penalty case."
                            if payload.get("accepted")
                            else "Remote validator rejected penalty case."
                        )
                    ),
                    policy=policy,
                )
            except ValueError:
                continue
            recorded.append(attestation)
    return recorded


def record_worker_payouts(
    *,
    settlement_id: str,
    receipt_id: str,
    model_id: str,
    participants: list[dict],
    money_policy: MoneyPolicy | None = None,
    policy: WalletPolicy | None = None,
) -> list[WorkerPayoutRecord]:
    active_money_policy = money_policy or MoneyPolicy()
    settlement = resolve_settlement(settlement_id, policy)
    existing_for_receipt = list_worker_payouts(
        settlement_id=settlement_id, receipt_id=receipt_id, policy=policy
    )
    if existing_for_receipt:
        if settlement is not None:
            settlement = _save_settlement_balance_audit(
                settlement,
                _with_worker_payout_accounting_audit(
                    settlement,
                    existing_for_receipt,
                    money_policy=active_money_policy,
                ),
                policy=policy,
            )
        if settlement is not None and (
            settlement.status == "applied" or settlement.applied_at
        ):
            _apply_finalized_worker_payouts(settlement, policy=policy)
            return list_worker_payouts(
                settlement_id=settlement_id, receipt_id=receipt_id, policy=policy
            )
        return existing_for_receipt

    existing = list_worker_payouts(policy=policy)
    created_at = _now_iso()
    records: list[WorkerPayoutRecord] = []
    for item in participants:
        node_id = str(item["node_id"])
        reward_atomic = int(item["reward_atomic"])
        recipient_address = resolve_worker_reward_address(node_id, policy)
        credited_wallet_id: str | None = None
        status = "pending_settlement" if recipient_address is not None else "unbound"

        records.append(
            WorkerPayoutRecord(
                payout_id=secrets.token_hex(12),
                created_at=created_at,
                settlement_id=settlement_id,
                receipt_id=receipt_id,
                model_id=model_id,
                node_id=node_id,
                runner_id=(
                    str(item["runner_id"]) if item.get("runner_id") is not None else None
                ),
                layer_start=item.get("layer_start"),
                layer_end=item.get("layer_end"),
                layer_count=int(item["layer_count"]),
                share_bps=int(item["share_bps"]),
                reward_atomic=reward_atomic,
                reward_token_code=active_money_policy.reward_token_code,
                recipient_address=recipient_address,
                credited_wallet_id=credited_wallet_id,
                status=status,
                note=item.get("note"),
            )
        )
    if settlement is not None:
        errors = validate_settlement_accounting(
            settlement,
            records,
            money_policy=active_money_policy,
        )
        if errors:
            raise ValueError(
                "Settlement payout accounting mismatch: " + " ".join(errors)
            )
    existing.extend(records)
    save_worker_payouts(existing, policy)
    if settlement is not None:
        settlement = _save_settlement_balance_audit(
            settlement,
            _with_worker_payout_accounting_audit(
                settlement,
                records,
                money_policy=active_money_policy,
            ),
            policy=policy,
        )
    if settlement is not None and (
        settlement.status == "applied" or settlement.applied_at
    ):
        _apply_finalized_worker_payouts(settlement, policy=policy)
        return list_worker_payouts(
            settlement_id=settlement_id, receipt_id=receipt_id, policy=policy
        )
    return records


def reconcile_worker_payouts(
    policy: WalletPolicy | None = None,
) -> list[WorkerPayoutRecord]:
    payouts = list_worker_payouts(policy=policy)
    if not payouts:
        return []

    updated = False
    reconciled: list[WorkerPayoutRecord] = []
    settlement_by_id = {item.settlement_id: item for item in list_settlements(policy)}
    for item in payouts:
        if item.status == "credited_local_wallet":
            continue

        recipient_address = resolve_worker_reward_address(item.node_id, policy)
        if recipient_address is None:
            continue

        item.recipient_address = recipient_address
        settlement = settlement_by_id.get(item.settlement_id)
        if settlement is None or (
            settlement.status != "applied" and not settlement.applied_at
        ):
            item.status = "pending_settlement"
            item.credited_wallet_id = None
        else:
            _apply_finalized_worker_payout_record(item, policy=policy, reconciled=True)
        updated = True
        reconciled.append(item)

    if updated:
        save_worker_payouts(payouts, policy)
    return reconciled


def _apply_finalized_ai_development_fee(
    settlement: SettlementRecord,
    *,
    policy: WalletPolicy | None = None,
) -> bool:
    amount_atomic = max(0, int(settlement.ai_development_fee_atomic or 0))
    if amount_atomic <= 0 or settlement.ai_development_credited_wallet_id:
        return False

    recipient_wallet = None
    if settlement.ai_development_wallet_id:
        recipient_wallet = find_wallet_by_id(settlement.ai_development_wallet_id, policy)
    if recipient_wallet is None and settlement.ai_development_address:
        recipient_wallet = find_wallet_by_address(
            settlement.ai_development_address,
            policy,
        )
    if recipient_wallet is None:
        return False

    recipient_wallet.spendable_balance_atomic += amount_atomic
    update_wallet(recipient_wallet, policy)
    append_journal_entry(
        JournalEntry(
            entry_id=secrets.token_hex(12),
            event_type="ai_development_fee_credit",
            created_at=_now_iso(),
            wallet_id=recipient_wallet.wallet_id,
            amount_atomic=amount_atomic,
            note=(
                f"AI development fee credited from finalized settlement "
                f"{settlement.settlement_id}."
            ),
        ),
        policy,
    )
    settlement.ai_development_credited_wallet_id = recipient_wallet.wallet_id
    return True


def _apply_finalized_worker_payouts(
    settlement: SettlementRecord,
    *,
    policy: WalletPolicy | None = None,
) -> list[WorkerPayoutRecord]:
    payouts = list_worker_payouts(policy=policy)
    changed = False
    applied: list[WorkerPayoutRecord] = []
    for item in payouts:
        if item.settlement_id != settlement.settlement_id:
            continue
        if item.status == "credited_local_wallet":
            continue
        if item.recipient_address is None:
            item.recipient_address = resolve_worker_reward_address(item.node_id, policy)
        if item.recipient_address is None:
            item.status = "unbound"
            item.credited_wallet_id = None
        else:
            _apply_finalized_worker_payout_record(item, policy=policy)
        changed = True
        applied.append(item)
    if changed:
        save_worker_payouts(payouts, policy)
    return applied


def _apply_finalized_worker_payout_record(
    item: WorkerPayoutRecord,
    *,
    policy: WalletPolicy | None = None,
    reconciled: bool = False,
) -> WorkerPayoutRecord:
    if item.recipient_address is None:
        item.status = "unbound"
        item.credited_wallet_id = None
        return item

    recipient_wallet = find_wallet_by_address(item.recipient_address, policy)
    if recipient_wallet is None:
        item.credited_wallet_id = None
        item.status = "recorded_external_address"
        return item

    if item.credited_wallet_id != recipient_wallet.wallet_id:
        recipient_wallet.spendable_balance_atomic += item.reward_atomic
        update_wallet(recipient_wallet, policy)
        action = "reconciled" if reconciled else "credited"
        append_journal_entry(
            JournalEntry(
                entry_id=secrets.token_hex(12),
                event_type="worker_reward_credit",
                created_at=_now_iso(),
                wallet_id=recipient_wallet.wallet_id,
                amount_atomic=item.reward_atomic,
                note=(
                    f"Worker reward {action} for node {item.node_id} "
                    f"from finalized settlement {item.settlement_id}."
                ),
            ),
            policy,
        )
        item.credited_wallet_id = recipient_wallet.wallet_id
    item.status = "credited_local_wallet"
    return item


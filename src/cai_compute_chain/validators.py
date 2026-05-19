# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

from .local_json_store import atomic_write_json_array_file, read_json_array_file
from .model import (
    MoneyPolicy,
    ValidatorLifecycleState,
    WalletPolicy,
    default_api_port,
    default_bootstrap_peers,
)
from .peer_payload import (
    add_peer_payload_metadata,
    peer_payload_hybrid_signatures_required,
    peer_payload_signatures_required,
    validate_peer_payload_network,
    verify_peer_payload_signature,
)
from .transport_endpoints import build_http_url, candidate_identity_http_urls
from .wallet import data_root


@dataclass
class ValidatorRecord:
    validator_id: str
    wallet_id: str | None
    address: str
    state: str
    bonded_atomic: int
    static_ip_confirmed: bool
    current_node_id: str | None
    advertised_api_host: str | None
    advertised_data_host: str | None
    activated_at: str | None
    ha_enabled: bool = False
    active_replica_node_id: str | None = None
    active_replica_lease_until: str | None = None
    replica_node_ids: list[str] = field(default_factory=list)
    unbonding_started_at: str | None = None
    unbonding_available_at: str | None = None
    jailed_at: str | None = None
    unjail_available_at: str | None = None
    last_slash_atomic: int = 0
    total_slashed_atomic: int = 0
    source: str = "local"
    source_url: str | None = None
    last_seen_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class ValidatorCommitteeSnapshot:
    validator_ids: list[str]
    bonded_atomic_by_validator_id: dict[str, int]
    total_bonded_atomic: int
    quorum_bond_atomic: int


@dataclass(frozen=True)
class ValidatorFeeShare:
    validator_id: str
    bonded_atomic: int
    amount_atomic: int


@dataclass(frozen=True)
class ValidatorSetSyncResult:
    attempted_peers: int
    successful_peers: int
    imported_records: int
    peer_urls: list[str]
    failed_peers: int = 0
    failed_peer_urls: list[str] = field(default_factory=list)
    peer_errors: list[dict[str, str]] = field(default_factory=list)


def validator_set_file_path(policy: WalletPolicy | None = None) -> Path:
    active_policy = policy or WalletPolicy()
    return data_root(active_policy) / active_policy.validator_set_file_name


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _coalesce_cai_url(cai_url: str | None = None, CAI_url: str | None = None) -> str:
    resolved = str(cai_url or CAI_url or "").strip()
    if not resolved:
        raise ValueError("CAI URL is required.")
    return resolved


def list_validator_records(policy: WalletPolicy | None = None) -> list[ValidatorRecord]:
    path = validator_set_file_path(policy)
    if not path.exists():
        return []
    raw = read_json_array_file(path, heal_corrupt=True)
    items: list[ValidatorRecord] = []
    for item in raw:
        item.setdefault("source", "local")
        item.setdefault("source_url", None)
        item.setdefault("last_seen_at", None)
        item.setdefault("unbonding_started_at", None)
        item.setdefault("unbonding_available_at", None)
        item.setdefault("ha_enabled", False)
        item.setdefault("active_replica_node_id", None)
        item.setdefault("active_replica_lease_until", None)
        item.setdefault("replica_node_ids", [])
        items.append(ValidatorRecord(**item))
    state_rank = {
        ValidatorLifecycleState.BONDED: 0,
        ValidatorLifecycleState.UNBONDING: 1,
        ValidatorLifecycleState.JAILED: 2,
        ValidatorLifecycleState.UNBONDED: 3,
    }
    items.sort(key=lambda item: (state_rank.get(item.state, 99), item.validator_id))
    return items


def save_validator_records(
    records: list[ValidatorRecord], policy: WalletPolicy | None = None
) -> None:
    path = validator_set_file_path(policy)
    atomic_write_json_array_file(path, [asdict(item) for item in records])


def get_validator_record(
    validator_id: str, policy: WalletPolicy | None = None
) -> ValidatorRecord | None:
    for item in list_validator_records(policy):
        if item.validator_id == validator_id:
            return item
    return None


def sync_validator_record(
    *,
    validator_id: str,
    wallet_id: str | None,
    address: str,
    state: str,
    bonded_atomic: int,
    static_ip_confirmed: bool,
    current_node_id: str | None = None,
    advertised_api_host: str | None = None,
    advertised_data_host: str | None = None,
    ha_enabled: bool | None = None,
    ha_role: str = "standalone",
    active_replica_node_id: str | None = None,
    active_replica_lease_until: str | None = None,
    replica_node_ids: list[str] | None = None,
    unbonding_started_at: str | None = None,
    unbonding_available_at: str | None = None,
    jailed_at: str | None = None,
    unjail_available_at: str | None = None,
    last_slash_atomic: int = 0,
    total_slashed_atomic: int = 0,
    source: str = "local",
    source_url: str | None = None,
    last_seen_at: str | None = None,
    updated_at: str | None = None,
    policy: WalletPolicy | None = None,
) -> ValidatorRecord:
    existing = list_validator_records(policy)
    index = next((i for i, item in enumerate(existing) if item.validator_id == validator_id), None)
    previous = existing[index] if index is not None else None
    activated_at = previous.activated_at if previous is not None else None
    if state == ValidatorLifecycleState.BONDED and activated_at is None:
        activated_at = _now_iso()

    normalized_ha_role = str(ha_role or "").strip().lower()
    record_ha_enabled = bool(
        previous.ha_enabled if ha_enabled is None and previous is not None else ha_enabled
    )
    record_current_node_id = (
        current_node_id if current_node_id is not None else (previous.current_node_id if previous else None)
    )
    record_active_replica_node_id = None
    record_active_replica_lease_until = None
    normalized_replica_node_ids: list[str] = []
    if record_ha_enabled:
        normalized_replica_node_ids = _merged_replica_node_ids(
            previous.replica_node_ids if previous else [],
            replica_node_ids or [],
            current_node_id,
            previous.current_node_id if previous else None,
        )
        record_active_replica_node_id = (
            active_replica_node_id
            if active_replica_node_id is not None
            else (previous.active_replica_node_id if previous else None)
        )
        record_active_replica_lease_until = (
            active_replica_lease_until
            if active_replica_lease_until is not None
            else (previous.active_replica_lease_until if previous else None)
        )
        if normalized_ha_role == "passive" and previous is not None:
            record_current_node_id = previous.current_node_id
            record_active_replica_node_id = previous.active_replica_node_id
            record_active_replica_lease_until = previous.active_replica_lease_until

    record = ValidatorRecord(
        validator_id=validator_id,
        wallet_id=wallet_id,
        address=address,
        state=state,
        bonded_atomic=max(0, int(bonded_atomic)),
        static_ip_confirmed=bool(static_ip_confirmed),
        current_node_id=record_current_node_id,
        advertised_api_host=advertised_api_host if advertised_api_host is not None else (previous.advertised_api_host if previous else None),
        advertised_data_host=advertised_data_host if advertised_data_host is not None else (previous.advertised_data_host if previous else None),
        activated_at=activated_at,
        ha_enabled=record_ha_enabled,
        active_replica_node_id=record_active_replica_node_id,
        active_replica_lease_until=record_active_replica_lease_until,
        replica_node_ids=normalized_replica_node_ids,
        unbonding_started_at=unbonding_started_at,
        unbonding_available_at=unbonding_available_at,
        jailed_at=jailed_at,
        unjail_available_at=unjail_available_at,
        last_slash_atomic=max(0, int(last_slash_atomic)),
        total_slashed_atomic=max(
            total_slashed_atomic,
            previous.total_slashed_atomic if previous else 0,
        ),
        source=source,
        source_url=source_url,
        last_seen_at=last_seen_at or _now_iso(),
        updated_at=updated_at or _now_iso(),
    )
    if index is None:
        existing.append(record)
    else:
        existing[index] = record
    save_validator_records(existing, policy)
    return record


def list_bonded_validators(policy: WalletPolicy | None = None) -> list[ValidatorRecord]:
    return [
        item
        for item in list_effective_validator_records(policy)
        if item.state == ValidatorLifecycleState.BONDED and item.bonded_atomic > 0
    ]


def list_effective_validator_records(
    policy: WalletPolicy | None = None,
) -> list[ValidatorRecord]:
    locked_bonds = _chain_locked_bonds_by_validator(policy)
    return [
        _chain_backed_validator_record(item, locked_bonds)
        for item in list_validator_records(policy)
    ]


def _chain_backed_validator_record(
    record: ValidatorRecord,
    locked_bonds: dict[str, int],
) -> ValidatorRecord:
    if record.state != ValidatorLifecycleState.BONDED:
        return record
    locked_atomic = locked_bonds.get(_normalize_validator_id(record.validator_id), 0)
    if record.bonded_atomic > 0 and locked_atomic >= record.bonded_atomic:
        return record
    payload = asdict(record)
    payload["state"] = ValidatorLifecycleState.UNBONDED
    payload["bonded_atomic"] = 0
    return ValidatorRecord(**payload)


def _chain_locked_bonds_by_validator(
    policy: WalletPolicy | None = None,
) -> dict[str, int]:
    try:
        from .chain import (
            chain_balance_atomic,
            validator_bond_pool_chain_address,
            validator_locked_bond_index,
        )
    except Exception:
        return {}

    locked_bonds = {
        _normalize_validator_id(validator_id): max(0, int(locked_atomic or 0))
        for validator_id, locked_atomic in validator_locked_bond_index(policy).items()
    }
    total_locked_atomic = sum(locked_bonds.values())
    if total_locked_atomic <= 0:
        return {}
    pool_atomic = chain_balance_atomic(
        validator_bond_pool_chain_address(
            MoneyPolicy(chain_network=(policy or WalletPolicy()).chain_network)
        ),
        policy,
    )
    if pool_atomic != total_locked_atomic:
        return {}
    return locked_bonds


def _quorum_bond_atomic(total_bonded_atomic: int) -> int:
    if total_bonded_atomic <= 0:
        return 0
    return ((2 * int(total_bonded_atomic)) + 2) // 3


def _normalize_validator_id(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized.startswith("cai_"):
        return normalized[4:]
    return normalized


def _merged_replica_node_ids(
    *groups: list[str] | tuple[str, ...] | set[str] | str | None,
) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        if group is None:
            continue
        values = [group] if isinstance(group, str) else list(group)
        for value in values:
            normalized = str(value or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
    return merged


def validator_ha_lease_is_active(
    record: ValidatorRecord | None,
    *,
    now: datetime | None = None,
) -> bool:
    if record is None or not record.ha_enabled or not record.active_replica_node_id:
        return False
    lease_until = _parse_iso_datetime(record.active_replica_lease_until)
    if lease_until is None:
        return False
    if lease_until.tzinfo is None:
        lease_until = lease_until.replace(tzinfo=UTC)
    active_now = now or datetime.now(tz=UTC)
    if active_now.tzinfo is None:
        active_now = active_now.replace(tzinfo=UTC)
    return lease_until > active_now


def split_amount_by_validator_bond(
    amount_atomic: int,
    bonded_atomic_by_validator_id: dict[str, int],
    *,
    validator_ids: list[str] | None = None,
) -> list[ValidatorFeeShare]:
    amount_atomic = max(0, int(amount_atomic or 0))
    if amount_atomic <= 0:
        return []

    raw_bonds = dict(bonded_atomic_by_validator_id or {})
    candidate_ids = list(validator_ids or raw_bonds.keys())
    bonded_by_validator: dict[str, int] = {}
    for raw_validator_id in candidate_ids:
        validator_id = _normalize_validator_id(str(raw_validator_id or ""))
        if not validator_id:
            continue
        bonded_atomic = raw_bonds.get(raw_validator_id)
        if bonded_atomic is None:
            bonded_atomic = raw_bonds.get(validator_id, 0)
        bonded_atomic = max(0, int(bonded_atomic or 0))
        if bonded_atomic <= 0:
            continue
        bonded_by_validator[validator_id] = (
            bonded_by_validator.get(validator_id, 0) + bonded_atomic
        )

    total_bonded_atomic = sum(bonded_by_validator.values())
    if total_bonded_atomic <= 0:
        return []

    shares: list[dict[str, int | str]] = []
    allocated_atomic = 0
    for validator_id in sorted(bonded_by_validator):
        bonded_atomic = bonded_by_validator[validator_id]
        numerator = amount_atomic * bonded_atomic
        share_atomic, remainder = divmod(numerator, total_bonded_atomic)
        allocated_atomic += share_atomic
        shares.append(
            {
                "validator_id": validator_id,
                "bonded_atomic": bonded_atomic,
                "amount_atomic": share_atomic,
                "remainder": remainder,
            }
        )

    remainder_atomic = amount_atomic - allocated_atomic
    for index in sorted(
        range(len(shares)),
        key=lambda item: (
            -int(shares[item]["remainder"]),
            str(shares[item]["validator_id"]),
        ),
    )[:remainder_atomic]:
        shares[index]["amount_atomic"] = int(shares[index]["amount_atomic"]) + 1

    return [
        ValidatorFeeShare(
            validator_id=str(item["validator_id"]),
            bonded_atomic=int(item["bonded_atomic"]),
            amount_atomic=int(item["amount_atomic"]),
        )
        for item in shares
        if int(item["amount_atomic"]) > 0
    ]


def build_validator_committee_snapshot(
    policy: WalletPolicy | None = None,
) -> ValidatorCommitteeSnapshot:
    bonded = list_bonded_validators(policy)
    bonded_map = {item.validator_id: item.bonded_atomic for item in bonded}
    total = sum(bonded_map.values())
    return ValidatorCommitteeSnapshot(
        validator_ids=sorted(bonded_map),
        bonded_atomic_by_validator_id=bonded_map,
        total_bonded_atomic=total,
        quorum_bond_atomic=_quorum_bond_atomic(total),
    )


def select_validator_committee_snapshot(
    *,
    selection_seed: str,
    policy: WalletPolicy | None = None,
    money_policy: MoneyPolicy | None = None,
    target_size: int | None = None,
) -> ValidatorCommitteeSnapshot:
    bonded = list_bonded_validators(policy)
    if not bonded:
        return ValidatorCommitteeSnapshot(
            validator_ids=[],
            bonded_atomic_by_validator_id={},
            total_bonded_atomic=0,
            quorum_bond_atomic=0,
        )

    active_money_policy = money_policy or MoneyPolicy()
    desired_size = int(
        target_size
        if target_size is not None
        else active_money_policy.validator_committee_target_size
    )
    if desired_size <= 0:
        desired_size = len(bonded)
    desired_size = min(desired_size, len(bonded))

    if desired_size >= len(bonded):
        selected = bonded
    else:
        selected = _select_weighted_committee(
            bonded,
            desired_size=desired_size,
            selection_seed=selection_seed,
        )

    bonded_map = {item.validator_id: item.bonded_atomic for item in selected}
    total = sum(bonded_map.values())
    return ValidatorCommitteeSnapshot(
        validator_ids=sorted(bonded_map),
        bonded_atomic_by_validator_id=bonded_map,
        total_bonded_atomic=total,
        quorum_bond_atomic=_quorum_bond_atomic(total),
    )


def export_validator_set_payload(policy: WalletPolicy | None = None) -> dict[str, Any]:
    committee = build_validator_committee_snapshot(policy)
    return add_peer_payload_metadata(
        {
            "exported_at": _now_iso(),
            "records": [
                asdict(item) for item in list_effective_validator_records(policy)
            ],
            "committee": {
                "validator_ids": list(committee.validator_ids),
                "bonded_atomic_by_validator_id": dict(
                    committee.bonded_atomic_by_validator_id
                ),
                "total_bonded_atomic": committee.total_bonded_atomic,
                "quorum_bond_atomic": committee.quorum_bond_atomic,
            },
        },
        policy=policy,
    )


def _format_host_for_url(host: str) -> str:
    normalized = str(host or "").strip()
    if ":" in normalized and not normalized.startswith("["):
        return f"[{normalized}]"
    return normalized


def resolve_validator_peer_url(
    *,
    source_url: str | None,
    advertised_api_host: str | None,
    endpoint_path: str = "/v1/cai/validators",
) -> str | None:
    normalized_source_url = str(source_url or "").strip()
    normalized_endpoint_path = "/" + str(endpoint_path or "").lstrip("/")
    normalized_advertised_host = _normalize_host(advertised_api_host)
    parsed = urlparse(normalized_source_url) if normalized_source_url else None
    if (
        normalized_advertised_host
        and parsed is not None
        and parsed.scheme
        and parsed.hostname
    ):
        formatted_host = _format_host_for_url(normalized_advertised_host)
        if parsed.port is not None:
            return f"{parsed.scheme}://{formatted_host}:{int(parsed.port)}{normalized_endpoint_path}"
        return f"{parsed.scheme}://{formatted_host}{normalized_endpoint_path}"
    if parsed is not None and parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}{normalized_endpoint_path}"
    return normalized_source_url or None


def merge_remote_validator_set_payload(
    payload: dict[str, Any],
    *,
    source_url: str,
    policy: WalletPolicy | None = None,
) -> int:
    validate_peer_payload_network(
        payload,
        policy=policy,
        payload_name="validator set",
    )
    signature_ok, signature_error = verify_peer_payload_signature(
        payload,
        payload_name="validator set",
        require_signature=peer_payload_signatures_required(policy=policy),
        require_hybrid_signature=peer_payload_hybrid_signatures_required(
            policy=policy
        ),
    )
    if not signature_ok:
        raise ValueError(signature_error or "Invalid validator set payload signature.")
    raw_records = _iter_remote_validator_records(payload)
    authoritative_node_ids = _authoritative_payload_validator_node_ids(
        payload,
        raw_records,
    )
    authoritative_sources = _authoritative_payload_validator_sources(
        payload,
        raw_records,
        source_url=source_url,
    )
    imported = 0
    imported_validator_ids: set[str] = set()
    for raw in raw_records:
        validator_id = str(raw.get("validator_id") or "").strip().lower()
        if not validator_id:
            continue
        imported_validator_ids.add(validator_id)
        resolved_record_source_url = resolve_validator_peer_url(
            source_url=str(raw.get("source_url") or source_url).strip() or None,
            advertised_api_host=raw.get("advertised_api_host"),
        ) or source_url
        incoming = ValidatorRecord(
            validator_id=validator_id,
            wallet_id=raw.get("wallet_id"),
            address=str(raw.get("address") or validator_id).strip().lower(),
            state=str(raw.get("state") or ValidatorLifecycleState.UNBONDED),
            bonded_atomic=max(0, int(raw.get("bonded_atomic") or 0)),
            static_ip_confirmed=bool(raw.get("static_ip_confirmed")),
            current_node_id=raw.get("current_node_id"),
            advertised_api_host=raw.get("advertised_api_host"),
            advertised_data_host=raw.get("advertised_data_host"),
            ha_enabled=raw.get("ha_enabled") if "ha_enabled" in raw else None,
            active_replica_node_id=raw.get("active_replica_node_id"),
            active_replica_lease_until=raw.get("active_replica_lease_until"),
            replica_node_ids=_merged_replica_node_ids(raw.get("replica_node_ids")),
            activated_at=raw.get("activated_at"),
            unbonding_started_at=raw.get("unbonding_started_at"),
            unbonding_available_at=raw.get("unbonding_available_at"),
            jailed_at=raw.get("jailed_at"),
            unjail_available_at=raw.get("unjail_available_at"),
            last_slash_atomic=max(0, int(raw.get("last_slash_atomic") or 0)),
            total_slashed_atomic=max(0, int(raw.get("total_slashed_atomic") or 0)),
            source="peer",
            source_url=resolved_record_source_url,
            last_seen_at=_now_iso(),
            updated_at=raw.get("updated_at"),
        )
        existing = get_validator_record(incoming.validator_id, policy)
        if existing is not None and _should_merge_remote_ha_lease(existing, incoming):
            sync_validator_record(
                validator_id=existing.validator_id,
                wallet_id=existing.wallet_id,
                address=existing.address,
                state=existing.state,
                bonded_atomic=existing.bonded_atomic,
                static_ip_confirmed=existing.static_ip_confirmed,
                current_node_id=incoming.current_node_id,
                advertised_api_host=incoming.advertised_api_host,
                advertised_data_host=incoming.advertised_data_host,
                ha_enabled=True,
                active_replica_node_id=incoming.active_replica_node_id,
                active_replica_lease_until=incoming.active_replica_lease_until,
                replica_node_ids=_merged_replica_node_ids(
                    existing.replica_node_ids,
                    incoming.replica_node_ids,
                    incoming.current_node_id,
                ),
                unbonding_started_at=existing.unbonding_started_at,
                unbonding_available_at=existing.unbonding_available_at,
                jailed_at=existing.jailed_at,
                unjail_available_at=existing.unjail_available_at,
                last_slash_atomic=existing.last_slash_atomic,
                total_slashed_atomic=existing.total_slashed_atomic,
                source=existing.source,
                source_url=existing.source_url,
                last_seen_at=existing.last_seen_at,
                updated_at=incoming.updated_at or _now_iso(),
                policy=policy,
            )
            imported += 1
            continue
        if existing is not None and not _should_replace_record(existing, incoming):
            continue
        sync_validator_record(
            validator_id=incoming.validator_id,
            wallet_id=incoming.wallet_id,
            address=incoming.address,
            state=incoming.state,
            bonded_atomic=incoming.bonded_atomic,
            static_ip_confirmed=incoming.static_ip_confirmed,
            current_node_id=incoming.current_node_id,
            advertised_api_host=incoming.advertised_api_host,
            advertised_data_host=incoming.advertised_data_host,
            ha_enabled=incoming.ha_enabled,
            active_replica_node_id=incoming.active_replica_node_id,
            active_replica_lease_until=incoming.active_replica_lease_until,
            replica_node_ids=incoming.replica_node_ids,
            unbonding_started_at=incoming.unbonding_started_at,
            unbonding_available_at=incoming.unbonding_available_at,
            jailed_at=incoming.jailed_at,
            unjail_available_at=incoming.unjail_available_at,
            last_slash_atomic=incoming.last_slash_atomic,
            total_slashed_atomic=incoming.total_slashed_atomic,
            source=incoming.source,
            source_url=incoming.source_url,
            last_seen_at=incoming.last_seen_at,
            updated_at=incoming.updated_at,
            policy=policy,
        )
        imported += 1
    if authoritative_node_ids and imported_validator_ids:
        _prune_peer_validator_records_for_authoritative_nodes(
            authoritative_node_ids=authoritative_node_ids,
            authoritative_validator_ids=imported_validator_ids,
            policy=policy,
        )
    if authoritative_sources:
        _prune_peer_validator_records_for_authoritative_sources(
            authoritative_validator_ids_by_source_url=authoritative_sources,
            policy=policy,
        )
    return imported


def _authoritative_payload_validator_node_ids(
    payload: dict[str, Any],
    raw_records: list[dict[str, Any]],
) -> set[str]:
    signer_addresses = _payload_signature_addresses(payload)
    if not signer_addresses:
        return set()

    node_ids: set[str] = set()
    for raw in raw_records:
        state = str(raw.get("state") or "").strip().lower()
        try:
            bonded_atomic = int(raw.get("bonded_atomic") or 0)
        except (TypeError, ValueError):
            bonded_atomic = 0
        if state != ValidatorLifecycleState.BONDED or bonded_atomic <= 0:
            continue
        record_addresses = {
            str(raw.get("validator_id") or "").strip().lower(),
            str(raw.get("address") or "").strip().lower(),
        }
        record_addresses.discard("")
        if not (record_addresses & signer_addresses):
            continue
        node_id = str(raw.get("current_node_id") or "").strip()
        if node_id:
            node_ids.add(node_id)
    return node_ids


def _payload_signature_addresses(payload: dict[str, Any]) -> set[str]:
    signature = payload.get("signature")
    if not isinstance(signature, dict):
        return set()
    addresses = {
        str(signature.get("signer_address") or "").strip().lower(),
        str(signature.get("public_key_address") or "").strip().lower(),
    }
    addresses.discard("")
    return addresses


def _prune_peer_validator_records_for_authoritative_nodes(
    *,
    authoritative_node_ids: set[str],
    authoritative_validator_ids: set[str],
    policy: WalletPolicy | None = None,
) -> None:
    existing = list_validator_records(policy)
    if not existing:
        return
    pruned = [
        item
        for item in existing
        if not (
            item.source == "peer"
            and item.validator_id not in authoritative_validator_ids
            and str(item.current_node_id or "").strip() in authoritative_node_ids
        )
    ]
    if len(pruned) != len(existing):
        save_validator_records(pruned, policy)


def _prune_peer_validator_records_for_authoritative_sources(
    *,
    authoritative_validator_ids_by_source_url: dict[str, set[str]],
    policy: WalletPolicy | None = None,
) -> None:
    if not authoritative_validator_ids_by_source_url:
        return
    existing = list_validator_records(policy)
    if not existing:
        return
    pruned = [
        item
        for item in existing
        if not (
            item.source == "peer"
            and (resolved_source_url := resolve_validator_peer_url(
                source_url=getattr(item, "source_url", None),
                advertised_api_host=getattr(item, "advertised_api_host", None),
            ))
            and resolved_source_url in authoritative_validator_ids_by_source_url
            and item.validator_id
            not in authoritative_validator_ids_by_source_url[resolved_source_url]
        )
    ]
    if len(pruned) != len(existing):
        save_validator_records(pruned, policy)


def _iter_remote_validator_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("records"):
        return list(payload.get("records") or [])

    if payload.get("validators"):
        normalized: list[dict[str, Any]] = []
        for raw in payload.get("validators") or []:
            validator_id = str(raw.get("validatorId") or raw.get("address") or "").strip().lower()
            if not validator_id:
                continue
            normalized.append(
                {
                    "validator_id": validator_id,
                    "wallet_id": raw.get("walletId"),
                    "address": str(raw.get("address") or validator_id).strip().lower(),
                    "state": raw.get("state"),
                    "source_url": raw.get("sourceUrl"),
                    "bonded_atomic": raw.get("bondedAtomic"),
                    "static_ip_confirmed": raw.get("staticIpConfirmed"),
                    "current_node_id": raw.get("nodeId"),
                    "advertised_api_host": raw.get("apiHost"),
                    "advertised_data_host": raw.get("dataHost"),
                    "ha_enabled": raw.get("haEnabled") if "haEnabled" in raw else None,
                    "active_replica_node_id": raw.get("activeReplicaNodeId"),
                    "active_replica_lease_until": raw.get("activeReplicaLeaseUntil"),
                    "replica_node_ids": raw.get("replicaNodeIds") or [],
                    "activated_at": raw.get("activatedAt"),
                    "unbonding_started_at": raw.get("unbondingStartedAt"),
                    "unbonding_available_at": raw.get("unbondingAvailableAt"),
                    "jailed_at": raw.get("jailedAt"),
                    "unjail_available_at": raw.get("unjailAvailableAt"),
                    "last_slash_atomic": raw.get("lastSlashAtomic"),
                    "total_slashed_atomic": raw.get("totalSlashedAtomic"),
                    "updated_at": raw.get("updatedAt"),
                }
            )
        return normalized

    return []


def _authoritative_payload_validator_sources(
    payload: dict[str, Any],
    raw_records: list[dict[str, Any]],
    *,
    source_url: str,
) -> dict[str, set[str]]:
    signer_addresses = _payload_signature_addresses(payload)
    if not signer_addresses:
        return {}

    authoritative: dict[str, set[str]] = {}
    for raw in raw_records:
        validator_id = str(raw.get("validator_id") or "").strip().lower()
        if not validator_id:
            continue
        state = str(raw.get("state") or "").strip().lower()
        try:
            bonded_atomic = int(raw.get("bonded_atomic") or 0)
        except (TypeError, ValueError):
            bonded_atomic = 0
        if state != ValidatorLifecycleState.BONDED or bonded_atomic <= 0:
            continue
        record_addresses = {
            validator_id,
            str(raw.get("address") or "").strip().lower(),
        }
        record_addresses.discard("")
        if not (record_addresses & signer_addresses):
            continue
        resolved_record_source_url = resolve_validator_peer_url(
            source_url=str(raw.get("source_url") or source_url).strip() or None,
            advertised_api_host=raw.get("advertised_api_host"),
        ) or source_url
        authoritative.setdefault(resolved_record_source_url, set()).add(validator_id)
    return authoritative


def _select_weighted_committee(
    bonded: list[ValidatorRecord],
    *,
    desired_size: int,
    selection_seed: str,
) -> list[ValidatorRecord]:
    ranked: list[tuple[float, str, ValidatorRecord]] = []
    for item in bonded:
        weight = max(1, int(item.bonded_atomic))
        uniform_value = _deterministic_uniform(selection_seed, item.validator_id)
        key = math.log(uniform_value) / float(weight)
        ranked.append((key, item.validator_id, item))
    ranked.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
    return [entry[2] for entry in ranked[:desired_size]]


def _deterministic_uniform(selection_seed: str, validator_id: str) -> float:
    digest = hashlib.sha256(
        f"{selection_seed}:{validator_id}".encode("utf-8")
    ).digest()
    numerator = int.from_bytes(digest, "big") + 1
    denominator = float((1 << (8 * len(digest))) + 1)
    return numerator / denominator


def _peer_error_payload(peer_url: str, exc: Exception) -> dict[str, str]:
    return {
        "peerUrl": peer_url,
        "errorType": type(exc).__name__,
        "message": str(exc),
    }


def sync_validator_set_from_cai_peers(
    *,
    state_payload: dict[str, Any],
    cai_url: str | None = None,
    CAI_url: str | None = None,
    policy: WalletPolicy | None = None,
    local_node_id: str | None = None,
    timeout_sec: int = 5,
) -> ValidatorSetSyncResult:
    resolved_cai_url = _coalesce_cai_url(cai_url, CAI_url)
    peer_urls = discover_peer_cai_urls(
        state_payload=state_payload,
        cai_url=resolved_cai_url,
        endpoint_path="/v1/cai/validators",
        local_node_id=local_node_id,
    )
    imported_records = 0
    successful_peers = 0
    failed_peer_urls: list[str] = []
    peer_errors: list[dict[str, str]] = []
    for peer_url in peer_urls:
        try:
            with urlopen(peer_url, timeout=timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
            imported_records += merge_remote_validator_set_payload(
                payload,
                source_url=peer_url,
                policy=policy,
            )
            successful_peers += 1
        except Exception as exc:
            failed_peer_urls.append(peer_url)
            peer_errors.append(_peer_error_payload(peer_url, exc))
            continue
    return ValidatorSetSyncResult(
        attempted_peers=len(peer_urls),
        successful_peers=successful_peers,
        imported_records=imported_records,
        peer_urls=peer_urls,
        failed_peers=len(failed_peer_urls),
        failed_peer_urls=failed_peer_urls,
        peer_errors=peer_errors,
    )


def apply_remote_validator_penalty_to_registry(
    *,
    validator_id: str,
    slash_atomic: int = 0,
    jailed: bool = False,
    observed_at: str | None = None,
    source_url: str | None = None,
    policy: WalletPolicy | None = None,
) -> bool:
    normalized_validator_id = str(validator_id or "").strip().lower()
    if not normalized_validator_id:
        return False

    existing = get_validator_record(normalized_validator_id, policy)
    if existing is None:
        return False

    # Local validator state remains authoritative on the node that owns it.
    if existing.source == "local":
        return False

    effective_slash_atomic = max(0, int(slash_atomic))
    target_state = existing.state
    target_bonded_atomic = existing.bonded_atomic
    target_unbonding_started_at = existing.unbonding_started_at
    target_unbonding_available_at = existing.unbonding_available_at
    target_jailed_at = existing.jailed_at

    if jailed:
        target_state = ValidatorLifecycleState.JAILED
        target_bonded_atomic = 0
        target_unbonding_started_at = None
        target_unbonding_available_at = None
        target_jailed_at = observed_at or existing.jailed_at
    elif effective_slash_atomic > 0:
        target_bonded_atomic = max(0, existing.bonded_atomic - effective_slash_atomic)

    target_last_slash_atomic = (
        effective_slash_atomic if effective_slash_atomic > 0 else existing.last_slash_atomic
    )
    target_total_slashed_atomic = (
        existing.total_slashed_atomic + effective_slash_atomic
        if effective_slash_atomic > 0
        else existing.total_slashed_atomic
    )

    if (
        target_state == existing.state
        and target_bonded_atomic == existing.bonded_atomic
        and target_unbonding_started_at == existing.unbonding_started_at
        and target_unbonding_available_at == existing.unbonding_available_at
        and target_jailed_at == existing.jailed_at
        and target_last_slash_atomic == existing.last_slash_atomic
        and target_total_slashed_atomic == existing.total_slashed_atomic
    ):
        return False

    sync_validator_record(
        validator_id=existing.validator_id,
        wallet_id=existing.wallet_id,
        address=existing.address,
        state=target_state,
        bonded_atomic=target_bonded_atomic,
        static_ip_confirmed=existing.static_ip_confirmed,
        current_node_id=existing.current_node_id,
        advertised_api_host=existing.advertised_api_host,
        advertised_data_host=existing.advertised_data_host,
        unbonding_started_at=target_unbonding_started_at,
        unbonding_available_at=target_unbonding_available_at,
        jailed_at=target_jailed_at,
        unjail_available_at=existing.unjail_available_at,
        last_slash_atomic=target_last_slash_atomic,
        total_slashed_atomic=target_total_slashed_atomic,
        source=existing.source,
        source_url=existing.source_url or source_url,
        last_seen_at=existing.last_seen_at,
        policy=policy,
    )
    return True


def discover_peer_cai_urls(
    *,
    state_payload: dict[str, Any],
    cai_url: str | None = None,
    CAI_url: str | None = None,
    endpoint_path: str,
    local_node_id: str | None = None,
) -> list[str]:
    resolved_cai_url = _coalesce_cai_url(cai_url, CAI_url)
    identities = state_payload.get("nodeIdentities") or {}
    resolved_local_node_id = _resolve_local_node_id(
        state_payload=state_payload,
        cai_url=resolved_cai_url,
        local_node_id=local_node_id,
    )
    normalized_path = "/" + str(endpoint_path or "").lstrip("/")
    urls: list[str] = []
    local_identity_urls: set[str] = set()
    local_identity = (
        identities.get(resolved_local_node_id) if resolved_local_node_id else None
    )
    if isinstance(local_identity, dict):
        local_identity_urls.update(
            candidate_identity_http_urls(
                local_identity,
                endpoint_path=normalized_path,
            )
        )
    for node_id, identity in identities.items():
        if node_id == resolved_local_node_id:
            continue
        if not isinstance(identity, dict):
            continue
        urls.extend(
            candidate_identity_http_urls(
                identity,
                endpoint_path=normalized_path,
            )
        )
    for url in _bootstrap_peer_cai_urls(endpoint_path=normalized_path):
        if url in local_identity_urls:
            continue
        urls.append(url)
    seen: set[str] = set()
    deduped: list[str] = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def _bootstrap_peer_cai_urls(*, endpoint_path: str) -> list[str]:
    normalized_path = "/" + str(endpoint_path or "").lstrip("/")
    urls: list[str] = []
    for peer in default_bootstrap_peers():
        base_url = _api_base_url_from_multiaddr(peer, default_api_port())
        if not base_url:
            continue
        urls.append(base_url.rstrip("/") + normalized_path)
    return urls


def _api_base_url_from_multiaddr(peer: str, api_port: int) -> str | None:
    normalized = str(peer or "").strip()
    if not normalized:
        return None

    ip4_match = re.match(r"^/ip4/([^/]+)", normalized)
    if ip4_match:
        return f"http://{ip4_match.group(1)}:{int(api_port)}"

    ip6_match = re.match(r"^/ip6/([^/]+)", normalized)
    if ip6_match:
        return f"http://[{ip6_match.group(1)}]:{int(api_port)}"

    dns_match = re.match(r"^/dns(?:4|6)?/([^/]+)", normalized)
    if dns_match:
        return f"http://{dns_match.group(1)}:{int(api_port)}"

    return None


def _normalize_host(value: object) -> str:
    return str(value or "").strip().lower()


def _is_loopback_host(host: str) -> bool:
    normalized = _normalize_host(host)
    if not normalized:
        return False
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _hosts_match(left: object, right: object) -> bool:
    left_host = _normalize_host(left)
    right_host = _normalize_host(right)
    if not left_host or not right_host:
        return False
    if left_host == right_host:
        return True
    if _is_loopback_host(left_host) and _is_loopback_host(right_host):
        return True
    try:
        return ipaddress.ip_address(left_host) == ipaddress.ip_address(right_host)
    except ValueError:
        return False


def _resolve_local_node_id(
    *,
    state_payload: dict[str, Any],
    cai_url: str | None = None,
    CAI_url: str | None = None,
    local_node_id: str | None = None,
) -> str | None:
    normalized_local_node_id = str(local_node_id or "").strip()
    if normalized_local_node_id:
        return normalized_local_node_id

    parsed = urlparse(_coalesce_cai_url(cai_url, CAI_url))
    target_port = parsed.port
    target_host = _normalize_host(parsed.hostname)
    identities = state_payload.get("nodeIdentities") or {}
    if target_port is not None:
        exact_matches: list[str] = []
        port_matches: list[str] = []
        for node_id, info in identities.items():
            try:
                api_port = int(info.get("apiPort", -1))
            except (TypeError, ValueError):
                continue
            if api_port != int(target_port):
                continue
            normalized_node_id = str(node_id).strip()
            if not normalized_node_id:
                continue
            port_matches.append(normalized_node_id)
            if _hosts_match(info.get("apiHost"), target_host):
                exact_matches.append(normalized_node_id)
        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(port_matches) == 1 and (
            len(identities) > 1 or not _is_loopback_host(target_host)
        ):
            return port_matches[0]
    if len(identities) == 1:
        node_id, info = next(iter(identities.items()))
        normalized_node_id = str(node_id).strip()
        if normalized_node_id and (
            not target_host or _hosts_match(info.get("apiHost"), target_host)
        ):
            return normalized_node_id
    return None


def _should_replace_record(existing: ValidatorRecord, incoming: ValidatorRecord) -> bool:
    if existing.source == "local" and incoming.source != "local":
        return False
    if incoming.source == "local" and existing.source != "local":
        return True

    existing_updated = _parse_iso_datetime(existing.updated_at)
    incoming_updated = _parse_iso_datetime(incoming.updated_at)
    if existing_updated and incoming_updated and incoming_updated > existing_updated:
        return True
    if existing_updated and incoming_updated and incoming_updated < existing_updated:
        return False
    if incoming.total_slashed_atomic > existing.total_slashed_atomic:
        return True
    severity = {
        ValidatorLifecycleState.JAILED: 3,
        ValidatorLifecycleState.UNBONDING: 2,
        ValidatorLifecycleState.BONDED: 1,
        ValidatorLifecycleState.UNBONDED: 0,
    }
    return severity.get(incoming.state, -1) >= severity.get(existing.state, -1)


def _should_merge_remote_ha_lease(
    existing: ValidatorRecord,
    incoming: ValidatorRecord,
) -> bool:
    if incoming.source == "local" or not incoming.ha_enabled:
        return False
    if incoming.state != ValidatorLifecycleState.BONDED:
        return False
    if not incoming.active_replica_node_id or not incoming.active_replica_lease_until:
        return False
    if not existing.ha_enabled and existing.source == "local":
        return False

    incoming_lease = _parse_iso_datetime(incoming.active_replica_lease_until)
    if incoming_lease is None:
        return False
    existing_lease = _parse_iso_datetime(existing.active_replica_lease_until)
    if existing_lease is None:
        return True
    return incoming_lease >= existing_lease


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None

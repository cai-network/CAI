# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.request import Request, urlopen

from .local_json_store import atomic_write_json_object_file, read_json_object_file
from .model import MoneyPolicy, ValidatorLifecycleState, WalletPolicy
from .peer_payload import (
    peer_payload_hybrid_signatures_required,
    peer_payload_signatures_required,
    validate_peer_payload_network,
    verify_peer_payload_signature,
)
from .validators import discover_peer_cai_urls, list_bonded_validators
from .wallet import atomic_to_coins, coins_to_atomic, data_root, normalize_address
from .wallet_signing import (
    ADDRESS_SCHEME_FIXED_WALLET_ID_SHA256,
    ADDRESS_SCHEME_ED25519,
    HYBRID_ADDRESS_SCHEMES,
    SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65,
    SIGNING_SCHEME_ML_DSA_65,
    address_from_public_key_b64,
    hybrid_address_from_public_keys_b64,
    verify_hybrid_payload_signature,
    verify_payload_signature,
)


GENESIS_PREVIOUS_HASH = "0" * 64
GENESIS_CREATED_AT = "2026-01-01T00:00:00+00:00"
COMPUTE_RESERVE_ADDRESS_PREFIX = "system:compute-reserve"
DEVELOPER_CONTRIBUTION_FUND_ADDRESS_PREFIX = "system:developer-contribution-fund"
TX_FEE_POOL_ADDRESS_PREFIX = "system:tx-fee-pool"
VALIDATOR_SETTLEMENT_FEE_POOL_ADDRESS_PREFIX = "system:validator-settlement-fee-pool"
VALIDATOR_BOND_POOL_ADDRESS_PREFIX = "system:validator-bond-pool"
VALIDATOR_SLASH_POOL_ADDRESS_PREFIX = "system:validator-slash-pool"
CHAIN_SCHEMA_VERSION = 1
CHAIN_INDEX_SCHEMA_VERSION = 1
CHAIN_SNAPSHOT_SCHEMA_VERSION = 1
CHAIN_INDEX_FILE_NAME = "chain-index.json"
CHAIN_SNAPSHOTS_FILE_NAME = "chain-snapshots.json"
DEFAULT_CHAIN_SNAPSHOT_INTERVAL = 100


LOGGER = logging.getLogger(__name__)


@dataclass
class ChainTransaction:
    tx_id: str
    created_at: str
    tx_type: str
    address: str
    delta_atomic: int
    settlement_id: str | None = None
    payout_id: str | None = None
    wallet_id: str | None = None
    counterparty_address: str | None = None
    note: str | None = None
    nonce: str | None = None
    chain_id: str = ""
    schema_version: int = CHAIN_SCHEMA_VERSION
    public_key_b64: str | None = None
    signature_b64: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChainBlock:
    block_id: str
    height: int
    created_at: str
    previous_hash: str
    validator_id: str | None
    transactions: list[ChainTransaction]
    block_hash: str
    chain_id: str = ""
    schema_version: int = CHAIN_SCHEMA_VERSION
    tx_root: str = ""
    state_root: str = ""
    validator_public_key_b64: str | None = None
    validator_signature_b64: str | None = None


@dataclass(frozen=True)
class ChainSyncResult:
    attempted_peers: int
    successful_peers: int
    imported_blocks: int
    imported_transactions: int
    peer_urls: list[str]
    failed_peers: int = 0
    failed_peer_urls: list[str] = field(default_factory=list)
    peer_errors: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class ChainPushResult:
    attempted_peers: int
    successful_peers: int
    peer_urls: list[str]
    failed_peers: int = 0
    failed_peer_urls: list[str] = field(default_factory=list)
    peer_errors: list[dict[str, str]] = field(default_factory=list)


def chain_file_path(policy: WalletPolicy | None = None):
    active_policy = policy or WalletPolicy()
    return data_root(active_policy) / active_policy.chain_file_name


def chain_index_file_path(policy: WalletPolicy | None = None):
    return data_root(policy or WalletPolicy()) / CHAIN_INDEX_FILE_NAME


def chain_snapshots_file_path(policy: WalletPolicy | None = None):
    return data_root(policy or WalletPolicy()) / CHAIN_SNAPSHOTS_FILE_NAME


def _money_policy_for_wallet_policy(
    wallet_policy: WalletPolicy | None = None,
    money_policy: MoneyPolicy | None = None,
) -> MoneyPolicy:
    if money_policy is not None:
        return money_policy
    if wallet_policy is not None:
        return MoneyPolicy(chain_network=wallet_policy.chain_network)
    return MoneyPolicy()


def _default_chain_id() -> str:
    return MoneyPolicy().chain_network.value


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _canonical(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _transaction_payload(tx: ChainTransaction) -> dict[str, Any]:
    payload = asdict(tx)
    payload["address"] = normalize_address(payload["address"])
    if payload.get("counterparty_address"):
        payload["counterparty_address"] = normalize_address(payload["counterparty_address"])
    payload["delta_atomic"] = int(payload["delta_atomic"])
    payload["chain_id"] = str(payload.get("chain_id") or _default_chain_id())
    payload["schema_version"] = int(payload.get("schema_version") or CHAIN_SCHEMA_VERSION)
    return payload


def transaction_signing_payload(tx: ChainTransaction) -> dict[str, Any]:
    payload = _transaction_payload(tx)
    payload["signature_b64"] = None
    metadata = dict(payload.get("metadata") or {})
    metadata.pop("pq_signature_b64", None)
    payload["metadata"] = metadata
    return payload


def signable_wallet_transaction_types() -> set[str]:
    return {"wallet_transfer_debit"}


def verify_chain_transaction_signature(
    tx: ChainTransaction,
    *,
    policy: WalletPolicy | None = None,
) -> tuple[bool, str | None]:
    metadata = dict(tx.metadata or {})
    signed_tx_type = tx.tx_type in signable_wallet_transaction_types()
    signature_required = bool(metadata.get("signature_required")) or signed_tx_type
    if not signature_required:
        return True, None
    if not tx.public_key_b64:
        return False, f"transaction {tx.tx_id} is missing public_key_b64"
    if not tx.signature_b64:
        return False, f"transaction {tx.tx_id} is missing signature_b64"
    address_scheme = str(metadata.get("address_scheme") or "").strip()
    if not address_scheme:
        return False, f"transaction {tx.tx_id} is missing address scheme"
    active_policy = policy or WalletPolicy()
    requires_post_quantum = bool(
        getattr(active_policy, "require_post_quantum_wallet_signatures", False)
    )
    if address_scheme == ADDRESS_SCHEME_ED25519:
        if requires_post_quantum:
            return False, f"transaction {tx.tx_id} requires a post-quantum signature"
        expected_address = address_from_public_key_b64(tx.public_key_b64)
        if normalize_address(tx.address) != expected_address:
            return False, f"transaction {tx.tx_id} public key does not match address"
    elif address_scheme in HYBRID_ADDRESS_SCHEMES:
        pq_public_key_b64 = str(metadata.get("pq_public_key_b64") or "").strip()
        pq_signature_b64 = str(metadata.get("pq_signature_b64") or "").strip()
        pq_signature_scheme = str(metadata.get("pq_signature_scheme") or "").strip()
        if pq_signature_scheme and pq_signature_scheme != SIGNING_SCHEME_ML_DSA_65:
            return False, f"transaction {tx.tx_id} has unsupported post-quantum signature scheme"
        if not pq_public_key_b64 or not pq_signature_b64:
            return False, f"transaction {tx.tx_id} is missing post-quantum signature"
        expected_address = hybrid_address_from_public_keys_b64(
            ed25519_public_key_b64=tx.public_key_b64,
            pq_public_key_b64=pq_public_key_b64,
            address_scheme=address_scheme,
        )
        if normalize_address(tx.address) != expected_address:
            return False, f"transaction {tx.tx_id} public keys do not match address"
        if not verify_hybrid_payload_signature(
            ed25519_public_key_b64=tx.public_key_b64,
            ed25519_signature_b64=tx.signature_b64,
            pq_public_key_b64=pq_public_key_b64,
            pq_signature_b64=pq_signature_b64,
            payload=transaction_signing_payload(tx),
        ):
            return False, f"transaction {tx.tx_id} has invalid hybrid signature"
        return True, None
    elif address_scheme == ADDRESS_SCHEME_FIXED_WALLET_ID_SHA256:
        signing_scheme = str(metadata.get("signing_scheme") or "").strip()
        pq_public_key_b64 = str(metadata.get("pq_public_key_b64") or "").strip()
        pq_signature_b64 = str(metadata.get("pq_signature_b64") or "").strip()
        pq_signature_scheme = str(metadata.get("pq_signature_scheme") or "").strip()
        if signing_scheme != SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65:
            return False, f"transaction {tx.tx_id} fixed address requires hybrid signing scheme"
        if pq_signature_scheme != SIGNING_SCHEME_ML_DSA_65:
            return False, f"transaction {tx.tx_id} has unsupported post-quantum signature scheme"
        if not pq_public_key_b64 or not pq_signature_b64:
            return False, f"transaction {tx.tx_id} is missing post-quantum signature"
        if not verify_hybrid_payload_signature(
            ed25519_public_key_b64=tx.public_key_b64,
            ed25519_signature_b64=tx.signature_b64,
            pq_public_key_b64=pq_public_key_b64,
            pq_signature_b64=pq_signature_b64,
            payload=transaction_signing_payload(tx),
        ):
            return False, f"transaction {tx.tx_id} has invalid hybrid signature"
        return True, None
    elif signature_required:
        return False, f"transaction {tx.tx_id} has unsupported address scheme"
    if not verify_payload_signature(
        public_key_b64=tx.public_key_b64,
        signature_b64=tx.signature_b64,
        payload=transaction_signing_payload(tx),
    ):
        return False, f"transaction {tx.tx_id} has invalid signature"
    return True, None


def deterministic_tx_id(
    *,
    tx_type: str,
    address: str,
    delta_atomic: int,
    settlement_id: str | None = None,
    payout_id: str | None = None,
    wallet_id: str | None = None,
    counterparty_address: str | None = None,
    nonce: str | None = None,
    public_key_b64: str | None = None,
    chain_id: str | None = None,
    schema_version: int = CHAIN_SCHEMA_VERSION,
) -> str:
    resolved_chain_id = str(chain_id or _default_chain_id())
    payload = {
        "chain_id": resolved_chain_id,
        "schema_version": int(schema_version),
        "tx_type": str(tx_type),
        "address": normalize_address(address),
        "delta_atomic": int(delta_atomic),
        "settlement_id": settlement_id,
        "payout_id": payout_id,
        "wallet_id": wallet_id,
        "counterparty_address": (
            normalize_address(counterparty_address) if counterparty_address else None
        ),
        "nonce": str(nonce) if nonce is not None else None,
        "public_key_b64": str(public_key_b64) if public_key_b64 else None,
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()[:32]


def make_chain_transaction(
    *,
    tx_type: str,
    address: str,
    delta_atomic: int,
    settlement_id: str | None = None,
    payout_id: str | None = None,
    wallet_id: str | None = None,
    counterparty_address: str | None = None,
    note: str | None = None,
    nonce: str | None = None,
    public_key_b64: str | None = None,
    signature_b64: str | None = None,
    metadata: dict[str, Any] | None = None,
    tx_id: str | None = None,
    created_at: str | None = None,
    chain_id: str | None = None,
    schema_version: int = CHAIN_SCHEMA_VERSION,
) -> ChainTransaction:
    normalized_address = normalize_address(address)
    resolved_chain_id = str(
        chain_id
        or (metadata or {}).get("network")
        or _default_chain_id()
    )
    resolved_tx_id = tx_id or deterministic_tx_id(
        tx_type=tx_type,
        address=normalized_address,
        delta_atomic=delta_atomic,
        settlement_id=settlement_id,
        payout_id=payout_id,
        wallet_id=wallet_id,
        counterparty_address=counterparty_address,
        nonce=nonce,
        public_key_b64=public_key_b64,
        chain_id=resolved_chain_id,
        schema_version=schema_version,
    )
    return ChainTransaction(
        tx_id=resolved_tx_id,
        created_at=created_at or _now_iso(),
        tx_type=str(tx_type),
        address=normalized_address,
        delta_atomic=int(delta_atomic),
        settlement_id=settlement_id,
        payout_id=payout_id,
        wallet_id=wallet_id,
        counterparty_address=(
            normalize_address(counterparty_address) if counterparty_address else None
        ),
        note=note,
        nonce=str(nonce) if nonce is not None else None,
        chain_id=resolved_chain_id,
        schema_version=int(schema_version),
        public_key_b64=str(public_key_b64) if public_key_b64 else None,
        signature_b64=str(signature_b64) if signature_b64 else None,
        metadata=dict(metadata or {}),
    )


def _block_hash_payload(block: ChainBlock) -> dict[str, Any]:
    return {
        "chain_id": str(block.chain_id or _default_chain_id()),
        "schema_version": int(block.schema_version or CHAIN_SCHEMA_VERSION),
        "block_id": block.block_id,
        "height": int(block.height),
        "created_at": block.created_at,
        "previous_hash": block.previous_hash,
        "validator_id": block.validator_id,
        "transactions": [_transaction_payload(tx) for tx in block.transactions],
    }


def compute_block_hash(block: ChainBlock) -> str:
    return hashlib.sha256(_canonical(_block_hash_payload(block)).encode("utf-8")).hexdigest()


def compute_transaction_root(transactions: list[ChainTransaction]) -> str:
    payload = [_transaction_payload(tx) for tx in transactions]
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def compute_state_root_from_balances(balances: dict[str, int]) -> str:
    payload = [
        {"address": normalize_address(address), "balance_atomic": int(balance)}
        for address, balance in sorted(balances.items())
        if int(balance) != 0
    ]
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def compute_chain_state_root(blocks: list[ChainBlock]) -> str:
    return compute_state_root_from_balances(chain_balance_index(blocks=blocks))


def _coerce_transaction(
    raw: dict[str, Any],
    *,
    default_chain_id: str | None = None,
) -> ChainTransaction:
    raw = dict(raw)
    raw.setdefault("settlement_id", None)
    raw.setdefault("payout_id", None)
    raw.setdefault("wallet_id", None)
    raw.setdefault("counterparty_address", None)
    raw.setdefault("note", None)
    raw.setdefault("nonce", None)
    raw.setdefault("public_key_b64", None)
    raw.setdefault("signature_b64", None)
    raw.setdefault("metadata", {})
    metadata = dict(raw.get("metadata") or {})
    chain_id = str(
        raw.get("chain_id")
        or metadata.get("network")
        or default_chain_id
        or _default_chain_id()
    )
    return ChainTransaction(
        tx_id=str(raw["tx_id"]),
        created_at=str(raw.get("created_at") or _now_iso()),
        tx_type=str(raw["tx_type"]),
        address=normalize_address(str(raw["address"])),
        delta_atomic=int(raw["delta_atomic"]),
        settlement_id=raw.get("settlement_id"),
        payout_id=raw.get("payout_id"),
        wallet_id=raw.get("wallet_id"),
        counterparty_address=(
            normalize_address(str(raw["counterparty_address"]))
            if raw.get("counterparty_address")
            else None
        ),
        note=raw.get("note"),
        nonce=str(raw.get("nonce")) if raw.get("nonce") is not None else None,
        chain_id=chain_id,
        schema_version=int(raw.get("schema_version") or CHAIN_SCHEMA_VERSION),
        public_key_b64=(
            str(raw.get("public_key_b64"))
            if raw.get("public_key_b64") is not None
            else None
        ),
        signature_b64=(
            str(raw.get("signature_b64"))
            if raw.get("signature_b64") is not None
            else None
        ),
        metadata=metadata,
    )


def _coerce_block(
    raw: dict[str, Any],
    *,
    default_chain_id: str | None = None,
) -> ChainBlock:
    chain_id = str(raw.get("chain_id") or default_chain_id or _default_chain_id())
    block = ChainBlock(
        block_id=str(raw["block_id"]),
        height=int(raw["height"]),
        created_at=str(raw.get("created_at") or _now_iso()),
        previous_hash=str(raw.get("previous_hash") or GENESIS_PREVIOUS_HASH),
        validator_id=raw.get("validator_id"),
        transactions=[
            _coerce_transaction(item, default_chain_id=chain_id)
            for item in list(raw.get("transactions") or [])
        ],
        block_hash=str(raw.get("block_hash") or ""),
        chain_id=chain_id,
        schema_version=int(raw.get("schema_version") or CHAIN_SCHEMA_VERSION),
        tx_root=str(raw.get("tx_root") or raw.get("txRoot") or ""),
        state_root=str(raw.get("state_root") or raw.get("stateRoot") or ""),
    )
    if not block.block_hash:
        block.block_hash = compute_block_hash(block)
    return block


def _fill_missing_block_roots(blocks: list[ChainBlock]) -> None:
    prefix: list[ChainBlock] = []
    for block in blocks:
        if not block.tx_root:
            block.tx_root = compute_transaction_root(block.transactions)
        if not block.state_root:
            block.state_root = compute_chain_state_root([*prefix, block])
        prefix.append(block)


def _chain_snapshot_interval() -> int:
    raw = str(os.getenv("CAI_CHAIN_SNAPSHOT_INTERVAL") or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError as exc:
            _log_best_effort_failure("chain snapshot interval parse", exc)
    return DEFAULT_CHAIN_SNAPSHOT_INTERVAL


def _chain_file_stat_payload(policy: WalletPolicy | None = None) -> dict[str, Any]:
    path = chain_file_path(policy)
    try:
        stat = path.stat()
    except OSError:
        return {
            "path": path.name,
            "exists": False,
            "size": 0,
            "mtimeNs": 0,
        }
    return {
        "path": path.name,
        "exists": True,
        "size": int(stat.st_size),
        "mtimeNs": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
    }


def _index_matches_current_chain_file(
    payload: dict[str, Any],
    policy: WalletPolicy | None = None,
) -> bool:
    if not isinstance(payload, dict):
        return False
    if int(payload.get("schemaVersion") or 0) != CHAIN_INDEX_SCHEMA_VERSION:
        return False
    if int(payload.get("chainSchemaVersion") or 0) != CHAIN_SCHEMA_VERSION:
        return False
    active_chain_id = _money_policy_for_wallet_policy(policy).chain_network.value
    if str(payload.get("chainId") or "") != active_chain_id:
        return False
    current = _chain_file_stat_payload(policy)
    if not current.get("exists"):
        return False
    recorded = payload.get("chainFile") if isinstance(payload.get("chainFile"), dict) else {}
    return (
        bool(recorded.get("exists"))
        and int(recorded.get("size") or -1) == int(current.get("size") or 0)
        and int(recorded.get("mtimeNs") or -1) == int(current.get("mtimeNs") or 0)
    )


def _int_mapping(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    result: dict[str, int] = {}
    for key, value in raw.items():
        normalized_key = normalize_address(str(key))
        if not normalized_key:
            continue
        try:
            result[normalized_key] = int(value)
        except (TypeError, ValueError):
            continue
    return result


def _chain_index_balances(index_payload: dict[str, Any]) -> dict[str, int]:
    return _int_mapping(index_payload.get("balancesAtomic"))


def _chain_index_validator_locked_bonds(index_payload: dict[str, Any]) -> dict[str, int]:
    return _int_mapping(index_payload.get("validatorLockedBondAtomic"))


def _build_chain_snapshot_payload(
    block: ChainBlock,
    *,
    balances: dict[str, int],
    validator_locked_bonds: dict[str, int],
    transaction_count: int,
) -> dict[str, Any]:
    non_zero_balance_count = sum(1 for value in balances.values() if int(value) != 0)
    return {
        "schemaVersion": CHAIN_SNAPSHOT_SCHEMA_VERSION,
        "chainSchemaVersion": CHAIN_SCHEMA_VERSION,
        "chainId": str(block.chain_id or _default_chain_id()),
        "height": int(block.height),
        "blockHash": block.block_hash,
        "createdAt": block.created_at,
        "stateRoot": block.state_root,
        "txRoot": block.tx_root,
        "transactionCount": int(transaction_count),
        "balanceAddressCount": len(balances),
        "nonZeroBalanceAddressCount": non_zero_balance_count,
        "totalBalanceAtomic": sum(int(value) for value in balances.values()),
        "balancesAtomic": dict(sorted(balances.items())),
        "validatorLockedBondAtomic": dict(sorted(validator_locked_bonds.items())),
    }


def _apply_validator_bond_index_delta(
    validator_locked_bonds: dict[str, int],
    tx: ChainTransaction,
) -> None:
    if tx.tx_type not in {"validator_bond_lock", "validator_bond_pool_debit"}:
        return
    validator_id = _validator_id_from_bond_metadata(tx)
    if validator_id is None:
        return
    if tx.tx_type == "validator_bond_lock":
        delta = max(0, -int(tx.delta_atomic))
    else:
        delta = int(tx.delta_atomic)
    if delta == 0:
        return
    validator_locked_bonds[validator_id] = (
        validator_locked_bonds.get(validator_id, 0) + delta
    )


def _build_chain_index_payloads(
    blocks: list[ChainBlock],
    policy: WalletPolicy | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    active_chain_id = _money_policy_for_wallet_policy(policy).chain_network.value
    snapshot_interval = _chain_snapshot_interval()
    balances: dict[str, int] = {}
    validator_locked_bonds: dict[str, int] = {}
    seen_tx_ids: set[str] = set()
    transaction_index: dict[str, dict[str, Any]] = {}
    address_history: dict[str, list[dict[str, Any]]] = {}
    settlement_history: dict[str, list[dict[str, Any]]] = {}
    snapshots: list[dict[str, Any]] = []
    ordered_blocks = sorted(
        blocks,
        key=lambda item: (int(item.height), item.created_at, item.block_hash),
    )
    tip_block = ordered_blocks[-1] if ordered_blocks else None

    for block in ordered_blocks:
        for tx_index, tx in enumerate(block.transactions):
            if tx.tx_id in seen_tx_ids:
                continue
            seen_tx_ids.add(tx.tx_id)
            address = normalize_address(tx.address)
            balances[address] = balances.get(address, 0) + int(tx.delta_atomic)
            _apply_validator_bond_index_delta(validator_locked_bonds, tx)
            entry = _chain_transaction_history_entry(
                block,
                tx,
                balance_after_atomic=balances[address],
            )
            address_history.setdefault(address, []).append(entry)
            if tx.settlement_id:
                settlement_history.setdefault(str(tx.settlement_id), []).append(
                    _chain_transaction_history_entry(block, tx)
                )
            transaction_index[tx.tx_id] = {
                "blockHeight": int(block.height),
                "blockHash": block.block_hash,
                "blockCreatedAt": block.created_at,
                "txIndex": int(tx_index),
                "address": address,
                "txType": tx.tx_type,
                "settlementId": tx.settlement_id,
            }
        if (
            int(block.height) % snapshot_interval == 0
            or block is tip_block
        ):
            snapshots.append(
                _build_chain_snapshot_payload(
                    block,
                    balances=balances,
                    validator_locked_bonds=validator_locked_bonds,
                    transaction_count=len(seen_tx_ids),
                )
            )

    validation_errors = validate_chain_blocks(ordered_blocks, policy=policy)
    latest_snapshot = snapshots[-1] if snapshots else None
    index_payload = {
        "schemaVersion": CHAIN_INDEX_SCHEMA_VERSION,
        "chainSchemaVersion": CHAIN_SCHEMA_VERSION,
        "chainId": active_chain_id,
        "indexedAt": _now_iso(),
        "chainFile": _chain_file_stat_payload(policy),
        "blockCount": len(ordered_blocks),
        "transactionCount": len(seen_tx_ids),
        "tipHeight": tip_block.height if tip_block is not None else None,
        "tipHash": tip_block.block_hash if tip_block is not None else None,
        "tipCreatedAt": tip_block.created_at if tip_block is not None else None,
        "tipTxRoot": tip_block.tx_root if tip_block is not None else None,
        "tipStateRoot": tip_block.state_root if tip_block is not None else None,
        "valid": not validation_errors,
        "validationErrors": validation_errors[:20],
        "balancesAtomic": dict(sorted(balances.items())),
        "validatorLockedBondAtomic": dict(sorted(validator_locked_bonds.items())),
        "transactionIndex": dict(sorted(transaction_index.items())),
        "addressHistory": {
            address: entries
            for address, entries in sorted(address_history.items())
        },
        "settlementHistory": {
            settlement_id: entries
            for settlement_id, entries in sorted(settlement_history.items())
        },
        "snapshotInterval": snapshot_interval,
        "snapshotCount": len(snapshots),
        "latestSnapshotHeight": (
            latest_snapshot.get("height") if latest_snapshot is not None else None
        ),
        "latestSnapshotHash": (
            latest_snapshot.get("blockHash") if latest_snapshot is not None else None
        ),
        "latestSnapshotStateRoot": (
            latest_snapshot.get("stateRoot") if latest_snapshot is not None else None
        ),
    }
    snapshots_payload = {
        "schemaVersion": CHAIN_SNAPSHOT_SCHEMA_VERSION,
        "chainSchemaVersion": CHAIN_SCHEMA_VERSION,
        "chainId": active_chain_id,
        "createdAt": _now_iso(),
        "chainFile": _chain_file_stat_payload(policy),
        "snapshotInterval": snapshot_interval,
        "latest": latest_snapshot,
        "snapshots": snapshots,
    }
    return index_payload, snapshots_payload


def rebuild_chain_indexes(
    policy: WalletPolicy | None = None,
    *,
    blocks: list[ChainBlock] | None = None,
) -> dict[str, Any]:
    resolved_blocks = list_chain_blocks(policy) if blocks is None else blocks
    index_payload, snapshots_payload = _build_chain_index_payloads(
        resolved_blocks,
        policy,
    )
    atomic_write_json_object_file(chain_snapshots_file_path(policy), snapshots_payload)
    atomic_write_json_object_file(chain_index_file_path(policy), index_payload)
    return index_payload


def load_chain_index(policy: WalletPolicy | None = None) -> dict[str, Any]:
    path = chain_index_file_path(policy)
    if not path.exists():
        return {}
    return read_json_object_file(path, heal_corrupt=True)


def load_or_rebuild_chain_index(
    policy: WalletPolicy | None = None,
    *,
    blocks: list[ChainBlock] | None = None,
) -> dict[str, Any]:
    if blocks is None:
        cached = load_chain_index(policy)
        if _index_matches_current_chain_file(cached, policy):
            return cached
        chain_path = chain_file_path(policy)
        if not chain_path.exists():
            return {}
        blocks = list_chain_blocks(policy)
    if not blocks:
        return {}
    return rebuild_chain_indexes(policy, blocks=blocks)


def list_chain_blocks(policy: WalletPolicy | None = None) -> list[ChainBlock]:
    active_money_policy = _money_policy_for_wallet_policy(policy)
    path = chain_file_path(policy)
    if not path.exists():
        return []
    raw = read_json_object_file(path, heal_corrupt=True)
    raw_blocks = raw.get("blocks") if isinstance(raw, dict) else raw
    blocks = [
        _coerce_block(item, default_chain_id=active_money_policy.chain_network.value)
        for item in list(raw_blocks or [])
    ]
    blocks.sort(key=lambda item: (int(item.height), item.created_at, item.block_hash))
    _fill_missing_block_roots(blocks)
    return blocks


def save_chain_blocks(blocks: list[ChainBlock], policy: WalletPolicy | None = None) -> None:
    path = chain_file_path(policy)
    atomic_write_json_object_file(
        path,
        {"blocks": [asdict(item) for item in blocks]},
    )
    rebuild_chain_indexes(policy, blocks=blocks)


def chain_transaction_ids(policy: WalletPolicy | None = None) -> set[str]:
    return {
        tx.tx_id
        for block in list_chain_blocks(policy)
        for tx in block.transactions
    }


def _chain_transaction_nonces(blocks: list[ChainBlock]) -> set[tuple[str, str]]:
    return {
        (normalize_address(tx.address), str(tx.nonce))
        for block in blocks
        for tx in block.transactions
        if tx.nonce is not None
    }


def append_chain_block(
    transactions: list[ChainTransaction],
    *,
    validator_id: str | None = None,
    policy: WalletPolicy | None = None,
) -> ChainBlock | None:
    if not transactions:
        return None

    active_chain_id = _money_policy_for_wallet_policy(policy).chain_network.value
    blocks = list_chain_blocks(policy)
    existing_tx_ids = {tx.tx_id for block in blocks for tx in block.transactions}
    existing_nonces = _chain_transaction_nonces(blocks)
    new_transactions = [tx for tx in transactions if tx.tx_id not in existing_tx_ids]
    if not new_transactions:
        return None
    new_nonces: set[tuple[str, str]] = set()
    for tx in new_transactions:
        if not tx.chain_id:
            tx.chain_id = active_chain_id
        if tx.chain_id != active_chain_id:
            raise ValueError(
                f"Refusing transaction for chain '{tx.chain_id}' on '{active_chain_id}'."
            )
        if int(tx.schema_version or 0) != CHAIN_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported transaction schema_version {tx.schema_version}."
            )
        signature_ok, signature_error = verify_chain_transaction_signature(
            tx,
            policy=policy,
        )
        if not signature_ok:
            raise ValueError(signature_error or "Invalid chain transaction signature.")
        if tx.nonce is not None:
            nonce_key = (normalize_address(tx.address), str(tx.nonce))
            if nonce_key in existing_nonces or nonce_key in new_nonces:
                raise ValueError(
                    f"Refusing duplicate nonce '{tx.nonce}' for address '{tx.address}'."
                )
            new_nonces.add(nonce_key)

    previous_hash = blocks[-1].block_hash if blocks else GENESIS_PREVIOUS_HASH
    height = (blocks[-1].height + 1) if blocks else 0
    block_id = hashlib.sha256(
        _canonical(
            {
                "height": height,
                "chain_id": active_chain_id,
                "schema_version": CHAIN_SCHEMA_VERSION,
                "previous_hash": previous_hash,
                "tx_ids": [tx.tx_id for tx in new_transactions],
                "nonce": secrets.token_hex(8),
            }
        ).encode("utf-8")
    ).hexdigest()[:32]
    block = ChainBlock(
        block_id=block_id,
        height=height,
        created_at=_now_iso(),
        previous_hash=previous_hash,
        validator_id=validator_id,
        transactions=new_transactions,
        block_hash="",
        chain_id=active_chain_id,
        schema_version=CHAIN_SCHEMA_VERSION,
        tx_root=compute_transaction_root(new_transactions),
    )
    block.state_root = compute_chain_state_root([*blocks, block])
    block.block_hash = compute_block_hash(block)
    blocks.append(block)
    save_chain_blocks(blocks, policy)
    return block


def make_genesis_block(money_policy: MoneyPolicy | None = None) -> ChainBlock:
    active_money_policy = money_policy or MoneyPolicy()
    transactions = genesis_chain_transactions(active_money_policy)
    block_id = hashlib.sha256(
        _canonical(
            {
                "kind": "genesis",
                "network": active_money_policy.chain_network.value,
                "height": 0,
                "created_at": GENESIS_CREATED_AT,
                "previous_hash": GENESIS_PREVIOUS_HASH,
                "validator_id": "genesis",
                "tx_ids": [tx.tx_id for tx in transactions],
            }
        ).encode("utf-8")
    ).hexdigest()[:32]
    block = ChainBlock(
        block_id=block_id,
        height=0,
        created_at=GENESIS_CREATED_AT,
        previous_hash=GENESIS_PREVIOUS_HASH,
        validator_id="genesis",
        transactions=transactions,
        block_hash="",
        chain_id=active_money_policy.chain_network.value,
        schema_version=CHAIN_SCHEMA_VERSION,
        tx_root=compute_transaction_root(transactions),
    )
    block.state_root = compute_chain_state_root([block])
    block.block_hash = compute_block_hash(block)
    return block


def expected_genesis_hash(
    *,
    policy: WalletPolicy | None = None,
    money_policy: MoneyPolicy | None = None,
) -> str:
    return make_genesis_block(
        _money_policy_for_wallet_policy(policy, money_policy)
    ).block_hash


def record_chain_transaction(
    transaction: ChainTransaction,
    *,
    validator_id: str | None = None,
    policy: WalletPolicy | None = None,
) -> bool:
    if not str(transaction.tx_type).startswith("genesis_"):
        ensure_chain_genesis(policy=policy)
    if transaction.tx_id in chain_transaction_ids(policy):
        return False
    return append_chain_block([transaction], validator_id=validator_id, policy=policy) is not None


def compute_reserve_chain_address(money_policy: MoneyPolicy | None = None) -> str:
    active_money_policy = money_policy or MoneyPolicy()
    return normalize_address(
        f"{COMPUTE_RESERVE_ADDRESS_PREFIX}:{active_money_policy.chain_network.value}"
    )


def developer_contribution_fund_chain_address(
    money_policy: MoneyPolicy | None = None,
) -> str:
    active_money_policy = money_policy or MoneyPolicy()
    return normalize_address(
        f"{DEVELOPER_CONTRIBUTION_FUND_ADDRESS_PREFIX}:"
        f"{active_money_policy.chain_network.value}"
    )


def developer_fund_distribution_round_ids(
    policy: WalletPolicy | None = None,
) -> set[str]:
    round_ids: set[str] = set()
    for block in list_chain_blocks(policy):
        for tx in block.transactions:
            if not str(tx.tx_type).startswith("developer_fund_distribution_"):
                continue
            round_id = str((tx.metadata or {}).get("round_id") or tx.payout_id or "")
            if round_id:
                round_ids.add(round_id)
    return round_ids


def record_developer_fund_distribution(
    *,
    round_id: str,
    recipients: list[dict[str, Any]],
    round_hash: str,
    participants_hash: str,
    source_commit: str | None = None,
    validator_id: str | None = None,
    policy: WalletPolicy | None = None,
    money_policy: MoneyPolicy | None = None,
) -> ChainBlock:
    active_money_policy = _money_policy_for_wallet_policy(policy, money_policy)
    normalized_round_id = str(round_id).strip()
    if not normalized_round_id:
        raise ValueError("Developer fund round_id is required.")
    if not str(round_hash or "").strip():
        raise ValueError("Developer fund round_hash is required.")
    if not str(participants_hash or "").strip():
        raise ValueError("Developer fund participants_hash is required.")
    ensure_chain_genesis(policy=policy, money_policy=active_money_policy)
    if normalized_round_id in developer_fund_distribution_round_ids(policy):
        raise ValueError(
            f"Developer fund round '{normalized_round_id}' was already distributed."
        )

    fund_address = developer_contribution_fund_chain_address(active_money_policy)
    prepared_recipients: list[dict[str, Any]] = []
    seen_addresses: set[str] = set()
    total_atomic = 0
    for index, recipient in enumerate(recipients):
        github = str(recipient.get("github") or "").strip().lstrip("@").lower()
        address = normalize_address(str(recipient.get("address") or ""))
        amount_atomic = int(recipient.get("amount_atomic") or 0)
        if not github:
            raise ValueError(f"Developer fund recipient[{index}] is missing github.")
        if not address:
            raise ValueError(f"Developer fund recipient '{github}' is missing address.")
        if amount_atomic <= 0:
            raise ValueError(
                f"Developer fund recipient '{github}' amount must be positive."
            )
        if address in seen_addresses:
            raise ValueError(
                f"Developer fund recipient address '{address}' is duplicated."
            )
        seen_addresses.add(address)
        prepared_recipients.append(
            {
                "github": github,
                "address": address,
                "amount_atomic": amount_atomic,
                "amount_coins": str(recipient.get("amount_coins") or ""),
                "category": str(recipient.get("category") or "general").strip()
                or "general",
                "reason": str(recipient.get("reason") or "").strip(),
            }
        )
        total_atomic += amount_atomic

    if not prepared_recipients:
        raise ValueError("Developer fund distribution requires at least one recipient.")
    fund_balance_atomic = chain_balance_atomic(fund_address, policy)
    if total_atomic > fund_balance_atomic:
        raise ValueError("Developer fund distribution exceeds available fund balance.")

    base_metadata: dict[str, Any] = {
        "round_id": normalized_round_id,
        "round_hash": str(round_hash),
        "participants_hash": str(participants_hash),
        "network": active_money_policy.chain_network.value,
        "reward_token_code": active_money_policy.reward_token_code,
    }
    if source_commit:
        base_metadata["source_commit"] = str(source_commit)

    debit = make_chain_transaction(
        tx_type="developer_fund_distribution_debit",
        address=fund_address,
        delta_atomic=-total_atomic,
        payout_id=normalized_round_id,
        wallet_id=f"system-developer-contribution-fund-{active_money_policy.chain_network.value}",
        note="Developer contribution fund distribution debit.",
        nonce=f"developer-fund:{normalized_round_id}:debit",
        metadata={
            **base_metadata,
            "recipient_count": len(prepared_recipients),
            "total_amount_atomic": total_atomic,
        },
        chain_id=active_money_policy.chain_network.value,
    )
    credit_transactions = [
        make_chain_transaction(
            tx_type="developer_fund_distribution_credit",
            address=recipient["address"],
            delta_atomic=recipient["amount_atomic"],
            payout_id=f"{normalized_round_id}:{recipient['github']}",
            counterparty_address=fund_address,
            note="Developer contribution fund distribution credit.",
            nonce=f"developer-fund:{normalized_round_id}:credit:{recipient['github']}",
            metadata={
                **base_metadata,
                "github": recipient["github"],
                "amount_coins": recipient["amount_coins"],
                "category": recipient["category"],
                "reason": recipient["reason"],
            },
            chain_id=active_money_policy.chain_network.value,
        )
        for recipient in prepared_recipients
    ]
    block = append_chain_block(
        [debit, *credit_transactions],
        validator_id=validator_id,
        policy=policy,
    )
    if block is None:
        raise ValueError("Developer fund distribution was not recorded.")
    return block


def tx_fee_pool_chain_address(money_policy: MoneyPolicy | None = None) -> str:
    active_money_policy = money_policy or MoneyPolicy()
    return normalize_address(
        f"{TX_FEE_POOL_ADDRESS_PREFIX}:{active_money_policy.chain_network.value}"
    )


def validator_settlement_fee_pool_chain_address(
    money_policy: MoneyPolicy | None = None,
) -> str:
    active_money_policy = money_policy or MoneyPolicy()
    return normalize_address(
        f"{VALIDATOR_SETTLEMENT_FEE_POOL_ADDRESS_PREFIX}:"
        f"{active_money_policy.chain_network.value}"
    )


def validator_bond_pool_chain_address(money_policy: MoneyPolicy | None = None) -> str:
    active_money_policy = money_policy or MoneyPolicy()
    return normalize_address(
        f"{VALIDATOR_BOND_POOL_ADDRESS_PREFIX}:{active_money_policy.chain_network.value}"
    )


def validator_slash_pool_chain_address(money_policy: MoneyPolicy | None = None) -> str:
    active_money_policy = money_policy or MoneyPolicy()
    return normalize_address(
        f"{VALIDATOR_SLASH_POOL_ADDRESS_PREFIX}:{active_money_policy.chain_network.value}"
    )


def genesis_chain_transactions(
    money_policy: MoneyPolicy | None = None,
) -> list[ChainTransaction]:
    active_money_policy = money_policy or MoneyPolicy()
    reserve_atomic = coins_to_atomic(
        str(active_money_policy.compute_reserve_coins),
        active_money_policy,
    )
    developer_treasury_atomic = coins_to_atomic(
        str(active_money_policy.developer_treasury_coins),
        active_money_policy,
    )
    developer_contribution_fund_atomic = coins_to_atomic(
        str(active_money_policy.developer_contribution_fund_coins),
        active_money_policy,
    )
    return [
        make_chain_transaction(
            tx_type="genesis_compute_reserve_credit",
            address=compute_reserve_chain_address(active_money_policy),
            delta_atomic=reserve_atomic,
            wallet_id=f"system-compute-reserve-{active_money_policy.chain_network.value}",
            note=(
                "Genesis allocation for the CAI compute reserve. This is the "
                "chain source of reserve-funded jobs."
            ),
            metadata={
                "network": active_money_policy.chain_network.value,
                "reward_token_code": active_money_policy.reward_token_code,
            },
            created_at=GENESIS_CREATED_AT,
            chain_id=active_money_policy.chain_network.value,
        ),
        make_chain_transaction(
            tx_type="genesis_developer_contribution_fund_credit",
            address=developer_contribution_fund_chain_address(active_money_policy),
            delta_atomic=developer_contribution_fund_atomic,
            wallet_id=(
                "system-developer-contribution-fund-"
                f"{active_money_policy.chain_network.value}"
            ),
            note=(
                "Genesis allocation for the developer contribution fund. "
                "Distribution is governed by the public contribution rules."
            ),
            metadata={
                "network": active_money_policy.chain_network.value,
                "reward_token_code": active_money_policy.reward_token_code,
            },
            created_at=GENESIS_CREATED_AT,
            chain_id=active_money_policy.chain_network.value,
        ),
        make_chain_transaction(
            tx_type="genesis_developer_treasury_credit",
            address=active_money_policy.developer_treasury_address,
            delta_atomic=developer_treasury_atomic,
            wallet_id=active_money_policy.developer_treasury_wallet_id,
            note="Genesis allocation for the fixed founder treasury wallet.",
            metadata={
                "network": active_money_policy.chain_network.value,
                "reward_token_code": active_money_policy.reward_token_code,
            },
            created_at=GENESIS_CREATED_AT,
            chain_id=active_money_policy.chain_network.value,
        ),
    ]


def ensure_chain_genesis(
    *,
    policy: WalletPolicy | None = None,
    money_policy: MoneyPolicy | None = None,
) -> int:
    active_money_policy = _money_policy_for_wallet_policy(policy, money_policy)
    genesis_block = make_genesis_block(active_money_policy)
    genesis_tx_ids = {tx.tx_id for tx in genesis_block.transactions}
    existing_blocks = list_chain_blocks(policy)
    if not existing_blocks:
        save_chain_blocks([genesis_block], policy)
        return len(genesis_block.transactions)

    if len(existing_blocks) == 1:
        existing_block = existing_blocks[0]
        existing_block_tx_ids = {tx.tx_id for tx in existing_block.transactions}
        if _is_stale_genesis_only_chain(existing_blocks, genesis_block):
            save_chain_blocks([genesis_block], policy)
            return 0
        if (
            int(existing_block.height) == 0
            and existing_block.previous_hash == GENESIS_PREVIOUS_HASH
            and existing_block.validator_id == "genesis"
            and existing_block_tx_ids == genesis_tx_ids
        ):
            if existing_block.block_hash != genesis_block.block_hash:
                save_chain_blocks([genesis_block], policy)
            return 0

    existing_tx_ids = chain_transaction_ids(policy)
    missing_transactions = [
        tx
        for tx in genesis_block.transactions
        if tx.tx_id not in existing_tx_ids
    ]
    block = append_chain_block(
        missing_transactions,
        validator_id="genesis",
        policy=policy,
    )
    return len(block.transactions) if block is not None else 0


def export_chain_payload(policy: WalletPolicy | None = None) -> dict[str, Any]:
    active_money_policy = _money_policy_for_wallet_policy(policy)
    ensure_chain_genesis(policy=policy, money_policy=active_money_policy)
    blocks = list_chain_blocks(policy)
    transactions = [tx for block in blocks for tx in block.transactions]
    genesis_hash = (
        blocks[0].block_hash
        if blocks
        else expected_genesis_hash(money_policy=active_money_policy)
    )
    return {
        "exported_at": _now_iso(),
        "network": active_money_policy.chain_network.value,
        "chain_id": active_money_policy.chain_network.value,
        "genesis_hash": genesis_hash,
        "schema_version": CHAIN_SCHEMA_VERSION,
        "chain": {
            "blocks": [asdict(item) for item in blocks],
            "block_count": len(blocks),
            "transaction_count": len(transactions),
            "tip_hash": blocks[-1].block_hash if blocks else None,
            "tip_tx_root": blocks[-1].tx_root if blocks else None,
            "tip_state_root": blocks[-1].state_root if blocks else None,
            "network": active_money_policy.chain_network.value,
            "chain_id": active_money_policy.chain_network.value,
            "genesis_hash": genesis_hash,
            "schema_version": CHAIN_SCHEMA_VERSION,
        },
    }


def merge_remote_chain_payload(
    payload: dict[str, Any],
    *,
    policy: WalletPolicy | None = None,
) -> tuple[int, int]:
    if isinstance(payload, dict):
        validate_peer_payload_network(
            payload,
            policy=policy,
            payload_name="chain",
        )
        signature_ok, signature_error = verify_peer_payload_signature(
            payload,
            payload_name="chain",
            require_signature=peer_payload_signatures_required(),
            require_hybrid_signature=peer_payload_hybrid_signatures_required(
                policy=policy
            ),
        )
        if not signature_ok:
            raise ValueError(signature_error or "Invalid chain payload signature.")
    raw_chain = payload.get("chain") if isinstance(payload.get("chain"), dict) else payload
    if not isinstance(raw_chain, dict):
        return 0, 0
    payload_chain_id = str(raw_chain.get("chain_id") or "").strip()
    payload_network_name = str(raw_chain.get("network") or "").strip()
    if payload_chain_id and payload_network_name and payload_chain_id != payload_network_name:
        raise ValueError(
            f"Refusing chain payload with mismatched chain_id '{payload_chain_id}' "
            f"and network '{payload_network_name}'."
        )
    payload_network = payload_chain_id or payload_network_name
    local_network = _money_policy_for_wallet_policy(policy).chain_network.value
    if payload_network and payload_network != local_network:
        raise ValueError(
            f"Refusing chain payload for network '{payload_network}' on '{local_network}'."
        )
    raw_blocks = list(raw_chain.get("blocks") or []) if isinstance(raw_chain, dict) else []
    if not raw_blocks:
        return 0, 0

    local_blocks = list_chain_blocks(policy)
    remote_blocks = _coerce_valid_remote_chain(raw_blocks, local_network, policy=policy)
    fork_imported = _adopt_authoritative_remote_fork(
        remote_blocks,
        local_blocks=local_blocks,
        signed_by_bonded_validator=_chain_payload_signed_by_bonded_validator(
            payload,
            policy=policy,
        ),
        policy=policy,
    )
    if fork_imported is not None:
        return fork_imported

    known_block_heights_by_hash = {
        block.block_hash: int(block.height)
        for block in local_blocks
        if str(block.block_hash).strip()
    }
    known_block_heights = {int(block.height) for block in local_blocks}
    existing_block_hashes = set(known_block_heights_by_hash)
    existing_tx_ids = {tx.tx_id for block in local_blocks for tx in block.transactions}
    existing_nonces = _chain_transaction_nonces(local_blocks)
    imported_blocks = 0
    imported_transactions = 0
    for raw in sorted(raw_blocks, key=_raw_block_sort_key):
        try:
            if not str(raw.get("block_hash") or "").strip():
                continue
            block = _coerce_block(raw, default_chain_id=local_network)
        except Exception as exc:
            _log_best_effort_failure("remote chain block import coercion", exc)
            continue
        if block.chain_id != local_network:
            continue
        if int(block.schema_version or 0) != CHAIN_SCHEMA_VERSION:
            continue
        if compute_block_hash(block) != block.block_hash:
            continue
        if block.block_hash in existing_block_hashes:
            continue
        if not _block_transactions_are_valid(block, policy=policy):
            continue
        if not _block_links_to_known_chain(
            block,
            known_block_heights_by_hash,
            known_block_heights,
        ):
            continue
        if not _block_roots_match_known_chain(block, local_blocks):
            continue
        if not _block_nonces_are_new(block, existing_nonces):
            continue
        new_transactions = [
            tx for tx in block.transactions if tx.tx_id not in existing_tx_ids
        ]
        if not new_transactions:
            continue
        local_blocks.append(block)
        existing_block_hashes.add(block.block_hash)
        known_block_heights_by_hash[block.block_hash] = int(block.height)
        known_block_heights.add(int(block.height))
        imported_blocks += 1
        for tx in new_transactions:
            existing_tx_ids.add(tx.tx_id)
            if tx.nonce is not None:
                existing_nonces.add((normalize_address(tx.address), str(tx.nonce)))
            imported_transactions += 1

    if imported_blocks:
        local_blocks.sort(key=lambda item: (int(item.height), item.created_at, item.block_hash))
        save_chain_blocks(local_blocks, policy)
    return imported_blocks, imported_transactions


def _is_stale_genesis_only_chain(
    local_blocks: list[ChainBlock],
    expected_genesis_block: ChainBlock,
) -> bool:
    if len(local_blocks) != 1:
        return False
    local_genesis = local_blocks[0]
    if (
        int(local_genesis.height) != 0
        or local_genesis.previous_hash != GENESIS_PREVIOUS_HASH
        or local_genesis.validator_id != "genesis"
        or local_genesis.block_hash == expected_genesis_block.block_hash
    ):
        return False
    return all(
        str(tx.tx_type or "").startswith("genesis_")
        for tx in local_genesis.transactions
    )


def _coerce_valid_remote_chain(
    raw_blocks: list[Any],
    local_network: str,
    *,
    policy: WalletPolicy | None = None,
) -> list[ChainBlock]:
    remote_blocks: list[ChainBlock] = []
    for raw in sorted(raw_blocks, key=_raw_block_sort_key):
        try:
            if not isinstance(raw, dict) or not str(raw.get("block_hash") or "").strip():
                return []
            block = _coerce_block(raw, default_chain_id=local_network)
        except Exception as exc:
            _log_best_effort_failure("remote chain fork validation block coercion", exc)
            return []
        if block.chain_id != local_network:
            return []
        if int(block.schema_version or 0) != CHAIN_SCHEMA_VERSION:
            return []
        if compute_block_hash(block) != block.block_hash:
            return []
        if not _block_transactions_are_valid(block, policy=policy):
            return []
        remote_blocks.append(block)
    if validate_chain_blocks(remote_blocks, policy=policy):
        return []
    return remote_blocks


def _adopt_authoritative_remote_fork(
    remote_blocks: list[ChainBlock],
    *,
    local_blocks: list[ChainBlock],
    signed_by_bonded_validator: bool,
    policy: WalletPolicy | None = None,
) -> tuple[int, int] | None:
    if not remote_blocks or not local_blocks:
        return None
    expected_genesis_block = make_genesis_block(_money_policy_for_wallet_policy(policy))
    if (
        remote_blocks[0].block_hash == expected_genesis_block.block_hash
        and _is_stale_genesis_only_chain(local_blocks, expected_genesis_block)
    ):
        imported_blocks = len(remote_blocks)
        imported_transactions = sum(len(block.transactions) for block in remote_blocks)
        save_chain_blocks(remote_blocks, policy)
        return imported_blocks, imported_transactions
    if not signed_by_bonded_validator:
        return None
    if remote_blocks[0].block_hash != local_blocks[0].block_hash:
        return None
    local_tip_hash = str(local_blocks[-1].block_hash or "").strip()
    remote_hashes = {str(block.block_hash) for block in remote_blocks}
    if local_tip_hash in remote_hashes:
        return None

    old_block_hashes = {str(block.block_hash) for block in local_blocks}
    old_tx_ids = {tx.tx_id for block in local_blocks for tx in block.transactions}
    imported_blocks = sum(
        1 for block in remote_blocks if str(block.block_hash) not in old_block_hashes
    )
    imported_transactions = sum(
        1
        for block in remote_blocks
        for tx in block.transactions
        if tx.tx_id not in old_tx_ids
    )
    save_chain_blocks(remote_blocks, policy)
    return imported_blocks, imported_transactions


def _chain_payload_signed_by_bonded_validator(
    payload: dict[str, Any],
    *,
    policy: WalletPolicy | None = None,
) -> bool:
    if not isinstance(payload, dict):
        return False
    signature = payload.get("signature")
    if not isinstance(signature, dict):
        return False
    signer_addresses = {
        str(signature.get("signer_address") or "").strip().lower(),
        str(signature.get("public_key_address") or "").strip().lower(),
    }
    signer_addresses.discard("")
    if not signer_addresses:
        return False

    for record in list_bonded_validators(policy):
        if record.state != ValidatorLifecycleState.BONDED:
            continue
        record_addresses = {
            str(record.validator_id or "").strip().lower(),
            str(record.address or "").strip().lower(),
        }
        record_addresses.discard("")
        if signer_addresses & record_addresses:
            return True
    return False


def _raw_block_sort_key(raw: Any) -> tuple[int, str, str]:
    if not isinstance(raw, dict):
        return (10**12, "", "")
    try:
        height = int(raw.get("height"))
    except (TypeError, ValueError):
        height = 10**12
    return (
        height,
        str(raw.get("created_at") or ""),
        str(raw.get("block_hash") or ""),
    )


def _block_transactions_are_valid(
    block: ChainBlock,
    *,
    policy: WalletPolicy | None = None,
) -> bool:
    if not str(block.chain_id).strip():
        return False
    if int(block.schema_version or 0) != CHAIN_SCHEMA_VERSION:
        return False
    tx_ids = [str(tx.tx_id).strip() for tx in block.transactions]
    if not tx_ids or any(not tx_id for tx_id in tx_ids):
        return False
    seen_nonces: set[tuple[str, str]] = set()
    for tx in block.transactions:
        if tx.chain_id != block.chain_id:
            return False
        if int(tx.schema_version or 0) != int(block.schema_version):
            return False
        signature_ok, _signature_error = verify_chain_transaction_signature(
            tx,
            policy=policy,
        )
        if not signature_ok:
            return False
        if tx.nonce is not None:
            nonce_key = (normalize_address(tx.address), str(tx.nonce))
            if nonce_key in seen_nonces:
                return False
            seen_nonces.add(nonce_key)
    if block.tx_root and block.tx_root != compute_transaction_root(block.transactions):
        return False
    return len(tx_ids) == len(set(tx_ids))


def _block_links_to_known_chain(
    block: ChainBlock,
    known_block_heights_by_hash: dict[str, int],
    known_block_heights: set[int],
) -> bool:
    height = int(block.height)
    if height == 0:
        return (
            block.previous_hash == GENESIS_PREVIOUS_HASH
            and 0 not in known_block_heights
        )
    previous_height = known_block_heights_by_hash.get(block.previous_hash)
    return (
        previous_height is not None
        and height == previous_height + 1
        and height not in known_block_heights
    )


def _block_roots_match_known_chain(
    block: ChainBlock,
    previous_blocks: list[ChainBlock],
) -> bool:
    expected_tx_root = compute_transaction_root(block.transactions)
    if block.tx_root and block.tx_root != expected_tx_root:
        return False
    block.tx_root = expected_tx_root
    expected_state_root = compute_chain_state_root([*previous_blocks, block])
    if block.state_root and block.state_root != expected_state_root:
        return False
    block.state_root = expected_state_root
    return True


def _block_nonces_are_new(
    block: ChainBlock,
    existing_nonces: set[tuple[str, str]],
) -> bool:
    for tx in block.transactions:
        if tx.nonce is None:
            continue
        if (normalize_address(tx.address), str(tx.nonce)) in existing_nonces:
            return False
    return True


def validate_chain_blocks(
    blocks: list[ChainBlock],
    *,
    policy: WalletPolicy | None = None,
) -> list[str]:
    if not blocks:
        return []

    errors: list[str] = []
    seen_block_hashes: set[str] = set()
    seen_tx_ids: set[str] = set()
    seen_nonces: set[tuple[str, str]] = set()
    prefix_blocks: list[ChainBlock] = []
    ordered_blocks = sorted(
        blocks,
        key=lambda item: (int(item.height), item.created_at, item.block_hash),
    )
    previous_block: ChainBlock | None = None
    expected_chain_id = str(ordered_blocks[0].chain_id or _default_chain_id())
    for index, block in enumerate(ordered_blocks):
        height = int(block.height)
        if not str(block.chain_id).strip():
            errors.append(f"block[{index}] has empty chain_id")
        elif str(block.chain_id) != expected_chain_id:
            errors.append(f"block[{index}] has mismatched chain_id")
        if int(block.schema_version or 0) != CHAIN_SCHEMA_VERSION:
            errors.append(f"block[{index}] has unsupported schema_version")
        if not str(block.block_hash).strip():
            errors.append(f"block[{index}] has empty block_hash")
            continue
        if block.block_hash in seen_block_hashes:
            errors.append(f"block[{index}] duplicates block_hash {block.block_hash}")
        seen_block_hashes.add(block.block_hash)
        if compute_block_hash(block) != block.block_hash:
            errors.append(f"block[{index}] has invalid block_hash")

        if index == 0:
            if height != 0:
                errors.append(f"block[{index}] must start at height 0")
            if block.previous_hash != GENESIS_PREVIOUS_HASH:
                errors.append(f"block[{index}] has invalid genesis previous_hash")
        elif previous_block is not None:
            expected_height = int(previous_block.height) + 1
            if height != expected_height:
                errors.append(
                    f"block[{index}] height {height} does not follow {expected_height - 1}"
                )
            if block.previous_hash != previous_block.block_hash:
                errors.append(f"block[{index}] previous_hash does not match parent")

        tx_ids = [str(tx.tx_id).strip() for tx in block.transactions]
        if not tx_ids:
            errors.append(f"block[{index}] has no transactions")
        if any(not tx_id for tx_id in tx_ids):
            errors.append(f"block[{index}] has empty transaction id")
        if len(tx_ids) != len(set(tx_ids)):
            errors.append(f"block[{index}] has duplicate transaction ids")
        expected_tx_root = compute_transaction_root(block.transactions)
        if not str(block.tx_root or "").strip():
            errors.append(f"block[{index}] has empty tx_root")
        elif block.tx_root != expected_tx_root:
            errors.append(f"block[{index}] has invalid tx_root")
        for tx in block.transactions:
            if not str(tx.chain_id).strip():
                errors.append(f"transaction {tx.tx_id} has empty chain_id")
            elif str(tx.chain_id) != str(block.chain_id):
                errors.append(f"transaction {tx.tx_id} chain_id does not match block")
            if int(tx.schema_version or 0) != CHAIN_SCHEMA_VERSION:
                errors.append(f"transaction {tx.tx_id} has unsupported schema_version")
            signature_ok, signature_error = verify_chain_transaction_signature(
                tx,
                policy=policy,
            )
            if not signature_ok:
                errors.append(signature_error or f"transaction {tx.tx_id} has invalid signature")
            if tx.nonce is not None:
                nonce_key = (normalize_address(tx.address), str(tx.nonce))
                if nonce_key in seen_nonces:
                    errors.append(
                        f"transaction {tx.tx_id} reuses nonce {tx.nonce} for address {tx.address}"
                    )
                seen_nonces.add(nonce_key)
        for tx_id in tx_ids:
            if tx_id in seen_tx_ids:
                errors.append(f"transaction {tx_id} appears in multiple blocks")
            seen_tx_ids.add(tx_id)
        expected_state_root = compute_chain_state_root([*prefix_blocks, block])
        if not str(block.state_root or "").strip():
            errors.append(f"block[{index}] has empty state_root")
        elif block.state_root != expected_state_root:
            errors.append(f"block[{index}] has invalid state_root")
        prefix_blocks.append(block)
        previous_block = block
    for address, balance_atomic in sorted(
        chain_balance_index(blocks=ordered_blocks).items()
    ):
        if balance_atomic < 0:
            errors.append(f"address {address} has negative balance")
    for validator_id, locked_atomic in sorted(
        validator_locked_bond_index(blocks=ordered_blocks).items()
    ):
        if locked_atomic < 0:
            errors.append(f"validator {validator_id} has negative locked bond")
    return errors


def chain_balance_index(
    policy: WalletPolicy | None = None,
    *,
    blocks: list[ChainBlock] | None = None,
) -> dict[str, int]:
    if blocks is None:
        index_payload = load_or_rebuild_chain_index(policy)
        if index_payload:
            return _chain_index_balances(index_payload)
    seen_tx_ids: set[str] = set()
    balances: dict[str, int] = {}
    for block in (blocks if blocks is not None else list_chain_blocks(policy)):
        for tx in block.transactions:
            if tx.tx_id in seen_tx_ids:
                continue
            seen_tx_ids.add(tx.tx_id)
            address = normalize_address(tx.address)
            balances[address] = balances.get(address, 0) + int(tx.delta_atomic)
    return balances


def _validator_id_from_bond_metadata(tx: ChainTransaction) -> str | None:
    metadata = tx.metadata or {}
    raw_validator_id = (
        metadata.get("validator_id")
        or metadata.get("validator_address")
        or metadata.get("validatorAddress")
    )
    if raw_validator_id is None:
        return None
    validator_id = normalize_address(str(raw_validator_id))
    return validator_id or None


def validator_locked_bond_index(
    policy: WalletPolicy | None = None,
    *,
    blocks: list[ChainBlock] | None = None,
) -> dict[str, int]:
    if blocks is None:
        index_payload = load_or_rebuild_chain_index(policy)
        if index_payload:
            return _chain_index_validator_locked_bonds(index_payload)
    seen_tx_ids: set[str] = set()
    locked_by_validator: dict[str, int] = {}
    for block in (blocks if blocks is not None else list_chain_blocks(policy)):
        for tx in block.transactions:
            if tx.tx_id in seen_tx_ids:
                continue
            seen_tx_ids.add(tx.tx_id)
            if tx.tx_type not in {"validator_bond_lock", "validator_bond_pool_debit"}:
                continue
            validator_id = _validator_id_from_bond_metadata(tx)
            if validator_id is None:
                continue
            if tx.tx_type == "validator_bond_lock":
                delta = max(0, -int(tx.delta_atomic))
            else:
                delta = int(tx.delta_atomic)
            if delta == 0:
                continue
            locked_by_validator[validator_id] = (
                locked_by_validator.get(validator_id, 0) + delta
            )
    return locked_by_validator


def chain_balance_atomic(address: str, policy: WalletPolicy | None = None) -> int:
    normalized_address = normalize_address(address)
    return chain_balance_index(policy).get(normalized_address, 0)


def _chain_transaction_history_entry(
    block: ChainBlock,
    tx: ChainTransaction,
    *,
    balance_after_atomic: int | None = None,
) -> dict[str, Any]:
    payload = asdict(tx)
    payload["block_height"] = int(block.height)
    payload["block_hash"] = block.block_hash
    payload["block_created_at"] = block.created_at
    if balance_after_atomic is not None:
        payload["balance_after_atomic"] = int(balance_after_atomic)
    return payload


def chain_address_history(
    address: str,
    policy: WalletPolicy | None = None,
    *,
    tx_types: set[str] | None = None,
    limit: int | None = None,
    newest_first: bool = True,
) -> list[dict[str, Any]]:
    normalized_address = normalize_address(address)
    allowed_types = set(tx_types) if tx_types is not None else None
    index_payload = load_or_rebuild_chain_index(policy)
    indexed_history = (
        index_payload.get("addressHistory", {}).get(normalized_address)
        if isinstance(index_payload.get("addressHistory"), dict)
        else None
    )
    if isinstance(indexed_history, list):
        history = [dict(item) for item in indexed_history if isinstance(item, dict)]
        if allowed_types is not None:
            history = [
                item
                for item in history
                if str(item.get("tx_type") or item.get("txType") or "") in allowed_types
            ]
        if newest_first:
            history.reverse()
        if limit is not None:
            return history[: max(0, int(limit))]
        return history
    seen_tx_ids: set[str] = set()
    balance_after_atomic = 0
    history: list[dict[str, Any]] = []
    for block in list_chain_blocks(policy):
        for tx in block.transactions:
            if tx.tx_id in seen_tx_ids:
                continue
            seen_tx_ids.add(tx.tx_id)
            if normalize_address(tx.address) != normalized_address:
                continue
            balance_after_atomic += int(tx.delta_atomic)
            if allowed_types is not None and tx.tx_type not in allowed_types:
                continue
            history.append(
                _chain_transaction_history_entry(
                    block,
                    tx,
                    balance_after_atomic=balance_after_atomic,
                )
            )
    if newest_first:
        history.reverse()
    if limit is not None:
        return history[: max(0, int(limit))]
    return history


def chain_settlement_history(
    settlement_id: str,
    policy: WalletPolicy | None = None,
    *,
    limit: int | None = None,
    newest_first: bool = True,
) -> list[dict[str, Any]]:
    normalized_settlement_id = str(settlement_id).strip()
    index_payload = load_or_rebuild_chain_index(policy)
    indexed_history = (
        index_payload.get("settlementHistory", {}).get(normalized_settlement_id)
        if isinstance(index_payload.get("settlementHistory"), dict)
        else None
    )
    if isinstance(indexed_history, list):
        history = [dict(item) for item in indexed_history if isinstance(item, dict)]
        if newest_first:
            history.reverse()
        if limit is not None:
            return history[: max(0, int(limit))]
        return history
    seen_tx_ids: set[str] = set()
    history: list[dict[str, Any]] = []
    for block in list_chain_blocks(policy):
        for tx in block.transactions:
            if tx.tx_id in seen_tx_ids:
                continue
            seen_tx_ids.add(tx.tx_id)
            if str(tx.settlement_id or "").strip() != normalized_settlement_id:
                continue
            history.append(_chain_transaction_history_entry(block, tx))
    if newest_first:
        history.reverse()
    if limit is not None:
        return history[: max(0, int(limit))]
    return history


def chain_is_initialized(policy: WalletPolicy | None = None) -> bool:
    return int(load_or_rebuild_chain_index(policy).get("blockCount") or 0) > 0


def wallet_balance_source(policy: WalletPolicy | None = None) -> str:
    return "chain" if chain_is_initialized(policy) else "local"


def has_chain_activity_for_address(address: str, policy: WalletPolicy | None = None) -> bool:
    normalized_address = normalize_address(address)
    return any(
        normalize_address(tx.address) == normalized_address
        for block in list_chain_blocks(policy)
        for tx in block.transactions
    )


def wallet_chain_balance_or_local_atomic(
    wallet,
    policy: WalletPolicy | None = None,
) -> int:
    if wallet is None:
        return 0
    index_payload = load_or_rebuild_chain_index(policy)
    if int(index_payload.get("blockCount") or 0) > 0:
        normalized_address = normalize_address(wallet.address)
        return _chain_index_balances(index_payload).get(normalized_address, 0)
    return int(getattr(wallet, "spendable_balance_atomic", 0) or 0)


def _peer_error_payload(peer_url: str, exc: Exception) -> dict[str, str]:
    return {
        "peerUrl": peer_url,
        "errorType": type(exc).__name__,
        "message": str(exc),
    }


def _log_best_effort_failure(operation: str, exc: Exception) -> None:
    LOGGER.warning(
        "Best-effort %s failed: %s",
        operation,
        exc,
        exc_info=LOGGER.isEnabledFor(logging.DEBUG),
    )


def sync_chain_from_cai_peers(
    *,
    state_payload: dict[str, Any],
    cai_url: str,
    policy: WalletPolicy | None = None,
    local_node_id: str | None = None,
    timeout_sec: float = 2.0,
    max_peers: int = 4,
) -> ChainSyncResult:
    discovered_peer_urls = discover_peer_cai_urls(
        state_payload=state_payload,
        cai_url=cai_url,
        endpoint_path="/v1/cai/chain",
        local_node_id=local_node_id,
    )
    peer_urls = discovered_peer_urls[: max(0, int(max_peers))]
    imported_blocks = 0
    imported_transactions = 0
    successful_peers = 0
    failed_peer_urls: list[str] = []
    peer_errors: list[dict[str, str]] = []
    for peer_url in peer_urls:
        try:
            with urlopen(peer_url, timeout=timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
            blocks, transactions = merge_remote_chain_payload(payload, policy=policy)
            imported_blocks += blocks
            imported_transactions += transactions
            successful_peers += 1
        except Exception as exc:
            failed_peer_urls.append(peer_url)
            peer_errors.append(_peer_error_payload(peer_url, exc))
            continue
    return ChainSyncResult(
        attempted_peers=len(peer_urls),
        successful_peers=successful_peers,
        imported_blocks=imported_blocks,
        imported_transactions=imported_transactions,
        peer_urls=peer_urls,
        failed_peers=len(failed_peer_urls),
        failed_peer_urls=failed_peer_urls,
        peer_errors=peer_errors,
    )


def push_chain_to_cai_peers(
    *,
    state_payload: dict[str, Any],
    cai_url: str,
    policy: WalletPolicy | None = None,
    local_node_id: str | None = None,
    timeout_sec: float = 1.0,
    max_peers: int = 4,
) -> ChainPushResult:
    discovered_peer_urls = discover_peer_cai_urls(
        state_payload=state_payload,
        cai_url=cai_url,
        endpoint_path="/v1/cai/chain/sync",
        local_node_id=local_node_id,
    )
    peer_urls = discovered_peer_urls[: max(0, int(max_peers))]
    payload = json.dumps(export_chain_payload(policy), ensure_ascii=False).encode("utf-8")
    successful_peers = 0
    failed_peer_urls: list[str] = []
    peer_errors: list[dict[str, str]] = []
    for peer_url in peer_urls:
        request = Request(
            url=peer_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_sec):
                pass
            successful_peers += 1
        except Exception as exc:
            failed_peer_urls.append(peer_url)
            peer_errors.append(_peer_error_payload(peer_url, exc))
            continue
    return ChainPushResult(
        attempted_peers=len(peer_urls),
        successful_peers=successful_peers,
        peer_urls=peer_urls,
        failed_peers=len(failed_peer_urls),
        failed_peer_urls=failed_peer_urls,
        peer_errors=peer_errors,
    )


def chain_summary(policy: WalletPolicy | None = None) -> dict[str, Any]:
    money_policy = _money_policy_for_wallet_policy(policy)
    blocks = list_chain_blocks(policy)
    validation_errors = validate_chain_blocks(blocks, policy=policy)
    index_payload = load_or_rebuild_chain_index(policy, blocks=blocks)
    balances = _chain_index_balances(index_payload) if index_payload else {}
    tip_block = blocks[-1] if blocks else None
    transaction_count = int(index_payload.get("transactionCount") or 0)
    reserve_address = compute_reserve_chain_address(money_policy)
    developer_contribution_fund_address = developer_contribution_fund_chain_address(
        money_policy
    )
    tx_fee_pool_address = tx_fee_pool_chain_address(money_policy)
    validator_settlement_fee_pool_address = validator_settlement_fee_pool_chain_address(
        money_policy
    )
    validator_bond_pool_address = validator_bond_pool_chain_address(money_policy)
    validator_slash_pool_address = validator_slash_pool_chain_address(money_policy)
    developer_address = normalize_address(money_policy.developer_treasury_address)
    ai_development_address = normalize_address(money_policy.ai_development_address)
    reserve_balance_atomic = balances.get(reserve_address, 0)
    developer_contribution_fund_balance_atomic = balances.get(
        developer_contribution_fund_address,
        0,
    )
    tx_fee_pool_balance_atomic = balances.get(tx_fee_pool_address, 0)
    validator_settlement_fee_pool_balance_atomic = balances.get(
        validator_settlement_fee_pool_address,
        0,
    )
    validator_bond_pool_balance_atomic = balances.get(validator_bond_pool_address, 0)
    validator_slash_pool_balance_atomic = balances.get(validator_slash_pool_address, 0)
    developer_balance_atomic = balances.get(developer_address, 0)
    ai_development_balance_atomic = balances.get(ai_development_address, 0)
    validator_bond_index = (
        _chain_index_validator_locked_bonds(index_payload)
        if index_payload
        else {}
    )
    non_zero_validator_bond_index = {
        validator_id: int(locked_atomic)
        for validator_id, locked_atomic in sorted(validator_bond_index.items())
        if int(locked_atomic) != 0
    }
    validator_bond_index_total_atomic = sum(
        int(locked_atomic) for locked_atomic in validator_bond_index.values()
    )
    total_balance_atomic = sum(int(balance) for balance in balances.values())
    expected_total_supply_atomic = coins_to_atomic(
        str(money_policy.total_supply_coins),
        money_policy,
    )
    supply_delta_atomic = total_balance_atomic - expected_total_supply_atomic
    non_zero_balance_count = sum(
        1 for balance in balances.values() if int(balance) != 0
    )
    return {
        "network": money_policy.chain_network.value,
        "chainId": money_policy.chain_network.value,
        "schemaVersion": CHAIN_SCHEMA_VERSION,
        "indexSchemaVersion": CHAIN_INDEX_SCHEMA_VERSION,
        "snapshotSchemaVersion": CHAIN_SNAPSHOT_SCHEMA_VERSION,
        "indexReady": bool(index_payload),
        "indexTipHeight": index_payload.get("tipHeight"),
        "indexTipHash": index_payload.get("tipHash"),
        "indexUpdatedAt": index_payload.get("indexedAt"),
        "snapshotInterval": index_payload.get("snapshotInterval"),
        "snapshotCount": index_payload.get("snapshotCount"),
        "latestSnapshotHeight": index_payload.get("latestSnapshotHeight"),
        "latestSnapshotHash": index_payload.get("latestSnapshotHash"),
        "latestSnapshotStateRoot": index_payload.get("latestSnapshotStateRoot"),
        "blockCount": len(blocks),
        "transactionCount": transaction_count,
        "tipHeight": tip_block.height if tip_block is not None else None,
        "tipHash": tip_block.block_hash if tip_block is not None else None,
        "tipCreatedAt": tip_block.created_at if tip_block is not None else None,
        "tipTxRoot": tip_block.tx_root if tip_block is not None else None,
        "tipStateRoot": tip_block.state_root if tip_block is not None else None,
        "finalizedHeight": tip_block.height if tip_block is not None and not validation_errors else None,
        "lastSyncAt": tip_block.created_at if tip_block is not None else None,
        "valid": not validation_errors,
        "validationErrors": validation_errors[:20],
        "balanceAddressCount": len(balances),
        "nonZeroBalanceAddressCount": non_zero_balance_count,
        "totalBalanceCoins": atomic_to_coins(total_balance_atomic, money_policy),
        "expectedTotalSupplyCoins": atomic_to_coins(
            expected_total_supply_atomic,
            money_policy,
        ),
        "supplyDeltaCoins": atomic_to_coins(supply_delta_atomic, money_policy),
        "supplyMatchesPolicy": supply_delta_atomic == 0,
        "computeReserveAddress": reserve_address,
        "computeReserveBalanceCoins": atomic_to_coins(reserve_balance_atomic, money_policy),
        "developerContributionFundAddress": developer_contribution_fund_address,
        "developerContributionFundBalanceCoins": atomic_to_coins(
            developer_contribution_fund_balance_atomic,
            money_policy,
        ),
        "txFeePoolAddress": tx_fee_pool_address,
        "txFeePoolBalanceCoins": atomic_to_coins(
            tx_fee_pool_balance_atomic,
            money_policy,
        ),
        "validatorSettlementFeePoolAddress": validator_settlement_fee_pool_address,
        "validatorSettlementFeePoolBalanceCoins": atomic_to_coins(
            validator_settlement_fee_pool_balance_atomic,
            money_policy,
        ),
        "validatorBondPoolAddress": validator_bond_pool_address,
        "validatorBondLockedCoins": atomic_to_coins(
            validator_bond_pool_balance_atomic,
            money_policy,
        ),
        "validatorBondLockedValidatorCount": len(non_zero_validator_bond_index),
        "validatorBondLockedByValidatorAtomic": non_zero_validator_bond_index,
        "validatorBondLockedByValidatorCoins": {
            validator_id: atomic_to_coins(locked_atomic, money_policy)
            for validator_id, locked_atomic in non_zero_validator_bond_index.items()
        },
        "validatorBondLockedIndexTotalCoins": atomic_to_coins(
            validator_bond_index_total_atomic,
            money_policy,
        ),
        "validatorBondLockedIndexMatchesPool": (
            validator_bond_index_total_atomic == validator_bond_pool_balance_atomic
        ),
        "validatorSlashPoolAddress": validator_slash_pool_address,
        "validatorSlashedCoins": atomic_to_coins(
            validator_slash_pool_balance_atomic,
            money_policy,
        ),
        "developerTreasuryAddress": developer_address,
        "developerTreasuryBalanceCoins": atomic_to_coins(
            developer_balance_atomic,
            money_policy,
        ),
        "founderTreasuryAddress": developer_address,
        "founderTreasuryBalanceCoins": atomic_to_coins(
            developer_balance_atomic,
            money_policy,
        ),
        "aiDevelopmentAddress": ai_development_address,
        "aiDevelopmentBalanceCoins": atomic_to_coins(
            ai_development_balance_atomic,
            money_policy,
        ),
    }

# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
import hashlib
import json
import secrets
import shutil
from pathlib import Path
from typing import Any

from .cai_owned_transport_common import (
    cai_owned_transport_chain_id as _cai_owned_transport_chain_id,
    clean_node_ids as _clean_node_ids,
    is_safe_transport_file_id as _is_safe_transport_file_id,
    parse_cai_owned_transport_datetime as _parse_cai_owned_transport_datetime,
    require_safe_transport_file_id as _require_safe_transport_file_id,
)
from .cai_owned_transport_protocol import (
    CAI_OWNED_TRANSPORT_PAYLOAD_RETENTION_SECONDS,
    CAI_OWNED_TRANSPORT_PROTOCOL,
    CAI_OWNED_TRANSPORT_PROTOCOL_VERSION,
    CAI_OWNED_TRANSPORT_REPLAY_CACHE_RETENTION_SECONDS,
    EXECUTION_MODE_CAI_OWNED_TRANSPORT_REQUIRED,
)
from .local_json_store import atomic_write_json_array_file, read_json_array_file
from .model import WalletPolicy
from .peer_payload import peer_payload_signing_body
from .wallet import data_root


@dataclass
class CaiOwnedTransportSessionRecord:
    session_id: str
    instance_id: str
    participant_node_ids: list[str]
    status: str
    created_at: str
    updated_at: str
    executor_node_ids: list[str] = field(default_factory=list)
    chain_id: str | None = None
    model_id: str | None = None
    task_id: str | None = None
    source_node_id: str | None = None
    execution_mode: str | None = None
    route_policy: dict[str, Any] = field(default_factory=dict)
    dispatch_records: list[dict[str, Any]] = field(default_factory=list)
    batch_records: list[dict[str, Any]] = field(default_factory=list)
    shard_receipts: list[dict[str, Any]] = field(default_factory=list)
    proof: dict[str, Any] | None = None
    completed_at: str | None = None
    last_error: str | None = None


def cai_owned_transport_sessions_file_path(
    policy: WalletPolicy | None = None,
) -> Path:
    return data_root(policy) / "cai-owned-transport-sessions.json"


def cai_owned_transport_replay_cache_file_path(
    policy: WalletPolicy | None = None,
) -> Path:
    return data_root(policy) / "cai-owned-transport-replay-cache.json"


def cai_owned_transport_batch_payload_path(
    session_id: str,
    batch_id: str,
    policy: WalletPolicy | None = None,
) -> Path:
    safe_session_id = _require_safe_transport_file_id(session_id, prefix="caiot_")
    safe_batch_id = _require_safe_transport_file_id(batch_id, prefix="caibatch_")
    return (
        cai_owned_transport_payload_storage_root(policy)
        / safe_session_id
        / f"{safe_batch_id}.bin"
    )


def cai_owned_transport_batch_output_payload_path(
    session_id: str,
    batch_id: str,
    policy: WalletPolicy | None = None,
) -> Path:
    safe_session_id = _require_safe_transport_file_id(session_id, prefix="caiot_")
    safe_batch_id = _require_safe_transport_file_id(batch_id, prefix="caibatch_")
    return (
        cai_owned_transport_payload_storage_root(policy)
        / safe_session_id
        / f"{safe_batch_id}.out.bin"
    )


def cai_owned_transport_payload_storage_root(
    policy: WalletPolicy | None = None,
) -> Path:
    return data_root(policy) / "cai-owned-transport-payloads"


def cleanup_cai_owned_transport_payload_storage(
    *,
    retention_seconds: float | int | None = None,
    now: datetime | None = None,
    policy: WalletPolicy | None = None,
) -> dict[str, Any]:
    root = cai_owned_transport_payload_storage_root(policy)
    reference_now = now or datetime.now(tz=UTC)
    retention = (
        CAI_OWNED_TRANSPORT_PAYLOAD_RETENTION_SECONDS
        if retention_seconds is None
        else max(0.0, float(retention_seconds))
    )
    result = {
        "status": "ok",
        "root": str(root),
        "retentionSeconds": retention,
        "deletedSessionIds": [],
        "skippedSessionIds": [],
        "skippedActiveSessionIds": [],
        "skippedUnknownSessionIds": [],
        "errorCount": 0,
        "errors": [],
    }
    if not root.exists():
        return result

    records_by_session_id = {
        record.session_id: record
        for record in list_cai_owned_transport_sessions(policy)
        if record.session_id
    }
    for child in root.iterdir():
        if not child.is_dir():
            continue
        session_id = child.name
        if not _is_safe_transport_file_id(session_id, prefix="caiot_"):
            result["skippedSessionIds"].append(session_id)
            continue
        record = records_by_session_id.get(session_id)
        if record is None:
            result["skippedUnknownSessionIds"].append(session_id)
            continue
        status = str(record.status or "").strip()
        if status not in {"completed", "failed"}:
            result["skippedActiveSessionIds"].append(session_id)
            continue
        reference_time = (
            _parse_cai_owned_transport_datetime(record.completed_at)
            or _parse_cai_owned_transport_datetime(record.updated_at)
            or _parse_cai_owned_transport_datetime(record.created_at)
        )
        if reference_time is None or reference_time + timedelta(
            seconds=retention
        ) > reference_now:
            result["skippedSessionIds"].append(session_id)
            continue
        try:
            _delete_cai_owned_transport_payload_session_dir(child, root)
            result["deletedSessionIds"].append(session_id)
        except Exception as exc:
            result["errorCount"] += 1
            result["errors"].append(f"{session_id}: {exc}")
    return result


def list_cai_owned_transport_sessions(
    policy: WalletPolicy | None = None,
) -> list[CaiOwnedTransportSessionRecord]:
    path = cai_owned_transport_sessions_file_path(policy)
    if not path.exists():
        return []
    raw = _read_json_array_file(path, heal_corrupt=True)
    records: list[CaiOwnedTransportSessionRecord] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        item.setdefault("model_id", item.get("modelId"))
        item.setdefault("task_id", item.get("taskId"))
        item.setdefault("source_node_id", item.get("sourceNodeId"))
        item.setdefault("execution_mode", item.get("executionMode"))
        item.setdefault("route_policy", item.get("routePolicy") or {})
        item.setdefault("dispatch_records", item.get("dispatchRecords") or [])
        item.setdefault("batch_records", item.get("batchRecords") or [])
        item.setdefault("shard_receipts", item.get("shardReceipts") or [])
        item.setdefault(
            "executor_node_ids",
            item.get("executorNodeIds")
            or item.get("executor_node_ids")
            or item.get("participantNodeIds")
            or item.get("participant_node_ids")
            or [],
        )
        item.setdefault("proof", None)
        item.setdefault("completed_at", item.get("completedAt"))
        item.setdefault("last_error", item.get("lastError"))
        item.setdefault(
            "chain_id",
            item.get("chainId")
            or item.get("network")
            or _cai_owned_transport_chain_id(policy),
        )
        records.append(CaiOwnedTransportSessionRecord(**item))
    records.sort(key=lambda item: (item.created_at, item.session_id), reverse=True)
    return records


def save_cai_owned_transport_sessions(
    records: list[CaiOwnedTransportSessionRecord],
    policy: WalletPolicy | None = None,
) -> None:
    path = cai_owned_transport_sessions_file_path(policy)
    _atomic_write_json_file(path, [asdict(item) for item in records])


def list_cai_owned_transport_replay_cache(
    policy: WalletPolicy | None = None,
) -> list[dict[str, Any]]:
    path = cai_owned_transport_replay_cache_file_path(policy)
    if not path.exists():
        return []
    raw = _read_json_array_file(path, heal_corrupt=True)
    return [dict(item) for item in raw if isinstance(item, dict)]


def save_cai_owned_transport_replay_cache(
    records: list[dict[str, Any]],
    policy: WalletPolicy | None = None,
) -> None:
    path = cai_owned_transport_replay_cache_file_path(policy)
    _atomic_write_json_file(path, records)


def cleanup_cai_owned_transport_replay_cache(
    *,
    retention_seconds: float | int | None = None,
    now: datetime | None = None,
    policy: WalletPolicy | None = None,
) -> int:
    records = list_cai_owned_transport_replay_cache(policy)
    if not records:
        return 0
    reference_now = (now or datetime.now(tz=UTC)).astimezone(UTC)
    retention = (
        CAI_OWNED_TRANSPORT_REPLAY_CACHE_RETENTION_SECONDS
        if retention_seconds is None
        else max(0.0, float(retention_seconds))
    )
    kept = [
        record
        for record in records
        if not _cai_owned_transport_replay_record_expired(
            record,
            reference_now=reference_now,
            retention_seconds=retention,
        )
    ]
    pruned = len(records) - len(kept)
    if pruned:
        save_cai_owned_transport_replay_cache(kept, policy)
    return pruned


def cai_owned_transport_payload_replay_key(payload: dict[str, Any]) -> str | None:
    if not isinstance(payload, dict):
        return None
    signature = payload.get("signature")
    if not isinstance(signature, dict):
        return None
    signature_b64 = str(signature.get("signature_b64") or "").strip()
    public_key_b64 = str(signature.get("public_key_b64") or "").strip()
    if not signature_b64 or not public_key_b64:
        return None
    replay_payload = {
        "body": peer_payload_signing_body(payload),
        "signature": {
            "scheme": signature.get("scheme"),
            "public_key_b64": public_key_b64,
            "signature_b64": signature_b64,
        },
    }
    encoded = json.dumps(
        replay_payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def record_cai_owned_transport_payload_replay(
    payload: dict[str, Any],
    *,
    payload_name: str,
    policy: WalletPolicy | None = None,
    retention_seconds: float | int | None = None,
    now: datetime | None = None,
) -> tuple[bool, str | None]:
    replay_key = cai_owned_transport_payload_replay_key(payload)
    if not replay_key:
        return False, f"{payload_name} payload signature is required for replay cache"
    reference_now = (now or datetime.now(tz=UTC)).astimezone(UTC)
    retention = (
        CAI_OWNED_TRANSPORT_REPLAY_CACHE_RETENTION_SECONDS
        if retention_seconds is None
        else max(0.0, float(retention_seconds))
    )
    records = [
        record
        for record in list_cai_owned_transport_replay_cache(policy)
        if not _cai_owned_transport_replay_record_expired(
            record,
            reference_now=reference_now,
            retention_seconds=retention,
        )
    ]
    if any(str(record.get("replayKey") or "") == replay_key for record in records):
        return False, f"{payload_name} payload signature replay detected"

    signature = payload.get("signature") if isinstance(payload, dict) else {}
    signature_dict = signature if isinstance(signature, dict) else {}
    expires_at = reference_now + timedelta(seconds=retention)
    records.append(
        {
            "replayKey": replay_key,
            "payloadName": payload_name,
            "signerNodeId": str(payload.get("signerNodeId") or "").strip() or None,
            "publicKeyAddress": str(
                signature_dict.get("public_key_address") or ""
            ).strip().lower()
            or None,
            "sessionId": str(payload.get("sessionId") or "").strip() or None,
            "batchId": str(payload.get("batchId") or "").strip() or None,
            "createdAt": str(payload.get("createdAt") or "").strip() or None,
            "seenAt": reference_now.isoformat(),
            "expiresAt": expires_at.isoformat(),
        }
    )
    save_cai_owned_transport_replay_cache(records, policy)
    return True, None


def create_cai_owned_transport_session(
    *,
    instance_id: str,
    participant_node_ids: Sequence[str],
    executor_node_ids: Sequence[str] | None = None,
    session_id: str | None = None,
    chain_id: str | None = None,
    model_id: str | None = None,
    task_id: str | None = None,
    source_node_id: str | None = None,
    execution_mode: str | None = EXECUTION_MODE_CAI_OWNED_TRANSPORT_REQUIRED,
    route_policy: dict[str, Any] | None = None,
    policy: WalletPolicy | None = None,
) -> CaiOwnedTransportSessionRecord:
    clean_instance_id = str(instance_id or "").strip()
    if not clean_instance_id:
        raise ValueError("CAI-owned transport session requires an instance id.")
    participants = _clean_node_ids(participant_node_ids)
    if not participants:
        raise ValueError("CAI-owned transport session requires participant nodes.")
    executors = _clean_node_ids(executor_node_ids or participants)
    if not executors:
        raise ValueError("CAI-owned transport session requires executor nodes.")
    if not set(executors).issubset(set(participants)):
        raise ValueError(
            "CAI-owned transport session executors must be participants."
        )
    resolved_chain_id = _cai_owned_transport_chain_id(policy, chain_id)
    policy_chain_id = _cai_owned_transport_chain_id(policy)
    if chain_id is not None and resolved_chain_id != policy_chain_id:
        raise ValueError(
            f"CAI-owned transport session is for chain '{resolved_chain_id}', "
            f"expected '{policy_chain_id}'."
        )
    clean_session_id = str(session_id or "").strip()
    if not clean_session_id:
        clean_session_id = f"caiot_{secrets.token_hex(12)}"
    records = list_cai_owned_transport_sessions(policy)
    for record in records:
        if record.session_id != clean_session_id:
            continue
        if (
            record.instance_id != clean_instance_id
            or record.participant_node_ids != participants
            or (record.executor_node_ids or record.participant_node_ids) != executors
            or (record.chain_id and record.chain_id != resolved_chain_id)
        ):
            raise ValueError(
                "CAI-owned transport session id is already used by another "
                "instance, participant set, or chain id."
            )
        return record
    now = datetime.now(tz=UTC).isoformat()
    record = CaiOwnedTransportSessionRecord(
        session_id=clean_session_id,
        instance_id=clean_instance_id,
        chain_id=resolved_chain_id,
        executor_node_ids=executors,
        model_id=str(model_id or "").strip() or None,
        task_id=str(task_id or "").strip() or None,
        source_node_id=str(source_node_id or "").strip() or participants[0],
        participant_node_ids=participants,
        execution_mode=str(execution_mode or "").strip() or None,
        route_policy=dict(route_policy or {}),
        status="created",
        created_at=now,
        updated_at=now,
    )
    records.append(record)
    save_cai_owned_transport_sessions(records, policy)
    return record


def deterministic_cai_owned_transport_session_id(
    instance_id: str,
    participant_node_ids: Sequence[str],
    *,
    executor_node_ids: Sequence[str] | None = None,
    task_id: str | None = None,
    chain_id: str | None = None,
) -> str:
    clean_instance_id = str(instance_id or "").strip()
    if not clean_instance_id:
        raise ValueError(
            "CAI-owned transport deterministic session requires instance id."
        )
    participants = _clean_node_ids(participant_node_ids)
    if not participants:
        raise ValueError(
            "CAI-owned transport deterministic session requires participant nodes."
        )
    executors = _clean_node_ids(executor_node_ids or participants)
    if not executors or not set(executors).issubset(set(participants)):
        raise ValueError(
            "CAI-owned transport deterministic session requires executor participants."
        )
    payload = {
        "instanceId": clean_instance_id,
        "participantNodeIds": participants,
        "executorNodeIds": executors,
        "protocol": CAI_OWNED_TRANSPORT_PROTOCOL,
        "protocolVersion": CAI_OWNED_TRANSPORT_PROTOCOL_VERSION,
        "chainId": _cai_owned_transport_chain_id(None, chain_id),
        "taskId": str(task_id or "").strip() or None,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"caiot_{hashlib.sha256(encoded).hexdigest()[:24]}"


def _atomic_write_json_file(path: Path, payload: list[Any]) -> None:
    atomic_write_json_array_file(path, payload)


def _read_json_array_file(path: Path, *, heal_corrupt: bool = False) -> list[Any]:
    return read_json_array_file(path, heal_corrupt=heal_corrupt)


def _delete_cai_owned_transport_payload_session_dir(path: Path, root: Path) -> None:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if resolved_path == resolved_root:
        raise ValueError("Refusing to delete CAI-owned transport payload root.")
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            "Refusing to delete CAI-owned transport payload path outside root."
        ) from exc
    shutil.rmtree(resolved_path)


def _cai_owned_transport_replay_record_expired(
    record: Mapping[str, Any],
    *,
    reference_now: datetime,
    retention_seconds: float,
) -> bool:
    expires_at = _parse_cai_owned_transport_datetime(record.get("expiresAt"))
    if expires_at is not None:
        return expires_at <= reference_now
    seen_at = _parse_cai_owned_transport_datetime(record.get("seenAt"))
    if seen_at is None:
        return False
    return seen_at + timedelta(seconds=retention_seconds) <= reference_now

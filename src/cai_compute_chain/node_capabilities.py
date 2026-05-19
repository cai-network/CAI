# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
import re
import socket
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse, urlunparse
from urllib.request import urlopen

from .decentralized_compute import cai_owned_transport_runtime_readiness
from .model import NetworkModelPolicy, WalletPolicy, normalize_network_model_id
from .peer_payload import (
    add_peer_payload_metadata,
    peer_payload_signatures_required,
    validate_peer_payload_network,
    verify_peer_payload_signature,
)
from .transport_endpoints import (
    candidate_identity_http_urls,
    identity_transport_endpoints,
)
from .validators import discover_peer_cai_urls
from .wallet import data_root
from .wallet_signing import address_from_public_key_b64

OVERLAY_MULTIADDR_P2P_COMPONENT_RE = re.compile(r"/p2p/([^/]+)", re.IGNORECASE)


def _peer_error_payload(peer_url: str, exc: Exception) -> dict[str, str]:
    return {
        "peerUrl": peer_url,
        "errorType": type(exc).__name__,
        "message": str(exc),
    }


@dataclass(frozen=True)
class NodeCapabilitySyncResult:
    attempted_peers: int
    successful_peers: int
    imported_records: int
    pruned_records: int
    peer_urls: list[str]
    failed_peers: int = 0
    failed_peer_urls: list[str] = field(default_factory=list)
    peer_errors: list[dict[str, str]] = field(default_factory=list)
    convergence_status: str = "unknown"
    convergence_repair_recommended: bool = False
    convergence_repair_actions: list[str] = field(default_factory=list)
    convergence_audit: dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeCapabilityRecord:
    node_id: str
    source: str
    source_url: str | None
    last_seen_at: str
    updated_at: str
    friendly_name: str | None = None
    node_public_key_b64: str | None = None
    node_public_key_address: str | None = None
    api_urls: list[str] = field(default_factory=list)
    data_endpoints: list[dict[str, Any]] = field(default_factory=list)
    worker_enabled: bool | None = None
    relay_enabled: bool | None = None
    validator_enabled: bool | None = None
    validator_id: str | None = None
    validator_state: str | None = None
    worker_reward_address: str | None = None
    worker_allowed_model_ids: list[str] = field(default_factory=list)
    model_ids: list[str] = field(default_factory=list)
    resource_summary: dict[str, Any] = field(default_factory=dict)
    readiness: dict[str, Any] = field(default_factory=dict)
    route_hints: dict[str, Any] = field(default_factory=dict)
    payload_signature_valid: bool | None = None
    payload_signer_address: str | None = None
    payload_public_key_address: str | None = None
    worker_verified: bool = False
    worker_verification_reason: str | None = None


def node_capabilities_file_path(policy: WalletPolicy | None = None) -> Path:
    return data_root(policy) / "node-capabilities.json"


def list_node_capabilities(
    policy: WalletPolicy | None = None,
) -> list[NodeCapabilityRecord]:
    path = node_capabilities_file_path(policy)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    records: list[NodeCapabilityRecord] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        item.setdefault("friendly_name", None)
        item.setdefault("api_urls", [])
        item.setdefault("data_endpoints", [])
        item.setdefault("worker_enabled", None)
        item.setdefault("relay_enabled", None)
        item.setdefault("validator_enabled", None)
        item.setdefault("validator_id", None)
        item.setdefault("validator_state", None)
        item.setdefault("worker_reward_address", None)
        item.setdefault("node_public_key_b64", None)
        item.setdefault("node_public_key_address", None)
        item.setdefault("worker_allowed_model_ids", [])
        item.setdefault("model_ids", [])
        item.setdefault("resource_summary", {})
        item.setdefault("readiness", {})
        item.setdefault("route_hints", {})
        item.setdefault("payload_signature_valid", None)
        item.setdefault("payload_signer_address", None)
        item.setdefault("payload_public_key_address", None)
        item.setdefault("worker_verified", False)
        item.setdefault("worker_verification_reason", None)
        records.append(NodeCapabilityRecord(**item))
    records.sort(key=lambda item: (item.node_id, item.source_url or ""))
    return records


def save_node_capabilities(
    records: list[NodeCapabilityRecord],
    policy: WalletPolicy | None = None,
) -> None:
    path = node_capabilities_file_path(policy)
    path.write_text(
        json.dumps([asdict(item) for item in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def export_node_capabilities_payload(
    *,
    state_payload: dict[str, Any] | None,
    cai_url: str | None = None,
    local_node_id: str | None = None,
    policy: WalletPolicy | None = None,
) -> dict[str, Any]:
    now = _now_iso()
    records = _records_from_state_payload(
        state_payload=state_payload,
        cai_url=cai_url,
        local_node_id=local_node_id,
        observed_at=now,
    )
    if not records:
        records = [
            _minimal_local_record(
                cai_url=cai_url,
                local_node_id=local_node_id,
                observed_at=now,
            )
        ]
    active_policy = policy or WalletPolicy()
    return add_peer_payload_metadata(
        {
            "chain_network": active_policy.chain_network.value,
            "exported_at": now,
            "records": [asdict(item) for item in records],
        },
        policy=active_policy,
    )


def refresh_local_node_capabilities(
    *,
    state_payload: dict[str, Any] | None,
    cai_url: str | None = None,
    local_node_id: str | None = None,
    policy: WalletPolicy | None = None,
) -> list[NodeCapabilityRecord]:
    payload = export_node_capabilities_payload(
        state_payload=state_payload,
        cai_url=cai_url,
        local_node_id=local_node_id,
        policy=policy,
    )
    merge_remote_node_capabilities_payload(
        payload,
        source_url=cai_url or "local",
        policy=policy,
        source="local_state",
    )
    return list_node_capabilities(policy)


def merge_remote_node_capabilities_payload(
    payload: dict[str, Any],
    *,
    source_url: str,
    policy: WalletPolicy | None = None,
    source: str = "peer",
    only_node_id: str | None = None,
) -> int:
    validate_peer_payload_network(
        payload,
        policy=policy,
        payload_name="node capabilities",
    )
    remote_source = source != "local_state"
    require_signature = (
        peer_payload_signatures_required(policy=policy) if remote_source else False
    )
    signature_ok, signature_error = verify_peer_payload_signature(
        payload,
        payload_name="node capabilities",
        require_signature=require_signature,
    )
    if not signature_ok:
        raise ValueError(
            signature_error or "Invalid node capabilities payload signature."
        )
    signature_context = _payload_signature_context(payload, signature_ok=signature_ok)
    existing = {item.node_id: item for item in list_node_capabilities(policy)}
    imported = 0
    now = _now_iso()
    requested_node_id = str(only_node_id or "").strip()
    for raw in _iter_capability_records(payload):
        node_id = str(raw.get("node_id") or raw.get("nodeId") or "").strip()
        if not node_id:
            continue
        if requested_node_id and node_id != requested_node_id:
            continue
        record = _record_from_raw(
            raw,
            source=source,
            source_url=source_url,
            observed_at=now,
        )
        _apply_record_verification_metadata(
            record,
            signature_context=signature_context,
            local_source=not remote_source,
        )
        previous = existing.get(node_id)
        if previous is not None and not _record_is_newer(record, previous):
            continue
        existing[node_id] = record
        imported += 1
    if imported:
        save_node_capabilities(list(existing.values()), policy)
    return imported


def worker_capability_verification_required(value: str | None = None) -> bool:
    raw = (
        value
        if value is not None
        else os.getenv("CAI_REQUIRE_VERIFIED_WORKER_CAPABILITIES")
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


def worker_capability_validator_attestation_required(
    value: str | None = None,
) -> bool:
    raw = (
        value
        if value is not None
        else os.getenv("CAI_REQUIRE_VALIDATOR_WORKER_CAPABILITY_ATTESTATION")
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
    return worker_capability_verification_required()


def list_verified_worker_node_ids(
    policy: WalletPolicy | None = None,
    *,
    accepted_model_ids: set[str] | None = None,
    max_age_seconds: int | None = None,
    require_validator_attestation: bool | None = None,
) -> set[str]:
    accepted = {
        _normalize_capability_model_id(model_id)
        for model_id in (accepted_model_ids or set())
        if str(model_id or "").strip()
    }
    records = list_node_capabilities(policy)
    require_attestation = (
        worker_capability_validator_attestation_required()
        if require_validator_attestation is None
        else bool(require_validator_attestation)
    )
    validator_attested_node_ids: set[str] | None = None
    if require_attestation:
        from .worker_capability_attestations import (
            list_validator_attested_worker_node_ids,
        )

        validator_attested_node_ids = list_validator_attested_worker_node_ids(
            records=records,
            accepted_model_ids=accepted,
            max_age_seconds=max_age_seconds,
            policy=policy,
        )
    worker_node_ids: set[str] = set()
    for record in records:
        node_id = str(record.node_id or "").strip()
        if not node_id:
            continue
        if validator_attested_node_ids is not None:
            if node_id not in validator_attested_node_ids:
                continue
            if not bool(record.worker_enabled):
                continue
            if not str(record.worker_reward_address or "").strip():
                continue
        elif not bool(record.worker_verified):
            continue
        if max_age_seconds is not None and _record_is_stale(
            record,
            max_age_seconds=max_age_seconds,
        ):
            continue
        allowed_model_ids = set(record.worker_allowed_model_ids or [])
        if accepted and allowed_model_ids and not accepted.intersection(
            allowed_model_ids
        ):
            continue
        worker_node_ids.add(node_id)
    return worker_node_ids


def prune_stale_node_capabilities(
    *,
    max_age_seconds: int = 3600,
    policy: WalletPolicy | None = None,
) -> int:
    now = datetime.now(tz=UTC)
    kept: list[NodeCapabilityRecord] = []
    pruned = 0
    for item in list_node_capabilities(policy):
        parsed = _parse_iso_datetime(item.last_seen_at)
        if parsed is None:
            kept.append(item)
            continue
        if (now - parsed).total_seconds() > max(0, int(max_age_seconds)):
            pruned += 1
            continue
        kept.append(item)
    if pruned:
        save_node_capabilities(kept, policy)
    return pruned


def node_capability_convergence_audit(
    *,
    state_payload: dict[str, Any] | None,
    records: Sequence[NodeCapabilityRecord] | None = None,
    local_node_id: str | None = None,
    max_age_seconds: int = 3600,
    policy: WalletPolicy | None = None,
) -> dict[str, Any]:
    cached_records = list(records) if records is not None else list_node_capabilities(policy)
    state_node_ids = _state_node_identity_ids(state_payload)
    overlay_node_ids = _state_overlay_reference_node_ids(state_payload)
    fresh_record_ids: set[str] = set()
    stale_record_ids: set[str] = set()
    now = datetime.now(tz=UTC)
    max_age = max(0, int(max_age_seconds))
    for record in cached_records:
        node_id = str(record.node_id or "").strip()
        if not node_id:
            continue
        parsed = _parse_iso_datetime(record.last_seen_at)
        if parsed is not None and (now - parsed).total_seconds() > max_age:
            stale_record_ids.add(node_id)
            continue
        fresh_record_ids.add(node_id)
    expected_node_ids = state_node_ids | overlay_node_ids | fresh_record_ids
    missing_from_state = sorted((fresh_record_ids | overlay_node_ids) - state_node_ids)
    missing_from_capabilities = sorted((state_node_ids | overlay_node_ids) - fresh_record_ids)
    actions: list[str] = []
    if missing_from_capabilities:
        actions.append("sync_node_capabilities")
    if missing_from_state:
        actions.append("request_full_sync")
        actions.append("re_advertise_overlay_peers")
    if stale_record_ids:
        actions.append("prune_stale_node_capabilities")
        actions.append("sync_node_capabilities")
    actions = list(dict.fromkeys(actions))
    status = "converged" if not actions else "repair_recommended"
    return {
        "schemaVersion": 1,
        "status": status,
        "repairRecommended": bool(actions),
        "repairActions": actions,
        "localNodeId": str(local_node_id or "").strip() or None,
        "expectedNodeIds": sorted(expected_node_ids),
        "stateNodeIds": sorted(state_node_ids),
        "overlayReferenceNodeIds": sorted(overlay_node_ids),
        "capabilityNodeIds": sorted(fresh_record_ids),
        "staleCapabilityNodeIds": sorted(stale_record_ids),
        "missingFromStateNodeIds": missing_from_state,
        "missingFromCapabilitiesNodeIds": missing_from_capabilities,
    }


def sync_node_capabilities_from_cai_peers(
    *,
    state_payload: dict[str, Any],
    cai_url: str | None = None,
    CAI_url: str | None = None,
    policy: WalletPolicy | None = None,
    local_node_id: str | None = None,
    timeout_sec: int = 5,
    prune_after_seconds: int = 3600,
    max_peer_exchange_rounds: int = 1,
    max_peers: int = 16,
) -> NodeCapabilitySyncResult:
    resolved_cai_url = str(cai_url or CAI_url or "").strip()
    endpoint_path = "/v1/cai/node-capabilities"
    seed_peer_urls = discover_peer_cai_urls(
        state_payload=state_payload,
        cai_url=resolved_cai_url,
        endpoint_path=endpoint_path,
        local_node_id=local_node_id,
    )
    pending_peer_urls: list[tuple[str, int]] = [
        (url, 0)
        for url in _dedupe_peer_urls(
            [
                *seed_peer_urls,
                *_capability_peer_urls(
                    endpoint_path=endpoint_path,
                    policy=policy,
                    local_node_id=local_node_id,
                    cai_url=resolved_cai_url,
                ),
            ]
        )
    ]
    seen_peer_urls = {url for url, _depth in pending_peer_urls}
    attempted_peer_urls: list[str] = []
    imported_records = 0
    successful_peers = 0
    failed_peer_urls: list[str] = []
    peer_errors: list[dict[str, str]] = []
    max_attempts = max(0, int(max_peers))
    max_exchange_depth = max(0, int(max_peer_exchange_rounds))
    while pending_peer_urls and len(attempted_peer_urls) < max_attempts:
        peer_url, exchange_depth = pending_peer_urls.pop(0)
        attempted_peer_urls.append(peer_url)
        try:
            with urlopen(peer_url, timeout=timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
            imported_records += merge_remote_node_capabilities_payload(
                payload,
                source_url=peer_url,
                policy=policy,
            )
            successful_peers += 1
        except Exception as exc:
            failed_peer_urls.append(peer_url)
            peer_errors.append(_peer_error_payload(peer_url, exc))
            continue
        if exchange_depth >= max_exchange_depth:
            continue
        for discovered_url in _capability_peer_urls(
            endpoint_path=endpoint_path,
            policy=policy,
            local_node_id=local_node_id,
            cai_url=resolved_cai_url,
        ):
            if discovered_url in seen_peer_urls:
                continue
            seen_peer_urls.add(discovered_url)
            pending_peer_urls.append((discovered_url, exchange_depth + 1))
    pruned_records = prune_stale_node_capabilities(
        max_age_seconds=prune_after_seconds,
        policy=policy,
    )
    convergence_audit = node_capability_convergence_audit(
        state_payload=state_payload,
        records=list_node_capabilities(policy),
        local_node_id=local_node_id,
        max_age_seconds=prune_after_seconds,
        policy=policy,
    )
    return NodeCapabilitySyncResult(
        attempted_peers=len(attempted_peer_urls),
        successful_peers=successful_peers,
        imported_records=imported_records,
        pruned_records=pruned_records,
        peer_urls=attempted_peer_urls,
        failed_peers=len(failed_peer_urls),
        failed_peer_urls=failed_peer_urls,
        peer_errors=peer_errors,
        convergence_status=str(convergence_audit.get("status") or "unknown"),
        convergence_repair_recommended=bool(
            convergence_audit.get("repairRecommended")
        ),
        convergence_repair_actions=list(
            convergence_audit.get("repairActions") or []
        ),
        convergence_audit=convergence_audit,
    )


def _records_from_state_payload(
    *,
    state_payload: dict[str, Any] | None,
    cai_url: str | None,
    local_node_id: str | None,
    observed_at: str,
) -> list[NodeCapabilityRecord]:
    identities = (
        state_payload.get("nodeIdentities")
        if isinstance(state_payload, dict)
        else None
    )
    if not isinstance(identities, dict):
        return []
    records: list[NodeCapabilityRecord] = []
    for node_id, identity in identities.items():
        if not isinstance(identity, dict):
            continue
        records.append(
            _record_from_identity(
                node_id=str(node_id),
                identity=identity,
                state_payload=state_payload or {},
                cai_url=cai_url,
                local_node_id=local_node_id,
                observed_at=observed_at,
            )
        )
    return records


def _state_node_identity_ids(state_payload: dict[str, Any] | None) -> set[str]:
    identities = (
        state_payload.get("nodeIdentities")
        if isinstance(state_payload, dict)
        else None
    )
    if not isinstance(identities, dict):
        return set()
    return {str(node_id).strip() for node_id in identities if str(node_id).strip()}


def _state_overlay_reference_node_ids(state_payload: dict[str, Any] | None) -> set[str]:
    if not isinstance(state_payload, dict):
        return set()
    node_ids: set[str] = set()
    overlay_peers = state_payload.get("overlayPeers")
    if isinstance(overlay_peers, dict):
        for node_id, peers in overlay_peers.items():
            clean_node_id = str(node_id or "").strip()
            if clean_node_id:
                node_ids.add(clean_node_id)
            for peer_id in _raw_string_list(peers):
                node_ids.add(peer_id)
    advertised_peers = state_payload.get("overlayAdvertisedPeers")
    if isinstance(advertised_peers, dict):
        for node_id, peers in advertised_peers.items():
            clean_node_id = str(node_id or "").strip()
            if clean_node_id:
                node_ids.add(clean_node_id)
            for peer_id in _overlay_advertised_peer_node_ids(peers):
                node_ids.add(peer_id)
    return node_ids


def _record_from_identity(
    *,
    node_id: str,
    identity: dict[str, Any],
    state_payload: dict[str, Any],
    cai_url: str | None,
    local_node_id: str | None,
    observed_at: str,
) -> NodeCapabilityRecord:
    route_hints = _route_hints_for_node(state_payload, node_id)
    api_urls = candidate_identity_http_urls(identity)
    if local_node_id and node_id == local_node_id and cai_url:
        api_urls = list(dict.fromkeys([str(cai_url).rstrip("/"), *api_urls]))
    node_public_key_b64 = _identity_node_public_key_b64(identity)
    return NodeCapabilityRecord(
        node_id=node_id,
        source="local_state",
        source_url=str(cai_url).rstrip("/") if cai_url else None,
        last_seen_at=_identity_text(identity, "lastSeenAt", "last_seen_at")
        or observed_at,
        updated_at=observed_at,
        friendly_name=_identity_text(identity, "friendlyName", "friendly_name"),
        node_public_key_b64=node_public_key_b64,
        node_public_key_address=_node_public_key_address(
            public_key_b64=node_public_key_b64,
            declared_address=_identity_text(
                identity,
                "nodePublicKeyAddress",
                "node_public_key_address",
                "publicKeyAddress",
                "public_key_address",
                "signingPublicKeyAddress",
                "signing_public_key_address",
            ),
        ),
        api_urls=api_urls,
        data_endpoints=identity_transport_endpoints(identity, purpose="data"),
        worker_enabled=_identity_bool(identity, "workerEnabled", "worker_enabled"),
        relay_enabled=_identity_bool(identity, "relayEnabled", "relay_enabled"),
        validator_enabled=_identity_bool(
            identity,
            "validatorEnabled",
            "validator_enabled",
        ),
        validator_id=_identity_text(identity, "validatorId", "validator_id"),
        validator_state=_identity_text(identity, "validatorState", "validator_state"),
        worker_reward_address=_identity_text(
            identity,
            "workerRewardAddress",
            "worker_reward_address",
        ),
        worker_allowed_model_ids=_identity_string_list(
            identity,
            "workerAllowedModelIds",
            "worker_allowed_model_ids",
            normalize_model_ids=True,
        ),
        model_ids=_identity_string_list(
            identity,
            "modelIds",
            "model_ids",
            "models",
            normalize_model_ids=True,
        ),
        resource_summary=_resource_summary_for_node(
            identity=identity,
            state_payload=state_payload,
            node_id=node_id,
        ),
        readiness=_readiness_summary_with_cai_transport(identity),
        route_hints=route_hints,
    )


def _minimal_local_record(
    *,
    cai_url: str | None,
    local_node_id: str | None,
    observed_at: str,
) -> NodeCapabilityRecord:
    node_id = str(local_node_id or socket.gethostname()).strip() or "local-node"
    return NodeCapabilityRecord(
        node_id=node_id,
        source="local_state",
        source_url=str(cai_url).rstrip("/") if cai_url else None,
        last_seen_at=observed_at,
        updated_at=observed_at,
        api_urls=[str(cai_url).rstrip("/")] if cai_url else [],
    )


def _record_from_raw(
    raw: dict[str, Any],
    *,
    source: str,
    source_url: str,
    observed_at: str,
) -> NodeCapabilityRecord:
    node_id = str(raw.get("node_id") or raw.get("nodeId") or "").strip()
    node_public_key_b64 = _raw_text(
        raw,
        "node_public_key_b64",
        "nodePublicKeyB64",
        "public_key_b64",
        "publicKeyB64",
        "signing_public_key_b64",
        "signingPublicKeyB64",
    )
    return NodeCapabilityRecord(
        node_id=node_id,
        source=source,
        source_url=source_url,
        last_seen_at=str(
            raw.get("last_seen_at") or raw.get("lastSeenAt") or observed_at
        ),
        updated_at=str(raw.get("updated_at") or raw.get("updatedAt") or observed_at),
        friendly_name=raw.get("friendly_name") or raw.get("friendlyName"),
        api_urls=_raw_string_list(raw.get("api_urls") or raw.get("apiUrls")),
        data_endpoints=_raw_dict_list(
            raw.get("data_endpoints") or raw.get("dataEndpoints")
        ),
        worker_enabled=_raw_optional_bool(
            raw.get("worker_enabled", raw.get("workerEnabled"))
        ),
        relay_enabled=_raw_optional_bool(
            raw.get("relay_enabled", raw.get("relayEnabled"))
        ),
        validator_enabled=_raw_optional_bool(
            raw.get("validator_enabled", raw.get("validatorEnabled"))
        ),
        validator_id=raw.get("validator_id") or raw.get("validatorId"),
        validator_state=raw.get("validator_state") or raw.get("validatorState"),
        node_public_key_b64=node_public_key_b64,
        node_public_key_address=_node_public_key_address(
            public_key_b64=node_public_key_b64,
            declared_address=_raw_text(
                raw,
                "node_public_key_address",
                "nodePublicKeyAddress",
                "public_key_address",
                "publicKeyAddress",
                "signing_public_key_address",
                "signingPublicKeyAddress",
            ),
        ),
        worker_reward_address=(
            raw.get("worker_reward_address") or raw.get("workerRewardAddress")
        ),
        worker_allowed_model_ids=_raw_string_list(
            raw.get("worker_allowed_model_ids") or raw.get("workerAllowedModelIds")
        ),
        model_ids=_raw_string_list(raw.get("model_ids") or raw.get("modelIds")),
        resource_summary=(
            raw.get("resource_summary")
            if isinstance(raw.get("resource_summary"), dict)
            else raw.get("resourceSummary")
            if isinstance(raw.get("resourceSummary"), dict)
            else {}
        ),
        readiness=(
            raw.get("readiness") if isinstance(raw.get("readiness"), dict) else {}
        ),
        route_hints=(
            raw.get("route_hints")
            if isinstance(raw.get("route_hints"), dict)
            else raw.get("routeHints")
            if isinstance(raw.get("routeHints"), dict)
            else {}
        ),
    )


def _iter_capability_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("records") if isinstance(payload, dict) else None
    if isinstance(records, list):
        return [item for item in records if isinstance(item, dict)]
    record = payload.get("record") if isinstance(payload, dict) else None
    if isinstance(record, dict):
        return [record]
    return []


def _capability_peer_urls(
    *,
    endpoint_path: str,
    policy: WalletPolicy | None,
    local_node_id: str | None,
    cai_url: str | None,
) -> list[str]:
    normalized_local_node_id = str(local_node_id or "").strip()
    normalized_local_base = _normalize_base_url(cai_url)
    urls: list[str] = []
    for record in list_node_capabilities(policy):
        if normalized_local_node_id and record.node_id == normalized_local_node_id:
            continue
        for api_url in record.api_urls:
            peer_url = _endpoint_url_from_api_url(api_url, endpoint_path=endpoint_path)
            if not peer_url:
                continue
            if (
                normalized_local_base
                and _normalize_base_url(peer_url) == normalized_local_base
            ):
                continue
            urls.append(peer_url)
    return _dedupe_peer_urls(urls)


def _endpoint_url_from_api_url(api_url: str, *, endpoint_path: str) -> str | None:
    parsed = urlparse(str(api_url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    normalized_path = "/" + str(endpoint_path or "").lstrip("/")
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            normalized_path,
            "",
            "",
            "",
        )
    )


def _normalize_base_url(api_url: str | None) -> str | None:
    parsed = urlparse(str(api_url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))


def _dedupe_peer_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for url in urls:
        normalized = str(url or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _record_is_newer(
    incoming: NodeCapabilityRecord,
    previous: NodeCapabilityRecord,
) -> bool:
    incoming_time = _parse_iso_datetime(incoming.updated_at)
    previous_time = _parse_iso_datetime(previous.updated_at)
    if incoming_time is None or previous_time is None:
        return asdict(incoming) != asdict(previous)
    return incoming_time >= previous_time and asdict(incoming) != asdict(previous)


def _record_is_stale(
    record: NodeCapabilityRecord,
    *,
    max_age_seconds: int,
) -> bool:
    parsed = _parse_iso_datetime(record.last_seen_at)
    if parsed is None:
        return False
    return (datetime.now(tz=UTC) - parsed).total_seconds() > max(
        0,
        int(max_age_seconds),
    )


def _payload_signature_context(
    payload: dict[str, Any],
    *,
    signature_ok: bool,
) -> dict[str, Any]:
    signature = payload.get("signature")
    if not isinstance(signature, dict) or not signature_ok:
        return {
            "signature_present": False,
            "signature_valid": False,
            "signer_address": None,
            "public_key_address": None,
        }
    public_key_address = str(
        signature.get("public_key_address") or ""
    ).strip().lower()
    signer_address = str(signature.get("signer_address") or "").strip().lower()
    return {
        "signature_present": True,
        "signature_valid": True,
        "signer_address": signer_address or public_key_address or None,
        "public_key_address": public_key_address or None,
    }


def _apply_record_verification_metadata(
    record: NodeCapabilityRecord,
    *,
    signature_context: dict[str, Any],
    local_source: bool,
) -> None:
    signature_valid = bool(signature_context.get("signature_valid"))
    record.payload_signature_valid = (
        signature_valid if signature_context.get("signature_present") else None
    )
    record.payload_signer_address = (
        str(signature_context.get("signer_address") or "").strip().lower() or None
    )
    record.payload_public_key_address = (
        str(signature_context.get("public_key_address") or "").strip().lower()
        or None
    )

    verified, reason = _worker_verification_status(
        record,
        local_source=local_source,
    )
    record.worker_verified = verified
    record.worker_verification_reason = reason


def _worker_verification_status(
    record: NodeCapabilityRecord,
    *,
    local_source: bool,
) -> tuple[bool, str | None]:
    if not bool(record.worker_enabled):
        return False, "worker mode is not enabled"
    reward_address = str(record.worker_reward_address or "").strip().lower()
    if not reward_address:
        return False, "worker reward address is missing"
    if local_source:
        return True, "local worker state"
    if record.payload_signature_valid is not True:
        return False, "node capability payload is unsigned"

    signer_address = str(record.payload_signer_address or "").strip().lower()
    if not signer_address:
        return False, "node capability signer address is missing"
    if signer_address != reward_address:
        return False, "node capability signer does not match worker reward address"

    record_public_key_address = str(
        record.node_public_key_address or ""
    ).strip().lower()
    payload_public_key_address = str(
        record.payload_public_key_address or ""
    ).strip().lower()
    if (
        record_public_key_address
        and payload_public_key_address
        and record_public_key_address != payload_public_key_address
    ):
        return False, "node public key does not match capability signer"
    return True, "signed worker capability"


def _route_hints_for_node(
    state_payload: dict[str, Any],
    node_id: str,
) -> dict[str, Any]:
    overlay_peers = state_payload.get("overlayPeers") or {}
    advertised_peers = state_payload.get("overlayAdvertisedPeers") or {}
    topology = state_payload.get("topology") or {}
    connections = topology.get("connections") if isinstance(topology, dict) else {}
    return {
        "overlayPeerIds": _peer_ids_for_node(overlay_peers, node_id),
        "overlayAdvertisedPeers": _overlay_advertised_peer_texts(
            advertised_peers.get(node_id) if isinstance(advertised_peers, dict) else []
        ),
        "directPeerIds": sorted(
            str(peer_id)
            for peer_id in (
                connections.get(node_id, {}).keys()
                if isinstance(connections, dict)
                and isinstance(connections.get(node_id), dict)
                else []
            )
            if str(peer_id).strip()
        ),
    }


def _peer_ids_for_node(raw_peers: Any, node_id: str) -> list[str]:
    if not isinstance(raw_peers, dict):
        return []
    value = raw_peers.get(node_id)
    if isinstance(value, dict):
        return sorted(str(key) for key in value.keys() if str(key).strip())
    return _raw_string_list(value)


def _identity_bool(identity: dict[str, Any], *keys: str) -> bool | None:
    for key in keys:
        if key in identity and identity[key] is not None:
            return bool(identity[key])
    return None


def _identity_text(identity: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        raw = identity.get(key)
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    return None


def _identity_node_public_key_b64(identity: dict[str, Any]) -> str | None:
    return _identity_text(
        identity,
        "nodePublicKeyB64",
        "node_public_key_b64",
        "publicKeyB64",
        "public_key_b64",
        "signingPublicKeyB64",
        "signing_public_key_b64",
    )


def _node_public_key_address(
    *,
    public_key_b64: str | None,
    declared_address: str | None,
) -> str | None:
    normalized_declared = str(declared_address or "").strip().lower() or None
    normalized_public_key = str(public_key_b64 or "").strip()
    if not normalized_public_key:
        return normalized_declared
    try:
        return address_from_public_key_b64(normalized_public_key)
    except Exception:
        return normalized_declared


def _identity_string_list(
    identity: dict[str, Any],
    *keys: str,
    normalize_model_ids: bool = False,
) -> list[str]:
    for key in keys:
        raw = identity.get(key)
        values = _raw_string_list(raw)
        if values:
            if normalize_model_ids:
                return [_normalize_capability_model_id(value) for value in values]
            return values
    return []


def _normalize_capability_model_id(value: str) -> str:
    policy = NetworkModelPolicy()
    if str(value or "").strip() == "Qwen/Qwen3-0.6B-GGUF":
        return policy.network_default_model_id
    normalized = normalize_network_model_id(value, policy)
    if normalized == policy.network_default_execution_model_id:
        return policy.network_default_model_id
    return normalized


def _overlay_advertised_peer_texts(value: Any) -> list[str]:
    texts: list[str] = []
    for item in _overlay_advertised_peer_entries(value):
        text = _overlay_advertised_peer_text(item)
        if text:
            texts.append(text)
    return sorted(dict.fromkeys(texts))


def _overlay_advertised_peer_node_ids(value: Any) -> list[str]:
    node_ids: list[str] = []
    for item in _overlay_advertised_peer_entries(value):
        node_id = _overlay_advertised_peer_node_id(item)
        if node_id:
            node_ids.append(node_id)
    return sorted(dict.fromkeys(node_ids))


def _overlay_advertised_peer_entries(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return []


def _overlay_advertised_peer_text(value: Any) -> str | None:
    if isinstance(value, str):
        clean = value.strip()
        return clean or None
    if not isinstance(value, dict):
        return None
    for key in (
        "address",
        "multiaddr",
        "peerAddress",
        "peer_address",
        "peer",
        "nodeId",
        "node_id",
        "peerId",
        "peer_id",
        "id",
    ):
        raw = value.get(key)
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    return None


def _overlay_advertised_peer_node_id(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("nodeId", "node_id", "peerId", "peer_id", "id"):
            raw = value.get(key)
            clean = str(raw or "").strip()
            if clean:
                return clean
        for key in ("address", "multiaddr", "peerAddress", "peer_address", "peer"):
            node_id = _overlay_advertised_peer_node_id_from_text(value.get(key))
            if node_id:
                return node_id
        return None
    return _overlay_advertised_peer_node_id_from_text(value)


def _overlay_advertised_peer_node_id_from_text(value: Any) -> str | None:
    clean = str(value or "").strip()
    if not clean:
        return None
    matches = OVERLAY_MULTIADDR_P2P_COMPONENT_RE.findall(clean)
    if matches:
        return str(matches[-1]).strip() or None
    if clean.startswith("/") or "://" in clean or clean.startswith("{") or clean.startswith("["):
        return None
    return clean


def _raw_string_list(value: Any) -> list[str]:
    if isinstance(value, dict):
        iterable = value.keys()
    elif isinstance(value, (list, tuple, set)):
        iterable = value
    else:
        return []
    return sorted(dict.fromkeys(str(item).strip() for item in iterable if str(item).strip()))


def _raw_text(raw: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = raw.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _raw_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _raw_optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _resource_summary(identity: dict[str, Any]) -> dict[str, Any]:
    resources = identity.get("resources")
    if isinstance(resources, dict):
        return dict(resources)
    summary: dict[str, Any] = {}
    for source_key, target_key in (
        ("ramBytes", "ramBytes"),
        ("ram_bytes", "ramBytes"),
        ("vramBytes", "vramBytes"),
        ("vram_bytes", "vramBytes"),
        ("cpuCores", "cpuCores"),
        ("cpu_cores", "cpuCores"),
    ):
        if source_key in identity:
            summary[target_key] = identity[source_key]
    return summary


def _resource_summary_for_node(
    *,
    identity: dict[str, Any],
    state_payload: dict[str, Any],
    node_id: str,
) -> dict[str, Any]:
    state_summary = _node_memory_resource_summary(state_payload, node_id)
    identity_summary = _resource_summary(identity)
    return {**state_summary, **identity_summary}


def _node_memory_resource_summary(
    state_payload: dict[str, Any],
    node_id: str,
) -> dict[str, Any]:
    node_memory = state_payload.get("nodeMemory")
    if node_memory is None:
        node_memory = state_payload.get("node_memory")
    if not isinstance(node_memory, dict):
        return {}

    memory = node_memory.get(node_id)
    if memory is None:
        memory = node_memory.get(str(node_id))
    if memory is None:
        return {}

    ram_total = _memory_field_bytes(memory, "ramTotal", "ram_total")
    ram_available = _memory_field_bytes(memory, "ramAvailable", "ram_available")
    swap_total = _memory_field_bytes(memory, "swapTotal", "swap_total")
    swap_available = _memory_field_bytes(
        memory,
        "swapAvailable",
        "swap_available",
    )

    summary: dict[str, Any] = {}
    if ram_total is not None:
        summary["ramBytes"] = ram_total
        summary["ramTotalBytes"] = ram_total
    if ram_available is not None:
        summary["ramAvailableBytes"] = ram_available
    if swap_total is not None:
        summary["swapBytes"] = swap_total
        summary["swapTotalBytes"] = swap_total
    if swap_available is not None:
        summary["swapAvailableBytes"] = swap_available
    return summary


def _memory_field_bytes(memory: Any, *field_names: str) -> int | None:
    raw_value: Any = None
    for field_name in field_names:
        if isinstance(memory, dict):
            if field_name in memory:
                raw_value = memory[field_name]
                break
        elif hasattr(memory, field_name):
            raw_value = getattr(memory, field_name)
            break
    if raw_value is None:
        return None
    return _memory_value_bytes(raw_value)


def _memory_value_bytes(value: Any) -> int | None:
    if hasattr(value, "in_bytes"):
        try:
            return int(getattr(value, "in_bytes"))
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict):
        for key in ("inBytes", "in_bytes", "bytes"):
            if key not in value:
                continue
            try:
                return int(value[key])
            except (TypeError, ValueError):
                return None
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _readiness_summary(identity: dict[str, Any]) -> dict[str, Any]:
    readiness = identity.get("readiness")
    if isinstance(readiness, dict):
        return dict(readiness)
    model_readiness = identity.get("modelReadiness") or identity.get("model_readiness")
    if isinstance(model_readiness, dict):
        return {"models": model_readiness}
    return {}


def _readiness_summary_with_cai_transport(identity: dict[str, Any]) -> dict[str, Any]:
    readiness = _readiness_summary(identity)
    if not isinstance(readiness.get("caiOwnedTransport"), dict):
        readiness["caiOwnedTransport"] = cai_owned_transport_runtime_readiness()
    return readiness


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


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()

# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from .model import (
    curated_model_for_id,
    curated_model_registry,
    normalize_network_model_id,
    resolve_execution_model_id,
)


def worker_identity_state(identity: Any) -> tuple[bool | None, str | None]:
    if not isinstance(identity, dict):
        return None, None
    worker_enabled = identity.get("workerEnabled")
    if worker_enabled is None:
        worker_enabled = identity.get("worker_enabled")
    if worker_enabled is None:
        normalized_enabled = None
    else:
        normalized_enabled = bool(worker_enabled)
    reward_address = (
        str(
            identity.get("workerRewardAddress")
            or identity.get("worker_reward_address")
            or ""
        ).strip()
        or None
    )
    return normalized_enabled, reward_address


def identity_allowed_model_ids(identity: Any) -> list[str] | None:
    if not isinstance(identity, dict):
        return None
    raw = (
        identity.get("workerAllowedModelIds")
        or identity.get("worker_allowed_model_ids")
        or identity.get("allowedModelIds")
        or identity.get("allowed_model_ids")
    )
    if raw is None:
        return None
    if not isinstance(raw, list):
        return []
    return [
        normalize_network_model_id(str(item))
        for item in raw
        if str(item or "").strip()
    ]


def accepted_worker_model_ids(model_id: str) -> set[str]:
    normalized_model_id = normalize_network_model_id(model_id)
    accepted_model_ids = {
        normalized_model_id,
        normalize_network_model_id(resolve_execution_model_id(model_id)),
    }
    curated_model = curated_model_for_id(model_id)
    if curated_model is not None:
        accepted_model_ids.add(curated_model.model_id)
        accepted_model_ids.add(curated_model.execution_model_id)
        accepted_model_ids.update(curated_model.runtime_model_ids)
    for registry_model in curated_model_registry():
        normalized_runtime_ids = {
            normalize_network_model_id(item)
            for item in (
                registry_model.model_id,
                registry_model.execution_model_id,
                *registry_model.runtime_model_ids,
            )
        }
        if normalized_model_id not in normalized_runtime_ids:
            continue
        accepted_model_ids.add(registry_model.model_id)
        accepted_model_ids.add(registry_model.execution_model_id)
        accepted_model_ids.update(registry_model.runtime_model_ids)
    return {item for item in accepted_model_ids if item}


def worker_model_allowed(
    allowed_model_ids: list[str] | None,
    accepted_model_ids: set[str],
) -> bool | None:
    if allowed_model_ids is None:
        return None
    if not allowed_model_ids:
        return True
    return bool(accepted_model_ids.intersection(set(allowed_model_ids)))


def identity_last_seen_at(identity: Any) -> str | None:
    if not isinstance(identity, dict):
        return None
    raw = (
        identity.get("lastSeenAt")
        or identity.get("last_seen_at")
        or identity.get("lastSeen")
        or identity.get("last_seen")
    )
    return str(raw).strip() if raw is not None and str(raw).strip() else None


def capability_records_by_node_id(
    *,
    accepted_model_ids: set[str],
    stale_after_seconds: int,
    require_verified_capabilities: bool,
    wallet_policy_factory: Callable[[], Any],
    list_verified_worker_node_ids_func: Callable[..., set[str]],
    list_node_capabilities_func: Callable[..., list[Any]],
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    policy = wallet_policy_factory()
    verified_node_ids: set[str] | None = None
    if require_verified_capabilities:
        verified_node_ids = list_verified_worker_node_ids_func(
            policy,
            accepted_model_ids=accepted_model_ids,
            max_age_seconds=stale_after_seconds,
        )
        if not verified_node_ids:
            return {}
    for record in list_node_capabilities_func(policy):
        node_id = str(record.node_id or "").strip()
        if not node_id:
            continue
        if capability_record_is_stale(
            record,
            stale_after_seconds=stale_after_seconds,
        ):
            continue
        if verified_node_ids is not None and node_id not in verified_node_ids:
            continue
        records[node_id] = record
    return records


def capability_record_is_stale(
    record: Any,
    *,
    stale_after_seconds: int,
) -> bool:
    parsed_last_seen = parse_iso_datetime(str(record.last_seen_at or "").strip() or None)
    if parsed_last_seen is None:
        return False
    return max(
        0,
        int((datetime.now(tz=UTC) - parsed_last_seen).total_seconds()),
    ) > max(0, int(stale_after_seconds))


def capability_identity_from_record(record: Any) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "workerEnabled": record.worker_enabled,
        "workerRewardAddress": record.worker_reward_address,
        "workerAllowedModelIds": list(record.worker_allowed_model_ids or []),
        "lastSeenAt": record.last_seen_at,
    }
    if record.friendly_name:
        identity["friendlyName"] = record.friendly_name
    if record.node_public_key_b64:
        identity["nodePublicKeyB64"] = record.node_public_key_b64
    if record.node_public_key_address:
        identity["nodePublicKeyAddress"] = record.node_public_key_address
    if record.relay_enabled is not None:
        identity["relayEnabled"] = record.relay_enabled
    if record.readiness:
        identity["readiness"] = dict(record.readiness)
    return identity


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def participant_route_reachable(network_audit: dict[str, Any]) -> bool | None:
    participant_count = int(network_audit.get("participantCount") or 0)
    if participant_count <= 1:
        return True
    if bool(network_audit.get("caiOwnedTransportExecuted")):
        return True
    proof = network_audit.get("caiOwnedTransportExecutionProof")
    if isinstance(proof, dict):
        execution_audit = proof.get("executionAudit")
        if isinstance(execution_audit, dict) and bool(execution_audit.get("verified")):
            return True
    transport_mode = str(network_audit.get("transportMode") or "").strip()
    if transport_mode in {
        "multi_worker_direct",
        "multi_worker_relay",
        "multi_worker_partial_direct",
    }:
        return True
    if transport_mode in {"multi_worker_disconnected", "multi_worker_overlay_only"}:
        return False
    return None


def build_participant_eligibility_audit(
    *,
    state_payload: dict[str, Any] | None,
    instance_snapshot: dict[str, Any] | None,
    requested_model_id: str,
    execution_model_id: str,
    network_audit: dict[str, Any],
    participant_node_ids_func: Callable[[dict[str, Any] | None], list[str]],
    worker_capability_verification_required_func: Callable[[], bool],
    capability_records_by_node_id_func: Callable[..., dict[str, Any]],
    verified_worker_node_ids_func: Callable[..., set[str]] | None = None,
    wallet_policy_factory: Callable[[], Any] | None = None,
    stale_after_seconds: int = 300,
) -> dict[str, Any]:
    participant_node_ids = participant_node_ids_func(instance_snapshot)
    identities = (
        state_payload.get("nodeIdentities")
        if isinstance(state_payload, dict)
        else None
    )
    identities = identities if isinstance(identities, dict) else {}
    accepted_model_ids = accepted_worker_model_ids(requested_model_id)
    accepted_model_ids.update(accepted_worker_model_ids(execution_model_id))
    now = datetime.now(tz=UTC)
    route_reachable = participant_route_reachable(network_audit)
    require_verified_capabilities = worker_capability_verification_required_func()
    capability_records = capability_records_by_node_id_func(
        accepted_model_ids=accepted_model_ids,
        stale_after_seconds=stale_after_seconds,
        require_verified_capabilities=require_verified_capabilities,
    )
    verified_worker_node_ids = set(capability_records)
    if (
        require_verified_capabilities
        and verified_worker_node_ids_func is not None
        and wallet_policy_factory is not None
    ):
        verified_worker_node_ids.update(
            str(node_id).strip()
            for node_id in verified_worker_node_ids_func(
                wallet_policy_factory(),
                accepted_model_ids=accepted_model_ids,
                max_age_seconds=stale_after_seconds,
            )
            if str(node_id or "").strip()
        )
    fatal_reasons: list[str] = []
    warnings: list[str] = []
    participant_items: list[dict[str, Any]] = []

    for node_id in participant_node_ids:
        capability_record = capability_records.get(str(node_id))
        identity = identities.get(node_id)
        capability_backed = False
        if identity is None and capability_record is not None:
            identity = capability_identity_from_record(capability_record)
            capability_backed = True
        worker_enabled, reward_address = worker_identity_state(identity)
        allowed_model_ids = identity_allowed_model_ids(identity)
        last_seen_at = identity_last_seen_at(identity)
        last_seen_age_seconds: int | None = None
        parsed_last_seen = parse_iso_datetime(last_seen_at)
        if parsed_last_seen is not None:
            last_seen_age_seconds = max(0, int((now - parsed_last_seen).total_seconds()))

        reasons: list[str] = []
        node_warnings: list[str] = []
        if identity is None:
            node_warnings.append("node identity is missing from state payload")
        elif capability_backed:
            node_warnings.append(
                "node identity is missing from live state; using validator-attested capability record"
            )
        if worker_enabled is False:
            reasons.append("worker mode is explicitly disabled")
        elif worker_enabled is None:
            node_warnings.append("worker mode is unknown in state payload")
        if not reward_address:
            node_warnings.append("worker reward address is unknown in state payload")
        model_allowed = None
        if allowed_model_ids is not None:
            model_allowed = worker_model_allowed(allowed_model_ids, accepted_model_ids)
            if not model_allowed:
                reasons.append("execution model is not in worker allowed model ids")
        verified_capability = (
            node_id in verified_worker_node_ids if require_verified_capabilities else None
        )
        if require_verified_capabilities and not verified_capability:
            reasons.append("worker capability is not verified")
        if (
            last_seen_age_seconds is not None
            and last_seen_age_seconds > max(0, int(stale_after_seconds))
        ):
            reasons.append(
                f"worker identity is stale: last seen {last_seen_age_seconds}s ago"
            )

        if reasons:
            fatal_reasons.extend(f"{node_id}: {reason}" for reason in reasons)
        warnings.extend(f"{node_id}: {warning}" for warning in node_warnings)
        participant_items.append(
            {
                "nodeId": node_id,
                "identityKnown": not capability_backed and identity is not None,
                "capabilityBacked": capability_backed,
                "workerEnabled": worker_enabled,
                "workerRewardAddressKnown": bool(reward_address),
                "verifiedCapability": verified_capability,
                "allowedModelIds": allowed_model_ids,
                "modelAllowed": model_allowed,
                "lastSeenAt": last_seen_at,
                "lastSeenAgeSeconds": last_seen_age_seconds,
                "fatalReasons": reasons,
                "warnings": node_warnings,
            }
        )

    if route_reachable is False:
        fatal_reasons.append(
            "multi-worker route was not proven executable by direct or active relay audit"
        )
    execution_strategy = network_audit.get("llamaCppExecutionStrategy")
    cai_owned_transport_executed = bool(
        isinstance(execution_strategy, dict)
        and execution_strategy.get("caiOwnedTransportExecuted")
    )
    compute_cell = network_audit.get("llamaCppComputeCell")
    if isinstance(compute_cell, dict):
        profile = str(compute_cell.get("profile") or "").strip()
        if (
            profile in {"failed_sharded_cell", "wan_risky_sharded_cell"}
            and not cai_owned_transport_executed
        ):
            fatal_reasons.append(
                "multi-worker llama.cpp compute-cell is not settlement-safe: "
                f"{profile}"
            )
        elif (
            bool(compute_cell.get("readyForLlamaCppRpc")) is False
            and not cai_owned_transport_executed
        ):
            warnings.append(
                "multi-worker llama.cpp compute-cell has no successful runtime "
                "RPC proof in RouteHealth"
            )
    if (
        isinstance(execution_strategy, dict)
        and bool(execution_strategy.get("requiresCaiOwnedTransport"))
        and not cai_owned_transport_executed
    ):
        fatal_reasons.append(
            "multi-worker execution requires CAI-owned WAN-safe transport, "
            "but no CAI transport execution proof was recorded"
        )

    return {
        "schemaVersion": 1,
        "canSettle": not fatal_reasons,
        "participantCount": len(participant_node_ids),
        "checkedModelIds": sorted(accepted_model_ids),
        "routeReachable": route_reachable,
        "fatalReasons": fatal_reasons,
        "warnings": warnings,
        "participants": participant_items,
    }

# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Callable
from urllib.parse import urlsplit

from .model import WalletPolicy


NowIsoFunc = Callable[[], str]
PeerSyncFunc = Callable[..., Any]
GetJsonFunc = Callable[..., Any]
LogBestEffortFailureFunc = Callable[[str, Exception], None]


def participant_node_ids(instance_snapshot: dict[str, Any] | None) -> list[str]:
    if not isinstance(instance_snapshot, dict):
        return []
    participants = instance_snapshot.get("participants") or []
    if not isinstance(participants, list):
        return []
    ordered: list[str] = []
    seen: set[str] = set()
    for item in participants:
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("node_id") or "").strip()
        if not node_id or node_id in seen:
            continue
        ordered.append(node_id)
        seen.add(node_id)
    return ordered


def cai_owned_transport_expected_participant_node_ids(
    instance_snapshot: dict[str, Any] | None,
    *,
    default_node_ids: list[str],
) -> list[str]:
    if not isinstance(instance_snapshot, dict):
        return list(default_node_ids)
    explicit = instance_snapshot.get("caiOwnedTransportParticipantNodeIds")
    if not isinstance(explicit, list):
        explicit = instance_snapshot.get("cai_owned_transport_participant_node_ids")
    if not isinstance(explicit, list):
        return list(default_node_ids)
    cleaned = [
        str(node_id or "").strip()
        for node_id in explicit
        if str(node_id or "").strip()
    ]
    return list(dict.fromkeys(cleaned)) or list(default_node_ids)


def cai_owned_transport_expected_executor_node_ids(
    instance_snapshot: dict[str, Any] | None,
    *,
    default_node_ids: list[str],
) -> list[str]:
    if not isinstance(instance_snapshot, dict):
        return list(default_node_ids)
    explicit = instance_snapshot.get("caiOwnedTransportExecutorNodeIds")
    if not isinstance(explicit, list):
        explicit = instance_snapshot.get("cai_owned_transport_executor_node_ids")
    if not isinstance(explicit, list):
        return list(default_node_ids)
    cleaned = [
        str(node_id or "").strip()
        for node_id in explicit
        if str(node_id or "").strip()
    ]
    return list(dict.fromkeys(cleaned)) or list(default_node_ids)


def instance_model_id(instance_snapshot: dict[str, Any] | None) -> str | None:
    if not isinstance(instance_snapshot, dict):
        return None
    model_id = str(
        instance_snapshot.get("model_id") or instance_snapshot.get("modelId") or ""
    ).strip()
    if model_id:
        return model_id
    shard_assignments = instance_snapshot.get("shardAssignments") or {}
    if not isinstance(shard_assignments, dict):
        return None
    model_id = str(
        shard_assignments.get("model_id") or shard_assignments.get("modelId") or ""
    ).strip()
    return model_id or None


def execution_cai_owned_transport_proof(
    instance_snapshot: dict[str, Any] | None,
    participant_node_ids: list[str],
    *,
    wallet_policy: WalletPolicy | None = None,
    latest_completed_cai_owned_transport_proof_for_instance_func: PeerSyncFunc,
    validate_cai_owned_transport_execution_proof_func: PeerSyncFunc,
) -> tuple[dict[str, Any] | None, bool, str | None]:
    proof = None
    if isinstance(instance_snapshot, dict):
        candidate = instance_snapshot.get("caiOwnedTransportProof")
        if not isinstance(candidate, dict):
            candidate = instance_snapshot.get("caiOwnedTransportExecutionProof")
        if isinstance(candidate, dict):
            proof = candidate
    if proof is None:
        instance_id = (
            str(instance_snapshot.get("instance_id") or "").strip()
            if isinstance(instance_snapshot, dict)
            else ""
        )
        proof = latest_completed_cai_owned_transport_proof_for_instance_func(
            instance_id,
            participant_node_ids=participant_node_ids,
            model_id=instance_model_id(instance_snapshot),
            policy=wallet_policy,
        )
    if proof is None:
        return None, False, None
    expected_transport_participants = cai_owned_transport_expected_participant_node_ids(
        instance_snapshot,
        default_node_ids=participant_node_ids,
    )
    expected_executors = cai_owned_transport_expected_executor_node_ids(
        instance_snapshot,
        default_node_ids=participant_node_ids,
    )
    valid, error = validate_cai_owned_transport_execution_proof_func(
        proof,
        participant_node_ids=expected_transport_participants,
        executor_node_ids=expected_executors,
    )
    if valid:
        execution_audit = proof.get("executionAudit")
        if not isinstance(execution_audit, dict) or not bool(
            execution_audit.get("verified")
        ):
            return (
                proof,
                False,
                "CAI-owned transport execution audit is missing or not verified.",
            )
    return proof, valid, error


def audit_error_status(exc: Exception) -> dict[str, str]:
    return {
        "status": "failed",
        "errorType": type(exc).__name__,
        "message": str(exc),
    }


def result_int_attr(result: Any, name: str) -> int:
    value = getattr(result, name, 0)
    if not isinstance(value, int):
        return 0
    return value


def result_list_attr(result: Any, name: str) -> list[Any]:
    value = getattr(result, name, [])
    return list(value) if isinstance(value, (list, tuple)) else []


def chain_sync_result_audit(result: Any) -> dict[str, Any]:
    return {
        "attemptedPeers": result_int_attr(result, "attempted_peers"),
        "successfulPeers": result_int_attr(result, "successful_peers"),
        "failedPeers": result_int_attr(result, "failed_peers"),
        "importedBlocks": result_int_attr(result, "imported_blocks"),
        "importedTransactions": result_int_attr(result, "imported_transactions"),
        "peerUrls": result_list_attr(result, "peer_urls"),
        "failedPeerUrls": result_list_attr(result, "failed_peer_urls"),
        "peerErrors": result_list_attr(result, "peer_errors"),
    }


def validator_sync_result_audit(result: Any) -> dict[str, Any]:
    return {
        "attemptedPeers": result_int_attr(result, "attempted_peers"),
        "successfulPeers": result_int_attr(result, "successful_peers"),
        "failedPeers": result_int_attr(result, "failed_peers"),
        "importedRecords": result_int_attr(result, "imported_records"),
        "peerUrls": result_list_attr(result, "peer_urls"),
        "failedPeerUrls": result_list_attr(result, "failed_peer_urls"),
        "peerErrors": result_list_attr(result, "peer_errors"),
    }


def node_capability_sync_result_audit(result: Any) -> dict[str, Any]:
    return {
        "attemptedPeers": result_int_attr(result, "attempted_peers"),
        "successfulPeers": result_int_attr(result, "successful_peers"),
        "failedPeers": result_int_attr(result, "failed_peers"),
        "importedRecords": result_int_attr(result, "imported_records"),
        "prunedRecords": result_int_attr(result, "pruned_records"),
        "peerUrls": result_list_attr(result, "peer_urls"),
        "failedPeerUrls": result_list_attr(result, "failed_peer_urls"),
        "peerErrors": result_list_attr(result, "peer_errors"),
        "convergenceStatus": str(
            getattr(result, "convergence_status", "unknown") or "unknown"
        ),
        "convergenceRepairRecommended": bool(
            getattr(result, "convergence_repair_recommended", False)
        ),
        "convergenceRepairActions": result_list_attr(
            result,
            "convergence_repair_actions",
        ),
    }


def chain_push_result_audit(result: Any) -> dict[str, Any]:
    attempted_peers = result_int_attr(result, "attempted_peers")
    successful_peers = result_int_attr(result, "successful_peers")
    failed_peers = result_int_attr(result, "failed_peers")
    if failed_peers <= 0 and attempted_peers > successful_peers:
        failed_peers = attempted_peers - successful_peers
    return {
        "attemptedPeers": attempted_peers,
        "successfulPeers": successful_peers,
        "failedPeers": failed_peers,
        "peerUrls": result_list_attr(result, "peer_urls"),
        "failedPeerUrls": result_list_attr(result, "failed_peer_urls"),
        "peerErrors": result_list_attr(result, "peer_errors"),
    }


def run_preflight_peer_sync(
    *,
    state_payload: dict[str, Any] | None,
    cai_url: str,
    wallet_policy: WalletPolicy | None = None,
    sync_validator_set_from_cai_peers_func: PeerSyncFunc,
    sync_chain_from_cai_peers_func: PeerSyncFunc,
    now_iso_func: NowIsoFunc,
) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "attempted": False,
        "statePayloadAvailable": state_payload is not None,
        "validatorSet": {"status": "skipped"},
        "chain": {"status": "skipped"},
    }
    if state_payload is None:
        return audit

    audit["attempted"] = True
    audit["checkedAt"] = now_iso_func()

    try:
        result = sync_validator_set_from_cai_peers_func(
            state_payload=state_payload,
            cai_url=cai_url,
            policy=wallet_policy,
        )
        audit["validatorSet"] = {"status": "ok", **validator_sync_result_audit(result)}
    except Exception as exc:
        audit["validatorSet"] = audit_error_status(exc)

    try:
        result = sync_chain_from_cai_peers_func(
            state_payload=state_payload,
            cai_url=cai_url,
            policy=wallet_policy,
            timeout_sec=2.5,
        )
        audit["chain"] = {"status": "ok", **chain_sync_result_audit(result)}
    except Exception as exc:
        audit["chain"] = audit_error_status(exc)

    return audit


def run_chain_push_audit(
    *,
    state_payload: dict[str, Any] | None,
    cai_url: str,
    wallet_policy: WalletPolicy | None = None,
    timeout_sec: float = 0.75,
    push_chain_to_cai_peers_func: PeerSyncFunc,
    now_iso_func: NowIsoFunc,
) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "attempted": False,
        "statePayloadAvailable": state_payload is not None,
        "status": "skipped",
        "attemptedPeers": 0,
        "successfulPeers": 0,
        "peerUrls": [],
    }
    if state_payload is None:
        return audit

    audit["attempted"] = True
    audit["checkedAt"] = now_iso_func()
    try:
        result = push_chain_to_cai_peers_func(
            state_payload=state_payload,
            cai_url=cai_url,
            policy=wallet_policy,
            timeout_sec=timeout_sec,
        )
    except Exception as exc:
        audit.update(audit_error_status(exc))
        return audit

    audit.update({"status": "ok", **chain_push_result_audit(result)})
    return audit


def execution_compute_cell_strategy(
    participant_node_ids: list[str],
    route_health_records: list[Any] | None,
    *,
    plan_llama_cpp_distributed_execution_func: PeerSyncFunc,
    log_best_effort_failure_func: LogBestEffortFailureFunc,
) -> dict[str, Any] | None:
    if len(participant_node_ids) <= 1:
        return None
    source_node_id = participant_node_ids[0]
    sink_node_ids = participant_node_ids[1:]
    try:
        strategy = plan_llama_cpp_distributed_execution_func(
            source_node_id,
            sink_node_ids,
            route_health_records or [],
        )
    except Exception as exc:
        log_best_effort_failure_func("distributed compute cell strategy planning", exc)
        return None
    return strategy if isinstance(strategy, dict) else None


def augment_route_health_records_from_worker_attestations(
    route_health_records: list[Any],
    *,
    participant_node_ids: list[str],
    wallet_policy: WalletPolicy | None,
    list_worker_capability_attestations_func: PeerSyncFunc,
    get_json_func: GetJsonFunc,
    log_best_effort_failure_func: LogBestEffortFailureFunc,
) -> list[Any]:
    if len(participant_node_ids) <= 1:
        return list(route_health_records or [])

    records = list(route_health_records or [])
    endpoints = route_health_endpoints_from_worker_attestations(
        participant_node_ids,
        wallet_policy=wallet_policy,
        list_worker_capability_attestations_func=list_worker_capability_attestations_func,
        log_best_effort_failure_func=log_best_effort_failure_func,
    )
    if not endpoints:
        return records

    seen = {_route_health_record_key(record) for record in records}
    for endpoint in endpoints:
        try:
            payload = get_json_func(endpoint, timeout=5)
        except Exception as exc:
            log_best_effort_failure_func(
                f"route health peer sync from {endpoint}",
                exc,
            )
            continue
        peer_records = payload.get("records") if isinstance(payload, dict) else None
        if not isinstance(peer_records, list):
            continue
        for record in peer_records:
            if not isinstance(record, dict):
                continue
            key = _route_health_record_key(record)
            if key in seen:
                continue
            records.append(record)
            seen.add(key)
    return records


def route_health_endpoints_from_worker_attestations(
    participant_node_ids: list[str],
    *,
    wallet_policy: WalletPolicy | None,
    list_worker_capability_attestations_func: PeerSyncFunc,
    log_best_effort_failure_func: LogBestEffortFailureFunc,
    now: datetime | None = None,
) -> list[str]:
    participant_node_id_set = {
        str(node_id or "").strip()
        for node_id in participant_node_ids
        if str(node_id or "").strip()
    }
    if not participant_node_id_set:
        return []
    try:
        attestations = list_worker_capability_attestations_func(policy=wallet_policy)
    except Exception as exc:
        log_best_effort_failure_func(
            "route health worker capability attestation lookup",
            exc,
        )
        return []

    latest_by_worker: dict[str, tuple[datetime, str]] = {}
    active_now = now or datetime.now(tz=UTC)
    for attestation in attestations:
        worker_node_id = str(getattr(attestation, "worker_node_id", "") or "").strip()
        if worker_node_id not in participant_node_id_set:
            continue
        if not bool(getattr(attestation, "accepted", False)):
            continue
        expires_at = _parse_iso_datetime(getattr(attestation, "expires_at", None))
        if expires_at is not None and expires_at <= active_now:
            continue
        endpoint = _route_health_endpoint_from_worker_attestation(attestation)
        if endpoint is None:
            continue
        updated_at = (
            _parse_iso_datetime(getattr(attestation, "updated_at", None))
            or _parse_iso_datetime(getattr(attestation, "last_seen_at", None))
            or _parse_iso_datetime(getattr(attestation, "created_at", None))
            or datetime.min.replace(tzinfo=UTC)
        )
        current = latest_by_worker.get(worker_node_id)
        if current is None or updated_at >= current[0]:
            latest_by_worker[worker_node_id] = (updated_at, endpoint)

    ordered: list[str] = []
    seen: set[str] = set()
    for node_id in participant_node_ids:
        item = latest_by_worker.get(str(node_id))
        if item is None:
            continue
        endpoint = item[1]
        if endpoint in seen:
            continue
        ordered.append(endpoint)
        seen.add(endpoint)
    return ordered


def _route_health_endpoint_from_worker_attestation(attestation: Any) -> str | None:
    source_url = str(getattr(attestation, "source_url", "") or "").strip()
    if not source_url:
        probe_result = getattr(attestation, "probe_result", None)
        if isinstance(probe_result, dict):
            source_url = str(
                probe_result.get("sourceUrl")
                or probe_result.get("source_url")
                or ""
            ).strip()
    return _route_health_endpoint_from_worker_source_url(source_url)


def _route_health_endpoint_from_worker_source_url(source_url: str) -> str | None:
    normalized = str(source_url or "").strip().rstrip("/")
    if not normalized:
        return None
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    endpoint_suffixes = (
        "/v1/cai/route-health",
        "/v1/cai/node-capabilities",
        "/v1/cai/worker-capability/challenge",
    )
    for suffix in endpoint_suffixes:
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)].rstrip("/") + "/v1/cai/route-health"
    return normalized + "/v1/cai/route-health"


def _route_health_record_key(record: Any) -> tuple[Any, ...]:
    return (
        _route_health_record_field(record, "route_id"),
        _route_health_record_field(record, "source_node_id"),
        _route_health_record_field(record, "sink_node_id"),
        _route_health_record_field(record, "route_type"),
        _route_health_record_field(record, "endpoint_url"),
        _route_health_record_field(record, "checked_at"),
    )


def _route_health_record_field(record: Any, field_name: str) -> Any:
    if isinstance(record, dict):
        return record.get(field_name)
    return getattr(record, field_name, None)


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


def participant_socket_adjacency(
    state_payload: dict[str, Any] | None,
    participant_node_ids: list[str],
) -> dict[str, set[str]]:
    adjacency = {node_id: set() for node_id in participant_node_ids}
    if not state_payload or not participant_node_ids:
        return adjacency

    topology = state_payload.get("topology") or {}
    if not isinstance(topology, dict):
        return adjacency
    connections = topology.get("connections") or {}
    if not isinstance(connections, dict):
        return adjacency

    participant_node_id_set = set(participant_node_ids)
    for source_node_id, sink_payload in connections.items():
        normalized_source_node_id = str(source_node_id or "").strip()
        if normalized_source_node_id not in participant_node_id_set:
            continue
        if not isinstance(sink_payload, dict):
            continue
        for sink_node_id, edge_payload in sink_payload.items():
            normalized_sink_node_id = str(sink_node_id or "").strip()
            if normalized_sink_node_id not in participant_node_id_set:
                continue
            if normalized_sink_node_id == normalized_source_node_id:
                continue
            if not isinstance(edge_payload, list) or not edge_payload:
                continue
            if any(
                isinstance(edge_item, dict)
                and isinstance(edge_item.get("sinkMultiaddr"), dict)
                for edge_item in edge_payload
            ):
                adjacency[normalized_source_node_id].add(normalized_sink_node_id)
    return adjacency


def checked_direct_socket_links(
    adjacency: dict[str, set[str]],
) -> list[dict[str, Any]]:
    checked_links: list[dict[str, Any]] = []
    emitted_pairs: set[tuple[str, str]] = set()

    for source_node_id, sink_node_ids in adjacency.items():
        for sink_node_id in sink_node_ids:
            pair = tuple(sorted((source_node_id, sink_node_id)))
            if pair in emitted_pairs:
                continue
            emitted_pairs.add(pair)
            reverse_present = source_node_id in adjacency.get(sink_node_id, set())
            checked_links.append(
                {
                    "sourceNodeId": pair[0],
                    "sinkNodeId": pair[1],
                    "forwardObserved": sink_node_id in adjacency.get(
                        source_node_id,
                        set(),
                    ),
                    "reverseObserved": reverse_present,
                    "bidirectional": reverse_present,
                }
            )

    checked_links.sort(key=lambda item: (item["sourceNodeId"], item["sinkNodeId"]))
    return checked_links


def checked_overlay_links(
    state_payload: dict[str, Any] | None,
    participant_node_ids: list[str],
) -> list[dict[str, str]]:
    if not state_payload or not participant_node_ids:
        return []

    overlay_peers = state_payload.get("overlayPeers") or {}
    if not isinstance(overlay_peers, dict):
        return []

    participant_node_id_set = set(participant_node_ids)
    checked_links: list[dict[str, str]] = []
    emitted_pairs: set[tuple[str, str]] = set()

    for source_node_id, peer_payload in overlay_peers.items():
        normalized_source_node_id = str(source_node_id or "").strip()
        if normalized_source_node_id not in participant_node_id_set:
            continue
        if not isinstance(peer_payload, (list, tuple, set, dict)):
            continue
        if isinstance(peer_payload, dict):
            peer_iterable = peer_payload.keys()
        else:
            peer_iterable = peer_payload
        for peer_node_id in peer_iterable:
            normalized_peer_node_id = str(peer_node_id or "").strip()
            if (
                normalized_peer_node_id not in participant_node_id_set
                or normalized_peer_node_id == normalized_source_node_id
            ):
                continue
            pair = tuple(sorted((normalized_source_node_id, normalized_peer_node_id)))
            if pair in emitted_pairs:
                continue
            emitted_pairs.add(pair)
            checked_links.append(
                {
                    "sourceNodeId": pair[0],
                    "sinkNodeId": pair[1],
                }
            )

    checked_links.sort(key=lambda item: (item["sourceNodeId"], item["sinkNodeId"]))
    return checked_links


def is_strongly_connected_participant_graph(
    adjacency: dict[str, set[str]],
    participant_node_ids: list[str],
) -> bool:
    if len(participant_node_ids) <= 1:
        return True

    participant_node_id_set = set(participant_node_ids)

    def _reachable_from(start_node_id: str, graph: dict[str, set[str]]) -> set[str]:
        seen = {start_node_id}
        stack = [start_node_id]
        while stack:
            current_node_id = stack.pop()
            for next_node_id in graph.get(current_node_id, set()):
                if (
                    next_node_id not in participant_node_id_set
                    or next_node_id in seen
                ):
                    continue
                seen.add(next_node_id)
                stack.append(next_node_id)
        return seen

    start_node_id = participant_node_ids[0]
    forward_seen = _reachable_from(start_node_id, adjacency)
    if len(forward_seen) != len(participant_node_ids):
        return False

    reverse_adjacency = {node_id: set() for node_id in participant_node_ids}
    for source_node_id, sink_node_ids in adjacency.items():
        for sink_node_id in sink_node_ids:
            if sink_node_id in reverse_adjacency:
                reverse_adjacency[sink_node_id].add(source_node_id)

    reverse_seen = _reachable_from(start_node_id, reverse_adjacency)
    return len(reverse_seen) == len(participant_node_ids)


def coordinator_direct_fanout_candidate_node_ids(
    adjacency: dict[str, set[str]],
    participant_node_ids: list[str],
) -> list[str]:
    if len(participant_node_ids) <= 1:
        return list(participant_node_ids)

    participant_node_id_set = set(participant_node_ids)
    candidate_node_ids: list[str] = []
    for source_node_id in participant_node_ids:
        reachable = adjacency.get(source_node_id, set())
        if all(
            other_node_id == source_node_id or other_node_id in reachable
            for other_node_id in participant_node_id_set
        ):
            candidate_node_ids.append(source_node_id)
    return candidate_node_ids


def relay_capability_snapshot(
    state_payload: dict[str, Any] | None,
    participant_node_ids: list[str],
) -> tuple[list[str], list[str]]:
    if not isinstance(state_payload, dict):
        return [], []

    node_identities = state_payload.get("nodeIdentities")
    if not isinstance(node_identities, dict):
        return [], []

    relay_capable_node_ids: list[str] = []
    participant_node_id_set = set(participant_node_ids)
    for node_id, identity_payload in node_identities.items():
        normalized_node_id = str(node_id or "").strip()
        if not normalized_node_id or not isinstance(identity_payload, dict):
            continue
        relay_enabled = identity_payload.get("relayEnabled")
        if relay_enabled is None:
            relay_enabled = identity_payload.get("relay_enabled")
        if bool(relay_enabled):
            relay_capable_node_ids.append(normalized_node_id)

    relay_capable_node_ids = sorted(dict.fromkeys(relay_capable_node_ids))
    relay_transit_candidate_node_ids = [
        node_id
        for node_id in relay_capable_node_ids
        if node_id not in participant_node_id_set
    ]
    return relay_capable_node_ids, relay_transit_candidate_node_ids


def active_relay_routes(
    instance_snapshot: dict[str, Any] | None,
    participant_node_ids: list[str],
) -> list[dict[str, Any]]:
    if not isinstance(instance_snapshot, dict):
        return []

    relay_routes_by_node = (
        instance_snapshot.get("relay_routes_by_node")
        or instance_snapshot.get("relayRoutesByNode")
        or {}
    )
    if not isinstance(relay_routes_by_node, dict):
        return []

    participant_node_id_set = set(participant_node_ids)
    active_routes: list[dict[str, Any]] = []
    for source_node_id, routes in relay_routes_by_node.items():
        normalized_source_node_id = str(source_node_id or "").strip()
        if normalized_source_node_id not in participant_node_id_set:
            continue
        if not isinstance(routes, list):
            continue
        for route in routes:
            if not isinstance(route, dict):
                continue
            normalized_sink_node_id = str(route.get("sinkNodeId") or "").strip()
            if normalized_sink_node_id not in participant_node_id_set:
                continue
            transit_node_id = str(route.get("transitNodeId") or "").strip()
            active_routes.append(
                {
                    "candidateOnly": False,
                    "pathNodeIds": [
                        normalized_source_node_id,
                        transit_node_id,
                        normalized_sink_node_id,
                    ],
                    "sinkNodeId": normalized_sink_node_id,
                    "sinkSegmentType": str(route.get("sinkSegmentType") or "direct"),
                    "sourceNodeId": normalized_source_node_id,
                    "sourceSegmentType": str(route.get("sourceSegmentType") or "direct"),
                    "transitNodeId": transit_node_id,
                    "transitParticipates": transit_node_id in participant_node_id_set,
                }
            )

    active_routes.sort(
        key=lambda item: (
            str(item.get("sourceNodeId") or ""),
            str(item.get("sinkNodeId") or ""),
            str(item.get("transitNodeId") or ""),
        )
    )
    return active_routes


def execution_transport_mode(
    *,
    participant_count: int,
    relay_hops_used: bool,
    coordinator_direct_fanout: bool,
    direct_socket_link_count: int,
    overlay_link_count: int,
) -> str:
    if participant_count <= 1:
        return "single_worker"
    if relay_hops_used:
        return "multi_worker_relay"
    if coordinator_direct_fanout and direct_socket_link_count > 0:
        return "multi_worker_direct"
    if overlay_link_count > 0 and direct_socket_link_count == 0:
        return "multi_worker_overlay_only"
    if direct_socket_link_count > 0:
        return "multi_worker_partial_direct"
    return "multi_worker_disconnected"


def relay_transport_note(
    *,
    relay_hops_used: bool,
    relay_bottleneck_risk: bool,
    relay_coordinator_candidate_count: int,
    relay_route_candidate_count: int,
    relay_transit_candidate_count: int,
    relay_capable_node_count: int,
) -> str:
    if relay_hops_used:
        if relay_bottleneck_risk:
            return (
                "Relay transport was used, but all active relay routes crossed the "
                "same transit node; add or enable more relay-capable peers to avoid "
                "a bottleneck."
            )
        return (
            "Coordinator-to-relay-to-worker transport was used for the current "
            "execution path."
        )
    if relay_coordinator_candidate_count > 0:
        return (
            "Coordinator-to-relay-to-worker candidate routes were visible, but "
            "relay hops were not selected for the current execution path."
        )
    if relay_route_candidate_count > 0:
        return (
            "Partial relay candidate routes were visible, but no single "
            "coordinator relay path was used for the current execution path."
        )
    if relay_transit_candidate_count > 0:
        return (
            "Relay-capable transit nodes were visible, but overlay and non-worker "
            "relay hops are not used for the current execution path yet."
        )
    if relay_capable_node_count > 0:
        return (
            "Participant nodes advertise relay capability, but relay hops are not "
            "used for the current execution path yet."
        )
    return (
        "Overlay and non-worker relay hops are not used for the current "
        "execution path yet."
    )

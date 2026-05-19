# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .cai_owned_transport_peer_urls import (
    cai_owned_transport_peer_url_route_class,
    prioritized_cai_owned_transport_peer_urls,
)


def preflight_cai_owned_transport_data_plane_routes(
    *,
    requester_node_id: str,
    executor_node_ids: Sequence[str],
    peer_cai_urls_by_node: Mapping[str, Sequence[str]],
    route_policy: Mapping[str, Any] | None = None,
    route_health_records: Sequence[Any] | None = None,
    require_route_health: bool = False,
    minimum_route_health_score: int = 2,
) -> dict[str, Any]:
    requester = str(requester_node_id or "").strip()
    executors = _clean_node_ids(executor_node_ids)
    participants = _clean_node_ids([requester, *executors])
    node_audits = [
        _preflight_cai_owned_transport_node_route(
            node_id,
            peer_cai_urls_by_node.get(node_id) or [],
        )
        for node_id in participants
    ]
    fatal_reasons = [
        str(item.get("error") or item.get("reason") or item.get("nodeId"))
        for item in node_audits
        if not bool(item.get("ready"))
    ]
    required_route_hops = _cai_owned_transport_required_route_hops(
        requester,
        executors,
        route_policy,
    )
    route_health_audits = [
        _preflight_cai_owned_transport_route_health(
            source_node_id=source_node_id,
            sink_node_id=sink_node_id,
            route_health_records=route_health_records,
            minimum_route_health_score=minimum_route_health_score,
        )
        for source_node_id, sink_node_id in required_route_hops
    ]
    if require_route_health:
        fatal_reasons.extend(
            str(item.get("error") or item.get("reason") or item.get("sinkNodeId"))
            for item in route_health_audits
            if not bool(item.get("ready"))
        )

    avoid_bottleneck = bool(
        isinstance(route_policy, Mapping)
        and route_policy.get("avoidSingleTransitBottleneck")
    )
    minimum_relay_quorum = 0
    if isinstance(route_policy, Mapping):
        minimum_relay_quorum = max(
            0,
            _optional_int(
                route_policy.get("minimumRelayQuorum")
                if route_policy.get("minimumRelayQuorum") is not None
                else route_policy.get("minimumIndependentTransitCount")
            )
            or 0,
        )
    relay_quorum_audits: list[dict[str, Any]] = []
    if avoid_bottleneck and len(executors) > 1:
        executor_route_classes = [
            str(item.get("selectedRouteClass") or "")
            for item in node_audits
            if item.get("nodeId") in set(executors)
        ]
        if executor_route_classes and all(
            route_class == "overlay_bootstrap"
            for route_class in executor_route_classes
        ):
            fatal_reasons.append(
                "all executor routes depend on bootstrap/VPS overlay relay"
            )
        transit_node_ids = [
            str(item.get("transitNodeId") or "")
            for item in route_health_audits
            if item.get("sourceNodeId") == requester
            and item.get("sinkNodeId") in set(executors)
            and str(item.get("transitNodeId") or "")
        ]
        if (
            require_route_health
            and len(transit_node_ids) == len(executors)
            and len(set(transit_node_ids)) == 1
        ):
            fatal_reasons.append(
                "all executor data-plane routes depend on one transit node"
            )
    if require_route_health and minimum_relay_quorum > 0 and len(executors) > 1:
        relay_quorum_audits = [
            _preflight_cai_owned_transport_relay_quorum(
                source_node_id=requester,
                sink_node_id=executor,
                route_health_records=route_health_records,
                minimum_relay_quorum=minimum_relay_quorum,
            )
            for executor in executors
            if executor != requester
        ]
        fatal_reasons.extend(
            str(item.get("error") or item.get("reason") or item.get("sinkNodeId"))
            for item in relay_quorum_audits
            if not bool(item.get("ready"))
        )

    return {
        "status": "ready" if not fatal_reasons else "failed",
        "requesterNodeId": requester,
        "executorNodeIds": executors,
        "participantNodeIds": participants,
        "avoidSingleTransitBottleneck": avoid_bottleneck,
        "requireRouteHealth": bool(require_route_health),
        "minimumRouteHealthScore": int(minimum_route_health_score),
        "minimumRelayQuorum": minimum_relay_quorum,
        "requiredRouteHops": [
            {"sourceNodeId": source_node_id, "sinkNodeId": sink_node_id}
            for source_node_id, sink_node_id in required_route_hops
        ],
        "fatalReasons": fatal_reasons,
        "nodeAudits": node_audits,
        "routeHealthAudits": route_health_audits,
        "relayQuorumAudits": relay_quorum_audits,
    }


def cai_owned_transport_route_health_readiness(
    *,
    source_node_id: str,
    sink_node_ids: Sequence[str],
    route_health_records: Sequence[Any] | None,
    route_policy: Mapping[str, Any] | None = None,
    minimum_route_health_score: int = 2,
) -> dict[str, Any]:
    source = str(source_node_id or "").strip()
    sinks = _clean_sink_node_ids(source, sink_node_ids)
    effective_route_policy = dict(route_policy or {})
    required_route_hops = _cai_owned_transport_required_route_hops(
        source,
        sinks,
        effective_route_policy,
    )
    route_health_audits = [
        _preflight_cai_owned_transport_route_health(
            source_node_id=hop_source,
            sink_node_id=hop_sink,
            route_health_records=route_health_records,
            minimum_route_health_score=minimum_route_health_score,
        )
        for hop_source, hop_sink in required_route_hops
    ]
    minimum_relay_quorum = max(
        0,
        _optional_int(
            effective_route_policy.get("minimumRelayQuorum")
            if effective_route_policy.get("minimumRelayQuorum") is not None
            else effective_route_policy.get("minimumIndependentTransitCount")
        )
        or 0,
    )
    relay_quorum_audits: list[dict[str, Any]] = []
    if minimum_relay_quorum > 0 and len(sinks) > 1:
        relay_quorum_audits = [
            _preflight_cai_owned_transport_relay_quorum(
                source_node_id=source,
                sink_node_id=sink,
                route_health_records=route_health_records,
                minimum_relay_quorum=minimum_relay_quorum,
            )
            for sink in sinks
            if sink != source
        ]
    fatal_reasons = [
        str(item.get("error") or item.get("reason") or item.get("sinkNodeId"))
        for item in route_health_audits
        if not bool(item.get("ready"))
    ]
    fatal_reasons.extend(
        str(item.get("error") or item.get("reason") or item.get("sinkNodeId"))
        for item in relay_quorum_audits
        if not bool(item.get("ready"))
    )
    status = "ready" if not fatal_reasons else "failed"
    if not sinks:
        status = "single_node"
    return {
        "status": status,
        "ready": status in {"ready", "single_node"},
        "sourceNodeId": source,
        "sinkNodeIds": sinks,
        "requiredRouteHops": [
            {"sourceNodeId": hop_source, "sinkNodeId": hop_sink}
            for hop_source, hop_sink in required_route_hops
        ],
        "minimumRouteHealthScore": max(2, int(minimum_route_health_score)),
        "minimumRelayQuorum": minimum_relay_quorum,
        "fatalReasons": list(dict.fromkeys(item for item in fatal_reasons if item)),
        "routeHealthAudits": route_health_audits,
        "relayQuorumAudits": relay_quorum_audits,
    }


def _preflight_cai_owned_transport_node_route(
    node_id: str,
    peer_cai_urls: Sequence[str],
) -> dict[str, Any]:
    urls = prioritized_cai_owned_transport_peer_urls(peer_cai_urls)
    if not urls:
        return {
            "nodeId": str(node_id or "").strip(),
            "ready": False,
            "reason": "missing_peer_cai_url",
            "error": "No CAI API route is available for participant.",
            "peerCaiUrls": [],
        }
    selected_url = urls[0]
    return {
        "nodeId": str(node_id or "").strip(),
        "ready": True,
        "reason": "route_available",
        "selectedPeerCaiUrl": selected_url,
        "selectedRouteClass": cai_owned_transport_peer_url_route_class(selected_url),
        "peerCaiUrls": urls,
    }


def _cai_owned_transport_required_route_hops(
    requester_node_id: str,
    executor_node_ids: Sequence[str],
    route_policy: Mapping[str, Any] | None,
) -> list[tuple[str, str]]:
    requester = str(requester_node_id or "").strip()
    executors = _clean_node_ids(executor_node_ids)
    hops: list[tuple[str, str]] = []

    def add_hop(source: Any, sink: Any) -> None:
        clean_source = str(source or "").strip()
        clean_sink = str(sink or "").strip()
        if not clean_source or not clean_sink or clean_source == clean_sink:
            return
        hop = (clean_source, clean_sink)
        if hop not in hops:
            hops.append(hop)

    for executor in executors:
        add_hop(requester, executor)

    dag = (
        route_policy.get("executionDag")
        if isinstance(route_policy, Mapping)
        and isinstance(route_policy.get("executionDag"), Mapping)
        else None
    )
    stages = dag.get("stages") if isinstance(dag, Mapping) else None
    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, Mapping):
                continue
            add_hop(stage.get("sourceNodeId"), stage.get("sinkNodeId"))
            add_hop(stage.get("sinkNodeId"), stage.get("outputToNodeId"))
        if hops:
            return hops

    for index, executor in enumerate(executors):
        next_node = executors[index + 1] if index + 1 < len(executors) else requester
        add_hop(executor, next_node)
    return hops


def _preflight_cai_owned_transport_route_health(
    *,
    source_node_id: str,
    sink_node_id: str,
    route_health_records: Sequence[Any] | None,
    minimum_route_health_score: int,
) -> dict[str, Any]:
    source = str(source_node_id or "").strip()
    sink = str(sink_node_id or "").strip()
    score = _cai_owned_transport_route_health_score_for_pair(
        source,
        sink,
        route_health_records,
    )
    selected_record = _best_cai_owned_transport_route_health_record(
        source,
        sink,
        route_health_records,
    )
    ready = score >= max(2, int(minimum_route_health_score))
    route_type = (
        str(_cai_owned_route_health_field(selected_record, "route_type") or "")
        if selected_record is not None
        else None
    )
    transit_node_id = (
        str(_cai_owned_route_health_field(selected_record, "transit_node_id") or "")
        or None
        if selected_record is not None
        else None
    )
    if ready:
        reason = "proven_data_plane_route"
        error = None
    elif not route_health_records:
        reason = "missing_route_health"
        error = "No RouteHealth records are available for required data-plane hop."
    else:
        reason = "unproven_data_plane_route"
        error = "No proven RouteHealth data-plane route is available for required hop."
    return {
        "sourceNodeId": source,
        "sinkNodeId": sink,
        "ready": ready,
        "reason": reason,
        "error": error,
        "routeHealthScore": score,
        "minimumRouteHealthScore": max(2, int(minimum_route_health_score)),
        "routeType": route_type,
        "endpointUrl": (
            _cai_owned_route_health_field(selected_record, "endpoint_url")
            if selected_record is not None
            else None
        ),
        "transitNodeId": transit_node_id,
        "latencyMs": (
            _cai_owned_route_health_field(selected_record, "latency_ms")
            if selected_record is not None
            else None
        ),
        "checkedAt": (
            _cai_owned_route_health_field(selected_record, "checked_at")
            if selected_record is not None
            else None
        ),
    }


def _cai_owned_transport_route_health_score_for_pair(
    source_node_id: str,
    sink_node_id: str,
    route_health_records: Sequence[Any] | None,
) -> int:
    if not route_health_records:
        return 1

    source = str(source_node_id or "").strip()
    sink = str(sink_node_id or "").strip()
    if not source or not sink or source == sink:
        return 1

    best_score = 1
    explicit_unhealthy = False
    for record in route_health_records:
        record_source = str(
            _cai_owned_route_health_field(record, "source_node_id") or ""
        ).strip()
        record_sink = str(
            _cai_owned_route_health_field(record, "sink_node_id") or ""
        ).strip()
        if record_source != source or record_sink != sink:
            continue

        route_type = str(
            _cai_owned_route_health_field(record, "route_type") or ""
        ).strip()
        reachable = bool(_cai_owned_route_health_field(record, "reachable"))
        if route_type in {"direct_data", "direct_socket"}:
            if reachable:
                best_score = max(best_score, 4)
            else:
                explicit_unhealthy = True
            continue
        if route_type in {"direct_api", "relay_active", "reverse_relay_available"}:
            if reachable:
                best_score = max(best_score, 3)
            else:
                explicit_unhealthy = True
            continue
        if route_type == "overlay_peer":
            if reachable:
                best_score = max(best_score, 2)
            else:
                explicit_unhealthy = True
            continue
        if route_type in {
            "llama_cpp_rpc",
            "llama_cpp_rpc_direct",
            "llama_cpp_rpc_relay",
        } and reachable:
            best_score = max(
                best_score,
                _cai_owned_route_health_score_for_route_type(route_type),
            )

    if best_score > 1:
        return best_score
    return 0 if explicit_unhealthy else 1


def _preflight_cai_owned_transport_relay_quorum(
    *,
    source_node_id: str,
    sink_node_id: str,
    route_health_records: Sequence[Any] | None,
    minimum_relay_quorum: int,
) -> dict[str, Any]:
    source = str(source_node_id or "").strip()
    sink = str(sink_node_id or "").strip()
    selected_record = _best_cai_owned_transport_route_health_record(
        source,
        sink,
        route_health_records,
    )
    selected_route_type = (
        str(_cai_owned_route_health_field(selected_record, "route_type") or "")
        if selected_record is not None
        else None
    )
    selected_transit = (
        str(_cai_owned_route_health_field(selected_record, "transit_node_id") or "")
        or None
        if selected_record is not None
        else None
    )
    direct_bypass = bool(
        selected_record is not None
        and not selected_transit
        and selected_route_type in {
            "direct_api",
            "direct_data",
            "direct_socket",
            "llama_cpp_rpc",
            "llama_cpp_rpc_direct",
        }
    )
    transit_node_ids = _cai_owned_transport_transit_candidates_for_pair(
        source,
        sink,
        route_health_records,
    )
    required = max(1, int(minimum_relay_quorum))
    ready = direct_bypass or len(transit_node_ids) >= required
    if ready and direct_bypass:
        reason = "direct_route_bypasses_relay_quorum"
        error = None
    elif ready:
        reason = "relay_quorum_satisfied"
        error = None
    else:
        reason = "relay_quorum_missing"
        error = (
            f"Need at least {required} independent transit route(s) "
            f"for '{source}' -> '{sink}'."
        )
    return {
        "sourceNodeId": source,
        "sinkNodeId": sink,
        "ready": ready,
        "reason": reason,
        "error": error,
        "minimumRelayQuorum": required,
        "transitNodeIds": transit_node_ids,
        "transitNodeCount": len(transit_node_ids),
        "directBypass": direct_bypass,
        "selectedRouteType": selected_route_type,
        "selectedTransitNodeId": selected_transit,
    }


def _cai_owned_transport_transit_candidates_for_pair(
    source_node_id: str,
    sink_node_id: str,
    route_health_records: Sequence[Any] | None,
) -> list[str]:
    if not route_health_records:
        return []
    source = str(source_node_id or "").strip()
    sink = str(sink_node_id or "").strip()
    transit_node_ids: set[str] = set()
    for record in route_health_records:
        record_source = str(
            _cai_owned_route_health_field(record, "source_node_id") or ""
        ).strip()
        record_sink = str(
            _cai_owned_route_health_field(record, "sink_node_id") or ""
        ).strip()
        route_type = str(
            _cai_owned_route_health_field(record, "route_type") or ""
        ).strip()
        reachable = bool(_cai_owned_route_health_field(record, "reachable"))
        transit_node_id = str(
            _cai_owned_route_health_field(record, "transit_node_id") or ""
        ).strip()
        if (
            not reachable
            or record_source != source
            or record_sink != sink
            or not transit_node_id
            or route_type not in {
                "relay_active",
                "reverse_relay_available",
                "llama_cpp_rpc_relay",
            }
        ):
            continue
        transit_node_ids.add(transit_node_id)
    return sorted(transit_node_ids)


def _best_cai_owned_transport_route_health_record(
    source_node_id: str,
    sink_node_id: str,
    route_health_records: Sequence[Any] | None,
) -> Any | None:
    if not route_health_records:
        return None
    source = str(source_node_id or "").strip()
    sink = str(sink_node_id or "").strip()
    best_key: tuple[int, int, str, int] | None = None
    best_record: Any | None = None
    for index, record in enumerate(route_health_records):
        record_source = str(
            _cai_owned_route_health_field(record, "source_node_id") or ""
        ).strip()
        record_sink = str(
            _cai_owned_route_health_field(record, "sink_node_id") or ""
        ).strip()
        route_type = str(
            _cai_owned_route_health_field(record, "route_type") or ""
        ).strip()
        reachable = bool(_cai_owned_route_health_field(record, "reachable"))
        if not reachable:
            continue
        same_direction = record_source == source and record_sink == sink
        score = 0
        if route_type in {
            "llama_cpp_rpc",
            "llama_cpp_rpc_direct",
            "llama_cpp_rpc_relay",
        }:
            score = (
                _cai_owned_route_health_score_for_route_type(route_type)
                if same_direction
                else 0
            )
        elif route_type in {"direct_data", "direct_socket"}:
            score = 4 if same_direction else 0
        elif route_type in {"direct_api", "relay_active", "reverse_relay_available"}:
            score = 3 if same_direction else 0
        elif route_type == "overlay_peer":
            score = 2 if same_direction else 0
        if score <= 0:
            continue
        checked_at = str(_cai_owned_route_health_field(record, "checked_at") or "")
        key = (
            score,
            _cai_owned_route_health_direct_preference(route_type),
            checked_at,
            index,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_record = record
    return best_record


def _cai_owned_route_health_score_for_route_type(route_type: str) -> int:
    clean = str(route_type or "").strip()
    if clean in {"llama_cpp_rpc", "llama_cpp_rpc_direct"}:
        return 5
    if clean == "llama_cpp_rpc_relay":
        return 3
    if clean in {"direct_data", "direct_socket"}:
        return 4
    if clean in {"direct_api", "relay_active", "reverse_relay_available"}:
        return 3
    if clean == "overlay_peer":
        return 2
    return 0


def _cai_owned_route_health_direct_preference(route_type: str) -> int:
    clean = str(route_type or "").strip()
    if clean in {
        "llama_cpp_rpc",
        "llama_cpp_rpc_direct",
        "direct_data",
        "direct_socket",
    }:
        return 2
    if clean == "direct_api":
        return 1
    return 0


def _cai_owned_route_health_field(record: Any, field_name: str) -> Any:
    if isinstance(record, Mapping):
        if field_name in record:
            return record[field_name]
        return record.get(_snake_to_camel(field_name))
    return getattr(record, field_name, None)


def _clean_sink_node_ids(source_node_id: str, sink_node_ids: Sequence[str]) -> list[str]:
    source = str(source_node_id or "").strip()
    sinks: list[str] = []
    seen: set[str] = set()
    for node_id in sink_node_ids:
        clean = str(node_id or "").strip()
        if not clean or clean == source or clean in seen:
            continue
        seen.add(clean)
        sinks.append(clean)
    return sinks


def _clean_node_ids(node_ids: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for node_id in node_ids:
        clean = str(node_id or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        cleaned.append(clean)
    return cleaned


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _snake_to_camel(value: str) -> str:
    parts = str(value).split("_")
    return parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])

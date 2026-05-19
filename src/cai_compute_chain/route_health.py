# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import os
import socket
import struct
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen

from .local_json_store import atomic_write_json_array_file, read_json_array_file
from .model import WalletPolicy
from .network_routes import relay_route_candidates
from .transport_endpoints import (
    candidate_identity_http_urls,
    format_host_for_url,
    identity_transport_endpoints,
)
from .wallet import data_root


@dataclass
class RouteHealthRecord:
    route_id: str
    source_node_id: str
    sink_node_id: str
    route_type: str
    endpoint_url: str | None
    reachable: bool
    checked_at: str
    latency_ms: float | None = None
    error: str | None = None
    transit_node_id: str | None = None
    source: str = "probe"
    consecutive_failures: int = 0


LLAMA_CPP_RPC_ROUTE_TYPES = {
    "llama_cpp_rpc",
    "llama_cpp_rpc_direct",
    "llama_cpp_rpc_relay",
}
DEFAULT_LLAMA_CPP_RPC_FAILURE_BACKOFF_SECONDS = 1800
LLAMA_CPP_RPC_PROBE_ROUTE_TYPE = "llama_cpp_rpc_probe"
LLAMA_CPP_RPC_PROTOCOL_FAILURE_REASON = "rpc_protocol_failed"
_LLAMA_CPP_RPC_CMD_HELLO = 14
_LLAMA_CPP_RPC_CONN_CAPS_SIZE = 24
_LLAMA_CPP_RPC_HELLO_RESPONSE_SIZE = 4 + _LLAMA_CPP_RPC_CONN_CAPS_SIZE
DEFAULT_LOW_LATENCY_COMPUTE_CELL_MAX_RTT_MS = 20.0
DEFAULT_WAN_RISKY_COMPUTE_CELL_MAX_RTT_MS = 60.0
_DIRECT_LLAMA_CPP_RPC_ROUTE_TYPES = {
    "llama_cpp_rpc",
    "llama_cpp_rpc_direct",
}
_RELAY_LLAMA_CPP_RPC_ROUTE_TYPES = {"llama_cpp_rpc_relay"}


def route_health_file_path(policy: WalletPolicy | None = None) -> Path:
    return data_root(policy) / "route-health.json"


def list_route_health_records(
    policy: WalletPolicy | None = None,
) -> list[RouteHealthRecord]:
    path = route_health_file_path(policy)
    if not path.exists():
        return []
    raw = read_json_array_file(path, heal_corrupt=True)
    records: list[RouteHealthRecord] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        item.setdefault("latency_ms", None)
        item.setdefault("error", None)
        item.setdefault("transit_node_id", None)
        item.setdefault("source", "probe")
        item.setdefault("consecutive_failures", 0)
        records.append(RouteHealthRecord(**item))
    records.sort(key=lambda item: (item.source_node_id, item.sink_node_id, item.route_type))
    return records


def save_route_health_records(
    records: list[RouteHealthRecord],
    policy: WalletPolicy | None = None,
) -> None:
    path = route_health_file_path(policy)
    atomic_write_json_array_file(path, [asdict(item) for item in records])


def prune_stale_route_health_records(
    *,
    max_age_seconds: int = 3600,
    policy: WalletPolicy | None = None,
) -> int:
    now = datetime.now(tz=UTC)
    kept: list[RouteHealthRecord] = []
    pruned = 0
    for item in list_route_health_records(policy):
        parsed = _parse_iso_datetime(item.checked_at)
        if parsed is None:
            kept.append(item)
            continue
        if (now - parsed).total_seconds() > max(0, int(max_age_seconds)):
            pruned += 1
            continue
        kept.append(item)
    if pruned:
        save_route_health_records(kept, policy)
    return pruned


def record_route_health(
    *,
    source_node_id: str,
    sink_node_id: str,
    route_type: str,
    reachable: bool,
    endpoint_url: str | None = None,
    latency_ms: float | None = None,
    error: str | None = None,
    transit_node_id: str | None = None,
    source: str = "probe",
    policy: WalletPolicy | None = None,
) -> RouteHealthRecord:
    route_id = deterministic_route_id(
        source_node_id=source_node_id,
        sink_node_id=sink_node_id,
        route_type=route_type,
        endpoint_url=endpoint_url,
        transit_node_id=transit_node_id,
    )
    records = list_route_health_records(policy)
    previous = next((item for item in records if item.route_id == route_id), None)
    consecutive_failures = 0
    if not reachable:
        consecutive_failures = (previous.consecutive_failures if previous else 0) + 1
    record = RouteHealthRecord(
        route_id=route_id,
        source_node_id=str(source_node_id),
        sink_node_id=str(sink_node_id),
        route_type=str(route_type),
        endpoint_url=endpoint_url,
        reachable=bool(reachable),
        checked_at=_now_iso(),
        latency_ms=latency_ms,
        error=error,
        transit_node_id=transit_node_id,
        source=source,
        consecutive_failures=consecutive_failures,
    )
    replaced = False
    for index, item in enumerate(records):
        if item.route_id == route_id:
            records[index] = record
            replaced = True
            break
    if not replaced:
        records.append(record)
    save_route_health_records(records, policy)
    return record


def record_llama_cpp_rpc_result(
    *,
    source_node_id: str,
    sink_node_id: str,
    reachable: bool,
    endpoint_url: str | None = None,
    transit_node_id: str | None = None,
    error: str | None = None,
    policy: WalletPolicy | None = None,
) -> RouteHealthRecord:
    route_type = "llama_cpp_rpc_relay" if transit_node_id else "llama_cpp_rpc_direct"
    return record_route_health(
        source_node_id=source_node_id,
        sink_node_id=sink_node_id,
        transit_node_id=transit_node_id,
        route_type=route_type,
        endpoint_url=endpoint_url,
        reachable=reachable,
        error=error,
        source="llama_cpp_rpc",
        policy=policy,
    )


def record_route_health_from_network_audit(
    network_audit: dict[str, Any],
    *,
    policy: WalletPolicy | None = None,
) -> list[RouteHealthRecord]:
    records: list[RouteHealthRecord] = []
    direct_links = network_audit.get("checkedDirectSocketLinks")
    if isinstance(direct_links, list):
        for item in direct_links:
            if not isinstance(item, dict):
                continue
            source_node_id = str(item.get("sourceNodeId") or "").strip()
            sink_node_id = str(item.get("sinkNodeId") or "").strip()
            if not source_node_id or not sink_node_id:
                continue
            records.append(
                record_route_health(
                    source_node_id=source_node_id,
                    sink_node_id=sink_node_id,
                    route_type="direct_data",
                    reachable=bool(item.get("bidirectional")),
                    source="network_audit",
                    policy=policy,
                )
            )

    relay_routes = network_audit.get("checkedRelayRoutes")
    if isinstance(relay_routes, list):
        for item in relay_routes:
            if not isinstance(item, dict):
                continue
            source_node_id = str(item.get("sourceNodeId") or "").strip()
            sink_node_id = str(item.get("sinkNodeId") or "").strip()
            transit_node_id = str(item.get("transitNodeId") or "").strip()
            if not source_node_id or not sink_node_id or not transit_node_id:
                continue
            candidate_only = bool(item.get("candidateOnly"))
            records.append(
                record_route_health(
                    source_node_id=source_node_id,
                    sink_node_id=sink_node_id,
                    transit_node_id=transit_node_id,
                    route_type="relay_candidate" if candidate_only else "relay_active",
                    reachable=not candidate_only,
                    source="network_audit",
                    policy=policy,
                )
            )
    return records


def route_health_score_for_pair(
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
    best_rpc = _best_llama_cpp_rpc_record_for_pair(
        source,
        sink,
        route_health_records,
    )
    if best_rpc is not None:
        best_rpc_type = str(_route_health_field(best_rpc, "route_type") or "")
        best_rpc_reachable = bool(_route_health_field(best_rpc, "reachable"))
        if best_rpc_reachable:
            best_score = max(
                best_score,
                _llama_cpp_rpc_route_health_score(best_rpc_type),
            )
        elif best_rpc_type in _DIRECT_LLAMA_CPP_RPC_ROUTE_TYPES:
            return 0
        else:
            explicit_unhealthy = True
    if best_score >= 5:
        return best_score

    for record in route_health_records:
        record_source = str(_route_health_field(record, "source_node_id") or "").strip()
        record_sink = str(_route_health_field(record, "sink_node_id") or "").strip()
        route_type = str(_route_health_field(record, "route_type") or "").strip()
        reachable = bool(_route_health_field(record, "reachable"))
        same_direction = record_source == source and record_sink == sink

        if route_type in {"direct_data", "direct_socket"} and same_direction:
            if reachable:
                best_score = max(best_score, 4)
            else:
                explicit_unhealthy = True
            continue
        if route_type == "direct_api" and same_direction:
            if reachable:
                best_score = max(best_score, 3)
            else:
                explicit_unhealthy = True
            continue
        if route_type == "overlay_peer" and same_direction:
            if reachable:
                best_score = max(best_score, 2)
            else:
                explicit_unhealthy = True
            continue
        if route_type == "relay_active" and same_direction:
            if reachable:
                best_score = max(best_score, 3)
            else:
                explicit_unhealthy = True
            continue
        if route_type == "reverse_relay_available" and same_direction:
            if reachable:
                best_score = max(best_score, 3)
            else:
                explicit_unhealthy = True

    if best_score > 1:
        return best_score
    return 0 if explicit_unhealthy else 1


def llama_cpp_rpc_status_for_pair(
    source_node_id: str,
    sink_node_id: str,
    route_health_records: Sequence[Any] | None,
) -> bool | None:
    record = _best_llama_cpp_rpc_record_for_pair(
        source_node_id,
        sink_node_id,
        route_health_records,
    )
    if record is None:
        return None
    return bool(_route_health_field(record, "reachable"))


def llama_cpp_compute_cell_profile_for_path(
    source_node_id: str,
    sink_node_ids: Sequence[str],
    route_health_records: Sequence[Any] | None,
    *,
    low_latency_max_ms: float | None = None,
    wan_risky_max_ms: float | None = None,
) -> dict[str, Any]:
    sinks = [
        str(node_id or "").strip()
        for node_id in sink_node_ids
        if str(node_id or "").strip() and str(node_id) != str(source_node_id)
    ]
    if not sinks:
        return {
            "profile": "single_node",
            "readyForLlamaCppRpc": True,
            "reason": "No remote shard participants.",
            "pairProfiles": [],
            "maxLatencyMs": None,
        }
    if not route_health_records:
        return {
            "profile": "unproven_sharded_cell",
            "readyForLlamaCppRpc": False,
            "reason": "No RouteHealth records are available for shard participants.",
            "pairProfiles": [
                _compute_cell_pair_profile(
                    str(source_node_id),
                    sink,
                    [],
                    low_latency_max_ms=low_latency_max_ms,
                    wan_risky_max_ms=wan_risky_max_ms,
                )
                for sink in sinks
            ],
            "maxLatencyMs": None,
        }

    pair_profiles = [
        _compute_cell_pair_profile(
            str(source_node_id),
            sink,
            route_health_records,
            low_latency_max_ms=low_latency_max_ms,
            wan_risky_max_ms=wan_risky_max_ms,
        )
        for sink in sinks
    ]
    max_latency_ms = _max_known_latency_ms(pair_profiles)
    statuses = {str(item.get("status") or "") for item in pair_profiles}
    if "failed" in statuses:
        profile = "failed_sharded_cell"
        ready = False
        reason = "At least one shard participant has a failed llama.cpp RPC proof."
    elif "wan_unsuitable" in statuses or "wan_risky" in statuses:
        profile = "wan_risky_sharded_cell"
        ready = False
        reason = (
            "Shard participants are routable, but measured RPC latency is too high "
            "for standard llama.cpp model-parallel decode."
        )
    elif "unproven" in statuses:
        profile = "unproven_sharded_cell"
        ready = False
        reason = "At least one shard participant still needs a runtime RPC proof."
    elif "proven_unknown_latency" in statuses:
        profile = "proven_unknown_latency_sharded_cell"
        ready = True
        reason = "All shard participants have RPC proof, but latency is not measured."
    else:
        profile = "low_latency_sharded_cell"
        ready = True
        reason = "All shard participants have low-latency RPC proof."

    return {
        "profile": profile,
        "readyForLlamaCppRpc": ready,
        "reason": reason,
        "pairProfiles": pair_profiles,
        "maxLatencyMs": max_latency_ms,
        "lowLatencyMaxMs": _low_latency_compute_cell_max_ms(low_latency_max_ms),
        "wanRiskyMaxMs": _wan_risky_compute_cell_max_ms(wan_risky_max_ms),
    }


def _compute_cell_pair_profile(
    source_node_id: str,
    sink_node_id: str,
    route_health_records: Sequence[Any] | None,
    *,
    low_latency_max_ms: float | None,
    wan_risky_max_ms: float | None,
) -> dict[str, Any]:
    latest = _latest_llama_cpp_rpc_record_for_pair(
        source_node_id,
        sink_node_id,
        route_health_records,
    )
    if latest is None:
        score = route_health_score_for_pair(
            source_node_id,
            sink_node_id,
            route_health_records,
        )
        return {
            "sourceNodeId": source_node_id,
            "sinkNodeId": sink_node_id,
            "status": "unproven",
            "readyForLlamaCppRpc": False,
            "routeScore": score,
            "routeType": None,
            "latencyMs": None,
            "endpointUrl": None,
            "reason": (
                "Route candidate exists but no successful runtime/probe "
                "llama.cpp RPC proof is available."
                if score > 1
                else "No usable route proof is available."
            ),
        }

    route_type = str(_route_health_field(latest, "route_type") or "").strip()
    latency_ms = _optional_float(_route_health_field(latest, "latency_ms"))
    endpoint_url = _route_health_field(latest, "endpoint_url")
    checked_at = str(_route_health_field(latest, "checked_at") or "")
    reachable = bool(_route_health_field(latest, "reachable"))
    if not reachable:
        return {
            "sourceNodeId": source_node_id,
            "sinkNodeId": sink_node_id,
            "status": "failed",
            "readyForLlamaCppRpc": False,
            "routeType": route_type,
            "latencyMs": latency_ms,
            "endpointUrl": endpoint_url,
            "checkedAt": checked_at,
            "error": _route_health_field(latest, "error"),
            "reason": "Latest llama.cpp RPC proof failed.",
        }

    if latency_ms is None:
        return {
            "sourceNodeId": source_node_id,
            "sinkNodeId": sink_node_id,
            "status": "proven_unknown_latency",
            "readyForLlamaCppRpc": True,
            "routeType": route_type,
            "latencyMs": None,
            "endpointUrl": endpoint_url,
            "checkedAt": checked_at,
            "reason": "RPC proof succeeded, but latency was not measured.",
        }

    low_latency_max = _low_latency_compute_cell_max_ms(low_latency_max_ms)
    wan_risky_max = _wan_risky_compute_cell_max_ms(wan_risky_max_ms)
    if latency_ms <= low_latency_max:
        status = "low_latency"
        ready = True
        reason = "RPC latency is within low-latency compute-cell threshold."
    elif latency_ms <= wan_risky_max:
        status = "wan_risky"
        ready = False
        reason = (
            "RPC proof succeeded, but latency is risky for standard llama.cpp "
            "model-parallel decode."
        )
    else:
        status = "wan_unsuitable"
        ready = False
        reason = (
            "RPC proof succeeded, but latency is above the WAN-risk threshold "
            "for standard llama.cpp model-parallel decode."
        )

    return {
        "sourceNodeId": source_node_id,
        "sinkNodeId": sink_node_id,
        "status": status,
        "readyForLlamaCppRpc": ready,
        "routeType": route_type,
        "latencyMs": latency_ms,
        "endpointUrl": endpoint_url,
        "checkedAt": checked_at,
        "reason": reason,
    }


def _latest_llama_cpp_rpc_record_for_pair(
    source_node_id: str,
    sink_node_id: str,
    route_health_records: Sequence[Any] | None,
) -> Any | None:
    return _best_llama_cpp_rpc_record_for_pair(
        source_node_id,
        sink_node_id,
        route_health_records,
    )


def _best_llama_cpp_rpc_record_for_pair(
    source_node_id: str,
    sink_node_id: str,
    route_health_records: Sequence[Any] | None,
) -> Any | None:
    if not route_health_records:
        return None
    source = str(source_node_id or "").strip()
    sink = str(sink_node_id or "").strip()
    if not source or not sink or source == sink:
        return None
    best_key: tuple[int, str, int] | None = None
    best_record: Any | None = None
    for index, record in enumerate(route_health_records):
        route_type = str(_route_health_field(record, "route_type") or "").strip()
        if route_type not in LLAMA_CPP_RPC_ROUTE_TYPES:
            continue
        record_source = str(_route_health_field(record, "source_node_id") or "").strip()
        record_sink = str(_route_health_field(record, "sink_node_id") or "").strip()
        if record_source != source or record_sink != sink:
            continue
        checked_at = str(_route_health_field(record, "checked_at") or "")
        if (
            not bool(_route_health_field(record, "reachable"))
            and _llama_cpp_rpc_failure_backoff_expired(checked_at)
        ):
            continue
        key = (_llama_cpp_rpc_route_preference(route_type), checked_at, index)
        if best_key is None or key > best_key:
            best_key = key
            best_record = record
    return best_record


def _llama_cpp_rpc_route_preference(route_type: str) -> int:
    clean = str(route_type or "").strip()
    if clean in _DIRECT_LLAMA_CPP_RPC_ROUTE_TYPES:
        return 2
    if clean in _RELAY_LLAMA_CPP_RPC_ROUTE_TYPES:
        return 1
    return 0


def _llama_cpp_rpc_route_health_score(route_type: str) -> int:
    clean = str(route_type or "").strip()
    if clean in _DIRECT_LLAMA_CPP_RPC_ROUTE_TYPES:
        return 5
    if clean in _RELAY_LLAMA_CPP_RPC_ROUTE_TYPES:
        return 3
    return 1


def _max_known_latency_ms(pair_profiles: Sequence[Mapping[str, Any]]) -> float | None:
    values = [
        float(item["latencyMs"])
        for item in pair_profiles
        if item.get("latencyMs") is not None
    ]
    return max(values) if values else None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _low_latency_compute_cell_max_ms(value: float | None = None) -> float:
    if value is not None:
        return max(0.0, float(value))
    try:
        return max(
            0.0,
            float(
                os.environ.get(
                    "CAI_LOW_LATENCY_COMPUTE_CELL_MAX_RTT_MS",
                    str(DEFAULT_LOW_LATENCY_COMPUTE_CELL_MAX_RTT_MS),
                )
                or DEFAULT_LOW_LATENCY_COMPUTE_CELL_MAX_RTT_MS
            ),
        )
    except ValueError:
        return DEFAULT_LOW_LATENCY_COMPUTE_CELL_MAX_RTT_MS


def _wan_risky_compute_cell_max_ms(value: float | None = None) -> float:
    if value is not None:
        return max(0.0, float(value))
    try:
        return max(
            0.0,
            float(
                os.environ.get(
                    "CAI_WAN_RISKY_COMPUTE_CELL_MAX_RTT_MS",
                    str(DEFAULT_WAN_RISKY_COMPUTE_CELL_MAX_RTT_MS),
                )
                or DEFAULT_WAN_RISKY_COMPUTE_CELL_MAX_RTT_MS
            ),
        )
    except ValueError:
        return DEFAULT_WAN_RISKY_COMPUTE_CELL_MAX_RTT_MS


def _llama_cpp_rpc_failure_backoff_expired(checked_at: str | None) -> bool:
    parsed = _parse_iso_datetime(checked_at)
    if parsed is None:
        return False
    try:
        backoff_seconds = int(
            os.environ.get(
                "CAI_LLAMA_CPP_RPC_FAILURE_BACKOFF_SECONDS",
                str(DEFAULT_LLAMA_CPP_RPC_FAILURE_BACKOFF_SECONDS),
            )
            or DEFAULT_LLAMA_CPP_RPC_FAILURE_BACKOFF_SECONDS
        )
    except ValueError:
        backoff_seconds = DEFAULT_LLAMA_CPP_RPC_FAILURE_BACKOFF_SECONDS
    if backoff_seconds <= 0:
        return False
    return (datetime.now(tz=UTC) - parsed).total_seconds() > backoff_seconds


def route_health_score_for_path(
    source_node_id: str,
    sink_node_ids: Sequence[str],
    route_health_records: Sequence[Any] | None,
) -> tuple[int, int, int]:
    if not sink_node_ids or not route_health_records:
        return (0, 0, 0)
    pair_scores = [
        route_health_score_for_pair(source_node_id, sink_node_id, route_health_records)
        for sink_node_id in sink_node_ids
        if str(sink_node_id or "").strip() and str(sink_node_id) != str(source_node_id)
    ]
    if not pair_scores:
        return (0, 0, 0)
    return (
        min(pair_scores),
        sum(pair_scores),
        -sum(1 for score in pair_scores if score == 0),
    )


def record_overlay_routes_from_state(
    *,
    state_payload: dict[str, Any],
    visible_node_ids: Sequence[str] | set[str] | None = None,
    policy: WalletPolicy | None = None,
) -> list[RouteHealthRecord]:
    overlay_peers = state_payload.get("overlayPeers") if isinstance(state_payload, dict) else None
    if not isinstance(overlay_peers, Mapping):
        return []

    visible_set = (
        {str(node_id).strip() for node_id in visible_node_ids if str(node_id).strip()}
        if visible_node_ids is not None
        else None
    )
    records: list[RouteHealthRecord] = []
    emitted: set[tuple[str, str]] = set()
    for source_node_id, peer_payload in overlay_peers.items():
        normalized_source = str(source_node_id or "").strip()
        if not normalized_source or (visible_set is not None and normalized_source not in visible_set):
            continue
        if isinstance(peer_payload, Mapping):
            peer_iterable = peer_payload.keys()
        elif isinstance(peer_payload, (list, tuple, set)):
            peer_iterable = peer_payload
        else:
            continue
        for peer_node_id in peer_iterable:
            normalized_peer = str(peer_node_id or "").strip()
            if (
                not normalized_peer
                or normalized_peer == normalized_source
                or (visible_set is not None and normalized_peer not in visible_set)
            ):
                continue
            key = (normalized_source, normalized_peer)
            if key in emitted:
                continue
            emitted.add(key)
            records.append(
                record_route_health(
                    source_node_id=normalized_source,
                    sink_node_id=normalized_peer,
                    route_type="overlay_peer",
                    reachable=True,
                    source="state_overlay",
                    policy=policy,
                )
            )
    return records


def score_relay_route_candidates(
    *,
    state_payload: dict[str, Any],
    participant_node_ids: Sequence[str] | set[str] | None = None,
    route_health_records: Sequence[Any] | None = None,
) -> dict[str, Any]:
    participants = _normalized_participant_node_ids(
        participant_node_ids
        if participant_node_ids is not None
        else _worker_node_ids_from_state(state_payload)
    )
    candidates = relay_route_candidates(
        state_payload,
        participants,
        include_alternatives=True,
    )
    records = list(route_health_records or [])
    transit_counts = Counter(
        str(route.get("transitNodeId") or "").strip()
        for route in candidates
        if str(route.get("transitNodeId") or "").strip()
    )
    scored_candidates: list[dict[str, Any]] = []
    for route in candidates:
        source_node_id = str(route.get("sourceNodeId") or "").strip()
        sink_node_id = str(route.get("sinkNodeId") or "").strip()
        transit_node_id = str(route.get("transitNodeId") or "").strip()
        source_segment_score = route_health_score_for_pair(
            source_node_id,
            transit_node_id,
            records,
        )
        sink_segment_score = route_health_score_for_pair(
            transit_node_id,
            sink_node_id,
            records,
        )
        active_route_score = route_health_score_for_pair(
            source_node_id,
            sink_node_id,
            [
                record
                for record in records
                if str(_route_health_field(record, "route_type") or "") == "relay_active"
            ],
        )
        health_score = (
            min(source_segment_score, sink_segment_score),
            source_segment_score + sink_segment_score + active_route_score,
            active_route_score,
            -int(transit_counts.get(transit_node_id, 0)),
        )
        enriched = dict(route)
        enriched.update(
            {
                "sourceSegmentHealthScore": source_segment_score,
                "sinkSegmentHealthScore": sink_segment_score,
                "activeRouteHealthScore": active_route_score,
                "relayHealthScore": list(health_score),
                "transitRouteCount": int(transit_counts.get(transit_node_id, 0)),
            }
        )
        scored_candidates.append(enriched)

    scored_candidates.sort(
        key=lambda item: (
            item.get("relayHealthScore") or [0, 0, 0, 0],
            str(item.get("sourceNodeId") or ""),
            str(item.get("sinkNodeId") or ""),
            str(item.get("transitNodeId") or ""),
        ),
        reverse=True,
    )
    bottleneck_transit_node_ids = [
        node_id
        for node_id, count in sorted(transit_counts.items())
        if node_id and count == len(candidates) and len(candidates) > 0
    ]
    return {
        "participantNodeIds": participants,
        "candidateCount": len(scored_candidates),
        "candidates": scored_candidates,
        "transitNodeCounts": dict(sorted(transit_counts.items())),
        "bottleneckRisk": bool(
            scored_candidates
            and len(transit_counts) == 1
            and len(participants) > 1
        ),
        "bottleneckTransitNodeIds": bottleneck_transit_node_ids,
    }


def record_relay_probe_result(
    *,
    source_node_id: str | None,
    sink_node_id: str | None,
    transit_node_id: str | None,
    target_host: str | None,
    target_port: int | None,
    ready: bool,
    mode: str | None,
    reverse_channels: int = 0,
    error: str | None = None,
    policy: WalletPolicy | None = None,
) -> RouteHealthRecord | None:
    normalized_source = str(source_node_id or "").strip()
    normalized_sink = str(sink_node_id or "").strip()
    normalized_transit = str(transit_node_id or "").strip()
    if not normalized_source or not normalized_sink or not normalized_transit:
        return None

    normalized_mode = str(mode or "").strip().lower()
    route_type = "relay_active"
    if ready and normalized_mode == "reverse" and int(reverse_channels) > 0:
        route_type = "reverse_relay_available"

    endpoint_url = None
    host = str(target_host or "").strip()
    port = int(target_port or 0)
    if host and port > 0:
        endpoint_url = f"relay://{normalized_transit}/{host}:{port}"

    return record_route_health(
        source_node_id=normalized_source,
        sink_node_id=normalized_sink,
        transit_node_id=normalized_transit,
        route_type=route_type,
        endpoint_url=endpoint_url,
        reachable=bool(ready),
        error=error,
        source="relay_probe",
        policy=policy,
    )


def probe_direct_api_routes(
    *,
    state_payload: dict[str, Any],
    local_node_id: str | None = None,
    timeout_sec: float = 1.0,
    policy: WalletPolicy | None = None,
) -> list[RouteHealthRecord]:
    identities = (
        state_payload.get("nodeIdentities")
        if isinstance(state_payload, dict)
        else None
    )
    if not isinstance(identities, dict):
        return []
    source_node_id = str(local_node_id or "local").strip() or "local"
    records: list[RouteHealthRecord] = []
    for node_id, identity in identities.items():
        sink_node_id = str(node_id).strip()
        if not sink_node_id or sink_node_id == source_node_id:
            continue
        urls = candidate_identity_http_urls(identity)
        if not urls:
            records.append(
                record_route_health(
                    source_node_id=source_node_id,
                    sink_node_id=sink_node_id,
                    route_type="direct_api",
                    reachable=False,
                    error="no candidate API endpoint",
                    policy=policy,
                )
            )
            continue
        records.append(
            _probe_first_reachable_http_url(
                source_node_id=source_node_id,
                sink_node_id=sink_node_id,
                route_type="direct_api",
                urls=urls,
                timeout_sec=timeout_sec,
                expected_node_id=sink_node_id,
                policy=policy,
            )
        )
    return records


def probe_direct_data_routes(
    *,
    state_payload: dict[str, Any],
    local_node_id: str | None = None,
    timeout_sec: float = 1.0,
    policy: WalletPolicy | None = None,
) -> list[RouteHealthRecord]:
    identities = (
        state_payload.get("nodeIdentities")
        if isinstance(state_payload, dict)
        else None
    )
    if not isinstance(identities, dict):
        return []
    source_node_id = str(local_node_id or "local").strip() or "local"
    records: list[RouteHealthRecord] = []
    for node_id, identity in identities.items():
        sink_node_id = str(node_id).strip()
        if not sink_node_id or sink_node_id == source_node_id:
            continue
        urls = _candidate_identity_data_urls(identity)
        if not urls:
            records.append(
                record_route_health(
                    source_node_id=source_node_id,
                    sink_node_id=sink_node_id,
                    route_type="direct_data",
                    reachable=False,
                    error="no candidate data endpoint",
                    policy=policy,
                )
            )
            continue
        records.append(
            _probe_first_reachable_url(
                source_node_id=source_node_id,
                sink_node_id=sink_node_id,
                route_type="direct_data",
                urls=urls,
                timeout_sec=timeout_sec,
                policy=policy,
            )
        )
    return records


def probe_llama_cpp_rpc_routes(
    *,
    state_payload: dict[str, Any],
    local_node_id: str | None = None,
    timeout_sec: float = 1.0,
    policy: WalletPolicy | None = None,
    max_relay_probes: int = 8,
) -> list[RouteHealthRecord]:
    identities = (
        state_payload.get("nodeIdentities")
        if isinstance(state_payload, dict)
        else None
    )
    if not isinstance(identities, dict):
        return []
    source_node_id = str(local_node_id or "local").strip() or "local"
    records: list[RouteHealthRecord] = []
    for node_id, identity in identities.items():
        sink_node_id = str(node_id).strip()
        if not sink_node_id or sink_node_id == source_node_id:
            continue
        if not _identity_worker_enabled(identity):
            continue
        urls = _candidate_identity_rpc_urls(identity)
        if not urls:
            records.append(
                record_route_health(
                    source_node_id=source_node_id,
                    sink_node_id=sink_node_id,
                    route_type=LLAMA_CPP_RPC_PROBE_ROUTE_TYPE,
                    reachable=False,
                    error="no candidate llama.cpp RPC endpoint",
                    source="llama_cpp_rpc_probe",
                    policy=policy,
                )
            )
            continue
        records.append(
            _probe_first_llama_cpp_rpc_url(
                source_node_id=source_node_id,
                sink_node_id=sink_node_id,
                urls=urls,
                timeout_sec=timeout_sec,
                policy=policy,
            )
        )
    records.extend(
        _probe_llama_cpp_rpc_relay_routes(
            state_payload=state_payload,
            local_node_id=source_node_id,
            timeout_sec=timeout_sec,
            policy=policy,
            max_relay_probes=max_relay_probes,
        )
    )
    return records


def _probe_llama_cpp_rpc_relay_routes(
    *,
    state_payload: dict[str, Any],
    local_node_id: str,
    timeout_sec: float,
    policy: WalletPolicy | None,
    max_relay_probes: int,
) -> list[RouteHealthRecord]:
    identities = (
        state_payload.get("nodeIdentities")
        if isinstance(state_payload, dict)
        else None
    )
    if not isinstance(identities, Mapping):
        return []
    source_node_id = str(local_node_id or "").strip()
    if not source_node_id:
        return []

    records: list[RouteHealthRecord] = []
    attempted = 0
    for route in relay_route_candidates(
        state_payload,
        _worker_node_ids_from_state(state_payload),
        include_alternatives=True,
    ):
        if attempted >= max(0, int(max_relay_probes)):
            break
        route_source = str(route.get("sourceNodeId") or "").strip()
        if route_source != source_node_id:
            continue
        transit_node_id = str(route.get("transitNodeId") or "").strip()
        sink_node_id = str(route.get("sinkNodeId") or "").strip()
        if not transit_node_id or not sink_node_id:
            continue

        relay_urls = candidate_identity_http_urls(
            identities.get(transit_node_id),
            endpoint_path="/v1/cai/relay/rpc/probe",
        )
        target_url = _first_identity_rpc_url_for_route(
            identities.get(sink_node_id),
            route_type=str(route.get("sinkSegmentType") or ""),
        )
        if not relay_urls or not target_url:
            continue
        attempted += 1
        records.append(
            _probe_llama_cpp_rpc_relay_url(
                source_node_id=source_node_id,
                sink_node_id=sink_node_id,
                transit_node_id=transit_node_id,
                relay_urls=relay_urls,
                target_url=target_url,
                timeout_sec=timeout_sec,
                policy=policy,
            )
        )
    return records


def _probe_llama_cpp_rpc_relay_url(
    *,
    source_node_id: str,
    sink_node_id: str,
    transit_node_id: str,
    relay_urls: list[str],
    target_url: str,
    timeout_sec: float,
    policy: WalletPolicy | None,
) -> RouteHealthRecord:
    target = urlparse(target_url)
    target_host = target.hostname
    target_port = target.port
    endpoint_url = (
        f"relay://{transit_node_id}/{format_host_for_url(target_host or '')}:"
        f"{target_port or 0}"
    )
    if not target_host or target_port is None:
        return record_route_health(
            source_node_id=source_node_id,
            sink_node_id=sink_node_id,
            transit_node_id=transit_node_id,
            route_type=LLAMA_CPP_RPC_PROBE_ROUTE_TYPE,
            endpoint_url=endpoint_url,
            reachable=False,
            error=f"{target_url}: missing host or port",
            source="llama_cpp_rpc_probe",
            policy=policy,
        )

    transport_errors: list[str] = []
    protocol_errors: list[tuple[str, str]] = []
    query = urlencode(
        {
            "source_node_id": source_node_id,
            "transit_node_id": transit_node_id,
            "sink_node_id": sink_node_id,
            "target_host": target_host,
            "target_port": int(target_port),
            "protocol": "llama_cpp_rpc",
        }
    )
    for relay_url in relay_urls:
        probe_url = f"{relay_url}?{query}"
        started = time.perf_counter()
        try:
            with urlopen(probe_url, timeout=timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("relay probe returned non-object payload")
            if payload.get("ready") and not payload.get("protocolReady"):
                protocol_errors.append(
                    (
                        probe_url,
                        f"{LLAMA_CPP_RPC_PROTOCOL_FAILURE_REASON}: {payload}",
                    )
                )
                continue
            if not payload.get("ready"):
                raise ValueError(f"relay probe returned not ready: {payload}")
            latency_ms = (time.perf_counter() - started) * 1000.0
            return record_route_health(
                source_node_id=source_node_id,
                sink_node_id=sink_node_id,
                transit_node_id=transit_node_id,
                route_type="llama_cpp_rpc_relay",
                endpoint_url=endpoint_url,
                reachable=True,
                latency_ms=round(latency_ms, 3),
                source="llama_cpp_rpc_probe",
                policy=policy,
            )
        except Exception as exc:
            transport_errors.append(f"{probe_url}: {exc}")
            continue

    if protocol_errors:
        return record_route_health(
            source_node_id=source_node_id,
            sink_node_id=sink_node_id,
            transit_node_id=transit_node_id,
            route_type="llama_cpp_rpc_relay",
            endpoint_url=endpoint_url,
            reachable=False,
            error="; ".join(f"{url}: {error}" for url, error in protocol_errors)[
                :1000
            ],
            source="llama_cpp_rpc_probe",
            policy=policy,
        )

    return record_route_health(
        source_node_id=source_node_id,
        sink_node_id=sink_node_id,
        transit_node_id=transit_node_id,
        route_type=LLAMA_CPP_RPC_PROBE_ROUTE_TYPE,
        endpoint_url=endpoint_url,
        reachable=False,
        error="; ".join(transport_errors)[:1000],
        source="llama_cpp_rpc_probe",
        policy=policy,
    )


def deterministic_route_id(
    *,
    source_node_id: str,
    sink_node_id: str,
    route_type: str,
    endpoint_url: str | None = None,
    transit_node_id: str | None = None,
) -> str:
    payload = {
        "source_node_id": str(source_node_id),
        "sink_node_id": str(sink_node_id),
        "route_type": str(route_type),
        "endpoint_url": endpoint_url,
        "transit_node_id": transit_node_id,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:32]


def _probe_first_llama_cpp_rpc_url(
    *,
    source_node_id: str,
    sink_node_id: str,
    urls: list[str],
    timeout_sec: float,
    policy: WalletPolicy | None,
) -> RouteHealthRecord:
    transport_errors: list[str] = []
    protocol_errors: list[tuple[str, str]] = []
    for url in urls:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port
        if not host or port is None:
            transport_errors.append(f"{url}: missing host or port")
            continue
        started = time.perf_counter()
        try:
            _probe_llama_cpp_rpc_hello(
                host,
                int(port),
                timeout_sec=timeout_sec,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            return record_route_health(
                source_node_id=source_node_id,
                sink_node_id=sink_node_id,
                route_type="llama_cpp_rpc_direct",
                endpoint_url=url,
                reachable=True,
                latency_ms=round(latency_ms, 3),
                error=None,
                source="llama_cpp_rpc_probe",
                policy=policy,
            )
        except OSError as exc:
            transport_errors.append(f"{url}: {exc}")
            continue
        except ValueError as exc:
            protocol_errors.append(
                (url, f"{LLAMA_CPP_RPC_PROTOCOL_FAILURE_REASON}: {exc}")
            )
            continue
    if protocol_errors:
        return record_route_health(
            source_node_id=source_node_id,
            sink_node_id=sink_node_id,
            route_type="llama_cpp_rpc_direct",
            endpoint_url=protocol_errors[0][0],
            reachable=False,
            error="; ".join(f"{url}: {error}" for url, error in protocol_errors)[
                :1000
            ],
            source="llama_cpp_rpc_probe",
            policy=policy,
        )
    return record_route_health(
        source_node_id=source_node_id,
        sink_node_id=sink_node_id,
        route_type=LLAMA_CPP_RPC_PROBE_ROUTE_TYPE,
        endpoint_url=urls[0],
        reachable=False,
        error="; ".join(transport_errors)[:1000],
        source="llama_cpp_rpc_probe",
        policy=policy,
    )


def _probe_llama_cpp_rpc_hello(
    host: str,
    port: int,
    *,
    timeout_sec: float,
) -> str:
    with socket.create_connection((host, int(port)), timeout=timeout_sec) as sock:
        sock.settimeout(timeout_sec)
        sock.sendall(bytes([_LLAMA_CPP_RPC_CMD_HELLO]))
        sock.sendall(struct.pack("<Q", _LLAMA_CPP_RPC_CONN_CAPS_SIZE))
        sock.sendall(bytes(_LLAMA_CPP_RPC_CONN_CAPS_SIZE))
        response_size = struct.unpack("<Q", _recv_exact(sock, 8))[0]
        if response_size != _LLAMA_CPP_RPC_HELLO_RESPONSE_SIZE:
            raise ValueError(f"unexpected HELLO response size {response_size}")
        response = _recv_exact(sock, response_size)
    major, minor, patch = response[0], response[1], response[2]
    if major <= 0:
        raise ValueError("invalid HELLO protocol version")

    expected_major = _optional_env_int("CAI_LLAMA_CPP_RPC_EXPECTED_MAJOR")
    if expected_major is not None and major != expected_major:
        raise ValueError(
            f"RPC protocol major mismatch: remote={major}, expected={expected_major}"
        )
    expected_minor = _optional_env_int("CAI_LLAMA_CPP_RPC_EXPECTED_MINOR")
    if (
        expected_major is not None
        and major == expected_major
        and expected_minor is not None
        and minor > expected_minor
    ):
        raise ValueError(
            f"RPC protocol minor mismatch: remote={major}.{minor}, "
            f"expected<= {expected_major}.{expected_minor}"
        )
    return f"{major}.{minor}.{patch}"


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(size)
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ValueError("connection closed during HELLO")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _probe_first_reachable_url(
    *,
    source_node_id: str,
    sink_node_id: str,
    route_type: str,
    urls: list[str],
    timeout_sec: float,
    policy: WalletPolicy | None,
) -> RouteHealthRecord:
    errors: list[str] = []
    for url in urls:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port
        if not host or port is None:
            errors.append(f"{url}: missing host or port")
            continue
        started = time.perf_counter()
        try:
            with socket.create_connection((host, int(port)), timeout=timeout_sec):
                latency_ms = (time.perf_counter() - started) * 1000.0
            return record_route_health(
                source_node_id=source_node_id,
                sink_node_id=sink_node_id,
                route_type=route_type,
                endpoint_url=url,
                reachable=True,
                latency_ms=round(latency_ms, 3),
                policy=policy,
            )
        except OSError as exc:
            errors.append(f"{url}: {exc}")
            continue
    return record_route_health(
        source_node_id=source_node_id,
        sink_node_id=sink_node_id,
        route_type=route_type,
        endpoint_url=urls[0],
        reachable=False,
        error="; ".join(errors)[:1000],
        policy=policy,
    )


def _probe_first_reachable_http_url(
    *,
    source_node_id: str,
    sink_node_id: str,
    route_type: str,
    urls: list[str],
    timeout_sec: float,
    expected_node_id: str | None,
    policy: WalletPolicy | None,
) -> RouteHealthRecord:
    errors: list[str] = []
    for url in urls:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            errors.append(f"{url}: unsupported scheme")
            continue
        if not parsed.hostname or parsed.port is None:
            errors.append(f"{url}: missing host or port")
            continue
        probe_url = f"{url.rstrip('/')}/node_id"
        started = time.perf_counter()
        try:
            with urlopen(probe_url, timeout=timeout_sec) as response:
                status_code = _http_response_status_code(response)
                if status_code is not None and status_code >= 400:
                    raise ValueError(f"HTTP status {status_code}")
                body = response.read()
            if not body:
                raise ValueError("empty HTTP response")
            observed_node_id = _decode_node_id_probe_payload(body)
            normalized_expected = str(expected_node_id or "").strip()
            if normalized_expected and observed_node_id != normalized_expected:
                raise ValueError(
                    f"node_id mismatch: expected {normalized_expected}, "
                    f"got {observed_node_id or '<empty>'}"
                )
            latency_ms = (time.perf_counter() - started) * 1000.0
            return record_route_health(
                source_node_id=source_node_id,
                sink_node_id=sink_node_id,
                route_type=route_type,
                endpoint_url=url,
                reachable=True,
                latency_ms=round(latency_ms, 3),
                policy=policy,
            )
        except (OSError, TimeoutError, ValueError) as exc:
            errors.append(f"{probe_url}: {exc}")
            continue
    return record_route_health(
        source_node_id=source_node_id,
        sink_node_id=sink_node_id,
        route_type=route_type,
        endpoint_url=urls[0],
        reachable=False,
        error="; ".join(errors)[:1000],
        policy=policy,
    )


def _http_response_status_code(response: Any) -> int | None:
    raw_status = getattr(response, "status", None)
    if raw_status is None:
        getcode = getattr(response, "getcode", None)
        if callable(getcode):
            raw_status = getcode()
    try:
        return int(raw_status)
    except (TypeError, ValueError):
        return None


def _decode_node_id_probe_payload(payload: bytes) -> str:
    text = payload.decode("utf-8", errors="replace").strip()
    if not text:
        return ""
    try:
        decoded = json.loads(text)
    except ValueError:
        return text.strip('"').strip()
    if isinstance(decoded, str):
        return decoded.strip()
    if isinstance(decoded, Mapping):
        for key in ("nodeId", "node_id", "id"):
            node_id = str(decoded.get(key) or "").strip()
            if node_id:
                return node_id
    return ""


def _candidate_identity_rpc_urls(
    identity: Mapping[str, Any] | None,
    *,
    route_type: str | None = None,
) -> list[str]:
    urls: list[str] = []
    for url in _candidate_identity_data_urls(identity, route_type=route_type):
        parsed = urlparse(url)
        if not parsed.hostname or parsed.port is None:
            continue
        urls.append(
            f"llama-cpp-rpc://{format_host_for_url(parsed.hostname)}:{parsed.port}"
        )
    return urls


def _first_identity_rpc_url_for_route(
    identity: Mapping[str, Any] | None,
    *,
    route_type: str | None,
) -> str | None:
    urls = _candidate_identity_rpc_urls(identity, route_type=route_type)
    if urls:
        return urls[0]
    return next(iter(_candidate_identity_rpc_urls(identity)), None)


def _candidate_identity_data_urls(
    identity: Mapping[str, Any] | None,
    *,
    route_type: str | None = None,
) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    route_types = [route_type] if route_type else None
    for endpoint in identity_transport_endpoints(
        identity,
        purpose="data",
        route_types=route_types,
        require_port=True,
    ):
        host = str(endpoint.get("host") or "").strip()
        port = endpoint.get("port")
        if not host or host in {"0.0.0.0", "::"} or port is None or int(port) <= 0:
            continue
        url = f"tcp://{format_host_for_url(host)}:{int(port)}"
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)

    if route_type or not isinstance(identity, Mapping):
        return urls

    host = str(identity.get("dataHost") or identity.get("data_host") or "").strip()
    port = identity.get("dataPort")
    try:
        normalized_port = int(port)
    except (TypeError, ValueError):
        normalized_port = -1
    if host and host not in {"0.0.0.0", "::"} and normalized_port > 0:
        fallback_url = f"tcp://{format_host_for_url(host)}:{normalized_port}"
        if fallback_url not in seen:
            urls.append(fallback_url)
    return urls


def _identity_worker_enabled(identity: Any) -> bool:
    if not isinstance(identity, Mapping):
        return False
    if "workerEnabled" in identity:
        return bool(identity.get("workerEnabled"))
    if "worker_enabled" in identity:
        return bool(identity.get("worker_enabled"))
    return False


def _route_health_field(record: Any, field_name: str) -> Any:
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
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _optional_env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _normalized_participant_node_ids(
    node_ids: Sequence[str] | set[str] | None,
) -> list[str]:
    return sorted(
        {
            str(node_id or "").strip()
            for node_id in (node_ids or [])
            if str(node_id or "").strip()
        }
    )


def _worker_node_ids_from_state(state_payload: dict[str, Any]) -> list[str]:
    identities = (
        state_payload.get("nodeIdentities")
        if isinstance(state_payload, dict)
        else None
    )
    if not isinstance(identities, Mapping):
        return []
    worker_node_ids: list[str] = []
    for node_id, identity in identities.items():
        if not isinstance(identity, Mapping):
            continue
        worker_enabled = identity.get("workerEnabled")
        if worker_enabled is None:
            worker_enabled = identity.get("worker_enabled")
        if bool(worker_enabled):
            normalized_node_id = str(node_id or "").strip()
            if normalized_node_id:
                worker_node_ids.append(normalized_node_id)
    return _normalized_participant_node_ids(worker_node_ids)


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()

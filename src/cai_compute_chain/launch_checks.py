# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.request import urlopen

from .network_routes import (
    relay_coordinator_candidate_node_ids,
    relay_route_candidates,
)


@dataclass(frozen=True)
class LaunchCheckResult:
    key: str
    status: str
    detail: str


@dataclass(frozen=True)
class LaunchCheckReport:
    ready: bool
    results: list[LaunchCheckResult]


def _fetch_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode('utf-8'))


def _count_connections(state_payload: dict[str, Any]) -> int:
    topology = state_payload.get('topology') or {}
    connections = topology.get('connections') or {}
    if isinstance(connections, dict):
        return len(connections)
    return 0


def _count_nodes(state_payload: dict[str, Any]) -> int:
    topology = state_payload.get('topology') or {}
    nodes = topology.get('nodes') or []
    return len(nodes)


def _count_overlay_peers(state_payload: dict[str, Any]) -> int:
    peers = state_payload.get('overlayPeers') or {}
    if isinstance(peers, dict):
        return len(peers)
    return 0


def _worker_node_ids(state_payload: dict[str, Any]) -> list[str]:
    node_identities = state_payload.get('nodeIdentities') or {}
    if not isinstance(node_identities, dict):
        return []
    worker_ids: list[str] = []
    for node_id, identity in node_identities.items():
        if not isinstance(identity, dict):
            continue
        if bool(identity.get('workerEnabled')):
            worker_ids.append(str(node_id))
    return sorted(dict.fromkeys(worker_ids))


def _relay_node_ids(state_payload: dict[str, Any]) -> list[str]:
    node_identities = state_payload.get('nodeIdentities') or {}
    if not isinstance(node_identities, dict):
        return []
    relay_ids: list[str] = []
    for node_id, identity in node_identities.items():
        if not isinstance(identity, dict):
            continue
        relay_enabled = identity.get('relayEnabled')
        if relay_enabled is None:
            relay_enabled = identity.get('relay_enabled')
        if bool(relay_enabled):
            relay_ids.append(str(node_id))
    return sorted(dict.fromkeys(relay_ids))


def _worker_socket_adjacency(
    state_payload: dict[str, Any],
    worker_node_ids: list[str],
) -> dict[str, set[str]]:
    adjacency = {node_id: set() for node_id in worker_node_ids}
    topology = state_payload.get('topology') or {}
    if not isinstance(topology, dict):
        return adjacency
    connections = topology.get('connections') or {}
    if not isinstance(connections, dict):
        return adjacency

    worker_node_id_set = set(worker_node_ids)
    for source_node_id, sink_payload in connections.items():
        normalized_source_node_id = str(source_node_id or '').strip()
        if normalized_source_node_id not in worker_node_id_set:
            continue
        if not isinstance(sink_payload, dict):
            continue
        for sink_node_id, edge_payload in sink_payload.items():
            normalized_sink_node_id = str(sink_node_id or '').strip()
            if normalized_sink_node_id not in worker_node_id_set:
                continue
            if normalized_sink_node_id == normalized_source_node_id:
                continue
            if not isinstance(edge_payload, list) or not edge_payload:
                continue
            if any(
                isinstance(edge_item, dict) and isinstance(edge_item.get('sinkMultiaddr'), dict)
                for edge_item in edge_payload
            ):
                adjacency[normalized_source_node_id].add(normalized_sink_node_id)
    return adjacency


def _worker_coordinator_candidate_count(state_payload: dict[str, Any]) -> int:
    worker_node_ids = _worker_node_ids(state_payload)
    if len(worker_node_ids) <= 1:
        return len(worker_node_ids)

    adjacency = _worker_socket_adjacency(state_payload, worker_node_ids)
    worker_node_id_set = set(worker_node_ids)
    candidates = 0
    for source_node_id in worker_node_ids:
        reachable = adjacency.get(source_node_id, set())
        if all(
            other_node_id == source_node_id or other_node_id in reachable
            for other_node_id in worker_node_id_set
        ):
            candidates += 1
    return candidates


def _validator_set_size(summary_payload: dict[str, Any]) -> int:
    validator = summary_payload.get('validator') or {}
    value = validator.get('validator_set_size')
    return int(value or 0)


def _execution_receipt_count(summary_payload: dict[str, Any]) -> int:
    compute = summary_payload.get('compute') or {}
    value = compute.get('execution_receipts')
    return int(value or 0)


def _make_result(key: str, passed: bool, detail: str) -> LaunchCheckResult:
    return LaunchCheckResult(key=key, status='PASS' if passed else 'FAIL', detail=detail)


def _make_info(key: str, detail: str) -> LaunchCheckResult:
    return LaunchCheckResult(key=key, status='INFO', detail=detail)


def run_alpha_launch_checks(
    *,
    local_state_url: str = 'http://127.0.0.1:52415/state',
    local_summary_url: str = 'http://127.0.0.1:52415/v1/cai/summary',
    remote_state_url: str | None = 'http://192.145.29.212:52415/state',
    remote_summary_url: str | None = 'http://192.145.29.212:52415/v1/cai/summary',
    expected_cluster_nodes: int = 2,
    min_overlay_peers: int = 1,
    min_connections: int = 2,
    min_validator_set_size: int = 1,
) -> LaunchCheckReport:
    results: list[LaunchCheckResult] = []
    failures = 0
    local_validator_set_size = 0
    remote_validator_set_size = 0

    try:
        local_state = _fetch_json(local_state_url)
        local_nodes = _count_nodes(local_state)
        local_overlay = _count_overlay_peers(local_state)
        local_connections = _count_connections(local_state)
        local_worker_count = len(_worker_node_ids(local_state))
        local_relay_count = len(_relay_node_ids(local_state))
        coordinator_candidates = _worker_coordinator_candidate_count(local_state)
        relay_coordinator_candidates = relay_coordinator_candidate_node_ids(
            local_state,
            _worker_node_ids(local_state),
        )
        relay_candidate_routes = relay_route_candidates(
            local_state,
            _worker_node_ids(local_state),
        )
        results.append(_make_result('local_state', True, f'{local_state_url} reachable'))
        node_ok = local_nodes >= expected_cluster_nodes
        overlay_ok = local_overlay >= min_overlay_peers
        conn_ok = local_connections >= min_connections
        worker_ok = local_worker_count >= 1
        distributed_worker_ok = local_worker_count >= 2
        distributed_direct_ok = (
            local_worker_count < 2
            or coordinator_candidates >= 1
            or len(relay_coordinator_candidates) >= 1
        )
        results.append(_make_result('local_cluster_nodes', node_ok, f'{local_nodes} node(s), expected >= {expected_cluster_nodes}'))
        results.append(_make_result('local_overlay_peers', overlay_ok, f'{local_overlay} overlay peer(s), expected >= {min_overlay_peers}'))
        results.append(_make_result('local_connections', conn_ok, f'{local_connections} connection(s), expected >= {min_connections}'))
        results.append(_make_result('local_worker_pool', worker_ok, f'{local_worker_count} worker node(s), expected >= 1'))
        results.append(_make_result('distributed_worker_pool', distributed_worker_ok, f'{local_worker_count} worker node(s), expected >= 2'))
        results.append(_make_result('distributed_direct_path', distributed_direct_ok, f'direct_coordinator_candidates={coordinator_candidates}, relay_coordinator_candidates={len(relay_coordinator_candidates)}, workers={local_worker_count}'))
        results.append(_make_info('relay_candidates', f'{local_relay_count} relay-capable node(s) visible'))
        results.append(_make_info('relay_candidate_paths', f'{len(relay_candidate_routes)} candidate relay route(s), coordinators={", ".join(relay_coordinator_candidates) if relay_coordinator_candidates else "<none>"}'))
        failures += (
            int(not node_ok)
            + int(not overlay_ok)
            + int(not conn_ok)
            + int(not worker_ok)
            + int(not distributed_worker_ok)
            + int(not distributed_direct_ok)
        )
    except Exception as exc:
        results.append(_make_result('local_state', False, f'{local_state_url} failed: {exc}'))
        failures += 1
        local_state = None

    try:
        local_summary = _fetch_json(local_summary_url)
        local_validator_set_size = _validator_set_size(local_summary)
        receipt_count = _execution_receipt_count(local_summary)
        results.append(_make_result('local_summary', True, f'{local_summary_url} reachable'))
        results.append(_make_info('local_validator_set', f'{local_validator_set_size} validator(s) visible on local summary'))
        results.append(_make_info('execution_history', f'{receipt_count} execution receipt(s) recorded locally'))
        if local_validator_set_size == 1:
            results.append(_make_info('alpha_mode', 'single-validator alpha mode is active'))
    except Exception as exc:
        results.append(_make_result('local_summary', False, f'{local_summary_url} failed: {exc}'))
        failures += 1
        local_summary = None

    if remote_state_url:
        try:
            remote_state = _fetch_json(remote_state_url)
            remote_nodes = _count_nodes(remote_state)
            remote_overlay = _count_overlay_peers(remote_state)
            remote_connections = _count_connections(remote_state)
            results.append(_make_result('remote_state', True, f'{remote_state_url} reachable'))
            remote_ok = (
                remote_nodes >= expected_cluster_nodes
                and remote_overlay >= min_overlay_peers
                and remote_connections >= min_connections
            )
            results.append(
                _make_result(
                    'remote_cluster_health',
                    remote_ok,
                    f'nodes={remote_nodes}, overlay={remote_overlay}, connections={remote_connections}',
                )
            )
            failures += int(not remote_ok)
        except Exception as exc:
            results.append(_make_result('remote_state', False, f'{remote_state_url} failed: {exc}'))
            failures += 1

    if remote_summary_url:
        try:
            remote_summary = _fetch_json(remote_summary_url)
            validator = remote_summary.get('validator') or {}
            validator_state = validator.get('validator_state') or '<unknown>'
            remote_validator_set_size = _validator_set_size(remote_summary)
            remote_validator_ok = remote_validator_set_size >= min_validator_set_size
            results.append(_make_result('remote_summary', True, f'{remote_summary_url} reachable'))
            results.append(
                _make_result(
                    'remote_validator_set',
                    remote_validator_ok,
                    f'{remote_validator_set_size} validator(s), expected >= {min_validator_set_size}',
                )
            )
            results.append(
                _make_info(
                    'remote_validator_state',
                    f'state={validator_state}, validator_set={remote_validator_set_size}',
                )
            )
            failures += int(not remote_validator_ok)
        except Exception as exc:
            results.append(_make_result('remote_summary', False, f'{remote_summary_url} failed: {exc}'))
            failures += 1

    effective_validator_set_size = max(local_validator_set_size, remote_validator_set_size)
    validator_path_ok = effective_validator_set_size >= min_validator_set_size
    if remote_validator_set_size >= min_validator_set_size:
        validator_path_detail = (
            f'remote validator path ready ({remote_validator_set_size} validator(s))'
        )
    elif local_validator_set_size >= min_validator_set_size:
        validator_path_detail = (
            f'local validator path ready ({local_validator_set_size} validator(s))'
        )
    else:
        validator_path_detail = (
            f'no validator path ready; visible validator set size is {effective_validator_set_size}'
        )
    results.append(_make_result('settlement_validator_path', validator_path_ok, validator_path_detail))
    failures += int(not validator_path_ok)

    return LaunchCheckReport(ready=(failures == 0), results=results)


def render_alpha_launch_report(report: LaunchCheckReport) -> str:
    lines = ['Alpha launch check']
    for result in report.results:
        lines.append(f'[{result.status}] {result.key}: {result.detail}')
    lines.append('')
    lines.append(f'Ready: {"yes" if report.ready else "no"}')
    return "\n".join(lines)

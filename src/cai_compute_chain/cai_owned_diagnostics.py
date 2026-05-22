# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .cai_owned_runtime import load_cai_owned_llm_shard_self_test_result
from .decentralized_compute import (
    CaiOwnedTransportSessionRecord,
    list_cai_owned_transport_batch_inbox,
    list_cai_owned_transport_sessions,
)
from .model import WalletPolicy
from .node_capabilities import NodeCapabilityRecord, list_node_capabilities
from .route_health import (
    RouteHealthRecord,
    list_route_health_records,
    route_health_score_for_pair,
)
from .wallet import data_root, get_active_wallet, load_session


CAI_OWNED_DIAGNOSTICS_SCHEMA_VERSION = 1
DEFAULT_CAI_OWNED_DIAGNOSTICS_MAX_RECORDS = 50

_SENSITIVE_KEY_PARTS = (
    "auth",
    "encrypted",
    "nonce",
    "password",
    "private",
    "secret",
    "seed",
    "salt",
    "token",
)
_MAX_NESTED_RECORDS = 20
_MAX_STRING_LENGTH = 4096


def build_cai_owned_diagnostics_snapshot(
    *,
    local_node_id: str | None = None,
    model_id: str | None = None,
    max_records: int | None = None,
    policy: WalletPolicy | None = None,
) -> dict[str, Any]:
    """Build a read-only, secret-safe snapshot for CAI-owned network debugging."""

    active_policy = policy or WalletPolicy()
    limit = _coerce_max_records(max_records)
    sessions = list_cai_owned_transport_sessions(active_policy)
    route_health = list_route_health_records(active_policy)
    node_capabilities = list_node_capabilities(active_policy)
    llm_self_test = load_cai_owned_llm_shard_self_test_result(policy=active_policy)
    local_node = str(local_node_id or "").strip() or None
    worker_runtime_queue = build_cai_owned_worker_runtime_queue_snapshot(
        local_node_id=local_node,
        max_records=limit,
        policy=active_policy,
    )
    inbox_records = worker_runtime_queue["records"]

    batch_status_counts = Counter(
        _batch_status(batch)
        for session in sessions
        for batch in session.batch_records
        if isinstance(batch, dict)
    )
    route_type_counts = Counter(record.route_type for record in route_health)
    reachable_route_count = sum(1 for record in route_health if record.reachable)
    worker_count = sum(1 for record in node_capabilities if record.worker_enabled)
    relay_count = sum(1 for record in node_capabilities if record.relay_enabled)
    validator_count = sum(1 for record in node_capabilities if record.validator_enabled)
    runtime_ready_count = sum(
        1 for record in node_capabilities if _node_runtime_ready(record)
    )
    llm_contract_ready_count = sum(
        1 for record in node_capabilities if _node_llm_contract_ready(record)
    )
    llm_production_ready_count = sum(
        1 for record in node_capabilities if _node_llm_production_ready(record)
    )

    return {
        "schemaVersion": CAI_OWNED_DIAGNOSTICS_SCHEMA_VERSION,
        "generatedAt": datetime.now(tz=UTC).isoformat(),
        "chainNetwork": active_policy.chain_network.value,
        "walletDataDirname": active_policy.wallet_data_dirname,
        "dataRoot": str(data_root(active_policy)),
        "wallet": _wallet_summary(active_policy),
        "summary": {
            "sessionCount": len(sessions),
            "activeSessionCount": sum(
                1
                for session in sessions
                if session.status not in {"completed", "failed"}
            ),
            "batchRecordCount": sum(
                len(session.batch_records) for session in sessions
            ),
            "batchStatusCounts": dict(sorted(batch_status_counts.items())),
            "batchInboxCount": int(worker_runtime_queue.get("recordCount") or 0),
            "batchInboxStatusCounts": dict(
                sorted(worker_runtime_queue["statusCounts"].items())
            ),
            "routeHealthCount": len(route_health),
            "reachableRouteCount": reachable_route_count,
            "routeTypeCounts": dict(sorted(route_type_counts.items())),
            "nodeCapabilityCount": len(node_capabilities),
            "workerCount": worker_count,
            "relayCount": relay_count,
            "validatorCount": validator_count,
            "runtimeReadyNodeCount": runtime_ready_count,
            "llmContractReadyNodeCount": llm_contract_ready_count,
            "llmProductionReadyNodeCount": llm_production_ready_count,
        },
        "caiOwnedTransport": {
            "localNodeId": local_node,
            "sessions": [_session_summary(item, limit) for item in sessions[:limit]],
            "recentBatches": _recent_batch_summaries(sessions, limit),
            "batchInbox": [
                _batch_inbox_summary(item) for item in inbox_records[:limit]
            ],
            "workerRuntimeQueue": worker_runtime_queue,
        },
        "routeHealth": {
            "records": [
                _route_health_summary(item) for item in route_health[:limit]
            ],
        },
        "nodeCapabilities": {
            "records": [
                _node_capability_summary(item) for item in node_capabilities[:limit]
            ],
        },
        "distributedInference": build_distributed_inference_diagnostics(
            local_node_id=local_node,
            model_id=model_id,
            node_capabilities=node_capabilities,
            route_health_records=route_health,
            max_records=limit,
        ),
        "llmShardSelfTest": _llm_self_test_summary(llm_self_test),
    }


def build_distributed_inference_diagnostics(
    *,
    local_node_id: str | None,
    model_id: str | None = None,
    node_capabilities: Sequence[NodeCapabilityRecord] | None = None,
    route_health_records: Sequence[RouteHealthRecord] | None = None,
    max_records: int | None = None,
    policy: WalletPolicy | None = None,
) -> dict[str, Any]:
    active_policy = policy or WalletPolicy()
    limit = _coerce_max_records(max_records)
    capabilities = list(node_capabilities or list_node_capabilities(active_policy))
    routes = list(route_health_records or list_route_health_records(active_policy))
    local_node = str(local_node_id or "").strip() or None
    requested_model = str(model_id or "").strip() or None
    worker_records = [item for item in capabilities if item.worker_enabled is True]
    executor_audits = [
        _distributed_executor_audit(
            item,
            local_node_id=local_node,
            model_id=requested_model,
            route_health_records=routes,
        )
        for item in worker_records
    ]
    ready_executors = [
        item for item in executor_audits if bool(item.get("readyForDistributedInference"))
    ]
    blockers = sorted(
        {
            str(reason)
            for item in executor_audits
            for reason in item.get("blockingReasons", [])
            if str(reason or "").strip()
        }
    )
    ready_route_classes = sorted(
        {
            str(item.get("routeClass") or "")
            for item in ready_executors
            if str(item.get("routeClass") or "").strip()
        }
    )
    status = "ready" if ready_executors else "blocked"
    if ready_executors and blockers:
        status = "partial"

    return {
        "schemaVersion": 1,
        "status": status,
        "localNodeId": local_node,
        "modelId": requested_model,
        "workerCount": len(worker_records),
        "readyExecutorCount": len(ready_executors),
        "runtimeReadyExecutorCount": sum(
            1 for item in executor_audits if bool(item.get("runtimeReady"))
        ),
        "modelReadyExecutorCount": sum(
            1 for item in executor_audits if bool(item.get("modelReady"))
        ),
        "routeReadyExecutorCount": sum(
            1 for item in executor_audits if bool(item.get("routeReady"))
        ),
        "directRouteReadyExecutorCount": sum(
            1
            for item in executor_audits
            if item.get("routeClass") == "direct" and bool(item.get("routeReady"))
        ),
        "relayRouteReadyExecutorCount": sum(
            1
            for item in executor_audits
            if item.get("routeClass") == "relay" and bool(item.get("routeReady"))
        ),
        "readyRouteClasses": ready_route_classes,
        "blockingReasons": blockers,
        "executors": executor_audits[:limit],
    }


def build_cai_owned_worker_runtime_queue_snapshot(
    *,
    local_node_id: str | None,
    max_records: int | None = None,
    policy: WalletPolicy | None = None,
) -> dict[str, Any]:
    active_policy = policy or WalletPolicy()
    limit = _coerce_max_records(max_records)
    local_node = str(local_node_id or "").strip() or None
    if not local_node:
        return {
            "localNodeId": None,
            "ready": False,
            "reason": "Local node id is required for worker runtime queue.",
            "statusCounts": {},
            "recordCount": 0,
            "receivedCount": 0,
            "processingCount": 0,
            "processedCount": 0,
            "failedCount": 0,
            "timedOutCount": 0,
            "deliveredCount": 0,
            "currentBatch": None,
            "lastError": None,
            "records": [],
        }

    inbox_records = list_cai_owned_transport_batch_inbox(
        local_node,
        status=None,
        policy=active_policy,
    )
    summarized_records = [_batch_inbox_summary(item) for item in inbox_records]
    summarized_records.sort(
        key=lambda item: str(
            item["batch"].get("updatedAt") or item["batch"].get("createdAt") or ""
        ),
        reverse=True,
    )
    status_counts = Counter(
        _batch_status(item["batch"])
        for item in summarized_records
        if isinstance(item.get("batch"), Mapping)
    )
    processing_records = [
        item for item in summarized_records if _batch_status(item["batch"]) == "processing"
    ]
    error_records = [
        item
        for item in summarized_records
        if _batch_error_text(item["batch"]) is not None
    ]
    last_error = _batch_error_text(error_records[0]["batch"]) if error_records else None

    return {
        "localNodeId": local_node,
        "ready": True,
        "reason": None,
        "statusCounts": dict(sorted(status_counts.items())),
        "recordCount": len(inbox_records),
        "receivedCount": int(status_counts.get("received", 0)),
        "processingCount": int(status_counts.get("processing", 0)),
        "processedCount": int(status_counts.get("processed", 0)),
        "failedCount": int(status_counts.get("failed", 0)),
        "timedOutCount": int(status_counts.get("timed_out", 0)),
        "deliveredCount": int(status_counts.get("delivered", 0)),
        "currentBatch": processing_records[0] if processing_records else None,
        "lastError": last_error,
        "records": summarized_records[:limit],
    }


def _distributed_executor_audit(
    record: NodeCapabilityRecord,
    *,
    local_node_id: str | None,
    model_id: str | None,
    route_health_records: Sequence[RouteHealthRecord],
) -> dict[str, Any]:
    node_id = str(record.node_id or "").strip()
    route = _distributed_route_audit(
        local_node_id,
        node_id,
        route_health_records,
    )
    model = _distributed_model_audit(record, model_id)
    runtime_ready = _node_runtime_ready(record)
    contract_ready = _node_llm_contract_ready(record)
    production_ready = _node_llm_production_ready(record)
    blocking_reasons: list[str] = []
    if record.worker_enabled is not True:
        blocking_reasons.append("worker_disabled")
    if not bool(route["ready"]):
        blocking_reasons.append(str(route["reason"]))
    if not runtime_ready:
        blocking_reasons.append("cai_owned_transport_not_runtime_ready")
    if not bool(model["ready"]):
        blocking_reasons.append(str(model["reason"]))

    return {
        "nodeId": node_id,
        "friendlyName": record.friendly_name,
        "readyForDistributedInference": not blocking_reasons,
        "blockingReasons": blocking_reasons,
        "workerEnabled": record.worker_enabled is True,
        "runtimeReady": runtime_ready,
        "contractReady": contract_ready,
        "productionReady": production_ready,
        "modelReady": bool(model["ready"]),
        "modelReason": model["reason"],
        "routeReady": bool(route["ready"]),
        "routeClass": route["routeClass"],
        "routeReason": route["reason"],
        "routeHealthScore": route["score"],
        "selectedRoute": route["selectedRoute"],
        "workerAllowedModelIds": list(record.worker_allowed_model_ids),
        "modelIds": list(record.model_ids),
    }


def _distributed_route_audit(
    local_node_id: str | None,
    node_id: str,
    route_health_records: Sequence[RouteHealthRecord],
) -> dict[str, Any]:
    if not node_id:
        return _route_audit_payload(False, "unknown", "node_id_missing", 0, None)
    if not local_node_id:
        return _route_audit_payload(
            False,
            "unknown",
            "local_node_id_missing",
            0,
            None,
        )
    if node_id == local_node_id:
        return _route_audit_payload(True, "local", "local_executor", 5, None)

    selected = _best_distributed_route_record(
        local_node_id,
        node_id,
        route_health_records,
    )
    score = route_health_score_for_pair(
        local_node_id,
        node_id,
        route_health_records,
    )
    if selected is None:
        return _route_audit_payload(
            False,
            "unknown",
            "route_health_missing",
            score,
            None,
        )
    route_class = _distributed_route_class(selected.route_type)
    if selected.reachable and route_class in {"direct", "relay"}:
        return _route_audit_payload(
            True,
            route_class,
            f"{route_class}_route_ready",
            score,
            selected,
        )
    return _route_audit_payload(
        False,
        route_class,
        "route_unreachable",
        score,
        selected,
    )


def _route_audit_payload(
    ready: bool,
    route_class: str,
    reason: str,
    score: int,
    record: RouteHealthRecord | None,
) -> dict[str, Any]:
    return {
        "ready": ready,
        "routeClass": route_class,
        "reason": reason,
        "score": score,
        "selectedRoute": _route_health_summary(record) if record else None,
    }


def _best_distributed_route_record(
    source_node_id: str,
    sink_node_id: str,
    route_health_records: Sequence[RouteHealthRecord],
) -> RouteHealthRecord | None:
    candidates = [
        item
        for item in route_health_records
        if item.source_node_id == source_node_id and item.sink_node_id == sink_node_id
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            1 if item.reachable else 0,
            _distributed_route_priority(item.route_type),
            str(item.checked_at or ""),
        ),
        reverse=True,
    )
    return candidates[0]


def _distributed_route_priority(route_type: str) -> int:
    if route_type in {"direct_data", "direct_socket", "llama_cpp_rpc_direct"}:
        return 4
    if route_type in {"relay_active", "reverse_relay_available"}:
        return 3
    if route_type in {"direct_api", "overlay_peer"}:
        return 2
    if route_type == "relay_candidate":
        return 1
    return 0


def _distributed_route_class(route_type: str) -> str:
    if route_type in {"direct_data", "direct_socket", "llama_cpp_rpc_direct"}:
        return "direct"
    if route_type in {"relay_active", "reverse_relay_available", "relay_candidate"}:
        return "relay"
    if route_type in {"direct_api", "overlay_peer"}:
        return "overlay"
    return "unknown"


def _distributed_model_audit(
    record: NodeCapabilityRecord,
    model_id: str | None,
) -> dict[str, Any]:
    if not model_id:
        return {"ready": True, "reason": "model_not_requested"}
    allowed_model_ids = list(record.worker_allowed_model_ids or [])
    if allowed_model_ids and not _model_id_in_values(model_id, allowed_model_ids):
        return {"ready": False, "reason": "model_not_allowed"}
    if _model_id_in_values(model_id, record.model_ids):
        return {"ready": True, "reason": "model_advertised"}
    if _model_id_in_readiness(model_id, record.readiness):
        return {"ready": True, "reason": "model_readiness_advertised"}
    if allowed_model_ids:
        return {"ready": True, "reason": "model_allowed"}
    return {"ready": False, "reason": "model_not_advertised"}


def _model_id_in_readiness(model_id: str, readiness: Mapping[str, Any]) -> bool:
    for field_name in (
        "models",
        "modelReadiness",
        "model_readiness",
        "modelShardInventory",
        "model_shard_inventory",
    ):
        value = readiness.get(field_name)
        if isinstance(value, Mapping) and _model_id_in_values(model_id, value.keys()):
            return True
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                if isinstance(item, Mapping) and _model_id_in_values(
                    model_id,
                    [
                        item.get("modelId"),
                        item.get("model_id"),
                        item.get("id"),
                        item.get("name"),
                    ],
                ):
                    return True

    cai_owned = readiness.get("caiOwnedTransport")
    if isinstance(cai_owned, Mapping):
        return _model_id_in_readiness(model_id, cai_owned)
    return False


def _model_id_in_values(model_id: str, values: Iterable[Any]) -> bool:
    expected = str(model_id or "").strip().lower()
    return any(str(item or "").strip().lower() == expected for item in values)


def _wallet_summary(policy: WalletPolicy) -> dict[str, Any]:
    session = load_session(policy)
    active_wallet = get_active_wallet(policy)
    return {
        "activeWalletId": active_wallet.wallet_id if active_wallet else None,
        "activeWalletName": active_wallet.name if active_wallet else None,
        "activeWalletAddress": active_wallet.address if active_wallet else None,
        "unlocked": bool(
            active_wallet and session.unlocked_wallet_id == active_wallet.wallet_id
        ),
        "unlockedWalletId": session.unlocked_wallet_id,
        "unlockedAt": session.unlocked_at,
    }


def _session_summary(
    record: CaiOwnedTransportSessionRecord,
    max_records: int,
) -> dict[str, Any]:
    batch_status_counts = Counter(
        _batch_status(batch)
        for batch in record.batch_records
        if isinstance(batch, dict)
    )
    return {
        "sessionId": record.session_id,
        "instanceId": record.instance_id,
        "modelId": record.model_id,
        "taskId": record.task_id,
        "sourceNodeId": record.source_node_id,
        "chainId": record.chain_id,
        "participantNodeIds": list(record.participant_node_ids),
        "executorNodeIds": list(record.executor_node_ids or record.participant_node_ids),
        "executionMode": record.execution_mode,
        "status": record.status,
        "createdAt": record.created_at,
        "updatedAt": record.updated_at,
        "completedAt": record.completed_at,
        "lastError": record.last_error,
        "dispatchRecordCount": len(record.dispatch_records),
        "batchRecordCount": len(record.batch_records),
        "batchStatusCounts": dict(sorted(batch_status_counts.items())),
        "shardReceiptCount": len(record.shard_receipts),
        "proofPresent": bool(record.proof),
        "batchRecords": [
            _batch_summary(item)
            for item in record.batch_records[:max_records]
            if isinstance(item, dict)
        ],
    }


def _recent_batch_summaries(
    sessions: Sequence[CaiOwnedTransportSessionRecord],
    max_records: int,
) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    for session in sessions:
        for batch in session.batch_records:
            if not isinstance(batch, dict):
                continue
            item = {
                "sessionId": session.session_id,
                "instanceId": session.instance_id,
                "modelId": session.model_id,
                "taskId": session.task_id,
                "batch": _batch_summary(batch),
            }
            batches.append(item)
    batches.sort(
        key=lambda item: str(
            item["batch"].get("updatedAt") or item["batch"].get("createdAt") or ""
        ),
        reverse=True,
    )
    return batches[:max_records]


def _batch_inbox_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    batch = record.get("batch")
    return {
        "sessionId": _safe_text(record.get("sessionId")),
        "instanceId": _safe_text(record.get("instanceId")),
        "modelId": _safe_text(record.get("modelId")),
        "taskId": _safe_text(record.get("taskId")),
        "sourceNodeId": _safe_text(record.get("sourceNodeId")),
        "batch": _batch_summary(batch if isinstance(batch, dict) else {}),
    }


def _batch_summary(batch: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "batchId",
        "chainId",
        "status",
        "phase",
        "sourceNodeId",
        "sinkNodeId",
        "payloadSizeBytes",
        "payloadSha256Hex",
        "outputPayloadSizeBytes",
        "outputPayloadSha256Hex",
        "inputPayloadSha256Hex",
        "previousBatchId",
        "hashChainSha256Hex",
        "createdAt",
        "updatedAt",
        "claimedAt",
        "startedAt",
        "processedAt",
        "heartbeatAt",
        "leaseExpiresAt",
        "leaseSeconds",
        "runtimeId",
        "claimedByNodeId",
        "attemptCount",
        "maxAttempts",
        "retryable",
        "retryScheduledAt",
        "timeoutReason",
        "lastError",
        "error",
    ):
        if key in batch:
            summary[key] = _safe_json(batch.get(key))
    for key in ("metadata", "metrics", "routeAudit", "runtimeAudit"):
        value = batch.get(key)
        if isinstance(value, Mapping):
            summary[key] = _safe_json(value)
    return summary


def _route_health_summary(record: RouteHealthRecord) -> dict[str, Any]:
    return {
        "routeId": record.route_id,
        "sourceNodeId": record.source_node_id,
        "sinkNodeId": record.sink_node_id,
        "routeType": record.route_type,
        "endpointUrl": _safe_url(record.endpoint_url),
        "reachable": record.reachable,
        "checkedAt": record.checked_at,
        "latencyMs": record.latency_ms,
        "error": record.error,
        "transitNodeId": record.transit_node_id,
        "source": record.source,
        "consecutiveFailures": record.consecutive_failures,
    }


def _node_capability_summary(record: NodeCapabilityRecord) -> dict[str, Any]:
    return {
        "nodeId": record.node_id,
        "source": record.source,
        "sourceUrl": _safe_url(record.source_url),
        "lastSeenAt": record.last_seen_at,
        "updatedAt": record.updated_at,
        "friendlyName": record.friendly_name,
        "nodePublicKeyAddress": record.node_public_key_address,
        "apiUrls": [_safe_url(item) for item in record.api_urls],
        "dataEndpoints": _safe_json(record.data_endpoints),
        "workerEnabled": record.worker_enabled,
        "relayEnabled": record.relay_enabled,
        "validatorEnabled": record.validator_enabled,
        "validatorId": record.validator_id,
        "validatorState": record.validator_state,
        "workerRewardAddress": record.worker_reward_address,
        "workerAllowedModelIds": list(record.worker_allowed_model_ids),
        "modelIds": list(record.model_ids),
        "resourceSummary": _safe_json(record.resource_summary),
        "readiness": _safe_json(record.readiness),
        "routeHints": _safe_json(record.route_hints),
    }


def _llm_self_test_summary(result: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        return {"status": "missing", "cached": False}
    return {
        "status": _safe_text(result.get("status")) or "unknown",
        "cached": True,
        "contractReady": bool(result.get("contractReady")),
        "productionReady": bool(result.get("productionReady")),
        "generationProbeReady": bool(result.get("generationProbeReady")),
        "adapterId": _safe_text(result.get("adapterId")),
        "adapterVersion": _safe_text(result.get("adapterVersion")),
        "runtimeVersion": _safe_text(result.get("runtimeVersion")),
        "productionReadinessError": _safe_text(
            result.get("productionReadinessError"),
        ),
        "productionReadinessChecks": _safe_json(
            result.get("productionReadinessChecks"),
        ),
        "generationProbe": _safe_json(result.get("generationProbe")),
        "backendHealthReady": (
            None
            if "backendHealthReady" not in result
            else (
                None
                if result.get("backendHealthReady") is None
                else bool(result.get("backendHealthReady"))
            )
        ),
        "backendHealth": _safe_json(result.get("backendHealth")),
        "outputFrameMetadataReady": bool(result.get("outputFrameMetadataReady")),
        "finalDecodeOutputReady": bool(result.get("finalDecodeOutputReady")),
        "error": _safe_text(result.get("error")),
        "savedAt": _safe_text(result.get("savedAt")),
        "testedAt": _safe_text(result.get("testedAt")),
        "recordedAt": _safe_text(result.get("recordedAt")),
        "patchBoundary": _safe_json(result.get("patchBoundary")),
    }


def _node_runtime_ready(record: NodeCapabilityRecord) -> bool:
    readiness = record.readiness.get("caiOwnedTransport")
    if not isinstance(readiness, Mapping) or not readiness.get("runtimeReady"):
        return False
    runtime_proof = readiness.get("runtimeReadyProof")
    if not (
        isinstance(runtime_proof, Mapping)
        and runtime_proof.get("verified") is True
    ):
        return False
    self_test = readiness.get("llmShardSelfTest")
    if not (
        isinstance(self_test, Mapping)
        and self_test.get("productionReady")
        and self_test.get("generationProbeReady") is True
    ):
        return False
    if isinstance(self_test, Mapping) and self_test.get("backendHealthReady") is False:
        return False
    return True


def _node_llm_contract_ready(record: NodeCapabilityRecord) -> bool:
    readiness = record.readiness.get("caiOwnedTransport")
    if not isinstance(readiness, Mapping):
        return False
    self_test = readiness.get("llmShardSelfTest")
    return bool(
        isinstance(self_test, Mapping)
        and self_test.get("contractReady")
        and self_test.get("backendHealthReady") is not False
    )


def _node_llm_production_ready(record: NodeCapabilityRecord) -> bool:
    readiness = record.readiness.get("caiOwnedTransport")
    if not isinstance(readiness, Mapping):
        return False
    self_test = readiness.get("llmShardSelfTest")
    return bool(
        isinstance(self_test, Mapping)
        and self_test.get("productionReady")
        and self_test.get("generationProbeReady") is True
        and self_test.get("backendHealthReady") is not False
    )


def _batch_status(batch: Mapping[str, Any]) -> str:
    return str(batch.get("status") or "recorded").strip() or "recorded"


def _batch_error_text(batch: Mapping[str, Any]) -> str | None:
    for key in ("lastError", "error", "timeoutReason"):
        value = batch.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _safe_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            clean_key = str(key)
            if _is_sensitive_key(clean_key):
                continue
            safe[clean_key] = _safe_json(item)
        return safe
    if isinstance(value, (list, tuple, set)):
        return [_safe_json(item) for item in list(value)[:_MAX_NESTED_RECORDS]]
    if isinstance(value, str):
        return _safe_url(value) if _looks_like_url(value) else _safe_text(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _safe_text(value)


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) <= _MAX_STRING_LENGTH:
        return text
    return text[:_MAX_STRING_LENGTH] + "...<truncated>"


def _safe_url(value: Any) -> str | None:
    text = _safe_text(value)
    if not text:
        return text
    try:
        parsed = urlsplit(text)
    except ValueError:
        return text.split("?", 1)[0].split("#", 1)[0]
    if not parsed.scheme or not parsed.netloc:
        return text.split("?", 1)[0].split("#", 1)[0]
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port is not None:
        netloc = f"{netloc}:{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _looks_like_url(value: str) -> bool:
    lowered = value.lower()
    return lowered.startswith(("http://", "https://", "ws://", "wss://"))


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in _SENSITIVE_KEY_PARTS)


def _coerce_max_records(value: int | None) -> int:
    if value is None:
        return DEFAULT_CAI_OWNED_DIAGNOSTICS_MAX_RECORDS
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return DEFAULT_CAI_OWNED_DIAGNOSTICS_MAX_RECORDS
    return max(1, min(500, parsed))

# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.cai_owned_diagnostics import (  # noqa: E402
    build_cai_owned_diagnostics_snapshot,
    build_distributed_inference_diagnostics,
    build_cai_owned_worker_runtime_queue_snapshot,
)
from cai_compute_chain.cai_owned_runtime import (  # noqa: E402
    save_cai_owned_llm_shard_self_test_result,
)
from cai_compute_chain.cli import handle_cai_owned_diagnostics  # noqa: E402
from cai_compute_chain.decentralized_compute import (  # noqa: E402
    create_cai_owned_transport_session,
    mark_cai_owned_transport_batch_status,
    record_cai_owned_transport_batch,
)
from cai_compute_chain.model import WalletPolicy  # noqa: E402
from cai_compute_chain.node_capabilities import (  # noqa: E402
    NodeCapabilityRecord,
    save_node_capabilities,
)
from cai_compute_chain.route_health import record_route_health  # noqa: E402
from cai_compute_chain.route_health import RouteHealthRecord  # noqa: E402
from cai_compute_chain.wallet import create_wallet, unlock_wallet  # noqa: E402


def test_cai_owned_diagnostics_snapshot_exports_secret_safe_network_state() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-diagnostics")
        wallet = create_wallet(
            "diagnostics",
            "pass",
            select=True,
            wallet_policy=policy,
        )
        unlock_wallet("pass", wallet_policy=policy)
        session = create_cai_owned_transport_session(
            instance_id="instance-diag",
            session_id="caiot_diag",
            participant_node_ids=["node-a", "node-b"],
            executor_node_ids=["node-b"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-diag",
            source_node_id="node-a",
            policy=policy,
        )
        record_cai_owned_transport_batch(
            session.session_id,
            batch_id="caibatch_diag",
            phase="prefill_activation_batches",
            source_node_id="node-a",
            sink_node_id="node-b",
            payload_size_bytes=128,
            payload_sha256_hex="aa" * 32,
            metadata={
                "sequence": 1,
                "endpointUrl": "http://user:pass@127.0.0.1:52415/batch?token=abc",
                "secretToken": "transport-token",
                "privateMaterial": "private-value",
            },
            route_audit={
                "endpointUrl": "http://127.0.0.1:52415/route?token=abc",
                "token": "transport-token",
            },
            status="received",
            policy=policy,
        )
        record_cai_owned_transport_batch(
            session.session_id,
            batch_id="caibatch_processing",
            phase="decode_activation_batches",
            source_node_id="node-a",
            sink_node_id="node-b",
            payload_size_bytes=64,
            payload_sha256_hex="bb" * 32,
            status="processing",
            policy=policy,
        )
        record_cai_owned_transport_batch(
            session.session_id,
            batch_id="caibatch_failed",
            phase="decode_activation_batches",
            source_node_id="node-a",
            sink_node_id="node-b",
            payload_size_bytes=64,
            payload_sha256_hex="cc" * 32,
            status="received",
            policy=policy,
        )
        mark_cai_owned_transport_batch_status(
            session.session_id,
            "caibatch_failed",
            status="failed",
            node_id="node-b",
            error="adapter failed",
            policy=policy,
        )
        record_route_health(
            source_node_id="node-a",
            sink_node_id="node-b",
            route_type="direct_data",
            reachable=True,
            endpoint_url="http://user:pass@127.0.0.1:52415/health?token=abc",
            latency_ms=12.5,
            policy=policy,
        )
        save_node_capabilities(
            [
                NodeCapabilityRecord(
                    node_id="node-b",
                    source="test",
                    source_url="http://user:pass@127.0.0.1:52415/state?token=abc",
                    last_seen_at="2026-05-04T00:00:00+00:00",
                    updated_at="2026-05-04T00:00:00+00:00",
                    friendly_name="worker-b",
                    api_urls=["http://user:pass@127.0.0.1:52415"],
                    worker_enabled=True,
                    relay_enabled=True,
                    validator_enabled=False,
                    worker_reward_address=wallet.address,
                    worker_allowed_model_ids=["cai-network/Qwen3-0.6B-GGUF"],
                    resource_summary={
                        "ramBytes": 8 * 1024 * 1024 * 1024,
                        "vramBytes": 4 * 1024 * 1024 * 1024,
                        "cpuCores": 8,
                    },
                    readiness={
                        "caiOwnedTransport": {
                            "runtimeReady": True,
                            "llmShardSelfTest": {
                                "contractReady": True,
                                "productionReady": True,
                                "generationProbeReady": True,
                                "backendHealthReady": False,
                                "backendHealth": {
                                    "status": "degraded",
                                    "error": "native engine down",
                                },
                            },
                            "authToken": "transport-token",
                        }
                    },
                    route_hints={"directUrl": "http://127.0.0.1:52415?token=abc"},
                )
            ],
            policy=policy,
        )
        save_cai_owned_llm_shard_self_test_result(
            {
                "status": "passed",
                "contractReady": True,
                "productionReady": False,
                "generationProbeReady": False,
                "adapterId": "deterministic-bytes",
                "adapterVersion": "deterministic-bytes/0.1",
                "runtimeVersion": "cai-owned-runtime/0.1",
                "backendHealthReady": False,
                "backendHealth": {
                    "status": "degraded",
                    "healthEndpointUrl": (
                        "http://user:pass@127.0.0.1:9258/health?token=abc"
                    ),
                    "error": "native engine down",
                    "secretToken": "transport-token",
                },
                "secretToken": "transport-token",
                "patchBoundary": {"status": "validated", "privateKey": "bad"},
            },
            policy=policy,
        )

        snapshot = build_cai_owned_diagnostics_snapshot(
            local_node_id="node-b",
            model_id="cai-network/Qwen3-0.6B-GGUF",
            max_records=10,
            policy=policy,
        )
        queue = build_cai_owned_worker_runtime_queue_snapshot(
            local_node_id="node-b",
            max_records=10,
            policy=policy,
        )

    encoded = json.dumps(snapshot, ensure_ascii=False)
    assert snapshot["chainNetwork"] == "mainnet"
    assert snapshot["wallet"]["activeWalletId"] == wallet.wallet_id
    assert snapshot["wallet"]["activeWalletAddress"] == wallet.address
    assert snapshot["wallet"]["unlocked"] is True
    assert snapshot["summary"]["sessionCount"] == 1
    assert snapshot["summary"]["batchInboxCount"] == 3
    assert snapshot["summary"]["batchInboxStatusCounts"] == {
        "failed": 1,
        "processing": 1,
        "received": 1,
    }
    assert snapshot["summary"]["workerCount"] == 1
    assert snapshot["summary"]["runtimeReadyNodeCount"] == 0
    assert snapshot["summary"]["llmContractReadyNodeCount"] == 0
    assert snapshot["summary"]["llmProductionReadyNodeCount"] == 0
    assert snapshot["distributedInference"]["status"] == "blocked"
    assert snapshot["distributedInference"]["workerCount"] == 1
    assert snapshot["distributedInference"]["readyExecutorCount"] == 0
    assert "cai_owned_transport_not_runtime_ready" in snapshot[
        "distributedInference"
    ]["blockingReasons"]
    assert snapshot["llmShardSelfTest"]["backendHealthReady"] is False
    assert snapshot["llmShardSelfTest"]["generationProbeReady"] is False
    assert snapshot["llmShardSelfTest"]["backendHealth"]["status"] == "degraded"
    assert snapshot["llmShardSelfTest"]["backendHealth"]["healthEndpointUrl"] == (
        "http://127.0.0.1:9258/health"
    )
    inbox_statuses = {
        item["batch"]["status"] for item in snapshot["caiOwnedTransport"]["batchInbox"]
    }
    assert inbox_statuses == {"failed", "processing", "received"}
    assert snapshot["caiOwnedTransport"]["workerRuntimeQueue"]["receivedCount"] == 1
    assert snapshot["caiOwnedTransport"]["workerRuntimeQueue"]["processingCount"] == 1
    assert snapshot["caiOwnedTransport"]["workerRuntimeQueue"]["failedCount"] == 1
    assert snapshot["caiOwnedTransport"]["workerRuntimeQueue"]["recordCount"] == 3
    assert queue["receivedCount"] == 1
    assert queue["currentBatch"]["batch"]["batchId"] == "caibatch_processing"
    assert queue["lastError"] == "adapter failed"
    assert snapshot["routeHealth"]["records"][0]["endpointUrl"] == (
        "http://127.0.0.1:52415/health"
    )
    assert "transport-token" not in encoded
    assert "private-value" not in encoded
    assert "user:pass" not in encoded
    assert "privateKey" not in encoded


def test_cai_owned_diagnostics_cli_outputs_json_snapshot() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-diagnostics-cli")
        create_cai_owned_transport_session(
            instance_id="instance-cli",
            session_id="caiot_cli",
            participant_node_ids=["node-a", "node-b"],
            executor_node_ids=["node-b"],
            policy=policy,
        )

        output = handle_cai_owned_diagnostics(
            wallet_data_dirname=".tmp-cai-owned-diagnostics-cli",
            local_node_id="node-b",
            model_id="cai-network/Qwen3-0.6B-GGUF",
            max_records=5,
        )

    snapshot = json.loads(output)
    assert snapshot["schemaVersion"] == 1
    assert snapshot["summary"]["sessionCount"] == 1
    assert snapshot["caiOwnedTransport"]["localNodeId"] == "node-b"
    assert snapshot["distributedInference"]["modelId"] == "cai-network/Qwen3-0.6B-GGUF"


def test_distributed_inference_diagnostics_reports_ready_direct_executors() -> None:
    readiness = {
        "caiOwnedTransport": {
            "runtimeReady": True,
            "runtimeReadyProof": {"verified": True},
            "llmShardSelfTest": {
                "contractReady": True,
                "productionReady": True,
                "generationProbeReady": True,
                "backendHealthReady": True,
            },
        },
        "modelShardInventory": {
            "cai-network/Qwen3-0.6B-GGUF": {
                "status": "ready",
            }
        },
    }
    records = [
        NodeCapabilityRecord(
            node_id="node-a",
            source="test",
            source_url=None,
            last_seen_at="2026-05-04T00:00:00+00:00",
            updated_at="2026-05-04T00:00:00+00:00",
            worker_enabled=True,
            worker_allowed_model_ids=["cai-network/Qwen3-0.6B-GGUF"],
            readiness=readiness,
        ),
        NodeCapabilityRecord(
            node_id="node-b",
            source="test",
            source_url=None,
            last_seen_at="2026-05-04T00:00:00+00:00",
            updated_at="2026-05-04T00:00:00+00:00",
            worker_enabled=True,
            model_ids=["cai-network/Qwen3-0.6B-GGUF"],
            readiness=readiness,
        ),
    ]
    routes = [
        RouteHealthRecord(
            route_id="route-a",
            source_node_id="requester",
            sink_node_id="node-a",
            route_type="direct_data",
            reachable=True,
            checked_at="2026-05-04T00:00:00+00:00",
            endpoint_url="http://node-a:52415/data?token=secret",
        ),
        RouteHealthRecord(
            route_id="route-b",
            source_node_id="requester",
            sink_node_id="node-b",
            route_type="direct_data",
            reachable=True,
            checked_at="2026-05-04T00:00:00+00:00",
            endpoint_url="http://node-b:52415/data?token=secret",
        ),
    ]

    diagnostics = build_distributed_inference_diagnostics(
        local_node_id="requester",
        model_id="cai-network/Qwen3-0.6B-GGUF",
        node_capabilities=records,
        route_health_records=routes,
    )

    assert diagnostics["status"] == "ready"
    assert diagnostics["workerCount"] == 2
    assert diagnostics["readyExecutorCount"] == 2
    assert diagnostics["directRouteReadyExecutorCount"] == 2
    assert diagnostics["relayRouteReadyExecutorCount"] == 0
    assert diagnostics["runtimeReadyExecutorCount"] == 2
    assert diagnostics["modelReadyExecutorCount"] == 2
    assert diagnostics["blockingReasons"] == []
    assert diagnostics["executors"][0]["selectedRoute"]["endpointUrl"] == (
        "http://node-a:52415/data"
    )


def test_distributed_inference_diagnostics_reports_single_ready_direct_executor() -> None:
    model_id = "cai-network/Qwen3-0.6B-GGUF"
    diagnostics = build_distributed_inference_diagnostics(
        local_node_id="requester",
        model_id=model_id,
        node_capabilities=[
            NodeCapabilityRecord(
                node_id="node-a",
                source="test",
                source_url=None,
                last_seen_at="2026-05-04T00:00:00+00:00",
                updated_at="2026-05-04T00:00:00+00:00",
                worker_enabled=True,
                model_ids=[model_id],
                readiness={
                    "caiOwnedTransport": {
                        "runtimeReady": True,
                        "runtimeReadyProof": {"verified": True},
                        "llmShardSelfTest": {
                            "contractReady": True,
                            "productionReady": True,
                            "generationProbeReady": True,
                            "backendHealthReady": True,
                        },
                    },
                },
            )
        ],
        route_health_records=[
            RouteHealthRecord(
                route_id="route-a",
                source_node_id="requester",
                sink_node_id="node-a",
                route_type="direct_data",
                reachable=True,
                checked_at="2026-05-04T00:00:00+00:00",
                endpoint_url="http://node-a:52415/data",
            )
        ],
    )

    assert diagnostics["status"] == "ready"
    assert diagnostics["readyExecutorCount"] == 1
    assert diagnostics["directRouteReadyExecutorCount"] == 1
    assert diagnostics["executors"][0]["readyForDistributedInference"] is True
    assert diagnostics["executors"][0]["routeReason"] == "direct_route_ready"


def test_distributed_inference_diagnostics_reports_overlay_only_executor_blocker() -> None:
    model_id = "cai-network/Qwen3-0.6B-GGUF"
    diagnostics = build_distributed_inference_diagnostics(
        local_node_id="requester",
        model_id=model_id,
        node_capabilities=[
            NodeCapabilityRecord(
                node_id="node-a",
                source="test",
                source_url=None,
                last_seen_at="2026-05-04T00:00:00+00:00",
                updated_at="2026-05-04T00:00:00+00:00",
                worker_enabled=True,
                model_ids=[model_id],
                readiness={
                    "caiOwnedTransport": {
                        "runtimeReady": True,
                        "runtimeReadyProof": {"verified": True},
                        "llmShardSelfTest": {
                            "contractReady": True,
                            "productionReady": True,
                            "generationProbeReady": True,
                            "backendHealthReady": True,
                        },
                    },
                },
            )
        ],
        route_health_records=[
            RouteHealthRecord(
                route_id="route-overlay",
                source_node_id="requester",
                sink_node_id="node-a",
                route_type="overlay_peer",
                reachable=True,
                checked_at="2026-05-04T00:00:00+00:00",
                endpoint_url="http://node-a:52415/overlay",
            )
        ],
    )

    assert diagnostics["status"] == "blocked"
    assert diagnostics["readyExecutorCount"] == 0
    assert diagnostics["routeReadyExecutorCount"] == 0
    assert "data_plane_route_missing" in diagnostics["blockingReasons"]
    assert diagnostics["executors"][0]["routeClass"] == "overlay"
    assert diagnostics["executors"][0]["routeReason"] == "data_plane_route_missing"


def test_distributed_inference_diagnostics_reports_model_shard_missing() -> None:
    diagnostics = build_distributed_inference_diagnostics(
        local_node_id="requester",
        model_id="cai-network/Qwen3-0.6B-GGUF",
        node_capabilities=[
            NodeCapabilityRecord(
                node_id="node-a",
                source="test",
                source_url=None,
                last_seen_at="2026-05-04T00:00:00+00:00",
                updated_at="2026-05-04T00:00:00+00:00",
                worker_enabled=True,
                readiness={
                    "caiOwnedTransport": {
                        "runtimeReady": True,
                        "runtimeReadyProof": {"verified": True},
                        "llmShardSelfTest": {
                            "contractReady": True,
                            "productionReady": True,
                            "generationProbeReady": True,
                            "backendHealthReady": True,
                        },
                    },
                },
            )
        ],
        route_health_records=[
            RouteHealthRecord(
                route_id="route-a",
                source_node_id="requester",
                sink_node_id="node-a",
                route_type="direct_data",
                reachable=True,
                checked_at="2026-05-04T00:00:00+00:00",
                endpoint_url="http://node-a:52415/data",
            )
        ],
    )

    assert diagnostics["status"] == "blocked"
    assert diagnostics["routeReadyExecutorCount"] == 1
    assert diagnostics["runtimeReadyExecutorCount"] == 1
    assert diagnostics["modelReadyExecutorCount"] == 0
    assert "model_not_advertised" in diagnostics["blockingReasons"]
    assert diagnostics["executors"][0]["modelReason"] == "model_not_advertised"


def test_distributed_inference_diagnostics_reports_transport_not_runtime_ready() -> None:
    model_id = "cai-network/Qwen3-0.6B-GGUF"
    diagnostics = build_distributed_inference_diagnostics(
        local_node_id="requester",
        model_id=model_id,
        node_capabilities=[
            NodeCapabilityRecord(
                node_id="node-a",
                source="test",
                source_url=None,
                last_seen_at="2026-05-04T00:00:00+00:00",
                updated_at="2026-05-04T00:00:00+00:00",
                worker_enabled=True,
                model_ids=[model_id],
                readiness={
                    "caiOwnedTransport": {
                        "runtimeReady": False,
                        "runtimeReadyProof": {"verified": False},
                    },
                },
            )
        ],
        route_health_records=[
            RouteHealthRecord(
                route_id="route-a",
                source_node_id="requester",
                sink_node_id="node-a",
                route_type="direct_data",
                reachable=True,
                checked_at="2026-05-04T00:00:00+00:00",
                endpoint_url="http://node-a:52415/data",
            )
        ],
    )

    assert diagnostics["status"] == "blocked"
    assert diagnostics["routeReadyExecutorCount"] == 1
    assert diagnostics["modelReadyExecutorCount"] == 1
    assert diagnostics["runtimeReadyExecutorCount"] == 0
    assert "cai_owned_transport_not_runtime_ready" in diagnostics["blockingReasons"]
    assert diagnostics["executors"][0]["runtimeReady"] is False


def test_distributed_inference_diagnostics_reports_model_and_route_blockers() -> None:
    diagnostics = build_distributed_inference_diagnostics(
        local_node_id="requester",
        model_id="cai-network/Qwen3-0.6B-GGUF",
        node_capabilities=[
            NodeCapabilityRecord(
                node_id="node-a",
                source="test",
                source_url=None,
                last_seen_at="2026-05-04T00:00:00+00:00",
                updated_at="2026-05-04T00:00:00+00:00",
                worker_enabled=True,
                worker_allowed_model_ids=["other/model"],
                readiness={},
            )
        ],
        route_health_records=[],
    )

    assert diagnostics["status"] == "blocked"
    assert diagnostics["readyExecutorCount"] == 0
    assert diagnostics["routeReadyExecutorCount"] == 0
    assert diagnostics["modelReadyExecutorCount"] == 0
    assert "model_not_allowed" in diagnostics["blockingReasons"]
    assert "route_health_missing" in diagnostics["blockingReasons"]
    assert "cai_owned_transport_not_runtime_ready" in diagnostics["blockingReasons"]

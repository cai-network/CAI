# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import hashlib
import json

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.decentralized_compute import (  # noqa: E402
    CAI_OWNED_TRANSPORT_EXECUTION_PIPELINE_SINGLE_PASS_FINAL_DECODE,
    CAI_OWNED_TRANSPORT_PROTOCOL,
    EXECUTION_MODE_CAI_OWNED_TRANSPORT_REQUIRED,
    EXECUTION_MODE_LLAMA_CPP_RPC_LOW_LATENCY,
    EXECUTION_MODE_SINGLE_NODE,
    await_cai_owned_transport_session_final_result,
    accept_cai_owned_transport_completion_notice,
    build_cai_owned_transport_batch_envelope,
    build_cai_owned_transport_batch_hash_chain,
    build_cai_owned_transport_execution_proof,
    build_cai_owned_transport_execution_dag,
    build_cai_owned_llm_handoff_metadata,
    build_cai_owned_llm_shard_frame_metadata_from_runtime,
    build_cai_owned_transport_frame_metadata,
    build_cai_owned_transport_output_batch_envelope,
    build_cai_owned_transport_session_offer,
    cai_owned_transport_auth_headers,
    cai_owned_transport_batch_output_payload_path,
    cai_owned_transport_batch_payload_bytes,
    cai_owned_transport_batch_payload_path,
    cai_owned_transport_sessions_file_path,
    cai_owned_transport_session_to_dict,
    cai_owned_transport_runtime_readiness,
    cai_owned_transport_version_compatibility,
    cai_owned_transport_shard_receipts_from_processed_batches,
    claim_cai_owned_transport_batch,
    claim_next_cai_owned_transport_batch,
    cleanup_cai_owned_transport_payload_storage,
    complete_cai_owned_transport_batch_processing,
    complete_cai_owned_transport_session,
    complete_cai_owned_transport_work_item,
    create_cai_owned_transport_session,
    create_cai_owned_transport_session_from_offer,
    deterministic_cai_owned_transport_session_id,
    dispatch_cai_owned_transport_execution_dag,
    fail_cai_owned_transport_work_item,
    heartbeat_cai_owned_transport_batch,
    latest_cai_owned_transport_final_output,
    list_cai_owned_transport_replay_cache,
    list_cai_owned_transport_sessions,
    list_cai_owned_transport_batch_inbox,
    mark_cai_owned_transport_batch_status,
    plan_llama_cpp_distributed_execution,
    preflight_cai_owned_transport_data_plane_routes,
    preflight_cai_owned_transport_executor_readiness,
    read_cai_owned_transport_batch_output_payload,
    read_cai_owned_transport_batch_payload,
    record_cai_owned_transport_batch,
    record_cai_owned_transport_batch_envelope,
    record_cai_owned_transport_shard_receipt,
    reconcile_cai_owned_transport_session_timeouts,
    save_cai_owned_transport_sessions,
    sign_cai_owned_transport_batch_envelope,
    sign_cai_owned_transport_execution_proof,
    sign_cai_owned_transport_session_offer,
    sign_cai_owned_transport_shard_receipt,
    submit_cai_owned_transport_batch_envelope,
    submit_cai_owned_transport_batch_envelope_to_any,
    submit_cai_owned_transport_session_offer,
    submit_cai_owned_transport_shard_receipt,
    validate_cai_owned_transport_batch_envelope,
    validate_cai_owned_transport_execution_dag,
    validate_cai_owned_transport_execution_proof,
    validate_cai_owned_transport_frame_metadata,
    validate_cai_owned_llm_handoff_metadata,
    validate_cai_owned_transport_local_runtime_auth,
    validate_cai_owned_transport_request_auth,
    validate_cai_owned_transport_session_execution_audit,
    validate_cai_owned_transport_session_offer,
    wait_for_cai_owned_transport_final_output,
)
from cai_compute_chain.model import ChainNetwork, WalletPolicy  # noqa: E402
from cai_compute_chain.route_health import RouteHealthRecord  # noqa: E402
from cai_compute_chain.wallet_signing import (  # noqa: E402
    encode_bytes,
    generate_signing_seed,
    public_key_b64_from_seed,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _signing_material() -> tuple[str, str]:
    signing_seed = generate_signing_seed()
    return public_key_b64_from_seed(signing_seed), encode_bytes(signing_seed)


def _runtime_ready_live_proof() -> dict[str, object]:
    return {
        "status": "ok",
        "sessionId": "caiot_live_proof",
        "instanceId": "instance-live-proof",
        "requesterNodeId": "node-user",
        "executorNodeIds": ["node-a", "node-b"],
        "finalResult": {
            "proofVerified": True,
            "finalOutput": {
                "payloadBase64": "b2s=",
                "payloadSha256Hex": hashlib.sha256(b"ok").hexdigest(),
            },
        },
    }


def _production_llm_self_test() -> dict[str, object]:
    return {
        "status": "passed",
        "contractReady": True,
        "productionReady": True,
        "generationProbeReady": True,
        "patchBoundaryVerified": True,
    }


def test_cai_owned_transport_sessions_recover_from_trailing_json_junk() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        path = cai_owned_transport_sessions_file_path(policy)
        payload = [
            {
                "session_id": "caiot_recovered",
                "instance_id": "instance-recovered",
                "participant_node_ids": ["node-a"],
                "status": "created",
                "created_at": "2026-05-12T00:00:00+00:00",
                "updated_at": "2026-05-12T00:00:01+00:00",
            }
        ]
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n]\n",
            encoding="utf-8",
        )

        records = list_cai_owned_transport_sessions(policy)

        assert [record.session_id for record in records] == ["caiot_recovered"]
        assert json.loads(path.read_text(encoding="utf-8")) == payload
        backups = list(path.parent.glob("cai-owned-transport-sessions.corrupt-*.json"))
        assert len(backups) == 1
        assert backups[0].read_text(encoding="utf-8").endswith("\n]\n")


def test_cai_owned_transport_sessions_save_falls_back_when_replace_is_locked() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch(
        "cai_compute_chain.local_json_store.os.replace",
        side_effect=PermissionError(13, "Access is denied"),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")

        create_cai_owned_transport_session(
            instance_id="instance-locked",
            participant_node_ids=["node-a"],
            session_id="caiot_locked",
            policy=policy,
        )

        records = list_cai_owned_transport_sessions(policy)
        assert [record.session_id for record in records] == ["caiot_locked"]


def _rpc_record(
    source: str,
    sink: str,
    *,
    reachable: bool = True,
    latency_ms: float | None = 7.0,
) -> RouteHealthRecord:
    return RouteHealthRecord(
        route_id=f"{source}->{sink}",
        source_node_id=source,
        sink_node_id=sink,
        route_type="llama_cpp_rpc_direct",
        endpoint_url="llama-cpp-rpc://198.51.100.11:52435",
        reachable=reachable,
        checked_at="2026-05-03T00:00:00+00:00",
        latency_ms=latency_ms,
    )


def _route_record(
    source: str,
    sink: str,
    *,
    route_type: str = "direct_api",
    transit_node_id: str | None = None,
    reachable: bool = True,
) -> RouteHealthRecord:
    transit = f":{transit_node_id}" if transit_node_id else ""
    return RouteHealthRecord(
        route_id=f"{source}->{sink}:{route_type}{transit}",
        source_node_id=source,
        sink_node_id=sink,
        route_type=route_type,
        endpoint_url=f"http://{sink}:52415",
        reachable=reachable,
        checked_at="2026-05-03T00:00:00+00:00",
        latency_ms=8.0,
        transit_node_id=transit_node_id,
    )


def test_single_node_execution_does_not_require_transport() -> None:
    plan = plan_llama_cpp_distributed_execution("node-a", [], [])

    assert plan["executionMode"] == EXECUTION_MODE_SINGLE_NODE
    assert plan["standardLlamaCppRpcReady"] is True
    assert plan["requiresCaiOwnedTransport"] is False
    assert plan["caiOwnedTransport"] is None


def test_low_latency_compute_cell_uses_standard_llama_cpp_rpc() -> None:
    plan = plan_llama_cpp_distributed_execution(
        "node-a",
        ["node-b"],
        [_rpc_record("node-a", "node-b", latency_ms=8.0)],
        model_id="cai-network/Qwen3-0.6B-GGUF",
    )

    assert plan["executionMode"] == EXECUTION_MODE_LLAMA_CPP_RPC_LOW_LATENCY
    assert plan["standardLlamaCppRpcReady"] is True
    assert plan["requiresCaiOwnedTransport"] is False
    assert plan["computeCellProfile"]["profile"] == "low_latency_sharded_cell"


def test_wan_risky_compute_cell_requires_cai_owned_transport() -> None:
    plan = plan_llama_cpp_distributed_execution(
        "node-a",
        ["node-b"],
        [
            _rpc_record("node-a", "node-b", latency_ms=45.0),
            _rpc_record("node-b", "node-a", latency_ms=45.0),
        ],
        model_id="cai-network/Qwen3-0.6B-GGUF",
    )

    assert plan["executionMode"] == EXECUTION_MODE_CAI_OWNED_TRANSPORT_REQUIRED
    assert plan["standardLlamaCppRpcReady"] is False
    assert plan["requiresCaiOwnedTransport"] is True
    assert plan["computeCellProfile"]["profile"] == "wan_risky_sharded_cell"
    assert plan["caiOwnedTransport"]["protocol"] == CAI_OWNED_TRANSPORT_PROTOCOL
    assert "activation_batch_stream" in plan["caiOwnedTransport"]["requiredCapabilities"]
    assert plan["caiOwnedTransport"]["routeHealthReadiness"]["status"] == "ready"
    assert plan["caiOwnedTransport"]["routeHealthReadiness"]["ready"] is True


def test_unproven_compute_cell_requires_cai_owned_transport() -> None:
    plan = plan_llama_cpp_distributed_execution(
        "node-a",
        ["node-b"],
        [],
        model_id="cai-network/Qwen3-0.6B-GGUF",
    )

    assert plan["executionMode"] == EXECUTION_MODE_CAI_OWNED_TRANSPORT_REQUIRED
    assert plan["standardLlamaCppRpcReady"] is False
    assert plan["requiresCaiOwnedTransport"] is True
    assert plan["computeCellProfile"]["profile"] == "unproven_sharded_cell"
    assert plan["caiOwnedTransport"]["routeHealthReadiness"]["status"] == "failed"
    assert plan["caiOwnedTransport"]["routeHealthReadiness"]["ready"] is False


def test_cai_owned_transport_runtime_readiness_defaults_to_planned() -> None:
    readiness = cai_owned_transport_runtime_readiness()

    assert readiness["protocol"] == CAI_OWNED_TRANSPORT_PROTOCOL
    assert readiness["implemented"] is False
    assert readiness["runtimeReady"] is False
    assert readiness["status"] == "planned"
    assert readiness["versionCompatible"] is True


def test_cai_owned_transport_runtime_readiness_reports_version_compatibility() -> None:
    ready = cai_owned_transport_runtime_readiness(
        runtime_ready=True,
        implemented=True,
        runtime_version="cai-owned-runtime/0.1",
        adapter_id="deterministic-bytes",
        adapter_version="deterministic-bytes/0.1",
        compatible_protocol_versions=[1],
        runtime_ready_proof=_runtime_ready_live_proof(),
        llm_shard_self_test=_production_llm_self_test(),
    )
    missing_versions = cai_owned_transport_runtime_readiness(
        runtime_ready=True,
        implemented=True,
        runtime_ready_proof=_runtime_ready_live_proof(),
        llm_shard_self_test=_production_llm_self_test(),
    )
    incompatible = cai_owned_transport_version_compatibility(
        {
            "protocol": CAI_OWNED_TRANSPORT_PROTOCOL,
            "protocolVersion": 999,
            "runtimeVersion": "cai-owned-runtime/0.1",
            "adapterVersion": "deterministic-bytes/0.1",
        },
        require_runtime_versions=True,
    )

    assert ready["versionCompatible"] is True
    assert ready["versionCompatibility"]["runtimeVersion"] == "cai-owned-runtime/0.1"
    assert missing_versions["versionCompatible"] is False
    assert "CAI-owned transport runtime version is missing." in (
        missing_versions["versionCompatibility"]["errors"]
    )
    assert incompatible["compatible"] is False
    assert "CAI-owned transport protocol version is unsupported." in (
        incompatible["errors"]
    )


def test_cai_owned_transport_runtime_readiness_promotes_self_test_adapter_versions() -> None:
    readiness = cai_owned_transport_runtime_readiness(
        runtime_ready=True,
        implemented=True,
        runtime_ready_proof=_runtime_ready_live_proof(),
        llm_shard_self_test={
            **_production_llm_self_test(),
            "adapterId": "llama.cpp-external-shard",
            "adapterVersion": "llama.cpp-external-shard/0.1",
        },
    )

    assert readiness["runtimeVersion"] == "cai-owned-runtime/0.1"
    assert readiness["adapterId"] == "llama.cpp-external-shard"
    assert readiness["adapterVersion"] == "llama.cpp-external-shard/0.1"
    assert readiness["versionCompatible"] is True
    assert readiness["versionCompatibility"]["errors"] == []


def test_cai_owned_transport_runtime_readiness_summarizes_llm_self_test() -> None:
    readiness = cai_owned_transport_runtime_readiness(
        implemented=True,
        status="test_adapter_ready",
        llm_shard_self_test={
            "status": "passed",
            "contractReady": True,
            "productionReady": False,
            "generationProbeReady": False,
            "patchBoundaryVerified": True,
            "patchBoundaryPatchId": "cai-llama-cpp-shard-smoke-runner",
            "outputPayloadSizeBytes": 32,
            "backendHealthReady": True,
            "backendHealth": {"status": "ok", "nativeEngineMode": "persistent_jsonl"},
            "error": "x" * 800,
        },
    )

    self_test = readiness["llmShardSelfTest"]
    assert self_test["status"] == "passed"
    assert self_test["contractReady"] is True
    assert self_test["productionReady"] is False
    assert self_test["generationProbeReady"] is False
    assert self_test["patchBoundaryVerified"] is True
    assert self_test["patchBoundaryPatchId"] == "cai-llama-cpp-shard-smoke-runner"
    assert self_test["outputPayloadSizeBytes"] == 32
    assert self_test["backendHealthReady"] is True
    assert self_test["backendHealthStatus"] == "ok"
    assert self_test["backendHealth"]["nativeEngineMode"] == "persistent_jsonl"
    assert len(self_test["error"]) == 500


def test_cai_owned_transport_runtime_readiness_self_test_prevents_false_ready() -> None:
    smoke_ready = cai_owned_transport_runtime_readiness(
        runtime_ready=True,
        implemented=True,
        status="ready",
        runtime_ready_proof=_runtime_ready_live_proof(),
        llm_shard_self_test={
            "status": "passed",
            "contractReady": True,
            "productionReady": False,
            "patchBoundaryVerified": True,
        },
    )
    production_ready = cai_owned_transport_runtime_readiness(
        runtime_ready=True,
        implemented=True,
        status="ready",
        runtime_ready_proof=_runtime_ready_live_proof(),
        llm_shard_self_test=_production_llm_self_test(),
    )
    missing_live_proof = cai_owned_transport_runtime_readiness(
        runtime_ready=True,
        implemented=True,
        status="ready",
        llm_shard_self_test={
            "status": "passed",
            "contractReady": True,
            "productionReady": True,
            "generationProbeReady": True,
            "patchBoundaryVerified": True,
        },
    )
    stale_production_ready = cai_owned_transport_runtime_readiness(
        runtime_ready=True,
        implemented=True,
        status="ready",
        runtime_ready_proof=_runtime_ready_live_proof(),
        llm_shard_self_test={
            "status": "passed",
            "contractReady": True,
            "productionReady": True,
            "patchBoundaryVerified": True,
        },
    )
    failed = cai_owned_transport_runtime_readiness(
        runtime_ready=True,
        implemented=True,
        llm_shard_self_test={
            "status": "failed",
            "contractReady": False,
            "productionReady": False,
            "error": "patch boundary missing",
        },
    )
    degraded_backend = cai_owned_transport_runtime_readiness(
        runtime_ready=True,
        implemented=True,
        status="ready",
        llm_shard_self_test={
            "status": "passed",
            "contractReady": True,
            "productionReady": True,
            "generationProbeReady": True,
            "patchBoundaryVerified": True,
            "backendHealthReady": False,
            "backendHealth": {"status": "degraded", "error": "native engine down"},
        },
    )

    assert smoke_ready["status"] == "test_adapter_ready"
    assert smoke_ready["runtimeReady"] is False
    assert production_ready["status"] == "ready"
    assert production_ready["runtimeReady"] is True
    assert production_ready["runtimeReadyProof"]["verified"] is True
    assert missing_live_proof["status"] == "test_adapter_ready"
    assert missing_live_proof["runtimeReady"] is False
    assert "live PC-to-PC proof" in missing_live_proof["runtimeReadyProofError"]
    assert stale_production_ready["status"] == "test_adapter_ready"
    assert stale_production_ready["runtimeReady"] is False
    assert failed["status"] == "failed"
    assert failed["runtimeReady"] is False
    assert degraded_backend["status"] == "failed"
    assert degraded_backend["runtimeReady"] is False
    assert degraded_backend["llmShardSelfTest"]["backendHealthReady"] is False


def test_cai_owned_transport_execution_dag_plans_shard_order() -> None:
    dag = build_cai_owned_transport_execution_dag(
        session_id="caiot_dag",
        requester_node_id="node-user",
        executor_node_ids=["node-a", "node-b"],
        total_layer_count=28,
        model_id="cai-network/Qwen3-0.6B-GGUF",
        task_id="task-dag",
        input_payload_sha256_hex="11" * 32,
        expected_output_payload_sha256_hex="22" * 32,
        created_at="2026-05-03T00:00:00+00:00",
    )
    repeated = build_cai_owned_transport_execution_dag(
        session_id="caiot_dag",
        requester_node_id="node-user",
        executor_node_ids=["node-a", "node-b"],
        total_layer_count=28,
        model_id="cai-network/Qwen3-0.6B-GGUF",
        task_id="task-dag",
        input_payload_sha256_hex="11" * 32,
        expected_output_payload_sha256_hex="22" * 32,
        created_at="2026-05-03T00:00:00+00:00",
    )
    valid, error = validate_cai_owned_transport_execution_dag(
        dag,
        session_id="caiot_dag",
        participant_node_ids=["node-user", "node-a", "node-b"],
    )
    wrong_chain_valid, wrong_chain_error = validate_cai_owned_transport_execution_dag(
        dag,
        chain_id=ChainNetwork.TESTNET.value,
        session_id="caiot_dag",
    )
    tampered = dict(dag)
    tampered["stages"] = [dict(item) for item in dag["stages"]]
    tampered["stages"][1]["stageId"] = "tampered-stage-id"
    tampered_valid, tampered_error = validate_cai_owned_transport_execution_dag(
        tampered,
        session_id="caiot_dag",
    )

    assert valid is True
    assert error is None
    assert dag["dagHashSha256Hex"] == repeated["dagHashSha256Hex"]
    assert dag["chainId"] == "mainnet"
    assert dag["requesterNodeId"] == "node-user"
    assert dag["executorNodeIds"] == ["node-a", "node-b"]
    assert dag["participantNodeIds"] == ["node-user", "node-a", "node-b"]
    assert dag["shardRanges"] == [
        {"nodeId": "node-a", "layerStart": 0, "layerEnd": 14, "layerCount": 14},
        {"nodeId": "node-b", "layerStart": 14, "layerEnd": 28, "layerCount": 14},
    ]
    assert [stage["sequence"] for stage in dag["stages"]] == [0, 1, 2, 3]
    assert dag["stages"][0]["phase"] == "prefill_activation_batches"
    assert dag["stages"][0]["sourceNodeId"] == "node-user"
    assert dag["stages"][0]["sinkNodeId"] == "node-a"
    assert dag["stages"][0]["expectedInputPayloadSha256Hex"] == "11" * 32
    assert dag["stages"][1]["dependsOnStageIds"] == [dag["stages"][0]["stageId"]]
    assert dag["stages"][2]["phase"] == "decode_activation_batches"
    assert dag["stages"][2]["dependsOnStageIds"] == [dag["stages"][1]["stageId"]]
    assert dag["stages"][3]["outputToNodeId"] == "node-user"
    assert dag["stages"][3]["expectedOutputPayloadSha256Hex"] == "22" * 32
    assert wrong_chain_valid is False
    assert wrong_chain_error == (
        "CAI-owned transport execution DAG is for chain 'mainnet', "
        "expected 'testnet'."
    )
    assert tampered_valid is False
    assert tampered_error == "CAI-owned transport execution DAG stage id does not match."


def test_cai_owned_transport_execution_dag_allows_single_executor_direct_final_output() -> None:
    dag = build_cai_owned_transport_execution_dag(
        session_id="caiot_direct_final",
        requester_node_id="node-user",
        executor_node_ids=["node-a"],
        total_layer_count=28,
        model_id="cai-network/Qwen3-0.6B-GGUF",
        task_id="task-direct-final",
        input_payload_sha256_hex="11" * 32,
        expected_output_payload_sha256_hex="22" * 32,
        created_at="2026-05-03T00:00:00+00:00",
        single_executor_direct_final_output=True,
    )

    valid, error = validate_cai_owned_transport_execution_dag(
        dag,
        session_id="caiot_direct_final",
        participant_node_ids=["node-user", "node-a"],
    )

    assert valid is True
    assert error is None
    assert dag["singleExecutorDirectFinalOutput"] is True
    assert dag["stageCount"] == 1
    stage = dag["stages"][0]
    assert stage["phase"] == "decode_activation_batches"
    assert stage["sourceNodeId"] == "node-user"
    assert stage["sinkNodeId"] == "node-a"
    assert stage["outputToNodeId"] == "node-user"
    assert stage["dependsOnStageIds"] == []
    assert stage["payloadRole"] == "final_output"
    assert stage["expectedInputPayloadSha256Hex"] == "11" * 32
    assert stage["expectedOutputPayloadSha256Hex"] == "22" * 32


def test_cai_owned_transport_execution_dag_supports_single_pass_final_decode() -> None:
    dag = build_cai_owned_transport_execution_dag(
        session_id="caiot_single_pass",
        requester_node_id="node-user",
        executor_node_ids=["node-a", "node-b"],
        total_layer_count=28,
        model_id="cai-network/Qwen3-0.6B-GGUF",
        task_id="task-single-pass",
        input_payload_sha256_hex="11" * 32,
        expected_output_payload_sha256_hex="22" * 32,
        created_at="2026-05-03T00:00:00+00:00",
        execution_pipeline_mode=(
            CAI_OWNED_TRANSPORT_EXECUTION_PIPELINE_SINGLE_PASS_FINAL_DECODE
        ),
    )

    valid, error = validate_cai_owned_transport_execution_dag(
        dag,
        session_id="caiot_single_pass",
        participant_node_ids=["node-user", "node-a", "node-b"],
    )

    assert valid is True
    assert error is None
    assert dag["executionPipelineMode"] == (
        CAI_OWNED_TRANSPORT_EXECUTION_PIPELINE_SINGLE_PASS_FINAL_DECODE
    )
    assert dag["stageCount"] == 2
    assert [stage["sequence"] for stage in dag["stages"]] == [0, 1]
    assert dag["stages"][0]["phase"] == "prefill_activation_batches"
    assert dag["stages"][0]["sourceNodeId"] == "node-user"
    assert dag["stages"][0]["sinkNodeId"] == "node-a"
    assert dag["stages"][0]["outputToNodeId"] == "node-b"
    assert dag["stages"][1]["phase"] == "decode_activation_batches"
    assert dag["stages"][1]["sourceNodeId"] == "node-a"
    assert dag["stages"][1]["sinkNodeId"] == "node-b"
    assert dag["stages"][1]["outputToNodeId"] == "node-user"
    assert dag["stages"][1]["dependsOnStageIds"] == [dag["stages"][0]["stageId"]]
    assert dag["stages"][1]["payloadRole"] == "final_output"
    assert dag["stages"][1]["expectedOutputPayloadSha256Hex"] == "22" * 32


def test_cai_owned_transport_execution_dag_accepts_custom_shard_ranges() -> None:
    custom_ranges = [
        {"nodeId": "node-a", "layerStart": 0, "layerEnd": 10, "layerCount": 10},
        {"nodeId": "node-b", "layerStart": 10, "layerEnd": 28, "layerCount": 18},
    ]
    dag = build_cai_owned_transport_execution_dag(
        session_id="caiot_custom_dag",
        requester_node_id="node-user",
        executor_node_ids=["node-a", "node-b"],
        total_layer_count=28,
        shard_ranges=custom_ranges,
        model_id="cai-network/Qwen3-0.6B-GGUF",
        task_id="task-custom-dag",
    )
    valid, error = validate_cai_owned_transport_execution_dag(
        dag,
        session_id="caiot_custom_dag",
        participant_node_ids=["node-user", "node-a", "node-b"],
    )

    assert valid is True
    assert error is None
    assert dag["shardRanges"] == custom_ranges
    assert [stage["layerEnd"] for stage in dag["stages"][:2]] == [10, 28]
    assert [stage["layerStart"] for stage in dag["stages"][2:4]] == [0, 10]


def test_cai_owned_transport_execution_dag_rejects_non_contiguous_custom_shard_ranges() -> None:
    with pytest.raises(
        ValueError,
        match="shard ranges must be contiguous",
    ):
        build_cai_owned_transport_execution_dag(
            session_id="caiot_bad_custom_dag",
            requester_node_id="node-user",
            executor_node_ids=["node-a", "node-b"],
            total_layer_count=28,
            shard_ranges=[
                {
                    "nodeId": "node-a",
                    "layerStart": 0,
                    "layerEnd": 12,
                    "layerCount": 12,
                },
                {
                    "nodeId": "node-b",
                    "layerStart": 14,
                    "layerEnd": 28,
                    "layerCount": 14,
                },
            ],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-bad-custom-dag",
        )


def test_cai_owned_transport_execution_dag_rejects_executor_count_above_total_layers() -> None:
    with pytest.raises(
        ValueError,
        match="executor count exceeds total layer count",
    ):
        build_cai_owned_transport_execution_dag(
            session_id="caiot_too_many_executors",
            requester_node_id="node-user",
            executor_node_ids=["node-a", "node-b", "node-c"],
            total_layer_count=2,
            model_id="Example/TinyLlama-GGUF",
            task_id="task-too-many-executors",
        )


def test_dispatch_cai_owned_transport_execution_dag_uses_custom_shard_ranges() -> None:
    captured: list[dict[str, object]] = []
    custom_ranges = [
        {"nodeId": "node-a", "layerStart": 0, "layerEnd": 10, "layerCount": 10},
        {"nodeId": "node-b", "layerStart": 10, "layerEnd": 28, "layerCount": 18},
    ]

    def fake_urlopen(request, timeout: float):
        body = json.loads(request.data.decode("utf-8"))
        captured.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "body": body,
            }
        )
        return _FakeResponse({"status": "ok", "sessionId": body.get("sessionId")})

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        result = dispatch_cai_owned_transport_execution_dag(
            instance_id="instance-dispatch-custom",
            requester_node_id="node-user",
            executor_node_ids=["node-a", "node-b"],
            peer_cai_urls_by_node={
                "node-user": ["http://node-user:52415"],
                "node-a": ["http://node-a:52415"],
                "node-b": ["http://node-b:52415"],
            },
            initial_payload=b"user-prompt",
            total_layer_count=28,
            shard_ranges=custom_ranges,
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-dispatch-custom",
            policy=policy,
        )

    initial_body = captured[2]["body"]
    route_plan = initial_body["metadata"]["outputRoutePlan"]

    assert result["dag"]["shardRanges"] == custom_ranges
    assert initial_body["metadata"]["layerStart"] == 0
    assert initial_body["metadata"]["layerEnd"] == 10
    assert route_plan[0]["layerStart"] == 10
    assert route_plan[0]["layerEnd"] == 28
    assert route_plan[1]["layerStart"] == 0
    assert route_plan[1]["layerEnd"] == 10


def test_dispatch_cai_owned_transport_execution_dag_posts_offer_and_first_batch() -> None:
    captured: list[dict[str, object]] = []

    def fake_urlopen(request, timeout: float):
        body = json.loads(request.data.decode("utf-8"))
        captured.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "body": body,
            }
        )
        return _FakeResponse({"status": "ok", "sessionId": body.get("sessionId")})

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        result = dispatch_cai_owned_transport_execution_dag(
            instance_id="instance-dispatch",
            requester_node_id="node-user",
            executor_node_ids=["node-a", "node-b"],
            peer_cai_urls_by_node={
                "node-user": ["http://node-user:52415"],
                "node-a": ["http://node-a:52415"],
                "node-b": ["http://node-b:52415"],
            },
            initial_payload=b"user-prompt",
            total_layer_count=28,
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-dispatch",
            tokenizer_config_hash="ab" * 32,
            require_data_plane_route=True,
            policy=policy,
        )
        records = list_cai_owned_transport_sessions(policy)

    assert result["status"] == "dispatched"
    assert result["executorNodeIds"] == ["node-a", "node-b"]
    assert result["participantNodeIds"] == ["node-user", "node-a", "node-b"]
    assert result["routePreflight"]["status"] == "ready"
    assert result["routePreflight"]["participantNodeIds"] == [
        "node-user",
        "node-a",
        "node-b",
    ]
    assert len(result["localSession"]["dispatchRecords"]) == 1
    assert result["localSession"]["dispatchRecords"][0]["status"] == "sent"
    assert result["dispatchRecord"]["dispatchKind"] == "initial_batch"
    assert result["dag"]["stageCount"] == 4
    assert len(records) == 1
    assert records[0].participant_node_ids == ["node-user", "node-a", "node-b"]
    assert records[0].executor_node_ids == ["node-a", "node-b"]
    assert len(captured) == 3
    assert captured[0]["url"].endswith(f"/{result['sessionId']}/offer")
    assert captured[1]["url"].endswith(f"/{result['sessionId']}/offer")
    assert captured[2]["url"].endswith(f"/{result['sessionId']}/batch-envelopes")
    initial_body = captured[2]["body"]
    assert initial_body["sourceNodeId"] == "node-user"
    assert initial_body["sinkNodeId"] == "node-a"
    assert initial_body["metadata"]["requesterNodeId"] == "node-user"
    assert initial_body["metadata"]["coordinatorNodeId"] == "node-user"
    assert initial_body["metadata"]["nextSinkNodeId"] == "node-b"
    assert initial_body["metadata"]["remainingSinkNodeIds"] == [
        "node-a",
        "node-b",
        "node-user",
    ]
    assert initial_body["metadata"]["nextOutputPhase"] == "prefill_activation_batches"
    assert initial_body["metadata"]["nextOutputSequence"] == 1
    route_plan = initial_body["metadata"]["outputRoutePlan"]
    assert [item["sinkNodeId"] for item in route_plan] == [
        "node-b",
        "node-a",
        "node-b",
        "node-user",
    ]
    assert route_plan[-1]["finalOutput"] is True
    assert initial_body["metadata"]["peerCaiUrlsByNode"]["node-b"] == [
        "http://node-b:52415"
    ]
    assert cai_owned_transport_batch_payload_bytes(initial_body) == b"user-prompt"


def test_dispatch_cai_owned_transport_execution_dag_builds_llm_frame_templates() -> None:
    captured: list[dict[str, object]] = []

    def fake_urlopen(request, timeout: float):
        body = json.loads(request.data.decode("utf-8"))
        captured.append({"url": request.full_url, "body": body})
        return _FakeResponse({"status": "ok", "sessionId": body.get("sessionId")})

    runtime_metadata = {
        "modelId": "cai-network/Qwen3-0.6B-GGUF",
        "totalLayerCount": 28,
        "hiddenSize": 1024,
        "activationDtype": "f16",
        "tensorEncoding": "ggml-tensor-v1",
        "modelSha256Hex": "12" * 32,
        "backend": "llama.cpp-patched",
        "backendVersion": "llama.cpp/cai-shard-0.1",
    }

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        result = dispatch_cai_owned_transport_execution_dag(
            instance_id="instance-dispatch-llm",
            requester_node_id="node-user",
            executor_node_ids=["node-a", "node-b"],
            peer_cai_urls_by_node={
                "node-a": ["http://node-a:52415"],
                "node-b": ["http://node-b:52415"],
            },
            initial_payload=b"user-prompt",
            total_layer_count=28,
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-dispatch-llm",
            tokenizer_config_hash="ab" * 32,
            llm_runtime_metadata=runtime_metadata,
            initial_token_count=8,
            policy=policy,
        )

    initial_body = captured[2]["body"]
    metadata = initial_body["metadata"]
    route_plan = metadata["outputRoutePlan"]
    next_template = metadata["nextFrameTemplate"]
    nested_decode_template = next_template["nextFrameTemplate"]

    assert result["status"] == "dispatched"
    assert metadata["llmHandoff"]["modelId"] == "cai-network/Qwen3-0.6B-GGUF"
    assert metadata["llmHandoff"]["tensor"]["shape"] == [1, 8, 1024]
    assert metadata["tokenStart"] == 0
    assert metadata["tokenEnd"] == 8
    assert next_template["frameKind"] == "activation"
    assert next_template["layerStart"] == 14
    assert next_template["layerEnd"] == 28
    assert next_template["tokenStart"] == 0
    assert next_template["tokenEnd"] == 8
    assert next_template["llmHandoff"]["tensor"]["sha256Hex"] == (
        hashlib.sha256(b"").hexdigest()
    )
    assert nested_decode_template["frameKind"] == "decode"
    assert nested_decode_template["layerStart"] == 0
    assert nested_decode_template["layerEnd"] == 14
    assert nested_decode_template["tokenStart"] == 8
    assert nested_decode_template["tokenEnd"] == 9
    assert route_plan[0]["frameTemplate"]["layerStart"] == 14
    assert route_plan[1]["frameTemplate"]["frameKind"] == "decode"


def test_dispatch_cai_owned_transport_execution_dag_uses_direct_final_for_single_executor() -> None:
    captured: list[dict[str, object]] = []

    def fake_urlopen(request, timeout: float):
        body = json.loads(request.data.decode("utf-8"))
        captured.append({"url": request.full_url, "body": body})
        return _FakeResponse({"status": "ok", "sessionId": body.get("sessionId")})

    runtime_metadata = {
        "modelId": "cai-network/Qwen3-0.6B-GGUF",
        "totalLayerCount": 28,
        "hiddenSize": 1024,
        "activationDtype": "f16",
        "tensorEncoding": "ggml-tensor-v1",
        "modelSha256Hex": "12" * 32,
        "backend": "llama.cpp-patched",
        "backendVersion": "llama.cpp/cai-shard-0.1",
    }

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        result = dispatch_cai_owned_transport_execution_dag(
            instance_id="instance-dispatch-direct-final",
            requester_node_id="node-user",
            executor_node_ids=["node-a"],
            peer_cai_urls_by_node={
                "node-user": ["http://node-user:52415"],
                "node-a": ["http://node-a:52415"],
            },
            initial_payload=b"2+3=",
            total_layer_count=28,
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-dispatch-direct-final",
            tokenizer_config_hash="ab" * 32,
            llm_runtime_metadata=runtime_metadata,
            initial_token_count=1,
            policy=policy,
            single_executor_direct_final_output=True,
        )

    initial_body = captured[-1]["body"]
    metadata = initial_body["metadata"]

    assert result["dag"]["singleExecutorDirectFinalOutput"] is True
    assert result["dag"]["stageCount"] == 1
    assert initial_body["phase"] == "decode_activation_batches"
    assert metadata["frameKind"] == "decode"
    assert metadata["stageId"] == result["dag"]["stages"][0]["stageId"]
    assert metadata["singleExecutorDirectFinalOutput"] is True
    assert metadata["nextSinkNodeId"] == "node-user"
    assert metadata["nextOutputPhase"] == "decode_activation_batches"
    assert "nextFrameTemplate" not in metadata
    assert metadata["outputRoutePlan"][-1]["finalOutput"] is True
    assert cai_owned_transport_batch_payload_bytes(initial_body) == b"2+3="


def test_dispatch_cai_owned_transport_execution_dag_can_offer_requester_api() -> None:
    captured: list[dict[str, object]] = []

    def fake_urlopen(request, timeout: float):
        body = json.loads(request.data.decode("utf-8"))
        captured.append({"url": request.full_url, "body": body})
        return _FakeResponse({"status": "ok", "sessionId": body.get("sessionId")})

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        result = dispatch_cai_owned_transport_execution_dag(
            instance_id="instance-dispatch-requester-offer",
            requester_node_id="node-user",
            executor_node_ids=["node-a", "node-b"],
            peer_cai_urls_by_node={
                "node-user": ["http://node-user:52415"],
                "node-a": ["http://node-a:52415"],
                "node-b": ["http://node-b:52415"],
            },
            initial_payload=b"user-prompt",
            total_layer_count=28,
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-dispatch-requester-offer",
            submit_requester_offer=True,
            policy=policy,
        )

    assert result["status"] == "dispatched"
    assert set(result["offerResponses"]) == {"node-user", "node-a", "node-b"}
    assert len(captured) == 4
    assert captured[0]["url"].endswith(f"/{result['sessionId']}/offer")
    assert captured[0]["body"]["sourceNodeId"] == "node-user"
    assert captured[-1]["url"].endswith(f"/{result['sessionId']}/batch-envelopes")


def test_dispatch_cai_owned_transport_execution_dag_signs_offer_and_first_batch() -> None:
    public_key_b64, signing_seed_b64 = _signing_material()
    captured: list[dict[str, object]] = []

    def fake_urlopen(request, timeout: float):
        body = json.loads(request.data.decode("utf-8"))
        captured.append({"url": request.full_url, "body": body})
        return _FakeResponse({"status": "ok", "sessionId": body.get("sessionId")})

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        result = dispatch_cai_owned_transport_execution_dag(
            instance_id="instance-dispatch-signed",
            requester_node_id="node-user",
            executor_node_ids=["node-a"],
            peer_cai_urls_by_node={
                "node-user": ["http://node-user:52415"],
                "node-a": ["http://node-a:52415"],
            },
            initial_payload=b"user-prompt",
            total_layer_count=14,
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-dispatch-signed",
            signing_material={
                "public_key_b64": public_key_b64,
                "signing_seed_b64": signing_seed_b64,
                "wallet_id": "wallet-user",
                "address": "abcd1234",
            },
            policy=policy,
        )

    offer_body = captured[0]["body"]
    initial_body = captured[1]["body"]
    offer_valid, offer_error = validate_cai_owned_transport_session_offer(
        offer_body,
        session_id=result["sessionId"],
        local_node_id="node-a",
        require_signature=True,
    )
    batch_valid, batch_error = validate_cai_owned_transport_batch_envelope(
        initial_body,
        session_id=result["sessionId"],
        participant_node_ids=["node-user", "node-a"],
        require_signature=True,
    )

    assert offer_valid is True
    assert offer_error is None
    assert batch_valid is True
    assert batch_error is None
    assert offer_body["signerNodeId"] == "node-user"
    assert initial_body["signerNodeId"] == "node-user"
    assert offer_body["signature"]["signer_wallet_id"] == "wallet-user"
    assert initial_body["signature"]["public_key_b64"] == public_key_b64


def test_dispatch_cai_owned_transport_execution_dag_resumes_sent_initial_batch() -> None:
    captured: list[str] = []

    def fake_urlopen(request, timeout: float):
        body = json.loads(request.data.decode("utf-8"))
        captured.append(request.full_url)
        return _FakeResponse({"status": "ok", "sessionId": body.get("sessionId")})

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        with patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
            first = dispatch_cai_owned_transport_execution_dag(
                instance_id="instance-dispatch-resume",
                requester_node_id="node-user",
                executor_node_ids=["node-a", "node-b"],
                peer_cai_urls_by_node={
                    "node-user": ["http://node-user:52415"],
                    "node-a": ["http://node-a:52415"],
                    "node-b": ["http://node-b:52415"],
                },
                initial_payload=b"user-prompt",
                total_layer_count=28,
                model_id="cai-network/Qwen3-0.6B-GGUF",
                task_id="task-dispatch-resume",
                policy=policy,
            )

        resume_requests: list[tuple[str, str]] = []

        def resume_urlopen(request, timeout: float):
            resume_requests.append((request.get_method(), request.full_url))
            if request.get_method() == "GET" and request.full_url.endswith(
                "/v1/cai/transport/sessions"
            ):
                return _FakeResponse({"sessions": []})
            raise AssertionError("Resumed dispatch should not re-submit work.")

        with patch("cai_compute_chain.decentralized_compute.urlopen", resume_urlopen):
            resumed = dispatch_cai_owned_transport_execution_dag(
                instance_id="instance-dispatch-resume",
                requester_node_id="node-user",
                executor_node_ids=["node-a", "node-b"],
                peer_cai_urls_by_node={
                    "node-user": ["http://node-user:52415"],
                    "node-a": ["http://node-a:52415"],
                    "node-b": ["http://node-b:52415"],
                },
                initial_payload=b"user-prompt",
                total_layer_count=28,
                model_id="cai-network/Qwen3-0.6B-GGUF",
                task_id="task-dispatch-resume",
                policy=policy,
            )

        records = list_cai_owned_transport_sessions(policy)

    assert first["status"] == "dispatched"
    assert resumed["status"] == "resumed"
    assert resumed["resumeReason"] == "initial_batch_already_dispatched"
    assert resumed["initialDispatchResponse"]["status"] == "ok"
    assert captured == [
        f"http://node-a:52415/v1/cai/transport/sessions/{first['sessionId']}/offer",
        f"http://node-b:52415/v1/cai/transport/sessions/{first['sessionId']}/offer",
        (
            "http://node-a:52415/v1/cai/transport/sessions/"
            f"{first['sessionId']}/batch-envelopes"
        ),
    ]
    assert len(records) == 1
    assert len(records[0].dispatch_records) == 1
    assert records[0].dispatch_records[0]["status"] == "sent"
    assert resume_requests
    assert all(method == "GET" for method, _url in resume_requests)


def test_dispatch_cai_owned_transport_execution_dag_resume_completed_session() -> None:
    def fake_urlopen(request, timeout: float):
        body = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"status": "ok", "sessionId": body.get("sessionId")})

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        with patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
            first = dispatch_cai_owned_transport_execution_dag(
                instance_id="instance-dispatch-completed-resume",
                requester_node_id="node-user",
                executor_node_ids=["node-a"],
                peer_cai_urls_by_node={"node-a": ["http://node-a:52415"]},
                initial_payload=b"user-prompt",
                total_layer_count=14,
                model_id="cai-network/Qwen3-0.6B-GGUF",
                task_id="task-dispatch-completed-resume",
                policy=policy,
            )

        records = list_cai_owned_transport_sessions(policy)
        records[0].status = "completed"
        records[0].completed_at = "2026-05-03T00:00:00+00:00"
        save_cai_owned_transport_sessions(records, policy)

        def fail_urlopen(request, timeout: float):
            raise AssertionError("Completed resume should not touch the network.")

        with patch("cai_compute_chain.decentralized_compute.urlopen", fail_urlopen):
            resumed = dispatch_cai_owned_transport_execution_dag(
                instance_id="instance-dispatch-completed-resume",
                requester_node_id="node-user",
                executor_node_ids=["node-a"],
                peer_cai_urls_by_node={"node-a": ["http://node-a:52415"]},
                initial_payload=b"user-prompt",
                total_layer_count=14,
                model_id="cai-network/Qwen3-0.6B-GGUF",
                task_id="task-dispatch-completed-resume",
                policy=policy,
            )

    assert first["status"] == "dispatched"
    assert resumed["status"] == "completed"
    assert resumed["resumeReason"] == "local_session_already_completed"
    assert resumed["recovery"]["checkedRemote"] is False


def test_dispatch_cai_owned_transport_execution_dag_resume_checks_remote_completion() -> None:
    def fake_urlopen(request, timeout: float):
        body = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"status": "ok", "sessionId": body.get("sessionId")})

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        with patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
            first = dispatch_cai_owned_transport_execution_dag(
                instance_id="instance-dispatch-remote-completed",
                requester_node_id="node-user",
                executor_node_ids=["node-a"],
                peer_cai_urls_by_node={"node-a": ["http://node-a:52415"]},
                initial_payload=b"user-prompt",
                total_layer_count=14,
                model_id="cai-network/Qwen3-0.6B-GGUF",
                task_id="task-dispatch-remote-completed",
                policy=policy,
            )

        resume_requests: list[tuple[str, str]] = []

        def resume_urlopen(request, timeout: float):
            resume_requests.append((request.get_method(), request.full_url))
            if request.get_method() == "GET" and request.full_url.endswith(
                "/v1/cai/transport/sessions"
            ):
                return _FakeResponse(
                    {
                        "sessions": [
                            {
                                "sessionId": first["sessionId"],
                                "status": "completed",
                                "proof": {},
                                "batchRecords": [
                                    {
                                        "batchId": "caibatch_final",
                                        "metadata": {"finalOutput": True},
                                    }
                                ],
                                "shardReceipts": [{"batchId": "caibatch_a"}],
                                "completedAt": "2026-05-03T00:00:00+00:00",
                            }
                        ]
                    }
                )
            raise AssertionError("Remote completion recovery should not POST work.")

        with patch("cai_compute_chain.decentralized_compute.urlopen", resume_urlopen):
            resumed = dispatch_cai_owned_transport_execution_dag(
                instance_id="instance-dispatch-remote-completed",
                requester_node_id="node-user",
                executor_node_ids=["node-a"],
                peer_cai_urls_by_node={"node-a": ["http://node-a:52415"]},
                initial_payload=b"user-prompt",
                total_layer_count=14,
                model_id="cai-network/Qwen3-0.6B-GGUF",
                task_id="task-dispatch-remote-completed",
                policy=policy,
            )

    assert resumed["status"] == "resumed"
    assert resumed["resumeReason"] == "remote_session_completed"
    assert resumed["recovery"]["checkedRemote"] is True
    assert resumed["recovery"]["remoteSessions"][0]["sessionStatus"] == "completed"
    assert resume_requests
    assert all(method == "GET" for method, _url in resume_requests)


def test_dispatch_cai_owned_transport_execution_dag_retries_prepared_initial_batch() -> None:
    public_key_b64, signing_seed_b64 = _signing_material()
    signing_material = {
        "public_key_b64": public_key_b64,
        "signing_seed_b64": signing_seed_b64,
        "wallet_id": "wallet-user",
        "address": "abcd1234",
    }
    first_attempt_urls: list[str] = []
    second_attempt_urls: list[str] = []

    def failing_urlopen(request, timeout: float):
        body = json.loads(request.data.decode("utf-8"))
        first_attempt_urls.append(request.full_url)
        if request.full_url.endswith("/batch-envelopes"):
            raise TimeoutError("temporary network failure")
        return _FakeResponse({"status": "ok", "sessionId": body.get("sessionId")})

    def succeeding_urlopen(request, timeout: float):
        body = json.loads(request.data.decode("utf-8"))
        second_attempt_urls.append(request.full_url)
        return _FakeResponse({"status": "ok", "sessionId": body.get("sessionId")})

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        with patch("cai_compute_chain.decentralized_compute.urlopen", failing_urlopen):
            with pytest.raises(ValueError, match="temporary network failure"):
                dispatch_cai_owned_transport_execution_dag(
                    instance_id="instance-dispatch-retry-prepared",
                    requester_node_id="node-user",
                    executor_node_ids=["node-a", "node-b"],
                    peer_cai_urls_by_node={
                        "node-user": ["http://node-user:52415"],
                        "node-a": ["http://node-a:52415"],
                        "node-b": ["http://node-b:52415"],
                    },
                    initial_payload=b"user-prompt",
                    total_layer_count=28,
                    model_id="cai-network/Qwen3-0.6B-GGUF",
                    task_id="task-dispatch-retry-prepared",
                    signing_material=signing_material,
                    policy=policy,
                )
        prepared = list_cai_owned_transport_sessions(policy)[0]

        with patch(
            "cai_compute_chain.decentralized_compute.urlopen",
            succeeding_urlopen,
        ):
            retried = dispatch_cai_owned_transport_execution_dag(
                instance_id="instance-dispatch-retry-prepared",
                requester_node_id="node-user",
                executor_node_ids=["node-a", "node-b"],
                peer_cai_urls_by_node={
                    "node-user": ["http://node-user:52415"],
                    "node-a": ["http://node-a:52415"],
                    "node-b": ["http://node-b:52415"],
                },
                initial_payload=b"user-prompt",
                total_layer_count=28,
                model_id="cai-network/Qwen3-0.6B-GGUF",
                task_id="task-dispatch-retry-prepared",
                signing_material=signing_material,
                policy=policy,
            )

        records = list_cai_owned_transport_sessions(policy)

    assert prepared.dispatch_records[0]["status"] == "prepared"
    assert retried["status"] == "dispatched"
    assert first_attempt_urls[-1].endswith("/batch-envelopes")
    assert second_attempt_urls[-1].endswith("/batch-envelopes")
    assert len(records) == 1
    assert len(records[0].dispatch_records) == 1
    assert records[0].dispatch_records[0]["status"] == "sent"


def test_dispatch_cai_owned_transport_execution_dag_checks_executor_readiness() -> None:
    captured: list[dict[str, object]] = []

    def fake_urlopen(request, timeout: float):
        captured.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "method": request.get_method(),
            }
        )
        if request.full_url.endswith("/v1/cai/summary"):
            return _FakeResponse(
                {
                    "worker": {
                        "workerEnabled": True,
                        "allowedModelIds": ["cai-network/Qwen3-0.6B-GGUF"],
                        "readiness": {
                            "caiOwnedTransport": {
                                "protocol": CAI_OWNED_TRANSPORT_PROTOCOL,
                                "protocolVersion": 1,
                                "runtimeReady": True,
                                "status": "ready",
                                "runtimeVersion": "cai-owned-runtime/0.1",
                                "adapterVersion": "deterministic-bytes/0.1",
                            }
                        },
                    }
                }
            )
        body = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"status": "ok", "sessionId": body.get("sessionId")})

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        result = dispatch_cai_owned_transport_execution_dag(
            instance_id="instance-dispatch-readiness",
            requester_node_id="node-user",
            executor_node_ids=["node-a", "node-b"],
            peer_cai_urls_by_node={
                "node-a": ["http://node-a:52415"],
                "node-b": ["http://node-b:52415"],
            },
            initial_payload=b"user-prompt",
            total_layer_count=28,
            model_id="cai-network/Qwen3-0.6B-GGUF",
            require_executor_readiness=True,
            require_cai_owned_runtime_ready=True,
            policy=policy,
        )

    summary_urls = [
        item["url"]
        for item in captured
        if str(item["url"]).endswith("/v1/cai/summary")
    ]

    assert result["status"] == "dispatched"
    assert result["readinessPreflight"]["status"] == "ready"
    assert result["readinessPreflight"]["executorNodeIds"] == ["node-a", "node-b"]
    assert summary_urls == [
        "http://node-a:52415/v1/cai/summary",
        "http://node-b:52415/v1/cai/summary",
    ]
    assert captured[0]["method"] == "GET"
    assert captured[2]["url"].endswith(f"/{result['sessionId']}/offer")


def test_dispatch_cai_owned_transport_rejects_incompatible_runtime_version() -> None:
    def fake_urlopen(request, timeout: float):
        if request.full_url.endswith("/v1/cai/summary"):
            return _FakeResponse(
                {
                    "worker": {
                        "workerEnabled": True,
                        "allowedModelIds": ["cai-network/Qwen3-0.6B-GGUF"],
                        "readiness": {
                            "caiOwnedTransport": {
                                "protocol": CAI_OWNED_TRANSPORT_PROTOCOL,
                                "protocolVersion": 999,
                                "runtimeReady": True,
                                "status": "ready",
                                "runtimeVersion": "cai-owned-runtime/0.1",
                                "adapterVersion": "deterministic-bytes/0.1",
                            }
                        },
                    }
                }
            )
        body = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"status": "ok", "sessionId": body.get("sessionId")})

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        with pytest.raises(ValueError) as exc_info:
            dispatch_cai_owned_transport_execution_dag(
                instance_id="instance-dispatch-bad-version",
                requester_node_id="node-user",
                executor_node_ids=["node-a"],
                peer_cai_urls_by_node={"node-a": ["http://node-a:52415"]},
                initial_payload=b"user-prompt",
                total_layer_count=28,
                model_id="cai-network/Qwen3-0.6B-GGUF",
                require_executor_readiness=True,
                require_cai_owned_runtime_ready=True,
                policy=policy,
            )

    assert "protocol version is unsupported" in str(exc_info.value)


def test_dispatch_cai_owned_transport_execution_dag_blocks_disabled_executor() -> None:
    captured: list[str] = []

    def fake_urlopen(request, timeout: float):
        captured.append(request.full_url)
        if request.full_url.endswith("/v1/cai/summary"):
            return _FakeResponse({"worker": {"workerEnabled": False}})
        raise AssertionError("Dispatch should stop before submitting work.")

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        with pytest.raises(ValueError, match="readiness preflight failed"):
            dispatch_cai_owned_transport_execution_dag(
                instance_id="instance-dispatch-readiness-blocked",
                requester_node_id="node-user",
                executor_node_ids=["node-a"],
                peer_cai_urls_by_node={"node-a": ["http://node-a:52415"]},
                initial_payload=b"user-prompt",
                total_layer_count=14,
                model_id="cai-network/Qwen3-0.6B-GGUF",
                require_executor_readiness=True,
                policy=policy,
            )

    assert captured == ["http://node-a:52415/v1/cai/summary"]


def test_dispatch_cai_owned_transport_execution_dag_checks_shard_inventory() -> None:
    captured: list[str] = []

    def fake_urlopen(request, timeout: float):
        captured.append(request.full_url)
        if request.full_url.endswith("/v1/cai/summary"):
            if "node-a" in request.full_url:
                layer_start, layer_end = 0, 14
            else:
                layer_start, layer_end = 14, 28
            return _FakeResponse(
                {
                    "worker": {
                        "workerEnabled": True,
                        "allowedModelIds": ["cai-network/Qwen3-0.6B-GGUF"],
                        "modelShardInventory": {
                            "cai-network/Qwen3-0.6B-GGUF": {
                                "shards": [
                                    {
                                        "layerStart": layer_start,
                                        "layerEnd": layer_end,
                                        "status": "ready",
                                    }
                                ]
                            }
                        },
                    }
                }
            )
        body = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"status": "ok", "sessionId": body.get("sessionId")})

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        result = dispatch_cai_owned_transport_execution_dag(
            instance_id="instance-dispatch-shard-ready",
            requester_node_id="node-user",
            executor_node_ids=["node-a", "node-b"],
            peer_cai_urls_by_node={
                "node-a": ["http://node-a:52415"],
                "node-b": ["http://node-b:52415"],
            },
            initial_payload=b"user-prompt",
            total_layer_count=28,
            model_id="cai-network/Qwen3-0.6B-GGUF",
            require_executor_readiness=True,
            require_executor_shard_readiness=True,
            policy=policy,
        )

    node_audit = result["readinessPreflight"]["nodeAudits"][0]

    assert result["readinessPreflight"]["status"] == "ready"
    assert result["readinessPreflight"]["requireShardReadiness"] is True
    assert node_audit["modelShardReadiness"]["reason"] == (
        "required_shard_ranges_ready"
    )
    assert node_audit["modelShardReadiness"]["requiredRanges"] == [
        {"layerStart": 0, "layerEnd": 14}
    ]
    assert captured[0] == "http://node-a:52415/v1/cai/summary"
    assert captured[2].endswith(f"/{result['sessionId']}/offer")


def test_dispatch_cai_owned_transport_execution_dag_blocks_missing_shard_range() -> None:
    captured: list[str] = []

    def fake_urlopen(request, timeout: float):
        captured.append(request.full_url)
        if request.full_url.endswith("/v1/cai/summary"):
            return _FakeResponse(
                {
                    "worker": {
                        "workerEnabled": True,
                        "allowedModelIds": ["cai-network/Qwen3-0.6B-GGUF"],
                        "modelShardInventory": {
                            "cai-network/Qwen3-0.6B-GGUF": {
                                "shards": [
                                    {
                                        "layerStart": 0,
                                        "layerEnd": 7,
                                        "status": "ready",
                                    }
                                ]
                            }
                        },
                    }
                }
            )
        raise AssertionError("Dispatch should stop before submitting work.")

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        with pytest.raises(ValueError, match="assigned model shard range"):
            dispatch_cai_owned_transport_execution_dag(
                instance_id="instance-dispatch-shard-blocked",
                requester_node_id="node-user",
                executor_node_ids=["node-a"],
                peer_cai_urls_by_node={"node-a": ["http://node-a:52415"]},
                initial_payload=b"user-prompt",
                total_layer_count=14,
                model_id="cai-network/Qwen3-0.6B-GGUF",
                require_executor_readiness=True,
                require_executor_shard_readiness=True,
                policy=policy,
            )

    assert captured == ["http://node-a:52415/v1/cai/summary"]


def test_preflight_blocks_unverified_encrypted_shard_cache_inventory() -> None:
    def fake_urlopen(request, timeout: float):
        return _FakeResponse(
            {
                "worker": {
                    "workerEnabled": True,
                    "allowedModelIds": ["cai-network/Qwen3-0.6B-GGUF"],
                    "modelShardInventory": {
                        "cai-network/Qwen3-0.6B-GGUF": {
                            "shards": [
                                {
                                    "layerStart": 0,
                                    "layerEnd": 14,
                                    "status": "cached",
                                    "chunkManifestVerified": False,
                                    "cacheVerified": True,
                                    "encryptedAtRest": True,
                                }
                            ]
                        }
                    },
                }
            }
        )

    with patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        result = preflight_cai_owned_transport_executor_readiness(
            executor_node_ids=["node-a"],
            peer_cai_urls_by_node={"node-a": ["http://node-a:52415"]},
            model_id="cai-network/Qwen3-0.6B-GGUF",
            required_shard_ranges_by_node={
                "node-a": [{"layerStart": 0, "layerEnd": 14}]
            },
        )

    audit = result["nodeAudits"][0]["modelShardReadiness"]
    assert result["status"] == "failed"
    assert audit["reason"] == "model_shard_range_not_ready"
    assert audit["blockedRanges"][0]["reason"] == "chunk_manifest_not_verified"


def test_preflight_accepts_verified_encrypted_cache_with_key() -> None:
    def fake_urlopen(request, timeout: float):
        return _FakeResponse(
            {
                "worker": {
                    "workerEnabled": True,
                    "allowedModelIds": ["cai-network/Qwen3-0.6B-GGUF"],
                    "modelShardInventory": {
                        "cai-network/Qwen3-0.6B-GGUF": {
                            "shards": [
                                {
                                    "layerStart": 0,
                                    "layerEnd": 14,
                                    "status": "cached",
                                    "chunkManifestVerified": True,
                                    "cacheVerified": True,
                                    "encryptedAtRest": True,
                                    "decryptionKeyAvailable": True,
                                    "downloadDeadlineAt": "2999-01-01T00:00:00Z",
                                }
                            ]
                        }
                    },
                }
            }
        )

    with patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        result = preflight_cai_owned_transport_executor_readiness(
            executor_node_ids=["node-a"],
            peer_cai_urls_by_node={"node-a": ["http://node-a:52415"]},
            model_id="cai-network/Qwen3-0.6B-GGUF",
            required_shard_ranges_by_node={
                "node-a": [{"layerStart": 0, "layerEnd": 14}]
            },
        )

    audit = result["nodeAudits"][0]["modelShardReadiness"]
    assert result["status"] == "ready"
    assert audit["reason"] == "required_shard_ranges_ready"


def test_preflight_blocks_insufficient_executor_ram_headroom() -> None:
    def fake_urlopen(request, timeout: float):
        return _FakeResponse(
            {
                "worker": {
                    "workerEnabled": True,
                    "allowedModelIds": ["cai-network/Qwen3-0.6B-GGUF"],
                    "resources": {
                        "ramAvailableBytes": 128 * 1024 * 1024,
                        "ramBytes": 8 * 1024 * 1024 * 1024,
                    },
                }
            }
        )

    with patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        result = preflight_cai_owned_transport_executor_readiness(
            executor_node_ids=["node-a"],
            peer_cai_urls_by_node={"node-a": ["http://node-a:52415"]},
            model_id="cai-network/Qwen3-0.6B-GGUF",
        )

    audit = result["nodeAudits"][0]["resourceAudit"]
    assert result["status"] == "failed"
    assert audit["reason"] == "resource_headroom_insufficient"
    assert audit["minimumRamHeadroomBytes"] == 256 * 1024 * 1024
    assert audit["insufficientResources"] == ["ram"]


def test_preflight_blocks_insufficient_executor_vram_headroom() -> None:
    def fake_urlopen(request, timeout: float):
        return _FakeResponse(
            {
                "worker": {
                    "workerEnabled": True,
                    "allowedModelIds": ["cai-network/Qwen3-0.6B-GGUF"],
                    "resourceSummary": {
                        "ramAvailable": {"inBytes": 1024 * 1024 * 1024},
                        "vramAvailableBytes": 32 * 1024 * 1024,
                    },
                }
            }
        )

    with patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        result = preflight_cai_owned_transport_executor_readiness(
            executor_node_ids=["node-a"],
            peer_cai_urls_by_node={"node-a": ["http://node-a:52415"]},
            model_id="cai-network/Qwen3-0.6B-GGUF",
            minimum_vram_headroom_bytes=64 * 1024 * 1024,
        )

    audit = result["nodeAudits"][0]["resourceAudit"]
    assert result["status"] == "failed"
    assert audit["reason"] == "resource_headroom_insufficient"
    assert audit["minimumVramHeadroomBytes"] == 64 * 1024 * 1024
    assert audit["insufficientResources"] == ["vram"]


def test_preflight_uses_node_capabilities_when_summary_lacks_runtime_readiness() -> None:
    requested_urls: list[str] = []

    def fake_urlopen(request, timeout: float):
        url = str(getattr(request, "full_url", request))
        requested_urls.append(url)
        if url.endswith("/v1/cai/node-capabilities"):
            return _FakeResponse(
                {
                    "records": [
                        {
                            "nodeId": "node-a",
                            "workerEnabled": True,
                            "readiness": {
                                "caiOwnedTransport": {
                                    "protocol": CAI_OWNED_TRANSPORT_PROTOCOL,
                                    "protocolVersion": 1,
                                    "implemented": True,
                                    "runtimeReady": True,
                                    "status": "ready",
                                    "runtimeVersion": "0.1.0",
                                    "adapterVersion": "native-bridge-v1",
                                }
                            },
                        }
                    ]
                }
            )
        return _FakeResponse(
            {
                "worker": {
                    "workerEnabled": True,
                    "allowedModelIds": ["cai-network/Qwen3-0.6B-GGUF"],
                    "resources": {
                        "ramAvailableBytes": 1024 * 1024 * 1024,
                        "ramBytes": 8 * 1024 * 1024 * 1024,
                    },
                }
            }
        )

    with patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        result = preflight_cai_owned_transport_executor_readiness(
            executor_node_ids=["node-a"],
            peer_cai_urls_by_node={"node-a": ["http://node-a:52415"]},
            model_id="cai-network/Qwen3-0.6B-GGUF",
            require_cai_owned_runtime_ready=True,
        )

    audit = result["nodeAudits"][0]
    assert result["status"] == "ready"
    assert audit["caiOwnedTransport"]["runtimeReady"] is True
    assert audit["nodeCapabilityReadinessAttempt"]["status"] == "ok"
    assert audit["nodeCapabilityReadinessAttempt"]["reason"] == "readiness_attached"
    assert requested_urls == [
        "http://node-a:52415/v1/cai/summary",
        "http://node-a:52415/v1/cai/node-capabilities",
    ]


def test_preflight_accepts_production_adapter_ready_without_live_runtime_proof() -> None:
    def fake_urlopen(request, timeout: float):
        assert request.full_url == "http://node-a:52415/v1/cai/summary"
        return _FakeResponse(
            {
                "worker": {
                    "workerEnabled": True,
                    "allowedModelIds": ["cai-network/Qwen3-0.6B-GGUF"],
                    "readiness": {
                        "caiOwnedTransport": {
                            "protocol": CAI_OWNED_TRANSPORT_PROTOCOL,
                            "protocolVersion": 1,
                            "implemented": True,
                            "runtimeReady": False,
                            "status": "test_adapter_ready",
                            "runtimeVersion": "cai-owned-runtime/0.1",
                            "adapterId": "llama.cpp-external-shard",
                            "adapterVersion": "llama.cpp-external-shard/0.1",
                            "llmShardSelfTest": {
                                **_production_llm_self_test(),
                                "adapterId": "llama.cpp-external-shard",
                                "adapterVersion": "llama.cpp-external-shard/0.1",
                            },
                        }
                    },
                }
            }
        )

    with patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        result = preflight_cai_owned_transport_executor_readiness(
            executor_node_ids=["node-a"],
            peer_cai_urls_by_node={"node-a": ["http://node-a:52415"]},
            model_id="cai-network/Qwen3-0.6B-GGUF",
            require_cai_owned_runtime_ready=True,
        )

    audit = result["nodeAudits"][0]
    assert result["status"] == "ready"
    assert audit["reason"] == "ready"
    assert audit["caiOwnedTransport"]["status"] == "test_adapter_ready"
    assert audit["caiOwnedTransport"]["runtimeReady"] is False


def test_preflight_uses_state_payload_when_direct_summary_times_out() -> None:
    def fake_urlopen(request, timeout: float):
        raise TimeoutError("direct summary timed out")

    state_payload = {
        "nodeIdentities": {
            "node-b": {
                "workerEnabled": True,
                "workerRewardAddress": "bbbb1234bbbb1234bbbb1234bbbb1234",
                "workerAllowedModelIds": ["cai-network/Qwen3-0.6B-GGUF"],
                "totalVramBytes": 128 * 1024 * 1024,
                "readiness": {
                    "caiOwnedTransport": {
                        "protocol": CAI_OWNED_TRANSPORT_PROTOCOL,
                        "protocolVersion": 1,
                        "implemented": True,
                        "runtimeReady": False,
                        "status": "test_adapter_ready",
                    }
                },
            }
        },
        "nodeMemory": {
            "node-b": {
                "ramAvailable": {"inBytes": 2 * 1024 * 1024 * 1024},
                "ramTotal": {"inBytes": 4 * 1024 * 1024 * 1024},
            }
        },
    }

    with patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        result = preflight_cai_owned_transport_executor_readiness(
            executor_node_ids=["node-b"],
            peer_cai_urls_by_node={
                "node-b": [
                    "http://node-b-bad:52415",
                    "cai-overlay:http://relay:52415?targetNodeId=node-b&relayRole=bootstrap"
                ]
            },
            model_id="cai-network/Qwen3-0.6B-GGUF",
            state_payload=state_payload,
            minimum_ram_headroom_bytes=256 * 1024 * 1024,
        )

    audit = result["nodeAudits"][0]
    assert result["status"] == "ready"
    assert audit["summarySource"] == "state_payload"
    assert audit["overlayPreflightDeferredToDispatch"] is True
    assert audit["resourceAudit"]["reason"] == "resource_headroom_ok"
    assert audit["attempts"][0]["status"] == "failed"
    assert audit["attempts"][1]["routeClass"] == "overlay_bootstrap"


def test_preflight_keeps_runtime_ready_strict_for_overlay_state_fallback() -> None:
    state_payload = {
        "nodeIdentities": {
            "node-b": {
                "workerEnabled": True,
                "workerRewardAddress": "bbbb1234bbbb1234bbbb1234bbbb1234",
                "readiness": {
                    "caiOwnedTransport": {
                        "protocol": CAI_OWNED_TRANSPORT_PROTOCOL,
                        "protocolVersion": 1,
                        "implemented": True,
                        "runtimeReady": False,
                        "status": "test_adapter_ready",
                    }
                },
            }
        }
    }

    result = preflight_cai_owned_transport_executor_readiness(
        executor_node_ids=["node-b"],
        peer_cai_urls_by_node={
            "node-b": [
                "cai-overlay:http://relay:52415?targetNodeId=node-b&relayRole=bootstrap"
            ]
        },
        model_id="cai-network/Qwen3-0.6B-GGUF",
        require_cai_owned_runtime_ready=True,
        state_payload=state_payload,
    )

    audit = result["nodeAudits"][0]
    assert result["status"] == "failed"
    assert audit["reason"] == "no_direct_summary_url"


def test_preflight_blocks_expired_download_deadline() -> None:
    def fake_urlopen(request, timeout: float):
        return _FakeResponse(
            {
                "worker": {
                    "workerEnabled": True,
                    "allowedModelIds": ["cai-network/Qwen3-0.6B-GGUF"],
                    "modelShardInventory": {
                        "cai-network/Qwen3-0.6B-GGUF": {
                            "downloadableRanges": [
                                {
                                    "layerStart": 0,
                                    "layerEnd": 14,
                                    "status": "downloading",
                                    "canLoadBeforeDeadline": True,
                                    "downloadDeadlineAt": "2000-01-01T00:00:00Z",
                                }
                            ]
                        }
                    },
                }
            }
        )

    with patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        result = preflight_cai_owned_transport_executor_readiness(
            executor_node_ids=["node-a"],
            peer_cai_urls_by_node={"node-a": ["http://node-a:52415"]},
            model_id="cai-network/Qwen3-0.6B-GGUF",
            required_shard_ranges_by_node={
                "node-a": [{"layerStart": 0, "layerEnd": 14}]
            },
        )

    audit = result["nodeAudits"][0]["modelShardReadiness"]
    assert result["status"] == "failed"
    assert audit["blockedRanges"][0]["reason"] == "download_deadline_expired"


def test_dispatch_cai_owned_transport_execution_dag_blocks_missing_requester_route() -> None:
    def fake_urlopen(request, timeout: float):
        raise AssertionError("Dispatch should stop before submitting work.")

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        with pytest.raises(ValueError, match="route preflight failed"):
            dispatch_cai_owned_transport_execution_dag(
                instance_id="instance-dispatch-route-blocked",
                requester_node_id="node-user",
                executor_node_ids=["node-a"],
                peer_cai_urls_by_node={"node-a": ["http://node-a:52415"]},
                initial_payload=b"user-prompt",
                total_layer_count=14,
                model_id="cai-network/Qwen3-0.6B-GGUF",
                require_data_plane_route=True,
                policy=policy,
            )


def test_dispatch_cai_owned_transport_execution_dag_blocks_unproven_route_health() -> None:
    def fake_urlopen(request, timeout: float):
        raise AssertionError("Dispatch should stop before submitting work.")

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        with pytest.raises(ValueError, match="RouteHealth"):
            dispatch_cai_owned_transport_execution_dag(
                instance_id="instance-dispatch-route-health-blocked",
                requester_node_id="node-user",
                executor_node_ids=["node-a"],
                peer_cai_urls_by_node={
                    "node-user": ["http://node-user:52415"],
                    "node-a": ["http://node-a:52415"],
                },
                initial_payload=b"user-prompt",
                total_layer_count=14,
                model_id="cai-network/Qwen3-0.6B-GGUF",
                require_proven_data_plane_route=True,
                policy=policy,
            )


def test_dispatch_cai_owned_transport_execution_dag_requires_directional_route_health() -> None:
    def fake_urlopen(request, timeout: float):
        raise AssertionError("Dispatch should stop before submitting work.")

    route_records = [
        _route_record("node-a", "node-user", route_type="direct_data"),
    ]
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        with pytest.raises(ValueError, match="RouteHealth"):
            dispatch_cai_owned_transport_execution_dag(
                instance_id="instance-dispatch-asymmetric-route-health",
                requester_node_id="node-user",
                executor_node_ids=["node-a"],
                peer_cai_urls_by_node={
                    "node-user": ["http://node-user:52415"],
                    "node-a": ["http://node-a:52415"],
                },
                initial_payload=b"user-prompt",
                total_layer_count=14,
                model_id="cai-network/Qwen3-0.6B-GGUF",
                require_proven_data_plane_route=True,
                route_health_records=route_records,
                policy=policy,
            )


def test_dispatch_cai_owned_transport_execution_dag_accepts_proven_route_health() -> None:
    captured: list[str] = []

    def fake_urlopen(request, timeout: float):
        body = json.loads(request.data.decode("utf-8"))
        captured.append(request.full_url)
        return _FakeResponse({"status": "ok", "sessionId": body.get("sessionId")})

    route_records = [
        _route_record("node-user", "node-a"),
        _route_record("node-user", "node-b"),
        _route_record("node-a", "node-b"),
        _route_record("node-b", "node-user"),
    ]
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        result = dispatch_cai_owned_transport_execution_dag(
            instance_id="instance-dispatch-route-health-ready",
            requester_node_id="node-user",
            executor_node_ids=["node-a", "node-b"],
            peer_cai_urls_by_node={
                "node-user": ["http://node-user:52415"],
                "node-a": ["http://node-a:52415"],
                "node-b": ["http://node-b:52415"],
            },
            initial_payload=b"user-prompt",
            total_layer_count=28,
            model_id="cai-network/Qwen3-0.6B-GGUF",
            require_proven_data_plane_route=True,
            route_health_records=route_records,
            policy=policy,
        )

    assert result["routePreflight"]["status"] == "ready"
    assert result["routePreflight"]["requireRouteHealth"] is True
    assert result["routePreflight"]["requiredRouteHops"] == [
        {"sourceNodeId": "node-user", "sinkNodeId": "node-a"},
        {"sourceNodeId": "node-user", "sinkNodeId": "node-b"},
        {"sourceNodeId": "node-a", "sinkNodeId": "node-b"},
        {"sourceNodeId": "node-b", "sinkNodeId": "node-user"},
    ]
    assert all(
        item["ready"] for item in result["routePreflight"]["routeHealthAudits"]
    )
    assert len(captured) == 3


def test_dispatch_cai_owned_transport_ignores_failed_llama_rpc_when_cai_route_ready() -> None:
    captured: list[str] = []

    def fake_urlopen(request, timeout: float):
        body = json.loads(request.data.decode("utf-8"))
        captured.append(request.full_url)
        return _FakeResponse({"status": "ok", "sessionId": body.get("sessionId")})

    route_records = [
        _route_record("node-user", "node-a", route_type="direct_data"),
        _rpc_record("node-user", "node-a", reachable=False),
        _route_record("node-user", "node-b", route_type="direct_data"),
        _route_record("node-a", "node-b", route_type="direct_data"),
        _route_record("node-b", "node-user", route_type="overlay_peer"),
    ]
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        result = dispatch_cai_owned_transport_execution_dag(
            instance_id="instance-dispatch-cai-route-ready-rpc-failed",
            requester_node_id="node-user",
            executor_node_ids=["node-a", "node-b"],
            peer_cai_urls_by_node={
                "node-user": ["http://node-user:52415"],
                "node-a": ["http://node-a:52415"],
                "node-b": ["http://node-b:52415"],
            },
            initial_payload=b"user-prompt",
            total_layer_count=28,
            model_id="cai-network/Qwen3-0.6B-GGUF",
            require_proven_data_plane_route=True,
            route_health_records=route_records,
            policy=policy,
        )

    assert result["routePreflight"]["status"] == "ready"
    first_hop = result["routePreflight"]["routeHealthAudits"][0]
    assert first_hop["sourceNodeId"] == "node-user"
    assert first_hop["sinkNodeId"] == "node-a"
    assert first_hop["routeHealthScore"] == 4
    assert len(captured) == 3


def test_cai_owned_route_preflight_prefers_direct_rpc_over_newer_relay() -> None:
    route_records = [
        RouteHealthRecord(
            route_id="direct-rpc",
            source_node_id="node-user",
            sink_node_id="node-a",
            route_type="llama_cpp_rpc_direct",
            endpoint_url="llama-cpp-rpc://node-a:52435",
            reachable=True,
            checked_at="2026-05-03T00:00:00+00:00",
            latency_ms=7.0,
        ),
        RouteHealthRecord(
            route_id="relay-rpc",
            source_node_id="node-user",
            sink_node_id="node-a",
            route_type="llama_cpp_rpc_relay",
            endpoint_url="relay://node-vps/node-a:52435",
            reachable=True,
            checked_at="2026-05-03T00:10:00+00:00",
            latency_ms=12.0,
            transit_node_id="node-vps",
        ),
        _route_record("node-a", "node-user", route_type="direct_data"),
    ]

    preflight = preflight_cai_owned_transport_data_plane_routes(
        requester_node_id="node-user",
        executor_node_ids=["node-a"],
        peer_cai_urls_by_node={
            "node-user": ["http://node-user:52415"],
            "node-a": ["http://node-a:52415"],
        },
        require_route_health=True,
        route_health_records=route_records,
    )

    assert preflight["status"] == "ready"
    hop = preflight["routeHealthAudits"][0]
    assert hop["routeType"] == "llama_cpp_rpc_direct"
    assert hop["transitNodeId"] is None
    assert hop["routeHealthScore"] == 5


def test_dispatch_cai_owned_transport_execution_dag_blocks_single_transit_route_health() -> None:
    def fake_urlopen(request, timeout: float):
        raise AssertionError("Dispatch should stop before submitting work.")

    route_records = [
        _route_record(
            "node-user",
            "node-a",
            route_type="relay_active",
            transit_node_id="node-vps",
        ),
        _route_record(
            "node-user",
            "node-b",
            route_type="relay_active",
            transit_node_id="node-vps",
        ),
        _route_record(
            "node-a",
            "node-b",
            route_type="relay_active",
            transit_node_id="node-vps",
        ),
        _route_record(
            "node-b",
            "node-user",
            route_type="relay_active",
            transit_node_id="node-vps",
        ),
    ]
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        with pytest.raises(ValueError, match="one transit node"):
            dispatch_cai_owned_transport_execution_dag(
                instance_id="instance-dispatch-route-health-bottleneck",
                requester_node_id="node-user",
                executor_node_ids=["node-a", "node-b"],
                peer_cai_urls_by_node={
                    "node-user": ["http://node-user:52415"],
                    "node-a": ["http://node-a:52415"],
                    "node-b": ["http://node-b:52415"],
                },
                initial_payload=b"user-prompt",
                total_layer_count=28,
                model_id="cai-network/Qwen3-0.6B-GGUF",
                route_policy={"avoidSingleTransitBottleneck": True},
                require_proven_data_plane_route=True,
                route_health_records=route_records,
                policy=policy,
            )


def test_dispatch_cai_owned_transport_execution_dag_blocks_missing_relay_quorum() -> None:
    def fake_urlopen(request, timeout: float):
        raise AssertionError("Dispatch should stop before submitting work.")

    route_records = [
        _route_record(
            "node-user",
            "node-a",
            route_type="relay_active",
            transit_node_id="node-relay-1",
        ),
        _route_record("node-user", "node-b"),
        _route_record("node-a", "node-b"),
        _route_record("node-b", "node-user"),
    ]
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        with pytest.raises(ValueError, match="at least 2 independent transit"):
            dispatch_cai_owned_transport_execution_dag(
                instance_id="instance-dispatch-route-health-quorum-blocked",
                requester_node_id="node-user",
                executor_node_ids=["node-a", "node-b"],
                peer_cai_urls_by_node={
                    "node-user": ["http://node-user:52415"],
                    "node-a": ["http://node-a:52415"],
                    "node-b": ["http://node-b:52415"],
                },
                initial_payload=b"user-prompt",
                total_layer_count=28,
                model_id="cai-network/Qwen3-0.6B-GGUF",
                route_policy={"minimumRelayQuorum": 2},
                require_proven_data_plane_route=True,
                route_health_records=route_records,
                policy=policy,
            )


def test_dispatch_cai_owned_transport_execution_dag_accepts_relay_quorum() -> None:
    captured: list[str] = []

    def fake_urlopen(request, timeout: float):
        body = json.loads(request.data.decode("utf-8"))
        captured.append(request.full_url)
        return _FakeResponse({"status": "ok", "sessionId": body.get("sessionId")})

    route_records = [
        _route_record(
            "node-user",
            "node-a",
            route_type="relay_active",
            transit_node_id="node-relay-1",
        ),
        _route_record(
            "node-user",
            "node-a",
            route_type="relay_active",
            transit_node_id="node-relay-2",
        ),
        _route_record("node-user", "node-b"),
        _route_record("node-a", "node-b"),
        _route_record("node-b", "node-user"),
    ]
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        result = dispatch_cai_owned_transport_execution_dag(
            instance_id="instance-dispatch-route-health-quorum-ready",
            requester_node_id="node-user",
            executor_node_ids=["node-a", "node-b"],
            peer_cai_urls_by_node={
                "node-user": ["http://node-user:52415"],
                "node-a": ["http://node-a:52415"],
                "node-b": ["http://node-b:52415"],
            },
            initial_payload=b"user-prompt",
            total_layer_count=28,
            model_id="cai-network/Qwen3-0.6B-GGUF",
            route_policy={"minimumRelayQuorum": 2},
            require_proven_data_plane_route=True,
            route_health_records=route_records,
            policy=policy,
        )

    assert result["routePreflight"]["status"] == "ready"
    assert result["routePreflight"]["minimumRelayQuorum"] == 2
    relay_audit = result["routePreflight"]["relayQuorumAudits"][0]
    assert relay_audit["transitNodeIds"] == ["node-relay-1", "node-relay-2"]
    assert relay_audit["ready"] is True
    assert len(captured) == 3


def test_dispatch_cai_owned_transport_execution_dag_can_compress_initial_payload() -> None:
    captured: list[dict[str, object]] = []

    def fake_urlopen(request, timeout: float):
        body = json.loads(request.data.decode("utf-8"))
        captured.append({"url": request.full_url, "body": body})
        return _FakeResponse({"status": "ok", "sessionId": body.get("sessionId")})

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        result = dispatch_cai_owned_transport_execution_dag(
            instance_id="instance-dispatch-gzip",
            requester_node_id="node-user",
            executor_node_ids=["node-a", "node-b"],
            peer_cai_urls_by_node={
                "node-a": ["http://node-a:52415"],
                "node-b": ["http://node-b:52415"],
            },
            initial_payload=b"user-prompt-" * 64,
            total_layer_count=28,
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-dispatch-gzip",
            payload_compression="gzip",
            policy=policy,
        )

    initial_body = captured[-1]["body"]

    assert result["status"] == "dispatched"
    assert initial_body["payloadCompression"] == "gzip"
    assert initial_body["payloadEncodedSizeBytes"] < initial_body["payloadSizeBytes"]
    assert cai_owned_transport_batch_payload_bytes(initial_body) == b"user-prompt-" * 64


def test_cai_owned_transport_execution_proof_validates_participants() -> None:
    proof = build_cai_owned_transport_execution_proof(
        session_id="session-1",
        instance_id="instance-1",
        participant_node_ids=["node-a", "node-b"],
        model_id="cai-network/Qwen3-0.6B-GGUF",
        activation_batch_count=2,
        decode_batch_count=3,
    )

    valid, error = validate_cai_owned_transport_execution_proof(
        proof,
        participant_node_ids=["node-a", "node-b"],
        model_id="cai-network/Qwen3-0.6B-GGUF",
    )
    wrong_valid, wrong_error = validate_cai_owned_transport_execution_proof(
        proof,
        participant_node_ids=["node-a", "node-c"],
    )
    wrong_receipt_chain = dict(proof)
    wrong_receipt_chain["shardReceipts"] = [
        {
            **dict(proof["shardReceipts"][0]),
            "network": ChainNetwork.TESTNET.value,
            "chainId": ChainNetwork.TESTNET.value,
        },
        dict(proof["shardReceipts"][1]),
    ]
    wrong_receipt_chain_valid, wrong_receipt_chain_error = (
        validate_cai_owned_transport_execution_proof(
            wrong_receipt_chain,
            participant_node_ids=["node-a", "node-b"],
        )
    )

    assert valid is True
    assert error is None
    assert proof["chainId"] == "mainnet"
    assert proof["shardReceipts"][0]["chainId"] == "mainnet"
    assert wrong_valid is False
    assert wrong_error == "CAI-owned transport proof participant set does not match."
    assert wrong_receipt_chain_valid is False
    assert wrong_receipt_chain_error == (
        "CAI-owned transport proof shard receipt is for chain 'testnet', "
        "expected 'mainnet'."
    )


def test_cai_owned_transport_execution_proof_validates_audited_receipts() -> None:
    proof = build_cai_owned_transport_execution_proof(
        session_id="session-audited",
        instance_id="instance-audited",
        participant_node_ids=["node-a", "node-b"],
        executor_node_ids=["node-a", "node-b"],
        model_id="cai-network/Qwen3-0.6B-GGUF",
        shard_receipts=[
            {
                "nodeId": "node-a",
                "network": "mainnet",
                "chainId": "mainnet",
                "status": "completed",
                "batchIds": ["caibatch_a"],
                "stageIds": ["caistage_a"],
                "hashChainSha256Hexes": ["a" * 64],
            },
            {
                "nodeId": "node-b",
                "network": "mainnet",
                "chainId": "mainnet",
                "status": "completed",
                "batchIds": ["caibatch_b"],
                "stageIds": ["caistage_b"],
                "hashChainSha256Hexes": ["b" * 64],
            },
        ],
    )
    proof["executionAudit"] = {
        "verified": True,
        "errorCount": 0,
        "errors": [],
        "processedBatchIds": ["caibatch_a", "caibatch_b"],
        "receiptBatchIds": ["caibatch_a", "caibatch_b"],
        "hashChainSha256Hexes": ["a" * 64, "b" * 64],
    }

    valid, error = validate_cai_owned_transport_execution_proof(
        proof,
        participant_node_ids=["node-a", "node-b"],
        executor_node_ids=["node-a", "node-b"],
        model_id="cai-network/Qwen3-0.6B-GGUF",
    )

    assert valid is True
    assert error is None


def test_cai_owned_transport_execution_proof_validates_node_signatures() -> None:
    public_key_b64, signing_seed_b64 = _signing_material()
    receipt = sign_cai_owned_transport_shard_receipt(
        {
            "nodeId": "node-a",
            "network": "mainnet",
            "chainId": "mainnet",
            "status": "completed",
            "batchIds": ["caibatch_signed"],
        },
        public_key_b64=public_key_b64,
        signing_seed_b64=signing_seed_b64,
    )
    proof = build_cai_owned_transport_execution_proof(
        session_id="session-signed-proof",
        instance_id="instance-signed-proof",
        participant_node_ids=["node-a"],
        executor_node_ids=["node-a"],
        shard_receipts=[receipt],
    )
    signed_proof = sign_cai_owned_transport_execution_proof(
        proof,
        public_key_b64=public_key_b64,
        signing_seed_b64=signing_seed_b64,
        signer_node_id="node-a",
    )
    tampered_receipt = dict(receipt)
    tampered_receipt["batchIds"] = ["caibatch_tampered"]
    tampered_receipt_proof = sign_cai_owned_transport_execution_proof(
        build_cai_owned_transport_execution_proof(
            session_id="session-tampered-receipt",
            instance_id="instance-tampered-receipt",
            participant_node_ids=["node-a"],
            executor_node_ids=["node-a"],
            shard_receipts=[tampered_receipt],
        ),
        public_key_b64=public_key_b64,
        signing_seed_b64=signing_seed_b64,
        signer_node_id="node-a",
    )
    unsigned_receipt_proof = sign_cai_owned_transport_execution_proof(
        build_cai_owned_transport_execution_proof(
            session_id="session-unsigned-receipt",
            instance_id="instance-unsigned-receipt",
            participant_node_ids=["node-a"],
            executor_node_ids=["node-a"],
            shard_receipts=[
                {
                    "nodeId": "node-a",
                    "network": "mainnet",
                    "chainId": "mainnet",
                    "status": "completed",
                    "batchIds": ["caibatch_unsigned"],
                }
            ],
        ),
        public_key_b64=public_key_b64,
        signing_seed_b64=signing_seed_b64,
        signer_node_id="node-a",
    )

    valid, error = validate_cai_owned_transport_execution_proof(
        signed_proof,
        participant_node_ids=["node-a"],
        executor_node_ids=["node-a"],
        require_signature=True,
    )
    tampered_valid, tampered_error = validate_cai_owned_transport_execution_proof(
        tampered_receipt_proof,
        participant_node_ids=["node-a"],
        executor_node_ids=["node-a"],
        require_signature=True,
    )
    unsigned_receipt_valid, unsigned_receipt_error = (
        validate_cai_owned_transport_execution_proof(
            unsigned_receipt_proof,
            participant_node_ids=["node-a"],
            executor_node_ids=["node-a"],
            require_signature=True,
        )
    )

    assert valid is True
    assert error is None
    assert signed_proof["signerNodeId"] == "node-a"
    assert tampered_valid is False
    assert tampered_error == "CAI-owned transport shard receipt payload signature is invalid"
    assert unsigned_receipt_valid is False
    assert (
        unsigned_receipt_error
        == "CAI-owned transport shard receipt payload signature is missing"
    )


def test_cai_owned_transport_execution_proof_binds_receipts_to_node_identity() -> None:
    executor_public_key_b64, executor_seed_b64 = _signing_material()
    coordinator_public_key_b64, coordinator_seed_b64 = _signing_material()
    wrong_public_key_b64, _wrong_seed_b64 = _signing_material()
    receipt = sign_cai_owned_transport_shard_receipt(
        {
            "nodeId": "node-a",
            "network": "mainnet",
            "chainId": "mainnet",
            "status": "completed",
            "batchIds": ["caibatch_trusted_receipt"],
        },
        public_key_b64=executor_public_key_b64,
        signing_seed_b64=executor_seed_b64,
    )
    proof = build_cai_owned_transport_execution_proof(
        session_id="session-trusted-proof",
        instance_id="instance-trusted-proof",
        participant_node_ids=["node-a", "node-b"],
        executor_node_ids=["node-a"],
        shard_receipts=[receipt],
    )
    signed_proof = sign_cai_owned_transport_execution_proof(
        proof,
        public_key_b64=coordinator_public_key_b64,
        signing_seed_b64=coordinator_seed_b64,
        signer_node_id="node-b",
    )

    valid, error = validate_cai_owned_transport_execution_proof(
        signed_proof,
        participant_node_ids=["node-a", "node-b"],
        executor_node_ids=["node-a"],
        require_trusted_signer=True,
        trusted_signer_identities_by_node={
            "node-a": {"node_public_key_b64": executor_public_key_b64},
            "node-b": {"node_public_key_b64": coordinator_public_key_b64},
        },
    )
    wrong_receipt_key_valid, wrong_receipt_key_error = (
        validate_cai_owned_transport_execution_proof(
            signed_proof,
            participant_node_ids=["node-a", "node-b"],
            executor_node_ids=["node-a"],
            require_trusted_signer=True,
            trusted_signer_identities_by_node={
                "node-a": {"node_public_key_b64": wrong_public_key_b64},
                "node-b": {"node_public_key_b64": coordinator_public_key_b64},
            },
        )
    )

    assert valid is True
    assert error is None
    assert wrong_receipt_key_valid is False
    assert wrong_receipt_key_error == (
        "CAI-owned transport shard receipt payload signer public key "
        "is not trusted for node"
    )


def test_cai_owned_transport_execution_proof_rejects_missing_or_foreign_receipts() -> None:
    missing_receipt_proof = build_cai_owned_transport_execution_proof(
        session_id="session-missing-receipt",
        instance_id="instance-missing-receipt",
        participant_node_ids=["node-a", "node-b"],
        executor_node_ids=["node-a", "node-b"],
        shard_receipts=[
            {
                "nodeId": "node-a",
                "network": "mainnet",
                "chainId": "mainnet",
                "status": "completed",
                "batchIds": ["caibatch_a"],
            }
        ],
    )
    foreign_receipt_proof = build_cai_owned_transport_execution_proof(
        session_id="session-foreign-receipt",
        instance_id="instance-foreign-receipt",
        participant_node_ids=["node-a", "node-b"],
        executor_node_ids=["node-a", "node-b"],
        shard_receipts=[
            {
                "nodeId": "node-a",
                "network": "mainnet",
                "chainId": "mainnet",
                "status": "completed",
                "batchIds": ["caibatch_a"],
            },
            {
                "nodeId": "node-b",
                "network": "mainnet",
                "chainId": "mainnet",
                "status": "completed",
                "batchIds": ["caibatch_b"],
            },
            {
                "nodeId": "node-c",
                "network": "mainnet",
                "chainId": "mainnet",
                "status": "completed",
                "batchIds": ["caibatch_c"],
            },
        ],
    )

    missing_valid, missing_error = validate_cai_owned_transport_execution_proof(
        missing_receipt_proof,
        participant_node_ids=["node-a", "node-b"],
        executor_node_ids=["node-a", "node-b"],
    )
    foreign_valid, foreign_error = validate_cai_owned_transport_execution_proof(
        foreign_receipt_proof,
        participant_node_ids=["node-a", "node-b"],
        executor_node_ids=["node-a", "node-b"],
    )

    assert missing_valid is False
    assert missing_error == "CAI-owned transport proof does not cover every executor."
    assert foreign_valid is False
    assert foreign_error == (
        "CAI-owned transport proof shard receipt node is not a participant."
    )


def test_cai_owned_transport_execution_proof_rejects_duplicate_batch_ids() -> None:
    proof = build_cai_owned_transport_execution_proof(
        session_id="session-duplicate-batch",
        instance_id="instance-duplicate-batch",
        participant_node_ids=["node-a", "node-b"],
        executor_node_ids=["node-a", "node-b"],
        shard_receipts=[
            {
                "nodeId": "node-a",
                "network": "mainnet",
                "chainId": "mainnet",
                "status": "completed",
                "batchIds": ["caibatch_shared"],
            },
            {
                "nodeId": "node-b",
                "network": "mainnet",
                "chainId": "mainnet",
                "status": "completed",
                "batchIds": ["caibatch_shared"],
            },
        ],
    )

    valid, error = validate_cai_owned_transport_execution_proof(
        proof,
        participant_node_ids=["node-a", "node-b"],
        executor_node_ids=["node-a", "node-b"],
    )

    assert valid is False
    assert error == "CAI-owned transport proof duplicates batch id 'caibatch_shared'."


def test_cai_owned_transport_execution_proof_rejects_wrong_hashes() -> None:
    invalid_hash_proof = build_cai_owned_transport_execution_proof(
        session_id="session-invalid-hash",
        instance_id="instance-invalid-hash",
        participant_node_ids=["node-a"],
        executor_node_ids=["node-a"],
        shard_receipts=[
            {
                "nodeId": "node-a",
                "network": "mainnet",
                "chainId": "mainnet",
                "status": "completed",
                "batchIds": ["caibatch_a"],
                "hashChainSha256Hexes": ["not-a-sha256"],
            }
        ],
    )
    mismatch_proof = build_cai_owned_transport_execution_proof(
        session_id="session-hash-mismatch",
        instance_id="instance-hash-mismatch",
        participant_node_ids=["node-a"],
        executor_node_ids=["node-a"],
        shard_receipts=[
            {
                "nodeId": "node-a",
                "network": "mainnet",
                "chainId": "mainnet",
                "status": "completed",
                "batchIds": ["caibatch_a"],
                "hashChainSha256Hexes": ["a" * 64],
            }
        ],
    )
    mismatch_proof["executionAudit"] = {
        "verified": True,
        "errorCount": 0,
        "errors": [],
        "processedBatchIds": ["caibatch_a"],
        "receiptBatchIds": ["caibatch_a"],
        "hashChainSha256Hexes": ["b" * 64],
    }

    invalid_hash_valid, invalid_hash_error = (
        validate_cai_owned_transport_execution_proof(
            invalid_hash_proof,
            participant_node_ids=["node-a"],
            executor_node_ids=["node-a"],
        )
    )
    mismatch_valid, mismatch_error = validate_cai_owned_transport_execution_proof(
        mismatch_proof,
        participant_node_ids=["node-a"],
        executor_node_ids=["node-a"],
    )

    assert invalid_hash_valid is False
    assert invalid_hash_error == (
        "CAI-owned transport hashChainSha256Hexes must be sha256 hex."
    )
    assert mismatch_valid is False
    assert mismatch_error == (
        "CAI-owned transport proof hash chain does not match execution audit."
    )


def test_cai_owned_transport_execution_proof_rejects_bad_runtime_audit_version() -> None:
    proof = build_cai_owned_transport_execution_proof(
        session_id="session-bad-runtime-audit",
        instance_id="instance-bad-runtime-audit",
        participant_node_ids=["node-a"],
        executor_node_ids=["node-a"],
        shard_receipts=[
            {
                "nodeId": "node-a",
                "network": "mainnet",
                "chainId": "mainnet",
                "status": "completed",
                "batchIds": ["caibatch_a"],
                "runtimeAudits": [
                    {
                        "protocolVersion": 999,
                        "runtimeVersion": "cai-owned-runtime/0.1",
                        "adapterVersion": "deterministic-bytes/0.1",
                    }
                ],
            }
        ],
    )

    valid, error = validate_cai_owned_transport_execution_proof(
        proof,
        participant_node_ids=["node-a"],
        executor_node_ids=["node-a"],
    )

    assert valid is False
    assert error == "CAI-owned transport protocol version is unsupported."


def test_cai_owned_transport_session_execution_audit_rejects_wrong_hash_chain() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        created = create_cai_owned_transport_session(
            session_id="caiot_wrong_hash_chain",
            instance_id="instance-wrong-hash-chain",
            participant_node_ids=["node-a"],
            executor_node_ids=["node-a"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            source_node_id="node-a",
            policy=policy,
        )
        record_cai_owned_transport_batch(
            created.session_id,
            batch_id="caibatch_wrong_hash",
            phase="prefill_activation_batches",
            source_node_id="node-a",
            sink_node_id="node-a",
            payload_size_bytes=32,
            payload_sha256_hex="a" * 64,
            status="received",
            policy=policy,
        )
        complete_cai_owned_transport_batch_processing(
            created.session_id,
            "caibatch_wrong_hash",
            node_id="node-a",
            output_payload=b"worker-output",
            policy=policy,
        )
        record = list_cai_owned_transport_sessions(policy)[0]
        record.batch_records[0]["hashChainSha256Hex"] = "0" * 64
        valid, error, audit = validate_cai_owned_transport_session_execution_audit(
            record,
            policy=policy,
        )

    assert valid is False
    assert error == (
        "CAI-owned transport batch 'caibatch_wrong_hash' hash chain does not match."
    )
    assert audit["verified"] is False


def test_deterministic_cai_owned_transport_session_id_is_stable() -> None:
    first = deterministic_cai_owned_transport_session_id(
        "instance-1",
        ["node-a", "node-b"],
        task_id="task-1",
    )
    second = deterministic_cai_owned_transport_session_id(
        "instance-1",
        ["node-a", "node-b"],
        task_id="task-1",
    )
    reversed_participants = deterministic_cai_owned_transport_session_id(
        "instance-1",
        ["node-b", "node-a"],
        task_id="task-1",
    )

    assert first == second
    assert first.startswith("caiot_")
    assert len(first) == len("caiot_") + 24
    assert reversed_participants != first


def test_cai_owned_transport_session_create_is_idempotent_for_session_id() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        session_id = deterministic_cai_owned_transport_session_id(
            "instance-idempotent",
            ["node-a", "node-b"],
            task_id="task-1",
        )
        created = create_cai_owned_transport_session(
            session_id=session_id,
            instance_id="instance-idempotent",
            participant_node_ids=["node-a", "node-b"],
            task_id="task-1",
            policy=policy,
        )
        repeated = create_cai_owned_transport_session(
            session_id=session_id,
            instance_id="instance-idempotent",
            participant_node_ids=["node-a", "node-b"],
            task_id="task-1",
            policy=policy,
        )
        records = list_cai_owned_transport_sessions(policy)

    assert created.session_id == session_id
    assert repeated.session_id == session_id
    assert len(records) == 1


def test_cai_owned_transport_batch_envelope_validates_payload_integrity() -> None:
    envelope = build_cai_owned_transport_batch_envelope(
        session_id="session-1",
        phase="prefill_activation_batches",
        source_node_id="node-a",
        sink_node_id="node-b",
        sequence=2,
        payload=b"activation-batch",
        metadata={"layerStart": 0, "layerEnd": 4},
    )

    valid, error = validate_cai_owned_transport_batch_envelope(
        envelope,
        session_id="session-1",
        participant_node_ids=["node-a", "node-b"],
    )
    tampered = dict(envelope)
    tampered["payloadBase64"] = "dGFtcGVyZWQ="
    tampered_valid, tampered_error = validate_cai_owned_transport_batch_envelope(
        tampered,
        session_id="session-1",
        participant_node_ids=["node-a", "node-b"],
    )
    wrong_participant_valid, wrong_participant_error = (
        validate_cai_owned_transport_batch_envelope(
            envelope,
            session_id="session-1",
            participant_node_ids=["node-a", "node-c"],
        )
    )
    wrong_batch_id = dict(envelope)
    wrong_batch_id["batchId"] = "caibatch_" + ("0" * 24)
    wrong_batch_id_valid, wrong_batch_id_error = (
        validate_cai_owned_transport_batch_envelope(
            wrong_batch_id,
            session_id="session-1",
            participant_node_ids=["node-a", "node-b"],
        )
    )

    assert valid is True
    assert error is None
    assert envelope["chainId"] == "mainnet"
    assert envelope["batchId"].startswith("caibatch_")
    assert envelope["payloadSizeBytes"] == len(b"activation-batch")
    assert tampered_valid is False
    assert (
        tampered_error
        == "CAI-owned transport batch envelope payload size does not match."
    )
    assert wrong_participant_valid is False
    assert (
        wrong_participant_error
        == "CAI-owned transport batch envelope participant is not allowed."
    )
    assert wrong_batch_id_valid is False
    assert (
        wrong_batch_id_error
        == "CAI-owned transport batch envelope batch id does not match."
    )


def test_cai_owned_transport_batch_envelope_rejects_stale_created_at() -> None:
    envelope = build_cai_owned_transport_batch_envelope(
        session_id="session-stale-envelope",
        phase="prefill_activation_batches",
        source_node_id="node-a",
        sink_node_id="node-b",
        sequence=0,
        payload=b"activation-batch",
        created_at="2000-01-01T00:00:00+00:00",
    )

    valid, error = validate_cai_owned_transport_batch_envelope(
        envelope,
        session_id="session-stale-envelope",
        participant_node_ids=["node-a", "node-b"],
    )

    assert valid is False
    assert error == "CAI-owned transport batch envelope has expired."


def test_cai_owned_transport_batch_envelope_validates_node_signature() -> None:
    public_key_b64, signing_seed_b64 = _signing_material()
    envelope = build_cai_owned_transport_batch_envelope(
        session_id="session-signed-envelope",
        phase="prefill_activation_batches",
        source_node_id="node-a",
        sink_node_id="node-b",
        sequence=1,
        payload=b"activation-batch",
    )
    signed_envelope = sign_cai_owned_transport_batch_envelope(
        envelope,
        public_key_b64=public_key_b64,
        signing_seed_b64=signing_seed_b64,
    )
    tampered = dict(signed_envelope)
    tampered["metadata"] = {"tampered": True}

    valid, error = validate_cai_owned_transport_batch_envelope(
        signed_envelope,
        session_id="session-signed-envelope",
        participant_node_ids=["node-a", "node-b"],
        require_signature=True,
    )
    tampered_valid, tampered_error = validate_cai_owned_transport_batch_envelope(
        tampered,
        session_id="session-signed-envelope",
        participant_node_ids=["node-a", "node-b"],
        require_signature=True,
    )
    wrong_signer = sign_cai_owned_transport_batch_envelope(
        envelope,
        public_key_b64=public_key_b64,
        signing_seed_b64=signing_seed_b64,
        signer_node_id="node-b",
    )
    wrong_signer_valid, wrong_signer_error = validate_cai_owned_transport_batch_envelope(
        wrong_signer,
        session_id="session-signed-envelope",
        participant_node_ids=["node-a", "node-b"],
        require_signature=True,
    )

    assert valid is True
    assert error is None
    assert signed_envelope["signerNodeId"] == "node-a"
    assert tampered_valid is False
    assert tampered_error == "CAI-owned transport batch envelope payload signature is invalid"
    assert wrong_signer_valid is False
    assert wrong_signer_error == (
        "CAI-owned transport batch envelope payload signer node id does not match"
    )


def test_cai_owned_transport_batch_envelope_supports_gzip_payload_compression() -> None:
    payload = b"activation-batch-" * 64
    envelope = build_cai_owned_transport_batch_envelope(
        session_id="session-gzip",
        phase="prefill_activation_batches",
        source_node_id="node-a",
        sink_node_id="node-b",
        sequence=1,
        payload=payload,
        payload_compression="gzip",
    )

    valid, error = validate_cai_owned_transport_batch_envelope(
        envelope,
        session_id="session-gzip",
        participant_node_ids=["node-a", "node-b"],
    )
    decoded = cai_owned_transport_batch_payload_bytes(envelope)
    unsupported = dict(envelope)
    unsupported["payloadCompression"] = "brotli"
    unsupported_valid, unsupported_error = validate_cai_owned_transport_batch_envelope(
        unsupported,
        session_id="session-gzip",
        participant_node_ids=["node-a", "node-b"],
    )

    assert valid is True
    assert error is None
    assert envelope["payloadCompression"] == "gzip"
    assert envelope["payloadSizeBytes"] == len(payload)
    assert envelope["payloadEncodedSizeBytes"] < len(payload)
    assert decoded == payload
    assert unsupported_valid is False
    assert (
        unsupported_error
        == "CAI-owned transport batch envelope payload compression is unsupported."
    )


def test_cai_owned_transport_batch_envelope_supports_payload_chunks() -> None:
    payload = b"chunked-activation-frame-" * 32
    envelope = build_cai_owned_transport_batch_envelope(
        session_id="session-chunked",
        phase="decode_activation_batches",
        source_node_id="node-a",
        sink_node_id="node-b",
        sequence=7,
        payload=payload,
        payload_chunk_size_bytes=64,
    )

    valid, error = validate_cai_owned_transport_batch_envelope(
        envelope,
        session_id="session-chunked",
        participant_node_ids=["node-a", "node-b"],
    )
    decoded = cai_owned_transport_batch_payload_bytes(envelope)
    tampered = dict(envelope)
    tampered["payloadChunkCount"] = int(envelope["payloadChunkCount"]) + 1
    tampered_valid, tampered_error = validate_cai_owned_transport_batch_envelope(
        tampered,
        session_id="session-chunked",
        participant_node_ids=["node-a", "node-b"],
    )

    assert valid is True
    assert error is None
    assert "payloadBase64" not in envelope
    assert len(envelope["payloadChunksBase64"]) == envelope["payloadChunkCount"]
    assert envelope["payloadChunkSizeBytes"] == 64
    assert envelope["payloadEncodedSizeBytes"] == len(payload)
    assert decoded == payload
    assert tampered_valid is False
    assert (
        tampered_error
        == "CAI-owned transport batch envelope payload chunk count does not match."
    )


def test_cai_owned_transport_frame_metadata_validates_before_batch_acceptance() -> None:
    payload = b"activation-frame"
    metadata = build_cai_owned_transport_frame_metadata(
        model_id="cai-network/Qwen3-0.6B-GGUF",
        frame_kind="activation",
        tokenizer_config_hash="ab" * 32,
        layer_start=0,
        layer_end=8,
        token_start=0,
        token_end=16,
        dtype="f16",
        shape=[1, 16, 64],
        sequence=3,
        payload_sha256_hex=hashlib.sha256(payload).hexdigest(),
    )
    envelope = build_cai_owned_transport_batch_envelope(
        session_id="session-frame",
        phase="prefill_activation_batches",
        source_node_id="node-a",
        sink_node_id="node-b",
        sequence=3,
        payload=payload,
        metadata=metadata,
    )
    valid, error = validate_cai_owned_transport_batch_envelope(
        envelope,
        session_id="session-frame",
        participant_node_ids=["node-a", "node-b"],
    )
    wrong_model_valid, wrong_model_error = validate_cai_owned_transport_frame_metadata(
        metadata,
        expected_model_id="other/model",
    )
    wrong_hash = dict(envelope)
    wrong_hash["metadata"] = {
        **metadata,
        "payloadSha256Hex": "cd" * 32,
    }
    wrong_hash_valid, wrong_hash_error = validate_cai_owned_transport_batch_envelope(
        wrong_hash,
        session_id="session-frame",
        participant_node_ids=["node-a", "node-b"],
    )

    assert valid is True
    assert error is None
    assert wrong_model_valid is False
    assert wrong_model_error == "CAI-owned transport frame model id does not match."
    assert wrong_hash_valid is False
    assert wrong_hash_error == "CAI-owned transport frame payload hash does not match."


def test_cai_owned_llm_handoff_metadata_validates_production_contract() -> None:
    payload = b"real-activation-bytes"
    payload_hash = hashlib.sha256(payload).hexdigest()
    handoff = build_cai_owned_llm_handoff_metadata(
        model_id="cai-network/Qwen3-0.6B-GGUF",
        backend="llama.cpp-patched",
        backend_version="llama.cpp/cai-shard-0.1",
        model_sha256_hex="12" * 32,
        tokenizer_config_hash="ab" * 32,
        layer_start=0,
        layer_end=8,
        token_start=0,
        token_end=16,
        tensor_dtype="f16",
        tensor_shape=[1, 16, 768],
        tensor_encoding="ggml-tensor-v1",
        tensor_sha256_hex=payload_hash,
        kv_cache={
            "layout": "llama.cpp-kv-cache-v1",
            "dtype": "f16",
            "shape": [2, 8, 16, 64],
        },
        decode_state={"position": 16, "sequenceId": "seq-1"},
    )
    metadata = build_cai_owned_transport_frame_metadata(
        model_id="cai-network/Qwen3-0.6B-GGUF",
        frame_kind="activation",
        tokenizer_config_hash="ab" * 32,
        layer_start=0,
        layer_end=8,
        token_start=0,
        token_end=16,
        dtype="f16",
        shape=[1, 16, 768],
        payload_sha256_hex=payload_hash,
        extra_metadata={"production": True},
    )
    metadata["llmHandoff"] = handoff

    valid, error = validate_cai_owned_transport_frame_metadata(
        metadata,
        expected_model_id="cai-network/Qwen3-0.6B-GGUF",
        require_llm_handoff=True,
    )
    missing_valid, missing_error = validate_cai_owned_transport_frame_metadata(
        {key: value for key, value in metadata.items() if key != "llmHandoff"},
        expected_model_id="cai-network/Qwen3-0.6B-GGUF",
        require_llm_handoff=True,
    )
    tampered = dict(handoff)
    tampered["tensor"] = {**dict(handoff["tensor"]), "sha256Hex": "cd" * 32}
    tampered_valid, tampered_error = validate_cai_owned_llm_handoff_metadata(
        tampered,
        expected_model_id="cai-network/Qwen3-0.6B-GGUF",
        expected_frame_metadata=metadata,
    )

    assert valid is True
    assert error is None
    assert missing_valid is False
    assert missing_error == "CAI-owned LLM handoff metadata is missing."
    assert tampered_valid is False
    assert tampered_error == (
        "CAI-owned LLM handoff tensor hash does not match frame payload."
    )


def test_qwen_private_model_builds_runtime_shard_handoff_metadata() -> None:
    payload = b"qwen-runtime-activation"
    runtime_metadata = {
        "modelId": "cai-network/Qwen3-0.6B-GGUF",
        "totalLayerCount": 28,
        "hiddenSize": 1024,
        "activationDtype": "f16",
        "tensorEncoding": "ggml-tensor-v1",
        "tokenizerConfigHash": "ab" * 32,
        "modelSha256Hex": "12" * 32,
        "backend": "llama.cpp-patched",
        "backendVersion": "llama.cpp/cai-shard-0.1",
        "metadataSource": "qwen3-0.6b-gguf-runtime",
        "preferredFilename": "Qwen3-0.6B-Q8_0.gguf",
        "family": "Qwen3",
        "quantization": "Q8_0",
        "contextLength": 32768,
        "ggufArchitecture": "qwen3",
        "shardCompatibility": "layer_range_supported",
        "layerRangeSupported": True,
        "layerRangeProbeAbi": "cai-layer-range-v1",
        "layerRangeEquivalenceProbeReport": (
            "docs/reports/qwen3-layer-range-equivalence-probe-2026-05-10.json"
        ),
        "modelArtifactPath": "C:/private/models/Qwen3-0.6B-Q8_0.gguf",
    }
    dag = build_cai_owned_transport_execution_dag(
        session_id="caiot_qwen_runtime",
        requester_node_id="node-user",
        executor_node_ids=["node-a", "node-b"],
        total_layer_count=28,
        model_id="cai-network/Qwen3-0.6B-GGUF",
        task_id="task-qwen-runtime",
        input_payload_sha256_hex="11" * 32,
        created_at="2026-05-03T00:00:00+00:00",
    )
    first_range = dag["shardRanges"][0]

    metadata = build_cai_owned_llm_shard_frame_metadata_from_runtime(
        model_id="cai-network/Qwen3-0.6B-GGUF",
        runtime_metadata=runtime_metadata,
        payload=payload,
        layer_start=first_range["layerStart"],
        layer_end=first_range["layerEnd"],
        token_start=0,
        token_end=8,
        sequence=0,
    )
    valid, error = validate_cai_owned_transport_frame_metadata(
        metadata,
        expected_model_id="cai-network/Qwen3-0.6B-GGUF",
        require_llm_handoff=True,
    )

    assert valid is True
    assert error is None
    assert metadata["layerStart"] == 0
    assert metadata["layerEnd"] == 14
    assert metadata["shape"] == [1, 8, 1024]
    assert metadata["payloadSha256Hex"] == hashlib.sha256(payload).hexdigest()
    assert metadata["llmHandoff"]["abi"] == "cai-llm-shard-handoff-v1"
    assert metadata["llmHandoff"]["modelId"] == "cai-network/Qwen3-0.6B-GGUF"
    assert metadata["llmHandoff"]["tensor"]["shape"] == [1, 8, 1024]
    assert metadata["llmHandoff"]["tensor"]["encoding"] == "ggml-tensor-v1"
    assert metadata["llmHandoff"]["extraMetadata"]["totalLayerCount"] == 28
    assert metadata["llmHandoff"]["extraMetadata"]["preferredFilename"] == (
        "Qwen3-0.6B-Q8_0.gguf"
    )
    assert metadata["llmHandoff"]["extraMetadata"]["family"] == "Qwen3"
    assert metadata["llmHandoff"]["extraMetadata"]["quantization"] == "Q8_0"
    assert metadata["llmHandoff"]["extraMetadata"]["contextLength"] == 32768
    assert metadata["llmHandoff"]["extraMetadata"]["ggufArchitecture"] == "qwen3"
    assert metadata["llmHandoff"]["extraMetadata"]["shardCompatibility"] == (
        "layer_range_supported"
    )
    assert metadata["llmHandoff"]["extraMetadata"]["layerRangeSupported"] is True
    assert metadata["llmHandoff"]["extraMetadata"][
        "layerRangeEquivalenceProbeReport"
    ].endswith("qwen3-layer-range-equivalence-probe-2026-05-10.json")
    assert "modelArtifactPath" not in metadata["llmHandoff"]["extraMetadata"]
    assert metadata["extraMetadata"]["preferredFilename"] == "Qwen3-0.6B-Q8_0.gguf"
    assert metadata["extraMetadata"]["family"] == "Qwen3"
    assert metadata["extraMetadata"]["quantization"] == "Q8_0"
    assert metadata["extraMetadata"]["contextLength"] == 32768
    assert metadata["extraMetadata"]["ggufArchitecture"] == "qwen3"
    assert metadata["extraMetadata"]["shardCompatibility"] == "layer_range_supported"
    assert metadata["extraMetadata"]["layerRangeSupported"] is True
    assert metadata["extraMetadata"]["layerRangeEquivalenceProbeReport"].endswith(
        "qwen3-layer-range-equivalence-probe-2026-05-10.json"
    )
    assert "modelArtifactPath" not in metadata["extraMetadata"]

    qwen2_runtime_metadata = {
        **runtime_metadata,
        "modelId": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "totalLayerCount": 24,
        "hiddenSize": 896,
        "metadataSource": "qwen2.5-0.5b-gguf-runtime",
        "preferredFilename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "family": "qwen",
        "quantization": "Q4_K_M",
        "contextLength": 8192,
        "ggufArchitecture": "qwen2",
        "shardCompatibility": "layer_range_supported",
        "layerRangeSupported": True,
        "layerRangeProbeAbi": "cai-layer-range-v1",
        "layerRangeEquivalenceProbeReport": (
            "docs/reports/qwen2.5-layer-range-equivalence-probe-2026-05-11.json"
        ),
    }
    qwen2_metadata = build_cai_owned_llm_shard_frame_metadata_from_runtime(
        model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        runtime_metadata=qwen2_runtime_metadata,
        payload=payload,
        layer_start=0,
        layer_end=12,
        token_start=0,
        token_end=8,
    )

    assert qwen2_metadata["shape"] == [1, 8, 896]
    assert qwen2_metadata["llmHandoff"]["extraMetadata"]["ggufArchitecture"] == "qwen2"
    assert qwen2_metadata["llmHandoff"]["extraMetadata"]["shardCompatibility"] == (
        "layer_range_supported"
    )
    assert qwen2_metadata["llmHandoff"]["extraMetadata"]["layerRangeSupported"] is True

    unsupported_runtime_metadata = {
        **runtime_metadata,
        "modelId": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "family": "qwen",
        "preferredFilename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "ggufArchitecture": "qwen2",
        "shardCompatibility": "unsupported_for_sharding",
        "layerRangeSupported": False,
    }
    with pytest.raises(ValueError, match="unsupported_for_sharding"):
        build_cai_owned_llm_shard_frame_metadata_from_runtime(
            model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            runtime_metadata=unsupported_runtime_metadata,
            payload=payload,
            layer_start=0,
            layer_end=12,
            token_start=0,
            token_end=8,
        )

    full_model_local_runtime_metadata = {
        **runtime_metadata,
        "modelId": "TheBloke/Mistral-7B-GGUF",
        "family": "mistral",
        "preferredFilename": "mistral-7b-instruct.Q4_K_M.gguf",
        "ggufArchitecture": "mistral",
        "shardCompatibility": "full_model_local",
        "layerRangeSupported": False,
    }
    with pytest.raises(ValueError, match="full_model_local"):
        build_cai_owned_llm_shard_frame_metadata_from_runtime(
            model_id="TheBloke/Mistral-7B-GGUF",
            runtime_metadata=full_model_local_runtime_metadata,
            payload=payload,
            layer_start=0,
            layer_end=12,
            token_start=0,
            token_end=8,
        )

    multimodal_runtime_metadata = {
        **runtime_metadata,
        "modelId": "Qwen/Qwen2.5-Omni-7B-GGUF",
        "family": "qwen",
        "preferredFilename": "qwen2.5-omni-7b-q4_k_m.gguf",
        "ggufArchitecture": "qwen",
        "shardCompatibility": "layer_range_supported",
        "layerRangeSupported": True,
    }
    with pytest.raises(ValueError, match="unsupported_for_sharding"):
        build_cai_owned_llm_shard_frame_metadata_from_runtime(
            model_id="Qwen/Qwen2.5-Omni-7B-GGUF",
            runtime_metadata=multimodal_runtime_metadata,
            payload=payload,
            layer_start=0,
            layer_end=12,
            token_start=0,
            token_end=8,
        )

    with pytest.raises(ValueError, match="layer range exceeds runtime metadata"):
        build_cai_owned_llm_shard_frame_metadata_from_runtime(
            model_id="cai-network/Qwen3-0.6B-GGUF",
            runtime_metadata=runtime_metadata,
            payload=payload,
            layer_start=14,
            layer_end=29,
            token_start=0,
            token_end=8,
        )


def test_cai_owned_transport_session_offer_validates_and_creates_record() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        offer = build_cai_owned_transport_session_offer(
            instance_id="instance-offer",
            participant_node_ids=["node-a", "node-b"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-1",
            source_node_id="node-a",
        )
        valid, error = validate_cai_owned_transport_session_offer(
            offer,
            session_id=offer["sessionId"],
            local_node_id="node-b",
        )
        record = create_cai_owned_transport_session_from_offer(
            offer,
            session_id=offer["sessionId"],
            local_node_id="node-b",
            policy=policy,
        )
        records = list_cai_owned_transport_sessions(policy)

    assert valid is True
    assert error is None
    assert offer["chainId"] == "mainnet"
    assert record.session_id == offer["sessionId"]
    assert record.chain_id == "mainnet"
    assert record.participant_node_ids == ["node-a", "node-b"]
    assert len(records) == 1


def test_cai_owned_transport_session_offer_rejects_stale_created_at() -> None:
    offer = build_cai_owned_transport_session_offer(
        instance_id="instance-stale-offer",
        participant_node_ids=["node-a", "node-b"],
        source_node_id="node-a",
        created_at="2000-01-01T00:00:00+00:00",
    )

    valid, error = validate_cai_owned_transport_session_offer(
        offer,
        session_id=offer["sessionId"],
        local_node_id="node-b",
    )

    assert valid is False
    assert error == "CAI-owned transport session offer has expired."


def test_cai_owned_transport_session_offer_validates_node_signature() -> None:
    public_key_b64, signing_seed_b64 = _signing_material()
    offer = build_cai_owned_transport_session_offer(
        instance_id="instance-signed-offer",
        participant_node_ids=["node-a", "node-b"],
        source_node_id="node-a",
    )
    signed_offer = sign_cai_owned_transport_session_offer(
        offer,
        public_key_b64=public_key_b64,
        signing_seed_b64=signing_seed_b64,
    )
    tampered = dict(signed_offer)
    tampered["modelId"] = "cai-network/Other-GGUF"

    valid, error = validate_cai_owned_transport_session_offer(
        signed_offer,
        session_id=offer["sessionId"],
        local_node_id="node-b",
        require_signature=True,
    )
    tampered_valid, tampered_error = validate_cai_owned_transport_session_offer(
        tampered,
        session_id=offer["sessionId"],
        local_node_id="node-b",
        require_signature=True,
    )
    unsigned_valid, unsigned_error = validate_cai_owned_transport_session_offer(
        offer,
        session_id=offer["sessionId"],
        local_node_id="node-b",
        require_signature=True,
    )

    assert valid is True
    assert error is None
    assert signed_offer["signerNodeId"] == "node-a"
    assert tampered_valid is False
    assert tampered_error == "CAI-owned transport session offer payload signature is invalid"
    assert unsigned_valid is False
    assert unsigned_error == "CAI-owned transport session offer payload signature is missing"


def test_cai_owned_transport_session_offer_binds_signature_to_node_identity() -> None:
    public_key_b64, signing_seed_b64 = _signing_material()
    wrong_public_key_b64, _wrong_seed_b64 = _signing_material()
    offer = build_cai_owned_transport_session_offer(
        instance_id="instance-trusted-signed-offer",
        participant_node_ids=["node-a", "node-b"],
        source_node_id="node-a",
    )
    signed_offer = sign_cai_owned_transport_session_offer(
        offer,
        public_key_b64=public_key_b64,
        signing_seed_b64=signing_seed_b64,
    )

    valid, error = validate_cai_owned_transport_session_offer(
        signed_offer,
        session_id=offer["sessionId"],
        local_node_id="node-b",
        require_trusted_signer=True,
        trusted_signer_identities_by_node={
            "node-a": {"node_public_key_b64": public_key_b64},
        },
    )
    wrong_key_valid, wrong_key_error = validate_cai_owned_transport_session_offer(
        signed_offer,
        session_id=offer["sessionId"],
        local_node_id="node-b",
        require_trusted_signer=True,
        trusted_signer_identities_by_node={
            "node-a": {"node_public_key_b64": wrong_public_key_b64},
        },
    )
    missing_identity_valid, missing_identity_error = (
        validate_cai_owned_transport_session_offer(
            signed_offer,
            session_id=offer["sessionId"],
            local_node_id="node-b",
            require_trusted_signer=True,
            trusted_signer_identities_by_node={},
        )
    )

    assert valid is True
    assert error is None
    assert wrong_key_valid is False
    assert wrong_key_error == (
        "CAI-owned transport session offer payload signer public key "
        "is not trusted for node"
    )
    assert missing_identity_valid is False
    assert missing_identity_error == (
        "CAI-owned transport session offer payload signer is not trusted for node"
    )


def test_cai_owned_transport_signed_payload_replay_cache_rejects_duplicate() -> None:
    public_key_b64, signing_seed_b64 = _signing_material()
    offer = build_cai_owned_transport_session_offer(
        instance_id="instance-replay-signed-offer",
        participant_node_ids=["node-a", "node-b"],
        source_node_id="node-a",
    )
    signed_offer = sign_cai_owned_transport_session_offer(
        offer,
        public_key_b64=public_key_b64,
        signing_seed_b64=signing_seed_b64,
    )

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-replay-cache")
        first_valid, first_error = validate_cai_owned_transport_session_offer(
            signed_offer,
            session_id=offer["sessionId"],
            local_node_id="node-b",
            require_signature=True,
            record_replay_cache=True,
            replay_cache_policy=policy,
        )
        second_valid, second_error = validate_cai_owned_transport_session_offer(
            signed_offer,
            session_id=offer["sessionId"],
            local_node_id="node-b",
            require_signature=True,
            record_replay_cache=True,
            replay_cache_policy=policy,
        )
        cache_records = list_cai_owned_transport_replay_cache(policy)

    assert first_valid is True
    assert first_error is None
    assert second_valid is False
    assert second_error == (
        "CAI-owned transport session offer payload signature replay detected"
    )
    assert len(cache_records) == 1
    assert cache_records[0]["signerNodeId"] == "node-a"
    assert cache_records[0]["sessionId"] == offer["sessionId"]


def test_cai_owned_transport_receive_helpers_can_record_replay_cache() -> None:
    public_key_b64, signing_seed_b64 = _signing_material()
    offer = build_cai_owned_transport_session_offer(
        instance_id="instance-receive-replay-cache",
        participant_node_ids=["node-a", "node-b"],
        source_node_id="node-a",
    )
    signed_offer = sign_cai_owned_transport_session_offer(
        offer,
        public_key_b64=public_key_b64,
        signing_seed_b64=signing_seed_b64,
    )
    envelope = build_cai_owned_transport_batch_envelope(
        session_id=offer["sessionId"],
        phase="prefill_activation_batches",
        source_node_id="node-a",
        sink_node_id="node-b",
        sequence=0,
        payload=b"signed receive payload",
    )
    signed_envelope = sign_cai_owned_transport_batch_envelope(
        envelope,
        public_key_b64=public_key_b64,
        signing_seed_b64=signing_seed_b64,
    )

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-receive-replay")
        created = create_cai_owned_transport_session_from_offer(
            signed_offer,
            session_id=offer["sessionId"],
            local_node_id="node-b",
            record_replay_cache=True,
            policy=policy,
        )
        recorded = record_cai_owned_transport_batch_envelope(
            created.session_id,
            signed_envelope,
            local_node_id="node-b",
            record_replay_cache=True,
            policy=policy,
        )
        duplicate_offer_error = None
        duplicate_batch_error = None
        try:
            create_cai_owned_transport_session_from_offer(
                signed_offer,
                session_id=offer["sessionId"],
                local_node_id="node-b",
                record_replay_cache=True,
                policy=policy,
            )
        except ValueError as exc:
            duplicate_offer_error = str(exc)
        try:
            record_cai_owned_transport_batch_envelope(
                created.session_id,
                signed_envelope,
                local_node_id="node-b",
                record_replay_cache=True,
                policy=policy,
            )
        except ValueError as exc:
            duplicate_batch_error = str(exc)
        cache_records = list_cai_owned_transport_replay_cache(policy)

    assert recorded.batch_records[0]["batchId"] == signed_envelope["batchId"]
    assert duplicate_offer_error == (
        "CAI-owned transport session offer payload signature replay detected"
    )
    assert duplicate_batch_error == (
        "CAI-owned transport batch envelope payload signature replay detected"
    )
    assert len(cache_records) == 2


def test_cai_owned_transport_session_separates_requester_and_executors() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        offer = build_cai_owned_transport_session_offer(
            instance_id="instance-requester-executors",
            participant_node_ids=["node-user", "node-a", "node-b"],
            executor_node_ids=["node-a", "node-b"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-requester-executors",
            source_node_id="node-user",
        )
        valid, error = validate_cai_owned_transport_session_offer(
            offer,
            session_id=offer["sessionId"],
            local_node_id="node-a",
        )
        created = create_cai_owned_transport_session_from_offer(
            offer,
            session_id=offer["sessionId"],
            local_node_id="node-a",
            policy=policy,
        )
        for node_id, payload_hash, output_payload, layer_start, layer_end in [
            ("node-a", "aa" * 32, b"node-a-output", 0, 14),
            ("node-b", "bb" * 32, b"node-b-output", 14, 28),
        ]:
            batch_id = f"caibatch_{node_id.replace('-', '_')}_request"
            record_cai_owned_transport_batch(
                created.session_id,
                batch_id=batch_id,
                phase="prefill_activation_batches",
                source_node_id="node-user",
                sink_node_id=node_id,
                payload_size_bytes=32,
                payload_sha256_hex=payload_hash,
                metadata={"layerStart": layer_start, "layerEnd": layer_end},
                status="received",
                policy=policy,
            )
            complete_cai_owned_transport_batch_processing(
                created.session_id,
                batch_id,
                node_id=node_id,
                output_payload=output_payload,
                policy=policy,
            )
        completed = complete_cai_owned_transport_session(
            created.session_id,
            policy=policy,
        )
        payload = cai_owned_transport_session_to_dict(completed)

    assert valid is True
    assert error is None
    assert offer["participantNodeIds"] == ["node-user", "node-a", "node-b"]
    assert offer["executorNodeIds"] == ["node-a", "node-b"]
    assert completed.participant_node_ids == ["node-user", "node-a", "node-b"]
    assert completed.executor_node_ids == ["node-a", "node-b"]
    assert completed.proof is not None
    assert completed.proof["participantNodeIds"] == ["node-user", "node-a", "node-b"]
    assert completed.proof["executorNodeIds"] == ["node-a", "node-b"]
    assert [item["nodeId"] for item in completed.proof["shardReceipts"]] == [
        "node-a",
        "node-b",
    ]
    assert payload["executorNodeIds"] == ["node-a", "node-b"]


def test_cai_owned_transport_final_output_delivery_does_not_block_proof() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        offer = build_cai_owned_transport_session_offer(
            instance_id="instance-final-output",
            participant_node_ids=["node-user", "node-a", "node-b"],
            executor_node_ids=["node-a", "node-b"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-final-output",
            source_node_id="node-user",
        )
        create_cai_owned_transport_session_from_offer(
            offer,
            session_id=offer["sessionId"],
            local_node_id="node-user",
            policy=policy,
        )
        first_payload = b"user-prompt"
        first_envelope = build_cai_owned_transport_batch_envelope(
            session_id=offer["sessionId"],
            phase="prefill_activation_batches",
            source_node_id="node-user",
            sink_node_id="node-a",
            sequence=0,
            payload=first_payload,
        )
        record_cai_owned_transport_batch_envelope(
            offer["sessionId"],
            first_envelope,
            local_node_id="node-a",
            policy=policy,
        )
        first_result = complete_cai_owned_transport_batch_processing(
            offer["sessionId"],
            first_envelope["batchId"],
            node_id="node-a",
            output_payload=b"a:user-prompt",
            policy=policy,
        )
        second_envelope = build_cai_owned_transport_output_batch_envelope(
            session_id=offer["sessionId"],
            source_batch_id=first_envelope["batchId"],
            sink_node_id="node-b",
            metadata={"nextStage": "executor"},
            policy=policy,
        )
        record_cai_owned_transport_batch_envelope(
            offer["sessionId"],
            second_envelope,
            local_node_id="node-b",
            policy=policy,
        )
        second_result = complete_cai_owned_transport_batch_processing(
            offer["sessionId"],
            second_envelope["batchId"],
            node_id="node-b",
            output_payload=b"b:a:user-prompt",
            policy=policy,
        )
        final_envelope = build_cai_owned_transport_output_batch_envelope(
            session_id=offer["sessionId"],
            source_batch_id=second_envelope["batchId"],
            sink_node_id="node-user",
            metadata={"nextStage": "requester"},
            policy=policy,
        )
        delivered = record_cai_owned_transport_batch_envelope(
            offer["sessionId"],
            final_envelope,
            local_node_id="node-user",
            policy=policy,
        )
        latest_output = latest_cai_owned_transport_final_output(
            offer["sessionId"],
            requester_node_id="node-user",
            policy=policy,
        )
        waited_output = wait_for_cai_owned_transport_final_output(
            offer["sessionId"],
            requester_node_id="node-user",
            timeout_sec=0.01,
            policy=policy,
        )
        final_result = await_cai_owned_transport_session_final_result(
            offer["sessionId"],
            requester_node_id="node-user",
            timeout_sec=0.01,
            policy=policy,
        )
        delivered_batches = list_cai_owned_transport_batch_inbox(
            "node-user",
            status="delivered",
            session_id=offer["sessionId"],
            policy=policy,
        )
        received_user_batches = list_cai_owned_transport_batch_inbox(
            "node-user",
            session_id=offer["sessionId"],
            policy=policy,
        )

    assert first_result["receipt"]["nodeId"] == "node-a"
    assert second_result["receipt"]["nodeId"] == "node-b"
    assert delivered.batch_records[-1]["status"] == "delivered"
    assert delivered.batch_records[-1]["metadata"]["finalOutput"] is True
    assert delivered.batch_records[-1]["metadata"]["deliveredToNodeId"] == "node-user"
    assert latest_output is not None
    assert latest_output["payload"] == b"b:a:user-prompt"
    assert waited_output["batchId"] == final_envelope["batchId"]
    assert waited_output["payload"] == b"b:a:user-prompt"
    assert delivered_batches[0]["batch"]["batchId"] == final_envelope["batchId"]
    assert received_user_batches == []
    assert final_result["status"] == "completed"
    assert final_result["proofVerified"] is True
    assert final_result["session"]["status"] == "completed"
    assert final_result["proof"]["executionAudit"]["verified"] is True
    assert final_result["proof"]["executionAudit"]["processedBatchCount"] == 2
    assert final_result["proof"]["executionAudit"]["finalOutputBatchIds"] == [
        final_envelope["batchId"]
    ]
    assert [item["nodeId"] for item in final_result["proof"]["shardReceipts"]] == [
        "node-a",
        "node-b",
    ]


def test_cai_owned_transport_completion_notice_completes_peer_without_full_audit() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        offer = build_cai_owned_transport_session_offer(
            instance_id="instance-completion-notice",
            participant_node_ids=["node-user", "node-a", "node-b"],
            executor_node_ids=["node-a", "node-b"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-completion-notice",
            source_node_id="node-user",
        )
        created = create_cai_owned_transport_session_from_offer(
            offer,
            session_id=offer["sessionId"],
            local_node_id="node-a",
            policy=policy,
        )
        record_cai_owned_transport_batch(
            created.session_id,
            batch_id="caibatch_local_notice",
            phase="prefill_activation_batches",
            source_node_id="node-user",
            sink_node_id="node-a",
            payload_size_bytes=8,
            payload_sha256_hex="11" * 32,
            metadata={"layerStart": 0, "layerEnd": 14},
            status="received",
            policy=policy,
        )
        processed = complete_cai_owned_transport_batch_processing(
            created.session_id,
            "caibatch_local_notice",
            node_id="node-a",
            output_payload=b"node-a-state",
            policy=policy,
        )
        local_receipt = dict(processed["receipt"])
        remote_receipt = {
            "nodeId": "node-b",
            "chainId": offer["chainId"],
            "network": offer["chainId"],
            "status": "completed",
            "activationBatchCount": 0,
            "decodeBatchCount": 1,
            "batchIds": ["caibatch_remote_notice"],
            "outputPayloadSha256Hexes": ["22" * 32],
            "metrics": {"processedBatchCount": 1},
        }
        bad_local_receipt = dict(local_receipt)
        bad_local_receipt["batchIds"] = ["caibatch_wrong_notice"]
        bad_proof = build_cai_owned_transport_execution_proof(
            session_id=created.session_id,
            instance_id=created.instance_id,
            participant_node_ids=created.participant_node_ids,
            executor_node_ids=created.executor_node_ids,
            chain_id=created.chain_id,
            model_id=created.model_id,
            task_id=created.task_id,
            shard_receipts=[bad_local_receipt, remote_receipt],
        )
        with pytest.raises(ValueError, match="does not cover local processed"):
            accept_cai_owned_transport_completion_notice(
                created.session_id,
                bad_proof,
                policy=policy,
            )
        after_bad_notice = list_cai_owned_transport_sessions(policy)[0]
        good_proof = build_cai_owned_transport_execution_proof(
            session_id=created.session_id,
            instance_id=created.instance_id,
            participant_node_ids=created.participant_node_ids,
            executor_node_ids=created.executor_node_ids,
            chain_id=created.chain_id,
            model_id=created.model_id,
            task_id=created.task_id,
            shard_receipts=[local_receipt, remote_receipt],
        )
        completed = accept_cai_owned_transport_completion_notice(
            created.session_id,
            good_proof,
            policy=policy,
        )

    assert after_bad_notice.status == "running"
    assert after_bad_notice.last_error is None
    assert completed.status == "completed"
    assert completed.proof == good_proof


def test_cai_owned_transport_final_output_records_embedded_receipts() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        offer = build_cai_owned_transport_session_offer(
            instance_id="instance-embedded-receipts",
            participant_node_ids=["node-user", "node-a", "node-b"],
            executor_node_ids=["node-a", "node-b"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-embedded-receipts",
            source_node_id="node-user",
        )
        dag = build_cai_owned_transport_execution_dag(
            session_id=offer["sessionId"],
            requester_node_id="node-user",
            executor_node_ids=["node-a", "node-b"],
            total_layer_count=2,
            chain_id=offer["chainId"],
            model_id=offer["modelId"],
            task_id=offer["taskId"],
        )
        offer = build_cai_owned_transport_session_offer(
            instance_id="instance-embedded-receipts",
            participant_node_ids=["node-user", "node-a", "node-b"],
            executor_node_ids=["node-a", "node-b"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-embedded-receipts",
            source_node_id="node-user",
            route_policy={
                "executionDag": dag,
                "executionDagHashSha256Hex": dag["dagHashSha256Hex"],
            },
        )
        create_cai_owned_transport_session_from_offer(
            offer,
            session_id=offer["sessionId"],
            local_node_id="node-user",
            policy=policy,
        )
        stage_by_node: dict[str, list[str]] = {"node-a": [], "node-b": []}
        for stage in dag["stages"]:
            stage_by_node[stage["executorNodeId"]].append(stage["stageId"])
        public_key_b64, signing_seed_b64 = _signing_material()
        receipts = [
            {
                "nodeId": "node-a",
                "chainId": offer["chainId"],
                "network": offer["chainId"],
                "status": "completed",
                "activationBatchCount": 1,
                "decodeBatchCount": 1,
                "layerStart": None,
                "layerEnd": None,
                "batchIds": ["caibatch_node_a_prefill", "caibatch_node_a_decode"],
                "stageIds": stage_by_node["node-a"],
                "sequences": [0, 2],
                "inputPayloadSha256Hexes": [],
                "outputPayloadSha256Hexes": [],
                "hashChainSha256Hexes": [],
                "routeAudits": [],
                "runtimeAudits": [],
                "metrics": {"processedBatchCount": 2},
            },
            {
                "nodeId": "node-b",
                "chainId": offer["chainId"],
                "network": offer["chainId"],
                "status": "completed",
                "activationBatchCount": 1,
                "decodeBatchCount": 1,
                "layerStart": None,
                "layerEnd": None,
                "batchIds": ["caibatch_node_b_prefill", "caibatch_node_b_decode"],
                "stageIds": stage_by_node["node-b"],
                "sequences": [1, 3],
                "inputPayloadSha256Hexes": [],
                "outputPayloadSha256Hexes": [],
                "hashChainSha256Hexes": [],
                "routeAudits": [],
                "runtimeAudits": [],
                "metrics": {"processedBatchCount": 2},
            },
        ]
        receipts = [
            sign_cai_owned_transport_shard_receipt(
                receipt,
                public_key_b64=public_key_b64,
                signing_seed_b64=signing_seed_b64,
                signer_node_id=receipt["nodeId"],
            )
            for receipt in receipts
        ]
        final_envelope = build_cai_owned_transport_batch_envelope(
            session_id=offer["sessionId"],
            phase="decode_activation_batches",
            source_node_id="node-b",
            sink_node_id="node-user",
            sequence=4,
            payload=b"final-answer",
            metadata={
                "finalOutput": True,
                "payloadRole": "shard_output",
                "upstreamShardReceipts": receipts,
            },
        )
        delivered = record_cai_owned_transport_batch_envelope(
            offer["sessionId"],
            final_envelope,
            local_node_id="node-user",
            policy=policy,
        )
        final_result = await_cai_owned_transport_session_final_result(
            offer["sessionId"],
            requester_node_id="node-user",
            timeout_sec=0.01,
            policy=policy,
        )
        signed_proof = sign_cai_owned_transport_execution_proof(
            final_result["proof"],
            public_key_b64=public_key_b64,
            signing_seed_b64=signing_seed_b64,
            signer_node_id="node-user",
        )
        signature_valid, signature_error = validate_cai_owned_transport_execution_proof(
            signed_proof,
            participant_node_ids=offer["participantNodeIds"],
            executor_node_ids=offer["executorNodeIds"],
            model_id=offer["modelId"],
            chain_id=offer["chainId"],
            require_signature=True,
        )

    assert [item["nodeId"] for item in delivered.shard_receipts] == [
        "node-a",
        "node-b",
    ]
    assert final_result["status"] == "completed"
    assert final_result["proofVerified"] is True
    assert final_result["proof"]["executionAudit"]["processedBatchCount"] == 0
    assert final_result["proof"]["executionAudit"]["receiptBatchIds"] == [
        "caibatch_node_a_decode",
        "caibatch_node_a_prefill",
        "caibatch_node_b_decode",
        "caibatch_node_b_prefill",
    ]
    assert final_result["proof"]["executionAudit"]["executionDag"]["verified"] is True
    assert [item["nodeId"] for item in final_result["proof"]["shardReceipts"]] == [
        "node-a",
        "node-b",
    ]
    assert signature_valid is True
    assert signature_error is None


def test_cai_owned_transport_final_output_supports_requester_that_is_executor() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        offer = build_cai_owned_transport_session_offer(
            instance_id="instance-requester-executor",
            participant_node_ids=["node-a", "node-b"],
            executor_node_ids=["node-a", "node-b"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-requester-executor",
            source_node_id="node-a",
        )
        create_cai_owned_transport_session_from_offer(
            offer,
            session_id=offer["sessionId"],
            local_node_id="node-a",
            policy=policy,
        )
        first_envelope = build_cai_owned_transport_batch_envelope(
            session_id=offer["sessionId"],
            phase="prefill_activation_batches",
            source_node_id="node-a",
            sink_node_id="node-a",
            sequence=0,
            payload=b"worker-client-prompt",
        )
        record_cai_owned_transport_batch_envelope(
            offer["sessionId"],
            first_envelope,
            local_node_id="node-a",
            policy=policy,
        )
        complete_cai_owned_transport_batch_processing(
            offer["sessionId"],
            first_envelope["batchId"],
            node_id="node-a",
            output_payload=b"a:worker-client-prompt",
            policy=policy,
        )
        second_envelope = build_cai_owned_transport_output_batch_envelope(
            session_id=offer["sessionId"],
            source_batch_id=first_envelope["batchId"],
            sink_node_id="node-b",
            metadata={"nextStage": "executor"},
            policy=policy,
        )
        record_cai_owned_transport_batch_envelope(
            offer["sessionId"],
            second_envelope,
            local_node_id="node-b",
            policy=policy,
        )
        complete_cai_owned_transport_batch_processing(
            offer["sessionId"],
            second_envelope["batchId"],
            node_id="node-b",
            output_payload=b"b:a:worker-client-prompt",
            policy=policy,
        )
        final_envelope = build_cai_owned_transport_output_batch_envelope(
            session_id=offer["sessionId"],
            source_batch_id=second_envelope["batchId"],
            sink_node_id="node-a",
            metadata={"nextStage": "requester", "finalOutput": True},
            policy=policy,
        )
        delivered = record_cai_owned_transport_batch_envelope(
            offer["sessionId"],
            final_envelope,
            local_node_id="node-a",
            policy=policy,
        )
        final_output = latest_cai_owned_transport_final_output(
            offer["sessionId"],
            requester_node_id="node-a",
            policy=policy,
        )
        final_result = await_cai_owned_transport_session_final_result(
            offer["sessionId"],
            requester_node_id="node-a",
            timeout_sec=0,
            policy=policy,
        )
        received_work_for_requester = list_cai_owned_transport_batch_inbox(
            "node-a",
            session_id=offer["sessionId"],
            policy=policy,
        )

    assert delivered.batch_records[-1]["status"] == "delivered"
    assert delivered.batch_records[-1]["sinkNodeId"] == "node-a"
    assert delivered.batch_records[-1]["metadata"]["finalOutput"] is True
    assert final_output is not None
    assert final_output["payload"] == b"b:a:worker-client-prompt"
    assert received_work_for_requester == []
    assert final_result["status"] == "completed"
    assert final_result["proofVerified"] is True
    assert final_result["proof"]["executionAudit"]["finalOutputBatchIds"] == [
        final_envelope["batchId"]
    ]
    assert [item["nodeId"] for item in final_result["proof"]["shardReceipts"]] == [
        "node-a",
        "node-b",
    ]


def test_cai_owned_transport_completion_accepts_remote_receipt_chain_gaps() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        offer = build_cai_owned_transport_session_offer(
            instance_id="instance-remote-receipts",
            participant_node_ids=["node-user", "node-a", "node-b"],
            executor_node_ids=["node-a", "node-b"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-remote-receipts",
            source_node_id="node-user",
        )
        dag = build_cai_owned_transport_execution_dag(
            session_id=offer["sessionId"],
            requester_node_id="node-user",
            executor_node_ids=["node-a", "node-b"],
            total_layer_count=28,
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-remote-receipts",
        )
        offer["routePolicy"] = {
            "executionDag": dag,
            "executionDagHashSha256Hex": dag["dagHashSha256Hex"],
        }
        stages = dag["stages"]
        create_cai_owned_transport_session_from_offer(
            offer,
            session_id=offer["sessionId"],
            local_node_id="node-user",
            policy=policy,
        )

        first_payload = b"user-prompt"
        first_metadata = build_cai_owned_transport_frame_metadata(
            model_id="cai-network/Qwen3-0.6B-GGUF",
            frame_kind="activation",
            tokenizer_config_hash="ef" * 32,
            layer_start=0,
            layer_end=14,
            token_start=0,
            token_end=1,
            dtype="bytes",
            shape=[len(first_payload)],
            sequence=0,
            payload_sha256_hex=hashlib.sha256(first_payload).hexdigest(),
        )
        first_metadata["stageId"] = stages[0]["stageId"]
        first_envelope = build_cai_owned_transport_batch_envelope(
            session_id=offer["sessionId"],
            phase="prefill_activation_batches",
            source_node_id="node-user",
            sink_node_id="node-a",
            sequence=0,
            payload=first_payload,
            metadata=first_metadata,
        )
        record_cai_owned_transport_batch_envelope(
            offer["sessionId"],
            first_envelope,
            local_node_id="node-a",
            policy=policy,
        )
        complete_cai_owned_transport_batch_processing(
            offer["sessionId"],
            first_envelope["batchId"],
            node_id="node-a",
            output_payload=b"local-prefill",
            policy=policy,
        )

        remote_prefill_batch_id = "caibatch_remote_prefill"
        local_decode_payload = b"remote-prefill-output"
        local_decode_metadata = build_cai_owned_transport_frame_metadata(
            model_id="cai-network/Qwen3-0.6B-GGUF",
            frame_kind="decode",
            tokenizer_config_hash="ef" * 32,
            layer_start=0,
            layer_end=14,
            token_start=0,
            token_end=1,
            dtype="bytes",
            shape=[len(local_decode_payload)],
            sequence=2,
            payload_sha256_hex=hashlib.sha256(local_decode_payload).hexdigest(),
        )
        local_decode_metadata["previousBatchId"] = remote_prefill_batch_id
        local_decode_metadata["stageId"] = stages[2]["stageId"]
        local_decode_envelope = build_cai_owned_transport_batch_envelope(
            session_id=offer["sessionId"],
            phase="decode_activation_batches",
            source_node_id="node-b",
            sink_node_id="node-a",
            sequence=2,
            payload=local_decode_payload,
            metadata=local_decode_metadata,
        )
        record_cai_owned_transport_batch_envelope(
            offer["sessionId"],
            local_decode_envelope,
            local_node_id="node-a",
            policy=policy,
        )
        complete_cai_owned_transport_batch_processing(
            offer["sessionId"],
            local_decode_envelope["batchId"],
            node_id="node-a",
            output_payload=b"local-decode",
            policy=policy,
        )

        local_record = list_cai_owned_transport_sessions(policy)[0]
        local_receipt = next(
            item
            for item in cai_owned_transport_shard_receipts_from_processed_batches(
                local_record
            )
            if item["nodeId"] == "node-a"
        )
        record_cai_owned_transport_shard_receipt(
            offer["sessionId"],
            node_id="node-a",
            chain_id="mainnet",
            activation_batch_count=local_receipt["activationBatchCount"],
            decode_batch_count=local_receipt["decodeBatchCount"],
            layer_start=local_receipt["layerStart"],
            layer_end=local_receipt["layerEnd"],
            metrics=local_receipt["metrics"],
            batch_ids=local_receipt["batchIds"],
            stage_ids=local_receipt["stageIds"],
            sequences=local_receipt["sequences"],
            input_payload_sha256_hexes=local_receipt["inputPayloadSha256Hexes"],
            output_payload_sha256_hexes=local_receipt["outputPayloadSha256Hexes"],
            hash_chain_sha256_hexes=local_receipt["hashChainSha256Hexes"],
            route_audits=local_receipt["routeAudits"],
            runtime_audits=local_receipt["runtimeAudits"],
            policy=policy,
        )
        remote_decode_batch_id = "caibatch_remote_decode"
        record_cai_owned_transport_shard_receipt(
            offer["sessionId"],
            node_id="node-b",
            chain_id="mainnet",
            activation_batch_count=1,
            decode_batch_count=1,
            layer_start=14,
            layer_end=28,
            metrics={"processedBatchCount": 2},
            batch_ids=[remote_prefill_batch_id, remote_decode_batch_id],
            stage_ids=[stages[1]["stageId"], stages[3]["stageId"]],
            sequences=[1, 3],
            policy=policy,
        )

        final_envelope = build_cai_owned_transport_batch_envelope(
            session_id=offer["sessionId"],
            phase="decode_activation_batches",
            source_node_id="node-b",
            sink_node_id="node-user",
            sequence=4,
            payload=b"final-output",
            metadata={
                "payloadRole": "shard_output",
                "finalOutput": True,
                "previousBatchId": remote_decode_batch_id,
                "stageId": "final_result",
            },
        )
        record_cai_owned_transport_batch_envelope(
            offer["sessionId"],
            final_envelope,
            local_node_id="node-user",
            policy=policy,
        )
        final_result = await_cai_owned_transport_session_final_result(
            offer["sessionId"],
            requester_node_id="node-user",
            timeout_sec=0.5,
            poll_interval_sec=0.01,
            policy=policy,
        )

    assert final_result["status"] == "completed"
    assert final_result["proofVerified"] is True
    assert set(final_result["proof"]["executionAudit"]["receiptBatchIds"]) >= {
        first_envelope["batchId"],
        local_decode_envelope["batchId"],
        remote_prefill_batch_id,
        remote_decode_batch_id,
    }
    assert final_result["proof"]["executionAudit"]["executionDag"]["missingStageIds"] == []


def test_cai_owned_transport_completion_accepts_remote_only_executor_receipt() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        offer = build_cai_owned_transport_session_offer(
            instance_id="instance-remote-only-receipt",
            participant_node_ids=["node-user", "node-b"],
            executor_node_ids=["node-b"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-remote-only-receipt",
            source_node_id="node-user",
        )
        dag = build_cai_owned_transport_execution_dag(
            session_id=offer["sessionId"],
            requester_node_id="node-user",
            executor_node_ids=["node-b"],
            total_layer_count=28,
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-remote-only-receipt",
        )
        offer["routePolicy"] = {
            "executionDag": dag,
            "executionDagHashSha256Hex": dag["dagHashSha256Hex"],
        }
        create_cai_owned_transport_session_from_offer(
            offer,
            session_id=offer["sessionId"],
            local_node_id="node-user",
            policy=policy,
        )
        record_cai_owned_transport_shard_receipt(
            offer["sessionId"],
            node_id="node-b",
            chain_id="mainnet",
            activation_batch_count=1,
            decode_batch_count=1,
            layer_start=0,
            layer_end=28,
            metrics={"processedBatchCount": 2},
            batch_ids=["caibatch_remote_prefill", "caibatch_remote_decode"],
            stage_ids=[dag["stages"][0]["stageId"], dag["stages"][1]["stageId"]],
            sequences=[0, 1],
            policy=policy,
        )
        final_envelope = build_cai_owned_transport_batch_envelope(
            session_id=offer["sessionId"],
            phase="decode_activation_batches",
            source_node_id="node-b",
            sink_node_id="node-user",
            sequence=2,
            payload=b"remote-final-output",
            metadata={
                "payloadRole": "shard_output",
                "finalOutput": True,
                "previousBatchId": "caibatch_remote_decode",
                "stageId": "final_result",
            },
        )
        record_cai_owned_transport_batch_envelope(
            offer["sessionId"],
            final_envelope,
            local_node_id="node-user",
            policy=policy,
        )
        final_result = await_cai_owned_transport_session_final_result(
            offer["sessionId"],
            requester_node_id="node-user",
            timeout_sec=0,
            policy=policy,
        )

    assert final_result["status"] == "completed"
    assert final_result["proofVerified"] is True
    assert final_result["finalOutput"]["payload"] == b"remote-final-output"
    assert final_result["proof"]["executionAudit"]["processedBatchCount"] == 0
    assert set(final_result["proof"]["executionAudit"]["receiptBatchIds"]) == {
        "caibatch_remote_prefill",
        "caibatch_remote_decode",
    }


def test_cai_owned_transport_session_completion_requires_execution_dag_stage_coverage() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        dag = build_cai_owned_transport_execution_dag(
            session_id="caiot_dag_missing_stage",
            requester_node_id="node-user",
            executor_node_ids=["node-a"],
            total_layer_count=28,
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-dag-missing-stage",
        )
        created = create_cai_owned_transport_session(
            session_id=dag["sessionId"],
            instance_id="instance-dag-missing-stage",
            participant_node_ids=["node-user", "node-a"],
            executor_node_ids=["node-a"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-dag-missing-stage",
            source_node_id="node-user",
            route_policy={"executionDag": dag},
            policy=policy,
        )
        first_envelope = build_cai_owned_transport_batch_envelope(
            session_id=created.session_id,
            phase="prefill_activation_batches",
            source_node_id="node-user",
            sink_node_id="node-a",
            sequence=0,
            payload=b"user-prompt",
            metadata={"stageId": dag["stages"][0]["stageId"]},
        )
        record_cai_owned_transport_batch_envelope(
            created.session_id,
            first_envelope,
            local_node_id="node-a",
            policy=policy,
        )
        complete_cai_owned_transport_batch_processing(
            created.session_id,
            first_envelope["batchId"],
            node_id="node-a",
            output_payload=b"prefill-output",
            policy=policy,
        )
        completion_error = None
        try:
            complete_cai_owned_transport_session(created.session_id, policy=policy)
        except ValueError as exc:
            completion_error = str(exc)
        record = list_cai_owned_transport_sessions(policy)[0]

    assert completion_error is not None
    assert dag["stages"][1]["stageId"] in completion_error
    assert record.status == "failed"
    assert record.last_error == completion_error


def test_cai_owned_transport_session_completion_accepts_execution_dag_and_final_output() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        dag = build_cai_owned_transport_execution_dag(
            session_id="caiot_dag_happy",
            requester_node_id="node-user",
            executor_node_ids=["node-a"],
            total_layer_count=28,
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-dag-happy",
        )
        created = create_cai_owned_transport_session(
            session_id=dag["sessionId"],
            instance_id="instance-dag-happy",
            participant_node_ids=["node-user", "node-a"],
            executor_node_ids=["node-a"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-dag-happy",
            source_node_id="node-user",
            route_policy={"executionDag": dag},
            policy=policy,
        )
        first_envelope = build_cai_owned_transport_batch_envelope(
            session_id=created.session_id,
            phase="prefill_activation_batches",
            source_node_id="node-user",
            sink_node_id="node-a",
            sequence=0,
            payload=b"user-prompt",
            metadata={"stageId": dag["stages"][0]["stageId"]},
        )
        record_cai_owned_transport_batch_envelope(
            created.session_id,
            first_envelope,
            local_node_id="node-a",
            policy=policy,
        )
        complete_cai_owned_transport_batch_processing(
            created.session_id,
            first_envelope["batchId"],
            node_id="node-a",
            output_payload=b"prefill-output",
            policy=policy,
        )
        second_envelope = build_cai_owned_transport_output_batch_envelope(
            session_id=created.session_id,
            source_batch_id=first_envelope["batchId"],
            sink_node_id="node-a",
            phase="decode_activation_batches",
            sequence=1,
            metadata={"stageId": dag["stages"][1]["stageId"]},
            policy=policy,
        )
        record_cai_owned_transport_batch_envelope(
            created.session_id,
            second_envelope,
            local_node_id="node-a",
            policy=policy,
        )
        complete_cai_owned_transport_batch_processing(
            created.session_id,
            second_envelope["batchId"],
            node_id="node-a",
            output_payload=b"decode-output",
            policy=policy,
        )
        final_envelope = build_cai_owned_transport_output_batch_envelope(
            session_id=created.session_id,
            source_batch_id=second_envelope["batchId"],
            sink_node_id="node-user",
            metadata={"nextStage": "requester"},
            policy=policy,
        )
        record_cai_owned_transport_batch_envelope(
            created.session_id,
            final_envelope,
            local_node_id="node-user",
            policy=policy,
        )
        result = await_cai_owned_transport_session_final_result(
            created.session_id,
            requester_node_id="node-user",
            timeout_sec=0,
            policy=policy,
        )

    assert result["status"] == "completed"
    assert result["proofVerified"] is True
    assert result["finalOutput"]["payload"] == b"decode-output"
    assert result["proof"]["executionAudit"]["executionDag"]["verified"] is True
    assert result["proof"]["executionAudit"]["executionDag"]["processedStageIds"] == [
        dag["stages"][0]["stageId"],
        dag["stages"][1]["stageId"],
    ]
    assert result["proof"]["executionAudit"]["executionDag"]["finalOutputBatchIds"] == [
        final_envelope["batchId"]
    ]


def test_cai_owned_transport_final_result_timeout_marks_session_failed() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        created = create_cai_owned_transport_session(
            session_id="caiot_final_timeout",
            instance_id="instance-final-timeout",
            participant_node_ids=["node-user", "node-a"],
            executor_node_ids=["node-a"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-final-timeout",
            source_node_id="node-user",
            policy=policy,
        )
        result = await_cai_owned_transport_session_final_result(
            created.session_id,
            requester_node_id="node-user",
            timeout_sec=0,
            policy=policy,
        )
        record = list_cai_owned_transport_sessions(policy)[0]

    assert result["status"] == "failed"
    assert result["proofVerified"] is False
    assert result["session"]["status"] == "failed"
    assert result["session"]["lastError"] == result["error"]
    assert record.status == "failed"
    assert record.last_error == result["error"]


def test_cai_owned_transport_rejects_wrong_chain_id() -> None:
    offer = build_cai_owned_transport_session_offer(
        instance_id="instance-offer-testnet",
        participant_node_ids=["node-a", "node-b"],
        task_id="task-chain",
        source_node_id="node-a",
        chain_id=ChainNetwork.TESTNET.value,
    )
    offer_valid, offer_error = validate_cai_owned_transport_session_offer(
        offer,
        session_id=offer["sessionId"],
        local_node_id="node-b",
    )
    envelope = build_cai_owned_transport_batch_envelope(
        session_id="session-chain",
        phase="prefill_activation_batches",
        source_node_id="node-a",
        sink_node_id="node-b",
        sequence=1,
        payload=b"activation-batch",
        chain_id=ChainNetwork.TESTNET.value,
    )
    envelope_valid, envelope_error = validate_cai_owned_transport_batch_envelope(
        envelope,
        session_id="session-chain",
        participant_node_ids=["node-a", "node-b"],
    )
    proof = build_cai_owned_transport_execution_proof(
        session_id="session-chain",
        instance_id="instance-chain",
        participant_node_ids=["node-a", "node-b"],
        chain_id=ChainNetwork.TESTNET.value,
        activation_batch_count=1,
        decode_batch_count=1,
    )
    proof_valid, proof_error = validate_cai_owned_transport_execution_proof(
        proof,
        participant_node_ids=["node-a", "node-b"],
    )

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        testnet_policy = WalletPolicy(
            chain_network=ChainNetwork.TESTNET,
            wallet_data_dirname=".tmp-cai-owned-transport-testnet",
        )
        accepted_record = create_cai_owned_transport_session_from_offer(
            offer,
            session_id=offer["sessionId"],
            local_node_id="node-b",
            policy=testnet_policy,
        )
        missing_receipt_chain_error = None
        try:
            record_cai_owned_transport_shard_receipt(
                offer["sessionId"],
                node_id="node-b",
                policy=testnet_policy,
            )
        except ValueError as exc:
            missing_receipt_chain_error = str(exc)
        wrong_receipt_chain_error = None
        try:
            record_cai_owned_transport_shard_receipt(
                offer["sessionId"],
                node_id="node-b",
                chain_id=ChainNetwork.MAINNET.value,
                policy=testnet_policy,
            )
        except ValueError as exc:
            wrong_receipt_chain_error = str(exc)
        accepted_receipt_record = record_cai_owned_transport_shard_receipt(
            offer["sessionId"],
            node_id="node-b",
            chain_id=ChainNetwork.TESTNET.value,
            policy=testnet_policy,
        )

    assert offer_valid is False
    assert offer_error == (
        "CAI-owned transport session offer is for chain 'testnet', "
        "expected 'mainnet'."
    )
    assert envelope_valid is False
    assert envelope_error == (
        "CAI-owned transport batch envelope is for chain 'testnet', "
        "expected 'mainnet'."
    )
    assert proof_valid is False
    assert proof_error == (
        "CAI-owned transport execution proof is for chain 'testnet', "
        "expected 'mainnet'."
    )
    assert accepted_record.chain_id == "testnet"
    assert missing_receipt_chain_error == (
        "CAI-owned transport shard receipt chain id is missing."
    )
    assert wrong_receipt_chain_error == (
        "CAI-owned transport shard receipt chain id does not match."
    )
    assert accepted_receipt_record.shard_receipts[0]["chainId"] == "testnet"


def test_cai_owned_transport_batch_replay_rejects_different_payload_hash() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        created = create_cai_owned_transport_session(
            session_id="caiot_replay",
            instance_id="instance-replay",
            participant_node_ids=["node-a", "node-b"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-replay",
            policy=policy,
        )
        first_record = record_cai_owned_transport_batch(
            created.session_id,
            batch_id="caibatch_replay",
            phase="prefill_activation_batches",
            source_node_id="node-a",
            sink_node_id="node-b",
            payload_size_bytes=64,
            payload_sha256_hex="aa" * 32,
            status="received",
            policy=policy,
        )
        repeated_record = record_cai_owned_transport_batch(
            created.session_id,
            batch_id="caibatch_replay",
            phase="prefill_activation_batches",
            source_node_id="node-a",
            sink_node_id="node-b",
            payload_size_bytes=64,
            payload_sha256_hex="aa" * 32,
            status="received",
            policy=policy,
        )
        replay_error = None
        try:
            record_cai_owned_transport_batch(
                created.session_id,
                batch_id="caibatch_replay",
                phase="prefill_activation_batches",
                source_node_id="node-a",
                sink_node_id="node-b",
                payload_size_bytes=64,
                payload_sha256_hex="bb" * 32,
                status="received",
                policy=policy,
            )
        except ValueError as exc:
            replay_error = str(exc)

    assert len(first_record.batch_records) == 1
    assert len(repeated_record.batch_records) == 1
    assert replay_error == (
        "CAI-owned transport batch replay payload hash does not match."
    )


def test_cai_owned_transport_session_offer_rejects_wrong_local_node() -> None:
    offer = build_cai_owned_transport_session_offer(
        instance_id="instance-offer",
        participant_node_ids=["node-a", "node-b"],
        task_id="task-1",
        source_node_id="node-a",
    )

    valid, error = validate_cai_owned_transport_session_offer(
        offer,
        session_id=offer["sessionId"],
        local_node_id="node-c",
    )

    assert valid is False
    assert error == "CAI-owned transport session offer does not include local node."


def test_cai_owned_transport_batch_envelope_records_metadata_on_recipient() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        offer = build_cai_owned_transport_session_offer(
            instance_id="instance-offer",
            participant_node_ids=["node-a", "node-b"],
            task_id="task-1",
            source_node_id="node-a",
        )
        create_cai_owned_transport_session_from_offer(
            offer,
            session_id=offer["sessionId"],
            local_node_id="node-b",
            policy=policy,
        )
        envelope = build_cai_owned_transport_batch_envelope(
            session_id=offer["sessionId"],
            phase="prefill_activation_batches",
            source_node_id="node-a",
            sink_node_id="node-b",
            sequence=1,
            payload=b"activation-batch",
        )
        record = record_cai_owned_transport_batch_envelope(
            offer["sessionId"],
            envelope,
            local_node_id="node-b",
            policy=policy,
        )
        payload_path = cai_owned_transport_batch_payload_path(
            offer["sessionId"],
            envelope["batchId"],
            policy,
        )
        stored_payload = read_cai_owned_transport_batch_payload(
            offer["sessionId"],
            envelope["batchId"],
            policy,
        )
        payload_exists = payload_path.exists()
        received_inbox = list_cai_owned_transport_batch_inbox(
            "node-b",
            policy=policy,
        )
        claimed_record = claim_cai_owned_transport_batch(
            offer["sessionId"],
            envelope["batchId"],
            node_id="node-b",
            runtime_id="runtime-1",
            policy=policy,
        )
        repeated_claim = claim_cai_owned_transport_batch(
            offer["sessionId"],
            envelope["batchId"],
            node_id="node-b",
            runtime_id="runtime-1",
            policy=policy,
        )
        wrong_node_error = None
        try:
            claim_cai_owned_transport_batch(
                offer["sessionId"],
                envelope["batchId"],
                node_id="node-a",
                runtime_id="runtime-a",
                policy=policy,
            )
        except ValueError as exc:
            wrong_node_error = str(exc)
        processed_record = mark_cai_owned_transport_batch_status(
            offer["sessionId"],
            envelope["batchId"],
            status="processed",
            node_id="node-b",
            metrics={"tokens": 3},
            policy=policy,
        )
        received_after_processing = list_cai_owned_transport_batch_inbox(
            "node-b",
            policy=policy,
        )
        all_inbox = list_cai_owned_transport_batch_inbox(
            "node-b",
            status=None,
            policy=policy,
        )
        payload_path.write_bytes(b"corrupt")
        corrupt_error = None
        try:
            read_cai_owned_transport_batch_payload(
                offer["sessionId"],
                envelope["batchId"],
                policy,
            )
        except ValueError as exc:
            corrupt_error = str(exc)

    assert record.status == "running"
    assert len(record.batch_records) == 1
    assert record.batch_records[0]["sourceNodeId"] == "node-a"
    assert record.batch_records[0]["sinkNodeId"] == "node-b"
    assert record.batch_records[0]["payloadSha256Hex"] == envelope["payloadSha256Hex"]
    assert payload_exists is True
    assert stored_payload == b"activation-batch"
    assert cai_owned_transport_batch_payload_bytes(envelope) == b"activation-batch"
    assert (
        record.batch_records[0]["metadata"]["transportBatchId"]
        == envelope["batchId"]
    )
    assert (
        record.batch_records[0]["metadata"]["payloadStorageKey"]
        == f"{offer['sessionId']}/{envelope['batchId']}.bin"
    )
    assert record.batch_records[0]["chainId"] == "mainnet"
    assert record.batch_records[0]["metadata"]["chainId"] == "mainnet"
    assert len(received_inbox) == 1
    assert received_inbox[0]["batch"]["status"] == "received"
    assert claimed_record.batch_records[0]["status"] == "processing"
    assert claimed_record.batch_records[0]["runtimeId"] == "runtime-1"
    assert claimed_record.batch_records[0]["attemptCount"] == 1
    assert claimed_record.batch_records[0]["heartbeatAt"]
    assert claimed_record.batch_records[0]["leaseExpiresAt"]
    assert claimed_record.batch_records[0]["leaseSeconds"] == 60.0
    assert repeated_claim.batch_records[0]["attemptCount"] == 1
    assert (
        wrong_node_error
        == "CAI-owned transport batch is not assigned to local node."
    )
    assert processed_record.batch_records[0]["status"] == "processed"
    assert processed_record.batch_records[0]["metrics"] == {"tokens": 3}
    assert received_after_processing == []
    assert len(all_inbox) == 1
    assert all_inbox[0]["batch"]["status"] == "processed"
    assert corrupt_error == "CAI-owned transport payload hash does not match."


def test_cleanup_cai_owned_transport_payload_storage_keeps_active_sessions() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        active = create_cai_owned_transport_session(
            session_id="caiot_active_cleanup",
            instance_id="instance-active-cleanup",
            participant_node_ids=["node-a", "node-b"],
            executor_node_ids=["node-b"],
            source_node_id="node-a",
            policy=policy,
        )
        completed = create_cai_owned_transport_session(
            session_id="caiot_completed_cleanup",
            instance_id="instance-completed-cleanup",
            participant_node_ids=["node-a", "node-b"],
            executor_node_ids=["node-b"],
            source_node_id="node-a",
            policy=policy,
        )
        active_envelope = build_cai_owned_transport_batch_envelope(
            session_id=active.session_id,
            phase="prefill_activation_batches",
            source_node_id="node-a",
            sink_node_id="node-b",
            sequence=1,
            payload=b"active-payload",
        )
        completed_envelope = build_cai_owned_transport_batch_envelope(
            session_id=completed.session_id,
            phase="prefill_activation_batches",
            source_node_id="node-a",
            sink_node_id="node-b",
            sequence=1,
            payload=b"completed-payload",
        )
        record_cai_owned_transport_batch_envelope(
            active.session_id,
            active_envelope,
            local_node_id="node-b",
            policy=policy,
        )
        record_cai_owned_transport_batch_envelope(
            completed.session_id,
            completed_envelope,
            local_node_id="node-b",
            policy=policy,
        )
        active_payload_path = cai_owned_transport_batch_payload_path(
            active.session_id,
            active_envelope["batchId"],
            policy,
        )
        completed_payload_path = cai_owned_transport_batch_payload_path(
            completed.session_id,
            completed_envelope["batchId"],
            policy,
        )
        records = list_cai_owned_transport_sessions(policy)
        for record in records:
            if record.session_id == completed.session_id:
                record.status = "completed"
                record.completed_at = "2000-01-01T00:00:00+00:00"
                record.updated_at = "2000-01-01T00:00:00+00:00"
        save_cai_owned_transport_sessions(records, policy)

        result = cleanup_cai_owned_transport_payload_storage(
            retention_seconds=0,
            policy=policy,
        )
        active_payload_exists = active_payload_path.exists()
        completed_payload_exists = completed_payload_path.exists()

    assert completed.session_id in result["deletedSessionIds"]
    assert active.session_id in result["skippedActiveSessionIds"]
    assert active_payload_exists is True
    assert completed_payload_exists is False


def test_submit_cai_owned_transport_session_offer_posts_to_peer() -> None:
    captured: dict[str, object] = {}
    offer = build_cai_owned_transport_session_offer(
        instance_id="instance-offer",
        participant_node_ids=["node-a", "node-b"],
        task_id="task-1",
        source_node_id="node-a",
    )

    def fake_urlopen(request, timeout: float):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["headers"] = {
            str(key).lower(): value for key, value in request.header_items()
        }
        return _FakeResponse({"status": "created", "sessionId": offer["sessionId"]})

    with patch.dict(
        "os.environ",
        {"CAI_PEER_TRANSPORT_TOKEN": "peer-secret"},
        clear=False,
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        response = submit_cai_owned_transport_session_offer(
            "http://worker:52415/",
            offer,
            timeout_sec=4.0,
        )

    assert response["status"] == "created"
    assert captured["url"] == (
        "http://worker:52415/v1/cai/transport/sessions/"
        f"{offer['sessionId']}/offer"
    )
    assert captured["timeout"] == 4.0
    assert captured["body"]["sessionId"] == offer["sessionId"]
    assert captured["headers"]["x-cai-chain-id"] == "mainnet"
    assert captured["headers"]["x-cai-transport-token"] == "peer-secret"


def test_cai_owned_transport_request_auth_validates_chain_and_token() -> None:
    headers = cai_owned_transport_auth_headers(
        chain_id="mainnet",
        auth_token="peer-secret",
    )
    valid, error = validate_cai_owned_transport_request_auth(
        headers,
        chain_id="mainnet",
        auth_token="peer-secret",
    )
    wrong_token_valid, wrong_token_error = validate_cai_owned_transport_request_auth(
        headers,
        chain_id="mainnet",
        auth_token="other-secret",
    )
    wrong_chain_valid, wrong_chain_error = validate_cai_owned_transport_request_auth(
        headers,
        chain_id=ChainNetwork.TESTNET.value,
        auth_token="peer-secret",
    )
    missing_required_valid, missing_required_error = (
        validate_cai_owned_transport_request_auth(
            {"X-CAI-Chain-Id": "mainnet"},
            chain_id="mainnet",
            require_auth=True,
        )
    )

    assert valid is True
    assert error is None
    assert wrong_token_valid is False
    assert wrong_token_error == "CAI-owned transport peer auth token is invalid."
    assert wrong_chain_valid is False
    assert wrong_chain_error == (
        "CAI-owned transport request is for chain 'mainnet', expected 'testnet'."
    )
    assert missing_required_valid is False
    assert missing_required_error == (
        "CAI-owned transport peer auth token is not configured."
    )


def test_submit_cai_owned_transport_helpers_accept_payload_chain_id() -> None:
    captured_bodies: list[dict[str, object]] = []
    offer = build_cai_owned_transport_session_offer(
        instance_id="instance-offer-testnet-submit",
        participant_node_ids=["node-a", "node-b"],
        task_id="task-submit-chain",
        source_node_id="node-a",
        chain_id=ChainNetwork.TESTNET.value,
    )
    envelope = build_cai_owned_transport_batch_envelope(
        session_id=offer["sessionId"],
        phase="decode_activation_batches",
        source_node_id="node-a",
        sink_node_id="node-b",
        sequence=2,
        payload=b"decode-testnet",
        chain_id=ChainNetwork.TESTNET.value,
    )

    def fake_urlopen(request, timeout: float):
        captured_bodies.append(json.loads(request.data.decode("utf-8")))
        return _FakeResponse({"status": "ok"})

    with patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        offer_response = submit_cai_owned_transport_session_offer(
            "http://worker:52415/",
            offer,
        )
        envelope_response = submit_cai_owned_transport_batch_envelope(
            "http://worker:52415/",
            offer["sessionId"],
            envelope,
        )

    assert offer_response["status"] == "ok"
    assert envelope_response["status"] == "ok"
    assert captured_bodies[0]["chainId"] == "testnet"
    assert captured_bodies[1]["chainId"] == "testnet"


def test_claim_next_cai_owned_transport_batch_returns_work_item() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        offer = build_cai_owned_transport_session_offer(
            instance_id="instance-claim-next",
            participant_node_ids=["node-a", "node-b"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-claim-next",
            source_node_id="node-a",
        )
        create_cai_owned_transport_session_from_offer(
            offer,
            session_id=offer["sessionId"],
            local_node_id="node-b",
            policy=policy,
        )
        envelope = build_cai_owned_transport_batch_envelope(
            session_id=offer["sessionId"],
            phase="decode_activation_batches",
            source_node_id="node-a",
            sink_node_id="node-b",
            sequence=7,
            payload=b"decode-work",
        )
        record_cai_owned_transport_batch_envelope(
            offer["sessionId"],
            envelope,
            local_node_id="node-b",
            policy=policy,
        )
        work_item = claim_next_cai_owned_transport_batch(
            "node-b",
            runtime_id="runtime-next",
            policy=policy,
        )
        no_more_work = claim_next_cai_owned_transport_batch(
            "node-b",
            runtime_id="runtime-next",
            policy=policy,
        )

    assert work_item is not None
    assert work_item["sessionId"] == offer["sessionId"]
    assert work_item["batch"]["batchId"] == envelope["batchId"]
    assert work_item["batch"]["status"] == "processing"
    assert work_item["batch"]["runtimeId"] == "runtime-next"
    assert work_item["payloadEndpoint"] == (
        f"/v1/cai/transport/sessions/{offer['sessionId']}/batches/"
        f"{envelope['batchId']}/payload"
    )
    assert work_item["payloadStorageKey"] == (
        f"{offer['sessionId']}/{envelope['batchId']}.bin"
    )
    assert no_more_work is None


def test_cai_owned_transport_local_runtime_auth_guards_claims() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        offer = build_cai_owned_transport_session_offer(
            instance_id="instance-runtime-auth",
            participant_node_ids=["node-a", "node-b"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-runtime-auth",
            source_node_id="node-a",
        )
        create_cai_owned_transport_session_from_offer(
            offer,
            session_id=offer["sessionId"],
            local_node_id="node-b",
            policy=policy,
        )
        envelope = build_cai_owned_transport_batch_envelope(
            session_id=offer["sessionId"],
            phase="decode_activation_batches",
            source_node_id="node-a",
            sink_node_id="node-b",
            sequence=8,
            payload=b"decode-runtime-auth",
        )
        record_cai_owned_transport_batch_envelope(
            offer["sessionId"],
            envelope,
            local_node_id="node-b",
            policy=policy,
        )
        with patch.dict(
            "os.environ",
            {"CAI_LOCAL_RUNTIME_TOKEN": "local-secret"},
            clear=False,
        ):
            missing_valid, missing_error = validate_cai_owned_transport_local_runtime_auth(
                require_auth=True,
            )
            wrong_valid, wrong_error = validate_cai_owned_transport_local_runtime_auth(
                "wrong-secret",
                require_auth=True,
            )
            with pytest.raises(ValueError, match="auth token is missing"):
                claim_next_cai_owned_transport_batch(
                    "node-b",
                    runtime_id="runtime-auth",
                    require_runtime_auth=True,
                    policy=policy,
                )
            with pytest.raises(ValueError, match="auth token is invalid"):
                claim_next_cai_owned_transport_batch(
                    "node-b",
                    runtime_id="runtime-auth",
                    runtime_auth_token="wrong-secret",
                    require_runtime_auth=True,
                    policy=policy,
                )
            work_item = claim_next_cai_owned_transport_batch(
                "node-b",
                runtime_id="runtime-auth",
                runtime_auth_token="local-secret",
                require_runtime_auth=True,
                policy=policy,
            )

    assert missing_valid is False
    assert missing_error == (
        "CAI-owned transport local runtime auth token is missing."
    )
    assert wrong_valid is False
    assert wrong_error == "CAI-owned transport local runtime auth token is invalid."
    assert work_item is not None
    assert work_item["batch"]["status"] == "processing"
    assert work_item["batch"]["runtimeId"] == "runtime-auth"


def test_cai_owned_transport_batch_heartbeat_and_stale_reclaim() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        offer = build_cai_owned_transport_session_offer(
            instance_id="instance-lease",
            participant_node_ids=["node-a", "node-b"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-lease",
            source_node_id="node-a",
        )
        create_cai_owned_transport_session_from_offer(
            offer,
            session_id=offer["sessionId"],
            local_node_id="node-b",
            policy=policy,
        )
        live_envelope = build_cai_owned_transport_batch_envelope(
            session_id=offer["sessionId"],
            phase="prefill_activation_batches",
            source_node_id="node-a",
            sink_node_id="node-b",
            sequence=1,
            payload=b"live-batch",
        )
        stale_envelope = build_cai_owned_transport_batch_envelope(
            session_id=offer["sessionId"],
            phase="prefill_activation_batches",
            source_node_id="node-a",
            sink_node_id="node-b",
            sequence=2,
            payload=b"stale-batch",
        )
        record_cai_owned_transport_batch_envelope(
            offer["sessionId"],
            live_envelope,
            local_node_id="node-b",
            policy=policy,
        )
        record_cai_owned_transport_batch_envelope(
            offer["sessionId"],
            stale_envelope,
            local_node_id="node-b",
            policy=policy,
        )
        claim_cai_owned_transport_batch(
            offer["sessionId"],
            live_envelope["batchId"],
            node_id="node-b",
            runtime_id="runtime-live",
            lease_seconds=30,
            policy=policy,
        )
        heartbeat_record = heartbeat_cai_owned_transport_batch(
            offer["sessionId"],
            live_envelope["batchId"],
            node_id="node-b",
            runtime_id="runtime-live",
            lease_seconds=45,
            policy=policy,
        )
        wrong_runtime_error = None
        try:
            heartbeat_cai_owned_transport_batch(
                offer["sessionId"],
                live_envelope["batchId"],
                node_id="node-b",
                runtime_id="runtime-other",
                policy=policy,
            )
        except ValueError as exc:
            wrong_runtime_error = str(exc)
        claim_cai_owned_transport_batch(
            offer["sessionId"],
            stale_envelope["batchId"],
            node_id="node-b",
            runtime_id="runtime-stale",
            lease_seconds=0,
            policy=policy,
        )
        reclaimed = claim_next_cai_owned_transport_batch(
            "node-b",
            runtime_id="runtime-new",
            lease_seconds=60,
            policy=policy,
        )

    live_batch = next(
        item
        for item in heartbeat_record.batch_records
        if item["batchId"] == live_envelope["batchId"]
    )
    assert live_batch["runtimeId"] == "runtime-live"
    assert live_batch["leaseSeconds"] == 45.0
    assert live_batch["heartbeatAt"]
    assert live_batch["leaseExpiresAt"]
    assert (
        wrong_runtime_error
        == "CAI-owned transport batch runtime id does not match."
    )
    assert reclaimed is not None
    assert reclaimed["batch"]["batchId"] == stale_envelope["batchId"]
    assert reclaimed["batch"]["runtimeId"] == "runtime-new"
    assert reclaimed["batch"]["previousRuntimeId"] == "runtime-stale"
    assert reclaimed["batch"]["attemptCount"] == 2
    assert reclaimed["batch"]["reclaimedAt"]


def test_reconcile_cai_owned_transport_session_timeouts_marks_unclaimed_batch_failed() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        offer = build_cai_owned_transport_session_offer(
            instance_id="instance-claim-timeout",
            participant_node_ids=["node-a", "node-b"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-claim-timeout",
            source_node_id="node-a",
        )
        create_cai_owned_transport_session_from_offer(
            offer,
            session_id=offer["sessionId"],
            local_node_id="node-b",
            policy=policy,
        )
        envelope = build_cai_owned_transport_batch_envelope(
            session_id=offer["sessionId"],
            phase="prefill_activation_batches",
            source_node_id="node-a",
            sink_node_id="node-b",
            sequence=1,
            payload=b"unclaimed-batch",
        )
        record_cai_owned_transport_batch_envelope(
            offer["sessionId"],
            envelope,
            local_node_id="node-b",
            policy=policy,
        )
        result = reconcile_cai_owned_transport_session_timeouts(
            offer["sessionId"],
            received_timeout_sec=0,
            policy=policy,
        )
        claim_after_timeout = claim_next_cai_owned_transport_batch(
            "node-b",
            runtime_id="runtime-after-timeout",
            policy=policy,
        )
        record = list_cai_owned_transport_sessions(policy)[0]

    assert result["status"] == "failed"
    assert result["timedOutBatchIds"] == [envelope["batchId"]]
    assert result["retryScheduledBatchIds"] == []
    assert claim_after_timeout is None
    assert record.status == "failed"
    assert record.batch_records[0]["status"] == "timed_out"
    assert record.batch_records[0]["timeoutReason"] == "claim_timeout"
    assert record.batch_records[0]["retryable"] is False
    assert record.last_error == (
        "CAI-owned transport batch was not claimed before coordinator timeout."
    )


def test_reconcile_cai_owned_transport_session_timeouts_retries_then_times_out_processing_batch() -> None:
    public_key_b64, signing_seed_b64 = _signing_material()
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        offer = build_cai_owned_transport_session_offer(
            instance_id="instance-processing-timeout",
            participant_node_ids=["node-a", "node-b"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-processing-timeout",
            source_node_id="node-a",
        )
        offer = sign_cai_owned_transport_session_offer(
            offer,
            signer_node_id="node-a",
            public_key_b64=public_key_b64,
            signing_seed_b64=signing_seed_b64,
            signer_wallet_id="wallet-node-a",
            signer_address="abcd1234",
        )
        create_cai_owned_transport_session_from_offer(
            offer,
            session_id=offer["sessionId"],
            local_node_id="node-b",
            policy=policy,
        )
        envelope = build_cai_owned_transport_batch_envelope(
            session_id=offer["sessionId"],
            phase="decode_activation_batches",
            source_node_id="node-a",
            sink_node_id="node-b",
            sequence=1,
            payload=b"processing-batch",
        )
        envelope = sign_cai_owned_transport_batch_envelope(
            envelope,
            signer_node_id="node-a",
            public_key_b64=public_key_b64,
            signing_seed_b64=signing_seed_b64,
            signer_wallet_id="wallet-node-a",
            signer_address="abcd1234",
        )
        record_cai_owned_transport_batch_envelope(
            offer["sessionId"],
            envelope,
            local_node_id="node-b",
            policy=policy,
        )
        claim_cai_owned_transport_batch(
            offer["sessionId"],
            envelope["batchId"],
            node_id="node-b",
            runtime_id="runtime-stale-1",
            lease_seconds=0,
            policy=policy,
        )
        retry_result = reconcile_cai_owned_transport_session_timeouts(
            offer["sessionId"],
            max_attempts=2,
            policy=policy,
        )
        retry_batch = list_cai_owned_transport_sessions(policy)[0].batch_records[0]
        claim_cai_owned_transport_batch(
            offer["sessionId"],
            envelope["batchId"],
            node_id="node-b",
            runtime_id="runtime-stale-2",
            lease_seconds=0,
            policy=policy,
        )
        timeout_result = reconcile_cai_owned_transport_session_timeouts(
            offer["sessionId"],
            max_attempts=2,
            policy=policy,
        )
        record = list_cai_owned_transport_sessions(policy)[0]

    assert retry_result["status"] == "running"
    assert retry_result["timedOutBatchIds"] == []
    assert retry_result["retryScheduledBatchIds"] == [envelope["batchId"]]
    assert retry_batch["status"] == "received"
    assert retry_batch["previousRuntimeId"] == "runtime-stale-1"
    assert "runtimeId" not in retry_batch
    assert retry_batch["lastError"] == (
        "CAI-owned transport batch lease expired; retry scheduled."
    )
    assert timeout_result["status"] == "failed"
    assert timeout_result["timedOutBatchIds"] == [envelope["batchId"]]
    assert record.status == "failed"
    assert record.batch_records[0]["status"] == "timed_out"
    assert record.batch_records[0]["timeoutReason"] == "lease_timeout"
    assert record.batch_records[0]["retryable"] is False


def test_submit_cai_owned_transport_batch_envelope_posts_to_peer() -> None:
    captured: dict[str, object] = {}
    envelope = build_cai_owned_transport_batch_envelope(
        session_id="session-1",
        phase="decode_activation_batches",
        source_node_id="node-a",
        sink_node_id="node-b",
        sequence=3,
        payload=b"decode-batch",
    )

    def fake_urlopen(request, timeout: float):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"status": "running", "sessionId": "session-1"})

    with patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        response = submit_cai_owned_transport_batch_envelope(
            "http://worker:52415/",
            "session-1",
            envelope,
            timeout_sec=4.5,
        )

    assert response["status"] == "running"
    assert captured["url"] == (
        "http://worker:52415/v1/cai/transport/sessions/"
        "session-1/batch-envelopes"
    )
    assert captured["timeout"] == 4.5
    assert captured["body"]["batchId"] == envelope["batchId"]


def test_submit_cai_owned_transport_batch_envelope_tries_peer_urls() -> None:
    captured_urls: list[str] = []
    envelope = build_cai_owned_transport_batch_envelope(
        session_id="session-1",
        phase="decode_activation_batches",
        source_node_id="node-a",
        sink_node_id="node-b",
        sequence=4,
        payload=b"decode-batch",
    )

    def fake_urlopen(request, timeout: float):
        captured_urls.append(request.full_url)
        if "bad-peer" in request.full_url:
            raise OSError("peer unavailable")
        return _FakeResponse({"status": "running", "sessionId": "session-1"})

    with patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        response = submit_cai_owned_transport_batch_envelope_to_any(
            ["http://bad-peer:52415", "http://good-peer:52415"],
            "session-1",
            envelope,
            timeout_sec=4.5,
        )

    assert response["status"] == "running"
    assert response["peerCaiUrl"] == "http://good-peer:52415"
    assert response["attemptedPeerCaiUrlCount"] == 2
    assert response["routeAudit"]["selectedPeerCaiUrl"] == "http://good-peer:52415"
    assert response["routeAudit"]["fallbackCount"] == 1
    assert response["routeAudit"]["attempts"][0]["status"] == "failed"
    assert response["routeAudit"]["attempts"][1]["status"] == "ok"
    assert captured_urls == [
        "http://bad-peer:52415/v1/cai/transport/sessions/session-1/batch-envelopes",
        "http://good-peer:52415/v1/cai/transport/sessions/session-1/batch-envelopes",
    ]


def test_submit_cai_owned_transport_batch_envelope_prefers_direct_over_overlay() -> None:
    captured_urls: list[str] = []
    envelope = build_cai_owned_transport_batch_envelope(
        session_id="session-direct-first",
        phase="decode_activation_batches",
        source_node_id="node-a",
        sink_node_id="node-b",
        sequence=4,
        payload=b"decode-batch",
    )

    def fake_urlopen(request, timeout: float):
        captured_urls.append(request.full_url)
        return _FakeResponse(
            {"status": "running", "sessionId": "session-direct-first"}
        )

    with patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        response = submit_cai_owned_transport_batch_envelope_to_any(
            [
                "cai-overlay:http://bootstrap:52415?targetNodeId=node-b&relayRole=bootstrap",
                "http://direct-peer:52415",
            ],
            "session-direct-first",
            envelope,
            timeout_sec=4.5,
        )

    assert response["status"] == "running"
    assert response["peerCaiUrl"] == "http://direct-peer:52415"
    assert response["attemptedPeerCaiUrlCount"] == 2
    assert response["routeAudit"]["fallbackCount"] == 0
    assert captured_urls == [
        "http://direct-peer:52415/v1/cai/transport/sessions/"
        "session-direct-first/batch-envelopes",
    ]


def test_submit_cai_owned_transport_batch_envelope_prefers_ordinary_relay() -> None:
    captured_urls: list[str] = []
    envelope = build_cai_owned_transport_batch_envelope(
        session_id="session-ordinary-relay",
        phase="decode_activation_batches",
        source_node_id="node-a",
        sink_node_id="node-b",
        sequence=4,
        payload=b"decode-batch",
    )

    def fake_urlopen(request, timeout: float):
        captured_urls.append(request.full_url)
        return _FakeResponse(
            {"status": "running", "sessionId": "session-ordinary-relay"}
        )

    with patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        response = submit_cai_owned_transport_batch_envelope_to_any(
            [
                "cai-overlay:http://bootstrap:52415?targetNodeId=node-b&relayRole=bootstrap",
                "cai-overlay:http://ordinary:52415?targetNodeId=node-b&relayRole=ordinary",
            ],
            "session-ordinary-relay",
            envelope,
            timeout_sec=4.5,
        )

    assert response["status"] == "running"
    assert response["peerCaiUrl"] == (
        "cai-overlay:http://ordinary:52415?targetNodeId=node-b&relayRole=ordinary"
    )
    assert captured_urls == [
        "http://ordinary:52415/v1/cai/transport/overlay/send",
    ]


def test_cai_owned_transport_completion_can_use_processed_batches() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        created = create_cai_owned_transport_session(
            instance_id="instance-batches",
            participant_node_ids=["node-a", "node-b"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-1",
            policy=policy,
        )
        for node_id, layer_start, layer_end in [
            ("node-a", 0, 14),
            ("node-b", 14, 28),
        ]:
            prefill_batch_id = f"caibatch_{node_id.replace('-', '_')}_prefill"
            decode_batch_id = f"caibatch_{node_id.replace('-', '_')}_decode"
            record_cai_owned_transport_batch(
                created.session_id,
                batch_id=prefill_batch_id,
                phase="prefill_activation_batches",
                source_node_id="node-a",
                sink_node_id=node_id,
                payload_size_bytes=64,
                payload_sha256_hex="cd" * 32,
                metadata={"layerStart": layer_start, "layerEnd": layer_end},
                status="received",
                policy=policy,
            )
            complete_cai_owned_transport_batch_processing(
                created.session_id,
                prefill_batch_id,
                node_id=node_id,
                output_payload=f"{node_id}:prefill".encode("utf-8"),
                policy=policy,
            )
            record_cai_owned_transport_batch(
                created.session_id,
                batch_id=decode_batch_id,
                phase="decode_activation_batches",
                source_node_id="node-a",
                sink_node_id=node_id,
                payload_size_bytes=32,
                payload_sha256_hex="ef" * 32,
                metadata={"layerStart": layer_start, "layerEnd": layer_end},
                status="received",
                policy=policy,
            )
            complete_cai_owned_transport_batch_processing(
                created.session_id,
                decode_batch_id,
                node_id=node_id,
                output_payload=f"{node_id}:decode".encode("utf-8"),
                policy=policy,
            )
        record = list_cai_owned_transport_sessions(policy)[0]
        receipts = cai_owned_transport_shard_receipts_from_processed_batches(record)
        audit_valid, audit_error, execution_audit = (
            validate_cai_owned_transport_session_execution_audit(record, policy=policy)
        )
        completed = complete_cai_owned_transport_session(
            created.session_id,
            policy=policy,
        )

    assert [item["nodeId"] for item in receipts] == ["node-a", "node-b"]
    assert audit_valid is True
    assert audit_error is None
    assert execution_audit["processedBatchCount"] == 4
    assert receipts[0]["activationBatchCount"] == 1
    assert receipts[0]["decodeBatchCount"] == 1
    assert receipts[0]["metrics"]["processedBatchCount"] == 2
    assert completed.status == "completed"
    assert completed.proof is not None
    assert completed.proof["activationBatchCount"] == 1
    assert completed.proof["decodeBatchCount"] == 1
    assert len(completed.proof["shardReceipts"]) == 2
    assert completed.proof["executionAudit"]["verified"] is True


def test_direct_final_decode_batch_counts_input_as_prompt_tokens() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        created = create_cai_owned_transport_session(
            session_id="caiot_direct_final_receipt",
            instance_id="instance-direct-final-receipt",
            participant_node_ids=["node-user", "node-a"],
            executor_node_ids=["node-a"],
            source_node_id="node-user",
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-direct-final-receipt",
            policy=policy,
        )
        record_cai_owned_transport_batch(
            created.session_id,
            batch_id="caibatch_direct_final",
            phase="decode_activation_batches",
            source_node_id="node-user",
            sink_node_id="node-a",
            payload_size_bytes=4,
            payload_sha256_hex="de" * 32,
            metadata={
                "layerStart": 0,
                "layerEnd": 28,
                "singleExecutorDirectFinalOutput": True,
            },
            status="received",
            policy=policy,
        )
        complete_cai_owned_transport_batch_processing(
            created.session_id,
            "caibatch_direct_final",
            node_id="node-a",
            output_payload=b"5",
            metrics={"inputTokenCount": 1, "outputTokenCount": 1},
            policy=policy,
        )
        record = list_cai_owned_transport_sessions(policy)[0]
        receipts = cai_owned_transport_shard_receipts_from_processed_batches(record)

    assert receipts[0]["activationBatchCount"] == 0
    assert receipts[0]["decodeBatchCount"] == 1
    assert receipts[0]["metrics"]["promptTokenCount"] == 1
    assert receipts[0]["metrics"]["completionTokenCount"] == 1


def test_complete_cai_owned_transport_batch_processing_submits_receipt() -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout: float):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"status": "running", "sessionId": "session-process"})

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        created = create_cai_owned_transport_session(
            session_id="caiot_process",
            instance_id="instance-process",
            participant_node_ids=["node-a", "node-b"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-1",
            policy=policy,
        )
        record_cai_owned_transport_batch(
            created.session_id,
            batch_id="caibatch_process",
            phase="prefill_activation_batches",
            source_node_id="node-a",
            sink_node_id="node-b",
            payload_size_bytes=64,
            payload_sha256_hex="12" * 32,
            metadata={"layerStart": 14, "layerEnd": 28},
            status="received",
            policy=policy,
        )
        result = complete_cai_owned_transport_batch_processing(
            created.session_id,
            "caibatch_process",
            node_id="node-b",
            coordinator_cai_url="http://coordinator:52415/",
            metrics={"latencyMs": 12, "inputTokens": 4, "outputTokens": 2},
            output_payload=b"worker-output",
            route_audit={
                "selectedRoute": "direct",
                "directUrl": "http://node-b:52415",
                "attemptCount": 1,
                "latencyMs": 7.5,
                "fallbackCount": 0,
            },
            runtime_audit={
                "runtimeId": "runtime-process",
                "runtimeVersion": "test-runtime/1",
                "adapterId": "test-adapter",
                "adapterVersion": "test-adapter/1",
            },
            timeout_sec=2.5,
            policy=policy,
        )
        output_payload_path = cai_owned_transport_batch_output_payload_path(
            created.session_id,
            "caibatch_process",
            policy,
        )
        stored_output_payload = read_cai_owned_transport_batch_output_payload(
            created.session_id,
            "caibatch_process",
            policy,
        )
        output_payload_exists = output_payload_path.exists()
        processed_batch = list_cai_owned_transport_sessions(policy)[0].batch_records[0]
        expected_hash_chain = build_cai_owned_transport_batch_hash_chain(
            session_id=created.session_id,
            batch_id="caibatch_process",
            input_payload_sha256_hex="12" * 32,
            output_payload_sha256_hex=result["outputPayload"][
                "outputPayloadSha256Hex"
            ],
            sequence=0,
        )
        output_envelope = build_cai_owned_transport_output_batch_envelope(
            session_id=created.session_id,
            source_batch_id="caibatch_process",
            sink_node_id="node-a",
            metadata={"nextStage": "coordinator"},
            policy=policy,
        )

    assert result["receipt"]["nodeId"] == "node-b"
    assert result["receipt"]["chainId"] == "mainnet"
    assert result["receipt"]["activationBatchCount"] == 1
    assert result["receipt"]["decodeBatchCount"] == 0
    assert result["outputPayload"]["outputPayloadSizeBytes"] == len(b"worker-output")
    assert (
        result["outputPayload"]["outputPayloadStorageKey"]
        == "caiot_process/caibatch_process.out.bin"
    )
    assert output_payload_exists is True
    assert stored_output_payload == b"worker-output"
    assert (
        processed_batch["outputPayloadSha256Hex"]
        == result["outputPayload"]["outputPayloadSha256Hex"]
    )
    assert processed_batch["outputPayloadSizeBytes"] == len(b"worker-output")
    assert (
        processed_batch["outputPayloadStorageKey"]
        == "caiot_process/caibatch_process.out.bin"
    )
    assert processed_batch["inputPayloadSha256Hex"] == "12" * 32
    assert processed_batch["hashChainSha256Hex"] == (
        expected_hash_chain["hashChainSha256Hex"]
    )
    assert processed_batch["routeAudit"]["selectedRoute"] == "direct"
    assert processed_batch["runtimeAudit"]["adapterId"] == "test-adapter"
    assert result["receipt"]["inputPayloadSha256Hexes"] == ["12" * 32]
    assert result["receipt"]["outputPayloadSha256Hexes"] == [
        result["outputPayload"]["outputPayloadSha256Hex"]
    ]
    assert result["receipt"]["hashChainSha256Hexes"] == [
        expected_hash_chain["hashChainSha256Hex"]
    ]
    assert result["receipt"]["metrics"]["outputPayloadSizeBytes"] == len(
        b"worker-output"
    )
    assert result["receipt"]["metrics"]["inputTokenCount"] == 4
    assert result["receipt"]["metrics"]["outputTokenCount"] == 2
    assert result["receipt"]["metrics"]["promptTokenCount"] == 4
    assert result["receipt"]["metrics"]["completionTokenCount"] == 0
    assert result["receipt"]["metrics"]["adapterIds"] == ["test-adapter"]
    assert result["receipt"]["metrics"]["runtimeVersions"] == ["test-runtime/1"]
    assert output_envelope["sourceNodeId"] == "node-b"
    assert output_envelope["sinkNodeId"] == "node-a"
    assert output_envelope["chainId"] == "mainnet"
    assert output_envelope["payloadSizeBytes"] == len(b"worker-output")
    assert output_envelope["payloadSha256Hex"] == (
        result["outputPayload"]["outputPayloadSha256Hex"]
    )
    assert output_envelope["metadata"]["payloadRole"] == "shard_output"
    assert output_envelope["metadata"]["previousBatchId"] == "caibatch_process"
    assert output_envelope["metadata"]["hashChainSha256Hex"] == (
        expected_hash_chain["hashChainSha256Hex"]
    )
    assert output_envelope["metadata"]["nextStage"] == "coordinator"
    assert result["coordinatorResponse"]["status"] == "running"
    assert captured["url"] == (
        "http://coordinator:52415/v1/cai/transport/sessions/"
        "caiot_process/shard-receipts"
    )
    assert captured["timeout"] == 2.5
    assert captured["body"]["nodeId"] == "node-b"
    assert captured["body"]["chainId"] == "mainnet"
    assert captured["body"]["activationBatchCount"] == 1
    assert captured["body"]["metrics"]["processedBatchCount"] == 1


def test_complete_cai_owned_transport_work_item_requires_runtime_owner() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        created = create_cai_owned_transport_session(
            session_id="caiot_workitem",
            instance_id="instance-workitem",
            participant_node_ids=["node-a", "node-b"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-workitem",
            policy=policy,
        )
        record_cai_owned_transport_batch(
            created.session_id,
            batch_id="caibatch_workitem",
            phase="decode_activation_batches",
            source_node_id="node-a",
            sink_node_id="node-b",
            payload_size_bytes=32,
            payload_sha256_hex="34" * 32,
            metadata={"layerStart": 14, "layerEnd": 28},
            status="received",
            policy=policy,
        )
        claim_cai_owned_transport_batch(
            created.session_id,
            "caibatch_workitem",
            node_id="node-b",
            runtime_id="runtime-owner",
            lease_seconds=30,
            policy=policy,
        )
        wrong_runtime_error = None
        try:
            complete_cai_owned_transport_work_item(
                created.session_id,
                "caibatch_workitem",
                node_id="node-b",
                runtime_id="runtime-other",
                output_payload=b"wrong",
                policy=policy,
            )
        except ValueError as exc:
            wrong_runtime_error = str(exc)
        result = complete_cai_owned_transport_work_item(
            created.session_id,
            "caibatch_workitem",
            node_id="node-b",
            runtime_id="runtime-owner",
            output_payload=b"decode-output",
            metrics={"tokens": 2},
            policy=policy,
        )
        stored_output_payload = read_cai_owned_transport_batch_output_payload(
            created.session_id,
            "caibatch_workitem",
            policy,
        )
        processed_batch = list_cai_owned_transport_sessions(policy)[0].batch_records[0]

    assert (
        wrong_runtime_error
        == "CAI-owned transport batch runtime id does not match."
    )
    assert result["receipt"]["nodeId"] == "node-b"
    assert result["receipt"]["chainId"] == "mainnet"
    assert result["receipt"]["decodeBatchCount"] == 1
    assert result["outputPayload"]["outputPayloadSizeBytes"] == len(b"decode-output")
    assert stored_output_payload == b"decode-output"
    assert processed_batch["status"] == "processed"
    assert processed_batch["metrics"] == {"tokens": 2}
    assert processed_batch["outputPayloadStorageKey"] == (
        "caiot_workitem/caibatch_workitem.out.bin"
    )


def test_cai_owned_transport_session_completion_rejects_failed_batch() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        created = create_cai_owned_transport_session(
            session_id="caiot_failed_audit",
            instance_id="instance-failed-audit",
            participant_node_ids=["node-a", "node-b"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            policy=policy,
        )
        record_cai_owned_transport_batch(
            created.session_id,
            batch_id="caibatch_failed_audit",
            phase="decode_activation_batches",
            source_node_id="node-a",
            sink_node_id="node-b",
            payload_size_bytes=16,
            payload_sha256_hex="90" * 32,
            status="failed",
            policy=policy,
        )
        completion_error = None
        try:
            complete_cai_owned_transport_session(created.session_id, policy=policy)
        except ValueError as exc:
            completion_error = str(exc)
        record = list_cai_owned_transport_sessions(policy)[0]

    assert completion_error == (
        "CAI-owned transport batch 'caibatch_failed_audit' is failed."
    )
    assert record.status == "failed"
    assert record.last_error == completion_error


def test_cai_owned_transport_session_completion_rejects_missing_output() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        created = create_cai_owned_transport_session(
            session_id="caiot_missing_output",
            instance_id="instance-missing-output",
            participant_node_ids=["node-a", "node-b"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            policy=policy,
        )
        record_cai_owned_transport_batch(
            created.session_id,
            batch_id="caibatch_missing_output",
            phase="decode_activation_batches",
            source_node_id="node-a",
            sink_node_id="node-b",
            payload_size_bytes=16,
            payload_sha256_hex="92" * 32,
            status="processed",
            policy=policy,
        )
        completion_error = None
        try:
            complete_cai_owned_transport_session(created.session_id, policy=policy)
        except ValueError as exc:
            completion_error = str(exc)
        record = list_cai_owned_transport_sessions(policy)[0]

    assert completion_error == (
        "CAI-owned transport batch 'caibatch_missing_output' is missing output hash."
    )
    assert record.status == "failed"
    assert record.last_error == completion_error


def test_cai_owned_transport_session_completion_rejects_bad_hash_chain() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        created = create_cai_owned_transport_session(
            session_id="caiot_bad_chain",
            instance_id="instance-bad-chain",
            participant_node_ids=["node-a", "node-b"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            policy=policy,
        )
        record_cai_owned_transport_batch(
            created.session_id,
            batch_id="caibatch_bad_chain",
            phase="decode_activation_batches",
            source_node_id="node-a",
            sink_node_id="node-b",
            payload_size_bytes=16,
            payload_sha256_hex="91" * 32,
            status="received",
            policy=policy,
        )
        complete_cai_owned_transport_batch_processing(
            created.session_id,
            "caibatch_bad_chain",
            node_id="node-b",
            output_payload=b"bad-chain-output",
            policy=policy,
        )
        mark_cai_owned_transport_batch_status(
            created.session_id,
            "caibatch_bad_chain",
            status="processed",
            node_id="node-b",
            hash_chain_sha256_hex="00" * 32,
            policy=policy,
        )
        completion_error = None
        try:
            complete_cai_owned_transport_session(created.session_id, policy=policy)
        except ValueError as exc:
            completion_error = str(exc)
        record = list_cai_owned_transport_sessions(policy)[0]

    assert completion_error == (
        "CAI-owned transport batch 'caibatch_bad_chain' hash chain does not match."
    )
    assert record.status == "failed"
    assert record.last_error == completion_error


def test_fail_cai_owned_transport_work_item_retries_then_final_fails() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        created = create_cai_owned_transport_session(
            session_id="caiot_retry",
            instance_id="instance-retry",
            participant_node_ids=["node-a", "node-b"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-retry",
            policy=policy,
        )
        record_cai_owned_transport_batch(
            created.session_id,
            batch_id="caibatch_retry",
            phase="decode_activation_batches",
            source_node_id="node-a",
            sink_node_id="node-b",
            payload_size_bytes=32,
            payload_sha256_hex="56" * 32,
            status="received",
            policy=policy,
        )
        claim_cai_owned_transport_batch(
            created.session_id,
            "caibatch_retry",
            node_id="node-b",
            runtime_id="runtime-1",
            policy=policy,
        )
        retry_result = fail_cai_owned_transport_work_item(
            created.session_id,
            "caibatch_retry",
            node_id="node-b",
            runtime_id="runtime-1",
            error="transient adapter error",
            retryable=True,
            max_attempts=2,
            metrics={"latencyMs": 4},
            policy=policy,
        )
        retry_batch = list_cai_owned_transport_sessions(policy)[0].batch_records[0]
        reclaimed = claim_next_cai_owned_transport_batch(
            "node-b",
            runtime_id="runtime-2",
            policy=policy,
        )
        final_result = fail_cai_owned_transport_work_item(
            created.session_id,
            "caibatch_retry",
            node_id="node-b",
            runtime_id="runtime-2",
            error="adapter failed again",
            retryable=True,
            max_attempts=2,
            policy=policy,
        )
        final_batch = list_cai_owned_transport_sessions(policy)[0].batch_records[0]

    assert retry_result["status"] == "received"
    assert retry_result["retryScheduled"] is True
    assert retry_result["attemptCount"] == 1
    assert retry_batch["status"] == "received"
    assert retry_batch["previousRuntimeId"] == "runtime-1"
    assert "runtimeId" not in retry_batch
    assert retry_batch["lastError"] == "transient adapter error"
    assert retry_batch["failureCount"] == 1
    assert retry_batch["metrics"] == {"latencyMs": 4}
    assert reclaimed is not None
    assert reclaimed["batch"]["runtimeId"] == "runtime-2"
    assert reclaimed["batch"]["attemptCount"] == 2
    assert final_result["status"] == "failed"
    assert final_result["retryScheduled"] is False
    assert final_result["attemptCount"] == 2
    assert final_batch["status"] == "failed"
    assert final_batch["lastError"] == "adapter failed again"
    assert final_batch["failureCount"] == 2
    assert final_batch["failedAt"]


def test_cai_owned_transport_session_lifecycle_persists_proof() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-owned-transport")
        created = create_cai_owned_transport_session(
            instance_id="instance-1",
            participant_node_ids=["node-a", "node-b"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-1",
            policy=policy,
        )
        record_cai_owned_transport_batch(
            created.session_id,
            batch_id="caibatch_lifecycle_a",
            phase="prefill_activation_batches",
            source_node_id="node-a",
            sink_node_id="node-a",
            payload_size_bytes=128,
            payload_sha256_hex="ab" * 32,
            metadata={"layerStart": 0, "layerEnd": 14},
            status="received",
            policy=policy,
        )
        complete_cai_owned_transport_batch_processing(
            created.session_id,
            "caibatch_lifecycle_a",
            node_id="node-a",
            output_payload=b"node-a-output",
            policy=policy,
        )
        record_cai_owned_transport_batch(
            created.session_id,
            batch_id="caibatch_lifecycle_b",
            phase="prefill_activation_batches",
            source_node_id="node-a",
            sink_node_id="node-b",
            payload_size_bytes=128,
            payload_sha256_hex="bc" * 32,
            metadata={"layerStart": 14, "layerEnd": 28},
            status="received",
            policy=policy,
        )
        complete_cai_owned_transport_batch_processing(
            created.session_id,
            "caibatch_lifecycle_b",
            node_id="node-b",
            output_payload=b"node-b-output",
            policy=policy,
        )
        completed = complete_cai_owned_transport_session(
            created.session_id,
            policy=policy,
        )
        records = list_cai_owned_transport_sessions(policy)
        payload = cai_owned_transport_session_to_dict(records[0])

    assert completed.status == "completed"
    assert completed.proof is not None
    assert records[0].session_id == created.session_id
    assert payload["status"] == "completed"
    assert payload["chainId"] == "mainnet"
    assert payload["network"] == "mainnet"
    assert len(payload["batchRecords"]) == 2
    assert len(payload["shardReceipts"]) == 0
    assert payload["proof"]["sessionId"] == created.session_id
    assert payload["proof"]["activationBatchCount"] == 1
    assert payload["proof"]["executionAudit"]["verified"] is True
    assert payload["proof"]["executionAudit"]["processedBatchCount"] == 2


def test_submit_cai_owned_transport_shard_receipt_posts_to_coordinator() -> None:
    public_key_b64, signing_seed_b64 = _signing_material()
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout: float):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"status": "running", "sessionId": "session-1"})

    with patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        response = submit_cai_owned_transport_shard_receipt(
            "http://coordinator:52415/",
            "session-1",
            node_id="node-b",
            activation_batch_count=2,
            decode_batch_count=1,
            signing_material={
                "public_key_b64": public_key_b64,
                "signing_seed_b64": signing_seed_b64,
            },
            timeout_sec=3.5,
        )

    assert response["status"] == "running"
    assert captured["url"] == (
        "http://coordinator:52415/v1/cai/transport/sessions/"
        "session-1/shard-receipts"
    )
    assert captured["timeout"] == 3.5
    assert captured["body"]["nodeId"] == "node-b"
    assert captured["body"]["chainId"] == "mainnet"
    assert captured["body"]["network"] == "mainnet"
    assert captured["body"]["signerNodeId"] == "node-b"
    assert captured["body"]["signature"]["public_key_b64"] == public_key_b64


def test_submit_cai_owned_transport_offer_can_use_overlay_relay_url() -> None:
    captured: dict[str, object] = {}
    offer = build_cai_owned_transport_session_offer(
        instance_id="instance-overlay",
        participant_node_ids=["node-a", "node-b"],
        executor_node_ids=["node-b"],
        model_id="cai-network/Qwen3-0.6B-GGUF",
        task_id="task-overlay",
        source_node_id="node-a",
    )

    def fake_urlopen(request, timeout: float):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"status": "queued"})

    with patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        response = submit_cai_owned_transport_session_offer(
            "cai-overlay:http://relay:52415?targetNodeId=node-b",
            offer,
            timeout_sec=4.0,
        )

    assert response["status"] == "queued"
    assert response["selectedRoute"] == "cai_overlay_gossipsub"
    assert captured["url"] == "http://relay:52415/v1/cai/transport/overlay/send"
    assert captured["timeout"] == 4.0
    assert captured["body"]["kind"] == "session_offer"
    assert captured["body"]["sourceNodeId"] == "node-a"
    assert captured["body"]["targetNodeId"] == "node-b"
    assert captured["body"]["sessionId"] == offer["sessionId"]
    assert captured["body"]["payload"]["sessionId"] == offer["sessionId"]

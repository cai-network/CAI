# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.cai_llama_cpp_assignment_artifact_engine import (  # noqa: E402
    ASSIGNMENT_EXECUTOR_REQUEST_ABI,
)
from cai_compute_chain.cai_llama_cpp_assignment_slot_state_executor import (  # noqa: E402
    ASSIGNMENT_SLOT_STATE_EXECUTOR_ID,
    handle_assignment_slot_state_executor_request,
)
from cai_compute_chain.cai_llama_cpp_slot_state_engine import (  # noqa: E402
    SlotStateEngineConfig,
)


def _request(tmp_path: Path) -> dict:
    payload = b"hello slot executor"
    payload_path = (tmp_path / "inputs" / "input.bin").resolve()
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes(payload)
    output_path = (tmp_path / "outputs" / "output.bin").resolve()
    assignment_path = (tmp_path / "assignment.gguf").resolve()
    assignment_path.write_bytes(b"assignment")
    model_path = (tmp_path / "model.gguf").resolve()
    model_path.write_bytes(b"model")
    return {
        "schemaVersion": 1,
        "abi": ASSIGNMENT_EXECUTOR_REQUEST_ABI,
        "action": "process_decode",
        "sessionId": "session-slot-executor",
        "modelId": "cai-network/Qwen3-0.6B-GGUF",
        "layerStart": 1,
        "layerEnd": 2,
        "tokenStart": 0,
        "tokenEnd": 1,
        "requiresFinalOutput": True,
        "expectedOutputKind": "final_output",
        "inputPayloadFile": {
            "path": str(payload_path),
            "sizeBytes": len(payload),
            "sha256Hex": hashlib.sha256(payload).hexdigest(),
        },
        "expectedOutputPayloadPath": str(output_path),
        "assignmentArtifact": {
            "artifactId": "assignment-main",
            "source": "materialized_assignment",
            "localPath": str(assignment_path),
            "sizeBytes": int(assignment_path.stat().st_size),
        },
        "localArtifactResolution": {
            "modelId": "cai-network/Qwen3-0.6B-GGUF",
            "assignmentArtifact": {
                "artifactId": "assignment-main",
                "source": "materialized_assignment",
                "localPath": str(assignment_path),
                "sizeBytes": int(assignment_path.stat().st_size),
            },
            "modelArtifact": {
                "artifactId": "gguf-main",
                "source": "local_binding",
                "localPath": str(model_path),
                "expectedSizeBytes": int(model_path.stat().st_size),
            },
        },
        "frame": {
            "sessionId": "session-slot-executor",
            "batchId": "batch-slot-executor",
            "phase": "decode_activation_batches",
            "modelId": "cai-network/Qwen3-0.6B-GGUF",
            "frameKind": "decode",
            "layerStart": 1,
            "layerEnd": 2,
            "tokenStart": 0,
            "tokenEnd": 1,
            "payloadSha256Hex": hashlib.sha256(payload).hexdigest(),
            "metadata": {},
        },
        "shardSpec": {
            "modelId": "cai-network/Qwen3-0.6B-GGUF",
            "backend": "llama.cpp-patched",
            "requiresPatchedBackend": True,
            "frameKind": "decode",
            "phase": "decode_activation_batches",
            "layerStart": 1,
            "layerEnd": 2,
            "tokenStart": 0,
            "tokenEnd": 1,
        },
        "outputContract": {"schemaVersion": 1, "requiresFinalOutput": True},
        "productionRequirements": {"schemaVersion": 1, "handoffAbi": "test"},
        "managedRuntime": {
            "schemaVersion": 1,
            "abi": "cai-llama-cpp-managed-runtime-v1",
            "platform": "nt",
            "repoRoot": str(tmp_path.resolve()),
            "runtimeRoot": str((tmp_path / "runtime").resolve()),
            "modelId": "cai-network/Qwen3-0.6B-GGUF",
            "sessionPaths": {
                "root": str((tmp_path / "session-root").resolve()),
                "stateDir": str((tmp_path / "session-root" / "state").resolve()),
                "cacheDir": str((tmp_path / "session-root" / "cache").resolve()),
                "logsDir": str((tmp_path / "session-root" / "logs").resolve()),
                "stdoutLog": str((tmp_path / "session-root" / "logs" / "stdout.log").resolve()),
                "stderrLog": str((tmp_path / "session-root" / "logs" / "stderr.log").resolve()),
            },
        },
        "executionWorkspace": {
            "schemaVersion": 1,
            "abi": "cai-llama-cpp-execution-workspace-v1",
            "root": str(tmp_path.resolve()),
            "inputsDir": str((tmp_path / "inputs").resolve()),
            "outputsDir": str((tmp_path / "outputs").resolve()),
            "stateFilesDir": str((tmp_path / "state").resolve()),
            "manifestPath": str((tmp_path / "execution-workspace.json").resolve()),
        },
    }


def test_assignment_slot_state_executor_translates_request(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request = _request(tmp_path)
    captured: dict[str, object] = {}
    expected_output = b"slot-state-final-output"

    def fake_slot_state(request_payload, *, config):
        captured["request"] = dict(request_payload)
        captured["config"] = config
        response_output_path = Path(
            request_payload["localFileContract"]["responseOutputPath"]
        ).resolve()
        response_output_path.parent.mkdir(parents=True, exist_ok=True)
        response_output_path.write_bytes(expected_output)
        return {
            "status": "ok",
            "outputPayloadFile": {
                "path": str(response_output_path),
                "sizeBytes": len(expected_output),
                "sha256Hex": hashlib.sha256(expected_output).hexdigest(),
            },
            "outputPayloadSha256Hex": hashlib.sha256(expected_output).hexdigest(),
            "metrics": {"backendMode": "llama.cpp-slot-state"},
        }

    monkeypatch.setattr(
        "cai_compute_chain.cai_llama_cpp_assignment_slot_state_executor.handle_slot_state_engine_request",
        fake_slot_state,
    )

    response = handle_assignment_slot_state_executor_request(
        request,
        config=SlotStateEngineConfig(
            server_url="",
            state_dir=None,
            slot_id=0,
            timeout_sec=10,
            decode_tokens=3,
        ),
    )

    handoff_request = captured["request"]
    assert response["status"] == "ok"
    assert response["outputKind"] == "final_output"
    assert response["realModelExecution"] is True
    assert response["metrics"]["executorBackendMode"] == ASSIGNMENT_SLOT_STATE_EXECUTOR_ID
    assert response["metrics"]["slotStateMetrics"]["backendMode"] == "llama.cpp-slot-state"
    assert response["outputPayloadSha256Hex"] == hashlib.sha256(expected_output).hexdigest()
    assert Path(response["outputPayloadFile"]["path"]).read_bytes() == expected_output

    assert handoff_request["abi"] == "cai-llama-cpp-external-shard-v1"
    assert handoff_request["action"] == "process_decode"
    assert handoff_request["payloadFile"]["path"] == request["inputPayloadFile"]["path"]
    assert handoff_request["localFileContract"]["responseOutputPath"] == request["expectedOutputPayloadPath"]
    assert handoff_request["managedRuntime"]["modelId"] == "cai-network/Qwen3-0.6B-GGUF"
    assert handoff_request["localArtifactResolution"]["modelArtifact"]["localPath"]
    assert handoff_request["productionRequirements"]["handoffAbi"] == "test"


def test_assignment_slot_state_executor_supports_load_shard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request = _request(tmp_path)
    request["action"] = "load_shard"
    request.pop("inputPayloadFile", None)
    request.pop("expectedOutputPayloadPath", None)
    captured: dict[str, object] = {}

    def fake_slot_state(request_payload, *, config):
        captured["request"] = dict(request_payload)
        return {
            "status": "ready",
            "metrics": {"backendMode": "llama.cpp-slot-state", "slotId": 0},
        }

    monkeypatch.setattr(
        "cai_compute_chain.cai_llama_cpp_assignment_slot_state_executor.handle_slot_state_engine_request",
        fake_slot_state,
    )

    response = handle_assignment_slot_state_executor_request(
        request,
        config=SlotStateEngineConfig(
            server_url="",
            state_dir=None,
            slot_id=0,
            timeout_sec=10,
            decode_tokens=3,
        ),
    )

    handoff_request = captured["request"]
    assert response["status"] == "ready"
    assert response["realModelExecution"] is True
    assert response["metrics"]["executorBackendMode"] == ASSIGNMENT_SLOT_STATE_EXECUTOR_ID
    assert response["metrics"]["slotStateMetrics"]["slotId"] == 0
    assert handoff_request["action"] == "load_shard"
    assert "payloadFile" not in handoff_request
    assert "localFileContract" not in handoff_request


def test_assignment_slot_state_executor_rejects_shard_only_production(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request = _request(tmp_path)
    request["productionRequirements"] = {
        "schemaVersion": 1,
        "requiresShardOnlyLoading": True,
        "forbidFullModelFallback": True,
    }
    captured: dict[str, object] = {}

    def fake_slot_state(request_payload, *, config):
        captured["request"] = dict(request_payload)
        return {"status": "ok", "metrics": {"backendMode": "llama.cpp-slot-state"}}

    monkeypatch.setattr(
        "cai_compute_chain.cai_llama_cpp_assignment_slot_state_executor.handle_slot_state_engine_request",
        fake_slot_state,
    )

    try:
        handle_assignment_slot_state_executor_request(
            request,
            config=SlotStateEngineConfig(
                server_url="",
                state_dir=None,
                slot_id=0,
                timeout_sec=10,
                decode_tokens=3,
            ),
        )
    except ValueError as exc:
        assert "reference-only" in str(exc)
    else:
        raise AssertionError("expected production shard-only slot_state rejection")

    assert "request" not in captured

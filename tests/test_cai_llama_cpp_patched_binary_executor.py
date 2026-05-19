# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cai_compute_chain.cai_llama_cpp_assignment_artifact_engine import (
    ASSIGNMENT_EXECUTOR_REQUEST_ABI,
)
from cai_compute_chain.cai_llama_cpp_patched_binary_executor import (
    CAI_LLM_PATCHED_BINARY_TIMEOUT_ENV,
    CAI_LLM_SHARD_NATIVE_TIMEOUT_ENV,
    PATCHED_BINARY_EXECUTOR_ID,
    _patched_binary_timeout_seconds,
    handle_patched_binary_executor_request,
    reset_patched_binary_executor_runtime_state,
)
from cai_compute_chain.cai_llama_cpp_patched_executor_host import (
    handle_patched_executor_host_request,
    reset_patched_executor_host_clients,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "cai-network/Qwen3-0.6B-GGUF"


def test_patched_binary_timeout_follows_native_timeout_env(monkeypatch) -> None:
    monkeypatch.delenv(CAI_LLM_PATCHED_BINARY_TIMEOUT_ENV, raising=False)
    monkeypatch.setenv(CAI_LLM_SHARD_NATIVE_TIMEOUT_ENV, "900")

    assert _patched_binary_timeout_seconds() == 900.0


def test_patched_binary_timeout_dedicated_env_wins(monkeypatch) -> None:
    monkeypatch.setenv(CAI_LLM_PATCHED_BINARY_TIMEOUT_ENV, "240")
    monkeypatch.setenv(CAI_LLM_SHARD_NATIVE_TIMEOUT_ENV, "900")

    assert _patched_binary_timeout_seconds() == 240.0


def _fake_binary_script(tmp_path: Path) -> Path:
    script = tmp_path / "fake_patched_binary.py"
    script.write_text(
        """
import hashlib
import json
import sys
from pathlib import Path

request = json.loads(sys.stdin.read() or "{}")
action = str(request.get("action") or "")
io_targets = request.get("ioTargets") or {}
response = {
    "status": "ready" if action == "load_shard" else "ok",
    "realModelExecution": True,
    "metrics": {"binary": "fake-patched-binary", "action": action},
}
if action == "process_prefill":
    state_payload = b"patched-layer-range-state"
    state_path = Path(io_targets["outputStateFilePath"])
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_bytes(state_payload)
    manifest = {
        "schemaVersion": 1,
        "abi": "cai-llama-cpp-real-state-payload-v1",
        "stateKind": request["expectedOutputKind"],
        "producedByAction": action,
        "modelId": request["modelId"],
        "sessionId": request["sessionId"],
        "layerStart": request["layerStart"],
        "layerEnd": request["layerEnd"],
        "tokenStart": request["tokenStart"],
        "tokenEnd": request["tokenEnd"],
        "stateFormat": "ggml-kv-cache-v1/token-step-kv-cache-v1",
        "stateFile": {
            "path": str(state_path.resolve()),
            "sha256Hex": hashlib.sha256(state_payload).hexdigest(),
            "sizeBytes": len(state_payload),
        },
        "metadata": {"binary": "fake-patched-binary"},
    }
    manifest_payload = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest_path = Path(io_targets["outputStateManifestPath"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(manifest_payload)
    response.update(
        {
            "outputKind": request["expectedOutputKind"],
            "outputPayloadFile": {
                "path": str(manifest_path.resolve()),
                "sizeBytes": len(manifest_payload),
                "sha256Hex": hashlib.sha256(manifest_payload).hexdigest(),
            },
            "outputPayloadSha256Hex": hashlib.sha256(manifest_payload).hexdigest(),
        }
    )
elif action == "process_decode":
    state = request.get("inputState") or {}
    assert state.get("stateFile", {}).get("path")
    final_payload = b"Paris"
    output_path = Path(io_targets["outputPayloadPath"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(final_payload)
    response.update(
        {
            "outputKind": request["expectedOutputKind"],
            "outputPayloadFile": {
                "path": str(output_path.resolve()),
                "sizeBytes": len(final_payload),
                "sha256Hex": hashlib.sha256(final_payload).hexdigest(),
            },
            "outputPayloadSha256Hex": hashlib.sha256(final_payload).hexdigest(),
        }
    )
print(json.dumps(response, sort_keys=True))
""".strip(),
        encoding="utf-8",
    )
    return script


def _fake_persistent_binary_script(tmp_path: Path) -> Path:
    script = tmp_path / "fake_patched_binary_persistent.py"
    script.write_text(
        """
import hashlib
import json
import os
import sys
from pathlib import Path

call_count = 0
loaded_sessions = {}
for line in sys.stdin:
    request = json.loads(line or "{}")
    action = str(request.get("action") or "")
    session_id = str(request.get("sessionId") or "")
    io_targets = request.get("ioTargets") or {}
    if action == "load_shard":
        loaded_sessions[session_id] = {
            "assignment": (request.get("assignmentArtifact") or {}).get("localPath")
        }
    response = {
        "status": "ready" if action == "load_shard" else "ok",
        "realModelExecution": True,
        "metrics": {
            "binary": "fake-patched-binary-persistent",
            "pid": os.getpid(),
            "callCount": call_count + 1,
            "loadedSessions": sorted(loaded_sessions),
        },
    }
    if action == "process_prefill":
        assert session_id in loaded_sessions
        state_payload = f"persistent-state-{call_count + 1}".encode("utf-8")
        state_path = Path(io_targets["outputStateFilePath"])
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_bytes(state_payload)
        manifest = {
            "schemaVersion": 1,
            "abi": "cai-llama-cpp-real-state-payload-v1",
            "stateKind": request["expectedOutputKind"],
            "producedByAction": action,
            "modelId": request["modelId"],
            "sessionId": request["sessionId"],
            "layerStart": request["layerStart"],
            "layerEnd": request["layerEnd"],
            "tokenStart": request["tokenStart"],
            "tokenEnd": request["tokenEnd"],
            "stateFormat": "ggml-kv-cache-v1/token-step-kv-cache-v1",
            "stateFile": {
                "path": str(state_path.resolve()),
                "sha256Hex": hashlib.sha256(state_payload).hexdigest(),
                "sizeBytes": len(state_payload),
            },
            "metadata": {"binary": "fake-patched-binary-persistent"},
        }
        manifest_payload = json.dumps(manifest, sort_keys=True).encode("utf-8")
        manifest_path = Path(io_targets["outputStateManifestPath"])
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(manifest_payload)
        response.update(
            {
                "outputKind": request["expectedOutputKind"],
                "outputPayloadFile": {
                    "path": str(manifest_path.resolve()),
                    "sizeBytes": len(manifest_payload),
                    "sha256Hex": hashlib.sha256(manifest_payload).hexdigest(),
                },
                "outputPayloadSha256Hex": hashlib.sha256(manifest_payload).hexdigest(),
            }
        )
    elif action == "process_decode":
        assert session_id in loaded_sessions
        state = request.get("inputState") or {}
        assert state.get("stateFile", {}).get("path")
        final_payload = b"Paris"
        output_path = Path(io_targets["outputPayloadPath"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(final_payload)
        response.update(
            {
                "outputKind": request["expectedOutputKind"],
                "outputPayloadFile": {
                    "path": str(output_path.resolve()),
                    "sizeBytes": len(final_payload),
                    "sha256Hex": hashlib.sha256(final_payload).hexdigest(),
                },
                "outputPayloadSha256Hex": hashlib.sha256(final_payload).hexdigest(),
            }
        )
    elif action == "finalize":
        loaded_sessions.pop(session_id, None)
        response["metrics"]["loadedSessions"] = sorted(loaded_sessions)
    call_count += 1
    print(json.dumps(response, sort_keys=True), flush=True)
""".strip(),
        encoding="utf-8",
    )
    return script


def _request(
    tmp_path: Path,
    *,
    action: str,
    expected_output_kind: str,
    input_payload: bytes,
    validated_input_state: dict | None = None,
) -> dict:
    workspace_root = tmp_path / "workspace"
    inputs_dir = workspace_root / "inputs"
    outputs_dir = workspace_root / "outputs"
    state_dir = workspace_root / "state"
    runtime_session = tmp_path / "runtime" / "session"
    for path in (
        inputs_dir,
        outputs_dir,
        state_dir,
        runtime_session / "state",
        runtime_session / "cache",
        runtime_session / "logs",
    ):
        path.mkdir(parents=True, exist_ok=True)
    input_path = inputs_dir / f"{action}-input.bin"
    input_path.write_bytes(input_payload)
    output_path = outputs_dir / f"{action}-output.bin"
    state_output_path = state_dir / f"{action}-output.state.bin"
    assignment_path = tmp_path / "assignment.gguf"
    assignment_path.write_bytes(b"assignment-bytes")
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"model-bytes")
    assignment_payload = assignment_path.read_bytes()
    base_request = {
        "schemaVersion": 1,
        "abi": ASSIGNMENT_EXECUTOR_REQUEST_ABI,
        "action": action,
        "sessionId": "session-patched-binary",
        "modelId": MODEL_ID,
        "layerStart": 0,
        "layerEnd": 14,
        "tokenStart": 0,
        "tokenEnd": 4,
        "requiresFinalOutput": expected_output_kind == "final_output",
        "expectedOutputKind": expected_output_kind,
        "inputPayloadFile": {
            "path": str(input_path.resolve()),
            "sizeBytes": len(input_payload),
            "sha256Hex": hashlib.sha256(input_payload).hexdigest(),
        },
        "expectedOutputPayloadPath": str(output_path.resolve()),
        "executionWorkspace": {
            "schemaVersion": 1,
            "abi": "cai-llama-cpp-execution-workspace-v1",
            "root": str(workspace_root.resolve()),
            "inputsDir": str(inputs_dir.resolve()),
            "outputsDir": str(outputs_dir.resolve()),
            "stateFilesDir": str(state_dir.resolve()),
            "manifestPath": str((workspace_root / "execution-workspace.json").resolve()),
        },
        "assignmentArtifact": {
            "artifactId": "assignment-main",
            "source": "materialized_assignment",
            "localPath": str(assignment_path.resolve()),
            "sizeBytes": int(assignment_path.stat().st_size),
            "layerStart": 0,
            "layerEnd": 14,
            "chunkRanges": [
                {
                    "chunkId": "chunk-0",
                    "offsetBytes": 0,
                    "sizeBytes": len(assignment_payload),
                    "sha256Hex": hashlib.sha256(assignment_payload).hexdigest(),
                    "layerStart": 0,
                    "layerEnd": 14,
                    "tensorNames": ["blk.0.attn_q.weight"],
                }
            ],
            "coverage": {
                "abi": "cai-llama-cpp-assignment-coverage-v1",
                "materializationMode": "sparse_full_size",
                "artifactSizeBytes": int(assignment_path.stat().st_size),
                "coveredByteCount": int(assignment_path.stat().st_size),
                "coveredRangeCount": 1,
                "zeroFilledOutsideCoveredRanges": True,
            },
        },
        "localArtifactResolution": {
            "assignmentArtifact": {
                "artifactId": "assignment-main",
                "source": "materialized_assignment",
                "localPath": str(assignment_path.resolve()),
                "sizeBytes": int(assignment_path.stat().st_size),
                "layerStart": 0,
                "layerEnd": 14,
                "chunkRanges": [
                    {
                        "chunkId": "chunk-0",
                        "offsetBytes": 0,
                        "sizeBytes": len(assignment_payload),
                        "sha256Hex": hashlib.sha256(assignment_payload).hexdigest(),
                        "layerStart": 0,
                        "layerEnd": 14,
                        "tensorNames": ["blk.0.attn_q.weight"],
                    }
                ],
                "coverage": {
                    "abi": "cai-llama-cpp-assignment-coverage-v1",
                    "materializationMode": "sparse_full_size",
                    "artifactSizeBytes": int(assignment_path.stat().st_size),
                    "coveredByteCount": int(assignment_path.stat().st_size),
                    "coveredRangeCount": 1,
                    "zeroFilledOutsideCoveredRanges": True,
                },
            },
            "modelArtifact": {
                "artifactId": "gguf-main",
                "source": "local_binding",
                "localPath": str(model_path.resolve()),
                "expectedSizeBytes": int(model_path.stat().st_size),
            },
        },
        "managedRuntime": {
            "abi": "cai-llama-cpp-managed-runtime-v1",
            "platform": os.name,
            "repoRoot": str(REPO_ROOT.resolve()),
            "runtimeRoot": str((tmp_path / "runtime").resolve()),
            "modelId": MODEL_ID,
            "sessionPaths": {
                "root": str(runtime_session.resolve()),
                "stateDir": str((runtime_session / "state").resolve()),
                "cacheDir": str((runtime_session / "cache").resolve()),
                "logsDir": str((runtime_session / "logs").resolve()),
                "stdoutLog": str((runtime_session / "logs" / "stdout.log").resolve()),
                "stderrLog": str((runtime_session / "logs" / "stderr.log").resolve()),
            },
        },
        "frame": {
            "sessionId": "session-patched-binary",
            "batchId": f"batch-{action}",
            "phase": (
                "prefill_activation_batches"
                if action == "process_prefill"
                else "decode_activation_batches"
            ),
            "modelId": MODEL_ID,
            "frameKind": "activation" if action == "process_prefill" else "decode",
            "layerStart": 0,
            "layerEnd": 14,
            "tokenStart": 0,
            "tokenEnd": 4,
            "payloadSha256Hex": hashlib.sha256(input_payload).hexdigest(),
            "metadata": {},
        },
        "shardSpec": {
            "modelId": MODEL_ID,
            "backend": "llama.cpp-patched",
            "requiresPatchedBackend": True,
            "frameKind": "activation" if action == "process_prefill" else "decode",
            "phase": (
                "prefill_activation_batches"
                if action == "process_prefill"
                else "decode_activation_batches"
            ),
            "layerStart": 0,
            "layerEnd": 14,
            "tokenStart": 0,
            "tokenEnd": 4,
        },
        "validatedExecutionContext": {
            "schemaVersion": 1,
            "abi": "cai-llama-cpp-patched-execution-context-v1",
            "action": action,
            "sessionId": "session-patched-binary",
            "modelId": MODEL_ID,
            "layerStart": 0,
            "layerEnd": 14,
            "tokenStart": 0,
            "tokenEnd": 4,
            "requiresFinalOutput": expected_output_kind == "final_output",
            "expectedOutputKind": expected_output_kind,
            "executionWorkspace": {
                "schemaVersion": 1,
                "abi": "cai-llama-cpp-execution-workspace-v1",
                "root": str(workspace_root.resolve()),
                "inputsDir": str(inputs_dir.resolve()),
                "outputsDir": str(outputs_dir.resolve()),
                "stateFilesDir": str(state_dir.resolve()),
                "manifestPath": str((workspace_root / "execution-workspace.json").resolve()),
            },
            "ioTargets": {
                "schemaVersion": 1,
                "abi": "cai-llama-cpp-patched-io-targets-v1",
                "inputPayloadPath": str(input_path.resolve()),
                "outputPayloadPath": str(output_path.resolve()),
                "outputStateManifestPath": (
                    str(output_path.resolve())
                    if expected_output_kind in {"activation_state", "decode_state"}
                    else None
                ),
                "outputStateFilePath": (
                    str(state_output_path.resolve())
                    if expected_output_kind in {"activation_state", "decode_state"}
                    else None
                ),
                "inputStateFilePath": (
                    str(((validated_input_state or {}).get("stateFile") or {}).get("path") or "")
                    or None
                ),
            },
            "assignmentArtifact": {
                "artifactId": "assignment-main",
                "source": "materialized_assignment",
                "localPath": str(assignment_path.resolve()),
                "sizeBytes": int(assignment_path.stat().st_size),
                "layerStart": 0,
                "layerEnd": 14,
                "chunkRanges": [
                    {
                        "chunkId": "chunk-0",
                        "offsetBytes": 0,
                        "sizeBytes": len(assignment_payload),
                        "sha256Hex": hashlib.sha256(assignment_payload).hexdigest(),
                        "layerStart": 0,
                        "layerEnd": 14,
                        "tensorNames": ["blk.0.attn_q.weight"],
                    }
                ],
                "coverage": {
                    "abi": "cai-llama-cpp-assignment-coverage-v1",
                    "materializationMode": "sparse_full_size",
                    "artifactSizeBytes": int(assignment_path.stat().st_size),
                    "coveredByteCount": int(assignment_path.stat().st_size),
                    "coveredRangeCount": 1,
                    "zeroFilledOutsideCoveredRanges": True,
                },
            },
            "modelArtifact": {
                "artifactId": "gguf-main",
                "source": "local_binding",
                "localPath": str(model_path.resolve()),
                "sizeBytes": int(model_path.stat().st_size),
                "expectedSizeBytes": int(model_path.stat().st_size),
            },
            "managedRuntime": {
                "abi": "cai-llama-cpp-managed-runtime-v1",
                "platform": os.name,
                "repoRoot": str(REPO_ROOT.resolve()),
                "runtimeRoot": str((tmp_path / "runtime").resolve()),
                "modelId": MODEL_ID,
                "sessionPaths": {
                    "root": str(runtime_session.resolve()),
                    "stateDir": str((runtime_session / "state").resolve()),
                    "cacheDir": str((runtime_session / "cache").resolve()),
                    "logsDir": str((runtime_session / "logs").resolve()),
                    "stdoutLog": str((runtime_session / "logs" / "stdout.log").resolve()),
                    "stderrLog": str((runtime_session / "logs" / "stderr.log").resolve()),
                },
            },
            "frame": {
                "sessionId": "session-patched-binary",
                "batchId": f"batch-{action}",
                "phase": (
                    "prefill_activation_batches"
                    if action == "process_prefill"
                    else "decode_activation_batches"
                ),
                "modelId": MODEL_ID,
                "frameKind": "activation" if action == "process_prefill" else "decode",
                "layerStart": 0,
                "layerEnd": 14,
                "tokenStart": 0,
                "tokenEnd": 4,
                "payloadSha256Hex": hashlib.sha256(input_payload).hexdigest(),
                "metadata": {},
            },
            "shardSpec": {
                "modelId": MODEL_ID,
                "backend": "llama.cpp-patched",
                "requiresPatchedBackend": True,
                "frameKind": "activation" if action == "process_prefill" else "decode",
                "phase": (
                    "prefill_activation_batches"
                    if action == "process_prefill"
                    else "decode_activation_batches"
                ),
                "layerStart": 0,
                "layerEnd": 14,
                "tokenStart": 0,
                "tokenEnd": 4,
            },
            "outputContract": {},
        },
        "productionRequirements": {"schemaVersion": 1, "handoffAbi": "test"},
    }
    if validated_input_state is not None:
        base_request["validatedInputState"] = dict(validated_input_state)
        base_request["validatedExecutionContext"]["inputState"] = dict(
            validated_input_state
        )
    if action in {"load_shard", "finalize"}:
        base_request.pop("inputPayloadFile", None)
        base_request.pop("expectedOutputPayloadPath", None)
    return base_request


def test_patched_binary_executor_runs_lifecycle_through_fake_binary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_patched_binary_executor_runtime_state()
    binary_script = _fake_binary_script(tmp_path)
    monkeypatch.setenv(
        "CAI_LLM_PATCHED_BINARY_COMMAND",
        subprocess.list2cmdline([sys.executable, str(binary_script)]),
    )

    load = handle_patched_binary_executor_request(
        _request(
            tmp_path,
            action="load_shard",
            expected_output_kind="decode_state",
            input_payload=b"",
        )
    )
    prefill = handle_patched_binary_executor_request(
        _request(
            tmp_path,
            action="process_prefill",
            expected_output_kind="decode_state",
            input_payload=b"The capital of France is",
        )
    )
    prefill_manifest = json.loads(
        Path(prefill["outputPayloadFile"]["path"]).read_text(encoding="utf-8")
    )
    decode = handle_patched_binary_executor_request(
        _request(
            tmp_path,
            action="process_decode",
            expected_output_kind="final_output",
            input_payload=Path(prefill["outputPayloadFile"]["path"]).read_bytes(),
            validated_input_state=prefill_manifest,
        )
    )
    finalize = handle_patched_binary_executor_request(
        _request(
            tmp_path,
            action="finalize",
            expected_output_kind="final_output",
            input_payload=b"",
        )
    )

    assert load["status"] == "ready"
    assert load["metrics"]["engineBackendMode"] == PATCHED_BINARY_EXECUTOR_ID
    assert prefill["outputKind"] == "decode_state"
    assert prefill_manifest["abi"] == "cai-llama-cpp-real-state-payload-v1"
    assert Path(prefill_manifest["stateFile"]["path"]).exists()
    assert decode["outputKind"] == "final_output"
    assert Path(decode["outputPayloadFile"]["path"]).read_bytes() == b"Paris"
    assert finalize["status"] == "ok"


def test_patched_executor_host_works_through_patched_binary_executor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_patched_executor_host_clients()
    reset_patched_binary_executor_runtime_state()
    binary_script = _fake_binary_script(tmp_path)
    monkeypatch.setenv(
        "CAI_LLM_SHARD_PATCHED_ENGINE_COMMAND",
        subprocess.list2cmdline(
            [
                sys.executable,
                "-m",
                "cai_compute_chain.cai_llama_cpp_patched_binary_executor",
            ]
        ),
    )
    monkeypatch.setenv(
        "CAI_LLM_PATCHED_BINARY_COMMAND",
        subprocess.list2cmdline([sys.executable, str(binary_script)]),
    )

    host_request = _request(
        tmp_path,
        action="process_prefill",
        expected_output_kind="decode_state",
        input_payload=b"The capital of France is",
    )
    response = handle_patched_executor_host_request(host_request)

    manifest = json.loads(
        Path(response["outputPayloadFile"]["path"]).read_text(encoding="utf-8")
    )
    assert response["status"] == "ok"
    assert response["metrics"]["patchedEngineMetrics"]["engineBackendMode"] == (
        PATCHED_BINARY_EXECUTOR_ID
    )
    assert manifest["abi"] == "cai-llama-cpp-real-state-payload-v1"


def test_patched_binary_executor_reuses_persistent_binary_through_lifecycle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_patched_binary_executor_runtime_state()
    binary_script = _fake_persistent_binary_script(tmp_path)
    monkeypatch.setenv(
        "CAI_LLM_PATCHED_BINARY_COMMAND",
        subprocess.list2cmdline([sys.executable, str(binary_script)]),
    )
    monkeypatch.setenv("CAI_LLM_PATCHED_BINARY_PERSISTENT", "1")

    load = handle_patched_binary_executor_request(
        _request(
            tmp_path,
            action="load_shard",
            expected_output_kind="decode_state",
            input_payload=b"",
        )
    )
    prefill = handle_patched_binary_executor_request(
        _request(
            tmp_path,
            action="process_prefill",
            expected_output_kind="decode_state",
            input_payload=b"The capital of France is",
        )
    )
    prefill_manifest = json.loads(
        Path(prefill["outputPayloadFile"]["path"]).read_text(encoding="utf-8")
    )
    decode = handle_patched_binary_executor_request(
        _request(
            tmp_path,
            action="process_decode",
            expected_output_kind="final_output",
            input_payload=Path(prefill["outputPayloadFile"]["path"]).read_bytes(),
            validated_input_state=prefill_manifest,
        )
    )
    finalize = handle_patched_binary_executor_request(
        _request(
            tmp_path,
            action="finalize",
            expected_output_kind="final_output",
            input_payload=b"",
        )
    )

    assert load["status"] == "ready"
    assert load["metrics"]["patchedBinaryMode"] == "persistent_jsonl"
    assert prefill["metrics"]["patchedBinaryMode"] == "persistent_jsonl"
    assert decode["metrics"]["patchedBinaryMode"] == "persistent_jsonl"
    assert finalize["metrics"]["patchedBinaryMode"] == "persistent_jsonl"
    assert load["metrics"]["patchedBinaryMetrics"]["callCount"] == 1
    assert prefill["metrics"]["patchedBinaryMetrics"]["callCount"] == 2
    assert decode["metrics"]["patchedBinaryMetrics"]["callCount"] == 3
    assert finalize["metrics"]["patchedBinaryMetrics"]["callCount"] == 4
    assert load["metrics"]["patchedBinaryMetrics"]["loadedSessions"] == [
        "session-patched-binary"
    ]
    assert finalize["metrics"]["patchedBinaryMetrics"]["loadedSessions"] == []
    assert (
        load["metrics"]["patchedBinaryMetrics"]["pid"]
        == prefill["metrics"]["patchedBinaryMetrics"]["pid"]
        == decode["metrics"]["patchedBinaryMetrics"]["pid"]
        == finalize["metrics"]["patchedBinaryMetrics"]["pid"]
    )


def test_patched_executor_host_reuses_persistent_patched_binary_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_patched_executor_host_clients()
    reset_patched_binary_executor_runtime_state()
    binary_script = _fake_persistent_binary_script(tmp_path)
    monkeypatch.setenv(
        "CAI_LLM_SHARD_PATCHED_ENGINE_COMMAND",
        subprocess.list2cmdline(
            [
                sys.executable,
                "-m",
                "cai_compute_chain.cai_llama_cpp_patched_binary_executor",
                "--jsonl",
            ]
        ),
    )
    monkeypatch.setenv("CAI_LLM_SHARD_PATCHED_ENGINE_PERSISTENT", "1")
    monkeypatch.setenv(
        "CAI_LLM_PATCHED_BINARY_COMMAND",
        subprocess.list2cmdline([sys.executable, str(binary_script)]),
    )
    monkeypatch.setenv("CAI_LLM_PATCHED_BINARY_PERSISTENT", "1")

    load = handle_patched_executor_host_request(
        _request(
            tmp_path,
            action="load_shard",
            expected_output_kind="decode_state",
            input_payload=b"",
        )
    )
    prefill = handle_patched_executor_host_request(
        _request(
            tmp_path,
            action="process_prefill",
            expected_output_kind="decode_state",
            input_payload=b"The capital of France is",
        )
    )
    decode = handle_patched_executor_host_request(
        _request(
            tmp_path,
            action="process_decode",
            expected_output_kind="final_output",
            input_payload=Path(prefill["outputPayloadFile"]["path"]).read_bytes(),
        )
    )
    finalize = handle_patched_executor_host_request(
        _request(
            tmp_path,
            action="finalize",
            expected_output_kind="final_output",
            input_payload=b"",
        )
    )

    assert load["status"] == "ready"
    assert prefill["status"] == "ok"
    assert decode["status"] == "ok"
    assert finalize["status"] == "ok"
    assert Path(decode["outputPayloadFile"]["path"]).read_bytes() == b"Paris"
    assert load["metrics"]["patchedEngineMode"] == "persistent_jsonl"
    assert prefill["metrics"]["patchedEngineMode"] == "persistent_jsonl"
    assert decode["metrics"]["patchedEngineMode"] == "persistent_jsonl"
    assert finalize["metrics"]["patchedEngineMode"] == "persistent_jsonl"
    assert load["metrics"]["patchedEngineMetrics"]["patchedBinaryMode"] == (
        "persistent_jsonl"
    )
    assert load["metrics"]["patchedEngineMetrics"]["engineProcessRequestCount"] == 1
    assert prefill["metrics"]["patchedEngineMetrics"]["engineProcessRequestCount"] == 2
    assert decode["metrics"]["patchedEngineMetrics"]["engineProcessRequestCount"] == 3
    assert finalize["metrics"]["patchedEngineMetrics"]["engineProcessRequestCount"] == 4
    assert (
        load["metrics"]["patchedEngineMetrics"]["engineProcessId"]
        == prefill["metrics"]["patchedEngineMetrics"]["engineProcessId"]
        == decode["metrics"]["patchedEngineMetrics"]["engineProcessId"]
        == finalize["metrics"]["patchedEngineMetrics"]["engineProcessId"]
    )
    assert (
        load["metrics"]["patchedEngineMetrics"]["patchedBinaryMetrics"]["pid"]
        == prefill["metrics"]["patchedEngineMetrics"]["patchedBinaryMetrics"]["pid"]
        == decode["metrics"]["patchedEngineMetrics"]["patchedBinaryMetrics"]["pid"]
        == finalize["metrics"]["patchedEngineMetrics"]["patchedBinaryMetrics"]["pid"]
    )
    assert (
        load["metrics"]["patchedEngineMetrics"]["patchedBinaryMetrics"]["callCount"]
        == 1
    )
    assert (
        prefill["metrics"]["patchedEngineMetrics"]["patchedBinaryMetrics"]["callCount"]
        == 2
    )
    assert (
        decode["metrics"]["patchedEngineMetrics"]["patchedBinaryMetrics"]["callCount"]
        == 3
    )
    assert (
        finalize["metrics"]["patchedEngineMetrics"]["patchedBinaryMetrics"]["callCount"]
        == 4
    )


def test_reference_patched_binary_restores_loaded_shard_context_from_disk(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_patched_binary_executor_runtime_state()
    monkeypatch.setenv(
        "CAI_LLM_PATCHED_BINARY_COMMAND",
        subprocess.list2cmdline(
            [
                sys.executable,
                "-m",
                "cai_compute_chain.cai_llama_cpp_reference_patched_binary",
            ]
        ),
    )

    load = handle_patched_binary_executor_request(
        _request(
            tmp_path,
            action="load_shard",
            expected_output_kind="decode_state",
            input_payload=b"",
        )
    )
    prefill = handle_patched_binary_executor_request(
        _request(
            tmp_path,
            action="process_prefill",
            expected_output_kind="decode_state",
            input_payload=b"The capital of France is",
        )
    )
    manifest = json.loads(
        Path(prefill["outputPayloadFile"]["path"]).read_text(encoding="utf-8")
    )
    prepared_shard_path = Path(
        load["metrics"]["patchedBinaryMetrics"]["referencePreparedShardWindowPath"]
    )
    prefill_receipt_path = Path(
        prefill["metrics"]["patchedBinaryMetrics"]["referenceBinaryPrefillReceiptPath"]
    )

    assert load["status"] == "ready"
    assert load["metrics"]["patchedBinaryMetrics"]["binary"] == "reference_patched_binary"
    assert load["metrics"]["patchedBinaryMetrics"]["referenceBinarySessionPrepared"] is True
    assert (
        load["metrics"]["patchedBinaryMetrics"]["referencePreparedShardWindowPrepared"]
        is True
    )
    assert (
        load["metrics"]["patchedBinaryMetrics"]["referenceAssignmentArtifactChunkCount"]
        == 1
    )
    assert (
        load["metrics"]["patchedBinaryMetrics"]["referenceAssignmentArtifactBytesRead"]
        > 0
    )
    assert prepared_shard_path.exists()
    assert (
        load["metrics"]["patchedBinaryMetrics"]["referencePreparedShardWindowSizeBytes"]
        > 0
    )
    assert prefill["status"] == "ok"
    assert (
        prefill["metrics"]["patchedBinaryMetrics"]["sessionContextSource"] == "disk"
    )
    assert (
        prefill["metrics"]["patchedBinaryMetrics"]["referencePreparedShardWindowVerified"]
        is True
    )
    assert prefill_receipt_path.exists()
    assert (
        prefill["metrics"]["patchedBinaryMetrics"]["referenceBinaryPrefillReceiptWritten"]
        is True
    )
    assert manifest["abi"] == "cai-llama-cpp-real-state-payload-v1"
    assert manifest["metadata"]["referenceBinary"] is True
    assert manifest["metadata"]["productionReady"] is False
    assert manifest["metadata"]["prefillExecutionDigest"]
    assert (
        manifest["metadata"]["prefillReceiptPath"] == str(prefill_receipt_path)
    )
    assert (
        manifest["metadata"]["preparedShardSha256Hex"]
        == load["metrics"]["patchedBinaryMetrics"]["referencePreparedShardWindowSha256Hex"]
    )
    finalize = handle_patched_binary_executor_request(
        _request(
            tmp_path,
            action="finalize",
            expected_output_kind="final_output",
            input_payload=b"",
        )
    )
    assert finalize["status"] == "ok"
    assert finalize["metrics"]["patchedBinaryMetrics"]["referencePreparedShardWindowDeleted"] is True
    assert finalize["metrics"]["patchedBinaryMetrics"]["referenceBinaryPrefillReceiptDeleted"] is True
    assert not prepared_shard_path.exists()
    assert not prefill_receipt_path.exists()


def test_reference_patched_binary_reuses_loaded_shard_context_in_memory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_patched_binary_executor_runtime_state()
    monkeypatch.setenv(
        "CAI_LLM_PATCHED_BINARY_COMMAND",
        subprocess.list2cmdline(
            [
                sys.executable,
                "-m",
                "cai_compute_chain.cai_llama_cpp_reference_patched_binary",
                "--jsonl",
            ]
        ),
    )
    monkeypatch.setenv("CAI_LLM_PATCHED_BINARY_PERSISTENT", "1")

    load = handle_patched_binary_executor_request(
        _request(
            tmp_path,
            action="load_shard",
            expected_output_kind="decode_state",
            input_payload=b"",
        )
    )
    prefill = handle_patched_binary_executor_request(
        _request(
            tmp_path,
            action="process_prefill",
            expected_output_kind="decode_state",
            input_payload=b"The capital of France is",
        )
    )
    decode = handle_patched_binary_executor_request(
        _request(
            tmp_path,
            action="process_decode",
            expected_output_kind="final_output",
            input_payload=Path(prefill["outputPayloadFile"]["path"]).read_bytes(),
            validated_input_state=json.loads(
                Path(prefill["outputPayloadFile"]["path"]).read_text(encoding="utf-8")
            ),
        )
    )

    assert load["status"] == "ready"
    assert load["metrics"]["patchedBinaryMode"] == "persistent_jsonl"
    assert prefill["metrics"]["patchedBinaryMetrics"]["sessionContextSource"] == "memory"
    assert decode["metrics"]["patchedBinaryMetrics"]["sessionContextSource"] == "memory"
    assert (
        decode["metrics"]["patchedBinaryMetrics"]["referenceBinaryPrefillExecutionDigest"]
        == prefill["metrics"]["patchedBinaryMetrics"]["referenceBinaryPrefillExecutionDigest"]
    )
    assert (
        load["metrics"]["patchedBinaryMetrics"]["callCount"]
        == 1
    )
    assert (
        prefill["metrics"]["patchedBinaryMetrics"]["callCount"]
        == 2
    )
    assert (
        decode["metrics"]["patchedBinaryMetrics"]["callCount"]
        == 3
    )
    assert (
        prefill["metrics"]["patchedBinaryMetrics"]["referencePreparedShardWindowVerified"]
        is True
    )
    assert Path(decode["outputPayloadFile"]["path"]).read_bytes().startswith(
        b"reference:"
    )


def test_reference_patched_binary_rejects_decode_without_matching_prefill_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_patched_binary_executor_runtime_state()
    monkeypatch.setenv(
        "CAI_LLM_PATCHED_BINARY_COMMAND",
        subprocess.list2cmdline(
            [
                sys.executable,
                "-m",
                "cai_compute_chain.cai_llama_cpp_reference_patched_binary",
                "--jsonl",
            ]
        ),
    )
    monkeypatch.setenv("CAI_LLM_PATCHED_BINARY_PERSISTENT", "1")

    handle_patched_binary_executor_request(
        _request(
            tmp_path,
            action="load_shard",
            expected_output_kind="decode_state",
            input_payload=b"",
        )
    )
    prefill = handle_patched_binary_executor_request(
        _request(
            tmp_path,
            action="process_prefill",
            expected_output_kind="decode_state",
            input_payload=b"The capital of France is",
        )
    )
    manifest = json.loads(
        Path(prefill["outputPayloadFile"]["path"]).read_text(encoding="utf-8")
    )
    manifest["metadata"]["prefillExecutionDigest"] = "broken-digest"

    with pytest.raises(
        ValueError,
        match="prefillExecutionDigest mismatch",
    ):
        handle_patched_binary_executor_request(
            _request(
                tmp_path,
                action="process_decode",
                expected_output_kind="final_output",
                input_payload=Path(prefill["outputPayloadFile"]["path"]).read_bytes(),
                validated_input_state=manifest,
            )
        )


def test_patched_binary_executor_rejects_reference_prefill_when_real_layer_required(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_patched_binary_executor_runtime_state()
    monkeypatch.setenv(
        "CAI_LLM_PATCHED_BINARY_COMMAND",
        subprocess.list2cmdline(
            [
                sys.executable,
                "-m",
                "cai_compute_chain.cai_llama_cpp_reference_patched_binary",
                "--jsonl",
            ]
        ),
    )
    monkeypatch.setenv("CAI_LLM_PATCHED_BINARY_PERSISTENT", "1")
    monkeypatch.setenv("CAI_LLM_PATCHED_BINARY_REQUIRE_REAL_LAYER_EXECUTION", "1")

    load = handle_patched_binary_executor_request(
        _request(
            tmp_path,
            action="load_shard",
            expected_output_kind="decode_state",
            input_payload=b"",
        )
    )

    assert load["status"] == "ready"
    with pytest.raises(
        ValueError,
        match="requires realLayerExecution=true",
    ):
        handle_patched_binary_executor_request(
            _request(
                tmp_path,
                action="process_prefill",
                expected_output_kind="decode_state",
                input_payload=b"The capital of France is",
            )
        )
    request_plan = json.loads(
        (tmp_path / "workspace" / "state" / "patched-binary-process_prefill.request.json")
        .read_text(encoding="utf-8")
    )
    assert request_plan["requireRealLayerExecution"] is True


def test_patched_binary_executor_rejects_full_model_when_shard_only_required(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_patched_binary_executor_runtime_state()
    binary_script = tmp_path / "fake_full_model_proving_binary.py"
    binary_script.write_text(
        """
import json
import sys

request = json.loads(sys.stdin.read() or "{}")
print(
    json.dumps(
        {
            "status": "ready",
            "realModelExecution": True,
            "metrics": {
                "binary": "fake-full-model-proving",
                "assignmentArtifactPresent": True,
                "usedFullModelForLayerRange": True,
                "shardOnlyLoadingReady": False,
            },
        },
        sort_keys=True,
    )
)
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "CAI_LLM_PATCHED_BINARY_COMMAND",
        subprocess.list2cmdline([sys.executable, str(binary_script)]),
    )
    monkeypatch.setenv("CAI_LLM_PATCHED_BINARY_REQUIRE_SHARD_ONLY_LOADING", "1")

    with pytest.raises(
        ValueError,
        match="requires shard-only loading",
    ):
        handle_patched_binary_executor_request(
            _request(
                tmp_path,
                action="load_shard",
                expected_output_kind="decode_state",
                input_payload=b"",
            )
        )
    request_plan = json.loads(
        (tmp_path / "workspace" / "state" / "patched-binary-load_shard.request.json")
        .read_text(encoding="utf-8")
    )
    assert request_plan["requireShardOnlyLoading"] is True


def test_patched_binary_executor_accepts_shard_only_loading_proof(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_patched_binary_executor_runtime_state()
    binary_script = tmp_path / "fake_shard_only_binary.py"
    binary_script.write_text(
        """
import json
import sys

request = json.loads(sys.stdin.read() or "{}")
print(
    json.dumps(
        {
            "status": "ready",
            "realModelExecution": True,
            "metrics": {
                "binary": "fake-shard-only",
                "assignmentArtifactPresent": True,
                "usedFullModelForLayerRange": False,
                "shardOnlyLoadingReady": True,
            },
        },
        sort_keys=True,
    )
)
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "CAI_LLM_PATCHED_BINARY_COMMAND",
        subprocess.list2cmdline([sys.executable, str(binary_script)]),
    )
    request = _request(
        tmp_path,
        action="load_shard",
        expected_output_kind="decode_state",
        input_payload=b"",
    )
    request["productionRequirements"]["requiresShardOnlyLoading"] = True
    request["productionRequirements"]["forbidFullModelFallback"] = True

    response = handle_patched_binary_executor_request(request)

    assert response["status"] == "ready"
    assert response["metrics"]["patchedBinaryMetrics"]["shardOnlyLoadingReady"] is True
    request_plan = json.loads(
        (tmp_path / "workspace" / "state" / "patched-binary-load_shard.request.json")
        .read_text(encoding="utf-8")
    )
    assert request_plan["requireShardOnlyLoading"] is True
    assert request_plan["productionRequirements"]["requiresShardOnlyLoading"] is True


def test_patched_binary_executor_allows_strict_finalize_without_loading_proof(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_patched_binary_executor_runtime_state()
    binary_script = tmp_path / "fake_finalize_binary.py"
    binary_script.write_text(
        """
import json
import sys

request = json.loads(sys.stdin.read() or "{}")
print(
    json.dumps(
        {
            "status": "ok",
            "realModelExecution": True,
            "metrics": {
                "binary": "fake-finalize",
                "residentModelFinalized": True,
            },
        },
        sort_keys=True,
    )
)
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "CAI_LLM_PATCHED_BINARY_COMMAND",
        subprocess.list2cmdline([sys.executable, str(binary_script)]),
    )
    request = _request(
        tmp_path,
        action="finalize",
        expected_output_kind="final_output",
        input_payload=b"",
    )
    request["productionRequirements"]["requiresShardOnlyLoading"] = True
    request["productionRequirements"]["forbidFullModelFallback"] = True

    response = handle_patched_binary_executor_request(request)

    assert response["status"] == "ok"
    assert response["metrics"]["patchedBinaryMetrics"]["residentModelFinalized"] is True
    request_plan = json.loads(
        (tmp_path / "workspace" / "state" / "patched-binary-finalize.request.json")
        .read_text(encoding="utf-8")
    )
    assert request_plan["requireShardOnlyLoading"] is True

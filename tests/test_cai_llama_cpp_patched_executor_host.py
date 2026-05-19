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
from cai_compute_chain.cai_llama_cpp_patched_executor_host import (
    CAI_LLM_PATCHED_ENGINE_TIMEOUT_ENV,
    CAI_LLM_SHARD_NATIVE_TIMEOUT_ENV,
    _patched_engine_timeout_seconds,
    handle_patched_executor_host_request,
    reset_patched_executor_host_clients,
)
from cai_compute_chain.cai_llama_cpp_real_state_contract import (
    build_real_state_manifest_payload,
)


MODEL_ID = "cai-network/Qwen3-0.6B-GGUF"


def test_patched_engine_timeout_follows_native_timeout_env(monkeypatch) -> None:
    monkeypatch.delenv(CAI_LLM_PATCHED_ENGINE_TIMEOUT_ENV, raising=False)
    monkeypatch.setenv(CAI_LLM_SHARD_NATIVE_TIMEOUT_ENV, "900")

    assert _patched_engine_timeout_seconds() == 900.0


def test_patched_engine_timeout_dedicated_env_wins(monkeypatch) -> None:
    monkeypatch.setenv(CAI_LLM_PATCHED_ENGINE_TIMEOUT_ENV, "240")
    monkeypatch.setenv(CAI_LLM_SHARD_NATIVE_TIMEOUT_ENV, "900")

    assert _patched_engine_timeout_seconds() == 240.0


def _request(
    tmp_path: Path,
    *,
    action: str = "process_prefill",
    expected_output_kind: str = "decode_state",
    input_payload: bytes | None = None,
) -> dict:
    workspace_root = tmp_path / "workspace"
    inputs_dir = workspace_root / "inputs"
    outputs_dir = workspace_root / "outputs"
    state_dir = workspace_root / "state"
    for path in (inputs_dir, outputs_dir, state_dir):
        path.mkdir(parents=True, exist_ok=True)
    input_payload = (
        bytes(input_payload)
        if input_payload is not None
        else b"patched executor input"
    )
    input_path = inputs_dir / f"{action}-input.bin"
    input_path.write_bytes(input_payload)
    output_path = outputs_dir / f"{action}-output.bin"
    assignment_path = tmp_path / "assignment.gguf"
    assignment_path.write_bytes(b"assignment-bytes")
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"model-bytes")
    request = {
        "schemaVersion": 1,
        "abi": ASSIGNMENT_EXECUTOR_REQUEST_ABI,
        "action": action,
        "sessionId": "session-patched-host",
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
                    "sizeBytes": int(assignment_path.stat().st_size),
                    "sha256Hex": hashlib.sha256(assignment_path.read_bytes()).hexdigest(),
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
                        "sizeBytes": int(assignment_path.stat().st_size),
                        "sha256Hex": hashlib.sha256(assignment_path.read_bytes()).hexdigest(),
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
        "executionWorkspace": {
            "schemaVersion": 1,
            "abi": "cai-llama-cpp-execution-workspace-v1",
            "root": str(workspace_root.resolve()),
            "inputsDir": str(inputs_dir.resolve()),
            "outputsDir": str(outputs_dir.resolve()),
            "stateFilesDir": str(state_dir.resolve()),
            "manifestPath": str((workspace_root / "execution-workspace.json").resolve()),
        },
        "frame": {
            "sessionId": "session-patched-host",
            "batchId": f"batch-{action}",
            "modelId": MODEL_ID,
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
        "managedRuntime": {
            "schemaVersion": 1,
            "abi": "cai-llama-cpp-managed-runtime-v1",
            "platform": os.name,
            "repoRoot": str(tmp_path.resolve()),
            "runtimeRoot": str((tmp_path / "runtime").resolve()),
            "modelId": MODEL_ID,
            "sessionPaths": {
                "root": str((tmp_path / "runtime" / "session").resolve()),
                "stateDir": str((tmp_path / "runtime" / "session" / "state").resolve()),
                "cacheDir": str((tmp_path / "runtime" / "session" / "cache").resolve()),
                "logsDir": str((tmp_path / "runtime" / "session" / "logs").resolve()),
                "stdoutLog": str((tmp_path / "runtime" / "session" / "logs" / "stdout.log").resolve()),
                "stderrLog": str((tmp_path / "runtime" / "session" / "logs" / "stderr.log").resolve()),
            },
        },
    }
    if action in {"load_shard", "finalize"}:
        request.pop("inputPayloadFile", None)
        request.pop("expectedOutputPayloadPath", None)
    return request


def _set_request_layer_range(request: dict, layer_start: int, layer_end: int) -> None:
    for payload in (
        request,
        request["frame"],
        request["shardSpec"],
        request["assignmentArtifact"],
        request["localArtifactResolution"]["assignmentArtifact"],
    ):
        payload["layerStart"] = layer_start
        payload["layerEnd"] = layer_end
    for ranges in (
        request["assignmentArtifact"]["chunkRanges"],
        request["localArtifactResolution"]["assignmentArtifact"]["chunkRanges"],
    ):
        ranges[0]["layerStart"] = layer_start
        ranges[0]["layerEnd"] = layer_end


def test_patched_executor_host_validates_real_decode_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_patched_executor_host_clients()
    engine_script = tmp_path / "fake_engine_valid.py"
    engine_script.write_text(
        """
import hashlib
import json
import sys
from pathlib import Path

request = json.loads(sys.stdin.read() or "{}")
state_dir = Path(request["executionWorkspace"]["stateFilesDir"])
state_dir.mkdir(parents=True, exist_ok=True)
state_payload = b"real decode state bytes"
state_path = Path(
    request["validatedExecutionContext"]["ioTargets"]["outputStateFilePath"]
)
state_path.write_bytes(state_payload)
manifest = {
    "schemaVersion": 1,
    "abi": "cai-llama-cpp-real-state-payload-v1",
    "stateKind": request["expectedOutputKind"],
    "producedByAction": request["action"],
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
}
payload = json.dumps(manifest, sort_keys=True).encode("utf-8")
output_path = Path(request["expectedOutputPayloadPath"])
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_bytes(payload)
print(
    json.dumps(
        {
            "status": "ok",
            "outputKind": request["expectedOutputKind"],
            "realModelExecution": True,
            "outputPayloadFile": {
                "path": str(output_path.resolve()),
                "sizeBytes": len(payload),
                "sha256Hex": hashlib.sha256(payload).hexdigest(),
            },
            "outputPayloadSha256Hex": hashlib.sha256(payload).hexdigest(),
            "metrics": {"engine": "fake-valid"},
        },
        sort_keys=True,
    )
)
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "CAI_LLM_SHARD_PATCHED_ENGINE_COMMAND",
        subprocess.list2cmdline([sys.executable, str(engine_script)]),
    )

    response = handle_patched_executor_host_request(_request(tmp_path))
    output_payload = Path(response["outputPayloadFile"]["path"]).read_bytes()

    assert response["status"] == "ok"
    assert response["outputKind"] == "decode_state"
    assert response["realModelExecution"] is True
    assert response["metrics"]["executorBackendMode"] == "patched_executor_host"
    assert response["metrics"]["patchedEngineMetrics"]["engine"] == "fake-valid"
    assert response["metrics"]["validatedStateKind"] == "decode_state"
    assert response["metrics"]["validatedStateFormat"] == (
        "ggml-kv-cache-v1/token-step-kv-cache-v1"
    )
    assert json.loads(output_payload.decode("utf-8"))["abi"] == (
        "cai-llama-cpp-real-state-payload-v1"
    )


def test_patched_executor_host_validates_and_forwards_input_real_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_patched_executor_host_clients()
    input_state_path = tmp_path / "workspace" / "state" / "input-decode.bin"
    input_state_path.parent.mkdir(parents=True, exist_ok=True)
    input_state_bytes = b"incoming decode state"
    input_state_path.write_bytes(input_state_bytes)
    input_manifest_payload = build_real_state_manifest_payload(
        output_kind="decode_state",
        action="process_prefill",
        model_id=MODEL_ID,
        session_id="session-patched-host",
        layer_start=0,
        layer_end=14,
        token_start=0,
        token_end=4,
        state_file_path=input_state_path,
    )
    engine_script = tmp_path / "fake_engine_input_state.py"
    engine_script.write_text(
        """
import hashlib
import json
import sys
from pathlib import Path

request = json.loads(sys.stdin.read() or "{}")
validated = request.get("validatedInputState") or {}
assert validated.get("stateKind") == "decode_state"
assert validated.get("stateFile", {}).get("path")
payload = b"Paris"
output_path = Path(request["expectedOutputPayloadPath"])
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_bytes(payload)
print(
    json.dumps(
        {
            "status": "ok",
            "outputKind": "final_output",
            "realModelExecution": True,
            "outputPayloadFile": {
                "path": str(output_path.resolve()),
                "sizeBytes": len(payload),
                "sha256Hex": hashlib.sha256(payload).hexdigest(),
            },
            "outputPayloadSha256Hex": hashlib.sha256(payload).hexdigest(),
            "metrics": {
                "engine": "fake-input-state",
                "validatedInputStateKind": validated.get("stateKind"),
            },
        },
        sort_keys=True,
    )
)
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "CAI_LLM_SHARD_PATCHED_ENGINE_COMMAND",
        subprocess.list2cmdline([sys.executable, str(engine_script)]),
    )

    response = handle_patched_executor_host_request(
        _request(
            tmp_path,
            action="process_decode",
            expected_output_kind="final_output",
            input_payload=input_manifest_payload,
        )
    )

    assert response["status"] == "ok"
    assert response["metrics"]["engineInputPayloadKind"] == "real_state_manifest"
    assert response["metrics"]["validatedInputStateKind"] == "decode_state"
    assert response["metrics"]["patchedEngineMetrics"]["engine"] == "fake-input-state"
    assert (
        response["metrics"]["patchedEngineMetrics"]["validatedInputStateKind"]
        == "decode_state"
    )


def test_patched_executor_host_accepts_prefill_decode_state_handoff_to_first_decode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_patched_executor_host_clients()
    input_state_path = tmp_path / "workspace" / "state" / "prefill-decode.bin"
    input_state_path.parent.mkdir(parents=True, exist_ok=True)
    input_state_path.write_bytes(b"incoming prefill decode state")
    input_manifest_payload = build_real_state_manifest_payload(
        output_kind="decode_state",
        action="process_prefill",
        model_id=MODEL_ID,
        session_id="session-patched-host",
        layer_start=14,
        layer_end=28,
        token_start=0,
        token_end=4,
        state_file_path=input_state_path,
    )
    engine_script = tmp_path / "fake_engine_prefill_decode_wrap.py"
    engine_script.write_text(
        """
import hashlib
import json
import sys
from pathlib import Path

request = json.loads(sys.stdin.read() or "{}")
validated = request.get("validatedInputState") or {}
context = request.get("validatedExecutionContext") or {}
assert validated.get("stateKind") == "decode_state"
assert validated.get("producedByAction") == "process_prefill"
assert validated.get("layerStart") == 14
assert validated.get("layerEnd") == 28
assert context.get("action") == "process_decode"
assert context.get("layerStart") == 0
payload = b"decode-start-ok"
output_path = Path(request["expectedOutputPayloadPath"])
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_bytes(payload)
print(
    json.dumps(
        {
            "status": "ok",
            "outputKind": "final_output",
            "realModelExecution": True,
            "outputPayloadFile": {
                "path": str(output_path.resolve()),
                "sizeBytes": len(payload),
                "sha256Hex": hashlib.sha256(payload).hexdigest(),
            },
            "outputPayloadSha256Hex": hashlib.sha256(payload).hexdigest(),
            "metrics": {
                "engine": "fake-prefill-decode-wrap",
                "validatedInputProducerLayerStart": validated.get("layerStart"),
            },
        },
        sort_keys=True,
    )
)
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "CAI_LLM_SHARD_PATCHED_ENGINE_COMMAND",
        subprocess.list2cmdline([sys.executable, str(engine_script)]),
    )

    response = handle_patched_executor_host_request(
        _request(
            tmp_path,
            action="process_decode",
            expected_output_kind="final_output",
            input_payload=input_manifest_payload,
        )
    )

    assert response["status"] == "ok"
    assert response["metrics"]["engineInputPayloadKind"] == "real_state_manifest"
    assert response["metrics"]["validatedInputStateKind"] == "decode_state"
    assert (
        response["metrics"]["patchedEngineMetrics"][
            "validatedInputProducerLayerStart"
        ]
        == 14
    )


def test_patched_executor_host_accepts_cross_layer_decode_handoff(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_patched_executor_host_clients()
    input_state_path = tmp_path / "workspace" / "state" / "input-decode.bin"
    input_state_path.parent.mkdir(parents=True, exist_ok=True)
    input_state_path.write_bytes(b"incoming cross-layer decode state")
    input_manifest_payload = build_real_state_manifest_payload(
        output_kind="decode_state",
        action="process_decode",
        model_id=MODEL_ID,
        session_id="session-patched-host",
        layer_start=0,
        layer_end=14,
        token_start=4,
        token_end=5,
        state_file_path=input_state_path,
    )
    engine_script = tmp_path / "fake_engine_decode_cross_layer.py"
    engine_script.write_text(
        """
import hashlib
import json
import sys
from pathlib import Path

request = json.loads(sys.stdin.read() or "{}")
validated = request.get("validatedInputState") or {}
context = request.get("validatedExecutionContext") or {}
assert validated.get("stateKind") == "decode_state"
assert validated.get("layerEnd") == 14
assert context.get("layerStart") == 14
payload = b"decode-cross-layer-ok"
output_path = Path(request["expectedOutputPayloadPath"])
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_bytes(payload)
print(
    json.dumps(
        {
            "status": "ok",
            "outputKind": "final_output",
            "realModelExecution": True,
            "outputPayloadFile": {
                "path": str(output_path.resolve()),
                "sizeBytes": len(payload),
                "sha256Hex": hashlib.sha256(payload).hexdigest(),
            },
            "outputPayloadSha256Hex": hashlib.sha256(payload).hexdigest(),
            "metrics": {
                "engine": "fake-decode-cross-layer",
                "validatedInputProducerLayerEnd": validated.get("layerEnd"),
            },
        },
        sort_keys=True,
    )
)
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "CAI_LLM_SHARD_PATCHED_ENGINE_COMMAND",
        subprocess.list2cmdline([sys.executable, str(engine_script)]),
    )
    request = _request(
        tmp_path,
        action="process_decode",
        expected_output_kind="final_output",
        input_payload=input_manifest_payload,
    )
    _set_request_layer_range(request, 14, 28)

    response = handle_patched_executor_host_request(request)

    assert response["status"] == "ok"
    assert response["metrics"]["engineInputPayloadKind"] == "real_state_manifest"
    assert response["metrics"]["validatedInputStateKind"] == "decode_state"
    assert (
        response["metrics"]["patchedEngineMetrics"][
            "validatedInputProducerLayerEnd"
        ]
        == 14
    )


def test_patched_executor_host_accepts_cross_layer_activation_handoff(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_patched_executor_host_clients()
    input_state_path = tmp_path / "workspace" / "state" / "input-activation.bin"
    input_state_path.parent.mkdir(parents=True, exist_ok=True)
    input_state_path.write_bytes(b"incoming activation state")
    input_manifest_payload = build_real_state_manifest_payload(
        output_kind="activation_state",
        action="process_prefill",
        model_id=MODEL_ID,
        session_id="session-patched-host",
        layer_start=0,
        layer_end=14,
        token_start=0,
        token_end=4,
        state_file_path=input_state_path,
    )
    engine_script = tmp_path / "fake_engine_activation_input.py"
    engine_script.write_text(
        """
import hashlib
import json
import sys
from pathlib import Path

request = json.loads(sys.stdin.read() or "{}")
validated = request.get("validatedInputState") or {}
context = request.get("validatedExecutionContext") or {}
assert validated.get("stateKind") == "activation_state"
assert validated.get("layerEnd") == 14
assert context.get("layerStart") == 14
payload = b"Paris"
output_path = Path(request["expectedOutputPayloadPath"])
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_bytes(payload)
print(
    json.dumps(
        {
            "status": "ok",
            "outputKind": "final_output",
            "realModelExecution": True,
            "outputPayloadFile": {
                "path": str(output_path.resolve()),
                "sizeBytes": len(payload),
                "sha256Hex": hashlib.sha256(payload).hexdigest(),
            },
            "outputPayloadSha256Hex": hashlib.sha256(payload).hexdigest(),
            "metrics": {
                "engine": "fake-activation-input",
                "validatedInputStateKind": validated.get("stateKind"),
                "validatedInputProducerLayerEnd": validated.get("layerEnd"),
            },
        },
        sort_keys=True,
    )
)
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "CAI_LLM_SHARD_PATCHED_ENGINE_COMMAND",
        subprocess.list2cmdline([sys.executable, str(engine_script)]),
    )
    request = _request(
        tmp_path,
        action="process_decode",
        expected_output_kind="final_output",
        input_payload=input_manifest_payload,
    )
    _set_request_layer_range(request, 14, 28)
    request["frame"]["frameKind"] = "activation"
    request["shardSpec"]["frameKind"] = "activation"

    response = handle_patched_executor_host_request(request)

    assert response["status"] == "ok"
    assert response["metrics"]["engineInputPayloadKind"] == "real_state_manifest"
    assert response["metrics"]["validatedInputStateKind"] == "activation_state"
    assert (
        response["metrics"]["patchedEngineMetrics"][
            "validatedInputProducerLayerEnd"
        ]
        == 14
    )


def test_patched_executor_host_forwards_validated_execution_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_patched_executor_host_clients()
    engine_script = tmp_path / "fake_engine_context.py"
    engine_script.write_text(
        """
import hashlib
import json
import sys
from pathlib import Path

request = json.loads(sys.stdin.read() or "{}")
context = request.get("validatedExecutionContext") or {}
assignment = context.get("assignmentArtifact") or {}
model_artifact = context.get("modelArtifact") or {}
workspace = context.get("executionWorkspace") or {}
managed = context.get("managedRuntime") or {}
io_targets = context.get("ioTargets") or {}
assert context.get("abi") == "cai-llama-cpp-patched-execution-context-v1"
assert assignment.get("coverage", {}).get("materializationMode") == "sparse_full_size"
assert assignment.get("chunkRanges", [{}])[0].get("tensorNames") == ["blk.0.attn_q.weight"]
assert model_artifact.get("localPath")
assert workspace.get("stateFilesDir")
assert managed.get("sessionPaths", {}).get("stateDir")
assert io_targets.get("abi") == "cai-llama-cpp-patched-io-targets-v1"
assert io_targets.get("outputPayloadPath") == request.get("expectedOutputPayloadPath")
assert not io_targets.get("outputStateFilePath")
payload = b"context-ok"
output_path = Path(request["expectedOutputPayloadPath"])
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_bytes(payload)
print(
    json.dumps(
        {
            "status": "ok",
            "outputKind": "final_output",
            "realModelExecution": True,
            "outputPayloadFile": {
                "path": str(output_path.resolve()),
                "sizeBytes": len(payload),
                "sha256Hex": hashlib.sha256(payload).hexdigest(),
            },
            "outputPayloadSha256Hex": hashlib.sha256(payload).hexdigest(),
            "metrics": {"engine": "fake-context"},
        },
        sort_keys=True,
    )
)
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "CAI_LLM_SHARD_PATCHED_ENGINE_COMMAND",
        subprocess.list2cmdline([sys.executable, str(engine_script)]),
    )

    response = handle_patched_executor_host_request(
        _request(
            tmp_path,
            action="process_decode",
            expected_output_kind="final_output",
        )
    )

    assert response["status"] == "ok"
    assert response["metrics"]["patchedEngineMetrics"]["engine"] == "fake-context"


def test_patched_executor_host_rejects_state_file_target_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_patched_executor_host_clients()
    engine_script = tmp_path / "fake_engine_state_target_drift.py"
    engine_script.write_text(
        """
import hashlib
import json
import sys
from pathlib import Path

request = json.loads(sys.stdin.read() or "{}")
state_dir = Path(request["executionWorkspace"]["stateFilesDir"])
state_dir.mkdir(parents=True, exist_ok=True)
state_payload = b"wrong-state-target"
state_path = state_dir / "wrong-target.bin"
state_path.write_bytes(state_payload)
manifest = {
    "schemaVersion": 1,
    "abi": "cai-llama-cpp-real-state-payload-v1",
    "stateKind": request["expectedOutputKind"],
    "producedByAction": request["action"],
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
}
payload = json.dumps(manifest, sort_keys=True).encode("utf-8")
output_path = Path(request["expectedOutputPayloadPath"])
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_bytes(payload)
print(
    json.dumps(
        {
            "status": "ok",
            "outputKind": request["expectedOutputKind"],
            "realModelExecution": True,
            "outputPayloadFile": {
                "path": str(output_path.resolve()),
                "sizeBytes": len(payload),
                "sha256Hex": hashlib.sha256(payload).hexdigest(),
            },
            "outputPayloadSha256Hex": hashlib.sha256(payload).hexdigest(),
        },
        sort_keys=True,
    )
)
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "CAI_LLM_SHARD_PATCHED_ENGINE_COMMAND",
        subprocess.list2cmdline([sys.executable, str(engine_script)]),
    )

    with pytest.raises(
        ValueError,
        match="state file path does not match ioTargets.outputStateFilePath",
    ):
        handle_patched_executor_host_request(_request(tmp_path))


def test_patched_executor_host_passes_final_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_patched_executor_host_clients()
    engine_script = tmp_path / "fake_engine_final.py"
    engine_script.write_text(
        """
import hashlib
import json
import sys
from pathlib import Path

request = json.loads(sys.stdin.read() or "{}")
payload = b"Paris"
output_path = Path(request["expectedOutputPayloadPath"])
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_bytes(payload)
print(
    json.dumps(
        {
            "status": "ok",
            "outputKind": "final_output",
            "realModelExecution": True,
            "outputPayloadFile": {
                "path": str(output_path.resolve()),
                "sizeBytes": len(payload),
                "sha256Hex": hashlib.sha256(payload).hexdigest(),
            },
            "outputPayloadSha256Hex": hashlib.sha256(payload).hexdigest(),
            "metrics": {"engine": "fake-final"},
        },
        sort_keys=True,
    )
)
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "CAI_LLM_SHARD_PATCHED_ENGINE_COMMAND",
        subprocess.list2cmdline([sys.executable, str(engine_script)]),
    )

    response = handle_patched_executor_host_request(
        _request(tmp_path, action="process_decode", expected_output_kind="final_output")
    )

    assert response["status"] == "ok"
    assert response["outputKind"] == "final_output"
    assert Path(response["outputPayloadFile"]["path"]).read_bytes() == b"Paris"
    assert "validatedStateKind" not in response["metrics"]


def test_patched_executor_host_rejects_mismatched_input_state_kind(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_patched_executor_host_clients()
    input_state_path = tmp_path / "workspace" / "state" / "input-activation.bin"
    input_state_path.parent.mkdir(parents=True, exist_ok=True)
    input_state_bytes = b"incoming activation state"
    input_state_path.write_bytes(input_state_bytes)
    input_manifest_payload = build_real_state_manifest_payload(
        output_kind="activation_state",
        action="process_prefill",
        model_id=MODEL_ID,
        session_id="session-patched-host",
        layer_start=0,
        layer_end=14,
        token_start=0,
        token_end=4,
        state_file_path=input_state_path,
    )
    engine_script = tmp_path / "fake_engine_unused.py"
    engine_script.write_text(
        "print('{\"status\": \"ok\", \"realModelExecution\": true}')",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "CAI_LLM_SHARD_PATCHED_ENGINE_COMMAND",
        subprocess.list2cmdline([sys.executable, str(engine_script)]),
    )

    with pytest.raises(
        ValueError,
        match="input real state payload stateKind does not match output kind",
    ):
        handle_patched_executor_host_request(
            _request(
                tmp_path,
                action="process_decode",
                expected_output_kind="final_output",
                input_payload=input_manifest_payload,
            )
        )


def test_patched_executor_host_rejects_invalid_state_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_patched_executor_host_clients()
    engine_script = tmp_path / "fake_engine_invalid.py"
    engine_script.write_text(
        """
import hashlib
import json
import sys
from pathlib import Path

request = json.loads(sys.stdin.read() or "{}")
escape_path = Path(request["executionWorkspace"]["root"]).resolve().parent / "escape.bin"
escape_path.write_bytes(b"escape")
manifest = {
    "schemaVersion": 1,
    "abi": "cai-llama-cpp-real-state-payload-v1",
    "stateKind": request["expectedOutputKind"],
    "producedByAction": request["action"],
    "modelId": request["modelId"],
    "sessionId": request["sessionId"],
    "layerStart": request["layerStart"],
    "layerEnd": request["layerEnd"],
    "tokenStart": request["tokenStart"],
    "tokenEnd": request["tokenEnd"],
    "stateFormat": "ggml-kv-cache-v1/token-step-kv-cache-v1",
    "stateFile": {
        "path": str(escape_path.resolve()),
        "sha256Hex": hashlib.sha256(escape_path.read_bytes()).hexdigest(),
        "sizeBytes": int(escape_path.stat().st_size),
    },
}
payload = json.dumps(manifest, sort_keys=True).encode("utf-8")
output_path = Path(request["expectedOutputPayloadPath"])
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_bytes(payload)
print(
    json.dumps(
        {
            "status": "ok",
            "outputKind": request["expectedOutputKind"],
            "realModelExecution": True,
            "outputPayloadFile": {
                "path": str(output_path.resolve()),
                "sizeBytes": len(payload),
                "sha256Hex": hashlib.sha256(payload).hexdigest(),
            },
        },
        sort_keys=True,
    )
)
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "CAI_LLM_SHARD_PATCHED_ENGINE_COMMAND",
        subprocess.list2cmdline([sys.executable, str(engine_script)]),
    )

    with pytest.raises(
        ValueError,
        match="stateFile path must stay within executionWorkspace.stateFilesDir",
    ):
        handle_patched_executor_host_request(_request(tmp_path))


def test_patched_executor_host_reuses_persistent_engine(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_patched_executor_host_clients()
    engine_script = tmp_path / "fake_engine_persistent.py"
    engine_script.write_text(
        """
import hashlib
import json
import os
import sys
from pathlib import Path

call_count = 0
for line in sys.stdin:
    request = json.loads(line or "{}")
    call_count += 1
    action = str(request.get("action") or "")
    response = {
        "status": "ready" if action == "load_shard" else "ok",
        "realModelExecution": True,
        "metrics": {"pid": os.getpid(), "callCount": call_count},
    }
    if action == "process_prefill":
        state_dir = Path(request["executionWorkspace"]["stateFilesDir"])
        state_dir.mkdir(parents=True, exist_ok=True)
        state_payload = f"state-{call_count}".encode("utf-8")
        state_path = Path(
            request["validatedExecutionContext"]["ioTargets"]["outputStateFilePath"]
        )
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
        }
        payload = json.dumps(manifest, sort_keys=True).encode("utf-8")
        output_path = Path(request["expectedOutputPayloadPath"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(payload)
        response.update(
            {
                "outputKind": request["expectedOutputKind"],
                "outputPayloadFile": {
                    "path": str(output_path.resolve()),
                    "sizeBytes": len(payload),
                    "sha256Hex": hashlib.sha256(payload).hexdigest(),
                },
                "outputPayloadSha256Hex": hashlib.sha256(payload).hexdigest(),
            }
        )
    print(json.dumps(response, sort_keys=True), flush=True)
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "CAI_LLM_SHARD_PATCHED_ENGINE_COMMAND",
        subprocess.list2cmdline([sys.executable, str(engine_script)]),
    )
    monkeypatch.setenv("CAI_LLM_SHARD_PATCHED_ENGINE_PERSISTENT", "1")

    load = handle_patched_executor_host_request(
        _request(tmp_path, action="load_shard")
    )
    prefill = handle_patched_executor_host_request(_request(tmp_path))
    finalize = handle_patched_executor_host_request(
        _request(tmp_path, action="finalize")
    )

    assert load["status"] == "ready"
    assert load["metrics"]["patchedEngineMode"] == "persistent_jsonl"
    assert prefill["metrics"]["patchedEngineMode"] == "persistent_jsonl"
    assert finalize["metrics"]["patchedEngineMode"] == "persistent_jsonl"
    assert load["metrics"]["patchedEngineMetrics"]["callCount"] == 1
    assert prefill["metrics"]["patchedEngineMetrics"]["callCount"] == 2
    assert finalize["metrics"]["patchedEngineMetrics"]["callCount"] == 3
    assert (
        load["metrics"]["patchedEngineMetrics"]["pid"]
        == prefill["metrics"]["patchedEngineMetrics"]["pid"]
        == finalize["metrics"]["patchedEngineMetrics"]["pid"]
    )

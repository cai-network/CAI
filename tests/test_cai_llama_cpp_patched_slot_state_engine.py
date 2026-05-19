# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.cai_llama_cpp_assignment_artifact_engine import (  # noqa: E402
    ASSIGNMENT_EXECUTOR_REQUEST_ABI,
)
from cai_compute_chain.cai_llama_cpp_patched_executor_host import (  # noqa: E402
    handle_patched_executor_host_request,
    reset_patched_executor_host_clients,
)
from cai_compute_chain.cai_llama_cpp_patched_slot_state_engine import (  # noqa: E402
    handle_patched_slot_state_engine_request,
    reset_patched_slot_state_engine_runtime_state,
)
from cai_compute_chain.cai_llama_cpp_real_state_contract import (  # noqa: E402
    REAL_STATE_PAYLOAD_ABI,
)
from cai_compute_chain.cai_llama_cpp_slot_state_engine import (  # noqa: E402
    SlotStateEngineConfig,
)


MODEL_ID = "cai-network/Qwen3-0.6B-GGUF"


class _FakeLlamaServer:
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.restored_bytes: bytes | None = None
        self.completion_prompts: list[str] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def start(self) -> "_FakeLlamaServer":
        self.thread.start()
        return self

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/slots":
                    self._send(200, [{"id": 0, "n_ctx": 256, "is_processing": False}])
                    return
                self._send(404, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                payload = json.loads(raw_body.decode("utf-8") or "{}")
                if self.path == "/completion":
                    prompt = str(payload.get("prompt") or "")
                    parent.completion_prompts.append(prompt)
                    content = " Paris" if prompt else " ok"
                    self._send(
                        200,
                        {
                            "content": content,
                            "tokens_predicted": max(1, int(payload.get("n_predict") or 1)),
                            "tokens_evaluated": max(1, len(prompt.split())),
                            "id_slot": 0,
                        },
                    )
                    return
                if self.path == "/slots/0?action=save":
                    filename = str(payload.get("filename") or "")
                    (parent.state_dir / filename).write_bytes(b"real-llama-slot-state")
                    self._send(
                        200,
                        {
                            "id_slot": 0,
                            "filename": filename,
                            "n_saved": 5,
                            "n_written": len(b"real-llama-slot-state"),
                        },
                    )
                    return
                if self.path == "/slots/0?action=restore":
                    filename = str(payload.get("filename") or "")
                    parent.restored_bytes = (parent.state_dir / filename).read_bytes()
                    self._send(
                        200,
                        {
                            "id_slot": 0,
                            "filename": filename,
                            "n_restored": 5,
                            "n_read": len(parent.restored_bytes),
                        },
                    )
                    return
                self._send(404, {"error": "not found"})

            def log_message(self, format: str, *args: object) -> None:
                return

            def _send(self, status: int, payload: object) -> None:
                encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        return Handler


def _config(server: _FakeLlamaServer, state_dir: Path) -> SlotStateEngineConfig:
    return SlotStateEngineConfig(
        server_url=server.url,
        state_dir=state_dir,
        slot_id=0,
        timeout_sec=10,
        decode_tokens=3,
    )


def _request(
    tmp_path: Path,
    *,
    action: str,
    expected_output_kind: str,
    input_payload: bytes,
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
        "sessionId": "session-patched-slot-state",
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
        "executionWorkspace": {
            "schemaVersion": 1,
            "abi": "cai-llama-cpp-execution-workspace-v1",
            "root": str(workspace_root.resolve()),
            "inputsDir": str(inputs_dir.resolve()),
            "outputsDir": str(outputs_dir.resolve()),
            "stateFilesDir": str(state_dir.resolve()),
            "manifestPath": str((workspace_root / "execution-workspace.json").resolve()),
        },
        "managedRuntime": {
            "schemaVersion": 1,
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
            "sessionId": "session-patched-slot-state",
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
            "sessionId": "session-patched-slot-state",
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
                "outputStateManifestPath": str(output_path.resolve()),
                "outputStateFilePath": str(state_output_path.resolve()),
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
                "sessionId": "session-patched-slot-state",
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
    if action in {"load_shard", "finalize"}:
        base_request.pop("inputPayloadFile", None)
        base_request.pop("expectedOutputPayloadPath", None)
    return base_request


def test_patched_slot_state_engine_wraps_prefill_as_real_state(tmp_path: Path) -> None:
    reset_patched_slot_state_engine_runtime_state()
    state_root = tmp_path / "slot-state"
    server = _FakeLlamaServer(state_root).start()
    try:
        load_request = _request(
            tmp_path,
            action="load_shard",
            expected_output_kind="decode_state",
            input_payload=b"",
        )
        load_response = handle_patched_slot_state_engine_request(
            load_request,
            config=_config(server, state_root),
        )
        request = _request(
            tmp_path,
            action="process_prefill",
            expected_output_kind="decode_state",
            input_payload=b"The capital of France is",
        )
        response = handle_patched_slot_state_engine_request(
            request,
            config=_config(server, state_root),
        )
        manifest = json.loads(
            Path(response["outputPayloadFile"]["path"]).read_text(encoding="utf-8")
        )
        state_file = Path(manifest["stateFile"]["path"])
        state_payload = json.loads(state_file.read_text(encoding="utf-8"))
        session_manifest_path = Path(load_response["metrics"]["sessionManifestPath"])
    finally:
        server.close()

    assert load_response["status"] == "ready"
    assert session_manifest_path.exists()
    assert response["status"] == "ok"
    assert response["outputKind"] == "decode_state"
    assert response["metrics"]["engineBackendMode"] == "patched_slot_state_engine"
    assert response["metrics"]["engineProcessRequestCount"] == 2
    assert manifest["abi"] == REAL_STATE_PAYLOAD_ABI
    assert manifest["stateKind"] == "decode_state"
    assert manifest["metadata"]["wrappedBackend"] == "llama.cpp-slot-state"
    assert manifest["stateFile"]["path"] == str(
        (tmp_path / "workspace" / "state" / "process_prefill-output.state.bin").resolve()
    )
    assert state_payload["abi"] == "cai-llama-cpp-slot-state-payload-v1"


def test_patched_slot_state_engine_runs_decode_to_final_output(tmp_path: Path) -> None:
    reset_patched_slot_state_engine_runtime_state()
    state_root = tmp_path / "slot-state"
    server = _FakeLlamaServer(state_root).start()
    try:
        load_request = _request(
            tmp_path,
            action="load_shard",
            expected_output_kind="decode_state",
            input_payload=b"",
        )
        load = handle_patched_slot_state_engine_request(
            load_request,
            config=_config(server, state_root),
        )
        prefill_request = _request(
            tmp_path,
            action="process_prefill",
            expected_output_kind="decode_state",
            input_payload=b"The capital of France is",
        )
        prefill = handle_patched_slot_state_engine_request(
            prefill_request,
            config=_config(server, state_root),
        )
        prefill_manifest_path = Path(prefill["outputPayloadFile"]["path"])
        prefill_manifest_payload = prefill_manifest_path.read_bytes()
        prefill_manifest = json.loads(prefill_manifest_payload.decode("utf-8"))
        decode_request = _request(
            tmp_path,
            action="process_decode",
            expected_output_kind="final_output",
            input_payload=prefill_manifest_payload,
        )
        decode_request["validatedInputState"] = prefill_manifest
        decode = handle_patched_slot_state_engine_request(
            decode_request,
            config=_config(server, state_root),
        )
        output_bytes = Path(decode["outputPayloadFile"]["path"]).read_bytes()
        finalize_request = _request(
            tmp_path,
            action="finalize",
            expected_output_kind="final_output",
            input_payload=b"",
        )
        finalize = handle_patched_slot_state_engine_request(
            finalize_request,
            config=_config(server, state_root),
        )
        session_manifest_path = Path(load["metrics"]["sessionManifestPath"])
    finally:
        server.close()

    assert decode["status"] == "ok"
    assert decode["outputKind"] == "final_output"
    assert decode["metrics"]["engineBackendMode"] == "patched_slot_state_engine"
    assert decode["metrics"]["engineProcessRequestCount"] == 3
    assert finalize["status"] == "ok"
    assert finalize["metrics"]["engineProcessRequestCount"] == 4
    assert not session_manifest_path.exists()
    assert output_bytes == b" Paris"


def test_patched_executor_host_works_through_patched_slot_state_engine(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_patched_slot_state_engine_runtime_state()
    reset_patched_executor_host_clients()
    state_root = tmp_path / "slot-state"
    server = _FakeLlamaServer(state_root).start()
    existing_pythonpath = str(os.environ.get("PYTHONPATH") or "")
    pythonpath_parts = [str(SRC_ROOT)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(pythonpath_parts))
    monkeypatch.setenv(
        "CAI_LLM_SHARD_PATCHED_ENGINE_COMMAND",
        subprocess.list2cmdline(
            [
                sys.executable,
                "-m",
                "cai_compute_chain.cai_llama_cpp_patched_slot_state_engine",
                "--server-url",
                server.url,
                "--state-dir",
                str(state_root.resolve()),
                "--slot-id",
                "0",
                "--timeout-sec",
                "10",
                "--decode-tokens",
                "3",
            ]
        ),
    )
    try:
        prefill = handle_patched_executor_host_request(
            _request(
                tmp_path,
                action="process_prefill",
                expected_output_kind="decode_state",
                input_payload=b"The capital of France is",
            )
        )
        manifest_payload = Path(prefill["outputPayloadFile"]["path"]).read_bytes()
        decode = handle_patched_executor_host_request(
            _request(
                tmp_path,
                action="process_decode",
                expected_output_kind="final_output",
                input_payload=manifest_payload,
            )
        )
        output_bytes = Path(decode["outputPayloadFile"]["path"]).read_bytes()
    finally:
        server.close()

    assert prefill["status"] == "ok"
    assert prefill["metrics"]["patchedEngineMetrics"]["engineBackendMode"] == (
        "patched_slot_state_engine"
    )
    assert decode["status"] == "ok"
    assert decode["outputKind"] == "final_output"
    assert output_bytes == b" Paris"


def test_patched_executor_host_reuses_persistent_patched_slot_state_engine(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_patched_slot_state_engine_runtime_state()
    reset_patched_executor_host_clients()
    state_root = tmp_path / "slot-state"
    server = _FakeLlamaServer(state_root).start()
    existing_pythonpath = str(os.environ.get("PYTHONPATH") or "")
    pythonpath_parts = [str(SRC_ROOT)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(pythonpath_parts))
    monkeypatch.setenv("CAI_LLM_SHARD_PATCHED_ENGINE_PERSISTENT", "1")
    monkeypatch.setenv(
        "CAI_LLM_SHARD_PATCHED_ENGINE_COMMAND",
        subprocess.list2cmdline(
            [
                sys.executable,
                "-m",
                "cai_compute_chain.cai_llama_cpp_patched_slot_state_engine",
                "--jsonl",
                "--server-url",
                server.url,
                "--state-dir",
                str(state_root.resolve()),
                "--slot-id",
                "0",
                "--timeout-sec",
                "10",
                "--decode-tokens",
                "3",
            ]
        ),
    )
    try:
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
        manifest_payload = Path(prefill["outputPayloadFile"]["path"]).read_bytes()
        decode = handle_patched_executor_host_request(
            _request(
                tmp_path,
                action="process_decode",
                expected_output_kind="final_output",
                input_payload=manifest_payload,
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
    finally:
        server.close()

    assert load["metrics"]["patchedEngineMode"] == "persistent_jsonl"
    assert prefill["metrics"]["patchedEngineMode"] == "persistent_jsonl"
    assert decode["metrics"]["patchedEngineMode"] == "persistent_jsonl"
    assert finalize["metrics"]["patchedEngineMode"] == "persistent_jsonl"
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

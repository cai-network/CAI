# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import os
import sys
import tempfile
import threading
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.cai_llama_cpp_slot_state_engine import (  # noqa: E402
    SLOT_STATE_ENGINE_ID,
    SLOT_STATE_PAYLOAD_ABI,
    SlotStateEngineConfig,
    handle_slot_state_engine_request,
    reset_slot_state_engine_managed_servers,
)
from cai_compute_chain.cai_llama_cpp_shard_native_bridge import (  # noqa: E402
    _handler_class,
)
from cai_compute_chain.cai_owned_runtime import (  # noqa: E402
    ExternalLlamaCppShardAdapter,
    LLAMA_CPP_EXTERNAL_SHARD_GENERATION_PROBE_ABI,
    LLAMA_CPP_EXTERNAL_SHARD_HANDOFF_ABI,
    LLAMA_CPP_EXTERNAL_SHARD_LOCAL_FILE_IO_ABI,
    cai_owned_shard_adapter_from_env,
    run_cai_owned_llm_shard_adapter_self_test,
)
from cai_compute_chain.cai_slot_state_handoff_smoke import (  # noqa: E402
    SlotStateEndpointConfig,
    SlotStateHandoffSmokeConfig,
    run_slot_state_handoff_smoke,
)


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


def _request(
    action: str,
    payload: bytes = b"",
    *,
    local_artifact_resolution: dict | None = None,
) -> dict:
    payload_hash = hashlib.sha256(payload).hexdigest()
    request = {
        "schemaVersion": 1,
        "abi": LLAMA_CPP_EXTERNAL_SHARD_HANDOFF_ABI,
        "action": action,
        "backend": "llama.cpp-patched",
        "frame": {
            "sessionId": "caiot_slot_state_test",
            "batchId": f"caibatch_slot_state_test_{action}",
            "phase": "prefill_activation_batches",
            "sourceNodeId": "node-a",
            "sinkNodeId": "node-b",
            "sequence": 0,
            "modelId": "cai-network/Qwen3-0.6B-GGUF",
            "frameKind": "activation",
            "layerStart": 0,
            "layerEnd": 1,
            "tokenStart": 0,
            "tokenEnd": 4,
            "payloadSha256Hex": payload_hash,
            "metadata": {},
        },
        "payloadBase64": base64.b64encode(payload).decode("ascii"),
        "payloadSha256Hex": payload_hash,
        "outputContract": {
            "schemaVersion": 1,
            "requiresOutputFrameMetadata": True,
            "frameMetadataTemplate": {
                "payloadSha256Hex": "<computed-output-sha256>",
                "llmHandoff": {"tensor": {"sha256Hex": "<computed-output-sha256>"}},
            },
        },
        "productionRequirements": {"schemaVersion": 1},
    }
    if local_artifact_resolution is not None:
        request["localArtifactResolution"] = dict(local_artifact_resolution)
    return request


def _config(server: _FakeLlamaServer, state_dir: Path) -> SlotStateEngineConfig:
    return SlotStateEngineConfig(
        server_url=server.url,
        state_dir=state_dir,
        slot_id=0,
        timeout_sec=10,
        decode_tokens=3,
    )


def _file_request(action: str, payload: bytes, io_root: Path) -> dict:
    io_root.mkdir(parents=True, exist_ok=True)
    request = _request(action, payload)
    payload_path = (io_root / f"{action}-payload.bin").resolve()
    payload_path.write_bytes(payload)
    request.pop("payloadBase64", None)
    request["payloadFile"] = {
        "path": str(payload_path),
        "sizeBytes": len(payload),
        "sha256Hex": hashlib.sha256(payload).hexdigest(),
    }
    request["localFileContract"] = {
        "schemaVersion": 1,
        "abi": LLAMA_CPP_EXTERNAL_SHARD_LOCAL_FILE_IO_ABI,
        "ioRoot": str(io_root.resolve()),
        "responseOutputPath": str((io_root / f"{action}-output.bin").resolve()),
    }
    return request


def _response_payload_bytes(response: dict) -> bytes:
    output_file = response.get("outputPayloadFile")
    if isinstance(output_file, dict):
        return Path(str(output_file["path"])).read_bytes()
    return base64.b64decode(response["outputPayloadBase64"])


def _managed_runtime_request(
    action: str,
    payload: bytes,
    *,
    runtime_root: Path,
    server_script: Path,
    model_path: Path,
) -> dict:
    session_root = runtime_root / "session-root"
    state_dir = session_root / "state"
    cache_dir = session_root / "cache"
    logs_dir = session_root / "logs"
    for path in (state_dir, cache_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    request = _request(
        action,
        payload,
        local_artifact_resolution={
            "modelArtifact": {
                "artifactId": "gguf-main",
                "source": "local_binding",
                "localPath": str(model_path.resolve()),
            }
        },
    )
    request["managedRuntime"] = {
        "schemaVersion": 1,
        "abi": "cai-llama-cpp-managed-runtime-v1",
        "platform": os.name,
        "repoRoot": str(REPO_ROOT.resolve()),
        "runtimeRoot": str(runtime_root.resolve()),
        "modelId": "cai-network/Qwen3-0.6B-GGUF",
        "llamaCpp": {
            "llamaServerPath": sys.executable,
            "llamaServerArgs": ["-u", str(server_script.resolve())],
        },
        "sessionPaths": {
            "root": str(session_root.resolve()),
            "stateDir": str(state_dir.resolve()),
            "cacheDir": str(cache_dir.resolve()),
            "logsDir": str(logs_dir.resolve()),
            "stdoutLog": str((logs_dir / "stdout.log").resolve()),
            "stderrLog": str((logs_dir / "stderr.log").resolve()),
        },
    }
    return request


def _write_managed_llama_server_script(path: Path) -> Path:
    path.write_text(
        """
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest="model_path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("-np", dest="parallel_slots", type=int, default=1)
    parser.add_argument("--slot-save-path", required=True)
    args = parser.parse_args()
    state_dir = Path(args.slot_save_path)
    state_dir.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/slots":
                self._send(200, [{"id": 0, "n_ctx": 256, "is_processing": False}])
                return
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            payload = json.loads(body.decode("utf-8") or "{}")
            if parsed.path == "/completion":
                prompt = str(payload.get("prompt") or "")
                self._send(
                    200,
                    {
                        "content": " Paris" if prompt else " ok",
                        "tokens_predicted": max(1, int(payload.get("n_predict") or 1)),
                        "tokens_evaluated": max(1, len(prompt.split())),
                        "id_slot": 0,
                    },
                )
                return
            if parsed.path == "/slots/0":
                action = parse_qs(parsed.query).get("action", [""])[0]
                filename = str(payload.get("filename") or "")
                target = state_dir / filename
                if action == "save":
                    target.write_bytes(b"real-llama-slot-state")
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
                if action == "restore":
                    restored = target.read_bytes()
                    self._send(
                        200,
                        {
                            "id_slot": 0,
                            "filename": filename,
                            "n_restored": 5,
                            "n_read": len(restored),
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

    server = ThreadingHTTPServer((args.host, int(args.port)), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_cai_owned_shard_adapter_from_env_selects_slot_state() -> None:
    adapter = cai_owned_shard_adapter_from_env(
        {
            "CAI_LLM_SHARD_ADAPTER": "slot_state",
            "CAI_LLM_SHARD_ADAPTER_TIMEOUT_SEC": "12",
            "CAI_LLM_SHARD_SLOT_SERVER_URL": "http://127.0.0.1:8080",
            "CAI_LLM_SHARD_SLOT_STATE_DIR": "C:/CAI/slot-state",
            "CAI_LLM_SHARD_SLOT_ID": "2",
            "CAI_LLM_SHARD_SLOT_DECODE_TOKENS": "5",
        }
    )

    assert isinstance(adapter, ExternalLlamaCppShardAdapter)
    assert adapter.command == [
        sys.executable,
        "-m",
        "cai_compute_chain.cai_llama_cpp_slot_state_engine",
    ]
    assert adapter.timeout_sec == 12
    assert adapter.require_handoff_contract is True
    assert adapter.require_patch_boundary is True
    assert adapter.env["CAI_LLM_SHARD_SLOT_SERVER_URL"] == "http://127.0.0.1:8080"
    assert adapter.env["CAI_LLM_SHARD_SLOT_STATE_DIR"] == "C:/CAI/slot-state"
    assert adapter.env["CAI_LLM_SHARD_SLOT_ID"] == "2"
    assert adapter.env["CAI_LLM_SHARD_SLOT_DECODE_TOKENS"] == "5"
    assert "PYTHONPATH" in adapter.env


def test_slot_state_engine_transfers_real_slot_state_payload() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        state_dir = Path(tempdir)
        gguf_path = state_dir / "model.gguf"
        gguf_path.write_bytes(b"fake-gguf")
        server = _FakeLlamaServer(state_dir).start()
        try:
            local_artifact_resolution = {
                "modelArtifact": {
                    "artifactId": "gguf-main",
                    "source": "local_binding",
                    "localPath": str(gguf_path.resolve()),
                }
            }
            prefill = handle_slot_state_engine_request(
                _request(
                    "process_prefill",
                    b"The capital of France is",
                    local_artifact_resolution=local_artifact_resolution,
                ),
                config=_config(server, state_dir),
            )
            state_payload = base64.b64decode(prefill["outputPayloadBase64"])
            state_envelope = json.loads(state_payload.decode("utf-8"))
            decode = handle_slot_state_engine_request(
                _request(
                    "process_decode",
                    state_payload,
                    local_artifact_resolution=local_artifact_resolution,
                ),
                config=_config(server, state_dir),
            )
        finally:
            server.close()

    assert prefill["status"] == "ok"
    assert state_envelope["abi"] == SLOT_STATE_PAYLOAD_ABI
    assert state_envelope["slotStateSha256Hex"] == hashlib.sha256(
        b"real-llama-slot-state"
    ).hexdigest()
    assert decode["status"] == "ok"
    assert base64.b64decode(decode["outputPayloadBase64"]) == b" Paris"
    assert prefill["nativeExecution"]["artifactKind"] == "model"
    assert prefill["nativeExecution"]["artifactSource"] == "local_binding"
    assert prefill["nativeExecution"]["fallbackMode"] == "full_model"
    assert decode["nativeExecution"]["artifactKind"] == "model"
    assert server.restored_bytes == b"real-llama-slot-state"


def test_slot_state_engine_transfers_real_slot_state_payload_via_local_file_io() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        state_dir = Path(tempdir) / "state"
        io_root = Path(tempdir) / "io"
        server = _FakeLlamaServer(state_dir).start()
        try:
            prefill = handle_slot_state_engine_request(
                _file_request("process_prefill", b"The capital of France is", io_root),
                config=_config(server, state_dir),
            )
            state_payload = _response_payload_bytes(prefill)
            state_envelope = json.loads(state_payload.decode("utf-8"))
            decode = handle_slot_state_engine_request(
                _file_request("process_decode", state_payload, io_root),
                config=_config(server, state_dir),
            )
            decode_output = _response_payload_bytes(decode)
        finally:
            server.close()

    assert prefill["status"] == "ok"
    assert "outputPayloadFile" in prefill
    assert state_envelope["abi"] == SLOT_STATE_PAYLOAD_ABI
    assert state_envelope["slotStateSha256Hex"] == hashlib.sha256(
        b"real-llama-slot-state"
    ).hexdigest()
    assert decode["status"] == "ok"
    assert "outputPayloadFile" in decode
    assert decode_output == b" Paris"
    assert server.restored_bytes == b"real-llama-slot-state"


def test_slot_state_engine_generation_probe_uses_llama_server_completion() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        state_dir = Path(tempdir)
        server = _FakeLlamaServer(state_dir).start()
        try:
            response = handle_slot_state_engine_request(
                {
                    "schemaVersion": 1,
                    "abi": LLAMA_CPP_EXTERNAL_SHARD_HANDOFF_ABI,
                    "action": "probe_generation",
                    "generationProbe": {
                        "schemaVersion": 1,
                        "abi": LLAMA_CPP_EXTERNAL_SHARD_GENERATION_PROBE_ABI,
                        "modelId": "cai-network/Qwen3-0.6B-GGUF",
                        "prompt": "The capital of France is",
                        "maxTokens": 2,
                        "temperature": 0.0,
                        "requiresRealModelExecution": True,
                    },
                },
                config=_config(server, state_dir),
            )
        finally:
            server.close()

    assert response["generationProbe"]["realModelExecution"] is True
    assert response["generationProbe"]["outputText"] == " Paris"
    assert response["metrics"]["backendMode"] == SLOT_STATE_ENGINE_ID


def test_slot_state_engine_self_test_is_contract_ready_not_production_ready() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        state_dir = Path(tempdir)
        server = _FakeLlamaServer(state_dir).start()
        try:
            result = run_cai_owned_llm_shard_adapter_self_test(
                ExternalLlamaCppShardAdapter(
                    command=[
                        sys.executable,
                        "-m",
                        "cai_compute_chain.cai_llama_cpp_slot_state_engine",
                        "--server-url",
                        server.url,
                        "--state-dir",
                        str(state_dir),
                        "--decode-tokens",
                        "3",
                    ],
                    timeout_sec=10,
                ),
                payload=b"The capital of France is",
            )
        finally:
            server.close()

    assert result["contractReady"] is True
    assert result["generationProbeReady"] is True
    assert result["productionReady"] is False
    assert "real_activation_state" in result["productionReadinessError"]


def test_slot_state_engine_self_test_supports_local_file_io() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        state_dir = Path(tempdir) / "state"
        io_root = Path(tempdir) / "adapter-io"
        server = _FakeLlamaServer(state_dir).start()
        try:
            result = run_cai_owned_llm_shard_adapter_self_test(
                ExternalLlamaCppShardAdapter(
                    command=[
                        sys.executable,
                        "-m",
                        "cai_compute_chain.cai_llama_cpp_slot_state_engine",
                        "--server-url",
                        server.url,
                        "--state-dir",
                        str(state_dir),
                        "--decode-tokens",
                        "3",
                    ],
                    timeout_sec=10,
                    file_io_root=str(io_root),
                    file_io_threshold_bytes=1,
                ),
                payload=b"The capital of France is",
            )
            io_children = list(io_root.iterdir()) if io_root.exists() else []
        finally:
            server.close()

    assert result["contractReady"] is True
    assert result["generationProbeReady"] is True
    assert result["productionReady"] is False
    assert io_children == []


def test_slot_state_engine_can_manage_local_llama_server_from_runtime_context() -> None:
    reset_slot_state_engine_managed_servers()
    with tempfile.TemporaryDirectory() as tempdir:
        root = Path(tempdir)
        runtime_root = root / "runtime"
        model_path = root / "model.gguf"
        model_path.write_bytes(b"fake-gguf")
        server_script = _write_managed_llama_server_script(root / "fake_llama_server.py")
        config = SlotStateEngineConfig(
            server_url="",
            state_dir=None,
            slot_id=0,
            timeout_sec=10,
            decode_tokens=3,
        )
        try:
            prefill_request = _managed_runtime_request(
                "process_prefill",
                b"The capital of France is",
                runtime_root=runtime_root,
                server_script=server_script,
                model_path=model_path,
            )
            prefill = handle_slot_state_engine_request(prefill_request, config=config)
            state_payload = _response_payload_bytes(prefill)
            decode_request = _managed_runtime_request(
                "process_decode",
                state_payload,
                runtime_root=runtime_root,
                server_script=server_script,
                model_path=model_path,
            )
            decode = handle_slot_state_engine_request(decode_request, config=config)
            finalize_request = _managed_runtime_request(
                "finalize",
                b"",
                runtime_root=runtime_root,
                server_script=server_script,
                model_path=model_path,
            )
            finalize = handle_slot_state_engine_request(finalize_request, config=config)
        finally:
            reset_slot_state_engine_managed_servers()
        assert prefill["status"] == "ok"
        assert prefill["metrics"]["backendMode"] == SLOT_STATE_ENGINE_ID
        assert base64.b64decode(decode["outputPayloadBase64"]) == b" Paris"
        assert finalize["status"] == "ok"
        assert (runtime_root / "session-root" / "logs" / "stdout.log").exists()
        assert (runtime_root / "session-root" / "state").exists()

def test_slot_state_handoff_smoke_uses_two_workers_and_verifies_proof() -> None:
    with tempfile.TemporaryDirectory() as prefill_tempdir:
        with tempfile.TemporaryDirectory() as decode_tempdir:
            prefill_state_dir = Path(prefill_tempdir)
            decode_state_dir = Path(decode_tempdir)
            prefill_server = _FakeLlamaServer(prefill_state_dir).start()
            decode_server = _FakeLlamaServer(decode_state_dir).start()
            try:
                report = run_slot_state_handoff_smoke(
                    SlotStateHandoffSmokeConfig(
                        prefill_endpoint=SlotStateEndpointConfig(
                            server_url=prefill_server.url,
                            state_dir=str(prefill_state_dir),
                        ),
                        decode_endpoint=SlotStateEndpointConfig(
                            server_url=decode_server.url,
                            state_dir=str(decode_state_dir),
                        ),
                        prompt="The capital of France is",
                        decode_tokens=3,
                        timeout_sec=10,
                    )
                )
            finally:
                prefill_server.close()
                decode_server.close()

    assert report["status"] == "ok"
    assert report["executorNodeIds"] == ["node-prefill", "node-decode"]
    assert report["shardReceiptNodeIds"] == ["node-prefill", "node-decode"]
    assert report["proofVerified"] is True
    assert report["slotStateHandoff"] is True
    assert report["productionLayerShard"] is False
    assert report["finalPayloadUtf8"] == " Paris"
    assert decode_server.restored_bytes == b"real-llama-slot-state"


def test_slot_state_handoff_smoke_can_forward_envelopes_over_http() -> None:
    with tempfile.TemporaryDirectory() as prefill_tempdir:
        with tempfile.TemporaryDirectory() as decode_tempdir:
            prefill_state_dir = Path(prefill_tempdir)
            decode_state_dir = Path(decode_tempdir)
            prefill_server = _FakeLlamaServer(prefill_state_dir).start()
            decode_server = _FakeLlamaServer(decode_state_dir).start()
            try:
                report = run_slot_state_handoff_smoke(
                    SlotStateHandoffSmokeConfig(
                        prefill_endpoint=SlotStateEndpointConfig(
                            server_url=prefill_server.url,
                            state_dir=str(prefill_state_dir),
                        ),
                        decode_endpoint=SlotStateEndpointConfig(
                            server_url=decode_server.url,
                            state_dir=str(decode_state_dir),
                        ),
                        prompt="The capital of France is",
                        decode_tokens=3,
                        timeout_sec=10,
                        use_http_forwarding=True,
                    )
                )
            finally:
                prefill_server.close()
                decode_server.close()

    assert report["status"] == "ok"
    assert report["httpForwarding"] is True
    assert report["receivedEnvelopeCount"] == 2
    assert report["receivedShardReceiptCount"] == 2
    assert report["proofVerified"] is True
    assert [
        item["outputForwardStatus"] for item in report["workerRuns"]
    ] == ["submitted", "submitted"]
    assert report["finalPayloadUtf8"] == " Paris"
    assert decode_server.restored_bytes == b"real-llama-slot-state"


def test_slot_state_engine_works_behind_native_bridge_endpoint() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        state_dir = Path(tempdir)
        gguf_path = state_dir / "model.gguf"
        gguf_path.write_bytes(b"fake-gguf")
        server = _FakeLlamaServer(state_dir).start()
        bridge = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            _handler_class(
                native_command=[
                    sys.executable,
                    "-m",
                    "cai_compute_chain.cai_llama_cpp_slot_state_engine",
                    "--server-url",
                    server.url,
                    "--state-dir",
                    str(state_dir),
                    "--decode-tokens",
                    "3",
                ],
                timeout_sec=10,
            ),
        )
        thread = threading.Thread(target=bridge.serve_forever, daemon=True)
        thread.start()
        try:
            endpoint_url = f"http://127.0.0.1:{bridge.server_address[1]}/cai-shard"
            result = run_cai_owned_llm_shard_adapter_self_test(
                ExternalLlamaCppShardAdapter(
                    endpoint_url=endpoint_url,
                    timeout_sec=10,
                    shard_artifact_hint={
                        "modelArtifactPath": str(gguf_path.resolve()),
                        "artifactId": "gguf-main",
                    },
                ),
                payload=b"The capital of France is",
            )
        finally:
            bridge.shutdown()
            bridge.server_close()
            thread.join(timeout=2)
            server.close()

    assert result["contractReady"] is True
    assert result["generationProbeReady"] is True
    assert result["productionReady"] is False
    assert result["backendMode"] == SLOT_STATE_ENGINE_ID

# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import tempfile
import threading
from collections.abc import Mapping, Sequence
from typing import Any

from .cai_owned_runtime import (
    LLAMA_CPP_EXTERNAL_SHARD_ACTIVATION_BOUNDARY,
    LLAMA_CPP_EXTERNAL_SHARD_DECODE_STATE_BOUNDARY,
    LLAMA_CPP_EXTERNAL_SHARD_HANDOFF_ABI,
    LLAMA_CPP_EXTERNAL_SHARD_GENERATION_PROBE_ABI,
    LLAMA_CPP_EXTERNAL_SHARD_GENERATION_PROBE_ACTION,
    LLAMA_CPP_EXTERNAL_SHARD_PATCH_BOUNDARY_ABI,
    LLAMA_CPP_EXTERNAL_SHARD_SPEC_ABI,
    LLAMA_CPP_EXTERNAL_SHARD_PRODUCTION_CAPABILITIES,
    LLAMA_CPP_EXTERNAL_SHARD_PRODUCTION_STATE_CONTRACT_ABI,
    LLAMA_CPP_EXTERNAL_SHARD_REQUIRED_CAPABILITIES,
    LLAMA_CPP_EXTERNAL_SHARD_SUPPORTED_TENSOR_ENCODINGS,
    validate_llama_cpp_external_shard_spec,
)
from .cai_llama_cpp_native_engine_contract import (
    NATIVE_EXECUTION_SCHEMA_VERSION,
)
from .cai_llama_cpp_backend_runtime import (
    default_llama_cpp_runtime_root,
    prepare_llama_cpp_session_paths,
    resolve_llama_cpp_binary_set,
    resolve_llama_cpp_repo_root,
)
from .decentralized_compute import validate_cai_owned_transport_frame_metadata
from .model import curated_model_for_id
from .model_distribution import (
    ChunkCacheClass,
    ModelPackageKind,
    ModelShardAssignment,
    build_gguf_model_package_manifest,
    get_cached_chunk_record,
    load_local_artifact_bindings,
    load_model_package_manifest,
    materialize_default_assignment_artifact_from_store,
    materialized_artifact_path,
    put_cached_chunk,
    save_local_artifact_binding,
    save_model_package_manifest,
    select_default_materialized_artifact_id,
    select_model_package_manifest_for_model,
)


CAI_LLM_SHARD_NATIVE_COMMAND_ENV = "CAI_LLM_SHARD_NATIVE_COMMAND"
CAI_LLM_SHARD_NATIVE_TIMEOUT_ENV = "CAI_LLM_SHARD_NATIVE_TIMEOUT_SEC"
CAI_LLM_SHARD_NATIVE_PERSISTENT_ENV = "CAI_LLM_SHARD_NATIVE_PERSISTENT"
CAI_LLM_SHARD_RUNTIME_ROOT_ENV = "CAI_LLM_SHARD_RUNTIME_ROOT"
CAI_LLM_SHARD_MODEL_ARTIFACT_PATH_ENV = "CAI_LLM_SHARD_MODEL_ARTIFACT_PATH"
MANAGED_RUNTIME_ABI = "cai-llama-cpp-managed-runtime-v1"
EXECUTION_WORKSPACE_ABI = "cai-llama-cpp-execution-workspace-v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a local CAI LLM shard bridge for a native patched backend.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9258)
    parser.add_argument(
        "--oneshot",
        action="store_true",
        help="Read one CAI shard JSON request from stdin and return one JSON response.",
    )
    parser.add_argument(
        "--native-command",
        default=os.getenv(CAI_LLM_SHARD_NATIVE_COMMAND_ENV, ""),
        help="Command that reads one CAI shard JSON request from stdin.",
    )
    parser.add_argument(
        "--persistent-engine",
        action="store_true",
        default=_env_bool(CAI_LLM_SHARD_NATIVE_PERSISTENT_ENV, False),
        help=(
            "Keep native command alive and exchange one JSON request/response "
            "per line over stdin/stdout."
        ),
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=float(os.getenv(CAI_LLM_SHARD_NATIVE_TIMEOUT_ENV, "120") or 120),
    )
    args = parser.parse_args(argv)
    command = _split_command(args.native_command)
    timeout_sec = max(0.1, float(args.timeout_sec or 120.0))
    if bool(args.oneshot):
        stdin_buffer = getattr(sys.stdin, "buffer", None)
        raw_body = (
            stdin_buffer.read()
            if stdin_buffer is not None
            else str(sys.stdin.read() or "").encode("utf-8")
        )
        _status_code, payload = handle_native_bridge_request_body(
            raw_body,
            native_command=command,
            timeout_sec=timeout_sec,
        )
        print(json.dumps(payload, sort_keys=True), flush=True)
        return 0
    persistent_engine = (
        PersistentNativeEngineClient(command, timeout_sec=timeout_sec)
        if bool(args.persistent_engine) and command
        else None
    )
    handler = _handler_class(
        native_command=command,
        timeout_sec=timeout_sec,
        persistent_engine=persistent_engine,
    )
    server = ThreadingHTTPServer(
        (str(args.host or "127.0.0.1"), int(args.port or 9258)),
        handler,
    )
    print(
        json.dumps(
            {
                "status": "listening",
                "host": server.server_address[0],
                "port": server.server_address[1],
                "endpoint": (
                    f"http://{server.server_address[0]}:"
                    f"{server.server_address[1]}/cai-shard"
                ),
                "nativeCommandConfigured": bool(command),
                "nativeEngineMode": (
                    "persistent_jsonl"
                    if persistent_engine is not None
                    else "subprocess_per_request"
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
        if persistent_engine is not None:
            persistent_engine.close()
    return 0


def handle_native_bridge_request_body(
    raw_body: bytes,
    *,
    native_command: Sequence[str] | None = None,
    timeout_sec: float = 120.0,
    env: Mapping[str, str] | None = None,
    persistent_engine: "PersistentNativeEngineClient | None" = None,
) -> tuple[int, dict[str, Any]]:
    try:
        request = json.loads(bytes(raw_body or b"").decode("utf-8") or "{}")
        if not isinstance(request, dict):
            raise ValueError("Shard adapter request must be an object.")
        _validate_native_bridge_request(request)
        engine_request = _build_native_engine_request(request)
    except Exception as exc:
        return 400, {"status": "error", "error": str(exc)}

    command = [str(item) for item in (native_command or []) if str(item).strip()]
    if not command:
        return (
            200,
            {
                "status": "error",
                "error": "CAI LLM shard native engine command is not configured.",
                "metrics": {"backendMode": "native_bridge_missing_engine"},
            },
        )
    if persistent_engine is not None:
        return persistent_engine.call(engine_request)
    return _call_native_engine(
        command,
        engine_request,
        timeout_sec=max(0.1, float(timeout_sec or 120.0)),
        env=env,
    )


def _handler_class(
    *,
    native_command: Sequence[str],
    timeout_sec: float,
    persistent_engine: "PersistentNativeEngineClient | None" = None,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path not in {"/health", "/v1/health"}:
                self._send_json(
                    404,
                    {
                        "status": "error",
                        "error": "CAI LLM shard native bridge endpoint is unknown.",
                    },
                )
                return
            status_code, payload = handle_native_bridge_health(
                native_command=native_command,
                timeout_sec=timeout_sec,
                persistent_engine=persistent_engine,
            )
            self._send_json(status_code, payload)

        def do_POST(self) -> None:  # noqa: N802
            raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            status_code, payload = handle_native_bridge_request_body(
                raw_body,
                native_command=native_command,
                timeout_sec=timeout_sec,
                persistent_engine=persistent_engine,
            )
            self._send_json(status_code, payload)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return Handler


def handle_native_bridge_health(
    *,
    native_command: Sequence[str] | None = None,
    timeout_sec: float = 120.0,
    persistent_engine: "PersistentNativeEngineClient | None" = None,
) -> tuple[int, dict[str, Any]]:
    command = [str(item) for item in (native_command or []) if str(item).strip()]
    mode = (
        "persistent_jsonl"
        if persistent_engine is not None
        else "subprocess_per_request"
    )
    payload: dict[str, Any] = {
        "schemaVersion": 1,
        "status": "ok" if command else "degraded",
        "bridge": "cai_llm_shard_native_bridge",
        "nativeCommandConfigured": bool(command),
        "nativeEngineMode": mode,
        "timeoutSec": max(0.1, float(timeout_sec or 120.0)),
        "endpoint": "/cai-shard",
    }
    if persistent_engine is not None:
        persistent_health = persistent_engine.health()
        payload["persistentEngine"] = persistent_health
        if not bool(persistent_health.get("alive")):
            payload["status"] = "degraded"
    if not command:
        payload["error"] = "CAI LLM shard native engine command is not configured."
    return (200 if payload["status"] == "ok" else 503), payload


class PersistentNativeEngineClient:
    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout_sec: float,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.command = [str(item) for item in command if str(item).strip()]
        self.timeout_sec = max(0.1, float(timeout_sec or 120.0))
        self.env = {str(k): str(v) for k, v in (env or {}).items()}
        self._lock = threading.Lock()
        self._stdout_lines: queue.Queue[str] = queue.Queue()
        self._stderr_tail: list[str] = []
        self._closed = False
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={**os.environ, **self.env},
        )
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    def call(self, request: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        with self._lock:
            if self._closed:
                return self._error("native_bridge_persistent_closed")
            if self._process.poll() is not None:
                return self._error("native_bridge_persistent_exited")
            if self._process.stdin is None:
                return self._error("native_bridge_persistent_stdin_missing")
            try:
                self._process.stdin.write(json.dumps(dict(request), sort_keys=True))
                self._process.stdin.write("\n")
                self._process.stdin.flush()
                response_text = self._stdout_lines.get(timeout=self.timeout_sec)
            except queue.Empty:
                self.close(kill=True)
                return self._error("native_bridge_persistent_timeout")
            except Exception as exc:
                return self._error("native_bridge_persistent_io_error", str(exc))
            return _parse_native_engine_response(
                response_text,
                invalid_mode="native_bridge_persistent_invalid_json",
                invalid_response_mode="native_bridge_persistent_invalid_response",
                request=request,
            )

    def close(self, *, kill: bool = False) -> None:
        self._closed = True
        process = self._process
        if process.poll() is not None:
            return
        try:
            if kill:
                process.kill()
            else:
                process.terminate()
            process.wait(timeout=2)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def health(self) -> dict[str, Any]:
        return_code = self._process.poll()
        alive = not self._closed and return_code is None
        payload: dict[str, Any] = {
            "mode": "persistent_jsonl",
            "alive": alive,
            "pid": self._process.pid,
            "closed": self._closed,
            "returnCode": return_code,
            "stderrTail": list(self._stderr_tail),
        }
        if not alive:
            payload["error"] = "CAI LLM shard persistent native engine is not alive."
        return payload

    def _read_stdout(self) -> None:
        stream = self._process.stdout
        if stream is None:
            return
        for line in stream:
            self._stdout_lines.put(line)

    def _read_stderr(self) -> None:
        stream = self._process.stderr
        if stream is None:
            return
        for line in stream:
            clean = str(line or "").strip()
            if not clean:
                continue
            self._stderr_tail.append(clean)
            del self._stderr_tail[:-10]

    def _error(
        self,
        backend_mode: str,
        detail: str | None = None,
    ) -> tuple[int, dict[str, Any]]:
        stderr_tail = "\n".join(self._stderr_tail).strip()
        error = "CAI LLM shard persistent native engine failed."
        if detail:
            error += f" {detail[:500]}"
        elif stderr_tail:
            error += f" {stderr_tail[:500]}"
        return (
            200,
            {
                "status": "error",
                "error": error,
                "metrics": {"backendMode": backend_mode},
            },
        )


def _validate_native_bridge_request(request: Mapping[str, Any]) -> None:
    if str(request.get("abi") or "").strip() != LLAMA_CPP_EXTERNAL_SHARD_HANDOFF_ABI:
        raise ValueError("CAI LLM shard native bridge request ABI is unsupported.")
    action = str(request.get("action") or "").strip()
    if action not in {
        "load_shard",
        "process_prefill",
        "process_decode",
        "finalize",
        LLAMA_CPP_EXTERNAL_SHARD_GENERATION_PROBE_ACTION,
    }:
        raise ValueError("CAI LLM shard native bridge action is unsupported.")
    _validate_production_requirements(request.get("productionRequirements"))
    if action == LLAMA_CPP_EXTERNAL_SHARD_GENERATION_PROBE_ACTION:
        _validate_generation_probe_request(request.get("generationProbe"))
        return
    frame = request.get("frame")
    if not isinstance(frame, Mapping):
        raise ValueError("CAI LLM shard native bridge frame is missing.")
    shard_spec_valid, shard_spec_error = validate_llama_cpp_external_shard_spec(
        request.get("shardSpec"),
        expected_model_id=str(frame.get("modelId") or "").strip() or None,
        expected_frame=frame,
    )
    if not shard_spec_valid:
        raise ValueError(
            shard_spec_error or "CAI LLM shard native bridge shardSpec is invalid."
        )
    if action in {"process_prefill", "process_decode"}:
        metadata = frame.get("metadata")
        valid, error = validate_cai_owned_transport_frame_metadata(
            metadata if isinstance(metadata, Mapping) else None,
            expected_model_id=str(frame.get("modelId") or "").strip() or None,
            require_llm_handoff=True,
        )
        if not valid:
            raise ValueError(
                error or "CAI LLM shard native bridge frame metadata is invalid."
            )


def _validate_generation_probe_request(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("CAI LLM shard generationProbe is missing.")
    if str(value.get("abi") or "").strip() != LLAMA_CPP_EXTERNAL_SHARD_GENERATION_PROBE_ABI:
        raise ValueError("CAI LLM shard generationProbe ABI is unsupported.")
    if not str(value.get("modelId") or "").strip():
        raise ValueError("CAI LLM shard generationProbe modelId is missing.")
    if not str(value.get("prompt") or "").strip():
        raise ValueError("CAI LLM shard generationProbe prompt is missing.")
    if not bool(value.get("requiresRealModelExecution")):
        raise ValueError(
            "CAI LLM shard generationProbe must require real model execution."
        )


def _validate_production_requirements(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("CAI LLM shard productionRequirements are missing.")
    required_fields = {
        "handoffAbi": LLAMA_CPP_EXTERNAL_SHARD_HANDOFF_ABI,
        "shardSpecAbi": LLAMA_CPP_EXTERNAL_SHARD_SPEC_ABI,
        "patchBoundaryAbi": LLAMA_CPP_EXTERNAL_SHARD_PATCH_BOUNDARY_ABI,
        "productionStateContractAbi": (
            LLAMA_CPP_EXTERNAL_SHARD_PRODUCTION_STATE_CONTRACT_ABI
        ),
        "activationBoundary": LLAMA_CPP_EXTERNAL_SHARD_ACTIVATION_BOUNDARY,
        "decodeStateBoundary": LLAMA_CPP_EXTERNAL_SHARD_DECODE_STATE_BOUNDARY,
    }
    for field_name, expected in required_fields.items():
        if str(value.get(field_name) or "").strip() != expected:
            raise ValueError(
                f"CAI LLM shard productionRequirements {field_name} mismatch."
            )
    _require_all(
        value.get("requiredCapabilities"),
        LLAMA_CPP_EXTERNAL_SHARD_REQUIRED_CAPABILITIES,
        "requiredCapabilities",
    )
    _require_all(
        value.get("requiredProductionCapabilities"),
        LLAMA_CPP_EXTERNAL_SHARD_PRODUCTION_CAPABILITIES,
        "requiredProductionCapabilities",
    )
    _require_all(
        value.get("supportedTensorEncodings"),
        LLAMA_CPP_EXTERNAL_SHARD_SUPPORTED_TENSOR_ENCODINGS,
        "supportedTensorEncodings",
    )
    if not bool(value.get("requiresRealStateContract")):
        raise ValueError(
            "CAI LLM shard productionRequirements must require real state contract."
        )
    if not bool(value.get("requiresShardOnlyLoading")):
        raise ValueError(
            "CAI LLM shard productionRequirements must require shard-only loading."
        )
    if not bool(value.get("forbidFullModelFallback")):
        raise ValueError(
            "CAI LLM shard productionRequirements must forbid full-model fallback."
        )


def _build_native_engine_request(request: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(request)
    local_artifact_resolution = _resolve_local_artifact_resolution(request)
    if local_artifact_resolution is not None:
        payload["localArtifactResolution"] = local_artifact_resolution
    managed_runtime = _build_managed_runtime_payload(request)
    if managed_runtime is not None:
        payload["managedRuntime"] = managed_runtime
        execution_workspace = _build_execution_workspace_payload(
            request,
            managed_runtime=managed_runtime,
        )
        if execution_workspace is not None:
            payload["executionWorkspace"] = execution_workspace
    return payload


def _build_managed_runtime_payload(
    request: Mapping[str, Any],
) -> dict[str, Any] | None:
    repo_root = resolve_llama_cpp_repo_root()
    binary_set = resolve_llama_cpp_binary_set(repo_root=repo_root)
    runtime_root = _managed_runtime_root(repo_root)
    action = str(request.get("action") or "").strip()
    model_id = _request_model_id(request)
    if not model_id and action == LLAMA_CPP_EXTERNAL_SHARD_GENERATION_PROBE_ACTION:
        generation_probe = request.get("generationProbe")
        if isinstance(generation_probe, Mapping):
            model_id = str(generation_probe.get("modelId") or "").strip() or None
    session_paths_payload: dict[str, Any] | None = None
    frame = request.get("frame")
    if isinstance(frame, Mapping):
        session_id = str(frame.get("sessionId") or "").strip() or f"bridge-{action or 'session'}"
        session_paths = prepare_llama_cpp_session_paths(
            base_root=runtime_root,
            session_id=session_id,
            model_id=model_id or "unknown-model",
            layer_start=_request_layer_bound(request, "layerStart"),
            layer_end=_request_layer_bound(request, "layerEnd"),
        )
        session_paths_payload = {
            "root": str(session_paths.root),
            "stateDir": str(session_paths.state_dir),
            "cacheDir": str(session_paths.cache_dir),
            "logsDir": str(session_paths.logs_dir),
            "stdoutLog": str(session_paths.stdout_log),
            "stderrLog": str(session_paths.stderr_log),
        }
    binaries_payload: dict[str, Any] = {}
    if binary_set.llama_server is not None:
        binaries_payload["llamaServerPath"] = str(binary_set.llama_server)
    if binary_set.rpc_server is not None:
        binaries_payload["rpcServerPath"] = str(binary_set.rpc_server)
    payload: dict[str, Any] = {
        "schemaVersion": 1,
        "abi": MANAGED_RUNTIME_ABI,
        "platform": os.name,
        "repoRoot": str(repo_root),
        "runtimeRoot": str(runtime_root),
    }
    if model_id:
        payload["modelId"] = model_id
    if binaries_payload:
        payload["llamaCpp"] = binaries_payload
    if session_paths_payload is not None:
        payload["sessionPaths"] = session_paths_payload
    return payload


def _build_execution_workspace_payload(
    request: Mapping[str, Any],
    *,
    managed_runtime: Mapping[str, Any],
) -> dict[str, Any] | None:
    session_paths = managed_runtime.get("sessionPaths")
    if not isinstance(session_paths, Mapping):
        return None
    state_dir = str(session_paths.get("stateDir") or "").strip()
    if not state_dir:
        return None
    root = (Path(state_dir).expanduser().resolve() / "llm-shard-execution").resolve()
    inputs_dir = (root / "inputs").resolve()
    outputs_dir = (root / "outputs").resolve()
    state_files_dir = (root / "state").resolve()
    for path in (root, inputs_dir, outputs_dir, state_files_dir):
        path.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schemaVersion": 1,
        "abi": EXECUTION_WORKSPACE_ABI,
        "root": str(root),
        "inputsDir": str(inputs_dir),
        "outputsDir": str(outputs_dir),
        "stateFilesDir": str(state_files_dir),
        "manifestPath": str((root / "execution-workspace.json").resolve()),
        "action": str(request.get("action") or "").strip(),
        "modelId": _request_model_id(request),
        "layerStart": _request_layer_bound(request, "layerStart"),
        "layerEnd": _request_layer_bound(request, "layerEnd"),
        "tokenStart": _request_layer_bound(request, "tokenStart"),
        "tokenEnd": _request_layer_bound(request, "tokenEnd"),
        "requiresFinalOutput": _request_requires_final_output(request),
        "expectedOutputKind": _expected_output_kind(request),
    }
    frame = request.get("frame")
    if isinstance(frame, Mapping):
        session_id = str(frame.get("sessionId") or "").strip()
        if session_id:
            payload["sessionId"] = session_id
    return payload


def _managed_runtime_root(repo_root: Path) -> Path:
    configured = str(os.getenv(CAI_LLM_SHARD_RUNTIME_ROOT_ENV) or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    default_root = default_llama_cpp_runtime_root(repo_root=repo_root)
    try:
        default_root.mkdir(parents=True, exist_ok=True)
        return default_root.resolve()
    except Exception:
        fallback_root = (
            Path(tempfile.gettempdir()).resolve() / "cai-llama-shard-runtime"
        )
        fallback_root.mkdir(parents=True, exist_ok=True)
        return fallback_root.resolve()


def _resolve_local_artifact_resolution(
    request: Mapping[str, Any],
) -> dict[str, Any] | None:
    action = str(request.get("action") or "").strip()
    if action == LLAMA_CPP_EXTERNAL_SHARD_GENERATION_PROBE_ACTION:
        generation_probe = request.get("generationProbe")
        model_id = (
            str(generation_probe.get("modelId") or "").strip()
            if isinstance(generation_probe, Mapping)
            else ""
        )
        return _resolve_local_artifact_resolution_for_model(model_id)
    shard_spec = request.get("shardSpec")
    frame = request.get("frame")
    model_id = ""
    artifact_hint: Mapping[str, Any] | None = None
    if isinstance(shard_spec, Mapping):
        model_id = str(shard_spec.get("modelId") or "").strip()
        raw_artifact_hint = shard_spec.get("artifactHint")
        if isinstance(raw_artifact_hint, Mapping):
            artifact_hint = raw_artifact_hint
    layer_start = _optional_non_negative_int(
        shard_spec.get("layerStart") if isinstance(shard_spec, Mapping) else None
    )
    layer_end = _optional_non_negative_int(
        shard_spec.get("layerEnd") if isinstance(shard_spec, Mapping) else None
    )
    if layer_start is None and isinstance(frame, Mapping):
        layer_start = _optional_non_negative_int(frame.get("layerStart"))
    if layer_end is None and isinstance(frame, Mapping):
        layer_end = _optional_non_negative_int(frame.get("layerEnd"))
    if not model_id and isinstance(frame, Mapping):
        model_id = str(frame.get("modelId") or "").strip()
    return _resolve_local_artifact_resolution_for_model(
        model_id,
        artifact_hint=artifact_hint,
        layer_start=layer_start,
        layer_end=layer_end,
    )


def _resolve_local_artifact_resolution_for_model(
    model_id: str,
    *,
    artifact_hint: Mapping[str, Any] | None = None,
    layer_start: int | None = None,
    layer_end: int | None = None,
) -> dict[str, Any] | None:
    clean_model_id = str(model_id or "").strip()
    if not clean_model_id:
        return None
    hint = artifact_hint if isinstance(artifact_hint, Mapping) else None
    catalog_id = str(hint.get("catalogId") or "").strip() if hint else ""
    version = str(hint.get("version") or "").strip() if hint else ""
    artifact_id = str(hint.get("artifactId") or "").strip() if hint else ""
    preferred_filename = (
        str(hint.get("preferredFilename") or "").strip() if hint else ""
    )
    explicit_model_artifact_path = (
        str(hint.get("modelArtifactPath") or "").strip() if hint else ""
    )
    explicit_hint = bool(
        hint
        and any(
            str(hint.get(field_name) or "").strip()
            for field_name in (
                "catalogId",
                "version",
                "artifactId",
                "modelArtifactPath",
                "assignmentArtifactPath",
            )
        )
    )
    manifest = None
    if catalog_id and version and not explicit_model_artifact_path:
        try:
            manifest = load_model_package_manifest(catalog_id, version)
        except FileNotFoundError as exc:
            raise ValueError(
                "CAI LLM shard local artifact manifest is missing: "
                f"{catalog_id}@{version}"
            ) from exc
    else:
        manifest = _select_or_prepare_local_model_package_manifest(
            clean_model_id,
            artifact_hint=hint,
            preferred_filename=preferred_filename,
            explicit_model_artifact_path=explicit_model_artifact_path,
        )
        if manifest is not None:
            catalog_id = str(manifest.catalog_id or "").strip()
            version = str(manifest.version or "").strip()
    if manifest is not None and not artifact_id:
        artifact_id = str(select_default_materialized_artifact_id(manifest) or "").strip()
        if not preferred_filename:
            preferred_filename = str(manifest.preferred_filename or "").strip()
    assignment_artifact = _resolve_assignment_artifact(
        hint,
        manifest=manifest,
        artifact_id=artifact_id,
        layer_start=layer_start,
        layer_end=layer_end,
    )
    model_artifact = _resolve_local_model_artifact(
        manifest=manifest,
        artifact_id=artifact_id,
        artifact_hint=hint,
        explicit_hint=explicit_hint and assignment_artifact is None,
    )
    if model_artifact is None and assignment_artifact is None:
        return None
    payload: dict[str, Any] = {
        "schemaVersion": 1,
        "modelId": clean_model_id,
    }
    if catalog_id:
        payload["catalogId"] = catalog_id
    if version:
        payload["version"] = version
    if preferred_filename:
        payload["preferredFilename"] = preferred_filename
    if manifest is not None:
        payload["manifestBackend"] = str(manifest.backend or "").strip()
    if model_artifact is not None:
        payload["modelArtifact"] = model_artifact
    if assignment_artifact is not None:
        payload["assignmentArtifact"] = assignment_artifact
    return payload


def _select_or_prepare_local_model_package_manifest(
    model_id: str,
    *,
    artifact_hint: Mapping[str, Any] | None,
    preferred_filename: str,
    explicit_model_artifact_path: str,
) -> Any:
    clean_model_id = str(model_id or "").strip()
    model_ids = _local_model_package_manifest_model_ids(clean_model_id)
    existing_manifest: Any = None
    for candidate_model_id in model_ids:
        manifest = select_model_package_manifest_for_model(candidate_model_id)
        if manifest is None:
            continue
        if existing_manifest is None:
            existing_manifest = manifest
        if _manifest_has_local_model_artifact(manifest):
            return manifest

    local_path = _resolve_local_gguf_model_artifact_path(
        clean_model_id,
        artifact_hint=artifact_hint,
        preferred_filename=preferred_filename,
        explicit_model_artifact_path=explicit_model_artifact_path,
    )
    if local_path is not None and existing_manifest is not None:
        artifact_id = str(select_default_materialized_artifact_id(existing_manifest) or "")
        if artifact_id:
            save_local_artifact_binding(
                existing_manifest.catalog_id,
                existing_manifest.version,
                artifact_id=artifact_id,
                local_path=local_path,
            )
        return existing_manifest
    if local_path is None:
        return existing_manifest

    curated = curated_model_for_id(clean_model_id)
    stat = local_path.stat()
    catalog_id = _local_model_package_catalog_id(clean_model_id)
    version = f"local-{int(stat.st_size)}-{int(stat.st_mtime)}"
    manifest = build_gguf_model_package_manifest(
        catalog_id=catalog_id,
        model_id=clean_model_id,
        version=version,
        gguf_path=local_path,
        total_layers=(
            int(getattr(curated, "total_layers", 0) or 0)
            if curated is not None
            else None
        ),
        package_kind=(
            ModelPackageKind.PRIVATE_CURATED
            if bool(getattr(curated, "private_network", False))
            else ModelPackageKind.PUBLIC_SHARED
        ),
        target_chunk_count=_env_positive_int(
            "CAI_LLM_SHARD_MODEL_PACKAGE_TARGET_CHUNK_COUNT",
            4,
        ),
        source_repo_id=(
            str(getattr(curated, "source_repo_id", "") or "").strip()
            if curated is not None
            else clean_model_id
        ),
        family=str(getattr(curated, "gguf_architecture", "") or "").strip(),
        quantization=str(getattr(curated, "quantization", "") or "").strip(),
    )
    save_model_package_manifest(manifest)
    save_local_artifact_binding(
        manifest.catalog_id,
        manifest.version,
        artifact_id=str(select_default_materialized_artifact_id(manifest) or "gguf-main"),
        local_path=local_path,
    )
    return manifest


def _local_model_package_manifest_model_ids(model_id: str) -> list[str]:
    clean_model_id = str(model_id or "").strip()
    candidates: list[str] = []

    def add(value: object) -> None:
        clean = str(value or "").strip()
        if clean and clean not in candidates:
            candidates.append(clean)

    add(clean_model_id)
    curated = curated_model_for_id(clean_model_id)
    if curated is not None:
        add(getattr(curated, "model_id", ""))
        add(getattr(curated, "execution_model_id", ""))
        for item in getattr(curated, "runtime_model_ids", ()) or ():
            add(item)
    if clean_model_id in {
        "Qwen/Qwen3-0.6B-GGUF",
        "cai-network/Qwen3-0.6B-GGUF",
    }:
        add("Qwen/Qwen3-0.6B-GGUF")
        add("cai-network/Qwen3-0.6B-GGUF")
    return candidates


def _manifest_has_local_model_artifact(manifest: Any) -> bool:
    artifact_id = str(select_default_materialized_artifact_id(manifest) or "").strip()
    if not artifact_id:
        return False
    for binding in load_local_artifact_bindings(
        str(manifest.catalog_id),
        str(manifest.version),
    ).bindings:
        if binding.artifact_id != artifact_id:
            continue
        path = Path(binding.local_path).expanduser()
        if path.exists() and path.is_file():
            return True
    candidate = materialized_artifact_path(manifest, artifact_id)
    return candidate.exists() and candidate.is_file()


def _resolve_local_gguf_model_artifact_path(
    model_id: str,
    *,
    artifact_hint: Mapping[str, Any] | None,
    preferred_filename: str,
    explicit_model_artifact_path: str,
) -> Path | None:
    if explicit_model_artifact_path:
        return _require_local_file(
            explicit_model_artifact_path,
            field_name="CAI LLM shard artifactHint modelArtifactPath",
        )
    for candidate in _candidate_local_gguf_model_artifact_paths(
        model_id,
        artifact_hint=artifact_hint,
        preferred_filename=preferred_filename,
    ):
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def _candidate_local_gguf_model_artifact_paths(
    model_id: str,
    *,
    artifact_hint: Mapping[str, Any] | None,
    preferred_filename: str,
) -> list[Path]:
    clean_model_id = str(model_id or "").strip()
    curated = curated_model_for_id(clean_model_id)
    filename = (
        str(preferred_filename or "").strip()
        or (
            str(artifact_hint.get("preferredFilename") or "").strip()
            if isinstance(artifact_hint, Mapping)
            else ""
        )
        or str(getattr(curated, "preferred_filename", "") or "").strip()
    )
    if not filename:
        return []

    candidates: list[Path] = []

    def add(raw: object) -> None:
        text = str(raw or "").strip()
        if not text:
            return
        path = Path(text).expanduser()
        if path.is_dir():
            path = path / filename
        try:
            resolved = path.resolve()
        except OSError:
            return
        if resolved not in candidates:
            candidates.append(resolved)

    for env_name in (
        CAI_LLM_SHARD_MODEL_ARTIFACT_PATH_ENV,
        "CAI_MODEL_ARTIFACT_PATH",
        "CAI_QWEN3_GGUF_PATH",
    ):
        for item in str(os.getenv(env_name) or "").split(os.pathsep):
            add(item)

    roots: list[Path] = []
    try:
        roots.append(resolve_llama_cpp_repo_root())
    except Exception:
        pass
    roots.append(Path.cwd())
    for env_name in ("CAI_REPO_ROOT", "CAI_RUNTIME_REPO", "CAI_WALLET_HOME"):
        raw = str(os.getenv(env_name) or "").strip()
        if raw:
            roots.append(Path(raw).expanduser())

    model_segments = _candidate_model_cache_segments(clean_model_id, curated)
    for root in roots:
        for base in (
            root / "models",
            root / ".cai" / "models",
            root / ".cai-local" / "models",
            root / "data" / ".cai" / "models",
        ):
            add(base / filename)
            for segment in model_segments:
                add(base / segment / filename)
                add(base / "caches" / segment / filename)

    home = Path.home()
    for segment in model_segments:
        snapshots = home / ".cache" / "huggingface" / "hub" / f"models--{segment}" / "snapshots"
        if snapshots.exists():
            for snapshot_dir in snapshots.iterdir():
                add(snapshot_dir / filename)
    return candidates


def _candidate_model_cache_segments(model_id: str, curated: Any) -> list[str]:
    values: list[str] = []

    def add(value: object) -> None:
        clean = str(value or "").strip()
        if not clean:
            return
        segment = clean.replace("/", "--")
        if segment not in values:
            values.append(segment)

    add(model_id)
    if curated is not None:
        add(getattr(curated, "model_id", ""))
        add(getattr(curated, "execution_model_id", ""))
        add(getattr(curated, "source_repo_id", ""))
        for item in getattr(curated, "runtime_model_ids", ()) or ():
            add(item)
    if model_id in {"Qwen/Qwen3-0.6B-GGUF", "cai-network/Qwen3-0.6B-GGUF"}:
        add("Qwen/Qwen3-0.6B-GGUF")
        add("cai-network/Qwen3-0.6B-GGUF")
    return values


def _local_model_package_catalog_id(model_id: str) -> str:
    safe = "".join(
        ch.lower() if ch.isalnum() else "-"
        for ch in str(model_id or "").strip()
    ).strip("-")
    while "--" in safe:
        safe = safe.replace("--", "-")
    return f"cai-local-{safe or 'gguf-model'}"


def _env_positive_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default)) or default))
    except ValueError:
        return max(1, int(default))


def _resolve_local_model_artifact(
    *,
    manifest: Any,
    artifact_id: str,
    artifact_hint: Mapping[str, Any] | None,
    explicit_hint: bool,
) -> dict[str, Any] | None:
    hint = artifact_hint if isinstance(artifact_hint, Mapping) else None
    explicit_path = str(hint.get("modelArtifactPath") or "").strip() if hint else ""
    expected_sha = (
        str(hint.get("modelArtifactSha256Hex") or "").strip().lower() if hint else ""
    )
    expected_size = _optional_positive_int(
        hint.get("modelArtifactSizeBytes") if hint else None
    )
    resolved_path: Path | None = None
    source = ""
    if explicit_path:
        resolved_path = _require_local_file(
            explicit_path,
            field_name="CAI LLM shard artifactHint modelArtifactPath",
        )
        source = "artifact_hint_path"
    elif manifest is not None and artifact_id:
        bindings = load_local_artifact_bindings(
            str(manifest.catalog_id),
            str(manifest.version),
        )
        binding = next(
            (item for item in bindings.bindings if item.artifact_id == artifact_id),
            None,
        )
        if binding is not None:
            resolved_path = _require_local_file(
                binding.local_path,
                field_name="CAI LLM shard local artifact binding",
            )
            source = "local_binding"
        else:
            candidate = materialized_artifact_path(
                manifest,
                artifact_id,
            )
            if candidate.exists() and candidate.is_file():
                resolved_path = candidate.resolve()
                source = "materialized_artifact"
    if resolved_path is None:
        if explicit_hint:
            raise ValueError(
                "CAI LLM shard local model artifact is not available for native bridge."
            )
        return None
    actual_size = int(resolved_path.stat().st_size)
    if expected_size is None and manifest is not None:
        manifest_artifact = next(
            (item for item in manifest.files if item.artifact_id == artifact_id),
            None,
        )
        if manifest_artifact is not None and int(manifest_artifact.size_bytes or 0) > 0:
            expected_size = int(manifest_artifact.size_bytes)
        if not expected_sha and manifest_artifact is not None:
            expected_sha = str(manifest_artifact.sha256_hex or "").strip().lower()
    if expected_size is not None and actual_size != expected_size:
        raise ValueError(
            "CAI LLM shard local model artifact size mismatch for native bridge."
        )
    payload: dict[str, Any] = {
        "artifactId": artifact_id or "gguf-main",
        "localPath": str(resolved_path),
        "source": source,
        "sizeBytes": actual_size,
    }
    if expected_size is not None:
        payload["expectedSizeBytes"] = expected_size
    if expected_sha:
        payload["expectedSha256Hex"] = expected_sha
    if hint is not None:
        preferred_filename = str(hint.get("preferredFilename") or "").strip()
        if preferred_filename:
            payload["preferredFilename"] = preferred_filename
    return payload


def _resolve_assignment_artifact(
    artifact_hint: Mapping[str, Any] | None,
    *,
    manifest: Any,
    artifact_id: str,
    layer_start: int | None,
    layer_end: int | None,
) -> dict[str, Any] | None:
    hint = artifact_hint if isinstance(artifact_hint, Mapping) else None
    if hint is None:
        assignment_path = ""
    else:
        assignment_path = str(hint.get("assignmentArtifactPath") or "").strip()
    chunk_ranges = _assignment_chunk_ranges(
        manifest=manifest,
        artifact_id=artifact_id,
        layer_start=layer_start,
        layer_end=layer_end,
    )
    if assignment_path:
        resolved_path = _require_local_file(
            assignment_path,
            field_name="CAI LLM shard artifactHint assignmentArtifactPath",
        )
        actual_size = int(resolved_path.stat().st_size)
        expected_size = _optional_positive_int(hint.get("assignmentArtifactSizeBytes"))
        if expected_size is not None and actual_size != expected_size:
            raise ValueError(
                "CAI LLM shard local assignment artifact size mismatch for native bridge."
            )
        payload: dict[str, Any] = {
            "localPath": str(resolved_path),
            "source": "artifact_hint_path",
            "sizeBytes": actual_size,
        }
        if artifact_id:
            payload["artifactId"] = artifact_id
        digest = str(hint.get("assignmentArtifactDigest") or "").strip()
        if digest:
            payload["expectedDigest"] = digest
        if expected_size is not None:
            payload["expectedSizeBytes"] = expected_size
        if layer_start is not None:
            payload["layerStart"] = layer_start
        if layer_end is not None:
            payload["layerEnd"] = layer_end
        if chunk_ranges:
            payload["chunkRanges"] = chunk_ranges
        return payload
    assignment = _build_local_assignment(
        layer_start=layer_start,
        layer_end=layer_end,
    )
    if manifest is None or not artifact_id or assignment is None:
        return None
    try:
        materialized = materialize_default_assignment_artifact_from_store(
            manifest,
            assignment,
            overwrite=False,
        )
    except (FileNotFoundError, IOError, OSError):
        if not _cache_assignment_chunks_from_local_artifact(
            manifest,
            artifact_id=artifact_id,
            layer_start=layer_start,
            layer_end=layer_end,
            artifact_hint=hint,
        ):
            return None
        try:
            materialized = materialize_default_assignment_artifact_from_store(
                manifest,
                assignment,
                overwrite=False,
            )
        except (FileNotFoundError, IOError, OSError):
            return None
    payload = {
        "artifactId": artifact_id,
        "localPath": str(Path(materialized.output_path).resolve()),
        "source": "materialized_assignment",
        "sizeBytes": int(materialized.size_bytes),
        "expectedDigest": str(materialized.sha256_hex),
        "layerStart": assignment.start_layer,
        "layerEnd": assignment.end_layer,
        "deviceRank": assignment.device_rank,
        "worldSize": assignment.world_size,
    }
    if chunk_ranges:
        payload["chunkRanges"] = chunk_ranges
        payload["coverage"] = _assignment_artifact_coverage_payload(
            artifact_size_bytes=int(materialized.size_bytes),
            chunk_ranges=chunk_ranges,
        )
    return payload


def _cache_assignment_chunks_from_local_artifact(
    manifest: Any,
    *,
    artifact_id: str,
    layer_start: int | None,
    layer_end: int | None,
    artifact_hint: Mapping[str, Any] | None,
) -> bool:
    if layer_start is None or layer_end is None or layer_end <= layer_start:
        return False
    local_artifact_path = _local_artifact_path_for_manifest(
        manifest,
        artifact_id=artifact_id,
        artifact_hint=artifact_hint,
    )
    if local_artifact_path is None:
        return False
    try:
        chunks = sorted(
            (
                chunk
                for chunk in manifest.required_chunks_for_layers(
                    layer_start,
                    layer_end,
                    include_default_chunks=True,
                )
                if str(chunk.artifact_id or "").strip() == artifact_id
            ),
            key=lambda chunk: int(chunk.offset_bytes),
        )
    except Exception:
        return False
    if not chunks:
        return False
    try:
        with local_artifact_path.open("rb") as handle:
            for chunk in chunks:
                if get_cached_chunk_record(chunk.chunk_id) is not None:
                    continue
                handle.seek(int(chunk.offset_bytes))
                payload = handle.read(int(chunk.size_bytes))
                if len(payload) != int(chunk.size_bytes):
                    return False
                if hashlib.sha256(payload).hexdigest() != str(chunk.sha256_hex):
                    return False
                put_cached_chunk(
                    catalog_id=str(manifest.catalog_id),
                    version=str(manifest.version),
                    chunk_id=str(chunk.chunk_id),
                    sha256_hex=str(chunk.sha256_hex),
                    content=payload,
                    pinned=True,
                    cache_class=ChunkCacheClass.HOT,
                )
    except Exception:
        return False
    return True


def _local_artifact_path_for_manifest(
    manifest: Any,
    *,
    artifact_id: str,
    artifact_hint: Mapping[str, Any] | None,
) -> Path | None:
    hint = artifact_hint if isinstance(artifact_hint, Mapping) else None
    explicit_path = str(hint.get("modelArtifactPath") or "").strip() if hint else ""
    if explicit_path:
        return _require_local_file(
            explicit_path,
            field_name="CAI LLM shard artifactHint modelArtifactPath",
        )
    for binding in load_local_artifact_bindings(
        str(manifest.catalog_id),
        str(manifest.version),
    ).bindings:
        if binding.artifact_id != artifact_id:
            continue
        path = Path(binding.local_path).expanduser().resolve()
        if path.exists() and path.is_file():
            return path
    return None


def _assignment_chunk_ranges(
    *,
    manifest: Any,
    artifact_id: str,
    layer_start: int | None,
    layer_end: int | None,
) -> list[dict[str, Any]] | None:
    if (
        manifest is None
        or not artifact_id
        or layer_start is None
        or layer_end is None
        or layer_start < 0
        or layer_end <= layer_start
    ):
        return None
    try:
        required_chunks = sorted(
            (
                chunk
                for chunk in manifest.required_chunks_for_layers(
                    layer_start,
                    layer_end,
                )
                if str(chunk.artifact_id or "").strip() == artifact_id
            ),
            key=lambda chunk: int(chunk.offset_bytes),
        )
    except Exception:
        return None
    if not required_chunks:
        return None
    output: list[dict[str, Any]] = []
    for chunk in required_chunks:
        item: dict[str, Any] = {
            "chunkId": str(chunk.chunk_id),
            "offsetBytes": int(chunk.offset_bytes),
            "sizeBytes": int(chunk.size_bytes),
            "sha256Hex": str(chunk.sha256_hex),
        }
        if chunk.layer_start is not None:
            item["layerStart"] = int(chunk.layer_start)
        if chunk.layer_end is not None:
            item["layerEnd"] = int(chunk.layer_end)
        if chunk.tensor_names:
            item["tensorNames"] = [str(name) for name in chunk.tensor_names if str(name).strip()]
        output.append(item)
    return output


def _assignment_artifact_coverage_payload(
    *,
    artifact_size_bytes: int,
    chunk_ranges: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    covered_byte_count = 0
    covered_range_count = 0
    for item in chunk_ranges:
        try:
            covered_byte_count += int(item.get("sizeBytes") or 0)
            covered_range_count += 1
        except (TypeError, ValueError):
            continue
    return {
        "abi": "cai-llama-cpp-assignment-coverage-v1",
        "materializationMode": "sparse_full_size",
        "artifactSizeBytes": int(max(0, artifact_size_bytes)),
        "coveredByteCount": int(max(0, covered_byte_count)),
        "coveredRangeCount": int(max(0, covered_range_count)),
        "zeroFilledOutsideCoveredRanges": True,
    }


def _build_local_assignment(
    *,
    layer_start: int | None,
    layer_end: int | None,
) -> ModelShardAssignment | None:
    if (
        layer_start is None
        or layer_end is None
        or layer_start < 0
        or layer_end <= layer_start
    ):
        return None
    return ModelShardAssignment(
        start_layer=layer_start,
        end_layer=layer_end,
        device_rank=0,
        world_size=1,
    )


def _require_local_file(path: str, *, field_name: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"{field_name} does not exist: {resolved}")
    return resolved


def _optional_positive_int(value: Any) -> int | None:
    try:
        clean = int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
    if clean is None or clean <= 0:
        return None
    return clean


def _optional_non_negative_int(value: Any) -> int | None:
    try:
        clean = int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
    if clean is None or clean < 0:
        return None
    return clean


def _require_all(raw: Any, required: Sequence[str], field_name: str) -> None:
    values = {str(item or "").strip() for item in raw or []}
    missing = [item for item in required if item not in values]
    if missing:
        raise ValueError(
            f"CAI LLM shard productionRequirements {field_name} missing: "
            + ", ".join(missing)
        )


def _call_native_engine(
    command: Sequence[str],
    request: Mapping[str, Any],
    *,
    timeout_sec: float,
    env: Mapping[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    completed = subprocess.run(
        list(command),
        input=json.dumps(dict(request), sort_keys=True),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        env={**os.environ, **{str(k): str(v) for k, v in (env or {}).items()}},
        check=False,
    )
    if completed.returncode != 0:
        error = str(completed.stderr or completed.stdout or "").strip()
        return (
            200,
            {
                "status": "error",
                "error": "CAI LLM shard native engine failed"
                + (f": {error[:500]}" if error else "."),
                "metrics": {"backendMode": "native_bridge_engine_error"},
            },
        )
    try:
        response = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        return _native_engine_invalid_json_response(
            exc,
            backend_mode="native_bridge_invalid_json",
        )
    return _native_engine_parsed_response(
        response,
        invalid_response_mode="native_bridge_invalid_response",
        request=request,
    )


def _parse_native_engine_response(
    response_text: str,
    *,
    invalid_mode: str,
    invalid_response_mode: str,
    request: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    try:
        response = json.loads(response_text or "{}")
    except json.JSONDecodeError as exc:
        return _native_engine_invalid_json_response(exc, backend_mode=invalid_mode)
    return _native_engine_parsed_response(
        response,
        invalid_response_mode=invalid_response_mode,
        request=request,
    )


def _native_engine_invalid_json_response(
    exc: json.JSONDecodeError,
    *,
    backend_mode: str,
) -> tuple[int, dict[str, Any]]:
    return (
        200,
        {
            "status": "error",
            "error": f"CAI LLM shard native engine returned invalid JSON: {exc}",
            "metrics": {"backendMode": backend_mode},
        },
    )


def _native_engine_parsed_response(
    response: Any,
    *,
    invalid_response_mode: str,
    request: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    if not isinstance(response, dict):
        return (
            200,
            {
                "status": "error",
                "error": "CAI LLM shard native engine response must be an object.",
                "metrics": {"backendMode": invalid_response_mode},
            },
        )
    try:
        return 200, _normalize_native_engine_response(
            request,
            response,
        )
    except Exception as exc:
        return (
            200,
            {
                "status": "error",
                "error": str(exc),
                "metrics": {"backendMode": invalid_response_mode},
            },
        )


def _normalize_native_engine_response(
    request: Mapping[str, Any],
    response: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(response)
    status = str(payload.get("status") or "ok").strip().lower()
    if status not in {"ok", "ready"}:
        return payload
    native_execution = _validate_native_execution_contract(
        request,
        payload.get("nativeExecution"),
    )
    if native_execution is None:
        return payload
    metrics = (
        dict(payload.get("metrics"))
        if isinstance(payload.get("metrics"), Mapping)
        else {}
    )
    metrics.setdefault("nativeExecutionValidated", True)
    metrics.setdefault("nativeExecutionMode", native_execution.get("executionMode"))
    metrics.setdefault("nativeExecutionArtifactKind", native_execution.get("artifactKind"))
    metrics.setdefault("nativeExecutionArtifactSource", native_execution.get("artifactSource"))
    fallback_mode = str(native_execution.get("fallbackMode") or "").strip()
    if fallback_mode:
        metrics.setdefault("nativeExecutionFallbackMode", fallback_mode)
    payload["metrics"] = metrics
    return payload


def _validate_native_execution_contract(
    request: Mapping[str, Any],
    value: Any,
) -> dict[str, Any] | None:
    action = str(request.get("action") or "").strip()
    if action not in {"process_prefill", "process_decode"}:
        return None
    local_resolution = request.get("localArtifactResolution")
    if not isinstance(local_resolution, Mapping):
        return None
    if not isinstance(value, Mapping):
        raise ValueError("CAI LLM shard native engine nativeExecution is missing.")
    try:
        schema_version = int(value.get("schemaVersion") or 0)
    except (TypeError, ValueError):
        raise ValueError("CAI LLM shard native engine nativeExecution schema is invalid.")
    if schema_version != NATIVE_EXECUTION_SCHEMA_VERSION:
        raise ValueError("CAI LLM shard native engine nativeExecution schema is unsupported.")
    if str(value.get("executionMode") or "").strip() != "layer_range":
        raise ValueError("CAI LLM shard native engine nativeExecution mode is unsupported.")
    if str(value.get("action") or "").strip() != action:
        raise ValueError("CAI LLM shard native engine nativeExecution action mismatch.")
    if not bool(value.get("usedPatchedBackend")):
        raise ValueError(
            "CAI LLM shard native engine nativeExecution must confirm patched backend usage."
        )
    expected_model_id = _request_model_id(request)
    model_id = str(value.get("modelId") or "").strip()
    if expected_model_id and model_id != expected_model_id:
        raise ValueError("CAI LLM shard native engine nativeExecution modelId mismatch.")
    expected_layer_start = _request_layer_bound(request, "layerStart")
    expected_layer_end = _request_layer_bound(request, "layerEnd")
    layer_start = _optional_non_negative_int(value.get("layerStart"))
    layer_end = _optional_non_negative_int(value.get("layerEnd"))
    if expected_layer_start is not None and layer_start != expected_layer_start:
        raise ValueError("CAI LLM shard native engine nativeExecution layerStart mismatch.")
    if expected_layer_end is not None and layer_end != expected_layer_end:
        raise ValueError("CAI LLM shard native engine nativeExecution layerEnd mismatch.")
    artifact_kind = str(value.get("artifactKind") or "").strip().lower()
    if artifact_kind not in {"assignment", "model"}:
        raise ValueError("CAI LLM shard native engine nativeExecution artifactKind is invalid.")
    artifact_source = str(value.get("artifactSource") or "").strip()
    artifact_path = str(value.get("artifactPath") or "").strip()
    if not artifact_source:
        raise ValueError("CAI LLM shard native engine nativeExecution artifactSource is missing.")
    if not artifact_path:
        raise ValueError("CAI LLM shard native engine nativeExecution artifactPath is missing.")
    assignment_resolution = (
        local_resolution.get("assignmentArtifact")
        if isinstance(local_resolution.get("assignmentArtifact"), Mapping)
        else None
    )
    model_resolution = (
        local_resolution.get("modelArtifact")
        if isinstance(local_resolution.get("modelArtifact"), Mapping)
        else None
    )
    if artifact_kind == "assignment":
        if assignment_resolution is None:
            raise ValueError(
                "CAI LLM shard native engine nativeExecution cannot use assignment artifact because none is available."
            )
        _validate_native_execution_artifact_match(
            value,
            assignment_resolution,
            artifact_path=artifact_path,
            artifact_source=artifact_source,
            artifact_kind="assignment",
        )
    else:
        if model_resolution is None:
            raise ValueError(
                "CAI LLM shard native engine nativeExecution cannot use model artifact because none is available."
            )
        _validate_native_execution_artifact_match(
            value,
            model_resolution,
            artifact_path=artifact_path,
            artifact_source=artifact_source,
            artifact_kind="model",
        )
        if assignment_resolution is not None:
            fallback_mode = str(value.get("fallbackMode") or "").strip()
            if fallback_mode != "full_model":
                raise ValueError(
                    "CAI LLM shard native engine nativeExecution must declare fallbackMode=full_model when assignment artifact is available but model artifact was used."
                )
    return dict(value)


def _validate_native_execution_artifact_match(
    native_execution: Mapping[str, Any],
    artifact_resolution: Mapping[str, Any],
    *,
    artifact_path: str,
    artifact_source: str,
    artifact_kind: str,
) -> None:
    expected_source = str(artifact_resolution.get("source") or "").strip()
    expected_path = str(artifact_resolution.get("localPath") or "").strip()
    if artifact_source != expected_source:
        raise ValueError(
            f"CAI LLM shard native engine nativeExecution {artifact_kind} artifactSource mismatch."
        )
    if not _paths_match(artifact_path, expected_path):
        raise ValueError(
            f"CAI LLM shard native engine nativeExecution {artifact_kind} artifactPath mismatch."
        )
    expected_artifact_id = str(artifact_resolution.get("artifactId") or "").strip()
    declared_artifact_id = str(native_execution.get("artifactId") or "").strip()
    if expected_artifact_id and declared_artifact_id and declared_artifact_id != expected_artifact_id:
        raise ValueError(
            f"CAI LLM shard native engine nativeExecution {artifact_kind} artifactId mismatch."
        )


def _request_requires_final_output(request: Mapping[str, Any]) -> bool:
    contract = request.get("outputContract")
    if isinstance(contract, Mapping):
        if contract.get("requiresFinalOutput") is not None:
            return bool(contract.get("requiresFinalOutput"))
        if contract.get("requiresOutputFrameMetadata") is not None:
            return not bool(contract.get("requiresOutputFrameMetadata"))
    return _request_output_frame_template(request) is None


def _request_output_frame_template(
    request: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    contract = request.get("outputContract")
    if not isinstance(contract, Mapping):
        return None
    template = contract.get("frameMetadataTemplate")
    return template if isinstance(template, Mapping) else None


def _expected_output_kind(request: Mapping[str, Any]) -> str:
    if _request_requires_final_output(request):
        return "final_output"
    template = _request_output_frame_template(request)
    frame_kind = str(template.get("frameKind") or "").strip().lower() if isinstance(
        template,
        Mapping,
    ) else ""
    if frame_kind == "decode":
        return "decode_state"
    if frame_kind == "activation":
        return "activation_state"
    return "state"


def _request_model_id(request: Mapping[str, Any]) -> str | None:
    shard_spec = request.get("shardSpec")
    if isinstance(shard_spec, Mapping):
        model_id = str(shard_spec.get("modelId") or "").strip()
        if model_id:
            return model_id
    frame = request.get("frame")
    if isinstance(frame, Mapping):
        model_id = str(frame.get("modelId") or "").strip()
        if model_id:
            return model_id
    return None


def _request_layer_bound(request: Mapping[str, Any], field_name: str) -> int | None:
    shard_spec = request.get("shardSpec")
    if isinstance(shard_spec, Mapping):
        value = _optional_non_negative_int(shard_spec.get(field_name))
        if value is not None:
            return value
    frame = request.get("frame")
    if isinstance(frame, Mapping):
        value = _optional_non_negative_int(frame.get(field_name))
        if value is not None:
            return value
    return None


def _paths_match(left: str, right: str) -> bool:
    try:
        return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
    except Exception:
        return str(left).strip() == str(right).strip()


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on", "required"}


def _split_command(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        clean = value.strip()
        if not clean:
            return []
        return [
            _strip_command_token_quotes(item)
            for item in shlex.split(clean, posix=(os.name != "nt"))
            if str(item or "").strip()
        ]
    return [
        _strip_command_token_quotes(str(item))
        for item in value
        if str(item or "").strip()
    ]


def _strip_command_token_quotes(value: str) -> str:
    token = str(value or "").strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        return token[1:-1]
    return token


if __name__ == "__main__":
    raise SystemExit(main())

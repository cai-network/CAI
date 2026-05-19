# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import atexit
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from typing import Any

from .cai_llama_cpp_assignment_artifact_engine import ASSIGNMENT_EXECUTOR_REQUEST_ABI
from .cai_llama_cpp_backend_runtime import (
    split_llama_cpp_subprocess_command,
    windows_subprocess_creation_flags,
    windows_subprocess_startupinfo,
)
from .cai_llama_cpp_patched_executor_host import (
    PATCHED_EXECUTION_CONTEXT_ABI,
    PATCHED_IO_TARGETS_ABI,
)


PATCHED_BINARY_EXECUTOR_ID = "patched_binary_executor"
PATCHED_BINARY_EXECUTOR_VERSION = "patched-binary-executor/0.1"
PATCHED_BINARY_SESSION_ABI = "cai-llama-cpp-patched-binary-session-v1"
PATCHED_BINARY_REQUEST_ABI = "cai-llama-cpp-patched-binary-request-v1"
CAI_LLM_PATCHED_BINARY_COMMAND_ENV = "CAI_LLM_PATCHED_BINARY_COMMAND"
CAI_LLM_PATCHED_BINARY_PERSISTENT_ENV = "CAI_LLM_PATCHED_BINARY_PERSISTENT"
CAI_LLM_PATCHED_BINARY_TIMEOUT_ENV = "CAI_LLM_PATCHED_BINARY_TIMEOUT_SEC"
CAI_LLM_SHARD_NATIVE_TIMEOUT_ENV = "CAI_LLM_SHARD_NATIVE_TIMEOUT_SEC"
CAI_LLM_PATCHED_BINARY_REQUIRE_REAL_LAYER_EXECUTION_ENV = (
    "CAI_LLM_PATCHED_BINARY_REQUIRE_REAL_LAYER_EXECUTION"
)
CAI_LLM_PATCHED_BINARY_REQUIRE_SHARD_ONLY_LOADING_ENV = (
    "CAI_LLM_PATCHED_BINARY_REQUIRE_SHARD_ONLY_LOADING"
)


@dataclass(frozen=True)
class PatchedBinaryExecutorConfig:
    binary_command: tuple[str, ...]
    persistent_binary: bool
    require_real_layer_execution: bool
    require_shard_only_loading: bool


class PersistentPatchedBinaryClient:
    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | None,
        timeout_sec: float = 120.0,
    ) -> None:
        self.command = [str(item) for item in command if str(item).strip()]
        self.cwd = str(cwd) if cwd else None
        self.timeout_sec = max(0.1, float(timeout_sec or 120.0))
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
            cwd=self.cwd,
            creationflags=windows_subprocess_creation_flags(),
            startupinfo=windows_subprocess_startupinfo(),
        )
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def call(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        with self._lock:
            if self._closed:
                raise ValueError("CAI patched binary executor persistent binary is closed.")
            if self._process.poll() is not None:
                raise ValueError(
                    "CAI patched binary executor persistent binary exited unexpectedly."
                )
            if self._process.stdin is None:
                raise ValueError(
                    "CAI patched binary executor persistent binary stdin is unavailable."
                )
            try:
                self._process.stdin.write(json.dumps(dict(request), sort_keys=True))
                self._process.stdin.write("\n")
                self._process.stdin.flush()
                response_text = self._stdout_lines.get(timeout=self.timeout_sec)
            except queue.Empty as exc:
                self.close(kill=True)
                raise ValueError(
                    "CAI patched binary executor persistent binary timed out."
                ) from exc
            except Exception as exc:
                raise ValueError(
                    "CAI patched binary executor persistent binary I/O failed."
                ) from exc
        try:
            parsed = json.loads(response_text or "{}")
        except Exception as exc:
            raise ValueError(
                "CAI patched binary executor persistent binary returned invalid JSON."
            ) from exc
        if not isinstance(parsed, Mapping):
            raise ValueError(
                "CAI patched binary executor persistent binary response must be an object."
            )
        return parsed

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

    @property
    def alive(self) -> bool:
        return not self._closed and self._process.poll() is None

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


_PERSISTENT_PATCHED_BINARY_CLIENTS: dict[
    tuple[tuple[str, ...], str | None, float],
    PersistentPatchedBinaryClient,
] = {}
_ENGINE_REQUEST_COUNT = 0


def reset_patched_binary_executor_runtime_state() -> None:
    global _ENGINE_REQUEST_COUNT
    _ENGINE_REQUEST_COUNT = 0
    for client in list(_PERSISTENT_PATCHED_BINARY_CLIENTS.values()):
        client.close(kill=True)
    _PERSISTENT_PATCHED_BINARY_CLIENTS.clear()


def build_patched_binary_executor_config(
    *,
    binary_command: str | Sequence[str] | None = None,
    persistent_binary: bool | None = None,
    require_real_layer_execution: bool | None = None,
    require_shard_only_loading: bool | None = None,
) -> PatchedBinaryExecutorConfig:
    return PatchedBinaryExecutorConfig(
        binary_command=tuple(_resolve_binary_command(binary_command)),
        persistent_binary=(
            _patched_binary_persistent_enabled()
            if persistent_binary is None
            else bool(persistent_binary)
        ),
        require_real_layer_execution=(
            _real_layer_execution_required()
            if require_real_layer_execution is None
            else bool(require_real_layer_execution)
        ),
        require_shard_only_loading=(
            _shard_only_loading_required_from_env()
            if require_shard_only_loading is None
            else bool(require_shard_only_loading)
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a CAI patched lower executor that consumes validatedExecutionContext "
            "from cai_llama_cpp_patched_executor_host and forwards shard actions to "
            "a real patched binary command."
        ),
    )
    parser.add_argument("--jsonl", action="store_true")
    parser.add_argument("--binary-command", default="")
    parser.add_argument("--persistent-binary", action="store_true")
    parser.add_argument("--require-real-layer-execution", action="store_true")
    parser.add_argument("--require-shard-only-loading", action="store_true")
    args = parser.parse_args(argv)
    config = build_patched_binary_executor_config(
        binary_command=str(args.binary_command or "").strip() or None,
        persistent_binary=(True if bool(args.persistent_binary) else None),
        require_real_layer_execution=(
            True if bool(args.require_real_layer_execution) else None
        ),
        require_shard_only_loading=(
            True if bool(args.require_shard_only_loading) else None
        ),
    )
    if bool(args.jsonl):
        return _jsonl_loop(config)
    try:
        request = json.loads(sys.stdin.read() or "{}")
        response = handle_patched_binary_executor_request(
            request,
            config=config,
        )
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), flush=True)
        return 1
    print(json.dumps(response, sort_keys=True), flush=True)
    return 0


def _jsonl_loop(config: PatchedBinaryExecutorConfig) -> int:
    for line in sys.stdin:
        try:
            request = json.loads(line or "{}")
            response = handle_patched_binary_executor_request(
                request,
                config=config,
            )
        except Exception as exc:
            response = {"status": "error", "error": str(exc)}
        print(json.dumps(response, sort_keys=True), flush=True)
    return 0


def handle_patched_binary_executor_request(
    request: Mapping[str, Any],
    *,
    config: PatchedBinaryExecutorConfig | None = None,
) -> dict[str, Any]:
    engine_runtime_metrics = _next_engine_runtime_metrics()
    if not isinstance(request, Mapping):
        raise ValueError("CAI patched binary executor request must be an object.")
    if str(request.get("abi") or "").strip() != ASSIGNMENT_EXECUTOR_REQUEST_ABI:
        raise ValueError("CAI patched binary executor ABI is unsupported.")
    action = str(request.get("action") or "").strip()
    if action not in {"load_shard", "process_prefill", "process_decode", "finalize"}:
        raise ValueError(
            f"CAI patched binary executor action is unsupported: {action}"
        )
    resolved_config = config or build_patched_binary_executor_config()
    if not resolved_config.binary_command:
        raise ValueError(
            "CAI patched binary executor binary command is not configured."
        )
    context = _validated_execution_context(request)
    session_manifest_path = _session_manifest_path(context)
    context_signature = _context_signature(context)
    _validate_session_manifest_if_present(
        session_manifest_path,
        context_signature=context_signature,
        action=action,
    )
    require_shard_only_loading = _request_requires_shard_only_loading(
        request,
        config=resolved_config,
    )
    binary_request = _build_binary_request(
        request,
        context=context,
        session_manifest_path=session_manifest_path,
        config=resolved_config,
        require_shard_only_loading=require_shard_only_loading,
    )
    _write_binary_request_plan(
        binary_request,
        context=context,
        action=action,
    )
    binary_response = _call_binary_command(
        resolved_config,
        request=binary_request,
        cwd=_binary_cwd(context),
    )
    if action == "load_shard":
        _write_session_manifest(
            session_manifest_path,
            context=context,
            context_signature=context_signature,
            last_action="load_shard",
            last_output_kind=None,
        )
    if action in {"load_shard", "finalize"}:
        if action == "finalize":
            _safe_unlink(session_manifest_path)
        return _normalize_lifecycle_response(
            binary_response,
            action=action,
            session_manifest_path=session_manifest_path,
            config=resolved_config,
            require_shard_only_loading=require_shard_only_loading,
            engine_runtime_metrics=engine_runtime_metrics,
        )
    response = _normalize_process_response(
        request,
        context=context,
        binary_response=binary_response,
        session_manifest_path=session_manifest_path,
        config=resolved_config,
        require_shard_only_loading=require_shard_only_loading,
        engine_runtime_metrics=engine_runtime_metrics,
    )
    _write_session_manifest(
        session_manifest_path,
        context=context,
        context_signature=context_signature,
        last_action=action,
        last_output_kind=str(response.get("outputKind") or "").strip() or None,
        last_state_manifest_path=(
            str((response.get("outputPayloadFile") or {}).get("path") or "").strip()
            if isinstance(response.get("outputPayloadFile"), Mapping)
            else None
        ),
    )
    return response


def _call_binary_command(
    config: PatchedBinaryExecutorConfig,
    *,
    request: Mapping[str, Any],
    cwd: str | None,
) -> Mapping[str, Any]:
    if config.persistent_binary:
        client = _persistent_patched_binary_client(config.binary_command, cwd=cwd)
        return client.call(request)
    completed = subprocess.run(
        list(config.binary_command),
        input=json.dumps(dict(request), sort_keys=True).encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        check=False,
        creationflags=windows_subprocess_creation_flags(),
        startupinfo=windows_subprocess_startupinfo(),
    )
    if int(completed.returncode) != 0:
        stderr_text = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(
            "CAI patched binary executor command failed"
            + (f" ({stderr_text})" if stderr_text else "")
            + "."
        )
    try:
        parsed = json.loads(completed.stdout.decode("utf-8") or "{}")
    except Exception as exc:
        raise ValueError(
            "CAI patched binary executor response is invalid JSON."
        ) from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(
            "CAI patched binary executor response must be an object."
        )
    return parsed


def _normalize_lifecycle_response(
    response: Mapping[str, Any],
    *,
    action: str,
    session_manifest_path: Path,
    config: PatchedBinaryExecutorConfig,
    require_shard_only_loading: bool,
    engine_runtime_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    status = str(response.get("status") or "").strip().lower()
    if status not in {"ready", "ok"}:
        detail = str(response.get("error") or "").strip()
        raise ValueError(
            "CAI patched binary executor returned non-ready status"
            + (f": {detail}" if detail else ".")
        )
    if not bool(response.get("realModelExecution")):
        raise ValueError(
            "CAI patched binary executor must declare realModelExecution=true."
        )
    if require_shard_only_loading and action == "load_shard":
        _validate_shard_only_loading_response(response)
    return {
        "status": status,
        "realModelExecution": True,
        "metrics": _executor_metrics(
            config,
            response=response,
            session_manifest_path=session_manifest_path,
            engine_runtime_metrics=engine_runtime_metrics,
        ),
    }


def _normalize_process_response(
    request: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    binary_response: Mapping[str, Any],
    session_manifest_path: Path,
    config: PatchedBinaryExecutorConfig,
    require_shard_only_loading: bool,
    engine_runtime_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    status = str(binary_response.get("status") or "").strip().lower()
    if status != "ok":
        detail = str(binary_response.get("error") or "").strip()
        raise ValueError(
            "CAI patched binary executor returned non-ok status"
            + (f": {detail}" if detail else ".")
        )
    if not bool(binary_response.get("realModelExecution")):
        raise ValueError(
            "CAI patched binary executor must declare realModelExecution=true."
        )
    real_layer_execution = bool(binary_response.get("realLayerExecution"))
    if config.require_real_layer_execution and not real_layer_execution:
        raise ValueError(
            "CAI patched binary executor requires realLayerExecution=true."
        )
    if require_shard_only_loading:
        _validate_shard_only_loading_response(binary_response)
    expected_output_kind = str(context.get("expectedOutputKind") or "").strip()
    if not expected_output_kind:
        raise ValueError(
            "CAI patched binary executor expectedOutputKind is missing."
        )
    output_kind = (
        str(binary_response.get("outputKind") or "").strip() or expected_output_kind
    )
    if output_kind != expected_output_kind:
        raise ValueError(
            "CAI patched binary executor outputKind does not match expectedOutputKind."
        )
    payload_file = binary_response.get("outputPayloadFile")
    payload_base64 = str(binary_response.get("outputPayloadBase64") or "").strip()
    if not isinstance(payload_file, Mapping) and not payload_base64:
        raise ValueError(
            "CAI patched binary executor did not return output payload."
        )
    result: dict[str, Any] = {
        "status": "ok",
        "outputKind": output_kind,
        "realModelExecution": True,
        "realLayerExecution": real_layer_execution,
        "metrics": _executor_metrics(
            config,
            response=binary_response,
            session_manifest_path=session_manifest_path,
            engine_runtime_metrics=engine_runtime_metrics,
        ),
    }
    if isinstance(payload_file, Mapping):
        result["outputPayloadFile"] = dict(payload_file)
    if payload_base64:
        result["outputPayloadBase64"] = payload_base64
    output_hash = str(binary_response.get("outputPayloadSha256Hex") or "").strip()
    if output_hash:
        result["outputPayloadSha256Hex"] = output_hash
    return result


def _executor_metrics(
    config: PatchedBinaryExecutorConfig,
    *,
    response: Mapping[str, Any],
    session_manifest_path: Path,
    engine_runtime_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "engineBackendMode": PATCHED_BINARY_EXECUTOR_ID,
        "engineBackendVersion": PATCHED_BINARY_EXECUTOR_VERSION,
        "patchedBinaryCommand": str(config.binary_command[0]),
        "patchedBinaryMode": (
            "persistent_jsonl"
            if config.persistent_binary
            else "subprocess_per_request"
        ),
        "binaryRequestAbi": PATCHED_BINARY_REQUEST_ABI,
        "requireRealLayerExecution": bool(config.require_real_layer_execution),
        "requireShardOnlyLoading": bool(config.require_shard_only_loading),
        "realModelExecution": bool(response.get("realModelExecution")),
        "realLayerExecution": bool(response.get("realLayerExecution")),
        "sessionManifestPath": str(session_manifest_path),
        **dict(engine_runtime_metrics),
    }
    response_metrics = response.get("metrics")
    if isinstance(response_metrics, Mapping):
        metrics["patchedBinaryMetrics"] = dict(response_metrics)
    return metrics


def _build_binary_request(
    request: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    session_manifest_path: Path,
    config: PatchedBinaryExecutorConfig,
    require_shard_only_loading: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schemaVersion": 1,
        "abi": PATCHED_BINARY_REQUEST_ABI,
        "action": str(request.get("action") or "").strip(),
        "sessionId": str(context.get("sessionId") or "").strip(),
        "modelId": str(context.get("modelId") or "").strip(),
        "layerStart": _optional_int(context.get("layerStart")),
        "layerEnd": _optional_int(context.get("layerEnd")),
        "tokenStart": _optional_int(context.get("tokenStart")),
        "tokenEnd": _optional_int(context.get("tokenEnd")),
        "requiresFinalOutput": bool(context.get("requiresFinalOutput")),
        "expectedOutputKind": str(context.get("expectedOutputKind") or "").strip(),
        "requireRealLayerExecution": bool(
            config.require_real_layer_execution
            or request.get("requireRealLayerExecution")
        ),
        "requireShardOnlyLoading": bool(require_shard_only_loading),
        "sessionManifestPath": str(session_manifest_path),
        "executionWorkspace": _mapping_copy(context.get("executionWorkspace")),
        "ioTargets": _mapping_copy(context.get("ioTargets")),
        "assignmentArtifact": _mapping_copy(context.get("assignmentArtifact")),
        "modelArtifact": _mapping_copy(context.get("modelArtifact")),
        "managedRuntime": _mapping_copy(context.get("managedRuntime")),
        "shardSpec": _mapping_copy(context.get("shardSpec")),
        "frame": _mapping_copy(context.get("frame")),
        "outputContract": _mapping_copy(context.get("outputContract")),
        "productionRequirements": _mapping_copy(request.get("productionRequirements")),
    }
    if isinstance(request.get("inputPayloadFile"), Mapping):
        payload["inputPayloadFile"] = dict(request.get("inputPayloadFile") or {})
    if isinstance(request.get("validatedInputState"), Mapping):
        payload["inputState"] = dict(request.get("validatedInputState") or {})
    return {
        key: value
        for key, value in payload.items()
        if value not in (None, {}, "")
    }


def _write_binary_request_plan(
    request: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    action: str,
) -> None:
    workspace = _execution_workspace(context)
    plan_path = (
        workspace["stateFilesDir"] / f"patched-binary-{str(action or '').strip()}.request.json"
    ).resolve()
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(dict(request), sort_keys=True), encoding="utf-8")


def _validated_execution_context(request: Mapping[str, Any]) -> Mapping[str, Any]:
    context = request.get("validatedExecutionContext")
    if not isinstance(context, Mapping):
        raise ValueError(
            "CAI patched binary executor validatedExecutionContext is missing."
        )
    if str(context.get("abi") or "").strip() != PATCHED_EXECUTION_CONTEXT_ABI:
        raise ValueError(
            "CAI patched binary executor validatedExecutionContext ABI is unsupported."
        )
    io_targets = context.get("ioTargets")
    if not isinstance(io_targets, Mapping):
        raise ValueError("CAI patched binary executor ioTargets are missing.")
    if str(io_targets.get("abi") or "").strip() != PATCHED_IO_TARGETS_ABI:
        raise ValueError("CAI patched binary executor ioTargets ABI is unsupported.")
    return context


def _execution_workspace(context: Mapping[str, Any]) -> dict[str, Path]:
    workspace = context.get("executionWorkspace")
    if not isinstance(workspace, Mapping):
        raise ValueError(
            "CAI patched binary executor executionWorkspace is missing."
        )
    root = _path_from_mapping(workspace, "root")
    inputs_dir = _path_from_mapping(workspace, "inputsDir")
    outputs_dir = _path_from_mapping(workspace, "outputsDir")
    state_files_dir = _path_from_mapping(workspace, "stateFilesDir")
    return {
        "root": root,
        "inputsDir": inputs_dir,
        "outputsDir": outputs_dir,
        "stateFilesDir": state_files_dir,
    }


def _session_manifest_path(context: Mapping[str, Any]) -> Path:
    workspace = _execution_workspace(context)
    return (workspace["stateFilesDir"] / "patched-binary-session.json").resolve()


def _context_signature(context: Mapping[str, Any]) -> str:
    payload = {
        "sessionId": str(context.get("sessionId") or "").strip(),
        "modelId": str(context.get("modelId") or "").strip(),
        "layerStart": _optional_int(context.get("layerStart")),
        "layerEnd": _optional_int(context.get("layerEnd")),
        "assignmentArtifact": _artifact_identity_payload(
            context.get("assignmentArtifact"),
            include_chunks=True,
        ),
        "modelArtifact": _artifact_identity_payload(
            context.get("modelArtifact"),
            include_chunks=False,
        ),
    }
    return _sha256_json(payload)


def _validate_session_manifest_if_present(
    manifest_path: Path,
    *,
    context_signature: str,
    action: str,
) -> None:
    if not manifest_path.exists():
        return
    try:
        parsed = json.loads(manifest_path.read_text(encoding="utf-8") or "{}")
    except Exception as exc:
        raise ValueError(
            "CAI patched binary executor session manifest is invalid."
        ) from exc
    if not isinstance(parsed, Mapping):
        raise ValueError("CAI patched binary executor session manifest is invalid.")
    if str(parsed.get("abi") or "").strip() != PATCHED_BINARY_SESSION_ABI:
        raise ValueError(
            "CAI patched binary executor session manifest ABI is unsupported."
        )
    if action == "load_shard":
        return
    declared_signature = str(parsed.get("contextSignature") or "").strip()
    if declared_signature and declared_signature != context_signature:
        raise ValueError(
            "CAI patched binary executor detected execution context drift."
        )


def _write_session_manifest(
    manifest_path: Path,
    *,
    context: Mapping[str, Any],
    context_signature: str,
    last_action: str,
    last_output_kind: str | None,
    last_state_manifest_path: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schemaVersion": 1,
        "abi": PATCHED_BINARY_SESSION_ABI,
        "sessionId": str(context.get("sessionId") or "").strip(),
        "modelId": str(context.get("modelId") or "").strip(),
        "layerStart": _optional_int(context.get("layerStart")),
        "layerEnd": _optional_int(context.get("layerEnd")),
        "tokenStart": _optional_int(context.get("tokenStart")),
        "tokenEnd": _optional_int(context.get("tokenEnd")),
        "expectedOutputKind": str(context.get("expectedOutputKind") or "").strip(),
        "contextSignature": context_signature,
        "lastAction": str(last_action or "").strip(),
        "lastOutputKind": str(last_output_kind or "").strip() or None,
        "lastStateManifestPath": str(last_state_manifest_path or "").strip() or None,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _artifact_identity_payload(
    artifact: Any,
    *,
    include_chunks: bool,
) -> dict[str, Any] | None:
    if not isinstance(artifact, Mapping):
        return None
    payload: dict[str, Any] = {
        "artifactId": str(artifact.get("artifactId") or "").strip() or None,
        "source": str(artifact.get("source") or "").strip() or None,
        "localPath": str(artifact.get("localPath") or "").strip() or None,
        "layerStart": _optional_int(artifact.get("layerStart")),
        "layerEnd": _optional_int(artifact.get("layerEnd")),
    }
    coverage = artifact.get("coverage")
    if isinstance(coverage, Mapping):
        payload["coverage"] = dict(coverage)
    if include_chunks:
        chunk_ranges = artifact.get("chunkRanges")
        if isinstance(chunk_ranges, Sequence) and not isinstance(
            chunk_ranges, (str, bytes, bytearray)
        ):
            payload["chunkRanges"] = [
                dict(item)
                for item in chunk_ranges
                if isinstance(item, Mapping)
            ]
    return payload


def _binary_cwd(context: Mapping[str, Any]) -> str | None:
    managed_runtime = context.get("managedRuntime")
    if not isinstance(managed_runtime, Mapping):
        return None
    repo_root = str(managed_runtime.get("repoRoot") or "").strip()
    if repo_root:
        return str(Path(repo_root).expanduser().resolve())
    runtime_root = str(managed_runtime.get("runtimeRoot") or "").strip()
    if runtime_root:
        return str(Path(runtime_root).expanduser().resolve())
    return None


def _resolve_binary_command(
    binary_command: str | Sequence[str] | None,
) -> list[str]:
    if isinstance(binary_command, Sequence) and not isinstance(
        binary_command,
        (str, bytes, bytearray),
    ):
        return [str(item) for item in binary_command if str(item).strip()]
    raw = str(
        binary_command
        if binary_command is not None
        else (os.getenv(CAI_LLM_PATCHED_BINARY_COMMAND_ENV) or "")
    ).strip()
    return split_llama_cpp_subprocess_command(raw)


def _patched_binary_persistent_enabled() -> bool:
    raw = str(os.getenv(CAI_LLM_PATCHED_BINARY_PERSISTENT_ENV) or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _patched_binary_timeout_seconds() -> float:
    for name in (
        CAI_LLM_PATCHED_BINARY_TIMEOUT_ENV,
        CAI_LLM_SHARD_NATIVE_TIMEOUT_ENV,
    ):
        raw = str(os.getenv(name) or "").strip()
        if not raw:
            continue
        try:
            return max(0.1, float(raw))
        except ValueError:
            continue
    return 120.0


def _real_layer_execution_required() -> bool:
    raw = str(
        os.getenv(CAI_LLM_PATCHED_BINARY_REQUIRE_REAL_LAYER_EXECUTION_ENV) or ""
    ).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _shard_only_loading_required_from_env() -> bool:
    raw = str(
        os.getenv(CAI_LLM_PATCHED_BINARY_REQUIRE_SHARD_ONLY_LOADING_ENV) or ""
    ).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _request_requires_shard_only_loading(
    request: Mapping[str, Any],
    *,
    config: PatchedBinaryExecutorConfig,
) -> bool:
    if bool(config.require_shard_only_loading):
        return True
    if _truthy(request.get("requireShardOnlyLoading")):
        return True
    requirements = request.get("productionRequirements")
    if isinstance(requirements, Mapping):
        return bool(
            _truthy(requirements.get("requiresShardOnlyLoading"))
            or _truthy(requirements.get("forbidFullModelFallback"))
        )
    return False


def _validate_shard_only_loading_response(response: Mapping[str, Any]) -> None:
    metrics = response.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(
            "CAI patched binary executor requires shard-only loading proof."
        )
    used_full_model = _nested_bool(metrics, "usedFullModelForLayerRange")
    shard_only_ready = _nested_bool(metrics, "shardOnlyLoadingReady")
    assignment_present = _nested_bool(metrics, "assignmentArtifactPresent")
    if used_full_model is True:
        raise ValueError(
            "CAI patched binary executor requires shard-only loading; lower "
            "binary used a full model for layer-range execution."
        )
    if shard_only_ready is not True:
        raise ValueError(
            "CAI patched binary executor requires shard-only loading proof."
        )
    if assignment_present is False:
        raise ValueError(
            "CAI patched binary executor requires assignmentArtifact-backed loading."
        )


def _nested_bool(value: Any, key: str) -> bool | None:
    if isinstance(value, Mapping):
        if key in value and isinstance(value.get(key), bool):
            return bool(value.get(key))
        for child in value.values():
            found = _nested_bool(child, key)
            if found is not None:
                return found
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            found = _nested_bool(child, key)
            if found is not None:
                return found
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _persistent_patched_binary_client(
    command: Sequence[str],
    *,
    cwd: str | None,
) -> PersistentPatchedBinaryClient:
    timeout_sec = _patched_binary_timeout_seconds()
    key = (tuple(str(item) for item in command if str(item).strip()), cwd, timeout_sec)
    client = _PERSISTENT_PATCHED_BINARY_CLIENTS.get(key)
    if client is not None and client.alive:
        return client
    if client is not None:
        client.close(kill=True)
    client = PersistentPatchedBinaryClient(
        command,
        cwd=cwd,
        timeout_sec=timeout_sec,
    )
    _PERSISTENT_PATCHED_BINARY_CLIENTS[key] = client
    return client


def _next_engine_runtime_metrics() -> dict[str, Any]:
    global _ENGINE_REQUEST_COUNT
    _ENGINE_REQUEST_COUNT += 1
    return {
        "engineProcessId": int(os.getpid()),
        "engineProcessRequestCount": int(_ENGINE_REQUEST_COUNT),
    }


def _path_from_mapping(payload: Mapping[str, Any], field_name: str) -> Path:
    raw = str(payload.get(field_name) or "").strip()
    if not raw:
        raise ValueError(
            f"CAI patched binary executor workspace field is missing: {field_name}"
        )
    return Path(raw).expanduser().resolve()


def _mapping_copy(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _sha256_json(payload: Mapping[str, Any]) -> str:
    return subprocess_hash_json(payload)


def subprocess_hash_json(payload: Mapping[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


atexit.register(reset_patched_binary_executor_runtime_state)


if __name__ == "__main__":
    raise SystemExit(main())

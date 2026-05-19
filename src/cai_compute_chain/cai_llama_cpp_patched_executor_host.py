# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import atexit
import argparse
import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
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
    resolve_managed_llama_cpp_runtime,
    split_llama_cpp_subprocess_command,
    windows_subprocess_creation_flags,
    windows_subprocess_startupinfo,
)
from .cai_llama_cpp_native_engine_contract import (
    resolve_assignment_artifact_chunk_ranges,
    resolve_assignment_artifact_coverage,
    validate_assignment_artifact_chunk_layer_coverage,
)
from .cai_llama_cpp_real_state_contract import validate_real_state_payload
from .cai_llama_cpp_real_state_contract import looks_like_real_state_payload


PATCHED_EXECUTOR_HOST_ID = "patched_executor_host"
PATCHED_EXECUTOR_HOST_VERSION = "patched-executor-host/0.1"
PATCHED_EXECUTION_CONTEXT_ABI = "cai-llama-cpp-patched-execution-context-v1"
PATCHED_IO_TARGETS_ABI = "cai-llama-cpp-patched-io-targets-v1"
CAI_LLM_PATCHED_ENGINE_COMMAND_ENV = "CAI_LLM_SHARD_PATCHED_ENGINE_COMMAND"
CAI_LLM_PATCHED_ENGINE_PERSISTENT_ENV = "CAI_LLM_SHARD_PATCHED_ENGINE_PERSISTENT"
CAI_LLM_PATCHED_ENGINE_TIMEOUT_ENV = "CAI_LLM_SHARD_PATCHED_ENGINE_TIMEOUT_SEC"
CAI_LLM_SHARD_NATIVE_TIMEOUT_ENV = "CAI_LLM_SHARD_NATIVE_TIMEOUT_SEC"


@dataclass(frozen=True)
class PatchedExecutorHostConfig:
    engine_command: tuple[str, ...]
    persistent_engine: bool


@dataclass(frozen=True)
class OutputPayloadEnvelope:
    payload: bytes
    output_file: dict[str, Any] | None
    output_base64: str | None
    sha256_hex: str


@dataclass(frozen=True)
class PreparedEngineRequest:
    payload: dict[str, Any]
    metrics: dict[str, Any]


class PersistentPatchedEngineClient:
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
                raise ValueError("CAI patched executor host persistent engine is closed.")
            if self._process.poll() is not None:
                raise ValueError(
                    "CAI patched executor host persistent engine exited unexpectedly."
                )
            if self._process.stdin is None:
                raise ValueError(
                    "CAI patched executor host persistent engine stdin is unavailable."
                )
            try:
                self._process.stdin.write(json.dumps(dict(request), sort_keys=True))
                self._process.stdin.write("\n")
                self._process.stdin.flush()
                response_text = self._stdout_lines.get(timeout=self.timeout_sec)
            except queue.Empty as exc:
                self.close(kill=True)
                raise ValueError(
                    "CAI patched executor host persistent engine timed out."
                ) from exc
            except Exception as exc:
                raise ValueError(
                    "CAI patched executor host persistent engine I/O failed."
                ) from exc
        try:
            parsed = json.loads(response_text or "{}")
        except Exception as exc:
            raise ValueError(
                "CAI patched executor host persistent engine returned invalid JSON."
            ) from exc
        if not isinstance(parsed, Mapping):
            raise ValueError(
                "CAI patched executor host persistent engine response must be an object."
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


_PERSISTENT_PATCHED_ENGINE_CLIENTS: dict[
    tuple[tuple[str, ...], str | None, float],
    PersistentPatchedEngineClient,
] = {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a CAI patched-executor host. This host speaks the "
            "assignment-executor ABI, forwards requests to a local patched "
            "engine command, and validates real activation/decode state "
            "manifests inside the execution workspace."
        ),
    )
    parser.add_argument("--jsonl", action="store_true")
    parser.add_argument("--engine-command", default="")
    parser.add_argument("--persistent-engine", action="store_true")
    args = parser.parse_args(argv)
    config = build_patched_executor_host_config(
        engine_command=str(args.engine_command or "").strip() or None,
        persistent_engine=(
            True if bool(args.persistent_engine) else None
        ),
    )
    if bool(args.jsonl):
        return _jsonl_loop(config)
    try:
        request = json.loads(sys.stdin.read() or "{}")
        response = handle_patched_executor_host_request(
            request,
            config=config,
        )
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), flush=True)
        return 1
    print(json.dumps(response, sort_keys=True), flush=True)
    return 0


def build_patched_executor_host_config(
    *,
    engine_command: str | Sequence[str] | None = None,
    persistent_engine: bool | None = None,
) -> PatchedExecutorHostConfig:
    command = _resolve_engine_command(engine_command)
    return PatchedExecutorHostConfig(
        engine_command=tuple(command),
        persistent_engine=(
            _patched_engine_persistent_enabled()
            if persistent_engine is None
            else bool(persistent_engine)
        ),
    )


def reset_patched_executor_host_clients() -> None:
    for client in list(_PERSISTENT_PATCHED_ENGINE_CLIENTS.values()):
        client.close(kill=True)
    _PERSISTENT_PATCHED_ENGINE_CLIENTS.clear()


def handle_patched_executor_host_request(
    request: Mapping[str, Any],
    *,
    config: PatchedExecutorHostConfig | None = None,
) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        raise ValueError("CAI patched executor host request must be an object.")
    if str(request.get("abi") or "").strip() != ASSIGNMENT_EXECUTOR_REQUEST_ABI:
        raise ValueError("CAI patched executor host ABI is unsupported.")
    action = str(request.get("action") or "").strip()
    if action not in {"load_shard", "process_prefill", "process_decode", "finalize"}:
        raise ValueError(
            f"CAI patched executor host action is unsupported: {action}"
        )
    resolved_config = config or build_patched_executor_host_config()
    if not resolved_config.engine_command:
        raise ValueError(
            "CAI patched executor host engine command is not configured."
        )
    prepared_request = _prepare_engine_request(request)
    engine_response = _call_patched_engine(
        resolved_config,
        request=prepared_request.payload,
    )
    if action in {"load_shard", "finalize"}:
        return _normalize_lifecycle_response(
            engine_response,
            config=resolved_config,
            preflight_metrics=prepared_request.metrics,
        )
    return _normalize_process_response(
        prepared_request.payload,
        engine_response,
        config=resolved_config,
        preflight_metrics=prepared_request.metrics,
    )


def _jsonl_loop(config: PatchedExecutorHostConfig) -> int:
    for line in sys.stdin:
        try:
            request = json.loads(line or "{}")
            response = handle_patched_executor_host_request(
                request,
                config=config,
            )
        except Exception as exc:
            response = {"status": "error", "error": str(exc)}
        print(json.dumps(response, sort_keys=True), flush=True)
    return 0


def _call_patched_engine(
    config: PatchedExecutorHostConfig,
    *,
    request: Mapping[str, Any],
) -> Mapping[str, Any]:
    cwd = _engine_cwd(request)
    if config.persistent_engine:
        client = _persistent_patched_engine_client(
            config.engine_command,
            cwd=cwd,
        )
        return client.call(request)
    completed = subprocess.run(
        list(config.engine_command),
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
            "CAI patched executor host engine command failed"
            + (f" ({stderr_text})" if stderr_text else "")
            + "."
        )
    try:
        parsed = json.loads(completed.stdout.decode("utf-8") or "{}")
    except Exception as exc:
        raise ValueError(
            "CAI patched executor host engine response is invalid JSON."
        ) from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(
            "CAI patched executor host engine response must be an object."
        )
    return parsed


def _normalize_lifecycle_response(
    response: Mapping[str, Any],
    *,
    config: PatchedExecutorHostConfig,
    preflight_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    status = str(response.get("status") or "").strip().lower()
    if status not in {"ready", "ok"}:
        detail = str(response.get("error") or response.get("message") or "").strip()
        if not detail:
            detail = json.dumps(dict(response), sort_keys=True)[:500]
        raise ValueError(
            "CAI patched executor host engine returned non-ready status"
            + (f": {detail}" if detail else ".")
        )
    if not bool(response.get("realModelExecution")):
        raise ValueError(
            "CAI patched executor host engine must declare realModelExecution=true."
        )
    return {
        "status": status,
        "realModelExecution": True,
        "metrics": _host_metrics(
            config,
            response=response,
            preflight_metrics=preflight_metrics,
        ),
    }


def _normalize_process_response(
    request: Mapping[str, Any],
    response: Mapping[str, Any],
    *,
    config: PatchedExecutorHostConfig,
    preflight_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    status = str(response.get("status") or "").strip().lower()
    if status != "ok":
        detail = str(response.get("error") or response.get("message") or "").strip()
        if not detail:
            detail = json.dumps(dict(response), sort_keys=True)[:500]
        raise ValueError(
            "CAI patched executor host engine returned non-ok status"
            + (f": {detail}" if detail else ".")
        )
    if not bool(response.get("realModelExecution")):
        raise ValueError(
            "CAI patched executor host engine must declare realModelExecution=true."
        )
    expected_output_kind = str(request.get("expectedOutputKind") or "").strip()
    if not expected_output_kind:
        raise ValueError("CAI patched executor host expectedOutputKind is missing.")
    output_kind = str(response.get("outputKind") or "").strip() or expected_output_kind
    if output_kind != expected_output_kind:
        raise ValueError(
            "CAI patched executor host engine outputKind does not match request."
        )
    envelope = _engine_output_payload(response, request=request)
    metrics = _host_metrics(
        config,
        response=response,
        preflight_metrics=preflight_metrics,
    )
    if output_kind in {"activation_state", "decode_state"}:
        manifest = validate_real_state_payload(
            envelope.payload,
            request=request,
            output_kind=output_kind,
            error_prefix="CAI patched executor host real state payload",
        )
        _validate_state_file_target(
            request,
            state_file_path=manifest.state_file.path,
        )
        metrics["validatedStateKind"] = manifest.state_kind
        metrics["validatedStateFormat"] = manifest.state_format
        metrics["validatedStateFileBytes"] = manifest.state_file.size_bytes
        metrics["validatedStateFilePath"] = str(manifest.state_file.path)
    result: dict[str, Any] = {
        "status": "ok",
        "outputKind": output_kind,
        "realModelExecution": True,
        "outputPayloadSha256Hex": envelope.sha256_hex,
        "metrics": metrics,
    }
    if envelope.output_file is not None:
        result["outputPayloadFile"] = envelope.output_file
    elif envelope.output_base64 is not None:
        result["outputPayloadBase64"] = envelope.output_base64
    else:
        raise ValueError(
            "CAI patched executor host did not preserve engine output payload."
        )
    return result


def _engine_output_payload(
    response: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
) -> OutputPayloadEnvelope:
    payload_file = response.get("outputPayloadFile")
    if isinstance(payload_file, Mapping):
        workspace = request.get("executionWorkspace")
        if not isinstance(workspace, Mapping):
            raise ValueError(
                "CAI patched executor host executionWorkspace is missing."
            )
        outputs_dir = _workspace_dir(
            workspace,
            field_name="outputsDir",
            fallback_field="root",
        )
        raw_path = str(payload_file.get("path") or "").strip()
        if not raw_path:
            raise ValueError(
                "CAI patched executor host engine outputPayloadFile path is missing."
            )
        output_path = _workspace_path(raw_path, base_dir=outputs_dir)
        if not _path_is_within(output_path, outputs_dir):
            raise ValueError(
                "CAI patched executor host engine output path must stay within "
                "executionWorkspace.outputsDir."
            )
        if not output_path.exists() or not output_path.is_file():
            raise ValueError(
                "CAI patched executor host engine output file is unavailable."
            )
        _validate_output_payload_target(
            request,
            output_path=output_path,
        )
        payload = output_path.read_bytes()
        size_bytes = _mapping_non_negative_int(
            payload_file.get("sizeBytes"),
            field_name=(
                "CAI patched executor host engine outputPayloadFile sizeBytes"
            ),
        )
        if size_bytes != len(payload):
            raise ValueError(
                "CAI patched executor host engine output file size mismatch."
            )
        output_hash = hashlib.sha256(payload).hexdigest()
        declared_hash = str(payload_file.get("sha256Hex") or "").strip().lower()
        if declared_hash and declared_hash != output_hash:
            raise ValueError(
                "CAI patched executor host engine output file hash mismatch."
            )
        _validate_response_payload_hash(response, actual_hash=output_hash)
        return OutputPayloadEnvelope(
            payload=payload,
            output_file={
                "path": str(output_path),
                "sizeBytes": len(payload),
                "sha256Hex": output_hash,
            },
            output_base64=None,
            sha256_hex=output_hash,
        )
    raw_base64 = str(response.get("outputPayloadBase64") or "").strip()
    if not raw_base64:
        raise ValueError(
            "CAI patched executor host engine did not return output payload."
        )
    try:
        payload = base64.b64decode(raw_base64.encode("ascii"), validate=True)
    except Exception as exc:
        raise ValueError(
            "CAI patched executor host engine outputPayloadBase64 is invalid."
        ) from exc
    output_hash = hashlib.sha256(payload).hexdigest()
    _validate_response_payload_hash(response, actual_hash=output_hash)
    return OutputPayloadEnvelope(
        payload=payload,
        output_file=None,
        output_base64=raw_base64,
        sha256_hex=output_hash,
    )


def _validate_response_payload_hash(
    response: Mapping[str, Any],
    *,
    actual_hash: str,
) -> None:
    declared_hash = str(response.get("outputPayloadSha256Hex") or "").strip().lower()
    if declared_hash and declared_hash != actual_hash:
        raise ValueError(
            "CAI patched executor host engine outputPayloadSha256Hex mismatch."
        )


def _host_metrics(
    config: PatchedExecutorHostConfig,
    *,
    response: Mapping[str, Any],
    preflight_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "executorBackendMode": PATCHED_EXECUTOR_HOST_ID,
        "executorBackendVersion": PATCHED_EXECUTOR_HOST_VERSION,
        "patchedEngineCommand": str(config.engine_command[0]),
        "patchedEngineMode": (
            "persistent_jsonl"
            if config.persistent_engine
            else "subprocess_per_request"
        ),
    }
    response_metrics = response.get("metrics")
    if isinstance(response_metrics, Mapping):
        metrics["patchedEngineMetrics"] = dict(response_metrics)
    if isinstance(preflight_metrics, Mapping):
        metrics.update(dict(preflight_metrics))
    return metrics


def _prepare_engine_request(request: Mapping[str, Any]) -> PreparedEngineRequest:
    action = str(request.get("action") or "").strip()
    payload = dict(request)
    metrics: dict[str, Any] = {}
    if action not in {"process_prefill", "process_decode"}:
        payload["validatedExecutionContext"] = _build_validated_execution_context(
            request,
            validated_input_state=None,
        )
        return PreparedEngineRequest(payload=payload, metrics=metrics)
    input_payload, input_file_metadata = _read_input_payload(request)
    metrics.update(input_file_metadata)
    validated_input_state = _validated_input_state_payload(
        input_payload,
        request=request,
    )
    if validated_input_state is not None:
        payload["validatedInputState"] = validated_input_state
        metrics["engineInputPayloadKind"] = "real_state_manifest"
        metrics["validatedInputStateKind"] = str(
            validated_input_state.get("stateKind") or ""
        )
        metrics["validatedInputStateFormat"] = str(
            validated_input_state.get("stateFormat") or ""
        )
        state_file = validated_input_state.get("stateFile")
        if isinstance(state_file, Mapping):
            metrics["validatedInputStateFileBytes"] = int(
                state_file.get("sizeBytes") or 0
            )
            metrics["validatedInputStateFilePath"] = str(
                state_file.get("path") or ""
            )
    else:
        metrics["engineInputPayloadKind"] = "raw_bytes"
    payload["validatedExecutionContext"] = _build_validated_execution_context(
        request,
        validated_input_state=validated_input_state,
    )
    return PreparedEngineRequest(payload=payload, metrics=metrics)


def _resolve_engine_command(
    engine_command: str | Sequence[str] | None,
) -> list[str]:
    if isinstance(engine_command, Sequence) and not isinstance(
        engine_command,
        (str, bytes, bytearray),
    ):
        return [
            str(item)
            for item in engine_command
            if str(item).strip()
        ]
    raw = str(
        engine_command
        if engine_command is not None
        else (os.getenv(CAI_LLM_PATCHED_ENGINE_COMMAND_ENV) or "")
    ).strip()
    return split_llama_cpp_subprocess_command(raw)


def _read_input_payload(
    request: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    input_payload_file = request.get("inputPayloadFile")
    if not isinstance(input_payload_file, Mapping):
        raise ValueError(
            "CAI patched executor host inputPayloadFile is missing."
        )
    workspace = request.get("executionWorkspace")
    if not isinstance(workspace, Mapping):
        raise ValueError(
            "CAI patched executor host executionWorkspace is missing."
        )
    inputs_dir = _workspace_dir(
        workspace,
        field_name="inputsDir",
        fallback_field="root",
    )
    raw_path = str(input_payload_file.get("path") or "").strip()
    if not raw_path:
        raise ValueError(
            "CAI patched executor host inputPayloadFile path is missing."
        )
    input_path = _workspace_path(raw_path, base_dir=inputs_dir)
    if not _path_is_within(input_path, inputs_dir):
        raise ValueError(
            "CAI patched executor host input path must stay within "
            "executionWorkspace.inputsDir."
        )
    if not input_path.exists() or not input_path.is_file():
        raise ValueError(
            "CAI patched executor host input file is unavailable."
        )
    payload = input_path.read_bytes()
    size_bytes = _mapping_non_negative_int(
        input_payload_file.get("sizeBytes"),
        field_name="CAI patched executor host inputPayloadFile sizeBytes",
    )
    if size_bytes != len(payload):
        raise ValueError(
            "CAI patched executor host input file size mismatch."
        )
    declared_hash = str(input_payload_file.get("sha256Hex") or "").strip().lower()
    payload_hash = hashlib.sha256(payload).hexdigest()
    if declared_hash and declared_hash != payload_hash:
        raise ValueError(
            "CAI patched executor host input file hash mismatch."
        )
    return payload, {
        "inputPayloadSha256Hex": payload_hash,
        "inputPayloadSizeBytes": len(payload),
        "inputPayloadPath": str(input_path),
    }


def _validated_input_state_payload(
    input_payload: bytes,
    *,
    request: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not looks_like_real_state_payload(input_payload):
        return None
    manifest = validate_real_state_payload(
        input_payload,
        request=request,
        output_kind=_expected_input_state_kind(request),
        error_prefix="CAI patched executor host input real state payload",
        match_request_action=False,
        match_request_bounds=False,
        allow_external_state_file=True,
    )
    _validate_input_state_handoff_bounds(
        manifest.payload,
        request=request,
        error_prefix="CAI patched executor host input real state payload",
    )
    return dict(manifest.payload)


def _expected_input_state_kind(request: Mapping[str, Any]) -> str:
    frame = request.get("frame")
    frame_kind = (
        str(frame.get("frameKind") or "").strip().lower()
        if isinstance(frame, Mapping)
        else ""
    )
    if frame_kind == "activation":
        return "activation_state"
    if frame_kind == "decode":
        return "decode_state"
    action = str(request.get("action") or "").strip().lower()
    if action == "process_decode":
        return "decode_state"
    return "activation_state"


def _validate_input_state_handoff_bounds(
    manifest: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    error_prefix: str,
) -> None:
    state_kind = str(manifest.get("stateKind") or "").strip().lower()
    current_layer_start = _request_int(request, "layerStart")
    current_layer_end = _request_int(request, "layerEnd")
    state_layer_start = _optional_int(manifest.get("layerStart"))
    state_layer_end = _optional_int(manifest.get("layerEnd"))
    if state_kind == "activation_state":
        if current_layer_start is not None and state_layer_end is None:
            raise ValueError(f"{error_prefix} producer layerEnd is missing.")
        if state_layer_start is None:
            raise ValueError(f"{error_prefix} producer layerStart is missing.")
        if current_layer_start is not None and state_layer_end != current_layer_start:
            raise ValueError(
                f"{error_prefix} producer layerEnd must match current layerStart."
            )
        if (
            state_layer_start is not None
            and state_layer_end is not None
            and state_layer_end <= state_layer_start
        ):
            raise ValueError(f"{error_prefix} producer layer range is invalid.")
        return
    if state_kind == "decode_state":
        if state_layer_start is None:
            raise ValueError(f"{error_prefix} producer layerStart is missing.")
        if state_layer_end is None:
            raise ValueError(f"{error_prefix} producer layerEnd is missing.")
        if state_layer_end <= state_layer_start:
            raise ValueError(f"{error_prefix} producer layer range is invalid.")
        same_layer_range = (
            (current_layer_start is None or state_layer_start == current_layer_start)
            and (current_layer_end is None or state_layer_end == current_layer_end)
        )
        if same_layer_range:
            return
        if current_layer_start is not None and state_layer_end == current_layer_start:
            return
        producer_action = str(manifest.get("producedByAction") or "").strip()
        current_action = str(request.get("action") or "").strip()
        prefill_to_first_decode = (
            producer_action == "process_prefill"
            and current_action == "process_decode"
            and current_layer_start == 0
            and current_layer_end is not None
            and state_layer_start >= current_layer_end
        )
        if prefill_to_first_decode:
            return
        if current_layer_start is not None and state_layer_start != current_layer_start:
            raise ValueError(f"{error_prefix} layerStart does not match request.")
        if current_layer_end is not None and state_layer_end != current_layer_end:
            raise ValueError(f"{error_prefix} layerEnd does not match request.")


def _build_validated_execution_context(
    request: Mapping[str, Any],
    *,
    validated_input_state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    action = str(request.get("action") or "").strip()
    execution_workspace = _validated_execution_workspace(request)
    context: dict[str, Any] = {
        "schemaVersion": 1,
        "abi": PATCHED_EXECUTION_CONTEXT_ABI,
        "action": action,
        "sessionId": _request_string(request, "sessionId"),
        "modelId": _request_string(request, "modelId"),
        "layerStart": _request_int(request, "layerStart"),
        "layerEnd": _request_int(request, "layerEnd"),
        "tokenStart": _request_int(request, "tokenStart"),
        "tokenEnd": _request_int(request, "tokenEnd"),
        "requiresFinalOutput": bool(request.get("requiresFinalOutput")),
        "expectedOutputKind": str(request.get("expectedOutputKind") or "").strip(),
        "executionWorkspace": execution_workspace,
    }
    io_targets = _validated_io_targets(
        request,
        execution_workspace=execution_workspace,
        validated_input_state=validated_input_state,
    )
    if io_targets is not None:
        context["ioTargets"] = io_targets
    assignment_artifact = _validated_assignment_artifact(request)
    if assignment_artifact is not None:
        context["assignmentArtifact"] = assignment_artifact
    model_artifact = _validated_model_artifact(request)
    if model_artifact is not None:
        context["modelArtifact"] = model_artifact
    managed_runtime = _validated_managed_runtime(request)
    if managed_runtime is not None:
        context["managedRuntime"] = managed_runtime
    if validated_input_state is not None:
        context["inputState"] = dict(validated_input_state)
    shard_spec = request.get("shardSpec")
    if isinstance(shard_spec, Mapping):
        context["shardSpec"] = dict(shard_spec)
    frame = request.get("frame")
    if isinstance(frame, Mapping):
        context["frame"] = dict(frame)
    output_contract = request.get("outputContract")
    if isinstance(output_contract, Mapping):
        context["outputContract"] = dict(output_contract)
    return context


def _validated_io_targets(
    request: Mapping[str, Any],
    *,
    execution_workspace: Mapping[str, Any],
    validated_input_state: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    action = str(request.get("action") or "").strip().lower()
    if action not in {"process_prefill", "process_decode"}:
        return {
            "schemaVersion": 1,
            "abi": PATCHED_IO_TARGETS_ABI,
            "inputPayloadPath": None,
            "outputPayloadPath": None,
            "outputStateManifestPath": None,
            "outputStateFilePath": None,
            "inputStateFilePath": None,
        }
    outputs_dir = _workspace_dir(
        execution_workspace,
        field_name="outputsDir",
        fallback_field="root",
    )
    state_dir = _workspace_dir(
        execution_workspace,
        field_name="stateFilesDir",
        fallback_field="root",
    )
    input_payload_path = None
    input_payload_file = request.get("inputPayloadFile")
    if isinstance(input_payload_file, Mapping):
        raw_input_path = str(input_payload_file.get("path") or "").strip()
        if raw_input_path:
            input_payload_path = _workspace_path(
                raw_input_path,
                base_dir=_workspace_dir(
                    execution_workspace,
                    field_name="inputsDir",
                    fallback_field="root",
                ),
            )
    expected_output_payload_path = str(
        request.get("expectedOutputPayloadPath") or ""
    ).strip()
    if not expected_output_payload_path:
        raise ValueError(
            "CAI patched executor host expectedOutputPayloadPath is missing."
        )
    output_payload_path = _workspace_path(
        expected_output_payload_path,
        base_dir=outputs_dir,
    )
    if not _path_is_within(output_payload_path, outputs_dir):
        raise ValueError(
            "CAI patched executor host expected output path must stay within executionWorkspace.outputsDir."
        )
    output_kind = str(request.get("expectedOutputKind") or "").strip().lower()
    output_state_file_path = None
    output_state_manifest_path = None
    if output_kind in {"activation_state", "decode_state"}:
        output_state_manifest_path = output_payload_path
        output_state_file_path = _derive_output_state_file_path(
            output_payload_path=output_payload_path,
            state_files_dir=state_dir,
        )
    input_state_file_path = None
    if isinstance(validated_input_state, Mapping):
        state_file = validated_input_state.get("stateFile")
        if isinstance(state_file, Mapping):
            raw_state_path = str(state_file.get("path") or "").strip()
            if raw_state_path:
                input_state_file_path = _workspace_path(
                    raw_state_path,
                    base_dir=state_dir,
                )
    payload: dict[str, Any] = {
        "schemaVersion": 1,
        "abi": PATCHED_IO_TARGETS_ABI,
        "inputPayloadPath": str(input_payload_path) if input_payload_path else None,
        "outputPayloadPath": str(output_payload_path),
        "outputStateManifestPath": (
            str(output_state_manifest_path) if output_state_manifest_path else None
        ),
        "outputStateFilePath": (
            str(output_state_file_path) if output_state_file_path else None
        ),
        "inputStateFilePath": (
            str(input_state_file_path) if input_state_file_path else None
        ),
    }
    return {
        key: value
        for key, value in payload.items()
        if value not in (None, "")
    }


def _derive_output_state_file_path(
    *,
    output_payload_path: Path,
    state_files_dir: Path,
) -> Path:
    stem = output_payload_path.stem or "output"
    suffix = output_payload_path.suffix or ".bin"
    return (state_files_dir / f"{stem}.state{suffix}").resolve()


def _validate_state_file_target(
    request: Mapping[str, Any],
    *,
    state_file_path: Path,
) -> None:
    validated_context = request.get("validatedExecutionContext")
    if not isinstance(validated_context, Mapping):
        return
    io_targets = validated_context.get("ioTargets")
    if not isinstance(io_targets, Mapping):
        return
    raw_target = str(io_targets.get("outputStateFilePath") or "").strip()
    if not raw_target:
        return
    expected_path = Path(raw_target).expanduser().resolve()
    if state_file_path.resolve() != expected_path:
        raise ValueError(
            "CAI patched executor host engine state file path does not match ioTargets.outputStateFilePath."
        )


def _validate_output_payload_target(
    request: Mapping[str, Any],
    *,
    output_path: Path,
) -> None:
    validated_context = request.get("validatedExecutionContext")
    if not isinstance(validated_context, Mapping):
        return
    io_targets = validated_context.get("ioTargets")
    if not isinstance(io_targets, Mapping):
        return
    raw_target = str(io_targets.get("outputPayloadPath") or "").strip()
    if not raw_target:
        return
    expected_path = Path(raw_target).expanduser().resolve()
    if output_path.resolve() != expected_path:
        raise ValueError(
            "CAI patched executor host engine output path does not match ioTargets.outputPayloadPath."
        )


def _validated_execution_workspace(request: Mapping[str, Any]) -> dict[str, Any]:
    workspace = request.get("executionWorkspace")
    if not isinstance(workspace, Mapping):
        raise ValueError(
            "CAI patched executor host executionWorkspace is missing."
        )
    root = _workspace_dir(workspace, field_name="root", fallback_field="root")
    inputs_dir = _workspace_dir(
        workspace,
        field_name="inputsDir",
        fallback_field="root",
    )
    outputs_dir = _workspace_dir(
        workspace,
        field_name="outputsDir",
        fallback_field="root",
    )
    state_dir = _workspace_dir(
        workspace,
        field_name="stateFilesDir",
        fallback_field="root",
    )
    manifest_path_raw = str(workspace.get("manifestPath") or "").strip()
    manifest_path = (
        _workspace_path(manifest_path_raw, base_dir=root)
        if manifest_path_raw
        else None
    )
    if manifest_path is not None and not _path_is_within(manifest_path, root):
        raise ValueError(
            "CAI patched executor host execution workspace manifest path must stay within workspace root."
        )
    return {
        "schemaVersion": 1,
        "abi": "cai-llama-cpp-execution-workspace-v1",
        "root": str(root),
        "inputsDir": str(inputs_dir),
        "outputsDir": str(outputs_dir),
        "stateFilesDir": str(state_dir),
        "manifestPath": str(manifest_path) if manifest_path is not None else None,
    }


def _validated_assignment_artifact(
    request: Mapping[str, Any],
) -> dict[str, Any] | None:
    artifact = None
    local_artifact_resolution = request.get("localArtifactResolution")
    if isinstance(local_artifact_resolution, Mapping):
        local_assignment = local_artifact_resolution.get("assignmentArtifact")
        if isinstance(local_assignment, Mapping):
            artifact = local_assignment
    if artifact is None:
        request_assignment = request.get("assignmentArtifact")
        if isinstance(request_assignment, Mapping):
            artifact = request_assignment
    if not isinstance(artifact, Mapping):
        return None
    local_path = _artifact_local_path(
        artifact,
        field_prefix="CAI patched executor host assignment artifact",
    )
    coverage = resolve_assignment_artifact_coverage(
        artifact,
        error_prefix="CAI patched executor host assignment artifact",
    )
    chunk_ranges = resolve_assignment_artifact_chunk_ranges(
        artifact,
        error_prefix="CAI patched executor host assignment artifact",
    )
    layer_start = _optional_int(artifact.get("layerStart"))
    layer_end = _optional_int(artifact.get("layerEnd"))
    request_layer_start = _request_int(request, "layerStart")
    request_layer_end = _request_int(request, "layerEnd")
    if request_layer_start is not None and layer_start is not None:
        if request_layer_start != layer_start:
            raise ValueError(
                "CAI patched executor host assignment artifact layerStart mismatch."
            )
    if request_layer_end is not None and layer_end is not None:
        if request_layer_end != layer_end:
            raise ValueError(
                "CAI patched executor host assignment artifact layerEnd mismatch."
            )
    validate_assignment_artifact_chunk_layer_coverage(
        artifact,
        layer_start=(
            request_layer_start if request_layer_start is not None else layer_start
        ),
        layer_end=request_layer_end if request_layer_end is not None else layer_end,
        error_prefix="CAI patched executor host assignment artifact",
    )
    normalized: dict[str, Any] = {
        "artifactId": str(artifact.get("artifactId") or "").strip() or None,
        "source": str(artifact.get("source") or "").strip() or None,
        "localPath": str(local_path),
        "sizeBytes": int(local_path.stat().st_size),
        "layerStart": layer_start,
        "layerEnd": layer_end,
        "expectedDigest": str(artifact.get("expectedDigest") or "").strip() or None,
    }
    if coverage is not None:
        normalized["coverage"] = {
            "abi": coverage.abi,
            "materializationMode": coverage.materialization_mode,
            "artifactSizeBytes": coverage.artifact_size_bytes,
            "coveredByteCount": coverage.covered_byte_count,
            "coveredRangeCount": coverage.covered_range_count,
            "zeroFilledOutsideCoveredRanges": (
                coverage.zero_filled_outside_covered_ranges
            ),
        }
    if chunk_ranges:
        normalized["chunkRanges"] = [
            {
                "chunkId": item.chunk_id,
                "offsetBytes": item.offset_bytes,
                "sizeBytes": item.size_bytes,
                "sha256Hex": item.sha256_hex,
                "layerStart": item.layer_start,
                "layerEnd": item.layer_end,
                "tensorNames": list(item.tensor_names),
            }
            for item in chunk_ranges
        ]
    return normalized


def _validated_model_artifact(
    request: Mapping[str, Any],
) -> dict[str, Any] | None:
    local_artifact_resolution = request.get("localArtifactResolution")
    if not isinstance(local_artifact_resolution, Mapping):
        return None
    artifact = local_artifact_resolution.get("modelArtifact")
    if not isinstance(artifact, Mapping):
        return None
    local_path = _artifact_local_path(
        artifact,
        field_prefix="CAI patched executor host model artifact",
    )
    return {
        "artifactId": str(artifact.get("artifactId") or "").strip() or None,
        "source": str(artifact.get("source") or "").strip() or None,
        "localPath": str(local_path),
        "sizeBytes": int(local_path.stat().st_size),
        "expectedSizeBytes": _mapping_optional_non_negative_int(
            artifact.get("expectedSizeBytes")
        ),
    }


def _validated_managed_runtime(
    request: Mapping[str, Any],
) -> dict[str, Any] | None:
    managed_runtime = resolve_managed_llama_cpp_runtime(request)
    if managed_runtime is None:
        return None
    payload: dict[str, Any] = {
        "abi": managed_runtime.abi,
        "platform": managed_runtime.platform,
        "repoRoot": (
            str(managed_runtime.repo_root.resolve())
            if managed_runtime.repo_root is not None
            else None
        ),
        "runtimeRoot": (
            str(managed_runtime.runtime_root.resolve())
            if managed_runtime.runtime_root is not None
            else None
        ),
        "modelId": managed_runtime.model_id,
    }
    if managed_runtime.session_paths is not None:
        payload["sessionPaths"] = {
            "root": str(managed_runtime.session_paths.root.resolve()),
            "stateDir": str(managed_runtime.session_paths.state_dir.resolve()),
            "cacheDir": str(managed_runtime.session_paths.cache_dir.resolve()),
            "logsDir": str(managed_runtime.session_paths.logs_dir.resolve()),
            "stdoutLog": str(managed_runtime.session_paths.stdout_log.resolve()),
            "stderrLog": str(managed_runtime.session_paths.stderr_log.resolve()),
        }
    llama_cpp: dict[str, Any] = {}
    if managed_runtime.llama_server.path is not None:
        llama_cpp["llamaServerPath"] = str(
            managed_runtime.llama_server.path.resolve()
        )
    if managed_runtime.llama_server.args:
        llama_cpp["llamaServerArgs"] = list(managed_runtime.llama_server.args)
    if managed_runtime.rpc_server.path is not None:
        llama_cpp["rpcServerPath"] = str(managed_runtime.rpc_server.path.resolve())
    if managed_runtime.rpc_server.args:
        llama_cpp["rpcServerArgs"] = list(managed_runtime.rpc_server.args)
    if llama_cpp:
        payload["llamaCpp"] = llama_cpp
    return payload


def _artifact_local_path(
    artifact: Mapping[str, Any],
    *,
    field_prefix: str,
) -> Path:
    raw_path = str(artifact.get("localPath") or "").strip()
    if not raw_path:
        raise ValueError(f"{field_prefix} localPath is missing.")
    path = Path(raw_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise ValueError(f"{field_prefix} localPath is unavailable.")
    return path


def _request_int(request: Mapping[str, Any], field_name: str) -> int | None:
    for payload in (
        request,
        request.get("shardSpec"),
        request.get("frame"),
    ):
        if not isinstance(payload, Mapping):
            continue
        value = _optional_int(payload.get(field_name))
        if value is not None:
            return value
    return None


def _request_string(request: Mapping[str, Any], field_name: str) -> str | None:
    for payload in (
        request,
        request.get("shardSpec"),
        request.get("frame"),
    ):
        if not isinstance(payload, Mapping):
            continue
        value = str(payload.get(field_name) or "").strip()
        if value:
            return value
    return None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mapping_optional_non_negative_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return _mapping_non_negative_int(
        value,
        field_name="CAI patched executor host optional integer",
    )


def _patched_engine_persistent_enabled() -> bool:
    raw = str(os.getenv(CAI_LLM_PATCHED_ENGINE_PERSISTENT_ENV) or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _patched_engine_timeout_seconds() -> float:
    for name in (
        CAI_LLM_PATCHED_ENGINE_TIMEOUT_ENV,
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


def _persistent_patched_engine_client(
    command: Sequence[str],
    *,
    cwd: str | None,
) -> PersistentPatchedEngineClient:
    timeout_sec = _patched_engine_timeout_seconds()
    key = (tuple(str(item) for item in command), cwd, timeout_sec)
    existing = _PERSISTENT_PATCHED_ENGINE_CLIENTS.get(key)
    if existing is not None and existing.alive:
        return existing
    if existing is not None:
        existing.close(kill=True)
        _PERSISTENT_PATCHED_ENGINE_CLIENTS.pop(key, None)
    client = PersistentPatchedEngineClient(
        command,
        cwd=cwd,
        timeout_sec=timeout_sec,
    )
    _PERSISTENT_PATCHED_ENGINE_CLIENTS[key] = client
    return client


def _engine_cwd(request: Mapping[str, Any]) -> str | None:
    managed_runtime = resolve_managed_llama_cpp_runtime(request)
    if managed_runtime is None:
        return None
    if managed_runtime.repo_root is not None:
        return str(managed_runtime.repo_root)
    if managed_runtime.runtime_root is not None:
        return str(managed_runtime.runtime_root)
    return None


def _workspace_dir(
    workspace: Mapping[str, Any],
    *,
    field_name: str,
    fallback_field: str,
) -> Path:
    raw = str(
        workspace.get(field_name) or workspace.get(fallback_field) or ""
    ).strip()
    if not raw:
        raise ValueError(
            "CAI patched executor host executionWorkspace paths are incomplete."
        )
    return Path(raw).expanduser().resolve()


def _workspace_path(raw_path: str, *, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except Exception:
        return False
    return True


def _mapping_non_negative_int(value: Any, *, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} is invalid.") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return parsed


atexit.register(reset_patched_executor_host_clients)


if __name__ == "__main__":
    raise SystemExit(main())

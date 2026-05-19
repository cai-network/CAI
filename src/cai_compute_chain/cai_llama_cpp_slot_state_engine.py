# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .cai_owned_runtime import (
    LLAMA_CPP_EXTERNAL_SHARD_GENERATION_PROBE_ABI,
    LLAMA_CPP_EXTERNAL_SHARD_GENERATION_PROBE_ACTION,
    LLAMA_CPP_EXTERNAL_SHARD_HANDOFF_ABI,
    LLAMA_CPP_EXTERNAL_SHARD_REQUIRED_CAPABILITIES,
    build_llama_cpp_external_shard_patch_boundary,
)
from .cai_llama_cpp_native_engine_contract import (
    build_native_engine_process_response,
    decode_native_engine_input_payload,
)
from .cai_llama_cpp_backend_runtime import (
    build_llama_server_command,
    choose_loopback_port,
    resolve_managed_llama_cpp_runtime,
    resolve_request_local_artifact_path,
    wait_for_llama_server_ready,
    windows_subprocess_creation_flags,
    windows_subprocess_startupinfo,
)


SLOT_STATE_ENGINE_ID = "llama.cpp-slot-state"
SLOT_STATE_ENGINE_VERSION = "llama.cpp-slot-state/0.1"
SLOT_STATE_PAYLOAD_ABI = "cai-llama-cpp-slot-state-payload-v1"
PRODUCTION_STATE_CONTRACT_ABI = "cai-llama-cpp-production-state-contract-v1"

SERVER_URL_ENV = "CAI_LLM_SHARD_SLOT_SERVER_URL"
STATE_DIR_ENV = "CAI_LLM_SHARD_SLOT_STATE_DIR"
SLOT_ID_ENV = "CAI_LLM_SHARD_SLOT_ID"
TIMEOUT_ENV = "CAI_LLM_SHARD_SLOT_TIMEOUT_SEC"
DECODE_TOKENS_ENV = "CAI_LLM_SHARD_SLOT_DECODE_TOKENS"


@dataclass
class ManagedSlotServerHandle:
    key: str
    server_url: str
    state_dir: Path
    process: subprocess.Popen[str]
    stdout_log: Path | None
    stderr_log: Path | None
    stdout_stream: Any | None = None
    stderr_stream: Any | None = None


_MANAGED_SLOT_SERVERS: dict[str, ManagedSlotServerHandle] = {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a CAI llama.cpp slot-state engine. This is a real KV/slot "
            "state handoff step for smoke/live validation, not a production "
            "layer-shard backend."
        ),
    )
    parser.add_argument("--jsonl", action="store_true")
    parser.add_argument("--server-url", default=os.getenv(SERVER_URL_ENV, ""))
    parser.add_argument("--state-dir", default=os.getenv(STATE_DIR_ENV, ""))
    parser.add_argument(
        "--slot-id",
        type=int,
        default=int(os.getenv(SLOT_ID_ENV, "0") or 0),
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=float(os.getenv(TIMEOUT_ENV, "120") or 120),
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=int(os.getenv(DECODE_TOKENS_ENV, "8") or 8),
    )
    args = parser.parse_args(argv)
    config = SlotStateEngineConfig(
        server_url=args.server_url,
        state_dir=Path(args.state_dir) if str(args.state_dir or "").strip() else None,
        slot_id=max(0, int(args.slot_id or 0)),
        timeout_sec=max(0.1, float(args.timeout_sec or 120.0)),
        decode_tokens=max(1, int(args.decode_tokens or 8)),
    )
    if bool(args.jsonl):
        return _jsonl_loop(config)
    try:
        request = json.loads(sys.stdin.read() or "{}")
        response = handle_slot_state_engine_request(request, config=config)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), flush=True)
        return 1
    print(json.dumps(response, sort_keys=True), flush=True)
    return 0


class SlotStateEngineConfig:
    def __init__(
        self,
        *,
        server_url: str,
        state_dir: Path | None,
        slot_id: int,
        timeout_sec: float,
        decode_tokens: int,
    ) -> None:
        self.server_url = _normalize_server_url(server_url)
        self.state_dir = state_dir
        self.slot_id = max(0, int(slot_id or 0))
        self.timeout_sec = max(0.1, float(timeout_sec or 120.0))
        self.decode_tokens = max(1, int(decode_tokens or 8))


def handle_slot_state_engine_request(
    request: Mapping[str, Any],
    *,
    config: SlotStateEngineConfig,
) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        raise ValueError("CAI llama.cpp slot-state request must be an object.")
    if str(request.get("abi") or "").strip() != LLAMA_CPP_EXTERNAL_SHARD_HANDOFF_ABI:
        raise ValueError("CAI llama.cpp slot-state request ABI is unsupported.")
    config = _resolved_slot_state_config(request, config)
    action = str(request.get("action") or "").strip()
    if action == "load_shard":
        return _load_shard(config)
    if action == LLAMA_CPP_EXTERNAL_SHARD_GENERATION_PROBE_ACTION:
        return _probe_generation(request, config)
    if action == "process_prefill":
        return _process_prefill(request, config)
    if action == "process_decode":
        return _process_decode(request, config)
    if action == "finalize":
        _release_managed_slot_server(request)
        return {
            "status": "ok",
            "metrics": {
                "backendMode": SLOT_STATE_ENGINE_ID,
                "backendFinalized": True,
            },
        }
    raise ValueError(f"CAI llama.cpp slot-state action is unsupported: {action}")


def _jsonl_loop(config: SlotStateEngineConfig) -> int:
    for line in sys.stdin:
        try:
            request = json.loads(line or "{}")
            response = handle_slot_state_engine_request(request, config=config)
        except Exception as exc:
            response = {"status": "error", "error": str(exc)}
        print(json.dumps(response, sort_keys=True), flush=True)
    return 0


def _load_shard(config: SlotStateEngineConfig) -> dict[str, Any]:
    slots = _get_json(config, "/slots")
    if not isinstance(slots, list):
        raise ValueError("llama.cpp slot-state engine could not read /slots.")
    slot_ids = {
        int(item.get("id"))
        for item in slots
        if isinstance(item, Mapping) and _optional_int(item.get("id")) is not None
    }
    if config.slot_id not in slot_ids:
        raise ValueError(f"llama.cpp slot id is unavailable: {config.slot_id}")
    capabilities = [
        *LLAMA_CPP_EXTERNAL_SHARD_REQUIRED_CAPABILITIES,
        "real_decode_state",
        "llama_cpp_slot_save_restore",
    ]
    return {
        "status": "ready",
        "capabilities": capabilities,
        "patchBoundary": build_llama_cpp_external_shard_patch_boundary(
            backend="llama.cpp-patched",
            backend_version=SLOT_STATE_ENGINE_VERSION,
            patch_id="cai-llama-cpp-slot-state-handoff",
            runner_protocol_version="slot-state-0.1",
            capabilities=capabilities,
            extra_metadata={
                "mode": SLOT_STATE_ENGINE_ID,
                "productionStateContract": _slot_state_contract(),
                "productionReady": False,
                "productionReadyReason": (
                    "Slot-state handoff uses real llama.cpp KV state, but it "
                    "does not execute CAI layer-range activation shards."
                ),
            },
        ),
        "metrics": {
            "backendLoaded": True,
            "backendMode": SLOT_STATE_ENGINE_ID,
            "backendCapabilities": capabilities,
            "slotId": config.slot_id,
        },
    }


def _probe_generation(
    request: Mapping[str, Any],
    config: SlotStateEngineConfig,
) -> dict[str, Any]:
    probe = request.get("generationProbe")
    if not isinstance(probe, Mapping):
        raise ValueError("llama.cpp slot-state generationProbe is missing.")
    prompt = str(probe.get("prompt") or "").strip()
    model_id = str(probe.get("modelId") or "").strip()
    max_tokens = max(1, int(_optional_int(probe.get("maxTokens"), 8) or 8))
    response = _completion(
        config,
        {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": float(probe.get("temperature") or 0.0),
            "cache_prompt": True,
        },
    )
    text = str(response.get("content") or "")
    output_tokens = _optional_int(response.get("tokens_predicted"), None)
    return {
        "status": "ok",
        "generationProbe": {
            "schemaVersion": 1,
            "abi": LLAMA_CPP_EXTERNAL_SHARD_GENERATION_PROBE_ABI,
            "ready": bool(text),
            "modelId": model_id or None,
            "outputText": text,
            "outputTokenCount": max(0, int(output_tokens or 0)),
            "realModelExecution": True,
        },
        "metrics": {
            "backendAction": LLAMA_CPP_EXTERNAL_SHARD_GENERATION_PROBE_ACTION,
            "backendMode": SLOT_STATE_ENGINE_ID,
            "generationProbeReady": bool(text),
            "slotId": config.slot_id,
        },
    }


def _process_prefill(
    request: Mapping[str, Any],
    config: SlotStateEngineConfig,
) -> dict[str, Any]:
    frame = _frame(request)
    prompt_bytes = _payload_bytes(request)
    prompt = prompt_bytes.decode("utf-8", errors="replace")
    completion = _completion(
        config,
        {
            "prompt": prompt,
            "n_predict": 1,
            "temperature": 0.0,
            "cache_prompt": True,
        },
    )
    state_bytes, save_audit = _save_slot_state(config, frame)
    output_payload = json.dumps(
        {
            "schemaVersion": 1,
            "abi": SLOT_STATE_PAYLOAD_ABI,
            "promptUtf8": prompt,
            "slotStateBase64": base64.b64encode(state_bytes).decode("ascii"),
            "slotStateSha256Hex": hashlib.sha256(state_bytes).hexdigest(),
            "slotId": config.slot_id,
            "nSaved": save_audit.get("n_saved"),
            "prefillContentPreview": str(completion.get("content") or "")[:120],
        },
        sort_keys=True,
    ).encode("utf-8")
    return _frame_response(
        request,
        frame,
        output_payload,
        metrics={
            "backendAction": "process_prefill",
            "backendMode": SLOT_STATE_ENGINE_ID,
            "inputTokenCount": max(0, int(save_audit.get("n_saved") or 0)),
            "outputTokenCount": 0,
            "slotStateBytes": len(state_bytes),
            "slotId": config.slot_id,
        },
        native_artifact_kind="model",
    )


def _process_decode(
    request: Mapping[str, Any],
    config: SlotStateEngineConfig,
) -> dict[str, Any]:
    frame = _frame(request)
    state_payload = _slot_state_payload(_payload_bytes(request))
    state_bytes = base64.b64decode(
        str(state_payload.get("slotStateBase64") or "").encode("ascii"),
        validate=True,
    )
    declared_hash = str(state_payload.get("slotStateSha256Hex") or "").strip().lower()
    actual_hash = hashlib.sha256(state_bytes).hexdigest()
    if declared_hash and declared_hash != actual_hash:
        raise ValueError("llama.cpp slot-state payload hash mismatch.")
    restore_audit = _restore_slot_state(config, frame, state_bytes)
    prompt = str(state_payload.get("promptUtf8") or "")
    completion = _completion(
        config,
        {
            "prompt": prompt,
            "n_predict": config.decode_tokens,
            "temperature": 0.0,
            "cache_prompt": True,
        },
    )
    output_text = str(completion.get("content") or "")
    output_payload = output_text.encode("utf-8")
    return _frame_response(
        request,
        frame,
        output_payload,
        metrics={
            "backendAction": "process_decode",
            "backendMode": SLOT_STATE_ENGINE_ID,
            "inputTokenCount": max(0, int(restore_audit.get("n_restored") or 0)),
            "outputTokenCount": max(
                0,
                int(_optional_int(completion.get("tokens_predicted"), 0) or 0),
            ),
            "slotStateBytes": len(state_bytes),
            "slotId": config.slot_id,
        },
        native_artifact_kind="model",
    )


def _frame_response(
    request: Mapping[str, Any],
    frame: Mapping[str, Any],
    output_payload: bytes,
    *,
    metrics: Mapping[str, Any],
    native_artifact_kind: str | None = None,
) -> dict[str, Any]:
    return build_native_engine_process_response(
        request,
        output_payload,
        metrics=metrics,
        artifact_kind=native_artifact_kind,
        fallback_mode="full_model" if native_artifact_kind == "model" else None,
        error_prefix="llama.cpp slot-state",
    )


def _save_slot_state(
    config: SlotStateEngineConfig,
    frame: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    filename = _state_filename(frame, suffix="prefill.bin")
    response = _post_json(
        config,
        f"/slots/{config.slot_id}?action=save",
        {"filename": filename},
    )
    path = _state_path(config, filename)
    data = path.read_bytes()
    _safe_unlink(path)
    return data, response


def _restore_slot_state(
    config: SlotStateEngineConfig,
    frame: Mapping[str, Any],
    state_bytes: bytes,
) -> dict[str, Any]:
    filename = _state_filename(frame, suffix="decode.bin")
    path = _state_path(config, filename)
    path.write_bytes(bytes(state_bytes or b""))
    try:
        return _post_json(
            config,
            f"/slots/{config.slot_id}?action=restore",
            {"filename": filename},
        )
    finally:
        _safe_unlink(path)


def _completion(config: SlotStateEngineConfig, payload: Mapping[str, Any]) -> dict[str, Any]:
    response = _post_json(config, "/completion", payload)
    if not isinstance(response, dict):
        raise ValueError("llama.cpp completion response is invalid.")
    return response


def _get_json(config: SlotStateEngineConfig, path: str) -> Any:
    with urlopen(_url(config, path), timeout=config.timeout_sec) as response:
        return json.loads(response.read().decode("utf-8") or "null")


def _post_json(
    config: SlotStateEngineConfig,
    path: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    request = Request(
        _url(config, path),
        data=json.dumps(dict(payload), sort_keys=True).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=config.timeout_sec) as response:
        parsed = json.loads(response.read().decode("utf-8") or "{}")
    if not isinstance(parsed, dict):
        raise ValueError("llama.cpp JSON response is invalid.")
    return parsed


def _url(config: SlotStateEngineConfig, path: str) -> str:
    if not config.server_url:
        raise ValueError(
            f"{SERVER_URL_ENV} or --server-url is required for slot-state engine."
        )
    clean_path = "/" + str(path or "").lstrip("/")
    return config.server_url + clean_path


def _normalize_server_url(value: str) -> str:
    clean = str(value or "").strip().rstrip("/")
    if not clean:
        return ""
    parsed = urlparse(clean)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("llama.cpp slot-state server URL is invalid.")
    host = (parsed.hostname or "").strip().lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("llama.cpp slot-state server URL must be loopback/local.")
    return clean


def _state_path(config: SlotStateEngineConfig, filename: str) -> Path:
    if config.state_dir is None:
        raise ValueError(f"{STATE_DIR_ENV} or --state-dir is required.")
    config.state_dir.mkdir(parents=True, exist_ok=True)
    path = (config.state_dir / filename).resolve()
    root = config.state_dir.resolve()
    if root not in path.parents and path != root:
        raise ValueError("llama.cpp slot-state path escaped state directory.")
    return path


def _state_filename(frame: Mapping[str, Any], *, suffix: str) -> str:
    batch_id = str(frame.get("batchId") or f"batch-{time.time_ns()}").strip()
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in batch_id)
    return f"cai-slot-{safe}-{suffix}"


def _frame(request: Mapping[str, Any]) -> Mapping[str, Any]:
    frame = request.get("frame")
    if not isinstance(frame, Mapping):
        raise ValueError("llama.cpp slot-state frame is missing.")
    return frame


def _payload_bytes(request: Mapping[str, Any]) -> bytes:
    return decode_native_engine_input_payload(
        request,
        error_prefix="llama.cpp slot-state",
    )


def _slot_state_payload(payload: bytes) -> Mapping[str, Any]:
    try:
        parsed = json.loads(bytes(payload or b"").decode("utf-8"))
    except Exception as exc:
        raise ValueError("llama.cpp slot-state payload is invalid JSON.") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError("llama.cpp slot-state payload must be an object.")
    if str(parsed.get("abi") or "").strip() != SLOT_STATE_PAYLOAD_ABI:
        raise ValueError("llama.cpp slot-state payload ABI is unsupported.")
    if not str(parsed.get("slotStateBase64") or "").strip():
        raise ValueError("llama.cpp slot-state payload is missing state bytes.")
    return parsed


def _local_file_contract(request: Mapping[str, Any]) -> Mapping[str, Any] | None:
    contract = request.get("localFileContract")
    return contract if isinstance(contract, Mapping) else None


def reset_slot_state_engine_managed_servers() -> None:
    handles = list(_MANAGED_SLOT_SERVERS.values())
    _MANAGED_SLOT_SERVERS.clear()
    for handle in handles:
        _terminate_managed_slot_server(handle)


def _slot_state_contract() -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "abi": PRODUCTION_STATE_CONTRACT_ABI,
        "activationStateFormat": "llama.cpp-slot-kv-cache-v1/full-model-prefill",
        "decodeStateFormat": "llama.cpp-slot-kv-cache-v1/slot-save-restore",
        "modelExecutionBackend": SLOT_STATE_ENGINE_ID,
        "tensorEncoding": "raw-le",
        "shardExecutionMode": "full_model_replica",
        "fullModelReplicaRequired": True,
        "activationStateIsSynthetic": True,
        "decodeStateIsSynthetic": False,
        "extraMetadata": {
            "productionReady": False,
            "reason": "This engine transfers real KV state but not layer activations.",
        },
    }


def _optional_int(value: object, default: int | None = None) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _resolved_slot_state_config(
    request: Mapping[str, Any],
    config: SlotStateEngineConfig,
) -> SlotStateEngineConfig:
    if config.server_url:
        return config
    managed_runtime = resolve_managed_llama_cpp_runtime(request)
    if managed_runtime is None:
        return config
    session_paths = managed_runtime.session_paths
    if session_paths is None:
        raise ValueError("llama.cpp slot-state managedRuntime sessionPaths are missing.")
    model_path = resolve_request_local_artifact_path(request, preferred_kind="model")
    if model_path is None:
        raise ValueError("llama.cpp slot-state managed runtime model artifact is unavailable.")
    handle = _managed_slot_server_handle(
        request,
        managed_runtime=managed_runtime,
        model_path=model_path,
        timeout_sec=config.timeout_sec,
    )
    return SlotStateEngineConfig(
        server_url=handle.server_url,
        state_dir=session_paths.state_dir,
        slot_id=config.slot_id,
        timeout_sec=config.timeout_sec,
        decode_tokens=config.decode_tokens,
    )


def _managed_slot_server_handle(
    request: Mapping[str, Any],
    *,
    managed_runtime,
    model_path: Path,
    timeout_sec: float,
) -> ManagedSlotServerHandle:
    session_paths = managed_runtime.session_paths
    if session_paths is None:
        raise ValueError("llama.cpp slot-state managedRuntime sessionPaths are missing.")
    key = "|".join(
        [
            str(managed_runtime.llama_server.path or ""),
            str(model_path.resolve()),
            str(session_paths.state_dir.resolve()),
        ]
    )
    existing = _MANAGED_SLOT_SERVERS.get(key)
    if existing is not None and existing.process.poll() is None:
        return existing
    if existing is not None:
        _terminate_managed_slot_server(existing)
        _MANAGED_SLOT_SERVERS.pop(key, None)

    port = choose_loopback_port()
    server_url = f"http://127.0.0.1:{port}"
    session_paths.state_dir.mkdir(parents=True, exist_ok=True)
    session_paths.logs_dir.mkdir(parents=True, exist_ok=True)
    command = build_llama_server_command(
        managed_runtime,
        model_path=model_path,
        host="127.0.0.1",
        port=port,
        slot_save_path=session_paths.state_dir,
        parallel_slots=1,
    )
    stdout_handle = session_paths.stdout_log.open("a", encoding="utf-8")
    stderr_handle = session_paths.stderr_log.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            cwd=str(session_paths.root),
            creationflags=windows_subprocess_creation_flags(),
            startupinfo=windows_subprocess_startupinfo(),
        )
    except Exception:
        stdout_handle.close()
        stderr_handle.close()
        raise
    handle = ManagedSlotServerHandle(
        key=key,
        server_url=server_url,
        state_dir=session_paths.state_dir,
        process=process,
        stdout_log=session_paths.stdout_log,
        stderr_log=session_paths.stderr_log,
        stdout_stream=stdout_handle,
        stderr_stream=stderr_handle,
    )
    try:
        wait_for_llama_server_ready(server_url, timeout_sec=timeout_sec, probe_path="/slots")
    except Exception:
        _terminate_managed_slot_server(handle)
        raise
    _MANAGED_SLOT_SERVERS[key] = handle
    return handle


def _release_managed_slot_server(request: Mapping[str, Any]) -> None:
    managed_runtime = resolve_managed_llama_cpp_runtime(request)
    if managed_runtime is None or managed_runtime.session_paths is None:
        return
    model_path = resolve_request_local_artifact_path(request, preferred_kind="model")
    if model_path is None:
        return
    key = "|".join(
        [
            str(managed_runtime.llama_server.path or ""),
            str(model_path.resolve()),
            str(managed_runtime.session_paths.state_dir.resolve()),
        ]
    )
    handle = _MANAGED_SLOT_SERVERS.pop(key, None)
    if handle is not None:
        _terminate_managed_slot_server(handle)


def _terminate_managed_slot_server(handle: ManagedSlotServerHandle) -> None:
    process = handle.process
    try:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=3)
    except Exception:
        try:
            if process.poll() is None:
                process.kill()
        except Exception:
            pass
    for stream in (
        handle.stdout_stream,
        handle.stderr_stream,
        process.stdout,
        process.stderr,
    ):
        try:
            if stream is not None:
                stream.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

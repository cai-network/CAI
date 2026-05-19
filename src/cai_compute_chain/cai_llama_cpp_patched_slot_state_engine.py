# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .cai_llama_cpp_assignment_artifact_engine import ASSIGNMENT_EXECUTOR_REQUEST_ABI
from .cai_llama_cpp_patched_executor_host import (
    PATCHED_EXECUTION_CONTEXT_ABI,
    PATCHED_IO_TARGETS_ABI,
)
from .cai_llama_cpp_real_state_contract import build_real_state_manifest_payload
from .cai_llama_cpp_slot_state_engine import (
    SLOT_STATE_ENGINE_ID,
    SlotStateEngineConfig,
    handle_slot_state_engine_request,
)
from .cai_owned_runtime import (
    LLAMA_CPP_EXTERNAL_SHARD_HANDOFF_ABI,
    LLAMA_CPP_EXTERNAL_SHARD_LOCAL_FILE_IO_ABI,
)


PATCHED_SLOT_STATE_ENGINE_ID = "patched_slot_state_engine"
PATCHED_SLOT_STATE_ENGINE_VERSION = "patched-slot-state-engine/0.1"
PATCHED_SLOT_STATE_SESSION_ABI = "cai-llama-cpp-patched-slot-session-v1"


_ENGINE_REQUEST_COUNT = 0


def reset_patched_slot_state_engine_runtime_state() -> None:
    global _ENGINE_REQUEST_COUNT
    _ENGINE_REQUEST_COUNT = 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a CAI lower patched-engine adapter that consumes "
            "validatedExecutionContext from cai_llama_cpp_patched_executor_host "
            "and executes the request through the real slot-state backend."
        ),
    )
    parser.add_argument("--jsonl", action="store_true")
    parser.add_argument("--server-url", default="")
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--slot-id", type=int, default=0)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--decode-tokens", type=int, default=8)
    args = parser.parse_args(argv)
    config = SlotStateEngineConfig(
        server_url=str(args.server_url or ""),
        state_dir=(
            Path(str(args.state_dir)).expanduser().resolve()
            if str(args.state_dir or "").strip()
            else None
        ),
        slot_id=max(0, int(args.slot_id or 0)),
        timeout_sec=max(0.1, float(args.timeout_sec or 120.0)),
        decode_tokens=max(1, int(args.decode_tokens or 8)),
    )
    if bool(args.jsonl):
        return _jsonl_loop(config)
    try:
        request = json.loads(sys.stdin.read() or "{}")
        response = handle_patched_slot_state_engine_request(
            request,
            config=config,
        )
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), flush=True)
        return 1
    print(json.dumps(response, sort_keys=True), flush=True)
    return 0


def _jsonl_loop(config: SlotStateEngineConfig) -> int:
    for line in sys.stdin:
        try:
            request = json.loads(line or "{}")
            response = handle_patched_slot_state_engine_request(
                request,
                config=config,
            )
        except Exception as exc:
            response = {"status": "error", "error": str(exc)}
        print(json.dumps(response, sort_keys=True), flush=True)
    return 0


def handle_patched_slot_state_engine_request(
    request: Mapping[str, Any],
    *,
    config: SlotStateEngineConfig,
) -> dict[str, Any]:
    engine_runtime_metrics = _next_engine_runtime_metrics()
    if not isinstance(request, Mapping):
        raise ValueError("CAI patched slot-state engine request must be an object.")
    if str(request.get("abi") or "").strip() != ASSIGNMENT_EXECUTOR_REQUEST_ABI:
        raise ValueError("CAI patched slot-state engine ABI is unsupported.")
    action = str(request.get("action") or "").strip()
    if action not in {"load_shard", "process_prefill", "process_decode", "finalize"}:
        raise ValueError(
            f"CAI patched slot-state engine action is unsupported: {action}"
        )
    context = _validated_execution_context(request)
    session_manifest_path = _session_manifest_path(context)
    context_signature = _context_signature(context)
    _validate_session_manifest_if_present(
        session_manifest_path,
        context=context,
        context_signature=context_signature,
        action=action,
    )
    slot_state_request = _build_slot_state_request(request, context)
    slot_state_response = handle_slot_state_engine_request(
        slot_state_request,
        config=config,
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
        return {
            "status": str(slot_state_response.get("status") or "ok").strip() or "ok",
            "realModelExecution": True,
            "metrics": {
                "engineBackendMode": PATCHED_SLOT_STATE_ENGINE_ID,
                "contextAbi": PATCHED_EXECUTION_CONTEXT_ABI,
                "sessionManifestPath": str(session_manifest_path),
                **engine_runtime_metrics,
                "slotStateMetrics": dict(slot_state_response.get("metrics") or {})
                if isinstance(slot_state_response.get("metrics"), Mapping)
                else {},
            },
        }
    expected_output_kind = str(context.get("expectedOutputKind") or "").strip().lower()
    if expected_output_kind == "final_output":
        response = _final_output_response(
            slot_state_response,
            slot_state_response_metrics=slot_state_response.get("metrics"),
            session_manifest_path=session_manifest_path,
            engine_runtime_metrics=engine_runtime_metrics,
        )
        _write_session_manifest(
            session_manifest_path,
            context=context,
            context_signature=context_signature,
            last_action="process_decode",
            last_output_kind="final_output",
        )
        return response
    if expected_output_kind != "decode_state":
        raise ValueError(
            "CAI patched slot-state engine only supports decode_state non-final output."
        )
    response = _real_state_response(
        request,
        context,
        slot_state_response,
        session_manifest_path=session_manifest_path,
        engine_runtime_metrics=engine_runtime_metrics,
    )
    _write_session_manifest(
        session_manifest_path,
        context=context,
        context_signature=context_signature,
        last_action=action,
        last_output_kind="decode_state",
        last_state_manifest_path=(
            str((response.get("outputPayloadFile") or {}).get("path") or "").strip()
            if isinstance(response.get("outputPayloadFile"), Mapping)
            else None
        ),
    )
    return response


def _validated_execution_context(request: Mapping[str, Any]) -> Mapping[str, Any]:
    context = request.get("validatedExecutionContext")
    if not isinstance(context, Mapping):
        raise ValueError(
            "CAI patched slot-state engine validatedExecutionContext is missing."
        )
    if str(context.get("abi") or "").strip() != PATCHED_EXECUTION_CONTEXT_ABI:
        raise ValueError(
            "CAI patched slot-state engine validatedExecutionContext ABI is unsupported."
        )
    return context


def _build_slot_state_request(
    request: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    action = str(request.get("action") or "").strip()
    workspace = _execution_workspace(context)
    io_targets = _io_targets(context)
    frame = context.get("frame")
    shard_spec = context.get("shardSpec")
    output_contract = context.get("outputContract")
    if not isinstance(frame, Mapping) or not isinstance(shard_spec, Mapping):
        raise ValueError(
            "CAI patched slot-state engine frame/shardSpec are missing from validatedExecutionContext."
        )
    local_artifact_resolution = _local_artifact_resolution_from_context(context)
    payload_file = None
    response_output_path = None
    if action == "process_prefill":
        payload_file = _mapping_copy(request.get("inputPayloadFile"))
        if payload_file is None:
            raise ValueError(
                "CAI patched slot-state engine inputPayloadFile is missing for process_prefill."
            )
        response_output_path = (
            str(io_targets["outputStateFilePath"])
            if io_targets.get("outputStateFilePath") is not None
            else str((workspace["stateFilesDir"] / "slot-state-prefill.bin").resolve())
        )
    elif action == "process_decode":
        validated_input_state = request.get("validatedInputState")
        payload_file = _payload_file_from_validated_input_state(validated_input_state)
        response_output_path = (
            str(io_targets["outputPayloadPath"])
            if io_targets.get("outputPayloadPath") is not None
            else str((workspace["outputsDir"] / "slot-state-final.bin").resolve())
        )
    handoff_request: dict[str, Any] = {
        "schemaVersion": 1,
        "abi": LLAMA_CPP_EXTERNAL_SHARD_HANDOFF_ABI,
        "action": action,
        "backend": "llama.cpp-patched",
        "frame": dict(frame),
        "shardSpec": dict(shard_spec),
        "payloadFile": payload_file,
        "payloadSha256Hex": (
            str(payload_file.get("sha256Hex") or "").strip()
            if isinstance(payload_file, Mapping)
            else None
        ),
        "outputContract": dict(output_contract) if isinstance(output_contract, Mapping) else None,
        "managedRuntime": _mapping_copy(context.get("managedRuntime")),
        "localArtifactResolution": local_artifact_resolution,
        "localFileContract": (
            {
                "schemaVersion": 1,
                "abi": LLAMA_CPP_EXTERNAL_SHARD_LOCAL_FILE_IO_ABI,
                "ioRoot": str(workspace["root"].resolve()),
                "responseOutputPath": str(Path(response_output_path).resolve()),
            }
            if response_output_path
            else None
        ),
        "productionRequirements": _mapping_copy(request.get("productionRequirements")),
    }
    return {
        key: value
        for key, value in handoff_request.items()
        if value not in (None, {})
    }


def _final_output_response(
    slot_state_response: Mapping[str, Any],
    *,
    slot_state_response_metrics: Any,
    session_manifest_path: Path,
    engine_runtime_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "status": "ok",
        "outputKind": "final_output",
        "realModelExecution": True,
        "metrics": {
            "engineBackendMode": PATCHED_SLOT_STATE_ENGINE_ID,
            "slotStateBackendMode": SLOT_STATE_ENGINE_ID,
            "sessionManifestPath": str(session_manifest_path),
            **dict(engine_runtime_metrics),
            "slotStateMetrics": (
                dict(slot_state_response_metrics)
                if isinstance(slot_state_response_metrics, Mapping)
                else {}
            ),
        },
    }
    output_file = slot_state_response.get("outputPayloadFile")
    if isinstance(output_file, Mapping):
        response["outputPayloadFile"] = dict(output_file)
    else:
        output_base64 = str(slot_state_response.get("outputPayloadBase64") or "").strip()
        if output_base64:
            response["outputPayloadBase64"] = output_base64
    output_hash = str(slot_state_response.get("outputPayloadSha256Hex") or "").strip()
    if output_hash:
        response["outputPayloadSha256Hex"] = output_hash
    return response


def _real_state_response(
    request: Mapping[str, Any],
    context: Mapping[str, Any],
    slot_state_response: Mapping[str, Any],
    *,
    session_manifest_path: Path,
    engine_runtime_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    workspace = _execution_workspace(context)
    io_targets = _io_targets(context)
    slot_state_file = slot_state_response.get("outputPayloadFile")
    if not isinstance(slot_state_file, Mapping):
        raise ValueError(
            "CAI patched slot-state engine expected slot-state outputPayloadFile."
        )
    state_file_path = Path(str(slot_state_file.get("path") or "")).expanduser().resolve()
    if not state_file_path.exists() or not state_file_path.is_file():
        raise ValueError("CAI patched slot-state engine slot-state output file is unavailable.")
    expected_output_path = str(
        (io_targets.get("outputStateManifestPath") or "")
        if isinstance(io_targets, Mapping)
        else ""
    ).strip() or str(request.get("expectedOutputPayloadPath") or "").strip()
    if not expected_output_path:
        raise ValueError(
            "CAI patched slot-state engine expectedOutputPayloadPath is missing."
        )
    manifest_path = Path(expected_output_path).expanduser().resolve()
    if not _path_is_within(manifest_path, workspace["outputsDir"]):
        raise ValueError(
            "CAI patched slot-state engine manifest output path must stay within outputsDir."
        )
    state_payload_bytes = state_file_path.read_bytes()
    manifest_payload = build_real_state_manifest_payload(
        output_kind="decode_state",
        action=str(request.get("action") or "").strip(),
        model_id=str(context.get("modelId") or "").strip(),
        session_id=str(context.get("sessionId") or "").strip(),
        layer_start=_optional_int(context.get("layerStart")),
        layer_end=_optional_int(context.get("layerEnd")),
        token_start=_optional_int(context.get("tokenStart")),
        token_end=_optional_int(context.get("tokenEnd")),
        state_file_path=state_file_path,
        metadata={
            "sourceEngine": PATCHED_SLOT_STATE_ENGINE_ID,
            "wrappedBackend": SLOT_STATE_ENGINE_ID,
            "slotStatePayloadAbi": "cai-llama-cpp-slot-state-payload-v1",
            "slotStatePayloadSha256Hex": hashlib.sha256(state_payload_bytes).hexdigest(),
        },
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(manifest_payload)
    manifest_hash = hashlib.sha256(manifest_payload).hexdigest()
    return {
        "status": "ok",
        "outputKind": "decode_state",
        "realModelExecution": True,
        "outputPayloadFile": {
            "path": str(manifest_path),
            "sizeBytes": len(manifest_payload),
            "sha256Hex": manifest_hash,
        },
        "outputPayloadSha256Hex": manifest_hash,
        "metrics": {
            "engineBackendMode": PATCHED_SLOT_STATE_ENGINE_ID,
            "slotStateBackendMode": SLOT_STATE_ENGINE_ID,
            "wrappedStateBytes": len(state_payload_bytes),
            "sessionManifestPath": str(session_manifest_path),
            **dict(engine_runtime_metrics),
            "slotStateMetrics": dict(slot_state_response.get("metrics") or {})
            if isinstance(slot_state_response.get("metrics"), Mapping)
            else {},
        },
    }


def _payload_file_from_validated_input_state(
    validated_input_state: Any,
) -> dict[str, Any]:
    if not isinstance(validated_input_state, Mapping):
        raise ValueError(
            "CAI patched slot-state engine validatedInputState is missing for process_decode."
        )
    state_file = validated_input_state.get("stateFile")
    if not isinstance(state_file, Mapping):
        raise ValueError(
            "CAI patched slot-state engine validatedInputState.stateFile is missing."
        )
    return {
        "path": str(state_file.get("path") or "").strip(),
        "sizeBytes": int(state_file.get("sizeBytes") or 0),
        "sha256Hex": str(state_file.get("sha256Hex") or "").strip(),
    }


def _local_artifact_resolution_from_context(
    context: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    assignment = context.get("assignmentArtifact")
    if isinstance(assignment, Mapping):
        payload["assignmentArtifact"] = dict(assignment)
    model = context.get("modelArtifact")
    if isinstance(model, Mapping):
        payload["modelArtifact"] = dict(model)
    return payload


def _execution_workspace(context: Mapping[str, Any]) -> dict[str, Path]:
    workspace = context.get("executionWorkspace")
    if not isinstance(workspace, Mapping):
        raise ValueError(
            "CAI patched slot-state engine executionWorkspace is missing from validatedExecutionContext."
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


def _io_targets(context: Mapping[str, Any]) -> dict[str, Path | None]:
    payload = context.get("ioTargets")
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(
            "CAI patched slot-state engine ioTargets must be an object."
        )
    if str(payload.get("abi") or "").strip() != PATCHED_IO_TARGETS_ABI:
        raise ValueError(
            "CAI patched slot-state engine ioTargets ABI is unsupported."
        )
    return {
        "inputPayloadPath": _optional_path_from_mapping(payload, "inputPayloadPath"),
        "outputPayloadPath": _optional_path_from_mapping(payload, "outputPayloadPath"),
        "outputStateManifestPath": _optional_path_from_mapping(
            payload,
            "outputStateManifestPath",
        ),
        "outputStateFilePath": _optional_path_from_mapping(
            payload,
            "outputStateFilePath",
        ),
        "inputStateFilePath": _optional_path_from_mapping(payload, "inputStateFilePath"),
    }


def _path_from_mapping(payload: Mapping[str, Any], field_name: str) -> Path:
    raw = str(payload.get(field_name) or "").strip()
    if not raw:
        raise ValueError(
            f"CAI patched slot-state engine workspace field is missing: {field_name}"
        )
    return Path(raw).expanduser().resolve()


def _optional_path_from_mapping(payload: Mapping[str, Any], field_name: str) -> Path | None:
    raw = str(payload.get(field_name) or "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _mapping_copy(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except Exception:
        return False
    return True


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _next_engine_runtime_metrics() -> dict[str, Any]:
    global _ENGINE_REQUEST_COUNT
    _ENGINE_REQUEST_COUNT += 1
    return {
        "engineProcessId": int(os.getpid()),
        "engineProcessRequestCount": int(_ENGINE_REQUEST_COUNT),
    }


def _session_manifest_path(context: Mapping[str, Any]) -> Path:
    workspace = _execution_workspace(context)
    return (workspace["stateFilesDir"] / "patched-slot-state-session.json").resolve()


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
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_session_manifest_if_present(
    manifest_path: Path,
    *,
    context: Mapping[str, Any],
    context_signature: str,
    action: str,
) -> None:
    if not manifest_path.exists():
        return
    try:
        parsed = json.loads(manifest_path.read_text(encoding="utf-8") or "{}")
    except Exception as exc:
        raise ValueError(
            "CAI patched slot-state engine session manifest is invalid."
        ) from exc
    if not isinstance(parsed, Mapping):
        raise ValueError("CAI patched slot-state engine session manifest is invalid.")
    if str(parsed.get("abi") or "").strip() != PATCHED_SLOT_STATE_SESSION_ABI:
        raise ValueError(
            "CAI patched slot-state engine session manifest ABI is unsupported."
        )
    if action == "load_shard":
        return
    declared_signature = str(parsed.get("contextSignature") or "").strip()
    if declared_signature and declared_signature != context_signature:
        raise ValueError(
            "CAI patched slot-state engine detected execution context drift."
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
        "abi": PATCHED_SLOT_STATE_SESSION_ABI,
        "sessionId": str(context.get("sessionId") or "").strip(),
        "modelId": str(context.get("modelId") or "").strip(),
        "layerStart": _optional_int(context.get("layerStart")),
        "layerEnd": _optional_int(context.get("layerEnd")),
        "tokenStart": _optional_int(context.get("tokenStart")),
        "tokenEnd": _optional_int(context.get("tokenEnd")),
        "expectedOutputKind": str(context.get("expectedOutputKind") or "").strip(),
        "contextSignature": context_signature,
        "lastAction": str(last_action or "").strip(),
    }
    if last_output_kind:
        payload["lastOutputKind"] = str(last_output_kind)
    if last_state_manifest_path:
        payload["lastStateManifestPath"] = str(last_state_manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


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
        "sizeBytes": _optional_int(artifact.get("sizeBytes")),
        "layerStart": _optional_int(artifact.get("layerStart")),
        "layerEnd": _optional_int(artifact.get("layerEnd")),
    }
    coverage = artifact.get("coverage")
    if isinstance(coverage, Mapping):
        payload["coverage"] = {
            "materializationMode": str(coverage.get("materializationMode") or "").strip()
            or None,
            "artifactSizeBytes": _optional_int(coverage.get("artifactSizeBytes")),
            "coveredByteCount": _optional_int(coverage.get("coveredByteCount")),
            "coveredRangeCount": _optional_int(coverage.get("coveredRangeCount")),
        }
    if include_chunks:
        chunk_ranges = artifact.get("chunkRanges")
        if isinstance(chunk_ranges, list):
            payload["chunkRanges"] = [
                {
                    "chunkId": str(item.get("chunkId") or "").strip() or None,
                    "offsetBytes": _optional_int(item.get("offsetBytes"))
                    if isinstance(item, Mapping)
                    else None,
                    "sizeBytes": _optional_int(item.get("sizeBytes"))
                    if isinstance(item, Mapping)
                    else None,
                    "sha256Hex": (
                        str(item.get("sha256Hex") or "").strip().lower() or None
                        if isinstance(item, Mapping)
                        else None
                    ),
                    "layerStart": _optional_int(item.get("layerStart"))
                    if isinstance(item, Mapping)
                    else None,
                    "layerEnd": _optional_int(item.get("layerEnd"))
                    if isinstance(item, Mapping)
                    else None,
                    "tensorNames": (
                        list(item.get("tensorNames") or [])
                        if isinstance(item, Mapping)
                        else []
                    ),
                }
                for item in chunk_ranges
            ]
    return payload


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

from .cai_llama_cpp_native_engine_contract import (
    resolve_assignment_artifact_chunk_ranges,
    resolve_assignment_artifact_coverage,
)
from .cai_llama_cpp_patched_binary_executor import PATCHED_BINARY_REQUEST_ABI
from .cai_llama_cpp_real_state_contract import build_real_state_manifest_payload


REFERENCE_PATCHED_BINARY_ID = "reference_patched_binary"
REFERENCE_PATCHED_BINARY_VERSION = "reference-patched-binary/0.1"
REFERENCE_PATCHED_BINARY_SESSION_ABI = "cai-llama-cpp-reference-patched-session-v1"
REFERENCE_PATCHED_PREFILL_RECEIPT_ABI = (
    "cai-llama-cpp-reference-prefill-receipt-v1"
)


@dataclass(frozen=True)
class ReferenceLoadedShardContext:
    session_id: str
    model_id: str
    layer_start: int | None
    layer_end: int | None
    assignment_local_path: Path
    assignment_source: str
    assignment_size_bytes: int
    artifact_digest: str
    digest_source: str
    bytes_read: int
    chunk_count: int
    coverage_mode: str | None
    covered_byte_count: int | None
    covered_range_count: int | None
    tensor_names: tuple[str, ...]
    prepared_shard_path: Path
    prepared_shard_sha256_hex: str
    prepared_shard_size_bytes: int
    prepared_shard_source_mode: str
    prefill_receipt_path: Path
    session_state_path: Path


_CALL_COUNT = 0
_LOADED_SHARD_SESSIONS: dict[str, ReferenceLoadedShardContext] = {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reference CAI patched binary command. It validates assignment shard "
            "artifacts, persists resident shard context across load_shard/process_* "
            "steps, and emits real-state manifests, but it does not execute real "
            "llama.cpp layer math."
        )
    )
    parser.add_argument("--jsonl", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.jsonl):
        return _jsonl_loop()
    try:
        request = json.loads(sys.stdin.read() or "{}")
        response = handle_reference_patched_binary_request(request)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), flush=True)
        return 1
    print(json.dumps(response, sort_keys=True), flush=True)
    return 0


def _jsonl_loop() -> int:
    for line in sys.stdin:
        try:
            request = json.loads(line or "{}")
            response = handle_reference_patched_binary_request(request)
        except Exception as exc:
            response = {"status": "error", "error": str(exc)}
        print(json.dumps(response, sort_keys=True), flush=True)
    return 0


def handle_reference_patched_binary_request(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    global _CALL_COUNT
    _CALL_COUNT += 1
    if not isinstance(request, Mapping):
        raise ValueError("CAI reference patched binary request must be an object.")
    if str(request.get("abi") or "").strip() != PATCHED_BINARY_REQUEST_ABI:
        raise ValueError("CAI reference patched binary ABI is unsupported.")
    action = str(request.get("action") or "").strip()
    if action not in {"load_shard", "process_prefill", "process_decode", "finalize"}:
        raise ValueError(
            f"CAI reference patched binary action is unsupported: {action}"
        )
    if action == "load_shard":
        return _handle_load_shard(request)
    if action == "process_prefill":
        return _handle_process_prefill(request)
    if action == "process_decode":
        return _handle_process_decode(request)
    return _handle_finalize(request)


def _handle_load_shard(request: Mapping[str, Any]) -> dict[str, Any]:
    context = _prepare_loaded_shard_context(request)
    _LOADED_SHARD_SESSIONS[context.session_id] = context
    _write_session_state(context)
    return {
        "status": "ready",
        "realModelExecution": True,
        "realLayerExecution": False,
        "metrics": _reference_metrics(
            action="load_shard",
            context=context,
            session_context_source="prepared",
            extra={
                "referenceBinarySessionPrepared": True,
                "referenceBinarySessionStatePath": str(context.session_state_path),
                "referencePreparedShardWindowPrepared": True,
            },
        ),
    }


def _handle_process_prefill(request: Mapping[str, Any]) -> dict[str, Any]:
    context, context_source = _load_shard_context(request)
    _ensure_prepared_shard_window(context)
    input_payload = _read_input_payload(request)
    prefill_receipt = _write_prefill_execution_receipt(
        request,
        context=context,
        input_payload=input_payload,
    )
    output_kind = _expected_output_kind(request)
    if output_kind == "final_output":
        return _final_output_response(
            request,
            context=context,
            session_context_source=context_source,
            state_input_digest=_input_state_digest(request),
            prefill_execution_digest=str(
                prefill_receipt.get("prefillExecutionDigest") or ""
            ).strip()
            or None,
        )
    return _real_state_response(
        request,
        context=context,
        session_context_source=context_source,
        source_payload=input_payload,
        state_kind=output_kind,
        state_label="prefill",
        state_metadata={
            "prefillExecutionDigest": str(
                prefill_receipt.get("prefillExecutionDigest") or ""
            ).strip()
            or None,
            "prefillReceiptPath": str(context.prefill_receipt_path),
        },
        extra_metrics={
            "referenceBinaryPrefillExecutionDigest": str(
                prefill_receipt.get("prefillExecutionDigest") or ""
            ).strip()
            or None,
            "referenceBinaryPrefillReceiptPath": str(context.prefill_receipt_path),
            "referenceBinaryPrefillReceiptWritten": True,
        },
    )


def _handle_process_decode(request: Mapping[str, Any]) -> dict[str, Any]:
    context, context_source = _load_shard_context(request)
    _ensure_prepared_shard_window(context)
    prefill_receipt = _read_prefill_execution_receipt(context.prefill_receipt_path)
    input_payload = _read_input_payload(request)
    _validate_decode_input_against_prefill_receipt(
        request,
        context=context,
        prefill_receipt=prefill_receipt,
    )
    output_kind = _expected_output_kind(request)
    if output_kind == "final_output" or bool(request.get("requiresFinalOutput")):
        return _final_output_response(
            request,
            context=context,
            session_context_source=context_source,
            state_input_digest=_input_state_digest(request),
            prefill_execution_digest=str(
                prefill_receipt.get("prefillExecutionDigest") or ""
            ).strip()
            or None,
        )
    return _real_state_response(
        request,
        context=context,
        session_context_source=context_source,
        source_payload=input_payload,
        state_kind=output_kind,
        state_label="decode",
        state_metadata={
            "prefillExecutionDigest": str(
                prefill_receipt.get("prefillExecutionDigest") or ""
            ).strip()
            or None,
            "prefillReceiptPath": str(context.prefill_receipt_path),
        },
        extra_metrics={
            "referenceBinaryPrefillExecutionDigest": str(
                prefill_receipt.get("prefillExecutionDigest") or ""
            ).strip()
            or None,
            "referenceBinaryPrefillReceiptPath": str(context.prefill_receipt_path),
            "referenceBinaryPrefillReceiptValidated": True,
        },
    )


def _handle_finalize(request: Mapping[str, Any]) -> dict[str, Any]:
    session_id = _require_string(request, "sessionId")
    context = _LOADED_SHARD_SESSIONS.pop(session_id, None)
    session_state_path = _session_state_path(request)
    context_source = "memory" if context is not None else None
    if context is None and session_state_path.exists():
        context = _read_session_state(session_state_path)
        context_source = "disk"
    _safe_unlink(session_state_path)
    if context is not None:
        _safe_unlink(context.prepared_shard_path)
        _safe_unlink(context.prefill_receipt_path)
    return {
        "status": "ok",
        "realModelExecution": True,
        "realLayerExecution": False,
        "metrics": _reference_metrics(
            action="finalize",
            context=context,
            session_context_source=context_source or "none",
            extra={
                "referenceBinarySessionFinalized": True,
                "referenceBinarySessionStatePath": str(session_state_path),
                "referencePreparedShardWindowDeleted": True,
                "referenceBinaryPrefillReceiptDeleted": True,
            },
        ),
    }


def _prepare_loaded_shard_context(
    request: Mapping[str, Any],
) -> ReferenceLoadedShardContext:
    assignment = request.get("assignmentArtifact")
    if not isinstance(assignment, Mapping):
        raise ValueError("CAI reference patched binary assignmentArtifact is missing.")
    local_path = Path(_require_string(assignment, "localPath")).expanduser().resolve()
    if not local_path.exists() or not local_path.is_file():
        raise ValueError(
            "CAI reference patched binary assignmentArtifact path is unavailable."
        )
    coverage = resolve_assignment_artifact_coverage(
        assignment,
        error_prefix="CAI reference patched binary",
    )
    chunk_ranges = resolve_assignment_artifact_chunk_ranges(
        assignment,
        error_prefix="CAI reference patched binary",
    )
    artifact_digest, digest_source, bytes_read = _assignment_window_digest(
        local_path,
        chunk_ranges=chunk_ranges,
        expected_digest=str(assignment.get("expectedDigest") or "").strip() or None,
    )
    prepared_shard_path = _prepared_shard_path(
        request,
        artifact_digest=artifact_digest,
    )
    prepared_shard_sha256_hex, prepared_shard_size_bytes, prepared_shard_source_mode = (
        _materialize_prepared_shard_window(
            local_path,
            chunk_ranges=chunk_ranges,
            destination_path=prepared_shard_path,
        )
    )
    tensor_names = tuple(
        dict.fromkeys(
            tensor_name
            for item in chunk_ranges
            for tensor_name in item.tensor_names
            if str(tensor_name).strip()
        )
    )
    return ReferenceLoadedShardContext(
        session_id=_require_string(request, "sessionId"),
        model_id=_require_string(request, "modelId"),
        layer_start=_optional_int(request.get("layerStart")),
        layer_end=_optional_int(request.get("layerEnd")),
        assignment_local_path=local_path,
        assignment_source=_require_string(assignment, "source"),
        assignment_size_bytes=int(local_path.stat().st_size),
        artifact_digest=artifact_digest,
        digest_source=digest_source,
        bytes_read=bytes_read,
        chunk_count=len(chunk_ranges),
        coverage_mode=(
            str(coverage.materialization_mode).strip() if coverage is not None else None
        ),
        covered_byte_count=(
            int(coverage.covered_byte_count) if coverage is not None else None
        ),
        covered_range_count=(
            int(coverage.covered_range_count) if coverage is not None else None
        ),
        tensor_names=tensor_names,
        prepared_shard_path=prepared_shard_path,
        prepared_shard_sha256_hex=prepared_shard_sha256_hex,
        prepared_shard_size_bytes=prepared_shard_size_bytes,
        prepared_shard_source_mode=prepared_shard_source_mode,
        prefill_receipt_path=_prefill_receipt_path(request),
        session_state_path=_session_state_path(request),
    )


def _load_shard_context(
    request: Mapping[str, Any],
) -> tuple[ReferenceLoadedShardContext, str]:
    session_id = _require_string(request, "sessionId")
    existing = _LOADED_SHARD_SESSIONS.get(session_id)
    if existing is not None:
        _validate_request_matches_context(request, existing)
        return existing, "memory"
    session_state_path = _session_state_path(request)
    if session_state_path.exists():
        restored = _read_session_state(session_state_path)
        _validate_request_matches_context(request, restored)
        _LOADED_SHARD_SESSIONS[session_id] = restored
        return restored, "disk"
    raise ValueError(
        "CAI reference patched binary missing resident shard context. "
        "Call load_shard first."
    )


def _write_session_state(context: ReferenceLoadedShardContext) -> None:
    payload = {
        "schemaVersion": 1,
        "abi": REFERENCE_PATCHED_BINARY_SESSION_ABI,
        "sessionId": context.session_id,
        "modelId": context.model_id,
        "layerStart": context.layer_start,
        "layerEnd": context.layer_end,
        "assignmentArtifact": {
            "localPath": str(context.assignment_local_path),
            "source": context.assignment_source,
            "sizeBytes": context.assignment_size_bytes,
            "digest": context.artifact_digest,
            "digestSource": context.digest_source,
            "bytesRead": context.bytes_read,
            "chunkCount": context.chunk_count,
            "coverageMode": context.coverage_mode,
            "coveredByteCount": context.covered_byte_count,
            "coveredRangeCount": context.covered_range_count,
            "tensorNames": list(context.tensor_names),
            "preparedShardPath": str(context.prepared_shard_path),
            "preparedShardSha256Hex": context.prepared_shard_sha256_hex,
            "preparedShardSizeBytes": context.prepared_shard_size_bytes,
            "preparedShardSourceMode": context.prepared_shard_source_mode,
            "prefillReceiptPath": str(context.prefill_receipt_path),
        },
    }
    context.session_state_path.parent.mkdir(parents=True, exist_ok=True)
    context.session_state_path.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )


def _read_session_state(path: Path) -> ReferenceLoadedShardContext:
    try:
        payload = json.loads(path.read_text(encoding="utf-8") or "{}")
    except Exception as exc:
        raise ValueError(
            "CAI reference patched binary session state is invalid."
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("CAI reference patched binary session state is invalid.")
    if str(payload.get("abi") or "").strip() != REFERENCE_PATCHED_BINARY_SESSION_ABI:
        raise ValueError(
            "CAI reference patched binary session state ABI is unsupported."
        )
    artifact = payload.get("assignmentArtifact")
    if not isinstance(artifact, Mapping):
        raise ValueError(
            "CAI reference patched binary session state assignmentArtifact is missing."
        )
    assignment_local_path = Path(_require_string(artifact, "localPath")).expanduser().resolve()
    return ReferenceLoadedShardContext(
        session_id=_require_string(payload, "sessionId"),
        model_id=_require_string(payload, "modelId"),
        layer_start=_optional_int(payload.get("layerStart")),
        layer_end=_optional_int(payload.get("layerEnd")),
        assignment_local_path=assignment_local_path,
        assignment_source=_require_string(artifact, "source"),
        assignment_size_bytes=_require_int(artifact, "sizeBytes"),
        artifact_digest=_require_string(artifact, "digest"),
        digest_source=_require_string(artifact, "digestSource"),
        bytes_read=_require_int(artifact, "bytesRead"),
        chunk_count=_require_int(artifact, "chunkCount"),
        coverage_mode=str(artifact.get("coverageMode") or "").strip() or None,
        covered_byte_count=_optional_int(artifact.get("coveredByteCount")),
        covered_range_count=_optional_int(artifact.get("coveredRangeCount")),
        tensor_names=tuple(
            str(item).strip()
            for item in (artifact.get("tensorNames") or [])
            if str(item).strip()
        ),
        prepared_shard_path=Path(
            _require_string(artifact, "preparedShardPath")
        ).expanduser().resolve(),
        prepared_shard_sha256_hex=_require_string(artifact, "preparedShardSha256Hex"),
        prepared_shard_size_bytes=_require_int(artifact, "preparedShardSizeBytes"),
        prepared_shard_source_mode=_require_string(artifact, "preparedShardSourceMode"),
        prefill_receipt_path=Path(
            _require_string(artifact, "prefillReceiptPath")
        ).expanduser().resolve(),
        session_state_path=path,
    )


def _validate_request_matches_context(
    request: Mapping[str, Any],
    context: ReferenceLoadedShardContext,
) -> None:
    if _require_string(request, "sessionId") != context.session_id:
        raise ValueError("CAI reference patched binary sessionId drifted.")
    model_id = _require_string(request, "modelId")
    if model_id != context.model_id:
        raise ValueError("CAI reference patched binary modelId drifted.")
    if _optional_int(request.get("layerStart")) != context.layer_start:
        raise ValueError("CAI reference patched binary layerStart drifted.")
    if _optional_int(request.get("layerEnd")) != context.layer_end:
        raise ValueError("CAI reference patched binary layerEnd drifted.")
    assignment = request.get("assignmentArtifact")
    if isinstance(assignment, Mapping):
        local_path = Path(_require_string(assignment, "localPath")).expanduser().resolve()
        if local_path != context.assignment_local_path:
            raise ValueError(
                "CAI reference patched binary assignmentArtifact path drifted."
            )


def _real_state_response(
    request: Mapping[str, Any],
    *,
    context: ReferenceLoadedShardContext,
    session_context_source: str,
    source_payload: bytes,
    state_kind: str,
    state_label: str,
    state_metadata: Mapping[str, Any] | None = None,
    extra_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    io_targets = _io_targets(request)
    output_payload_path = Path(_require_string(io_targets, "outputPayloadPath")).resolve()
    output_state_file_path = Path(_require_string(io_targets, "outputStateFilePath")).resolve()
    state_payload = json.dumps(
        {
            "schemaVersion": 1,
            "referenceBinary": True,
            "productionReady": False,
            "stateLabel": state_label,
            "artifactDigest": context.artifact_digest,
            "preparedShardSha256Hex": context.prepared_shard_sha256_hex,
            "preparedShardSizeBytes": context.prepared_shard_size_bytes,
            "inputPayloadSha256Hex": hashlib.sha256(source_payload).hexdigest(),
            "inputStateDigest": _input_state_digest(request),
            "tensorNames": list(context.tensor_names),
            **(
                {
                    key: value
                    for key, value in dict(state_metadata).items()
                    if value not in (None, "")
                }
                if isinstance(state_metadata, Mapping)
                else {}
            ),
        },
        sort_keys=True,
    ).encode("utf-8")
    output_state_file_path.parent.mkdir(parents=True, exist_ok=True)
    output_state_file_path.write_bytes(state_payload)
    manifest_payload = build_real_state_manifest_payload(
        output_kind=state_kind,
        action=_require_string(request, "action"),
        model_id=context.model_id,
        session_id=context.session_id,
        layer_start=context.layer_start,
        layer_end=context.layer_end,
        token_start=_optional_int(request.get("tokenStart")),
        token_end=_optional_int(request.get("tokenEnd")),
        state_file_path=output_state_file_path,
        metadata={
            "referenceBinary": True,
            "productionReady": False,
            "artifactDigest": context.artifact_digest,
            "preparedShardSha256Hex": context.prepared_shard_sha256_hex,
            "preparedShardSizeBytes": context.prepared_shard_size_bytes,
            "sessionContextSource": session_context_source,
            **(
                {
                    key: value
                    for key, value in dict(state_metadata).items()
                    if value not in (None, "")
                }
                if isinstance(state_metadata, Mapping)
                else {}
            ),
        },
    )
    output_payload_path.parent.mkdir(parents=True, exist_ok=True)
    output_payload_path.write_bytes(manifest_payload)
    output_hash = hashlib.sha256(manifest_payload).hexdigest()
    return {
        "status": "ok",
        "outputKind": state_kind,
        "realModelExecution": True,
        "realLayerExecution": False,
        "outputPayloadFile": {
            "path": str(output_payload_path),
            "sizeBytes": len(manifest_payload),
            "sha256Hex": output_hash,
        },
        "outputPayloadSha256Hex": output_hash,
        "metrics": _reference_metrics(
            action=_require_string(request, "action"),
            context=context,
            session_context_source=session_context_source,
            extra={
                "referenceBinaryOutputStateKind": state_kind,
                "referenceBinaryOutputStateFilePath": str(output_state_file_path),
                "referenceBinarySessionStatePath": str(context.session_state_path),
                "referencePreparedShardWindowVerified": True,
                **(dict(extra_metrics) if isinstance(extra_metrics, Mapping) else {}),
            },
        ),
    }


def _final_output_response(
    request: Mapping[str, Any],
    *,
    context: ReferenceLoadedShardContext,
    session_context_source: str,
    state_input_digest: str | None,
    prefill_execution_digest: str | None = None,
) -> dict[str, Any]:
    io_targets = _io_targets(request)
    output_payload_path = Path(_require_string(io_targets, "outputPayloadPath")).resolve()
    output_payload = (
        "reference:"
        + context.prepared_shard_sha256_hex[:16]
        + ":"
        + (str(prefill_execution_digest or "")[:16] or "no-prefill")
        + ":"
        + (str(state_input_digest or "")[:16] or "no-state")
    ).encode("utf-8")
    output_payload_path.parent.mkdir(parents=True, exist_ok=True)
    output_payload_path.write_bytes(output_payload)
    output_hash = hashlib.sha256(output_payload).hexdigest()
    return {
        "status": "ok",
        "outputKind": "final_output",
        "realModelExecution": True,
        "realLayerExecution": False,
        "outputPayloadFile": {
            "path": str(output_payload_path),
            "sizeBytes": len(output_payload),
            "sha256Hex": output_hash,
        },
        "outputPayloadSha256Hex": output_hash,
        "metrics": _reference_metrics(
            action=_require_string(request, "action"),
            context=context,
            session_context_source=session_context_source,
            extra={
                "referenceBinaryFinalOutput": True,
                "referenceBinarySessionStatePath": str(context.session_state_path),
                "referencePreparedShardWindowVerified": True,
                "referenceBinaryPrefillExecutionDigest": (
                    str(prefill_execution_digest or "").strip() or None
                ),
                "referenceBinaryPrefillReceiptPath": str(context.prefill_receipt_path),
            },
        ),
    }


def _reference_metrics(
    *,
    action: str,
    context: ReferenceLoadedShardContext | None,
    session_context_source: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "binary": REFERENCE_PATCHED_BINARY_ID,
        "binaryVersion": REFERENCE_PATCHED_BINARY_VERSION,
        "callCount": int(_CALL_COUNT),
        "sessionContextSource": session_context_source,
        "productionReady": False,
        "realLayerExecution": False,
        "productionReadyReason": (
            "reference_patched_binary_prepares shard context and moves real-state "
            "files, but it does not execute real llama.cpp layer math yet."
        ),
    }
    if context is not None:
        metrics.update(
            {
                "referenceAssignmentArtifactPath": str(context.assignment_local_path),
                "referenceAssignmentArtifactDigest": context.artifact_digest,
                "referenceAssignmentDigestSource": context.digest_source,
                "referenceAssignmentArtifactBytesRead": context.bytes_read,
                "referenceAssignmentArtifactChunkCount": context.chunk_count,
                "referenceAssignmentArtifactCoverageMode": context.coverage_mode,
                "referenceAssignmentArtifactCoveredByteCount": context.covered_byte_count,
                "referenceAssignmentArtifactCoveredRangeCount": context.covered_range_count,
                "referenceAssignmentTensorNames": list(context.tensor_names),
                "referenceAssignmentTensorCount": len(context.tensor_names),
                "referencePreparedShardWindowPath": str(context.prepared_shard_path),
                "referencePreparedShardWindowSha256Hex": (
                    context.prepared_shard_sha256_hex
                ),
                "referencePreparedShardWindowSizeBytes": (
                    context.prepared_shard_size_bytes
                ),
                "referencePreparedShardWindowSourceMode": (
                    context.prepared_shard_source_mode
                ),
                "referenceBinaryPrefillReceiptPath": str(context.prefill_receipt_path),
            }
        )
    if extra is not None:
        metrics.update(dict(extra))
    return metrics


def _assignment_window_digest(
    path: Path,
    *,
    chunk_ranges: tuple[Any, ...],
    expected_digest: str | None,
) -> tuple[str, str, int]:
    if chunk_ranges:
        hasher = hashlib.sha256()
        bytes_read = 0
        with path.open("rb") as handle:
            for chunk in chunk_ranges:
                handle.seek(int(chunk.offset_bytes))
                payload = handle.read(int(chunk.size_bytes))
                if len(payload) != int(chunk.size_bytes):
                    raise ValueError(
                        "CAI reference patched binary could not read a full chunk range."
                    )
                payload_hash = hashlib.sha256(payload).hexdigest()
                expected_hash = str(chunk.sha256_hex or "").strip().lower()
                if expected_hash and payload_hash != expected_hash:
                    raise ValueError(
                        "CAI reference patched binary assignment chunk hash mismatch."
                    )
                hasher.update(str(chunk.chunk_id or "").encode("utf-8"))
                hasher.update(payload_hash.encode("ascii"))
                bytes_read += int(chunk.size_bytes)
        return f"chunk_ranges:{hasher.hexdigest()}", "chunk_ranges", bytes_read
    if expected_digest:
        return expected_digest, "expected_digest", 0
    raise ValueError(
        "CAI reference patched binary needs assignment chunkRanges or expectedDigest."
    )


def _session_state_path(request: Mapping[str, Any]) -> Path:
    workspace = request.get("executionWorkspace")
    if not isinstance(workspace, Mapping):
        raise ValueError("CAI reference patched binary executionWorkspace is missing.")
    state_dir = Path(_require_string(workspace, "stateFilesDir")).expanduser().resolve()
    session_key = hashlib.sha256(
        (
            _require_string(request, "sessionId")
            + ":"
            + _require_string(request, "modelId")
            + ":"
            + str(_optional_int(request.get("layerStart")))
            + ":"
            + str(_optional_int(request.get("layerEnd")))
        ).encode("utf-8")
    ).hexdigest()[:16]
    return (state_dir / f"reference-patched-binary-{session_key}.json").resolve()


def _prepared_shard_path(
    request: Mapping[str, Any],
    *,
    artifact_digest: str,
) -> Path:
    workspace = request.get("executionWorkspace")
    if not isinstance(workspace, Mapping):
        raise ValueError("CAI reference patched binary executionWorkspace is missing.")
    state_dir = Path(_require_string(workspace, "stateFilesDir")).expanduser().resolve()
    safe_digest = hashlib.sha256(artifact_digest.encode("utf-8")).hexdigest()[:16]
    return (state_dir / f"reference-prepared-shard-{safe_digest}.bin").resolve()


def _prefill_receipt_path(request: Mapping[str, Any]) -> Path:
    workspace = request.get("executionWorkspace")
    if not isinstance(workspace, Mapping):
        raise ValueError("CAI reference patched binary executionWorkspace is missing.")
    state_dir = Path(_require_string(workspace, "stateFilesDir")).expanduser().resolve()
    session_key = hashlib.sha256(
        (
            _require_string(request, "sessionId")
            + ":"
            + _require_string(request, "modelId")
            + ":prefill:"
            + str(_optional_int(request.get("layerStart")))
            + ":"
            + str(_optional_int(request.get("layerEnd")))
        ).encode("utf-8")
    ).hexdigest()[:16]
    return (state_dir / f"reference-prefill-receipt-{session_key}.json").resolve()


def _materialize_prepared_shard_window(
    assignment_path: Path,
    *,
    chunk_ranges: tuple[Any, ...],
    destination_path: Path,
) -> tuple[str, int, str]:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    bytes_written = 0
    source_mode = "chunk_ranges" if chunk_ranges else "full_artifact"
    with assignment_path.open("rb") as source, destination_path.open("wb") as target:
        if chunk_ranges:
            for chunk in chunk_ranges:
                source.seek(int(chunk.offset_bytes))
                payload = source.read(int(chunk.size_bytes))
                if len(payload) != int(chunk.size_bytes):
                    raise ValueError(
                        "CAI reference patched binary could not materialize full chunk range."
                    )
                target.write(payload)
                hasher.update(payload)
                bytes_written += len(payload)
        else:
            payload = source.read()
            target.write(payload)
            hasher.update(payload)
            bytes_written = len(payload)
    return hasher.hexdigest(), int(bytes_written), source_mode


def _ensure_prepared_shard_window(context: ReferenceLoadedShardContext) -> None:
    if not context.prepared_shard_path.exists() or not context.prepared_shard_path.is_file():
        raise ValueError(
            "CAI reference patched binary prepared shard window is unavailable."
        )
    actual_size = int(context.prepared_shard_path.stat().st_size)
    if actual_size != int(context.prepared_shard_size_bytes):
        raise ValueError(
            "CAI reference patched binary prepared shard window size mismatch."
        )
    actual_hash = hashlib.sha256(context.prepared_shard_path.read_bytes()).hexdigest()
    if actual_hash != context.prepared_shard_sha256_hex:
        raise ValueError(
            "CAI reference patched binary prepared shard window hash mismatch."
        )


def _write_prefill_execution_receipt(
    request: Mapping[str, Any],
    *,
    context: ReferenceLoadedShardContext,
    input_payload: bytes,
) -> dict[str, Any]:
    input_payload_hash = hashlib.sha256(input_payload).hexdigest()
    execution_digest = hashlib.sha256(
        (
            context.prepared_shard_sha256_hex
            + ":"
            + input_payload_hash
            + ":"
            + str(context.layer_start)
            + ":"
            + str(context.layer_end)
            + ":"
            + str(_optional_int(request.get("tokenStart")))
            + ":"
            + str(_optional_int(request.get("tokenEnd")))
        ).encode("utf-8")
    ).hexdigest()
    payload = {
        "schemaVersion": 1,
        "abi": REFERENCE_PATCHED_PREFILL_RECEIPT_ABI,
        "sessionId": context.session_id,
        "modelId": context.model_id,
        "layerStart": context.layer_start,
        "layerEnd": context.layer_end,
        "tokenStart": _optional_int(request.get("tokenStart")),
        "tokenEnd": _optional_int(request.get("tokenEnd")),
        "preparedShardPath": str(context.prepared_shard_path),
        "preparedShardSha256Hex": context.prepared_shard_sha256_hex,
        "preparedShardSizeBytes": context.prepared_shard_size_bytes,
        "inputPayloadSha256Hex": input_payload_hash,
        "prefillExecutionDigest": execution_digest,
    }
    context.prefill_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    context.prefill_receipt_path.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def _read_prefill_execution_receipt(path: Path) -> Mapping[str, Any]:
    if not path.exists() or not path.is_file():
        raise ValueError(
            "CAI reference patched binary prefill receipt is unavailable. "
            "Run process_prefill first."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8") or "{}")
    except Exception as exc:
        raise ValueError(
            "CAI reference patched binary prefill receipt is invalid."
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("CAI reference patched binary prefill receipt is invalid.")
    if str(payload.get("abi") or "").strip() != REFERENCE_PATCHED_PREFILL_RECEIPT_ABI:
        raise ValueError(
            "CAI reference patched binary prefill receipt ABI is unsupported."
        )
    return payload


def _validate_decode_input_against_prefill_receipt(
    request: Mapping[str, Any],
    *,
    context: ReferenceLoadedShardContext,
    prefill_receipt: Mapping[str, Any],
) -> None:
    declared_shard_hash = str(
        prefill_receipt.get("preparedShardSha256Hex") or ""
    ).strip().lower()
    if declared_shard_hash != context.prepared_shard_sha256_hex:
        raise ValueError(
            "CAI reference patched binary prepared shard hash drifted from prefill receipt."
        )
    input_state = request.get("inputState")
    if not isinstance(input_state, Mapping):
        raise ValueError(
            "CAI reference patched binary process_decode needs validated inputState."
        )
    metadata = input_state.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(
            "CAI reference patched binary inputState metadata is missing."
        )
    expected_digest = str(prefill_receipt.get("prefillExecutionDigest") or "").strip()
    actual_digest = str(metadata.get("prefillExecutionDigest") or "").strip()
    if expected_digest and actual_digest != expected_digest:
        raise ValueError(
            "CAI reference patched binary inputState prefillExecutionDigest mismatch."
        )
    actual_shard_hash = str(metadata.get("preparedShardSha256Hex") or "").strip().lower()
    if actual_shard_hash != context.prepared_shard_sha256_hex:
        raise ValueError(
            "CAI reference patched binary inputState prepared shard hash mismatch."
        )


def _read_input_payload(request: Mapping[str, Any]) -> bytes:
    input_payload_file = request.get("inputPayloadFile")
    if not isinstance(input_payload_file, Mapping):
        return b""
    input_path = Path(_require_string(input_payload_file, "path")).expanduser().resolve()
    payload = input_path.read_bytes()
    expected_size = _optional_int(input_payload_file.get("sizeBytes"))
    if expected_size is not None and expected_size != len(payload):
        raise ValueError(
            "CAI reference patched binary input payload size mismatch."
        )
    expected_hash = str(input_payload_file.get("sha256Hex") or "").strip().lower()
    if expected_hash and expected_hash != hashlib.sha256(payload).hexdigest():
        raise ValueError(
            "CAI reference patched binary input payload hash mismatch."
        )
    return payload


def _input_state_digest(request: Mapping[str, Any]) -> str | None:
    input_state = request.get("inputState")
    if not isinstance(input_state, Mapping):
        return None
    state_file = input_state.get("stateFile")
    if not isinstance(state_file, Mapping):
        return None
    return str(state_file.get("sha256Hex") or "").strip().lower() or None


def _expected_output_kind(request: Mapping[str, Any]) -> str:
    output_kind = str(request.get("expectedOutputKind") or "").strip().lower()
    if not output_kind:
        raise ValueError(
            "CAI reference patched binary expectedOutputKind is missing."
        )
    return output_kind


def _io_targets(request: Mapping[str, Any]) -> Mapping[str, Any]:
    io_targets = request.get("ioTargets")
    if not isinstance(io_targets, Mapping):
        raise ValueError("CAI reference patched binary ioTargets are missing.")
    return io_targets


def _require_string(payload: Mapping[str, Any], field_name: str) -> str:
    value = str(payload.get(field_name) or "").strip()
    if not value:
        raise ValueError(
            f"CAI reference patched binary field is missing: {field_name}"
        )
    return value


def _require_int(payload: Mapping[str, Any], field_name: str) -> int:
    value = _optional_int(payload.get(field_name))
    if value is None:
        raise ValueError(
            f"CAI reference patched binary field is invalid: {field_name}"
        )
    return value


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
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


if __name__ == "__main__":
    raise SystemExit(main())

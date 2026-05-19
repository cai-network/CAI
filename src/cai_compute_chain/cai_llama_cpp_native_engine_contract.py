# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from .cai_owned_runtime import LLAMA_CPP_EXTERNAL_SHARD_LOCAL_FILE_IO_ABI


NATIVE_EXECUTION_SCHEMA_VERSION = 1
ASSIGNMENT_ARTIFACT_COVERAGE_ABI = "cai-llama-cpp-assignment-coverage-v1"


@dataclass(frozen=True)
class NativeEngineArtifactChoice:
    kind: str
    artifact_id: str | None
    source: str
    local_path: str
    fallback_mode: str | None = None


@dataclass(frozen=True)
class AssignmentArtifactCoverage:
    abi: str
    materialization_mode: str
    artifact_size_bytes: int
    covered_byte_count: int
    covered_range_count: int
    zero_filled_outside_covered_ranges: bool


@dataclass(frozen=True)
class AssignmentArtifactChunkRange:
    chunk_id: str
    offset_bytes: int
    size_bytes: int
    sha256_hex: str | None
    layer_start: int | None
    layer_end: int | None
    tensor_names: tuple[str, ...]


def select_native_engine_artifact(
    request: Mapping[str, Any],
    *,
    artifact_kind: str | None = None,
) -> NativeEngineArtifactChoice | None:
    action = str(request.get("action") or "").strip()
    if action not in {"process_prefill", "process_decode"}:
        return None
    local_resolution = request.get("localArtifactResolution")
    if not isinstance(local_resolution, Mapping):
        return None
    assignment = (
        local_resolution.get("assignmentArtifact")
        if isinstance(local_resolution.get("assignmentArtifact"), Mapping)
        else None
    )
    model = (
        local_resolution.get("modelArtifact")
        if isinstance(local_resolution.get("modelArtifact"), Mapping)
        else None
    )
    preferred_kind = str(artifact_kind or "").strip().lower()
    if preferred_kind == "assignment":
        if assignment is None:
            raise ValueError(
                "CAI llama.cpp native engine assignment artifact is unavailable."
            )
        return _artifact_choice_from_mapping(assignment, kind="assignment")
    if preferred_kind == "model":
        if model is None:
            raise ValueError("CAI llama.cpp native engine model artifact is unavailable.")
        return _artifact_choice_from_mapping(
            model,
            kind="model",
            fallback_mode="full_model" if assignment is not None else None,
        )
    if preferred_kind:
        raise ValueError(
            f"CAI llama.cpp native engine artifact kind is unsupported: {preferred_kind}"
        )
    if assignment is not None:
        return _artifact_choice_from_mapping(assignment, kind="assignment")
    if model is not None:
        return _artifact_choice_from_mapping(model, kind="model")
    return None


def build_native_execution_receipt(
    request: Mapping[str, Any],
    *,
    artifact_kind: str | None = None,
    used_patched_backend: bool = True,
    fallback_mode: str | None = None,
) -> dict[str, Any] | None:
    action = str(request.get("action") or "").strip()
    if action not in {"process_prefill", "process_decode"}:
        return None
    choice = select_native_engine_artifact(request, artifact_kind=artifact_kind)
    if choice is None:
        return None
    receipt: dict[str, Any] = {
        "schemaVersion": NATIVE_EXECUTION_SCHEMA_VERSION,
        "executionMode": "layer_range",
        "action": action,
        "modelId": _request_model_id(request),
        "layerStart": _request_layer_bound(request, "layerStart"),
        "layerEnd": _request_layer_bound(request, "layerEnd"),
        "artifactKind": choice.kind,
        "artifactId": choice.artifact_id,
        "artifactSource": choice.source,
        "artifactPath": choice.local_path,
        "usedPatchedBackend": bool(used_patched_backend),
    }
    resolved_fallback = str(fallback_mode or choice.fallback_mode or "").strip()
    if resolved_fallback:
        receipt["fallbackMode"] = resolved_fallback
    return receipt


def resolve_assignment_artifact_coverage(
    artifact: Mapping[str, Any],
    *,
    error_prefix: str = "CAI llama.cpp native engine",
) -> AssignmentArtifactCoverage | None:
    coverage = artifact.get("coverage")
    if not isinstance(coverage, Mapping):
        if str(artifact.get("source") or "").strip() == "materialized_assignment":
            chunk_ranges = resolve_assignment_artifact_chunk_ranges(
                artifact,
                error_prefix=error_prefix,
            )
            if chunk_ranges:
                raise ValueError(
                    f"{error_prefix} assignment artifact coverage is missing."
                )
        return None
    abi = str(coverage.get("abi") or "").strip()
    if abi != ASSIGNMENT_ARTIFACT_COVERAGE_ABI:
        raise ValueError(f"{error_prefix} assignment artifact coverage ABI is invalid.")
    materialization_mode = str(coverage.get("materializationMode") or "").strip()
    if not materialization_mode:
        raise ValueError(
            f"{error_prefix} assignment artifact coverage mode is missing."
        )
    artifact_size_bytes = _mapping_positive_int(
        coverage.get("artifactSizeBytes"),
        field_name=f"{error_prefix} assignment artifact coverage artifactSizeBytes",
    )
    covered_byte_count = _mapping_non_negative_int(
        coverage.get("coveredByteCount"),
        field_name=f"{error_prefix} assignment artifact coverage coveredByteCount",
    )
    covered_range_count = _mapping_non_negative_int(
        coverage.get("coveredRangeCount"),
        field_name=f"{error_prefix} assignment artifact coverage coveredRangeCount",
    )
    zero_filled = _mapping_bool(
        coverage.get("zeroFilledOutsideCoveredRanges"),
        field_name=(
            f"{error_prefix} assignment artifact coverage "
            "zeroFilledOutsideCoveredRanges"
        ),
    )
    chunk_ranges = resolve_assignment_artifact_chunk_ranges(
        artifact,
        error_prefix=error_prefix,
    )
    if chunk_ranges:
        for item in chunk_ranges:
            if item.offset_bytes + item.size_bytes > artifact_size_bytes:
                raise ValueError(
                    f"{error_prefix} assignment artifact chunk range exceeds "
                    "artifact bounds."
                )
        expected_covered_bytes = sum(int(item.size_bytes) for item in chunk_ranges)
        if covered_byte_count != expected_covered_bytes:
            raise ValueError(
                f"{error_prefix} assignment artifact coverage byte count mismatch."
            )
        if covered_range_count != len(chunk_ranges):
            raise ValueError(
                f"{error_prefix} assignment artifact coverage range count mismatch."
            )
    artifact_size = _mapping_optional_non_negative_int(artifact.get("sizeBytes"))
    if artifact_size is not None and artifact_size_bytes != artifact_size:
        raise ValueError(
            f"{error_prefix} assignment artifact coverage size mismatch."
        )
    if materialization_mode == "sparse_full_size" and not zero_filled:
        raise ValueError(
            f"{error_prefix} sparse assignment artifact must declare zero-filled gaps."
        )
    return AssignmentArtifactCoverage(
        abi=abi,
        materialization_mode=materialization_mode,
        artifact_size_bytes=artifact_size_bytes,
        covered_byte_count=covered_byte_count,
        covered_range_count=covered_range_count,
        zero_filled_outside_covered_ranges=zero_filled,
    )


def resolve_assignment_artifact_chunk_ranges(
    artifact: Mapping[str, Any],
    *,
    error_prefix: str = "CAI llama.cpp native engine",
) -> tuple[AssignmentArtifactChunkRange, ...]:
    raw_ranges = artifact.get("chunkRanges")
    if not isinstance(raw_ranges, list):
        return ()
    output: list[AssignmentArtifactChunkRange] = []
    for item in raw_ranges:
        if not isinstance(item, Mapping):
            continue
        chunk_id = str(item.get("chunkId") or "").strip()
        offset_bytes = _mapping_non_negative_int(
            item.get("offsetBytes"),
            field_name=f"{error_prefix} assignment chunk offsetBytes",
        )
        size_bytes = _mapping_positive_int(
            item.get("sizeBytes"),
            field_name=f"{error_prefix} assignment chunk sizeBytes",
        )
        sha256_hex = str(item.get("sha256Hex") or "").strip().lower() or None
        layer_start = _mapping_optional_non_negative_int(item.get("layerStart"))
        layer_end = _mapping_optional_non_negative_int(item.get("layerEnd"))
        if (layer_start is None) != (layer_end is None):
            raise ValueError(
                f"{error_prefix} assignment chunk layer range is incomplete."
            )
        if (
            layer_start is not None
            and layer_end is not None
            and int(layer_end) <= int(layer_start)
        ):
            raise ValueError(
                f"{error_prefix} assignment chunk layer range is invalid."
            )
        output.append(
            AssignmentArtifactChunkRange(
                chunk_id=chunk_id,
                offset_bytes=offset_bytes,
                size_bytes=size_bytes,
                sha256_hex=sha256_hex,
                layer_start=layer_start,
                layer_end=layer_end,
                tensor_names=_mapping_string_sequence(
                    item.get("tensorNames"),
                    field_name=f"{error_prefix} assignment chunk tensorNames",
                ),
            )
        )
    return tuple(output)


def validate_assignment_artifact_chunk_layer_coverage(
    artifact: Mapping[str, Any],
    *,
    layer_start: int | None,
    layer_end: int | None,
    error_prefix: str = "CAI llama.cpp native engine",
) -> None:
    if layer_start is None or layer_end is None:
        return
    if layer_start < 0 or layer_end <= layer_start:
        raise ValueError(f"{error_prefix} layer range is invalid.")
    ranges = resolve_assignment_artifact_chunk_ranges(
        artifact,
        error_prefix=error_prefix,
    )
    scoped_ranges = sorted(
        (
            item
            for item in ranges
            if item.layer_start is not None
            and item.layer_end is not None
            and item.layer_start < layer_end
            and item.layer_end > layer_start
        ),
        key=lambda item: (
            int(item.layer_start or 0),
            int(item.layer_end or 0),
            int(item.offset_bytes),
            str(item.chunk_id),
        ),
    )
    if not scoped_ranges:
        raise ValueError(
            f"{error_prefix} assignment chunk ranges do not include "
            "layer-scoped weights for the requested range."
        )
    cursor = int(layer_start)
    for item in scoped_ranges:
        item_start = max(int(item.layer_start or 0), int(layer_start))
        item_end = min(int(item.layer_end or 0), int(layer_end))
        if item_end <= cursor:
            continue
        if item_start > cursor:
            raise ValueError(
                f"{error_prefix} assignment chunk ranges have a layer gap: "
                f"missing layers {cursor}..{item_start}."
            )
        cursor = max(cursor, item_end)
        if cursor >= int(layer_end):
            return
    raise ValueError(
        f"{error_prefix} assignment chunk ranges have a layer gap: "
        f"missing layers {cursor}..{layer_end}."
    )


def decode_native_engine_input_payload(
    request: Mapping[str, Any],
    *,
    error_prefix: str = "CAI llama.cpp native engine",
) -> bytes:
    local_file_contract = _local_file_contract(request)
    payload_file = request.get("payloadFile")
    if isinstance(payload_file, Mapping):
        return _read_payload_file(
            payload_file,
            local_file_contract=local_file_contract,
            error_prefix=error_prefix,
        )
    raw = request.get("payloadBase64")
    if raw is None:
        return b""
    try:
        return base64.b64decode(str(raw or "").encode("ascii"), validate=True)
    except Exception as exc:
        raise ValueError(f"{error_prefix} payload is invalid.") from exc


def build_native_engine_process_response(
    request: Mapping[str, Any],
    output_payload: bytes,
    *,
    metrics: Mapping[str, Any],
    artifact_kind: str | None = None,
    fallback_mode: str | None = None,
    error_prefix: str = "CAI llama.cpp native engine",
    allow_empty_output_without_metadata: bool = False,
) -> dict[str, Any]:
    output_hash = hashlib.sha256(output_payload).hexdigest()
    response: dict[str, Any] = {
        "status": "ok",
        "outputPayloadSha256Hex": output_hash,
        "metrics": dict(metrics),
    }
    native_execution = build_native_execution_receipt(
        request,
        artifact_kind=artifact_kind,
        fallback_mode=fallback_mode,
    )
    if native_execution is not None:
        response["nativeExecution"] = native_execution
    response.update(
        _encode_output_payload(
            request,
            output_payload,
            output_hash,
            error_prefix=error_prefix,
        )
    )
    output_metadata = _output_frame_metadata(request, output_hash)
    if output_metadata is not None:
        response["outputFrameMetadata"] = output_metadata
    elif not output_payload and not allow_empty_output_without_metadata:
        raise ValueError(f"{error_prefix} decode output is empty.")
    return response


def _artifact_choice_from_mapping(
    artifact: Mapping[str, Any],
    *,
    kind: str,
    fallback_mode: str | None = None,
) -> NativeEngineArtifactChoice:
    source = str(artifact.get("source") or "").strip()
    local_path = str(artifact.get("localPath") or "").strip()
    if not source:
        raise ValueError("CAI llama.cpp native engine artifact source is missing.")
    if not local_path:
        raise ValueError("CAI llama.cpp native engine artifact path is missing.")
    try:
        resolved_path = str(Path(local_path).expanduser().resolve())
    except Exception:
        resolved_path = local_path
    artifact_id = str(artifact.get("artifactId") or "").strip() or None
    return NativeEngineArtifactChoice(
        kind=kind,
        artifact_id=artifact_id,
        source=source,
        local_path=resolved_path,
        fallback_mode=str(fallback_mode or "").strip() or None,
    )


def _local_file_contract(request: Mapping[str, Any]) -> Mapping[str, Any] | None:
    contract = request.get("localFileContract")
    return contract if isinstance(contract, Mapping) else None


def _read_payload_file(
    payload_file: Mapping[str, Any],
    *,
    local_file_contract: Mapping[str, Any] | None,
    error_prefix: str,
) -> bytes:
    payload_path = _resolve_local_file_path(
        path_value=payload_file.get("path"),
        local_file_contract=local_file_contract,
        field_name="payload",
        error_prefix=error_prefix,
    )
    try:
        payload = payload_path.read_bytes()
    except Exception as exc:
        raise ValueError(f"{error_prefix} payload file is unreadable.") from exc
    expected_size = payload_file.get("sizeBytes")
    if expected_size is not None:
        try:
            size_value = int(expected_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{error_prefix} payload file size is invalid.") from exc
        if size_value != len(payload):
            raise ValueError(f"{error_prefix} payload file size mismatch.")
    expected_hash = str(payload_file.get("sha256Hex") or "").strip().lower()
    if expected_hash and expected_hash != hashlib.sha256(payload).hexdigest():
        raise ValueError(f"{error_prefix} payload file hash mismatch.")
    return payload


def _encode_output_payload(
    request: Mapping[str, Any],
    output_payload: bytes,
    output_hash: str,
    *,
    error_prefix: str,
) -> dict[str, Any]:
    local_file_contract = _local_file_contract(request)
    if isinstance(local_file_contract, Mapping):
        output_path = _resolve_local_file_path(
            path_value=local_file_contract.get("responseOutputPath"),
            local_file_contract=local_file_contract,
            field_name="output",
            error_prefix=error_prefix,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(output_payload)
        return {
            "outputPayloadFile": {
                "path": str(output_path),
                "sizeBytes": len(output_payload),
                "sha256Hex": output_hash,
            }
        }
    return {"outputPayloadBase64": base64.b64encode(output_payload).decode("ascii")}


def _resolve_local_file_path(
    *,
    path_value: object,
    local_file_contract: Mapping[str, Any] | None,
    field_name: str,
    error_prefix: str,
) -> Path:
    raw_path = str(path_value or "").strip()
    if not raw_path:
        raise ValueError(f"{error_prefix} {field_name} file path is missing.")
    path = Path(raw_path)
    if not path.is_absolute():
        raise ValueError(f"{error_prefix} {field_name} file path must be absolute.")
    if isinstance(local_file_contract, Mapping):
        abi = str(local_file_contract.get("abi") or "").strip()
        if abi and abi != LLAMA_CPP_EXTERNAL_SHARD_LOCAL_FILE_IO_ABI:
            raise ValueError(f"{error_prefix} local file contract ABI is invalid.")
        io_root = str(local_file_contract.get("ioRoot") or "").strip()
        if io_root:
            root_path = Path(io_root).resolve()
            resolved_path = path.resolve()
            try:
                resolved_path.relative_to(root_path)
            except ValueError as exc:
                raise ValueError(
                    f"{error_prefix} {field_name} file escaped IO root."
                ) from exc
    return path


def _output_frame_metadata(
    request: Mapping[str, Any],
    output_hash: str,
) -> dict[str, Any] | None:
    contract = request.get("outputContract")
    template = (
        contract.get("frameMetadataTemplate")
        if isinstance(contract, Mapping)
        else None
    )
    if not isinstance(template, Mapping):
        return None
    output = json.loads(json.dumps(dict(template), sort_keys=True))
    output["payloadSha256Hex"] = output_hash
    handoff = dict(output.get("llmHandoff") or {})
    tensor = dict(handoff.get("tensor") or {})
    tensor["sha256Hex"] = output_hash
    handoff["tensor"] = tensor
    output["llmHandoff"] = handoff
    return output


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


def _optional_non_negative_int(value: Any) -> int | None:
    try:
        clean = int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
    if clean is None or clean < 0:
        return None
    return clean


def _mapping_optional_non_negative_int(value: Any) -> int | None:
    return _optional_non_negative_int(value)


def _mapping_non_negative_int(value: Any, *, field_name: str) -> int:
    clean = _optional_non_negative_int(value)
    if clean is None:
        raise ValueError(f"{field_name} is invalid.")
    return clean


def _mapping_positive_int(value: Any, *, field_name: str) -> int:
    clean = _optional_non_negative_int(value)
    if clean is None or clean <= 0:
        raise ValueError(f"{field_name} is invalid.")
    return clean


def _mapping_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field_name} is invalid.")


def _mapping_string_sequence(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{field_name} is invalid.")
    output: list[str] = []
    for item in value:
        clean = str(item or "").strip()
        if not clean:
            raise ValueError(f"{field_name} is invalid.")
        output.append(clean)
    return tuple(output)

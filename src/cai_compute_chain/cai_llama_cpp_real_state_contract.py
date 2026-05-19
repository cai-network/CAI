# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cai_owned_runtime import (
    LLAMA_CPP_EXTERNAL_SHARD_ACTIVATION_BOUNDARY,
    LLAMA_CPP_EXTERNAL_SHARD_DECODE_STATE_BOUNDARY,
)


REAL_STATE_PAYLOAD_SCHEMA_VERSION = 1
REAL_STATE_PAYLOAD_ABI = "cai-llama-cpp-real-state-payload-v1"
REAL_STATE_OUTPUT_KINDS = frozenset({"activation_state", "decode_state"})
REAL_STATE_FORMATS = {
    "activation_state": (
        f"ggml-tensor-v1/{LLAMA_CPP_EXTERNAL_SHARD_ACTIVATION_BOUNDARY}"
    ),
    "decode_state": f"ggml-kv-cache-v1/{LLAMA_CPP_EXTERNAL_SHARD_DECODE_STATE_BOUNDARY}",
}


@dataclass(frozen=True)
class RealStateFileRef:
    path: Path
    sha256_hex: str
    size_bytes: int


@dataclass(frozen=True)
class RealStateManifest:
    state_kind: str
    produced_by_action: str
    model_id: str
    session_id: str
    layer_start: int | None
    layer_end: int | None
    token_start: int | None
    token_end: int | None
    state_format: str
    state_file: RealStateFileRef
    payload: dict[str, Any]


def build_real_state_manifest(
    *,
    output_kind: str,
    action: str,
    model_id: str,
    session_id: str,
    layer_start: int | None,
    layer_end: int | None,
    token_start: int | None,
    token_end: int | None,
    state_file_path: str | Path,
    state_file_sha256_hex: str | None = None,
    state_file_size_bytes: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_kind = _normalize_output_kind(output_kind)
    resolved_path = Path(state_file_path).expanduser().resolve()
    if state_file_sha256_hex is None or state_file_size_bytes is None:
        payload = resolved_path.read_bytes()
        if state_file_sha256_hex is None:
            state_file_sha256_hex = hashlib.sha256(payload).hexdigest()
        if state_file_size_bytes is None:
            state_file_size_bytes = len(payload)
    manifest: dict[str, Any] = {
        "schemaVersion": REAL_STATE_PAYLOAD_SCHEMA_VERSION,
        "abi": REAL_STATE_PAYLOAD_ABI,
        "stateKind": normalized_kind,
        "producedByAction": str(action or "").strip(),
        "modelId": str(model_id or "").strip(),
        "sessionId": str(session_id or "").strip(),
        "layerStart": layer_start,
        "layerEnd": layer_end,
        "tokenStart": token_start,
        "tokenEnd": token_end,
        "stateFormat": expected_real_state_format(normalized_kind),
        "stateFile": {
            "path": str(resolved_path),
            "sha256Hex": str(state_file_sha256_hex or "").strip().lower(),
            "sizeBytes": int(state_file_size_bytes or 0),
        },
    }
    if metadata is not None:
        manifest["metadata"] = dict(metadata)
    return manifest


def build_real_state_manifest_payload(**kwargs: Any) -> bytes:
    return json.dumps(
        build_real_state_manifest(**kwargs),
        sort_keys=True,
    ).encode("utf-8")


def validate_real_state_payload(
    payload: bytes,
    *,
    request: Mapping[str, Any],
    output_kind: str,
    error_prefix: str = "CAI llama.cpp real state payload",
    match_request_action: bool = True,
    match_request_bounds: bool = True,
    allow_external_state_file: bool = False,
) -> RealStateManifest:
    normalized_kind = _normalize_output_kind(output_kind)
    try:
        parsed = json.loads(bytes(payload or b"").decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"{error_prefix} is not valid UTF-8 JSON.") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{error_prefix} manifest must be an object.")
    schema_version = _positive_int(
        parsed.get("schemaVersion"),
        field_name=f"{error_prefix} schemaVersion",
    )
    if schema_version != REAL_STATE_PAYLOAD_SCHEMA_VERSION:
        raise ValueError(f"{error_prefix} schemaVersion is unsupported.")
    if str(parsed.get("abi") or "").strip() != REAL_STATE_PAYLOAD_ABI:
        raise ValueError(f"{error_prefix} ABI is unsupported.")
    state_kind = str(parsed.get("stateKind") or "").strip().lower()
    if state_kind != normalized_kind:
        raise ValueError(f"{error_prefix} stateKind does not match output kind.")
    produced_by_action = str(parsed.get("producedByAction") or "").strip()
    if not produced_by_action:
        raise ValueError(f"{error_prefix} producedByAction is missing.")
    if match_request_action:
        expected_action = str(request.get("action") or "").strip()
        if produced_by_action != expected_action:
            raise ValueError(f"{error_prefix} producedByAction does not match request.")
    model_id = str(parsed.get("modelId") or "").strip()
    expected_model_id = _request_string(request, "modelId")
    if expected_model_id and model_id != expected_model_id:
        raise ValueError(f"{error_prefix} modelId does not match request.")
    session_id = str(parsed.get("sessionId") or "").strip()
    expected_session_id = _request_string(request, "sessionId")
    if expected_session_id and session_id != expected_session_id:
        raise ValueError(f"{error_prefix} sessionId does not match request.")
    layer_start = _optional_int(parsed.get("layerStart"))
    layer_end = _optional_int(parsed.get("layerEnd"))
    token_start = _optional_int(parsed.get("tokenStart"))
    token_end = _optional_int(parsed.get("tokenEnd"))
    if match_request_bounds:
        _validate_optional_bound_match(
            actual=layer_start,
            expected=_request_bound(request, "layerStart"),
            field_name=f"{error_prefix} layerStart",
        )
        _validate_optional_bound_match(
            actual=layer_end,
            expected=_request_bound(request, "layerEnd"),
            field_name=f"{error_prefix} layerEnd",
        )
        _validate_optional_bound_match(
            actual=token_start,
            expected=_request_bound(request, "tokenStart"),
            field_name=f"{error_prefix} tokenStart",
        )
        _validate_optional_bound_match(
            actual=token_end,
            expected=_request_bound(request, "tokenEnd"),
            field_name=f"{error_prefix} tokenEnd",
        )
    state_format = str(parsed.get("stateFormat") or "").strip()
    if state_format != expected_real_state_format(normalized_kind):
        raise ValueError(f"{error_prefix} stateFormat is unsupported.")
    metadata = parsed.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise ValueError(f"{error_prefix} metadata must be an object.")
    state_file = _validate_state_file_ref(
        parsed.get("stateFile"),
        request=request,
        error_prefix=error_prefix,
        allow_external_state_file=allow_external_state_file,
    )
    return RealStateManifest(
        state_kind=state_kind,
        produced_by_action=produced_by_action,
        model_id=model_id,
        session_id=session_id,
        layer_start=layer_start,
        layer_end=layer_end,
        token_start=token_start,
        token_end=token_end,
        state_format=state_format,
        state_file=state_file,
        payload=dict(parsed),
    )


def expected_real_state_format(output_kind: str) -> str:
    normalized_kind = _normalize_output_kind(output_kind)
    return REAL_STATE_FORMATS[normalized_kind]


def looks_like_real_state_payload(payload: bytes) -> bool:
    stripped = bytes(payload or b"").lstrip()
    if not stripped or stripped[:1] != b"{":
        return False
    try:
        parsed = json.loads(stripped.decode("utf-8"))
    except Exception:
        return False
    return (
        isinstance(parsed, Mapping)
        and str(parsed.get("abi") or "").strip() == REAL_STATE_PAYLOAD_ABI
    )


def _validate_state_file_ref(
    payload: Any,
    *,
    request: Mapping[str, Any],
    error_prefix: str,
    allow_external_state_file: bool,
) -> RealStateFileRef:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{error_prefix} stateFile is missing.")
    raw_path = str(payload.get("path") or "").strip()
    if not raw_path:
        raise ValueError(f"{error_prefix} stateFile path is missing.")
    workspace = request.get("executionWorkspace")
    if not isinstance(workspace, Mapping):
        raise ValueError(f"{error_prefix} executionWorkspace is missing.")
    state_files_dir = _workspace_dir(
        workspace,
        field_name="stateFilesDir",
        fallback_field="root",
        error_prefix=error_prefix,
    )
    state_path = _workspace_path(raw_path, base_dir=state_files_dir)
    allowed_roots = [state_files_dir]
    if allow_external_state_file:
        allowed_roots.extend(_managed_runtime_state_file_roots(request))
    if not any(_path_is_within(state_path, root) for root in allowed_roots):
        raise ValueError(
            f"{error_prefix} stateFile path must stay within executionWorkspace.stateFilesDir."
        )
    if not state_path.exists() or not state_path.is_file():
        raise ValueError(f"{error_prefix} stateFile is unavailable.")
    size_bytes = _positive_int(
        payload.get("sizeBytes"),
        field_name=f"{error_prefix} stateFile sizeBytes",
    )
    actual_size = int(state_path.stat().st_size)
    if actual_size != size_bytes:
        raise ValueError(f"{error_prefix} stateFile sizeBytes mismatch.")
    sha256_hex = str(payload.get("sha256Hex") or "").strip().lower()
    if not sha256_hex:
        raise ValueError(f"{error_prefix} stateFile sha256Hex is missing.")
    actual_hash = hashlib.sha256(state_path.read_bytes()).hexdigest()
    if actual_hash != sha256_hex:
        raise ValueError(f"{error_prefix} stateFile sha256Hex mismatch.")
    return RealStateFileRef(
        path=state_path,
        sha256_hex=sha256_hex,
        size_bytes=size_bytes,
    )


def _managed_runtime_state_file_roots(request: Mapping[str, Any]) -> list[Path]:
    managed_runtime = request.get("managedRuntime")
    if not isinstance(managed_runtime, Mapping):
        return []
    roots: list[Path] = []
    for field_name in ("runtimeRoot",):
        raw = str(managed_runtime.get(field_name) or "").strip()
        if raw:
            roots.append(Path(raw).expanduser().resolve())
    session_paths = managed_runtime.get("sessionPaths")
    if isinstance(session_paths, Mapping):
        for field_name in ("root", "stateDir", "cacheDir"):
            raw = str(session_paths.get(field_name) or "").strip()
            if raw:
                roots.append(Path(raw).expanduser().resolve())
    return roots


def _request_string(request: Mapping[str, Any], field_name: str) -> str | None:
    raw = request.get(field_name)
    if raw is not None:
        value = str(raw or "").strip()
        if value:
            return value
    for parent in ("shardSpec", "frame"):
        payload = request.get(parent)
        if not isinstance(payload, Mapping):
            continue
        value = str(payload.get(field_name) or "").strip()
        if value:
            return value
    return None


def _request_bound(request: Mapping[str, Any], field_name: str) -> int | None:
    raw = request.get(field_name)
    value = _optional_int(raw)
    if value is not None:
        return value
    for parent in ("shardSpec", "frame"):
        payload = request.get(parent)
        if not isinstance(payload, Mapping):
            continue
        value = _optional_int(payload.get(field_name))
        if value is not None:
            return value
    return None


def _workspace_dir(
    workspace: Mapping[str, Any],
    *,
    field_name: str,
    fallback_field: str,
    error_prefix: str,
) -> Path:
    raw = str(
        workspace.get(field_name) or workspace.get(fallback_field) or ""
    ).strip()
    if not raw:
        raise ValueError(f"{error_prefix} executionWorkspace path is incomplete.")
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


def _normalize_output_kind(output_kind: str) -> str:
    normalized = str(output_kind or "").strip().lower()
    if normalized not in REAL_STATE_OUTPUT_KINDS:
        raise ValueError(
            "CAI llama.cpp real state output kind is unsupported: "
            f"{output_kind!r}"
        )
    return normalized


def _positive_int(value: Any, *, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} is invalid.") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return parsed


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_optional_bound_match(
    *,
    actual: int | None,
    expected: int | None,
    field_name: str,
) -> None:
    if expected is None:
        return
    if actual != expected:
        raise ValueError(f"{field_name} does not match request.")

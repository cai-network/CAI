# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import atexit
import argparse
import base64
from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
from typing import Any

from .cai_llama_cpp_backend_runtime import (
    resolve_managed_llama_cpp_runtime,
    split_llama_cpp_subprocess_command,
    windows_subprocess_creation_flags,
    windows_subprocess_startupinfo,
)
from .cai_llama_cpp_native_engine_contract import (
    resolve_assignment_artifact_coverage,
    build_native_engine_process_response,
    decode_native_engine_input_payload,
    validate_assignment_artifact_chunk_layer_coverage,
)
from .cai_owned_runtime import (
    LLAMA_CPP_EXTERNAL_SHARD_GENERATION_PROBE_ABI,
    LLAMA_CPP_EXTERNAL_SHARD_GENERATION_PROBE_ACTION,
    LLAMA_CPP_EXTERNAL_SHARD_HANDOFF_ABI,
    LLAMA_CPP_EXTERNAL_SHARD_PRODUCTION_CAPABILITIES,
    LLAMA_CPP_EXTERNAL_SHARD_REQUIRED_CAPABILITIES,
    build_llama_cpp_external_shard_patch_boundary,
    build_llama_cpp_external_shard_production_state_contract,
)
from .model_distribution import (
    ModelPackageManifest,
    ModelShardAssignment,
    ensure_assignment_ready_from_store,
    load_model_package_manifest,
    materialize_default_assignment_artifact_from_store,
    select_default_materialized_artifact_id,
    select_model_package_manifest_for_model,
)


ASSIGNMENT_ARTIFACT_ENGINE_ID = "assignment_artifact_engine"
ASSIGNMENT_ARTIFACT_ENGINE_VERSION = "assignment-artifact-engine/0.1"
ASSIGNMENT_ARTIFACT_ENGINE_CAPABILITIES = (
    *LLAMA_CPP_EXTERNAL_SHARD_REQUIRED_CAPABILITIES,
    "assignment_artifact_execution",
)
ASSIGNMENT_ARTIFACT_ENGINE_PRODUCTION_CAPABILITIES = tuple(
    dict.fromkeys(
        (
            *ASSIGNMENT_ARTIFACT_ENGINE_CAPABILITIES,
            *LLAMA_CPP_EXTERNAL_SHARD_PRODUCTION_CAPABILITIES,
        )
    )
)
ASSIGNMENT_STATE_PAYLOAD_ABI = "cai-llama-cpp-assignment-state-v1"
ASSIGNMENT_OUTPUT_PAYLOAD_ABI = "cai-llama-cpp-assignment-output-v1"
ASSIGNMENT_MANAGED_SESSION_ABI = "cai-llama-cpp-assignment-managed-session-v1"
EXECUTION_WORKSPACE_ABI = "cai-llama-cpp-execution-workspace-v1"
ASSIGNMENT_EXECUTOR_REQUEST_ABI = "cai-llama-cpp-assignment-executor-v1"
CAI_LLM_ASSIGNMENT_EXECUTOR_COMMAND_ENV = (
    "CAI_LLM_SHARD_ASSIGNMENT_EXECUTOR_COMMAND"
)
CAI_LLM_ASSIGNMENT_EXECUTOR_PERSISTENT_ENV = (
    "CAI_LLM_SHARD_ASSIGNMENT_EXECUTOR_PERSISTENT"
)
CAI_LLM_ASSIGNMENT_EXECUTOR_TIMEOUT_ENV = (
    "CAI_LLM_SHARD_ASSIGNMENT_EXECUTOR_TIMEOUT_SEC"
)
CAI_LLM_SHARD_NATIVE_TIMEOUT_ENV = "CAI_LLM_SHARD_NATIVE_TIMEOUT_SEC"
REFERENCE_EXECUTOR_BACKEND_MODES = {
    "assignment_slot_state_executor",
    "llama.cpp-slot-state",
    "llama_cpp_slot_state",
    "patched_slot_state_engine",
    "slot_state",
}


@dataclass(frozen=True)
class AssignmentArtifactContext:
    artifact_id: str | None
    artifact_source: str
    local_path: Path
    size_bytes: int
    layer_start: int | None
    layer_end: int | None
    expected_digest: str | None
    chunk_ranges: tuple[dict[str, Any], ...]
    window_digest: str
    digest_source: str
    bytes_read: int
    coverage_mode: str | None
    covered_byte_count: int | None
    covered_range_count: int | None
    zero_filled_outside_covered_ranges: bool | None


@dataclass(frozen=True)
class AssignmentSessionKey:
    session_id: str
    model_id: str | None
    layer_start: int | None
    layer_end: int | None


@dataclass
class AssignmentLoadedSession:
    key: AssignmentSessionKey
    context: AssignmentArtifactContext
    loaded_action: str


@dataclass(frozen=True)
class AssignmentResidentArtifactKey:
    model_id: str | None
    local_path: str
    layer_start: int | None
    layer_end: int | None
    expected_digest: str | None
    chunk_ranges_signature: tuple[
        tuple[str, int, int, str | None, tuple[str, ...]],
        ...,
    ]


@dataclass(frozen=True)
class AssignmentExecutorResult:
    output_payload: bytes
    output_kind: str | None
    metrics: dict[str, Any]


class PersistentAssignmentExecutorClient:
    def __init__(
        self,
        command: list[str],
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
                raise ValueError("CAI assignment executor persistent client is closed.")
            if self._process.poll() is not None:
                raise ValueError(
                    "CAI assignment executor persistent client exited unexpectedly."
                )
            if self._process.stdin is None:
                raise ValueError(
                    "CAI assignment executor persistent client stdin is unavailable."
                )
            try:
                self._process.stdin.write(json.dumps(dict(request), sort_keys=True))
                self._process.stdin.write("\n")
                self._process.stdin.flush()
                response_text = self._stdout_lines.get(timeout=self.timeout_sec)
            except queue.Empty as exc:
                self.close(kill=True)
                raise ValueError(
                    "CAI assignment executor persistent client timed out."
                ) from exc
            except Exception as exc:
                raise ValueError(
                    "CAI assignment executor persistent client I/O failed."
                ) from exc
        try:
            parsed = json.loads(response_text or "{}")
        except Exception as exc:
            raise ValueError(
                "CAI assignment executor persistent client returned invalid JSON."
            ) from exc
        if not isinstance(parsed, Mapping):
            raise ValueError(
                "CAI assignment executor persistent client response must be an object."
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
    def pid(self) -> int:
        return int(self._process.pid)

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


_ENGINE_SESSIONS: dict[AssignmentSessionKey, AssignmentLoadedSession] = {}
_ENGINE_RESIDENT_ARTIFACTS: dict[
    AssignmentResidentArtifactKey,
    AssignmentArtifactContext,
] = {}
_PERSISTENT_ASSIGNMENT_EXECUTORS: dict[
    tuple[tuple[str, ...], str | None, float],
    PersistentAssignmentExecutorClient,
] = {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a CAI assignment-artifact engine. This backend consumes "
            "assignmentArtifact chunk windows directly and never falls back "
            "to a full model replica."
        ),
    )
    parser.add_argument("--jsonl", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.jsonl):
        return _jsonl_loop()
    try:
        request = json.loads(sys.stdin.read() or "{}")
        response = handle_assignment_artifact_engine_request(request)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), flush=True)
        return 1
    print(json.dumps(response, sort_keys=True), flush=True)
    return 0


def handle_assignment_artifact_engine_request(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        raise ValueError("CAI assignment artifact engine request must be an object.")
    if str(request.get("abi") or "").strip() != LLAMA_CPP_EXTERNAL_SHARD_HANDOFF_ABI:
        raise ValueError("CAI assignment artifact engine request ABI is unsupported.")
    action = str(request.get("action") or "").strip()
    if action == "load_shard":
        return _load_shard(request)
    if action == LLAMA_CPP_EXTERNAL_SHARD_GENERATION_PROBE_ACTION:
        return _probe_generation(request)
    if action == "process_prefill":
        return _process_prefill(request)
    if action == "process_decode":
        return _process_decode(request)
    if action == "finalize":
        return _finalize(request)
    raise ValueError(
        f"CAI assignment artifact engine action is unsupported: {action}"
    )


def _jsonl_loop() -> int:
    for line in sys.stdin:
        try:
            request = json.loads(line or "{}")
            response = handle_assignment_artifact_engine_request(request)
        except Exception as exc:
            response = {"status": "error", "error": str(exc)}
        print(json.dumps(response, sort_keys=True), flush=True)
    return 0


def reset_assignment_artifact_engine_sessions() -> None:
    _ENGINE_SESSIONS.clear()
    _ENGINE_RESIDENT_ARTIFACTS.clear()
    reset_assignment_executor_clients()


def reset_assignment_executor_clients() -> None:
    clients = list(_PERSISTENT_ASSIGNMENT_EXECUTORS.values())
    _PERSISTENT_ASSIGNMENT_EXECUTORS.clear()
    for client in clients:
        client.close(kill=True)


atexit.register(reset_assignment_executor_clients)


def _assignment_patch_boundary(
    *,
    production_ready: bool,
    reason: str | None = None,
) -> dict[str, Any]:
    capabilities = (
        ASSIGNMENT_ARTIFACT_ENGINE_PRODUCTION_CAPABILITIES
        if production_ready
        else ASSIGNMENT_ARTIFACT_ENGINE_CAPABILITIES
    )
    extra_metadata: dict[str, Any] = {
        "mode": ASSIGNMENT_ARTIFACT_ENGINE_ID,
        "productionReady": bool(production_ready),
    }
    if production_ready:
        extra_metadata["productionReadyReason"] = (
            reason
            or "Assignment artifacts execute through a patched llama.cpp binary "
            "with shard-only loading proof."
        )
        extra_metadata["productionStateContract"] = (
            build_llama_cpp_external_shard_production_state_contract()
        )
    else:
        extra_metadata["productionReadyReason"] = (
            reason
            or "Assignment artifact engine requires a shard-only patched "
            "llama.cpp executor before it can advertise production state."
        )
    return build_llama_cpp_external_shard_patch_boundary(
        backend="llama.cpp-patched",
        backend_version=ASSIGNMENT_ARTIFACT_ENGINE_VERSION,
        patch_id="cai-llama-cpp-assignment-artifact-engine",
        runner_protocol_version="assignment-artifact-0.1",
        capabilities=capabilities,
        extra_metadata=extra_metadata,
    )


def _apply_assignment_patch_boundary(
    response: dict[str, Any],
    *,
    production_ready: bool,
    reason: str | None = None,
) -> None:
    capabilities = (
        ASSIGNMENT_ARTIFACT_ENGINE_PRODUCTION_CAPABILITIES
        if production_ready
        else ASSIGNMENT_ARTIFACT_ENGINE_CAPABILITIES
    )
    response["capabilities"] = list(capabilities)
    response["patchBoundary"] = _assignment_patch_boundary(
        production_ready=production_ready,
        reason=reason,
    )
    metrics = response.get("metrics")
    if isinstance(metrics, dict):
        metrics["backendCapabilities"] = list(capabilities)
        metrics["productionReady"] = bool(production_ready)


def _load_shard(request: Mapping[str, Any]) -> dict[str, Any]:
    response: dict[str, Any] = {
        "status": "ready",
        "metrics": {
            "backendLoaded": True,
            "backendMode": ASSIGNMENT_ARTIFACT_ENGINE_ID,
            "backendCapabilities": list(ASSIGNMENT_ARTIFACT_ENGINE_CAPABILITIES),
        },
    }
    _apply_assignment_patch_boundary(response, production_ready=False)
    metrics = response["metrics"]
    session_key = _request_session_key(request)
    if session_key is not None:
        metrics["assignmentArtifactSessionKey"] = _session_key_text(session_key)
    assignment_mapping = _request_assignment_artifact_mapping(request)
    if assignment_mapping is not None:
        context, assignment_metrics = _resolved_assignment_artifact_context(
            request,
            action="load_shard",
        )
        metrics.update(assignment_metrics)
        metrics["assignmentArtifactBytesRead"] = context.bytes_read
        metrics["assignmentArtifactChunkCount"] = len(context.chunk_ranges)
        metrics["assignmentArtifactCoverageMode"] = context.coverage_mode
        metrics["assignmentArtifactCoveredByteCount"] = context.covered_byte_count
        metrics["assignmentArtifactCoveredRangeCount"] = context.covered_range_count
        metrics.update(
            _stage_managed_session_files(
                request,
                action="load_shard",
                artifact=context,
            )
        )
        executor_metrics = _run_assignment_executor_lifecycle(
            request,
            action="load_shard",
            artifact=context,
        )
        if executor_metrics:
            metrics.update(executor_metrics)
            if _assignment_executor_metrics_prove_shard_only(executor_metrics):
                _apply_assignment_patch_boundary(response, production_ready=True)
    else:
        metrics["assignmentArtifactSessionLoaded"] = False
        metrics["assignmentArtifactSessionCacheHit"] = False
        metrics["assignmentArtifactResidentShardHit"] = False
        metrics["assignmentArtifactResidentShardLoaded"] = False
    return response


def _probe_generation(request: Mapping[str, Any]) -> dict[str, Any]:
    probe = request.get("generationProbe")
    if not isinstance(probe, Mapping):
        raise ValueError("CAI assignment artifact generationProbe is missing.")
    model_id = (
        str(probe.get("modelId") or "").strip()
        or _request_model_id(request)
        or ""
    )
    if not model_id:
        raise ValueError("CAI assignment artifact generationProbe modelId is missing.")
    if not _assignment_executor_command():
        return _generation_probe_not_ready(
            model_id=model_id,
            reason=(
                f"{CAI_LLM_ASSIGNMENT_EXECUTOR_COMMAND_ENV} is not configured; "
                "real generation cannot be proven through assignment artifacts."
            ),
        )
    try:
        return _run_assignment_generation_probe(request, probe, model_id=model_id)
    except Exception as exc:
        return _generation_probe_not_ready(
            model_id=model_id,
            reason=str(exc),
            error_class=exc.__class__.__name__,
        )


def _generation_probe_not_ready(
    *,
    model_id: str | None,
    reason: str,
    error_class: str | None = None,
) -> dict[str, Any]:
    probe_payload: dict[str, Any] = {
        "schemaVersion": 1,
        "abi": LLAMA_CPP_EXTERNAL_SHARD_GENERATION_PROBE_ABI,
        "ready": False,
        "modelId": str(model_id or "").strip() or None,
        "outputText": "",
        "outputTokenCount": 0,
        "realModelExecution": False,
        "reason": str(reason or "").strip(),
    }
    if error_class:
        probe_payload["errorClass"] = error_class
    return {
        "status": "ok",
        "generationProbe": probe_payload,
        "metrics": {
            "backendAction": LLAMA_CPP_EXTERNAL_SHARD_GENERATION_PROBE_ACTION,
            "backendMode": ASSIGNMENT_ARTIFACT_ENGINE_ID,
            "generationProbeReady": False,
            "generationProbeReason": str(reason or "").strip(),
        },
    }


def _run_assignment_generation_probe(
    request: Mapping[str, Any],
    probe: Mapping[str, Any],
    *,
    model_id: str,
) -> dict[str, Any]:
    manifest = _generation_probe_manifest(request, model_id=model_id)
    total_layers = _generation_probe_total_layers(manifest)
    if total_layers < 2:
        raise ValueError("CAI assignment generationProbe needs at least two layers.")
    split_layer = max(1, min(total_layers - 1, total_layers // 2))
    max_tokens = max(1, _optional_int(probe.get("maxTokens")) or 8)
    prompt_text = str(probe.get("prompt") or "")
    prompt_payload = prompt_text.encode("utf-8")
    session_id = _generation_probe_session_id(
        model_id=model_id,
        prompt=prompt_text,
        split_layer=split_layer,
        total_layers=total_layers,
    )
    workspace_root = _generation_probe_root(request, session_id=session_id)
    shard_a_artifact = _generation_probe_assignment_artifact(
        manifest,
        ModelShardAssignment(
            start_layer=0,
            end_layer=split_layer,
            device_rank=0,
            world_size=2,
        ),
    )
    shard_b_artifact = _generation_probe_assignment_artifact(
        manifest,
        ModelShardAssignment(
            start_layer=split_layer,
            end_layer=total_layers,
            device_rank=1,
            world_size=2,
        ),
    )

    finalize_requests: list[dict[str, Any]] = []
    try:
        load_a_request = _generation_probe_handoff_request(
            request,
            action="load_shard",
            model_id=model_id,
            session_id=session_id,
            assignment_artifact=shard_a_artifact,
            layer_start=0,
            layer_end=split_layer,
            token_start=0,
            token_end=max_tokens,
            payload=b"",
            workspace_root=workspace_root,
            final_output=False,
            next_frame_kind="activation",
        )
        finalize_requests.append(load_a_request)
        load_a = _load_shard(load_a_request)

        prefill_request = _generation_probe_handoff_request(
            request,
            action="process_prefill",
            model_id=model_id,
            session_id=session_id,
            assignment_artifact=shard_a_artifact,
            layer_start=0,
            layer_end=split_layer,
            token_start=0,
            token_end=max_tokens,
            payload=prompt_payload,
            workspace_root=workspace_root,
            final_output=False,
            next_frame_kind="activation",
        )
        prefill = _process_prefill(prefill_request)
        if not _generation_probe_process_proves_real(prefill):
            raise ValueError(
                "CAI assignment generationProbe prefill did not prove real "
                "layer execution."
            )
        prefill_payload = _generation_probe_response_payload(prefill)

        load_b_request = _generation_probe_handoff_request(
            request,
            action="load_shard",
            model_id=model_id,
            session_id=session_id,
            assignment_artifact=shard_b_artifact,
            layer_start=split_layer,
            layer_end=total_layers,
            token_start=0,
            token_end=max_tokens,
            payload=b"",
            workspace_root=workspace_root,
            final_output=True,
            next_frame_kind=None,
        )
        finalize_requests.append(load_b_request)
        load_b = _load_shard(load_b_request)

        decode_request = _generation_probe_handoff_request(
            request,
            action="process_decode",
            model_id=model_id,
            session_id=session_id,
            assignment_artifact=shard_b_artifact,
            layer_start=split_layer,
            layer_end=total_layers,
            token_start=0,
            token_end=max_tokens,
            payload=prefill_payload,
            workspace_root=workspace_root,
            final_output=True,
            next_frame_kind=None,
        )
        decode = _process_decode(decode_request)
        if not _generation_probe_process_proves_real(decode):
            raise ValueError(
                "CAI assignment generationProbe decode did not prove real "
                "layer execution."
            )
        final_payload = _generation_probe_response_payload(decode)
    finally:
        for loaded_request in reversed(finalize_requests):
            try:
                _finalize(dict(loaded_request, action="finalize"))
            except Exception:
                pass

    output_text = _generation_probe_output_text(final_payload)
    if not output_text:
        raise ValueError("CAI assignment generationProbe produced empty output.")
    token_count = (
        _nested_int(decode.get("metrics"), "outputTokenCount")
        or _nested_int(decode.get("metrics"), "tokensGenerated")
        or 1
    )
    shard_only_ready = (
        _assignment_executor_metrics_prove_shard_only(load_a.get("metrics") or {})
        and _assignment_executor_metrics_prove_shard_only(load_b.get("metrics") or {})
    )
    return {
        "status": "ok",
        "generationProbe": {
            "schemaVersion": 1,
            "abi": LLAMA_CPP_EXTERNAL_SHARD_GENERATION_PROBE_ABI,
            "ready": True,
            "modelId": model_id,
            "outputText": output_text,
            "outputTokenCount": max(1, int(token_count)),
            "realModelExecution": True,
            "realLayerExecution": True,
            "layerSplit": {
                "firstShard": [0, split_layer],
                "secondShard": [split_layer, total_layers],
            },
        },
        "metrics": {
            "backendAction": LLAMA_CPP_EXTERNAL_SHARD_GENERATION_PROBE_ACTION,
            "backendMode": ASSIGNMENT_ARTIFACT_ENGINE_ID,
            "generationProbeReady": True,
            "generationProbeLayerSplit": split_layer,
            "generationProbeTotalLayers": total_layers,
            "generationProbeShardOnlyLoadingReady": bool(shard_only_ready),
            "generationProbeArtifactSource": "materialized_assignment",
            "generationProbeWorkspaceRoot": str(workspace_root),
        },
    }


def _generation_probe_manifest(
    request: Mapping[str, Any],
    *,
    model_id: str,
) -> ModelPackageManifest:
    local_resolution = request.get("localArtifactResolution")
    if isinstance(local_resolution, Mapping):
        catalog_id = str(local_resolution.get("catalogId") or "").strip()
        version = str(local_resolution.get("version") or "").strip()
        if catalog_id and version:
            manifest = load_model_package_manifest(catalog_id, version)
            if manifest.model_id != model_id:
                raise ValueError(
                    "CAI assignment generationProbe manifest modelId mismatch."
                )
            return manifest
    manifest = select_model_package_manifest_for_model(model_id)
    if manifest is None:
        raise FileNotFoundError(
            "CAI assignment generationProbe model package manifest is missing."
        )
    return manifest


def _generation_probe_total_layers(manifest: ModelPackageManifest) -> int:
    for key in ("total_layers", "totalLayerCount", "n_layer", "nLayers"):
        value = _optional_int(manifest.metadata.get(key))
        if value is not None and value > 0:
            return value
    layer_end_values = [
        int(chunk.layer_end)
        for chunk in manifest.chunks
        if chunk.layer_end is not None and int(chunk.layer_end) > 0
    ]
    if layer_end_values:
        return max(layer_end_values)
    raise ValueError("CAI assignment generationProbe total layer count is missing.")


def _generation_probe_session_id(
    *,
    model_id: str,
    prompt: str,
    split_layer: int,
    total_layers: int,
) -> str:
    digest = hashlib.sha256(
        f"{model_id}:{prompt}:{split_layer}:{total_layers}".encode("utf-8")
    ).hexdigest()
    return f"cai-generation-probe-{digest[:16]}"


def _generation_probe_root(
    request: Mapping[str, Any],
    *,
    session_id: str,
) -> Path:
    managed_runtime = request.get("managedRuntime")
    runtime_root = ""
    if isinstance(managed_runtime, Mapping):
        runtime_root = str(managed_runtime.get("runtimeRoot") or "").strip()
    base = (
        Path(runtime_root).expanduser().resolve()
        if runtime_root
        else Path(tempfile.gettempdir()).resolve() / "cai-assignment-generation-probe"
    )
    root = (base / "generation-probe" / session_id).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _generation_probe_assignment_artifact(
    manifest: ModelPackageManifest,
    assignment: ModelShardAssignment,
) -> dict[str, Any]:
    artifact_id = str(select_default_materialized_artifact_id(manifest) or "").strip()
    if not artifact_id:
        raise FileNotFoundError(
            "CAI assignment generationProbe has no materializable artifact."
        )
    ready_result = ensure_assignment_ready_from_store(
        manifest,
        assignment,
        include_default_chunks=True,
    )
    if not ready_result.final_plan.ready:
        missing = ", ".join(ready_result.final_plan.coverage.missing_chunk_ids)
        raise FileNotFoundError(
            "CAI assignment generationProbe chunks are not ready"
            + (f": {missing}" if missing else ".")
        )
    materialized = materialize_default_assignment_artifact_from_store(
        manifest,
        assignment,
        include_default_chunks=True,
        overwrite=False,
    )
    chunk_ranges = _generation_probe_chunk_ranges(
        manifest,
        artifact_id=artifact_id,
        layer_start=assignment.start_layer,
        layer_end=assignment.end_layer,
    )
    payload: dict[str, Any] = {
        "artifactId": artifact_id,
        "localPath": str(Path(materialized.output_path).resolve()),
        "source": "materialized_assignment",
        "sizeBytes": int(materialized.size_bytes),
        "expectedDigest": str(materialized.sha256_hex),
        "layerStart": int(assignment.start_layer),
        "layerEnd": int(assignment.end_layer),
        "deviceRank": int(assignment.device_rank),
        "worldSize": int(assignment.world_size),
        "chunkRanges": chunk_ranges,
        "coverage": _generation_probe_assignment_coverage(
            artifact_size_bytes=int(materialized.size_bytes),
            chunk_ranges=chunk_ranges,
        ),
    }
    return payload


def _generation_probe_chunk_ranges(
    manifest: ModelPackageManifest,
    *,
    artifact_id: str,
    layer_start: int,
    layer_end: int,
) -> list[dict[str, Any]]:
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
        key=lambda item: int(item.offset_bytes),
    )
    if not chunks:
        raise FileNotFoundError(
            "CAI assignment generationProbe has no chunks for layer range."
        )
    output: list[dict[str, Any]] = []
    for chunk in chunks:
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
            item["tensorNames"] = [
                str(name) for name in chunk.tensor_names if str(name).strip()
            ]
        output.append(item)
    return output


def _generation_probe_assignment_coverage(
    *,
    artifact_size_bytes: int,
    chunk_ranges: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "abi": "cai-llama-cpp-assignment-coverage-v1",
        "materializationMode": "sparse_full_size",
        "artifactSizeBytes": int(artifact_size_bytes),
        "coveredByteCount": sum(int(item.get("sizeBytes") or 0) for item in chunk_ranges),
        "coveredRangeCount": len(chunk_ranges),
        "zeroFilledOutsideCoveredRanges": True,
    }


def _generation_probe_handoff_request(
    source_request: Mapping[str, Any],
    *,
    action: str,
    model_id: str,
    session_id: str,
    assignment_artifact: Mapping[str, Any],
    layer_start: int,
    layer_end: int,
    token_start: int,
    token_end: int,
    payload: bytes,
    workspace_root: Path,
    final_output: bool,
    next_frame_kind: str | None,
) -> dict[str, Any]:
    payload_hash = hashlib.sha256(bytes(payload or b"")).hexdigest()
    frame_kind = "activation"
    phase = (
        "decode_activation_batches"
        if action == "process_decode"
        else "prefill_activation_batches"
    )
    workspace = _generation_probe_workspace_contract(
        workspace_root,
        action=action,
        session_id=session_id,
        model_id=model_id,
        layer_start=layer_start,
        layer_end=layer_end,
        token_start=token_start,
        token_end=token_end,
        final_output=final_output,
        next_frame_kind=next_frame_kind,
    )
    request: dict[str, Any] = {
        "schemaVersion": 1,
        "abi": LLAMA_CPP_EXTERNAL_SHARD_HANDOFF_ABI,
        "action": action,
        "adapterId": str(source_request.get("adapterId") or "generation-probe"),
        "adapterVersion": str(source_request.get("adapterVersion") or "generation-probe/0.1"),
        "backend": str(source_request.get("backend") or "llama.cpp-patched"),
        "backendVersion": source_request.get("backendVersion"),
        "frame": {
            "sessionId": session_id,
            "batchId": f"{session_id}-{action}-{layer_start}-{layer_end}",
            "modelId": model_id,
            "frameKind": frame_kind,
            "phase": phase,
            "layerStart": layer_start,
            "layerEnd": layer_end,
            "tokenStart": token_start,
            "tokenEnd": token_end,
            "payloadSha256Hex": payload_hash,
            "metadata": {},
        },
        "shardSpec": {
            "modelId": model_id,
            "backend": "llama.cpp-patched",
            "requiresPatchedBackend": True,
            "frameKind": frame_kind,
            "phase": phase,
            "layerStart": layer_start,
            "layerEnd": layer_end,
            "tokenStart": token_start,
            "tokenEnd": token_end,
        },
        "payloadSha256Hex": payload_hash,
        "payloadBase64": base64.b64encode(bytes(payload or b"")).decode("ascii"),
        "localArtifactResolution": {
            "schemaVersion": 1,
            "modelId": model_id,
            "assignmentArtifact": dict(assignment_artifact),
        },
        "outputContract": _generation_probe_output_contract(
            final_output=final_output,
            next_frame_kind=next_frame_kind,
        ),
        "productionRequirements": (
            dict(source_request.get("productionRequirements"))
            if isinstance(source_request.get("productionRequirements"), Mapping)
            else None
        ),
        "managedRuntime": (
            dict(source_request.get("managedRuntime"))
            if isinstance(source_request.get("managedRuntime"), Mapping)
            else None
        ),
        "executionWorkspace": workspace,
    }
    local_resolution = source_request.get("localArtifactResolution")
    if isinstance(local_resolution, Mapping):
        for key in ("catalogId", "version", "preferredFilename", "manifestBackend"):
            if local_resolution.get(key) is not None:
                request["localArtifactResolution"][key] = local_resolution.get(key)
    return request


def _generation_probe_workspace_contract(
    workspace_root: Path,
    *,
    action: str,
    session_id: str,
    model_id: str,
    layer_start: int,
    layer_end: int,
    token_start: int,
    token_end: int,
    final_output: bool,
    next_frame_kind: str | None,
) -> dict[str, Any]:
    root = (workspace_root / f"layers-{layer_start}-{layer_end}").resolve()
    inputs_dir = (root / "inputs").resolve()
    outputs_dir = (root / "outputs").resolve()
    state_dir = (root / "state").resolve()
    for path in (root, inputs_dir, outputs_dir, state_dir):
        path.mkdir(parents=True, exist_ok=True)
    expected_kind = "final_output"
    if not final_output:
        expected_kind = (
            "decode_state" if str(next_frame_kind or "").lower() == "decode"
            else "activation_state"
        )
    return {
        "schemaVersion": 1,
        "abi": EXECUTION_WORKSPACE_ABI,
        "root": str(root),
        "inputsDir": str(inputs_dir),
        "outputsDir": str(outputs_dir),
        "stateFilesDir": str(state_dir),
        "manifestPath": str((root / "execution-workspace.json").resolve()),
        "action": action,
        "sessionId": session_id,
        "modelId": model_id,
        "layerStart": layer_start,
        "layerEnd": layer_end,
        "tokenStart": token_start,
        "tokenEnd": token_end,
        "requiresFinalOutput": bool(final_output),
        "expectedOutputKind": expected_kind,
    }


def _generation_probe_output_contract(
    *,
    final_output: bool,
    next_frame_kind: str | None,
) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "schemaVersion": 1,
        "requiresFinalOutput": bool(final_output),
        "requiresOutputFrameMetadata": not bool(final_output),
        "outputPayloadHashSource": "computed_output_payload",
    }
    if not final_output:
        frame_kind = str(next_frame_kind or "activation").strip() or "activation"
        contract["frameMetadataTemplate"] = {
            "frameKind": frame_kind,
            "payloadSha256Hex": "<computed-output-sha256>",
            "llmHandoff": {
                "tensor": {"sha256Hex": "<computed-output-sha256>"},
            },
        }
    return contract


def _generation_probe_response_payload(response: Mapping[str, Any]) -> bytes:
    payload_file = response.get("outputPayloadFile")
    if isinstance(payload_file, Mapping):
        raw_path = str(payload_file.get("path") or "").strip()
        if not raw_path:
            raise ValueError("CAI assignment generationProbe output path is missing.")
        path = Path(raw_path).expanduser().resolve()
        payload = path.read_bytes()
        declared_hash = str(payload_file.get("sha256Hex") or "").strip().lower()
        if declared_hash and declared_hash != hashlib.sha256(payload).hexdigest():
            raise ValueError("CAI assignment generationProbe output hash mismatch.")
        return payload
    raw_base64 = str(response.get("outputPayloadBase64") or "").strip()
    if raw_base64:
        return base64.b64decode(raw_base64.encode("ascii"), validate=True)
    raise ValueError("CAI assignment generationProbe output payload is missing.")


def _generation_probe_output_text(payload: bytes) -> str:
    text = bytes(payload or b"").decode("utf-8", errors="replace")
    if not text:
        return ""
    stripped = text.strip()
    if not stripped:
        return text
    try:
        parsed = json.loads(stripped)
    except Exception:
        return text
    if isinstance(parsed, Mapping):
        for key in ("textUtf8", "outputText", "text", "answer"):
            value = str(parsed.get(key) or "")
            if value:
                return value
    return text


def _generation_probe_process_proves_real(response: Mapping[str, Any]) -> bool:
    metrics = response.get("metrics")
    if not isinstance(metrics, Mapping):
        return False
    if _nested_bool(metrics, "assignmentExecutorRealModelExecution") is not True:
        return False
    if _nested_bool(metrics, "realLayerExecution") is not True:
        return False
    if _nested_reference_executor_backend(metrics):
        return False
    if _nested_bool(metrics, "usedFullModelForLayerRange") is True:
        return False
    if _nested_bool(metrics, "assignmentArtifactPresent") is not True:
        return False
    return _nested_bool(metrics, "shardOnlyLoadingReady") is True


def _process_prefill(request: Mapping[str, Any]) -> dict[str, Any]:
    artifact, session_metrics = _resolved_assignment_artifact_context(
        request,
        action="process_prefill",
    )
    payload = decode_native_engine_input_payload(
        request,
        error_prefix="CAI assignment artifact engine",
    )
    executor_result = _run_assignment_executor(
        request,
        action="process_prefill",
        artifact=artifact,
        input_payload=payload,
    )
    if executor_result is not None:
        output_state_kind = (
            executor_result.output_kind
            or _request_expected_output_kind(request)
        )
        output_payload = executor_result.output_payload
        executor_metrics = executor_result.metrics
        native_artifact_kind, native_fallback_mode = (
            _native_execution_artifact_from_executor_metrics(
                request,
                executor_metrics,
            )
        )
    else:
        output_state_kind = _assignment_state_kind_for_output(
            request,
            default="assignment_activation",
        )
        output_payload = _build_assignment_state_payload(
            request,
            artifact,
            state_kind=output_state_kind,
            source_payload=payload,
        )
        executor_metrics = {}
        native_artifact_kind = "assignment"
        native_fallback_mode = None
    managed_session_metrics = _stage_managed_session_files(
        request,
        action="process_prefill",
        artifact=artifact,
        input_payload=payload,
        output_payload=output_payload,
    )
    return build_native_engine_process_response(
        request,
        output_payload,
        metrics={
            "backendAction": "process_prefill",
            "backendMode": ASSIGNMENT_ARTIFACT_ENGINE_ID,
            "inputTokenCount": _request_token_count(request, payload),
            "outputTokenCount": 0,
            "assignmentStateKind": output_state_kind,
            "assignmentFinalOutput": False,
            "assignmentArtifactBytesRead": artifact.bytes_read,
            "assignmentArtifactChunkCount": len(artifact.chunk_ranges),
            "assignmentArtifactDigestSource": artifact.digest_source,
            "assignmentArtifactCoverageMode": artifact.coverage_mode,
            "assignmentArtifactCoveredByteCount": artifact.covered_byte_count,
            "assignmentArtifactCoveredRangeCount": artifact.covered_range_count,
            **executor_metrics,
            **session_metrics,
            **managed_session_metrics,
        },
        artifact_kind=native_artifact_kind,
        fallback_mode=native_fallback_mode,
        error_prefix="CAI assignment artifact engine",
    )


def _process_decode(request: Mapping[str, Any]) -> dict[str, Any]:
    artifact, session_metrics = _resolved_assignment_artifact_context(
        request,
        action="process_decode",
    )
    payload = decode_native_engine_input_payload(
        request,
        error_prefix="CAI assignment artifact engine",
    )
    final_output = _request_requires_final_output(request)
    executor_result = _run_assignment_executor(
        request,
        action="process_decode",
        artifact=artifact,
        input_payload=payload,
    )
    if executor_result is not None:
        output_state_kind = (
            executor_result.output_kind
            or _request_expected_output_kind(request)
        )
        output_payload = executor_result.output_payload
        source_payload = payload
        executor_metrics = executor_result.metrics
        native_artifact_kind, native_fallback_mode = (
            _native_execution_artifact_from_executor_metrics(
                request,
                executor_metrics,
            )
        )
    else:
        envelope = _assignment_state_payload(payload)
        expected_model_id = str(envelope.get("modelId") or "").strip()
        actual_model_id = _request_model_id(request)
        if expected_model_id and actual_model_id and expected_model_id != actual_model_id:
            raise ValueError(
                "CAI assignment artifact engine modelId does not match input state."
            )
        source_payload = _assignment_source_payload(envelope)
        source_text = _assignment_source_text(envelope, source_payload)
        if not source_text and source_payload:
            source_text = source_payload.decode("utf-8", errors="replace")
        input_state_digest = _assignment_state_digest(envelope)
        input_state_kind = str(envelope.get("stateKind") or "").strip()
        if final_output:
            prompt_preview = source_text[:80]
            output_text = (
                f"assignment:{artifact.window_digest[:16]}:{input_state_digest[:16]}"
            )
            if prompt_preview:
                output_text += ":" + prompt_preview
            output_payload = json.dumps(
                {
                    "schemaVersion": 1,
                    "abi": ASSIGNMENT_OUTPUT_PAYLOAD_ABI,
                    "textUtf8": output_text,
                    "assignmentDigest": artifact.window_digest,
                    "inputAssignmentDigest": str(
                        ((envelope.get("assignmentArtifact") or {}).get("digest")) or ""
                    ).strip()
                    or None,
                    "activationStateDigest": str(
                        envelope.get("activationStateDigest") or ""
                    ),
                    "decodeStateDigest": str(envelope.get("decodeStateDigest") or ""),
                    "stateDigest": input_state_digest,
                    "inputStateKind": input_state_kind or None,
                    "modelId": actual_model_id,
                    "layerStart": artifact.layer_start,
                    "layerEnd": artifact.layer_end,
                },
                sort_keys=True,
            ).encode("utf-8")
            output_state_kind = "final_output"
        else:
            output_state_kind = _assignment_state_kind_for_output(
                request,
                default="assignment_decode",
            )
            output_payload = _build_assignment_state_payload(
                request,
                artifact,
                state_kind=output_state_kind,
                source_payload=source_payload,
                input_state_digest=input_state_digest,
                input_state_kind=input_state_kind or None,
                )
        executor_metrics = {}
        native_artifact_kind = "assignment"
        native_fallback_mode = None
    managed_session_metrics = _stage_managed_session_files(
        request,
        action="process_decode",
        artifact=artifact,
        input_payload=payload,
        output_payload=output_payload,
    )
    return build_native_engine_process_response(
        request,
        output_payload,
        metrics={
            "backendAction": "process_decode",
            "backendMode": ASSIGNMENT_ARTIFACT_ENGINE_ID,
            "inputTokenCount": _request_token_count(request, source_payload or payload),
            "outputTokenCount": 1 if final_output else 0,
            "assignmentStateKind": output_state_kind,
            "assignmentFinalOutput": final_output,
            "assignmentArtifactBytesRead": artifact.bytes_read,
            "assignmentArtifactChunkCount": len(artifact.chunk_ranges),
            "assignmentArtifactDigestSource": artifact.digest_source,
            "assignmentArtifactCoverageMode": artifact.coverage_mode,
            "assignmentArtifactCoveredByteCount": artifact.covered_byte_count,
            "assignmentArtifactCoveredRangeCount": artifact.covered_range_count,
            **executor_metrics,
            **session_metrics,
            **managed_session_metrics,
        },
        artifact_kind=native_artifact_kind,
        fallback_mode=native_fallback_mode,
        error_prefix="CAI assignment artifact engine",
    )


def _finalize(request: Mapping[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "backendFinalized": True,
        "backendMode": ASSIGNMENT_ARTIFACT_ENGINE_ID,
    }
    session_key = _request_session_key(request)
    if session_key is not None:
        metrics["assignmentArtifactSessionKey"] = _session_key_text(session_key)
        released = _ENGINE_SESSIONS.pop(session_key, None)
        metrics["assignmentArtifactSessionReleased"] = released is not None
        if released is not None:
            metrics.update(
                _stage_managed_session_files(
                    request,
                    action="finalize",
                    artifact=released.context,
                    finalized=True,
                )
            )
            executor_metrics = _run_assignment_executor_lifecycle(
                request,
                action="finalize",
                artifact=released.context,
            )
            if executor_metrics:
                metrics.update(executor_metrics)
    return {
        "status": "ok",
        "metrics": metrics,
    }


def _native_execution_artifact_from_executor_metrics(
    request: Mapping[str, Any],
    executor_metrics: Mapping[str, Any],
) -> tuple[str, str | None]:
    used_full_model = _nested_bool(executor_metrics, "usedFullModelForLayerRange")
    shard_only_ready = _nested_bool(executor_metrics, "shardOnlyLoadingReady")
    assignment_present = _nested_bool(executor_metrics, "assignmentArtifactPresent")
    reference_backend = _nested_reference_executor_backend(executor_metrics)
    if _request_requires_shard_only_loading(request):
        if reference_backend:
            raise ValueError(
                "CAI assignment artifact engine executor used reference "
                f"{reference_backend} backend while shard-only loading is required."
            )
        if used_full_model is True:
            raise ValueError(
                "CAI assignment artifact engine executor used full-model "
                "layer-range execution while shard-only loading is required."
            )
        if shard_only_ready is not True:
            raise ValueError(
                "CAI assignment artifact engine executor did not prove "
                "shard-only loading."
            )
        if assignment_present is False:
            raise ValueError(
                "CAI assignment artifact engine executor did not prove "
                "assignmentArtifact-backed loading."
            )
    if used_full_model is True:
        local_resolution = request.get("localArtifactResolution")
        model_artifact = (
            local_resolution.get("modelArtifact")
            if isinstance(local_resolution, Mapping)
            and isinstance(local_resolution.get("modelArtifact"), Mapping)
            else None
        )
        if model_artifact is None:
            raise ValueError(
                "CAI assignment artifact engine executor used full-model "
                "layer-range execution but modelArtifact is unavailable."
            )
        return "model", "full_model"
    if reference_backend:
        return "assignment", "slot_state_reference"
    return "assignment", None


def _assignment_executor_metrics_prove_shard_only(
    executor_metrics: Mapping[str, Any],
) -> bool:
    if _nested_reference_executor_backend(executor_metrics):
        return False
    if _nested_bool(executor_metrics, "usedFullModelForLayerRange") is True:
        return False
    if _nested_bool(executor_metrics, "shardOnlyLoadingReady") is not True:
        return False
    if _nested_bool(executor_metrics, "assignmentArtifactPresent") is not True:
        return False
    return True


def _request_requires_shard_only_loading(request: Mapping[str, Any]) -> bool:
    requirements = request.get("productionRequirements")
    if not isinstance(requirements, Mapping):
        return False
    return bool(
        _truthy(requirements.get("requiresShardOnlyLoading"))
        or _truthy(requirements.get("forbidFullModelFallback"))
    )


def _nested_bool(value: Any, key: str) -> bool | None:
    if isinstance(value, Mapping):
        if key in value and isinstance(value.get(key), bool):
            return bool(value.get(key))
        for child in value.values():
            found = _nested_bool(child, key)
            if found is not None:
                return found
    if isinstance(value, list | tuple):
        for child in value:
            found = _nested_bool(child, key)
            if found is not None:
                return found
    return None


def _nested_int(value: Any, key: str) -> int | None:
    if isinstance(value, Mapping):
        if key in value:
            parsed = _optional_int(value.get(key))
            if parsed is not None:
                return parsed
        for child in value.values():
            found = _nested_int(child, key)
            if found is not None:
                return found
    if isinstance(value, list | tuple):
        for child in value:
            found = _nested_int(child, key)
            if found is not None:
                return found
    return None


def _nested_reference_executor_backend(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            clean_key = str(key or "").strip()
            if clean_key in {
                "slotStateMetrics",
                "slotStateBackendMode",
                "slotStateReferenceBackend",
                "usedSlotStateBackend",
            }:
                if isinstance(child, Mapping):
                    nested = _nested_reference_executor_backend(child)
                    return nested or "slot_state"
                if _truthy(child):
                    return "slot_state"
            if clean_key in {
                "backendMode",
                "engineBackendMode",
                "executorBackendMode",
                "slotStateBackendMode",
            }:
                mode = str(child or "").strip()
                if mode in REFERENCE_EXECUTOR_BACKEND_MODES:
                    return mode
            nested = _nested_reference_executor_backend(child)
            if nested is not None:
                return nested
    if isinstance(value, list | tuple):
        for child in value:
            nested = _nested_reference_executor_backend(child)
            if nested is not None:
                return nested
    return None


def _validate_executor_metrics_for_shard_only_request(
    request: Mapping[str, Any],
    executor_metrics: Mapping[str, Any],
) -> None:
    if not _request_requires_shard_only_loading(request):
        return
    reference_backend = _nested_reference_executor_backend(executor_metrics)
    if reference_backend:
        raise ValueError(
            "CAI assignment artifact engine executor used reference "
            f"{reference_backend} backend while shard-only loading is required."
        )
    used_full_model = _nested_bool(executor_metrics, "usedFullModelForLayerRange")
    if used_full_model is True:
        raise ValueError(
            "CAI assignment artifact engine executor used full-model "
            "layer-range execution while shard-only loading is required."
        )
    shard_only_ready = _nested_bool(executor_metrics, "shardOnlyLoadingReady")
    if shard_only_ready is not True:
        raise ValueError(
            "CAI assignment artifact engine executor did not prove "
            "shard-only loading."
        )
    assignment_present = _nested_bool(executor_metrics, "assignmentArtifactPresent")
    if assignment_present is False:
        raise ValueError(
            "CAI assignment artifact engine executor did not prove "
            "assignmentArtifact-backed loading."
        )


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _assignment_artifact_context(
    request: Mapping[str, Any],
) -> AssignmentArtifactContext:
    assignment = _request_assignment_artifact_mapping(request)
    if not isinstance(assignment, Mapping):
        raise ValueError(
            "CAI assignment artifact engine assignmentArtifact is missing."
        )
    artifact_source = str(assignment.get("source") or "").strip()
    if not artifact_source:
        raise ValueError(
            "CAI assignment artifact engine assignmentArtifact source is missing."
        )
    local_path = Path(str(assignment.get("localPath") or "")).expanduser().resolve()
    if not local_path.exists() or not local_path.is_file():
        raise ValueError(
            "CAI assignment artifact engine assignmentArtifact path is unavailable."
        )
    chunk_ranges = _assignment_chunk_ranges(assignment)
    coverage = resolve_assignment_artifact_coverage(
        assignment,
        error_prefix="CAI assignment artifact engine",
    )
    layer_start = _request_bound(request, "layerStart")
    layer_end = _request_bound(request, "layerEnd")
    validate_assignment_artifact_chunk_layer_coverage(
        assignment,
        layer_start=layer_start,
        layer_end=layer_end,
        error_prefix="CAI assignment artifact engine",
    )
    expected_digest = str(assignment.get("expectedDigest") or "").strip() or None
    window_digest, digest_source, bytes_read = _assignment_window_digest(
        local_path,
        chunk_ranges=chunk_ranges,
        expected_digest=expected_digest,
    )
    return AssignmentArtifactContext(
        artifact_id=str(assignment.get("artifactId") or "").strip() or None,
        artifact_source=artifact_source,
        local_path=local_path,
        size_bytes=int(
            _positive_int(assignment.get("sizeBytes")) or local_path.stat().st_size
        ),
        layer_start=layer_start,
        layer_end=layer_end,
        expected_digest=expected_digest,
        chunk_ranges=chunk_ranges,
        window_digest=window_digest,
        digest_source=digest_source,
        bytes_read=bytes_read,
        coverage_mode=(
            str(coverage.materialization_mode).strip() if coverage is not None else None
        ),
        covered_byte_count=(
            int(coverage.covered_byte_count) if coverage is not None else None
        ),
        covered_range_count=(
            int(coverage.covered_range_count) if coverage is not None else None
        ),
        zero_filled_outside_covered_ranges=(
            bool(coverage.zero_filled_outside_covered_ranges)
            if coverage is not None
            else None
        ),
    )


def _resolved_assignment_artifact_context(
    request: Mapping[str, Any],
    *,
    action: str,
) -> tuple[AssignmentArtifactContext, dict[str, Any]]:
    session_key = _request_session_key(request)
    model_id = _request_model_id(request)
    session_metrics: dict[str, Any] = {
        "assignmentArtifactSessionCacheHit": False,
        "assignmentArtifactSessionLoaded": False,
        "assignmentArtifactResidentShardHit": False,
        "assignmentArtifactResidentShardLoaded": False,
    }
    if session_key is not None:
        session_metrics["assignmentArtifactSessionKey"] = _session_key_text(session_key)
    assignment_mapping = _request_assignment_artifact_mapping(request)
    resident_key = _assignment_resident_artifact_key_from_request(
        request,
        assignment_mapping=assignment_mapping,
    )
    cached = _ENGINE_SESSIONS.get(session_key) if session_key is not None else None
    if cached is not None:
        mismatch_reason = (
            _assignment_mapping_mismatch_reason(assignment_mapping, cached.context)
            if assignment_mapping is not None
            else None
        )
        if mismatch_reason:
            raise ValueError(
                "CAI assignment artifact engine assignment session drifted between load_shard and "
                f"{action}: {mismatch_reason}."
            )
        session_metrics["assignmentArtifactSessionCacheHit"] = True
        session_metrics["assignmentArtifactSessionLoaded"] = True
        return _resident_assignment_artifact_context(cached.context), session_metrics
    if resident_key is not None:
        resident_context = _ENGINE_RESIDENT_ARTIFACTS.get(resident_key)
        if resident_context is not None:
            if assignment_mapping is not None and not _assignment_mapping_matches_context(
                assignment_mapping,
                resident_context,
            ):
                raise ValueError(
                    "CAI assignment artifact engine resident shard mapping drifted from the request."
                )
            if session_key is not None:
                _ENGINE_SESSIONS[session_key] = AssignmentLoadedSession(
                    key=session_key,
                    context=resident_context,
                    loaded_action=action,
                )
                session_metrics["assignmentArtifactSessionLoaded"] = True
            session_metrics["assignmentArtifactResidentShardHit"] = True
            return _resident_assignment_artifact_context(resident_context), session_metrics
    context = _assignment_artifact_context(request)
    resident_context = _resident_assignment_artifact_context(context)
    resident_key = resident_key or _assignment_resident_artifact_key_from_context(
        model_id,
        resident_context,
    )
    if resident_key is not None:
        _ENGINE_RESIDENT_ARTIFACTS[resident_key] = resident_context
        session_metrics["assignmentArtifactResidentShardLoaded"] = True
    if session_key is not None:
        _ENGINE_SESSIONS[session_key] = AssignmentLoadedSession(
            key=session_key,
            context=resident_context,
            loaded_action=action,
        )
        session_metrics["assignmentArtifactSessionLoaded"] = True
    return context, session_metrics


def _resident_assignment_artifact_context(
    context: AssignmentArtifactContext,
) -> AssignmentArtifactContext:
    return replace(context, bytes_read=0)


def _stage_managed_session_files(
    request: Mapping[str, Any],
    *,
    action: str,
    artifact: AssignmentArtifactContext,
    input_payload: bytes | None = None,
    output_payload: bytes | None = None,
    finalized: bool = False,
) -> dict[str, Any]:
    workspace_contract = _execution_workspace_contract(request)
    workspace_root = _managed_session_workspace_root(
        request,
        workspace_contract=workspace_contract,
    )
    if workspace_root is None:
        return {"assignmentManagedRuntimeUsed": False}
    inputs_dir = _workspace_subdir(
        workspace_contract,
        field_name="inputsDir",
        default=workspace_root / "inputs",
    )
    outputs_dir = _workspace_subdir(
        workspace_contract,
        field_name="outputsDir",
        default=workspace_root / "outputs",
    )
    state_files_dir = _workspace_subdir(
        workspace_contract,
        field_name="stateFilesDir",
        default=workspace_root / "state",
    )
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    state_files_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "assignmentManagedRuntimeUsed": True,
        "assignmentManagedSessionWorkspaceRoot": str(workspace_root),
    }
    input_path: Path | None = None
    output_path: Path | None = None
    if input_payload is not None:
        input_hash = hashlib.sha256(input_payload).hexdigest()
        input_path = (
            inputs_dir / f"{action}-input-{input_hash[:16]}.bin"
        ).resolve()
        input_path.write_bytes(bytes(input_payload or b""))
        metrics["assignmentManagedInputPath"] = str(input_path)
        metrics["assignmentManagedInputSha256Hex"] = input_hash
    if output_payload is not None:
        output_hash = hashlib.sha256(output_payload).hexdigest()
        output_suffix = ".json" if _looks_like_json_payload(output_payload) else ".bin"
        output_path = (
            outputs_dir / f"{action}-output-{output_hash[:16]}{output_suffix}"
        ).resolve()
        output_path.write_bytes(bytes(output_payload or b""))
        metrics["assignmentManagedOutputPath"] = str(output_path)
        metrics["assignmentManagedOutputSha256Hex"] = output_hash
    manifest_path = (state_files_dir / "assignment-session.json").resolve()
    manifest = _read_managed_session_manifest(manifest_path)
    manifest.update(
        {
            "schemaVersion": 1,
            "abi": ASSIGNMENT_MANAGED_SESSION_ABI,
            "engineId": ASSIGNMENT_ARTIFACT_ENGINE_ID,
            "sessionId": _request_session_id(request),
            "modelId": _request_model_id(request),
            "layerStart": artifact.layer_start,
            "layerEnd": artifact.layer_end,
            "artifact": {
                "artifactId": artifact.artifact_id,
                "source": artifact.artifact_source,
                "localPath": str(artifact.local_path.resolve()),
                "digest": artifact.window_digest,
                "digestSource": artifact.digest_source,
                "sizeBytes": artifact.size_bytes,
                "chunkCount": len(artifact.chunk_ranges),
                "coverageMode": artifact.coverage_mode,
                "coveredByteCount": artifact.covered_byte_count,
                "coveredRangeCount": artifact.covered_range_count,
            },
            "lastAction": action,
            "finalized": bool(finalized),
        }
    )
    if input_path is not None:
        manifest["lastInputPath"] = str(input_path)
    if output_path is not None:
        manifest["lastOutputPath"] = str(output_path)
    manifest.update(_managed_payload_manifest_fields(output_payload))
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    metrics["assignmentManagedSessionManifestPath"] = str(manifest_path)
    if "latestStateDigest" in manifest:
        metrics["assignmentManagedLatestStateDigest"] = manifest["latestStateDigest"]
    if "latestStateKind" in manifest:
        metrics["assignmentManagedLatestStateKind"] = manifest["latestStateKind"]
    if finalized:
        metrics["assignmentManagedSessionFinalized"] = True
    workspace_manifest_path = _workspace_manifest_path(
        workspace_contract,
        default=workspace_root / "execution-workspace.json",
    )
    _write_execution_workspace_manifest(
        workspace_manifest_path,
        workspace_root=workspace_root,
        workspace_contract=workspace_contract,
        action=action,
        artifact=artifact,
        session_id=_request_session_id(request),
        model_id=_request_model_id(request),
        input_path=input_path,
        output_path=output_path,
        payload_fields=_managed_payload_manifest_fields(output_payload),
        finalized=finalized,
    )
    metrics["assignmentExecutionWorkspaceManifestPath"] = str(workspace_manifest_path)
    return metrics


def _execution_workspace_contract(request: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = request.get("executionWorkspace")
    if not isinstance(value, Mapping):
        return None
    if str(value.get("abi") or "").strip() != EXECUTION_WORKSPACE_ABI:
        return None
    return value


def _managed_session_workspace_root(
    request: Mapping[str, Any],
    *,
    workspace_contract: Mapping[str, Any] | None,
) -> Path | None:
    if isinstance(workspace_contract, Mapping):
        raw_root = str(workspace_contract.get("root") or "").strip()
        if raw_root:
            return Path(raw_root).expanduser().resolve()
    managed_runtime = resolve_managed_llama_cpp_runtime(request)
    if managed_runtime is None or managed_runtime.session_paths is None:
        return None
    return (managed_runtime.session_paths.state_dir / "assignment-engine").resolve()


def _workspace_subdir(
    workspace_contract: Mapping[str, Any] | None,
    *,
    field_name: str,
    default: Path,
) -> Path:
    if isinstance(workspace_contract, Mapping):
        raw_path = str(workspace_contract.get(field_name) or "").strip()
        if raw_path:
            return Path(raw_path).expanduser().resolve()
    return default.resolve()


def _workspace_manifest_path(
    workspace_contract: Mapping[str, Any] | None,
    *,
    default: Path,
) -> Path:
    if isinstance(workspace_contract, Mapping):
        raw_path = str(workspace_contract.get("manifestPath") or "").strip()
        if raw_path:
            return Path(raw_path).expanduser().resolve()
    return default.resolve()


def _read_managed_session_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8") or "{}")
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _managed_payload_manifest_fields(payload: bytes | None) -> dict[str, Any]:
    if payload is None:
        return {}
    try:
        parsed = json.loads(bytes(payload or b"").decode("utf-8") or "{}")
    except Exception:
        return {}
    if not isinstance(parsed, Mapping):
        return {}
    fields: dict[str, Any] = {}
    abi = str(parsed.get("abi") or "").strip()
    if abi == ASSIGNMENT_STATE_PAYLOAD_ABI:
        state_digest = str(parsed.get("stateDigest") or "").strip()
        state_kind = str(parsed.get("stateKind") or "").strip()
        if state_digest:
            fields["latestStateDigest"] = state_digest
        if state_kind:
            fields["latestStateKind"] = state_kind
    elif abi == ASSIGNMENT_OUTPUT_PAYLOAD_ABI:
        state_digest = str(parsed.get("stateDigest") or "").strip()
        if state_digest:
            fields["latestStateDigest"] = state_digest
        fields["latestStateKind"] = "final_output"
        text_utf8 = str(parsed.get("textUtf8") or "")
        if text_utf8:
            fields["latestOutputPreview"] = text_utf8[:160]
    return fields


def _write_execution_workspace_manifest(
    path: Path,
    *,
    workspace_root: Path,
    workspace_contract: Mapping[str, Any] | None,
    action: str,
    artifact: AssignmentArtifactContext,
    session_id: str | None,
    model_id: str | None,
    input_path: Path | None,
    output_path: Path | None,
    payload_fields: Mapping[str, Any],
    finalized: bool,
) -> None:
    manifest = _read_managed_session_manifest(path)
    manifest.update(
        {
            "schemaVersion": 1,
            "abi": EXECUTION_WORKSPACE_ABI,
            "root": str(workspace_root.resolve()),
            "action": action,
            "sessionId": session_id,
            "modelId": model_id,
            "layerStart": artifact.layer_start,
            "layerEnd": artifact.layer_end,
            "artifactDigest": artifact.window_digest,
            "artifactLocalPath": str(artifact.local_path.resolve()),
            "artifactSource": artifact.artifact_source,
            "artifactChunkCount": len(artifact.chunk_ranges),
            "finalized": bool(finalized),
        }
    )
    if isinstance(workspace_contract, Mapping) and action != "finalize":
        for field_name in (
            "expectedOutputKind",
            "requiresFinalOutput",
            "tokenStart",
            "tokenEnd",
            "inputsDir",
            "outputsDir",
            "stateFilesDir",
        ):
            if workspace_contract.get(field_name) is not None:
                manifest[field_name] = workspace_contract.get(field_name)
    if input_path is not None:
        manifest["lastInputPath"] = str(input_path)
    if output_path is not None:
        manifest["lastOutputPath"] = str(output_path)
    manifest.update(dict(payload_fields))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _run_assignment_executor(
    request: Mapping[str, Any],
    *,
    action: str,
    artifact: AssignmentArtifactContext,
    input_payload: bytes,
) -> AssignmentExecutorResult | None:
    command = _assignment_executor_command()
    if not command:
        return None
    workspace_contract = _execution_workspace_contract(request)
    if not isinstance(workspace_contract, Mapping):
        raise ValueError(
            "CAI assignment artifact engine executionWorkspace is required when "
            f"{CAI_LLM_ASSIGNMENT_EXECUTOR_COMMAND_ENV} is configured."
        )
    staged_metrics = _stage_managed_session_files(
        request,
        action=action,
        artifact=artifact,
        input_payload=input_payload,
    )
    input_path = Path(str(staged_metrics.get("assignmentManagedInputPath") or "")).resolve()
    if not input_path.exists() or not input_path.is_file():
        raise ValueError(
            "CAI assignment artifact engine could not stage executor input payload."
        )
    input_hash = str(staged_metrics.get("assignmentManagedInputSha256Hex") or "").strip()
    expected_output_path = (
        _workspace_subdir(
            workspace_contract,
            field_name="outputsDir",
            default=Path(str(workspace_contract.get("root") or "")).resolve() / "outputs",
        )
        / f"{action}-executor-output.bin"
    ).resolve()
    expected_output_path.parent.mkdir(parents=True, exist_ok=True)
    executor_request = {
        "schemaVersion": 1,
        "abi": ASSIGNMENT_EXECUTOR_REQUEST_ABI,
        "action": action,
        "backendMode": ASSIGNMENT_ARTIFACT_ENGINE_ID,
        "sessionId": _request_session_id(request),
        "modelId": _request_model_id(request),
        "layerStart": artifact.layer_start,
        "layerEnd": artifact.layer_end,
        "tokenStart": _request_bound(request, "tokenStart"),
        "tokenEnd": _request_bound(request, "tokenEnd"),
        "requiresFinalOutput": _request_requires_final_output(request),
        "expectedOutputKind": _request_expected_output_kind(request),
        "inputPayloadFile": {
            "path": str(input_path),
            "sizeBytes": int(len(input_payload)),
            "sha256Hex": input_hash or hashlib.sha256(input_payload).hexdigest(),
        },
        "expectedOutputPayloadPath": str(expected_output_path),
        "assignmentArtifact": dict(_request_assignment_artifact_mapping(request) or {}),
        "localArtifactResolution": (
            dict(request.get("localArtifactResolution"))
            if isinstance(request.get("localArtifactResolution"), Mapping)
            else None
        ),
        "frame": dict(request.get("frame")) if isinstance(request.get("frame"), Mapping) else None,
        "shardSpec": dict(request.get("shardSpec")) if isinstance(request.get("shardSpec"), Mapping) else None,
        "outputContract": (
            dict(request.get("outputContract"))
            if isinstance(request.get("outputContract"), Mapping)
            else None
        ),
        "productionRequirements": (
            dict(request.get("productionRequirements"))
            if isinstance(request.get("productionRequirements"), Mapping)
            else None
        ),
        "managedRuntime": (
            dict(request.get("managedRuntime"))
            if isinstance(request.get("managedRuntime"), Mapping)
            else None
        ),
        "executionWorkspace": dict(workspace_contract),
    }
    response = _call_assignment_executor(command, executor_request, request)
    status = str(response.get("status") or "").strip().lower()
    if status not in {"ok", "ready", ""}:
        detail = str(response.get("error") or response.get("message") or "").strip()
        raise ValueError(
            "CAI assignment artifact engine executor returned non-ok status"
            + (f": {detail}" if detail else ".")
        )
    output_payload = _assignment_executor_output_payload(
        response,
        workspace_contract=workspace_contract,
        expected_output_path=expected_output_path,
    )
    output_kind = str(response.get("outputKind") or "").strip() or None
    response_metrics = response.get("metrics")
    metrics_payload = (
        dict(response_metrics) if isinstance(response_metrics, Mapping) else {}
    )
    reference_backend = _nested_reference_executor_backend(metrics_payload)
    result_metrics: dict[str, Any] = {
        "assignmentExecutorUsed": True,
        "assignmentExecutorCommand": str(command[0]),
        "assignmentExecutorMode": (
            "persistent_jsonl"
            if _assignment_executor_persistent_enabled()
            else "subprocess_per_request"
        ),
        "assignmentExecutorOutputKind": output_kind
        or _request_expected_output_kind(request),
    }
    if response.get("realModelExecution") is not None:
        result_metrics["assignmentExecutorRealModelExecution"] = bool(
            response.get("realModelExecution")
        )
    if reference_backend:
        result_metrics["assignmentExecutorReferenceBackend"] = reference_backend
    if metrics_payload:
        result_metrics["assignmentExecutorMetrics"] = metrics_payload
    _validate_executor_metrics_for_shard_only_request(request, result_metrics)
    return AssignmentExecutorResult(
        output_payload=output_payload,
        output_kind=output_kind,
        metrics=result_metrics,
    )


def _run_assignment_executor_lifecycle(
    request: Mapping[str, Any],
    *,
    action: str,
    artifact: AssignmentArtifactContext,
) -> dict[str, Any] | None:
    command = _assignment_executor_command()
    if not command:
        return None
    workspace_contract = _execution_workspace_contract(request)
    if not isinstance(workspace_contract, Mapping):
        raise ValueError(
            "CAI assignment artifact engine executionWorkspace is required when "
            f"{CAI_LLM_ASSIGNMENT_EXECUTOR_COMMAND_ENV} is configured."
        )
    executor_request = {
        "schemaVersion": 1,
        "abi": ASSIGNMENT_EXECUTOR_REQUEST_ABI,
        "action": action,
        "backendMode": ASSIGNMENT_ARTIFACT_ENGINE_ID,
        "sessionId": _request_session_id(request),
        "modelId": _request_model_id(request),
        "layerStart": artifact.layer_start,
        "layerEnd": artifact.layer_end,
        "tokenStart": _request_bound(request, "tokenStart"),
        "tokenEnd": _request_bound(request, "tokenEnd"),
        "requiresFinalOutput": _request_requires_final_output(request),
        "expectedOutputKind": _request_expected_output_kind(request),
        "assignmentArtifact": dict(_request_assignment_artifact_mapping(request) or {}),
        "localArtifactResolution": (
            dict(request.get("localArtifactResolution"))
            if isinstance(request.get("localArtifactResolution"), Mapping)
            else None
        ),
        "frame": dict(request.get("frame")) if isinstance(request.get("frame"), Mapping) else None,
        "shardSpec": dict(request.get("shardSpec")) if isinstance(request.get("shardSpec"), Mapping) else None,
        "outputContract": (
            dict(request.get("outputContract"))
            if isinstance(request.get("outputContract"), Mapping)
            else None
        ),
        "productionRequirements": (
            dict(request.get("productionRequirements"))
            if isinstance(request.get("productionRequirements"), Mapping)
            else None
        ),
        "managedRuntime": (
            dict(request.get("managedRuntime"))
            if isinstance(request.get("managedRuntime"), Mapping)
            else None
        ),
        "executionWorkspace": dict(workspace_contract),
    }
    response = _call_assignment_executor(command, executor_request, request)
    status = str(response.get("status") or "").strip().lower()
    if status not in {"ok", "ready"}:
        detail = str(response.get("error") or response.get("message") or "").strip()
        raise ValueError(
            "CAI assignment artifact engine executor returned non-ok status"
            + (f": {detail}" if detail else ".")
        )
    response_metrics = response.get("metrics")
    metrics_payload = (
        dict(response_metrics) if isinstance(response_metrics, Mapping) else {}
    )
    reference_backend = _nested_reference_executor_backend(metrics_payload)
    result_metrics: dict[str, Any] = {
        "assignmentExecutorUsed": True,
        "assignmentExecutorCommand": str(command[0]),
        "assignmentExecutorMode": (
            "persistent_jsonl"
            if _assignment_executor_persistent_enabled()
            else "subprocess_per_request"
        ),
        "assignmentExecutorStatus": status or ("ready" if action == "load_shard" else "ok"),
    }
    if response.get("realModelExecution") is not None:
        result_metrics["assignmentExecutorRealModelExecution"] = bool(
            response.get("realModelExecution")
        )
    if reference_backend:
        result_metrics["assignmentExecutorReferenceBackend"] = reference_backend
    if metrics_payload:
        result_metrics["assignmentExecutorMetrics"] = metrics_payload
    _validate_executor_metrics_for_shard_only_request(request, result_metrics)
    return result_metrics


def _assignment_executor_command() -> list[str]:
    raw = str(os.getenv(CAI_LLM_ASSIGNMENT_EXECUTOR_COMMAND_ENV) or "").strip()
    return split_llama_cpp_subprocess_command(raw)


def _call_assignment_executor(
    command: list[str],
    payload: Mapping[str, Any],
    request: Mapping[str, Any],
) -> Mapping[str, Any]:
    managed_runtime = resolve_managed_llama_cpp_runtime(request)
    cwd = None
    if managed_runtime is not None:
        if managed_runtime.repo_root is not None:
            cwd = str(managed_runtime.repo_root)
        elif managed_runtime.runtime_root is not None:
            cwd = str(managed_runtime.runtime_root)
    if _assignment_executor_persistent_enabled():
        client = _persistent_assignment_executor_client(command, cwd=cwd)
        return client.call(payload)
    completed = subprocess.run(
        command,
        input=json.dumps(dict(payload), sort_keys=True).encode("utf-8"),
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
            "CAI assignment artifact engine executor command failed"
            + (
                f" ({stderr_text})"
                if stderr_text
                else ""
            )
            + "."
        )
    try:
        parsed = json.loads(completed.stdout.decode("utf-8") or "{}")
    except Exception as exc:
        raise ValueError(
            "CAI assignment artifact engine executor response is invalid JSON."
        ) from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(
            "CAI assignment artifact engine executor response must be an object."
        )
    return parsed


def _assignment_executor_persistent_enabled() -> bool:
    raw = str(os.getenv(CAI_LLM_ASSIGNMENT_EXECUTOR_PERSISTENT_ENV) or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _assignment_executor_timeout_seconds() -> float:
    for name in (
        CAI_LLM_ASSIGNMENT_EXECUTOR_TIMEOUT_ENV,
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


def _persistent_assignment_executor_client(
    command: list[str],
    *,
    cwd: str | None,
) -> PersistentAssignmentExecutorClient:
    timeout_sec = _assignment_executor_timeout_seconds()
    key = (tuple(command), cwd, timeout_sec)
    existing = _PERSISTENT_ASSIGNMENT_EXECUTORS.get(key)
    if existing is not None and existing.alive:
        return existing
    if existing is not None:
        existing.close(kill=True)
        _PERSISTENT_ASSIGNMENT_EXECUTORS.pop(key, None)
    client = PersistentAssignmentExecutorClient(
        command,
        cwd=cwd,
        timeout_sec=timeout_sec,
    )
    _PERSISTENT_ASSIGNMENT_EXECUTORS[key] = client
    return client


def _assignment_executor_output_payload(
    response: Mapping[str, Any],
    *,
    workspace_contract: Mapping[str, Any],
    expected_output_path: Path,
) -> bytes:
    payload_file = response.get("outputPayloadFile")
    if isinstance(payload_file, Mapping):
        raw_path = str(payload_file.get("path") or "").strip()
        if not raw_path:
            raise ValueError(
                "CAI assignment artifact engine executor outputPayloadFile path is missing."
            )
        output_path = Path(raw_path).expanduser().resolve()
        outputs_dir = _workspace_subdir(
            workspace_contract,
            field_name="outputsDir",
            default=expected_output_path.parent,
        )
        if not _path_is_within(output_path, outputs_dir):
            raise ValueError(
                "CAI assignment artifact engine executor output path must stay within "
                "executionWorkspace.outputsDir."
            )
        if not output_path.exists() or not output_path.is_file():
            raise ValueError(
                "CAI assignment artifact engine executor output file is unavailable."
            )
        payload = output_path.read_bytes()
        _validate_executor_payload_hash(payload, payload_file)
        return payload
    raw_base64 = str(response.get("outputPayloadBase64") or "").strip()
    if raw_base64:
        try:
            return base64.b64decode(raw_base64.encode("ascii"), validate=True)
        except Exception as exc:
            raise ValueError(
                "CAI assignment artifact engine executor outputPayloadBase64 is invalid."
            ) from exc
    raise ValueError(
        "CAI assignment artifact engine executor did not return output payload."
    )


def _validate_executor_payload_hash(
    payload: bytes,
    payload_file: Mapping[str, Any],
) -> None:
    declared_hash = str(payload_file.get("sha256Hex") or "").strip()
    if not declared_hash:
        return
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != declared_hash:
        raise ValueError(
            "CAI assignment artifact engine executor output payload hash mismatch."
        )


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except Exception:
        return False
    return True


def _looks_like_json_payload(payload: bytes) -> bool:
    stripped = bytes(payload or b"").lstrip()
    return bool(stripped) and stripped[:1] in {b"{", b"["}


def _request_assignment_artifact_mapping(
    request: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    local_resolution = request.get("localArtifactResolution")
    if not isinstance(local_resolution, Mapping):
        return None
    assignment = local_resolution.get("assignmentArtifact")
    return assignment if isinstance(assignment, Mapping) else None


def _request_session_key(request: Mapping[str, Any]) -> AssignmentSessionKey | None:
    frame = request.get("frame")
    if not isinstance(frame, Mapping):
        return None
    session_id = str(frame.get("sessionId") or "").strip()
    if not session_id:
        return None
    return AssignmentSessionKey(
        session_id=session_id,
        model_id=_request_model_id(request),
        layer_start=_request_bound(request, "layerStart"),
        layer_end=_request_bound(request, "layerEnd"),
    )


def _request_session_id(request: Mapping[str, Any]) -> str | None:
    frame = request.get("frame")
    if not isinstance(frame, Mapping):
        return None
    session_id = str(frame.get("sessionId") or "").strip()
    return session_id or None


def _session_key_text(value: AssignmentSessionKey) -> str:
    return (
        f"{value.session_id}:{value.model_id or ''}:"
        f"{'' if value.layer_start is None else value.layer_start}-"
        f"{'' if value.layer_end is None else value.layer_end}"
    )


def _assignment_resident_artifact_key_from_request(
    request: Mapping[str, Any],
    *,
    assignment_mapping: Mapping[str, Any] | None,
) -> AssignmentResidentArtifactKey | None:
    if not isinstance(assignment_mapping, Mapping):
        return None
    local_path = str(assignment_mapping.get("localPath") or "").strip()
    if not local_path:
        return None
    try:
        resolved_path = str(Path(local_path).expanduser().resolve())
    except Exception:
        return None
    return AssignmentResidentArtifactKey(
        model_id=_request_model_id(request),
        local_path=resolved_path,
        layer_start=_request_bound(request, "layerStart"),
        layer_end=_request_bound(request, "layerEnd"),
        expected_digest=str(assignment_mapping.get("expectedDigest") or "").strip() or None,
        chunk_ranges_signature=_assignment_chunk_range_signature(
            _assignment_chunk_ranges(assignment_mapping)
        ),
    )


def _assignment_resident_artifact_key_from_context(
    model_id: str | None,
    context: AssignmentArtifactContext,
) -> AssignmentResidentArtifactKey:
    return AssignmentResidentArtifactKey(
        model_id=model_id,
        local_path=str(context.local_path.resolve()),
        layer_start=context.layer_start,
        layer_end=context.layer_end,
        expected_digest=context.expected_digest,
        chunk_ranges_signature=_assignment_chunk_range_signature(context.chunk_ranges),
    )


def _assignment_mapping_matches_context(
    assignment: Mapping[str, Any],
    context: AssignmentArtifactContext,
) -> bool:
    return _assignment_mapping_mismatch_reason(assignment, context) is None


def _assignment_mapping_mismatch_reason(
    assignment: Mapping[str, Any],
    context: AssignmentArtifactContext,
) -> str | None:
    local_path = str(assignment.get("localPath") or "").strip()
    if local_path:
        try:
            if Path(local_path).expanduser().resolve() != context.local_path.resolve():
                return "localPath mismatch"
        except Exception:
            return "localPath mismatch"
    layer_start = _optional_int(assignment.get("layerStart"))
    if layer_start is not None and layer_start != context.layer_start:
        return f"layerStart mismatch ({layer_start} != {context.layer_start})"
    layer_end = _optional_int(assignment.get("layerEnd"))
    if layer_end is not None and layer_end != context.layer_end:
        return f"layerEnd mismatch ({layer_end} != {context.layer_end})"
    expected_digest = str(assignment.get("expectedDigest") or "").strip()
    if expected_digest and context.expected_digest and expected_digest != context.expected_digest:
        return "expectedDigest mismatch"
    requested_ranges = _assignment_chunk_ranges(assignment)
    if requested_ranges and requested_ranges != context.chunk_ranges:
        return "chunkRanges mismatch"
    return None


def _assignment_chunk_range_signature(
    ranges: tuple[dict[str, Any], ...],
) -> tuple[
    tuple[str, int, int, str | None, int | None, int | None, tuple[str, ...]],
    ...,
]:
    return tuple(
        (
            str(item.get("chunkId") or "").strip(),
            int(item["offsetBytes"]),
            int(item["sizeBytes"]),
            str(item.get("sha256Hex") or "").strip().lower() or None,
            _optional_int(item.get("layerStart")),
            _optional_int(item.get("layerEnd")),
            tuple(
                str(name).strip()
                for name in item.get("tensorNames") or []
                if str(name).strip()
            ),
        )
        for item in ranges
    )


def _assignment_chunk_ranges(value: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_ranges = value.get("chunkRanges")
    if not isinstance(raw_ranges, list):
        return ()
    output: list[dict[str, Any]] = []
    for item in raw_ranges:
        if not isinstance(item, Mapping):
            continue
        offset_bytes = _positive_int(item.get("offsetBytes"))
        size_bytes = _positive_int(item.get("sizeBytes"))
        if offset_bytes is None or size_bytes is None:
            continue
        chunk: dict[str, Any] = {
            "chunkId": str(item.get("chunkId") or "").strip(),
            "offsetBytes": offset_bytes,
            "sizeBytes": size_bytes,
        }
        sha256_hex = str(item.get("sha256Hex") or "").strip().lower()
        if sha256_hex:
            chunk["sha256Hex"] = sha256_hex
        layer_start = _optional_int(item.get("layerStart"))
        layer_end = _optional_int(item.get("layerEnd"))
        if layer_start is not None:
            chunk["layerStart"] = layer_start
        if layer_end is not None:
            chunk["layerEnd"] = layer_end
        tensor_names = item.get("tensorNames")
        if isinstance(tensor_names, list):
            clean_tensor_names = [
                str(name).strip() for name in tensor_names if str(name).strip()
            ]
            if clean_tensor_names:
                chunk["tensorNames"] = clean_tensor_names
        output.append(chunk)
    return tuple(output)


def _assignment_window_digest(
    path: Path,
    *,
    chunk_ranges: tuple[dict[str, Any], ...],
    expected_digest: str | None,
) -> tuple[str, str, int]:
    if chunk_ranges:
        hasher = hashlib.sha256()
        bytes_read = 0
        with path.open("rb") as handle:
            for chunk in chunk_ranges:
                offset_bytes = int(chunk["offsetBytes"])
                size_bytes = int(chunk["sizeBytes"])
                handle.seek(offset_bytes)
                payload = handle.read(size_bytes)
                if len(payload) != size_bytes:
                    raise ValueError(
                        "CAI assignment artifact engine could not read a full chunk range."
                    )
                payload_hash = hashlib.sha256(payload).hexdigest()
                expected_hash = str(chunk.get("sha256Hex") or "").strip().lower()
                if expected_hash and payload_hash != expected_hash:
                    raise ValueError(
                        "CAI assignment artifact engine assignment chunk hash mismatch."
                    )
                hasher.update(str(chunk.get("chunkId") or "").encode("utf-8"))
                hasher.update(payload_hash.encode("ascii"))
                bytes_read += size_bytes
        return f"chunk_ranges:{hasher.hexdigest()}", "chunk_ranges", bytes_read
    if expected_digest:
        return expected_digest, "expected_digest", 0
    raise ValueError(
        "CAI assignment artifact engine needs assignment chunkRanges or expectedDigest."
    )


def _assignment_state_payload(payload: bytes) -> Mapping[str, Any]:
    try:
        parsed = json.loads(bytes(payload or b"").decode("utf-8") or "{}")
    except Exception as exc:
        raise ValueError(
            "CAI assignment artifact engine input state payload is invalid."
        ) from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(
            "CAI assignment artifact engine input state payload must be an object."
        )
    if str(parsed.get("abi") or "").strip() != ASSIGNMENT_STATE_PAYLOAD_ABI:
        raise ValueError(
            "CAI assignment artifact engine input state ABI is unsupported."
        )
    state_kind = str(parsed.get("stateKind") or "").strip()
    if state_kind not in {"assignment_activation", "assignment_decode"}:
        raise ValueError(
            "CAI assignment artifact engine input state kind is unsupported."
        )
    return parsed


def _build_assignment_state_payload(
    request: Mapping[str, Any],
    artifact: AssignmentArtifactContext,
    *,
    state_kind: str,
    source_payload: bytes,
    input_state_digest: str | None = None,
    input_state_kind: str | None = None,
) -> bytes:
    source_payload_hash = hashlib.sha256(source_payload).hexdigest()
    state_digest = hashlib.sha256(
        (
            state_kind
            + ":"
            + artifact.window_digest
            + ":"
            + source_payload_hash
            + ":"
            + str(input_state_digest or "")
            + ":"
            + str(artifact.layer_start)
            + ":"
            + str(artifact.layer_end)
            + ":"
            + str(_request_bound(request, "tokenStart"))
            + ":"
            + str(_request_bound(request, "tokenEnd"))
        ).encode("utf-8")
    ).hexdigest()
    payload: dict[str, Any] = {
        "schemaVersion": 1,
        "abi": ASSIGNMENT_STATE_PAYLOAD_ABI,
        "stateKind": state_kind,
        "modelId": _request_model_id(request),
        "layerStart": artifact.layer_start,
        "layerEnd": artifact.layer_end,
        "tokenStart": _request_bound(request, "tokenStart"),
        "tokenEnd": _request_bound(request, "tokenEnd"),
        "assignmentArtifact": {
            "artifactId": artifact.artifact_id,
            "source": artifact.artifact_source,
            "digest": artifact.window_digest,
            "sizeBytes": artifact.size_bytes,
            "chunkCount": len(artifact.chunk_ranges),
            "digestSource": artifact.digest_source,
        },
        "sourcePayloadSha256Hex": source_payload_hash,
        "sourcePayloadBase64": base64.b64encode(source_payload).decode("ascii"),
        "sourcePayloadUtf8": source_payload.decode("utf-8", errors="replace"),
        "stateDigest": state_digest,
    }
    if state_kind == "assignment_activation":
        payload["activationStateDigest"] = state_digest
    if state_kind == "assignment_decode":
        payload["decodeStateDigest"] = state_digest
    if input_state_digest:
        payload["inputStateDigest"] = input_state_digest
    if input_state_kind:
        payload["inputStateKind"] = input_state_kind
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def _assignment_source_payload(envelope: Mapping[str, Any]) -> bytes:
    for field_name in ("sourcePayloadBase64", "inputPayloadBase64"):
        payload = _decode_optional_base64(envelope.get(field_name))
        if payload:
            return payload
    return b""


def _assignment_source_text(
    envelope: Mapping[str, Any],
    source_payload: bytes,
) -> str:
    for field_name in ("sourcePayloadUtf8", "inputPayloadUtf8"):
        value = str(envelope.get(field_name) or "").strip()
        if value:
            return value
    if source_payload:
        return source_payload.decode("utf-8", errors="replace")
    return ""


def _assignment_state_digest(envelope: Mapping[str, Any]) -> str:
    for field_name in ("stateDigest", "decodeStateDigest", "activationStateDigest"):
        value = str(envelope.get(field_name) or "").strip()
        if value:
            return value
    return ""


def _decode_optional_base64(value: Any) -> bytes:
    raw = str(value or "").strip()
    if not raw:
        return b""
    try:
        return base64.b64decode(raw.encode("ascii"), validate=True)
    except Exception as exc:
        raise ValueError(
            "CAI assignment artifact engine input state payloadBase64 is invalid."
        ) from exc


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


def _request_output_frame_template(
    request: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    contract = request.get("outputContract")
    if not isinstance(contract, Mapping):
        return None
    template = contract.get("frameMetadataTemplate")
    return template if isinstance(template, Mapping) else None


def _request_requires_final_output(request: Mapping[str, Any]) -> bool:
    contract = request.get("outputContract")
    if isinstance(contract, Mapping):
        if contract.get("requiresFinalOutput") is not None:
            return bool(contract.get("requiresFinalOutput"))
        if contract.get("requiresOutputFrameMetadata") is not None:
            return not bool(contract.get("requiresOutputFrameMetadata"))
    return _request_output_frame_template(request) is None


def _request_expected_output_kind(request: Mapping[str, Any]) -> str:
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


def _assignment_state_kind_for_output(
    request: Mapping[str, Any],
    *,
    default: str,
) -> str:
    template = _request_output_frame_template(request)
    frame_kind = str(template.get("frameKind") or "").strip().lower() if isinstance(
        template,
        Mapping,
    ) else ""
    if frame_kind == "decode":
        return "assignment_decode"
    if frame_kind == "activation":
        return "assignment_activation"
    return default


def _request_bound(request: Mapping[str, Any], field_name: str) -> int | None:
    shard_spec = request.get("shardSpec")
    if isinstance(shard_spec, Mapping):
        value = _optional_int(shard_spec.get(field_name))
        if value is not None:
            return value
    frame = request.get("frame")
    if isinstance(frame, Mapping):
        value = _optional_int(frame.get(field_name))
        if value is not None:
            return value
    return None


def _request_token_count(request: Mapping[str, Any], payload: bytes) -> int:
    token_start = _request_bound(request, "tokenStart")
    token_end = _request_bound(request, "tokenEnd")
    if (
        token_start is not None
        and token_end is not None
        and token_end >= token_start
    ):
        return max(1, token_end - token_start)
    return 1 if payload else 0


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _positive_int(value: Any) -> int | None:
    clean = _optional_int(value)
    if clean is None or clean < 0:
        return None
    return clean


if __name__ == "__main__":
    raise SystemExit(main())

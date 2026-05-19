# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from .decentralized_compute import (
    build_cai_owned_llm_shard_frame_metadata_from_runtime,
    build_cai_owned_transport_batch_hash_chain,
    build_cai_owned_transport_output_batch_envelope,
    cai_owned_transport_peer_signing_kwargs,
    claim_next_cai_owned_transport_batch,
    complete_cai_owned_transport_work_item,
    fail_cai_owned_transport_work_item,
    heartbeat_cai_owned_transport_batch,
    list_cai_owned_transport_batch_inbox,
    read_cai_owned_transport_batch_payload,
    sign_cai_owned_transport_batch_envelope,
    sign_cai_owned_transport_shard_receipt,
    submit_cai_owned_transport_batch_envelope_to_any,
    validate_cai_owned_llm_handoff_metadata,
    validate_cai_owned_transport_frame_metadata,
)
from .model import NetworkModelPolicy, WalletPolicy
from .wallet import data_root


CAI_OWNED_SHARD_RUNTIME_VERSION = "cai-owned-runtime/0.1"
DETERMINISTIC_BYTES_ADAPTER_ID = "deterministic-bytes"
DETERMINISTIC_BYTES_ADAPTER_VERSION = "deterministic-bytes/0.1"
TASK_LEVEL_HTTP_INFERENCE_ADAPTER_ID = "cai-task-inference-http"
TASK_LEVEL_HTTP_INFERENCE_ADAPTER_VERSION = "cai-task-inference-http/0.1"
LLAMA_CPP_EXTERNAL_SHARD_ADAPTER_ID = "llama.cpp-external-shard"
LLAMA_CPP_EXTERNAL_SHARD_ADAPTER_VERSION = "llama.cpp-external-shard/0.1"
LLAMA_CPP_EXTERNAL_SHARD_HANDOFF_ABI = "cai-llama-cpp-external-shard-v1"
LLAMA_CPP_EXTERNAL_SHARD_LOCAL_FILE_IO_ABI = "cai-llama-cpp-local-file-io-v1"
LLAMA_CPP_EXTERNAL_SHARD_SPEC_SCHEMA_VERSION = 1
LLAMA_CPP_EXTERNAL_SHARD_SPEC_ABI = "cai-llama-cpp-shard-spec-v1"
LLAMA_CPP_EXTERNAL_SHARD_PATCH_BOUNDARY_SCHEMA_VERSION = 1
LLAMA_CPP_EXTERNAL_SHARD_PATCH_BOUNDARY_ABI = (
    "cai-llama-cpp-shard-patch-boundary-v1"
)
LLAMA_CPP_EXTERNAL_SHARD_ACTIVATION_BOUNDARY = "layer-range-activation-v1"
LLAMA_CPP_EXTERNAL_SHARD_DECODE_STATE_BOUNDARY = "token-step-kv-cache-v1"
LLAMA_CPP_EXTERNAL_SHARD_SUPPORTED_TENSOR_ENCODINGS = (
    "ggml-tensor-v1",
    "raw-le",
    "safetensors-fragment-v1",
)
LLAMA_CPP_EXTERNAL_SHARD_REQUIRED_CAPABILITIES = (
    "layer_range_execution",
    "activation_handoff",
    "decode_state_handoff",
    "output_frame_metadata",
)
LLAMA_CPP_EXTERNAL_SHARD_PRODUCTION_CAPABILITIES = (
    "gguf_layer_execution",
    "real_activation_state",
    "real_decode_state",
)
LLAMA_CPP_EXTERNAL_SHARD_GENERATION_PROBE_ACTION = "probe_generation"
LLAMA_CPP_EXTERNAL_SHARD_GENERATION_PROBE_ABI = (
    "cai-llama-cpp-generation-probe-v1"
)
LLAMA_CPP_EXTERNAL_SHARD_PRODUCTION_STATE_CONTRACT_SCHEMA_VERSION = 1
LLAMA_CPP_EXTERNAL_SHARD_PRODUCTION_STATE_CONTRACT_ABI = (
    "cai-llama-cpp-production-state-contract-v1"
)
LLAMA_CPP_EXTERNAL_SHARD_PRODUCTION_STATE_CONTRACT_FIELDS = (
    "activationStateFormat",
    "decodeStateFormat",
    "modelExecutionBackend",
    "tensorEncoding",
)
LLAMA_CPP_EXTERNAL_SHARD_PRODUCTION_EXECUTION_MODE = "layer_range"
CAI_LLM_SHARD_ADAPTER_ENV = "CAI_LLM_SHARD_ADAPTER"
CAI_LLM_SHARD_ADAPTER_COMMAND_ENV = "CAI_LLM_SHARD_ADAPTER_COMMAND"
CAI_LLM_SHARD_ADAPTER_URL_ENV = "CAI_LLM_SHARD_ADAPTER_URL"
CAI_ALLOW_REMOTE_LLM_SHARD_ADAPTER_URL_ENV = (
    "CAI_ALLOW_REMOTE_LLM_SHARD_ADAPTER_URL"
)
CAI_LLM_SHARD_ADAPTER_TIMEOUT_ENV = "CAI_LLM_SHARD_ADAPTER_TIMEOUT_SEC"
CAI_LLM_SHARD_FILE_IO_THRESHOLD_BYTES_ENV = "CAI_LLM_SHARD_FILE_IO_THRESHOLD_BYTES"
CAI_LLM_SHARD_IO_ROOT_ENV = "CAI_LLM_SHARD_IO_ROOT"
CAI_LLM_SHARD_ARTIFACT_HINT_JSON_ENV = "CAI_LLM_SHARD_ARTIFACT_HINT_JSON"
CAI_REQUIRE_PRODUCTION_LLM_HANDOFF_ENV = "CAI_REQUIRE_PRODUCTION_LLM_HANDOFF"
CAI_REQUIRE_LLM_PATCH_BOUNDARY_ENV = "CAI_REQUIRE_LLM_PATCH_BOUNDARY"
CAI_DETERMINISTIC_SHARD_PREFIX_ENV = "CAI_DETERMINISTIC_SHARD_PREFIX"
CAI_TASK_INFERENCE_ADAPTER_URL_ENV = "CAI_TASK_INFERENCE_ADAPTER_URL"
CAI_TASK_INFERENCE_ADAPTER_MODEL_ENV = "CAI_TASK_INFERENCE_ADAPTER_MODEL"
CAI_TASK_INFERENCE_ADAPTER_TIMEOUT_ENV = "CAI_TASK_INFERENCE_ADAPTER_TIMEOUT_SEC"
CAI_TASK_INFERENCE_ADAPTER_MAX_TOKENS_ENV = "CAI_TASK_INFERENCE_ADAPTER_MAX_TOKENS"
CAI_TASK_INFERENCE_ADAPTER_TEMPERATURE_ENV = (
    "CAI_TASK_INFERENCE_ADAPTER_TEMPERATURE"
)
CAI_ALLOW_REMOTE_TASK_INFERENCE_ADAPTER_URL_ENV = (
    "CAI_ALLOW_REMOTE_TASK_INFERENCE_ADAPTER_URL"
)
CAI_LLM_SHARD_NATIVE_COMMAND_ENV = "CAI_LLM_SHARD_NATIVE_COMMAND"
CAI_LLM_SHARD_NATIVE_TIMEOUT_ENV = "CAI_LLM_SHARD_NATIVE_TIMEOUT_SEC"
CAI_LLM_SHARD_NATIVE_PERSISTENT_ENV = "CAI_LLM_SHARD_NATIVE_PERSISTENT"
CAI_LLM_SHARD_SLOT_SERVER_URL_ENV = "CAI_LLM_SHARD_SLOT_SERVER_URL"
CAI_LLM_SHARD_SLOT_STATE_DIR_ENV = "CAI_LLM_SHARD_SLOT_STATE_DIR"
CAI_LLM_SHARD_SLOT_ID_ENV = "CAI_LLM_SHARD_SLOT_ID"
CAI_LLM_SHARD_SLOT_TIMEOUT_ENV = "CAI_LLM_SHARD_SLOT_TIMEOUT_SEC"
CAI_LLM_SHARD_SLOT_DECODE_TOKENS_ENV = "CAI_LLM_SHARD_SLOT_DECODE_TOKENS"
CAI_LLM_SHARD_SELF_TEST_CACHE_SCHEMA_VERSION = 2
CAI_LLM_SHARD_SELF_TEST_CACHE_TTL_SECONDS = 24 * 60 * 60
CAI_OWNED_TRANSPORT_LIVE_PROOF_CACHE_SCHEMA_VERSION = 1
CAI_OWNED_TRANSPORT_LIVE_PROOF_CACHE_TTL_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class CaiOwnedShardRuntimeConfig:
    node_id: str
    runtime_id: str
    coordinator_cai_url: str | None = None
    output_peer_cai_urls_by_node: Mapping[str, Sequence[str]] = field(
        default_factory=dict
    )
    output_forward_timeout_sec: float = 5.0
    max_concurrent_batches: int = 1
    max_payload_size_bytes: int = 16 * 1024 * 1024
    lease_seconds: float = 60.0
    max_attempts: int = 3
    local_runtime_auth_token: str | None = None
    require_local_runtime_auth: bool | str | None = None
    signing_material: Mapping[str, Any] | None = None
    require_production_llm_handoff: bool = False
    policy: WalletPolicy | None = None


@dataclass(frozen=True)
class CaiOwnedShardAdapterResult:
    output_payload: bytes = b""
    metrics: dict[str, Any] = field(default_factory=dict)
    output_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaiOwnedShardFrame:
    session_id: str
    batch_id: str
    phase: str
    source_node_id: str
    sink_node_id: str
    sequence: int
    model_id: str | None
    frame_kind: str
    layer_start: int | None
    layer_end: int | None
    token_start: int | None
    token_end: int | None
    payload_sha256_hex: str
    payload: bytes
    metadata: dict[str, Any] = field(default_factory=dict)


class CaiOwnedLlmShardBackendHealthError(ValueError):
    def __init__(self, backend_health: Mapping[str, Any] | None) -> None:
        self.backend_health = (
            dict(backend_health) if isinstance(backend_health, Mapping) else None
        )
        super().__init__(_llm_shard_adapter_health_error(backend_health))


class CaiOwnedShardAdapter(Protocol):
    def process(
        self,
        work_item: Mapping[str, Any],
        payload: bytes,
    ) -> CaiOwnedShardAdapterResult:
        ...


class CaiOwnedLlmShardAdapter(Protocol):
    def load_shard(self, frame: CaiOwnedShardFrame) -> Mapping[str, Any] | None:
        ...

    def process_prefill(self, frame: CaiOwnedShardFrame) -> CaiOwnedShardAdapterResult:
        ...

    def process_decode(self, frame: CaiOwnedShardFrame) -> CaiOwnedShardAdapterResult:
        ...

    def finalize(
        self,
        frame: CaiOwnedShardFrame,
        result: CaiOwnedShardAdapterResult,
    ) -> Mapping[str, Any] | None:
        ...


@dataclass(frozen=True)
class DeterministicBytesShardAdapter:
    prefix: bytes = b"cai-shard-output:"

    def process(
        self,
        work_item: Mapping[str, Any],
        payload: bytes,
    ) -> CaiOwnedShardAdapterResult:
        return self._process_frame(_work_item_frame(work_item, payload))

    def load_shard(self, frame: CaiOwnedShardFrame) -> Mapping[str, Any]:
        return {
            "adapterShardLoaded": True,
            "adapterLayerStart": frame.layer_start,
            "adapterLayerEnd": frame.layer_end,
        }

    def process_prefill(self, frame: CaiOwnedShardFrame) -> CaiOwnedShardAdapterResult:
        return self._process_frame(frame, adapter_phase="prefill")

    def process_decode(self, frame: CaiOwnedShardFrame) -> CaiOwnedShardAdapterResult:
        return self._process_frame(frame, adapter_phase="decode")

    def finalize(
        self,
        frame: CaiOwnedShardFrame,
        result: CaiOwnedShardAdapterResult,
    ) -> Mapping[str, Any]:
        return {
            "adapterFinalized": True,
            "adapterFinalizedBatchId": frame.batch_id,
        }

    def _process_frame(
        self,
        frame: CaiOwnedShardFrame,
        *,
        adapter_phase: str | None = None,
    ) -> CaiOwnedShardAdapterResult:
        return CaiOwnedShardAdapterResult(
            output_payload=self.prefix + frame.payload,
            metrics={
                "adapter": DETERMINISTIC_BYTES_ADAPTER_ID,
                "adapterId": DETERMINISTIC_BYTES_ADAPTER_ID,
                "adapterVersion": DETERMINISTIC_BYTES_ADAPTER_VERSION,
                "adapterPhase": adapter_phase or frame.phase,
                "frameKind": frame.frame_kind,
                "inputBytes": len(frame.payload),
                "outputBytes": len(self.prefix) + len(frame.payload),
                "batchId": frame.batch_id,
                "modelId": frame.model_id,
                "layerStart": frame.layer_start,
                "layerEnd": frame.layer_end,
                "sequence": frame.sequence,
            },
        )


@dataclass(frozen=True)
class TaskLevelHttpInferenceAdapter:
    endpoint_url: str
    model_id: str | None = None
    timeout_sec: float = 120.0
    allow_remote_endpoint_url: bool = False
    max_tokens: int | None = 64
    temperature: float | None = 0.0
    adapter_id: str = TASK_LEVEL_HTTP_INFERENCE_ADAPTER_ID
    adapter_version: str = TASK_LEVEL_HTTP_INFERENCE_ADAPTER_VERSION

    def __post_init__(self) -> None:
        _validate_llm_shard_adapter_endpoint_url(
            str(self.endpoint_url or "").strip(),
            allow_remote=bool(self.allow_remote_endpoint_url),
        )

    def probe_health(self) -> dict[str, Any] | None:
        endpoint_url = str(self.endpoint_url or "").strip()
        if not endpoint_url:
            return None
        health_url = _llm_shard_adapter_health_url(endpoint_url)
        http_request = Request(
            health_url,
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with urlopen(
                http_request,
                timeout=max(0.1, float(self.timeout_sec or 120.0)),
            ) as response:
                response_text = response.read().decode("utf-8")
            return _parse_llm_shard_adapter_health_response(
                response_text,
                health_url=health_url,
                http_status=_http_response_status(response, 200),
                endpoint_available=True,
            )
        except HTTPError as exc:
            if int(exc.code or 0) in {404, 405}:
                return {
                    "status": "unknown",
                    "ready": None,
                    "healthEndpointAvailable": False,
                    "healthEndpointUrl": health_url,
                    "httpStatus": int(exc.code or 0),
                    "error": "Task inference adapter health endpoint is not available.",
                }
            response_text = exc.read().decode("utf-8", errors="replace")
            health = _parse_llm_shard_adapter_health_response(
                response_text,
                health_url=health_url,
                http_status=int(exc.code or 0),
                endpoint_available=True,
            )
            health.setdefault("status", "degraded")
            health.setdefault("ready", False)
            return health
        except Exception as exc:
            return {
                "status": "unknown",
                "ready": None,
                "healthEndpointAvailable": False,
                "healthEndpointUrl": health_url,
                "errorClass": exc.__class__.__name__,
                "error": str(exc)[:500],
            }

    def process(
        self,
        work_item: Mapping[str, Any],
        payload: bytes,
    ) -> CaiOwnedShardAdapterResult:
        batch = work_item.get("batch") if isinstance(work_item, Mapping) else None
        metadata = batch.get("metadata") if isinstance(batch, Mapping) else None
        if not _task_level_inference_stage(metadata):
            return CaiOwnedShardAdapterResult(
                output_payload=bytes(payload or b""),
                metrics={
                    "adapterId": self.adapter_id,
                    "adapterVersion": self.adapter_version,
                    "adapterMode": "task_level_http_inference",
                    "taskLevelFallback": True,
                    "taskLevelPassThrough": True,
                    "modelParallel": False,
                    "inferenceExecutor": False,
                    "inputTokenCount": 0,
                    "outputTokenCount": 0,
                    "promptTokenCount": 0,
                    "completionTokenCount": 0,
                },
            )

        request_payload, prompt_text = _task_inference_request_payload(
            work_item,
            payload,
            model_id=self.model_id,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        started = time.perf_counter()
        response_payload = self._call_inference_endpoint(request_payload)
        latency_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        output_text = _task_inference_response_text(response_payload)
        output_payload = output_text.encode("utf-8")
        metrics = _task_inference_response_metrics(
            response_payload,
            prompt_text=prompt_text,
            output_text=output_text,
        )
        metrics.update(
            {
                "adapterId": self.adapter_id,
                "adapterVersion": self.adapter_version,
                "adapterMode": "task_level_http_inference",
                "taskLevelFallback": True,
                "taskLevelPassThrough": False,
                "modelParallel": False,
                "inferenceExecutor": True,
                "inferenceEndpoint": _safe_endpoint_label(self.endpoint_url),
                "inferenceLatencyMs": latency_ms,
                "modelId": request_payload.get("model"),
            }
        )
        return CaiOwnedShardAdapterResult(
            output_payload=output_payload,
            metrics=metrics,
        )

    def _call_inference_endpoint(self, request_payload: Mapping[str, Any]) -> dict[str, Any]:
        request_body = json.dumps(request_payload, sort_keys=True).encode("utf-8")
        http_request = Request(
            str(self.endpoint_url or "").strip(),
            data=request_body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-CAI-Task-Inference-Adapter": self.adapter_version,
            },
            method="POST",
        )
        try:
            with urlopen(
                http_request,
                timeout=max(0.1, float(self.timeout_sec or 120.0)),
            ) as response:
                response_text = response.read().decode("utf-8")
        except Exception as exc:
            raise ValueError(
                "Task-level HTTP inference adapter endpoint failed: "
                f"{str(exc)[:500]}"
            ) from exc
        try:
            parsed = json.loads(response_text or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Task-level HTTP inference adapter returned invalid JSON."
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError(
                "Task-level HTTP inference adapter returned unsupported response."
            )
        return parsed


@dataclass(frozen=True)
class ExternalLlamaCppShardAdapter:
    command: Sequence[str] = field(default_factory=tuple)
    endpoint_url: str | None = None
    allow_remote_endpoint_url: bool = False
    timeout_sec: float = 120.0
    env: Mapping[str, str] = field(default_factory=dict)
    adapter_id: str = LLAMA_CPP_EXTERNAL_SHARD_ADAPTER_ID
    adapter_version: str = LLAMA_CPP_EXTERNAL_SHARD_ADAPTER_VERSION
    backend: str = "llama.cpp-patched"
    backend_version: str | None = None
    require_handoff_contract: bool = True
    require_patch_boundary: bool = True
    required_capabilities: Sequence[str] = LLAMA_CPP_EXTERNAL_SHARD_REQUIRED_CAPABILITIES
    file_io_root: str | None = None
    file_io_threshold_bytes: int | None = None
    shard_artifact_hint: Mapping[str, Any] | None = None
    defer_finalize_when_persistent: bool = True

    def load_shard(self, frame: CaiOwnedShardFrame) -> Mapping[str, Any]:
        response = self._call_backend("load_shard", frame, b"")
        self._validate_required_capabilities(response)
        self._validate_patch_boundary(response)
        return self._response_metrics(response)

    def process_prefill(self, frame: CaiOwnedShardFrame) -> CaiOwnedShardAdapterResult:
        return self._process_frame("process_prefill", frame)

    def process_decode(self, frame: CaiOwnedShardFrame) -> CaiOwnedShardAdapterResult:
        return self._process_frame("process_decode", frame)

    def finalize(
        self,
        frame: CaiOwnedShardFrame,
        result: CaiOwnedShardAdapterResult,
    ) -> Mapping[str, Any]:
        response = self._call_backend("finalize", frame, result.output_payload)
        return self._response_metrics(response)

    def probe_generation(
        self,
        *,
        model_id: str,
        prompt: str,
        max_tokens: int = 8,
    ) -> Mapping[str, Any]:
        request_payload = {
            "schemaVersion": 1,
            "abi": LLAMA_CPP_EXTERNAL_SHARD_HANDOFF_ABI,
            "action": LLAMA_CPP_EXTERNAL_SHARD_GENERATION_PROBE_ACTION,
            "adapterId": self.adapter_id,
            "adapterVersion": self.adapter_version,
            "backend": self.backend,
            "backendVersion": self.backend_version,
            "generationProbe": {
                "schemaVersion": 1,
                "abi": LLAMA_CPP_EXTERNAL_SHARD_GENERATION_PROBE_ABI,
                "modelId": str(model_id or "").strip(),
                "prompt": str(prompt or ""),
                "maxTokens": max(1, int(max_tokens or 8)),
                "temperature": 0.0,
                "requiresRealModelExecution": True,
            },
            "productionRequirements": _external_adapter_production_requirements(
                self.required_capabilities,
            ),
        }
        return self._call_backend_request(request_payload)

    def probe_health(self) -> dict[str, Any] | None:
        endpoint_url = str(self.endpoint_url or "").strip()
        if not endpoint_url:
            return None
        _validate_llm_shard_adapter_endpoint_url(
            endpoint_url,
            allow_remote=bool(self.allow_remote_endpoint_url),
        )
        health_url = _llm_shard_adapter_health_url(endpoint_url)
        http_request = Request(
            health_url,
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with urlopen(
                http_request,
                timeout=max(0.1, float(self.timeout_sec or 120.0)),
            ) as response:
                response_text = response.read().decode("utf-8")
                return _parse_llm_shard_adapter_health_response(
                    response_text,
                    health_url=health_url,
                    http_status=_http_response_status(response, 200),
                    endpoint_available=True,
                )
        except HTTPError as exc:
            response_text = exc.read().decode("utf-8", errors="replace")
            if int(exc.code or 0) in {404, 405}:
                return {
                    "status": "unknown",
                    "ready": None,
                    "healthEndpointAvailable": False,
                    "healthEndpointUrl": health_url,
                    "httpStatus": int(exc.code or 0),
                    "error": "LLM shard adapter health endpoint is not available.",
                }
            health = _parse_llm_shard_adapter_health_response(
                response_text,
                health_url=health_url,
                http_status=int(exc.code or 0),
                endpoint_available=True,
            )
            health.setdefault("status", "degraded")
            health.setdefault("ready", False)
            return health
        except Exception as exc:
            return {
                "status": "unknown",
                "ready": None,
                "healthEndpointAvailable": False,
                "healthEndpointUrl": health_url,
                "errorClass": exc.__class__.__name__,
                "error": str(exc)[:500],
            }

    def _process_frame(
        self,
        action: str,
        frame: CaiOwnedShardFrame,
    ) -> CaiOwnedShardAdapterResult:
        response = self._call_backend(action, frame, frame.payload)
        output_payload = _external_adapter_output_payload(response)
        metrics = self._response_metrics(response)
        declared_hash = str(response.get("outputPayloadSha256Hex") or "").strip().lower()
        if declared_hash and declared_hash != hashlib.sha256(output_payload).hexdigest():
            raise ValueError("External llama.cpp shard adapter output hash mismatch.")
        output_metadata = _external_adapter_output_metadata(
            response,
            output_payload,
            expected_model_id=frame.model_id,
            require_handoff_contract=self.require_handoff_contract,
            expected_frame_template=_next_frame_template(frame.metadata),
        )
        return CaiOwnedShardAdapterResult(
            output_payload=output_payload,
            metrics=metrics,
            output_metadata=output_metadata,
        )

    def _call_backend(
        self,
        action: str,
        frame: CaiOwnedShardFrame,
        payload: bytes,
    ) -> dict[str, Any]:
        if self.require_handoff_contract:
            valid, error = validate_cai_owned_llm_handoff_metadata(
                frame.metadata.get("llmHandoff")
                if isinstance(frame.metadata, Mapping)
                else None,
                expected_model_id=frame.model_id,
                expected_frame_metadata=frame.metadata,
            )
            if not valid:
                raise ValueError(error or "CAI-owned LLM handoff metadata is invalid.")
        request_payload = {
            "schemaVersion": 1,
            "abi": LLAMA_CPP_EXTERNAL_SHARD_HANDOFF_ABI,
            "action": action,
            "adapterId": self.adapter_id,
            "adapterVersion": self.adapter_version,
            "backend": self.backend,
            "backendVersion": self.backend_version,
            "frame": _frame_request_payload(frame),
            "shardSpec": build_llama_cpp_external_shard_spec(
                frame,
                artifact_hint=self.shard_artifact_hint,
            ),
            "payloadSha256Hex": hashlib.sha256(bytes(payload or b"")).hexdigest(),
            "outputContract": _external_adapter_output_contract(frame),
            "productionRequirements": _external_adapter_production_requirements(
                self.required_capabilities,
            ),
        }
        endpoint_url = str(self.endpoint_url or "").strip()
        local_file_context = _prepare_external_adapter_local_file_context(
            payload=payload,
            endpoint_url=endpoint_url or None,
            allow_remote_endpoint_url=bool(self.allow_remote_endpoint_url),
            file_io_root=self.file_io_root,
            file_io_threshold_bytes=self.file_io_threshold_bytes,
        )
        if local_file_context is not None:
            request_payload["payloadFile"] = dict(local_file_context["payloadFile"])
            request_payload["localFileContract"] = dict(
                local_file_context["localFileContract"]
            )
        else:
            request_payload["payloadBase64"] = base64.b64encode(
                bytes(payload or b"")
            ).decode("ascii")
        try:
            if endpoint_url:
                _validate_llm_shard_adapter_endpoint_url(
                    endpoint_url,
                    allow_remote=bool(self.allow_remote_endpoint_url),
                )
                response = self._call_http_backend(endpoint_url, request_payload)
            else:
                response = self._call_backend_request(request_payload)
        except Exception:
            _cleanup_external_adapter_local_file_context(local_file_context)
            raise
        if local_file_context is not None:
            if action in {"process_prefill", "process_decode"}:
                response["_localFileContract"] = dict(
                    local_file_context["localFileContract"]
                )
            else:
                _cleanup_external_adapter_local_file_context(local_file_context)
        return response

    def _call_backend_request(
        self,
        request_payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        endpoint_url = str(self.endpoint_url or "").strip()
        if endpoint_url:
            _validate_llm_shard_adapter_endpoint_url(
                endpoint_url,
                allow_remote=bool(self.allow_remote_endpoint_url),
            )
            return self._call_http_backend(endpoint_url, request_payload)
        command = [str(item) for item in self.command if str(item or "").strip()]
        if not command:
            raise ValueError(
                "External llama.cpp shard adapter command or endpoint URL is required."
            )
        return self._call_command_backend(command, request_payload)

    def _call_command_backend(
        self,
        command: Sequence[str],
        request_payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        completed = subprocess.run(
            list(command),
            input=json.dumps(request_payload, sort_keys=True),
            capture_output=True,
            text=True,
            timeout=max(0.1, float(self.timeout_sec or 120.0)),
            env={**os.environ, **{str(k): str(v) for k, v in self.env.items()}},
            check=False,
        )
        if completed.returncode != 0:
            stderr = str(completed.stderr or "").strip()
            raise ValueError(
                "External llama.cpp shard adapter failed"
                + (f": {stderr[:500]}" if stderr else ".")
            )
        return _parse_external_llama_cpp_shard_adapter_response(completed.stdout)

    def _call_http_backend(
        self,
        endpoint_url: str,
        request_payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        request_body = json.dumps(request_payload, sort_keys=True).encode("utf-8")
        http_request = Request(
            endpoint_url,
            data=request_body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-CAI-LLM-Shard-ABI": LLAMA_CPP_EXTERNAL_SHARD_HANDOFF_ABI,
            },
            method="POST",
        )
        try:
            with urlopen(
                http_request,
                timeout=max(0.1, float(self.timeout_sec or 120.0)),
            ) as response:
                response_text = response.read().decode("utf-8")
        except Exception as exc:
            raise ValueError(
                "External llama.cpp shard adapter HTTP endpoint failed: "
                f"{str(exc)[:500]}"
            ) from exc
        return _parse_external_llama_cpp_shard_adapter_response(response_text)

    def _validate_required_capabilities(self, response: Mapping[str, Any]) -> None:
        required = [
            str(item or "").strip()
            for item in self.required_capabilities
            if str(item or "").strip()
        ]
        if not required:
            return
        available = _external_adapter_capability_set(response)
        missing = [item for item in required if item not in available]
        if missing:
            raise ValueError(
                "External llama.cpp shard adapter missing required capabilities: "
                + ", ".join(missing)
            )

    def _validate_patch_boundary(self, response: Mapping[str, Any]) -> None:
        if not self.require_patch_boundary:
            return
        boundary = _external_adapter_patch_boundary(response)
        valid, error = validate_llama_cpp_external_shard_patch_boundary(
            boundary,
            expected_backend=self.backend,
            required_capabilities=self.required_capabilities,
        )
        if not valid:
            raise ValueError(
                error or "External llama.cpp shard adapter patch boundary is invalid."
            )

    def _response_metrics(self, response: Mapping[str, Any]) -> dict[str, Any]:
        metrics = (
            dict(response.get("metrics"))
            if isinstance(response.get("metrics"), Mapping)
            else {}
        )
        capabilities = sorted(_external_adapter_capability_set(response))
        if capabilities:
            metrics.setdefault("backendCapabilities", capabilities)
        patch_boundary = _external_adapter_patch_boundary(response)
        if isinstance(patch_boundary, Mapping):
            valid, _error = validate_llama_cpp_external_shard_patch_boundary(
                patch_boundary,
                expected_backend=self.backend,
                required_capabilities=self.required_capabilities,
            )
            patch_boundary_capabilities = _clean_string_sequence(
                patch_boundary.get("capabilities")
            )
            metrics.setdefault("patchBoundaryVerified", bool(valid))
            metrics.setdefault("patchBoundaryAbi", patch_boundary.get("abi"))
            metrics.setdefault("patchBoundaryPatchId", patch_boundary.get("patchId"))
            metrics.setdefault(
                "patchBoundaryCapabilities",
                patch_boundary_capabilities,
            )
            metrics.setdefault(
                "patchBoundaryHash",
                _stable_json_sha256_hex(patch_boundary),
            )
            extra_metadata = patch_boundary.get("extraMetadata")
            if isinstance(extra_metadata, Mapping):
                metrics.setdefault("patchBoundaryExtraMetadata", dict(extra_metadata))
        metrics.setdefault("adapterId", self.adapter_id)
        metrics.setdefault("adapterVersion", self.adapter_version)
        metrics.setdefault("backend", self.backend)
        if self.backend_version:
            metrics.setdefault("backendVersion", self.backend_version)
        metrics.setdefault("productionLlmHandoff", bool(self.require_handoff_contract))
        return metrics


def _parse_external_llama_cpp_shard_adapter_response(
    response_text: str,
) -> dict[str, Any]:
    try:
        parsed = json.loads(response_text or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(
            "External llama.cpp shard adapter returned invalid JSON."
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(
            "External llama.cpp shard adapter returned unsupported response."
        )
    status = str(parsed.get("status") or "ok").strip().lower()
    if status not in {"ok", "ready"}:
        error = str(parsed.get("error") or parsed.get("message") or status)
        raise ValueError(f"External llama.cpp shard adapter rejected work: {error}")
    return parsed


def _llm_shard_adapter_health_url(endpoint_url: str) -> str:
    parsed = urlsplit(str(endpoint_url or "").strip())
    path = parsed.path or "/cai-shard"
    if path.endswith("/cai-shard"):
        health_path = path[: -len("/cai-shard")] + "/health"
    elif path.endswith("/"):
        health_path = path + "health"
    else:
        health_path = "/health"
    if not health_path.startswith("/"):
        health_path = "/" + health_path
    return urlunsplit((parsed.scheme, parsed.netloc, health_path, "", ""))


def _http_response_status(response: object, default: int) -> int:
    getcode = getattr(response, "getcode", None)
    if callable(getcode):
        try:
            return int(getcode())
        except (TypeError, ValueError):
            pass
    try:
        return int(getattr(response, "status"))
    except (TypeError, ValueError):
        return int(default)


def _parse_llm_shard_adapter_health_response(
    response_text: str,
    *,
    health_url: str,
    http_status: int,
    endpoint_available: bool,
) -> dict[str, Any]:
    try:
        parsed = json.loads(response_text or "{}")
    except json.JSONDecodeError:
        parsed = {}
    health = dict(parsed) if isinstance(parsed, Mapping) else {}
    health.setdefault("status", "ok" if 200 <= int(http_status) < 300 else "degraded")
    health.setdefault("healthEndpointAvailable", bool(endpoint_available))
    health.setdefault("healthEndpointUrl", health_url)
    health.setdefault("httpStatus", int(http_status))
    if "ready" not in health:
        health["ready"] = _llm_shard_adapter_health_ready(health)
    return health


def _validate_llm_shard_adapter_endpoint_url(
    endpoint_url: str,
    *,
    allow_remote: bool = False,
) -> None:
    clean_url = str(endpoint_url or "").strip()
    parsed = urlsplit(clean_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("CAI LLM shard adapter endpoint URL must be HTTP(S).")
    if parsed.username or parsed.password:
        raise ValueError(
            "CAI LLM shard adapter endpoint URL must not contain credentials."
        )
    host = str(parsed.hostname or "").strip()
    if not host:
        raise ValueError("CAI LLM shard adapter endpoint URL host is required.")
    if allow_remote or _is_loopback_host(host):
        return
    raise ValueError(
        "CAI LLM shard adapter endpoint URL must be loopback by default; "
        f"set {CAI_ALLOW_REMOTE_LLM_SHARD_ADAPTER_URL_ENV}=1 only for a guarded "
        "lab bridge."
    )


def _is_loopback_host(host: str) -> bool:
    clean_host = str(host or "").strip().lower()
    if clean_host == "localhost" or clean_host.endswith(".localhost"):
        return True
    try:
        return bool(ipaddress.ip_address(clean_host).is_loopback)
    except ValueError:
        return False


def cai_owned_shard_adapter_from_env(
    env: Mapping[str, str] | None = None,
) -> object:
    source = env or os.environ
    kind = str(source.get(CAI_LLM_SHARD_ADAPTER_ENV) or "deterministic").strip().lower()
    if kind in {"", "deterministic", "deterministic_bytes", "bytes"}:
        prefix = str(source.get(CAI_DETERMINISTIC_SHARD_PREFIX_ENV) or "")
        return DeterministicBytesShardAdapter(
            prefix=prefix.encode("utf-8") if prefix else b"cai-shard-output:",
        )
    if kind in {"smoke", "smoke_runner", "llm_shard_smoke_runner"}:
        return ExternalLlamaCppShardAdapter(
            command=_python_module_command(
                "cai_compute_chain.cai_llama_cpp_shard_smoke_runner"
            ),
            timeout_sec=_env_float(source, CAI_LLM_SHARD_ADAPTER_TIMEOUT_ENV, 120.0),
            env=_smoke_runner_env(source),
            require_handoff_contract=True,
            require_patch_boundary=True,
            file_io_root=_env_optional_text(source, CAI_LLM_SHARD_IO_ROOT_ENV),
            file_io_threshold_bytes=_env_optional_int(
                source,
                CAI_LLM_SHARD_FILE_IO_THRESHOLD_BYTES_ENV,
                None,
            ),
            shard_artifact_hint=_env_optional_json_mapping(
                source,
                CAI_LLM_SHARD_ARTIFACT_HINT_JSON_ENV,
            ),
        )
    if kind in {"native_bridge", "llama_cpp_native_bridge"}:
        endpoint_url = str(source.get(CAI_LLM_SHARD_ADAPTER_URL_ENV) or "").strip()
        allow_remote_endpoint_url = _env_bool(
            source,
            CAI_ALLOW_REMOTE_LLM_SHARD_ADAPTER_URL_ENV,
            False,
        )
        if endpoint_url:
            _validate_llm_shard_adapter_endpoint_url(
                endpoint_url,
                allow_remote=allow_remote_endpoint_url,
            )
            return ExternalLlamaCppShardAdapter(
                endpoint_url=endpoint_url,
                allow_remote_endpoint_url=allow_remote_endpoint_url,
                timeout_sec=_env_float(
                    source,
                    CAI_LLM_SHARD_ADAPTER_TIMEOUT_ENV,
                    120.0,
                ),
                require_handoff_contract=True,
                require_patch_boundary=True,
                file_io_root=_env_optional_text(source, CAI_LLM_SHARD_IO_ROOT_ENV),
                file_io_threshold_bytes=_env_optional_int(
                    source,
                    CAI_LLM_SHARD_FILE_IO_THRESHOLD_BYTES_ENV,
                    None,
                ),
                shard_artifact_hint=_env_optional_json_mapping(
                    source,
                    CAI_LLM_SHARD_ARTIFACT_HINT_JSON_ENV,
                ),
            )
        native_env = _runtime_src_pythonpath_env(source)
        for key in (
            CAI_LLM_SHARD_NATIVE_COMMAND_ENV,
            CAI_LLM_SHARD_NATIVE_TIMEOUT_ENV,
            CAI_LLM_SHARD_NATIVE_PERSISTENT_ENV,
        ):
            if source.get(key) is not None:
                native_env[key] = str(source.get(key) or "")
        return ExternalLlamaCppShardAdapter(
            command=_python_module_command(
                "cai_compute_chain.cai_llama_cpp_shard_native_bridge",
                "--oneshot",
            ),
            timeout_sec=_env_float(source, CAI_LLM_SHARD_ADAPTER_TIMEOUT_ENV, 120.0),
            env=native_env,
            require_handoff_contract=True,
            require_patch_boundary=True,
            file_io_root=_env_optional_text(source, CAI_LLM_SHARD_IO_ROOT_ENV),
            file_io_threshold_bytes=_env_optional_int(
                source,
                CAI_LLM_SHARD_FILE_IO_THRESHOLD_BYTES_ENV,
                None,
            ),
            shard_artifact_hint=_env_optional_json_mapping(
                source,
                CAI_LLM_SHARD_ARTIFACT_HINT_JSON_ENV,
            ),
        )
    if kind in {"slot_state", "llama_cpp_slot_state", "llama_cpp_slot"}:
        slot_env = _slot_state_engine_env(source)
        return ExternalLlamaCppShardAdapter(
            command=_python_module_command(
                "cai_compute_chain.cai_llama_cpp_slot_state_engine"
            ),
            timeout_sec=_env_float(source, CAI_LLM_SHARD_ADAPTER_TIMEOUT_ENV, 120.0),
            env=slot_env,
            require_handoff_contract=True,
            require_patch_boundary=True,
            file_io_root=_env_optional_text(source, CAI_LLM_SHARD_IO_ROOT_ENV),
            file_io_threshold_bytes=_env_optional_int(
                source,
                CAI_LLM_SHARD_FILE_IO_THRESHOLD_BYTES_ENV,
                None,
            ),
            shard_artifact_hint=_env_optional_json_mapping(
                source,
                CAI_LLM_SHARD_ARTIFACT_HINT_JSON_ENV,
            ),
        )
    if kind in {
        "task_http",
        "task_inference_http",
        "job_http",
        "job_level_http",
        "openai_compatible",
    }:
        endpoint_url = str(
            source.get(CAI_TASK_INFERENCE_ADAPTER_URL_ENV)
            or source.get(CAI_LLM_SHARD_ADAPTER_URL_ENV)
            or ""
        ).strip()
        if not endpoint_url:
            raise ValueError(
                "CAI task inference adapter URL is required."
            )
        allow_remote_endpoint_url = _env_bool(
            source,
            CAI_ALLOW_REMOTE_TASK_INFERENCE_ADAPTER_URL_ENV,
            False,
        )
        return TaskLevelHttpInferenceAdapter(
            endpoint_url=endpoint_url,
            model_id=str(source.get(CAI_TASK_INFERENCE_ADAPTER_MODEL_ENV) or "").strip()
            or None,
            timeout_sec=_env_float(
                source,
                CAI_TASK_INFERENCE_ADAPTER_TIMEOUT_ENV,
                _env_float(source, CAI_LLM_SHARD_ADAPTER_TIMEOUT_ENV, 120.0),
            ),
            allow_remote_endpoint_url=allow_remote_endpoint_url,
            max_tokens=_env_optional_int(
                source,
                CAI_TASK_INFERENCE_ADAPTER_MAX_TOKENS_ENV,
                64,
            ),
            temperature=_env_optional_float(
                source,
                CAI_TASK_INFERENCE_ADAPTER_TEMPERATURE_ENV,
                0.0,
            ),
        )
    if kind in {"external", "external_llama_cpp", "llama_cpp_external", "production"}:
        command = _env_command(source, CAI_LLM_SHARD_ADAPTER_COMMAND_ENV)
        endpoint_url = str(source.get(CAI_LLM_SHARD_ADAPTER_URL_ENV) or "").strip()
        allow_remote_endpoint_url = _env_bool(
            source,
            CAI_ALLOW_REMOTE_LLM_SHARD_ADAPTER_URL_ENV,
            False,
        )
        if not command and not endpoint_url:
            raise ValueError(
                "CAI LLM shard adapter command or endpoint URL is required."
            )
        if endpoint_url:
            _validate_llm_shard_adapter_endpoint_url(
                endpoint_url,
                allow_remote=allow_remote_endpoint_url,
            )
        return ExternalLlamaCppShardAdapter(
            command=command,
            endpoint_url=endpoint_url or None,
            allow_remote_endpoint_url=allow_remote_endpoint_url,
            timeout_sec=_env_float(source, CAI_LLM_SHARD_ADAPTER_TIMEOUT_ENV, 120.0),
            require_handoff_contract=_env_bool(
                source,
                CAI_REQUIRE_PRODUCTION_LLM_HANDOFF_ENV,
                True,
            ),
            require_patch_boundary=_env_bool(
                source,
                CAI_REQUIRE_LLM_PATCH_BOUNDARY_ENV,
                True,
            ),
            file_io_root=_env_optional_text(source, CAI_LLM_SHARD_IO_ROOT_ENV),
            file_io_threshold_bytes=_env_optional_int(
                source,
                CAI_LLM_SHARD_FILE_IO_THRESHOLD_BYTES_ENV,
                None,
            ),
            shard_artifact_hint=_env_optional_json_mapping(
                source,
                CAI_LLM_SHARD_ARTIFACT_HINT_JSON_ENV,
            ),
        )
    raise ValueError(f"CAI LLM shard adapter kind is unsupported: {kind}")


def run_cai_owned_llm_shard_adapter_self_test(
    adapter: object | None = None,
    *,
    model_id: str | None = None,
    runtime_metadata: Mapping[str, Any] | None = None,
    payload: bytes | None = None,
    require_production_llm_handoff: bool = True,
    require_generation_probe: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    recorded_at = datetime.now(tz=UTC).isoformat()
    resolved_model_id = (
        str(model_id or "").strip()
        or NetworkModelPolicy().network_default_model_id
    )
    probe_payload = bytes(payload or b"cai-llm-shard-self-test")
    adapter_instance = (
        adapter if adapter is not None else cai_owned_shard_adapter_from_env()
    )
    adapter_class = adapter_instance.__class__.__name__
    backend_health = _probe_llm_shard_adapter_health(adapter_instance)
    if isinstance(adapter_instance, DeterministicBytesShardAdapter):
        return {
            "status": "not_production_ready",
            "ok": False,
            "contractReady": False,
            "productionReady": False,
            "productionReadinessError": (
                "Deterministic test adapter is not a production LLM shard backend."
            ),
            "productionReadinessChecks": {
                "contractReady": False,
                "generationProbeReady": False,
            },
            "selfTestKind": "llm_shard_adapter_contract",
            "modelId": resolved_model_id,
            "adapterClass": adapter_class,
            "adapterId": DETERMINISTIC_BYTES_ADAPTER_ID,
            "adapterVersion": DETERMINISTIC_BYTES_ADAPTER_VERSION,
            "reason": (
                "Deterministic test adapter is not a production LLM shard backend."
            ),
            "generationProbeReady": False,
            "generationProbe": {
                "schemaVersion": 1,
                "ready": False,
                "skipped": True,
                "reason": "deterministic_adapter",
            },
            "recordedAt": recorded_at,
            "latencyMs": _elapsed_ms(started),
        }
    try:
        if _llm_shard_adapter_health_blocks_self_test(backend_health):
            raise CaiOwnedLlmShardBackendHealthError(backend_health)
        metadata_source = _self_test_runtime_metadata(
            resolved_model_id,
            runtime_metadata,
        )
        total_layers = max(
            1,
            _optional_int(metadata_source.get("totalLayerCount"), 2) or 2,
        )
        prefill_layer_end = (
            max(1, min(total_layers - 1, total_layers // 2))
            if total_layers > 1
            else 1
        )
        metadata = build_cai_owned_llm_shard_frame_metadata_from_runtime(
            model_id=resolved_model_id,
            runtime_metadata=metadata_source,
            payload=probe_payload,
            layer_start=0,
            layer_end=prefill_layer_end,
            frame_kind="activation",
            token_start=0,
            token_end=1,
            sequence=0,
        )
        next_layer_start = prefill_layer_end if total_layers > 1 else 0
        next_layer_end = total_layers if total_layers > 1 else 1
        metadata["nextFrameTemplate"] = (
            build_cai_owned_llm_shard_frame_metadata_from_runtime(
                model_id=resolved_model_id,
                runtime_metadata=metadata_source,
                payload=b"",
                layer_start=next_layer_start,
                layer_end=next_layer_end,
                frame_kind="activation",
                token_start=0,
                token_end=1,
                sequence=1,
            )
        )
        prefill_work_item = {
            "sessionId": "caiot_llm_shard_self_test",
            "modelId": resolved_model_id,
            "batch": {
                "batchId": "caibatch_llm_shard_self_test_prefill",
                "phase": "prefill_activation_batches",
                "sourceNodeId": "self-test-requester",
                "sinkNodeId": "self-test-executor",
                "sequence": 0,
                "metadata": metadata,
            },
        }
        config = CaiOwnedShardRuntimeConfig(
            node_id="self-test-executor",
            runtime_id="cai-owned-llm-shard-self-test",
            require_production_llm_handoff=require_production_llm_handoff,
        )
        prefill_result = _run_adapter(
            config,
            adapter_instance,
            prefill_work_item,
            probe_payload,
        )
        prefill_metrics = dict(prefill_result.metrics or {})
        patch_boundary_verified = bool(prefill_metrics.get("patchBoundaryVerified"))
        output_metadata_ready, output_metadata_error = (
            _llm_shard_self_test_output_metadata_ready(
                prefill_result.output_metadata,
                prefill_result.output_payload,
                expected_model_id=resolved_model_id,
            )
        )
        decode_result: CaiOwnedShardAdapterResult | None = None
        decode_metrics: dict[str, Any] = {}
        final_decode_output_ready = False
        final_decode_output_error: str | None = None
        if output_metadata_ready:
            decode_work_item = {
                "sessionId": "caiot_llm_shard_self_test",
                "modelId": resolved_model_id,
                "batch": {
                    "batchId": "caibatch_llm_shard_self_test_decode",
                    "phase": "decode_activation_batches",
                    "sourceNodeId": "self-test-executor",
                    "sinkNodeId": "self-test-executor",
                    "sequence": 1,
                    "metadata": dict(prefill_result.output_metadata),
                },
            }
            decode_result = _run_adapter(
                config,
                adapter_instance,
                decode_work_item,
                prefill_result.output_payload,
            )
            decode_metrics = dict(decode_result.metrics or {})
            final_decode_output_ready, final_decode_output_error = (
                _llm_shard_self_test_final_output_ready(
                    decode_result.output_payload,
                )
            )
        else:
            final_decode_output_error = (
                "Skipping decode self-test because output frame metadata is invalid."
            )
        metrics = {**prefill_metrics, **decode_metrics}
        contract_ready = (
            bool(prefill_result.output_payload)
            and output_metadata_ready
            and final_decode_output_ready
            and (
                patch_boundary_verified
                if isinstance(adapter_instance, ExternalLlamaCppShardAdapter)
                else True
            )
        )
        (
            production_ready,
            production_readiness_error,
            production_readiness_checks,
        ) = _llm_shard_self_test_production_readiness(
            metrics,
            contract_ready=contract_ready,
            patch_boundary_verified=patch_boundary_verified,
            backend_health=backend_health,
        )
        generation_probe = _llm_shard_self_test_generation_probe(
            adapter_instance,
            model_id=resolved_model_id,
            enabled=bool(production_ready or require_generation_probe),
        )
        production_readiness_checks["generationProbeReady"] = bool(
            generation_probe.get("ready")
        )
        production_readiness_checks["generationProbe"] = dict(generation_probe)
        if bool(require_generation_probe) and production_ready and not bool(
            generation_probe.get("ready")
        ):
            production_ready = False
            production_readiness_error = str(
                generation_probe.get("error")
                or "LLM shard backend generation probe did not pass."
            )
        result = {
            "status": "passed" if contract_ready else "failed",
            "ok": contract_ready,
            "contractReady": contract_ready,
            "productionReady": production_ready,
            "productionReadinessError": production_readiness_error,
            "productionReadinessChecks": production_readiness_checks,
            "selfTestKind": "llm_shard_adapter_contract",
            "modelId": resolved_model_id,
            "adapterClass": adapter_class,
            "adapterId": metrics.get("adapterId"),
            "adapterVersion": metrics.get("adapterVersion"),
            "backend": metrics.get("backend"),
            "backendVersion": metrics.get("backendVersion"),
            "backendMode": metrics.get("backendMode"),
            "patchBoundaryVerified": patch_boundary_verified,
            "patchBoundaryAbi": metrics.get("patchBoundaryAbi"),
            "patchBoundaryPatchId": metrics.get("patchBoundaryPatchId"),
            "patchBoundaryHash": metrics.get("patchBoundaryHash"),
            "outputFrameMetadataReady": output_metadata_ready,
            "outputFrameMetadataError": output_metadata_error,
            "finalDecodeOutputReady": final_decode_output_ready,
            "finalDecodeOutputError": final_decode_output_error,
            "generationProbeReady": bool(generation_probe.get("ready")),
            "generationProbe": dict(generation_probe),
            "prefillOutputPayloadSizeBytes": len(prefill_result.output_payload),
            "prefillOutputPayloadSha256Hex": hashlib.sha256(
                prefill_result.output_payload,
            ).hexdigest(),
            "decodeOutputPayloadSizeBytes": (
                len(decode_result.output_payload)
                if decode_result is not None
                else 0
            ),
            "decodeOutputPayloadSha256Hex": (
                hashlib.sha256(decode_result.output_payload).hexdigest()
                if decode_result is not None
                else None
            ),
            "outputPayloadSizeBytes": (
                len(decode_result.output_payload)
                if decode_result is not None
                else len(prefill_result.output_payload)
            ),
            "outputPayloadSha256Hex": (
                hashlib.sha256(decode_result.output_payload).hexdigest()
                if decode_result is not None
                else hashlib.sha256(prefill_result.output_payload).hexdigest()
            ),
            "recordedAt": recorded_at,
            "latencyMs": _elapsed_ms(started),
        }
        _attach_llm_shard_backend_health(result, backend_health)
        return result
    except Exception as exc:
        error_text = str(exc)
        metadata_error = (
            error_text
            if "output frame metadata" in error_text.lower()
            or "output metadata" in error_text.lower()
            else None
        )
        result = {
            "status": "failed",
            "ok": False,
            "contractReady": False,
            "productionReady": False,
            "productionReadinessError": error_text,
            "productionReadinessChecks": {
                "contractReady": False,
                "generationProbeReady": False,
            },
            "selfTestKind": "llm_shard_adapter_contract",
            "modelId": resolved_model_id,
            "adapterClass": adapter_class,
            "errorClass": exc.__class__.__name__,
            "error": error_text,
            "patchBoundaryVerified": False,
            "outputFrameMetadataReady": False,
            "outputFrameMetadataError": metadata_error,
            "finalDecodeOutputReady": False,
            "finalDecodeOutputError": "Self-test failed before decode.",
            "generationProbeReady": False,
            "generationProbe": {
                "schemaVersion": 1,
                "ready": False,
                "skipped": True,
                "reason": "self_test_failed_before_generation_probe",
            },
            "recordedAt": recorded_at,
            "latencyMs": _elapsed_ms(started),
        }
        _attach_llm_shard_backend_health(result, backend_health)
        return result


def _exception_backend_health_metrics(exc: Exception) -> dict[str, Any]:
    backend_health = getattr(exc, "backend_health", None)
    return _llm_shard_backend_health_metrics(
        backend_health if isinstance(backend_health, Mapping) else None,
    )


def _llm_shard_backend_health_metrics(
    backend_health: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(backend_health, Mapping):
        return {}
    metrics: dict[str, Any] = {
        "backendHealth": dict(backend_health),
        "backendHealthReady": _llm_shard_adapter_health_ready(backend_health),
    }
    backend_health_status = str(backend_health.get("status") or "").strip()
    if backend_health_status:
        metrics["backendHealthStatus"] = backend_health_status
    return metrics


def _probe_llm_shard_adapter_health(adapter: object) -> dict[str, Any] | None:
    probe = getattr(adapter, "probe_health", None)
    if not callable(probe):
        return None
    try:
        health = probe()
    except Exception as exc:
        return {
            "status": "unknown",
            "ready": None,
            "healthEndpointAvailable": False,
            "errorClass": exc.__class__.__name__,
            "error": str(exc)[:500],
        }
    return dict(health) if isinstance(health, Mapping) else None


def _attach_llm_shard_backend_health(
    result: dict[str, Any],
    backend_health: Mapping[str, Any] | None,
) -> None:
    if backend_health is None:
        return
    result["backendHealth"] = dict(backend_health)
    result["backendHealthReady"] = _llm_shard_adapter_health_ready(backend_health)


def _llm_shard_adapter_health_ready(
    backend_health: Mapping[str, Any] | None,
) -> bool | None:
    if not isinstance(backend_health, Mapping):
        return None
    if backend_health.get("healthEndpointAvailable") is False:
        return None
    status = str(backend_health.get("status") or "").strip().lower()
    if status in {"degraded", "failed", "failure", "error", "unhealthy"}:
        return False
    if backend_health.get("nativeCommandConfigured") is False:
        return False
    persistent_engine = backend_health.get("persistentEngine")
    if (
        isinstance(persistent_engine, Mapping)
        and persistent_engine.get("alive") is False
    ):
        return False
    explicit_ready = backend_health.get("ready")
    if explicit_ready is not None:
        return bool(explicit_ready)
    if status in {"ok", "ready", "healthy"}:
        return True
    return None


def _llm_shard_adapter_health_blocks_self_test(
    backend_health: Mapping[str, Any] | None,
) -> bool:
    return _llm_shard_adapter_health_ready(backend_health) is False


def _llm_shard_adapter_health_error(
    backend_health: Mapping[str, Any] | None,
) -> str:
    detail = ""
    if isinstance(backend_health, Mapping):
        detail = str(
            backend_health.get("error")
            or backend_health.get("message")
            or backend_health.get("status")
            or ""
        ).strip()
    return (
        "External llama.cpp shard adapter health check failed"
        + (f": {detail}" if detail else ".")
    )


def _llm_shard_adapter_prefers_deferred_finalize(
    adapter: object,
    backend_health: Mapping[str, Any] | None,
) -> bool:
    if not isinstance(backend_health, Mapping):
        return False
    if not bool(getattr(adapter, "defer_finalize_when_persistent", True)):
        return False
    mode = str(backend_health.get("nativeEngineMode") or "").strip().lower()
    if mode != "persistent_jsonl":
        return False
    ready = _llm_shard_adapter_health_ready(backend_health)
    return ready is not False


def _llm_shard_self_test_output_metadata_ready(
    metadata: Mapping[str, Any] | None,
    output_payload: bytes,
    *,
    expected_model_id: str,
) -> tuple[bool, str | None]:
    if not isinstance(metadata, Mapping) or not metadata:
        return False, "External shard adapter output frame metadata is missing."
    metadata_dict = dict(metadata)
    output_hash = hashlib.sha256(bytes(output_payload or b"")).hexdigest()
    declared_hash = str(metadata_dict.get("payloadSha256Hex") or "").strip().lower()
    if declared_hash != output_hash:
        return False, "External shard adapter output frame metadata hash mismatch."
    handoff = metadata_dict.get("llmHandoff")
    tensor = handoff.get("tensor") if isinstance(handoff, Mapping) else None
    tensor_hash = (
        str(tensor.get("sha256Hex") or "").strip().lower()
        if isinstance(tensor, Mapping)
        else ""
    )
    if tensor_hash != output_hash:
        return False, "External shard adapter output handoff tensor hash mismatch."
    valid, error = validate_cai_owned_transport_frame_metadata(
        metadata_dict,
        expected_model_id=expected_model_id,
        require_llm_handoff=True,
    )
    if not valid:
        return False, error or "External shard adapter output frame metadata is invalid."
    return True, None


def _llm_shard_self_test_final_output_ready(
    output_payload: bytes,
) -> tuple[bool, str | None]:
    if not bytes(output_payload or b""):
        return False, "External shard adapter final decode output is empty."
    return True, None


def _llm_shard_self_test_generation_probe(
    adapter: object,
    *,
    model_id: str,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {
            "schemaVersion": 1,
            "ready": False,
            "skipped": True,
            "reason": "production_readiness_baseline_not_ready",
        }
    probe = getattr(adapter, "probe_generation", None)
    if not callable(probe):
        return {
            "schemaVersion": 1,
            "ready": False,
            "skipped": True,
            "error": "LLM shard adapter does not expose generation probe.",
        }
    started = time.perf_counter()
    prompt = "CAI generation probe: reply with one short word."
    try:
        response = probe(model_id=model_id, prompt=prompt, max_tokens=8)
        ready, error, audit = _llm_shard_generation_probe_ready(
            response if isinstance(response, Mapping) else {},
            expected_model_id=model_id,
        )
        audit["latencyMs"] = _elapsed_ms(started)
        audit["ready"] = bool(ready)
        if error:
            audit["error"] = error
        return audit
    except Exception as exc:
        return {
            "schemaVersion": 1,
            "ready": False,
            "errorClass": exc.__class__.__name__,
            "error": (
                "LLM shard backend generation probe failed: "
                + str(exc)[:500]
            ),
            "latencyMs": _elapsed_ms(started),
        }


def _llm_shard_generation_probe_ready(
    response: Mapping[str, Any],
    *,
    expected_model_id: str,
) -> tuple[bool, str | None, dict[str, Any]]:
    payload = response.get("generationProbe")
    if not isinstance(payload, Mapping):
        payload = response
    model_id = str(
        payload.get("modelId")
        or payload.get("model_id")
        or response.get("modelId")
        or ""
    ).strip()
    output_text = str(
        payload.get("outputText")
        or payload.get("output_text")
        or payload.get("text")
        or payload.get("answer")
        or ""
    )
    output_tokens = _optional_int(
        payload.get("outputTokenCount")
        or payload.get("completionTokenCount")
        or payload.get("output_tokens")
        or payload.get("completion_tokens")
        or payload.get("tokensGenerated")
    )
    real_model_execution = bool(
        payload.get("realModelExecution")
        or payload.get("real_model_execution")
        or response.get("realModelExecution")
    )
    real_layer_execution = bool(
        payload.get("realLayerExecution")
        or payload.get("real_layer_execution")
        or response.get("realLayerExecution")
    )
    probe_reason = str(
        payload.get("reason")
        or payload.get("error")
        or response.get("reason")
        or response.get("error")
        or ""
    ).strip()
    audit: dict[str, Any] = {
        "schemaVersion": 1,
        "probeAbi": str(
            payload.get("abi") or LLAMA_CPP_EXTERNAL_SHARD_GENERATION_PROBE_ABI
        ).strip(),
        "modelId": model_id or None,
        "outputTextPreview": output_text[:120],
        "outputTokenCount": max(0, int(output_tokens or 0)),
        "realModelExecution": real_model_execution,
        "realLayerExecution": real_layer_execution,
    }
    if probe_reason:
        audit["reason"] = probe_reason[:500]
    error_class = str(payload.get("errorClass") or response.get("errorClass") or "").strip()
    if error_class:
        audit["errorClass"] = error_class[:120]
    metrics = response.get("metrics")
    if isinstance(metrics, Mapping):
        backend_mode = str(metrics.get("backendMode") or "").strip()
        if backend_mode:
            audit["backendMode"] = backend_mode
    expected = str(expected_model_id or "").strip()
    if model_id and expected and model_id != expected:
        return False, "LLM shard backend generation probe model id mismatch.", audit
    if not real_model_execution:
        return (
            False,
            probe_reason
            or "LLM shard backend generation probe did not prove real model execution.",
            audit,
        )
    if not output_text:
        return False, "LLM shard backend generation probe returned empty output.", audit
    if int(output_tokens or 0) <= 0:
        return (
            False,
            "LLM shard backend generation probe did not report generated tokens.",
            audit,
        )
    return True, None, audit


def _llm_shard_self_test_production_readiness(
    metrics: Mapping[str, Any],
    *,
    contract_ready: bool,
    patch_boundary_verified: bool,
    backend_health: Mapping[str, Any] | None = None,
) -> tuple[bool, str | None, dict[str, Any]]:
    backend_mode = str(metrics.get("backendMode") or "").strip()
    patch_id = str(metrics.get("patchBoundaryPatchId") or "").strip()
    capabilities = set(_clean_string_sequence(metrics.get("backendCapabilities")))
    patch_boundary_capabilities = set(
        _clean_string_sequence(metrics.get("patchBoundaryCapabilities"))
    )
    missing_capabilities = sorted(
        item
        for item in LLAMA_CPP_EXTERNAL_SHARD_PRODUCTION_CAPABILITIES
        if item not in patch_boundary_capabilities
    )
    extra_metadata = metrics.get("patchBoundaryExtraMetadata")
    if not isinstance(extra_metadata, Mapping):
        extra_metadata = {}
    state_contract = extra_metadata.get("productionStateContract")
    state_contract_ready, state_contract_error = _production_state_contract_ready(
        state_contract,
    )
    backend_health_ready = _llm_shard_adapter_health_ready(backend_health)
    checks: dict[str, Any] = {
        "contractReady": bool(contract_ready),
        "patchBoundaryVerified": bool(patch_boundary_verified),
        "backendMode": backend_mode or None,
        "patchBoundaryPatchId": patch_id or None,
        "requiredProductionCapabilities": list(
            LLAMA_CPP_EXTERNAL_SHARD_PRODUCTION_CAPABILITIES,
        ),
        "backendCapabilities": sorted(capabilities),
        "patchBoundaryCapabilities": sorted(patch_boundary_capabilities),
        "missingProductionCapabilities": missing_capabilities,
        "productionStateContractReady": state_contract_ready,
    }
    if backend_health is not None:
        checks["backendHealthReady"] = backend_health_ready
        checks["backendHealthStatus"] = str(
            backend_health.get("status") or "unknown"
        ).strip()
    if state_contract_error:
        checks["productionStateContractError"] = state_contract_error

    if not contract_ready:
        return False, "LLM shard adapter contract self-test did not pass.", checks
    if not patch_boundary_verified:
        return False, "LLM shard adapter patch boundary is not verified.", checks
    if backend_mode.lower() in {"", "smoke_runner", "unit_test", "test"}:
        return False, "LLM shard adapter backend mode is not production.", checks
    if "smoke" in patch_id.lower() or "test" in patch_id.lower():
        return False, "LLM shard adapter patch id is not production.", checks
    if missing_capabilities:
        return (
            False,
            "LLM shard adapter missing production capabilities: "
            + ", ".join(missing_capabilities),
            checks,
        )
    if backend_health_ready is False:
        return False, _llm_shard_adapter_health_error(backend_health), checks
    if not state_contract_ready:
        return False, state_contract_error, checks
    return True, None, checks


def _production_state_contract_ready(value: Any) -> tuple[bool, str | None]:
    valid, error = validate_llama_cpp_external_shard_production_state_contract(value)
    if not valid:
        return False, error
    return True, None


def cai_owned_llm_shard_self_test_file_path(
    policy: WalletPolicy | None = None,
) -> Path:
    return data_root(policy) / "cai-owned-llm-shard-self-test.json"


def cai_owned_transport_live_proof_file_path(
    policy: WalletPolicy | None = None,
) -> Path:
    return data_root(policy) / "cai-owned-transport-live-proof.json"


def save_cai_owned_llm_shard_self_test_result(
    result: Mapping[str, Any],
    *,
    policy: WalletPolicy | None = None,
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise ValueError("CAI-owned LLM shard self-test result is invalid.")
    payload = {
        "schemaVersion": CAI_LLM_SHARD_SELF_TEST_CACHE_SCHEMA_VERSION,
        "recordedAt": str(
            result.get("recordedAt") or datetime.now(tz=UTC).isoformat()
        ),
        "result": dict(result),
    }
    path = cai_owned_llm_shard_self_test_file_path(policy)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "path": str(path),
        "recordedAt": payload["recordedAt"],
        "schemaVersion": payload["schemaVersion"],
    }


def load_cai_owned_llm_shard_self_test_result(
    *,
    max_age_seconds: float | int | None = CAI_LLM_SHARD_SELF_TEST_CACHE_TTL_SECONDS,
    policy: WalletPolicy | None = None,
) -> dict[str, Any] | None:
    path = cai_owned_llm_shard_self_test_file_path(policy)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    try:
        schema_version = int(payload.get("schemaVersion") or 0)
    except (TypeError, ValueError):
        return None
    if schema_version != CAI_LLM_SHARD_SELF_TEST_CACHE_SCHEMA_VERSION:
        return None
    result = payload.get("result")
    if not isinstance(result, Mapping):
        return None
    if not _llm_shard_self_test_cache_result_supported(result):
        return None
    recorded_at = _parse_datetime(payload.get("recordedAt") or result.get("recordedAt"))
    if recorded_at is None:
        return None
    age_seconds = max(0.0, (datetime.now(tz=UTC) - recorded_at).total_seconds())
    if max_age_seconds is not None and age_seconds > max(0.0, float(max_age_seconds)):
        return None
    loaded = dict(result)
    loaded.setdefault("recordedAt", recorded_at.isoformat())
    loaded["cacheAgeSeconds"] = age_seconds
    return loaded


def _llm_shard_self_test_cache_result_supported(result: Mapping[str, Any]) -> bool:
    if not bool(result.get("productionReady")):
        return True
    if result.get("generationProbeReady") is not True:
        return False
    checks = result.get("productionReadinessChecks")
    if not isinstance(checks, Mapping):
        return False
    if checks.get("generationProbeReady") is not True:
        return False
    probe = result.get("generationProbe")
    if not isinstance(probe, Mapping):
        return False
    if probe.get("ready") is not True:
        return False
    return probe.get("realModelExecution") is True


def save_cai_owned_transport_live_proof_result(
    result: Mapping[str, Any],
    *,
    policy: WalletPolicy | None = None,
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise ValueError("CAI-owned transport live proof result is invalid.")
    audit = cai_owned_transport_live_proof_audit(result)
    if not audit.get("verified"):
        raise ValueError(
            "CAI-owned transport live proof is not verified: "
            + str(audit.get("error") or "unknown error")
        )
    payload = {
        "schemaVersion": CAI_OWNED_TRANSPORT_LIVE_PROOF_CACHE_SCHEMA_VERSION,
        "recordedAt": str(
            result.get("recordedAt") or datetime.now(tz=UTC).isoformat()
        ),
        "result": dict(result),
    }
    path = cai_owned_transport_live_proof_file_path(policy)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "path": str(path),
        "recordedAt": payload["recordedAt"],
        "schemaVersion": payload["schemaVersion"],
        "audit": audit,
    }


def load_cai_owned_transport_live_proof_result(
    *,
    max_age_seconds: float | int | None = CAI_OWNED_TRANSPORT_LIVE_PROOF_CACHE_TTL_SECONDS,
    policy: WalletPolicy | None = None,
) -> dict[str, Any] | None:
    path = cai_owned_transport_live_proof_file_path(policy)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    try:
        schema_version = int(payload.get("schemaVersion") or 0)
    except (TypeError, ValueError):
        return None
    if schema_version != CAI_OWNED_TRANSPORT_LIVE_PROOF_CACHE_SCHEMA_VERSION:
        return None
    result = payload.get("result")
    if not isinstance(result, Mapping):
        return None
    audit = cai_owned_transport_live_proof_audit(result)
    if not audit.get("verified"):
        return None
    recorded_at = _parse_datetime(payload.get("recordedAt") or result.get("recordedAt"))
    if recorded_at is None:
        return None
    age_seconds = max(0.0, (datetime.now(tz=UTC) - recorded_at).total_seconds())
    if max_age_seconds is not None and age_seconds > max(0.0, float(max_age_seconds)):
        return None
    loaded = dict(result)
    loaded.setdefault("recordedAt", recorded_at.isoformat())
    loaded["cacheAgeSeconds"] = age_seconds
    loaded["runtimeReadyProofAudit"] = audit
    return loaded


def cai_owned_transport_live_proof_audit(
    result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        return {
            "verified": False,
            "error": "CAI-owned transport live proof is missing.",
        }
    final_result = result.get("finalResult")
    if not isinstance(final_result, Mapping):
        final_result = {}
    executor_ids = _clean_string_sequence(
        result.get("executorNodeIds")
        or final_result.get("executorNodeIds")
        or final_result.get("executorIds")
    )
    unique_executor_ids = sorted(set(executor_ids))
    proof_verified = bool(
        result.get("proofVerified")
        or final_result.get("proofVerified")
        or result.get("verified")
    )
    final_output = final_result.get("finalOutput") or result.get("finalOutput")
    has_final_output = isinstance(final_output, Mapping) and bool(final_output)
    status = str(result.get("status") or "").strip().lower()
    status_ok = status in {"ok", "passed", "verified", "ready"} or proof_verified
    audit: dict[str, Any] = {
        "verified": False,
        "status": status or None,
        "proofVerified": proof_verified,
        "requesterNodeId": str(result.get("requesterNodeId") or "").strip() or None,
        "sessionId": str(result.get("sessionId") or "").strip() or None,
        "instanceId": str(result.get("instanceId") or "").strip() or None,
        "executorNodeIds": unique_executor_ids,
        "executorCount": len(unique_executor_ids),
        "hasFinalOutput": has_final_output,
    }
    if not status_ok:
        audit["error"] = "Live proof status is not ok."
        return audit
    if not proof_verified:
        audit["error"] = "Live proof is not verified."
        return audit
    if len(unique_executor_ids) < 2:
        audit["error"] = "Live proof requires at least two executor nodes."
        return audit
    if not has_final_output:
        audit["error"] = "Live proof final output is missing."
        return audit
    audit["verified"] = True
    return audit


def cai_owned_transport_runtime_capacity_status(
    config: CaiOwnedShardRuntimeConfig,
) -> dict[str, Any]:
    node_id = _require_runtime_node_id(config.node_id)
    max_concurrent_batches = _positive_int(config.max_concurrent_batches, 1)
    max_payload_size_bytes = _positive_int(
        config.max_payload_size_bytes,
        16 * 1024 * 1024,
    )
    processing = list_cai_owned_transport_batch_inbox(
        node_id,
        status="processing",
        policy=config.policy,
    )
    queued = list_cai_owned_transport_batch_inbox(
        node_id,
        status="received",
        policy=config.policy,
    )
    now = datetime.now(tz=UTC)
    active_processing = [
        item
        for item in processing
        if _batch_lease_active(item.get("batch"), now)
    ]
    oversized_queued = [
        item
        for item in queued
        if _batch_payload_size(item.get("batch")) > max_payload_size_bytes
    ]
    can_claim = len(active_processing) < max_concurrent_batches
    return {
        "nodeId": node_id,
        "status": "available" if can_claim else "busy",
        "canClaim": can_claim,
        "activeProcessingCount": len(active_processing),
        "queuedBatchCount": len(queued),
        "oversizedQueuedBatchCount": len(oversized_queued),
        "maxConcurrentBatches": max_concurrent_batches,
        "maxPayloadSizeBytes": max_payload_size_bytes,
        "activeBatchIds": [
            str((item.get("batch") or {}).get("batchId") or "")
            for item in active_processing
            if isinstance(item.get("batch"), dict)
        ],
    }


def run_cai_owned_shard_runtime_once(
    config: CaiOwnedShardRuntimeConfig,
    adapter: CaiOwnedShardAdapter,
) -> dict[str, Any]:
    node_id = _require_runtime_node_id(config.node_id)
    runtime_id = _require_runtime_id(config.runtime_id)
    capacity = cai_owned_transport_runtime_capacity_status(config)
    if not capacity["canClaim"]:
        return {
            "status": "busy",
            "capacity": capacity,
            "workItem": None,
        }

    work_item = claim_next_cai_owned_transport_batch(
        node_id,
        runtime_id=runtime_id,
        runtime_auth_token=config.local_runtime_auth_token,
        require_runtime_auth=config.require_local_runtime_auth,
        lease_seconds=config.lease_seconds,
        policy=config.policy,
    )
    if work_item is None:
        return {
            "status": "idle",
            "capacity": capacity,
            "workItem": None,
        }

    session_id = str(work_item.get("sessionId") or "").strip()
    batch = work_item.get("batch")
    if not isinstance(batch, dict):
        raise ValueError("CAI-owned runtime work item batch is missing.")
    batch_id = str(batch.get("batchId") or "").strip()
    payload_size = _batch_payload_size(batch)
    max_payload_size = _positive_int(config.max_payload_size_bytes, 16 * 1024 * 1024)
    if payload_size > max_payload_size:
        failure = fail_cai_owned_transport_work_item(
            session_id,
            batch_id,
            node_id=node_id,
            runtime_id=runtime_id,
            runtime_auth_token=config.local_runtime_auth_token,
            require_runtime_auth=config.require_local_runtime_auth,
            error=(
                "CAI-owned transport payload exceeds runtime capacity: "
                f"{payload_size} > {max_payload_size} bytes."
            ),
            retryable=False,
            max_attempts=config.max_attempts,
            metrics={
                "errorClass": "PayloadTooLarge",
                "runtimeId": runtime_id,
                "runtimeVersion": CAI_OWNED_SHARD_RUNTIME_VERSION,
                "payloadSizeBytes": payload_size,
                "maxPayloadSizeBytes": max_payload_size,
            },
            policy=config.policy,
        )
        return {
            "status": "failed",
            "reason": "payload_too_large",
            "capacity": capacity,
            "workItem": work_item,
            "failure": failure,
        }

    started = time.perf_counter()
    heartbeat_stop = threading.Event()
    heartbeat_errors: list[str] = []
    heartbeat_thread = _start_work_item_heartbeat(
        config,
        session_id=session_id,
        batch_id=batch_id,
        node_id=node_id,
        runtime_id=runtime_id,
        stop_event=heartbeat_stop,
        errors=heartbeat_errors,
    )
    try:
        payload = read_cai_owned_transport_batch_payload(
            session_id,
            batch_id,
            config.policy,
        )
        adapter_result = _run_adapter(config, adapter, work_item, payload)
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)
        processing_latency_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        metrics = _runtime_metrics(
            config,
            runtime_id=runtime_id,
            adapter_metrics=adapter_result.metrics,
            payload_size_bytes=len(payload),
            output_payload_size_bytes=len(adapter_result.output_payload),
            processing_latency_ms=processing_latency_ms,
        )
        if heartbeat_errors:
            metrics["leaseHeartbeatErrors"] = list(heartbeat_errors[-3:])
        route_audit = _work_item_route_audit(work_item)
        runtime_audit = _runtime_audit(config, runtime_id, metrics)
        output_forward = _forward_output_if_requested(
            config,
            work_item,
            adapter_result,
            metrics=metrics,
            route_audit=route_audit,
            runtime_audit=runtime_audit,
        )
        output_forward_error = _blocking_output_forward_error(output_forward)
        if output_forward_error:
            raise RuntimeError(output_forward_error)
        coordinator_cai_url = config.coordinator_cai_url or _coordinator_url_for_work_item(
            config,
            work_item,
        )
        completion = complete_cai_owned_transport_work_item(
            session_id,
            batch_id,
            node_id=node_id,
            runtime_id=runtime_id,
            runtime_auth_token=config.local_runtime_auth_token,
            require_runtime_auth=config.require_local_runtime_auth,
            coordinator_cai_url=coordinator_cai_url,
            metrics=metrics,
            output_payload=adapter_result.output_payload,
            route_audit=route_audit,
            runtime_audit=runtime_audit,
            signing_material=config.signing_material,
            policy=config.policy,
        )
        return {
            "status": "processed",
            "capacity": capacity,
            "workItem": work_item,
            "completion": completion,
            "outputForward": output_forward,
            "outputPayloadSizeBytes": len(adapter_result.output_payload),
        }
    except Exception as exc:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)
        processing_latency_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        failure_metrics = {
            "errorClass": exc.__class__.__name__,
            "runtimeId": runtime_id,
            "runtimeVersion": CAI_OWNED_SHARD_RUNTIME_VERSION,
            "processingLatencyMs": processing_latency_ms,
        }
        if heartbeat_errors:
            failure_metrics["leaseHeartbeatErrors"] = list(heartbeat_errors[-3:])
        failure_metrics.update(_exception_backend_health_metrics(exc))
        failure = fail_cai_owned_transport_work_item(
            session_id,
            batch_id,
            node_id=node_id,
            runtime_id=runtime_id,
            runtime_auth_token=config.local_runtime_auth_token,
            require_runtime_auth=config.require_local_runtime_auth,
            error=str(exc),
            retryable=_runtime_error_retryable(exc),
            max_attempts=config.max_attempts,
            metrics=failure_metrics,
            policy=config.policy,
        )
        return {
            "status": "retry_scheduled" if failure["retryScheduled"] else "failed",
            "capacity": capacity,
            "workItem": work_item,
            "failure": failure,
        }


def _start_work_item_heartbeat(
    config: CaiOwnedShardRuntimeConfig,
    *,
    session_id: str,
    batch_id: str,
    node_id: str,
    runtime_id: str,
    stop_event: threading.Event,
    errors: list[str],
) -> threading.Thread:
    try:
        lease_seconds = float(config.lease_seconds)
    except (TypeError, ValueError):
        lease_seconds = 60.0
    lease_seconds = max(1.0, lease_seconds)
    interval_seconds = max(1.0, min(15.0, lease_seconds / 3.0))

    def heartbeat_loop() -> None:
        while not stop_event.wait(interval_seconds):
            try:
                heartbeat_cai_owned_transport_batch(
                    session_id,
                    batch_id,
                    node_id=node_id,
                    runtime_id=runtime_id,
                    runtime_auth_token=config.local_runtime_auth_token,
                    require_runtime_auth=config.require_local_runtime_auth,
                    lease_seconds=lease_seconds,
                    policy=config.policy,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
                if "lease has expired" in str(exc).lower():
                    return

    thread = threading.Thread(
        target=heartbeat_loop,
        name=f"cai-owned-heartbeat-{batch_id}",
        daemon=True,
    )
    thread.start()
    return thread


def _task_level_inference_stage(metadata: object) -> bool:
    if not isinstance(metadata, Mapping):
        return False
    if metadata.get("finalOutput") is True:
        return True
    route_plan = _output_route_plan(metadata)
    return bool(route_plan and route_plan[0].get("finalOutput") is True)


def _task_inference_request_payload(
    work_item: Mapping[str, Any],
    payload: bytes,
    *,
    model_id: str | None,
    max_tokens: int | None,
    temperature: float | None,
) -> tuple[dict[str, Any], str]:
    text = bytes(payload or b"").decode("utf-8", errors="replace")
    parsed: Any = None
    try:
        parsed = json.loads(text) if text.strip() else None
    except json.JSONDecodeError:
        parsed = None

    resolved_model_id = (
        str(model_id or "").strip()
        or str(work_item.get("modelId") or "").strip()
        or NetworkModelPolicy().network_default_model_id
    )
    if isinstance(parsed, Mapping):
        request_payload = dict(parsed)
        if isinstance(request_payload.get("messages"), Sequence) and not isinstance(
            request_payload.get("messages"),
            (str, bytes),
        ):
            prompt_text = _messages_prompt_text(request_payload.get("messages"))
        else:
            prompt_text = str(
                request_payload.get("prompt")
                or request_payload.get("input")
                or request_payload.get("text")
                or ""
            )
            request_payload["messages"] = [
                {"role": "user", "content": prompt_text},
            ]
            request_payload.pop("prompt", None)
            request_payload.pop("input", None)
            request_payload.pop("text", None)
    else:
        prompt_text = text
        request_payload = {
            "messages": [{"role": "user", "content": prompt_text}],
        }

    request_payload["model"] = (
        str(request_payload.get("model") or "").strip() or resolved_model_id
    )
    request_payload["stream"] = False
    if max_tokens is not None and "max_tokens" not in request_payload:
        request_payload["max_tokens"] = max(1, int(max_tokens))
    if temperature is not None and "temperature" not in request_payload:
        request_payload["temperature"] = float(temperature)
    return request_payload, prompt_text


def _messages_prompt_text(messages: object) -> str:
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return ""
    parts: list[str] = []
    for item in messages:
        if not isinstance(item, Mapping):
            continue
        content = item.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
            for segment in content:
                if isinstance(segment, Mapping):
                    text = segment.get("text")
                    if isinstance(text, str):
                        parts.append(text)
    return "\n".join(part for part in parts if part)


def _task_inference_response_text(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, Sequence) and not isinstance(choices, (str, bytes)):
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            message = choice.get("message")
            if isinstance(message, Mapping):
                content = message.get("content")
                if isinstance(content, str) and content:
                    return content
            text = choice.get("text")
            if isinstance(text, str) and text:
                return text
    for field_name in ("answer", "response", "output", "text", "content"):
        value = payload.get(field_name)
        if isinstance(value, str) and value:
            return value
    raise ValueError("Task-level HTTP inference adapter response text is missing.")


def _task_inference_response_metrics(
    payload: Mapping[str, Any],
    *,
    prompt_text: str,
    output_text: str,
) -> dict[str, Any]:
    usage = payload.get("usage")
    metrics: dict[str, Any] = {
        "promptBytes": len(prompt_text.encode("utf-8")),
        "completionBytes": len(output_text.encode("utf-8")),
    }
    if isinstance(usage, Mapping):
        prompt_tokens = _optional_int(
            usage.get("prompt_tokens")
            or usage.get("promptTokens")
            or usage.get("input_tokens")
            or usage.get("inputTokens")
        )
        completion_tokens = _optional_int(
            usage.get("completion_tokens")
            or usage.get("completionTokens")
            or usage.get("output_tokens")
            or usage.get("outputTokens")
        )
        total_tokens = _optional_int(
            usage.get("total_tokens") or usage.get("totalTokens")
        )
        if total_tokens is None and (
            prompt_tokens is not None or completion_tokens is not None
        ):
            total_tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)
        if prompt_tokens is not None:
            metrics["promptTokenCount"] = max(0, int(prompt_tokens))
            metrics["inputTokenCount"] = max(0, int(prompt_tokens))
        if completion_tokens is not None:
            metrics["completionTokenCount"] = max(0, int(completion_tokens))
            metrics["outputTokenCount"] = max(0, int(completion_tokens))
        if total_tokens is not None:
            metrics["totalTokenCount"] = max(0, int(total_tokens))
        if any(
            key in metrics
            for key in ("promptTokenCount", "completionTokenCount", "totalTokenCount")
        ):
            metrics["tokenUsageSource"] = "task_inference_endpoint_usage"
            return metrics

    metrics["tokenUsageSource"] = "not_reported"
    return metrics


def _safe_endpoint_label(endpoint_url: str) -> str:
    parsed = urlsplit(str(endpoint_url or "").strip())
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _ensure_stable_patch_boundary_metrics(
    load_metrics: Mapping[str, Any],
    stage_metrics: Mapping[str, Any] | None,
    *,
    stage_name: str,
) -> None:
    if not isinstance(stage_metrics, Mapping):
        return
    load_hash = str(load_metrics.get("patchBoundaryHash") or "").strip().lower()
    stage_hash = str(stage_metrics.get("patchBoundaryHash") or "").strip().lower()
    if load_hash and stage_hash and load_hash != stage_hash:
        raise ValueError(
            "External llama.cpp shard adapter patch boundary changed between "
            f"load_shard and {stage_name}."
        )


def _run_adapter(
    config: CaiOwnedShardRuntimeConfig,
    adapter: object,
    work_item: Mapping[str, Any],
    payload: bytes,
) -> CaiOwnedShardAdapterResult:
    if _has_llm_adapter_interface(adapter):
        frame = _work_item_frame(
            work_item,
            payload,
            require_production_llm_handoff=config.require_production_llm_handoff,
        )
        backend_health = _probe_llm_shard_adapter_health(adapter)
        if _llm_shard_adapter_health_blocks_self_test(backend_health):
            raise CaiOwnedLlmShardBackendHealthError(backend_health)
        metrics: dict[str, Any] = _llm_shard_backend_health_metrics(backend_health)
        load_metrics = adapter.load_shard(frame)  # type: ignore[attr-defined]
        if isinstance(load_metrics, Mapping):
            metrics.update(dict(load_metrics))
        load_metrics_snapshot = dict(metrics)
        if frame.phase == "prefill_activation_batches":
            result = adapter.process_prefill(frame)  # type: ignore[attr-defined]
        elif frame.phase == "decode_activation_batches":
            result = adapter.process_decode(frame)  # type: ignore[attr-defined]
        else:
            raise ValueError(
                f"CAI-owned shard adapter phase is unsupported: {frame.phase}"
            )
        result = _coerce_adapter_result(result)
        _ensure_stable_patch_boundary_metrics(
            load_metrics_snapshot,
            result.metrics,
            stage_name=(
                "process_prefill"
                if frame.phase == "prefill_activation_batches"
                else "process_decode"
            ),
        )
        metrics.update(result.metrics)
        if _llm_shard_adapter_prefers_deferred_finalize(adapter, backend_health):
            metrics.setdefault("backendFinalizeDeferred", True)
            metrics.setdefault("backendRetainedResidentShard", True)
            metrics.setdefault("backendFinalized", False)
        else:
            finalize_metrics = adapter.finalize(frame, result)  # type: ignore[attr-defined]
            if isinstance(finalize_metrics, Mapping):
                _ensure_stable_patch_boundary_metrics(
                    load_metrics_snapshot,
                    finalize_metrics,
                    stage_name="finalize",
                )
                metrics.update(dict(finalize_metrics))
        metrics.setdefault("frameKind", frame.frame_kind)
        metrics.setdefault("modelId", frame.model_id)
        metrics.setdefault("layerStart", frame.layer_start)
        metrics.setdefault("layerEnd", frame.layer_end)
        metrics.setdefault("sequence", frame.sequence)
        return CaiOwnedShardAdapterResult(
            output_payload=result.output_payload,
            metrics=metrics,
            output_metadata=result.output_metadata,
        )
    if not hasattr(adapter, "process") or not callable(getattr(adapter, "process")):
        raise ValueError("CAI-owned shard adapter does not implement process.")
    if config.require_production_llm_handoff:
        raise ValueError(
            "CAI-owned production LLM handoff requires the LLM shard adapter "
            "interface."
        )
    backend_health = _probe_llm_shard_adapter_health(adapter)
    if _llm_shard_adapter_health_blocks_self_test(backend_health):
        raise CaiOwnedLlmShardBackendHealthError(backend_health)
    result = _coerce_adapter_result(adapter.process(work_item, payload))  # type: ignore[attr-defined]
    health_metrics = _llm_shard_backend_health_metrics(backend_health)
    if health_metrics:
        metrics = dict(health_metrics)
        metrics.update(result.metrics)
        return CaiOwnedShardAdapterResult(
            output_payload=result.output_payload,
            metrics=metrics,
            output_metadata=result.output_metadata,
        )
    return result


def _forward_output_if_requested(
    config: CaiOwnedShardRuntimeConfig,
    work_item: Mapping[str, Any],
    adapter_result: CaiOwnedShardAdapterResult | None = None,
    *,
    metrics: Mapping[str, Any] | None = None,
    route_audit: Mapping[str, Any] | None = None,
    runtime_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    next_sink_node_id = _next_sink_node_id(work_item)
    if not next_sink_node_id:
        return None
    session_id = str(work_item.get("sessionId") or "").strip()
    batch = work_item.get("batch") if isinstance(work_item, Mapping) else None
    if not isinstance(batch, Mapping):
        return {
            "status": "failed",
            "sinkNodeId": next_sink_node_id,
            "error": "CAI-owned runtime work item batch is missing.",
        }
    batch_id = str(batch.get("batchId") or "").strip()
    metadata = _output_forward_metadata(
        work_item,
        next_sink_node_id,
        adapter_result.output_metadata if adapter_result is not None else None,
        forwarded_shard_receipts=_forwarded_shard_receipts_for_output(
            config,
            work_item,
            adapter_result,
            metrics=metrics,
            route_audit=route_audit,
            runtime_audit=runtime_audit,
        ),
    )
    try:
        envelope = build_cai_owned_transport_output_batch_envelope(
            session_id=session_id,
            source_batch_id=batch_id,
            sink_node_id=next_sink_node_id,
            phase=_next_output_phase(work_item),
            sequence=_next_output_sequence(work_item),
            metadata=metadata,
            output_payload=(
                adapter_result.output_payload if adapter_result is not None else None
            ),
            policy=config.policy,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "sinkNodeId": next_sink_node_id,
            "error": str(exc),
        }
    signing_kwargs = cai_owned_transport_peer_signing_kwargs(config.signing_material)
    if signing_kwargs:
        envelope = sign_cai_owned_transport_batch_envelope(
            envelope,
            signer_node_id=config.node_id,
            **signing_kwargs,
        )
    peer_urls = _peer_urls_for_node(config, work_item, next_sink_node_id)
    if not peer_urls:
        return {
            "status": "no_peer_urls",
            "sinkNodeId": next_sink_node_id,
            "envelope": envelope,
        }
    try:
        response = submit_cai_owned_transport_batch_envelope_to_any(
            peer_urls,
            session_id,
            envelope,
            chain_id=str(work_item.get("chainId") or work_item.get("network") or ""),
            timeout_sec=max(0.1, float(config.output_forward_timeout_sec or 5.0)),
        )
    except Exception as exc:
        return {
            "status": "failed",
            "sinkNodeId": next_sink_node_id,
            "envelope": envelope,
            "peerCaiUrls": peer_urls,
            "error": str(exc),
        }
    return {
        "status": "submitted",
        "sinkNodeId": next_sink_node_id,
        "envelope": envelope,
        "peerCaiUrls": peer_urls,
        "response": response,
    }


def _blocking_output_forward_error(output_forward: Mapping[str, Any] | None) -> str | None:
    if not isinstance(output_forward, Mapping):
        return None
    status = str(output_forward.get("status") or "").strip().lower()
    if status != "failed":
        return None
    sink = str(output_forward.get("sinkNodeId") or "").strip() or "<unknown>"
    error = str(output_forward.get("error") or "").strip()
    if error:
        return f"CAI-owned output forward to {sink} failed: {error}"
    return f"CAI-owned output forward to {sink} failed."


def _next_sink_node_id(work_item: Mapping[str, Any]) -> str | None:
    batch = work_item.get("batch") if isinstance(work_item, Mapping) else None
    metadata = batch.get("metadata") if isinstance(batch, Mapping) else None
    if isinstance(metadata, Mapping):
        route_plan = _output_route_plan(metadata)
        if route_plan:
            value = str(route_plan[0].get("sinkNodeId") or "").strip()
            if value:
                return value
        for field_name in ("nextSinkNodeId", "outputToNodeId", "nextNodeId"):
            value = str(metadata.get(field_name) or "").strip()
            if value:
                return value
    if isinstance(batch, Mapping):
        value = str(batch.get("outputToNodeId") or "").strip()
        if value:
            return value
    return None


def _output_forward_metadata(
    work_item: Mapping[str, Any],
    next_sink_node_id: str,
    adapter_output_metadata: Mapping[str, Any] | None = None,
    forwarded_shard_receipts: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    batch = work_item.get("batch") if isinstance(work_item, Mapping) else None
    metadata = batch.get("metadata") if isinstance(batch, Mapping) else {}
    output_metadata = {
        "nextStage": "forwarded_output",
        "forwardedByRuntime": True,
        "forwardedToNodeId": next_sink_node_id,
    }
    if isinstance(metadata, Mapping):
        route_plan = _output_route_plan(metadata)
        if route_plan:
            current_route = route_plan[0]
            remaining_route = route_plan[1:]
            if remaining_route:
                output_metadata["outputRoutePlan"] = remaining_route
                output_metadata["nextSinkNodeId"] = remaining_route[0]["sinkNodeId"]
            for field_name in (
                "stageId",
                "outputStageId",
                "executorNodeId",
                "layerStart",
                "layerEnd",
                "finalOutput",
            ):
                if current_route.get(field_name) is not None:
                    output_metadata[field_name] = current_route.get(field_name)
        remaining_sinks = metadata.get("remainingSinkNodeIds")
        carry_next_sink = ""
        if isinstance(remaining_sinks, Sequence) and not isinstance(
            remaining_sinks,
            str,
        ):
            cleaned_remaining = [
                str(item or "").strip()
                for item in remaining_sinks
                if str(item or "").strip()
            ]
            if cleaned_remaining:
                carry_next_sink = cleaned_remaining[0]
                if len(cleaned_remaining) > 1:
                    output_metadata["remainingSinkNodeIds"] = cleaned_remaining[1:]
        if not carry_next_sink:
            carry_next_sink = str(
                metadata.get("nextOutputSinkNodeId")
                or metadata.get("nextNextSinkNodeId")
                or metadata.get("finalSinkNodeId")
                or ""
            ).strip()
        if carry_next_sink:
            output_metadata["nextSinkNodeId"] = carry_next_sink
        carry_next_phase = str(metadata.get("nextNextOutputPhase") or "").strip()
        if carry_next_phase:
            output_metadata["nextOutputPhase"] = carry_next_phase
        if isinstance(metadata.get("peerCaiUrlsByNode"), Mapping):
            output_metadata["peerCaiUrlsByNode"] = dict(
                metadata.get("peerCaiUrlsByNode") or {}
            )
        for field_name in ("requesterNodeId", "coordinatorNodeId"):
            if metadata.get(field_name) and field_name not in output_metadata:
                output_metadata[field_name] = metadata.get(field_name)
        for field_name in ("stageId", "nextStageId", "outputStageId"):
            if metadata.get(field_name) and field_name not in output_metadata:
                output_metadata[field_name] = metadata.get(field_name)
    if isinstance(adapter_output_metadata, Mapping):
        output_metadata.update(dict(adapter_output_metadata))
    cleaned_receipts = [
        dict(receipt)
        for receipt in forwarded_shard_receipts or []
        if isinstance(receipt, Mapping)
    ]
    if cleaned_receipts:
        output_metadata["upstreamShardReceipts"] = cleaned_receipts
        output_metadata["upstreamShardReceiptCount"] = len(cleaned_receipts)
    return output_metadata


def _forwarded_shard_receipts_for_output(
    config: CaiOwnedShardRuntimeConfig,
    work_item: Mapping[str, Any],
    adapter_result: CaiOwnedShardAdapterResult | None,
    *,
    metrics: Mapping[str, Any] | None = None,
    route_audit: Mapping[str, Any] | None = None,
    runtime_audit: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    batch = work_item.get("batch") if isinstance(work_item, Mapping) else None
    metadata = batch.get("metadata") if isinstance(batch, Mapping) else None
    receipts = _metadata_upstream_shard_receipts(metadata)
    current_receipt = _current_work_item_shard_receipt(
        config,
        work_item,
        adapter_result,
        metrics=metrics,
        route_audit=route_audit,
        runtime_audit=runtime_audit,
    )
    if current_receipt is not None:
        receipts.append(current_receipt)
    return _merge_forwarded_shard_receipts(
        receipts,
        current_node_id=str(config.node_id or "").strip(),
        signing_material=config.signing_material,
    )


def _metadata_upstream_shard_receipts(metadata: object) -> list[dict[str, Any]]:
    if not isinstance(metadata, Mapping):
        return []
    raw_receipts = None
    for field_name in (
        "upstreamShardReceipts",
        "caiOwnedShardReceipts",
        "shardReceipts",
    ):
        value = metadata.get(field_name)
        if value is not None:
            raw_receipts = value
            break
    if raw_receipts is None:
        return []
    if isinstance(raw_receipts, Mapping):
        return [dict(raw_receipts)]
    if isinstance(raw_receipts, (str, bytes)) or not isinstance(
        raw_receipts,
        Sequence,
    ):
        return []
    return [dict(item) for item in raw_receipts if isinstance(item, Mapping)]


def _current_work_item_shard_receipt(
    config: CaiOwnedShardRuntimeConfig,
    work_item: Mapping[str, Any],
    adapter_result: CaiOwnedShardAdapterResult | None,
    *,
    metrics: Mapping[str, Any] | None = None,
    route_audit: Mapping[str, Any] | None = None,
    runtime_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    batch = work_item.get("batch") if isinstance(work_item, Mapping) else None
    if not isinstance(batch, Mapping):
        return None
    batch_id = str(batch.get("batchId") or "").strip()
    node_id = str(config.node_id or "").strip()
    if not batch_id or not node_id:
        return None
    metadata = batch.get("metadata") if isinstance(batch.get("metadata"), Mapping) else {}
    phase = str(batch.get("phase") or "").strip()
    sequence = _optional_int(batch.get("sequence"))
    if sequence is None:
        sequence = _optional_int(metadata.get("sequence"))
    input_payload_sha256_hex = str(
        batch.get("payloadSha256Hex")
        or work_item.get("payloadSha256Hex")
        or metadata.get("payloadSha256Hex")
        or ""
    ).strip().lower()
    output_payload = (
        adapter_result.output_payload if adapter_result is not None else b""
    )
    output_payload_sha256_hex = hashlib.sha256(bytes(output_payload or b"")).hexdigest()
    hash_chain_sha256_hex: str | None = None
    previous_batch_id = (
        str(batch.get("previousBatchId") or "").strip()
        or str(metadata.get("previousBatchId") or "").strip()
        or None
    )
    if input_payload_sha256_hex and output_payload_sha256_hex:
        try:
            hash_chain = build_cai_owned_transport_batch_hash_chain(
                session_id=str(work_item.get("sessionId") or "").strip(),
                batch_id=batch_id,
                input_payload_sha256_hex=input_payload_sha256_hex,
                output_payload_sha256_hex=output_payload_sha256_hex,
                sequence=sequence or 0,
                previous_batch_id=previous_batch_id,
            )
            hash_chain_sha256_hex = str(
                hash_chain.get("hashChainSha256Hex") or ""
            ).strip() or None
        except Exception:
            hash_chain_sha256_hex = None
    receipt_metrics = dict(metrics or {})
    receipt_metrics.setdefault("processedBatchCount", 1)
    try:
        receipt_metrics.setdefault(
            "payloadSizeBytes",
            max(0, int(batch.get("payloadSizeBytes") or 0)),
        )
    except (TypeError, ValueError):
        pass
    receipt_metrics.setdefault("outputPayloadSizeBytes", len(output_payload))
    receipt: dict[str, Any] = {
        "nodeId": node_id,
        "network": str(work_item.get("chainId") or work_item.get("network") or ""),
        "chainId": str(work_item.get("chainId") or work_item.get("network") or ""),
        "status": "completed",
        "activationBatchCount": 1 if phase == "prefill_activation_batches" else 0,
        "decodeBatchCount": 1 if phase == "decode_activation_batches" else 0,
        "layerStart": _optional_int(metadata.get("layerStart")),
        "layerEnd": _optional_int(metadata.get("layerEnd")),
        "metrics": receipt_metrics,
        "batchIds": [batch_id],
        "stageIds": [
            stage_id
            for stage_id in [str(metadata.get("stageId") or "").strip()]
            if stage_id
        ],
        "sequences": [sequence] if sequence is not None else [],
        "inputPayloadSha256Hexes": (
            [input_payload_sha256_hex] if input_payload_sha256_hex else []
        ),
        "outputPayloadSha256Hexes": [output_payload_sha256_hex],
        "hashChainSha256Hexes": (
            [hash_chain_sha256_hex] if hash_chain_sha256_hex else []
        ),
        "routeAudits": [dict(route_audit)] if isinstance(route_audit, Mapping) else [],
        "runtimeAudits": (
            [dict(runtime_audit)] if isinstance(runtime_audit, Mapping) else []
        ),
    }
    return receipt


def _merge_forwarded_shard_receipts(
    receipts: Sequence[Mapping[str, Any]],
    *,
    current_node_id: str,
    signing_material: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    merged_by_node: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for receipt in receipts:
        node_id = str(receipt.get("nodeId") or "").strip()
        if not node_id:
            continue
        if node_id not in merged_by_node:
            merged_by_node[node_id] = (
                _unsigned_receipt_copy(receipt)
                if node_id == current_node_id
                else dict(receipt)
            )
            order.append(node_id)
            continue
        _merge_shard_receipt(merged_by_node[node_id], receipt)

    signing_kwargs = cai_owned_transport_peer_signing_kwargs(signing_material)
    if signing_kwargs and current_node_id in merged_by_node:
        merged_by_node[current_node_id] = sign_cai_owned_transport_shard_receipt(
            _unsigned_receipt_copy(merged_by_node[current_node_id]),
            signer_node_id=current_node_id,
            **signing_kwargs,
        )
    return [merged_by_node[node_id] for node_id in order]


def _unsigned_receipt_copy(receipt: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(receipt)
    copied.pop("signature", None)
    return copied


def _merge_shard_receipt(target: dict[str, Any], receipt: Mapping[str, Any]) -> None:
    target.pop("signature", None)
    for field_name in ("activationBatchCount", "decodeBatchCount"):
        target[field_name] = _positive_int(target.get(field_name), 0) + _positive_int(
            receipt.get(field_name),
            0,
        )
    for field_name in (
        "batchIds",
        "stageIds",
        "sequences",
        "inputPayloadSha256Hexes",
        "outputPayloadSha256Hexes",
        "hashChainSha256Hexes",
        "routeAudits",
        "runtimeAudits",
    ):
        target_values = target.setdefault(field_name, [])
        if not isinstance(target_values, list):
            target_values = []
            target[field_name] = target_values
        _append_unique_receipt_values(target_values, receipt.get(field_name))
    for field_name, reducer in (("layerStart", min), ("layerEnd", max)):
        incoming = _optional_int(receipt.get(field_name))
        existing = _optional_int(target.get(field_name))
        if incoming is None:
            continue
        target[field_name] = incoming if existing is None else reducer(existing, incoming)
    target_metrics = target.setdefault("metrics", {})
    if not isinstance(target_metrics, dict):
        target_metrics = {}
        target["metrics"] = target_metrics
    incoming_metrics = receipt.get("metrics")
    if isinstance(incoming_metrics, Mapping):
        _merge_receipt_metrics(target_metrics, incoming_metrics)


def _append_unique_receipt_values(target: list[Any], values: object) -> None:
    if values is None:
        return
    candidates: Sequence[Any]
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        candidates = [values]
    else:
        candidates = values
    for item in candidates:
        if isinstance(item, Mapping):
            candidate: Any = dict(item)
            marker = json.dumps(candidate, sort_keys=True, default=str)
        else:
            candidate = item
            marker = str(item)
        existing_markers = {
            json.dumps(value, sort_keys=True, default=str)
            if isinstance(value, Mapping)
            else str(value)
            for value in target
        }
        if marker not in existing_markers:
            target.append(candidate)


def _merge_receipt_metrics(
    target: dict[str, Any],
    incoming: Mapping[str, Any],
) -> None:
    additive_fields = {
        "processedBatchCount",
        "payloadSizeBytes",
        "outputPayloadSizeBytes",
        "promptTokenCount",
        "completionTokenCount",
        "inputTokenCount",
        "outputTokenCount",
        "tokenCount",
    }
    for key, value in incoming.items():
        if key in additive_fields:
            target[key] = _positive_int(target.get(key), 0) + _positive_int(value, 0)
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            target_values = target.setdefault(key, [])
            if not isinstance(target_values, list):
                target_values = []
                target[key] = target_values
            _append_unique_receipt_values(target_values, value)
            continue
        target.setdefault(key, value)


def _next_output_phase(work_item: Mapping[str, Any]) -> str | None:
    batch = work_item.get("batch") if isinstance(work_item, Mapping) else None
    metadata = batch.get("metadata") if isinstance(batch, Mapping) else None
    if isinstance(metadata, Mapping):
        route_plan = _output_route_plan(metadata)
        if route_plan:
            phase = str(route_plan[0].get("phase") or "").strip()
            if phase:
                return phase
        phase = str(metadata.get("nextOutputPhase") or metadata.get("outputPhase") or "").strip()
        if phase:
            return phase
    return None


def _next_output_sequence(work_item: Mapping[str, Any]) -> int | None:
    batch = work_item.get("batch") if isinstance(work_item, Mapping) else None
    metadata = batch.get("metadata") if isinstance(batch, Mapping) else None
    if isinstance(metadata, Mapping):
        route_plan = _output_route_plan(metadata)
        if route_plan:
            sequence = _optional_int(route_plan[0].get("sequence"))
            if sequence is not None:
                return sequence
        sequence = _optional_int(metadata.get("nextOutputSequence"))
        if sequence is not None:
            return sequence
    if isinstance(batch, Mapping):
        sequence = _optional_int(batch.get("sequence"))
        if sequence is not None:
            return sequence + 1
    return None


def _output_route_plan(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    plan = metadata.get("outputRoutePlan")
    if not isinstance(plan, Sequence) or isinstance(plan, (str, bytes)):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in plan:
        if not isinstance(item, Mapping):
            continue
        sink = str(
            item.get("sinkNodeId")
            or item.get("outputToNodeId")
            or item.get("nodeId")
            or ""
        ).strip()
        if not sink:
            continue
        entry = dict(item)
        entry["sinkNodeId"] = sink
        sequence = _optional_int(entry.get("sequence"))
        if sequence is not None:
            entry["sequence"] = sequence
        elif "sequence" in entry:
            entry.pop("sequence", None)
        phase = str(entry.get("phase") or "").strip()
        if phase:
            entry["phase"] = phase
        elif "phase" in entry:
            entry.pop("phase", None)
        cleaned.append(entry)
    return cleaned


def _peer_urls_for_node(
    config: CaiOwnedShardRuntimeConfig,
    work_item: Mapping[str, Any],
    node_id: str,
) -> list[str]:
    urls: list[str] = []
    batch = work_item.get("batch") if isinstance(work_item, Mapping) else None
    metadata = batch.get("metadata") if isinstance(batch, Mapping) else None
    if isinstance(metadata, Mapping):
        explicit_urls = metadata.get("nextPeerCaiUrls") or metadata.get("peerCaiUrls")
        if isinstance(explicit_urls, Sequence) and not isinstance(explicit_urls, str):
            urls.extend(str(item or "").strip() for item in explicit_urls)
        urls_by_node = metadata.get("peerCaiUrlsByNode")
        if isinstance(urls_by_node, Mapping):
            mapped_urls = urls_by_node.get(node_id)
            if isinstance(mapped_urls, Sequence) and not isinstance(mapped_urls, str):
                urls.extend(str(item or "").strip() for item in mapped_urls)
            elif mapped_urls:
                urls.append(str(mapped_urls).strip())
    configured_urls = config.output_peer_cai_urls_by_node.get(node_id)
    if isinstance(configured_urls, Sequence) and not isinstance(configured_urls, str):
        urls.extend(str(item or "").strip() for item in configured_urls)
    elif configured_urls:
        urls.append(str(configured_urls).strip())
    cleaned: list[str] = []
    seen: set[str] = set()
    for url in urls:
        clean_url = str(url or "").strip()
        if not clean_url or clean_url in seen:
            continue
        seen.add(clean_url)
        cleaned.append(clean_url)
    if str(node_id or "").strip() != str(config.node_id or "").strip():
        routable = [
            url for url in cleaned if not _is_loopback_peer_cai_url(url)
        ]
        if routable:
            return routable
    return cleaned


def _coordinator_url_for_work_item(
    config: CaiOwnedShardRuntimeConfig,
    work_item: Mapping[str, Any],
) -> str | None:
    batch = work_item.get("batch") if isinstance(work_item, Mapping) else None
    metadata = batch.get("metadata") if isinstance(batch, Mapping) else None
    coordinator_node_id = ""
    if isinstance(metadata, Mapping):
        coordinator_node_id = str(
            metadata.get("coordinatorNodeId")
            or metadata.get("requesterNodeId")
            or metadata.get("sourceNodeId")
            or ""
        ).strip()
    if not coordinator_node_id:
        coordinator_node_id = str(work_item.get("sourceNodeId") or "").strip()
    if not coordinator_node_id:
        return None
    peer_urls = _peer_urls_for_node(config, work_item, coordinator_node_id)
    if coordinator_node_id != str(config.node_id or "").strip():
        routable_urls = [
            url for url in peer_urls if not _is_loopback_peer_cai_url(url)
        ]
        if routable_urls:
            return routable_urls[0]
    return peer_urls[0] if peer_urls else None


def _is_loopback_peer_cai_url(value: object) -> bool:
    raw = str(value or "").strip()
    if raw.startswith("cai-overlay:"):
        raw = raw[len("cai-overlay:") :].strip()
    try:
        parsed = urlsplit(raw)
    except Exception:
        return False
    host = (parsed.hostname or "").strip().lower()
    return host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def _has_llm_adapter_interface(adapter: object) -> bool:
    return all(
        callable(getattr(adapter, method_name, None))
        for method_name in (
            "load_shard",
            "process_prefill",
            "process_decode",
            "finalize",
        )
    )


def _work_item_frame(
    work_item: Mapping[str, Any],
    payload: bytes,
    *,
    require_production_llm_handoff: bool = False,
) -> CaiOwnedShardFrame:
    batch = work_item.get("batch") if isinstance(work_item, Mapping) else None
    if not isinstance(batch, Mapping):
        raise ValueError("CAI-owned runtime work item batch is missing.")
    metadata = batch.get("metadata") if isinstance(batch.get("metadata"), Mapping) else {}
    metadata_dict = dict(metadata or {})
    expected_model_id = str(work_item.get("modelId") or "").strip() or None
    if metadata_dict.get("frameSchemaVersion") is not None:
        valid, error = validate_cai_owned_transport_frame_metadata(
            metadata_dict,
            expected_model_id=expected_model_id,
            require_llm_handoff=require_production_llm_handoff,
        )
        if not valid:
            raise ValueError(error or "CAI-owned transport frame metadata is invalid.")
    elif require_production_llm_handoff:
        raise ValueError("CAI-owned transport frame metadata is missing.")
    payload_sha256_hex = hashlib.sha256(bytes(payload or b"")).hexdigest()
    metadata_payload_hash = str(
        metadata_dict.get("payloadSha256Hex") or ""
    ).strip().lower()
    if metadata_payload_hash and metadata_payload_hash != payload_sha256_hex:
        raise ValueError("CAI-owned transport frame payload hash does not match.")
    phase = str(batch.get("phase") or "").strip()
    model_id = str(metadata_dict.get("modelId") or expected_model_id or "").strip()
    return CaiOwnedShardFrame(
        session_id=str(work_item.get("sessionId") or "").strip(),
        batch_id=str(batch.get("batchId") or "").strip(),
        phase=phase,
        source_node_id=str(batch.get("sourceNodeId") or "").strip(),
        sink_node_id=str(batch.get("sinkNodeId") or "").strip(),
        sequence=_optional_int(batch.get("sequence"), default=0),
        model_id=model_id or None,
        frame_kind=str(metadata_dict.get("frameKind") or _frame_kind_for_phase(phase)),
        layer_start=_optional_int(metadata_dict.get("layerStart")),
        layer_end=_optional_int(metadata_dict.get("layerEnd")),
        token_start=_optional_int(metadata_dict.get("tokenStart")),
        token_end=_optional_int(metadata_dict.get("tokenEnd")),
        payload_sha256_hex=payload_sha256_hex,
        payload=bytes(payload or b""),
        metadata=metadata_dict,
    )


def _frame_kind_for_phase(phase: str) -> str:
    if phase == "decode_activation_batches":
        return "decode"
    if phase == "prefill_activation_batches":
        return "activation"
    return "bytes_test"


def _frame_request_payload(frame: CaiOwnedShardFrame) -> dict[str, Any]:
    return {
        "sessionId": frame.session_id,
        "batchId": frame.batch_id,
        "phase": frame.phase,
        "sourceNodeId": frame.source_node_id,
        "sinkNodeId": frame.sink_node_id,
        "sequence": frame.sequence,
        "modelId": frame.model_id,
        "frameKind": frame.frame_kind,
        "layerStart": frame.layer_start,
        "layerEnd": frame.layer_end,
        "tokenStart": frame.token_start,
        "tokenEnd": frame.token_end,
        "payloadSha256Hex": frame.payload_sha256_hex,
        "metadata": dict(frame.metadata or {}),
    }


def _next_frame_template(metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(metadata, Mapping):
        return None
    template = metadata.get("nextFrameTemplate")
    if not isinstance(template, Mapping):
        return None
    return _json_clone_mapping(template)


def _external_adapter_output_contract(frame: CaiOwnedShardFrame) -> dict[str, Any]:
    next_template = _next_frame_template(frame.metadata)
    contract: dict[str, Any] = {
        "schemaVersion": 1,
        "requiresOutputFrameMetadata": next_template is not None,
        "requiresFinalOutput": next_template is None,
        "outputPayloadHashSource": "computed_output_payload",
    }
    if next_template is not None:
        contract["frameMetadataTemplate"] = _frame_template_with_output_hash(
            next_template,
            "<computed-output-sha256>",
        )
    return contract


def _external_adapter_production_requirements(
    required_capabilities: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "handoffAbi": LLAMA_CPP_EXTERNAL_SHARD_HANDOFF_ABI,
        "shardSpecAbi": LLAMA_CPP_EXTERNAL_SHARD_SPEC_ABI,
        "patchBoundaryAbi": LLAMA_CPP_EXTERNAL_SHARD_PATCH_BOUNDARY_ABI,
        "productionStateContractAbi": (
            LLAMA_CPP_EXTERNAL_SHARD_PRODUCTION_STATE_CONTRACT_ABI
        ),
        "activationBoundary": LLAMA_CPP_EXTERNAL_SHARD_ACTIVATION_BOUNDARY,
        "decodeStateBoundary": LLAMA_CPP_EXTERNAL_SHARD_DECODE_STATE_BOUNDARY,
        "supportedTensorEncodings": list(
            LLAMA_CPP_EXTERNAL_SHARD_SUPPORTED_TENSOR_ENCODINGS,
        ),
        "requiredCapabilities": _clean_string_sequence(
            required_capabilities or LLAMA_CPP_EXTERNAL_SHARD_REQUIRED_CAPABILITIES,
        ),
        "requiredProductionCapabilities": list(
            LLAMA_CPP_EXTERNAL_SHARD_PRODUCTION_CAPABILITIES,
        ),
        "requiresRealStateContract": True,
        "requiresShardOnlyLoading": True,
        "forbidFullModelFallback": True,
    }


def _frame_template_with_output_hash(
    template: Mapping[str, Any],
    output_payload_sha256_hex: str,
) -> dict[str, Any]:
    output = _json_clone_mapping(template)
    output["payloadSha256Hex"] = output_payload_sha256_hex
    handoff = output.get("llmHandoff")
    if isinstance(handoff, Mapping):
        handoff_dict = dict(handoff)
        tensor = handoff_dict.get("tensor")
        if isinstance(tensor, Mapping):
            tensor_dict = dict(tensor)
            tensor_dict["sha256Hex"] = output_payload_sha256_hex
            handoff_dict["tensor"] = tensor_dict
        output["llmHandoff"] = handoff_dict
    return output


def _json_clone_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), sort_keys=True))


def _build_external_shard_spec_extra_metadata(
    handoff_extra_metadata: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(handoff_extra_metadata, Mapping):
        return None
    output: dict[str, Any] = {}
    for key in ("preferredFilename", "family", "quantization"):
        value = handoff_extra_metadata.get(key)
        if isinstance(value, str):
            clean = value.strip()
            if clean:
                output[key] = clean
    context_length = _optional_int(handoff_extra_metadata.get("contextLength"))
    if context_length is not None and context_length > 0:
        output["contextLength"] = context_length
    return output or None


def _normalize_llama_cpp_external_shard_artifact_hint(
    value: Any,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("External llama.cpp shard artifact hint is invalid.")
    output: dict[str, Any] = {}
    for field_name in (
        "catalogId",
        "version",
        "artifactId",
        "preferredFilename",
        "modelArtifactPath",
        "assignmentArtifactPath",
        "assignmentArtifactDigest",
    ):
        raw = value.get(field_name)
        if raw is None:
            continue
        clean = str(raw).strip()
        if clean:
            output[field_name] = clean
    model_sha = value.get("modelArtifactSha256Hex")
    if model_sha is not None:
        output["modelArtifactSha256Hex"] = _normalize_sha256_hex(
            model_sha,
            field_name="External llama.cpp shard artifact hint modelArtifactSha256Hex",
        )
    for field_name in ("modelArtifactSizeBytes", "assignmentArtifactSizeBytes"):
        raw = value.get(field_name)
        if raw is None:
            continue
        clean_int = _optional_int(raw)
        if clean_int is None or clean_int <= 0:
            raise ValueError(
                f"External llama.cpp shard artifact hint {field_name} is invalid."
            )
        output[field_name] = clean_int
    return output or None


def _expected_frame_field(
    expected_frame: CaiOwnedShardFrame | Mapping[str, Any],
    attr_name: str,
    field_name: str,
) -> int | None:
    if isinstance(expected_frame, CaiOwnedShardFrame):
        return _optional_int(getattr(expected_frame, attr_name))
    if isinstance(expected_frame, Mapping):
        return _optional_int(expected_frame.get(field_name))
    return None


def build_llama_cpp_external_shard_patch_boundary(
    *,
    backend: str = "llama.cpp-patched",
    backend_version: str | None = None,
    patch_id: str = "cai-llama-cpp-shard",
    runner_protocol_version: str = "0.1",
    model_format: str = "gguf",
    activation_boundary: str = LLAMA_CPP_EXTERNAL_SHARD_ACTIVATION_BOUNDARY,
    decode_state_boundary: str = LLAMA_CPP_EXTERNAL_SHARD_DECODE_STATE_BOUNDARY,
    supported_tensor_encodings: Sequence[str] | None = None,
    capabilities: Sequence[str] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    boundary: dict[str, Any] = {
        "schemaVersion": LLAMA_CPP_EXTERNAL_SHARD_PATCH_BOUNDARY_SCHEMA_VERSION,
        "abi": LLAMA_CPP_EXTERNAL_SHARD_PATCH_BOUNDARY_ABI,
        "backend": str(backend or "").strip(),
        "backendVersion": str(backend_version or "").strip() or None,
        "patchId": str(patch_id or "").strip(),
        "runnerProtocolVersion": str(runner_protocol_version or "").strip(),
        "modelFormat": str(model_format or "").strip(),
        "requiresPatchedBackend": True,
        "activationBoundary": str(activation_boundary or "").strip(),
        "decodeStateBoundary": str(decode_state_boundary or "").strip(),
        "supportedTensorEncodings": _clean_string_sequence(
            supported_tensor_encodings
            or LLAMA_CPP_EXTERNAL_SHARD_SUPPORTED_TENSOR_ENCODINGS,
        ),
        "capabilities": _clean_string_sequence(
            capabilities or LLAMA_CPP_EXTERNAL_SHARD_REQUIRED_CAPABILITIES,
        ),
    }
    if extra_metadata is not None:
        boundary["extraMetadata"] = dict(extra_metadata)
    valid, error = validate_llama_cpp_external_shard_patch_boundary(boundary)
    if not valid:
        raise ValueError(error or "llama.cpp external shard patch boundary is invalid.")
    return boundary


def build_llama_cpp_external_shard_production_state_contract(
    *,
    activation_state_format: str = "ggml-tensor-v1/layer-range-activation-v1",
    decode_state_format: str = "ggml-kv-cache-v1/token-step-kv-cache-v1",
    model_execution_backend: str = "llama.cpp-cai-shard",
    tensor_encoding: str = "ggml-tensor-v1",
    shard_execution_mode: str = LLAMA_CPP_EXTERNAL_SHARD_PRODUCTION_EXECUTION_MODE,
    full_model_replica_required: bool = False,
    activation_state_is_synthetic: bool = False,
    decode_state_is_synthetic: bool = False,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "schemaVersion": (
            LLAMA_CPP_EXTERNAL_SHARD_PRODUCTION_STATE_CONTRACT_SCHEMA_VERSION
        ),
        "abi": LLAMA_CPP_EXTERNAL_SHARD_PRODUCTION_STATE_CONTRACT_ABI,
        "activationStateFormat": str(activation_state_format or "").strip(),
        "decodeStateFormat": str(decode_state_format or "").strip(),
        "modelExecutionBackend": str(model_execution_backend or "").strip(),
        "tensorEncoding": str(tensor_encoding or "").strip(),
        "shardExecutionMode": str(shard_execution_mode or "").strip(),
        "fullModelReplicaRequired": bool(full_model_replica_required),
        "activationStateIsSynthetic": bool(activation_state_is_synthetic),
        "decodeStateIsSynthetic": bool(decode_state_is_synthetic),
    }
    if extra_metadata is not None:
        contract["extraMetadata"] = dict(extra_metadata)
    valid, error = validate_llama_cpp_external_shard_production_state_contract(
        contract,
    )
    if not valid:
        raise ValueError(
            error or "llama.cpp production state contract is invalid."
        )
    return contract


def validate_llama_cpp_external_shard_production_state_contract(
    contract: Any,
) -> tuple[bool, str | None]:
    if not isinstance(contract, Mapping):
        return False, "LLM shard adapter production state contract is missing."
    try:
        schema_version = int(contract.get("schemaVersion") or 0)
    except (TypeError, ValueError):
        return False, "LLM shard adapter production state contract schema is invalid."
    if (
        schema_version
        != LLAMA_CPP_EXTERNAL_SHARD_PRODUCTION_STATE_CONTRACT_SCHEMA_VERSION
    ):
        return (
            False,
            "LLM shard adapter production state contract schema is unsupported.",
        )
    if (
        str(contract.get("abi") or "").strip()
        != LLAMA_CPP_EXTERNAL_SHARD_PRODUCTION_STATE_CONTRACT_ABI
    ):
        return False, "LLM shard adapter production state contract ABI is unsupported."
    missing = [
        field_name
        for field_name in LLAMA_CPP_EXTERNAL_SHARD_PRODUCTION_STATE_CONTRACT_FIELDS
        if not str(contract.get(field_name) or "").strip()
    ]
    if missing:
        return (
            False,
            "LLM shard adapter production state contract missing fields: "
            + ", ".join(missing),
        )
    tensor_encoding = str(contract.get("tensorEncoding") or "").strip()
    if tensor_encoding not in LLAMA_CPP_EXTERNAL_SHARD_SUPPORTED_TENSOR_ENCODINGS:
        return False, "LLM shard adapter production tensor encoding is unsupported."
    activation_state_format = str(contract.get("activationStateFormat") or "").strip()
    if not activation_state_format.endswith(
        f"/{LLAMA_CPP_EXTERNAL_SHARD_ACTIVATION_BOUNDARY}"
    ):
        return (
            False,
            "LLM shard adapter activation state format does not match layer-range boundary.",
        )
    decode_state_format = str(contract.get("decodeStateFormat") or "").strip()
    if not decode_state_format.endswith(
        f"/{LLAMA_CPP_EXTERNAL_SHARD_DECODE_STATE_BOUNDARY}"
    ):
        return (
            False,
            "LLM shard adapter decode state format does not match token-step boundary.",
        )
    shard_execution_mode = str(contract.get("shardExecutionMode") or "").strip()
    if shard_execution_mode != LLAMA_CPP_EXTERNAL_SHARD_PRODUCTION_EXECUTION_MODE:
        return (
            False,
            "LLM shard adapter production shard execution mode must be layer_range.",
        )
    full_model_replica_required = contract.get("fullModelReplicaRequired")
    if not isinstance(full_model_replica_required, bool):
        return (
            False,
            "LLM shard adapter production fullModelReplicaRequired flag is invalid.",
        )
    if full_model_replica_required:
        return (
            False,
            "LLM shard adapter production backend requires a full model replica.",
        )
    if bool(contract.get("activationStateIsSynthetic")):
        return False, "LLM shard adapter activation state is synthetic."
    if bool(contract.get("decodeStateIsSynthetic")):
        return False, "LLM shard adapter decode state is synthetic."
    if contract.get("extraMetadata") is not None and not isinstance(
        contract.get("extraMetadata"),
        Mapping,
    ):
        return (
            False,
            "LLM shard adapter production state contract extraMetadata is invalid.",
        )
    return True, None


def validate_llama_cpp_external_shard_patch_boundary(
    boundary: Any,
    *,
    expected_backend: str | None = None,
    required_capabilities: Sequence[str] | None = None,
) -> tuple[bool, str | None]:
    if not isinstance(boundary, Mapping):
        return False, "External llama.cpp shard adapter patch boundary is missing."
    try:
        schema_version = int(boundary.get("schemaVersion") or 0)
    except (TypeError, ValueError):
        return False, "External llama.cpp shard adapter patch boundary schema is invalid."
    if schema_version != LLAMA_CPP_EXTERNAL_SHARD_PATCH_BOUNDARY_SCHEMA_VERSION:
        return False, "External llama.cpp shard adapter patch boundary schema is unsupported."
    if str(boundary.get("abi") or "").strip() != LLAMA_CPP_EXTERNAL_SHARD_PATCH_BOUNDARY_ABI:
        return False, "External llama.cpp shard adapter patch boundary ABI is unsupported."
    if not bool(boundary.get("requiresPatchedBackend")):
        return False, "External llama.cpp shard adapter must require a patched backend."

    backend = str(boundary.get("backend") or "").strip()
    if not backend or not backend.isascii():
        return False, "External llama.cpp shard adapter patch boundary backend is invalid."
    expected = str(expected_backend or "").strip()
    if expected and backend != expected:
        return False, "External llama.cpp shard adapter patch boundary backend does not match."

    for field_name in ("patchId", "runnerProtocolVersion", "modelFormat"):
        value = str(boundary.get(field_name) or "").strip()
        if not value or not value.isascii():
            return False, (
                "External llama.cpp shard adapter patch boundary "
                f"{field_name} is invalid."
            )
    if (
        str(boundary.get("activationBoundary") or "").strip()
        != LLAMA_CPP_EXTERNAL_SHARD_ACTIVATION_BOUNDARY
    ):
        return False, "External llama.cpp shard adapter activation boundary is unsupported."
    if (
        str(boundary.get("decodeStateBoundary") or "").strip()
        != LLAMA_CPP_EXTERNAL_SHARD_DECODE_STATE_BOUNDARY
    ):
        return False, "External llama.cpp shard adapter decode state boundary is unsupported."

    encodings = _clean_string_sequence(boundary.get("supportedTensorEncodings"))
    if not encodings:
        return False, "External llama.cpp shard adapter tensor encodings are missing."
    unsupported_encodings = [
        item
        for item in encodings
        if item not in LLAMA_CPP_EXTERNAL_SHARD_SUPPORTED_TENSOR_ENCODINGS
    ]
    if unsupported_encodings:
        return False, (
            "External llama.cpp shard adapter tensor encoding is unsupported: "
            + ", ".join(unsupported_encodings)
        )

    available = set(_clean_string_sequence(boundary.get("capabilities")))
    required = set(
        _clean_string_sequence(
            required_capabilities or LLAMA_CPP_EXTERNAL_SHARD_REQUIRED_CAPABILITIES,
        )
    )
    missing = sorted(required - available)
    if missing:
        return False, (
            "External llama.cpp shard adapter patch boundary missing capabilities: "
            + ", ".join(missing)
        )
    if boundary.get("extraMetadata") is not None and not isinstance(
        boundary.get("extraMetadata"),
        Mapping,
    ):
        return False, "External llama.cpp shard adapter patch boundary extraMetadata is invalid."
    return True, None


def build_llama_cpp_external_shard_spec(
    frame: CaiOwnedShardFrame,
    *,
    artifact_hint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = frame.metadata if isinstance(frame.metadata, Mapping) else {}
    handoff = metadata.get("llmHandoff") if isinstance(metadata, Mapping) else None
    if not isinstance(handoff, Mapping):
        raise ValueError("CAI-owned LLM handoff metadata is missing.")
    valid, error = validate_cai_owned_llm_handoff_metadata(
        handoff,
        expected_model_id=frame.model_id,
        expected_frame_metadata=metadata,
    )
    if not valid:
        raise ValueError(error or "CAI-owned LLM handoff metadata is invalid.")
    tensor = handoff.get("tensor")
    if not isinstance(tensor, Mapping):
        raise ValueError("CAI-owned LLM handoff tensor is missing.")
    layer_start = _coalesce_optional_int(frame.layer_start, handoff.get("layerStart"))
    layer_end = _coalesce_optional_int(frame.layer_end, handoff.get("layerEnd"))
    token_start = _coalesce_optional_int(frame.token_start, handoff.get("tokenStart"))
    token_end = _coalesce_optional_int(frame.token_end, handoff.get("tokenEnd"))
    if layer_start is None or layer_end is None:
        raise ValueError("External llama.cpp shard spec layer range is missing.")
    if token_start is None or token_end is None:
        raise ValueError("External llama.cpp shard spec token window is missing.")
    handoff_extra = handoff.get("extraMetadata")
    handoff_extra_mapping = (
        handoff_extra if isinstance(handoff_extra, Mapping) else None
    )
    spec: dict[str, Any] = {
        "schemaVersion": LLAMA_CPP_EXTERNAL_SHARD_SPEC_SCHEMA_VERSION,
        "abi": LLAMA_CPP_EXTERNAL_SHARD_SPEC_ABI,
        "modelId": str(frame.model_id or handoff.get("modelId") or "").strip(),
        "modelFormat": "gguf",
        "requiresPatchedBackend": True,
        "backend": str(handoff.get("backend") or "").strip(),
        "backendVersion": str(handoff.get("backendVersion") or "").strip() or None,
        "shardExecutionMode": LLAMA_CPP_EXTERNAL_SHARD_PRODUCTION_EXECUTION_MODE,
        "frameKind": str(frame.frame_kind or metadata.get("frameKind") or "").strip(),
        "phase": str(frame.phase or "").strip(),
        "layerStart": layer_start,
        "layerEnd": layer_end,
        "tokenStart": token_start,
        "tokenEnd": token_end,
        "tensor": _json_clone_mapping(tensor),
    }
    total_layers = (
        _optional_int(handoff_extra_mapping.get("totalLayerCount"))
        if handoff_extra_mapping is not None
        else None
    )
    if total_layers is not None:
        spec["totalLayerCount"] = total_layers
    runtime_source = (
        str(handoff_extra_mapping.get("runtimeMetadataSource") or "").strip()
        if handoff_extra_mapping is not None
        else ""
    )
    if runtime_source:
        spec["runtimeMetadataSource"] = runtime_source
    for field_name in ("modelSha256Hex", "tokenizerConfigHash"):
        value = handoff.get(field_name)
        if value is not None:
            spec[field_name] = str(value)
    for field_name in ("kvCache", "decodeState"):
        value = handoff.get(field_name)
        if isinstance(value, Mapping):
            spec[field_name] = _json_clone_mapping(value)
    shard_extra = _build_external_shard_spec_extra_metadata(
        handoff_extra_mapping,
    )
    if shard_extra:
        spec["extraMetadata"] = shard_extra
    normalized_artifact_hint = _normalize_llama_cpp_external_shard_artifact_hint(
        artifact_hint,
    )
    if normalized_artifact_hint:
        spec["artifactHint"] = normalized_artifact_hint
    valid, error = validate_llama_cpp_external_shard_spec(
        spec,
        expected_model_id=frame.model_id,
        expected_frame=frame,
    )
    if not valid:
        raise ValueError(error or "External llama.cpp shard spec is invalid.")
    return spec


def validate_llama_cpp_external_shard_spec(
    spec: Any,
    *,
    expected_model_id: str | None = None,
    expected_frame: CaiOwnedShardFrame | Mapping[str, Any] | None = None,
) -> tuple[bool, str | None]:
    if not isinstance(spec, Mapping):
        return False, "External llama.cpp shard spec is missing."
    try:
        schema_version = int(spec.get("schemaVersion") or 0)
    except (TypeError, ValueError):
        return False, "External llama.cpp shard spec schema is invalid."
    if schema_version != LLAMA_CPP_EXTERNAL_SHARD_SPEC_SCHEMA_VERSION:
        return False, "External llama.cpp shard spec schema is unsupported."
    if str(spec.get("abi") or "").strip() != LLAMA_CPP_EXTERNAL_SHARD_SPEC_ABI:
        return False, "External llama.cpp shard spec ABI is unsupported."
    if not bool(spec.get("requiresPatchedBackend")):
        return False, "External llama.cpp shard spec must require a patched backend."
    model_id = str(spec.get("modelId") or "").strip()
    if not model_id:
        return False, "External llama.cpp shard spec modelId is missing."
    expected = str(expected_model_id or "").strip()
    if expected and model_id != expected:
        return False, "External llama.cpp shard spec modelId does not match."
    if str(spec.get("modelFormat") or "").strip() != "gguf":
        return False, "External llama.cpp shard spec modelFormat is unsupported."
    backend = str(spec.get("backend") or "").strip()
    if not backend or not backend.isascii():
        return False, "External llama.cpp shard spec backend is invalid."
    if (
        str(spec.get("shardExecutionMode") or "").strip()
        != LLAMA_CPP_EXTERNAL_SHARD_PRODUCTION_EXECUTION_MODE
    ):
        return False, "External llama.cpp shard spec execution mode is unsupported."
    for field_name, label in (
        ("layerStart", "layer range"),
        ("layerEnd", "layer range"),
        ("tokenStart", "token window"),
        ("tokenEnd", "token window"),
    ):
        if _optional_int(spec.get(field_name)) is None:
            return False, f"External llama.cpp shard spec {label} is invalid."
    layer_start = int(spec.get("layerStart") or 0)
    layer_end = int(spec.get("layerEnd") or 0)
    token_start = int(spec.get("tokenStart") or 0)
    token_end = int(spec.get("tokenEnd") or 0)
    if layer_start < 0 or layer_end <= layer_start:
        return False, "External llama.cpp shard spec layer range is invalid."
    if token_start < 0 or token_end < token_start:
        return False, "External llama.cpp shard spec token window is invalid."
    total_layers = _optional_int(spec.get("totalLayerCount"))
    if spec.get("totalLayerCount") is not None:
        if total_layers is None or total_layers <= 0:
            return False, "External llama.cpp shard spec totalLayerCount is invalid."
        if layer_end > total_layers:
            return False, "External llama.cpp shard spec layer range exceeds totalLayerCount."
    tensor = spec.get("tensor")
    if not isinstance(tensor, Mapping):
        return False, "External llama.cpp shard spec tensor is missing."
    tensor_name = str(tensor.get("name") or "").strip()
    if not tensor_name or not tensor_name.isascii():
        return False, "External llama.cpp shard spec tensor name is invalid."
    tensor_dtype = str(tensor.get("dtype") or "").strip()
    if not tensor_dtype or not tensor_dtype.isascii():
        return False, "External llama.cpp shard spec tensor dtype is invalid."
    tensor_encoding = str(tensor.get("encoding") or "").strip()
    if tensor_encoding not in LLAMA_CPP_EXTERNAL_SHARD_SUPPORTED_TENSOR_ENCODINGS:
        return False, "External llama.cpp shard spec tensor encoding is unsupported."
    shape = tensor.get("shape")
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes, bytearray)):
        return False, "External llama.cpp shard spec tensor shape is invalid."
    clean_shape: list[int] = []
    for item in shape:
        clean_int = _optional_int(item)
        if clean_int is None or clean_int <= 0:
            return False, "External llama.cpp shard spec tensor shape is invalid."
        clean_shape.append(clean_int)
    if not clean_shape:
        return False, "External llama.cpp shard spec tensor shape is invalid."
    tensor_hash = tensor.get("sha256Hex")
    if tensor_hash is not None:
        try:
            _normalize_sha256_hex(
                tensor_hash,
                field_name="External llama.cpp shard spec tensor sha256Hex",
            )
        except ValueError as exc:
            return False, str(exc)
    for field_name in ("modelSha256Hex", "tokenizerConfigHash"):
        value = spec.get(field_name)
        if value is not None:
            try:
                _normalize_sha256_hex(
                    value,
                    field_name=f"External llama.cpp shard spec {field_name}",
                )
            except ValueError as exc:
                return False, str(exc)
    for field_name in ("kvCache", "decodeState", "extraMetadata"):
        value = spec.get(field_name)
        if value is not None and not isinstance(value, Mapping):
            return False, f"External llama.cpp shard spec {field_name} is invalid."
    try:
        _normalize_llama_cpp_external_shard_artifact_hint(spec.get("artifactHint"))
    except ValueError as exc:
        return False, str(exc)
    if expected_frame is not None:
        for attr_name, field_name in (
            ("layer_start", "layerStart"),
            ("layer_end", "layerEnd"),
            ("token_start", "tokenStart"),
            ("token_end", "tokenEnd"),
        ):
            expected_value = _expected_frame_field(expected_frame, attr_name, field_name)
            if expected_value is not None and _optional_int(spec.get(field_name)) != int(
                expected_value
            ):
                label = "layer range" if "layer" in field_name else "token window"
                return (
                    False,
                    f"External llama.cpp shard spec {label} does not match frame.",
                )
    return True, None


def _prepare_external_adapter_local_file_context(
    *,
    payload: bytes,
    endpoint_url: str | None,
    allow_remote_endpoint_url: bool,
    file_io_root: str | None,
    file_io_threshold_bytes: int | None,
) -> dict[str, Any] | None:
    threshold = (
        max(1, int(file_io_threshold_bytes))
        if file_io_threshold_bytes is not None
        else 256 * 1024
    )
    payload_bytes = bytes(payload or b"")
    if len(payload_bytes) < threshold:
        return None
    if endpoint_url:
        _validate_llm_shard_adapter_endpoint_url(
            endpoint_url,
            allow_remote=allow_remote_endpoint_url,
        )
        parsed = urlsplit(endpoint_url)
        if not _is_loopback_host(str(parsed.hostname or "").strip()):
            return None
    base_root = Path(
        str(file_io_root or "").strip()
        or os.getenv(CAI_LLM_SHARD_IO_ROOT_ENV, "").strip()
        or Path(tempfile.gettempdir()) / "cai-llm-shard-io"
    ).resolve()
    base_root.mkdir(parents=True, exist_ok=True)
    io_root = Path(tempfile.mkdtemp(prefix="cai-llm-shard-io-", dir=str(base_root)))
    payload_path = io_root / "payload.bin"
    payload_path.write_bytes(payload_bytes)
    payload_hash = hashlib.sha256(payload_bytes).hexdigest()
    response_output_path = io_root / "output.bin"
    return {
        "ioRoot": str(io_root),
        "payloadFile": {
            "path": str(payload_path),
            "sha256Hex": payload_hash,
            "sizeBytes": len(payload_bytes),
        },
        "localFileContract": {
            "schemaVersion": 1,
            "abi": LLAMA_CPP_EXTERNAL_SHARD_LOCAL_FILE_IO_ABI,
            "ioRoot": str(io_root),
            "responseOutputPath": str(response_output_path),
        },
    }


def _cleanup_external_adapter_local_file_context(
    context: Mapping[str, Any] | None,
) -> None:
    if not isinstance(context, Mapping):
        return
    io_root = str(context.get("ioRoot") or "").strip()
    if not io_root:
        return
    try:
        shutil.rmtree(io_root, ignore_errors=True)
    except Exception:
        return


def _cleanup_external_adapter_local_file_contract(
    contract: Mapping[str, Any] | None,
) -> None:
    if not isinstance(contract, Mapping):
        return
    io_root = str(contract.get("ioRoot") or "").strip()
    if not io_root:
        return
    try:
        shutil.rmtree(io_root, ignore_errors=True)
    except Exception:
        return


def _read_external_adapter_output_payload_file(
    output_file: Mapping[str, Any],
    *,
    local_file_contract: Mapping[str, Any] | None,
) -> bytes:
    path_value = str(output_file.get("path") or "").strip()
    if not path_value:
        raise ValueError("External llama.cpp shard adapter output payload file is missing.")
    output_path = Path(path_value)
    if not output_path.is_absolute():
        raise ValueError(
            "External llama.cpp shard adapter output payload file path must be absolute."
        )
    io_root = str(local_file_contract.get("ioRoot") or "").strip() if isinstance(
        local_file_contract, Mapping
    ) else ""
    if io_root:
        root_path = Path(io_root).resolve()
        resolved_output_path = output_path.resolve()
        try:
            resolved_output_path.relative_to(root_path)
        except ValueError as exc:
            raise ValueError(
                "External llama.cpp shard adapter output payload file escaped IO root."
            ) from exc
    try:
        payload = output_path.read_bytes()
    except Exception as exc:
        raise ValueError(
            "External llama.cpp shard adapter output payload file is unreadable."
        ) from exc
    expected_size = output_file.get("sizeBytes")
    if expected_size is not None:
        try:
            size_value = int(expected_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "External llama.cpp shard adapter output payload file size is invalid."
            ) from exc
        if size_value != len(payload):
            raise ValueError(
                "External llama.cpp shard adapter output payload file size mismatch."
            )
    expected_hash = str(output_file.get("sha256Hex") or "").strip().lower()
    if expected_hash and expected_hash != hashlib.sha256(payload).hexdigest():
        raise ValueError(
            "External llama.cpp shard adapter output payload file hash mismatch."
        )
    return payload


def _external_adapter_output_payload(response: Mapping[str, Any]) -> bytes:
    local_file_contract = (
        response.get("_localFileContract")
        if isinstance(response.get("_localFileContract"), Mapping)
        else None
    )
    try:
        output_file = response.get("outputPayloadFile")
        if isinstance(output_file, Mapping):
            return _read_external_adapter_output_payload_file(
                output_file,
                local_file_contract=local_file_contract,
            )
        raw = response.get("outputPayloadBase64")
        if raw is None:
            raw = response.get("output_payload_b64")
        if raw is None:
            raise ValueError(
                "External llama.cpp shard adapter output payload is missing."
            )
        try:
            return base64.b64decode(str(raw or "").encode("ascii"), validate=True)
        except Exception as exc:
            raise ValueError(
                "External llama.cpp shard adapter output payload is invalid."
            ) from exc
    finally:
        _cleanup_external_adapter_local_file_contract(local_file_contract)


def _external_adapter_output_metadata(
    response: Mapping[str, Any],
    output_payload: bytes,
    *,
    expected_model_id: str | None = None,
    require_handoff_contract: bool = False,
    expected_frame_template: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw = response.get("outputFrameMetadata")
    if raw is None:
        raw = response.get("outputMetadata")
    if raw is None:
        if isinstance(expected_frame_template, Mapping):
            raise ValueError(
                "External llama.cpp shard adapter output frame metadata is "
                "required for the next LLM shard frame."
            )
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(
            "External llama.cpp shard adapter output metadata is invalid."
        )
    metadata = dict(raw)
    payload_hash = hashlib.sha256(bytes(output_payload or b"")).hexdigest()
    declared_hash = str(metadata.get("payloadSha256Hex") or "").strip().lower()
    if declared_hash and declared_hash != payload_hash:
        raise ValueError(
            "External llama.cpp shard adapter output metadata hash mismatch."
        )
    if metadata.get("payloadSha256Hex") is None:
        metadata["payloadSha256Hex"] = payload_hash
    if metadata.get("frameSchemaVersion") is not None:
        valid, error = validate_cai_owned_transport_frame_metadata(
            metadata,
            expected_model_id=expected_model_id,
            require_llm_handoff=require_handoff_contract,
        )
        if not valid:
            raise ValueError(
                error or "External llama.cpp shard adapter output metadata is invalid."
            )
    if isinstance(expected_frame_template, Mapping):
        expected_metadata = _frame_template_with_output_hash(
            expected_frame_template,
            payload_hash,
        )
        mismatch = _metadata_template_mismatch(metadata, expected_metadata)
        if mismatch:
            raise ValueError(
                "External llama.cpp shard adapter output metadata does not match "
                f"next frame template: {mismatch}"
            )
    return metadata


def _metadata_template_mismatch(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    path: str = "metadata",
) -> str | None:
    for key, expected_value in expected.items():
        field_path = f"{path}.{key}"
        if key not in actual:
            return f"{field_path} is missing"
        actual_value = actual.get(key)
        if isinstance(expected_value, Mapping):
            if not isinstance(actual_value, Mapping):
                return f"{field_path} is not an object"
            nested = _metadata_template_mismatch(
                actual_value,
                expected_value,
                path=field_path,
            )
            if nested:
                return nested
            continue
        if (
            isinstance(expected_value, Sequence)
            and not isinstance(expected_value, (str, bytes, bytearray))
        ):
            if not (
                isinstance(actual_value, Sequence)
                and not isinstance(actual_value, (str, bytes, bytearray))
            ):
                return f"{field_path} is not a sequence"
            if list(actual_value) != list(expected_value):
                return f"{field_path} differs from template"
            continue
        if actual_value != expected_value:
            return f"{field_path} differs from template"
    return None


def _external_adapter_capability_set(response: Mapping[str, Any]) -> set[str]:
    capabilities: set[str] = set()

    def collect(raw: Any) -> None:
        if isinstance(raw, Mapping):
            supported = raw.get("supported")
            if supported is not None:
                collect(supported)
                return
            for key, value in raw.items():
                if bool(value):
                    capability = str(key or "").strip()
                    if capability:
                        capabilities.add(capability)
            return
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            for item in raw:
                capability = str(item or "").strip()
                if capability:
                    capabilities.add(capability)

    collect(response.get("capabilities"))
    patch_boundary = _external_adapter_patch_boundary(response)
    if isinstance(patch_boundary, Mapping):
        collect(patch_boundary.get("capabilities"))
    metrics = response.get("metrics")
    if isinstance(metrics, Mapping):
        collect(metrics.get("backendCapabilities"))
        collect(metrics.get("capabilities"))
    return capabilities


def _external_adapter_patch_boundary(response: Mapping[str, Any]) -> Any:
    boundary = response.get("patchBoundary")
    if boundary is None:
        boundary = response.get("patch_boundary")
    if boundary is None:
        metrics = response.get("metrics")
        if isinstance(metrics, Mapping):
            boundary = metrics.get("patchBoundary") or metrics.get("patch_boundary")
    return boundary


def _clean_string_sequence(values: Any) -> list[str]:
    if values is None or isinstance(values, (str, bytes)):
        return []
    if not isinstance(values, Sequence):
        return []
    cleaned: list[str] = []
    for item in values:
        value = str(item or "").strip()
        if value and value.isascii() and value not in cleaned:
            cleaned.append(value)
    return cleaned


def _stable_json_sha256_hex(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _self_test_runtime_metadata(
    model_id: str,
    runtime_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metadata = dict(runtime_metadata or {})
    metadata.setdefault("modelId", str(model_id or "").strip())
    metadata.setdefault("totalLayerCount", 2)
    metadata.setdefault("hiddenSize", 8)
    metadata.setdefault("activationDtype", "f16")
    metadata.setdefault("tensorEncoding", "ggml-tensor-v1")
    metadata.setdefault("tokenizerConfigHash", "00" * 32)
    metadata.setdefault("backend", "llama.cpp-patched")
    metadata.setdefault("backendVersion", "llama.cpp/cai-shard-self-test")
    metadata.setdefault("metadataSource", "adapter_self_test")
    return metadata


def _elapsed_ms(started: float) -> float:
    return max(0.0, (time.perf_counter() - started) * 1000.0)


def _runtime_metrics(
    config: CaiOwnedShardRuntimeConfig,
    *,
    runtime_id: str,
    adapter_metrics: dict[str, Any],
    payload_size_bytes: int,
    output_payload_size_bytes: int,
    processing_latency_ms: float,
) -> dict[str, Any]:
    metrics = dict(adapter_metrics or {})
    metrics["runtimeId"] = runtime_id
    metrics["runtimeVersion"] = CAI_OWNED_SHARD_RUNTIME_VERSION
    metrics["processingLatencyMs"] = processing_latency_ms
    metrics["payloadSizeBytes"] = max(0, int(payload_size_bytes or 0))
    metrics["outputPayloadSizeBytes"] = max(0, int(output_payload_size_bytes or 0))
    seconds = processing_latency_ms / 1000.0 if processing_latency_ms > 0 else 0.0
    metrics["batchesPerSecond"] = 1.0 / seconds if seconds > 0 else 0.0
    metrics["bytesPerSecond"] = (
        (payload_size_bytes + output_payload_size_bytes) / seconds
        if seconds > 0
        else 0.0
    )
    metrics.setdefault("adapterId", _adapter_id(metrics))
    metrics.setdefault("adapterVersion", _adapter_version(metrics))
    metrics["maxConcurrentBatches"] = _positive_int(config.max_concurrent_batches, 1)
    metrics["maxPayloadSizeBytes"] = _positive_int(
        config.max_payload_size_bytes,
        16 * 1024 * 1024,
    )
    return metrics


def _runtime_audit(
    config: CaiOwnedShardRuntimeConfig,
    runtime_id: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "nodeId": _require_runtime_node_id(config.node_id),
        "runtimeId": runtime_id,
        "runtimeVersion": CAI_OWNED_SHARD_RUNTIME_VERSION,
        "adapterId": _adapter_id(metrics),
        "adapterVersion": _adapter_version(metrics),
        "maxConcurrentBatches": _positive_int(config.max_concurrent_batches, 1),
        "maxPayloadSizeBytes": _positive_int(
            config.max_payload_size_bytes,
            16 * 1024 * 1024,
        ),
        "recordedAt": datetime.now(tz=UTC).isoformat(),
    }


def _work_item_route_audit(work_item: Mapping[str, Any]) -> dict[str, Any]:
    batch = work_item.get("batch") if isinstance(work_item, Mapping) else None
    existing: dict[str, Any] = {}
    if isinstance(batch, Mapping) and isinstance(batch.get("routeAudit"), Mapping):
        existing.update(dict(batch.get("routeAudit") or {}))
    existing.setdefault("selectedRoute", "local_inbox_payload")
    existing.setdefault("transportMode", "cai_owned_batch_payload")
    existing["payloadEndpoint"] = str(work_item.get("payloadEndpoint") or "")
    if isinstance(batch, Mapping):
        existing["sourceNodeId"] = str(batch.get("sourceNodeId") or "")
        existing["sinkNodeId"] = str(batch.get("sinkNodeId") or "")
        existing["attemptCount"] = _positive_int(batch.get("attemptCount"), 1)
    return existing


def _adapter_id(metrics: Mapping[str, Any]) -> str:
    return str(
        metrics.get("adapterId")
        or metrics.get("adapter")
        or DETERMINISTIC_BYTES_ADAPTER_ID
    ).strip()


def _adapter_version(metrics: Mapping[str, Any]) -> str:
    return str(
        metrics.get("adapterVersion")
        or DETERMINISTIC_BYTES_ADAPTER_VERSION
    ).strip()


def _coerce_adapter_result(
    value: CaiOwnedShardAdapterResult | bytes | bytearray,
) -> CaiOwnedShardAdapterResult:
    if isinstance(value, CaiOwnedShardAdapterResult):
        return value
    if isinstance(value, (bytes, bytearray)):
        return CaiOwnedShardAdapterResult(output_payload=bytes(value))
    raise ValueError("CAI-owned shard adapter returned unsupported result.")


def _runtime_error_retryable(exc: Exception) -> bool:
    message = str(exc).lower()
    if "hash does not match" in message:
        return False
    if "payload exceeds runtime capacity" in message:
        return False
    return True


def _batch_payload_size(batch: object) -> int:
    if not isinstance(batch, Mapping):
        return 0
    try:
        return max(0, int(batch.get("payloadSizeBytes") or 0))
    except (TypeError, ValueError):
        return 0


def _batch_lease_active(batch: object, now: datetime) -> bool:
    if not isinstance(batch, Mapping):
        return False
    expires_at = _parse_datetime(batch.get("leaseExpiresAt"))
    if expires_at is None:
        return False
    return expires_at > now


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _positive_int(value: object, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    try:
        return max(0.1, float(env.get(key) or default))
    except (TypeError, ValueError):
        return default


def _env_optional_float(
    env: Mapping[str, str],
    key: str,
    default: float | None,
) -> float | None:
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_optional_int(
    env: Mapping[str, str],
    key: str,
    default: int | None,
) -> int | None:
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_optional_text(
    env: Mapping[str, str],
    key: str,
    default: str | None = None,
) -> str | None:
    raw = env.get(key)
    if raw is None:
        return default
    text = str(raw).strip()
    if not text:
        return default
    return text


def _env_optional_json_mapping(
    env: Mapping[str, str],
    key: str,
) -> dict[str, Any] | None:
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        parsed = json.loads(str(raw))
    except Exception as exc:
        raise ValueError(f"{key} must be valid JSON.") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{key} must be a JSON object.")
    return _normalize_llama_cpp_external_shard_artifact_hint(parsed)


def _env_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on", "required"}


def _env_command(env: Mapping[str, str], key: str) -> list[str]:
    raw = str(env.get(key) or "").strip()
    if not raw:
        return []
    return shlex.split(raw, posix=True)


def _python_module_command(module_name: str, *args: str) -> list[str]:
    module = str(module_name or "").strip()
    if not module:
        return [sys.executable, *[str(arg) for arg in args]]
    if getattr(sys, "frozen", False):
        inline_code = f"from {module} import main; raise SystemExit(main())"
        return [sys.executable, "-c", inline_code, *[str(arg) for arg in args]]
    return [sys.executable, "-m", module, *[str(arg) for arg in args]]


def _smoke_runner_env(env: Mapping[str, str]) -> dict[str, str]:
    extra = _runtime_src_pythonpath_env(env)
    for key in ("CAI_SHARD_SMOKE_PREFILL_PREFIX", "CAI_SHARD_SMOKE_DECODE_PREFIX"):
        if env.get(key) is not None:
            extra[key] = str(env.get(key) or "")
    return extra


def _slot_state_engine_env(env: Mapping[str, str]) -> dict[str, str]:
    extra = _runtime_src_pythonpath_env(env)
    for key in (
        CAI_LLM_SHARD_SLOT_SERVER_URL_ENV,
        CAI_LLM_SHARD_SLOT_STATE_DIR_ENV,
        CAI_LLM_SHARD_SLOT_ID_ENV,
        CAI_LLM_SHARD_SLOT_TIMEOUT_ENV,
        CAI_LLM_SHARD_SLOT_DECODE_TOKENS_ENV,
    ):
        if env.get(key) is not None:
            extra[key] = str(env.get(key) or "")
    return extra


def _runtime_src_pythonpath_env(env: Mapping[str, str]) -> dict[str, str]:
    extra: dict[str, str] = {}
    src_root = str(_runtime_src_root())
    existing = str(env.get("PYTHONPATH") or os.environ.get("PYTHONPATH") or "")
    if src_root:
        extra["PYTHONPATH"] = (
            src_root if not existing else src_root + os.pathsep + existing
        )
    return extra


def _runtime_src_root() -> str:
    package_dir = os.path.dirname(os.path.abspath(__file__))
    src_root = os.path.dirname(package_dir)
    return src_root if src_root else package_dir


def _optional_int(value: object, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coalesce_optional_int(*values: object) -> int | None:
    for value in values:
        clean = _optional_int(value)
        if clean is not None:
            return clean
    return None


def _normalize_sha256_hex(value: object, *, field_name: str) -> str:
    clean = str(value or "").strip().lower()
    if len(clean) != 64 or any(ch not in "0123456789abcdef" for ch in clean):
        raise ValueError(f"{field_name} is invalid.")
    return clean


def _require_runtime_node_id(value: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError("CAI-owned shard runtime requires node id.")
    return clean


def _require_runtime_id(value: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError("CAI-owned shard runtime requires runtime id.")
    return clean

# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import base64
import hashlib
import json


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.cai_owned_runtime import (  # noqa: E402
    CAI_LLM_SHARD_SELF_TEST_CACHE_SCHEMA_VERSION,
    CAI_OWNED_SHARD_RUNTIME_VERSION,
    DETERMINISTIC_BYTES_ADAPTER_ID,
    DETERMINISTIC_BYTES_ADAPTER_VERSION,
    TASK_LEVEL_HTTP_INFERENCE_ADAPTER_ID,
    TASK_LEVEL_HTTP_INFERENCE_ADAPTER_VERSION,
    CaiOwnedShardAdapterResult,
    CaiOwnedShardFrame,
    CaiOwnedShardRuntimeConfig,
    DeterministicBytesShardAdapter,
    ExternalLlamaCppShardAdapter,
    LLAMA_CPP_EXTERNAL_SHARD_SPEC_ABI,
    TaskLevelHttpInferenceAdapter,
    build_llama_cpp_external_shard_patch_boundary,
    build_llama_cpp_external_shard_production_state_contract,
    build_llama_cpp_external_shard_spec,
    cai_owned_shard_adapter_from_env,
    cai_owned_llm_shard_self_test_file_path,
    cai_owned_transport_live_proof_audit,
    cai_owned_transport_live_proof_file_path,
    cai_owned_transport_runtime_capacity_status,
    load_cai_owned_transport_live_proof_result,
    load_cai_owned_llm_shard_self_test_result,
    run_cai_owned_llm_shard_adapter_self_test,
    run_cai_owned_shard_runtime_once,
    save_cai_owned_llm_shard_self_test_result,
    save_cai_owned_transport_live_proof_result,
    validate_llama_cpp_external_shard_patch_boundary,
    validate_llama_cpp_external_shard_production_state_contract,
    validate_llama_cpp_external_shard_spec,
    _llm_shard_generation_probe_ready,
    _peer_urls_for_node,
)
from cai_compute_chain.cli import (  # noqa: E402
    handle_cai_owned_llm_shard_conformance,
    handle_cai_owned_llm_shard_self_test,
    handle_cai_owned_runtime,
)
from cai_compute_chain.cai_llm_shard_conformance import (  # noqa: E402
    run_cai_owned_llm_shard_conformance,
)
from cai_compute_chain.cai_llama_cpp_shard_http_smoke_bridge import (  # noqa: E402
    handle_http_smoke_bridge_request_body,
)
from cai_compute_chain.decentralized_compute import (  # noqa: E402
    build_cai_owned_llm_shard_frame_metadata_from_runtime,
    build_cai_owned_transport_batch_envelope,
    build_cai_owned_transport_frame_metadata,
    build_cai_owned_transport_output_batch_envelope,
    build_cai_owned_transport_session_offer,
    build_cai_owned_llm_handoff_metadata,
    cai_owned_transport_batch_payload_bytes,
    complete_cai_owned_transport_session,
    claim_cai_owned_transport_batch,
    create_cai_owned_transport_session_from_offer,
    dispatch_cai_owned_transport_execution_dag,
    list_cai_owned_transport_sessions,
    read_cai_owned_transport_batch_output_payload,
    record_cai_owned_transport_batch_envelope,
    validate_cai_owned_transport_batch_envelope,
)
from cai_compute_chain.model import WalletPolicy  # noqa: E402
from cai_compute_chain.wallet_signing import (  # noqa: E402
    encode_bytes,
    generate_signing_seed,
    public_key_b64_from_seed,
)


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def getcode(self) -> int:
        return self.status


def _signing_material() -> dict[str, str]:
    signing_seed = generate_signing_seed()
    return {
        "public_key_b64": public_key_b64_from_seed(signing_seed),
        "signing_seed_b64": encode_bytes(signing_seed),
    }


def _verified_live_transport_proof(
    *,
    executor_node_ids: list[str] | None = None,
) -> dict[str, object]:
    return {
        "status": "ok",
        "sessionId": "caiot_live_proof",
        "instanceId": "instance-live-proof",
        "requesterNodeId": "node-user",
        "executorNodeIds": executor_node_ids or ["node-a", "node-b"],
        "finalResult": {
            "proofVerified": True,
            "finalOutput": {
                "payloadBase64": "b2s=",
                "payloadSha256Hex": hashlib.sha256(b"ok").hexdigest(),
            },
        },
    }


def _smoke_runner_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "cai_compute_chain.cai_llama_cpp_shard_smoke_runner",
    ]


def _output_contract_backend_code() -> str:
    return r"""
import base64
import hashlib
import json
import os
import sys

request = json.loads(sys.stdin.read() or "{}")
capture_path = os.environ.get("CAI_CAPTURE_REQUEST_PATH")
if capture_path:
    with open(capture_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(request, sort_keys=True) + "\n")

if request.get("action") == "load_shard":
    print(json.dumps({
        "status": "ready",
        "capabilities": [
            "layer_range_execution",
            "activation_handoff",
            "decode_state_handoff",
            "output_frame_metadata",
        ],
        "patchBoundary": {
            "schemaVersion": 1,
            "abi": "cai-llama-cpp-shard-patch-boundary-v1",
            "backend": "llama.cpp-patched",
            "backendVersion": "llama.cpp/cai-test",
            "patchId": "cai-llama-cpp-shard-output-contract-test",
            "runnerProtocolVersion": "0.1",
            "modelFormat": "gguf",
            "requiresPatchedBackend": True,
            "activationBoundary": "layer-range-activation-v1",
            "decodeStateBoundary": "token-step-kv-cache-v1",
            "supportedTensorEncodings": ["ggml-tensor-v1"],
            "capabilities": [
                "layer_range_execution",
                "activation_handoff",
                "decode_state_handoff",
                "output_frame_metadata",
            ],
        },
        "metrics": {"backendMode": "unit_test"},
    }))
    raise SystemExit(0)

payload = base64.b64decode(request.get("payloadBase64") or "")
output = b"state:" + payload
output_hash = hashlib.sha256(output).hexdigest()
contract = request.get("outputContract") or {}
template = json.loads(json.dumps(contract.get("frameMetadataTemplate") or {}))
if os.environ.get("CAI_BAD_OUTPUT_TEMPLATE") == "1":
    template["stageId"] = "wrong-stage"
template["payloadSha256Hex"] = output_hash
handoff = dict(template.get("llmHandoff") or {})
tensor = dict(handoff.get("tensor") or {})
tensor["sha256Hex"] = output_hash
handoff["tensor"] = tensor
template["llmHandoff"] = handoff
print(json.dumps({
    "status": "ok",
    "outputPayloadBase64": base64.b64encode(output).decode("ascii"),
    "outputPayloadSha256Hex": output_hash,
    "outputFrameMetadata": template,
    "metrics": {"backendMode": "unit_test"},
}))
"""


def _production_manifest_self_test_backend_code() -> str:
    return r"""
import base64
import hashlib
import json
import os
import sys

request = json.loads(sys.stdin.read() or "{}")
backend_mode = os.environ.get("CAI_BACKEND_MODE", "real_llama_cpp")
capabilities = [
    "layer_range_execution",
    "activation_handoff",
    "decode_state_handoff",
    "output_frame_metadata",
]
extra_metadata = {}
if os.environ.get("CAI_REAL_STATE_MANIFEST") == "1":
    capabilities.extend([
        "gguf_layer_execution",
        "real_activation_state",
        "real_decode_state",
    ])
    extra_metadata["productionStateContract"] = {
        "schemaVersion": 1,
        "abi": "cai-llama-cpp-production-state-contract-v1",
        "activationStateFormat": "ggml-tensor-v1/layer-range-activation-v1",
        "decodeStateFormat": "ggml-kv-cache-v1/token-step-kv-cache-v1",
        "modelExecutionBackend": "llama.cpp-cai-shard",
        "tensorEncoding": "ggml-tensor-v1",
        "shardExecutionMode": "layer_range",
        "fullModelReplicaRequired": False,
        "activationStateIsSynthetic": False,
        "decodeStateIsSynthetic": False,
    }

if request.get("action") == "load_shard":
    patch_boundary = {
        "schemaVersion": 1,
        "abi": "cai-llama-cpp-shard-patch-boundary-v1",
        "backend": "llama.cpp-patched",
        "backendVersion": "llama.cpp/cai-real",
        "patchId": "cai-llama-cpp-shard-real",
        "runnerProtocolVersion": "0.1",
        "modelFormat": "gguf",
        "requiresPatchedBackend": True,
        "activationBoundary": "layer-range-activation-v1",
        "decodeStateBoundary": "token-step-kv-cache-v1",
        "supportedTensorEncodings": ["ggml-tensor-v1"],
        "capabilities": capabilities,
    }
    if extra_metadata:
        patch_boundary["extraMetadata"] = extra_metadata
    print(json.dumps({
        "status": "ready",
        "capabilities": capabilities,
        "patchBoundary": patch_boundary,
        "metrics": {"backendLoaded": True, "backendMode": backend_mode},
    }))
    raise SystemExit(0)

if request.get("action") == "probe_generation":
    if os.environ.get("CAI_DISABLE_GENERATION_PROBE") == "1":
        print(json.dumps({
            "status": "error",
            "error": "generation probe disabled",
            "metrics": {"backendMode": backend_mode},
        }))
        raise SystemExit(0)
    probe = request.get("generationProbe") or {}
    print(json.dumps({
        "status": "ok",
        "generationProbe": {
            "schemaVersion": 1,
            "abi": "cai-llama-cpp-generation-probe-v1",
            "ready": True,
            "modelId": probe.get("modelId"),
            "outputText": "ok",
            "outputTokenCount": 1,
            "realModelExecution": True,
            "realLayerExecution": True,
        },
        "metrics": {
            "backendMode": backend_mode,
            "generationProbeReady": True,
        },
    }))
    raise SystemExit(0)

if request.get("action") in {"process_prefill", "process_decode"}:
    payload = base64.b64decode(request.get("payloadBase64") or "")
    prefix = b"decoded-real:" if request.get("action") == "process_decode" else b"real-state:"
    output = prefix + payload
    output_hash = hashlib.sha256(output).hexdigest()
    contract = request.get("outputContract") or {}
    template = json.loads(json.dumps(contract.get("frameMetadataTemplate") or {}))
    if template:
        template["payloadSha256Hex"] = output_hash
        handoff = dict(template.get("llmHandoff") or {})
        tensor = dict(handoff.get("tensor") or {})
        tensor["sha256Hex"] = output_hash
        handoff["tensor"] = tensor
        template["llmHandoff"] = handoff
    print(json.dumps({
        "status": "ok",
        "outputPayloadBase64": base64.b64encode(output).decode("ascii"),
        "outputPayloadSha256Hex": output_hash,
        "outputFrameMetadata": template,
        "metrics": {"backendMode": backend_mode},
    }))
    raise SystemExit(0)

print(json.dumps({"status": "ok", "metrics": {"backendMode": backend_mode}}))
"""


def _mismatched_process_patch_boundary_backend_code() -> str:
    return r"""
import base64
import hashlib
import json
import sys

request = json.loads(sys.stdin.read() or "{}")
base_boundary = {
    "schemaVersion": 1,
    "abi": "cai-llama-cpp-shard-patch-boundary-v1",
    "backend": "llama.cpp-patched",
    "backendVersion": "llama.cpp/cai-test",
    "runnerProtocolVersion": "0.1",
    "modelFormat": "gguf",
    "requiresPatchedBackend": True,
    "activationBoundary": "layer-range-activation-v1",
    "decodeStateBoundary": "token-step-kv-cache-v1",
    "supportedTensorEncodings": ["ggml-tensor-v1"],
    "capabilities": [
        "layer_range_execution",
        "activation_handoff",
        "decode_state_handoff",
        "output_frame_metadata",
    ],
}

if request.get("action") == "load_shard":
    boundary = dict(base_boundary)
    boundary["patchId"] = "cai-llama-cpp-shard-stable"
    print(json.dumps({
        "status": "ready",
        "capabilities": list(boundary["capabilities"]),
        "patchBoundary": boundary,
        "metrics": {"backendMode": "unit_test"},
    }))
    raise SystemExit(0)

payload = base64.b64decode(request.get("payloadBase64") or "")
output = b"state:" + payload
output_hash = hashlib.sha256(output).hexdigest()
contract = request.get("outputContract") or {}
template = json.loads(json.dumps(contract.get("frameMetadataTemplate") or {}))
if template:
    template["payloadSha256Hex"] = output_hash
    handoff = dict(template.get("llmHandoff") or {})
    tensor = dict(handoff.get("tensor") or {})
    tensor["sha256Hex"] = output_hash
    handoff["tensor"] = tensor
    template["llmHandoff"] = handoff
boundary = dict(base_boundary)
boundary["patchId"] = "cai-llama-cpp-shard-drifted"
print(json.dumps({
    "status": "ok",
    "outputPayloadBase64": base64.b64encode(output).decode("ascii"),
    "outputPayloadSha256Hex": output_hash,
    "outputFrameMetadata": template,
    "patchBoundary": boundary,
    "metrics": {"backendMode": "unit_test"},
}))
"""


def _metrics_only_production_capabilities_backend_code() -> str:
    return r"""
import base64
import hashlib
import json
import sys

request = json.loads(sys.stdin.read() or "{}")
required_capabilities = [
    "layer_range_execution",
    "activation_handoff",
    "decode_state_handoff",
    "output_frame_metadata",
]
production_capabilities = [
    "gguf_layer_execution",
    "real_activation_state",
    "real_decode_state",
]
state_contract = {
    "schemaVersion": 1,
    "abi": "cai-llama-cpp-production-state-contract-v1",
    "activationStateFormat": "ggml-tensor-v1/layer-range-activation-v1",
    "decodeStateFormat": "ggml-kv-cache-v1/token-step-kv-cache-v1",
    "modelExecutionBackend": "llama.cpp-cai-shard",
    "tensorEncoding": "ggml-tensor-v1",
    "shardExecutionMode": "layer_range",
    "fullModelReplicaRequired": False,
    "activationStateIsSynthetic": False,
    "decodeStateIsSynthetic": False,
}

if request.get("action") == "load_shard":
    print(json.dumps({
        "status": "ready",
        "capabilities": required_capabilities + production_capabilities,
        "patchBoundary": {
            "schemaVersion": 1,
            "abi": "cai-llama-cpp-shard-patch-boundary-v1",
            "backend": "llama.cpp-patched",
            "backendVersion": "llama.cpp/cai-real",
            "patchId": "cai-llama-cpp-shard-real",
            "runnerProtocolVersion": "0.1",
            "modelFormat": "gguf",
            "requiresPatchedBackend": True,
            "activationBoundary": "layer-range-activation-v1",
            "decodeStateBoundary": "token-step-kv-cache-v1",
            "supportedTensorEncodings": ["ggml-tensor-v1"],
            "capabilities": required_capabilities,
            "extraMetadata": {"productionStateContract": state_contract},
        },
        "metrics": {
            "backendLoaded": True,
            "backendMode": "real_llama_cpp",
            "backendCapabilities": required_capabilities + production_capabilities,
        },
    }))
    raise SystemExit(0)

if request.get("action") == "probe_generation":
    probe = request.get("generationProbe") or {}
    print(json.dumps({
        "status": "ok",
        "generationProbe": {
            "schemaVersion": 1,
            "abi": "cai-llama-cpp-generation-probe-v1",
            "ready": True,
            "modelId": probe.get("modelId"),
            "outputText": "ok",
            "outputTokenCount": 1,
            "realModelExecution": True,
        },
        "metrics": {
            "backendMode": "real_llama_cpp",
            "generationProbeReady": True,
        },
    }))
    raise SystemExit(0)

if request.get("action") in {"process_prefill", "process_decode"}:
    payload = base64.b64decode(request.get("payloadBase64") or "")
    prefix = b"decoded-real:" if request.get("action") == "process_decode" else b"real-state:"
    output = prefix + payload
    output_hash = hashlib.sha256(output).hexdigest()
    contract = request.get("outputContract") or {}
    template = json.loads(json.dumps(contract.get("frameMetadataTemplate") or {}))
    if template:
        template["payloadSha256Hex"] = output_hash
        handoff = dict(template.get("llmHandoff") or {})
        tensor = dict(handoff.get("tensor") or {})
        tensor["sha256Hex"] = output_hash
        handoff["tensor"] = tensor
        template["llmHandoff"] = handoff
    print(json.dumps({
        "status": "ok",
        "outputPayloadBase64": base64.b64encode(output).decode("ascii"),
        "outputPayloadSha256Hex": output_hash,
        "outputFrameMetadata": template,
        "metrics": {"backendMode": "real_llama_cpp"},
    }))
    raise SystemExit(0)

print(json.dumps({"status": "ok", "metrics": {"backendMode": "real_llama_cpp"}}))
"""


class _SpyLlmShardAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def load_shard(self, frame: CaiOwnedShardFrame) -> dict[str, object]:
        self.calls.append(("load_shard", frame.phase, frame.batch_id))
        return {
            "spyLoaded": True,
            "spyLayerStart": frame.layer_start,
            "spyLayerEnd": frame.layer_end,
        }

    def process_prefill(self, frame: CaiOwnedShardFrame) -> CaiOwnedShardAdapterResult:
        self.calls.append(("process_prefill", frame.phase, frame.batch_id))
        return CaiOwnedShardAdapterResult(
            output_payload=b"prefill:" + frame.payload,
            metrics={"adapterId": "spy-llm", "adapterVersion": "spy-llm/0.1"},
        )

    def process_decode(self, frame: CaiOwnedShardFrame) -> CaiOwnedShardAdapterResult:
        self.calls.append(("process_decode", frame.phase, frame.batch_id))
        return CaiOwnedShardAdapterResult(
            output_payload=b"decode:" + frame.payload,
            metrics={"adapterId": "spy-llm", "adapterVersion": "spy-llm/0.1"},
        )

    def finalize(
        self,
        frame: CaiOwnedShardFrame,
        result: CaiOwnedShardAdapterResult,
    ) -> dict[str, object]:
        self.calls.append(("finalize", frame.phase, frame.batch_id))
        return {"spyFinalized": True, "spyOutputBytes": len(result.output_payload)}


class _PersistentSpyLlmShardAdapter(_SpyLlmShardAdapter):
    def probe_health(self) -> dict[str, object]:
        return {
            "status": "ok",
            "ready": True,
            "healthEndpointAvailable": True,
            "nativeCommandConfigured": True,
            "nativeEngineMode": "persistent_jsonl",
            "persistentEngine": {"alive": True, "pid": 42},
        }


def test_cai_owned_shard_adapter_from_env_selects_deterministic_adapter() -> None:
    adapter = cai_owned_shard_adapter_from_env(
        {
            "CAI_LLM_SHARD_ADAPTER": "deterministic",
            "CAI_DETERMINISTIC_SHARD_PREFIX": "env:",
        }
    )

    assert isinstance(adapter, DeterministicBytesShardAdapter)
    assert adapter.prefix == b"env:"


def test_cai_owned_shard_adapter_from_env_selects_smoke_runner() -> None:
    adapter = cai_owned_shard_adapter_from_env(
        {
            "CAI_LLM_SHARD_ADAPTER": "smoke_runner",
            "CAI_LLM_SHARD_ADAPTER_TIMEOUT_SEC": "7",
            "CAI_SHARD_SMOKE_PREFILL_PREFIX": "prefill-env:",
        }
    )

    assert isinstance(adapter, ExternalLlamaCppShardAdapter)
    assert adapter.command == _smoke_runner_command()
    assert adapter.timeout_sec == 7
    assert adapter.require_handoff_contract is True
    assert adapter.require_patch_boundary is True
    assert "PYTHONPATH" in adapter.env
    assert adapter.env["CAI_SHARD_SMOKE_PREFILL_PREFIX"] == "prefill-env:"


def test_cai_owned_shard_adapter_from_env_reads_file_io_settings() -> None:
    adapter = cai_owned_shard_adapter_from_env(
        {
            "CAI_LLM_SHARD_ADAPTER": "smoke_runner",
            "CAI_LLM_SHARD_IO_ROOT": "  local-file-io-root  ",
            "CAI_LLM_SHARD_FILE_IO_THRESHOLD_BYTES": "4096",
        }
    )

    assert isinstance(adapter, ExternalLlamaCppShardAdapter)
    assert adapter.file_io_root == "local-file-io-root"
    assert adapter.file_io_threshold_bytes == 4096


def test_cai_owned_shard_adapter_from_env_reads_artifact_hint_json() -> None:
    adapter = cai_owned_shard_adapter_from_env(
        {
            "CAI_LLM_SHARD_ADAPTER": "smoke_runner",
            "CAI_LLM_SHARD_ARTIFACT_HINT_JSON": json.dumps(
                {
                    "catalogId": "cai-private",
                    "version": "2026.05",
                    "artifactId": "gguf-main",
                    "modelArtifactPath": "C:/CAI/models/model.gguf",
                    "modelArtifactSha256Hex": "ab" * 32,
                    "modelArtifactSizeBytes": 123456,
                }
            ),
        }
    )

    assert isinstance(adapter, ExternalLlamaCppShardAdapter)
    assert adapter.shard_artifact_hint == {
        "catalogId": "cai-private",
        "version": "2026.05",
        "artifactId": "gguf-main",
        "modelArtifactPath": "C:/CAI/models/model.gguf",
        "modelArtifactSha256Hex": "ab" * 32,
        "modelArtifactSizeBytes": 123456,
    }


def test_cai_owned_shard_adapter_from_env_selects_native_bridge() -> None:
    adapter = cai_owned_shard_adapter_from_env(
        {
            "CAI_LLM_SHARD_ADAPTER": "native_bridge",
            "CAI_LLM_SHARD_ADAPTER_TIMEOUT_SEC": "9",
            "CAI_LLM_SHARD_NATIVE_COMMAND": '"C:/CAI/patched-engine.exe" --json',
            "CAI_LLM_SHARD_NATIVE_TIMEOUT_SEC": "11",
        }
    )

    assert isinstance(adapter, ExternalLlamaCppShardAdapter)
    assert adapter.command == [
        sys.executable,
        "-m",
        "cai_compute_chain.cai_llama_cpp_shard_native_bridge",
        "--oneshot",
    ]
    assert adapter.timeout_sec == 9
    assert adapter.require_handoff_contract is True
    assert adapter.require_patch_boundary is True
    assert adapter.env["CAI_LLM_SHARD_NATIVE_COMMAND"] == (
        '"C:/CAI/patched-engine.exe" --json'
    )
    assert adapter.env["CAI_LLM_SHARD_NATIVE_TIMEOUT_SEC"] == "11"
    assert "PYTHONPATH" in adapter.env


def test_cai_owned_shard_adapter_from_env_uses_inline_module_command_when_frozen(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    adapter = cai_owned_shard_adapter_from_env(
        {
            "CAI_LLM_SHARD_ADAPTER": "native_bridge",
            "CAI_LLM_SHARD_NATIVE_COMMAND": '"C:/CAI/patched-engine.exe" --json',
        }
    )

    assert isinstance(adapter, ExternalLlamaCppShardAdapter)
    assert adapter.command[0] == sys.executable
    assert adapter.command[1] == "-c"
    assert "cai_compute_chain.cai_llama_cpp_shard_native_bridge" in adapter.command[2]
    assert "main" in adapter.command[2]
    assert adapter.command[-1] == "--oneshot"


def test_cai_owned_shard_adapter_from_env_selects_native_bridge_endpoint() -> None:
    adapter = cai_owned_shard_adapter_from_env(
        {
            "CAI_LLM_SHARD_ADAPTER": "native_bridge",
            "CAI_LLM_SHARD_ADAPTER_URL": "http://127.0.0.1:9258/cai-shard",
            "CAI_LLM_SHARD_ADAPTER_TIMEOUT_SEC": "8",
        }
    )

    assert isinstance(adapter, ExternalLlamaCppShardAdapter)
    assert list(adapter.command) == []
    assert adapter.endpoint_url == "http://127.0.0.1:9258/cai-shard"
    assert adapter.timeout_sec == 8
    assert adapter.require_handoff_contract is True
    assert adapter.require_patch_boundary is True


def test_cai_owned_shard_adapter_from_env_selects_external_command() -> None:
    adapter = cai_owned_shard_adapter_from_env(
        {
            "CAI_LLM_SHARD_ADAPTER": "external_llama_cpp",
            "CAI_LLM_SHARD_ADAPTER_COMMAND": '"C:/Program Files/CAI/runner.exe" --json',
            "CAI_LLM_SHARD_ADAPTER_TIMEOUT_SEC": "3.5",
        }
    )

    assert isinstance(adapter, ExternalLlamaCppShardAdapter)
    assert adapter.command == ["C:/Program Files/CAI/runner.exe", "--json"]
    assert adapter.timeout_sec == 3.5
    assert adapter.require_handoff_contract is True
    assert adapter.require_patch_boundary is True


def test_cai_owned_shard_adapter_from_env_selects_external_http_endpoint() -> None:
    adapter = cai_owned_shard_adapter_from_env(
        {
            "CAI_LLM_SHARD_ADAPTER": "external_llama_cpp",
            "CAI_LLM_SHARD_ADAPTER_URL": "http://127.0.0.1:9257/cai-shard",
            "CAI_LLM_SHARD_ADAPTER_TIMEOUT_SEC": "4",
        }
    )

    assert isinstance(adapter, ExternalLlamaCppShardAdapter)
    assert list(adapter.command) == []
    assert adapter.endpoint_url == "http://127.0.0.1:9257/cai-shard"
    assert adapter.timeout_sec == 4
    assert adapter.require_handoff_contract is True
    assert adapter.require_patch_boundary is True


def test_cai_owned_shard_adapter_from_env_selects_task_http_inference() -> None:
    adapter = cai_owned_shard_adapter_from_env(
        {
            "CAI_LLM_SHARD_ADAPTER": "task_http",
            "CAI_TASK_INFERENCE_ADAPTER_URL": "http://127.0.0.1:8080/v1/chat/completions",
            "CAI_TASK_INFERENCE_ADAPTER_MODEL": "cai-network/Qwen3-0.6B-GGUF",
            "CAI_TASK_INFERENCE_ADAPTER_TIMEOUT_SEC": "6",
            "CAI_TASK_INFERENCE_ADAPTER_MAX_TOKENS": "12",
            "CAI_TASK_INFERENCE_ADAPTER_TEMPERATURE": "0.2",
        }
    )

    assert isinstance(adapter, TaskLevelHttpInferenceAdapter)
    assert adapter.endpoint_url == "http://127.0.0.1:8080/v1/chat/completions"
    assert adapter.model_id == "cai-network/Qwen3-0.6B-GGUF"
    assert adapter.timeout_sec == 6
    assert adapter.max_tokens == 12
    assert adapter.temperature == 0.2


def test_cai_owned_shard_adapter_from_env_rejects_remote_http_endpoint_by_default() -> None:
    try:
        cai_owned_shard_adapter_from_env(
            {
                "CAI_LLM_SHARD_ADAPTER": "external_llama_cpp",
                "CAI_LLM_SHARD_ADAPTER_URL": "http://198.51.100.20:9257/cai-shard",
            }
        )
    except ValueError as exc:
        error = str(exc)
    else:
        raise AssertionError("Expected remote LLM shard adapter URL to be rejected.")

    assert "must be loopback by default" in error
    assert "CAI_ALLOW_REMOTE_LLM_SHARD_ADAPTER_URL" in error


def test_cai_owned_shard_adapter_from_env_allows_remote_http_endpoint_with_guard() -> None:
    adapter = cai_owned_shard_adapter_from_env(
        {
            "CAI_LLM_SHARD_ADAPTER": "external_llama_cpp",
            "CAI_LLM_SHARD_ADAPTER_URL": "http://198.51.100.20:9257/cai-shard",
            "CAI_ALLOW_REMOTE_LLM_SHARD_ADAPTER_URL": "1",
        }
    )

    assert isinstance(adapter, ExternalLlamaCppShardAdapter)
    assert adapter.endpoint_url == "http://198.51.100.20:9257/cai-shard"
    assert adapter.allow_remote_endpoint_url is True


def test_cai_owned_shard_adapter_from_env_rejects_endpoint_credentials() -> None:
    try:
        cai_owned_shard_adapter_from_env(
            {
                "CAI_LLM_SHARD_ADAPTER": "external_llama_cpp",
                "CAI_LLM_SHARD_ADAPTER_URL": (
                    "http://user:secret@127.0.0.1:9257/cai-shard"
                ),
            }
        )
    except ValueError as exc:
        error = str(exc)
    else:
        raise AssertionError("Expected credentialed LLM shard adapter URL rejection.")

    assert error == "CAI LLM shard adapter endpoint URL must not contain credentials."


def test_llama_cpp_http_smoke_bridge_handles_load_shard_request() -> None:
    status_code, response = handle_http_smoke_bridge_request_body(
        json.dumps(
            {
                "schemaVersion": 1,
                "abi": "cai-llama-cpp-external-shard-v1",
                "action": "load_shard",
                "backend": "llama.cpp-patched",
                "frame": {"batchId": "caibatch_http_bridge_test"},
            },
            sort_keys=True,
        ).encode("utf-8")
    )

    assert status_code == 200
    assert response["status"] == "ready"
    assert response["metrics"]["backendMode"] == "smoke_runner"
    assert response["patchBoundary"]["abi"] == "cai-llama-cpp-shard-patch-boundary-v1"


def test_llama_cpp_external_patch_boundary_contract_validates() -> None:
    boundary = build_llama_cpp_external_shard_patch_boundary(
        backend_version="llama.cpp/cai-shard-0.1",
        patch_id="cai-llama-cpp-shard-test",
    )

    valid, error = validate_llama_cpp_external_shard_patch_boundary(
        boundary,
        expected_backend="llama.cpp-patched",
    )
    unsupported = dict(boundary)
    unsupported["activationBoundary"] = "plain-rpc"
    unsupported_valid, unsupported_error = (
        validate_llama_cpp_external_shard_patch_boundary(unsupported)
    )

    assert valid is True
    assert error is None
    assert boundary["abi"] == "cai-llama-cpp-shard-patch-boundary-v1"
    assert boundary["requiresPatchedBackend"] is True
    assert unsupported_valid is False
    assert unsupported_error == (
        "External llama.cpp shard adapter activation boundary is unsupported."
    )


def test_llama_cpp_external_shard_spec_contract_validates() -> None:
    payload = b"activation-state"
    runtime_metadata = {
        "modelId": "cai-network/Qwen3-0.6B-GGUF",
        "totalLayerCount": 28,
        "hiddenSize": 1024,
        "activationDtype": "f16",
        "tensorEncoding": "ggml-tensor-v1",
        "tokenizerConfigHash": "ab" * 32,
        "modelSha256Hex": "12" * 32,
        "backend": "llama.cpp-patched",
        "backendVersion": "llama.cpp/cai-shard-0.1",
        "metadataSource": "unit-test-runtime",
        "preferredFilename": "Qwen3-0.6B-Q8_0.gguf",
        "quantization": "Q8_0",
        "contextLength": 32768,
    }
    metadata = build_cai_owned_llm_shard_frame_metadata_from_runtime(
        model_id="cai-network/Qwen3-0.6B-GGUF",
        runtime_metadata=runtime_metadata,
        payload=payload,
        frame_kind="activation",
        layer_start=0,
        layer_end=14,
        token_start=0,
        token_end=8,
        sequence=0,
    )
    frame = CaiOwnedShardFrame(
        session_id="caiot_spec_test",
        batch_id="caibatch_spec_test",
        phase="prefill_activation_batches",
        source_node_id="node-user",
        sink_node_id="node-a",
        sequence=0,
        model_id="cai-network/Qwen3-0.6B-GGUF",
        frame_kind="activation",
        layer_start=0,
        layer_end=14,
        token_start=0,
        token_end=8,
        payload_sha256_hex=hashlib.sha256(payload).hexdigest(),
        payload=payload,
        metadata=metadata,
    )
    spec = build_llama_cpp_external_shard_spec(
        frame,
        artifact_hint={
            "catalogId": "cai-private",
            "version": "2026.05",
            "artifactId": "gguf-main",
            "preferredFilename": "Qwen3-0.6B-Q8_0.gguf",
            "modelArtifactPath": "C:/CAI/models/Qwen3-0.6B-Q8_0.gguf",
            "modelArtifactSha256Hex": "cd" * 32,
            "modelArtifactSizeBytes": 123456,
        },
    )
    valid, error = validate_llama_cpp_external_shard_spec(
        spec,
        expected_model_id="cai-network/Qwen3-0.6B-GGUF",
        expected_frame=frame,
    )
    unsupported = dict(spec)
    unsupported["modelFormat"] = "plain-rpc"
    unsupported_valid, unsupported_error = validate_llama_cpp_external_shard_spec(
        unsupported,
    )

    assert valid is True
    assert error is None
    assert spec["abi"] == LLAMA_CPP_EXTERNAL_SHARD_SPEC_ABI
    assert spec["totalLayerCount"] == 28
    assert spec["extraMetadata"]["preferredFilename"] == "Qwen3-0.6B-Q8_0.gguf"
    assert spec["extraMetadata"]["quantization"] == "Q8_0"
    assert spec["artifactHint"]["catalogId"] == "cai-private"
    assert unsupported_valid is False
    assert unsupported_error == (
        "External llama.cpp shard spec modelFormat is unsupported."
    )


def test_llama_cpp_external_production_state_contract_validates() -> None:
    contract = build_llama_cpp_external_shard_production_state_contract()
    valid, error = validate_llama_cpp_external_shard_production_state_contract(
        contract,
    )
    unsupported = dict(contract)
    unsupported["abi"] = "plain-state"
    unsupported_valid, unsupported_error = (
        validate_llama_cpp_external_shard_production_state_contract(unsupported)
    )

    assert valid is True
    assert error is None
    assert contract["abi"] == "cai-llama-cpp-production-state-contract-v1"
    assert contract["shardExecutionMode"] == "layer_range"
    assert contract["fullModelReplicaRequired"] is False
    assert contract["activationStateIsSynthetic"] is False
    assert contract["decodeStateIsSynthetic"] is False
    assert unsupported_valid is False
    assert unsupported_error == (
        "LLM shard adapter production state contract ABI is unsupported."
    )


def test_llama_cpp_external_production_state_contract_rejects_synthetic_state() -> None:
    try:
        build_llama_cpp_external_shard_production_state_contract(
            activation_state_is_synthetic=True,
        )
    except ValueError as exc:
        error = str(exc)
    else:
        raise AssertionError("Expected synthetic production state contract rejection.")

    assert error == "LLM shard adapter activation state is synthetic."


def test_llama_cpp_external_production_state_contract_rejects_full_model_replica() -> None:
    try:
        build_llama_cpp_external_shard_production_state_contract(
            full_model_replica_required=True,
        )
    except ValueError as exc:
        error = str(exc)
    else:
        raise AssertionError("Expected full model replica production state rejection.")

    assert error == (
        "LLM shard adapter production backend requires a full model replica."
    )


def test_cai_owned_llm_shard_self_test_rejects_deterministic_adapter() -> None:
    result = run_cai_owned_llm_shard_adapter_self_test(
        DeterministicBytesShardAdapter(),
    )

    assert result["status"] == "not_production_ready"
    assert result["contractReady"] is False
    assert result["productionReady"] is False
    assert result["adapterId"] == "deterministic-bytes"


def test_cai_owned_llm_shard_self_test_passes_smoke_runner_contract() -> None:
    result = run_cai_owned_llm_shard_adapter_self_test(
        ExternalLlamaCppShardAdapter(
            command=_smoke_runner_command(),
            env={"PYTHONPATH": str(SRC_ROOT)},
            timeout_sec=10,
        ),
        payload=b"self-test-prompt",
    )

    assert result["status"] == "passed"
    assert result["contractReady"] is True
    assert result["productionReady"] is False
    assert result["productionReadinessError"] == (
        "LLM shard adapter backend mode is not production."
    )
    assert result["backendMode"] == "smoke_runner"
    assert result["patchBoundaryVerified"] is True
    assert result["patchBoundaryPatchId"] == "cai-llama-cpp-shard-smoke-runner"
    assert result["outputFrameMetadataReady"] is True
    assert result["outputFrameMetadataError"] is None
    assert result["finalDecodeOutputReady"] is True
    assert result["finalDecodeOutputError"] is None
    assert result["prefillOutputPayloadSizeBytes"] > len(b"self-test-prompt")
    assert result["decodeOutputPayloadSizeBytes"] > result["prefillOutputPayloadSizeBytes"]
    assert result["outputPayloadSizeBytes"] > len(b"self-test-prompt")


def test_cai_owned_llm_shard_self_test_uses_balanced_layer_split() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        capture_path = Path(tmpdir) / "requests.jsonl"
        result = run_cai_owned_llm_shard_adapter_self_test(
            ExternalLlamaCppShardAdapter(
                command=[sys.executable, "-c", _output_contract_backend_code()],
                env={"CAI_CAPTURE_REQUEST_PATH": str(capture_path)},
                timeout_sec=10,
            ),
            runtime_metadata={"totalLayerCount": 28},
            payload=b"self-test-prompt",
        )

        requests = [
            json.loads(line)
            for line in capture_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    prefill = next(item for item in requests if item["action"] == "process_prefill")
    decode = next(item for item in requests if item["action"] == "process_decode")

    assert result["contractReady"] is True
    assert prefill["shardSpec"]["layerStart"] == 0
    assert prefill["shardSpec"]["layerEnd"] == 14
    assert decode["shardSpec"]["layerStart"] == 14
    assert decode["shardSpec"]["layerEnd"] == 28


def test_cai_owned_llm_shard_self_test_requires_real_state_manifest() -> None:
    result = run_cai_owned_llm_shard_adapter_self_test(
        ExternalLlamaCppShardAdapter(
            command=[
                sys.executable,
                "-c",
                _production_manifest_self_test_backend_code(),
            ],
            timeout_sec=10,
        ),
        payload=b"self-test-prompt",
    )

    assert result["status"] == "passed"
    assert result["contractReady"] is True
    assert result["productionReady"] is False
    assert result["patchBoundaryVerified"] is True
    assert result["backendMode"] == "real_llama_cpp"
    assert result["productionReadinessError"] == (
        "LLM shard adapter missing production capabilities: "
        "gguf_layer_execution, real_activation_state, real_decode_state"
    )
    assert result["productionReadinessChecks"]["productionStateContractReady"] is False
    assert result["productionReadinessChecks"]["missingProductionCapabilities"] == [
        "gguf_layer_execution",
        "real_activation_state",
        "real_decode_state",
    ]


def test_cai_owned_llm_shard_self_test_accepts_real_state_manifest() -> None:
    result = run_cai_owned_llm_shard_adapter_self_test(
        ExternalLlamaCppShardAdapter(
            command=[
                sys.executable,
                "-c",
                _production_manifest_self_test_backend_code(),
            ],
            env={"CAI_REAL_STATE_MANIFEST": "1"},
            timeout_sec=10,
        ),
        payload=b"self-test-prompt",
    )

    assert result["status"] == "passed"
    assert result["contractReady"] is True
    assert result["productionReady"] is True
    assert result["productionReadinessError"] is None
    assert result["productionReadinessChecks"]["productionStateContractReady"] is True
    assert result["productionReadinessChecks"]["missingProductionCapabilities"] == []
    assert result["productionReadinessChecks"]["generationProbeReady"] is True
    assert result["generationProbeReady"] is True
    assert result["generationProbe"]["realModelExecution"] is True
    assert result["generationProbe"]["realLayerExecution"] is True
    assert result["outputFrameMetadataReady"] is True
    assert result["finalDecodeOutputReady"] is True


def test_cai_owned_llm_shard_self_test_requires_production_caps_in_patch_boundary() -> None:
    result = run_cai_owned_llm_shard_adapter_self_test(
        ExternalLlamaCppShardAdapter(
            command=[
                sys.executable,
                "-c",
                _metrics_only_production_capabilities_backend_code(),
            ],
            timeout_sec=10,
        ),
        payload=b"self-test-prompt",
    )

    assert result["status"] == "passed"
    assert result["contractReady"] is True
    assert result["productionReady"] is False
    assert result["productionReadinessError"] == (
        "LLM shard adapter missing production capabilities: "
        "gguf_layer_execution, real_activation_state, real_decode_state"
    )
    assert result["productionReadinessChecks"]["backendCapabilities"] == [
        "activation_handoff",
        "decode_state_handoff",
        "gguf_layer_execution",
        "layer_range_execution",
        "output_frame_metadata",
        "real_activation_state",
        "real_decode_state",
    ]
    assert result["productionReadinessChecks"]["patchBoundaryCapabilities"] == [
        "activation_handoff",
        "decode_state_handoff",
        "layer_range_execution",
        "output_frame_metadata",
    ]
    assert result["productionReadinessChecks"]["productionStateContractReady"] is True


def test_cai_owned_llm_shard_self_test_requires_generation_probe_for_production() -> None:
    result = run_cai_owned_llm_shard_adapter_self_test(
        ExternalLlamaCppShardAdapter(
            command=[
                sys.executable,
                "-c",
                _production_manifest_self_test_backend_code(),
            ],
            env={
                "CAI_REAL_STATE_MANIFEST": "1",
                "CAI_DISABLE_GENERATION_PROBE": "1",
            },
            timeout_sec=10,
        ),
        payload=b"self-test-prompt",
    )

    assert result["status"] == "passed"
    assert result["contractReady"] is True
    assert result["productionReady"] is False
    assert result["generationProbeReady"] is False
    assert result["productionReadinessChecks"]["generationProbeReady"] is False
    assert "generation probe" in result["productionReadinessError"]


def test_cai_owned_llm_shard_self_test_rejects_missing_output_frame_metadata() -> None:
    backend_code = r"""
import base64
import json
import sys

request = json.loads(sys.stdin.read() or "{}")
if request.get("action") == "load_shard":
    print(json.dumps({
        "status": "ready",
        "capabilities": [
            "layer_range_execution",
            "activation_handoff",
            "decode_state_handoff",
            "output_frame_metadata",
        ],
        "patchBoundary": {
            "schemaVersion": 1,
            "abi": "cai-llama-cpp-shard-patch-boundary-v1",
            "backend": "llama.cpp-patched",
            "backendVersion": "llama.cpp/cai-test",
            "patchId": "cai-llama-cpp-shard-test",
            "runnerProtocolVersion": "0.1",
            "modelFormat": "gguf",
            "requiresPatchedBackend": True,
            "activationBoundary": "layer-range-activation-v1",
            "decodeStateBoundary": "token-step-kv-cache-v1",
            "supportedTensorEncodings": ["ggml-tensor-v1"],
            "capabilities": [
                "layer_range_execution",
                "activation_handoff",
                "decode_state_handoff",
                "output_frame_metadata",
            ],
        },
        "metrics": {"backendMode": "unit_test"},
    }))
elif request.get("action") in {"process_prefill", "process_decode"}:
    payload = base64.b64decode(request.get("payloadBase64") or "")
    output = b"state:" + payload
    print(json.dumps({
        "status": "ok",
        "outputPayloadBase64": base64.b64encode(output).decode("ascii"),
        "outputPayloadSha256Hex": __import__("hashlib").sha256(output).hexdigest(),
        "metrics": {"backendMode": "unit_test"},
    }))
else:
    print(json.dumps({"status": "ok", "metrics": {"backendMode": "unit_test"}}))
"""
    result = run_cai_owned_llm_shard_adapter_self_test(
        ExternalLlamaCppShardAdapter(
            command=[sys.executable, "-c", backend_code],
            timeout_sec=10,
        ),
        payload=b"self-test-prompt",
    )

    assert result["status"] == "failed"
    assert result["contractReady"] is False
    assert result["productionReady"] is False
    assert result["patchBoundaryVerified"] is False
    assert result["outputFrameMetadataReady"] is False
    assert result["outputFrameMetadataError"] == (
        "External llama.cpp shard adapter output frame metadata is "
        "required for the next LLM shard frame."
    )
    assert result["finalDecodeOutputReady"] is False
    assert result["finalDecodeOutputError"] == "Self-test failed before decode."


def test_external_llama_cpp_shard_adapter_probes_http_health_endpoint() -> None:
    calls: list[tuple[str, str]] = []

    def fake_urlopen(http_request, timeout):  # noqa: ANN001
        calls.append((http_request.get_method(), http_request.full_url))
        assert timeout == 7
        return _FakeResponse(
            {
                "status": "ok",
                "nativeCommandConfigured": True,
                "nativeEngineMode": "persistent_jsonl",
                "persistentEngine": {"alive": True, "pid": 42},
            },
        )

    adapter = ExternalLlamaCppShardAdapter(
        endpoint_url="http://127.0.0.1:9257/cai-shard",
        timeout_sec=7,
    )
    with patch("cai_compute_chain.cai_owned_runtime.urlopen", fake_urlopen):
        health = adapter.probe_health()

    assert calls == [("GET", "http://127.0.0.1:9257/health")]
    assert health is not None
    assert health["status"] == "ok"
    assert health["ready"] is True
    assert health["healthEndpointAvailable"] is True
    assert health["persistentEngine"]["alive"] is True


def test_cai_owned_llm_shard_self_test_blocks_degraded_http_health() -> None:
    calls: list[tuple[str, str]] = []

    def fake_urlopen(http_request, timeout):  # noqa: ANN001
        calls.append((http_request.get_method(), http_request.full_url))
        return _FakeResponse(
            {
                "status": "degraded",
                "error": "native engine down",
                "nativeCommandConfigured": False,
            },
            status=503,
        )

    adapter = ExternalLlamaCppShardAdapter(
        endpoint_url="http://127.0.0.1:9257/cai-shard",
        timeout_sec=10,
    )
    with patch("cai_compute_chain.cai_owned_runtime.urlopen", fake_urlopen):
        result = run_cai_owned_llm_shard_adapter_self_test(
            adapter,
            payload=b"self-test-prompt",
        )

    assert calls == [("GET", "http://127.0.0.1:9257/health")]
    assert result["status"] == "failed"
    assert result["contractReady"] is False
    assert result["productionReady"] is False
    assert result["backendHealthReady"] is False
    assert result["backendHealth"]["status"] == "degraded"
    assert "health check failed" in result["productionReadinessError"]


def test_cai_owned_llm_shard_self_test_cache_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        result = run_cai_owned_llm_shard_adapter_self_test(
            DeterministicBytesShardAdapter(),
        )
        saved = save_cai_owned_llm_shard_self_test_result(result, policy=policy)
        loaded = load_cai_owned_llm_shard_self_test_result(policy=policy)
        old_result = dict(result)
        old_result["recordedAt"] = "2000-01-01T00:00:00+00:00"
        save_cai_owned_llm_shard_self_test_result(old_result, policy=policy)
        missing_when_expired = load_cai_owned_llm_shard_self_test_result(
            policy=policy,
        )

    assert saved["path"].endswith("cai-owned-llm-shard-self-test.json")
    assert loaded is not None
    assert loaded["status"] == "not_production_ready"
    assert loaded["adapterId"] == "deterministic-bytes"
    assert loaded["cacheAgeSeconds"] >= 0
    assert missing_when_expired is None


def test_cai_owned_llm_shard_self_test_cache_rejects_stale_production_ready() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        path = cai_owned_llm_shard_self_test_file_path(policy)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schemaVersion": CAI_LLM_SHARD_SELF_TEST_CACHE_SCHEMA_VERSION,
                    "recordedAt": "2026-05-04T00:00:00+00:00",
                    "result": {
                        "status": "passed",
                        "contractReady": True,
                        "productionReady": True,
                        "productionReadinessChecks": {
                            "contractReady": True,
                            "productionStateContractReady": True,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )

        loaded = load_cai_owned_llm_shard_self_test_result(
            max_age_seconds=None,
            policy=policy,
        )

    assert loaded is None


def test_cai_owned_transport_live_proof_cache_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        proof = _verified_live_transport_proof()
        saved = save_cai_owned_transport_live_proof_result(proof, policy=policy)
        loaded = load_cai_owned_transport_live_proof_result(policy=policy)

    assert saved["path"].endswith("cai-owned-transport-live-proof.json")
    assert saved["audit"]["verified"] is True
    assert loaded is not None
    assert loaded["runtimeReadyProofAudit"]["verified"] is True
    assert loaded["runtimeReadyProofAudit"]["executorCount"] == 2
    assert loaded["cacheAgeSeconds"] >= 0


def test_cai_owned_transport_live_proof_rejects_single_executor() -> None:
    proof = _verified_live_transport_proof(executor_node_ids=["node-a"])
    audit = cai_owned_transport_live_proof_audit(proof)

    assert audit["verified"] is False
    assert "at least two executor" in audit["error"]


def test_cai_owned_transport_live_proof_cache_rejects_unverified_file() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        path = cai_owned_transport_live_proof_file_path(policy)
        path.parent.mkdir(parents=True, exist_ok=True)
        proof = _verified_live_transport_proof()
        proof["finalResult"] = {"proofVerified": False, "finalOutput": {}}
        path.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "recordedAt": "2026-05-04T00:00:00+00:00",
                    "result": proof,
                }
            ),
            encoding="utf-8",
        )

        loaded = load_cai_owned_transport_live_proof_result(
            max_age_seconds=None,
            policy=policy,
        )

    assert loaded is None


def test_cai_owned_shard_runtime_processes_one_work_item() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        session_id, batch_id = _create_received_batch(policy, payload=b"runtime-input")
        config = CaiOwnedShardRuntimeConfig(
            node_id="node-b",
            runtime_id="runtime-loop",
            policy=policy,
        )

        result = run_cai_owned_shard_runtime_once(
            config,
            DeterministicBytesShardAdapter(prefix=b"done:"),
        )
        output_payload = read_cai_owned_transport_batch_output_payload(
            session_id,
            batch_id,
            policy,
        )
        record = list_cai_owned_transport_sessions(policy)[0]
        batch = record.batch_records[0]

    assert result["status"] == "processed"
    assert result["completion"]["receipt"]["nodeId"] == "node-b"
    assert result["outputPayloadSizeBytes"] == len(b"done:runtime-input")
    assert output_payload == b"done:runtime-input"
    assert batch["status"] == "processed"
    assert batch["runtimeId"] == "runtime-loop"
    assert batch["metrics"]["adapter"] == "deterministic-bytes"
    assert batch["metrics"]["adapterId"] == DETERMINISTIC_BYTES_ADAPTER_ID
    assert batch["metrics"]["adapterVersion"] == DETERMINISTIC_BYTES_ADAPTER_VERSION
    assert batch["metrics"]["adapterPhase"] == "decode"
    assert batch["metrics"]["runtimeVersion"] == CAI_OWNED_SHARD_RUNTIME_VERSION
    assert batch["metrics"]["processingLatencyMs"] >= 0
    assert batch["metrics"]["batchesPerSecond"] >= 0
    assert batch["routeAudit"]["selectedRoute"] == "local_inbox_payload"
    assert batch["runtimeAudit"]["runtimeId"] == "runtime-loop"
    assert batch["runtimeAudit"]["adapterId"] == DETERMINISTIC_BYTES_ADAPTER_ID
    assert batch["hashChainSha256Hex"]
    assert result["completion"]["receipt"]["metrics"]["adapterIds"] == [
        DETERMINISTIC_BYTES_ADAPTER_ID
    ]
    assert result["completion"]["receipt"]["metrics"]["runtimeVersions"] == [
        CAI_OWNED_SHARD_RUNTIME_VERSION
    ]
    assert batch["outputPayloadSizeBytes"] == len(b"done:runtime-input")


def test_cai_owned_shard_runtime_blocks_degraded_llm_backend_health() -> None:
    calls: list[tuple[str, str]] = []

    def fake_urlopen(http_request, timeout):  # noqa: ANN001
        calls.append((http_request.get_method(), http_request.full_url))
        return _FakeResponse(
            {
                "status": "degraded",
                "error": "native engine down",
                "nativeCommandConfigured": False,
            },
            status=503,
        )

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        _session_id, _batch_id = _create_received_batch(
            policy,
            payload=b"runtime-input",
        )
        config = CaiOwnedShardRuntimeConfig(
            node_id="node-b",
            runtime_id="runtime-health-gate",
            max_attempts=1,
            policy=policy,
        )

        with patch("cai_compute_chain.cai_owned_runtime.urlopen", fake_urlopen):
            result = run_cai_owned_shard_runtime_once(
                config,
                ExternalLlamaCppShardAdapter(
                    endpoint_url="http://127.0.0.1:9257/cai-shard",
                    timeout_sec=10,
                ),
            )
        record = list_cai_owned_transport_sessions(policy)[0]
        batch = record.batch_records[0]

    assert calls == [("GET", "http://127.0.0.1:9257/health")]
    assert result["status"] == "failed"
    assert result["failure"]["retryScheduled"] is False
    assert batch["status"] == "failed"
    assert "health check failed" in batch["lastError"]
    assert batch["metrics"]["errorClass"] == "CaiOwnedLlmShardBackendHealthError"
    assert batch["metrics"]["backendHealthReady"] is False
    assert batch["metrics"]["backendHealth"]["status"] == "degraded"


def test_cai_owned_runtime_cli_processes_one_work_item_from_env_adapter() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch.dict(
        "os.environ",
        {
            "CAI_LLM_SHARD_ADAPTER": "deterministic",
            "CAI_DETERMINISTIC_SHARD_PREFIX": "cli:",
        },
        clear=False,
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        session_id, batch_id = _create_received_batch(
            policy,
            payload=b"runtime-cli-input",
        )
        output_text = handle_cai_owned_runtime(
            node_id="node-b",
            runtime_id="runtime-cli",
            wallet_data_dirname=".tmp-cai-runtime",
        )
        output = read_cai_owned_transport_batch_output_payload(
            session_id,
            batch_id,
            policy,
        )
        result = json.loads(output_text)

    assert result["status"] == "processed"
    assert result["nodeId"] == "node-b"
    assert result["runtimeId"] == "runtime-cli"
    assert output == b"cli:runtime-cli-input"


def test_cai_owned_llm_shard_self_test_cli_uses_env_adapter() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch.dict(
        "os.environ",
        {"CAI_LLM_SHARD_ADAPTER": "smoke_runner"},
        clear=False,
    ):
        output_text = handle_cai_owned_llm_shard_self_test(
            model_id="cai-network/Qwen3-0.6B-GGUF",
            payload="cli-self-test",
            save_readiness=True,
            wallet_data_dirname=".tmp-cai-runtime",
        )
        result = json.loads(output_text)
        cached_text = handle_cai_owned_llm_shard_self_test(
            show_cached=True,
            wallet_data_dirname=".tmp-cai-runtime",
        )
        cached = json.loads(cached_text)

    assert result["status"] == "passed"
    assert result["contractReady"] is True
    assert result["productionReady"] is False
    assert result["backendMode"] == "smoke_runner"
    assert result["savedReadiness"]["path"].endswith(
        "cai-owned-llm-shard-self-test.json"
    )
    assert cached["status"] == "cached"
    assert cached["cached"]["contractReady"] is True


def test_cai_owned_llm_shard_conformance_passes_contract_smoke_runner() -> None:
    with patch.dict(
        "os.environ",
        {"CAI_LLM_SHARD_ADAPTER": "smoke_runner"},
        clear=False,
    ):
        report = run_cai_owned_llm_shard_conformance(
            model_id="cai-network/Qwen3-0.6B-GGUF",
            payload="cli-conformance",
        )

    assert report["status"] == "passed"
    assert report["ok"] is True
    assert report["requireProduction"] is False
    assert report["errors"] == []
    assert report["checks"]["contractReady"] is True
    assert report["checks"]["productionReady"] is False
    assert report["selfTest"]["backendMode"] == "smoke_runner"


def test_cai_owned_llm_shard_conformance_fails_when_production_required() -> None:
    with patch.dict(
        "os.environ",
        {"CAI_LLM_SHARD_ADAPTER": "smoke_runner"},
        clear=False,
    ):
        report = run_cai_owned_llm_shard_conformance(
            model_id="cai-network/Qwen3-0.6B-GGUF",
            payload="cli-conformance",
            require_production=True,
        )

    assert report["status"] == "failed"
    assert report["ok"] is False
    assert report["requireProduction"] is True
    assert any("not productionReady" in item for item in report["errors"])
    assert report["checks"]["contractReady"] is True


def test_cai_owned_llm_shard_conformance_passes_production_manifest_backend() -> None:
    report = run_cai_owned_llm_shard_conformance(
        adapter=ExternalLlamaCppShardAdapter(
            command=[
                sys.executable,
                "-c",
                _production_manifest_self_test_backend_code(),
            ],
            env={"CAI_REAL_STATE_MANIFEST": "1"},
            timeout_sec=10,
        ),
        model_id="cai-network/Qwen3-0.6B-GGUF",
        payload="cli-conformance",
        require_production=True,
    )

    assert report["status"] == "passed"
    assert report["ok"] is True
    assert report["requireProduction"] is True
    assert report["errors"] == []
    assert report["checks"]["contractReady"] is True
    assert report["checks"]["productionReady"] is True
    assert report["checks"]["generationProbeReady"] is True
    assert report["checks"]["productionReadinessChecks"][
        "productionStateContractReady"
    ] is True


def test_llm_shard_generation_probe_accepts_whitespace_token_output() -> None:
    ready, error, audit = _llm_shard_generation_probe_ready(
        {
            "generationProbe": {
                "schemaVersion": 1,
                "abi": "cai-llama-cpp-generation-probe-v1",
                "ready": True,
                "modelId": "cai-network/TinyLlama-1.1B-Chat-v1.0-GGUF",
                "outputText": "\n",
                "outputTokenCount": 1,
                "realModelExecution": True,
                "realLayerExecution": True,
            },
            "metrics": {"backendMode": "assignment_artifact_engine"},
        },
        expected_model_id="cai-network/TinyLlama-1.1B-Chat-v1.0-GGUF",
    )

    assert ready is True
    assert error is None
    assert audit["outputTextPreview"] == "\n"
    assert audit["outputTokenCount"] == 1
    assert audit["realLayerExecution"] is True


def test_cai_owned_llm_shard_conformance_cli_writes_json_report() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch.dict(
        "os.environ",
        {"CAI_LLM_SHARD_ADAPTER": "smoke_runner"},
        clear=False,
    ):
        report_path = Path(tempdir) / "llm-shard-conformance.json"
        output_text = handle_cai_owned_llm_shard_conformance(
            model_id="cai-network/Qwen3-0.6B-GGUF",
            payload="cli-conformance",
            json_report=str(report_path),
        )
        output = json.loads(output_text)
        saved = json.loads(report_path.read_text(encoding="utf-8"))

    assert output["status"] == "passed"
    assert output["checks"]["contractReady"] is True
    assert saved["status"] == output["status"]
    assert saved["selfTest"]["backendMode"] == "smoke_runner"


def test_cai_owned_runtime_derives_coordinator_url_from_peer_map() -> None:
    signing_material = _signing_material()
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout: float):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"status": "running", "sessionId": "session-runtime"})

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        session_id, _batch_id = _create_received_batch(
            policy,
            payload=b"runtime-input",
            metadata={
                "peerCaiUrlsByNode": {
                    "node-a": ["http://coordinator:52415"],
                }
            },
        )
        result = run_cai_owned_shard_runtime_once(
            CaiOwnedShardRuntimeConfig(
                node_id="node-b",
                runtime_id="runtime-derived-coordinator",
                signing_material=signing_material,
                policy=policy,
            ),
            DeterministicBytesShardAdapter(prefix=b"done:"),
        )

    assert result["status"] == "processed"
    assert result["completion"]["coordinatorResponse"]["status"] == "running"
    assert captured["url"] == (
        "http://coordinator:52415/v1/cai/transport/sessions/"
        f"{session_id}/shard-receipts"
    )
    assert captured["body"]["nodeId"] == "node-b"
    assert captured["body"]["signerNodeId"] == "node-b"
    assert (
        captured["body"]["signature"]["public_key_b64"]
        == signing_material["public_key_b64"]
    )


def test_cai_owned_runtime_skips_loopback_for_remote_coordinator_receipt() -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout: float):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"status": "queued", "sessionId": "session-runtime"})

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        session_id, batch_id = _create_received_batch(
            policy,
            payload=b"runtime-input",
            metadata={
                "stageId": "caistage_runtime_remote",
                "peerCaiUrlsByNode": {
                    "node-a": [
                        "http://127.0.0.1:52435",
                        "cai-overlay:http://relay:52415?targetNodeId=node-a",
                    ],
                },
            },
        )
        result = run_cai_owned_shard_runtime_once(
            CaiOwnedShardRuntimeConfig(
                node_id="node-b",
                runtime_id="runtime-overlay-coordinator",
                policy=policy,
            ),
            DeterministicBytesShardAdapter(prefix=b"done:"),
        )

    body = captured["body"]
    assert result["status"] == "processed"
    assert captured["url"] == "http://relay:52415/v1/cai/transport/overlay/send"
    assert isinstance(body, dict)
    assert body["kind"] == "shard_receipt"
    assert body["targetNodeId"] == "node-a"
    assert body["sessionId"] == session_id
    assert body["payload"]["batchIds"] == [batch_id]
    assert body["payload"]["stageIds"] == ["caistage_runtime_remote"]


def test_cai_owned_runtime_keeps_original_coordinator_for_forwarded_receipt() -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout: float):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"status": "queued", "sessionId": "session-runtime"})

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        session_id, batch_id = _create_received_batch(
            policy,
            payload=b"runtime-input",
            metadata={
                "stageId": "caistage_runtime_forwarded",
                "requesterNodeId": "node-user",
                "coordinatorNodeId": "node-user",
                "peerCaiUrlsByNode": {
                    "node-a": ["http://previous-worker:52415"],
                    "node-user": [
                        "cai-overlay:http://relay:52415?targetNodeId=node-user"
                    ],
                },
            },
        )
        result = run_cai_owned_shard_runtime_once(
            CaiOwnedShardRuntimeConfig(
                node_id="node-b",
                runtime_id="runtime-original-coordinator",
                policy=policy,
            ),
            DeterministicBytesShardAdapter(prefix=b"done:"),
        )

    body = captured["body"]
    assert result["status"] == "processed"
    assert captured["url"] == "http://relay:52415/v1/cai/transport/overlay/send"
    assert body["kind"] == "shard_receipt"
    assert body["targetNodeId"] == "node-user"
    assert body["sessionId"] == session_id
    assert body["payload"]["batchIds"] == [batch_id]
    assert body["payload"]["stageIds"] == ["caistage_runtime_forwarded"]


def test_cai_owned_runtime_embeds_forwarded_receipt_in_output_envelope() -> None:
    captured: list[dict[str, object]] = []

    def fake_urlopen(request, timeout: float):
        captured.append(
            {
                "url": request.full_url,
                "body": json.loads(request.data.decode("utf-8")),
                "timeout": timeout,
            }
        )
        return _FakeResponse({"status": "running", "sessionId": "session-runtime"})

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        session_id, batch_id = _create_received_batch(
            policy,
            payload=b"runtime-input",
            phase="prefill_activation_batches",
            metadata={
                "stageId": "caistage_runtime_embedded",
                "nextSinkNodeId": "node-a",
                "peerCaiUrlsByNode": {
                    "node-a": ["http://coordinator:52415"],
                },
            },
        )
        result = run_cai_owned_shard_runtime_once(
            CaiOwnedShardRuntimeConfig(
                node_id="node-b",
                runtime_id="runtime-embedded-receipt",
                signing_material=_signing_material(),
                policy=policy,
            ),
            DeterministicBytesShardAdapter(prefix=b"done:"),
        )

    output_posts = [
        item for item in captured if str(item["url"]).endswith("/batch-envelopes")
    ]
    receipt_posts = [
        item for item in captured if str(item["url"]).endswith("/shard-receipts")
    ]
    output_metadata = output_posts[0]["body"]["metadata"]
    embedded_receipts = output_metadata["upstreamShardReceipts"]

    assert result["status"] == "processed"
    assert output_posts[0]["body"]["sessionId"] == session_id
    assert receipt_posts[0]["body"]["batchIds"] == [batch_id]
    assert embedded_receipts[0]["nodeId"] == "node-b"
    assert embedded_receipts[0]["batchIds"] == [batch_id]
    assert embedded_receipts[0]["stageIds"] == ["caistage_runtime_embedded"]
    assert embedded_receipts[0]["activationBatchCount"] == 1
    assert embedded_receipts[0]["signature"]["public_key_b64"]


def test_peer_urls_for_node_skips_loopback_for_remote_output_target() -> None:
    config = CaiOwnedShardRuntimeConfig(
        node_id="node-b",
        runtime_id="runtime-peer-url-filter",
    )
    work_item = {
        "batch": {
            "metadata": {
                "peerCaiUrlsByNode": {
                    "node-a": [
                        "http://127.0.0.1:52435",
                        "cai-overlay:http://relay:52415?targetNodeId=node-a",
                    ],
                },
            },
        },
    }

    assert _peer_urls_for_node(config, work_item, "node-a") == [
        "cai-overlay:http://relay:52415?targetNodeId=node-a",
    ]


def test_cai_owned_runtime_keeps_processed_batch_when_receipt_submit_fails() -> None:
    def fake_urlopen(_request, timeout: float):
        raise OSError("coordinator unavailable")

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        _session_id, _batch_id = _create_received_batch(
            policy,
            payload=b"runtime-input",
            metadata={
                "peerCaiUrlsByNode": {
                    "node-a": ["http://coordinator:52415"],
                }
            },
        )
        result = run_cai_owned_shard_runtime_once(
            CaiOwnedShardRuntimeConfig(
                node_id="node-b",
                runtime_id="runtime-coordinator-failure",
                policy=policy,
            ),
            DeterministicBytesShardAdapter(prefix=b"done:"),
        )
        batch = list_cai_owned_transport_sessions(policy)[0].batch_records[0]

    assert result["status"] == "processed"
    assert result["completion"]["coordinatorResponse"]["status"] == "failed"
    assert (
        "coordinator unavailable"
        in result["completion"]["coordinatorResponse"]["error"]
    )
    assert batch["status"] == "processed"


def test_cai_owned_llm_shard_adapter_interface_receives_frame() -> None:
    payload = b"prefill-frame"
    metadata = build_cai_owned_transport_frame_metadata(
        model_id="cai-network/Qwen3-0.6B-GGUF",
        frame_kind="activation",
        tokenizer_config_hash="ab" * 32,
        layer_start=0,
        layer_end=14,
        token_start=0,
        token_end=4,
        dtype="bytes",
        shape=[len(payload)],
        sequence=1,
        payload_sha256_hex=hashlib.sha256(payload).hexdigest(),
    )
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        session_id, batch_id = _create_received_batch(
            policy,
            payload=payload,
            phase="prefill_activation_batches",
            metadata=metadata,
        )
        config = CaiOwnedShardRuntimeConfig(
            node_id="node-b",
            runtime_id="runtime-llm-frame",
            policy=policy,
        )
        adapter = _SpyLlmShardAdapter()

        result = run_cai_owned_shard_runtime_once(config, adapter)
        output_payload = read_cai_owned_transport_batch_output_payload(
            session_id,
            batch_id,
            policy,
        )
        batch = list_cai_owned_transport_sessions(policy)[0].batch_records[0]

    assert result["status"] == "processed"
    assert output_payload == b"prefill:prefill-frame"
    assert [call[0] for call in adapter.calls] == [
        "load_shard",
        "process_prefill",
        "finalize",
    ]
    assert batch["metrics"]["adapterId"] == "spy-llm"
    assert batch["metrics"]["adapterVersion"] == "spy-llm/0.1"
    assert batch["metrics"]["spyLoaded"] is True
    assert batch["metrics"]["spyFinalized"] is True
    assert batch["metrics"]["frameKind"] == "activation"
    assert batch["metrics"]["modelId"] == "cai-network/Qwen3-0.6B-GGUF"
    assert batch["metrics"]["layerStart"] == 0
    assert batch["metrics"]["layerEnd"] == 14


def test_cai_owned_llm_shard_adapter_defers_finalize_for_persistent_backend() -> None:
    payload = b"prefill-frame"
    metadata = build_cai_owned_transport_frame_metadata(
        model_id="cai-network/Qwen3-0.6B-GGUF",
        frame_kind="activation",
        tokenizer_config_hash="ab" * 32,
        layer_start=0,
        layer_end=14,
        token_start=0,
        token_end=4,
        dtype="bytes",
        shape=[len(payload)],
        sequence=1,
        payload_sha256_hex=hashlib.sha256(payload).hexdigest(),
    )
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        session_id, batch_id = _create_received_batch(
            policy,
            payload=payload,
            phase="prefill_activation_batches",
            metadata=metadata,
        )
        config = CaiOwnedShardRuntimeConfig(
            node_id="node-b",
            runtime_id="runtime-llm-frame",
            policy=policy,
        )
        adapter = _PersistentSpyLlmShardAdapter()

        result = run_cai_owned_shard_runtime_once(config, adapter)
        output_payload = read_cai_owned_transport_batch_output_payload(
            session_id,
            batch_id,
            policy,
        )
        batch = list_cai_owned_transport_sessions(policy)[0].batch_records[0]

    assert result["status"] == "processed"
    assert output_payload == b"prefill:prefill-frame"
    assert [call[0] for call in adapter.calls] == [
        "load_shard",
        "process_prefill",
    ]
    assert batch["metrics"]["backendFinalizeDeferred"] is True
    assert batch["metrics"]["backendRetainedResidentShard"] is True
    assert batch["metrics"]["backendFinalized"] is False


def test_cai_owned_shard_runtime_forwards_output_envelope_to_next_peer() -> None:
    signing_material = _signing_material()
    captured: list[dict[str, object]] = []

    def fake_urlopen(request, timeout: float):
        captured.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "body": json.loads(request.data.decode("utf-8")),
            },
        )
        return _FakeResponse({"status": "running", "sessionId": "session-forward"})

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        session_id, _batch_id = _create_received_batch(
            policy,
            payload=b"forward-input",
            metadata={
                "outputRoutePlan": [
                    {
                        "sinkNodeId": "node-a",
                        "phase": "prefill_activation_batches",
                        "sequence": 9,
                        "stageId": "stage-forward",
                    },
                    {
                        "sinkNodeId": "node-b",
                        "phase": "decode_activation_batches",
                        "sequence": 10,
                        "stageId": "stage-return",
                    },
                ],
            },
        )
        config = CaiOwnedShardRuntimeConfig(
            node_id="node-b",
            runtime_id="runtime-forward",
            output_peer_cai_urls_by_node={"node-a": ["http://node-a:52415"]},
            signing_material=signing_material,
            policy=policy,
        )

        result = run_cai_owned_shard_runtime_once(
            config,
            DeterministicBytesShardAdapter(prefix=b"done:"),
        )

    assert result["status"] == "processed"
    assert result["outputForward"]["status"] == "submitted"
    assert result["outputForward"]["sinkNodeId"] == "node-a"
    assert result["outputForward"]["response"]["status"] == "running"
    forward_call = next(
        item for item in captured if str(item["url"]).endswith("/batch-envelopes")
    )
    assert forward_call["url"] == (
        "http://node-a:52415/v1/cai/transport/sessions/"
        f"{session_id}/batch-envelopes"
    )
    assert forward_call["timeout"] == 5.0
    assert forward_call["body"]["sessionId"] == session_id
    assert forward_call["body"]["sourceNodeId"] == "node-b"
    assert forward_call["body"]["sinkNodeId"] == "node-a"
    assert forward_call["body"]["phase"] == "prefill_activation_batches"
    assert forward_call["body"]["sequence"] == 9
    assert cai_owned_transport_batch_payload_bytes(forward_call["body"]) == (
        b"done:forward-input"
    )
    assert forward_call["body"]["metadata"]["payloadRole"] == "shard_output"
    assert forward_call["body"]["metadata"]["forwardedByRuntime"] is True
    assert forward_call["body"]["metadata"]["stageId"] == "stage-forward"
    assert forward_call["body"]["metadata"]["nextSinkNodeId"] == "node-b"
    assert forward_call["body"]["metadata"]["outputRoutePlan"][0]["stageId"] == (
        "stage-return"
    )
    signed_valid, signed_error = validate_cai_owned_transport_batch_envelope(
        forward_call["body"],
        session_id=session_id,
        participant_node_ids=["node-a", "node-b"],
        require_signature=True,
    )
    assert signed_valid is True
    assert signed_error is None
    assert forward_call["body"]["signerNodeId"] == "node-b"


def test_task_level_http_inference_adapter_calls_endpoint_on_final_stage() -> None:
    captured: list[dict[str, object]] = []

    def fake_urlopen(request, timeout: float):  # noqa: ANN001
        captured.append(
            {
                "method": request.get_method(),
                "url": request.full_url,
                "timeout": timeout,
                "body": (
                    json.loads(request.data.decode("utf-8"))
                    if getattr(request, "data", None)
                    else None
                ),
            }
        )
        if request.get_method() == "GET":
            return _FakeResponse({"status": "ok", "ready": True})
        return _FakeResponse(
            {
                "choices": [
                    {"message": {"content": "network answer"}},
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            }
        )

    payload = json.dumps(
        {"messages": [{"role": "user", "content": "ping network"}]},
        sort_keys=True,
    ).encode("utf-8")
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.cai_owned_runtime.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        session_id, batch_id = _create_received_batch(
            policy,
            payload=payload,
            metadata={
                "outputRoutePlan": [
                    {
                        "sinkNodeId": "node-a",
                        "phase": "decode_activation_batches",
                        "sequence": 2,
                        "finalOutput": True,
                    }
                ],
            },
        )
        result = run_cai_owned_shard_runtime_once(
            CaiOwnedShardRuntimeConfig(
                node_id="node-b",
                runtime_id="runtime-task-http",
                policy=policy,
            ),
            TaskLevelHttpInferenceAdapter(
                endpoint_url="http://127.0.0.1:8080/v1/chat/completions",
                model_id="cai-network/Qwen3-0.6B-GGUF",
                timeout_sec=5,
                max_tokens=16,
                temperature=0.0,
            ),
        )
        output = read_cai_owned_transport_batch_output_payload(
            session_id,
            batch_id,
            policy,
        )
        batch = list_cai_owned_transport_sessions(policy)[0].batch_records[0]

    assert result["status"] == "processed"
    assert output == b"network answer"
    assert captured[0]["method"] == "GET"
    assert captured[0]["url"] == "http://127.0.0.1:8080/health"
    assert captured[1]["method"] == "POST"
    assert captured[1]["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    assert captured[1]["body"]["model"] == "cai-network/Qwen3-0.6B-GGUF"
    assert captured[1]["body"]["max_tokens"] == 16
    assert captured[1]["body"]["messages"][0]["content"] == "ping network"
    assert batch["metrics"]["adapterId"] == TASK_LEVEL_HTTP_INFERENCE_ADAPTER_ID
    assert batch["metrics"]["adapterVersion"] == TASK_LEVEL_HTTP_INFERENCE_ADAPTER_VERSION
    assert batch["metrics"]["taskLevelFallback"] is True
    assert batch["metrics"]["modelParallel"] is False
    assert batch["metrics"]["inferenceExecutor"] is True
    assert batch["metrics"]["promptTokenCount"] == 3
    assert batch["metrics"]["completionTokenCount"] == 2
    assert batch["metrics"]["totalTokenCount"] == 5
    assert batch["runtimeAudit"]["adapterId"] == TASK_LEVEL_HTTP_INFERENCE_ADAPTER_ID


def test_task_level_http_inference_adapter_passes_through_non_final_stage() -> None:
    captured: list[str] = []

    def fake_urlopen(request, timeout: float):  # noqa: ANN001
        _ = timeout
        captured.append(request.full_url)
        if request.get_method() == "GET":
            return _FakeResponse({"status": "ok", "ready": True})
        raise AssertionError("Non-final task-level stage must not call inference.")

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.cai_owned_runtime.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        session_id, batch_id = _create_received_batch(
            policy,
            payload=b"prompt to carry",
            metadata={
                "outputRoutePlan": [
                    {
                        "sinkNodeId": "node-a",
                        "phase": "decode_activation_batches",
                        "sequence": 2,
                        "stageId": "stage-next",
                    }
                ],
            },
        )
        result = run_cai_owned_shard_runtime_once(
            CaiOwnedShardRuntimeConfig(
                node_id="node-b",
                runtime_id="runtime-task-http-pass",
                policy=policy,
            ),
            TaskLevelHttpInferenceAdapter(
                endpoint_url="http://127.0.0.1:8080/v1/chat/completions",
            ),
        )
        output = read_cai_owned_transport_batch_output_payload(
            session_id,
            batch_id,
            policy,
        )
        batch = list_cai_owned_transport_sessions(policy)[0].batch_records[0]

    assert result["status"] == "processed"
    assert output == b"prompt to carry"
    assert captured == ["http://127.0.0.1:8080/health"]
    assert batch["metrics"]["taskLevelPassThrough"] is True
    assert batch["metrics"]["inputTokenCount"] == 0
    assert batch["metrics"]["outputTokenCount"] == 0


def test_cai_owned_shard_runtime_retries_when_output_forward_fails() -> None:
    def fake_urlopen(request, timeout: float):
        _ = request, timeout
        raise OSError("route unavailable")

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        _session_id, batch_id = _create_received_batch(
            policy,
            payload=b"forward-input",
            metadata={
                "outputRoutePlan": [
                    {
                        "sinkNodeId": "node-a",
                        "phase": "prefill_activation_batches",
                        "sequence": 9,
                    },
                ],
            },
        )
        result = run_cai_owned_shard_runtime_once(
            CaiOwnedShardRuntimeConfig(
                node_id="node-b",
                runtime_id="runtime-forward-fails",
                output_peer_cai_urls_by_node={"node-a": ["http://node-a:52415"]},
                policy=policy,
            ),
            DeterministicBytesShardAdapter(prefix=b"done:"),
        )
        batch = list_cai_owned_transport_sessions(policy)[0].batch_records[0]

    assert result["status"] == "retry_scheduled"
    assert result["failure"]["batchId"] == batch_id
    assert result["failure"]["retryScheduled"] is True
    assert batch["status"] == "received"
    assert batch["retryable"] is True
    assert "CAI-owned output forward to node-a failed" in batch["lastError"]
    assert "route unavailable" in batch["lastError"]
    assert "processedAt" not in batch


def test_cai_owned_two_runtime_chain_returns_final_envelope_and_proof() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        offer = build_cai_owned_transport_session_offer(
            instance_id="instance-two-runtime",
            participant_node_ids=["node-user", "node-a", "node-b"],
            executor_node_ids=["node-a", "node-b"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-two-runtime",
            source_node_id="node-user",
        )
        create_cai_owned_transport_session_from_offer(
            offer,
            session_id=offer["sessionId"],
            local_node_id="node-user",
            policy=policy,
        )
        initial_payload = b"user-prompt"
        initial_metadata = build_cai_owned_transport_frame_metadata(
            model_id="cai-network/Qwen3-0.6B-GGUF",
            frame_kind="activation",
            tokenizer_config_hash="cd" * 32,
            layer_start=0,
            layer_end=14,
            token_start=0,
            token_end=2,
            dtype="bytes",
            shape=[len(initial_payload)],
            sequence=0,
            payload_sha256_hex=hashlib.sha256(initial_payload).hexdigest(),
        )
        first_envelope = build_cai_owned_transport_batch_envelope(
            session_id=offer["sessionId"],
            phase="prefill_activation_batches",
            source_node_id="node-user",
            sink_node_id="node-a",
            sequence=0,
            payload=initial_payload,
            metadata=initial_metadata,
        )
        record_cai_owned_transport_batch_envelope(
            offer["sessionId"],
            first_envelope,
            local_node_id="node-a",
            policy=policy,
        )
        node_a_result = run_cai_owned_shard_runtime_once(
            CaiOwnedShardRuntimeConfig(
                node_id="node-a",
                runtime_id="runtime-node-a",
                policy=policy,
            ),
            DeterministicBytesShardAdapter(prefix=b"a:"),
        )
        node_a_output = read_cai_owned_transport_batch_output_payload(
            session_id=offer["sessionId"],
            batch_id=first_envelope["batchId"],
            policy=policy,
        )
        decode_metadata = build_cai_owned_transport_frame_metadata(
            model_id="cai-network/Qwen3-0.6B-GGUF",
            frame_kind="decode",
            tokenizer_config_hash="cd" * 32,
            layer_start=14,
            layer_end=28,
            token_start=0,
            token_end=2,
            dtype="bytes",
            shape=[len(node_a_output)],
            sequence=1,
            payload_sha256_hex=hashlib.sha256(node_a_output).hexdigest(),
        )
        decode_metadata["previousBatchId"] = first_envelope["batchId"]
        to_node_b = build_cai_owned_transport_batch_envelope(
            session_id=offer["sessionId"],
            phase="decode_activation_batches",
            source_node_id="node-a",
            sink_node_id="node-b",
            sequence=1,
            payload=node_a_output,
            metadata=decode_metadata,
        )
        record_cai_owned_transport_batch_envelope(
            offer["sessionId"],
            to_node_b,
            local_node_id="node-b",
            policy=policy,
        )
        node_b_result = run_cai_owned_shard_runtime_once(
            CaiOwnedShardRuntimeConfig(
                node_id="node-b",
                runtime_id="runtime-node-b",
                policy=policy,
            ),
            DeterministicBytesShardAdapter(prefix=b"b:"),
        )
        final_envelope = build_cai_owned_transport_output_batch_envelope(
            session_id=offer["sessionId"],
            source_batch_id=to_node_b["batchId"],
            sink_node_id="node-user",
            metadata={"nextStage": "requester"},
            policy=policy,
        )
        completed = complete_cai_owned_transport_session(
            offer["sessionId"],
            policy=policy,
        )

    assert node_a_result["status"] == "processed"
    assert node_b_result["status"] == "processed"
    assert cai_owned_transport_batch_payload_bytes(final_envelope) == b"b:a:user-prompt"
    assert final_envelope["sourceNodeId"] == "node-b"
    assert final_envelope["sinkNodeId"] == "node-user"
    assert completed.status == "completed"
    assert completed.proof is not None
    assert completed.proof["executorNodeIds"] == ["node-a", "node-b"]
    assert completed.proof["executionAudit"]["verified"] is True
    assert [item["nodeId"] for item in completed.proof["shardReceipts"]] == [
        "node-a",
        "node-b",
    ]


def test_external_llama_cpp_shard_adapter_processes_handoff_frame() -> None:
    payload = b"activation-state"
    payload_hash = hashlib.sha256(payload).hexdigest()
    handoff = build_cai_owned_llm_handoff_metadata(
        model_id="cai-network/Qwen3-0.6B-GGUF",
        backend="llama.cpp-patched",
        backend_version="llama.cpp/cai-shard-0.1",
        tokenizer_config_hash="ab" * 32,
        layer_start=0,
        layer_end=14,
        token_start=0,
        token_end=8,
        tensor_dtype="f16",
        tensor_shape=[1, 8, 768],
        tensor_encoding="ggml-tensor-v1",
        tensor_sha256_hex=payload_hash,
    )
    metadata = build_cai_owned_transport_frame_metadata(
        model_id="cai-network/Qwen3-0.6B-GGUF",
        frame_kind="activation",
        tokenizer_config_hash="ab" * 32,
        layer_start=0,
        layer_end=14,
        token_start=0,
        token_end=8,
        dtype="f16",
        shape=[1, 8, 768],
        payload_sha256_hex=payload_hash,
    )
    metadata["llmHandoff"] = handoff

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        session_id, batch_id = _create_received_batch(
            policy,
            payload=payload,
            phase="prefill_activation_batches",
            metadata=metadata,
        )
        result = run_cai_owned_shard_runtime_once(
            CaiOwnedShardRuntimeConfig(
                node_id="node-b",
                runtime_id="runtime-external-llama",
                require_production_llm_handoff=True,
                policy=policy,
            ),
            ExternalLlamaCppShardAdapter(
                command=_smoke_runner_command(),
                env={
                    "PYTHONPATH": str(SRC_ROOT),
                    "CAI_SHARD_SMOKE_PREFILL_PREFIX": "llm-shard:",
                },
                timeout_sec=10,
            ),
        )
        output = read_cai_owned_transport_batch_output_payload(
            session_id,
            batch_id,
            policy,
        )
        batch = list_cai_owned_transport_sessions(policy)[0].batch_records[0]

    assert result["status"] == "processed"
    assert output == b"llm-shard:activation-state"
    assert batch["metrics"]["adapterId"] == "llama.cpp-external-shard"
    assert batch["metrics"]["backendAction"] == "process_prefill"
    assert "activation_handoff" in batch["metrics"]["backendCapabilities"]
    assert batch["metrics"]["backendMode"] == "smoke_runner"
    assert batch["metrics"]["patchBoundaryVerified"] is True
    assert batch["metrics"]["patchBoundaryPatchId"] == (
        "cai-llama-cpp-shard-smoke-runner"
    )
    assert batch["runtimeAudit"]["adapterId"] == "llama.cpp-external-shard"


def test_external_llama_cpp_shard_adapter_processes_http_endpoint() -> None:
    payload = b"prompt-tokens"
    runtime_metadata = {
        "modelId": "cai-network/Qwen3-0.6B-GGUF",
        "totalLayerCount": 28,
        "hiddenSize": 1024,
        "activationDtype": "f16",
        "tensorEncoding": "ggml-tensor-v1",
        "tokenizerConfigHash": "ab" * 32,
        "modelSha256Hex": "12" * 32,
        "backend": "llama.cpp-patched",
        "backendVersion": "llama.cpp/cai-shard-0.1",
    }
    metadata = build_cai_owned_llm_shard_frame_metadata_from_runtime(
        model_id="cai-network/Qwen3-0.6B-GGUF",
        runtime_metadata=runtime_metadata,
        payload=payload,
        frame_kind="activation",
        layer_start=0,
        layer_end=14,
        token_start=0,
        token_end=4,
        sequence=0,
    )
    metadata["nextFrameTemplate"] = build_cai_owned_llm_shard_frame_metadata_from_runtime(
        model_id="cai-network/Qwen3-0.6B-GGUF",
        runtime_metadata=runtime_metadata,
        payload=b"",
        frame_kind="decode",
        layer_start=14,
        layer_end=28,
        token_start=4,
        token_end=5,
        sequence=1,
    )
    frame = CaiOwnedShardFrame(
        session_id="caiot_http_adapter",
        batch_id="caibatch_http_adapter_prefill",
        phase="prefill_activation_batches",
        source_node_id="node-user",
        sink_node_id="node-a",
        sequence=0,
        model_id="cai-network/Qwen3-0.6B-GGUF",
        frame_kind="activation",
        layer_start=0,
        layer_end=14,
        token_start=0,
        token_end=4,
        payload_sha256_hex=hashlib.sha256(payload).hexdigest(),
        payload=payload,
        metadata=metadata,
    )
    patch_boundary = build_llama_cpp_external_shard_patch_boundary(
        backend_version="llama.cpp/cai-http",
        patch_id="cai-llama-cpp-shard-http",
    )
    seen_actions: list[str] = []

    def fake_urlopen(http_request, timeout):  # noqa: ANN001
        assert timeout == 10
        assert http_request.full_url == "http://127.0.0.1:9257/cai-shard"
        request_payload = json.loads(http_request.data.decode("utf-8"))
        seen_actions.append(request_payload["action"])
        if request_payload["action"] == "load_shard":
            return _FakeResponse(
                {
                    "status": "ready",
                    "capabilities": patch_boundary["capabilities"],
                    "patchBoundary": patch_boundary,
                    "metrics": {"backendMode": "http_bridge"},
                }
            )
        if request_payload["action"] == "finalize":
            return _FakeResponse(
                {"status": "ok", "metrics": {"backendMode": "http_bridge"}}
            )
        input_payload = base64.b64decode(request_payload["payloadBase64"])
        output = b"http-state:" + input_payload
        output_hash = hashlib.sha256(output).hexdigest()
        contract = request_payload["outputContract"]
        template = json.loads(json.dumps(contract["frameMetadataTemplate"]))
        template["payloadSha256Hex"] = output_hash
        handoff = dict(template["llmHandoff"])
        tensor = dict(handoff["tensor"])
        tensor["sha256Hex"] = output_hash
        handoff["tensor"] = tensor
        template["llmHandoff"] = handoff
        return _FakeResponse(
            {
                "status": "ok",
                "outputPayloadBase64": base64.b64encode(output).decode("ascii"),
                "outputPayloadSha256Hex": output_hash,
                "outputFrameMetadata": template,
                "metrics": {
                    "backendAction": request_payload["action"],
                    "backendMode": "http_bridge",
                },
            }
        )

    adapter = ExternalLlamaCppShardAdapter(
        endpoint_url="http://127.0.0.1:9257/cai-shard",
        timeout_sec=10,
    )
    with patch("cai_compute_chain.cai_owned_runtime.urlopen", fake_urlopen):
        load_metrics = adapter.load_shard(frame)
        result = adapter.process_prefill(frame)
        finalize_metrics = adapter.finalize(frame, result)

    assert seen_actions == ["load_shard", "process_prefill", "finalize"]
    assert load_metrics["backendMode"] == "http_bridge"
    assert result.output_payload == b"http-state:prompt-tokens"
    assert result.metrics["backendAction"] == "process_prefill"
    assert result.output_metadata["frameKind"] == "decode"
    assert finalize_metrics["backendMode"] == "http_bridge"


def test_external_llama_cpp_shard_adapter_uses_local_file_io_for_large_payloads() -> None:
    payload = b"activation-state:" + (b"0123456789abcdef" * 4096)
    payload_hash = hashlib.sha256(payload).hexdigest()
    handoff = build_cai_owned_llm_handoff_metadata(
        model_id="cai-network/Qwen3-0.6B-GGUF",
        backend="llama.cpp-patched",
        backend_version="llama.cpp/cai-shard-0.1",
        tokenizer_config_hash="ab" * 32,
        layer_start=0,
        layer_end=14,
        token_start=0,
        token_end=8,
        tensor_dtype="f16",
        tensor_shape=[1, 8, 768],
        tensor_encoding="ggml-tensor-v1",
        tensor_sha256_hex=payload_hash,
    )
    metadata = build_cai_owned_transport_frame_metadata(
        model_id="cai-network/Qwen3-0.6B-GGUF",
        frame_kind="activation",
        tokenizer_config_hash="ab" * 32,
        layer_start=0,
        layer_end=14,
        token_start=0,
        token_end=8,
        dtype="f16",
        shape=[1, 8, 768],
        payload_sha256_hex=payload_hash,
    )
    metadata["llmHandoff"] = handoff

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        io_root = Path(tempdir) / "adapter-file-io"
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        session_id, batch_id = _create_received_batch(
            policy,
            payload=payload,
            phase="prefill_activation_batches",
            metadata=metadata,
        )
        result = run_cai_owned_shard_runtime_once(
            CaiOwnedShardRuntimeConfig(
                node_id="node-b",
                runtime_id="runtime-external-file-io",
                require_production_llm_handoff=True,
                policy=policy,
            ),
            ExternalLlamaCppShardAdapter(
                command=_smoke_runner_command(),
                env={
                    "PYTHONPATH": str(SRC_ROOT),
                    "CAI_SHARD_SMOKE_REQUIRE_FILE_INPUT": "1",
                    "CAI_SHARD_SMOKE_PREFER_OUTPUT_FILE": "1",
                },
                timeout_sec=10,
                file_io_root=str(io_root),
                file_io_threshold_bytes=1,
            ),
        )
        output = read_cai_owned_transport_batch_output_payload(
            session_id,
            batch_id,
            policy,
        )
        batch = list_cai_owned_transport_sessions(policy)[0].batch_records[0]
        io_children = list(io_root.iterdir()) if io_root.exists() else []

    assert result["status"] == "processed"
    assert output == b"prefill-state:" + payload
    assert batch["metrics"]["backendMode"] == "smoke_runner"
    assert batch["metrics"]["patchBoundaryVerified"] is True
    assert io_children == []


def test_external_llama_cpp_shard_adapter_sends_output_contract_for_next_frame() -> None:
    payload = b"prompt-tokens"
    runtime_metadata = {
        "modelId": "cai-network/Qwen3-0.6B-GGUF",
        "totalLayerCount": 28,
        "hiddenSize": 1024,
        "activationDtype": "f16",
        "tensorEncoding": "ggml-tensor-v1",
        "tokenizerConfigHash": "ab" * 32,
        "backend": "llama.cpp-patched",
        "backendVersion": "llama.cpp/cai-shard-0.1",
        "preferredFilename": "Qwen3-0.6B-Q8_0.gguf",
        "quantization": "Q8_0",
        "contextLength": 32768,
    }
    metadata = build_cai_owned_llm_shard_frame_metadata_from_runtime(
        model_id="cai-network/Qwen3-0.6B-GGUF",
        runtime_metadata=runtime_metadata,
        payload=payload,
        frame_kind="activation",
        layer_start=0,
        layer_end=14,
        token_start=0,
        token_end=4,
        sequence=0,
    )
    metadata["nextFrameTemplate"] = build_cai_owned_llm_shard_frame_metadata_from_runtime(
        model_id="cai-network/Qwen3-0.6B-GGUF",
        runtime_metadata=runtime_metadata,
        payload=b"",
        frame_kind="decode",
        layer_start=14,
        layer_end=28,
        token_start=4,
        token_end=5,
        sequence=1,
    )
    metadata["nextFrameTemplate"]["stageId"] = "expected-stage"

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        capture_path = Path(tempdir) / "external-requests.jsonl"
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        session_id, batch_id = _create_received_batch(
            policy,
            payload=payload,
            phase="prefill_activation_batches",
            metadata=metadata,
        )
        result = run_cai_owned_shard_runtime_once(
            CaiOwnedShardRuntimeConfig(
                node_id="node-b",
                runtime_id="runtime-external-output-contract",
                require_production_llm_handoff=True,
                policy=policy,
            ),
            ExternalLlamaCppShardAdapter(
                command=[sys.executable, "-c", _output_contract_backend_code()],
                env={"CAI_CAPTURE_REQUEST_PATH": str(capture_path)},
                timeout_sec=10,
            ),
        )
        output = read_cai_owned_transport_batch_output_payload(
            session_id,
            batch_id,
            policy,
        )
        captured = [
            json.loads(line)
            for line in capture_path.read_text(encoding="utf-8").splitlines()
        ]

    process_request = next(
        item for item in captured if item.get("action") == "process_prefill"
    )
    contract = process_request["outputContract"]
    requirements = process_request["productionRequirements"]
    shard_spec = process_request["shardSpec"]
    template = contract["frameMetadataTemplate"]
    assert result["status"] == "processed"
    assert output == b"state:prompt-tokens"
    assert contract["requiresOutputFrameMetadata"] is True
    assert contract["requiresFinalOutput"] is False
    assert requirements["handoffAbi"] == "cai-llama-cpp-external-shard-v1"
    assert requirements["shardSpecAbi"] == "cai-llama-cpp-shard-spec-v1"
    assert requirements["patchBoundaryAbi"] == "cai-llama-cpp-shard-patch-boundary-v1"
    assert requirements["productionStateContractAbi"] == (
        "cai-llama-cpp-production-state-contract-v1"
    )
    assert requirements["requiresRealStateContract"] is True
    assert requirements["requiresShardOnlyLoading"] is True
    assert requirements["forbidFullModelFallback"] is True
    assert requirements["requiredCapabilities"] == [
        "layer_range_execution",
        "activation_handoff",
        "decode_state_handoff",
        "output_frame_metadata",
    ]
    assert requirements["requiredProductionCapabilities"] == [
        "gguf_layer_execution",
        "real_activation_state",
        "real_decode_state",
    ]
    assert shard_spec["abi"] == "cai-llama-cpp-shard-spec-v1"
    assert shard_spec["modelFormat"] == "gguf"
    assert shard_spec["layerStart"] == 0
    assert shard_spec["layerEnd"] == 14
    assert shard_spec["totalLayerCount"] == 28
    assert shard_spec["extraMetadata"]["preferredFilename"] == "Qwen3-0.6B-Q8_0.gguf"
    assert shard_spec["extraMetadata"]["quantization"] == "Q8_0"
    assert shard_spec["extraMetadata"]["contextLength"] == 32768
    assert template["layerStart"] == 14
    assert template["layerEnd"] == 28
    assert template["payloadSha256Hex"] == "<computed-output-sha256>"
    assert template["llmHandoff"]["tensor"]["sha256Hex"] == (
        "<computed-output-sha256>"
    )


def test_external_llama_cpp_shard_adapter_rejects_wrong_output_frame_template() -> None:
    payload = b"prompt-tokens"
    runtime_metadata = {
        "modelId": "cai-network/Qwen3-0.6B-GGUF",
        "totalLayerCount": 28,
        "hiddenSize": 1024,
        "activationDtype": "f16",
        "tensorEncoding": "ggml-tensor-v1",
        "tokenizerConfigHash": "ab" * 32,
        "backend": "llama.cpp-patched",
        "backendVersion": "llama.cpp/cai-shard-0.1",
    }
    metadata = build_cai_owned_llm_shard_frame_metadata_from_runtime(
        model_id="cai-network/Qwen3-0.6B-GGUF",
        runtime_metadata=runtime_metadata,
        payload=payload,
        frame_kind="activation",
        layer_start=0,
        layer_end=14,
        token_start=0,
        token_end=4,
        sequence=0,
    )
    metadata["nextFrameTemplate"] = build_cai_owned_llm_shard_frame_metadata_from_runtime(
        model_id="cai-network/Qwen3-0.6B-GGUF",
        runtime_metadata=runtime_metadata,
        payload=b"",
        frame_kind="decode",
        layer_start=14,
        layer_end=28,
        token_start=4,
        token_end=5,
        sequence=1,
    )
    metadata["nextFrameTemplate"]["stageId"] = "expected-stage"

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        _session_id, _batch_id = _create_received_batch(
            policy,
            payload=payload,
            phase="prefill_activation_batches",
            metadata=metadata,
        )
        result = run_cai_owned_shard_runtime_once(
            CaiOwnedShardRuntimeConfig(
                node_id="node-b",
                runtime_id="runtime-external-bad-output-contract",
                require_production_llm_handoff=True,
                policy=policy,
            ),
            ExternalLlamaCppShardAdapter(
                command=[sys.executable, "-c", _output_contract_backend_code()],
                env={"CAI_BAD_OUTPUT_TEMPLATE": "1"},
                timeout_sec=10,
            ),
        )
        batch = list_cai_owned_transport_sessions(policy)[0].batch_records[0]

    assert result["status"] == "retry_scheduled"
    assert "output metadata does not match next frame template" in batch["lastError"]
    assert "metadata.stageId differs from template" in batch["lastError"]
    assert batch["status"] == "received"


def test_external_llama_cpp_shard_adapter_requires_patched_capabilities() -> None:
    payload = b"activation-state"
    metadata = build_cai_owned_llm_shard_frame_metadata_from_runtime(
        model_id="cai-network/Qwen3-0.6B-GGUF",
        runtime_metadata={
            "modelId": "cai-network/Qwen3-0.6B-GGUF",
            "totalLayerCount": 28,
            "hiddenSize": 1024,
            "activationDtype": "f16",
            "tensorEncoding": "ggml-tensor-v1",
            "tokenizerConfigHash": "ab" * 32,
            "backend": "llama.cpp-patched",
            "backendVersion": "llama.cpp/cai-shard-0.1",
        },
        payload=payload,
        frame_kind="activation",
        layer_start=0,
        layer_end=14,
        token_start=0,
        token_end=8,
    )
    backend_code = r"""
import json
import sys

request = json.loads(sys.stdin.read() or "{}")
if request.get("action") == "load_shard":
    print(json.dumps({
        "status": "ready",
        "capabilities": ["activation_handoff"],
        "metrics": {"backendLoaded": True},
    }))
else:
    print(json.dumps({"status": "ok", "outputPayloadBase64": ""}))
"""

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        _session_id, _batch_id = _create_received_batch(
            policy,
            payload=payload,
            phase="prefill_activation_batches",
            metadata=metadata,
        )
        result = run_cai_owned_shard_runtime_once(
            CaiOwnedShardRuntimeConfig(
                node_id="node-b",
                runtime_id="runtime-external-missing-capabilities",
                require_production_llm_handoff=True,
                policy=policy,
            ),
            ExternalLlamaCppShardAdapter(
                command=[sys.executable, "-c", backend_code],
                timeout_sec=10,
            ),
        )
        batch = list_cai_owned_transport_sessions(policy)[0].batch_records[0]

    assert result["status"] == "retry_scheduled"
    assert "missing required capabilities" in batch["lastError"]
    assert "layer_range_execution" in batch["lastError"]
    assert batch["status"] == "received"


def test_external_llama_cpp_shard_adapter_requires_patch_boundary() -> None:
    payload = b"activation-state"
    metadata = build_cai_owned_llm_shard_frame_metadata_from_runtime(
        model_id="cai-network/Qwen3-0.6B-GGUF",
        runtime_metadata={
            "modelId": "cai-network/Qwen3-0.6B-GGUF",
            "totalLayerCount": 28,
            "hiddenSize": 1024,
            "activationDtype": "f16",
            "tensorEncoding": "ggml-tensor-v1",
            "tokenizerConfigHash": "ab" * 32,
            "backend": "llama.cpp-patched",
            "backendVersion": "llama.cpp/cai-shard-0.1",
        },
        payload=payload,
        frame_kind="activation",
        layer_start=0,
        layer_end=14,
        token_start=0,
        token_end=8,
    )
    backend_code = r"""
import json
import sys

request = json.loads(sys.stdin.read() or "{}")
if request.get("action") == "load_shard":
    print(json.dumps({
        "status": "ready",
        "capabilities": [
            "layer_range_execution",
            "activation_handoff",
            "decode_state_handoff",
            "output_frame_metadata",
        ],
        "metrics": {"backendLoaded": True},
    }))
else:
    print(json.dumps({"status": "ok", "outputPayloadBase64": ""}))
"""

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        _session_id, _batch_id = _create_received_batch(
            policy,
            payload=payload,
            phase="prefill_activation_batches",
            metadata=metadata,
        )
        result = run_cai_owned_shard_runtime_once(
            CaiOwnedShardRuntimeConfig(
                node_id="node-b",
                runtime_id="runtime-external-missing-patch-boundary",
                require_production_llm_handoff=True,
                policy=policy,
            ),
            ExternalLlamaCppShardAdapter(
                command=[sys.executable, "-c", backend_code],
                timeout_sec=10,
            ),
        )
        batch = list_cai_owned_transport_sessions(policy)[0].batch_records[0]

    assert result["status"] == "retry_scheduled"
    assert batch["lastError"] == (
        "External llama.cpp shard adapter patch boundary is missing."
    )
    assert batch["status"] == "received"


def test_external_llama_cpp_shard_adapter_rejects_patch_boundary_drift() -> None:
    payload = b"activation-state"
    metadata = build_cai_owned_llm_shard_frame_metadata_from_runtime(
        model_id="cai-network/Qwen3-0.6B-GGUF",
        runtime_metadata={
            "modelId": "cai-network/Qwen3-0.6B-GGUF",
            "totalLayerCount": 28,
            "hiddenSize": 1024,
            "activationDtype": "f16",
            "tensorEncoding": "ggml-tensor-v1",
            "tokenizerConfigHash": "ab" * 32,
            "backend": "llama.cpp-patched",
            "backendVersion": "llama.cpp/cai-shard-0.1",
        },
        payload=payload,
        frame_kind="activation",
        layer_start=0,
        layer_end=14,
        token_start=0,
        token_end=8,
    )

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        _session_id, _batch_id = _create_received_batch(
            policy,
            payload=payload,
            phase="prefill_activation_batches",
            metadata=metadata,
        )
        result = run_cai_owned_shard_runtime_once(
            CaiOwnedShardRuntimeConfig(
                node_id="node-b",
                runtime_id="runtime-external-drifted-patch-boundary",
                require_production_llm_handoff=True,
                policy=policy,
            ),
            ExternalLlamaCppShardAdapter(
                command=[
                    sys.executable,
                    "-c",
                    _mismatched_process_patch_boundary_backend_code(),
                ],
                timeout_sec=10,
            ),
        )
        batch = list_cai_owned_transport_sessions(policy)[0].batch_records[0]

    assert result["status"] == "retry_scheduled"
    assert batch["lastError"] == (
        "External llama.cpp shard adapter patch boundary changed between "
        "load_shard and process_prefill."
    )
    assert batch["status"] == "received"


def test_external_llama_cpp_two_runtime_handoff_chain_forwards_state() -> None:
    runtime_metadata = {
        "modelId": "cai-network/Qwen3-0.6B-GGUF",
        "totalLayerCount": 28,
        "hiddenSize": 1024,
        "activationDtype": "f16",
        "tensorEncoding": "ggml-tensor-v1",
        "tokenizerConfigHash": "ab" * 32,
        "modelSha256Hex": "12" * 32,
        "backend": "llama.cpp-patched",
        "backendVersion": "llama.cpp/cai-shard-0.1",
    }

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        offer = build_cai_owned_transport_session_offer(
            instance_id="instance-external-two-runtime",
            participant_node_ids=["node-user", "node-a", "node-b"],
            executor_node_ids=["node-a", "node-b"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-external-two-runtime",
            source_node_id="node-user",
        )
        create_cai_owned_transport_session_from_offer(
            offer,
            session_id=offer["sessionId"],
            local_node_id="node-user",
            policy=policy,
        )
        initial_payload = b"prompt-tokens"
        decode_template = build_cai_owned_llm_shard_frame_metadata_from_runtime(
            model_id="cai-network/Qwen3-0.6B-GGUF",
            runtime_metadata=runtime_metadata,
            payload=b"placeholder-prefill-output",
            frame_kind="decode",
            layer_start=14,
            layer_end=28,
            token_start=4,
            token_end=5,
            sequence=1,
            decode_state={"position": 4, "sequenceId": "seq-smoke"},
        )
        initial_metadata = build_cai_owned_llm_shard_frame_metadata_from_runtime(
            model_id="cai-network/Qwen3-0.6B-GGUF",
            runtime_metadata=runtime_metadata,
            payload=initial_payload,
            frame_kind="activation",
            layer_start=0,
            layer_end=14,
            token_start=0,
            token_end=4,
            sequence=0,
        )
        initial_metadata["nextSinkNodeId"] = "node-b"
        initial_metadata["nextOutputPhase"] = "decode_activation_batches"
        initial_metadata["nextOutputSequence"] = 1
        initial_metadata["nextFrameTemplate"] = decode_template
        first_envelope = build_cai_owned_transport_batch_envelope(
            session_id=offer["sessionId"],
            phase="prefill_activation_batches",
            source_node_id="node-user",
            sink_node_id="node-a",
            sequence=0,
            payload=initial_payload,
            metadata=initial_metadata,
        )
        record_cai_owned_transport_batch_envelope(
            offer["sessionId"],
            first_envelope,
            local_node_id="node-a",
            policy=policy,
        )
        adapter = ExternalLlamaCppShardAdapter(
            command=_smoke_runner_command(),
            env={"PYTHONPATH": str(SRC_ROOT)},
            timeout_sec=10,
        )
        node_a_result = run_cai_owned_shard_runtime_once(
            CaiOwnedShardRuntimeConfig(
                node_id="node-a",
                runtime_id="runtime-external-node-a",
                require_production_llm_handoff=True,
                policy=policy,
            ),
            adapter,
        )
        forwarded = node_a_result["outputForward"]["envelope"]
        record_cai_owned_transport_batch_envelope(
            offer["sessionId"],
            forwarded,
            local_node_id="node-b",
            policy=policy,
        )
        node_b_result = run_cai_owned_shard_runtime_once(
            CaiOwnedShardRuntimeConfig(
                node_id="node-b",
                runtime_id="runtime-external-node-b",
                require_production_llm_handoff=True,
                policy=policy,
            ),
            adapter,
        )
        assert node_b_result["status"] == "processed", node_b_result
        final_envelope = build_cai_owned_transport_output_batch_envelope(
            session_id=offer["sessionId"],
            source_batch_id=forwarded["batchId"],
            sink_node_id="node-user",
            metadata={"nextStage": "requester"},
            policy=policy,
        )
        completed = complete_cai_owned_transport_session(
            offer["sessionId"],
            policy=policy,
        )

    forwarded_payload = cai_owned_transport_batch_payload_bytes(forwarded)
    forwarded_metadata = forwarded["metadata"]
    assert node_a_result["status"] == "processed"
    assert node_a_result["outputForward"]["status"] == "no_peer_urls"
    assert forwarded_payload == b"prefill-state:prompt-tokens"
    assert forwarded_metadata["layerStart"] == 14
    assert forwarded_metadata["layerEnd"] == 28
    assert forwarded_metadata["llmHandoff"]["tensor"]["sha256Hex"] == (
        hashlib.sha256(forwarded_payload).hexdigest()
    )
    assert node_b_result["status"] == "processed"
    assert cai_owned_transport_batch_payload_bytes(final_envelope) == (
        b"decoded-answer:prefill-state:prompt-tokens"
    )
    assert completed.status == "completed"
    assert completed.proof is not None
    assert completed.proof["executionAudit"]["verified"] is True
    assert [item["metrics"]["adapterIds"] for item in completed.proof["shardReceipts"]] == [
        ["llama.cpp-external-shard"],
        ["llama.cpp-external-shard"],
    ]


def test_dispatch_smoke_runner_completes_four_stage_llm_shard_route() -> None:
    captured: list[dict[str, object]] = []

    def fake_urlopen(request, timeout: float):
        body = json.loads(request.data.decode("utf-8"))
        captured.append({"url": request.full_url, "body": body})
        return _FakeResponse({"status": "ok", "sessionId": body.get("sessionId")})

    runtime_metadata = {
        "modelId": "cai-network/Qwen3-0.6B-GGUF",
        "totalLayerCount": 28,
        "hiddenSize": 1024,
        "activationDtype": "f16",
        "tensorEncoding": "ggml-tensor-v1",
        "tokenizerConfigHash": "ab" * 32,
        "modelSha256Hex": "12" * 32,
        "backend": "llama.cpp-patched",
        "backendVersion": "llama.cpp/cai-shard-0.1",
    }

    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ), patch("cai_compute_chain.decentralized_compute.urlopen", fake_urlopen):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        dispatch = dispatch_cai_owned_transport_execution_dag(
            instance_id="instance-smoke-four-stage",
            requester_node_id="node-user",
            executor_node_ids=["node-a", "node-b"],
            peer_cai_urls_by_node={
                "node-a": ["http://node-a:52415"],
                "node-b": ["http://node-b:52415"],
            },
            initial_payload=b"prompt",
            total_layer_count=28,
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-smoke-four-stage",
            tokenizer_config_hash="ab" * 32,
            llm_runtime_metadata=runtime_metadata,
            initial_token_count=4,
            policy=policy,
        )
        adapter = cai_owned_shard_adapter_from_env(
            {"CAI_LLM_SHARD_ADAPTER": "smoke_runner"}
        )
        current_envelope = dispatch["initialBatchEnvelope"]
        processed_results: list[dict[str, object]] = []
        for node_id in ("node-a", "node-b", "node-a", "node-b"):
            record_cai_owned_transport_batch_envelope(
                dispatch["sessionId"],
                current_envelope,
                local_node_id=node_id,
                policy=policy,
            )
            result = run_cai_owned_shard_runtime_once(
                CaiOwnedShardRuntimeConfig(
                    node_id=node_id,
                    runtime_id=f"runtime-{node_id}",
                    require_production_llm_handoff=True,
                    policy=policy,
                ),
                adapter,
            )
            assert result["status"] == "processed", result
            processed_results.append(result)
            forward = result["outputForward"]
            assert forward["status"] in {"submitted", "no_peer_urls"}
            current_envelope = forward["envelope"]
        record_cai_owned_transport_batch_envelope(
            dispatch["sessionId"],
            current_envelope,
            local_node_id="node-user",
            policy=policy,
        )
        completed = complete_cai_owned_transport_session(
            dispatch["sessionId"],
            policy=policy,
        )

    assert dispatch["status"] == "dispatched"
    assert len(captured) == 6
    assert current_envelope["sinkNodeId"] == "node-user"
    assert cai_owned_transport_batch_payload_bytes(current_envelope) == (
        b"decoded-answer:decoded-answer:prefill-state:prefill-state:prompt"
    )
    assert [result["workItem"]["batch"]["sinkNodeId"] for result in processed_results] == [
        "node-a",
        "node-b",
        "node-a",
        "node-b",
    ]
    assert completed.status == "completed"
    assert completed.proof is not None
    assert completed.proof["executionAudit"]["verified"] is True
    assert completed.proof["executionAudit"]["processedBatchCount"] == 4
    assert completed.proof["executionAudit"]["executionDag"]["verified"] is True
    assert len(
        completed.proof["executionAudit"]["executionDag"]["processedStageIds"]
    ) == 4
    assert len(completed.proof["shardReceipts"]) == 2
    assert sum(
        len(receipt["batchIds"]) for receipt in completed.proof["shardReceipts"]
    ) == 4
    assert [receipt["metrics"]["inputTokenCount"] for receipt in completed.proof["shardReceipts"]] == [
        5,
        5,
    ]
    assert [receipt["metrics"]["outputTokenCount"] for receipt in completed.proof["shardReceipts"]] == [
        1,
        1,
    ]
    assert [receipt["metrics"]["promptTokenCount"] for receipt in completed.proof["shardReceipts"]] == [
        4,
        4,
    ]
    assert [receipt["metrics"]["completionTokenCount"] for receipt in completed.proof["shardReceipts"]] == [
        1,
        1,
    ]


def test_production_llm_runtime_rejects_missing_handoff_contract() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        _session_id, _batch_id = _create_received_batch(
            policy,
            payload=b"missing-handoff",
            metadata=build_cai_owned_transport_frame_metadata(
                model_id="cai-network/Qwen3-0.6B-GGUF",
                frame_kind="activation",
                dtype="f16",
                shape=[1, 1, 1],
                payload_sha256_hex=hashlib.sha256(b"missing-handoff").hexdigest(),
            ),
        )
        result = run_cai_owned_shard_runtime_once(
            CaiOwnedShardRuntimeConfig(
                node_id="node-b",
                runtime_id="runtime-require-handoff",
                require_production_llm_handoff=True,
                policy=policy,
            ),
            DeterministicBytesShardAdapter(prefix=b"unsafe:"),
        )
        batch = list_cai_owned_transport_sessions(policy)[0].batch_records[0]

    assert result["status"] == "retry_scheduled"
    assert batch["status"] == "received"
    assert batch["lastError"] == "CAI-owned LLM handoff metadata is missing."


def test_cai_owned_shard_runtime_reports_busy_when_capacity_full() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        session_id, batch_id = _create_received_batch(policy, payload=b"busy-input")
        claim_cai_owned_transport_batch(
            session_id,
            batch_id,
            node_id="node-b",
            runtime_id="runtime-active",
            lease_seconds=60,
            policy=policy,
        )
        config = CaiOwnedShardRuntimeConfig(
            node_id="node-b",
            runtime_id="runtime-loop",
            max_concurrent_batches=1,
            policy=policy,
        )

        capacity = cai_owned_transport_runtime_capacity_status(config)
        result = run_cai_owned_shard_runtime_once(
            config,
            DeterministicBytesShardAdapter(),
        )

    assert capacity["status"] == "busy"
    assert capacity["activeProcessingCount"] == 1
    assert capacity["activeBatchIds"] == [batch_id]
    assert result["status"] == "busy"
    assert result["workItem"] is None


def test_cai_owned_shard_runtime_fails_oversized_payload_without_retry() -> None:
    with tempfile.TemporaryDirectory() as tempdir, patch(
        "cai_compute_chain.wallet.repo_root",
        return_value=Path(tempdir),
    ):
        policy = WalletPolicy(wallet_data_dirname=".tmp-cai-runtime")
        _session_id, _batch_id = _create_received_batch(
            policy,
            payload=b"payload-too-large",
        )
        config = CaiOwnedShardRuntimeConfig(
            node_id="node-b",
            runtime_id="runtime-loop",
            max_payload_size_bytes=4,
            policy=policy,
        )

        result = run_cai_owned_shard_runtime_once(
            config,
            DeterministicBytesShardAdapter(),
        )
        batch = list_cai_owned_transport_sessions(policy)[0].batch_records[0]

    assert result["status"] == "failed"
    assert result["reason"] == "payload_too_large"
    assert result["failure"]["retryScheduled"] is False
    assert batch["status"] == "failed"
    assert batch["retryable"] is False
    assert "payload exceeds runtime capacity" in batch["lastError"]


def _create_received_batch(
    policy: WalletPolicy,
    *,
    payload: bytes,
    phase: str = "decode_activation_batches",
    metadata: dict | None = None,
) -> tuple[str, str]:
    offer = build_cai_owned_transport_session_offer(
        instance_id="instance-runtime",
        participant_node_ids=["node-a", "node-b"],
        model_id="cai-network/Qwen3-0.6B-GGUF",
        task_id="task-runtime",
        source_node_id="node-a",
    )
    create_cai_owned_transport_session_from_offer(
        offer,
        session_id=offer["sessionId"],
        local_node_id="node-b",
        policy=policy,
    )
    envelope = build_cai_owned_transport_batch_envelope(
        session_id=offer["sessionId"],
        phase=phase,
        source_node_id="node-a",
        sink_node_id="node-b",
        sequence=1,
        payload=payload,
        metadata=metadata,
    )
    record_cai_owned_transport_batch_envelope(
        offer["sessionId"],
        envelope,
        local_node_id="node-b",
        policy=policy,
    )
    return offer["sessionId"], envelope["batchId"]

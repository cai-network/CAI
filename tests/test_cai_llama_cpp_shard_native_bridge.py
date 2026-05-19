# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Sequence
from http.server import ThreadingHTTPServer
import io
import json
import os
import struct
import sys
import threading
from pathlib import Path
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.cai_llama_cpp_shard_native_bridge import (  # noqa: E402
    PersistentNativeEngineClient,
    _handler_class,
    handle_native_bridge_health,
    handle_native_bridge_request_body,
    main as native_bridge_main,
)
from cai_compute_chain.cai_llm_shard_conformance import (  # noqa: E402
    run_cai_owned_llm_shard_conformance,
)
from cai_compute_chain.decentralized_compute import (  # noqa: E402
    build_cai_owned_llm_shard_frame_metadata_from_runtime,
)
from cai_compute_chain.model_distribution import (  # noqa: E402
    ModelShardAssignment,
    build_gguf_model_package_manifest,
    materialized_assignment_artifact_path,
    put_cached_chunk,
    save_local_artifact_binding,
    save_model_package_manifest,
)
from cai_compute_chain.cai_owned_runtime import (  # noqa: E402
    ExternalLlamaCppShardAdapter,
    LLAMA_CPP_EXTERNAL_SHARD_ACTIVATION_BOUNDARY,
    LLAMA_CPP_EXTERNAL_SHARD_DECODE_STATE_BOUNDARY,
    LLAMA_CPP_EXTERNAL_SHARD_HANDOFF_ABI,
    LLAMA_CPP_EXTERNAL_SHARD_PATCH_BOUNDARY_ABI,
    LLAMA_CPP_EXTERNAL_SHARD_SPEC_ABI,
    LLAMA_CPP_EXTERNAL_SHARD_PRODUCTION_CAPABILITIES,
    LLAMA_CPP_EXTERNAL_SHARD_PRODUCTION_STATE_CONTRACT_ABI,
    LLAMA_CPP_EXTERNAL_SHARD_REQUIRED_CAPABILITIES,
    LLAMA_CPP_EXTERNAL_SHARD_SUPPORTED_TENSOR_ENCODINGS,
)


def _write_minimal_gguf_file(path: Path) -> Path:
    tensors = (
        ("token_embd.weight", b"EMBED000"),
        ("blk.0.attn_q.weight", b"LAYER000"),
        ("blk.1.attn_q.weight", b"LAYER111"),
        ("output.weight", b"OUTPUT00"),
    )

    def _gguf_string(value: str) -> bytes:
        encoded = value.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    payload = bytearray()
    payload.extend(b"GGUF")
    payload.extend(struct.pack("<I", 3))
    payload.extend(struct.pack("<Q", len(tensors)))
    payload.extend(struct.pack("<Q", 1))
    payload.extend(_gguf_string("general.alignment"))
    payload.extend(struct.pack("<I", 4))
    payload.extend(struct.pack("<I", 32))

    relative_offset = 0
    tensor_payload = bytearray()
    for name, tensor_bytes in tensors:
        payload.extend(_gguf_string(name))
        payload.extend(struct.pack("<I", 1))
        payload.extend(struct.pack("<Q", len(tensor_bytes)))
        payload.extend(struct.pack("<I", 0))
        payload.extend(struct.pack("<Q", relative_offset))
        tensor_payload.extend(tensor_bytes)
        relative_offset += len(tensor_bytes)

    while len(payload) % 32 != 0:
        payload.append(0)
    payload.extend(tensor_payload)
    path.write_bytes(bytes(payload))
    return path


def _production_native_command_code() -> str:
    return r"""
import base64
import hashlib
import json
import sys
from pathlib import Path

request = json.loads(sys.stdin.read() or "{}")
capabilities = [
    "layer_range_execution",
    "activation_handoff",
    "decode_state_handoff",
    "output_frame_metadata",
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
        "capabilities": capabilities,
        "patchBoundary": {
            "schemaVersion": 1,
            "abi": "cai-llama-cpp-shard-patch-boundary-v1",
            "backend": "llama.cpp-patched",
            "backendVersion": "llama.cpp/cai-native-bridge-test",
            "patchId": "cai-llama-cpp-shard-native-bridge",
            "runnerProtocolVersion": "0.1",
            "modelFormat": "gguf",
            "requiresPatchedBackend": True,
            "activationBoundary": "layer-range-activation-v1",
            "decodeStateBoundary": "token-step-kv-cache-v1",
            "supportedTensorEncodings": ["ggml-tensor-v1"],
            "capabilities": capabilities,
            "extraMetadata": {"productionStateContract": state_contract},
        },
        "metrics": {"backendLoaded": True, "backendMode": "real_llama_cpp"},
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
            "backendAction": request.get("action"),
            "backendMode": "real_llama_cpp",
            "generationProbeReady": True,
        },
    }))
    raise SystemExit(0)

if request.get("action") in {"process_prefill", "process_decode"}:
    local_file_contract = request.get("localFileContract") or {}
    local_artifact_resolution = request.get("localArtifactResolution") or {}
    payload_file = request.get("payloadFile")
    if isinstance(payload_file, dict):
        payload = Path(str(payload_file.get("path") or "")).read_bytes()
    else:
        payload = base64.b64decode(request.get("payloadBase64") or "")
    prefix = b"native-decode:" if request.get("action") == "process_decode" else b"native-state:"
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
    response = {
        "status": "ok",
        "outputPayloadSha256Hex": output_hash,
        "outputFrameMetadata": template,
        "nativeExecution": {
            "schemaVersion": 1,
            "executionMode": "layer_range",
            "action": request.get("action"),
            "modelId": ((request.get("shardSpec") or {}).get("modelId")),
            "layerStart": ((request.get("shardSpec") or {}).get("layerStart")),
            "layerEnd": ((request.get("shardSpec") or {}).get("layerEnd")),
            "artifactKind": (
                "assignment"
                if isinstance(local_artifact_resolution.get("assignmentArtifact"), dict)
                else "model"
            ),
            "artifactId": (
                ((local_artifact_resolution.get("assignmentArtifact") or {}).get("artifactId"))
                or ((local_artifact_resolution.get("modelArtifact") or {}).get("artifactId"))
            ),
            "artifactSource": (
                ((local_artifact_resolution.get("assignmentArtifact") or {}).get("source"))
                or ((local_artifact_resolution.get("modelArtifact") or {}).get("source"))
            ),
            "artifactPath": (
                ((local_artifact_resolution.get("assignmentArtifact") or {}).get("localPath"))
                or ((local_artifact_resolution.get("modelArtifact") or {}).get("localPath"))
            ),
            "usedPatchedBackend": True,
            "fallbackMode": (
                "none"
                if isinstance(local_artifact_resolution.get("assignmentArtifact"), dict)
                else ""
            ),
        },
        "metrics": {
            "backendAction": request.get("action"),
            "backendMode": "real_llama_cpp",
        },
    }
    output_path = str(local_file_contract.get("responseOutputPath") or "").strip()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(output)
        response["outputPayloadFile"] = {
            "path": output_path,
            "sizeBytes": len(output),
            "sha256Hex": output_hash,
        }
    else:
        response["outputPayloadBase64"] = base64.b64encode(output).decode("ascii")
    print(json.dumps(response))
    raise SystemExit(0)

print(json.dumps({"status": "ok", "metrics": {"backendMode": "real_llama_cpp"}}))
"""


def _production_native_jsonl_script(path: Path) -> Path:
    path.write_text(
        r"""
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

capabilities = [
    "layer_range_execution",
    "activation_handoff",
    "decode_state_handoff",
    "output_frame_metadata",
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
counter = 0


def response_for(request):
    global counter
    counter += 1
    action = request.get("action")
    metrics = {
        "backendAction": action,
        "backendMode": "real_llama_cpp",
        "persistentBackendPid": os.getpid(),
        "persistentCallCounter": counter,
    }
    if action == "load_shard":
        return {
            "status": "ready",
            "capabilities": capabilities,
            "patchBoundary": {
                "schemaVersion": 1,
                "abi": "cai-llama-cpp-shard-patch-boundary-v1",
                "backend": "llama.cpp-patched",
                "backendVersion": "llama.cpp/cai-native-bridge-test",
                "patchId": "cai-llama-cpp-shard-native-bridge",
                "runnerProtocolVersion": "0.1",
                "modelFormat": "gguf",
                "requiresPatchedBackend": True,
                "activationBoundary": "layer-range-activation-v1",
                "decodeStateBoundary": "token-step-kv-cache-v1",
                "supportedTensorEncodings": ["ggml-tensor-v1"],
                "capabilities": capabilities,
                "extraMetadata": {"productionStateContract": state_contract},
            },
            "metrics": {**metrics, "backendLoaded": True},
        }
    if action == "probe_generation":
        probe = request.get("generationProbe") or {}
        return {
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
            "metrics": {**metrics, "generationProbeReady": True},
        }
    if action in {"process_prefill", "process_decode"}:
        local_file_contract = request.get("localFileContract") or {}
        local_artifact_resolution = request.get("localArtifactResolution") or {}
        payload_file = request.get("payloadFile")
        if isinstance(payload_file, dict):
            payload = Path(str(payload_file.get("path") or "")).read_bytes()
        else:
            payload = base64.b64decode(request.get("payloadBase64") or "")
        prefix = b"native-decode:" if action == "process_decode" else b"native-state:"
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
        response = {
            "status": "ok",
            "outputPayloadSha256Hex": output_hash,
            "outputFrameMetadata": template,
            "nativeExecution": {
                "schemaVersion": 1,
                "executionMode": "layer_range",
                "action": action,
                "modelId": ((request.get("shardSpec") or {}).get("modelId")),
                "layerStart": ((request.get("shardSpec") or {}).get("layerStart")),
                "layerEnd": ((request.get("shardSpec") or {}).get("layerEnd")),
                "artifactKind": (
                    "assignment"
                    if isinstance(local_artifact_resolution.get("assignmentArtifact"), dict)
                    else "model"
                ),
                "artifactId": (
                    ((local_artifact_resolution.get("assignmentArtifact") or {}).get("artifactId"))
                    or ((local_artifact_resolution.get("modelArtifact") or {}).get("artifactId"))
                ),
                "artifactSource": (
                    ((local_artifact_resolution.get("assignmentArtifact") or {}).get("source"))
                    or ((local_artifact_resolution.get("modelArtifact") or {}).get("source"))
                ),
                "artifactPath": (
                    ((local_artifact_resolution.get("assignmentArtifact") or {}).get("localPath"))
                    or ((local_artifact_resolution.get("modelArtifact") or {}).get("localPath"))
                ),
                "usedPatchedBackend": True,
                "fallbackMode": (
                    "none"
                    if isinstance(local_artifact_resolution.get("assignmentArtifact"), dict)
                    else ""
                ),
            },
            "metrics": metrics,
        }
        output_path = str(local_file_contract.get("responseOutputPath") or "").strip()
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(output)
            response["outputPayloadFile"] = {
                "path": output_path,
                "sizeBytes": len(output),
                "sha256Hex": output_hash,
            }
        else:
            response["outputPayloadBase64"] = base64.b64encode(output).decode("ascii")
        return response
    return {"status": "ok", "metrics": metrics}


for line in sys.stdin:
    if not line.strip():
        continue
    print(json.dumps(response_for(json.loads(line))), flush=True)
""",
        encoding="utf-8",
    )
    return path


def _start_native_bridge_server(
    native_command: Sequence[str],
    *,
    persistent_engine: PersistentNativeEngineClient | None = None,
) -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    command = [str(item) for item in native_command]
    handler = _handler_class(
        native_command=command,
        timeout_sec=10,
        persistent_engine=persistent_engine,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint_url = f"http://127.0.0.1:{server.server_address[1]}/cai-shard"
    return server, thread, endpoint_url


def _native_bridge_request(action: str = "load_shard") -> dict:
    return {
        "schemaVersion": 1,
        "abi": LLAMA_CPP_EXTERNAL_SHARD_HANDOFF_ABI,
        "action": action,
        "adapterId": "llama.cpp-external-shard",
        "adapterVersion": "llama.cpp-external-shard/0.1",
        "backend": "llama.cpp-patched",
        "frame": {
            "sessionId": "caiot_native_bridge_test",
            "batchId": "caibatch_native_bridge_test",
            "phase": "prefill_activation_batches",
            "sourceNodeId": "node-user",
            "sinkNodeId": "node-a",
            "sequence": 0,
            "modelId": "cai-network/Qwen3-0.6B-GGUF",
            "frameKind": "activation",
            "layerStart": 0,
            "layerEnd": 1,
            "tokenStart": 0,
            "tokenEnd": 1,
            "payloadSha256Hex": "ab" * 32,
            "metadata": {},
        },
        "shardSpec": {
            "schemaVersion": 1,
            "abi": LLAMA_CPP_EXTERNAL_SHARD_SPEC_ABI,
            "modelId": "cai-network/Qwen3-0.6B-GGUF",
            "modelFormat": "gguf",
            "requiresPatchedBackend": True,
            "backend": "llama.cpp-patched",
            "backendVersion": "llama.cpp/cai-native-bridge-test",
            "shardExecutionMode": "layer_range",
            "frameKind": "activation",
            "phase": "prefill_activation_batches",
            "layerStart": 0,
            "layerEnd": 1,
            "tokenStart": 0,
            "tokenEnd": 1,
            "tensor": {
                "name": "layer_range_activation",
                "dtype": "f16",
                "encoding": "ggml-tensor-v1",
                "shape": [1, 1, 1024],
                "sha256Hex": "ab" * 32,
            },
            "totalLayerCount": 28,
            "modelSha256Hex": "cd" * 32,
            "tokenizerConfigHash": "ef" * 32,
        },
        "payloadBase64": "",
        "payloadSha256Hex": "ab" * 32,
        "outputContract": {"schemaVersion": 1},
        "productionRequirements": {
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
                LLAMA_CPP_EXTERNAL_SHARD_SUPPORTED_TENSOR_ENCODINGS
            ),
            "requiredCapabilities": list(LLAMA_CPP_EXTERNAL_SHARD_REQUIRED_CAPABILITIES),
            "requiredProductionCapabilities": list(
                LLAMA_CPP_EXTERNAL_SHARD_PRODUCTION_CAPABILITIES
            ),
            "requiresRealStateContract": True,
            "requiresShardOnlyLoading": True,
            "forbidFullModelFallback": True,
        },
    }


def _native_bridge_process_request(action: str = "process_prefill") -> dict:
    request = _native_bridge_request(action)
    payload = b"prompt"
    runtime_metadata = {
        "modelId": "cai-network/Qwen3-0.6B-GGUF",
        "totalLayerCount": 28,
        "hiddenSize": 1024,
        "activationDtype": "f16",
        "tensorEncoding": "ggml-tensor-v1",
        "tokenizerConfigHash": "ef" * 32,
        "backend": "llama.cpp-patched",
        "backendVersion": "llama.cpp/cai-native-bridge-test",
    }
    metadata = build_cai_owned_llm_shard_frame_metadata_from_runtime(
        model_id="cai-network/Qwen3-0.6B-GGUF",
        runtime_metadata=runtime_metadata,
        payload=payload,
        frame_kind="activation" if action == "process_prefill" else "decode",
        layer_start=0,
        layer_end=1,
        token_start=0,
        token_end=1,
        sequence=0,
    )
    request["frame"]["metadata"] = metadata
    request["frame"]["payloadSha256Hex"] = metadata["payloadSha256Hex"]
    request["payloadSha256Hex"] = metadata["payloadSha256Hex"]
    return request


def test_native_bridge_reports_missing_native_engine() -> None:
    status_code, response = handle_native_bridge_request_body(
        json.dumps(_native_bridge_request()).encode("utf-8"),
    )

    assert status_code == 200
    assert response["status"] == "error"
    assert response["error"] == (
        "CAI LLM shard native engine command is not configured."
    )
    assert response["metrics"]["backendMode"] == "native_bridge_missing_engine"


def test_native_bridge_rejects_missing_production_requirements() -> None:
    request = _native_bridge_request()
    del request["productionRequirements"]

    status_code, response = handle_native_bridge_request_body(
        json.dumps(request).encode("utf-8"),
    )

    assert status_code == 400
    assert response["status"] == "error"
    assert response["error"] == "CAI LLM shard productionRequirements are missing."


def test_native_bridge_health_reports_missing_native_engine() -> None:
    status_code, response = handle_native_bridge_health()

    assert status_code == 503
    assert response["status"] == "degraded"
    assert response["nativeCommandConfigured"] is False
    assert response["nativeEngineMode"] == "subprocess_per_request"
    assert response["error"] == (
        "CAI LLM shard native engine command is not configured."
    )


def test_native_bridge_invokes_native_command() -> None:
    native_code = (
        "import json, sys; "
        "request=json.loads(sys.stdin.read() or '{}'); "
        "print(json.dumps({'status':'ready','metrics':{"
        "'backendMode':'native_command','action':request.get('action')}}))"
    )

    status_code, response = handle_native_bridge_request_body(
        json.dumps(_native_bridge_request()).encode("utf-8"),
        native_command=[sys.executable, "-c", native_code],
        timeout_sec=10,
    )

    assert status_code == 200
    assert response["status"] == "ready"
    assert response["metrics"] == {
        "backendMode": "native_command",
        "action": "load_shard",
    }


def test_native_bridge_rejects_process_response_without_native_execution(
    tmp_path: Path,
) -> None:
    native_code = (
        "import json, sys; "
        "request=json.loads(sys.stdin.read() or '{}'); "
        "print(json.dumps({'status':'ok','outputPayloadBase64':'','outputPayloadSha256Hex':'"
        + ("ab" * 32)
        + "','outputFrameMetadata':{},'metrics':{'backendMode':'native_command'}}))"
    )
    request = _native_bridge_process_request("process_prefill")
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"model")
    request["shardSpec"]["artifactHint"] = {
        "modelArtifactPath": str(gguf_path),
        "artifactId": "gguf-main",
    }

    status_code, response = handle_native_bridge_request_body(
        json.dumps(request).encode("utf-8"),
        native_command=[sys.executable, "-c", native_code],
        timeout_sec=10,
    )

    assert status_code == 200
    assert response["status"] == "error"
    assert response["metrics"]["backendMode"] == "native_bridge_invalid_response"
    assert "nativeExecution is missing" in response["error"]


def test_native_bridge_injects_local_artifact_resolution_from_manifest_binding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gguf_path = tmp_path / "Qwen3-0.6B-Q8_0.gguf"
    gguf_path.write_bytes(b"gguf-test-payload")
    manifest = build_gguf_model_package_manifest(
        catalog_id="cai-private",
        model_id="cai-network/Qwen3-0.6B-GGUF",
        version="2026.05",
        gguf_path=gguf_path,
        total_layers=28,
        family="Qwen3",
        quantization="Q8_0",
    )
    monkeypatch.setattr("cai_compute_chain.wallet.repo_root", lambda: tmp_path)
    save_model_package_manifest(manifest)
    save_local_artifact_binding(
        manifest.catalog_id,
        manifest.version,
        artifact_id="gguf-main",
        local_path=gguf_path,
    )
    native_code = (
        "import json, sys; "
        "request=json.loads(sys.stdin.read() or '{}'); "
        "print(json.dumps({'status':'ready','metrics':{"
        "'backendMode':'native_command',"
        "'artifactResolution':request.get('localArtifactResolution')}}))"
    )

    status_code, response = handle_native_bridge_request_body(
        json.dumps(_native_bridge_request()).encode("utf-8"),
        native_command=[sys.executable, "-c", native_code],
        timeout_sec=10,
    )

    resolution = response["metrics"]["artifactResolution"]
    assert status_code == 200
    assert resolution["modelId"] == "cai-network/Qwen3-0.6B-GGUF"
    assert resolution["catalogId"] == "cai-private"
    assert resolution["version"] == "2026.05"
    assert resolution["preferredFilename"] == "Qwen3-0.6B-Q8_0.gguf"
    assert resolution["modelArtifact"]["artifactId"] == "gguf-main"
    assert resolution["modelArtifact"]["localPath"] == str(gguf_path.resolve())
    assert resolution["modelArtifact"]["source"] == "local_binding"
    assert resolution["modelArtifact"]["expectedSizeBytes"] == len(b"gguf-test-payload")


def test_native_bridge_auto_prepares_curated_qwen3_manifest_from_local_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    model_dir = repo_root / "models"
    model_dir.mkdir(parents=True)
    gguf_path = _write_minimal_gguf_file(model_dir / "Qwen3-0.6B-Q8_0.gguf")
    monkeypatch.setattr(
        "cai_compute_chain.cai_llama_cpp_shard_native_bridge.resolve_llama_cpp_repo_root",
        lambda: repo_root,
    )
    monkeypatch.setattr("cai_compute_chain.wallet.repo_root", lambda: tmp_path)
    native_code = (
        "import json, sys; "
        "request=json.loads(sys.stdin.read() or '{}'); "
        "print(json.dumps({'status':'ready','metrics':{"
        "'backendMode':'native_command',"
        "'artifactResolution':request.get('localArtifactResolution')}}))"
    )

    status_code, response = handle_native_bridge_request_body(
        json.dumps(_native_bridge_request()).encode("utf-8"),
        native_command=[sys.executable, "-c", native_code],
        timeout_sec=10,
    )

    resolution = response["metrics"]["artifactResolution"]
    assignment = resolution["assignmentArtifact"]
    assert status_code == 200
    assert resolution["modelArtifact"]["localPath"] == str(gguf_path.resolve())
    assert resolution["modelArtifact"]["source"] == "local_binding"
    assert assignment["source"] == "materialized_assignment"
    assert assignment["layerStart"] == 0
    assert assignment["layerEnd"] == 1
    assert Path(assignment["localPath"]).exists()


def test_native_bridge_injects_managed_runtime_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    binary_name = "llama-server.exe" if os.name == "nt" else "llama-server"
    rpc_name = "rpc-server.exe" if os.name == "nt" else "rpc-server"
    llama_server = repo_root / "runtime" / "llama.cpp" / binary_name
    rpc_server = repo_root / "runtime" / "llama.cpp" / rpc_name
    llama_server.parent.mkdir(parents=True, exist_ok=True)
    llama_server.write_bytes(b"server")
    rpc_server.write_bytes(b"rpc")
    runtime_root = tmp_path / "managed-runtime"
    monkeypatch.setattr(
        "cai_compute_chain.cai_llama_cpp_shard_native_bridge.resolve_llama_cpp_repo_root",
        lambda: repo_root,
    )
    monkeypatch.setenv("CAI_LLM_SHARD_RUNTIME_ROOT", str(runtime_root))
    native_code = (
        "import json, sys; "
        "request=json.loads(sys.stdin.read() or '{}'); "
        "print(json.dumps({'status':'ready','metrics':{"
        "'backendMode':'native_command',"
        "'managedRuntime':request.get('managedRuntime'),"
        "'executionWorkspace':request.get('executionWorkspace')}}))"
    )

    status_code, response = handle_native_bridge_request_body(
        json.dumps(_native_bridge_request()).encode("utf-8"),
        native_command=[sys.executable, "-c", native_code],
        timeout_sec=10,
    )

    managed_runtime = response["metrics"]["managedRuntime"]
    execution_workspace = response["metrics"]["executionWorkspace"]
    assert status_code == 200
    assert managed_runtime["abi"] == "cai-llama-cpp-managed-runtime-v1"
    assert managed_runtime["platform"] == os.name
    assert managed_runtime["repoRoot"] == str(repo_root)
    assert managed_runtime["runtimeRoot"] == str(runtime_root.resolve())
    assert managed_runtime["modelId"] == "cai-network/Qwen3-0.6B-GGUF"
    assert managed_runtime["llamaCpp"]["llamaServerPath"] == str(llama_server.resolve())
    assert managed_runtime["llamaCpp"]["rpcServerPath"] == str(rpc_server.resolve())
    assert Path(managed_runtime["sessionPaths"]["root"]).exists()
    assert Path(managed_runtime["sessionPaths"]["stateDir"]).exists()
    assert Path(managed_runtime["sessionPaths"]["logsDir"]).exists()
    assert execution_workspace["abi"] == "cai-llama-cpp-execution-workspace-v1"
    assert execution_workspace["sessionId"] == "caiot_native_bridge_test"
    assert execution_workspace["expectedOutputKind"] == "final_output"
    assert Path(execution_workspace["root"]).exists()
    assert Path(execution_workspace["inputsDir"]).exists()
    assert Path(execution_workspace["outputsDir"]).exists()


def test_native_bridge_rejects_invalid_explicit_artifact_hint_path() -> None:
    request = _native_bridge_request()
    request["shardSpec"]["artifactHint"] = {
        "catalogId": "cai-private",
        "version": "2026.05",
        "artifactId": "gguf-main",
        "modelArtifactPath": "C:/missing/path/model.gguf",
    }

    status_code, response = handle_native_bridge_request_body(
        json.dumps(request).encode("utf-8"),
    )

    assert status_code == 400
    assert response["status"] == "error"
    assert "modelArtifactPath does not exist" in response["error"]


def test_native_bridge_injects_local_artifact_resolution_into_generation_probe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gguf_path = tmp_path / "Qwen3-0.6B-Q8_0.gguf"
    gguf_path.write_bytes(b"gguf-test-probe")
    manifest = build_gguf_model_package_manifest(
        catalog_id="cai-private",
        model_id="cai-network/Qwen3-0.6B-GGUF",
        version="2026.05",
        gguf_path=gguf_path,
        total_layers=28,
        family="Qwen3",
        quantization="Q8_0",
    )
    monkeypatch.setattr("cai_compute_chain.wallet.repo_root", lambda: tmp_path)
    save_model_package_manifest(manifest)
    save_local_artifact_binding(
        manifest.catalog_id,
        manifest.version,
        artifact_id="gguf-main",
        local_path=gguf_path,
    )
    request = {
        "schemaVersion": 1,
        "abi": LLAMA_CPP_EXTERNAL_SHARD_HANDOFF_ABI,
        "action": "probe_generation",
        "adapterId": "llama.cpp-external-shard",
        "adapterVersion": "llama.cpp-external-shard/0.1",
        "backend": "llama.cpp-patched",
        "generationProbe": {
            "schemaVersion": 1,
            "abi": "cai-llama-cpp-generation-probe-v1",
            "modelId": "cai-network/Qwen3-0.6B-GGUF",
            "prompt": "ping",
            "maxTokens": 4,
            "temperature": 0.0,
            "requiresRealModelExecution": True,
        },
        "productionRequirements": dict(_native_bridge_request()["productionRequirements"]),
    }
    native_code = (
        "import json, sys; "
        "request=json.loads(sys.stdin.read() or '{}'); "
        "print(json.dumps({'status':'ok','generationProbe':{"
        "'schemaVersion':1,"
        "'abi':'cai-llama-cpp-generation-probe-v1',"
        "'ready':True,"
        "'modelId':request.get('generationProbe',{}).get('modelId'),"
        "'outputText':'ok',"
        "'outputTokenCount':1,"
        "'realModelExecution':True},"
        "'metrics':{"
        "'backendMode':'native_command',"
        "'artifactResolution':request.get('localArtifactResolution')}}))"
    )

    status_code, response = handle_native_bridge_request_body(
        json.dumps(request).encode("utf-8"),
        native_command=[sys.executable, "-c", native_code],
        timeout_sec=10,
    )

    resolution = response["metrics"]["artifactResolution"]
    assert status_code == 200
    assert response["status"] == "ok"
    assert resolution["modelArtifact"]["localPath"] == str(gguf_path.resolve())
    assert resolution["modelArtifact"]["source"] == "local_binding"


def test_native_bridge_materializes_assignment_artifact_from_local_chunk_store(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gguf_payload = b"ABCDEFGH"
    gguf_path = tmp_path / "Qwen3-0.6B-Q8_0.gguf"
    gguf_path.write_bytes(gguf_payload)
    manifest = build_gguf_model_package_manifest(
        catalog_id="cai-private",
        model_id="cai-network/Qwen3-0.6B-GGUF",
        version="2026.05",
        gguf_path=gguf_path,
        total_layers=2,
        min_chunk_bytes=1,
        max_chunk_bytes=4,
        target_chunk_count=2,
        family="Qwen3",
        quantization="Q8_0",
    )
    monkeypatch.setattr("cai_compute_chain.wallet.repo_root", lambda: tmp_path)
    save_model_package_manifest(manifest)
    save_local_artifact_binding(
        manifest.catalog_id,
        manifest.version,
        artifact_id="gguf-main",
        local_path=gguf_path,
    )
    required_chunks = manifest.required_chunks_for_layers(0, 1)
    for chunk in required_chunks:
        payload = gguf_payload[
            chunk.offset_bytes : chunk.offset_bytes + chunk.size_bytes
        ]
        put_cached_chunk(
            catalog_id=manifest.catalog_id,
            version=manifest.version,
            chunk_id=chunk.chunk_id,
            sha256_hex=chunk.sha256_hex,
            content=payload,
        )
    request = _native_bridge_request()
    request["shardSpec"]["artifactHint"] = {
        "catalogId": manifest.catalog_id,
        "version": manifest.version,
        "artifactId": "gguf-main",
    }
    native_code = (
        "import json, sys; "
        "request=json.loads(sys.stdin.read() or '{}'); "
        "print(json.dumps({'status':'ready','metrics':{"
        "'backendMode':'native_command',"
        "'artifactResolution':request.get('localArtifactResolution')}}))"
    )

    status_code, response = handle_native_bridge_request_body(
        json.dumps(request).encode("utf-8"),
        native_command=[sys.executable, "-c", native_code],
        timeout_sec=10,
    )

    assignment_context = ModelShardAssignment(
        start_layer=0,
        end_layer=1,
        device_rank=0,
        world_size=1,
    )
    assignment_path = materialized_assignment_artifact_path(
        manifest,
        "gguf-main",
        assignment=assignment_context,
    )
    resolution = response["metrics"]["artifactResolution"]
    assignment = resolution["assignmentArtifact"]
    assert status_code == 200
    assert assignment["source"] == "materialized_assignment"
    assert assignment["layerStart"] == 0
    assert assignment["layerEnd"] == 1
    assert Path(assignment["localPath"]).exists()
    assert Path(assignment["localPath"]).resolve() == assignment_path.resolve()
    assert assignment["sizeBytes"] == len(gguf_payload)
    assert len(assignment["chunkRanges"]) == len(required_chunks)
    assert assignment["chunkRanges"][0]["offsetBytes"] == required_chunks[0].offset_bytes
    assert assignment["chunkRanges"][0]["sizeBytes"] == required_chunks[0].size_bytes
    assert assignment["chunkRanges"][0]["sha256Hex"] == required_chunks[0].sha256_hex
    assert assignment["coverage"]["abi"] == "cai-llama-cpp-assignment-coverage-v1"
    assert assignment["coverage"]["materializationMode"] == "sparse_full_size"
    assert assignment["coverage"]["artifactSizeBytes"] == len(gguf_payload)
    assert assignment["coverage"]["coveredByteCount"] == sum(
        int(chunk.size_bytes) for chunk in required_chunks
    )
    assert assignment["coverage"]["coveredRangeCount"] == len(required_chunks)
    assert assignment["coverage"]["zeroFilledOutsideCoveredRanges"] is True


def test_native_bridge_forwards_tensor_names_in_assignment_chunk_ranges(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gguf_path = _write_minimal_gguf_file(tmp_path / "Qwen3-0.6B-Q8_0.gguf")
    gguf_payload = gguf_path.read_bytes()
    manifest = build_gguf_model_package_manifest(
        catalog_id="cai-private",
        model_id="cai-network/Qwen3-0.6B-GGUF",
        version="2026.05",
        gguf_path=gguf_path,
        total_layers=2,
        min_chunk_bytes=80,
        max_chunk_bytes=80,
        family="Qwen3",
        quantization="Q8_0",
    )
    monkeypatch.setattr("cai_compute_chain.wallet.repo_root", lambda: tmp_path)
    save_model_package_manifest(manifest)
    save_local_artifact_binding(
        manifest.catalog_id,
        manifest.version,
        artifact_id="gguf-main",
        local_path=gguf_path,
    )
    required_chunks = manifest.required_chunks_for_layers(0, 1)
    for chunk in required_chunks:
        payload = gguf_payload[
            chunk.offset_bytes : chunk.offset_bytes + chunk.size_bytes
        ]
        put_cached_chunk(
            catalog_id=manifest.catalog_id,
            version=manifest.version,
            chunk_id=chunk.chunk_id,
            sha256_hex=chunk.sha256_hex,
            content=payload,
        )
    request = _native_bridge_request()
    request["shardSpec"]["layerEnd"] = 1
    request["frame"]["layerEnd"] = 1
    request["shardSpec"]["artifactHint"] = {
        "catalogId": manifest.catalog_id,
        "version": manifest.version,
        "artifactId": "gguf-main",
    }
    native_code = (
        "import json, sys; "
        "request=json.loads(sys.stdin.read() or '{}'); "
        "print(json.dumps({'status':'ready','metrics':{"
        "'backendMode':'native_command',"
        "'artifactResolution':request.get('localArtifactResolution')}}))"
    )

    status_code, response = handle_native_bridge_request_body(
        json.dumps(request).encode("utf-8"),
        native_command=[sys.executable, "-c", native_code],
        timeout_sec=10,
    )

    assignment = response["metrics"]["artifactResolution"]["assignmentArtifact"]
    assert status_code == 200
    chunk_tensor_names = [chunk["tensorNames"] for chunk in assignment["chunkRanges"]]
    assert chunk_tensor_names[0] == ["token_embd.weight"]
    assert "blk.0.attn_q.weight" in chunk_tensor_names[1]


def test_native_bridge_validates_native_execution_assignment_usage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gguf_payload = b"ABCDEFGH"
    gguf_path = tmp_path / "Qwen3-0.6B-Q8_0.gguf"
    gguf_path.write_bytes(gguf_payload)
    manifest = build_gguf_model_package_manifest(
        catalog_id="cai-private",
        model_id="cai-network/Qwen3-0.6B-GGUF",
        version="2026.05",
        gguf_path=gguf_path,
        total_layers=2,
        min_chunk_bytes=1,
        max_chunk_bytes=4,
        target_chunk_count=2,
        family="Qwen3",
        quantization="Q8_0",
    )
    monkeypatch.setattr("cai_compute_chain.wallet.repo_root", lambda: tmp_path)
    save_model_package_manifest(manifest)
    save_local_artifact_binding(
        manifest.catalog_id,
        manifest.version,
        artifact_id="gguf-main",
        local_path=gguf_path,
    )
    required_chunks = manifest.required_chunks_for_layers(0, 1)
    for chunk in required_chunks:
        payload = gguf_payload[
            chunk.offset_bytes : chunk.offset_bytes + chunk.size_bytes
        ]
        put_cached_chunk(
            catalog_id=manifest.catalog_id,
            version=manifest.version,
            chunk_id=chunk.chunk_id,
            sha256_hex=chunk.sha256_hex,
            content=payload,
        )
    request = _native_bridge_process_request("process_prefill")
    request["shardSpec"]["artifactHint"] = {
        "catalogId": manifest.catalog_id,
        "version": manifest.version,
        "artifactId": "gguf-main",
    }
    output_hash = "ab" * 32
    native_code = (
        "import json, sys; "
        "request=json.loads(sys.stdin.read() or '{}'); "
        "assignment=(request.get('localArtifactResolution',{}).get('assignmentArtifact') or {}); "
        "print(json.dumps({'status':'ok','outputPayloadBase64':'','outputPayloadSha256Hex':'"
        + output_hash
        + "','outputFrameMetadata':{},'nativeExecution':{"
        "'schemaVersion':1,"
        "'executionMode':'layer_range',"
        "'action':request.get('action'),"
        "'modelId':(request.get('shardSpec') or {}).get('modelId'),"
        "'layerStart':(request.get('shardSpec') or {}).get('layerStart'),"
        "'layerEnd':(request.get('shardSpec') or {}).get('layerEnd'),"
        "'artifactKind':'assignment',"
        "'artifactId':assignment.get('artifactId'),"
        "'artifactSource':assignment.get('source'),"
        "'artifactPath':assignment.get('localPath'),"
        "'usedPatchedBackend':True,"
        "'fallbackMode':'none'},"
        "'metrics':{'backendMode':'native_command'}}))"
    )

    status_code, response = handle_native_bridge_request_body(
        json.dumps(request).encode("utf-8"),
        native_command=[sys.executable, "-c", native_code],
        timeout_sec=10,
    )

    assert status_code == 200
    assert response["status"] == "ok"
    assert response["metrics"]["nativeExecutionValidated"] is True
    assert response["metrics"]["nativeExecutionArtifactKind"] == "assignment"
    assert response["metrics"]["nativeExecutionArtifactSource"] == "materialized_assignment"


def test_native_bridge_oneshot_invokes_native_command(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    native_script = tmp_path / "native_backend.py"
    native_script.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "request = json.loads(sys.stdin.read() or '{}')",
                "print(json.dumps({'status': 'ready', 'metrics': {",
                "    'backendMode': 'native_oneshot',",
                "    'action': request.get('action'),",
                "}}))",
            ],
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps(_native_bridge_request())),
    )

    exit_code = native_bridge_main(
        [
            "--oneshot",
            "--native-command",
            f'"{sys.executable}" "{native_script}"',
            "--timeout-sec",
            "10",
        ],
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["status"] == "ready"
    assert output["metrics"] == {
        "backendMode": "native_oneshot",
        "action": "load_shard",
    }


def test_persistent_native_engine_reuses_process(tmp_path: Path) -> None:
    native_script = tmp_path / "persistent_backend.py"
    native_script.write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "import sys",
                "counter = 0",
                "for line in sys.stdin:",
                "    counter += 1",
                "    request = json.loads(line or '{}')",
                "    print(json.dumps({'status': 'ready', 'metrics': {",
                "        'backendMode': 'persistent_native',",
                "        'action': request.get('action'),",
                "        'pid': os.getpid(),",
                "        'counter': counter,",
                "    }}), flush=True)",
            ],
        ),
        encoding="utf-8",
    )
    client = PersistentNativeEngineClient(
        [sys.executable, str(native_script)],
        timeout_sec=10,
    )
    try:
        first_status, first = client.call(_native_bridge_request("load_shard"))
        second_status, second = client.call(_native_bridge_request("finalize"))
    finally:
        client.close()

    assert first_status == 200
    assert second_status == 200
    assert first["status"] == "ready"
    assert second["status"] == "ready"
    assert first["metrics"]["backendMode"] == "persistent_native"
    assert first["metrics"]["pid"] == second["metrics"]["pid"]
    assert first["metrics"]["counter"] == 1
    assert second["metrics"]["counter"] == 2


def test_persistent_native_engine_health_reports_lifecycle(tmp_path: Path) -> None:
    native_script = tmp_path / "persistent_health_backend.py"
    native_script.write_text(
        "\n".join(
            [
                "import sys",
                "for line in sys.stdin:",
                "    pass",
            ],
        ),
        encoding="utf-8",
    )
    command = [sys.executable, str(native_script)]
    client = PersistentNativeEngineClient(command, timeout_sec=10)
    try:
        healthy_status, healthy = handle_native_bridge_health(
            native_command=command,
            persistent_engine=client,
        )
    finally:
        client.close()
    closed_status, closed = handle_native_bridge_health(
        native_command=command,
        persistent_engine=client,
    )

    assert healthy_status == 200
    assert healthy["status"] == "ok"
    assert healthy["nativeEngineMode"] == "persistent_jsonl"
    assert healthy["persistentEngine"]["alive"] is True
    assert healthy["persistentEngine"]["pid"] > 0
    assert closed_status == 503
    assert closed["status"] == "degraded"
    assert closed["persistentEngine"]["alive"] is False


def test_native_bridge_http_health_endpoint() -> None:
    server, thread, endpoint_url = _start_native_bridge_server(
        [sys.executable, "-c", "import json; print(json.dumps({'status':'ok'}))"],
    )
    try:
        health_url = endpoint_url.replace("/cai-shard", "/health")
        with urlopen(health_url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert payload["status"] == "ok"
    assert payload["bridge"] == "cai_llm_shard_native_bridge"
    assert payload["nativeCommandConfigured"] is True
    assert payload["nativeEngineMode"] == "subprocess_per_request"


def test_native_bridge_http_path_passes_production_conformance() -> None:
    server, thread, endpoint_url = _start_native_bridge_server(
        [sys.executable, "-c", _production_native_command_code()],
    )
    try:
        report = run_cai_owned_llm_shard_conformance(
            adapter=ExternalLlamaCppShardAdapter(
                endpoint_url=endpoint_url,
                timeout_sec=10,
            ),
            model_id="cai-network/Qwen3-0.6B-GGUF",
            payload="native-bridge-conformance",
            require_production=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert report["status"] == "passed"
    assert report["ok"] is True
    assert report["requireProduction"] is True
    assert report["errors"] == []
    assert report["checks"]["contractReady"] is True
    assert report["checks"]["productionReady"] is True
    assert report["checks"]["generationProbeReady"] is True
    assert report["checks"]["backendHealthReady"] is True
    assert report["checks"]["backendHealth"]["status"] == "ok"
    assert report["checks"]["backendHealth"]["nativeCommandConfigured"] is True
    assert report["checks"]["backendMode"] == "real_llama_cpp"
    assert report["checks"]["productionReadinessChecks"][
        "productionStateContractReady"
    ] is True


def test_native_bridge_http_persistent_path_passes_production_conformance(
    tmp_path: Path,
) -> None:
    native_script = _production_native_jsonl_script(
        tmp_path / "persistent_production_backend.py",
    )
    command = [sys.executable, str(native_script)]
    persistent_engine = PersistentNativeEngineClient(command, timeout_sec=10)
    server, thread, endpoint_url = _start_native_bridge_server(
        command,
        persistent_engine=persistent_engine,
    )
    try:
        report = run_cai_owned_llm_shard_conformance(
            adapter=ExternalLlamaCppShardAdapter(
                endpoint_url=endpoint_url,
                timeout_sec=10,
            ),
            model_id="cai-network/Qwen3-0.6B-GGUF",
            payload="native-bridge-persistent-conformance",
            require_production=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        persistent_engine.close()

    assert report["status"] == "passed"
    assert report["ok"] is True
    assert report["requireProduction"] is True
    assert report["errors"] == []
    assert report["checks"]["contractReady"] is True
    assert report["checks"]["productionReady"] is True
    assert report["checks"]["generationProbeReady"] is True
    assert report["checks"]["backendHealthReady"] is True
    assert report["checks"]["backendHealth"]["status"] == "ok"
    assert report["checks"]["backendHealth"]["nativeEngineMode"] == "persistent_jsonl"
    assert report["checks"]["backendHealth"]["persistentEngine"]["alive"] is True
    assert report["checks"]["backendMode"] == "real_llama_cpp"
    assert report["checks"]["productionReadinessChecks"][
        "productionStateContractReady"
    ] is True


def test_native_bridge_http_path_passes_production_conformance_with_local_file_io(
    tmp_path: Path,
) -> None:
    server, thread, endpoint_url = _start_native_bridge_server(
        [sys.executable, "-c", _production_native_command_code()],
    )
    io_root = tmp_path / "native-bridge-io"
    try:
        report = run_cai_owned_llm_shard_conformance(
            adapter=ExternalLlamaCppShardAdapter(
                endpoint_url=endpoint_url,
                timeout_sec=10,
                file_io_root=str(io_root),
                file_io_threshold_bytes=1,
            ),
            model_id="cai-network/Qwen3-0.6B-GGUF",
            payload="native-bridge-conformance",
            require_production=True,
        )
        io_children = list(io_root.iterdir()) if io_root.exists() else []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert report["status"] == "passed"
    assert report["ok"] is True
    assert report["checks"]["contractReady"] is True
    assert report["checks"]["productionReady"] is True
    assert report["checks"]["generationProbeReady"] is True
    assert io_children == []

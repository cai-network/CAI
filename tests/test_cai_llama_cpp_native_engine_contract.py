# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.cai_llama_cpp_native_engine_contract import (  # noqa: E402
    ASSIGNMENT_ARTIFACT_COVERAGE_ABI,
    build_native_engine_process_response,
    build_native_execution_receipt,
    decode_native_engine_input_payload,
    resolve_assignment_artifact_chunk_ranges,
    resolve_assignment_artifact_coverage,
    select_native_engine_artifact,
)


def _request() -> dict:
    return {
        "action": "process_prefill",
        "frame": {
            "modelId": "cai-network/Qwen3-0.6B-GGUF",
            "layerStart": 0,
            "layerEnd": 14,
        },
        "shardSpec": {
            "modelId": "cai-network/Qwen3-0.6B-GGUF",
            "layerStart": 0,
            "layerEnd": 14,
        },
        "localArtifactResolution": {
            "modelArtifact": {
                "artifactId": "gguf-main",
                "source": "local_binding",
                "localPath": "C:/models/model.gguf",
            },
            "assignmentArtifact": {
                "artifactId": "gguf-main",
                "source": "materialized_assignment",
                "localPath": "C:/models/model.layers-0-14.rank-0-of-1.gguf",
                "sizeBytes": 1024,
                "chunkRanges": [
                    {
                        "chunkId": "chunk-0",
                        "offsetBytes": 0,
                        "sizeBytes": 512,
                        "layerStart": 0,
                        "layerEnd": 7,
                        "tensorNames": [
                            "token_embd.weight",
                            "blk.0.attn_q.weight",
                        ],
                    },
                    {
                        "chunkId": "chunk-1",
                        "offsetBytes": 512,
                        "sizeBytes": 128,
                        "layerStart": 7,
                        "layerEnd": 14,
                        "tensorNames": ["blk.7.attn_q.weight"],
                    },
                ],
                "coverage": {
                    "abi": ASSIGNMENT_ARTIFACT_COVERAGE_ABI,
                    "materializationMode": "sparse_full_size",
                    "artifactSizeBytes": 1024,
                    "coveredByteCount": 640,
                    "coveredRangeCount": 2,
                    "zeroFilledOutsideCoveredRanges": True,
                },
            },
        },
    }


def test_select_native_engine_artifact_prefers_assignment() -> None:
    choice = select_native_engine_artifact(_request())

    assert choice is not None
    assert choice.kind == "assignment"
    assert choice.source == "materialized_assignment"
    assert choice.local_path.endswith(".layers-0-14.rank-0-of-1.gguf")


def test_select_native_engine_artifact_model_fallback_marks_full_model() -> None:
    choice = select_native_engine_artifact(_request(), artifact_kind="model")

    assert choice is not None
    assert choice.kind == "model"
    assert choice.source == "local_binding"
    assert choice.fallback_mode == "full_model"


def test_build_native_execution_receipt_uses_assignment_choice() -> None:
    receipt = build_native_execution_receipt(_request())

    assert receipt is not None
    assert receipt["executionMode"] == "layer_range"
    assert receipt["artifactKind"] == "assignment"
    assert receipt["artifactSource"] == "materialized_assignment"
    assert receipt["layerStart"] == 0
    assert receipt["layerEnd"] == 14


def test_resolve_assignment_artifact_chunk_ranges_preserves_tensor_metadata() -> None:
    chunk_ranges = resolve_assignment_artifact_chunk_ranges(
        _request()["localArtifactResolution"]["assignmentArtifact"],
    )

    assert len(chunk_ranges) == 2
    assert chunk_ranges[0].chunk_id == "chunk-0"
    assert chunk_ranges[0].layer_start == 0
    assert chunk_ranges[0].layer_end == 7
    assert chunk_ranges[0].tensor_names == (
        "token_embd.weight",
        "blk.0.attn_q.weight",
    )


def test_resolve_assignment_artifact_chunk_ranges_rejects_invalid_tensor_names() -> None:
    artifact = dict(_request()["localArtifactResolution"]["assignmentArtifact"])
    artifact["chunkRanges"] = [
        {
            "chunkId": "chunk-0",
            "offsetBytes": 0,
            "sizeBytes": 512,
            "tensorNames": ["", 123],
        }
    ]
    try:
        resolve_assignment_artifact_chunk_ranges(artifact)
    except ValueError as exc:
        error_text = str(exc)
    else:
        error_text = ""

    assert "tensorNames is invalid" in error_text


def test_build_native_execution_receipt_returns_none_without_local_resolution() -> None:
    request = _request()
    request.pop("localArtifactResolution", None)

    assert build_native_execution_receipt(request) is None


def test_resolve_assignment_artifact_coverage_reads_sparse_layout() -> None:
    coverage = resolve_assignment_artifact_coverage(
        _request()["localArtifactResolution"]["assignmentArtifact"],
    )

    assert coverage is not None
    assert coverage.abi == ASSIGNMENT_ARTIFACT_COVERAGE_ABI
    assert coverage.materialization_mode == "sparse_full_size"
    assert coverage.covered_byte_count == 640
    assert coverage.covered_range_count == 2


def test_resolve_assignment_artifact_coverage_rejects_missing_sparse_layout() -> None:
    artifact = dict(_request()["localArtifactResolution"]["assignmentArtifact"])
    artifact.pop("coverage", None)
    try:
        resolve_assignment_artifact_coverage(artifact)
    except ValueError as exc:
        error_text = str(exc)
    else:
        error_text = ""

    assert "coverage is missing" in error_text


def test_decode_native_engine_input_payload_reads_payload_file() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        io_root = Path(tempdir).resolve()
        payload_path = (io_root / "payload.bin").resolve()
        payload = b"activation-payload"
        payload_path.write_bytes(payload)
        request = {
            "payloadFile": {
                "path": str(payload_path),
                "sizeBytes": len(payload),
                "sha256Hex": hashlib.sha256(payload).hexdigest(),
            },
            "localFileContract": {
                "abi": "cai-llama-cpp-local-file-io-v1",
                "ioRoot": str(io_root),
                "responseOutputPath": str((io_root / "out.bin").resolve()),
            },
        }

        assert decode_native_engine_input_payload(request) == payload


def test_build_native_engine_process_response_writes_output_file_and_metadata() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        io_root = Path(tempdir).resolve()
        output_path = (io_root / "out.bin").resolve()
        request = _request()
        request["localArtifactResolution"] = {
            "modelArtifact": {
                "artifactId": "gguf-main",
                "source": "local_binding",
                "localPath": "C:/models/model.gguf",
            }
        }
        request["localFileContract"] = {
            "abi": "cai-llama-cpp-local-file-io-v1",
            "ioRoot": str(io_root),
            "responseOutputPath": str(output_path),
        }
        request["outputContract"] = {
            "frameMetadataTemplate": {
                "payloadSha256Hex": "<computed-output-sha256>",
                "llmHandoff": {"tensor": {"sha256Hex": "<computed-output-sha256>"}},
            }
        }
        output = b"state-output"

        response = build_native_engine_process_response(
            request,
            output,
            metrics={"backendMode": "test"},
            artifact_kind="model",
            fallback_mode="full_model",
        )

        assert output_path.read_bytes() == output
        assert response["outputPayloadFile"]["path"] == str(output_path)
        assert response["nativeExecution"]["artifactKind"] == "model"
        assert response["nativeExecution"]["fallbackMode"] == "full_model"
        assert response["outputFrameMetadata"]["payloadSha256Hex"] == hashlib.sha256(
            output
        ).hexdigest()

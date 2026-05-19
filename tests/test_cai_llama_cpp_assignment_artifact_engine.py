# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import base64
import hashlib
from http.server import ThreadingHTTPServer
import json
import os
import sys
import tempfile
import threading
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.cai_llama_cpp_assignment_artifact_engine import (  # noqa: E402
    ASSIGNMENT_ARTIFACT_ENGINE_ID,
    ASSIGNMENT_STATE_PAYLOAD_ABI,
    CAI_LLM_ASSIGNMENT_EXECUTOR_TIMEOUT_ENV,
    CAI_LLM_SHARD_NATIVE_TIMEOUT_ENV,
    _assignment_executor_timeout_seconds,
    handle_assignment_artifact_engine_request,
    reset_assignment_artifact_engine_sessions,
)
from cai_compute_chain.cai_llama_cpp_patched_executor_host import (  # noqa: E402
    reset_patched_executor_host_clients,
)
from cai_compute_chain.cai_llama_cpp_shard_native_bridge import (  # noqa: E402
    _handler_class,
)
from cai_compute_chain.cai_owned_runtime import (  # noqa: E402
    ExternalLlamaCppShardAdapter,
    LLAMA_CPP_EXTERNAL_SHARD_HANDOFF_ABI,
    run_cai_owned_llm_shard_adapter_self_test,
)
from cai_compute_chain.model_distribution import (  # noqa: E402
    build_gguf_model_package_manifest,
    put_cached_chunk,
    save_local_artifact_binding,
    save_model_package_manifest,
)


MODEL_ID = "cai-network/Qwen3-0.6B-GGUF"


def test_assignment_executor_timeout_follows_native_timeout_env(monkeypatch) -> None:
    monkeypatch.delenv(CAI_LLM_ASSIGNMENT_EXECUTOR_TIMEOUT_ENV, raising=False)
    monkeypatch.setenv(CAI_LLM_SHARD_NATIVE_TIMEOUT_ENV, "900")

    assert _assignment_executor_timeout_seconds() == 900.0


def test_assignment_executor_timeout_dedicated_env_wins(monkeypatch) -> None:
    monkeypatch.setenv(CAI_LLM_ASSIGNMENT_EXECUTOR_TIMEOUT_ENV, "240")
    monkeypatch.setenv(CAI_LLM_SHARD_NATIVE_TIMEOUT_ENV, "900")

    assert _assignment_executor_timeout_seconds() == 240.0


def _assignment_artifact(
    path: Path,
    *,
    layer_start: int,
    layer_end: int,
    chunk_ranges: list[tuple[int, int]],
) -> dict:
    payload = path.read_bytes()
    chunks = []
    for index, (offset_bytes, size_bytes) in enumerate(chunk_ranges):
        chunk_payload = payload[offset_bytes : offset_bytes + size_bytes]
        chunks.append(
            {
                "chunkId": f"chunk-{layer_start}-{layer_end}-{index}",
                "offsetBytes": offset_bytes,
                "sizeBytes": size_bytes,
                "sha256Hex": hashlib.sha256(chunk_payload).hexdigest(),
                "layerStart": layer_start,
                "layerEnd": layer_end,
            }
        )
    covered_byte_count = sum(item["sizeBytes"] for item in chunks)
    return {
        "artifactId": "gguf-main",
        "localPath": str(path.resolve()),
        "source": "materialized_assignment",
        "sizeBytes": len(payload),
        "expectedDigest": f"assignment:{layer_start}-{layer_end}",
        "layerStart": layer_start,
        "layerEnd": layer_end,
        "chunkRanges": chunks,
        "coverage": {
            "abi": "cai-llama-cpp-assignment-coverage-v1",
            "materializationMode": "sparse_full_size",
            "artifactSizeBytes": len(payload),
            "coveredByteCount": covered_byte_count,
            "coveredRangeCount": len(chunks),
            "zeroFilledOutsideCoveredRanges": True,
        },
    }


def _request(
    action: str,
    payload: bytes,
    assignment_artifact: dict,
    *,
    layer_start: int,
    layer_end: int,
    io_root: Path | None = None,
    session_id: str = "session-assignment-test",
    final_output: bool | None = None,
    next_frame_kind: str | None = None,
    managed_runtime_root: Path | None = None,
) -> dict:
    payload_hash = hashlib.sha256(payload).hexdigest()
    if final_output is None:
        final_output = action == "process_decode"
    request = {
        "schemaVersion": 1,
        "abi": LLAMA_CPP_EXTERNAL_SHARD_HANDOFF_ABI,
        "action": action,
        "adapterId": "llama.cpp-external-shard",
        "adapterVersion": "llama.cpp-external-shard/0.1",
        "backend": "llama.cpp-patched",
        "frame": {
            "sessionId": session_id,
            "batchId": f"{session_id}-{action}",
            "modelId": MODEL_ID,
            "frameKind": "activation" if action == "process_prefill" else "decode",
            "phase": (
                "prefill_activation_batches"
                if action == "process_prefill"
                else "decode_activation_batches"
            ),
            "layerStart": layer_start,
            "layerEnd": layer_end,
            "tokenStart": 0,
            "tokenEnd": 1,
            "payloadSha256Hex": payload_hash,
            "metadata": {},
        },
        "shardSpec": {
            "modelId": MODEL_ID,
            "backend": "llama.cpp-patched",
            "requiresPatchedBackend": True,
            "frameKind": "activation" if action == "process_prefill" else "decode",
            "phase": (
                "prefill_activation_batches"
                if action == "process_prefill"
                else "decode_activation_batches"
            ),
            "layerStart": layer_start,
            "layerEnd": layer_end,
            "tokenStart": 0,
            "tokenEnd": 1,
        },
        "payloadSha256Hex": payload_hash,
        "localArtifactResolution": {
            "assignmentArtifact": dict(assignment_artifact),
        },
        "outputContract": {
            "requiresFinalOutput": bool(final_output),
            "requiresOutputFrameMetadata": not bool(final_output),
        },
    }
    if not final_output:
        request["outputContract"]["frameMetadataTemplate"] = {
            "frameKind": (
                next_frame_kind
                or ("decode" if action == "process_prefill" else "activation")
            ),
            "payloadSha256Hex": "<computed-output-sha256>",
            "llmHandoff": {
                "tensor": {"sha256Hex": "<computed-output-sha256>"},
            },
        }
    if managed_runtime_root is not None:
        session_root = managed_runtime_root / "session-root"
        state_dir = session_root / "state"
        cache_dir = session_root / "cache"
        logs_dir = session_root / "logs"
        workspace_root = state_dir / "llm-shard-execution"
        for path in (state_dir, cache_dir, logs_dir):
            path.mkdir(parents=True, exist_ok=True)
        request["managedRuntime"] = {
            "schemaVersion": 1,
            "abi": "cai-llama-cpp-managed-runtime-v1",
            "platform": os.name,
            "repoRoot": str(REPO_ROOT.resolve()),
            "runtimeRoot": str(managed_runtime_root.resolve()),
            "modelId": MODEL_ID,
            "sessionPaths": {
                "root": str(session_root.resolve()),
                "stateDir": str(state_dir.resolve()),
                "cacheDir": str(cache_dir.resolve()),
                "logsDir": str(logs_dir.resolve()),
                "stdoutLog": str((logs_dir / "stdout.log").resolve()),
                "stderrLog": str((logs_dir / "stderr.log").resolve()),
            },
        }
        request["executionWorkspace"] = {
            "schemaVersion": 1,
            "abi": "cai-llama-cpp-execution-workspace-v1",
            "root": str(workspace_root.resolve()),
            "inputsDir": str((workspace_root / "inputs").resolve()),
            "outputsDir": str((workspace_root / "outputs").resolve()),
            "stateFilesDir": str((workspace_root / "state").resolve()),
            "manifestPath": str((workspace_root / "execution-workspace.json").resolve()),
            "action": action,
            "sessionId": session_id,
            "modelId": MODEL_ID,
            "layerStart": layer_start,
            "layerEnd": layer_end,
            "tokenStart": 0,
            "tokenEnd": 1,
            "requiresFinalOutput": bool(final_output),
            "expectedOutputKind": (
                "final_output"
                if final_output
                else ("decode_state" if next_frame_kind == "decode" else "activation_state")
            ),
        }
    if io_root is None:
        request["payloadBase64"] = base64.b64encode(payload).decode("ascii")
        return request
    io_root.mkdir(parents=True, exist_ok=True)
    payload_path = (io_root / f"{action}-payload.bin").resolve()
    output_path = (io_root / f"{action}-output.bin").resolve()
    payload_path.write_bytes(payload)
    request["payloadFile"] = {
        "path": str(payload_path),
        "sizeBytes": len(payload),
        "sha256Hex": payload_hash,
    }
    request["localFileContract"] = {
        "schemaVersion": 1,
        "abi": "cai-llama-cpp-local-file-io-v1",
        "ioRoot": str(io_root.resolve()),
        "responseOutputPath": str(output_path),
    }
    return request


def _response_payload_bytes(response: dict) -> bytes:
    if "outputPayloadFile" in response:
        return Path(response["outputPayloadFile"]["path"]).read_bytes()
    return base64.b64decode(response["outputPayloadBase64"])


def test_assignment_artifact_engine_transfers_cross_shard_state() -> None:
    reset_assignment_artifact_engine_sessions()
    with tempfile.TemporaryDirectory() as tempdir:
        artifact_path = Path(tempdir) / "assignment.gguf"
        artifact_path.write_bytes(b"abcdefghijklmnop")
        prefill_artifact = _assignment_artifact(
            artifact_path,
            layer_start=0,
            layer_end=1,
            chunk_ranges=[(0, 4), (4, 4)],
        )
        decode_artifact = _assignment_artifact(
            artifact_path,
            layer_start=1,
            layer_end=2,
            chunk_ranges=[(8, 4), (12, 4)],
        )
        prefill = handle_assignment_artifact_engine_request(
            _request(
                "process_prefill",
                b"The capital of France is",
                prefill_artifact,
                layer_start=0,
                layer_end=1,
                next_frame_kind="decode",
            )
        )
        prefill_state = json.loads(_response_payload_bytes(prefill).decode("utf-8"))
        decode = handle_assignment_artifact_engine_request(
            _request(
                "process_decode",
                _response_payload_bytes(prefill),
                decode_artifact,
                layer_start=1,
                layer_end=2,
                final_output=True,
            )
        )
        decode_payload = json.loads(_response_payload_bytes(decode).decode("utf-8"))

    assert prefill["status"] == "ok"
    assert prefill["nativeExecution"]["artifactKind"] == "assignment"
    assert prefill["metrics"]["assignmentArtifactBytesRead"] == 8
    assert prefill_state["abi"] == ASSIGNMENT_STATE_PAYLOAD_ABI
    assert prefill_state["stateKind"] == "assignment_decode"
    assert prefill_state["assignmentArtifact"]["chunkCount"] == 2
    assert decode["status"] == "ok"
    assert decode["nativeExecution"]["artifactKind"] == "assignment"
    assert decode_payload["assignmentDigest"] != decode_payload["inputAssignmentDigest"]
    assert decode_payload["textUtf8"].startswith("assignment:")


def test_assignment_artifact_engine_supports_local_file_io() -> None:
    reset_assignment_artifact_engine_sessions()
    with tempfile.TemporaryDirectory() as tempdir:
        temp_root = Path(tempdir)
        artifact_path = temp_root / "assignment.gguf"
        artifact_path.write_bytes(b"abcdefghijklmnop")
        prefill_artifact = _assignment_artifact(
            artifact_path,
            layer_start=0,
            layer_end=1,
            chunk_ranges=[(0, 4), (4, 4)],
        )
        decode_artifact = _assignment_artifact(
            artifact_path,
            layer_start=1,
            layer_end=2,
            chunk_ranges=[(8, 4), (12, 4)],
        )
        prefill = handle_assignment_artifact_engine_request(
            _request(
                "process_prefill",
                b"hello",
                prefill_artifact,
                layer_start=0,
                layer_end=1,
                io_root=temp_root / "prefill-io",
                next_frame_kind="decode",
            )
        )
        decode = handle_assignment_artifact_engine_request(
            _request(
                "process_decode",
                _response_payload_bytes(prefill),
                decode_artifact,
                layer_start=1,
                layer_end=2,
                io_root=temp_root / "decode-io",
                final_output=True,
            )
        )
        decode_output = _response_payload_bytes(decode)

    assert prefill["status"] == "ok"
    assert "outputPayloadFile" in prefill
    assert decode["status"] == "ok"
    assert "outputPayloadFile" in decode
    assert b"assignment:" in decode_output


def test_assignment_artifact_engine_reuses_loaded_session_and_clears_on_finalize() -> None:
    reset_assignment_artifact_engine_sessions()
    with tempfile.TemporaryDirectory() as tempdir:
        artifact_path = Path(tempdir) / "assignment.gguf"
        artifact_path.write_bytes(b"abcdefghijklmnop")
        prefill_artifact = _assignment_artifact(
            artifact_path,
            layer_start=0,
            layer_end=1,
            chunk_ranges=[(0, 4), (4, 4)],
        )
        load_request = _request(
            "load_shard",
            b"",
            prefill_artifact,
            layer_start=0,
            layer_end=1,
            session_id="session-loaded",
        )
        prefill_request = _request(
            "process_prefill",
            b"hello",
            prefill_artifact,
            layer_start=0,
            layer_end=1,
            session_id="session-loaded",
            next_frame_kind="decode",
        )
        finalize_request = _request(
            "finalize",
            b"",
            prefill_artifact,
            layer_start=0,
            layer_end=1,
            session_id="session-loaded",
        )
        load_response = handle_assignment_artifact_engine_request(load_request)
        prefill_response = handle_assignment_artifact_engine_request(prefill_request)
        finalize_response = handle_assignment_artifact_engine_request(finalize_request)
        second_prefill = handle_assignment_artifact_engine_request(prefill_request)

    assert load_response["status"] == "ready"
    assert load_response["metrics"]["assignmentArtifactSessionLoaded"] is True
    assert load_response["metrics"]["assignmentArtifactBytesRead"] == 8
    assert load_response["metrics"]["assignmentArtifactResidentShardLoaded"] is True
    assert load_response["metrics"]["assignmentArtifactCoverageMode"] == "sparse_full_size"
    assert prefill_response["metrics"]["assignmentArtifactSessionCacheHit"] is True
    assert prefill_response["metrics"]["assignmentArtifactBytesRead"] == 0
    assert finalize_response["metrics"]["assignmentArtifactSessionReleased"] is True
    assert second_prefill["metrics"]["assignmentArtifactSessionCacheHit"] is False
    assert second_prefill["metrics"]["assignmentArtifactResidentShardHit"] is True
    assert second_prefill["metrics"]["assignmentArtifactBytesRead"] == 0


def test_assignment_artifact_engine_reuses_resident_shard_across_sessions() -> None:
    reset_assignment_artifact_engine_sessions()
    with tempfile.TemporaryDirectory() as tempdir:
        artifact_path = Path(tempdir) / "assignment.gguf"
        artifact_path.write_bytes(b"abcdefghijklmnop")
        prefill_artifact = _assignment_artifact(
            artifact_path,
            layer_start=0,
            layer_end=1,
            chunk_ranges=[(0, 4), (4, 4)],
        )
        first_prefill = handle_assignment_artifact_engine_request(
            _request(
                "process_prefill",
                b"hello",
                prefill_artifact,
                layer_start=0,
                layer_end=1,
                session_id="session-first",
                next_frame_kind="decode",
            )
        )
        handle_assignment_artifact_engine_request(
            _request(
                "finalize",
                b"",
                prefill_artifact,
                layer_start=0,
                layer_end=1,
                session_id="session-first",
            )
        )
        second_prefill = handle_assignment_artifact_engine_request(
            _request(
                "process_prefill",
                b"hello",
                prefill_artifact,
                layer_start=0,
                layer_end=1,
                session_id="session-second",
                next_frame_kind="decode",
            )
        )

    assert first_prefill["metrics"]["assignmentArtifactResidentShardLoaded"] is True
    assert first_prefill["metrics"]["assignmentArtifactResidentShardHit"] is False
    assert first_prefill["metrics"]["assignmentArtifactBytesRead"] == 8
    assert first_prefill["metrics"]["assignmentArtifactCoveredByteCount"] == 8
    assert second_prefill["metrics"]["assignmentArtifactSessionCacheHit"] is False
    assert second_prefill["metrics"]["assignmentArtifactSessionLoaded"] is True
    assert second_prefill["metrics"]["assignmentArtifactResidentShardHit"] is True
    assert second_prefill["metrics"]["assignmentArtifactResidentShardLoaded"] is False
    assert second_prefill["metrics"]["assignmentArtifactBytesRead"] == 0


def test_assignment_artifact_engine_rejects_session_drift_after_load() -> None:
    reset_assignment_artifact_engine_sessions()
    with tempfile.TemporaryDirectory() as tempdir:
        artifact_path = Path(tempdir) / "assignment.gguf"
        artifact_path.write_bytes(b"abcdefghijklmnop")
        loaded_artifact = _assignment_artifact(
            artifact_path,
            layer_start=0,
            layer_end=1,
            chunk_ranges=[(0, 4), (4, 4)],
        )
        drifted_artifact = _assignment_artifact(
            artifact_path,
            layer_start=0,
            layer_end=1,
            chunk_ranges=[(8, 4), (12, 4)],
        )
        handle_assignment_artifact_engine_request(
            _request(
                "load_shard",
                b"",
                loaded_artifact,
                layer_start=0,
                layer_end=1,
                session_id="session-drift",
            )
        )
        try:
            handle_assignment_artifact_engine_request(
                _request(
                    "process_prefill",
                    b"hello",
                    drifted_artifact,
                    layer_start=0,
                    layer_end=1,
                    session_id="session-drift",
                    next_frame_kind="decode",
                )
            )
        except ValueError as exc:
            error_text = str(exc)
        else:
            error_text = ""

    assert "assignment session drifted" in error_text


def test_assignment_artifact_engine_rejects_missing_sparse_coverage() -> None:
    reset_assignment_artifact_engine_sessions()
    with tempfile.TemporaryDirectory() as tempdir:
        artifact_path = Path(tempdir) / "assignment.gguf"
        artifact_path.write_bytes(b"abcdefghijklmnop")
        artifact = _assignment_artifact(
            artifact_path,
            layer_start=0,
            layer_end=1,
            chunk_ranges=[(0, 4), (4, 4)],
        )
        artifact.pop("coverage", None)
        try:
            handle_assignment_artifact_engine_request(
                _request(
                    "process_prefill",
                    b"hello",
                    artifact,
                    layer_start=0,
                    layer_end=1,
                    next_frame_kind="decode",
                )
            )
        except ValueError as exc:
            error_text = str(exc)
        else:
            error_text = ""

    assert "coverage is missing" in error_text


def test_assignment_artifact_engine_rejects_layer_gap_in_chunk_ranges() -> None:
    reset_assignment_artifact_engine_sessions()
    with tempfile.TemporaryDirectory() as tempdir:
        artifact_path = Path(tempdir) / "assignment.gguf"
        artifact_path.write_bytes(b"abcdefghijklmnop")
        artifact = _assignment_artifact(
            artifact_path,
            layer_start=0,
            layer_end=3,
            chunk_ranges=[(0, 4), (4, 4)],
        )
        artifact["chunkRanges"][0]["layerEnd"] = 1
        artifact["chunkRanges"][1]["layerStart"] = 2
        artifact["chunkRanges"][1]["layerEnd"] = 3
        try:
            handle_assignment_artifact_engine_request(
                _request(
                    "process_prefill",
                    b"hello",
                    artifact,
                    layer_start=0,
                    layer_end=3,
                    next_frame_kind="decode",
                )
            )
        except ValueError as exc:
            error_text = str(exc)
        else:
            error_text = ""

    assert "missing layers 1..2" in error_text


def test_assignment_artifact_engine_preserves_intermediate_decode_state() -> None:
    reset_assignment_artifact_engine_sessions()
    with tempfile.TemporaryDirectory() as tempdir:
        artifact_path = Path(tempdir) / "assignment.gguf"
        artifact_path.write_bytes(b"abcdefghijklmnop")
        shard_a = _assignment_artifact(
            artifact_path,
            layer_start=0,
            layer_end=1,
            chunk_ranges=[(0, 4), (4, 4)],
        )
        shard_b = _assignment_artifact(
            artifact_path,
            layer_start=1,
            layer_end=2,
            chunk_ranges=[(8, 4), (12, 4)],
        )
        prefill = handle_assignment_artifact_engine_request(
            _request(
                "process_prefill",
                b"hello decentralized world",
                shard_b,
                layer_start=1,
                layer_end=2,
                next_frame_kind="decode",
            )
        )
        intermediate_decode = handle_assignment_artifact_engine_request(
            _request(
                "process_decode",
                _response_payload_bytes(prefill),
                shard_a,
                layer_start=0,
                layer_end=1,
                final_output=False,
                next_frame_kind="decode",
            )
        )
        intermediate_state = json.loads(
            _response_payload_bytes(intermediate_decode).decode("utf-8")
        )
        final_decode = handle_assignment_artifact_engine_request(
            _request(
                "process_decode",
                _response_payload_bytes(intermediate_decode),
                shard_b,
                layer_start=1,
                layer_end=2,
                final_output=True,
            )
        )
        final_payload = json.loads(_response_payload_bytes(final_decode).decode("utf-8"))

    assert prefill["metrics"]["assignmentStateKind"] == "assignment_decode"
    assert prefill["metrics"]["assignmentFinalOutput"] is False
    assert intermediate_decode["metrics"]["assignmentStateKind"] == "assignment_decode"
    assert intermediate_decode["metrics"]["assignmentFinalOutput"] is False
    assert intermediate_state["stateKind"] == "assignment_decode"
    assert intermediate_state["inputStateKind"] == "assignment_decode"
    assert intermediate_state["stateDigest"]
    assert final_decode["metrics"]["assignmentStateKind"] == "final_output"
    assert final_decode["metrics"]["assignmentFinalOutput"] is True
    assert final_payload["stateDigest"] == intermediate_state["stateDigest"]
    assert final_payload["textUtf8"].startswith("assignment:")


def test_assignment_artifact_engine_stages_managed_runtime_session_files() -> None:
    reset_assignment_artifact_engine_sessions()
    with tempfile.TemporaryDirectory() as tempdir:
        temp_root = Path(tempdir)
        artifact_path = temp_root / "assignment.gguf"
        artifact_path.write_bytes(b"abcdefghijklmnop")
        runtime_root = temp_root / "managed-runtime"
        shard_a = _assignment_artifact(
            artifact_path,
            layer_start=0,
            layer_end=1,
            chunk_ranges=[(0, 4), (4, 4)],
        )
        shard_b = _assignment_artifact(
            artifact_path,
            layer_start=1,
            layer_end=2,
            chunk_ranges=[(8, 4), (12, 4)],
        )
        prefill = handle_assignment_artifact_engine_request(
            _request(
                "process_prefill",
                b"hello managed runtime",
                shard_a,
                layer_start=0,
                layer_end=1,
                next_frame_kind="decode",
                managed_runtime_root=runtime_root,
                session_id="session-managed",
            )
        )
        decode = handle_assignment_artifact_engine_request(
            _request(
                "process_decode",
                _response_payload_bytes(prefill),
                shard_b,
                layer_start=1,
                layer_end=2,
                final_output=True,
                managed_runtime_root=runtime_root,
                session_id="session-managed",
            )
        )
        finalize = handle_assignment_artifact_engine_request(
            _request(
                "finalize",
                b"",
                shard_b,
                layer_start=1,
                layer_end=2,
                managed_runtime_root=runtime_root,
                session_id="session-managed",
            )
        )
        manifest_path = Path(
            prefill["metrics"]["assignmentManagedSessionManifestPath"]
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        workspace_manifest_path = Path(
            prefill["metrics"]["assignmentExecutionWorkspaceManifestPath"]
        )
        workspace_manifest = json.loads(
            workspace_manifest_path.read_text(encoding="utf-8")
        )
        assert prefill["metrics"]["assignmentManagedRuntimeUsed"] is True
        assert decode["metrics"]["assignmentManagedRuntimeUsed"] is True
        assert finalize["metrics"]["assignmentManagedSessionFinalized"] is True
        assert manifest["abi"] == "cai-llama-cpp-assignment-managed-session-v1"
        assert manifest["sessionId"] == "session-managed"
        assert manifest["latestStateKind"] == "final_output"
        assert workspace_manifest["abi"] == "cai-llama-cpp-execution-workspace-v1"
        assert workspace_manifest["expectedOutputKind"] == "final_output"
        assert workspace_manifest["lastOutputPath"] == manifest["lastOutputPath"]
        assert Path(manifest["lastInputPath"]).exists()
        assert Path(manifest["lastOutputPath"]).exists()
        assert "llm-shard-execution" in str(workspace_manifest_path)


def test_assignment_artifact_engine_uses_external_executor_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_assignment_artifact_engine_sessions()
    artifact_path = tmp_path / "assignment.gguf"
    artifact_path.write_bytes(b"abcdefghijklmnop")
    runtime_root = tmp_path / "managed-runtime"
    shard_a = _assignment_artifact(
        artifact_path,
        layer_start=0,
        layer_end=1,
        chunk_ranges=[(0, 4), (4, 4)],
    )
    shard_b = _assignment_artifact(
        artifact_path,
        layer_start=1,
        layer_end=2,
        chunk_ranges=[(8, 4), (12, 4)],
    )
    executor_script = tmp_path / "assignment_executor.py"
    executor_script.write_text(
        """
import hashlib
import json
import sys
from pathlib import Path

request = json.loads(sys.stdin.read() or "{}")
input_payload = Path(request["inputPayloadFile"]["path"]).read_bytes()
output_path = Path(request["expectedOutputPayloadPath"])
if request["action"] == "process_prefill":
    payload = json.dumps(
        {
            "schemaVersion": 1,
            "abi": "executor-state-v1",
            "stateKind": "decode_state",
            "textUtf8": input_payload.decode("utf-8", errors="replace"),
        },
        sort_keys=True,
    ).encode("utf-8")
    output_kind = "decode_state"
else:
    state = json.loads(input_payload.decode("utf-8") or "{}")
    payload = json.dumps(
        {
            "schemaVersion": 1,
            "abi": "executor-output-v1",
            "textUtf8": "executor:" + str(state.get("textUtf8") or ""),
        },
        sort_keys=True,
    ).encode("utf-8")
    output_kind = "final_output"
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_bytes(payload)
print(
    json.dumps(
        {
            "status": "ok",
            "outputKind": output_kind,
            "realModelExecution": True,
            "outputPayloadFile": {
                "path": str(output_path),
                "sizeBytes": len(payload),
                "sha256Hex": hashlib.sha256(payload).hexdigest(),
            },
            "metrics": {"executorMode": "fake"},
        },
        sort_keys=True,
    )
)
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "CAI_LLM_SHARD_ASSIGNMENT_EXECUTOR_COMMAND",
        subprocess.list2cmdline([sys.executable, str(executor_script)]),
    )

    prefill = handle_assignment_artifact_engine_request(
        _request(
            "process_prefill",
            b"hello executor path",
            shard_a,
            layer_start=0,
            layer_end=1,
            next_frame_kind="decode",
            managed_runtime_root=runtime_root,
            session_id="session-executor",
        )
    )
    prefill_payload = json.loads(_response_payload_bytes(prefill).decode("utf-8"))
    decode = handle_assignment_artifact_engine_request(
        _request(
            "process_decode",
            _response_payload_bytes(prefill),
            shard_b,
            layer_start=1,
            layer_end=2,
            final_output=True,
            managed_runtime_root=runtime_root,
            session_id="session-executor",
        )
    )
    decode_payload = json.loads(_response_payload_bytes(decode).decode("utf-8"))

    assert prefill["metrics"]["assignmentExecutorUsed"] is True
    assert prefill["metrics"]["assignmentExecutorOutputKind"] == "decode_state"
    assert prefill["metrics"]["assignmentExecutorRealModelExecution"] is True
    assert prefill["metrics"]["assignmentExecutorMetrics"]["executorMode"] == "fake"
    assert prefill_payload["abi"] == "executor-state-v1"
    assert decode["metrics"]["assignmentExecutorUsed"] is True
    assert decode["metrics"]["assignmentExecutorOutputKind"] == "final_output"
    assert decode_payload["abi"] == "executor-output-v1"
    assert decode_payload["textUtf8"] == "executor:hello executor path"


def test_assignment_artifact_engine_advertises_production_boundary_after_shard_only_load(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_assignment_artifact_engine_sessions()
    artifact_path = tmp_path / "assignment.gguf"
    artifact_path.write_bytes(b"abcdefghijklmnop")
    shard_a = _assignment_artifact(
        artifact_path,
        layer_start=0,
        layer_end=1,
        chunk_ranges=[(0, 4), (4, 4)],
    )
    executor_script = tmp_path / "assignment_shard_only_executor.py"
    executor_script.write_text(
        """
import json
import sys

request = json.loads(sys.stdin.read() or "{}")
print(
    json.dumps(
        {
            "status": "ready",
            "realModelExecution": True,
            "metrics": {
                "assignmentArtifactPresent": True,
                "usedFullModelForLayerRange": False,
                "shardOnlyLoadingReady": True,
            },
        },
        sort_keys=True,
    )
)
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "CAI_LLM_SHARD_ASSIGNMENT_EXECUTOR_COMMAND",
        subprocess.list2cmdline([sys.executable, str(executor_script)]),
    )
    request = _request(
        "load_shard",
        b"",
        shard_a,
        layer_start=0,
        layer_end=1,
        managed_runtime_root=tmp_path / "managed-runtime",
        session_id="session-production-boundary",
    )
    request["productionRequirements"] = {
        "requiresShardOnlyLoading": True,
        "forbidFullModelFallback": True,
    }

    response = handle_assignment_artifact_engine_request(request)

    assert response["patchBoundary"]["extraMetadata"]["productionReady"] is True
    assert "gguf_layer_execution" in response["patchBoundary"]["capabilities"]
    assert "real_activation_state" in response["patchBoundary"]["capabilities"]
    assert "real_decode_state" in response["patchBoundary"]["capabilities"]
    assert (
        response["patchBoundary"]["extraMetadata"]["productionStateContract"]["abi"]
        == "cai-llama-cpp-production-state-contract-v1"
    )


def test_assignment_artifact_engine_generation_probe_uses_assignment_executor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_assignment_artifact_engine_sessions()
    wallet_home = tmp_path / "wallet-home"
    monkeypatch.setenv("CAI_WALLET_HOME", str(wallet_home))
    gguf_payload = b"abcdefghijklmnop"
    gguf_path = tmp_path / "Qwen3-0.6B-Q8_0.gguf"
    gguf_path.write_bytes(gguf_payload)
    manifest = build_gguf_model_package_manifest(
        catalog_id="cai-private",
        model_id=MODEL_ID,
        version="2026.05",
        gguf_path=gguf_path,
        total_layers=2,
        min_chunk_bytes=1,
        max_chunk_bytes=8,
        target_chunk_count=2,
        family="Qwen3",
        quantization="Q8_0",
    )
    save_model_package_manifest(manifest)
    save_local_artifact_binding(
        manifest.catalog_id,
        manifest.version,
        artifact_id=manifest.files[0].artifact_id,
        local_path=gguf_path,
    )
    executor_script = tmp_path / "assignment_generation_executor.py"
    executor_script.write_text(
        """
import hashlib
import json
import sys
from pathlib import Path

request = json.loads(sys.stdin.read() or "{}")
action = str(request.get("action") or "")
metrics = {
    "assignmentArtifactPresent": True,
    "usedFullModelForLayerRange": False,
    "shardOnlyLoadingReady": True,
    "realLayerExecution": True,
}
if action == "load_shard":
    print(json.dumps({"status": "ready", "realModelExecution": True, "metrics": metrics}))
    raise SystemExit(0)
if action == "process_prefill":
    input_payload = Path(request["inputPayloadFile"]["path"]).read_bytes()
    payload = json.dumps(
        {
            "schemaVersion": 1,
            "stateKind": request["expectedOutputKind"],
            "textUtf8": input_payload.decode("utf-8", errors="replace"),
        },
        sort_keys=True,
    ).encode("utf-8")
else:
    payload = b"ok"
output_path = Path(request["expectedOutputPayloadPath"])
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_bytes(payload)
print(
    json.dumps(
        {
            "status": "ok",
            "outputKind": request["expectedOutputKind"],
            "realModelExecution": True,
            "outputPayloadFile": {
                "path": str(output_path),
                "sizeBytes": len(payload),
                "sha256Hex": hashlib.sha256(payload).hexdigest(),
            },
            "metrics": metrics,
        },
        sort_keys=True,
    )
)
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "CAI_LLM_SHARD_ASSIGNMENT_EXECUTOR_COMMAND",
        subprocess.list2cmdline([sys.executable, str(executor_script)]),
    )

    response = handle_assignment_artifact_engine_request(
        {
            "schemaVersion": 1,
            "abi": LLAMA_CPP_EXTERNAL_SHARD_HANDOFF_ABI,
            "action": "probe_generation",
            "backend": "llama.cpp-patched",
            "generationProbe": {
                "schemaVersion": 1,
                "abi": "cai-llama-cpp-generation-probe-v1",
                "modelId": MODEL_ID,
                "prompt": "CAI generation probe",
                "maxTokens": 1,
                "requiresRealModelExecution": True,
            },
            "localArtifactResolution": {
                "catalogId": manifest.catalog_id,
                "version": manifest.version,
            },
            "productionRequirements": {
                "requiresShardOnlyLoading": True,
                "forbidFullModelFallback": True,
            },
        }
    )

    assert response["generationProbe"]["ready"] is True
    assert response["generationProbe"]["realModelExecution"] is True
    assert response["generationProbe"]["realLayerExecution"] is True
    assert response["generationProbe"]["outputText"] == "ok"
    assert response["metrics"]["generationProbeShardOnlyLoadingReady"] is True


def test_assignment_artifact_engine_marks_full_model_executor_as_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_assignment_artifact_engine_sessions()
    artifact_path = tmp_path / "assignment.gguf"
    artifact_path.write_bytes(b"abcdefghijklmnop")
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"full-model-bytes")
    shard_a = _assignment_artifact(
        artifact_path,
        layer_start=0,
        layer_end=1,
        chunk_ranges=[(0, 4), (4, 4)],
    )
    executor_script = tmp_path / "assignment_full_model_executor.py"
    executor_script.write_text(
        """
import hashlib
import json
import sys
from pathlib import Path

request = json.loads(sys.stdin.read() or "{}")
payload = json.dumps(
    {
        "schemaVersion": 1,
        "abi": "executor-state-v1",
        "stateKind": request["expectedOutputKind"],
    },
    sort_keys=True,
).encode("utf-8")
output_path = Path(request["expectedOutputPayloadPath"])
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_bytes(payload)
print(
    json.dumps(
        {
            "status": "ok",
            "outputKind": request["expectedOutputKind"],
            "realModelExecution": True,
            "outputPayloadFile": {
                "path": str(output_path),
                "sizeBytes": len(payload),
                "sha256Hex": hashlib.sha256(payload).hexdigest(),
            },
            "metrics": {
                "executorMode": "fake-full-model",
                "patchedEngineMetrics": {
                    "patchedBinaryMetrics": {
                        "assignmentArtifactPresent": True,
                        "usedFullModelForLayerRange": True,
                        "shardOnlyLoadingReady": False,
                    }
                },
            },
        },
        sort_keys=True,
    )
)
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "CAI_LLM_SHARD_ASSIGNMENT_EXECUTOR_COMMAND",
        subprocess.list2cmdline([sys.executable, str(executor_script)]),
    )
    request = _request(
        "process_prefill",
        b"hello executor path",
        shard_a,
        layer_start=0,
        layer_end=1,
        next_frame_kind="decode",
        managed_runtime_root=tmp_path / "managed-runtime",
        session_id="session-full-model-fallback",
    )
    request["localArtifactResolution"]["modelArtifact"] = {
        "artifactId": "gguf-main-full",
        "source": "local_binding",
        "localPath": str(model_path.resolve()),
        "sizeBytes": int(model_path.stat().st_size),
        "expectedSizeBytes": int(model_path.stat().st_size),
    }

    prefill = handle_assignment_artifact_engine_request(request)

    assert prefill["nativeExecution"]["artifactKind"] == "model"
    assert prefill["nativeExecution"]["fallbackMode"] == "full_model"
    assert (
        prefill["metrics"]["assignmentExecutorMetrics"]["patchedEngineMetrics"][
            "patchedBinaryMetrics"
        ]["usedFullModelForLayerRange"]
        is True
    )


def test_assignment_artifact_engine_rejects_full_model_when_shard_only_required(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_assignment_artifact_engine_sessions()
    artifact_path = tmp_path / "assignment.gguf"
    artifact_path.write_bytes(b"abcdefghijklmnop")
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"full-model-bytes")
    shard_a = _assignment_artifact(
        artifact_path,
        layer_start=0,
        layer_end=1,
        chunk_ranges=[(0, 4), (4, 4)],
    )
    executor_script = tmp_path / "assignment_full_model_executor.py"
    executor_script.write_text(
        """
import hashlib
import json
import sys
from pathlib import Path

request = json.loads(sys.stdin.read() or "{}")
payload = b"state"
output_path = Path(request["expectedOutputPayloadPath"])
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_bytes(payload)
print(
    json.dumps(
        {
            "status": "ok",
            "outputKind": request["expectedOutputKind"],
            "realModelExecution": True,
            "outputPayloadFile": {
                "path": str(output_path),
                "sizeBytes": len(payload),
                "sha256Hex": hashlib.sha256(payload).hexdigest(),
            },
            "metrics": {
                "usedFullModelForLayerRange": True,
                "shardOnlyLoadingReady": False,
            },
        },
        sort_keys=True,
    )
)
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "CAI_LLM_SHARD_ASSIGNMENT_EXECUTOR_COMMAND",
        subprocess.list2cmdline([sys.executable, str(executor_script)]),
    )
    request = _request(
        "process_prefill",
        b"hello executor path",
        shard_a,
        layer_start=0,
        layer_end=1,
        next_frame_kind="decode",
        managed_runtime_root=tmp_path / "managed-runtime",
        session_id="session-shard-only-required",
    )
    request["localArtifactResolution"]["modelArtifact"] = {
        "artifactId": "gguf-main-full",
        "source": "local_binding",
        "localPath": str(model_path.resolve()),
        "sizeBytes": int(model_path.stat().st_size),
        "expectedSizeBytes": int(model_path.stat().st_size),
    }
    request["productionRequirements"] = {
        "schemaVersion": 1,
        "requiresShardOnlyLoading": True,
        "forbidFullModelFallback": True,
    }

    try:
        handle_assignment_artifact_engine_request(request)
    except ValueError as exc:
        assert "shard-only loading is required" in str(exc)
    else:
        raise AssertionError("expected shard-only guard to reject full-model executor")


def test_assignment_artifact_engine_rejects_slot_state_load_when_shard_only_required(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_assignment_artifact_engine_sessions()
    artifact_path = tmp_path / "assignment.gguf"
    artifact_path.write_bytes(b"abcdefghijklmnop")
    shard_a = _assignment_artifact(
        artifact_path,
        layer_start=0,
        layer_end=1,
        chunk_ranges=[(0, 4), (4, 4)],
    )
    executor_script = tmp_path / "assignment_slot_state_executor.py"
    executor_script.write_text(
        """
import json
import sys

json.loads(sys.stdin.read() or "{}")
print(
    json.dumps(
        {
            "status": "ready",
            "realModelExecution": True,
            "metrics": {
                "executorBackendMode": "assignment_slot_state_executor",
                "slotStateReferenceBackend": True,
                "shardOnlyLoadingReady": False,
                "slotStateMetrics": {"backendMode": "llama.cpp-slot-state"},
            },
        },
        sort_keys=True,
    )
)
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "CAI_LLM_SHARD_ASSIGNMENT_EXECUTOR_COMMAND",
        subprocess.list2cmdline([sys.executable, str(executor_script)]),
    )
    request = _request(
        "load_shard",
        b"",
        shard_a,
        layer_start=0,
        layer_end=1,
        managed_runtime_root=tmp_path / "managed-runtime",
        session_id="session-slot-state-production-reject",
    )
    request["productionRequirements"] = {
        "schemaVersion": 1,
        "requiresShardOnlyLoading": True,
        "forbidFullModelFallback": True,
    }

    try:
        handle_assignment_artifact_engine_request(request)
    except ValueError as exc:
        assert "reference" in str(exc)
        assert "shard-only loading is required" in str(exc)
    else:
        raise AssertionError("expected shard-only guard to reject slot_state executor")


def test_assignment_artifact_engine_uses_patched_executor_host(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_assignment_artifact_engine_sessions()
    reset_patched_executor_host_clients()
    existing_pythonpath = str(os.environ.get("PYTHONPATH") or "")
    pythonpath_parts = [str(SRC_ROOT)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(pythonpath_parts))
    artifact_path = tmp_path / "assignment.gguf"
    artifact_path.write_bytes(b"abcdefghijklmnop")
    shard_a = _assignment_artifact(
        artifact_path,
        layer_start=0,
        layer_end=1,
        chunk_ranges=[(0, 4), (4, 4)],
    )
    patched_engine_script = tmp_path / "fake_patched_engine.py"
    patched_engine_script.write_text(
        """
import hashlib
import json
import sys
from pathlib import Path

request = json.loads(sys.stdin.read() or "{}")
state_dir = Path(request["executionWorkspace"]["stateFilesDir"])
state_dir.mkdir(parents=True, exist_ok=True)
state_payload = b"host-wrapped-real-state"
state_path = Path(
    request["validatedExecutionContext"]["ioTargets"]["outputStateFilePath"]
)
state_path.write_bytes(state_payload)
manifest = {
    "schemaVersion": 1,
    "abi": "cai-llama-cpp-real-state-payload-v1",
    "stateKind": request["expectedOutputKind"],
    "producedByAction": request["action"],
    "modelId": request["modelId"],
    "sessionId": request["sessionId"],
    "layerStart": request["layerStart"],
    "layerEnd": request["layerEnd"],
    "tokenStart": request["tokenStart"],
    "tokenEnd": request["tokenEnd"],
    "stateFormat": "ggml-kv-cache-v1/token-step-kv-cache-v1",
    "stateFile": {
        "path": str(state_path.resolve()),
        "sha256Hex": hashlib.sha256(state_payload).hexdigest(),
        "sizeBytes": len(state_payload),
    },
}
payload = json.dumps(manifest, sort_keys=True).encode("utf-8")
output_path = Path(request["expectedOutputPayloadPath"])
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_bytes(payload)
print(
    json.dumps(
        {
            "status": "ok",
            "outputKind": request["expectedOutputKind"],
            "realModelExecution": True,
            "outputPayloadFile": {
                "path": str(output_path.resolve()),
                "sizeBytes": len(payload),
                "sha256Hex": hashlib.sha256(payload).hexdigest(),
            },
            "outputPayloadSha256Hex": hashlib.sha256(payload).hexdigest(),
            "metrics": {"engine": "fake-patched"},
        },
        sort_keys=True,
    )
)
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "CAI_LLM_SHARD_ASSIGNMENT_EXECUTOR_COMMAND",
        subprocess.list2cmdline(
            [
                sys.executable,
                "-m",
                "cai_compute_chain.cai_llama_cpp_patched_executor_host",
            ]
        ),
    )
    monkeypatch.setenv(
        "CAI_LLM_SHARD_PATCHED_ENGINE_COMMAND",
        subprocess.list2cmdline([sys.executable, str(patched_engine_script)]),
    )

    prefill = handle_assignment_artifact_engine_request(
        _request(
            "process_prefill",
            b"hello patched host",
            shard_a,
            layer_start=0,
            layer_end=1,
            next_frame_kind="decode",
            managed_runtime_root=tmp_path / "managed-runtime",
            session_id="session-patched-host",
        )
    )
    prefill_payload = json.loads(_response_payload_bytes(prefill).decode("utf-8"))

    assert prefill["metrics"]["assignmentExecutorUsed"] is True
    assert prefill["metrics"]["assignmentExecutorRealModelExecution"] is True
    assert (
        prefill["metrics"]["assignmentExecutorMetrics"]["executorBackendMode"]
        == "patched_executor_host"
    )
    assert (
        prefill["metrics"]["assignmentExecutorMetrics"]["patchedEngineMetrics"][
            "engine"
        ]
        == "fake-patched"
    )
    assert (
        prefill["metrics"]["assignmentExecutorMetrics"]["validatedStateKind"]
        == "decode_state"
    )
    assert prefill_payload["abi"] == "cai-llama-cpp-real-state-payload-v1"


def test_assignment_artifact_engine_reuses_persistent_external_executor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_assignment_artifact_engine_sessions()
    artifact_path = tmp_path / "assignment.gguf"
    artifact_path.write_bytes(b"abcdefghijklmnop")
    shard_a = _assignment_artifact(
        artifact_path,
        layer_start=0,
        layer_end=1,
        chunk_ranges=[(0, 4), (4, 4)],
    )
    shard_b = _assignment_artifact(
        artifact_path,
        layer_start=1,
        layer_end=2,
        chunk_ranges=[(8, 4), (12, 4)],
    )
    executor_script = tmp_path / "assignment_executor_jsonl.py"
    executor_script.write_text(
        """
import base64
import json
import os
import sys

call_count = 0
for line in sys.stdin:
    request = json.loads(line or "{}")
    call_count += 1
    action = str(request.get("action") or "")
    payload = json.dumps(
        {
            "schemaVersion": 1,
            "abi": "persistent-executor-output-v1",
            "callCount": call_count,
            "pid": os.getpid(),
            "action": action,
        },
        sort_keys=True,
    ).encode("utf-8")
    response = {
        "status": "ready" if action == "load_shard" else "ok",
        "realModelExecution": True,
        "metrics": {"executorPid": os.getpid(), "callCount": call_count},
    }
    if action in {"process_prefill", "process_decode"}:
        response["outputKind"] = request.get("expectedOutputKind")
        response["outputPayloadBase64"] = base64.b64encode(payload).decode("ascii")
    print(
        json.dumps(response, sort_keys=True),
        flush=True,
    )
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "CAI_LLM_SHARD_ASSIGNMENT_EXECUTOR_COMMAND",
        subprocess.list2cmdline([sys.executable, str(executor_script)]),
    )
    monkeypatch.setenv("CAI_LLM_SHARD_ASSIGNMENT_EXECUTOR_PERSISTENT", "1")
    runtime_root = tmp_path / "persistent-managed-runtime"
    load = handle_assignment_artifact_engine_request(
        _request(
            "load_shard",
            b"",
            shard_a,
            layer_start=0,
            layer_end=1,
            session_id="session-persistent-executor",
            managed_runtime_root=runtime_root,
        )
    )

    prefill = handle_assignment_artifact_engine_request(
        _request(
            "process_prefill",
            b"hello persistent executor",
            shard_a,
            layer_start=0,
            layer_end=1,
            next_frame_kind="decode",
            session_id="session-persistent-executor",
            managed_runtime_root=runtime_root,
        )
    )
    decode = handle_assignment_artifact_engine_request(
        _request(
            "process_decode",
            _response_payload_bytes(prefill),
            shard_b,
            layer_start=1,
            layer_end=2,
            final_output=True,
            session_id="session-persistent-executor",
            managed_runtime_root=runtime_root,
        )
    )
    finalize = handle_assignment_artifact_engine_request(
        _request(
            "finalize",
            b"",
            shard_b,
            layer_start=1,
            layer_end=2,
            session_id="session-persistent-executor",
            managed_runtime_root=runtime_root,
        )
    )
    prefill_payload = json.loads(_response_payload_bytes(prefill).decode("utf-8"))
    decode_payload = json.loads(_response_payload_bytes(decode).decode("utf-8"))

    assert load["metrics"]["assignmentExecutorMode"] == "persistent_jsonl"
    assert prefill["metrics"]["assignmentExecutorMode"] == "persistent_jsonl"
    assert decode["metrics"]["assignmentExecutorMode"] == "persistent_jsonl"
    assert finalize["metrics"]["assignmentExecutorMode"] == "persistent_jsonl"
    assert load["metrics"]["assignmentExecutorStatus"] == "ready"
    assert finalize["metrics"]["assignmentExecutorStatus"] == "ok"
    assert load["metrics"]["assignmentExecutorMetrics"]["callCount"] == 1
    assert prefill["metrics"]["assignmentExecutorMetrics"]["callCount"] == 2
    assert decode["metrics"]["assignmentExecutorMetrics"]["callCount"] == 3
    assert finalize["metrics"]["assignmentExecutorMetrics"]["callCount"] == 4
    assert (
        load["metrics"]["assignmentExecutorMetrics"]["executorPid"]
        == prefill["metrics"]["assignmentExecutorMetrics"]["executorPid"]
        == decode["metrics"]["assignmentExecutorMetrics"]["executorPid"]
        == finalize["metrics"]["assignmentExecutorMetrics"]["executorPid"]
    )
    assert prefill_payload["callCount"] == 2
    assert decode_payload["callCount"] == 3
    assert prefill_payload["pid"] == decode_payload["pid"]


def test_assignment_artifact_engine_self_test_passes_contract_behind_native_bridge(
    tmp_path: Path,
    monkeypatch,
) -> None:
    wallet_home = tmp_path / "wallet-home"
    monkeypatch.setenv("CAI_WALLET_HOME", str(wallet_home))
    existing_pythonpath = str(os.environ.get("PYTHONPATH") or "")
    pythonpath_parts = [str(SRC_ROOT)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(pythonpath_parts))
    gguf_payload = b"abcdefghijklmnop"
    gguf_path = tmp_path / "Qwen3-0.6B-Q8_0.gguf"
    gguf_path.write_bytes(gguf_payload)
    manifest = build_gguf_model_package_manifest(
        catalog_id="cai-private",
        model_id=MODEL_ID,
        version="2026.05",
        gguf_path=gguf_path,
        total_layers=2,
        min_chunk_bytes=1,
        max_chunk_bytes=8,
        target_chunk_count=2,
        family="Qwen3",
        quantization="Q8_0",
    )
    save_model_package_manifest(manifest)
    for chunk in manifest.chunks:
        payload = gguf_payload[chunk.offset_bytes : chunk.offset_bytes + chunk.size_bytes]
        put_cached_chunk(
            catalog_id=manifest.catalog_id,
            version=manifest.version,
            chunk_id=chunk.chunk_id,
            sha256_hex=chunk.sha256_hex,
            content=payload,
        )
    bridge = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _handler_class(
            native_command=[
                sys.executable,
                "-m",
                "cai_compute_chain.cai_llama_cpp_assignment_artifact_engine",
            ],
            timeout_sec=10,
        ),
    )
    thread = threading.Thread(target=bridge.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint_url = f"http://127.0.0.1:{bridge.server_address[1]}/cai-shard"
        result = run_cai_owned_llm_shard_adapter_self_test(
            ExternalLlamaCppShardAdapter(
                endpoint_url=endpoint_url,
                timeout_sec=10,
                shard_artifact_hint={
                    "catalogId": manifest.catalog_id,
                    "version": manifest.version,
                    "artifactId": "gguf-main",
                },
            ),
            model_id=MODEL_ID,
            runtime_metadata={
                "modelId": MODEL_ID,
                "totalLayerCount": 2,
                "hiddenSize": 1024,
                "activationDtype": "f16",
                "tensorEncoding": "ggml-tensor-v1",
                "tokenizerConfigHash": "ef" * 32,
                "backend": "llama.cpp-patched",
                "backendVersion": "assignment-artifact-engine/0.1",
            },
            payload=b"The capital of France is",
        )
    finally:
        bridge.shutdown()
        bridge.server_close()
        thread.join(timeout=2)

    assert result["contractReady"] is True
    assert result["patchBoundaryVerified"] is True
    assert result["productionReady"] is False
    assert result["backendMode"] == ASSIGNMENT_ARTIFACT_ENGINE_ID
    assert result["generationProbeReady"] is False

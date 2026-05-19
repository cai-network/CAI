# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from cai_compute_chain.cai_llama_cpp_real_state_contract import (
    build_real_state_manifest_payload,
    validate_real_state_payload,
)


def _request(tmp_path: Path) -> dict:
    workspace_root = tmp_path / "workspace"
    state_dir = workspace_root / "state"
    outputs_dir = workspace_root / "outputs"
    inputs_dir = workspace_root / "inputs"
    for path in (state_dir, outputs_dir, inputs_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "action": "process_prefill",
        "sessionId": "session-real-state",
        "modelId": "cai-network/Qwen3-0.6B-GGUF",
        "layerStart": 0,
        "layerEnd": 14,
        "tokenStart": 0,
        "tokenEnd": 4,
        "executionWorkspace": {
            "schemaVersion": 1,
            "abi": "cai-llama-cpp-execution-workspace-v1",
            "root": str(workspace_root.resolve()),
            "inputsDir": str(inputs_dir.resolve()),
            "outputsDir": str(outputs_dir.resolve()),
            "stateFilesDir": str(state_dir.resolve()),
            "manifestPath": str((workspace_root / "execution-workspace.json").resolve()),
        },
    }


def test_validate_real_state_payload_accepts_workspace_manifest(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    state_path = Path(request["executionWorkspace"]["stateFilesDir"]) / "decode.bin"
    state_payload = b"real decode state bytes"
    state_path.write_bytes(state_payload)
    manifest_payload = build_real_state_manifest_payload(
        output_kind="decode_state",
        action="process_prefill",
        model_id=request["modelId"],
        session_id=request["sessionId"],
        layer_start=request["layerStart"],
        layer_end=request["layerEnd"],
        token_start=request["tokenStart"],
        token_end=request["tokenEnd"],
        state_file_path=state_path,
    )

    manifest = validate_real_state_payload(
        manifest_payload,
        request=request,
        output_kind="decode_state",
    )

    assert manifest.state_kind == "decode_state"
    assert manifest.state_file.path == state_path.resolve()
    assert manifest.state_file.sha256_hex == hashlib.sha256(state_payload).hexdigest()
    assert manifest.state_file.size_bytes == len(state_payload)


def test_validate_real_state_payload_rejects_escape_from_workspace(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    outside_path = tmp_path / "outside.bin"
    outside_path.write_bytes(b"escape")
    manifest_payload = build_real_state_manifest_payload(
        output_kind="decode_state",
        action="process_prefill",
        model_id=request["modelId"],
        session_id=request["sessionId"],
        layer_start=request["layerStart"],
        layer_end=request["layerEnd"],
        token_start=request["tokenStart"],
        token_end=request["tokenEnd"],
        state_file_path=outside_path,
    )

    with pytest.raises(
        ValueError,
        match="stateFile path must stay within executionWorkspace.stateFilesDir",
    ):
        validate_real_state_payload(
            manifest_payload,
            request=request,
            output_kind="decode_state",
        )

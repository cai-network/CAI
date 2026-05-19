# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .cai_llama_cpp_assignment_artifact_engine import (
    ASSIGNMENT_EXECUTOR_REQUEST_ABI,
)
from .cai_llama_cpp_slot_state_engine import (
    SlotStateEngineConfig,
    handle_slot_state_engine_request,
)
from .cai_owned_runtime import (
    LLAMA_CPP_EXTERNAL_SHARD_HANDOFF_ABI,
    LLAMA_CPP_EXTERNAL_SHARD_LOCAL_FILE_IO_ABI,
)


ASSIGNMENT_SLOT_STATE_EXECUTOR_ID = "assignment_slot_state_executor"
ASSIGNMENT_SLOT_STATE_REFERENCE_REASON = (
    "slot_state is a smoke/reference backend, not production shard-only "
    "layer-range execution."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a CAI assignment executor that translates the assignment "
            "executor ABI into the real llama.cpp slot-state backend."
        ),
    )
    parser.add_argument("--jsonl", action="store_true")
    parser.add_argument("--server-url", default="")
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--slot-id", type=int, default=0)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--decode-tokens", type=int, default=8)
    args = parser.parse_args(argv)
    config = SlotStateEngineConfig(
        server_url=str(args.server_url or ""),
        state_dir=(
            Path(str(args.state_dir)).expanduser().resolve()
            if str(args.state_dir or "").strip()
            else None
        ),
        slot_id=max(0, int(args.slot_id or 0)),
        timeout_sec=max(0.1, float(args.timeout_sec or 120.0)),
        decode_tokens=max(1, int(args.decode_tokens or 8)),
    )
    if bool(args.jsonl):
        return _jsonl_loop(config)
    try:
        request = json.loads(sys.stdin.read() or "{}")
        response = handle_assignment_slot_state_executor_request(
            request,
            config=config,
        )
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), flush=True)
        return 1
    print(json.dumps(response, sort_keys=True), flush=True)
    return 0


def _jsonl_loop(config: SlotStateEngineConfig) -> int:
    for line in sys.stdin:
        try:
            request = json.loads(line or "{}")
            response = handle_assignment_slot_state_executor_request(
                request,
                config=config,
            )
        except Exception as exc:
            response = {"status": "error", "error": str(exc)}
        print(json.dumps(response, sort_keys=True), flush=True)
    return 0


def handle_assignment_slot_state_executor_request(
    request: Mapping[str, Any],
    *,
    config: SlotStateEngineConfig,
) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        raise ValueError("CAI assignment slot-state executor request must be an object.")
    if str(request.get("abi") or "").strip() != ASSIGNMENT_EXECUTOR_REQUEST_ABI:
        raise ValueError("CAI assignment slot-state executor ABI is unsupported.")
    action = str(request.get("action") or "").strip()
    if action not in {"load_shard", "process_prefill", "process_decode", "finalize"}:
        raise ValueError(
            f"CAI assignment slot-state executor action is unsupported: {action}"
        )
    if _request_requires_shard_only_loading(request):
        raise ValueError(
            "CAI assignment slot-state executor is reference-only and cannot "
            "satisfy production shard-only loading."
        )
    handoff_request = _build_slot_state_handoff_request(request)
    slot_state_response = handle_slot_state_engine_request(
        handoff_request,
        config=config,
    )
    slot_state_metrics = (
        dict(slot_state_response.get("metrics") or {})
        if isinstance(slot_state_response.get("metrics"), Mapping)
        else {}
    )
    if action in {"load_shard", "finalize"}:
        return {
            "status": str(slot_state_response.get("status") or "ok").strip() or "ok",
            "realModelExecution": True,
            "metrics": _slot_state_reference_metrics(slot_state_metrics),
        }
    output_kind = str(request.get("expectedOutputKind") or "").strip() or None
    response: dict[str, Any] = {
        "status": "ok",
        "outputKind": output_kind,
        "realModelExecution": True,
        "metrics": _slot_state_reference_metrics(slot_state_metrics),
    }
    output_file = slot_state_response.get("outputPayloadFile")
    if isinstance(output_file, Mapping):
        response["outputPayloadFile"] = dict(output_file)
    else:
        output_base64 = str(slot_state_response.get("outputPayloadBase64") or "").strip()
        if output_base64:
            response["outputPayloadBase64"] = output_base64
    output_hash = str(slot_state_response.get("outputPayloadSha256Hex") or "").strip()
    if output_hash:
        response["outputPayloadSha256Hex"] = output_hash
    return response


def _build_slot_state_handoff_request(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    action = str(request.get("action") or "").strip()
    input_payload_file = request.get("inputPayloadFile")
    has_payload = action in {"process_prefill", "process_decode"}
    if has_payload and not isinstance(input_payload_file, Mapping):
        raise ValueError(
            "CAI assignment slot-state executor inputPayloadFile is missing."
        )
    execution_workspace = request.get("executionWorkspace")
    if not isinstance(execution_workspace, Mapping):
        raise ValueError(
            "CAI assignment slot-state executor executionWorkspace is missing."
        )
    workspace_root = str(execution_workspace.get("root") or "").strip()
    response_output_path = str(request.get("expectedOutputPayloadPath") or "").strip()
    if not workspace_root or (has_payload and not response_output_path):
        raise ValueError(
            "CAI assignment slot-state executor workspace paths are incomplete."
        )
    payload_hash = (
        str(input_payload_file.get("sha256Hex") or "").strip()
        if isinstance(input_payload_file, Mapping)
        else ""
    )
    local_artifact_resolution = request.get("localArtifactResolution")
    handoff_request: dict[str, Any] = {
        "schemaVersion": 1,
        "abi": LLAMA_CPP_EXTERNAL_SHARD_HANDOFF_ABI,
        "action": action,
        "backend": "llama.cpp-patched",
        "frame": _mapping_copy(request.get("frame")),
        "shardSpec": _mapping_copy(request.get("shardSpec")),
        "payloadFile": dict(input_payload_file) if isinstance(input_payload_file, Mapping) else None,
        "payloadSha256Hex": payload_hash or None,
        "outputContract": _mapping_copy(request.get("outputContract")),
        "managedRuntime": _mapping_copy(request.get("managedRuntime")),
        "localArtifactResolution": (
            _mapping_copy(local_artifact_resolution)
            if isinstance(local_artifact_resolution, Mapping)
            else _local_artifact_resolution_from_assignment(request)
        ),
        "localFileContract": (
            {
                "schemaVersion": 1,
                "abi": LLAMA_CPP_EXTERNAL_SHARD_LOCAL_FILE_IO_ABI,
                "ioRoot": str(Path(workspace_root).expanduser().resolve()),
                "responseOutputPath": str(
                    Path(response_output_path).expanduser().resolve()
                ),
            }
            if has_payload
            else None
        ),
        "productionRequirements": _mapping_copy(request.get("productionRequirements")),
    }
    return {
        key: value
        for key, value in handoff_request.items()
        if value not in (None, {})
    }


def _local_artifact_resolution_from_assignment(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    assignment_artifact = request.get("assignmentArtifact")
    if not isinstance(assignment_artifact, Mapping):
        return {}
    return {"assignmentArtifact": dict(assignment_artifact)}


def _mapping_copy(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _slot_state_reference_metrics(slot_state_metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "executorBackendMode": ASSIGNMENT_SLOT_STATE_EXECUTOR_ID,
        "slotStateReferenceBackend": True,
        "referenceBackend": True,
        "productionReady": False,
        "productionReadyReason": ASSIGNMENT_SLOT_STATE_REFERENCE_REASON,
        "realLayerExecution": False,
        "shardOnlyLoadingReady": False,
        "slotStateMetrics": dict(slot_state_metrics),
    }


def _request_requires_shard_only_loading(request: Mapping[str, Any]) -> bool:
    if _truthy(request.get("requireShardOnlyLoading")):
        return True
    requirements = request.get("productionRequirements")
    if not isinstance(requirements, Mapping):
        return False
    return bool(
        _truthy(requirements.get("requiresShardOnlyLoading"))
        or _truthy(requirements.get("forbidFullModelFallback"))
    )


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    raise SystemExit(main())

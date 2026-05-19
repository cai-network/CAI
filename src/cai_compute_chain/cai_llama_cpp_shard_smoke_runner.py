# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .cai_owned_runtime import (
    LLAMA_CPP_EXTERNAL_SHARD_HANDOFF_ABI,
    LLAMA_CPP_EXTERNAL_SHARD_LOCAL_FILE_IO_ABI,
    LLAMA_CPP_EXTERNAL_SHARD_REQUIRED_CAPABILITIES,
    build_llama_cpp_external_shard_patch_boundary,
)
from .decentralized_compute import validate_cai_owned_transport_frame_metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run CAI LLM shard smoke runner.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Read one request per line and write one response per line.",
    )
    args = parser.parse_args(argv)
    if bool(args.jsonl):
        return _jsonl_loop()
    try:
        request = json.loads(sys.stdin.read() or "{}")
        response = handle_smoke_runner_request(request)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 1
    print(json.dumps(response, sort_keys=True))
    return 0


def _jsonl_loop() -> int:
    for line in sys.stdin:
        try:
            request = json.loads(line or "{}")
            response = handle_smoke_runner_request(request)
        except Exception as exc:
            response = {"status": "error", "error": str(exc)}
        print(json.dumps(response, sort_keys=True), flush=True)
    return 0


def handle_smoke_runner_request(request: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        raise ValueError("CAI shard smoke runner request must be an object.")
    if str(request.get("abi") or "").strip() != LLAMA_CPP_EXTERNAL_SHARD_HANDOFF_ABI:
        raise ValueError("CAI shard smoke runner request ABI is unsupported.")
    action = str(request.get("action") or "").strip()
    frame = request.get("frame")
    if not isinstance(frame, Mapping):
        raise ValueError("CAI shard smoke runner frame is missing.")

    if action == "load_shard":
        return {
            "status": "ready",
            "capabilities": list(LLAMA_CPP_EXTERNAL_SHARD_REQUIRED_CAPABILITIES),
            "patchBoundary": build_llama_cpp_external_shard_patch_boundary(
                backend=str(request.get("backend") or "llama.cpp-patched"),
                backend_version=str(request.get("backendVersion") or "").strip()
                or "llama.cpp/cai-shard-smoke",
                patch_id="cai-llama-cpp-shard-smoke-runner",
                runner_protocol_version="smoke-0.1",
                extra_metadata={"mode": "smoke_runner"},
            ),
            "metrics": {
                "backendLoaded": True,
                "backendMode": "smoke_runner",
            },
        }
    if action in {"process_prefill", "process_decode"}:
        return _process_frame(action, request, frame)
    if action == "finalize":
        return {
            "status": "ok",
            "metrics": {
                "backendFinalized": True,
                "backendMode": "smoke_runner",
            },
        }
    raise ValueError(f"CAI shard smoke runner action is unsupported: {action}")


def _process_frame(
    action: str,
    request: Mapping[str, Any],
    frame: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _decode_payload(request)
    declared_hash = str(request.get("payloadSha256Hex") or "").strip().lower()
    payload_hash = hashlib.sha256(payload).hexdigest()
    if declared_hash and declared_hash != payload_hash:
        raise ValueError("CAI shard smoke runner payload hash does not match.")

    metadata = frame.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("CAI shard smoke runner frame metadata is missing.")
    valid, error = validate_cai_owned_transport_frame_metadata(
        metadata,
        expected_model_id=str(frame.get("modelId") or "").strip() or None,
        require_llm_handoff=True,
    )
    if not valid:
        raise ValueError(error or "CAI shard smoke runner frame metadata is invalid.")

    prefix = _output_prefix(action)
    output = prefix + payload
    output_hash = hashlib.sha256(output).hexdigest()
    response: dict[str, Any] = {
        "status": "ok",
        "outputPayloadSha256Hex": output_hash,
        "metrics": {
            "backendMode": "smoke_runner",
            "backendAction": action,
            "handoffAbi": request.get("abi"),
            "frameBatchId": frame.get("batchId"),
            "frameLayerStart": frame.get("layerStart"),
            "frameLayerEnd": frame.get("layerEnd"),
            "inputTokenCount": _token_count(metadata),
            "outputTokenCount": 1 if action == "process_decode" else 0,
        },
    }
    response.update(_encode_output_payload(request, output, output_hash))
    output_metadata = _output_frame_metadata(request, metadata, output_hash)
    if output_metadata:
        response["outputFrameMetadata"] = output_metadata
    return response


def _decode_payload(request: Mapping[str, Any]) -> bytes:
    payload_file = request.get("payloadFile")
    if _env_flag("CAI_SHARD_SMOKE_REQUIRE_FILE_INPUT") and not isinstance(
        payload_file,
        Mapping,
    ):
        raise ValueError("CAI shard smoke runner expected payloadFile input.")
    if isinstance(payload_file, Mapping):
        return _read_payload_file(
            payload_file,
            request.get("localFileContract")
            if isinstance(request.get("localFileContract"), Mapping)
            else None,
        )
    raw = request.get("payloadBase64")
    if raw is None:
        return b""
    try:
        return base64.b64decode(str(raw or "").encode("ascii"), validate=True)
    except Exception as exc:
        raise ValueError("CAI shard smoke runner payload is invalid.") from exc


def _output_prefix(action: str) -> bytes:
    if action == "process_decode":
        raw = os.getenv("CAI_SHARD_SMOKE_DECODE_PREFIX", "decoded-answer:")
    else:
        raw = os.getenv("CAI_SHARD_SMOKE_PREFILL_PREFIX", "prefill-state:")
    return raw.encode("utf-8")


def _read_payload_file(
    payload_file: Mapping[str, Any],
    local_file_contract: Mapping[str, Any] | None,
) -> bytes:
    payload_path = _resolve_local_file_path(
        path_value=payload_file.get("path"),
        local_file_contract=local_file_contract,
        field_name="payload",
    )
    try:
        payload = payload_path.read_bytes()
    except Exception as exc:
        raise ValueError("CAI shard smoke runner payload file is unreadable.") from exc
    expected_size = payload_file.get("sizeBytes")
    if expected_size is not None:
        try:
            size_value = int(expected_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "CAI shard smoke runner payload file size is invalid."
            ) from exc
        if size_value != len(payload):
            raise ValueError("CAI shard smoke runner payload file size mismatch.")
    expected_hash = str(payload_file.get("sha256Hex") or "").strip().lower()
    if expected_hash and expected_hash != hashlib.sha256(payload).hexdigest():
        raise ValueError("CAI shard smoke runner payload file hash mismatch.")
    return payload


def _encode_output_payload(
    request: Mapping[str, Any],
    output: bytes,
    output_hash: str,
) -> dict[str, Any]:
    local_file_contract = (
        request.get("localFileContract")
        if isinstance(request.get("localFileContract"), Mapping)
        else None
    )
    if _env_flag("CAI_SHARD_SMOKE_PREFER_OUTPUT_FILE") and isinstance(
        local_file_contract,
        Mapping,
    ):
        output_path = _resolve_local_file_path(
            path_value=local_file_contract.get("responseOutputPath"),
            local_file_contract=local_file_contract,
            field_name="output",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(output)
        return {
            "outputPayloadFile": {
                "path": str(output_path),
                "sizeBytes": len(output),
                "sha256Hex": output_hash,
            }
        }
    return {"outputPayloadBase64": base64.b64encode(output).decode("ascii")}


def _resolve_local_file_path(
    *,
    path_value: object,
    local_file_contract: Mapping[str, Any] | None,
    field_name: str,
) -> Path:
    raw_path = str(path_value or "").strip()
    if not raw_path:
        raise ValueError(f"CAI shard smoke runner {field_name} file path is missing.")
    path = Path(raw_path)
    if not path.is_absolute():
        raise ValueError(
            f"CAI shard smoke runner {field_name} file path must be absolute."
        )
    if isinstance(local_file_contract, Mapping):
        abi = str(local_file_contract.get("abi") or "").strip()
        if abi and abi != LLAMA_CPP_EXTERNAL_SHARD_LOCAL_FILE_IO_ABI:
            raise ValueError("CAI shard smoke runner local file contract ABI is invalid.")
        io_root = str(local_file_contract.get("ioRoot") or "").strip()
        if io_root:
            root_path = Path(io_root).resolve()
            resolved_path = path.resolve()
            try:
                resolved_path.relative_to(root_path)
            except ValueError as exc:
                raise ValueError(
                    f"CAI shard smoke runner {field_name} file escaped IO root."
                ) from exc
    return path


def _env_flag(name: str) -> bool:
    return str(os.getenv(name) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _output_frame_metadata(
    request: Mapping[str, Any],
    metadata: Mapping[str, Any],
    output_hash: str,
) -> dict[str, Any] | None:
    contract = request.get("outputContract")
    template = (
        contract.get("frameMetadataTemplate")
        if isinstance(contract, Mapping)
        else None
    )
    if template is None:
        template = metadata.get("nextFrameTemplate")
    if not isinstance(template, Mapping):
        if isinstance(contract, Mapping) and bool(
            contract.get("requiresOutputFrameMetadata"),
        ):
            raise ValueError(
                "CAI shard smoke runner output contract requires frame metadata."
            )
        return None
    output_metadata = json.loads(json.dumps(dict(template), sort_keys=True))
    output_metadata["payloadSha256Hex"] = output_hash
    handoff = dict(output_metadata.get("llmHandoff") or {})
    tensor = dict(handoff.get("tensor") or {})
    tensor["sha256Hex"] = output_hash
    handoff["tensor"] = tensor
    output_metadata["llmHandoff"] = handoff
    return output_metadata


def _token_count(metadata: Mapping[str, Any]) -> int:
    try:
        start = int(metadata.get("tokenStart") or 0)
        end = int(metadata.get("tokenEnd") or start)
    except (TypeError, ValueError):
        return 0
    return max(0, end - start)


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from typing import Any

from .cai_owned_runtime import (
    CaiOwnedShardRuntimeConfig,
    cai_owned_shard_adapter_from_env,
    run_cai_owned_shard_runtime_once,
)
from .decentralized_compute import (
    build_cai_owned_llm_shard_frame_metadata_from_runtime,
    build_cai_owned_transport_batch_envelope,
    build_cai_owned_transport_session_offer,
    cai_owned_transport_batch_payload_bytes,
    complete_cai_owned_transport_session,
    create_cai_owned_transport_session_from_offer,
    record_cai_owned_transport_batch_envelope,
)
from .model import WalletPolicy


PREFILL_SLOT_SERVER_URL_ENV = "CAI_LLM_SHARD_SLOT_PREFILL_SERVER_URL"
PREFILL_SLOT_STATE_DIR_ENV = "CAI_LLM_SHARD_SLOT_PREFILL_STATE_DIR"
PREFILL_SLOT_ID_ENV = "CAI_LLM_SHARD_SLOT_PREFILL_ID"
DECODE_SLOT_SERVER_URL_ENV = "CAI_LLM_SHARD_SLOT_DECODE_SERVER_URL"
DECODE_SLOT_STATE_DIR_ENV = "CAI_LLM_SHARD_SLOT_DECODE_STATE_DIR"
DECODE_SLOT_ID_ENV = "CAI_LLM_SHARD_SLOT_DECODE_ID"

MODEL_ID_DEFAULT = "cai-network/Qwen3-0.6B-GGUF"
REQUESTER_NODE_ID_DEFAULT = "node-user"
PREFILL_NODE_ID_DEFAULT = "node-prefill"
DECODE_NODE_ID_DEFAULT = "node-decode"


@dataclass(frozen=True)
class SlotStateEndpointConfig:
    server_url: str
    state_dir: str
    slot_id: int = 0


@dataclass(frozen=True)
class SlotStateHandoffSmokeConfig:
    prefill_endpoint: SlotStateEndpointConfig
    decode_endpoint: SlotStateEndpointConfig
    prompt: str = "The capital of France is"
    model_id: str = MODEL_ID_DEFAULT
    requester_node_id: str = REQUESTER_NODE_ID_DEFAULT
    prefill_node_id: str = PREFILL_NODE_ID_DEFAULT
    decode_node_id: str = DECODE_NODE_ID_DEFAULT
    instance_id: str = "instance-slot-state-handoff-smoke"
    task_id: str = "task-slot-state-handoff-smoke"
    wallet_data_dirname: str = ".tmp-cai-slot-state-handoff-smoke"
    total_layer_count: int = 28
    hidden_size: int = 1024
    token_count_hint: int = 4
    decode_tokens: int = 4
    timeout_sec: float = 120.0
    json_report_path: str | None = None
    use_http_forwarding: bool = False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a CAI-owned two-worker llama.cpp slot-state handoff smoke. "
            "This validates real slot/KV payload transfer through CAI envelopes; "
            "it is not a production layer-shard backend."
        ),
    )
    parser.add_argument("--worker-once", action="store_true")
    parser.add_argument("--node-id", default="")
    parser.add_argument("--runtime-id", default="")
    parser.add_argument("--output-peer", action="append", default=[])
    parser.add_argument("--wallet-data-dirname", default="")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--model-id", default=MODEL_ID_DEFAULT)
    parser.add_argument("--requester-node-id", default=REQUESTER_NODE_ID_DEFAULT)
    parser.add_argument("--prefill-node-id", default=PREFILL_NODE_ID_DEFAULT)
    parser.add_argument("--decode-node-id", default=DECODE_NODE_ID_DEFAULT)
    parser.add_argument(
        "--prefill-server-url",
        default=os.getenv(PREFILL_SLOT_SERVER_URL_ENV, ""),
    )
    parser.add_argument(
        "--prefill-state-dir",
        default=os.getenv(PREFILL_SLOT_STATE_DIR_ENV, ""),
    )
    parser.add_argument(
        "--prefill-slot-id",
        type=int,
        default=int(os.getenv(PREFILL_SLOT_ID_ENV, "0") or 0),
    )
    parser.add_argument(
        "--decode-server-url",
        default=os.getenv(DECODE_SLOT_SERVER_URL_ENV, ""),
    )
    parser.add_argument(
        "--decode-state-dir",
        default=os.getenv(DECODE_SLOT_STATE_DIR_ENV, ""),
    )
    parser.add_argument(
        "--decode-slot-id",
        type=int,
        default=int(os.getenv(DECODE_SLOT_ID_ENV, "0") or 0),
    )
    parser.add_argument("--total-layer-count", type=int, default=28)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--token-count-hint", type=int, default=4)
    parser.add_argument("--decode-tokens", type=int, default=4)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument(
        "--use-http-forwarding",
        action="store_true",
        help=(
            "Send worker output envelopes through a local HTTP receiver instead "
            "of only returning them to the coordinator process."
        ),
    )
    parser.add_argument("--json-report", default="")
    args = parser.parse_args(argv)
    if args.worker_once:
        return _worker_once(args)

    config = SlotStateHandoffSmokeConfig(
        prefill_endpoint=SlotStateEndpointConfig(
            server_url=_required(args.prefill_server_url, "--prefill-server-url"),
            state_dir=_required(args.prefill_state_dir, "--prefill-state-dir"),
            slot_id=max(0, int(args.prefill_slot_id or 0)),
        ),
        decode_endpoint=SlotStateEndpointConfig(
            server_url=_required(args.decode_server_url, "--decode-server-url"),
            state_dir=_required(args.decode_state_dir, "--decode-state-dir"),
            slot_id=max(0, int(args.decode_slot_id or 0)),
        ),
        prompt=str(args.prompt or ""),
        model_id=str(args.model_id or MODEL_ID_DEFAULT).strip() or MODEL_ID_DEFAULT,
        requester_node_id=str(
            args.requester_node_id or REQUESTER_NODE_ID_DEFAULT,
        ).strip(),
        prefill_node_id=str(args.prefill_node_id or PREFILL_NODE_ID_DEFAULT).strip(),
        decode_node_id=str(args.decode_node_id or DECODE_NODE_ID_DEFAULT).strip(),
        total_layer_count=max(1, int(args.total_layer_count or 28)),
        hidden_size=max(1, int(args.hidden_size or 1024)),
        token_count_hint=max(1, int(args.token_count_hint or 4)),
        decode_tokens=max(1, int(args.decode_tokens or 4)),
        timeout_sec=max(0.1, float(args.timeout_sec or 120.0)),
        json_report_path=str(args.json_report or "").strip() or None,
        use_http_forwarding=bool(args.use_http_forwarding),
        wallet_data_dirname=str(
            args.wallet_data_dirname or ".tmp-cai-slot-state-handoff-smoke"
        ),
    )
    report = run_slot_state_handoff_smoke(config)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if report.get("status") == "ok" else 2


def run_slot_state_handoff_smoke(
    config: SlotStateHandoffSmokeConfig,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="cai-slot-state-handoff-") as tempdir:
        previous_repo_root = os.environ.get("CAI_REPO_ROOT")
        os.environ["CAI_REPO_ROOT"] = tempdir
        try:
            report = _run_slot_state_handoff_smoke_in_repo(config, tempdir)
        finally:
            if previous_repo_root is None:
                os.environ.pop("CAI_REPO_ROOT", None)
            else:
                os.environ["CAI_REPO_ROOT"] = previous_repo_root
    if config.json_report_path:
        Path(config.json_report_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return report


def _run_slot_state_handoff_smoke_in_repo(
    config: SlotStateHandoffSmokeConfig,
    tempdir: str,
) -> dict[str, Any]:
    policy = WalletPolicy(wallet_data_dirname=config.wallet_data_dirname)
    participants = [
        config.requester_node_id,
        config.prefill_node_id,
        config.decode_node_id,
    ]
    executors = [config.prefill_node_id, config.decode_node_id]
    offer = build_cai_owned_transport_session_offer(
        instance_id=config.instance_id,
        participant_node_ids=participants,
        executor_node_ids=executors,
        model_id=config.model_id,
        task_id=config.task_id,
        source_node_id=config.requester_node_id,
        route_policy={
            "runtime": "slot_state_handoff_smoke",
            "dataPlane": "cai_owned_slot_state_handoff",
            "slotStateHandoff": True,
            "productionLayerShard": False,
        },
    )
    create_cai_owned_transport_session_from_offer(
        offer,
        session_id=offer["sessionId"],
        local_node_id=config.requester_node_id,
        policy=policy,
    )
    prompt_payload = config.prompt.encode("utf-8")
    prefill_metadata = _prefill_metadata(config, prompt_payload)
    first_envelope = build_cai_owned_transport_batch_envelope(
        session_id=offer["sessionId"],
        phase="prefill_activation_batches",
        source_node_id=config.requester_node_id,
        sink_node_id=config.prefill_node_id,
        sequence=0,
        payload=prompt_payload,
        metadata=prefill_metadata,
    )
    record_cai_owned_transport_batch_envelope(
        offer["sessionId"],
        first_envelope,
        local_node_id=config.prefill_node_id,
        policy=policy,
    )

    base_env = _base_worker_env(tempdir)
    receiver_state: dict[str, Any] | None = None
    receiver: ThreadingHTTPServer | None = None
    receiver_url: str | None = None
    if config.use_http_forwarding:
        receiver_state = {
            "receivedEnvelopes": [],
            "receivedShardReceipts": [],
            "finalPayload": None,
            "finalEnvelope": None,
        }
        receiver = _start_envelope_receiver(
            policy=policy,
            final_sink_node_id=config.requester_node_id,
            state=receiver_state,
        )
        receiver_url = (
            f"http://{receiver.server_address[0]}:{receiver.server_address[1]}"
        )

    prefill_result = _run_worker_process(
        base_env,
        node_id=config.prefill_node_id,
        runtime_id="runtime-slot-state-prefill",
        wallet_data_dirname=config.wallet_data_dirname,
        endpoint=config.prefill_endpoint,
        decode_tokens=config.decode_tokens,
        timeout_sec=config.timeout_sec,
        output_peers=_http_output_peers(
            receiver_url,
            config.decode_node_id,
            config.requester_node_id,
        ),
    )
    decode_envelope = _forwarded_envelope(prefill_result, config.decode_node_id)
    if not config.use_http_forwarding:
        record_cai_owned_transport_batch_envelope(
            offer["sessionId"],
            decode_envelope,
            local_node_id=config.decode_node_id,
            policy=policy,
        )

    decode_result = _run_worker_process(
        base_env,
        node_id=config.decode_node_id,
        runtime_id="runtime-slot-state-decode",
        wallet_data_dirname=config.wallet_data_dirname,
        endpoint=config.decode_endpoint,
        decode_tokens=config.decode_tokens,
        timeout_sec=config.timeout_sec,
        output_peers=_http_output_peers(receiver_url, config.requester_node_id),
    )
    final_envelope = _forwarded_envelope(decode_result, config.requester_node_id)
    if not config.use_http_forwarding:
        record_cai_owned_transport_batch_envelope(
            offer["sessionId"],
            final_envelope,
            local_node_id=config.requester_node_id,
            policy=policy,
        )

    completed = complete_cai_owned_transport_session(offer["sessionId"], policy=policy)
    final_payload = (
        bytes(receiver_state.get("finalPayload") or b"")
        if isinstance(receiver_state, Mapping)
        and isinstance(receiver_state.get("finalPayload"), bytes)
        else cai_owned_transport_batch_payload_bytes(final_envelope)
    )
    proof = completed.proof or {}
    execution_audit = proof.get("executionAudit") if isinstance(proof, Mapping) else {}
    report = {
        "status": "ok",
        "repoRoot": tempdir,
        "sessionId": offer["sessionId"],
        "instanceId": config.instance_id,
        "requesterNodeId": config.requester_node_id,
        "executorNodeIds": executors,
        "modelId": config.model_id,
        "slotStateHandoff": True,
        "productionLayerShard": False,
        "httpForwarding": bool(config.use_http_forwarding),
        "receivedEnvelopeCount": (
            len(receiver_state["receivedEnvelopes"])
            if isinstance(receiver_state, Mapping)
            and isinstance(receiver_state.get("receivedEnvelopes"), list)
            else 0
        ),
        "receivedShardReceiptCount": (
            len(receiver_state["receivedShardReceipts"])
            if isinstance(receiver_state, Mapping)
            and isinstance(receiver_state.get("receivedShardReceipts"), list)
            else 0
        ),
        "finalPayloadUtf8": final_payload.decode("utf-8", errors="replace"),
        "finalPayloadSha256Hex": hashlib.sha256(final_payload).hexdigest(),
        "proofVerified": bool(
            isinstance(execution_audit, Mapping) and execution_audit.get("verified")
        ),
        "workerRuns": [
            _worker_summary(prefill_result),
            _worker_summary(decode_result),
        ],
        "shardReceiptNodeIds": [
            item.get("nodeId")
            for item in proof.get("shardReceipts", [])
            if isinstance(item, Mapping)
        ],
        "executionAudit": dict(execution_audit)
        if isinstance(execution_audit, Mapping)
        else {},
    }
    if not report["proofVerified"]:
        raise RuntimeError("CAI-owned slot-state proof was not verified.")
    if receiver is not None:
        receiver.shutdown()
        receiver.server_close()
    return report


def _worker_once(args: argparse.Namespace) -> int:
    if not str(args.node_id or "").strip():
        raise ValueError("--node-id is required for worker mode.")
    if not str(args.runtime_id or "").strip():
        raise ValueError("--runtime-id is required for worker mode.")
    env = dict(os.environ)
    env["CAI_LLM_SHARD_ADAPTER"] = "slot_state"
    os.environ["CAI_LLM_SHARD_ADAPTER"] = "slot_state"
    adapter = cai_owned_shard_adapter_from_env(env)
    result = run_cai_owned_shard_runtime_once(
        CaiOwnedShardRuntimeConfig(
            node_id=str(args.node_id or "").strip(),
            runtime_id=str(args.runtime_id or "").strip(),
            output_peer_cai_urls_by_node=_parse_output_peers(
                getattr(args, "output_peer", []),
            ),
            require_production_llm_handoff=True,
            policy=WalletPolicy(
                wallet_data_dirname=str(
                    args.wallet_data_dirname
                    or ".tmp-cai-slot-state-handoff-smoke"
                ),
            ),
        ),
        adapter,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result.get("status") == "processed" else 2


def _prefill_metadata(
    config: SlotStateHandoffSmokeConfig,
    prompt_payload: bytes,
) -> dict[str, Any]:
    runtime_metadata = _runtime_metadata(config)
    decode_template = build_cai_owned_llm_shard_frame_metadata_from_runtime(
        model_id=config.model_id,
        runtime_metadata=runtime_metadata,
        payload=b"",
        frame_kind="decode",
        layer_start=0,
        layer_end=config.total_layer_count,
        token_start=config.token_count_hint,
        token_end=config.token_count_hint + 1,
        sequence=1,
    )
    decode_template["stageId"] = "caistage_slot_state_decode"
    decode_template["outputRoutePlan"] = [
        {
            "sinkNodeId": config.requester_node_id,
            "phase": "decode_activation_batches",
            "sequence": 2,
            "stageId": "final_result",
            "finalOutput": True,
        }
    ]
    metadata = build_cai_owned_llm_shard_frame_metadata_from_runtime(
        model_id=config.model_id,
        runtime_metadata=runtime_metadata,
        payload=prompt_payload,
        frame_kind="activation",
        layer_start=0,
        layer_end=config.total_layer_count,
        token_start=0,
        token_end=config.token_count_hint,
        sequence=0,
    )
    metadata["stageId"] = "caistage_slot_state_prefill"
    metadata["nextSinkNodeId"] = config.decode_node_id
    metadata["nextOutputPhase"] = "decode_activation_batches"
    metadata["nextOutputSequence"] = 1
    metadata["nextFrameTemplate"] = decode_template
    metadata["outputRoutePlan"] = [
        {
            "sinkNodeId": config.decode_node_id,
            "phase": "decode_activation_batches",
            "sequence": 1,
            "stageId": "caistage_slot_state_decode",
            "executorNodeId": config.decode_node_id,
            "layerStart": 0,
            "layerEnd": config.total_layer_count,
            "frameTemplate": decode_template,
        },
        {
            "sinkNodeId": config.requester_node_id,
            "phase": "decode_activation_batches",
            "sequence": 2,
            "stageId": "final_result",
            "finalOutput": True,
        },
    ]
    return metadata


def _runtime_metadata(config: SlotStateHandoffSmokeConfig) -> dict[str, Any]:
    return {
        "modelId": config.model_id,
        "totalLayerCount": config.total_layer_count,
        "hiddenSize": config.hidden_size,
        "activationDtype": "f16",
        "tensorEncoding": "ggml-tensor-v1",
        "tokenizerConfigHash": "ef" * 32,
        "backend": "llama.cpp-patched",
        "backendVersion": "llama.cpp-slot-state/0.1",
        "metadataSource": "slot_state_handoff_smoke",
    }


def _base_worker_env(tempdir: str) -> dict[str, str]:
    env = dict(os.environ)
    env["CAI_REPO_ROOT"] = tempdir
    env["PYTHONPATH"] = _pythonpath_with_src_root(env)
    return env


def _run_worker_process(
    env: Mapping[str, str],
    *,
    node_id: str,
    runtime_id: str,
    wallet_data_dirname: str,
    endpoint: SlotStateEndpointConfig,
    decode_tokens: int,
    timeout_sec: float,
    output_peers: Mapping[str, list[str]] | None = None,
) -> dict[str, Any]:
    worker_env = dict(env)
    worker_env.update(
        {
            "CAI_LLM_SHARD_ADAPTER": "slot_state",
            "CAI_LLM_SHARD_SLOT_SERVER_URL": endpoint.server_url,
            "CAI_LLM_SHARD_SLOT_STATE_DIR": endpoint.state_dir,
            "CAI_LLM_SHARD_SLOT_ID": str(max(0, int(endpoint.slot_id or 0))),
            "CAI_LLM_SHARD_SLOT_TIMEOUT_SEC": str(max(0.1, float(timeout_sec))),
            "CAI_LLM_SHARD_SLOT_DECODE_TOKENS": str(max(1, int(decode_tokens))),
            "CAI_LLM_SHARD_ADAPTER_TIMEOUT_SEC": str(max(0.1, float(timeout_sec))),
        }
    )
    command = [
        sys.executable,
        "-m",
        "cai_compute_chain.cai_slot_state_handoff_smoke",
        "--worker-once",
        "--node-id",
        node_id,
        "--runtime-id",
        runtime_id,
        "--wallet-data-dirname",
        wallet_data_dirname,
    ]
    for peer_node_id, peer_urls in (output_peers or {}).items():
        for peer_url in peer_urls:
            command.extend(["--output-peer", f"{peer_node_id}={peer_url}"])
    completed = subprocess.run(
        command,
        env=worker_env,
        text=True,
        capture_output=True,
        timeout=max(0.1, float(timeout_sec)) + 5.0,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Worker {node_id} failed: "
            + (completed.stderr or completed.stdout or "").strip()[:800]
        )
    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError(f"Worker {node_id} returned empty output.")
    return json.loads(stdout.splitlines()[-1])


def _start_envelope_receiver(
    *,
    policy: WalletPolicy,
    final_sink_node_id: str,
    state: dict[str, Any],
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            try:
                raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                payload = json.loads(raw_body.decode("utf-8") or "{}")
                if not isinstance(payload, dict):
                    raise ValueError("HTTP payload must be an object.")
                if self.path.rstrip("/").endswith("/shard-receipts"):
                    state.setdefault("receivedShardReceipts", []).append(payload)
                    self._send_json(200, {"status": "recorded"})
                    return
                envelope = payload
                if not isinstance(envelope, dict):
                    raise ValueError("Envelope payload must be an object.")
                session_id = str(envelope.get("sessionId") or "").strip()
                sink_node_id = str(envelope.get("sinkNodeId") or "").strip()
                record_cai_owned_transport_batch_envelope(
                    session_id,
                    envelope,
                    local_node_id=sink_node_id,
                    policy=policy,
                )
                state["receivedEnvelopes"].append(envelope)
                if sink_node_id == final_sink_node_id:
                    state["finalEnvelope"] = envelope
                    state["finalPayload"] = cai_owned_transport_batch_payload_bytes(
                        envelope,
                    )
                self._send_json(
                    200,
                    {
                        "status": "running",
                        "sessionId": session_id,
                        "sinkNodeId": sink_node_id,
                    },
                )
            except Exception as exc:
                self._send_json(400, {"status": "error", "error": str(exc)})

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _http_output_peers(
    receiver_url: str | None,
    *node_ids: str,
) -> dict[str, list[str]]:
    clean_url = str(receiver_url or "").strip()
    if not clean_url:
        return {}
    return {
        clean_node_id: [clean_url]
        for raw_node_id in node_ids
        if (clean_node_id := str(raw_node_id or "").strip())
    }


def _parse_output_peers(values: object) -> dict[str, list[str]]:
    peers: dict[str, list[str]] = {}
    if not isinstance(values, list):
        return peers
    for value in values:
        raw = str(value or "").strip()
        if not raw or "=" not in raw:
            continue
        node_id, url = raw.split("=", 1)
        clean_node_id = node_id.strip()
        clean_url = url.strip()
        if not clean_node_id or not clean_url:
            continue
        peers.setdefault(clean_node_id, []).append(clean_url)
    return peers


def _forwarded_envelope(
    worker_result: Mapping[str, Any],
    expected_sink_node_id: str,
) -> dict[str, Any]:
    output_forward = worker_result.get("outputForward")
    if not isinstance(output_forward, Mapping):
        raise RuntimeError("Worker did not expose output forward envelope.")
    envelope = output_forward.get("envelope")
    if not isinstance(envelope, dict):
        raise RuntimeError("Worker output forward envelope is missing.")
    sink = str(envelope.get("sinkNodeId") or "").strip()
    if sink != str(expected_sink_node_id or "").strip():
        raise RuntimeError(f"Worker forwarded to unexpected sink: {sink}")
    return envelope


def _worker_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    work_item = result.get("workItem") if isinstance(result, Mapping) else None
    batch = work_item.get("batch") if isinstance(work_item, Mapping) else None
    completion = result.get("completion") if isinstance(result, Mapping) else None
    receipt = completion.get("receipt") if isinstance(completion, Mapping) else None
    metrics = batch.get("metrics") if isinstance(batch, Mapping) else None
    output_forward = (
        result.get("outputForward") if isinstance(result, Mapping) else None
    )
    return {
        "status": result.get("status"),
        "nodeId": receipt.get("nodeId") if isinstance(receipt, Mapping) else None,
        "batchId": batch.get("batchId") if isinstance(batch, Mapping) else None,
        "phase": batch.get("phase") if isinstance(batch, Mapping) else None,
        "outputPayloadSizeBytes": result.get("outputPayloadSizeBytes"),
        "outputForwardStatus": (
            output_forward.get("status")
            if isinstance(output_forward, Mapping)
            else None
        ),
        "outputForwardSinkNodeId": (
            output_forward.get("sinkNodeId")
            if isinstance(output_forward, Mapping)
            else None
        ),
        "adapterMode": (
            metrics.get("backendMode") if isinstance(metrics, Mapping) else None
        ),
    }


def _pythonpath_with_src_root(env: Mapping[str, str]) -> str:
    src_root = str(Path(__file__).resolve().parents[1])
    existing = str(env.get("PYTHONPATH") or "")
    return src_root if not existing else src_root + os.pathsep + existing


def _required(value: object, flag_name: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError(f"{flag_name} is required.")
    return clean


if __name__ == "__main__":
    raise SystemExit(main())

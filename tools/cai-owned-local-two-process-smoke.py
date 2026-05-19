# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.cai_owned_runtime import (  # noqa: E402
    CaiOwnedShardRuntimeConfig,
    DeterministicBytesShardAdapter,
    cai_owned_shard_adapter_from_env,
    run_cai_owned_shard_runtime_once,
)
from cai_compute_chain.cai_llama_cpp_shard_http_smoke_bridge import (  # noqa: E402
    handle_http_smoke_bridge_request_body,
)
from cai_compute_chain.cai_llama_cpp_shard_native_bridge import (  # noqa: E402
    PersistentNativeEngineClient,
    handle_native_bridge_health,
    handle_native_bridge_request_body,
)
from cai_compute_chain.decentralized_compute import (  # noqa: E402
    build_cai_owned_llm_shard_frame_metadata_from_runtime,
    build_cai_owned_transport_batch_envelope,
    build_cai_owned_transport_execution_dag,
    build_cai_owned_transport_frame_metadata,
    build_cai_owned_transport_session_offer,
    cai_owned_transport_batch_payload_bytes,
    complete_cai_owned_transport_session,
    create_cai_owned_transport_session_from_offer,
    record_cai_owned_transport_batch_envelope,
)
from cai_compute_chain.model import WalletPolicy  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a local CAI-owned two-process shard runtime smoke test.",
    )
    parser.add_argument("--worker-once", action="store_true")
    parser.add_argument("--node-id")
    parser.add_argument("--runtime-id")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--output-peer", action="append", default=[])
    parser.add_argument(
        "--adapter",
        choices=(
            "deterministic",
            "smoke_runner",
            "http_smoke",
            "native_bridge_smoke",
            "native_bridge_persistent_smoke",
            "env",
        ),
        default="deterministic",
        help=(
            "Shard adapter for worker subprocesses. Use env with "
            "CAI_LLM_SHARD_ADAPTER_COMMAND or CAI_LLM_SHARD_ADAPTER_URL for a "
            "real backend."
        ),
    )
    parser.add_argument(
        "--production-llm-smoke",
        action="store_true",
        help="Build llmHandoff/nextFrameTemplate metadata for the local smoke.",
    )
    parser.add_argument(
        "--require-production-llm-handoff",
        action="store_true",
        help="Require production llmHandoff validation inside worker subprocesses.",
    )
    parser.add_argument("--wallet-data-dirname", default=".tmp-cai-two-process-smoke")
    parser.add_argument("--json-report", default="")
    args = parser.parse_args()
    if args.production_llm_smoke and args.adapter == "deterministic":
        args.adapter = "smoke_runner"

    if args.worker_once:
        return _worker_once(args)
    return _coordinator_smoke(args)


def _worker_once(args: argparse.Namespace) -> int:
    policy = WalletPolicy(wallet_data_dirname=args.wallet_data_dirname)
    env = dict(os.environ)
    if args.adapter == "smoke_runner":
        env["CAI_LLM_SHARD_ADAPTER"] = "smoke_runner"
        os.environ["CAI_LLM_SHARD_ADAPTER"] = "smoke_runner"
    if args.adapter in {
        "http_smoke",
        "native_bridge_smoke",
        "native_bridge_persistent_smoke",
    }:
        env["CAI_LLM_SHARD_ADAPTER"] = "external_llama_cpp"
        os.environ["CAI_LLM_SHARD_ADAPTER"] = "external_llama_cpp"
    env_adapter_choices = {
        "smoke_runner",
        "http_smoke",
        "native_bridge_smoke",
        "native_bridge_persistent_smoke",
        "env",
    }
    adapter = (
        cai_owned_shard_adapter_from_env(env)
        if args.adapter in env_adapter_choices
        else DeterministicBytesShardAdapter(prefix=str(args.prefix).encode("utf-8"))
    )
    result = run_cai_owned_shard_runtime_once(
        CaiOwnedShardRuntimeConfig(
            node_id=_required(args.node_id, "--node-id"),
            runtime_id=_required(args.runtime_id, "--runtime-id"),
            output_peer_cai_urls_by_node=_parse_output_peers(args.output_peer),
            require_production_llm_handoff=bool(
                args.require_production_llm_handoff
                or args.production_llm_smoke
            ),
            policy=policy,
        ),
        adapter,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status") == "processed" else 2


def _coordinator_smoke(args: argparse.Namespace) -> int:
    wallet_data_dirname = str(args.wallet_data_dirname or ".tmp-cai-two-process-smoke")
    with tempfile.TemporaryDirectory(prefix="cai-two-process-smoke-") as tempdir:
        env = dict(os.environ)
        env["CAI_REPO_ROOT"] = tempdir
        os.environ["CAI_REPO_ROOT"] = tempdir
        policy = WalletPolicy(wallet_data_dirname=wallet_data_dirname)
        receiver_state: dict[str, Any] = {
            "receivedEnvelopes": [],
            "finalPayload": None,
            "finalEnvelope": None,
        }
        receiver = _start_envelope_receiver(
            policy=policy,
            final_sink_node_id="node-user",
            state=receiver_state,
        )
        receiver_url = (
            f"http://{receiver.server_address[0]}:{receiver.server_address[1]}"
        )
        shard_http_server = None
        worker_adapter = args.adapter
        if args.adapter in {
            "http_smoke",
            "native_bridge_smoke",
            "native_bridge_persistent_smoke",
        }:
            if args.adapter == "native_bridge_persistent_smoke":
                shard_http_server = _start_llm_shard_native_bridge_smoke_server(
                    persistent=True,
                )
            elif args.adapter == "native_bridge_smoke":
                shard_http_server = _start_llm_shard_native_bridge_smoke_server()
            else:
                shard_http_server = _start_llm_shard_http_smoke_server()
            shard_http_url = (
                "http://"
                f"{shard_http_server.server_address[0]}:"
                f"{shard_http_server.server_address[1]}/cai-shard"
            )
            env["CAI_LLM_SHARD_ADAPTER"] = "external_llama_cpp"
            env["CAI_LLM_SHARD_ADAPTER_URL"] = shard_http_url
            worker_adapter = "env"
        offer = build_cai_owned_transport_session_offer(
            instance_id="instance-two-process-smoke",
            participant_node_ids=["node-user", "node-a", "node-b"],
            executor_node_ids=["node-a", "node-b"],
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-two-process-smoke",
            source_node_id="node-user",
        )
        dag = build_cai_owned_transport_execution_dag(
            session_id=offer["sessionId"],
            requester_node_id="node-user",
            executor_node_ids=["node-a", "node-b"],
            total_layer_count=28,
            model_id="cai-network/Qwen3-0.6B-GGUF",
            task_id="task-two-process-smoke",
        )
        offer["routePolicy"] = {
            "runtime": "local-two-process-smoke",
            "dataPlane": "cai_owned_transport_execution_dag",
            "executionDag": dag,
            "executionDagHashSha256Hex": dag.get("dagHashSha256Hex"),
        }
        create_cai_owned_transport_session_from_offer(
            offer,
            session_id=offer["sessionId"],
            local_node_id="node-user",
            policy=policy,
        )
        initial_payload = b"user-prompt"
        runtime_metadata = _runtime_metadata(args)
        initial_metadata = _frame_metadata(
            payload=initial_payload,
            frame_kind="activation",
            layer_start=0,
            layer_end=14,
            sequence=0,
            runtime_metadata=runtime_metadata,
        )
        stages = [stage for stage in dag.get("stages", []) if isinstance(stage, dict)]
        if stages:
            initial_metadata["stageId"] = stages[0].get("stageId")
        initial_metadata["nextSinkNodeId"] = "node-b"
        initial_metadata["nextOutputPhase"] = "prefill_activation_batches"
        initial_metadata["nextOutputSequence"] = 1
        initial_metadata["remainingSinkNodeIds"] = [
            "node-a",
            "node-b",
            "node-user",
        ]
        initial_metadata["outputRoutePlan"] = _output_route_plan_from_dag(
            stages,
            requester_node_id="node-user",
        )
        if runtime_metadata is not None:
            _attach_llm_route_frame_templates(
                initial_metadata["outputRoutePlan"],
                runtime_metadata=runtime_metadata,
                initial_token_count=2,
            )
            initial_metadata["nextFrameTemplate"] = dict(
                initial_metadata["outputRoutePlan"][0]["frameTemplate"]
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

        node_a_prefill = _run_worker_process(
            env,
            node_id="node-a",
            runtime_id="runtime-node-a-prefill",
            prefix="a:",
            adapter=worker_adapter,
            production_llm_smoke=bool(args.production_llm_smoke),
            wallet_data_dirname=wallet_data_dirname,
            output_peers={"node-b": [receiver_url], "node-user": [receiver_url]},
        )
        node_b_prefill = _run_worker_process(
            env,
            node_id="node-b",
            runtime_id="runtime-node-b-prefill",
            prefix="b:",
            adapter=worker_adapter,
            production_llm_smoke=bool(args.production_llm_smoke),
            wallet_data_dirname=wallet_data_dirname,
            output_peers={"node-a": [receiver_url], "node-user": [receiver_url]},
        )
        node_a_decode = _run_worker_process(
            env,
            node_id="node-a",
            runtime_id="runtime-node-a-decode",
            prefix="a:",
            adapter=worker_adapter,
            production_llm_smoke=bool(args.production_llm_smoke),
            wallet_data_dirname=wallet_data_dirname,
            output_peers={"node-b": [receiver_url], "node-user": [receiver_url]},
        )
        node_b_decode = _run_worker_process(
            env,
            node_id="node-b",
            runtime_id="runtime-node-b-decode",
            prefix="b:",
            adapter=worker_adapter,
            production_llm_smoke=bool(args.production_llm_smoke),
            wallet_data_dirname=wallet_data_dirname,
            output_peers={"node-a": [receiver_url], "node-user": [receiver_url]},
        )
        final_payload = receiver_state.get("finalPayload")
        if not isinstance(final_payload, bytes):
            raise RuntimeError("Requester did not receive final output payload.")
        completed = complete_cai_owned_transport_session(offer["sessionId"], policy=policy)
        report = {
            "status": "ok",
            "repoRoot": tempdir,
            "sessionId": offer["sessionId"],
            "workerRuns": [
                _worker_summary(node_a_prefill),
                _worker_summary(node_b_prefill),
                _worker_summary(node_a_decode),
                _worker_summary(node_b_decode),
            ],
            "finalPayloadUtf8": final_payload.decode("utf-8", errors="replace"),
            "receivedEnvelopeCount": len(receiver_state["receivedEnvelopes"]),
            "proofVerified": bool(
                (completed.proof or {}).get("executionAudit", {}).get("verified")
            ),
            "finalOutputBatchCount": int(
                (completed.proof or {})
                .get("executionAudit", {})
                .get("finalOutputBatchCount", 0)
            ),
            "executorNodeIds": (completed.proof or {}).get("executorNodeIds"),
            "productionLlmSmoke": bool(args.production_llm_smoke),
            "adapter": args.adapter,
            "shardReceiptNodeIds": [
                item.get("nodeId")
                for item in (completed.proof or {}).get("shardReceipts", [])
                if isinstance(item, dict)
            ],
        }
        expected_final_payload = _expected_final_payload(args)
        if (
            expected_final_payload is not None
            and final_payload != expected_final_payload
        ):
            raise RuntimeError(f"Unexpected final payload: {final_payload!r}")
        if not report["proofVerified"]:
            raise RuntimeError("CAI-owned proof was not verified.")
        if args.json_report:
            Path(args.json_report).write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        if shard_http_server is not None:
            shard_http_server.shutdown()
            persistent_engine = getattr(
                shard_http_server,
                "cai_persistent_engine",
                None,
            )
            if isinstance(persistent_engine, PersistentNativeEngineClient):
                persistent_engine.close()
            shard_http_server.server_close()
        receiver.shutdown()
        receiver.server_close()
    return 0


def _frame_metadata(
    *,
    payload: bytes,
    frame_kind: str,
    layer_start: int,
    layer_end: int,
    sequence: int,
    runtime_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if runtime_metadata is not None:
        return build_cai_owned_llm_shard_frame_metadata_from_runtime(
            model_id="cai-network/Qwen3-0.6B-GGUF",
            runtime_metadata=runtime_metadata,
            payload=payload,
            frame_kind=frame_kind,
            layer_start=layer_start,
            layer_end=layer_end,
            token_start=0,
            token_end=2,
            sequence=sequence,
        )
    return build_cai_owned_transport_frame_metadata(
        model_id="cai-network/Qwen3-0.6B-GGUF",
        frame_kind=frame_kind,
        tokenizer_config_hash="ef" * 32,
        layer_start=layer_start,
        layer_end=layer_end,
        token_start=0,
        token_end=2,
        dtype="bytes",
        shape=[len(payload)],
        sequence=sequence,
        payload_sha256_hex=hashlib.sha256(payload).hexdigest(),
    )


def _runtime_metadata(args: argparse.Namespace) -> dict[str, Any] | None:
    if not bool(args.production_llm_smoke):
        return None
    return {
        "modelId": "cai-network/Qwen3-0.6B-GGUF",
        "totalLayerCount": 28,
        "hiddenSize": 1024,
        "activationDtype": "f16",
        "tensorEncoding": "ggml-tensor-v1",
        "tokenizerConfigHash": "ef" * 32,
        "backend": "llama.cpp-patched",
        "backendVersion": "llama.cpp/cai-shard-0.1",
        "metadataSource": "cai-owned-local-two-process-smoke",
    }


def _attach_llm_route_frame_templates(
    route_plan: list[dict[str, Any]],
    *,
    runtime_metadata: dict[str, Any],
    initial_token_count: int,
) -> None:
    next_template: dict[str, Any] | None = None
    token_count = max(0, int(initial_token_count or 0))
    for item in reversed(route_plan):
        if bool(item.get("finalOutput")):
            next_template = None
            continue
        phase = str(item.get("phase") or "")
        template = build_cai_owned_llm_shard_frame_metadata_from_runtime(
            model_id="cai-network/Qwen3-0.6B-GGUF",
            runtime_metadata=runtime_metadata,
            payload=b"",
            frame_kind=(
                "decode" if phase == "decode_activation_batches" else "activation"
            ),
            layer_start=int(item.get("layerStart") or 0),
            layer_end=int(item.get("layerEnd") or 0),
            token_start=token_count if phase == "decode_activation_batches" else 0,
            token_end=(
                token_count + 1
                if phase == "decode_activation_batches"
                else token_count
            ),
            sequence=int(item.get("sequence") or 0),
        )
        if item.get("stageId") is not None:
            template["stageId"] = item.get("stageId")
        if next_template is not None:
            template["nextFrameTemplate"] = next_template
        item["frameTemplate"] = template
        next_template = template


def _output_route_plan_from_dag(
    stages: list[dict[str, Any]],
    *,
    requester_node_id: str,
) -> list[dict[str, Any]]:
    if not stages:
        return []
    plan = [
        {
            "sinkNodeId": str(stage.get("sinkNodeId") or "").strip(),
            "phase": str(stage.get("phase") or "").strip(),
            "sequence": int(stage.get("sequence") or 0),
            "stageId": stage.get("stageId"),
            "executorNodeId": stage.get("executorNodeId"),
            "layerStart": stage.get("layerStart"),
            "layerEnd": stage.get("layerEnd"),
        }
        for stage in stages[1:]
    ]
    final_stage = stages[-1]
    plan.append(
        {
            "sinkNodeId": requester_node_id,
            "phase": str(final_stage.get("phase") or "").strip(),
            "sequence": int(final_stage.get("sequence") or 0) + 1,
            "stageId": "final_result",
            "finalOutput": True,
        }
    )
    return [
        item
        for item in plan
        if str(item.get("sinkNodeId") or "").strip()
        and str(item.get("phase") or "").strip()
    ]


def _expected_final_payload(args: argparse.Namespace) -> bytes | None:
    if not bool(args.production_llm_smoke):
        return b"b:a:b:a:user-prompt"
    if args.adapter in {
        "smoke_runner",
        "http_smoke",
        "native_bridge_smoke",
        "native_bridge_persistent_smoke",
    }:
        return b"decoded-answer:decoded-answer:prefill-state:prefill-state:user-prompt"
    return None


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
                envelope = json.loads(raw_body.decode("utf-8") or "{}")
                if not isinstance(envelope, dict):
                    raise ValueError("Envelope payload must be an object.")
                session_id = str(envelope.get("sessionId") or "").strip()
                sink_node_id = str(envelope.get("sinkNodeId") or "").strip()
                state["receivedEnvelopes"].append(envelope)
                record_cai_owned_transport_batch_envelope(
                    session_id,
                    envelope,
                    local_node_id=sink_node_id,
                    policy=policy,
                )
                if sink_node_id == final_sink_node_id:
                    state["finalEnvelope"] = envelope
                    state["finalPayload"] = cai_owned_transport_batch_payload_bytes(
                        envelope
                    )
                response = {
                    "status": "running",
                    "sessionId": session_id,
                    "sinkNodeId": sink_node_id,
                }
                self._send_json(200, response)
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


def _start_llm_shard_http_smoke_server() -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path not in {"/health", "/v1/health"}:
                self._send_json(
                    404,
                    {
                        "status": "error",
                        "error": "CAI LLM shard smoke bridge endpoint is unknown.",
                    },
                )
                return
            self._send_json(
                200,
                {
                    "schemaVersion": 1,
                    "status": "ok",
                    "bridge": "cai_llm_shard_http_smoke_bridge",
                    "nativeCommandConfigured": True,
                    "nativeEngineMode": "in_process_smoke",
                    "endpoint": "/cai-shard",
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            status_code, response = handle_http_smoke_bridge_request_body(raw_body)
            self._send_json(status_code, response)

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


def _start_llm_shard_native_bridge_smoke_server(
    *,
    persistent: bool = False,
) -> ThreadingHTTPServer:
    native_command = [
        sys.executable,
        "-m",
        "cai_compute_chain.cai_llama_cpp_shard_smoke_runner",
    ]
    if persistent:
        native_command.append("--jsonl")
    native_env = {"PYTHONPATH": _pythonpath_with_src_root()}
    persistent_engine = (
        PersistentNativeEngineClient(
            native_command,
            timeout_sec=10,
            env=native_env,
        )
        if persistent
        else None
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path not in {"/health", "/v1/health"}:
                self._send_json(
                    404,
                    {
                        "status": "error",
                        "error": "CAI LLM shard native bridge endpoint is unknown.",
                    },
                )
                return
            status_code, response = handle_native_bridge_health(
                native_command=native_command,
                timeout_sec=10,
                persistent_engine=persistent_engine,
            )
            self._send_json(status_code, response)

        def do_POST(self) -> None:  # noqa: N802
            raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            status_code, response = handle_native_bridge_request_body(
                raw_body,
                native_command=native_command,
                timeout_sec=10,
                env=native_env,
                persistent_engine=persistent_engine,
            )
            self._send_json(status_code, response)

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
    if persistent_engine is not None:
        setattr(server, "cai_persistent_engine", persistent_engine)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _pythonpath_with_src_root() -> str:
    existing = str(os.environ.get("PYTHONPATH") or "")
    return str(SRC_ROOT) if not existing else str(SRC_ROOT) + os.pathsep + existing


def _run_worker_process(
    env: dict[str, str],
    *,
    node_id: str,
    runtime_id: str,
    prefix: str,
    adapter: str,
    production_llm_smoke: bool,
    wallet_data_dirname: str,
    output_peers: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-once",
        "--node-id",
        node_id,
        "--runtime-id",
        runtime_id,
        "--prefix",
        prefix,
        "--adapter",
        adapter,
        "--wallet-data-dirname",
        wallet_data_dirname,
    ]
    if production_llm_smoke:
        command.append("--production-llm-smoke")
        command.append("--require-production-llm-handoff")
    for peer_node_id, peer_urls in (output_peers or {}).items():
        for peer_url in peer_urls:
            command.extend(["--output-peer", f"{peer_node_id}={peer_url}"])
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if int(completed.returncode or 0) != 0:
        parsed_output: dict[str, Any] | None = None
        stdout_lines = str(completed.stdout or "").strip().splitlines()
        if stdout_lines:
            try:
                parsed = json.loads(stdout_lines[-1])
                if isinstance(parsed, dict):
                    parsed_output = parsed
            except Exception:
                parsed_output = None
        stdout_tail = str(completed.stdout or "").strip()[-5000:]
        stderr_tail = str(completed.stderr or "").strip()[-5000:]
        detail = {
            "nodeId": node_id,
            "runtimeId": runtime_id,
            "returnCode": int(completed.returncode or 0),
            "stdoutTail": stdout_tail,
            "stderrTail": stderr_tail,
        }
        if parsed_output is not None:
            detail["resultStatus"] = parsed_output.get("status")
            detail["resultError"] = parsed_output.get("error")
            failure = parsed_output.get("failure")
            if isinstance(failure, dict):
                detail["failureError"] = failure.get("error")
                detail["failureRetryScheduled"] = failure.get("retryScheduled")
            batch = parsed_output.get("batch")
            if isinstance(batch, dict):
                detail["batchStatus"] = batch.get("status")
                detail["batchLastError"] = batch.get("lastError")
        raise RuntimeError(
            "Worker subprocess failed: "
            + json.dumps(detail, ensure_ascii=False, sort_keys=True)
        )
    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError(f"Worker {node_id} returned empty output.")
    return json.loads(stdout.splitlines()[-1])


def _parse_output_peers(items: list[str]) -> dict[str, list[str]]:
    peers: dict[str, list[str]] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid --output-peer value: {item}")
        node_id, url = item.split("=", 1)
        clean_node_id = node_id.strip()
        clean_url = url.strip()
        if not clean_node_id or not clean_url:
            raise ValueError(f"Invalid --output-peer value: {item}")
        peers.setdefault(clean_node_id, []).append(clean_url)
    return peers


def _worker_summary(result: dict[str, Any]) -> dict[str, Any]:
    completion = result.get("completion") if isinstance(result, dict) else None
    receipt = completion.get("receipt") if isinstance(completion, dict) else None
    return {
        "status": result.get("status") if isinstance(result, dict) else None,
        "batchId": completion.get("batchId") if isinstance(completion, dict) else None,
        "nodeId": receipt.get("nodeId") if isinstance(receipt, dict) else None,
        "activationBatchCount": (
            receipt.get("activationBatchCount") if isinstance(receipt, dict) else None
        ),
        "decodeBatchCount": (
            receipt.get("decodeBatchCount") if isinstance(receipt, dict) else None
        ),
        "outputPayloadSizeBytes": result.get("outputPayloadSizeBytes")
        if isinstance(result, dict)
        else None,
    }


def _required(value: str | None, name: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError(f"{name} is required.")
    return clean


if __name__ == "__main__":
    raise SystemExit(main())

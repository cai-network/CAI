# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.decentralized_compute import (  # noqa: E402
    dispatch_cai_owned_transport_execution_dag,
)
from cai_compute_chain.cai_owned_runtime import (  # noqa: E402
    save_cai_owned_transport_live_proof_result,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a CAI-owned transport HTTP smoke through live CAI API nodes. "
            "This uses the deterministic/pass-through adapter path and does not "
            "load a GPU model."
        )
    )
    parser.add_argument("--requester-url", required=True)
    parser.add_argument("--requester-node-id", required=True)
    parser.add_argument(
        "--executor",
        action="append",
        required=True,
        help="Executor in node_id=http://host:port[,http://fallback:port] form.",
    )
    parser.add_argument(
        "--requester-peer-url",
        action="append",
        default=[],
        help=(
            "Extra requester CAI URL advertised to executors for result return, "
            "for example cai-overlay:http://relay:52415?targetNodeId=<requester>."
        ),
    )
    parser.add_argument("--model-id", default="cai-network/Qwen3-0.6B-GGUF")
    parser.add_argument("--payload", default="user-prompt")
    parser.add_argument("--total-layer-count", type=int, default=28)
    parser.add_argument(
        "--input-token-count",
        type=int,
        default=0,
        help="Declared prompt token count for frame metadata and token accounting.",
    )
    parser.add_argument("--tokenizer-config-hash", default="ab" * 32)
    parser.add_argument(
        "--production-llm-smoke",
        action="store_true",
        help="Dispatch production llmHandoff/nextFrameTemplate metadata.",
    )
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--tensor-dtype", default="f16")
    parser.add_argument("--tensor-encoding", default="ggml-tensor-v1")
    parser.add_argument("--model-sha256-hex", default="")
    parser.add_argument("--backend", default="llama.cpp-patched")
    parser.add_argument("--backend-version", default="llama.cpp/cai-shard-0.1")
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--poll-sec", type=float, default=0.5)
    parser.add_argument(
        "--offer-settle-sec",
        type=float,
        default=1.5,
        help="Seconds to wait after session offers before dispatching the first batch.",
    )
    parser.add_argument("--json-report", default="")
    parser.add_argument(
        "--save-runtime-ready-proof",
        action="store_true",
        help=(
            "When the smoke succeeds, save a local runtime-ready proof cache used "
            "by CAI-owned transport readiness advertisement."
        ),
    )
    args = parser.parse_args()

    requester_url = _clean_url(args.requester_url)
    requester_node_id = _required(args.requester_node_id, "--requester-node-id")
    executor_urls = _parse_executor_urls(args.executor)
    executor_node_ids = list(executor_urls)
    if not executor_node_ids:
        raise ValueError("At least one --executor is required.")

    now = int(time.time())
    instance_id = f"cai-owned-http-smoke-{now}"
    task_id = f"task-cai-owned-http-smoke-{now}"
    peer_urls_by_node = _peer_urls_by_node(
        requester_node_id=requester_node_id,
        requester_url=requester_url,
        requester_peer_urls=args.requester_peer_url,
        executor_urls=executor_urls,
    )
    initial_payload = str(args.payload).encode("utf-8")
    dispatch = dispatch_cai_owned_transport_execution_dag(
        instance_id=instance_id,
        requester_node_id=requester_node_id,
        executor_node_ids=executor_node_ids,
        peer_cai_urls_by_node=peer_urls_by_node,
        initial_payload=initial_payload,
        total_layer_count=args.total_layer_count,
        model_id=args.model_id,
        task_id=task_id,
        tokenizer_config_hash=args.tokenizer_config_hash,
        llm_runtime_metadata=_runtime_metadata(args),
        initial_token_count=args.input_token_count,
        timeout_sec=args.timeout_sec,
        submit_requester_offer=True,
        offer_settle_sec=args.offer_settle_sec,
    )

    final_result = _post_json(
        (
            f"{requester_url}/v1/cai/transport/sessions/"
            f"{quote(dispatch['sessionId'], safe='')}/await-final-result"
        ),
        {
            "requesterNodeId": requester_node_id,
            "timeoutSec": args.timeout_sec,
            "pollIntervalSec": args.poll_sec,
        },
        timeout_sec=args.timeout_sec + 5.0,
    )
    final_output = final_result.get("finalOutput")
    report = {
        "status": "ok" if final_result.get("proofVerified") else "failed",
        "sessionId": dispatch["sessionId"],
        "instanceId": instance_id,
        "requesterNodeId": requester_node_id,
        "executorNodeIds": executor_node_ids,
        "productionLlmSmoke": bool(args.production_llm_smoke),
        "offerResponses": dispatch.get("offerResponses"),
        "initialDispatchResponse": dispatch.get("initialDispatchResponse"),
        "dag": dispatch.get("dag"),
        "finalResult": final_result,
        "finalPayloadUtf8": _final_payload_text(final_output),
    }
    if args.save_runtime_ready_proof and report["status"] == "ok":
        report["savedRuntimeReadyProof"] = save_cai_owned_transport_live_proof_result(
            report
        )
    if args.json_report:
        Path(args.json_report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "ok" else 2


def _parse_executor_urls(items: list[str]) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --executor value: {item}")
        node_id, urls_raw = item.split("=", 1)
        clean_node_id = _required(node_id, "--executor node_id")
        urls = [_clean_url(url) for url in urls_raw.split(",") if url.strip()]
        if not urls:
            raise ValueError(f"Invalid --executor URLs for {clean_node_id}")
        parsed[clean_node_id] = urls
    return parsed


def _peer_urls_by_node(
    *,
    requester_node_id: str,
    requester_url: str,
    requester_peer_urls: list[str],
    executor_urls: dict[str, list[str]],
) -> dict[str, list[str]]:
    peer_urls_by_node = {
        node_id: _dedupe_urls(list(urls))
        for node_id, urls in executor_urls.items()
    }
    existing_requester_urls = peer_urls_by_node.get(requester_node_id, [])
    peer_urls_by_node[requester_node_id] = _dedupe_urls(
        [
            requester_url,
            *(_clean_url(url) for url in requester_peer_urls),
            *existing_requester_urls,
        ]
    )
    return peer_urls_by_node


def _dedupe_urls(urls: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        clean = str(url or "").strip().rstrip("/")
        if not clean or clean in seen:
            continue
        seen.add(clean)
        deduped.append(clean)
    return deduped


def _runtime_metadata(args: argparse.Namespace) -> dict[str, Any] | None:
    if not bool(args.production_llm_smoke):
        return None
    metadata: dict[str, Any] = {
        "modelId": args.model_id,
        "totalLayerCount": int(args.total_layer_count),
        "hiddenSize": int(args.hidden_size),
        "activationDtype": str(args.tensor_dtype or "").strip() or "f16",
        "tensorEncoding": str(args.tensor_encoding or "").strip() or "ggml-tensor-v1",
        "tokenizerConfigHash": str(args.tokenizer_config_hash or "").strip(),
        "backend": str(args.backend or "").strip() or "llama.cpp-patched",
        "backendVersion": str(args.backend_version or "").strip(),
        "metadataSource": "cai-owned-http-smoke",
    }
    model_hash = str(args.model_sha256_hex or "").strip()
    if model_hash:
        metadata["modelSha256Hex"] = model_hash
    return metadata


def _output_route_plan_from_dag(
    stages: list[dict[str, Any]],
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


def _post_json(url: str, payload: dict[str, Any], *, timeout_sec: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_sec) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw or "{}")
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _final_payload_text(final_output: object) -> str | None:
    if not isinstance(final_output, dict):
        return None
    payload_base64 = final_output.get("payloadBase64") or final_output.get("payload")
    if not isinstance(payload_base64, str):
        return None
    try:
        import base64

        return base64.b64decode(payload_base64).decode("utf-8", errors="replace")
    except Exception:
        return None


def _clean_url(value: str) -> str:
    clean = str(value or "").strip().rstrip("/")
    if not clean:
        raise ValueError("URL is required.")
    if clean.startswith("cai-overlay:"):
        overlay_target = clean[len("cai-overlay:") :]
        if overlay_target.startswith(("http://", "https://")):
            return clean
    if not clean.startswith(("http://", "https://")):
        raise ValueError(f"URL must start with http:// or https://: {value}")
    return clean


def _required(value: str | None, label: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError(f"{label} is required.")
    return clean


if __name__ == "__main__":
    raise SystemExit(main())

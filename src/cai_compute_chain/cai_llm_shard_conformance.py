# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .cai_owned_runtime import (
    run_cai_owned_llm_shard_adapter_self_test,
    save_cai_owned_llm_shard_self_test_result,
)
from .model import NetworkModelPolicy, WalletPolicy


def run_cai_owned_llm_shard_conformance(
    *,
    adapter: object | None = None,
    model_id: str | None = None,
    payload: str | bytes | None = None,
    runtime_metadata: Mapping[str, Any] | None = None,
    require_production: bool = False,
    require_production_llm_handoff: bool = True,
    save_readiness: bool = False,
    wallet_data_dirname: str | None = None,
) -> dict[str, Any]:
    payload_bytes = (
        bytes(payload)
        if isinstance(payload, (bytes, bytearray))
        else str(payload or "cai-llm-shard-conformance").encode("utf-8")
    )
    self_test = run_cai_owned_llm_shard_adapter_self_test(
        adapter,
        model_id=model_id,
        runtime_metadata=runtime_metadata,
        payload=payload_bytes,
        require_production_llm_handoff=require_production_llm_handoff,
    )
    errors = _conformance_errors(
        self_test,
        require_production=bool(require_production),
    )
    report: dict[str, Any] = {
        "status": "passed" if not errors else "failed",
        "ok": not errors,
        "conformanceKind": "llm_shard_backend_conformance",
        "requireProduction": bool(require_production),
        "checks": _conformance_checks(self_test),
        "errors": errors,
        "selfTest": dict(self_test),
    }
    if save_readiness:
        policy = (
            WalletPolicy(wallet_data_dirname=wallet_data_dirname)
            if wallet_data_dirname
            else WalletPolicy()
        )
        report["savedReadiness"] = save_cai_owned_llm_shard_self_test_result(
            self_test,
            policy=policy,
        )
    return report


def conformance_report_to_json(report: Mapping[str, Any]) -> str:
    return json.dumps(dict(report), ensure_ascii=False, indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run CAI LLM shard backend conformance checks.",
    )
    parser.add_argument(
        "--model-id",
        default=NetworkModelPolicy().network_default_model_id,
        help="Model id for the synthetic llmHandoff frame.",
    )
    parser.add_argument(
        "--payload",
        default="cai-llm-shard-conformance",
        help="Small UTF-8 payload used for the conformance probe.",
    )
    parser.add_argument(
        "--runtime-metadata-json",
        help="Optional JSON object merged into the synthetic llmHandoff frame metadata.",
    )
    parser.add_argument(
        "--runtime-metadata-json-file",
        help="Optional path to a JSON object merged into the synthetic llmHandoff frame metadata.",
    )
    parser.add_argument(
        "--require-production",
        action="store_true",
        help="Fail unless the backend is productionReady, not only contractReady.",
    )
    parser.add_argument(
        "--allow-non-production-handoff",
        action="store_true",
        help="Do not require strict production llmHandoff validation.",
    )
    parser.add_argument(
        "--save-readiness",
        action="store_true",
        help="Persist the underlying self-test result for node readiness.",
    )
    parser.add_argument(
        "--wallet-data-dirname",
        help="Wallet/runtime data directory name used when saving readiness.",
    )
    parser.add_argument(
        "--json-report",
        help="Optional path to write the conformance JSON report.",
    )
    args = parser.parse_args(argv)
    runtime_metadata = None
    if args.runtime_metadata_json_file:
        runtime_metadata = json.loads(
            Path(args.runtime_metadata_json_file).read_text(encoding="utf-8")
        )
        if not isinstance(runtime_metadata, Mapping):
            raise ValueError("--runtime-metadata-json-file must contain a JSON object.")
    elif args.runtime_metadata_json:
        runtime_metadata = json.loads(args.runtime_metadata_json)
        if not isinstance(runtime_metadata, Mapping):
            raise ValueError("--runtime-metadata-json must be a JSON object.")
    report = run_cai_owned_llm_shard_conformance(
        model_id=args.model_id,
        payload=args.payload,
        runtime_metadata=runtime_metadata,
        require_production=bool(args.require_production),
        require_production_llm_handoff=not bool(args.allow_non_production_handoff),
        save_readiness=bool(args.save_readiness),
        wallet_data_dirname=args.wallet_data_dirname,
    )
    rendered = conformance_report_to_json(report)
    if args.json_report:
        Path(args.json_report).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report.get("ok") else 2


def _conformance_errors(
    self_test: Mapping[str, Any],
    *,
    require_production: bool,
) -> list[str]:
    errors: list[str] = []
    if not bool(self_test.get("contractReady")):
        errors.append("LLM shard backend contract self-test did not pass.")
    if not bool(self_test.get("patchBoundaryVerified")):
        errors.append("LLM shard backend patch boundary is not verified.")
    if not bool(self_test.get("outputFrameMetadataReady")):
        errors.append("LLM shard backend did not return valid output frame metadata.")
    if not bool(self_test.get("finalDecodeOutputReady")):
        errors.append("LLM shard backend did not return final decode output.")
    if require_production and not bool(self_test.get("productionReady")):
        reason = str(self_test.get("productionReadinessError") or "").strip()
        errors.append(
            "LLM shard backend is not productionReady"
            + (f": {reason}" if reason else ".")
        )
    return errors


def _conformance_checks(self_test: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for field_name in (
        "status",
        "modelId",
        "adapterId",
        "adapterVersion",
        "backend",
        "backendVersion",
        "backendMode",
        "patchBoundaryAbi",
        "patchBoundaryPatchId",
        "patchBoundaryHash",
        "productionReadinessError",
    ):
        value = self_test.get(field_name)
        if value is not None and str(value).strip():
            checks[field_name] = str(value).strip()
    for field_name in (
        "contractReady",
        "productionReady",
        "patchBoundaryVerified",
        "outputFrameMetadataReady",
        "finalDecodeOutputReady",
        "generationProbeReady",
    ):
        checks[field_name] = bool(self_test.get(field_name))
    if "backendHealthReady" in self_test:
        checks["backendHealthReady"] = self_test.get("backendHealthReady")
    for field_name in (
        "prefillOutputPayloadSizeBytes",
        "decodeOutputPayloadSizeBytes",
        "outputPayloadSizeBytes",
    ):
        value = self_test.get(field_name)
        if value is not None:
            try:
                checks[field_name] = max(0, int(value))
            except (TypeError, ValueError):
                pass
    production_checks = self_test.get("productionReadinessChecks")
    if isinstance(production_checks, Mapping):
        checks["productionReadinessChecks"] = dict(production_checks)
    backend_health = self_test.get("backendHealth")
    if isinstance(backend_health, Mapping):
        checks["backendHealth"] = dict(backend_health)
    generation_probe = self_test.get("generationProbe")
    if isinstance(generation_probe, Mapping):
        checks["generationProbe"] = dict(generation_probe)
    return checks


if __name__ == "__main__":
    raise SystemExit(main())

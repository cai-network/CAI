# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent
from urllib.request import urlopen

from .economics import chain_backed_ledger_snapshot, plan_funding, resolve_compute_price
from .chain import (
    chain_balance_atomic,
    chain_summary,
    developer_fund_distribution_round_ids,
    developer_contribution_fund_chain_address,
    ensure_chain_genesis,
    record_developer_fund_distribution,
    wallet_balance_source,
    wallet_chain_balance_or_local_atomic,
)
from .developer_fund import (
    DEFAULT_PARTICIPANTS_PATH,
    DEFAULT_ROUNDS_DIR,
    DEFAULT_ROUND_FILE_NAME,
    canonical_json_hash,
    developer_fund_founder_confirmation_signing_payload,
    developer_fund_vote_signing_payload,
    developer_fund_recipients_for_chain,
    resolve_developer_fund_path,
    validate_developer_fund_files,
)
from .cai_runtime_launcher import (
    add_peer_to_book,
    import_peer_book_from_state_url,
    launch_cai_runtime,
    load_peer_book,
    peer_book_path,
    sync_peer_book_from_bootstrap,
)
from .cai_owned_runtime import (
    CaiOwnedShardRuntimeConfig,
    cai_owned_shard_adapter_from_env,
    load_cai_owned_llm_shard_self_test_result,
    run_cai_owned_llm_shard_adapter_self_test,
    run_cai_owned_shard_runtime_once,
    save_cai_owned_llm_shard_self_test_result,
)
from .cai_llm_shard_conformance import (
    conformance_report_to_json,
    run_cai_owned_llm_shard_conformance,
)
from .cai_owned_diagnostics import build_cai_owned_diagnostics_snapshot
from .model import (
    ChunkCachePolicy,
    ChunkStorageAccountingPolicy,
    CaiNetworkConfig,
    ModelPrivacyPrinciples,
    MoneyPolicy,
    NetworkPrinciples,
    NetworkModelPolicy,
    NodeRole,
    PaymentPreference,
    RoadmapStatus,
    VerificationMode,
    WalletPolicy,
)
from .jobs import (
    apply_local_validator_attestation,
    create_job_intent,
    execute_job_intent,
    list_execution_receipts,
    list_job_intents,
)
from .launch_checks import render_alpha_launch_report, run_alpha_launch_checks
from .model_distribution import (
    ChunkDownloadTaskStatus,
    ChunkFetchSourceKind,
    ChunkInventorySourceKind,
    ModelShardAssignment,
    build_assignment_chunk_plan_from_store,
    build_chunk_inventory_index,
    build_assignment_fetch_plan_from_store,
    build_chunk_download_tasks_from_fetch_plan,
    build_local_chunk_inventory_payload,
    build_hf_gguf_model_package_manifest,
    build_gguf_model_package_manifest,
    chunk_download_queue_snapshot,
    chunk_store_snapshot,
    discover_and_import_hf_model_package_manifest,
    evict_chunks_to_policy_target,
    ensure_assignment_ready_from_store,
    execute_chunk_download_queue,
    import_model_package_manifest_from_url,
    load_chunk_source_bindings,
    load_local_artifact_bindings,
    list_chunk_download_tasks,
    list_chunk_source_health_records,
    list_chunk_storage_accounting_records,
    import_chunk_inventory_payload,
    list_recent_shard_hints,
    list_model_package_manifests,
    list_imported_chunk_inventory_payloads,
    load_chunk_inventory_payload,
    load_model_package_manifest,
    prefetch_recent_shard_hints,
    queue_assignment_fetch_plan,
    record_chunk_storage_accounting_snapshot,
    save_chunk_source_binding,
    save_local_artifact_binding,
    save_model_package_manifest,
    save_local_chunk_inventory_payload,
    sync_chunk_inventory_from_cai_peers,
    sync_chunk_inventory_from_urls,
    update_chunk_download_task_status,
)
from .node_config import (
    bind_worker_reward_address,
    clear_validator_jail,
    complete_validator_unbond,
    load_or_create_node_config,
    set_relay_mode,
    set_validator_ha_mode,
    set_validator_static_ip_confirmation,
    set_validator_mode,
    set_worker_mode,
)
from .settlement import (
    list_attestations,
    list_validator_evidence_cases,
    list_settlements,
    list_validator_evidence,
    list_worker_payouts,
    reconcile_worker_payouts,
    record_funding_settlement,
    sync_validator_evidence_from_cai_peers,
)
from .validators import build_validator_committee_snapshot, list_validator_records
from .validators import export_validator_set_payload, sync_validator_set_from_cai_peers
from .ui_state import build_interface_snapshot
from .update_channel import apply_remote_update, check_for_updates
from .wallet import (
    JournalEntry,
    apply_wallet_transfer,
    append_journal_entry,
    atomic_to_coins,
    coins_to_atomic,
    create_seed_wallet,
    credit_wallet,
    data_root,
    ai_development_password_file_path,
    ai_development_seed_file_path,
    developer_treasury_password_file_path,
    developer_treasury_seed_file_path,
    ensure_local_ai_development_wallet,
    ensure_local_developer_treasury_wallet,
    get_active_wallet,
    list_wallets,
    load_or_create_ledger,
    load_unlocked_wallet_signing_material,
    load_session,
    list_journal_entries,
    lock_wallet,
    normalize_address,
    restore_wallet_from_seed,
    resolve_wallet,
    save_ledger,
    select_active_wallet,
    unlock_wallet,
    update_wallet,
    wallets_file_path,
)
from .wallet_signing import (
    SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65,
    SIGNING_SCHEME_ML_DSA_65,
    decode_bytes,
    mldsa65_available,
    sign_payload_b64,
    sign_payload_mldsa65_b64,
)


def _chunk_task_source_kind(task) -> str:
    source_kind = str(getattr(task, "selected_source_kind", "") or "").strip()
    sources = getattr(task, "sources", ()) or ()
    if not source_kind and sources:
        source_kind = str(getattr(sources[0], "kind", "") or "").strip()
    return source_kind or "none"


def _chunk_task_source_lines(prefix: str, tasks) -> list[str]:
    counts = Counter(_chunk_task_source_kind(task) for task in tasks)
    return [
        f"- {prefix}_source_peer_cache={counts.get(ChunkFetchSourceKind.PEER_CACHE, 0)}",
        f"- {prefix}_source_storage_seed={counts.get(ChunkFetchSourceKind.STORAGE_SEED, 0)}",
        f"- {prefix}_source_origin={counts.get(ChunkFetchSourceKind.ORIGIN, 0)}",
        f"- {prefix}_source_none={counts.get('none', 0)}",
    ]


def _supports_color() -> bool:
    return sys.stdout.isatty() and str(os.getenv("NO_COLOR") or "").strip() == ""


def _ansi(text: str, code: str) -> str:
    if not _supports_color():
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def _bool_status(value: bool) -> str:
    return _ansi("enabled", "92") if value else _ansi("disabled", "90")


def _validator_state_label(state: str) -> str:
    normalized = str(state or "").strip().lower()
    if normalized == "bonded":
        return _ansi(normalized, "92")
    if normalized == "unbonding":
        return _ansi(normalized, "93")
    if normalized == "jailed":
        return _ansi(normalized, "91")
    return _ansi(normalized or "unbonded", "90")


def _add_cai_url_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str = "http://127.0.0.1:52425",
    help_text: str,
) -> None:
    parser.add_argument(
        "--cai-url",
        dest="cai_url",
        default=default,
        help=help_text,
    )
    parser.add_argument("--CAI-url", dest="cai_url", help=argparse.SUPPRESS)


def _add_cai_executable_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cai-executable",
        dest="cai_executable",
        help="Path to the CAI runtime executable",
    )
    parser.add_argument("--CAI-executable", dest="cai_executable", help=argparse.SUPPRESS)


def _add_cai_home_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cai-home",
        dest="cai_home",
        help="Explicit CAI runtime home directory",
    )
    parser.add_argument("--CAI-home", dest="cai_home", help=argparse.SUPPRESS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cai-chain")
    subparsers = parser.add_subparsers(dest="command", required=True)

    role_parser = subparsers.add_parser("role", help="Show role summary")
    role_parser.add_argument("role", choices=[role.value for role in NodeRole])

    verify_parser = subparsers.add_parser(
        "verify-mode", help="Show verification mode summary"
    )
    verify_parser.add_argument(
        "mode", choices=[mode.value for mode in VerificationMode]
    )

    subparsers.add_parser("network", help="Show CAI runtime network bootstrap settings")
    subparsers.add_parser(
        "network-model", help="Show default network model and privacy policy"
    )
    subparsers.add_parser("topology", help="Show target network topology")
    subparsers.add_parser("money-policy", help="Show monetary policy summary")
    subparsers.add_parser("status", help="Show bootstrap roadmap status")
    subparsers.add_parser("ledger", help="Show local reserve/treasury ledger status")
    developer_fund_parser = subparsers.add_parser(
        "developer-fund",
        help="Validate and distribute developer contribution fund rounds",
    )
    developer_fund_subparsers = developer_fund_parser.add_subparsers(
        dest="developer_fund_command",
        required=True,
    )
    developer_fund_validate = developer_fund_subparsers.add_parser(
        "validate",
        help="Validate participants registry and a developer fund round file",
    )
    developer_fund_validate.add_argument("--repo-root", default=".")
    developer_fund_validate.add_argument("--participants")
    developer_fund_validate.add_argument("--round")
    developer_fund_distribute = developer_fund_subparsers.add_parser(
        "distribute",
        help="Record an approved developer fund round on the local chain",
    )
    developer_fund_distribute.add_argument("--repo-root", default=".")
    developer_fund_distribute.add_argument("--participants")
    developer_fund_distribute.add_argument("--round")
    developer_fund_distribute.add_argument("--validator-id")
    developer_fund_distribute.add_argument("--source-commit")
    developer_fund_distribute.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the planned payout without writing a chain block",
    )
    developer_fund_sign_vote = developer_fund_subparsers.add_parser(
        "sign-vote",
        help="Create a signed developer fund vote from the active unlocked wallet",
    )
    developer_fund_sign_vote.add_argument("--round-id", default="year-1")
    developer_fund_sign_vote.add_argument("--github", required=True)
    developer_fund_sign_vote.add_argument(
        "--choice",
        action="append",
        required=True,
        help="GitHub login to vote for; may be repeated",
    )
    developer_fund_confirm = developer_fund_subparsers.add_parser(
        "sign-founder-confirmation",
        help="Create a founder signature for an approved developer fund round",
    )
    developer_fund_confirm.add_argument("--repo-root", default=".")
    developer_fund_confirm.add_argument("--participants")
    developer_fund_confirm.add_argument("--round")
    developer_fund_confirm.add_argument("--confirmed-by", default="founder")
    subparsers.add_parser(
        "model-package-list",
        help="List local CAI model package manifests",
    )
    subparsers.add_parser(
        "chunk-store",
        help="Show local CAI chunk store summary",
    )
    subparsers.add_parser(
        "chunk-store-gc",
        help="Enforce local CAI chunk store cache policy and evict old chunks if needed",
    )
    chunk_inventory_local_parser = subparsers.add_parser(
        "chunk-inventory-local",
        help="Build and save a local CAI chunk inventory payload",
    )
    chunk_inventory_local_parser.add_argument("source_id")
    chunk_inventory_local_parser.add_argument(
        "--source-kind",
        choices=[kind.value for kind in ChunkInventorySourceKind],
        default=ChunkInventorySourceKind.LOCAL_CACHE.value,
    )
    chunk_inventory_import_parser = subparsers.add_parser(
        "chunk-inventory-import",
        help="Import a peer or storage-seed chunk inventory payload",
    )
    chunk_inventory_import_parser.add_argument("path")
    chunk_inventory_list_parser = subparsers.add_parser(
        "chunk-inventory-list",
        help="List imported CAI chunk inventory payloads",
    )
    chunk_inventory_list_parser.add_argument(
        "--source-kind",
        choices=[
            ChunkInventorySourceKind.PEER_CACHE.value,
            ChunkInventorySourceKind.STORAGE_SEED.value,
        ],
    )
    chunk_inventory_sync_parser = subparsers.add_parser(
        "chunk-inventory-sync",
        help="Import chunk inventory payloads from reachable CAI peers",
    )
    _add_cai_url_argument(
        chunk_inventory_sync_parser,
        help_text="Local CAI runtime API base URL used for peer discovery",
    )
    chunk_inventory_sync_parser.add_argument(
        "--inventory-url",
        action="append",
        default=[],
        help="Explicit CAI chunk inventory URL or base URL; may be repeated",
    )
    chunk_inventory_sync_parser.add_argument(
        "--source-kind",
        choices=[
            ChunkInventorySourceKind.PEER_CACHE.value,
            ChunkInventorySourceKind.STORAGE_SEED.value,
        ],
        default=ChunkInventorySourceKind.PEER_CACHE.value,
    )
    subparsers.add_parser(
        "chunk-download-queue",
        help="Show local CAI chunk download queue summary",
    )
    chunk_source_health_parser = subparsers.add_parser(
        "chunk-source-health",
        help="Show local CAI chunk source health and cooldown state",
    )
    chunk_source_health_parser.add_argument(
        "--source-kind",
        choices=[kind.value for kind in ChunkFetchSourceKind],
    )
    chunk_shard_hints_parser = subparsers.add_parser(
        "chunk-shard-hints",
        help="Show recent shard prefetch hints remembered for a worker node",
    )
    chunk_shard_hints_parser.add_argument("node_id")
    chunk_shard_hints_parser.add_argument(
        "--prefetch",
        action="store_true",
        help="Run a bounded prefetch pass from the remembered hints",
    )
    chunk_storage_accounting_parser = subparsers.add_parser(
        "chunk-storage-accounting",
        help="Show or record local chunk storage accounting for a worker node",
    )
    chunk_storage_accounting_parser.add_argument("node_id")
    chunk_storage_accounting_parser.add_argument(
        "--record",
        action="store_true",
        help="Record a new accounting snapshot for currently cached chunks",
    )
    chunk_download_run_parser = subparsers.add_parser(
        "chunk-download-run",
        help="Execute queued local chunk downloads using supported local sources",
    )
    chunk_download_run_parser.add_argument("--max-tasks", type=int)
    chunk_source_bind_parser = subparsers.add_parser(
        "chunk-source-bind",
        help="Bind a local CAI data root as a peer or storage-seed chunk source",
    )
    chunk_source_bind_parser.add_argument(
        "source_kind",
        choices=[
            ChunkFetchSourceKind.PEER_CACHE.value,
            ChunkFetchSourceKind.STORAGE_SEED.value,
        ],
    )
    chunk_source_bind_parser.add_argument("source_id")
    chunk_source_bind_parser.add_argument("data_root_path")
    chunk_source_bindings_parser = subparsers.add_parser(
        "chunk-source-bindings",
        help="List bound peer and storage-seed chunk source roots",
    )
    chunk_source_bindings_parser.add_argument(
        "--source-kind",
        choices=[
            ChunkFetchSourceKind.PEER_CACHE.value,
            ChunkFetchSourceKind.STORAGE_SEED.value,
        ],
    )
    model_package_enqueue_fetch_parser = subparsers.add_parser(
        "model-package-enqueue-fetch",
        help="Convert a model-package fetch plan into queued local chunk download tasks",
    )
    model_package_enqueue_fetch_parser.add_argument("catalog_id")
    model_package_enqueue_fetch_parser.add_argument("version")
    model_package_enqueue_fetch_parser.add_argument("--start-layer", type=int, required=True)
    model_package_enqueue_fetch_parser.add_argument("--end-layer", type=int, required=True)
    model_package_enqueue_fetch_parser.add_argument("--device-rank", type=int, default=0)
    model_package_enqueue_fetch_parser.add_argument("--world-size", type=int, default=1)
    model_package_enqueue_fetch_parser.add_argument("--node-id")
    model_package_enqueue_fetch_parser.add_argument("--peer-inventory-json")
    model_package_enqueue_fetch_parser.add_argument("--seed-inventory-json")
    model_package_enqueue_fetch_parser.add_argument(
        "--use-imported-peer-inventory",
        action="store_true",
    )
    model_package_enqueue_fetch_parser.add_argument(
        "--use-imported-seed-inventory",
        action="store_true",
    )
    model_package_ensure_ready_parser = subparsers.add_parser(
        "model-package-ensure-ready",
        help="Ensure shard chunk coverage is ready by planning, queuing, and executing chunk fetches",
    )
    model_package_ensure_ready_parser.add_argument("catalog_id")
    model_package_ensure_ready_parser.add_argument("version")
    model_package_ensure_ready_parser.add_argument("--start-layer", type=int, required=True)
    model_package_ensure_ready_parser.add_argument("--end-layer", type=int, required=True)
    model_package_ensure_ready_parser.add_argument("--device-rank", type=int, default=0)
    model_package_ensure_ready_parser.add_argument("--world-size", type=int, default=1)
    model_package_ensure_ready_parser.add_argument("--node-id")
    model_package_ensure_ready_parser.add_argument("--peer-inventory-json")
    model_package_ensure_ready_parser.add_argument("--seed-inventory-json")
    model_package_ensure_ready_parser.add_argument(
        "--use-imported-peer-inventory",
        action="store_true",
    )
    model_package_ensure_ready_parser.add_argument(
        "--use-imported-seed-inventory",
        action="store_true",
    )
    model_package_ensure_ready_parser.add_argument("--max-tasks", type=int)
    model_package_cache_all_parser = subparsers.add_parser(
        "model-package-cache-all",
        help="Fetch and cache every chunk required to serve a full model package as a public seed",
    )
    model_package_cache_all_parser.add_argument("catalog_id")
    model_package_cache_all_parser.add_argument("version")
    model_package_cache_all_parser.add_argument("--node-id")
    model_package_cache_all_parser.add_argument("--max-tasks", type=int)
    model_package_cache_all_parser.add_argument(
        "--use-imported-peer-inventory",
        action="store_true",
    )
    model_package_cache_all_parser.add_argument(
        "--use-imported-seed-inventory",
        action="store_true",
    )
    chunk_download_mark_parser = subparsers.add_parser(
        "chunk-download-mark",
        help="Update local chunk download task status",
    )
    chunk_download_mark_parser.add_argument("task_id")
    chunk_download_mark_parser.add_argument(
        "status",
        choices=[status.value for status in ChunkDownloadTaskStatus],
    )
    chunk_download_mark_parser.add_argument("--source-kind")
    chunk_download_mark_parser.add_argument("--source-id")
    chunk_download_mark_parser.add_argument("--last-error")
    model_package_bind_artifact_parser = subparsers.add_parser(
        "model-package-bind-artifact",
        help="Bind a local artifact file path to a CAI model package artifact id",
    )
    model_package_bind_artifact_parser.add_argument("catalog_id")
    model_package_bind_artifact_parser.add_argument("version")
    model_package_bind_artifact_parser.add_argument("artifact_id")
    model_package_bind_artifact_parser.add_argument("path")
    subparsers.add_parser(
        "model-package-bindings",
        help="List local artifact bindings for a CAI model package",
    ).add_argument("catalog_id")
    # parser returned above, so attach version separately
    bindings_parser = subparsers.choices["model-package-bindings"]
    bindings_parser.add_argument("version")
    model_package_show_parser = subparsers.add_parser(
        "model-package-show",
        help="Show a local CAI model package manifest",
    )
    model_package_show_parser.add_argument("catalog_id")
    model_package_show_parser.add_argument("version")
    model_package_plan_parser = subparsers.add_parser(
        "model-package-plan",
        help="Show required chunk coverage for a shard assignment",
    )
    model_package_plan_parser.add_argument("catalog_id")
    model_package_plan_parser.add_argument("version")
    model_package_plan_parser.add_argument("--start-layer", type=int, required=True)
    model_package_plan_parser.add_argument("--end-layer", type=int, required=True)
    model_package_plan_parser.add_argument("--device-rank", type=int, default=0)
    model_package_plan_parser.add_argument("--world-size", type=int, default=1)
    model_package_plan_parser.add_argument("--node-id")
    model_package_fetch_plan_parser = subparsers.add_parser(
        "model-package-fetch-plan",
        help="Show missing chunk fetch sources for a shard assignment",
    )
    model_package_fetch_plan_parser.add_argument("catalog_id")
    model_package_fetch_plan_parser.add_argument("version")
    model_package_fetch_plan_parser.add_argument("--start-layer", type=int, required=True)
    model_package_fetch_plan_parser.add_argument("--end-layer", type=int, required=True)
    model_package_fetch_plan_parser.add_argument("--device-rank", type=int, default=0)
    model_package_fetch_plan_parser.add_argument("--world-size", type=int, default=1)
    model_package_fetch_plan_parser.add_argument("--node-id")
    model_package_fetch_plan_parser.add_argument(
        "--peer-inventory-json",
        help="JSON file mapping peer ids to chunk id lists",
    )
    model_package_fetch_plan_parser.add_argument(
        "--seed-inventory-json",
        help="JSON file mapping storage seed ids to chunk id lists",
    )
    model_package_fetch_plan_parser.add_argument(
        "--use-imported-peer-inventory",
        action="store_true",
        help="Load saved imported peer chunk inventories from the local CAI data directory",
    )
    model_package_fetch_plan_parser.add_argument(
        "--use-imported-seed-inventory",
        action="store_true",
        help="Load saved imported storage-seed chunk inventories from the local CAI data directory",
    )
    model_package_create_gguf_parser = subparsers.add_parser(
        "model-package-create-gguf",
        help="Build and save a CAI model package manifest from a local GGUF file",
    )
    model_package_create_gguf_parser.add_argument("catalog_id")
    model_package_create_gguf_parser.add_argument("model_id")
    model_package_create_gguf_parser.add_argument("version")
    model_package_create_gguf_parser.add_argument("gguf_path")
    model_package_create_gguf_parser.add_argument("--n-layers", type=int, required=True)
    model_package_create_gguf_parser.add_argument(
        "--package-kind",
        choices=["public_shared", "private_curated"],
        default="public_shared",
    )
    model_package_create_gguf_parser.add_argument(
        "--chunk-size-policy",
        choices=["small", "balanced", "large", "adaptive"],
        default="adaptive",
    )
    model_package_create_gguf_parser.add_argument("--min-chunk-mb", type=int, default=64)
    model_package_create_gguf_parser.add_argument("--max-chunk-mb", type=int, default=512)
    model_package_create_gguf_parser.add_argument("--target-chunks", type=int)
    model_package_create_gguf_parser.add_argument("--source-repo-id")
    model_package_create_gguf_parser.add_argument("--source-revision", default="main")
    model_package_create_gguf_parser.add_argument("--family", default="")
    model_package_create_gguf_parser.add_argument("--quantization", default="")
    model_package_create_hf_gguf_parser = subparsers.add_parser(
        "model-package-create-hf-gguf",
        help="Build and save a CAI model package manifest from a Hugging Face GGUF artifact using range requests",
    )
    model_package_create_hf_gguf_parser.add_argument("model_id")
    model_package_create_hf_gguf_parser.add_argument("version")
    model_package_create_hf_gguf_parser.add_argument("--catalog-id")
    model_package_create_hf_gguf_parser.add_argument("--preferred-filename")
    model_package_create_hf_gguf_parser.add_argument("--n-layers", type=int)
    model_package_create_hf_gguf_parser.add_argument(
        "--package-kind",
        choices=["public_shared", "private_curated"],
        default="public_shared",
    )
    model_package_create_hf_gguf_parser.add_argument(
        "--chunk-size-policy",
        choices=["small", "balanced", "large", "adaptive"],
        default="adaptive",
    )
    model_package_create_hf_gguf_parser.add_argument("--min-chunk-mb", type=int, default=64)
    model_package_create_hf_gguf_parser.add_argument("--max-chunk-mb", type=int, default=512)
    model_package_create_hf_gguf_parser.add_argument("--target-chunks", type=int)
    model_package_create_hf_gguf_parser.add_argument("--source-revision", default="main")
    model_package_create_hf_gguf_parser.add_argument("--family", default="")
    model_package_create_hf_gguf_parser.add_argument("--quantization", default="")
    model_package_create_hf_gguf_parser.add_argument("--timeout-sec", type=int, default=30)
    model_package_create_hf_gguf_parser.add_argument("--cache-chunks", action="store_true")
    model_package_create_hf_gguf_parser.add_argument("--pin-chunks", action="store_true")
    model_package_import_url_parser = subparsers.add_parser(
        "model-package-import-url",
        help="Import a CAI model package manifest from a URL",
    )
    model_package_import_url_parser.add_argument("manifest_url")
    model_package_import_url_parser.add_argument("--expected-model-id")
    model_package_import_url_parser.add_argument("--expected-preferred-filename")
    model_package_import_url_parser.add_argument("--timeout-sec", type=int, default=15)
    model_package_import_hf_parser = subparsers.add_parser(
        "model-package-import-hf",
        help="Discover and import a CAI model package manifest from a Hugging Face model repo",
    )
    model_package_import_hf_parser.add_argument("model_id")
    model_package_import_hf_parser.add_argument("--preferred-filename")
    model_package_import_hf_parser.add_argument("--source-revision", default="main")
    model_package_import_hf_parser.add_argument("--timeout-sec", type=int, default=15)
    launch_check_local_port = CaiNetworkConfig().default_api_port
    launch_check_parser = subparsers.add_parser(
        "launch-check", help="Run alpha launch readiness checks"
    )
    launch_check_parser.add_argument(
        "--local-state-url",
        default=f"http://127.0.0.1:{launch_check_local_port}/state",
        help="Local CAI runtime state URL",
    )
    launch_check_parser.add_argument(
        "--local-summary-url",
        default=f"http://127.0.0.1:{launch_check_local_port}/v1/cai/summary",
        help="Local CAI summary URL",
    )
    launch_check_parser.add_argument(
        "--remote-state-url",
        default="http://192.145.29.212:52415/state",
        help="Remote CAI runtime state URL",
    )
    launch_check_parser.add_argument(
        "--remote-summary-url",
        default="http://192.145.29.212:52415/v1/cai/summary",
        help="Remote CAI summary URL",
    )
    subparsers.add_parser("node-config", help="Show persisted validator/worker mode config")
    subparsers.add_parser("validator-set", help="Show local validator set and quorum")
    validator_sync_parser = subparsers.add_parser(
        "validator-set-sync", help="Import validator records from reachable CAI peers"
    )
    _add_cai_url_argument(
        validator_sync_parser,
        help_text="Local CAI runtime API base URL used to discover peer validator endpoints",
    )
    bind_parser = subparsers.add_parser(
        "worker-reward-bind",
        help="Bind a worker node id to a wallet payout address",
    )
    bind_parser.add_argument("--node-id", required=True)
    bind_parser.add_argument("--address", required=True)
    job_parser = subparsers.add_parser("job", help="Manage local job intents")
    job_subparsers = job_parser.add_subparsers(dest="job_command", required=True)
    job_create = job_subparsers.add_parser("create", help="Create a local job intent")
    job_create.add_argument("--prompt", required=True)
    job_create.add_argument(
        "--amount",
        help="Requested compute amount in coins; if omitted, bounded network auto-pricing is used",
    )
    job_create.add_argument(
        "--payment",
        choices=[mode.value for mode in PaymentPreference],
        default=PaymentPreference.AUTO.value,
    )
    _add_cai_url_argument(
        job_create,
        help_text="Base URL of the CAI runtime API",
    )
    job_create.add_argument("--model", help="Explicit model id")

    job_run = job_subparsers.add_parser(
        "run", help="Execute a local job intent through the CAI runtime"
    )
    job_run.add_argument("job_id")
    job_run.add_argument(
        "--timeout-sec",
        type=int,
        default=1800,
        help="HTTP timeout for the CAI runtime completion request",
    )

    job_subparsers.add_parser("list", help="List local job intents")
    receipt_parser = job_subparsers.add_parser(
        "receipts", help="List local execution receipts"
    )
    receipt_parser.add_argument("--limit", type=int, default=10)
    verify_parser = job_subparsers.add_parser(
        "verify",
        help="Verify the latest or selected execution receipt, network path, and reward accounting",
    )
    verify_target_group = verify_parser.add_mutually_exclusive_group()
    verify_target_group.add_argument("--receipt-id")
    verify_target_group.add_argument("--job-id")
    settlement_parser = subparsers.add_parser(
        "settlement", help="Show local settlement records"
    )
    settlement_parser.add_argument("--limit", type=int, default=10)
    attestation_parser = subparsers.add_parser(
        "attestation", help="Show local validator attestations"
    )
    attestation_parser.add_argument("--limit", type=int, default=10)
    evidence_parser = subparsers.add_parser(
        "validator-evidence", help="Show local validator evidence and penalties"
    )
    evidence_parser.add_argument("--limit", type=int, default=10)
    evidence_parser.add_argument("--validator-id")
    evidence_cases_parser = subparsers.add_parser(
        "validator-evidence-cases",
        help="Show aggregated validator evidence cases with quorum status",
    )
    evidence_cases_parser.add_argument("--limit", type=int, default=10)
    evidence_cases_parser.add_argument("--validator-id")
    evidence_sync_parser = subparsers.add_parser(
        "validator-evidence-sync",
        help="Import validator evidence from reachable CAI peers",
    )
    _add_cai_url_argument(
        evidence_sync_parser,
        help_text="Local CAI runtime API base URL used for peer discovery",
    )
    worker_payout_parser = subparsers.add_parser(
        "worker-payouts", help="Show local per-worker payout records"
    )
    worker_payout_parser.add_argument("--limit", type=int, default=10)
    subparsers.add_parser(
        "worker-payout-reconcile",
        help="Apply newly configured reward bindings to existing payout records",
    )
    validator_mode_parser = subparsers.add_parser(
        "validator-mode", help="Enable or disable validator mode"
    )
    validator_mode_group = validator_mode_parser.add_mutually_exclusive_group(required=True)
    validator_mode_group.add_argument("--enable", action="store_true")
    validator_mode_group.add_argument("--disable", action="store_true")
    validator_mode_parser.add_argument(
        "--state-url",
        default=f"http://127.0.0.1:{launch_check_local_port}/state",
        help="CAI runtime state URL used for validator eligibility checks",
    )
    validator_config_parser = subparsers.add_parser(
        "validator-config", help="Manage validator-specific network readiness flags"
    )
    validator_config_group = validator_config_parser.add_mutually_exclusive_group(required=True)
    validator_config_group.add_argument("--confirm-static-ip", action="store_true")
    validator_config_group.add_argument("--clear-static-ip", action="store_true")
    validator_ha_parser = subparsers.add_parser(
        "validator-ha",
        help="Configure active/passive HA replicas for one validator identity",
    )
    validator_ha_group = validator_ha_parser.add_mutually_exclusive_group(required=True)
    validator_ha_group.add_argument("--active", action="store_true")
    validator_ha_group.add_argument("--passive", action="store_true")
    validator_ha_group.add_argument("--disable", action="store_true")
    validator_ha_parser.add_argument(
        "--replica-id",
        help="Stable local replica identifier; generated automatically when omitted",
    )
    validator_ha_parser.add_argument(
        "--no-auto-failover",
        action="store_true",
        help="Keep this replica passive until it is promoted manually",
    )
    validator_ha_parser.add_argument(
        "--lease-seconds",
        type=int,
        help="Active replica lease duration in seconds",
    )
    validator_ha_parser.add_argument(
        "--state-url",
        default=f"http://127.0.0.1:{launch_check_local_port}/state",
        help="CAI runtime state URL used to resolve the local replica node id",
    )
    subparsers.add_parser(
        "validator-unjail",
        help="Clear validator jail state after cooldown has elapsed",
    )
    subparsers.add_parser(
        "validator-unbond-complete",
        help="Release validator bond after the unbonding delay has elapsed",
    )

    worker_mode_parser = subparsers.add_parser(
        "worker-mode", help="Enable or disable worker mode and resource hints"
    )
    worker_mode_group = worker_mode_parser.add_mutually_exclusive_group(required=True)
    worker_mode_group.add_argument("--enable", action="store_true")
    worker_mode_group.add_argument("--disable", action="store_true")
    worker_mode_parser.add_argument(
        "--allow-model",
        action="append",
        default=[],
        help="Model id to allow on this worker",
    )
    worker_mode_parser.add_argument(
        "--clear-models",
        action="store_true",
        help="Clear current worker model allow-list before applying updates",
    )
    worker_mode_parser.add_argument(
        "--max-parallel-jobs",
        type=int,
        help="Persist a local max-parallel-jobs hint",
    )
    worker_mode_parser.add_argument(
        "--max-memory-mb",
        type=int,
        help="Persist a local memory ceiling hint for worker execution",
    )
    relay_mode_parser = subparsers.add_parser(
        "relay-mode", help="Enable or disable relay capability for this node"
    )
    relay_mode_group = relay_mode_parser.add_mutually_exclusive_group(required=True)
    relay_mode_group.add_argument("--enable", action="store_true")
    relay_mode_group.add_argument("--disable", action="store_true")

    interface_parser = subparsers.add_parser(
        "interface-state", help="Show aggregated wallet/network/interface snapshot"
    )
    interface_parser.add_argument(
        "--state-url",
        default="http://127.0.0.1:52415/state",
        help="CAI runtime /state URL to inspect",
    )
    interface_parser.add_argument(
        "--quote-amount",
        help="Optional compute amount in coins to include in the compute panel",
    )
    interface_parser.add_argument(
        "--quote-prompt",
        help="Optional prompt to let the compute panel build an automatic network-priced quote",
    )
    interface_parser.add_argument(
        "--model",
        dest="quote_model",
        help="Model id to use with automatic quote preview",
    )
    _add_cai_url_argument(
        interface_parser,
        help_text="CAI runtime API base URL for automatic quote preview",
    )
    interface_parser.add_argument(
        "--payment",
        choices=[mode.value for mode in PaymentPreference],
        default=PaymentPreference.AUTO.value,
        help="Funding preference for compute quote preview",
    )
    interface_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit snapshot as JSON",
    )

    peer_book_parser = subparsers.add_parser(
        "peer-book", help="Show or update local bootstrap peer book"
    )
    peer_book_parser.add_argument(
        "--add",
        dest="peer_to_add",
        help="Add a bootstrap peer multiaddr to the local peer book",
    )
    peer_book_parser.add_argument(
        "--import-state-url",
        help="Import overlay-advertised peers from a CAI runtime /state endpoint",
    )
    peer_book_parser.add_argument(
        "--sync-bootstrap",
        action="store_true",
        help="Pull overlay-advertised peers from known bootstrap nodes",
    )

    update_parser = subparsers.add_parser(
        "update", help="Check or apply CAI source updates from the validator channel"
    )
    update_subparsers = update_parser.add_subparsers(dest="update_command", required=True)

    update_check_parser = update_subparsers.add_parser(
        "check", help="Check whether a validator update is available"
    )
    update_check_parser.add_argument(
        "--base-url",
        help="Explicit validator CAI API base URL",
    )
    update_check_parser.add_argument(
        "--repo-root",
        default=str(Path.cwd()),
        help="Local CAI source checkout root",
    )
    update_check_parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="HTTP timeout in seconds",
    )

    update_apply_parser = update_subparsers.add_parser(
        "apply", help="Download and apply the validator update package"
    )
    update_apply_parser.add_argument(
        "--base-url",
        help="Explicit validator CAI API base URL",
    )
    update_apply_parser.add_argument(
        "--repo-root",
        default=str(Path.cwd()),
        help="Local CAI source checkout root",
    )
    update_apply_parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="HTTP timeout in seconds",
    )

    wallet_parser = subparsers.add_parser("wallet", help="Manage local wallets")
    wallet_subparsers = wallet_parser.add_subparsers(dest="wallet_command", required=True)

    wallet_create = wallet_subparsers.add_parser("create", help="Create a local wallet scaffold")
    wallet_create.add_argument("--name", required=True, help="Wallet display name")
    wallet_create.add_argument(
        "--password",
        required=True,
        help="Wallet password for the local prototype store",
    )
    wallet_create.add_argument(
        "--select",
        action="store_true",
        help="Select the created wallet as active",
    )

    wallet_restore = wallet_subparsers.add_parser(
        "restore", help="Restore a wallet from an existing seed phrase"
    )
    wallet_restore.add_argument("--name", required=True, help="Wallet display name")
    wallet_restore.add_argument("--password", required=True)
    wallet_restore.add_argument("--seed-phrase", required=True)
    wallet_restore.add_argument(
        "--select",
        action="store_true",
        help="Select the restored wallet as active",
    )

    wallet_subparsers.add_parser("list", help="List local wallets")

    wallet_select = wallet_subparsers.add_parser("select", help="Select active wallet")
    wallet_select.add_argument("selector", help="Wallet id, address, or name")

    wallet_unlock = wallet_subparsers.add_parser("unlock", help="Unlock active wallet")
    wallet_unlock.add_argument("--password", required=True)
    wallet_unlock.add_argument(
        "--wallet",
        dest="wallet_selector",
        help="Optional wallet id, address, or name to unlock and select",
    )

    wallet_subparsers.add_parser("lock", help="Lock current wallet session")
    wallet_subparsers.add_parser("status", help="Show active wallet and session state")
    wallet_history = wallet_subparsers.add_parser(
        "history", help="Show local wallet journal entries"
    )
    wallet_history.add_argument(
        "--wallet",
        dest="wallet_selector",
        help="Wallet id, address, or name; defaults to active wallet",
    )
    wallet_history.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of entries to show",
    )

    wallet_credit = wallet_subparsers.add_parser(
        "credit", help="Credit a wallet balance for local testing"
    )
    wallet_credit.add_argument("--amount", required=True, help="Amount in coins, e.g. 12.50000000")
    wallet_credit.add_argument(
        "--wallet",
        dest="wallet_selector",
        help="Wallet id, address, or name; defaults to active wallet",
    )

    wallet_send = wallet_subparsers.add_parser(
        "send", help="Send coins to another wallet address"
    )
    wallet_send.add_argument("--to", required=True, dest="recipient_address")
    wallet_send.add_argument("--amount", required=True, help="Amount in coins to send")

    wallet_subparsers.add_parser(
        "developer-treasury",
        help="Provision the fixed developer treasury wallet from local secret files",
    )
    wallet_subparsers.add_parser(
        "ai-development",
        help="Provision the fixed AI development wallet from local secret files",
    )

    quote_parser = subparsers.add_parser(
        "job-quote", help="Quote how a compute job would be funded"
    )
    quote_parser.add_argument(
        "--amount",
        help="Requested compute amount in coins; if omitted, bounded network auto-pricing is used",
    )
    quote_parser.add_argument(
        "--prompt",
        help="Prompt for automatic pricing when --amount is omitted",
    )
    quote_parser.add_argument(
        "--model",
        help="Model id for automatic pricing",
    )
    _add_cai_url_argument(
        quote_parser,
        help_text="CAI runtime API base URL used to read live network state for automatic pricing",
    )
    quote_parser.add_argument(
        "--payment",
        choices=[mode.value for mode in PaymentPreference],
        default=PaymentPreference.AUTO.value,
        help="Funding preference",
    )

    fund_parser = subparsers.add_parser(
        "job-fund", help="Apply a local compute funding decision to wallet/ledger state"
    )
    fund_parser.add_argument(
        "--amount",
        help="Requested compute amount in coins; if omitted, bounded network auto-pricing is used",
    )
    fund_parser.add_argument(
        "--prompt",
        help="Prompt for automatic pricing when --amount is omitted",
    )
    fund_parser.add_argument(
        "--model",
        help="Model id for automatic pricing",
    )
    _add_cai_url_argument(
        fund_parser,
        help_text="CAI runtime API base URL used to read live network state for automatic pricing",
    )
    fund_parser.add_argument(
        "--payment",
        choices=[mode.value for mode in PaymentPreference],
        default=PaymentPreference.AUTO.value,
        help="Funding preference",
    )

    run_cai_parser = subparsers.add_parser(
        "run-cai", help="Launch the CAI runtime with built-in CAI network bootstrap"
    )
    _add_cai_executable_argument(run_cai_parser)
    _add_cai_home_argument(run_cai_parser)
    run_cai_parser.add_argument("--api-port", type=int)
    run_cai_parser.add_argument("--libp2p-port", type=int)
    run_cai_parser.add_argument("--no-downloads", action="store_true")
    run_cai_parser.add_argument("--no-worker", action="store_true")
    run_cai_parser.add_argument("--offline", action="store_true")
    run_cai_parser.add_argument("--force-master", action="store_true")
    run_cai_parser.add_argument("--verbose", action="store_true")
    run_cai_parser.add_argument("--dry-run", action="store_true")
    run_cai_parser.add_argument(
        "--sync-peer-book",
        action="store_true",
        help="Sync local peer book from bootstrap nodes before launch",
    )
    run_cai_parser.add_argument(
        "--advertise-peer",
        action="append",
        default=[],
        help="Advertise a public bootstrap multiaddr through the CAI runtime peer env",
    )

    owned_runtime_parser = subparsers.add_parser(
        "cai-owned-runtime",
        help="Run the CAI-owned shard runtime loop for local queued batches",
    )
    owned_runtime_parser.add_argument("--node-id", help="Local CAI node id")
    owned_runtime_parser.add_argument("--runtime-id", help="Runtime id for receipts")
    owned_runtime_parser.add_argument(
        "--coordinator-cai-url",
        help="Coordinator CAI API URL for receipt submission",
    )
    owned_runtime_parser.add_argument(
        "--wallet-data-dirname",
        help="Wallet/runtime data directory name; defaults to the active network policy",
    )
    owned_runtime_parser.add_argument(
        "--max-iterations",
        type=int,
        default=1,
        help="Maximum loop iterations; use 0 with --loop for an infinite worker loop",
    )
    owned_runtime_parser.add_argument(
        "--loop",
        action="store_true",
        help="Continue polling after the first iteration",
    )
    owned_runtime_parser.add_argument(
        "--idle-sleep-sec",
        type=float,
        default=1.0,
        help="Sleep between loop iterations when no work is claimed",
    )
    owned_runtime_parser.add_argument(
        "--max-payload-size-bytes",
        type=int,
        default=16 * 1024 * 1024,
    )
    owned_runtime_parser.add_argument("--lease-seconds", type=float, default=60.0)
    owned_runtime_parser.add_argument("--max-attempts", type=int, default=3)
    owned_runtime_parser.add_argument(
        "--local-runtime-token",
        help="Local runtime auth token; defaults to CAI_LOCAL_RUNTIME_TOKEN",
    )
    owned_runtime_parser.add_argument(
        "--require-local-runtime-auth",
        action="store_true",
    )
    owned_runtime_parser.add_argument(
        "--require-production-llm-handoff",
        action="store_true",
        help="Require production llmHandoff metadata before adapter execution",
    )
    owned_self_test_parser = subparsers.add_parser(
        "cai-owned-llm-shard-self-test",
        help="Run a local CAI-owned LLM shard adapter contract self-test",
    )
    owned_self_test_parser.add_argument(
        "--model-id",
        default=NetworkModelPolicy().network_default_model_id,
        help="Model id for the synthetic llmHandoff frame",
    )
    owned_self_test_parser.add_argument(
        "--payload",
        default="cai-llm-shard-self-test",
        help="Small UTF-8 payload used for the adapter contract probe",
    )
    owned_self_test_parser.add_argument(
        "--wallet-data-dirname",
        help="Wallet/runtime data directory name used when saving readiness",
    )
    owned_self_test_parser.add_argument(
        "--save-readiness",
        action="store_true",
        help="Persist the self-test result so node readiness can advertise it",
    )
    owned_self_test_parser.add_argument(
        "--show-cached",
        action="store_true",
        help="Show the cached self-test result instead of running a new probe",
    )
    owned_self_test_parser.add_argument(
        "--allow-non-production-handoff",
        action="store_true",
        help="Do not require strict production llmHandoff validation",
    )
    owned_conformance_parser = subparsers.add_parser(
        "cai-owned-llm-shard-conformance",
        help="Run CAI-owned LLM shard backend conformance checks",
    )
    owned_conformance_parser.add_argument(
        "--model-id",
        default=NetworkModelPolicy().network_default_model_id,
        help="Model id for the synthetic llmHandoff frame",
    )
    owned_conformance_parser.add_argument(
        "--payload",
        default="cai-llm-shard-conformance",
        help="Small UTF-8 payload used for the conformance probe",
    )
    owned_conformance_parser.add_argument(
        "--require-production",
        action="store_true",
        help="Fail unless the backend is productionReady, not only contractReady",
    )
    owned_conformance_parser.add_argument(
        "--allow-non-production-handoff",
        action="store_true",
        help="Do not require strict production llmHandoff validation",
    )
    owned_conformance_parser.add_argument(
        "--save-readiness",
        action="store_true",
        help="Persist the self-test result so node readiness can advertise it",
    )
    owned_conformance_parser.add_argument(
        "--wallet-data-dirname",
        help="Wallet/runtime data directory name used when saving readiness",
    )
    owned_conformance_parser.add_argument(
        "--json-report",
        help="Optional path to write the conformance JSON report",
    )
    owned_diagnostics_parser = subparsers.add_parser(
        "cai-owned-diagnostics",
        help="Export a secret-safe CAI-owned transport diagnostics snapshot",
    )
    owned_diagnostics_parser.add_argument(
        "--wallet-data-dirname",
        help="Wallet/runtime data directory name; defaults to the active network policy",
    )
    owned_diagnostics_parser.add_argument(
        "--local-node-id",
        help="Local node id used to include this node's batch inbox",
    )
    owned_diagnostics_parser.add_argument(
        "--model-id",
        help="Optional model id used to audit distributed inference readiness",
    )
    owned_diagnostics_parser.add_argument(
        "--max-records",
        type=int,
        default=50,
        help="Maximum records per snapshot section",
    )

    return parser


def handle_role(role: str) -> str:
    descriptions = {
        NodeRole.SEED.value: "Seed: bootstrap discovery only, no special trust.",
        NodeRole.PEER.value: "Peer/full-node: stores chain state and validates settlement.",
        NodeRole.VALIDATOR.value: "Validator: confirms blocks, network state, and settlement.",
        NodeRole.WORKER.value: "Worker: executes AI compute through the CAI runtime layer.",
        NodeRole.CLIENT.value: "Client: launches jobs, pays fees, and uses reserve or wallet funds.",
    }
    return descriptions[role]


def handle_verify_mode(mode: str) -> str:
    descriptions = {
        VerificationMode.DETERMINISTIC.value: "Deterministic verification: the task can be replayed or rechecked cheaply.",
        VerificationMode.REDUNDANT.value: "Redundant verification: several workers execute the same task independently.",
        VerificationMode.CHALLENGE.value: "Challenge verification: optimistic execution with dispute window.",
        VerificationMode.RECEIPT_ONLY.value: "Receipt-only: useful for prototypes, but unsafe for production rewards.",
    }
    return descriptions[mode]


def handle_cai_owned_runtime(
    *,
    node_id: str | None,
    runtime_id: str | None = None,
    coordinator_cai_url: str | None = None,
    wallet_data_dirname: str | None = None,
    max_iterations: int = 1,
    loop: bool = False,
    idle_sleep_sec: float = 1.0,
    max_payload_size_bytes: int = 16 * 1024 * 1024,
    lease_seconds: float = 60.0,
    max_attempts: int = 3,
    local_runtime_token: str | None = None,
    require_local_runtime_auth: bool | str | None = None,
    require_production_llm_handoff: bool | str | None = None,
) -> str:
    results = list(
        iter_cai_owned_runtime_results(
            node_id=node_id,
            runtime_id=runtime_id,
            coordinator_cai_url=coordinator_cai_url,
            wallet_data_dirname=wallet_data_dirname,
            max_iterations=max_iterations,
            loop=loop,
            idle_sleep_sec=idle_sleep_sec,
            max_payload_size_bytes=max_payload_size_bytes,
            lease_seconds=lease_seconds,
            max_attempts=max_attempts,
            local_runtime_token=local_runtime_token,
            require_local_runtime_auth=require_local_runtime_auth,
            require_production_llm_handoff=require_production_llm_handoff,
        )
    )
    if len(results) == 1:
        return json.dumps(results[0], ensure_ascii=False, indent=2, sort_keys=True)
    return json.dumps(
        {"iterationCount": len(results), "results": results},
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def handle_cai_owned_llm_shard_self_test(
    *,
    model_id: str | None = None,
    payload: str | bytes | None = None,
    require_production_llm_handoff: bool = True,
    save_readiness: bool = False,
    show_cached: bool = False,
    wallet_data_dirname: str | None = None,
) -> str:
    policy = (
        WalletPolicy(wallet_data_dirname=wallet_data_dirname)
        if wallet_data_dirname
        else WalletPolicy()
    )
    if show_cached:
        cached = load_cai_owned_llm_shard_self_test_result(policy=policy)
        return json.dumps(
            {
                "status": "cached" if cached is not None else "missing",
                "cached": cached,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    payload_bytes = (
        bytes(payload)
        if isinstance(payload, (bytes, bytearray))
        else str(payload or "cai-llm-shard-self-test").encode("utf-8")
    )
    result = run_cai_owned_llm_shard_adapter_self_test(
        model_id=model_id,
        payload=payload_bytes,
        require_production_llm_handoff=require_production_llm_handoff,
    )
    if save_readiness:
        result = dict(result)
        result["savedReadiness"] = save_cai_owned_llm_shard_self_test_result(
            result,
            policy=policy,
        )
    return json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)


def handle_cai_owned_llm_shard_conformance(
    *,
    model_id: str | None = None,
    payload: str | bytes | None = None,
    require_production: bool = False,
    require_production_llm_handoff: bool = True,
    save_readiness: bool = False,
    wallet_data_dirname: str | None = None,
    json_report: str | None = None,
) -> str:
    report = run_cai_owned_llm_shard_conformance(
        model_id=model_id,
        payload=payload,
        require_production=require_production,
        require_production_llm_handoff=require_production_llm_handoff,
        save_readiness=save_readiness,
        wallet_data_dirname=wallet_data_dirname,
    )
    rendered = conformance_report_to_json(report)
    if str(json_report or "").strip():
        Path(str(json_report).strip()).write_text(rendered + "\n", encoding="utf-8")
    return rendered


def handle_cai_owned_diagnostics(
    *,
    wallet_data_dirname: str | None = None,
    local_node_id: str | None = None,
    model_id: str | None = None,
    max_records: int | None = 50,
) -> str:
    policy = (
        WalletPolicy(wallet_data_dirname=wallet_data_dirname)
        if wallet_data_dirname
        else WalletPolicy()
    )
    snapshot = build_cai_owned_diagnostics_snapshot(
        local_node_id=local_node_id,
        model_id=model_id,
        max_records=max_records,
        policy=policy,
    )
    return json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True)


def iter_cai_owned_runtime_results(
    *,
    node_id: str | None,
    runtime_id: str | None = None,
    coordinator_cai_url: str | None = None,
    wallet_data_dirname: str | None = None,
    max_iterations: int = 1,
    loop: bool = False,
    idle_sleep_sec: float = 1.0,
    max_payload_size_bytes: int = 16 * 1024 * 1024,
    lease_seconds: float = 60.0,
    max_attempts: int = 3,
    local_runtime_token: str | None = None,
    require_local_runtime_auth: bool | str | None = None,
    require_production_llm_handoff: bool | str | None = None,
):
    resolved_node_id = (
        str(node_id or "").strip()
        or os.getenv("CAI_NODE_ID", "").strip()
        or os.getenv("CAI_LOCAL_NODE_ID", "").strip()
    )
    if not resolved_node_id:
        raise ValueError("CAI-owned runtime requires --node-id or CAI_NODE_ID.")
    resolved_runtime_id = (
        str(runtime_id or "").strip()
        or os.getenv("CAI_OWNED_RUNTIME_ID", "").strip()
        or f"cai-owned-runtime-{resolved_node_id}"
    )
    policy = (
        WalletPolicy(wallet_data_dirname=wallet_data_dirname)
        if wallet_data_dirname
        else WalletPolicy()
    )
    config = CaiOwnedShardRuntimeConfig(
        node_id=resolved_node_id,
        runtime_id=resolved_runtime_id,
        coordinator_cai_url=str(coordinator_cai_url or "").strip() or None,
        max_payload_size_bytes=max(1, int(max_payload_size_bytes or 1)),
        lease_seconds=max(0.1, float(lease_seconds or 60.0)),
        max_attempts=max(1, int(max_attempts or 1)),
        local_runtime_auth_token=(
            local_runtime_token or os.getenv("CAI_LOCAL_RUNTIME_TOKEN")
        ),
        require_local_runtime_auth=bool(require_local_runtime_auth),
        require_production_llm_handoff=(
            bool(require_production_llm_handoff)
            or _env_truthy("CAI_REQUIRE_PRODUCTION_LLM_HANDOFF")
        ),
        policy=policy,
    )
    adapter = cai_owned_shard_adapter_from_env()
    limit = int(max_iterations or 0)
    iteration = 0
    while True:
        iteration += 1
        result = run_cai_owned_shard_runtime_once(config, adapter)
        result = dict(result)
        result.setdefault("iteration", iteration)
        result.setdefault("nodeId", resolved_node_id)
        result.setdefault("runtimeId", resolved_runtime_id)
        yield result
        if not loop:
            break
        if limit > 0 and iteration >= limit:
            break
        if str(result.get("status") or "") == "idle":
            time.sleep(max(0.0, float(idle_sleep_sec or 0.0)))


def _env_truthy(key: str) -> bool:
    return str(os.getenv(key) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        "required",
    }


def handle_topology() -> str:
    return dedent(
        """
        Target topology:
        - CAI runtime is the execution base for distributed AI compute
        - peer/full-nodes maintain chain state and balances
        - validators confirm blocks and settlement
        - workers execute useful AI compute
        - clients launch jobs using reserve or wallet funds
        - transaction fees are always paid by the transaction initiator
        """
    ).strip()


def handle_network() -> str:
    config = CaiNetworkConfig()
    peer_book = load_peer_book()
    peers = "\n".join(f"- {peer}" for peer in config.bootstrap_peers)
    remembered_peers = (
        "\n".join(f"- {peer}" for peer in peer_book) if peer_book else "- <empty>"
    )
    return dedent(
        f"""
        CAI runtime config:
        - chain_network={config.chain_network.value}
        - namespace={config.namespace}
        - overlay_first={config.overlay_first}
        - direct_inbound_required={config.direct_inbound_required}
        - default_api_port={config.default_api_port}
        - default_libp2p_port={config.default_libp2p_port}
        - default_cai_home_dirname={config.default_cai_home_dirname}
        - advertise_env_var_name={config.advertise_env_var_name}
        - bootstrap_peers:
        {peers}
        - peer_book_path={peer_book_path()}
        - peer_book_entries:
        {remembered_peers}
        """
    ).strip()


def handle_network_model() -> str:
    policy = NetworkModelPolicy()
    privacy = ModelPrivacyPrinciples()
    return dedent(
        f"""
        Network model policy:
        - mode={policy.mode.value}
        - network_default_model_id={policy.network_default_model_id}
        - user_must_download_model={policy.user_must_download_model}
        - client_receives_weights={policy.client_receives_weights}
        - private_model_allows_full_single_worker_copy={policy.private_model_allows_full_single_worker_copy}
        - private_model_requires_sharded_distribution={policy.private_model_requires_sharded_distribution}
        - minimum_worker_shards={policy.minimum_worker_shards}
        - allow_single_node_private_inference={policy.allow_single_node_private_inference}
        - shard_cache_encrypted_at_rest={policy.shard_cache_encrypted_at_rest}
        - shard_keys_released_per_assignment={policy.shard_keys_released_per_assignment}
        - model_owner_can_publish_public_model={policy.model_owner_can_publish_public_model}
        - absolute_leak_protection_on_untrusted_hardware={policy.absolute_leak_protection_on_untrusted_hardware}
        - v1_delivery_strategy={policy.v1_delivery_strategy}

        Model privacy principles:
        - no_client_weight_distribution={privacy.no_client_weight_distribution}
        - no_full_model_distribution_to_untrusted_workers={privacy.no_full_model_distribution_to_untrusted_workers}
        - overlay_network_model_access={privacy.overlay_network_model_access}
        - validator_gated_shard_assignment={privacy.validator_gated_shard_assignment}
        - per_shard_integrity_verification={privacy.per_shard_integrity_verification}
        - per_shard_decryption_keys={privacy.per_shard_decryption_keys}
        - confidential_compute_required_for_strongest_protection={privacy.confidential_compute_required_for_strongest_protection}
        """
    ).strip()


def handle_status() -> str:
    status = RoadmapStatus(
        execution_base="cai_runtime_locked_in",
        currency_layer="chain_event_ledger_scaffold",
        validator_consensus="model_fixed",
        settlement_layer="validator_settlement_chain_feed",
    )
    principles = NetworkPrinciples()
    return dedent(
        f"""
        Roadmap status:
        - execution_base={status.execution_base}
        - currency_layer={status.currency_layer}
        - validator_consensus={status.validator_consensus}
        - settlement_layer={status.settlement_layer}

        Network principles:
        - uses_cai_runtime_base={principles.uses_cai_runtime_base}
        - minimize_extra_protocol_logic={principles.minimize_extra_protocol_logic}
        - has_validator_role={principles.has_validator_role}
        - has_dedicated_centers={principles.has_dedicated_centers}
        - has_claim_protocol={principles.has_claim_protocol}
        - worker_payments_settled_on_chain={principles.worker_payments_settled_on_chain}
        - reserve_can_fund_compute={principles.reserve_can_fund_compute}
        - tx_fee_paid_by_initiator={principles.tx_fee_paid_by_initiator}
        """
    ).strip()


def handle_money_policy() -> str:
    policy = MoneyPolicy()
    return dedent(
        f"""
        Monetary policy:
        - chain_network={policy.chain_network.value}
        - currency_code={policy.currency_code}
        - reward_token_code={policy.reward_token_code}
        - decimals={policy.decimals}
        - total_supply_coins={policy.total_supply_coins}
        - compute_reserve_share={policy.compute_reserve_share}
        - compute_reserve_coins={policy.compute_reserve_coins}
        - founder_treasury_share={policy.developer_treasury_share}
        - founder_treasury_coins={policy.developer_treasury_coins}
        - developer_treasury_share={policy.developer_treasury_share}
        - developer_treasury_coins={policy.developer_treasury_coins}
        - developer_contribution_fund_share={policy.developer_contribution_fund_share}
        - developer_contribution_fund_coins={policy.developer_contribution_fund_coins}
        - developer_treasury_wallet_id={policy.developer_treasury_wallet_id}
        - developer_treasury_address={policy.developer_treasury_address}
        - ai_development_wallet_id={policy.ai_development_wallet_id}
        - ai_development_address={policy.ai_development_address}
        - ai_development_fee_bps={policy.ai_development_fee_bps}
        - daily_user_reserve_limit_enabled={policy.daily_user_reserve_limit_enabled}
        - daily_user_reserve_limit_coins={policy.daily_user_reserve_limit_coins}
        - daily_ip_reserve_limit_enabled={policy.daily_ip_reserve_limit_enabled}
        - daily_ip_reserve_limit_coins={policy.daily_ip_reserve_limit_coins}
        - reserve_funds_compute_first={policy.reserve_funds_compute_first}
        - user_can_fund_jobs_from_wallet={policy.user_can_fund_jobs_from_wallet}
        - tx_fee_paid_by_initiator={policy.tx_fee_paid_by_initiator}
        - default_tx_fee_coins={policy.default_tx_fee_coins}
        - validator_settlement_fee_bps={policy.validator_settlement_fee_bps}
        - validator_min_bond_coins={policy.validator_min_bond_coins}
        - validator_committee_target_size={policy.validator_committee_target_size}
        - validator_committee_selection_mode={policy.validator_committee_selection_mode}
        - validator_jail_slash_bps={policy.validator_jail_slash_bps}
        - validator_conflicting_attestation_slash_bps={policy.validator_conflicting_attestation_slash_bps}
        - validator_unbonding_seconds={policy.validator_unbonding_seconds}
        - validator_unjail_cooldown_seconds={policy.validator_unjail_cooldown_seconds}
        - automatic_pricing_enabled={policy.automatic_pricing_enabled}
        - automatic_token_pricing_enabled={policy.automatic_token_pricing_enabled}
        - automatic_price_floor_coins={policy.automatic_price_floor_coins}
        - automatic_price_cap_coins={policy.automatic_price_cap_coins}
        - automatic_price_per_input_token_coins={policy.automatic_price_per_input_token_coins}
        - automatic_price_per_output_token_coins={policy.automatic_price_per_output_token_coins}
        - automatic_price_default_reserved_output_tokens={policy.automatic_price_default_reserved_output_tokens}
        - automatic_price_prompt_unit_chars={policy.automatic_price_prompt_unit_chars}
        - automatic_price_per_prompt_unit_coins={policy.automatic_price_per_prompt_unit_coins}
        """
    ).strip()


def handle_ledger() -> str:
    money_policy = MoneyPolicy()
    wallet_policy = WalletPolicy()
    ensure_chain_genesis(policy=wallet_policy, money_policy=money_policy)
    ledger = load_or_create_ledger(money_policy, wallet_policy)
    chain_state = chain_summary(wallet_policy)
    chain_initialized = int(chain_state.get("blockCount") or 0) > 0

    def balance_coins(chain_key: str, fallback_atomic: int) -> str:
        value = chain_state.get(chain_key)
        if chain_initialized and isinstance(value, str) and value:
            return value
        return atomic_to_coins(fallback_atomic, money_policy)

    return dedent(
        f"""
        Ledger status:
        - chain_network={money_policy.chain_network.value}
        - data_root={data_root(wallet_policy)}
        - balance_source={'chain' if chain_initialized else 'ledger_cache'}
        - chain_block_count={chain_state.get('blockCount', 0)}
        - chain_transaction_count={chain_state.get('transactionCount', 0)}
        - chain_tip_height={chain_state.get('tipHeight') if chain_state.get('tipHeight') is not None else '<none>'}
        - chain_valid={str(bool(chain_state.get('valid'))).lower()}
        - compute_reserve_balance={balance_coins('computeReserveBalanceCoins', ledger.compute_reserve_balance_atomic)}
        - project_treasury_balance={balance_coins('developerTreasuryBalanceCoins', ledger.project_treasury_balance_atomic)}
        - developer_treasury_wallet_id={ledger.developer_treasury_wallet_id or '<none>'}
        - developer_treasury_address={ledger.developer_treasury_address or '<none>'}
        - developer_treasury_allocated={atomic_to_coins(ledger.developer_treasury_allocated_atomic, money_policy)}
        - developer_treasury_provisioned_locally={str(ledger.developer_treasury_provisioned_locally).lower()}
        - developer_treasury_seed_file={ledger.developer_treasury_seed_file or developer_treasury_seed_file_path()}
        - developer_treasury_password_file={ledger.developer_treasury_password_file or developer_treasury_password_file_path()}
        - ai_development_wallet_id={ledger.ai_development_wallet_id or '<none>'}
        - ai_development_address={ledger.ai_development_address or '<none>'}
        - ai_development_provisioned_locally={str(ledger.ai_development_provisioned_locally).lower()}
        - ai_development_seed_file={ledger.ai_development_seed_file or ai_development_seed_file_path()}
        - ai_development_password_file={ledger.ai_development_password_file or ai_development_password_file_path()}
        - ai_development_fee_pool={balance_coins('aiDevelopmentBalanceCoins', ledger.ai_development_fee_pool_atomic)}
        - validator_fee_pool={balance_coins('validatorSettlementFeePoolBalanceCoins', ledger.validator_fee_pool_atomic)}
        - tx_fee_pool={balance_coins('txFeePoolBalanceCoins', ledger.tx_fee_pool_atomic)}
        - worker_distributed_ledger_cache={atomic_to_coins(ledger.worker_distributed_atomic, money_policy)}
        - settlements_applied={ledger.settlements_applied}
        """
    ).strip()


def _developer_fund_validation_text(result) -> str:
    money_policy = MoneyPolicy()
    lines = [
        "Developer fund validation:",
        f"- participants_file={result.participants_path}",
        f"- round_file={result.round_path}",
        f"- round_id={result.round_id or '<none>'}",
        f"- round_status={result.round_status}",
        f"- round_type={result.round_type or '<none>'}",
        f"- participants_count={result.participants_count}",
        f"- winner_count={result.winner_count}",
        f"- total_amount={atomic_to_coins(result.total_amount_atomic, money_policy)}",
        f"- vote_outcome={result.vote_outcome or '<none>'}",
        (
            "- founder_confirmation="
            f"{result.founder_confirmation_status or '<none>'}"
        ),
        f"- signed_vote_count={result.signed_vote_count}",
        f"- participants_hash={result.participants_hash or '<none>'}",
        f"- round_hash={result.round_hash or '<none>'}",
        f"- distributable={str(result.distributable).lower()}",
    ]
    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {item}" for item in result.warnings)
    if result.errors:
        lines.append("Errors:")
        lines.extend(f"- {item}" for item in result.errors)
    if result.recipients:
        lines.append("Recipients:")
        for recipient in result.recipients[:25]:
            lines.append(
                f"- @{recipient.github} amount={recipient.amount_coins} "
                f"category={recipient.category} address={recipient.address}"
            )
        if len(result.recipients) > 25:
            lines.append(f"- ... {len(result.recipients) - 25} more")
    return "\n".join(lines)


def handle_developer_fund_validate(
    *,
    repo_root: str,
    participants_path: str | None,
    round_path: str | None,
) -> tuple[str, bool]:
    result = validate_developer_fund_files(
        repo_root=repo_root,
        participants_path=participants_path,
        round_path=round_path,
    )
    return _developer_fund_validation_text(result), result.ok


def handle_developer_fund_distribute(
    *,
    repo_root: str,
    participants_path: str | None,
    round_path: str | None,
    validator_id: str | None,
    source_commit: str | None,
    dry_run: bool,
) -> tuple[str, bool]:
    money_policy = MoneyPolicy()
    wallet_policy = WalletPolicy()
    result = validate_developer_fund_files(
        repo_root=repo_root,
        participants_path=participants_path,
        round_path=round_path,
        money_policy=money_policy,
    )
    lines = [_developer_fund_validation_text(result)]
    if not result.ok:
        return "\n".join(lines), False
    if not result.distributable:
        lines.append(
            "Distribution blocked: round status must be approved."
        )
        return "\n".join(lines), False

    ensure_chain_genesis(policy=wallet_policy, money_policy=money_policy)
    paid_rounds = developer_fund_distribution_round_ids(wallet_policy)
    if result.round_id in paid_rounds:
        lines.append(f"Distribution blocked: round '{result.round_id}' is already paid.")
        return "\n".join(lines), False

    fund_address = developer_contribution_fund_chain_address(money_policy)
    fund_balance_atomic = chain_balance_atomic(fund_address, wallet_policy)
    if result.total_amount_atomic > fund_balance_atomic:
        lines.append("Distribution blocked: developer fund balance is too low.")
        lines.append(
            f"- fund_balance={atomic_to_coins(fund_balance_atomic, money_policy)}"
        )
        return "\n".join(lines), False

    if dry_run:
        lines.append("Dry run:")
        lines.append("- chain_write=false")
        lines.append(f"- planned_transactions={len(result.recipients) + 1}")
        lines.append(f"- fund_address={fund_address}")
        return "\n".join(lines), True

    block = record_developer_fund_distribution(
        round_id=result.round_id,
        recipients=developer_fund_recipients_for_chain(result),
        round_hash=result.round_hash,
        participants_hash=result.participants_hash,
        source_commit=source_commit,
        validator_id=validator_id,
        policy=wallet_policy,
        money_policy=money_policy,
    )
    lines.append("Distribution recorded:")
    lines.append("- chain_write=true")
    lines.append(f"- block_height={block.height}")
    lines.append(f"- block_hash={block.block_hash}")
    lines.append(f"- transactions={len(block.transactions)}")
    lines.append(f"- fund_address={fund_address}")
    return "\n".join(lines), True


def handle_developer_fund_sign_vote(
    *,
    round_id: str,
    github: str,
    choices: list[str],
) -> str:
    wallet, session = _require_active_wallet()
    if session.unlocked_wallet_id != wallet.wallet_id:
        raise SystemExit("Active wallet must be unlocked before signing a vote.")
    signing_material = load_unlocked_wallet_signing_material(wallet)
    if signing_material is None:
        raise SystemExit("Unlocked wallet signing material is not available.")
    payload = developer_fund_vote_signing_payload(
        round_id=round_id,
        github=github,
        cai_address=wallet.address,
        choices=choices,
    )
    signature_b64 = sign_payload_b64(
        decode_bytes(str(signing_material["signing_seed_b64"])),
        payload,
    )
    vote = {
        "github": str(github).strip().lstrip("@").lower(),
        "choices": [str(choice).strip().lstrip("@").lower() for choice in choices],
        "address_scheme": wallet.address_scheme,
        "public_key_b64": wallet.public_key_b64,
        "signature_b64": signature_b64,
    }
    if wallet.signing_scheme == SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65:
        pq_private_key_b64 = str(signing_material.get("pq_private_key_b64") or "")
        if not pq_private_key_b64:
            raise SystemExit("Hybrid wallet vote requires post-quantum signing material.")
        vote["pq_public_key_b64"] = wallet.pq_public_key_b64
        vote["pq_signature_scheme"] = SIGNING_SCHEME_ML_DSA_65
        vote["pq_signature_b64"] = sign_payload_mldsa65_b64(
            pq_private_key_b64,
            payload,
        )
    return json.dumps(vote, ensure_ascii=False, indent=2, sort_keys=True)


def handle_developer_fund_sign_founder_confirmation(
    *,
    repo_root: str,
    participants_path: str | None,
    round_path: str | None,
    confirmed_by: str,
) -> str:
    wallet, session = _require_active_wallet()
    if session.unlocked_wallet_id != wallet.wallet_id:
        raise SystemExit(
            "Active founder wallet must be unlocked before signing confirmation."
        )
    money_policy = MoneyPolicy()
    if normalize_address(wallet.address) != normalize_address(
        money_policy.developer_treasury_address
    ):
        raise SystemExit(
            "Active wallet address must match the developer treasury/founder address."
        )
    signing_material = load_unlocked_wallet_signing_material(wallet)
    if signing_material is None:
        raise SystemExit("Unlocked wallet signing material is not available.")

    root = Path(repo_root)
    resolved_participants_path = resolve_developer_fund_path(
        root,
        participants_path,
        DEFAULT_PARTICIPANTS_PATH,
    )
    resolved_round_path = resolve_developer_fund_path(
        root,
        round_path,
        DEFAULT_ROUNDS_DIR / DEFAULT_ROUND_FILE_NAME,
    )
    participants_payload = json.loads(
        resolved_participants_path.read_text(encoding="utf-8")
    )
    round_payload = json.loads(resolved_round_path.read_text(encoding="utf-8"))
    if not isinstance(round_payload, dict):
        raise SystemExit("Round file must be a JSON object.")

    confirmed_at = datetime.now(UTC).isoformat()
    payload = developer_fund_founder_confirmation_signing_payload(
        round_payload=round_payload,
        participants_hash=canonical_json_hash(participants_payload),
        confirmed_by=confirmed_by,
        confirmed_at=confirmed_at,
    )
    confirmation = {
        "confirmed_by": str(confirmed_by).strip(),
        "confirmed_at": confirmed_at,
        "address_scheme": wallet.address_scheme,
        "public_key_b64": wallet.public_key_b64,
        "signature_b64": sign_payload_b64(
            decode_bytes(str(signing_material["signing_seed_b64"])),
            payload,
        ),
    }
    if wallet.signing_scheme == SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65:
        pq_private_key_b64 = str(signing_material.get("pq_private_key_b64") or "")
        if not pq_private_key_b64:
            raise SystemExit(
                "Hybrid founder confirmation requires post-quantum signing material."
            )
        confirmation["pq_public_key_b64"] = wallet.pq_public_key_b64
        confirmation["pq_signature_scheme"] = SIGNING_SCHEME_ML_DSA_65
        confirmation["pq_signature_b64"] = sign_payload_mldsa65_b64(
            pq_private_key_b64,
            payload,
        )
    return json.dumps(confirmation, ensure_ascii=False, indent=2, sort_keys=True)


def handle_model_package_list() -> str:
    manifests = list_model_package_manifests()
    if not manifests:
        return "<empty>"
    lines = []
    for manifest in manifests:
        lines.append(
            (
                f"- {manifest.catalog_id}@{manifest.version} "
                f"backend={manifest.backend} "
                f"kind={manifest.package_kind} "
                f"chunks={len(manifest.chunks)} "
                f"total_bytes={manifest.total_size_bytes}"
            )
        )
    return "\n".join(lines)


def handle_model_package_show(catalog_id: str, version: str) -> str:
    manifest = load_model_package_manifest(catalog_id, version)
    return json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2)


def handle_model_package_plan(
    catalog_id: str,
    version: str,
    *,
    start_layer: int,
    end_layer: int,
    device_rank: int,
    world_size: int,
    node_id: str | None,
) -> str:
    manifest = load_model_package_manifest(catalog_id, version)
    plan = build_assignment_chunk_plan_from_store(
        manifest,
        ModelShardAssignment(
            start_layer=start_layer,
            end_layer=end_layer,
            device_rank=device_rank,
            world_size=world_size,
            node_id=node_id,
        ),
    )
    return dedent(
        f"""
        Assignment chunk plan:
        - catalog_id={catalog_id}
        - version={version}
        - layer_range={start_layer}:{end_layer}
        - device_rank={device_rank}
        - world_size={world_size}
        - node_id={node_id or '<none>'}
        - ready={str(plan.ready).lower()}
        - required_chunks={len(plan.coverage.required_chunk_ids)}
        - present_chunks={len(plan.coverage.present_chunk_ids)}
        - missing_chunks={len(plan.coverage.missing_chunk_ids)}
        - required_bytes={plan.coverage.required_bytes}
        - present_bytes={plan.coverage.present_bytes}
        - fetch_bytes={plan.estimated_fetch_bytes}
        """
    ).strip()


def _load_chunk_inventory(path: str | None) -> dict[str, tuple[str, ...]] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Chunk inventory JSON must be an object mapping source ids to chunk lists.")
    normalized: dict[str, tuple[str, ...]] = {}
    for source_id, chunk_ids in payload.items():
        if not isinstance(chunk_ids, list):
            raise ValueError("Chunk inventory values must be arrays of chunk ids.")
        normalized[str(source_id)] = tuple(str(chunk_id) for chunk_id in chunk_ids)
    return normalized


def _manifest_total_layer_count(manifest) -> int:
    for key in ("total_layers", "layer_count", "n_layers"):
        try:
            value = int(manifest.metadata.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value

    layer_ends = [
        int(chunk.layer_end)
        for chunk in manifest.chunks
        if chunk.layer_end is not None and int(chunk.layer_end) > 0
    ]
    if layer_ends:
        return max(layer_ends)

    raise ValueError(
        "Model package does not expose a total layer count; "
        "cannot derive a full-model shard assignment."
    )


def handle_chunk_inventory_local(source_id: str, *, source_kind: str) -> str:
    payload = build_local_chunk_inventory_payload(
        source_id,
        source_kind=source_kind,
    )
    saved_path = save_local_chunk_inventory_payload(
        source_id,
        source_kind=source_kind,
    )
    return dedent(
        f"""
        Local chunk inventory saved:
        - source_id={payload.source_id}
        - source_kind={payload.source_kind}
        - inventory_path={saved_path}
        - manifest_count={len(payload.records)}
        - package_manifest_count={len(payload.manifests)}
        - chunk_count={sum(record.chunk_count for record in payload.records)}
        - total_bytes={sum(record.total_bytes for record in payload.records)}
        """
    ).strip()


def handle_chunk_inventory_import(path: str) -> str:
    payload = load_chunk_inventory_payload(path)
    saved_path = import_chunk_inventory_payload(payload)
    return dedent(
        f"""
        Chunk inventory imported:
        - source_id={payload.source_id}
        - source_kind={payload.source_kind}
        - saved_path={saved_path}
        - manifest_count={len(payload.records)}
        - package_manifest_count={len(payload.manifests)}
        - chunk_count={sum(record.chunk_count for record in payload.records)}
        - total_bytes={sum(record.total_bytes for record in payload.records)}
        """
    ).strip()


def handle_chunk_inventory_list(source_kind: str | None = None) -> str:
    payloads = list_imported_chunk_inventory_payloads(source_kind=source_kind)
    if not payloads:
        return "<empty>"
    lines = ["Imported chunk inventories:"]
    for payload in payloads:
        lines.extend(
            [
                f"- source_id={payload.source_id}",
                f"  source_kind={payload.source_kind}",
                f"  manifest_count={len(payload.records)}",
                f"  chunk_count={sum(record.chunk_count for record in payload.records)}",
                f"  total_bytes={sum(record.total_bytes for record in payload.records)}",
            ]
        )
    return "\n".join(lines)


def handle_chunk_inventory_sync(
    cai_url: str,
    *,
    source_kind: str,
    inventory_urls: list[str] | tuple[str, ...] | None = None,
) -> str:
    if inventory_urls:
        result = sync_chunk_inventory_from_urls(
            inventory_urls=inventory_urls,
            source_kind=source_kind,
        )
    else:
        state_url = cai_url.rstrip("/") + "/state"
        with urlopen(state_url, timeout=5) as response:
            state_payload = json.loads(response.read().decode("utf-8"))

        result = sync_chunk_inventory_from_cai_peers(
            state_payload=state_payload,
            cai_url=cai_url,
            source_kind=source_kind,
    )
    lines = [
        dedent(
            f"""
            Chunk inventory sync completed:
            - source_kind={source_kind}
            - attempted_peers={result.attempted_peers}
            - successful_peers={result.successful_peers}
            - failed_peers={result.failed_peers}
            - imported_payloads={result.imported_payloads}
            - pruned_payloads={result.pruned_payloads}
            """
        ).strip()
    ]
    if result.peer_errors:
        lines.append("- peer_errors:")
        lines.extend(
            f"  - {item['peerUrl']}: {item['errorType']}: {item['message']}"
            for item in result.peer_errors
        )
    return "\n".join(lines)


def handle_chunk_download_queue() -> str:
    snapshot = chunk_download_queue_snapshot()
    stats = snapshot.stats
    lines = dedent(
        f"""
        Chunk download queue:
        - tasks={stats.task_count}
        - queued={stats.queued_count}
        - in_progress={stats.in_progress_count}
        - completed={stats.completed_count}
        - failed={stats.failed_count}
        - total_bytes={stats.total_bytes}
        - queued_bytes={stats.queued_bytes}
        - completed_bytes={stats.completed_bytes}
        """
    ).strip().splitlines()
    lines.extend(_chunk_task_source_lines("queue", snapshot.tasks))
    return "\n".join(lines)


def handle_chunk_source_health(source_kind: str | None = None) -> str:
    records = list_chunk_source_health_records()
    if source_kind:
        records = [record for record in records if record.source_kind == source_kind]
    if not records:
        return "<empty>"
    lines = ["Chunk source health:"]
    for record in sorted(
        records,
        key=lambda item: (item.source_kind, item.source_id, str(item.locator or "")),
    ):
        lines.extend(
            [
                f"- source_kind={record.source_kind}",
                f"  source_id={record.source_id}",
                f"  locator={record.locator or '<none>'}",
                f"  success_count={record.success_count}",
                f"  failure_count={record.failure_count}",
                f"  consecutive_failures={record.consecutive_failures}",
                f"  cooldown_until={record.cooldown_until or '<none>'}",
                f"  last_error={record.last_error or '<none>'}",
            ]
        )
    return "\n".join(lines)


def handle_model_package_fetch_plan(
    catalog_id: str,
    version: str,
    *,
    start_layer: int,
    end_layer: int,
    device_rank: int,
    world_size: int,
    node_id: str | None,
    peer_inventory_json: str | None,
    seed_inventory_json: str | None,
    use_imported_peer_inventory: bool,
    use_imported_seed_inventory: bool,
) -> str:
    manifest = load_model_package_manifest(catalog_id, version)
    plan = build_assignment_fetch_plan_from_store(
        manifest,
        ModelShardAssignment(
            start_layer=start_layer,
            end_layer=end_layer,
            device_rank=device_rank,
            world_size=world_size,
            node_id=node_id,
        ),
        peer_chunk_inventory=_load_chunk_inventory(peer_inventory_json),
        seed_chunk_inventory=_load_chunk_inventory(seed_inventory_json),
        use_imported_peer_inventory=use_imported_peer_inventory,
        use_imported_seed_inventory=use_imported_seed_inventory,
    )

    peer_first = sum(
        1
        for request in plan.fetch_requests
        if request.sources
        and request.sources[0].kind == ChunkFetchSourceKind.PEER_CACHE
    )
    seed_first = sum(
        1
        for request in plan.fetch_requests
        if request.sources
        and request.sources[0].kind == ChunkFetchSourceKind.STORAGE_SEED
    )
    origin_first = sum(
        1
        for request in plan.fetch_requests
        if request.sources
        and request.sources[0].kind == ChunkFetchSourceKind.ORIGIN
    )
    no_source = sum(1 for request in plan.fetch_requests if not request.sources)
    imported_peer_sources = (
        len(
            build_chunk_inventory_index(
                catalog_id,
                version,
                source_kind=ChunkInventorySourceKind.PEER_CACHE,
            )
        )
        if use_imported_peer_inventory
        else 0
    )
    imported_seed_sources = (
        len(
            build_chunk_inventory_index(
                catalog_id,
                version,
                source_kind=ChunkInventorySourceKind.STORAGE_SEED,
            )
        )
        if use_imported_seed_inventory
        else 0
    )

    return dedent(
        f"""
        Assignment fetch plan:
        - catalog_id={catalog_id}
        - version={version}
        - layer_range={start_layer}:{end_layer}
        - device_rank={device_rank}
        - world_size={world_size}
        - node_id={node_id or '<none>'}
        - ready={str(plan.ready).lower()}
        - missing_chunks={len(plan.fetch_requests)}
        - fetch_bytes={plan.estimated_fetch_bytes}
        - peer_first={peer_first}
        - seed_first={seed_first}
        - origin_first={origin_first}
        - no_source={no_source}
        - imported_peer_sources={imported_peer_sources}
        - imported_seed_sources={imported_seed_sources}
        """
    ).strip()


def handle_model_package_enqueue_fetch(
    catalog_id: str,
    version: str,
    *,
    start_layer: int,
    end_layer: int,
    device_rank: int,
    world_size: int,
    node_id: str | None,
    peer_inventory_json: str | None,
    seed_inventory_json: str | None,
    use_imported_peer_inventory: bool,
    use_imported_seed_inventory: bool,
) -> str:
    manifest = load_model_package_manifest(catalog_id, version)
    fetch_plan = build_assignment_fetch_plan_from_store(
        manifest,
        ModelShardAssignment(
            start_layer=start_layer,
            end_layer=end_layer,
            device_rank=device_rank,
            world_size=world_size,
            node_id=node_id,
        ),
        peer_chunk_inventory=_load_chunk_inventory(peer_inventory_json),
        seed_chunk_inventory=_load_chunk_inventory(seed_inventory_json),
        use_imported_peer_inventory=use_imported_peer_inventory,
        use_imported_seed_inventory=use_imported_seed_inventory,
    )
    queued = queue_assignment_fetch_plan(manifest, fetch_plan)
    snapshot = chunk_download_queue_snapshot()
    return dedent(
        f"""
        Chunk downloads enqueued:
        - catalog_id={catalog_id}
        - version={version}
        - queued_now={len(queued)}
        - fetch_requests={len(fetch_plan.fetch_requests)}
        - queue_tasks={snapshot.stats.task_count}
        - queue_queued={snapshot.stats.queued_count}
        - queue_bytes={snapshot.stats.queued_bytes}
        """
    ).strip()


def handle_model_package_ensure_ready(
    catalog_id: str,
    version: str,
    *,
    start_layer: int,
    end_layer: int,
    device_rank: int,
    world_size: int,
    node_id: str | None,
    peer_inventory_json: str | None,
    seed_inventory_json: str | None,
    use_imported_peer_inventory: bool,
    use_imported_seed_inventory: bool,
    max_tasks: int | None,
) -> str:
    manifest = load_model_package_manifest(catalog_id, version)
    result = ensure_assignment_ready_from_store(
        manifest,
        ModelShardAssignment(
            start_layer=start_layer,
            end_layer=end_layer,
            device_rank=device_rank,
            world_size=world_size,
            node_id=node_id,
        ),
        peer_chunk_inventory=_load_chunk_inventory(peer_inventory_json),
        seed_chunk_inventory=_load_chunk_inventory(seed_inventory_json),
        use_imported_peer_inventory=use_imported_peer_inventory,
        use_imported_seed_inventory=use_imported_seed_inventory,
        max_tasks=max_tasks,
    )
    snapshot = chunk_download_queue_snapshot()
    processed_completed = sum(
        1
        for task in result.processed_tasks
        if task.status == ChunkDownloadTaskStatus.COMPLETED
    )
    processed_failed = sum(
        1
        for task in result.processed_tasks
        if task.status == ChunkDownloadTaskStatus.FAILED
    )
    lines = dedent(
        f"""
        Assignment ensure ready:
        - catalog_id={catalog_id}
        - version={version}
        - layer_range={start_layer}:{end_layer}
        - initial_ready={str(result.initial_plan.ready).lower()}
        - final_ready={str(result.final_plan.ready).lower()}
        - initial_missing_chunks={len(result.initial_plan.coverage.missing_chunk_ids)}
        - final_missing_chunks={len(result.final_plan.coverage.missing_chunk_ids)}
        - initial_fetch_bytes={result.initial_plan.estimated_fetch_bytes}
        - final_fetch_bytes={result.final_plan.estimated_fetch_bytes}
        - queued_now={len(result.queued_tasks)}
        - processed_now={len(result.processed_tasks)}
        - processed_completed={processed_completed}
        - processed_failed={processed_failed}
        - queue_tasks={snapshot.stats.task_count}
        - queue_completed={snapshot.stats.completed_count}
        - queue_failed={snapshot.stats.failed_count}
        """
    ).strip().splitlines()
    insertion_index = lines.index(f"- queue_tasks={snapshot.stats.task_count}")
    lines[insertion_index:insertion_index] = _chunk_task_source_lines(
        "processed",
        result.processed_tasks,
    )
    return "\n".join(lines)


def handle_model_package_cache_all(
    catalog_id: str,
    version: str,
    *,
    node_id: str | None,
    use_imported_peer_inventory: bool,
    use_imported_seed_inventory: bool,
    max_tasks: int | None,
) -> str:
    manifest = load_model_package_manifest(catalog_id, version)
    total_layers = _manifest_total_layer_count(manifest)
    resolved_node_id = str(node_id or "").strip() or "local-model-package-seed"
    result = ensure_assignment_ready_from_store(
        manifest,
        ModelShardAssignment(
            start_layer=0,
            end_layer=total_layers,
            device_rank=0,
            world_size=1,
            node_id=resolved_node_id,
        ),
        use_imported_peer_inventory=use_imported_peer_inventory,
        use_imported_seed_inventory=use_imported_seed_inventory,
        max_tasks=max_tasks,
    )
    inventory_path = save_local_chunk_inventory_payload(
        resolved_node_id,
        source_kind=ChunkInventorySourceKind.LOCAL_CACHE,
    )
    snapshot = chunk_download_queue_snapshot()
    processed_completed = sum(
        1
        for task in result.processed_tasks
        if task.status == ChunkDownloadTaskStatus.COMPLETED
    )
    processed_failed = sum(
        1
        for task in result.processed_tasks
        if task.status == ChunkDownloadTaskStatus.FAILED
    )
    lines = dedent(
        f"""
        Model package full cache:
        - catalog_id={catalog_id}
        - version={version}
        - source_id={resolved_node_id}
        - layer_range=0:{total_layers}
        - initial_ready={str(result.initial_plan.ready).lower()}
        - final_ready={str(result.final_plan.ready).lower()}
        - initial_missing_chunks={len(result.initial_plan.coverage.missing_chunk_ids)}
        - final_missing_chunks={len(result.final_plan.coverage.missing_chunk_ids)}
        - initial_fetch_bytes={result.initial_plan.estimated_fetch_bytes}
        - final_fetch_bytes={result.final_plan.estimated_fetch_bytes}
        - queued_now={len(result.queued_tasks)}
        - processed_now={len(result.processed_tasks)}
        - processed_completed={processed_completed}
        - processed_failed={processed_failed}
        - local_inventory_path={inventory_path}
        - queue_tasks={snapshot.stats.task_count}
        - queue_completed={snapshot.stats.completed_count}
        - queue_failed={snapshot.stats.failed_count}
        """
    ).strip().splitlines()
    insertion_index = lines.index(f"- queue_tasks={snapshot.stats.task_count}")
    lines[insertion_index:insertion_index] = _chunk_task_source_lines(
        "processed",
        result.processed_tasks,
    )
    return "\n".join(lines)


def handle_chunk_download_mark(
    task_id: str,
    status: str,
    *,
    source_kind: str | None,
    source_id: str | None,
    last_error: str | None,
) -> str:
    task = update_chunk_download_task_status(
        task_id,
        status,
        selected_source_kind=source_kind,
        selected_source_id=source_id,
        last_error=last_error,
    )
    if task is None:
        raise ValueError(f"Unknown chunk download task: {task_id}")
    return dedent(
        f"""
        Chunk download updated:
        - task_id={task.task_id}
        - status={task.status}
        - source_kind={task.selected_source_kind or '<none>'}
        - source_id={task.selected_source_id or '<none>'}
        - attempts={task.attempt_count}
        - last_error={task.last_error or '<none>'}
        """
    ).strip()


def handle_chunk_download_run(max_tasks: int | None) -> str:
    processed = execute_chunk_download_queue(max_tasks=max_tasks)
    snapshot = chunk_download_queue_snapshot()
    processed_completed = sum(
        1 for task in processed if task.status == ChunkDownloadTaskStatus.COMPLETED
    )
    processed_failed = sum(
        1 for task in processed if task.status == ChunkDownloadTaskStatus.FAILED
    )
    lines = dedent(
        f"""
        Chunk download run:
        - processed={len(processed)}
        - processed_completed={processed_completed}
        - processed_failed={processed_failed}
        - queue_tasks={snapshot.stats.task_count}
        - queued={snapshot.stats.queued_count}
        - completed={snapshot.stats.completed_count}
        - failed={snapshot.stats.failed_count}
        """
    ).strip().splitlines()
    insertion_index = lines.index(f"- queue_tasks={snapshot.stats.task_count}")
    lines[insertion_index:insertion_index] = _chunk_task_source_lines(
        "processed",
        processed,
    )
    return "\n".join(lines)


def handle_chunk_source_bind(
    source_kind: str,
    source_id: str,
    data_root_path: str,
) -> str:
    saved_path = save_chunk_source_binding(
        source_kind,
        source_id,
        data_root_path=data_root_path,
    )
    return dedent(
        f"""
        Chunk source binding saved:
        - source_kind={source_kind}
        - source_id={source_id}
        - bindings_path={saved_path}
        - data_root_path={Path(data_root_path).expanduser().resolve()}
        """
    ).strip()


def handle_chunk_source_bindings(source_kind: str | None = None) -> str:
    bindings = load_chunk_source_bindings()
    filtered = [
        binding
        for binding in bindings.bindings
        if source_kind is None or binding.source_kind == source_kind
    ]
    if not filtered:
        return "<empty>"
    lines = ["Chunk source bindings:"]
    for binding in filtered:
        lines.extend(
            [
                f"- source_kind={binding.source_kind}",
                f"  source_id={binding.source_id}",
                f"  data_root_path={binding.data_root_path}",
                f"  updated_at={binding.updated_at}",
            ]
        )
    return "\n".join(lines)


def handle_model_package_bind_artifact(
    catalog_id: str,
    version: str,
    artifact_id: str,
    path: str,
) -> str:
    saved_path = save_local_artifact_binding(
        catalog_id,
        version,
        artifact_id=artifact_id,
        local_path=path,
    )
    return dedent(
        f"""
        Local artifact binding saved:
        - catalog_id={catalog_id}
        - version={version}
        - artifact_id={artifact_id}
        - bindings_path={saved_path}
        - local_path={Path(path).expanduser().resolve()}
        """
    ).strip()


def handle_model_package_bindings(catalog_id: str, version: str) -> str:
    bindings = load_local_artifact_bindings(catalog_id, version)
    if not bindings.bindings:
        return "<empty>"
    lines = ["Local artifact bindings:"]
    for binding in bindings.bindings:
        lines.extend(
            [
                f"- artifact_id={binding.artifact_id}",
                f"  local_path={binding.local_path}",
                f"  updated_at={binding.updated_at}",
            ]
        )
    return "\n".join(lines)


def handle_model_package_create_gguf(
    catalog_id: str,
    model_id: str,
    version: str,
    gguf_path: str,
    *,
    n_layers: int,
    package_kind: str,
    chunk_size_policy: str,
    min_chunk_mb: int,
    max_chunk_mb: int,
    target_chunks: int | None,
    source_repo_id: str | None,
    source_revision: str,
    family: str,
    quantization: str,
) -> str:
    manifest = build_gguf_model_package_manifest(
        catalog_id=catalog_id,
        model_id=model_id,
        version=version,
        gguf_path=gguf_path,
        total_layers=n_layers,
        package_kind=package_kind,
        chunk_size_policy=chunk_size_policy,
        min_chunk_bytes=min_chunk_mb * 1024 * 1024,
        max_chunk_bytes=max_chunk_mb * 1024 * 1024,
        target_chunk_count=target_chunks,
        source_repo_id=source_repo_id,
        source_revision=source_revision,
        family=family,
        quantization=quantization,
    )
    saved_path = save_model_package_manifest(manifest)
    save_local_artifact_binding(
        manifest.catalog_id,
        manifest.version,
        artifact_id="gguf-main",
        local_path=gguf_path,
    )
    return dedent(
        f"""
        GGUF model package manifest created:
        - catalog_id={manifest.catalog_id}
        - model_id={manifest.model_id}
        - version={manifest.version}
        - manifest_path={saved_path}
        - file={manifest.preferred_filename or '<unknown>'}
        - total_size_bytes={manifest.total_size_bytes}
        - chunk_count={len(manifest.chunks)}
        - chunk_size_policy={manifest.chunk_size_policy}
        """
    ).strip()


def handle_model_package_create_hf_gguf(
    model_id: str,
    version: str,
    *,
    catalog_id: str | None,
    preferred_filename: str | None,
    n_layers: int | None,
    package_kind: str,
    chunk_size_policy: str,
    min_chunk_mb: int,
    max_chunk_mb: int,
    target_chunks: int | None,
    source_revision: str,
    family: str,
    quantization: str,
    timeout_sec: int,
    cache_chunks: bool,
    pin_chunks: bool,
) -> str:
    manifest = build_hf_gguf_model_package_manifest(
        model_id=model_id,
        version=version,
        catalog_id=catalog_id,
        preferred_filename=preferred_filename,
        total_layers=n_layers,
        package_kind=package_kind,
        chunk_size_policy=chunk_size_policy,
        min_chunk_bytes=min_chunk_mb * 1024 * 1024,
        max_chunk_bytes=max_chunk_mb * 1024 * 1024,
        target_chunk_count=target_chunks,
        source_revision=source_revision,
        family=family,
        quantization=quantization,
        timeout_sec=timeout_sec,
        cache_downloaded_chunks=cache_chunks,
        pin_cached_chunks=pin_chunks,
    )
    saved_path = save_model_package_manifest(manifest)
    return dedent(
        f"""
        Hugging Face GGUF model package manifest created:
        - catalog_id={manifest.catalog_id}
        - model_id={manifest.model_id}
        - version={manifest.version}
        - manifest_path={saved_path}
        - file={manifest.preferred_filename or '<unknown>'}
        - total_size_bytes={manifest.total_size_bytes}
        - chunk_count={len(manifest.chunks)}
        - chunk_size_policy={manifest.chunk_size_policy}
        - cached_chunks={str(cache_chunks).lower()}
        """
    ).strip()


def _render_imported_model_package_manifest(manifest, saved_from: str) -> str:
    return dedent(
        f"""
        Model package manifest imported:
        - source={saved_from}
        - catalog_id={manifest.catalog_id}
        - model_id={manifest.model_id}
        - version={manifest.version}
        - file={manifest.preferred_filename or '<unknown>'}
        - total_size_bytes={manifest.total_size_bytes}
        - chunk_count={len(manifest.chunks)}
        """
    ).strip()


def handle_model_package_import_url(
    manifest_url: str,
    *,
    expected_model_id: str | None,
    expected_preferred_filename: str | None,
    timeout_sec: int,
) -> str:
    manifest = import_model_package_manifest_from_url(
        manifest_url,
        expected_model_id=expected_model_id,
        expected_preferred_filename=expected_preferred_filename,
        timeout_sec=timeout_sec,
    )
    return _render_imported_model_package_manifest(manifest, manifest_url)


def handle_model_package_import_hf(
    model_id: str,
    *,
    preferred_filename: str | None,
    source_revision: str,
    timeout_sec: int,
) -> str:
    manifest = discover_and_import_hf_model_package_manifest(
        model_id,
        preferred_filename=preferred_filename,
        source_revision=source_revision,
        timeout_sec=timeout_sec,
    )
    if manifest is None:
        return f"No CAI model package manifest discovered for {model_id}."
    return _render_imported_model_package_manifest(manifest, f"hf:{model_id}")


def handle_chunk_store() -> str:
    snapshot = chunk_store_snapshot()
    stats = snapshot.stats
    cache_policy = ChunkCachePolicy()
    active_lease_chunks = sum(
        1 for record in snapshot.records if record.lease_status == "active"
    )
    return dedent(
        f"""
        Chunk store:
        - chunks={stats.chunk_count}
        - total_bytes={stats.total_bytes}
        - pinned_chunks={stats.pinned_chunk_count}
        - pinned_bytes={stats.pinned_bytes}
        - hot_chunks={stats.hot_chunk_count}
        - warm_chunks={stats.warm_chunk_count}
        - cold_chunks={stats.cold_chunk_count}
        - active_lease_chunks={active_lease_chunks}
        - max_store_bytes={cache_policy.max_store_bytes}
        - target_store_bytes={cache_policy.target_store_bytes}
        """
    ).strip()


def handle_chunk_store_gc() -> str:
    result = evict_chunks_to_policy_target()
    return dedent(
        f"""
        Chunk store GC:
        - before_bytes={result.before.stats.total_bytes}
        - after_bytes={result.after.stats.total_bytes}
        - evicted_chunks={len(result.evicted_chunk_ids)}
        - evicted_bytes={result.evicted_bytes}
        """
    ).strip()


def handle_chunk_shard_hints(node_id: str, *, prefetch: bool = False) -> str:
    hints = list_recent_shard_hints(node_id)
    lines = [f"Recent shard hints for {node_id}:"]
    if not hints:
        lines.append("- <empty>")
    else:
        for hint in hints:
            lines.append(
                "- "
                + ", ".join(
                    [
                        f"model={hint.model_id}",
                        f"layers={hint.start_layer}:{hint.end_layer}",
                        f"rank={hint.device_rank}/{hint.world_size}",
                        f"use_count={hint.use_count}",
                        f"last_seen={hint.last_seen_at or '<unknown>'}",
                    ]
                )
            )
    if prefetch:
        result = prefetch_recent_shard_hints(node_id)
        lines.extend(
            [
                "",
                "Prefetch:",
                f"- manifests_considered={result.manifests_considered}",
                f"- manifests_prefetched={result.manifests_prefetched}",
                f"- queued_tasks={result.queued_tasks}",
                f"- processed_tasks={result.processed_tasks}",
            ]
        )
    return "\n".join(lines)


def handle_chunk_storage_accounting(node_id: str, *, record: bool = False) -> str:
    accounting_policy = ChunkStorageAccountingPolicy()
    if record:
        result = record_chunk_storage_accounting_snapshot(node_id)
        lines = [
            f"Chunk storage accounting recorded for {node_id}:",
            f"- new_records={len(result.records)}",
            f"- byte_seconds={result.summary.total_byte_seconds}",
            f"- max_interval_seconds={accounting_policy.max_accounting_interval_seconds}",
            f"- min_accounting_seconds={accounting_policy.min_accounting_seconds}",
        ]
        return "\n".join(lines)

    records = list_chunk_storage_accounting_records(node_id)
    total_byte_seconds = sum(item.byte_seconds for item in records)
    lines = [
        f"Chunk storage accounting for {node_id}:",
        f"- records={len(records)}",
        f"- byte_seconds={total_byte_seconds}",
        f"- max_interval_seconds={accounting_policy.max_accounting_interval_seconds}",
        f"- min_accounting_seconds={accounting_policy.min_accounting_seconds}",
    ]
    for item in records[:10]:
        lines.append(
            "- "
            + ", ".join(
                [
                    f"id={item.accounting_id}",
                    f"model={item.catalog_id}@{item.version}",
                    f"chunk={item.chunk_id[:12]}",
                    f"bytes={item.size_bytes}",
                    f"seconds={item.accounted_seconds}",
                    f"byte_seconds={item.byte_seconds}",
                ]
            )
        )
    if len(records) > 10:
        lines.append(f"- ... {len(records) - 10} more")
    return "\n".join(lines)


def handle_launch_check(
    local_state_url: str,
    local_summary_url: str,
    remote_state_url: str | None,
    remote_summary_url: str | None,
) -> tuple[str, bool]:
    report = run_alpha_launch_checks(
        local_state_url=local_state_url,
        local_summary_url=local_summary_url,
        remote_state_url=remote_state_url,
        remote_summary_url=remote_summary_url,
    )
    return render_alpha_launch_report(report), report.ready


def handle_node_config() -> str:
    config = load_or_create_node_config()
    wallet_policy = WalletPolicy()
    lines = [
        _ansi("Node config:", "96"),
        f"- validator_enabled={_bool_status(config.validator_enabled)}",
        f"- validator_state={_validator_state_label(config.validator_state)}",
        f"- validator_wallet_id={config.validator_wallet_id or '<none>'}",
        f"- validator_address={config.validator_address or '<none>'}",
        f"- validator_bond_atomic={config.validator_bond_atomic}",
        f"- validator_static_ip_confirmed={config.validator_static_ip_confirmed}",
        f"- validator_unbonding_started_at={config.validator_unbonding_started_at or '<none>'}",
        f"- validator_unbonding_available_at={config.validator_unbonding_available_at or '<none>'}",
        f"- validator_jailed_at={config.validator_jailed_at or '<none>'}",
        f"- validator_unjail_available_at={config.validator_unjail_available_at or '<none>'}",
        f"- validator_jail_reason={config.validator_jail_reason or '<none>'}",
        f"- validator_last_slash_atomic={config.validator_last_slash_atomic}",
        f"- validator_total_slashed_atomic={config.validator_total_slashed_atomic}",
        f"- validator_ha_enabled={_bool_status(getattr(config, 'validator_ha_enabled', False))}",
        f"- validator_ha_role={getattr(config, 'validator_ha_role', 'standalone')}",
        f"- validator_ha_replica_id={getattr(config, 'validator_ha_replica_id', None) or '<none>'}",
        f"- validator_ha_auto_failover_enabled={_bool_status(getattr(config, 'validator_ha_auto_failover_enabled', True))}",
        f"- validator_ha_lease_seconds={getattr(config, 'validator_ha_lease_seconds', 90)}",
        f"- worker_enabled={_bool_status(config.worker_enabled)}",
        f"- relay_enabled={_bool_status(getattr(config, 'relay_enabled', False))}",
        f"- pq_backend_available={_bool_status(mldsa65_available())}",
        f"- require_post_quantum_wallet_signatures={_bool_status(wallet_policy.require_post_quantum_wallet_signatures)}",
        f"- require_hybrid_peer_payload_signatures={_bool_status(wallet_policy.require_hybrid_peer_payload_signatures)}",
        f"- worker_max_parallel_jobs={config.worker_max_parallel_jobs}",
        f"- worker_max_memory_mb={config.worker_max_memory_mb or '<none>'}",
        _ansi("- worker_allowed_model_ids:", "94"),
    ]
    if config.worker_allowed_model_ids:
        lines.extend(f"  - {model_id}" for model_id in config.worker_allowed_model_ids)
    else:
        lines.append("  - <empty>")
    lines.append(_ansi("- worker_reward_address_by_node_id:", "94"))
    if config.worker_reward_address_by_node_id:
        lines.extend(
            f"  - {node_id} -> {address}"
            for node_id, address in sorted(config.worker_reward_address_by_node_id.items())
        )
    else:
        lines.append("  - <empty>")
    return "\n".join(lines)


def handle_settlements(limit: int) -> str:
    money_policy = MoneyPolicy()
    settlements = list_settlements()[:limit]
    if not settlements:
        return "Settlements:\n- <empty>"

    lines = ["Settlements:"]
    for item in settlements:
        lines.append(
            "- "
            + ", ".join(
                [
                    f"id={item.settlement_id}",
                    f"at={item.created_at}",
                    f"wallet={item.source_wallet_address}",
                    f"funding_source={item.funding_source}",
                    f"compute_cost={atomic_to_coins(item.compute_cost_atomic, money_policy)}",
                    f"tx_fee={atomic_to_coins(item.tx_fee_atomic, money_policy)}",
                    f"settlement_fee={atomic_to_coins(item.settlement_fee_atomic, money_policy)}",
                    f"ai_development_fee={atomic_to_coins(item.ai_development_fee_atomic, money_policy)}",
                    f"worker_reward={atomic_to_coins(item.worker_reward_atomic, money_policy)}",
                    f"committee_size={len(item.committee_validator_ids or [])}",
                    f"committee_mode={item.committee_selection_mode}",
                    f"accepted_attestations={item.accepted_attestations}",
                    f"rejected_attestations={item.rejected_attestations}",
                    f"accepted_bond={atomic_to_coins(item.accepted_bond_atomic, money_policy)}",
                    f"rejected_bond={atomic_to_coins(item.rejected_bond_atomic, money_policy)}",
                    f"quorum_bond={atomic_to_coins(item.committee_quorum_bond_atomic, money_policy)}",
                    f"status={item.status}",
                    f"note={item.note or '<none>'}",
                ]
            )
        )
    return "\n".join(lines)


def handle_validator_set() -> str:
    money_policy = MoneyPolicy()
    committee = build_validator_committee_snapshot()
    records = list_validator_records()
    lines = ["Validator set:"]
    lines.append(f"- validators={len(records)}")
    lines.append(
        f"- bonded_total={atomic_to_coins(committee.total_bonded_atomic, money_policy)}"
    )
    lines.append(
        f"- quorum_bond={atomic_to_coins(committee.quorum_bond_atomic, money_policy)}"
    )
    if not records:
        lines.append("- <empty>")
        return "\n".join(lines)

    for item in records:
        lines.append(
            "- "
            + ", ".join(
                [
                    f"id={item.validator_id}",
                    f"state={item.state}",
                    f"bond={atomic_to_coins(item.bonded_atomic, money_policy)}",
                    f"static_ip_confirmed={item.static_ip_confirmed}",
                    f"node_id={item.current_node_id or '<none>'}",
                    f"ha_enabled={_bool_status(getattr(item, 'ha_enabled', False))}",
                    f"active_replica={getattr(item, 'active_replica_node_id', None) or '<none>'}",
                    f"active_lease_until={getattr(item, 'active_replica_lease_until', None) or '<none>'}",
                    f"replicas={','.join(getattr(item, 'replica_node_ids', []) or []) or '<none>'}",
                    f"api_host={item.advertised_api_host or '<none>'}",
                    f"data_host={item.advertised_data_host or '<none>'}",
                    f"last_slash={atomic_to_coins(item.last_slash_atomic, money_policy)}",
                    f"total_slashed={atomic_to_coins(item.total_slashed_atomic, money_policy)}",
                ]
            )
        )
    return "\n".join(lines)


def handle_validator_set_sync(cai_url: str) -> str:
    state_url = cai_url.rstrip("/") + "/state"
    with urlopen(state_url, timeout=5) as response:
        state_payload = json.loads(response.read().decode("utf-8"))
    result = sync_validator_set_from_cai_peers(
        state_payload=state_payload,
        cai_url=cai_url.rstrip("/"),
    )
    lines = ["Validator set sync:"]
    lines.append(f"- attempted_peers={result.attempted_peers}")
    lines.append(f"- successful_peers={result.successful_peers}")
    lines.append(f"- failed_peers={result.failed_peers}")
    lines.append(f"- imported_records={result.imported_records}")
    if result.peer_urls:
        lines.extend(f"  - {item}" for item in result.peer_urls)
    else:
        lines.append("  - <no reachable peer validator endpoints discovered>")
    if result.peer_errors:
        lines.append("- peer_errors:")
        lines.extend(
            f"  - {item['peerUrl']}: {item['errorType']}: {item['message']}"
            for item in result.peer_errors
        )
    return "\n".join(lines)


def handle_job_list() -> str:
    items = list_job_intents()
    if not items:
        return "Job intents:\n- <empty>"

    money_policy = MoneyPolicy()
    lines = ["Job intents:"]
    for item in items:
        lines.append(
            "- "
            + ", ".join(
                [
                    f"id={item.job_id}",
                    f"at={item.created_at}",
                    f"model={item.model_id}",
                    f"status={item.status}",
                    f"payment={item.payment_preference}",
                    f"compute_cost={atomic_to_coins(item.requested_compute_cost_atomic, money_policy)}",
                    f"pricing_mode={item.pricing_mode}",
                    f"cai_url={item.cai_url}",
                    f"receipt_id={item.receipt_id or '<none>'}",
                    f"settlement_id={item.settlement_id or '<none>'}",
                    f"error={item.last_error or '<none>'}",
                ]
            )
        )
    return "\n".join(lines)


def handle_receipts(limit: int) -> str:
    items = list_execution_receipts()[:limit]
    if not items:
        return "Execution receipts:\n- <empty>"

    money_policy = MoneyPolicy()
    lines = ["Execution receipts:"]
    for item in items:
        output_preview = item.output_text.replace("\n", " ").strip()
        if len(output_preview) > 80:
            output_preview = output_preview[:77] + "..."
        payout_preview = (
            "; ".join(
                [
                    f"{payout['node_id']}={atomic_to_coins(int(payout['reward_atomic']), money_policy)}"
                    for payout in item.worker_payouts
                ]
            )
            if item.worker_payouts
            else "<none>"
        )
        lines.append(
            "- "
            + ", ".join(
                [
                    f"id={item.receipt_id}",
                    f"at={item.created_at}",
                    f"job_id={item.job_id}",
                    f"model={item.model_id}",
                    f"execution_model={item.execution_model_id}",
                    f"instance_id={item.instance_id or '<none>'}",
                    f"pricing_mode={item.pricing_mode}",
                    f"pricing_basis={item.pricing_basis}",
                    (
                        f"usage={item.prompt_tokens}/{item.completion_tokens}/{item.total_tokens}"
                        if item.total_tokens is not None
                        else "usage=<none>"
                    ),
                    (
                        "compute_cost="
                        f"{atomic_to_coins(item.actual_compute_cost_atomic, money_policy)}"
                        if item.actual_compute_cost_atomic is not None
                        else "compute_cost=<none>"
                    ),
                    (
                        "reserved_compute="
                        f"{atomic_to_coins(item.reserved_compute_cost_atomic, money_policy)}"
                        if item.reserved_compute_cost_atomic is not None
                        else "reserved_compute=<none>"
                    ),
                    f"finish_reason={item.finish_reason or '<none>'}",
                    f"worker_payouts={payout_preview}",
                    f"output={output_preview or '<empty>'}",
                ]
            )
        )
    return "\n".join(lines)


def _bool_yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _preview_text(value: str | None, *, limit: int = 120) -> str:
    preview = str(value or "").replace("\n", " ").strip()
    if not preview:
        return "<empty>"
    if len(preview) > limit:
        return preview[: limit - 3] + "..."
    return preview


def _join_compact(values) -> str:
    normalized = [str(item).strip() for item in values if str(item).strip()]
    return ", ".join(normalized) if normalized else "<none>"


def _format_checked_direct_links(items) -> str:
    if not isinstance(items, list) or not items:
        return "<none>"

    rendered: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        source_node_id = str(item.get("sourceNodeId") or "").strip()
        sink_node_id = str(item.get("sinkNodeId") or "").strip()
        if not source_node_id or not sink_node_id:
            continue
        connector = "<->" if bool(item.get("bidirectional")) else "->"
        rendered.append(f"{source_node_id}{connector}{sink_node_id}")
    return ", ".join(rendered) if rendered else "<none>"


def _format_checked_overlay_links(items) -> str:
    if not isinstance(items, list) or not items:
        return "<none>"

    rendered: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        source_node_id = str(item.get("sourceNodeId") or "").strip()
        sink_node_id = str(item.get("sinkNodeId") or "").strip()
        if not source_node_id or not sink_node_id:
            continue
        rendered.append(f"{source_node_id}~{sink_node_id}")
    return ", ".join(rendered) if rendered else "<none>"


def _format_checked_relay_routes(items) -> str:
    if not isinstance(items, list) or not items:
        return "<none>"

    rendered: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        source_node_id = str(item.get("sourceNodeId") or "").strip()
        transit_node_id = str(item.get("transitNodeId") or "").strip()
        sink_node_id = str(item.get("sinkNodeId") or "").strip()
        if not source_node_id or not transit_node_id or not sink_node_id:
            continue
        source_segment = str(item.get("sourceSegmentType") or "?").strip() or "?"
        sink_segment = str(item.get("sinkSegmentType") or "?").strip() or "?"
        rendered.append(
            f"{source_node_id}-[{source_segment}]->{transit_node_id}-[{sink_segment}]->{sink_node_id}"
        )
    return ", ".join(rendered) if rendered else "<none>"


def _resolve_receipt_for_verification(
    *, receipt_id: str | None = None, job_id: str | None = None
):
    receipts = list_execution_receipts()
    jobs = list_job_intents()

    if receipt_id is not None:
        receipt = next((item for item in receipts if item.receipt_id == receipt_id), None)
        if receipt is None:
            raise ValueError(f"Execution receipt {receipt_id} was not found.")
        job = next(
            (
                item
                for item in jobs
                if item.receipt_id == receipt.receipt_id or item.job_id == receipt.job_id
            ),
            None,
        )
        return receipt, job

    if job_id is not None:
        job = next((item for item in jobs if item.job_id == job_id), None)
        if job is None:
            raise ValueError(f"Job {job_id} was not found.")
        if not job.receipt_id:
            raise ValueError(f"Job {job_id} does not have an execution receipt yet.")
        receipt = next((item for item in receipts if item.receipt_id == job.receipt_id), None)
        if receipt is None:
            raise ValueError(
                f"Execution receipt {job.receipt_id} for job {job_id} was not found."
            )
        return receipt, job

    if not receipts:
        return None, None

    latest_receipt = receipts[0]
    latest_job = next(
        (
            item
            for item in jobs
            if item.receipt_id == latest_receipt.receipt_id
            or item.job_id == latest_receipt.job_id
        ),
        None,
    )
    return latest_receipt, latest_job


def handle_job_verify(
    *, receipt_id: str | None = None, job_id: str | None = None
) -> str:
    receipt, job = _resolve_receipt_for_verification(
        receipt_id=receipt_id,
        job_id=job_id,
    )
    if receipt is None:
        return "Execution verification:\n- <empty>"

    network_audit = receipt.network_audit or {}
    payout_records = list_worker_payouts(receipt_id=receipt.receipt_id)
    payout_status_counts = Counter(item.status for item in payout_records)
    payout_status_preview = (
        ", ".join(
            f"{status}={count}" for status, count in sorted(payout_status_counts.items())
        )
        if payout_status_counts
        else "<none>"
    )
    payout_record_node_ids = [item.node_id for item in payout_records]
    receipt_reward_atomic = sum(
        int(payout.get("reward_atomic") or 0) for payout in (receipt.worker_payouts or [])
        if isinstance(payout, dict)
    )
    payout_record_reward_atomic = sum(item.reward_atomic for item in payout_records)
    payout_count_matches_receipt = len(receipt.worker_payouts or []) == len(payout_records)
    payout_sum_matches_receipt = receipt_reward_atomic == payout_record_reward_atomic

    settlement_id_candidates: list[str] = []
    if job is not None and job.settlement_id:
        settlement_id_candidates.append(job.settlement_id)
    settlement_id_candidates.extend(
        item.settlement_id for item in payout_records if item.settlement_id
    )
    settlement_id_candidates = list(dict.fromkeys(settlement_id_candidates))
    settlement_id = (
        settlement_id_candidates[0] if len(settlement_id_candidates) == 1 else None
    )
    settlement = next(
        (
            item
            for item in list_settlements()
            if settlement_id is not None and item.settlement_id == settlement_id
        ),
        None,
    )
    settlement_reward_matches_receipt = (
        settlement is not None and settlement.worker_reward_atomic == receipt_reward_atomic
    )
    settlement_reward_matches_payout_records = (
        settlement is not None
        and settlement.worker_reward_atomic == payout_record_reward_atomic
    )
    reward_accounting_consistent = (
        payout_count_matches_receipt
        and payout_sum_matches_receipt
        and (
            settlement is None
            or (
                settlement_reward_matches_receipt
                and settlement_reward_matches_payout_records
            )
        )
    )

    response_received = bool(_preview_text(receipt.output_text) != "<empty>")
    participant_node_ids = network_audit.get("participantNodeIds") or []
    participant_eligibility = network_audit.get("participantEligibility")
    if not isinstance(participant_eligibility, dict):
        participant_eligibility = {}
    execution_strategy = network_audit.get("llamaCppExecutionStrategy")
    if not isinstance(execution_strategy, dict):
        execution_strategy = {}
    decentralized_execution = bool(network_audit.get("decentralizedExecution"))
    direct_paths_checked = _format_checked_direct_links(
        network_audit.get("checkedDirectSocketLinks")
    )
    overlay_paths_checked = _format_checked_overlay_links(
        network_audit.get("checkedOverlayLinks")
    )
    relay_paths_checked = _format_checked_relay_routes(
        network_audit.get("checkedRelayRoutes")
    )

    lines = [
        "Execution verification:",
        f"- receipt_id={receipt.receipt_id}",
        f"- job_id={receipt.job_id}",
        f"- settlement_id={settlement_id or '<none>'}",
        f"- at={receipt.created_at}",
        f"- model={receipt.model_id}",
        f"- execution_model={receipt.execution_model_id}",
        f"- instance_id={receipt.instance_id or '<none>'}",
        f"- pricing_mode={receipt.pricing_mode}",
        f"- pricing_basis={receipt.pricing_basis}",
        (
            f"- prompt_tokens={receipt.prompt_tokens}"
            if receipt.prompt_tokens is not None
            else "- prompt_tokens=<none>"
        ),
        (
            f"- completion_tokens={receipt.completion_tokens}"
            if receipt.completion_tokens is not None
            else "- completion_tokens=<none>"
        ),
        (
            f"- total_tokens={receipt.total_tokens}"
            if receipt.total_tokens is not None
            else "- total_tokens=<none>"
        ),
        (
            "- reserved_compute_cost="
            f"{atomic_to_coins(receipt.reserved_compute_cost_atomic, MoneyPolicy())}"
            if receipt.reserved_compute_cost_atomic is not None
            else "- reserved_compute_cost=<none>"
        ),
        (
            "- actual_compute_cost="
            f"{atomic_to_coins(receipt.actual_compute_cost_atomic, MoneyPolicy())}"
            if receipt.actual_compute_cost_atomic is not None
            else "- actual_compute_cost=<none>"
        ),
        (
            "- reservation_surplus="
            f"{atomic_to_coins(receipt.reservation_surplus_atomic, MoneyPolicy())}"
            if receipt.reservation_surplus_atomic
            else "- reservation_surplus=0.00000000"
        ),
        f"- usage_priced={_bool_yes_no(bool(receipt.usage_priced))}",
        f"- finish_reason={receipt.finish_reason or '<none>'}",
        f"- response_received={_bool_yes_no(response_received)}",
        f"- output={_preview_text(receipt.output_text)}",
        f"- participant_count={network_audit.get('participantCount', len(receipt.worker_payouts or []))}",
        f"- participant_nodes={_join_compact(participant_node_ids)}",
        f"- participant_eligibility_can_settle={_bool_yes_no(bool(participant_eligibility.get('canSettle')))}",
        f"- participant_eligibility_route_reachable={participant_eligibility.get('routeReachable', '<none>')}",
        f"- participant_eligibility_fatal_reasons={_join_compact(participant_eligibility.get('fatalReasons') or [])}",
        f"- participant_eligibility_warnings={_join_compact(participant_eligibility.get('warnings') or [])}",
        f"- transport_mode={network_audit.get('transportMode') or '<none>'}",
        f"- decentralized_execution={_bool_yes_no(decentralized_execution)}",
        f"- llama_cpp_execution_mode={execution_strategy.get('executionMode') or '<none>'}",
        f"- cai_owned_transport_executed={_bool_yes_no(bool(network_audit.get('caiOwnedTransportExecuted')))}",
        f"- cai_owned_transport_proof_error={network_audit.get('caiOwnedTransportProofError') or '<none>'}",
        f"- coordinator_direct_fanout={_bool_yes_no(bool(network_audit.get('coordinatorDirectFanout')))}",
        f"- strongly_connected_direct_graph={_bool_yes_no(bool(network_audit.get('stronglyConnectedDirectGraph')))}",
        f"- direct_socket_links={network_audit.get('directSocketLinkCount', 0)}",
        f"- direct_bidirectional_links={network_audit.get('directBidirectionalLinkCount', 0)}",
        f"- overlay_links={network_audit.get('overlayLinkCount', 0)}",
        f"- direct_paths_checked={direct_paths_checked}",
        f"- overlay_paths_checked={overlay_paths_checked}",
        f"- relay_paths_checked={relay_paths_checked}",
        f"- coordinator_candidates={_join_compact(network_audit.get('coordinatorCandidateNodeIds') or [])}",
        f"- relay_coordinator_candidates={_join_compact(network_audit.get('relayCoordinatorCandidateNodeIds') or [])}",
        f"- relay_capable_nodes={_join_compact(network_audit.get('relayCapableNodeIds') or [])}",
        f"- relay_transit_candidates={_join_compact(network_audit.get('relayTransitCandidateNodeIds') or [])}",
        f"- relay_route_candidate_count={network_audit.get('relayRouteCandidateCount', 0)}",
        f"- relay_hops_used={_bool_yes_no(bool(network_audit.get('relayHopsUsed')))}",
        f"- reward_sum_receipt={atomic_to_coins(receipt_reward_atomic, MoneyPolicy())}",
        f"- reward_sum_payout_records={atomic_to_coins(payout_record_reward_atomic, MoneyPolicy())}",
        f"- payout_record_nodes={_join_compact(payout_record_node_ids)}",
        f"- payout_statuses={payout_status_preview}",
        f"- payout_count_matches_receipt={_bool_yes_no(payout_count_matches_receipt)}",
        f"- payout_sum_matches_receipt={_bool_yes_no(payout_sum_matches_receipt)}",
        f"- reward_accounting_consistent={_bool_yes_no(reward_accounting_consistent)}",
    ]

    if settlement is None:
        lines.extend(
            [
                "- settlement_found=no",
                "- settlement_status=<none>",
                "- settlement_worker_reward=<none>",
                "- settlement_ai_development_fee=<none>",
                "- settlement_reward_matches_receipt=<none>",
                "- settlement_reward_matches_payout_records=<none>",
                "- settlement_attestations=<none>",
            ]
        )
    else:
        lines.extend(
            [
                "- settlement_found=yes",
                f"- settlement_status={settlement.status}",
                f"- settlement_funding_source={settlement.funding_source}",
                f"- settlement_worker_reward={atomic_to_coins(settlement.worker_reward_atomic, MoneyPolicy())}",
                f"- settlement_ai_development_fee={atomic_to_coins(settlement.ai_development_fee_atomic, MoneyPolicy())}",
                f"- settlement_ai_development_wallet={settlement.ai_development_address or '<none>'}",
                f"- settlement_reward_matches_receipt={_bool_yes_no(settlement_reward_matches_receipt)}",
                f"- settlement_reward_matches_payout_records={_bool_yes_no(settlement_reward_matches_payout_records)}",
                (
                    "- settlement_attestations="
                    f"{settlement.accepted_attestations} accepted / "
                    f"{settlement.rejected_attestations} rejected"
                ),
            ]
        )

    return "\n".join(lines)


def handle_attestations(limit: int) -> str:
    attestations = list_attestations(limit=limit)
    if not attestations:
        return "Attestations:\n- <empty>"

    lines = ["Attestations:"]
    for item in attestations:
        lines.append(
            "- "
            + ", ".join(
                [
                    f"id={item.attestation_id}",
                    f"at={item.created_at}",
                    f"settlement_id={item.settlement_id}",
                    f"validator_id={item.validator_id}",
                    f"accepted={item.accepted}",
                    f"note={item.note or '<none>'}",
                ]
            )
        )
    return "\n".join(lines)


def handle_worker_payouts(limit: int) -> str:
    money_policy = MoneyPolicy()
    payouts = list_worker_payouts(limit=limit)
    if not payouts:
        return "Worker payouts:\n- <empty>"

    lines = ["Worker payouts:"]
    for item in payouts:
        lines.append(
            "- "
            + ", ".join(
                [
                    f"id={item.payout_id}",
                    f"at={item.created_at}",
                    f"settlement_id={item.settlement_id}",
                    f"receipt_id={item.receipt_id}",
                    f"node_id={item.node_id}",
                    f"runner_id={item.runner_id or '<none>'}",
                    f"layer_count={item.layer_count}",
                    f"share_bps={item.share_bps}",
                    f"reward={atomic_to_coins(item.reward_atomic, money_policy)}",
                    f"recipient_address={item.recipient_address or '<none>'}",
                    f"credited_wallet_id={item.credited_wallet_id or '<none>'}",
                    f"status={item.status}",
                ]
            )
        )
    return "\n".join(lines)


def handle_validator_evidence(limit: int, validator_id: str | None = None) -> str:
    money_policy = MoneyPolicy()
    evidence_items = list_validator_evidence(limit=limit, validator_id=validator_id)
    if not evidence_items:
        return "Validator evidence:\n- <empty>"

    lines = ["Validator evidence:"]
    for item in evidence_items:
        lines.append(
            "- "
            + ", ".join(
                [
                    f"id={item.evidence_id}",
                    f"at={item.created_at}",
                    f"validator_id={item.validator_id}",
                    f"reporter={item.reporter_validator_id or '<unknown>'}",
                    f"type={item.evidence_type}",
                    f"settlement_id={item.settlement_id or '<none>'}",
                    f"attestation_id={item.attestation_id or '<none>'}",
                    f"conflicting_attestation_id={item.conflicting_attestation_id or '<none>'}",
                    f"slash={atomic_to_coins(item.slash_atomic, money_policy)}",
                    f"jailed={item.jailed}",
                    f"source={item.source}",
                    f"applied={item.applied_to_registry}",
                    f"note={item.note or '<none>'}",
                ]
            )
        )
    return "\n".join(lines)


def handle_validator_evidence_cases(limit: int, validator_id: str | None = None) -> str:
    money_policy = MoneyPolicy()
    cases = list_validator_evidence_cases(limit=limit, validator_id=validator_id)
    if not cases:
        return "Validator evidence cases:\n- <empty>"

    lines = ["Validator evidence cases:"]
    for item in cases:
        lines.append(
            "- "
            + ", ".join(
                [
                    f"case_id={item.case_id}",
                    f"validator_id={item.validator_id}",
                    f"type={item.evidence_type}",
                    f"settlement_id={item.settlement_id or '<none>'}",
                    f"slash={atomic_to_coins(item.slash_atomic, money_policy)}",
                    f"jailed={item.jailed}",
                    f"support_mode={item.support_mode}",
                    f"support_scope={item.support_scope}",
                    f"support={item.supporting_validator_count if item.support_mode == 'validator' else item.supporting_sources_count}/{item.required_sources}",
                    f"evidence_quorum_reached={item.evidence_quorum_reached}",
                    f"penalty_support={item.penalty_attestation_count}/{item.penalty_attestation_required}",
                    f"status={item.status}",
                    f"quorum_reached={item.quorum_reached}",
                    f"finalized_at={item.finalized_at or '<none>'}",
                    f"applied_at={item.applied_at or '<none>'}",
                    f"evidence_count={item.evidence_count}",
                    f"applied={item.applied_to_registry}",
                ]
            )
        )
    return "\n".join(lines)


def handle_validator_evidence_sync(cai_url: str) -> str:
    state_url = cai_url.rstrip("/") + "/state"
    with urlopen(state_url, timeout=5) as response:
        state_payload = json.loads(response.read().decode("utf-8"))
    result = sync_validator_evidence_from_cai_peers(
        state_payload=state_payload,
        cai_url=cai_url.rstrip("/"),
    )
    lines = ["Validator evidence sync:"]
    lines.append(f"- attempted_peers={result.attempted_peers}")
    lines.append(f"- successful_peers={result.successful_peers}")
    lines.append(f"- failed_peers={result.failed_peers}")
    lines.append(f"- imported_records={result.imported_records}")
    lines.append(f"- applied_records={result.applied_records}")
    if result.peer_urls:
        lines.extend(f"  - {item}" for item in result.peer_urls)
    else:
        lines.append("  - <no reachable peer evidence endpoints discovered>")
    if result.peer_errors:
        lines.append("- peer_errors:")
        lines.extend(
            f"  - {item['peerUrl']}: {item['errorType']}: {item['message']}"
            for item in result.peer_errors
        )
    if result.validator_set_sync_error:
        item = result.validator_set_sync_error
        lines.append(
            f"- validator_set_sync_error={item['errorType']}: {item['message']}"
        )
    if result.penalty_attestation_sync_error:
        item = result.penalty_attestation_sync_error
        lines.append(
            f"- penalty_attestation_sync_error={item['errorType']}: {item['message']}"
        )
    return "\n".join(lines)


def handle_worker_payout_reconcile() -> str:
    money_policy = MoneyPolicy()
    reconciled = reconcile_worker_payouts()
    if not reconciled:
        return "Worker payout reconciliation:\n- <no updates>"

    lines = ["Worker payout reconciliation:"]
    for item in reconciled:
        lines.append(
            "- "
            + ", ".join(
                [
                    f"id={item.payout_id}",
                    f"node_id={item.node_id}",
                    f"reward={atomic_to_coins(item.reward_atomic, money_policy)}",
                    f"recipient_address={item.recipient_address or '<none>'}",
                    f"credited_wallet_id={item.credited_wallet_id or '<none>'}",
                    f"status={item.status}",
                ]
            )
        )
    return "\n".join(lines)


def handle_interface_state(
    *,
    state_url: str,
    quote_amount: str | None,
    quote_prompt: str | None,
    quote_model: str | None,
    cai_url: str | None,
    payment: str,
    as_json: bool,
) -> str:
    snapshot = build_interface_snapshot(
        state_url=state_url,
        quote_amount_coins=quote_amount,
        quote_prompt=quote_prompt,
        quote_model_id=quote_model,
        cai_url=cai_url,
        payment_preference=PaymentPreference(payment),
    )
    if as_json:
        return json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2)

    wallet = snapshot.wallet
    network = snapshot.network
    validator = snapshot.validator
    worker = snapshot.worker
    compute = snapshot.compute
    return dedent(
        f"""
        Interface snapshot:
        [wallet]
        - has_active_wallet={wallet.has_active_wallet}
        - wallet_name={wallet.wallet_name or '<none>'}
        - address={wallet.address or '<none>'}
        - balance={wallet.balance_coins or '<none>'}
        - unlocked={wallet.unlocked}
        - history_entries={wallet.history_entries}

        [network]
        - state_url={network.state_url}
        - reachable={network.reachable}
        - namespace={network.namespace}
        - bootstrap_peers={network.bootstrap_peers}
        - peer_book_entries={network.peer_book_entries}
        - overlay_peers={network.overlay_peers}
        - topology_nodes={network.topology_nodes}
        - topology_connections={network.topology_connections}
        - node_system_entries={network.node_system_entries}
        - error={network.error or '<none>'}

        [validator]
        - validator_enabled={validator.validator_enabled}
        - validator_fee_pool={validator.validator_fee_pool_coins}
        - project_treasury_balance={validator.project_treasury_balance_coins}
        - settlement_count={validator.settlement_count}
        - attestation_count={validator.attestation_count}

        [worker]
        - worker_enabled={worker.worker_enabled}
        - network_default_model_id={worker.network_default_model_id}
        - private_model_minimum_shards={worker.private_model_minimum_shards}
        - allowed_model_ids={', '.join(worker.allowed_model_ids) if worker.allowed_model_ids else '<empty>'}
        - max_parallel_jobs={worker.max_parallel_jobs}
        - max_memory_mb={worker.max_memory_mb or '<none>'}
        - reward_bindings={worker.reward_bindings}
        - reserve_balance={worker.reserve_balance_coins}
        - worker_paid_out={worker.worker_paid_out_coins}
        - local_worker_earnings={worker.local_worker_earnings_coins}
        - external_payout_records={worker.external_payout_records}
        - unbound_payout_records={worker.unbound_payout_records}
        - settlement_records={worker.settlement_records}
        - payout_records={worker.payout_records}

        [compute]
        - pricing_mode={compute.pricing_mode}
        - payment_preference={compute.payment_preference}
        - quote_available={compute.quote_available}
        - quote_reason={compute.quote_reason}
        - funding_source={compute.funding_source or '<none>'}
        - compute_cost={compute.compute_cost_coins or '<none>'}
        - tx_fee={compute.tx_fee_coins or '<none>'}
        - settlement_fee={compute.settlement_fee_coins or '<none>'}
        - ai_development_fee={compute.ai_development_fee_coins or '<none>'}
        - worker_reward={compute.worker_reward_coins or '<none>'}
        - automatic_price_reason={compute.automatic_price_reason or '<none>'}
        - job_intents={compute.job_intents}
        - execution_receipts={compute.execution_receipts}
        """
    ).strip()


def handle_wallet_list() -> str:
    money_policy = MoneyPolicy()
    ensure_chain_genesis(money_policy=money_policy)
    wallets = list_wallets()
    session = load_session()
    if not wallets:
        return f"Wallets:\n- <empty>\n- data_root={data_root()}\n- wallets_file={wallets_file_path()}"

    lines = [
        "Wallets:",
        f"- data_root={data_root()}",
        f"- wallets_file={wallets_file_path()}",
        f"- balance_source={wallet_balance_source()}",
    ]
    for wallet in wallets:
        tags: list[str] = []
        if session.active_wallet_id == wallet.wallet_id:
            tags.append("active")
        if session.unlocked_wallet_id == wallet.wallet_id:
            tags.append("unlocked")
        tags_text = f" [{' '.join(tags)}]" if tags else ""
        balance_atomic = wallet_chain_balance_or_local_atomic(wallet)
        lines.append(
            f"- {wallet.name}{tags_text}: address={wallet.address} "
            f"balance={atomic_to_coins(balance_atomic, money_policy)} "
            f"local_cached_balance={atomic_to_coins(wallet.spendable_balance_atomic, money_policy)}"
        )
    return "\n".join(lines)


def handle_wallet_status() -> str:
    money_policy = MoneyPolicy()
    wallet_policy = WalletPolicy()
    ensure_chain_genesis(money_policy=money_policy)
    session = load_session()
    wallet = get_active_wallet()
    if wallet is None:
        return dedent(
            f"""
            Wallet status:
            - active_wallet=<none>
            - unlocked_wallet_id={session.unlocked_wallet_id or '<none>'}
            - pq_backend_available={str(mldsa65_available()).lower()}
            - require_post_quantum_wallet_signatures={str(wallet_policy.require_post_quantum_wallet_signatures).lower()}
            - require_hybrid_peer_payload_signatures={str(wallet_policy.require_hybrid_peer_payload_signatures).lower()}
            - data_root={data_root()}
            """
        ).strip()

    is_unlocked = session.unlocked_wallet_id == wallet.wallet_id
    balance_atomic = wallet_chain_balance_or_local_atomic(wallet)
    return dedent(
        f"""
        Wallet status:
        - active_wallet_name={wallet.name}
        - active_wallet_id={wallet.wallet_id}
        - address={wallet.address}
        - signing_scheme={wallet.signing_scheme or '<none>'}
        - address_scheme={wallet.address_scheme or '<none>'}
        - pq_signing_scheme={wallet.pq_signing_scheme or '<none>'}
        - pq_public_key_available={str(bool(wallet.pq_public_key_b64)).lower()}
        - pq_backend_available={str(mldsa65_available()).lower()}
        - require_post_quantum_wallet_signatures={str(wallet_policy.require_post_quantum_wallet_signatures).lower()}
        - require_hybrid_peer_payload_signatures={str(wallet_policy.require_hybrid_peer_payload_signatures).lower()}
        - balance={atomic_to_coins(balance_atomic, money_policy)}
        - balance_source={wallet_balance_source()}
        - local_cached_balance={atomic_to_coins(wallet.spendable_balance_atomic, money_policy)}
        - unlocked={is_unlocked}
        - unlocked_at={session.unlocked_at or '<none>'}
        - data_root={data_root()}
        """
    ).strip()


def handle_wallet_history(wallet_selector: str | None, limit: int) -> str:
    money_policy = MoneyPolicy()
    if wallet_selector is None:
        wallet = get_active_wallet()
        if wallet is None:
            raise SystemExit("Active wallet is not set.")
    else:
        wallet = resolve_wallet(wallet_selector, list_wallets())
        if wallet is None:
            raise SystemExit(f"Wallet '{wallet_selector}' not found.")

    entries = list_journal_entries(wallet_id=wallet.wallet_id, limit=limit)
    if not entries:
        return (
            "Wallet history:\n"
            f"- wallet={wallet.name}\n"
            "- <empty>"
        )

    lines = [
        "Wallet history:",
        f"- wallet={wallet.name}",
        f"- address={wallet.address}",
    ]
    for entry in entries:
        parts = [f"type={entry.event_type}", f"at={entry.created_at}"]
        if entry.amount_atomic is not None:
            parts.append(
                f"amount={atomic_to_coins(entry.amount_atomic, money_policy)}"
            )
        if entry.counterparty_address is not None:
            parts.append(f"counterparty={normalize_address(entry.counterparty_address)}")
        if entry.funding_source is not None:
            parts.append(f"funding_source={entry.funding_source}")
        if entry.compute_cost_atomic is not None:
            parts.append(
                f"compute_cost={atomic_to_coins(entry.compute_cost_atomic, money_policy)}"
            )
        if entry.tx_fee_atomic is not None:
            parts.append(
                f"tx_fee={atomic_to_coins(entry.tx_fee_atomic, money_policy)}"
            )
        if entry.settlement_fee_atomic is not None:
            parts.append(
                f"settlement_fee={atomic_to_coins(entry.settlement_fee_atomic, money_policy)}"
            )
        if entry.worker_reward_atomic is not None:
            parts.append(
                f"worker_reward={atomic_to_coins(entry.worker_reward_atomic, money_policy)}"
            )
        if entry.note:
            parts.append(f"note={entry.note}")
        lines.append("- " + ", ".join(parts))
    return "\n".join(lines)


def _require_active_wallet() -> tuple:
    wallet = get_active_wallet()
    if wallet is None:
        raise SystemExit("Active wallet is not set. Create or select a wallet first.")
    session = load_session()
    return wallet, session


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _resolve_compute_quote_for_cli(
    *,
    amount: str | None,
    prompt: str | None,
    model: str | None,
    cai_url: str | None,
):
    money_policy = MoneyPolicy()
    ensure_chain_genesis(money_policy=money_policy)
    ledger = chain_backed_ledger_snapshot(
        load_or_create_ledger(money_policy),
        money_policy=money_policy,
    )
    wallet, _ = _require_active_wallet()
    wallet.spendable_balance_atomic = chain_balance_atomic(wallet.address)
    network_model_policy = NetworkModelPolicy()
    resolved_model = model or network_model_policy.network_default_model_id
    try:
        resolved_price = resolve_compute_price(
            compute_amount_coins=amount,
            prompt=prompt,
            model_id=resolved_model,
            cai_url=cai_url,
            ledger=ledger,
            money_policy=money_policy,
            network_model_policy=network_model_policy,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return resolved_price, money_policy, ledger, wallet


def _handle_job_quote(
    amount: str | None,
    payment: str,
    prompt: str | None,
    model: str | None,
    cai_url: str | None,
) -> str:
    resolved_price, money_policy, ledger, wallet = _resolve_compute_quote_for_cli(
        amount=amount,
        prompt=prompt,
        model=model,
        cai_url=cai_url,
    )
    decision = plan_funding(
        ledger=ledger,
        wallet=wallet,
        compute_cost_atomic=resolved_price.compute_cost_atomic,
        payment_preference=PaymentPreference(payment),
        money_policy=money_policy,
    )
    lines = ["Job funding quote:", *decision.pretty_lines(money_policy=money_policy)]
    if resolved_price.automatic_quote is not None:
        lines.extend(resolved_price.automatic_quote.pretty_lines(money_policy=money_policy))
    else:
        lines.insert(1, f"- pricing_reason={resolved_price.pricing_reason}")
        lines.insert(1, f"- pricing_mode={resolved_price.pricing_mode}")
    return "\n".join(lines)


def _handle_job_fund(
    amount: str | None,
    payment: str,
    prompt: str | None,
    model: str | None,
    cai_url: str | None,
) -> str:
    resolved_price, money_policy, ledger, wallet = _resolve_compute_quote_for_cli(
        amount=amount,
        prompt=prompt,
        model=model,
        cai_url=cai_url,
    )
    _, session = _require_active_wallet()
    if session.unlocked_wallet_id != wallet.wallet_id:
        raise SystemExit("Active wallet must be unlocked before applying a funding decision.")

    decision = plan_funding(
        ledger=ledger,
        wallet=wallet,
        compute_cost_atomic=resolved_price.compute_cost_atomic,
        payment_preference=PaymentPreference(payment),
        money_policy=money_policy,
    )
    if not decision.can_fund:
        raise SystemExit("Cannot fund job:\n" + "\n".join(decision.pretty_lines(money_policy=money_policy)))

    wallet.spendable_balance_atomic = decision.wallet_after_atomic
    ledger.compute_reserve_balance_atomic = decision.reserve_after_atomic
    ledger.validator_fee_pool_atomic += decision.fee_quote.settlement_fee_atomic
    ledger.ai_development_fee_pool_atomic += decision.fee_quote.ai_development_fee_atomic
    ledger.tx_fee_pool_atomic += decision.fee_quote.tx_fee_atomic
    ledger.worker_distributed_atomic += decision.fee_quote.worker_reward_atomic
    ledger.settlements_applied += 1
    update_wallet(wallet)
    save_ledger(ledger)
    settlement = record_funding_settlement(
        source_wallet_id=wallet.wallet_id,
        source_wallet_address=wallet.address,
        decision=decision,
        note=decision.reason,
    )
    state_payload = None
    try:
        with urlopen(cai_url.rstrip("/") + "/state", timeout=5) as response:
            state_payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        state_payload = None

    attestation = apply_local_validator_attestation(
        settlement_id=settlement.settlement_id,
        accepted_note="Local bonded validator accepted settlement.",
        money_policy=money_policy,
        state_payload=state_payload,
        cai_url=cai_url,
        fallback_validator_address=wallet.address,
    )
    append_journal_entry(
        JournalEntry(
            entry_id=secrets.token_hex(12),
            event_type="compute_job_funded",
            created_at=_now_iso(),
            wallet_id=wallet.wallet_id,
            funding_source=(
                decision.funding_source.value if decision.funding_source else None
            ),
            compute_cost_atomic=decision.fee_quote.compute_cost_atomic,
            tx_fee_atomic=decision.fee_quote.tx_fee_atomic,
            settlement_fee_atomic=decision.fee_quote.settlement_fee_atomic,
            worker_reward_atomic=decision.fee_quote.worker_reward_atomic,
            note=decision.reason,
        )
    )
    lines = ["Job funding applied:", *decision.pretty_lines(money_policy=money_policy)]
    if resolved_price.automatic_quote is not None:
        lines.extend(resolved_price.automatic_quote.pretty_lines(money_policy=money_policy))
    else:
        lines.insert(1, f"- pricing_reason={resolved_price.pricing_reason}")
        lines.insert(1, f"- pricing_mode={resolved_price.pricing_mode}")
    lines.append(f"- settlement_id={settlement.settlement_id}")
    lines.append(f"- attestation_id={attestation.attestation_id if attestation else '<none>'}")
    return "\n".join(lines)


def _default_tx_fee_atomic(money_policy: MoneyPolicy) -> int:
    return coins_to_atomic(money_policy.default_tx_fee_coins, money_policy)


def _handle_wallet_send(recipient_address: str, amount: str) -> str:
    money_policy = MoneyPolicy()
    ensure_chain_genesis(money_policy=money_policy)
    sender, session = _require_active_wallet()
    if session.unlocked_wallet_id != sender.wallet_id:
        raise SystemExit("Active wallet must be unlocked before sending funds.")

    recipient_address = normalize_address(recipient_address)
    amount_atomic = coins_to_atomic(amount, money_policy)
    tx_fee_atomic = _default_tx_fee_atomic(money_policy)
    total_atomic = amount_atomic + tx_fee_atomic

    ledger = load_or_create_ledger(money_policy)
    sender_after, recipient = apply_wallet_transfer(
        sender_wallet_id=sender.wallet_id,
        recipient_address=recipient_address,
        amount_atomic=amount_atomic,
        tx_fee_atomic=tx_fee_atomic,
    )
    ledger = chain_backed_ledger_snapshot(ledger, money_policy=money_policy)
    save_ledger(ledger)

    append_journal_entry(
        JournalEntry(
            entry_id=secrets.token_hex(12),
            event_type="wallet_send",
            created_at=_now_iso(),
            wallet_id=sender.wallet_id,
                counterparty_address=recipient_address,
            amount_atomic=amount_atomic,
            tx_fee_atomic=tx_fee_atomic,
            note="Local wallet transfer prepared.",
        )
    )
    if recipient is not None:
        append_journal_entry(
            JournalEntry(
                entry_id=secrets.token_hex(12),
                event_type="wallet_receive",
                created_at=_now_iso(),
                wallet_id=recipient.wallet_id,
                counterparty_address=sender.address,
                amount_atomic=amount_atomic,
                note="Local wallet transfer received.",
            )
        )

    delivery = "local-delivered" if recipient is not None else "external-address-recorded"
    return dedent(
        f"""
        Wallet transfer applied:
        - from={sender.address}
        - to={recipient_address}
        - delivery={delivery}
        - amount={atomic_to_coins(amount_atomic, money_policy)}
        - tx_fee={atomic_to_coins(tx_fee_atomic, money_policy)}
        - total_debited={atomic_to_coins(total_atomic, money_policy)}
        - sender_balance_after={atomic_to_coins(sender_after.spendable_balance_atomic, money_policy)}
        - recipient_local={recipient.name if recipient is not None else '<none>'}
        """
    ).strip()


def handle_update_check(repo_root: str, *, base_url: str | None, timeout: int) -> str:
    result = check_for_updates(
        Path(repo_root),
        base_url=base_url,
        timeout_sec=timeout,
    )
    return dedent(
        f"""
        CAI update check:
        - channel={result.get('channel') or '<none>'}
        - repository={result.get('repository') or '<none>'}
        - target_branch={result.get('targetBranch') or '<none>'}
        - source_url={result.get('sourceUrl') or '<none>'}
        - base_url={result.get('baseUrl') or '<none>'}
        - local_git_commit={result['localGitCommit'] or '<none>'}
        - local_git_branch={result['localGitBranch'] or '<none>'}
        - local_git_dirty={result['localGitDirty']}
        - remote_git_commit={result['remoteGitCommit'] or '<none>'}
        - remote_git_branch={result['remoteGitBranch'] or '<none>'}
        - remote_version={result['remoteVersion'] or '<none>'}
        - update_available={result['updateAvailable']}
        - can_apply={result['canApply']}
        - apply_reason={result['applyReason']}
        """
    ).strip()


def handle_update_apply(repo_root: str, *, base_url: str | None, timeout: int) -> str:
    result = apply_remote_update(
        Path(repo_root),
        base_url=base_url,
        timeout_sec=timeout,
    )
    return dedent(
        f"""
        CAI update apply:
        - updated={result.get('updated', False)}
        - channel={result.get('channel') or '<none>'}
        - repository={result.get('repository') or '<none>'}
        - target_branch={result.get('targetBranch') or '<none>'}
        - source_url={result.get('sourceUrl') or '<none>'}
        - message={result.get('message', '<none>')}
        - remote_git_commit={result.get('remoteGitCommit') or '<none>'}
        - local_git_commit={result.get('localGitCommit') or '<none>'}
        - archive_path={result.get('archivePath') or '<none>'}
        - written_file_count={result.get('writtenFileCount', 0)}
        """
    ).strip()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "role":
        print(handle_role(args.role))
        return

    if args.command == "verify-mode":
        print(handle_verify_mode(args.mode))
        return

    if args.command == "network":
        print(handle_network())
        return

    if args.command == "network-model":
        print(handle_network_model())
        return

    if args.command == "peer-book":
        if args.sync_bootstrap:
            config = CaiNetworkConfig()
            path, imported, tried_urls = sync_peer_book_from_bootstrap(config)
            print(f"peer_book_path={path}")
            print(f"tried_state_urls={len(tried_urls)}")
            for url in tried_urls:
                print(f"state_url={url}")
            print(f"imported_count={len(imported)}")
            if imported:
                for peer in imported:
                    print(peer)
            else:
                print("<no advertised peers imported>")
            return
        if args.import_state_url:
            path, imported = import_peer_book_from_state_url(args.import_state_url)
            print(f"peer_book_path={path}")
            print(f"imported_count={len(imported)}")
            if imported:
                for peer in imported:
                    print(peer)
            else:
                print("<no overlay_advertised_peers found>")
            return
        if args.peer_to_add:
            path = add_peer_to_book(args.peer_to_add)
            print(f"Peer added to {path}")
            return
        peers = load_peer_book()
        print(f"peer_book_path={peer_book_path()}")
        if peers:
            for peer in peers:
                print(peer)
        else:
            print("<empty>")
        return

    if args.command == "update":
        if args.update_command == "check":
            print(
                handle_update_check(
                    args.repo_root,
                    base_url=args.base_url,
                    timeout=args.timeout,
                )
            )
            return
        if args.update_command == "apply":
            print(
                handle_update_apply(
                    args.repo_root,
                    base_url=args.base_url,
                    timeout=args.timeout,
                )
            )
            return

    if args.command == "topology":
        print(handle_topology())
        return

    if args.command == "money-policy":
        print(handle_money_policy())
        return

    if args.command == "status":
        print(handle_status())
        return

    if args.command == "ledger":
        print(handle_ledger())
        return

    if args.command == "developer-fund":
        if args.developer_fund_command == "validate":
            text, ok = handle_developer_fund_validate(
                repo_root=args.repo_root,
                participants_path=args.participants,
                round_path=args.round,
            )
            print(text)
            if not ok:
                raise SystemExit(1)
            return
        if args.developer_fund_command == "distribute":
            text, ok = handle_developer_fund_distribute(
                repo_root=args.repo_root,
                participants_path=args.participants,
                round_path=args.round,
                validator_id=args.validator_id,
                source_commit=args.source_commit,
                dry_run=args.dry_run,
            )
            print(text)
            if not ok:
                raise SystemExit(1)
            return
        if args.developer_fund_command == "sign-vote":
            print(
                handle_developer_fund_sign_vote(
                    round_id=args.round_id,
                    github=args.github,
                    choices=args.choice,
                )
            )
            return
        if args.developer_fund_command == "sign-founder-confirmation":
            print(
                handle_developer_fund_sign_founder_confirmation(
                    repo_root=args.repo_root,
                    participants_path=args.participants,
                    round_path=args.round,
                    confirmed_by=args.confirmed_by,
                )
            )
            return

    if args.command == "model-package-list":
        print(handle_model_package_list())
        return

    if args.command == "chunk-store":
        print(handle_chunk_store())
        return

    if args.command == "chunk-store-gc":
        print(handle_chunk_store_gc())
        return

    if args.command == "chunk-inventory-local":
        print(
            handle_chunk_inventory_local(
                args.source_id,
                source_kind=args.source_kind,
            )
        )
        return

    if args.command == "chunk-inventory-import":
        print(handle_chunk_inventory_import(args.path))
        return

    if args.command == "chunk-inventory-list":
        print(handle_chunk_inventory_list(args.source_kind))
        return

    if args.command == "chunk-inventory-sync":
        print(
            handle_chunk_inventory_sync(
                args.cai_url,
                source_kind=args.source_kind,
                inventory_urls=args.inventory_url,
            )
        )
        return

    if args.command == "chunk-download-queue":
        print(handle_chunk_download_queue())
        return

    if args.command == "chunk-source-health":
        print(handle_chunk_source_health(args.source_kind))
        return

    if args.command == "chunk-shard-hints":
        print(handle_chunk_shard_hints(args.node_id, prefetch=args.prefetch))
        return

    if args.command == "chunk-storage-accounting":
        print(handle_chunk_storage_accounting(args.node_id, record=args.record))
        return

    if args.command == "chunk-download-run":
        print(handle_chunk_download_run(args.max_tasks))
        return


    if args.command == "chunk-source-bind":
        print(
            handle_chunk_source_bind(
                args.source_kind,
                args.source_id,
                args.data_root_path,
            )
        )
        return

    if args.command == "chunk-source-bindings":
        print(handle_chunk_source_bindings(args.source_kind))
        return

    if args.command == "model-package-show":
        print(handle_model_package_show(args.catalog_id, args.version))
        return

    if args.command == "model-package-plan":
        print(
            handle_model_package_plan(
                args.catalog_id,
                args.version,
                start_layer=args.start_layer,
                end_layer=args.end_layer,
                device_rank=args.device_rank,
                world_size=args.world_size,
                node_id=args.node_id,
            )
        )
        return

    if args.command == "model-package-fetch-plan":
        print(
            handle_model_package_fetch_plan(
                args.catalog_id,
                args.version,
                start_layer=args.start_layer,
                end_layer=args.end_layer,
                device_rank=args.device_rank,
                world_size=args.world_size,
                node_id=args.node_id,
                peer_inventory_json=args.peer_inventory_json,
                seed_inventory_json=args.seed_inventory_json,
                use_imported_peer_inventory=args.use_imported_peer_inventory,
                use_imported_seed_inventory=args.use_imported_seed_inventory,
            )
        )
        return

    if args.command == "model-package-enqueue-fetch":
        print(
            handle_model_package_enqueue_fetch(
                args.catalog_id,
                args.version,
                start_layer=args.start_layer,
                end_layer=args.end_layer,
                device_rank=args.device_rank,
                world_size=args.world_size,
                node_id=args.node_id,
                peer_inventory_json=args.peer_inventory_json,
                seed_inventory_json=args.seed_inventory_json,
                use_imported_peer_inventory=args.use_imported_peer_inventory,
                use_imported_seed_inventory=args.use_imported_seed_inventory,
            )
        )
        return

    if args.command == "model-package-ensure-ready":
        print(
            handle_model_package_ensure_ready(
                args.catalog_id,
                args.version,
                start_layer=args.start_layer,
                end_layer=args.end_layer,
                device_rank=args.device_rank,
                world_size=args.world_size,
                node_id=args.node_id,
                peer_inventory_json=args.peer_inventory_json,
                seed_inventory_json=args.seed_inventory_json,
                use_imported_peer_inventory=args.use_imported_peer_inventory,
                use_imported_seed_inventory=args.use_imported_seed_inventory,
                max_tasks=args.max_tasks,
            )
        )
        return

    if args.command == "model-package-cache-all":
        print(
            handle_model_package_cache_all(
                args.catalog_id,
                args.version,
                node_id=args.node_id,
                use_imported_peer_inventory=args.use_imported_peer_inventory,
                use_imported_seed_inventory=args.use_imported_seed_inventory,
                max_tasks=args.max_tasks,
            )
        )
        return

    if args.command == "chunk-download-mark":
        print(
            handle_chunk_download_mark(
                args.task_id,
                args.status,
                source_kind=args.source_kind,
                source_id=args.source_id,
                last_error=args.last_error,
            )
        )
        return

    if args.command == "model-package-bind-artifact":
        print(
            handle_model_package_bind_artifact(
                args.catalog_id,
                args.version,
                args.artifact_id,
                args.path,
            )
        )
        return

    if args.command == "model-package-bindings":
        print(handle_model_package_bindings(args.catalog_id, args.version))
        return

    if args.command == "model-package-create-gguf":
        print(
            handle_model_package_create_gguf(
                args.catalog_id,
                args.model_id,
                args.version,
                args.gguf_path,
                n_layers=args.n_layers,
                package_kind=args.package_kind,
                chunk_size_policy=args.chunk_size_policy,
                min_chunk_mb=args.min_chunk_mb,
                max_chunk_mb=args.max_chunk_mb,
                target_chunks=args.target_chunks,
                source_repo_id=args.source_repo_id,
                source_revision=args.source_revision,
                family=args.family,
                quantization=args.quantization,
            )
        )
        return

    if args.command == "model-package-create-hf-gguf":
        print(
            handle_model_package_create_hf_gguf(
                args.model_id,
                args.version,
                catalog_id=args.catalog_id,
                preferred_filename=args.preferred_filename,
                n_layers=args.n_layers,
                package_kind=args.package_kind,
                chunk_size_policy=args.chunk_size_policy,
                min_chunk_mb=args.min_chunk_mb,
                max_chunk_mb=args.max_chunk_mb,
                target_chunks=args.target_chunks,
                source_revision=args.source_revision,
                family=args.family,
                quantization=args.quantization,
                timeout_sec=args.timeout_sec,
                cache_chunks=args.cache_chunks,
                pin_chunks=args.pin_chunks,
            )
        )
        return

    if args.command == "model-package-import-url":
        print(
            handle_model_package_import_url(
                args.manifest_url,
                expected_model_id=args.expected_model_id,
                expected_preferred_filename=args.expected_preferred_filename,
                timeout_sec=args.timeout_sec,
            )
        )
        return

    if args.command == "model-package-import-hf":
        print(
            handle_model_package_import_hf(
                args.model_id,
                preferred_filename=args.preferred_filename,
                source_revision=args.source_revision,
                timeout_sec=args.timeout_sec,
            )
        )
        return

    if args.command == "launch-check":
        report_text, ready = handle_launch_check(
            local_state_url=args.local_state_url,
            local_summary_url=args.local_summary_url,
            remote_state_url=args.remote_state_url,
            remote_summary_url=args.remote_summary_url,
        )
        print(report_text)
        if not ready:
            raise SystemExit(1)
        return

    if args.command == "node-config":
        print(handle_node_config())
        return

    if args.command == "validator-set":
        print(handle_validator_set())
        return

    if args.command == "validator-set-sync":
        print(handle_validator_set_sync(args.cai_url))
        return

    if args.command == "worker-reward-bind":
        bind_worker_reward_address(args.node_id, args.address)
        normalized_address = normalize_address(args.address)
        print(
            f"Worker reward binding saved: {args.node_id} -> {normalized_address}"
        )
        print(handle_node_config())
        return

    if args.command == "job":
        if args.job_command == "create":
            job = create_job_intent(
                prompt=args.prompt,
                compute_amount_coins=args.amount,
                payment_preference=PaymentPreference(args.payment),
                cai_url=args.cai_url,
                model_id=args.model,
            )
            print(
                dedent(
                    f"""
                    Job intent created:
                    - job_id={job.job_id}
                    - created_at={job.created_at}
                    - model={job.model_id}
                    - cai_url={job.cai_url}
                    - payment={job.payment_preference}
                    - compute_cost={atomic_to_coins(job.requested_compute_cost_atomic, MoneyPolicy())}
                    - pricing_mode={job.pricing_mode}
                    - pricing_reason={job.pricing_reason or '<none>'}
                    - status={job.status}
                    """
                ).strip()
            )
            return
        if args.job_command == "run":
            try:
                job, receipt = execute_job_intent(
                    args.job_id, request_timeout_sec=args.timeout_sec
                )
            except Exception as exc:
                raise SystemExit(f"Job execution failed: {exc}") from exc
            print(
                dedent(
                    f"""
                    Job executed:
                    - job_id={job.job_id}
                    - status={job.status}
                    - receipt_id={receipt.receipt_id}
                    - settlement_id={job.settlement_id or '<none>'}
                    - finish_reason={receipt.finish_reason or '<none>'}
                    - output={receipt.output_text or '<empty>'}
                    """
                ).strip()
            )
            return
        if args.job_command == "list":
            print(handle_job_list())
            return
        if args.job_command == "receipts":
            print(handle_receipts(args.limit))
            return
        if args.job_command == "verify":
            print(handle_job_verify(receipt_id=args.receipt_id, job_id=args.job_id))
            return

    if args.command == "settlement":
        print(handle_settlements(args.limit))
        return

    if args.command == "attestation":
        print(handle_attestations(args.limit))
        return

    if args.command == "validator-evidence":
        print(handle_validator_evidence(args.limit, args.validator_id))
        return

    if args.command == "validator-evidence-cases":
        print(handle_validator_evidence_cases(args.limit, args.validator_id))
        return

    if args.command == "validator-evidence-sync":
        print(handle_validator_evidence_sync(args.cai_url))
        return

    if args.command == "worker-payouts":
        print(handle_worker_payouts(args.limit))
        return

    if args.command == "worker-payout-reconcile":
        print(handle_worker_payout_reconcile())
        return

    if args.command == "validator-mode":
        state_url = args.state_url
        cai_url = state_url[:-6] if state_url.endswith("/state") else state_url
        snapshot = build_interface_snapshot(state_url=state_url, cai_url=cai_url)
        if args.enable and not snapshot.validator.validator_can_enable:
            raise ValueError(snapshot.validator.validator_status_note)
        state_payload = None
        if args.enable:
            try:
                with urlopen(state_url, timeout=5) as response:
                    state_payload = json.loads(response.read().decode("utf-8"))
            except Exception:
                state_payload = None
        config = set_validator_mode(
            enabled=args.enable,
            state_payload=state_payload,
            cai_url=cai_url,
        )
        if args.enable:
            print(_ansi("Validator mode enabled.", "96"))
        elif config.validator_state == "unbonding":
            print(
                _ansi(
                    "Validator mode disabled. Unbonding started and bond remains locked.",
                    "93",
                )
            )
        else:
            print(_ansi("Validator mode disabled.", "90"))
        print(handle_node_config())
        return

    if args.command == "validator-config":
        config = set_validator_static_ip_confirmation(
            confirmed=bool(args.confirm_static_ip)
        )
        state = "confirmed" if config.validator_static_ip_confirmed else "cleared"
        print(
            _ansi(
                f"Validator static IP confirmation {state}.",
                "96" if config.validator_static_ip_confirmed else "90",
            )
        )
        print(handle_node_config())
        return

    if args.command == "validator-ha":
        state_payload = None
        cai_url = args.state_url[:-6] if args.state_url.endswith("/state") else args.state_url
        try:
            with urlopen(args.state_url, timeout=5) as response:
                state_payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            state_payload = None
        if args.disable:
            config = set_validator_ha_mode(
                enabled=False,
                auto_failover=None,
                lease_seconds=args.lease_seconds,
                state_payload=state_payload,
                cai_url=cai_url,
            )
            print(_ansi("Validator HA disabled; this node is standalone.", "90"))
        else:
            role = "active" if args.active else "passive"
            config = set_validator_ha_mode(
                enabled=True,
                role=role,
                replica_id=args.replica_id,
                auto_failover=not args.no_auto_failover,
                lease_seconds=args.lease_seconds,
                state_payload=state_payload,
                cai_url=cai_url,
            )
            print(
                _ansi(
                    f"Validator HA {role} replica configured.",
                    "96" if role == "active" else "93",
                )
            )
        print(handle_node_config())
        return

    if args.command == "validator-unjail":
        config = clear_validator_jail()
        print(_ansi("Validator jail cleared.", "96"))
        print(handle_node_config())
        return

    if args.command == "validator-unbond-complete":
        config = complete_validator_unbond()
        print(_ansi("Validator unbonding completed and bond released.", "96"))
        print(handle_node_config())
        return

    if args.command == "worker-mode":
        config = set_worker_mode(
            enabled=args.enable,
            allowed_model_ids=args.allow_model,
            clear_models=args.clear_models,
            max_parallel_jobs=args.max_parallel_jobs,
            max_memory_mb=args.max_memory_mb,
        )
        state = "enabled" if config.worker_enabled else "disabled"
        print(_ansi(f"Worker mode {state}.", "96" if config.worker_enabled else "90"))
        print(handle_node_config())
        return

    if args.command == "relay-mode":
        config = set_relay_mode(enabled=args.enable)
        relay_enabled = bool(getattr(config, "relay_enabled", False))
        state = "enabled" if relay_enabled else "disabled"
        print(_ansi(f"Relay mode {state}.", "96" if relay_enabled else "90"))
        print(handle_node_config())
        return

    if args.command == "interface-state":
        print(
            handle_interface_state(
                state_url=args.state_url,
                quote_amount=args.quote_amount,
                quote_prompt=args.quote_prompt,
                quote_model=args.quote_model,
                cai_url=args.cai_url,
                payment=args.payment,
                as_json=args.json,
            )
        )
        return

    if args.command == "wallet":
        if args.wallet_command == "create":
            wallet, seed_phrase = create_seed_wallet(
                args.name, args.password, select=args.select
            )
            print(
                dedent(
                    f"""
                    Wallet created:
                    - name={wallet.name}
                    - wallet_id={wallet.wallet_id}
                    - address={wallet.address}
                    - created_at={wallet.created_at}
                    - seed_phrase={seed_phrase}
                    """
                ).strip()
            )
            return
        if args.wallet_command == "restore":
            wallet = restore_wallet_from_seed(
                args.name,
                args.password,
                seed_phrase=args.seed_phrase,
                select=args.select,
            )
            print(
                dedent(
                    f"""
                    Wallet restored:
                    - name={wallet.name}
                    - wallet_id={wallet.wallet_id}
                    - address={wallet.address}
                    - created_at={wallet.created_at}
                    """
                ).strip()
            )
            return
        if args.wallet_command == "list":
            print(handle_wallet_list())
            return
        if args.wallet_command == "select":
            wallet = select_active_wallet(args.selector)
            print(f"Active wallet set to {wallet.name} ({wallet.address})")
            return
        if args.wallet_command == "unlock":
            wallet = unlock_wallet(args.password, selector=args.wallet_selector)
            print(f"Wallet unlocked: {wallet.name} ({wallet.address})")
            return
        if args.wallet_command == "lock":
            session = lock_wallet()
            print(f"Wallet session locked. Active wallet remains: {session.active_wallet_id or '<none>'}")
            return
        if args.wallet_command == "status":
            print(handle_wallet_status())
            return
        if args.wallet_command == "history":
            print(handle_wallet_history(args.wallet_selector, args.limit))
            return
        if args.wallet_command == "credit":
            if args.wallet_selector is None:
                wallet = get_active_wallet()
            else:
                wallet = resolve_wallet(args.wallet_selector, list_wallets())
            if wallet is None:
                raise SystemExit("Active wallet is not set.")
            credited = credit_wallet(
                wallet.wallet_id,
                coins_to_atomic(args.amount, MoneyPolicy()),
            )
            print(
                f"Wallet credited: {credited.name} -> {atomic_to_coins(credited.spendable_balance_atomic, MoneyPolicy())}"
            )
            return
        if args.wallet_command == "send":
            print(_handle_wallet_send(args.recipient_address, args.amount))
            return
        if args.wallet_command == "developer-treasury":
            wallet = ensure_local_developer_treasury_wallet(
                money_policy=MoneyPolicy(),
                wallet_policy=WalletPolicy(),
            )
            print(
                dedent(
                    f"""
                    Developer treasury provisioned:
                    - chain_network={MoneyPolicy().chain_network.value}
                    - wallet_id={wallet.wallet_id}
                    - address={wallet.address}
                    - seed_file={developer_treasury_seed_file_path()}
                    - password_file={developer_treasury_password_file_path()}
                    """
                ).strip()
            )
            return
        if args.wallet_command == "ai-development":
            wallet = ensure_local_ai_development_wallet(
                money_policy=MoneyPolicy(),
                wallet_policy=WalletPolicy(),
            )
            print(
                dedent(
                    f"""
                    AI development wallet provisioned:
                    - chain_network={MoneyPolicy().chain_network.value}
                    - wallet_id={wallet.wallet_id}
                    - address={wallet.address}
                    - seed_file={ai_development_seed_file_path()}
                    - password_file={ai_development_password_file_path()}
                    """
                ).strip()
            )
            return

    if args.command == "job-quote":
        print(
            _handle_job_quote(
                args.amount,
                args.payment,
                args.prompt,
                args.model,
                args.cai_url,
            )
        )
        return

    if args.command == "job-fund":
        print(
            _handle_job_fund(
                args.amount,
                args.payment,
                args.prompt,
                args.model,
                args.cai_url,
            )
        )
        return

    if args.command == "run-cai":
        config = CaiNetworkConfig()
        if args.sync_peer_book:
            path, imported, tried_urls = sync_peer_book_from_bootstrap(config)
            print(f"peer_book_path={path}")
            print(f"tried_state_urls={len(tried_urls)}")
            print(f"imported_count={len(imported)}")
        raise SystemExit(
            launch_cai_runtime(
                cai_executable=args.cai_executable,
                cai_home=args.cai_home,
                config=config,
                api_port=args.api_port or config.default_api_port,
                libp2p_port=args.libp2p_port or config.default_libp2p_port,
                verbose=args.verbose,
                no_downloads=args.no_downloads,
                no_worker=args.no_worker,
                force_master=args.force_master,
                offline=args.offline,
                dry_run=args.dry_run,
                advertise_peers=args.advertise_peer,
            )
        )

    if args.command == "cai-owned-runtime":
        if args.loop and int(args.max_iterations or 0) <= 0:
            for result in iter_cai_owned_runtime_results(
                node_id=args.node_id,
                runtime_id=args.runtime_id,
                coordinator_cai_url=args.coordinator_cai_url,
                wallet_data_dirname=args.wallet_data_dirname,
                max_iterations=args.max_iterations,
                loop=True,
                idle_sleep_sec=args.idle_sleep_sec,
                max_payload_size_bytes=args.max_payload_size_bytes,
                lease_seconds=args.lease_seconds,
                max_attempts=args.max_attempts,
                local_runtime_token=args.local_runtime_token,
                require_local_runtime_auth=args.require_local_runtime_auth,
                require_production_llm_handoff=args.require_production_llm_handoff,
            ):
                print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
            return
        print(
            handle_cai_owned_runtime(
                node_id=args.node_id,
                runtime_id=args.runtime_id,
                coordinator_cai_url=args.coordinator_cai_url,
                wallet_data_dirname=args.wallet_data_dirname,
                max_iterations=args.max_iterations,
                loop=args.loop,
                idle_sleep_sec=args.idle_sleep_sec,
                max_payload_size_bytes=args.max_payload_size_bytes,
                lease_seconds=args.lease_seconds,
                max_attempts=args.max_attempts,
                local_runtime_token=args.local_runtime_token,
                require_local_runtime_auth=args.require_local_runtime_auth,
                require_production_llm_handoff=args.require_production_llm_handoff,
            )
        )
        return

    if args.command == "cai-owned-llm-shard-self-test":
        print(
            handle_cai_owned_llm_shard_self_test(
                model_id=args.model_id,
                payload=args.payload,
                require_production_llm_handoff=(
                    not args.allow_non_production_handoff
                ),
                save_readiness=args.save_readiness,
                show_cached=args.show_cached,
                wallet_data_dirname=args.wallet_data_dirname,
            )
        )
        return

    if args.command == "cai-owned-llm-shard-conformance":
        output_text = handle_cai_owned_llm_shard_conformance(
            model_id=args.model_id,
            payload=args.payload,
            require_production=args.require_production,
            require_production_llm_handoff=(
                not args.allow_non_production_handoff
            ),
            save_readiness=args.save_readiness,
            wallet_data_dirname=args.wallet_data_dirname,
            json_report=args.json_report,
        )
        print(output_text)
        try:
            report = json.loads(output_text)
        except json.JSONDecodeError:
            report = {}
        if not bool(report.get("ok")):
            raise SystemExit(2)
        return

    if args.command == "cai-owned-diagnostics":
        print(
            handle_cai_owned_diagnostics(
                wallet_data_dirname=args.wallet_data_dirname,
                local_node_id=args.local_node_id,
                model_id=args.model_id,
                max_records=args.max_records,
            )
        )
        return

    parser.error("Unknown command")


if __name__ == "__main__":
    main()

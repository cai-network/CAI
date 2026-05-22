# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from http.client import IncompleteRead
import json
import logging
import math
import os
import secrets
import time
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from .economics import (
    build_reserve_limit_identity_keys,
    calculate_token_priced_cost,
    chain_backed_ledger_snapshot,
    plan_funding,
    reserve_client_ip_hash,
    resolve_compute_price,
)
from .execution_performance import (
    execution_performance_preference_key,
    list_execution_performance_records,
    record_execution_attempt_performance,
)
from .decentralized_compute import (
    CAI_OWNED_TRANSPORT_EXECUTION_PIPELINE_SINGLE_PASS_FINAL_DECODE,
    await_cai_owned_transport_session_final_result,
    dispatch_cai_owned_transport_execution_dag,
    latest_completed_cai_owned_transport_proof_for_instance,
    plan_llama_cpp_distributed_execution,
    submit_cai_owned_transport_completion_notice,
    validate_cai_owned_transport_execution_proof,
)
from .gguf_shard_policy import (
    gguf_shard_compatibility,
)
from .job_network_audit import (
    active_relay_routes as _active_relay_routes,
    audit_error_status as _audit_error_status,
    augment_route_health_records_from_worker_attestations as _augment_route_health_records_from_worker_attestations_impl,
    checked_direct_socket_links as _checked_direct_socket_links,
    checked_overlay_links as _checked_overlay_links,
    chain_sync_result_audit as _chain_sync_result_audit,
    coordinator_direct_fanout_candidate_node_ids as _coordinator_direct_fanout_candidate_node_ids,
    execution_cai_owned_transport_proof as _execution_cai_owned_transport_proof_impl,
    execution_compute_cell_strategy as _execution_compute_cell_strategy_impl,
    execution_transport_mode as _execution_transport_mode,
    is_strongly_connected_participant_graph as _is_strongly_connected_participant_graph,
    node_capability_sync_result_audit as _node_capability_sync_result_audit,
    participant_node_ids as _participant_node_ids,
    participant_socket_adjacency as _participant_socket_adjacency,
    route_health_endpoints_from_worker_attestations as _route_health_endpoints_from_worker_attestations_impl,
    _route_health_record_field,
    relay_capability_snapshot as _relay_capability_snapshot,
    relay_transport_note as _relay_transport_note,
    run_chain_push_audit as _run_chain_push_audit_impl,
    run_preflight_peer_sync as _run_preflight_peer_sync_impl,
    validator_sync_result_audit as _validator_sync_result_audit,
)
from .job_http import (
    decode_json_http_payload as _decode_json_http_payload,
    delete_json as _delete_json,
    extract_finish_reason as _extract_finish_reason,
    extract_output_text as _extract_output_text,
    get_json as _get_json,
    http_error_detail as _http_error_detail,
    post_json as _post_json,
)
from .job_instance_placement import (
    cai_cluster_node_count as _cai_cluster_node_count_impl,
    dedupe_execution_node_id_attempts as _dedupe_execution_node_id_attempts_impl,
    execution_node_id_attempts as _execution_node_id_attempts_impl,
    extract_instance_participants as _extract_instance_participants_impl,
    find_model_instance as _find_model_instance_impl,
    instance_definition_participant_count as _instance_definition_participant_count_impl,
    instance_is_ready as _instance_is_ready_impl,
    instances_have_model as _instances_have_model_impl,
    preview_execution_preference_key as _preview_execution_preference_key_impl,
    preview_execution_preference_penalty_key as _preview_execution_preference_penalty_key_impl,
    preview_participant_count as _preview_participant_count_impl,
    preview_participant_node_ids as _preview_participant_node_ids_impl,
    preview_preference_key as _preview_preference_key_impl,
    private_network_node_id_attempts as _private_network_node_id_attempts_impl,
    select_preferred_preview as _select_preferred_preview_impl,
    single_node_preview_preference_key as _single_node_preview_preference_key_impl,
    require_settleable_instance_snapshot as _require_settleable_instance_snapshot_impl,
    runner_status_name as _runner_status_name_impl,
    snapshot_from_instance_definition as _snapshot_from_instance_definition_impl,
    snapshot_from_instance_state_item as _snapshot_from_instance_state_item_impl,
)
from .job_instance_readiness import (
    CAI_INSTANCE_READINESS_PROTOCOL_VERSION as _CAI_INSTANCE_READINESS_PROTOCOL_VERSION,
    CAI_INSTANCE_READINESS_STAGES as _CAI_INSTANCE_READINESS_STAGES,
    attach_cai_instance_readiness_state as _attach_cai_instance_readiness_state_impl,
    cai_instance_readiness_stage_for_status as _cai_instance_readiness_stage_for_status_impl,
    cai_instance_readiness_stage_items as _cai_instance_readiness_stage_items_impl,
    completed_model_download_node_labels as _completed_model_download_node_labels_impl,
    describe_pending_model_downloads as _describe_pending_model_downloads_impl,
    download_progress_equivalent_model_ids as _download_progress_equivalent_model_ids_impl,
    download_progress_is_completed as _download_progress_is_completed_impl,
    download_progress_is_pending as _download_progress_is_pending_impl,
    download_progress_matches_model as _download_progress_matches_model_impl,
    model_download_node_labels as _model_download_node_labels_impl,
    pending_model_download_node_labels as _pending_model_download_node_labels_impl,
)
from .job_execution_config import (
    cai_instance_ready_timeout_sec as _cai_instance_ready_timeout_sec,
    env_flag as _env_flag,
    env_positive_int as _env_positive_int,
    job_execution_attempt_timeout_sec as _job_execution_attempt_timeout_sec,
    job_execution_first_response_timeout_sec as _job_execution_first_response_timeout_sec,
    job_execution_max_attempts as _job_execution_max_attempts,
    job_execution_retry_backoff_sec as _job_execution_retry_backoff_sec,
    job_execution_total_timeout_sec as _job_execution_total_timeout_sec,
    task_level_route_probe_timeout_sec as _task_level_route_probe_timeout_sec,
    task_level_transport_executor_count as _task_level_transport_executor_count,
    task_level_transport_jobs_enabled as _task_level_transport_jobs_enabled,
    task_level_transport_jobs_required as _task_level_transport_jobs_required,
    task_level_transport_private_models_allowed as _task_level_transport_private_models_allowed,
    task_level_transport_require_data_plane_route as _task_level_transport_require_data_plane_route,
    task_level_transport_require_proven_data_plane_route as _task_level_transport_require_proven_data_plane_route,
    task_level_transport_require_runtime_ready as _task_level_transport_require_runtime_ready,
    task_level_transport_require_shard_readiness as _task_level_transport_require_shard_readiness,
    task_level_transport_timeout_sec as _task_level_transport_timeout_sec,
    task_level_transport_wait_timeout_sec as _task_level_transport_wait_timeout_sec,
    worker_identity_stale_after_seconds as _worker_identity_stale_after_seconds,
)
from .job_execution_attempts import (
    best_effort_attempt_participant_node_ids as _best_effort_attempt_participant_node_ids_impl,
    elapsed_ms as _elapsed_ms,
    execution_attempt_record as _execution_attempt_record_impl,
    record_execution_attempt_performance_best_effort as _record_execution_attempt_performance_best_effort_impl,
    should_retry_cai_startup_error as _should_retry_cai_startup_error_impl,
    should_retry_job_execution_error as _should_retry_job_execution_error_impl,
    update_latest_execution_attempt_phase as _update_latest_execution_attempt_phase_impl,
)
from .job_pricing import (
    extract_cai_owned_transport_token_usage as _extract_cai_owned_transport_token_usage,
    extract_llm_token_usage as _extract_llm_token_usage,
    extract_reserved_output_tokens as _extract_reserved_output_tokens,
    first_int_value as _first_int_value,
    merge_llm_token_usage_audit as _merge_llm_token_usage_audit,
    pricing_cap_atomic as _pricing_cap_atomic,
    pricing_floor_atomic as _pricing_floor_atomic,
)
from .job_request_payload import (
    build_text_job_request_payload as _build_text_job_request_payload,
    latest_user_message_text as _latest_user_message_text,
    message_content_text as _message_content_text,
    request_payload_prompt_text as _request_payload_prompt_text,
    task_level_transport_initial_prompt_text as _task_level_transport_initial_prompt_text,
)
from .job_node_urls import (
    cai_api_urls_by_node_id as _cai_api_urls_by_node_id_impl,
    candidate_cai_chat_base_urls as _candidate_cai_chat_base_urls_impl,
    cai_owned_overlay_peer_urls_for_target as _cai_owned_overlay_peer_urls_for_target,
    cai_owned_overlay_relay_role as _cai_owned_overlay_relay_role,
    cai_summary_urls_by_node_id as _cai_summary_urls_by_node_id,
    direct_cai_api_urls_for_overlay_relay as _direct_cai_api_urls_for_overlay_relay,
    identity_bool as _identity_bool,
    overlay_peer_set as _overlay_peer_set,
    relay_has_overlay_path_to_target as _relay_has_overlay_path_to_target,
    resolve_local_node_id_from_state_payload as _resolve_local_node_id_from_state_payload,
)
from .job_task_transport import (
    clean_task_level_peer_cai_urls as _clean_task_level_peer_cai_urls_impl,
    estimated_prompt_token_count as _estimated_prompt_token_count_impl,
    executor_candidate_route_preference_key as _executor_candidate_route_preference_key_impl,
    format_worker_node_rejection_summary as _format_worker_node_rejection_summary_impl,
    latest_known_route_latency_ms as _latest_known_route_latency_ms_impl,
    select_task_level_transport_executor_node_ids as _select_task_level_transport_executor_node_ids_impl,
    sort_executor_candidates_by_route_health as _sort_executor_candidates_by_route_health_impl,
    task_level_transport_effective_executor_count as _task_level_transport_effective_executor_count_impl,
    task_level_transport_executor_fallback_attempts as _task_level_transport_executor_fallback_attempts_impl,
    task_level_transport_gguf_shard_compatibility as _task_level_transport_gguf_shard_compatibility_impl,
    task_level_transport_instance_snapshot as _task_level_transport_instance_snapshot_impl,
    task_level_transport_llm_runtime_metadata as _task_level_transport_llm_runtime_metadata_impl,
    task_level_transport_participants_from_dag as _task_level_transport_participants_from_dag_impl,
    task_level_transport_participants_from_proof as _task_level_transport_participants_from_proof_impl,
    task_level_transport_planned_shard_ranges as _task_level_transport_planned_shard_ranges_impl,
    task_level_transport_final_output_text as _task_level_transport_final_output_text_impl,
    task_level_transport_response as _task_level_transport_response_impl,
    task_level_transport_total_layer_count as _task_level_transport_total_layer_count_impl,
    task_level_transport_total_layer_count_from_dag as _task_level_transport_total_layer_count_from_dag_impl,
    task_level_transport_total_layer_count_from_participants as _task_level_transport_total_layer_count_from_participants_impl,
    task_level_transport_usage_from_proof as _task_level_transport_usage_from_proof_impl,
    worker_node_ids_from_audit as _worker_node_ids_from_audit_impl,
)
from .job_reward_distribution import (
    distribute_worker_reward as _distribute_worker_reward,
    layer_count_from_metadata as _layer_count_from_metadata,
    optional_int_field_value as _optional_int_field_value,
    optional_int_value as _optional_int_value,
    settlement_participants_for_reward as _settlement_participants_for_reward,
    unwrap_shard_metadata as _unwrap_shard_metadata,
)
from .job_worker_eligibility import (
    accepted_worker_model_ids as _accepted_worker_model_ids,
    build_participant_eligibility_audit as _build_participant_eligibility_audit_impl,
    capability_identity_from_record as _capability_identity_from_record,
    capability_record_is_stale as _capability_record_is_stale,
    capability_records_by_node_id as _capability_records_by_node_id_impl,
    identity_allowed_model_ids as _identity_allowed_model_ids,
    identity_last_seen_at as _identity_last_seen_at,
    parse_iso_datetime as _parse_iso_datetime,
    participant_route_reachable as _participant_route_reachable,
    worker_identity_state as _worker_identity_state,
    worker_model_allowed as _worker_model_allowed,
)
from .job_storage import (
    ExecutionReceipt,
    JobIntent,
    _normalize_execution_receipt_payload,
    _normalize_job_intent_payload,
    execution_receipt_file_path,
    job_intent_file_path,
    list_execution_receipts,
    list_job_intents,
    resolve_job_intent,
    save_execution_receipts,
    save_job_intents,
    update_execution_receipt,
    update_job_intent,
)
from .model import (
    effective_private_worker_shard_minimum,
    MoneyPolicy,
    NetworkModelPolicy,
    PaymentPreference,
    curated_model_for_id,
    curated_model_registry,
    is_private_curated_model_id,
    normalize_network_model_id,
    WalletPolicy,
    resolve_execution_model_id,
)
from .model_distribution import select_model_package_manifest_for_model
from .network_routes import (
    relay_coordinator_candidate_node_ids,
    relay_route_candidates,
)
from .runtime_cleanup import (
    cleanup_orphan_llama_cpp_processes as _cleanup_orphan_llama_cpp_processes,
)
from .node_capabilities import (
    NodeCapabilityRecord,
    list_node_capabilities,
    list_verified_worker_node_ids,
    refresh_local_node_capabilities,
    sync_node_capabilities_from_cai_peers,
    worker_capability_verification_required,
)
from .route_health import (
    list_route_health_records,
    probe_direct_api_routes,
    probe_direct_data_routes,
    record_overlay_routes_from_state,
    record_route_health_from_network_audit,
    route_health_score_for_path,
)
from .worker_capability_attestations import list_worker_capability_attestations
from .node_config import (
    bind_worker_reward_address,
    get_validator_attestation_status,
    jail_validator,
    load_or_create_node_config,
)
from .settlement import (
    apply_finalized_settlement,
    ConflictingAttestationError,
    ensure_settlement_committee,
    export_settlement_proposal_payload,
    list_worker_payouts,
    list_settlements,
    reconcile_worker_payouts,
    record_chain_entries_for_finalized_settlements,
    record_funding_settlement,
    record_settlement_execution_audit,
    record_worker_payouts,
    record_validator_attestation,
    record_validator_evidence,
    reset_retryable_settlement_rejection,
    resolve_settlement,
    sign_settlement_envelope,
)
from .chain import (
    chain_balance_atomic,
    chain_settlement_history,
    ensure_chain_genesis,
    merge_remote_chain_payload,
    push_chain_to_cai_peers,
    sync_chain_from_cai_peers,
)
from .validators import (
    get_validator_record,
    resolve_validator_peer_url,
    sync_validator_set_from_cai_peers,
)
from .wallet import (
    JournalEntry,
    append_journal_entry,
    create_wallet,
    get_active_wallet,
    load_or_create_ledger,
    load_session,
    list_wallets,
    save_session,
    update_wallet,
)


CAI_INSTANCE_READINESS_PROTOCOL_VERSION = _CAI_INSTANCE_READINESS_PROTOCOL_VERSION
CAI_INSTANCE_READINESS_STAGES = _CAI_INSTANCE_READINESS_STAGES
CAI_TASK_LEVEL_TRANSPORT_JOB_PROTOCOL_VERSION = 1
CAI_TASK_LEVEL_TRANSPORT_JOB_SOURCE = "cai_owned_task_level_transport"

LOGGER = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _coalesce_cai_url(cai_url: str | None = None, CAI_url: str | None = None) -> str | None:
    resolved = str(cai_url or CAI_url or "").strip()
    return resolved or None


def _settlement_id_for_receipt_or_job(
    *,
    receipt_id: str | None,
    job_id: str | None,
    policy: WalletPolicy | None = None,
) -> str | None:
    normalized_receipt_id = str(receipt_id or "").strip()
    normalized_job_id = str(job_id or "").strip()
    if not normalized_receipt_id and not normalized_job_id:
        return None
    for settlement in list_settlements(policy):
        audit = getattr(settlement, "balance_audit", None)
        if not isinstance(audit, dict):
            continue
        execution = audit.get("execution")
        if not isinstance(execution, dict):
            continue
        audit_receipt_id = str(
            execution.get("receipt_id") or execution.get("receiptId") or ""
        ).strip()
        audit_job_id = str(
            execution.get("job_id") or execution.get("jobId") or ""
        ).strip()
        if normalized_receipt_id and audit_receipt_id == normalized_receipt_id:
            return settlement.settlement_id
        if normalized_job_id and audit_job_id == normalized_job_id:
            return settlement.settlement_id
    return None


def _ensure_active_job_wallet(wallet_policy: WalletPolicy | None = None):
    wallet = get_active_wallet(wallet_policy)
    if wallet is not None:
        return wallet

    wallets = list_wallets(wallet_policy)
    if wallets:
        session = load_session(wallet_policy)
        session.active_wallet_id = wallets[0].wallet_id
        save_session(session, wallet_policy)
        return wallets[0]

    return create_wallet(
        "CAI Device Wallet",
        secrets.token_urlsafe(32),
        select=True,
        wallet_policy=wallet_policy,
    )


def create_job_intent(
    *,
    prompt: str,
    compute_amount_coins: str | None,
    payment_preference: PaymentPreference,
    cai_url: str | None = None,
    execution_cai_url: str | None = None,
    CAI_url: str | None = None,
    model_id: str | None = None,
    requester_node_id: str | None = None,
    request_payload_preview: dict[str, Any] | None = None,
    reserve_client_ip: str | None = None,
    money_policy: MoneyPolicy | None = None,
    network_model_policy: NetworkModelPolicy | None = None,
    wallet_policy: WalletPolicy | None = None,
) -> JobIntent:
    active_money_policy = money_policy or MoneyPolicy()
    active_network_model_policy = network_model_policy or NetworkModelPolicy()
    wallet = _ensure_active_job_wallet(wallet_policy)
    ledger = chain_backed_ledger_snapshot(
        load_or_create_ledger(active_money_policy, wallet_policy),
        money_policy=active_money_policy,
        wallet_policy=wallet_policy,
    )
    resolved_model_id = normalize_network_model_id(
        model_id or active_network_model_policy.network_default_model_id,
        active_network_model_policy,
    )
    resolved_cai_url = _coalesce_cai_url(cai_url, CAI_url)
    if resolved_cai_url is None:
        raise ValueError("CAI URL is required.")
    resolved_execution_cai_url = _coalesce_cai_url(
        execution_cai_url,
        None,
    ) or resolved_cai_url
    normalized_cai_url = resolved_cai_url.rstrip("/")
    normalized_execution_cai_url = resolved_execution_cai_url.rstrip("/")
    reserve_identity_keys = build_reserve_limit_identity_keys(
        wallet.wallet_id,
        reserve_client_ip=reserve_client_ip,
        money_policy=active_money_policy,
    )
    reserve_client_hash = reserve_client_ip_hash(reserve_client_ip)
    reserved_output_tokens = _extract_reserved_output_tokens(
        request_payload_preview,
        active_money_policy,
    )
    resolved_price = resolve_compute_price(
        compute_amount_coins=compute_amount_coins,
        prompt=prompt,
        model_id=resolved_model_id,
        max_output_tokens=reserved_output_tokens,
        cai_url=normalized_execution_cai_url,
        ledger=ledger,
        money_policy=active_money_policy,
        network_model_policy=active_network_model_policy,
    )
    automatic_quote = resolved_price.automatic_quote
    resolved_pricing_basis = getattr(resolved_price, "pricing_basis", "manual")
    if not isinstance(resolved_pricing_basis, str) or not resolved_pricing_basis.strip():
        resolved_pricing_basis = "manual"
    pricing_floor_atomic = _pricing_floor_atomic(active_money_policy)
    pricing_cap_atomic = _pricing_cap_atomic(active_money_policy)

    job = JobIntent(
        job_id=secrets.token_hex(12),
        created_at=_now_iso(),
        source_wallet_id=wallet.wallet_id,
        source_wallet_address=wallet.address,
        model_id=resolved_model_id,
        cai_url=normalized_cai_url,
        execution_cai_url=normalized_execution_cai_url,
        requester_node_id=(str(requester_node_id).strip() or None)
        if requester_node_id is not None
        else None,
        prompt=prompt,
        payment_preference=payment_preference.value,
        requested_compute_cost_atomic=resolved_price.compute_cost_atomic,
        pricing_mode=resolved_price.pricing_mode,
        pricing_basis=resolved_pricing_basis,
        pricing_reason=resolved_price.pricing_reason,
        reserved_prompt_tokens=(
            automatic_quote.prompt_tokens_estimate if automatic_quote is not None else None
        ),
        reserved_completion_tokens=(
            automatic_quote.reserved_output_tokens
            if automatic_quote is not None
            else None
        ),
        pricing_final_multiplier_bps=(
            automatic_quote.final_multiplier_bps if automatic_quote is not None else None
        ),
        pricing_floor_atomic=(
            pricing_floor_atomic if automatic_quote is not None else None
        ),
        pricing_cap_atomic=(
            pricing_cap_atomic if automatic_quote is not None else None
        ),
        pricing_input_token_price_atomic=(
            automatic_quote.input_token_price_atomic if automatic_quote is not None else None
        ),
        pricing_output_token_price_atomic=(
            automatic_quote.output_token_price_atomic if automatic_quote is not None else None
        ),
        reserve_limit_identity_keys=list(reserve_identity_keys),
        reserve_client_ip_hash=reserve_client_hash,
    )
    jobs = list_job_intents(wallet_policy)
    jobs.append(job)
    save_job_intents(jobs, wallet_policy)
    append_journal_entry(
        JournalEntry(
            entry_id=secrets.token_hex(12),
            event_type="job_intent_created",
            created_at=_now_iso(),
            wallet_id=wallet.wallet_id,
            amount_atomic=job.requested_compute_cost_atomic,
            note=(
                f"Job intent {job.job_id} created for model {job.model_id} "
                f"using {job.pricing_mode} pricing."
            ),
        ),
        wallet_policy,
    )
    return job


def reconcile_stale_running_job_intents(
    policy: WalletPolicy | None = None,
    *,
    stale_after_seconds: int | None = None,
) -> int:
    raw_timeout = (
        stale_after_seconds
        if stale_after_seconds is not None
        else (
            os.getenv("CAI_STALE_RUNNING_JOB_SECONDS")
            or os.getenv("CAI_CHAT_COMPLETION_TIMEOUT_SECONDS")
            or "1800"
        )
    )
    try:
        timeout_seconds = max(60, int(raw_timeout))
    except (TypeError, ValueError):
        timeout_seconds = 1800

    now = datetime.now(tz=UTC)
    changed = 0
    items = list_job_intents(policy)
    receipt_by_job_id = {
        str(item.job_id).strip(): item
        for item in list_execution_receipts(policy)
        if str(item.job_id).strip()
    }
    for job in items:
        if job.status != "running":
            continue
        if not job.receipt_id:
            receipt = receipt_by_job_id.get(str(job.job_id).strip())
            if receipt is not None:
                job.status = "completed"
                job.receipt_id = receipt.receipt_id
                job.settlement_id = (
                    job.settlement_id
                    or _settlement_id_for_receipt_or_job(
                        receipt_id=receipt.receipt_id,
                        job_id=job.job_id,
                        policy=policy,
                    )
                )
                job.last_error = None
                changed += 1
                continue
        if job.receipt_id or job.settlement_id:
            continue
        created_at = _parse_iso_datetime(job.created_at)
        if created_at is None:
            continue
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        age_seconds = (now - created_at.astimezone(UTC)).total_seconds()
        if age_seconds <= timeout_seconds:
            continue
        job.status = "failed"
        job.last_error = (
            "Job was marked failed because it remained running beyond "
            f"{timeout_seconds} seconds without an execution receipt."
        )
        changed += 1
    if changed:
        save_job_intents(items, policy)
    return changed


def _execution_remaining_timeout_sec(deadline_monotonic: float) -> float:
    return max(0.0, deadline_monotonic - time.monotonic())


def _bounded_execution_timeout_sec(
    requested_timeout_sec: int | float,
    deadline_monotonic: float,
    *,
    phase: str,
) -> int:
    remaining = _execution_remaining_timeout_sec(deadline_monotonic)
    if remaining <= 0:
        raise TimeoutError(f"CAI job execution timeout expired before {phase}.")
    return max(1, int(math.ceil(min(float(requested_timeout_sec), remaining))))


def _execution_attempt_deadline_monotonic(
    total_deadline_monotonic: float,
    attempt_timeout_sec: int | float,
) -> float:
    return min(
        total_deadline_monotonic,
        time.monotonic() + max(1.0, float(attempt_timeout_sec)),
    )


def _chain_backed_funding_wallet(
    wallet,
    *,
    money_policy: MoneyPolicy,
    wallet_policy: WalletPolicy | None = None,
):
    ensure_chain_genesis(policy=wallet_policy, money_policy=money_policy)
    return replace(
        wallet,
        spendable_balance_atomic=chain_balance_atomic(wallet.address, wallet_policy),
    )


def _request_payload_model_id(
    request_payload_override: Mapping[str, Any] | None,
) -> str | None:
    if not isinstance(request_payload_override, Mapping):
        return None
    model_id = str(request_payload_override.get("model") or "").strip()
    return model_id or None


def _request_payload_model_matches_job(
    payload_model_id: str | None,
    *,
    job_model_id: str,
    execution_model_id: str,
    network_model_policy: NetworkModelPolicy,
) -> bool:
    if not payload_model_id:
        return True
    normalized_payload_model_id = normalize_network_model_id(
        payload_model_id,
        network_model_policy,
    )
    payload_execution_model_id = resolve_execution_model_id(
        payload_model_id,
        network_model_policy,
    )
    return (
        payload_model_id == job_model_id
        or payload_model_id == execution_model_id
        or normalized_payload_model_id == job_model_id
        or payload_execution_model_id == execution_model_id
    )


def _validate_request_payload_model_matches_job(
    *,
    job_model_id: str,
    execution_model_id: str,
    request_payload_override: Mapping[str, Any] | None,
    network_model_policy: NetworkModelPolicy | None = None,
) -> None:
    active_network_model_policy = network_model_policy or NetworkModelPolicy()
    payload_model_id = _request_payload_model_id(request_payload_override)
    if _request_payload_model_matches_job(
        payload_model_id,
        job_model_id=job_model_id,
        execution_model_id=execution_model_id,
        network_model_policy=active_network_model_policy,
    ):
        return
    raise ValueError(
        f"Request payload model '{payload_model_id}' does not match metered job "
        f"model '{job_model_id}' or execution model '{execution_model_id}'. "
        "Refusing to execute a different model than the job selected."
    )


def _model_selection_audit(
    *,
    job_model_id: str,
    execution_model_id: str,
    request_payload_override: Mapping[str, Any] | None,
    network_model_policy: NetworkModelPolicy | None = None,
) -> dict[str, Any]:
    active_network_model_policy = network_model_policy or NetworkModelPolicy()
    payload_model_id = _request_payload_model_id(request_payload_override)
    normalized_payload_model_id = (
        normalize_network_model_id(payload_model_id, active_network_model_policy)
        if payload_model_id
        else None
    )
    payload_execution_model_id = (
        resolve_execution_model_id(payload_model_id, active_network_model_policy)
        if payload_model_id
        else None
    )
    matches_job = _request_payload_model_matches_job(
        payload_model_id,
        job_model_id=job_model_id,
        execution_model_id=execution_model_id,
        network_model_policy=active_network_model_policy,
    )
    return {
        "status": "matched" if matches_job else "mismatch",
        "jobModelId": job_model_id,
        "executionModelId": execution_model_id,
        "requestPayloadModelId": payload_model_id,
        "normalizedRequestPayloadModelId": normalized_payload_model_id,
        "requestPayloadExecutionModelId": payload_execution_model_id,
        "requestPayloadMatchesJob": matches_job,
        "requestPayloadModelOverridden": bool(
            payload_model_id and payload_model_id != execution_model_id
        ),
    }


def _log_best_effort_failure(operation: str, exc: Exception) -> None:
    LOGGER.warning("%s failed: %s: %s", operation, type(exc).__name__, exc)


def _run_preflight_peer_sync(
    *,
    state_payload: dict[str, Any] | None,
    cai_url: str,
    wallet_policy: WalletPolicy | None = None,
) -> dict[str, Any]:
    return _run_preflight_peer_sync_impl(
        state_payload=state_payload,
        cai_url=cai_url,
        wallet_policy=wallet_policy,
        sync_validator_set_from_cai_peers_func=sync_validator_set_from_cai_peers,
        sync_chain_from_cai_peers_func=sync_chain_from_cai_peers,
        now_iso_func=_now_iso,
    )


def _run_chain_push_audit(
    *,
    state_payload: dict[str, Any] | None,
    cai_url: str,
    wallet_policy: WalletPolicy | None = None,
    timeout_sec: float = 0.75,
) -> dict[str, Any]:
    return _run_chain_push_audit_impl(
        state_payload=state_payload,
        cai_url=cai_url,
        wallet_policy=wallet_policy,
        timeout_sec=timeout_sec,
        push_chain_to_cai_peers_func=push_chain_to_cai_peers,
        now_iso_func=_now_iso,
    )


def _execution_attempt_record(
    *,
    attempt: int,
    status: str,
    started_at: str,
    completed_at: str | None,
    participant_node_ids: list[str],
    excluded_node_ids: list[str],
    instance_id: str | None = None,
    error: Exception | None = None,
    retry_scheduled: bool = False,
    timeout_sec: int | None = None,
    phase: str | None = None,
    phase_started_at: str | None = None,
    phase_message: str | None = None,
    attempt_duration_ms: int | None = None,
    readiness_duration_ms: int | None = None,
    response_duration_ms: int | None = None,
) -> dict[str, Any]:
    return _execution_attempt_record_impl(
        attempt=attempt,
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        participant_node_ids=participant_node_ids,
        excluded_node_ids=excluded_node_ids,
        now_iso_func=_now_iso,
        instance_id=instance_id,
        error=error,
        retry_scheduled=retry_scheduled,
        timeout_sec=timeout_sec,
        phase=phase,
        phase_started_at=phase_started_at,
        phase_message=phase_message,
        attempt_duration_ms=attempt_duration_ms,
        readiness_duration_ms=readiness_duration_ms,
        response_duration_ms=response_duration_ms,
    )


def _update_latest_execution_attempt_phase(
    job: JobIntent,
    execution_attempts: list[dict[str, Any]],
    *,
    phase: str,
    wallet_policy: WalletPolicy | None,
    phase_message: str | None = None,
    participant_node_ids: list[str] | None = None,
    instance_id: str | None = None,
    timeout_sec: int | None = None,
) -> None:
    _update_latest_execution_attempt_phase_impl(
        job,
        execution_attempts,
        phase=phase,
        wallet_policy=wallet_policy,
        now_iso_func=_now_iso,
        update_job_intent_func=update_job_intent,
        phase_message=phase_message,
        participant_node_ids=participant_node_ids,
        instance_id=instance_id,
        timeout_sec=timeout_sec,
    )


def _record_execution_attempt_performance_best_effort(
    *,
    model_id: str,
    requester_node_id: str | None,
    executor_node_ids: list[str] | tuple[str, ...] | set[str] | None,
    status: str,
    attempt_duration_ms: int | None = None,
    readiness_duration_ms: int | None = None,
    response_duration_ms: int | None = None,
    timeout_sec: int | float | None = None,
    error_type: str | None = None,
    wallet_policy: WalletPolicy | None = None,
) -> None:
    _record_execution_attempt_performance_best_effort_impl(
        model_id=model_id,
        requester_node_id=requester_node_id,
        executor_node_ids=executor_node_ids,
        status=status,
        record_execution_attempt_performance_func=record_execution_attempt_performance,
        log_best_effort_failure_func=_log_best_effort_failure,
        attempt_duration_ms=attempt_duration_ms,
        readiness_duration_ms=readiness_duration_ms,
        response_duration_ms=response_duration_ms,
        timeout_sec=timeout_sec,
        error_type=error_type,
        wallet_policy=wallet_policy,
    )


def _best_effort_attempt_participant_node_ids(
    *,
    instance_snapshot: dict[str, Any] | None,
    cai_url: str,
    model_id: str,
) -> list[str]:
    return _best_effort_attempt_participant_node_ids_impl(
        instance_snapshot=instance_snapshot,
        cai_url=cai_url,
        model_id=model_id,
        participant_node_ids_func=_participant_node_ids,
        resolve_cai_instance_snapshot_func=resolve_cai_instance_snapshot,
        log_best_effort_failure_func=_log_best_effort_failure,
    )


def _should_retry_job_execution_error(exc: Exception) -> bool:
    return _should_retry_job_execution_error_impl(
        exc,
        should_retry_cai_startup_error_func=_should_retry_cai_startup_error,
    )


def execute_job_intent(
    job_id: str,
    *,
    money_policy: MoneyPolicy | None = None,
    wallet_policy: WalletPolicy | None = None,
    request_timeout_sec: int = 1800,
    request_payload_override: dict[str, Any] | None = None,
) -> tuple[JobIntent, ExecutionReceipt]:
    active_money_policy = money_policy or MoneyPolicy()
    active_network_model_policy = NetworkModelPolicy()
    request_timeout_sec = _env_positive_int(
        "CAI_JOB_EXECUTION_REQUEST_TIMEOUT_SECONDS",
        int(max(1, request_timeout_sec)),
    )
    execution_deadline_monotonic = (
        time.monotonic() + _job_execution_total_timeout_sec(request_timeout_sec)
    )
    job = resolve_job_intent(job_id, wallet_policy)
    if job is None:
        raise ValueError(f"Job intent '{job_id}' not found.")

    session = load_session(wallet_policy)
    wallet = get_active_wallet(wallet_policy)
    if wallet is None or wallet.wallet_id != job.source_wallet_id:
        raise ValueError("Active wallet must match the job owner.")

    preflight_state_payload = _load_cai_state_payload(job.cai_url)
    preflight_peer_sync_audit = _run_preflight_peer_sync(
        state_payload=preflight_state_payload,
        cai_url=job.cai_url,
        wallet_policy=wallet_policy,
    )

    funding_wallet = _chain_backed_funding_wallet(
        wallet,
        money_policy=active_money_policy,
        wallet_policy=wallet_policy,
    )
    ledger = chain_backed_ledger_snapshot(
        load_or_create_ledger(active_money_policy, wallet_policy),
        money_policy=active_money_policy,
        wallet_policy=wallet_policy,
    )
    reservation_decision = plan_funding(
        ledger=ledger,
        wallet=funding_wallet,
        compute_cost_atomic=job.requested_compute_cost_atomic,
        payment_preference=PaymentPreference(job.payment_preference),
        money_policy=active_money_policy,
        wallet_policy=wallet_policy,
        reserve_limit_identity_keys=tuple(job.reserve_limit_identity_keys or []),
        reserve_client_hash=job.reserve_client_ip_hash,
    )
    if not reservation_decision.can_fund:
        job.status = "failed"
        job.last_error = reservation_decision.reason
        update_job_intent(job, wallet_policy)
        raise ValueError(reservation_decision.reason)
    if (
        reservation_decision.wallet_after_atomic
        != funding_wallet.spendable_balance_atomic
        and session.unlocked_wallet_id != wallet.wallet_id
    ):
        raise ValueError("Active wallet must be unlocked before executing a job.")

    job.status = "running"
    job.last_error = None
    normalized_job_model_id = normalize_network_model_id(
        job.model_id,
        active_network_model_policy,
    )
    if normalized_job_model_id != job.model_id:
        job.model_id = normalized_job_model_id
    update_job_intent(job, wallet_policy)

    execution_model_id = resolve_execution_model_id(
        job.model_id, active_network_model_policy
    )
    try:
        _validate_request_payload_model_matches_job(
            job_model_id=job.model_id,
            execution_model_id=execution_model_id,
            request_payload_override=request_payload_override,
            network_model_policy=active_network_model_policy,
        )
    except ValueError as exc:
        job.status = "failed"
        job.last_error = str(exc)
        update_job_intent(job, wallet_policy)
        raise
    execution_cai_url = (job.execution_cai_url or job.cai_url).rstrip("/")
    private_network_model = is_private_curated_model_id(
        job.model_id,
        active_network_model_policy,
    )

    instance_snapshot: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    if _task_level_transport_jobs_enabled() and (
        not private_network_model or _task_level_transport_private_models_allowed()
    ):
        task_level_state_payload = preflight_state_payload
        if execution_cai_url != job.cai_url.rstrip("/"):
            task_level_state_payload = (
                _load_cai_state_payload(execution_cai_url) or preflight_state_payload
            )
        try:
            task_transport_result = _try_execute_task_level_transport_job(
                job,
                execution_cai_url=execution_cai_url,
                execution_model_id=execution_model_id,
                state_payload=task_level_state_payload,
                request_timeout_sec=_bounded_execution_timeout_sec(
                    request_timeout_sec,
                    execution_deadline_monotonic,
                    phase="CAI-owned task-level transport",
                ),
                request_payload_override=request_payload_override,
                wallet_policy=wallet_policy,
            )
        except Exception as exc:
            if _task_level_transport_jobs_required():
                job.status = "failed"
                job.last_error = str(exc)
                update_job_intent(job, wallet_policy)
                raise
            task_transport_result = None
        if task_transport_result is None and _task_level_transport_jobs_required():
            message = (
                "CAI-owned task-level transport was required, but no usable "
                "requester/executor route was available."
            )
            job.status = "failed"
            job.last_error = message
            update_job_intent(job, wallet_policy)
            raise RuntimeError(message)
        if isinstance(task_transport_result, dict):
            response = task_transport_result.get("response")
            instance_snapshot = task_transport_result.get("instance_snapshot")

    if response is None or instance_snapshot is None:
        max_execution_attempts = _job_execution_max_attempts()
        last_execution_exception: Exception | None = None
        failed_execution_node_ids: set[str] = set()
        execution_attempts = list(job.execution_attempts or [])
        for attempt in range(max_execution_attempts):
            attempt_started_monotonic = time.monotonic()
            attempt_number = attempt + 1
            started_at = _now_iso()
            excluded_node_ids = sorted(failed_execution_node_ids)
            attempt_instance_snapshot: dict[str, Any] | None = None
            readiness_started_monotonic: float | None = None
            readiness_duration_ms: int | None = None
            response_started_monotonic: float | None = None
            response_duration_ms: int | None = None
            attempt_budget_sec = _job_execution_attempt_timeout_sec(request_timeout_sec)
            attempt_deadline_monotonic = _execution_attempt_deadline_monotonic(
                execution_deadline_monotonic,
                attempt_budget_sec,
            )
            try:
                ready_timeout_sec = _bounded_execution_timeout_sec(
                    _cai_instance_ready_timeout_sec(
                        private_network_model=private_network_model,
                    ),
                    attempt_deadline_monotonic,
                    phase=f"attempt {attempt_number} instance readiness",
                )
                attempt_timeout_sec = _bounded_execution_timeout_sec(
                    attempt_budget_sec,
                    attempt_deadline_monotonic,
                    phase=f"attempt {attempt_number} model response",
                )
            except TimeoutError as exc:
                execution_attempts.append(
                    _execution_attempt_record(
                        attempt=attempt_number,
                        status="failed",
                        started_at=started_at,
                        completed_at=_now_iso(),
                        participant_node_ids=[],
                        excluded_node_ids=excluded_node_ids,
                        error=exc,
                        retry_scheduled=False,
                        phase="deadline_expired",
                        phase_message="Execution deadline expired before a new attempt could start.",
                        attempt_duration_ms=_elapsed_ms(attempt_started_monotonic),
                    )
                )
                job.status = "failed"
                job.last_error = str(exc)
                job.execution_attempts = execution_attempts
                update_job_intent(job, wallet_policy)
                raise
            submit_timeout_sec = attempt_timeout_sec
            execution_attempts.append(
                _execution_attempt_record(
                    attempt=attempt_number,
                    status="running",
                    started_at=started_at,
                    completed_at=None,
                    participant_node_ids=[],
                    excluded_node_ids=excluded_node_ids,
                    retry_scheduled=False,
                    timeout_sec=attempt_timeout_sec,
                    phase="instance_readiness",
                    phase_message="Preparing execution route and waiting for CAI instance readiness.",
                )
            )
            job.status = "running"
            job.execution_attempts = execution_attempts
            update_job_intent(job, wallet_policy)
            try:
                ensure_kwargs: dict[str, Any] = {
                    "ready_timeout_sec": ready_timeout_sec,
                    "private_network_model": private_network_model,
                    "requester_node_id": job.requester_node_id,
                }
                if excluded_node_ids:
                    ensure_kwargs["excluded_node_ids"] = excluded_node_ids
                readiness_started_monotonic = time.monotonic()
                _update_latest_execution_attempt_phase(
                    job,
                    execution_attempts,
                    phase="instance_readiness",
                    phase_message="Preparing execution route and waiting for CAI instance readiness.",
                    wallet_policy=wallet_policy,
                )
                attempt_instance_snapshot = ensure_cai_instance(
                    execution_cai_url,
                    execution_model_id,
                    **ensure_kwargs,
                )
                attempt_instance_snapshot = (
                    resolve_cai_instance_snapshot(execution_cai_url, execution_model_id)
                    or attempt_instance_snapshot
                )
                readiness_duration_ms = _elapsed_ms(readiness_started_monotonic)
                ready_participant_node_ids = _participant_node_ids(
                    attempt_instance_snapshot
                )
                ready_instance_id = (
                    str(attempt_instance_snapshot.get("instance_id") or "")
                    if isinstance(attempt_instance_snapshot, dict)
                    else None
                )
                response_budget_sec = _job_execution_first_response_timeout_sec(
                    attempt_budget_sec
                )
                submit_timeout_sec = _bounded_execution_timeout_sec(
                    response_budget_sec,
                    attempt_deadline_monotonic,
                    phase=f"attempt {attempt_number} model response",
                )
                response_started_monotonic = time.monotonic()
                _update_latest_execution_attempt_phase(
                    job,
                    execution_attempts,
                    phase="first_response_wait",
                    phase_message="CAI instance is ready; waiting for the model response.",
                    participant_node_ids=ready_participant_node_ids,
                    instance_id=ready_instance_id,
                    timeout_sec=submit_timeout_sec,
                    wallet_policy=wallet_policy,
                )
                response = _submit_text_job_to_cai(
                    execution_cai_url,
                    execution_model_id,
                    job.prompt,
                    timeout_sec=submit_timeout_sec,
                    request_payload_override=request_payload_override,
                )
                response_duration_ms = _elapsed_ms(response_started_monotonic)
                command_instance_snapshot = None
                response_id = response.get("id")
                if isinstance(response_id, str) and response_id.strip():
                    command_instance_snapshot = resolve_cai_command_instance_snapshot(
                        execution_cai_url,
                        response_id,
                        model_id=execution_model_id,
                    )
                attempt_instance_snapshot = (
                    command_instance_snapshot
                    or resolve_cai_instance_snapshot(
                        execution_cai_url,
                        execution_model_id,
                    )
                    or attempt_instance_snapshot
                )
                _require_settleable_instance_snapshot(attempt_instance_snapshot)
                instance_snapshot = attempt_instance_snapshot
                last_execution_exception = None
                participant_node_ids = _participant_node_ids(attempt_instance_snapshot)
                execution_attempts[-1] = _execution_attempt_record(
                    attempt=attempt_number,
                    status="completed",
                    started_at=started_at,
                    completed_at=_now_iso(),
                    participant_node_ids=participant_node_ids,
                    excluded_node_ids=excluded_node_ids,
                    instance_id=(
                        str(attempt_instance_snapshot.get("instance_id") or "")
                        if isinstance(attempt_instance_snapshot, dict)
                        else None
                    ),
                    timeout_sec=submit_timeout_sec,
                    phase="completed",
                    phase_message="Execution attempt completed successfully.",
                    attempt_duration_ms=_elapsed_ms(attempt_started_monotonic),
                    readiness_duration_ms=readiness_duration_ms,
                    response_duration_ms=response_duration_ms,
                )
                _record_execution_attempt_performance_best_effort(
                    model_id=execution_model_id,
                    requester_node_id=job.requester_node_id
                    or _resolve_local_node_id_from_state_payload(
                        preflight_state_payload or {},
                        job.cai_url,
                    ),
                    executor_node_ids=participant_node_ids,
                    status="completed",
                    attempt_duration_ms=execution_attempts[-1].get(
                        "attemptDurationMs"
                    ),
                    readiness_duration_ms=readiness_duration_ms,
                    response_duration_ms=response_duration_ms,
                    timeout_sec=submit_timeout_sec,
                    wallet_policy=wallet_policy,
                )
                job.execution_attempts = execution_attempts
                update_job_intent(job, wallet_policy)
                break
            except Exception as exc:
                if (
                    readiness_started_monotonic is not None
                    and readiness_duration_ms is None
                ):
                    readiness_duration_ms = _elapsed_ms(readiness_started_monotonic)
                if (
                    response_started_monotonic is not None
                    and response_duration_ms is None
                ):
                    response_duration_ms = _elapsed_ms(response_started_monotonic)
                last_execution_exception = exc
                participant_node_ids = _best_effort_attempt_participant_node_ids(
                    instance_snapshot=attempt_instance_snapshot,
                    cai_url=execution_cai_url,
                    model_id=execution_model_id,
                )
                failed_execution_node_ids.update(participant_node_ids)
                cleanup_cai_model_instances(
                    execution_cai_url,
                    execution_model_id,
                    best_effort=True,
                )
                retry_scheduled = (
                    attempt + 1 < max_execution_attempts
                    and _should_retry_job_execution_error(exc)
                    and _execution_remaining_timeout_sec(execution_deadline_monotonic)
                    > 1.0
                )
                execution_attempts[-1] = _execution_attempt_record(
                    attempt=attempt_number,
                    status="retrying" if retry_scheduled else "failed",
                    started_at=started_at,
                    completed_at=_now_iso(),
                    participant_node_ids=participant_node_ids,
                    excluded_node_ids=excluded_node_ids,
                    instance_id=(
                        str(attempt_instance_snapshot.get("instance_id") or "")
                        if isinstance(attempt_instance_snapshot, dict)
                        else None
                    ),
                    error=exc,
                    retry_scheduled=retry_scheduled,
                    timeout_sec=submit_timeout_sec,
                    phase="retry_scheduled" if retry_scheduled else "failed",
                    phase_message=(
                        "Execution attempt failed; retrying with another route."
                        if retry_scheduled
                        else "Execution attempt failed and no retry is scheduled."
                    ),
                    attempt_duration_ms=_elapsed_ms(attempt_started_monotonic),
                    readiness_duration_ms=readiness_duration_ms,
                    response_duration_ms=response_duration_ms,
                )
                _record_execution_attempt_performance_best_effort(
                    model_id=execution_model_id,
                    requester_node_id=job.requester_node_id
                    or _resolve_local_node_id_from_state_payload(
                        preflight_state_payload or {},
                        job.cai_url,
                    ),
                    executor_node_ids=participant_node_ids,
                    status="failed",
                    attempt_duration_ms=execution_attempts[-1].get(
                        "attemptDurationMs"
                    ),
                    readiness_duration_ms=readiness_duration_ms,
                    response_duration_ms=response_duration_ms,
                    timeout_sec=submit_timeout_sec,
                    error_type=type(exc).__name__,
                    wallet_policy=wallet_policy,
                )
                job.execution_attempts = execution_attempts
                if retry_scheduled:
                    job.status = "running"
                    job.last_error = (
                        f"Execution attempt {attempt_number} failed; retrying: {exc}"
                    )
                    update_job_intent(job, wallet_policy)
                    cleanup_orphan_llama_cpp_processes(
                        cai_url=execution_cai_url,
                        model_id=execution_model_id,
                    )
                    sleep_sec = min(
                        _job_execution_retry_backoff_sec(),
                        max(
                            0.0,
                            _execution_remaining_timeout_sec(
                                execution_deadline_monotonic
                            )
                            - 1.0,
                        ),
                    )
                    if sleep_sec > 0:
                        time.sleep(sleep_sec)
                    continue
                job.status = "failed"
                job.last_error = str(exc)
                update_job_intent(job, wallet_policy)
                raise

        if last_execution_exception is not None:
            raise last_execution_exception

    _require_settleable_instance_snapshot(instance_snapshot)

    participants = list(instance_snapshot["participants"] if instance_snapshot else [])
    execution_state_payload = _load_cai_state_payload(execution_cai_url)
    route_health_records = _route_health_records_for_execution_settlement(
        instance_snapshot=instance_snapshot,
        wallet_policy=wallet_policy,
    )
    network_audit = _build_execution_network_audit(
        state_payload=execution_state_payload,
        instance_snapshot=instance_snapshot,
        requester_node_id=job.requester_node_id
        or _resolve_local_node_id_from_state_payload(
            execution_state_payload or {},
            job.cai_url,
        ),
        route_health_records=route_health_records,
        wallet_policy=wallet_policy,
    )
    if job.execution_attempts:
        network_audit["executionAttempts"] = list(job.execution_attempts)
        network_audit["executionAttemptCount"] = len(job.execution_attempts)
    network_audit["preflightPeerSync"] = preflight_peer_sync_audit
    network_audit["modelSelection"] = _model_selection_audit(
        job_model_id=job.model_id,
        execution_model_id=execution_model_id,
        request_payload_override=request_payload_override,
        network_model_policy=active_network_model_policy,
    )
    record_route_health_from_network_audit(network_audit, policy=wallet_policy)
    participant_eligibility = _build_participant_eligibility_audit(
        state_payload=execution_state_payload,
        instance_snapshot=instance_snapshot,
        requested_model_id=job.model_id,
        execution_model_id=execution_model_id,
        network_audit=network_audit,
    )
    network_audit["participantEligibility"] = participant_eligibility
    if not participant_eligibility["canSettle"]:
        message = (
            "Cannot settle execution reward: worker participant eligibility failed: "
            + "; ".join(participant_eligibility["fatalReasons"])
        )
        job.status = "failed"
        job.last_error = message
        update_job_intent(job, wallet_policy)
        raise RuntimeError(message)
    participants = _settlement_participants_for_reward(
        participants,
        network_audit=network_audit,
    )
    response_usage = _extract_llm_token_usage(response)
    proof_usage = _extract_cai_owned_transport_token_usage(network_audit)
    usage, token_usage_source, token_usage_audit = _merge_llm_token_usage_audit(
        response_usage=response_usage,
        proof_usage=proof_usage,
    )
    actual_compute_cost_atomic = job.requested_compute_cost_atomic
    usage_priced = False
    if (
        job.pricing_basis == "llm_tokens"
        and usage is not None
        and job.pricing_final_multiplier_bps is not None
    ):
        actual_price = calculate_token_priced_cost(
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            final_multiplier_bps=job.pricing_final_multiplier_bps,
            money_policy=active_money_policy,
            input_token_price_atomic=job.pricing_input_token_price_atomic,
            output_token_price_atomic=job.pricing_output_token_price_atomic,
            floor_atomic=job.pricing_floor_atomic,
            cap_atomic=job.pricing_cap_atomic,
        )
        actual_compute_cost_atomic = actual_price.compute_cost_atomic
        usage_priced = True

    decision = reservation_decision
    if actual_compute_cost_atomic != job.requested_compute_cost_atomic:
        decision = plan_funding(
            ledger=ledger,
            wallet=funding_wallet,
            compute_cost_atomic=actual_compute_cost_atomic,
            payment_preference=PaymentPreference(job.payment_preference),
            money_policy=active_money_policy,
            wallet_policy=wallet_policy,
            reserve_limit_identity_keys=tuple(job.reserve_limit_identity_keys or []),
            reserve_client_hash=job.reserve_client_ip_hash,
        )
        if not decision.can_fund:
            job.status = "failed"
            job.last_error = (
                "Execution completed, but final token-priced settlement could not be funded: "
                f"{decision.reason}"
            )
            update_job_intent(job, wallet_policy)
            raise ValueError(job.last_error)

    reservation_surplus_atomic = max(
        job.requested_compute_cost_atomic - decision.fee_quote.compute_cost_atomic,
        0,
    )
    worker_payouts = _distribute_worker_reward(
        decision.fee_quote.worker_reward_atomic,
        participants,
    )
    _sync_worker_reward_bindings_from_cai(
        execution_cai_url,
        worker_payouts,
        wallet_policy,
    )
    if decision.fee_quote.worker_reward_atomic > 0 and not worker_payouts:
        message = (
            "Cannot settle execution reward: CAI did not report any worker "
            "participants for the completed job."
        )
        job.status = "failed"
        job.last_error = message
        update_job_intent(job, wallet_policy)
        raise RuntimeError(message)

    receipt = ExecutionReceipt(
        receipt_id=secrets.token_hex(12),
        created_at=_now_iso(),
        job_id=job.job_id,
        cai_url=execution_cai_url,
        model_id=job.model_id,
        execution_model_id=execution_model_id,
        response_id=response.get("id"),
        finish_reason=_extract_finish_reason(response),
        output_text=_extract_output_text(response),
        instance_id=instance_snapshot["instance_id"] if instance_snapshot else None,
        pricing_mode=job.pricing_mode,
        pricing_basis=job.pricing_basis,
        prompt_tokens=(usage["prompt_tokens"] if usage is not None else None),
        completion_tokens=(usage["completion_tokens"] if usage is not None else None),
        total_tokens=(usage["total_tokens"] if usage is not None else None),
        reserved_prompt_tokens=job.reserved_prompt_tokens,
        reserved_completion_tokens=job.reserved_completion_tokens,
        reserved_compute_cost_atomic=job.requested_compute_cost_atomic,
        actual_compute_cost_atomic=decision.fee_quote.compute_cost_atomic,
        reservation_surplus_atomic=reservation_surplus_atomic,
        usage_priced=usage_priced,
        token_usage_source=token_usage_source,
        token_usage_audit=token_usage_audit,
        worker_payouts=worker_payouts,
        raw_response=response,
        network_audit=network_audit,
    )
    receipts = list_execution_receipts(wallet_policy)
    receipts.append(receipt)
    save_execution_receipts(receipts, wallet_policy)

    state_payload = _load_cai_state_payload(job.cai_url)
    settlement = record_funding_settlement(
        source_wallet_id=wallet.wallet_id,
        source_wallet_address=wallet.address,
        decision=decision,
        note=f"CAI job {job.job_id} completed",
        money_policy=active_money_policy,
        policy=wallet_policy,
        state_payload=state_payload,
        cai_url=job.cai_url,
    )
    payout_records = record_worker_payouts(
        settlement_id=settlement.settlement_id,
        receipt_id=receipt.receipt_id,
        model_id=receipt.execution_model_id,
        participants=receipt.worker_payouts,
        money_policy=active_money_policy,
        policy=wallet_policy,
    )
    record_settlement_execution_audit(
        settlement_id=settlement.settlement_id,
        receipt_id=receipt.receipt_id,
        job_id=job.job_id,
        model_id=receipt.model_id,
        execution_model_id=receipt.execution_model_id,
        pricing_mode=receipt.pricing_mode,
        pricing_basis=receipt.pricing_basis,
        prompt_tokens=receipt.prompt_tokens,
        completion_tokens=receipt.completion_tokens,
        total_tokens=receipt.total_tokens,
        reserved_prompt_tokens=receipt.reserved_prompt_tokens,
        reserved_completion_tokens=receipt.reserved_completion_tokens,
        reserved_compute_cost_atomic=receipt.reserved_compute_cost_atomic,
        actual_compute_cost_atomic=receipt.actual_compute_cost_atomic,
        reservation_surplus_atomic=receipt.reservation_surplus_atomic,
        usage_priced=receipt.usage_priced,
        token_usage_source=receipt.token_usage_source,
        token_usage_audit=receipt.token_usage_audit,
        network_audit=receipt.network_audit,
        worker_payouts=payout_records,
        policy=wallet_policy,
    )
    sign_settlement_envelope(
        settlement.settlement_id,
        policy=wallet_policy,
        money_policy=active_money_policy,
    )
    apply_local_validator_attestation(
        settlement_id=settlement.settlement_id,
        accepted_note="Local bonded validator accepted CAI execution receipt.",
        money_policy=active_money_policy,
        wallet_policy=wallet_policy,
        state_payload=state_payload,
        cai_url=job.cai_url,
        fallback_validator_address=wallet.address,
    )
    remote_committee_attestation_audit: dict[str, Any] = {
        "settlementId": settlement.settlement_id,
    }
    request_remote_committee_attestations(
        settlement_id=settlement.settlement_id,
        accepted_note="Remote committee validator accepted CAI execution receipt.",
        wallet_policy=wallet_policy,
        cai_url=job.cai_url,
        audit=remote_committee_attestation_audit,
    )
    settlement_warning: str | None = None
    settlement_tail_audit: dict[str, Any] = {
        "settlementId": settlement.settlement_id,
        "remoteCommitteeAttestation": remote_committee_attestation_audit,
        "canonicalChainApply": {"status": "pending"},
        "chainPush": {"status": "skipped"},
    }
    try:
        _apply_settlement_after_canonical_chain_sync(
            settlement_id=settlement.settlement_id,
            wallet_policy=wallet_policy,
            money_policy=active_money_policy,
        )
        settlement_tail_audit["canonicalChainApply"] = {"status": "ok"}
        push_state_payload = state_payload or _load_cai_state_payload(job.cai_url)
        settlement_tail_audit["chainPush"] = _run_chain_push_audit(
            state_payload=push_state_payload,
            cai_url=job.cai_url,
            wallet_policy=wallet_policy,
        )
    except Exception as exc:
        # The model response and receipt already exist at this point. Keep chat
        # delivery successful and let later repair/finality passes reconcile the
        # settlement tail instead of surfacing a 400 to the user.
        settlement_warning = str(exc)
        settlement_tail_audit["canonicalChainApply"] = _audit_error_status(exc)

    network_audit["settlementTail"] = settlement_tail_audit
    receipt.network_audit = network_audit
    try:
        update_execution_receipt(receipt, wallet_policy)
    except Exception as exc:
        warning = f"receipt network audit update failed: {exc}"
        settlement_warning = (
            f"{settlement_warning}; {warning}" if settlement_warning else warning
        )

    job.status = "completed"
    job.receipt_id = receipt.receipt_id
    job.settlement_id = settlement.settlement_id
    job.last_error = None
    update_job_intent(job, wallet_policy)

    append_journal_entry(
        JournalEntry(
            entry_id=secrets.token_hex(12),
            event_type="execution_receipt_recorded",
            created_at=_now_iso(),
            wallet_id=wallet.wallet_id,
            amount_atomic=decision.fee_quote.compute_cost_atomic,
            note=(
                f"Receipt {receipt.receipt_id} recorded for job {job.job_id} "
                f"with {len(receipt.worker_payouts)} worker payout(s)."
                + (
                    f" Reservation surplus released logically: {reservation_surplus_atomic} atomic."
                    if reservation_surplus_atomic > 0
                    else ""
                )
            ),
        ),
        wallet_policy,
    )
    if settlement_warning:
        append_journal_entry(
            JournalEntry(
                entry_id=secrets.token_hex(12),
                event_type="execution_settlement_warning",
                created_at=_now_iso(),
                wallet_id=wallet.wallet_id,
                note=(
                    f"Settlement tail warning for receipt {receipt.receipt_id} "
                    f"and settlement {settlement.settlement_id}: {settlement_warning}"
                ),
            ),
            wallet_policy,
        )
    return job, receipt


def _apply_settlement_after_canonical_chain_sync(
    *,
    settlement_id: str,
    wallet_policy: WalletPolicy | None = None,
    money_policy: MoneyPolicy | None = None,
) -> None:
    active_money_policy = money_policy or MoneyPolicy()
    settlement = resolve_settlement(settlement_id, wallet_policy)
    if settlement is None:
        return

    canonical_chain_recorded = bool(
        chain_settlement_history(settlement_id, policy=wallet_policy, limit=1)
    )
    if settlement.status == "finalized" and canonical_chain_recorded:
        apply_finalized_settlement(
            settlement_id=settlement_id,
            policy=wallet_policy,
            money_policy=active_money_policy,
        )
        settlement = resolve_settlement(settlement_id, wallet_policy) or settlement

    if settlement.status == "applied":
        reconcile_worker_payouts(wallet_policy)
        record_chain_entries_for_finalized_settlements(
            policy=wallet_policy,
            money_policy=active_money_policy,
        )


def repair_local_worker_reward_state(
    *,
    cai_url: str,
    money_policy: MoneyPolicy | None = None,
    wallet_policy: WalletPolicy | None = None,
    state_payload: dict[str, Any] | None = None,
    timeout_sec: int = 5,
) -> dict[str, Any]:
    active_money_policy = money_policy or MoneyPolicy()
    resolved_cai_url = cai_url.rstrip("/")
    resolved_state_payload = state_payload or _load_cai_state_payload(resolved_cai_url)

    peer_sync_audit: dict[str, Any] = {
        "attempted": False,
        "statePayloadAvailable": resolved_state_payload is not None,
        "validatorSet": {"status": "skipped"},
        "chain": {"status": "skipped"},
    }
    if resolved_state_payload is not None:
        peer_sync_audit["attempted"] = True
        peer_sync_audit["checkedAt"] = _now_iso()
        try:
            validator_result = sync_validator_set_from_cai_peers(
                state_payload=resolved_state_payload,
                cai_url=resolved_cai_url,
                policy=wallet_policy,
                timeout_sec=timeout_sec,
            )
            peer_sync_audit["validatorSet"] = {
                "status": "ok",
                **_validator_sync_result_audit(validator_result),
            }
        except Exception as exc:
            peer_sync_audit["validatorSet"] = _audit_error_status(exc)
        try:
            chain_result = sync_chain_from_cai_peers(
                state_payload=resolved_state_payload,
                cai_url=resolved_cai_url,
                policy=wallet_policy,
                timeout_sec=min(float(timeout_sec), 1.0),
            )
            peer_sync_audit["chain"] = {
                "status": "ok",
                **_chain_sync_result_audit(chain_result),
            }
        except Exception as exc:
            peer_sync_audit["chain"] = _audit_error_status(exc)

    committee_backfilled = 0
    local_attestations = 0
    remote_attestations = 0
    settlements_applied = 0
    attestation_repair_audit: dict[str, Any] = {
        "local": [],
        "remote": [],
    }
    for settlement in list_settlements(wallet_policy):
        had_committee = bool(settlement.committee_validator_ids)
        refreshed = ensure_settlement_committee(
            settlement.settlement_id,
            policy=wallet_policy,
            money_policy=active_money_policy,
        )
        current = refreshed or settlement
        if not had_committee and current.committee_validator_ids:
            committee_backfilled += 1
        if current.status == "rejected":
            current = (
                reset_retryable_settlement_rejection(
                    current.settlement_id,
                    policy=wallet_policy,
                    money_policy=active_money_policy,
                )
                or current
            )

        current = (
            _prepare_pending_settlement_for_attestation(
                current,
                money_policy=active_money_policy,
                wallet_policy=wallet_policy,
            )
            or current
        )
        attestation_ready = _settlement_attestation_ready(
            current,
            wallet_policy=wallet_policy,
        )

        if (
            attestation_ready
            and current.status == "pending"
            and current.committee_validator_ids
        ):
            attestation_status = get_validator_attestation_status(
                policy=wallet_policy,
                state_payload=resolved_state_payload,
                cai_url=resolved_cai_url,
            )
            if (
                attestation_status.can_attest
                and attestation_status.validator_id in current.committee_validator_ids
            ):
                try:
                    attestation = apply_local_validator_attestation(
                        settlement_id=current.settlement_id,
                        accepted_note=(
                            "Local bonded validator accepted repaired CAI settlement."
                        ),
                        money_policy=active_money_policy,
                        wallet_policy=wallet_policy,
                        state_payload=resolved_state_payload,
                        cai_url=resolved_cai_url,
                    )
                    if attestation is not None and getattr(attestation, "accepted", False):
                        local_attestations += 1
                except Exception as exc:
                    attestation_repair_audit["local"].append(
                        {
                            "settlementId": current.settlement_id,
                            **_audit_error_status(exc),
                        }
                    )
                    _log_best_effort_failure(
                        "repair local validator attestation",
                        exc,
                    )
                current = resolve_settlement(current.settlement_id, wallet_policy) or current

        if (
            attestation_ready
            and current.status == "pending"
            and current.committee_validator_ids
        ):
            remote_audit: dict[str, Any] = {
                "settlementId": current.settlement_id,
            }
            try:
                responses = request_remote_committee_attestations(
                    settlement_id=current.settlement_id,
                    accepted_note=(
                        "Remote committee validator accepted CAI execution receipt."
                    ),
                    wallet_policy=wallet_policy,
                    cai_url=resolved_cai_url,
                    audit=remote_audit,
                )
                remote_attestations += len(responses)
            except Exception as exc:
                remote_audit.update(_audit_error_status(exc))
                _log_best_effort_failure(
                    "repair remote committee attestation",
                    exc,
                )
            attestation_repair_audit["remote"].append(remote_audit)
            current = resolve_settlement(current.settlement_id, wallet_policy) or current

        canonical_chain_recorded = bool(
            chain_settlement_history(current.settlement_id, policy=wallet_policy, limit=1)
        )
        local_validator_can_apply = bool(
            load_or_create_node_config(wallet_policy).validator_enabled
        )
        if current.status == "finalized" and (
            canonical_chain_recorded or local_validator_can_apply
        ):
            applied = apply_finalized_settlement(
                settlement_id=current.settlement_id,
                policy=wallet_policy,
                money_policy=active_money_policy,
            )
            if applied is not None and applied.status == "applied":
                settlements_applied += 1

    local_validator_can_apply = bool(
        load_or_create_node_config(wallet_policy).validator_enabled
    )
    reconciled = reconcile_worker_payouts(wallet_policy)
    chain_entries_recorded = record_chain_entries_for_finalized_settlements(
        policy=wallet_policy,
        money_policy=active_money_policy,
        only_if_chain_recorded=not local_validator_can_apply,
    )
    return {
        "committeeBackfilled": committee_backfilled,
        "localAttestations": local_attestations,
        "remoteAttestations": remote_attestations,
        "settlementsApplied": settlements_applied,
        "payoutsReconciled": len(reconciled),
        "chainEntriesRecorded": chain_entries_recorded,
        "peerSync": peer_sync_audit,
        "attestationRepair": attestation_repair_audit,
    }


def _settlement_attestation_ready(
    settlement: Any | None,
    *,
    wallet_policy: WalletPolicy | None = None,
) -> bool:
    if settlement is None:
        return False
    audit = dict(getattr(settlement, "balance_audit", {}) or {})
    execution_audit = audit.get("execution")
    if not isinstance(execution_audit, dict) or not execution_audit:
        return False
    if not list_worker_payouts(
        settlement_id=getattr(settlement, "settlement_id", None),
        policy=wallet_policy,
    ):
        return False
    envelope = audit.get("signed_envelope")
    return isinstance(envelope, dict) and envelope.get("status") == "signed"


def _prepare_pending_settlement_for_attestation(
    settlement: Any | None,
    *,
    money_policy: MoneyPolicy | None = None,
    wallet_policy: WalletPolicy | None = None,
) -> Any | None:
    if settlement is None or getattr(settlement, "status", None) != "pending":
        return settlement
    if not list(getattr(settlement, "committee_validator_ids", []) or []):
        return settlement
    if _settlement_attestation_ready(settlement, wallet_policy=wallet_policy):
        return settlement

    audit = dict(getattr(settlement, "balance_audit", {}) or {})
    execution_audit = audit.get("execution")
    has_execution_audit = isinstance(execution_audit, dict) and bool(execution_audit)
    has_worker_payouts = bool(
        list_worker_payouts(
            settlement_id=getattr(settlement, "settlement_id", None),
            policy=wallet_policy,
        )
    )
    if not has_execution_audit or not has_worker_payouts:
        return settlement

    return (
        sign_settlement_envelope(
            getattr(settlement, "settlement_id"),
            policy=wallet_policy,
            money_policy=money_policy,
        )
        or settlement
    )


def apply_local_validator_attestation(
    *,
    settlement_id: str,
    accepted_note: str,
    money_policy: MoneyPolicy | None = None,
    wallet_policy: WalletPolicy | None = None,
    state_payload: dict[str, Any] | None = None,
    cai_url: str | None = None,
    CAI_url: str | None = None,
    fallback_validator_address: str | None = None,
) -> Any | None:
    active_money_policy = money_policy or MoneyPolicy()
    node_config = load_or_create_node_config(wallet_policy)
    if not node_config.validator_enabled:
        return None

    attestation_status = get_validator_attestation_status(
        policy=wallet_policy,
        state_payload=state_payload,
        cai_url=_coalesce_cai_url(cai_url, CAI_url),
    )
    validator_id = (
        attestation_status.validator_id
        or node_config.validator_address
        or fallback_validator_address
    )
    if validator_id is None:
        return None

    settlement = resolve_settlement(settlement_id, wallet_policy)
    committee_validator_ids = list(getattr(settlement, "committee_validator_ids", []) or [])
    if committee_validator_ids and validator_id not in committee_validator_ids:
        return None

    if attestation_status.can_attest:
        try:
            return record_validator_attestation(
                settlement_id=settlement_id,
                validator_id=validator_id,
                accepted=True,
                note=accepted_note,
                policy=wallet_policy,
            )
        except ConflictingAttestationError as exc:
            jailed_config = jail_validator(
                reason=(
                    "Validator produced a conflicting settlement attestation and was slashed."
                ),
                money_policy=active_money_policy,
                policy=wallet_policy,
                slash_bps=active_money_policy.validator_conflicting_attestation_slash_bps,
            )
            record_validator_evidence(
                validator_id=validator_id,
                evidence_type="conflicting_attestation",
                settlement_id=settlement_id,
                attestation_id=exc.existing_attestation.attestation_id,
                conflicting_attestation_id=exc.existing_attestation.attestation_id,
                slash_atomic=jailed_config.validator_last_slash_atomic,
                jailed=True,
                note=str(exc),
                policy=wallet_policy,
            )
            return None

    if getattr(attestation_status, "passive_replica", False):
        return None

    try:
        rejected_attestation = record_validator_attestation(
            settlement_id=settlement_id,
            validator_id=validator_id,
            accepted=False,
            note=attestation_status.reason,
            policy=wallet_policy,
        )
        jailed_config = jail_validator(
            reason=(
                "Validator failed settlement attestation eligibility check: "
                f"{attestation_status.reason}"
            ),
            money_policy=active_money_policy,
            policy=wallet_policy,
        )
        record_validator_evidence(
            validator_id=validator_id,
            evidence_type="eligibility_failure",
            settlement_id=settlement_id,
            attestation_id=rejected_attestation.attestation_id,
            slash_atomic=jailed_config.validator_last_slash_atomic,
            jailed=True,
            note=attestation_status.reason,
            policy=wallet_policy,
        )
        return rejected_attestation
    except ConflictingAttestationError as exc:
        jailed_config = jail_validator(
            reason=(
                "Validator produced a conflicting settlement attestation and was slashed."
            ),
            money_policy=active_money_policy,
            policy=wallet_policy,
            slash_bps=active_money_policy.validator_conflicting_attestation_slash_bps,
        )
        record_validator_evidence(
            validator_id=validator_id,
            evidence_type="conflicting_attestation",
            settlement_id=settlement_id,
            attestation_id=exc.existing_attestation.attestation_id,
            conflicting_attestation_id=exc.existing_attestation.attestation_id,
            slash_atomic=jailed_config.validator_last_slash_atomic,
            jailed=True,
            note=str(exc),
            policy=wallet_policy,
        )
        return None


def request_remote_committee_attestations(
    *,
    settlement_id: str,
    accepted_note: str,
    wallet_policy: WalletPolicy | None = None,
    cai_url: str | None = None,
    CAI_url: str | None = None,
    audit: dict[str, Any] | None = None,
) -> list[Any]:
    settlement = resolve_settlement(settlement_id, wallet_policy)
    if settlement is None:
        raise ValueError(f"Settlement '{settlement_id}' not found.")
    committee_validator_ids = list(getattr(settlement, "committee_validator_ids", []) or [])
    if audit is not None:
        audit.setdefault("settlementId", settlement.settlement_id)
        audit.setdefault("attempted", bool(committee_validator_ids))
        audit.setdefault("validatorSetSync", {"status": "skipped"})
        audit.setdefault("validators", [])
        audit.setdefault("acceptedResponses", 0)
    if not committee_validator_ids:
        return []

    resolved_cai_url = _coalesce_cai_url(cai_url, CAI_url)
    if resolved_cai_url is None:
        raise ValueError("CAI URL is required.")
    state_payload = _load_cai_state_payload(resolved_cai_url)
    if state_payload is not None:
        try:
            validator_sync_result = sync_validator_set_from_cai_peers(
                state_payload=state_payload,
                cai_url=resolved_cai_url,
                policy=wallet_policy,
            )
            if audit is not None:
                audit["validatorSetSync"] = {
                    "status": "ok",
                    **_validator_sync_result_audit(validator_sync_result),
                }
        except Exception as exc:
            if audit is not None:
                audit["validatorSetSync"] = _audit_error_status(exc)
            _log_best_effort_failure("remote committee validator set sync", exc)
    elif audit is not None:
        audit["validatorSetSync"] = {
            "status": "skipped",
            "reason": "state_payload_unavailable",
        }

    node_config = load_or_create_node_config(wallet_policy)
    local_validator_id = (
        str(node_config.validator_address).strip().lower()
        if getattr(node_config, "validator_address", None)
        else None
    )
    settlement_proposal = export_settlement_proposal_payload(
        settlement.settlement_id,
        policy=wallet_policy,
    )
    responses: list[Any] = []
    validator_audits = (
        audit.setdefault("validators", []) if audit is not None else None
    )
    for validator_id in committee_validator_ids:
        normalized_validator_id = str(validator_id).strip().lower()
        validator_audit: dict[str, Any] | None = None
        if isinstance(validator_audits, list):
            validator_audit = {
                "validatorId": normalized_validator_id,
                "status": "pending",
            }
            validator_audits.append(validator_audit)
        if local_validator_id is not None and normalized_validator_id == local_validator_id:
            if validator_audit is not None:
                validator_audit.update(
                    {"status": "skipped", "reason": "local_validator"}
                )
            continue
        record = get_validator_record(normalized_validator_id, wallet_policy)
        if record is None:
            if validator_audit is not None:
                validator_audit.update(
                    {"status": "skipped", "reason": "validator_record_missing"}
                )
            continue
        record_source_url = resolve_validator_peer_url(
            source_url=getattr(record, "source_url", None),
            advertised_api_host=getattr(record, "advertised_api_host", None),
        )
        if not record_source_url:
            if validator_audit is not None:
                validator_audit.update(
                    {"status": "skipped", "reason": "validator_endpoint_missing"}
                )
            continue
        endpoint = _validator_attestation_endpoint(record_source_url)
        if validator_audit is not None:
            validator_audit["endpoint"] = endpoint
        try:
            payload = _post_json(
                endpoint,
                {
                    "settlement_id": settlement.settlement_id,
                    "accepted_note": accepted_note,
                    "committee_validator_ids": list(settlement.committee_validator_ids),
                    "committee_bonded_atomic_by_validator_id": dict(
                        settlement.committee_bonded_atomic_by_validator_id
                    ),
                    "committee_total_bonded_atomic": settlement.committee_total_bonded_atomic,
                    "committee_quorum_bond_atomic": settlement.committee_quorum_bond_atomic,
                    "committee_selection_seed": settlement.committee_selection_seed,
                    "committee_selection_mode": settlement.committee_selection_mode,
                    "settlement_proposal": settlement_proposal,
                },
                timeout=30,
            )
        except Exception as exc:
            if validator_audit is not None:
                validator_audit.update(_audit_error_status(exc))
            _log_best_effort_failure("remote committee validator request", exc)
            continue

        if not payload.get("attested"):
            if validator_audit is not None:
                validator_audit.update(
                    {
                        "status": "skipped",
                        "reason": str(
                            payload.get("reason")
                            or payload.get("message")
                            or "validator_not_attested"
                        ),
                    }
                )
            continue
        try:
            attestation = record_validator_attestation(
                settlement_id=settlement.settlement_id,
                validator_id=str(
                    payload.get("validatorId") or normalized_validator_id
                ).strip().lower(),
                accepted=bool(payload.get("accepted")),
                note=str(
                    payload.get("note")
                    or (accepted_note if payload.get("accepted") else "Remote validator rejected settlement.")
                ),
                policy=wallet_policy,
                apply_on_finalize=False,
            )
        except ConflictingAttestationError as exc:
            if validator_audit is not None:
                validator_audit.update(
                    {
                        "status": "failed",
                        "errorType": type(exc).__name__,
                        "message": str(exc),
                    }
                )
            _log_best_effort_failure(
                "remote committee validator attestation record",
                exc,
            )
            continue
        if validator_audit is not None:
            validator_audit.update(
                {
                    "status": "ok",
                    "accepted": bool(payload.get("accepted")),
                    "recordedValidatorId": str(
                        getattr(attestation, "validator_id", normalized_validator_id)
                    ),
                    "chainMerge": {"status": "skipped"},
                }
            )
        chain_payload = payload.get("chain")
        if isinstance(chain_payload, dict):
            try:
                merge_remote_chain_payload(chain_payload, policy=wallet_policy)
                if validator_audit is not None:
                    validator_audit["chainMerge"] = {"status": "ok"}
            except Exception as exc:
                if validator_audit is not None:
                    validator_audit["chainMerge"] = _audit_error_status(exc)
                _log_best_effort_failure(
                    "remote committee validator chain merge",
                    exc,
                )
        responses.append(attestation)
    if audit is not None:
        audit["acceptedResponses"] = len(responses)
        audit["status"] = "ok" if responses else "no_attestations"
    return responses


def _submit_text_job_to_cai(
    cai_url: str,
    model_id: str,
    prompt: str,
    *,
    timeout_sec: int = 1800,
    request_payload_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_payload = _build_text_job_request_payload(
        model_id,
        prompt,
        request_payload_override=request_payload_override,
    )
    last_http_error: HTTPError | None = None
    for candidate_url in _candidate_cai_chat_base_urls(cai_url):
        request = Request(
            url=f"{candidate_url.rstrip('/')}/v1/chat/completions",
            data=json.dumps(request_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_sec) as response:
                return _decode_json_http_payload(
                    response.read(),
                    context=(
                        "CAI text generation endpoint "
                        f"{candidate_url.rstrip('/')}/v1/chat/completions"
                    ),
                )
        except HTTPError as exc:
            if exc.code == 404:
                last_http_error = exc
                continue
            detail = _http_error_detail(exc)
            if detail:
                raise RuntimeError(detail) from exc
            raise
        except IncompleteRead as exc:
            raise RuntimeError(
                "CAI text generation endpoint closed the response before returning a complete JSON body."
            ) from exc
    if last_http_error is not None:
        raise last_http_error
    raise RuntimeError("No CAI chat completion endpoint candidates were available.")


def _load_cai_state_payload(
    cai_url: str,
    *,
    log_operation: str | None = None,
) -> dict[str, Any] | None:
    state_url = cai_url.rstrip("/") + "/state"
    try:
        with urlopen(state_url, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        if log_operation:
            _log_best_effort_failure(log_operation, exc)
        return None


def _candidate_cai_chat_base_urls(cai_url: str) -> list[str]:
    return _candidate_cai_chat_base_urls_impl(
        cai_url,
        load_cai_state_payload_func=_load_cai_state_payload,
    )


def _cai_api_urls_by_node_id(
    cai_url: str,
    state_payload: dict[str, Any],
) -> dict[str, list[str]]:
    return _cai_api_urls_by_node_id_impl(
        cai_url,
        state_payload,
        list_node_capabilities_func=list_node_capabilities,
    )


def _capability_records_by_node_id(
    *,
    accepted_model_ids: set[str],
    stale_after_seconds: int,
    require_verified_capabilities: bool,
) -> dict[str, NodeCapabilityRecord]:
    return _capability_records_by_node_id_impl(
        accepted_model_ids=accepted_model_ids,
        stale_after_seconds=stale_after_seconds,
        require_verified_capabilities=require_verified_capabilities,
        wallet_policy_factory=WalletPolicy,
        list_verified_worker_node_ids_func=list_verified_worker_node_ids,
        list_node_capabilities_func=list_node_capabilities,
    )


def _build_participant_eligibility_audit(
    *,
    state_payload: dict[str, Any] | None,
    instance_snapshot: dict[str, Any] | None,
    requested_model_id: str,
    execution_model_id: str,
    network_audit: dict[str, Any],
    stale_after_seconds: int = 300,
) -> dict[str, Any]:
    return _build_participant_eligibility_audit_impl(
        state_payload=state_payload,
        instance_snapshot=instance_snapshot,
        requested_model_id=requested_model_id,
        execution_model_id=execution_model_id,
        network_audit=network_audit,
        participant_node_ids_func=_participant_node_ids,
        worker_capability_verification_required_func=worker_capability_verification_required,
        capability_records_by_node_id_func=_capability_records_by_node_id,
        verified_worker_node_ids_func=list_verified_worker_node_ids,
        wallet_policy_factory=WalletPolicy,
        stale_after_seconds=stale_after_seconds,
    )


def _resolve_worker_execution_node_ids(cai_url: str, model_id: str) -> list[str] | None:
    audit = _resolve_worker_execution_node_audit(cai_url, model_id)
    if audit is None:
        return None
    eligible_node_ids = audit.get("eligibleNodeIds")
    if isinstance(eligible_node_ids, list) and eligible_node_ids:
        return sorted(str(node_id) for node_id in eligible_node_ids)
    if int(audit.get("checkedNodeCount") or 0) > 0:
        return []
    return None


def _resolve_worker_execution_node_audit(cai_url: str, model_id: str) -> dict[str, Any] | None:
    try:
        identities_payload = _get_json(
            f"{cai_url.rstrip('/')}/state/nodeIdentities",
            timeout=5,
        )
    except AssertionError:
        return None
    except Exception as exc:
        _log_best_effort_failure("worker execution node identity audit", exc)
        return None
    if not isinstance(identities_payload, dict) or not identities_payload:
        return None

    identities = identities_payload
    state_payload = {"nodeIdentities": identities}
    eligible_node_ids: list[str] = []
    checked_nodes = 0
    summary_urls = _cai_summary_urls_by_node_id(cai_url, state_payload)
    accepted_model_ids = _accepted_worker_model_ids(model_id)
    node_items: list[dict[str, Any]] = []
    now = datetime.now(tz=UTC)
    stale_after_seconds = _worker_identity_stale_after_seconds()
    require_verified_capabilities = worker_capability_verification_required()
    capability_records = _capability_records_by_node_id(
        accepted_model_ids=accepted_model_ids,
        stale_after_seconds=stale_after_seconds,
        require_verified_capabilities=require_verified_capabilities,
    )
    verified_worker_node_ids = set(capability_records)
    state_node_ids: set[str] = set()
    for node_id, identity in identities.items():
        normalized_node_id = str(node_id).strip()
        if not normalized_node_id:
            continue
        state_node_ids.add(normalized_node_id)
        identity = identities.get(node_id)
        worker_enabled, reward_address = _worker_identity_state(identity)
        allowed_model_ids = _identity_allowed_model_ids(identity)
        last_seen_at = _identity_last_seen_at(identity)
        last_seen_age_seconds: int | None = None
        parsed_last_seen = _parse_iso_datetime(last_seen_at)
        if parsed_last_seen is not None:
            last_seen_age_seconds = max(
                0,
                int((now - parsed_last_seen).total_seconds()),
            )
        summary_url = summary_urls.get(normalized_node_id)
        summary_checked = False
        summary_error: dict[str, str] | None = None
        if worker_enabled is None and summary_url:
            summary_checked = True
            try:
                summary_payload = _get_json(summary_url, timeout=5)
            except Exception as exc:
                summary_error = _audit_error_status(exc)
                _log_best_effort_failure(
                    f"worker summary fetch for node {normalized_node_id}",
                    exc,
                )
                summary_payload = {}
            worker = (
                summary_payload.get("worker")
                if isinstance(summary_payload, dict)
                and isinstance(summary_payload.get("worker"), dict)
                else {}
            )
            if "worker_enabled" in worker:
                worker_enabled = bool(worker.get("worker_enabled"))
            elif "workerEnabled" in worker:
                worker_enabled = bool(worker.get("workerEnabled"))
            reward_address = reward_address or (
                str(
                    worker.get("worker_reward_address")
                    or worker.get("workerRewardAddress")
                    or ""
                ).strip()
                or None
            )
            if allowed_model_ids is None:
                raw_allowed_model_ids = (
                    worker.get("worker_allowed_model_ids")
                    or worker.get("workerAllowedModelIds")
                    or worker.get("allowed_model_ids")
                    or worker.get("allowedModelIds")
                )
                allowed_model_ids = _identity_allowed_model_ids(
                    {"workerAllowedModelIds": raw_allowed_model_ids}
                )

        if worker_enabled is not None or summary_checked:
            checked_nodes += 1

        model_allowed = _worker_model_allowed(allowed_model_ids, accepted_model_ids)
        reasons: list[str] = []
        if worker_enabled is False:
            reasons.append("worker mode is disabled")
        elif worker_enabled is None:
            reasons.append("worker mode is unknown")
        if not reward_address:
            reasons.append("worker reward address is missing")
        if model_allowed is False:
            allowed_text = ", ".join(allowed_model_ids or [])
            accepted_text = ", ".join(sorted(accepted_model_ids))
            reasons.append(
                f"model is not in worker allow-list: allowed=[{allowed_text}], accepted=[{accepted_text}]"
            )
        verified_capability = (
            normalized_node_id in verified_worker_node_ids
            if require_verified_capabilities
            else None
        )
        if require_verified_capabilities and not verified_capability:
            reasons.append("worker capability is not verified")
        if (
            last_seen_age_seconds is not None
            and last_seen_age_seconds > stale_after_seconds
        ):
            reasons.append(
                f"worker identity is stale: last seen {last_seen_age_seconds}s ago"
            )

        eligible = bool(
            worker_enabled
            and reward_address
            and model_allowed is not False
            and not reasons
        )
        if eligible:
            eligible_node_ids.append(normalized_node_id)
        node_items.append(
            {
                "nodeId": normalized_node_id,
                "eligible": eligible,
                "identityKnown": True,
                "capabilityBacked": False,
                "workerEnabled": worker_enabled,
                "workerRewardAddressKnown": bool(reward_address),
                "verifiedCapability": verified_capability,
                "allowedModelIds": allowed_model_ids,
                "modelAllowed": model_allowed,
                "lastSeenAt": last_seen_at,
                "lastSeenAgeSeconds": last_seen_age_seconds,
                "summaryStatus": (
                    "failed"
                    if summary_error
                    else ("checked" if summary_checked else "skipped")
                ),
                "summaryError": summary_error,
                "reasons": reasons,
            }
        )

    for normalized_node_id, record in sorted(capability_records.items()):
        if normalized_node_id in state_node_ids:
            continue
        identity = _capability_identity_from_record(record)
        worker_enabled, reward_address = _worker_identity_state(identity)
        allowed_model_ids = _identity_allowed_model_ids(identity)
        last_seen_at = _identity_last_seen_at(identity)
        last_seen_age_seconds: int | None = None
        parsed_last_seen = _parse_iso_datetime(last_seen_at)
        if parsed_last_seen is not None:
            last_seen_age_seconds = max(
                0,
                int((now - parsed_last_seen).total_seconds()),
            )
        checked_nodes += 1

        model_allowed = _worker_model_allowed(allowed_model_ids, accepted_model_ids)
        reasons: list[str] = []
        warnings = [
            "node identity is missing from live state; using validator-attested capability record"
        ]
        verified_capability = (
            True if require_verified_capabilities else bool(record.worker_verified)
        )
        if not verified_capability:
            reasons.append("worker capability is not verified")
        if worker_enabled is False:
            reasons.append("worker mode is disabled")
        elif worker_enabled is None:
            reasons.append("worker mode is unknown")
        if not reward_address:
            reasons.append("worker reward address is missing")
        if model_allowed is False:
            allowed_text = ", ".join(allowed_model_ids or [])
            accepted_text = ", ".join(sorted(accepted_model_ids))
            reasons.append(
                f"model is not in worker allow-list: allowed=[{allowed_text}], accepted=[{accepted_text}]"
            )
        if (
            last_seen_age_seconds is not None
            and last_seen_age_seconds > stale_after_seconds
        ):
            reasons.append(
                f"worker identity is stale: last seen {last_seen_age_seconds}s ago"
            )

        eligible = bool(
            worker_enabled
            and reward_address
            and model_allowed is not False
            and not reasons
        )
        if eligible:
            eligible_node_ids.append(normalized_node_id)
        node_items.append(
            {
                "nodeId": normalized_node_id,
                "eligible": eligible,
                "identityKnown": False,
                "capabilityBacked": True,
                "workerEnabled": worker_enabled,
                "workerRewardAddressKnown": bool(reward_address),
                "verifiedCapability": verified_capability,
                "allowedModelIds": allowed_model_ids,
                "modelAllowed": model_allowed,
                "lastSeenAt": last_seen_at,
                "lastSeenAgeSeconds": last_seen_age_seconds,
                "warnings": warnings,
                "reasons": reasons,
            }
        )

    return {
        "schemaVersion": 1,
        "modelId": model_id,
        "acceptedModelIds": sorted(accepted_model_ids),
        "checkedNodeCount": checked_nodes,
        "eligibleNodeIds": sorted(set(eligible_node_ids)),
        "nodes": node_items,
    }


def _worker_node_ids_from_audit(audit: dict[str, Any] | None) -> list[str] | None:
    return _worker_node_ids_from_audit_impl(audit)


def _task_level_transport_gguf_shard_compatibility(
    model_id: str,
    *,
    wallet_policy: WalletPolicy | None = None,
):
    return _task_level_transport_gguf_shard_compatibility_impl(
        model_id,
        wallet_policy=wallet_policy,
        curated_model_for_id_func=curated_model_for_id,
        select_model_package_manifest_for_model_func=select_model_package_manifest_for_model,
        gguf_shard_compatibility_func=gguf_shard_compatibility,
    )


def _task_level_transport_effective_executor_count(
    model_id: str,
    *,
    requested_executor_count: int | None = None,
    wallet_policy: WalletPolicy | None = None,
) -> int:
    return _task_level_transport_effective_executor_count_impl(
        model_id,
        requested_executor_count=requested_executor_count,
        wallet_policy=wallet_policy,
        task_level_transport_executor_count_func=_task_level_transport_executor_count,
        gguf_shard_compatibility_func=_task_level_transport_gguf_shard_compatibility,
        total_layer_count_func=_task_level_transport_total_layer_count,
        curated_model_for_id_func=curated_model_for_id,
    )


def _task_level_transport_total_layer_count(
    model_id: str,
    *,
    executor_count: int,
    wallet_policy: WalletPolicy | None = None,
) -> int:
    return _task_level_transport_total_layer_count_impl(
        model_id,
        executor_count=executor_count,
        wallet_policy=wallet_policy,
        curated_model_for_id_func=curated_model_for_id,
        select_model_package_manifest_for_model_func=select_model_package_manifest_for_model,
        optional_int_value_func=_optional_int_value,
        optional_int_field_value_func=_optional_int_field_value,
    )


def _task_level_transport_llm_runtime_metadata(
    model_id: str,
    *,
    total_layer_count: int,
    wallet_policy: WalletPolicy | None = None,
) -> dict[str, Any] | None:
    return _task_level_transport_llm_runtime_metadata_impl(
        model_id,
        total_layer_count=total_layer_count,
        wallet_policy=wallet_policy,
        curated_model_for_id_func=curated_model_for_id,
        select_model_package_manifest_for_model_func=select_model_package_manifest_for_model,
        optional_int_value_func=_optional_int_value,
        optional_int_field_value_func=_optional_int_field_value,
    )


def _task_level_transport_planned_shard_ranges(
    cai_url: str,
    model_id: str,
    *,
    executor_node_ids: list[str],
    total_layer_count: int,
) -> tuple[list[str], list[dict[str, Any]] | None]:
    return _task_level_transport_planned_shard_ranges_impl(
        cai_url,
        model_id,
        executor_node_ids=executor_node_ids,
        total_layer_count=total_layer_count,
        resolve_cai_instance_create_payload_for_nodes_func=_resolve_cai_instance_create_payload_for_nodes,
        snapshot_from_instance_definition_func=_snapshot_from_instance_definition,
        optional_int_field_value_func=_optional_int_field_value,
        is_private_curated_model_id_func=is_private_curated_model_id,
    )


def _try_execute_task_level_transport_job(
    job: JobIntent,
    *,
    execution_cai_url: str,
    execution_model_id: str,
    state_payload: dict[str, Any] | None,
    request_timeout_sec: int,
    request_payload_override: dict[str, Any] | None,
    wallet_policy: WalletPolicy | None,
) -> dict[str, Any] | None:
    state = state_payload or _load_cai_state_payload(execution_cai_url)
    if not isinstance(state, dict):
        return None
    requester_node_id = (
        str(job.requester_node_id or "").strip()
        or _resolve_local_node_id_from_state_payload(state, execution_cai_url)
    )
    if not requester_node_id:
        return None
    node_capability_sync_audit = _sync_task_level_transport_node_capabilities_best_effort(
        state_payload=state,
        execution_cai_url=execution_cai_url,
        requester_node_id=requester_node_id,
        wallet_policy=wallet_policy,
    )
    peer_cai_urls_by_node = _cai_api_urls_by_node_id(execution_cai_url, state)
    if requester_node_id not in peer_cai_urls_by_node:
        peer_cai_urls_by_node[requester_node_id] = [execution_cai_url.rstrip("/")]
    route_health_records = _refresh_task_level_route_health_records_best_effort(
        state_payload=state,
        requester_node_id=requester_node_id,
        wallet_policy=wallet_policy,
    )
    performance_records = list_execution_performance_records(wallet_policy)

    worker_node_audit = _resolve_worker_execution_node_audit(
        execution_cai_url,
        execution_model_id,
    )
    worker_node_ids = _worker_node_ids_from_audit(worker_node_audit)
    node_id_attempts = _execution_node_id_attempts(
        worker_node_ids,
        state_payload=state,
        cai_url=execution_cai_url,
        private_network_model=False,
        requester_node_id=requester_node_id,
    )
    executor_count = _task_level_transport_effective_executor_count(
        execution_model_id,
        wallet_policy=wallet_policy,
    )
    executor_node_ids = _select_task_level_transport_executor_node_ids(
        node_id_attempts,
        peer_cai_urls_by_node=peer_cai_urls_by_node,
        requester_node_id=requester_node_id,
        executor_count=executor_count,
        route_health_records=route_health_records,
        model_id=execution_model_id,
        performance_records=performance_records,
    )
    if not executor_node_ids:
        return None
    executor_node_id_attempts = _task_level_transport_executor_fallback_attempts(
        executor_node_ids,
        requester_node_id=requester_node_id,
    )

    request_payload = _build_text_job_request_payload(
        execution_model_id,
        job.prompt,
        request_payload_override=request_payload_override,
    )
    initial_prompt_text = _task_level_transport_initial_prompt_text(
        request_payload,
        fallback_prompt=job.prompt,
    )
    initial_payload = initial_prompt_text.encode("utf-8")
    last_transport_exception: Exception | None = None
    for executor_attempt_index, attempted_executor_node_ids in enumerate(
        executor_node_id_attempts
    ):
        instance_id = f"caitask_{job.job_id}"
        if executor_attempt_index > 0:
            instance_id = f"{instance_id}_r{executor_attempt_index + 1}"
        executor_node_ids = list(attempted_executor_node_ids)
        total_layer_count = _task_level_transport_total_layer_count(
            execution_model_id,
            executor_count=len(executor_node_ids),
            wallet_policy=wallet_policy,
        )
        executor_node_ids, planned_shard_ranges = (
            _task_level_transport_planned_shard_ranges(
                execution_cai_url,
                execution_model_id,
                executor_node_ids=executor_node_ids,
                total_layer_count=total_layer_count,
            )
        )
        llm_runtime_metadata = _task_level_transport_llm_runtime_metadata(
            execution_model_id,
            total_layer_count=total_layer_count,
            wallet_policy=wallet_policy,
        )
        attempt_started_monotonic = time.monotonic()
        try:
            dispatch_started_monotonic = time.monotonic()
            dispatch_result = dispatch_cai_owned_transport_execution_dag(
                instance_id=instance_id,
                requester_node_id=requester_node_id,
                executor_node_ids=executor_node_ids,
                peer_cai_urls_by_node=peer_cai_urls_by_node,
                initial_payload=initial_payload,
                total_layer_count=total_layer_count,
                shard_ranges=planned_shard_ranges,
                model_id=execution_model_id,
                task_id=job.job_id,
                llm_runtime_metadata=llm_runtime_metadata,
                initial_token_count=_estimated_prompt_token_count(
                    initial_prompt_text,
                ),
                require_executor_readiness=True,
                require_cai_owned_runtime_ready=(
                    _task_level_transport_require_runtime_ready()
                ),
                require_executor_shard_readiness=bool(
                    llm_runtime_metadata
                    and _task_level_transport_require_shard_readiness()
                ),
                require_data_plane_route=_task_level_transport_require_data_plane_route(),
                require_proven_data_plane_route=(
                    _task_level_transport_require_proven_data_plane_route()
                ),
                route_health_records=route_health_records,
                executor_readiness_state_payload=state,
                timeout_sec=_task_level_transport_timeout_sec(5.0),
                submit_requester_offer=True,
                offer_settle_sec=0.0,
                policy=wallet_policy,
                single_executor_direct_final_output=(len(executor_node_ids) == 1),
                execution_pipeline_mode=(
                    CAI_OWNED_TRANSPORT_EXECUTION_PIPELINE_SINGLE_PASS_FINAL_DECODE
                    if llm_runtime_metadata is not None
                    else None
                ),
            )
            dispatch_result["nodeCapabilitySync"] = node_capability_sync_audit
            dispatch_duration_ms = _elapsed_ms(dispatch_started_monotonic)
            session_id = str(dispatch_result.get("sessionId") or "").strip()
            if not session_id:
                raise RuntimeError(
                    "CAI-owned task-level transport dispatch did not return a session id."
                )
            response_started_monotonic = time.monotonic()
            final_result = await_cai_owned_transport_session_final_result(
                session_id,
                requester_node_id=requester_node_id,
                timeout_sec=_task_level_transport_wait_timeout_sec(
                    request_timeout_sec
                ),
                poll_interval_sec=0.25,
                policy=wallet_policy,
            )
            response_duration_ms = _elapsed_ms(response_started_monotonic)
            attempt_duration_ms = _elapsed_ms(attempt_started_monotonic)
            if str(final_result.get("status") or "").strip() != "completed":
                _record_execution_attempt_performance_best_effort(
                    model_id=execution_model_id,
                    requester_node_id=requester_node_id,
                    executor_node_ids=executor_node_ids,
                    status="failed",
                    attempt_duration_ms=attempt_duration_ms,
                    readiness_duration_ms=dispatch_duration_ms,
                    response_duration_ms=response_duration_ms,
                    timeout_sec=request_timeout_sec,
                    error_type=str(
                        final_result.get("error")
                        or final_result.get("status")
                        or ""
                    ),
                    wallet_policy=wallet_policy,
                )
                raise RuntimeError(
                    "CAI-owned task-level transport did not complete: "
                    + str(
                        final_result.get("error")
                        or final_result.get("status")
                        or "unknown"
                    )
                )
            proof = final_result.get("proof")
            if not isinstance(proof, dict):
                raise RuntimeError(
                    "CAI-owned task-level transport completed without proof."
                )
            completion_notice = _notify_task_level_transport_completion_to_peers(
                session_id=session_id,
                proof=proof,
                requester_node_id=requester_node_id,
                executor_node_ids=executor_node_ids,
                peer_cai_urls_by_node=peer_cai_urls_by_node,
                timeout_sec=5.0,
            )
            dispatch_result["completionNotice"] = completion_notice
            final_output = final_result.get("finalOutput")
            response = _task_level_transport_response(
                final_output,
                model_id=execution_model_id,
                session_id=session_id,
                proof=proof,
            )
            instance_snapshot = _task_level_transport_instance_snapshot(
                instance_id=instance_id,
                execution_model_id=execution_model_id,
                requester_node_id=requester_node_id,
                executor_node_ids=executor_node_ids,
                proof=proof,
                dispatch_result=dispatch_result,
            )
            _record_execution_attempt_performance_best_effort(
                model_id=execution_model_id,
                requester_node_id=requester_node_id,
                executor_node_ids=executor_node_ids,
                status="completed",
                attempt_duration_ms=attempt_duration_ms,
                readiness_duration_ms=dispatch_duration_ms,
                response_duration_ms=response_duration_ms,
                timeout_sec=request_timeout_sec,
                wallet_policy=wallet_policy,
            )
            return {
                "response": response,
                "instance_snapshot": instance_snapshot,
                "dispatch_result": dispatch_result,
                "final_result": final_result,
                "node_capability_sync": node_capability_sync_audit,
            }
        except Exception as exc:
            last_transport_exception = exc
            if (
                executor_attempt_index + 1 >= len(executor_node_id_attempts)
                or not _should_retry_job_execution_error(exc)
            ):
                raise
            _log_best_effort_failure(
                "task-level transport executor attempt",
                exc,
            )
            continue
    if last_transport_exception is not None:
        raise last_transport_exception
    return None


def _notify_task_level_transport_completion_to_peers(
    *,
    session_id: str,
    proof: Mapping[str, Any],
    requester_node_id: str,
    executor_node_ids: Sequence[str],
    peer_cai_urls_by_node: Mapping[str, Sequence[str]],
    timeout_sec: float = 5.0,
) -> dict[str, Any]:
    requester = str(requester_node_id or "").strip()
    attempts: list[dict[str, Any]] = []
    notified_nodes: list[str] = []
    for node_id in dict.fromkeys(
        str(item or "").strip()
        for item in executor_node_ids
        if str(item or "").strip()
    ):
        if not node_id or node_id == requester:
            continue
        peer_urls = _clean_task_level_peer_cai_urls(
            peer_cai_urls_by_node.get(node_id) or []
        )
        if not peer_urls:
            attempts.append(
                {
                    "nodeId": node_id,
                    "status": "skipped",
                    "reason": "missing_peer_cai_url",
                }
            )
            continue
        last_error = ""
        for peer_url in peer_urls:
            try:
                response = submit_cai_owned_transport_completion_notice(
                    peer_url,
                    session_id,
                    proof=proof,
                    source_node_id=requester,
                    target_node_id=node_id,
                    timeout_sec=timeout_sec,
                )
                attempts.append(
                    {
                        "nodeId": node_id,
                        "peerCaiUrl": peer_url,
                        "status": "notified",
                        "response": response,
                    }
                )
                notified_nodes.append(node_id)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                attempts.append(
                    {
                        "nodeId": node_id,
                        "peerCaiUrl": peer_url,
                        "status": "failed",
                        "error": last_error,
                    }
                )
        if last_error and node_id not in notified_nodes:
            _log_best_effort_failure(
                f"task-level transport completion notice to {node_id}",
                RuntimeError(last_error),
            )
    return {
        "status": "ok",
        "sessionId": session_id,
        "notifiedNodeIds": list(dict.fromkeys(notified_nodes)),
        "attempts": attempts,
    }


def _clean_task_level_peer_cai_urls(peer_cai_urls: Sequence[str]) -> list[str]:
    return _clean_task_level_peer_cai_urls_impl(peer_cai_urls)


def _task_level_transport_executor_fallback_attempts(
    executor_node_ids: list[str],
    *,
    requester_node_id: str,
) -> list[list[str]]:
    return _task_level_transport_executor_fallback_attempts_impl(
        executor_node_ids,
        requester_node_id=requester_node_id,
    )


def _sync_task_level_transport_node_capabilities_best_effort(
    *,
    state_payload: dict[str, Any],
    execution_cai_url: str,
    requester_node_id: str,
    wallet_policy: WalletPolicy | None,
) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "attempted": True,
        "checkedAt": _now_iso(),
        "refreshLocal": {"status": "pending"},
        "peerSync": {"status": "pending"},
    }
    try:
        records = refresh_local_node_capabilities(
            state_payload=state_payload,
            cai_url=execution_cai_url,
            local_node_id=requester_node_id,
            policy=wallet_policy,
        )
        audit["refreshLocal"] = {
            "status": "ok",
            "recordCount": len(records) if isinstance(records, list) else 0,
        }
    except Exception as exc:
        audit["refreshLocal"] = _audit_error_status(exc)
        _log_best_effort_failure("task-level node capability refresh", exc)
    try:
        result = sync_node_capabilities_from_cai_peers(
            state_payload=state_payload,
            cai_url=execution_cai_url,
            local_node_id=requester_node_id,
            policy=wallet_policy,
        )
        audit["peerSync"] = {
            "status": "ok",
            **_node_capability_sync_result_audit(result),
        }
    except Exception as exc:
        audit["peerSync"] = _audit_error_status(exc)
        _log_best_effort_failure("task-level node capability peer sync", exc)
    return audit


def _refresh_task_level_route_health_records_best_effort(
    *,
    state_payload: dict[str, Any],
    requester_node_id: str,
    wallet_policy: WalletPolicy | None,
) -> list[Any]:
    try:
        probe_direct_api_routes(
            state_payload=state_payload,
            local_node_id=requester_node_id,
            timeout_sec=_task_level_route_probe_timeout_sec(),
            policy=wallet_policy,
        )
        probe_direct_data_routes(
            state_payload=state_payload,
            local_node_id=requester_node_id,
            timeout_sec=_task_level_route_probe_timeout_sec(),
            policy=wallet_policy,
        )
        record_overlay_routes_from_state(
            state_payload=state_payload,
            policy=wallet_policy,
        )
    except Exception as exc:
        _log_best_effort_failure("task-level route health refresh", exc)
    return list_route_health_records(wallet_policy)


def _select_task_level_transport_executor_node_ids(
    node_id_attempts: list[list[str] | None],
    *,
    peer_cai_urls_by_node: dict[str, list[str]],
    requester_node_id: str,
    executor_count: int | None = None,
    route_health_records: list[Any] | None = None,
    model_id: str | None = None,
    performance_records: list[Any] | None = None,
) -> list[str]:
    return _select_task_level_transport_executor_node_ids_impl(
        node_id_attempts,
        peer_cai_urls_by_node=peer_cai_urls_by_node,
        requester_node_id=requester_node_id,
        executor_count=executor_count,
        route_health_records=route_health_records,
        model_id=model_id,
        performance_records=performance_records,
        task_level_transport_executor_count_func=_task_level_transport_executor_count,
        sort_executor_candidates_by_route_health_func=_sort_executor_candidates_by_route_health,
    )


def _sort_executor_candidates_by_route_health(
    candidates: list[str],
    *,
    requester_node_id: str,
    route_health_records: list[Any] | None,
    model_id: str | None = None,
    performance_records: list[Any] | None = None,
) -> list[str]:
    return _sort_executor_candidates_by_route_health_impl(
        candidates,
        requester_node_id=requester_node_id,
        route_health_records=route_health_records,
        model_id=model_id,
        performance_records=performance_records,
        executor_candidate_route_preference_key_func=(
            _executor_candidate_route_preference_key
        ),
    )


def _executor_candidate_route_preference_key(
    node_id: str,
    *,
    requester_node_id: str,
    route_health_records: list[Any] | None,
    model_id: str | None,
    performance_records: list[Any] | None,
    original_index: int,
) -> tuple[int, int, int, int, int, int, float, float, int, int, float, int]:
    return _executor_candidate_route_preference_key_impl(
        node_id,
        requester_node_id=requester_node_id,
        route_health_records=route_health_records,
        model_id=model_id,
        performance_records=performance_records,
        original_index=original_index,
        route_health_score_for_path_func=route_health_score_for_path,
        execution_performance_preference_key_func=execution_performance_preference_key,
        latest_known_route_latency_ms_func=_latest_known_route_latency_ms,
    )


def _latest_known_route_latency_ms(
    source_node_id: str,
    sink_node_id: str,
    route_health_records: list[Any] | None,
) -> float | None:
    return _latest_known_route_latency_ms_impl(
        source_node_id,
        sink_node_id,
        route_health_records,
        route_health_record_field_func=_route_health_record_field,
    )


def _task_level_transport_response(
    final_output: Any,
    *,
    model_id: str,
    session_id: str,
    proof: dict[str, Any],
) -> dict[str, Any]:
    return _task_level_transport_response_impl(
        final_output,
        model_id=model_id,
        session_id=session_id,
        proof=proof,
        protocol_version=CAI_TASK_LEVEL_TRANSPORT_JOB_PROTOCOL_VERSION,
        current_time_func=time.time,
        final_output_text_func=_task_level_transport_final_output_text,
        usage_from_proof_func=_task_level_transport_usage_from_proof,
    )


def _task_level_transport_final_output_text(final_output: Any) -> str:
    return _task_level_transport_final_output_text_impl(
        final_output,
        base64_decode_func=base64.b64decode,
        log_best_effort_failure_func=_log_best_effort_failure,
    )


def _task_level_transport_usage_from_proof(
    proof: dict[str, Any],
) -> dict[str, int] | None:
    return _task_level_transport_usage_from_proof_impl(
        proof,
        extract_cai_owned_transport_token_usage_func=_extract_cai_owned_transport_token_usage,
    )


def _task_level_transport_instance_snapshot(
    *,
    instance_id: str,
    execution_model_id: str,
    requester_node_id: str,
    executor_node_ids: list[str],
    proof: dict[str, Any],
    dispatch_result: dict[str, Any],
) -> dict[str, Any]:
    return _task_level_transport_instance_snapshot_impl(
        instance_id=instance_id,
        execution_model_id=execution_model_id,
        requester_node_id=requester_node_id,
        executor_node_ids=executor_node_ids,
        proof=proof,
        dispatch_result=dispatch_result,
        snapshot_source=CAI_TASK_LEVEL_TRANSPORT_JOB_SOURCE,
        protocol_version=CAI_TASK_LEVEL_TRANSPORT_JOB_PROTOCOL_VERSION,
        participants_from_proof_func=_task_level_transport_participants_from_proof,
        participants_from_dag_func=_task_level_transport_participants_from_dag,
        total_layer_count_func=_task_level_transport_total_layer_count,
        total_layer_count_from_dag_func=_task_level_transport_total_layer_count_from_dag,
        total_layer_count_from_participants_func=(
            _task_level_transport_total_layer_count_from_participants
        ),
    )


def _task_level_transport_participants_from_proof(
    proof: dict[str, Any] | None,
    *,
    executor_node_ids: list[str],
) -> list[dict[str, Any]]:
    return _task_level_transport_participants_from_proof_impl(
        proof,
        executor_node_ids=executor_node_ids,
        optional_int_field_value_func=_optional_int_field_value,
        optional_int_value_func=_optional_int_value,
    )


def _task_level_transport_participants_from_dag(
    dag: Any,
    *,
    executor_node_ids: list[str],
) -> list[dict[str, Any]]:
    return _task_level_transport_participants_from_dag_impl(
        dag,
        executor_node_ids=executor_node_ids,
        optional_int_field_value_func=_optional_int_field_value,
        optional_int_value_func=_optional_int_value,
    )


def _task_level_transport_total_layer_count_from_dag(dag: Any) -> int | None:
    return _task_level_transport_total_layer_count_from_dag_impl(
        dag,
        optional_int_field_value_func=_optional_int_field_value,
    )


def _task_level_transport_total_layer_count_from_participants(
    participants: list[dict[str, Any]],
    *,
    fallback: int,
) -> int:
    return _task_level_transport_total_layer_count_from_participants_impl(
        participants,
        fallback=fallback,
        optional_int_value_func=_optional_int_value,
    )


def _estimated_prompt_token_count(prompt: str) -> int:
    return _estimated_prompt_token_count_impl(prompt)


def _format_worker_node_rejection_summary(audit: dict[str, Any] | None) -> str:
    return _format_worker_node_rejection_summary_impl(audit)


def _sync_worker_reward_bindings_from_cai(
    cai_url: str,
    participants: list[dict[str, Any]],
    wallet_policy: WalletPolicy | None = None,
) -> None:
    if not participants:
        return

    state_payload = _load_cai_state_payload(
        cai_url,
        log_operation="worker reward binding CAI state fetch",
    ) or {}
    if not state_payload:
        return

    summary_urls = _cai_summary_urls_by_node_id(cai_url, state_payload)
    identities = state_payload.get("nodeIdentities") or {}
    for item in participants:
        node_id = str(item.get("node_id") or "").strip()
        if not node_id:
            continue
        _worker_enabled, reward_address = _worker_identity_state(identities.get(node_id))
        if reward_address:
            bind_worker_reward_address(node_id, reward_address, policy=wallet_policy)
            continue
        summary_url = summary_urls.get(node_id)
        if not summary_url:
            continue
        try:
            summary_payload = _get_json(summary_url, timeout=5)
        except Exception as exc:
            _log_best_effort_failure(
                f"worker reward binding summary fetch for node {node_id}",
                exc,
            )
            continue
        worker = summary_payload.get("worker") or {}
        if not isinstance(worker, dict):
            continue
        reward_address = str(worker.get("worker_reward_address") or "").strip()
        if not reward_address:
            continue
        bind_worker_reward_address(node_id, reward_address, policy=wallet_policy)


def ensure_cai_instance(
    cai_url: str,
    model_id: str,
    *,
    ready_timeout_sec: int = 180,
    recreate_on_timeout: bool = True,
    private_network_model: bool = False,
    requester_node_id: str | None = None,
    excluded_node_ids: list[str] | set[str] | tuple[str, ...] | None = None,
) -> dict[str, Any] | None:
    instances_payload = _get_json(f"{cai_url.rstrip('/')}/state/instances")
    instance_became_visible = _instances_have_model(instances_payload, model_id)
    if instance_became_visible:
        if _wait_for_cai_instance_ready(
            cai_url,
            model_id,
            timeout_sec=ready_timeout_sec,
        ):
            return resolve_cai_instance_snapshot(cai_url, model_id)
        if not recreate_on_timeout:
            raise TimeoutError(
                _format_cai_instance_timeout_message(
                    cai_url,
                    model_id,
                    phase="ready CAI instance",
                )
            )
        cleanup_cai_model_instances(
            cai_url,
            model_id,
            best_effort=True,
            wait_timeout_sec=min(ready_timeout_sec, 60),
        )
        instances_payload = {}
        instance_became_visible = False

    if not instance_became_visible:
        create_payload = _resolve_cai_instance_create_payload(
            cai_url,
            model_id,
            private_network_model=private_network_model,
            requester_node_id=requester_node_id,
            excluded_node_ids=excluded_node_ids,
        )
        planned_snapshot = _snapshot_from_instance_definition(create_payload.get("instance"))
        create_timeout_sec = max(60, min(int(ready_timeout_sec), 300))
        try:
            _post_json(
                f"{cai_url.rstrip('/')}/instance",
                create_payload,
                timeout=create_timeout_sec,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                _format_cai_instance_timeout_message(
                    cai_url,
                    model_id,
                    phase="CAI instance creation",
                )
            ) from exc

        for _ in range(20):
            instances_payload = _get_json(f"{cai_url.rstrip('/')}/state/instances")
            instance_became_visible = _instances_have_model(instances_payload, model_id)
            if instance_became_visible:
                break
            time.sleep(1)
    else:
        planned_snapshot = None

    if _wait_for_cai_instance_ready(
        cai_url,
        model_id,
        timeout_sec=ready_timeout_sec,
    ):
        return resolve_cai_instance_snapshot(cai_url, model_id) or planned_snapshot

    if planned_snapshot is not None and not instance_became_visible:
        return planned_snapshot

    phase = (
        "ready CAI instance"
        if instance_became_visible
        else "CAI instance state visibility"
    )
    raise TimeoutError(
        _format_cai_instance_timeout_message(cai_url, model_id, phase=phase)
    )


def _resolve_cai_instance_create_payload(
    cai_url: str,
    model_id: str,
    *,
    private_network_model: bool = False,
    requester_node_id: str | None = None,
    excluded_node_ids: list[str] | set[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    worker_node_audit = _resolve_worker_execution_node_audit(cai_url, model_id)
    worker_node_ids = _worker_node_ids_from_audit(worker_node_audit)
    if private_network_model and worker_node_ids is not None:
        available_worker_nodes = len(worker_node_ids)
        minimum_worker_nodes = effective_private_worker_shard_minimum(
            available_worker_count=available_worker_nodes
        )
        if available_worker_nodes < minimum_worker_nodes:
            raise ValueError(
                f"Private network model {model_id} requires at least "
                f"{minimum_worker_nodes} worker-enabled CAI node(s), but only "
                f"{available_worker_nodes} eligible node(s) are currently available."
                f"{_format_worker_node_rejection_summary(worker_node_audit)}"
            )
    if worker_node_ids == []:
        raise ValueError(
            f"No worker-enabled CAI nodes are currently available for model {model_id}."
            f"{_format_worker_node_rejection_summary(worker_node_audit)}"
        )

    state_payload = _load_cai_state_payload(cai_url) or {}
    cluster_node_count = _cai_cluster_node_count(state_payload)
    requester_node_id = str(requester_node_id or "").strip() or (
        _resolve_local_node_id_from_state_payload(state_payload, cai_url)
    )
    route_health_records = list_route_health_records()
    performance_records = list_execution_performance_records()
    if worker_node_ids:
        worker_node_ids = _sort_executor_candidates_by_route_health(
            worker_node_ids,
            requester_node_id=requester_node_id,
            route_health_records=route_health_records,
            model_id=model_id,
            performance_records=performance_records,
        )
    prefer_multi_node = bool(private_network_model)
    node_id_attempts = _execution_node_id_attempts(
        worker_node_ids,
        state_payload=state_payload,
        cai_url=cai_url,
        private_network_model=private_network_model,
        requester_node_id=requester_node_id,
        excluded_node_ids=excluded_node_ids,
    )
    last_error: ValueError | None = None
    for node_ids in node_id_attempts:
        try:
            return _resolve_cai_instance_create_payload_for_nodes(
                cai_url,
                model_id,
                node_ids=node_ids,
                private_network_model=private_network_model,
                cluster_node_count=cluster_node_count,
                prefer_multi_node=prefer_multi_node,
                requester_node_id=requester_node_id,
                route_health_records=route_health_records,
                performance_records=performance_records,
            )
        except ValueError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise ValueError(f"No usable CAI placement preview found for model {model_id}.")


def _resolve_cai_instance_create_payload_for_nodes(
    cai_url: str,
    model_id: str,
    *,
    node_ids: list[str] | None,
    private_network_model: bool,
    cluster_node_count: int,
    prefer_multi_node: bool,
    requester_node_id: str | None = None,
    route_health_records: list[Any] | None = None,
    performance_records: list[Any] | None = None,
) -> dict[str, Any]:
    query_items: list[tuple[str, str]] = [("model_id", model_id)]
    if private_network_model:
        query_items.append(("private_network_model", "true"))
    if node_ids:
        query_items.extend(("node_ids", node_id) for node_id in node_ids)
    query_suffix = f"?{urlencode(query_items, doseq=True)}"
    previews_url = f"{cai_url.rstrip('/')}/instance/previews{query_suffix}"
    selected: dict[str, Any] | None = None
    for attempt in range(10):
        previews_payload = _get_json(previews_url)
        previews = previews_payload.get("previews") or []
        selected = _select_preferred_preview(
            previews,
            prefer_multi_node=prefer_multi_node,
            requester_node_id=requester_node_id,
            model_id=model_id,
            route_health_records=route_health_records,
            performance_records=performance_records,
        )
        if selected is None:
            break
        if cluster_node_count <= 1 or not prefer_multi_node:
            break
        if _preview_participant_count(selected) >= 2:
            break
        if attempt == 9:
            break
        time.sleep(1)
    if selected is not None and selected.get("instance"):
        return {"instance": selected["instance"]}

    try:
        placement = _get_json(
            f"{cai_url.rstrip('/')}/instance/placement{query_suffix}"
        )
    except HTTPError as exc:
        detail = _http_error_detail(exc)
        if detail:
            raise ValueError(
                f"No usable CAI placement preview found for model {model_id}. "
                f"Direct placement failed: {detail}"
            ) from exc
        raise ValueError(
            f"No usable CAI placement preview found for model {model_id}."
        ) from exc

    if placement:
        return {"instance": placement}

    raise ValueError(f"No usable CAI placement preview found for model {model_id}.")


def _execution_node_id_attempts(
    worker_node_ids: list[str] | None,
    *,
    state_payload: dict[str, Any],
    cai_url: str,
    private_network_model: bool,
    requester_node_id: str | None = None,
    excluded_node_ids: list[str] | set[str] | tuple[str, ...] | None = None,
) -> list[list[str] | None]:
    return _execution_node_id_attempts_impl(
        worker_node_ids,
        state_payload=state_payload,
        cai_url=cai_url,
        private_network_model=private_network_model,
        requester_node_id=requester_node_id,
        excluded_node_ids=excluded_node_ids,
        env_flag_func=_env_flag,
        resolve_local_node_id_from_state_payload_func=(
            _resolve_local_node_id_from_state_payload
        ),
        private_network_node_id_attempts_func=_private_network_node_id_attempts,
        dedupe_execution_node_id_attempts_func=_dedupe_execution_node_id_attempts,
    )


def _dedupe_execution_node_id_attempts(
    attempts: list[list[str] | None],
) -> list[list[str] | None]:
    return _dedupe_execution_node_id_attempts_impl(attempts)


def _private_network_node_id_attempts(
    worker_node_ids: list[str] | None,
) -> list[list[str] | None]:
    return _private_network_node_id_attempts_impl(
        worker_node_ids,
        effective_private_worker_shard_minimum_func=effective_private_worker_shard_minimum,
        env_positive_int_func=_env_positive_int,
    )


def _select_preferred_preview(
    previews: list[dict[str, Any]],
    *,
    prefer_multi_node: bool = False,
    requester_node_id: str | None = None,
    model_id: str | None = None,
    route_health_records: list[Any] | None = None,
    performance_records: list[Any] | None = None,
) -> dict[str, Any] | None:
    return _select_preferred_preview_impl(
        previews,
        prefer_multi_node=prefer_multi_node,
        requester_node_id=requester_node_id,
        model_id=model_id,
        route_health_records=route_health_records,
        performance_records=performance_records,
        preview_preference_key_func=_preview_preference_key,
        single_node_preview_preference_key_func=_single_node_preview_preference_key,
        preview_execution_preference_key_func=_preview_execution_preference_key,
        preview_execution_preference_penalty_key_func=(
            _preview_execution_preference_penalty_key
        ),
    )


def _preview_preference_key(item: dict[str, Any]) -> tuple[int, int]:
    return _preview_preference_key_impl(
        item,
        preview_participant_count_func=_preview_participant_count,
    )


def _single_node_preview_preference_key(item: dict[str, Any]) -> tuple[int, int]:
    return _single_node_preview_preference_key_impl(
        item,
        preview_participant_count_func=_preview_participant_count,
    )


def _preview_participant_count(item: dict[str, Any]) -> int:
    return _preview_participant_count_impl(
        item,
        instance_definition_participant_count_func=(
            _instance_definition_participant_count
        ),
    )


def _preview_participant_node_ids(item: dict[str, Any]) -> list[str]:
    return _preview_participant_node_ids_impl(item)


def _preview_execution_preference_key(
    item: dict[str, Any],
    *,
    requester_node_id: str | None,
    model_id: str | None,
    route_health_records: list[Any] | None,
    performance_records: list[Any] | None,
) -> tuple[int, int, int, int, int, int, float, float, int]:
    return _preview_execution_preference_key_impl(
        item,
        requester_node_id=requester_node_id,
        model_id=model_id,
        route_health_records=route_health_records,
        performance_records=performance_records,
        preview_participant_node_ids_func=_preview_participant_node_ids,
        route_health_score_for_path_func=route_health_score_for_path,
        execution_performance_preference_key_func=execution_performance_preference_key,
    )


def _preview_execution_preference_penalty_key(
    item: dict[str, Any],
    *,
    requester_node_id: str | None,
    model_id: str | None,
    route_health_records: list[Any] | None,
    performance_records: list[Any] | None,
) -> tuple[float, ...]:
    return _preview_execution_preference_penalty_key_impl(
        item,
        requester_node_id=requester_node_id,
        model_id=model_id,
        route_health_records=route_health_records,
        performance_records=performance_records,
        preview_execution_preference_key_func=_preview_execution_preference_key,
    )


def _instance_definition_participant_count(instance_definition: Any) -> int:
    return _instance_definition_participant_count_impl(instance_definition)


def _cai_cluster_node_count(state_payload: dict[str, Any]) -> int:
    return _cai_cluster_node_count_impl(state_payload)


def cleanup_cai_model_instances(
    cai_url: str,
    model_id: str,
    *,
    best_effort: bool = False,
    wait_timeout_sec: int = 30,
) -> None:
    try:
        matching_instance_ids = [
            item["instance_id"] for item in list_cai_instances(cai_url, model_id=model_id)
        ]
        if not matching_instance_ids:
            return

        for instance_id in matching_instance_ids:
            _delete_json(f"{cai_url.rstrip('/')}/instance/{instance_id}", timeout=30)

        deadline = time.time() + wait_timeout_sec
        while time.time() < deadline:
            remaining = list_cai_instances(cai_url, model_id=model_id)
            if not remaining:
                return
            time.sleep(1)
        raise TimeoutError(f"Timed out deleting CAI instances for model {model_id}.")
    except Exception as exc:
        if best_effort:
            _log_best_effort_failure("CAI model instance cleanup", exc)
            return
        raise


def list_cai_instances(
    cai_url: str, *, model_id: str | None = None
) -> list[dict[str, Any]]:
    payload = _get_json(f"{cai_url.rstrip('/')}/state/instances")
    items: list[dict[str, Any]] = []
    for instance_id, item in payload.items():
        if not isinstance(item, dict):
            continue
        for instance_type, instance_payload in item.items():
            if not isinstance(instance_payload, dict):
                continue
            shard_assignments = instance_payload.get("shardAssignments") or {}
            current_model_id = shard_assignments.get("modelId")
            if model_id is not None and current_model_id != model_id:
                continue
            items.append(
                {
                    "instance_id": str(instance_id),
                    "instance_type": str(instance_type),
                    "model_id": str(current_model_id) if current_model_id is not None else None,
                }
            )
    return items


def _instances_have_model(payload: dict[str, Any], model_id: str) -> bool:
    return _instances_have_model_impl(payload, model_id)


def resolve_cai_instance_snapshot(cai_url: str, model_id: str) -> dict[str, Any] | None:
    payload = _get_json(f"{cai_url.rstrip('/')}/state/instances")
    for instance_id, item in payload.items():
        snapshot = _snapshot_from_instance_state_item(
            instance_id=str(instance_id),
            instance_item=item,
            model_id=model_id,
        )
        if snapshot is not None:
            return snapshot
    return None


def resolve_cai_command_instance_snapshot(
    cai_url: str,
    command_id: str,
    *,
    model_id: str | None = None,
    timeout_sec: float = 1.0,
) -> dict[str, Any] | None:
    try:
        state_payload = _get_json(f"{cai_url.rstrip('/')}/state", timeout=timeout_sec)
    except Exception as exc:
        _log_best_effort_failure("CAI command instance snapshot state fetch", exc)
        return None
    tasks_payload = state_payload.get("tasks") or {}
    if not isinstance(tasks_payload, dict):
        return None

    matched_instance_id: str | None = None
    for task_wrapper in tasks_payload.values():
        if not isinstance(task_wrapper, dict):
            continue
        for task_payload in task_wrapper.values():
            if not isinstance(task_payload, dict):
                continue
            if str(task_payload.get("commandId") or "").strip() != command_id:
                continue
            matched_instance_id = str(task_payload.get("instanceId") or "").strip() or None
            bound_instance = task_payload.get("boundInstance") or {}
            if isinstance(bound_instance, dict):
                instance_definition = bound_instance.get("instance")
                snapshot = _snapshot_from_instance_definition(
                    instance_definition,
                    snapshot_source="command_bound_instance",
                )
                if snapshot is not None:
                    if matched_instance_id:
                        snapshot["instance_id"] = matched_instance_id
                    return snapshot

    if matched_instance_id is None:
        return None

    instances_payload = state_payload.get("instances") or {}
    if not isinstance(instances_payload, dict):
        return None
    instance_item = instances_payload.get(matched_instance_id)
    if not isinstance(instance_item, dict):
        return None
    return _snapshot_from_instance_state_item(
        instance_id=matched_instance_id,
        instance_item=instance_item,
        model_id=model_id,
    )


def _snapshot_from_instance_state_item(
    *,
    instance_id: str,
    instance_item: dict[str, Any],
    model_id: str | None = None,
) -> dict[str, Any] | None:
    return _snapshot_from_instance_state_item_impl(
        instance_id=instance_id,
        instance_item=instance_item,
        model_id=model_id,
        extract_instance_participants_func=_extract_instance_participants,
    )


def _snapshot_from_instance_definition(
    instance_definition: dict[str, Any] | None,
    *,
    snapshot_source: str = "planned_definition",
) -> dict[str, Any] | None:
    return _snapshot_from_instance_definition_impl(
        instance_definition,
        snapshot_source=snapshot_source,
        extract_instance_participants_func=_extract_instance_participants,
    )


def _require_settleable_instance_snapshot(instance_snapshot: dict[str, Any] | None) -> None:
    _require_settleable_instance_snapshot_impl(
        instance_snapshot,
        task_level_transport_job_source=CAI_TASK_LEVEL_TRANSPORT_JOB_SOURCE,
    )


def _wait_for_cai_instance_ready(
    cai_url: str,
    model_id: str,
    *,
    timeout_sec: int,
) -> bool:
    deadline = time.time() + timeout_sec
    polling_failure_count = 0
    last_polling_error: Exception | None = None
    while time.time() < deadline:
        try:
            state = _get_json(f"{cai_url.rstrip('/')}/state")
            audit = _cai_instance_readiness_from_state(
                state,
                model_id,
                route_health_records=list_route_health_records(),
            )
            if audit.get("status") == "inference_ready":
                return True
        except Exception as exc:
            polling_failure_count += 1
            last_polling_error = exc
        time.sleep(1)
    if last_polling_error is not None:
        _log_best_effort_failure(
            f"CAI instance readiness polling ({polling_failure_count} failed attempt(s))",
            last_polling_error,
        )
    return False


def cai_instance_readiness_audit(
    cai_url: str,
    model_id: str,
    *,
    wallet_policy: WalletPolicy | None = None,
) -> dict[str, Any]:
    state_payload = _load_cai_state_payload(cai_url)
    if not isinstance(state_payload, dict):
        return _attach_cai_instance_readiness_state(
            {
                "status": "state_unavailable",
                "ready": False,
                "modelId": model_id,
                "reason": "CAI state endpoint is unavailable.",
            }
        )
    return _attach_cai_instance_readiness_state(
        _cai_instance_readiness_from_state(
            state_payload,
            model_id,
            route_health_records=list_route_health_records(wallet_policy),
        )
    )


def _cai_instance_readiness_from_state(
    state_payload: dict[str, Any],
    model_id: str,
    *,
    route_health_records: list[Any] | None = None,
) -> dict[str, Any]:
    instances_payload = state_payload.get("instances") or {}
    runners_payload = state_payload.get("runners") or {}
    instance = _find_model_instance(instances_payload, model_id)
    pending_nodes = _pending_model_download_node_labels(state_payload, model_id)
    completed_download_nodes = _completed_model_download_node_labels(
        state_payload,
        model_id,
    )

    if instance is None:
        if pending_nodes:
            return {
                "status": "shard_loading",
                "ready": False,
                "modelId": model_id,
                "reason": "Model shards are still downloading.",
                "pendingDownloadNodes": pending_nodes,
            }
        if completed_download_nodes:
            return {
                "status": "model_materializing",
                "ready": False,
                "modelId": model_id,
                "reason": (
                    "Model shards are downloaded, but no materialized CAI "
                    "instance is visible yet."
                ),
                "completedDownloadNodes": completed_download_nodes,
            }
        return {
            "status": "model_missing",
            "ready": False,
            "modelId": model_id,
            "reason": "No CAI instance for this model is visible in state.",
        }

    payload = instance.get("payload") or {}
    shard_assignments = payload.get("shardAssignments") or {}
    runner_to_shard = shard_assignments.get("runnerToShard") or {}
    runner_ids = [str(runner_id) for runner_id in runner_to_shard.keys()]
    if not runner_ids:
        return {
            "status": "runner_missing",
            "ready": False,
            "modelId": model_id,
            "instanceId": instance.get("instance_id"),
            "reason": "CAI instance has no runner assignment.",
        }

    runner_audits = [
        {
            "runnerId": runner_id,
            "status": _runner_status_name(runners_payload.get(runner_id))
            or "unknown",
        }
        for runner_id in runner_ids
    ]
    if all(
        item["status"] in {"RunnerReady", "RunnerRunning"}
        for item in runner_audits
    ):
        return {
            "status": "inference_ready",
            "ready": True,
            "modelId": model_id,
            "instanceId": instance.get("instance_id"),
            "reason": "All assigned runners are ready.",
            "runners": runner_audits,
        }
    instance_snapshot = _snapshot_from_instance_definition(
        {str(instance.get("instance_type") or "CAIInstance"): payload},
        snapshot_source="state",
    )
    network_audit = _build_execution_network_audit(
        state_payload=state_payload,
        instance_snapshot=instance_snapshot,
        route_health_records=route_health_records,
    )
    transport_mode = str(network_audit.get("transportMode") or "").strip()
    compute_cell = network_audit.get("llamaCppComputeCell")
    rpc_ready = bool(
        isinstance(compute_cell, dict)
        and compute_cell.get("readyForLlamaCppRpc")
    )
    execution_strategy = network_audit.get("llamaCppExecutionStrategy")
    cai_owned_transport = (
        execution_strategy.get("caiOwnedTransport")
        if isinstance(execution_strategy, dict)
        else None
    )
    cai_owned_route_readiness = (
        cai_owned_transport.get("routeHealthReadiness")
        if isinstance(cai_owned_transport, dict)
        else None
    )
    if (
        isinstance(execution_strategy, dict)
        and bool(execution_strategy.get("requiresCaiOwnedTransport"))
        and isinstance(cai_owned_route_readiness, dict)
    ):
        if not bool(cai_owned_route_readiness.get("ready")):
            fatal_reasons = cai_owned_route_readiness.get("fatalReasons")
            reason_detail = ""
            if isinstance(fatal_reasons, list) and fatal_reasons:
                reason_detail = " " + "; ".join(str(item) for item in fatal_reasons[:3])
            return {
                "status": "cai_owned_route_blocked",
                "ready": False,
                "modelId": model_id,
                "instanceId": instance.get("instance_id"),
                "reason": (
                    "CAI-owned transport is required, but its data-plane "
                    f"route is not proven.{reason_detail}"
                ),
                "runners": runner_audits,
                "networkAudit": network_audit,
                "pendingDownloadNodes": pending_nodes,
            }
        return {
            "status": "cai_owned_route_ready",
            "ready": False,
            "modelId": model_id,
            "instanceId": instance.get("instance_id"),
            "reason": (
                "CAI-owned transport route is proven, but assigned runners "
                "are not inference-ready yet."
            ),
            "runners": runner_audits,
            "networkAudit": network_audit,
            "pendingDownloadNodes": pending_nodes,
        }
    if (
        transport_mode in {"multi_worker_disconnected", "multi_worker_overlay_only"}
        and not rpc_ready
    ):
        return {
            "status": "route_blocked",
            "ready": False,
            "modelId": model_id,
            "instanceId": instance.get("instance_id"),
            "reason": (
                "Instance is visible, but participant route is not proven "
                f"for transport mode {transport_mode}."
            ),
            "runners": runner_audits,
            "networkAudit": network_audit,
            "pendingDownloadNodes": pending_nodes,
        }
    if rpc_ready:
        return {
            "status": "rpc_ready",
            "ready": False,
            "modelId": model_id,
            "instanceId": instance.get("instance_id"),
            "reason": (
                "llama.cpp RPC route is ready, but assigned runners are not "
                "inference-ready yet."
            ),
            "runners": runner_audits,
            "networkAudit": network_audit,
            "pendingDownloadNodes": pending_nodes,
        }
    if pending_nodes:
        status = "shard_loading"
        reason = "Instance is visible, but model shards are still downloading."
    else:
        status = "model_loading"
        reason = "Instance is visible, but assigned runners are not ready."
    return {
        "status": status,
        "ready": False,
        "modelId": model_id,
        "instanceId": instance.get("instance_id"),
        "reason": reason,
        "runners": runner_audits,
        "pendingDownloadNodes": pending_nodes,
    }


def _attach_cai_instance_readiness_state(
    audit: dict[str, Any],
) -> dict[str, Any]:
    return _attach_cai_instance_readiness_state_impl(audit)


def _cai_instance_readiness_stage_for_status(
    status: str,
    *,
    ready: bool,
) -> tuple[str, list[str], list[str], str | None]:
    return _cai_instance_readiness_stage_for_status_impl(status, ready=ready)


def _cai_instance_readiness_stage_items(
    *,
    current_stage: str,
    completed_stages: list[str],
    blocked_stages: list[str],
) -> list[dict[str, Any]]:
    return _cai_instance_readiness_stage_items_impl(
        current_stage=current_stage,
        completed_stages=completed_stages,
        blocked_stages=blocked_stages,
    )


def _format_cai_instance_timeout_message(
    cai_url: str,
    model_id: str,
    *,
    phase: str,
) -> str:
    message = f"Timed out waiting for {phase} for model {model_id}."
    readiness = cai_instance_readiness_audit(cai_url, model_id)
    readiness_status = str(readiness.get("status") or "").strip()
    readiness_reason = str(readiness.get("reason") or "").strip()
    if readiness_status:
        message = f"{message} Readiness status={readiness_status}."
    readiness_state = readiness.get("readinessState")
    if isinstance(readiness_state, dict):
        current_stage = str(readiness_state.get("currentStage") or "").strip()
        next_stage = str(readiness_state.get("nextStage") or "").strip()
        if current_stage:
            message = f"{message} Readiness stage={current_stage}."
        if next_stage and next_stage != current_stage:
            message = f"{message} Next stage={next_stage}."
    if readiness_reason:
        message = f"{message} {readiness_reason}"
    download_detail = _describe_pending_model_downloads(cai_url, model_id)
    if download_detail:
        return f"{message} {download_detail}"
    return message


def _describe_pending_model_downloads(cai_url: str, model_id: str) -> str | None:
    return _describe_pending_model_downloads_impl(
        cai_url,
        model_id,
        load_cai_state_payload_func=_load_cai_state_payload,
        pending_model_download_node_labels_func=_pending_model_download_node_labels,
    )


def _pending_model_download_node_labels(
    state_payload: dict[str, Any],
    model_id: str,
) -> list[str]:
    return _pending_model_download_node_labels_impl(
        state_payload,
        model_id,
        model_download_node_labels_func=_model_download_node_labels,
        download_progress_is_completed_func=_download_progress_is_completed,
        download_progress_is_pending_func=_download_progress_is_pending,
    )


def _completed_model_download_node_labels(
    state_payload: dict[str, Any],
    model_id: str,
) -> list[str]:
    return _completed_model_download_node_labels_impl(
        state_payload,
        model_id,
        model_download_node_labels_func=_model_download_node_labels,
        download_progress_is_completed_func=_download_progress_is_completed,
    )


def _model_download_node_labels(
    state_payload: dict[str, Any],
    model_id: str,
    *,
    include_node: Any,
) -> list[str]:
    return _model_download_node_labels_impl(
        state_payload,
        model_id,
        include_node=include_node,
        download_progress_matches_model_func=_download_progress_matches_model,
    )


def _download_progress_matches_model(progress_item: Any, model_id: str) -> bool:
    return _download_progress_matches_model_impl(
        progress_item,
        model_id,
        normalize_network_model_id_func=normalize_network_model_id,
        download_progress_equivalent_model_ids_func=(
            _download_progress_equivalent_model_ids
        ),
    )


def _download_progress_equivalent_model_ids(model_id: str) -> set[str]:
    return _download_progress_equivalent_model_ids_impl(
        model_id,
        accepted_worker_model_ids_func=_accepted_worker_model_ids,
        curated_model_for_id_func=curated_model_for_id,
        curated_model_registry_func=curated_model_registry,
        normalize_network_model_id_func=normalize_network_model_id,
    )


def _download_progress_is_pending(progress_item: Any) -> bool:
    return _download_progress_is_pending_impl(progress_item)


def _download_progress_is_completed(progress_item: Any) -> bool:
    return _download_progress_is_completed_impl(progress_item)


def _should_retry_cai_startup_error(exc: Exception) -> bool:
    return _should_retry_cai_startup_error_impl(exc)


def _find_model_instance(
    instances_payload: dict[str, Any],
    model_id: str,
) -> dict[str, Any] | None:
    return _find_model_instance_impl(instances_payload, model_id)


def _instance_is_ready(
    instance: dict[str, Any],
    runners_payload: dict[str, Any],
) -> bool:
    return _instance_is_ready_impl(
        instance,
        runners_payload,
        runner_status_name_func=_runner_status_name,
    )


def _runner_status_name(runner_status: Any) -> str | None:
    return _runner_status_name_impl(runner_status)


def _extract_instance_participants(shard_assignments: dict[str, Any]) -> list[dict[str, Any]]:
    return _extract_instance_participants_impl(
        shard_assignments,
        unwrap_shard_metadata_func=_unwrap_shard_metadata,
        layer_count_from_metadata_func=_layer_count_from_metadata,
    )


def _route_health_records_for_execution_settlement(
    *,
    instance_snapshot: dict[str, Any] | None,
    wallet_policy: WalletPolicy | None,
) -> list[Any]:
    records: list[Any] = list(list_route_health_records(wallet_policy))
    participant_node_ids = _participant_node_ids(instance_snapshot)
    if len(participant_node_ids) <= 1:
        return records

    records = _augment_route_health_records_from_worker_attestations(
        records,
        participant_node_ids=participant_node_ids,
        wallet_policy=wallet_policy,
    )
    return records


def _augment_route_health_records_from_worker_attestations(
    route_health_records: list[Any],
    *,
    participant_node_ids: list[str],
    wallet_policy: WalletPolicy | None,
) -> list[Any]:
    return _augment_route_health_records_from_worker_attestations_impl(
        route_health_records,
        participant_node_ids=participant_node_ids,
        wallet_policy=wallet_policy,
        list_worker_capability_attestations_func=list_worker_capability_attestations,
        get_json_func=_get_json,
        log_best_effort_failure_func=_log_best_effort_failure,
    )


def _route_health_endpoints_from_worker_attestations(
    participant_node_ids: list[str],
    *,
    wallet_policy: WalletPolicy | None,
) -> list[str]:
    return _route_health_endpoints_from_worker_attestations_impl(
        participant_node_ids,
        wallet_policy=wallet_policy,
        list_worker_capability_attestations_func=list_worker_capability_attestations,
        log_best_effort_failure_func=_log_best_effort_failure,
    )


def _build_execution_network_audit(
    *,
    state_payload: dict[str, Any] | None,
    instance_snapshot: dict[str, Any] | None,
    requester_node_id: str | None = None,
    route_health_records: list[Any] | None = None,
    wallet_policy: WalletPolicy | None = None,
) -> dict[str, Any]:
    participant_node_ids = _participant_node_ids(instance_snapshot)
    participant_count = len(participant_node_ids)
    normalized_requester_node_id = str(requester_node_id or "").strip() or None
    single_worker_remote = (
        participant_count == 1
        and normalized_requester_node_id is not None
        and participant_node_ids[0] != normalized_requester_node_id
    )
    single_worker_self_execution = (
        participant_count == 1
        and normalized_requester_node_id is not None
        and participant_node_ids[0] == normalized_requester_node_id
    )
    socket_adjacency = _participant_socket_adjacency(
        state_payload,
        participant_node_ids,
    )
    checked_direct_socket_links = _checked_direct_socket_links(socket_adjacency)
    checked_overlay_links = _checked_overlay_links(state_payload, participant_node_ids)
    direct_socket_link_count = sum(len(peers) for peers in socket_adjacency.values())
    direct_bidirectional_link_count = sum(
        1 for item in checked_direct_socket_links if bool(item.get("bidirectional"))
    )
    strongly_connected_direct_graph = _is_strongly_connected_participant_graph(
        socket_adjacency,
        participant_node_ids,
    )
    coordinator_candidate_node_ids = _coordinator_direct_fanout_candidate_node_ids(
        socket_adjacency,
        participant_node_ids,
    )
    relay_route_candidate_items = relay_route_candidates(
        state_payload or {},
        participant_node_ids,
    )
    relay_coordinator_candidates = relay_coordinator_candidate_node_ids(
        state_payload or {},
        participant_node_ids,
    )
    active_relay_routes = _active_relay_routes(instance_snapshot, participant_node_ids)
    active_relay_transit_node_ids = sorted(
        {
            str(route.get("transitNodeId") or "").strip()
            for route in active_relay_routes
            if str(route.get("transitNodeId") or "").strip()
        }
    )
    coordinator_direct_fanout = (
        participant_count > 1 and len(coordinator_candidate_node_ids) > 0
    )
    relay_capable_node_ids, relay_transit_candidate_node_ids = (
        _relay_capability_snapshot(state_payload, participant_node_ids)
    )
    relay_hops_used = bool(active_relay_routes)
    relay_bottleneck_risk = (
        relay_hops_used
        and participant_count > 1
        and direct_socket_link_count == 0
        and len(active_relay_transit_node_ids) <= 1
    )
    execution_strategy = _execution_compute_cell_strategy(
        participant_node_ids,
        route_health_records,
    )
    (
        cai_owned_transport_proof,
        cai_owned_transport_executed,
        cai_owned_transport_proof_error,
    ) = _execution_cai_owned_transport_proof(
        instance_snapshot,
        participant_node_ids,
        wallet_policy=wallet_policy,
    )
    if isinstance(execution_strategy, dict):
        execution_strategy = dict(execution_strategy)
        execution_strategy["caiOwnedTransportExecuted"] = (
            cai_owned_transport_executed
        )
        execution_strategy["caiOwnedTransportProofError"] = (
            cai_owned_transport_proof_error
        )
    compute_cell_profile = (
        execution_strategy.get("computeCellProfile")
        if isinstance(execution_strategy, dict)
        else None
    )
    transport_mode = _execution_transport_mode(
        participant_count=participant_count,
        relay_hops_used=relay_hops_used,
        coordinator_direct_fanout=coordinator_direct_fanout,
        direct_socket_link_count=direct_socket_link_count,
        overlay_link_count=len(checked_overlay_links),
    )
    relay_note = _relay_transport_note(
        relay_hops_used=relay_hops_used,
        relay_bottleneck_risk=relay_bottleneck_risk,
        relay_coordinator_candidate_count=len(relay_coordinator_candidates),
        relay_route_candidate_count=len(relay_route_candidate_items),
        relay_transit_candidate_count=len(relay_transit_candidate_node_ids),
        relay_capable_node_count=len(relay_capable_node_ids),
    )

    return {
        "participantCount": participant_count,
        "participantNodeIds": participant_node_ids,
        "instanceSnapshotSource": (
            instance_snapshot.get("snapshot_source")
            if isinstance(instance_snapshot, dict)
            else None
        ),
        "requesterNodeId": normalized_requester_node_id,
        "singleWorkerRemote": single_worker_remote,
        "singleWorkerSelfExecution": single_worker_self_execution,
        "directSocketLinkCount": direct_socket_link_count,
        "directBidirectionalLinkCount": direct_bidirectional_link_count,
        "overlayLinkCount": len(checked_overlay_links),
        "stronglyConnectedDirectGraph": strongly_connected_direct_graph,
        "coordinatorDirectFanout": coordinator_direct_fanout,
        "coordinatorCandidateNodeIds": coordinator_candidate_node_ids,
        "relayCoordinatorCandidateCount": len(relay_coordinator_candidates),
        "relayCoordinatorCandidateNodeIds": relay_coordinator_candidates,
        "relayRouteCandidateCount": len(relay_route_candidate_items),
        "checkedRelayRoutes": active_relay_routes or relay_route_candidate_items,
        "transportMode": transport_mode,
        "decentralizedExecution": (
            single_worker_remote or coordinator_direct_fanout or relay_hops_used
        ),
        "relayHopsUsed": relay_hops_used,
        "relayBottleneckRisk": relay_bottleneck_risk,
        "activeRelayTransitNodeIds": active_relay_transit_node_ids,
        "relayCapableNodeCount": len(relay_capable_node_ids),
        "relayCapableNodeIds": relay_capable_node_ids,
        "relayTransitCandidateCount": len(relay_transit_candidate_node_ids),
        "relayTransitCandidateNodeIds": relay_transit_candidate_node_ids,
        "relayNote": relay_note,
        "checkedDirectSocketLinks": checked_direct_socket_links,
        "checkedOverlayLinks": checked_overlay_links,
        "llamaCppComputeCell": compute_cell_profile,
        "llamaCppExecutionStrategy": execution_strategy,
        "caiOwnedTransportExecuted": cai_owned_transport_executed,
        "caiOwnedTransportExecutionProof": cai_owned_transport_proof,
        "caiOwnedTransportProofError": cai_owned_transport_proof_error,
    }


def _execution_compute_cell_strategy(
    participant_node_ids: list[str],
    route_health_records: list[Any] | None,
) -> dict[str, Any] | None:
    return _execution_compute_cell_strategy_impl(
        participant_node_ids,
        route_health_records,
        plan_llama_cpp_distributed_execution_func=plan_llama_cpp_distributed_execution,
        log_best_effort_failure_func=_log_best_effort_failure,
    )


def _execution_cai_owned_transport_proof(
    instance_snapshot: dict[str, Any] | None,
    participant_node_ids: list[str],
    *,
    wallet_policy: WalletPolicy | None = None,
) -> tuple[dict[str, Any] | None, bool, str | None]:
    return _execution_cai_owned_transport_proof_impl(
        instance_snapshot,
        participant_node_ids,
        wallet_policy=wallet_policy,
        latest_completed_cai_owned_transport_proof_for_instance_func=latest_completed_cai_owned_transport_proof_for_instance,
        validate_cai_owned_transport_execution_proof_func=validate_cai_owned_transport_execution_proof,
    )


def cleanup_orphan_llama_cpp_processes(
    *,
    cai_url: str | None = None,
    model_id: str | None = None,
) -> int:
    return _cleanup_orphan_llama_cpp_processes(
        cai_url=cai_url,
        model_id=model_id,
        list_cai_instances_func=list_cai_instances,
        log_best_effort_failure=_log_best_effort_failure,
    )


def _validator_attestation_endpoint(source_url: str) -> str:
    parsed = urlsplit(source_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Unsupported validator source URL: {source_url}")
    return f"{parsed.scheme}://{parsed.netloc}/v1/cai/settlement/attest"

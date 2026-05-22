# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
import secrets
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from .model import (
    NetworkModelPolicy,
    WalletPolicy,
    is_private_curated_model_id,
)
from .route_health import (
    list_route_health_records,
    llama_cpp_compute_cell_profile_for_path,
)
from .cai_owned_transport_auth import (
    cai_owned_transport_auth_headers,
    cai_owned_transport_auth_required,
    cai_owned_transport_json_headers as _cai_owned_transport_json_headers,
    cai_owned_transport_peer_signing_kwargs,
    require_cai_owned_transport_local_runtime_auth as _require_cai_owned_transport_local_runtime_auth,
    sign_cai_owned_transport_batch_envelope,
    sign_cai_owned_transport_execution_proof,
    sign_cai_owned_transport_payload,
    sign_cai_owned_transport_session_offer,
    sign_cai_owned_transport_shard_receipt,
    validate_cai_owned_transport_local_runtime_auth,
    validate_cai_owned_transport_payload_signature,
    validate_cai_owned_transport_request_auth,
)
from .cai_owned_transport_batch_lifecycle import (
    apply_cai_owned_transport_batch_lease as _apply_cai_owned_transport_batch_lease,
    cai_owned_transport_batch_attempt_count as _cai_owned_transport_batch_attempt_count,
    cai_owned_transport_batch_claim_expired as _cai_owned_transport_batch_claim_expired,
    cai_owned_transport_batch_lease_expired as _cai_owned_transport_batch_lease_expired,
    clear_cai_owned_transport_batch_runtime_claim as _clear_cai_owned_transport_batch_runtime_claim,
    coerce_cai_owned_transport_batch_claim_timeout_seconds as _coerce_cai_owned_transport_batch_claim_timeout_seconds,
    coerce_cai_owned_transport_batch_lease_seconds as _coerce_cai_owned_transport_batch_lease_seconds,
    coerce_cai_owned_transport_max_attempts as _coerce_cai_owned_transport_max_attempts,
    mark_cai_owned_transport_batch_timed_out as _mark_cai_owned_transport_batch_timed_out,
)
from .cai_owned_llm_runtime_metadata import (
    cai_owned_transport_llm_runtime_metadata as _cai_owned_transport_llm_runtime_metadata,
    require_runtime_metadata_layer_range_supported as _require_runtime_metadata_layer_range_supported,
    runtime_metadata_bool as _runtime_metadata_bool,
    runtime_metadata_external_shard_descriptor as _runtime_metadata_external_shard_descriptor,
    runtime_metadata_int as _runtime_metadata_int,
    runtime_metadata_mapping as _runtime_metadata_mapping,
    runtime_metadata_shape as _runtime_metadata_shape,
    runtime_metadata_text as _runtime_metadata_text,
)
from .cai_owned_transport_peer_urls import (
    cai_owned_transport_peer_url_priority as _cai_owned_transport_peer_url_priority,
    cai_owned_transport_peer_url_route_class as _cai_owned_transport_peer_url_route_class,
    clean_peer_cai_urls as _clean_peer_cai_urls,
    parse_cai_owned_transport_overlay_url as _parse_cai_owned_transport_overlay_url,
    prioritized_cai_owned_transport_peer_urls as _prioritized_cai_owned_transport_peer_urls,
)
from .cai_owned_transport_common import (
    cai_owned_transport_chain_id as _cai_owned_transport_chain_id,
    cai_owned_transport_payload_chain_id as _cai_owned_transport_payload_chain_id,
    clean_node_ids as _clean_node_ids,
    is_safe_transport_file_id as _is_safe_transport_file_id,
    jsonable_dict as _jsonable_dict,
    normalize_sha256_hex as _normalize_sha256_hex,
    optional_int as _optional_int,
    optional_sha256_hex as _optional_sha256_hex,
    parse_cai_owned_transport_datetime as _parse_cai_owned_transport_datetime,
    require_safe_transport_file_id as _require_safe_transport_file_id,
    validate_cai_owned_transport_batch_replay as _validate_cai_owned_transport_batch_replay,
    validate_cai_owned_transport_chain_id as _validate_cai_owned_transport_chain_id,
    validate_cai_owned_transport_created_at as _validate_cai_owned_transport_created_at,
)
from .cai_owned_transport_execution_plan import (
    cai_owned_transport_frame_kind_for_phase as _cai_owned_transport_frame_kind_for_phase,
    cai_owned_transport_layer_ranges as _cai_owned_transport_layer_ranges,
    cai_owned_transport_output_route_plan_from_dag as _cai_owned_transport_output_route_plan_from_dag,
    cai_owned_transport_template_token_end as _cai_owned_transport_template_token_end,
    cai_owned_transport_template_token_start as _cai_owned_transport_template_token_start,
    clean_sink_node_ids as _clean_sink_node_ids,
    execution_mode_for_compute_cell as _execution_mode_for_compute_cell,
    execution_reason as _execution_reason,
    normalize_cai_owned_transport_shard_ranges as _normalize_cai_owned_transport_shard_ranges,
)
from .cai_owned_transport_ids import (
    cai_owned_transport_batch_id as _cai_owned_transport_batch_id,
    cai_owned_transport_dag_hash as _cai_owned_transport_dag_hash,
    cai_owned_transport_stage_id as _cai_owned_transport_stage_id,
)
from .cai_owned_transport_payload_codec import (
    cai_owned_transport_encoded_payload_fields as _cai_owned_transport_encoded_payload_fields,
    decode_cai_owned_transport_batch_payload as _decode_cai_owned_transport_batch_payload,
    decode_cai_owned_transport_payload_bytes as _decode_cai_owned_transport_payload_bytes,
    encode_cai_owned_transport_payload_bytes as _encode_cai_owned_transport_payload_bytes,
    encoded_cai_owned_transport_payload_bytes_from_envelope as _encoded_cai_owned_transport_payload_bytes_from_envelope,
    normalize_cai_owned_transport_payload_compression as _normalize_cai_owned_transport_payload_compression,
)
from .cai_owned_transport_protocol import (
    CAI_OWNED_LLM_HANDOFF_ABI,
    CAI_OWNED_LLM_HANDOFF_SCHEMA_VERSION,
    CAI_OWNED_LLM_HANDOFF_TENSOR_ENCODINGS,
    CAI_OWNED_TRANSPORT_BATCH_ENVELOPE_MAX_AGE_SECONDS,
    CAI_OWNED_TRANSPORT_BATCH_ENVELOPE_SCHEMA_VERSION,
    CAI_OWNED_TRANSPORT_BATCH_PHASES,
    CAI_OWNED_TRANSPORT_EXECUTION_DAG_SCHEMA_VERSION,
    CAI_OWNED_TRANSPORT_EXECUTION_PIPELINE_FULL_PREFILL_DECODE,
    CAI_OWNED_TRANSPORT_EXECUTION_PIPELINE_SINGLE_PASS_FINAL_DECODE,
    CAI_OWNED_TRANSPORT_FRAME_KINDS,
    CAI_OWNED_TRANSPORT_FRAME_SCHEMA_VERSION,
    CAI_OWNED_TRANSPORT_HASH_CHAIN_SCHEMA_VERSION,
    CAI_OWNED_TRANSPORT_OVERLAY_URL_PREFIX,
    CAI_OWNED_TRANSPORT_PAYLOAD_COMPRESSIONS,
    CAI_OWNED_TRANSPORT_PAYLOAD_RETENTION_SECONDS,
    CAI_OWNED_TRANSPORT_PROOF_SCHEMA_VERSION,
    CAI_OWNED_TRANSPORT_PROTOCOL,
    CAI_OWNED_TRANSPORT_PROTOCOL_VERSION,
    CAI_OWNED_TRANSPORT_REPLAY_CACHE_RETENTION_SECONDS,
    CAI_OWNED_TRANSPORT_REQUIRED_CAPABILITIES,
    CAI_OWNED_TRANSPORT_RUNTIME_PHASES,
    CAI_OWNED_TRANSPORT_RUNTIME_VERSION,
    CAI_OWNED_TRANSPORT_SESSION_OFFER_MAX_AGE_SECONDS,
    CAI_OWNED_TRANSPORT_SESSION_OFFER_SCHEMA_VERSION,
    EXECUTION_MODE_CAI_OWNED_TRANSPORT_REQUIRED,
    EXECUTION_MODE_LLAMA_CPP_RPC_LOW_LATENCY,
    EXECUTION_MODE_LLAMA_CPP_RPC_PROVEN_UNKNOWN_LATENCY,
    EXECUTION_MODE_SINGLE_NODE,
)
from .cai_owned_transport_receipts import (
    append_unique as _append_unique,
    append_unique_metric as _append_unique_metric,
    cai_owned_transport_proof_batch_ids as _cai_owned_transport_proof_batch_ids,
    cai_owned_transport_receipt_values as _cai_owned_transport_receipt_values,
    cai_owned_transport_shard_receipt_batch_ids as _cai_owned_transport_shard_receipt_batch_ids,
    clean_cai_owned_transport_receipt_audits as _clean_cai_owned_transport_receipt_audits,
    clean_cai_owned_transport_receipt_batch_ids as _clean_cai_owned_transport_receipt_batch_ids,
    clean_cai_owned_transport_receipt_hashes as _clean_cai_owned_transport_receipt_hashes,
    clean_cai_owned_transport_receipt_sequences as _clean_cai_owned_transport_receipt_sequences,
    clean_cai_owned_transport_receipt_stage_ids as _clean_cai_owned_transport_receipt_stage_ids,
    first_metric_value as _first_metric_value,
    max_receipt_count as _max_receipt_count,
)
from .cai_owned_transport_route_readiness import (
    cai_owned_transport_route_health_readiness,
    preflight_cai_owned_transport_data_plane_routes,
)
from .cai_owned_transport_versioning import (
    cai_owned_transport_int_list as _cai_owned_transport_int_list,
    cai_owned_transport_version_compatibility,
    cai_owned_transport_version_label as _cai_owned_transport_version_label,
)
from .cai_owned_transport_storage import (
    CaiOwnedTransportSessionRecord,
    cai_owned_transport_batch_output_payload_path,
    cai_owned_transport_batch_payload_path,
    cai_owned_transport_payload_replay_key,
    cai_owned_transport_payload_storage_root,
    cai_owned_transport_replay_cache_file_path,
    cai_owned_transport_sessions_file_path,
    cleanup_cai_owned_transport_payload_storage,
    cleanup_cai_owned_transport_replay_cache,
    create_cai_owned_transport_session,
    deterministic_cai_owned_transport_session_id,
    list_cai_owned_transport_replay_cache,
    list_cai_owned_transport_sessions,
    record_cai_owned_transport_payload_replay,
    save_cai_owned_transport_replay_cache,
    save_cai_owned_transport_sessions,
)
from .cai_owned_transport_summary_helpers import (
    coerce_byte_count as _coerce_byte_count,
    non_negative_int_or_default as _non_negative_int_or_default,
    summary_bool as _summary_bool,
    summary_available_ranges_from_candidate as _summary_available_ranges_from_candidate,
    summary_blocked_ranges_from_candidate as _summary_blocked_ranges_from_candidate,
    summary_cai_owned_transport_readiness as _summary_cai_owned_transport_readiness,
    summary_cached_shard_integrity_satisfied as _summary_cached_shard_integrity_satisfied,
    summary_can_load_before_deadline as _summary_can_load_before_deadline,
    summary_candidate_range_items as _summary_candidate_range_items,
    summary_deadline_expired as _summary_deadline_expired,
    summary_encrypted_cache_accessible as _summary_encrypted_cache_accessible,
    summary_item_already_materialized as _summary_item_already_materialized,
    summary_layer_range_item_block_error as _summary_layer_range_item_block_error,
    summary_layer_range_item_block_reason as _summary_layer_range_item_block_reason,
    summary_layer_range_item_ready as _summary_layer_range_item_ready,
    summary_layer_ranges as _summary_layer_ranges,
    summary_merge_layer_ranges as _summary_merge_layer_ranges,
    summary_missing_required_ranges as _summary_missing_required_ranges,
    summary_model_id_in_list as _summary_model_id_in_list,
    summary_model_id_matches as _summary_model_id_matches,
    summary_payload_model_matches as _summary_payload_model_matches,
    summary_resource_bytes as _summary_resource_bytes,
    summary_resource_headroom_audit as _summary_resource_headroom_audit,
    summary_resource_payload as _summary_resource_payload,
    summary_string_list as _summary_string_list,
    summary_string_values as _summary_string_values,
)


def build_cai_owned_transport_session_offer(
    *,
    instance_id: str,
    participant_node_ids: Sequence[str],
    executor_node_ids: Sequence[str] | None = None,
    chain_id: str | None = None,
    model_id: str | None = None,
    task_id: str | None = None,
    source_node_id: str | None = None,
    execution_mode: str | None = EXECUTION_MODE_CAI_OWNED_TRANSPORT_REQUIRED,
    route_policy: dict[str, Any] | None = None,
    session_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    clean_instance_id = str(instance_id or "").strip()
    participants = _clean_node_ids(participant_node_ids)
    executors = _clean_node_ids(executor_node_ids or participants)
    if not executors or not set(executors).issubset(set(participants)):
        raise ValueError(
            "CAI-owned transport session offer executors must be participants."
        )
    resolved_chain_id = _cai_owned_transport_chain_id(None, chain_id)
    deterministic_session_id = deterministic_cai_owned_transport_session_id(
        clean_instance_id,
        participants,
        executor_node_ids=executors,
        task_id=task_id,
        chain_id=resolved_chain_id,
    )
    clean_session_id = str(session_id or "").strip() or deterministic_session_id
    if clean_session_id != deterministic_session_id:
        raise ValueError("CAI-owned transport session offer id must be deterministic.")
    source = str(source_node_id or "").strip() or participants[0]
    if source not in participants:
        raise ValueError(
            "CAI-owned transport session offer source is not a participant."
        )
    return {
        "schemaVersion": CAI_OWNED_TRANSPORT_SESSION_OFFER_SCHEMA_VERSION,
        "protocol": CAI_OWNED_TRANSPORT_PROTOCOL,
        "protocolVersion": CAI_OWNED_TRANSPORT_PROTOCOL_VERSION,
        "network": resolved_chain_id,
        "chainId": resolved_chain_id,
        "sessionId": clean_session_id,
        "instanceId": clean_instance_id,
        "modelId": str(model_id or "").strip() or None,
        "taskId": str(task_id or "").strip() or None,
        "sourceNodeId": source,
        "participantNodeIds": participants,
        "executorNodeIds": executors,
        "executionMode": str(execution_mode or "").strip() or None,
        "routePolicy": dict(route_policy or {}),
        "createdAt": created_at or datetime.now(tz=UTC).isoformat(),
    }


def validate_cai_owned_transport_session_offer(
    offer: dict[str, Any] | None,
    *,
    session_id: str | None = None,
    local_node_id: str | None = None,
    chain_id: str | None = None,
    max_age_seconds: float | None = CAI_OWNED_TRANSPORT_SESSION_OFFER_MAX_AGE_SECONDS,
    now: datetime | None = None,
    require_signature: bool | None = None,
    trusted_signer_identities_by_node: Mapping[str, Any] | None = None,
    require_trusted_signer: bool = False,
    record_replay_cache: bool = False,
    replay_cache_policy: WalletPolicy | None = None,
    replay_cache_retention_seconds: float | int | None = None,
) -> tuple[bool, str | None]:
    if not isinstance(offer, dict):
        return False, "CAI-owned transport session offer is missing."
    if (
        int(offer.get("schemaVersion") or 0)
        != CAI_OWNED_TRANSPORT_SESSION_OFFER_SCHEMA_VERSION
    ):
        return False, "CAI-owned transport session offer schema is unsupported."
    if str(offer.get("protocol") or "").strip() != CAI_OWNED_TRANSPORT_PROTOCOL:
        return False, "CAI-owned transport session offer protocol is invalid."
    if int(offer.get("protocolVersion") or 0) != CAI_OWNED_TRANSPORT_PROTOCOL_VERSION:
        return (
            False,
            "CAI-owned transport session offer protocol version is unsupported.",
        )
    chain_valid, chain_error, offer_chain_id = _validate_cai_owned_transport_chain_id(
        offer,
        expected_chain_id=_cai_owned_transport_chain_id(None, chain_id),
        payload_name="session offer",
    )
    if not chain_valid:
        return False, chain_error
    created_valid, created_error = _validate_cai_owned_transport_created_at(
        offer,
        payload_name="session offer",
        max_age_seconds=max_age_seconds,
        now=now,
    )
    if not created_valid:
        return False, created_error

    clean_session_id = str(offer.get("sessionId") or "").strip()
    expected_session_id = str(session_id or "").strip()
    if not clean_session_id:
        return False, "CAI-owned transport session offer session id is missing."
    if expected_session_id and clean_session_id != expected_session_id:
        return False, "CAI-owned transport session offer session id does not match."

    clean_instance_id = str(offer.get("instanceId") or "").strip()
    if not clean_instance_id:
        return False, "CAI-owned transport session offer instance id is missing."
    participants = _clean_node_ids(offer.get("participantNodeIds") or [])
    if not participants:
        return False, "CAI-owned transport session offer participants are missing."
    executors = _clean_node_ids(offer.get("executorNodeIds") or participants)
    if not executors:
        return False, "CAI-owned transport session offer executors are missing."
    if not set(executors).issubset(set(participants)):
        return (
            False,
            "CAI-owned transport session offer executors must be participants.",
        )
    source = str(offer.get("sourceNodeId") or "").strip()
    if source and source not in participants:
        return False, "CAI-owned transport session offer source is not a participant."
    local = str(local_node_id or "").strip()
    if local and local not in participants:
        return False, "CAI-owned transport session offer does not include local node."
    signature_valid, signature_error = validate_cai_owned_transport_payload_signature(
        offer,
        payload_name="CAI-owned transport session offer",
        expected_signer_node_id=source or participants[0],
        allowed_signer_node_ids=participants,
        require_signature=require_signature,
        trusted_signer_identities_by_node=trusted_signer_identities_by_node,
        require_trusted_signer=require_trusted_signer,
        record_replay_cache=record_replay_cache,
        replay_cache_policy=replay_cache_policy,
        replay_cache_retention_seconds=replay_cache_retention_seconds,
    )
    if not signature_valid:
        return False, signature_error

    try:
        deterministic_session_id = deterministic_cai_owned_transport_session_id(
            clean_instance_id,
            participants,
            executor_node_ids=executors,
            task_id=offer.get("taskId"),
            chain_id=offer_chain_id,
        )
    except ValueError as exc:
        return False, str(exc)
    if clean_session_id != deterministic_session_id:
        return False, "CAI-owned transport session offer id is not deterministic."
    return True, None


def create_cai_owned_transport_session_from_offer(
    offer: dict[str, Any],
    *,
    session_id: str | None = None,
    local_node_id: str | None = None,
    record_replay_cache: bool = False,
    replay_cache_retention_seconds: float | int | None = None,
    policy: WalletPolicy | None = None,
) -> CaiOwnedTransportSessionRecord:
    valid, error = validate_cai_owned_transport_session_offer(
        offer,
        session_id=session_id,
        local_node_id=local_node_id,
        chain_id=_cai_owned_transport_chain_id(policy),
        record_replay_cache=record_replay_cache,
        replay_cache_policy=policy,
        replay_cache_retention_seconds=replay_cache_retention_seconds,
    )
    if not valid:
        raise ValueError(error or "CAI-owned transport session offer is invalid.")
    return create_cai_owned_transport_session(
        session_id=str(offer.get("sessionId") or "").strip(),
        instance_id=str(offer.get("instanceId") or "").strip(),
        participant_node_ids=offer.get("participantNodeIds") or [],
        executor_node_ids=offer.get("executorNodeIds") or None,
        chain_id=_cai_owned_transport_payload_chain_id(offer),
        model_id=offer.get("modelId"),
        task_id=offer.get("taskId"),
        source_node_id=offer.get("sourceNodeId"),
        execution_mode=offer.get("executionMode"),
        route_policy=offer.get("routePolicy")
        if isinstance(offer.get("routePolicy"), dict)
        else None,
        policy=policy,
    )


def submit_cai_owned_transport_session_offer(
    peer_cai_url: str,
    offer: dict[str, Any],
    *,
    chain_id: str | None = None,
    timeout_sec: float = 5.0,
) -> dict[str, Any]:
    resolved_chain_id = chain_id or _cai_owned_transport_payload_chain_id(offer)
    valid, error = validate_cai_owned_transport_session_offer(
        offer,
        chain_id=resolved_chain_id,
    )
    if not valid:
        raise ValueError(error or "CAI-owned transport session offer is invalid.")
    overlay_target = _parse_cai_owned_transport_overlay_url(peer_cai_url)
    if overlay_target is not None:
        relay_url, target_node_id = overlay_target
        return submit_cai_owned_transport_overlay_message(
            relay_url,
            kind="session_offer",
            source_node_id=str(offer.get("sourceNodeId") or "").strip(),
            target_node_id=target_node_id,
            session_id=str(offer.get("sessionId") or "").strip(),
            payload=offer,
            timeout_sec=timeout_sec,
        )
    base_url = str(peer_cai_url or "").strip().rstrip("/")
    if not base_url:
        raise ValueError("Peer CAI URL is required.")
    session_id = str(offer.get("sessionId") or "").strip()
    url = (
        f"{base_url}/v1/cai/transport/sessions/"
        f"{quote(session_id, safe='')}/offer"
    )
    request = Request(
        url,
        data=json.dumps(offer).encode("utf-8"),
        headers=_cai_owned_transport_json_headers(chain_id=resolved_chain_id),
        method="POST",
    )
    with urlopen(request, timeout=timeout_sec) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw or "{}")
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def submit_cai_owned_transport_session_offer_to_any(
    peer_cai_urls: Sequence[str],
    offer: dict[str, Any],
    *,
    chain_id: str | None = None,
    timeout_sec: float = 5.0,
) -> dict[str, Any]:
    urls = _prioritized_cai_owned_transport_peer_urls(peer_cai_urls)
    if not urls:
        raise ValueError("At least one peer CAI URL is required.")
    last_error: Exception | None = None
    attempts: list[dict[str, Any]] = []
    for url in urls:
        try:
            response = submit_cai_owned_transport_session_offer(
                url,
                offer,
                chain_id=chain_id,
                timeout_sec=timeout_sec,
            )
            attempts.append(
                {
                    "peerCaiUrl": url,
                    "status": "ok",
                    "selected": True,
                }
            )
            response.setdefault("peerCaiUrl", url)
            response.setdefault("attemptedPeerCaiUrlCount", len(urls))
            response.setdefault(
                "routeAudit",
                {
                    "selectedRoute": response.get("selectedRoute") or "peer_cai_api",
                    "selectedPeerCaiUrl": url,
                    "attemptedPeerCaiUrlCount": len(urls),
                    "fallbackCount": max(0, len(attempts) - 1),
                    "attempts": attempts,
                },
            )
            return response
        except Exception as exc:
            last_error = exc
            attempts.append(
                {
                    "peerCaiUrl": url,
                    "status": "failed",
                    "selected": False,
                    "error": str(exc),
                }
            )
    raise ValueError(
        "Unable to submit CAI-owned transport session offer to any peer URL: "
        f"{last_error}"
    ) from last_error


def submit_cai_owned_transport_batch_envelope(
    peer_cai_url: str,
    session_id: str,
    envelope: dict[str, Any],
    *,
    chain_id: str | None = None,
    timeout_sec: float = 5.0,
) -> dict[str, Any]:
    clean_session_id = str(session_id or "").strip()
    resolved_chain_id = chain_id or _cai_owned_transport_payload_chain_id(envelope)
    valid, error = validate_cai_owned_transport_batch_envelope(
        envelope,
        session_id=clean_session_id,
        chain_id=resolved_chain_id,
    )
    if not valid:
        raise ValueError(error or "CAI-owned transport batch envelope is invalid.")
    overlay_target = _parse_cai_owned_transport_overlay_url(peer_cai_url)
    if overlay_target is not None:
        relay_url, target_node_id = overlay_target
        return submit_cai_owned_transport_overlay_message(
            relay_url,
            kind="batch_envelope",
            source_node_id=str(envelope.get("sourceNodeId") or "").strip(),
            target_node_id=target_node_id,
            session_id=clean_session_id,
            payload=envelope,
            timeout_sec=timeout_sec,
        )
    base_url = str(peer_cai_url or "").strip().rstrip("/")
    if not base_url:
        raise ValueError("Peer CAI URL is required.")
    url = (
        f"{base_url}/v1/cai/transport/sessions/"
        f"{quote(clean_session_id, safe='')}/batch-envelopes"
    )
    request = Request(
        url,
        data=json.dumps(envelope).encode("utf-8"),
        headers=_cai_owned_transport_json_headers(chain_id=resolved_chain_id),
        method="POST",
    )
    with urlopen(request, timeout=timeout_sec) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw or "{}")
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def submit_cai_owned_transport_batch_envelope_to_any(
    peer_cai_urls: Sequence[str],
    session_id: str,
    envelope: dict[str, Any],
    *,
    chain_id: str | None = None,
    timeout_sec: float = 5.0,
) -> dict[str, Any]:
    urls = _prioritized_cai_owned_transport_peer_urls(peer_cai_urls)
    if not urls:
        raise ValueError("At least one peer CAI URL is required.")
    last_error: Exception | None = None
    attempts: list[dict[str, Any]] = []
    for url in urls:
        try:
            response = submit_cai_owned_transport_batch_envelope(
                url,
                session_id,
                envelope,
                chain_id=chain_id,
                timeout_sec=timeout_sec,
            )
            attempts.append(
                {
                    "peerCaiUrl": url,
                    "status": "ok",
                    "selected": True,
                }
            )
            response.setdefault("peerCaiUrl", url)
            response.setdefault("attemptedPeerCaiUrlCount", len(urls))
            response.setdefault(
                "routeAudit",
                {
                    "selectedRoute": response.get("selectedRoute") or "peer_cai_api",
                    "selectedPeerCaiUrl": url,
                    "attemptedPeerCaiUrlCount": len(urls),
                    "fallbackCount": max(0, len(attempts) - 1),
                    "attempts": attempts,
                },
            )
            return response
        except Exception as exc:
            last_error = exc
            attempts.append(
                {
                    "peerCaiUrl": url,
                    "status": "failed",
                    "selected": False,
                    "error": str(exc),
                }
            )
    raise ValueError(
        "Unable to submit CAI-owned transport batch envelope to any peer URL: "
        f"{last_error}"
    ) from last_error


def submit_cai_owned_transport_overlay_message(
    relay_cai_url: str,
    *,
    kind: str,
    source_node_id: str,
    target_node_id: str,
    session_id: str,
    payload: dict[str, Any],
    timeout_sec: float = 5.0,
) -> dict[str, Any]:
    base_url = str(relay_cai_url or "").strip().rstrip("/")
    clean_kind = str(kind or "").strip()
    clean_source = str(source_node_id or "").strip()
    clean_target = str(target_node_id or "").strip()
    clean_session_id = str(session_id or "").strip()
    if not base_url:
        raise ValueError("CAI-owned transport overlay relay CAI URL is required.")
    if clean_kind not in {
        "session_offer",
        "batch_envelope",
        "shard_receipt",
        "completion_notice",
    }:
        raise ValueError("CAI-owned transport overlay message kind is unsupported.")
    if not clean_target:
        raise ValueError("CAI-owned transport overlay target node id is required.")
    if not clean_session_id:
        raise ValueError("CAI-owned transport overlay session id is required.")

    message = {
        "messageId": f"caiovl_{secrets.token_hex(12)}",
        "kind": clean_kind,
        "sourceNodeId": clean_source,
        "targetNodeId": clean_target,
        "sessionId": clean_session_id,
        "payload": dict(payload or {}),
        "createdAt": datetime.now(tz=UTC).isoformat(),
    }
    url = f"{base_url}/v1/cai/transport/overlay/send"
    request = Request(
        url,
        data=json.dumps(message).encode("utf-8"),
        headers=_cai_owned_transport_json_headers(
            chain_id=_cai_owned_transport_payload_chain_id(payload),
        ),
        method="POST",
    )
    with urlopen(request, timeout=timeout_sec) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw or "{}")
    response_payload = parsed if isinstance(parsed, dict) else {"value": parsed}
    response_payload.setdefault("selectedRoute", "cai_overlay_gossipsub")
    response_payload.setdefault("relayCaiUrl", base_url)
    response_payload.setdefault("targetNodeId", clean_target)
    response_payload.setdefault("messageId", message["messageId"])
    return response_payload


def build_cai_owned_transport_frame_metadata(
    *,
    model_id: str,
    frame_kind: str = "activation",
    tokenizer_config_hash: str | None = None,
    layer_start: int | None = None,
    layer_end: int | None = None,
    token_start: int | None = None,
    token_end: int | None = None,
    dtype: str | None = None,
    shape: Sequence[int] | None = None,
    sequence: int = 0,
    payload_sha256_hex: str | None = None,
    kv_cache_metadata: dict[str, Any] | None = None,
    activation_metadata: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "frameSchemaVersion": CAI_OWNED_TRANSPORT_FRAME_SCHEMA_VERSION,
        "frameKind": str(frame_kind or "").strip(),
        "modelId": str(model_id or "").strip(),
        "sequence": max(0, int(sequence or 0)),
    }
    if tokenizer_config_hash is not None:
        metadata["tokenizerConfigHash"] = _normalize_sha256_hex(
            tokenizer_config_hash,
            field_name="tokenizerConfigHash",
        )
    if layer_start is not None:
        metadata["layerStart"] = int(layer_start)
    if layer_end is not None:
        metadata["layerEnd"] = int(layer_end)
    if token_start is not None:
        metadata["tokenStart"] = int(token_start)
    if token_end is not None:
        metadata["tokenEnd"] = int(token_end)
    if dtype is not None:
        metadata["dtype"] = str(dtype or "").strip()
    if shape is not None:
        metadata["shape"] = [int(item) for item in shape]
    if payload_sha256_hex is not None:
        metadata["payloadSha256Hex"] = _normalize_sha256_hex(
            payload_sha256_hex,
            field_name="payloadSha256Hex",
        )
    if kv_cache_metadata is not None:
        metadata["kvCacheMetadata"] = _jsonable_dict(
            kv_cache_metadata,
            field_name="kvCacheMetadata",
        )
    if activation_metadata is not None:
        metadata["activationMetadata"] = _jsonable_dict(
            activation_metadata,
            field_name="activationMetadata",
        )
    if extra_metadata is not None:
        metadata["extraMetadata"] = _jsonable_dict(
            extra_metadata,
            field_name="extraMetadata",
        )
    valid, error = validate_cai_owned_transport_frame_metadata(metadata)
    if not valid:
        raise ValueError(error or "CAI-owned transport frame metadata is invalid.")
    return metadata


def build_cai_owned_llm_handoff_metadata(
    *,
    model_id: str,
    backend: str = "llama.cpp-patched",
    backend_version: str | None = None,
    model_sha256_hex: str | None = None,
    tokenizer_config_hash: str | None = None,
    layer_start: int | None = None,
    layer_end: int | None = None,
    token_start: int | None = None,
    token_end: int | None = None,
    tensor_name: str = "activation",
    tensor_dtype: str,
    tensor_shape: Sequence[int],
    tensor_encoding: str = "raw-le",
    tensor_sha256_hex: str | None = None,
    kv_cache: dict[str, Any] | None = None,
    decode_state: dict[str, Any] | None = None,
    requires_patched_backend: bool = True,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    clean_model_id = str(model_id or "").strip()
    if not clean_model_id:
        raise ValueError("CAI-owned LLM handoff model id is required.")
    handoff = {
        "schemaVersion": CAI_OWNED_LLM_HANDOFF_SCHEMA_VERSION,
        "abi": CAI_OWNED_LLM_HANDOFF_ABI,
        "backend": str(backend or "").strip(),
        "backendVersion": str(backend_version or "").strip() or None,
        "modelId": clean_model_id,
        "requiresPatchedBackend": bool(requires_patched_backend),
        "tensor": {
            "name": str(tensor_name or "").strip() or "activation",
            "dtype": str(tensor_dtype or "").strip(),
            "shape": [int(item) for item in tensor_shape],
            "encoding": str(tensor_encoding or "").strip(),
        },
    }
    if model_sha256_hex is not None:
        handoff["modelSha256Hex"] = _normalize_sha256_hex(
            model_sha256_hex,
            field_name="modelSha256Hex",
        )
    if tokenizer_config_hash is not None:
        handoff["tokenizerConfigHash"] = _normalize_sha256_hex(
            tokenizer_config_hash,
            field_name="tokenizerConfigHash",
        )
    if layer_start is not None:
        handoff["layerStart"] = int(layer_start)
    if layer_end is not None:
        handoff["layerEnd"] = int(layer_end)
    if token_start is not None:
        handoff["tokenStart"] = int(token_start)
    if token_end is not None:
        handoff["tokenEnd"] = int(token_end)
    if tensor_sha256_hex is not None:
        handoff["tensor"]["sha256Hex"] = _normalize_sha256_hex(
            tensor_sha256_hex,
            field_name="tensor.sha256Hex",
        )
    if kv_cache is not None:
        handoff["kvCache"] = _jsonable_dict(kv_cache, field_name="kvCache")
    if decode_state is not None:
        handoff["decodeState"] = _jsonable_dict(
            decode_state,
            field_name="decodeState",
        )
    if extra_metadata is not None:
        handoff["extraMetadata"] = _jsonable_dict(
            extra_metadata,
            field_name="extraMetadata",
        )
    valid, error = validate_cai_owned_llm_handoff_metadata(handoff)
    if not valid:
        raise ValueError(error or "CAI-owned LLM handoff metadata is invalid.")
    return handoff


def build_cai_owned_llm_shard_frame_metadata_from_runtime(
    *,
    model_id: str,
    runtime_metadata: Mapping[str, Any],
    payload: bytes,
    layer_start: int,
    layer_end: int,
    frame_kind: str = "activation",
    token_start: int = 0,
    token_end: int | None = None,
    sequence: int = 0,
    backend: str | None = None,
    backend_version: str | None = None,
    tensor_name: str | None = None,
    tensor_shape: Sequence[int] | None = None,
    tensor_dtype: str | None = None,
    tensor_encoding: str | None = None,
    kv_cache: dict[str, Any] | None = None,
    decode_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(runtime_metadata, Mapping):
        raise ValueError("CAI-owned LLM runtime metadata is required.")
    clean_model_id = str(model_id or "").strip()
    if not clean_model_id:
        raise ValueError("CAI-owned LLM runtime metadata model id is required.")
    runtime_model_id = _runtime_metadata_text(runtime_metadata, "modelId", "model_id")
    if runtime_model_id and runtime_model_id != clean_model_id:
        raise ValueError("CAI-owned LLM runtime metadata model id does not match.")
    total_layers = _runtime_metadata_int(
        runtime_metadata,
        "totalLayerCount",
        "totalLayers",
        "total_layers",
        "nLayer",
        "n_layer",
        "blockCount",
        "block_count",
    )
    clean_layer_start = int(layer_start)
    clean_layer_end = int(layer_end)
    if total_layers is None or total_layers <= 0:
        raise ValueError("CAI-owned LLM runtime metadata total layers is required.")
    if (
        clean_layer_start < 0
        or clean_layer_end <= clean_layer_start
        or clean_layer_end > total_layers
    ):
        raise ValueError("CAI-owned LLM shard layer range exceeds runtime metadata.")
    _require_runtime_metadata_layer_range_supported(
        runtime_metadata,
        model_id=clean_model_id,
    )
    clean_token_start = int(token_start)
    clean_token_end = (
        int(token_end)
        if token_end is not None
        else clean_token_start + 1
    )
    if clean_token_start < 0 or clean_token_end < clean_token_start:
        raise ValueError("CAI-owned LLM shard token window is invalid.")
    token_count = max(1, clean_token_end - clean_token_start)

    resolved_tensor_dtype = (
        str(tensor_dtype or "").strip()
        or _runtime_metadata_text(
            runtime_metadata,
            "activationDtype",
            "activation_dtype",
            "tensorDtype",
            "tensor_dtype",
        )
        or "f16"
    )
    resolved_tensor_encoding = (
        str(tensor_encoding or "").strip()
        or _runtime_metadata_text(
            runtime_metadata,
            "tensorEncoding",
            "tensor_encoding",
            "activationTensorEncoding",
            "activation_tensor_encoding",
        )
        or "ggml-tensor-v1"
    )
    resolved_tensor_shape = (
        [int(item) for item in tensor_shape]
        if tensor_shape is not None
        else _runtime_metadata_shape(runtime_metadata)
    )
    if resolved_tensor_shape is None:
        hidden_size = _runtime_metadata_int(
            runtime_metadata,
            "hiddenSize",
            "hidden_size",
            "embeddingLength",
            "embedding_length",
            "nEmbd",
            "n_embd",
        )
        if hidden_size is None or hidden_size <= 0:
            raise ValueError("CAI-owned LLM runtime metadata hidden size is required.")
        resolved_tensor_shape = [1, token_count, hidden_size]
    payload_hash = hashlib.sha256(bytes(payload or b"")).hexdigest()
    tokenizer_config_hash = _runtime_metadata_text(
        runtime_metadata,
        "tokenizerConfigHash",
        "tokenizer_config_hash",
    )
    model_sha256_hex = _runtime_metadata_text(
        runtime_metadata,
        "modelSha256Hex",
        "model_sha256_hex",
        "ggufSha256Hex",
        "gguf_sha256_hex",
    )
    resolved_backend = (
        str(backend or "").strip()
        or _runtime_metadata_text(runtime_metadata, "backend")
        or "llama.cpp-patched"
    )
    resolved_backend_version = (
        str(backend_version or "").strip()
        or _runtime_metadata_text(
            runtime_metadata,
            "backendVersion",
            "backend_version",
        )
    )
    handoff = build_cai_owned_llm_handoff_metadata(
        model_id=clean_model_id,
        backend=resolved_backend,
        backend_version=resolved_backend_version,
        model_sha256_hex=model_sha256_hex,
        tokenizer_config_hash=tokenizer_config_hash,
        layer_start=clean_layer_start,
        layer_end=clean_layer_end,
        token_start=clean_token_start,
        token_end=clean_token_end,
        tensor_name=tensor_name or str(frame_kind or "").strip() or "activation",
        tensor_dtype=resolved_tensor_dtype,
        tensor_shape=resolved_tensor_shape,
        tensor_encoding=resolved_tensor_encoding,
        tensor_sha256_hex=payload_hash,
        kv_cache=kv_cache
        if kv_cache is not None
        else _runtime_metadata_mapping(runtime_metadata, "kvCache", "kv_cache"),
        decode_state=decode_state
        if decode_state is not None
        else _runtime_metadata_mapping(
            runtime_metadata,
            "decodeState",
            "decode_state",
        ),
        extra_metadata={
            "totalLayerCount": total_layers,
            "runtimeMetadataSource": _runtime_metadata_text(
                runtime_metadata,
                "metadataSource",
                "metadata_source",
            )
            or "runtime",
            **_runtime_metadata_external_shard_descriptor(runtime_metadata),
        },
    )
    metadata = build_cai_owned_transport_frame_metadata(
        model_id=clean_model_id,
        frame_kind=frame_kind,
        tokenizer_config_hash=tokenizer_config_hash,
        layer_start=clean_layer_start,
        layer_end=clean_layer_end,
        token_start=clean_token_start,
        token_end=clean_token_end,
        dtype=resolved_tensor_dtype,
        shape=resolved_tensor_shape,
        sequence=sequence,
        payload_sha256_hex=payload_hash,
        extra_metadata={
            "productionLlmHandoff": True,
            "totalLayerCount": total_layers,
            "backend": resolved_backend,
            **_runtime_metadata_external_shard_descriptor(runtime_metadata),
        },
    )
    metadata["llmHandoff"] = handoff
    valid, error = validate_cai_owned_transport_frame_metadata(
        metadata,
        expected_model_id=clean_model_id,
        require_llm_handoff=True,
    )
    if not valid:
        raise ValueError(error or "CAI-owned LLM shard frame metadata is invalid.")
    return metadata


def validate_cai_owned_transport_frame_metadata(
    metadata: dict[str, Any] | None,
    *,
    expected_model_id: str | None = None,
    require_llm_handoff: bool = False,
) -> tuple[bool, str | None]:
    if not isinstance(metadata, dict):
        return False, "CAI-owned transport frame metadata is missing."
    try:
        schema_version = int(metadata.get("frameSchemaVersion") or 0)
    except (TypeError, ValueError):
        return False, "CAI-owned transport frame schema is invalid."
    if schema_version != CAI_OWNED_TRANSPORT_FRAME_SCHEMA_VERSION:
        return False, "CAI-owned transport frame schema is unsupported."

    frame_kind = str(metadata.get("frameKind") or "").strip()
    if frame_kind not in CAI_OWNED_TRANSPORT_FRAME_KINDS:
        return False, "CAI-owned transport frame kind is unsupported."
    model_id = str(metadata.get("modelId") or "").strip()
    if not model_id:
        return False, "CAI-owned transport frame model id is missing."
    expected = str(expected_model_id or "").strip()
    if expected and model_id != expected:
        return False, "CAI-owned transport frame model id does not match."

    for field_name in ("tokenizerConfigHash", "payloadSha256Hex"):
        if metadata.get(field_name) is not None:
            try:
                _normalize_sha256_hex(metadata.get(field_name), field_name=field_name)
            except ValueError as exc:
                return False, str(exc)

    layer_start = _optional_int(metadata.get("layerStart"))
    layer_end = _optional_int(metadata.get("layerEnd"))
    if metadata.get("layerStart") is not None and layer_start is None:
        return False, "CAI-owned transport frame layer range is invalid."
    if metadata.get("layerEnd") is not None and layer_end is None:
        return False, "CAI-owned transport frame layer range is invalid."
    if layer_start is not None and layer_start < 0:
        return False, "CAI-owned transport frame layer range is invalid."
    if layer_end is not None and layer_end < 0:
        return False, "CAI-owned transport frame layer range is invalid."
    if layer_start is not None and layer_end is not None and layer_end < layer_start:
        return False, "CAI-owned transport frame layer range is invalid."

    token_start = _optional_int(metadata.get("tokenStart"))
    token_end = _optional_int(metadata.get("tokenEnd"))
    if metadata.get("tokenStart") is not None and token_start is None:
        return False, "CAI-owned transport frame token window is invalid."
    if metadata.get("tokenEnd") is not None and token_end is None:
        return False, "CAI-owned transport frame token window is invalid."
    if token_start is not None and token_start < 0:
        return False, "CAI-owned transport frame token window is invalid."
    if token_end is not None and token_end < 0:
        return False, "CAI-owned transport frame token window is invalid."
    if token_start is not None and token_end is not None and token_end < token_start:
        return False, "CAI-owned transport frame token window is invalid."

    dtype = metadata.get("dtype")
    if dtype is not None:
        clean_dtype = str(dtype or "").strip()
        if not clean_dtype or not clean_dtype.isascii():
            return False, "CAI-owned transport frame dtype is invalid."
    shape = metadata.get("shape")
    if shape is not None:
        if not isinstance(shape, (list, tuple)):
            return False, "CAI-owned transport frame shape is invalid."
        for item in shape:
            try:
                dimension = int(item)
            except (TypeError, ValueError):
                return False, "CAI-owned transport frame shape is invalid."
            if dimension < 0:
                return False, "CAI-owned transport frame shape is invalid."
    for field_name in ("kvCacheMetadata", "activationMetadata", "extraMetadata"):
        if metadata.get(field_name) is not None and not isinstance(
            metadata.get(field_name),
            dict,
        ):
            return False, f"CAI-owned transport frame {field_name} is invalid."
    handoff = metadata.get("llmHandoff")
    if handoff is not None or require_llm_handoff:
        handoff_valid, handoff_error = validate_cai_owned_llm_handoff_metadata(
            handoff,
            expected_model_id=model_id,
            expected_frame_metadata=metadata,
        )
        if not handoff_valid:
            return False, handoff_error
    return True, None


def validate_cai_owned_llm_handoff_metadata(
    handoff: Any,
    *,
    expected_model_id: str | None = None,
    expected_frame_metadata: Mapping[str, Any] | None = None,
) -> tuple[bool, str | None]:
    if not isinstance(handoff, Mapping):
        return False, "CAI-owned LLM handoff metadata is missing."
    try:
        schema_version = int(handoff.get("schemaVersion") or 0)
    except (TypeError, ValueError):
        return False, "CAI-owned LLM handoff schema is invalid."
    if schema_version != CAI_OWNED_LLM_HANDOFF_SCHEMA_VERSION:
        return False, "CAI-owned LLM handoff schema is unsupported."
    if str(handoff.get("abi") or "").strip() != CAI_OWNED_LLM_HANDOFF_ABI:
        return False, "CAI-owned LLM handoff ABI is unsupported."

    backend = str(handoff.get("backend") or "").strip()
    if not backend or not backend.isascii():
        return False, "CAI-owned LLM handoff backend is invalid."
    if not bool(handoff.get("requiresPatchedBackend")):
        return False, "CAI-owned LLM handoff must require a patched backend."

    model_id = str(handoff.get("modelId") or "").strip()
    if not model_id:
        return False, "CAI-owned LLM handoff model id is missing."
    expected = str(expected_model_id or "").strip()
    if expected and model_id != expected:
        return False, "CAI-owned LLM handoff model id does not match."

    for field_name in ("modelSha256Hex", "tokenizerConfigHash"):
        if handoff.get(field_name) is not None:
            try:
                _normalize_sha256_hex(handoff.get(field_name), field_name=field_name)
            except ValueError as exc:
                return False, str(exc)

    for start_field, end_field, label in (
        ("layerStart", "layerEnd", "layer range"),
        ("tokenStart", "tokenEnd", "token window"),
    ):
        start = _optional_int(handoff.get(start_field))
        end = _optional_int(handoff.get(end_field))
        if handoff.get(start_field) is not None and start is None:
            return False, f"CAI-owned LLM handoff {label} is invalid."
        if handoff.get(end_field) is not None and end is None:
            return False, f"CAI-owned LLM handoff {label} is invalid."
        if start is not None and start < 0:
            return False, f"CAI-owned LLM handoff {label} is invalid."
        if end is not None and end < 0:
            return False, f"CAI-owned LLM handoff {label} is invalid."
        if start is not None and end is not None and end < start:
            return False, f"CAI-owned LLM handoff {label} is invalid."
        if isinstance(expected_frame_metadata, Mapping):
            frame_start = _optional_int(expected_frame_metadata.get(start_field))
            frame_end = _optional_int(expected_frame_metadata.get(end_field))
            if frame_start is not None and start is not None and start != frame_start:
                return False, f"CAI-owned LLM handoff {label} does not match frame."
            if frame_end is not None and end is not None and end != frame_end:
                return False, f"CAI-owned LLM handoff {label} does not match frame."

    tensor = handoff.get("tensor")
    if not isinstance(tensor, Mapping):
        return False, "CAI-owned LLM handoff tensor is missing."
    tensor_dtype = str(tensor.get("dtype") or "").strip()
    if not tensor_dtype or not tensor_dtype.isascii():
        return False, "CAI-owned LLM handoff tensor dtype is invalid."
    tensor_encoding = str(tensor.get("encoding") or "").strip()
    if tensor_encoding not in CAI_OWNED_LLM_HANDOFF_TENSOR_ENCODINGS:
        return False, "CAI-owned LLM handoff tensor encoding is unsupported."
    tensor_shape = tensor.get("shape")
    if not isinstance(tensor_shape, Sequence) or isinstance(
        tensor_shape,
        (str, bytes),
    ):
        return False, "CAI-owned LLM handoff tensor shape is invalid."
    if not tensor_shape:
        return False, "CAI-owned LLM handoff tensor shape is invalid."
    for value in tensor_shape:
        try:
            dimension = int(value)
        except (TypeError, ValueError):
            return False, "CAI-owned LLM handoff tensor shape is invalid."
        if dimension <= 0:
            return False, "CAI-owned LLM handoff tensor shape is invalid."
    if tensor.get("sha256Hex") is not None:
        try:
            tensor_hash = _normalize_sha256_hex(
                tensor.get("sha256Hex"),
                field_name="tensor.sha256Hex",
            )
        except ValueError as exc:
            return False, str(exc)
        if isinstance(expected_frame_metadata, Mapping):
            frame_payload_hash = str(
                expected_frame_metadata.get("payloadSha256Hex") or ""
            ).strip().lower()
            if frame_payload_hash and tensor_hash != frame_payload_hash:
                return False, (
                    "CAI-owned LLM handoff tensor hash does not match frame payload."
                )
    for field_name in ("kvCache", "decodeState", "extraMetadata"):
        if handoff.get(field_name) is not None and not isinstance(
            handoff.get(field_name),
            Mapping,
        ):
            return False, f"CAI-owned LLM handoff {field_name} is invalid."
    return True, None


def build_cai_owned_transport_batch_hash_chain(
    *,
    session_id: str,
    batch_id: str,
    input_payload_sha256_hex: str,
    output_payload_sha256_hex: str,
    sequence: int = 0,
    previous_batch_id: str | None = None,
) -> dict[str, Any]:
    clean_session_id = str(session_id or "").strip()
    if not clean_session_id:
        raise ValueError("CAI-owned transport hash chain session id is required.")
    clean_batch_id = _require_safe_transport_file_id(batch_id, prefix="caibatch_")
    clean_previous_batch_id = str(previous_batch_id or "").strip() or None
    if clean_previous_batch_id is not None:
        clean_previous_batch_id = _require_safe_transport_file_id(
            clean_previous_batch_id,
            prefix="caibatch_",
        )
    chain_without_digest = {
        "schemaVersion": CAI_OWNED_TRANSPORT_HASH_CHAIN_SCHEMA_VERSION,
        "sessionId": clean_session_id,
        "batchId": clean_batch_id,
        "sequence": max(0, int(sequence or 0)),
        "previousBatchId": clean_previous_batch_id,
        "inputPayloadSha256Hex": _normalize_sha256_hex(
            input_payload_sha256_hex,
            field_name="inputPayloadSha256Hex",
        ),
        "outputPayloadSha256Hex": _normalize_sha256_hex(
            output_payload_sha256_hex,
            field_name="outputPayloadSha256Hex",
        ),
    }
    digest = hashlib.sha256(
        json.dumps(
            chain_without_digest,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        **chain_without_digest,
        "hashChainSha256Hex": digest,
    }


def build_cai_owned_transport_batch_envelope(
    *,
    session_id: str,
    phase: str,
    source_node_id: str,
    sink_node_id: str,
    sequence: int,
    payload: bytes,
    chain_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    route_audit: dict[str, Any] | None = None,
    payload_compression: str | None = None,
    payload_chunk_size_bytes: int | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    clean_session_id = str(session_id or "").strip()
    clean_phase = str(phase or "").strip()
    source = str(source_node_id or "").strip()
    sink = str(sink_node_id or "").strip()
    if not clean_session_id:
        raise ValueError("CAI-owned transport batch envelope requires session id.")
    if clean_phase not in CAI_OWNED_TRANSPORT_BATCH_PHASES:
        raise ValueError("CAI-owned transport batch envelope phase is unsupported.")
    if not source or not sink:
        raise ValueError("CAI-owned transport batch envelope requires source and sink.")
    resolved_chain_id = _cai_owned_transport_chain_id(None, chain_id)
    payload_bytes = bytes(payload or b"")
    payload_sha256_hex = hashlib.sha256(payload_bytes).hexdigest()
    compression = _normalize_cai_owned_transport_payload_compression(
        payload_compression,
    )
    encoded_payload_bytes = _encode_cai_owned_transport_payload_bytes(
        payload_bytes,
        compression=compression,
    )
    payload_fields = _cai_owned_transport_encoded_payload_fields(
        encoded_payload_bytes,
        chunk_size_bytes=payload_chunk_size_bytes,
    )
    clean_sequence = max(0, int(sequence or 0))
    batch_id = _cai_owned_transport_batch_id(
        session_id=clean_session_id,
        phase=clean_phase,
        source_node_id=source,
        sink_node_id=sink,
        sequence=clean_sequence,
        payload_sha256_hex=payload_sha256_hex,
    )
    envelope = {
        "schemaVersion": CAI_OWNED_TRANSPORT_BATCH_ENVELOPE_SCHEMA_VERSION,
        "protocol": CAI_OWNED_TRANSPORT_PROTOCOL,
        "protocolVersion": CAI_OWNED_TRANSPORT_PROTOCOL_VERSION,
        "network": resolved_chain_id,
        "chainId": resolved_chain_id,
        "batchId": batch_id,
        "sessionId": clean_session_id,
        "phase": clean_phase,
        "sourceNodeId": source,
        "sinkNodeId": sink,
        "sequence": clean_sequence,
        "payloadEncoding": "base64",
        "payloadSizeBytes": len(payload_bytes),
        "payloadSha256Hex": payload_sha256_hex,
        "metadata": dict(metadata or {}),
        "createdAt": created_at or datetime.now(tz=UTC).isoformat(),
    }
    envelope.update(payload_fields)
    if compression is not None or "payloadChunksBase64" in payload_fields:
        envelope["payloadEncodedSizeBytes"] = len(encoded_payload_bytes)
    if compression is not None:
        envelope["payloadCompression"] = compression
    if route_audit is not None:
        envelope["routeAudit"] = _jsonable_dict(
            route_audit,
            field_name="routeAudit",
        )
    return envelope


def build_cai_owned_transport_output_batch_envelope(
    *,
    session_id: str,
    source_batch_id: str,
    sink_node_id: str,
    source_node_id: str | None = None,
    phase: str | None = None,
    sequence: int | None = None,
    metadata: dict[str, Any] | None = None,
    output_payload: bytes | None = None,
    output_payload_sha256_hex: str | None = None,
    policy: WalletPolicy | None = None,
) -> dict[str, Any]:
    clean_session_id = str(session_id or "").strip()
    clean_source_batch_id = _require_safe_transport_file_id(
        source_batch_id,
        prefix="caibatch_",
    )
    record = _cai_owned_transport_session_record(clean_session_id, policy)
    if record is None:
        raise ValueError(
            f"CAI-owned transport session '{clean_session_id}' not found."
        )
    source_batch = _find_cai_owned_transport_batch(record, clean_source_batch_id)
    if source_batch is None:
        raise FileNotFoundError(
            f"CAI-owned transport batch not found: {clean_source_batch_id}"
        )
    source = _require_session_participant(
        record,
        source_node_id or str(source_batch.get("sinkNodeId") or ""),
    )
    sink = _require_session_participant(record, sink_node_id)
    clean_phase = str(phase or source_batch.get("phase") or "").strip()
    if not clean_phase:
        raise ValueError("CAI-owned transport output batch phase is required.")
    try:
        clean_sequence = (
            int(sequence)
            if sequence is not None
            else int(source_batch.get("sequence") or 0)
        )
    except (TypeError, ValueError):
        clean_sequence = 0
    if output_payload is None:
        payload = read_cai_owned_transport_batch_output_payload(
            clean_session_id,
            clean_source_batch_id,
            policy,
        )
        resolved_output_payload_sha256_hex = str(
            output_payload_sha256_hex
            or source_batch.get("outputPayloadSha256Hex")
            or ""
        ).strip()
    else:
        payload = bytes(output_payload or b"")
        resolved_output_payload_sha256_hex = (
            str(output_payload_sha256_hex or "").strip().lower()
            or hashlib.sha256(payload).hexdigest()
        )
    output_metadata = dict(metadata or {})
    output_metadata.update(
        {
            "payloadRole": "shard_output",
            "previousBatchId": clean_source_batch_id,
            "inputPayloadSha256Hex": source_batch.get("payloadSha256Hex"),
            "outputPayloadSha256Hex": resolved_output_payload_sha256_hex,
            "hashChainSha256Hex": source_batch.get("hashChainSha256Hex"),
        }
    )
    return build_cai_owned_transport_batch_envelope(
        session_id=clean_session_id,
        phase=clean_phase,
        source_node_id=source,
        sink_node_id=sink,
        sequence=clean_sequence,
        payload=payload,
        chain_id=record.chain_id,
        metadata=output_metadata,
    )


def validate_cai_owned_transport_batch_envelope(
    envelope: dict[str, Any] | None,
    *,
    session_id: str | None = None,
    participant_node_ids: Sequence[str] | None = None,
    chain_id: str | None = None,
    max_age_seconds: float | None = CAI_OWNED_TRANSPORT_BATCH_ENVELOPE_MAX_AGE_SECONDS,
    now: datetime | None = None,
    require_signature: bool | None = None,
    trusted_signer_identities_by_node: Mapping[str, Any] | None = None,
    require_trusted_signer: bool = False,
    record_replay_cache: bool = False,
    replay_cache_policy: WalletPolicy | None = None,
    replay_cache_retention_seconds: float | int | None = None,
) -> tuple[bool, str | None]:
    if not isinstance(envelope, dict):
        return False, "CAI-owned transport batch envelope is missing."
    if (
        int(envelope.get("schemaVersion") or 0)
        != CAI_OWNED_TRANSPORT_BATCH_ENVELOPE_SCHEMA_VERSION
    ):
        return False, "CAI-owned transport batch envelope schema is unsupported."
    if str(envelope.get("protocol") or "").strip() != CAI_OWNED_TRANSPORT_PROTOCOL:
        return False, "CAI-owned transport batch envelope protocol is invalid."
    if (
        int(envelope.get("protocolVersion") or 0)
        != CAI_OWNED_TRANSPORT_PROTOCOL_VERSION
    ):
        return (
            False,
            "CAI-owned transport batch envelope protocol version is unsupported.",
        )
    chain_valid, chain_error, _envelope_chain_id = (
        _validate_cai_owned_transport_chain_id(
            envelope,
            expected_chain_id=_cai_owned_transport_chain_id(None, chain_id),
            payload_name="batch envelope",
        )
    )
    if not chain_valid:
        return False, chain_error
    created_valid, created_error = _validate_cai_owned_transport_created_at(
        envelope,
        payload_name="batch envelope",
        max_age_seconds=max_age_seconds,
        now=now,
    )
    if not created_valid:
        return False, created_error

    expected_session_id = str(session_id or "").strip()
    envelope_session_id = str(envelope.get("sessionId") or "").strip()
    if not envelope_session_id:
        return False, "CAI-owned transport batch envelope session id is missing."
    if expected_session_id and envelope_session_id != expected_session_id:
        return False, "CAI-owned transport batch envelope session id does not match."
    batch_id = str(envelope.get("batchId") or "").strip()
    if not _is_safe_transport_file_id(batch_id, prefix="caibatch_"):
        return False, "CAI-owned transport batch envelope batch id is invalid."

    phase = str(envelope.get("phase") or "").strip()
    if phase not in CAI_OWNED_TRANSPORT_BATCH_PHASES:
        return False, "CAI-owned transport batch envelope phase is unsupported."
    source = str(envelope.get("sourceNodeId") or "").strip()
    sink = str(envelope.get("sinkNodeId") or "").strip()
    if not source or not sink:
        return False, "CAI-owned transport batch envelope source or sink is missing."

    participants = set(_clean_node_ids(participant_node_ids or []))
    if participants and (source not in participants or sink not in participants):
        return False, "CAI-owned transport batch envelope participant is not allowed."
    signature_valid, signature_error = validate_cai_owned_transport_payload_signature(
        envelope,
        payload_name="CAI-owned transport batch envelope",
        expected_signer_node_id=source,
        allowed_signer_node_ids=participant_node_ids or [],
        require_signature=require_signature,
        trusted_signer_identities_by_node=trusted_signer_identities_by_node,
        require_trusted_signer=require_trusted_signer,
        record_replay_cache=record_replay_cache,
        replay_cache_policy=replay_cache_policy,
        replay_cache_retention_seconds=replay_cache_retention_seconds,
    )
    if not signature_valid:
        return False, signature_error

    try:
        if int(envelope.get("sequence") or 0) < 0:
            return False, "CAI-owned transport batch envelope sequence is invalid."
    except (TypeError, ValueError):
        return False, "CAI-owned transport batch envelope sequence is invalid."
    if str(envelope.get("payloadEncoding") or "").strip() != "base64":
        return (
            False,
            "CAI-owned transport batch envelope payload encoding is unsupported.",
        )

    try:
        encoded_payload_bytes = _encoded_cai_owned_transport_payload_bytes_from_envelope(
            envelope,
        )
    except ValueError as exc:
        return False, str(exc)
    compression = str(envelope.get("payloadCompression") or "").strip().lower()
    if compression and compression not in CAI_OWNED_TRANSPORT_PAYLOAD_COMPRESSIONS:
        return (
            False,
            "CAI-owned transport batch envelope payload compression is unsupported.",
        )
    try:
        payload_bytes = _decode_cai_owned_transport_payload_bytes(
            encoded_payload_bytes,
            compression=compression or None,
        )
    except ValueError as exc:
        return False, str(exc)
    if envelope.get("payloadEncodedSizeBytes") is not None:
        try:
            declared_encoded_size = int(envelope.get("payloadEncodedSizeBytes") or 0)
        except (TypeError, ValueError):
            return (
                False,
                "CAI-owned transport batch envelope encoded payload size is invalid.",
            )
        if declared_encoded_size != len(encoded_payload_bytes):
            return (
                False,
                "CAI-owned transport batch envelope encoded payload size does not match.",
            )

    try:
        declared_size = int(envelope.get("payloadSizeBytes") or 0)
    except (TypeError, ValueError):
        return False, "CAI-owned transport batch envelope payload size is invalid."
    if declared_size != len(payload_bytes):
        return False, "CAI-owned transport batch envelope payload size does not match."
    declared_hash = str(envelope.get("payloadSha256Hex") or "").strip().lower()
    if declared_hash != hashlib.sha256(payload_bytes).hexdigest():
        return False, "CAI-owned transport batch envelope payload hash does not match."
    metadata = envelope.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        return False, "CAI-owned transport batch envelope metadata is invalid."
    if isinstance(metadata, dict) and metadata.get("frameSchemaVersion") is not None:
        frame_valid, frame_error = validate_cai_owned_transport_frame_metadata(
            metadata,
        )
        if not frame_valid:
            return False, frame_error
        frame_payload_hash = metadata.get("payloadSha256Hex")
        if frame_payload_hash is not None and (
            _normalize_sha256_hex(
                frame_payload_hash,
                field_name="payloadSha256Hex",
            )
            != declared_hash
        ):
            return False, "CAI-owned transport frame payload hash does not match."
    route_audit = envelope.get("routeAudit")
    if route_audit is not None and not isinstance(route_audit, dict):
        return False, "CAI-owned transport batch envelope route audit is invalid."
    expected_batch_id = _cai_owned_transport_batch_id(
        session_id=envelope_session_id,
        phase=phase,
        source_node_id=source,
        sink_node_id=sink,
        sequence=int(envelope.get("sequence") or 0),
        payload_sha256_hex=declared_hash,
    )
    if batch_id != expected_batch_id:
        return False, "CAI-owned transport batch envelope batch id does not match."
    return True, None


def cai_owned_transport_batch_payload_bytes(
    envelope: dict[str, Any],
) -> bytes:
    valid, error = validate_cai_owned_transport_batch_envelope(envelope)
    if not valid:
        raise ValueError(error or "CAI-owned transport batch envelope is invalid.")
    return _decode_cai_owned_transport_batch_payload(envelope)


def read_cai_owned_transport_batch_payload(
    session_id: str,
    batch_id: str,
    policy: WalletPolicy | None = None,
) -> bytes:
    path = verified_cai_owned_transport_batch_payload_path(session_id, batch_id, policy)
    return path.read_bytes()


def verified_cai_owned_transport_batch_payload_path(
    session_id: str,
    batch_id: str,
    policy: WalletPolicy | None = None,
) -> Path:
    clean_session_id = str(session_id or "").strip()
    clean_batch_id = _require_safe_transport_file_id(batch_id, prefix="caibatch_")
    batch_record: dict[str, Any] | None = None
    for record in list_cai_owned_transport_sessions(policy):
        if record.session_id != clean_session_id:
            continue
        for batch in record.batch_records:
            if not isinstance(batch, dict):
                continue
            if str(batch.get("batchId") or "").strip() == clean_batch_id:
                batch_record = batch
                break
        break
    if batch_record is None:
        raise FileNotFoundError(
            f"CAI-owned transport batch not found: {clean_batch_id}"
        )
    path = cai_owned_transport_batch_payload_path(
        clean_session_id,
        clean_batch_id,
        policy,
    )
    if not path.exists():
        raise FileNotFoundError(
            f"CAI-owned transport payload not found: {clean_batch_id}"
        )
    expected_hash = str(batch_record.get("payloadSha256Hex") or "").strip().lower()
    if expected_hash:
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError("CAI-owned transport payload hash does not match.")
    return path


def store_cai_owned_transport_batch_output_payload(
    session_id: str,
    batch_id: str,
    payload: bytes,
    policy: WalletPolicy | None = None,
) -> dict[str, Any]:
    clean_session_id = str(session_id or "").strip()
    clean_batch_id = _require_safe_transport_file_id(batch_id, prefix="caibatch_")
    batch_record = _cai_owned_transport_batch_record(
        clean_session_id,
        clean_batch_id,
        policy,
    )
    if batch_record is None:
        raise FileNotFoundError(
            f"CAI-owned transport batch not found: {clean_batch_id}"
        )
    payload_bytes = bytes(payload or b"")
    payload_hash = hashlib.sha256(payload_bytes).hexdigest()
    path = cai_owned_transport_batch_output_payload_path(
        clean_session_id,
        clean_batch_id,
        policy,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload_bytes)
    return {
        "outputPayloadStorage": "local_file",
        "outputPayloadStorageKey": f"{clean_session_id}/{clean_batch_id}.out.bin",
        "outputPayloadSizeBytes": len(payload_bytes),
        "outputPayloadSha256Hex": payload_hash,
    }


def read_cai_owned_transport_batch_output_payload(
    session_id: str,
    batch_id: str,
    policy: WalletPolicy | None = None,
) -> bytes:
    path = verified_cai_owned_transport_batch_output_payload_path(
        session_id,
        batch_id,
        policy,
    )
    return path.read_bytes()


def latest_cai_owned_transport_final_output(
    session_id: str,
    *,
    requester_node_id: str | None = None,
    policy: WalletPolicy | None = None,
) -> dict[str, Any] | None:
    clean_session_id = str(session_id or "").strip()
    record = _cai_owned_transport_session_record(clean_session_id, policy)
    if record is None:
        raise ValueError(f"CAI-owned transport session '{clean_session_id}' not found.")
    requester = str(requester_node_id or "").strip()
    for batch in reversed(record.batch_records):
        if not isinstance(batch, dict):
            continue
        if not _is_cai_owned_transport_final_output_batch(record, batch):
            continue
        sink = str(batch.get("sinkNodeId") or "").strip()
        if requester and sink != requester:
            continue
        batch_id = _require_safe_transport_file_id(
            batch.get("batchId"),
            prefix="caibatch_",
        )
        payload = read_cai_owned_transport_batch_payload(
            record.session_id,
            batch_id,
            policy,
        )
        return {
            "sessionId": record.session_id,
            "batchId": batch_id,
            "sourceNodeId": str(batch.get("sourceNodeId") or "").strip(),
            "sinkNodeId": sink,
            "payload": payload,
            "payloadBase64": base64.b64encode(payload).decode("ascii"),
            "payloadSizeBytes": len(payload),
            "payloadSha256Hex": hashlib.sha256(payload).hexdigest(),
            "batch": dict(batch),
        }
    return None


def wait_for_cai_owned_transport_final_output(
    session_id: str,
    *,
    requester_node_id: str | None = None,
    timeout_sec: float | int = 30.0,
    poll_interval_sec: float | int = 0.25,
    policy: WalletPolicy | None = None,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.0, float(timeout_sec or 0.0))
    interval = max(0.01, float(poll_interval_sec or 0.01))
    while True:
        output = latest_cai_owned_transport_final_output(
            session_id,
            requester_node_id=requester_node_id,
            policy=policy,
        )
        if output is not None:
            return output
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"CAI-owned transport final output was not delivered for "
                f"session '{session_id}'."
            )
        time.sleep(interval)


def await_cai_owned_transport_session_final_result(
    session_id: str,
    *,
    requester_node_id: str | None = None,
    timeout_sec: float | int = 30.0,
    poll_interval_sec: float | int = 0.25,
    policy: WalletPolicy | None = None,
) -> dict[str, Any]:
    clean_session_id = str(session_id or "").strip()
    timeout = max(0.0, float(timeout_sec or 0.0))
    interval = max(0.01, float(poll_interval_sec or 0.01))
    deadline = time.monotonic() + timeout
    try:
        final_output = wait_for_cai_owned_transport_final_output(
            clean_session_id,
            requester_node_id=requester_node_id,
            timeout_sec=timeout,
            poll_interval_sec=interval,
            policy=policy,
        )
    except TimeoutError as exc:
        failed = fail_cai_owned_transport_session(
            clean_session_id,
            error=str(exc),
            policy=policy,
        )
        return {
            "status": "failed",
            "sessionId": clean_session_id,
            "error": str(exc),
            "finalOutput": None,
            "session": cai_owned_transport_session_to_dict(failed),
            "proofVerified": False,
        }

    while True:
        try:
            completed = complete_cai_owned_transport_session(
                clean_session_id,
                policy=policy,
            )
            break
        except ValueError as exc:
            if time.monotonic() >= deadline:
                failed = _cai_owned_transport_session_record(clean_session_id, policy)
                return {
                    "status": "failed",
                    "sessionId": clean_session_id,
                    "error": str(exc),
                    "finalOutput": final_output,
                    "session": (
                        cai_owned_transport_session_to_dict(failed)
                        if failed is not None
                        else None
                    ),
                    "proofVerified": False,
                }
            time.sleep(interval)

    execution_audit = (completed.proof or {}).get("executionAudit", {})
    return {
        "status": completed.status,
        "sessionId": clean_session_id,
        "finalOutput": final_output,
        "session": cai_owned_transport_session_to_dict(completed),
        "proof": completed.proof,
        "proofVerified": bool(
            isinstance(execution_audit, dict)
            and execution_audit.get("verified")
        ),
    }


def verified_cai_owned_transport_batch_output_payload_path(
    session_id: str,
    batch_id: str,
    policy: WalletPolicy | None = None,
) -> Path:
    clean_session_id = str(session_id or "").strip()
    clean_batch_id = _require_safe_transport_file_id(batch_id, prefix="caibatch_")
    batch_record = _cai_owned_transport_batch_record(
        clean_session_id,
        clean_batch_id,
        policy,
    )
    if batch_record is None:
        raise FileNotFoundError(
            f"CAI-owned transport batch not found: {clean_batch_id}"
        )
    path = cai_owned_transport_batch_output_payload_path(
        clean_session_id,
        clean_batch_id,
        policy,
    )
    if not path.exists():
        raise FileNotFoundError(
            f"CAI-owned transport output payload not found: {clean_batch_id}"
        )
    expected_hash = (
        str(batch_record.get("outputPayloadSha256Hex") or "").strip().lower()
    )
    if expected_hash:
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError("CAI-owned transport output payload hash does not match.")
    return path


def complete_cai_owned_transport_session(
    session_id: str,
    *,
    activation_batch_count: int = 0,
    decode_batch_count: int = 0,
    shard_receipts: Sequence[dict[str, Any]] | None = None,
    proof: dict[str, Any] | None = None,
    policy: WalletPolicy | None = None,
) -> CaiOwnedTransportSessionRecord:
    clean_session_id = str(session_id or "").strip()
    records = list_cai_owned_transport_sessions(policy)
    for index, record in enumerate(records):
        if record.session_id != clean_session_id:
            continue
        if proof is None and shard_receipts is None:
            shard_receipts = (
                list(record.shard_receipts)
                if record.shard_receipts
                else cai_owned_transport_shard_receipts_from_processed_batches(record)
            )
        if proof is None and shard_receipts is not None:
            activation_batch_count = activation_batch_count or _max_receipt_count(
                shard_receipts,
                "activationBatchCount",
            )
            decode_batch_count = decode_batch_count or _max_receipt_count(
                shard_receipts,
                "decodeBatchCount",
            )
        resolved_proof = proof or build_cai_owned_transport_execution_proof(
            session_id=record.session_id,
            instance_id=record.instance_id,
            participant_node_ids=record.participant_node_ids,
            executor_node_ids=record.executor_node_ids or record.participant_node_ids,
            chain_id=record.chain_id,
            model_id=record.model_id,
            task_id=record.task_id,
            activation_batch_count=activation_batch_count,
            decode_batch_count=decode_batch_count,
            shard_receipts=shard_receipts,
        )
        audit_valid, audit_error, execution_audit = (
            validate_cai_owned_transport_session_execution_audit(
                record,
                proof=resolved_proof,
                policy=policy,
            )
        )
        now = datetime.now(tz=UTC).isoformat()
        if not audit_valid:
            record.status = "failed"
            record.last_error = audit_error
            record.updated_at = now
            records[index] = record
            save_cai_owned_transport_sessions(records, policy)
            raise ValueError(
                audit_error or "CAI-owned transport execution audit is invalid."
            )
        resolved_proof = dict(resolved_proof)
        resolved_proof["executionAudit"] = execution_audit
        valid, error = validate_cai_owned_transport_execution_proof(
            resolved_proof,
            participant_node_ids=record.participant_node_ids,
            executor_node_ids=record.executor_node_ids or record.participant_node_ids,
            model_id=record.model_id,
            chain_id=record.chain_id,
        )
        if not valid:
            record.status = "failed"
            record.last_error = error
            record.updated_at = now
            records[index] = record
            save_cai_owned_transport_sessions(records, policy)
            raise ValueError(error or "CAI-owned transport proof is invalid.")
        record.status = "completed"
        record.proof = resolved_proof
        record.completed_at = str(resolved_proof.get("completedAt") or now)
        record.updated_at = now
        record.last_error = None
        records[index] = record
        save_cai_owned_transport_sessions(records, policy)
        return record
    raise ValueError(f"CAI-owned transport session '{clean_session_id}' not found.")


def accept_cai_owned_transport_completion_notice(
    session_id: str,
    proof: dict[str, Any],
    *,
    policy: WalletPolicy | None = None,
) -> CaiOwnedTransportSessionRecord:
    clean_session_id = str(session_id or "").strip()
    if not isinstance(proof, dict):
        raise ValueError("CAI-owned transport completion proof must be an object.")
    if str(proof.get("sessionId") or "").strip() != clean_session_id:
        raise ValueError("CAI-owned transport completion proof session id mismatch.")
    records = list_cai_owned_transport_sessions(policy)
    for index, record in enumerate(records):
        if record.session_id != clean_session_id:
            continue
        if record.status == "completed":
            return record
        if record.status == "failed":
            raise ValueError("Cannot complete a failed CAI-owned transport session.")
        valid, error = validate_cai_owned_transport_execution_proof(
            proof,
            participant_node_ids=record.participant_node_ids,
            executor_node_ids=record.executor_node_ids or record.participant_node_ids,
            model_id=record.model_id,
            chain_id=record.chain_id,
        )
        if not valid:
            raise ValueError(error or "CAI-owned transport completion proof is invalid.")
        execution_audit = proof.get("executionAudit")
        if isinstance(execution_audit, Mapping) and execution_audit.get("verified") is False:
            raise ValueError("CAI-owned transport completion proof audit is not verified.")
        coverage_error = _validate_cai_owned_transport_completion_notice_coverage(
            record,
            proof,
        )
        if coverage_error:
            raise ValueError(coverage_error)
        now = datetime.now(tz=UTC).isoformat()
        record.status = "completed"
        record.proof = dict(proof)
        record.completed_at = str(proof.get("completedAt") or now)
        record.updated_at = now
        record.last_error = None
        records[index] = record
        save_cai_owned_transport_sessions(records, policy)
        return record
    raise ValueError(f"CAI-owned transport session '{clean_session_id}' not found.")


def _validate_cai_owned_transport_completion_notice_coverage(
    record: CaiOwnedTransportSessionRecord,
    proof: Mapping[str, Any],
) -> str | None:
    local_batch_ids: set[str] = set()
    local_output_hashes: set[str] = set()
    for batch in record.batch_records:
        if not isinstance(batch, Mapping):
            continue
        if str(batch.get("status") or "").strip() not in {"processed", "delivered"}:
            continue
        batch_id = str(batch.get("batchId") or "").strip()
        if batch_id:
            local_batch_ids.add(batch_id)
        output_hash = str(batch.get("outputPayloadSha256Hex") or "").strip().lower()
        if output_hash:
            local_output_hashes.add(output_hash)

    proof_batch_ids, proof_batch_id_errors = _cai_owned_transport_shard_receipt_batch_ids(
        proof.get("shardReceipts") if isinstance(proof.get("shardReceipts"), list) else []
    )
    if proof_batch_id_errors:
        return proof_batch_id_errors[0]
    audit = proof.get("executionAudit")
    if isinstance(audit, Mapping):
        for batch_id in _clean_cai_owned_transport_receipt_batch_ids(
            audit.get("receiptBatchIds") if isinstance(audit.get("receiptBatchIds"), list) else []
        ):
            proof_batch_ids.add(batch_id)

    missing_batch_ids = sorted(local_batch_ids - proof_batch_ids)
    if missing_batch_ids:
        return (
            "CAI-owned transport completion proof does not cover local processed "
            "batch ids: "
            + ", ".join(missing_batch_ids[:5])
        )

    proof_output_hashes: set[str] = set()
    for receipt in proof.get("shardReceipts") or []:
        if not isinstance(receipt, Mapping):
            continue
        for output_hash in _clean_cai_owned_transport_receipt_hashes(
            receipt.get("outputPayloadSha256Hexes")
            if isinstance(receipt.get("outputPayloadSha256Hexes"), list)
            else [],
            field_name="outputPayloadSha256Hexes",
        ):
            proof_output_hashes.add(output_hash)
    missing_output_hashes = sorted(local_output_hashes - proof_output_hashes)
    if missing_output_hashes:
        return (
            "CAI-owned transport completion proof does not cover local output "
            "payload hashes: "
            + ", ".join(missing_output_hashes[:5])
        )
    return None


def fail_cai_owned_transport_session(
    session_id: str,
    *,
    error: str | None = None,
    policy: WalletPolicy | None = None,
) -> CaiOwnedTransportSessionRecord:
    clean_session_id = str(session_id or "").strip()
    records = list_cai_owned_transport_sessions(policy)
    for index, record in enumerate(records):
        if record.session_id != clean_session_id:
            continue
        if record.status == "completed":
            raise ValueError("Cannot fail a completed CAI-owned transport session.")
        now = datetime.now(tz=UTC).isoformat()
        record.status = "failed"
        record.last_error = str(error or "").strip() or None
        record.updated_at = now
        records[index] = record
        save_cai_owned_transport_sessions(records, policy)
        return record
    raise ValueError(f"CAI-owned transport session '{clean_session_id}' not found.")


def reconcile_cai_owned_transport_session_timeouts(
    session_id: str,
    *,
    received_timeout_sec: float | int | None = None,
    max_attempts: int | None = None,
    now: datetime | None = None,
    policy: WalletPolicy | None = None,
) -> dict[str, Any]:
    clean_session_id = str(session_id or "").strip()
    now_dt = (now or datetime.now(tz=UTC)).astimezone(UTC)
    timeout_seconds = _coerce_cai_owned_transport_batch_claim_timeout_seconds(
        received_timeout_sec,
    )
    resolved_max_attempts = _coerce_cai_owned_transport_max_attempts(max_attempts)
    records = list_cai_owned_transport_sessions(policy)
    for record_index, record in enumerate(records):
        if record.session_id != clean_session_id:
            continue
        if record.status == "completed":
            return {
                "status": "completed",
                "sessionId": clean_session_id,
                "timedOutBatchIds": [],
                "retryScheduledBatchIds": [],
                "session": cai_owned_transport_session_to_dict(record),
            }

        timed_out_batch_ids: list[str] = []
        retry_scheduled_batch_ids: list[str] = []
        first_error: str | None = None
        updated_batches: list[dict[str, Any]] = []
        for batch in record.batch_records:
            if not isinstance(batch, dict):
                continue
            updated_batch = dict(batch)
            batch_id = str(updated_batch.get("batchId") or "").strip()
            status = str(updated_batch.get("status") or "recorded").strip()
            if status == "received" and _cai_owned_transport_batch_claim_expired(
                updated_batch,
                now_dt,
                timeout_seconds,
            ):
                error = (
                    "CAI-owned transport batch was not claimed before coordinator "
                    "timeout."
                )
                _mark_cai_owned_transport_batch_timed_out(
                    updated_batch,
                    now_dt,
                    error=error,
                    reason="claim_timeout",
                )
                if batch_id:
                    timed_out_batch_ids.append(batch_id)
                first_error = first_error or error
            elif status == "processing" and _cai_owned_transport_batch_lease_expired(
                updated_batch,
                now_dt,
            ):
                attempt_count = _cai_owned_transport_batch_attempt_count(updated_batch)
                if attempt_count < resolved_max_attempts:
                    updated_batch["status"] = "received"
                    updated_batch["updatedAt"] = now_dt.isoformat()
                    updated_batch["retryScheduledAt"] = now_dt.isoformat()
                    updated_batch["lastError"] = (
                        "CAI-owned transport batch lease expired; retry scheduled."
                    )
                    updated_batch["retryable"] = True
                    updated_batch["maxAttempts"] = resolved_max_attempts
                    _clear_cai_owned_transport_batch_runtime_claim(updated_batch)
                    if batch_id:
                        retry_scheduled_batch_ids.append(batch_id)
                else:
                    error = (
                        "CAI-owned transport batch lease expired and max attempts "
                        "were exhausted."
                    )
                    updated_batch["maxAttempts"] = resolved_max_attempts
                    _mark_cai_owned_transport_batch_timed_out(
                        updated_batch,
                        now_dt,
                        error=error,
                        reason="lease_timeout",
                    )
                    if batch_id:
                        timed_out_batch_ids.append(batch_id)
                    first_error = first_error or error
            updated_batches.append(updated_batch)

        record.batch_records = updated_batches
        if timed_out_batch_ids:
            record.status = "failed"
            record.last_error = first_error
        elif retry_scheduled_batch_ids and record.status == "created":
            record.status = "running"
        if timed_out_batch_ids or retry_scheduled_batch_ids:
            record.updated_at = now_dt.isoformat()
            records[record_index] = record
            save_cai_owned_transport_sessions(records, policy)
        return {
            "status": record.status,
            "sessionId": clean_session_id,
            "timedOutBatchIds": timed_out_batch_ids,
            "retryScheduledBatchIds": retry_scheduled_batch_ids,
            "session": cai_owned_transport_session_to_dict(record),
        }
    raise ValueError(f"CAI-owned transport session '{clean_session_id}' not found.")


def validate_cai_owned_transport_session_execution_audit(
    record: CaiOwnedTransportSessionRecord,
    *,
    proof: dict[str, Any] | None = None,
    policy: WalletPolicy | None = None,
) -> tuple[bool, str | None, dict[str, Any]]:
    if not isinstance(record, CaiOwnedTransportSessionRecord):
        return (
            False,
            "CAI-owned transport session record is missing.",
            {"verified": False, "errors": ["session record is missing"]},
        )
    errors: list[str] = []
    batch_records = [
        dict(item)
        for item in record.batch_records
        if isinstance(item, dict)
    ]
    known_batch_ids = {
        str(item.get("batchId") or "").strip()
        for item in batch_records
        if str(item.get("batchId") or "").strip()
    }
    receipt_known_batch_ids, receipt_known_errors = (
        _cai_owned_transport_shard_receipt_batch_ids(record.shard_receipts)
    )
    errors.extend(receipt_known_errors)
    known_batch_ids.update(receipt_known_batch_ids)
    processed_batch_ids: list[str] = []
    final_output_batch_ids: list[str] = []
    hash_chain_sha256_hexes: list[str] = []
    blocked_batch_ids: list[str] = []

    for batch in batch_records:
        batch_id = str(batch.get("batchId") or "").strip()
        status = str(batch.get("status") or "recorded").strip()
        if not batch_id:
            errors.append("CAI-owned transport batch id is missing.")
            continue
        if status in {"failed", "timed_out", "hash_mismatch"}:
            blocked_batch_ids.append(batch_id)
            errors.append(
                f"CAI-owned transport batch '{batch_id}' is {status}."
            )
            continue
        if _is_cai_owned_transport_final_output_batch(record, batch):
            final_output_batch_ids.append(batch_id)
            continue
        if status != "processed":
            errors.append(
                f"CAI-owned transport batch '{batch_id}' is not processed."
            )
            continue
        processed_batch_ids.append(batch_id)
        batch_valid, batch_error, chain_hash = (
            _validate_cai_owned_transport_processed_batch_execution_audit(
                record.session_id,
                batch,
                known_batch_ids=known_batch_ids,
                policy=policy,
            )
        )
        if not batch_valid and batch_error:
            errors.append(batch_error)
        if chain_hash:
            hash_chain_sha256_hexes.append(chain_hash)

    dag_audit, dag_errors = _validate_cai_owned_transport_execution_dag_coverage(
        record,
        batch_records,
        final_output_batch_ids=final_output_batch_ids,
    )
    errors.extend(dag_errors)

    resolved_proof = proof
    if resolved_proof is None:
        shard_receipts = cai_owned_transport_shard_receipts_from_processed_batches(
            record
        )
        resolved_proof = build_cai_owned_transport_execution_proof(
            session_id=record.session_id,
            instance_id=record.instance_id,
            participant_node_ids=record.participant_node_ids,
            executor_node_ids=record.executor_node_ids or record.participant_node_ids,
            chain_id=record.chain_id,
            model_id=record.model_id,
            task_id=record.task_id,
            activation_batch_count=_max_receipt_count(
                shard_receipts,
                "activationBatchCount",
            ),
            decode_batch_count=_max_receipt_count(
                shard_receipts,
                "decodeBatchCount",
            ),
            shard_receipts=shard_receipts,
        )
    receipt_batch_ids, receipt_errors = _cai_owned_transport_proof_batch_ids(
        resolved_proof
    )
    errors.extend(receipt_errors)
    missing_from_proof = [
        batch_id
        for batch_id in processed_batch_ids
        if batch_id not in receipt_batch_ids
    ]
    if processed_batch_ids and not receipt_batch_ids:
        errors.append("CAI-owned transport proof does not cover processed batches.")
    elif missing_from_proof:
        errors.append(
            "CAI-owned transport proof is missing processed batch ids: "
            + ", ".join(missing_from_proof)
        )
    if not batch_records:
        errors.append("CAI-owned transport session has no batch records.")
    if not processed_batch_ids and not receipt_batch_ids:
        errors.append("CAI-owned transport session has no processed batches.")

    audit = {
        "verified": not errors,
        "sessionId": record.session_id,
        "processedBatchIds": processed_batch_ids,
        "finalOutputBatchIds": final_output_batch_ids,
        "blockedBatchIds": blocked_batch_ids,
        "receiptBatchIds": sorted(receipt_batch_ids),
        "hashChainSha256Hexes": hash_chain_sha256_hexes,
        "executionDag": dag_audit,
        "batchRecordCount": len(batch_records),
        "processedBatchCount": len(processed_batch_ids),
        "finalOutputBatchCount": len(final_output_batch_ids),
        "errorCount": len(errors),
        "errors": errors,
        "verifiedAt": datetime.now(tz=UTC).isoformat(),
    }
    if errors:
        return False, errors[0], audit
    return True, None, audit


def record_cai_owned_transport_batch(
    session_id: str,
    *,
    phase: str,
    source_node_id: str,
    sink_node_id: str,
    batch_id: str | None = None,
    payload_size_bytes: int = 0,
    payload_sha256_hex: str | None = None,
    metadata: dict[str, Any] | None = None,
    route_audit: dict[str, Any] | None = None,
    status: str = "recorded",
    policy: WalletPolicy | None = None,
) -> CaiOwnedTransportSessionRecord:
    clean_session_id = str(session_id or "").strip()
    records = list_cai_owned_transport_sessions(policy)
    for index, record in enumerate(records):
        if record.session_id != clean_session_id:
            continue
        if record.status == "completed":
            raise ValueError("Cannot append batch records to a completed session.")
        source = _require_session_participant(record, source_node_id)
        sink = _require_session_participant(record, sink_node_id)
        clean_phase = str(phase or "").strip()
        if not clean_phase:
            raise ValueError("CAI-owned transport batch phase is required.")
        clean_batch_id = str(batch_id or "").strip()
        if clean_batch_id:
            clean_batch_id = _require_safe_transport_file_id(
                clean_batch_id,
                prefix="caibatch_",
            )
        else:
            clean_batch_id = f"caibatch_{secrets.token_hex(10)}"
        for existing_batch in record.batch_records:
            if not isinstance(existing_batch, dict):
                continue
            if str(existing_batch.get("batchId") or "").strip() != clean_batch_id:
                continue
            replay_valid, replay_error = _validate_cai_owned_transport_batch_replay(
                existing_batch,
                phase=clean_phase,
                source_node_id=source,
                sink_node_id=sink,
                payload_sha256_hex=payload_sha256_hex,
            )
            if not replay_valid:
                raise ValueError(replay_error)
            return record
        now = datetime.now(tz=UTC).isoformat()
        clean_status = str(status or "").strip() or "recorded"
        batch_record = {
            "batchId": clean_batch_id,
            "chainId": record.chain_id,
            "status": clean_status,
            "phase": clean_phase,
            "sourceNodeId": source,
            "sinkNodeId": sink,
            "payloadSizeBytes": max(0, int(payload_size_bytes or 0)),
            "payloadSha256Hex": str(payload_sha256_hex or "").strip() or None,
            "metadata": dict(metadata or {}),
            "createdAt": now,
        }
        if route_audit is not None:
            batch_record["routeAudit"] = _jsonable_dict(
                route_audit,
                field_name="routeAudit",
            )
        record.batch_records.append(batch_record)
        if record.status == "created":
            record.status = "running"
        record.updated_at = now
        records[index] = record
        save_cai_owned_transport_sessions(records, policy)
        return record
    raise ValueError(f"CAI-owned transport session '{clean_session_id}' not found.")


def record_cai_owned_transport_batch_envelope(
    session_id: str,
    envelope: dict[str, Any],
    *,
    local_node_id: str | None = None,
    record_replay_cache: bool = False,
    replay_cache_retention_seconds: float | int | None = None,
    policy: WalletPolicy | None = None,
) -> CaiOwnedTransportSessionRecord:
    clean_session_id = str(session_id or "").strip()
    records = list_cai_owned_transport_sessions(policy)
    for record in records:
        if record.session_id != clean_session_id:
            continue
        valid, error = validate_cai_owned_transport_batch_envelope(
            envelope,
            session_id=clean_session_id,
            participant_node_ids=record.participant_node_ids,
            chain_id=record.chain_id,
            record_replay_cache=record_replay_cache,
            replay_cache_policy=policy,
            replay_cache_retention_seconds=replay_cache_retention_seconds,
        )
        if not valid:
            raise ValueError(error or "CAI-owned transport batch envelope is invalid.")
        local = str(local_node_id or "").strip()
        sink = str(envelope.get("sinkNodeId") or "").strip()
        if local and sink != local:
            raise ValueError(
                "CAI-owned transport batch envelope is not addressed to local node."
            )
        batch_id = _require_safe_transport_file_id(
            envelope.get("batchId"),
            prefix="caibatch_",
        )
        payload_bytes = _decode_cai_owned_transport_batch_payload(envelope)
        payload_path = cai_owned_transport_batch_payload_path(
            clean_session_id,
            batch_id,
            policy,
        )
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_bytes(payload_bytes)
        metadata = dict(envelope.get("metadata") or {})
        metadata.update(
            {
                "transportBatchId": batch_id,
                "chainId": envelope.get("chainId"),
                "sequence": envelope.get("sequence"),
                "payloadEncoding": envelope.get("payloadEncoding"),
                "payloadCompression": envelope.get("payloadCompression"),
                "payloadEncodedSizeBytes": envelope.get("payloadEncodedSizeBytes"),
                "payloadChunkCount": envelope.get("payloadChunkCount"),
                "payloadChunkSizeBytes": envelope.get("payloadChunkSizeBytes"),
                "envelopeSchemaVersion": envelope.get("schemaVersion"),
                "payloadStorage": "local_file",
                "payloadStorageKey": f"{clean_session_id}/{batch_id}.bin",
            }
        )
        route_audit = envelope.get("routeAudit")
        if route_audit is None and isinstance(metadata.get("routeAudit"), dict):
            route_audit = metadata.get("routeAudit")
        status = "received"
        preview_batch = {
            "batchId": batch_id,
            "status": status,
            "sourceNodeId": str(envelope.get("sourceNodeId") or ""),
            "sinkNodeId": sink,
            "metadata": metadata,
        }
        if _is_cai_owned_transport_final_output_batch(record, preview_batch):
            status = "delivered"
            metadata["finalOutput"] = True
            metadata["deliveredToNodeId"] = sink
        updated_record = record_cai_owned_transport_batch(
            clean_session_id,
            phase=str(envelope.get("phase") or ""),
            source_node_id=str(envelope.get("sourceNodeId") or ""),
            sink_node_id=sink,
            batch_id=batch_id,
            payload_size_bytes=int(envelope.get("payloadSizeBytes") or 0),
            payload_sha256_hex=str(envelope.get("payloadSha256Hex") or ""),
            metadata=metadata,
            route_audit=route_audit if isinstance(route_audit, dict) else None,
            status=status,
            policy=policy,
        )
        if status == "delivered":
            return _record_cai_owned_transport_embedded_shard_receipts(
                updated_record,
                metadata,
                policy=policy,
            )
        return updated_record
    raise ValueError(f"CAI-owned transport session '{clean_session_id}' not found.")


def _record_cai_owned_transport_embedded_shard_receipts(
    record: CaiOwnedTransportSessionRecord,
    metadata: Mapping[str, Any],
    *,
    policy: WalletPolicy | None = None,
) -> CaiOwnedTransportSessionRecord:
    updated_record = record
    for receipt in _cai_owned_transport_embedded_shard_receipts(metadata):
        updated_record = record_cai_owned_transport_shard_receipt(
            record.session_id,
            node_id=str(receipt.get("nodeId") or receipt.get("node_id") or ""),
            chain_id=receipt.get("chainId")
            or receipt.get("chain_id")
            or receipt.get("network"),
            status=str(receipt.get("status") or "completed"),
            activation_batch_count=int(receipt.get("activationBatchCount") or 0),
            decode_batch_count=int(receipt.get("decodeBatchCount") or 0),
            layer_start=_optional_int(receipt.get("layerStart")),
            layer_end=_optional_int(receipt.get("layerEnd")),
            metrics=receipt.get("metrics")
            if isinstance(receipt.get("metrics"), dict)
            else None,
            batch_ids=receipt.get("batchIds") or receipt.get("batch_ids"),
            stage_ids=receipt.get("stageIds") or receipt.get("stage_ids"),
            sequences=receipt.get("sequences"),
            input_payload_sha256_hexes=receipt.get("inputPayloadSha256Hexes")
            or receipt.get("input_payload_sha256_hexes"),
            output_payload_sha256_hexes=receipt.get("outputPayloadSha256Hexes")
            or receipt.get("output_payload_sha256_hexes"),
            hash_chain_sha256_hexes=receipt.get("hashChainSha256Hexes")
            or receipt.get("hash_chain_sha256_hexes"),
            route_audits=receipt.get("routeAudits") or receipt.get("route_audits"),
            runtime_audits=receipt.get("runtimeAudits")
            or receipt.get("runtime_audits"),
            signature=receipt.get("signature")
            if isinstance(receipt.get("signature"), Mapping)
            else None,
            signer_node_id=receipt.get("signerNodeId")
            or receipt.get("signer_node_id"),
            recorded_at=receipt.get("recordedAt") or receipt.get("recorded_at"),
            policy=policy,
        )
    return updated_record


def _cai_owned_transport_embedded_shard_receipts(
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(metadata, Mapping):
        return []
    raw_receipts = None
    for field_name in (
        "upstreamShardReceipts",
        "caiOwnedShardReceipts",
        "shardReceipts",
    ):
        value = metadata.get(field_name)
        if value is not None:
            raw_receipts = value
            break
    if raw_receipts is None:
        return []
    if isinstance(raw_receipts, Mapping):
        return [dict(raw_receipts)]
    if isinstance(raw_receipts, (str, bytes)) or not isinstance(
        raw_receipts,
        Sequence,
    ):
        raise ValueError("CAI-owned transport embedded shard receipts are invalid.")
    receipts = [dict(item) for item in raw_receipts if isinstance(item, Mapping)]
    if len(receipts) != len(raw_receipts):
        raise ValueError("CAI-owned transport embedded shard receipt is invalid.")
    return receipts


def list_cai_owned_transport_batch_inbox(
    local_node_id: str,
    *,
    status: str | None = "received",
    session_id: str | None = None,
    policy: WalletPolicy | None = None,
) -> list[dict[str, Any]]:
    local = str(local_node_id or "").strip()
    if not local:
        raise ValueError("CAI-owned transport inbox requires local node id.")
    clean_status = str(status or "").strip() if status is not None else None
    clean_session_id = str(session_id or "").strip()
    inbox: list[dict[str, Any]] = []
    for record in list_cai_owned_transport_sessions(policy):
        if clean_session_id and record.session_id != clean_session_id:
            continue
        for batch in record.batch_records:
            if not isinstance(batch, dict):
                continue
            if str(batch.get("sinkNodeId") or "").strip() != local:
                continue
            batch_status = str(batch.get("status") or "recorded").strip()
            if clean_status is not None and batch_status != clean_status:
                continue
            inbox.append(
                {
                    "sessionId": record.session_id,
                    "instanceId": record.instance_id,
                    "modelId": record.model_id,
                    "taskId": record.task_id,
                    "sourceNodeId": record.source_node_id,
                    "batch": dict(batch),
                }
            )
    return inbox


def mark_cai_owned_transport_batch_status(
    session_id: str,
    batch_id: str,
    *,
    status: str,
    node_id: str | None = None,
    metrics: dict[str, Any] | None = None,
    error: str | None = None,
    input_payload_sha256_hex: str | None = None,
    output_payload_sha256_hex: str | None = None,
    output_payload_size_bytes: int | None = None,
    output_payload_storage_key: str | None = None,
    previous_batch_id: str | None = None,
    hash_chain_sha256_hex: str | None = None,
    route_audit: dict[str, Any] | None = None,
    runtime_audit: dict[str, Any] | None = None,
    policy: WalletPolicy | None = None,
) -> CaiOwnedTransportSessionRecord:
    clean_session_id = str(session_id or "").strip()
    clean_batch_id = _require_safe_transport_file_id(batch_id, prefix="caibatch_")
    clean_status = str(status or "").strip()
    if not clean_status:
        raise ValueError("CAI-owned transport batch status is required.")
    records = list_cai_owned_transport_sessions(policy)
    for record_index, record in enumerate(records):
        if record.session_id != clean_session_id:
            continue
        if record.status == "completed":
            raise ValueError("Cannot update batches on a completed session.")
        local = str(node_id or "").strip()
        if local:
            _require_session_participant(record, local)
        for batch_index, batch in enumerate(record.batch_records):
            if str(batch.get("batchId") or "").strip() != clean_batch_id:
                continue
            if local and str(batch.get("sinkNodeId") or "").strip() != local:
                raise ValueError(
                    "CAI-owned transport batch is not assigned to local node."
                )
            now = datetime.now(tz=UTC).isoformat()
            updated_batch = dict(batch)
            updated_batch["status"] = clean_status
            updated_batch["updatedAt"] = now
            if clean_status in {"processed", "failed"}:
                updated_batch["processedAt"] = now
            if metrics is not None:
                updated_batch["metrics"] = dict(metrics)
            if error is not None:
                updated_batch["error"] = str(error)
            if input_payload_sha256_hex is not None:
                updated_batch["inputPayloadSha256Hex"] = _normalize_sha256_hex(
                    input_payload_sha256_hex,
                    field_name="inputPayloadSha256Hex",
                )
            if output_payload_sha256_hex is not None:
                updated_batch["outputPayloadSha256Hex"] = _normalize_sha256_hex(
                    output_payload_sha256_hex,
                    field_name="outputPayloadSha256Hex",
                )
            if output_payload_size_bytes is not None:
                updated_batch["outputPayloadSizeBytes"] = max(
                    0,
                    int(output_payload_size_bytes or 0),
                )
            if output_payload_storage_key is not None:
                updated_batch["outputPayloadStorage"] = "local_file"
                updated_batch["outputPayloadStorageKey"] = str(
                    output_payload_storage_key
                ).strip()
            if previous_batch_id is not None:
                clean_previous_batch_id = str(previous_batch_id or "").strip()
                updated_batch["previousBatchId"] = (
                    _require_safe_transport_file_id(
                        clean_previous_batch_id,
                        prefix="caibatch_",
                    )
                    if clean_previous_batch_id
                    else None
                )
            if hash_chain_sha256_hex is not None:
                updated_batch["hashChainSha256Hex"] = _normalize_sha256_hex(
                    hash_chain_sha256_hex,
                    field_name="hashChainSha256Hex",
                )
            if route_audit is not None:
                updated_batch["routeAudit"] = _jsonable_dict(
                    route_audit,
                    field_name="routeAudit",
                )
            if runtime_audit is not None:
                updated_batch["runtimeAudit"] = _jsonable_dict(
                    runtime_audit,
                    field_name="runtimeAudit",
                )
            record.batch_records[batch_index] = updated_batch
            record.updated_at = now
            records[record_index] = record
            save_cai_owned_transport_sessions(records, policy)
            return record
        raise ValueError(
            f"CAI-owned transport batch '{clean_batch_id}' not found in session "
            f"'{clean_session_id}'."
        )
    raise ValueError(f"CAI-owned transport session '{clean_session_id}' not found.")


def claim_cai_owned_transport_batch(
    session_id: str,
    batch_id: str,
    *,
    node_id: str,
    runtime_id: str | None = None,
    runtime_auth_token: str | None = None,
    require_runtime_auth: bool | str | None = None,
    lease_seconds: float | int | None = None,
    policy: WalletPolicy | None = None,
) -> CaiOwnedTransportSessionRecord:
    _require_cai_owned_transport_local_runtime_auth(
        runtime_auth_token,
        require_runtime_auth=require_runtime_auth,
    )
    clean_session_id = str(session_id or "").strip()
    clean_batch_id = _require_safe_transport_file_id(batch_id, prefix="caibatch_")
    clean_runtime_id = str(runtime_id or "").strip() or None
    records = list_cai_owned_transport_sessions(policy)
    for record_index, record in enumerate(records):
        if record.session_id != clean_session_id:
            continue
        if record.status == "completed":
            raise ValueError("Cannot claim batches on a completed session.")
        local = _require_session_participant(record, node_id)
        for batch_index, batch in enumerate(record.batch_records):
            if str(batch.get("batchId") or "").strip() != clean_batch_id:
                continue
            if str(batch.get("sinkNodeId") or "").strip() != local:
                raise ValueError(
                    "CAI-owned transport batch is not assigned to local node."
                )
            current_status = str(batch.get("status") or "recorded").strip()
            existing_runtime_id = str(batch.get("runtimeId") or "").strip()
            now_dt = datetime.now(tz=UTC)
            stale_processing = False
            if current_status == "processing":
                stale_processing = _cai_owned_transport_batch_lease_expired(
                    batch,
                    now_dt,
                )
                if (
                    not stale_processing
                    and clean_runtime_id
                    and existing_runtime_id == clean_runtime_id
                ):
                    updated_batch = dict(batch)
                    updated_batch["updatedAt"] = now_dt.isoformat()
                    _apply_cai_owned_transport_batch_lease(
                        updated_batch,
                        now_dt,
                        lease_seconds,
                    )
                    record.batch_records[batch_index] = updated_batch
                    record.updated_at = now_dt.isoformat()
                    records[record_index] = record
                    save_cai_owned_transport_sessions(records, policy)
                    return record
                if not stale_processing:
                    raise ValueError(
                        "CAI-owned transport batch is already processing."
                    )
            if current_status in {"processed", "failed", "timed_out", "delivered"}:
                raise ValueError(
                    f"CAI-owned transport batch is already {current_status}."
                )
            now = now_dt.isoformat()
            updated_batch = dict(batch)
            updated_batch["status"] = "processing"
            updated_batch["claimedAt"] = now
            updated_batch["startedAt"] = now
            updated_batch["updatedAt"] = now
            updated_batch["claimedByNodeId"] = local
            if stale_processing:
                updated_batch["reclaimedAt"] = now
                if existing_runtime_id:
                    updated_batch["previousRuntimeId"] = existing_runtime_id
            if clean_runtime_id:
                updated_batch["runtimeId"] = clean_runtime_id
            try:
                attempt_count = int(updated_batch.get("attemptCount") or 0)
            except (TypeError, ValueError):
                attempt_count = 0
            updated_batch["attemptCount"] = attempt_count + 1
            _apply_cai_owned_transport_batch_lease(
                updated_batch,
                now_dt,
                lease_seconds,
            )
            record.batch_records[batch_index] = updated_batch
            if record.status == "created":
                record.status = "running"
            record.updated_at = now
            records[record_index] = record
            save_cai_owned_transport_sessions(records, policy)
            return record
        raise ValueError(
            f"CAI-owned transport batch '{clean_batch_id}' not found in session "
            f"'{clean_session_id}'."
        )
    raise ValueError(f"CAI-owned transport session '{clean_session_id}' not found.")


def heartbeat_cai_owned_transport_batch(
    session_id: str,
    batch_id: str,
    *,
    node_id: str,
    runtime_id: str | None = None,
    runtime_auth_token: str | None = None,
    require_runtime_auth: bool | str | None = None,
    lease_seconds: float | int | None = None,
    policy: WalletPolicy | None = None,
) -> CaiOwnedTransportSessionRecord:
    _require_cai_owned_transport_local_runtime_auth(
        runtime_auth_token,
        require_runtime_auth=require_runtime_auth,
    )
    clean_session_id = str(session_id or "").strip()
    clean_batch_id = _require_safe_transport_file_id(batch_id, prefix="caibatch_")
    clean_runtime_id = str(runtime_id or "").strip()
    records = list_cai_owned_transport_sessions(policy)
    for record_index, record in enumerate(records):
        if record.session_id != clean_session_id:
            continue
        if record.status == "completed":
            raise ValueError("Cannot heartbeat batches on a completed session.")
        local = _require_session_participant(record, node_id)
        for batch_index, batch in enumerate(record.batch_records):
            if str(batch.get("batchId") or "").strip() != clean_batch_id:
                continue
            if str(batch.get("sinkNodeId") or "").strip() != local:
                raise ValueError(
                    "CAI-owned transport batch is not assigned to local node."
                )
            if str(batch.get("status") or "").strip() != "processing":
                raise ValueError("CAI-owned transport batch is not processing.")
            existing_runtime_id = str(batch.get("runtimeId") or "").strip()
            if clean_runtime_id and existing_runtime_id != clean_runtime_id:
                raise ValueError(
                    "CAI-owned transport batch runtime id does not match."
                )
            now_dt = datetime.now(tz=UTC)
            if _cai_owned_transport_batch_lease_expired(batch, now_dt):
                raise ValueError("CAI-owned transport batch lease has expired.")
            updated_batch = dict(batch)
            updated_batch["updatedAt"] = now_dt.isoformat()
            _apply_cai_owned_transport_batch_lease(
                updated_batch,
                now_dt,
                lease_seconds,
            )
            record.batch_records[batch_index] = updated_batch
            record.updated_at = now_dt.isoformat()
            records[record_index] = record
            save_cai_owned_transport_sessions(records, policy)
            return record
        raise ValueError(
            f"CAI-owned transport batch '{clean_batch_id}' not found in session "
            f"'{clean_session_id}'."
        )
    raise ValueError(f"CAI-owned transport session '{clean_session_id}' not found.")


def claim_next_cai_owned_transport_batch(
    local_node_id: str,
    *,
    status: str | None = "received",
    session_id: str | None = None,
    runtime_id: str | None = None,
    runtime_auth_token: str | None = None,
    require_runtime_auth: bool | str | None = None,
    lease_seconds: float | int | None = None,
    policy: WalletPolicy | None = None,
) -> dict[str, Any] | None:
    _require_cai_owned_transport_local_runtime_auth(
        runtime_auth_token,
        require_runtime_auth=require_runtime_auth,
    )
    local = str(local_node_id or "").strip()
    if not local:
        raise ValueError("CAI-owned transport claim-next requires local node id.")
    clean_status = str(status or "").strip() if status is not None else None
    candidate_statuses = [clean_status]
    if clean_status == "received":
        candidate_statuses.append("processing")
    for candidate_status in candidate_statuses:
        for inbox_item in list_cai_owned_transport_batch_inbox(
            local,
            status=candidate_status,
            session_id=session_id,
            policy=policy,
        ):
            batch = inbox_item.get("batch")
            if not isinstance(batch, dict):
                continue
            if candidate_status == "processing" and not (
                _cai_owned_transport_batch_lease_expired(
                    batch,
                    datetime.now(tz=UTC),
                )
            ):
                continue
            batch_id = str(batch.get("batchId") or "").strip()
            item_session_id = str(inbox_item.get("sessionId") or "").strip()
            if not batch_id or not item_session_id:
                continue
            try:
                record = claim_cai_owned_transport_batch(
                    item_session_id,
                    batch_id,
                    node_id=local,
                    runtime_id=runtime_id,
                    runtime_auth_token=runtime_auth_token,
                    require_runtime_auth=require_runtime_auth,
                    lease_seconds=lease_seconds,
                    policy=policy,
                )
            except ValueError as exc:
                if "already processing" in str(exc):
                    continue
                raise
            claimed_batch = _find_cai_owned_transport_batch(record, batch_id)
            if claimed_batch is None:
                continue
            return cai_owned_transport_batch_work_item(record, claimed_batch)
    return None


def cai_owned_transport_batch_work_item(
    record: CaiOwnedTransportSessionRecord,
    batch: dict[str, Any],
) -> dict[str, Any]:
    batch_copy = dict(batch)
    batch_id = str(batch_copy.get("batchId") or "").strip()
    metadata = (
        batch_copy.get("metadata")
        if isinstance(batch_copy.get("metadata"), dict)
        else {}
    )
    return {
        "sessionId": record.session_id,
        "instanceId": record.instance_id,
        "network": record.chain_id,
        "chainId": record.chain_id,
        "modelId": record.model_id,
        "taskId": record.task_id,
        "sourceNodeId": record.source_node_id,
        "participantNodeIds": list(record.participant_node_ids),
        "executorNodeIds": list(record.executor_node_ids or record.participant_node_ids),
        "batch": batch_copy,
        "payloadEndpoint": (
            f"/v1/cai/transport/sessions/{quote(record.session_id, safe='')}"
            f"/batches/{quote(batch_id, safe='')}/payload"
        ),
        "payloadStorageKey": metadata.get("payloadStorageKey"),
        "payloadSha256Hex": batch_copy.get("payloadSha256Hex"),
        "payloadSizeBytes": batch_copy.get("payloadSizeBytes"),
    }


def cai_owned_transport_shard_receipts_from_processed_batches(
    record: CaiOwnedTransportSessionRecord,
) -> list[dict[str, Any]]:
    receipts_by_node: dict[str, dict[str, Any]] = {}
    for batch in record.batch_records:
        if not isinstance(batch, dict):
            continue
        if str(batch.get("status") or "").strip() != "processed":
            continue
        node_id = str(batch.get("sinkNodeId") or "").strip()
        if not node_id:
            continue
        receipt = receipts_by_node.setdefault(
            node_id,
            {
                "nodeId": node_id,
                "network": record.chain_id,
                "chainId": record.chain_id,
                "status": "completed",
                "activationBatchCount": 0,
                "decodeBatchCount": 0,
                "layerStart": None,
                "layerEnd": None,
                "batchIds": [],
                "stageIds": [],
                "sequences": [],
                "inputPayloadSha256Hexes": [],
                "outputPayloadSha256Hexes": [],
                "hashChainSha256Hexes": [],
                "routeAudits": [],
                "runtimeAudits": [],
                "metrics": {
                    "processedBatchCount": 0,
                    "payloadSizeBytes": 0,
                    "outputPayloadSizeBytes": 0,
                    "promptTokenCount": 0,
                    "completionTokenCount": 0,
                    "inputTokenCount": 0,
                    "outputTokenCount": 0,
                    "tokenCount": 0,
                },
            },
        )
        phase = str(batch.get("phase") or "").strip()
        if phase == "prefill_activation_batches":
            receipt["activationBatchCount"] += 1
        elif phase == "decode_activation_batches":
            receipt["decodeBatchCount"] += 1
        receipt["batchIds"].append(str(batch.get("batchId") or "").strip())
        sequence = _optional_int(batch.get("sequence"))
        if sequence is None:
            metadata = batch.get("metadata")
            if isinstance(metadata, dict):
                sequence = _optional_int(metadata.get("sequence"))
        if sequence is not None:
            receipt["sequences"].append(sequence)
        for batch_field, receipt_field in (
            ("payloadSha256Hex", "inputPayloadSha256Hexes"),
            ("inputPayloadSha256Hex", "inputPayloadSha256Hexes"),
            ("outputPayloadSha256Hex", "outputPayloadSha256Hexes"),
            ("hashChainSha256Hex", "hashChainSha256Hexes"),
        ):
            value = str(batch.get(batch_field) or "").strip().lower()
            if value:
                _append_unique(receipt[receipt_field], value)
        route_audit = batch.get("routeAudit")
        if isinstance(route_audit, dict):
            receipt["routeAudits"].append(dict(route_audit))
        runtime_audit = batch.get("runtimeAudit")
        if isinstance(runtime_audit, dict):
            receipt["runtimeAudits"].append(dict(runtime_audit))
            _append_unique_metric(
                receipt["metrics"],
                "runtimeVersions",
                runtime_audit.get("runtimeVersion"),
            )
            _append_unique_metric(
                receipt["metrics"],
                "adapterIds",
                runtime_audit.get("adapterId"),
            )
            _append_unique_metric(
                receipt["metrics"],
                "adapterVersions",
                runtime_audit.get("adapterVersion"),
            )
        receipt["metrics"]["processedBatchCount"] += 1
        try:
            receipt["metrics"]["payloadSizeBytes"] += max(
                0,
                int(batch.get("payloadSizeBytes") or 0),
            )
        except (TypeError, ValueError):
            pass
        try:
            receipt["metrics"]["outputPayloadSizeBytes"] += max(
                0,
                int(batch.get("outputPayloadSizeBytes") or 0),
            )
        except (TypeError, ValueError):
            pass
        metadata = (
            batch.get("metadata")
            if isinstance(batch.get("metadata"), dict)
            else {}
        )
        direct_final_output = bool(metadata.get("singleExecutorDirectFinalOutput"))
        batch_metrics = batch.get("metrics")
        if isinstance(batch_metrics, dict):
            input_metric_value = _first_metric_value(
                batch_metrics,
                ("inputTokens", "inputTokenCount"),
            )
            output_metric_value = _first_metric_value(
                batch_metrics,
                ("outputTokens", "outputTokenCount"),
            )
            prompt_metric_value = _first_metric_value(
                batch_metrics,
                ("promptTokens", "promptTokenCount"),
            )
            completion_metric_value = _first_metric_value(
                batch_metrics,
                ("completionTokens", "completionTokenCount"),
            )
            if (
                prompt_metric_value is None
                and phase == "prefill_activation_batches"
            ):
                prompt_metric_value = input_metric_value
            if (
                prompt_metric_value is None
                and phase == "decode_activation_batches"
                and direct_final_output
            ):
                prompt_metric_value = input_metric_value
            if (
                completion_metric_value is None
                and phase == "decode_activation_batches"
            ):
                completion_metric_value = output_metric_value
            for metric_value, receipt_field in (
                (input_metric_value, "inputTokenCount"),
                (output_metric_value, "outputTokenCount"),
                (
                    _first_metric_value(
                        batch_metrics,
                        ("tokens", "tokenCount", "totalTokens"),
                    ),
                    "tokenCount",
                ),
                (prompt_metric_value, "promptTokenCount"),
                (completion_metric_value, "completionTokenCount"),
            ):
                if metric_value is not None:
                    try:
                        receipt["metrics"][receipt_field] += max(
                            0,
                            int(metric_value or 0),
                        )
                    except (TypeError, ValueError):
                        pass
            latency_ms = batch_metrics.get("processingLatencyMs")
            if latency_ms is None:
                latency_ms = batch_metrics.get("latencyMs")
            if latency_ms is not None:
                try:
                    receipt["metrics"].setdefault("processingLatencyMs", []).append(
                        max(0.0, float(latency_ms))
                    )
                except (TypeError, ValueError):
                    pass
            _append_unique_metric(
                receipt["metrics"],
                "adapterIds",
                batch_metrics.get("adapterId") or batch_metrics.get("adapter"),
            )
            _append_unique_metric(
                receipt["metrics"],
                "adapterVersions",
                batch_metrics.get("adapterVersion"),
            )
            _append_unique_metric(
                receipt["metrics"],
                "runtimeVersions",
                batch_metrics.get("runtimeVersion"),
            )
        stage_id = str(metadata.get("stageId") or "").strip()
        if stage_id:
            _append_unique(receipt["stageIds"], stage_id)
        layer_start = _optional_int(metadata.get("layerStart"))
        layer_end = _optional_int(metadata.get("layerEnd"))
        if layer_start is not None:
            existing_start = receipt.get("layerStart")
            receipt["layerStart"] = (
                layer_start
                if existing_start is None
                else min(int(existing_start), layer_start)
            )
        if layer_end is not None:
            existing_end = receipt.get("layerEnd")
            receipt["layerEnd"] = (
                layer_end if existing_end is None else max(int(existing_end), layer_end)
            )
    return [
        receipts_by_node[node_id]
        for node_id in record.participant_node_ids
        if node_id in receipts_by_node
    ]


def complete_cai_owned_transport_batch_processing(
    session_id: str,
    batch_id: str,
    *,
    node_id: str,
    runtime_id: str | None = None,
    coordinator_cai_url: str | None = None,
    metrics: dict[str, Any] | None = None,
    output_payload: bytes | None = None,
    output_payload_sha256_hex: str | None = None,
    route_audit: dict[str, Any] | None = None,
    runtime_audit: dict[str, Any] | None = None,
    signing_material: Mapping[str, Any] | None = None,
    policy: WalletPolicy | None = None,
    timeout_sec: float = 5.0,
) -> dict[str, Any]:
    _require_cai_owned_transport_batch_completion_owner(
        session_id,
        batch_id,
        node_id=node_id,
        runtime_id=runtime_id,
        policy=policy,
    )
    source_batch = _cai_owned_transport_batch_record(session_id, batch_id, policy)
    if source_batch is None:
        raise ValueError(f"CAI-owned transport batch '{batch_id}' not found.")
    output_payload_metadata: dict[str, Any] | None = None
    output_payload_size_bytes: int | None = None
    output_payload_storage_key: str | None = None
    if output_payload is not None:
        output_payload_metadata = store_cai_owned_transport_batch_output_payload(
            session_id,
            batch_id,
            output_payload,
            policy,
        )
        output_payload_sha256_hex = (
            output_payload_sha256_hex
            or output_payload_metadata["outputPayloadSha256Hex"]
        )
        output_payload_size_bytes = int(
            output_payload_metadata["outputPayloadSizeBytes"]
        )
        output_payload_storage_key = str(
            output_payload_metadata["outputPayloadStorageKey"]
        )
    input_payload_sha256_hex = str(source_batch.get("payloadSha256Hex") or "").strip()
    metadata = (
        source_batch.get("metadata")
        if isinstance(source_batch.get("metadata"), dict)
        else {}
    )
    sequence = _optional_int(source_batch.get("sequence"))
    if sequence is None:
        sequence = _optional_int(metadata.get("sequence"))
    if sequence is None:
        sequence = 0
    previous_batch_id = (
        str(source_batch.get("previousBatchId") or "").strip()
        or str(metadata.get("previousBatchId") or "").strip()
        or None
    )
    hash_chain: dict[str, Any] | None = None
    if output_payload_sha256_hex and input_payload_sha256_hex:
        hash_chain = build_cai_owned_transport_batch_hash_chain(
            session_id=session_id,
            batch_id=batch_id,
            input_payload_sha256_hex=input_payload_sha256_hex,
            output_payload_sha256_hex=output_payload_sha256_hex,
            sequence=sequence,
            previous_batch_id=previous_batch_id,
        )
    record = mark_cai_owned_transport_batch_status(
        session_id,
        batch_id,
        status="processed",
        node_id=node_id,
        metrics=metrics,
        input_payload_sha256_hex=hash_chain["inputPayloadSha256Hex"]
        if hash_chain is not None
        else None,
        output_payload_sha256_hex=output_payload_sha256_hex,
        output_payload_size_bytes=output_payload_size_bytes,
        output_payload_storage_key=output_payload_storage_key,
        previous_batch_id=hash_chain["previousBatchId"]
        if hash_chain is not None
        else None,
        hash_chain_sha256_hex=hash_chain["hashChainSha256Hex"]
        if hash_chain is not None
        else None,
        route_audit=route_audit,
        runtime_audit=runtime_audit,
        policy=policy,
    )
    clean_node_id = str(node_id or "").strip()
    receipts = cai_owned_transport_shard_receipts_from_processed_batches(record)
    receipt = next(
        (
            item
            for item in receipts
            if str(item.get("nodeId") or "").strip() == clean_node_id
        ),
        None,
    )
    if receipt is None:
        raise ValueError("CAI-owned transport processed batch receipt was not found.")

    coordinator_response: dict[str, Any] | None = None
    if str(coordinator_cai_url or "").strip():
        try:
            coordinator_response = submit_cai_owned_transport_shard_receipt(
                str(coordinator_cai_url),
                record.session_id,
                node_id=clean_node_id,
                chain_id=record.chain_id,
                activation_batch_count=int(receipt.get("activationBatchCount") or 0),
                decode_batch_count=int(receipt.get("decodeBatchCount") or 0),
                layer_start=_optional_int(receipt.get("layerStart")),
                layer_end=_optional_int(receipt.get("layerEnd")),
                metrics=receipt.get("metrics")
                if isinstance(receipt.get("metrics"), dict)
                else None,
                batch_ids=receipt.get("batchIds")
                if isinstance(receipt.get("batchIds"), Sequence)
                and not isinstance(receipt.get("batchIds"), (str, bytes))
                else None,
                stage_ids=receipt.get("stageIds")
                if isinstance(receipt.get("stageIds"), Sequence)
                and not isinstance(receipt.get("stageIds"), (str, bytes))
                else None,
                sequences=receipt.get("sequences")
                if isinstance(receipt.get("sequences"), Sequence)
                and not isinstance(receipt.get("sequences"), (str, bytes))
                else None,
                input_payload_sha256_hexes=receipt.get("inputPayloadSha256Hexes")
                if isinstance(receipt.get("inputPayloadSha256Hexes"), Sequence)
                and not isinstance(
                    receipt.get("inputPayloadSha256Hexes"), (str, bytes)
                )
                else None,
                output_payload_sha256_hexes=receipt.get("outputPayloadSha256Hexes")
                if isinstance(receipt.get("outputPayloadSha256Hexes"), Sequence)
                and not isinstance(
                    receipt.get("outputPayloadSha256Hexes"), (str, bytes)
                )
                else None,
                hash_chain_sha256_hexes=receipt.get("hashChainSha256Hexes")
                if isinstance(receipt.get("hashChainSha256Hexes"), Sequence)
                and not isinstance(
                    receipt.get("hashChainSha256Hexes"), (str, bytes)
                )
                else None,
                route_audits=receipt.get("routeAudits")
                if isinstance(receipt.get("routeAudits"), Sequence)
                and not isinstance(receipt.get("routeAudits"), (str, bytes))
                else None,
                runtime_audits=receipt.get("runtimeAudits")
                if isinstance(receipt.get("runtimeAudits"), Sequence)
                and not isinstance(receipt.get("runtimeAudits"), (str, bytes))
                else None,
                signing_material=signing_material,
                timeout_sec=timeout_sec,
            )
        except Exception as exc:
            coordinator_response = {
                "status": "failed",
                "error": str(exc),
                "retryable": True,
            }
    return {
        "sessionId": record.session_id,
        "batchId": str(batch_id or "").strip(),
        "nodeId": clean_node_id,
        "receipt": receipt,
        "outputPayload": output_payload_metadata,
        "coordinatorResponse": coordinator_response,
    }


def complete_cai_owned_transport_work_item(
    session_id: str,
    batch_id: str,
    *,
    node_id: str,
    runtime_id: str | None = None,
    runtime_auth_token: str | None = None,
    require_runtime_auth: bool | str | None = None,
    coordinator_cai_url: str | None = None,
    metrics: dict[str, Any] | None = None,
    output_payload: bytes | None = None,
    output_payload_sha256_hex: str | None = None,
    route_audit: dict[str, Any] | None = None,
    runtime_audit: dict[str, Any] | None = None,
    signing_material: Mapping[str, Any] | None = None,
    policy: WalletPolicy | None = None,
    timeout_sec: float = 5.0,
) -> dict[str, Any]:
    _require_cai_owned_transport_local_runtime_auth(
        runtime_auth_token,
        require_runtime_auth=require_runtime_auth,
    )
    return complete_cai_owned_transport_batch_processing(
        session_id,
        batch_id,
        node_id=node_id,
        runtime_id=runtime_id,
        coordinator_cai_url=coordinator_cai_url,
        metrics=metrics,
        output_payload=output_payload,
        output_payload_sha256_hex=output_payload_sha256_hex,
        route_audit=route_audit,
        runtime_audit=runtime_audit,
        signing_material=signing_material,
        policy=policy,
        timeout_sec=timeout_sec,
    )


def fail_cai_owned_transport_work_item(
    session_id: str,
    batch_id: str,
    *,
    node_id: str,
    runtime_id: str | None = None,
    runtime_auth_token: str | None = None,
    require_runtime_auth: bool | str | None = None,
    error: str | None = None,
    retryable: bool = True,
    max_attempts: int | None = None,
    metrics: dict[str, Any] | None = None,
    policy: WalletPolicy | None = None,
) -> dict[str, Any]:
    _require_cai_owned_transport_local_runtime_auth(
        runtime_auth_token,
        require_runtime_auth=require_runtime_auth,
    )
    _require_cai_owned_transport_batch_completion_owner(
        session_id,
        batch_id,
        node_id=node_id,
        runtime_id=runtime_id,
        policy=policy,
    )
    clean_session_id = str(session_id or "").strip()
    clean_batch_id = _require_safe_transport_file_id(batch_id, prefix="caibatch_")
    clean_error = str(error or "").strip() or "CAI-owned transport work item failed."
    clean_node_id = str(node_id or "").strip()
    resolved_max_attempts = _coerce_cai_owned_transport_max_attempts(max_attempts)
    records = list_cai_owned_transport_sessions(policy)
    for record_index, record in enumerate(records):
        if record.session_id != clean_session_id:
            continue
        local = _require_session_participant(record, clean_node_id)
        for batch_index, batch in enumerate(record.batch_records):
            if str(batch.get("batchId") or "").strip() != clean_batch_id:
                continue
            if str(batch.get("sinkNodeId") or "").strip() != local:
                raise ValueError(
                    "CAI-owned transport batch is not assigned to local node."
                )
            now = datetime.now(tz=UTC).isoformat()
            updated_batch = dict(batch)
            try:
                attempt_count = int(updated_batch.get("attemptCount") or 0)
            except (TypeError, ValueError):
                attempt_count = 0
            try:
                failure_count = int(updated_batch.get("failureCount") or 0)
            except (TypeError, ValueError):
                failure_count = 0
            final_failure = (
                (not bool(retryable)) or attempt_count >= resolved_max_attempts
            )
            updated_batch["updatedAt"] = now
            updated_batch["lastFailedAt"] = now
            updated_batch["lastError"] = clean_error
            updated_batch["error"] = clean_error
            updated_batch["retryable"] = bool(retryable)
            updated_batch["maxAttempts"] = resolved_max_attempts
            updated_batch["failureCount"] = failure_count + 1
            if metrics is not None:
                updated_batch["metrics"] = dict(metrics)
            if final_failure:
                updated_batch["status"] = "failed"
                updated_batch["failedAt"] = now
                updated_batch["processedAt"] = now
            else:
                updated_batch["status"] = "received"
                updated_batch["retryScheduledAt"] = now
                _clear_cai_owned_transport_batch_runtime_claim(updated_batch)
            record.batch_records[batch_index] = updated_batch
            record.updated_at = now
            records[record_index] = record
            save_cai_owned_transport_sessions(records, policy)
            return {
                "sessionId": record.session_id,
                "batchId": clean_batch_id,
                "nodeId": local,
                "status": updated_batch["status"],
                "retryScheduled": not final_failure,
                "attemptCount": attempt_count,
                "maxAttempts": resolved_max_attempts,
                "error": clean_error,
            }
        raise ValueError(
            f"CAI-owned transport batch '{clean_batch_id}' not found in session "
            f"'{clean_session_id}'."
        )
    raise ValueError(f"CAI-owned transport session '{clean_session_id}' not found.")


def record_cai_owned_transport_shard_receipt(
    session_id: str,
    *,
    node_id: str,
    chain_id: str | None = None,
    status: str = "completed",
    activation_batch_count: int = 0,
    decode_batch_count: int = 0,
    layer_start: int | None = None,
    layer_end: int | None = None,
    metrics: dict[str, Any] | None = None,
    batch_ids: Sequence[str] | None = None,
    stage_ids: Sequence[str] | None = None,
    sequences: Sequence[int] | None = None,
    input_payload_sha256_hexes: Sequence[str] | None = None,
    output_payload_sha256_hexes: Sequence[str] | None = None,
    hash_chain_sha256_hexes: Sequence[str] | None = None,
    route_audits: Sequence[dict[str, Any]] | None = None,
    runtime_audits: Sequence[dict[str, Any]] | None = None,
    signature: Mapping[str, Any] | None = None,
    signer_node_id: str | None = None,
    recorded_at: str | None = None,
    policy: WalletPolicy | None = None,
) -> CaiOwnedTransportSessionRecord:
    clean_session_id = str(session_id or "").strip()
    records = list_cai_owned_transport_sessions(policy)
    for index, record in enumerate(records):
        if record.session_id != clean_session_id:
            continue
        if record.status == "completed":
            raise ValueError("Cannot append shard receipts to a completed session.")
        record_chain_id = record.chain_id or _cai_owned_transport_chain_id(policy)
        if chain_id is None:
            raise ValueError("CAI-owned transport shard receipt chain id is missing.")
        if _cai_owned_transport_chain_id(None, chain_id) != record_chain_id:
            raise ValueError("CAI-owned transport shard receipt chain id does not match.")
        clean_node_id = _require_session_participant(record, node_id)
        clean_status = str(status or "").strip() or "completed"
        now = datetime.now(tz=UTC).isoformat()
        receipt = {
            "nodeId": clean_node_id,
            "network": record_chain_id,
            "chainId": record_chain_id,
            "status": clean_status,
            "activationBatchCount": max(0, int(activation_batch_count or 0)),
            "decodeBatchCount": max(0, int(decode_batch_count or 0)),
            "layerStart": layer_start,
            "layerEnd": layer_end,
            "metrics": dict(metrics or {}),
            "batchIds": _clean_cai_owned_transport_receipt_batch_ids(batch_ids),
            "stageIds": _clean_cai_owned_transport_receipt_stage_ids(stage_ids),
            "sequences": _clean_cai_owned_transport_receipt_sequences(sequences),
            "inputPayloadSha256Hexes": _clean_cai_owned_transport_receipt_hashes(
                input_payload_sha256_hexes,
                field_name="inputPayloadSha256Hexes",
            ),
            "outputPayloadSha256Hexes": _clean_cai_owned_transport_receipt_hashes(
                output_payload_sha256_hexes,
                field_name="outputPayloadSha256Hexes",
            ),
            "hashChainSha256Hexes": _clean_cai_owned_transport_receipt_hashes(
                hash_chain_sha256_hexes,
                field_name="hashChainSha256Hexes",
            ),
            "routeAudits": _clean_cai_owned_transport_receipt_audits(route_audits),
            "runtimeAudits": _clean_cai_owned_transport_receipt_audits(
                runtime_audits
            ),
            "recordedAt": now,
        }
        if isinstance(signature, Mapping):
            clean_signer_node_id = str(signer_node_id or "").strip()
            if clean_signer_node_id:
                receipt["signerNodeId"] = clean_signer_node_id
            receipt["signature"] = dict(signature)
            if recorded_at is not None:
                receipt["recordedAt"] = str(recorded_at or "").strip() or None
            else:
                receipt.pop("recordedAt", None)
        record.shard_receipts = [
            item
            for item in record.shard_receipts
            if str(item.get("nodeId") or "").strip() != clean_node_id
        ]
        record.shard_receipts.append(receipt)
        if record.status == "created":
            record.status = "running"
        record.updated_at = now
        records[index] = record
        save_cai_owned_transport_sessions(records, policy)
        return record
    raise ValueError(f"CAI-owned transport session '{clean_session_id}' not found.")


def submit_cai_owned_transport_shard_receipt(
    coordinator_cai_url: str,
    session_id: str,
    *,
    node_id: str,
    chain_id: str | None = None,
    status: str = "completed",
    activation_batch_count: int = 0,
    decode_batch_count: int = 0,
    layer_start: int | None = None,
    layer_end: int | None = None,
    metrics: dict[str, Any] | None = None,
    batch_ids: Sequence[str] | None = None,
    stage_ids: Sequence[str] | None = None,
    sequences: Sequence[int] | None = None,
    input_payload_sha256_hexes: Sequence[str] | None = None,
    output_payload_sha256_hexes: Sequence[str] | None = None,
    hash_chain_sha256_hexes: Sequence[str] | None = None,
    route_audits: Sequence[dict[str, Any]] | None = None,
    runtime_audits: Sequence[dict[str, Any]] | None = None,
    signing_material: Mapping[str, Any] | None = None,
    timeout_sec: float = 5.0,
) -> dict[str, Any]:
    base_url = str(coordinator_cai_url or "").strip().rstrip("/")
    clean_session_id = str(session_id or "").strip()
    resolved_chain_id = _cai_owned_transport_chain_id(None, chain_id)
    if not clean_session_id:
        raise ValueError("CAI-owned transport session id is required.")
    payload = {
        "nodeId": str(node_id or "").strip(),
        "chainId": resolved_chain_id,
        "network": resolved_chain_id,
        "status": str(status or "completed").strip() or "completed",
        "activationBatchCount": max(0, int(activation_batch_count or 0)),
        "decodeBatchCount": max(0, int(decode_batch_count or 0)),
        "layerStart": layer_start,
        "layerEnd": layer_end,
        "metrics": dict(metrics or {}),
        "batchIds": _clean_cai_owned_transport_receipt_batch_ids(batch_ids),
        "stageIds": _clean_cai_owned_transport_receipt_stage_ids(stage_ids),
        "sequences": _clean_cai_owned_transport_receipt_sequences(sequences),
        "inputPayloadSha256Hexes": _clean_cai_owned_transport_receipt_hashes(
            input_payload_sha256_hexes,
            field_name="inputPayloadSha256Hexes",
        ),
        "outputPayloadSha256Hexes": _clean_cai_owned_transport_receipt_hashes(
            output_payload_sha256_hexes,
            field_name="outputPayloadSha256Hexes",
        ),
        "hashChainSha256Hexes": _clean_cai_owned_transport_receipt_hashes(
            hash_chain_sha256_hexes,
            field_name="hashChainSha256Hexes",
        ),
        "routeAudits": _clean_cai_owned_transport_receipt_audits(route_audits),
        "runtimeAudits": _clean_cai_owned_transport_receipt_audits(runtime_audits),
    }
    signing_kwargs = cai_owned_transport_peer_signing_kwargs(signing_material)
    if signing_kwargs:
        payload = sign_cai_owned_transport_shard_receipt(
            payload,
            signer_node_id=str(node_id or "").strip(),
            **signing_kwargs,
        )
    overlay_target = _parse_cai_owned_transport_overlay_url(coordinator_cai_url)
    if overlay_target is not None:
        relay_url, target_node_id = overlay_target
        return submit_cai_owned_transport_overlay_message(
            relay_url,
            kind="shard_receipt",
            source_node_id=str(node_id or "").strip(),
            target_node_id=target_node_id,
            session_id=clean_session_id,
            payload=payload,
            timeout_sec=timeout_sec,
        )
    if not base_url:
        raise ValueError("Coordinator CAI URL is required.")
    url = (
        f"{base_url}/v1/cai/transport/sessions/"
        f"{quote(clean_session_id, safe='')}/shard-receipts"
    )
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=_cai_owned_transport_json_headers(chain_id=resolved_chain_id),
        method="POST",
    )
    with urlopen(request, timeout=timeout_sec) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw or "{}")
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def submit_cai_owned_transport_completion_notice(
    peer_cai_url: str,
    session_id: str,
    *,
    proof: Mapping[str, Any],
    source_node_id: str | None = None,
    target_node_id: str | None = None,
    chain_id: str | None = None,
    timeout_sec: float = 5.0,
) -> dict[str, Any]:
    base_url = str(peer_cai_url or "").strip().rstrip("/")
    clean_session_id = str(session_id or "").strip()
    if not clean_session_id:
        raise ValueError("CAI-owned transport session id is required.")
    if not isinstance(proof, Mapping):
        raise ValueError("CAI-owned transport completion proof must be an object.")
    resolved_chain_id = _cai_owned_transport_chain_id(
        None,
        chain_id
        or proof.get("chainId")
        or proof.get("network"),
    )
    payload = {
        "proof": dict(proof),
        "chainId": resolved_chain_id,
        "network": resolved_chain_id,
    }
    overlay_target = _parse_cai_owned_transport_overlay_url(peer_cai_url)
    if overlay_target is not None:
        relay_url, overlay_target_node_id = overlay_target
        return submit_cai_owned_transport_overlay_message(
            relay_url,
            kind="completion_notice",
            source_node_id=str(source_node_id or "").strip(),
            target_node_id=(
                str(target_node_id or "").strip() or overlay_target_node_id
            ),
            session_id=clean_session_id,
            payload=payload,
            timeout_sec=timeout_sec,
        )
    if not base_url:
        raise ValueError("Peer CAI URL is required.")
    url = (
        f"{base_url}/v1/cai/transport/sessions/"
        f"{quote(clean_session_id, safe='')}/completion-notice"
    )
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=_cai_owned_transport_json_headers(chain_id=resolved_chain_id),
        method="POST",
    )
    with urlopen(request, timeout=timeout_sec) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw or "{}")
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def latest_completed_cai_owned_transport_proof_for_instance(
    instance_id: str,
    *,
    participant_node_ids: Sequence[str] | None = None,
    model_id: str | None = None,
    policy: WalletPolicy | None = None,
) -> dict[str, Any] | None:
    clean_instance_id = str(instance_id or "").strip()
    if not clean_instance_id:
        return None
    for record in list_cai_owned_transport_sessions(policy):
        if record.instance_id != clean_instance_id or record.status != "completed":
            continue
        valid, _error = validate_cai_owned_transport_execution_proof(
            record.proof,
            executor_node_ids=participant_node_ids,
            model_id=model_id,
            chain_id=record.chain_id,
        )
        if valid:
            return record.proof
    return None


def cai_owned_transport_session_to_dict(
    record: CaiOwnedTransportSessionRecord,
) -> dict[str, Any]:
    return {
        "sessionId": record.session_id,
        "instanceId": record.instance_id,
        "modelId": record.model_id,
        "taskId": record.task_id,
        "sourceNodeId": record.source_node_id,
        "network": record.chain_id,
        "chainId": record.chain_id,
        "participantNodeIds": list(record.participant_node_ids),
        "executorNodeIds": list(record.executor_node_ids or record.participant_node_ids),
        "executionMode": record.execution_mode,
        "routePolicy": dict(record.route_policy),
        "dispatchRecords": list(record.dispatch_records),
        "batchRecords": list(record.batch_records),
        "shardReceipts": list(record.shard_receipts),
        "status": record.status,
        "createdAt": record.created_at,
        "updatedAt": record.updated_at,
        "completedAt": record.completed_at,
        "lastError": record.last_error,
        "proof": record.proof,
    }


def plan_decentralized_llm_execution(
    source_node_id: str,
    sink_node_ids: Sequence[str],
    route_health_records: Sequence[Any] | None,
    *,
    model_id: str | None = None,
    backend: str = "llama.cpp",
    low_latency_max_ms: float | None = None,
    wan_risky_max_ms: float | None = None,
) -> dict[str, Any]:
    source = str(source_node_id or "").strip()
    sinks = _clean_sink_node_ids(source, sink_node_ids)
    compute_cell_profile = llama_cpp_compute_cell_profile_for_path(
        source,
        sinks,
        route_health_records,
        low_latency_max_ms=low_latency_max_ms,
        wan_risky_max_ms=wan_risky_max_ms,
    )
    execution_mode = _execution_mode_for_compute_cell(compute_cell_profile)
    requires_transport = execution_mode == EXECUTION_MODE_CAI_OWNED_TRANSPORT_REQUIRED
    reason = _execution_reason(execution_mode, compute_cell_profile)
    cai_owned_transport = None
    if requires_transport:
        cai_owned_transport = build_cai_owned_transport_contract(
            source_node_id=source,
            sink_node_ids=sinks,
            reason=reason,
            model_id=model_id,
            backend=backend,
        )
        cai_owned_transport["routeHealthReadiness"] = (
            cai_owned_transport_route_health_readiness(
                source_node_id=source,
                sink_node_ids=sinks,
                route_health_records=route_health_records,
                route_policy=cai_owned_transport.get("routePolicy"),
            )
        )

    return {
        "modelId": model_id,
        "backend": backend,
        "sourceNodeId": source,
        "sinkNodeIds": sinks,
        "participantNodeIds": [source, *sinks] if source else sinks,
        "executionMode": execution_mode,
        "standardLlamaCppRpcReady": bool(
            compute_cell_profile.get("readyForLlamaCppRpc")
        ),
        "requiresCaiOwnedTransport": requires_transport,
        "reason": reason,
        "computeCellProfile": compute_cell_profile,
        "caiOwnedTransport": cai_owned_transport,
    }


def plan_llama_cpp_distributed_execution(
    source_node_id: str,
    sink_node_ids: Sequence[str],
    route_health_records: Sequence[Any] | None,
    *,
    model_id: str | None = None,
    low_latency_max_ms: float | None = None,
    wan_risky_max_ms: float | None = None,
) -> dict[str, Any]:
    return plan_decentralized_llm_execution(
        source_node_id,
        sink_node_ids,
        route_health_records,
        model_id=model_id,
        backend="llama.cpp",
        low_latency_max_ms=low_latency_max_ms,
        wan_risky_max_ms=wan_risky_max_ms,
    )


def build_cai_owned_transport_execution_dag(
    *,
    session_id: str,
    requester_node_id: str,
    executor_node_ids: Sequence[str],
    total_layer_count: int,
    shard_ranges: Sequence[Mapping[str, Any]] | None = None,
    chain_id: str | None = None,
    model_id: str | None = None,
    task_id: str | None = None,
    coordinator_node_id: str | None = None,
    input_payload_sha256_hex: str | None = None,
    expected_output_payload_sha256_hex: str | None = None,
    created_at: str | None = None,
    single_executor_direct_final_output: bool = False,
    execution_pipeline_mode: str | None = None,
) -> dict[str, Any]:
    clean_session_id = str(session_id or "").strip()
    if not clean_session_id:
        raise ValueError("CAI-owned transport execution DAG requires session id.")
    requester = str(requester_node_id or "").strip()
    if not requester:
        raise ValueError("CAI-owned transport execution DAG requires requester node.")
    executors = _clean_node_ids(executor_node_ids)
    if not executors:
        raise ValueError("CAI-owned transport execution DAG requires executors.")
    try:
        total_layers = int(total_layer_count)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "CAI-owned transport execution DAG total layer count is invalid."
        ) from exc
    if total_layers <= 0:
        raise ValueError(
            "CAI-owned transport execution DAG total layer count is invalid."
        )
    if len(executors) > total_layers:
        raise ValueError(
            "CAI-owned transport execution DAG executor count exceeds total layer count."
        )
    resolved_chain_id = _cai_owned_transport_chain_id(None, chain_id)
    coordinator = str(coordinator_node_id or "").strip() or requester
    participants = _clean_node_ids([requester, coordinator, *executors])
    input_hash = _optional_sha256_hex(
        input_payload_sha256_hex,
        field_name="inputPayloadSha256Hex",
    )
    expected_output_hash = _optional_sha256_hex(
        expected_output_payload_sha256_hex,
        field_name="expectedOutputPayloadSha256Hex",
    )
    shard_ranges = _normalize_cai_owned_transport_shard_ranges(
        executors,
        total_layers,
        shard_ranges=shard_ranges,
    )
    stages: list[dict[str, Any]] = []
    sequence = 0
    last_prefill_stage_id: str | None = None
    direct_final_output = (
        bool(single_executor_direct_final_output) and len(shard_ranges) == 1
    )
    pipeline_mode = (
        str(execution_pipeline_mode or "").strip()
        or CAI_OWNED_TRANSPORT_EXECUTION_PIPELINE_FULL_PREFILL_DECODE
    )
    supported_pipeline_modes = {
        CAI_OWNED_TRANSPORT_EXECUTION_PIPELINE_FULL_PREFILL_DECODE,
        CAI_OWNED_TRANSPORT_EXECUTION_PIPELINE_SINGLE_PASS_FINAL_DECODE,
    }
    if pipeline_mode not in supported_pipeline_modes:
        raise ValueError("CAI-owned transport execution DAG pipeline mode is invalid.")
    if direct_final_output:
        pipeline_mode = CAI_OWNED_TRANSPORT_EXECUTION_PIPELINE_SINGLE_PASS_FINAL_DECODE

    if direct_final_output:
        shard_range = shard_ranges[0]
        executor = str(shard_range["nodeId"])
        layer_start = int(shard_range["layerStart"])
        layer_end = int(shard_range["layerEnd"])
        phase = "decode_activation_batches"
        stage_id = _cai_owned_transport_stage_id(
            session_id=clean_session_id,
            phase=phase,
            sequence=sequence,
            executor_node_id=executor,
            layer_start=layer_start,
            layer_end=layer_end,
        )
        stages.append(
            {
                "stageId": stage_id,
                "sequence": sequence,
                "phase": phase,
                "sourceNodeId": requester,
                "sinkNodeId": executor,
                "executorNodeId": executor,
                "outputToNodeId": requester,
                "dependsOnStageIds": [],
                "layerStart": layer_start,
                "layerEnd": layer_end,
                "expectedInputPayloadSha256Hex": input_hash,
                "expectedOutputPayloadSha256Hex": expected_output_hash,
                "payloadRole": "final_output",
                "singleExecutorDirectFinalOutput": True,
            }
        )
        sequence += 1
    elif pipeline_mode == CAI_OWNED_TRANSPORT_EXECUTION_PIPELINE_SINGLE_PASS_FINAL_DECODE:
        previous_stage_id: str | None = None
        for index, shard_range in enumerate(shard_ranges):
            executor = str(shard_range["nodeId"])
            layer_start = int(shard_range["layerStart"])
            layer_end = int(shard_range["layerEnd"])
            is_final_shard = index + 1 == len(shard_ranges)
            phase = (
                "decode_activation_batches"
                if is_final_shard
                else "prefill_activation_batches"
            )
            source = requester if index == 0 else str(shard_ranges[index - 1]["nodeId"])
            output_to = (
                requester if is_final_shard else str(shard_ranges[index + 1]["nodeId"])
            )
            depends_on_stage_ids = [previous_stage_id] if previous_stage_id else []
            stage_id = _cai_owned_transport_stage_id(
                session_id=clean_session_id,
                phase=phase,
                sequence=sequence,
                executor_node_id=executor,
                layer_start=layer_start,
                layer_end=layer_end,
            )
            stage = {
                "stageId": stage_id,
                "sequence": sequence,
                "phase": phase,
                "sourceNodeId": source,
                "sinkNodeId": executor,
                "executorNodeId": executor,
                "outputToNodeId": output_to,
                "dependsOnStageIds": depends_on_stage_ids,
                "layerStart": layer_start,
                "layerEnd": layer_end,
                "expectedInputPayloadSha256Hex": input_hash if index == 0 else None,
                "expectedOutputPayloadSha256Hex": (
                    expected_output_hash if is_final_shard else None
                ),
                "payloadRole": "final_output" if is_final_shard else "prefill_activation",
            }
            stages.append(stage)
            previous_stage_id = stage_id
            sequence += 1
    else:
        for phase in CAI_OWNED_TRANSPORT_BATCH_PHASES:
            previous_stage_id: str | None = None
            for index, shard_range in enumerate(shard_ranges):
                executor = str(shard_range["nodeId"])
                layer_start = int(shard_range["layerStart"])
                layer_end = int(shard_range["layerEnd"])
                source = (
                    requester if index == 0 else str(shard_ranges[index - 1]["nodeId"])
                )
                output_to = (
                    str(shard_ranges[index + 1]["nodeId"])
                    if index + 1 < len(shard_ranges)
                    else requester
                )
                depends_on_stage_ids = []
                if previous_stage_id:
                    depends_on_stage_ids.append(previous_stage_id)
                elif phase == "decode_activation_batches" and last_prefill_stage_id:
                    depends_on_stage_ids.append(last_prefill_stage_id)
                stage_id = _cai_owned_transport_stage_id(
                    session_id=clean_session_id,
                    phase=phase,
                    sequence=sequence,
                    executor_node_id=executor,
                    layer_start=layer_start,
                    layer_end=layer_end,
                )
                stage = {
                    "stageId": stage_id,
                    "sequence": sequence,
                    "phase": phase,
                    "sourceNodeId": source,
                    "sinkNodeId": executor,
                    "executorNodeId": executor,
                    "outputToNodeId": output_to,
                    "dependsOnStageIds": depends_on_stage_ids,
                    "layerStart": layer_start,
                    "layerEnd": layer_end,
                    "expectedInputPayloadSha256Hex": (
                        input_hash
                        if phase == "prefill_activation_batches" and index == 0
                        else None
                    ),
                    "expectedOutputPayloadSha256Hex": (
                        expected_output_hash
                        if phase == "decode_activation_batches"
                        and index + 1 == len(shard_ranges)
                        else None
                    ),
                    "payloadRole": (
                        "prefill_activation"
                        if phase == "prefill_activation_batches"
                        else "decode_activation"
                    ),
                }
                stages.append(stage)
                previous_stage_id = stage_id
                if phase == "prefill_activation_batches":
                    last_prefill_stage_id = stage_id
                sequence += 1

    dag = {
        "schemaVersion": CAI_OWNED_TRANSPORT_EXECUTION_DAG_SCHEMA_VERSION,
        "protocol": CAI_OWNED_TRANSPORT_PROTOCOL,
        "protocolVersion": CAI_OWNED_TRANSPORT_PROTOCOL_VERSION,
        "network": resolved_chain_id,
        "chainId": resolved_chain_id,
        "sessionId": clean_session_id,
        "taskId": str(task_id or "").strip() or None,
        "modelId": str(model_id or "").strip() or None,
        "requesterNodeId": requester,
        "coordinatorNodeId": coordinator,
        "executorNodeIds": executors,
        "participantNodeIds": participants,
        "totalLayerCount": total_layers,
        "shardRanges": shard_ranges,
        "stages": stages,
        "stageCount": len(stages),
        "singleExecutorDirectFinalOutput": direct_final_output,
        "executionPipelineMode": pipeline_mode,
        "createdAt": created_at or datetime.now(tz=UTC).isoformat(),
    }
    dag["dagHashSha256Hex"] = _cai_owned_transport_dag_hash(dag)
    return dag


def validate_cai_owned_transport_execution_dag(
    dag: dict[str, Any] | None,
    *,
    chain_id: str | None = None,
    session_id: str | None = None,
    participant_node_ids: Sequence[str] | None = None,
) -> tuple[bool, str | None]:
    if not isinstance(dag, dict):
        return False, "CAI-owned transport execution DAG is missing."
    if (
        int(dag.get("schemaVersion") or 0)
        != CAI_OWNED_TRANSPORT_EXECUTION_DAG_SCHEMA_VERSION
    ):
        return False, "CAI-owned transport execution DAG schema is unsupported."
    if str(dag.get("protocol") or "").strip() != CAI_OWNED_TRANSPORT_PROTOCOL:
        return False, "CAI-owned transport execution DAG protocol is invalid."
    if int(dag.get("protocolVersion") or 0) != CAI_OWNED_TRANSPORT_PROTOCOL_VERSION:
        return False, "CAI-owned transport execution DAG protocol version is unsupported."
    chain_valid, chain_error, _dag_chain_id = _validate_cai_owned_transport_chain_id(
        dag,
        expected_chain_id=_cai_owned_transport_chain_id(None, chain_id),
        payload_name="execution DAG",
    )
    if not chain_valid:
        return False, chain_error
    clean_session_id = str(dag.get("sessionId") or "").strip()
    expected_session_id = str(session_id or "").strip()
    if not clean_session_id:
        return False, "CAI-owned transport execution DAG session id is missing."
    if expected_session_id and clean_session_id != expected_session_id:
        return False, "CAI-owned transport execution DAG session id does not match."
    requester = str(dag.get("requesterNodeId") or "").strip()
    coordinator = str(dag.get("coordinatorNodeId") or "").strip()
    executors = _clean_node_ids(dag.get("executorNodeIds") or [])
    participants = _clean_node_ids(dag.get("participantNodeIds") or [])
    expected_participants = _clean_node_ids(participant_node_ids or [])
    if not requester or not coordinator or not executors:
        return False, "CAI-owned transport execution DAG participants are missing."
    if expected_participants and set(participants) != set(expected_participants):
        return False, "CAI-owned transport execution DAG participant set does not match."
    if not set([requester, coordinator, *executors]).issubset(set(participants)):
        return False, "CAI-owned transport execution DAG participant list is incomplete."
    try:
        total_layers = int(dag.get("totalLayerCount") or 0)
    except (TypeError, ValueError):
        return False, "CAI-owned transport execution DAG total layer count is invalid."
    if total_layers <= 0:
        return False, "CAI-owned transport execution DAG total layer count is invalid."
    if len(executors) > total_layers:
        return (
            False,
            "CAI-owned transport execution DAG executor count exceeds total layer count.",
        )

    try:
        shard_ranges = _normalize_cai_owned_transport_shard_ranges(
            executors,
            total_layers,
            shard_ranges=dag.get("shardRanges"),
        )
    except ValueError as exc:
        return False, str(exc)
    if dag.get("shardRanges") != shard_ranges:
        return False, "CAI-owned transport execution DAG shard ranges are not normalized."

    stages = dag.get("stages")
    if not isinstance(stages, list) or not stages:
        return False, "CAI-owned transport execution DAG stages are missing."
    if int(dag.get("stageCount") or 0) != len(stages):
        return False, "CAI-owned transport execution DAG stage count does not match."
    direct_final_output = bool(dag.get("singleExecutorDirectFinalOutput"))
    if direct_final_output and len(executors) != 1:
        return (
            False,
            "CAI-owned transport execution DAG direct final output requires one executor.",
        )
    if direct_final_output:
        expected_stage_count = 1
    elif (
        str(dag.get("executionPipelineMode") or "").strip()
        == CAI_OWNED_TRANSPORT_EXECUTION_PIPELINE_SINGLE_PASS_FINAL_DECODE
    ):
        expected_stage_count = len(executors)
    else:
        expected_stage_count = len(executors) * len(CAI_OWNED_TRANSPORT_BATCH_PHASES)
    if len(stages) != expected_stage_count:
        return False, "CAI-owned transport execution DAG stage count is invalid."

    seen_stage_ids: set[str] = set()
    last_prefill_stage_id: str | None = None
    pipeline_mode = (
        str(dag.get("executionPipelineMode") or "").strip()
        or CAI_OWNED_TRANSPORT_EXECUTION_PIPELINE_FULL_PREFILL_DECODE
    )
    if direct_final_output:
        pipeline_mode = CAI_OWNED_TRANSPORT_EXECUTION_PIPELINE_SINGLE_PASS_FINAL_DECODE
    if pipeline_mode not in {
        CAI_OWNED_TRANSPORT_EXECUTION_PIPELINE_FULL_PREFILL_DECODE,
        CAI_OWNED_TRANSPORT_EXECUTION_PIPELINE_SINGLE_PASS_FINAL_DECODE,
    }:
        return False, "CAI-owned transport execution DAG pipeline mode is invalid."
    for expected_sequence, stage in enumerate(stages):
        if not isinstance(stage, dict):
            return False, "CAI-owned transport execution DAG stage is invalid."
        sequence_value = _optional_int(stage.get("sequence"))
        if sequence_value != expected_sequence:
            return False, "CAI-owned transport execution DAG stage sequence is invalid."
        if direct_final_output:
            shard_index = 0
            expected_phase = "decode_activation_batches"
        elif (
            pipeline_mode
            == CAI_OWNED_TRANSPORT_EXECUTION_PIPELINE_SINGLE_PASS_FINAL_DECODE
        ):
            shard_index = expected_sequence
            expected_phase = (
                "decode_activation_batches"
                if expected_sequence + 1 == len(shard_ranges)
                else "prefill_activation_batches"
            )
        else:
            shard_index = expected_sequence % len(shard_ranges)
            phase_index = expected_sequence // len(shard_ranges)
            expected_phase = CAI_OWNED_TRANSPORT_BATCH_PHASES[phase_index]
        expected_range = shard_ranges[shard_index]
        expected_source = (
            requester
            if shard_index == 0
            else str(shard_ranges[shard_index - 1]["nodeId"])
        )
        expected_output_to = (
            requester
            if shard_index + 1 == len(shard_ranges)
            else str(shard_ranges[shard_index + 1]["nodeId"])
        )
        phase = str(stage.get("phase") or "").strip()
        if phase != expected_phase:
            return False, "CAI-owned transport execution DAG stage phase is unsupported."
        executor = str(stage.get("executorNodeId") or "").strip()
        if executor != str(expected_range["nodeId"]):
            return False, "CAI-owned transport execution DAG stage executor is invalid."
        if str(stage.get("sinkNodeId") or "").strip() != executor:
            return False, "CAI-owned transport execution DAG stage sink is invalid."
        if str(stage.get("sourceNodeId") or "").strip() != expected_source:
            return False, "CAI-owned transport execution DAG stage route is invalid."
        if str(stage.get("outputToNodeId") or "").strip() != expected_output_to:
            return False, "CAI-owned transport execution DAG stage route is invalid."
        layer_start = _optional_int(stage.get("layerStart"))
        layer_end = _optional_int(stage.get("layerEnd"))
        if layer_start is None or layer_end is None or layer_end <= layer_start:
            return False, "CAI-owned transport execution DAG stage layer range is invalid."
        if (
            layer_start != int(expected_range["layerStart"])
            or layer_end != int(expected_range["layerEnd"])
        ):
            return False, "CAI-owned transport execution DAG stage layer range is invalid."
        expected_stage_id = _cai_owned_transport_stage_id(
            session_id=clean_session_id,
            phase=phase,
            sequence=expected_sequence,
            executor_node_id=executor,
            layer_start=layer_start,
            layer_end=layer_end,
        )
        if str(stage.get("stageId") or "").strip() != expected_stage_id:
            return False, "CAI-owned transport execution DAG stage id does not match."
        depends_on_stage_ids = stage.get("dependsOnStageIds") or []
        if not isinstance(depends_on_stage_ids, list):
            return False, "CAI-owned transport execution DAG stage dependencies are invalid."
        expected_dependencies: list[str] = []
        if direct_final_output:
            expected_dependencies = []
        elif (
            pipeline_mode
            == CAI_OWNED_TRANSPORT_EXECUTION_PIPELINE_SINGLE_PASS_FINAL_DECODE
        ):
            if expected_sequence > 0:
                expected_dependencies = [
                    str(stages[expected_sequence - 1].get("stageId") or "")
                ]
        elif shard_index > 0:
            expected_dependencies = [str(stages[expected_sequence - 1].get("stageId") or "")]
        elif phase == "decode_activation_batches" and last_prefill_stage_id:
            expected_dependencies = [last_prefill_stage_id]
        if depends_on_stage_ids != expected_dependencies:
            return (
                False,
                "CAI-owned transport execution DAG stage dependencies are invalid.",
            )
        for dependency in depends_on_stage_ids:
            if str(dependency or "").strip() not in seen_stage_ids:
                return False, "CAI-owned transport execution DAG stage dependency is invalid."
        for hash_field in (
            "expectedInputPayloadSha256Hex",
            "expectedOutputPayloadSha256Hex",
        ):
            if stage.get(hash_field):
                try:
                    _normalize_sha256_hex(stage.get(hash_field), field_name=hash_field)
                except ValueError as exc:
                    return False, str(exc)
        seen_stage_ids.add(expected_stage_id)
        if phase == "prefill_activation_batches" and shard_index + 1 == len(shard_ranges):
            last_prefill_stage_id = expected_stage_id

    expected_hash = _cai_owned_transport_dag_hash(dag)
    if str(dag.get("dagHashSha256Hex") or "").strip().lower() != expected_hash:
        return False, "CAI-owned transport execution DAG hash does not match."
    return True, None


def dispatch_cai_owned_transport_execution_dag(
    *,
    instance_id: str,
    requester_node_id: str,
    executor_node_ids: Sequence[str],
    peer_cai_urls_by_node: Mapping[str, Sequence[str]],
    initial_payload: bytes,
    total_layer_count: int,
    shard_ranges: Sequence[Mapping[str, Any]] | None = None,
    chain_id: str | None = None,
    model_id: str | None = None,
    task_id: str | None = None,
    tokenizer_config_hash: str | None = None,
    llm_runtime_metadata: Mapping[str, Any] | None = None,
    initial_token_count: int | None = None,
    route_policy: dict[str, Any] | None = None,
    payload_compression: str | None = None,
    payload_chunk_size_bytes: int | None = None,
    require_executor_readiness: bool = False,
    require_cai_owned_runtime_ready: bool = False,
    require_executor_shard_readiness: bool = False,
    require_data_plane_route: bool = False,
    require_proven_data_plane_route: bool = False,
    route_health_records: Sequence[Any] | None = None,
    executor_readiness_state_payload: Mapping[str, Any] | None = None,
    timeout_sec: float = 5.0,
    submit_requester_offer: bool = False,
    offer_settle_sec: float = 0.0,
    signing_material: Mapping[str, Any] | None = None,
    policy: WalletPolicy | None = None,
    single_executor_direct_final_output: bool = False,
    execution_pipeline_mode: str | None = None,
) -> dict[str, Any]:
    requester = str(requester_node_id or "").strip()
    executors = _clean_node_ids(executor_node_ids)
    if not requester:
        raise ValueError("CAI-owned dispatch requires requester node id.")
    if not executors:
        raise ValueError("CAI-owned dispatch requires executor nodes.")
    if not isinstance(peer_cai_urls_by_node, Mapping):
        raise ValueError("CAI-owned dispatch peer URL map is required.")
    participants = _clean_node_ids([requester, *executors])
    resolved_chain_id = _cai_owned_transport_chain_id(policy, chain_id)
    require_executor_readiness = bool(
        require_executor_readiness
        or require_cai_owned_runtime_ready
        or require_executor_shard_readiness
    )
    expected_shard_ranges = _normalize_cai_owned_transport_shard_ranges(
        executors,
        total_layer_count,
        shard_ranges=shard_ranges,
    )
    readiness_preflight: dict[str, Any] | None = None
    offer = build_cai_owned_transport_session_offer(
        instance_id=instance_id,
        participant_node_ids=participants,
        executor_node_ids=executors,
        chain_id=resolved_chain_id,
        model_id=model_id,
        task_id=task_id,
        source_node_id=requester,
        route_policy=route_policy,
    )
    dag = build_cai_owned_transport_execution_dag(
        session_id=offer["sessionId"],
        requester_node_id=requester,
        executor_node_ids=executors,
        total_layer_count=total_layer_count,
        shard_ranges=expected_shard_ranges,
        chain_id=resolved_chain_id,
        model_id=model_id,
        task_id=task_id,
        single_executor_direct_final_output=(
            bool(single_executor_direct_final_output) and len(executors) == 1
        ),
        execution_pipeline_mode=execution_pipeline_mode,
    )
    route_policy = dict(offer.get("routePolicy") or {})
    route_policy["executionDag"] = dag
    route_policy["executionDagHashSha256Hex"] = dag.get("dagHashSha256Hex")
    offer["routePolicy"] = route_policy
    signing_kwargs = cai_owned_transport_peer_signing_kwargs(signing_material)
    if signing_kwargs:
        offer = sign_cai_owned_transport_session_offer(
            offer,
            signer_node_id=requester,
            **signing_kwargs,
        )
    route_preflight: dict[str, Any] | None = None
    first_executor = executors[0]
    first_range = dag["shardRanges"][0]
    stages = [
        dict(stage)
        for stage in dag.get("stages") or []
        if isinstance(stage, dict)
    ]
    first_stage_phase = (
        str((stages[0] if stages else {}).get("phase") or "prefill_activation_batches")
        .strip()
        or "prefill_activation_batches"
    )
    first_frame_kind = _cai_owned_transport_frame_kind_for_phase(first_stage_phase)
    resolved_model_id = str(model_id or "").strip() or "unknown"
    resolved_initial_token_count = _optional_int(initial_token_count)
    if resolved_initial_token_count is None:
        resolved_initial_token_count = 0
    resolved_initial_token_count = max(0, resolved_initial_token_count)
    resolved_runtime_metadata = _cai_owned_transport_llm_runtime_metadata(
        llm_runtime_metadata,
        model_id=resolved_model_id,
        total_layer_count=total_layer_count,
        tokenizer_config_hash=tokenizer_config_hash,
    )
    if resolved_runtime_metadata is not None:
        initial_metadata = build_cai_owned_llm_shard_frame_metadata_from_runtime(
            model_id=resolved_model_id,
            runtime_metadata=resolved_runtime_metadata,
            payload=bytes(initial_payload or b""),
            frame_kind=first_frame_kind,
            layer_start=int(first_range["layerStart"]),
            layer_end=int(first_range["layerEnd"]),
            token_start=0,
            token_end=resolved_initial_token_count,
            sequence=0,
        )
    else:
        initial_metadata = build_cai_owned_transport_frame_metadata(
            model_id=resolved_model_id,
            frame_kind=first_frame_kind,
            tokenizer_config_hash=tokenizer_config_hash,
            layer_start=int(first_range["layerStart"]),
            layer_end=int(first_range["layerEnd"]),
            token_start=0,
            token_end=resolved_initial_token_count,
            dtype="bytes",
            shape=[len(bytes(initial_payload or b""))],
            sequence=0,
            payload_sha256_hex=hashlib.sha256(bytes(initial_payload or b"")).hexdigest(),
        )
    if stages:
        initial_metadata["stageId"] = stages[0].get("stageId")
    if bool(dag.get("singleExecutorDirectFinalOutput")):
        initial_metadata["singleExecutorDirectFinalOutput"] = True
    output_route_plan = _cai_owned_transport_output_route_plan_from_dag(
        dag,
        requester_node_id=requester,
    )
    if resolved_runtime_metadata is not None:
        _attach_cai_owned_transport_llm_route_frame_templates(
            output_route_plan,
            model_id=resolved_model_id,
            runtime_metadata=resolved_runtime_metadata,
            initial_token_count=resolved_initial_token_count,
        )
    if output_route_plan:
        initial_metadata["outputRoutePlan"] = output_route_plan
        initial_metadata["nextSinkNodeId"] = output_route_plan[0]["sinkNodeId"]
        initial_metadata["remainingSinkNodeIds"] = [
            str(item.get("sinkNodeId") or "")
            for item in output_route_plan[1:]
            if str(item.get("sinkNodeId") or "")
        ]
        initial_metadata["nextOutputPhase"] = output_route_plan[0].get("phase")
        initial_metadata["nextOutputSequence"] = output_route_plan[0].get("sequence")
        if isinstance(output_route_plan[0].get("frameTemplate"), dict):
            initial_metadata["nextFrameTemplate"] = dict(
                output_route_plan[0].get("frameTemplate") or {}
            )
    initial_metadata["requesterNodeId"] = requester
    initial_metadata["coordinatorNodeId"] = requester
    initial_metadata["peerCaiUrlsByNode"] = {
        node_id: _clean_peer_cai_urls(urls)
        for node_id, urls in peer_cai_urls_by_node.items()
        if _clean_peer_cai_urls(urls)
    }
    initial_envelope = build_cai_owned_transport_batch_envelope(
        session_id=offer["sessionId"],
        phase=first_stage_phase,
        source_node_id=requester,
        sink_node_id=first_executor,
        sequence=0,
        payload=bytes(initial_payload or b""),
        chain_id=resolved_chain_id,
        metadata=initial_metadata,
        payload_compression=payload_compression,
        payload_chunk_size_bytes=payload_chunk_size_bytes,
    )
    if signing_kwargs:
        initial_envelope = sign_cai_owned_transport_batch_envelope(
            initial_envelope,
            signer_node_id=requester,
            **signing_kwargs,
        )
    existing_record = _cai_owned_transport_session_record(offer["sessionId"], policy)
    existing_dispatch = _find_cai_owned_transport_dispatch_record(
        existing_record,
        initial_envelope,
    )
    if _cai_owned_transport_dispatch_record_was_sent(existing_dispatch):
        recovery = _recover_cai_owned_transport_sent_dispatch(
            existing_record,
            session_id=offer["sessionId"],
            requester_node_id=requester,
            participant_node_ids=participants,
            peer_cai_urls_by_node=peer_cai_urls_by_node,
            chain_id=resolved_chain_id,
            timeout_sec=timeout_sec,
            policy=policy,
        )
        recovered_record = _cai_owned_transport_session_record(
            offer["sessionId"],
            policy,
        )
        resume_status = "resumed"
        resume_reason = "initial_batch_already_dispatched"
        if recovery.get("status") in {
            "local_session_already_completed",
            "local_final_output_completed",
        }:
            resume_status = "completed"
            resume_reason = str(recovery.get("status") or resume_reason)
        elif recovery.get("status") in {
            "remote_session_completed",
            "remote_final_state_observed",
        }:
            resume_reason = str(recovery.get("status") or resume_reason)
        return {
            "status": resume_status,
            "resumeReason": resume_reason,
            "sessionId": offer["sessionId"],
            "instanceId": str(instance_id or "").strip(),
            "requesterNodeId": requester,
            "executorNodeIds": executors,
            "participantNodeIds": participants,
            "chainId": resolved_chain_id,
            "offer": offer,
            "dag": dag,
            "localSession": (
                cai_owned_transport_session_to_dict(recovered_record)
                if recovered_record is not None
                else None
            ),
            "offerResponses": {},
            "initialBatchEnvelope": initial_envelope,
            "initialDispatchResponse": (
                existing_dispatch.get("response")
                if isinstance(existing_dispatch, dict)
                else None
            ),
            "dispatchRecord": existing_dispatch,
            "recovery": recovery,
            "readinessPreflight": None,
            "routePreflight": None,
        }

    if require_executor_readiness:
        required_shard_ranges_by_node = (
            {
                str(item["nodeId"]): [item]
                for item in expected_shard_ranges
                if str(item.get("nodeId") or "").strip()
            }
            if require_executor_shard_readiness
            else None
        )
        readiness_preflight = preflight_cai_owned_transport_executor_readiness(
            executor_node_ids=executors,
            peer_cai_urls_by_node=peer_cai_urls_by_node,
            model_id=model_id,
            require_cai_owned_runtime_ready=require_cai_owned_runtime_ready,
            required_shard_ranges_by_node=required_shard_ranges_by_node,
            state_payload=executor_readiness_state_payload,
            timeout_sec=timeout_sec,
        )
        if readiness_preflight["status"] != "ready":
            errors = [
                str(item.get("error") or item.get("reason") or item.get("nodeId"))
                for item in readiness_preflight.get("nodeAudits", [])
                if isinstance(item, dict) and not bool(item.get("ready"))
            ]
            raise ValueError(
                "CAI-owned dispatch readiness preflight failed: "
                + "; ".join(errors)
            )
    if require_data_plane_route or require_proven_data_plane_route:
        if route_health_records is None and require_proven_data_plane_route:
            route_health_records = list_route_health_records(policy)
        route_preflight = preflight_cai_owned_transport_data_plane_routes(
            requester_node_id=requester,
            executor_node_ids=executors,
            peer_cai_urls_by_node=peer_cai_urls_by_node,
            route_policy=route_policy,
            route_health_records=route_health_records,
            require_route_health=require_proven_data_plane_route,
        )
        if route_preflight["status"] != "ready":
            errors = [
                str(item.get("error") or item.get("reason") or item.get("nodeId"))
                for item in route_preflight.get("nodeAudits", [])
                if isinstance(item, dict) and not bool(item.get("ready"))
            ]
            errors.extend(
                str(
                    item.get("error")
                    or item.get("reason")
                    or item.get("sinkNodeId")
                )
                for item in route_preflight.get("routeHealthAudits", [])
                if isinstance(item, dict) and not bool(item.get("ready"))
            )
            errors.extend(
                str(
                    item.get("error")
                    or item.get("reason")
                    or item.get("sinkNodeId")
                )
                for item in route_preflight.get("relayQuorumAudits", [])
                if isinstance(item, dict) and not bool(item.get("ready"))
            )
            errors.extend(str(item) for item in route_preflight.get("fatalReasons", []))
            errors = list(dict.fromkeys(item for item in errors if item))
            raise ValueError(
                "CAI-owned dispatch route preflight failed: "
                + "; ".join(errors)
            )

    local_record = create_cai_owned_transport_session_from_offer(
        offer,
        session_id=offer["sessionId"],
        local_node_id=requester,
        policy=policy,
    )
    local_record = _upsert_cai_owned_transport_dispatch_record(
        offer["sessionId"],
        _cai_owned_transport_initial_dispatch_record(
            initial_envelope,
            status="prepared",
        ),
        policy=policy,
    )
    offer_responses: dict[str, Any] = {}
    if submit_requester_offer:
        requester_peer_urls = _clean_peer_cai_urls(
            peer_cai_urls_by_node.get(requester) or []
        )
        if not requester_peer_urls:
            raise ValueError(
                f"CAI-owned dispatch has no requester peer URL for '{requester}'."
            )
        offer_responses[requester] = submit_cai_owned_transport_session_offer_to_any(
            requester_peer_urls,
            offer,
            chain_id=resolved_chain_id,
            timeout_sec=timeout_sec,
        )
    for node_id in executors:
        peer_urls = _clean_peer_cai_urls(peer_cai_urls_by_node.get(node_id) or [])
        if not peer_urls:
            raise ValueError(f"CAI-owned dispatch has no peer URL for '{node_id}'.")
        offer_responses[node_id] = submit_cai_owned_transport_session_offer_to_any(
            peer_urls,
            offer,
            chain_id=resolved_chain_id,
            timeout_sec=timeout_sec,
        )
    if offer_settle_sec and float(offer_settle_sec or 0.0) > 0:
        time.sleep(float(offer_settle_sec or 0.0))
    first_peer_urls = _clean_peer_cai_urls(peer_cai_urls_by_node.get(first_executor) or [])
    dispatch_response = submit_cai_owned_transport_batch_envelope_to_any(
        first_peer_urls,
        offer["sessionId"],
        initial_envelope,
        chain_id=resolved_chain_id,
        timeout_sec=timeout_sec,
    )
    local_record = _upsert_cai_owned_transport_dispatch_record(
        offer["sessionId"],
        _cai_owned_transport_initial_dispatch_record(
            initial_envelope,
            status="sent",
            response=dispatch_response,
        ),
        policy=policy,
    )
    return {
        "status": "dispatched",
        "sessionId": offer["sessionId"],
        "instanceId": str(instance_id or "").strip(),
        "requesterNodeId": requester,
        "executorNodeIds": executors,
        "participantNodeIds": participants,
        "chainId": resolved_chain_id,
        "offer": offer,
        "dag": dag,
        "localSession": cai_owned_transport_session_to_dict(local_record),
        "offerResponses": offer_responses,
        "initialBatchEnvelope": initial_envelope,
        "initialDispatchResponse": dispatch_response,
        "dispatchRecord": _find_cai_owned_transport_dispatch_record(
            local_record,
            initial_envelope,
        ),
        "readinessPreflight": readiness_preflight,
        "routePreflight": route_preflight,
    }


def _recover_cai_owned_transport_sent_dispatch(
    existing_record: CaiOwnedTransportSessionRecord | None,
    *,
    session_id: str,
    requester_node_id: str,
    participant_node_ids: Sequence[str],
    peer_cai_urls_by_node: Mapping[str, Sequence[str]],
    chain_id: str | None,
    timeout_sec: float,
    policy: WalletPolicy | None,
) -> dict[str, Any]:
    clean_session_id = str(session_id or "").strip()
    requester = str(requester_node_id or "").strip()
    if existing_record is not None and existing_record.status == "completed":
        return {
            "status": "local_session_already_completed",
            "sessionId": clean_session_id,
            "checkedRemote": False,
            "localSession": cai_owned_transport_session_to_dict(existing_record),
        }

    local_final_output = latest_cai_owned_transport_final_output(
        clean_session_id,
        requester_node_id=requester,
        policy=policy,
    )
    if local_final_output is not None:
        try:
            completed = complete_cai_owned_transport_session(
                clean_session_id,
                policy=policy,
            )
            return {
                "status": "local_final_output_completed",
                "sessionId": clean_session_id,
                "checkedRemote": False,
                "finalOutput": _cai_owned_transport_compact_final_output(
                    local_final_output
                ),
                "localSession": cai_owned_transport_session_to_dict(completed),
            }
        except Exception as exc:
            local_error = str(exc)
    else:
        local_error = None

    remote_audits = _fetch_cai_owned_transport_remote_session_audits(
        clean_session_id,
        participant_node_ids=participant_node_ids,
        peer_cai_urls_by_node=peer_cai_urls_by_node,
        chain_id=chain_id,
        timeout_sec=timeout_sec,
    )
    remote_completed = [
        item
        for item in remote_audits
        if item.get("status") == "found"
        and str(item.get("sessionStatus") or "").strip() == "completed"
    ]
    remote_final = [
        item
        for item in remote_audits
        if item.get("status") == "found"
        and (
            bool(item.get("hasProof"))
            or int(item.get("finalOutputBatchCount") or 0) > 0
            or int(item.get("shardReceiptCount") or 0) > 0
        )
    ]
    if remote_completed:
        return {
            "status": "remote_session_completed",
            "sessionId": clean_session_id,
            "checkedRemote": True,
            "localCompletionError": local_error,
            "remoteSessions": remote_audits,
        }
    if remote_final:
        return {
            "status": "remote_final_state_observed",
            "sessionId": clean_session_id,
            "checkedRemote": True,
            "localCompletionError": local_error,
            "remoteSessions": remote_audits,
        }
    return {
        "status": "sent_dispatch_observed",
        "sessionId": clean_session_id,
        "checkedRemote": bool(remote_audits),
        "localCompletionError": local_error,
        "remoteSessions": remote_audits,
    }


def _fetch_cai_owned_transport_remote_session_audits(
    session_id: str,
    *,
    participant_node_ids: Sequence[str],
    peer_cai_urls_by_node: Mapping[str, Sequence[str]],
    chain_id: str | None,
    timeout_sec: float,
) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for node_id in _clean_node_ids(participant_node_ids):
        for peer_url in _prioritized_cai_owned_transport_peer_urls(
            peer_cai_urls_by_node.get(node_id) or []
        ):
            if str(peer_url or "").strip().startswith(
                CAI_OWNED_TRANSPORT_OVERLAY_URL_PREFIX
            ):
                continue
            base_url = str(peer_url or "").strip().rstrip("/")
            if not base_url:
                continue
            url = f"{base_url}/v1/cai/transport/sessions"
            if url in seen_urls:
                continue
            seen_urls.add(url)
            audits.append(
                _fetch_cai_owned_transport_remote_session_audit(
                    url,
                    session_id,
                    node_id=node_id,
                    chain_id=chain_id,
                    timeout_sec=timeout_sec,
                )
            )
    return audits


def _fetch_cai_owned_transport_remote_session_audit(
    sessions_url: str,
    session_id: str,
    *,
    node_id: str,
    chain_id: str | None,
    timeout_sec: float,
) -> dict[str, Any]:
    try:
        request = Request(
            sessions_url,
            headers=_cai_owned_transport_json_headers(chain_id=chain_id),
            method="GET",
        )
        with urlopen(request, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8")
        payload = json.loads(raw or "{}")
        session = _find_cai_owned_transport_session_payload(payload, session_id)
        if session is None:
            return {
                "nodeId": str(node_id or "").strip(),
                "peerCaiUrl": sessions_url,
                "status": "not_found",
            }
        return {
            "nodeId": str(node_id or "").strip(),
            "peerCaiUrl": sessions_url,
            "status": "found",
            "sessionStatus": str(session.get("status") or "").strip() or None,
            "hasProof": isinstance(session.get("proof"), Mapping),
            "batchRecordCount": len(session.get("batchRecords") or [])
            if isinstance(session.get("batchRecords"), list)
            else 0,
            "shardReceiptCount": len(session.get("shardReceipts") or [])
            if isinstance(session.get("shardReceipts"), list)
            else 0,
            "finalOutputBatchCount": _remote_session_final_output_batch_count(
                session
            ),
            "completedAt": session.get("completedAt"),
            "lastError": session.get("lastError"),
        }
    except Exception as exc:
        return {
            "nodeId": str(node_id or "").strip(),
            "peerCaiUrl": sessions_url,
            "status": "failed",
            "error": str(exc),
        }


def _find_cai_owned_transport_session_payload(
    payload: Any,
    session_id: str,
) -> dict[str, Any] | None:
    if isinstance(payload, Mapping):
        sessions = payload.get("sessions")
    else:
        sessions = payload
    if not isinstance(sessions, Sequence) or isinstance(sessions, (str, bytes)):
        return None
    clean_session_id = str(session_id or "").strip()
    for item in sessions:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("sessionId") or "").strip() == clean_session_id:
            return dict(item)
    return None


def _remote_session_final_output_batch_count(session: Mapping[str, Any]) -> int:
    batch_records = session.get("batchRecords")
    if not isinstance(batch_records, list):
        return 0
    count = 0
    for item in batch_records:
        if not isinstance(item, Mapping):
            continue
        metadata = item.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        if bool(item.get("finalOutput")) or bool(metadata.get("finalOutput")):
            count += 1
    return count


def _cai_owned_transport_compact_final_output(
    final_output: Mapping[str, Any],
) -> dict[str, Any]:
    batch = final_output.get("batch")
    batch = batch if isinstance(batch, Mapping) else {}
    return {
        "batchId": final_output.get("batchId"),
        "sourceNodeId": final_output.get("sourceNodeId"),
        "sinkNodeId": final_output.get("sinkNodeId"),
        "payloadSizeBytes": final_output.get("payloadSizeBytes"),
        "payloadSha256Hex": final_output.get("payloadSha256Hex"),
        "batchStatus": batch.get("status"),
    }


def preflight_cai_owned_transport_executor_readiness(
    *,
    executor_node_ids: Sequence[str],
    peer_cai_urls_by_node: Mapping[str, Sequence[str]],
    model_id: str | None = None,
    require_cai_owned_runtime_ready: bool = False,
    required_shard_ranges_by_node: Mapping[
        str,
        Sequence[Mapping[str, Any]],
    ]
    | None = None,
    minimum_ram_headroom_bytes: int | None = None,
    minimum_vram_headroom_bytes: int | None = None,
    state_payload: Mapping[str, Any] | None = None,
    timeout_sec: float = 5.0,
) -> dict[str, Any]:
    executors = _clean_node_ids(executor_node_ids)
    resource_requirements = _cai_owned_transport_resource_requirements(
        model_id,
        minimum_ram_headroom_bytes=minimum_ram_headroom_bytes,
        minimum_vram_headroom_bytes=minimum_vram_headroom_bytes,
    )
    node_audits = [
        _preflight_cai_owned_transport_executor_node_readiness(
            node_id,
            _prioritized_cai_owned_transport_peer_urls(
                peer_cai_urls_by_node.get(node_id) or []
            ),
            model_id=model_id,
            require_cai_owned_runtime_ready=require_cai_owned_runtime_ready,
            required_shard_ranges=(
                (required_shard_ranges_by_node or {}).get(node_id) or []
            ),
            minimum_ram_headroom_bytes=resource_requirements["minimumRamHeadroomBytes"],
            minimum_vram_headroom_bytes=resource_requirements[
                "minimumVramHeadroomBytes"
            ],
            state_payload=state_payload,
            timeout_sec=timeout_sec,
        )
        for node_id in executors
    ]
    ready = bool(node_audits) and all(
        bool(item.get("ready")) for item in node_audits
    )
    return {
        "status": "ready" if ready else "failed",
        "executorNodeIds": executors,
        "modelId": str(model_id or "").strip() or None,
        "requireCaiOwnedRuntimeReady": bool(require_cai_owned_runtime_ready),
        "requireShardReadiness": bool(required_shard_ranges_by_node),
        **resource_requirements,
        "nodeAudits": node_audits,
    }


def _preflight_cai_owned_transport_executor_node_readiness(
    node_id: str,
    peer_cai_urls: Sequence[str],
    *,
    model_id: str | None,
    require_cai_owned_runtime_ready: bool,
    required_shard_ranges: Sequence[Mapping[str, Any]],
    minimum_ram_headroom_bytes: int,
    minimum_vram_headroom_bytes: int,
    state_payload: Mapping[str, Any] | None,
    timeout_sec: float,
) -> dict[str, Any]:
    clean_node_id = str(node_id or "").strip()
    prioritized_urls = _prioritized_cai_owned_transport_peer_urls(peer_cai_urls)
    overlay_urls = [
        url
        for url in prioritized_urls
        if str(url or "").strip().startswith(CAI_OWNED_TRANSPORT_OVERLAY_URL_PREFIX)
    ]
    direct_urls = [
        url
        for url in prioritized_urls
        if not str(url or "").strip().startswith(
            CAI_OWNED_TRANSPORT_OVERLAY_URL_PREFIX
        )
    ]
    overlay_fallback_summary = _cai_owned_transport_executor_summary_from_state(
        clean_node_id,
        state_payload,
    )
    can_use_overlay_state_fallback = (
        bool(overlay_urls)
        and overlay_fallback_summary is not None
        and not require_cai_owned_runtime_ready
        and not required_shard_ranges
    )
    if not direct_urls:
        if can_use_overlay_state_fallback:
            return _evaluate_cai_owned_transport_executor_state_fallback(
                clean_node_id,
                overlay_urls,
                overlay_fallback_summary,
                model_id=model_id,
                require_cai_owned_runtime_ready=require_cai_owned_runtime_ready,
                required_shard_ranges=required_shard_ranges,
                minimum_ram_headroom_bytes=minimum_ram_headroom_bytes,
                minimum_vram_headroom_bytes=minimum_vram_headroom_bytes,
                attempts=[],
            )
        return {
            "nodeId": clean_node_id,
            "ready": False,
            "reason": "no_direct_summary_url",
            "error": "No direct CAI summary URL is available for executor.",
            "checkedUrls": [],
            "overlayUrls": list(overlay_urls),
        }
    attempts: list[dict[str, Any]] = []
    last_error = ""
    last_audit: dict[str, Any] | None = None
    for base_url in direct_urls:
        summary_url = f"{base_url.rstrip('/')}/v1/cai/summary"
        try:
            request = Request(summary_url, method="GET")
            with urlopen(request, timeout=timeout_sec) as response:
                raw = response.read().decode("utf-8")
            payload = json.loads(raw or "{}")
            if not isinstance(payload, dict):
                payload = {"value": payload}
            capability_readiness_attempt: dict[str, Any] | None = None
            if require_cai_owned_runtime_ready:
                payload, capability_readiness_attempt = (
                    _cai_owned_transport_summary_with_node_capability_readiness(
                        clean_node_id,
                        base_url,
                        payload,
                        timeout_sec=timeout_sec,
                    )
                )
            audit = _evaluate_cai_owned_transport_executor_summary(
                clean_node_id,
                summary_url,
                payload,
                model_id=model_id,
                require_cai_owned_runtime_ready=require_cai_owned_runtime_ready,
                required_shard_ranges=required_shard_ranges,
                minimum_ram_headroom_bytes=minimum_ram_headroom_bytes,
                minimum_vram_headroom_bytes=minimum_vram_headroom_bytes,
            )
            attempts.append(
                {
                    "url": summary_url,
                    "status": "ok" if audit["ready"] else "not_ready",
                    "reason": audit.get("reason"),
                }
            )
            if capability_readiness_attempt is not None:
                attempts.append(capability_readiness_attempt)
                audit["nodeCapabilityReadinessAttempt"] = dict(
                    capability_readiness_attempt
                )
            audit["checkedUrls"] = [item["url"] for item in attempts]
            if audit["ready"]:
                return audit
            last_audit = audit
            last_error = str(audit.get("error") or audit.get("reason") or "")
        except Exception as exc:
            last_error = str(exc)
            attempts.append(
                {
                    "url": summary_url,
                    "status": "failed",
                    "error": last_error,
                }
            )
    if last_audit is not None:
        last_audit["checkedUrls"] = [item["url"] for item in attempts]
        last_audit["attempts"] = attempts
        return last_audit
    if can_use_overlay_state_fallback:
        return _evaluate_cai_owned_transport_executor_state_fallback(
            clean_node_id,
            overlay_urls,
            overlay_fallback_summary,
            model_id=model_id,
            require_cai_owned_runtime_ready=require_cai_owned_runtime_ready,
            required_shard_ranges=required_shard_ranges,
            minimum_ram_headroom_bytes=minimum_ram_headroom_bytes,
            minimum_vram_headroom_bytes=minimum_vram_headroom_bytes,
            attempts=attempts,
        )
    return {
        "nodeId": clean_node_id,
        "ready": False,
        "reason": "summary_unavailable",
        "error": last_error or "Executor summary is unavailable.",
        "checkedUrls": [item["url"] for item in attempts],
        "overlayUrls": list(overlay_urls),
        "attempts": attempts,
    }


def _evaluate_cai_owned_transport_executor_state_fallback(
    node_id: str,
    overlay_urls: Sequence[str],
    summary: Mapping[str, Any],
    *,
    model_id: str | None,
    require_cai_owned_runtime_ready: bool,
    required_shard_ranges: Sequence[Mapping[str, Any]],
    minimum_ram_headroom_bytes: int,
    minimum_vram_headroom_bytes: int,
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    audit = _evaluate_cai_owned_transport_executor_summary(
        node_id,
        f"state://{node_id}/v1/cai/summary",
        summary,
        model_id=model_id,
        require_cai_owned_runtime_ready=require_cai_owned_runtime_ready,
        required_shard_ranges=required_shard_ranges,
        minimum_ram_headroom_bytes=minimum_ram_headroom_bytes,
        minimum_vram_headroom_bytes=minimum_vram_headroom_bytes,
    )
    overlay_attempts = [
        {
            "url": str(url),
            "status": "deferred_to_dispatch" if audit["ready"] else "not_ready",
            "reason": audit.get("reason"),
            "routeClass": _cai_owned_transport_peer_url_route_class(str(url)),
        }
        for url in overlay_urls
    ]
    all_attempts = [dict(item) for item in attempts] + overlay_attempts
    audit["summarySource"] = "state_payload"
    audit["overlayPreflightDeferredToDispatch"] = bool(audit["ready"])
    audit["overlayUrls"] = [str(url) for url in overlay_urls]
    audit["checkedUrls"] = [
        str(item.get("url") or "")
        for item in all_attempts
        if str(item.get("url") or "").strip()
    ]
    audit["attempts"] = all_attempts
    return audit


def _cai_owned_transport_executor_summary_from_state(
    node_id: str,
    state_payload: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(state_payload, Mapping):
        return None
    identities = state_payload.get("nodeIdentities")
    if not isinstance(identities, Mapping):
        identities = state_payload.get("node_identities")
    if not isinstance(identities, Mapping):
        return None
    identity = identities.get(str(node_id or "").strip())
    if not isinstance(identity, Mapping):
        return None

    worker: dict[str, Any] = {}
    for source_key, target_key in (
        ("workerEnabled", "workerEnabled"),
        ("worker_enabled", "workerEnabled"),
        ("workerRewardAddress", "workerRewardAddress"),
        ("worker_reward_address", "workerRewardAddress"),
        ("workerAllowedModelIds", "allowedModelIds"),
        ("worker_allowed_model_ids", "allowedModelIds"),
        ("allowedModelIds", "allowedModelIds"),
        ("allowed_model_ids", "allowedModelIds"),
    ):
        if source_key in identity and target_key not in worker:
            worker[target_key] = identity.get(source_key)

    readiness = identity.get("readiness")
    if isinstance(readiness, Mapping):
        worker["readiness"] = dict(readiness)
    elif isinstance(identity.get("caiOwnedTransport"), Mapping):
        worker["readiness"] = {
            "caiOwnedTransport": dict(identity.get("caiOwnedTransport") or {})
        }

    resource_summary: dict[str, Any] = {}
    node_memory = state_payload.get("nodeMemory")
    if not isinstance(node_memory, Mapping):
        node_memory = state_payload.get("node_memory")
    memory = node_memory.get(str(node_id or "").strip()) if isinstance(node_memory, Mapping) else None
    if isinstance(memory, Mapping):
        if "ramAvailable" in memory:
            resource_summary["ramAvailable"] = memory.get("ramAvailable")
        if "ram_available" in memory:
            resource_summary["ramAvailable"] = memory.get("ram_available")
        if "ramTotal" in memory:
            resource_summary["ramTotal"] = memory.get("ramTotal")
        if "ram_total" in memory:
            resource_summary["ramTotal"] = memory.get("ram_total")
    if "totalVramBytes" in identity:
        resource_summary["vramBytes"] = identity.get("totalVramBytes")
    if "total_vram_bytes" in identity:
        resource_summary["vramBytes"] = identity.get("total_vram_bytes")

    summary: dict[str, Any] = {"worker": worker}
    if resource_summary:
        summary["resourceSummary"] = dict(resource_summary)
        worker["resourceSummary"] = dict(resource_summary)
    return summary


def _cai_owned_transport_summary_with_node_capability_readiness(
    node_id: str,
    base_url: str,
    summary: Mapping[str, Any],
    *,
    timeout_sec: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    summary_payload = dict(summary)
    worker = (
        summary_payload.get("worker")
        if isinstance(summary_payload.get("worker"), Mapping)
        else {}
    )
    if _summary_cai_owned_transport_readiness(summary_payload, worker):
        return summary_payload, None

    capabilities_url = f"{str(base_url or '').rstrip('/')}/v1/cai/node-capabilities"
    attempt: dict[str, Any] = {
        "url": capabilities_url,
        "status": "failed",
        "kind": "node_capabilities_readiness",
    }
    try:
        request = Request(capabilities_url, method="GET")
        with urlopen(request, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8")
        payload = json.loads(raw or "{}")
    except Exception as exc:  # noqa: BLE001
        attempt["error"] = str(exc)
        return summary_payload, attempt

    record = _cai_owned_transport_node_capability_record(payload, node_id)
    if record is None:
        attempt.update(
            {
                "status": "not_found",
                "reason": "node_capability_missing",
            }
        )
        return summary_payload, attempt

    augmented = _cai_owned_transport_summary_from_node_capability_record(
        summary_payload,
        record,
    )
    augmented_worker = (
        augmented.get("worker") if isinstance(augmented.get("worker"), Mapping) else {}
    )
    readiness = _summary_cai_owned_transport_readiness(augmented, augmented_worker)
    attempt.update(
        {
            "status": "ok" if readiness else "not_ready",
            "reason": "readiness_attached" if readiness else "readiness_missing",
        }
    )
    return augmented, attempt


def _cai_owned_transport_node_capability_record(
    payload: Any,
    node_id: str,
) -> Mapping[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    records = payload.get("records")
    if records is None:
        records = payload.get("nodeCapabilities")
    if records is None:
        records = payload.get("node_capabilities")
    if records is None:
        records = payload.get("items")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return None

    clean_node_id = str(node_id or "").strip()
    for item in records:
        if not isinstance(item, Mapping):
            continue
        candidate_node_id = str(
            item.get("nodeId") or item.get("node_id") or item.get("id") or ""
        ).strip()
        if candidate_node_id == clean_node_id:
            return item
    return None


def _cai_owned_transport_summary_from_node_capability_record(
    summary: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    augmented = dict(summary)
    worker = (
        dict(augmented.get("worker"))
        if isinstance(augmented.get("worker"), Mapping)
        else {}
    )

    for source_key, target_key in (
        ("workerEnabled", "workerEnabled"),
        ("worker_enabled", "workerEnabled"),
        ("workerRewardAddress", "workerRewardAddress"),
        ("worker_reward_address", "workerRewardAddress"),
        ("workerAllowedModelIds", "allowedModelIds"),
        ("worker_allowed_model_ids", "allowedModelIds"),
        ("allowedModelIds", "allowedModelIds"),
        ("allowed_model_ids", "allowedModelIds"),
    ):
        if source_key in record and target_key not in worker:
            worker[target_key] = record.get(source_key)

    readiness = record.get("readiness")
    if isinstance(readiness, Mapping):
        worker["readiness"] = dict(readiness)
        cai_owned_transport = readiness.get("caiOwnedTransport")
        if not isinstance(cai_owned_transport, Mapping):
            cai_owned_transport = readiness.get("cai_owned_transport")
        if isinstance(cai_owned_transport, Mapping):
            worker["caiOwnedTransport"] = dict(cai_owned_transport)
            worker["cai_owned_transport"] = dict(cai_owned_transport)
    elif isinstance(record.get("caiOwnedTransport"), Mapping):
        cai_owned_transport = dict(record.get("caiOwnedTransport") or {})
        worker["readiness"] = {"caiOwnedTransport": cai_owned_transport}
        worker["caiOwnedTransport"] = cai_owned_transport
        worker["cai_owned_transport"] = dict(cai_owned_transport)

    for source_key, target_key in (
        ("resources", "resources"),
        ("resourceSummary", "resourceSummary"),
        ("resource_summary", "resourceSummary"),
        ("modelShardInventory", "modelShardInventory"),
        ("model_shard_inventory", "modelShardInventory"),
    ):
        value = record.get(source_key)
        if isinstance(value, Mapping):
            worker.setdefault(target_key, dict(value))
            if target_key == "resourceSummary":
                worker.setdefault("resources", dict(value))
                augmented.setdefault("resourceSummary", dict(value))
            if target_key == "modelShardInventory":
                worker.setdefault("model_shard_inventory", dict(value))

    augmented["worker"] = worker
    return augmented


def _evaluate_cai_owned_transport_executor_summary(
    node_id: str,
    summary_url: str,
    summary: Mapping[str, Any],
    *,
    model_id: str | None,
    require_cai_owned_runtime_ready: bool,
    required_shard_ranges: Sequence[Mapping[str, Any]],
    minimum_ram_headroom_bytes: int,
    minimum_vram_headroom_bytes: int,
) -> dict[str, Any]:
    worker = summary.get("worker") if isinstance(summary.get("worker"), Mapping) else {}
    worker_enabled = _summary_bool(
        worker,
        "worker_enabled",
        "workerEnabled",
        "enabled",
    )
    if worker_enabled is not True:
        return {
            "nodeId": node_id,
            "ready": False,
            "summaryUrl": summary_url,
            "reason": "worker_disabled",
            "error": "Executor worker mode is disabled or missing.",
        }
    requested_model = str(model_id or "").strip()
    allowed_model_ids = _summary_string_list(
        worker,
        "allowed_model_ids",
        "allowedModelIds",
        "workerAllowedModelIds",
    )
    network_default_model_id = str(
        worker.get("network_default_model_id")
        or worker.get("networkDefaultModelId")
        or ""
    ).strip()
    if requested_model and allowed_model_ids:
        allowed = requested_model in allowed_model_ids
        default_allowed = requested_model == network_default_model_id
        if not (allowed or default_allowed):
            return {
                "nodeId": node_id,
                "ready": False,
                "summaryUrl": summary_url,
                "reason": "model_not_allowed",
                "error": f"Executor does not allow model {requested_model}.",
                "allowedModelIds": allowed_model_ids,
            }
    resource_audit = _summary_resource_headroom_audit(
        summary,
        worker,
        minimum_ram_headroom_bytes=minimum_ram_headroom_bytes,
        minimum_vram_headroom_bytes=minimum_vram_headroom_bytes,
    )
    if not bool(resource_audit.get("ready")):
        return {
            "nodeId": node_id,
            "ready": False,
            "summaryUrl": summary_url,
            "reason": resource_audit.get("reason") or "resource_headroom_insufficient",
            "error": resource_audit.get("error")
            or "Executor does not have enough RAM/VRAM headroom.",
            "resourceAudit": resource_audit,
        }
    cai_readiness = _summary_cai_owned_transport_readiness(summary, worker)
    cai_status = (
        str(cai_readiness.get("status") or "").strip().lower()
        if cai_readiness
        else ""
    )
    cai_runtime_ready = (
        _summary_bool(cai_readiness, "runtimeReady", "runtime_ready")
        if cai_readiness
        else None
    )
    cai_implemented = (
        _summary_bool(cai_readiness, "implemented")
        if cai_readiness
        else None
    )
    version_required = bool(
        cai_readiness
        and (
            require_cai_owned_runtime_ready
            or cai_runtime_ready is True
            or cai_implemented is True
            or cai_status in {"ready", "test_adapter_ready"}
        )
    )
    version_compatibility = (
        cai_owned_transport_version_compatibility(
            cai_readiness,
            require_runtime_versions=bool(
                require_cai_owned_runtime_ready or cai_runtime_ready is True
            ),
            require_protocol_version=version_required,
        )
        if cai_readiness
        else None
    )
    if version_required and not bool(version_compatibility["compatible"]):
        return {
            "nodeId": node_id,
            "ready": False,
            "summaryUrl": summary_url,
            "reason": "runtime_version_incompatible",
            "error": "; ".join(version_compatibility.get("errors") or [])
            or "Executor CAI-owned transport version is incompatible.",
            "caiOwnedTransport": dict(cai_readiness),
            "versionCompatibility": version_compatibility,
        }
    if require_cai_owned_runtime_ready:
        if not cai_readiness:
            return {
                "nodeId": node_id,
                "ready": False,
                "summaryUrl": summary_url,
                "reason": "runtime_readiness_missing",
                "error": "Executor CAI-owned transport readiness is missing.",
            }
        production_adapter_ready = _cai_owned_transport_production_adapter_ready(
            cai_readiness,
        )
        if (
            cai_runtime_ready is not True
            and cai_status != "ready"
            and not production_adapter_ready
        ):
            return {
                "nodeId": node_id,
                "ready": False,
                "summaryUrl": summary_url,
                "reason": "runtime_not_ready",
                "error": "Executor CAI-owned transport runtime is not ready.",
                "caiOwnedTransport": dict(cai_readiness),
            }
    shard_readiness = None
    if requested_model and required_shard_ranges:
        shard_readiness = _summary_model_shard_readiness_audit(
            summary,
            worker,
            model_id=requested_model,
            required_shard_ranges=required_shard_ranges,
        )
        if not bool(shard_readiness.get("ready")):
            return {
                "nodeId": node_id,
                "ready": False,
                "summaryUrl": summary_url,
                "reason": shard_readiness.get("reason") or "model_shard_not_ready",
                "error": shard_readiness.get("error")
                or "Executor model shard inventory is not ready.",
                "modelShardReadiness": shard_readiness,
            }
    return {
        "nodeId": node_id,
        "ready": True,
        "summaryUrl": summary_url,
        "reason": "ready",
        "workerEnabled": True,
        "allowedModelIds": allowed_model_ids,
        "resourceAudit": resource_audit,
        "caiOwnedTransport": dict(cai_readiness) if cai_readiness else None,
        "versionCompatibility": version_compatibility,
        "modelShardReadiness": shard_readiness,
    }


def _cai_owned_transport_production_adapter_ready(
    cai_readiness: Mapping[str, Any],
) -> bool:
    status = str(cai_readiness.get("status") or "").strip().lower()
    if status not in {"ready", "test_adapter_ready"}:
        return False
    if cai_readiness.get("implemented") is not True:
        return False
    self_test = cai_readiness.get("llmShardSelfTest")
    if not isinstance(self_test, Mapping):
        return False
    if self_test.get("backendHealthReady") is False:
        return False
    return bool(
        self_test.get("productionReady")
        and self_test.get("generationProbeReady") is True
        and self_test.get("patchBoundaryVerified") is not False
    )


def _cai_owned_transport_resource_requirements(
    model_id: str | None,
    *,
    minimum_ram_headroom_bytes: int | None,
    minimum_vram_headroom_bytes: int | None,
) -> dict[str, int]:
    policy = NetworkModelPolicy()
    private_model = bool(
        str(model_id or "").strip()
        and is_private_curated_model_id(str(model_id or "").strip(), policy)
    )
    default_ram_bytes = (
        max(0, int(policy.minimum_worker_ram_headroom_mb)) * 1024 * 1024
        if private_model
        else 0
    )
    default_vram_bytes = (
        max(0, int(getattr(policy, "minimum_worker_vram_headroom_mb", 0))) * 1024 * 1024
        if private_model
        else 0
    )
    return {
        "minimumRamHeadroomBytes": _non_negative_int_or_default(
            minimum_ram_headroom_bytes,
            default_ram_bytes,
        ),
        "minimumVramHeadroomBytes": _non_negative_int_or_default(
            minimum_vram_headroom_bytes,
            default_vram_bytes,
        ),
    }


def _summary_model_shard_readiness_audit(
    summary: Mapping[str, Any],
    worker: Mapping[str, Any],
    *,
    model_id: str,
    required_shard_ranges: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required_ranges = _summary_layer_ranges(required_shard_ranges, assume_ready=True)
    if not required_ranges:
        return {
            "ready": True,
            "reason": "no_required_ranges",
            "requiredRanges": [],
        }

    candidates = _summary_model_readiness_candidates(summary, worker, model_id)
    if not candidates:
        return {
            "ready": False,
            "reason": "model_shard_inventory_missing",
            "error": "Executor summary does not advertise model shard inventory.",
            "modelId": model_id,
            "requiredRanges": required_ranges,
        }

    fallback_audit: dict[str, Any] | None = None
    for candidate in candidates:
        audit = _summary_model_candidate_shard_readiness(
            candidate,
            model_id=model_id,
            required_ranges=required_ranges,
        )
        if audit["ready"]:
            return audit
        fallback_audit = audit
    return fallback_audit or {
        "ready": False,
        "reason": "model_shard_inventory_missing",
        "error": "Executor summary does not advertise usable model shard inventory.",
        "modelId": model_id,
        "requiredRanges": required_ranges,
    }


def _summary_model_readiness_candidates(
    summary: Mapping[str, Any],
    worker: Mapping[str, Any],
    model_id: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for source_name, payload in (
        ("worker", worker),
        ("summary", summary),
    ):
        candidates.extend(
            _summary_model_candidates_from_payload(payload, model_id, source_name)
        )
    return candidates


def _summary_model_candidates_from_payload(
    payload: Mapping[str, Any],
    model_id: str,
    source_name: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for field_name in (
        "modelReadiness",
        "model_readiness",
        "modelShardInventory",
        "model_shard_inventory",
        "shardInventory",
        "shard_inventory",
        "models",
    ):
        value = payload.get(field_name)
        candidates.extend(
            _summary_model_candidates_from_value(
                value,
                model_id=model_id,
                source=f"{source_name}.{field_name}",
            )
        )

    readiness = payload.get("readiness")
    if isinstance(readiness, Mapping):
        models = readiness.get("models") or readiness.get("modelReadiness")
        candidates.extend(
            _summary_model_candidates_from_value(
                models,
                model_id=model_id,
                source=f"{source_name}.readiness.models",
            )
        )

    for field_name in (
        "loadedModelIds",
        "loaded_model_ids",
        "downloadedModelIds",
        "downloaded_model_ids",
        "cachedModelIds",
        "cached_model_ids",
    ):
        if _summary_model_id_in_list(payload.get(field_name), model_id):
            candidates.append(
                {
                    "modelId": model_id,
                    "loaded": True,
                    "source": f"{source_name}.{field_name}",
                }
            )
    return candidates


def _summary_model_candidates_from_value(
    value: Any,
    *,
    model_id: str,
    source: str,
) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        if _summary_payload_model_matches(value, model_id):
            candidate = dict(value)
            candidate.setdefault("source", source)
            return [candidate]
        matches: list[dict[str, Any]] = []
        for key, item in value.items():
            if not _summary_model_id_matches(str(key), model_id):
                continue
            candidate = dict(item) if isinstance(item, Mapping) else {"status": item}
            candidate.setdefault("modelId", str(key))
            candidate.setdefault("source", f"{source}.{key}")
            matches.append(candidate)
        return matches
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        matches = []
        for item in value:
            if isinstance(item, Mapping) and _summary_payload_model_matches(
                item,
                model_id,
            ):
                candidate = dict(item)
                candidate.setdefault("source", source)
                matches.append(candidate)
        return matches
    return []


def _summary_model_candidate_shard_readiness(
    candidate: Mapping[str, Any],
    *,
    model_id: str,
    required_ranges: list[dict[str, int]],
) -> dict[str, Any]:
    source = str(candidate.get("source") or "").strip() or None
    blocked_ranges = _summary_blocked_ranges_from_candidate(candidate)
    candidate_block_reason = _summary_layer_range_item_block_reason(candidate)
    if candidate_block_reason:
        return {
            "ready": False,
            "reason": candidate_block_reason,
            "error": _summary_layer_range_item_block_error(candidate_block_reason),
            "modelId": model_id,
            "source": source,
            "requiredRanges": required_ranges,
            "blockedRanges": blocked_ranges,
        }

    if _summary_model_candidate_full_ready(candidate):
        return {
            "ready": True,
            "reason": "full_model_ready",
            "modelId": model_id,
            "source": source,
            "requiredRanges": required_ranges,
        }

    available_ranges = _summary_available_ranges_from_candidate(candidate)
    missing_ranges = _summary_missing_required_ranges(
        required_ranges,
        available_ranges,
    )
    if not missing_ranges:
        return {
            "ready": True,
            "reason": "required_shard_ranges_ready",
            "modelId": model_id,
            "source": source,
            "requiredRanges": required_ranges,
            "availableRanges": available_ranges,
        }

    if _summary_can_load_before_deadline(candidate):
        return {
            "ready": True,
            "reason": "can_load_before_deadline",
            "modelId": model_id,
            "source": source,
            "requiresLoading": True,
            "requiredRanges": required_ranges,
            "availableRanges": available_ranges,
            "missingRanges": missing_ranges,
        }

    return {
        "ready": False,
        "reason": "model_shard_range_not_ready",
        "error": "Executor does not advertise the assigned model shard range.",
        "modelId": model_id,
        "source": source,
        "requiredRanges": required_ranges,
        "availableRanges": available_ranges,
        "missingRanges": missing_ranges,
        "blockedRanges": blocked_ranges,
    }


def _summary_model_candidate_full_ready(candidate: Mapping[str, Any]) -> bool:
    if _summary_layer_range_item_block_reason(candidate):
        return False
    if _summary_bool(
        candidate,
        "fullModelReady",
        "full_model_ready",
        "modelReady",
        "model_ready",
        "loaded",
        "downloaded",
        "cached",
        "available",
    ):
        return True
    status = str(candidate.get("status") or "").strip().lower()
    return status in {"ready", "loaded", "cached", "downloaded", "available"}


def build_cai_owned_transport_contract(
    *,
    source_node_id: str,
    sink_node_ids: Sequence[str],
    reason: str,
    model_id: str | None = None,
    backend: str = "llama.cpp",
) -> dict[str, Any]:
    sinks = _clean_sink_node_ids(str(source_node_id or "").strip(), sink_node_ids)
    return {
        "protocol": CAI_OWNED_TRANSPORT_PROTOCOL,
        "protocolVersion": CAI_OWNED_TRANSPORT_PROTOCOL_VERSION,
        "status": "required",
        "modelId": model_id,
        "backend": backend,
        "sourceNodeId": str(source_node_id or "").strip(),
        "sinkNodeIds": sinks,
        "participantNodeIds": [
            node_id
            for node_id in [str(source_node_id or "").strip(), *sinks]
            if node_id
        ],
        "requiredCapabilities": list(CAI_OWNED_TRANSPORT_REQUIRED_CAPABILITIES),
        "runtimePhases": list(CAI_OWNED_TRANSPORT_RUNTIME_PHASES),
        "routePolicy": {
            "allowDirect": True,
            "allowRelay": True,
            "manualTunnelRequired": False,
            "avoidSingleTransitBottleneck": True,
            "minimumRelayQuorum": 0,
        },
        "dataPlaneGoal": (
            "Move CAI-owned batched shard activations and receipts over the "
            "network instead of relying on per-token standard llama.cpp RPC "
            "handshakes across WAN links."
        ),
        "reason": reason,
    }


def cai_owned_transport_runtime_readiness(
    *,
    runtime_ready: bool = False,
    implemented: bool = False,
    proof_kind: str | None = None,
    status: str | None = None,
    runtime_version: str | None = None,
    adapter_id: str | None = None,
    adapter_version: str | None = None,
    compatible_protocol_versions: Sequence[int] | None = None,
    llm_shard_self_test: Mapping[str, Any] | None = None,
    runtime_ready_proof: Mapping[str, Any] | None = None,
    require_runtime_ready_proof: bool = True,
) -> dict[str, Any]:
    self_test_summary = _cai_owned_llm_shard_self_test_summary(llm_shard_self_test)
    runtime_proof_summary = _cai_owned_transport_runtime_ready_proof_summary(
        runtime_ready_proof
    )
    if self_test_summary is not None:
        if not str(adapter_id or "").strip():
            adapter_id = str(self_test_summary.get("adapterId") or "").strip() or None
        if not str(adapter_version or "").strip():
            adapter_version = (
                str(self_test_summary.get("adapterVersion") or "").strip() or None
            )
        if (
            not str(runtime_version or "").strip()
            and bool(self_test_summary.get("productionReady"))
            and str(adapter_version or "").strip()
        ):
            runtime_version = CAI_OWNED_TRANSPORT_RUNTIME_VERSION
    effective_runtime_ready = bool(runtime_ready)
    resolved_status = status
    runtime_ready_error: str | None = None
    runtime_ready_proof_error: str | None = None
    if effective_runtime_ready and bool(require_runtime_ready_proof):
        if not (
            isinstance(runtime_proof_summary, Mapping)
            and runtime_proof_summary.get("verified") is True
        ):
            effective_runtime_ready = False
            runtime_ready_proof_error = (
                "CAI-owned transport runtimeReady requires a fresh verified "
                "live PC-to-PC proof."
            )
            if resolved_status in {None, "", "ready"}:
                resolved_status = "test_adapter_ready"
    if self_test_summary is not None:
        self_test_status = str(self_test_summary.get("status") or "").strip().lower()
        production_ready = bool(self_test_summary.get("productionReady"))
        backend_health_ready = self_test_summary.get("backendHealthReady")
        if backend_health_ready is False and resolved_status in {
            None,
            "",
            "ready",
            "test_adapter_ready",
        }:
            resolved_status = "failed"
            effective_runtime_ready = False
        elif self_test_status in {"failed", "error"} and resolved_status in {
            None,
            "",
            "ready",
            "test_adapter_ready",
        }:
            resolved_status = "failed"
            effective_runtime_ready = False
        elif not production_ready:
            effective_runtime_ready = False
            if resolved_status in {None, "", "ready"}:
                resolved_status = "test_adapter_ready"
    elif effective_runtime_ready:
        effective_runtime_ready = False
        runtime_ready_error = (
            "CAI-owned transport runtimeReady requires production LLM shard "
            "self-test readiness."
        )
        if resolved_status in {None, "", "ready"}:
            resolved_status = "test_adapter_ready"
    if resolved_status is None:
        resolved_status = (
            "ready" if effective_runtime_ready and implemented else "planned"
        )
    readiness = {
        "protocol": CAI_OWNED_TRANSPORT_PROTOCOL,
        "protocolVersion": CAI_OWNED_TRANSPORT_PROTOCOL_VERSION,
        "implemented": bool(implemented),
        "runtimeReady": bool(effective_runtime_ready),
        "runtimeReadyProofRequired": bool(require_runtime_ready_proof),
        "proofKind": str(proof_kind or "").strip() or None,
        "status": resolved_status,
        "requiredCapabilities": list(CAI_OWNED_TRANSPORT_REQUIRED_CAPABILITIES),
        "runtimePhases": list(CAI_OWNED_TRANSPORT_RUNTIME_PHASES),
    }
    if compatible_protocol_versions is not None:
        readiness["compatibleProtocolVersions"] = [
            int(item)
            for item in compatible_protocol_versions
            if _optional_int(item) is not None
        ]
    if str(runtime_version or "").strip():
        readiness["runtimeVersion"] = str(runtime_version or "").strip()
    if str(adapter_id or "").strip():
        readiness["adapterId"] = str(adapter_id or "").strip()
    if str(adapter_version or "").strip():
        readiness["adapterVersion"] = str(adapter_version or "").strip()
    if self_test_summary is not None:
        readiness["llmShardSelfTest"] = self_test_summary
    if runtime_proof_summary is not None:
        readiness["runtimeReadyProof"] = runtime_proof_summary
    if runtime_ready_proof_error:
        readiness["runtimeReadyProofError"] = runtime_ready_proof_error
    if runtime_ready_error:
        readiness["runtimeReadyError"] = runtime_ready_error
    version_compatibility = cai_owned_transport_version_compatibility(
        readiness,
        require_runtime_versions=bool(effective_runtime_ready),
        require_protocol_version=True,
    )
    readiness["versionCompatible"] = bool(version_compatibility["compatible"])
    readiness["versionCompatibility"] = version_compatibility
    return readiness


def _cai_owned_llm_shard_self_test_summary(
    self_test: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(self_test, Mapping):
        return None
    summary: dict[str, Any] = {}
    for field_name in (
        "status",
        "selfTestKind",
        "modelId",
        "adapterClass",
        "adapterId",
        "adapterVersion",
        "backend",
        "backendVersion",
        "backendMode",
        "patchBoundaryAbi",
        "patchBoundaryPatchId",
        "patchBoundaryHash",
        "productionReadinessError",
        "errorClass",
        "recordedAt",
        "backendHealthStatus",
    ):
        value = self_test.get(field_name)
        if value is not None and str(value).strip():
            summary[field_name] = str(value).strip()[:500]
    for field_name in (
        "ok",
        "contractReady",
        "productionReady",
        "patchBoundaryVerified",
        "outputFrameMetadataReady",
        "finalDecodeOutputReady",
        "generationProbeReady",
    ):
        if self_test.get(field_name) is not None:
            summary[field_name] = bool(self_test.get(field_name))
    if summary.get("productionReady") and summary.get("generationProbeReady") is not True:
        summary["productionReady"] = False
        summary.setdefault(
            "productionReadinessError",
            "LLM shard self-test is missing generation probe readiness.",
        )
    if "backendHealthReady" in self_test:
        backend_health_ready = self_test.get("backendHealthReady")
        summary["backendHealthReady"] = (
            None if backend_health_ready is None else bool(backend_health_ready)
        )
    backend_health = self_test.get("backendHealth")
    if isinstance(backend_health, Mapping):
        summary["backendHealth"] = json.loads(
            json.dumps(dict(backend_health), sort_keys=True, default=str),
        )
        backend_health_status = str(backend_health.get("status") or "").strip()
        if backend_health_status and "backendHealthStatus" not in summary:
            summary["backendHealthStatus"] = backend_health_status[:500]
    for field_name in (
        "outputPayloadSizeBytes",
        "prefillOutputPayloadSizeBytes",
        "decodeOutputPayloadSizeBytes",
    ):
        value = _optional_int(self_test.get(field_name))
        if value is not None:
            summary[field_name] = max(0, value)
    for field_name in ("productionReadinessChecks", "generationProbe"):
        value = self_test.get(field_name)
        if isinstance(value, Mapping):
            summary[field_name] = json.loads(
                json.dumps(dict(value), sort_keys=True, default=str),
            )
    latency = self_test.get("latencyMs")
    if latency is not None:
        try:
            summary["latencyMs"] = max(0.0, float(latency))
        except (TypeError, ValueError):
            pass
    error = self_test.get("error")
    if error is not None and str(error).strip():
        summary["error"] = str(error).strip()[:500]
    return summary or None


def _cai_owned_transport_runtime_ready_proof_summary(
    proof: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(proof, Mapping):
        return None
    final_result = proof.get("finalResult")
    if not isinstance(final_result, Mapping):
        final_result = {}
    executor_ids = [
        item.strip()
        for item in _summary_string_values(
            proof.get("executorNodeIds")
            or final_result.get("executorNodeIds")
            or final_result.get("executorIds")
        )
        if item.strip()
    ]
    unique_executor_ids = sorted(set(executor_ids))
    proof_verified = bool(
        proof.get("proofVerified")
        or final_result.get("proofVerified")
        or proof.get("verified")
    )
    final_output = final_result.get("finalOutput") or proof.get("finalOutput")
    has_final_output = isinstance(final_output, Mapping) and bool(final_output)
    status = str(proof.get("status") or "").strip().lower()
    status_ok = status in {"ok", "passed", "verified", "ready"} or proof_verified
    verified = bool(
        status_ok
        and proof_verified
        and len(unique_executor_ids) >= 2
        and has_final_output
    )
    summary: dict[str, Any] = {
        "verified": verified,
        "status": status or None,
        "proofVerified": proof_verified,
        "executorNodeIds": unique_executor_ids,
        "executorCount": len(unique_executor_ids),
        "hasFinalOutput": has_final_output,
    }
    for field_name in ("sessionId", "instanceId", "requesterNodeId", "recordedAt"):
        value = proof.get(field_name)
        if value is not None and str(value).strip():
            summary[field_name] = str(value).strip()[:500]
    if "cacheAgeSeconds" in proof:
        try:
            summary["cacheAgeSeconds"] = max(0.0, float(proof["cacheAgeSeconds"]))
        except (TypeError, ValueError):
            pass
    if not verified:
        if not status_ok:
            summary["error"] = "Live proof status is not ok."
        elif not proof_verified:
            summary["error"] = "Live proof is not verified."
        elif len(unique_executor_ids) < 2:
            summary["error"] = "Live proof requires at least two executor nodes."
        elif not has_final_output:
            summary["error"] = "Live proof final output is missing."
    return summary


def build_cai_owned_transport_execution_proof(
    *,
    session_id: str,
    instance_id: str,
    participant_node_ids: Sequence[str],
    executor_node_ids: Sequence[str] | None = None,
    chain_id: str | None = None,
    model_id: str | None = None,
    task_id: str | None = None,
    activation_batch_count: int = 0,
    decode_batch_count: int = 0,
    shard_receipts: Sequence[dict[str, Any]] | None = None,
    completed_at: str | None = None,
) -> dict[str, Any]:
    participants = _clean_node_ids(participant_node_ids)
    executors = _clean_node_ids(executor_node_ids or participants)
    if not set(executors).issubset(set(participants)):
        raise ValueError(
            "CAI-owned transport proof executors must be participants."
        )
    resolved_chain_id = _cai_owned_transport_chain_id(None, chain_id)
    receipts = (
        list(shard_receipts)
        if shard_receipts is not None
        else [
            {
                "nodeId": node_id,
                "network": resolved_chain_id,
                "chainId": resolved_chain_id,
                "status": "completed",
                "activationBatchCount": max(0, int(activation_batch_count)),
                "decodeBatchCount": max(0, int(decode_batch_count)),
            }
            for node_id in executors
        ]
    )
    return {
        "schemaVersion": CAI_OWNED_TRANSPORT_PROOF_SCHEMA_VERSION,
        "protocol": CAI_OWNED_TRANSPORT_PROTOCOL,
        "protocolVersion": CAI_OWNED_TRANSPORT_PROTOCOL_VERSION,
        "network": resolved_chain_id,
        "chainId": resolved_chain_id,
        "sessionId": str(session_id or "").strip(),
        "instanceId": str(instance_id or "").strip(),
        "taskId": str(task_id or "").strip() or None,
        "modelId": str(model_id or "").strip() or None,
        "participantNodeIds": participants,
        "executorNodeIds": executors,
        "activationBatchCount": max(0, int(activation_batch_count)),
        "decodeBatchCount": max(0, int(decode_batch_count)),
        "shardReceipts": receipts,
        "completed": True,
        "completedAt": completed_at or datetime.now(tz=UTC).isoformat(),
    }


def validate_cai_owned_transport_execution_proof(
    proof: dict[str, Any] | None,
    *,
    participant_node_ids: Sequence[str] | None = None,
    executor_node_ids: Sequence[str] | None = None,
    model_id: str | None = None,
    chain_id: str | None = None,
    require_signature: bool | None = None,
    trusted_signer_identities_by_node: Mapping[str, Any] | None = None,
    require_trusted_signer: bool = False,
    record_replay_cache: bool = False,
    replay_cache_policy: WalletPolicy | None = None,
    replay_cache_retention_seconds: float | int | None = None,
) -> tuple[bool, str | None]:
    if not isinstance(proof, dict):
        return False, "CAI-owned transport proof is missing."
    if int(proof.get("schemaVersion") or 0) != CAI_OWNED_TRANSPORT_PROOF_SCHEMA_VERSION:
        return False, "CAI-owned transport proof schema is unsupported."
    if str(proof.get("protocol") or "").strip() != CAI_OWNED_TRANSPORT_PROTOCOL:
        return False, "CAI-owned transport proof protocol is invalid."
    if int(proof.get("protocolVersion") or 0) != CAI_OWNED_TRANSPORT_PROTOCOL_VERSION:
        return False, "CAI-owned transport proof protocol version is unsupported."
    chain_valid, chain_error, proof_chain_id = _validate_cai_owned_transport_chain_id(
        proof,
        expected_chain_id=_cai_owned_transport_chain_id(None, chain_id),
        payload_name="execution proof",
    )
    if not chain_valid:
        return False, chain_error
    if not str(proof.get("sessionId") or "").strip():
        return False, "CAI-owned transport proof session id is missing."
    if not bool(proof.get("completed")):
        return False, "CAI-owned transport proof is not marked completed."

    expected_participants = _clean_node_ids(participant_node_ids or [])
    proof_participants = _clean_node_ids(proof.get("participantNodeIds") or [])
    if expected_participants and set(proof_participants) != set(expected_participants):
        return False, "CAI-owned transport proof participant set does not match."
    proof_executors = _clean_node_ids(proof.get("executorNodeIds") or proof_participants)
    expected_executors = _clean_node_ids(executor_node_ids or [])
    if not proof_executors:
        return False, "CAI-owned transport proof executors are missing."
    if not set(proof_executors).issubset(set(proof_participants)):
        return False, "CAI-owned transport proof executors must be participants."
    if expected_executors and set(proof_executors) != set(expected_executors):
        return False, "CAI-owned transport proof executor set does not match."
    signature_valid, signature_error = validate_cai_owned_transport_payload_signature(
        proof,
        payload_name="CAI-owned transport execution proof",
        allowed_signer_node_ids=proof_participants,
        require_signature=require_signature,
        trusted_signer_identities_by_node=trusted_signer_identities_by_node,
        require_trusted_signer=require_trusted_signer,
        record_replay_cache=record_replay_cache,
        replay_cache_policy=replay_cache_policy,
        replay_cache_retention_seconds=replay_cache_retention_seconds,
    )
    if not signature_valid:
        return False, signature_error

    expected_model = str(model_id or "").strip()
    proof_model = str(proof.get("modelId") or "").strip()
    if expected_model and proof_model and proof_model != expected_model:
        return False, "CAI-owned transport proof model id does not match."

    shard_receipts = proof.get("shardReceipts")
    if not isinstance(shard_receipts, list) or not shard_receipts:
        return False, "CAI-owned transport proof shard receipts are missing."
    receipt_node_ids: set[str] = set()
    receipt_stage_ids: set[str] = set()
    receipt_hash_chain_sha256_hexes: set[str] = set()
    for item in shard_receipts:
        if not isinstance(item, dict):
            return False, "CAI-owned transport proof shard receipt is invalid."
        receipt_chain_valid, receipt_chain_error, _receipt_chain_id = (
            _validate_cai_owned_transport_chain_id(
                item,
                expected_chain_id=proof_chain_id or "",
                payload_name="proof shard receipt",
            )
        )
        if not receipt_chain_valid:
            return False, receipt_chain_error
        receipt_node_id = str(item.get("nodeId") or "").strip()
        if not receipt_node_id:
            return False, "CAI-owned transport proof shard receipt node id is missing."
        receipt_node_ids.add(receipt_node_id)
        receipt_signature_valid, receipt_signature_error = (
            validate_cai_owned_transport_payload_signature(
                item,
                payload_name="CAI-owned transport shard receipt",
                expected_signer_node_id=receipt_node_id,
                allowed_signer_node_ids=proof_executors,
                require_signature=require_signature,
                trusted_signer_identities_by_node=trusted_signer_identities_by_node,
                require_trusted_signer=require_trusted_signer,
                record_replay_cache=record_replay_cache,
                replay_cache_policy=replay_cache_policy,
                replay_cache_retention_seconds=replay_cache_retention_seconds,
            )
        )
        if not receipt_signature_valid:
            return False, receipt_signature_error
        raw_stage_ids = item.get("stageIds") or []
        if isinstance(raw_stage_ids, (str, bytes)) or not isinstance(
            raw_stage_ids,
            Sequence,
        ):
            return False, (
                "CAI-owned transport proof shard receipt stage ids are invalid."
            )
        for raw_stage_id in raw_stage_ids:
            stage_id = str(raw_stage_id or "").strip()
            if not stage_id:
                continue
            if not _is_safe_transport_file_id(stage_id, prefix="caistage_"):
                return False, (
                    "CAI-owned transport proof shard receipt stage id is invalid."
                )
            if stage_id in receipt_stage_ids:
                return False, (
                    f"CAI-owned transport proof duplicates stage id '{stage_id}'."
                )
            receipt_stage_ids.add(stage_id)
        for field_name in (
            "inputPayloadSha256Hexes",
            "outputPayloadSha256Hexes",
            "hashChainSha256Hexes",
        ):
            raw_hashes = item.get(field_name)
            if raw_hashes is None:
                continue
            if isinstance(raw_hashes, (str, bytes)) or not isinstance(
                raw_hashes,
                Sequence,
            ):
                return False, (
                    f"CAI-owned transport proof shard receipt {field_name} is invalid."
                )
            for raw_hash in raw_hashes:
                try:
                    normalized_hash = _normalize_sha256_hex(
                        raw_hash,
                        field_name=field_name,
                    )
                except ValueError as exc:
                    return False, str(exc)
                if field_name == "hashChainSha256Hexes":
                    receipt_hash_chain_sha256_hexes.add(normalized_hash)
        raw_runtime_audits = item.get("runtimeAudits")
        if raw_runtime_audits is not None:
            if isinstance(raw_runtime_audits, Mapping):
                runtime_audits = [raw_runtime_audits]
            elif isinstance(raw_runtime_audits, (str, bytes)) or not isinstance(
                raw_runtime_audits,
                Sequence,
            ):
                return False, (
                    "CAI-owned transport proof shard receipt runtime audits "
                    "are invalid."
                )
            else:
                runtime_audits = list(raw_runtime_audits)
            for runtime_audit in runtime_audits:
                if not isinstance(runtime_audit, Mapping):
                    return False, (
                        "CAI-owned transport proof shard receipt runtime audit "
                        "is invalid."
                    )
                compatibility = cai_owned_transport_version_compatibility(
                    runtime_audit,
                    require_runtime_versions=True,
                    require_protocol_version=False,
                )
                if not bool(compatibility["compatible"]):
                    return False, "; ".join(compatibility.get("errors") or [])
    required_receipt_node_ids = set(expected_executors or proof_executors)
    if not receipt_node_ids.issubset(set(proof_participants)):
        return False, (
            "CAI-owned transport proof shard receipt node is not a participant."
        )
    if not receipt_node_ids.issubset(set(proof_executors)):
        return False, (
            "CAI-owned transport proof shard receipt node is not an executor."
        )
    if required_receipt_node_ids and not required_receipt_node_ids.issubset(
        receipt_node_ids
    ):
        return False, "CAI-owned transport proof does not cover every executor."
    receipt_batch_ids, receipt_errors = _cai_owned_transport_proof_batch_ids(proof)
    if receipt_errors:
        return False, receipt_errors[0]
    if any(
        isinstance(item, dict) and str(item.get("status") or "").strip() != "completed"
        for item in shard_receipts
    ):
        return False, "CAI-owned transport proof contains incomplete shard receipts."
    execution_audit = proof.get("executionAudit")
    if execution_audit is not None:
        if not isinstance(execution_audit, dict):
            return False, "CAI-owned transport execution audit is invalid."
        if not bool(execution_audit.get("verified")):
            return False, "CAI-owned transport execution audit is not verified."
        try:
            audit_error_count = int(execution_audit.get("errorCount") or 0)
        except (TypeError, ValueError):
            return False, "CAI-owned transport execution audit error count is invalid."
        if audit_error_count != 0:
            return False, "CAI-owned transport execution audit contains errors."
        audit_errors = execution_audit.get("errors")
        if isinstance(audit_errors, list) and audit_errors:
            return False, "CAI-owned transport execution audit contains errors."
        raw_audit_receipt_batch_ids = execution_audit.get("receiptBatchIds") or []
        raw_audit_processed_batch_ids = execution_audit.get("processedBatchIds") or []
        if isinstance(
            raw_audit_receipt_batch_ids,
            (str, bytes),
        ) or not isinstance(raw_audit_receipt_batch_ids, Sequence):
            return False, (
                "CAI-owned transport execution audit receipt batch ids are invalid."
            )
        if isinstance(
            raw_audit_processed_batch_ids,
            (str, bytes),
        ) or not isinstance(raw_audit_processed_batch_ids, Sequence):
            return False, (
                "CAI-owned transport execution audit processed batch ids are invalid."
            )
        audit_receipt_batch_ids = {
            str(value or "").strip()
            for value in raw_audit_receipt_batch_ids
            if str(value or "").strip()
        }
        audit_processed_batch_ids = {
            str(value or "").strip()
            for value in raw_audit_processed_batch_ids
            if str(value or "").strip()
        }
        if audit_receipt_batch_ids and audit_receipt_batch_ids != receipt_batch_ids:
            return False, (
                "CAI-owned transport proof receipt batch ids do not match "
                "execution audit."
            )
        if audit_processed_batch_ids and not audit_processed_batch_ids.issubset(
            receipt_batch_ids
        ):
            return False, (
                "CAI-owned transport proof is missing audited processed batch ids."
            )
        audit_hash_chain_sha256_hexes: set[str] = set()
        raw_audit_hash_chain_sha256_hexes = (
            execution_audit.get("hashChainSha256Hexes") or []
        )
        if isinstance(
            raw_audit_hash_chain_sha256_hexes,
            (str, bytes),
        ) or not isinstance(raw_audit_hash_chain_sha256_hexes, Sequence):
            return False, (
                "CAI-owned transport execution audit hash chains are invalid."
            )
        for raw_hash in raw_audit_hash_chain_sha256_hexes:
            try:
                audit_hash_chain_sha256_hexes.add(
                    _normalize_sha256_hex(
                        raw_hash,
                        field_name="hashChainSha256Hexes",
                    )
                )
            except ValueError as exc:
                return False, str(exc)
        if audit_hash_chain_sha256_hexes and not audit_hash_chain_sha256_hexes.issubset(
            receipt_hash_chain_sha256_hexes
        ):
            return False, (
                "CAI-owned transport proof hash chain does not match execution audit."
            )
    return True, None


def _attach_cai_owned_transport_llm_route_frame_templates(
    route_plan: list[dict[str, Any]],
    *,
    model_id: str,
    runtime_metadata: Mapping[str, Any],
    initial_token_count: int = 0,
) -> None:
    next_template: dict[str, Any] | None = None
    token_count = max(0, int(initial_token_count or 0))
    for item in reversed(route_plan):
        if bool(item.get("finalOutput")):
            next_template = None
            continue
        try:
            layer_start = int(item.get("layerStart"))
            layer_end = int(item.get("layerEnd"))
            sequence = int(item.get("sequence") or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "CAI-owned dispatch LLM route frame template is invalid."
            ) from exc
        template = build_cai_owned_llm_shard_frame_metadata_from_runtime(
            model_id=model_id,
            runtime_metadata=runtime_metadata,
            payload=b"",
            frame_kind=_cai_owned_transport_frame_kind_for_phase(
                str(item.get("phase") or "")
            ),
            layer_start=layer_start,
            layer_end=layer_end,
            token_start=_cai_owned_transport_template_token_start(
                str(item.get("phase") or ""),
                token_count,
            ),
            token_end=_cai_owned_transport_template_token_end(
                str(item.get("phase") or ""),
                token_count,
            ),
            sequence=sequence,
        )
        if item.get("stageId") is not None:
            template["stageId"] = item.get("stageId")
        if next_template is not None:
            template["nextFrameTemplate"] = next_template
        item["frameTemplate"] = template
        next_template = template


def _validate_cai_owned_transport_execution_dag_coverage(
    record: CaiOwnedTransportSessionRecord,
    batch_records: Sequence[dict[str, Any]],
    *,
    final_output_batch_ids: Sequence[str],
) -> tuple[dict[str, Any] | None, list[str]]:
    route_policy = (
        record.route_policy
        if isinstance(record.route_policy, Mapping)
        else {}
    )
    dag = (
        route_policy.get("executionDag")
        if isinstance(route_policy, Mapping)
        else None
    )
    if not isinstance(dag, dict):
        return None, []

    errors: list[str] = []
    valid, error = validate_cai_owned_transport_execution_dag(
        dag,
        chain_id=record.chain_id,
        session_id=record.session_id,
        participant_node_ids=record.participant_node_ids,
    )
    if not valid and error:
        errors.append(error)

    stages = [
        dict(stage)
        for stage in dag.get("stages") or []
        if isinstance(stage, dict)
    ]
    expected_stage_ids = [
        str(stage.get("stageId") or "").strip()
        for stage in stages
        if str(stage.get("stageId") or "").strip()
    ]
    processed_stage_ids: list[str] = []
    for batch in batch_records:
        if str(batch.get("status") or "").strip() != "processed":
            continue
        metadata = batch.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        stage_id = str(metadata.get("stageId") or "").strip()
        if stage_id:
            processed_stage_ids.append(stage_id)
    for receipt in record.shard_receipts:
        if not isinstance(receipt, Mapping):
            continue
        for raw_stage_id in receipt.get("stageIds") or []:
            stage_id = str(raw_stage_id or "").strip()
            if stage_id:
                processed_stage_ids.append(stage_id)

    processed_stage_id_set = set(processed_stage_ids)
    expected_stage_id_set = set(expected_stage_ids)
    missing_stage_ids = [
        stage_id
        for stage_id in expected_stage_ids
        if stage_id not in processed_stage_id_set
    ]
    unknown_stage_ids = [
        stage_id
        for stage_id in processed_stage_ids
        if stage_id not in expected_stage_id_set
    ]
    if missing_stage_ids:
        errors.append(
            "CAI-owned transport execution DAG is missing processed stage ids: "
            + ", ".join(missing_stage_ids)
        )
    if unknown_stage_ids:
        errors.append(
            "CAI-owned transport execution DAG has unknown processed stage ids: "
            + ", ".join(unknown_stage_ids)
        )
    if expected_stage_ids and not final_output_batch_ids:
        errors.append("CAI-owned transport execution DAG final output is missing.")

    return (
        {
            "verified": not errors,
            "dagHashSha256Hex": dag.get("dagHashSha256Hex"),
            "expectedStageIds": expected_stage_ids,
            "processedStageIds": processed_stage_ids,
            "missingStageIds": missing_stage_ids,
            "unknownStageIds": unknown_stage_ids,
            "finalOutputBatchIds": list(final_output_batch_ids),
        },
        errors,
    )


def _require_session_participant(
    record: CaiOwnedTransportSessionRecord,
    node_id: str,
) -> str:
    clean = str(node_id or "").strip()
    if not clean:
        raise ValueError("CAI-owned transport participant node id is required.")
    if clean not in set(record.participant_node_ids):
        raise ValueError(
            f"Node '{clean}' is not part of CAI-owned transport session "
            f"'{record.session_id}'."
        )
    return clean


def _find_cai_owned_transport_batch(
    record: CaiOwnedTransportSessionRecord,
    batch_id: str,
) -> dict[str, Any] | None:
    clean_batch_id = str(batch_id or "").strip()
    for batch in record.batch_records:
        if not isinstance(batch, dict):
            continue
        if str(batch.get("batchId") or "").strip() == clean_batch_id:
            return dict(batch)
    return None


def _is_cai_owned_transport_final_output_batch(
    record: CaiOwnedTransportSessionRecord,
    batch: Mapping[str, Any],
) -> bool:
    if not isinstance(batch, Mapping):
        return False
    metadata = batch.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    if str(metadata.get("payloadRole") or "").strip() != "shard_output":
        return False
    source = str(batch.get("sourceNodeId") or "").strip()
    sink = str(batch.get("sinkNodeId") or "").strip()
    if not source or not sink:
        return False
    executors = {
        str(node_id or "").strip()
        for node_id in (record.executor_node_ids or record.participant_node_ids)
        if str(node_id or "").strip()
    }
    if not executors:
        return False
    explicit_final_output = bool(metadata.get("finalOutput"))
    return source in executors and (sink not in executors or explicit_final_output)


def _cai_owned_transport_batch_record(
    session_id: str,
    batch_id: str,
    policy: WalletPolicy | None = None,
) -> dict[str, Any] | None:
    clean_batch_id = str(batch_id or "").strip()
    record = _cai_owned_transport_session_record(session_id, policy)
    if record is not None:
        return _find_cai_owned_transport_batch(record, clean_batch_id)
    return None


def _cai_owned_transport_session_record(
    session_id: str,
    policy: WalletPolicy | None = None,
) -> CaiOwnedTransportSessionRecord | None:
    clean_session_id = str(session_id or "").strip()
    for record in list_cai_owned_transport_sessions(policy):
        if record.session_id == clean_session_id:
            return record
    return None


def _find_cai_owned_transport_dispatch_record(
    record: CaiOwnedTransportSessionRecord | None,
    envelope: Mapping[str, Any],
    *,
    dispatch_kind: str = "initial_batch",
) -> dict[str, Any] | None:
    if record is None:
        return None
    batch_id = str(envelope.get("batchId") or "").strip()
    payload_hash = str(envelope.get("payloadSha256Hex") or "").strip().lower()
    for item in record.dispatch_records:
        if not isinstance(item, dict):
            continue
        if str(item.get("dispatchKind") or "").strip() != dispatch_kind:
            continue
        if str(item.get("batchId") or "").strip() != batch_id:
            continue
        if str(item.get("payloadSha256Hex") or "").strip().lower() != payload_hash:
            continue
        return dict(item)
    return None


def _cai_owned_transport_dispatch_record_was_sent(
    record: dict[str, Any] | None,
) -> bool:
    if not isinstance(record, dict):
        return False
    return str(record.get("status") or "").strip() in {
        "sent",
        "submitted",
        "delivered",
        "completed",
    }


def _upsert_cai_owned_transport_dispatch_record(
    session_id: str,
    dispatch_record: Mapping[str, Any],
    *,
    policy: WalletPolicy | None = None,
) -> CaiOwnedTransportSessionRecord:
    clean_session_id = str(session_id or "").strip()
    dispatch_id = str(dispatch_record.get("dispatchId") or "").strip()
    if not clean_session_id:
        raise ValueError("CAI-owned transport dispatch journal requires session id.")
    if not dispatch_id:
        raise ValueError("CAI-owned transport dispatch journal requires dispatch id.")
    records = list_cai_owned_transport_sessions(policy)
    for record_index, record in enumerate(records):
        if record.session_id != clean_session_id:
            continue
        now = datetime.now(tz=UTC).isoformat()
        updated_dispatch = dict(dispatch_record)
        updated_dispatch["dispatchId"] = dispatch_id
        updated_dispatch["updatedAt"] = now
        replaced = False
        updated_records: list[dict[str, Any]] = []
        for existing in record.dispatch_records:
            if not isinstance(existing, dict):
                continue
            if str(existing.get("dispatchId") or "").strip() != dispatch_id:
                updated_records.append(existing)
                continue
            merged = dict(existing)
            merged.update(updated_dispatch)
            merged.setdefault("createdAt", existing.get("createdAt") or now)
            updated_records.append(merged)
            replaced = True
        if not replaced:
            updated_dispatch.setdefault("createdAt", now)
            updated_records.append(updated_dispatch)
        record.dispatch_records = updated_records
        if record.status == "created":
            record.status = "running"
        record.updated_at = now
        records[record_index] = record
        save_cai_owned_transport_sessions(records, policy)
        return record
    raise ValueError(f"CAI-owned transport session '{clean_session_id}' not found.")


def _cai_owned_transport_initial_dispatch_record(
    envelope: Mapping[str, Any],
    *,
    status: str,
    response: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    batch_id = str(envelope.get("batchId") or "").strip()
    record = {
        "dispatchId": f"initial_batch:{batch_id}",
        "dispatchKind": "initial_batch",
        "status": str(status or "").strip() or "prepared",
        "batchId": batch_id,
        "phase": str(envelope.get("phase") or "").strip(),
        "sourceNodeId": str(envelope.get("sourceNodeId") or "").strip(),
        "sinkNodeId": str(envelope.get("sinkNodeId") or "").strip(),
        "sequence": int(envelope.get("sequence") or 0),
        "payloadSha256Hex": str(envelope.get("payloadSha256Hex") or "").strip(),
        "payloadSizeBytes": int(envelope.get("payloadSizeBytes") or 0),
    }
    if response is not None:
        record["response"] = _jsonable_dict(response, field_name="dispatchResponse")
        route_audit = response.get("routeAudit") if isinstance(response, Mapping) else None
        if isinstance(route_audit, Mapping):
            record["routeAudit"] = _jsonable_dict(route_audit, field_name="routeAudit")
        peer_url = response.get("peerCaiUrl") if isinstance(response, Mapping) else None
        if peer_url:
            record["peerCaiUrl"] = str(peer_url)
    return record


def _require_cai_owned_transport_batch_completion_owner(
    session_id: str,
    batch_id: str,
    *,
    node_id: str,
    runtime_id: str | None,
    policy: WalletPolicy | None = None,
) -> None:
    clean_runtime_id = str(runtime_id or "").strip()
    if not clean_runtime_id:
        return
    record = _cai_owned_transport_session_record(session_id, policy)
    if record is None:
        raise ValueError(f"CAI-owned transport session '{session_id}' not found.")
    local = _require_session_participant(record, node_id)
    clean_batch_id = _require_safe_transport_file_id(batch_id, prefix="caibatch_")
    batch = _find_cai_owned_transport_batch(record, clean_batch_id)
    if batch is None:
        raise ValueError(
            f"CAI-owned transport batch '{clean_batch_id}' not found in session "
            f"'{record.session_id}'."
        )
    if str(batch.get("sinkNodeId") or "").strip() != local:
        raise ValueError("CAI-owned transport batch is not assigned to local node.")
    if str(batch.get("status") or "").strip() != "processing":
        raise ValueError("CAI-owned transport batch is not processing.")
    if str(batch.get("runtimeId") or "").strip() != clean_runtime_id:
        raise ValueError("CAI-owned transport batch runtime id does not match.")
    if _cai_owned_transport_batch_lease_expired(batch, datetime.now(tz=UTC)):
        raise ValueError("CAI-owned transport batch lease has expired.")


def _validate_cai_owned_transport_processed_batch_execution_audit(
    session_id: str,
    batch: dict[str, Any],
    *,
    known_batch_ids: set[str],
    policy: WalletPolicy | None = None,
) -> tuple[bool, str | None, str | None]:
    batch_id = str(batch.get("batchId") or "").strip()
    metadata = batch.get("metadata") if isinstance(batch.get("metadata"), dict) else {}
    input_hash = str(
        batch.get("inputPayloadSha256Hex") or batch.get("payloadSha256Hex") or ""
    ).strip()
    output_hash = str(batch.get("outputPayloadSha256Hex") or "").strip()
    chain_hash = str(batch.get("hashChainSha256Hex") or "").strip()
    if not input_hash:
        return (
            False,
            f"CAI-owned transport batch '{batch_id}' is missing input hash.",
            None,
        )
    if not output_hash:
        return (
            False,
            f"CAI-owned transport batch '{batch_id}' is missing output hash.",
            None,
        )
    if not chain_hash:
        return (
            False,
            f"CAI-owned transport batch '{batch_id}' is missing hash chain.",
            None,
        )
    sequence = _optional_int(batch.get("sequence"))
    if sequence is None:
        sequence = _optional_int(metadata.get("sequence"))
    if sequence is None:
        sequence = 0
    previous_batch_id = (
        str(batch.get("previousBatchId") or "").strip()
        or str(metadata.get("previousBatchId") or "").strip()
        or None
    )
    if previous_batch_id and previous_batch_id not in known_batch_ids:
        return (
            False,
            (
                f"CAI-owned transport batch '{batch_id}' previous batch "
                f"'{previous_batch_id}' is not in this session."
            ),
            None,
        )
    try:
        expected_chain = build_cai_owned_transport_batch_hash_chain(
            session_id=session_id,
            batch_id=batch_id,
            input_payload_sha256_hex=input_hash,
            output_payload_sha256_hex=output_hash,
            sequence=sequence,
            previous_batch_id=previous_batch_id,
        )
        normalized_chain_hash = _normalize_sha256_hex(
            chain_hash,
            field_name="hashChainSha256Hex",
        )
    except ValueError as exc:
        return False, str(exc), None
    if expected_chain["hashChainSha256Hex"] != normalized_chain_hash:
        return (
            False,
            f"CAI-owned transport batch '{batch_id}' hash chain does not match.",
            normalized_chain_hash,
        )
    if str(batch.get("outputPayloadStorageKey") or "").strip():
        try:
            verified_cai_owned_transport_batch_output_payload_path(
                session_id,
                batch_id,
                policy,
            )
        except Exception as exc:
            return (
                False,
                (
                    f"CAI-owned transport batch '{batch_id}' output payload "
                    f"is not verifiable: {exc}"
                ),
                normalized_chain_hash,
            )
    return True, None, normalized_chain_hash

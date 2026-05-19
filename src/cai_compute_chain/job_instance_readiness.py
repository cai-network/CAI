# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Callable
from typing import Any


CAI_INSTANCE_READINESS_PROTOCOL_VERSION = 1
CAI_INSTANCE_READINESS_STAGES = (
    "download",
    "materialize",
    "load",
    "rpc_ready",
    "inference_ready",
)


def attach_cai_instance_readiness_state(
    audit: dict[str, Any],
) -> dict[str, Any]:
    status = str(audit.get("status") or "").strip()
    ready = bool(audit.get("ready"))
    current_stage, completed_stages, blocked_stages, next_stage = (
        cai_instance_readiness_stage_for_status(status, ready=ready)
    )
    stage_items = cai_instance_readiness_stage_items(
        current_stage=current_stage,
        completed_stages=completed_stages,
        blocked_stages=blocked_stages,
    )
    state = {
        "protocolVersion": CAI_INSTANCE_READINESS_PROTOCOL_VERSION,
        "status": status,
        "ready": ready,
        "stageOrder": list(CAI_INSTANCE_READINESS_STAGES),
        "currentStage": current_stage,
        "currentStageIndex": (
            CAI_INSTANCE_READINESS_STAGES.index(current_stage)
            if current_stage in CAI_INSTANCE_READINESS_STAGES
            else -1
        ),
        "nextStage": next_stage,
        "completedStages": completed_stages,
        "blockedStages": blocked_stages,
        "stages": stage_items,
    }
    enriched = dict(audit)
    enriched["readinessProtocolVersion"] = CAI_INSTANCE_READINESS_PROTOCOL_VERSION
    enriched["currentStage"] = current_stage
    enriched["nextStage"] = next_stage
    enriched["readinessState"] = state
    return enriched


def cai_instance_readiness_stage_for_status(
    status: str,
    *,
    ready: bool,
) -> tuple[str, list[str], list[str], str | None]:
    if ready or status == "inference_ready":
        return (
            "inference_ready",
            list(CAI_INSTANCE_READINESS_STAGES),
            [],
            None,
        )
    if status == "model_materializing":
        return ("materialize", ["download"], [], "load")
    if status == "runner_missing":
        return ("materialize", ["download"], ["materialize"], "load")
    if status in {"model_loading"}:
        return ("load", ["download", "materialize"], [], "rpc_ready")
    if status in {"route_blocked", "cai_owned_route_blocked"}:
        return (
            "rpc_ready",
            ["download", "materialize", "load"],
            ["rpc_ready"],
            "rpc_ready",
        )
    if status in {"rpc_ready", "cai_owned_route_ready"}:
        return (
            "rpc_ready",
            ["download", "materialize", "load", "rpc_ready"],
            [],
            "inference_ready",
        )
    if status == "shard_loading":
        return ("download", [], [], "materialize")
    if status in {"state_unavailable", "model_missing"}:
        return ("download", [], ["download"], "download")
    return ("load", ["download", "materialize"], ["load"], "rpc_ready")


def cai_instance_readiness_stage_items(
    *,
    current_stage: str,
    completed_stages: list[str],
    blocked_stages: list[str],
) -> list[dict[str, Any]]:
    completed = set(completed_stages)
    blocked = set(blocked_stages)
    items: list[dict[str, Any]] = []
    for index, stage in enumerate(CAI_INSTANCE_READINESS_STAGES):
        if stage == current_stage:
            stage_status = "current"
        elif stage in blocked:
            stage_status = "blocked"
        elif stage in completed:
            stage_status = "completed"
        else:
            stage_status = "pending"
        items.append(
            {
                "stage": stage,
                "index": index,
                "status": stage_status,
            }
        )
    return items


def describe_pending_model_downloads(
    cai_url: str,
    model_id: str,
    *,
    load_cai_state_payload_func: Callable[[str], dict[str, Any] | None],
    pending_model_download_node_labels_func: Callable[
        [dict[str, Any], str],
        list[str],
    ],
) -> str | None:
    state_payload = load_cai_state_payload_func(cai_url) or {}
    pending_nodes = pending_model_download_node_labels_func(state_payload, model_id)
    if not pending_nodes:
        return None

    listed_nodes = ", ".join(pending_nodes[:3])
    if len(pending_nodes) > 3:
        listed_nodes = f"{listed_nodes}, +{len(pending_nodes) - 3} more"
    return (
        "Model shards are still downloading on "
        f"{len(pending_nodes)} node(s): {listed_nodes}."
    )


def pending_model_download_node_labels(
    state_payload: dict[str, Any],
    model_id: str,
    *,
    model_download_node_labels_func: Callable[..., list[str]],
    download_progress_is_completed_func: Callable[[Any], bool],
    download_progress_is_pending_func: Callable[[Any], bool],
) -> list[str]:
    return model_download_node_labels_func(
        state_payload,
        model_id,
        include_node=lambda matching_progress: (
            not any(
                download_progress_is_completed_func(progress_item)
                for progress_item in matching_progress
            )
            and any(
                download_progress_is_pending_func(progress_item)
                for progress_item in matching_progress
            )
        ),
    )


def completed_model_download_node_labels(
    state_payload: dict[str, Any],
    model_id: str,
    *,
    model_download_node_labels_func: Callable[..., list[str]],
    download_progress_is_completed_func: Callable[[Any], bool],
) -> list[str]:
    return model_download_node_labels_func(
        state_payload,
        model_id,
        include_node=lambda matching_progress: any(
            download_progress_is_completed_func(progress_item)
            for progress_item in matching_progress
        ),
    )


def model_download_node_labels(
    state_payload: dict[str, Any],
    model_id: str,
    *,
    include_node: Callable[[list[Any]], bool],
    download_progress_matches_model_func: Callable[[Any, str], bool],
) -> list[str]:
    downloads_payload = state_payload.get("downloads") or {}
    if not isinstance(downloads_payload, dict):
        return []
    node_identities = state_payload.get("nodeIdentities") or {}
    if not isinstance(node_identities, dict):
        node_identities = {}

    node_labels: list[str] = []
    for node_id, progress_items in downloads_payload.items():
        if not isinstance(progress_items, list):
            continue
        matching_progress = [
            progress_item
            for progress_item in progress_items
            if download_progress_matches_model_func(progress_item, model_id)
        ]
        if not matching_progress or not bool(include_node(matching_progress)):
            continue
        identity = node_identities.get(node_id) or {}
        friendly_name = None
        if isinstance(identity, dict):
            friendly_name = str(identity.get("friendlyName") or "").strip() or None
        node_labels.append(friendly_name or str(node_id))

    return list(dict.fromkeys(node_labels))


def download_progress_matches_model(
    progress_item: Any,
    model_id: str,
    *,
    normalize_network_model_id_func: Callable[[str], str],
    download_progress_equivalent_model_ids_func: Callable[[str], set[str]],
) -> bool:
    if not isinstance(progress_item, dict) or not progress_item:
        return False
    _status_name, payload = next(iter(progress_item.items()))
    if not isinstance(payload, dict):
        return False
    shard_metadata = payload.get("shardMetadata") or {}
    if not isinstance(shard_metadata, dict) or not shard_metadata:
        return False
    _metadata_type, metadata_payload = next(iter(shard_metadata.items()))
    if not isinstance(metadata_payload, dict):
        return False
    model_card = metadata_payload.get("modelCard") or {}
    if not isinstance(model_card, dict):
        return False
    progress_model_id = str(model_card.get("modelId") or "").strip()
    if not progress_model_id:
        return False
    return (
        normalize_network_model_id_func(progress_model_id)
        in download_progress_equivalent_model_ids_func(model_id)
    )


def download_progress_equivalent_model_ids(
    model_id: str,
    *,
    accepted_worker_model_ids_func: Callable[[str], set[str]],
    curated_model_for_id_func: Callable[[str], Any],
    curated_model_registry_func: Callable[[], list[Any]],
    normalize_network_model_id_func: Callable[[str], str],
) -> set[str]:
    equivalent_ids = set(accepted_worker_model_ids_func(model_id))
    requested_model = curated_model_for_id_func(model_id)
    preferred_filename = (
        str(getattr(requested_model, "preferred_filename", "") or "").strip().lower()
        if requested_model is not None
        else ""
    )
    if preferred_filename:
        for registry_model in curated_model_registry_func():
            registry_filename = str(
                getattr(registry_model, "preferred_filename", "") or ""
            ).strip().lower()
            if registry_filename != preferred_filename:
                continue
            equivalent_ids.add(registry_model.model_id)
            equivalent_ids.add(registry_model.execution_model_id)
            equivalent_ids.update(registry_model.runtime_model_ids)
    return {normalize_network_model_id_func(item) for item in equivalent_ids if item}


def download_progress_is_pending(progress_item: Any) -> bool:
    if not isinstance(progress_item, dict) or not progress_item:
        return False
    status_name = str(next(iter(progress_item.keys())))
    return status_name in {"DownloadPending", "DownloadOngoing"}


def download_progress_is_completed(progress_item: Any) -> bool:
    if not isinstance(progress_item, dict) or not progress_item:
        return False
    status_name = str(next(iter(progress_item.keys())))
    return status_name == "DownloadCompleted"

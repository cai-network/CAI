# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import contextlib
import hashlib
import importlib
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import anyio
from anyio import BrokenResourceError, ClosedResourceError, fail_after, to_thread
from loguru import logger

from cai.api.types import ImageEditsTaskParams
from cai.download.download_utils import is_read_only_model_dir, resolve_existing_model
from cai.shared.apply import apply
from cai.shared.constants import CAI_MAX_INSTANCE_RETRIES
from cai.shared.models.model_cards import (
    InferenceBackend,
    ModelId,
    add_to_card_cache,
    delete_custom_card,
    delete_custom_model_local_path,
    get_custom_model_local_path,
    set_custom_model_local_path,
)
from cai.shared.types.chunks import InputImageChunk
from cai.shared.types.commands import (
    DeleteInstance,
    ForwarderCommand,
    ForwarderDownloadCommand,
    StartDownload,
)
from cai.shared.types.common import CommandId, NodeId, SystemId
from cai.shared.types.events import (
    CustomModelCardAdded,
    CustomModelCardDeleted,
    Event,
    IndexedEvent,
    InputChunkReceived,
    InstanceDeleted,
    NodeDownloadProgress,
    NodeGatheredInfo,
    TaskCreated,
    TaskStatusUpdated,
    TopologyEdgeCreated,
    TopologyEdgeDeleted,
)
from cai.shared.types.multiaddr import Multiaddr
from cai.shared.types.state import State
from cai.shared.types.tasks import (
    CancelTask,
    CreateRunner,
    DownloadModel,
    ImageEdits,
    LoadModel,
    Shutdown,
    Task,
    TaskStatus,
    TextGeneration,
)
from cai.shared.types.text_generation import Base64Image, Base64ImageHash
from cai.shared.types.topology import Connection, SocketConnection
from cai.shared.types.worker.downloads import DownloadCompleted, DownloadFailed
from cai.shared.types.worker.instances import InstanceId
from cai.shared.types.worker.runners import RunnerId
from cai.utils.channels import Receiver, Sender, channel
from cai.utils.info_gatherer.info_gatherer import GatheredInfo, InfoGatherer
from cai.utils.info_gatherer.net_profile import check_reachable
from cai.utils.keyed_backoff import KeyedBackoff
from cai.utils.task_group import TaskGroup
from cai.worker.plan import plan
from cai.worker.runner.runner_supervisor import RunnerSupervisor


CAI_CHUNK_SEED_URLS_ENV = "CAI_CHUNK_SEED_URLS"
CAI_DISABLE_HF_MODEL_PACKAGE_DISCOVERY_ENV = "CAI_DISABLE_HF_MODEL_PACKAGE_DISCOVERY"
_CAI_OWNED_TRANSPORT_FALSE_VALUES = {"0", "false", "no", "off", "disabled"}
_REMOTE_MODEL_PACKAGE_IMPORT_ATTEMPTS: set[tuple[str, str, str]] = set()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _CAI_OWNED_TRANSPORT_FALSE_VALUES


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(minimum, float(raw))
    except ValueError:
        return default


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _cai_owned_transport_runtime_enabled() -> bool:
    return _env_bool("CAI_OWNED_TRANSPORT_RUNTIME_ENABLED", True)


def _cai_owned_transport_runtime_requires_worker_enabled() -> bool:
    return _env_bool("CAI_OWNED_TRANSPORT_REQUIRE_WORKER_ENABLED", True)


def _cai_owned_transport_startup_self_test_enabled() -> bool:
    return _env_bool("CAI_OWNED_TRANSPORT_STARTUP_SELF_TEST", False)


def _cai_owned_transport_self_test_retry_seconds() -> float:
    return _env_float(
        "CAI_OWNED_TRANSPORT_SELF_TEST_RETRY_SECONDS",
        300.0,
        minimum=5.0,
    )


def _cai_owned_transport_self_test_model_id(wallet_model) -> str:
    configured = str(os.getenv("CAI_OWNED_TRANSPORT_SELF_TEST_MODEL_ID") or "").strip()
    if configured:
        return configured
    try:
        policy = wallet_model.NetworkModelPolicy()
        return str(
            getattr(policy, "network_default_execution_model_id", "")
            or getattr(policy, "network_default_model_id", "")
        ).strip()
    except Exception:
        return "Qwen/Qwen3-0.6B-GGUF"


def _cached_llm_shard_self_test_ready(cached: object, *, model_id: str) -> bool:
    if not isinstance(cached, dict):
        return False
    cached_model_id = str(cached.get("modelId") or "").strip()
    if cached_model_id and cached_model_id != str(model_id or "").strip():
        return False
    return bool(cached.get("productionReady")) and bool(
        cached.get("generationProbeReady")
    )


async def _ensure_cai_owned_transport_startup_self_test(
    *,
    runtime,
    wallet_model,
    adapter,
) -> None:
    if not _cai_owned_transport_startup_self_test_enabled():
        return
    if (
        _cai_owned_transport_runtime_requires_worker_enabled()
        and not await to_thread.run_sync(_local_cai_worker_enabled)
    ):
        return
    policy = wallet_model.WalletPolicy()
    model_id = _cai_owned_transport_self_test_model_id(wallet_model)
    try:
        cached = await to_thread.run_sync(
            lambda: runtime.load_cai_owned_llm_shard_self_test_result(policy=policy)
        )
        if _cached_llm_shard_self_test_ready(cached, model_id=model_id):
            return
        logger.info("Running CAI-owned LLM shard startup self-test for {}", model_id)
        result = await to_thread.run_sync(
            lambda: runtime.run_cai_owned_llm_shard_adapter_self_test(
                adapter,
                model_id=model_id,
                require_production_llm_handoff=True,
                require_generation_probe=True,
            )
        )
        await to_thread.run_sync(
            lambda: runtime.save_cai_owned_llm_shard_self_test_result(
                result,
                policy=policy,
            )
        )
        if result.get("productionReady"):
            logger.info("CAI-owned LLM shard startup self-test passed")
        else:
            logger.warning(
                "CAI-owned LLM shard startup self-test did not reach production readiness: {}",
                result.get("productionReadinessError") or result.get("error"),
            )
    except Exception:
        logger.exception("CAI-owned LLM shard startup self-test failed")


def _local_cai_worker_enabled() -> bool:
    try:
        from cai_compute_chain.model import WalletPolicy
        from cai_compute_chain.node_config import load_or_create_node_config

        return bool(load_or_create_node_config(WalletPolicy()).worker_enabled)
    except Exception:
        return False


def _safe_runtime_id(value: object) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in {"_", "-"} else "_"
        for ch in str(value or "").strip()
    )
    return cleaned or "unknown"


def _cai_owned_transport_runtime_id(node_id: NodeId) -> str:
    configured = str(os.getenv("CAI_OWNED_TRANSPORT_RUNTIME_ID") or "").strip()
    if configured:
        return configured
    return f"cai-owned-worker-{_safe_runtime_id(node_id)}-{os.getpid()}"


def _cai_owned_transport_adapter(node_id: NodeId) -> object:
    runtime = importlib.import_module("cai_compute_chain.cai_owned_runtime")
    adapter_env = dict(os.environ)
    prefix = os.getenv("CAI_OWNED_TRANSPORT_DETERMINISTIC_PREFIX")
    if prefix is None:
        prefix_value = ""
    elif "{node_id}" in prefix:
        try:
            prefix_value = prefix.format(node_id=str(node_id))
        except Exception:
            prefix_value = prefix
    else:
        prefix_value = prefix
    if (
        prefix_value
        and not str(adapter_env.get("CAI_LLM_SHARD_ADAPTER") or "").strip()
        and not str(adapter_env.get("CAI_DETERMINISTIC_SHARD_PREFIX") or "").strip()
    ):
        adapter_env["CAI_DETERMINISTIC_SHARD_PREFIX"] = prefix_value
    return runtime.cai_owned_shard_adapter_from_env(adapter_env)


def _cai_owned_transport_runtime_requires_production_handoff() -> bool:
    return _env_bool("CAI_REQUIRE_PRODUCTION_LLM_HANDOFF", False)


def _http_url(host: object, port: object) -> str | None:
    clean_host = str(host or "").strip()
    if not clean_host or clean_host in {"0.0.0.0", "::"}:
        return None
    try:
        clean_port = int(port)
    except (TypeError, ValueError):
        return None
    if clean_port <= 0:
        return None
    if ":" in clean_host and not clean_host.startswith("["):
        clean_host = f"[{clean_host}]"
    return f"http://{clean_host}:{clean_port}"


def _identity_api_urls(identity: object) -> list[str]:
    urls: list[str] = []
    transport_endpoints_for = getattr(identity, "transport_endpoints_for", None)
    if callable(transport_endpoints_for):
        try:
            endpoints = transport_endpoints_for(purpose="api", require_port=True)
        except TypeError:
            endpoints = []
        for endpoint in endpoints or []:
            url = _http_url(
                getattr(endpoint, "host", None),
                getattr(endpoint, "port", None),
            )
            if url:
                urls.append(url)
    for host, port in (
        (getattr(identity, "api_host", None), getattr(identity, "api_port", None)),
        (getattr(identity, "data_host", None), getattr(identity, "data_port", None)),
    ):
        url = _http_url(host, port)
        if url:
            urls.append(url)
    return list(dict.fromkeys(urls))


def _cai_owned_transport_peer_urls_by_node(
    state: State,
    *,
    local_node_id: NodeId,
    api_port: int,
) -> dict[str, list[str]]:
    urls_by_node: dict[str, list[str]] = {
        str(local_node_id): [f"http://127.0.0.1:{int(api_port)}"]
    }
    for node_id, identity in state.node_identities.items():
        urls = _identity_api_urls(identity)
        if not urls:
            continue
        urls_by_node[str(node_id)] = list(
            dict.fromkeys([*urls_by_node.get(str(node_id), []), *urls])
        )
    return urls_by_node


def _configured_chunk_seed_urls() -> tuple[str, ...]:
    raw = str(os.getenv(CAI_CHUNK_SEED_URLS_ENV) or "").strip()
    if not raw:
        return ()
    normalized: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[\r\n,;]+", raw):
        item = str(part or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return tuple(normalized)


def _sync_chunk_backed_peer_inventories(
    *,
    api_port: int | None = None,
) -> object | None:
    if api_port is None:
        return None

    try:
        model_distribution = importlib.import_module(
            "cai_compute_chain.model_distribution"
        )
    except Exception:
        return None

    try:
        with urlopen(f"http://127.0.0.1:{int(api_port)}/state", timeout=5) as response:
            state_payload = json.loads(response.read().decode("utf-8"))
        return model_distribution.sync_chunk_inventory_from_cai_peers(
            state_payload=state_payload,
            CAI_url=f"http://127.0.0.1:{int(api_port)}",
            source_kind="peer_cache",
            prune_missing_peers=True,
        )
    except Exception:
        return None


def _sync_chunk_backed_seed_inventories() -> object | None:
    seed_urls = _configured_chunk_seed_urls()
    if not seed_urls:
        return None

    try:
        model_distribution = importlib.import_module(
            "cai_compute_chain.model_distribution"
        )
    except Exception:
        return None

    try:
        return model_distribution.sync_chunk_inventory_from_urls(
            inventory_urls=seed_urls,
            source_kind="storage_seed",
            prune_missing_endpoint_base_urls=model_distribution.inventory_endpoint_base_urls(
                seed_urls,
                source_kind="storage_seed",
            ),
        )
    except Exception:
        return None

def _prefetch_chunk_backed_default_chunks(
    node_id: NodeId,
) -> object | None:
    try:
        model_distribution = importlib.import_module(
            "cai_compute_chain.model_distribution"
        )
    except Exception:
        return None

    try:
        return model_distribution.prefetch_default_chunks_from_fresh_inventories(
            node_id=str(node_id),
            max_manifests=4,
            max_tasks=8,
        )
    except Exception:
        return None


def _prefetch_chunk_backed_bootstrap_chunks(
    node_id: NodeId,
) -> object | None:
    try:
        model_distribution = importlib.import_module(
            "cai_compute_chain.model_distribution"
        )
    except Exception:
        return None

    try:
        return model_distribution.prefetch_bootstrap_chunks_from_fresh_inventories(
            node_id=str(node_id),
            max_manifests=2,
            max_tasks=4,
        )
    except Exception:
        return None


def _prefetch_chunk_backed_local_shard_hints(
    node_id: NodeId,
    shard_metadatas: list[object] | tuple[object, ...],
) -> object | None:
    normalized_hints: list[dict[str, object]] = []
    for shard in shard_metadatas:
        model_id = getattr(getattr(shard, "model_card", None), "model_id", None)
        start_layer = getattr(shard, "start_layer", None)
        end_layer = getattr(shard, "end_layer", None)
        if model_id is None or start_layer is None or end_layer is None:
            continue
        normalized_hints.append(
            {
                "model_id": str(model_id),
                "start_layer": int(start_layer),
                "end_layer": int(end_layer),
                "device_rank": int(getattr(shard, "device_rank", 0) or 0),
                "world_size": int(getattr(shard, "world_size", 1) or 1),
                "node_id": str(node_id),
            }
        )
    if not normalized_hints:
        return None
    try:
        model_distribution = importlib.import_module(
            "cai_compute_chain.model_distribution"
        )
    except Exception:
        return None
    try:
        return model_distribution.prefetch_hinted_bootstrap_chunks(normalized_hints)
    except Exception:
        return None


def _extract_chunk_backed_instance_shard_hints(
    node_id: NodeId,
    instances: list[object] | tuple[object, ...],
) -> list[dict[str, object]]:
    normalized_hints: list[dict[str, object]] = []
    seen: set[tuple[str, int, int, int, int]] = set()
    for instance in instances:
        shard_assignments = getattr(instance, "shard_assignments", None)
        if shard_assignments is None:
            continue
        node_to_runner = getattr(shard_assignments, "node_to_runner", {}) or {}
        runner_to_shard = getattr(shard_assignments, "runner_to_shard", {}) or {}
        runner_id = node_to_runner.get(node_id) or node_to_runner.get(str(node_id))
        if runner_id is None:
            continue
        shard = runner_to_shard.get(runner_id)
        if shard is None:
            continue
        model_id = getattr(getattr(shard, "model_card", None), "model_id", None)
        start_layer = getattr(shard, "start_layer", None)
        end_layer = getattr(shard, "end_layer", None)
        if model_id is None or start_layer is None or end_layer is None:
            continue
        hint_key = (
            str(model_id),
            int(start_layer),
            int(end_layer),
            int(getattr(shard, "device_rank", 0) or 0),
            int(getattr(shard, "world_size", 1) or 1),
        )
        if hint_key in seen:
            continue
        seen.add(hint_key)
        normalized_hints.append(
            {
                "model_id": str(model_id),
                "start_layer": int(start_layer),
                "end_layer": int(end_layer),
                "device_rank": int(getattr(shard, "device_rank", 0) or 0),
                "world_size": int(getattr(shard, "world_size", 1) or 1),
                "node_id": str(node_id),
            }
        )
    return normalized_hints


def _remember_chunk_backed_instance_shard_hints(
    node_id: NodeId,
    instances: list[object] | tuple[object, ...],
) -> object | None:
    normalized_hints = _extract_chunk_backed_instance_shard_hints(node_id, instances)
    if not normalized_hints:
        return None
    try:
        model_distribution = importlib.import_module(
            "cai_compute_chain.model_distribution"
        )
    except Exception:
        return None
    try:
        return model_distribution.remember_recent_shard_hints(
            str(node_id),
            normalized_hints,
        )
    except Exception:
        return None


def _prefetch_chunk_backed_instance_shard_hints(
    node_id: NodeId,
    instances: list[object] | tuple[object, ...],
) -> object | None:
    normalized_hints = _extract_chunk_backed_instance_shard_hints(node_id, instances)
    if not normalized_hints:
        return None
    try:
        model_distribution = importlib.import_module(
            "cai_compute_chain.model_distribution"
        )
    except Exception:
        return None
    try:
        return model_distribution.prefetch_hinted_bootstrap_chunks(normalized_hints)
    except Exception:
        return None


def _remember_chunk_backed_local_shard_hints(
    node_id: NodeId,
    shard_metadatas: list[object] | tuple[object, ...],
) -> object | None:
    normalized_hints: list[dict[str, object]] = []
    for shard in shard_metadatas:
        model_id = getattr(getattr(shard, "model_card", None), "model_id", None)
        start_layer = getattr(shard, "start_layer", None)
        end_layer = getattr(shard, "end_layer", None)
        if model_id is None or start_layer is None or end_layer is None:
            continue
        normalized_hints.append(
            {
                "model_id": str(model_id),
                "start_layer": int(start_layer),
                "end_layer": int(end_layer),
                "device_rank": int(getattr(shard, "device_rank", 0) or 0),
                "world_size": int(getattr(shard, "world_size", 1) or 1),
            }
        )
    if not normalized_hints:
        return None
    try:
        model_distribution = importlib.import_module(
            "cai_compute_chain.model_distribution"
        )
    except Exception:
        return None
    try:
        return model_distribution.remember_recent_shard_hints(
            str(node_id),
            normalized_hints,
        )
    except Exception:
        return None


def _prefetch_chunk_backed_recent_shard_hints(
    node_id: NodeId,
) -> object | None:
    try:
        model_distribution = importlib.import_module(
            "cai_compute_chain.model_distribution"
        )
    except Exception:
        return None
    try:
        return model_distribution.prefetch_recent_shard_hints(str(node_id))
    except Exception:
        return None


def _select_or_import_chunk_backed_manifest(
    model_distribution,
    shard,
) -> object | None:
    model_id = str(shard.model_card.model_id)
    manifest = model_distribution.select_model_package_manifest_for_model(model_id)
    if manifest is not None:
        return manifest

    preferred_filename = str(
        getattr(shard.model_card, "preferred_filename", None) or ""
    ).strip()
    source_revision = str(
        getattr(shard.model_card, "model_package_version", None) or "main"
    ).strip() or "main"
    explicit_manifest_url = str(
        getattr(shard.model_card, "model_package_manifest_url", None) or ""
    ).strip()
    if explicit_manifest_url:
        attempt_key = (model_id, preferred_filename, explicit_manifest_url)
        if attempt_key not in _REMOTE_MODEL_PACKAGE_IMPORT_ATTEMPTS:
            _REMOTE_MODEL_PACKAGE_IMPORT_ATTEMPTS.add(attempt_key)
            importer = getattr(
                model_distribution,
                "import_model_package_manifest_from_url",
                None,
            )
            if callable(importer):
                try:
                    imported = importer(
                        explicit_manifest_url,
                        expected_model_id=model_id,
                        expected_preferred_filename=preferred_filename or None,
                    )
                    if imported is not None:
                        return imported
                except Exception as exc:
                    logger.warning(
                        "Failed to import CAI model package manifest for {} from {}: {}",
                        model_id,
                        explicit_manifest_url,
                        exc,
                    )

    if _env_bool(CAI_DISABLE_HF_MODEL_PACKAGE_DISCOVERY_ENV, False):
        return None

    attempt_key = (model_id, preferred_filename, "hf-discovery")
    if attempt_key in _REMOTE_MODEL_PACKAGE_IMPORT_ATTEMPTS:
        return None
    _REMOTE_MODEL_PACKAGE_IMPORT_ATTEMPTS.add(attempt_key)
    discoverer = getattr(
        model_distribution,
        "discover_and_import_hf_model_package_manifest",
        None,
    )
    if not callable(discoverer):
        return None
    try:
        return discoverer(
            model_id,
            preferred_filename=preferred_filename or None,
            source_revision=source_revision,
            timeout_sec=5,
        )
    except Exception as exc:
        logger.debug(
            "No importable CAI model package manifest discovered for {}: {}",
            model_id,
            exc,
        )
        return None


def _try_prepare_chunk_backed_llama_cpp_download(
    node_id: NodeId,
    shard,
    *,
    api_port: int | None = None,
) -> DownloadCompleted | DownloadFailed | None:
    if shard.model_card.inference_backend != InferenceBackend.LlamaCpp:
        return None

    try:
        model_distribution = importlib.import_module(
            "cai_compute_chain.model_distribution"
        )
    except Exception:
        return None

    manifest = _select_or_import_chunk_backed_manifest(model_distribution, shard)
    if manifest is None:
        return None

    try:
        _remember_chunk_backed_local_shard_hints(node_id, [shard])
        _sync_chunk_backed_peer_inventories(api_port=api_port)
        _sync_chunk_backed_seed_inventories()

        assignment = model_distribution.ModelShardAssignment(
            start_layer=shard.start_layer,
            end_layer=shard.end_layer,
            device_rank=shard.device_rank,
            world_size=shard.world_size,
            node_id=str(node_id),
        )

        result = model_distribution.ensure_assignment_ready_from_store(
            manifest,
            assignment,
            use_imported_peer_inventory=True,
            use_imported_seed_inventory=True,
        )
    except Exception:
        logger.exception(
            f"Chunk-backed llama.cpp preparation failed for {shard.model_card.model_id}"
        )
        return DownloadFailed(
            node_id=node_id,
            shard_metadata=shard,
            error_message=(
                "Chunk-backed model preparation failed; full-model fallback is disabled "
                "for curated distributed manifests."
            ),
        )

    if not result.ready:
        final_plan = getattr(result, "final_plan", None)
        coverage = getattr(final_plan, "coverage", None)
        missing_chunks = getattr(coverage, "missing_chunk_ids", ()) or ()
        return DownloadFailed(
            node_id=node_id,
            shard_metadata=shard,
            error_message=(
                "Chunk-backed model preparation is incomplete; full-model fallback is "
                f"disabled for curated distributed manifests. missing_chunks={len(missing_chunks)}"
            ),
        )

    try:
        materialized = (
            model_distribution.materialize_default_assignment_artifact_from_store(
                manifest,
                assignment,
            )
        )
        set_custom_model_local_path(
            shard.model_card.model_id,
            materialized.output_path,
        )
    except Exception:
        logger.exception(
            f"Chunk-backed llama.cpp materialization failed for {shard.model_card.model_id}"
        )
        return DownloadFailed(
            node_id=node_id,
            shard_metadata=shard,
            error_message=(
                "Chunk-backed model materialization failed; full-model fallback is disabled "
                "for curated distributed manifests."
            ),
        )

    return DownloadCompleted(
        node_id=node_id,
        shard_metadata=shard,
        model_directory=str(materialized.output_path),
        total=shard.model_card.storage_size,
        read_only=False,
    )


def _release_chunk_backed_assignment_cache(
    node_id: NodeId,
    shard,
    *,
    other_shards: list[object] | None = None,
) -> bool:
    if shard.model_card.inference_backend != InferenceBackend.LlamaCpp:
        return False

    try:
        model_distribution = importlib.import_module(
            "cai_compute_chain.model_distribution"
        )
    except Exception:
        return False

    manifest = model_distribution.select_model_package_manifest_for_model(
        str(shard.model_card.model_id)
    )
    if manifest is None:
        return False

    protected_chunk_ids: set[str] = set()
    same_model_active = False
    for other_shard in other_shards or []:
        other_model_id = getattr(getattr(other_shard, "model_card", None), "model_id", None)
        if str(other_model_id) != str(shard.model_card.model_id):
            continue
        same_model_active = True
        for chunk in manifest.required_chunks_for_layers(
            int(other_shard.start_layer),
            int(other_shard.end_layer),
        ):
            protected_chunk_ids.add(str(chunk.chunk_id))

    released = model_distribution.release_assignment_cache_policy_from_store(
        manifest,
        model_distribution.ModelShardAssignment(
            start_layer=shard.start_layer,
            end_layer=shard.end_layer,
            device_rank=shard.device_rank,
            world_size=shard.world_size,
            node_id=str(node_id),
        ),
        protected_chunk_ids=protected_chunk_ids,
    )

    cleaned_up_materialized = False
    if not same_model_active:
        try:
            artifact_id = model_distribution.select_default_materialized_artifact_id(
                manifest
            )
            if artifact_id is not None:
                materialized_path = Path(
                    model_distribution.materialized_artifact_path(manifest, artifact_id)
                )
                custom_model_path = get_custom_model_local_path(
                    shard.model_card.model_id
                )
                if custom_model_path is not None and custom_model_path == materialized_path:
                    delete_custom_model_local_path(shard.model_card.model_id)
                    materialized_path.unlink(missing_ok=True)
                    cleaned_up_materialized = True
        except Exception:
            cleaned_up_materialized = False

    return bool(released) or cleaned_up_materialized


class Worker:
    def __init__(
        self,
        node_id: NodeId,
        *,
        event_receiver: Receiver[IndexedEvent],
        event_sender: Sender[Event],
        # This is for requesting updates. It doesn't need to be a general command sender right now,
        # but I think it's the correct way to be thinking about commands
        command_sender: Sender[ForwarderCommand],
        download_command_sender: Sender[ForwarderDownloadCommand],
        api_port: int,
    ):
        self.node_id: NodeId = node_id
        self.event_receiver = event_receiver
        self.event_sender = event_sender
        self.command_sender = command_sender
        self.download_command_sender = download_command_sender
        self.api_port = api_port

        self.state: State = State()
        self.runners: dict[RunnerId, RunnerSupervisor] = {}
        self._tg: TaskGroup = TaskGroup()

        self._system_id = SystemId()

        # Buffer for input image chunks (for image editing)
        self.input_chunk_buffer: dict[CommandId, dict[int, InputImageChunk]] = {}
        self.input_chunk_counts: dict[CommandId, int] = {}
        self.image_cache: dict[Base64ImageHash, Base64Image] = {}

        self._download_backoff: KeyedBackoff[ModelId] = KeyedBackoff(base=0.5, cap=10.0)
        self._instance_backoff: KeyedBackoff[InstanceId] = KeyedBackoff(
            base=0.5, cap=10.0
        )
        self._stopped: anyio.Event = anyio.Event()

    async def run(self):
        logger.info("Starting Worker")

        info_send, info_recv = channel[GatheredInfo]()
        info_gatherer: InfoGatherer = InfoGatherer(
            info_send,
            api_port=self.api_port,
            node_id=str(self.node_id),
        )

        try:
            async with self._tg as tg:
                tg.start_soon(info_gatherer.run)
                tg.start_soon(self._forward_info, info_recv)
                tg.start_soon(self.plan_step)
                tg.start_soon(self._event_applier)
                tg.start_soon(self._poll_connection_updates)
                tg.start_soon(self._sync_chunk_inventory_updates)
                tg.start_soon(self._run_cai_owned_transport_runtime_loop)
        finally:
            # Actual shutdown code - waits for all tasks to complete before executing.
            logger.info("Stopping Worker")
            self.event_sender.close()
            self.command_sender.close()
            self.download_command_sender.close()
            for runner in self.runners.values():
                with contextlib.suppress(Exception):
                    _release_chunk_backed_assignment_cache(
                        self.node_id,
                        runner.shard_metadata,
                    )
                runner.shutdown()
            self._stopped.set()

    async def _forward_info(self, recv: Receiver[GatheredInfo]):
        with recv as info_stream:
            async for info in info_stream:
                try:
                    await self.event_sender.send(
                        NodeGatheredInfo(
                            node_id=self.node_id,
                            when=str(datetime.now(tz=timezone.utc)),
                            info=info,
                        )
                    )
                except (BrokenResourceError, ClosedResourceError):
                    logger.debug(
                        "worker info forwarder stopping because event channel is closed"
                    )
                    return

    async def _event_applier(self):
        with self.event_receiver as events:
            async for event in events:
                # 2. for each event, apply it to the state
                self.state = apply(self.state, event=event)
                event = event.event

                if isinstance(event, InstanceDeleted):
                    self._instance_backoff.reset(event.instance_id)

                # Buffer input image chunks for image editing
                if isinstance(event, InputChunkReceived):
                    cmd_id = event.command_id
                    if cmd_id not in self.input_chunk_buffer:
                        self.input_chunk_buffer[cmd_id] = {}
                        self.input_chunk_counts[cmd_id] = event.chunk.total_chunks

                    self.input_chunk_buffer[cmd_id][event.chunk.chunk_index] = (
                        event.chunk
                    )

                if isinstance(event, CustomModelCardAdded):
                    await event.model_card.save_to_custom_dir()
                    add_to_card_cache(event.model_card)

                if isinstance(event, CustomModelCardDeleted):
                    await delete_custom_card(event.model_id)

    async def plan_step(self):
        while True:
            await anyio.sleep(0.1)
            task: Task | None = plan(
                self.node_id,
                self.runners,
                self.state.downloads,
                self.state.instances,
                self.state.runners,
                self.state.tasks,
                self.input_chunk_buffer,
                self._instance_backoff,
                self._download_backoff,
            )
            if task is None:
                continue

            if isinstance(task, CreateRunner):
                iid = task.instance_id
                if self._instance_backoff.attempts(iid) >= CAI_MAX_INSTANCE_RETRIES:
                    logger.warning(
                        f"Instance {iid} exceeded {CAI_MAX_INSTANCE_RETRIES} retries, requesting deletion"
                    )
                    await self.command_sender.send(
                        ForwarderCommand(
                            origin=self._system_id,
                            command=DeleteInstance(instance_id=iid),
                        )
                    )
                    continue

            logger.info(f"Worker plan: {task.__class__.__name__}")
            assert task.task_status
            await self.event_sender.send(TaskCreated(task_id=task.task_id, task=task))

            # lets not kill the worker if a runner is unresponsive
            match task:
                case CreateRunner():
                    self._create_supervisor(task)
                    self._instance_backoff.record_attempt(task.instance_id)
                    await self.event_sender.send(
                        TaskStatusUpdated(
                            task_id=task.task_id, task_status=TaskStatus.Complete
                        )
                    )
                case DownloadModel(shard_metadata=shard):
                    model_id = shard.model_card.model_id
                    self._download_backoff.record_attempt(model_id)

                    chunk_backed_download = await to_thread.run_sync(
                        lambda: _try_prepare_chunk_backed_llama_cpp_download(
                            self.node_id,
                            shard,
                            api_port=self.api_port,
                        )
                    )
                    if chunk_backed_download is not None:
                        await self.event_sender.send(
                            NodeDownloadProgress(
                                download_progress=chunk_backed_download
                            )
                        )
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id,
                                task_status=(
                                    TaskStatus.Complete
                                    if isinstance(
                                        chunk_backed_download,
                                        DownloadCompleted,
                                    )
                                    else TaskStatus.Failed
                                ),
                            )
                        )
                        continue

                    found_path = await to_thread.run_sync(
                        resolve_existing_model, model_id
                    )
                    if found_path is not None:
                        logger.info(f"Model {model_id} found at {found_path}")
                        await self.event_sender.send(
                            NodeDownloadProgress(
                                download_progress=DownloadCompleted(
                                    node_id=self.node_id,
                                    shard_metadata=shard,
                                    model_directory=str(found_path),
                                    total=shard.model_card.storage_size,
                                    read_only=is_read_only_model_dir(found_path),
                                )
                            )
                        )
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id,
                                task_status=TaskStatus.Complete,
                            )
                        )
                    else:
                        await self.download_command_sender.send(
                            ForwarderDownloadCommand(
                                origin=self._system_id,
                                command=StartDownload(
                                    target_node_id=self.node_id,
                                    shard_metadata=shard,
                                ),
                            )
                        )
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id,
                                task_status=TaskStatus.Running,
                            )
                        )
                case Shutdown(runner_id=runner_id):
                    runner = self.runners.pop(runner_id)
                    other_shards = [
                        other_runner.shard_metadata
                        for other_runner in self.runners.values()
                    ]
                    try:
                        with fail_after(3):
                            await runner.start_task(task)
                    except TimeoutError:
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id, task_status=TaskStatus.TimedOut
                            )
                        )
                    finally:
                        with contextlib.suppress(Exception):
                            _release_chunk_backed_assignment_cache(
                                self.node_id,
                                runner.shard_metadata,
                                other_shards=other_shards,
                            )
                        runner.shutdown()
                case CancelTask(
                    cancelled_task_id=cancelled_task_id, runner_id=runner_id
                ):
                    await self.runners[runner_id].cancel_task(cancelled_task_id)
                    await self.event_sender.send(
                        TaskStatusUpdated(
                            task_id=task.task_id, task_status=TaskStatus.Complete
                        )
                    )
                case ImageEdits() if task.task_params.total_input_chunks > 0:
                    # Assemble image from chunks and inject into task
                    cmd_id = task.command_id
                    chunks = self.input_chunk_buffer.get(cmd_id, {})
                    assembled = "".join(chunks[i].data for i in range(len(chunks)))
                    logger.info(
                        f"Assembled input image from {len(chunks)} chunks, "
                        f"total size: {len(assembled)} bytes"
                    )
                    # Create modified task with assembled image data
                    modified_task = ImageEdits(
                        task_id=task.task_id,
                        command_id=task.command_id,
                        instance_id=task.instance_id,
                        task_status=task.task_status,
                        task_params=ImageEditsTaskParams(
                            image_data=assembled,
                            total_input_chunks=task.task_params.total_input_chunks,
                            prompt=task.task_params.prompt,
                            model=task.task_params.model,
                            n=task.task_params.n,
                            quality=task.task_params.quality,
                            output_format=task.task_params.output_format,
                            response_format=task.task_params.response_format,
                            size=task.task_params.size,
                            image_strength=task.task_params.image_strength,
                            bench=task.task_params.bench,
                            stream=task.task_params.stream,
                            partial_images=task.task_params.partial_images,
                            advanced_params=task.task_params.advanced_params,
                        ),
                    )
                    # Cleanup buffers
                    if cmd_id in self.input_chunk_buffer:
                        del self.input_chunk_buffer[cmd_id]
                    if cmd_id in self.input_chunk_counts:
                        del self.input_chunk_counts[cmd_id]
                    await self._start_runner_task(modified_task)

                case TextGeneration() if (
                    task.task_params.image_hashes
                    or task.task_params.total_input_chunks > 0
                ):
                    cmd_id = task.command_id
                    by_index: dict[int, Base64Image] = {}

                    for idx, h in task.task_params.image_hashes.items():
                        assert h in self.image_cache
                        by_index[idx] = self.image_cache[h]

                    if task.task_params.total_input_chunks > 0:
                        chunk_buffer = self.input_chunk_buffer.get(cmd_id, {})
                        per_image: defaultdict[int, list[InputImageChunk]] = (
                            defaultdict(list)
                        )
                        for chunk in chunk_buffer.values():
                            per_image[chunk.image_index].append(chunk)
                        for img_idx in sorted(per_image):
                            sorted_chunks = sorted(
                                per_image[img_idx], key=lambda c: c.chunk_index
                            )
                            img = Base64Image("".join(c.data for c in sorted_chunks))
                            self.image_cache[
                                Base64ImageHash(
                                    hashlib.sha256(img.encode("ascii")).hexdigest()
                                )
                            ] = img
                            by_index[img_idx] = img
                        logger.info(
                            f"Assembled {len(per_image)} VLM image(s) "
                            f"from {len(chunk_buffer)} chunks"
                        )

                    resolved_images = [
                        Base64Image(by_index[i]) for i in sorted(by_index)
                    ]
                    modified_task = task.model_copy(
                        update={
                            "task_params": task.task_params.model_copy(
                                update={"images": resolved_images}
                            )
                        }
                    )
                    if cmd_id in self.input_chunk_buffer:
                        del self.input_chunk_buffer[cmd_id]
                    if cmd_id in self.input_chunk_counts:
                        del self.input_chunk_counts[cmd_id]
                    await self._start_runner_task(modified_task)
                case LoadModel(instance_id=instance_id):
                    if (instance := self.state.instances.get(instance_id)) is not None:
                        model_id = instance.shard_assignments.model_id
                        self._download_backoff.reset(model_id)

                    await self._start_runner_task(task)
                case task:
                    await self._start_runner_task(task)

    async def shutdown(self):
        self._tg.cancel_tasks()
        await self._stopped.wait()

    async def _start_runner_task(self, task: Task):
        if (instance := self.state.instances.get(task.instance_id)) is not None:
            await self.runners[
                instance.shard_assignments.node_to_runner[self.node_id]
            ].start_task(task)

    def _create_supervisor(self, task: CreateRunner) -> RunnerSupervisor:
        """Creates and stores a new AssignedRunner with initial downloading status."""
        runner = RunnerSupervisor.create(
            bound_instance=task.bound_instance,
            event_sender=self.event_sender.clone(),
        )
        self.runners[task.bound_instance.bound_runner_id] = runner
        self._tg.start_soon(runner.run)
        return runner

    async def _poll_connection_updates(self):
        while True:
            edges = set(
                conn.edge for conn in self.state.topology.out_edges(self.node_id)
            )
            conns: defaultdict[NodeId, set[tuple[str, int]]] = defaultdict(set)
            async for ip, port, nid in check_reachable(
                self.state.topology,
                self.node_id,
                self.state.node_network,
                self.state.node_identities,
                api_port=self.api_port,
                overlay_peers=self.state.overlay_peers,
                include_overlay_fallback=False,
            ):
                target = (ip, port)
                if target in conns[nid]:
                    continue
                conns[nid].add(target)
                edge = SocketConnection(
                    sink_multiaddr=Multiaddr(address=f"/ip4/{ip}/tcp/{port}")
                    if "." in ip
                    else Multiaddr(address=f"/ip6/{ip}/tcp/{port}"),
                )
                if edge not in edges:
                    logger.debug(f"ping discovered {edge=}")
                    await self.event_sender.send(
                        TopologyEdgeCreated(
                            conn=Connection(source=self.node_id, sink=nid, edge=edge)
                        )
                    )

            for conn in self.state.topology.out_edges(self.node_id):
                if not isinstance(conn.edge, SocketConnection):
                    continue
                if (
                    conn.sink not in conns
                    or (
                        conn.edge.sink_multiaddr.ip_address,
                        conn.edge.sink_multiaddr.port,
                    )
                    not in conns[conn.sink]
                ):
                    logger.debug(f"ping failed to discover {conn=}")
                    await self.event_sender.send(TopologyEdgeDeleted(conn=conn))

            await anyio.sleep(10)

    async def _sync_chunk_inventory_updates(self):
        while True:
            try:
                result = await to_thread.run_sync(
                    lambda: _sync_chunk_backed_peer_inventories(
                        api_port=self.api_port,
                    )
                )
                if result is not None and (
                    getattr(result, "imported_payloads", 0) > 0
                    or getattr(result, "pruned_payloads", 0) > 0
                ):
                    logger.debug(
                        "chunk inventory sync updated peer cache state: "
                        f"imported={getattr(result, 'imported_payloads', 0)} "
                        f"pruned={getattr(result, 'pruned_payloads', 0)}"
                    )
                seed_result = await to_thread.run_sync(
                    _sync_chunk_backed_seed_inventories
                )
                if seed_result is not None and (
                    getattr(seed_result, "imported_payloads", 0) > 0
                    or getattr(seed_result, "pruned_payloads", 0) > 0
                ):
                    logger.debug(
                        "chunk inventory sync updated storage seed state: "
                        f"imported={getattr(seed_result, 'imported_payloads', 0)} "
                        f"pruned={getattr(seed_result, 'pruned_payloads', 0)}"
                    )
                prefetch_result = await to_thread.run_sync(
                    lambda: _prefetch_chunk_backed_default_chunks(self.node_id)
                )
                if prefetch_result is not None and (
                    getattr(prefetch_result, "manifests_prefetched", 0) > 0
                    or getattr(prefetch_result, "processed_tasks", 0) > 0
                ):
                    logger.debug(
                        "chunk inventory background prefetch warmed default chunks: "
                        f"manifests_prefetched={getattr(prefetch_result, 'manifests_prefetched', 0)} "
                        f"processed_tasks={getattr(prefetch_result, 'processed_tasks', 0)}"
                    )
                bootstrap_prefetch_result = await to_thread.run_sync(
                    lambda: _prefetch_chunk_backed_bootstrap_chunks(self.node_id)
                )
                if bootstrap_prefetch_result is not None and (
                    getattr(bootstrap_prefetch_result, "manifests_prefetched", 0) > 0
                    or getattr(bootstrap_prefetch_result, "processed_tasks", 0) > 0
                ):
                    logger.debug(
                        "chunk inventory background bootstrap prefetch warmed likely shard chunks: "
                        f"manifests_prefetched={getattr(bootstrap_prefetch_result, 'manifests_prefetched', 0)} "
                        f"processed_tasks={getattr(bootstrap_prefetch_result, 'processed_tasks', 0)}"
                    )
                instance_values = list(self.state.instances.values())
                if instance_values:
                    remembered_instance_hint_result = await to_thread.run_sync(
                        lambda: _remember_chunk_backed_instance_shard_hints(
                            self.node_id,
                            instance_values,
                        )
                    )
                    if remembered_instance_hint_result is not None and (
                        getattr(remembered_instance_hint_result, "records_upserted", 0) > 0
                        or getattr(remembered_instance_hint_result, "records_pruned", 0) > 0
                    ):
                        logger.debug(
                            "chunk inventory updated instance-derived shard hints: "
                            f"upserted={getattr(remembered_instance_hint_result, 'records_upserted', 0)} "
                            f"pruned={getattr(remembered_instance_hint_result, 'records_pruned', 0)} "
                            f"stored={getattr(remembered_instance_hint_result, 'stored_records', 0)}"
                        )
                    instance_hint_prefetch_result = await to_thread.run_sync(
                        lambda: _prefetch_chunk_backed_instance_shard_hints(
                            self.node_id,
                            instance_values,
                        )
                    )
                    if instance_hint_prefetch_result is not None and (
                        getattr(instance_hint_prefetch_result, "manifests_prefetched", 0) > 0
                        or getattr(instance_hint_prefetch_result, "processed_tasks", 0) > 0
                    ):
                        logger.debug(
                            "chunk inventory placement-hint prefetch warmed assigned chunk ranges: "
                            f"manifests_prefetched={getattr(instance_hint_prefetch_result, 'manifests_prefetched', 0)} "
                            f"processed_tasks={getattr(instance_hint_prefetch_result, 'processed_tasks', 0)}"
                        )
                shard_metadatas = [runner.shard_metadata for runner in self.runners.values()]
                if shard_metadatas:
                    remembered_hint_result = await to_thread.run_sync(
                        lambda: _remember_chunk_backed_local_shard_hints(
                            self.node_id,
                            shard_metadatas,
                        )
                    )
                    if remembered_hint_result is not None and (
                        getattr(remembered_hint_result, "records_upserted", 0) > 0
                        or getattr(remembered_hint_result, "records_pruned", 0) > 0
                    ):
                        logger.debug(
                            "chunk inventory updated recent shard hints: "
                            f"upserted={getattr(remembered_hint_result, 'records_upserted', 0)} "
                            f"pruned={getattr(remembered_hint_result, 'records_pruned', 0)} "
                            f"stored={getattr(remembered_hint_result, 'stored_records', 0)}"
                        )
                hint_prefetch_result = await to_thread.run_sync(
                    lambda: _prefetch_chunk_backed_local_shard_hints(
                        self.node_id,
                        shard_metadatas,
                    )
                )
                if hint_prefetch_result is not None and (
                    getattr(hint_prefetch_result, "manifests_prefetched", 0) > 0
                    or getattr(hint_prefetch_result, "processed_tasks", 0) > 0
                ):
                    logger.debug(
                        "chunk inventory hint prefetch warmed node-specific chunk ranges: "
                        f"manifests_prefetched={getattr(hint_prefetch_result, 'manifests_prefetched', 0)} "
                        f"processed_tasks={getattr(hint_prefetch_result, 'processed_tasks', 0)}"
                    )
                elif not shard_metadatas:
                    recent_hint_prefetch_result = await to_thread.run_sync(
                        lambda: _prefetch_chunk_backed_recent_shard_hints(self.node_id)
                    )
                    if recent_hint_prefetch_result is not None and (
                        getattr(recent_hint_prefetch_result, "manifests_prefetched", 0) > 0
                        or getattr(recent_hint_prefetch_result, "processed_tasks", 0) > 0
                    ):
                        logger.debug(
                            "chunk inventory recent-hint prefetch warmed remembered shard ranges: "
                            f"manifests_prefetched={getattr(recent_hint_prefetch_result, 'manifests_prefetched', 0)} "
                            f"processed_tasks={getattr(recent_hint_prefetch_result, 'processed_tasks', 0)}"
                        )
            except Exception:
                pass
            await anyio.sleep(15)

    async def _run_cai_owned_transport_runtime_loop(self):
        if not _cai_owned_transport_runtime_enabled():
            logger.info("CAI-owned transport shard runtime loop is disabled")
            return

        try:
            runtime = importlib.import_module("cai_compute_chain.cai_owned_runtime")
            wallet_model = importlib.import_module("cai_compute_chain.model")
            adapter = _cai_owned_transport_adapter(self.node_id)
        except Exception:
            logger.exception("Unable to initialize CAI-owned transport shard runtime")
            return

        runtime_id = _cai_owned_transport_runtime_id(self.node_id)
        idle_sleep = _env_float("CAI_OWNED_TRANSPORT_POLL_SECONDS", 1.0, minimum=0.05)
        active_sleep = _env_float(
            "CAI_OWNED_TRANSPORT_ACTIVE_POLL_SECONDS",
            0.05,
            minimum=0.0,
        )
        disabled_sleep = _env_float(
            "CAI_OWNED_TRANSPORT_DISABLED_POLL_SECONDS",
            5.0,
            minimum=0.25,
        )
        error_sleep = _env_float(
            "CAI_OWNED_TRANSPORT_ERROR_POLL_SECONDS",
            3.0,
            minimum=0.25,
        )
        logger.info("Starting CAI-owned transport shard runtime loop")
        await _ensure_cai_owned_transport_startup_self_test(
            runtime=runtime,
            wallet_model=wallet_model,
            adapter=adapter,
        )
        self._cai_owned_transport_next_self_test_at = (
            time.monotonic() + _cai_owned_transport_self_test_retry_seconds()
        )

        while True:
            try:
                if (
                    _cai_owned_transport_runtime_requires_worker_enabled()
                    and not await to_thread.run_sync(_local_cai_worker_enabled)
                ):
                    await anyio.sleep(disabled_sleep)
                    continue

                now = time.monotonic()
                next_self_test_at = float(
                    getattr(self, "_cai_owned_transport_next_self_test_at", 0.0)
                )
                if now >= next_self_test_at:
                    self._cai_owned_transport_next_self_test_at = (
                        now + _cai_owned_transport_self_test_retry_seconds()
                    )
                    await _ensure_cai_owned_transport_startup_self_test(
                        runtime=runtime,
                        wallet_model=wallet_model,
                        adapter=adapter,
                    )

                policy = wallet_model.WalletPolicy()
                config = runtime.CaiOwnedShardRuntimeConfig(
                    node_id=str(self.node_id),
                    runtime_id=runtime_id,
                    output_peer_cai_urls_by_node=(
                        _cai_owned_transport_peer_urls_by_node(
                            self.state,
                            local_node_id=self.node_id,
                            api_port=self.api_port,
                        )
                    ),
                    output_forward_timeout_sec=_env_float(
                        "CAI_OWNED_TRANSPORT_OUTPUT_FORWARD_TIMEOUT_SECONDS",
                        5.0,
                        minimum=0.1,
                    ),
                    max_concurrent_batches=_env_int(
                        "CAI_OWNED_TRANSPORT_MAX_CONCURRENT_BATCHES",
                        1,
                    ),
                    max_payload_size_bytes=_env_int(
                        "CAI_OWNED_TRANSPORT_MAX_PAYLOAD_BYTES",
                        16 * 1024 * 1024,
                    ),
                    lease_seconds=_env_float(
                        "CAI_OWNED_TRANSPORT_BATCH_LEASE_SECONDS",
                        60.0,
                        minimum=1.0,
                    ),
                    max_attempts=_env_int(
                        "CAI_OWNED_TRANSPORT_MAX_BATCH_ATTEMPTS",
                        3,
                    ),
                    require_production_llm_handoff=(
                        _cai_owned_transport_runtime_requires_production_handoff()
                    ),
                    policy=policy,
                )
                result = await to_thread.run_sync(
                    lambda: runtime.run_cai_owned_shard_runtime_once(
                        config,
                        adapter,
                    )
                )
                status = str(result.get("status") or "").strip()
                if status == "processed":
                    completion = result.get("completion")
                    batch_id = (
                        completion.get("batchId")
                        if isinstance(completion, dict)
                        else None
                    )
                    logger.info(
                        "CAI-owned transport processed shard batch {}",
                        batch_id or "<unknown>",
                    )
                    await anyio.sleep(active_sleep)
                    continue
                if status in {"failed", "retry_scheduled"}:
                    logger.warning(
                        "CAI-owned transport runtime reported {}: {}",
                        status,
                        result,
                    )
                    await anyio.sleep(error_sleep if status == "failed" else idle_sleep)
                    continue
                await anyio.sleep(idle_sleep)
            except Exception:
                logger.exception("CAI-owned transport shard runtime loop iteration failed")
                await anyio.sleep(error_sleep)


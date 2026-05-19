# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from cai.shared.models.model_cards import ModelCard, ModelId
from cai.shared.types.memory import Memory
from cai.shared.types.worker.instances import Instance
from cai.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata, Sharding


DEFAULT_PRIVATE_NETWORK_MODEL_IDS = frozenset(
    {ModelId("cai-network/Qwen3-0.6B-GGUF")}
)
PRIVATE_NETWORK_DOWNLOAD_REPO_IDS = {
    ModelId("cai-network/Qwen3-0.6B-GGUF"): ModelId("Qwen/Qwen3-0.6B-GGUF"),
}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_model_ids(
    name: str,
    *,
    default: frozenset[ModelId] = frozenset(),
) -> frozenset[ModelId]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    parts = [item.strip() for item in raw.replace(";", ",").split(",")]
    return frozenset(ModelId(item) for item in parts if item)


@dataclass(frozen=True)
class PrivateNetworkModelPolicy:
    model_ids: frozenset[ModelId]
    minimum_nodes: int = 1
    require_pipeline_sharding: bool = True
    disallow_single_node_fallback: bool = True
    allow_single_node_bootstrap: bool = False
    require_filtered_downloads: bool = True
    minimum_ram_headroom: Memory = Memory()
    minimum_pipeline_layers_per_node: int = 2


def _is_private_network_model(
    model_id: ModelId,
    *,
    model_card: ModelCard | None = None,
    private_override: bool = False,
) -> bool:
    if private_override:
        return True
    if model_card is not None and model_card.is_custom:
        return True
    return model_id in get_private_network_model_policy().model_ids


@lru_cache(maxsize=1)
def get_private_network_model_policy() -> PrivateNetworkModelPolicy:
    minimum_nodes_raw = os.getenv("CAI_PRIVATE_NETWORK_MODEL_MIN_NODES", "1").strip()
    try:
        minimum_nodes = max(1, int(minimum_nodes_raw))
    except ValueError:
        minimum_nodes = 1

    minimum_ram_headroom_raw = os.getenv(
        "CAI_PRIVATE_NETWORK_MODEL_MIN_RAM_HEADROOM_MB", "0"
    ).strip()
    try:
        minimum_ram_headroom = Memory.from_mb(
            max(0, float(minimum_ram_headroom_raw))
        )
    except ValueError:
        minimum_ram_headroom = Memory()

    minimum_pipeline_layers_raw = os.getenv(
        "CAI_PRIVATE_NETWORK_MODEL_MIN_PIPELINE_LAYERS_PER_NODE", "2"
    ).strip()
    try:
        minimum_pipeline_layers_per_node = max(1, int(minimum_pipeline_layers_raw))
    except ValueError:
        minimum_pipeline_layers_per_node = 2

    return PrivateNetworkModelPolicy(
        model_ids=_env_model_ids(
            "CAI_PRIVATE_NETWORK_MODEL_IDS",
            default=DEFAULT_PRIVATE_NETWORK_MODEL_IDS,
        ),
        minimum_nodes=minimum_nodes,
        require_pipeline_sharding=_env_flag(
            "CAI_PRIVATE_NETWORK_MODEL_REQUIRE_PIPELINE", True
        ),
        disallow_single_node_fallback=_env_flag(
            "CAI_PRIVATE_NETWORK_MODEL_DISABLE_SINGLE_NODE", True
        ),
        allow_single_node_bootstrap=_env_flag(
            "CAI_PRIVATE_NETWORK_MODEL_ALLOW_BOOTSTRAP_SINGLE_NODE", False
        ),
        require_filtered_downloads=_env_flag(
            "CAI_PRIVATE_NETWORK_MODEL_FILTERED_DOWNLOADS", True
        ),
        minimum_ram_headroom=minimum_ram_headroom,
        minimum_pipeline_layers_per_node=minimum_pipeline_layers_per_node,
    )


def private_network_model_effective_min_nodes(
    model_id: ModelId,
    *,
    model_card: ModelCard | None = None,
    private_override: bool = False,
    available_nodes: int | None = None,
) -> int:
    policy = get_private_network_model_policy()
    if not _is_private_network_model(
        model_id,
        model_card=model_card,
        private_override=private_override,
    ):
        return 1
    minimum_nodes = max(1, policy.minimum_nodes)
    if (
        policy.allow_single_node_bootstrap
        and available_nodes is not None
        and 0 < int(available_nodes) < minimum_nodes
    ):
        return max(1, int(available_nodes))
    return minimum_nodes


def is_private_network_model(
    model_id: ModelId,
    *,
    model_card: ModelCard | None = None,
    private_override: bool = False,
) -> bool:
    return _is_private_network_model(
        model_id,
        model_card=model_card,
        private_override=private_override,
    )


def private_network_download_repo_id(model_id: ModelId) -> ModelId:
    return PRIVATE_NETWORK_DOWNLOAD_REPO_IDS.get(ModelId(model_id), ModelId(model_id))


def private_network_model_ram_headroom(
    model_id: ModelId,
    *,
    model_card: ModelCard | None = None,
    private_override: bool = False,
) -> Memory:
    policy = get_private_network_model_policy()
    if not _is_private_network_model(
        model_id,
        model_card=model_card,
        private_override=private_override,
    ):
        return Memory()
    return policy.minimum_ram_headroom


def private_network_model_min_pipeline_layers_per_node(
    model_id: ModelId,
    *,
    model_card: ModelCard | None = None,
    private_override: bool = False,
) -> int:
    policy = get_private_network_model_policy()
    if not _is_private_network_model(
        model_id,
        model_card=model_card,
        private_override=private_override,
    ):
        return 1
    return max(1, policy.minimum_pipeline_layers_per_node)


def enforce_private_network_model_request(
    model_id: ModelId,
    sharding: Sharding,
    min_nodes: int,
    *,
    model_card: ModelCard | None = None,
    private_override: bool = False,
    available_nodes: int | None = None,
) -> tuple[Sharding, int]:
    policy = get_private_network_model_policy()
    if not _is_private_network_model(
        model_id,
        model_card=model_card,
        private_override=private_override,
    ):
        return sharding, min_nodes

    if policy.require_pipeline_sharding and sharding != Sharding.Pipeline:
        raise ValueError(
            f"Private network model {model_id} requires pipeline sharding."
        )

    effective_min_nodes = private_network_model_effective_min_nodes(
        model_id,
        model_card=model_card,
        private_override=private_override,
        available_nodes=available_nodes,
    )
    return Sharding.Pipeline, max(min_nodes, effective_min_nodes)


def validate_private_network_instance(instance: Instance) -> None:
    shard_assignments = instance.shard_assignments
    model_id = shard_assignments.model_id
    shard_cards = [
        shard.model_card
        for shard in shard_assignments.runner_to_shard.values()
        if getattr(shard, "model_card", None) is not None
    ]
    model_card = shard_cards[0] if shard_cards else None
    policy = get_private_network_model_policy()
    if not _is_private_network_model(model_id, model_card=model_card):
        return

    node_count = len(shard_assignments.node_to_runner)
    if node_count < policy.minimum_nodes:
        if not (policy.allow_single_node_bootstrap and node_count == 1):
            raise ValueError(
                f"Private network model {model_id} requires at least "
                f"{policy.minimum_nodes} nodes, got {node_count}."
            )

    for shard in shard_assignments.runner_to_shard.values():
        if policy.require_pipeline_sharding and not isinstance(
            shard, PipelineShardMetadata
        ):
            raise ValueError(
                f"Private network model {model_id} requires pipeline shard metadata."
            )
        if (
            isinstance(shard, PipelineShardMetadata)
            and (shard.end_layer - shard.start_layer)
            < policy.minimum_pipeline_layers_per_node
        ):
            raise ValueError(
                f"Private network model {model_id} requires at least "
                f"{policy.minimum_pipeline_layers_per_node} pipeline layer(s) per node."
            )


def requires_filtered_downloads(shard: ShardMetadata) -> bool:
    policy = get_private_network_model_policy()
    return (
        _is_private_network_model(shard.model_card.model_id, model_card=shard.model_card)
        and policy.require_filtered_downloads
    )


def validate_private_network_shard_download(
    shard: ShardMetadata, *, skip_download: bool = False
) -> None:
    if skip_download:
        return

    policy = get_private_network_model_policy()
    model_id = shard.model_card.model_id
    if not _is_private_network_model(model_id, model_card=shard.model_card):
        return

    if policy.require_pipeline_sharding and not isinstance(
        shard, PipelineShardMetadata
    ):
        raise ValueError(
            f"Private network model {model_id} requires pipeline shard downloads."
        )

    if policy.disallow_single_node_fallback and shard.world_size < policy.minimum_nodes:
        if policy.allow_single_node_bootstrap and shard.world_size == 1:
            return
        raise ValueError(
            f"Private network model {model_id} requires at least "
            f"{policy.minimum_nodes} distributed shards."
        )


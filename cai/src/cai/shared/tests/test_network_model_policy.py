# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import pytest

from cai.shared.models.model_cards import ModelCard, ModelId, ModelTask
from cai.shared.network_model_policy import (
    DEFAULT_PRIVATE_NETWORK_MODEL_IDS,
    enforce_private_network_model_request,
    get_private_network_model_policy,
    private_network_download_repo_id,
    private_network_model_effective_min_nodes,
    private_network_model_ram_headroom,
    validate_private_network_instance,
)
from cai.shared.types.common import NodeId
from cai.shared.types.memory import Memory
from cai.shared.types.worker.instances import InstanceId, MlxRingInstance
from cai.shared.types.worker.runners import RunnerId, ShardAssignments
from cai.shared.types.worker.shards import PipelineShardMetadata, Sharding


def _model_card(model_id: str = "private/test-model") -> ModelCard:
    return ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_mb(100),
        n_layers=8,
        hidden_size=1024,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )


@pytest.fixture(autouse=True)
def _reset_policy_cache(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CAI_PRIVATE_NETWORK_MODEL_IDS", raising=False)
    monkeypatch.delenv("CAI_PRIVATE_NETWORK_MODEL_MIN_NODES", raising=False)
    monkeypatch.delenv("CAI_PRIVATE_NETWORK_MODEL_REQUIRE_PIPELINE", raising=False)
    monkeypatch.delenv("CAI_PRIVATE_NETWORK_MODEL_DISABLE_SINGLE_NODE", raising=False)
    monkeypatch.delenv(
        "CAI_PRIVATE_NETWORK_MODEL_ALLOW_BOOTSTRAP_SINGLE_NODE", raising=False
    )
    monkeypatch.delenv("CAI_PRIVATE_NETWORK_MODEL_FILTERED_DOWNLOADS", raising=False)
    monkeypatch.delenv(
        "CAI_PRIVATE_NETWORK_MODEL_MIN_RAM_HEADROOM_MB", raising=False
    )
    get_private_network_model_policy.cache_clear()
    yield
    get_private_network_model_policy.cache_clear()


def test_enforce_private_network_model_request_honors_explicit_two_node_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAI_PRIVATE_NETWORK_MODEL_IDS", "private/test-model")
    monkeypatch.setenv("CAI_PRIVATE_NETWORK_MODEL_MIN_NODES", "2")
    get_private_network_model_policy.cache_clear()

    sharding, min_nodes = enforce_private_network_model_request(
        ModelId("private/test-model"), Sharding.Pipeline, 1
    )
    assert sharding == Sharding.Pipeline
    assert min_nodes == 2

    with pytest.raises(ValueError, match="requires pipeline sharding"):
        enforce_private_network_model_request(
            ModelId("private/test-model"), Sharding.Tensor, 2
        )


def test_default_private_network_model_is_registered_without_env() -> None:
    policy = get_private_network_model_policy()

    assert DEFAULT_PRIVATE_NETWORK_MODEL_IDS.issubset(policy.model_ids)

    sharding, min_nodes = enforce_private_network_model_request(
        ModelId("cai-network/Qwen3-0.6B-GGUF"),
        Sharding.Pipeline,
        1,
    )

    assert sharding == Sharding.Pipeline
    assert min_nodes == 1


def test_private_network_download_repo_uses_public_execution_model() -> None:
    assert private_network_download_repo_id(
        ModelId("cai-network/Qwen3-0.6B-GGUF")
    ) == ModelId("Qwen/Qwen3-0.6B-GGUF")
    assert private_network_download_repo_id(
        ModelId("Qwen/Qwen2.5-0.5B-Instruct-GGUF")
    ) == ModelId("Qwen/Qwen2.5-0.5B-Instruct-GGUF")


def test_private_network_model_allows_single_node_only_for_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAI_PRIVATE_NETWORK_MODEL_IDS", "private/test-model")
    monkeypatch.setenv("CAI_PRIVATE_NETWORK_MODEL_MIN_NODES", "2")
    monkeypatch.setenv("CAI_PRIVATE_NETWORK_MODEL_ALLOW_BOOTSTRAP_SINGLE_NODE", "true")
    get_private_network_model_policy.cache_clear()

    assert (
        private_network_model_effective_min_nodes(
            ModelId("private/test-model"),
            available_nodes=1,
        )
        == 1
    )
    assert (
        private_network_model_effective_min_nodes(
            ModelId("private/test-model"),
            available_nodes=3,
        )
        == 2
    )

    sharding, min_nodes = enforce_private_network_model_request(
        ModelId("private/test-model"),
        Sharding.Pipeline,
        1,
        available_nodes=1,
    )

    assert sharding == Sharding.Pipeline
    assert min_nodes == 1


def test_validate_private_network_instance_rejects_single_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAI_PRIVATE_NETWORK_MODEL_IDS", "private/test-model")
    monkeypatch.setenv("CAI_PRIVATE_NETWORK_MODEL_MIN_NODES", "2")
    get_private_network_model_policy.cache_clear()

    runner_id = RunnerId()
    node_id = NodeId()
    card = _model_card()
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=card.n_layers,
        n_layers=card.n_layers,
    )
    instance = MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            runner_to_shard={runner_id: shard},
            node_to_runner={node_id: runner_id},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )

    with pytest.raises(ValueError, match="requires at least 2 nodes"):
        validate_private_network_instance(instance)


def test_private_network_model_ram_headroom_reads_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAI_PRIVATE_NETWORK_MODEL_IDS", "private/test-model")
    monkeypatch.setenv("CAI_PRIVATE_NETWORK_MODEL_MIN_RAM_HEADROOM_MB", "384")
    get_private_network_model_policy.cache_clear()

    assert private_network_model_ram_headroom(ModelId("private/test-model")) == Memory.from_mb(384)
    assert private_network_model_ram_headroom(ModelId("public/test-model")) == Memory()


def test_custom_model_card_is_treated_as_private_without_explicit_env() -> None:
    card = _model_card("custom/test-model").model_copy(update={"is_custom": True})

    sharding, min_nodes = enforce_private_network_model_request(
        card.model_id,
        Sharding.Pipeline,
        1,
        model_card=card,
    )

    assert sharding == Sharding.Pipeline
    assert min_nodes == 1


def test_validate_private_network_instance_allows_single_custom_model_node_by_default() -> None:
    runner_id = RunnerId()
    node_id = NodeId()
    card = _model_card("custom/test-model").model_copy(update={"is_custom": True})
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=card.n_layers,
        n_layers=card.n_layers,
    )
    instance = MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            runner_to_shard={runner_id: shard},
            node_to_runner={node_id: runner_id},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )

    validate_private_network_instance(instance)


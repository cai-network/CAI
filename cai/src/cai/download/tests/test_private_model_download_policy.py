# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, patch

import pytest

from cai.download.download_utils import resolve_allow_patterns
from cai.shared.models.model_cards import (
    InferenceBackend,
    ModelCard,
    ModelId,
    ModelTask,
)
from cai.shared.network_model_policy import (
    get_private_network_model_policy,
    validate_private_network_shard_download,
)
from cai.shared.types.memory import Memory
from cai.shared.types.worker.shards import PipelineShardMetadata


def _card(model_id: str = "private/test-model") -> ModelCard:
    return ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_mb(100),
        n_layers=4,
        hidden_size=256,
        supports_tensor=False,
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
    get_private_network_model_policy.cache_clear()
    yield
    get_private_network_model_policy.cache_clear()


async def test_resolve_allow_patterns_uses_shard_subset_for_private_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAI_PRIVATE_NETWORK_MODEL_IDS", "private/test-model")
    get_private_network_model_policy.cache_clear()

    card = _card()
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=2,
        start_layer=0,
        end_layer=2,
        n_layers=card.n_layers,
    )
    weight_map = {
        "model.layers.0.weight": "model-00001.safetensors",
        "model.layers.1.weight": "model-00002.safetensors",
        "model.layers.2.weight": "model-00003.safetensors",
        "model.layers.3.weight": "model-00004.safetensors",
        "model.embed_tokens.weight": "model-shared.safetensors",
    }

    mock_get_weight_map = AsyncMock(return_value=weight_map)
    with patch(
        "cai.download.download_utils.get_weight_map",
        mock_get_weight_map,
    ):
        patterns = await resolve_allow_patterns(shard)

    mock_get_weight_map.assert_awaited_once_with(ModelId("private/test-model"))
    assert "model-00001.safetensors" in patterns
    assert "model-00002.safetensors" in patterns
    assert "model-00003.safetensors" not in patterns
    assert "model-00004.safetensors" not in patterns
    assert "model-shared.safetensors" in patterns
    assert "*.json" in patterns


def test_validate_private_network_shard_download_rejects_full_single_node() -> None:
    with patch.dict(
        "os.environ",
        {
            "CAI_PRIVATE_NETWORK_MODEL_IDS": "private/test-model",
            "CAI_PRIVATE_NETWORK_MODEL_MIN_NODES": "2",
        },
        clear=False,
    ):
        get_private_network_model_policy.cache_clear()
        card = _card()
        shard = PipelineShardMetadata(
            model_card=card,
            device_rank=0,
            world_size=1,
            start_layer=0,
            end_layer=card.n_layers,
            n_layers=card.n_layers,
        )

        with pytest.raises(ValueError, match="requires at least 2 distributed shards"):
            validate_private_network_shard_download(shard, skip_download=False)


def test_custom_model_shard_download_allows_single_node_by_default() -> None:
    card = _card("custom/test-model").model_copy(update={"is_custom": True})
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=card.n_layers,
        n_layers=card.n_layers,
    )

    validate_private_network_shard_download(shard, skip_download=False)


async def test_resolve_allow_patterns_uses_preferred_gguf_for_llama_backend() -> None:
    card = ModelCard(
        model_id=ModelId("Qwen/Qwen2.5-0.5B-Instruct-GGUF"),
        storage_size=Memory.from_mb(500),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        is_custom=False,
        inference_backend=InferenceBackend.LlamaCpp,
        preferred_filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
    )
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )

    patterns = await resolve_allow_patterns(shard)

    assert patterns == ["qwen2.5-0.5b-instruct-q4_k_m.gguf"]


# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

from fastapi import HTTPException

from cai.api.model_placement_policy import (
    apply_private_network_model_override,
    llama_cpp_layer_range_supported,
    model_card_from_instance,
    validate_llama_cpp_multi_node_sharding,
)
from cai.shared.models.model_cards import ModelCard, ModelId, ModelTask
from cai.shared.types.memory import Memory


def _card(model_id: str = "Qwen/Qwen3-0.6B-GGUF") -> ModelCard:
    card = ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_mb(512),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    return card.model_copy(
        update={
            "gguf_architecture": "qwen3",
            "shard_compatibility": "unsupported_for_sharding",
            "layer_range_supported": False,
        }
    )


def _instance_with_cards(model_id: ModelId, cards: list[ModelCard]):
    return SimpleNamespace(
        shard_assignments=SimpleNamespace(
            model_id=model_id,
            runner_to_shard={
                f"runner-{index}": SimpleNamespace(model_card=card)
                for index, card in enumerate(cards)
            },
        )
    )


def test_private_network_override_marks_public_card_as_custom() -> None:
    result = apply_private_network_model_override(
        _card(),
        private_network_model=True,
    )

    assert result.is_custom is True


def test_model_card_from_instance_requires_all_shards_to_match_instance_model() -> None:
    card = _card()
    instance = _instance_with_cards(card.model_id, [card])

    assert model_card_from_instance(instance) == card

    mismatch = _instance_with_cards(card.model_id, [_card("other/model")])
    assert model_card_from_instance(mismatch) is None


def test_llama_cpp_multi_node_sharding_requires_layer_range_support() -> None:
    card = _card()

    try:
        validate_llama_cpp_multi_node_sharding(card, min_nodes=2)
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "multi-node layer-range placement" in str(exc.detail)
    else:
        raise AssertionError("Expected unsupported multi-node GGUF placement to fail")

    validate_llama_cpp_multi_node_sharding(card, min_nodes=1)

    supported = card.model_copy(
        update={
            "shard_compatibility": "layer_range_supported",
            "layer_range_supported": True,
        }
    )
    assert llama_cpp_layer_range_supported(supported) is True
    validate_llama_cpp_multi_node_sharding(supported, min_nodes=2)

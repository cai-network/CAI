# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from fastapi import HTTPException

from cai.shared.models.model_cards import InferenceBackend, ModelCard
from cai.shared.types.worker.instances import Instance


def apply_private_network_model_override(
    model_card: ModelCard,
    *,
    private_network_model: bool,
) -> ModelCard:
    if private_network_model and not model_card.is_custom:
        return model_card.model_copy(update={"is_custom": True})
    return model_card


def model_card_from_instance(instance: Instance) -> ModelCard | None:
    model_id = instance.shard_assignments.model_id
    cards: list[ModelCard] = []
    for shard in instance.shard_assignments.runner_to_shard.values():
        model_card = getattr(shard, "model_card", None)
        if isinstance(model_card, ModelCard):
            cards.append(model_card)
    if not cards:
        return None
    if any(card.model_id != model_id for card in cards):
        return None
    return cards[0]


def llama_cpp_layer_range_supported(model_card: ModelCard) -> bool:
    if model_card.inference_backend != InferenceBackend.LlamaCpp:
        return True
    return (
        bool(getattr(model_card, "layer_range_supported", False))
        and str(getattr(model_card, "shard_compatibility", "") or "").strip()
        == "layer_range_supported"
    )


def validate_llama_cpp_multi_node_sharding(
    model_card: ModelCard,
    *,
    min_nodes: int,
) -> None:
    if (
        model_card.inference_backend != InferenceBackend.LlamaCpp
        or int(min_nodes) <= 1
        or llama_cpp_layer_range_supported(model_card)
    ):
        return

    architecture = str(getattr(model_card, "gguf_architecture", "") or "unknown").strip()
    compatibility = str(
        getattr(model_card, "shard_compatibility", "") or "unsupported_for_sharding"
    ).strip()
    raise HTTPException(
        status_code=400,
        detail=(
            f"GGUF architecture '{architecture}' is {compatibility}; "
            "multi-node layer-range placement requires a successful "
            "CAI layer-range conformance probe. Use single-node full-model "
            "GGUF mode or add an architecture-specific probe first."
        ),
    )

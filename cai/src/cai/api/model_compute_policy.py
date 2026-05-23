# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from cai_compute_chain.gguf_shard_policy import gguf_shard_compatibility
from cai.shared.models.model_cards import InferenceBackend, ModelCard


def model_card_supported_for_cai_gguf_compute(card: ModelCard) -> bool:
    return (
        card.inference_backend == InferenceBackend.LlamaCpp
        and card.layer_range_supported
        and card.shard_compatibility == "layer_range_supported"
    )


def unsupported_gguf_model_detail(card: ModelCard) -> str:
    architecture = card.gguf_architecture or card.family or "unknown"
    reason = card.shard_compatibility_reason or (
        "No checked CAI layer-range proof is registered for this GGUF architecture."
    )
    return (
        f"Model '{card.model_id}' is not supported for CAI distributed GGUF "
        f"compute yet. Architecture: {architecture}. {reason}"
    )


def model_info_is_gguf(model: Any) -> bool:
    model_id = str(getattr(model, "id", "") or "").lower()
    tags = [str(tag).lower() for tag in getattr(model, "tags", []) or []]
    return "gguf" in model_id or any("gguf" in tag for tag in tags)


def model_info_supported_for_cai_gguf_compute(model: Any) -> bool:
    if not model_info_is_gguf(model):
        return False
    model_id = str(getattr(model, "id", "") or "").strip()
    tags = [str(tag) for tag in getattr(model, "tags", []) or []]
    compatibility = gguf_shard_compatibility(
        model_id=model_id,
        family=" ".join(tags),
        filename=model_id,
        allow_full_model_local=False,
    )
    return (
        compatibility.layer_range_supported
        and compatibility.shard_compatibility == "layer_range_supported"
    )

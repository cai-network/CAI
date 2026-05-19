# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import asyncio

from cai.shared.models import model_cards as model_cards_module
from cai.shared.models.model_cards import InferenceBackend, ModelCard, ModelTask
from cai.shared.types.common import ModelId
from cai.shared.types.memory import Memory


def _card(model_id: str, backend: InferenceBackend) -> ModelCard:
    return ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_bytes(1024),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=backend,
    )


def test_get_model_cards_respects_allowed_inference_backends(monkeypatch) -> None:
    llama_card = _card("Qwen/Qwen3-0.6B-GGUF", InferenceBackend.LlamaCpp)
    mlx_card = _card("mlx-community/Qwen3-0.6B-4bit", InferenceBackend.Mlx)

    monkeypatch.setattr(
        model_cards_module,
        "_card_cache",
        {
            llama_card.model_id: llama_card,
            mlx_card.model_id: mlx_card,
        },
    )
    monkeypatch.setenv("CAI_ALLOWED_INFERENCE_BACKENDS", "llama_cpp")

    result = asyncio.run(model_cards_module.get_model_cards())

    assert [card.model_id for card in result] == [llama_card.model_id]


def test_qwen3_gguf_cards_use_real_runtime_shape_metadata(monkeypatch) -> None:
    monkeypatch.setattr(model_cards_module, "_card_cache", {})
    monkeypatch.delenv("CAI_ALLOWED_INFERENCE_BACKENDS", raising=False)

    cards = asyncio.run(model_cards_module.get_model_cards())
    cards_by_id = {str(card.model_id): card for card in cards}

    for model_id in ("Qwen/Qwen3-0.6B-GGUF", "cai-network/Qwen3-0.6B-GGUF"):
        card = cards_by_id[model_id]
        assert card.n_layers == 28
        assert card.hidden_size == 1024
        assert card.num_key_value_heads == 8
        assert card.gguf_architecture == "qwen3"
        assert card.shard_compatibility == "layer_range_supported"
        assert card.layer_range_supported is True
        assert card.layer_range_probe_abi == "cai-layer-range-v1"
        assert "qwen3-production-binary-conformance" in (
            card.layer_range_probe_report or ""
        )
        assert "qwen3-layer-range-equivalence-probe" in (
            card.layer_range_equivalence_probe_report or ""
        )


def test_qwen2_gguf_cards_use_checked_layer_range_policy(
    monkeypatch,
) -> None:
    monkeypatch.setattr(model_cards_module, "_card_cache", {})
    monkeypatch.delenv("CAI_ALLOWED_INFERENCE_BACKENDS", raising=False)

    cards = asyncio.run(model_cards_module.get_model_cards())
    cards_by_id = {str(card.model_id): card for card in cards}
    card = cards_by_id["Qwen/Qwen2.5-0.5B-Instruct-GGUF"]

    assert card.gguf_architecture == "qwen2"
    assert card.shard_compatibility == "layer_range_supported"
    assert card.layer_range_supported is True
    assert card.layer_range_probe_abi == "cai-layer-range-v1"
    assert "qwen2.5-production-binary-conformance" in (
        card.layer_range_probe_report or ""
    )
    assert "qwen2.5-layer-range-equivalence-probe" in (
        card.layer_range_equivalence_probe_report or ""
    )


def test_qwen15_gguf_cards_inherit_qwen2_layer_range_policy() -> None:
    card = ModelCard(
        model_id=ModelId("Qwen/Qwen1.5-0.5B-Chat-GGUF"),
        storage_size=Memory.from_bytes(1024),
        n_layers=24,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family="qwen",
        inference_backend=InferenceBackend.LlamaCpp,
        preferred_filename="qwen1_5-0_5b-chat-q4_k_m.gguf",
    )

    assert card.gguf_architecture == "qwen2"
    assert card.shard_compatibility == "layer_range_supported"
    assert card.layer_range_supported is True
    assert card.layer_range_probe_abi == "cai-layer-range-v1"
    assert "qwen2.5-production-binary-conformance" in (
        card.layer_range_probe_report or ""
    )


def test_explicit_gguf_architecture_drives_model_card_shard_policy() -> None:
    card = ModelCard(
        model_id=ModelId("local/custom-gguf"),
        storage_size=Memory.from_bytes(1024),
        n_layers=24,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family="qwen",
        inference_backend=InferenceBackend.LlamaCpp,
        preferred_filename="weights.gguf",
        gguf_architecture="qwen2",
    )

    assert card.gguf_architecture == "qwen2"
    assert card.shard_compatibility == "layer_range_supported"
    assert card.layer_range_supported is True


def test_model_card_policy_overrides_stale_unsupported_gguf_metadata() -> None:
    card = ModelCard(
        model_id=ModelId("Qwen/Qwen2.5-Omni-7B-GGUF"),
        storage_size=Memory.from_bytes(1024),
        n_layers=36,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family="qwen",
        inference_backend=InferenceBackend.LlamaCpp,
        preferred_filename="qwen2.5-omni-7b-q4_k_m.gguf",
        gguf_architecture="qwen",
        shard_compatibility="layer_range_supported",
        layer_range_supported=True,
        layer_range_probe_report="docs/reports/stale-report.json",
    )

    assert card.gguf_architecture == "multimodal"
    assert card.shard_compatibility == "unsupported_for_sharding"
    assert card.layer_range_supported is False
    assert card.layer_range_probe_report is None


def test_unproven_qwen3next_gguf_cards_are_explicitly_unsupported_for_sharding() -> None:
    card = ModelCard(
        model_id=ModelId("Qwen/Qwen3-Next-80B-A3B-GGUF"),
        storage_size=Memory.from_bytes(1024),
        n_layers=48,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family="qwen",
        inference_backend=InferenceBackend.LlamaCpp,
        preferred_filename="qwen3-next-80b-a3b-q4_k_m.gguf",
    )

    assert card.gguf_architecture == "qwen3next"
    assert card.shard_compatibility == "unsupported_for_sharding"
    assert card.layer_range_supported is False


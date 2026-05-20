# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import asyncio
from unittest.mock import AsyncMock, patch

from cai.api.main import API
from cai.api.types import ModelListModel
from cai.shared.models.model_cards import InferenceBackend, ModelCard, ModelTask
from cai.shared.types.memory import Memory
from cai.shared.types.state import State


def _make_api() -> API:
    api = object.__new__(API)
    api.state = State()
    return api


def _model_card(model_id: str, backend: InferenceBackend) -> ModelCard:
    return ModelCard(
        model_id=model_id,
        storage_size=Memory.from_mb(512),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=backend,
    )


def test_model_list_model_defaults_to_llama_cpp_backend() -> None:
    assert ModelListModel(id="Qwen/Qwen3-0.6B-GGUF").inference_backend == "llama_cpp"


def test_get_models_filters_catalog_to_llama_cpp() -> None:
    api = _make_api()
    cards = [
        _model_card("cai-network/Qwen3-0.6B-GGUF", InferenceBackend.LlamaCpp),
        _model_card("Qwen/Qwen3-0.6B-GGUF", InferenceBackend.LlamaCpp),
        _model_card("Qwen/Qwen2.5-0.5B-Instruct-GGUF", InferenceBackend.LlamaCpp),
        _model_card("mlx-community/DeepSeek-V3.1-4bit", InferenceBackend.LlamaCpp),
    ]

    with patch("cai.api.main.get_model_cards", AsyncMock(return_value=cards)):
        result = asyncio.run(api.get_models())

    assert [model.id for model in result.data] == [
        "cai-network/Qwen3-0.6B-GGUF",
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    ]
    assert result.data[0].inference_backend == "llama_cpp"
    assert result.data[0].gguf_architecture == "qwen3"
    assert result.data[0].shard_compatibility == "layer_range_supported"
    assert result.data[0].layer_range_supported is True
    assert result.data[0].layer_range_probe_abi == "cai-layer-range-v1"
    assert "qwen3-layer-range-equivalence-probe" in (
        result.data[0].layer_range_equivalence_probe_report or ""
    )


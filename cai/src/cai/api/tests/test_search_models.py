# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from cai.api.main import API
from cai.api.types import AddCustomModelParams
from cai.shared.models.model_cards import InferenceBackend, ModelCard, ModelTask
from cai.shared.types.common import ModelId
from cai.shared.types.memory import Memory


def _make_api() -> API:
    return object.__new__(API)


def test_search_models_prefers_gguf_results() -> None:
    api = _make_api()

    def fake_list_models(*, search=None, sort=None, limit=None):
        del sort, limit
        if search == "Qwen GGUF":
            return [
                SimpleNamespace(
                    id="Qwen/Qwen3-0.6B-GGUF",
                    author="Qwen",
                    downloads=100,
                    likes=10,
                    last_modified="2026-04-23",
                    tags=["gguf", "text-generation"],
                ),
                SimpleNamespace(
                    id="mlx-community/Qwen3-0.6B-4bit",
                    author="mlx-community",
                    downloads=200,
                    likes=20,
                    last_modified="2026-04-23",
                    tags=["text-generation"],
                ),
            ]
        return []

    with patch("huggingface_hub.list_models", side_effect=fake_list_models):
        results = asyncio.run(api.search_models("Qwen", limit=10))

    assert [result.id for result in results] == ["Qwen/Qwen3-0.6B-GGUF"]


def test_search_models_filters_to_supported_gguf_architectures() -> None:
    api = _make_api()

    def fake_list_models(*, search=None, sort=None, limit=None):
        del sort, limit
        if search == "Qwen GGUF":
            return [
                SimpleNamespace(
                    id="Qwen/Qwen3-0.6B-GGUF",
                    author="Qwen",
                    downloads=100,
                    likes=10,
                    last_modified="2026-04-23",
                    tags=["gguf", "text-generation"],
                ),
                SimpleNamespace(
                    id="Qwen/Qwen3-Next-80B-A3B-GGUF",
                    author="Qwen",
                    downloads=200,
                    likes=20,
                    last_modified="2026-04-23",
                    tags=["gguf", "text-generation"],
                ),
                SimpleNamespace(
                    id="openai/gpt-oss-20b-GGUF",
                    author="openai",
                    downloads=300,
                    likes=30,
                    last_modified="2026-04-23",
                    tags=["gguf", "text-generation"],
                ),
            ]
        return []

    with patch("huggingface_hub.list_models", side_effect=fake_list_models):
        results = asyncio.run(api.search_models("Qwen", limit=10))

    assert [result.id for result in results] == ["Qwen/Qwen3-0.6B-GGUF"]


def test_search_models_falls_back_to_query_results_and_filters_to_gguf() -> None:
    api = _make_api()

    def fake_list_models(*, search=None, sort=None, limit=None):
        del sort, limit
        if search == "Llama GGUF":
            return []
        if search == "Llama":
            return [
                SimpleNamespace(
                    id="meta-llama/Llama-3.2-3B-Instruct",
                    author="meta-llama",
                    downloads=300,
                    likes=30,
                    last_modified="2026-04-23",
                    tags=["text-generation"],
                ),
                SimpleNamespace(
                    id="bartowski/Llama-3.2-3B-Instruct-GGUF",
                    author="bartowski",
                    downloads=400,
                    likes=40,
                    last_modified="2026-04-23",
                    tags=["gguf", "text-generation"],
                ),
            ]
        return []

    with patch("huggingface_hub.list_models", side_effect=fake_list_models):
        results = asyncio.run(api.search_models("Llama", limit=10))

    assert [result.id for result in results] == [
        "bartowski/Llama-3.2-3B-Instruct-GGUF"
    ]


def test_add_custom_model_rejects_unsupported_gguf_architecture() -> None:
    api = _make_api()
    unsupported_card = ModelCard(
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

    async def fake_fetch_from_hf(model_id):
        assert model_id == "Qwen/Qwen3-Next-80B-A3B-GGUF"
        return unsupported_card

    payload = AddCustomModelParams(model_id=ModelId("Qwen/Qwen3-Next-80B-A3B-GGUF"))

    with patch("cai.api.main.ModelCard.fetch_from_hf", side_effect=fake_fetch_from_hf):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(api.add_custom_model(payload))

    assert exc_info.value.status_code == 400
    assert "not supported for CAI distributed GGUF compute yet" in str(
        exc_info.value.detail
    )


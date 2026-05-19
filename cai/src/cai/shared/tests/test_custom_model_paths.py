# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cai.download.download_utils import resolve_existing_model
from cai.shared.models.model_cards import (
    InferenceBackend,
    ModelCard,
    ModelId,
    delete_custom_model_local_path,
    get_custom_model_local_path,
    set_custom_model_local_path,
)


def _create_local_model_dir(root: Path) -> Path:
    model_dir = root / "My Custom Model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "num_hidden_layers": 4,
                "hidden_size": 128,
                "num_key_value_heads": 8,
                "max_position_embeddings": 4096,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {"layers.0.weight": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    return model_dir


def _create_local_gguf_dir(root: Path) -> Path:
    model_dir = root / "GGUF Model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "Qwen3-0.6B-Q4_K_M.gguf").write_bytes(b"gguf")
    return model_dir


async def test_load_from_local_directory_rejects_non_gguf_model(
    tmp_path: Path,
) -> None:
    model_dir = _create_local_model_dir(tmp_path)

    with pytest.raises(ValueError, match="Only GGUF"):
        await ModelCard.load_from_local_directory(
            ModelId("local/my-custom-model"),
            model_dir,
        )


async def test_load_from_local_directory_builds_llama_cpp_card_for_gguf_dir(
    tmp_path: Path,
) -> None:
    model_dir = _create_local_gguf_dir(tmp_path)

    card = await ModelCard.load_from_local_directory(
        ModelId("local/qwen-gguf"),
        model_dir,
    )

    assert card.model_id == "local/qwen-gguf"
    assert card.is_custom is True
    assert card.inference_backend == InferenceBackend.LlamaCpp
    assert card.supports_tensor is False
    assert card.n_layers > 1
    assert card.family == "qwen"
    assert card.quantization == "Q4_K_M"
    assert card.preferred_filename == "Qwen3-0.6B-Q4_K_M.gguf"


async def test_load_from_local_directory_builds_llama_cpp_card_for_gguf_file(
    tmp_path: Path,
) -> None:
    gguf_path = tmp_path / "Meta-Llama-3.2-1B-Instruct-Q8_0.gguf"
    gguf_path.write_bytes(b"gguf")

    card = await ModelCard.load_from_local_directory(
        ModelId("local/llama-gguf"),
        gguf_path,
    )

    assert card.inference_backend == InferenceBackend.LlamaCpp
    assert card.n_layers > 1
    assert card.family == "llama"
    assert card.quantization == "Q8_0"
    assert card.preferred_filename == "Meta-Llama-3.2-1B-Instruct-Q8_0.gguf"


async def test_fetch_from_hf_builds_public_llama_cpp_card_for_gguf_repo() -> None:
    info = SimpleNamespace(
        gguf={"architecture": "qwen2", "context_length": 8192},
        tags=[
            "gguf",
            "base_model:Qwen/Qwen2.5-0.5B-Instruct",
            "base_model:quantized:Qwen/Qwen2.5-0.5B-Instruct",
        ],
        siblings=[
            SimpleNamespace(
                rfilename="qwen2.5-0.5b-instruct-q2_k.gguf",
                size=415182688,
                lfs=None,
            ),
            SimpleNamespace(
                rfilename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
                size=491400032,
                lfs=None,
            ),
        ],
    )

    with patch("cai.shared.models.model_cards.model_info", return_value=info):
        card = await ModelCard.fetch_from_hf(
            ModelId("Qwen/Qwen2.5-0.5B-Instruct-GGUF")
        )

    assert card.model_id == "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    assert card.is_custom is False
    assert card.inference_backend == InferenceBackend.LlamaCpp
    assert card.supports_tensor is False
    assert card.n_layers > 1
    assert card.storage_size.in_bytes == 491400032
    assert card.context_length == 8192
    assert card.family == "qwen2"
    assert card.quantization == "Q4_K_M"
    assert card.base_model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert card.preferred_filename == "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    assert card.gguf_architecture == "qwen2"
    assert card.shard_compatibility == "layer_range_supported"
    assert card.layer_range_supported is True
    assert "qwen2.5-production-binary-conformance" in (
        card.layer_range_probe_report or ""
    )


def test_custom_model_path_registry_and_resolve_existing_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from cai.shared.models import model_cards as model_cards_module

    model_dir = _create_local_model_dir(tmp_path)
    registry_file = tmp_path / "custom_model_paths.json"
    monkeypatch.setattr(model_cards_module, "_custom_model_paths_file", registry_file)

    model_id = ModelId("local/my-custom-model")
    stored_path = set_custom_model_local_path(model_id, str(model_dir))

    assert stored_path == model_dir
    assert get_custom_model_local_path(model_id) == model_dir
    assert resolve_existing_model(model_id) == model_dir

    assert delete_custom_model_local_path(model_id) is True
    assert get_custom_model_local_path(model_id) is None


def test_custom_gguf_path_registry_and_resolve_existing_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from cai.shared.models import model_cards as model_cards_module

    gguf_path = tmp_path / "Tiny-Q4_K_M.gguf"
    gguf_path.write_bytes(b"gguf")
    registry_file = tmp_path / "custom_model_paths.json"
    monkeypatch.setattr(model_cards_module, "_custom_model_paths_file", registry_file)

    model_id = ModelId("local/tiny-gguf")
    stored_path = set_custom_model_local_path(model_id, str(gguf_path))

    assert stored_path == gguf_path
    assert get_custom_model_local_path(model_id) == gguf_path
    assert resolve_existing_model(model_id) == gguf_path


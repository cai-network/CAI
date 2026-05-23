# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

from cai.api import model_compute_policy
from cai.api.model_compute_policy import (
    model_card_supported_for_cai_gguf_compute,
    model_info_is_gguf,
    model_info_supported_for_cai_gguf_compute,
    unsupported_gguf_model_detail,
)
from cai.shared.models.model_cards import InferenceBackend


def test_model_card_supported_for_cai_gguf_compute_requires_llama_layer_range() -> None:
    supported = SimpleNamespace(
        inference_backend=InferenceBackend.LlamaCpp,
        layer_range_supported=True,
        shard_compatibility="layer_range_supported",
    )
    unsupported_backend = SimpleNamespace(
        inference_backend="other",
        layer_range_supported=True,
        shard_compatibility="layer_range_supported",
    )
    unsupported_range = SimpleNamespace(
        inference_backend=InferenceBackend.LlamaCpp,
        layer_range_supported=False,
        shard_compatibility="unsupported",
    )

    assert model_card_supported_for_cai_gguf_compute(supported) is True
    assert model_card_supported_for_cai_gguf_compute(unsupported_backend) is False
    assert model_card_supported_for_cai_gguf_compute(unsupported_range) is False


def test_unsupported_gguf_model_detail_includes_architecture_and_reason() -> None:
    detail = unsupported_gguf_model_detail(
        SimpleNamespace(
            model_id="Qwen/Test-GGUF",
            gguf_architecture="qwen2",
            family="qwen",
            shard_compatibility_reason="layer-range proof failed",
        )
    )

    assert "Qwen/Test-GGUF" in detail
    assert "qwen2" in detail
    assert "layer-range proof failed" in detail


def test_model_info_is_gguf_accepts_model_id_or_tags() -> None:
    assert model_info_is_gguf(SimpleNamespace(id="Qwen/Test-GGUF", tags=[])) is True
    assert model_info_is_gguf(SimpleNamespace(id="Qwen/Test", tags=["GGUF"])) is True
    assert model_info_is_gguf(SimpleNamespace(id="Qwen/Test", tags=["safetensors"])) is False


def test_model_info_supported_for_cai_gguf_compute_uses_shard_policy() -> None:
    calls: list[dict[str, object]] = []
    original = model_compute_policy.gguf_shard_compatibility

    def _fake_compatibility(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            layer_range_supported=True,
            shard_compatibility="layer_range_supported",
        )

    try:
        model_compute_policy.gguf_shard_compatibility = _fake_compatibility
        supported = model_info_supported_for_cai_gguf_compute(
            SimpleNamespace(id="Qwen/Test-GGUF", tags=["qwen"])
        )
    finally:
        model_compute_policy.gguf_shard_compatibility = original

    assert supported is True
    assert calls == [
        {
            "model_id": "Qwen/Test-GGUF",
            "family": "qwen",
            "filename": "Qwen/Test-GGUF",
            "allow_full_model_local": False,
        }
    ]

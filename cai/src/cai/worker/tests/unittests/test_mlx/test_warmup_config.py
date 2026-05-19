# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import cai.worker.engines.mlx.generator.generate as mlx_generate


def test_warmup_max_output_tokens_defaults_to_50(monkeypatch):
    monkeypatch.delenv("CAI_WARMUP_MAX_OUTPUT_TOKENS", raising=False)
    assert mlx_generate._warmup_max_output_tokens() == 50


def test_warmup_max_output_tokens_can_be_lowered(monkeypatch):
    monkeypatch.setenv("CAI_WARMUP_MAX_OUTPUT_TOKENS", "1")
    assert mlx_generate._warmup_max_output_tokens() == 1


def test_warmup_max_output_tokens_can_disable_warmup(monkeypatch):
    monkeypatch.setenv("CAI_WARMUP_MAX_OUTPUT_TOKENS", "0")
    assert mlx_generate._warmup_max_output_tokens() == 0


def test_warmup_max_output_tokens_ignores_invalid_values(monkeypatch):
    monkeypatch.setenv("CAI_WARMUP_MAX_OUTPUT_TOKENS", "nope")
    assert mlx_generate._warmup_max_output_tokens() == 50


# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from cai_compute_chain.gguf_architecture_matrix import (
    CANDIDATE_STATUS,
    NON_DECODER_STATUS,
    PROVEN_STATUS,
    SEPARATE_CONTRACT_STATUS,
    build_gguf_architecture_matrix,
    parse_llama_architecture_names,
    render_markdown_matrix,
)
from cai_compute_chain.gguf_shard_policy import (
    GGUF_SHARD_MODE_UNSUPPORTED_FOR_SHARDING,
    gguf_shard_compatibility,
    normalize_gguf_architecture,
)


def test_parse_llama_architecture_names_from_llama_cpp_registry() -> None:
    text = '''
static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
    { LLM_ARCH_LLAMA,            "llama"            },
    { LLM_ARCH_QWEN3,            "qwen3"            },
    { LLM_ARCH_FALCON,           "falcon"           },
    { LLM_ARCH_UNKNOWN,          "(unknown)"        },
};
'''

    assert parse_llama_architecture_names(text) == ("llama", "qwen3", "falcon")


def test_architecture_matrix_marks_proven_and_candidate_tiers() -> None:
    rows = {
        row.architecture: row
        for row in build_gguf_architecture_matrix(
            (
                "llama",
                "mistral3",
                "gemma",
                "gemma2",
                "gemma3",
                "gptneox",
                "phi2",
                "phi3",
                "falcon",
                "qwen3vl",
                "qwen3moe",
                "mamba",
                "qwen35",
                "qwen3next",
                "falcon-h1",
                "gemma3n",
                "dream",
            )
        )
    }

    assert rows["llama"].status == PROVEN_STATUS
    assert rows["mistral3"].status == PROVEN_STATUS
    assert rows["gemma"].status == PROVEN_STATUS
    assert rows["gemma2"].status == PROVEN_STATUS
    assert rows["gemma3"].status == PROVEN_STATUS
    assert rows["gptneox"].status == PROVEN_STATUS
    assert rows["phi2"].status == PROVEN_STATUS
    assert rows["phi3"].status == PROVEN_STATUS
    assert rows["falcon"].status == CANDIDATE_STATUS
    assert rows["qwen3vl"].status == NON_DECODER_STATUS
    assert rows["gemma3n"].status == NON_DECODER_STATUS
    assert rows["qwen3moe"].status == SEPARATE_CONTRACT_STATUS
    assert rows["mamba"].status == SEPARATE_CONTRACT_STATUS
    assert rows["qwen35"].status == SEPARATE_CONTRACT_STATUS
    assert rows["qwen3next"].status == SEPARATE_CONTRACT_STATUS
    assert rows["falcon-h1"].status == SEPARATE_CONTRACT_STATUS
    assert rows["dream"].status == SEPARATE_CONTRACT_STATUS


def test_explicit_unknown_dense_architecture_is_preserved_but_not_whitelisted() -> None:
    assert normalize_gguf_architecture(architecture="falcon") == "falcon"
    assert normalize_gguf_architecture(architecture="modern-bert") == "modern-bert"

    compatibility = gguf_shard_compatibility(
        model_id="tiiuae/falcon-7b-gguf",
        gguf_architecture="falcon",
        filename="falcon-7b-q4_k_m.gguf",
    )

    assert compatibility.gguf_architecture == "falcon"
    assert compatibility.shard_compatibility == GGUF_SHARD_MODE_UNSUPPORTED_FOR_SHARDING
    assert compatibility.layer_range_supported is False


def test_render_markdown_matrix_includes_add_flow_and_rows() -> None:
    rows = build_gguf_architecture_matrix(("llama", "falcon"))
    rendered = render_markdown_matrix(rows, source="llama-arch.cpp")

    assert "Add-New-Architecture Flow" in rendered
    assert "| llama | layer_range_supported | proven |" in rendered
    assert "| falcon | candidate_requires_hook_probe_conformance |" in rendered

# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


GGUF_SHARD_MODE_FULL_MODEL_LOCAL = "full_model_local"
GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED = "layer_range_supported"
GGUF_SHARD_MODE_LOW_LATENCY_RPC_CELL = "low_latency_rpc_cell"
GGUF_SHARD_MODE_UNSUPPORTED_FOR_SHARDING = "unsupported_for_sharding"

GGUF_LAYER_RANGE_PROBE_ABI = "cai-layer-range-v1"
GGUF_QWEN3_LAYER_RANGE_LEGACY_PROBE_ABI = "cai-qwen3-layer-range-v1"
GGUF_QWEN3_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT = (
    "docs/reports/qwen3-layer-range-equivalence-probe-2026-05-10.json"
)
GGUF_QWEN3_LAYER_RANGE_PROBE_REPORT = (
    "docs/reports/qwen3-production-binary-conformance-2026-05-10.json"
)
GGUF_QWEN2_5_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT = (
    "docs/reports/qwen2.5-layer-range-equivalence-probe-2026-05-11.json"
)
GGUF_QWEN2_5_LAYER_RANGE_PROBE_REPORT = (
    "docs/reports/qwen2.5-production-binary-conformance-2026-05-11.json"
)
GGUF_QWEN_1_8B_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT = (
    "docs/reports/qwen-1.8b-chat-layer-range-equivalence-probe-2026-05-11.json"
)
GGUF_QWEN_1_8B_LAYER_RANGE_PROBE_REPORT = (
    "docs/reports/qwen-1.8b-chat-production-binary-conformance-2026-05-11.json"
)
GGUF_LLAMA_TINYLLAMA_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT = (
    "docs/reports/llama-tinyllama-layer-range-equivalence-probe-2026-05-11.json"
)
GGUF_LLAMA_TINYLLAMA_LAYER_RANGE_PROBE_REPORT = (
    "docs/reports/llama-tinyllama-production-binary-conformance-2026-05-11.json"
)
GGUF_MISTRAL3_MINISTRAL_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT = (
    "docs/reports/mistral3-ministral-3-3b-layer-range-equivalence-probe-2026-05-11.json"
)
GGUF_MISTRAL3_MINISTRAL_LAYER_RANGE_PROBE_REPORT = (
    "docs/reports/mistral3-ministral-3-3b-production-binary-conformance-2026-05-11.json"
)
GGUF_GEMMA_2B_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT = (
    "docs/reports/gemma-2b-layer-range-equivalence-probe-2026-05-11.json"
)
GGUF_GEMMA_2B_LAYER_RANGE_PROBE_REPORT = (
    "docs/reports/gemma-2b-production-binary-conformance-2026-05-11.json"
)
GGUF_GEMMA2_2B_IT_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT = (
    "docs/reports/gemma2-2b-it-layer-range-equivalence-probe-2026-05-11.json"
)
GGUF_GEMMA2_2B_IT_LAYER_RANGE_PROBE_REPORT = (
    "docs/reports/gemma2-2b-it-production-binary-conformance-2026-05-11.json"
)
GGUF_GEMMA3_1B_IT_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT = (
    "docs/reports/gemma3-1b-it-layer-range-equivalence-probe-2026-05-11.json"
)
GGUF_GEMMA3_1B_IT_LAYER_RANGE_PROBE_REPORT = (
    "docs/reports/gemma3-1b-it-production-binary-conformance-2026-05-11.json"
)
GGUF_PHI3_MINI_4K_INSTRUCT_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT = (
    "docs/reports/phi3-mini-4k-instruct-layer-range-equivalence-probe-2026-05-11.json"
)
GGUF_PHI3_MINI_4K_INSTRUCT_LAYER_RANGE_PROBE_REPORT = (
    "docs/reports/phi3-mini-4k-instruct-production-binary-conformance-2026-05-11.json"
)
GGUF_PHI2_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT = (
    "docs/reports/phi2-layer-range-equivalence-probe-2026-05-11.json"
)
GGUF_PHI2_LAYER_RANGE_PROBE_REPORT = (
    "docs/reports/phi2-production-binary-conformance-2026-05-11.json"
)
GGUF_GPTNEOX_PYTHIA_14M_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT = (
    "docs/reports/gptneox-pythia-14m-layer-range-equivalence-probe-2026-05-11.json"
)
GGUF_GPTNEOX_PYTHIA_14M_LAYER_RANGE_PROBE_REPORT = (
    "docs/reports/gptneox-pythia-14m-production-binary-conformance-2026-05-11.json"
)
GGUF_LAYER_RANGE_ACTIVATION_STATE_FORMAT = "ggml-tensor-v1/layer-range-activation-v1"
GGUF_LAYER_RANGE_DECODE_STATE_FORMAT = "ggml-kv-cache-v1/token-step-kv-cache-v1"
GGUF_QWEN3_ACTIVATION_STATE_FORMAT = GGUF_LAYER_RANGE_ACTIVATION_STATE_FORMAT
GGUF_QWEN3_DECODE_STATE_FORMAT = GGUF_LAYER_RANGE_DECODE_STATE_FORMAT
GGUF_LAYER_RANGE_REQUIRED_PRODUCTION_CAPABILITIES = (
    "gguf_layer_execution",
    "layer_range_execution",
    "real_activation_state",
    "real_decode_state",
)
GGUF_LAYER_RANGE_MAX_ABS_DIFF_THRESHOLD = 1e-4
GGUF_LAYER_RANGE_MEAN_ABS_DIFF_THRESHOLD = 1e-5
GGUF_QWEN_DENSE_ARCHITECTURES = ("qwen", "qwen2", "qwen3")
GGUF_QWEN_RECURRENT_ARCHITECTURES = ("qwen35", "qwen3next")
GGUF_QWEN_MOE_ARCHITECTURES = ("qwen2moe", "qwen3moe", "qwen35moe")
GGUF_MISTRAL_DENSE_ARCHITECTURES = ("mistral", "mistral3", "mistral4")
GGUF_QWEN_MULTIMODAL_ARCHITECTURES = (
    "qwen2audio",
    "qwen2vl",
    "qwen25omni",
    "qwen25vl",
    "qwen3omni",
    "qwen3vl",
    "qwen3vlmoe",
)


@dataclass(frozen=True)
class GgufShardCompatibility:
    model_format: str
    gguf_architecture: str
    shard_compatibility: str
    layer_range_supported: bool
    layer_range_probe_abi: str | None = None
    layer_range_probe_report: str | None = None
    layer_range_equivalence_probe_report: str | None = None
    state_format: str | None = None
    activation_state_format: str | None = None
    decode_state_format: str | None = None
    reason: str = ""

    def to_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "model_format": self.model_format,
            "gguf_architecture": self.gguf_architecture,
            "shard_compatibility": self.shard_compatibility,
            "layer_range_supported": self.layer_range_supported,
        }
        if self.layer_range_probe_abi:
            metadata["layer_range_probe_abi"] = self.layer_range_probe_abi
        if self.layer_range_probe_report:
            metadata["layer_range_probe_report"] = self.layer_range_probe_report
        if self.layer_range_equivalence_probe_report:
            metadata["layer_range_equivalence_probe_report"] = (
                self.layer_range_equivalence_probe_report
            )
        if self.state_format:
            metadata["state_format"] = self.state_format
        if self.activation_state_format:
            metadata["activation_state_format"] = self.activation_state_format
        if self.decode_state_format:
            metadata["decode_state_format"] = self.decode_state_format
        if self.reason:
            metadata["shard_compatibility_reason"] = self.reason
        return metadata


@dataclass(frozen=True)
class GgufLayerRangeArchitectureProof:
    gguf_architecture: str
    reference_model_id: str
    layer_range_probe_abi: str
    equivalence_probe_report: str
    conformance_probe_report: str
    state_format: str
    activation_state_format: str
    decode_state_format: str

    def to_compatibility(self) -> GgufShardCompatibility:
        return GgufShardCompatibility(
            model_format="gguf",
            gguf_architecture=self.gguf_architecture,
            shard_compatibility=GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED,
            layer_range_supported=True,
            layer_range_probe_abi=self.layer_range_probe_abi,
            layer_range_probe_report=self.conformance_probe_report,
            layer_range_equivalence_probe_report=self.equivalence_probe_report,
            state_format=self.state_format,
            activation_state_format=self.activation_state_format,
            decode_state_format=self.decode_state_format,
            reason=(
                f"{self.gguf_architecture} GGUF layer-range architecture has "
                f"checked CAI equivalence and production probes on "
                f"{self.reference_model_id}."
            ),
        )


GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS: Mapping[
    str,
    GgufLayerRangeArchitectureProof,
] = {
    "qwen3": GgufLayerRangeArchitectureProof(
        gguf_architecture="qwen3",
        reference_model_id="cai-network/Qwen3-0.6B-GGUF",
        layer_range_probe_abi=GGUF_LAYER_RANGE_PROBE_ABI,
        equivalence_probe_report=GGUF_QWEN3_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT,
        conformance_probe_report=GGUF_QWEN3_LAYER_RANGE_PROBE_REPORT,
        state_format=GGUF_QWEN3_ACTIVATION_STATE_FORMAT,
        activation_state_format=GGUF_QWEN3_ACTIVATION_STATE_FORMAT,
        decode_state_format=GGUF_QWEN3_DECODE_STATE_FORMAT,
    ),
    "qwen2": GgufLayerRangeArchitectureProof(
        gguf_architecture="qwen2",
        reference_model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        layer_range_probe_abi=GGUF_LAYER_RANGE_PROBE_ABI,
        equivalence_probe_report=GGUF_QWEN2_5_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT,
        conformance_probe_report=GGUF_QWEN2_5_LAYER_RANGE_PROBE_REPORT,
        state_format=GGUF_LAYER_RANGE_ACTIVATION_STATE_FORMAT,
        activation_state_format=GGUF_LAYER_RANGE_ACTIVATION_STATE_FORMAT,
        decode_state_format=GGUF_LAYER_RANGE_DECODE_STATE_FORMAT,
    ),
    "qwen": GgufLayerRangeArchitectureProof(
        gguf_architecture="qwen",
        reference_model_id="mradermacher/Qwen-1_8B-Chat-GGUF",
        layer_range_probe_abi=GGUF_LAYER_RANGE_PROBE_ABI,
        equivalence_probe_report=GGUF_QWEN_1_8B_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT,
        conformance_probe_report=GGUF_QWEN_1_8B_LAYER_RANGE_PROBE_REPORT,
        state_format=GGUF_LAYER_RANGE_ACTIVATION_STATE_FORMAT,
        activation_state_format=GGUF_LAYER_RANGE_ACTIVATION_STATE_FORMAT,
        decode_state_format=GGUF_LAYER_RANGE_DECODE_STATE_FORMAT,
    ),
    "llama": GgufLayerRangeArchitectureProof(
        gguf_architecture="llama",
        reference_model_id="cai-network/TinyLlama-1.1B-Chat-v1.0-GGUF",
        layer_range_probe_abi=GGUF_LAYER_RANGE_PROBE_ABI,
        equivalence_probe_report=(
            GGUF_LLAMA_TINYLLAMA_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT
        ),
        conformance_probe_report=GGUF_LLAMA_TINYLLAMA_LAYER_RANGE_PROBE_REPORT,
        state_format=GGUF_LAYER_RANGE_ACTIVATION_STATE_FORMAT,
        activation_state_format=GGUF_LAYER_RANGE_ACTIVATION_STATE_FORMAT,
        decode_state_format=GGUF_LAYER_RANGE_DECODE_STATE_FORMAT,
    ),
    "mistral3": GgufLayerRangeArchitectureProof(
        gguf_architecture="mistral3",
        reference_model_id="mistralai/Ministral-3-3B-Instruct-2512-GGUF",
        layer_range_probe_abi=GGUF_LAYER_RANGE_PROBE_ABI,
        equivalence_probe_report=(
            GGUF_MISTRAL3_MINISTRAL_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT
        ),
        conformance_probe_report=GGUF_MISTRAL3_MINISTRAL_LAYER_RANGE_PROBE_REPORT,
        state_format=GGUF_LAYER_RANGE_ACTIVATION_STATE_FORMAT,
        activation_state_format=GGUF_LAYER_RANGE_ACTIVATION_STATE_FORMAT,
        decode_state_format=GGUF_LAYER_RANGE_DECODE_STATE_FORMAT,
    ),
    "gemma": GgufLayerRangeArchitectureProof(
        gguf_architecture="gemma",
        reference_model_id="tensorblock/gemma-2b-GGUF",
        layer_range_probe_abi=GGUF_LAYER_RANGE_PROBE_ABI,
        equivalence_probe_report=GGUF_GEMMA_2B_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT,
        conformance_probe_report=GGUF_GEMMA_2B_LAYER_RANGE_PROBE_REPORT,
        state_format=GGUF_LAYER_RANGE_ACTIVATION_STATE_FORMAT,
        activation_state_format=GGUF_LAYER_RANGE_ACTIVATION_STATE_FORMAT,
        decode_state_format=GGUF_LAYER_RANGE_DECODE_STATE_FORMAT,
    ),
    "gemma2": GgufLayerRangeArchitectureProof(
        gguf_architecture="gemma2",
        reference_model_id="MaziyarPanahi/gemma-2-2b-it-GGUF",
        layer_range_probe_abi=GGUF_LAYER_RANGE_PROBE_ABI,
        equivalence_probe_report=(
            GGUF_GEMMA2_2B_IT_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT
        ),
        conformance_probe_report=GGUF_GEMMA2_2B_IT_LAYER_RANGE_PROBE_REPORT,
        state_format=GGUF_LAYER_RANGE_ACTIVATION_STATE_FORMAT,
        activation_state_format=GGUF_LAYER_RANGE_ACTIVATION_STATE_FORMAT,
        decode_state_format=GGUF_LAYER_RANGE_DECODE_STATE_FORMAT,
    ),
    "gemma3": GgufLayerRangeArchitectureProof(
        gguf_architecture="gemma3",
        reference_model_id="second-state/gemma-3-1b-it-GGUF",
        layer_range_probe_abi=GGUF_LAYER_RANGE_PROBE_ABI,
        equivalence_probe_report=(
            GGUF_GEMMA3_1B_IT_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT
        ),
        conformance_probe_report=GGUF_GEMMA3_1B_IT_LAYER_RANGE_PROBE_REPORT,
        state_format=GGUF_LAYER_RANGE_ACTIVATION_STATE_FORMAT,
        activation_state_format=GGUF_LAYER_RANGE_ACTIVATION_STATE_FORMAT,
        decode_state_format=GGUF_LAYER_RANGE_DECODE_STATE_FORMAT,
    ),
    "phi3": GgufLayerRangeArchitectureProof(
        gguf_architecture="phi3",
        reference_model_id="tensorblock/Phi-3-mini-4k-instruct-GGUF",
        layer_range_probe_abi=GGUF_LAYER_RANGE_PROBE_ABI,
        equivalence_probe_report=(
            GGUF_PHI3_MINI_4K_INSTRUCT_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT
        ),
        conformance_probe_report=GGUF_PHI3_MINI_4K_INSTRUCT_LAYER_RANGE_PROBE_REPORT,
        state_format=GGUF_LAYER_RANGE_ACTIVATION_STATE_FORMAT,
        activation_state_format=GGUF_LAYER_RANGE_ACTIVATION_STATE_FORMAT,
        decode_state_format=GGUF_LAYER_RANGE_DECODE_STATE_FORMAT,
    ),
    "phi2": GgufLayerRangeArchitectureProof(
        gguf_architecture="phi2",
        reference_model_id="TheBloke/phi-2-GGUF",
        layer_range_probe_abi=GGUF_LAYER_RANGE_PROBE_ABI,
        equivalence_probe_report=GGUF_PHI2_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT,
        conformance_probe_report=GGUF_PHI2_LAYER_RANGE_PROBE_REPORT,
        state_format=GGUF_LAYER_RANGE_ACTIVATION_STATE_FORMAT,
        activation_state_format=GGUF_LAYER_RANGE_ACTIVATION_STATE_FORMAT,
        decode_state_format=GGUF_LAYER_RANGE_DECODE_STATE_FORMAT,
    ),
    "gptneox": GgufLayerRangeArchitectureProof(
        gguf_architecture="gptneox",
        reference_model_id="DevQuasar-3/EleutherAI.pythia-14m-GGUF",
        layer_range_probe_abi=GGUF_LAYER_RANGE_PROBE_ABI,
        equivalence_probe_report=(
            GGUF_GPTNEOX_PYTHIA_14M_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT
        ),
        conformance_probe_report=GGUF_GPTNEOX_PYTHIA_14M_LAYER_RANGE_PROBE_REPORT,
        state_format=GGUF_LAYER_RANGE_ACTIVATION_STATE_FORMAT,
        activation_state_format=GGUF_LAYER_RANGE_ACTIVATION_STATE_FORMAT,
        decode_state_format=GGUF_LAYER_RANGE_DECODE_STATE_FORMAT,
    ),
}


def normalize_gguf_architecture(
    architecture: str | None = None,
    *,
    model_id: str | None = None,
    family: str | None = None,
    filename: str | None = None,
) -> str:
    text = _normalize_gguf_architecture_from_text(
        " ".join(
            str(part or "").strip().lower()
            for part in (family, filename, model_id)
            if str(part or "").strip()
        )
    )
    explicit_architecture = _normalize_explicit_gguf_architecture(architecture)
    if explicit_architecture != "unknown":
        if (
            explicit_architecture == "qwen"
            and text
            in {
                "moe",
                "multimodal",
                "qwen2",
                "qwen3",
                "qwen35",
                "qwen3next",
            }
        ):
            return text
        return explicit_architecture
    return text


def _normalize_explicit_gguf_architecture(architecture: str | None) -> str:
    raw_value = str(architecture or "").strip().lower()
    if not raw_value:
        return "unknown"
    value = raw_value.replace("-", "").replace("_", "").replace(".", "")
    if value in GGUF_QWEN_MULTIMODAL_ARCHITECTURES:
        return "multimodal"
    if value in GGUF_QWEN_MOE_ARCHITECTURES:
        return "moe"
    if value in GGUF_QWEN_DENSE_ARCHITECTURES:
        return value
    if value in GGUF_QWEN_RECURRENT_ARCHITECTURES:
        return value
    if value in GGUF_MISTRAL_DENSE_ARCHITECTURES:
        return value
    if value in {"qwen15", "qwen1p5", "qwen25", "qwen2p5"}:
        return "qwen2"
    if value in {"qwen35", "qwen3p5"}:
        return "qwen35"
    if _is_safe_explicit_gguf_architecture(raw_value):
        return raw_value
    return _normalize_gguf_architecture_from_text(raw_value)


def _is_safe_explicit_gguf_architecture(value: str) -> bool:
    clean = str(value or "").strip().lower()
    if not clean or len(clean) > 64:
        return False
    if clean[0] in {"-", ".", "_"} or clean[-1] in {"-", ".", "_"}:
        return False
    return all(char.isascii() and (char.isalnum() or char in {"-", ".", "_"}) for char in clean)


def _normalize_gguf_architecture_from_text(text: str) -> str:
    text = str(text or "").strip().lower()
    if not text:
        return "unknown"
    compact = text.replace("-", "").replace("_", "").replace(".", "").replace(" ", "")
    if "qwen3next" in compact:
        return "qwen3next"
    if any(
        token in text
        for token in (
            "audio",
            "image",
            "multimodal",
            "omni",
            "video",
            "vision",
            "vl",
        )
    ):
        return "multimodal"
    if any(token in text for token in ("mixtral", "moe", "a3b", "a22b")):
        return "moe"
    if "qwen35" in compact or "qwen3p5" in compact:
        return "qwen35"
    if "qwen3" in text and "qwen2.5" not in text:
        return "qwen3"
    if "qwen15" in compact or "qwen1p5" in compact:
        return "qwen2"
    if "qwen2" in text or "qwen2.5" in text:
        return "qwen2"
    if "qwen" in text:
        return "qwen"
    if "llama" in text:
        return "llama"
    if "mistral3" in compact:
        return "mistral3"
    if "mistral4" in compact:
        return "mistral4"
    if "mistral" in text:
        return "mistral"
    if "gemma3n" in compact or "gemma-3n" in text or "gemma 3n" in text:
        return "gemma3n"
    if "gemma-4" in text or "gemma4" in compact:
        return "gemma4"
    if "gemma-3" in text or "gemma3" in compact:
        return "gemma3"
    if (
        "gemma-2-" in text
        or "gemma 2 " in text
        or "gemma_2_" in text
        or "gemma2-" in text
        or "gemma2_" in text
        or "gemma2." in text
    ):
        return "gemma2"
    if "gemma" in text:
        return "gemma"
    if "phi-2" in text or "phi2" in compact:
        return "phi2"
    if "phi-3" in text or "phi3" in compact:
        return "phi3"
    if "gptneox" in compact or "gpt-neox" in text or "pythia" in text:
        return "gptneox"
    if "rwkv" in text:
        return "rwkv"
    if "bert" in text:
        return "bert"
    return "unknown"


def gguf_layer_range_architecture_proof(
    architecture: str | None,
) -> GgufLayerRangeArchitectureProof | None:
    normalized_architecture = normalize_gguf_architecture(architecture)
    return GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS.get(normalized_architecture)


def gguf_shard_compatibility(
    *,
    model_id: str,
    gguf_architecture: str | None = None,
    family: str | None = None,
    filename: str | None = None,
    allow_full_model_local: bool = False,
) -> GgufShardCompatibility:
    normalized_architecture = normalize_gguf_architecture(
        gguf_architecture,
        model_id=model_id,
        family=family,
        filename=filename,
    )
    proof = GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS.get(normalized_architecture)
    if proof is not None:
        return proof.to_compatibility()

    if allow_full_model_local:
        return GgufShardCompatibility(
            model_format="gguf",
            gguf_architecture=normalized_architecture,
            shard_compatibility=GGUF_SHARD_MODE_FULL_MODEL_LOCAL,
            layer_range_supported=False,
            reason=(
                f"GGUF architecture '{normalized_architecture}' has no checked CAI "
                "layer-range equivalence probe yet; policy allows it only as "
                "single-node full-model local inference."
            ),
        )

    return GgufShardCompatibility(
        model_format="gguf",
        gguf_architecture=normalized_architecture,
        shard_compatibility=GGUF_SHARD_MODE_UNSUPPORTED_FOR_SHARDING,
        layer_range_supported=False,
        reason=(
            f"GGUF architecture '{normalized_architecture}' has no checked CAI "
            "layer-range equivalence probe yet."
        ),
    )


def validate_gguf_layer_range_architecture_proof(
    *,
    architecture: str,
    repo_root: str | Path = ".",
) -> tuple[bool, str | None]:
    proof = gguf_layer_range_architecture_proof(architecture)
    if proof is None:
        return False, (
            f"GGUF architecture '{normalize_gguf_architecture(architecture)}' has "
            "no registered layer-range proof."
        )

    root = Path(repo_root)
    try:
        load_gguf_layer_range_equivalence_probe_report(
            root / proof.equivalence_probe_report,
            expected_model_id=proof.reference_model_id,
            expected_architecture=proof.gguf_architecture,
        )
        load_gguf_layer_range_conformance_report(
            root / proof.conformance_probe_report,
            expected_model_id=proof.reference_model_id,
            expected_architecture=proof.gguf_architecture,
        )
    except (OSError, ValueError) as exc:
        return False, (
            f"GGUF layer-range proof for architecture "
            f"'{proof.gguf_architecture}' is invalid: {exc}"
        )
    return True, None


def validate_gguf_layer_range_conformance_report(
    report: Mapping[str, Any],
    *,
    expected_model_id: str,
    expected_architecture: str,
) -> tuple[bool, str | None]:
    checks = _mapping(report.get("checks"))
    result = _mapping(report.get("result"))
    model = _mapping(report.get("model"))
    strict_guards = _mapping(report.get("strictGuards"))
    production_binary = _mapping(report.get("productionBinary"))

    if result.get("ok") is not True or str(result.get("status") or "") != "passed":
        return False, "GGUF layer-range conformance result is not passed."
    if str(checks.get("status") or "") != "passed":
        return False, "GGUF layer-range conformance checks are not passed."
    if checks.get("productionReady") is not True:
        return False, "GGUF layer-range conformance must be productionReady."
    if checks.get("patchBoundaryVerified") is not True:
        return False, "GGUF layer-range conformance patch boundary is not verified."

    model_id = str(model.get("modelId") or checks.get("modelId") or "").strip()
    expected_model = str(expected_model_id or "").strip()
    if not expected_model or model_id != expected_model:
        return False, "GGUF layer-range conformance model id does not match."

    architecture = normalize_gguf_architecture(
        model.get("family"),
        model_id=model_id,
        filename=model.get("ggufFile"),
    )
    expected_arch = str(expected_architecture or "").strip().lower()
    if architecture != expected_arch:
        return False, "GGUF layer-range conformance architecture does not match."

    production_checks = _mapping(checks.get("productionReadinessChecks"))
    capabilities = {
        str(item).strip()
        for item in production_checks.get("backendCapabilities") or ()
        if str(item).strip()
    }
    missing_capabilities = [
        item
        for item in GGUF_LAYER_RANGE_REQUIRED_PRODUCTION_CAPABILITIES
        if item not in capabilities
    ]
    if missing_capabilities:
        return False, (
            "GGUF layer-range conformance is missing production capabilities: "
            + ", ".join(missing_capabilities)
        )
    if production_checks.get("productionStateContractReady") is not True:
        return False, "GGUF layer-range conformance production state contract is not ready."

    probe = _mapping(
        checks.get("generationProbe")
        or production_checks.get("generationProbe")
    )
    if probe.get("ready") is not True:
        return False, "GGUF layer-range conformance generation probe is not ready."
    if probe.get("realModelExecution") is not True:
        return False, "GGUF layer-range conformance requires realModelExecution=true."
    if probe.get("realLayerExecution") is not True:
        return False, "GGUF layer-range conformance requires realLayerExecution=true."

    for guard_name in (
        "forbidFullModelFallback",
        "requireProduction",
        "requireRealLayerExecution",
    ):
        if strict_guards.get(guard_name) is not True:
            return False, f"GGUF layer-range conformance strict guard {guard_name} is missing."

    if not _is_sha256_hex(production_binary.get("sha256Hex")):
        return False, "GGUF layer-range conformance production binary hash is invalid."
    try:
        binary_size = int(production_binary.get("sizeBytes") or 0)
    except (TypeError, ValueError):
        binary_size = 0
    if binary_size <= 0:
        return False, "GGUF layer-range conformance production binary size is invalid."

    try:
        total_layers = int(model.get("totalLayers") or 0)
    except (TypeError, ValueError):
        total_layers = 0
    if total_layers <= 0:
        return False, "GGUF layer-range conformance total layer count is invalid."
    return True, None


def validate_gguf_layer_range_equivalence_probe_report(
    report: Mapping[str, Any],
    *,
    expected_model_id: str,
    expected_architecture: str,
    max_abs_diff_threshold: float = GGUF_LAYER_RANGE_MAX_ABS_DIFF_THRESHOLD,
    mean_abs_diff_threshold: float = GGUF_LAYER_RANGE_MEAN_ABS_DIFF_THRESHOLD,
) -> tuple[bool, str | None]:
    model = _mapping(report.get("model"))
    execution = _mapping(report.get("execution"))
    layer_range = _mapping(report.get("layerRange") or report.get("layer_range"))
    result = _mapping(report.get("result"))

    status = _text_value(report, "status") or _text_value(result, "status")
    if status not in {"ok", "passed"}:
        return False, "GGUF layer-range equivalence probe status is not passed."

    probe_abi = (
        _text_value(report, "probeAbi", "probe_abi", "probe", "abi")
        or _text_value(result, "probeAbi", "probe_abi", "probe", "abi")
    )
    if probe_abi not in {
        GGUF_LAYER_RANGE_PROBE_ABI,
        GGUF_QWEN3_LAYER_RANGE_LEGACY_PROBE_ABI,
    }:
        return False, "GGUF layer-range equivalence probe ABI is unsupported."

    model_id = _text_value(report, "modelId", "model_id") or _text_value(
        model,
        "modelId",
        "model_id",
    )
    expected_model = str(expected_model_id or "").strip()
    if not expected_model or model_id != expected_model:
        return False, "GGUF layer-range equivalence probe model id does not match."

    gguf_file = (
        _text_value(report, "ggufFile", "gguf_file", "modelPath", "modelFile")
        or _text_value(model, "ggufFile", "gguf_file", "modelPath", "modelFile")
    )
    if not gguf_file:
        return False, "GGUF layer-range equivalence probe model file is missing."

    architecture = normalize_gguf_architecture(
        _text_value(report, "architecture", "ggufArchitecture")
        or _text_value(model, "architecture", "ggufArchitecture", "family"),
        model_id=model_id,
        filename=gguf_file,
    )
    expected_arch = str(expected_architecture or "").strip().lower()
    if architecture != expected_arch:
        return False, "GGUF layer-range equivalence probe architecture does not match."

    gguf_size = _int_value(report, "ggufSizeBytes", "gguf_size_bytes") or _int_value(
        model,
        "ggufSizeBytes",
        "gguf_size_bytes",
        "sizeBytes",
    )
    if gguf_size is None or gguf_size <= 0:
        return False, "GGUF layer-range equivalence probe GGUF size is invalid."

    gguf_sha256 = (
        _text_value(report, "ggufSha256Hex", "gguf_sha256_hex", "sha256Hex")
        or _text_value(model, "ggufSha256Hex", "gguf_sha256_hex", "sha256Hex")
    )
    if not _is_sha256_hex(gguf_sha256):
        return False, "GGUF layer-range equivalence probe GGUF hash is invalid."

    device_mode = (
        _text_value(report, "deviceMode", "backendMode", "executionMode", "computeMode")
        or _text_value(
            execution,
            "deviceMode",
            "backendMode",
            "executionMode",
            "computeMode",
        )
    )
    if not device_mode:
        return False, "GGUF layer-range equivalence probe execution mode is missing."

    if not _bool_true(report, "realLayerExecution") and not _bool_true(
        result,
        "realLayerExecution",
    ):
        return False, "GGUF layer-range equivalence probe requires realLayerExecution=true."

    total_layers = (
        _int_value(report, "nLayer", "nLayers", "totalLayers", "totalLayerCount")
        or _int_value(
            layer_range,
            "nLayer",
            "nLayers",
            "totalLayers",
            "totalLayerCount",
        )
    )
    if total_layers is None or total_layers <= 0:
        return False, "GGUF layer-range equivalence probe total layer count is invalid."

    split_layer = _int_value(report, "splitLayer", "split_layer") or _int_value(
        layer_range,
        "splitLayer",
        "split_layer",
    )
    if split_layer is None or split_layer <= 0 or split_layer >= total_layers:
        return False, "GGUF layer-range equivalence probe split layer is invalid."

    token_count = _int_value(report, "tokenCount", "token_count", "nTokens") or _int_value(
        layer_range,
        "tokenCount",
        "token_count",
        "nTokens",
    )
    if token_count is None or token_count <= 0:
        return False, "GGUF layer-range equivalence probe token count is invalid."

    activation_shape = _shape_value(report, "activationShape", "activation_shape")
    if activation_shape is None:
        activation_shape = _shape_value(layer_range, "activationShape", "activation_shape")
    if activation_shape is None:
        activation_shape = _activation_shape_from_counts(
            token_count=token_count,
            embedding_size=(
                _int_value(report, "nEmbedding", "embeddingSize")
                or _int_value(layer_range, "nEmbedding", "embeddingSize")
            ),
            activation_float_count=(
                _int_value(report, "activationFloatCount")
                or _int_value(layer_range, "activationFloatCount")
            ),
        )
    if activation_shape is None:
        return False, "GGUF layer-range equivalence probe activation shape is invalid."

    activation_float_count = _int_value(report, "activationFloatCount") or _int_value(
        layer_range,
        "activationFloatCount",
    )
    if activation_float_count is not None:
        shape_float_count = 1
        for dimension in activation_shape:
            shape_float_count *= dimension
        if activation_float_count != shape_float_count:
            return False, (
                "GGUF layer-range equivalence probe activation shape does not match "
                "activationFloatCount."
            )

    if not _bool_true(report, "topTokenMatch") and not _bool_true(result, "topTokenMatch"):
        return False, "GGUF layer-range equivalence probe requires topTokenMatch=true."

    max_abs_diff = _float_value(report, "maxAbsDiff", "max_abs_diff")
    if max_abs_diff is None:
        max_abs_diff = _float_value(result, "maxAbsDiff", "max_abs_diff")
    if max_abs_diff is None or max_abs_diff > max_abs_diff_threshold:
        return False, "GGUF layer-range equivalence probe maxAbsDiff is too high."

    mean_abs_diff = _float_value(report, "meanAbsDiff", "mean_abs_diff")
    if mean_abs_diff is None:
        mean_abs_diff = _float_value(result, "meanAbsDiff", "mean_abs_diff")
    if mean_abs_diff is None or mean_abs_diff > mean_abs_diff_threshold:
        return False, "GGUF layer-range equivalence probe meanAbsDiff is too high."

    return True, None


def load_gguf_layer_range_equivalence_probe_report(
    report_path: str | Path,
    *,
    expected_model_id: str,
    expected_architecture: str,
    max_abs_diff_threshold: float = GGUF_LAYER_RANGE_MAX_ABS_DIFF_THRESHOLD,
    mean_abs_diff_threshold: float = GGUF_LAYER_RANGE_MEAN_ABS_DIFF_THRESHOLD,
) -> dict[str, Any]:
    path = Path(report_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("GGUF layer-range equivalence probe report must be a JSON object.")
    valid, error = validate_gguf_layer_range_equivalence_probe_report(
        report,
        expected_model_id=expected_model_id,
        expected_architecture=expected_architecture,
        max_abs_diff_threshold=max_abs_diff_threshold,
        mean_abs_diff_threshold=mean_abs_diff_threshold,
    )
    if not valid:
        raise ValueError(error or "GGUF layer-range equivalence probe report is invalid.")
    return report


def load_gguf_layer_range_conformance_report(
    report_path: str | Path,
    *,
    expected_model_id: str,
    expected_architecture: str,
) -> dict[str, Any]:
    path = Path(report_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("GGUF layer-range conformance report must be a JSON object.")
    valid, error = validate_gguf_layer_range_conformance_report(
        report,
        expected_model_id=expected_model_id,
        expected_architecture=expected_architecture,
    )
    if not valid:
        raise ValueError(error or "GGUF layer-range conformance report is invalid.")
    return report


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text_value(mapping: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _int_value(mapping: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = mapping.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _float_value(mapping: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _bool_true(mapping: Mapping[str, Any], key: str) -> bool:
    value = mapping.get(key)
    if isinstance(value, bool):
        return value is True
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _shape_value(mapping: Mapping[str, Any], *keys: str) -> list[int] | None:
    for key in keys:
        value = mapping.get(key)
        if value is None or isinstance(value, (str, bytes)):
            continue
        if not isinstance(value, Sequence):
            continue
        shape: list[int] = []
        valid = True
        for item in value:
            if isinstance(item, bool):
                valid = False
                break
            try:
                dimension = int(item)
            except (TypeError, ValueError):
                valid = False
                break
            if dimension <= 0:
                valid = False
                break
            shape.append(dimension)
        if valid and shape:
            return shape
    return None


def _activation_shape_from_counts(
    *,
    token_count: int,
    embedding_size: int | None,
    activation_float_count: int | None,
) -> list[int] | None:
    if embedding_size is None or embedding_size <= 0:
        return None
    if activation_float_count is None or activation_float_count <= 0:
        return None
    if token_count * embedding_size != activation_float_count:
        return None
    return [token_count, embedding_size]


def _is_sha256_hex(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)

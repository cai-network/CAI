# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .gguf_shard_policy import GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS


LLAMA_ARCH_ENTRY_RE = re.compile(
    r'\{\s*LLM_ARCH_[A-Z0-9_]+,\s*"(?P<name>[^"]+)"\s*\}'
)

PROVEN_STATUS = "layer_range_supported"
CANDIDATE_STATUS = "candidate_requires_hook_probe_conformance"
SEPARATE_CONTRACT_STATUS = "requires_separate_state_contract"
NON_DECODER_STATUS = "non_decoder_or_multimodal"

BIG_MODEL_DENSE_PRIORITY_ARCHITECTURES = (
    "llama",
    "qwen",
    "qwen2",
    "qwen3",
    "gemma",
    "gemma2",
    "gemma3",
    "mistral3",
    "mistral",
    "falcon",
    "bloom",
    "gptneox",
    "mpt",
    "command-r",
    "cohere2",
    "granite",
    "nemotron",
    "glm4",
    "chatglm",
    "deepseek",
    "deepseek2",
    "exaone",
    "exaone4",
    "internlm2",
    "baichuan",
    "xverse",
)


@dataclass(frozen=True)
class GgufArchitectureMatrixRow:
    architecture: str
    status: str
    tier: str
    note: str

    def to_json(self) -> dict[str, str]:
        return {
            "architecture": self.architecture,
            "status": self.status,
            "tier": self.tier,
            "note": self.note,
        }


def parse_llama_architecture_names(text: str) -> tuple[str, ...]:
    names: list[str] = []
    for match in LLAMA_ARCH_ENTRY_RE.finditer(str(text or "")):
        name = match.group("name").strip()
        if not name or name == "(unknown)" or name in names:
            continue
        names.append(name)
    return tuple(names)


def discover_llama_architecture_names(path: str | Path) -> tuple[str, ...]:
    return parse_llama_architecture_names(Path(path).read_text(encoding="utf-8"))


def default_llama_arch_cpp_candidates(repo_root: str | Path | None = None) -> tuple[Path, ...]:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    return (
        root / "cai/.runtime/llama.cpp/windows-patched/source/src/llama-arch.cpp",
        root / "cai/.runtime/llama.cpp/upstream-source/src/llama-arch.cpp",
        root / "cai/.runtime/llama.cpp/windows/source/src/llama-arch.cpp",
        root / "cai/.runtime/llama.cpp/wsl/source/src/llama-arch.cpp",
    )


def resolve_default_llama_arch_cpp(repo_root: str | Path | None = None) -> Path:
    for candidate in default_llama_arch_cpp_candidates(repo_root):
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find llama-arch.cpp under local cai/.runtime/llama.cpp trees. "
        "Pass --llama-arch-cpp explicitly after fetching/building llama.cpp."
    )


def classify_gguf_architecture(architecture: str) -> GgufArchitectureMatrixRow:
    name = str(architecture or "").strip().lower()
    if not name:
        return GgufArchitectureMatrixRow(
            architecture="unknown",
            status=SEPARATE_CONTRACT_STATUS,
            tier="unknown",
            note="Missing architecture must stay unsupported until a GGUF header is available.",
        )
    if name in GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS:
        proof = GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS[name]
        return GgufArchitectureMatrixRow(
            architecture=name,
            status=PROVEN_STATUS,
            tier="proven",
            note=f"Proof gate exists; reference model: {proof.reference_model_id}.",
        )
    if _looks_non_decoder_or_multimodal(name):
        return GgufArchitectureMatrixRow(
            architecture=name,
            status=NON_DECODER_STATUS,
            tier="not-first-wave",
            note="Needs a separate non-text-decoder or multimodal handoff plan.",
        )
    if _looks_moe_or_hybrid(name):
        return GgufArchitectureMatrixRow(
            architecture=name,
            status=SEPARATE_CONTRACT_STATUS,
            tier="moe-or-hybrid",
            note="Needs an architecture-specific state contract beyond dense layer-range.",
        )
    if _looks_recurrent_or_linear(name):
        return GgufArchitectureMatrixRow(
            architecture=name,
            status=SEPARATE_CONTRACT_STATUS,
            tier="recurrent-or-linear",
            note="Needs a recurrent/linear state boundary before layer-range admission.",
        )
    if _looks_non_autoregressive_or_diffusion(name):
        return GgufArchitectureMatrixRow(
            architecture=name,
            status=SEPARATE_CONTRACT_STATUS,
            tier="non-autoregressive",
            note="Needs a non-standard decode/state contract before layer-range admission.",
        )
    return GgufArchitectureMatrixRow(
        architecture=name,
        status=CANDIDATE_STATUS,
        tier="dense-decoder-candidate",
        note="Candidate: add graph hook, run local equivalence probe, then production conformance.",
    )


def build_gguf_architecture_matrix(
    architectures: Iterable[str],
) -> tuple[GgufArchitectureMatrixRow, ...]:
    rows = [classify_gguf_architecture(name) for name in architectures]
    return tuple(sorted(rows, key=lambda row: (row.tier, row.architecture)))


def render_markdown_matrix(
    rows: Sequence[GgufArchitectureMatrixRow],
    *,
    source: str,
) -> str:
    counts = Counter(row.status for row in rows)
    lines = [
        "# GGUF Architecture Support Matrix",
        "",
        "Generated from local `llama.cpp` architecture registry:",
        f"`{source}`",
        "",
        "This matrix is intentionally conservative: a GGUF architecture is marked",
        "`layer_range_supported` only when it has both a local equivalence probe and",
        "a production binary conformance report registered in",
        "`GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS`.",
        "",
        "## Summary",
        "",
        f"- Total known llama.cpp architectures: `{len(rows)}`",
        f"- Proven CAI layer-range architectures: `{counts.get(PROVEN_STATUS, 0)}`",
        f"- Dense decoder candidates: `{counts.get(CANDIDATE_STATUS, 0)}`",
        f"- Architectures requiring separate state contract: `{counts.get(SEPARATE_CONTRACT_STATUS, 0)}`",
        f"- Non-decoder/multimodal first-wave exclusions: `{counts.get(NON_DECODER_STATUS, 0)}`",
        "",
        "## Add-New-Architecture Flow",
        "",
        "1. Download or point to a small GGUF candidate for the exact",
        "   `general.architecture` value.",
        "2. Read GGUF metadata; `general.architecture` and `{architecture}.block_count`",
        "   are authoritative.",
        "3. If no graph hook exists, add an architecture-specific `llama.cpp` patch",
        "   for layer start/end, boundary activation output, activation input",
        "   continuation, and output-row selection.",
        "4. Run `tools/run-gguf-layer-range-probe.ps1` and keep the report under",
        "   `docs/reports/`.",
        "5. Run `tools/run-gguf-production-conformance.ps1` with strict production",
        "   guards and keep the report under `docs/reports/`.",
        "6. Only then add the architecture to `GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS`.",
        "",
        "Automation boundary: discovery, metadata reading, split-layer selection,",
        "report validation, and matrix updates can be automated. The graph hook and",
        "state contract still need human review because GGUF is a container, not one",
        "universal computation graph.",
        "",
        "## Big-Model Priority Orientation",
        "",
        "Expansion should favor architectures that have large real-world dense GGUF",
        "families, while proofs can use the smallest available GGUF with the same",
        "`general.architecture` header. Current priority queue:",
        "",
        "`" + "`, `".join(BIG_MODEL_DENSE_PRIORITY_ARCHITECTURES) + "`",
        "",
        "MoE, hybrid/recurrent, embedding, and multimodal variants stay out of this",
        "dense layer-range queue until they have a separate state-contract plan.",
        "",
        "## Matrix",
        "",
        "| Architecture | CAI Status | Tier | Note |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(value)
                for value in (row.architecture, row.status, row.tier, row.note)
            )
            + " |"
        )
    return "\n".join(lines)


def _looks_non_decoder_or_multimodal(name: str) -> bool:
    compact = name.replace("-", "").replace("_", "").replace(".", "")
    if name in {
        "clip",
        "paddleocr",
        "wavtokenizer-dec",
        "cogvlm",
        "chameleon",
        "gemma3n",
    }:
        return True
    if any(token in compact for token in ("vl", "vision", "ocr", "audio", "omni")):
        return True
    if any(token in name for token in ("bert", "encoder", "embedding", "embed")):
        return True
    return name in {"t5", "t5encoder", "plm", "pangu-embedded"}


def _looks_moe_or_hybrid(name: str) -> bool:
    compact = name.replace("-", "").replace("_", "").replace(".", "")
    if "moe" in compact:
        return True
    return name in {
        "arctic",
        "dbrx",
        "falcon-h1",
        "gpt-oss",
        "jamba",
        "llama4",
        "minimax-m2",
        "granitehybrid",
    }


def _looks_recurrent_or_linear(name: str) -> bool:
    compact = name.replace("-", "").replace("_", "").replace(".", "")
    return name in {"qwen35", "qwen3next"} or any(
        token in compact for token in ("mamba", "rwkv", "linear")
    )


def _looks_non_autoregressive_or_diffusion(name: str) -> bool:
    return name in {"dream", "llada"}


def _markdown_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _display_source_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    for base in (Path.cwd(), Path(__file__).resolve().parents[2]):
        try:
            return resolved.relative_to(base.expanduser().resolve()).as_posix()
        except ValueError:
            continue
    return str(resolved)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List GGUF architectures known to local llama.cpp and CAI support status.",
    )
    parser.add_argument("--llama-arch-cpp", default="", help="Path to llama-arch.cpp.")
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format.",
    )
    parser.add_argument("--output", default="", help="Optional output file path.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    source_path = (
        Path(args.llama_arch_cpp).expanduser().resolve()
        if str(args.llama_arch_cpp or "").strip()
        else resolve_default_llama_arch_cpp()
    )
    rows = build_gguf_architecture_matrix(discover_llama_architecture_names(source_path))
    if args.format == "json":
        rendered = json.dumps(
            {
                "source": str(source_path),
                "architectures": [row.to_json() for row in rows],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    else:
        rendered = render_markdown_matrix(rows, source=_display_source_path(source_path))
    if args.output:
        Path(args.output).expanduser().resolve().write_text(
            rendered + "\n",
            encoding="utf-8",
        )
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

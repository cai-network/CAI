# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from pathlib import Path

from cai_compute_chain.cai_llama_cpp_shard_native_bridge import (
    _candidate_local_gguf_model_artifact_paths,
)


def test_candidate_local_gguf_paths_include_standard_model_id_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gguf_path = (
        tmp_path
        / "data"
        / ".cai"
        / "models"
        / "Qwen--Qwen3-0.6B-GGUF"
        / "Qwen3-0.6B-Q8_0.gguf"
    )
    gguf_path.parent.mkdir(parents=True)
    gguf_path.write_bytes(b"gguf")
    monkeypatch.setenv("CAI_REPO_ROOT", str(tmp_path))

    candidates = _candidate_local_gguf_model_artifact_paths(
        "Qwen/Qwen3-0.6B-GGUF",
        artifact_hint=None,
        preferred_filename="Qwen3-0.6B-Q8_0.gguf",
    )

    assert gguf_path.resolve() in candidates

# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
from pathlib import Path

from cai_compute_chain.model_distribution import (
    ChunkCacheClass,
    ModelChunk,
    ModelChunkKind,
    ModelPackageManifest,
    ModelShardAssignment,
    SourceArtifact,
    materialize_default_assignment_artifact_from_store,
    put_cached_chunk,
)


def _chunk(
    *,
    chunk_id: str,
    artifact_id: str,
    payload: bytes,
    offset_bytes: int,
    layer_start: int | None = None,
    layer_end: int | None = None,
    required_by_default: bool = False,
) -> ModelChunk:
    return ModelChunk(
        chunk_id=chunk_id,
        artifact_id=artifact_id,
        kind=ModelChunkKind.METADATA
        if required_by_default
        else ModelChunkKind.WEIGHTS,
        offset_bytes=offset_bytes,
        size_bytes=len(payload),
        sha256_hex=hashlib.sha256(payload).hexdigest(),
        layer_start=layer_start,
        layer_end=layer_end,
        required_by_default=required_by_default,
    )


def test_materialize_assignment_repairs_corrupt_existing_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CAI_WALLET_HOME", str(tmp_path / ".cai-local"))
    artifact_id = "gguf-main"
    header_payload = b"GGUF-test-header"
    layer_payload = b"layer-0-weights"
    full_payload = header_payload + layer_payload
    chunks = [
        _chunk(
            chunk_id="header",
            artifact_id=artifact_id,
            payload=header_payload,
            offset_bytes=0,
            required_by_default=True,
        ),
        _chunk(
            chunk_id="layer-0",
            artifact_id=artifact_id,
            payload=layer_payload,
            offset_bytes=len(header_payload),
            layer_start=0,
            layer_end=1,
        ),
    ]
    manifest = ModelPackageManifest(
        catalog_id="test-qwen",
        model_id="Qwen/Qwen3-0.6B-GGUF",
        version="local-test",
        backend="llama.cpp",
        total_size_bytes=len(full_payload),
        files=[
            SourceArtifact(
                artifact_id=artifact_id,
                relative_path="Qwen3-0.6B-Q8_0.gguf",
                size_bytes=len(full_payload),
                sha256_hex=hashlib.sha256(full_payload).hexdigest(),
                media_type="application/gguf",
            )
        ],
        chunks=chunks,
    )
    for chunk, payload in zip(chunks, (header_payload, layer_payload), strict=True):
        put_cached_chunk(
            catalog_id=manifest.catalog_id,
            version=manifest.version,
            chunk_id=chunk.chunk_id,
            sha256_hex=chunk.sha256_hex,
            content=payload,
            cache_class=ChunkCacheClass.HOT,
        )

    assignment = ModelShardAssignment(start_layer=0, end_layer=1)
    first = materialize_default_assignment_artifact_from_store(
        manifest,
        assignment,
        overwrite=True,
    )
    output_path = Path(first.output_path)
    output_path.write_bytes(b"\0" * len(full_payload))

    repaired = materialize_default_assignment_artifact_from_store(
        manifest,
        assignment,
        overwrite=False,
    )

    assert Path(repaired.output_path) == output_path
    assert output_path.read_bytes() == full_payload
    assert repaired.sha256_hex == first.sha256_hex

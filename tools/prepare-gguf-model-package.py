# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.model import WalletPolicy  # noqa: E402
from cai_compute_chain.model_distribution import (  # noqa: E402
    ChunkCacheClass,
    ModelPackageKind,
    build_gguf_model_package_manifest,
    put_cached_chunk,
    save_local_artifact_binding,
    save_model_package_manifest,
)


def _read_chunk(path: Path, offset: int, size: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(int(offset))
        payload = handle.read(int(size))
    if len(payload) != int(size):
        raise ValueError(f"Could not read full GGUF chunk at offset {offset}.")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a local GGUF model package manifest and chunk cache.",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--catalog-id", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--architecture", default="")
    parser.add_argument("--total-layers", type=int, default=0)
    parser.add_argument("--quantization", default="")
    parser.add_argument("--source-repo-id", default="")
    parser.add_argument("--source-revision", default="main")
    parser.add_argument("--wallet-data-dirname", default="")
    parser.add_argument("--target-chunk-count", type=int, default=4)
    parser.add_argument("--min-chunk-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-chunk-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--cache-chunks", action="store_true")
    parser.add_argument("--pin-chunks", action="store_true")
    parser.add_argument("--json-report", default="")
    args = parser.parse_args(argv)

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"GGUF model file not found: {model_path}")
    policy = (
        WalletPolicy(wallet_data_dirname=args.wallet_data_dirname)
        if args.wallet_data_dirname.strip()
        else WalletPolicy()
    )
    manifest = build_gguf_model_package_manifest(
        catalog_id=args.catalog_id,
        model_id=args.model_id,
        version=args.version,
        gguf_path=model_path,
        total_layers=int(args.total_layers) if int(args.total_layers) > 0 else None,
        package_kind=ModelPackageKind.PRIVATE_CURATED,
        min_chunk_bytes=int(args.min_chunk_bytes),
        max_chunk_bytes=int(args.max_chunk_bytes),
        target_chunk_count=max(1, int(args.target_chunk_count)),
        source_repo_id=args.source_repo_id.strip() or args.model_id,
        source_revision=args.source_revision.strip() or "main",
        family=args.architecture.strip(),
        quantization=args.quantization,
    )
    manifest_path = save_model_package_manifest(manifest, policy)
    binding_path = save_local_artifact_binding(
        manifest.catalog_id,
        manifest.version,
        artifact_id="gguf-main",
        local_path=model_path,
        policy=policy,
    )
    cached_chunk_count = 0
    cached_bytes = 0
    if args.cache_chunks:
        for chunk in manifest.chunks:
            payload = _read_chunk(model_path, chunk.offset_bytes, chunk.size_bytes)
            put_cached_chunk(
                catalog_id=manifest.catalog_id,
                version=manifest.version,
                chunk_id=chunk.chunk_id,
                sha256_hex=chunk.sha256_hex,
                content=payload,
                pinned=bool(args.pin_chunks),
                cache_class=(
                    ChunkCacheClass.HOT if args.pin_chunks else ChunkCacheClass.WARM
                ),
                policy=policy,
            )
            cached_chunk_count += 1
            cached_bytes += len(payload)

    summary = {
        "status": "ok",
        "modelId": manifest.model_id,
        "catalogId": manifest.catalog_id,
        "version": manifest.version,
        "family": manifest.family,
        "ggufArchitecture": manifest.metadata.get("gguf_architecture"),
        "quantization": manifest.quantization,
        "preferredFilename": manifest.preferred_filename,
        "totalLayers": int(manifest.metadata.get("total_layers") or 0),
        "manifestPath": str(manifest_path),
        "localArtifactBindingPath": str(binding_path),
        "manifestChunkCount": len(manifest.chunks),
        "cachedChunkCount": cached_chunk_count,
        "cachedBytes": cached_bytes,
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.json_report.strip():
        output_path = Path(args.json_report).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import sys
import tempfile
import unittest
import hashlib
import struct
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.model import (
    ChunkCachePolicy,
    ChunkFetchPolicy,
    ChunkStorageAccountingPolicy,
    WalletPolicy,
)
from cai_compute_chain.model_distribution import (
    apply_assignment_cache_policy_from_store,
    AssignmentChunkPlan,
    AssignmentFetchPlan,
    ChunkCoverage,
    ChunkCacheClass,
    ChunkDownloadTaskStatus,
    ChunkFetchSource,
    ChunkFetchSourceKind,
    ChunkInventoryPayload,
    ChunkInventoryRecord,
    ChunkInventorySourceKind,
    ChunkLeaseStatus,
    ChunkSizePolicy,
    ModelShardAssignment,
    ModelChunk,
    ModelChunkKind,
    ModelManifestValidationError,
    ModelPackageKind,
    ModelPackageManifest,
    build_assignment_chunk_plan_from_store,
    build_chunk_inventory_index,
    build_chunk_inventory_locator_index,
    build_assignment_fetch_plan_from_store,
    build_bootstrap_chunk_fetch_plan_from_store,
    chunk_download_queue_snapshot,
    cached_chunk_path,
    build_gguf_model_package_manifest,
    build_local_chunk_inventory_payload,
    chunk_source_bindings_path,
    chunk_store_snapshot,
    ensure_default_chunks_ready_from_store,
    get_chunk_source_binding,
    ensure_assignment_ready_from_store,
    execute_chunk_download_queue,
    export_chunk_inventory_payload,
    build_source_artifact_from_file,
    build_weight_chunks_for_artifact,
    SourceArtifact,
    delete_cached_chunk,
    evict_chunks_to_policy_target,
    get_cached_chunk_record,
    get_chunk_source_health_record,
    load_local_artifact_bindings,
    load_model_package_manifest,
    load_chunk_inventory_payload,
    list_cached_chunks,
    list_chunk_source_health_records,
    list_chunk_storage_accounting_records,
    list_imported_chunk_inventory_payloads,
    list_chunk_download_tasks,
    load_chunk_source_bindings,
    materialize_artifact_from_store,
    materialize_default_assignment_artifact_from_store,
    materialize_default_artifact_from_store,
    materialized_assignment_artifact_path,
    materialized_artifact_path,
    mark_cached_chunk_used,
    make_chunk_id,
    list_recent_shard_hints,
    prefetch_recent_shard_hints,
    put_cached_chunk,
    prefetch_bootstrap_chunks_from_fresh_inventories,
    prefetch_hinted_bootstrap_chunks,
    prefetch_default_chunks_from_fresh_inventories,
    prune_imported_chunk_inventory_payloads,
    prune_chunk_source_health_records,
    queue_assignment_fetch_plan,
    read_gguf_model_metadata,
    record_chunk_source_failure,
    record_chunk_storage_accounting_snapshot,
    remember_recent_shard_hints,
    release_assignment_cache_policy_from_store,
    save_chunk_source_binding,
    save_local_artifact_binding,
    save_chunk_inventory_payload,
    save_model_package_manifest,
    import_chunk_inventory_payload,
    select_model_package_manifest_for_model,
    sync_chunk_inventory_from_cai_peers,
    sync_chunk_inventory_from_urls,
    update_chunk_download_task_status,
)
from cai_compute_chain.wallet import data_root


def _fresh_published_at() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


def _build_manifest() -> ModelPackageManifest:
    gguf_sha = "a" * 64
    metadata_sha = "b" * 64
    return ModelPackageManifest(
        catalog_id="qwen3-0.6b-q4",
        model_id="Qwen/Qwen3-0.6B-GGUF",
        version="2026.04.25",
        backend="llama_cpp",
        package_kind=ModelPackageKind.PUBLIC_SHARED,
        chunk_size_policy=ChunkSizePolicy.ADAPTIVE,
        total_size_bytes=1_000,
        source_repo_id="Qwen/Qwen3-0.6B-GGUF",
        preferred_filename="qwen3-0.6b-q4_k_m.gguf",
        files=[
            SourceArtifact(
                artifact_id="gguf-main",
                relative_path="qwen3-0.6b-q4_k_m.gguf",
                size_bytes=900,
                sha256_hex=gguf_sha,
            ),
            SourceArtifact(
                artifact_id="tokenizer",
                relative_path="tokenizer.json",
                size_bytes=100,
                sha256_hex=metadata_sha,
            ),
        ],
        chunks=[
            ModelChunk(
                chunk_id=make_chunk_id(
                    "tokenizer",
                    offset_bytes=0,
                    size_bytes=100,
                    sha256_hex=metadata_sha,
                ),
                artifact_id="tokenizer",
                kind=ModelChunkKind.TOKENIZER,
                offset_bytes=0,
                size_bytes=100,
                sha256_hex=metadata_sha,
                required_by_default=True,
            ),
            ModelChunk(
                chunk_id=make_chunk_id(
                    "gguf-main",
                    offset_bytes=0,
                    size_bytes=300,
                    sha256_hex="c" * 64,
                ),
                artifact_id="gguf-main",
                kind=ModelChunkKind.WEIGHTS,
                offset_bytes=0,
                size_bytes=300,
                sha256_hex="c" * 64,
                layer_start=0,
                layer_end=10,
            ),
            ModelChunk(
                chunk_id=make_chunk_id(
                    "gguf-main",
                    offset_bytes=300,
                    size_bytes=250,
                    sha256_hex="d" * 64,
                ),
                artifact_id="gguf-main",
                kind=ModelChunkKind.WEIGHTS,
                offset_bytes=300,
                size_bytes=250,
                sha256_hex="d" * 64,
                layer_start=10,
                layer_end=18,
            ),
            ModelChunk(
                chunk_id=make_chunk_id(
                    "gguf-main",
                    offset_bytes=550,
                    size_bytes=350,
                    sha256_hex="e" * 64,
                ),
                artifact_id="gguf-main",
                kind=ModelChunkKind.WEIGHTS,
                offset_bytes=550,
                size_bytes=350,
                sha256_hex="e" * 64,
                layer_start=18,
                layer_end=28,
            ),
        ],
    )


def _write_minimal_gguf_file(
    path: Path,
    *,
    architecture: str | None = None,
    block_count: int | None = None,
    embedding_length: int | None = None,
) -> Path:
    tensors = (
        ("token_embd.weight", b"EMBED000"),
        ("blk.0.attn_q.weight", b"LAYER000"),
        ("blk.1.attn_q.weight", b"LAYER111"),
        ("output.weight", b"OUTPUT00"),
    )

    def _gguf_string(value: str) -> bytes:
        encoded = value.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    kv_entries: list[tuple[str, int, bytes]] = [
        ("general.alignment", 4, struct.pack("<I", 32)),
    ]
    if architecture:
        kv_entries.append(("general.architecture", 8, _gguf_string(architecture)))
        if block_count is not None:
            kv_entries.append(
                (f"{architecture}.block_count", 4, struct.pack("<I", block_count))
            )
        if embedding_length is not None:
            kv_entries.append(
                (
                    f"{architecture}.embedding_length",
                    4,
                    struct.pack("<I", embedding_length),
                )
            )

    payload = bytearray()
    payload.extend(b"GGUF")
    payload.extend(struct.pack("<I", 3))
    payload.extend(struct.pack("<Q", len(tensors)))
    payload.extend(struct.pack("<Q", len(kv_entries)))
    for key, value_type, value_payload in kv_entries:
        payload.extend(_gguf_string(key))
        payload.extend(struct.pack("<I", value_type))
        payload.extend(value_payload)

    relative_offset = 0
    tensor_payload = bytearray()
    for name, tensor_bytes in tensors:
        payload.extend(_gguf_string(name))
        payload.extend(struct.pack("<I", 1))
        payload.extend(struct.pack("<Q", len(tensor_bytes)))
        payload.extend(struct.pack("<I", 0))
        payload.extend(struct.pack("<Q", relative_offset))
        tensor_payload.extend(tensor_bytes)
        relative_offset += len(tensor_bytes)

    while len(payload) % 32 != 0:
        payload.append(0)
    payload.extend(tensor_payload)
    path.write_bytes(bytes(payload))
    return path


class ModelDistributionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo_patch = patch(
            "cai_compute_chain.wallet.repo_root",
            return_value=Path(self.tempdir.name),
        )
        self.repo_patch.start()

    def tearDown(self) -> None:
        self.repo_patch.stop()
        self.tempdir.cleanup()

    def _write_temp_file(self, name: str, content: bytes) -> Path:
        path = Path(self.tempdir.name) / name
        path.write_bytes(content)
        return path

    def test_manifest_roundtrip_persists_variable_sized_chunks(self) -> None:
        manifest = _build_manifest()
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-test")
        saved_path = save_model_package_manifest(manifest, policy)
        loaded = load_model_package_manifest(
            manifest.catalog_id,
            manifest.version,
            policy,
        )
        self.assertEqual(loaded.catalog_id, manifest.catalog_id)
        self.assertEqual(loaded.chunk_size_policy, ChunkSizePolicy.ADAPTIVE)
        self.assertEqual(
            [chunk.size_bytes for chunk in loaded.chunks],
            [100, 300, 250, 350],
        )
        self.assertEqual(saved_path.name, "manifest.json")

    def test_select_model_package_manifest_for_model_prefers_latest_version(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-select")
        older = _build_manifest()
        older.version = "2026.04.24"
        newer = _build_manifest()
        newer.version = "2026.04.25"
        save_model_package_manifest(older, policy)
        save_model_package_manifest(newer, policy)

        selected = select_model_package_manifest_for_model(older.model_id, policy)

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.version, "2026.04.25")

    def test_required_chunks_for_layers_include_default_metadata(self) -> None:
        manifest = _build_manifest()
        chunks = manifest.required_chunks_for_layers(11, 20)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(any(chunk.kind == ModelChunkKind.TOKENIZER for chunk in chunks))
        self.assertTrue(
            any(chunk.layer_start == 10 and chunk.layer_end == 18 for chunk in chunks)
        )
        self.assertTrue(
            any(chunk.layer_start == 18 and chunk.layer_end == 28 for chunk in chunks)
        )

    def test_assignment_plan_rejects_manifest_layer_gaps(self) -> None:
        manifest = _build_manifest()
        manifest.chunks = [
            chunk
            for chunk in manifest.chunks
            if not (chunk.kind == ModelChunkKind.WEIGHTS and chunk.layer_start == 10)
        ]
        assignment = ModelShardAssignment(start_layer=0, end_layer=28)

        with self.assertRaises(ModelManifestValidationError) as context:
            manifest.build_assignment_chunk_plan(
                assignment,
                present_chunk_ids={chunk.chunk_id for chunk in manifest.chunks},
            )

        self.assertIn("missing layers 10..18", str(context.exception))

    def test_compute_chunk_coverage_counts_bytes_from_present_cache(self) -> None:
        manifest = _build_manifest()
        required = manifest.required_chunks_for_layers(0, 18)
        tokenizer_chunk = next(
            chunk for chunk in required if chunk.kind == ModelChunkKind.TOKENIZER
        )
        first_weight_chunk = next(
            chunk
            for chunk in required
            if chunk.layer_start == 0 and chunk.layer_end == 10
        )
        coverage = manifest.compute_chunk_coverage(
            {tokenizer_chunk.chunk_id, first_weight_chunk.chunk_id},
            start_layer=0,
            end_layer=18,
        )
        self.assertIsInstance(coverage, ChunkCoverage)
        self.assertFalse(coverage.ready)
        self.assertEqual(coverage.required_bytes, 650)
        self.assertEqual(coverage.present_bytes, 400)
        self.assertEqual(len(coverage.missing_chunk_ids), 1)

    def test_manifest_validation_rejects_unknown_artifact_reference(self) -> None:
        manifest = _build_manifest()
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-test-invalid")
        manifest.chunks.append(
            ModelChunk(
                chunk_id="broken",
                artifact_id="missing-artifact",
                kind=ModelChunkKind.WEIGHTS,
                offset_bytes=0,
                size_bytes=10,
                sha256_hex="f" * 64,
                layer_start=0,
                layer_end=1,
            )
        )
        with self.assertRaises(ModelManifestValidationError) as context:
            save_model_package_manifest(manifest, policy)
        self.assertIn("unknown artifact_id", str(context.exception))

    def test_manifest_validation_rejects_chunk_outside_artifact_bounds(self) -> None:
        manifest = _build_manifest()
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-test-bounds")
        manifest.chunks[1] = ModelChunk(
            chunk_id="oversized",
            artifact_id="gguf-main",
            kind=ModelChunkKind.WEIGHTS,
            offset_bytes=895,
            size_bytes=10,
            sha256_hex="f" * 64,
            layer_start=0,
            layer_end=1,
        )

        with self.assertRaises(ModelManifestValidationError) as context:
            save_model_package_manifest(manifest, policy)

        self.assertIn("exceeds artifact bounds", str(context.exception))

    def test_assignment_chunk_plan_reports_missing_fetch_bytes(self) -> None:
        manifest = _build_manifest()
        assignment = ModelShardAssignment(
            start_layer=10,
            end_layer=28,
            device_rank=1,
            world_size=3,
            node_id="node-b",
        )
        tokenizer_chunk = next(
            chunk for chunk in manifest.chunks if chunk.kind == ModelChunkKind.TOKENIZER
        )
        plan = manifest.build_assignment_chunk_plan(
            assignment,
            present_chunk_ids={tokenizer_chunk.chunk_id},
        )
        self.assertIsInstance(plan, AssignmentChunkPlan)
        self.assertFalse(plan.ready)
        self.assertEqual(plan.coverage.required_bytes, 700)
        self.assertEqual(plan.coverage.present_bytes, 100)
        self.assertEqual(plan.estimated_fetch_bytes, 600)
        self.assertEqual(len(plan.coverage.missing_chunk_ids), 2)

    def test_chunk_store_put_list_mark_used_and_delete(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-chunk-store")
        payload = b"hello chunk store"
        sha256_hex = "8f3d6a6d5ef3904db4318f9d6b4cbb7f0fd0f9c80d7a8fd2f0f81f9b1f9d4c6d"
        # recomputed once to keep the test explicit and stable
        import hashlib

        sha256_hex = hashlib.sha256(payload).hexdigest()
        record = put_cached_chunk(
            catalog_id="qwen3-0.6b-q4",
            version="2026.04.25",
            chunk_id="chunk-1",
            sha256_hex=sha256_hex,
            content=payload,
            pinned=True,
            cache_class=ChunkCacheClass.HOT,
            lease_status=ChunkLeaseStatus.ACTIVE,
            lease_expires_at="2026-04-26T00:00:00+00:00",
            policy=policy,
        )
        self.assertEqual(record.size_bytes, len(payload))
        self.assertEqual(len(list_cached_chunks(policy)), 1)

        loaded = get_cached_chunk_record("chunk-1", policy)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.cache_class, ChunkCacheClass.HOT)
        self.assertEqual(loaded.lease_status, ChunkLeaseStatus.ACTIVE)

        touched = mark_cached_chunk_used("chunk-1", policy)
        self.assertIsNotNone(touched)
        assert touched is not None
        self.assertEqual(touched.use_count, 1)

        snapshot = chunk_store_snapshot(policy)
        self.assertEqual(snapshot.stats.chunk_count, 1)
        self.assertEqual(snapshot.stats.total_bytes, len(payload))
        self.assertEqual(snapshot.stats.pinned_chunk_count, 1)
        self.assertEqual(snapshot.stats.hot_chunk_count, 1)

        self.assertTrue(delete_cached_chunk("chunk-1", policy))
        self.assertEqual(chunk_store_snapshot(policy).stats.chunk_count, 0)

    def test_chunk_storage_accounting_records_byte_seconds(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-storage-accounting")
        payload = b"a" * 1024
        put_cached_chunk(
            catalog_id="demo",
            version="v1",
            chunk_id="chunk-accounting",
            sha256_hex=hashlib.sha256(payload).hexdigest(),
            content=payload,
            policy=policy,
        )
        cached = get_cached_chunk_record("chunk-accounting", policy)
        assert cached is not None
        start = datetime.fromisoformat(cached.stored_at)

        result = record_chunk_storage_accounting_snapshot(
            "node-a",
            accounting_policy=ChunkStorageAccountingPolicy(
                max_accounting_interval_seconds=3600,
                min_accounting_seconds=1,
            ),
            policy=policy,
            now=start + timedelta(seconds=3600),
        )

        self.assertEqual(len(result.records), 1)
        record = result.records[0]
        self.assertEqual(record.node_id, "node-a")
        self.assertEqual(record.size_bytes, len(payload))
        self.assertEqual(record.accounted_seconds, 3600)
        self.assertEqual(record.byte_seconds, len(payload) * 3600)
        self.assertEqual(len(list_chunk_storage_accounting_records("node-a", policy)), 1)

    def test_chunk_storage_accounting_does_not_double_count_intervals(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-storage-accounting-dedup")
        payload = b"b" * 4096
        put_cached_chunk(
            catalog_id="demo",
            version="v1",
            chunk_id="chunk-dedup",
            sha256_hex=hashlib.sha256(payload).hexdigest(),
            content=payload,
            policy=policy,
        )
        cached = get_cached_chunk_record("chunk-dedup", policy)
        assert cached is not None
        start = datetime.fromisoformat(cached.stored_at)

        first = record_chunk_storage_accounting_snapshot(
            "node-a",
            accounting_policy=ChunkStorageAccountingPolicy(
                max_accounting_interval_seconds=3600,
                min_accounting_seconds=1,
            ),
            policy=policy,
            now=start + timedelta(seconds=120),
        )
        second = record_chunk_storage_accounting_snapshot(
            "node-a",
            accounting_policy=ChunkStorageAccountingPolicy(
                max_accounting_interval_seconds=3600,
                min_accounting_seconds=1,
            ),
            policy=policy,
            now=start + timedelta(seconds=180),
        )

        self.assertEqual(len(first.records), 1)
        self.assertEqual(len(second.records), 1)
        self.assertEqual(first.records[0].accounted_seconds, 120)
        self.assertEqual(second.records[0].accounted_seconds, 60)
        persisted = list_chunk_storage_accounting_records("node-a", policy)
        self.assertEqual(len(persisted), 2)

    def test_chunk_store_gc_evicts_oldest_cold_chunks_to_target(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-chunk-store-gc")
        put_cached_chunk(
            catalog_id="demo",
            version="v1",
            chunk_id="cold-a",
            sha256_hex=hashlib.sha256(b"aaaa").hexdigest(),
            content=b"aaaa",
            cache_class=ChunkCacheClass.COLD,
            policy=policy,
        )
        put_cached_chunk(
            catalog_id="demo",
            version="v1",
            chunk_id="warm-b",
            sha256_hex=hashlib.sha256(b"bbbb").hexdigest(),
            content=b"bbbb",
            cache_class=ChunkCacheClass.WARM,
            policy=policy,
        )
        put_cached_chunk(
            catalog_id="demo",
            version="v1",
            chunk_id="hot-c",
            sha256_hex=hashlib.sha256(b"cccc").hexdigest(),
            content=b"cccc",
            cache_class=ChunkCacheClass.HOT,
            policy=policy,
        )

        result = evict_chunks_to_policy_target(
            cache_policy=ChunkCachePolicy(max_store_bytes=10, target_store_bytes=6),
            policy=policy,
        )

        self.assertTrue(result.changed)
        self.assertEqual(result.before.stats.total_bytes, 12)
        self.assertEqual(result.after.stats.total_bytes, 4)
        self.assertEqual(result.evicted_chunk_ids, ("cold-a", "warm-b"))
        self.assertIsNotNone(get_cached_chunk_record("hot-c", policy))

    def test_chunk_store_gc_preserves_pinned_and_active_leases(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-chunk-store-gc-protect")
        put_cached_chunk(
            catalog_id="demo",
            version="v1",
            chunk_id="pinned-a",
            sha256_hex=hashlib.sha256(b"aaaa").hexdigest(),
            content=b"aaaa",
            pinned=True,
            cache_class=ChunkCacheClass.COLD,
            policy=policy,
        )
        put_cached_chunk(
            catalog_id="demo",
            version="v1",
            chunk_id="leased-b",
            sha256_hex=hashlib.sha256(b"bbbb").hexdigest(),
            content=b"bbbb",
            cache_class=ChunkCacheClass.COLD,
            lease_status=ChunkLeaseStatus.ACTIVE,
            policy=policy,
        )
        put_cached_chunk(
            catalog_id="demo",
            version="v1",
            chunk_id="evict-c",
            sha256_hex=hashlib.sha256(b"cccc").hexdigest(),
            content=b"cccc",
            cache_class=ChunkCacheClass.COLD,
            policy=policy,
        )

        result = evict_chunks_to_policy_target(
            cache_policy=ChunkCachePolicy(max_store_bytes=8, target_store_bytes=4),
            policy=policy,
        )

        self.assertTrue(result.changed)
        self.assertEqual(result.evicted_chunk_ids, ("evict-c",))
        self.assertEqual(result.after.stats.total_bytes, 8)
        self.assertIsNotNone(get_cached_chunk_record("pinned-a", policy))
        self.assertIsNotNone(get_cached_chunk_record("leased-b", policy))

    def test_apply_assignment_cache_policy_promotes_required_chunks(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-assignment-cache-policy")
        gguf_payload = b"abcdefghij"
        gguf_path = self._write_temp_file("assignment-cache-policy.gguf", gguf_payload)
        manifest = build_gguf_model_package_manifest(
            catalog_id="demo-assignment-cache-policy",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            version="v1",
            gguf_path=gguf_path,
            total_layers=5,
            chunk_size_policy=ChunkSizePolicy.ADAPTIVE,
            min_chunk_bytes=4,
            max_chunk_bytes=4,
            source_repo_id="Qwen/Qwen3-0.6B-GGUF",
        )
        save_model_package_manifest(manifest, policy)
        assignment = ModelShardAssignment(start_layer=0, end_layer=5, node_id="node-a")
        required_chunks = manifest.required_chunks_for_layers(0, 5)
        for chunk in required_chunks:
            put_cached_chunk(
                catalog_id=manifest.catalog_id,
                version=manifest.version,
                chunk_id=chunk.chunk_id,
                sha256_hex=chunk.sha256_hex,
                content=gguf_payload[
                    chunk.offset_bytes : chunk.offset_bytes + chunk.size_bytes
                ],
                cache_class=ChunkCacheClass.COLD,
                policy=policy,
            )

        updated = apply_assignment_cache_policy_from_store(
            manifest,
            assignment,
            cache_policy=ChunkCachePolicy(
                assignment_lease_seconds=600,
                pin_assignment_chunks=False,
            ),
            policy=policy,
        )

        self.assertEqual(len(updated), len(required_chunks))
        for chunk in required_chunks:
            record = get_cached_chunk_record(chunk.chunk_id, policy)
            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record.cache_class, ChunkCacheClass.HOT)
            self.assertEqual(record.lease_status, ChunkLeaseStatus.ACTIVE)
            self.assertFalse(record.pinned)
            self.assertIsNotNone(record.lease_expires_at)

    def test_release_assignment_cache_policy_demotes_required_chunks(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-assignment-cache-release")
        gguf_payload = b"abcdefghij"
        gguf_path = self._write_temp_file("assignment-cache-release.gguf", gguf_payload)
        manifest = build_gguf_model_package_manifest(
            catalog_id="demo-assignment-cache-release",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            version="v1",
            gguf_path=gguf_path,
            total_layers=5,
            chunk_size_policy=ChunkSizePolicy.ADAPTIVE,
            min_chunk_bytes=4,
            max_chunk_bytes=4,
            source_repo_id="Qwen/Qwen3-0.6B-GGUF",
        )
        save_model_package_manifest(manifest, policy)
        assignment = ModelShardAssignment(start_layer=0, end_layer=5, node_id="node-a")
        required_chunks = manifest.required_chunks_for_layers(0, 5)
        for chunk in required_chunks:
            put_cached_chunk(
                catalog_id=manifest.catalog_id,
                version=manifest.version,
                chunk_id=chunk.chunk_id,
                sha256_hex=chunk.sha256_hex,
                content=gguf_payload[
                    chunk.offset_bytes : chunk.offset_bytes + chunk.size_bytes
                ],
                cache_class=ChunkCacheClass.HOT,
                lease_status=ChunkLeaseStatus.ACTIVE,
                pinned=True,
                policy=policy,
            )

        updated = release_assignment_cache_policy_from_store(
            manifest,
            assignment,
            policy=policy,
        )

        self.assertEqual(len(updated), len(required_chunks))
        for chunk in required_chunks:
            record = get_cached_chunk_record(chunk.chunk_id, policy)
            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record.cache_class, ChunkCacheClass.WARM)
            self.assertEqual(record.lease_status, ChunkLeaseStatus.EXPIRED)
            self.assertFalse(record.pinned)
            self.assertIsNotNone(record.lease_expires_at)

    def test_release_assignment_cache_policy_respects_protected_chunk_ids(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-assignment-cache-release-protect")
        gguf_payload = b"abcdefghij"
        gguf_path = self._write_temp_file("assignment-cache-release-protect.gguf", gguf_payload)
        manifest = build_gguf_model_package_manifest(
            catalog_id="demo-assignment-cache-release-protect",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            version="v1",
            gguf_path=gguf_path,
            total_layers=5,
            chunk_size_policy=ChunkSizePolicy.ADAPTIVE,
            min_chunk_bytes=4,
            max_chunk_bytes=4,
            source_repo_id="Qwen/Qwen3-0.6B-GGUF",
        )
        save_model_package_manifest(manifest, policy)
        assignment = ModelShardAssignment(start_layer=0, end_layer=5, node_id="node-a")
        required_chunks = manifest.required_chunks_for_layers(0, 5)
        protected_chunk_id = required_chunks[0].chunk_id
        for chunk in required_chunks:
            put_cached_chunk(
                catalog_id=manifest.catalog_id,
                version=manifest.version,
                chunk_id=chunk.chunk_id,
                sha256_hex=chunk.sha256_hex,
                content=gguf_payload[
                    chunk.offset_bytes : chunk.offset_bytes + chunk.size_bytes
                ],
                cache_class=ChunkCacheClass.HOT,
                lease_status=ChunkLeaseStatus.ACTIVE,
                policy=policy,
            )

        release_assignment_cache_policy_from_store(
            manifest,
            assignment,
            protected_chunk_ids={protected_chunk_id},
            policy=policy,
        )

        protected_record = get_cached_chunk_record(protected_chunk_id, policy)
        self.assertIsNotNone(protected_record)
        assert protected_record is not None
        self.assertEqual(protected_record.cache_class, ChunkCacheClass.HOT)
        self.assertEqual(protected_record.lease_status, ChunkLeaseStatus.ACTIVE)

    def test_assignment_plan_from_store_filters_by_manifest_version(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-chunk-store-version")
        manifest = _build_manifest()
        tokenizer_chunk = next(
            chunk for chunk in manifest.chunks if chunk.kind == ModelChunkKind.TOKENIZER
        )
        put_cached_chunk(
            catalog_id=manifest.catalog_id,
            version="old-version",
            chunk_id=tokenizer_chunk.chunk_id,
            sha256_hex=hashlib.sha256(b"t" * tokenizer_chunk.size_bytes).hexdigest(),
            content=b"t" * tokenizer_chunk.size_bytes,
            policy=policy,
        )
        assignment = ModelShardAssignment(start_layer=0, end_layer=10)
        plan = build_assignment_chunk_plan_from_store(
            manifest,
            assignment,
            policy=policy,
        )
        self.assertFalse(plan.ready)
        self.assertEqual(plan.coverage.present_bytes, 0)

    def test_assignment_plan_from_store_can_verify_cached_chunk_files(self) -> None:
        policy = WalletPolicy(
            wallet_data_dirname=".tmp-model-distribution-chunk-store-verify"
        )
        manifest = _build_manifest()
        assignment = ModelShardAssignment(start_layer=0, end_layer=10)
        required_chunks = manifest.required_chunks_for_layers(0, 10)
        for chunk in required_chunks:
            put_cached_chunk(
                catalog_id=manifest.catalog_id,
                version=manifest.version,
                chunk_id=chunk.chunk_id,
                sha256_hex=hashlib.sha256(b"x" * chunk.size_bytes).hexdigest(),
                content=b"x" * chunk.size_bytes,
                policy=policy,
            )
        cached_chunk_path(required_chunks[0].chunk_id, policy=policy).write_bytes(
            b"y" * required_chunks[0].size_bytes
        )

        optimistic_plan = build_assignment_chunk_plan_from_store(
            manifest,
            assignment,
            policy=policy,
        )
        verified_plan = build_assignment_chunk_plan_from_store(
            manifest,
            assignment,
            verify_cached_chunks=True,
            policy=policy,
        )

        self.assertTrue(optimistic_plan.ready)
        self.assertFalse(verified_plan.ready)
        self.assertIn(
            required_chunks[0].chunk_id,
            verified_plan.coverage.missing_chunk_ids,
        )

    def test_build_weight_chunks_for_artifact_uses_real_hashes_and_layer_coverage(self) -> None:
        artifact_path = self._write_temp_file("tiny.gguf", b"abcdefghi")
        artifact = build_source_artifact_from_file(
            artifact_path,
            artifact_id="gguf-main",
            media_type="application/gguf",
        )
        chunks = build_weight_chunks_for_artifact(
            artifact,
            artifact_path,
            total_layers=9,
            policy=ChunkSizePolicy.ADAPTIVE,
            min_chunk_bytes=3,
            max_chunk_bytes=3,
        )
        self.assertEqual([chunk.size_bytes for chunk in chunks], [3, 3, 3])
        self.assertEqual(
            [(chunk.layer_start, chunk.layer_end) for chunk in chunks],
            [(0, 3), (3, 6), (6, 9)],
        )
        self.assertEqual(chunks[0].sha256_hex, hashlib.sha256(b"abc").hexdigest())
        self.assertEqual(chunks[2].sha256_hex, hashlib.sha256(b"ghi").hexdigest())

    def test_build_weight_chunks_for_real_gguf_preserves_tensor_boundaries(self) -> None:
        artifact_path = _write_minimal_gguf_file(
            Path(self.tempdir.name) / "tensor-aware.gguf"
        )
        artifact = build_source_artifact_from_file(
            artifact_path,
            artifact_id="gguf-main",
            media_type="application/gguf",
        )

        chunks = build_weight_chunks_for_artifact(
            artifact,
            artifact_path,
            total_layers=2,
            policy=ChunkSizePolicy.ADAPTIVE,
            min_chunk_bytes=80,
            max_chunk_bytes=80,
        )

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].offset_bytes, 0)
        self.assertTrue(chunks[0].required_by_default)
        self.assertIsNone(chunks[0].layer_start)
        self.assertEqual(chunks[0].tensor_names, ["token_embd.weight"])
        self.assertEqual(chunks[1].layer_start, 0)
        self.assertEqual(chunks[1].layer_end, 2)
        self.assertEqual(
            chunks[1].tensor_names,
            ["blk.0.attn_q.weight", "blk.1.attn_q.weight"],
        )
        self.assertTrue(chunks[2].required_by_default)
        self.assertIsNone(chunks[2].layer_start)
        self.assertIsNone(chunks[2].layer_end)
        self.assertEqual(chunks[2].tensor_names, ["output.weight"])
        self.assertGreater(chunks[1].offset_bytes, chunks[0].offset_bytes)

    def test_read_gguf_model_metadata_extracts_architecture_and_layers(self) -> None:
        gguf_path = _write_minimal_gguf_file(
            Path(self.tempdir.name) / "metadata.gguf",
            architecture="mistral",
            block_count=2,
            embedding_length=4096,
        )

        metadata = read_gguf_model_metadata(gguf_path)

        self.assertEqual(metadata.architecture, "mistral")
        self.assertEqual(metadata.total_layers, 2)
        self.assertEqual(metadata.hidden_size, 4096)
        self.assertEqual(metadata.metadata["general.architecture"], "mistral")
        self.assertEqual(metadata.metadata["mistral.block_count"], 2)
        self.assertEqual(metadata.metadata["mistral.embedding_length"], 4096)

    def test_build_gguf_manifest_can_discover_architecture_and_layers(self) -> None:
        gguf_path = _write_minimal_gguf_file(
            Path(self.tempdir.name) / "autodetect.gguf",
            architecture="mistral",
            block_count=2,
            embedding_length=4096,
        )
        manifest = build_gguf_model_package_manifest(
            catalog_id="mistral-autodetect",
            model_id="Example/Mistral-7B-GGUF",
            version="2026.05.11",
            gguf_path=gguf_path,
            chunk_size_policy=ChunkSizePolicy.ADAPTIVE,
            min_chunk_bytes=4,
            max_chunk_bytes=128,
            quantization="Q4_K_M",
        )

        self.assertEqual(manifest.family, "mistral")
        self.assertEqual(manifest.metadata["total_layers"], 2)
        self.assertEqual(manifest.metadata["hidden_size"], 4096)
        self.assertEqual(manifest.metadata["gguf_architecture"], "mistral")
        self.assertEqual(manifest.metadata["shard_compatibility"], "full_model_local")
        self.assertFalse(manifest.metadata["layer_range_supported"])

    def test_build_gguf_manifest_prefers_header_architecture_over_generic_family(
        self,
    ) -> None:
        gguf_path = _write_minimal_gguf_file(
            Path(self.tempdir.name) / "qwen2-header.gguf",
            architecture="qwen2",
            block_count=24,
            embedding_length=896,
        )
        manifest = build_gguf_model_package_manifest(
            catalog_id="local-qwen2",
            model_id="local/custom-qwen-family-gguf",
            version="2026.05.11",
            gguf_path=gguf_path,
            total_layers=5,
            chunk_size_policy=ChunkSizePolicy.ADAPTIVE,
            min_chunk_bytes=4,
            max_chunk_bytes=128,
            family="qwen",
            quantization="Q4_K_M",
        )

        self.assertEqual(manifest.family, "qwen2")
        self.assertEqual(manifest.metadata["total_layers"], 24)
        self.assertEqual(manifest.metadata["hidden_size"], 896)
        self.assertEqual(manifest.metadata["gguf_architecture"], "qwen2")
        self.assertEqual(
            manifest.metadata["shard_compatibility"],
            "layer_range_supported",
        )
        self.assertTrue(manifest.metadata["layer_range_supported"])

    def test_build_gguf_manifest_creates_adaptive_chunked_package(self) -> None:
        gguf_path = self._write_temp_file("model.gguf", b"a" * 10)
        manifest = build_gguf_model_package_manifest(
            catalog_id="qwen3-demo",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            version="2026.04.25",
            gguf_path=gguf_path,
            total_layers=5,
            chunk_size_policy=ChunkSizePolicy.ADAPTIVE,
            min_chunk_bytes=4,
            max_chunk_bytes=4,
            family="qwen",
            quantization="Q4_K_M",
        )
        self.assertEqual(manifest.total_size_bytes, 10)
        self.assertEqual(manifest.preferred_filename, "model.gguf")
        self.assertEqual(manifest.family, "qwen")
        self.assertEqual(manifest.quantization, "Q4_K_M")
        self.assertEqual([chunk.size_bytes for chunk in manifest.chunks], [4, 3, 3])
        self.assertEqual(manifest.metadata["chunk_count"], 3)
        self.assertEqual(manifest.metadata["total_layers"], 5)
        self.assertEqual(manifest.metadata["model_format"], "gguf")
        self.assertEqual(manifest.metadata["gguf_architecture"], "qwen3")
        self.assertEqual(
            manifest.metadata["shard_compatibility"],
            "layer_range_supported",
        )
        self.assertTrue(manifest.metadata["layer_range_supported"])
        self.assertEqual(
            manifest.metadata["layer_range_probe_abi"],
            "cai-layer-range-v1",
        )
        self.assertIn(
            "qwen3-layer-range-equivalence-probe",
            manifest.metadata["layer_range_equivalence_probe_report"],
        )
        self.assertEqual(
            manifest.metadata["activation_state_format"],
            "ggml-tensor-v1/layer-range-activation-v1",
        )
        self.assertEqual(manifest.files[0].sha256_hex, hashlib.sha256(b"a" * 10).hexdigest())

    def test_build_public_gguf_manifest_marks_unproven_architecture_full_model_local(
        self,
    ) -> None:
        gguf_path = self._write_temp_file("mistral.gguf", b"b" * 10)
        manifest = build_gguf_model_package_manifest(
            catalog_id="mistral-demo",
            model_id="Example/Mistral-7B-GGUF",
            version="2026.04.25",
            gguf_path=gguf_path,
            total_layers=4,
            chunk_size_policy=ChunkSizePolicy.ADAPTIVE,
            min_chunk_bytes=4,
            max_chunk_bytes=4,
            family="mistral",
            quantization="Q4_K_M",
        )

        self.assertEqual(manifest.metadata["gguf_architecture"], "mistral")
        self.assertEqual(
            manifest.metadata["shard_compatibility"],
            "full_model_local",
        )
        self.assertFalse(manifest.metadata["layer_range_supported"])
        self.assertIn(
            "single-node full-model local inference",
            manifest.metadata["shard_compatibility_reason"],
        )

    def test_build_private_gguf_manifest_keeps_unproven_architecture_unsupported(
        self,
    ) -> None:
        gguf_path = self._write_temp_file("private-mistral.gguf", b"c" * 10)
        manifest = build_gguf_model_package_manifest(
            catalog_id="private-mistral-demo",
            model_id="Example/Mistral-7B-GGUF",
            version="2026.04.25",
            gguf_path=gguf_path,
            total_layers=4,
            package_kind=ModelPackageKind.PRIVATE_CURATED,
            chunk_size_policy=ChunkSizePolicy.ADAPTIVE,
            min_chunk_bytes=4,
            max_chunk_bytes=4,
            family="mistral",
            quantization="Q4_K_M",
        )

        self.assertEqual(manifest.metadata["gguf_architecture"], "mistral")
        self.assertEqual(
            manifest.metadata["shard_compatibility"],
            "unsupported_for_sharding",
        )
        self.assertFalse(manifest.metadata["layer_range_supported"])
        self.assertIn(
            "no checked CAI layer-range equivalence probe",
            manifest.metadata["shard_compatibility_reason"],
        )

    def test_assignment_fetch_plan_prefers_peer_then_origin(self) -> None:
        manifest = _build_manifest()
        assignment = ModelShardAssignment(start_layer=10, end_layer=28)
        tokenizer_chunk = next(
            chunk for chunk in manifest.chunks if chunk.kind == ModelChunkKind.TOKENIZER
        )
        missing_chunk_ids = {
            chunk.chunk_id
            for chunk in manifest.chunks
            if chunk.kind == ModelChunkKind.WEIGHTS and chunk.layer_start in (10, 18)
        }
        plan = manifest.build_assignment_fetch_plan(
            assignment,
            present_chunk_ids={tokenizer_chunk.chunk_id},
            peer_chunk_inventory={"peer-a": missing_chunk_ids},
        )
        self.assertIsInstance(plan, AssignmentFetchPlan)
        self.assertFalse(plan.ready)
        self.assertEqual(plan.estimated_fetch_bytes, 600)
        self.assertEqual(len(plan.fetch_requests), 2)
        for request in plan.fetch_requests:
            self.assertEqual(request.sources[0].kind, ChunkFetchSourceKind.PEER_CACHE)
            self.assertEqual(request.sources[0].source_id, "peer-a")
            self.assertEqual(request.sources[-1].kind, ChunkFetchSourceKind.ORIGIN)

    def test_assignment_fetch_plan_from_store_ignores_old_version_and_uses_seed(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-fetch-plan")
        manifest = _build_manifest()
        tokenizer_chunk = next(
            chunk for chunk in manifest.chunks if chunk.kind == ModelChunkKind.TOKENIZER
        )
        put_cached_chunk(
            catalog_id=manifest.catalog_id,
            version="old-version",
            chunk_id=tokenizer_chunk.chunk_id,
            sha256_hex=hashlib.sha256(b"z" * tokenizer_chunk.size_bytes).hexdigest(),
            content=b"z" * tokenizer_chunk.size_bytes,
            policy=policy,
        )
        assignment = ModelShardAssignment(start_layer=0, end_layer=10)
        weight_chunk = next(
            chunk for chunk in manifest.chunks if chunk.kind == ModelChunkKind.WEIGHTS and chunk.layer_start == 0
        )
        plan = build_assignment_fetch_plan_from_store(
            manifest,
            assignment,
            seed_chunk_inventory={"seed-1": [tokenizer_chunk.chunk_id, weight_chunk.chunk_id]},
            policy=policy,
        )
        self.assertFalse(plan.ready)
        self.assertEqual(plan.coverage.present_bytes, 0)
        self.assertEqual(len(plan.fetch_requests), 2)
        self.assertTrue(
            all(request.sources[0].kind == ChunkFetchSourceKind.STORAGE_SEED for request in plan.fetch_requests)
        )

    def test_local_chunk_inventory_payload_groups_cached_chunks_by_manifest(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-inventory-local")
        put_cached_chunk(
            catalog_id="demo-a",
            version="v1",
            chunk_id="chunk-a1",
            sha256_hex=hashlib.sha256(b"a1").hexdigest(),
            content=b"a1",
            policy=policy,
        )
        put_cached_chunk(
            catalog_id="demo-a",
            version="v1",
            chunk_id="chunk-a2",
            sha256_hex=hashlib.sha256(b"a2").hexdigest(),
            content=b"a2",
            policy=policy,
        )
        put_cached_chunk(
            catalog_id="demo-b",
            version="v2",
            chunk_id="chunk-b1",
            sha256_hex=hashlib.sha256(b"bbb").hexdigest(),
            content=b"bbb",
            policy=policy,
        )
        payload = build_local_chunk_inventory_payload(
            "node-local",
            source_kind=ChunkInventorySourceKind.LOCAL_CACHE,
            policy=policy,
        )
        self.assertEqual(payload.source_id, "node-local")
        self.assertEqual(len(payload.records), 2)
        record_a = next(record for record in payload.records if record.catalog_id == "demo-a")
        self.assertEqual(record_a.version, "v1")
        self.assertEqual(record_a.chunk_count, 2)
        self.assertEqual(record_a.total_bytes, 4)
        self.assertEqual(set(record_a.chunk_ids), {"chunk-a1", "chunk-a2"})

    def test_chunk_inventory_payload_exports_and_imports_public_manifests(self) -> None:
        export_policy = WalletPolicy(
            wallet_data_dirname=".tmp-model-distribution-inventory-manifest-export"
        )
        import_policy = WalletPolicy(
            wallet_data_dirname=".tmp-model-distribution-inventory-manifest-import"
        )
        manifest = _build_manifest()
        save_model_package_manifest(manifest, export_policy)

        payload = build_local_chunk_inventory_payload(
            "node-local",
            source_kind=ChunkInventorySourceKind.LOCAL_CACHE,
            policy=export_policy,
        )

        self.assertEqual(len(payload.manifests), 1)
        self.assertEqual(payload.manifests[0].catalog_id, manifest.catalog_id)

        round_tripped = ChunkInventoryPayload.from_dict(payload.to_dict())
        import_chunk_inventory_payload(round_tripped, import_policy)
        imported_manifest = load_model_package_manifest(
            manifest.catalog_id,
            manifest.version,
            import_policy,
        )

        self.assertEqual(imported_manifest.model_id, manifest.model_id)
        self.assertEqual(len(imported_manifest.chunks), len(manifest.chunks))

    def test_export_and_sync_chunk_inventory_from_cai_peers(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-inventory-sync")
        stale_payload = ChunkInventoryPayload(
            source_id="node-stale",
            source_kind=ChunkInventorySourceKind.PEER_CACHE,
            published_at="2026-04-25T00:00:00+00:00",
            endpoint_base_url="http://198.51.100.11:52415",
            records=(),
        )
        import_chunk_inventory_payload(stale_payload, policy)
        put_cached_chunk(
            catalog_id="demo-a",
            version="v1",
            chunk_id="chunk-a1",
            sha256_hex=hashlib.sha256(b"a1").hexdigest(),
            content=b"a1",
            policy=policy,
        )
        payload = export_chunk_inventory_payload(
            "node-peer",
            source_kind=ChunkInventorySourceKind.PEER_CACHE,
            endpoint_base_url="http://198.51.100.10:52415",
            policy=policy,
        )

        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return json.dumps(payload.to_dict()).encode("utf-8")

        with patch(
            "cai_compute_chain.validators.discover_peer_cai_urls",
            return_value=[
                "http://198.51.100.10:52415/v1/cai/chunk-inventory?source_kind=peer_cache"
            ],
        ), patch("cai_compute_chain.model_distribution.urlopen", return_value=_FakeResponse()):
            result = sync_chunk_inventory_from_cai_peers(
                state_payload={"nodeIdentities": {"node-peer": {}}},
                cai_url="http://127.0.0.1:52415",
                source_kind=ChunkInventorySourceKind.PEER_CACHE,
                policy=policy,
            )

        self.assertEqual(result.attempted_peers, 1)
        self.assertEqual(result.successful_peers, 1)
        self.assertEqual(result.imported_payloads, 1)
        self.assertEqual(result.pruned_payloads, 1)
        imported = list_imported_chunk_inventory_payloads(
            source_kind=ChunkInventorySourceKind.PEER_CACHE,
            policy=policy,
        )
        self.assertEqual(len(imported), 1)
        self.assertEqual(imported[0].source_id, "node-peer")
        self.assertEqual(imported[0].endpoint_base_url, "http://198.51.100.10:52415")

    def test_sync_chunk_inventory_from_urls_records_peer_errors(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-inventory-errors")
        payload = ChunkInventoryPayload(
            source_id="seed-new",
            source_kind=ChunkInventorySourceKind.STORAGE_SEED,
            published_at="2026-04-25T00:00:00+00:00",
            endpoint_base_url="http://203.0.113.11:52415",
            records=(),
        )

        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return json.dumps(payload.to_dict()).encode("utf-8")

        def fake_urlopen(url: str, timeout: int = 0):
            if "203.0.113.10" in url:
                raise OSError("chunk inventory peer offline")
            return _FakeResponse()

        with patch(
            "cai_compute_chain.model_distribution.urlopen",
            side_effect=fake_urlopen,
        ):
            result = sync_chunk_inventory_from_urls(
                inventory_urls=[
                    "http://203.0.113.10:52415",
                    "http://203.0.113.11:52415",
                ],
                source_kind=ChunkInventorySourceKind.STORAGE_SEED,
                policy=policy,
            )

        self.assertEqual(result.attempted_peers, 2)
        self.assertEqual(result.successful_peers, 1)
        self.assertEqual(result.failed_peers, 1)
        self.assertEqual(
            result.failed_peer_urls,
            [
                "http://203.0.113.10:52415/v1/cai/chunk-inventory"
                "?source_kind=storage_seed"
            ],
        )
        self.assertEqual(result.peer_errors[0]["errorType"], "OSError")
        self.assertIn("chunk inventory peer offline", result.peer_errors[0]["message"])
        self.assertEqual(result.imported_payloads, 1)

    def test_prune_imported_chunk_inventory_payloads_removes_missing_peers(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-inventory-prune")
        keep_payload = ChunkInventoryPayload(
            source_id="peer-keep",
            source_kind=ChunkInventorySourceKind.PEER_CACHE,
            published_at="2026-04-25T00:00:00+00:00",
            endpoint_base_url="http://198.51.100.10:52415",
            records=(),
        )
        drop_payload = ChunkInventoryPayload(
            source_id="peer-drop",
            source_kind=ChunkInventorySourceKind.PEER_CACHE,
            published_at="2026-04-25T00:00:00+00:00",
            endpoint_base_url="http://198.51.100.11:52415",
            records=(),
        )
        import_chunk_inventory_payload(keep_payload, policy)
        import_chunk_inventory_payload(drop_payload, policy)

        removed = prune_imported_chunk_inventory_payloads(
            allowed_source_ids={"peer-keep"},
            source_kind=ChunkInventorySourceKind.PEER_CACHE,
            policy=policy,
        )

        self.assertEqual(removed, ("peer-drop",))
        imported = list_imported_chunk_inventory_payloads(
            source_kind=ChunkInventorySourceKind.PEER_CACHE,
            policy=policy,
        )
        self.assertEqual(len(imported), 1)
        self.assertEqual(imported[0].source_id, "peer-keep")

    def test_prune_chunk_source_health_records_removes_missing_peers(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-health-prune")
        record_chunk_source_failure(
            ChunkFetchSource(
                kind=ChunkFetchSourceKind.PEER_CACHE,
                source_id="peer-keep",
                locator="http://198.51.100.10:52415",
            ),
            error="timeout",
            policy=policy,
        )
        record_chunk_source_failure(
            ChunkFetchSource(
                kind=ChunkFetchSourceKind.PEER_CACHE,
                source_id="peer-drop",
                locator="http://198.51.100.11:52415",
            ),
            error="timeout",
            policy=policy,
        )

        removed = prune_chunk_source_health_records(
            allowed_source_ids={"peer-keep"},
            source_kind=ChunkInventorySourceKind.PEER_CACHE,
            policy=policy,
        )

        self.assertEqual(len(removed), 1)
        self.assertIn("peer-drop", removed[0])
        remaining = list_chunk_source_health_records(policy)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].source_id, "peer-keep")

    def test_sync_chunk_inventory_from_urls_prunes_stale_storage_seed_by_endpoint(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-seed-sync")
        stale_payload = ChunkInventoryPayload(
            source_id="seed-old",
            source_kind=ChunkInventorySourceKind.STORAGE_SEED,
            published_at="2026-04-25T00:00:00+00:00",
            endpoint_base_url="http://203.0.113.10:52415",
            records=(),
        )
        import_chunk_inventory_payload(stale_payload, policy)
        fresh_payload = ChunkInventoryPayload(
            source_id="seed-new",
            source_kind=ChunkInventorySourceKind.STORAGE_SEED,
            published_at="2026-04-25T00:00:00+00:00",
            endpoint_base_url="http://203.0.113.11:52415",
            records=(),
        )

        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return json.dumps(fresh_payload.to_dict()).encode("utf-8")

        with patch(
            "cai_compute_chain.model_distribution.urlopen",
            return_value=_FakeResponse(),
        ):
            result = sync_chunk_inventory_from_urls(
                inventory_urls=["http://203.0.113.11:52415"],
                source_kind=ChunkInventorySourceKind.STORAGE_SEED,
                prune_missing_endpoint_base_urls={"http://203.0.113.11:52415"},
                policy=policy,
            )

        self.assertEqual(result.attempted_peers, 1)
        self.assertEqual(result.successful_peers, 1)
        self.assertEqual(result.imported_payloads, 1)
        self.assertEqual(result.pruned_payloads, 1)
        imported = list_imported_chunk_inventory_payloads(
            source_kind=ChunkInventorySourceKind.STORAGE_SEED,
            policy=policy,
        )
        self.assertEqual(len(imported), 1)
        self.assertEqual(imported[0].source_id, "seed-new")
        self.assertEqual(imported[0].endpoint_base_url, "http://203.0.113.11:52415")

    def test_build_assignment_fetch_plan_from_store_ignores_stale_imported_peer_inventory(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-stale-peer-inventory")
        manifest = _build_manifest()
        save_model_package_manifest(manifest, policy)
        required_chunks = manifest.required_chunks_for_layers(0, 4)
        required_chunk_ids = tuple(chunk.chunk_id for chunk in required_chunks)
        required_total_bytes = sum(chunk.size_bytes for chunk in required_chunks)

        stale_payload = ChunkInventoryPayload(
            source_id="peer-stale",
            source_kind=ChunkInventorySourceKind.PEER_CACHE,
            published_at=(datetime.now(tz=UTC) - timedelta(hours=2)).replace(microsecond=0).isoformat(),
            endpoint_base_url="http://198.51.100.10:52415",
            records=(
                ChunkInventoryRecord(
                    catalog_id=manifest.catalog_id,
                    version=manifest.version,
                    chunk_ids=required_chunk_ids,
                    chunk_count=len(required_chunk_ids),
                    total_bytes=required_total_bytes,
                ),
            ),
        )
        fresh_payload = ChunkInventoryPayload(
            source_id="peer-fresh",
            source_kind=ChunkInventorySourceKind.PEER_CACHE,
            published_at=_fresh_published_at(),
            endpoint_base_url="http://198.51.100.11:52415",
            records=(
                ChunkInventoryRecord(
                    catalog_id=manifest.catalog_id,
                    version=manifest.version,
                    chunk_ids=required_chunk_ids,
                    chunk_count=len(required_chunk_ids),
                    total_bytes=required_total_bytes,
                ),
            ),
        )
        import_chunk_inventory_payload(stale_payload, policy)
        import_chunk_inventory_payload(fresh_payload, policy)

        fetch_plan = build_assignment_fetch_plan_from_store(
            manifest,
            ModelShardAssignment(start_layer=0, end_layer=4, node_id="node-a"),
            use_imported_peer_inventory=True,
            fetch_policy=ChunkFetchPolicy(max_inventory_age_seconds=300),
            policy=policy,
        )

        self.assertTrue(fetch_plan.fetch_requests)
        first_request_sources = fetch_plan.fetch_requests[0].sources
        self.assertEqual(first_request_sources[0].kind, ChunkFetchSourceKind.PEER_CACHE)
        self.assertEqual(first_request_sources[0].source_id, "peer-fresh")
        self.assertTrue(all(source.source_id != "peer-stale" for source in first_request_sources))

    def test_build_chunk_inventory_index_can_disable_freshness_filter(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-no-freshness-filter")
        manifest = _build_manifest()
        save_model_package_manifest(manifest, policy)
        required_chunks = manifest.required_chunks_for_layers(0, 4)
        required_chunk_ids = tuple(chunk.chunk_id for chunk in required_chunks)
        required_total_bytes = sum(chunk.size_bytes for chunk in required_chunks)
        stale_payload = ChunkInventoryPayload(
            source_id="peer-stale",
            source_kind=ChunkInventorySourceKind.PEER_CACHE,
            published_at=(datetime.now(tz=UTC) - timedelta(days=1)).replace(microsecond=0).isoformat(),
            endpoint_base_url="http://198.51.100.10:52415",
            records=(
                ChunkInventoryRecord(
                    catalog_id=manifest.catalog_id,
                    version=manifest.version,
                    chunk_ids=required_chunk_ids,
                    chunk_count=len(required_chunk_ids),
                    total_bytes=required_total_bytes,
                ),
            ),
        )
        import_chunk_inventory_payload(stale_payload, policy)

        index = build_chunk_inventory_index(
            manifest.catalog_id,
            manifest.version,
            source_kind=ChunkInventorySourceKind.PEER_CACHE,
            fetch_policy=ChunkFetchPolicy(max_inventory_age_seconds=0),
            policy=policy,
        )

        self.assertIn("peer-stale", index)

    def test_sync_chunk_inventory_from_urls_prunes_stale_storage_seed_health_by_endpoint(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-seed-health-sync")
        record_chunk_source_failure(
            ChunkFetchSource(
                kind=ChunkFetchSourceKind.STORAGE_SEED,
                source_id="seed-old",
                locator="http://203.0.113.10:52415",
            ),
            error="seed unavailable",
            policy=policy,
        )
        stale_payload = ChunkInventoryPayload(
            source_id="seed-old",
            source_kind=ChunkInventorySourceKind.STORAGE_SEED,
            published_at="2026-04-25T00:00:00+00:00",
            endpoint_base_url="http://203.0.113.10:52415",
            records=(),
        )
        import_chunk_inventory_payload(stale_payload, policy)
        fresh_payload = ChunkInventoryPayload(
            source_id="seed-new",
            source_kind=ChunkInventorySourceKind.STORAGE_SEED,
            published_at="2026-04-25T00:00:00+00:00",
            endpoint_base_url="http://203.0.113.11:52415",
            records=(),
        )

        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return json.dumps(fresh_payload.to_dict()).encode("utf-8")

        with patch(
            "cai_compute_chain.model_distribution.urlopen",
            return_value=_FakeResponse(),
        ):
            sync_chunk_inventory_from_urls(
                inventory_urls=["http://203.0.113.11:52415"],
                source_kind=ChunkInventorySourceKind.STORAGE_SEED,
                prune_missing_endpoint_base_urls={"http://203.0.113.11:52415"},
                policy=policy,
            )

        remaining = list_chunk_source_health_records(policy)
        self.assertEqual(len(remaining), 0)

    def test_imported_chunk_inventory_index_filters_by_kind_and_manifest(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-inventory-import")
        peer_payload = ChunkInventoryPayload(
            source_id="peer-a",
            source_kind=ChunkInventorySourceKind.PEER_CACHE,
            published_at=_fresh_published_at(),
            endpoint_base_url="http://198.51.100.10:52415",
            records=(
                ChunkInventoryRecord(
                    catalog_id="demo-qwen",
                    version="v1",
                    chunk_ids=("chunk-1", "chunk-2"),
                    chunk_count=2,
                    total_bytes=20,
                ),
            ),
        )
        seed_payload = ChunkInventoryPayload(
            source_id="seed-a",
            source_kind=ChunkInventorySourceKind.STORAGE_SEED,
            published_at=_fresh_published_at(),
            records=(
                ChunkInventoryRecord(
                    catalog_id="demo-qwen",
                    version="v1",
                    chunk_ids=("chunk-3",),
                    chunk_count=1,
                    total_bytes=10,
                ),
            ),
        )
        import_chunk_inventory_payload(peer_payload, policy)
        import_chunk_inventory_payload(seed_payload, policy)

        peer_index = build_chunk_inventory_index(
            "demo-qwen",
            "v1",
            source_kind=ChunkInventorySourceKind.PEER_CACHE,
            policy=policy,
        )
        seed_index = build_chunk_inventory_index(
            "demo-qwen",
            "v1",
            source_kind=ChunkInventorySourceKind.STORAGE_SEED,
            policy=policy,
        )
        peer_locators = build_chunk_inventory_locator_index(
            "demo-qwen",
            "v1",
            source_kind=ChunkInventorySourceKind.PEER_CACHE,
            policy=policy,
        )
        self.assertEqual(peer_index, {"peer-a": {"chunk-1", "chunk-2"}})
        self.assertEqual(seed_index, {"seed-a": {"chunk-3"}})
        self.assertEqual(peer_locators, {"peer-a": "http://198.51.100.10:52415"})
        self.assertEqual(len(list_imported_chunk_inventory_payloads(policy=policy)), 2)

    def test_fetch_plan_from_store_can_use_imported_inventories(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-inventory-fetch")
        manifest = _build_manifest()
        weight_chunk = next(
            chunk for chunk in manifest.chunks if chunk.kind == ModelChunkKind.WEIGHTS and chunk.layer_start == 0
        )
        inventory_path = self._write_temp_file(
            "peer-inventory.json",
            json.dumps(
                {
                    "source_id": "peer-a",
                    "source_kind": ChunkInventorySourceKind.PEER_CACHE,
                    "published_at": _fresh_published_at(),
                    "endpoint_base_url": "http://198.51.100.10:52415",
                    "records": [
                        {
                            "catalog_id": manifest.catalog_id,
                            "version": manifest.version,
                            "chunk_ids": [weight_chunk.chunk_id],
                            "chunk_count": 1,
                            "total_bytes": weight_chunk.size_bytes,
                        }
                    ],
                }
            ).encode("utf-8"),
        )
        import_chunk_inventory_payload(load_chunk_inventory_payload(inventory_path), policy)
        assignment = ModelShardAssignment(start_layer=0, end_layer=10)
        plan = build_assignment_fetch_plan_from_store(
            manifest,
            assignment,
            use_imported_peer_inventory=True,
            policy=policy,
        )
        self.assertEqual(len(plan.fetch_requests), 2)
        peer_backed = next(request for request in plan.fetch_requests if request.chunk_id == weight_chunk.chunk_id)
        self.assertEqual(peer_backed.sources[0].kind, ChunkFetchSourceKind.PEER_CACHE)
        self.assertEqual(peer_backed.sources[0].source_id, "peer-a")
        self.assertEqual(peer_backed.sources[0].locator, "http://198.51.100.10:52415")

    def test_queue_assignment_fetch_plan_creates_deduplicated_download_tasks(self) -> None:
        manifest = _build_manifest()
        assignment = ModelShardAssignment(start_layer=10, end_layer=28, node_id="node-a")
        tokenizer_chunk = next(
            chunk for chunk in manifest.chunks if chunk.kind == ModelChunkKind.TOKENIZER
        )
        fetch_plan = manifest.build_assignment_fetch_plan(
            assignment,
            present_chunk_ids={tokenizer_chunk.chunk_id},
            peer_chunk_inventory={
                "peer-a": {
                    chunk.chunk_id
                    for chunk in manifest.chunks
                    if chunk.kind == ModelChunkKind.WEIGHTS and chunk.layer_start in (10, 18)
                }
            },
        )
        queued_first = queue_assignment_fetch_plan(manifest, fetch_plan)
        queued_second = queue_assignment_fetch_plan(manifest, fetch_plan)
        self.assertEqual(len(queued_first), 2)
        self.assertEqual(len(queued_second), 2)
        tasks = list_chunk_download_tasks()
        self.assertEqual(len(tasks), 2)
        self.assertTrue(all(task.status == ChunkDownloadTaskStatus.QUEUED for task in tasks))
        self.assertTrue(all(task.selected_source_kind == ChunkFetchSourceKind.PEER_CACHE for task in tasks))

    def test_chunk_download_queue_snapshot_and_status_updates(self) -> None:
        manifest = _build_manifest()
        assignment = ModelShardAssignment(start_layer=0, end_layer=10, node_id="node-a")
        fetch_plan = manifest.build_assignment_fetch_plan(
            assignment,
            present_chunk_ids=set(),
            seed_chunk_inventory={
                "seed-a": {
                    chunk.chunk_id
                    for chunk in manifest.required_chunks_for_layers(0, 10)
                }
            },
        )
        queued = queue_assignment_fetch_plan(manifest, fetch_plan)
        self.assertEqual(len(queued), 2)
        first_task = queued[0]
        updated = update_chunk_download_task_status(
            first_task.task_id,
            ChunkDownloadTaskStatus.COMPLETED,
            selected_source_kind=ChunkFetchSourceKind.STORAGE_SEED,
            selected_source_id="seed-a",
        )
        assert updated is not None
        self.assertEqual(updated.attempt_count, 1)
        snapshot = chunk_download_queue_snapshot()
        self.assertEqual(snapshot.stats.task_count, 2)
        self.assertEqual(snapshot.stats.completed_count, 1)
        self.assertEqual(snapshot.stats.queued_count, 1)
        self.assertEqual(snapshot.stats.completed_bytes, first_task.size_bytes)

    def test_save_and_load_local_artifact_binding(self) -> None:
        artifact_path = self._write_temp_file("artifact.gguf", b"weights")
        save_local_artifact_binding(
            "demo-qwen",
            "v1",
            artifact_id="gguf-main",
            local_path=artifact_path,
        )
        bindings = load_local_artifact_bindings("demo-qwen", "v1")
        self.assertEqual(len(bindings.bindings), 1)
        self.assertEqual(bindings.bindings[0].artifact_id, "gguf-main")
        self.assertEqual(Path(bindings.bindings[0].local_path), artifact_path.resolve())

    def test_save_and_load_chunk_source_binding(self) -> None:
        remote_root = Path(self.tempdir.name) / "peer-data-root"
        remote_root.mkdir(parents=True, exist_ok=True)
        save_chunk_source_binding(
            ChunkFetchSourceKind.PEER_CACHE,
            "peer-a",
            data_root_path=remote_root,
        )
        bindings = load_chunk_source_bindings()
        self.assertEqual(len(bindings.bindings), 1)
        binding = get_chunk_source_binding(ChunkFetchSourceKind.PEER_CACHE, "peer-a")
        self.assertIsNotNone(binding)
        assert binding is not None
        self.assertEqual(Path(binding.data_root_path), remote_root.resolve())
        self.assertEqual(chunk_source_bindings_path().name, "chunk-source-bindings.json")

    def test_execute_chunk_download_queue_materializes_chunks_from_bound_origin(self) -> None:
        gguf_path = self._write_temp_file("materialize.gguf", b"abcdefghij")
        manifest = build_gguf_model_package_manifest(
            catalog_id="demo-materialize",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            version="v1",
            gguf_path=gguf_path,
            total_layers=5,
            chunk_size_policy=ChunkSizePolicy.ADAPTIVE,
            min_chunk_bytes=4,
            max_chunk_bytes=4,
            source_repo_id="Qwen/Qwen3-0.6B-GGUF",
        )
        save_model_package_manifest(manifest)
        save_local_artifact_binding(
            manifest.catalog_id,
            manifest.version,
            artifact_id="gguf-main",
            local_path=gguf_path,
        )
        fetch_plan = manifest.build_assignment_fetch_plan(
            ModelShardAssignment(start_layer=0, end_layer=5, node_id="node-a"),
            present_chunk_ids=set(),
        )
        queued = queue_assignment_fetch_plan(manifest, fetch_plan)
        self.assertEqual(len(queued), len(manifest.chunks))
        processed = execute_chunk_download_queue()
        self.assertEqual(len(processed), len(manifest.chunks))
        self.assertTrue(all(task.status == ChunkDownloadTaskStatus.COMPLETED for task in processed))
        snapshot = chunk_store_snapshot()
        self.assertEqual(snapshot.stats.chunk_count, len(manifest.chunks))
        first_chunk_record = get_cached_chunk_record(manifest.chunks[0].chunk_id)
        self.assertIsNotNone(first_chunk_record)
        assert first_chunk_record is not None
        self.assertEqual(first_chunk_record.cache_class, ChunkCacheClass.HOT)
        self.assertEqual(first_chunk_record.lease_status, ChunkLeaseStatus.ACTIVE)

    def test_materialize_artifact_from_store_rebuilds_original_gguf(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-materialize-artifact")
        gguf_payload = b"abcdefghij"
        gguf_path = self._write_temp_file("materialize-artifact.gguf", gguf_payload)
        manifest = build_gguf_model_package_manifest(
            catalog_id="demo-materialize-artifact",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            version="v1",
            gguf_path=gguf_path,
            total_layers=5,
            chunk_size_policy=ChunkSizePolicy.ADAPTIVE,
            min_chunk_bytes=4,
            max_chunk_bytes=4,
            source_repo_id="Qwen/Qwen3-0.6B-GGUF",
        )
        save_model_package_manifest(manifest, policy)
        for chunk in manifest.chunks:
            payload = gguf_payload[
                chunk.offset_bytes : chunk.offset_bytes + chunk.size_bytes
            ]
            put_cached_chunk(
                catalog_id=manifest.catalog_id,
                version=manifest.version,
                chunk_id=chunk.chunk_id,
                sha256_hex=chunk.sha256_hex,
                content=payload,
                policy=policy,
            )

        materialized = materialize_artifact_from_store(
            manifest,
            "gguf-main",
            policy=policy,
        )

        output_path = Path(materialized.output_path)
        self.assertTrue(output_path.exists())
        self.assertEqual(output_path.read_bytes(), gguf_payload)
        self.assertEqual(
            output_path,
            materialized_artifact_path(manifest, "gguf-main", policy=policy),
        )
        self.assertEqual(materialized.size_bytes, len(gguf_payload))

    def test_materialize_default_artifact_from_store_prefers_gguf(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-materialize-default")
        gguf_payload = b"abcdefghij"
        gguf_path = self._write_temp_file("materialize-default.gguf", gguf_payload)
        manifest = build_gguf_model_package_manifest(
            catalog_id="demo-materialize-default",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            version="v1",
            gguf_path=gguf_path,
            total_layers=5,
            chunk_size_policy=ChunkSizePolicy.ADAPTIVE,
            min_chunk_bytes=4,
            max_chunk_bytes=4,
            source_repo_id="Qwen/Qwen3-0.6B-GGUF",
        )
        save_model_package_manifest(manifest, policy)
        for chunk in manifest.chunks:
            payload = gguf_payload[
                chunk.offset_bytes : chunk.offset_bytes + chunk.size_bytes
            ]
            put_cached_chunk(
                catalog_id=manifest.catalog_id,
                version=manifest.version,
                chunk_id=chunk.chunk_id,
                sha256_hex=chunk.sha256_hex,
                content=payload,
                policy=policy,
            )

        materialized = materialize_default_artifact_from_store(
            manifest,
            policy=policy,
        )

        self.assertTrue(materialized.output_path.endswith(".gguf"))
        self.assertEqual(Path(materialized.output_path).read_bytes(), gguf_payload)

    def test_materialize_assignment_artifact_writes_only_required_chunks(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-materialize-assignment")
        gguf_payload = b"abcdefghijklmnop"
        gguf_path = self._write_temp_file(
            "materialize-assignment.gguf",
            gguf_payload,
        )
        manifest = build_gguf_model_package_manifest(
            catalog_id="demo-materialize-assignment",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            version="v1",
            gguf_path=gguf_path,
            total_layers=8,
            min_chunk_bytes=1,
            max_chunk_bytes=4,
            target_chunk_count=4,
            source_repo_id="Qwen/Qwen3-0.6B-GGUF",
        )
        save_model_package_manifest(manifest, policy)
        assignment = ModelShardAssignment(
            start_layer=0,
            end_layer=2,
            device_rank=0,
            world_size=4,
            node_id="node-a",
        )
        required_chunks = manifest.required_chunks_for_layers(0, 2)
        missing_chunks = [
            chunk for chunk in manifest.chunks if chunk.chunk_id not in {
                item.chunk_id for item in required_chunks
            }
        ]
        self.assertTrue(required_chunks)
        self.assertTrue(missing_chunks)

        for chunk in required_chunks:
            payload = gguf_payload[
                chunk.offset_bytes : chunk.offset_bytes + chunk.size_bytes
            ]
            put_cached_chunk(
                catalog_id=manifest.catalog_id,
                version=manifest.version,
                chunk_id=chunk.chunk_id,
                sha256_hex=chunk.sha256_hex,
                content=payload,
                policy=policy,
            )

        materialized = materialize_default_assignment_artifact_from_store(
            manifest,
            assignment,
            policy=policy,
        )

        output_path = Path(materialized.output_path)
        self.assertEqual(
            output_path,
            materialized_assignment_artifact_path(
                manifest,
                "gguf-main",
                assignment,
                policy=policy,
            ),
        )
        self.assertEqual(output_path.stat().st_size, len(gguf_payload))
        materialized_payload = output_path.read_bytes()
        for chunk in required_chunks:
            self.assertEqual(
                materialized_payload[
                    chunk.offset_bytes : chunk.offset_bytes + chunk.size_bytes
                ],
                gguf_payload[
                    chunk.offset_bytes : chunk.offset_bytes + chunk.size_bytes
                ],
            )
        for chunk in missing_chunks:
            self.assertEqual(
                materialized_payload[
                    chunk.offset_bytes : chunk.offset_bytes + chunk.size_bytes
                ],
                b"\0" * chunk.size_bytes,
            )
        self.assertNotEqual(materialized_payload, gguf_payload)

    def test_execute_chunk_download_queue_materializes_chunks_from_bound_peer_cache(self) -> None:
        local_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-local-peer-fetch")
        remote_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-remote-peer-fetch")
        gguf_payload = b"abcdefghij"
        gguf_path = self._write_temp_file("peer-materialize.gguf", gguf_payload)
        manifest = build_gguf_model_package_manifest(
            catalog_id="demo-peer-fetch",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            version="v1",
            gguf_path=gguf_path,
            total_layers=5,
            chunk_size_policy=ChunkSizePolicy.ADAPTIVE,
            min_chunk_bytes=4,
            max_chunk_bytes=4,
            source_repo_id="Qwen/Qwen3-0.6B-GGUF",
        )
        save_model_package_manifest(manifest, local_policy)
        required_chunks = manifest.required_chunks_for_layers(0, 5)
        for chunk in required_chunks:
            payload = gguf_payload[
                chunk.offset_bytes : chunk.offset_bytes + chunk.size_bytes
            ]
            put_cached_chunk(
                catalog_id=manifest.catalog_id,
                version=manifest.version,
                chunk_id=chunk.chunk_id,
                sha256_hex=chunk.sha256_hex,
                content=payload,
                policy=remote_policy,
            )
        remote_root = Path(self.tempdir.name) / remote_policy.wallet_data_dirname
        save_chunk_source_binding(
            ChunkFetchSourceKind.PEER_CACHE,
            "peer-a",
            data_root_path=remote_root,
            policy=local_policy,
        )
        fetch_plan = manifest.build_assignment_fetch_plan(
            ModelShardAssignment(start_layer=0, end_layer=5, node_id="node-a"),
            present_chunk_ids=set(),
            peer_chunk_inventory={
                "peer-a": {chunk.chunk_id for chunk in required_chunks}
            },
        )
        queued = queue_assignment_fetch_plan(manifest, fetch_plan, policy=local_policy)
        self.assertEqual(len(queued), len(required_chunks))
        processed = execute_chunk_download_queue(policy=local_policy)
        self.assertEqual(len(processed), len(required_chunks))
        self.assertTrue(all(task.status == ChunkDownloadTaskStatus.COMPLETED for task in processed))
        self.assertTrue(
            all(task.selected_source_kind == ChunkFetchSourceKind.PEER_CACHE for task in processed)
        )
        snapshot = chunk_store_snapshot(local_policy)
        self.assertEqual(snapshot.stats.chunk_count, len(required_chunks))
        first_chunk_record = get_cached_chunk_record(
            required_chunks[0].chunk_id,
            local_policy,
        )
        self.assertIsNotNone(first_chunk_record)
        assert first_chunk_record is not None
        self.assertEqual(first_chunk_record.cache_class, ChunkCacheClass.HOT)
        self.assertEqual(first_chunk_record.lease_status, ChunkLeaseStatus.ACTIVE)

    def test_execute_chunk_download_queue_falls_back_from_peer_to_seed_and_records_source_health(self) -> None:
        local_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-peer-seed-fallback-local")
        peer_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-peer-seed-fallback-peer")
        seed_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-peer-seed-fallback-seed")
        gguf_payload = b"abcdefghij"
        gguf_path = self._write_temp_file("peer-seed-fallback.gguf", gguf_payload)
        manifest = build_gguf_model_package_manifest(
            catalog_id="demo-peer-seed-fallback",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            version="v1",
            gguf_path=gguf_path,
            total_layers=5,
            chunk_size_policy=ChunkSizePolicy.ADAPTIVE,
            min_chunk_bytes=4,
            max_chunk_bytes=4,
            source_repo_id="Qwen/Qwen3-0.6B-GGUF",
        )
        save_model_package_manifest(manifest, local_policy)
        required_chunks = manifest.required_chunks_for_layers(0, 5)
        for chunk in required_chunks:
            payload = gguf_payload[
                chunk.offset_bytes : chunk.offset_bytes + chunk.size_bytes
            ]
            put_cached_chunk(
                catalog_id=manifest.catalog_id,
                version=manifest.version,
                chunk_id=chunk.chunk_id,
                sha256_hex=chunk.sha256_hex,
                content=payload,
                policy=seed_policy,
            )
        peer_root = Path(self.tempdir.name) / peer_policy.wallet_data_dirname
        seed_root = Path(self.tempdir.name) / seed_policy.wallet_data_dirname
        peer_root.mkdir(parents=True, exist_ok=True)
        save_chunk_source_binding(
            ChunkFetchSourceKind.PEER_CACHE,
            "peer-a",
            data_root_path=peer_root,
            policy=local_policy,
        )
        save_chunk_source_binding(
            ChunkFetchSourceKind.STORAGE_SEED,
            "seed-a",
            data_root_path=seed_root,
            policy=local_policy,
        )
        fetch_plan = manifest.build_assignment_fetch_plan(
            ModelShardAssignment(start_layer=0, end_layer=5, node_id="node-a"),
            present_chunk_ids=set(),
            peer_chunk_inventory={"peer-a": {chunk.chunk_id for chunk in required_chunks}},
            seed_chunk_inventory={"seed-a": {chunk.chunk_id for chunk in required_chunks}},
        )
        queue_assignment_fetch_plan(manifest, fetch_plan, policy=local_policy)

        processed = execute_chunk_download_queue(policy=local_policy)

        self.assertEqual(len(processed), len(required_chunks))
        self.assertTrue(
            all(task.selected_source_kind == ChunkFetchSourceKind.STORAGE_SEED for task in processed)
        )
        peer_health = get_chunk_source_health_record(
            ChunkFetchSourceKind.PEER_CACHE,
            "peer-a",
            policy=local_policy,
        )
        self.assertIsNotNone(peer_health)
        assert peer_health is not None
        self.assertEqual(peer_health.failure_count, 1)
        self.assertGreaterEqual(peer_health.consecutive_failures, 1)
        self.assertIsNotNone(peer_health.cooldown_until)
        seed_health = get_chunk_source_health_record(
            ChunkFetchSourceKind.STORAGE_SEED,
            "seed-a",
            policy=local_policy,
        )
        self.assertIsNotNone(seed_health)
        assert seed_health is not None
        self.assertEqual(seed_health.success_count, len(required_chunks))
        self.assertEqual(seed_health.consecutive_failures, 0)

    def test_execute_chunk_download_queue_skips_cooldown_peer_when_seed_is_available(self) -> None:
        local_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-peer-seed-cooldown-local")
        peer_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-peer-seed-cooldown-peer")
        seed_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-peer-seed-cooldown-seed")
        gguf_payload = b"abcdefghij"
        gguf_path = self._write_temp_file("peer-seed-cooldown.gguf", gguf_payload)
        manifest = build_gguf_model_package_manifest(
            catalog_id="demo-peer-seed-cooldown",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            version="v1",
            gguf_path=gguf_path,
            total_layers=5,
            chunk_size_policy=ChunkSizePolicy.ADAPTIVE,
            min_chunk_bytes=4,
            max_chunk_bytes=4,
            source_repo_id="Qwen/Qwen3-0.6B-GGUF",
        )
        save_model_package_manifest(manifest, local_policy)
        required_chunks = manifest.required_chunks_for_layers(0, 5)
        for chunk in required_chunks:
            payload = gguf_payload[
                chunk.offset_bytes : chunk.offset_bytes + chunk.size_bytes
            ]
            put_cached_chunk(
                catalog_id=manifest.catalog_id,
                version=manifest.version,
                chunk_id=chunk.chunk_id,
                sha256_hex=chunk.sha256_hex,
                content=payload,
                policy=seed_policy,
            )
        peer_root = Path(self.tempdir.name) / peer_policy.wallet_data_dirname
        seed_root = Path(self.tempdir.name) / seed_policy.wallet_data_dirname
        peer_root.mkdir(parents=True, exist_ok=True)
        save_chunk_source_binding(
            ChunkFetchSourceKind.PEER_CACHE,
            "peer-a",
            data_root_path=peer_root,
            policy=local_policy,
        )
        save_chunk_source_binding(
            ChunkFetchSourceKind.STORAGE_SEED,
            "seed-a",
            data_root_path=seed_root,
            policy=local_policy,
        )
        fetch_plan = manifest.build_assignment_fetch_plan(
            ModelShardAssignment(start_layer=0, end_layer=5, node_id="node-a"),
            present_chunk_ids=set(),
            peer_chunk_inventory={"peer-a": {chunk.chunk_id for chunk in required_chunks}},
            seed_chunk_inventory={"seed-a": {chunk.chunk_id for chunk in required_chunks}},
        )
        queue_assignment_fetch_plan(manifest, fetch_plan, policy=local_policy)
        execute_chunk_download_queue(
            policy=local_policy,
            fetch_policy=ChunkFetchPolicy(source_failure_cooldown_seconds=600),
        )

        for chunk in required_chunks:
            delete_cached_chunk(chunk.chunk_id, local_policy)
        queued = list_chunk_download_tasks(local_policy)
        for task in queued:
            update_chunk_download_task_status(
                task.task_id,
                ChunkDownloadTaskStatus.QUEUED,
                selected_source_kind=ChunkFetchSourceKind.PEER_CACHE,
                selected_source_id="peer-a",
                last_error=None,
                policy=local_policy,
            )

        processed = execute_chunk_download_queue(
            policy=local_policy,
            fetch_policy=ChunkFetchPolicy(source_failure_cooldown_seconds=600),
        )

        self.assertEqual(len(processed), len(required_chunks))
        self.assertTrue(
            all(task.selected_source_kind == ChunkFetchSourceKind.STORAGE_SEED for task in processed)
        )
        peer_health = get_chunk_source_health_record(
            ChunkFetchSourceKind.PEER_CACHE,
            "peer-a",
            policy=local_policy,
        )
        self.assertIsNotNone(peer_health)
        assert peer_health is not None
        self.assertEqual(peer_health.failure_count, 1)

    def test_build_assignment_fetch_plan_from_store_prefers_seed_when_peer_is_in_cooldown(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-fetch-plan-health")
        gguf_payload = b"abcdefghij"
        gguf_path = self._write_temp_file("fetch-plan-health.gguf", gguf_payload)
        manifest = build_gguf_model_package_manifest(
            catalog_id="demo-fetch-plan-health",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            version="v1",
            gguf_path=gguf_path,
            total_layers=5,
            chunk_size_policy=ChunkSizePolicy.ADAPTIVE,
            min_chunk_bytes=4,
            max_chunk_bytes=4,
            source_repo_id="Qwen/Qwen3-0.6B-GGUF",
        )
        save_model_package_manifest(manifest, policy)
        required_chunks = manifest.required_chunks_for_layers(0, 5)
        peer_source = manifest.build_assignment_fetch_plan(
            ModelShardAssignment(start_layer=0, end_layer=5, node_id="node-a"),
            present_chunk_ids=set(),
            peer_chunk_inventory={"peer-a": {chunk.chunk_id for chunk in required_chunks}},
            seed_chunk_inventory={"seed-a": {chunk.chunk_id for chunk in required_chunks}},
        ).fetch_requests[0].sources[0]
        record_chunk_source_failure(
            peer_source,
            error="peer down",
            fetch_policy=ChunkFetchPolicy(source_failure_cooldown_seconds=600),
            policy=policy,
        )

        fetch_plan = build_assignment_fetch_plan_from_store(
            manifest,
            ModelShardAssignment(start_layer=0, end_layer=5, node_id="node-a"),
            peer_chunk_inventory={"peer-a": {chunk.chunk_id for chunk in required_chunks}},
            seed_chunk_inventory={"seed-a": {chunk.chunk_id for chunk in required_chunks}},
            fetch_policy=ChunkFetchPolicy(source_failure_cooldown_seconds=600),
            policy=policy,
        )

        self.assertTrue(fetch_plan.fetch_requests)
        first_request_sources = fetch_plan.fetch_requests[0].sources
        self.assertEqual(first_request_sources[0].kind, ChunkFetchSourceKind.STORAGE_SEED)
        self.assertTrue(
            all(source.kind != ChunkFetchSourceKind.PEER_CACHE for source in first_request_sources[:-1])
        )

    def test_build_assignment_fetch_plan_from_store_keeps_only_peer_when_no_alternative_exists(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-fetch-plan-peer-only")
        gguf_payload = b"abcdefghij"
        gguf_path = self._write_temp_file("fetch-plan-peer-only.gguf", gguf_payload)
        manifest = build_gguf_model_package_manifest(
            catalog_id="demo-fetch-plan-peer-only",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            version="v1",
            gguf_path=gguf_path,
            total_layers=5,
            chunk_size_policy=ChunkSizePolicy.ADAPTIVE,
            min_chunk_bytes=4,
            max_chunk_bytes=4,
            source_repo_id=None,
        )
        save_model_package_manifest(manifest, policy)
        required_chunks = manifest.required_chunks_for_layers(0, 5)
        peer_source = manifest.build_assignment_fetch_plan(
            ModelShardAssignment(start_layer=0, end_layer=5, node_id="node-a"),
            present_chunk_ids=set(),
            peer_chunk_inventory={"peer-a": {chunk.chunk_id for chunk in required_chunks}},
        ).fetch_requests[0].sources[0]
        record_chunk_source_failure(
            peer_source,
            error="peer down",
            fetch_policy=ChunkFetchPolicy(source_failure_cooldown_seconds=600),
            policy=policy,
        )

        fetch_plan = build_assignment_fetch_plan_from_store(
            manifest,
            ModelShardAssignment(start_layer=0, end_layer=5, node_id="node-a"),
            peer_chunk_inventory={"peer-a": {chunk.chunk_id for chunk in required_chunks}},
            fetch_policy=ChunkFetchPolicy(source_failure_cooldown_seconds=600),
            policy=policy,
        )

        self.assertTrue(fetch_plan.fetch_requests)
        first_request_sources = fetch_plan.fetch_requests[0].sources
        self.assertEqual(first_request_sources[0].kind, ChunkFetchSourceKind.PEER_CACHE)

    def test_execute_chunk_download_queue_materializes_chunks_from_http_peer_transport(self) -> None:
        local_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-http-peer-fetch")
        gguf_payload = b"abcdefghij"
        gguf_path = self._write_temp_file("http-peer-materialize.gguf", gguf_payload)
        manifest = build_gguf_model_package_manifest(
            catalog_id="demo-http-peer-fetch",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            version="v1",
            gguf_path=gguf_path,
            total_layers=5,
            chunk_size_policy=ChunkSizePolicy.ADAPTIVE,
            min_chunk_bytes=4,
            max_chunk_bytes=4,
            source_repo_id="Qwen/Qwen3-0.6B-GGUF",
        )
        save_model_package_manifest(manifest, local_policy)
        required_chunks = manifest.required_chunks_for_layers(0, 5)
        remote_payloads = {
            chunk.chunk_id: gguf_payload[
                chunk.offset_bytes : chunk.offset_bytes + chunk.size_bytes
            ]
            for chunk in required_chunks
        }

        class _FakeResponse:
            def __init__(self, payload: bytes) -> None:
                self._payload = payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return self._payload

        def _fake_urlopen(url: str, timeout: int = 30):
            from urllib.parse import parse_qs, urlparse

            parsed = urlparse(url)
            query = parse_qs(parsed.query)
            chunk_id = query["chunk_id"][0]
            return _FakeResponse(remote_payloads[chunk_id])

        fetch_plan = manifest.build_assignment_fetch_plan(
            ModelShardAssignment(start_layer=0, end_layer=5, node_id="node-a"),
            present_chunk_ids=set(),
            peer_chunk_inventory={
                "peer-a": {chunk.chunk_id for chunk in required_chunks}
            },
            peer_chunk_locators={"peer-a": "http://198.51.100.10:52415"},
        )
        queued = queue_assignment_fetch_plan(manifest, fetch_plan, policy=local_policy)
        self.assertEqual(len(queued), len(required_chunks))
        with patch("cai_compute_chain.model_distribution.urlopen", side_effect=_fake_urlopen):
            processed = execute_chunk_download_queue(policy=local_policy)
        self.assertEqual(len(processed), len(required_chunks))
        self.assertTrue(all(task.status == ChunkDownloadTaskStatus.COMPLETED for task in processed))
        self.assertTrue(
            all(task.selected_source_kind == ChunkFetchSourceKind.PEER_CACHE for task in processed)
        )
        self.assertEqual(chunk_store_snapshot(local_policy).stats.chunk_count, len(required_chunks))
        first_chunk_record = get_cached_chunk_record(required_chunks[0].chunk_id, local_policy)
        self.assertIsNotNone(first_chunk_record)
        assert first_chunk_record is not None
        self.assertEqual(first_chunk_record.cache_class, ChunkCacheClass.HOT)
        self.assertEqual(first_chunk_record.lease_status, ChunkLeaseStatus.ACTIVE)

    def test_ensure_assignment_ready_from_store_materializes_from_origin(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-ensure-origin")
        gguf_path = self._write_temp_file("ensure-origin.gguf", b"abcdefghij")
        manifest = build_gguf_model_package_manifest(
            catalog_id="demo-ensure-origin",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            version="v1",
            gguf_path=gguf_path,
            total_layers=5,
            chunk_size_policy=ChunkSizePolicy.ADAPTIVE,
            min_chunk_bytes=4,
            max_chunk_bytes=4,
            source_repo_id="Qwen/Qwen3-0.6B-GGUF",
        )
        save_model_package_manifest(manifest, policy)
        save_local_artifact_binding(
            manifest.catalog_id,
            manifest.version,
            artifact_id="gguf-main",
            local_path=gguf_path,
            policy=policy,
        )
        result = ensure_assignment_ready_from_store(
            manifest,
            ModelShardAssignment(start_layer=0, end_layer=5, node_id="node-a"),
            policy=policy,
        )
        self.assertFalse(result.initial_plan.ready)
        self.assertTrue(result.final_plan.ready)
        self.assertTrue(result.ready)
        self.assertEqual(len(result.queued_tasks), len(manifest.chunks))
        self.assertEqual(len(result.processed_tasks), len(manifest.chunks))
        self.assertTrue(
            all(task.selected_source_kind == ChunkFetchSourceKind.ORIGIN for task in result.processed_tasks)
        )

    def test_ensure_assignment_ready_from_store_fetches_origin_ranges_without_local_artifact(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-origin-range")
        gguf_payload = b"abcdefghij"
        gguf_path = self._write_temp_file("origin-range.gguf", gguf_payload)
        manifest = build_gguf_model_package_manifest(
            catalog_id="demo-origin-range",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            version="v1",
            gguf_path=gguf_path,
            total_layers=5,
            chunk_size_policy=ChunkSizePolicy.ADAPTIVE,
            min_chunk_bytes=4,
            max_chunk_bytes=4,
            source_repo_id="Qwen/Qwen3-0.6B-GGUF",
            source_revision="pinned-rev",
        )
        save_model_package_manifest(manifest, policy)
        required_chunks = manifest.required_chunks_for_layers(0, 5)
        seen_requests: list[tuple[str, str]] = []

        class _RangeResponse:
            status = 206

            def __init__(self, payload: bytes) -> None:
                self._payload = payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def getcode(self) -> int:
                return self.status

            def read(self) -> bytes:
                return self._payload

        def _fake_urlopen(request, timeout: int = 30):
            range_header = request.get_header("Range")
            assert range_header is not None
            seen_requests.append((request.full_url, range_header))
            start_text, end_text = range_header.removeprefix("bytes=").split("-", 1)
            start = int(start_text)
            end = int(end_text)
            return _RangeResponse(gguf_payload[start : end + 1])

        with patch("cai_compute_chain.model_distribution.urlopen", side_effect=_fake_urlopen):
            result = ensure_assignment_ready_from_store(
                manifest,
                ModelShardAssignment(start_layer=0, end_layer=5, node_id="node-a"),
                policy=policy,
            )

        self.assertTrue(result.ready)
        self.assertEqual(len(result.processed_tasks), len(required_chunks))
        expected_url = (
            "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/"
            "resolve/pinned-rev/origin-range.gguf"
        )
        self.assertTrue(all(url == expected_url for url, _range in seen_requests))
        self.assertEqual(
            [range_header for _url, range_header in seen_requests],
            [
                f"bytes={chunk.offset_bytes}-{chunk.offset_bytes + chunk.size_bytes - 1}"
                for chunk in required_chunks
            ],
        )
        self.assertEqual(chunk_store_snapshot(policy).stats.chunk_count, len(required_chunks))

    def test_origin_range_fetch_rejects_full_response_without_reading_body(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-origin-no-range")
        gguf_path = self._write_temp_file("origin-no-range.gguf", b"abcdefghij")
        manifest = build_gguf_model_package_manifest(
            catalog_id="demo-origin-no-range",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            version="v1",
            gguf_path=gguf_path,
            total_layers=5,
            chunk_size_policy=ChunkSizePolicy.ADAPTIVE,
            min_chunk_bytes=4,
            max_chunk_bytes=4,
            source_repo_id="Qwen/Qwen3-0.6B-GGUF",
            source_revision="pinned-rev",
        )
        save_model_package_manifest(manifest, policy)
        read_calls = 0

        class _FullResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def getcode(self) -> int:
                return self.status

            def read(self) -> bytes:
                nonlocal read_calls
                read_calls += 1
                return b"this would be the full file"

        with patch("cai_compute_chain.model_distribution.urlopen", return_value=_FullResponse()):
            result = ensure_assignment_ready_from_store(
                manifest,
                ModelShardAssignment(start_layer=0, end_layer=5, node_id="node-a"),
                policy=policy,
            )

        self.assertFalse(result.ready)
        self.assertEqual(read_calls, 0)
        self.assertTrue(result.processed_tasks)
        self.assertTrue(
            all(task.status == ChunkDownloadTaskStatus.FAILED for task in result.processed_tasks)
        )
        self.assertTrue(
            all("expected HTTP 206" in (task.last_error or "") for task in result.processed_tasks)
        )

    def test_ensure_assignment_ready_from_store_promotes_already_cached_chunks(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-ensure-promote")
        gguf_payload = b"abcdefghij"
        gguf_path = self._write_temp_file("ensure-promote.gguf", gguf_payload)
        manifest = build_gguf_model_package_manifest(
            catalog_id="demo-ensure-promote",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            version="v1",
            gguf_path=gguf_path,
            total_layers=5,
            chunk_size_policy=ChunkSizePolicy.ADAPTIVE,
            min_chunk_bytes=4,
            max_chunk_bytes=4,
            source_repo_id="Qwen/Qwen3-0.6B-GGUF",
        )
        save_model_package_manifest(manifest, policy)
        required_chunks = manifest.required_chunks_for_layers(0, 5)
        for chunk in required_chunks:
            payload = gguf_payload[
                chunk.offset_bytes : chunk.offset_bytes + chunk.size_bytes
            ]
            put_cached_chunk(
                catalog_id=manifest.catalog_id,
                version=manifest.version,
                chunk_id=chunk.chunk_id,
                sha256_hex=chunk.sha256_hex,
                content=payload,
                cache_class=ChunkCacheClass.COLD,
                policy=policy,
            )

        result = ensure_assignment_ready_from_store(
            manifest,
            ModelShardAssignment(start_layer=0, end_layer=5, node_id="node-a"),
            policy=policy,
        )

        self.assertTrue(result.initial_plan.ready)
        self.assertTrue(result.final_plan.ready)
        self.assertEqual(result.queued_tasks, ())
        self.assertEqual(result.processed_tasks, ())
        for chunk in required_chunks:
            record = get_cached_chunk_record(chunk.chunk_id, policy)
            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record.cache_class, ChunkCacheClass.HOT)
            self.assertEqual(record.lease_status, ChunkLeaseStatus.ACTIVE)

    def test_ensure_assignment_ready_from_store_materializes_from_peer_cache(self) -> None:
        local_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-ensure-peer-local")
        remote_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-ensure-peer-remote")
        gguf_payload = b"abcdefghij"
        gguf_path = self._write_temp_file("ensure-peer.gguf", gguf_payload)
        manifest = build_gguf_model_package_manifest(
            catalog_id="demo-ensure-peer",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            version="v1",
            gguf_path=gguf_path,
            total_layers=5,
            chunk_size_policy=ChunkSizePolicy.ADAPTIVE,
            min_chunk_bytes=4,
            max_chunk_bytes=4,
            source_repo_id="Qwen/Qwen3-0.6B-GGUF",
        )
        save_model_package_manifest(manifest, local_policy)
        required_chunks = manifest.required_chunks_for_layers(0, 5)
        for chunk in required_chunks:
            payload = gguf_payload[
                chunk.offset_bytes : chunk.offset_bytes + chunk.size_bytes
            ]
            put_cached_chunk(
                catalog_id=manifest.catalog_id,
                version=manifest.version,
                chunk_id=chunk.chunk_id,
                sha256_hex=chunk.sha256_hex,
                content=payload,
                policy=remote_policy,
            )
        remote_root = Path(self.tempdir.name) / remote_policy.wallet_data_dirname
        save_chunk_source_binding(
            ChunkFetchSourceKind.PEER_CACHE,
            "peer-a",
            data_root_path=remote_root,
            policy=local_policy,
        )
        result = ensure_assignment_ready_from_store(
            manifest,
            ModelShardAssignment(start_layer=0, end_layer=5, node_id="node-a"),
            peer_chunk_inventory={
                "peer-a": {chunk.chunk_id for chunk in required_chunks}
            },
            policy=local_policy,
        )
        self.assertFalse(result.initial_plan.ready)
        self.assertTrue(result.final_plan.ready)
        self.assertTrue(result.ready)
        self.assertEqual(len(result.queued_tasks), len(required_chunks))
        self.assertEqual(len(result.processed_tasks), len(required_chunks))
        self.assertTrue(
            all(task.selected_source_kind == ChunkFetchSourceKind.PEER_CACHE for task in result.processed_tasks)
        )

    def test_ensure_default_chunks_ready_from_store_materializes_tokenizer_from_peer_cache(self) -> None:
        local_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-default-peer-local")
        remote_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-default-peer-remote")
        tokenizer_payload = b'{"vocab":["hello"]}'
        tokenizer_sha = hashlib.sha256(tokenizer_payload).hexdigest()
        manifest = ModelPackageManifest(
            catalog_id="demo-default-peer",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            version="v1",
            backend="llama_cpp",
            package_kind=ModelPackageKind.PUBLIC_SHARED,
            total_size_bytes=len(tokenizer_payload),
            source_repo_id=None,
            files=[
                SourceArtifact(
                    artifact_id="tokenizer",
                    relative_path="tokenizer.json",
                    size_bytes=len(tokenizer_payload),
                    sha256_hex=tokenizer_sha,
                )
            ],
            chunks=[
                ModelChunk(
                    chunk_id=make_chunk_id(
                        "tokenizer",
                        offset_bytes=0,
                        size_bytes=len(tokenizer_payload),
                        sha256_hex=tokenizer_sha,
                    ),
                    artifact_id="tokenizer",
                    kind=ModelChunkKind.TOKENIZER,
                    offset_bytes=0,
                    size_bytes=len(tokenizer_payload),
                    sha256_hex=tokenizer_sha,
                    required_by_default=True,
                )
            ],
        )
        save_model_package_manifest(manifest, local_policy)
        tokenizer_chunk = manifest.default_chunks()[0]
        put_cached_chunk(
            catalog_id=manifest.catalog_id,
            version=manifest.version,
            chunk_id=tokenizer_chunk.chunk_id,
            sha256_hex=tokenizer_chunk.sha256_hex,
            content=tokenizer_payload,
            policy=remote_policy,
        )
        save_chunk_source_binding(
            ChunkFetchSourceKind.PEER_CACHE,
            "peer-a",
            data_root_path=data_root(remote_policy),
            policy=local_policy,
        )
        import_chunk_inventory_payload(
            ChunkInventoryPayload(
                source_id="peer-a",
                source_kind=ChunkInventorySourceKind.PEER_CACHE,
                published_at=_fresh_published_at(),
                records=(
                    ChunkInventoryRecord(
                        catalog_id=manifest.catalog_id,
                        version=manifest.version,
                        chunk_ids=(tokenizer_chunk.chunk_id,),
                        chunk_count=1,
                        total_bytes=tokenizer_chunk.size_bytes,
                    ),
                ),
            ),
            local_policy,
        )

        result = ensure_default_chunks_ready_from_store(
            manifest,
            node_id="node-a",
            use_imported_peer_inventory=True,
            fetch_policy=ChunkFetchPolicy(max_inventory_age_seconds=300),
            policy=local_policy,
        )

        self.assertFalse(result.initial_plan.ready)
        self.assertTrue(result.final_plan.ready)
        self.assertTrue(result.ready)
        self.assertEqual(len(result.queued_tasks), 1)
        self.assertEqual(len(result.processed_tasks), 1)
        self.assertEqual(
            result.processed_tasks[0].selected_source_kind,
            ChunkFetchSourceKind.PEER_CACHE,
        )
        cached = get_cached_chunk_record(tokenizer_chunk.chunk_id, local_policy)
        self.assertIsNotNone(cached)

    def test_prefetch_default_chunks_from_fresh_inventories_prefetches_tokenizer(self) -> None:
        local_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-default-prefetch-local")
        remote_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-default-prefetch-remote")
        tokenizer_payload = b'{"vocab":["world"]}'
        tokenizer_sha = hashlib.sha256(tokenizer_payload).hexdigest()
        manifest = ModelPackageManifest(
            catalog_id="demo-default-prefetch",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            version="v1",
            backend="llama_cpp",
            package_kind=ModelPackageKind.PUBLIC_SHARED,
            total_size_bytes=len(tokenizer_payload),
            source_repo_id=None,
            files=[
                SourceArtifact(
                    artifact_id="tokenizer",
                    relative_path="tokenizer.json",
                    size_bytes=len(tokenizer_payload),
                    sha256_hex=tokenizer_sha,
                )
            ],
            chunks=[
                ModelChunk(
                    chunk_id=make_chunk_id(
                        "tokenizer",
                        offset_bytes=0,
                        size_bytes=len(tokenizer_payload),
                        sha256_hex=tokenizer_sha,
                    ),
                    artifact_id="tokenizer",
                    kind=ModelChunkKind.TOKENIZER,
                    offset_bytes=0,
                    size_bytes=len(tokenizer_payload),
                    sha256_hex=tokenizer_sha,
                    required_by_default=True,
                )
            ],
        )
        save_model_package_manifest(manifest, local_policy)
        tokenizer_chunk = manifest.default_chunks()[0]
        put_cached_chunk(
            catalog_id=manifest.catalog_id,
            version=manifest.version,
            chunk_id=tokenizer_chunk.chunk_id,
            sha256_hex=tokenizer_chunk.sha256_hex,
            content=tokenizer_payload,
            policy=remote_policy,
        )
        save_chunk_source_binding(
            ChunkFetchSourceKind.PEER_CACHE,
            "peer-a",
            data_root_path=data_root(remote_policy),
            policy=local_policy,
        )
        import_chunk_inventory_payload(
            ChunkInventoryPayload(
                source_id="peer-a",
                source_kind=ChunkInventorySourceKind.PEER_CACHE,
                published_at=_fresh_published_at(),
                records=(
                    ChunkInventoryRecord(
                        catalog_id=manifest.catalog_id,
                        version=manifest.version,
                        chunk_ids=(tokenizer_chunk.chunk_id,),
                        chunk_count=1,
                        total_bytes=tokenizer_chunk.size_bytes,
                    ),
                ),
            ),
            local_policy,
        )

        result = prefetch_default_chunks_from_fresh_inventories(
            node_id="node-a",
            max_manifests=4,
            max_tasks=4,
            fetch_policy=ChunkFetchPolicy(max_inventory_age_seconds=300),
            policy=local_policy,
        )

        self.assertEqual(result.manifests_considered, 1)
        self.assertEqual(result.manifests_prefetched, 1)
        self.assertEqual(result.queued_tasks, 1)
        self.assertEqual(result.processed_tasks, 1)
        cached = get_cached_chunk_record(tokenizer_chunk.chunk_id, local_policy)
        self.assertIsNotNone(cached)

    def test_build_bootstrap_chunk_fetch_plan_from_store_limits_weight_chunks(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-bootstrap-plan")
        manifest = _build_manifest()
        save_model_package_manifest(manifest, policy)
        assignment = ModelShardAssignment(start_layer=0, end_layer=28, node_id="node-a")
        all_chunk_ids = {chunk.chunk_id for chunk in manifest.chunks}

        fetch_plan = build_bootstrap_chunk_fetch_plan_from_store(
            manifest,
            node_id="node-a",
            peer_chunk_inventory={"peer-a": all_chunk_ids},
            max_weight_chunks=1,
            max_weight_bytes=1024,
            fetch_policy=ChunkFetchPolicy(max_inventory_age_seconds=300),
            policy=policy,
        )

        self.assertEqual(len(fetch_plan.fetch_requests), 2)
        chunk_ids = {request.chunk_id for request in fetch_plan.fetch_requests}
        tokenizer_chunk = next(chunk for chunk in manifest.chunks if chunk.kind == ModelChunkKind.TOKENIZER)
        first_weight_chunk = next(
            chunk
            for chunk in manifest.chunks
            if chunk.kind == ModelChunkKind.WEIGHTS and chunk.layer_start == 0
        )
        later_weight_chunk = next(
            chunk
            for chunk in manifest.chunks
            if chunk.kind == ModelChunkKind.WEIGHTS and chunk.layer_start == 10
        )
        self.assertIn(tokenizer_chunk.chunk_id, chunk_ids)
        self.assertIn(first_weight_chunk.chunk_id, chunk_ids)
        self.assertNotIn(later_weight_chunk.chunk_id, chunk_ids)

    def test_prefetch_bootstrap_chunks_from_fresh_inventories_prefetches_first_weight_only(self) -> None:
        local_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-bootstrap-prefetch-local")
        remote_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-bootstrap-prefetch-remote")
        tokenizer_payload = b'{"vocab":["bootstrap"]}'
        weight_a = b"AAAA"
        weight_b = b"BBBB"
        tokenizer_sha = hashlib.sha256(tokenizer_payload).hexdigest()
        weight_a_sha = hashlib.sha256(weight_a).hexdigest()
        weight_b_sha = hashlib.sha256(weight_b).hexdigest()
        manifest = ModelPackageManifest(
            catalog_id="demo-bootstrap-prefetch",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            version="v1",
            backend="llama_cpp",
            package_kind=ModelPackageKind.PUBLIC_SHARED,
            total_size_bytes=len(tokenizer_payload) + len(weight_a) + len(weight_b),
            source_repo_id=None,
            files=[
                SourceArtifact(
                    artifact_id="tokenizer",
                    relative_path="tokenizer.json",
                    size_bytes=len(tokenizer_payload),
                    sha256_hex=tokenizer_sha,
                ),
                SourceArtifact(
                    artifact_id="gguf-main",
                    relative_path="model.gguf",
                    size_bytes=len(weight_a) + len(weight_b),
                    sha256_hex=hashlib.sha256(weight_a + weight_b).hexdigest(),
                ),
            ],
            chunks=[
                ModelChunk(
                    chunk_id=make_chunk_id(
                        "tokenizer",
                        offset_bytes=0,
                        size_bytes=len(tokenizer_payload),
                        sha256_hex=tokenizer_sha,
                    ),
                    artifact_id="tokenizer",
                    kind=ModelChunkKind.TOKENIZER,
                    offset_bytes=0,
                    size_bytes=len(tokenizer_payload),
                    sha256_hex=tokenizer_sha,
                    required_by_default=True,
                ),
                ModelChunk(
                    chunk_id=make_chunk_id(
                        "gguf-main",
                        offset_bytes=0,
                        size_bytes=len(weight_a),
                        sha256_hex=weight_a_sha,
                    ),
                    artifact_id="gguf-main",
                    kind=ModelChunkKind.WEIGHTS,
                    offset_bytes=0,
                    size_bytes=len(weight_a),
                    sha256_hex=weight_a_sha,
                    layer_start=0,
                    layer_end=1,
                ),
                ModelChunk(
                    chunk_id=make_chunk_id(
                        "gguf-main",
                        offset_bytes=len(weight_a),
                        size_bytes=len(weight_b),
                        sha256_hex=weight_b_sha,
                    ),
                    artifact_id="gguf-main",
                    kind=ModelChunkKind.WEIGHTS,
                    offset_bytes=len(weight_a),
                    size_bytes=len(weight_b),
                    sha256_hex=weight_b_sha,
                    layer_start=1,
                    layer_end=2,
                ),
            ],
        )
        save_model_package_manifest(manifest, local_policy)
        tokenizer_chunk = next(chunk for chunk in manifest.chunks if chunk.kind == ModelChunkKind.TOKENIZER)
        first_weight_chunk = next(
            chunk for chunk in manifest.chunks if chunk.kind == ModelChunkKind.WEIGHTS and chunk.layer_start == 0
        )
        second_weight_chunk = next(
            chunk for chunk in manifest.chunks if chunk.kind == ModelChunkKind.WEIGHTS and chunk.layer_start == 1
        )
        put_cached_chunk(
            catalog_id=manifest.catalog_id,
            version=manifest.version,
            chunk_id=tokenizer_chunk.chunk_id,
            sha256_hex=tokenizer_chunk.sha256_hex,
            content=tokenizer_payload,
            policy=remote_policy,
        )
        put_cached_chunk(
            catalog_id=manifest.catalog_id,
            version=manifest.version,
            chunk_id=first_weight_chunk.chunk_id,
            sha256_hex=first_weight_chunk.sha256_hex,
            content=weight_a,
            policy=remote_policy,
        )
        put_cached_chunk(
            catalog_id=manifest.catalog_id,
            version=manifest.version,
            chunk_id=second_weight_chunk.chunk_id,
            sha256_hex=second_weight_chunk.sha256_hex,
            content=weight_b,
            policy=remote_policy,
        )
        save_chunk_source_binding(
            ChunkFetchSourceKind.PEER_CACHE,
            "peer-a",
            data_root_path=data_root(remote_policy),
            policy=local_policy,
        )
        import_chunk_inventory_payload(
            ChunkInventoryPayload(
                source_id="peer-a",
                source_kind=ChunkInventorySourceKind.PEER_CACHE,
                published_at=_fresh_published_at(),
                records=(
                    ChunkInventoryRecord(
                        catalog_id=manifest.catalog_id,
                        version=manifest.version,
                        chunk_ids=(
                            tokenizer_chunk.chunk_id,
                            first_weight_chunk.chunk_id,
                            second_weight_chunk.chunk_id,
                        ),
                        chunk_count=3,
                        total_bytes=(
                            tokenizer_chunk.size_bytes
                            + first_weight_chunk.size_bytes
                            + second_weight_chunk.size_bytes
                        ),
                    ),
                ),
            ),
            local_policy,
        )

        result = prefetch_bootstrap_chunks_from_fresh_inventories(
            node_id="node-a",
            max_manifests=2,
            max_tasks=4,
            fetch_policy=ChunkFetchPolicy(
                max_inventory_age_seconds=300,
                warm_prefetch_weight_chunk_count_per_manifest=1,
                warm_prefetch_max_weight_bytes_per_manifest=8,
            ),
            policy=local_policy,
        )

        self.assertEqual(result.manifests_considered, 1)
        self.assertEqual(result.manifests_prefetched, 1)
        self.assertEqual(result.queued_tasks, 2)
        self.assertEqual(result.processed_tasks, 2)
        self.assertIsNotNone(get_cached_chunk_record(tokenizer_chunk.chunk_id, local_policy))
        self.assertIsNotNone(get_cached_chunk_record(first_weight_chunk.chunk_id, local_policy))
        self.assertIsNone(get_cached_chunk_record(second_weight_chunk.chunk_id, local_policy))

    def test_background_prefetch_skips_private_curated_manifests(self) -> None:
        local_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-private-prefetch-local")
        remote_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-private-prefetch-remote")
        tokenizer_payload = b'{"vocab":["private"]}'
        tokenizer_sha = hashlib.sha256(tokenizer_payload).hexdigest()
        manifest = ModelPackageManifest(
            catalog_id="demo-private-prefetch",
            model_id="cai-network/Qwen3-0.6B-GGUF",
            version="v1",
            backend="llama_cpp",
            package_kind=ModelPackageKind.PRIVATE_CURATED,
            total_size_bytes=len(tokenizer_payload),
            source_repo_id="cai-network/Qwen3-0.6B-GGUF",
            files=[
                SourceArtifact(
                    artifact_id="tokenizer",
                    relative_path="tokenizer.json",
                    size_bytes=len(tokenizer_payload),
                    sha256_hex=tokenizer_sha,
                )
            ],
            chunks=[
                ModelChunk(
                    chunk_id=make_chunk_id(
                        "tokenizer",
                        offset_bytes=0,
                        size_bytes=len(tokenizer_payload),
                        sha256_hex=tokenizer_sha,
                    ),
                    artifact_id="tokenizer",
                    kind=ModelChunkKind.TOKENIZER,
                    offset_bytes=0,
                    size_bytes=len(tokenizer_payload),
                    sha256_hex=tokenizer_sha,
                    required_by_default=True,
                )
            ],
        )
        save_model_package_manifest(manifest, local_policy)
        tokenizer_chunk = manifest.chunks[0]
        put_cached_chunk(
            catalog_id=manifest.catalog_id,
            version=manifest.version,
            chunk_id=tokenizer_chunk.chunk_id,
            sha256_hex=tokenizer_chunk.sha256_hex,
            content=tokenizer_payload,
            policy=remote_policy,
        )
        save_chunk_source_binding(
            ChunkFetchSourceKind.PEER_CACHE,
            "peer-private",
            data_root_path=data_root(remote_policy),
            policy=local_policy,
        )
        import_chunk_inventory_payload(
            ChunkInventoryPayload(
                source_id="peer-private",
                source_kind=ChunkInventorySourceKind.PEER_CACHE,
                published_at=_fresh_published_at(),
                records=(
                    ChunkInventoryRecord(
                        catalog_id=manifest.catalog_id,
                        version=manifest.version,
                        chunk_ids=(tokenizer_chunk.chunk_id,),
                        chunk_count=1,
                        total_bytes=tokenizer_chunk.size_bytes,
                    ),
                ),
            ),
            local_policy,
        )

        default_result = prefetch_default_chunks_from_fresh_inventories(
            node_id="node-a",
            max_manifests=4,
            max_tasks=4,
            fetch_policy=ChunkFetchPolicy(max_inventory_age_seconds=300),
            policy=local_policy,
        )
        bootstrap_result = prefetch_bootstrap_chunks_from_fresh_inventories(
            node_id="node-a",
            max_manifests=4,
            max_tasks=4,
            fetch_policy=ChunkFetchPolicy(
                max_inventory_age_seconds=300,
                warm_prefetch_weight_chunk_count_per_manifest=1,
                warm_prefetch_max_weight_bytes_per_manifest=8,
            ),
            policy=local_policy,
        )

        self.assertEqual(default_result.manifests_considered, 0)
        self.assertEqual(default_result.queued_tasks, 0)
        self.assertEqual(default_result.processed_tasks, 0)
        self.assertEqual(bootstrap_result.manifests_considered, 0)
        self.assertEqual(bootstrap_result.queued_tasks, 0)
        self.assertEqual(bootstrap_result.processed_tasks, 0)
        self.assertIsNone(get_cached_chunk_record(tokenizer_chunk.chunk_id, local_policy))

    def test_assignment_ready_fetches_only_assigned_private_chunks(self) -> None:
        local_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-private-assignment-local")
        remote_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-private-assignment-remote")
        gguf_payload = b"abcdefghijkl"
        gguf_path = self._write_temp_file("private-assignment.gguf", gguf_payload)
        manifest = build_gguf_model_package_manifest(
            catalog_id="demo-private-assignment",
            model_id="cai-network/Qwen3-0.6B-GGUF",
            version="v1",
            gguf_path=gguf_path,
            total_layers=6,
            package_kind=ModelPackageKind.PRIVATE_CURATED,
            chunk_size_policy=ChunkSizePolicy.ADAPTIVE,
            min_chunk_bytes=4,
            max_chunk_bytes=4,
            source_repo_id="cai-network/Qwen3-0.6B-GGUF",
        )
        save_model_package_manifest(manifest, local_policy)
        for chunk in manifest.chunks:
            payload = gguf_payload[chunk.offset_bytes : chunk.offset_bytes + chunk.size_bytes]
            put_cached_chunk(
                catalog_id=manifest.catalog_id,
                version=manifest.version,
                chunk_id=chunk.chunk_id,
                sha256_hex=chunk.sha256_hex,
                content=payload,
                policy=remote_policy,
            )
        save_chunk_source_binding(
            ChunkFetchSourceKind.PEER_CACHE,
            "peer-private",
            data_root_path=data_root(remote_policy),
            policy=local_policy,
        )
        import_chunk_inventory_payload(
            ChunkInventoryPayload(
                source_id="peer-private",
                source_kind=ChunkInventorySourceKind.PEER_CACHE,
                published_at=_fresh_published_at(),
                records=(
                    ChunkInventoryRecord(
                        catalog_id=manifest.catalog_id,
                        version=manifest.version,
                        chunk_ids=tuple(chunk.chunk_id for chunk in manifest.chunks),
                        chunk_count=len(manifest.chunks),
                        total_bytes=sum(chunk.size_bytes for chunk in manifest.chunks),
                    ),
                ),
            ),
            local_policy,
        )
        assignment = ModelShardAssignment(start_layer=2, end_layer=4, node_id="node-a")
        required_chunks = manifest.required_chunks_for_layers(2, 4)
        unrelated_chunks = [chunk for chunk in manifest.chunks if chunk not in required_chunks]

        result = ensure_assignment_ready_from_store(
            manifest,
            assignment,
            use_imported_peer_inventory=True,
            max_tasks=8,
            fetch_policy=ChunkFetchPolicy(max_inventory_age_seconds=300),
            policy=local_policy,
        )

        self.assertTrue(result.ready)
        self.assertEqual(
            {task.chunk_id for task in result.processed_tasks},
            {chunk.chunk_id for chunk in required_chunks},
        )
        for chunk in required_chunks:
            self.assertIsNotNone(get_cached_chunk_record(chunk.chunk_id, local_policy))
        for chunk in unrelated_chunks:
            self.assertIsNone(get_cached_chunk_record(chunk.chunk_id, local_policy))

    def test_recent_shard_hints_prune_old_records_and_keep_recent_capacity(self) -> None:
        policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-recent-hints")
        base_now = datetime.now(tz=UTC).replace(microsecond=0)
        fetch_policy = ChunkFetchPolicy(
            recent_shard_hint_ttl_seconds=3600,
            recent_shard_hint_capacity_per_node=2,
        )

        remember_recent_shard_hints(
            "node-a",
            [
                {
                    "model_id": "Qwen/Qwen3-0.6B-GGUF",
                    "start_layer": 0,
                    "end_layer": 8,
                    "device_rank": 0,
                    "world_size": 2,
                }
            ],
            fetch_policy=fetch_policy,
            policy=policy,
            now=base_now - timedelta(hours=2),
        )
        remember_recent_shard_hints(
            "node-a",
            [
                {
                    "model_id": "Qwen/Qwen3-0.6B-GGUF",
                    "start_layer": 8,
                    "end_layer": 16,
                    "device_rank": 1,
                    "world_size": 2,
                }
            ],
            fetch_policy=fetch_policy,
            policy=policy,
            now=base_now - timedelta(minutes=5),
        )
        remember_recent_shard_hints(
            "node-a",
            [
                {
                    "model_id": "Qwen/Qwen3-0.6B-GGUF",
                    "start_layer": 16,
                    "end_layer": 24,
                    "device_rank": 0,
                    "world_size": 2,
                }
            ],
            fetch_policy=fetch_policy,
            policy=policy,
            now=base_now,
        )

        recent_records = list_recent_shard_hints(
            "node-a",
            fetch_policy=fetch_policy,
            policy=policy,
        )

        self.assertEqual(len(recent_records), 2)
        self.assertEqual(recent_records[0].start_layer, 16)
        self.assertEqual(recent_records[0].end_layer, 24)
        self.assertEqual(recent_records[1].start_layer, 8)
        self.assertEqual(recent_records[1].end_layer, 16)

    def test_prefetch_default_chunks_prioritizes_recent_hint_model(self) -> None:
        local_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-default-priority-local")
        remote_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-default-priority-remote")
        tokenizer_a = b'{"vocab":["older-qwen"]}'
        tokenizer_b = b'{"vocab":["hinted-gemma"]}'
        tokenizer_a_sha = hashlib.sha256(tokenizer_a).hexdigest()
        tokenizer_b_sha = hashlib.sha256(tokenizer_b).hexdigest()
        manifest_a = ModelPackageManifest(
            catalog_id="demo-priority-a",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            version="v1",
            backend="llama_cpp",
            package_kind=ModelPackageKind.PUBLIC_SHARED,
            total_size_bytes=len(tokenizer_a),
            source_repo_id=None,
            files=[
                SourceArtifact(
                    artifact_id="tokenizer",
                    relative_path="tokenizer-a.json",
                    size_bytes=len(tokenizer_a),
                    sha256_hex=tokenizer_a_sha,
                ),
            ],
            chunks=[
                ModelChunk(
                    chunk_id=make_chunk_id(
                        "tokenizer",
                        offset_bytes=0,
                        size_bytes=len(tokenizer_a),
                        sha256_hex=tokenizer_a_sha,
                    ),
                    artifact_id="tokenizer",
                    kind=ModelChunkKind.TOKENIZER,
                    offset_bytes=0,
                    size_bytes=len(tokenizer_a),
                    sha256_hex=tokenizer_a_sha,
                    required_by_default=True,
                ),
            ],
        )
        manifest_b = ModelPackageManifest(
            catalog_id="demo-priority-b",
            model_id="google/gemma-2b-it-GGUF",
            version="v1",
            backend="llama_cpp",
            package_kind=ModelPackageKind.PUBLIC_SHARED,
            total_size_bytes=len(tokenizer_b),
            source_repo_id=None,
            files=[
                SourceArtifact(
                    artifact_id="tokenizer",
                    relative_path="tokenizer-b.json",
                    size_bytes=len(tokenizer_b),
                    sha256_hex=tokenizer_b_sha,
                ),
            ],
            chunks=[
                ModelChunk(
                    chunk_id=make_chunk_id(
                        "tokenizer",
                        offset_bytes=0,
                        size_bytes=len(tokenizer_b),
                        sha256_hex=tokenizer_b_sha,
                    ),
                    artifact_id="tokenizer",
                    kind=ModelChunkKind.TOKENIZER,
                    offset_bytes=0,
                    size_bytes=len(tokenizer_b),
                    sha256_hex=tokenizer_b_sha,
                    required_by_default=True,
                ),
            ],
        )
        save_model_package_manifest(manifest_a, local_policy)
        save_model_package_manifest(manifest_b, local_policy)
        chunk_a = manifest_a.chunks[0]
        chunk_b = manifest_b.chunks[0]
        put_cached_chunk(
            catalog_id=manifest_a.catalog_id,
            version=manifest_a.version,
            chunk_id=chunk_a.chunk_id,
            sha256_hex=chunk_a.sha256_hex,
            content=tokenizer_a,
            policy=remote_policy,
        )
        put_cached_chunk(
            catalog_id=manifest_b.catalog_id,
            version=manifest_b.version,
            chunk_id=chunk_b.chunk_id,
            sha256_hex=chunk_b.sha256_hex,
            content=tokenizer_b,
            policy=remote_policy,
        )
        save_chunk_source_binding(
            ChunkFetchSourceKind.PEER_CACHE,
            "peer-a",
            data_root_path=data_root(remote_policy),
            policy=local_policy,
        )
        import_chunk_inventory_payload(
            ChunkInventoryPayload(
                source_id="peer-a",
                source_kind=ChunkInventorySourceKind.PEER_CACHE,
                published_at=(datetime.now(tz=UTC) - timedelta(minutes=1)).replace(microsecond=0).isoformat(),
                records=(
                    ChunkInventoryRecord(
                        catalog_id=manifest_a.catalog_id,
                        version=manifest_a.version,
                        chunk_ids=(chunk_a.chunk_id,),
                        chunk_count=1,
                        total_bytes=chunk_a.size_bytes,
                    ),
                ),
            ),
            local_policy,
        )
        import_chunk_inventory_payload(
            ChunkInventoryPayload(
                source_id="peer-b",
                source_kind=ChunkInventorySourceKind.PEER_CACHE,
                published_at=(datetime.now(tz=UTC) - timedelta(minutes=2)).replace(microsecond=0).isoformat(),
                records=(
                    ChunkInventoryRecord(
                        catalog_id=manifest_b.catalog_id,
                        version=manifest_b.version,
                        chunk_ids=(chunk_b.chunk_id,),
                        chunk_count=1,
                        total_bytes=chunk_b.size_bytes,
                    ),
                ),
            ),
            local_policy,
        )
        save_chunk_source_binding(
            ChunkFetchSourceKind.PEER_CACHE,
            "peer-b",
            data_root_path=data_root(remote_policy),
            policy=local_policy,
        )
        remember_recent_shard_hints(
            "node-a",
            [
                {
                    "model_id": manifest_b.model_id,
                    "start_layer": 0,
                    "end_layer": 1,
                    "device_rank": 0,
                    "world_size": 1,
                }
            ],
            policy=local_policy,
        )

        result = prefetch_default_chunks_from_fresh_inventories(
            node_id="node-a",
            max_manifests=1,
            max_tasks=1,
            fetch_policy=ChunkFetchPolicy(max_inventory_age_seconds=300),
            policy=local_policy,
        )

        self.assertEqual(result.manifests_considered, 1)
        self.assertEqual(result.manifests_prefetched, 1)
        self.assertIsNone(get_cached_chunk_record(chunk_a.chunk_id, local_policy))
        self.assertIsNotNone(get_cached_chunk_record(chunk_b.chunk_id, local_policy))

    def test_prefetch_bootstrap_chunks_prioritizes_recent_hint_model(self) -> None:
        local_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-bootstrap-priority-local")
        remote_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-bootstrap-priority-remote")
        older_weight = b"AAAA"
        hinted_weight = b"BBBB"
        older_sha = hashlib.sha256(older_weight).hexdigest()
        hinted_sha = hashlib.sha256(hinted_weight).hexdigest()
        manifest_a = ModelPackageManifest(
            catalog_id="demo-bootstrap-priority-a",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            version="v1",
            backend="llama_cpp",
            package_kind=ModelPackageKind.PUBLIC_SHARED,
            total_size_bytes=len(older_weight),
            source_repo_id=None,
            files=[
                SourceArtifact(
                    artifact_id="gguf-main",
                    relative_path="model-a.gguf",
                    size_bytes=len(older_weight),
                    sha256_hex=older_sha,
                ),
            ],
            chunks=[
                ModelChunk(
                    chunk_id=make_chunk_id(
                        "gguf-main",
                        offset_bytes=0,
                        size_bytes=len(older_weight),
                        sha256_hex=older_sha,
                    ),
                    artifact_id="gguf-main",
                    kind=ModelChunkKind.WEIGHTS,
                    offset_bytes=0,
                    size_bytes=len(older_weight),
                    sha256_hex=older_sha,
                    layer_start=0,
                    layer_end=1,
                ),
            ],
        )
        manifest_b = ModelPackageManifest(
            catalog_id="demo-bootstrap-priority-b",
            model_id="google/gemma-2b-it-GGUF",
            version="v1",
            backend="llama_cpp",
            package_kind=ModelPackageKind.PUBLIC_SHARED,
            total_size_bytes=len(hinted_weight),
            source_repo_id=None,
            files=[
                SourceArtifact(
                    artifact_id="gguf-main",
                    relative_path="model-b.gguf",
                    size_bytes=len(hinted_weight),
                    sha256_hex=hinted_sha,
                ),
            ],
            chunks=[
                ModelChunk(
                    chunk_id=make_chunk_id(
                        "gguf-main",
                        offset_bytes=0,
                        size_bytes=len(hinted_weight),
                        sha256_hex=hinted_sha,
                    ),
                    artifact_id="gguf-main",
                    kind=ModelChunkKind.WEIGHTS,
                    offset_bytes=0,
                    size_bytes=len(hinted_weight),
                    sha256_hex=hinted_sha,
                    layer_start=0,
                    layer_end=1,
                ),
            ],
        )
        save_model_package_manifest(manifest_a, local_policy)
        save_model_package_manifest(manifest_b, local_policy)
        chunk_a = manifest_a.chunks[0]
        chunk_b = manifest_b.chunks[0]
        put_cached_chunk(
            catalog_id=manifest_a.catalog_id,
            version=manifest_a.version,
            chunk_id=chunk_a.chunk_id,
            sha256_hex=chunk_a.sha256_hex,
            content=older_weight,
            policy=remote_policy,
        )
        put_cached_chunk(
            catalog_id=manifest_b.catalog_id,
            version=manifest_b.version,
            chunk_id=chunk_b.chunk_id,
            sha256_hex=chunk_b.sha256_hex,
            content=hinted_weight,
            policy=remote_policy,
        )
        save_chunk_source_binding(
            ChunkFetchSourceKind.PEER_CACHE,
            "peer-a",
            data_root_path=data_root(remote_policy),
            policy=local_policy,
        )
        save_chunk_source_binding(
            ChunkFetchSourceKind.PEER_CACHE,
            "peer-b",
            data_root_path=data_root(remote_policy),
            policy=local_policy,
        )
        import_chunk_inventory_payload(
            ChunkInventoryPayload(
                source_id="peer-a",
                source_kind=ChunkInventorySourceKind.PEER_CACHE,
                published_at=(datetime.now(tz=UTC) - timedelta(minutes=1)).replace(microsecond=0).isoformat(),
                records=(
                    ChunkInventoryRecord(
                        catalog_id=manifest_a.catalog_id,
                        version=manifest_a.version,
                        chunk_ids=(chunk_a.chunk_id,),
                        chunk_count=1,
                        total_bytes=chunk_a.size_bytes,
                    ),
                ),
            ),
            local_policy,
        )
        import_chunk_inventory_payload(
            ChunkInventoryPayload(
                source_id="peer-b",
                source_kind=ChunkInventorySourceKind.PEER_CACHE,
                published_at=(datetime.now(tz=UTC) - timedelta(minutes=2)).replace(microsecond=0).isoformat(),
                records=(
                    ChunkInventoryRecord(
                        catalog_id=manifest_b.catalog_id,
                        version=manifest_b.version,
                        chunk_ids=(chunk_b.chunk_id,),
                        chunk_count=1,
                        total_bytes=chunk_b.size_bytes,
                    ),
                ),
            ),
            local_policy,
        )
        remember_recent_shard_hints(
            "node-a",
            [
                {
                    "model_id": manifest_b.model_id,
                    "start_layer": 0,
                    "end_layer": 1,
                    "device_rank": 0,
                    "world_size": 1,
                }
            ],
            policy=local_policy,
        )

        result = prefetch_bootstrap_chunks_from_fresh_inventories(
            node_id="node-a",
            max_manifests=1,
            max_tasks=1,
            fetch_policy=ChunkFetchPolicy(
                max_inventory_age_seconds=300,
                warm_prefetch_weight_chunk_count_per_manifest=1,
                warm_prefetch_max_weight_bytes_per_manifest=8,
            ),
            policy=local_policy,
        )

        self.assertEqual(result.manifests_considered, 1)
        self.assertEqual(result.manifests_prefetched, 1)
        self.assertIsNone(get_cached_chunk_record(chunk_a.chunk_id, local_policy))
        self.assertIsNotNone(get_cached_chunk_record(chunk_b.chunk_id, local_policy))

    def test_prefetch_recent_shard_hints_uses_latest_remembered_range(self) -> None:
        local_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-recent-hint-prefetch-local")
        remote_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-recent-hint-prefetch-remote")
        gguf_payload = b"abcdefghij"
        gguf_path = self._write_temp_file("recent-hint-prefetch.gguf", gguf_payload)
        manifest = build_gguf_model_package_manifest(
            catalog_id="demo-recent-hint-prefetch",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            version="v1",
            gguf_path=gguf_path,
            total_layers=5,
            chunk_size_policy=ChunkSizePolicy.ADAPTIVE,
            min_chunk_bytes=2,
            max_chunk_bytes=2,
            source_repo_id=None,
        )
        save_model_package_manifest(manifest, local_policy)
        weight_chunks = [chunk for chunk in manifest.chunks if chunk.kind == ModelChunkKind.WEIGHTS]
        self.assertGreaterEqual(len(weight_chunks), 3)
        for chunk in weight_chunks:
            payload = gguf_payload[chunk.offset_bytes : chunk.offset_bytes + chunk.size_bytes]
            put_cached_chunk(
                catalog_id=manifest.catalog_id,
                version=manifest.version,
                chunk_id=chunk.chunk_id,
                sha256_hex=chunk.sha256_hex,
                content=payload,
                policy=remote_policy,
            )
        save_chunk_source_binding(
            ChunkFetchSourceKind.PEER_CACHE,
            "peer-a",
            data_root_path=data_root(remote_policy),
            policy=local_policy,
        )
        import_chunk_inventory_payload(
            ChunkInventoryPayload(
                source_id="peer-a",
                source_kind=ChunkInventorySourceKind.PEER_CACHE,
                published_at=_fresh_published_at(),
                records=(
                    ChunkInventoryRecord(
                        catalog_id=manifest.catalog_id,
                        version=manifest.version,
                        chunk_ids=tuple(chunk.chunk_id for chunk in weight_chunks),
                        chunk_count=len(weight_chunks),
                        total_bytes=sum(chunk.size_bytes for chunk in weight_chunks),
                    ),
                ),
            ),
            local_policy,
        )
        base_now = datetime.now(tz=UTC).replace(microsecond=0)
        remember_recent_shard_hints(
            "node-a",
            [
                {
                    "model_id": manifest.model_id,
                    "start_layer": 0,
                    "end_layer": 2,
                    "device_rank": 0,
                    "world_size": 2,
                }
            ],
            policy=local_policy,
            now=base_now - timedelta(minutes=10),
        )
        remember_recent_shard_hints(
            "node-a",
            [
                {
                    "model_id": manifest.model_id,
                    "start_layer": 2,
                    "end_layer": 4,
                    "device_rank": 1,
                    "world_size": 2,
                }
            ],
            policy=local_policy,
            now=base_now,
        )

        result = prefetch_recent_shard_hints(
            "node-a",
            max_hints=1,
            fetch_policy=ChunkFetchPolicy(
                max_inventory_age_seconds=300,
                hint_prefetch_weight_chunk_count_per_manifest=1,
                hint_prefetch_max_weight_bytes_per_manifest=8,
                recent_hint_prefetch_max_hints=1,
            ),
            policy=local_policy,
        )

        self.assertEqual(result.manifests_considered, 1)
        self.assertEqual(result.manifests_prefetched, 1)
        self.assertEqual(result.queued_tasks, 1)
        self.assertEqual(result.processed_tasks, 1)
        middle_chunk = next(
            chunk for chunk in weight_chunks if chunk.overlaps_layer_range(2, 4)
        )
        first_chunk = next(
            chunk for chunk in weight_chunks if chunk.overlaps_layer_range(0, 2)
        )
        self.assertIsNotNone(get_cached_chunk_record(middle_chunk.chunk_id, local_policy))
        self.assertIsNone(get_cached_chunk_record(first_chunk.chunk_id, local_policy))

    def test_prefetch_hinted_bootstrap_chunks_prefers_nearby_weight_chunk(self) -> None:
        local_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-hinted-prefetch-local")
        remote_policy = WalletPolicy(wallet_data_dirname=".tmp-model-distribution-hinted-prefetch-remote")
        gguf_payload = b"abcdefghij"
        gguf_path = self._write_temp_file("hinted-prefetch.gguf", gguf_payload)
        manifest = build_gguf_model_package_manifest(
            catalog_id="demo-hinted-prefetch",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            version="v1",
            gguf_path=gguf_path,
            total_layers=5,
            chunk_size_policy=ChunkSizePolicy.ADAPTIVE,
            min_chunk_bytes=2,
            max_chunk_bytes=2,
            source_repo_id=None,
        )
        save_model_package_manifest(manifest, local_policy)
        weight_chunks = [chunk for chunk in manifest.chunks if chunk.kind == ModelChunkKind.WEIGHTS]
        self.assertGreaterEqual(len(weight_chunks), 3)
        for chunk in weight_chunks:
            payload = gguf_payload[chunk.offset_bytes : chunk.offset_bytes + chunk.size_bytes]
            put_cached_chunk(
                catalog_id=manifest.catalog_id,
                version=manifest.version,
                chunk_id=chunk.chunk_id,
                sha256_hex=chunk.sha256_hex,
                content=payload,
                policy=remote_policy,
            )
        save_chunk_source_binding(
            ChunkFetchSourceKind.PEER_CACHE,
            "peer-a",
            data_root_path=data_root(remote_policy),
            policy=local_policy,
        )
        import_chunk_inventory_payload(
            ChunkInventoryPayload(
                source_id="peer-a",
                source_kind=ChunkInventorySourceKind.PEER_CACHE,
                published_at=_fresh_published_at(),
                records=(
                    ChunkInventoryRecord(
                        catalog_id=manifest.catalog_id,
                        version=manifest.version,
                        chunk_ids=tuple(chunk.chunk_id for chunk in weight_chunks),
                        chunk_count=len(weight_chunks),
                        total_bytes=sum(chunk.size_bytes for chunk in weight_chunks),
                    ),
                ),
            ),
            local_policy,
        )

        result = prefetch_hinted_bootstrap_chunks(
            [
                {
                    "model_id": manifest.model_id,
                    "start_layer": 2,
                    "end_layer": 4,
                    "device_rank": 1,
                    "world_size": 3,
                    "node_id": "node-a",
                }
            ],
            fetch_policy=ChunkFetchPolicy(
                max_inventory_age_seconds=300,
                hint_prefetch_weight_chunk_count_per_manifest=1,
                hint_prefetch_max_weight_bytes_per_manifest=8,
            ),
            policy=local_policy,
        )

        self.assertEqual(result.manifests_considered, 1)
        self.assertEqual(result.manifests_prefetched, 1)
        self.assertEqual(result.queued_tasks, 1)
        self.assertEqual(result.processed_tasks, 1)
        middle_chunk = next(
            chunk for chunk in weight_chunks if chunk.overlaps_layer_range(2, 4)
        )
        self.assertIsNotNone(get_cached_chunk_record(middle_chunk.chunk_id, local_policy))


if __name__ == "__main__":
    unittest.main()

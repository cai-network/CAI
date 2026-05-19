# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlparse
from urllib.request import Request, urlopen

from .gguf_shard_policy import gguf_shard_compatibility
from .model import ChunkCachePolicy, ChunkFetchPolicy, ChunkStorageAccountingPolicy, WalletPolicy
from .wallet import data_root


CAI_MODEL_PACKAGE_FORMAT_VERSION = 1
DEFAULT_CHUNK_TARGET_BYTES = 256 * 1024 * 1024
DEFAULT_MIN_CHUNK_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_CHUNK_BYTES = 512 * 1024 * 1024
MATERIALIZED_ASSIGNMENT_VALIDATION_READ_BYTES = 8 * 1024 * 1024
FSCTL_SET_SPARSE = 0x000900C4
GGUF_MAGIC = b"GGUF"
GGUF_DEFAULT_ALIGNMENT = 32
GGUF_METADATA_TYPE_UINT8 = 0
GGUF_METADATA_TYPE_INT8 = 1
GGUF_METADATA_TYPE_UINT16 = 2
GGUF_METADATA_TYPE_INT16 = 3
GGUF_METADATA_TYPE_UINT32 = 4
GGUF_METADATA_TYPE_INT32 = 5
GGUF_METADATA_TYPE_FLOAT32 = 6
GGUF_METADATA_TYPE_BOOL = 7
GGUF_METADATA_TYPE_STRING = 8
GGUF_METADATA_TYPE_ARRAY = 9
GGUF_METADATA_TYPE_UINT64 = 10
GGUF_METADATA_TYPE_INT64 = 11
GGUF_METADATA_TYPE_FLOAT64 = 12


class ChunkSizePolicy(StrEnum):
    SMALL = "small"
    BALANCED = "balanced"
    LARGE = "large"
    ADAPTIVE = "adaptive"


class ModelPackageKind(StrEnum):
    PUBLIC_SHARED = "public_shared"
    PRIVATE_CURATED = "private_curated"


class ModelChunkKind(StrEnum):
    WEIGHTS = "weights"
    TOKENIZER = "tokenizer"
    CONFIG = "config"
    METADATA = "metadata"


class ModelManifestValidationError(ValueError):
    """Raised when a CAI model manifest is structurally invalid."""


class ChunkCacheClass(StrEnum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


class ChunkLeaseStatus(StrEnum):
    NONE = "none"
    ACTIVE = "active"
    EXPIRED = "expired"


class ChunkFetchSourceKind(StrEnum):
    PEER_CACHE = "peer_cache"
    STORAGE_SEED = "storage_seed"
    ORIGIN = "origin"


class ChunkInventorySourceKind(StrEnum):
    LOCAL_CACHE = "local_cache"
    PEER_CACHE = "peer_cache"
    STORAGE_SEED = "storage_seed"


class ChunkDownloadTaskStatus(StrEnum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class SourceArtifact:
    artifact_id: str
    relative_path: str
    size_bytes: int
    sha256_hex: str | None = None
    media_type: str | None = None
    source_repo_id: str | None = None
    source_revision: str | None = None


@dataclass(frozen=True)
class ModelChunk:
    chunk_id: str
    artifact_id: str
    kind: str = ModelChunkKind.WEIGHTS
    offset_bytes: int = 0
    size_bytes: int = 0
    sha256_hex: str = ""
    layer_start: int | None = None
    layer_end: int | None = None
    tensor_names: list[str] = field(default_factory=list)
    required_by_default: bool = False
    compression: str | None = None
    encrypted_at_rest: bool = False

    def overlaps_layer_range(self, start_layer: int, end_layer: int) -> bool:
        if self.layer_start is None or self.layer_end is None:
            return False
        return self.layer_start < end_layer and self.layer_end > start_layer


@dataclass(frozen=True)
class ChunkCoverage:
    required_chunk_ids: tuple[str, ...]
    present_chunk_ids: tuple[str, ...]
    missing_chunk_ids: tuple[str, ...]
    required_bytes: int
    present_bytes: int

    @property
    def ready(self) -> bool:
        return len(self.missing_chunk_ids) == 0

    @property
    def coverage_ratio(self) -> float:
        if self.required_bytes <= 0:
            return 1.0
        return min(self.present_bytes / self.required_bytes, 1.0)


@dataclass
class CachedChunkRecord:
    chunk_id: str
    catalog_id: str
    version: str
    relative_path: str
    size_bytes: int
    sha256_hex: str
    stored_at: str
    last_used_at: str
    use_count: int = 0
    pinned: bool = False
    cache_class: str = ChunkCacheClass.WARM
    lease_status: str = ChunkLeaseStatus.NONE
    lease_expires_at: str | None = None


@dataclass(frozen=True)
class ChunkStoreStats:
    chunk_count: int
    total_bytes: int
    pinned_chunk_count: int
    pinned_bytes: int
    hot_chunk_count: int
    warm_chunk_count: int
    cold_chunk_count: int


@dataclass(frozen=True)
class ChunkStoreSnapshot:
    records: tuple[CachedChunkRecord, ...]
    stats: ChunkStoreStats


@dataclass(frozen=True)
class ChunkStoreEvictionResult:
    before: ChunkStoreSnapshot
    after: ChunkStoreSnapshot
    evicted_chunk_ids: tuple[str, ...]
    evicted_bytes: int

    @property
    def changed(self) -> bool:
        return bool(self.evicted_chunk_ids)


@dataclass
class ChunkStorageAccountingRecord:
    accounting_id: str
    created_at: str
    node_id: str
    catalog_id: str
    version: str
    chunk_id: str
    size_bytes: int
    stored_at: str
    last_used_at: str
    accounted_from: str
    accounted_until: str
    accounted_seconds: int
    byte_seconds: int
    cache_class: str
    lease_status: str
    pinned: bool = False


@dataclass(frozen=True)
class ChunkStorageAccountingSummary:
    node_id: str
    record_count: int
    total_size_bytes: int
    total_accounted_seconds: int
    total_byte_seconds: int
    period_started_at: str | None
    period_ended_at: str | None


@dataclass(frozen=True)
class ChunkStorageAccountingResult:
    summary: ChunkStorageAccountingSummary
    records: tuple[ChunkStorageAccountingRecord, ...]


@dataclass(frozen=True)
class ModelShardAssignment:
    start_layer: int
    end_layer: int
    device_rank: int = 0
    world_size: int = 1
    node_id: str | None = None
    runner_id: str | None = None


@dataclass(frozen=True)
class AssignmentChunkPlan:
    assignment: ModelShardAssignment
    coverage: ChunkCoverage
    estimated_fetch_bytes: int

    @property
    def ready(self) -> bool:
        return self.coverage.ready


@dataclass(frozen=True)
class ChunkFetchSource:
    kind: str
    source_id: str
    locator: str | None = None


@dataclass(frozen=True)
class ChunkFetchRequest:
    chunk_id: str
    artifact_id: str
    size_bytes: int
    sha256_hex: str
    layer_start: int | None
    layer_end: int | None
    sources: tuple[ChunkFetchSource, ...]


@dataclass(frozen=True)
class AssignmentFetchPlan:
    assignment: ModelShardAssignment
    coverage: ChunkCoverage
    fetch_requests: tuple[ChunkFetchRequest, ...]
    estimated_fetch_bytes: int

    @property
    def ready(self) -> bool:
        return len(self.fetch_requests) == 0


@dataclass(frozen=True)
class AssignmentEnsureReadyResult:
    manifest: "ModelPackageManifest"
    assignment: ModelShardAssignment
    initial_plan: AssignmentChunkPlan
    fetch_plan: AssignmentFetchPlan | None
    queued_tasks: tuple[ChunkDownloadTask, ...]
    processed_tasks: tuple[ChunkDownloadTask, ...]
    final_plan: AssignmentChunkPlan

    @property
    def ready(self) -> bool:
        return self.final_plan.ready


@dataclass(frozen=True)
class MaterializedArtifactResult:
    catalog_id: str
    version: str
    artifact_id: str
    output_path: str
    size_bytes: int
    sha256_hex: str


@dataclass(frozen=True)
class GgufTensorSpan:
    name: str
    start_offset: int
    end_offset: int
    layer_index: int | None = None


@dataclass(frozen=True)
class GgufTensorLayout:
    data_offset: int
    tensor_count: int
    tensors: tuple[GgufTensorSpan, ...]


@dataclass(frozen=True)
class GgufModelMetadata:
    architecture: str | None
    total_layers: int | None
    hidden_size: int | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ChunkInventoryRecord:
    catalog_id: str
    version: str
    chunk_ids: tuple[str, ...]
    chunk_count: int
    total_bytes: int


@dataclass(frozen=True)
class ChunkInventoryPayload:
    source_id: str
    source_kind: str
    published_at: str
    records: tuple[ChunkInventoryRecord, ...]
    endpoint_base_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_kind": self.source_kind,
            "published_at": self.published_at,
            "endpoint_base_url": self.endpoint_base_url,
            "records": [
                {
                    "catalog_id": record.catalog_id,
                    "version": record.version,
                    "chunk_ids": list(record.chunk_ids),
                    "chunk_count": record.chunk_count,
                    "total_bytes": record.total_bytes,
                }
                for record in self.records
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChunkInventoryPayload":
        return cls(
            source_id=str(payload["source_id"]),
            source_kind=str(payload["source_kind"]),
            published_at=str(payload["published_at"]),
            endpoint_base_url=(
                str(payload.get("endpoint_base_url")).strip()
                if payload.get("endpoint_base_url")
                else None
            ),
            records=tuple(
                ChunkInventoryRecord(
                    catalog_id=str(item["catalog_id"]),
                    version=str(item["version"]),
                    chunk_ids=tuple(str(chunk_id) for chunk_id in item.get("chunk_ids", [])),
                    chunk_count=int(item["chunk_count"]),
                    total_bytes=int(item["total_bytes"]),
                )
                for item in payload.get("records", [])
            ),
        )

    def chunk_ids_by_manifest(self, catalog_id: str, version: str) -> set[str]:
        for record in self.records:
            if record.catalog_id == catalog_id and record.version == version:
                return set(record.chunk_ids)
        return set()


@dataclass(frozen=True)
class ChunkInventorySyncResult:
    attempted_peers: int
    successful_peers: int
    imported_payloads: int
    pruned_payloads: int
    peer_urls: list[str]
    failed_peers: int = 0
    failed_peer_urls: list[str] = field(default_factory=list)
    peer_errors: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class DefaultChunkPrefetchResult:
    manifests_considered: int
    manifests_prefetched: int
    queued_tasks: int
    processed_tasks: int


@dataclass(frozen=True)
class BootstrapChunkPrefetchResult:
    manifests_considered: int
    manifests_prefetched: int
    queued_tasks: int
    processed_tasks: int


@dataclass(frozen=True)
class HintedChunkPrefetchResult:
    manifests_considered: int
    manifests_prefetched: int
    queued_tasks: int
    processed_tasks: int


@dataclass
class RecentShardHintRecord:
    model_id: str
    start_layer: int
    end_layer: int
    device_rank: int = 0
    world_size: int = 1
    first_seen_at: str = ""
    last_seen_at: str = ""
    use_count: int = 0


@dataclass(frozen=True)
class RecentShardHintUpdateResult:
    hints_received: int
    records_upserted: int
    records_pruned: int
    stored_records: int


@dataclass
class ChunkDownloadTask:
    task_id: str
    catalog_id: str
    version: str
    chunk_id: str
    artifact_id: str
    size_bytes: int
    sha256_hex: str
    status: str
    created_at: str
    updated_at: str
    assignment_start_layer: int
    assignment_end_layer: int
    assignment_device_rank: int = 0
    assignment_world_size: int = 1
    node_id: str | None = None
    sources: tuple[ChunkFetchSource, ...] = ()
    selected_source_kind: str | None = None
    selected_source_id: str | None = None
    attempt_count: int = 0
    last_error: str | None = None


@dataclass(frozen=True)
class ChunkDownloadQueueStats:
    task_count: int
    queued_count: int
    in_progress_count: int
    completed_count: int
    failed_count: int
    total_bytes: int
    queued_bytes: int
    completed_bytes: int


@dataclass(frozen=True)
class ChunkDownloadQueueSnapshot:
    tasks: tuple[ChunkDownloadTask, ...]
    stats: ChunkDownloadQueueStats


@dataclass(frozen=True)
class ChunkSourceHealthRecord:
    source_kind: str
    source_id: str
    locator: str | None = None
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    last_success_at: str | None = None
    last_failure_at: str | None = None
    cooldown_until: str | None = None
    last_error: str | None = None


@dataclass(frozen=True)
class LocalArtifactBinding:
    artifact_id: str
    local_path: str
    updated_at: str


@dataclass(frozen=True)
class LocalArtifactBindings:
    catalog_id: str
    version: str
    bindings: tuple[LocalArtifactBinding, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_id": self.catalog_id,
            "version": self.version,
            "bindings": [asdict(binding) for binding in self.bindings],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LocalArtifactBindings":
        return cls(
            catalog_id=str(payload["catalog_id"]),
            version=str(payload["version"]),
            bindings=tuple(
                LocalArtifactBinding(**item) for item in payload.get("bindings", [])
            ),
        )


@dataclass(frozen=True)
class ChunkSourceBinding:
    source_kind: str
    source_id: str
    data_root_path: str
    updated_at: str


@dataclass(frozen=True)
class ChunkSourceBindings:
    bindings: tuple[ChunkSourceBinding, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bindings": [asdict(binding) for binding in self.bindings],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChunkSourceBindings":
        return cls(
            bindings=tuple(
                ChunkSourceBinding(**item) for item in payload.get("bindings", [])
            ),
        )


@dataclass
class ModelPackageManifest:
    catalog_id: str
    model_id: str
    version: str
    backend: str
    manifest_version: int = CAI_MODEL_PACKAGE_FORMAT_VERSION
    package_kind: str = ModelPackageKind.PUBLIC_SHARED
    chunk_size_policy: str = ChunkSizePolicy.BALANCED
    total_size_bytes: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=UTC).isoformat()
    )
    source_repo_id: str | None = None
    source_revision: str | None = "main"
    preferred_filename: str | None = None
    family: str = ""
    quantization: str = ""
    files: list[SourceArtifact] = field(default_factory=list)
    chunks: list[ModelChunk] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelPackageManifest":
        files = [SourceArtifact(**item) for item in payload.get("files", [])]
        chunks = [ModelChunk(**item) for item in payload.get("chunks", [])]
        base_payload = dict(payload)
        base_payload["files"] = files
        base_payload["chunks"] = chunks
        manifest = cls(**base_payload)
        validate_model_package_manifest(manifest)
        return manifest

    def required_chunks_for_layers(
        self,
        start_layer: int,
        end_layer: int,
        *,
        include_default_chunks: bool = True,
    ) -> list[ModelChunk]:
        if start_layer < 0 or end_layer <= start_layer:
            raise ValueError("Layer range must satisfy 0 <= start < end.")

        required_by_id: dict[str, ModelChunk] = {}
        for chunk in self.chunks:
            if include_default_chunks and chunk.required_by_default:
                required_by_id.setdefault(chunk.chunk_id, chunk)
                continue
            if chunk.kind != ModelChunkKind.WEIGHTS:
                if include_default_chunks:
                    required_by_id.setdefault(chunk.chunk_id, chunk)
                continue
            if chunk.overlaps_layer_range(start_layer, end_layer):
                required_by_id.setdefault(chunk.chunk_id, chunk)

        return list(required_by_id.values())

    def compute_chunk_coverage(
        self,
        present_chunk_ids: set[str] | list[str] | tuple[str, ...],
        *,
        start_layer: int,
        end_layer: int,
        include_default_chunks: bool = True,
    ) -> ChunkCoverage:
        present_set = {str(item) for item in present_chunk_ids}
        required_chunks = self.required_chunks_for_layers(
            start_layer,
            end_layer,
            include_default_chunks=include_default_chunks,
        )
        required_ids = tuple(chunk.chunk_id for chunk in required_chunks)
        missing_ids = tuple(
            chunk.chunk_id for chunk in required_chunks if chunk.chunk_id not in present_set
        )
        present_ids = tuple(
            chunk.chunk_id for chunk in required_chunks if chunk.chunk_id in present_set
        )
        required_bytes = sum(chunk.size_bytes for chunk in required_chunks)
        present_bytes = sum(
            chunk.size_bytes for chunk in required_chunks if chunk.chunk_id in present_set
        )
        return ChunkCoverage(
            required_chunk_ids=required_ids,
            present_chunk_ids=present_ids,
            missing_chunk_ids=missing_ids,
            required_bytes=required_bytes,
            present_bytes=present_bytes,
        )

    def default_chunks(self) -> list[ModelChunk]:
        required_by_id: dict[str, ModelChunk] = {}
        for chunk in self.chunks:
            if chunk.required_by_default or chunk.kind != ModelChunkKind.WEIGHTS:
                required_by_id.setdefault(chunk.chunk_id, chunk)
        return list(required_by_id.values())

    def compute_default_chunk_coverage(
        self,
        present_chunk_ids: set[str] | list[str] | tuple[str, ...],
    ) -> ChunkCoverage:
        present_set = {str(item) for item in present_chunk_ids}
        required_chunks = self.default_chunks()
        required_ids = tuple(chunk.chunk_id for chunk in required_chunks)
        missing_ids = tuple(
            chunk.chunk_id for chunk in required_chunks if chunk.chunk_id not in present_set
        )
        present_ids = tuple(
            chunk.chunk_id for chunk in required_chunks if chunk.chunk_id in present_set
        )
        required_bytes = sum(chunk.size_bytes for chunk in required_chunks)
        present_bytes = sum(
            chunk.size_bytes for chunk in required_chunks if chunk.chunk_id in present_set
        )
        return ChunkCoverage(
            required_chunk_ids=required_ids,
            present_chunk_ids=present_ids,
            missing_chunk_ids=missing_ids,
            required_bytes=required_bytes,
            present_bytes=present_bytes,
        )

    def bootstrap_prefetch_chunks(
        self,
        *,
        max_weight_chunks: int = 1,
        max_weight_bytes: int | None = None,
        include_default_chunks: bool = True,
        hint_start_layer: int | None = None,
        hint_end_layer: int | None = None,
    ) -> list[ModelChunk]:
        selected_by_id: dict[str, ModelChunk] = {}
        if include_default_chunks:
            for chunk in self.default_chunks():
                selected_by_id.setdefault(chunk.chunk_id, chunk)

        if max_weight_chunks <= 0:
            return list(selected_by_id.values())

        weight_budget = None if max_weight_bytes is None else max(0, int(max_weight_bytes))
        weight_chunks = sorted(
            (chunk for chunk in self.chunks if chunk.kind == ModelChunkKind.WEIGHTS),
            key=lambda chunk: (
                _chunk_hint_distance(
                    chunk,
                    hint_start_layer=hint_start_layer,
                    hint_end_layer=hint_end_layer,
                ),
                0
                if _chunk_overlaps_hint_range(
                    chunk,
                    hint_start_layer=hint_start_layer,
                    hint_end_layer=hint_end_layer,
                )
                else 1,
                chunk.layer_start if chunk.layer_start is not None else 10**9,
                chunk.offset_bytes,
                chunk.chunk_id,
            ),
        )
        selected_weight_chunks = 0
        selected_weight_bytes = 0
        for chunk in weight_chunks:
            if selected_weight_chunks >= max_weight_chunks:
                break
            if weight_budget is not None and chunk.size_bytes + selected_weight_bytes > weight_budget:
                continue
            selected_by_id.setdefault(chunk.chunk_id, chunk)
            selected_weight_chunks += 1
            selected_weight_bytes += chunk.size_bytes
        return list(selected_by_id.values())

    def compute_bootstrap_chunk_coverage(
        self,
        present_chunk_ids: set[str] | list[str] | tuple[str, ...],
        *,
        max_weight_chunks: int = 1,
        max_weight_bytes: int | None = None,
        include_default_chunks: bool = True,
        hint_start_layer: int | None = None,
        hint_end_layer: int | None = None,
    ) -> ChunkCoverage:
        present_set = {str(item) for item in present_chunk_ids}
        required_chunks = self.bootstrap_prefetch_chunks(
            max_weight_chunks=max_weight_chunks,
            max_weight_bytes=max_weight_bytes,
            include_default_chunks=include_default_chunks,
            hint_start_layer=hint_start_layer,
            hint_end_layer=hint_end_layer,
        )
        required_ids = tuple(chunk.chunk_id for chunk in required_chunks)
        missing_ids = tuple(
            chunk.chunk_id for chunk in required_chunks if chunk.chunk_id not in present_set
        )
        present_ids = tuple(
            chunk.chunk_id for chunk in required_chunks if chunk.chunk_id in present_set
        )
        required_bytes = sum(chunk.size_bytes for chunk in required_chunks)
        present_bytes = sum(
            chunk.size_bytes for chunk in required_chunks if chunk.chunk_id in present_set
        )
        return ChunkCoverage(
            required_chunk_ids=required_ids,
            present_chunk_ids=present_ids,
            missing_chunk_ids=missing_ids,
            required_bytes=required_bytes,
            present_bytes=present_bytes,
        )

    def build_assignment_chunk_plan(
        self,
        assignment: ModelShardAssignment,
        present_chunk_ids: set[str] | list[str] | tuple[str, ...],
        *,
        include_default_chunks: bool = True,
    ) -> AssignmentChunkPlan:
        self.validate_assignment_layer_coverage(
            assignment.start_layer,
            assignment.end_layer,
            artifact_id=select_default_materialized_artifact_id(self),
        )
        coverage = self.compute_chunk_coverage(
            present_chunk_ids,
            start_layer=assignment.start_layer,
            end_layer=assignment.end_layer,
            include_default_chunks=include_default_chunks,
        )
        return AssignmentChunkPlan(
            assignment=assignment,
            coverage=coverage,
            estimated_fetch_bytes=max(
                coverage.required_bytes - coverage.present_bytes,
                0,
            ),
        )

    def build_assignment_fetch_plan(
        self,
        assignment: ModelShardAssignment,
        present_chunk_ids: set[str] | list[str] | tuple[str, ...],
        *,
        peer_chunk_inventory: dict[str, set[str] | list[str] | tuple[str, ...]] | None = None,
        seed_chunk_inventory: dict[str, set[str] | list[str] | tuple[str, ...]] | None = None,
        peer_chunk_locators: dict[str, str] | None = None,
        seed_chunk_locators: dict[str, str] | None = None,
        include_default_chunks: bool = True,
    ) -> AssignmentFetchPlan:
        self.validate_assignment_layer_coverage(
            assignment.start_layer,
            assignment.end_layer,
            artifact_id=select_default_materialized_artifact_id(self),
        )
        coverage = self.compute_chunk_coverage(
            present_chunk_ids,
            start_layer=assignment.start_layer,
            end_layer=assignment.end_layer,
            include_default_chunks=include_default_chunks,
        )
        chunk_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        normalized_peer_inventory = {
            source_id: {str(chunk_id) for chunk_id in chunk_ids}
            for source_id, chunk_ids in (peer_chunk_inventory or {}).items()
        }
        normalized_seed_inventory = {
            source_id: {str(chunk_id) for chunk_id in chunk_ids}
            for source_id, chunk_ids in (seed_chunk_inventory or {}).items()
        }

        fetch_requests: list[ChunkFetchRequest] = []
        for chunk_id in coverage.missing_chunk_ids:
            chunk = chunk_by_id[chunk_id]
            sources: list[ChunkFetchSource] = []

            for peer_id, chunk_ids in normalized_peer_inventory.items():
                if chunk_id in chunk_ids:
                    sources.append(
                        ChunkFetchSource(
                            kind=ChunkFetchSourceKind.PEER_CACHE,
                            source_id=peer_id,
                            locator=(peer_chunk_locators or {}).get(peer_id),
                        )
                    )

            for seed_id, chunk_ids in normalized_seed_inventory.items():
                if chunk_id in chunk_ids:
                    sources.append(
                        ChunkFetchSource(
                            kind=ChunkFetchSourceKind.STORAGE_SEED,
                            source_id=seed_id,
                            locator=(seed_chunk_locators or {}).get(seed_id),
                        )
                    )

            if self.source_repo_id:
                origin_locator = self.source_repo_id
                if self.source_revision:
                    origin_locator = f"{origin_locator}@{self.source_revision}"
                sources.append(
                    ChunkFetchSource(
                        kind=ChunkFetchSourceKind.ORIGIN,
                        source_id=self.source_repo_id,
                        locator=origin_locator,
                    )
                )

            fetch_requests.append(
                ChunkFetchRequest(
                    chunk_id=chunk.chunk_id,
                    artifact_id=chunk.artifact_id,
                    size_bytes=chunk.size_bytes,
                    sha256_hex=chunk.sha256_hex,
                    layer_start=chunk.layer_start,
                    layer_end=chunk.layer_end,
                    sources=tuple(sources),
                )
            )

        return AssignmentFetchPlan(
            assignment=assignment,
            coverage=coverage,
            fetch_requests=tuple(fetch_requests),
            estimated_fetch_bytes=sum(request.size_bytes for request in fetch_requests),
        )

    def validate_assignment_layer_coverage(
        self,
        start_layer: int,
        end_layer: int,
        *,
        artifact_id: str | None = None,
    ) -> None:
        if start_layer < 0 or end_layer <= start_layer:
            raise ValueError("Layer range must satisfy 0 <= start < end.")
        clean_artifact_id = str(artifact_id or "").strip()
        if not clean_artifact_id:
            raise ModelManifestValidationError(
                "Assignment layer coverage requires a materializable artifact."
            )
        layer_chunks = sorted(
            (
                chunk
                for chunk in self.chunks
                if chunk.kind == ModelChunkKind.WEIGHTS
                and chunk.artifact_id == clean_artifact_id
                and chunk.layer_start is not None
                and chunk.layer_end is not None
                and chunk.overlaps_layer_range(start_layer, end_layer)
            ),
            key=lambda chunk: (
                int(chunk.layer_start or 0),
                int(chunk.layer_end or 0),
                int(chunk.offset_bytes),
                str(chunk.chunk_id),
            ),
        )
        if not layer_chunks:
            raise ModelManifestValidationError(
                "Assignment layer coverage has no layer-scoped weight chunks "
                f"for {clean_artifact_id} layers {start_layer}..{end_layer}."
            )
        cursor = int(start_layer)
        for chunk in layer_chunks:
            chunk_start = max(int(chunk.layer_start or 0), int(start_layer))
            chunk_end = min(int(chunk.layer_end or 0), int(end_layer))
            if chunk_end <= cursor:
                continue
            if chunk_start > cursor:
                raise ModelManifestValidationError(
                    "Assignment layer coverage has a gap for "
                    f"{clean_artifact_id}: missing layers {cursor}..{chunk_start}."
                )
            cursor = max(cursor, chunk_end)
            if cursor >= int(end_layer):
                return
        raise ModelManifestValidationError(
            "Assignment layer coverage has a gap for "
            f"{clean_artifact_id}: missing layers {cursor}..{end_layer}."
        )

    def build_default_chunk_fetch_plan(
        self,
        assignment: ModelShardAssignment,
        present_chunk_ids: set[str] | list[str] | tuple[str, ...],
        *,
        peer_chunk_inventory: dict[str, set[str] | list[str] | tuple[str, ...]] | None = None,
        seed_chunk_inventory: dict[str, set[str] | list[str] | tuple[str, ...]] | None = None,
        peer_chunk_locators: dict[str, str] | None = None,
        seed_chunk_locators: dict[str, str] | None = None,
    ) -> AssignmentFetchPlan:
        coverage = self.compute_default_chunk_coverage(present_chunk_ids)
        chunk_by_id = {chunk.chunk_id: chunk for chunk in self.default_chunks()}
        normalized_peer_inventory = {
            source_id: {str(chunk_id) for chunk_id in chunk_ids}
            for source_id, chunk_ids in (peer_chunk_inventory or {}).items()
        }
        normalized_seed_inventory = {
            source_id: {str(chunk_id) for chunk_id in chunk_ids}
            for source_id, chunk_ids in (seed_chunk_inventory or {}).items()
        }

        fetch_requests: list[ChunkFetchRequest] = []
        for chunk_id in coverage.missing_chunk_ids:
            chunk = chunk_by_id[chunk_id]
            sources: list[ChunkFetchSource] = []

            for peer_id, chunk_ids in normalized_peer_inventory.items():
                if chunk_id in chunk_ids:
                    sources.append(
                        ChunkFetchSource(
                            kind=ChunkFetchSourceKind.PEER_CACHE,
                            source_id=peer_id,
                            locator=(peer_chunk_locators or {}).get(peer_id),
                        )
                    )

            for seed_id, chunk_ids in normalized_seed_inventory.items():
                if chunk_id in chunk_ids:
                    sources.append(
                        ChunkFetchSource(
                            kind=ChunkFetchSourceKind.STORAGE_SEED,
                            source_id=seed_id,
                            locator=(seed_chunk_locators or {}).get(seed_id),
                        )
                    )

            if self.source_repo_id:
                origin_locator = self.source_repo_id
                if self.source_revision:
                    origin_locator = f"{origin_locator}@{self.source_revision}"
                sources.append(
                    ChunkFetchSource(
                        kind=ChunkFetchSourceKind.ORIGIN,
                        source_id=self.source_repo_id,
                        locator=origin_locator,
                    )
                )

            fetch_requests.append(
                ChunkFetchRequest(
                    chunk_id=chunk.chunk_id,
                    artifact_id=chunk.artifact_id,
                    size_bytes=chunk.size_bytes,
                    sha256_hex=chunk.sha256_hex,
                    layer_start=chunk.layer_start,
                    layer_end=chunk.layer_end,
                    sources=tuple(sources),
                )
            )

        return AssignmentFetchPlan(
            assignment=assignment,
            coverage=coverage,
            fetch_requests=tuple(fetch_requests),
            estimated_fetch_bytes=sum(request.size_bytes for request in fetch_requests),
        )

    def build_bootstrap_chunk_fetch_plan(
        self,
        assignment: ModelShardAssignment,
        present_chunk_ids: set[str] | list[str] | tuple[str, ...],
        *,
        peer_chunk_inventory: dict[str, set[str] | list[str] | tuple[str, ...]] | None = None,
        seed_chunk_inventory: dict[str, set[str] | list[str] | tuple[str, ...]] | None = None,
        peer_chunk_locators: dict[str, str] | None = None,
        seed_chunk_locators: dict[str, str] | None = None,
        max_weight_chunks: int = 1,
        max_weight_bytes: int | None = None,
        include_default_chunks: bool = True,
        hint_start_layer: int | None = None,
        hint_end_layer: int | None = None,
    ) -> AssignmentFetchPlan:
        coverage = self.compute_bootstrap_chunk_coverage(
            present_chunk_ids,
            max_weight_chunks=max_weight_chunks,
            max_weight_bytes=max_weight_bytes,
            include_default_chunks=include_default_chunks,
            hint_start_layer=hint_start_layer,
            hint_end_layer=hint_end_layer,
        )
        chunk_by_id = {
            chunk.chunk_id: chunk
            for chunk in self.bootstrap_prefetch_chunks(
                max_weight_chunks=max_weight_chunks,
                max_weight_bytes=max_weight_bytes,
                include_default_chunks=include_default_chunks,
                hint_start_layer=hint_start_layer,
                hint_end_layer=hint_end_layer,
            )
        }
        normalized_peer_inventory = {
            source_id: {str(chunk_id) for chunk_id in chunk_ids}
            for source_id, chunk_ids in (peer_chunk_inventory or {}).items()
        }
        normalized_seed_inventory = {
            source_id: {str(chunk_id) for chunk_id in chunk_ids}
            for source_id, chunk_ids in (seed_chunk_inventory or {}).items()
        }

        fetch_requests: list[ChunkFetchRequest] = []
        for chunk_id in coverage.missing_chunk_ids:
            chunk = chunk_by_id[chunk_id]
            sources: list[ChunkFetchSource] = []

            for peer_id, chunk_ids in normalized_peer_inventory.items():
                if chunk_id in chunk_ids:
                    sources.append(
                        ChunkFetchSource(
                            kind=ChunkFetchSourceKind.PEER_CACHE,
                            source_id=peer_id,
                            locator=(peer_chunk_locators or {}).get(peer_id),
                        )
                    )

            for seed_id, chunk_ids in normalized_seed_inventory.items():
                if chunk_id in chunk_ids:
                    sources.append(
                        ChunkFetchSource(
                            kind=ChunkFetchSourceKind.STORAGE_SEED,
                            source_id=seed_id,
                            locator=(seed_chunk_locators or {}).get(seed_id),
                        )
                    )

            if self.source_repo_id:
                origin_locator = self.source_repo_id
                if self.source_revision:
                    origin_locator = f"{origin_locator}@{self.source_revision}"
                sources.append(
                    ChunkFetchSource(
                        kind=ChunkFetchSourceKind.ORIGIN,
                        source_id=self.source_repo_id,
                        locator=origin_locator,
                    )
                )

            fetch_requests.append(
                ChunkFetchRequest(
                    chunk_id=chunk.chunk_id,
                    artifact_id=chunk.artifact_id,
                    size_bytes=chunk.size_bytes,
                    sha256_hex=chunk.sha256_hex,
                    layer_start=chunk.layer_start,
                    layer_end=chunk.layer_end,
                    sources=tuple(sources),
                )
            )

        return AssignmentFetchPlan(
            assignment=assignment,
            coverage=coverage,
            fetch_requests=tuple(fetch_requests),
            estimated_fetch_bytes=sum(request.size_bytes for request in fetch_requests),
        )


def _safe_segment(text: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text).strip())
    normalized = normalized.strip("-._")
    return normalized or "default"


def _chunk_overlaps_hint_range(
    chunk: ModelChunk,
    *,
    hint_start_layer: int | None,
    hint_end_layer: int | None,
) -> bool:
    if hint_start_layer is None or hint_end_layer is None:
        return False
    return chunk.overlaps_layer_range(int(hint_start_layer), int(hint_end_layer))


def _chunk_hint_distance(
    chunk: ModelChunk,
    *,
    hint_start_layer: int | None,
    hint_end_layer: int | None,
) -> int:
    if hint_start_layer is None or hint_end_layer is None:
        return 0
    if chunk.layer_start is None or chunk.layer_end is None:
        return 10**9
    start = int(hint_start_layer)
    end = int(hint_end_layer)
    if chunk.overlaps_layer_range(start, end):
        return 0
    if chunk.layer_end <= start:
        return start - chunk.layer_end
    if chunk.layer_start >= end:
        return chunk.layer_start - end
    return 0


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _hash_file_range(path: Path, *, offset_bytes: int, size_bytes: int) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(offset_bytes)
        remaining = size_bytes
        while remaining > 0:
            chunk = handle.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                break
            hasher.update(chunk)
            remaining -= len(chunk)
    return hasher.hexdigest()


def build_source_artifact_from_file(
    path: str | Path,
    *,
    artifact_id: str,
    relative_path: str | None = None,
    media_type: str | None = None,
    source_repo_id: str | None = None,
    source_revision: str | None = None,
) -> SourceArtifact:
    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.exists() or not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    return SourceArtifact(
        artifact_id=artifact_id,
        relative_path=relative_path or artifact_path.name,
        size_bytes=artifact_path.stat().st_size,
        sha256_hex=_hash_file(artifact_path),
        media_type=media_type,
        source_repo_id=source_repo_id,
        source_revision=source_revision,
    )


def recommended_chunk_size_bytes(
    total_size_bytes: int,
    policy: str = ChunkSizePolicy.ADAPTIVE,
    *,
    min_chunk_bytes: int = DEFAULT_MIN_CHUNK_BYTES,
    max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
    target_chunk_count: int | None = None,
) -> int:
    if total_size_bytes <= 0:
        raise ValueError("total_size_bytes must be positive.")
    if min_chunk_bytes <= 0:
        raise ValueError("min_chunk_bytes must be positive.")
    if max_chunk_bytes < min_chunk_bytes:
        raise ValueError("max_chunk_bytes must be >= min_chunk_bytes.")
    if target_chunk_count is not None and target_chunk_count <= 0:
        raise ValueError("target_chunk_count must be positive when provided.")

    normalized_policy = str(policy or ChunkSizePolicy.ADAPTIVE)
    if target_chunk_count is not None:
        base_size = math.ceil(total_size_bytes / target_chunk_count)
    elif normalized_policy == ChunkSizePolicy.SMALL:
        base_size = min_chunk_bytes
    elif normalized_policy == ChunkSizePolicy.LARGE:
        base_size = max_chunk_bytes
    elif normalized_policy == ChunkSizePolicy.BALANCED:
        base_size = DEFAULT_CHUNK_TARGET_BYTES
    else:
        adaptive_target_chunks = max(
            1,
            math.ceil(total_size_bytes / DEFAULT_CHUNK_TARGET_BYTES),
        )
        base_size = math.ceil(total_size_bytes / adaptive_target_chunks)

    return _clamp(base_size, min_chunk_bytes, max_chunk_bytes)


def build_weight_chunks_for_artifact(
    artifact: SourceArtifact,
    artifact_path: str | Path,
    *,
    total_layers: int,
    policy: str = ChunkSizePolicy.ADAPTIVE,
    min_chunk_bytes: int = DEFAULT_MIN_CHUNK_BYTES,
    max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
    target_chunk_count: int | None = None,
) -> list[ModelChunk]:
    if total_layers <= 0:
        raise ValueError("total_layers must be positive.")
    if artifact.size_bytes <= 0:
        raise ValueError("artifact.size_bytes must be positive.")

    resolved_path = Path(artifact_path).expanduser().resolve()
    if not resolved_path.exists() or not resolved_path.is_file():
        raise FileNotFoundError(resolved_path)

    preferred_chunk_size = recommended_chunk_size_bytes(
        artifact.size_bytes,
        policy,
        min_chunk_bytes=min_chunk_bytes,
        max_chunk_bytes=max_chunk_bytes,
        target_chunk_count=target_chunk_count,
    )
    gguf_chunks = _build_gguf_weight_chunks_for_artifact(
        artifact,
        resolved_path,
        total_layers=total_layers,
        preferred_chunk_size=preferred_chunk_size,
    )
    if gguf_chunks is not None:
        return gguf_chunks
    chunk_count = max(1, math.ceil(artifact.size_bytes / preferred_chunk_size))
    chunk_count = min(chunk_count, total_layers)

    base_chunk_bytes = artifact.size_bytes // chunk_count
    remainder_bytes = artifact.size_bytes % chunk_count
    layer_boundaries = [
        (index * total_layers) // chunk_count for index in range(chunk_count + 1)
    ]

    chunks: list[ModelChunk] = []
    offset_bytes = 0
    for index in range(chunk_count):
        size_bytes = base_chunk_bytes + (1 if index < remainder_bytes else 0)
        layer_start = layer_boundaries[index]
        layer_end = layer_boundaries[index + 1]
        if layer_end <= layer_start:
            layer_end = min(layer_start + 1, total_layers)
        if index == chunk_count - 1:
            layer_end = total_layers
            size_bytes = artifact.size_bytes - offset_bytes

        sha256_hex = _hash_file_range(
            resolved_path,
            offset_bytes=offset_bytes,
            size_bytes=size_bytes,
        )
        chunks.append(
            ModelChunk(
                chunk_id=make_chunk_id(
                    artifact.artifact_id,
                    offset_bytes=offset_bytes,
                    size_bytes=size_bytes,
                    sha256_hex=sha256_hex,
                ),
                artifact_id=artifact.artifact_id,
                kind=ModelChunkKind.WEIGHTS,
                offset_bytes=offset_bytes,
                size_bytes=size_bytes,
                sha256_hex=sha256_hex,
                layer_start=layer_start,
                layer_end=layer_end,
            )
        )
        offset_bytes += size_bytes

    return chunks


def _build_gguf_weight_chunks_for_artifact(
    artifact: SourceArtifact,
    resolved_path: Path,
    *,
    total_layers: int,
    preferred_chunk_size: int,
) -> list[ModelChunk] | None:
    layout = _maybe_extract_gguf_tensor_layout(
        resolved_path,
        total_layers=total_layers,
    )
    if layout is None or not layout.tensors:
        return None

    chunks: list[ModelChunk] = []
    current_spans: list[GgufTensorSpan] = []
    current_start = 0

    def flush_current() -> None:
        nonlocal current_spans, current_start
        if not current_spans:
            return
        chunk = _build_model_chunk_from_gguf_spans(
            artifact,
            resolved_path,
            start_offset=current_start,
            spans=current_spans,
        )
        chunks.append(chunk)
        current_spans = []
        current_start = chunk.offset_bytes + chunk.size_bytes

    for span in layout.tensors:
        if not current_spans:
            current_spans = [span]
            current_start = 0 if not chunks else span.start_offset
            continue
        current_size = max(0, current_spans[-1].end_offset - current_start)
        current_has_layer = any(item.layer_index is not None for item in current_spans)
        next_has_layer = span.layer_index is not None
        if current_has_layer != next_has_layer:
            flush_current()
            current_spans = [span]
            current_start = span.start_offset
            continue
        if current_has_layer and next_has_layer and current_size >= preferred_chunk_size:
            flush_current()
            current_spans = [span]
            current_start = span.start_offset
            continue
        current_spans.append(span)

    flush_current()
    return chunks or None


def _build_model_chunk_from_gguf_spans(
    artifact: SourceArtifact,
    resolved_path: Path,
    *,
    start_offset: int,
    spans: list[GgufTensorSpan],
) -> ModelChunk:
    end_offset = max(int(spans[-1].end_offset), int(start_offset))
    size_bytes = max(1, end_offset - int(start_offset))
    sha256_hex = _hash_file_range(
        resolved_path,
        offset_bytes=int(start_offset),
        size_bytes=int(size_bytes),
    )
    layer_indices = [
        int(item.layer_index)
        for item in spans
        if item.layer_index is not None
    ]
    layer_start = min(layer_indices) if layer_indices else None
    layer_end = (max(layer_indices) + 1) if layer_indices else None
    return ModelChunk(
        chunk_id=make_chunk_id(
            artifact.artifact_id,
            offset_bytes=int(start_offset),
            size_bytes=int(size_bytes),
            sha256_hex=sha256_hex,
        ),
        artifact_id=artifact.artifact_id,
        kind=ModelChunkKind.WEIGHTS,
        offset_bytes=int(start_offset),
        size_bytes=int(size_bytes),
        sha256_hex=sha256_hex,
        layer_start=layer_start,
        layer_end=layer_end,
        tensor_names=[str(item.name) for item in spans],
        required_by_default=not layer_indices,
    )


def _maybe_extract_gguf_tensor_layout(
    path: Path,
    *,
    total_layers: int,
) -> GgufTensorLayout | None:
    try:
        file_size = int(path.stat().st_size)
    except OSError:
        return None
    try:
        with path.open("rb") as handle:
            if _read_exact(handle, 4) != GGUF_MAGIC:
                return None
            version = _read_u32_le(handle)
            if int(version) not in {2, 3}:
                return None
            tensor_count = _read_u64_le(handle)
            kv_count = _read_u64_le(handle)
            alignment = GGUF_DEFAULT_ALIGNMENT
            for _ in range(int(kv_count)):
                key = _read_gguf_string(handle)
                value_type = _read_u32_le(handle)
                value = _read_gguf_metadata_value(handle, int(value_type))
                if (
                    key == "general.alignment"
                    and isinstance(value, int)
                    and int(value) > 0
                ):
                    alignment = int(value)
            tensor_entries: list[tuple[str, int]] = []
            for _ in range(int(tensor_count)):
                name = _read_gguf_string(handle)
                n_dimensions = _read_u32_le(handle)
                for _ in range(int(n_dimensions)):
                    _read_u64_le(handle)
                _read_u32_le(handle)  # ggml type
                relative_offset = _read_u64_le(handle)
                tensor_entries.append((name, int(relative_offset)))
            data_offset = _align_offset(int(handle.tell()), int(alignment))
    except (OSError, ValueError, struct.error):
        return None

    if data_offset < 0 or data_offset > file_size:
        return None
    ordered_entries = sorted(tensor_entries, key=lambda item: int(item[1]))
    spans: list[GgufTensorSpan] = []
    for index, (name, relative_offset) in enumerate(ordered_entries):
        start_offset = int(data_offset + int(relative_offset))
        end_offset = (
            int(data_offset + int(ordered_entries[index + 1][1]))
            if index + 1 < len(ordered_entries)
            else int(file_size)
        )
        if start_offset < data_offset or start_offset >= file_size:
            return None
        if end_offset <= start_offset:
            return None
        spans.append(
            GgufTensorSpan(
                name=str(name),
                start_offset=start_offset,
                end_offset=end_offset,
                layer_index=_infer_gguf_tensor_layer_index(name, total_layers),
            )
        )
    return GgufTensorLayout(
        data_offset=int(data_offset),
        tensor_count=len(spans),
        tensors=tuple(spans),
    )


def read_gguf_model_metadata(path: str | Path) -> GgufModelMetadata:
    metadata = _read_gguf_metadata_map(Path(path))
    architecture = _gguf_metadata_text(metadata, "general.architecture")
    total_layers = _gguf_metadata_total_layers(metadata, architecture)
    hidden_size = _gguf_metadata_hidden_size(metadata, architecture)
    return GgufModelMetadata(
        architecture=architecture,
        total_layers=total_layers,
        hidden_size=hidden_size,
        metadata=metadata,
    )


def _read_gguf_metadata_map(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            if _read_exact(handle, 4) != GGUF_MAGIC:
                raise ValueError("GGUF file magic is invalid.")
            version = _read_u32_le(handle)
            if int(version) not in {2, 3}:
                raise ValueError("GGUF file version is unsupported.")
            _read_u64_le(handle)  # tensor count
            kv_count = _read_u64_le(handle)
            metadata: dict[str, Any] = {}
            for _ in range(int(kv_count)):
                key = _read_gguf_string(handle)
                value_type = _read_u32_le(handle)
                metadata[key] = _read_gguf_metadata_value(handle, int(value_type))
            return metadata
    except (OSError, ValueError, struct.error) as exc:
        raise ValueError(f"Could not read GGUF metadata from {path}.") from exc


def _gguf_metadata_text(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _gguf_metadata_total_layers(
    metadata: dict[str, Any],
    architecture: str | None,
) -> int | None:
    candidate_keys: list[str] = []
    if architecture:
        clean_architecture = str(architecture).strip().lower()
        if clean_architecture:
            candidate_keys.append(f"{clean_architecture}.block_count")
    candidate_keys.extend(
        str(key)
        for key in metadata.keys()
        if str(key).endswith(".block_count") and str(key) not in candidate_keys
    )
    for key in candidate_keys:
        value = metadata.get(key)
        if isinstance(value, bool):
            continue
        try:
            total_layers = int(value)
        except (TypeError, ValueError):
            continue
        if total_layers > 0:
            return total_layers
    return None


def _gguf_metadata_hidden_size(
    metadata: dict[str, Any],
    architecture: str | None,
) -> int | None:
    candidate_keys: list[str] = []
    if architecture:
        clean_architecture = str(architecture).strip().lower()
        if clean_architecture:
            candidate_keys.extend(
                (
                    f"{clean_architecture}.embedding_length",
                    f"{clean_architecture}.n_embd",
                    f"{clean_architecture}.hidden_size",
                )
            )
    candidate_keys.extend(
        str(key)
        for key in metadata.keys()
        if (
            str(key).endswith(".embedding_length")
            or str(key).endswith(".n_embd")
            or str(key).endswith(".hidden_size")
        )
        and str(key) not in candidate_keys
    )
    for key in candidate_keys:
        value = metadata.get(key)
        if isinstance(value, bool):
            continue
        try:
            hidden_size = int(value)
        except (TypeError, ValueError):
            continue
        if hidden_size > 0:
            return hidden_size
    return None


def _read_exact(handle, size: int) -> bytes:  # type: ignore[no-untyped-def]
    data = handle.read(int(size))
    if len(data) != int(size):
        raise ValueError("Unexpected end of GGUF file.")
    return data


def _read_u32_le(handle) -> int:  # type: ignore[no-untyped-def]
    return int(struct.unpack("<I", _read_exact(handle, 4))[0])


def _read_u64_le(handle) -> int:  # type: ignore[no-untyped-def]
    return int(struct.unpack("<Q", _read_exact(handle, 8))[0])


def _read_gguf_string(handle) -> str:  # type: ignore[no-untyped-def]
    length = _read_u64_le(handle)
    return _read_exact(handle, length).decode("utf-8", errors="replace")


def _read_gguf_metadata_value(handle, value_type: int) -> Any:  # type: ignore[no-untyped-def]
    scalar_formats: dict[int, tuple[str, int]] = {
        GGUF_METADATA_TYPE_UINT8: ("<B", 1),
        GGUF_METADATA_TYPE_INT8: ("<b", 1),
        GGUF_METADATA_TYPE_UINT16: ("<H", 2),
        GGUF_METADATA_TYPE_INT16: ("<h", 2),
        GGUF_METADATA_TYPE_UINT32: ("<I", 4),
        GGUF_METADATA_TYPE_INT32: ("<i", 4),
        GGUF_METADATA_TYPE_FLOAT32: ("<f", 4),
        GGUF_METADATA_TYPE_UINT64: ("<Q", 8),
        GGUF_METADATA_TYPE_INT64: ("<q", 8),
        GGUF_METADATA_TYPE_FLOAT64: ("<d", 8),
    }
    if value_type in scalar_formats:
        fmt, size = scalar_formats[value_type]
        return struct.unpack(fmt, _read_exact(handle, size))[0]
    if value_type == GGUF_METADATA_TYPE_BOOL:
        return bool(struct.unpack("<B", _read_exact(handle, 1))[0])
    if value_type == GGUF_METADATA_TYPE_STRING:
        return _read_gguf_string(handle)
    if value_type == GGUF_METADATA_TYPE_ARRAY:
        nested_type = _read_u32_le(handle)
        item_count = _read_u64_le(handle)
        return [
            _read_gguf_metadata_value(handle, int(nested_type))
            for _ in range(int(item_count))
        ]
    raise ValueError(f"Unsupported GGUF metadata type: {value_type}")


def _align_offset(value: int, alignment: int) -> int:
    if alignment <= 1:
        return int(value)
    remainder = int(value) % int(alignment)
    if remainder == 0:
        return int(value)
    return int(value) + (int(alignment) - remainder)


def _infer_gguf_tensor_layer_index(
    tensor_name: str,
    total_layers: int,
) -> int | None:
    clean_name = str(tensor_name or "").strip()
    if not clean_name:
        return None
    patterns = (
        r"(?:^|[./_])blk\.(\d+)\.",
        r"(?:^|[./_])model\.layers\.(\d+)\.",
        r"(?:^|[./_])layers\.(\d+)\.",
        r"(?:^|[./_])layer\.(\d+)\.",
        r"(?:^|[./_])h\.(\d+)\.",
    )
    for pattern in patterns:
        match = re.search(pattern, clean_name)
        if match is None:
            continue
        try:
            layer_index = int(match.group(1))
        except (TypeError, ValueError):
            continue
        if 0 <= layer_index < int(total_layers):
            return layer_index
    return None


def build_gguf_model_package_manifest(
    *,
    catalog_id: str,
    model_id: str,
    version: str,
    gguf_path: str | Path,
    total_layers: int | None = None,
    backend: str = "llama_cpp",
    package_kind: str = ModelPackageKind.PUBLIC_SHARED,
    chunk_size_policy: str = ChunkSizePolicy.ADAPTIVE,
    min_chunk_bytes: int = DEFAULT_MIN_CHUNK_BYTES,
    max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
    target_chunk_count: int | None = None,
    source_repo_id: str | None = None,
    source_revision: str | None = "main",
    family: str = "",
    quantization: str = "",
    allow_full_model_local: bool | None = None,
) -> ModelPackageManifest:
    resolved_total_layers = int(total_layers or 0)
    resolved_family = str(family or "").strip()
    gguf_metadata: GgufModelMetadata | None = None
    try:
        gguf_metadata = read_gguf_model_metadata(gguf_path)
    except ValueError:
        if resolved_total_layers <= 0:
            raise
    if gguf_metadata is not None:
        if gguf_metadata.architecture:
            resolved_family = gguf_metadata.architecture
        if gguf_metadata.total_layers:
            resolved_total_layers = int(gguf_metadata.total_layers)
    if resolved_total_layers <= 0:
        raise ValueError(
            "total_layers must be positive or discoverable from GGUF metadata."
        )

    artifact = build_source_artifact_from_file(
        gguf_path,
        artifact_id="gguf-main",
        relative_path=Path(gguf_path).name,
        media_type="application/gguf",
        source_repo_id=source_repo_id,
        source_revision=source_revision,
    )
    chunks = build_weight_chunks_for_artifact(
        artifact,
        gguf_path,
        total_layers=resolved_total_layers,
        policy=chunk_size_policy,
        min_chunk_bytes=min_chunk_bytes,
        max_chunk_bytes=max_chunk_bytes,
        target_chunk_count=target_chunk_count,
    )
    shard_compatibility = gguf_shard_compatibility(
        model_id=model_id,
        gguf_architecture=resolved_family,
        family=resolved_family,
        filename=artifact.relative_path,
        allow_full_model_local=(
            str(package_kind) == str(ModelPackageKind.PUBLIC_SHARED)
            if allow_full_model_local is None
            else bool(allow_full_model_local)
        ),
    )
    manifest_metadata: dict[str, Any] = {
        "builder": "cai_compute_chain",
        "chunk_target_bytes": recommended_chunk_size_bytes(
            artifact.size_bytes,
            chunk_size_policy,
            min_chunk_bytes=min_chunk_bytes,
            max_chunk_bytes=max_chunk_bytes,
            target_chunk_count=target_chunk_count,
        ),
        "chunk_count": len(chunks),
        "total_layers": resolved_total_layers,
        **shard_compatibility.to_metadata(),
    }
    if gguf_metadata is not None and gguf_metadata.hidden_size:
        manifest_metadata["hidden_size"] = int(gguf_metadata.hidden_size)
    manifest = ModelPackageManifest(
        catalog_id=catalog_id,
        model_id=model_id,
        version=version,
        backend=backend,
        package_kind=package_kind,
        chunk_size_policy=chunk_size_policy,
        total_size_bytes=artifact.size_bytes,
        source_repo_id=source_repo_id,
        source_revision=source_revision,
        preferred_filename=artifact.relative_path,
        family=resolved_family,
        quantization=quantization,
        files=[artifact],
        chunks=chunks,
        metadata=manifest_metadata,
    )
    validate_model_package_manifest(manifest)
    return manifest


def model_package_root(policy: WalletPolicy | None = None) -> Path:
    root = data_root(policy) / "model-packages"
    root.mkdir(parents=True, exist_ok=True)
    return root


def chunk_store_root(policy: WalletPolicy | None = None) -> Path:
    root = data_root(policy) / "chunk-store"
    root.mkdir(parents=True, exist_ok=True)
    return root


def chunk_store_chunks_dir(policy: WalletPolicy | None = None) -> Path:
    root = chunk_store_root(policy) / "chunks"
    root.mkdir(parents=True, exist_ok=True)
    return root


def chunk_store_index_path(policy: WalletPolicy | None = None) -> Path:
    return chunk_store_root(policy) / "index.json"


def chunk_inventory_root(policy: WalletPolicy | None = None) -> Path:
    root = data_root(policy) / "chunk-inventories"
    root.mkdir(parents=True, exist_ok=True)
    return root


def local_chunk_inventory_path(
    source_id: str,
    policy: WalletPolicy | None = None,
) -> Path:
    directory = chunk_inventory_root(policy) / "local"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{_safe_segment(source_id)}.json"


def imported_chunk_inventory_dir(
    source_kind: str,
    policy: WalletPolicy | None = None,
) -> Path:
    directory = chunk_inventory_root(policy) / "imported" / _safe_segment(source_kind)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def imported_chunk_inventory_path(
    source_id: str,
    source_kind: str,
    policy: WalletPolicy | None = None,
) -> Path:
    return imported_chunk_inventory_dir(source_kind, policy) / f"{_safe_segment(source_id)}.json"


def chunk_download_queue_root(policy: WalletPolicy | None = None) -> Path:
    root = data_root(policy) / "chunk-download-queue"
    root.mkdir(parents=True, exist_ok=True)
    return root


def chunk_download_queue_index_path(policy: WalletPolicy | None = None) -> Path:
    return chunk_download_queue_root(policy) / "index.json"


def chunk_source_health_index_path(policy: WalletPolicy | None = None) -> Path:
    return data_root(policy) / "chunk-source-health.json"


def chunk_storage_accounting_dir(policy: WalletPolicy | None = None) -> Path:
    root = data_root(policy) / "chunk-storage-accounting"
    root.mkdir(parents=True, exist_ok=True)
    return root


def chunk_storage_accounting_path(
    node_id: str,
    policy: WalletPolicy | None = None,
) -> Path:
    return chunk_storage_accounting_dir(policy) / f"{_safe_segment(node_id)}.json"


def recent_shard_hints_root(policy: WalletPolicy | None = None) -> Path:
    root = data_root(policy) / "recent-shard-hints"
    root.mkdir(parents=True, exist_ok=True)
    return root


def recent_shard_hints_path(
    node_id: str,
    policy: WalletPolicy | None = None,
) -> Path:
    return recent_shard_hints_root(policy) / f"{_safe_segment(node_id)}.json"


def model_package_dir(
    catalog_id: str, version: str, policy: WalletPolicy | None = None
) -> Path:
    path = model_package_root(policy) / _safe_segment(catalog_id) / _safe_segment(version)
    path.mkdir(parents=True, exist_ok=True)
    return path


def manifest_file_path(
    catalog_id: str, version: str, policy: WalletPolicy | None = None
) -> Path:
    return model_package_dir(catalog_id, version, policy) / "manifest.json"


def local_artifact_bindings_path(
    catalog_id: str, version: str, policy: WalletPolicy | None = None
) -> Path:
    return model_package_dir(catalog_id, version, policy) / "local-artifacts.json"


def materialized_artifacts_dir(
    catalog_id: str, version: str, policy: WalletPolicy | None = None
) -> Path:
    path = model_package_dir(catalog_id, version, policy) / "materialized"
    path.mkdir(parents=True, exist_ok=True)
    return path


def chunk_source_bindings_path(policy: WalletPolicy | None = None) -> Path:
    return data_root(policy) / "chunk-source-bindings.json"


def save_model_package_manifest(
    manifest: ModelPackageManifest, policy: WalletPolicy | None = None
) -> Path:
    validate_model_package_manifest(manifest)
    path = manifest_file_path(manifest.catalog_id, manifest.version, policy)
    path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_local_artifact_bindings(
    catalog_id: str,
    version: str,
    policy: WalletPolicy | None = None,
) -> LocalArtifactBindings:
    path = local_artifact_bindings_path(catalog_id, version, policy)
    if not path.exists():
        return LocalArtifactBindings(
            catalog_id=catalog_id,
            version=version,
            bindings=(),
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return LocalArtifactBindings.from_dict(payload)


def save_local_artifact_binding(
    catalog_id: str,
    version: str,
    *,
    artifact_id: str,
    local_path: str | Path,
    policy: WalletPolicy | None = None,
) -> Path:
    resolved = str(Path(local_path).expanduser().resolve())
    existing = load_local_artifact_bindings(catalog_id, version, policy)
    updated_bindings: list[LocalArtifactBinding] = []
    replaced = False
    for binding in existing.bindings:
        if binding.artifact_id != artifact_id:
            updated_bindings.append(binding)
            continue
        updated_bindings.append(
            LocalArtifactBinding(
                artifact_id=artifact_id,
                local_path=resolved,
                updated_at=_now_iso(),
            )
        )
        replaced = True
    if not replaced:
        updated_bindings.append(
            LocalArtifactBinding(
                artifact_id=artifact_id,
                local_path=resolved,
                updated_at=_now_iso(),
            )
        )
    payload = LocalArtifactBindings(
        catalog_id=catalog_id,
        version=version,
        bindings=tuple(updated_bindings),
    )
    path = local_artifact_bindings_path(catalog_id, version, policy)
    path.write_text(
        json.dumps(payload.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_chunk_source_bindings(
    policy: WalletPolicy | None = None,
) -> ChunkSourceBindings:
    path = chunk_source_bindings_path(policy)
    if not path.exists():
        return ChunkSourceBindings(bindings=())
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ChunkSourceBindings.from_dict(payload)


def get_chunk_source_binding(
    source_kind: str,
    source_id: str,
    policy: WalletPolicy | None = None,
) -> ChunkSourceBinding | None:
    bindings = load_chunk_source_bindings(policy)
    for binding in bindings.bindings:
        if binding.source_kind == source_kind and binding.source_id == source_id:
            return binding
    return None


def save_chunk_source_binding(
    source_kind: str,
    source_id: str,
    *,
    data_root_path: str | Path,
    policy: WalletPolicy | None = None,
) -> Path:
    resolved = str(Path(data_root_path).expanduser().resolve())
    existing = load_chunk_source_bindings(policy)
    updated_bindings: list[ChunkSourceBinding] = []
    replaced = False
    for binding in existing.bindings:
        if binding.source_kind != source_kind or binding.source_id != source_id:
            updated_bindings.append(binding)
            continue
        updated_bindings.append(
            ChunkSourceBinding(
                source_kind=source_kind,
                source_id=source_id,
                data_root_path=resolved,
                updated_at=_now_iso(),
            )
        )
        replaced = True
    if not replaced:
        updated_bindings.append(
            ChunkSourceBinding(
                source_kind=source_kind,
                source_id=source_id,
                data_root_path=resolved,
                updated_at=_now_iso(),
            )
        )
    payload = ChunkSourceBindings(bindings=tuple(updated_bindings))
    path = chunk_source_bindings_path(policy)
    path.write_text(
        json.dumps(payload.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _read_chunk_store_records(
    policy: WalletPolicy | None = None,
) -> list[CachedChunkRecord]:
    path = chunk_store_index_path(policy)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [CachedChunkRecord(**item) for item in raw]


def _write_chunk_store_records(
    records: list[CachedChunkRecord],
    policy: WalletPolicy | None = None,
) -> None:
    path = chunk_store_index_path(policy)
    path.write_text(
        json.dumps([asdict(item) for item in records], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_chunk_download_tasks(
    policy: WalletPolicy | None = None,
) -> list[ChunkDownloadTask]:
    path = chunk_download_queue_index_path(policy)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    tasks: list[ChunkDownloadTask] = []
    for item in raw:
        payload = dict(item)
        payload["sources"] = tuple(
            ChunkFetchSource(**source_item) for source_item in payload.get("sources", [])
        )
        tasks.append(ChunkDownloadTask(**payload))
    return tasks


def _write_chunk_download_tasks(
    tasks: list[ChunkDownloadTask],
    policy: WalletPolicy | None = None,
) -> None:
    path = chunk_download_queue_index_path(policy)
    path.write_text(
        json.dumps(
            [
                {
                    **asdict(task),
                    "sources": [asdict(source) for source in task.sources],
                }
                for task in tasks
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _source_health_key(source_kind: str, source_id: str, locator: str | None = None) -> str:
    normalized_locator = str(locator or "").strip().rstrip("/")
    return f"{source_kind}|{source_id}|{normalized_locator}"


def _read_chunk_source_health_records(
    policy: WalletPolicy | None = None,
) -> list[ChunkSourceHealthRecord]:
    path = chunk_source_health_index_path(policy)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [ChunkSourceHealthRecord(**item) for item in raw]


def _write_chunk_source_health_records(
    records: list[ChunkSourceHealthRecord],
    policy: WalletPolicy | None = None,
) -> None:
    path = chunk_source_health_index_path(policy)
    path.write_text(
        json.dumps([asdict(item) for item in records], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_chunk_storage_accounting_records(
    node_id: str,
    policy: WalletPolicy | None = None,
) -> list[ChunkStorageAccountingRecord]:
    path = chunk_storage_accounting_path(node_id, policy)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    allowed_fields = set(ChunkStorageAccountingRecord.__dataclass_fields__)
    return [
        ChunkStorageAccountingRecord(
            **{key: value for key, value in item.items() if key in allowed_fields}
        )
        for item in raw
    ]


def _write_chunk_storage_accounting_records(
    node_id: str,
    records: list[ChunkStorageAccountingRecord],
    policy: WalletPolicy | None = None,
) -> None:
    path = chunk_storage_accounting_path(node_id, policy)
    path.write_text(
        json.dumps([asdict(item) for item in records], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def list_chunk_storage_accounting_records(
    node_id: str,
    policy: WalletPolicy | None = None,
) -> list[ChunkStorageAccountingRecord]:
    return _read_chunk_storage_accounting_records(node_id, policy)


def list_chunk_source_health_records(
    policy: WalletPolicy | None = None,
) -> list[ChunkSourceHealthRecord]:
    return _read_chunk_source_health_records(policy)


def prune_chunk_source_health_records(
    *,
    allowed_source_ids: set[str] | list[str] | tuple[str, ...] | None = None,
    allowed_locators: set[str] | list[str] | tuple[str, ...] | None = None,
    source_kind: str | None = None,
    policy: WalletPolicy | None = None,
) -> tuple[str, ...]:
    normalized_allowed_source_ids = {
        str(source_id).strip()
        for source_id in (allowed_source_ids or ())
        if str(source_id).strip()
    }
    normalized_allowed_locators = {
        str(locator).strip().rstrip("/")
        for locator in (allowed_locators or ())
        if str(locator).strip()
    }
    records = _read_chunk_source_health_records(policy)
    kept: list[ChunkSourceHealthRecord] = []
    removed: list[str] = []
    for record in records:
        if source_kind and record.source_kind != source_kind:
            kept.append(record)
            continue
        should_prune = False
        if normalized_allowed_source_ids and record.source_id not in normalized_allowed_source_ids:
            should_prune = True
        normalized_locator = str(record.locator or "").strip().rstrip("/")
        if normalized_allowed_locators and normalized_locator not in normalized_allowed_locators:
            should_prune = True
        if not should_prune:
            kept.append(record)
            continue
        removed.append(
            _source_health_key(
                record.source_kind,
                record.source_id,
                record.locator,
            )
        )
    if len(kept) != len(records):
        _write_chunk_source_health_records(kept, policy)
    return tuple(removed)


def _recent_shard_hint_key(record: RecentShardHintRecord) -> tuple[str, int, int, int, int]:
    return (
        str(record.model_id),
        int(record.start_layer),
        int(record.end_layer),
        int(record.device_rank),
        int(record.world_size),
    )


def _read_recent_shard_hint_records(
    node_id: str,
    policy: WalletPolicy | None = None,
) -> list[RecentShardHintRecord]:
    path = recent_shard_hints_path(node_id, policy)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [RecentShardHintRecord(**item) for item in raw]


def _write_recent_shard_hint_records(
    node_id: str,
    records: list[RecentShardHintRecord],
    policy: WalletPolicy | None = None,
) -> None:
    path = recent_shard_hints_path(node_id, policy)
    path.write_text(
        json.dumps([asdict(item) for item in records], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def list_recent_shard_hints(
    node_id: str,
    *,
    fetch_policy: ChunkFetchPolicy | None = None,
    policy: WalletPolicy | None = None,
) -> tuple[RecentShardHintRecord, ...]:
    active_fetch_policy = fetch_policy or ChunkFetchPolicy()
    ttl_seconds = max(0, int(active_fetch_policy.recent_shard_hint_ttl_seconds))
    now = datetime.now(tz=UTC)
    records = _read_recent_shard_hint_records(node_id, policy)
    kept: list[RecentShardHintRecord] = []
    for record in records:
        if ttl_seconds > 0:
            last_seen = _parse_iso_datetime(record.last_seen_at)
            if last_seen is None:
                continue
            if last_seen.tzinfo is None:
                last_seen = last_seen.replace(tzinfo=UTC)
            age_seconds = (now - last_seen.astimezone(UTC)).total_seconds()
            if age_seconds > ttl_seconds:
                continue
        kept.append(record)
    kept.sort(
        key=lambda record: (
            _parse_iso_datetime(record.last_seen_at) or datetime.min.replace(tzinfo=UTC),
            int(record.use_count),
        ),
        reverse=True,
    )
    capacity = max(1, int(active_fetch_policy.recent_shard_hint_capacity_per_node))
    trimmed = kept[:capacity]
    if trimmed != records:
        _write_recent_shard_hint_records(node_id, trimmed, policy)
    return tuple(trimmed)


def remember_recent_shard_hints(
    node_id: str,
    hints: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    fetch_policy: ChunkFetchPolicy | None = None,
    policy: WalletPolicy | None = None,
    now: datetime | None = None,
) -> RecentShardHintUpdateResult:
    active_fetch_policy = fetch_policy or ChunkFetchPolicy()
    current_time = now or datetime.now(tz=UTC)
    current_time_iso = current_time.astimezone(UTC).isoformat()
    existing_records = list(_read_recent_shard_hint_records(node_id, policy))
    existing_by_key = {
        _recent_shard_hint_key(record): record
        for record in existing_records
    }
    records_upserted = 0
    hints_received = 0
    for hint in hints:
        model_id = str(hint.get("model_id") or "").strip()
        start_layer = hint.get("start_layer")
        end_layer = hint.get("end_layer")
        if not model_id or start_layer is None or end_layer is None:
            continue
        hints_received += 1
        key = (
            model_id,
            int(start_layer),
            int(end_layer),
            int(hint.get("device_rank", 0) or 0),
            int(hint.get("world_size", 1) or 1),
        )
        existing = existing_by_key.get(key)
        if existing is None:
            existing_by_key[key] = RecentShardHintRecord(
                model_id=model_id,
                start_layer=int(start_layer),
                end_layer=int(end_layer),
                device_rank=int(hint.get("device_rank", 0) or 0),
                world_size=int(hint.get("world_size", 1) or 1),
                first_seen_at=current_time_iso,
                last_seen_at=current_time_iso,
                use_count=1,
            )
        else:
            existing_by_key[key] = RecentShardHintRecord(
                model_id=existing.model_id,
                start_layer=existing.start_layer,
                end_layer=existing.end_layer,
                device_rank=existing.device_rank,
                world_size=existing.world_size,
                first_seen_at=existing.first_seen_at or current_time_iso,
                last_seen_at=current_time_iso,
                use_count=int(existing.use_count) + 1,
            )
        records_upserted += 1
    updated_records = list(existing_by_key.values())
    updated_records.sort(
        key=lambda record: (
            _parse_iso_datetime(record.last_seen_at) or datetime.min.replace(tzinfo=UTC),
            int(record.use_count),
        ),
        reverse=True,
    )
    capacity = max(1, int(active_fetch_policy.recent_shard_hint_capacity_per_node))
    updated_records = updated_records[:capacity]
    records_after_upsert = len(updated_records)
    _write_recent_shard_hint_records(node_id, updated_records, policy)
    trimmed_records = list_recent_shard_hints(
        node_id,
        fetch_policy=active_fetch_policy,
        policy=policy,
    )
    return RecentShardHintUpdateResult(
        hints_received=hints_received,
        records_upserted=records_upserted,
        records_pruned=max(0, records_after_upsert - len(trimmed_records)),
        stored_records=len(trimmed_records),
    )


def prefetch_recent_shard_hints(
    node_id: str,
    *,
    max_hints: int | None = None,
    fetch_policy: ChunkFetchPolicy | None = None,
    policy: WalletPolicy | None = None,
) -> HintedChunkPrefetchResult:
    active_fetch_policy = fetch_policy or ChunkFetchPolicy()
    recent_hints = list_recent_shard_hints(
        node_id,
        fetch_policy=active_fetch_policy,
        policy=policy,
    )
    if not recent_hints:
        return HintedChunkPrefetchResult(
            manifests_considered=0,
            manifests_prefetched=0,
            queued_tasks=0,
            processed_tasks=0,
        )
    limit = max_hints
    if limit is None:
        limit = max(1, int(active_fetch_policy.recent_hint_prefetch_max_hints))
    normalized_hints = [
        {
            "model_id": record.model_id,
            "start_layer": int(record.start_layer),
            "end_layer": int(record.end_layer),
            "device_rank": int(record.device_rank),
            "world_size": int(record.world_size),
            "node_id": str(node_id),
        }
        for record in recent_hints[: max(1, int(limit))]
    ]
    return prefetch_hinted_bootstrap_chunks(
        normalized_hints,
        fetch_policy=active_fetch_policy,
        policy=policy,
    )


def get_chunk_source_health_record(
    source_kind: str,
    source_id: str,
    locator: str | None = None,
    policy: WalletPolicy | None = None,
) -> ChunkSourceHealthRecord | None:
    key = _source_health_key(source_kind, source_id, locator)
    for record in _read_chunk_source_health_records(policy):
        if _source_health_key(record.source_kind, record.source_id, record.locator) == key:
            return record
    return None


def _source_cooldown_until(
    previous_record: ChunkSourceHealthRecord | None,
    fetch_policy: ChunkFetchPolicy,
) -> str | None:
    base_seconds = max(0, int(fetch_policy.source_failure_cooldown_seconds))
    if base_seconds <= 0:
        return None
    previous_failures = (
        previous_record.consecutive_failures if previous_record is not None else 0
    )
    multiplier_power = max(previous_failures, 0)
    cooldown_seconds = int(
        min(
            max(1, base_seconds * (fetch_policy.source_failure_backoff_multiplier ** multiplier_power)),
            max(1, int(fetch_policy.max_source_failure_cooldown_seconds)),
        )
    )
    return (
        datetime.now(tz=UTC).replace(microsecond=0)
        + timedelta(seconds=cooldown_seconds)
    ).isoformat()


def record_chunk_source_success(
    source: ChunkFetchSource,
    *,
    fetch_policy: ChunkFetchPolicy | None = None,
    policy: WalletPolicy | None = None,
) -> ChunkSourceHealthRecord:
    active_fetch_policy = fetch_policy or ChunkFetchPolicy()
    records = _read_chunk_source_health_records(policy)
    key = _source_health_key(source.kind, source.source_id, source.locator)
    previous = next(
        (
            record
            for record in records
            if _source_health_key(record.source_kind, record.source_id, record.locator) == key
        ),
        None,
    )
    updated = ChunkSourceHealthRecord(
        source_kind=source.kind,
        source_id=source.source_id,
        locator=source.locator,
        success_count=(previous.success_count if previous is not None else 0) + 1,
        failure_count=previous.failure_count if previous is not None else 0,
        consecutive_failures=0 if active_fetch_policy.reset_failures_on_success else (
            previous.consecutive_failures if previous is not None else 0
        ),
        last_success_at=_now_iso(),
        last_failure_at=previous.last_failure_at if previous is not None else None,
        cooldown_until=None,
        last_error=None,
    )
    replaced = False
    for index, record in enumerate(records):
        if _source_health_key(record.source_kind, record.source_id, record.locator) != key:
            continue
        records[index] = updated
        replaced = True
        break
    if not replaced:
        records.append(updated)
    _write_chunk_source_health_records(records, policy)
    return updated


def record_chunk_source_failure(
    source: ChunkFetchSource,
    *,
    error: str,
    fetch_policy: ChunkFetchPolicy | None = None,
    policy: WalletPolicy | None = None,
) -> ChunkSourceHealthRecord:
    active_fetch_policy = fetch_policy or ChunkFetchPolicy()
    records = _read_chunk_source_health_records(policy)
    key = _source_health_key(source.kind, source.source_id, source.locator)
    previous = next(
        (
            record
            for record in records
            if _source_health_key(record.source_kind, record.source_id, record.locator) == key
        ),
        None,
    )
    updated = ChunkSourceHealthRecord(
        source_kind=source.kind,
        source_id=source.source_id,
        locator=source.locator,
        success_count=previous.success_count if previous is not None else 0,
        failure_count=(previous.failure_count if previous is not None else 0) + 1,
        consecutive_failures=(previous.consecutive_failures if previous is not None else 0) + 1,
        last_success_at=previous.last_success_at if previous is not None else None,
        last_failure_at=_now_iso(),
        cooldown_until=_source_cooldown_until(previous, active_fetch_policy),
        last_error=str(error),
    )
    replaced = False
    for index, record in enumerate(records):
        if _source_health_key(record.source_kind, record.source_id, record.locator) != key:
            continue
        records[index] = updated
        replaced = True
        break
    if not replaced:
        records.append(updated)
    _write_chunk_source_health_records(records, policy)
    return updated


def _source_is_in_cooldown(
    source: ChunkFetchSource,
    *,
    policy: WalletPolicy | None = None,
) -> bool:
    record = get_chunk_source_health_record(
        source.kind,
        source.source_id,
        source.locator,
        policy,
    )
    if record is None:
        return False
    cooldown_until = _parse_iso_datetime(record.cooldown_until)
    if cooldown_until is None:
        return False
    return cooldown_until > datetime.now(tz=UTC)


def order_chunk_fetch_sources(
    sources: tuple[ChunkFetchSource, ...] | list[ChunkFetchSource],
    *,
    fetch_policy: ChunkFetchPolicy | None = None,
    policy: WalletPolicy | None = None,
) -> tuple[ChunkFetchSource, ...]:
    active_fetch_policy = fetch_policy or ChunkFetchPolicy()
    if not active_fetch_policy.prefer_healthy_sources:
        return tuple(sources)

    kind_rank = {
        ChunkFetchSourceKind.PEER_CACHE: 0,
        ChunkFetchSourceKind.STORAGE_SEED: 1,
        ChunkFetchSourceKind.ORIGIN: 2,
    }
    source_keys = {
        _source_health_key(source.kind, source.source_id, source.locator): source
        for source in sources
    }
    health_records = {
        key: get_chunk_source_health_record(
            source.kind,
            source.source_id,
            source.locator,
            policy,
        )
        for key, source in source_keys.items()
    }
    cooldown_keys = {
        key
        for key, source in source_keys.items()
        if _source_is_in_cooldown(source, policy=policy)
    }
    if (
        active_fetch_policy.skip_cooldowned_sources_when_alternatives_exist
        and cooldown_keys
        and len(cooldown_keys) < len(source_keys)
    ):
        filtered_sources = tuple(
            source
            for key, source in source_keys.items()
            if key not in cooldown_keys
        )
        if filtered_sources:
            sources = filtered_sources
            source_keys = {
                _source_health_key(source.kind, source.source_id, source.locator): source
                for source in sources
            }
            health_records = {
                key: health_records.get(key)
                for key in source_keys
            }
    ordered = sorted(
        enumerate(sources),
        key=lambda item: (
            1
            if (
                (
                    health_records.get(
                        _source_health_key(
                            item[1].kind,
                            item[1].source_id,
                            item[1].locator,
                        )
                    )
                )
                and _source_is_in_cooldown(item[1], policy=policy)
            )
            else 0,
            (
                health_records[
                    _source_health_key(
                        item[1].kind,
                        item[1].source_id,
                        item[1].locator,
                    )
                ].consecutive_failures
                if health_records.get(
                    _source_health_key(
                        item[1].kind,
                        item[1].source_id,
                        item[1].locator,
                    )
                )
                is not None
                else 0
            ),
            kind_rank.get(item[1].kind, 99),
            item[0],
        ),
    )
    return tuple(source for _index, source in ordered)


def adapt_assignment_fetch_plan_to_source_health(
    fetch_plan: AssignmentFetchPlan,
    *,
    fetch_policy: ChunkFetchPolicy | None = None,
    policy: WalletPolicy | None = None,
) -> AssignmentFetchPlan:
    active_fetch_policy = fetch_policy or ChunkFetchPolicy()
    if not active_fetch_policy.prefer_healthy_sources:
        return fetch_plan

    updated_requests = tuple(
        ChunkFetchRequest(
            chunk_id=request.chunk_id,
            artifact_id=request.artifact_id,
            size_bytes=request.size_bytes,
            sha256_hex=request.sha256_hex,
            layer_start=request.layer_start,
            layer_end=request.layer_end,
            sources=order_chunk_fetch_sources(
                request.sources,
                fetch_policy=active_fetch_policy,
                policy=policy,
            ),
        )
        for request in fetch_plan.fetch_requests
    )
    return AssignmentFetchPlan(
        assignment=fetch_plan.assignment,
        coverage=fetch_plan.coverage,
        fetch_requests=updated_requests,
        estimated_fetch_bytes=fetch_plan.estimated_fetch_bytes,
    )


def cached_chunk_path(
    chunk_id: str,
    *,
    suffix: str = ".bin",
    policy: WalletPolicy | None = None,
) -> Path:
    safe_id = _safe_segment(chunk_id)
    return chunk_store_chunks_dir(policy) / f"{safe_id}{suffix}"


def put_cached_chunk(
    *,
    catalog_id: str,
    version: str,
    chunk_id: str,
    sha256_hex: str,
    content: bytes,
    pinned: bool = False,
    cache_class: ChunkCacheClass = ChunkCacheClass.WARM,
    lease_status: ChunkLeaseStatus = ChunkLeaseStatus.NONE,
    lease_expires_at: str | None = None,
    policy: WalletPolicy | None = None,
) -> CachedChunkRecord:
    payload_hash = hashlib.sha256(content).hexdigest()
    if payload_hash != sha256_hex:
        raise ModelManifestValidationError(
            f"Chunk payload hash mismatch for {chunk_id}: expected {sha256_hex}, got {payload_hash}."
        )

    path = cached_chunk_path(chunk_id, policy=policy)
    path.write_bytes(content)

    records = _read_chunk_store_records(policy)
    now = _now_iso()
    relative_path = str(path.relative_to(chunk_store_root(policy))).replace("\\", "/")

    updated = CachedChunkRecord(
        chunk_id=chunk_id,
        catalog_id=catalog_id,
        version=version,
        relative_path=relative_path,
        size_bytes=len(content),
        sha256_hex=sha256_hex,
        stored_at=now,
        last_used_at=now,
        use_count=0,
        pinned=pinned,
        cache_class=cache_class,
        lease_status=lease_status,
        lease_expires_at=lease_expires_at,
    )

    replaced = False
    for index, record in enumerate(records):
        if record.chunk_id == chunk_id:
            records[index] = updated
            replaced = True
            break
    if not replaced:
        records.append(updated)

    _write_chunk_store_records(records, policy)
    evict_chunks_to_policy_target(policy=policy)
    return updated


def list_cached_chunks(policy: WalletPolicy | None = None) -> list[CachedChunkRecord]:
    return _read_chunk_store_records(policy)


def get_cached_chunk_record(
    chunk_id: str,
    policy: WalletPolicy | None = None,
) -> CachedChunkRecord | None:
    for record in _read_chunk_store_records(policy):
        if record.chunk_id == chunk_id:
            return record
    return None


def mark_cached_chunk_used(
    chunk_id: str,
    policy: WalletPolicy | None = None,
) -> CachedChunkRecord | None:
    records = _read_chunk_store_records(policy)
    updated_record: CachedChunkRecord | None = None
    for index, record in enumerate(records):
        if record.chunk_id != chunk_id:
            continue
        updated_record = CachedChunkRecord(
            chunk_id=record.chunk_id,
            catalog_id=record.catalog_id,
            version=record.version,
            relative_path=record.relative_path,
            size_bytes=record.size_bytes,
            sha256_hex=record.sha256_hex,
            stored_at=record.stored_at,
            last_used_at=_now_iso(),
            use_count=record.use_count + 1,
            pinned=record.pinned,
            cache_class=record.cache_class,
            lease_status=record.lease_status,
            lease_expires_at=record.lease_expires_at,
        )
        records[index] = updated_record
        break
    if updated_record is not None:
        _write_chunk_store_records(records, policy)
    return updated_record


def update_cached_chunk_record(
    chunk_id: str,
    *,
    pinned: bool | None = None,
    cache_class: str | None = None,
    lease_status: str | None = None,
    lease_expires_at: str | None = None,
    policy: WalletPolicy | None = None,
) -> CachedChunkRecord | None:
    records = _read_chunk_store_records(policy)
    updated_record: CachedChunkRecord | None = None
    for index, record in enumerate(records):
        if record.chunk_id != chunk_id:
            continue
        updated_record = CachedChunkRecord(
            chunk_id=record.chunk_id,
            catalog_id=record.catalog_id,
            version=record.version,
            relative_path=record.relative_path,
            size_bytes=record.size_bytes,
            sha256_hex=record.sha256_hex,
            stored_at=record.stored_at,
            last_used_at=_now_iso(),
            use_count=record.use_count + 1,
            pinned=record.pinned if pinned is None else bool(pinned),
            cache_class=record.cache_class if cache_class is None else str(cache_class),
            lease_status=record.lease_status if lease_status is None else str(lease_status),
            lease_expires_at=record.lease_expires_at if lease_expires_at is None else lease_expires_at,
        )
        records[index] = updated_record
        break
    if updated_record is not None:
        _write_chunk_store_records(records, policy)
    return updated_record


def delete_cached_chunk(
    chunk_id: str,
    policy: WalletPolicy | None = None,
) -> bool:
    records = _read_chunk_store_records(policy)
    kept: list[CachedChunkRecord] = []
    deleted = False
    for record in records:
        if record.chunk_id != chunk_id:
            kept.append(record)
            continue
        chunk_path = chunk_store_root(policy) / record.relative_path
        if chunk_path.exists():
            chunk_path.unlink()
        deleted = True
    if deleted:
        _write_chunk_store_records(kept, policy)
    return deleted


def chunk_store_snapshot(policy: WalletPolicy | None = None) -> ChunkStoreSnapshot:
    records = tuple(_read_chunk_store_records(policy))
    total_bytes = sum(record.size_bytes for record in records)
    pinned_records = [record for record in records if record.pinned]
    stats = ChunkStoreStats(
        chunk_count=len(records),
        total_bytes=total_bytes,
        pinned_chunk_count=len(pinned_records),
        pinned_bytes=sum(record.size_bytes for record in pinned_records),
        hot_chunk_count=sum(1 for record in records if record.cache_class == ChunkCacheClass.HOT),
        warm_chunk_count=sum(1 for record in records if record.cache_class == ChunkCacheClass.WARM),
        cold_chunk_count=sum(1 for record in records if record.cache_class == ChunkCacheClass.COLD),
    )
    return ChunkStoreSnapshot(records=records, stats=stats)


def _storage_accounting_id(
    *,
    node_id: str,
    chunk_id: str,
    accounted_from: str,
    accounted_until: str,
) -> str:
    payload = f"{node_id}|{chunk_id}|{accounted_from}|{accounted_until}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def summarize_chunk_storage_accounting(
    node_id: str,
    records: list[ChunkStorageAccountingRecord] | tuple[ChunkStorageAccountingRecord, ...] | None = None,
    policy: WalletPolicy | None = None,
) -> ChunkStorageAccountingSummary:
    active_records = tuple(records if records is not None else list_chunk_storage_accounting_records(node_id, policy))
    starts = [record.accounted_from for record in active_records]
    ends = [record.accounted_until for record in active_records]
    return ChunkStorageAccountingSummary(
        node_id=str(node_id),
        record_count=len(active_records),
        total_size_bytes=sum(record.size_bytes for record in active_records),
        total_accounted_seconds=sum(record.accounted_seconds for record in active_records),
        total_byte_seconds=sum(record.byte_seconds for record in active_records),
        period_started_at=min(starts) if starts else None,
        period_ended_at=max(ends) if ends else None,
    )


def record_chunk_storage_accounting_snapshot(
    node_id: str,
    *,
    accounting_policy: ChunkStorageAccountingPolicy | None = None,
    policy: WalletPolicy | None = None,
    now: datetime | None = None,
) -> ChunkStorageAccountingResult:
    active_accounting_policy = accounting_policy or ChunkStorageAccountingPolicy()
    now_dt = (now or datetime.now(tz=UTC)).astimezone(UTC)
    now_iso = now_dt.isoformat()
    existing = list_chunk_storage_accounting_records(node_id, policy)
    latest_until_by_chunk: dict[str, datetime] = {}
    for record in existing:
        accounted_until = _parse_iso_datetime(record.accounted_until)
        if accounted_until is None:
            continue
        if accounted_until.tzinfo is None:
            accounted_until = accounted_until.replace(tzinfo=UTC)
        accounted_until = accounted_until.astimezone(UTC)
        current = latest_until_by_chunk.get(record.chunk_id)
        if current is None or accounted_until > current:
            latest_until_by_chunk[record.chunk_id] = accounted_until

    new_records: list[ChunkStorageAccountingRecord] = []
    max_interval_seconds = max(1, int(active_accounting_policy.max_accounting_interval_seconds))
    min_accounting_seconds = max(0, int(active_accounting_policy.min_accounting_seconds))
    for chunk in list_cached_chunks(policy):
        stored_at = _parse_iso_datetime(chunk.stored_at)
        if stored_at is None:
            continue
        if stored_at.tzinfo is None:
            stored_at = stored_at.replace(tzinfo=UTC)
        stored_at = stored_at.astimezone(UTC)
        accounted_from = max(
            stored_at,
            latest_until_by_chunk.get(chunk.chunk_id, stored_at),
            now_dt - timedelta(seconds=max_interval_seconds),
        )
        accounted_seconds = int((now_dt - accounted_from).total_seconds())
        if accounted_seconds < min_accounting_seconds:
            continue
        byte_seconds = int(chunk.size_bytes) * accounted_seconds
        from_iso = accounted_from.isoformat()
        new_records.append(
            ChunkStorageAccountingRecord(
                accounting_id=_storage_accounting_id(
                    node_id=str(node_id),
                    chunk_id=chunk.chunk_id,
                    accounted_from=from_iso,
                    accounted_until=now_iso,
                ),
                created_at=now_iso,
                node_id=str(node_id),
                catalog_id=chunk.catalog_id,
                version=chunk.version,
                chunk_id=chunk.chunk_id,
                size_bytes=chunk.size_bytes,
                stored_at=chunk.stored_at,
                last_used_at=chunk.last_used_at,
                accounted_from=from_iso,
                accounted_until=now_iso,
                accounted_seconds=accounted_seconds,
                byte_seconds=byte_seconds,
                cache_class=chunk.cache_class,
                lease_status=chunk.lease_status,
                pinned=chunk.pinned,
            )
        )
    if new_records:
        existing.extend(new_records)
        _write_chunk_storage_accounting_records(node_id, existing, policy)
    return ChunkStorageAccountingResult(
        summary=summarize_chunk_storage_accounting(node_id, new_records, policy),
        records=tuple(new_records),
    )


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _inventory_payload_is_fresh(
    payload: ChunkInventoryPayload,
    *,
    fetch_policy: ChunkFetchPolicy,
) -> bool:
    max_age_seconds = max(0, int(fetch_policy.max_inventory_age_seconds))
    if max_age_seconds <= 0:
        return True
    published_at = _parse_iso_datetime(payload.published_at)
    if published_at is None:
        return False
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=UTC)
    age_seconds = (datetime.now(tz=UTC) - published_at.astimezone(UTC)).total_seconds()
    return age_seconds <= max_age_seconds


def _assignment_lease_expires_at(
    cache_policy: ChunkCachePolicy,
) -> str | None:
    if cache_policy.assignment_lease_seconds <= 0:
        return None
    return (
        datetime.now(tz=UTC).replace(microsecond=0)
        + timedelta(seconds=cache_policy.assignment_lease_seconds)
    ).isoformat()


def _chunk_eviction_priority(
    record: CachedChunkRecord,
    *,
    cache_policy: ChunkCachePolicy,
) -> tuple[int, int, float, float, int, str]:
    lease_expired = (
        record.lease_status == ChunkLeaseStatus.EXPIRED
        or (
            record.lease_expires_at is not None
            and (_parse_iso_datetime(record.lease_expires_at) or datetime.max.replace(tzinfo=UTC))
            <= datetime.now(tz=UTC)
        )
    )
    cache_rank = {
        ChunkCacheClass.COLD: 0,
        ChunkCacheClass.WARM: 1,
        ChunkCacheClass.HOT: 2,
    }.get(record.cache_class, 1)
    if cache_policy.protect_hot_chunks_when_possible and record.cache_class == ChunkCacheClass.HOT:
        cache_rank += 2

    last_used = _parse_iso_datetime(record.last_used_at)
    stored_at = _parse_iso_datetime(record.stored_at)
    last_used_ts = last_used.timestamp() if last_used is not None else 0.0
    stored_at_ts = stored_at.timestamp() if stored_at is not None else 0.0

    expired_rank = 0 if (cache_policy.evict_expired_leases_first and lease_expired) else 1
    active_lease_rank = 1 if record.lease_status == ChunkLeaseStatus.ACTIVE else 0
    return (
        expired_rank,
        cache_rank,
        active_lease_rank,
        last_used_ts,
        stored_at_ts,
        record.use_count,
        record.chunk_id,
    )


def evict_chunks_to_policy_target(
    *,
    cache_policy: ChunkCachePolicy | None = None,
    policy: WalletPolicy | None = None,
) -> ChunkStoreEvictionResult:
    active_cache_policy = cache_policy or ChunkCachePolicy()
    before = chunk_store_snapshot(policy)
    if before.stats.total_bytes <= active_cache_policy.max_store_bytes:
        return ChunkStoreEvictionResult(
            before=before,
            after=before,
            evicted_chunk_ids=(),
            evicted_bytes=0,
        )

    records = list(before.records)
    evictable: list[CachedChunkRecord] = []
    for record in records:
        if active_cache_policy.protect_pinned_chunks and record.pinned:
            continue
        if active_cache_policy.protect_active_leases and record.lease_status == ChunkLeaseStatus.ACTIVE:
            continue
        evictable.append(record)

    evictable.sort(
        key=lambda record: _chunk_eviction_priority(
            record,
            cache_policy=active_cache_policy,
        )
    )

    remaining_bytes = before.stats.total_bytes
    evicted_chunk_ids: list[str] = []
    evicted_bytes = 0
    for record in evictable:
        if remaining_bytes <= active_cache_policy.target_store_bytes:
            break
        if delete_cached_chunk(record.chunk_id, policy):
            remaining_bytes -= record.size_bytes
            evicted_bytes += record.size_bytes
            evicted_chunk_ids.append(record.chunk_id)

    after = chunk_store_snapshot(policy)
    return ChunkStoreEvictionResult(
        before=before,
        after=after,
        evicted_chunk_ids=tuple(evicted_chunk_ids),
        evicted_bytes=evicted_bytes,
    )


def apply_assignment_cache_policy_from_store(
    manifest: ModelPackageManifest,
    assignment: ModelShardAssignment,
    *,
    include_default_chunks: bool = True,
    cache_policy: ChunkCachePolicy | None = None,
    policy: WalletPolicy | None = None,
) -> tuple[CachedChunkRecord, ...]:
    active_cache_policy = cache_policy or ChunkCachePolicy()
    required_chunks = manifest.required_chunks_for_layers(
        assignment.start_layer,
        assignment.end_layer,
        include_default_chunks=include_default_chunks,
    )
    if not required_chunks:
        return ()

    lease_expires_at = _assignment_lease_expires_at(active_cache_policy)

    updated: list[CachedChunkRecord] = []
    for chunk in required_chunks:
        if get_cached_chunk_record(chunk.chunk_id, policy) is None:
            continue
        record = update_cached_chunk_record(
            chunk.chunk_id,
            pinned=active_cache_policy.pin_assignment_chunks,
            cache_class=ChunkCacheClass.HOT,
            lease_status=ChunkLeaseStatus.ACTIVE,
            lease_expires_at=lease_expires_at,
            policy=policy,
        )
        if record is not None:
            updated.append(record)
    return tuple(updated)


def release_assignment_cache_policy_from_store(
    manifest: ModelPackageManifest,
    assignment: ModelShardAssignment,
    *,
    include_default_chunks: bool = True,
    protected_chunk_ids: set[str] | list[str] | tuple[str, ...] | None = None,
    demoted_cache_class: str = ChunkCacheClass.WARM,
    policy: WalletPolicy | None = None,
) -> tuple[CachedChunkRecord, ...]:
    required_chunks = manifest.required_chunks_for_layers(
        assignment.start_layer,
        assignment.end_layer,
        include_default_chunks=include_default_chunks,
    )
    protected = {str(chunk_id) for chunk_id in (protected_chunk_ids or ())}
    released_at = datetime.now(tz=UTC).replace(microsecond=0).isoformat()
    updated: list[CachedChunkRecord] = []
    for chunk in required_chunks:
        if chunk.chunk_id in protected:
            continue
        if get_cached_chunk_record(chunk.chunk_id, policy) is None:
            continue
        record = update_cached_chunk_record(
            chunk.chunk_id,
            pinned=False,
            cache_class=demoted_cache_class,
            lease_status=ChunkLeaseStatus.EXPIRED,
            lease_expires_at=released_at,
            policy=policy,
        )
        if record is not None:
            updated.append(record)
    return tuple(updated)


def build_local_chunk_inventory_payload(
    source_id: str,
    *,
    source_kind: str = ChunkInventorySourceKind.LOCAL_CACHE,
    endpoint_base_url: str | None = None,
    policy: WalletPolicy | None = None,
) -> ChunkInventoryPayload:
    grouped: dict[tuple[str, str], list[CachedChunkRecord]] = {}
    for record in _read_chunk_store_records(policy):
        grouped.setdefault((record.catalog_id, record.version), []).append(record)

    records: list[ChunkInventoryRecord] = []
    for (catalog_id, version), items in sorted(grouped.items()):
        sorted_items = sorted(items, key=lambda item: item.chunk_id)
        records.append(
            ChunkInventoryRecord(
                catalog_id=catalog_id,
                version=version,
                chunk_ids=tuple(item.chunk_id for item in sorted_items),
                chunk_count=len(sorted_items),
                total_bytes=sum(item.size_bytes for item in sorted_items),
            )
        )

    return ChunkInventoryPayload(
        source_id=source_id,
        source_kind=source_kind,
        published_at=_now_iso(),
        endpoint_base_url=str(endpoint_base_url).strip().rstrip("/") if endpoint_base_url else None,
        records=tuple(records),
    )


def save_chunk_inventory_payload(
    payload: ChunkInventoryPayload,
    path: str | Path,
) -> Path:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(payload.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target_path


def load_chunk_inventory_payload(path: str | Path) -> ChunkInventoryPayload:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ChunkInventoryPayload.from_dict(payload)


def save_local_chunk_inventory_payload(
    source_id: str,
    *,
    source_kind: str = ChunkInventorySourceKind.LOCAL_CACHE,
    endpoint_base_url: str | None = None,
    policy: WalletPolicy | None = None,
) -> Path:
    payload = build_local_chunk_inventory_payload(
        source_id,
        source_kind=source_kind,
        endpoint_base_url=endpoint_base_url,
        policy=policy,
    )
    return save_chunk_inventory_payload(
        payload,
        local_chunk_inventory_path(source_id, policy),
    )


def export_chunk_inventory_payload(
    source_id: str,
    *,
    source_kind: str = ChunkInventorySourceKind.PEER_CACHE,
    endpoint_base_url: str | None = None,
    policy: WalletPolicy | None = None,
) -> ChunkInventoryPayload:
    return build_local_chunk_inventory_payload(
        source_id,
        source_kind=source_kind,
        endpoint_base_url=endpoint_base_url,
        policy=policy,
    )


def import_chunk_inventory_payload(
    payload: ChunkInventoryPayload,
    policy: WalletPolicy | None = None,
) -> Path:
    return save_chunk_inventory_payload(
        payload,
        imported_chunk_inventory_path(payload.source_id, payload.source_kind, policy),
    )


def delete_imported_chunk_inventory_payload(
    source_id: str,
    source_kind: str,
    policy: WalletPolicy | None = None,
) -> bool:
    path = imported_chunk_inventory_path(source_id, source_kind, policy)
    if not path.exists():
        return False
    path.unlink()
    return True


def list_imported_chunk_inventory_payloads(
    *,
    source_kind: str | None = None,
    policy: WalletPolicy | None = None,
) -> list[ChunkInventoryPayload]:
    if source_kind is None:
        paths = sorted((chunk_inventory_root(policy) / "imported").glob("*/*.json"))
    else:
        paths = sorted(imported_chunk_inventory_dir(source_kind, policy).glob("*.json"))
    payloads: list[ChunkInventoryPayload] = []
    for path in paths:
        try:
            payloads.append(load_chunk_inventory_payload(path))
        except Exception:
            continue
    return payloads


def prune_imported_chunk_inventory_payloads(
    *,
    allowed_source_ids: set[str] | list[str] | tuple[str, ...] | None = None,
    allowed_endpoint_base_urls: set[str] | list[str] | tuple[str, ...] | None = None,
    source_kind: str | None = None,
    policy: WalletPolicy | None = None,
) -> tuple[str, ...]:
    allowed_ids = {
        str(source_id).strip()
        for source_id in (allowed_source_ids or ())
        if str(source_id).strip()
    }
    normalized_allowed_endpoint_base_urls = {
        str(endpoint_base_url).strip().rstrip("/")
        for endpoint_base_url in (allowed_endpoint_base_urls or ())
        if str(endpoint_base_url).strip()
    }
    removed: list[str] = []
    for payload in list_imported_chunk_inventory_payloads(
        source_kind=source_kind,
        policy=policy,
    ):
        if allowed_ids and payload.source_id not in allowed_ids:
            should_prune = True
        else:
            should_prune = False
        if normalized_allowed_endpoint_base_urls:
            payload_endpoint_base_url = (
                str(payload.endpoint_base_url).strip().rstrip("/")
                if payload.endpoint_base_url
                else ""
            )
            if payload_endpoint_base_url not in normalized_allowed_endpoint_base_urls:
                should_prune = True
        if not should_prune:
            continue
        if delete_imported_chunk_inventory_payload(
            payload.source_id,
            payload.source_kind,
            policy,
        ):
            removed.append(payload.source_id)
    return tuple(removed)


def build_chunk_inventory_endpoint_url(
    base_url_or_inventory_url: str,
    *,
    source_kind: str = ChunkInventorySourceKind.PEER_CACHE,
) -> str:
    raw = str(base_url_or_inventory_url or "").strip()
    if not raw:
        raise ValueError("Chunk inventory base URL cannot be empty.")
    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid chunk inventory URL: {raw}")
    normalized_source_kind = (
        str(source_kind or "").strip() or ChunkInventorySourceKind.PEER_CACHE
    )
    base_path = parsed.path.rstrip("/")
    existing_query = {
        key: value
        for key, value in parse_qsl(parsed.query, keep_blank_values=False)
        if str(key).strip()
    }
    existing_query["source_kind"] = normalized_source_kind
    if base_path.endswith("/v1/cai/chunk-inventory"):
        endpoint_path = base_path
    else:
        endpoint_path = f"{base_path}/v1/cai/chunk-inventory"
    return (
        f"{parsed.scheme}://{parsed.netloc}{endpoint_path}"
        f"?{urlencode(existing_query)}"
    )


def normalize_chunk_inventory_sync_urls(
    inventory_urls: list[str] | tuple[str, ...] | set[str],
    *,
    source_kind: str = ChunkInventorySourceKind.PEER_CACHE,
) -> tuple[str, ...]:
    normalized_urls: list[str] = []
    seen: set[str] = set()
    for item in inventory_urls:
        raw = str(item or "").strip()
        if not raw:
            continue
        normalized = build_chunk_inventory_endpoint_url(
            raw,
            source_kind=source_kind,
        )
        if normalized in seen:
            continue
        seen.add(normalized)
        normalized_urls.append(normalized)
    return tuple(normalized_urls)


def inventory_endpoint_base_urls(
    inventory_urls: list[str] | tuple[str, ...] | set[str],
    *,
    source_kind: str = ChunkInventorySourceKind.PEER_CACHE,
) -> tuple[str, ...]:
    base_urls: list[str] = []
    seen: set[str] = set()
    for normalized_inventory_url in normalize_chunk_inventory_sync_urls(
        inventory_urls,
        source_kind=source_kind,
    ):
        parsed = urlparse(normalized_inventory_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        if base_url in seen:
            continue
        seen.add(base_url)
        base_urls.append(base_url)
    return tuple(base_urls)


def _peer_error_payload(peer_url: str, exc: Exception) -> dict[str, str]:
    return {
        "peerUrl": peer_url,
        "errorType": type(exc).__name__,
        "message": str(exc),
    }


def sync_chunk_inventory_from_urls(
    *,
    inventory_urls: list[str] | tuple[str, ...] | set[str],
    source_kind: str = ChunkInventorySourceKind.PEER_CACHE,
    policy: WalletPolicy | None = None,
    timeout_sec: int = 5,
    prune_missing_source_ids: set[str] | list[str] | tuple[str, ...] | None = None,
    prune_missing_endpoint_base_urls: set[str] | list[str] | tuple[str, ...] | None = None,
) -> ChunkInventorySyncResult:
    normalized_source_kind = (
        str(source_kind or "").strip() or ChunkInventorySourceKind.PEER_CACHE
    )
    normalized_inventory_urls = normalize_chunk_inventory_sync_urls(
        inventory_urls,
        source_kind=normalized_source_kind,
    )
    imported_payloads = 0
    successful_peers = 0
    pruned_payloads = 0
    failed_peer_urls: list[str] = []
    peer_errors: list[dict[str, str]] = []
    for inventory_url in normalized_inventory_urls:
        try:
            with urlopen(inventory_url, timeout=timeout_sec) as response:
                payload = ChunkInventoryPayload.from_dict(
                    json.loads(response.read().decode("utf-8"))
                )
            if not payload.endpoint_base_url:
                parsed = urlparse(inventory_url)
                if parsed.scheme and parsed.netloc:
                    payload = ChunkInventoryPayload(
                        source_id=payload.source_id,
                        source_kind=payload.source_kind,
                        published_at=payload.published_at,
                        endpoint_base_url=f"{parsed.scheme}://{parsed.netloc}",
                        records=payload.records,
                    )
            import_chunk_inventory_payload(payload, policy)
            imported_payloads += 1
            successful_peers += 1
        except Exception as exc:
            failed_peer_urls.append(inventory_url)
            peer_errors.append(_peer_error_payload(inventory_url, exc))
            continue

    if prune_missing_source_ids or prune_missing_endpoint_base_urls:
        pruned_payloads = len(
            prune_imported_chunk_inventory_payloads(
                allowed_source_ids=prune_missing_source_ids,
                allowed_endpoint_base_urls=prune_missing_endpoint_base_urls,
                source_kind=normalized_source_kind,
                policy=policy,
            )
        )
        prune_chunk_source_health_records(
            allowed_source_ids=prune_missing_source_ids,
            allowed_locators=prune_missing_endpoint_base_urls,
            source_kind=normalized_source_kind,
            policy=policy,
        )

    return ChunkInventorySyncResult(
        attempted_peers=len(normalized_inventory_urls),
        successful_peers=successful_peers,
        imported_payloads=imported_payloads,
        pruned_payloads=pruned_payloads,
        peer_urls=list(normalized_inventory_urls),
        failed_peers=len(failed_peer_urls),
        failed_peer_urls=failed_peer_urls,
        peer_errors=peer_errors,
    )


def sync_chunk_inventory_from_cai_peers(
    *,
    state_payload: dict[str, Any],
    cai_url: str,
    source_kind: str = ChunkInventorySourceKind.PEER_CACHE,
    policy: WalletPolicy | None = None,
    local_node_id: str | None = None,
    timeout_sec: int = 5,
    prune_missing_peers: bool = True,
) -> ChunkInventorySyncResult:
    from .validators import discover_peer_cai_urls

    normalized_source_kind = str(source_kind or "").strip() or ChunkInventorySourceKind.PEER_CACHE
    peer_urls = discover_peer_cai_urls(
        state_payload=state_payload,
        cai_url=cai_url,
        endpoint_path=f"/v1/cai/chunk-inventory?source_kind={normalized_source_kind}",
        local_node_id=local_node_id,
    )
    return sync_chunk_inventory_from_urls(
        inventory_urls=peer_urls,
        source_kind=normalized_source_kind,
        policy=policy,
        timeout_sec=timeout_sec,
        prune_missing_source_ids=(
            {
                str(peer_id).strip()
                for peer_id in (state_payload.get("nodeIdentities") or {}).keys()
                if str(peer_id).strip()
            }
            if prune_missing_peers
            and normalized_source_kind == ChunkInventorySourceKind.PEER_CACHE
            else None
        ),
    )

def build_chunk_inventory_index(
    catalog_id: str,
    version: str,
    *,
    source_kind: str | None = None,
    fetch_policy: ChunkFetchPolicy | None = None,
    policy: WalletPolicy | None = None,
) -> dict[str, set[str]]:
    active_fetch_policy = fetch_policy or ChunkFetchPolicy()
    index: dict[str, set[str]] = {}
    for payload in list_imported_chunk_inventory_payloads(
        source_kind=source_kind,
        policy=policy,
    ):
        if not _inventory_payload_is_fresh(payload, fetch_policy=active_fetch_policy):
            continue
        chunk_ids = payload.chunk_ids_by_manifest(catalog_id, version)
        if chunk_ids:
            index[payload.source_id] = chunk_ids
    return index


def build_chunk_inventory_locator_index(
    catalog_id: str,
    version: str,
    *,
    source_kind: str | None = None,
    fetch_policy: ChunkFetchPolicy | None = None,
    policy: WalletPolicy | None = None,
) -> dict[str, str]:
    active_fetch_policy = fetch_policy or ChunkFetchPolicy()
    locators: dict[str, str] = {}
    for payload in list_imported_chunk_inventory_payloads(
        source_kind=source_kind,
        policy=policy,
    ):
        if not _inventory_payload_is_fresh(payload, fetch_policy=active_fetch_policy):
            continue
        if not payload.endpoint_base_url:
            continue
        if payload.chunk_ids_by_manifest(catalog_id, version):
            locators[payload.source_id] = payload.endpoint_base_url.rstrip("/")
    return locators


def fresh_imported_manifest_refs(
    *,
    source_kind: str | None = None,
    fetch_policy: ChunkFetchPolicy | None = None,
    policy: WalletPolicy | None = None,
) -> tuple[tuple[str, str], ...]:
    active_fetch_policy = fetch_policy or ChunkFetchPolicy()
    latest_published_at: dict[tuple[str, str], datetime] = {}
    for payload in list_imported_chunk_inventory_payloads(
        source_kind=source_kind,
        policy=policy,
    ):
        if not _inventory_payload_is_fresh(payload, fetch_policy=active_fetch_policy):
            continue
        published_at = _parse_iso_datetime(payload.published_at)
        if published_at is None:
            continue
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=UTC)
        published_at = published_at.astimezone(UTC)
        for record in payload.records:
            key = (record.catalog_id, record.version)
            if key not in latest_published_at or published_at > latest_published_at[key]:
                latest_published_at[key] = published_at
    return tuple(
        key
        for key, _published_at in sorted(
            latest_published_at.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )


def prioritize_manifest_refs_for_node(
    manifest_refs: tuple[tuple[str, str], ...] | list[tuple[str, str]],
    *,
    node_id: str | None = None,
    fetch_policy: ChunkFetchPolicy | None = None,
    policy: WalletPolicy | None = None,
) -> tuple[tuple[str, str], ...]:
    ordered_refs = tuple(manifest_refs)
    normalized_node_id = str(node_id or "").strip()
    if not normalized_node_id or not ordered_refs:
        return ordered_refs

    active_fetch_policy = fetch_policy or ChunkFetchPolicy()
    recent_hints = list_recent_shard_hints(
        normalized_node_id,
        fetch_policy=active_fetch_policy,
        policy=policy,
    )
    if not recent_hints:
        return ordered_refs

    recent_model_rank: dict[str, tuple[int, int]] = {}
    for index, record in enumerate(recent_hints):
        model_id = str(record.model_id)
        weight = (index, -int(record.use_count))
        if model_id not in recent_model_rank or weight < recent_model_rank[model_id]:
            recent_model_rank[model_id] = weight

    decorated: list[tuple[tuple[int, int], int, tuple[str, str]]] = []
    for original_index, ref in enumerate(ordered_refs):
        try:
            manifest = load_model_package_manifest(ref[0], ref[1], policy)
        except FileNotFoundError:
            decorated.append(((10_000, 0), original_index, ref))
            continue
        priority = recent_model_rank.get(str(manifest.model_id), (10_000, 0))
        decorated.append((priority, original_index, ref))

    decorated.sort(key=lambda item: (item[0][0], item[0][1], item[1]))
    return tuple(item[2] for item in decorated)


def present_chunk_ids_for_manifest(
    catalog_id: str,
    version: str,
    policy: WalletPolicy | None = None,
    *,
    verify_cached_payloads: bool = False,
) -> set[str]:
    output: set[str] = set()
    for record in _read_chunk_store_records(policy):
        if record.catalog_id != catalog_id or record.version != version:
            continue
        if not _cached_chunk_record_is_present(
            record,
            verify_payload=verify_cached_payloads,
            policy=policy,
        ):
            continue
        output.add(record.chunk_id)
    return output


def _cached_chunk_record_is_present(
    record: CachedChunkRecord,
    *,
    verify_payload: bool,
    policy: WalletPolicy | None = None,
) -> bool:
    payload_path = chunk_store_root(policy) / Path(record.relative_path)
    if not payload_path.exists() or not payload_path.is_file():
        return False
    try:
        if int(payload_path.stat().st_size) != int(record.size_bytes):
            return False
        if verify_payload:
            payload_hash = hashlib.sha256(payload_path.read_bytes()).hexdigest()
            if payload_hash != str(record.sha256_hex).strip().lower():
                return False
    except OSError:
        return False
    return True


def default_chunk_assignment_context(
    *,
    node_id: str | None = None,
    device_rank: int = 0,
    world_size: int = 1,
) -> ModelShardAssignment:
    return ModelShardAssignment(
        start_layer=0,
        end_layer=1,
        device_rank=device_rank,
        world_size=world_size,
        node_id=node_id,
    )


def build_default_chunk_plan_from_store(
    manifest: ModelPackageManifest,
    *,
    node_id: str | None = None,
    device_rank: int = 0,
    world_size: int = 1,
    policy: WalletPolicy | None = None,
) -> AssignmentChunkPlan:
    assignment = default_chunk_assignment_context(
        node_id=node_id,
        device_rank=device_rank,
        world_size=world_size,
    )
    coverage = manifest.compute_default_chunk_coverage(
        present_chunk_ids_for_manifest(
            manifest.catalog_id,
            manifest.version,
            policy=policy,
        )
    )
    return AssignmentChunkPlan(
        assignment=assignment,
        coverage=coverage,
        estimated_fetch_bytes=sum(
            chunk.size_bytes
            for chunk in manifest.default_chunks()
            if chunk.chunk_id in coverage.missing_chunk_ids
        ),
    )


def build_bootstrap_chunk_plan_from_store(
    manifest: ModelPackageManifest,
    *,
    node_id: str | None = None,
    device_rank: int = 0,
    world_size: int = 1,
    max_weight_chunks: int = 1,
    max_weight_bytes: int | None = None,
    include_default_chunks: bool = True,
    hint_start_layer: int | None = None,
    hint_end_layer: int | None = None,
    policy: WalletPolicy | None = None,
) -> AssignmentChunkPlan:
    assignment = default_chunk_assignment_context(
        node_id=node_id,
        device_rank=device_rank,
        world_size=world_size,
    )
    coverage = manifest.compute_bootstrap_chunk_coverage(
        present_chunk_ids_for_manifest(
            manifest.catalog_id,
            manifest.version,
            policy=policy,
        ),
        max_weight_chunks=max_weight_chunks,
        max_weight_bytes=max_weight_bytes,
        include_default_chunks=include_default_chunks,
        hint_start_layer=hint_start_layer,
        hint_end_layer=hint_end_layer,
    )
    return AssignmentChunkPlan(
        assignment=assignment,
        coverage=coverage,
        estimated_fetch_bytes=sum(
            chunk.size_bytes
            for chunk in manifest.bootstrap_prefetch_chunks(
                max_weight_chunks=max_weight_chunks,
                max_weight_bytes=max_weight_bytes,
                include_default_chunks=include_default_chunks,
                hint_start_layer=hint_start_layer,
                hint_end_layer=hint_end_layer,
            )
            if chunk.chunk_id in coverage.missing_chunk_ids
        ),
    )


def build_assignment_chunk_plan_from_store(
    manifest: ModelPackageManifest,
    assignment: ModelShardAssignment,
    *,
    include_default_chunks: bool = True,
    verify_cached_chunks: bool = False,
    policy: WalletPolicy | None = None,
) -> AssignmentChunkPlan:
    present_chunk_ids = present_chunk_ids_for_manifest(
        manifest.catalog_id,
        manifest.version,
        policy,
        verify_cached_payloads=verify_cached_chunks,
    )
    return manifest.build_assignment_chunk_plan(
        assignment,
        present_chunk_ids,
        include_default_chunks=include_default_chunks,
    )


def build_assignment_fetch_plan_from_store(
    manifest: ModelPackageManifest,
    assignment: ModelShardAssignment,
    *,
    peer_chunk_inventory: dict[str, set[str] | list[str] | tuple[str, ...]] | None = None,
    seed_chunk_inventory: dict[str, set[str] | list[str] | tuple[str, ...]] | None = None,
    peer_chunk_locators: dict[str, str] | None = None,
    seed_chunk_locators: dict[str, str] | None = None,
    use_imported_peer_inventory: bool = False,
    use_imported_seed_inventory: bool = False,
    include_default_chunks: bool = True,
    verify_cached_chunks: bool = False,
    fetch_policy: ChunkFetchPolicy | None = None,
    policy: WalletPolicy | None = None,
) -> AssignmentFetchPlan:
    present_chunk_ids = present_chunk_ids_for_manifest(
        manifest.catalog_id,
        manifest.version,
        policy,
        verify_cached_payloads=verify_cached_chunks,
    )
    resolved_peer_inventory = (
        build_chunk_inventory_index(
            manifest.catalog_id,
            manifest.version,
            source_kind=ChunkInventorySourceKind.PEER_CACHE,
            fetch_policy=fetch_policy,
            policy=policy,
        )
        if use_imported_peer_inventory and peer_chunk_inventory is None
        else peer_chunk_inventory
    )
    resolved_seed_inventory = (
        build_chunk_inventory_index(
            manifest.catalog_id,
            manifest.version,
            source_kind=ChunkInventorySourceKind.STORAGE_SEED,
            fetch_policy=fetch_policy,
            policy=policy,
        )
        if use_imported_seed_inventory and seed_chunk_inventory is None
        else seed_chunk_inventory
    )
    resolved_peer_locators = (
        build_chunk_inventory_locator_index(
            manifest.catalog_id,
            manifest.version,
            source_kind=ChunkInventorySourceKind.PEER_CACHE,
            fetch_policy=fetch_policy,
            policy=policy,
        )
        if use_imported_peer_inventory and peer_chunk_locators is None
        else peer_chunk_locators
    )
    resolved_seed_locators = (
        build_chunk_inventory_locator_index(
            manifest.catalog_id,
            manifest.version,
            source_kind=ChunkInventorySourceKind.STORAGE_SEED,
            fetch_policy=fetch_policy,
            policy=policy,
        )
        if use_imported_seed_inventory and seed_chunk_locators is None
        else seed_chunk_locators
    )
    fetch_plan = manifest.build_assignment_fetch_plan(
        assignment,
        present_chunk_ids,
        peer_chunk_inventory=resolved_peer_inventory,
        seed_chunk_inventory=resolved_seed_inventory,
        peer_chunk_locators=resolved_peer_locators,
        seed_chunk_locators=resolved_seed_locators,
        include_default_chunks=include_default_chunks,
    )
    return adapt_assignment_fetch_plan_to_source_health(
        fetch_plan,
        fetch_policy=fetch_policy,
        policy=policy,
    )


def build_default_chunk_fetch_plan_from_store(
    manifest: ModelPackageManifest,
    *,
    node_id: str | None = None,
    device_rank: int = 0,
    world_size: int = 1,
    peer_chunk_inventory: dict[str, set[str] | list[str] | tuple[str, ...]] | None = None,
    seed_chunk_inventory: dict[str, set[str] | list[str] | tuple[str, ...]] | None = None,
    peer_chunk_locators: dict[str, str] | None = None,
    seed_chunk_locators: dict[str, str] | None = None,
    use_imported_peer_inventory: bool = False,
    use_imported_seed_inventory: bool = False,
    fetch_policy: ChunkFetchPolicy | None = None,
    policy: WalletPolicy | None = None,
) -> AssignmentFetchPlan:
    assignment = default_chunk_assignment_context(
        node_id=node_id,
        device_rank=device_rank,
        world_size=world_size,
    )
    present_chunk_ids = present_chunk_ids_for_manifest(
        manifest.catalog_id,
        manifest.version,
        policy=policy,
    )
    resolved_peer_inventory = (
        build_chunk_inventory_index(
            manifest.catalog_id,
            manifest.version,
            source_kind=ChunkInventorySourceKind.PEER_CACHE,
            fetch_policy=fetch_policy,
            policy=policy,
        )
        if use_imported_peer_inventory and peer_chunk_inventory is None
        else peer_chunk_inventory
    )
    resolved_seed_inventory = (
        build_chunk_inventory_index(
            manifest.catalog_id,
            manifest.version,
            source_kind=ChunkInventorySourceKind.STORAGE_SEED,
            fetch_policy=fetch_policy,
            policy=policy,
        )
        if use_imported_seed_inventory and seed_chunk_inventory is None
        else seed_chunk_inventory
    )
    resolved_peer_locators = (
        build_chunk_inventory_locator_index(
            manifest.catalog_id,
            manifest.version,
            source_kind=ChunkInventorySourceKind.PEER_CACHE,
            fetch_policy=fetch_policy,
            policy=policy,
        )
        if use_imported_peer_inventory and peer_chunk_locators is None
        else peer_chunk_locators
    )
    resolved_seed_locators = (
        build_chunk_inventory_locator_index(
            manifest.catalog_id,
            manifest.version,
            source_kind=ChunkInventorySourceKind.STORAGE_SEED,
            fetch_policy=fetch_policy,
            policy=policy,
        )
        if use_imported_seed_inventory and seed_chunk_locators is None
        else seed_chunk_locators
    )
    fetch_plan = manifest.build_default_chunk_fetch_plan(
        assignment,
        present_chunk_ids,
        peer_chunk_inventory=resolved_peer_inventory,
        seed_chunk_inventory=resolved_seed_inventory,
        peer_chunk_locators=resolved_peer_locators,
        seed_chunk_locators=resolved_seed_locators,
    )
    return adapt_assignment_fetch_plan_to_source_health(
        fetch_plan,
        fetch_policy=fetch_policy,
        policy=policy,
    )


def build_bootstrap_chunk_fetch_plan_from_store(
    manifest: ModelPackageManifest,
    *,
    node_id: str | None = None,
    device_rank: int = 0,
    world_size: int = 1,
    peer_chunk_inventory: dict[str, set[str] | list[str] | tuple[str, ...]] | None = None,
    seed_chunk_inventory: dict[str, set[str] | list[str] | tuple[str, ...]] | None = None,
    peer_chunk_locators: dict[str, str] | None = None,
    seed_chunk_locators: dict[str, str] | None = None,
    use_imported_peer_inventory: bool = False,
    use_imported_seed_inventory: bool = False,
    max_weight_chunks: int = 1,
    max_weight_bytes: int | None = None,
    include_default_chunks: bool = True,
    hint_start_layer: int | None = None,
    hint_end_layer: int | None = None,
    fetch_policy: ChunkFetchPolicy | None = None,
    policy: WalletPolicy | None = None,
) -> AssignmentFetchPlan:
    assignment = default_chunk_assignment_context(
        node_id=node_id,
        device_rank=device_rank,
        world_size=world_size,
    )
    present_chunk_ids = present_chunk_ids_for_manifest(
        manifest.catalog_id,
        manifest.version,
        policy=policy,
    )
    resolved_peer_inventory = (
        build_chunk_inventory_index(
            manifest.catalog_id,
            manifest.version,
            source_kind=ChunkInventorySourceKind.PEER_CACHE,
            fetch_policy=fetch_policy,
            policy=policy,
        )
        if use_imported_peer_inventory and peer_chunk_inventory is None
        else peer_chunk_inventory
    )
    resolved_seed_inventory = (
        build_chunk_inventory_index(
            manifest.catalog_id,
            manifest.version,
            source_kind=ChunkInventorySourceKind.STORAGE_SEED,
            fetch_policy=fetch_policy,
            policy=policy,
        )
        if use_imported_seed_inventory and seed_chunk_inventory is None
        else seed_chunk_inventory
    )
    resolved_peer_locators = (
        build_chunk_inventory_locator_index(
            manifest.catalog_id,
            manifest.version,
            source_kind=ChunkInventorySourceKind.PEER_CACHE,
            fetch_policy=fetch_policy,
            policy=policy,
        )
        if use_imported_peer_inventory and peer_chunk_locators is None
        else peer_chunk_locators
    )
    resolved_seed_locators = (
        build_chunk_inventory_locator_index(
            manifest.catalog_id,
            manifest.version,
            source_kind=ChunkInventorySourceKind.STORAGE_SEED,
            fetch_policy=fetch_policy,
            policy=policy,
        )
        if use_imported_seed_inventory and seed_chunk_locators is None
        else seed_chunk_locators
    )
    fetch_plan = manifest.build_bootstrap_chunk_fetch_plan(
        assignment,
        present_chunk_ids,
        peer_chunk_inventory=resolved_peer_inventory,
        seed_chunk_inventory=resolved_seed_inventory,
        peer_chunk_locators=resolved_peer_locators,
        seed_chunk_locators=resolved_seed_locators,
        max_weight_chunks=max_weight_chunks,
        max_weight_bytes=max_weight_bytes,
        include_default_chunks=include_default_chunks,
        hint_start_layer=hint_start_layer,
        hint_end_layer=hint_end_layer,
    )
    return adapt_assignment_fetch_plan_to_source_health(
        fetch_plan,
        fetch_policy=fetch_policy,
        policy=policy,
    )


def build_chunk_download_tasks_from_fetch_plan(
    manifest: ModelPackageManifest,
    fetch_plan: AssignmentFetchPlan,
) -> list[ChunkDownloadTask]:
    now = _now_iso()
    return [
        ChunkDownloadTask(
            task_id=hashlib.sha256(
                f"{manifest.catalog_id}:{manifest.version}:{request.chunk_id}".encode("utf-8")
            ).hexdigest()[:24],
            catalog_id=manifest.catalog_id,
            version=manifest.version,
            chunk_id=request.chunk_id,
            artifact_id=request.artifact_id,
            size_bytes=request.size_bytes,
            sha256_hex=request.sha256_hex,
            status=ChunkDownloadTaskStatus.QUEUED,
            created_at=now,
            updated_at=now,
            assignment_start_layer=fetch_plan.assignment.start_layer,
            assignment_end_layer=fetch_plan.assignment.end_layer,
            assignment_device_rank=fetch_plan.assignment.device_rank,
            assignment_world_size=fetch_plan.assignment.world_size,
            node_id=fetch_plan.assignment.node_id,
            sources=request.sources,
            selected_source_kind=request.sources[0].kind if request.sources else None,
            selected_source_id=request.sources[0].source_id if request.sources else None,
        )
        for request in fetch_plan.fetch_requests
    ]


def queue_assignment_fetch_plan(
    manifest: ModelPackageManifest,
    fetch_plan: AssignmentFetchPlan,
    policy: WalletPolicy | None = None,
) -> list[ChunkDownloadTask]:
    queued_tasks = _read_chunk_download_tasks(policy)
    queued_by_key = {
        (task.catalog_id, task.version, task.chunk_id): task
        for task in queued_tasks
    }
    result: list[ChunkDownloadTask] = []
    for task in build_chunk_download_tasks_from_fetch_plan(manifest, fetch_plan):
        key = (task.catalog_id, task.version, task.chunk_id)
        existing = queued_by_key.get(key)
        if existing is None:
            queued_tasks.append(task)
            queued_by_key[key] = task
            result.append(task)
            continue

        merged_sources = list(existing.sources)
        for source in task.sources:
            if source not in merged_sources:
                merged_sources.append(source)
        selected_source_kind = existing.selected_source_kind
        selected_source_id = existing.selected_source_id
        if selected_source_kind is None and merged_sources:
            selected_source_kind = merged_sources[0].kind
            selected_source_id = merged_sources[0].source_id
        updated = ChunkDownloadTask(
            task_id=existing.task_id,
            catalog_id=existing.catalog_id,
            version=existing.version,
            chunk_id=existing.chunk_id,
            artifact_id=existing.artifact_id,
            size_bytes=existing.size_bytes,
            sha256_hex=existing.sha256_hex,
            status=existing.status,
            created_at=existing.created_at,
            updated_at=_now_iso(),
            assignment_start_layer=min(existing.assignment_start_layer, task.assignment_start_layer),
            assignment_end_layer=max(existing.assignment_end_layer, task.assignment_end_layer),
            assignment_device_rank=existing.assignment_device_rank,
            assignment_world_size=max(existing.assignment_world_size, task.assignment_world_size),
            node_id=existing.node_id or task.node_id,
            sources=tuple(merged_sources),
            selected_source_kind=selected_source_kind,
            selected_source_id=selected_source_id,
            attempt_count=existing.attempt_count,
            last_error=existing.last_error,
        )
        queued_by_key[key] = updated
        for index, queued_task in enumerate(queued_tasks):
            if queued_task.task_id == existing.task_id:
                queued_tasks[index] = updated
                break
        result.append(updated)

    _write_chunk_download_tasks(queued_tasks, policy)
    return result


def list_chunk_download_tasks(policy: WalletPolicy | None = None) -> list[ChunkDownloadTask]:
    return _read_chunk_download_tasks(policy)


def update_chunk_download_task_status(
    task_id: str,
    status: str,
    *,
    selected_source_kind: str | None = None,
    selected_source_id: str | None = None,
    last_error: str | None = None,
    policy: WalletPolicy | None = None,
) -> ChunkDownloadTask | None:
    tasks = _read_chunk_download_tasks(policy)
    updated_task: ChunkDownloadTask | None = None
    for index, task in enumerate(tasks):
        if task.task_id != task_id:
            continue
        updated_task = ChunkDownloadTask(
            task_id=task.task_id,
            catalog_id=task.catalog_id,
            version=task.version,
            chunk_id=task.chunk_id,
            artifact_id=task.artifact_id,
            size_bytes=task.size_bytes,
            sha256_hex=task.sha256_hex,
            status=status,
            created_at=task.created_at,
            updated_at=_now_iso(),
            assignment_start_layer=task.assignment_start_layer,
            assignment_end_layer=task.assignment_end_layer,
            assignment_device_rank=task.assignment_device_rank,
            assignment_world_size=task.assignment_world_size,
            node_id=task.node_id,
            sources=task.sources,
            selected_source_kind=selected_source_kind or task.selected_source_kind,
            selected_source_id=selected_source_id or task.selected_source_id,
            attempt_count=task.attempt_count + 1,
            last_error=last_error,
        )
        tasks[index] = updated_task
        break
    if updated_task is not None:
        _write_chunk_download_tasks(tasks, policy)
    return updated_task


def chunk_download_queue_snapshot(
    policy: WalletPolicy | None = None,
) -> ChunkDownloadQueueSnapshot:
    tasks = tuple(_read_chunk_download_tasks(policy))
    stats = ChunkDownloadQueueStats(
        task_count=len(tasks),
        queued_count=sum(1 for task in tasks if task.status == ChunkDownloadTaskStatus.QUEUED),
        in_progress_count=sum(1 for task in tasks if task.status == ChunkDownloadTaskStatus.IN_PROGRESS),
        completed_count=sum(1 for task in tasks if task.status == ChunkDownloadTaskStatus.COMPLETED),
        failed_count=sum(1 for task in tasks if task.status == ChunkDownloadTaskStatus.FAILED),
        total_bytes=sum(task.size_bytes for task in tasks),
        queued_bytes=sum(task.size_bytes for task in tasks if task.status == ChunkDownloadTaskStatus.QUEUED),
        completed_bytes=sum(task.size_bytes for task in tasks if task.status == ChunkDownloadTaskStatus.COMPLETED),
    )
    return ChunkDownloadQueueSnapshot(tasks=tasks, stats=stats)


def _read_bound_artifact_bytes(
    manifest: ModelPackageManifest,
    *,
    artifact_id: str,
    offset_bytes: int,
    size_bytes: int,
    policy: WalletPolicy | None = None,
) -> bytes:
    bindings = load_local_artifact_bindings(manifest.catalog_id, manifest.version, policy)
    binding_map = {binding.artifact_id: binding for binding in bindings.bindings}
    binding = binding_map.get(artifact_id)
    if binding is None:
        raise FileNotFoundError(
            f"No local artifact binding found for {manifest.catalog_id}@{manifest.version}:{artifact_id}"
        )
    artifact_path = Path(binding.local_path)
    if not artifact_path.exists() or not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    with artifact_path.open("rb") as handle:
        handle.seek(offset_bytes)
        payload = handle.read(size_bytes)
    if len(payload) != size_bytes:
        raise IOError(
            f"Expected {size_bytes} bytes from {artifact_path}, got {len(payload)}."
        )
    return payload


def _huggingface_resolve_url(
    *,
    repo_id: str,
    revision: str | None,
    relative_path: str,
) -> str:
    normalized_repo_id = str(repo_id or "").strip().rstrip("/")
    normalized_revision = str(revision or "main").strip() or "main"
    normalized_relative_path = str(relative_path or "").replace("\\", "/").strip("/")
    if not normalized_repo_id:
        raise ValueError("Hugging Face repo id is required for origin chunk fetch.")
    if not normalized_relative_path:
        raise ValueError("Artifact relative path is required for origin chunk fetch.")

    revision_segment = quote(normalized_revision, safe="")
    path_segment = quote(normalized_relative_path, safe="/")
    if normalized_repo_id.lower().startswith(("http://", "https://")):
        parsed = urlparse(normalized_repo_id)
        if "/resolve/" in parsed.path:
            return normalized_repo_id
        return f"{normalized_repo_id}/resolve/{revision_segment}/{path_segment}"

    repo_segment = quote(normalized_repo_id, safe="/")
    return f"https://huggingface.co/{repo_segment}/resolve/{revision_segment}/{path_segment}"


def _response_status_code(response: Any) -> int | None:
    status = getattr(response, "status", None)
    if status is not None:
        try:
            return int(status)
        except (TypeError, ValueError):
            return None
    getcode = getattr(response, "getcode", None)
    if callable(getcode):
        try:
            value = getcode()
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None
    return None


def _download_origin_artifact_range(
    manifest: ModelPackageManifest,
    *,
    artifact_id: str,
    offset_bytes: int,
    size_bytes: int,
    timeout_sec: int = 30,
) -> bytes:
    artifact = next(
        (item for item in manifest.files if item.artifact_id == artifact_id),
        None,
    )
    if artifact is None:
        raise FileNotFoundError(
            f"Unknown artifact_id for origin chunk fetch: {artifact_id}"
        )

    repo_id = artifact.source_repo_id or manifest.source_repo_id
    if repo_id is None:
        raise FileNotFoundError(
            f"No origin repo configured for {manifest.catalog_id}@{manifest.version}:{artifact_id}"
        )

    revision = artifact.source_revision or manifest.source_revision or "main"
    artifact_url = _huggingface_resolve_url(
        repo_id=repo_id,
        revision=revision,
        relative_path=artifact.relative_path,
    )
    end_byte = int(offset_bytes) + int(size_bytes) - 1
    request = Request(
        artifact_url,
        headers={
            "Range": f"bytes={int(offset_bytes)}-{end_byte}",
            "User-Agent": "CAI-Compute-Chain/1.0",
        },
    )

    with urlopen(request, timeout=timeout_sec) as response:
        status_code = _response_status_code(response)
        if status_code != 206:
            raise IOError(
                f"Origin did not honor range request for {artifact_url}; "
                f"expected HTTP 206, got {status_code or 'unknown'}."
            )
        payload = response.read()

    if len(payload) != int(size_bytes):
        raise IOError(
            f"Expected {size_bytes} bytes from {artifact_url}, got {len(payload)}."
        )
    return payload


def _read_origin_chunk_bytes(
    manifest: ModelPackageManifest,
    chunk: ModelChunk,
    *,
    policy: WalletPolicy | None = None,
) -> bytes:
    try:
        return _read_bound_artifact_bytes(
            manifest,
            artifact_id=chunk.artifact_id,
            offset_bytes=chunk.offset_bytes,
            size_bytes=chunk.size_bytes,
            policy=policy,
        )
    except FileNotFoundError:
        return _download_origin_artifact_range(
            manifest,
            artifact_id=chunk.artifact_id,
            offset_bytes=chunk.offset_bytes,
            size_bytes=chunk.size_bytes,
        )


def _read_bound_source_chunk_bytes(
    *,
    source_kind: str,
    source_id: str,
    catalog_id: str,
    version: str,
    chunk_id: str,
    policy: WalletPolicy | None = None,
) -> bytes:
    binding = get_chunk_source_binding(source_kind, source_id, policy)
    if binding is None:
        raise FileNotFoundError(
            f"No chunk source binding found for {source_kind}:{source_id}"
        )
    source_root = Path(binding.data_root_path)
    index_path = source_root / "chunk-store" / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    record = next(
        (
            CachedChunkRecord(**item)
            for item in raw
            if item.get("chunk_id") == chunk_id
            and item.get("catalog_id") == catalog_id
            and item.get("version") == version
        ),
        None,
    )
    if record is None:
        raise FileNotFoundError(
            f"Chunk {chunk_id} not found in bound source {source_kind}:{source_id}"
        )
    payload_path = source_root / "chunk-store" / Path(record.relative_path)
    if not payload_path.exists() or not payload_path.is_file():
        raise FileNotFoundError(payload_path)
    payload = payload_path.read_bytes()
    if len(payload) != record.size_bytes:
        raise IOError(
            f"Expected {record.size_bytes} bytes from {payload_path}, got {len(payload)}."
        )
    return payload


def _download_chunk_bytes_from_locator(
    *,
    locator: str,
    catalog_id: str,
    version: str,
    chunk_id: str,
    size_bytes: int,
    sha256_hex: str,
    timeout_sec: int = 30,
) -> bytes:
    base_url = str(locator or "").strip().rstrip("/")
    if not base_url.lower().startswith(("http://", "https://")):
        raise ValueError(f"Unsupported chunk source locator: {locator}")

    query = urlencode(
        {
            "catalog_id": catalog_id,
            "version": version,
            "chunk_id": chunk_id,
        }
    )
    chunk_url = f"{base_url}/v1/cai/chunks?{query}"
    with urlopen(chunk_url, timeout=timeout_sec) as response:
        payload = response.read()

    if len(payload) != int(size_bytes):
        raise IOError(
            f"Expected {size_bytes} bytes from {chunk_url}, got {len(payload)}."
        )
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != sha256_hex:
        raise IOError(
            f"SHA256 mismatch for {chunk_url}: expected {sha256_hex}, got {actual_sha256}."
        )
    return payload


def _read_cached_chunk_payload(
    chunk_id: str,
    *,
    catalog_id: str,
    version: str,
    policy: WalletPolicy | None = None,
) -> bytes:
    record = get_cached_chunk_record(chunk_id, policy)
    if record is None or record.catalog_id != catalog_id or record.version != version:
        raise FileNotFoundError(
            f"Chunk {chunk_id} not found in cache for {catalog_id}@{version}"
        )
    payload_path = chunk_store_root(policy) / Path(record.relative_path)
    if not payload_path.exists() or not payload_path.is_file():
        raise FileNotFoundError(payload_path)
    payload = payload_path.read_bytes()
    if len(payload) != record.size_bytes:
        raise IOError(
            f"Expected {record.size_bytes} bytes from {payload_path}, got {len(payload)}."
        )
    return payload


def select_default_materialized_artifact_id(
    manifest: ModelPackageManifest,
) -> str | None:
    for artifact in manifest.files:
        relative_path = str(artifact.relative_path).lower()
        media_type = str(artifact.media_type or "").lower()
        if relative_path.endswith(".gguf") or media_type == "application/gguf":
            return artifact.artifact_id
    return manifest.files[0].artifact_id if manifest.files else None


def materialized_artifact_path(
    manifest: ModelPackageManifest,
    artifact_id: str,
    *,
    policy: WalletPolicy | None = None,
) -> Path:
    artifact = next(
        (item for item in manifest.files if item.artifact_id == artifact_id),
        None,
    )
    if artifact is None:
        raise FileNotFoundError(
            f"Unknown artifact_id for materialization: {artifact_id}"
        )
    filename = Path(artifact.relative_path).name or f"{artifact_id}.bin"
    return materialized_artifacts_dir(
        manifest.catalog_id,
        manifest.version,
        policy,
    ) / filename


def materialized_assignment_artifact_path(
    manifest: ModelPackageManifest,
    artifact_id: str,
    assignment: ModelShardAssignment,
    *,
    policy: WalletPolicy | None = None,
) -> Path:
    artifact = next(
        (item for item in manifest.files if item.artifact_id == artifact_id),
        None,
    )
    if artifact is None:
        raise FileNotFoundError(
            f"Unknown artifact_id for materialization: {artifact_id}"
        )

    source_name = Path(artifact.relative_path).name or f"{artifact_id}.bin"
    stem = Path(source_name).stem or artifact_id
    suffix = Path(source_name).suffix or ".bin"
    assignment_suffix = (
        f".layers-{assignment.start_layer}-{assignment.end_layer}"
        f".rank-{assignment.device_rank}-of-{assignment.world_size}"
    )
    return materialized_artifacts_dir(
        manifest.catalog_id,
        manifest.version,
        policy,
    ) / f"{stem}{assignment_suffix}{suffix}"


def _mark_file_handle_sparse(handle) -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        import msvcrt

        bytes_returned = ctypes.c_ulong(0)
        ctypes.windll.kernel32.DeviceIoControl(
            msvcrt.get_osfhandle(handle.fileno()),
            FSCTL_SET_SPARSE,
            None,
            0,
            None,
            0,
            ctypes.byref(bytes_returned),
            None,
        )
    except Exception:
        return


def _assignment_chunks_digest(required_chunks: list[ModelChunk]) -> str:
    assignment_hasher = hashlib.sha256()
    for chunk in required_chunks:
        assignment_hasher.update(chunk.chunk_id.encode("utf-8"))
        assignment_hasher.update(chunk.sha256_hex.encode("utf-8"))
    return f"assignment:{assignment_hasher.hexdigest()}"


def _hash_open_file_range(handle, *, offset_bytes: int, size_bytes: int) -> str:
    hasher = hashlib.sha256()
    remaining = int(size_bytes)
    handle.seek(int(offset_bytes))
    while remaining > 0:
        payload = handle.read(
            min(remaining, MATERIALIZED_ASSIGNMENT_VALIDATION_READ_BYTES)
        )
        if not payload:
            raise IOError("Unexpected EOF while validating materialized assignment.")
        hasher.update(payload)
        remaining -= len(payload)
    return hasher.hexdigest()


def _materialized_assignment_artifact_is_valid(
    output_path: Path,
    *,
    artifact_size_bytes: int,
    required_chunks: list[ModelChunk],
) -> bool:
    try:
        if not output_path.exists() or not output_path.is_file():
            return False
        actual_size = int(output_path.stat().st_size)
        expected_size = int(artifact_size_bytes)
        if expected_size >= 0 and actual_size != expected_size:
            return False
        with output_path.open("rb") as handle:
            for chunk in required_chunks:
                offset_bytes = int(chunk.offset_bytes)
                size_bytes = int(chunk.size_bytes)
                if (
                    offset_bytes < 0
                    or size_bytes <= 0
                    or offset_bytes + size_bytes > actual_size
                ):
                    return False
                expected_hash = str(chunk.sha256_hex or "").strip().lower()
                if not expected_hash:
                    return False
                actual_hash = _hash_open_file_range(
                    handle,
                    offset_bytes=offset_bytes,
                    size_bytes=size_bytes,
                )
                if actual_hash != expected_hash:
                    return False
    except (OSError, IOError, ValueError):
        return False
    return True


def materialize_artifact_from_store(
    manifest: ModelPackageManifest,
    artifact_id: str,
    *,
    overwrite: bool = True,
    policy: WalletPolicy | None = None,
) -> MaterializedArtifactResult:
    artifact = next(
        (item for item in manifest.files if item.artifact_id == artifact_id),
        None,
    )
    if artifact is None:
        raise FileNotFoundError(
            f"Unknown artifact_id for materialization: {artifact_id}"
        )

    artifact_chunks = sorted(
        (
            chunk
            for chunk in manifest.chunks
            if chunk.artifact_id == artifact_id
        ),
        key=lambda chunk: chunk.offset_bytes,
    )
    if not artifact_chunks:
        raise FileNotFoundError(
            f"No chunks available to materialize artifact {artifact_id}"
        )

    output_path = materialized_artifact_path(manifest, artifact_id, policy=policy)
    if output_path.exists() and not overwrite:
        payload_hash = _hash_file(output_path)
        return MaterializedArtifactResult(
            catalog_id=manifest.catalog_id,
            version=manifest.version,
            artifact_id=artifact_id,
            output_path=str(output_path),
            size_bytes=output_path.stat().st_size,
            sha256_hex=payload_hash,
        )

    expected_offset = 0
    hasher = hashlib.sha256()
    with output_path.open("wb") as handle:
        for chunk in artifact_chunks:
            if chunk.offset_bytes != expected_offset:
                raise IOError(
                    f"Chunk offsets are not contiguous for {artifact_id}: "
                    f"expected {expected_offset}, got {chunk.offset_bytes}"
                )
            payload = _read_cached_chunk_payload(
                chunk.chunk_id,
                catalog_id=manifest.catalog_id,
                version=manifest.version,
                policy=policy,
            )
            if hashlib.sha256(payload).hexdigest() != chunk.sha256_hex:
                raise IOError(
                    f"Cached chunk hash mismatch for {chunk.chunk_id}"
                )
            handle.write(payload)
            hasher.update(payload)
            expected_offset += len(payload)

    if expected_offset != artifact.size_bytes:
        raise IOError(
            f"Materialized artifact size mismatch for {artifact_id}: "
            f"expected {artifact.size_bytes}, got {expected_offset}"
        )

    artifact_hash = hasher.hexdigest()
    if artifact_hash != artifact.sha256_hex:
        raise IOError(
            f"Materialized artifact hash mismatch for {artifact_id}: "
            f"expected {artifact.sha256_hex}, got {artifact_hash}"
        )

    return MaterializedArtifactResult(
        catalog_id=manifest.catalog_id,
        version=manifest.version,
        artifact_id=artifact_id,
        output_path=str(output_path),
        size_bytes=expected_offset,
        sha256_hex=artifact_hash,
    )


def materialize_assignment_artifact_from_store(
    manifest: ModelPackageManifest,
    artifact_id: str,
    assignment: ModelShardAssignment,
    *,
    include_default_chunks: bool = True,
    overwrite: bool = True,
    policy: WalletPolicy | None = None,
) -> MaterializedArtifactResult:
    artifact = next(
        (item for item in manifest.files if item.artifact_id == artifact_id),
        None,
    )
    if artifact is None:
        raise FileNotFoundError(
            f"Unknown artifact_id for materialization: {artifact_id}"
        )
    manifest.validate_assignment_layer_coverage(
        assignment.start_layer,
        assignment.end_layer,
        artifact_id=artifact_id,
    )

    required_chunks = sorted(
        (
            chunk
            for chunk in manifest.required_chunks_for_layers(
                assignment.start_layer,
                assignment.end_layer,
                include_default_chunks=include_default_chunks,
            )
            if chunk.artifact_id == artifact_id
        ),
        key=lambda chunk: chunk.offset_bytes,
    )
    if not required_chunks:
        raise FileNotFoundError(
            f"No assignment chunks available to materialize artifact {artifact_id}"
        )

    output_path = materialized_assignment_artifact_path(
        manifest,
        artifact_id,
        assignment,
        policy=policy,
    )
    if output_path.exists() and not overwrite:
        if not _materialized_assignment_artifact_is_valid(
            output_path,
            artifact_size_bytes=int(artifact.size_bytes),
            required_chunks=required_chunks,
        ):
            overwrite = True
        else:
            return MaterializedArtifactResult(
                catalog_id=manifest.catalog_id,
                version=manifest.version,
                artifact_id=artifact_id,
                output_path=str(output_path),
                size_bytes=output_path.stat().st_size,
                sha256_hex=_assignment_chunks_digest(required_chunks),
            )

    with output_path.open("wb") as handle:
        _mark_file_handle_sparse(handle)
        if artifact.size_bytes > 0:
            handle.seek(artifact.size_bytes - 1)
            handle.write(b"\0")
        for chunk in required_chunks:
            payload = _read_cached_chunk_payload(
                chunk.chunk_id,
                catalog_id=manifest.catalog_id,
                version=manifest.version,
                policy=policy,
            )
            if hashlib.sha256(payload).hexdigest() != chunk.sha256_hex:
                raise IOError(
                    f"Cached chunk hash mismatch for {chunk.chunk_id}"
                )
            handle.seek(chunk.offset_bytes)
            handle.write(payload)

    return MaterializedArtifactResult(
        catalog_id=manifest.catalog_id,
        version=manifest.version,
        artifact_id=artifact_id,
        output_path=str(output_path),
        size_bytes=output_path.stat().st_size,
        sha256_hex=_assignment_chunks_digest(required_chunks),
    )


def materialize_default_artifact_from_store(
    manifest: ModelPackageManifest,
    *,
    overwrite: bool = True,
    policy: WalletPolicy | None = None,
) -> MaterializedArtifactResult:
    artifact_id = select_default_materialized_artifact_id(manifest)
    if artifact_id is None:
        raise FileNotFoundError(
            f"No materializable artifact available for {manifest.catalog_id}@{manifest.version}"
        )
    return materialize_artifact_from_store(
        manifest,
        artifact_id,
        overwrite=overwrite,
        policy=policy,
    )


def materialize_default_assignment_artifact_from_store(
    manifest: ModelPackageManifest,
    assignment: ModelShardAssignment,
    *,
    include_default_chunks: bool = True,
    overwrite: bool = True,
    policy: WalletPolicy | None = None,
) -> MaterializedArtifactResult:
    artifact_id = select_default_materialized_artifact_id(manifest)
    if artifact_id is None:
        raise FileNotFoundError(
            f"No materializable artifact available for {manifest.catalog_id}@{manifest.version}"
        )
    return materialize_assignment_artifact_from_store(
        manifest,
        artifact_id,
        assignment,
        include_default_chunks=include_default_chunks,
        overwrite=overwrite,
        policy=policy,
    )


def execute_chunk_download_queue(
    *,
    max_tasks: int | None = None,
    task_ids: set[str] | list[str] | tuple[str, ...] | None = None,
    cache_policy: ChunkCachePolicy | None = None,
    fetch_policy: ChunkFetchPolicy | None = None,
    policy: WalletPolicy | None = None,
) -> list[ChunkDownloadTask]:
    active_cache_policy = cache_policy or ChunkCachePolicy()
    active_fetch_policy = fetch_policy or ChunkFetchPolicy()
    tasks = _read_chunk_download_tasks(policy)
    if not tasks:
        return []

    selected_task_ids = (
        {str(task_id) for task_id in task_ids}
        if task_ids is not None
        else None
    )
    processed: list[ChunkDownloadTask] = []
    tasks_processed = 0
    for task in list(tasks):
        if max_tasks is not None and tasks_processed >= max_tasks:
            break
        if selected_task_ids is not None and task.task_id not in selected_task_ids:
            continue
        if task.status not in (
            ChunkDownloadTaskStatus.QUEUED,
            ChunkDownloadTaskStatus.IN_PROGRESS,
            ChunkDownloadTaskStatus.FAILED,
        ):
            continue

        manifest = load_model_package_manifest(task.catalog_id, task.version, policy)
        chunk = next(
            (item for item in manifest.chunks if item.chunk_id == task.chunk_id),
            None,
        )
        if chunk is None:
            updated = update_chunk_download_task_status(
                task.task_id,
                ChunkDownloadTaskStatus.FAILED,
                last_error=f"Unknown chunk id in manifest: {task.chunk_id}",
                policy=policy,
            )
            if updated is not None:
                processed.append(updated)
                tasks_processed += 1
            continue

        candidate_sources = order_chunk_fetch_sources(
            task.sources
            or (
                ChunkFetchSource(
                    kind=ChunkFetchSourceKind.ORIGIN,
                    source_id=manifest.source_repo_id or "origin",
                    locator=manifest.source_repo_id,
                ),
            ),
            fetch_policy=active_fetch_policy,
            policy=policy,
        )

        last_error: str | None = None
        for source in candidate_sources:
            try:
                if source.kind == ChunkFetchSourceKind.ORIGIN:
                    payload = _read_origin_chunk_bytes(
                        manifest,
                        chunk,
                        policy=policy,
                    )
                elif source.kind in (
                    ChunkFetchSourceKind.PEER_CACHE,
                    ChunkFetchSourceKind.STORAGE_SEED,
                ):
                    if source.locator:
                        payload = _download_chunk_bytes_from_locator(
                            locator=source.locator,
                            catalog_id=manifest.catalog_id,
                            version=manifest.version,
                            chunk_id=chunk.chunk_id,
                            size_bytes=chunk.size_bytes,
                            sha256_hex=chunk.sha256_hex,
                        )
                    else:
                        payload = _read_bound_source_chunk_bytes(
                            source_kind=source.kind,
                            source_id=source.source_id,
                            catalog_id=manifest.catalog_id,
                            version=manifest.version,
                            chunk_id=chunk.chunk_id,
                            policy=policy,
                        )
                else:
                    last_error = (
                        f"Source kind not yet executable locally: {source.kind}"
                    )
                    continue
                put_cached_chunk(
                    catalog_id=manifest.catalog_id,
                    version=manifest.version,
                    chunk_id=chunk.chunk_id,
                    sha256_hex=chunk.sha256_hex,
                    content=payload,
                    pinned=active_cache_policy.pin_assignment_chunks,
                    cache_class=ChunkCacheClass.HOT,
                    lease_status=(
                        ChunkLeaseStatus.ACTIVE
                        if active_cache_policy.assignment_lease_seconds > 0
                        else ChunkLeaseStatus.NONE
                    ),
                    lease_expires_at=_assignment_lease_expires_at(active_cache_policy),
                    policy=policy,
                )
                updated = update_chunk_download_task_status(
                    task.task_id,
                    ChunkDownloadTaskStatus.COMPLETED,
                    selected_source_kind=source.kind,
                    selected_source_id=source.source_id,
                    policy=policy,
                )
                record_chunk_source_success(
                    source,
                    fetch_policy=active_fetch_policy,
                    policy=policy,
                )
                if updated is not None:
                    processed.append(updated)
                    tasks_processed += 1
                last_error = None
                break
            except Exception as exc:
                last_error = str(exc)
                record_chunk_source_failure(
                    source,
                    error=last_error,
                    fetch_policy=active_fetch_policy,
                    policy=policy,
                )
                continue

        if last_error is not None:
            updated = update_chunk_download_task_status(
                task.task_id,
                ChunkDownloadTaskStatus.FAILED,
                last_error=last_error,
                policy=policy,
            )
            if updated is not None:
                processed.append(updated)
                tasks_processed += 1

    return processed


def ensure_assignment_ready_from_store(
    manifest: ModelPackageManifest,
    assignment: ModelShardAssignment,
    *,
    peer_chunk_inventory: dict[str, set[str] | list[str] | tuple[str, ...]] | None = None,
    seed_chunk_inventory: dict[str, set[str] | list[str] | tuple[str, ...]] | None = None,
    use_imported_peer_inventory: bool = False,
    use_imported_seed_inventory: bool = False,
    include_default_chunks: bool = True,
    max_tasks: int | None = None,
    fetch_policy: ChunkFetchPolicy | None = None,
    policy: WalletPolicy | None = None,
) -> AssignmentEnsureReadyResult:
    initial_plan = build_assignment_chunk_plan_from_store(
        manifest,
        assignment,
        include_default_chunks=include_default_chunks,
        verify_cached_chunks=True,
        policy=policy,
    )
    if initial_plan.ready:
        apply_assignment_cache_policy_from_store(
            manifest,
            assignment,
            include_default_chunks=include_default_chunks,
            policy=policy,
        )
        final_initial_plan = build_assignment_chunk_plan_from_store(
            manifest,
            assignment,
            include_default_chunks=include_default_chunks,
            policy=policy,
        )
        return AssignmentEnsureReadyResult(
            manifest=manifest,
            assignment=assignment,
            initial_plan=initial_plan,
            fetch_plan=None,
            queued_tasks=(),
            processed_tasks=(),
            final_plan=final_initial_plan,
        )

    fetch_plan = build_assignment_fetch_plan_from_store(
        manifest,
        assignment,
        peer_chunk_inventory=peer_chunk_inventory,
        seed_chunk_inventory=seed_chunk_inventory,
        use_imported_peer_inventory=use_imported_peer_inventory,
        use_imported_seed_inventory=use_imported_seed_inventory,
        include_default_chunks=include_default_chunks,
        verify_cached_chunks=True,
        fetch_policy=fetch_policy,
        policy=policy,
    )
    queued_tasks = tuple(queue_assignment_fetch_plan(manifest, fetch_plan, policy))
    processed_tasks = tuple(
        execute_chunk_download_queue(
            max_tasks=max_tasks,
            task_ids={task.task_id for task in queued_tasks},
            fetch_policy=fetch_policy,
            policy=policy,
        )
    )
    final_plan = build_assignment_chunk_plan_from_store(
        manifest,
        assignment,
        include_default_chunks=include_default_chunks,
        verify_cached_chunks=True,
        policy=policy,
    )
    if final_plan.ready:
        apply_assignment_cache_policy_from_store(
            manifest,
            assignment,
            include_default_chunks=include_default_chunks,
            policy=policy,
        )
        final_plan = build_assignment_chunk_plan_from_store(
            manifest,
            assignment,
            include_default_chunks=include_default_chunks,
            verify_cached_chunks=True,
            policy=policy,
        )
    return AssignmentEnsureReadyResult(
        manifest=manifest,
        assignment=assignment,
        initial_plan=initial_plan,
        fetch_plan=fetch_plan,
        queued_tasks=queued_tasks,
        processed_tasks=processed_tasks,
        final_plan=final_plan,
    )


def ensure_default_chunks_ready_from_store(
    manifest: ModelPackageManifest,
    *,
    node_id: str | None = None,
    device_rank: int = 0,
    world_size: int = 1,
    peer_chunk_inventory: dict[str, set[str] | list[str] | tuple[str, ...]] | None = None,
    seed_chunk_inventory: dict[str, set[str] | list[str] | tuple[str, ...]] | None = None,
    use_imported_peer_inventory: bool = False,
    use_imported_seed_inventory: bool = False,
    max_tasks: int | None = None,
    fetch_policy: ChunkFetchPolicy | None = None,
    policy: WalletPolicy | None = None,
) -> AssignmentEnsureReadyResult:
    initial_plan = build_default_chunk_plan_from_store(
        manifest,
        node_id=node_id,
        device_rank=device_rank,
        world_size=world_size,
        policy=policy,
    )
    if initial_plan.ready:
        return AssignmentEnsureReadyResult(
            manifest=manifest,
            assignment=initial_plan.assignment,
            initial_plan=initial_plan,
            fetch_plan=None,
            queued_tasks=(),
            processed_tasks=(),
            final_plan=initial_plan,
        )

    fetch_plan = build_default_chunk_fetch_plan_from_store(
        manifest,
        node_id=node_id,
        device_rank=device_rank,
        world_size=world_size,
        peer_chunk_inventory=peer_chunk_inventory,
        seed_chunk_inventory=seed_chunk_inventory,
        use_imported_peer_inventory=use_imported_peer_inventory,
        use_imported_seed_inventory=use_imported_seed_inventory,
        fetch_policy=fetch_policy,
        policy=policy,
    )
    queued_tasks = tuple(queue_assignment_fetch_plan(manifest, fetch_plan, policy))
    processed_tasks = tuple(
        execute_chunk_download_queue(
            max_tasks=max_tasks,
            task_ids={task.task_id for task in queued_tasks},
            fetch_policy=fetch_policy,
            policy=policy,
        )
    )
    final_plan = build_default_chunk_plan_from_store(
        manifest,
        node_id=node_id,
        device_rank=device_rank,
        world_size=world_size,
        policy=policy,
    )
    return AssignmentEnsureReadyResult(
        manifest=manifest,
        assignment=initial_plan.assignment,
        initial_plan=initial_plan,
        fetch_plan=fetch_plan,
        queued_tasks=queued_tasks,
        processed_tasks=processed_tasks,
        final_plan=final_plan,
    )


def _manifest_allows_background_chunk_prefetch(
    manifest: ModelPackageManifest,
    *,
    include_private_curated: bool = False,
) -> bool:
    if include_private_curated:
        return True
    return str(manifest.package_kind) != str(ModelPackageKind.PRIVATE_CURATED)


def prefetch_default_chunks_from_fresh_inventories(
    *,
    node_id: str | None = None,
    device_rank: int = 0,
    world_size: int = 1,
    max_manifests: int = 4,
    max_tasks: int = 8,
    include_private_curated: bool = False,
    fetch_policy: ChunkFetchPolicy | None = None,
    policy: WalletPolicy | None = None,
) -> DefaultChunkPrefetchResult:
    active_fetch_policy = fetch_policy or ChunkFetchPolicy()
    manifest_refs = fresh_imported_manifest_refs(
        fetch_policy=active_fetch_policy,
        policy=policy,
    )
    manifest_refs = prioritize_manifest_refs_for_node(
        manifest_refs,
        node_id=node_id,
        fetch_policy=active_fetch_policy,
        policy=policy,
    )
    manifests_considered = 0
    manifests_prefetched = 0
    queued_tasks = 0
    processed_tasks = 0
    remaining_tasks = max(0, int(max_tasks))
    for catalog_id, version in manifest_refs:
        if remaining_tasks == 0:
            break
        try:
            manifest = load_model_package_manifest(catalog_id, version, policy)
        except FileNotFoundError:
            continue
        if not _manifest_allows_background_chunk_prefetch(
            manifest,
            include_private_curated=include_private_curated,
        ):
            continue
        if max_manifests > 0 and manifests_considered >= max_manifests:
            break
        manifests_considered += 1
        result = ensure_default_chunks_ready_from_store(
            manifest,
            node_id=node_id,
            device_rank=device_rank,
            world_size=world_size,
            use_imported_peer_inventory=True,
            use_imported_seed_inventory=True,
            max_tasks=remaining_tasks,
            fetch_policy=active_fetch_policy,
            policy=policy,
        )
        queued_tasks += len(result.queued_tasks)
        processed_tasks += len(result.processed_tasks)
        if result.ready and not result.initial_plan.ready:
            manifests_prefetched += 1
        if max_tasks > 0:
            remaining_tasks = max(0, remaining_tasks - len(result.processed_tasks))
    return DefaultChunkPrefetchResult(
        manifests_considered=manifests_considered,
        manifests_prefetched=manifests_prefetched,
        queued_tasks=queued_tasks,
        processed_tasks=processed_tasks,
    )


def ensure_bootstrap_chunks_ready_from_store(
    manifest: ModelPackageManifest,
    *,
    node_id: str | None = None,
    device_rank: int = 0,
    world_size: int = 1,
    peer_chunk_inventory: dict[str, set[str] | list[str] | tuple[str, ...]] | None = None,
    seed_chunk_inventory: dict[str, set[str] | list[str] | tuple[str, ...]] | None = None,
    use_imported_peer_inventory: bool = False,
    use_imported_seed_inventory: bool = False,
    max_weight_chunks: int = 1,
    max_weight_bytes: int | None = None,
    include_default_chunks: bool = True,
    hint_start_layer: int | None = None,
    hint_end_layer: int | None = None,
    max_tasks: int | None = None,
    fetch_policy: ChunkFetchPolicy | None = None,
    policy: WalletPolicy | None = None,
) -> AssignmentEnsureReadyResult:
    initial_plan = build_bootstrap_chunk_plan_from_store(
        manifest,
        node_id=node_id,
        device_rank=device_rank,
        world_size=world_size,
        max_weight_chunks=max_weight_chunks,
        max_weight_bytes=max_weight_bytes,
        include_default_chunks=include_default_chunks,
        hint_start_layer=hint_start_layer,
        hint_end_layer=hint_end_layer,
        policy=policy,
    )
    if initial_plan.ready:
        return AssignmentEnsureReadyResult(
            manifest=manifest,
            assignment=initial_plan.assignment,
            initial_plan=initial_plan,
            fetch_plan=None,
            queued_tasks=(),
            processed_tasks=(),
            final_plan=initial_plan,
        )

    fetch_plan = build_bootstrap_chunk_fetch_plan_from_store(
        manifest,
        node_id=node_id,
        device_rank=device_rank,
        world_size=world_size,
        peer_chunk_inventory=peer_chunk_inventory,
        seed_chunk_inventory=seed_chunk_inventory,
        use_imported_peer_inventory=use_imported_peer_inventory,
        use_imported_seed_inventory=use_imported_seed_inventory,
        max_weight_chunks=max_weight_chunks,
        max_weight_bytes=max_weight_bytes,
        include_default_chunks=include_default_chunks,
        hint_start_layer=hint_start_layer,
        hint_end_layer=hint_end_layer,
        fetch_policy=fetch_policy,
        policy=policy,
    )
    queued_tasks = tuple(queue_assignment_fetch_plan(manifest, fetch_plan, policy))
    processed_tasks = tuple(
        execute_chunk_download_queue(
            max_tasks=max_tasks,
            task_ids={task.task_id for task in queued_tasks},
            fetch_policy=fetch_policy,
            policy=policy,
        )
    )
    final_plan = build_bootstrap_chunk_plan_from_store(
        manifest,
        node_id=node_id,
        device_rank=device_rank,
        world_size=world_size,
        max_weight_chunks=max_weight_chunks,
        max_weight_bytes=max_weight_bytes,
        include_default_chunks=include_default_chunks,
        hint_start_layer=hint_start_layer,
        hint_end_layer=hint_end_layer,
        policy=policy,
    )
    return AssignmentEnsureReadyResult(
        manifest=manifest,
        assignment=initial_plan.assignment,
        initial_plan=initial_plan,
        fetch_plan=fetch_plan,
        queued_tasks=queued_tasks,
        processed_tasks=processed_tasks,
        final_plan=final_plan,
    )


def prefetch_bootstrap_chunks_from_fresh_inventories(
    *,
    node_id: str | None = None,
    device_rank: int = 0,
    world_size: int = 1,
    max_manifests: int = 2,
    max_tasks: int = 4,
    include_private_curated: bool = False,
    fetch_policy: ChunkFetchPolicy | None = None,
    policy: WalletPolicy | None = None,
) -> BootstrapChunkPrefetchResult:
    active_fetch_policy = fetch_policy or ChunkFetchPolicy()
    manifest_refs = fresh_imported_manifest_refs(
        fetch_policy=active_fetch_policy,
        policy=policy,
    )
    manifest_refs = prioritize_manifest_refs_for_node(
        manifest_refs,
        node_id=node_id,
        fetch_policy=active_fetch_policy,
        policy=policy,
    )
    manifests_considered = 0
    manifests_prefetched = 0
    queued_tasks = 0
    processed_tasks = 0
    remaining_tasks = max(0, int(max_tasks))
    for catalog_id, version in manifest_refs:
        if remaining_tasks == 0:
            break
        try:
            manifest = load_model_package_manifest(catalog_id, version, policy)
        except FileNotFoundError:
            continue
        if not _manifest_allows_background_chunk_prefetch(
            manifest,
            include_private_curated=include_private_curated,
        ):
            continue
        if max_manifests > 0 and manifests_considered >= max_manifests:
            break
        manifests_considered += 1
        result = ensure_bootstrap_chunks_ready_from_store(
            manifest,
            node_id=node_id,
            device_rank=device_rank,
            world_size=world_size,
            use_imported_peer_inventory=True,
            use_imported_seed_inventory=True,
            max_weight_chunks=max(
                0, int(active_fetch_policy.warm_prefetch_weight_chunk_count_per_manifest)
            ),
            max_weight_bytes=max(
                0, int(active_fetch_policy.warm_prefetch_max_weight_bytes_per_manifest)
            ),
            max_tasks=remaining_tasks,
            fetch_policy=active_fetch_policy,
            policy=policy,
        )
        queued_tasks += len(result.queued_tasks)
        processed_tasks += len(result.processed_tasks)
        if result.ready and not result.initial_plan.ready:
            manifests_prefetched += 1
        if max_tasks > 0:
            remaining_tasks = max(0, remaining_tasks - len(result.processed_tasks))
    return BootstrapChunkPrefetchResult(
        manifests_considered=manifests_considered,
        manifests_prefetched=manifests_prefetched,
        queued_tasks=queued_tasks,
        processed_tasks=processed_tasks,
    )


def prefetch_hinted_bootstrap_chunks(
    hints: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    fetch_policy: ChunkFetchPolicy | None = None,
    policy: WalletPolicy | None = None,
) -> HintedChunkPrefetchResult:
    active_fetch_policy = fetch_policy or ChunkFetchPolicy()
    manifests_considered = 0
    manifests_prefetched = 0
    queued_tasks = 0
    processed_tasks = 0
    seen_keys: set[tuple[str, str, int, int, int, int, str]] = set()
    for raw_hint in hints:
        model_id = str(raw_hint.get("model_id") or "").strip()
        if not model_id:
            continue
        start_layer = int(raw_hint.get("start_layer") or 0)
        end_layer = int(raw_hint.get("end_layer") or 0)
        if end_layer <= start_layer:
            continue
        device_rank = int(raw_hint.get("device_rank") or 0)
        world_size = int(raw_hint.get("world_size") or 1)
        node_id = str(raw_hint.get("node_id") or "").strip() or None
        manifests = find_model_package_manifests_for_model(model_id, policy)
        if not manifests:
            continue
        manifest = sorted(
            manifests,
            key=lambda item: (item.version, item.created_at),
            reverse=True,
        )[0]
        key = (
            manifest.catalog_id,
            manifest.version,
            start_layer,
            end_layer,
            device_rank,
            world_size,
            str(node_id or ""),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        manifests_considered += 1
        result = ensure_bootstrap_chunks_ready_from_store(
            manifest,
            node_id=node_id,
            device_rank=device_rank,
            world_size=world_size,
            use_imported_peer_inventory=True,
            use_imported_seed_inventory=True,
            max_weight_chunks=max(
                0, int(active_fetch_policy.hint_prefetch_weight_chunk_count_per_manifest)
            ),
            max_weight_bytes=max(
                0, int(active_fetch_policy.hint_prefetch_max_weight_bytes_per_manifest)
            ),
            hint_start_layer=start_layer,
            hint_end_layer=end_layer,
            max_tasks=max(
                0, int(active_fetch_policy.hint_prefetch_weight_chunk_count_per_manifest) + 4
            ),
            fetch_policy=active_fetch_policy,
            policy=policy,
        )
        queued_tasks += len(result.queued_tasks)
        processed_tasks += len(result.processed_tasks)
        if result.ready and not result.initial_plan.ready:
            manifests_prefetched += 1
    return HintedChunkPrefetchResult(
        manifests_considered=manifests_considered,
        manifests_prefetched=manifests_prefetched,
        queued_tasks=queued_tasks,
        processed_tasks=processed_tasks,
    )


def load_model_package_manifest(
    catalog_id: str,
    version: str,
    policy: WalletPolicy | None = None,
) -> ModelPackageManifest:
    path = manifest_file_path(catalog_id, version, policy)
    if not path.exists():
        raise FileNotFoundError(path)
    return ModelPackageManifest.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )


def list_model_package_manifests(
    policy: WalletPolicy | None = None,
) -> list[ModelPackageManifest]:
    root = model_package_root(policy)
    manifests: list[ModelPackageManifest] = []
    for path in sorted(root.glob("*/*/manifest.json")):
        try:
            manifests.append(
                ModelPackageManifest.from_dict(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            )
        except Exception:
            continue
    return manifests


def find_model_package_manifests_for_model(
    model_id: str,
    policy: WalletPolicy | None = None,
) -> list[ModelPackageManifest]:
    normalized = str(model_id).strip()
    return [
        manifest
        for manifest in list_model_package_manifests(policy)
        if manifest.model_id == normalized
    ]


def select_model_package_manifest_for_model(
    model_id: str,
    policy: WalletPolicy | None = None,
) -> ModelPackageManifest | None:
    manifests = find_model_package_manifests_for_model(model_id, policy)
    if not manifests:
        return None

    def _sort_key(manifest: ModelPackageManifest) -> tuple[str, str]:
        return (str(manifest.version), str(manifest.created_at))

    return sorted(manifests, key=_sort_key, reverse=True)[0]


def make_chunk_id(
    artifact_id: str,
    *,
    offset_bytes: int,
    size_bytes: int,
    sha256_hex: str,
) -> str:
    payload = f"{artifact_id}:{offset_bytes}:{size_bytes}:{sha256_hex}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def validate_model_package_manifest(manifest: ModelPackageManifest) -> None:
    if manifest.manifest_version < 1:
        raise ModelManifestValidationError("Manifest version must be >= 1.")
    if not manifest.catalog_id.strip():
        raise ModelManifestValidationError("catalog_id must not be empty.")
    if not manifest.model_id.strip():
        raise ModelManifestValidationError("model_id must not be empty.")
    if not manifest.version.strip():
        raise ModelManifestValidationError("version must not be empty.")
    if not manifest.backend.strip():
        raise ModelManifestValidationError("backend must not be empty.")

    artifact_ids = set()
    artifact_size_by_id: dict[str, int] = {}
    for artifact in manifest.files:
        if not artifact.artifact_id.strip():
            raise ModelManifestValidationError("artifact_id must not be empty.")
        if artifact.artifact_id in artifact_ids:
            raise ModelManifestValidationError(
                f"Duplicate artifact_id in manifest: {artifact.artifact_id}"
            )
        artifact_ids.add(artifact.artifact_id)
        if artifact.size_bytes < 0:
            raise ModelManifestValidationError(
                f"Artifact size must be >= 0 for {artifact.artifact_id}."
            )
        artifact_size_by_id[artifact.artifact_id] = int(artifact.size_bytes)

    chunk_ids = set()
    for chunk in manifest.chunks:
        if not chunk.chunk_id.strip():
            raise ModelManifestValidationError("chunk_id must not be empty.")
        if chunk.chunk_id in chunk_ids:
            raise ModelManifestValidationError(
                f"Duplicate chunk_id in manifest: {chunk.chunk_id}"
            )
        chunk_ids.add(chunk.chunk_id)
        if chunk.artifact_id not in artifact_ids:
            raise ModelManifestValidationError(
                f"Chunk {chunk.chunk_id} references unknown artifact_id {chunk.artifact_id}."
            )
        if chunk.offset_bytes < 0:
            raise ModelManifestValidationError(
                f"Chunk {chunk.chunk_id} has negative offset."
            )
        if chunk.size_bytes <= 0:
            raise ModelManifestValidationError(
                f"Chunk {chunk.chunk_id} must have positive size."
            )
        artifact_size = artifact_size_by_id.get(chunk.artifact_id)
        if (
            artifact_size is not None
            and chunk.offset_bytes + chunk.size_bytes > artifact_size
        ):
            raise ModelManifestValidationError(
                f"Chunk {chunk.chunk_id} exceeds artifact bounds for {chunk.artifact_id}."
            )
        if (chunk.layer_start is None) != (chunk.layer_end is None):
            raise ModelManifestValidationError(
                f"Chunk {chunk.chunk_id} must specify both layer_start and layer_end or neither."
            )
        if chunk.layer_start is not None and chunk.layer_end is not None:
            if chunk.layer_start < 0 or chunk.layer_end <= chunk.layer_start:
                raise ModelManifestValidationError(
                    f"Chunk {chunk.chunk_id} has invalid layer range."
                )

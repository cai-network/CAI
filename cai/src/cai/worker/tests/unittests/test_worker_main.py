# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import anyio
import importlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cai.shared.models.model_cards import InferenceBackend, ModelCard, ModelId, ModelTask
from cai.shared.types.common import NodeId, SystemId
from cai.shared.types.events import NodeDownloadProgress, TaskCreated, TaskStatusUpdated
from cai.shared.types.memory import Memory
from cai.shared.types.state import State
from cai.shared.types.tasks import DownloadModel, TaskStatus
from cai.shared.types.worker.downloads import DownloadCompleted, DownloadFailed
from cai.shared.types.worker.instances import InstanceId, MlxRingInstance
from cai.shared.types.worker.runners import RunnerId, ShardAssignments
from cai.shared.types.worker.shards import PipelineShardMetadata
from cai.utils.channels import channel
from cai.utils.info_gatherer.info_gatherer import MiscData
from cai.utils.keyed_backoff import KeyedBackoff
from cai.worker.main import (
    CAI_CHUNK_SEED_URLS_ENV,
    Worker,
    _cai_owned_transport_adapter,
    _cai_owned_transport_runtime_requires_production_handoff,
    _prefetch_chunk_backed_bootstrap_chunks,
    _prefetch_chunk_backed_default_chunks,
    _prefetch_chunk_backed_instance_shard_hints,
    _prefetch_chunk_backed_local_shard_hints,
    _prefetch_chunk_backed_recent_shard_hints,
    _remember_chunk_backed_instance_shard_hints,
    _remember_chunk_backed_local_shard_hints,
    _release_chunk_backed_assignment_cache,
    _sync_chunk_backed_seed_inventories,
    _sync_chunk_backed_peer_inventories,
    _try_prepare_chunk_backed_llama_cpp_download,
)


def test_cai_owned_transport_adapter_uses_env_factory(monkeypatch) -> None:
    from cai_compute_chain.cai_owned_runtime import ExternalLlamaCppShardAdapter

    monkeypatch.setenv("CAI_LLM_SHARD_ADAPTER", "smoke_runner")

    adapter = _cai_owned_transport_adapter(NodeId("node-a"))

    assert isinstance(adapter, ExternalLlamaCppShardAdapter)
    assert "cai_compute_chain.cai_llama_cpp_shard_smoke_runner" in adapter.command


def test_cai_owned_transport_adapter_preserves_legacy_deterministic_prefix(
    monkeypatch,
) -> None:
    from cai_compute_chain.cai_owned_runtime import DeterministicBytesShardAdapter

    monkeypatch.delenv("CAI_LLM_SHARD_ADAPTER", raising=False)
    monkeypatch.delenv("CAI_DETERMINISTIC_SHARD_PREFIX", raising=False)
    monkeypatch.setenv("CAI_OWNED_TRANSPORT_DETERMINISTIC_PREFIX", "node:{node_id}:")

    adapter = _cai_owned_transport_adapter(NodeId("node-a"))

    assert isinstance(adapter, DeterministicBytesShardAdapter)
    assert adapter.prefix == b"node:node-a:"


def test_cai_owned_transport_runtime_production_handoff_guard(monkeypatch) -> None:
    monkeypatch.delenv("CAI_REQUIRE_PRODUCTION_LLM_HANDOFF", raising=False)
    assert _cai_owned_transport_runtime_requires_production_handoff() is False

    monkeypatch.setenv("CAI_REQUIRE_PRODUCTION_LLM_HANDOFF", "1")
    assert _cai_owned_transport_runtime_requires_production_handoff() is True


def test_forward_info_exits_cleanly_when_event_channel_is_closed() -> None:
    async def _run() -> None:
        info_send, info_recv = channel[MiscData]()
        event_send, _event_recv = channel[object]()

        worker = object.__new__(Worker)
        worker.node_id = NodeId("node-a")
        worker.event_sender = event_send

        event_send.close()

        async with anyio.create_task_group() as tg:
            tg.start_soon(worker._forward_info, info_recv)
            await info_send.send(MiscData(friendly_name="local"))
            info_send.close()

    anyio.run(_run)


def test_try_prepare_chunk_backed_llama_cpp_download_returns_completed() -> None:
    @dataclass(frozen=True)
    class FakeAssignment:
        start_layer: int
        end_layer: int
        device_rank: int = 0
        world_size: int = 1
        node_id: str | None = None

    fake_manifest = SimpleNamespace(catalog_id="demo", version="v1")
    fake_materialized = SimpleNamespace(output_path=str(Path("D:/tmp/materialized.gguf")))
    recorded_sync_calls: list[tuple[dict[str, object], str, str, bool]] = []
    recorded_hint_calls: list[tuple[str, list[dict[str, object]]]] = []

    class FakeModelDistributionModule:
        ModelShardAssignment = FakeAssignment

        @staticmethod
        def select_model_package_manifest_for_model(model_id: str):
            return fake_manifest if model_id == "Qwen/Qwen3-0.6B-GGUF" else None

        @staticmethod
        def ensure_assignment_ready_from_store(
            manifest,
            assignment,
            *,
            use_imported_peer_inventory: bool,
            use_imported_seed_inventory: bool,
        ):
            assert manifest is fake_manifest
            assert assignment.start_layer == 0
            assert assignment.end_layer == 28
            assert use_imported_peer_inventory is True
            assert use_imported_seed_inventory is True
            return SimpleNamespace(ready=True)

        @staticmethod
        def materialize_default_assignment_artifact_from_store(manifest, assignment):
            assert manifest is fake_manifest
            assert assignment.start_layer == 0
            assert assignment.end_layer == 28
            assert assignment.node_id == "node-a"
            return fake_materialized

        @staticmethod
        def remember_recent_shard_hints(node_id, hints):
            recorded_hint_calls.append((str(node_id), list(hints)))
            return SimpleNamespace(
                hints_received=len(hints),
                records_upserted=len(hints),
                records_pruned=0,
                stored_records=len(hints),
            )

        @staticmethod
        def sync_chunk_inventory_from_cai_peers(
            *,
            state_payload,
            CAI_url: str,
            source_kind: str,
            prune_missing_peers: bool,
        ):
            recorded_sync_calls.append(
                (state_payload, CAI_url, source_kind, prune_missing_peers)
            )
            return SimpleNamespace(imported_payloads=1)

    recorded_paths: list[tuple[ModelId, str]] = []
    real_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name == "cai_compute_chain.model_distribution":
            return FakeModelDistributionModule
        return real_import_module(name)

    def fake_set_custom_model_local_path(model_id: ModelId, local_path: str):
        recorded_paths.append((model_id, local_path))
        return Path(local_path)

    shard = PipelineShardMetadata(
        model_card=ModelCard(
            model_id=ModelId("Qwen/Qwen3-0.6B-GGUF"),
            storage_size=Memory.from_mb(1),
            n_layers=28,
            hidden_size=1,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
            inference_backend=InferenceBackend.LlamaCpp,
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=28,
        n_layers=28,
    )

    with patch("cai.worker.main.importlib.import_module", fake_import_module), patch(
        "cai.worker.main.set_custom_model_local_path",
        fake_set_custom_model_local_path,
    ), patch("cai.worker.main.urlopen") as urlopen_mock:
        urlopen_mock.return_value.__enter__.return_value.read.return_value = b'{"nodeIdentities": {}}'
        completed = _try_prepare_chunk_backed_llama_cpp_download(
            NodeId("node-a"),
            shard,
            api_port=52415,
        )

    assert completed is not None
    assert completed.model_directory == fake_materialized.output_path
    assert completed.read_only is False
    assert recorded_hint_calls == [
        (
            "node-a",
            [
                {
                    "model_id": "Qwen/Qwen3-0.6B-GGUF",
                    "start_layer": 0,
                    "end_layer": 28,
                    "device_rank": 0,
                    "world_size": 1,
                }
            ],
        )
    ]
    assert recorded_paths == [(ModelId("Qwen/Qwen3-0.6B-GGUF"), fake_materialized.output_path)]
    assert recorded_sync_calls == [
        ({"nodeIdentities": {}}, "http://127.0.0.1:52415", "peer_cache", True)
    ]


def test_try_prepare_chunk_backed_llama_cpp_download_returns_failed_when_not_ready() -> None:
    @dataclass(frozen=True)
    class FakeAssignment:
        start_layer: int
        end_layer: int
        device_rank: int = 0
        world_size: int = 1
        node_id: str | None = None

    fake_manifest = SimpleNamespace(catalog_id="demo", version="v1")

    class FakeModelDistributionModule:
        ModelShardAssignment = FakeAssignment

        @staticmethod
        def select_model_package_manifest_for_model(model_id: str):
            return fake_manifest if model_id == "Qwen/Qwen3-0.6B-GGUF" else None

        @staticmethod
        def ensure_assignment_ready_from_store(
            manifest,
            assignment,
            *,
            use_imported_peer_inventory: bool,
            use_imported_seed_inventory: bool,
        ):
            assert manifest is fake_manifest
            assert assignment.node_id == "node-a"
            assert use_imported_peer_inventory is True
            assert use_imported_seed_inventory is True
            return SimpleNamespace(
                ready=False,
                final_plan=SimpleNamespace(
                    coverage=SimpleNamespace(missing_chunk_ids=("chunk-a", "chunk-b"))
                ),
            )

        @staticmethod
        def remember_recent_shard_hints(node_id, hints):
            return SimpleNamespace(
                hints_received=len(hints),
                records_upserted=len(hints),
                records_pruned=0,
                stored_records=len(hints),
            )

        @staticmethod
        def sync_chunk_inventory_from_cai_peers(
            *,
            state_payload,
            CAI_url: str,
            source_kind: str,
            prune_missing_peers: bool,
        ):
            return SimpleNamespace(imported_payloads=0)

    real_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name == "cai_compute_chain.model_distribution":
            return FakeModelDistributionModule
        return real_import_module(name)

    shard = PipelineShardMetadata(
        model_card=ModelCard(
            model_id=ModelId("Qwen/Qwen3-0.6B-GGUF"),
            storage_size=Memory.from_mb(1),
            n_layers=28,
            hidden_size=1,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
            inference_backend=InferenceBackend.LlamaCpp,
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=28,
        n_layers=28,
    )

    with patch("cai.worker.main.importlib.import_module", fake_import_module), patch(
        "cai.worker.main.urlopen"
    ) as urlopen_mock:
        urlopen_mock.return_value.__enter__.return_value.read.return_value = b'{"nodeIdentities": {}}'
        failed = _try_prepare_chunk_backed_llama_cpp_download(
            NodeId("node-a"),
            shard,
            api_port=52415,
        )

    assert isinstance(failed, DownloadFailed)
    assert "full-model fallback is disabled" in failed.error_message
    assert "missing_chunks=2" in failed.error_message


def test_download_model_prefers_chunk_backed_manifest_over_existing_full_path() -> None:
    async def _run() -> None:
        event_send, event_recv = channel[object]()
        command_send, _command_recv = channel[object]()
        download_send, _download_recv = channel[object]()

        worker = object.__new__(Worker)
        worker.node_id = NodeId("node-a")
        worker.event_sender = event_send
        worker.command_sender = command_send
        worker.download_command_sender = download_send
        worker.api_port = 52415
        worker.state = State()
        worker.runners = {}
        worker.input_chunk_buffer = {}
        worker.input_chunk_counts = {}
        worker._system_id = SystemId("system-a")
        worker._download_backoff = KeyedBackoff(base=0.01, cap=0.1)
        worker._instance_backoff = KeyedBackoff(base=0.01, cap=0.1)

        shard = PipelineShardMetadata(
            model_card=ModelCard(
                model_id=ModelId("Qwen/Qwen3-0.6B-GGUF"),
                storage_size=Memory.from_mb(1),
                n_layers=28,
                hidden_size=1,
                supports_tensor=False,
                tasks=[ModelTask.TextGeneration],
                inference_backend=InferenceBackend.LlamaCpp,
            ),
            device_rank=0,
            world_size=1,
            start_layer=0,
            end_layer=28,
            n_layers=28,
        )
        download_task = DownloadModel(
            instance_id=InstanceId("instance-a"),
            shard_metadata=shard,
        )
        chunk_backed_completed = DownloadCompleted(
            node_id=NodeId("node-a"),
            shard_metadata=shard,
            model_directory="D:/tmp/chunk-backed.gguf",
            total=Memory.from_mb(1),
            read_only=False,
        )
        plan_calls = 0

        def fake_plan(*args, **kwargs):
            nonlocal plan_calls
            if plan_calls == 0:
                plan_calls += 1
                return download_task
            return None

        with patch("cai.worker.main.plan", side_effect=fake_plan), patch(
            "cai.worker.main._try_prepare_chunk_backed_llama_cpp_download",
            return_value=chunk_backed_completed,
        ), patch("cai.worker.main.resolve_existing_model") as resolve_existing_model_mock:
            async with anyio.create_task_group() as tg:
                tg.start_soon(worker.plan_step)
                created = await event_recv.receive()
                progress = await event_recv.receive()
                status = await event_recv.receive()
                tg.cancel_scope.cancel()

        assert isinstance(created, TaskCreated)
        assert isinstance(progress, NodeDownloadProgress)
        assert progress.download_progress is chunk_backed_completed
        assert isinstance(status, TaskStatusUpdated)
        assert status.task_status == TaskStatus.Complete
        resolve_existing_model_mock.assert_not_called()
        event_send.close()

    anyio.run(_run)


def test_sync_chunk_backed_peer_inventories_returns_sync_result() -> None:
    recorded_sync_calls: list[tuple[dict[str, object], str, str, bool]] = []

    class FakeModelDistributionModule:
        @staticmethod
        def sync_chunk_inventory_from_cai_peers(
            *,
            state_payload,
            CAI_url: str,
            source_kind: str,
            prune_missing_peers: bool,
        ):
            recorded_sync_calls.append(
                (state_payload, CAI_url, source_kind, prune_missing_peers)
            )
            return SimpleNamespace(imported_payloads=2, pruned_payloads=1)

    real_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name == "cai_compute_chain.model_distribution":
            return FakeModelDistributionModule
        return real_import_module(name)

    with patch("cai.worker.main.importlib.import_module", fake_import_module), patch(
        "cai.worker.main.urlopen"
    ) as urlopen_mock:
        urlopen_mock.return_value.__enter__.return_value.read.return_value = (
            b'{"nodeIdentities": {"peer-a": {}, "peer-b": {}}}'
        )
        result = _sync_chunk_backed_peer_inventories(api_port=52415)

    assert result is not None
    assert getattr(result, "imported_payloads", 0) == 2
    assert getattr(result, "pruned_payloads", 0) == 1
    assert recorded_sync_calls == [
        (
            {"nodeIdentities": {"peer-a": {}, "peer-b": {}}},
            "http://127.0.0.1:52415",
            "peer_cache",
            True,
        )
    ]


def test_sync_chunk_backed_seed_inventories_uses_configured_seed_urls() -> None:
    recorded_sync_calls: list[tuple[tuple[str, ...], str, tuple[str, ...]]] = []

    class FakeModelDistributionModule:
        @staticmethod
        def inventory_endpoint_base_urls(inventory_urls, *, source_kind: str):
            assert source_kind == "storage_seed"
            return tuple(str(item).rstrip("/") for item in inventory_urls)

        @staticmethod
        def sync_chunk_inventory_from_urls(
            *,
            inventory_urls,
            source_kind: str,
            prune_missing_endpoint_base_urls,
        ):
            recorded_sync_calls.append(
                (
                    tuple(inventory_urls),
                    source_kind,
                    tuple(prune_missing_endpoint_base_urls),
                )
            )
            return SimpleNamespace(imported_payloads=1, pruned_payloads=0)

    real_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name == "cai_compute_chain.model_distribution":
            return FakeModelDistributionModule
        return real_import_module(name)

    with patch.dict(
        "os.environ",
        {CAI_CHUNK_SEED_URLS_ENV: "http://203.0.113.20:52415, http://203.0.113.21:52415"},
        clear=False,
    ), patch("cai.worker.main.importlib.import_module", fake_import_module):
        result = _sync_chunk_backed_seed_inventories()

    assert result is not None
    assert getattr(result, "imported_payloads", 0) == 1
    assert recorded_sync_calls == [
        (
            ("http://203.0.113.20:52415", "http://203.0.113.21:52415"),
            "storage_seed",
            ("http://203.0.113.20:52415", "http://203.0.113.21:52415"),
        )
    ]


def test_prefetch_chunk_backed_default_chunks_returns_prefetch_result() -> None:
    recorded_prefetch_calls: list[tuple[str, int, int]] = []

    class FakeModelDistributionModule:
        @staticmethod
        def prefetch_default_chunks_from_fresh_inventories(
            *,
            node_id: str,
            max_manifests: int,
            max_tasks: int,
        ):
            recorded_prefetch_calls.append((node_id, max_manifests, max_tasks))
            return SimpleNamespace(
                manifests_considered=1,
                manifests_prefetched=1,
                queued_tasks=1,
                processed_tasks=1,
            )

    real_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name == "cai_compute_chain.model_distribution":
            return FakeModelDistributionModule
        return real_import_module(name)

    with patch("cai.worker.main.importlib.import_module", fake_import_module):
        result = _prefetch_chunk_backed_default_chunks(NodeId("node-a"))

    assert result is not None
    assert getattr(result, "manifests_prefetched", 0) == 1
    assert recorded_prefetch_calls == [("node-a", 4, 8)]


def test_prefetch_chunk_backed_bootstrap_chunks_returns_prefetch_result() -> None:
    recorded_prefetch_calls: list[tuple[str, int, int]] = []

    class FakeModelDistributionModule:
        @staticmethod
        def prefetch_bootstrap_chunks_from_fresh_inventories(
            *,
            node_id: str,
            max_manifests: int,
            max_tasks: int,
        ):
            recorded_prefetch_calls.append((node_id, max_manifests, max_tasks))
            return SimpleNamespace(
                manifests_considered=1,
                manifests_prefetched=1,
                queued_tasks=2,
                processed_tasks=2,
            )

    real_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name == "cai_compute_chain.model_distribution":
            return FakeModelDistributionModule
        return real_import_module(name)

    with patch("cai.worker.main.importlib.import_module", fake_import_module):
        result = _prefetch_chunk_backed_bootstrap_chunks(NodeId("node-a"))

    assert result is not None
    assert getattr(result, "manifests_prefetched", 0) == 1
    assert recorded_prefetch_calls == [("node-a", 2, 4)]


def test_prefetch_chunk_backed_local_shard_hints_uses_runner_ranges() -> None:
    recorded_prefetch_calls: list[list[dict[str, object]]] = []

    class FakeModelDistributionModule:
        @staticmethod
        def prefetch_hinted_bootstrap_chunks(hints):
            recorded_prefetch_calls.append(list(hints))
            return SimpleNamespace(
                manifests_considered=1,
                manifests_prefetched=1,
                queued_tasks=2,
                processed_tasks=2,
            )

    real_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name == "cai_compute_chain.model_distribution":
            return FakeModelDistributionModule
        return real_import_module(name)

    shard = PipelineShardMetadata(
        model_card=ModelCard(
            model_id=ModelId("Qwen/Qwen3-0.6B-GGUF"),
            storage_size=Memory.from_mb(1),
            n_layers=28,
            hidden_size=1,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
            inference_backend=InferenceBackend.LlamaCpp,
        ),
        device_rank=1,
        world_size=3,
        start_layer=10,
        end_layer=18,
        n_layers=28,
    )

    with patch("cai.worker.main.importlib.import_module", fake_import_module):
        result = _prefetch_chunk_backed_local_shard_hints(NodeId("node-a"), [shard])

    assert result is not None
    assert getattr(result, "manifests_prefetched", 0) == 1
    assert recorded_prefetch_calls == [
        [
            {
                "model_id": "Qwen/Qwen3-0.6B-GGUF",
                "start_layer": 10,
                "end_layer": 18,
                "device_rank": 1,
                "world_size": 3,
                "node_id": "node-a",
            }
        ]
    ]


def test_remember_chunk_backed_local_shard_hints_persists_runner_ranges() -> None:
    recorded_hint_calls: list[tuple[str, list[dict[str, object]]]] = []

    class FakeModelDistributionModule:
        @staticmethod
        def remember_recent_shard_hints(node_id, hints):
            recorded_hint_calls.append((str(node_id), list(hints)))
            return SimpleNamespace(
                hints_received=len(hints),
                records_upserted=len(hints),
                records_pruned=0,
                stored_records=len(hints),
            )

    real_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name == "cai_compute_chain.model_distribution":
            return FakeModelDistributionModule
        return real_import_module(name)

    shard = PipelineShardMetadata(
        model_card=ModelCard(
            model_id=ModelId("Qwen/Qwen3-0.6B-GGUF"),
            storage_size=Memory.from_mb(1),
            n_layers=28,
            hidden_size=1,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
            inference_backend=InferenceBackend.LlamaCpp,
        ),
        device_rank=2,
        world_size=4,
        start_layer=18,
        end_layer=28,
        n_layers=28,
    )

    with patch("cai.worker.main.importlib.import_module", fake_import_module):
        result = _remember_chunk_backed_local_shard_hints(NodeId("node-a"), [shard])

    assert result is not None
    assert getattr(result, "records_upserted", 0) == 1
    assert recorded_hint_calls == [
        (
            "node-a",
            [
                {
                    "model_id": "Qwen/Qwen3-0.6B-GGUF",
                    "start_layer": 18,
                    "end_layer": 28,
                    "device_rank": 2,
                    "world_size": 4,
                }
            ],
        )
    ]


def test_prefetch_chunk_backed_recent_shard_hints_uses_node_memory() -> None:
    recorded_prefetch_calls: list[str] = []

    class FakeModelDistributionModule:
        @staticmethod
        def prefetch_recent_shard_hints(node_id):
            recorded_prefetch_calls.append(str(node_id))
            return SimpleNamespace(
                manifests_considered=1,
                manifests_prefetched=1,
                queued_tasks=1,
                processed_tasks=1,
            )

    real_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name == "cai_compute_chain.model_distribution":
            return FakeModelDistributionModule
        return real_import_module(name)

    with patch("cai.worker.main.importlib.import_module", fake_import_module):
        result = _prefetch_chunk_backed_recent_shard_hints(NodeId("node-a"))

    assert result is not None
    assert getattr(result, "manifests_prefetched", 0) == 1
    assert recorded_prefetch_calls == ["node-a"]


def test_remember_chunk_backed_instance_shard_hints_uses_state_assignments() -> None:
    recorded_hint_calls: list[tuple[str, list[dict[str, object]]]] = []

    class FakeModelDistributionModule:
        @staticmethod
        def remember_recent_shard_hints(node_id, hints):
            recorded_hint_calls.append((str(node_id), list(hints)))
            return SimpleNamespace(
                hints_received=len(hints),
                records_upserted=len(hints),
                records_pruned=0,
                stored_records=len(hints),
            )

    real_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name == "cai_compute_chain.model_distribution":
            return FakeModelDistributionModule
        return real_import_module(name)

    runner_id = RunnerId("runner-a")
    shard = PipelineShardMetadata(
        model_card=ModelCard(
            model_id=ModelId("Qwen/Qwen3-0.6B-GGUF"),
            storage_size=Memory.from_mb(1),
            n_layers=28,
            hidden_size=1,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
            inference_backend=InferenceBackend.LlamaCpp,
        ),
        device_rank=0,
        world_size=2,
        start_layer=0,
        end_layer=14,
        n_layers=28,
    )
    instance = MlxRingInstance(
        instance_id=InstanceId("instance-a"),
        hosts_by_node={NodeId("node-a"): [], NodeId("node-b"): []},
        ephemeral_port=1,
        shard_assignments=ShardAssignments(
            model_id=ModelId("Qwen/Qwen3-0.6B-GGUF"),
            runner_to_shard={runner_id: shard},
            node_to_runner={NodeId("node-a"): runner_id},
        ),
    )

    with patch("cai.worker.main.importlib.import_module", fake_import_module):
        result = _remember_chunk_backed_instance_shard_hints(NodeId("node-a"), [instance])

    assert result is not None
    assert getattr(result, "records_upserted", 0) == 1
    assert recorded_hint_calls == [
        (
            "node-a",
            [
                {
                    "model_id": "Qwen/Qwen3-0.6B-GGUF",
                    "start_layer": 0,
                    "end_layer": 14,
                    "device_rank": 0,
                    "world_size": 2,
                    "node_id": "node-a",
                }
            ],
        )
    ]


def test_prefetch_chunk_backed_instance_shard_hints_uses_state_assignments() -> None:
    recorded_prefetch_calls: list[list[dict[str, object]]] = []

    class FakeModelDistributionModule:
        @staticmethod
        def prefetch_hinted_bootstrap_chunks(hints):
            recorded_prefetch_calls.append(list(hints))
            return SimpleNamespace(
                manifests_considered=1,
                manifests_prefetched=1,
                queued_tasks=1,
                processed_tasks=1,
            )

    real_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name == "cai_compute_chain.model_distribution":
            return FakeModelDistributionModule
        return real_import_module(name)

    runner_id = RunnerId("runner-a")
    shard = PipelineShardMetadata(
        model_card=ModelCard(
            model_id=ModelId("Qwen/Qwen3-0.6B-GGUF"),
            storage_size=Memory.from_mb(1),
            n_layers=28,
            hidden_size=1,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
            inference_backend=InferenceBackend.LlamaCpp,
        ),
        device_rank=1,
        world_size=2,
        start_layer=14,
        end_layer=28,
        n_layers=28,
    )
    instance = MlxRingInstance(
        instance_id=InstanceId("instance-a"),
        hosts_by_node={NodeId("node-a"): [], NodeId("node-b"): []},
        ephemeral_port=1,
        shard_assignments=ShardAssignments(
            model_id=ModelId("Qwen/Qwen3-0.6B-GGUF"),
            runner_to_shard={runner_id: shard},
            node_to_runner={NodeId("node-a"): runner_id},
        ),
    )

    with patch("cai.worker.main.importlib.import_module", fake_import_module):
        result = _prefetch_chunk_backed_instance_shard_hints(NodeId("node-a"), [instance])

    assert result is not None
    assert getattr(result, "manifests_prefetched", 0) == 1
    assert recorded_prefetch_calls == [
        [
            {
                "model_id": "Qwen/Qwen3-0.6B-GGUF",
                "start_layer": 14,
                "end_layer": 28,
                "device_rank": 1,
                "world_size": 2,
                "node_id": "node-a",
            }
        ]
    ]


def test_release_chunk_backed_assignment_cache_protects_chunks_for_other_local_shards() -> None:
    @dataclass(frozen=True)
    class FakeChunk:
        chunk_id: str

    @dataclass(frozen=True)
    class FakeManifest:
        catalog_id: str = "demo"
        version: str = "v1"

        @staticmethod
        def required_chunks_for_layers(start_layer: int, end_layer: int):
            if start_layer == 0 and end_layer == 14:
                return (
                    FakeChunk("shared"),
                    FakeChunk("left-only"),
                )
            if start_layer == 14 and end_layer == 28:
                return (
                    FakeChunk("shared"),
                    FakeChunk("right-only"),
                )
            return ()

    @dataclass(frozen=True)
    class FakeAssignment:
        start_layer: int
        end_layer: int
        device_rank: int = 0
        world_size: int = 1
        node_id: str | None = None

    recorded_release_calls: list[tuple[FakeManifest, FakeAssignment, set[str]]] = []
    real_import_module = importlib.import_module

    class FakeModelDistributionModule:
        ModelShardAssignment = FakeAssignment

        @staticmethod
        def select_model_package_manifest_for_model(model_id: str):
            return FakeManifest() if model_id == "Qwen/Qwen3-0.6B-GGUF" else None

        @staticmethod
        def release_assignment_cache_policy_from_store(
            manifest,
            assignment,
            *,
            protected_chunk_ids,
        ):
            recorded_release_calls.append(
                (manifest, assignment, set(protected_chunk_ids))
            )
            return (object(),)

    def fake_import_module(name: str):
        if name == "cai_compute_chain.model_distribution":
            return FakeModelDistributionModule
        return real_import_module(name)

    shard = PipelineShardMetadata(
        model_card=ModelCard(
            model_id=ModelId("Qwen/Qwen3-0.6B-GGUF"),
            storage_size=Memory.from_mb(1),
            n_layers=28,
            hidden_size=1,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
            inference_backend=InferenceBackend.LlamaCpp,
        ),
        device_rank=0,
        world_size=2,
        start_layer=0,
        end_layer=14,
        n_layers=28,
    )
    other_shard = PipelineShardMetadata(
        model_card=ModelCard(
            model_id=ModelId("Qwen/Qwen3-0.6B-GGUF"),
            storage_size=Memory.from_mb(1),
            n_layers=28,
            hidden_size=1,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
            inference_backend=InferenceBackend.LlamaCpp,
        ),
        device_rank=1,
        world_size=2,
        start_layer=14,
        end_layer=28,
        n_layers=28,
    )

    with patch("cai.worker.main.importlib.import_module", fake_import_module):
        released = _release_chunk_backed_assignment_cache(
            NodeId("node-a"),
            shard,
            other_shards=[other_shard],
        )

    assert released is True
    assert len(recorded_release_calls) == 1
    _manifest, assignment, protected_chunk_ids = recorded_release_calls[0]
    assert assignment.start_layer == 0
    assert assignment.end_layer == 14
    assert assignment.node_id == "node-a"
    assert protected_chunk_ids == {"shared", "right-only"}


def test_release_chunk_backed_assignment_cache_cleans_last_materialized_model_path() -> None:
    @dataclass(frozen=True)
    class FakeManifest:
        catalog_id: str = "demo"
        version: str = "v1"

        @staticmethod
        def required_chunks_for_layers(start_layer: int, end_layer: int):
            return ()

    @dataclass(frozen=True)
    class FakeAssignment:
        start_layer: int
        end_layer: int
        device_rank: int = 0
        world_size: int = 1
        node_id: str | None = None

    class FakeModelDistributionModule:
        ModelShardAssignment = FakeAssignment

        @staticmethod
        def select_model_package_manifest_for_model(model_id: str):
            return FakeManifest() if model_id == "Qwen/Qwen3-0.6B-GGUF" else None

        @staticmethod
        def release_assignment_cache_policy_from_store(
            manifest,
            assignment,
            *,
            protected_chunk_ids,
        ):
            return ()

        @staticmethod
        def select_default_materialized_artifact_id(manifest):
            return "gguf-main"

        @staticmethod
        def materialized_artifact_path(manifest, artifact_id):
            assert artifact_id == "gguf-main"
            return materialized_path

    real_import_module = importlib.import_module
    deleted_model_ids: list[ModelId] = []

    def fake_import_module(name: str):
        if name == "cai_compute_chain.model_distribution":
            return FakeModelDistributionModule
        return real_import_module(name)

    def fake_delete_custom_model_local_path(model_id: ModelId):
        deleted_model_ids.append(model_id)
        return True

    shard = PipelineShardMetadata(
        model_card=ModelCard(
            model_id=ModelId("Qwen/Qwen3-0.6B-GGUF"),
            storage_size=Memory.from_mb(1),
            n_layers=28,
            hidden_size=1,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
            inference_backend=InferenceBackend.LlamaCpp,
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=28,
        n_layers=28,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        materialized_path = Path(temp_dir) / "materialized.gguf"
        materialized_path.write_bytes(b"gguf")

        with patch("cai.worker.main.importlib.import_module", fake_import_module), patch(
            "cai.worker.main.get_custom_model_local_path",
            return_value=materialized_path,
        ), patch(
            "cai.worker.main.delete_custom_model_local_path",
            fake_delete_custom_model_local_path,
        ):
            released = _release_chunk_backed_assignment_cache(
                NodeId("node-a"),
                shard,
            )
        assert materialized_path.exists() is False

    assert released is True
    assert deleted_model_ids == [ModelId("Qwen/Qwen3-0.6B-GGUF")]


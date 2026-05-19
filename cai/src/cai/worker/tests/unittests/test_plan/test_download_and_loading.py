# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from datetime import UTC, datetime
from pathlib import Path

import cai_compute_chain.model_distribution as model_dist
import cai.worker.plan as plan_mod
from cai.shared.models.model_cards import InferenceBackend, ModelCard, ModelTask
from cai.shared.types.tasks import ConnectToGroup, TaskStatus
from cai.shared.types.common import NodeId
from cai.shared.types.memory import Memory
from cai.shared.types.tasks import LoadModel
from cai.shared.types.worker.downloads import DownloadCompleted, DownloadProgress
from cai.shared.types.worker.instances import BoundInstance, InstanceId, MlxRingInstance
from cai.shared.types.worker.runners import (
    RunnerConnected,
    RunnerIdle,
    RunnerId,
    ShardAssignments,
)
from cai.shared.types.worker.shards import PipelineShardMetadata
from cai.utils.keyed_backoff import KeyedBackoff
from cai.worker.tests.constants import (
    INSTANCE_1_ID,
    MODEL_A_ID,
    NODE_A,
    NODE_B,
    RUNNER_1_ID,
    RUNNER_2_ID,
)
from cai.worker.tests.unittests.conftest import (
    FakeRunnerSupervisor,
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)


def _write_cai_manifest_for_llama_model(tmp_path: Path, model_id: str) -> model_dist.ModelPackageManifest:
    artifact_bytes = b"llama-cpp-chunk-ready"
    artifact_path = tmp_path / "model.gguf"
    artifact_path.write_bytes(artifact_bytes)
    manifest = model_dist.build_gguf_model_package_manifest(
        catalog_id="qwen2-0.5b-ready",
        model_id=model_id,
        version="2026.04.25",
        gguf_path=artifact_path,
        total_layers=8,
        source_repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    )
    model_dist.save_model_package_manifest(manifest)
    for chunk in manifest.chunks:
        payload = artifact_bytes[
            chunk.offset_bytes : chunk.offset_bytes + chunk.size_bytes
        ]
        model_dist.put_cached_chunk(
            catalog_id=manifest.catalog_id,
            version=manifest.version,
            chunk_id=chunk.chunk_id,
            sha256_hex=chunk.sha256_hex,
            content=payload,
        )
    return manifest


def _build_single_node_llama_bound_instance(*, model_id: str) -> BoundInstance:
    node_id = NodeId("node-a")
    runner_id = RunnerId("runner-a")
    model_card = ModelCard(
        model_id=model_id,
        storage_size=Memory.from_mb(1),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )
    instance = MlxRingInstance(
        instance_id=InstanceId("instance-llama-single"),
        shard_assignments=ShardAssignments(
            model_id=model_id,
            node_to_runner={node_id: runner_id},
            runner_to_shard={
                runner_id: PipelineShardMetadata(
                    model_card=model_card,
                    device_rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=8,
                    n_layers=8,
                )
            },
        ),
        hosts_by_node={},
        ephemeral_port=50053,
    )
    return BoundInstance(
        instance=instance,
        bound_runner_id=runner_id,
        bound_node_id=node_id,
    )


def _build_two_node_llama_bound_instance(*, device_rank: int) -> BoundInstance:
    model_id = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    model_card = ModelCard(
        model_id=model_id,
        storage_size=Memory.from_mb(1),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    runner_a = RunnerId("runner-a")
    runner_b = RunnerId("runner-b")
    instance = MlxRingInstance(
        instance_id=InstanceId("instance-llama-distributed"),
        shard_assignments=ShardAssignments(
            model_id=model_id,
            node_to_runner={node_a: runner_a, node_b: runner_b},
            runner_to_shard={
                runner_a: PipelineShardMetadata(
                    model_card=model_card,
                    device_rank=0,
                    world_size=2,
                    start_layer=0,
                    end_layer=4,
                    n_layers=8,
                ),
                runner_b: PipelineShardMetadata(
                    model_card=model_card,
                    device_rank=1,
                    world_size=2,
                    start_layer=4,
                    end_layer=8,
                    n_layers=8,
                ),
            },
        ),
        hosts_by_node={},
        ephemeral_port=50054,
    )
    return BoundInstance(
        instance=instance,
        bound_runner_id=runner_a if device_rank == 0 else runner_b,
        bound_node_id=node_a if device_rank == 0 else node_b,
    )


def test_plan_requests_download_when_waiting_and_shard_not_downloaded():
    """
    When a runner is waiting for a model and its shard is not in the
    local download_status map, plan() should emit DownloadModel.
    """

    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID},
        runner_to_shard={RUNNER_1_ID: shard},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerIdle())

    runners = {RUNNER_1_ID: runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {RUNNER_1_ID: RunnerIdle()}

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert isinstance(result, plan_mod.DownloadModel)
    assert result.instance_id == INSTANCE_1_ID
    assert result.shard_metadata == shard


def test_plan_prefers_connect_to_group_before_download_for_multinode_instance():
    """
    Distributed runner lifecycle should advance to ConnectToGroup before
    spending another planner tick on local shard download work.
    """
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard2 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard1, RUNNER_2_ID: shard2},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerIdle())

    runners = {RUNNER_1_ID: runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerIdle(),
        RUNNER_2_ID: RunnerIdle(),
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        global_download_status={NODE_A: [], NODE_B: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert isinstance(result, ConnectToGroup)
    assert result.instance_id == INSTANCE_1_ID


def test_plan_does_not_duplicate_download_task_for_same_model():
    """
    Once a DownloadModel task is already Pending/Running for the local model,
    plan() should not emit another duplicate download task.
    """
    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID},
        runner_to_shard={RUNNER_1_ID: shard},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerIdle())

    existing_download = plan_mod.DownloadModel(
        instance_id=INSTANCE_1_ID,
        shard_metadata=shard,
        task_status=TaskStatus.Running,
    )

    result = plan_mod.plan(
        node_id=NODE_A,
        runners={RUNNER_1_ID: runner},  # type: ignore
        global_download_status={NODE_A: []},
        instances={INSTANCE_1_ID: instance},
        all_runners={RUNNER_1_ID: RunnerIdle()},
        tasks={existing_download.task_id: existing_download},
        input_chunk_buffer={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert result is None


def test_plan_loads_model_when_all_shards_downloaded_and_waiting():
    """
    When all shards for an instance are DownloadCompleted (globally) and
    all runners are in waiting/loading/loaded states, plan() should emit
    LoadModel once.
    """
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard2 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard1, RUNNER_2_ID: shard2},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerConnected()
    )

    runners = {RUNNER_1_ID: local_runner}
    instances = {INSTANCE_1_ID: instance}

    all_runners = {
        RUNNER_1_ID: RunnerConnected(),
        RUNNER_2_ID: RunnerConnected(),
    }

    global_download_status = {
        NODE_A: [
            DownloadCompleted(shard_metadata=shard1, node_id=NODE_A, total=Memory())
        ],
        NODE_B: [
            DownloadCompleted(shard_metadata=shard2, node_id=NODE_B, total=Memory())
        ],
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        global_download_status=global_download_status,
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert isinstance(result, LoadModel)
    assert result.instance_id == INSTANCE_1_ID


def test_plan_does_not_request_download_when_shard_already_downloaded():
    """
    If the local shard already has a DownloadCompleted entry, plan()
    should not re-emit DownloadModel while global state is still catching up.
    """
    shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID},
        runner_to_shard={RUNNER_1_ID: shard},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerIdle())

    runners = {RUNNER_1_ID: runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {RUNNER_1_ID: RunnerIdle()}

    # Global state shows shard is downloaded for NODE_A
    global_download_status: dict[NodeId, list[DownloadProgress]] = {
        NODE_A: [
            DownloadCompleted(shard_metadata=shard, node_id=NODE_A, total=Memory())
        ],
        NODE_B: [],
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        global_download_status=global_download_status,
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert not isinstance(result, plan_mod.DownloadModel)


def test_plan_does_not_load_model_until_all_shards_downloaded_globally():
    """
    LoadModel should not be emitted while some shards are still missing from
    the global_download_status.
    """
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard2 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard1, RUNNER_2_ID: shard2},
    )

    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerConnected()
    )

    runners = {RUNNER_1_ID: local_runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerConnected(),
        RUNNER_2_ID: RunnerConnected(),
    }

    global_download_status = {
        NODE_A: [
            DownloadCompleted(shard_metadata=shard1, node_id=NODE_A, total=Memory())
        ],
        NODE_B: [],  # NODE_B has no downloads completed yet
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        global_download_status=global_download_status,
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert result is None

    global_download_status = {
        NODE_A: [
            DownloadCompleted(shard_metadata=shard1, node_id=NODE_A, total=Memory())
        ],
        NODE_B: [
            DownloadCompleted(shard_metadata=shard2, node_id=NODE_B, total=Memory())
        ],  # NODE_B has no downloads completed yet
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        global_download_status=global_download_status,
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert result is not None


def test_plan_loads_single_node_llama_cpp_when_assigned_chunks_are_already_cached(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(model_dist, "data_root", lambda policy=None: tmp_path)
    plan_mod._get_cai_model_distribution_module.cache_clear()

    bound_instance = _build_single_node_llama_bound_instance(
        model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    )
    _write_cai_manifest_for_llama_model(
        tmp_path,
        str(bound_instance.bound_shard.model_card.model_id),
    )
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerIdle())
    instance = bound_instance.instance

    result = plan_mod.plan(
        node_id=bound_instance.bound_node_id,
        runners={bound_instance.bound_runner_id: runner},  # type: ignore[arg-type]
        global_download_status={bound_instance.bound_node_id: []},
        instances={instance.instance_id: instance},
        all_runners={bound_instance.bound_runner_id: RunnerIdle()},
        tasks={},
        input_chunk_buffer={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert isinstance(result, LoadModel)
    assert result.instance_id == instance.instance_id


def test_plan_loads_distributed_llama_cpp_when_remote_peer_inventory_covers_shard(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(model_dist, "data_root", lambda policy=None: tmp_path)
    plan_mod._get_cai_model_distribution_module.cache_clear()

    bound_instance = _build_two_node_llama_bound_instance(device_rank=0)
    manifest = _write_cai_manifest_for_llama_model(
        tmp_path,
        str(bound_instance.bound_shard.model_card.model_id),
    )
    peer_chunk_ids = [
        chunk.chunk_id
        for chunk in manifest.required_chunks_for_layers(4, 8)
        if chunk.kind == model_dist.ModelChunkKind.WEIGHTS
    ]
    model_dist.import_chunk_inventory_payload(
        model_dist.ChunkInventoryPayload(
            source_id="node-b",
            source_kind=model_dist.ChunkInventorySourceKind.PEER_CACHE,
            published_at=datetime.now(tz=UTC).isoformat(),
            records=(
                model_dist.ChunkInventoryRecord(
                    catalog_id=manifest.catalog_id,
                    version=manifest.version,
                    chunk_ids=tuple(peer_chunk_ids),
                    chunk_count=len(peer_chunk_ids),
                    total_bytes=sum(
                        chunk.size_bytes
                        for chunk in manifest.chunks
                        if chunk.chunk_id in peer_chunk_ids
                    ),
                ),
            ),
        )
    )

    runner = FakeRunnerSupervisor(
        bound_instance=bound_instance,
        status=RunnerConnected(),
    )
    instance = bound_instance.instance

    result = plan_mod.plan(
        node_id=bound_instance.bound_node_id,
        runners={bound_instance.bound_runner_id: runner},  # type: ignore[arg-type]
        global_download_status={
            bound_instance.bound_node_id: [],
            NodeId("node-b"): [],
        },
        instances={instance.instance_id: instance},
        all_runners={
            RunnerId("runner-a"): RunnerConnected(),
            RunnerId("runner-b"): RunnerConnected(),
        },
        tasks={},
        input_chunk_buffer={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert isinstance(result, LoadModel)
    assert result.instance_id == instance.instance_id


def test_plan_loads_distributed_llama_cpp_without_chunk_manifest(monkeypatch):
    class NonChunkedModelDistribution:
        @staticmethod
        def select_model_package_manifest_for_model(model_id: str):
            return None

    plan_mod._get_cai_model_distribution_module.cache_clear()
    monkeypatch.setattr(
        plan_mod,
        "_get_cai_model_distribution_module",
        lambda: NonChunkedModelDistribution,
    )
    monkeypatch.setattr(
        plan_mod,
        "resolve_existing_model",
        lambda model_id: Path("D:/tmp/model.gguf"),
    )

    bound_instance = _build_two_node_llama_bound_instance(device_rank=0)
    runner = FakeRunnerSupervisor(
        bound_instance=bound_instance,
        status=RunnerConnected(),
    )
    instance = bound_instance.instance

    result = plan_mod.plan(
        node_id=bound_instance.bound_node_id,
        runners={bound_instance.bound_runner_id: runner},  # type: ignore[arg-type]
        global_download_status={
            bound_instance.bound_node_id: [],
            NodeId("node-b"): [],
        },
        instances={instance.instance_id: instance},
        all_runners={
            RunnerId("runner-a"): RunnerConnected(),
            RunnerId("runner-b"): RunnerConnected(),
        },
        tasks={},
        input_chunk_buffer={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert isinstance(result, LoadModel)
    assert result.instance_id == instance.instance_id


def test_plan_downloads_single_node_llama_cpp_without_chunk_manifest_until_local_model_exists(
    monkeypatch,
):
    class NonChunkedModelDistribution:
        @staticmethod
        def select_model_package_manifest_for_model(model_id: str):
            return None

    plan_mod._get_cai_model_distribution_module.cache_clear()
    monkeypatch.setattr(
        plan_mod,
        "_get_cai_model_distribution_module",
        lambda: NonChunkedModelDistribution,
    )
    monkeypatch.setattr(plan_mod, "resolve_existing_model", lambda model_id: None)

    bound_instance = _build_single_node_llama_bound_instance(
        model_id="Qwen/Qwen3-0.6B-GGUF"
    )
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerIdle())
    instance = bound_instance.instance

    result = plan_mod.plan(
        node_id=bound_instance.bound_node_id,
        runners={bound_instance.bound_runner_id: runner},  # type: ignore[arg-type]
        global_download_status={bound_instance.bound_node_id: []},
        instances={instance.instance_id: instance},
        all_runners={bound_instance.bound_runner_id: RunnerIdle()},
        tasks={},
        input_chunk_buffer={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert isinstance(result, plan_mod.DownloadModel)
    assert result.instance_id == instance.instance_id


def test_plan_loads_single_node_llama_cpp_without_chunk_manifest_when_local_model_exists(
    monkeypatch,
):
    class NonChunkedModelDistribution:
        @staticmethod
        def select_model_package_manifest_for_model(model_id: str):
            return None

    plan_mod._get_cai_model_distribution_module.cache_clear()
    monkeypatch.setattr(
        plan_mod,
        "_get_cai_model_distribution_module",
        lambda: NonChunkedModelDistribution,
    )
    monkeypatch.setattr(
        plan_mod,
        "resolve_existing_model",
        lambda model_id: Path("D:/tmp/model.gguf"),
    )

    bound_instance = _build_single_node_llama_bound_instance(
        model_id="Qwen/Qwen3-0.6B-GGUF"
    )
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerIdle())
    instance = bound_instance.instance

    result = plan_mod.plan(
        node_id=bound_instance.bound_node_id,
        runners={bound_instance.bound_runner_id: runner},  # type: ignore[arg-type]
        global_download_status={bound_instance.bound_node_id: []},
        instances={instance.instance_id: instance},
        all_runners={bound_instance.bound_runner_id: RunnerIdle()},
        tasks={},
        input_chunk_buffer={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert isinstance(result, LoadModel)
    assert result.instance_id == instance.instance_id


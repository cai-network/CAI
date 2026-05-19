# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import cai.worker.plan as plan_mod
from cai.shared.models.model_cards import InferenceBackend, ModelCard, ModelId, ModelTask
from cai.shared.types.common import NodeId
from cai.shared.types.memory import Memory
from cai.shared.types.tasks import ConnectToGroup, StartWarmup
from cai.shared.types.worker.instances import BoundInstance, InstanceId, MlxRingInstance
from cai.shared.types.worker.runners import (
    RunnerId,
    RunnerIdle,
    RunnerLoaded,
    ShardAssignments,
)
from cai.shared.types.worker.shards import PipelineShardMetadata
from cai.utils.keyed_backoff import KeyedBackoff
from cai.worker.tests.unittests.conftest import FakeRunnerSupervisor


def _make_bound_instance(*, device_rank: int) -> BoundInstance:
    model_id = ModelId("Qwen/Qwen2.5-0.5B-Instruct-GGUF")
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
        instance_id=InstanceId("instance-a"),
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
        ephemeral_port=50052,
    )
    return BoundInstance(
        instance=instance,
        bound_runner_id=runner_a if device_rank == 0 else runner_b,
        bound_node_id=node_a if device_rank == 0 else node_b,
    )


def test_plan_connects_second_llama_cpp_rank_without_waiting_for_connecting_state() -> None:
    bound_instance = _make_bound_instance(device_rank=1)
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerIdle())
    instance = bound_instance.instance

    result = plan_mod.plan(
        node_id=bound_instance.bound_node_id,
        runners={bound_instance.bound_runner_id: runner},  # type: ignore[arg-type]
        global_download_status={bound_instance.bound_node_id: []},
        instances={instance.instance_id: instance},
        all_runners={
            RunnerId("runner-a"): RunnerIdle(),
            RunnerId("runner-b"): RunnerIdle(),
        },
        tasks={},
        input_chunk_buffer={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert isinstance(result, ConnectToGroup)
    assert result.instance_id == instance.instance_id


def test_plan_allows_llama_cpp_rank_zero_to_warmup_once_everyone_is_loaded() -> None:
    bound_instance = _make_bound_instance(device_rank=0)
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerLoaded())
    instance = bound_instance.instance

    result = plan_mod.plan(
        node_id=bound_instance.bound_node_id,
        runners={bound_instance.bound_runner_id: runner},  # type: ignore[arg-type]
        global_download_status={bound_instance.bound_node_id: []},
        instances={instance.instance_id: instance},
        all_runners={
            RunnerId("runner-a"): RunnerLoaded(),
            RunnerId("runner-b"): RunnerLoaded(),
        },
        tasks={},
        input_chunk_buffer={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert isinstance(result, StartWarmup)
    assert result.instance_id == instance.instance_id


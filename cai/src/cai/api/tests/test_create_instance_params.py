# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import json

from cai.api.types.api import CreateInstanceParams
from cai.shared.models.model_cards import InferenceBackend, ModelCard, ModelTask
from cai.shared.types.common import Host, ModelId, NodeId
from cai.shared.types.memory import Memory
from cai.shared.types.worker.instances import InstanceId, MlxRingInstance
from cai.shared.types.worker.runners import RunnerId, ShardAssignments
from cai.shared.types.worker.shards import PipelineShardMetadata


def test_create_instance_params_accepts_preview_payload_round_trip() -> None:
    runner_local = RunnerId("runner-local")
    runner_remote = RunnerId("runner-remote")
    node_local = NodeId("node-local")
    node_remote = NodeId("node-remote")

    model_card = ModelCard(
        model_id=ModelId("Qwen/Qwen3-0.6B-GGUF"),
        storage_size=Memory.from_bytes(123),
        n_layers=28,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family="qwen3",
        quantization="Q8_0",
        base_model="Qwen/Qwen3-0.6B",
        context_length=40960,
        trust_remote_code=False,
        is_custom=False,
        inference_backend=InferenceBackend.LlamaCpp,
        preferred_filename="Qwen3-0.6B-Q8_0.gguf",
    )
    shard_assignments = ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard={
            runner_local: PipelineShardMetadata(
                model_card=model_card,
                device_rank=0,
                world_size=2,
                start_layer=0,
                end_layer=25,
                n_layers=28,
            ),
            runner_remote: PipelineShardMetadata(
                model_card=model_card,
                device_rank=1,
                world_size=2,
                start_layer=25,
                end_layer=28,
                n_layers=28,
            ),
        },
        node_to_runner={
            node_local: runner_local,
            node_remote: runner_remote,
        },
    )
    instance = MlxRingInstance(
        instance_id=InstanceId("instance-1"),
        shard_assignments=shard_assignments,
        hosts_by_node={
            node_local: [Host(ip="127.0.0.1", port=52415)],
            node_remote: [Host(ip="85.137.164.250", port=52425)],
        },
        ephemeral_port=37111,
    )

    preview_payload = json.loads(instance.model_dump_json(by_alias=True))
    params = CreateInstanceParams.model_validate({"instance": preview_payload})

    assert params.instance.instance_id == instance.instance_id
    assert params.instance.shard_assignments.model_id == model_card.model_id
    local_shard = params.instance.shard_assignments.runner_to_shard[runner_local]
    assert local_shard.start_layer == 0
    assert local_shard.end_layer == 25
    assert local_shard.model_card.inference_backend == InferenceBackend.LlamaCpp


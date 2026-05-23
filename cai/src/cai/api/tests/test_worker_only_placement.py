# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from cai.api.main import API
from cai.api.model_placement_policy import llama_cpp_layer_range_supported
from cai.api.types.api import CreateInstanceParams
from cai.shared.types.commands import CreateInstance, TestCommand
from cai.master.tests.conftest import create_socket_connection
from cai.shared.topology import Topology
from cai.shared.models.model_cards import InferenceBackend, ModelCard, ModelTask
from cai.shared.types.profiling import (
    AdvertisedTransportEndpoint,
    MemoryUsage,
    NetworkInterfaceInfo,
    NodeIdentity,
    NodeNetworkInfo,
)
from cai.shared.types.common import Host, ModelId, NodeId, SystemId
from cai.shared.types.memory import Memory
from cai.shared.types.state import State
from cai.shared.types.topology import Connection
from cai.shared.types.worker.instances import InstanceId, InstanceMeta, MlxRingInstance
from cai.shared.types.worker.runners import RunnerId, ShardAssignments
from cai.shared.types.worker.shards import PipelineShardMetadata, Sharding
from cai_compute_chain.node_capabilities import NodeCapabilityRecord


def test_resolve_worker_required_nodes_filters_out_validator_nodes() -> None:
    api = object.__new__(API)
    api.port = 52415
    api.node_id = NodeId("node-local")
    api.state = SimpleNamespace(
        node_identities={
            NodeId("node-local"): {"apiPort": 52415, "apiHost": "127.0.0.1"},
            NodeId("node-validator"): {"apiPort": 52415, "apiHost": "85.137.164.250"},
        }
    )

    with (
        patch.object(API, "_load_local_worker_enabled", return_value=True),
        patch.object(
            API,
            "_load_local_worker_reward_address",
            return_value="ABCD1234ABCD1234ABCD1234ABCD1234",
        ),
        patch("cai.api.main._load_json_url", return_value={"worker": {"worker_enabled": False}}),
    ):
        required_nodes = api._resolve_worker_required_nodes()  # pyright: ignore[reportPrivateUsage]

    assert required_nodes == {NodeId("node-local")}


def test_validate_worker_only_instance_rejects_validator_participants() -> None:
    api = object.__new__(API)
    api.port = 52415
    api.node_id = NodeId("node-local")
    api.state = SimpleNamespace(
        node_identities={
            NodeId("node-local"): {"apiPort": 52415, "apiHost": "127.0.0.1"},
            NodeId("node-validator"): {"apiPort": 52416, "apiHost": "85.137.164.250"},
        }
    )

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

    instance = MlxRingInstance(
        instance_id=InstanceId("instance-1"),
        shard_assignments=ShardAssignments(
            model_id=model_card.model_id,
            runner_to_shard={
                RunnerId("runner-local"): PipelineShardMetadata(
                    model_card=model_card,
                    device_rank=0,
                    world_size=2,
                    start_layer=0,
                    end_layer=25,
                    n_layers=28,
                ),
                RunnerId("runner-validator"): PipelineShardMetadata(
                    model_card=model_card,
                    device_rank=1,
                    world_size=2,
                    start_layer=25,
                    end_layer=28,
                    n_layers=28,
                ),
            },
            node_to_runner={
                NodeId("node-local"): RunnerId("runner-local"),
                NodeId("node-validator"): RunnerId("runner-validator"),
            },
        ),
        hosts_by_node={
            NodeId("node-local"): [Host(ip="127.0.0.1", port=52415)],
            NodeId("node-validator"): [Host(ip="85.137.164.250", port=52415)],
        },
        ephemeral_port=37111,
    )

    def _fake_load_json(url: str, *, timeout: int = 5):  # noqa: ARG001
        if url.startswith("http://127.0.0.1:52415/"):
            return {"worker": {"worker_enabled": True}}
        return {"worker": {"worker_enabled": False}}

    with patch("cai.api.main._load_json_url", side_effect=_fake_load_json):
        with pytest.raises(HTTPException, match="worker-enabled CAI nodes"):
            api._validate_worker_only_instance(instance)  # pyright: ignore[reportPrivateUsage]


def test_get_placement_previews_excludes_validator_nodes_from_results() -> None:
    api = object.__new__(API)
    topology = Topology()
    topology.add_node(NodeId("node-local"))
    topology.add_node(NodeId("node-validator"))
    api.port = 52415
    api.state = SimpleNamespace(
        topology=topology,
        node_memory={
            NodeId("node-local"): MemoryUsage(
                ram_total=Memory.from_bytes(16 * 1024**3),
                ram_available=Memory.from_bytes(8 * 1024**3),
                swap_total=Memory(),
                swap_available=Memory(),
            ),
            NodeId("node-validator"): MemoryUsage(
                ram_total=Memory.from_bytes(16 * 1024**3),
                ram_available=Memory.from_bytes(8 * 1024**3),
                swap_total=Memory(),
                swap_available=Memory(),
            ),
        },
        node_network={
            NodeId("node-local"): NodeNetworkInfo(),
            NodeId("node-validator"): NodeNetworkInfo(),
        },
        node_identities={
            NodeId("node-local"): NodeIdentity(
                api_host="127.0.0.1",
                api_port=52415,
                data_host="127.0.0.1",
                data_port=52436,
            ),
            NodeId("node-validator"): NodeIdentity(
                api_host="85.137.164.250",
                api_port=52415,
                data_host="85.137.164.250",
                data_port=52436,
            ),
        },
        instances={},
        downloads={},
    )

    model_card = ModelCard(
        model_id=ModelId("Qwen/Qwen3-0.6B-GGUF"),
        storage_size=Memory.from_bytes(639_446_688),
        n_layers=28,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family="qwen",
        quantization="Q8_0",
        base_model="Qwen3 0.6B",
        context_length=40960,
        trust_remote_code=False,
        is_custom=False,
        inference_backend=InferenceBackend.LlamaCpp,
        preferred_filename="Qwen3-0.6B-Q8_0.gguf",
    )

    with (
        patch.object(
            API,
            "_resolve_worker_required_nodes",
            return_value={NodeId("node-local")},
        ),
        patch("cai.api.main.ModelCard.load", new=AsyncMock(return_value=model_card)),
    ):
        response = asyncio.run(
            api.get_placement_previews(  # pyright: ignore[reportAttributeAccessIssue]
                model_id=ModelId("Qwen/Qwen3-0.6B-GGUF"),
                node_ids=None,
                private_network_model=False,
            )
        )

    assert response.previews
    assert all(preview.instance is not None for preview in response.previews)
    for preview in response.previews:
        instance = preview.instance
        assert instance is not None
        assert set(instance.shard_assignments.node_to_runner.keys()) == {
            NodeId("node-local")
        }


def test_get_placement_previews_limits_unsupported_gguf_to_single_node() -> None:
    api = object.__new__(API)
    topology = Topology()
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_socket_connection(2))
    )
    api.port = 52415
    api.state = SimpleNamespace(
        topology=topology,
        node_memory={
            node_a: MemoryUsage(
                ram_total=Memory.from_bytes(16 * 1024**3),
                ram_available=Memory.from_bytes(8 * 1024**3),
                swap_total=Memory(),
                swap_available=Memory(),
            ),
            node_b: MemoryUsage(
                ram_total=Memory.from_bytes(16 * 1024**3),
                ram_available=Memory.from_bytes(8 * 1024**3),
                swap_total=Memory(),
                swap_available=Memory(),
            ),
        },
        node_network={node_a: NodeNetworkInfo(), node_b: NodeNetworkInfo()},
        node_identities={
            node_a: NodeIdentity(worker_enabled=True),
            node_b: NodeIdentity(worker_enabled=True),
        },
        instances={},
        downloads={},
        overlay_peers={},
    )
    model_card = ModelCard(
        model_id=ModelId("Qwen/Qwen3.5-2B-GGUF"),
        storage_size=Memory.from_bytes(491_400_032),
        n_layers=24,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family="qwen",
        quantization="Q4_K_M",
        base_model="Qwen3.5 2B",
        context_length=8192,
        trust_remote_code=False,
        is_custom=False,
        inference_backend=InferenceBackend.LlamaCpp,
        preferred_filename="qwen3.5-2b-q4_k_m.gguf",
    )

    with (
        patch.object(API, "_resolve_worker_required_nodes", return_value={node_a, node_b}),
        patch("cai.api.main.ModelCard.load", new=AsyncMock(return_value=model_card)),
    ):
        response = asyncio.run(
                api.get_placement_previews(  # pyright: ignore[reportAttributeAccessIssue]
                    model_id=ModelId("Qwen/Qwen3.5-2B-GGUF"),
                    node_ids=None,
                    private_network_model=False,
                )
        )

    assert response.previews
    assert all(preview.instance is not None for preview in response.previews)
    assert {
        len(preview.instance.shard_assignments.node_to_runner)
        for preview in response.previews
        if preview.instance is not None
    } == {1}


def test_create_instance_prefers_embedded_model_card_over_reload() -> None:
    api = object.__new__(API)
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    runner_a = RunnerId("runner-a")
    runner_b = RunnerId("runner-b")
    model_card = ModelCard(
        model_id=ModelId("Qwen/Qwen2.5-0.5B-Instruct-GGUF"),
        storage_size=Memory.from_bytes(491_400_032),
        n_layers=24,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family="qwen",
        quantization="Q4_K_M",
        base_model="Qwen2.5 0.5B Instruct",
        context_length=8192,
        trust_remote_code=False,
        is_custom=True,
        inference_backend=InferenceBackend.LlamaCpp,
        preferred_filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        gguf_architecture="qwen2",
        shard_compatibility="layer_range_supported",
        layer_range_supported=True,
    )
    instance = MlxRingInstance(
        instance_id=InstanceId("instance-qwen25"),
        shard_assignments=ShardAssignments(
            model_id=model_card.model_id,
            runner_to_shard={
                runner_a: PipelineShardMetadata(
                    model_card=model_card,
                    device_rank=0,
                    world_size=2,
                    start_layer=0,
                    end_layer=22,
                    n_layers=24,
                ),
                runner_b: PipelineShardMetadata(
                    model_card=model_card,
                    device_rank=1,
                    world_size=2,
                    start_layer=22,
                    end_layer=24,
                    n_layers=24,
                ),
            },
            node_to_runner={node_a: runner_a, node_b: runner_b},
        ),
        hosts_by_node={
            node_a: [Host(ip="127.0.0.1", port=52445)],
            node_b: [Host(ip="85.137.164.250", port=52435)],
        },
        ephemeral_port=53115,
    )
    api.state = SimpleNamespace(
        node_memory={
            node_a: MemoryUsage(
                ram_total=Memory.from_bytes(16 * 1024**3),
                ram_available=Memory.from_bytes(8 * 1024**3),
                swap_total=Memory(),
                swap_available=Memory(),
            ),
            node_b: MemoryUsage(
                ram_total=Memory.from_bytes(4 * 1024**3),
                ram_available=Memory.from_bytes(2 * 1024**3),
                swap_total=Memory(),
                swap_available=Memory(),
            ),
        },
        node_identities={
            node_a: NodeIdentity(worker_enabled=True),
            node_b: NodeIdentity(worker_enabled=True),
        },
    )
    sent: list[CreateInstance] = []

    async def send(command: CreateInstance) -> None:
        sent.append(command)

    api._send = send  # pyright: ignore[reportAttributeAccessIssue]

    with (
        patch.object(API, "_resolve_execution_node_scope", return_value={node_a, node_b}),
        patch(
            "cai.api.main.ModelCard.load",
            new=AsyncMock(side_effect=AssertionError("model card should be embedded")),
        ),
    ):
        response = asyncio.run(
            api.create_instance(CreateInstanceParams(instance=instance))  # pyright: ignore[reportAttributeAccessIssue]
        )

    assert response.model_card.model_id == model_card.model_id
    assert sent
    assert sent[0].instance.instance_id == instance.instance_id


def test_send_times_out_while_api_is_paused() -> None:
    api = object.__new__(API)
    api.paused = True
    api.paused_ev = asyncio.Event()
    api._system_id = SystemId("system-test")
    api.command_sender = SimpleNamespace(
        send=AsyncMock(side_effect=AssertionError("send should wait for unpause"))
    )

    with (
        patch.dict("os.environ", {"CAI_API_COMMAND_SEND_TIMEOUT_SECONDS": "0.01"}),
        pytest.raises(HTTPException) as excinfo,
    ):
        asyncio.run(api._send(TestCommand()))  # pyright: ignore[reportPrivateUsage]

    assert excinfo.value.status_code == 503
    assert "paused for master election" in str(excinfo.value.detail)
    api.command_sender.send.assert_not_called()


def test_send_times_out_when_command_sender_blocks() -> None:
    class BlockingSender:
        def __init__(self) -> None:
            self.calls = 0

        async def send(self, _value: object) -> None:
            self.calls += 1
            await asyncio.Event().wait()

    api = object.__new__(API)
    api.paused = False
    api.paused_ev = asyncio.Event()
    api._system_id = SystemId("system-test")
    api.command_sender = BlockingSender()

    with (
        patch.dict("os.environ", {"CAI_API_COMMAND_SEND_TIMEOUT_SECONDS": "0.01"}),
        pytest.raises(HTTPException) as excinfo,
    ):
        asyncio.run(api._send(TestCommand()))  # pyright: ignore[reportPrivateUsage]

    assert excinfo.value.status_code == 503
    assert "Timed out dispatching command" in str(excinfo.value.detail)
    assert api.command_sender.calls == 1


def test_llama_cpp_layer_range_support_requires_consistent_policy_fields() -> None:
    assert llama_cpp_layer_range_supported(
        SimpleNamespace(
            inference_backend=InferenceBackend.LlamaCpp,
            layer_range_supported=False,
            shard_compatibility="layer_range_supported",
        )
    ) is False
    assert llama_cpp_layer_range_supported(
        SimpleNamespace(
            inference_backend=InferenceBackend.LlamaCpp,
            layer_range_supported=True,
            shard_compatibility="unsupported_for_sharding",
        )
    ) is False
    assert llama_cpp_layer_range_supported(
        SimpleNamespace(
            inference_backend=InferenceBackend.LlamaCpp,
            layer_range_supported=True,
            shard_compatibility="layer_range_supported",
        )
    ) is True


def test_get_placement_rejects_multi_node_unsupported_gguf() -> None:
    api = object.__new__(API)
    model_card = ModelCard(
        model_id=ModelId("Qwen/Qwen3.5-2B-GGUF"),
        storage_size=Memory.from_bytes(491_400_032),
        n_layers=24,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family="qwen",
        inference_backend=InferenceBackend.LlamaCpp,
        preferred_filename="qwen3.5-2b-q4_k_m.gguf",
    )

    with patch("cai.api.main.ModelCard.load", new=AsyncMock(return_value=model_card)):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                api.get_placement(  # pyright: ignore[reportAttributeAccessIssue]
                    model_id=ModelId("Qwen/Qwen3.5-2B-GGUF"),
                    sharding=Sharding.Pipeline,
                    instance_meta=InstanceMeta.LlamaCpp,
                    min_nodes=2,
                    private_network_model=False,
                )
            )

    assert exc_info.value.status_code == 400
    assert "unsupported_for_sharding" in str(exc_info.value.detail)


def test_build_execution_view_keeps_non_worker_relay_route_context() -> None:
    api = object.__new__(API)
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    node_relay = NodeId("node-relay")
    topology = Topology()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_node(node_relay)
    api.state = State(
        topology=topology,
        overlay_peers={
            node_a: [node_relay],
            node_relay: [node_b],
        },
        node_memory={
            node_a: MemoryUsage(
                ram_total=Memory.from_bytes(16 * 1024**3),
                ram_available=Memory.from_bytes(8 * 1024**3),
                swap_total=Memory(),
                swap_available=Memory(),
            ),
            node_b: MemoryUsage(
                ram_total=Memory.from_bytes(16 * 1024**3),
                ram_available=Memory.from_bytes(8 * 1024**3),
                swap_total=Memory(),
                swap_available=Memory(),
            ),
            node_relay: MemoryUsage(
                ram_total=Memory.from_bytes(4 * 1024**3),
                ram_available=Memory.from_bytes(2 * 1024**3),
                swap_total=Memory(),
                swap_available=Memory(),
            ),
        },
        node_network={
            node_a: NodeNetworkInfo(),
            node_b: NodeNetworkInfo(),
            node_relay: NodeNetworkInfo(),
        },
        node_identities={
            node_a: NodeIdentity(worker_enabled=True),
            node_b: NodeIdentity(worker_enabled=True),
            node_relay: NodeIdentity(
                relay_enabled=True,
                transport_endpoints=[
                    AdvertisedTransportEndpoint(
                        purpose="api",
                        route_type="overlay",
                        host="203.0.113.10",
                        port=52415,
                    )
                ],
            ),
        },
        instances={},
        downloads={},
    )

    (
        execution_topology,
        execution_memory,
        execution_network,
        execution_identities,
        execution_overlay_peers,
        _execution_instances,
        _execution_downloads,
    ) = api._build_execution_view({node_a, node_b})  # pyright: ignore[reportPrivateUsage]

    assert set(execution_memory) == {node_a, node_b}
    assert node_relay in execution_network
    assert execution_identities is not None
    assert node_relay in execution_identities
    assert node_relay in set(execution_topology.list_nodes())
    assert execution_overlay_peers[node_a] == [node_relay]
    assert execution_overlay_peers[node_relay] == [node_b]


def test_build_execution_view_uses_verified_capability_record_missing_from_state() -> None:
    api = object.__new__(API)
    node_local = NodeId("node-local")
    node_worker = NodeId("node-worker")
    node_relay = NodeId("node-relay")
    topology = Topology()
    topology.add_node(node_local)
    topology.add_node(node_relay)
    api.state = State(
        topology=topology,
        overlay_peers={
            node_local: [node_relay],
            node_relay: [node_local],
        },
        node_memory={
            node_local: MemoryUsage(
                ram_total=Memory.from_bytes(16 * 1024**3),
                ram_available=Memory.from_bytes(8 * 1024**3),
                swap_total=Memory(),
                swap_available=Memory(),
            ),
        },
        node_network={
            node_local: NodeNetworkInfo(),
            node_relay: NodeNetworkInfo(),
        },
        node_identities={
            node_local: NodeIdentity(worker_enabled=True),
            node_relay: NodeIdentity(
                relay_enabled=True,
                transport_endpoints=[
                    AdvertisedTransportEndpoint(
                        purpose="api",
                        route_type="overlay",
                        host="203.0.113.10",
                        port=52415,
                    )
                ],
            ),
        },
        instances={},
        downloads={},
    )
    record = NodeCapabilityRecord(
        node_id=str(node_worker),
        source="peer",
        source_url="http://198.51.100.20:52415/v1/cai/node-capabilities",
        last_seen_at="2999-01-01T00:00:00+00:00",
        updated_at="2999-01-01T00:00:00+00:00",
        friendly_name="Worker",
        api_urls=["http://198.51.100.20:52415"],
        data_endpoints=[
            {
                "purpose": "data",
                "routeType": "overlay",
                "host": "198.51.100.20",
            }
        ],
        worker_enabled=True,
        worker_reward_address="worker-address",
        resource_summary={
            "ramBytes": 12 * 1024**3,
            "ramAvailableBytes": 6 * 1024**3,
        },
        route_hints={"overlayPeerIds": [str(node_relay)]},
        worker_verified=True,
    )

    with patch.object(
        API,
        "_load_verified_worker_capability_records",
        return_value={node_worker: record},
    ):
        (
            execution_topology,
            execution_memory,
            _execution_network,
            execution_identities,
            execution_overlay_peers,
            _execution_instances,
            _execution_downloads,
        ) = api._build_execution_view(  # pyright: ignore[reportPrivateUsage]
            {node_local, node_worker}
        )

    assert set(execution_memory) == {node_local, node_worker}
    assert execution_memory[node_worker].ram_available == Memory.from_bytes(6 * 1024**3)
    assert execution_identities is not None
    assert execution_identities[node_worker].worker_enabled is True
    assert node_worker in set(execution_topology.list_nodes())
    assert execution_overlay_peers[node_worker] == [node_relay]
    assert node_worker in execution_overlay_peers[node_relay]


def test_preplacement_llama_cpp_rpc_probe_runs_for_multi_node_model() -> None:
    api = object.__new__(API)
    api.port = 52415
    api.node_id = NodeId("node-local")
    api.state = {"nodeIdentities": {}}
    wallet_policy = object()
    calls: list[dict[str, object]] = []
    model_card = ModelCard(
        model_id=ModelId("Qwen/Qwen3-0.6B-GGUF"),
        storage_size=Memory.from_bytes(123),
        n_layers=28,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )
    service = SimpleNamespace(
        wallet_policy=wallet_policy,
        modules=SimpleNamespace(
            route_health=SimpleNamespace(
                probe_llama_cpp_rpc_routes=lambda **kwargs: calls.append(kwargs)
            )
        ),
    )

    with patch("cai.api.main.make_cai_service", return_value=service):
        api._probe_llama_cpp_rpc_route_health_before_placement(  # pyright: ignore[reportPrivateUsage]
            model_card=model_card,
            min_nodes=2,
        )

    assert calls == [
        {
            "state_payload": {"nodeIdentities": {}},
            "local_node_id": "node-local",
            "timeout_sec": 0.75,
            "policy": wallet_policy,
        }
    ]


def test_relay_target_allows_active_runtime_port_but_not_sink_api_port() -> None:
    api = object.__new__(API)
    sink = NodeId("node-sink")
    runner_id = RunnerId("runner-sink")
    model_card = ModelCard(
        model_id=ModelId("Qwen/Qwen3-0.6B-GGUF"),
        storage_size=Memory.from_bytes(123),
        n_layers=28,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )
    instance = MlxRingInstance(
        instance_id=InstanceId("instance-runtime"),
        shard_assignments=ShardAssignments(
            model_id=model_card.model_id,
            runner_to_shard={
                runner_id: PipelineShardMetadata(
                    model_card=model_card,
                    device_rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=28,
                    n_layers=28,
                ),
            },
            node_to_runner={sink: runner_id},
        ),
        hosts_by_node={sink: [Host(ip="0.0.0.0", port=59657)]},
        ephemeral_port=59657,
    )
    api.state = State(
        node_identities={
            sink: NodeIdentity(
                api_host="26.97.29.153",
                api_port=52425,
            )
        },
        node_network={
            sink: NodeNetworkInfo(
                interfaces=[
                    NetworkInterfaceInfo(
                        name="overlay0",
                        ip_address="26.97.29.153",
                    )
                ]
            )
        },
        instances={instance.instance_id: instance},
    )

    assert api._relay_target_allowed(  # pyright: ignore[reportPrivateUsage]
        str(sink),
        "26.97.29.153",
        59657,
    )
    assert not api._relay_target_allowed(  # pyright: ignore[reportPrivateUsage]
        str(sink),
        "26.97.29.153",
        52425,
    )
    assert not api._relay_target_allowed(  # pyright: ignore[reportPrivateUsage]
        str(sink),
        "198.51.100.99",
        59657,
    )


def test_resolve_worker_required_nodes_keeps_real_local_node_when_only_remote_identity_exists() -> None:
    api = object.__new__(API)
    api.port = 52415
    api.node_id = NodeId("node-local")
    api.state = SimpleNamespace(
        node_identities={
            NodeId("node-validator"): {"apiPort": 52415, "apiHost": "85.137.164.250"},
        }
    )

    with (
        patch.object(API, "_load_local_worker_enabled", return_value=True),
        patch.object(
            API,
            "_load_local_worker_reward_address",
            return_value="ABCD1234ABCD1234ABCD1234ABCD1234",
        ),
        patch("cai.api.main._load_json_url", return_value={"worker": {"worker_enabled": False}}),
    ):
        required_nodes = api._resolve_worker_required_nodes()  # pyright: ignore[reportPrivateUsage]

    assert required_nodes == {NodeId("node-local")}


def test_resolve_worker_required_nodes_uses_local_config_without_self_http() -> None:
    api = object.__new__(API)
    api.port = 52415
    api.node_id = NodeId("node-local")
    api.state = SimpleNamespace(
        node_identities={
            NodeId("node-local"): {"apiPort": 52415, "apiHost": "127.0.0.1"},
            NodeId("node-validator"): {"apiPort": 52415, "apiHost": "85.137.164.250"},
        }
    )

    def _fake_load_json(url: str, *, timeout: int = 5):  # noqa: ARG001
        assert not url.startswith("http://127.0.0.1:52415/")
        return {"worker": {"worker_enabled": False}}

    with (
        patch.object(API, "_load_local_worker_enabled", return_value=True),
        patch.object(
            API,
            "_load_local_worker_reward_address",
            return_value="ABCD1234ABCD1234ABCD1234ABCD1234",
        ),
        patch("cai.api.main._load_json_url", side_effect=_fake_load_json),
    ):
        required_nodes = api._resolve_worker_required_nodes()  # pyright: ignore[reportPrivateUsage]

    assert required_nodes == {NodeId("node-local")}


def test_resolve_worker_required_nodes_rejects_local_worker_without_reward_address() -> None:
    api = object.__new__(API)
    api.port = 52415
    api.node_id = NodeId("node-local")
    api.state = SimpleNamespace(
        node_identities={
            NodeId("node-local"): {"apiPort": 52415, "apiHost": "127.0.0.1"},
        }
    )

    with (
        patch.object(API, "_load_local_worker_enabled", return_value=True),
        patch.object(API, "_load_local_worker_reward_address", return_value=None),
        patch("cai.api.main._load_json_url", side_effect=AssertionError("HTTP fallback should not run")),
    ):
        required_nodes = api._resolve_worker_required_nodes()  # pyright: ignore[reportPrivateUsage]

    assert required_nodes == set()


def test_resolve_worker_required_nodes_uses_state_worker_metadata_for_remote_nodes() -> None:
    api = object.__new__(API)
    api.port = 52415
    api.node_id = NodeId("node-local")
    api.state = SimpleNamespace(
        node_identities={
            NodeId("node-local"): {
                "apiPort": 52415,
                "apiHost": "127.0.0.1",
                "workerEnabled": False,
            },
            NodeId("node-remote"): {
                "apiPort": 52415,
                "apiHost": "198.51.100.20",
                "workerEnabled": True,
                "workerRewardAddress": "ABCD1234ABCD1234ABCD1234ABCD1234",
            },
        }
    )

    with (
        patch.object(API, "_load_local_worker_enabled", return_value=False),
        patch("cai.api.main._load_json_url", side_effect=AssertionError("HTTP fallback should not run")),
    ):
        required_nodes = api._resolve_worker_required_nodes()  # pyright: ignore[reportPrivateUsage]

    assert required_nodes == {NodeId("node-remote")}


def test_resolve_worker_required_nodes_ignores_stale_remote_worker_nodes() -> None:
    api = object.__new__(API)
    api.port = 52415
    api.node_id = NodeId("node-local")
    api.state = SimpleNamespace(
        node_identities={
            NodeId("node-local"): {
                "apiPort": 52415,
                "apiHost": "127.0.0.1",
                "workerEnabled": False,
            },
            NodeId("node-stale"): {
                "apiPort": 52415,
                "apiHost": "198.51.100.20",
                "workerEnabled": True,
                "workerRewardAddress": "ABCD1234ABCD1234ABCD1234ABCD1234",
            },
        },
        last_seen={
            NodeId("node-stale"): datetime.now(tz=timezone.utc) - timedelta(hours=2),
        },
    )

    with (
        patch.object(API, "_load_local_worker_enabled", return_value=False),
        patch("cai.api.main._load_json_url", side_effect=AssertionError("stale worker should be ignored before HTTP fallback")),
    ):
        required_nodes = api._resolve_worker_required_nodes()  # pyright: ignore[reportPrivateUsage]

    assert required_nodes == set()


def test_resolve_worker_required_nodes_requires_verified_remote_capability_in_strict_mode() -> None:
    api = object.__new__(API)
    api.port = 52415
    api.node_id = NodeId("node-local")
    api.state = SimpleNamespace(
        node_identities={
            NodeId("node-local"): {
                "apiPort": 52415,
                "apiHost": "127.0.0.1",
                "workerEnabled": False,
            },
            NodeId("node-fake"): {
                "apiPort": 52415,
                "apiHost": "198.51.100.44",
                "workerEnabled": True,
                "workerRewardAddress": "ABCD1234ABCD1234ABCD1234ABCD1234",
            },
        }
    )

    with (
        patch.dict("os.environ", {"CAI_REQUIRE_SIGNED_PEER_PAYLOADS": "1"}),
        patch.object(API, "_load_local_worker_enabled", return_value=False),
        patch.object(API, "_load_verified_worker_capability_node_ids", return_value=set()),
        patch("cai.api.main._load_json_url", side_effect=AssertionError("unverified HTTP fallback should not run")),
    ):
        required_nodes = api._resolve_worker_required_nodes()  # pyright: ignore[reportPrivateUsage]

    assert required_nodes == set()


def test_resolve_worker_required_nodes_allows_verified_remote_capability_in_strict_mode() -> None:
    api = object.__new__(API)
    api.port = 52415
    api.node_id = NodeId("node-local")
    api.state = SimpleNamespace(
        node_identities={
            NodeId("node-local"): {
                "apiPort": 52415,
                "apiHost": "127.0.0.1",
                "workerEnabled": False,
            },
            NodeId("node-worker"): {
                "apiPort": 52415,
                "apiHost": "198.51.100.45",
                "workerEnabled": True,
                "workerRewardAddress": "ABCD1234ABCD1234ABCD1234ABCD1234",
            },
        }
    )

    with (
        patch.dict("os.environ", {"CAI_REQUIRE_SIGNED_PEER_PAYLOADS": "1"}),
        patch.object(API, "_load_local_worker_enabled", return_value=False),
        patch.object(
            API,
            "_load_verified_worker_capability_node_ids",
            return_value={"node-worker"},
        ),
        patch("cai.api.main._load_json_url", side_effect=AssertionError("verified capability should avoid HTTP fallback")),
    ):
        required_nodes = api._resolve_worker_required_nodes()  # pyright: ignore[reportPrivateUsage]

    assert required_nodes == {NodeId("node-worker")}


def test_resolve_worker_required_nodes_ignores_verified_nodes_missing_from_live_state() -> None:
    api = object.__new__(API)
    api.port = 52415
    api.node_id = NodeId("node-local")
    api.state = SimpleNamespace(
        node_identities={
            NodeId("node-local"): {
                "apiPort": 52415,
                "apiHost": "127.0.0.1",
                "workerEnabled": True,
                "workerRewardAddress": "ABCD1234ABCD1234ABCD1234ABCD1234",
            },
            NodeId("node-validator"): {
                "apiPort": 52415,
                "apiHost": "198.51.100.50",
                "workerEnabled": False,
            },
        }
    )

    with (
        patch.object(API, "_load_local_worker_enabled", return_value=True),
        patch.object(
            API,
            "_load_local_worker_reward_address",
            return_value="ABCD1234ABCD1234ABCD1234ABCD1234",
        ),
        patch.object(
            API,
            "_load_verified_worker_capability_node_ids",
            return_value={"node-stale"},
        ),
        patch("cai.api.main._load_json_url", return_value={"worker": {"worker_enabled": False}}),
    ):
        required_nodes = api._resolve_worker_required_nodes()  # pyright: ignore[reportPrivateUsage]

    assert required_nodes == {NodeId("node-local")}


def test_resolve_execution_cai_url_stays_local_after_demote() -> None:
    api = object.__new__(API)
    api.port = 52425
    api.node_id = NodeId("node-local")
    api.current_master_node_id = NodeId(
        "12D3KooWPi8Xi74nmpEE8yFSByr7KHAzuL9dD8G8akexuM2CnCyC"
    )
    api.state = SimpleNamespace(
        node_identities={
            NodeId("node-local"): {"apiPort": 52425, "apiHost": None},
        }
    )

    with patch.dict("os.environ", {}, clear=True):
        execution_url = api._resolve_execution_cai_url()  # pyright: ignore[reportPrivateUsage]

    assert execution_url == "http://127.0.0.1:52425"


def test_resolve_execution_cai_url_allows_explicit_override() -> None:
    api = object.__new__(API)
    api.port = 52425

    with patch.dict(
        "os.environ",
        {"CAI_EXECUTION_CAI_URL": "http://85.137.164.250:52415/"},
        clear=True,
    ):
        execution_url = api._resolve_execution_cai_url()  # pyright: ignore[reportPrivateUsage]

    assert execution_url == "http://85.137.164.250:52415"


def test_get_dashboard_state_counts_remote_worker_resources_from_state_metadata() -> None:
    api = object.__new__(API)
    api.dashboard_disabled = False
    api.port = 52415
    api.node_id = NodeId("node-local")
    topology = Topology()
    topology.add_node(NodeId("node-local"))
    topology.add_node(NodeId("node-remote"))
    api.state = State(
        topology=topology,
        overlay_peers={NodeId("node-local"): [NodeId("node-remote")]},
        node_memory={
            NodeId("node-local"): MemoryUsage(
                ram_total=Memory.from_bytes(16 * 1024**3),
                ram_available=Memory.from_bytes(8 * 1024**3),
                swap_total=Memory(),
                swap_available=Memory(),
            ),
            NodeId("node-remote"): MemoryUsage(
                ram_total=Memory.from_bytes(48 * 1024**3),
                ram_available=Memory.from_bytes(36 * 1024**3),
                swap_total=Memory(),
                swap_available=Memory(),
            ),
        },
        node_identities={
            NodeId("node-local"): NodeIdentity(
                api_host="127.0.0.1",
                api_port=52415,
                cpu_physical_cores=8,
                total_vram_bytes=0,
                worker_enabled=False,
            ),
            NodeId("node-remote"): NodeIdentity(
                api_host="198.51.100.20",
                api_port=52415,
                cpu_physical_cores=24,
                total_vram_bytes=16 * 1024**3,
                worker_enabled=True,
                worker_reward_address="ABCD1234ABCD1234ABCD1234ABCD1234",
            ),
        },
    )

    with (
        patch.object(API, "_load_local_worker_enabled", return_value=False),
        patch("cai.api.main._load_json_url", side_effect=AssertionError("HTTP fallback should not run")),
    ):
        payload = api.get_dashboard_state()  # pyright: ignore[reportAttributeAccessIssue]

    assert payload["networkSummary"]["knownWorkers"] == 1
    assert payload["networkSummary"]["workerTotalRamBytes"] == 48 * 1024**3
    assert payload["networkSummary"]["workerTotalAvailableRamBytes"] == 36 * 1024**3
    assert payload["networkSummary"]["workerTotalVramBytes"] == 16 * 1024**3
    assert payload["networkSummary"]["workerTotalCpuCores"] == 24


def test_get_dashboard_state_counts_verified_capability_worker_missing_from_state() -> None:
    api = object.__new__(API)
    api.dashboard_disabled = False
    api.port = 52415
    api.node_id = NodeId("node-local")
    topology = Topology()
    topology.add_node(NodeId("node-local"))
    api.state = State(
        topology=topology,
        node_memory={
            NodeId("node-local"): MemoryUsage(
                ram_total=Memory.from_bytes(16 * 1024**3),
                ram_available=Memory.from_bytes(8 * 1024**3),
                swap_total=Memory(),
                swap_available=Memory(),
            ),
        },
        node_identities={
            NodeId("node-local"): NodeIdentity(
                api_host="127.0.0.1",
                api_port=52415,
                cpu_physical_cores=8,
                total_vram_bytes=0,
                worker_enabled=False,
            ),
        },
    )
    record = NodeCapabilityRecord(
        node_id="node-remote",
        source="peer",
        source_url="http://198.51.100.20:52415/v1/cai/node-capabilities",
        last_seen_at="2999-01-01T00:00:00+00:00",
        updated_at="2999-01-01T00:00:00+00:00",
        friendly_name="Remote worker",
        api_urls=["http://198.51.100.20:52415"],
        worker_enabled=True,
        worker_reward_address="ABCD1234ABCD1234ABCD1234ABCD1234",
        worker_verified=True,
        resource_summary={
            "ramBytes": 48 * 1024**3,
            "ramAvailableBytes": 36 * 1024**3,
            "vramBytes": 16 * 1024**3,
            "cpuCores": 24,
        },
    )

    with (
        patch.object(
            API,
            "_resolve_worker_required_nodes",
            return_value={NodeId("node-remote")},
        ),
        patch.object(
            API,
            "_load_verified_worker_capability_records",
            return_value={NodeId("node-remote"): record},
        ),
    ):
        payload = api.get_dashboard_state()  # pyright: ignore[reportAttributeAccessIssue]

    assert set(payload["topology"]["nodes"]) == {"node-local", "node-remote"}
    assert payload["networkSummary"]["knownNodes"] == 2
    assert payload["networkSummary"]["knownWorkers"] == 1
    assert payload["networkSummary"]["workerTotalRamBytes"] == 48 * 1024**3
    assert payload["networkSummary"]["workerTotalAvailableRamBytes"] == 36 * 1024**3
    assert payload["networkSummary"]["workerTotalVramBytes"] == 16 * 1024**3
    assert payload["networkSummary"]["workerTotalCpuCores"] == 24


def test_get_dashboard_state_reports_worker_enabled_resources_only() -> None:
    api = object.__new__(API)
    api.dashboard_disabled = False
    api.node_id = NodeId("node-worker")
    topology = Topology()
    topology.add_node(NodeId("node-worker"))
    topology.add_node(NodeId("node-validator"))
    api.state = State(
        topology=topology,
        overlay_peers={NodeId("node-worker"): [NodeId("node-validator")]},
        node_memory={
            NodeId("node-worker"): MemoryUsage(
                ram_total=Memory.from_bytes(32 * 1024**3),
                ram_available=Memory.from_bytes(20 * 1024**3),
                swap_total=Memory(),
                swap_available=Memory(),
            ),
            NodeId("node-validator"): MemoryUsage(
                ram_total=Memory.from_bytes(96 * 1024**3),
                ram_available=Memory.from_bytes(72 * 1024**3),
                swap_total=Memory(),
                swap_available=Memory(),
            ),
        },
        node_identities={
            NodeId("node-worker"): NodeIdentity(
                cpu_physical_cores=16,
                total_vram_bytes=24 * 1024**3,
            ),
            NodeId("node-validator"): NodeIdentity(
                cpu_physical_cores=64,
                total_vram_bytes=0,
            ),
        },
    )

    with patch.object(
        API,
        "_resolve_worker_required_nodes",
        return_value={NodeId("node-worker")},
    ):
        payload = api.get_dashboard_state()  # pyright: ignore[reportAttributeAccessIssue]

    assert payload["networkSummary"]["knownNodes"] == 2
    assert payload["networkSummary"]["knownWorkers"] == 1
    assert payload["networkSummary"]["workerTotalRamBytes"] == 32 * 1024**3
    assert payload["networkSummary"]["workerTotalAvailableRamBytes"] == 20 * 1024**3
    assert payload["networkSummary"]["workerTotalVramBytes"] == 24 * 1024**3
    assert payload["networkSummary"]["workerTotalCpuCores"] == 16


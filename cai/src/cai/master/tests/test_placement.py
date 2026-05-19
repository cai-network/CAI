# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

from collections import Counter
from datetime import UTC, datetime

import pytest

from cai.master.placement import (
    _node_cai_owned_worker_readiness,
    get_transition_events,
    place_instance,
)
from cai.master.placement_utils import (
    get_cai_api_urls_by_node,
    get_llama_cpp_direct_candidate_cycles,
    get_llama_cpp_relay_routes_by_node,
)
from cai.shared.network_model_policy import get_private_network_model_policy
from cai.shared.network_routes import relay_route_candidates
from cai.master.tests.conftest import (
    create_node_memory,
    create_node_network,
    create_rdma_connection,
    create_socket_connection,
)
from cai.shared.models.model_cards import (
    InferenceBackend,
    ModelCard,
    ModelId,
    ModelTask,
)
from cai.shared.topology import Topology
from cai.shared.types.commands import PlaceInstance
from cai.shared.types.common import CommandId, NodeId
from cai.shared.types.events import (
    InstanceCreated,
    InstanceDeleted,
    TaskStatusUpdated,
)
from cai.shared.types.memory import Memory
from cai.shared.types.multiaddr import Multiaddr
from cai.shared.types.profiling import (
    AdvertisedTransportEndpoint,
    NetworkInterfaceInfo,
    NodeIdentity,
    NodeNetworkInfo,
)
from cai.shared.types.tasks import TaskId, TaskStatus, TextGeneration
from cai.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from cai.shared.types.topology import Connection, Cycle, SocketConnection
from cai.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadProgressData,
)
from cai.shared.types.worker.instances import (
    Instance,
    InstanceId,
    InstanceMeta,
    LlamaCppRelayRoute,
    MlxJacclInstance,
    MlxRingInstance,
)
from cai.shared.types.worker.runners import ShardAssignments
from cai.shared.types.worker.shards import PipelineShardMetadata, Sharding


@pytest.fixture
def instance() -> Instance:
    return MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=ModelId("test-model"), runner_to_shard={}, node_to_runner={}
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )


@pytest.fixture
def model_card() -> ModelCard:
    return ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=10,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )


def place_instance_command(model_card: ModelCard) -> PlaceInstance:
    return PlaceInstance(
        command_id=CommandId(),
        model_card=model_card,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
    )


def test_get_cai_api_urls_by_node_prefers_transport_endpoints() -> None:
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")

    api_urls = get_cai_api_urls_by_node(
        Cycle(node_ids=[node_a, node_b]),
        {
            node_a: NodeIdentity(
                api_host="198.51.100.10",
                api_port=52415,
                transport_endpoints=[
                    AdvertisedTransportEndpoint(
                        purpose="api",
                        route_type="direct",
                        host="10.0.0.10",
                        port=52415,
                    )
                ],
            ),
            node_b: NodeIdentity(api_host="0.0.0.0", api_port=52415),
        },
    )

    assert api_urls[node_a] == [
        "http://10.0.0.10:52415",
        "http://198.51.100.10:52415",
    ]
    assert api_urls[node_b] == []


def test_get_cai_api_urls_by_node_adds_overlay_urls_from_relay_routes() -> None:
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    relay = NodeId("node-relay")

    api_urls = get_cai_api_urls_by_node(
        Cycle(node_ids=[node_a, node_b]),
        {
            node_a: NodeIdentity(worker_enabled=True),
            node_b: NodeIdentity(worker_enabled=True),
            relay: NodeIdentity(relay_enabled=True),
        },
        relay_routes_by_node={
            node_a: [
                LlamaCppRelayRoute(
                    source_node_id=node_a,
                    transit_node_id=relay,
                    sink_node_id=node_b,
                    relay_api_host="85.137.164.250",
                    relay_api_port=52415,
                    target_host="203.0.113.42",
                    target_port=52435,
                    source_segment_type="overlay",
                    sink_segment_type="overlay",
                )
            ]
        },
    )

    assert api_urls[node_a] == []
    assert api_urls[node_b] == [
        "cai-overlay:http://85.137.164.250:52415?"
        "targetNodeId=node-b&relayRole=bootstrap&transitNodeId=node-relay"
    ]


def test_get_cai_api_urls_by_node_marks_worker_relay_as_ordinary() -> None:
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    relay = NodeId("node-worker-relay")

    api_urls = get_cai_api_urls_by_node(
        Cycle(node_ids=[node_a, node_b]),
        {
            node_a: NodeIdentity(worker_enabled=True),
            node_b: NodeIdentity(worker_enabled=True),
            relay: NodeIdentity(worker_enabled=True, relay_enabled=True),
        },
        relay_routes_by_node={
            node_a: [
                LlamaCppRelayRoute(
                    source_node_id=node_a,
                    transit_node_id=relay,
                    sink_node_id=node_b,
                    relay_api_host="203.0.113.10",
                    relay_api_port=52415,
                    target_host="203.0.113.42",
                    target_port=52435,
                    source_segment_type="overlay",
                    sink_segment_type="overlay",
                )
            ]
        },
    )

    assert api_urls[node_b] == [
        "cai-overlay:http://203.0.113.10:52415?"
        "targetNodeId=node-b&relayRole=ordinary&transitNodeId=node-worker-relay"
    ]


@pytest.mark.parametrize(
    "available_memory,total_layers,expected_layers",
    [
        ((500, 500, 1000), 12, (3, 3, 6)),
        ((500, 500, 500), 12, (4, 4, 4)),
        ((312, 468, 1092), 12, (2, 3, 7)),
    ],
)
def test_get_instance_placements_create_instance(
    available_memory: tuple[int, int, int],
    total_layers: int,
    expected_layers: tuple[int, int, int],
    model_card: ModelCard,
):
    # arrange
    model_card.n_layers = total_layers
    model_card.storage_size = Memory.from_bytes(
        sum(available_memory)
    )  # make it exactly fit across all nodes
    topology = Topology()

    cic = place_instance_command(model_card)
    node_id_a = NodeId()
    node_id_b = NodeId()
    node_id_c = NodeId()

    # fully connected (directed) between the 3 nodes
    conn_a_b = Connection(
        source=node_id_a, sink=node_id_b, edge=create_socket_connection(1)
    )
    conn_b_c = Connection(
        source=node_id_b, sink=node_id_c, edge=create_socket_connection(2)
    )
    conn_c_a = Connection(
        source=node_id_c, sink=node_id_a, edge=create_socket_connection(3)
    )
    conn_c_b = Connection(
        source=node_id_c, sink=node_id_b, edge=create_socket_connection(4)
    )
    conn_a_c = Connection(
        source=node_id_a, sink=node_id_c, edge=create_socket_connection(5)
    )
    conn_b_a = Connection(
        source=node_id_b, sink=node_id_a, edge=create_socket_connection(6)
    )

    node_memory = {
        node_id_a: create_node_memory(available_memory[0]),
        node_id_b: create_node_memory(available_memory[1]),
        node_id_c: create_node_memory(available_memory[2]),
    }
    node_network = {
        node_id_a: create_node_network(),
        node_id_b: create_node_network(),
        node_id_c: create_node_network(),
    }
    topology.add_node(node_id_a)
    topology.add_node(node_id_b)
    topology.add_node(node_id_c)
    topology.add_connection(conn_a_b)
    topology.add_connection(conn_b_c)
    topology.add_connection(conn_c_a)
    topology.add_connection(conn_c_b)
    topology.add_connection(conn_a_c)
    topology.add_connection(conn_b_a)

    # act
    placements = place_instance(cic, topology, {}, node_memory, node_network)

    # assert
    assert len(placements) == 1
    instance_id = list(placements.keys())[0]
    instance = placements[instance_id]
    assert instance.shard_assignments.model_id == model_card.model_id

    runner_id_a = instance.shard_assignments.node_to_runner[node_id_a]
    runner_id_b = instance.shard_assignments.node_to_runner[node_id_b]
    runner_id_c = instance.shard_assignments.node_to_runner[node_id_c]

    shard_a = instance.shard_assignments.runner_to_shard[runner_id_a]
    shard_b = instance.shard_assignments.runner_to_shard[runner_id_b]
    shard_c = instance.shard_assignments.runner_to_shard[runner_id_c]

    assert shard_a.end_layer - shard_a.start_layer == expected_layers[0]
    assert shard_b.end_layer - shard_b.start_layer == expected_layers[1]
    assert shard_c.end_layer - shard_c.start_layer == expected_layers[2]

    shards = [shard_a, shard_b, shard_c]
    shards_sorted = sorted(shards, key=lambda s: s.start_layer)
    assert shards_sorted[0].start_layer == 0
    assert shards_sorted[-1].end_layer == total_layers


def test_get_instance_placements_one_node_exact_fit() -> None:
    topology = Topology()
    node_id = NodeId()
    topology.add_node(node_id)
    node_memory = {node_id: create_node_memory(1000 * 1024)}
    node_network = {node_id: create_node_network()}
    cic = place_instance_command(
        ModelCard(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_kb(1000),
            n_layers=10,
            hidden_size=1000,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
        ),
    )
    placements = place_instance(cic, topology, {}, node_memory, node_network)

    assert len(placements) == 1
    instance_id = list(placements.keys())[0]
    instance = placements[instance_id]
    assert instance.shard_assignments.model_id == "test-model"
    assert len(instance.shard_assignments.node_to_runner) == 1
    assert len(instance.shard_assignments.runner_to_shard) == 1
    assert len(instance.shard_assignments.runner_to_shard) == 1


def test_get_instance_placements_one_node_fits_with_extra_memory() -> None:
    topology = Topology()
    node_id = NodeId()
    topology.add_node(node_id)
    node_memory = {node_id: create_node_memory(1001 * 1024)}
    node_network = {node_id: create_node_network()}
    cic = place_instance_command(
        ModelCard(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_kb(1000),
            n_layers=10,
            hidden_size=1000,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
        ),
    )
    placements = place_instance(cic, topology, {}, node_memory, node_network)

    assert len(placements) == 1
    instance_id = list(placements.keys())[0]
    instance = placements[instance_id]
    assert instance.shard_assignments.model_id == "test-model"
    assert len(instance.shard_assignments.node_to_runner) == 1
    assert len(instance.shard_assignments.runner_to_shard) == 1
    assert len(instance.shard_assignments.runner_to_shard) == 1


def test_llama_cpp_backend_requires_llama_cpp_instance_meta() -> None:
    topology = Topology()
    node_id = NodeId()
    topology.add_node(node_id)
    node_memory = {node_id: create_node_memory(1000 * 1024)}
    node_network = {node_id: create_node_network()}
    model_card = ModelCard(
        model_id=ModelId("local/gguf-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )

    with pytest.raises(
        ValueError,
        match="llama.cpp models require the LlamaCpp instance backend",
    ):
        place_instance(
            PlaceInstance(
                command_id=CommandId(),
                model_card=model_card,
                sharding=Sharding.Pipeline,
                instance_meta=InstanceMeta.MlxRing,
                min_nodes=1,
            ),
            topology,
            {},
            node_memory,
            node_network,
        )


def test_llama_cpp_backend_places_on_multiple_nodes() -> None:
    topology = Topology()
    node_a = NodeId()
    node_b = NodeId()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_socket_connection(2))
    )
    node_memory = {
        node_a: create_node_memory(1000 * 1024),
        node_b: create_node_memory(1000 * 1024),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }
    node_identities = {
        node_a: NodeIdentity(data_host="10.0.0.10"),
        node_b: NodeIdentity(data_host="10.0.0.11"),
    }
    model_card = ModelCard(
        model_id=ModelId("local/gguf-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )
    placements = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=model_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.LlamaCpp,
            min_nodes=2,
        ),
        topology,
        {},
        node_memory,
        node_network,
        node_identities=node_identities,
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxRingInstance)
    assert len(instance.shard_assignments.node_to_runner) == 2
    runner_a = instance.shard_assignments.node_to_runner[node_a]
    runner_b = instance.shard_assignments.node_to_runner[node_b]
    shard_a = instance.shard_assignments.runner_to_shard[runner_a]
    shard_b = instance.shard_assignments.runner_to_shard[runner_b]
    hosts_a = instance.hosts_by_node[node_a]
    hosts_b = instance.hosts_by_node[node_b]

    assert hosts_a[shard_a.device_rank].ip == "0.0.0.0"
    assert hosts_b[shard_b.device_rank].ip == "0.0.0.0"
    assert any(host.ip == "10.0.0.11" for host in hosts_a if host.ip != "0.0.0.0")
    assert any(host.ip == "10.0.0.10" for host in hosts_b if host.ip != "0.0.0.0")
    assert all(host.port == instance.ephemeral_port for host in hosts_a)
    assert all(host.port == instance.ephemeral_port for host in hosts_b)


def test_llama_cpp_backend_places_with_direct_coordinator_fanout() -> None:
    topology = Topology()
    node_a = NodeId()
    node_b = NodeId()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_socket_connection(1))
    )
    node_memory = {
        node_a: create_node_memory(2000 * 1024),
        node_b: create_node_memory(1000 * 1024),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }
    node_identities = {
        node_a: NodeIdentity(),
        node_b: NodeIdentity(data_host="10.0.0.11"),
    }
    model_card = ModelCard(
        model_id=ModelId("local/gguf-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )
    placements = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=model_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.LlamaCpp,
            min_nodes=2,
        ),
        topology,
        {},
        node_memory,
        node_network,
        node_identities=node_identities,
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxRingInstance)
    hosts_a = instance.hosts_by_node[node_a]
    hosts_b = instance.hosts_by_node[node_b]
    assert any(host.ip == "10.0.0.11" for host in hosts_a if host.ip != "0.0.0.0")
    assert all(host.ip == "0.0.0.0" for host in hosts_b)


def test_llama_cpp_backend_requires_direct_coordinator_fanout_for_multi_node() -> None:
    topology = Topology()
    node_a = NodeId()
    node_b = NodeId()
    node_c = NodeId()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_node(node_c)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_c, edge=create_socket_connection(2))
    )
    node_memory = {
        node_a: create_node_memory(2000 * 1024),
        node_b: create_node_memory(1000 * 1024),
        node_c: create_node_memory(1000 * 1024),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
        node_c: create_node_network(),
    }
    node_identities = {
        node_a: NodeIdentity(),
        node_b: NodeIdentity(data_host="10.0.0.11"),
        node_c: NodeIdentity(data_host="10.0.0.12"),
    }
    model_card = ModelCard(
        model_id=ModelId("local/gguf-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )
    with pytest.raises(
        ValueError,
        match="requires at least 3 worker node\\(s\\) where one coordinator can reach every other participant directly",
    ):
        place_instance(
            PlaceInstance(
                command_id=CommandId(),
                model_card=model_card,
                sharding=Sharding.Pipeline,
                instance_meta=InstanceMeta.LlamaCpp,
                min_nodes=3,
            ),
            topology,
            {},
            node_memory,
            node_network,
            node_identities=node_identities,
        )


def test_llama_cpp_backend_places_with_relay_coordinator_fanout() -> None:
    topology = Topology()
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    node_relay = NodeId("node-relay")
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_node(node_relay)
    topology.add_connection(
        Connection(source=node_a, sink=node_relay, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_relay, sink=node_b, edge=create_socket_connection(2))
    )
    node_memory = {
        node_a: create_node_memory(2000 * 1024),
        node_b: create_node_memory(1000 * 1024),
        node_relay: create_node_memory(1000 * 1024),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
        node_relay: create_node_network(),
    }
    node_identities = {
        node_a: NodeIdentity(worker_enabled=True),
        node_b: NodeIdentity(
            worker_enabled=True,
            transport_endpoints=[
                AdvertisedTransportEndpoint(
                    purpose="data",
                    route_type="overlay",
                    host="203.0.113.22",
                    port=52435,
                )
            ],
        ),
        node_relay: NodeIdentity(
            relay_enabled=True,
            transport_endpoints=[
                AdvertisedTransportEndpoint(
                    purpose="api",
                    route_type="overlay",
                    host="203.0.113.21",
                    port=52415,
                )
            ],
        ),
    }
    model_card = ModelCard(
        model_id=ModelId("local/relay-gguf-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )
    placements = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=model_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.LlamaCpp,
            min_nodes=2,
        ),
        topology,
        {},
        node_memory,
        node_network,
        node_identities=node_identities,
        overlay_peers={
            node_a: [node_relay],
            node_relay: [node_b],
        },
        required_nodes={node_a, node_b},
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxRingInstance)
    assert instance.hosts_by_node[node_a][1].ip == "198.51.100.1"
    assert instance.hosts_by_node[node_a][1].port == 0
    assert len(instance.relay_routes_by_node[node_a]) == 1
    relay_route = instance.relay_routes_by_node[node_a][0]
    assert relay_route.transit_node_id == node_relay
    assert relay_route.sink_node_id == node_b
    assert relay_route.relay_api_host == "203.0.113.21"
    assert relay_route.relay_api_port == 52415
    assert relay_route.target_host == "203.0.113.22"
    assert relay_route.target_port == 52435
    assert instance.cai_api_urls_by_node[node_b] == [
        "cai-overlay:http://203.0.113.21:52415?"
        "targetNodeId=node-b&relayRole=bootstrap&transitNodeId=node-relay"
    ]


def test_llama_cpp_backend_places_through_non_worker_overlay_relay() -> None:
    topology = Topology()
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    node_relay = NodeId("node-relay")
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_node(node_relay)
    node_memory = {
        node_a: create_node_memory(2000 * 1024),
        node_b: create_node_memory(1000 * 1024),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
        node_relay: create_node_network(),
    }
    node_identities = {
        node_a: NodeIdentity(worker_enabled=True),
        node_b: NodeIdentity(
            worker_enabled=True,
            transport_endpoints=[
                AdvertisedTransportEndpoint(
                    purpose="data",
                    route_type="overlay",
                    host="203.0.113.32",
                    port=52435,
                )
            ],
        ),
        node_relay: NodeIdentity(
            relay_enabled=True,
            transport_endpoints=[
                AdvertisedTransportEndpoint(
                    purpose="api",
                    route_type="overlay",
                    host="203.0.113.31",
                    port=52415,
                )
            ],
        ),
    }
    model_card = ModelCard(
        model_id=ModelId("local/non-worker-relay-gguf-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )

    placements = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=model_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.LlamaCpp,
            min_nodes=2,
        ),
        topology,
        {},
        node_memory,
        node_network,
        node_identities=node_identities,
        overlay_peers={
            node_a: [node_relay],
            node_relay: [node_b],
        },
        required_nodes={node_a, node_b},
    )

    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxRingInstance)
    relay_route = instance.relay_routes_by_node[node_a][0]
    assert relay_route.transit_node_id == node_relay
    assert relay_route.sink_node_id == node_b
    assert relay_route.relay_api_host == "203.0.113.31"
    assert relay_route.target_host == "203.0.113.32"
    assert instance.cai_api_urls_by_node[node_b] == [
        "cai-overlay:http://203.0.113.31:52415?"
        "targetNodeId=node-b&relayRole=bootstrap&transitNodeId=node-relay"
    ]


def test_llama_cpp_relay_uses_runtime_port_when_sink_only_advertises_api_endpoint() -> None:
    topology = Topology()
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    node_relay = NodeId("node-relay")
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_node(node_relay)
    node_memory = {
        node_a: create_node_memory(2000 * 1024),
        node_b: create_node_memory(1000 * 1024),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
        node_relay: create_node_network(),
    }
    node_identities = {
        node_a: NodeIdentity(worker_enabled=True),
        node_b: NodeIdentity(
            worker_enabled=True,
            transport_endpoints=[
                AdvertisedTransportEndpoint(
                    purpose="api",
                    route_type="overlay",
                    host="203.0.113.42",
                    port=52425,
                )
            ],
        ),
        node_relay: NodeIdentity(
            relay_enabled=True,
            transport_endpoints=[
                AdvertisedTransportEndpoint(
                    purpose="api",
                    route_type="overlay",
                    host="203.0.113.41",
                    port=52415,
                )
            ],
        ),
    }
    model_card = ModelCard(
        model_id=ModelId("local/api-only-relay-gguf-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )

    placements = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=model_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.LlamaCpp,
            min_nodes=2,
        ),
        topology,
        {},
        node_memory,
        node_network,
        node_identities=node_identities,
        overlay_peers={
            node_a: [node_relay],
            node_relay: [node_b],
        },
        required_nodes={node_a, node_b},
    )

    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxRingInstance)
    relay_route = instance.relay_routes_by_node[node_a][0]
    assert relay_route.target_host == "203.0.113.42"
    assert relay_route.target_port == instance.ephemeral_port
    assert relay_route.target_port != 52425


def test_shared_relay_route_candidates_balance_equivalent_transits() -> None:
    topology = Topology()
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    node_c = NodeId("node-c")
    relay_a = NodeId("relay-a")
    relay_b = NodeId("relay-b")
    for node_id in [node_a, node_b, node_c, relay_a, relay_b]:
        topology.add_node(node_id)

    overlay_peers = {
        node_a: [relay_a, relay_b],
        node_b: [relay_a, relay_b],
        node_c: [relay_a, relay_b],
        relay_a: [node_a, node_b, node_c],
        relay_b: [node_a, node_b, node_c],
    }
    node_identities = {
        relay_a: NodeIdentity(relay_enabled=True),
        relay_b: NodeIdentity(relay_enabled=True),
    }

    routes = relay_route_candidates(
        topology,
        overlay_peers,
        node_identities,
        [node_a, node_b, node_c],
    )
    all_routes = relay_route_candidates(
        topology,
        overlay_peers,
        node_identities,
        [node_a, node_b, node_c],
        include_alternatives=True,
    )
    transit_counts = Counter(route["transitNodeId"] for route in routes)

    assert len(routes) == 6
    assert len(all_routes) == 12
    assert set(transit_counts) == {"relay-a", "relay-b"}
    assert max(transit_counts.values()) - min(transit_counts.values()) <= 1


def test_shared_relay_route_candidates_allow_participant_transit() -> None:
    topology = Topology()
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    node_c = NodeId("node-c")
    for node_id in [node_a, node_b, node_c]:
        topology.add_node(node_id)

    routes = relay_route_candidates(
        topology,
        {
            node_a: [node_c],
            node_c: [node_b],
        },
        {
            node_a: NodeIdentity(worker_enabled=True),
            node_b: NodeIdentity(worker_enabled=True),
            node_c: NodeIdentity(worker_enabled=True, relay_enabled=True),
        },
        [node_a, node_b, node_c],
    )

    route = next(
        item
        for item in routes
        if item["sourceNodeId"] == "node-a" and item["sinkNodeId"] == "node-b"
    )
    assert route["transitNodeId"] == "node-c"
    assert route["transitParticipates"] is True


def test_llama_cpp_relay_routes_keep_fallback_alternatives_for_sink() -> None:
    topology = Topology()
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    relay_a = NodeId("relay-a")
    relay_b = NodeId("relay-b")
    for node_id in [node_a, node_b, relay_a, relay_b]:
        topology.add_node(node_id)

    overlay_peers = {
        node_a: [relay_a, relay_b],
        relay_a: [node_b],
        relay_b: [node_b],
    }
    node_identities = {
        node_a: NodeIdentity(worker_enabled=True),
        node_b: NodeIdentity(
            worker_enabled=True,
            transport_endpoints=[
                AdvertisedTransportEndpoint(
                    purpose="data",
                    route_type="overlay",
                    host="203.0.113.21",
                    port=52435,
                )
            ],
        ),
        relay_a: NodeIdentity(
            relay_enabled=True,
            transport_endpoints=[
                AdvertisedTransportEndpoint(
                    purpose="api",
                    route_type="overlay",
                    host="203.0.113.31",
                    port=52415,
                )
            ],
        ),
        relay_b: NodeIdentity(
            relay_enabled=True,
            transport_endpoints=[
                AdvertisedTransportEndpoint(
                    purpose="api",
                    route_type="overlay",
                    host="203.0.113.32",
                    port=52415,
                )
            ],
        ),
    }
    selected = relay_route_candidates(
        topology,
        overlay_peers,
        node_identities,
        [node_a, node_b],
    )

    routes_by_node = get_llama_cpp_relay_routes_by_node(
        selected_cycle=Cycle(node_ids=[node_a, node_b]),
        cycle_digraph=topology,
        ephemeral_port=59657,
        node_network={
            node_a: create_node_network(),
            node_b: create_node_network(),
            relay_a: create_node_network(),
            relay_b: create_node_network(),
        },
        node_identities=node_identities,
        overlay_peers=overlay_peers,
    )

    routes = routes_by_node[node_a]
    assert len(routes) == 2
    assert routes[0].transit_node_id == NodeId(selected[0]["transitNodeId"])
    assert {route.transit_node_id for route in routes} == {relay_a, relay_b}
    assert all(route.sink_node_id == node_b for route in routes)


def test_mlx_ring_backend_places_on_overlay_connected_nodes_without_direct_cycle() -> None:
    topology = Topology()
    node_a = NodeId()
    node_b = NodeId()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_socket_connection(1))
    )
    node_memory = {
        node_a: create_node_memory(2000 * 1024),
        node_b: create_node_memory(2000 * 1024),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }
    node_identities = {
        node_a: NodeIdentity(data_host="10.0.0.10"),
        node_b: NodeIdentity(data_host="10.0.0.11"),
    }
    model_card = ModelCard(
        model_id=ModelId("test-overlay-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )

    placements = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=model_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=2,
        ),
        topology,
        {},
        node_memory,
        node_network,
        node_identities=node_identities,
        overlay_peers={node_a: [node_b]},
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxRingInstance)
    assert set(instance.shard_assignments.node_to_runner.keys()) == {node_a, node_b}
    assert any(host.ip == "0.0.0.0" for host in instance.hosts_by_node[node_a])
    assert any(host.ip == "0.0.0.0" for host in instance.hosts_by_node[node_b])
    assert any(
        host.ip == "10.0.0.11"
        for host in instance.hosts_by_node[node_a]
        if host.ip != "0.0.0.0"
    )
    assert any(
        host.ip == "10.0.0.10"
        for host in instance.hosts_by_node[node_b]
        if host.ip != "0.0.0.0"
    )


def test_mlx_ring_backend_uses_advertised_api_host_without_data_host() -> None:
    topology = Topology()
    node_a = NodeId()
    node_b = NodeId()
    topology.add_node(node_a)
    topology.add_node(node_b)
    node_memory = {
        node_a: create_node_memory(2000 * 1024),
        node_b: create_node_memory(2000 * 1024),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }
    node_identities = {
        node_a: NodeIdentity(api_host="198.51.100.10"),
        node_b: NodeIdentity(api_host="198.51.100.20"),
    }
    model_card = ModelCard(
        model_id=ModelId("test-overlay-api-host-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )

    placements = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=model_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=2,
        ),
        topology,
        {},
        node_memory,
        node_network,
        node_identities=node_identities,
        overlay_peers={node_a: [node_b]},
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxRingInstance)
    assert any(
        host.ip == "198.51.100.20"
        for host in instance.hosts_by_node[node_a]
        if host.ip != "0.0.0.0"
    )
    assert any(
        host.ip == "198.51.100.10"
        for host in instance.hosts_by_node[node_b]
        if host.ip != "0.0.0.0"
    )


def test_llama_cpp_backend_uses_advertised_api_host_without_data_host() -> None:
    topology = Topology()
    node_a = NodeId()
    node_b = NodeId()
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_socket_connection(2))
    )
    node_memory = {
        node_a: create_node_memory(2000 * 1024),
        node_b: create_node_memory(1000 * 1024),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }
    node_identities = {
        node_a: NodeIdentity(api_host="198.51.100.10"),
        node_b: NodeIdentity(api_host="198.51.100.20"),
    }
    model_card = ModelCard(
        model_id=ModelId("local/api-host-gguf-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )

    placements = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=model_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.LlamaCpp,
            min_nodes=2,
        ),
        topology,
        {},
        node_memory,
        node_network,
        node_identities=node_identities,
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxRingInstance)
    assert any(
        host.ip == "198.51.100.20"
        for host in instance.hosts_by_node[node_a]
        if host.ip != "0.0.0.0"
    )
    assert any(
        host.ip == "198.51.100.10"
        for host in instance.hosts_by_node[node_b]
        if host.ip != "0.0.0.0"
    )


def test_llama_cpp_backend_prefers_transport_data_endpoints() -> None:
    topology = Topology()
    node_a = NodeId()
    node_b = NodeId()
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_socket_connection(2))
    )
    node_memory = {
        node_a: create_node_memory(2000 * 1024),
        node_b: create_node_memory(1000 * 1024),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }
    node_identities = {
        node_a: NodeIdentity(
            api_host="198.51.100.10",
            transport_endpoints=[
                AdvertisedTransportEndpoint(
                    purpose="data",
                    route_type="overlay",
                    host="26.97.29.153",
                    port=62001,
                    source="interface_scan",
                )
            ],
        ),
        node_b: NodeIdentity(
            api_host="198.51.100.20",
            transport_endpoints=[
                AdvertisedTransportEndpoint(
                    purpose="data",
                    route_type="overlay",
                    host="26.97.29.154",
                    port=62002,
                    source="interface_scan",
                )
            ],
        ),
    }
    model_card = ModelCard(
        model_id=ModelId("local/transport-endpoint-gguf-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )

    placements = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=model_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.LlamaCpp,
            min_nodes=2,
        ),
        topology,
        {},
        node_memory,
        node_network,
        node_identities=node_identities,
    )

    instance = next(iter(placements.values()))
    assert any(
        host.ip == "26.97.29.154" and host.port == 62002
        for host in instance.hosts_by_node[node_a]
        if host.ip != "0.0.0.0"
    )
    assert any(
        host.ip == "26.97.29.153" and host.port == 62001
        for host in instance.hosts_by_node[node_b]
        if host.ip != "0.0.0.0"
    )


def test_llama_cpp_backend_uses_healthy_direct_route_without_topology_edge() -> None:
    topology = Topology()
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    relay = NodeId("relay")
    for node_id in (node_a, node_b, relay):
        topology.add_node(node_id)
    topology.add_connection(
        Connection(source=node_a, sink=relay, edge=create_socket_connection(1))
    )

    node_memory = {
        node_a: create_node_memory(2000 * 1024),
        node_b: create_node_memory(1000 * 1024),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
        relay: create_node_network(),
    }
    node_identities = {
        node_a: NodeIdentity(api_host="26.242.160.75"),
        node_b: NodeIdentity(api_host="26.97.29.153"),
        relay: NodeIdentity(relay_enabled=True, api_host="85.137.164.250", api_port=52415),
    }
    route_health_records = [
        SimpleNamespace(
            source_node_id=str(node_a),
            sink_node_id=str(node_b),
            route_type="direct_data",
            reachable=True,
            endpoint_url="http://26.97.29.153:52425",
        )
    ]
    model_card = ModelCard(
        model_id=ModelId("local/direct-health-gguf-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )

    placements = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=model_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.LlamaCpp,
            min_nodes=2,
        ),
        topology,
        {},
        node_memory,
        node_network,
        node_identities=node_identities,
        overlay_peers={node_a: [relay], relay: [node_b]},
        route_health_records=route_health_records,
    )

    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxRingInstance)
    assert instance.relay_routes_by_node.get(node_a, []) == []
    assert any(
        host.ip == "26.97.29.153" and host.port == instance.ephemeral_port
        for host in instance.hosts_by_node[node_a]
        if host.ip != "0.0.0.0"
    )


def test_llama_cpp_direct_route_health_is_directional_for_nat() -> None:
    topology = Topology()
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    topology.add_node(node_a)
    topology.add_node(node_b)

    node_memory = {
        node_a: create_node_memory(2000 * 1024),
        node_b: create_node_memory(1000 * 1024),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }
    route_health_records = [
        SimpleNamespace(
            source_node_id=str(node_b),
            sink_node_id=str(node_a),
            route_type="direct_data",
            reachable=True,
            endpoint_url="http://26.242.160.75:52445",
            checked_at=datetime.now(tz=UTC).isoformat(),
        )
    ]
    cycles = get_llama_cpp_direct_candidate_cycles(
        topology,
        node_memory,
        min_nodes=2,
        route_health_records=route_health_records,
    )

    assert [cycle.node_ids for cycle in cycles] == [[node_b, node_a]]


def test_llama_cpp_hosts_prefer_proven_rpc_endpoint_host() -> None:
    topology = Topology()
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_socket_connection(1))
    )

    node_memory = {
        node_a: create_node_memory(2000 * 1024),
        node_b: create_node_memory(1000 * 1024),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }
    node_identities = {
        node_a: NodeIdentity(api_host="10.0.0.10"),
        node_b: NodeIdentity(api_host="198.51.100.200"),
    }
    route_health_records = [
        SimpleNamespace(
            source_node_id=str(node_a),
            sink_node_id=str(node_b),
            route_type="llama_cpp_rpc_direct",
            reachable=True,
            endpoint_url="llama-cpp-rpc://10.2.0.7:55779",
            checked_at=datetime.now(tz=UTC).isoformat(),
        )
    ]
    model_card = ModelCard(
        model_id=ModelId("local/direct-health-gguf-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )

    placements = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=model_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.LlamaCpp,
            min_nodes=2,
        ),
        topology,
        {},
        node_memory,
        node_network,
        node_identities=node_identities,
        route_health_records=route_health_records,
    )

    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxRingInstance)
    assert any(
        host.ip == "10.2.0.7" and host.port == instance.ephemeral_port
        for host in instance.hosts_by_node[node_a]
        if host.ip != "0.0.0.0"
    )


def test_llama_cpp_backend_rejects_known_failed_rpc_protocol_route() -> None:
    topology = Topology()
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    topology.add_node(node_a)
    topology.add_node(node_b)

    node_memory = {
        node_a: create_node_memory(2000 * 1024),
        node_b: create_node_memory(1000 * 1024),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }
    node_identities = {
        node_a: NodeIdentity(api_host="26.242.160.75"),
        node_b: NodeIdentity(api_host="26.97.29.153"),
    }
    checked_at = datetime.now(tz=UTC).isoformat()
    route_health_records = [
        SimpleNamespace(
            source_node_id=str(node_a),
            sink_node_id=str(node_b),
            route_type="direct_data",
            reachable=True,
            endpoint_url="http://26.97.29.153:52425",
            checked_at="2026-05-03T00:00:00+00:00",
        ),
        SimpleNamespace(
            source_node_id=str(node_a),
            sink_node_id=str(node_b),
            route_type="llama_cpp_rpc_direct",
            reachable=False,
            endpoint_url="llama-cpp-rpc://26.97.29.153:55779",
            checked_at=checked_at,
            error="Remote RPC server crashed or returned malformed response",
        ),
        SimpleNamespace(
            source_node_id=str(node_b),
            sink_node_id=str(node_a),
            route_type="llama_cpp_rpc_direct",
            reachable=False,
            endpoint_url="llama-cpp-rpc://26.242.160.75:55779",
            checked_at=checked_at,
            error="Remote RPC server crashed or returned malformed response",
        ),
    ]
    model_card = ModelCard(
        model_id=ModelId("local/direct-health-gguf-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )

    with pytest.raises(ValueError, match="No usable decentralized llama.cpp RPC route"):
        place_instance(
            PlaceInstance(
                command_id=CommandId(),
                model_card=model_card,
                sharding=Sharding.Pipeline,
                instance_meta=InstanceMeta.LlamaCpp,
                min_nodes=2,
            ),
            topology,
            {},
            node_memory,
            node_network,
            node_identities=node_identities,
            route_health_records=route_health_records,
        )


def test_llama_cpp_backend_requires_rpc_proof_when_strict_mode_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAI_REQUIRE_LLAMA_CPP_RPC_PROOF", "1")
    topology = Topology()
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    topology.add_node(node_a)
    topology.add_node(node_b)

    node_memory = {
        node_a: create_node_memory(2000 * 1024),
        node_b: create_node_memory(1000 * 1024),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }
    node_identities = {
        node_a: NodeIdentity(api_host="26.242.160.75"),
        node_b: NodeIdentity(api_host="26.97.29.153"),
    }
    route_health_records = [
        SimpleNamespace(
            source_node_id=str(node_a),
            sink_node_id=str(node_b),
            route_type="direct_data",
            reachable=True,
            endpoint_url="tcp://26.97.29.153:55779",
            checked_at=datetime.now(tz=UTC).isoformat(),
        )
    ]
    model_card = ModelCard(
        model_id=ModelId("local/direct-health-gguf-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )

    with pytest.raises(ValueError, match="Strict RPC proof is enabled"):
        place_instance(
            PlaceInstance(
                command_id=CommandId(),
                model_card=model_card,
                sharding=Sharding.Pipeline,
                instance_meta=InstanceMeta.LlamaCpp,
                min_nodes=2,
            ),
            topology,
            {},
            node_memory,
            node_network,
            node_identities=node_identities,
            route_health_records=route_health_records,
        )


def test_llama_cpp_backend_accepts_rpc_proof_when_strict_mode_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAI_REQUIRE_LLAMA_CPP_RPC_PROOF", "1")
    topology = Topology()
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    topology.add_node(node_a)
    topology.add_node(node_b)

    node_memory = {
        node_a: create_node_memory(2000 * 1024),
        node_b: create_node_memory(1000 * 1024),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }
    node_identities = {
        node_a: NodeIdentity(api_host="26.242.160.75"),
        node_b: NodeIdentity(api_host="26.97.29.153"),
    }
    checked_at = datetime.now(tz=UTC).isoformat()
    route_health_records = [
        SimpleNamespace(
            source_node_id=str(node_a),
            sink_node_id=str(node_b),
            route_type="direct_data",
            reachable=True,
            endpoint_url="tcp://26.97.29.153:55779",
            checked_at=checked_at,
        ),
        SimpleNamespace(
            source_node_id=str(node_a),
            sink_node_id=str(node_b),
            route_type="llama_cpp_rpc_direct",
            reachable=True,
            endpoint_url="llama-cpp-rpc://26.97.29.153:55779",
            checked_at=checked_at,
        ),
    ]
    model_card = ModelCard(
        model_id=ModelId("local/direct-health-gguf-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )

    placements = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=model_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.LlamaCpp,
            min_nodes=2,
        ),
        topology,
        {},
        node_memory,
        node_network,
        node_identities=node_identities,
        route_health_records=route_health_records,
    )

    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxRingInstance)
    assert len(instance.shard_assignments.node_to_runner) == 2


def test_llama_cpp_backend_prefers_low_latency_compute_cell() -> None:
    topology = Topology()
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    node_c = NodeId("node-c")
    for node_id in (node_a, node_b, node_c):
        topology.add_node(node_id)

    node_memory = {
        node_a: create_node_memory(1000 * 1024),
        node_b: create_node_memory(3000 * 1024),
        node_c: create_node_memory(1200 * 1024),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
        node_c: create_node_network(),
    }
    node_identities = {
        node_a: NodeIdentity(api_host="10.0.0.10"),
        node_b: NodeIdentity(api_host="10.0.0.20"),
        node_c: NodeIdentity(api_host="10.0.0.30"),
    }
    checked_at = datetime.now(tz=UTC).isoformat()
    route_health_records = [
        SimpleNamespace(
            source_node_id=str(node_b),
            sink_node_id=str(node_a),
            route_type="llama_cpp_rpc_direct",
            reachable=True,
            endpoint_url="llama-cpp-rpc://10.0.0.10:55779",
            checked_at=checked_at,
            latency_ms=48.0,
        ),
        SimpleNamespace(
            source_node_id=str(node_c),
            sink_node_id=str(node_a),
            route_type="llama_cpp_rpc_direct",
            reachable=True,
            endpoint_url="llama-cpp-rpc://10.0.0.10:55779",
            checked_at=checked_at,
            latency_ms=6.0,
        ),
    ]
    model_card = ModelCard(
        model_id=ModelId("local/direct-health-gguf-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )

    placements = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=model_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.LlamaCpp,
            min_nodes=2,
        ),
        topology,
        {},
        node_memory,
        node_network,
        node_identities=node_identities,
        route_health_records=route_health_records,
    )

    instance = next(iter(placements.values()))
    assert set(instance.shard_assignments.node_to_runner) == {node_a, node_c}


def test_llama_cpp_backend_rejects_wan_risky_compute_cell() -> None:
    topology = Topology()
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    topology.add_node(node_a)
    topology.add_node(node_b)

    node_memory = {
        node_a: create_node_memory(2000 * 1024),
        node_b: create_node_memory(1000 * 1024),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }
    node_identities = {
        node_a: NodeIdentity(
            api_host="10.0.0.10",
            worker_enabled=True,
            readiness={
                "caiOwnedTransport": {
                    "implemented": True,
                    "runtimeReady": False,
                    "status": "test_adapter_ready",
                }
            },
        ),
        node_b: NodeIdentity(
            api_host="10.0.0.20",
            worker_enabled=True,
            readiness={
                "caiOwnedTransport": {
                    "implemented": True,
                    "runtimeReady": False,
                    "status": "test_adapter_ready",
                }
            },
        ),
    }
    route_health_records = [
        SimpleNamespace(
            source_node_id=str(node_a),
            sink_node_id=str(node_b),
            route_type="llama_cpp_rpc_direct",
            reachable=True,
            endpoint_url="llama-cpp-rpc://10.0.0.20:55779",
            checked_at=datetime.now(tz=UTC).isoformat(),
            latency_ms=48.0,
        )
    ]
    model_card = ModelCard(
        model_id=ModelId("local/direct-health-gguf-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )

    with pytest.raises(ValueError, match="low-latency decentralized llama.cpp"):
        place_instance(
            PlaceInstance(
                command_id=CommandId(),
                model_card=model_card,
                sharding=Sharding.Pipeline,
                instance_meta=InstanceMeta.LlamaCpp,
                min_nodes=2,
            ),
            topology,
            {},
            node_memory,
            node_network,
            node_identities=node_identities,
            route_health_records=route_health_records,
        )


def test_llama_cpp_backend_accepts_wan_risky_cell_with_cai_owned_route_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAI_OWNED_TRANSPORT_GENERATION_ENABLED", "1")
    topology = Topology()
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    topology.add_node(node_a)
    topology.add_node(node_b)

    node_memory = {
        node_a: create_node_memory(2000 * 1024),
        node_b: create_node_memory(1000 * 1024),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }
    node_identities = {
        node_a: NodeIdentity(
            api_host="10.0.0.10",
            worker_enabled=True,
            readiness={
                "caiOwnedTransport": {
                    "implemented": True,
                    "runtimeReady": False,
                    "status": "test_adapter_ready",
                }
            },
        ),
        node_b: NodeIdentity(
            api_host="10.0.0.20",
            worker_enabled=True,
            readiness={
                "caiOwnedTransport": {
                    "implemented": True,
                    "runtimeReady": False,
                    "status": "test_adapter_ready",
                }
            },
        ),
    }
    route_health_records = [
        SimpleNamespace(
            source_node_id=str(node_a),
            sink_node_id=str(node_b),
            route_type="llama_cpp_rpc_direct",
            reachable=True,
            endpoint_url="llama-cpp-rpc://10.0.0.20:55779",
            checked_at=datetime.now(tz=UTC).isoformat(),
            latency_ms=48.0,
        ),
        SimpleNamespace(
            source_node_id=str(node_b),
            sink_node_id=str(node_a),
            route_type="llama_cpp_rpc_direct",
            reachable=True,
            endpoint_url="llama-cpp-rpc://10.0.0.10:55779",
            checked_at=datetime.now(tz=UTC).isoformat(),
            latency_ms=48.0,
        ),
    ]
    model_card = ModelCard(
        model_id=ModelId("local/direct-health-gguf-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )

    placements = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=model_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.LlamaCpp,
            min_nodes=2,
        ),
        topology,
        {},
        node_memory,
        node_network,
        node_identities=node_identities,
        route_health_records=route_health_records,
    )

    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxRingInstance)
    assert set(instance.shard_assignments.node_to_runner) == {node_a, node_b}


def test_cai_owned_worker_readiness_requires_runtime_live_proof() -> None:
    stale_identity = NodeIdentity(
        worker_enabled=True,
        readiness={
            "caiOwnedTransport": {
                "implemented": True,
                "runtimeReady": True,
                "status": "ready",
            }
        },
    )
    verified_identity = NodeIdentity(
        worker_enabled=True,
        readiness={
            "caiOwnedTransport": {
                "implemented": True,
                "runtimeReady": True,
                "runtimeReadyProof": {"verified": True},
                "llmShardSelfTest": {
                    "productionReady": True,
                    "generationProbeReady": True,
                    "backendHealthReady": True,
                },
                "status": "ready",
            }
        },
    )

    stale = _node_cai_owned_worker_readiness(
        NodeId("node-a"),
        stale_identity,
        require_runtime_ready=True,
    )
    verified = _node_cai_owned_worker_readiness(
        NodeId("node-b"),
        verified_identity,
        require_runtime_ready=True,
    )

    assert stale["ready"] is False
    assert stale["reason"] == "cai_owned_runtime_not_ready"
    assert "verified live proof" in stale["error"]
    assert verified["ready"] is True


def test_llama_cpp_backend_rejects_wan_risky_cell_without_cai_owned_reverse_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAI_OWNED_TRANSPORT_GENERATION_ENABLED", "1")
    topology = Topology()
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    topology.add_node(node_a)
    topology.add_node(node_b)

    node_memory = {
        node_a: create_node_memory(2000 * 1024),
        node_b: create_node_memory(1000 * 1024),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }
    node_identities = {
        node_a: NodeIdentity(api_host="10.0.0.10"),
        node_b: NodeIdentity(api_host="10.0.0.20"),
    }
    route_health_records = [
        SimpleNamespace(
            source_node_id=str(node_a),
            sink_node_id=str(node_b),
            route_type="llama_cpp_rpc_direct",
            reachable=True,
            endpoint_url="llama-cpp-rpc://10.0.0.20:55779",
            checked_at=datetime.now(tz=UTC).isoformat(),
            latency_ms=48.0,
        )
    ]
    model_card = ModelCard(
        model_id=ModelId("local/direct-health-gguf-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )

    with pytest.raises(ValueError, match="CAI-owned transport route proof"):
        place_instance(
            PlaceInstance(
                command_id=CommandId(),
                model_card=model_card,
                sharding=Sharding.Pipeline,
                instance_meta=InstanceMeta.LlamaCpp,
                min_nodes=2,
            ),
            topology,
            {},
            node_memory,
            node_network,
            node_identities=node_identities,
            route_health_records=route_health_records,
        )


def test_llama_cpp_backend_rejects_wan_risky_cell_without_cai_owned_worker_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAI_OWNED_TRANSPORT_GENERATION_ENABLED", "1")
    topology = Topology()
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    topology.add_node(node_a)
    topology.add_node(node_b)

    node_memory = {
        node_a: create_node_memory(2000 * 1024),
        node_b: create_node_memory(1000 * 1024),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }
    node_identities = {
        node_a: NodeIdentity(
            api_host="10.0.0.10",
            worker_enabled=True,
            readiness={
                "caiOwnedTransport": {
                    "implemented": True,
                    "runtimeReady": False,
                    "status": "test_adapter_ready",
                }
            },
        ),
        node_b: NodeIdentity(api_host="10.0.0.20", worker_enabled=True),
    }
    route_health_records = [
        SimpleNamespace(
            source_node_id=str(node_a),
            sink_node_id=str(node_b),
            route_type="llama_cpp_rpc_direct",
            reachable=True,
            endpoint_url="llama-cpp-rpc://10.0.0.20:55779",
            checked_at=datetime.now(tz=UTC).isoformat(),
            latency_ms=48.0,
        ),
        SimpleNamespace(
            source_node_id=str(node_b),
            sink_node_id=str(node_a),
            route_type="llama_cpp_rpc_direct",
            reachable=True,
            endpoint_url="llama-cpp-rpc://10.0.0.10:55779",
            checked_at=datetime.now(tz=UTC).isoformat(),
            latency_ms=48.0,
        ),
    ]
    model_card = ModelCard(
        model_id=ModelId("local/direct-health-gguf-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )

    with pytest.raises(ValueError, match="CAI-owned transport readiness"):
        place_instance(
            PlaceInstance(
                command_id=CommandId(),
                model_card=model_card,
                sharding=Sharding.Pipeline,
                instance_meta=InstanceMeta.LlamaCpp,
                min_nodes=2,
            ),
            topology,
            {},
            node_memory,
            node_network,
            node_identities=node_identities,
            route_health_records=route_health_records,
        )


def test_private_llama_cpp_model_allows_runtime_rpc_proof_bootstrap_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CAI_REQUIRE_LLAMA_CPP_RPC_PROOF", raising=False)
    monkeypatch.setenv("CAI_ALLOW_LLAMA_CPP_RPC_BOOTSTRAP_WITHOUT_PROOF", "1")
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
    node_memory = {
        node_a: create_node_memory(2000 * 1024),
        node_b: create_node_memory(1000 * 1024),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }
    model_card = ModelCard(
        model_id=ModelId("cai-network/Qwen3-0.6B-GGUF"),
        storage_size=Memory.from_kb(1000),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )

    placements = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=model_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.LlamaCpp,
            min_nodes=2,
        ),
        topology,
        {},
        node_memory,
        node_network,
    )

    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxRingInstance)
    assert len(instance.shard_assignments.node_to_runner) == 2


def test_private_llama_cpp_model_allows_single_node_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CAI_REQUIRE_LLAMA_CPP_RPC_PROOF", raising=False)
    topology = Topology()
    node_a = NodeId("node-a")
    topology.add_node(node_a)
    node_memory = {node_a: create_node_memory(2000 * 1024)}
    node_network = {node_a: create_node_network()}
    model_card = ModelCard(
        model_id=ModelId("cai-network/Qwen3-0.6B-GGUF"),
        storage_size=Memory.from_kb(1000),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )

    placements = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=model_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.LlamaCpp,
            min_nodes=1,
        ),
        topology,
        {},
        node_memory,
        node_network,
    )

    instance = next(iter(placements.values()))
    assert len(instance.shard_assignments.node_to_runner) == 1


def test_private_llama_cpp_model_requires_rpc_proof_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CAI_REQUIRE_LLAMA_CPP_RPC_PROOF", raising=False)
    monkeypatch.delenv("CAI_ALLOW_LLAMA_CPP_RPC_BOOTSTRAP_WITHOUT_PROOF", raising=False)
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
    node_memory = {
        node_a: create_node_memory(2000 * 1024),
        node_b: create_node_memory(1000 * 1024),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }
    model_card = ModelCard(
        model_id=ModelId("cai-network/Qwen3-0.6B-GGUF"),
        storage_size=Memory.from_kb(1000),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )

    with pytest.raises(ValueError, match="Strict RPC proof is enabled"):
        place_instance(
            PlaceInstance(
                command_id=CommandId(),
                model_card=model_card,
                sharding=Sharding.Pipeline,
                instance_meta=InstanceMeta.LlamaCpp,
                min_nodes=2,
            ),
            topology,
            {},
            node_memory,
            node_network,
        )


def test_private_llama_cpp_model_accepts_rpc_proof_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CAI_REQUIRE_LLAMA_CPP_RPC_PROOF", raising=False)
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
    node_memory = {
        node_a: create_node_memory(2000 * 1024),
        node_b: create_node_memory(1000 * 1024),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }
    checked_at = datetime.now(tz=UTC).isoformat()
    route_health_records = [
        SimpleNamespace(
            source_node_id=str(node_a),
            sink_node_id=str(node_b),
            route_type="llama_cpp_rpc_direct",
            reachable=True,
            endpoint_url="llama-cpp-rpc://node-b:55779",
            checked_at=checked_at,
        ),
        SimpleNamespace(
            source_node_id=str(node_b),
            sink_node_id=str(node_a),
            route_type="llama_cpp_rpc_direct",
            reachable=True,
            endpoint_url="llama-cpp-rpc://node-a:55779",
            checked_at=checked_at,
        ),
    ]
    model_card = ModelCard(
        model_id=ModelId("cai-network/Qwen3-0.6B-GGUF"),
        storage_size=Memory.from_kb(1000),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )

    placements = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=model_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.LlamaCpp,
            min_nodes=2,
        ),
        topology,
        {},
        node_memory,
        node_network,
        route_health_records=route_health_records,
    )

    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxRingInstance)
    assert len(instance.shard_assignments.node_to_runner) == 2


def test_llama_cpp_relay_rpc_proof_keeps_relay_route_when_strict_mode_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAI_REQUIRE_LLAMA_CPP_RPC_PROOF", "1")
    topology = Topology()
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    node_relay = NodeId("node-relay")
    for node_id in (node_a, node_b, node_relay):
        topology.add_node(node_id)

    node_memory = {
        node_a: create_node_memory(2000 * 1024),
        node_b: create_node_memory(1000 * 1024),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
        node_relay: create_node_network(),
    }
    node_identities = {
        node_a: NodeIdentity(worker_enabled=True),
        node_b: NodeIdentity(
            worker_enabled=True,
            transport_endpoints=[
                AdvertisedTransportEndpoint(
                    purpose="data",
                    route_type="overlay",
                    host="203.0.113.42",
                    port=50052,
                )
            ],
        ),
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
    }
    checked_at = datetime.now(tz=UTC).isoformat()
    route_health_records = [
        SimpleNamespace(
            source_node_id=str(node_a),
            sink_node_id=str(node_b),
            transit_node_id=str(node_relay),
            route_type="llama_cpp_rpc_relay",
            reachable=True,
            endpoint_url="relay://node-relay/203.0.113.42:50052",
            checked_at=checked_at,
        ),
    ]
    model_card = ModelCard(
        model_id=ModelId("local/relay-proof-gguf-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )

    placements = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=model_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.LlamaCpp,
            min_nodes=2,
        ),
        topology,
        {},
        node_memory,
        node_network,
        node_identities=node_identities,
        overlay_peers={node_a: [node_relay], node_relay: [node_b]},
        required_nodes={node_a, node_b},
        route_health_records=route_health_records,
    )

    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxRingInstance)
    assert instance.hosts_by_node[node_a][1].ip == "198.51.100.1"
    assert instance.hosts_by_node[node_a][1].port == 0
    relay_route = instance.relay_routes_by_node[node_a][0]
    assert relay_route.transit_node_id == node_relay
    assert relay_route.sink_node_id == node_b


def test_llama_cpp_placement_prefers_cached_assignment_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = Topology()
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    node_c = NodeId("node-c")
    node_d = NodeId("node-d")
    for node_id in (node_a, node_b, node_c, node_d):
        topology.add_node(node_id)
    for source, sink, port in (
        (node_a, node_b, 1),
        (node_b, node_a, 2),
        (node_c, node_d, 3),
        (node_d, node_c, 4),
    ):
        topology.add_connection(
            Connection(source=source, sink=sink, edge=create_socket_connection(port))
        )

    node_memory = {
        node_id: create_node_memory(1000 * 1024)
        for node_id in (node_a, node_b, node_c, node_d)
    }
    node_network = {
        node_id: create_node_network()
        for node_id in (node_a, node_b, node_c, node_d)
    }
    model_card = ModelCard(
        model_id=ModelId("Qwen/Qwen3-0.6B-GGUF"),
        storage_size=Memory.from_kb(1000),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )

    class _FakeSourceKind:
        PEER_CACHE = "peer_cache"

    class _FakeManifest:
        catalog_id = "qwen-demo"
        version = "v1"

        @staticmethod
        def compute_chunk_coverage(
            present_chunk_ids,
            *,
            start_layer: int,
            end_layer: int,
        ):
            required_id = f"layers-{start_layer}-{end_layer}"
            ready = required_id in set(present_chunk_ids)
            return SimpleNamespace(
                required_bytes=100,
                present_bytes=100 if ready else 0,
                missing_chunk_ids=() if ready else (required_id,),
                ready=ready,
            )

    class _FakeModelDistribution:
        ChunkInventorySourceKind = _FakeSourceKind

        @staticmethod
        def select_model_package_manifest_for_model(model_id: str):
            return _FakeManifest() if model_id == str(model_card.model_id) else None

        @staticmethod
        def build_chunk_inventory_index(catalog_id, version, *, source_kind):  # noqa: ARG004
            cached_chunks = {"layers-0-4", "layers-4-8"}
            return {
                str(node_c): cached_chunks,
                str(node_d): cached_chunks,
            }

    monkeypatch.setattr(
        "cai.master.placement._get_cai_model_distribution_module",
        lambda: _FakeModelDistribution,
    )

    placements = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=model_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.LlamaCpp,
            min_nodes=2,
        ),
        topology,
        {},
        node_memory,
        node_network,
    )

    instance = next(iter(placements.values()))
    assert set(instance.shard_assignments.node_to_runner.keys()) == {node_c, node_d}


def test_llama_cpp_backend_prefers_advertised_data_ports() -> None:
    topology = Topology()
    node_a = NodeId()
    node_b = NodeId()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_socket_connection(1))
    )
    node_memory = {
        node_a: create_node_memory(2000 * 1024),
        node_b: create_node_memory(1000 * 1024),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }
    node_identities = {
        node_a: NodeIdentity(data_host="85.137.164.250", data_port=52435),
        node_b: NodeIdentity(data_host="85.137.164.250"),
    }
    model_card = ModelCard(
        model_id=ModelId("local/gguf-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )

    placements = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=model_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.LlamaCpp,
            min_nodes=2,
        ),
        topology,
        {},
        node_memory,
        node_network,
        node_identities=node_identities,
    )

    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxRingInstance)
    hosts_a = instance.hosts_by_node[node_a]
    hosts_b = instance.hosts_by_node[node_b]

    assert any(host.ip == "0.0.0.0" and host.port == 52435 for host in hosts_a)
    assert any(
        host.ip == "85.137.164.250" and host.port == instance.ephemeral_port
        for host in hosts_a
        if host.ip != "0.0.0.0"
    )
    assert any(host.ip == "85.137.164.250" and host.port == 52435 for host in hosts_b if host.ip != "0.0.0.0")


def test_get_instance_placements_one_node_not_fit() -> None:
    topology = Topology()
    node_id = NodeId()
    topology.add_node(node_id)
    node_memory = {node_id: create_node_memory(1000 * 1024)}
    node_network = {node_id: create_node_network()}
    cic = place_instance_command(
        model_card=ModelCard(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_kb(1001),
            n_layers=10,
            hidden_size=1000,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
        ),
    )

    with pytest.raises(ValueError, match="No cycles found with sufficient memory"):
        place_instance(cic, topology, {}, node_memory, node_network)


def test_private_model_ram_headroom_biases_layers_toward_stronger_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAI_PRIVATE_NETWORK_MODEL_IDS", "private/test-model")
    monkeypatch.setenv("CAI_PRIVATE_NETWORK_MODEL_MIN_NODES", "2")
    monkeypatch.setenv("CAI_PRIVATE_NETWORK_MODEL_MIN_RAM_HEADROOM_MB", "256")
    monkeypatch.setenv("CAI_PRIVATE_NETWORK_MODEL_MIN_PIPELINE_LAYERS_PER_NODE", "2")
    get_private_network_model_policy.cache_clear()

    topology = Topology()
    node_a = NodeId()
    node_b = NodeId()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_socket_connection(2))
    )

    node_memory = {
        node_a: create_node_memory(Memory.from_mb(800).in_bytes),
        node_b: create_node_memory(Memory.from_mb(500).in_bytes),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }
    model_card = ModelCard(
        model_id=ModelId("private/test-model"),
        storage_size=Memory.from_mb(280),
        n_layers=28,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )

    placements = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=model_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=2,
        ),
        topology,
        {},
        node_memory,
        node_network,
    )

    instance = next(iter(placements.values()))
    runner_a = instance.shard_assignments.node_to_runner[node_a]
    runner_b = instance.shard_assignments.node_to_runner[node_b]
    shard_a = instance.shard_assignments.runner_to_shard[runner_a]
    shard_b = instance.shard_assignments.runner_to_shard[runner_b]

    assert shard_a.end_layer - shard_a.start_layer == 19
    assert shard_b.end_layer - shard_b.start_layer == 9
    get_private_network_model_policy.cache_clear()


def test_private_llama_cpp_model_allows_single_worker_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAI_PRIVATE_NETWORK_MODEL_IDS", "private/test-model")
    monkeypatch.setenv("CAI_PRIVATE_NETWORK_MODEL_MIN_NODES", "2")
    monkeypatch.setenv("CAI_PRIVATE_NETWORK_MODEL_ALLOW_BOOTSTRAP_SINGLE_NODE", "true")
    get_private_network_model_policy.cache_clear()

    topology = Topology()
    node_a = NodeId()
    topology.add_node(node_a)
    node_memory = {node_a: create_node_memory(Memory.from_mb(800).in_bytes)}
    node_network = {node_a: create_node_network()}
    model_card = ModelCard(
        model_id=ModelId("private/test-model"),
        storage_size=Memory.from_mb(280),
        n_layers=28,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )

    placements = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=model_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.LlamaCpp,
            min_nodes=1,
        ),
        topology,
        {},
        node_memory,
        node_network,
    )

    instance = next(iter(placements.values()))

    assert list(instance.shard_assignments.node_to_runner.keys()) == [node_a]
    get_private_network_model_policy.cache_clear()


def test_private_model_rebalances_pathological_one_layer_secondary_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAI_PRIVATE_NETWORK_MODEL_IDS", "private/test-model")
    monkeypatch.setenv("CAI_PRIVATE_NETWORK_MODEL_MIN_NODES", "2")
    monkeypatch.setenv("CAI_PRIVATE_NETWORK_MODEL_MIN_RAM_HEADROOM_MB", "256")
    monkeypatch.setenv("CAI_PRIVATE_NETWORK_MODEL_MIN_PIPELINE_LAYERS_PER_NODE", "2")
    get_private_network_model_policy.cache_clear()

    topology = Topology()
    node_a = NodeId()
    node_b = NodeId()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_socket_connection(2))
    )

    node_memory = {
        node_a: create_node_memory(Memory.from_mb(800).in_bytes),
        node_b: create_node_memory(Memory.from_mb(276).in_bytes),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }
    model_card = ModelCard(
        model_id=ModelId("private/test-model"),
        storage_size=Memory.from_mb(280),
        n_layers=28,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )

    placements = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=model_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=2,
        ),
        topology,
        {},
        node_memory,
        node_network,
    )

    instance = next(iter(placements.values()))
    runner_a = instance.shard_assignments.node_to_runner[node_a]
    runner_b = instance.shard_assignments.node_to_runner[node_b]
    shard_a = instance.shard_assignments.runner_to_shard[runner_a]
    shard_b = instance.shard_assignments.runner_to_shard[runner_b]

    assert shard_a.end_layer - shard_a.start_layer == 26
    assert shard_b.end_layer - shard_b.start_layer == 2

    get_private_network_model_policy.cache_clear()


def test_private_model_rejects_secondary_node_below_minimum_layer_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAI_PRIVATE_NETWORK_MODEL_IDS", "private/test-model")
    monkeypatch.setenv("CAI_PRIVATE_NETWORK_MODEL_MIN_NODES", "2")
    monkeypatch.setenv("CAI_PRIVATE_NETWORK_MODEL_MIN_RAM_HEADROOM_MB", "256")
    monkeypatch.setenv("CAI_PRIVATE_NETWORK_MODEL_MIN_PIPELINE_LAYERS_PER_NODE", "2")
    get_private_network_model_policy.cache_clear()

    topology = Topology()
    node_a = NodeId()
    node_b = NodeId()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_socket_connection(2))
    )

    node_memory = {
        node_a: create_node_memory(Memory.from_mb(800).in_bytes),
        node_b: create_node_memory(Memory.from_mb(275).in_bytes),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }
    model_card = ModelCard(
        model_id=ModelId("private/test-model"),
        storage_size=Memory.from_mb(280),
        n_layers=28,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )

    with pytest.raises(ValueError, match="insufficient memory"):
        place_instance(
            PlaceInstance(
                command_id=CommandId(),
                model_card=model_card,
                sharding=Sharding.Pipeline,
                instance_meta=InstanceMeta.MlxRing,
                min_nodes=2,
            ),
            topology,
            {},
            node_memory,
            node_network,
        )

    get_private_network_model_policy.cache_clear()


def test_get_transition_events_no_change(instance: Instance):
    # arrange
    instance_id = InstanceId()
    current_instances = {instance_id: instance}
    target_instances = {instance_id: instance}

    # act
    events = get_transition_events(current_instances, target_instances, {})

    # assert
    assert len(events) == 0


def test_get_transition_events_create_instance(instance: Instance):
    # arrange
    instance_id = InstanceId()
    current_instances: dict[InstanceId, Instance] = {}
    target_instances: dict[InstanceId, Instance] = {instance_id: instance}

    # act
    events = get_transition_events(current_instances, target_instances, {})

    # assert
    assert len(events) == 1
    assert isinstance(events[0], InstanceCreated)


def test_get_transition_events_delete_instance(instance: Instance):
    # arrange
    instance_id = InstanceId()
    current_instances: dict[InstanceId, Instance] = {instance_id: instance}
    target_instances: dict[InstanceId, Instance] = {}

    # act
    events = get_transition_events(current_instances, target_instances, {})

    # assert
    assert len(events) == 1
    assert isinstance(events[0], InstanceDeleted)
    assert events[0].instance_id == instance_id


def test_placement_selects_leaf_nodes(
    model_card: ModelCard,
):
    # arrange
    topology = Topology()

    model_card.storage_size = Memory.from_bytes(1000)

    node_id_a = NodeId()
    node_id_b = NodeId()
    node_id_c = NodeId()
    node_id_d = NodeId()

    node_memory = {
        node_id_a: create_node_memory(500),
        node_id_b: create_node_memory(600),
        node_id_c: create_node_memory(600),
        node_id_d: create_node_memory(500),
    }
    node_network = {
        node_id_a: create_node_network(),
        node_id_b: create_node_network(),
        node_id_c: create_node_network(),
        node_id_d: create_node_network(),
    }

    topology.add_node(node_id_a)
    topology.add_node(node_id_b)
    topology.add_node(node_id_c)
    topology.add_node(node_id_d)

    # Daisy chain topology (directed)
    topology.add_connection(
        Connection(source=node_id_a, sink=node_id_b, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_id_b, sink=node_id_a, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_id_b, sink=node_id_c, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_id_c, sink=node_id_b, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_id_c, sink=node_id_d, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_id_d, sink=node_id_c, edge=create_socket_connection(1))
    )

    cic = place_instance_command(model_card=model_card)

    # act
    placements = place_instance(cic, topology, {}, node_memory, node_network)

    # assert
    assert len(placements) == 1
    instance = list(placements.values())[0]

    assigned_nodes = set(instance.shard_assignments.node_to_runner.keys())
    assert assigned_nodes == set((node_id_a, node_id_b)) or assigned_nodes == set(
        (
            node_id_c,
            node_id_d,
        )
    )


def test_tensor_rdma_backend_connectivity_matrix(
    model_card: ModelCard,
):
    # arrange
    topology = Topology()
    model_card.n_layers = 12
    model_card.storage_size = Memory.from_bytes(1500)

    node_a = NodeId()
    node_b = NodeId()
    node_c = NodeId()

    node_memory = {
        node_a: create_node_memory(500),
        node_b: create_node_memory(500),
        node_c: create_node_memory(500),
    }

    ethernet_interface = NetworkInterfaceInfo(
        name="en0",
        ip_address="10.0.0.1",
    )
    ethernet_conn = SocketConnection(
        sink_multiaddr=Multiaddr(address="/ip4/10.0.0.1/tcp/8000")
    )

    node_network = {
        node_a: NodeNetworkInfo(interfaces=[ethernet_interface]),
        node_b: NodeNetworkInfo(interfaces=[ethernet_interface]),
        node_c: NodeNetworkInfo(interfaces=[ethernet_interface]),
    }

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_node(node_c)

    # RDMA connections (directed)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_rdma_connection(3))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_rdma_connection(3))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_c, edge=create_rdma_connection(4))
    )
    topology.add_connection(
        Connection(source=node_c, sink=node_b, edge=create_rdma_connection(4))
    )
    topology.add_connection(
        Connection(source=node_a, sink=node_c, edge=create_rdma_connection(5))
    )
    topology.add_connection(
        Connection(source=node_c, sink=node_a, edge=create_rdma_connection(5))
    )

    # Ethernet connections (directed)
    topology.add_connection(Connection(source=node_a, sink=node_b, edge=ethernet_conn))
    topology.add_connection(Connection(source=node_b, sink=node_c, edge=ethernet_conn))
    topology.add_connection(Connection(source=node_c, sink=node_a, edge=ethernet_conn))
    topology.add_connection(Connection(source=node_a, sink=node_c, edge=ethernet_conn))
    topology.add_connection(Connection(source=node_b, sink=node_a, edge=ethernet_conn))
    topology.add_connection(Connection(source=node_c, sink=node_b, edge=ethernet_conn))

    cic = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=model_card,
        min_nodes=1,
    )

    # act
    placements = place_instance(cic, topology, {}, node_memory, node_network)

    # assert
    assert len(placements) == 1
    instance_id = list(placements.keys())[0]
    instance = placements[instance_id]

    assert isinstance(instance, MlxJacclInstance)

    assert instance.jaccl_devices is not None
    assert instance.jaccl_coordinators is not None

    matrix = instance.jaccl_devices
    assert len(matrix) == 3
    for i in range(3):
        assert matrix[i][i] is None

    assigned_nodes = list(instance.shard_assignments.node_to_runner.keys())
    node_to_idx = {node_id: idx for idx, node_id in enumerate(assigned_nodes)}

    idx_a = node_to_idx[node_a]
    idx_b = node_to_idx[node_b]
    idx_c = node_to_idx[node_c]

    assert matrix[idx_a][idx_b] == "rdma_en3"
    assert matrix[idx_b][idx_c] == "rdma_en4"
    assert matrix[idx_c][idx_a] == "rdma_en5"

    # Verify coordinators are set for all nodes
    assert len(instance.jaccl_coordinators) == 3
    for node_id in assigned_nodes:
        assert node_id in instance.jaccl_coordinators
        coordinator = instance.jaccl_coordinators[node_id]
        assert ":" in coordinator
        # Rank 0 node should use 0.0.0.0, others should use connection-specific IPs
        if node_id == assigned_nodes[0]:
            assert coordinator.startswith("0.0.0.0:")
        else:
            ip_part = coordinator.split(":")[0]
            assert len(ip_part.split(".")) == 4


def _make_task(
    instance_id: InstanceId,
    status: TaskStatus = TaskStatus.Running,
) -> TextGeneration:
    return TextGeneration(
        task_id=TaskId(),
        task_status=status,
        instance_id=instance_id,
        command_id=CommandId(),
        task_params=TextGenerationTaskParams(
            model=ModelId("test-model"),
            input=[InputMessage(role="user", content=InputMessageContent("hello"))],
        ),
    )


def test_get_transition_events_delete_instance_cancels_running_tasks(
    instance: Instance,
):
    # arrange
    instance_id = InstanceId()
    current_instances: dict[InstanceId, Instance] = {instance_id: instance}
    target_instances: dict[InstanceId, Instance] = {}
    task = _make_task(instance_id, TaskStatus.Running)
    tasks = {task.task_id: task}

    # act
    events = get_transition_events(current_instances, target_instances, tasks)

    # assert – cancellation event should come before the deletion event
    assert len(events) == 2
    assert isinstance(events[0], TaskStatusUpdated)
    assert events[0].task_id == task.task_id
    assert events[0].task_status == TaskStatus.Cancelled
    assert isinstance(events[1], InstanceDeleted)
    assert events[1].instance_id == instance_id


def test_get_transition_events_delete_instance_cancels_pending_tasks(
    instance: Instance,
):
    # arrange
    instance_id = InstanceId()
    current_instances: dict[InstanceId, Instance] = {instance_id: instance}
    target_instances: dict[InstanceId, Instance] = {}
    task = _make_task(instance_id, TaskStatus.Pending)
    tasks = {task.task_id: task}

    # act
    events = get_transition_events(current_instances, target_instances, tasks)

    # assert
    assert len(events) == 2
    assert isinstance(events[0], TaskStatusUpdated)
    assert events[0].task_id == task.task_id
    assert events[0].task_status == TaskStatus.Cancelled
    assert isinstance(events[1], InstanceDeleted)


def test_get_transition_events_delete_instance_ignores_completed_tasks(
    instance: Instance,
):
    # arrange
    instance_id = InstanceId()
    current_instances: dict[InstanceId, Instance] = {instance_id: instance}
    target_instances: dict[InstanceId, Instance] = {}
    tasks = {
        t.task_id: t
        for t in [
            _make_task(instance_id, TaskStatus.Complete),
            _make_task(instance_id, TaskStatus.Failed),
            _make_task(instance_id, TaskStatus.TimedOut),
            _make_task(instance_id, TaskStatus.Cancelled),
        ]
    }

    # act
    events = get_transition_events(current_instances, target_instances, tasks)

    # assert – only the InstanceDeleted event, no cancellations
    assert len(events) == 1
    assert isinstance(events[0], InstanceDeleted)


def test_get_transition_events_delete_instance_cancels_only_matching_tasks(
    instance: Instance,
):
    # arrange
    instance_id_a = InstanceId()
    instance_id_b = InstanceId()
    current_instances: dict[InstanceId, Instance] = {
        instance_id_a: instance,
        instance_id_b: instance,
    }
    # only delete instance A, keep instance B
    target_instances: dict[InstanceId, Instance] = {instance_id_b: instance}

    task_a = _make_task(instance_id_a, TaskStatus.Running)
    task_b = _make_task(instance_id_b, TaskStatus.Running)
    tasks = {task_a.task_id: task_a, task_b.task_id: task_b}

    # act
    events = get_transition_events(current_instances, target_instances, tasks)

    # assert – only task_a should be cancelled
    cancel_events = [e for e in events if isinstance(e, TaskStatusUpdated)]
    delete_events = [e for e in events if isinstance(e, InstanceDeleted)]
    assert len(cancel_events) == 1
    assert cancel_events[0].task_id == task_a.task_id
    assert cancel_events[0].task_status == TaskStatus.Cancelled
    assert len(delete_events) == 1
    assert delete_events[0].instance_id == instance_id_a


def _make_shard_metadata(model_card: ModelCard) -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=model_card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=model_card.n_layers,
        n_layers=model_card.n_layers,
    )


def test_placement_prefers_cycle_with_downloaded_model(
    model_card: ModelCard,
) -> None:
    """When two cycles are otherwise equal, prefer the one with the model already downloaded."""
    topology = Topology()

    model_card.storage_size = Memory.from_bytes(500)

    node_a = NodeId()
    node_b = NodeId()

    node_memory = {
        node_a: create_node_memory(1000),
        node_b: create_node_memory(1000),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }

    topology.add_node(node_a)
    topology.add_node(node_b)
    # No connections between them — two single-node cycles

    shard_meta = _make_shard_metadata(model_card)

    # node_b has the model fully downloaded, node_a does not
    download_status = {
        node_b: [
            DownloadCompleted(
                node_id=node_b,
                shard_metadata=shard_meta,
                total=model_card.storage_size,
            ),
        ],
    }

    cic = place_instance_command(model_card)
    placements = place_instance(
        cic, topology, {}, node_memory, node_network, download_status=download_status
    )

    assert len(placements) == 1
    instance = list(placements.values())[0]
    assigned_nodes = set(instance.shard_assignments.node_to_runner.keys())
    assert assigned_nodes == {node_b}


def test_placement_prefers_cycle_with_higher_download_progress(
    model_card: ModelCard,
) -> None:
    """When two cycles are otherwise equal, prefer the one with more download progress."""
    topology = Topology()

    model_card.storage_size = Memory.from_bytes(1000)

    node_a = NodeId()
    node_b = NodeId()

    node_memory = {
        node_a: create_node_memory(1000),
        node_b: create_node_memory(1000),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }

    topology.add_node(node_a)
    topology.add_node(node_b)

    shard_meta = _make_shard_metadata(model_card)

    # node_a: 30% downloaded, node_b: 80% downloaded
    download_status = {
        node_a: [
            DownloadOngoing(
                node_id=node_a,
                shard_metadata=shard_meta,
                download_progress=DownloadProgressData(
                    total=Memory.from_bytes(1000),
                    downloaded=Memory.from_bytes(300),
                    downloaded_this_session=Memory.from_bytes(300),
                    completed_files=0,
                    total_files=1,
                    speed=0.0,
                    eta_ms=0,
                    files={},
                ),
            ),
        ],
        node_b: [
            DownloadOngoing(
                node_id=node_b,
                shard_metadata=shard_meta,
                download_progress=DownloadProgressData(
                    total=Memory.from_bytes(1000),
                    downloaded=Memory.from_bytes(800),
                    downloaded_this_session=Memory.from_bytes(800),
                    completed_files=0,
                    total_files=1,
                    speed=0.0,
                    eta_ms=0,
                    files={},
                ),
            ),
        ],
    }

    cic = place_instance_command(model_card)
    placements = place_instance(
        cic, topology, {}, node_memory, node_network, download_status=download_status
    )

    assert len(placements) == 1
    instance = list(placements.values())[0]
    assigned_nodes = set(instance.shard_assignments.node_to_runner.keys())
    assert assigned_nodes == {node_b}


def test_placement_does_not_prefer_cycle_with_failed_download(
    model_card: ModelCard,
) -> None:
    """A failed download should count as 0% — not preferred over a node with no download history."""
    topology = Topology()

    model_card.storage_size = Memory.from_bytes(500)

    node_a = NodeId()
    node_b = NodeId()

    # node_a has slightly more RAM so it would win on the RAM tiebreaker
    node_memory = {
        node_a: create_node_memory(1001),
        node_b: create_node_memory(1000),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }

    topology.add_node(node_a)
    topology.add_node(node_b)

    shard_meta = _make_shard_metadata(model_card)

    # node_b has a failed download — should not be preferred
    download_status = {
        node_b: [
            DownloadFailed(
                node_id=node_b,
                shard_metadata=shard_meta,
                error_message="connection reset",
            ),
        ],
    }

    cic = place_instance_command(model_card)
    placements = place_instance(
        cic, topology, {}, node_memory, node_network, download_status=download_status
    )

    assert len(placements) == 1
    instance = list(placements.values())[0]
    assigned_nodes = set(instance.shard_assignments.node_to_runner.keys())
    # node_a should win on RAM tiebreaker since failed download scores 0.0
    assert assigned_nodes == {node_a}


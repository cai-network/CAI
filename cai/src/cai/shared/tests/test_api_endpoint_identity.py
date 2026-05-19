# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from cai.shared.apply import apply
from cai.shared.types.common import NodeId
from cai.shared.types.events import IndexedEvent, NodeGatheredInfo
from cai.shared.types.profiling import AdvertisedTransportEndpoint
from cai.shared.types.state import State
from cai.utils.info_gatherer.info_gatherer import (
    ApiEndpointInfo,
    StaticNodeInformation,
    WorkerStateInfo,
)


def test_api_endpoint_info_updates_node_identity() -> None:
    node_id = NodeId("node-a")
    state = apply(
        State(),
        IndexedEvent(
            idx=0,
            event=NodeGatheredInfo(
                node_id=node_id,
                when="2026-04-20T00:00:00+00:00",
                info=ApiEndpointInfo(
                    host="203.0.113.10",
                    port=18080,
                    data_host="198.51.100.20",
                    data_port=19090,
                    transport_endpoints=[
                        AdvertisedTransportEndpoint(
                            purpose="api",
                            route_type="direct",
                            host="203.0.113.10",
                            port=18080,
                            source="explicit",
                        ),
                        AdvertisedTransportEndpoint(
                            purpose="data",
                            route_type="overlay",
                            host="26.97.29.153",
                            port=19090,
                            source="interface_scan",
                            interface_name="Radmin VPN",
                        ),
                    ],
                ),
            ),
        ),
    )

    assert state.node_identities[node_id].api_host == "203.0.113.10"
    assert state.node_identities[node_id].api_port == 18080
    assert state.node_identities[node_id].data_host == "198.51.100.20"
    assert state.node_identities[node_id].data_port == 19090
    assert len(state.node_identities[node_id].transport_endpoints) == 2
    assert state.node_identities[node_id].transport_endpoints[0].route_type == "direct"
    assert state.node_identities[node_id].transport_endpoints[1].route_type == "overlay"


def test_static_node_information_updates_hardware_identity() -> None:
    node_id = NodeId("node-a")
    state = apply(
        State(),
        IndexedEvent(
            idx=0,
            event=NodeGatheredInfo(
                node_id=node_id,
                when="2026-04-27T00:00:00+00:00",
                info=StaticNodeInformation(
                    model="Custom Box",
                    chip="Ryzen",
                    os_version="Windows",
                    os_build_version="11",
                    cpu_physical_cores=8,
                    cpu_logical_cores=16,
                    total_vram_bytes=12 * 1024**3,
                ),
            ),
        ),
    )

    identity = state.node_identities[node_id]
    assert identity.model_id == "Custom Box"
    assert identity.chip_id == "Ryzen"
    assert identity.os_version == "Windows"
    assert identity.os_build_version == "11"
    assert identity.cpu_physical_cores == 8
    assert identity.cpu_logical_cores == 16
    assert identity.total_vram_bytes == 12 * 1024**3


def test_worker_state_info_updates_worker_and_relay_identity() -> None:
    node_id = NodeId("node-a")
    state = apply(
        State(),
        IndexedEvent(
            idx=0,
            event=NodeGatheredInfo(
                node_id=node_id,
                when="2026-05-01T00:00:00+00:00",
                info=WorkerStateInfo(
                    worker_enabled=True,
                    relay_enabled=True,
                    worker_reward_address="abcd1234abcd1234abcd1234abcd1234",
                ),
            ),
        ),
    )

    identity = state.node_identities[node_id]
    assert identity.worker_enabled is True
    assert identity.relay_enabled is True
    assert identity.worker_reward_address == "abcd1234abcd1234abcd1234abcd1234"


def test_worker_state_info_updates_cai_owned_transport_readiness() -> None:
    node_id = NodeId("node-a")
    state = apply(
        State(),
        IndexedEvent(
            idx=0,
            event=NodeGatheredInfo(
                node_id=node_id,
                when="2026-05-04T00:00:00+00:00",
                info=WorkerStateInfo(
                    worker_enabled=True,
                    relay_enabled=True,
                    readiness={
                        "caiOwnedTransport": {
                            "implemented": True,
                            "runtimeReady": True,
                            "status": "ready",
                        }
                    },
                ),
            ),
        ),
    )

    identity = state.node_identities[node_id]
    assert identity.readiness["caiOwnedTransport"]["implemented"] is True
    assert identity.readiness["caiOwnedTransport"]["runtimeReady"] is True
    assert identity.readiness["caiOwnedTransport"]["status"] == "ready"


def test_relay_enabled_identity_synthesizes_relay_transport_endpoints() -> None:
    node_id = NodeId("node-a")
    state = State()
    state = apply(
        state,
        IndexedEvent(
            idx=0,
            event=NodeGatheredInfo(
                node_id=node_id,
                when="2026-05-01T00:00:00+00:00",
                info=ApiEndpointInfo(
                    host="203.0.113.10",
                    port=18080,
                    data_host="198.51.100.20",
                    data_port=19090,
                    transport_endpoints=[
                        AdvertisedTransportEndpoint(
                            purpose="api",
                            route_type="direct",
                            host="203.0.113.10",
                            port=18080,
                            source="explicit",
                        ),
                        AdvertisedTransportEndpoint(
                            purpose="data",
                            route_type="overlay",
                            host="26.97.29.153",
                            port=19090,
                            source="interface_scan",
                            interface_name="Radmin VPN",
                        ),
                    ],
                ),
            ),
        ),
    )
    state = apply(
        state,
        IndexedEvent(
            idx=1,
            event=NodeGatheredInfo(
                node_id=node_id,
                when="2026-05-01T00:00:01+00:00",
                info=WorkerStateInfo(
                    worker_enabled=False,
                    relay_enabled=True,
                ),
            ),
        ),
    )

    identity = state.node_identities[node_id]
    route_types = [endpoint.route_type for endpoint in identity.transport_endpoints]
    assert route_types.count("relay") == 2
    assert route_types.count("direct") == 1
    assert route_types.count("overlay") == 1


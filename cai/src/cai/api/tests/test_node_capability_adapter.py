# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from cai.api.node_capability_adapter import (
    capability_record_node_identity,
    capability_record_node_memory,
    capability_record_route_peers,
    worker_identity_state,
)
from cai.shared.types.common import NodeId
from cai_compute_chain.node_capabilities import NodeCapabilityRecord


def _record(**overrides) -> NodeCapabilityRecord:
    data = {
        "node_id": "node-1",
        "source": "test",
        "source_url": None,
        "last_seen_at": "2026-05-23T00:00:00+00:00",
        "updated_at": "2026-05-23T00:00:00+00:00",
    }
    data.update(overrides)
    return NodeCapabilityRecord(**data)


def test_worker_identity_state_accepts_mapping_and_object_shapes() -> None:
    assert worker_identity_state(
        {"workerEnabled": True, "workerRewardAddress": " CAI123 "}
    ) == (True, "CAI123")

    class Identity:
        worker_enabled = False
        worker_reward_address = ""

    assert worker_identity_state(Identity()) == (False, None)


def test_capability_record_node_memory_reads_nested_and_flat_resource_values() -> None:
    memory = capability_record_node_memory(
        _record(
            resource_summary={
                "ramTotalBytes": {"bytes": 4096},
                "ramAvailableBytes": 2048,
                "swapTotalBytes": 1024,
            },
        )
    )

    assert memory is not None
    assert memory.ram_total.in_bytes == 4096
    assert memory.ram_available.in_bytes == 2048
    assert memory.swap_total.in_bytes == 1024
    assert memory.swap_available.in_bytes == 1024


def test_capability_record_node_identity_builds_transport_endpoints() -> None:
    identity = capability_record_node_identity(
        _record(
            friendly_name="Worker One",
            worker_enabled=True,
            relay_enabled=True,
            worker_reward_address="CAI123",
            api_urls=["http://worker.example:52415/v1/cai/summary"],
            data_endpoints=[
                {
                    "purpose": "data",
                    "routeType": "direct",
                    "host": "worker-data.example",
                    "port": 47001,
                    "source": "explicit",
                }
            ],
            resource_summary={
                "totalVramBytes": 1024,
                "cpuPhysicalCores": 4,
                "cpuLogicalCores": 8,
            },
        )
    )

    assert identity.friendly_name == "Worker One"
    assert identity.api_host == "worker.example"
    assert identity.api_port == 52415
    assert identity.data_host == "worker-data.example"
    assert identity.data_port == 47001
    assert identity.total_vram_bytes == 1024
    assert identity.worker_enabled is True


def test_capability_record_route_peers_normalizes_list_and_mapping_hints() -> None:
    peers = capability_record_route_peers(
        _record(
            route_hints={
                "overlayPeerIds": ["node-2", ""],
                "direct_peer_ids": {"node-3": {"ok": True}},
            },
        )
    )

    assert peers == {NodeId("node-2"), NodeId("node-3")}

# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from cai.api.dashboard_state import (
    build_dashboard_state,
    sanitize_dashboard_state_payload,
)
from cai.shared.topology import Topology
from cai.shared.types.common import NodeId
from cai.shared.types.multiaddr import Multiaddr
from cai.shared.types.profiling import MemoryUsage, NodeIdentity
from cai.shared.types.topology import Connection, SocketConnection
from cai.shared.types.state import State
from datetime import datetime, timedelta, timezone


def _socket_connection(port: int) -> SocketConnection:
    return SocketConnection(
        sink_multiaddr=Multiaddr(address=f"/ip4/198.51.100.10/tcp/{port}")
    )


def _runtime_ready_proof() -> dict[str, object]:
    return {
        "verified": True,
        "proofVerified": True,
        "executorNodeIds": ["node-a", "node-b"],
        "executorCount": 2,
        "hasFinalOutput": True,
    }


def _production_llm_self_test() -> dict[str, object]:
    return {
        "contractReady": True,
        "productionReady": True,
        "generationProbeReady": True,
        "backendHealthReady": True,
    }


def test_sanitize_dashboard_state_payload_keeps_full_topology() -> None:
    payload = {
        "topology": {
            "nodes": ["node-local", "node-remote"],
            "connections": {
                "node-local": {"node-remote": [{"sinkMultiaddr": {"address": "/ip4/1.2.3.4/tcp/1234"}}]},
                "node-remote": {"node-local": [{"sinkMultiaddr": {"address": "/ip4/5.6.7.8/tcp/5678"}}]},
            },
        },
        "overlayPeers": {
            "node-local": ["node-remote"],
            "node-remote": ["node-local"],
        },
        "nodeIdentities": {
            "node-local": {"friendlyName": "Local"},
            "node-remote": {"friendlyName": "Remote"},
        },
        "nodeMemory": {
            "node-local": {"ramTotal": {"inBytes": 10}, "ramAvailable": {"inBytes": 5}},
            "node-remote": {"ramTotal": {"inBytes": 20}, "ramAvailable": {"inBytes": 10}},
        },
        "downloads": {
            "node-local": [{"nodeId": "node-local"}],
            "node-remote": [{"nodeId": "node-remote"}],
        },
        "instances": {
            "instance-1": {
                "MlxRingInstance": {
                    "instanceId": "instance-1",
                    "hostsByNode": {
                        "node-local": ["127.0.0.1"],
                        "node-remote": ["10.0.0.2"],
                    },
                    "ephemeralPort": 8000,
                    "shardAssignments": {
                        "modelId": "test-model",
                        "nodeToRunner": {
                            "node-local": "runner-local",
                            "node-remote": "runner-remote",
                        },
                        "runnerToShard": {
                            "runner-local": {"PipelineShardMetadata": {"startLayer": 0, "endLayer": 2}},
                            "runner-remote": {"PipelineShardMetadata": {"startLayer": 2, "endLayer": 4}},
                        },
                    },
                }
            }
        },
        "runners": {
            "runner-local": {"RunnerReady": {}},
            "runner-remote": {"RunnerReady": {}},
        },
        "thunderboltBridgeCycles": [["node-local", "node-remote"]],
    }

    sanitized = sanitize_dashboard_state_payload(payload, local_node_id="node-local")

    assert sanitized["topology"] == payload["topology"]
    assert sanitized["overlayPeers"] == {
        "node-local": ["node-remote"],
        "node-remote": ["node-local"],
    }
    assert sanitized["nodeIdentities"] == {
        "node-local": {"friendlyName": "Local"},
        "node-remote": {"friendlyName": "Remote"},
    }
    assert sanitized["nodeMemory"] == {
        "node-local": {"ramTotal": {"inBytes": 10}, "ramAvailable": {"inBytes": 5}},
        "node-remote": {"ramTotal": {"inBytes": 20}, "ramAvailable": {"inBytes": 10}},
    }
    assert sanitized["downloads"] == {
        "node-local": [{"nodeId": "node-local"}],
        "node-remote": [{"nodeId": "node-remote"}],
    }
    assert sanitized["thunderboltBridgeCycles"] == [["node-local", "node-remote"]]
    assert sanitized["runners"] == {
        "runner-local": {"RunnerReady": {}},
        "runner-remote": {"RunnerReady": {}},
    }
    assert sanitized["currentNodeId"] == "node-local"

    instance = sanitized["instances"]["instance-1"]["MlxRingInstance"]
    assert instance["hostsByNode"] == {
        "node-local": ["127.0.0.1"],
        "node-remote": ["10.0.0.2"],
    }
    assert instance["shardAssignments"]["nodeToRunner"] == {
        "node-local": "runner-local",
        "node-remote": "runner-remote",
    }
    assert instance["shardAssignments"]["runnerToShard"] == {
        "runner-local": {"PipelineShardMetadata": {"startLayer": 0, "endLayer": 2}},
        "runner-remote": {"PipelineShardMetadata": {"startLayer": 2, "endLayer": 4}},
    }


def test_sanitize_dashboard_state_payload_keeps_remote_only_instances() -> None:
    payload = {
        "topology": {"nodes": ["node-local", "node-remote"], "connections": {}},
        "instances": {
            "instance-remote": {
                "MlxRingInstance": {
                    "instanceId": "instance-remote",
                    "shardAssignments": {
                        "modelId": "test-model",
                        "nodeToRunner": {"node-remote": "runner-remote"},
                        "runnerToShard": {"runner-remote": {"PipelineShardMetadata": {"startLayer": 0, "endLayer": 4}}},
                    },
                }
            }
        },
        "runners": {"runner-remote": {"RunnerReady": {}}},
    }

    sanitized = sanitize_dashboard_state_payload(payload, local_node_id="node-local")

    assert "instance-remote" in sanitized["instances"]
    assert sanitized["runners"] == {"runner-remote": {"RunnerReady": {}}}


def test_sanitize_dashboard_state_payload_keeps_remote_node_without_inventing_local() -> None:
    payload = {
        "topology": {
            "nodes": ["node-remote"],
            "connections": {},
        },
        "nodeIdentities": {
            "node-remote": {"friendlyName": "Remote"},
        },
        "nodeMemory": {
            "node-remote": {"ramTotal": {"inBytes": 20}, "ramAvailable": {"inBytes": 10}},
        },
        "downloads": {
            "node-remote": [{"nodeId": "node-remote"}],
        },
    }

    sanitized = sanitize_dashboard_state_payload(payload, local_node_id="node-local")

    assert sanitized["topology"] == {"nodes": ["node-remote"], "connections": {}}
    assert sanitized["nodeIdentities"] == {"node-remote": {"friendlyName": "Remote"}}
    assert sanitized["nodeMemory"] == {
        "node-remote": {"ramTotal": {"inBytes": 20}, "ramAvailable": {"inBytes": 10}}
    }
    assert sanitized["downloads"] == {"node-remote": [{"nodeId": "node-remote"}]}
    assert "node-local" not in sanitized["topology"]["nodes"]
    assert sanitized["currentNodeId"] == "node-local"


def test_sanitize_dashboard_state_payload_adds_overlay_only_nodes_to_topology() -> None:
    payload = {
        "overlayPeers": {
            "node-local": ["node-remote"],
            "node-remote": ["node-local"],
        },
        "nodeIdentities": {
            "node-local": {"friendlyName": "Local"},
            "node-remote": {"friendlyName": "Remote"},
        },
    }

    sanitized = sanitize_dashboard_state_payload(payload, local_node_id="node-local")

    assert sanitized["topology"] == {
        "nodes": ["node-local", "node-remote"],
        "connections": {},
    }
    assert sanitized["currentNodeId"] == "node-local"


def test_sanitize_dashboard_state_payload_does_not_add_identity_only_nodes_to_topology() -> None:
    payload = {
        "topology": {
            "nodes": ["node-local", "node-remote"],
            "connections": {"node-local": {"node-remote": []}},
        },
        "nodeIdentities": {
            "node-local": {"friendlyName": "Local"},
            "node-remote": {"friendlyName": "Remote"},
            "node-stale-local": {"friendlyName": "Local"},
        },
        "lastSeen": {
            "node-local": 1,
            "node-remote": 1,
            "node-stale-local": 1,
        },
        "nodeMemory": {
            "node-local": {"ramTotal": {"inBytes": 10}, "ramAvailable": {"inBytes": 5}},
            "node-remote": {"ramTotal": {"inBytes": 10}, "ramAvailable": {"inBytes": 5}},
            "node-stale-local": {
                "ramTotal": {"inBytes": 10},
                "ramAvailable": {"inBytes": 5},
            },
        },
    }

    sanitized = sanitize_dashboard_state_payload(payload, local_node_id="node-local")

    assert sanitized["topology"] == {
        "nodes": ["node-local", "node-remote"],
        "connections": {"node-local": {"node-remote": []}},
    }
    assert "node-stale-local" in sanitized["nodeIdentities"]
    assert "node-stale-local" not in sanitized["topology"]["nodes"]


def test_sanitize_dashboard_state_payload_hides_stale_nodes_from_topology() -> None:
    stale_seen = (datetime.now(tz=timezone.utc) - timedelta(hours=2)).isoformat()
    fresh_seen = datetime.now(tz=timezone.utc).isoformat()
    payload = {
        "topology": {
            "nodes": ["node-local", "node-fresh", "node-stale"],
            "connections": {},
        },
        "nodeIdentities": {
            "node-local": {"friendlyName": "Local"},
            "node-fresh": {"friendlyName": "Fresh"},
            "node-stale": {"friendlyName": "Stale"},
        },
        "lastSeen": {
            "node-local": fresh_seen,
            "node-fresh": fresh_seen,
            "node-stale": stale_seen,
        },
    }

    sanitized = sanitize_dashboard_state_payload(payload, local_node_id="node-local")

    assert sanitized["topology"] == {
        "nodes": ["node-local", "node-fresh"],
        "connections": {},
    }
    assert "node-stale" not in sanitized["nodeIdentities"]


def test_build_dashboard_state_aggregates_network_resources() -> None:
    topology = Topology()
    topology.add_node(NodeId("node-local"))
    topology.add_node(NodeId("node-remote"))

    state = State(
        topology=topology,
        overlay_peers={NodeId("node-local"): [NodeId("node-remote")]},
        node_memory={
            NodeId("node-local"): MemoryUsage.from_bytes(
                ram_total=32 * 1024**3,
                ram_available=20 * 1024**3,
                swap_total=0,
                swap_available=0,
            ),
            NodeId("node-remote"): MemoryUsage.from_bytes(
                ram_total=64 * 1024**3,
                ram_available=40 * 1024**3,
                swap_total=0,
                swap_available=0,
            ),
        },
        node_identities={
            NodeId("node-local"): NodeIdentity(
                cpu_physical_cores=8,
                cpu_logical_cores=16,
                total_vram_bytes=12 * 1024**3,
                worker_enabled=True,
            ),
            NodeId("node-remote"): NodeIdentity(
                cpu_physical_cores=24,
                cpu_logical_cores=32,
                total_vram_bytes=48 * 1024**3,
                worker_enabled=True,
            ),
        },
    )

    payload = build_dashboard_state(state, NodeId("node-local"))

    assert payload["networkSummary"] == {
        "knownNodes": 2,
        "knownWorkers": 2,
        "knownRelays": 0,
        "knownConnections": 0,
        "localOverlayPeers": 1,
        "totalRamBytes": 96 * 1024**3,
        "totalAvailableRamBytes": 60 * 1024**3,
        "totalVramBytes": 60 * 1024**3,
        "totalCpuCores": 32,
        "workerTotalRamBytes": 96 * 1024**3,
        "workerTotalAvailableRamBytes": 60 * 1024**3,
        "workerTotalVramBytes": 60 * 1024**3,
        "workerTotalCpuCores": 32,
        "workerDirectSocketLinks": 0,
        "workerOverlayLinks": 1,
        "llamaCppLargestDirectWorkerCycle": 0,
        "llamaCppRelayCoordinatorCandidateCount": 0,
        "llamaCppRelayRouteCandidateCount": 0,
        "llamaCppDistributedReady": False,
        "llamaCppDistributedReason": (
            "Workers are visible through overlay, but distributed llama.cpp still "
            "needs an executable direct or relay coordinator-to-worker path."
        ),
        "caiOwnedTransportReadiness": {
            "protocol": None,
            "status": "planned",
            "ready": False,
            "runtimeReady": False,
            "reason": (
                "Worker nodes have not advertised CAI-owned transport readiness yet."
            ),
            "workerCount": 2,
            "runtimeReadyWorkerCount": 0,
            "implementedWorkerCount": 0,
            "contractReadyWorkerCount": 0,
            "productionReadyWorkerCount": 0,
            "backendHealthFailedWorkerCount": 0,
            "failedWorkerCount": 0,
            "missingWorkerCount": 2,
            "observedStatuses": ["planned"],
            "runtimeReadyWorkerIds": [],
            "implementedWorkerIds": [],
            "contractReadyWorkerIds": [],
            "productionReadyWorkerIds": [],
            "backendHealthFailedWorkerIds": [],
            "failedWorkerIds": [],
            "missingWorkerIds": ["node-local", "node-remote"],
        },
    }


def test_build_dashboard_state_reports_cai_owned_transport_readiness() -> None:
    state = State(
        node_identities={
            NodeId("node-a"): NodeIdentity(
                worker_enabled=True,
                readiness={
                    "caiOwnedTransport": {
                        "protocol": "cai-owned-activation-batch-v1",
                        "implemented": True,
                        "runtimeReady": False,
                        "status": "test_adapter_ready",
                    }
                },
            ),
            NodeId("node-b"): NodeIdentity(
                worker_enabled=True,
                readiness={
                    "caiOwnedTransport": {
                        "protocol": "cai-owned-activation-batch-v1",
                        "implemented": True,
                        "runtimeReady": False,
                        "status": "test_adapter_ready",
                    }
                },
            ),
        },
    )

    payload = build_dashboard_state(state, NodeId("node-a"))
    readiness = payload["networkSummary"]["caiOwnedTransportReadiness"]

    assert readiness["status"] == "test_adapter_ready"
    assert readiness["ready"] is False
    assert readiness["runtimeReady"] is False
    assert readiness["implementedWorkerCount"] == 2
    assert readiness["runtimeReadyWorkerCount"] == 0
    assert readiness["contractReadyWorkerCount"] == 0
    assert readiness["productionReadyWorkerCount"] == 0
    assert readiness["observedStatuses"] == ["test_adapter_ready"]


def test_build_dashboard_state_falls_back_to_visible_workers_when_selection_empty() -> None:
    state = State(
        node_identities={
            NodeId("node-a"): NodeIdentity(
                worker_enabled=True,
                readiness={
                    "caiOwnedTransport": {
                        "protocol": "cai-owned-llm-shard-transport",
                        "implemented": True,
                        "runtimeReady": False,
                        "status": "failed",
                    }
                },
            )
        },
    )

    payload = build_dashboard_state(state, NodeId("node-local"), worker_node_ids=set())
    readiness = payload["networkSummary"]["caiOwnedTransportReadiness"]

    assert payload["networkSummary"]["knownWorkers"] == 1
    assert readiness["status"] == "failed"
    assert readiness["workerCount"] == 1
    assert readiness["implementedWorkerCount"] == 1
    assert readiness["failedWorkerIds"] == ["node-a"]


def test_build_dashboard_state_reports_cai_owned_llm_shard_self_test_counts() -> None:
    state = State(
        node_identities={
            NodeId("node-a"): NodeIdentity(
                worker_enabled=True,
                readiness={
                    "caiOwnedTransport": {
                        "implemented": True,
                        "runtimeReady": False,
                        "status": "test_adapter_ready",
                        "llmShardSelfTest": {
                            "contractReady": True,
                            "productionReady": False,
                            "patchBoundaryVerified": True,
                            "patchBoundaryPatchId": (
                                "cai-llama-cpp-shard-smoke-runner"
                            ),
                        },
                    }
                },
            ),
            NodeId("node-b"): NodeIdentity(
                worker_enabled=True,
                readiness={
                    "caiOwnedTransport": {
                        "implemented": True,
                        "runtimeReady": False,
                        "status": "test_adapter_ready",
                        "llmShardSelfTest": {
                            "contractReady": True,
                            "productionReady": False,
                            "patchBoundaryVerified": True,
                        },
                    }
                },
            ),
        },
    )

    payload = build_dashboard_state(state, NodeId("node-a"))
    readiness = payload["networkSummary"]["caiOwnedTransportReadiness"]

    assert readiness["status"] == "test_adapter_ready"
    assert readiness["contractReadyWorkerCount"] == 2
    assert readiness["productionReadyWorkerCount"] == 0
    assert readiness["contractReadyWorkerIds"] == ["node-a", "node-b"]
    assert readiness["ready"] is False
    assert "contract self-test" in readiness["reason"]


def test_build_dashboard_state_blocks_degraded_llm_shard_backend_health() -> None:
    state = State(
        node_identities={
            NodeId("node-a"): NodeIdentity(
                worker_enabled=True,
                readiness={
                    "caiOwnedTransport": {
                        "implemented": True,
                        "runtimeReady": True,
                        "runtimeReadyProof": _runtime_ready_proof(),
                        "status": "ready",
                        "llmShardSelfTest": {
                            "contractReady": True,
                            "productionReady": True,
                            "generationProbeReady": True,
                            "backendHealthReady": False,
                            "backendHealth": {"status": "degraded"},
                        },
                    }
                },
            ),
            NodeId("node-b"): NodeIdentity(
                worker_enabled=True,
                readiness={
                    "caiOwnedTransport": {
                        "implemented": True,
                        "runtimeReady": True,
                        "runtimeReadyProof": _runtime_ready_proof(),
                        "status": "ready",
                        "llmShardSelfTest": {
                            "contractReady": True,
                            "productionReady": True,
                            "generationProbeReady": True,
                            "backendHealthReady": True,
                        },
                    }
                },
            ),
        },
    )

    payload = build_dashboard_state(state, NodeId("node-a"))
    readiness = payload["networkSummary"]["caiOwnedTransportReadiness"]

    assert readiness["status"] == "failed"
    assert readiness["runtimeReady"] is False
    assert readiness["runtimeReadyWorkerIds"] == ["node-b"]
    assert readiness["productionReadyWorkerIds"] == ["node-b"]
    assert readiness["backendHealthFailedWorkerIds"] == ["node-a"]
    assert "backend health" in readiness["reason"]


def test_build_dashboard_state_requires_generation_probe_for_production_count() -> None:
    state = State(
        node_identities={
            NodeId("node-a"): NodeIdentity(
                worker_enabled=True,
                readiness={
                    "caiOwnedTransport": {
                        "implemented": True,
                        "runtimeReady": True,
                        "runtimeReadyProof": _runtime_ready_proof(),
                        "status": "ready",
                        "llmShardSelfTest": {
                            "contractReady": True,
                            "productionReady": True,
                            "backendHealthReady": True,
                        },
                    }
                },
            )
        },
    )

    payload = build_dashboard_state(state, NodeId("node-a"))
    readiness = payload["networkSummary"]["caiOwnedTransportReadiness"]

    assert readiness["productionReadyWorkerCount"] == 0
    assert readiness["productionReadyWorkerIds"] == []


def test_build_dashboard_state_does_not_mark_single_cai_owned_runtime_ready() -> None:
    state = State(
        node_identities={
            NodeId("node-a"): NodeIdentity(
                worker_enabled=True,
                readiness={
                    "caiOwnedTransport": {
                        "implemented": True,
                        "runtimeReady": True,
                        "runtimeReadyProof": _runtime_ready_proof(),
                        "llmShardSelfTest": _production_llm_self_test(),
                        "status": "ready",
                    }
                },
            ),
        },
    )

    payload = build_dashboard_state(state, NodeId("node-a"))
    readiness = payload["networkSummary"]["caiOwnedTransportReadiness"]

    assert readiness["status"] == "test_adapter_ready"
    assert readiness["ready"] is False
    assert readiness["runtimeReady"] is False
    assert readiness["runtimeReadyWorkerCount"] == 1


def test_build_dashboard_state_requires_live_proof_for_runtime_count() -> None:
    state = State(
        node_identities={
            NodeId("node-a"): NodeIdentity(
                worker_enabled=True,
                readiness={
                    "caiOwnedTransport": {
                        "implemented": True,
                        "runtimeReady": True,
                        "status": "ready",
                    }
                },
            )
        },
    )

    payload = build_dashboard_state(state, NodeId("node-a"))
    readiness = payload["networkSummary"]["caiOwnedTransportReadiness"]

    assert readiness["status"] == "test_adapter_ready"
    assert readiness["runtimeReadyWorkerCount"] == 0
    assert readiness["runtimeReadyWorkerIds"] == []


def test_build_dashboard_state_marks_cai_owned_transport_ready_for_workers() -> None:
    state = State(
        node_identities={
            NodeId("node-a"): NodeIdentity(
                worker_enabled=True,
                readiness={
                    "caiOwnedTransport": {
                        "implemented": True,
                        "runtimeReady": True,
                        "runtimeReadyProof": _runtime_ready_proof(),
                        "llmShardSelfTest": _production_llm_self_test(),
                        "status": "ready",
                    }
                },
            ),
            NodeId("node-b"): NodeIdentity(
                worker_enabled=True,
                readiness={
                    "caiOwnedTransport": {
                        "implemented": True,
                        "runtimeReady": True,
                        "runtimeReadyProof": _runtime_ready_proof(),
                        "llmShardSelfTest": _production_llm_self_test(),
                        "status": "ready",
                    }
                },
            ),
        },
    )

    payload = build_dashboard_state(state, NodeId("node-a"))
    readiness = payload["networkSummary"]["caiOwnedTransportReadiness"]

    assert readiness["status"] == "ready"
    assert readiness["ready"] is True
    assert readiness["runtimeReady"] is True
    assert readiness["runtimeReadyWorkerCount"] == 2
    assert readiness["reason"] is None


def test_build_dashboard_state_counts_known_relays() -> None:
    topology = Topology()
    topology.add_node(NodeId("node-relay"))
    topology.add_node(NodeId("node-worker"))

    state = State(
        topology=topology,
        node_identities={
            NodeId("node-relay"): NodeIdentity(
                cpu_physical_cores=4,
                relay_enabled=True,
                worker_enabled=False,
            ),
            NodeId("node-worker"): NodeIdentity(
                cpu_physical_cores=8,
                relay_enabled=True,
                worker_enabled=True,
            ),
        },
    )

    payload = build_dashboard_state(state, NodeId("node-relay"))

    assert payload["networkSummary"]["knownRelays"] == 2


def test_build_dashboard_state_aggregates_worker_enabled_resources_only() -> None:
    topology = Topology()
    topology.add_node(NodeId("node-worker"))
    topology.add_node(NodeId("node-validator"))

    state = State(
        topology=topology,
        node_memory={
            NodeId("node-worker"): MemoryUsage.from_bytes(
                ram_total=24 * 1024**3,
                ram_available=12 * 1024**3,
                swap_total=0,
                swap_available=0,
            ),
            NodeId("node-validator"): MemoryUsage.from_bytes(
                ram_total=128 * 1024**3,
                ram_available=96 * 1024**3,
                swap_total=0,
                swap_available=0,
            ),
        },
        node_identities={
            NodeId("node-worker"): NodeIdentity(
                cpu_physical_cores=12,
                total_vram_bytes=16 * 1024**3,
                worker_enabled=True,
            ),
            NodeId("node-validator"): NodeIdentity(
                cpu_physical_cores=48,
                total_vram_bytes=80 * 1024**3,
                worker_enabled=False,
            ),
        },
    )

    payload = build_dashboard_state(state, NodeId("node-worker"))

    assert payload["networkSummary"]["knownNodes"] == 2
    assert payload["networkSummary"]["knownWorkers"] == 1
    assert payload["networkSummary"]["workerTotalRamBytes"] == 24 * 1024**3
    assert payload["networkSummary"]["workerTotalAvailableRamBytes"] == 12 * 1024**3
    assert payload["networkSummary"]["workerTotalVramBytes"] == 16 * 1024**3
    assert payload["networkSummary"]["workerTotalCpuCores"] == 12
    assert payload["networkSummary"]["totalRamBytes"] == 152 * 1024**3


def test_build_dashboard_state_counts_overlay_only_nodes_in_network_summary() -> None:
    state = State(
        overlay_peers={NodeId("node-local"): [NodeId("node-remote")]},
        node_identities={
            NodeId("node-local"): NodeIdentity(
                cpu_physical_cores=8,
                worker_enabled=True,
            ),
            NodeId("node-remote"): NodeIdentity(
                cpu_physical_cores=4,
                worker_enabled=True,
            ),
        },
    )

    payload = build_dashboard_state(state, NodeId("node-local"))

    assert payload["topology"] == {
        "nodes": ["node-local", "node-remote"],
        "connections": {},
    }
    assert payload["networkSummary"]["knownNodes"] == 2
    assert payload["networkSummary"]["localOverlayPeers"] == 1
    assert payload["networkSummary"]["llamaCppDistributedReady"] is False
    assert payload["networkSummary"]["workerOverlayLinks"] == 1
    assert payload["networkSummary"]["workerDirectSocketLinks"] == 0
    assert payload["networkSummary"]["llamaCppRelayCoordinatorCandidateCount"] == 0
    assert payload["networkSummary"]["llamaCppRelayRouteCandidateCount"] == 0


def test_build_dashboard_state_reports_distributed_llama_cpp_ready_for_direct_worker_cycle() -> None:
    topology = Topology()
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=_socket_connection(52431))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=_socket_connection(52432))
    )

    state = State(
        topology=topology,
        overlay_peers={node_a: [node_b]},
        node_identities={
            node_a: NodeIdentity(worker_enabled=True),
            node_b: NodeIdentity(worker_enabled=True),
        },
    )

    payload = build_dashboard_state(
        state,
        node_a,
        worker_node_ids={node_a, node_b},
    )

    assert payload["networkSummary"]["workerDirectSocketLinks"] == 1
    assert payload["networkSummary"]["workerOverlayLinks"] == 1
    assert payload["networkSummary"]["llamaCppLargestDirectWorkerCycle"] == 2
    assert payload["networkSummary"]["llamaCppDistributedReady"] is True
    assert payload["networkSummary"]["llamaCppDistributedReason"] is None
    assert payload["networkSummary"]["llamaCppRelayCoordinatorCandidateCount"] == 0
    assert payload["networkSummary"]["llamaCppRelayRouteCandidateCount"] == 0


def test_build_dashboard_state_reports_distributed_llama_cpp_ready_for_direct_coordinator_fanout() -> None:
    topology = Topology()
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=_socket_connection(52431))
    )

    state = State(
        topology=topology,
        overlay_peers={node_a: [node_b]},
        node_identities={
            node_a: NodeIdentity(worker_enabled=True),
            node_b: NodeIdentity(worker_enabled=True),
        },
    )

    payload = build_dashboard_state(
        state,
        node_a,
        worker_node_ids={node_a, node_b},
    )

    assert payload["networkSummary"]["workerDirectSocketLinks"] == 1
    assert payload["networkSummary"]["llamaCppLargestDirectWorkerCycle"] == 0
    assert payload["networkSummary"]["llamaCppDistributedReady"] is True
    assert payload["networkSummary"]["llamaCppDistributedReason"] is None
    assert payload["networkSummary"]["llamaCppRelayCoordinatorCandidateCount"] == 0
    assert payload["networkSummary"]["llamaCppRelayRouteCandidateCount"] == 0


def test_build_dashboard_state_reports_relay_coordinator_fanout_ready() -> None:
    topology = Topology()
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    node_relay = NodeId("node-relay")
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_node(node_relay)

    state = State(
        topology=topology,
        overlay_peers={
            node_a: [node_relay],
            node_relay: [node_a, node_b],
            node_b: [node_relay],
        },
        node_identities={
            node_a: NodeIdentity(worker_enabled=True),
            node_b: NodeIdentity(worker_enabled=True),
            node_relay: NodeIdentity(relay_enabled=True, worker_enabled=False),
        },
    )

    payload = build_dashboard_state(
        state,
        node_a,
        worker_node_ids={node_a, node_b},
    )

    assert payload["networkSummary"]["llamaCppDistributedReady"] is True
    assert payload["networkSummary"]["llamaCppRelayCoordinatorCandidateCount"] == 2
    assert payload["networkSummary"]["llamaCppRelayRouteCandidateCount"] == 2
    assert payload["networkSummary"]["llamaCppDistributedReason"] is None


def test_build_dashboard_state_keeps_empty_worker_scope_empty() -> None:
    topology = Topology()
    topology.add_node(NodeId("node-a"))
    topology.add_node(NodeId("node-b"))

    payload = build_dashboard_state(
        State(topology=topology),
        NodeId("node-a"),
        worker_node_ids=set(),
    )

    assert payload["networkSummary"]["knownWorkers"] == 0
    assert payload["networkSummary"]["workerDirectSocketLinks"] == 0
    assert payload["networkSummary"]["workerOverlayLinks"] == 0
    assert payload["networkSummary"]["llamaCppDistributedReady"] is False
    assert payload["networkSummary"]["llamaCppDistributedReason"] == (
        "Need at least 2 worker-enabled nodes for distributed llama.cpp."
    )


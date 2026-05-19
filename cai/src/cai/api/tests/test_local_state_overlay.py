# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from cai.api.main import API
from cai.shared.apply import apply
from cai.shared.topology import Topology
from cai.shared.types.common import NodeId
from cai.shared.types.events import IndexedEvent, NodeTimedOut
from cai.shared.types.profiling import MemoryUsage, NodeIdentity
from cai.shared.types.state import State


def _make_api() -> API:
    api = object.__new__(API)
    api.node_id = NodeId("node-local")
    api._local_state_cache = State()
    return api


def test_overlay_local_node_state_restores_cached_identity_after_timeout() -> None:
    api = _make_api()
    node_id = NodeId("node-local")
    topology = Topology()
    topology.add_node(node_id)
    initial_state = State(
        topology=topology,
        node_identities={
            node_id: NodeIdentity(cpu_physical_cores=8, cpu_logical_cores=16)
        },
        node_memory={
            node_id: MemoryUsage.from_bytes(
                ram_total=32 * 1024**3,
                ram_available=24 * 1024**3,
                swap_total=0,
                swap_available=0,
            )
        },
    )

    api._cache_local_node_state_from(initial_state)
    timed_out_state = apply(
        initial_state,
        IndexedEvent(idx=0, event=NodeTimedOut(node_id=node_id)),
    )
    restored_state = api._overlay_local_node_state(timed_out_state)

    assert node_id in restored_state.node_identities
    assert node_id in restored_state.node_memory
    assert node_id in set(restored_state.topology.list_nodes())


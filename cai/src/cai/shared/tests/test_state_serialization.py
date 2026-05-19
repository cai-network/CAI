# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from cai.shared.types.common import NodeId
from cai.shared.types.multiaddr import Multiaddr
from cai.shared.types.state import State
from cai.shared.types.topology import Connection, SocketConnection


def test_state_serialization_roundtrip() -> None:
    """Verify that State → JSON → State round-trip preserves topology."""

    # --- build a simple state ------------------------------------------------
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")

    connection = Connection(
        source=node_a,
        sink=node_b,
        edge=SocketConnection(
            sink_multiaddr=Multiaddr(address="/ip4/127.0.0.1/tcp/10001"),
        ),
    )

    state = State()
    state.topology.add_connection(connection)
    state = state.model_copy(
        update={
            "overlay_peers": {
                node_a: [node_b],
                node_b: [node_a],
            },
            "overlay_advertised_peers": {
                node_a: [Multiaddr(address="/ip4/10.0.0.1/tcp/52416/p2p/nodea123")],
            },
        }
    )

    json_repr = state.model_dump_json()
    restored_state = State.model_validate_json(json_repr)

    assert (
        state.topology.to_snapshot().nodes
        == restored_state.topology.to_snapshot().nodes
    )
    assert set(state.topology.to_snapshot().connections) == set(
        restored_state.topology.to_snapshot().connections
    )
    assert restored_state.overlay_peers == state.overlay_peers
    assert restored_state.overlay_advertised_peers == state.overlay_advertised_peers
    assert restored_state.model_dump_json() == json_repr


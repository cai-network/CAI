# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from cai.shared.apply import apply
from cai.shared.types.common import NodeId
from cai.shared.types.events import (
    IndexedEvent,
    OverlayBootstrapPeersAdvertised,
    OverlayPeerConnected,
    OverlayPeerDisconnected,
)
from cai.shared.types.multiaddr import Multiaddr
from cai.shared.types.state import State


def test_overlay_peer_connection_roundtrip() -> None:
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")

    state = State()
    state = apply(
        state,
        IndexedEvent(
            idx=0,
            event=OverlayPeerConnected(
                local_node_id=node_a,
                remote_node_id=node_b,
            ),
        ),
    )

    assert state.overlay_peers == {
        node_a: [node_b],
        node_b: [node_a],
    }

    state = apply(
        state,
        IndexedEvent(
            idx=1,
            event=OverlayPeerDisconnected(
                local_node_id=node_a,
                remote_node_id=node_b,
            ),
        ),
    )

    assert state.overlay_peers == {}


def test_overlay_bootstrap_peers_advertisement_roundtrip() -> None:
    node_a = NodeId("node-a")
    advertised_peer = Multiaddr(address="/ip4/203.0.113.10/tcp/52416/p2p/nodea123")

    state = apply(
        State(),
        IndexedEvent(
            idx=0,
            event=OverlayBootstrapPeersAdvertised(
                node_id=node_a,
                peers=[advertised_peer],
            ),
        ),
    )

    assert state.overlay_advertised_peers == {
        node_a: [advertised_peer],
    }


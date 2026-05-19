# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime, timezone

from cai.shared.apply import apply_node_timed_out
from cai.shared.types.common import NodeId
from cai.shared.types.events import NodeTimedOut
from cai.shared.types.profiling import NodeIdentity
from cai.shared.types.state import State


def test_apply_node_timed_out_removes_node_identity():
    node_id = NodeId("node-1")
    state = State(
        last_seen={node_id: datetime.now(tz=timezone.utc)},
        node_identities={node_id: NodeIdentity(friendly_name="Node 1")},
    )

    new_state = apply_node_timed_out(NodeTimedOut(node_id=node_id), state)

    assert node_id not in new_state.last_seen
    assert node_id not in new_state.node_identities


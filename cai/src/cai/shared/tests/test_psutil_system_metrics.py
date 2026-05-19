# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from cai.shared.apply import apply
from cai.shared.types.common import NodeId
from cai.shared.types.events import IndexedEvent, NodeGatheredInfo
from cai.shared.types.state import State
from cai.shared.types.profiling import SystemPerformanceProfile
from cai.utils.info_gatherer.info_gatherer import PsutilSystemMetrics


def test_psutil_system_metrics_updates_node_system() -> None:
    node_id = NodeId("node-a")
    state = apply(
        State(),
        IndexedEvent(
            idx=0,
            event=NodeGatheredInfo(
                node_id=node_id,
                when="2026-04-21T00:00:00+00:00",
                info=PsutilSystemMetrics(
                    system_profile=SystemPerformanceProfile(
                        pcpu_usage=0.42,
                        temp=67.5,
                    )
                ),
            ),
        ),
    )

    assert state.node_system[node_id].pcpu_usage == 0.42
    assert state.node_system[node_id].temp == 67.5


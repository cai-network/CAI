# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from cai_compute_chain.node_capabilities import export_node_capabilities_payload


def test_export_node_capabilities_includes_state_node_memory_summary() -> None:
    payload = export_node_capabilities_payload(
        state_payload={
            "nodeIdentities": {
                "node-worker": {
                    "workerEnabled": True,
                    "workerRewardAddress": "worker-address",
                }
            },
            "nodeMemory": {
                "node-worker": {
                    "ramTotal": {"inBytes": 16_000},
                    "ramAvailable": {"inBytes": 12_000},
                    "swapTotal": {"inBytes": 4_000},
                    "swapAvailable": {"inBytes": 3_000},
                }
            },
        },
        cai_url="http://127.0.0.1:52415",
        local_node_id="node-worker",
    )

    record = payload["records"][0]
    assert record["resource_summary"]["ramBytes"] == 16_000
    assert record["resource_summary"]["ramAvailableBytes"] == 12_000
    assert record["resource_summary"]["swapBytes"] == 4_000
    assert record["resource_summary"]["swapAvailableBytes"] == 3_000

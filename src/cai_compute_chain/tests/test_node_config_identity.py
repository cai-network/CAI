# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from cai_compute_chain.node_config import _resolve_current_node_identity


def test_resolve_current_node_identity_ignores_null_api_ports() -> None:
    node_id, identity = _resolve_current_node_identity(
        {
            "nodeIdentities": {
                "node-local": {"apiHost": None, "apiPort": None},
                "node-runtime": {"apiHost": "127.0.0.1", "apiPort": 52425},
            }
        },
        "http://127.0.0.1:52425",
    )

    assert node_id == "node-runtime"
    assert identity == {"apiHost": "127.0.0.1", "apiPort": 52425}

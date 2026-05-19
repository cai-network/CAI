# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "cai-owned-http-smoke.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location("cai_owned_http_smoke", TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load cai-owned-http-smoke tool module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_peer_urls_by_node_preserves_requester_overlay_when_requester_is_executor() -> None:
    module = _load_tool_module()

    result = module._peer_urls_by_node(
        requester_node_id="node-a",
        requester_url="http://127.0.0.1:52445",
        requester_peer_urls=[
            "cai-overlay:http://relay:52415?targetNodeId=node-a&relayRole=bootstrap",
        ],
        executor_urls={
            "node-a": ["http://127.0.0.1:52445/"],
            "node-b": [
                "cai-overlay:http://relay:52415?targetNodeId=node-b&relayRole=bootstrap",
            ],
        },
    )

    assert result["node-a"] == [
        "http://127.0.0.1:52445",
        "cai-overlay:http://relay:52415?targetNodeId=node-a&relayRole=bootstrap",
    ]
    assert result["node-b"] == [
        "cai-overlay:http://relay:52415?targetNodeId=node-b&relayRole=bootstrap",
    ]

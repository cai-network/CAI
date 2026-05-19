# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from cai.shared.single_instance import (
    NodeSingleInstanceGuard,
    load_node_instance_state,
    node_instance_error_message,
    node_instance_state_root,
    should_enforce_node_single_instance,
)


def test_node_single_instance_guard_blocks_second_process(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CAI_NODE_INSTANCE_STATE_DIR", str(tmp_path))
    guard = NodeSingleInstanceGuard()
    assert guard.acquire() is True
    guard.write_state({"apiPort": 52425, "dashboardUrl": "http://127.0.0.1:52425/"})

    env = os.environ.copy()
    env["CAI_NODE_INSTANCE_STATE_DIR"] = str(tmp_path)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parents[3]), env.get("PYTHONPATH", "")]
    )
    code = (
        "from cai.shared.single_instance import NodeSingleInstanceGuard;"
        "guard=NodeSingleInstanceGuard();"
        "raise SystemExit(0 if guard.acquire() else 17)"
    )
    result = subprocess.run([sys.executable, "-c", code], env=env, check=False)

    guard.release()
    assert result.returncode == 17


def test_node_instance_state_and_message(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CAI_NODE_INSTANCE_STATE_DIR", str(tmp_path))
    guard = NodeSingleInstanceGuard()
    assert guard.acquire() is True
    guard.write_state({"dashboardUrl": "http://127.0.0.1:52425/"})

    state = load_node_instance_state()
    message = node_instance_error_message()

    guard.clear_state()
    guard.release()
    assert state["pid"]
    assert state["dashboardUrl"] == "http://127.0.0.1:52425/"
    assert "already running" in message
    assert "dashboard=http://127.0.0.1:52425/" in message


def test_node_single_instance_state_root_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CAI_NODE_INSTANCE_STATE_DIR", str(tmp_path))
    assert node_instance_state_root() == tmp_path.resolve()
    assert should_enforce_node_single_instance() is True

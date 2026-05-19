# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import BinaryIO, Mapping


NODE_INSTANCE_LOCK_FILENAME = "node-instance.lock"
NODE_INSTANCE_STATE_FILENAME = "node-instance.json"


def should_enforce_node_single_instance() -> bool:
    return True


def node_instance_state_root() -> Path:
    override = os.getenv("CAI_NODE_INSTANCE_STATE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    desktop_override = os.getenv("CAI_DESKTOP_STATE_DIR", "").strip()
    if desktop_override:
        return Path(desktop_override).expanduser().resolve()
    if sys.platform.startswith("win"):
        for env_name in ("LOCALAPPDATA", "APPDATA"):
            value = os.getenv(env_name, "").strip()
            if value:
                return Path(value).expanduser().resolve() / "CAI"
    state_home = os.getenv("XDG_STATE_HOME", "").strip()
    if state_home:
        return Path(state_home).expanduser().resolve() / "cai"
    return Path.home().expanduser().resolve() / ".local" / "state" / "cai"


def node_instance_state_path() -> Path:
    return node_instance_state_root() / NODE_INSTANCE_STATE_FILENAME


def load_node_instance_state() -> dict[str, object]:
    path = node_instance_state_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def node_instance_error_message() -> str:
    state = load_node_instance_state()
    pid = str(state.get("pid") or "").strip()
    dashboard_url = str(state.get("dashboardUrl") or "").strip()
    details: list[str] = []
    if pid:
        details.append(f"pid={pid}")
    if dashboard_url:
        details.append(f"dashboard={dashboard_url}")
    suffix = f" ({', '.join(details)})" if details else ""
    return (
        "Another CAI node runtime is already running on this device"
        f"{suffix}. Stop it before starting a new CAI node."
    )


class NodeSingleInstanceGuard:
    def __init__(self, state_root: Path | None = None) -> None:
        self._state_root = (state_root or node_instance_state_root()).expanduser().resolve()
        self._lock_path = self._state_root / NODE_INSTANCE_LOCK_FILENAME
        self._state_path = self._state_root / NODE_INSTANCE_STATE_FILENAME
        self._handle: BinaryIO | None = None

    def acquire(self) -> bool:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self._lock_path.open("a+b")
        try:
            handle.seek(0)
            if handle.read(1) == b"":
                handle.seek(0)
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        self._handle = handle
        return True

    def write_state(self, payload: Mapping[str, object]) -> None:
        state = dict(payload)
        state["pid"] = os.getpid()
        state["updatedAt"] = int(time.time())
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clear_state(self) -> None:
        try:
            self._state_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            handle.close()

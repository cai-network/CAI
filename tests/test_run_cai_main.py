# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
import runpy
import sys
import types
from pathlib import Path


def _load_run_cai_main(monkeypatch, *, update_result: dict[str, object] | None = None) -> None:
    fake_update_channel = types.ModuleType("cai_compute_chain.update_channel")
    fake_update_channel.maybe_auto_update_on_launch = (
        lambda repo_root: update_result or {"updated": False}
    )

    fake_cai_package = types.ModuleType("cai")
    fake_cai_package.__path__ = []  # type: ignore[attr-defined]

    fake_cai_main = types.ModuleType("cai.main")
    fake_cai_main.main = lambda: 0

    monkeypatch.setitem(
        sys.modules, "cai_compute_chain.update_channel", fake_update_channel
    )
    monkeypatch.setitem(sys.modules, "cai", fake_cai_package)
    monkeypatch.setitem(sys.modules, "cai.main", fake_cai_main)

    script_path = Path(__file__).resolve().parents[1] / "tools" / "run-cai-main.py"
    runpy.run_path(str(script_path), run_name="__test__")


def test_run_cai_main_defaults_to_llama_cpp_backend(monkeypatch) -> None:
    monkeypatch.delenv("CAI_ALLOWED_INFERENCE_BACKENDS", raising=False)

    _load_run_cai_main(monkeypatch)

    assert os.environ["CAI_ALLOWED_INFERENCE_BACKENDS"] == "llama_cpp"


def test_run_cai_main_preserves_explicit_backend_override(monkeypatch) -> None:
    monkeypatch.setenv("CAI_ALLOWED_INFERENCE_BACKENDS", "custom_backend")

    _load_run_cai_main(monkeypatch)

    assert os.environ["CAI_ALLOWED_INFERENCE_BACKENDS"] == "custom_backend"


def test_run_cai_main_restarts_once_after_auto_update(monkeypatch) -> None:
    calls: list[tuple[str, list[str]]] = []

    def fake_execv(executable: str, argv: list[str]) -> None:
        calls.append((executable, argv))
        raise SystemExit("restart")

    monkeypatch.delenv("CAI_AUTO_UPDATE_RESTARTED", raising=False)
    monkeypatch.setattr(os, "execv", fake_execv)

    try:
        _load_run_cai_main(
            monkeypatch,
            update_result={"updated": True, "message": "updated"},
        )
    except SystemExit:
        pass
    else:
        raise AssertionError("Expected run-cai-main to restart after applying update.")

    assert os.environ["CAI_AUTO_UPDATE_RESTARTED"] == "1"
    assert calls
    assert calls[0][0] == sys.executable


def test_run_cai_main_does_not_restart_twice_after_auto_update(monkeypatch) -> None:
    monkeypatch.setenv("CAI_AUTO_UPDATE_RESTARTED", "1")

    def fail_execv(executable: str, argv: list[str]) -> None:
        raise AssertionError("execv should not be called twice for one update.")

    monkeypatch.setattr(os, "execv", fail_execv)

    _load_run_cai_main(
        monkeypatch,
        update_result={"updated": True, "message": "updated"},
    )

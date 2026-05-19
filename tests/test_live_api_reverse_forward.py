# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "live-api-reverse-forward.py"
    spec = importlib.util.spec_from_file_location("live_api_reverse_forward", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_forward_spec_rejects_out_of_range_ports() -> None:
    module = _load_module()

    assert module._parse_forward_spec("25445=127.0.0.1:52445") == (
        25445,
        "127.0.0.1",
        52445,
    )

    with pytest.raises(ValueError, match="between 1 and 65535"):
        module._parse_forward_spec("252445=127.0.0.1:52445")

    with pytest.raises(ValueError, match="between 1 and 65535"):
        module._parse_forward_spec("25445=127.0.0.1:99999")


def test_enable_transport_keepalive_configures_active_transport() -> None:
    module = _load_module()

    class Transport:
        def __init__(self) -> None:
            self.keepalive: int | None = None

        def is_active(self) -> bool:
            return True

        def set_keepalive(self, value: int) -> None:
            self.keepalive = value

    class Client:
        def __init__(self) -> None:
            self.transport = Transport()

        def get_transport(self):
            return self.transport

    client = Client()

    assert module._enable_transport_keepalive(client, 15) is True
    assert client.transport.keepalive == 15


def test_enable_transport_keepalive_can_be_disabled() -> None:
    module = _load_module()

    class Client:
        def get_transport(self):  # pragma: no cover - would fail the behavior
            raise AssertionError("transport should not be requested")

    assert module._enable_transport_keepalive(Client(), 0) is False

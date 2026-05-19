# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import sys

from cai.utils import banner


class BrokenStderr:
    def write(self, _: str) -> int:
        raise OSError(22, "Invalid argument")

    def flush(self) -> None:
        raise OSError(22, "Invalid argument")


def test_startup_banner_does_not_crash_when_stderr_is_unavailable(monkeypatch):
    monkeypatch.setattr(banner, "_is_first_run", lambda: False)
    monkeypatch.setattr(banner.logger, "debug", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "stderr", BrokenStderr())

    banner.print_startup_banner(52425)

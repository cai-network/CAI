# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cai_compute_chain.windows_firewall import (
    ensure_windows_firewall_rule,
    windows_firewall_rule_name,
)


def _completed(returncode: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class WindowsFirewallTests(unittest.TestCase):
    def test_rule_name_collapses_adjacent_runtime_ports(self) -> None:
        self.assertEqual(
            windows_firewall_rule_name([52426, 52425]),
            "CAI Runtime TCP 52425-52426",
        )

    def test_non_windows_skips_firewall_setup(self) -> None:
        with patch("cai_compute_chain.windows_firewall.sys.platform", "linux"):
            result = ensure_windows_firewall_rule([52425, 52426])

        self.assertEqual(result.status, "skipped")
        self.assertFalse(result.attempted)

    def test_existing_rule_does_not_request_elevation(self) -> None:
        commands: list[list[str]] = []

        def _runner(command):
            commands.append(list(command))
            return _completed(0, "Ok.")

        with patch("cai_compute_chain.windows_firewall.sys.platform", "win32"):
            result = ensure_windows_firewall_rule(
                [52425, 52426],
                runner=_runner,
                admin_probe=lambda: (_ for _ in ()).throw(AssertionError("unused")),
            )

        self.assertEqual(result.status, "already_configured")
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][0], "netsh.exe")

    def test_admin_configures_missing_rule_with_helper_script(self) -> None:
        commands: list[list[str]] = []

        def _runner(command):
            commands.append(list(command))
            return _completed(1) if len(commands) == 1 else _completed(0, "ready")

        with tempfile.TemporaryDirectory() as tmp:
            with patch("cai_compute_chain.windows_firewall.sys.platform", "win32"):
                result = ensure_windows_firewall_rule(
                    [52425, 52426],
                    runner=_runner,
                    admin_probe=lambda: True,
                    script_dir=Path(tmp),
                )
            remaining_scripts = list(Path(tmp).glob("cai-firewall-*.ps1"))

        self.assertEqual(result.status, "configured")
        self.assertEqual(commands[1][0], "powershell.exe")
        self.assertEqual(remaining_scripts, [])

    def test_non_admin_requests_uac_for_missing_rule(self) -> None:
        commands: list[list[str]] = []
        shell_calls: list[tuple[object, ...]] = []

        def _runner(command):
            commands.append(list(command))
            return _completed(1)

        def _shell_execute(*args):
            shell_calls.append(args)
            return 33

        with tempfile.TemporaryDirectory() as tmp:
            with patch("cai_compute_chain.windows_firewall.sys.platform", "win32"):
                result = ensure_windows_firewall_rule(
                    [52425, 52426],
                    runner=_runner,
                    admin_probe=lambda: False,
                    shell_execute=_shell_execute,
                    script_dir=Path(tmp),
                )
            remaining_scripts = list(Path(tmp).glob("cai-firewall-*.ps1"))

        self.assertEqual(result.status, "uac_requested")
        self.assertEqual(commands[0][0], "netsh.exe")
        self.assertEqual(shell_calls[0][1:3], ("runas", "powershell.exe"))
        self.assertEqual(len(remaining_scripts), 1)


if __name__ == "__main__":
    unittest.main()

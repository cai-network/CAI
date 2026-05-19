# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
AdminProbe = Callable[[], bool]
ShellExecute = Callable[[int | None, str, str, str, str | None, int], int]


@dataclass(frozen=True)
class WindowsFirewallRuleResult:
    attempted: bool
    status: str
    rule_name: str
    ports: tuple[int, ...]
    message: str = ""


def windows_firewall_rule_name(ports: Sequence[int]) -> str:
    normalized = _normalize_ports(ports)
    if len(normalized) == 2 and normalized[1] == normalized[0] + 1:
        port_text = f"{normalized[0]}-{normalized[1]}"
    else:
        port_text = ",".join(str(port) for port in normalized)
    return f"CAI Runtime TCP {port_text}"


def ensure_windows_firewall_rule(
    ports: Sequence[int],
    *,
    rule_name: str | None = None,
    allow_uac: bool | None = None,
    runner: Runner | None = None,
    admin_probe: AdminProbe | None = None,
    shell_execute: ShellExecute | None = None,
    script_dir: Path | None = None,
) -> WindowsFirewallRuleResult:
    normalized_ports = _normalize_ports(ports)
    resolved_rule_name = rule_name or windows_firewall_rule_name(normalized_ports)
    if not sys.platform.startswith("win"):
        return WindowsFirewallRuleResult(
            attempted=False,
            status="skipped",
            rule_name=resolved_rule_name,
            ports=normalized_ports,
            message="Windows Firewall is only configured on Windows.",
        )
    if _env_disabled("CAI_AUTO_WINDOWS_FIREWALL"):
        return WindowsFirewallRuleResult(
            attempted=False,
            status="disabled",
            rule_name=resolved_rule_name,
            ports=normalized_ports,
            message="CAI_AUTO_WINDOWS_FIREWALL disabled automatic firewall setup.",
        )

    command_runner = runner or _run_command
    if _firewall_rule_exists(resolved_rule_name, command_runner):
        return WindowsFirewallRuleResult(
            attempted=False,
            status="already_configured",
            rule_name=resolved_rule_name,
            ports=normalized_ports,
            message="Windows Firewall rule already exists.",
        )

    script_path = _write_firewall_script(
        normalized_ports,
        resolved_rule_name,
        script_dir=script_dir,
    )
    is_admin = (admin_probe or _is_windows_admin)()
    if is_admin:
        completed = command_runner(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
            ]
        )
        _unlink_best_effort(script_path)
        if completed.returncode == 0:
            return WindowsFirewallRuleResult(
                attempted=True,
                status="configured",
                rule_name=resolved_rule_name,
                ports=normalized_ports,
                message="Windows Firewall rule configured.",
            )
        output = (completed.stderr or completed.stdout or "").strip()
        return WindowsFirewallRuleResult(
            attempted=True,
            status="failed",
            rule_name=resolved_rule_name,
            ports=normalized_ports,
            message=output or f"Firewall helper failed with exit code {completed.returncode}.",
        )

    resolved_allow_uac = (
        not _env_disabled("CAI_WINDOWS_FIREWALL_UAC")
        if allow_uac is None
        else bool(allow_uac)
    )
    if not resolved_allow_uac:
        return WindowsFirewallRuleResult(
            attempted=False,
            status="needs_elevation",
            rule_name=resolved_rule_name,
            ports=normalized_ports,
            message="Administrator rights are required to create the Windows Firewall rule.",
        )

    shell_execute_fn = shell_execute or _shell_execute
    params = subprocess.list2cmdline(
        [
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script_path),
        ]
    )
    result = shell_execute_fn(None, "runas", "powershell.exe", params, None, 0)
    if int(result) <= 32:
        return WindowsFirewallRuleResult(
            attempted=True,
            status="uac_failed",
            rule_name=resolved_rule_name,
            ports=normalized_ports,
            message=f"Windows refused the elevated firewall helper request ({result}).",
        )
    return WindowsFirewallRuleResult(
        attempted=True,
        status="uac_requested",
        rule_name=resolved_rule_name,
        ports=normalized_ports,
        message="Requested elevation to create the Windows Firewall rule.",
    )


def _normalize_ports(ports: Sequence[int]) -> tuple[int, ...]:
    normalized: list[int] = []
    for raw_port in ports:
        port = int(raw_port)
        if port < 1 or port > 65535:
            raise ValueError(f"Invalid TCP port for Windows Firewall rule: {port}")
        if port not in normalized:
            normalized.append(port)
    if not normalized:
        raise ValueError("At least one TCP port is required for a Windows Firewall rule.")
    return tuple(sorted(normalized))


def _env_disabled(name: str) -> bool:
    return str(os.getenv(name) or "").strip().lower() in {"0", "false", "no", "off"}


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def _firewall_rule_exists(rule_name: str, runner: Runner) -> bool:
    try:
        completed = runner(
            [
                "netsh.exe",
                "advfirewall",
                "firewall",
                "show",
                "rule",
                f"name={rule_name}",
            ]
        )
    except OSError:
        return False
    return completed.returncode == 0


def _write_firewall_script(
    ports: tuple[int, ...],
    rule_name: str,
    *,
    script_dir: Path | None,
) -> Path:
    directory = script_dir or Path(tempfile.gettempdir())
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"cai-firewall-{os.getpid()}-{abs(hash((rule_name, ports)))}.ps1"
    escaped_name = rule_name.replace("'", "''")
    port_array = ", ".join(f"'{port}'" for port in ports)
    path.write_text(
        "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                f"$name = '{escaped_name}'",
                f"$ports = @({port_array})",
                "$rule = Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue",
                "if ($null -eq $rule) {",
                "  New-NetFirewallRule -DisplayName $name -Direction Inbound -Action Allow -Protocol TCP -LocalPort $ports -Profile Any | Out-Null",
                "} else {",
                "  Set-NetFirewallRule -DisplayName $name -Enabled True -Direction Inbound -Action Allow -Profile Any | Out-Null",
                "  Get-NetFirewallRule -DisplayName $name | Set-NetFirewallPortFilter -Protocol TCP -LocalPort $ports | Out-Null",
                "}",
                "Write-Output 'CAI firewall rule ready.'",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _is_windows_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _shell_execute(
    hwnd: int | None,
    operation: str,
    file: str,
    params: str,
    directory: str | None,
    show_cmd: int,
) -> int:
    return int(
        ctypes.windll.shell32.ShellExecuteW(
            hwnd,
            operation,
            file,
            params,
            directory,
            show_cmd,
        )
    )


def _unlink_best_effort(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass

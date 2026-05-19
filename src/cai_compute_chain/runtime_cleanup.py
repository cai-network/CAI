# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


LogBestEffortFailure = Callable[[str, Exception], None]
ListCaiInstances = Callable[..., Any]


def _noop_log_best_effort_failure(operation: str, exc: Exception) -> None:
    return None


def cleanup_orphan_llama_cpp_processes(
    *,
    cai_url: str | None = None,
    model_id: str | None = None,
    list_cai_instances_func: ListCaiInstances | None = None,
    log_best_effort_failure: LogBestEffortFailure | None = None,
    repo_root: Path | None = None,
) -> int:
    log_failure = log_best_effort_failure or _noop_log_best_effort_failure
    if cai_url and list_cai_instances_func is not None:
        try:
            if list_cai_instances_func(cai_url, model_id=model_id):
                return 0
        except Exception as exc:
            log_failure("orphan llama.cpp CAI instance lookup", exc)
    try:
        import psutil
    except Exception as exc:
        log_failure("orphan llama.cpp psutil import", exc)
        return 0

    active_repo_root = repo_root or Path(__file__).resolve().parents[2]
    candidate_roots = [
        active_repo_root / "cai" / ".runtime" / "llama.cpp",
        active_repo_root / "runtime" / "llama.cpp",
        active_repo_root / "data" / "runtime" / "llama.cpp",
        active_repo_root / ".dist" / "CAI-portable" / "llama.cpp",
    ]
    managed_roots = [root.resolve() for root in candidate_roots if root.exists()]
    if not managed_roots:
        return 0

    def _managed_runtime_process_path(process_info: dict[str, Any]) -> Path | None:
        raw_exe = process_info.get("exe")
        raw_cmdline = process_info.get("cmdline")
        candidates: list[str] = []
        if isinstance(raw_exe, str) and raw_exe.strip():
            candidates.append(raw_exe.strip())
        if isinstance(raw_cmdline, list) and raw_cmdline:
            first = raw_cmdline[0]
            if isinstance(first, str) and first.strip():
                candidates.append(first.strip())
        for raw_path in candidates:
            try:
                resolved = Path(raw_path).resolve()
            except Exception:
                continue
            if any(root == resolved or root in resolved.parents for root in managed_roots):
                return resolved
        return None

    terminated = 0
    for process in psutil.process_iter(["pid", "name", "exe", "cmdline"]):
        try:
            name = str(process.info.get("name") or "").strip().lower()
            if name not in {"llama-server.exe", "llama-server", "rpc-server.exe", "rpc-server"}:
                continue
            if _managed_runtime_process_path(process.info) is None:
                continue
            process.terminate()
            try:
                process.wait(timeout=10)
            except psutil.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            terminated += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return terminated

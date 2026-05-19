# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import re
import shlex
import socket
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from typing import Any
from urllib.request import urlopen


LLAMA_SERVER_ENV = "CAI_LLAMA_CPP_SERVER"
LLAMA_RPC_SERVER_ENV = "CAI_LLAMA_CPP_RPC_SERVER"


@dataclass(frozen=True)
class LlamaCppBinarySet:
    llama_server: Path | None
    rpc_server: Path | None


@dataclass(frozen=True)
class ManagedLlamaCppBinary:
    path: Path | None
    args: tuple[str, ...] = ()


@dataclass(frozen=True)
class LlamaCppSessionPaths:
    root: Path
    state_dir: Path
    cache_dir: Path
    logs_dir: Path
    stdout_log: Path
    stderr_log: Path


@dataclass(frozen=True)
class ManagedLlamaCppRuntime:
    abi: str
    platform: str | None
    repo_root: Path | None
    runtime_root: Path | None
    model_id: str | None
    llama_server: ManagedLlamaCppBinary
    rpc_server: ManagedLlamaCppBinary
    session_paths: LlamaCppSessionPaths | None


def split_llama_cpp_subprocess_command(raw: str) -> list[str]:
    command = str(raw or "").strip()
    if not command:
        return []
    return [
        _strip_wrapping_quotes(str(item).strip())
        for item in shlex.split(command, posix=(os.name != "nt"))
        if str(item).strip()
    ]


def _strip_wrapping_quotes(value: str) -> str:
    clean = str(value or "").strip()
    if len(clean) >= 2 and clean[0] == clean[-1] and clean[0] in {"'", '"'}:
        return clean[1:-1]
    return clean


def resolve_llama_cpp_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_llama_cpp_binary_set(
    *,
    repo_root: Path | None = None,
    env: dict[str, str] | None = None,
    os_name: str | None = None,
) -> LlamaCppBinarySet:
    return LlamaCppBinarySet(
        llama_server=resolve_llama_cpp_binary(
            "llama-server",
            env_var=LLAMA_SERVER_ENV,
            repo_root=repo_root,
            env=env,
            os_name=os_name,
        ),
        rpc_server=resolve_llama_cpp_binary(
            "rpc-server",
            env_var=LLAMA_RPC_SERVER_ENV,
            repo_root=repo_root,
            env=env,
            os_name=os_name,
        ),
    )


def default_llama_cpp_runtime_root(
    *,
    repo_root: Path | None = None,
) -> Path:
    return (repo_root or resolve_llama_cpp_repo_root()) / ".cai-local" / "llama-shard-runtime"


def resolve_llama_cpp_binary(
    binary_stem: str,
    *,
    env_var: str,
    repo_root: Path | None = None,
    env: dict[str, str] | None = None,
    os_name: str | None = None,
) -> Path | None:
    active_env = env or dict(os.environ)
    platform_name = str(os_name or os.name).strip().lower() or os.name
    suffix = ".exe" if platform_name == "nt" else ""
    binary_name = f"{binary_stem}{suffix}"
    candidates: list[Path] = []

    env_path = str(active_env.get(env_var) or "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())

    root = (repo_root or resolve_llama_cpp_repo_root()).expanduser().resolve()
    bundle_root = _bundle_root()
    if bundle_root is not None and platform_name == "nt":
        candidates.append(bundle_root / "llama.cpp" / binary_name)

    runtime_roots = [
        root / "data" / "runtime" / "llama.cpp",
        root / "runtime" / "llama.cpp",
    ]
    for runtime_root in runtime_roots:
        candidates.append(runtime_root / binary_name)
        candidates.append(runtime_root / "bin" / binary_name)

    cai_runtime_root = root / "cai" / ".runtime" / "llama.cpp"
    if platform_name == "nt":
        candidates.extend(
            [
                cai_runtime_root / "windows" / "build" / binary_name,
                cai_runtime_root / "windows" / "build" / "bin" / binary_name,
                cai_runtime_root / "build" / "bin" / binary_name,
            ]
        )
    else:
        candidates.extend(
            [
                cai_runtime_root / "wsl" / "build" / "bin" / binary_stem,
                cai_runtime_root / "build" / "bin" / binary_stem,
                cai_runtime_root / "windows" / "build" / binary_name,
                cai_runtime_root / "windows" / "build" / "bin" / binary_name,
            ]
        )

    which_name = binary_name if platform_name == "nt" else binary_stem
    which_path = shutil.which(which_name)
    if which_path:
        candidates.append(Path(which_path))

    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            continue
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def prepare_llama_cpp_session_paths(
    *,
    base_root: str | Path,
    session_id: str,
    model_id: str,
    layer_start: int | None,
    layer_end: int | None,
) -> LlamaCppSessionPaths:
    root = Path(base_root).expanduser().resolve()
    safe_session = _safe_segment(session_id, fallback="session")
    safe_model = _safe_segment(model_id, fallback="model")
    layer_segment = (
        f"layers-{'' if layer_start is None else int(layer_start)}-"
        f"{'' if layer_end is None else int(layer_end)}"
    )
    session_root = root / safe_model / layer_segment / safe_session
    state_dir = session_root / "state"
    cache_dir = session_root / "cache"
    logs_dir = session_root / "logs"
    for path in (state_dir, cache_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    return LlamaCppSessionPaths(
        root=session_root,
        state_dir=state_dir,
        cache_dir=cache_dir,
        logs_dir=logs_dir,
        stdout_log=logs_dir / "stdout.log",
        stderr_log=logs_dir / "stderr.log",
    )


def windows_subprocess_creation_flags(*, os_name: str | None = None) -> int:
    platform_name = str(os_name or os.name).strip().lower() or os.name
    if platform_name != "nt":
        return 0
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
        subprocess,
        "DETACHED_PROCESS",
        0x00000008,
    )


def windows_subprocess_startupinfo(
    *,
    os_name: str | None = None,
) -> subprocess.STARTUPINFO | None:
    platform_name = str(os_name or os.name).strip().lower() or os.name
    if platform_name != "nt" or not hasattr(subprocess, "STARTUPINFO"):
        return None
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return startupinfo


def resolve_managed_llama_cpp_runtime(
    request: Mapping[str, Any],
) -> ManagedLlamaCppRuntime | None:
    payload = request.get("managedRuntime")
    if not isinstance(payload, Mapping):
        return None
    session_paths = _managed_session_paths(payload.get("sessionPaths"))
    repo_root = _managed_optional_path(payload.get("repoRoot"))
    runtime_root = _managed_optional_path(payload.get("runtimeRoot"))
    llama_cpp = payload.get("llamaCpp")
    llama_cpp_mapping = llama_cpp if isinstance(llama_cpp, Mapping) else {}
    return ManagedLlamaCppRuntime(
        abi=str(payload.get("abi") or "").strip(),
        platform=str(payload.get("platform") or "").strip() or None,
        repo_root=repo_root,
        runtime_root=runtime_root,
        model_id=str(payload.get("modelId") or "").strip() or None,
        llama_server=ManagedLlamaCppBinary(
            path=_managed_optional_path(llama_cpp_mapping.get("llamaServerPath")),
            args=_managed_string_tuple(llama_cpp_mapping.get("llamaServerArgs")),
        ),
        rpc_server=ManagedLlamaCppBinary(
            path=_managed_optional_path(llama_cpp_mapping.get("rpcServerPath")),
            args=_managed_string_tuple(llama_cpp_mapping.get("rpcServerArgs")),
        ),
        session_paths=session_paths,
    )


def resolve_request_local_artifact_path(
    request: Mapping[str, Any],
    *,
    preferred_kind: str = "model",
) -> Path | None:
    local_resolution = request.get("localArtifactResolution")
    if not isinstance(local_resolution, Mapping):
        return None
    preferred = str(preferred_kind or "").strip().lower()
    ordered_keys = (
        ("assignmentArtifact", "modelArtifact")
        if preferred == "assignment"
        else ("modelArtifact", "assignmentArtifact")
    )
    for key in ordered_keys:
        artifact = local_resolution.get(key)
        if not isinstance(artifact, Mapping):
            continue
        local_path = _managed_optional_path(artifact.get("localPath"))
        if local_path is not None and local_path.exists() and local_path.is_file():
            return local_path
    return None


def choose_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_llama_server_command(
    runtime: ManagedLlamaCppRuntime,
    *,
    model_path: Path,
    host: str,
    port: int,
    slot_save_path: Path,
    parallel_slots: int = 1,
) -> list[str]:
    if runtime.llama_server.path is None:
        raise FileNotFoundError("Managed llama.cpp runtime does not define llama-server.")
    return [
        str(runtime.llama_server.path),
        *runtime.llama_server.args,
        "-m",
        str(model_path),
        "--host",
        str(host),
        "--port",
        str(int(port)),
        "-np",
        str(max(1, int(parallel_slots or 1))),
        "--slot-save-path",
        str(slot_save_path),
    ]


def wait_for_llama_server_ready(
    server_url: str,
    *,
    timeout_sec: float,
    probe_path: str = "/slots",
) -> None:
    deadline = time.monotonic() + max(0.1, float(timeout_sec or 0.1))
    last_error: Exception | None = None
    url = str(server_url or "").rstrip("/") + "/" + str(probe_path or "").lstrip("/")
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=min(2.0, max(0.1, float(timeout_sec or 0.1)))) as response:
                if int(response.status) >= 200 and int(response.status) < 500:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(0.2)
    if last_error is not None:
        raise TimeoutError(
            f"Timed out waiting for llama.cpp server readiness: {last_error}"
        ) from last_error
    raise TimeoutError("Timed out waiting for llama.cpp server readiness.")


def _bundle_root() -> Path | None:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root is None:
        return None
    try:
        return Path(frozen_root).resolve()
    except Exception:
        return None


def _repo_root() -> Path:
    return resolve_llama_cpp_repo_root()


def _safe_segment(value: str, *, fallback: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip())
    clean = clean.strip(" .-_")
    return clean or fallback


def _managed_optional_path(value: Any) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return Path(raw).expanduser().resolve()
    except Exception:
        return None


def _managed_string_tuple(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, list):
        return ()
    output: list[str] = []
    for item in value:
        clean = str(item or "").strip()
        if clean:
            output.append(clean)
    return tuple(output)


def _managed_session_paths(value: Any) -> LlamaCppSessionPaths | None:
    if not isinstance(value, Mapping):
        return None
    root = _managed_optional_path(value.get("root"))
    state_dir = _managed_optional_path(value.get("stateDir"))
    cache_dir = _managed_optional_path(value.get("cacheDir"))
    logs_dir = _managed_optional_path(value.get("logsDir"))
    stdout_log = _managed_optional_path(value.get("stdoutLog"))
    stderr_log = _managed_optional_path(value.get("stderrLog"))
    if not all((root, state_dir, cache_dir, logs_dir, stdout_log, stderr_log)):
        return None
    return LlamaCppSessionPaths(
        root=root,
        state_dir=state_dir,
        cache_dir=cache_dir,
        logs_dir=logs_dir,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
    )

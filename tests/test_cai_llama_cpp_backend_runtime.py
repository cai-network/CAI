# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.cai_llama_cpp_backend_runtime import (  # noqa: E402
    prepare_llama_cpp_session_paths,
    build_llama_server_command,
    resolve_managed_llama_cpp_runtime,
    resolve_request_local_artifact_path,
    resolve_llama_cpp_binary,
    resolve_llama_cpp_binary_set,
    split_llama_cpp_subprocess_command,
    windows_subprocess_creation_flags,
    windows_subprocess_startupinfo,
)


def test_resolve_llama_cpp_binary_prefers_explicit_env_path(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    explicit = tmp_path / "custom" / "llama-server"
    explicit.parent.mkdir(parents=True)
    explicit.write_bytes(b"binary")

    resolved = resolve_llama_cpp_binary(
        "llama-server",
        env_var="CAI_LLAMA_CPP_SERVER",
        repo_root=repo_root,
        env={"CAI_LLAMA_CPP_SERVER": str(explicit)},
        os_name="posix",
    )

    assert resolved == explicit.resolve()


def test_split_llama_cpp_subprocess_command_strips_windows_wrapping_quotes() -> None:
    command = (
        '"D:\\Program Files\\CAI\\python.exe" -m '
        'cai_compute_chain.cai_llama_cpp_patched_executor_host --jsonl'
    )

    parsed = split_llama_cpp_subprocess_command(command)

    assert parsed == [
        "D:\\Program Files\\CAI\\python.exe",
        "-m",
        "cai_compute_chain.cai_llama_cpp_patched_executor_host",
        "--jsonl",
    ]


def test_resolve_llama_cpp_binary_finds_wsl_build_on_linux(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    binary = (
        repo_root
        / "cai"
        / ".runtime"
        / "llama.cpp"
        / "wsl"
        / "build"
        / "bin"
        / "rpc-server"
    )
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"binary")

    resolved = resolve_llama_cpp_binary(
        "rpc-server",
        env_var="CAI_LLAMA_CPP_RPC_SERVER",
        repo_root=repo_root,
        env={},
        os_name="posix",
    )

    assert resolved == binary.resolve()


def test_resolve_llama_cpp_binary_set_finds_windows_runtime_payload(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    llama_server = repo_root / "runtime" / "llama.cpp" / "llama-server.exe"
    rpc_server = repo_root / "runtime" / "llama.cpp" / "rpc-server.exe"
    llama_server.parent.mkdir(parents=True)
    llama_server.write_bytes(b"server")
    rpc_server.write_bytes(b"rpc")

    binaries = resolve_llama_cpp_binary_set(
        repo_root=repo_root,
        env={},
        os_name="nt",
    )

    assert binaries.llama_server == llama_server.resolve()
    assert binaries.rpc_server == rpc_server.resolve()


def test_prepare_llama_cpp_session_paths_creates_runtime_tree(tmp_path: Path) -> None:
    paths = prepare_llama_cpp_session_paths(
        base_root=tmp_path / "runtime",
        session_id="session:one/two",
        model_id="cai-network/Qwen3-0.6B-GGUF",
        layer_start=0,
        layer_end=14,
    )

    assert paths.root.exists()
    assert paths.state_dir.exists()
    assert paths.cache_dir.exists()
    assert paths.logs_dir.exists()
    assert paths.stdout_log.name == "stdout.log"
    assert paths.stderr_log.name == "stderr.log"
    assert "cai-network-Qwen3-0.6B-GGUF" in str(paths.root)
    assert "session-one-two" in str(paths.root)


def test_resolve_managed_llama_cpp_runtime_reads_session_and_binary_metadata(
    tmp_path: Path,
) -> None:
    session_paths = prepare_llama_cpp_session_paths(
        base_root=tmp_path / "runtime",
        session_id="session-one",
        model_id="cai-network/Qwen3-0.6B-GGUF",
        layer_start=0,
        layer_end=14,
    )
    request = {
        "managedRuntime": {
            "abi": "cai-llama-cpp-managed-runtime-v1",
            "platform": "posix",
            "repoRoot": str((tmp_path / "repo").resolve()),
            "runtimeRoot": str((tmp_path / "runtime").resolve()),
            "modelId": "cai-network/Qwen3-0.6B-GGUF",
            "llamaCpp": {
                "llamaServerPath": str((tmp_path / "llama-server").resolve()),
                "llamaServerArgs": ["-u", "fake_server.py"],
            },
            "sessionPaths": {
                "root": str(session_paths.root),
                "stateDir": str(session_paths.state_dir),
                "cacheDir": str(session_paths.cache_dir),
                "logsDir": str(session_paths.logs_dir),
                "stdoutLog": str(session_paths.stdout_log),
                "stderrLog": str(session_paths.stderr_log),
            },
        }
    }

    runtime = resolve_managed_llama_cpp_runtime(request)

    assert runtime is not None
    assert runtime.abi == "cai-llama-cpp-managed-runtime-v1"
    assert runtime.platform == "posix"
    assert runtime.model_id == "cai-network/Qwen3-0.6B-GGUF"
    assert runtime.llama_server.args == ("-u", "fake_server.py")
    assert runtime.session_paths is not None
    assert runtime.session_paths.state_dir == session_paths.state_dir


def test_resolve_request_local_artifact_path_prefers_model_artifact(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    assignment_path = tmp_path / "assignment.gguf"
    model_path.write_bytes(b"model")
    assignment_path.write_bytes(b"assignment")
    request = {
        "localArtifactResolution": {
            "modelArtifact": {"localPath": str(model_path)},
            "assignmentArtifact": {"localPath": str(assignment_path)},
        }
    }

    resolved = resolve_request_local_artifact_path(request, preferred_kind="model")

    assert resolved == model_path.resolve()


def test_build_llama_server_command_uses_managed_runtime_binary_and_args(
    tmp_path: Path,
) -> None:
    session_paths = prepare_llama_cpp_session_paths(
        base_root=tmp_path / "runtime",
        session_id="session-one",
        model_id="cai-network/Qwen3-0.6B-GGUF",
        layer_start=0,
        layer_end=14,
    )
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"model")
    request = {
        "managedRuntime": {
            "abi": "cai-llama-cpp-managed-runtime-v1",
            "llamaCpp": {
                "llamaServerPath": str((tmp_path / "python").resolve()),
                "llamaServerArgs": ["-u", "fake_server.py"],
            },
            "sessionPaths": {
                "root": str(session_paths.root),
                "stateDir": str(session_paths.state_dir),
                "cacheDir": str(session_paths.cache_dir),
                "logsDir": str(session_paths.logs_dir),
                "stdoutLog": str(session_paths.stdout_log),
                "stderrLog": str(session_paths.stderr_log),
            },
        }
    }
    runtime = resolve_managed_llama_cpp_runtime(request)
    assert runtime is not None

    command = build_llama_server_command(
        runtime,
        model_path=model_path,
        host="127.0.0.1",
        port=8080,
        slot_save_path=session_paths.state_dir,
        parallel_slots=1,
    )

    assert command[:3] == [str((tmp_path / "python").resolve()), "-u", "fake_server.py"]
    assert "-m" in command
    assert str(model_path) in command
    assert "--slot-save-path" in command


def test_windows_helpers_are_noop_on_non_windows() -> None:
    assert windows_subprocess_creation_flags(os_name="posix") == 0
    assert windows_subprocess_startupinfo(os_name="posix") is None


def test_llama_cpp_runner_import_does_not_require_windows_startupinfo() -> None:
    script = f"""
import pathlib
import subprocess
import sys

repo = pathlib.Path({str(REPO_ROOT)!r})
sys.path.insert(0, str(repo / "src"))
sys.path.insert(0, str(repo / "cai" / "src"))

if hasattr(subprocess, "STARTUPINFO"):
    delattr(subprocess, "STARTUPINFO")

from cai.worker.runner.llama_cpp.runner import _windows_startupinfo

assert _windows_startupinfo() is None
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

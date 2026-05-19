# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing import freeze_support
from pathlib import Path
from typing import BinaryIO, Final, Sequence

from cai_compute_chain.cai_runtime_launcher import (
    _set_cai_llm_shard_production_adapter_defaults,
    bootstrap_peers_argument,
    discover_peer_book_peers,
    normalize_peers,
    set_cai_owned_task_level_env_defaults,
)
from cai_compute_chain.model import CaiNetworkConfig, NetworkModelPolicy, WalletPolicy
from cai_compute_chain.windows_firewall import (
    WindowsFirewallRuleResult,
    ensure_windows_firewall_rule,
)
from cai_compute_chain.wallet import lock_wallet


DEFAULT_DESKTOP_API_PORT = 52425
DEFAULT_DESKTOP_LIBP2P_PORT = 52426
CAI_INTERNAL_RUNTIME_FLAG: Final[str] = "--cai-runtime"
INLINE_CODE_FLAG: Final[str] = "-c"
PORTABLE_HOME_DIRNAME: Final[str] = ".cai"
AUTO_UPDATE_INTERVAL_ENV: Final[str] = "CAI_AUTO_UPDATE_INTERVAL_SECONDS"
AUTO_UPDATE_ERROR_INTERVAL_ENV: Final[str] = "CAI_AUTO_UPDATE_ERROR_INTERVAL_SECONDS"
AUTO_UPDATE_INITIAL_DELAY_ENV: Final[str] = "CAI_AUTO_UPDATE_INITIAL_DELAY_SECONDS"
DEFAULT_AUTO_UPDATE_INTERVAL_SECONDS: Final[int] = 2 * 60
DEFAULT_AUTO_UPDATE_ERROR_INTERVAL_SECONDS: Final[int] = 60
DESKTOP_PREFERENCES_FILENAME: Final[str] = "desktop-preferences.json"
DESKTOP_INSTANCE_LOCK_FILENAME: Final[str] = "desktop-instance.lock"
DESKTOP_INSTANCE_STATE_FILENAME: Final[str] = "desktop-instance.json"
DEFAULT_DESKTOP_LANGUAGE: Final[str] = "en"
DESKTOP_ICON_SIZES: Final[tuple[tuple[int, int], ...]] = (
    (16, 16),
    (20, 20),
    (24, 24),
    (32, 32),
    (40, 40),
    (48, 48),
    (64, 64),
    (128, 128),
    (256, 256),
)


@dataclass(frozen=True)
class CaiDesktopConfig:
    repo_root: Path
    api_port: int = DEFAULT_DESKTOP_API_PORT
    libp2p_port: int = DEFAULT_DESKTOP_LIBP2P_PORT
    python_executable: str | None = None
    cai_home: str | None = None
    no_downloads: bool = False
    no_worker: bool = False
    force_master: bool = False
    offline: bool = False
    verbose: bool = False
    open_browser: bool = True
    start_on_launch: bool = True
    language: str = DEFAULT_DESKTOP_LANGUAGE


def _desktop_state_root() -> Path:
    override = os.getenv("CAI_DESKTOP_STATE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform.startswith("win"):
        for env_name in ("LOCALAPPDATA", "APPDATA"):
            value = os.getenv(env_name, "").strip()
            if value:
                return Path(value).expanduser().resolve() / "CAI"
    state_home = os.getenv("XDG_STATE_HOME", "").strip()
    if state_home:
        return Path(state_home).expanduser().resolve() / "cai"
    return Path.home().expanduser().resolve() / ".local" / "state" / "cai"


def _desktop_preferences_path() -> Path:
    return _desktop_state_root() / DESKTOP_PREFERENCES_FILENAME


def _desktop_instance_lock_path() -> Path:
    return _desktop_state_root() / DESKTOP_INSTANCE_LOCK_FILENAME


def _desktop_instance_state_path() -> Path:
    return _desktop_state_root() / DESKTOP_INSTANCE_STATE_FILENAME


def _load_desktop_preferences() -> dict[str, object]:
    path = _desktop_preferences_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_desktop_preferences(payload: dict[str, object]) -> None:
    path = _desktop_preferences_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _stored_desktop_language() -> str | None:
    language = _load_desktop_preferences().get("language")
    return language if language in {"en", "ru"} else None


def save_desktop_language(language: str) -> None:
    normalized = (language or "").lower()
    if normalized not in {"en", "ru"}:
        return
    payload = _load_desktop_preferences()
    payload["language"] = normalized
    _save_desktop_preferences(payload)


def _load_running_instance_state() -> dict[str, object]:
    path = _desktop_instance_state_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _env_flag(name: str) -> bool:
    return str(os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_positive_int(name: str, default: int) -> int:
    try:
        value = int(str(os.getenv(name) or "").strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def should_enforce_single_instance() -> bool:
    if _env_flag("CAI_ALLOW_MULTIPLE_DESKTOP_INSTANCES"):
        return False
    return True


class DesktopSingleInstanceGuard:
    def __init__(self) -> None:
        self._handle: BinaryIO | None = None
        self._state_path = _desktop_instance_state_path()
        self._lock_path = _desktop_instance_lock_path()

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

    def write_state(self, config: CaiDesktopConfig) -> None:
        payload = {
            "pid": os.getpid(),
            "repoRoot": str(config.repo_root),
            "dashboardUrl": f"http://127.0.0.1:{config.api_port}/",
            "language": resolve_language(config.language),
            "updatedAt": int(time.time()),
        }
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
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


def _bundle_root() -> Path | None:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root is None:
        return None
    return Path(frozen_root).resolve()


def _portable_executable_root() -> Path | None:
    if not getattr(sys, "frozen", False):
        return None
    return Path(sys.executable).resolve().parent


def _is_embedded_portable_app() -> bool:
    bundle_root = _bundle_root()
    if bundle_root is None or _portable_executable_root() is None:
        return False
    return (bundle_root / "dashboard").is_dir() and (bundle_root / "resources").is_dir()


def _is_onefile_embedded_portable_app() -> bool:
    bundle_root = _bundle_root()
    executable_root = _portable_executable_root()
    if bundle_root is None or executable_root is None:
        return False
    return bundle_root.parent != executable_root


def _is_source_repo(root: Path) -> bool:
    runtime_main = root / "cai" / "src" / "cai" / "main.py"
    return runtime_main.exists() and (root / "tools" / "run-cai-main.py").exists()


def _is_portable_root(root: Path) -> bool:
    return (root / "runtime" / "cai").exists() or (root / "cai.exe").exists()


def _maybe_run_inline_code(argv: Sequence[str]) -> bool:
    try:
        flag_index = argv.index(INLINE_CODE_FLAG)
    except ValueError:
        return False

    code_index = flag_index + 1
    if code_index >= len(argv):
        return False

    inline_code = argv[code_index]
    sys.argv = ["-c", *argv[code_index + 1 :]]
    namespace: dict[str, object] = {"__name__": "__main__"}
    exec(inline_code, namespace, namespace)
    return True


def _candidate_repo_roots(explicit_repo_root: str | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit_repo_root:
        candidates.append(Path(explicit_repo_root))
    env_repo = os.getenv("CAI_REPO_ROOT")
    if env_repo:
        candidates.append(Path(env_repo))
    candidates.append(Path.cwd())

    executable = Path(sys.executable).resolve()
    candidates.extend(executable.parents)
    candidates.extend(Path(__file__).resolve().parents)
    return candidates


def resolve_repo_root(explicit_repo_root: str | None = None) -> Path:
    if _is_embedded_portable_app():
        if explicit_repo_root:
            return Path(explicit_repo_root).expanduser().resolve()
        portable_root = _portable_executable_root()
        if portable_root is not None:
            return portable_root

    for candidate in _candidate_repo_roots(explicit_repo_root):
        root = candidate.expanduser().resolve()
        if _is_source_repo(root) or _is_portable_root(root):
            return root
    raise FileNotFoundError(
        "Cannot locate CAI source repository or portable runtime. "
        "Pass --repo-root or set CAI_REPO_ROOT."
    )


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _python_executable_for_runtime(config: CaiDesktopConfig) -> Path:
    if config.python_executable:
        path = Path(config.python_executable).expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"Cannot find CAI runtime Python executable: {path}")

    repo_root = config.repo_root
    candidates = [
        repo_root / "cai" / ".venv-win" / "Scripts" / "python.exe",
        repo_root / "cai" / ".venv" / "Scripts" / "python.exe",
        repo_root / "cai" / ".venv" / "bin" / "python",
    ]
    found = _first_existing(candidates)
    if found is not None:
        return found
    raise FileNotFoundError(
        "Cannot find CAI runtime Python environment. Expected cai/.venv-win or cai/.venv. "
        "Run tools/install.ps1 or tools/install.sh first."
    )


def _bundled_runtime_executable(config: CaiDesktopConfig) -> Path | None:
    env_path = os.getenv("CAI_RUNTIME_EXECUTABLE")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            config.repo_root / "runtime" / "cai" / "cai.exe",
            config.repo_root / "cai.exe",
            config.repo_root / "runtime" / "cai" / "cai.exe",
            config.repo_root / "cai.exe",
        ]
    )
    return _first_existing([path.expanduser().resolve() for path in candidates])


def _build_runtime_flags(config: CaiDesktopConfig) -> list[str]:
    command: list[str] = []
    if config.verbose:
        command.append("-v")
    if config.force_master:
        command.append("-m")
    command.extend(
        [
            "--api-port",
            str(config.api_port),
            "--libp2p-port",
            str(config.libp2p_port),
        ]
    )
    if config.no_downloads:
        command.append("--no-downloads")
    if config.no_worker:
        command.append("--no-worker")
    if config.offline:
        command.append("--offline")

    network_config = CaiNetworkConfig()
    bootstrap_peers = normalize_peers(
        [*network_config.bootstrap_peers, *load_peer_book_for_repo(config.repo_root)]
    )
    if bootstrap_peers and not config.offline:
        command.extend(["--bootstrap-peers", bootstrap_peers_argument(bootstrap_peers)])
    return command


def _append_path(existing: str | None, paths: list[Path]) -> str:
    parts = [str(path) for path in paths if path.exists()]
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def _append_env_path(existing: str | None, values: list[Path]) -> str:
    parts = [str(value) for value in values]
    if existing:
        parts.extend(existing.split(os.pathsep))
    return os.pathsep.join(normalize_peers(parts))


def _lock_wallet_session_best_effort() -> None:
    try:
        lock_wallet()
    except Exception:
        # Wallet locking is a safety cleanup step; it must never block the
        # desktop shell from starting or exiting.
        pass


def _set_compat_env(
    env: dict[str, str], cai_key: str, value: str, *, legacy_key: str | None = None
) -> None:
    env[cai_key] = value
    if legacy_key and legacy_key != cai_key:
        env[legacy_key] = value


def _desktop_disconnect_grace_seconds(
    config: CaiDesktopConfig, env: dict[str, str]
) -> str:
    configured = (
        str(
            env.get("CAI_CONNECTION_DISCONNECT_GRACE_SECONDS")
            or env.get("CAI_CONNECTION_DISCONNECT_GRACE_SECONDS")
            or ""
        )
        .strip()
    )
    if configured:
        return configured

    # Desktop-managed nodes sit on noisier edge networks than the raw CLI/server
    # path, and each re-election cascades into worker churn. Portable is the most
    # sensitive because that churn is surfaced as visible process activity.
    if _is_portable_root(config.repo_root):
        return "300"
    return "120"


def _desktop_session_transition_debounce_seconds(
    config: CaiDesktopConfig, env: dict[str, str]
) -> str:
    configured = (
        str(
            env.get("CAI_SESSION_TRANSITION_DEBOUNCE_SECONDS")
            or env.get("CAI_SESSION_TRANSITION_DEBOUNCE_SECONDS")
            or ""
        )
        .strip()
    )
    if configured:
        return configured

    # Session/master churn is what causes visible worker/runtime restarts.
    # Portable gets a slightly longer debounce because its bundled process tree
    # makes that churn much more obvious to the user on Windows.
    if _is_portable_root(config.repo_root):
        return "8"
    return "5"


def _windows_hidden_startupinfo() -> subprocess.STARTUPINFO | None:
    if not sys.platform.startswith("win"):
        return None
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return startupinfo


def _windows_popen_flags(
    *, new_process_group: bool = False, new_console: bool = False
) -> int:
    if not sys.platform.startswith("win"):
        return 0
    creationflags = 0
    if new_process_group:
        creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    if new_console:
        creationflags |= getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    else:
        creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return creationflags


def _can_bind_tcp_port(port: int, host: str) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((host, port))
        return True
    except OSError:
        return False


def _find_available_port(
    preferred_port: int,
    *,
    host: str,
    reserved: set[int] | None = None,
    max_offset: int = 128,
) -> int:
    reserved_ports = reserved or set()
    for offset in range(max_offset + 1):
        candidate = preferred_port + offset
        if candidate in reserved_ports:
            continue
        if _can_bind_tcp_port(candidate, host):
            return candidate
    raise OSError(
        f"Unable to find a free TCP port near {preferred_port} for host {host!r}."
    )


def _user_models_dir() -> Path | None:
    try:
        return Path.home() / "models"
    except RuntimeError:
        return None


def cai_home_path(config: CaiDesktopConfig) -> Path:
    network_config = CaiNetworkConfig()
    if config.cai_home:
        return Path(config.cai_home).expanduser().resolve()
    if _is_portable_root(config.repo_root):
        return config.repo_root / "data" / PORTABLE_HOME_DIRNAME
    return config.repo_root / network_config.default_cai_home_dirname


def wallet_home_path(config: CaiDesktopConfig) -> Path:
    wallet_policy = WalletPolicy()
    if _is_portable_root(config.repo_root):
        return config.repo_root / "data" / wallet_policy.wallet_data_dirname
    return config.repo_root / wallet_policy.wallet_data_dirname


def peer_book_path_for_repo(repo_root: Path) -> Path:
    return repo_root / ".cai-peer-book.json"


def load_peer_book_for_repo(repo_root: Path) -> list[str]:
    path = peer_book_path_for_repo(repo_root)
    if not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"Peer book must contain a JSON list: {path}")

    peers = [item.strip() for item in data if isinstance(item, str) and item.strip()]
    return normalize_peers(peers)


def save_peer_book_for_repo(repo_root: Path, peers: Sequence[str]) -> Path:
    path = peer_book_path_for_repo(repo_root)
    path.write_text(
        json.dumps(normalize_peers(list(peers)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def sync_peer_book_for_repo_from_bootstrap(
    config: CaiDesktopConfig,
    *,
    timeout_sec: float = 2.0,
    max_state_urls: int = 4,
) -> tuple[Path, list[str], list[str]]:
    network_config = CaiNetworkConfig()
    source_peers = normalize_peers(
        [*network_config.bootstrap_peers, *load_peer_book_for_repo(config.repo_root)]
    )
    imported_total, tried_state_urls = discover_peer_book_peers(
        source_peers,
        network_config.default_api_port,
        max_state_urls=max_state_urls,
        timeout_sec=timeout_sec,
    )

    normalized_imported = normalize_peers(imported_total)
    merged_peers = normalize_peers(
        [*load_peer_book_for_repo(config.repo_root), *normalized_imported]
    )
    path = save_peer_book_for_repo(config.repo_root, merged_peers)
    return path, normalized_imported, tried_state_urls


def ensure_desktop_firewall(config: CaiDesktopConfig) -> WindowsFirewallRuleResult | None:
    if not sys.platform.startswith("win"):
        return None
    return ensure_windows_firewall_rule(
        [config.api_port, config.libp2p_port],
        script_dir=_desktop_state_root(),
    )


def build_cai_desktop_env(config: CaiDesktopConfig) -> dict[str, str]:
    env = dict(os.environ)
    repo_root = config.repo_root
    bundle_root = _bundle_root()
    runtime_src = repo_root / "cai" / "src"
    cai_src = repo_root / "src"
    network_config = CaiNetworkConfig()
    network_model_policy = NetworkModelPolicy()

    _set_compat_env(
        env,
        "CAI_LIBP2P_NAMESPACE",
        network_config.namespace,
        legacy_key="EXO_LIBP2P_NAMESPACE",
    )
    _set_compat_env(
        env,
        "CAI_HOME",
        str(cai_home_path(config)),
        legacy_key="CAI_HOME",
    )
    env["CAI_WALLET_HOME"] = str(wallet_home_path(config))
    env["CAI_REPO_ROOT"] = str(repo_root)
    _set_compat_env(env, "CAI_RUNTIME_REPO", str(repo_root))
    _set_compat_env(env, "CAI_RUNTIME_SRC", str(cai_src))
    _set_compat_env(
        env,
        "CAI_DEFAULT_MODELS_DIR",
        str(cai_home_path(config) / "models"),
        legacy_key="CAI_DEFAULT_MODELS_DIR",
    )
    disconnect_grace_seconds = _desktop_disconnect_grace_seconds(config, env)
    _set_compat_env(
        env,
        "CAI_CONNECTION_DISCONNECT_GRACE_SECONDS",
        disconnect_grace_seconds,
        legacy_key="CAI_CONNECTION_DISCONNECT_GRACE_SECONDS",
    )
    session_transition_debounce_seconds = _desktop_session_transition_debounce_seconds(
        config, env
    )
    _set_compat_env(
        env,
        "CAI_SESSION_TRANSITION_DEBOUNCE_SECONDS",
        session_transition_debounce_seconds,
        legacy_key="CAI_SESSION_TRANSITION_DEBOUNCE_SECONDS",
    )
    if sys.platform.startswith("win"):
        # NVIDIA's CLI helper can still create a visible conhost on some Windows
        # systems even when spawned with hidden/no-window flags. The VRAM summary
        # it feeds is non-critical for runtime placement, so desktop builds prefer
        # stable UX over this optional metric.
        _set_compat_env(
            env,
            "CAI_DISABLE_NVIDIA_SMI_VRAM_PROBE",
            "1",
            legacy_key="CAI_DISABLE_NVIDIA_SMI_VRAM_PROBE",
        )
    _set_compat_env(
        env,
        "CAI_ALLOWED_INFERENCE_BACKENDS",
        "llama_cpp",
        legacy_key="CAI_ALLOWED_INFERENCE_BACKENDS",
    )
    _set_compat_env(env, "CAI_NO_BATCH", "1", legacy_key="CAI_NO_BATCH")
    env["PYTHONPATH"] = _append_path(env.get("PYTHONPATH"), [runtime_src, cai_src])
    env["CAI_LANG"] = resolve_language(config.language)
    env.pop("CAI_DISABLE_DASHBOARD", None)

    _set_compat_env(
        env,
        "CAI_PRIVATE_NETWORK_MODEL_IDS",
        ",".join(network_model_policy.private_runtime_model_ids),
        legacy_key="CAI_PRIVATE_NETWORK_MODEL_IDS",
    )
    _set_compat_env(
        env,
        "CAI_PRIVATE_NETWORK_MODEL_MIN_NODES",
        str(network_model_policy.minimum_worker_shards),
        legacy_key="CAI_PRIVATE_NETWORK_MODEL_MIN_NODES",
    )
    _set_compat_env(
        env,
        "CAI_PRIVATE_NETWORK_MODEL_MIN_RAM_HEADROOM_MB",
        str(network_model_policy.minimum_worker_ram_headroom_mb),
        legacy_key="CAI_PRIVATE_NETWORK_MODEL_MIN_RAM_HEADROOM_MB",
    )
    _set_compat_env(
        env,
        "CAI_PRIVATE_NETWORK_MODEL_MIN_PIPELINE_LAYERS_PER_NODE",
        str(network_model_policy.minimum_worker_pipeline_layers_per_node),
        legacy_key="CAI_PRIVATE_NETWORK_MODEL_MIN_PIPELINE_LAYERS_PER_NODE",
    )
    _set_compat_env(
        env,
        "CAI_PRIVATE_NETWORK_MODEL_REQUIRE_PIPELINE",
        "true",
        legacy_key="CAI_PRIVATE_NETWORK_MODEL_REQUIRE_PIPELINE",
    )
    _set_compat_env(
        env,
        "CAI_PRIVATE_NETWORK_MODEL_DISABLE_SINGLE_NODE",
        "true",
        legacy_key="CAI_PRIVATE_NETWORK_MODEL_DISABLE_SINGLE_NODE",
    )
    _set_compat_env(
        env,
        "CAI_PRIVATE_NETWORK_MODEL_FILTERED_DOWNLOADS",
        "true",
        legacy_key="CAI_PRIVATE_NETWORK_MODEL_FILTERED_DOWNLOADS",
    )
    set_cai_owned_task_level_env_defaults(
        env,
        config=network_config,
        network_model_policy=network_model_policy,
    )
    _set_cai_llm_shard_production_adapter_defaults(
        env,
        resolved_repo_root=repo_root,
    )

    user_models_dir = _user_models_dir()
    if user_models_dir is not None and user_models_dir.exists():
        readonly_dirs = _append_env_path(
            env.get("CAI_MODELS_READ_ONLY_DIRS") or env.get("CAI_MODELS_READ_ONLY_DIRS"),
            [user_models_dir],
        )
        _set_compat_env(
            env,
            "CAI_MODELS_READ_ONLY_DIRS",
            readonly_dirs,
            legacy_key="CAI_MODELS_READ_ONLY_DIRS",
        )

    llama_server = _first_existing(
        [
            *(  # Prefer the bundled single-exe runtime payload when present.
                [bundle_root / "llama.cpp" / "llama-server.exe"] if bundle_root is not None else []
            ),
            repo_root / "data" / "runtime" / "llama.cpp" / "llama-server.exe",
            repo_root / "data" / "runtime" / "llama.cpp" / "bin" / "llama-server.exe",
            repo_root / "runtime" / "llama.cpp" / "llama-server.exe",
            repo_root / "runtime" / "llama.cpp" / "bin" / "llama-server.exe",
            repo_root
            / "cai"
            / ".runtime"
            / "llama.cpp"
            / "windows"
            / "build"
            / "llama-server.exe",
            repo_root
            / "cai"
            / ".runtime"
            / "llama.cpp"
            / "windows"
            / "build"
            / "bin"
            / "llama-server.exe",
        ]
    )
    if llama_server is not None:
        _set_compat_env(
            env,
            "CAI_LLAMA_CPP_SERVER",
            str(llama_server),
            legacy_key="CAI_LLAMA_CPP_SERVER",
        )

    llama_rpc = _first_existing(
        [
            *(
                [bundle_root / "llama.cpp" / "rpc-server.exe"] if bundle_root is not None else []
            ),
            repo_root / "data" / "runtime" / "llama.cpp" / "rpc-server.exe",
            repo_root / "data" / "runtime" / "llama.cpp" / "bin" / "rpc-server.exe",
            repo_root / "runtime" / "llama.cpp" / "rpc-server.exe",
            repo_root / "runtime" / "llama.cpp" / "bin" / "rpc-server.exe",
            repo_root
            / "cai"
            / ".runtime"
            / "llama.cpp"
            / "windows"
            / "build"
            / "rpc-server.exe",
            repo_root
            / "cai"
            / ".runtime"
            / "llama.cpp"
            / "windows"
            / "build"
            / "bin"
            / "rpc-server.exe",
        ]
    )
    if llama_rpc is not None:
        _set_compat_env(
            env,
            "CAI_LLAMA_CPP_RPC_SERVER",
            str(llama_rpc),
            legacy_key="CAI_LLAMA_CPP_RPC_SERVER",
        )

    return env


def build_cai_desktop_command(config: CaiDesktopConfig) -> list[str]:
    runtime_flags = _build_runtime_flags(config)
    if _is_embedded_portable_app():
        return [str(Path(sys.executable).resolve()), CAI_INTERNAL_RUNTIME_FLAG, *runtime_flags]

    bundled_runtime = _bundled_runtime_executable(config)
    if bundled_runtime is not None:
        return [str(bundled_runtime), *runtime_flags]

    return [
        str(_python_executable_for_runtime(config)),
        str(config.repo_root / "tools" / "run-cai-main.py"),
        *runtime_flags,
    ]


def _prepare_embedded_runtime_env() -> None:
    try:
        repo_root = resolve_repo_root(None)
    except Exception:
        return

    config = CaiDesktopConfig(
        repo_root=repo_root,
        open_browser=False,
        start_on_launch=False,
    )
    defaults = build_cai_desktop_env(config)
    for key, value in defaults.items():
        if key == "PYTHONPATH":
            os.environ[key] = value
            continue
        os.environ.setdefault(key, value)
    for path in (repo_root / "cai" / "src", repo_root / "src"):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def run_embedded_runtime(runtime_args: Sequence[str]) -> int:
    _prepare_embedded_runtime_env()
    from cai.main import main as runtime_main

    previous_argv = sys.argv[:]
    forwarded_args = [arg for arg in runtime_args if arg != CAI_INTERNAL_RUNTIME_FLAG]
    sys.argv = [str(Path(sys.executable).resolve()), *forwarded_args]
    try:
        runtime_main()
    finally:
        sys.argv = previous_argv
    return 0


def _portable_auto_update_interval_seconds() -> int:
    return _env_positive_int(
        AUTO_UPDATE_INTERVAL_ENV,
        DEFAULT_AUTO_UPDATE_INTERVAL_SECONDS,
    )


def _portable_auto_update_error_interval_seconds() -> int:
    return _env_positive_int(
        AUTO_UPDATE_ERROR_INTERVAL_ENV,
        DEFAULT_AUTO_UPDATE_ERROR_INTERVAL_SECONDS,
    )


def _maybe_schedule_embedded_portable_auto_update(
    config: CaiDesktopConfig,
    raw_args: Sequence[str],
) -> dict[str, object] | None:
    if not _is_embedded_portable_app():
        return None
    if _env_flag("CAI_PORTABLE_UPDATE_RESTARTED"):
        return None
    try:
        from cai_compute_chain.update_channel import (
            auto_update_check_timeout_seconds,
            auto_update_idle_seconds,
            auto_update_idle_timeout_seconds,
            maybe_stage_portable_auto_update_on_launch,
            record_portable_update_activity,
        )
    except Exception:
        return None

    try:
        record_portable_update_activity(
            config.repo_root,
            source="desktop",
            user_active=True,
            last_user_activity_at=datetime.now(tz=UTC).isoformat(),
            metadata={"reason": "desktop-startup"},
        )
    except Exception:
        pass

    def _run_auto_update_monitor() -> None:
        initial_delay = _env_positive_int(AUTO_UPDATE_INITIAL_DELAY_ENV, 1)
        interval_seconds = _portable_auto_update_interval_seconds()
        error_interval_seconds = _portable_auto_update_error_interval_seconds()
        if initial_delay > 0:
            time.sleep(initial_delay)
        while True:
            try:
                result = maybe_stage_portable_auto_update_on_launch(
                    config.repo_root,
                    relaunch_command=[str(Path(sys.executable).resolve()), *raw_args],
                    parent_pid=os.getpid(),
                    timeout_sec=auto_update_check_timeout_seconds(),
                    idle_seconds=auto_update_idle_seconds(),
                    idle_timeout_sec=auto_update_idle_timeout_seconds(),
                )
            except Exception:
                time.sleep(error_interval_seconds)
                continue
            if result.get("restartScheduled"):
                message = str(
                    result.get("message")
                    or "CAI portable update downloaded; applying it automatically."
                )
                print(message)
                return
            if str(result.get("status") or "").strip().lower() == "error":
                time.sleep(error_interval_seconds)
                continue
            time.sleep(interval_seconds)

    thread = threading.Thread(
        target=_run_auto_update_monitor,
        name="cai-portable-auto-update-monitor",
        daemon=True,
    )
    thread.start()
    return {
        "started": True,
        "message": "CAI portable update check started in the background.",
    }


def _maybe_resume_embedded_portable_auto_update(
    config: CaiDesktopConfig,
    raw_args: Sequence[str],
) -> dict[str, object] | None:
    if not _is_embedded_portable_app():
        return None
    if _env_flag("CAI_PORTABLE_UPDATE_RESTARTED"):
        return None
    try:
        from cai_compute_chain.update_channel import (
            resume_pending_portable_update_on_launch,
        )
    except Exception:
        return None
    try:
        return resume_pending_portable_update_on_launch(
            config.repo_root,
            relaunch_command=[str(Path(sys.executable).resolve()), *raw_args],
            parent_pid=os.getpid(),
        )
    except Exception:
        return None


def launch_config_payload(config: CaiDesktopConfig) -> dict[str, object]:
    return {
        "repoRoot": str(config.repo_root),
        "dashboardUrl": f"http://127.0.0.1:{config.api_port}/",
        "caiHome": str(cai_home_path(config)),
        "command": build_cai_desktop_command(config),
    }


def runtime_doctor_script_path(config: CaiDesktopConfig) -> Path | None:
    candidates = [
        *(
            [_bundle_root() / "install-runtime-deps-win.ps1"]
            if _bundle_root() is not None
            else []
        ),
        config.repo_root / "install-runtime-deps-win.ps1",
        config.repo_root / "tools" / "install-runtime-deps-win.ps1",
    ]
    return _first_existing(candidates)


def runtime_doctor_command(
    config: CaiDesktopConfig,
    *,
    install: bool,
    pause: bool,
    open_driver_page: bool = False,
) -> list[str]:
    script_path = runtime_doctor_script_path(config)
    if script_path is None:
        raise FileNotFoundError("Cannot find CAI runtime dependency installer script.")
    bundle_root = _bundle_root()

    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script_path),
        "-RepoRoot",
        str(config.repo_root),
    ]
    if bundle_root is not None:
        command.extend(["-BundleRoot", str(bundle_root)])
    if install:
        command.append("-Install")
    if open_driver_page:
        command.append("-OpenDriverPage")
    if pause:
        command.append("-Pause")
    return command


def run_runtime_doctor(
    config: CaiDesktopConfig,
    *,
    install: bool,
    pause: bool,
    open_driver_page: bool = False,
) -> int:
    return subprocess.call(
        runtime_doctor_command(
            config,
            install=install,
            pause=pause,
            open_driver_page=open_driver_page,
        ),
        cwd=str(config.repo_root),
    )


def resolve_language(language: str = "auto") -> str:
    requested = (language or "auto").lower()
    if requested in {"en", "ru"}:
        return requested
    env_language = os.getenv("CAI_LANG", "").lower()
    if env_language in {"en", "ru"}:
        return env_language
    stored_language = _stored_desktop_language()
    if stored_language in {"en", "ru"}:
        return stored_language
    return DEFAULT_DESKTOP_LANGUAGE


DESKTOP_TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "open_dashboard": "Open dashboard",
        "start_node": "Start node",
        "stop_node": "Stop node",
        "restart_node": "Restart node",
        "check_runtime": "Check/install requirements",
        "language": "Language",
        "language_english": "English",
        "language_russian": "Russian",
        "quit": "Quit CAI",
        "already_running": "CAI is already running on this system.",
        "opening_existing_dashboard": "Opening the active CAI dashboard: {url}",
        "console_fallback": "Falling back to console mode.",
        "tray_missing": (
            "pystray is required for tray mode. "
            "Use --no-tray or install cai[desktop]."
        ),
        "dashboard": "CAI dashboard",
        "press_ctrl_c": "Press Ctrl+C to stop.",
    },
    "ru": {
        "open_dashboard": "Открыть панель",
        "start_node": "Запустить узел",
        "stop_node": "Остановить узел",
        "restart_node": "Перезапустить узел",
        "check_runtime": "Проверить зависимости",
        "language": "Язык",
        "language_english": "Английский",
        "language_russian": "Русский",
        "quit": "Выйти из CAI",
        "already_running": "CAI уже запущен на этой системе.",
        "opening_existing_dashboard": "Открываю активную панель CAI: {url}",
        "console_fallback": "Перехожу в консольный режим.",
        "tray_missing": (
            "Для режима трея нужен pystray. "
            "Используйте --no-tray или установите cai[desktop]."
        ),
        "dashboard": "Панель CAI",
        "press_ctrl_c": "Нажмите Ctrl+C для остановки.",
    },
}


def desktop_text(config: CaiDesktopConfig, key: str) -> str:
    language = resolve_language(config.language)
    return DESKTOP_TRANSLATIONS[language].get(
        key, DESKTOP_TRANSLATIONS["en"].get(key, key)
    )


def handle_existing_instance(config: CaiDesktopConfig) -> int:
    print(desktop_text(config, "already_running"))
    instance_state = _load_running_instance_state()
    dashboard_url = str(
        instance_state.get("dashboardUrl") or f"http://127.0.0.1:{config.api_port}/"
    )
    if config.open_browser and dashboard_url:
        print(
            desktop_text(config, "opening_existing_dashboard").format(
                url=dashboard_url
            )
        )
        webbrowser.open(dashboard_url)
    return 0


class CaiDesktopController:
    def __init__(self, config: CaiDesktopConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._log_handle: BinaryIO | None = None

    @property
    def dashboard_url(self) -> str:
        return f"http://127.0.0.1:{self.config.api_port}/"

    @property
    def running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def _open_log(self) -> BinaryIO:
        log_dir = cai_home_path(self.config)
        log_dir.mkdir(parents=True, exist_ok=True)
        return (log_dir / "desktop.log").open("ab")

    def _close_log(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def start(self) -> None:
        with self._lock:
            if self.running:
                return
            firewall_result = ensure_desktop_firewall(self.config)
            if firewall_result is not None and firewall_result.status not in {
                "already_configured",
                "configured",
                "uac_requested",
                "skipped",
                "disabled",
            }:
                print(f"CAI firewall setup warning: {firewall_result.message}")
            if not self.config.offline:
                try:
                    sync_peer_book_for_repo_from_bootstrap(self.config)
                except Exception:
                    pass
            command = build_cai_desktop_command(self.config)
            env = build_cai_desktop_env(self.config)
            if _is_onefile_embedded_portable_app():
                # Independent self-spawn is required for one-file PyInstaller apps.
                env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
            self._log_handle = self._open_log()
            try:
                self._process = subprocess.Popen(
                    command,
                    cwd=str(self.config.repo_root),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=self._log_handle,
                    stderr=subprocess.STDOUT,
                    creationflags=_windows_popen_flags(new_process_group=True),
                    startupinfo=_windows_hidden_startupinfo(),
                )
            except Exception:
                self._close_log()
                raise

    def stop(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
        if process is None or process.poll() is not None:
            self._close_log()
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        finally:
            self._close_log()

    def restart(self) -> None:
        self.stop()
        self.start()

    def open_dashboard(self) -> None:
        webbrowser.open(self.dashboard_url)

    def open_runtime_doctor(self) -> None:
        command = runtime_doctor_command(
            self.config,
            install=True,
            open_driver_page=True,
            pause=True,
        )
        subprocess.Popen(
            command,
            cwd=str(self.config.repo_root),
            creationflags=_windows_popen_flags(new_console=True),
        )


def _desktop_icon_source_candidates(repo_root: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    env_icon = os.getenv("CAI_ICON_PATH", "").strip()
    if env_icon:
        candidates.append(Path(env_icon))

    for root in (repo_root, _bundle_root(), _portable_executable_root()):
        if root is None:
            continue
        candidates.extend(
            [
                root / "assets" / "cai.ico",
                root / "icon.ico",
                root / "cai" / "dashboard" / "static" / "cai-favicon.png",
                root / "dashboard" / "cai-favicon.png",
            ]
        )

    source_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            source_root / "icon.ico",
            source_root / "cai" / "dashboard" / "static" / "cai-favicon.png",
        ]
    )
    return candidates


def _coerce_square_icon_image(image, size: int):
    from PIL import Image

    rgba = image.convert("RGBA")
    width, height = rgba.size
    side = max(width, height)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.alpha_composite(rgba, ((side - width) // 2, (side - height) // 2))
    if side == size:
        return square
    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)
    return square.resize((size, size), resampling)


def _load_desktop_icon_image(repo_root: Path | None = None, *, size: int = 64):
    from PIL import Image

    for candidate in _desktop_icon_source_candidates(repo_root):
        try:
            path = candidate.expanduser().resolve()
        except OSError:
            continue
        if not path.is_file():
            continue
        try:
            with Image.open(path) as opened:
                if path.suffix.lower() == ".ico" and hasattr(opened, "ico"):
                    icon_sizes = sorted(opened.ico.sizes())
                    if icon_sizes:
                        chosen_size = max(icon_sizes, key=lambda item: item[0] * item[1])
                        opened = opened.ico.getimage(chosen_size)
                return _coerce_square_icon_image(opened, size)
        except OSError:
            continue

    image = Image.new("RGBA", (size, size), (5, 12, 18, 255))
    return image


def _make_tray_icon_image(repo_root: Path | None = None):
    return _load_desktop_icon_image(repo_root, size=64)


def _desktop_icon_frame(image, size: int):
    from PIL import ImageFilter

    frame = _coerce_square_icon_image(image, size)
    if size <= 64:
        frame = frame.filter(ImageFilter.UnsharpMask(radius=0.7, percent=180, threshold=2))
    return frame


def write_desktop_icon(path: Path, repo_root: Path | None = None) -> Path:
    image = _load_desktop_icon_image(repo_root, size=256)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [_desktop_icon_frame(image, size[0]) for size in DESKTOP_ICON_SIZES]
    frames[-1].save(
        path,
        format="ICO",
        append_images=frames[:-1],
        sizes=list(DESKTOP_ICON_SIZES),
    )
    return path


def run_tray(controller: CaiDesktopController) -> int:
    try:
        import pystray
    except ImportError as exc:
        raise RuntimeError(desktop_text(controller.config, "tray_missing")) from exc

    def _safe(action):
        def _wrapped(_icon=None, _item=None):
            action()

        return _wrapped

    def _quit(icon, _item=None):
        controller.stop()
        icon.stop()

    def _current_menu_language() -> str:
        return resolve_language(controller.config.language)

    def _set_language(icon, language: str) -> None:
        controller.config.language = language
        save_desktop_language(language)
        icon.menu = _build_menu()
        icon.update_menu()

    def _build_menu():
        return pystray.Menu(
            pystray.MenuItem(
                lambda item: desktop_text(controller.config, "open_dashboard"),
                _safe(controller.open_dashboard),
            ),
            pystray.MenuItem(
                lambda item: desktop_text(controller.config, "start_node"),
                _safe(controller.start),
            ),
            pystray.MenuItem(
                lambda item: desktop_text(controller.config, "stop_node"),
                _safe(controller.stop),
            ),
            pystray.MenuItem(
                lambda item: desktop_text(controller.config, "restart_node"),
                _safe(controller.restart),
            ),
            pystray.MenuItem(
                lambda item: desktop_text(controller.config, "check_runtime"),
                _safe(controller.open_runtime_doctor),
            ),
            pystray.MenuItem(
                lambda item: desktop_text(controller.config, "language"),
                pystray.Menu(
                    pystray.MenuItem(
                        lambda item: desktop_text(controller.config, "language_english"),
                        lambda icon, item: _set_language(icon, "en"),
                        radio=True,
                        checked=lambda item: _current_menu_language() == "en",
                    ),
                    pystray.MenuItem(
                        lambda item: desktop_text(controller.config, "language_russian"),
                        lambda icon, item: _set_language(icon, "ru"),
                        radio=True,
                        checked=lambda item: _current_menu_language() == "ru",
                    ),
                ),
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                lambda item: desktop_text(controller.config, "quit"),
                _quit,
            ),
        )

    icon = pystray.Icon(
        "CAI",
        _make_tray_icon_image(controller.config.repo_root),
        "CAI",
        _build_menu(),
    )
    if controller.config.start_on_launch:
        controller.start()
    if controller.config.open_browser:
        controller.open_dashboard()
    icon.run()
    return 0


def run_console(controller: CaiDesktopController) -> int:
    if controller.config.start_on_launch:
        controller.start()
    if controller.config.open_browser:
        controller.open_dashboard()
    print(f"{desktop_text(controller.config, 'dashboard')}: {controller.dashboard_url}")
    if not controller.config.start_on_launch:
        return 0
    print(desktop_text(controller.config, "press_ctrl_c"))
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cai-desktop")
    parser.add_argument("--repo-root")
    parser.add_argument("--api-port", type=int, default=DEFAULT_DESKTOP_API_PORT)
    parser.add_argument("--libp2p-port", type=int, default=DEFAULT_DESKTOP_LIBP2P_PORT)
    parser.add_argument("--python", dest="python_executable")
    parser.add_argument("--cai-home")
    parser.add_argument("--no-downloads", action="store_true")
    parser.add_argument("--no-worker", action="store_true")
    parser.add_argument("--force-master", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--language",
        choices=["auto", "en", "ru"],
        default=DEFAULT_DESKTOP_LANGUAGE,
    )
    parser.add_argument(CAI_INTERNAL_RUNTIME_FLAG, action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-start", action="store_true")
    parser.add_argument("--no-tray", action="store_true")
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Check local runtime dependencies and exit.",
    )
    parser.add_argument(
        "--install-missing",
        action="store_true",
        help="Install supported missing runtime dependencies when used with --doctor.",
    )
    parser.add_argument(
        "--pause",
        action="store_true",
        help="Wait for Enter before closing when used with --doctor.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print resolved CAI runtime launch config and exit.",
    )
    parser.add_argument(
        "--write-icon",
        help="Write the CAI tray icon as an .ico file and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    raw_argv = [sys.argv[0], *raw_args]

    if _maybe_run_inline_code(raw_argv):
        return 0
    freeze_support()

    if CAI_INTERNAL_RUNTIME_FLAG in raw_args:
        return run_embedded_runtime(raw_args)

    args = build_parser().parse_args(raw_args)
    repo_root = resolve_repo_root(args.repo_root)
    base_config = CaiDesktopConfig(
        repo_root=repo_root,
        api_port=args.api_port,
        libp2p_port=args.libp2p_port,
        python_executable=args.python_executable,
        cai_home=args.cai_home,
        no_downloads=args.no_downloads,
        no_worker=args.no_worker,
        force_master=args.force_master,
        offline=args.offline,
        verbose=args.verbose,
        open_browser=not args.no_browser,
        start_on_launch=not args.no_start,
        language=args.language,
    )
    if args.write_icon:
        write_desktop_icon(Path(args.write_icon), repo_root=base_config.repo_root)
        return 0
    if args.doctor:
        return run_runtime_doctor(
            base_config,
            install=args.install_missing,
            open_driver_page=args.install_missing,
            pause=args.pause,
        )
    if args.print_config:
        selected_api_port = _find_available_port(args.api_port, host="127.0.0.1")
        selected_libp2p_port = _find_available_port(
            args.libp2p_port,
            host="0.0.0.0",
            reserved={selected_api_port},
        )
        print(
            json.dumps(
                launch_config_payload(
                    CaiDesktopConfig(
                        **{
                            **base_config.__dict__,
                            "api_port": selected_api_port,
                            "libp2p_port": selected_libp2p_port,
                        }
                    )
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    enforce_single_instance = should_enforce_single_instance()
    instance_guard = DesktopSingleInstanceGuard()
    if enforce_single_instance and not instance_guard.acquire():
        return handle_existing_instance(base_config)
    selected_api_port = _find_available_port(args.api_port, host="127.0.0.1")
    selected_libp2p_port = _find_available_port(
        args.libp2p_port,
        host="0.0.0.0",
        reserved={selected_api_port},
    )
    config = CaiDesktopConfig(
        **{
            **base_config.__dict__,
            "api_port": selected_api_port,
            "libp2p_port": selected_libp2p_port,
        }
    )
    if enforce_single_instance:
        if args.language in {"en", "ru"}:
            save_desktop_language(args.language)
        instance_guard.write_state(config)

    controller = CaiDesktopController(config)
    try:
        if enforce_single_instance:
            _lock_wallet_session_best_effort()
        resumed_update = _maybe_resume_embedded_portable_auto_update(
            config,
            raw_args,
        )
        if resumed_update is not None and resumed_update.get("restartScheduled"):
            message = str(
                resumed_update.get("message")
                or "CAI portable update is already downloaded; applying it now."
            )
            print(message)
            return 0
        scheduled_update = _maybe_schedule_embedded_portable_auto_update(
            config,
            raw_args,
        )
        if scheduled_update is not None:
            message = str(
                scheduled_update.get("message")
                or "CAI portable update check started in the background."
            )
            print(message)
        if args.no_tray:
            return run_console(controller)
        try:
            return run_tray(controller)
        except RuntimeError as exc:
            print(f"{exc} {desktop_text(config, 'console_fallback')}")
            return run_console(controller)
    finally:
        if enforce_single_instance:
            _lock_wallet_session_best_effort()
            instance_guard.clear_state()
            instance_guard.release()


if __name__ == "__main__":
    raise SystemExit(main())



# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

from .model import CaiLaunchPlan, CaiNetworkConfig, NetworkModelPolicy


def _set_compat_env(
    env: dict[str, str], cai_key: str, value: str, *, legacy_key: str | None = None
) -> None:
    env[cai_key] = value
    if legacy_key and legacy_key != cai_key:
        env[legacy_key] = value


def _prepend_env_path(env: dict[str, str], key: str, value: str) -> None:
    clean_value = str(value or "").strip()
    if not clean_value:
        return
    existing = str(env.get(key) or "").strip()
    parts = [item for item in existing.split(os.pathsep) if item]
    if clean_value not in parts:
        parts.insert(0, clean_value)
    env[key] = os.pathsep.join(parts)


def repo_root() -> Path:
    configured = str(
        os.getenv("CAI_REPO_ROOT")
        or os.getenv("CAI_RUNTIME_REPO")
        or ""
    ).strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def cai_runtime_candidates() -> tuple[Path, ...]:
    root = repo_root()
    return (root / "cai",)


def peer_book_path() -> Path:
    return repo_root() / ".cai-peer-book.json"


def default_cai_launch_plan() -> CaiLaunchPlan:
    candidates: list[Path] = []
    for runtime_root in cai_runtime_candidates():
        candidates.extend(
            [
                runtime_root / ".venv-win" / "Scripts" / "cai.exe",
                runtime_root / ".venv" / "bin" / "cai",
            ]
        )
    return CaiLaunchPlan(executable_candidates=tuple(candidates))


def resolve_cai_runtime_command(explicit_executable: str | None) -> list[str]:
    if explicit_executable:
        return [explicit_executable]

    launch_plan = default_cai_launch_plan()
    env_executable = (
        os.getenv(launch_plan.env_var_name)
        or os.getenv("CAI_EXECUTABLE")
    )
    if env_executable:
        return [env_executable]

    for candidate in default_cai_launch_plan().executable_candidates:
        if candidate.exists():
            return [str(candidate)]

    cai_path = shutil.which("cai")
    if cai_path:
        return [cai_path]

    uv_path = shutil.which("uv")
    if uv_path:
        return [uv_path, "run", "cai"]

    raise FileNotFoundError(
        "Could not find the CAI runtime. Set --cai-executable or the CAI_RUNTIME_EXECUTABLE environment variable."
    )


def normalize_peers(peers: list[str]) -> list[str]:
    return list(dict.fromkeys(peer.strip() for peer in peers if peer.strip()))


def bootstrap_peers_argument(peers: list[str]) -> str:
    return ",".join(normalize_peers(peers))


def set_cai_owned_task_level_env_defaults(
    env: dict[str, str],
    *,
    config: CaiNetworkConfig,
    network_model_policy: NetworkModelPolicy | None = None,
) -> None:
    def set_default(name: str, value: object) -> None:
        if str(env.get(name) or "").strip():
            return
        env[name] = str(value)

    # User-facing inference must stay on the metered CAI job path so receipt,
    # settlement, and rewards are produced even when one executor is enough for
    # the selected model. Multi-executor routes are still selected by the
    # transport planner when policy and live readiness require them.
    set_default("CAI_ENABLE_TASK_LEVEL_TRANSPORT_JOBS", "1")
    set_default("CAI_REQUIRE_TASK_LEVEL_TRANSPORT_JOBS", "1")
    set_default("CAI_ALLOW_TASK_LEVEL_TRANSPORT_PRIVATE_MODELS", "1")
    set_default("CAI_TASK_LEVEL_TRANSPORT_REQUIRE_RUNTIME_READY", "1")
    set_default("CAI_TASK_LEVEL_TRANSPORT_REQUIRE_SHARD_READINESS", "1")
    set_default("CAI_TASK_LEVEL_TRANSPORT_REQUIRE_DATA_PLANE_ROUTE", "1")
    set_default("CAI_TASK_LEVEL_TRANSPORT_REQUIRE_PROVEN_DATA_PLANE_ROUTE", "1")
    set_default("CAI_TASK_LEVEL_TRANSPORT_EXECUTOR_COUNT", "1")
    set_default("CAI_TASK_LEVEL_TRANSPORT_TIMEOUT_SEC", "25")
    set_default("CAI_TASK_LEVEL_TRANSPORT_WAIT_TIMEOUT_SEC", "300")
    set_default("CAI_OWNED_TRANSPORT_GENERATION_ENABLED", "1")
    set_default("CAI_OWNED_TRANSPORT_GENERATION_REQUIRED", "1")
    set_default("CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_EXECUTOR_READINESS", "1")
    set_default("CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_RUNTIME_READY", "1")
    set_default("CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_SHARD_READINESS", "1")
    set_default("CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_DATA_PLANE_ROUTE", "1")
    set_default("CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_PROVEN_DATA_PLANE_ROUTE", "1")
    set_default("CAI_OWNED_TRANSPORT_STARTUP_SELF_TEST", "1")
    set_default("CAI_OWNED_TRANSPORT_REQUIRE_LIVE_PROOF", "0")
    # The classic llama.cpp route can still hang after an instance reports ready.
    # Keep the user-facing job moving to the next route instead of spending the
    # whole attempt budget on a silent response socket.
    set_default("CAI_JOB_EXECUTION_FIRST_RESPONSE_TIMEOUT_SECONDS", "120")
    set_default("CAI_CHAT_COMPLETION_TIMEOUT_SECONDS", "360")


def _first_existing_path(candidates: Sequence[Path]) -> Path | None:
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        if resolved.is_file():
            return resolved
    return None


def _inline_module_command(module_name: str) -> str:
    code = (
        f"from {module_name} import main; "
        "raise SystemExit(main())"
    )
    return " ".join([shlex.quote(str(Path(sys.executable).resolve())), "-c", shlex.quote(code)])


def _set_cai_llm_shard_production_adapter_defaults(
    env: dict[str, str],
    *,
    resolved_repo_root: Path,
) -> None:
    def set_default(name: str, value: object) -> None:
        if str(env.get(name) or "").strip():
            return
        env[name] = str(value)

    engine_path = _first_existing_path(
        (
            resolved_repo_root
            / "_internal"
            / "llama.cpp"
            / "llama-cai-shard-engine.exe",
            resolved_repo_root
            / "llama.cpp"
            / "llama-cai-shard-engine.exe",
            resolved_repo_root
            / "cai"
            / ".runtime"
            / "llama.cpp"
            / "windows-patched"
            / "build"
            / "bin"
            / "Release"
            / "llama-cai-shard-engine.exe",
        )
    )
    if engine_path is None:
        return

    set_default("CAI_LLM_SHARD_ADAPTER", "native_bridge")
    set_default("CAI_LLM_SHARD_ADAPTER_TIMEOUT_SEC", "900")
    set_default("CAI_LLM_SHARD_NATIVE_TIMEOUT_SEC", "900")
    set_default(
        "CAI_LLM_SHARD_NATIVE_COMMAND",
        _inline_module_command(
            "cai_compute_chain.cai_llama_cpp_assignment_artifact_engine"
        ),
    )
    set_default(
        "CAI_LLM_SHARD_ASSIGNMENT_EXECUTOR_COMMAND",
        _inline_module_command(
            "cai_compute_chain.cai_llama_cpp_patched_executor_host"
        )
        + " --jsonl",
    )
    set_default("CAI_LLM_SHARD_ASSIGNMENT_EXECUTOR_PERSISTENT", "1")
    set_default(
        "CAI_LLM_SHARD_PATCHED_ENGINE_COMMAND",
        _inline_module_command(
            "cai_compute_chain.cai_llama_cpp_patched_binary_executor"
        )
        + " --jsonl",
    )
    set_default("CAI_LLM_SHARD_PATCHED_ENGINE_PERSISTENT", "1")
    set_default(
        "CAI_LLM_PATCHED_BINARY_COMMAND",
        f"{shlex.quote(str(engine_path))} --jsonl",
    )
    set_default("CAI_LLM_PATCHED_BINARY_PERSISTENT", "1")
    set_default("CAI_LLM_PATCHED_BINARY_REQUIRE_REAL_LAYER_EXECUTION", "1")
    set_default("CAI_LLM_PATCHED_BINARY_REQUIRE_SHARD_ONLY_LOADING", "1")


def build_cai_runtime_env(
    cai_home: str | None,
    config: CaiNetworkConfig,
    advertise_peers: list[str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ)
    network_model_policy = NetworkModelPolicy()
    _set_compat_env(
        env,
        "CAI_LIBP2P_NAMESPACE",
        config.namespace,
        legacy_key="EXO_LIBP2P_NAMESPACE",
    )
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
        config=config,
        network_model_policy=network_model_policy,
    )
    _set_compat_env(
        env,
        "CAI_ALLOWED_INFERENCE_BACKENDS",
        "llama_cpp",
        legacy_key="CAI_ALLOWED_INFERENCE_BACKENDS",
    )
    if not env.get("CAI_NO_BATCH"):
        _set_compat_env(env, "CAI_NO_BATCH", "1", legacy_key="CAI_NO_BATCH")
    resolved_repo_root = repo_root()
    env["CAI_REPO_ROOT"] = str(resolved_repo_root)
    _set_compat_env(
        env, "CAI_RUNTIME_REPO", str(resolved_repo_root)
    )
    runtime_src = str(resolved_repo_root / "src")
    env["CAI_RUNTIME_SRC"] = runtime_src
    _prepend_env_path(env, "PYTHONPATH", runtime_src)
    _set_cai_llm_shard_production_adapter_defaults(
        env,
        resolved_repo_root=resolved_repo_root,
    )
    resolved_home = (
        cai_home
        or os.getenv("CAI_HOME")
        or str(resolved_repo_root / config.default_cai_home_dirname)
    )
    _set_compat_env(env, "CAI_HOME", resolved_home, legacy_key="CAI_HOME")

    normalized_advertise_peers = normalize_peers(advertise_peers or [])
    if normalized_advertise_peers:
        advertise_value = "\n".join(normalized_advertise_peers)
        _set_compat_env(
            env,
            config.advertise_env_var_name,
            advertise_value,
            legacy_key="CAI_ADVERTISE_PEERS",
        )

    return env


def build_cai_runtime_command(
    *,
    runtime_command: list[str],
    config: CaiNetworkConfig,
    api_port: int,
    libp2p_port: int,
    verbose: bool,
    no_downloads: bool,
    no_worker: bool,
    force_master: bool,
    offline: bool,
) -> list[str]:
    command = list(runtime_command)
    if verbose:
        command.append("-v")
    if force_master:
        command.append("-m")
    command.extend(
        [
            "--api-port",
            str(api_port),
            "--libp2p-port",
            str(libp2p_port),
        ]
    )
    if no_downloads:
        command.append("--no-downloads")
    if no_worker:
        command.append("--no-worker")
    if offline:
        command.append("--offline")
    peer_book_peers = load_peer_book()
    bootstrap_peers = normalize_peers([*config.bootstrap_peers, *peer_book_peers])
    if bootstrap_peers:
        command.append("--bootstrap-peers")
        command.append(bootstrap_peers_argument(bootstrap_peers))
    return command


def shell_render(command: list[str]) -> str:
    return subprocess.list2cmdline(command) if os.name == "nt" else " ".join(command)


def launch_cai_runtime(
    *,
    cai_executable: str | None,
    cai_home: str | None,
    config: CaiNetworkConfig,
    api_port: int,
    libp2p_port: int,
    verbose: bool,
    no_downloads: bool,
    no_worker: bool,
    force_master: bool,
    offline: bool,
    dry_run: bool,
    advertise_peers: list[str] | None,
) -> int:
    runtime_command = resolve_cai_runtime_command(cai_executable)
    command = build_cai_runtime_command(
        runtime_command=runtime_command,
        config=config,
        api_port=api_port,
        libp2p_port=libp2p_port,
        verbose=verbose,
        no_downloads=no_downloads,
        no_worker=no_worker,
        force_master=force_master,
        offline=offline,
    )
    env = build_cai_runtime_env(cai_home, config, advertise_peers=advertise_peers)

    print(f"CAI_LIBP2P_NAMESPACE={env['CAI_LIBP2P_NAMESPACE']}")
    print(f"CAI_HOME={env.get('CAI_HOME', '')}")
    if config.advertise_env_var_name in env:
        print(f"{config.advertise_env_var_name}={env[config.advertise_env_var_name]}")
    print(shell_render(command))

    if dry_run:
        return 0

    completed = subprocess.run(command, env=env, check=False)
    return int(completed.returncode)


def running_on_windows() -> bool:
    return sys.platform.startswith("win")


def load_peer_book() -> list[str]:
    path = peer_book_path()
    if not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("Peer book must contain a JSON list of multiaddrs.")

    peers = []
    for item in data:
        if isinstance(item, str) and item.strip():
            peers.append(item.strip())
    return normalize_peers(peers)


def save_peer_book(peers: list[str]) -> Path:
    path = peer_book_path()
    normalized = normalize_peers(peers)
    path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def add_peer_to_book(peer: str) -> Path:
    peers = load_peer_book()
    peers.append(peer)
    return save_peer_book(peers)


def read_state_payload_from_url(
    state_url: str,
    *,
    timeout_sec: float | None = None,
) -> dict[str, object]:
    kwargs = {}
    if timeout_sec is not None:
        kwargs["timeout"] = timeout_sec
    with urllib.request.urlopen(state_url, **kwargs) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("State endpoint must return a JSON object.")
    return data


def extract_overlay_advertised_peers(state_payload: dict[str, object]) -> list[str]:
    advertised = state_payload.get("overlayAdvertisedPeers", {})
    if not isinstance(advertised, dict):
        return []

    peers: list[str] = []
    for values in advertised.values():
        if not isinstance(values, list):
            continue
        for entry in values:
            if isinstance(entry, dict):
                address = entry.get("address")
                if isinstance(address, str):
                    peers.append(address)
            elif isinstance(entry, str):
                peers.append(entry)

    return normalize_peers(peers)


def import_peer_book_from_state_url(state_url: str) -> tuple[Path, list[str]]:
    state_payload = read_state_payload_from_url(state_url)
    imported_peers = extract_overlay_advertised_peers(state_payload)
    merged_peers = normalize_peers([*load_peer_book(), *imported_peers])
    path = save_peer_book(merged_peers)
    return path, imported_peers


def state_url_from_multiaddr(peer: str, api_port: int) -> str | None:
    ip4_match = re.match(r"^/ip4/([^/]+)", peer)
    if ip4_match:
        return f"http://{ip4_match.group(1)}:{api_port}/state"

    ip6_match = re.match(r"^/ip6/([^/]+)", peer)
    if ip6_match:
        return f"http://[{ip6_match.group(1)}]:{api_port}/state"

    dns_match = re.match(r"^/dns(?:4|6)?/([^/]+)", peer)
    if dns_match:
        return f"http://{dns_match.group(1)}:{api_port}/state"

    return None


StatePayloadReader = Callable[[str], dict[str, object]]


def discover_peer_book_peers(
    source_peers: Sequence[str],
    api_port: int,
    *,
    max_state_urls: int = 16,
    timeout_sec: float | None = None,
    read_state_payload: StatePayloadReader | None = None,
) -> tuple[list[str], list[str]]:
    imported_total: list[str] = []
    imported_seen: set[str] = set()
    tried_state_urls: list[str] = []
    tried_state_url_set: set[str] = set()
    pending_peers = normalize_peers(list(source_peers))
    queued_peers = set(pending_peers)
    state_url_limit = max(0, int(max_state_urls))
    deadline = (
        time.monotonic() + max(timeout_sec, 0.1)
        if timeout_sec is not None
        else None
    )

    while pending_peers and len(tried_state_urls) < state_url_limit:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            read_timeout = max(0.25, remaining)
        else:
            read_timeout = None

        peer = pending_peers.pop(0)
        state_url = state_url_from_multiaddr(peer, api_port)
        if not state_url or state_url in tried_state_url_set:
            continue

        tried_state_urls.append(state_url)
        tried_state_url_set.add(state_url)
        try:
            if read_state_payload is not None:
                state_payload = read_state_payload(state_url)
            else:
                state_payload = read_state_payload_from_url(
                    state_url,
                    timeout_sec=read_timeout,
                )
        except Exception:
            continue

        for imported_peer in extract_overlay_advertised_peers(state_payload):
            if imported_peer not in imported_seen:
                imported_total.append(imported_peer)
                imported_seen.add(imported_peer)
            if imported_peer not in queued_peers:
                pending_peers.append(imported_peer)
                queued_peers.add(imported_peer)

    return normalize_peers(imported_total), tried_state_urls


def sync_peer_book_from_bootstrap(
    config: CaiNetworkConfig,
    *,
    max_state_urls: int = 16,
    timeout_sec: float | None = None,
) -> tuple[Path, list[str], list[str]]:
    source_peers = normalize_peers([*config.bootstrap_peers, *load_peer_book()])
    normalized_imported, tried_state_urls = discover_peer_book_peers(
        source_peers,
        config.default_api_port,
        max_state_urls=max_state_urls,
        timeout_sec=timeout_sec,
    )
    merged_peers = normalize_peers([*load_peer_book(), *normalized_imported])
    path = save_peer_book(merged_peers)
    return path, normalized_imported, tried_state_urls



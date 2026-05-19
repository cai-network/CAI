# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import os
import sys
from collections.abc import Callable
from pathlib import Path

from cai.utils.dashboard_path import find_dashboard, find_resources


def _expand_cai_home(raw_value: str | None) -> Path | None:
    if raw_value is None:
        return None
    path = Path(raw_value).expanduser()
    if path.is_absolute():
        return path
    return Path.home() / path


def _get_xdg_dir(env_var: str, fallback: str) -> Path:
    """Resolve a CAI data root, honoring CAI_HOME before platform defaults."""

    cai_home = _expand_cai_home(os.environ.get("CAI_HOME"))
    if cai_home is not None:
        return cai_home

    if sys.platform != "linux":
        return Path.home() / ".cai"

    xdg_value = os.environ.get(env_var)
    if xdg_value is not None:
        return Path(xdg_value).expanduser() / "cai"
    return Path.home() / fallback / "cai"


def _parse_colon_dirs(env_var: str) -> tuple[Path, ...]:
    raw = os.environ.get(env_var)
    if raw is None:
        return ()
    return tuple(Path(part).expanduser() for part in raw.split(os.pathsep) if part)


def _resolve_env_path(env_var: str, fallback_factory: Callable[[], Path]) -> Path:
    raw = os.environ.get(env_var)
    if raw is None:
        return fallback_factory()
    return _expand_cai_home(raw) or fallback_factory()


CAI_CONFIG_HOME = _get_xdg_dir("XDG_CONFIG_HOME", ".config")
CAI_DATA_HOME = _get_xdg_dir("XDG_DATA_HOME", ".local/share")
CAI_CACHE_HOME = _get_xdg_dir("XDG_CACHE_HOME", ".cache")

_CAI_DEFAULT_MODELS_DIR_ENV = os.environ.get("CAI_DEFAULT_MODELS_DIR")
CAI_DEFAULT_MODELS_DIR = (
    Path(_CAI_DEFAULT_MODELS_DIR_ENV).expanduser()
    if _CAI_DEFAULT_MODELS_DIR_ENV is not None
    else CAI_DATA_HOME / "models"
)

_CAI_MODELS_READ_ONLY_DIRS_ENV = _parse_colon_dirs("CAI_MODELS_READ_ONLY_DIRS")
_CAI_MODELS_DIRS_ENV = _parse_colon_dirs("CAI_MODELS_DIRS")
_READ_ONLY_SET = frozenset(_CAI_MODELS_READ_ONLY_DIRS_ENV)

CAI_MODELS_DIRS: tuple[Path, ...] = tuple(
    path
    for path in (CAI_DEFAULT_MODELS_DIR, *_CAI_MODELS_DIRS_ENV)
    if path not in _READ_ONLY_SET
)
CAI_MODELS_READ_ONLY_DIRS: tuple[Path, ...] = _CAI_MODELS_READ_ONLY_DIRS_ENV

RESOURCES_DIR = _resolve_env_path("CAI_RESOURCES_DIR", find_resources)
DASHBOARD_DIR = _resolve_env_path("CAI_DASHBOARD_DIR", find_dashboard)

CAI_LOG_DIR = CAI_CACHE_HOME / "cai_log"
CAI_LOG = CAI_LOG_DIR / "cai.log"
CAI_TEST_LOG = CAI_CACHE_HOME / "cai_test.log"

CAI_NODE_ID_KEYPAIR = CAI_CONFIG_HOME / "node_id.keypair"
CAI_CONFIG_FILE = CAI_CONFIG_HOME / "config.toml"

LIBP2P_LOCAL_EVENTS_TOPIC = "worker_events"
LIBP2P_GLOBAL_EVENTS_TOPIC = "global_events"
LIBP2P_ELECTION_MESSAGES_TOPIC = "election_message"
LIBP2P_COMMANDS_TOPIC = "commands"

CAI_MAX_CHUNK_SIZE = 512 * 1024

CAI_CUSTOM_MODEL_CARDS_DIR = CAI_DATA_HOME / "custom_model_cards"

CAI_EVENT_LOG_DIR = CAI_DATA_HOME / "event_log"
CAI_IMAGE_CACHE_DIR = CAI_CACHE_HOME / "images"
CAI_TRACING_CACHE_DIR = CAI_CACHE_HOME / "traces"

CAI_ENABLE_IMAGE_MODELS = os.getenv("CAI_ENABLE_IMAGE_MODELS", "false").lower() == "true"
CAI_OFFLINE = os.getenv("CAI_OFFLINE", "false").lower() == "true"
CAI_TRACING_ENABLED = os.getenv("CAI_TRACING_ENABLED", "false").lower() == "true"
CAI_MAX_CONCURRENT_REQUESTS = int(os.getenv("CAI_MAX_CONCURRENT_REQUESTS", "8"))

CAI_MAX_INSTANCE_RETRIES = 5

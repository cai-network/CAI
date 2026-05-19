# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
#!/usr/bin/env python3

import os
import shutil
import sys
from pathlib import Path


AUTO_UPDATE_RESTART_ENV = "CAI_AUTO_UPDATE_RESTARTED"


def _maybe_apply_cai_auto_update(repo_root: Path, cai_src: Path) -> None:
    if str(cai_src) not in sys.path:
        sys.path.insert(0, str(cai_src))

    try:
        from cai_compute_chain.update_channel import maybe_auto_update_on_launch
    except Exception:
        return

    try:
        result = maybe_auto_update_on_launch(repo_root)
    except Exception:
        return

    if result.get("updated"):
        message = str(result.get("message") or "CAI source update applied from validator.")
        print(message, file=sys.stderr)
        if os.environ.get(AUTO_UPDATE_RESTART_ENV) != "1":
            os.environ[AUTO_UPDATE_RESTART_ENV] = "1"
            os.execv(sys.executable, [sys.executable, *sys.argv])


def _copy_if_missing(source: Path, destination: Path) -> None:
    if destination.exists() or not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination)
        return
    shutil.copy2(source, destination)


def _migrate_legacy_runtime_home(cai_home_path: Path) -> None:
    legacy_config_root = Path.home() / ".config" / "cai"
    legacy_data_root = Path.home() / ".local" / "share" / "cai"

    _copy_if_missing(
        legacy_config_root / "node_id.keypair",
        cai_home_path / "node_id.keypair",
    )
    _copy_if_missing(
        legacy_config_root / "config.toml",
        cai_home_path / "config.toml",
    )
    _copy_if_missing(
        legacy_data_root / "event_log",
        cai_home_path / "event_log",
    )


def _bootstrap_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runtime_src = repo_root / "cai" / "src"
    cai_src = repo_root / "src"

    for path in (str(runtime_src), str(cai_src)):
        if path not in sys.path:
            sys.path.insert(0, path)

    _maybe_apply_cai_auto_update(repo_root, cai_src)

    try:
        from cai_compute_chain.model import CaiNetworkConfig

        default_namespace = CaiNetworkConfig().namespace
    except Exception:
        default_namespace = "cai-ai-net"
    configured_namespace = (
        os.environ.get("CAI_LIBP2P_NAMESPACE")
        or os.environ.get("EXO_LIBP2P_NAMESPACE")
        or default_namespace
    )

    os.environ.setdefault("CAI_REPO_ROOT", str(repo_root))
    os.environ.setdefault("CAI_RUNTIME_REPO", str(repo_root))
    os.environ.setdefault("CAI_RUNTIME_SRC", str(cai_src))
    os.environ.setdefault("CAI_LIBP2P_NAMESPACE", configured_namespace)
    os.environ.setdefault("EXO_LIBP2P_NAMESPACE", configured_namespace)
    os.environ.setdefault("CAI_ALLOWED_INFERENCE_BACKENDS", "llama_cpp")
    os.environ.setdefault("CAI_CONNECTION_DISCONNECT_GRACE_SECONDS", "30")

    cai_home = (
        os.environ.get("CAI_HOME")
        or str(repo_root / ".cai")
    )
    os.environ.setdefault("CAI_HOME", cai_home)
    cai_home_path = Path(cai_home).expanduser().resolve()
    _migrate_legacy_runtime_home(cai_home_path)
    cai_home_path.mkdir(parents=True, exist_ok=True)
    (cai_home_path / "models").mkdir(parents=True, exist_ok=True)


_bootstrap_paths()

from cai.main import main


if __name__ == "__main__":
    main()



# -*- mode: python ; coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

CAI_ROOT = Path(os.environ.get("CAI_REPO_ROOT", Path.cwd())).resolve()
RUNTIME_ROOT = Path(os.environ.get("CAI_RUNTIME_ROOT", CAI_ROOT / "cai")).resolve()
RUNTIME_SOURCE_ROOT = RUNTIME_ROOT / "src"
CAI_SOURCE_ROOT = CAI_ROOT / "src"
ENTRYPOINT = RUNTIME_SOURCE_ROOT / "cai" / "__main__.py"
DASHBOARD_DIR = RUNTIME_ROOT / "dashboard" / "build"
RESOURCES_DIR = RUNTIME_ROOT / "resources"

if not ENTRYPOINT.is_file():
    raise SystemExit(f"Unable to locate CAI runtime entrypoint: {ENTRYPOINT}")

if not DASHBOARD_DIR.is_dir():
    raise SystemExit(f"Dashboard assets are missing: {DASHBOARD_DIR}")

if not RESOURCES_DIR.is_dir():
    raise SystemExit(f"Resource assets are missing: {RESOURCES_DIR}")


def _safe_collect(package_name: str) -> list[str]:
    try:
        return collect_submodules(package_name)
    except Exception:
        return []


def _keep_windows_runtime_module(name: str) -> bool:
    parts = name.split(".")
    if "tests" in parts or "test" in parts:
        return False

    excluded_prefixes = (
        "cai.app",
        "cai.bench",
        "cai.worker.tests",
        "cai.worker.engines.mlx",
        "cai.worker.engines.image",
        "cai.worker.runner.image_models",
    )
    return not name.startswith(excluded_prefixes)


HIDDEN_IMPORTS = sorted(
    set(
        module
        for module in (
            _safe_collect("cai")
            + _safe_collect("cai")
            + _safe_collect("cai_compute_chain")
            + [
                "cai.worker.runner.llama_cpp.runner",
                "cai.worker.runner.bootstrap",
                "cai_compute_chain.jobs",
                "cai_compute_chain.model",
                "cai_compute_chain.model_distribution",
                "cai_compute_chain.node_config",
                "cai_compute_chain.settlement",
                "cai_compute_chain.ui_state",
                "cai_compute_chain.validators",
                "cai_compute_chain.wallet",
            ]
        )
        if _keep_windows_runtime_module(module)
    )
)

DATAS = [
    (str(DASHBOARD_DIR), "dashboard"),
    (str(RESOURCES_DIR), "resources"),
]

block_cipher = None

a = Analysis(
    [str(ENTRYPOINT)],
    pathex=[str(RUNTIME_SOURCE_ROOT), str(CAI_SOURCE_ROOT)],
    binaries=[],
    datas=DATAS,
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "mlx",
        "mlx_lm",
        "mlx_vlm",
        "mflux",
        "pytest",
        "_pytest",
        "torch",
        "tensorflow",
    ],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="cai-runtime",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="cai-runtime",
)



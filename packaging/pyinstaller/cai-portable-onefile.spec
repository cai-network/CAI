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
ENTRYPOINT = CAI_SOURCE_ROOT / "cai_compute_chain" / "cai_desktop_app.py"
DASHBOARD_DIR = RUNTIME_ROOT / "dashboard" / "build"
RESOURCES_DIR = RUNTIME_ROOT / "resources"
RUNTIME_DOCTOR_SCRIPT = CAI_ROOT / "tools" / "install-runtime-deps-win.ps1"
ICON_PATH_VALUE = os.environ.get("CAI_ICON_PATH")
ICON_PATH = Path(ICON_PATH_VALUE).resolve() if ICON_PATH_VALUE else None


def _find_llama_cpp_root() -> Path | None:
    candidates = [
        RUNTIME_ROOT / ".runtime" / "llama.cpp" / "windows" / "build",
        RUNTIME_ROOT / ".runtime" / "llama.cpp" / "windows" / "build" / "bin",
    ]
    for candidate in candidates:
        if (candidate / "llama-server.exe").is_file():
            return candidate
    return None


def _find_llama_cpp_patched_root() -> Path | None:
    candidates = [
        RUNTIME_ROOT
        / ".runtime"
        / "llama.cpp"
        / "windows-patched"
        / "build"
        / "bin"
        / "Release",
        RUNTIME_ROOT / ".runtime" / "llama.cpp" / "windows-patched" / "build" / "bin",
        RUNTIME_ROOT / ".runtime" / "llama.cpp" / "windows-patched" / "build",
    ]
    for candidate in candidates:
        if (candidate / "llama-cai-shard-engine.exe").is_file():
            return candidate
    return None


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


if not ENTRYPOINT.is_file():
    raise SystemExit(f"Unable to locate CAI desktop entrypoint: {ENTRYPOINT}")

if not DASHBOARD_DIR.is_dir():
    raise SystemExit(f"Dashboard assets are missing: {DASHBOARD_DIR}")

if not RESOURCES_DIR.is_dir():
    raise SystemExit(f"Resource assets are missing: {RESOURCES_DIR}")

if not RUNTIME_DOCTOR_SCRIPT.is_file():
    raise SystemExit(f"Runtime doctor script is missing: {RUNTIME_DOCTOR_SCRIPT}")

LLAMA_CPP_ROOT = _find_llama_cpp_root()
LLAMA_CPP_PATCHED_ROOT = _find_llama_cpp_patched_root()
COMPAT_NATIVE_PACKAGE = "e" + "xo_pyo3_bindings"

HIDDEN_IMPORTS = sorted(
    set(
        module
        for module in (
            _safe_collect("pystray")
            + _safe_collect("cai")
            + _safe_collect("cai")
            + _safe_collect(COMPAT_NATIVE_PACKAGE)
            + _safe_collect("cai_compute_chain")
            + [
                "PIL.Image",
                "PIL.ImageDraw",
                "cai.worker.runner.llama_cpp.runner",
                "cai.worker.runner.bootstrap",
                "cai_compute_chain.jobs",
                "cai_compute_chain.cai_llama_cpp_assignment_artifact_engine",
                "cai_compute_chain.cai_llama_cpp_patched_binary_executor",
                "cai_compute_chain.cai_llama_cpp_patched_executor_host",
                "cai_compute_chain.cai_llama_cpp_shard_native_bridge",
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
    (str(RUNTIME_DOCTOR_SCRIPT), "."),
]
if ICON_PATH is not None and ICON_PATH.is_file():
    DATAS.append((str(ICON_PATH), "assets"))

BINARIES = []
if LLAMA_CPP_ROOT is not None:
    required_names = {"llama-server.exe", "rpc-server.exe"}
    for file in LLAMA_CPP_ROOT.iterdir():
        if not file.is_file():
            continue
        if file.suffix.lower() == ".dll" or file.name in required_names:
            BINARIES.append((str(file), "llama.cpp"))
if LLAMA_CPP_PATCHED_ROOT is not None:
    patched_engine = LLAMA_CPP_PATCHED_ROOT / "llama-cai-shard-engine.exe"
    BINARIES.append((str(patched_engine), "llama.cpp"))

block_cipher = None

a = Analysis(
    [str(ENTRYPOINT)],
    pathex=[str(RUNTIME_SOURCE_ROOT), str(CAI_SOURCE_ROOT)],
    binaries=BINARIES,
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
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="CAI",
    icon=str(ICON_PATH) if ICON_PATH is not None and ICON_PATH.exists() else None,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    runtime_tmpdir="data",
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)



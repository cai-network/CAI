#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path


MIN_PYTHON = (3, 13)


def run(command: list[str], *, cwd: Path | None = None) -> None:
    location = f" (cwd={cwd})" if cwd else ""
    print(f"[bootstrap] {' '.join(command)}{location}")
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def display_command(command: list[str]) -> str:
    return " ".join(str(part) for part in command)


def find_venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        candidate = venv_dir / "Scripts" / "python.exe"
    else:
        candidate = venv_dir / "bin" / "python"
    if not candidate.exists():
        raise FileNotFoundError(f"Virtualenv python not found: {candidate}")
    return candidate


def ensure_venv(venv_dir: Path) -> Path:
    if not venv_dir.exists():
        print(f"[bootstrap] creating virtualenv at {venv_dir}")
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(str(venv_dir))
    else:
        print(f"[bootstrap] reusing virtualenv at {venv_dir}")
    return find_venv_python(venv_dir)


def main() -> int:
    if sys.version_info < MIN_PYTHON:
        required = ".".join(str(part) for part in MIN_PYTHON)
        current = ".".join(str(part) for part in sys.version_info[:3])
        print(
            f"[bootstrap] Python {required}+ is required, current interpreter is {current}.",
            file=sys.stderr,
        )
        print(
            "[bootstrap] Install Python 3.13 or newer and rerun this bootstrap command.",
            file=sys.stderr,
        )
        return 2

    parser = argparse.ArgumentParser(
        description="Bootstrap the CAI monorepo from a single root.",
    )
    parser.add_argument(
        "--venv",
        default=".venv",
        help="Virtualenv directory to create or reuse (default: .venv).",
    )
    parser.add_argument(
        "--skip-dashboard",
        action="store_true",
        help="Skip npm install/build in cai/dashboard.",
    )
    parser.add_argument(
        "--skip-dashboard-build",
        action="store_true",
        help="Install dashboard dependencies but skip the production dashboard build.",
    )
    parser.add_argument(
        "--skip-pip-install",
        action="store_true",
        help="Skip editable pip install for the CAI Python package.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    venv_dir = repo_root / args.venv
    dashboard_dir = repo_root / "cai" / "dashboard"
    runtime_dir = repo_root / "cai"
    native_bindings_dir = runtime_dir / "rust" / "cai_pyo3_bindings"

    venv_python = ensure_venv(venv_dir)

    if not args.skip_pip_install:
        run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
        if native_bindings_dir.exists():
            run([str(venv_python), "-m", "pip", "install", "-e", str(native_bindings_dir)])
        if runtime_dir.exists():
            run([str(venv_python), "-m", "pip", "install", "-e", str(runtime_dir)])
        run([str(venv_python), "-m", "pip", "install", "-e", str(repo_root)])
    else:
        print("[bootstrap] skipping editable pip install")

    if args.skip_dashboard:
        print("[bootstrap] skipping dashboard dependencies")
    elif not dashboard_dir.exists():
        print(f"[bootstrap] dashboard directory not found: {dashboard_dir}")
    else:
        npm = shutil.which("npm")
        if npm is None:
            print("[bootstrap] npm not found, skipping cai/dashboard install")
        else:
            run([npm, "ci"], cwd=dashboard_dir)
            if args.skip_dashboard_build:
                print("[bootstrap] skipping dashboard build")
            else:
                run([npm, "run", "build"], cwd=dashboard_dir)

    print()
    print("[bootstrap] next steps")
    print(f"  1. {venv_python} -m cai_compute_chain.cli status")
    print(f"  2. {display_command([str(venv_python), str(repo_root / 'tools' / 'run-cai-main.py')])}")
    print(f"  3. {venv_python} -m cai_compute_chain.cli launch-check")
    print("  4. Linux/VPS validator: bash ./tools/join-mainnet-validator.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

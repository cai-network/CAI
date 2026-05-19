#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="$REPO_ROOT/cai/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

cd "$REPO_ROOT"

if [[ "$#" -gt 0 ]]; then
  exec "$PYTHON_BIN" -m cai_compute_chain.cli update "$@"
fi

exec "$PYTHON_BIN" -m cai_compute_chain.cli update apply

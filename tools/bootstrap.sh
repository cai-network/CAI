#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Python not found. Install Python 3 and rerun tools/bootstrap.sh" >&2
  exit 1
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/bootstrap.py" "$@"

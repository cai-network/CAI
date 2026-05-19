#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN=""
for candidate in python3.14 python3.13 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"
    break
  fi
done

if [[ -z "$PYTHON_BIN" ]]; then
  echo "Python not found. Install Python 3.13 or newer and rerun tools/bootstrap.sh" >&2
  exit 1
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/bootstrap.py" "$@"

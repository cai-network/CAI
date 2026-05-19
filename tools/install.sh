#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/bootstrap.sh" "$@"

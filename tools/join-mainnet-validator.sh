#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tools/join-mainnet-validator.sh [options]

Join the current CAI mainnet as a validator.

Options:
  --state-url URL              Local CAI state endpoint. Default: http://127.0.0.1:52415/state
  --confirm-static-ip          Mark this node as having stable public reachability.
  --skip-validator-set-sync    Do not sync the validator set before enabling validator mode.
  -h, --help                   Show this help.

This script does not create a genesis block, does not provision owner treasury
keys, and does not deploy official CAI infrastructure. Create, fund and unlock a
local wallet before enabling validator mode.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

STATE_URL="${CAI_STATE_URL:-http://127.0.0.1:52415/state}"
CONFIRM_STATIC_IP=0
SKIP_VALIDATOR_SET_SYNC=0

while (($# > 0)); do
  case "$1" in
    --state-url)
      shift
      if (($# == 0)); then
        echo "Missing value for --state-url" >&2
        exit 2
      fi
      STATE_URL="$1"
      ;;
    --state-url=*)
      STATE_URL="${1#--state-url=}"
      ;;
    --confirm-static-ip)
      CONFIRM_STATIC_IP=1
      ;;
    --skip-validator-set-sync)
      SKIP_VALIDATOR_SET_SYNC=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

resolve_python() {
  local candidates=(
    "$REPO_ROOT/.venv/bin/python"
    "$REPO_ROOT/.venv-desktop/bin/python"
    "$REPO_ROOT/cai/.venv-linux/bin/python"
    "$REPO_ROOT/cai/.venv-wsl/bin/python"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi

  echo "Python was not found. Run tools/bootstrap.sh first or install Python 3." >&2
  return 1
}

invoke_cai_cli() {
  PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m cai_compute_chain.cli "$@"
}

STATE_URL="${STATE_URL%/}"
if [[ "$STATE_URL" != */state ]]; then
  STATE_URL="$STATE_URL/state"
fi
CAI_URL="${STATE_URL%/state}"
PYTHON_BIN="$(resolve_python)"

echo "CAI mainnet validator onboarding"
echo "  state url: $STATE_URL"
echo "  note: this script joins the current network; it does not create a genesis block."
echo "  requirement: create, fund and unlock a local wallet before enabling validator mode."
echo

invoke_cai_cli status

if ((CONFIRM_STATIC_IP == 1)); then
  invoke_cai_cli validator-config --confirm-static-ip
else
  echo
  echo "Static IP confirmation was not changed."
  echo "Rerun with --confirm-static-ip only if this node has stable public reachability."
fi

if ((SKIP_VALIDATOR_SET_SYNC == 0)); then
  invoke_cai_cli validator-set-sync --cai-url "$CAI_URL"
fi

invoke_cai_cli validator-mode --enable --state-url "$STATE_URL"
invoke_cai_cli node-config

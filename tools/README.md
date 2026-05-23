<!--
SPDX-FileCopyrightText: 2026 CAI contributors
SPDX-License-Identifier: MIT
-->
# CAI Tools

This directory contains public user and developer tools only.

## Public Tools

- `bootstrap.ps1`, `bootstrap.sh`, `bootstrap.py` prepare a local development checkout.
- `run-cai-main.py` starts the local CAI runtime.
- `join-mainnet-validator.sh` helps an already configured Linux/macOS/VPS node join the current CAI mainnet as a validator. A configured node means the repository was bootstrapped, CAI runtime is running and reachable through the local API, a local wallet was created/restored, funded and unlocked, and public reachability/firewall settings are prepared when the node is expected to serve as a public validator. The script syncs the validator set and enables validator mode; it does not create a genesis block and does not provision owner treasury keys.
- `build-portable-win.ps1` builds the Windows portable package. On a fresh checkout it prepares `cai/.venv-win` and the Windows `llama.cpp` runtime automatically unless `-NoBootstrap` or `-NoInstallLlamaCpp` is passed.
- `install-llama-cpp*.ps1`, `build-llama-cpp-patched*.ps1` prepare llama.cpp runtime binaries.
- `generate-release-metadata.py`, `generate-release-notes.py`, `check-portable-clean.ps1` validate release artifacts without publishing official updates.

## Maintainer-Only Tools

Official VPS deployment, release publishing, update-server staging, live stand orchestration and private SSH helpers are intentionally not shipped in the public repository.

Public nodes should consume signed update metadata from the current CAI network. Official update publishing is controlled by project maintainers outside this repository.

<!--
SPDX-FileCopyrightText: 2026 CAI contributors
SPDX-License-Identifier: MIT
-->
# Code cleanup and refactor plan

This plan keeps refactoring work small, reviewable, and separate from behavior
changes. The goal is not cosmetic cleanup. The goal is to make network,
wallet, update, settlement, and distributed-compute work safer to change.

## Rules

- Keep each refactor step small enough to test and commit separately.
- Do not mix cleanup with tokenomics, consensus, wallet policy, or update
  protocol changes unless that is the explicit task.
- Preserve public imports and CLI/API surfaces unless a migration is planned.
- Add or run a focused verification step for each moved boundary.
- Stop when a stable area would only gain visual neatness from more code churn.

## Active focus

- Continue splitting oversized modules along real domain boundaries.
- Keep `jobs.py`, `decentralized_compute.py`, `api/main.py`, `cli.py`,
  `update_channel.py`, and `model_distribution.py` moving toward smaller
  modules with stable compatibility wrappers.
- Replace silent best-effort failures with explicit audit/status objects or
  clear logging where user flows must continue.
- Move JSON state access toward a dedicated storage/repository layer.

## Completed cleanup checkpoints

- `jobs.py` has been partially split into job storage, pricing, HTTP helpers,
  execution config, execution attempts, request payload helpers, reward
  distribution, node URL helpers, worker eligibility, task transport, instance
  placement, and readiness modules.
- Runtime cleanup for orphan `llama.cpp` processes is isolated in
  `runtime_cleanup.py`.
- Chain and peer-sync paths expose failed peer URLs and peer-level errors.
- Update summaries expose source-resolution errors instead of silently
  returning an empty update state.
- CAI-owned transport session storage, payload storage, and replay-cache logic
  is isolated in `cai_owned_transport_storage.py`.
- CAI-owned transport protocol constants and shared wire-level names are
  isolated in `cai_owned_transport_protocol.py`.
- CAI-owned transport shared helper functions for node IDs, chain IDs, and
  timestamps are isolated in `cai_owned_transport_common.py`.
- CAI-owned transport peer URL cleanup, prioritization, and route-class
  detection is isolated in `cai_owned_transport_peer_urls.py`.
- CAI-owned transport data-plane route readiness and route-health quorum checks
  are isolated in `cai_owned_transport_route_readiness.py`.

## Next checkpoints

- Continue splitting `decentralized_compute.py` into protocol DTOs, transport
  storage, dispatch, proof validation, route readiness, and execution planning.
- Split `api/main.py` into routers for wallet, chain, validators, updates,
  transport, and OpenAI-compatible APIs.
- Split `cli.py` into command modules without changing CLI commands.
- Move generated portable-update apply scripts out of large inline strings and
  into testable templates.
- Add concurrency smoke checks for Windows JSON writes around update status,
  settlements, chain state, and wallet session files.

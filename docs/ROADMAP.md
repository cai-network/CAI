<!--
SPDX-FileCopyrightText: 2026 CAI contributors
SPDX-License-Identifier: MIT
-->
# Roadmap

This roadmap describes the current project state and the active hardening tracks. It is not a promise of dates. It is a working map for bringing the alpha network to a stable public state.

## Implemented

- Wallet creation, recovery, locking, unlocking, balances, and transfers.
- `mainnet` and `testnet` modes.
- Local runtime API and web UI request path.
- Executor and validator roles.
- Relay/bootstrap support for peer discovery and connectivity.
- Settlement records for completed AI jobs.
- Reward split between executors and validator settlement pool.
- Genesis economy with `1,000,000,000 CAICN`.
- Founder treasury and developer contribution fund accounting.
- Blockchain indexes, snapshots, address history, and settlement history.
- Hybrid post-quantum wallet/signature direction using Ed25519 + ML-DSA-65.
- Portable build and update tooling.
- Developer fund registry, round files, and validation command.

## Alpha hardening

- Verify distributed inference between two or more physical PCs.
- Complete PC-to-PC data flow so validators do not become model-data bottlenecks.
- Improve executor disconnect handling with retry, redistribution, and clear request outcomes.
- Keep update UX visible, safe, and interruptible before the unsafe apply phase.
- Validate growing-chain behavior with snapshots, indexes, pruning, archive strategy, and restore tests.
- Strengthen multi-validator consensus and finality behavior.
- Verify mandatory post-quantum validation on governance, payout, validator, and transfer paths.
- Expand live-network tests, load tests, and security review.

## Network direction

CAI focuses on useful decentralized AI compute. The network path prioritizes:

- real executor readiness instead of declared-only availability;
- route choice based on feasibility and speed;
- direct PC-to-PC transport when possible;
- relay use only where it improves connectivity;
- transparent settlement and reward accounting.

## Economy direction

The base alpha model uses:

- `85%` compute reserve;
- `5%` founder treasury;
- `10%` developer contribution fund;
- `98%` of AI job price for executors;
- `2%` of AI job price for the validator settlement pool.

The market path for `CAICN/USDT` is part of the broader network direction so users and executors can buy, sell, and evaluate the token more conveniently.

## Public release readiness

The project is ready for broader public attention when:

- two or more physical machines can complete distributed inference reliably;
- validator settlement works without becoming a data-plane bottleneck;
- updates work predictably during normal app usage;
- chain recovery from snapshots is tested;
- sensitive paths have security review;
- repository rules protect `main` while keeping pull requests open to contributors.

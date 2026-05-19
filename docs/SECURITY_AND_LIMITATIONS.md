<!--
SPDX-FileCopyrightText: 2026 CAI contributors
SPDX-License-Identifier: MIT
-->
# Security and limitations

CAI is in mainnet alpha. The project has a working core, but operators and contributors should treat the public network as software under active stabilization.

## Sensitive data

Never commit or publish:

- seed phrases;
- private keys;
- wallet backup files;
- VPS passwords;
- API tokens;
- `.env` files with real secrets;
- local node state from `.cai-local`;
- release signing keys or update credentials.

If a secret is accidentally published, rotate it immediately. Deleting it from the latest commit is not enough if it reached git history.

## Wallet safety

Wallet seed phrases control funds. Store them offline or in a secure password manager. Do not paste seed phrases into issues, pull requests, logs, screenshots, CI output, or chat messages.

Validator wallets require extra care because they can sign operational actions. A validator should be recoverable after VPS failure, but the recovery material must stay outside public infrastructure.

## Network limitations

The alpha network still hardens:

- distributed inference across physical machines;
- PC-to-PC transport stability;
- executor disconnect handling;
- multi-validator consensus/finality;
- growing-chain maintenance;
- update UX and restart safety.

Users should expect clear errors when a request cannot be completed. Silent failure is treated as a bug.

## Blockchain limitations

Indexes and snapshots are implemented and used by the chain layer. Long-running networks still need ongoing checks for:

- snapshot restore;
- history growth;
- archival strategy;
- pruning strategy;
- settlement index consistency;
- large-history wallet balance reads.

## Post-quantum direction

CAI follows a hybrid post-quantum direction for wallets and signatures: Ed25519 + ML-DSA-65.

The project checks critical paths so post-quantum validation is not only present in wallet creation but also required where it protects real state changes.

## Contribution safety

Pull requests that affect sensitive areas need careful review:

- genesis and token allocation;
- wallet and seed handling;
- transfer validation;
- settlement and reward accounting;
- validator rules;
- update and release scripts;
- network transport and peer discovery;
- GitHub Actions and release automation.

Contributors should keep changes small enough to review. Large refactors are welcome when they improve the architecture, but they should not hide behavior changes in unrelated cleanup.

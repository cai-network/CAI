<!--
SPDX-FileCopyrightText: 2026 CAI contributors
SPDX-License-Identifier: MIT
-->
# Project overview

CAI is an open decentralized network for LLM compute. The project combines a wallet, local runtime, executor network, validator settlement, and a web interface into one practical system for running model requests and accounting for useful work.

The network token is `CAICN` (`CAI Network Credit`). It is used to account for model work, executor rewards, validator settlement fees, and developer contribution rewards.

## Purpose

Modern AI compute is often concentrated in a small number of centralized services. CAI takes a different direction: participants can connect real machines, provide CPU/GPU resources, execute model work, and receive rewards for completed tasks.

The same interface can be used by a regular user, a local application, or a custom AI agent. CAI provides the local API and network path that lets those clients use local or network-backed model execution.

## Current status

CAI is in **mainnet alpha**.

The core cycle already exists:

- wallet creation, recovery, locking, unlocking, balances, and transfers;
- `mainnet` and `testnet` network modes;
- local API and web UI for model requests;
- executor and validator roles;
- settlement records for completed AI work;
- reward accounting for executors and validator settlement;
- blockchain indexes, snapshots, address history, and settlement history;
- portable update and launch tooling.

The public network remains experimental. Operators should treat alpha builds as live software under active stabilization rather than as a finalized production network.

## Participants

- Users send model requests through the web UI, local API, or external AI agents.
- Executors provide compute resources and perform model work.
- Validators confirm settlement and record network accounting.
- Relay/bootstrap nodes help peers find a path to each other without becoming a permanent data center for model traffic.
- Developers improve the protocol, runtime, UI, tooling, tests, and documentation.

## What makes CAI different

- Rewards are connected to useful AI compute rather than empty activity.
- The project combines AI runtime, wallet, settlement, and web UI in one portable package.
- Executors can connect real machines and participate in request processing.
- The architecture supports local inference and distributed loading of model parts.
- External AI agents can use CAI as a compute backend.
- Developer contribution rewards are documented and can be governed transparently.
- Wallet and signature work follows a hybrid post-quantum direction: Ed25519 + ML-DSA-65.

## Practical use cases

- A user runs a local CAI interface and sends model requests.
- A team connects an AI agent to the CAI local API.
- An executor contributes CPU/GPU resources to the network.
- A validator confirms settlement and earns a settlement share.
- A developer contributes code, reviews, tests, or documentation and participates in the developer contribution process.

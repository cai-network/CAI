<!--
SPDX-FileCopyrightText: 2026 CAI contributors
SPDX-License-Identifier: MIT
-->
# Architecture

CAI is organized as a set of cooperating layers. Each layer has a narrow responsibility so the project can evolve without turning the validator, UI, or runtime into a single point of failure.

## Core layers

- Web UI and desktop/portable shell provide the user-facing application.
- Local API accepts model requests from the web UI and external AI agents.
- Runtime layer prepares model execution and connects to GGUF/llama.cpp backends.
- Network overlay discovers peers, checks executor readiness, and selects execution routes.
- PC-to-PC data plane carries working data between nodes directly when possible.
- Settlement layer records completed AI jobs and calculates rewards.
- Blockchain layer stores balances, transfers, settlement records, indexes, and snapshots.
- Update layer manages portable releases and node updates.

## Request flow

1. A client sends a model request through the web UI, local API, or an external AI agent.
2. The runtime checks whether the request can be executed locally or through the network.
3. The network layer discovers available executors and checks their readiness.
4. Route selection chooses a local, single-executor, or distributed path.
5. Executors load the required model or required model part and perform the assigned work.
6. Results return to the requester through the selected route.
7. CAI creates receipts for completed work.
8. Validators confirm settlement and write the accounting result into the chain state.

## Validator role

Validators confirm state and settlement. They are not designed to be permanent proxies for model data.

This distinction matters:

- validator traffic is for state, settlement, and consensus-related decisions;
- working model traffic should move directly between nodes when possible;
- relay/reverse-relay paths are used for connectivity when direct routes are not available;
- bootstrap peers help discovery and do not define ownership of the network.

For uptime, a validator can use active/passive HA replicas under one bonded validator identity. This protects availability without multiplying voting power: the validator remains one committee member, one bond, and one settlement vote. The active replica renews a short-lived lease; standby replicas stay silent while the lease is valid and promote themselves only after the lease expires.

## Distributed compute

CAI supports a runtime architecture where model work can be assigned to one executor or split across participants when the selected path benefits from distribution.

Model package manifests describe GGUF artifacts as verified, layer-aware chunks. A seed node can import or create a manifest, cache the full package once, and publish its chunk inventory. Executors then plan only the chunks required for their assigned layer range and fetch missing chunks from peers, storage seeds, or the original Hugging Face artifact when that origin is present in the manifest.

The operator flow is:

1. Create or import a model package manifest.
2. Cache the full package on at least one seed node.
3. Publish local chunk inventory from that seed node.
4. Let executors sync inventory and fetch only the chunks required by their assignment.

Example commands:

```bash
python -m cai_compute_chain.cli model-package-create-hf-gguf Qwen/Qwen2.5-0.5B-Instruct-GGUF 0.1.0 --preferred-filename qwen2.5-0.5b-instruct-q4_k_m.gguf
python -m cai_compute_chain.cli model-package-cache-all Qwen--Qwen2.5-0.5B-Instruct-GGUF 0.1.0 --node-id seed-node-1
python -m cai_compute_chain.cli chunk-inventory-local seed-node-1 --source-kind local_cache
```

The current alpha hardening track focuses on:

- stable PC-to-PC routing between physical machines;
- executor readiness checks that match real transport availability;
- model-part loading instead of unnecessary full loading where possible;
- retry and redistribution when an executor disconnects;
- clear user-facing errors when a request cannot be completed honestly.

## Blockchain and settlement

The blockchain layer keeps the accounting state:

- account balances;
- transfers;
- settlement entries;
- reward assignments;
- validator settlement pool accounting;
- address and settlement indexes;
- snapshots for faster state recovery.

Indexes and snapshots are part of normal operation. Long-term history still requires pruning, archiving, recovery checks, and load testing as the network grows.

## Update architecture

Portable builds use an update channel that can be served by release infrastructure. A node checks update metadata, downloads the new artifact, pauses unsafe user actions during apply, and restarts into the updated version.

The update path is part of the same operational model as the network: it must be visible to users, cancellable before the unsafe phase, and safe for normal node operators.

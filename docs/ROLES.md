<!--
SPDX-FileCopyrightText: 2026 CAI contributors
SPDX-License-Identifier: MIT
-->
# Participant roles

CAI has several roles. A single machine can run more than one role when configured that way, but each role has a different responsibility.

## User

A user sends model requests through the web UI, local API, or an external AI agent. The user normally creates or unlocks a wallet before sending requests that use network resources.

Users receive:

- a local interface for model requests;
- wallet and balance management;
- access to local or network-backed compute routes;
- clear status and errors when a request cannot be completed.

## Executor

An executor provides CPU/GPU resources to the network. It advertises readiness, receives assigned model work, executes the work, and returns the result.

Executors earn rewards when settlement confirms completed work. The reward is tied to useful computation, not to simply being online.

## Validator

A validator confirms settlement and records accounting state. Validators receive a settlement share for confirmed AI jobs.

Validators are responsible for:

- checking settlement records;
- writing accepted settlement results;
- maintaining chain state;
- participating in validator coordination when more validators are active.

Validator nodes should keep their operational keys safe. A validator wallet must be available to sign required actions, but seed phrases and private keys must not be stored in public repositories, shared scripts, logs, or issue comments.

Validator high availability is modeled as replicas of one validator identity. Multiple nodes can protect uptime for the same bonded validator, but they do not create multiple votes from one bond. Only the active replica signs settlement attestations; passive replicas stay ready without signing. Active replicas publish a short-lived lease, and passive replicas can take over automatically after the lease expires.

## Relay/bootstrap node

Relay/bootstrap nodes help peers find each other and maintain connectivity. They are not a replacement for direct PC-to-PC data flow.

Good relay/bootstrap behavior:

- helps new nodes discover peers;
- assists with reverse relay when direct connectivity is blocked;
- avoids becoming the permanent data path for all model traffic;
- does not define or replace consensus rules.

## Developer

Developers contribute code, tests, documentation, security review, runtime improvements, network logic, UI work, and release tooling.

Developer contribution rewards are governed by the developer contribution fund rules. Code that reaches the main branch is still subject to review, security expectations, and project governance.

## Founder and final maintainers

The founder and assigned final maintainers protect the public repository, final merge flow, release path, and sensitive network decisions. This role is intentionally narrow: it exists to keep the project coherent and safe while contributors participate through pull requests, reviews, and governance votes.

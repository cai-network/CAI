<!--
SPDX-FileCopyrightText: 2026 CAI contributors
SPDX-License-Identifier: MIT
-->
# GitHub setup

This document describes a safe repository setup for publishing CAI while keeping `main` protected.

## Recommended model

Do not give write access to everyone.

For a public GitHub repository, outside contributors can fork the repository and open pull requests without write access. This is the normal open-source model and keeps the project open without allowing arbitrary users to push branches into the upstream repository.

Use direct repository write access only for trusted recurring contributors.

## Access levels

Recommended access model:

- Public users: no direct write access; they fork and open pull requests.
- Regular contributors: no write access by default; promote only after trust is established.
- Reviewers: triage or write access when needed for issue/PR work.
- Maintainers: maintain access for trusted people who help manage PRs and releases.
- Founder/final maintainers: admin or final merge authority.

In an organization, use teams instead of individual permissions:

- `contributors` - read or triage;
- `reviewers` - triage or write;
- `maintainers` - maintain;
- `release-admins` - admin and protected release environment access.

## Protecting main

In GitHub, open repository settings and configure a ruleset or branch protection rule for `main`.

Recommended rules:

- require a pull request before merging;
- require review from selected reviewers or code owners;
- require status checks before merge;
- require all conversations to be resolved;
- block force pushes;
- block deletion of `main`;
- restrict direct pushes to founder/final maintainers;
- require signed commits if the team is ready for that workflow;
- require linear history if the project wants a clean commit graph.

## Pull requests

Anyone can open a pull request from a fork when the repository is public. Collaborators with write access can also create branches directly in the upstream repository, but that should be limited to trusted contributors.

Sensitive changes should receive extra review:

- `.github/workflows/*`;
- wallet and signature code;
- genesis and chain state code;
- settlement and rewards;
- update and release tooling;
- validator and network transport code;
- developer fund payout logic.

## GitHub Actions

Recommended settings:

- keep default workflow permissions read-only;
- require approval for first-time contributors before running workflows;
- do not expose secrets to pull requests from forks;
- store release and VPS credentials only in protected GitHub Environments;
- require reviewer approval for protected environments.

Python tests can remain gated while the public repository is being prepared, but the documentation and release checks should stay enabled so every PR receives basic validation.

## CODEOWNERS

A `CODEOWNERS` file can automatically request review from trusted maintainers for sensitive areas.

Recommended ownership areas:

- wallet and signature paths;
- chain, genesis, settlement, and rewards;
- update and release scripts;
- network transport;
- GitHub Actions;
- developer fund files.

## Public workflow

1. Contributor forks the repository.
2. Contributor creates a branch in their fork.
3. Contributor opens a pull request to `main`.
4. Checks run.
5. Reviewers discuss and request changes.
6. Governance voting is attached when the rules require it.
7. Founder or final maintainer merges when the PR is accepted.

This keeps CAI open to contributors without losing control of the main network code.

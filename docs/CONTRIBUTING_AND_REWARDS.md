<!--
SPDX-FileCopyrightText: 2026 CAI contributors
SPDX-License-Identifier: MIT
-->
# Developer contributions and rewards

CAI includes a developer contribution fund. Its purpose is to reward people whose work becomes part of the project and helps the network mature.

This document describes the public rules for contribution tracking, voting, and payout rounds.

## Fund size

The genesis economy contains a `10%` developer contribution fund.

This fund is separate from:

- the compute reserve;
- executor rewards;
- validator settlement rewards;
- founder treasury.

## Contribution contest

The initial developer contribution contest runs for one year.

The reward structure is:

- `7%` of the total supply is assigned to seven most significant project contributors, `1%` each.
- `1%` of the total supply is shared between ten active developers.
- `1%` of the total supply is shared between one hundred developers selected from broader project contributors.
- `1%` of the total supply is shared between up to one thousand authors whose code becomes part of the project.

If a tier has fewer eligible participants than available slots, the remaining amount for that tier is distributed among the eligible participants of that tier.

## Eligibility

A contributor is eligible when their contribution is accepted into the project or otherwise recognized by the documented governance process.

Examples of eligible work:

- protocol and network code;
- runtime and model execution improvements;
- wallet, settlement, blockchain, or validator work;
- UI and portable application improvements;
- tests, security review, documentation, and tooling;
- meaningful review work that prevents bugs or security issues.

Contributors can add their public GitHub account and CAI reward address to the developer fund registry. The registry is a practical way to connect a public contributor identity to a payout address.

## Voting rules

Voting happens by signed votes. A signed vote is the vote itself, not a separate later approval artifact.

Rules:

- a participant cannot vote for themselves;
- a participant who does not vote cannot claim a reward in that round;
- votes must be attributable to eligible voters for that round;
- votes must be verifiable before a payout round is finalized;
- payout execution requires the required governance/finalizer signature so a local CLI command cannot spend the fund by itself.

## Round structure

The contest has several logical rounds:

- The top-7 round selects the seven most significant contributors through broad participant voting and founder confirmation signature.
- The top-10 round is voted by the seven selected contributors and confirmed by the founder signature.
- The top-100 round is voted by the seventeen main authors from the previous two tiers.
- The up-to-1000 author round covers authors whose accepted code becomes part of the project; if the number of authors exceeds the tier capacity, selection is resolved by voting from the top-100 group.

Founder confirmation is used where the rules require it. This is not the same thing as an automatic approval flag.

## Pull request review

Contribution rewards do not bypass code review.

The repository workflow uses pull requests:

- contributors propose changes through PRs;
- selected reviewers review sensitive changes;
- conversations and checks are resolved before merge;
- final merge is performed by the founder or an assigned final maintainer;
- if a PR passed the required governance vote and the founder does not approve or reject it within 7 calendar days after the vote completion date, the PR is considered accepted under the governance rules.

The 7-day rule applies to PR governance flow. It does not replace required founder confirmation signatures for developer fund payout rounds.

## Payout rounds

A payout round should contain:

- round identifier;
- eligible participants;
- voter list;
- signed votes;
- calculated results;
- payout addresses;
- final governance/finalizer signature;
- resulting transaction or settlement reference.

This keeps voting meaningful while preventing arbitrary local payout commands from spending the developer fund.

## Repository files

The public repository contains helper files for the developer fund:

- `.github/developer-fund/participants.json`
- `.github/developer-fund/rounds/round-001.json`

These files document the public process and can be validated by project tooling. They are not a replacement for chain-level payout authorization.

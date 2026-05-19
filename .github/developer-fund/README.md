<!--
SPDX-FileCopyrightText: 2026 CAI contributors
SPDX-License-Identifier: MIT
-->
# Developer Fund Registry

This directory contains the public MVP registry for the CAI developer contribution fund.

- `participants.json` is the self-registration list. Contributors add their GitHub login and CAI address through a pull request.
- `rounds/round-001.json` is the first yearly distribution round. It stays `draft` until voting is complete.
- Approved rounds must declare `round_type`: `top-7`, `top-10`, `top-100`, or `code-authors`.
- Accepted code in `main` is the source of contribution facts. These files only connect contributors to payout addresses and record the approved payout result.
- Contest votes are signed by participant CAI wallets and stored in the round `votes` array; `vote_result.tallies` must match those signed votes.
- A participant cannot vote for themselves. Open rounds require winners to cast a valid vote; restricted rounds count only `voting.eligible_voters`.
- Founder confirmation is recorded in `founder_confirmation` and must be signed by the founder/developer treasury wallet; it is not counted as a normal contest vote.
- CAI address signatures are not a normal requirement. Contributors provide their payout address through a GitHub PR; extra address signatures are only an exceptional dispute/risk tool.

Validate the registry and round:

```powershell
python -m cai_compute_chain.cli developer-fund validate
```

Create a founder confirmation for an approved payout round:

```powershell
python -m cai_compute_chain.cli developer-fund sign-founder-confirmation --round .github/developer-fund/rounds/round-001.json
```

After a round is approved, record it on-chain:

```powershell
python -m cai_compute_chain.cli developer-fund distribute --round .github/developer-fund/rounds/round-001.json
```

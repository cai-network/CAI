# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.chain import (
    chain_balance_atomic,
    developer_contribution_fund_chain_address,
    record_developer_fund_distribution,
)
from cai_compute_chain.developer_fund import (
    canonical_json_hash,
    developer_fund_founder_confirmation_signing_payload,
    developer_fund_vote_signing_payload,
    developer_fund_recipients_for_chain,
    validate_developer_fund_files,
)
from cai_compute_chain.model import MoneyPolicy, WalletPolicy
from cai_compute_chain.wallet import coins_to_atomic
from cai_compute_chain.wallet_signing import (
    ADDRESS_SCHEME_ED25519,
    address_from_public_key_b64,
    generate_signing_seed,
    public_key_b64_from_seed,
    sign_payload_b64,
)


class DeveloperFundTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.tempdir.name)
        self.repo_patch = patch(
            "cai_compute_chain.wallet.repo_root",
            return_value=self.repo_root,
        )
        self.repo_patch.start()
        self.fund_dir = self.repo_root / ".github" / "developer-fund"
        self.rounds_dir = self.fund_dir / "rounds"
        self.rounds_dir.mkdir(parents=True)
        self.participants_path = self.fund_dir / "participants.json"
        self.round_path = self.rounds_dir / "round-001.json"
        self.founder_seed = generate_signing_seed()
        self.founder_public_key_b64 = public_key_b64_from_seed(self.founder_seed)
        self.founder_address = address_from_public_key_b64(
            self.founder_public_key_b64
        )
        self.money_policy = MoneyPolicy(
            developer_treasury_address=self.founder_address
        )

    def tearDown(self) -> None:
        self.repo_patch.stop()
        self.tempdir.cleanup()

    def _write_json(self, path: Path, payload) -> None:
        if path == self.round_path and isinstance(payload, dict):
            payload = dict(payload)
            payload.pop("approval", None)
            if (
                str(payload.get("status") or "").strip().lower() == "approved"
                and "founder_confirmation" not in payload
            ):
                payload["founder_confirmation"] = self._founder_confirmation(payload)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _founder_confirmation(self, round_payload: dict) -> dict:
        participants_payload = json.loads(
            self.participants_path.read_text(encoding="utf-8")
        )
        confirmed_at = "2026-05-20T00:00:00+00:00"
        payload = developer_fund_founder_confirmation_signing_payload(
            round_payload=round_payload,
            participants_hash=canonical_json_hash(participants_payload),
            confirmed_by="founder",
            confirmed_at=confirmed_at,
        )
        return {
            "confirmed_by": "founder",
            "confirmed_at": confirmed_at,
            "address_scheme": ADDRESS_SCHEME_ED25519,
            "public_key_b64": self.founder_public_key_b64,
            "signature_b64": sign_payload_b64(self.founder_seed, payload),
        }

    def _write_round(self, round_payload: dict) -> None:
        payload = dict(round_payload)
        if str(payload.get("status") or "").strip().lower() == "approved":
            payload["founder_confirmation"] = self._founder_confirmation(payload)
        self._write_json(self.round_path, payload)

    def _validate(self):
        return validate_developer_fund_files(
            repo_root=self.repo_root,
            money_policy=self.money_policy,
        )

    def _signed_vote(
        self,
        *,
        round_id: str,
        github: str,
        cai_address: str,
        public_key_b64: str,
        signing_seed: bytes,
        choices: list[str],
    ) -> dict:
        payload = developer_fund_vote_signing_payload(
            round_id=round_id,
            github=github,
            cai_address=cai_address,
            choices=choices,
        )
        return {
            "github": github,
            "choices": choices,
            "address_scheme": ADDRESS_SCHEME_ED25519,
            "public_key_b64": public_key_b64,
            "signature_b64": sign_payload_b64(signing_seed, payload),
        }

    def _wallets_for(self, names: list[str]) -> dict[str, dict]:
        wallets: dict[str, dict] = {}
        for name in names:
            signing_seed = generate_signing_seed()
            public_key_b64 = public_key_b64_from_seed(signing_seed)
            wallets[name] = {
                "signing_seed": signing_seed,
                "public_key_b64": public_key_b64,
                "address": address_from_public_key_b64(public_key_b64),
            }
        return wallets

    def _participants_for(self, wallets: dict[str, dict]) -> list[dict]:
        return [
            {
                "github": github,
                "cai_address": wallet["address"],
                "status": "active",
            }
            for github, wallet in wallets.items()
        ]

    def _vote_for(
        self,
        *,
        wallets: dict[str, dict],
        round_id: str,
        github: str,
        choices: list[str],
    ) -> dict:
        wallet = wallets[github]
        return self._signed_vote(
            round_id=round_id,
            github=github,
            cai_address=wallet["address"],
            public_key_b64=wallet["public_key_b64"],
            signing_seed=wallet["signing_seed"],
            choices=choices,
        )

    def test_empty_draft_round_is_valid_but_not_distributable(self) -> None:
        self._write_json(self.participants_path, [])
        self._write_round(
            {
                "round_id": "year-1",
                "status": "draft",
                "winners": [],
            },
        )

        result = self._validate()

        self.assertTrue(result.ok)
        self.assertFalse(result.distributable)
        self.assertEqual(result.participants_count, 0)
        self.assertIn("participants registry is empty", result.warnings)

    def test_approved_round_records_on_chain_distribution_once(self) -> None:
        money_policy = self.money_policy
        wallet_policy = WalletPolicy(wallet_data_dirname="developer-fund-node")
        alice_seed = generate_signing_seed()
        alice_public_key_b64 = public_key_b64_from_seed(alice_seed)
        recipient_address = address_from_public_key_b64(alice_public_key_b64)
        bob_seed = generate_signing_seed()
        bob_public_key_b64 = public_key_b64_from_seed(bob_seed)
        bob_address = address_from_public_key_b64(bob_public_key_b64)
        alice_vote = self._signed_vote(
            round_id="year-1",
            github="alice",
            cai_address=recipient_address,
            public_key_b64=alice_public_key_b64,
            signing_seed=alice_seed,
            choices=["bob"],
        )
        bob_vote = self._signed_vote(
            round_id="year-1",
            github="bob",
            cai_address=bob_address,
            public_key_b64=bob_public_key_b64,
            signing_seed=bob_seed,
            choices=["alice"],
        )
        self._write_json(
            self.participants_path,
            [
                {
                    "github": "alice",
                    "cai_address": recipient_address,
                    "status": "active",
                },
                {
                    "github": "bob",
                    "cai_address": bob_address,
                    "status": "active",
                },
            ],
        )
        self._write_round(
            {
                "round_id": "year-1",
                "round_type": "code-authors",
                "candidate_count": 1,
                "status": "approved",
                "github_issue": "https://github.com/example/cai/issues/1",
                "vote_result": {
                    "outcome": "passed",
                    "source": "https://github.com/example/cai/issues/1",
                    "completed_at": "2026-05-19T00:00:00+00:00",
                    "signed_vote_count": 2,
                    "tallies": {"alice": 1, "bob": 1},
                },
                "winners": [
                    {
                        "github": "alice",
                        "amount": "10000000.00000000",
                        "category": "code-authors",
                        "reason": "Core networking work",
                    }
                ],
                "voting": {"method": "signed_participant_votes"},
                "votes": [alice_vote, bob_vote],
            },
        )

        result = self._validate()
        self.assertTrue(result.ok)
        self.assertTrue(result.distributable)

        block = record_developer_fund_distribution(
            round_id=result.round_id,
            recipients=developer_fund_recipients_for_chain(result),
            round_hash=result.round_hash,
            participants_hash=result.participants_hash,
            policy=wallet_policy,
            money_policy=money_policy,
        )

        amount_atomic = coins_to_atomic("10000000.00000000", money_policy)
        fund_address = developer_contribution_fund_chain_address(money_policy)
        self.assertEqual(len(block.transactions), 2)
        self.assertEqual(
            chain_balance_atomic(fund_address, wallet_policy),
            coins_to_atomic(
                str(money_policy.developer_contribution_fund_coins),
                money_policy,
            )
            - amount_atomic,
        )
        self.assertEqual(
            chain_balance_atomic(recipient_address, wallet_policy),
            amount_atomic,
        )
        with self.assertRaises(ValueError):
            record_developer_fund_distribution(
                round_id=result.round_id,
                recipients=developer_fund_recipients_for_chain(result),
                round_hash=result.round_hash,
                participants_hash=result.participants_hash,
                policy=wallet_policy,
                money_policy=money_policy,
            )

    def test_round_rejects_unregistered_winner(self) -> None:
        signing_seed = generate_signing_seed()
        public_key_b64 = public_key_b64_from_seed(signing_seed)
        address = address_from_public_key_b64(public_key_b64)
        bob_seed = generate_signing_seed()
        bob_public_key_b64 = public_key_b64_from_seed(bob_seed)
        bob_address = address_from_public_key_b64(bob_public_key_b64)
        vote = self._signed_vote(
            round_id="year-1",
            github="alice",
            cai_address=address,
            public_key_b64=public_key_b64,
            signing_seed=signing_seed,
            choices=["bob"],
        )
        self._write_json(
            self.participants_path,
            [
                {
                    "github": "alice",
                    "cai_address": address,
                    "status": "active",
                },
                {
                    "github": "bob",
                    "cai_address": bob_address,
                    "status": "active",
                },
            ],
        )
        self._write_json(
            self.round_path,
            {
                "round_id": "year-1",
                "round_type": "code-authors",
                "candidate_count": 1,
                "status": "approved",
                "vote_result": {
                    "outcome": "passed",
                    "source": "https://github.com/example/cai/issues/1",
                    "completed_at": "2026-05-19T00:00:00+00:00",
                    "signed_vote_count": 1,
                    "tallies": {"bob": 1},
                },
                "winners": [
                    {
                        "github": "missing-user",
                        "amount": "10000000.00000000",
                        "category": "code-authors",
                    }
                ],
                "voting": {"method": "signed_participant_votes"},
                "votes": [vote],
            },
        )

        result = self._validate()

        self.assertFalse(result.ok)
        self.assertIn(
            "winner 'missing-user' is not registered in participants",
            result.errors,
        )

    def test_approved_round_requires_founder_confirmation(self) -> None:
        signing_seed = generate_signing_seed()
        public_key_b64 = public_key_b64_from_seed(signing_seed)
        address = address_from_public_key_b64(public_key_b64)
        self._write_json(
            self.participants_path,
            [
                {
                    "github": "alice",
                    "cai_address": address,
                    "status": "active",
                }
            ],
        )
        self._write_json(
            self.round_path,
            {
                "round_id": "year-1",
                "round_type": "code-authors",
                "candidate_count": 1,
                "status": "approved",
                "founder_confirmation": None,
                "winners": [
                    {
                        "github": "alice",
                        "amount": "10000000.00000000",
                        "category": "code-authors",
                    }
                ],
            },
        )

        result = self._validate()

        self.assertFalse(result.ok)
        self.assertIn(
            "approved round must include founder_confirmation object",
            result.errors,
        )

    def test_approved_round_requires_vote_result(self) -> None:
        alice_seed = generate_signing_seed()
        alice_public_key_b64 = public_key_b64_from_seed(alice_seed)
        alice_address = address_from_public_key_b64(alice_public_key_b64)
        bob_seed = generate_signing_seed()
        bob_public_key_b64 = public_key_b64_from_seed(bob_seed)
        bob_address = address_from_public_key_b64(bob_public_key_b64)
        alice_vote = self._signed_vote(
            round_id="year-1",
            github="alice",
            cai_address=alice_address,
            public_key_b64=alice_public_key_b64,
            signing_seed=alice_seed,
            choices=["bob"],
        )
        bob_vote = self._signed_vote(
            round_id="year-1",
            github="bob",
            cai_address=bob_address,
            public_key_b64=bob_public_key_b64,
            signing_seed=bob_seed,
            choices=["alice"],
        )
        self._write_json(
            self.participants_path,
            [
                {
                    "github": "alice",
                    "cai_address": alice_address,
                    "status": "active",
                },
                {
                    "github": "bob",
                    "cai_address": bob_address,
                    "status": "active",
                }
            ],
        )
        self._write_json(
            self.round_path,
            {
                "round_id": "year-1",
                "round_type": "top-7",
                "status": "approved",
                "winners": [
                    {
                        "github": "alice",
                        "amount": "10000000.00000000",
                        "category": "top-7",
                    }
                ],
                "voting": {"method": "signed_participant_votes"},
                "votes": [alice_vote, bob_vote],
            },
        )

        result = self._validate()

        self.assertFalse(result.ok)
        self.assertIn(
            "approved round must include vote_result object",
            result.errors,
        )

    def test_auto_approved_round_status_is_not_supported_for_payout(self) -> None:
        alice_seed = generate_signing_seed()
        alice_public_key_b64 = public_key_b64_from_seed(alice_seed)
        alice_address = address_from_public_key_b64(alice_public_key_b64)
        bob_seed = generate_signing_seed()
        bob_public_key_b64 = public_key_b64_from_seed(bob_seed)
        bob_address = address_from_public_key_b64(bob_public_key_b64)
        alice_vote = self._signed_vote(
            round_id="year-1",
            github="alice",
            cai_address=alice_address,
            public_key_b64=alice_public_key_b64,
            signing_seed=alice_seed,
            choices=["bob"],
        )
        bob_vote = self._signed_vote(
            round_id="year-1",
            github="bob",
            cai_address=bob_address,
            public_key_b64=bob_public_key_b64,
            signing_seed=bob_seed,
            choices=["alice"],
        )
        self._write_json(
            self.participants_path,
            [
                {
                    "github": "alice",
                    "cai_address": alice_address,
                    "status": "active",
                },
                {
                    "github": "bob",
                    "cai_address": bob_address,
                    "status": "active",
                }
            ],
        )
        self._write_json(
            self.round_path,
            {
                "round_id": "year-1",
                "round_type": "top-7",
                "status": "auto_approved",
                "vote_result": {
                    "outcome": "passed",
                    "source": "https://github.com/example/cai/issues/1",
                    "completed_at": "2026-05-19T00:00:00+00:00",
                    "signed_vote_count": 2,
                    "tallies": {"alice": 1, "bob": 1},
                },
                "winners": [
                    {
                        "github": "alice",
                        "amount": "10000000.00000000",
                        "category": "top-7",
                    }
                ],
                "voting": {"method": "signed_participant_votes"},
                "votes": [alice_vote, bob_vote],
            },
        )

        result = self._validate()

        self.assertFalse(result.ok)
        self.assertIn(
            "round has unsupported status 'auto_approved'",
            result.errors,
        )

    def test_approved_round_rejects_tally_mismatch(self) -> None:
        alice_seed = generate_signing_seed()
        alice_public_key_b64 = public_key_b64_from_seed(alice_seed)
        alice_address = address_from_public_key_b64(alice_public_key_b64)
        bob_seed = generate_signing_seed()
        bob_public_key_b64 = public_key_b64_from_seed(bob_seed)
        bob_address = address_from_public_key_b64(bob_public_key_b64)
        alice_vote = self._signed_vote(
            round_id="year-1",
            github="alice",
            cai_address=alice_address,
            public_key_b64=alice_public_key_b64,
            signing_seed=alice_seed,
            choices=["bob"],
        )
        bob_vote = self._signed_vote(
            round_id="year-1",
            github="bob",
            cai_address=bob_address,
            public_key_b64=bob_public_key_b64,
            signing_seed=bob_seed,
            choices=["alice"],
        )
        self._write_json(
            self.participants_path,
            [
                {
                    "github": "alice",
                    "cai_address": alice_address,
                    "status": "active",
                },
                {
                    "github": "bob",
                    "cai_address": bob_address,
                    "status": "active",
                }
            ],
        )
        self._write_json(
            self.round_path,
            {
                "round_id": "year-1",
                "round_type": "top-7",
                "status": "approved",
                "voting": {"method": "signed_participant_votes"},
                "vote_result": {
                    "outcome": "passed",
                    "source": "https://github.com/example/cai/issues/1",
                    "completed_at": "2026-05-19T00:00:00+00:00",
                    "signed_vote_count": 2,
                    "tallies": {"alice": 2},
                },
                "winners": [
                    {
                        "github": "alice",
                        "amount": "10000000.00000000",
                        "category": "top-7",
                    }
                ],
                "votes": [alice_vote, bob_vote],
            },
        )

        result = self._validate()

        self.assertFalse(result.ok)
        self.assertIn(
            "approved round vote_result.tallies must match signed votes",
            result.errors,
        )

    def test_approved_round_rejects_self_vote(self) -> None:
        alice_seed = generate_signing_seed()
        alice_public_key_b64 = public_key_b64_from_seed(alice_seed)
        alice_address = address_from_public_key_b64(alice_public_key_b64)
        vote = self._signed_vote(
            round_id="year-1",
            github="alice",
            cai_address=alice_address,
            public_key_b64=alice_public_key_b64,
            signing_seed=alice_seed,
            choices=["alice"],
        )
        self._write_json(
            self.participants_path,
            [
                {"github": "alice", "cai_address": alice_address, "status": "active"},
            ],
        )
        self._write_json(
            self.round_path,
            {
                "round_id": "year-1",
                "round_type": "top-7",
                "status": "approved",
                "voting": {"method": "signed_participant_votes"},
                "vote_result": {
                    "outcome": "passed",
                    "source": "https://github.com/example/cai/issues/1",
                    "completed_at": "2026-05-19T00:00:00+00:00",
                    "signed_vote_count": 1,
                    "tallies": {"alice": 1},
                },
                "winners": [
                    {
                        "github": "alice",
                        "amount": "10000000.00000000",
                        "category": "top-7",
                    }
                ],
                "votes": [vote],
            },
        )

        result = self._validate()

        self.assertFalse(result.ok)
        self.assertIn("vote[alice] cannot vote for self", result.errors)

    def test_approved_round_rejects_winner_without_valid_vote(self) -> None:
        alice_seed = generate_signing_seed()
        alice_public_key_b64 = public_key_b64_from_seed(alice_seed)
        alice_address = address_from_public_key_b64(alice_public_key_b64)
        bob_seed = generate_signing_seed()
        bob_public_key_b64 = public_key_b64_from_seed(bob_seed)
        bob_address = address_from_public_key_b64(bob_public_key_b64)
        vote = self._signed_vote(
            round_id="year-1",
            github="bob",
            cai_address=bob_address,
            public_key_b64=bob_public_key_b64,
            signing_seed=bob_seed,
            choices=["alice"],
        )
        self._write_json(
            self.participants_path,
            [
                {"github": "alice", "cai_address": alice_address, "status": "active"},
                {"github": "bob", "cai_address": bob_address, "status": "active"},
            ],
        )
        self._write_json(
            self.round_path,
            {
                "round_id": "year-1",
                "round_type": "top-7",
                "status": "approved",
                "voting": {"method": "signed_participant_votes"},
                "vote_result": {
                    "outcome": "passed",
                    "source": "https://github.com/example/cai/issues/1",
                    "completed_at": "2026-05-19T00:00:00+00:00",
                    "signed_vote_count": 1,
                    "tallies": {"alice": 1},
                },
                "winners": [
                    {
                        "github": "alice",
                        "amount": "10000000.00000000",
                        "category": "top-7",
                    }
                ],
                "votes": [vote],
            },
        )

        result = self._validate()

        self.assertFalse(result.ok)
        self.assertIn(
            "winner 'alice' did not cast a valid signed vote",
            result.errors,
        )

    def test_duplicate_voter_uses_last_valid_signed_vote(self) -> None:
        wallets = self._wallets_for([f"p{i}" for i in range(1, 9)])
        first_vote = self._signed_vote(
            round_id="year-1",
            github="p1",
            cai_address=wallets["p1"]["address"],
            public_key_b64=wallets["p1"]["public_key_b64"],
            signing_seed=wallets["p1"]["signing_seed"],
            choices=["p8"],
        )
        last_vote = self._signed_vote(
            round_id="year-1",
            github="p1",
            cai_address=wallets["p1"]["address"],
            public_key_b64=wallets["p1"]["public_key_b64"],
            signing_seed=wallets["p1"]["signing_seed"],
            choices=["p2"],
        )
        votes = [
            first_vote,
            last_vote,
            self._vote_for(wallets=wallets, round_id="year-1", github="p2", choices=["p1"]),
            self._vote_for(wallets=wallets, round_id="year-1", github="p3", choices=["p4"]),
            self._vote_for(wallets=wallets, round_id="year-1", github="p4", choices=["p3"]),
            self._vote_for(wallets=wallets, round_id="year-1", github="p5", choices=["p6"]),
            self._vote_for(wallets=wallets, round_id="year-1", github="p6", choices=["p5"]),
            self._vote_for(wallets=wallets, round_id="year-1", github="p7", choices=["p1"]),
            self._vote_for(wallets=wallets, round_id="year-1", github="p8", choices=["p7"]),
        ]
        self._write_json(
            self.participants_path,
            self._participants_for(wallets),
        )
        self._write_json(
            self.round_path,
            {
                "round_id": "year-1",
                "round_type": "top-7",
                "status": "approved",
                "voting": {"method": "signed_participant_votes"},
                "vote_result": {
                    "outcome": "passed",
                    "source": "https://github.com/example/cai/issues/1",
                    "completed_at": "2026-05-19T00:00:00+00:00",
                    "signed_vote_count": 8,
                    "tallies": {
                        "p1": 2,
                        "p2": 1,
                        "p3": 1,
                        "p4": 1,
                        "p5": 1,
                        "p6": 1,
                        "p7": 1,
                    },
                },
                "winners": [
                    {
                        "github": "p1",
                        "amount": "10000000.00000000",
                        "category": "top-7",
                    },
                    {
                        "github": "p2",
                        "amount": "10000000.00000000",
                        "category": "top-7",
                    },
                    {
                        "github": "p3",
                        "amount": "10000000.00000000",
                        "category": "top-7",
                    },
                    {
                        "github": "p4",
                        "amount": "10000000.00000000",
                        "category": "top-7",
                    },
                    {
                        "github": "p5",
                        "amount": "10000000.00000000",
                        "category": "top-7",
                    },
                    {
                        "github": "p6",
                        "amount": "10000000.00000000",
                        "category": "top-7",
                    },
                    {
                        "github": "p7",
                        "amount": "10000000.00000000",
                        "category": "top-7",
                    }
                ],
                "votes": votes,
            },
        )

        result = self._validate()

        self.assertTrue(result.ok, result.errors)
        self.assertEqual(result.signed_vote_count, 8)

    def test_top_10_round_accepts_only_top_7_eligible_voters(self) -> None:
        voters = [f"v{i}" for i in range(1, 8)]
        winners = [f"d{i}" for i in range(1, 11)]
        wallets = self._wallets_for(voters + winners)
        votes = [
            self._vote_for(
                wallets=wallets,
                round_id="year-1-top-10",
                github="v1",
                choices=winners,
            )
        ]
        votes.extend(
            self._vote_for(
                wallets=wallets,
                round_id="year-1-top-10",
                github=voter,
                choices=["d1"],
            )
            for voter in voters[1:]
        )
        self._write_json(self.participants_path, self._participants_for(wallets))
        self._write_json(
            self.round_path,
            {
                "round_id": "year-1-top-10",
                "round_type": "top-10",
                "status": "approved",
                "voting": {
                    "method": "signed_participant_votes",
                    "eligible_voters": voters,
                },
                "vote_result": {
                    "outcome": "passed",
                    "source": "https://github.com/example/cai/issues/2",
                    "completed_at": "2026-05-19T00:00:00+00:00",
                    "signed_vote_count": 7,
                    "tallies": {
                        **{winner: 1 for winner in winners},
                        "d1": 7,
                    },
                },
                "winners": [
                    {
                        "github": winner,
                        "amount": "1000000.00000000",
                        "category": "top-10",
                    }
                    for winner in winners
                ],
                "votes": votes,
            },
        )

        result = self._validate()

        self.assertTrue(result.ok, result.errors)
        self.assertEqual(result.signed_vote_count, 7)

    def test_top_10_round_rejects_vote_outside_top_7_electorate(self) -> None:
        voters = [f"v{i}" for i in range(1, 8)]
        winners = [f"d{i}" for i in range(1, 11)]
        wallets = self._wallets_for(voters + winners)
        valid_votes = [
            self._vote_for(
                wallets=wallets,
                round_id="year-1-top-10",
                github="v1",
                choices=winners,
            )
        ]
        valid_votes.extend(
            self._vote_for(
                wallets=wallets,
                round_id="year-1-top-10",
                github=voter,
                choices=["d1"],
            )
            for voter in voters[1:]
        )
        outsider_vote = self._vote_for(
            wallets=wallets,
            round_id="year-1-top-10",
            github="d1",
            choices=["d2"],
        )
        self._write_json(self.participants_path, self._participants_for(wallets))
        self._write_json(
            self.round_path,
            {
                "round_id": "year-1-top-10",
                "round_type": "top-10",
                "status": "approved",
                "voting": {
                    "method": "signed_participant_votes",
                    "eligible_voters": voters,
                },
                "vote_result": {
                    "outcome": "passed",
                    "source": "https://github.com/example/cai/issues/2",
                    "completed_at": "2026-05-19T00:00:00+00:00",
                    "signed_vote_count": 7,
                    "tallies": {
                        **{winner: 1 for winner in winners},
                        "d1": 7,
                    },
                },
                "winners": [
                    {
                        "github": winner,
                        "amount": "1000000.00000000",
                        "category": "top-10",
                    }
                    for winner in winners
                ],
                "votes": valid_votes + [outsider_vote],
            },
        )

        result = self._validate()

        self.assertFalse(result.ok)
        self.assertIn("vote[d1] is not eligible for this round", result.errors)


if __name__ == "__main__":
    unittest.main()

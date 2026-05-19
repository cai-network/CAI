# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .model import MoneyPolicy
from .wallet import coins_to_atomic, normalize_address
from .wallet_signing import (
    ADDRESS_SCHEME_ED25519,
    HYBRID_ADDRESS_SCHEMES,
    address_from_public_key_b64,
    hybrid_address_from_public_keys_b64,
    verify_hybrid_payload_signature,
    verify_payload_signature,
)


DEFAULT_DEVELOPER_FUND_DIR = Path(".github") / "developer-fund"
DEFAULT_PARTICIPANTS_PATH = DEFAULT_DEVELOPER_FUND_DIR / "participants.json"
DEFAULT_ROUNDS_DIR = DEFAULT_DEVELOPER_FUND_DIR / "rounds"
DEFAULT_ROUND_FILE_NAME = "round-001.json"
APPROVED_ROUND_STATUSES = {"approved"}
APPROVED_VOTE_OUTCOMES = {"passed", "approved"}
FOUNDER_CONFIRMATION_STATUS_SIGNED = "signed"
SIGNED_VOTE_METHOD = "signed_participant_votes"
ROUND_TYPE_TOP_7 = "top-7"
ROUND_TYPE_TOP_10 = "top-10"
ROUND_TYPE_TOP_100 = "top-100"
ROUND_TYPE_CODE_AUTHORS = "code-authors"
KNOWN_ROUND_TYPES = {
    ROUND_TYPE_TOP_7,
    ROUND_TYPE_TOP_10,
    ROUND_TYPE_TOP_100,
    ROUND_TYPE_CODE_AUTHORS,
}
ROUND_TOTAL_BPS = {
    ROUND_TYPE_TOP_7: 700,
    ROUND_TYPE_TOP_10: 100,
    ROUND_TYPE_TOP_100: 100,
    ROUND_TYPE_CODE_AUTHORS: 100,
}
ROUND_WINNER_COUNTS = {
    ROUND_TYPE_TOP_7: 7,
    ROUND_TYPE_TOP_10: 10,
    ROUND_TYPE_TOP_100: 100,
}
ROUND_ELIGIBLE_VOTER_COUNTS = {
    ROUND_TYPE_TOP_10: 7,
    ROUND_TYPE_TOP_100: 17,
}
CODE_AUTHORS_MAX_WINNERS = 1000
CODE_AUTHORS_VOTER_THRESHOLD = 1000
CODE_AUTHORS_ELIGIBLE_VOTER_COUNT = 100
KNOWN_ROUND_STATUSES = {
    "draft",
    "voting",
    "approved",
    "paid",
    "rejected",
    "cancelled",
}
GITHUB_LOGIN_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")


@dataclass(frozen=True)
class DeveloperFundParticipant:
    github: str
    cai_address: str
    status: str = "active"
    display_name: str | None = None


@dataclass(frozen=True)
class DeveloperFundRecipient:
    github: str
    address: str
    amount_atomic: int
    amount_coins: str
    category: str
    reason: str


@dataclass(frozen=True)
class DeveloperFundValidationResult:
    participants_path: Path
    round_path: Path
    round_id: str
    round_status: str
    participants_hash: str
    round_hash: str
    participants_count: int
    winner_count: int
    total_amount_atomic: int
    round_type: str = ""
    vote_outcome: str = ""
    founder_confirmation_status: str = ""
    signed_vote_count: int = 0
    recipients: tuple[DeveloperFundRecipient, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def distributable(self) -> bool:
        return self.ok and self.round_status in APPROVED_ROUND_STATUSES


def developer_fund_participants_path(repo_root: Path | str = ".") -> Path:
    return Path(repo_root) / DEFAULT_PARTICIPANTS_PATH


def developer_fund_default_round_path(repo_root: Path | str = ".") -> Path:
    return Path(repo_root) / DEFAULT_ROUNDS_DIR / DEFAULT_ROUND_FILE_NAME


def resolve_developer_fund_path(
    repo_root: Path | str,
    path: Path | str | None,
    default_path: Path,
) -> Path:
    root = Path(repo_root)
    resolved = Path(path) if path is not None else default_path
    if not resolved.is_absolute():
        resolved = root / resolved
    return resolved


def canonical_json_hash(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _github_key(value: str) -> str:
    return str(value).strip().lstrip("@").lower()


def _participant_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("participants"), list):
        return [item for item in payload["participants"] if isinstance(item, dict)]
    return []


def _validate_participants(
    payload: Any,
) -> tuple[dict[str, DeveloperFundParticipant], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    participants_by_github: dict[str, DeveloperFundParticipant] = {}
    addresses_by_normalized: dict[str, str] = {}

    if not isinstance(payload, (list, dict)):
        return {}, ["participants file must be a JSON array or object"], []
    items = _participant_items(payload)
    has_empty_array = payload == []
    has_participants_array = (
        isinstance(payload, dict) and isinstance(payload.get("participants"), list)
    )
    if not items and not has_empty_array and not has_participants_array:
        errors.append("participants file has no participants array")

    for index, item in enumerate(items):
        github_raw = str(item.get("github") or "").strip()
        github = _github_key(github_raw)
        if not github:
            errors.append(f"participant[{index}] is missing github")
            continue
        if not GITHUB_LOGIN_RE.match(github):
            errors.append(f"participant[{index}] has invalid github '{github_raw}'")
            continue
        if github in participants_by_github:
            errors.append(f"duplicate participant github '{github}'")
            continue

        status = str(item.get("status") or "active").strip().lower()
        if status not in {"active", "inactive", "superseded"}:
            errors.append(f"participant[{github}] has unsupported status '{status}'")

        address = normalize_address(str(item.get("cai_address") or "").strip())
        if not address:
            errors.append(f"participant[{github}] is missing cai_address")
            continue
        if address in addresses_by_normalized:
            errors.append(
                "duplicate participant cai_address "
                f"'{address}' for '{addresses_by_normalized[address]}' and '{github}'"
            )
            continue
        addresses_by_normalized[address] = github

        participants_by_github[github] = DeveloperFundParticipant(
            github=github,
            cai_address=address,
            status=status,
            display_name=(
                str(item.get("display_name")).strip()
                if item.get("display_name") is not None
                else None
            ),
        )

    if not participants_by_github:
        warnings.append("participants registry is empty")
    return participants_by_github, errors, warnings


def _round_winner_items(payload: Any) -> list[dict[str, Any]]:
    winners = payload.get("winners") if isinstance(payload, dict) else None
    return (
        [item for item in winners if isinstance(item, dict)]
        if isinstance(winners, list)
        else []
    )


def developer_fund_vote_signing_payload(
    *,
    round_id: str,
    github: str,
    cai_address: str,
    choices: list[str],
) -> dict[str, Any]:
    return {
        "type": "cai_developer_fund_vote",
        "version": 1,
        "round_id": str(round_id).strip(),
        "github": _github_key(github),
        "cai_address": normalize_address(cai_address),
        "choices": [_github_key(choice) for choice in choices],
    }


def _round_vote_items(payload: Any) -> list[dict[str, Any]]:
    votes = payload.get("votes") if isinstance(payload, dict) else None
    return (
        [item for item in votes if isinstance(item, dict)]
        if isinstance(votes, list)
        else []
    )


def _round_confirmation_voting_payload(payload: dict[str, Any]) -> dict[str, Any]:
    voting = payload.get("voting")
    if not isinstance(voting, dict):
        return {}
    confirmation_keys = (
        "method",
        "eligible_voters",
        "founder_timeout_days",
    )
    return {key: voting[key] for key in confirmation_keys if key in voting}


def _round_confirmation_decision_payload(payload: dict[str, Any]) -> dict[str, Any]:
    vote_result = payload.get("vote_result")
    return {
        "round_id": str(payload.get("round_id") or "").strip(),
        "round_type": _round_type_key(payload),
        "candidate_count": payload.get("candidate_count"),
        "status": str(payload.get("status") or "").strip().lower(),
        "github_issue": str(payload.get("github_issue") or "").strip(),
        "voting": _round_confirmation_voting_payload(payload),
        "vote_result": vote_result if isinstance(vote_result, dict) else {},
        "winners": _round_winner_items(payload),
        "votes_hash": canonical_json_hash(_round_vote_items(payload)),
    }


def developer_fund_founder_confirmation_signing_payload(
    *,
    round_payload: dict[str, Any],
    participants_hash: str,
    confirmed_by: str,
    confirmed_at: str,
) -> dict[str, Any]:
    return {
        "type": "cai_developer_fund_founder_confirmation",
        "version": 1,
        "confirmed_by": str(confirmed_by).strip(),
        "confirmed_at": str(confirmed_at).strip(),
        "round_id": str(round_payload.get("round_id") or "").strip(),
        "round_type": _round_type_key(round_payload),
        "round_status": str(round_payload.get("status") or "").strip().lower(),
        "participants_hash": str(participants_hash).strip(),
        "round_decision_hash": canonical_json_hash(
            _round_confirmation_decision_payload(round_payload)
        ),
    }


def _round_type_key(payload: dict[str, Any]) -> str:
    return str(payload.get("round_type") or "").strip().lower()


def _round_candidate_count(payload: dict[str, Any], winner_count: int) -> int:
    value = payload.get("candidate_count")
    if value is None:
        return winner_count
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return -1


def _round_eligible_voters(
    payload: dict[str, Any],
) -> tuple[set[str] | None, list[str]]:
    voting = payload.get("voting")
    raw_voters = None
    if isinstance(voting, dict):
        raw_voters = voting.get("eligible_voters")
    if raw_voters is None:
        raw_voters = payload.get("eligible_voters")
    if raw_voters is None:
        return None, []
    if not isinstance(raw_voters, list):
        return set(), ["voting.eligible_voters must be an array"]

    errors: list[str] = []
    voters: set[str] = set()
    for index, item in enumerate(raw_voters):
        github = _github_key(str(item))
        if not github:
            errors.append(f"voting.eligible_voters[{index}] is empty")
            continue
        if github in voters:
            errors.append(f"duplicate eligible voter '{github}'")
            continue
        voters.add(github)
    return voters, errors


def _total_supply_percent_atomic(
    policy: MoneyPolicy,
    *,
    bps: int,
) -> int:
    total_supply_atomic = coins_to_atomic(str(policy.total_supply_coins), policy)
    return (total_supply_atomic * int(bps)) // 10_000


def _expected_equal_amounts(total_atomic: int, count: int) -> list[int]:
    if count <= 0:
        return []
    base = total_atomic // count
    remainder = total_atomic % count
    return [base + 1] * remainder + [base] * (count - remainder)


def _requires_signed_votes(round_type: str, candidate_count: int) -> bool:
    if round_type == ROUND_TYPE_CODE_AUTHORS:
        return candidate_count > CODE_AUTHORS_VOTER_THRESHOLD
    return round_type in {
        ROUND_TYPE_TOP_7,
        ROUND_TYPE_TOP_10,
        ROUND_TYPE_TOP_100,
    }


def _expected_eligible_voter_count(round_type: str, candidate_count: int) -> int | None:
    if round_type == ROUND_TYPE_CODE_AUTHORS:
        if candidate_count > CODE_AUTHORS_VOTER_THRESHOLD:
            return CODE_AUTHORS_ELIGIBLE_VOTER_COUNT
        return None
    return ROUND_ELIGIBLE_VOTER_COUNTS.get(round_type)


def _validate_signed_votes(
    round_payload: dict[str, Any],
    *,
    round_id: str,
    participants_by_github: dict[str, DeveloperFundParticipant],
    eligible_voters: set[str] | None = None,
) -> tuple[int, dict[str, int], set[str], list[str]]:
    errors: list[str] = []
    tallies: dict[str, int] = {}
    voting = round_payload.get("voting")
    method = (
        str(voting.get("method") or "").strip()
        if isinstance(voting, dict)
        else ""
    )
    if method != SIGNED_VOTE_METHOD:
        errors.append(
            f"approved round voting.method must be {SIGNED_VOTE_METHOD}"
        )
        return 0, tallies, set(), errors

    if not isinstance(round_payload.get("votes", []), list):
        errors.append("round votes must be an array")
        return 0, tallies, set(), errors

    valid_votes_by_github: dict[str, list[str]] = {}
    for index, vote in enumerate(_round_vote_items(round_payload)):
        github = _github_key(str(vote.get("github") or ""))
        if not github:
            errors.append(f"vote[{index}] is missing github")
            continue

        participant = participants_by_github.get(github)
        if participant is None:
            errors.append(f"vote[{github}] is not registered in participants")
            continue
        if participant.status != "active":
            errors.append(f"vote[{github}] participant is not active")
            continue
        if eligible_voters is not None and github not in eligible_voters:
            errors.append(f"vote[{github}] is not eligible for this round")
            continue

        raw_choices = vote.get("choices")
        if not isinstance(raw_choices, list) or not raw_choices:
            errors.append(f"vote[{github}] must contain non-empty choices")
            continue
        choices = [_github_key(str(choice)) for choice in raw_choices]
        if any(not choice for choice in choices):
            errors.append(f"vote[{github}] contains an empty choice")
            continue
        if len(choices) != len(set(choices)):
            errors.append(f"vote[{github}] contains duplicate choices")
            continue
        if github in choices:
            errors.append(f"vote[{github}] cannot vote for self")
            continue
        missing_choices = [
            choice
            for choice in choices
            if choice not in participants_by_github
        ]
        if missing_choices:
            errors.append(
                f"vote[{github}] contains unregistered choices: "
                f"{', '.join(missing_choices)}"
            )
            continue

        public_key_b64 = str(vote.get("public_key_b64") or "").strip()
        signature_b64 = str(vote.get("signature_b64") or "").strip()
        address_scheme = str(vote.get("address_scheme") or "").strip()
        if not public_key_b64 or not signature_b64 or not address_scheme:
            errors.append(
                f"vote[{github}] requires public_key_b64, signature_b64 "
                "and address_scheme"
            )
            continue

        try:
            if address_scheme == ADDRESS_SCHEME_ED25519:
                signer_address = address_from_public_key_b64(public_key_b64)
                signature_ok = verify_payload_signature(
                    public_key_b64=public_key_b64,
                    signature_b64=signature_b64,
                    payload=developer_fund_vote_signing_payload(
                        round_id=round_id,
                        github=github,
                        cai_address=participant.cai_address,
                        choices=choices,
                    ),
                )
            elif address_scheme in HYBRID_ADDRESS_SCHEMES:
                pq_public_key_b64 = str(vote.get("pq_public_key_b64") or "").strip()
                pq_signature_b64 = str(vote.get("pq_signature_b64") or "").strip()
                if not pq_public_key_b64 or not pq_signature_b64:
                    errors.append(
                        f"vote[{github}] requires pq_public_key_b64 "
                        "and pq_signature_b64"
                    )
                    continue
                signer_address = hybrid_address_from_public_keys_b64(
                    ed25519_public_key_b64=public_key_b64,
                    pq_public_key_b64=pq_public_key_b64,
                    address_scheme=address_scheme,
                )
                signature_ok = verify_hybrid_payload_signature(
                    ed25519_public_key_b64=public_key_b64,
                    ed25519_signature_b64=signature_b64,
                    pq_public_key_b64=pq_public_key_b64,
                    pq_signature_b64=pq_signature_b64,
                    payload=developer_fund_vote_signing_payload(
                        round_id=round_id,
                        github=github,
                        cai_address=participant.cai_address,
                        choices=choices,
                    ),
                )
            else:
                errors.append(f"vote[{github}] has unsupported address_scheme")
                continue
        except Exception as exc:
            errors.append(f"vote[{github}] cannot be verified: {exc}")
            continue

        if normalize_address(signer_address) != participant.cai_address:
            errors.append(
                f"vote[{github}] signer address does not match participants registry"
            )
            continue
        if not signature_ok:
            errors.append(f"vote[{github}] has invalid signature")
            continue
        valid_votes_by_github[github] = choices

    signed_vote_count = len(valid_votes_by_github)
    for choices in valid_votes_by_github.values():
        for choice in choices:
            tallies[choice] = tallies.get(choice, 0) + 1

    if signed_vote_count <= 0:
        errors.append("approved round requires at least one valid signed vote")
    return signed_vote_count, tallies, set(valid_votes_by_github), errors


def _validate_founder_confirmation(
    round_payload: dict[str, Any],
    *,
    participants_hash: str,
    money_policy: MoneyPolicy,
) -> tuple[str, list[str]]:
    errors: list[str] = []
    confirmation = round_payload.get("founder_confirmation")
    if not isinstance(confirmation, dict):
        return "", ["approved round must include founder_confirmation object"]

    confirmed_by = str(confirmation.get("confirmed_by") or "").strip()
    confirmed_at = str(confirmation.get("confirmed_at") or "").strip()
    if not confirmed_by:
        errors.append("founder_confirmation.confirmed_by is required")
    if not confirmed_at:
        errors.append("founder_confirmation.confirmed_at is required")

    public_key_b64 = str(confirmation.get("public_key_b64") or "").strip()
    signature_b64 = str(confirmation.get("signature_b64") or "").strip()
    address_scheme = str(confirmation.get("address_scheme") or "").strip()
    if not public_key_b64 or not signature_b64 or not address_scheme:
        errors.append(
            "founder_confirmation requires public_key_b64, signature_b64 "
            "and address_scheme"
        )
        return "", errors
    if not confirmed_by or not confirmed_at:
        return "", errors

    payload = developer_fund_founder_confirmation_signing_payload(
        round_payload=round_payload,
        participants_hash=participants_hash,
        confirmed_by=confirmed_by,
        confirmed_at=confirmed_at,
    )
    try:
        if address_scheme == ADDRESS_SCHEME_ED25519:
            signer_address = address_from_public_key_b64(public_key_b64)
            signature_ok = verify_payload_signature(
                public_key_b64=public_key_b64,
                signature_b64=signature_b64,
                payload=payload,
            )
        elif address_scheme in HYBRID_ADDRESS_SCHEMES:
            pq_public_key_b64 = str(
                confirmation.get("pq_public_key_b64") or ""
            ).strip()
            pq_signature_b64 = str(
                confirmation.get("pq_signature_b64") or ""
            ).strip()
            if not pq_public_key_b64 or not pq_signature_b64:
                errors.append(
                    "founder_confirmation requires pq_public_key_b64 "
                    "and pq_signature_b64"
                )
                return "", errors
            signer_address = hybrid_address_from_public_keys_b64(
                ed25519_public_key_b64=public_key_b64,
                pq_public_key_b64=pq_public_key_b64,
                address_scheme=address_scheme,
            )
            signature_ok = verify_hybrid_payload_signature(
                ed25519_public_key_b64=public_key_b64,
                ed25519_signature_b64=signature_b64,
                pq_public_key_b64=pq_public_key_b64,
                pq_signature_b64=pq_signature_b64,
                payload=payload,
            )
        else:
            errors.append("founder_confirmation has unsupported address_scheme")
            return "", errors
    except Exception as exc:
        errors.append(f"founder_confirmation cannot be verified: {exc}")
        return "", errors

    expected_founder_address = normalize_address(money_policy.developer_treasury_address)
    if normalize_address(signer_address) != expected_founder_address:
        errors.append(
            "founder_confirmation signer address does not match "
            "developer treasury address"
        )
    if not signature_ok:
        errors.append("founder_confirmation has invalid signature")
    return (
        FOUNDER_CONFIRMATION_STATUS_SIGNED if not errors else "",
        errors,
    )


def _validate_round_decision(
    round_payload: dict[str, Any],
    *,
    round_status: str,
    requires_signed_votes: bool,
    signed_vote_count: int,
    signed_vote_tallies: dict[str, int],
    participants_hash: str,
    money_policy: MoneyPolicy,
) -> tuple[str, str, list[str]]:
    errors: list[str] = []
    vote_result = round_payload.get("vote_result")
    if "approval" in round_payload:
        errors.append(
            "round must use founder_confirmation or timeout instead of legacy "
            "approval object"
        )

    if round_status not in APPROVED_ROUND_STATUSES:
        vote_outcome = ""
        if isinstance(vote_result, dict):
            vote_outcome = str(vote_result.get("outcome") or "").strip().lower()
        return vote_outcome, "", errors

    if requires_signed_votes and not isinstance(vote_result, dict):
        errors.append("approved round must include vote_result object")
        vote_outcome = ""
    elif isinstance(vote_result, dict):
        vote_outcome = str(vote_result.get("outcome") or "").strip().lower()
        if requires_signed_votes and vote_outcome not in APPROVED_VOTE_OUTCOMES:
            errors.append("approved round vote_result.outcome must be passed")
        if requires_signed_votes and not str(vote_result.get("source") or "").strip():
            errors.append("approved round vote_result.source is required")
        if requires_signed_votes and not str(vote_result.get("completed_at") or "").strip():
            errors.append("approved round vote_result.completed_at is required")
        if requires_signed_votes and "signed_vote_count" not in vote_result:
            errors.append("approved round vote_result.signed_vote_count is required")
        elif requires_signed_votes:
            expected_vote_count = vote_result.get("signed_vote_count")
            try:
                if int(expected_vote_count) != int(signed_vote_count):
                    errors.append(
                        "approved round vote_result.signed_vote_count must match "
                        "the number of valid signed votes"
                    )
            except (TypeError, ValueError):
                errors.append("approved round vote_result.signed_vote_count is invalid")
        raw_tallies = vote_result.get("tallies")
        if requires_signed_votes and not isinstance(raw_tallies, dict):
            errors.append("approved round vote_result.tallies is required")
        elif requires_signed_votes:
            expected_tallies: dict[str, int] = {}
            for key, value in raw_tallies.items():
                try:
                    expected_tallies[_github_key(str(key))] = int(value)
                except (TypeError, ValueError):
                    errors.append(f"approved round vote_result.tallies[{key}] is invalid")
            if expected_tallies != signed_vote_tallies:
                errors.append(
                    "approved round vote_result.tallies must match signed votes"
                )
    else:
        vote_outcome = ""

    if round_status == "approved":
        founder_status, confirmation_errors = _validate_founder_confirmation(
            round_payload,
            participants_hash=participants_hash,
            money_policy=money_policy,
        )
        errors.extend(confirmation_errors)
        return vote_outcome, founder_status, errors

    return vote_outcome, "", errors


def _validate_structured_round_rules(
    *,
    round_type: str,
    round_status: str,
    candidate_count: int,
    eligible_voters: set[str] | None,
    founder_confirmation_status: str,
    participants_by_github: dict[str, DeveloperFundParticipant],
    recipients: list[DeveloperFundRecipient],
    total_amount_atomic: int,
    money_policy: MoneyPolicy,
) -> list[str]:
    errors: list[str] = []
    if round_status not in APPROVED_ROUND_STATUSES:
        return errors

    if not round_type:
        return ["approved round must include round_type"]
    if round_type not in KNOWN_ROUND_TYPES:
        return [f"round_type '{round_type}' is not supported"]

    if (
        round_status == "approved"
        and founder_confirmation_status != FOUNDER_CONFIRMATION_STATUS_SIGNED
    ):
        errors.append(
            f"{round_type} round requires signed founder_confirmation"
        )

    if candidate_count < 0:
        errors.append("round candidate_count is invalid")

    expected_voter_count = _expected_eligible_voter_count(round_type, candidate_count)
    if expected_voter_count is None:
        if round_type != ROUND_TYPE_TOP_7 and eligible_voters:
            for github in eligible_voters:
                participant = participants_by_github.get(github)
                if participant is None or participant.status != "active":
                    errors.append(
                        f"eligible voter '{github}' is not an active participant"
                    )
    else:
        if eligible_voters is None:
            errors.append(
                f"{round_type} round requires voting.eligible_voters"
            )
        elif len(eligible_voters) != expected_voter_count:
            errors.append(
                f"{round_type} round requires exactly "
                f"{expected_voter_count} eligible voters"
            )
        if eligible_voters:
            for github in eligible_voters:
                participant = participants_by_github.get(github)
                if participant is None or participant.status != "active":
                    errors.append(
                        f"eligible voter '{github}' is not an active participant"
                    )

    expected_winner_count = ROUND_WINNER_COUNTS.get(round_type)
    if expected_winner_count is not None:
        if len(recipients) != expected_winner_count:
            errors.append(
                f"{round_type} round requires exactly "
                f"{expected_winner_count} winners"
            )
    elif round_type == ROUND_TYPE_CODE_AUTHORS:
        if len(recipients) > CODE_AUTHORS_MAX_WINNERS:
            errors.append(
                f"{ROUND_TYPE_CODE_AUTHORS} round cannot include more than "
                f"{CODE_AUTHORS_MAX_WINNERS} winners"
            )
        if candidate_count > 0 and candidate_count <= CODE_AUTHORS_VOTER_THRESHOLD:
            if len(recipients) != candidate_count:
                errors.append(
                    f"{ROUND_TYPE_CODE_AUTHORS} round must include every code author "
                    "when candidate_count is not above the voting threshold"
                )
        if candidate_count > CODE_AUTHORS_VOTER_THRESHOLD:
            if len(recipients) != CODE_AUTHORS_MAX_WINNERS:
                errors.append(
                    f"{ROUND_TYPE_CODE_AUTHORS} round requires exactly "
                    f"{CODE_AUTHORS_MAX_WINNERS} winners when candidate_count "
                    "is above the voting threshold"
                )

    for recipient in recipients:
        if recipient.category != round_type:
            errors.append(
                f"winner '{recipient.github}' category must be {round_type}"
            )

    expected_total_atomic = _total_supply_percent_atomic(
        money_policy,
        bps=ROUND_TOTAL_BPS[round_type],
    )
    if total_amount_atomic != expected_total_atomic:
        errors.append(
            f"{round_type} round total must be "
            f"{expected_total_atomic} atomic units"
        )

    if recipients:
        expected_amounts = sorted(
            _expected_equal_amounts(expected_total_atomic, len(recipients))
        )
        actual_amounts = sorted(recipient.amount_atomic for recipient in recipients)
        if actual_amounts != expected_amounts:
            errors.append(f"{round_type} round amounts must be split equally")

    return errors


def validate_developer_fund_files(
    *,
    repo_root: Path | str = ".",
    participants_path: Path | str | None = None,
    round_path: Path | str | None = None,
    money_policy: MoneyPolicy | None = None,
) -> DeveloperFundValidationResult:
    root = Path(repo_root)
    resolved_participants_path = resolve_developer_fund_path(
        root,
        participants_path,
        DEFAULT_PARTICIPANTS_PATH,
    )
    resolved_round_path = resolve_developer_fund_path(
        root,
        round_path,
        DEFAULT_ROUNDS_DIR / DEFAULT_ROUND_FILE_NAME,
    )
    active_money_policy = money_policy or MoneyPolicy()
    errors: list[str] = []
    warnings: list[str] = []

    if not resolved_participants_path.exists():
        return DeveloperFundValidationResult(
            participants_path=resolved_participants_path,
            round_path=resolved_round_path,
            round_id="",
            round_status="missing",
            participants_hash="",
            round_hash="",
            participants_count=0,
            winner_count=0,
            total_amount_atomic=0,
            errors=(f"participants file does not exist: {resolved_participants_path}",),
        )
    if not resolved_round_path.exists():
        return DeveloperFundValidationResult(
            participants_path=resolved_participants_path,
            round_path=resolved_round_path,
            round_id="",
            round_status="missing",
            participants_hash="",
            round_hash="",
            participants_count=0,
            winner_count=0,
            total_amount_atomic=0,
            errors=(f"round file does not exist: {resolved_round_path}",),
        )

    try:
        participants_payload = _load_json(resolved_participants_path)
    except Exception as exc:  # pragma: no cover - json error text varies by Python.
        return DeveloperFundValidationResult(
            participants_path=resolved_participants_path,
            round_path=resolved_round_path,
            round_id="",
            round_status="invalid",
            participants_hash="",
            round_hash="",
            participants_count=0,
            winner_count=0,
            total_amount_atomic=0,
            errors=(f"participants file is not valid JSON: {exc}",),
        )
    try:
        round_payload = _load_json(resolved_round_path)
    except Exception as exc:  # pragma: no cover - json error text varies by Python.
        return DeveloperFundValidationResult(
            participants_path=resolved_participants_path,
            round_path=resolved_round_path,
            round_id="",
            round_status="invalid",
            participants_hash=canonical_json_hash(participants_payload),
            round_hash="",
            participants_count=0,
            winner_count=0,
            total_amount_atomic=0,
            errors=(f"round file is not valid JSON: {exc}",),
        )

    participants_by_github, participant_errors, participant_warnings = (
        _validate_participants(participants_payload)
    )
    errors.extend(participant_errors)
    warnings.extend(participant_warnings)

    if not isinstance(round_payload, dict):
        errors.append("round file must be a JSON object")
        round_payload = {}

    round_id = str(round_payload.get("round_id") or "").strip()
    if not round_id:
        errors.append("round is missing round_id")
        round_id = "<missing>"

    round_status = str(round_payload.get("status") or "draft").strip().lower()
    if round_status not in KNOWN_ROUND_STATUSES:
        errors.append(f"round has unsupported status '{round_status}'")
    round_type = _round_type_key(round_payload)
    if round_type and round_type not in KNOWN_ROUND_TYPES:
        errors.append(f"round_type '{round_type}' is not supported")
    eligible_voters, eligible_voter_errors = _round_eligible_voters(round_payload)
    errors.extend(eligible_voter_errors)
    candidate_count = _round_candidate_count(round_payload, len(_round_winner_items(round_payload)))
    requires_signed_votes = (
        round_status in APPROVED_ROUND_STATUSES
        and round_type in KNOWN_ROUND_TYPES
        and _requires_signed_votes(round_type, candidate_count)
    )
    signed_vote_count = 0
    signed_vote_tallies: dict[str, int] = {}
    signed_vote_voters: set[str] = set()
    if round_status in APPROVED_ROUND_STATUSES and requires_signed_votes:
        (
            signed_vote_count,
            signed_vote_tallies,
            signed_vote_voters,
            vote_errors,
        ) = _validate_signed_votes(
            round_payload,
            round_id=round_id,
            participants_by_github=participants_by_github,
            eligible_voters=eligible_voters,
        )
        errors.extend(vote_errors)
    participants_hash = canonical_json_hash(participants_payload)
    vote_outcome, founder_confirmation_status, decision_errors = (
        _validate_round_decision(
            round_payload,
            round_status=round_status,
            requires_signed_votes=requires_signed_votes,
            signed_vote_count=signed_vote_count,
            signed_vote_tallies=signed_vote_tallies,
            participants_hash=participants_hash,
            money_policy=active_money_policy,
        )
    )
    errors.extend(decision_errors)

    winners = _round_winner_items(round_payload)
    if not isinstance(round_payload.get("winners", []), list):
        errors.append("round winners must be an array")

    seen_winners: set[str] = set()
    recipients: list[DeveloperFundRecipient] = []
    total_amount_atomic = 0
    for index, winner in enumerate(winners):
        github = _github_key(str(winner.get("github") or ""))
        if not github:
            errors.append(f"winner[{index}] is missing github")
            continue
        if github in seen_winners:
            errors.append(f"duplicate round winner '{github}'")
            continue
        seen_winners.add(github)

        participant = participants_by_github.get(github)
        if participant is None:
            errors.append(f"winner '{github}' is not registered in participants")
            continue
        if participant.status != "active":
            errors.append(f"winner '{github}' is not active in participants")
            continue

        if "cai_address" in winner:
            winner_address = normalize_address(str(winner.get("cai_address") or ""))
            if winner_address and winner_address != participant.cai_address:
                errors.append(
                    f"winner '{github}' cai_address does not match participants registry"
                )
                continue

        amount_coins = str(winner.get("amount") or "").strip()
        if not amount_coins:
            errors.append(f"winner '{github}' is missing amount")
            continue
        try:
            amount_atomic = coins_to_atomic(amount_coins, active_money_policy)
        except Exception as exc:
            errors.append(f"winner '{github}' has invalid amount '{amount_coins}': {exc}")
            continue
        if amount_atomic <= 0:
            errors.append(f"winner '{github}' amount must be positive")
            continue

        category = str(winner.get("category") or "general").strip() or "general"
        reason = str(winner.get("reason") or "").strip()
        recipients.append(
            DeveloperFundRecipient(
                github=github,
                address=participant.cai_address,
                amount_atomic=amount_atomic,
                amount_coins=amount_coins,
                category=category,
                reason=reason,
            )
        )
        total_amount_atomic += amount_atomic

    if round_status in APPROVED_ROUND_STATUSES and not recipients:
        errors.append("approved round must contain at least one winner")
    if round_status in APPROVED_ROUND_STATUSES and requires_signed_votes:
        for recipient in recipients:
            if signed_vote_tallies.get(recipient.github, 0) <= 0:
                errors.append(
                    f"winner '{recipient.github}' has no signed votes in vote_result"
                )
            winner_can_vote_in_round = (
                eligible_voters is None or recipient.github in eligible_voters
            )
            if winner_can_vote_in_round and recipient.github not in signed_vote_voters:
                errors.append(
                    f"winner '{recipient.github}' did not cast a valid signed vote"
                )

    fund_total_atomic = coins_to_atomic(
        str(active_money_policy.developer_contribution_fund_coins),
        active_money_policy,
    )
    if total_amount_atomic > fund_total_atomic:
        errors.append("round total exceeds the developer contribution fund total")
    errors.extend(
        _validate_structured_round_rules(
            round_type=round_type,
            round_status=round_status,
            candidate_count=candidate_count,
            eligible_voters=eligible_voters,
            founder_confirmation_status=founder_confirmation_status,
            participants_by_github=participants_by_github,
            recipients=recipients,
            total_amount_atomic=total_amount_atomic,
            money_policy=active_money_policy,
        )
    )

    return DeveloperFundValidationResult(
        participants_path=resolved_participants_path,
        round_path=resolved_round_path,
        round_id=round_id,
        round_status=round_status,
        round_type=round_type,
        participants_hash=participants_hash,
        round_hash=canonical_json_hash(round_payload),
        participants_count=len(participants_by_github),
        winner_count=len(recipients),
        total_amount_atomic=total_amount_atomic,
        vote_outcome=vote_outcome,
        founder_confirmation_status=founder_confirmation_status,
        signed_vote_count=signed_vote_count,
        recipients=tuple(recipients),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def developer_fund_recipients_for_chain(
    result: DeveloperFundValidationResult,
) -> list[dict[str, Any]]:
    return [
        {
            "github": recipient.github,
            "address": recipient.address,
            "amount_atomic": recipient.amount_atomic,
            "amount_coins": recipient.amount_coins,
            "category": recipient.category,
            "reason": recipient.reason,
        }
        for recipient in result.recipients
    ]

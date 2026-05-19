# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import secrets

LEGACY_SEED_WORD_COUNT = 12
DEFAULT_SEED_WORD_COUNT = 32
SUPPORTED_SEED_WORD_COUNTS = (LEGACY_SEED_WORD_COUNT, DEFAULT_SEED_WORD_COUNT)

_PREFIXES = (
    "al",
    "be",
    "cor",
    "den",
    "el",
    "far",
    "gan",
    "hal",
    "is",
    "jor",
    "kel",
    "lor",
    "mor",
    "nel",
    "or",
    "pra",
)

_SUFFIXES = (
    "ba",
    "den",
    "fin",
    "gor",
    "hal",
    "ian",
    "jor",
    "kel",
    "lin",
    "mor",
    "nor",
    "or",
    "pra",
    "quil",
    "ran",
    "sor",
)

SEED_WORDS: tuple[str, ...] = tuple(
    f"{prefix}{suffix}" for prefix in _PREFIXES for suffix in _SUFFIXES
)
_SEED_WORD_SET = set(SEED_WORDS)


def normalize_seed_phrase(seed_phrase: str) -> str:
    return " ".join(seed_phrase.strip().lower().split())


def validate_seed_phrase(
    seed_phrase: str, *, word_count: int | None = None
) -> str:
    normalized = normalize_seed_phrase(seed_phrase)
    words = normalized.split()
    allowed_word_counts = (
        (int(word_count),)
        if word_count is not None
        else SUPPORTED_SEED_WORD_COUNTS
    )
    if len(words) not in allowed_word_counts:
        if word_count is not None:
            raise ValueError(f"Seed phrase must contain exactly {word_count} words.")
        expected = ", ".join(str(count) for count in allowed_word_counts)
        raise ValueError(f"Seed phrase must contain one of these word counts: {expected}.")
    invalid = [word for word in words if word not in _SEED_WORD_SET]
    if invalid:
        raise ValueError(f"Seed phrase contains unknown words: {', '.join(invalid[:3])}.")
    return normalized


def generate_seed_phrase(word_count: int = DEFAULT_SEED_WORD_COUNT) -> str:
    return " ".join(SEED_WORDS[index] for index in secrets.token_bytes(word_count))


def seed_fingerprint(seed_phrase: str) -> str:
    normalized = validate_seed_phrase(seed_phrase)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def derive_seed_wallet_id(seed_phrase: str) -> str:
    normalized = validate_seed_phrase(seed_phrase)
    return hashlib.sha256(f"cai-seed-v1:{normalized}".encode("utf-8")).hexdigest()[:32]

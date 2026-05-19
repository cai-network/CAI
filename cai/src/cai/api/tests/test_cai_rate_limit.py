# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from cai.api.rate_limit import InMemoryFixedWindowRateLimiter


def test_fixed_window_rate_limiter_allows_until_limit() -> None:
    limiter = InMemoryFixedWindowRateLimiter(limit=2, window_seconds=60)

    first = limiter.check("client-a", now=10.0)
    second = limiter.check("client-a", now=11.0)
    third = limiter.check("client-a", now=12.0)

    assert first.allowed is True
    assert first.remaining == 1
    assert second.allowed is True
    assert second.remaining == 0
    assert third.allowed is False
    assert third.retry_after_seconds == 48


def test_fixed_window_rate_limiter_resets_next_window() -> None:
    limiter = InMemoryFixedWindowRateLimiter(limit=1, window_seconds=60)

    assert limiter.check("client-a", now=59.0).allowed is True
    assert limiter.check("client-a", now=59.5).allowed is False
    reset = limiter.check("client-a", now=60.0)

    assert reset.allowed is True
    assert reset.remaining == 0


def test_fixed_window_rate_limiter_is_disabled_when_limit_is_zero() -> None:
    limiter = InMemoryFixedWindowRateLimiter(limit=0, window_seconds=60)

    assert limiter.check("client-a", now=1.0).allowed is True
    assert limiter.check("client-a", now=1.1).allowed is True

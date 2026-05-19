# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import time
from dataclasses import dataclass


PUBLIC_RATE_LIMIT_PER_MINUTE_ENV = "CAI_PUBLIC_RATE_LIMIT_PER_MINUTE"


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int = 0
    remaining: int = 0


class InMemoryFixedWindowRateLimiter:
    def __init__(
        self,
        *,
        limit: int,
        window_seconds: int = 60,
    ) -> None:
        self.limit = max(0, int(limit))
        self.window_seconds = max(1, int(window_seconds))
        self._counts: dict[tuple[str, int], int] = {}

    @property
    def enabled(self) -> bool:
        return self.limit > 0

    @classmethod
    def from_env(cls) -> "InMemoryFixedWindowRateLimiter":
        return cls(limit=_int_env(PUBLIC_RATE_LIMIT_PER_MINUTE_ENV, default=0))

    def check(self, key: str, *, now: float | None = None) -> RateLimitDecision:
        if not self.enabled:
            return RateLimitDecision(allowed=True, remaining=0)

        current_time = float(time.time() if now is None else now)
        window_id = int(current_time // self.window_seconds)
        self._prune_windows_before(window_id)
        bucket = (str(key or "unknown"), window_id)
        current_count = self._counts.get(bucket, 0)
        if current_count >= self.limit:
            retry_after = int(((window_id + 1) * self.window_seconds) - current_time)
            return RateLimitDecision(
                allowed=False,
                retry_after_seconds=max(1, retry_after),
                remaining=0,
            )

        next_count = current_count + 1
        self._counts[bucket] = next_count
        return RateLimitDecision(
            allowed=True,
            remaining=max(0, self.limit - next_count),
        )

    def _prune_windows_before(self, window_id: int) -> None:
        stale = [key for key in self._counts if key[1] < window_id]
        for key in stale:
            self._counts.pop(key, None)


def _int_env(name: str, *, default: int = 0) -> int:
    try:
        return int(str(os.getenv(name) or "").strip() or default)
    except ValueError:
        return default

# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from types import SimpleNamespace

from cai_compute_chain.worker_capability_attestations import (
    worker_capability_fingerprint_from_record,
)


def _record_with_resources(resource_summary: dict[str, int]) -> SimpleNamespace:
    return SimpleNamespace(
        node_id="node-worker",
        worker_enabled=True,
        worker_reward_address="worker-address",
        node_public_key_address="worker-address",
        worker_allowed_model_ids=[],
        model_ids=[],
        resource_summary=resource_summary,
        readiness={},
    )


def test_worker_capability_fingerprint_ignores_volatile_available_resources() -> None:
    first = worker_capability_fingerprint_from_record(
        _record_with_resources(
            {
                "ramBytes": 16_000,
                "ramAvailableBytes": 12_000,
                "swapBytes": 4_000,
                "swapAvailableBytes": 3_000,
            }
        )
    )
    second = worker_capability_fingerprint_from_record(
        _record_with_resources(
            {
                "ramBytes": 16_000,
                "ramAvailableBytes": 8_000,
                "swapBytes": 4_000,
                "swapAvailableBytes": 1_000,
            }
        )
    )

    assert second == first


def test_worker_capability_fingerprint_keeps_stable_resource_capacity() -> None:
    first = worker_capability_fingerprint_from_record(
        _record_with_resources({"ramBytes": 16_000})
    )
    second = worker_capability_fingerprint_from_record(
        _record_with_resources({"ramBytes": 32_000})
    )

    assert second != first

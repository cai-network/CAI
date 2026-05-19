# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import os


def env_positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, "") or default)
    except ValueError:
        return default
    return max(1, value)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def cai_instance_ready_timeout_sec(*, private_network_model: bool) -> int:
    if private_network_model:
        return env_positive_int("CAI_PRIVATE_INSTANCE_READY_TIMEOUT_SECONDS", 600)
    return env_positive_int("CAI_INSTANCE_READY_TIMEOUT_SECONDS", 180)


def job_execution_max_attempts() -> int:
    return env_positive_int("CAI_JOB_EXECUTION_MAX_ATTEMPTS", 3)


def job_execution_total_timeout_sec(default: int | float) -> float:
    return float(
        env_positive_int(
            "CAI_JOB_EXECUTION_TOTAL_TIMEOUT_SECONDS",
            int(max(1, float(default))),
        )
    )


def job_execution_attempt_timeout_sec(default: int | float) -> float:
    return float(
        env_positive_int(
            "CAI_JOB_EXECUTION_ATTEMPT_TIMEOUT_SECONDS",
            int(max(1, float(default))),
        )
    )


def job_execution_first_response_timeout_sec(default: int | float) -> float:
    return float(
        env_positive_int(
            "CAI_JOB_EXECUTION_FIRST_RESPONSE_TIMEOUT_SECONDS",
            int(max(1, float(default))),
        )
    )


def job_execution_retry_backoff_sec() -> float:
    return float(env_positive_int("CAI_JOB_EXECUTION_RETRY_BACKOFF_SECONDS", 2))


def worker_identity_stale_after_seconds() -> int:
    try:
        return max(0, int(os.getenv("CAI_WORKER_IDENTITY_STALE_SECONDS", "300") or "300"))
    except ValueError:
        return 300


def task_level_transport_jobs_enabled() -> bool:
    return env_flag("CAI_ENABLE_TASK_LEVEL_TRANSPORT_JOBS", False)


def task_level_transport_jobs_required() -> bool:
    return env_flag("CAI_REQUIRE_TASK_LEVEL_TRANSPORT_JOBS", False)


def task_level_transport_private_models_allowed() -> bool:
    return env_flag("CAI_ALLOW_TASK_LEVEL_TRANSPORT_PRIVATE_MODELS", False)


def task_level_transport_require_runtime_ready() -> bool:
    return env_flag("CAI_TASK_LEVEL_TRANSPORT_REQUIRE_RUNTIME_READY", True)


def task_level_transport_require_shard_readiness() -> bool:
    return env_flag("CAI_TASK_LEVEL_TRANSPORT_REQUIRE_SHARD_READINESS", True)


def task_level_transport_require_data_plane_route() -> bool:
    return env_flag("CAI_TASK_LEVEL_TRANSPORT_REQUIRE_DATA_PLANE_ROUTE", False)


def task_level_transport_require_proven_data_plane_route() -> bool:
    configured = os.getenv("CAI_TASK_LEVEL_TRANSPORT_REQUIRE_PROVEN_DATA_PLANE_ROUTE")
    if configured is None:
        return task_level_transport_require_data_plane_route()
    return str(configured).strip().lower() in {"1", "true", "yes", "on"}


def task_level_transport_timeout_sec(default: int | float) -> float:
    raw = os.getenv("CAI_TASK_LEVEL_TRANSPORT_TIMEOUT_SEC")
    try:
        return max(0.1, float(raw if raw is not None else default))
    except (TypeError, ValueError):
        return max(0.1, float(default))


def task_level_transport_wait_timeout_sec(default: int | float) -> float:
    raw = os.getenv("CAI_TASK_LEVEL_TRANSPORT_WAIT_TIMEOUT_SEC")
    try:
        return max(0.1, float(raw if raw is not None else default))
    except (TypeError, ValueError):
        return max(0.1, float(default))


def task_level_transport_executor_count() -> int:
    return env_positive_int("CAI_TASK_LEVEL_TRANSPORT_EXECUTOR_COUNT", 1)


def task_level_route_probe_timeout_sec() -> float:
    try:
        return max(
            0.1,
            float(os.getenv("CAI_TASK_LEVEL_ROUTE_PROBE_TIMEOUT_SEC", "1.5") or "1.5"),
        )
    except ValueError:
        return 1.5

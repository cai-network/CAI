# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

from cai.api.text_generation_failures import (
    runner_failure_message_for_model,
    task_failure_message,
    text_generation_failure_detail,
)
from cai.shared.types.worker.runners import RunnerFailed


def test_task_failure_message_returns_matching_command_error() -> None:
    state = SimpleNamespace(
        tasks={
            "other": SimpleNamespace(command_id="other", error_message="ignored"),
            "target": SimpleNamespace(command_id="cmd-1", error_message="runner crashed"),
        }
    )

    assert task_failure_message(state, "cmd-1") == "runner crashed"


def test_runner_failure_message_reports_failed_runner_for_model() -> None:
    model_id = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    state = SimpleNamespace(
        instances={
            "instance-1": SimpleNamespace(
                shard_assignments=SimpleNamespace(
                    model_id=model_id,
                    runner_to_shard={"runner-1": object()},
                )
            )
        },
        runners={"runner-1": RunnerFailed(error_message="llama-server missing")},
    )

    assert runner_failure_message_for_model(state, model_id) == (
        "Runner for model Qwen/Qwen2.5-0.5B-Instruct-GGUF failed: "
        "llama-server missing"
    )


def test_text_generation_failure_detail_prefers_task_then_runner_then_fallback() -> None:
    model_id = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    state = SimpleNamespace(
        tasks={"task-1": SimpleNamespace(command_id="cmd-1", error_message="task failed")},
        instances={},
        runners={},
    )

    assert (
        text_generation_failure_detail(
            state,
            command_id="cmd-1",
            model_id=model_id,
            fallback="no chunks",
        )
        == "task failed"
    )
    assert (
        text_generation_failure_detail(
            None,
            command_id="cmd-missing",
            model_id=model_id,
            fallback="no chunks",
        )
        == "no chunks"
    )

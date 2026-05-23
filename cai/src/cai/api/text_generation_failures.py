# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from cai.shared.models.model_cards import ModelId
from cai.shared.types.worker.runners import RunnerFailed


def text_generation_failure_detail(
    state: object | None,
    *,
    command_id: str,
    model_id: ModelId,
    fallback: str,
) -> str:
    task_failure = task_failure_message(state, command_id)
    if task_failure:
        return task_failure
    runner_failure = runner_failure_message_for_model(state, model_id)
    if runner_failure:
        return runner_failure
    return fallback


def task_failure_message(state: object | None, command_id: str) -> str | None:
    tasks = getattr(state, "tasks", {}) if state is not None else {}
    tasks = tasks or {}
    for task in getattr(tasks, "values", lambda: [])():
        if str(getattr(task, "command_id", "") or "") != str(command_id):
            continue
        error_message = str(getattr(task, "error_message", "") or "").strip()
        if error_message:
            return error_message
    return None


def runner_failure_message_for_model(
    state: object | None,
    model_id: ModelId,
) -> str | None:
    if state is None:
        return None

    instances = getattr(state, "instances", {}) or {}
    runners = getattr(state, "runners", {}) or {}
    runner_lookup = getattr(runners, "get", None)
    if not callable(runner_lookup):
        return None

    for instance in getattr(instances, "values", lambda: [])():
        assignments = getattr(instance, "shard_assignments", None)
        if getattr(assignments, "model_id", None) != model_id:
            continue
        runner_to_shard = getattr(assignments, "runner_to_shard", {}) or {}
        for runner_id in runner_to_shard:
            status = runner_lookup(runner_id)
            if status is None:
                status = runner_lookup(str(runner_id))
            if isinstance(status, RunnerFailed):
                message = str(status.error_message or "runner failed").strip()
                return f"Runner for model {model_id} failed: {message}"
    return None

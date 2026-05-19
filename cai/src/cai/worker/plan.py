# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
# pyright: reportUnusedImport = false

import importlib
from collections.abc import Mapping, Sequence
from functools import lru_cache

from cai.download.download_utils import resolve_existing_model
from cai.shared.models.model_cards import InferenceBackend
from cai.shared.types.chunks import InputImageChunk
from cai.shared.types.common import CommandId, ModelId, NodeId
from cai.shared.types.tasks import (
    CancelTask,
    ConnectToGroup,
    CreateRunner,
    DownloadModel,
    ImageEdits,
    ImageGeneration,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from cai.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadProgress,
)
from cai.shared.types.worker.instances import BoundInstance, Instance, InstanceId
from cai.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerFailed,
    RunnerId,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerStatus,
    RunnerWarmingUp,
)
from cai.utils.keyed_backoff import KeyedBackoff
from cai.worker.runner.runner_supervisor import RunnerSupervisor


@lru_cache(maxsize=1)
def _get_cai_model_distribution_module():
    try:
        return importlib.import_module("cai_compute_chain.model_distribution")
    except Exception:
        return None


def _has_download_completed(
    node_id: NodeId,
    model_id: ModelId,
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> bool:
    return node_id in global_download_status and any(
        isinstance(dp, DownloadCompleted)
        and dp.shard_metadata.model_card.model_id == model_id
        for dp in global_download_status[node_id]
    )


def _llama_cpp_node_chunk_ready(
    *,
    local_node_id: NodeId,
    target_node_id: NodeId,
    instance: Instance,
) -> bool:
    runner_id = instance.shard_assignments.node_to_runner.get(target_node_id)
    if runner_id is None:
        return False

    shard = instance.shard_assignments.runner_to_shard.get(runner_id)
    if shard is None or shard.model_card.inference_backend != InferenceBackend.LlamaCpp:
        return False

    model_distribution = _get_cai_model_distribution_module()
    if model_distribution is None:
        return False

    manifest = model_distribution.select_model_package_manifest_for_model(
        str(shard.model_card.model_id)
    )
    if manifest is None:
        # Non chunk-backed llama.cpp models still need a local GGUF before
        # the runner can load them. We can verify that only for the local node;
        # remote nodes continue to use the existing optimistic behavior because
        # the planner has no direct visibility into their local filesystem.
        if target_node_id != local_node_id:
            return True
        return resolve_existing_model(shard.model_card.model_id) is not None

    assignment = model_distribution.ModelShardAssignment(
        start_layer=shard.start_layer,
        end_layer=shard.end_layer,
        device_rank=shard.device_rank,
        world_size=shard.world_size,
        node_id=str(target_node_id),
        runner_id=str(runner_id),
    )

    if target_node_id == local_node_id:
        chunk_plan = model_distribution.build_assignment_chunk_plan_from_store(
            manifest,
            assignment,
        )
        return bool(chunk_plan.ready)

    imported_peer_inventory = model_distribution.build_chunk_inventory_index(
        manifest.catalog_id,
        manifest.version,
        source_kind=model_distribution.ChunkInventorySourceKind.PEER_CACHE,
    )
    chunk_plan = manifest.build_assignment_chunk_plan(
        assignment,
        present_chunk_ids=imported_peer_inventory.get(str(target_node_id), set()),
    )
    return bool(chunk_plan.ready)


def _node_ready_for_model_load(
    *,
    local_node_id: NodeId,
    target_node_id: NodeId,
    instance: Instance,
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> bool:
    model_id = instance.shard_assignments.model_id
    if _has_download_completed(target_node_id, model_id, global_download_status):
        return True
    return _llama_cpp_node_chunk_ready(
        local_node_id=local_node_id,
        target_node_id=target_node_id,
        instance=instance,
    )


def plan(
    node_id: NodeId,
    # Runners is expected to be FRESH and so should not come from state
    runners: Mapping[RunnerId, RunnerSupervisor],
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
    instances: Mapping[InstanceId, Instance],
    all_runners: Mapping[RunnerId, RunnerStatus],  # all global
    tasks: Mapping[TaskId, Task],
    input_chunk_buffer: Mapping[CommandId, Mapping[int, InputImageChunk]],
    instance_backoff: KeyedBackoff[InstanceId],
    download_backoff: KeyedBackoff[ModelId],
) -> Task | None:
    # Python short circuiting OR logic should evaluate these sequentially.
    return (
        _cancel_tasks(runners, tasks)
        or _kill_runner(runners, all_runners, instances)
        or _create_runner(node_id, runners, all_runners, instances, instance_backoff)
        or _init_distributed_backend(runners, all_runners)
        or _model_needs_download(
            node_id,
            runners,
            global_download_status,
            tasks,
            download_backoff,
        )
        or _load_model(runners, all_runners, global_download_status)
        or _ready_to_warmup(runners, all_runners)
        or _pending_tasks(runners, tasks, all_runners, input_chunk_buffer)
    )


def _kill_runner(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
    instances: Mapping[InstanceId, Instance],
) -> Shutdown | None:
    for runner in runners.values():
        runner_id = runner.bound_instance.bound_runner_id
        if (instance_id := runner.bound_instance.instance.instance_id) not in instances:
            return Shutdown(instance_id=instance_id, runner_id=runner_id)
        if isinstance(runner.status, RunnerFailed):
            return Shutdown(
                instance_id=runner.bound_instance.instance.instance_id,
                runner_id=runner_id,
            )

        for (
            global_runner_id
        ) in runner.bound_instance.instance.shard_assignments.node_to_runner.values():
            if runner_id == global_runner_id:
                continue

            if isinstance(all_runners.get(global_runner_id, None), RunnerFailed):
                return Shutdown(
                    instance_id=instance_id,
                    runner_id=runner_id,
                )


def _create_runner(
    node_id: NodeId,
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
    instances: Mapping[InstanceId, Instance],
    instance_backoff: KeyedBackoff[InstanceId],
) -> CreateRunner | None:
    for instance in instances.values():
        runner_id = instance.shard_assignments.node_to_runner.get(node_id, None)
        if runner_id is None:
            continue

        if runner_id in runners:
            continue

        # don't create runners if any other nodes have runners that have failed - wait for them to fix themselves first.
        instance_has_failed_runner = any(
            isinstance(all_runners.get(remote_runner_id), RunnerFailed)
            for remote_runner_id in instance.shard_assignments.node_to_runner.values()
            if remote_runner_id != runner_id
        )
        we_have_failed_before = isinstance(all_runners.get(runner_id), RunnerFailed)
        if instance_has_failed_runner and not we_have_failed_before:
            continue

        if not instance_backoff.should_proceed(instance.instance_id):
            continue

        return CreateRunner(
            instance_id=instance.instance_id,
            bound_instance=BoundInstance(
                instance=instance, bound_runner_id=runner_id, bound_node_id=node_id
            ),
        )


def _model_needs_download(
    node_id: NodeId,
    runners: Mapping[RunnerId, RunnerSupervisor],
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
    tasks: Mapping[TaskId, Task],
    download_backoff: KeyedBackoff[ModelId],
) -> DownloadModel | None:
    local_downloads = global_download_status.get(node_id, [])
    download_status = {
        dp.shard_metadata.model_card.model_id: dp for dp in local_downloads
    }
    pending_download_models = {
        task.shard_metadata.model_card.model_id
        for task in tasks.values()
        if isinstance(task, DownloadModel)
        if task.task_status in (TaskStatus.Pending, TaskStatus.Running)
    }

    for runner in runners.values():
        model_id = runner.bound_instance.bound_shard.model_card.model_id
        local_chunk_ready = (
            runner.bound_instance.bound_shard.model_card.inference_backend
            == InferenceBackend.LlamaCpp
            and _llama_cpp_node_chunk_ready(
                local_node_id=node_id,
                target_node_id=node_id,
                instance=runner.bound_instance.instance,
            )
        )
        if (
            isinstance(runner.status, RunnerIdle)
            and not local_chunk_ready
            and model_id not in pending_download_models
            and (
                model_id not in download_status
                or not isinstance(
                    download_status[model_id],
                    (DownloadOngoing, DownloadCompleted, DownloadFailed),
                )
            )
            and download_backoff.should_proceed(model_id)
        ):
            # We don't invalidate download_status randomly in case a file gets deleted on disk
            return DownloadModel(
                instance_id=runner.bound_instance.instance.instance_id,
                shard_metadata=runner.bound_instance.bound_shard,
            )


def _init_distributed_backend(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
):
    for runner in runners.values():
        instance = runner.bound_instance.instance
        shard_assignments = instance.shard_assignments

        is_single_node_instance = len(shard_assignments.runner_to_shard) == 1
        if is_single_node_instance:
            continue

        if (
            runner.bound_instance.bound_shard.model_card.inference_backend
            == InferenceBackend.LlamaCpp
        ):
            runner_is_idle = isinstance(runner.status, RunnerIdle)
            all_runners_connectable = all(
                isinstance(
                    all_runners.get(global_runner_id),
                    (RunnerIdle, RunnerConnecting, RunnerConnected),
                )
                for global_runner_id in shard_assignments.runner_to_shard
            )
            if runner_is_idle and all_runners_connectable:
                return ConnectToGroup(instance_id=instance.instance_id)
            continue

        runner_is_idle = isinstance(runner.status, RunnerIdle)
        all_runners_connecting = all(
            isinstance(
                all_runners.get(global_runner_id),
                (RunnerConnecting, RunnerIdle),
            )
            for global_runner_id in shard_assignments.runner_to_shard
        )

        if not (runner_is_idle and all_runners_connecting):
            continue

        runner_id = runner.bound_instance.bound_runner_id

        shard = runner.bound_instance.bound_shard
        device_rank = shard.device_rank
        world_size = shard.world_size

        assert device_rank < world_size
        assert device_rank >= 0

        accepting_ranks = device_rank < world_size - 1

        # Rank = n-1
        connecting_rank_ready = device_rank == world_size - 1 and all(
            isinstance(all_runners.get(global_runner_id, None), RunnerConnecting)
            for global_runner_id in shard_assignments.runner_to_shard
            if global_runner_id != runner_id
        )

        if not (accepting_ranks or connecting_rank_ready):
            continue

        return ConnectToGroup(instance_id=instance.instance_id)

    return None


def _load_model(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> LoadModel | None:
    for runner in runners.values():
        instance = runner.bound_instance.instance
        shard_assignments = instance.shard_assignments

        all_local_downloads_complete = all(
            _node_ready_for_model_load(
                local_node_id=runner.bound_instance.bound_node_id,
                target_node_id=nid,
                instance=instance,
                global_download_status=global_download_status,
            )
            for nid in shard_assignments.node_to_runner
        )
        if not all_local_downloads_complete:
            continue

        is_single_node_instance = len(instance.shard_assignments.runner_to_shard) == 1
        if is_single_node_instance and isinstance(runner.status, RunnerIdle):
            return LoadModel(instance_id=instance.instance_id)

        is_runner_waiting = isinstance(runner.status, RunnerConnected)

        all_ready_for_model = all(
            isinstance(
                all_runners.get(global_runner_id, None),
                (RunnerConnected, RunnerLoading, RunnerLoaded),
            )
            for global_runner_id in shard_assignments.runner_to_shard
        )

        if is_runner_waiting and all_ready_for_model:
            return LoadModel(instance_id=instance.instance_id)

    return None


def _ready_to_warmup(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
) -> StartWarmup | None:
    for runner in runners.values():
        instance = runner.bound_instance.instance
        shard_assignments = instance.shard_assignments
        shard = runner.bound_instance.bound_shard
        if shard.model_card.inference_backend == InferenceBackend.LlamaCpp:
            is_runner_loaded = isinstance(runner.status, RunnerLoaded)
            all_ready_for_warmup = all(
                isinstance(
                    all_runners.get(global_runner_id, None),
                    (RunnerLoaded, RunnerWarmingUp, RunnerReady),
                )
                for global_runner_id in shard_assignments.runner_to_shard
            )
            if is_runner_loaded and all_ready_for_warmup:
                return StartWarmup(instance_id=instance.instance_id)
            continue

        device_rank = shard.device_rank
        runner_id = runner.bound_instance.bound_runner_id
        world_size = shard.world_size

        is_runner_loaded = isinstance(runner.status, RunnerLoaded)

        assert device_rank < world_size
        assert device_rank >= 0

        # Rank != 0
        accepting_ranks_ready = device_rank > 0 and all(
            isinstance(
                all_runners.get(global_runner_id, None),
                (RunnerLoaded, RunnerWarmingUp),
            )
            for global_runner_id in shard_assignments.runner_to_shard
        )

        # Rank = 0
        connecting_rank_ready = device_rank == 0 and all(
            isinstance(all_runners.get(global_runner_id, None), RunnerWarmingUp)
            for global_runner_id in shard_assignments.runner_to_shard
            if global_runner_id != runner_id
        )

        if is_runner_loaded and (accepting_ranks_ready or connecting_rank_ready):
            return StartWarmup(instance_id=instance.instance_id)

    return None


def _pending_tasks(
    runners: Mapping[RunnerId, RunnerSupervisor],
    tasks: Mapping[TaskId, Task],
    all_runners: Mapping[RunnerId, RunnerStatus],
    input_chunk_buffer: Mapping[CommandId, Mapping[int, InputImageChunk]],
) -> Task | None:
    for task in tasks.values():
        # for now, just forward chat completions
        # TODO(ciaran): do this better!
        if not isinstance(task, (TextGeneration, ImageGeneration, ImageEdits)):
            continue
        if task.task_status not in (TaskStatus.Pending, TaskStatus.Running):
            continue

        # For tasks with images, verify all input chunks have been received
        expected_image_chunks = 0
        if isinstance(task, (ImageEdits, TextGeneration)):
            expected_image_chunks = task.task_params.total_input_chunks
        if expected_image_chunks > 0:
            cmd_id = task.command_id
            received = len(input_chunk_buffer.get(cmd_id, {}))
            if received < expected_image_chunks:
                continue  # Wait for all chunks to arrive

        for runner in runners.values():
            if task.instance_id != runner.bound_instance.instance.instance_id:
                continue

            if not _runner_accepts_pending_task(runner, task):
                continue

            # the task status _should_ be set to completed by the LAST runner
            # it is currently set by the first
            # this is definitely a hack
            if task.task_id in runner.completed or task.task_id in runner.in_progress:
                continue

            if isinstance(runner.status, (RunnerReady, RunnerRunning)) and all(
                isinstance(all_runners[global_runner_id], (RunnerReady, RunnerRunning))
                for global_runner_id in runner.bound_instance.instance.shard_assignments.runner_to_shard
            ):
                return task


def _runner_accepts_pending_task(
    runner: RunnerSupervisor,
    task: Task,
) -> bool:
    if not isinstance(task, TextGeneration):
        return True

    shard = runner.bound_instance.bound_shard
    if shard.model_card.inference_backend != InferenceBackend.LlamaCpp:
        return True
    if shard.world_size <= 1:
        return True

    return shard.device_rank == 0


def _cancel_tasks(
    runners: Mapping[RunnerId, RunnerSupervisor],
    tasks: Mapping[TaskId, Task],
) -> Task | None:
    for task in tasks.values():
        if task.task_status != TaskStatus.Cancelled:
            continue
        for runner_id, runner in runners.items():
            if task.instance_id != runner.bound_instance.instance.instance_id:
                continue
            if task.task_id in runner.cancelled:
                continue
            return CancelTask(
                instance_id=task.instance_id,
                cancelled_task_id=task.task_id,
                runner_id=runner_id,
            )


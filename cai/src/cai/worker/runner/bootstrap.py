# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import os

import loguru

try:
    import resource
except ImportError:
    resource = None

from cai.shared.types.events import Event, RunnerStatusUpdated
from cai.shared.types.tasks import Task, TaskId
from cai.shared.types.worker.instances import BoundInstance
from cai.shared.types.worker.runners import RunnerFailed
from cai.utils.channels import ClosedResourceError, MpReceiver, MpSender
from cai.shared.models.model_cards import InferenceBackend

logger: "loguru.Logger" = loguru.logger


def entrypoint(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    cancel_receiver: MpReceiver[TaskId],
    _logger: "loguru.Logger",
) -> None:
    global logger
    logger = _logger

    if resource is not None:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 2048), hard), hard))

    fast_synch_override = os.environ.get("CAI_FAST_SYNCH")
    if fast_synch_override == "false":
        os.environ["MLX_METAL_FAST_SYNCH"] = "0"
    else:
        os.environ["MLX_METAL_FAST_SYNCH"] = "1"

    logger.info(f"Fast synch flag: {os.environ['MLX_METAL_FAST_SYNCH']}")

    # Import main after setting global logger - this lets us just import logger from this module
    try:
        if bound_instance.is_image_model:
            from cai.worker.runner.image_models.runner import Runner as ImageRunner

            runner = ImageRunner(
                bound_instance, event_sender, task_receiver, cancel_receiver
            )
            runner.main()
        elif (
            bound_instance.bound_shard.model_card.inference_backend
            == InferenceBackend.LlamaCpp
        ):
            from cai.worker.runner.llama_cpp.runner import Runner as LlamaCppRunner

            runner = LlamaCppRunner(
                bound_instance, event_sender, task_receiver, cancel_receiver
            )
            runner.main()
        else:
            from cai.worker.runner.llm_inference.runner import Runner
            from cai.worker.engines.mlx.patches import apply_mlx_patches

            apply_mlx_patches()

            runner = Runner(
                bound_instance, event_sender, task_receiver, cancel_receiver
            )
            runner.main()

    except ClosedResourceError:
        logger.warning("Runner communication closed unexpectedly")
    except Exception as e:
        logger.opt(exception=e).warning(
            f"Runner {bound_instance.bound_runner_id} crashed with critical exception {e}"
        )
        event_sender.send(
            RunnerStatusUpdated(
                runner_id=bound_instance.bound_runner_id,
                runner_status=RunnerFailed(error_message=str(e)),
            )
        )
    finally:
        try:
            event_sender.close()
            task_receiver.close()
        finally:
            event_sender.join()
            task_receiver.join()
            logger.info("bye from the runner")


# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from pydantic import Field

from cai.api.types import (
    ImageEditsTaskParams,
    ImageGenerationTaskParams,
)
from cai.shared.models.model_cards import ModelCard, ModelId
from cai.shared.types.chunks import InputImageChunk
from cai.shared.types.common import CommandId, NodeId, SystemId
from cai.shared.types.text_generation import TextGenerationTaskParams
from cai.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from cai.shared.types.worker.shards import Sharding, ShardMetadata
from cai.utils.pydantic_ext import CamelCaseModel, TaggedModel


class BaseCommand(TaggedModel):
    command_id: CommandId = Field(default_factory=CommandId)


class TestCommand(BaseCommand):
    __test__ = False


class TextGeneration(BaseCommand):
    task_params: TextGenerationTaskParams


class ImageGeneration(BaseCommand):
    task_params: ImageGenerationTaskParams


class ImageEdits(BaseCommand):
    task_params: ImageEditsTaskParams


class PlaceInstance(BaseCommand):
    model_card: ModelCard
    sharding: Sharding
    instance_meta: InstanceMeta
    min_nodes: int


class CreateInstance(BaseCommand):
    instance: Instance


class DeleteInstance(BaseCommand):
    instance_id: InstanceId


class TaskCancelled(BaseCommand):
    cancelled_command_id: CommandId


class TaskFinished(BaseCommand):
    finished_command_id: CommandId


class SendInputChunk(BaseCommand):
    """Command to send an input image chunk (converted to event by master)."""

    chunk: InputImageChunk


class RequestEventLog(BaseCommand):
    since_idx: int


class StartDownload(BaseCommand):
    target_node_id: NodeId
    shard_metadata: ShardMetadata


class DeleteDownload(BaseCommand):
    target_node_id: NodeId
    model_id: ModelId


class CancelDownload(BaseCommand):
    target_node_id: NodeId
    model_id: ModelId


class AddCustomModelCard(BaseCommand):
    model_card: ModelCard


class DeleteCustomModelCard(BaseCommand):
    model_id: ModelId


DownloadCommand = StartDownload | DeleteDownload | CancelDownload


Command = (
    TestCommand
    | RequestEventLog
    | TextGeneration
    | ImageGeneration
    | ImageEdits
    | PlaceInstance
    | CreateInstance
    | DeleteInstance
    | TaskCancelled
    | TaskFinished
    | SendInputChunk
    | AddCustomModelCard
    | DeleteCustomModelCard
)


class ForwarderCommand(CamelCaseModel):
    origin: SystemId
    command: Command


class ForwarderDownloadCommand(CamelCaseModel):
    origin: SystemId
    command: DownloadCommand


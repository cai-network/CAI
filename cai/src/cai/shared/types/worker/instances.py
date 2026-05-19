# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from cai.shared.models.model_cards import ModelTask
from cai.shared.types.common import Host, Id, NodeId
from cai.shared.types.worker.runners import RunnerId, ShardAssignments, ShardMetadata
from cai.utils.pydantic_ext import CamelCaseModel, TaggedModel


class InstanceId(Id):
    pass


class InstanceMeta(str, Enum):
    MlxRing = "MlxRing"
    MlxJaccl = "MlxJaccl"
    LlamaCpp = "LlamaCpp"


class BaseInstance(TaggedModel):
    instance_id: InstanceId
    shard_assignments: ShardAssignments

    def shard(self, runner_id: RunnerId) -> ShardMetadata | None:
        return self.shard_assignments.runner_to_shard.get(runner_id, None)


class LlamaCppRelayRoute(CamelCaseModel):
    source_node_id: NodeId
    transit_node_id: NodeId
    sink_node_id: NodeId
    relay_api_host: str
    relay_api_port: int
    target_host: str
    target_port: int
    source_segment_type: Literal["direct", "overlay"]
    sink_segment_type: Literal["direct", "overlay"]


class MlxRingInstance(BaseInstance):
    hosts_by_node: dict[NodeId, list[Host]]
    ephemeral_port: int
    relay_routes_by_node: dict[NodeId, list[LlamaCppRelayRoute]] = Field(
        default_factory=dict
    )
    cai_api_urls_by_node: dict[NodeId, list[str]] = Field(default_factory=dict)


class MlxJacclInstance(BaseInstance):
    jaccl_devices: list[list[str | None]]
    jaccl_coordinators: dict[NodeId, str]


# TODO: Single node instance
Instance = MlxRingInstance | MlxJacclInstance


class BoundInstance(CamelCaseModel):
    instance: Instance
    bound_runner_id: RunnerId
    bound_node_id: NodeId

    @property
    def bound_shard(self) -> ShardMetadata:
        shard = self.instance.shard(self.bound_runner_id)
        assert shard is not None
        return shard

    @property
    def is_image_model(self) -> bool:
        return (
            ModelTask.TextToImage in self.bound_shard.model_card.tasks
            or ModelTask.ImageToImage in self.bound_shard.model_card.tasks
        )

    @model_validator(mode="after")
    def validate_shard_exists(self) -> "BoundInstance":
        assert (
            self.bound_runner_id in self.instance.shard_assignments.runner_to_shard
        ), (
            "Bound Instance must be constructed with a runner_id that is in the instances assigned shards"
        )
        return self


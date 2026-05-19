# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from enum import Enum

from cai.routing.cai_owned_transport_message import CaiOwnedTransportOverlayMessage
from cai.routing.connection_message import ConnectionMessage
from cai.shared.election import ElectionMessage
from cai.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from cai.shared.types.events import (
    GlobalForwarderEvent,
    LocalForwarderEvent,
)
from cai.utils.pydantic_ext import CamelCaseModel


class PublishPolicy(str, Enum):
    Never = "Never"
    """Never publish to the network - this is a local message"""
    Minimal = "Minimal"
    """Only publish when there is no local receiver for this type of message"""
    Always = "Always"
    """Always publish to the network"""


@dataclass  # (frozen=True)
class TypedTopic[T: CamelCaseModel]:
    topic: str
    publish_policy: PublishPolicy

    model_type: type[
        T
    ]  # This can be worked around with evil type hacking, see https://stackoverflow.com/a/71720366 - I don't think it's necessary here.

    @staticmethod
    def serialize(t: T) -> bytes:
        return t.model_dump_json().encode("utf-8")

    def deserialize(self, b: bytes) -> T:
        return self.model_type.model_validate_json(b.decode("utf-8"))


GLOBAL_EVENTS = TypedTopic("global_events", PublishPolicy.Always, GlobalForwarderEvent)
LOCAL_EVENTS = TypedTopic("local_events", PublishPolicy.Always, LocalForwarderEvent)
COMMANDS = TypedTopic("commands", PublishPolicy.Always, ForwarderCommand)
ELECTION_MESSAGES = TypedTopic(
    "election_messages", PublishPolicy.Always, ElectionMessage
)
CONNECTION_MESSAGES = TypedTopic(
    "connection_messages", PublishPolicy.Never, ConnectionMessage
)
DOWNLOAD_COMMANDS = TypedTopic(
    "download_commands", PublishPolicy.Always, ForwarderDownloadCommand
)
CAI_OWNED_TRANSPORT_MESSAGES = TypedTopic(
    "cai_owned_transport_messages",
    PublishPolicy.Always,
    CaiOwnedTransportOverlayMessage,
)


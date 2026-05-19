# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from cai.shared.types.common import NodeId
from cai.utils.pydantic_ext import CamelCaseModel

from .native_bindings import PyFromSwarm

"""Serialisable types for Connection Updates/Messages"""


class ConnectionMessage(CamelCaseModel):
    node_id: NodeId
    connected: bool

    @classmethod
    def from_update(cls, update: PyFromSwarm.Connection) -> "ConnectionMessage":
        return cls(node_id=NodeId(update.peer_id), connected=update.connected)

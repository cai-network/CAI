# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Iterator
from dataclasses import dataclass

from cai.shared.types.common import NodeId
from cai.shared.types.multiaddr import Multiaddr
from cai.utils.pydantic_ext import FrozenModel


@dataclass(frozen=True)
class Cycle:
    node_ids: list[NodeId]

    def __len__(self) -> int:
        return self.node_ids.__len__()

    def __iter__(self) -> Iterator[NodeId]:
        return self.node_ids.__iter__()


class RDMAConnection(FrozenModel):
    source_rdma_iface: str
    sink_rdma_iface: str


class SocketConnection(FrozenModel):
    sink_multiaddr: Multiaddr

    def __hash__(self):
        return hash((self.sink_multiaddr.ip_address, self.sink_multiaddr.port))


class Connection(FrozenModel):
    source: NodeId
    sink: NodeId
    edge: RDMAConnection | SocketConnection


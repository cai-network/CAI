# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Literal

from cai.utils.pydantic_ext import CamelCaseModel


CaiOwnedTransportOverlayKind = Literal[
    "session_offer",
    "batch_envelope",
    "shard_receipt",
    "completion_notice",
]


class CaiOwnedTransportOverlayMessage(CamelCaseModel):
    message_id: str
    kind: CaiOwnedTransportOverlayKind
    source_node_id: str
    target_node_id: str
    session_id: str
    payload: dict[str, Any]
    created_at: str

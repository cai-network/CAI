# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
from dataclasses import dataclass, field

from fastapi import WebSocket


RELAY_TARGET_CONNECT_TIMEOUT_SECONDS = float(
    os.getenv("CAI_RELAY_TARGET_CONNECT_TIMEOUT_SECONDS", "1") or "1"
)
REVERSE_RELAY_WAIT_TIMEOUT_SECONDS = float(
    os.getenv("CAI_REVERSE_RELAY_WAIT_TIMEOUT_SECONDS", "4") or "4"
)
RELAY_STREAM_CHUNK_SIZE = max(
    int(os.getenv("CAI_RELAY_STREAM_CHUNK_SIZE", "16384") or "16384"),
    1024,
)
RELAY_EOF_MESSAGE = "__cai_relay_eof__"
RELAY_TARGET_CONNECTED_MESSAGE = "__cai_relay_target_connected__"
REVERSE_RELAY_TARGET_READY_TIMEOUT_SECONDS = float(
    os.getenv("CAI_REVERSE_RELAY_TARGET_READY_TIMEOUT_SECONDS", "4") or "4"
)
LLAMA_CPP_RPC_CMD_HELLO = 14
LLAMA_CPP_RPC_CONN_CAPS_SIZE = 24
LLAMA_CPP_RPC_HELLO_RESPONSE_SIZE = 4 + LLAMA_CPP_RPC_CONN_CAPS_SIZE


@dataclass
class ReverseRelaySession:
    websocket: WebSocket
    done: asyncio.Event = field(default_factory=asyncio.Event)

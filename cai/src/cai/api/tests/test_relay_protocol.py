# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from cai.api.relay_protocol import (
    LLAMA_CPP_RPC_CMD_HELLO,
    LLAMA_CPP_RPC_CONN_CAPS_SIZE,
    LLAMA_CPP_RPC_HELLO_RESPONSE_SIZE,
    RELAY_EOF_MESSAGE,
    RELAY_STREAM_CHUNK_SIZE,
    RELAY_TARGET_CONNECTED_MESSAGE,
    ReverseRelaySession,
)


def test_relay_protocol_constants_match_wire_contract() -> None:
    assert RELAY_EOF_MESSAGE == "__cai_relay_eof__"
    assert RELAY_TARGET_CONNECTED_MESSAGE == "__cai_relay_target_connected__"
    assert RELAY_STREAM_CHUNK_SIZE >= 1024
    assert LLAMA_CPP_RPC_CMD_HELLO == 14
    assert LLAMA_CPP_RPC_CONN_CAPS_SIZE == 24
    assert LLAMA_CPP_RPC_HELLO_RESPONSE_SIZE == 28


def test_reverse_relay_session_starts_unset() -> None:
    session = ReverseRelaySession(websocket=object())

    assert session.websocket is not None
    assert session.done.is_set() is False

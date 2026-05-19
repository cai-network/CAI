# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import asyncio
import struct

from unittest.mock import patch

from cai.api.main import (
    API,
    _RELAY_TARGET_CONNECTED_MESSAGE,
    _ReverseRelaySession,
)


class FakeWebSocket:
    def __init__(self, messages: list[dict[str, object]]) -> None:
        self.messages = list(messages)
        self.sent_text: list[str] = []
        self.sent_bytes: list[bytes] = []
        self.accepted = False
        self.closed = False

    async def accept(self) -> None:
        self.accepted = True

    async def receive(self) -> dict[str, object]:
        if self.messages:
            return self.messages.pop(0)
        return {"type": "websocket.disconnect"}

    async def send_text(self, payload: str) -> None:
        self.sent_text.append(payload)

    async def send_bytes(self, payload: bytes) -> None:
        self.sent_bytes.append(payload)

    async def close(self, *_args, **_kwargs) -> None:
        self.closed = True


class FakeStream:
    def __init__(self) -> None:
        response = bytes([3, 6, 0, 0]) + bytes(24)
        self.response = struct.pack("<Q", len(response)) + response
        self.sent = bytearray()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args) -> None:
        return None

    async def send(self, payload: bytes) -> None:
        self.sent.extend(payload)

    async def receive(self, size: int) -> bytes:
        chunk = self.response[:size]
        self.response = self.response[size:]
        return chunk


def test_reverse_relay_source_connects_after_target_ready() -> None:
    async def scenario() -> tuple[FakeWebSocket, FakeWebSocket, bool]:
        api = object.__new__(API)
        source = FakeWebSocket([{"type": "websocket.disconnect"}])
        reverse = FakeWebSocket(
            [
                {"type": "websocket.disconnect"},
            ]
        )
        session = _ReverseRelaySession(websocket=reverse)

        await api._bridge_relay_websockets(source, session)

        return source, reverse, session.done.is_set()

    source, reverse, done = asyncio.run(scenario())

    assert reverse.sent_text[0] == "connected"
    assert source.sent_text == ["connected"]
    assert done is True


def test_reverse_relay_endpoint_queues_only_after_target_ready() -> None:
    async def scenario() -> tuple[FakeWebSocket, int, bool]:
        api = object.__new__(API)
        api.node_id = "node-relay"
        api._reverse_relay_queues = {}
        api._reverse_relay_queues_lock = asyncio.Lock()
        api._local_node_relay_enabled = lambda: True
        api._relay_target_allowed = lambda *_args: True

        reverse = FakeWebSocket(
            [
                {
                    "type": "websocket.receive",
                    "text": _RELAY_TARGET_CONNECTED_MESSAGE,
                },
            ]
        )
        task = asyncio.create_task(
            api.cai_reverse_relay_rpc_websocket(
                reverse,
                target_host="10.0.0.2",
                target_port=52435,
                sink_node_id="node-sink",
            )
        )
        key = api._reverse_relay_key("node-sink", "10.0.0.2", 52435)
        queue = await api._reverse_relay_queue(key)
        for _ in range(20):
            if queue.qsize():
                break
            await asyncio.sleep(0.01)
        queued = queue.qsize()
        session = await queue.get()
        session.done.set()
        await task
        return reverse, queued, session.done.is_set()

    reverse, queued, done = asyncio.run(scenario())

    assert reverse.accepted is True
    assert reverse.sent_text == ["registered"]
    assert queued == 1
    assert done is True


def test_relay_probe_can_verify_llama_cpp_rpc_protocol() -> None:
    async def scenario() -> tuple[dict[str, object], FakeStream, list[dict[str, object]]]:
        api = object.__new__(API)
        api.node_id = "node-relay"
        api._local_node_relay_enabled = lambda: True
        api._relay_target_allowed = lambda *_args: True
        api._reverse_relay_queue_size = lambda *_args: asyncio.sleep(0, result=0)
        route_health_calls: list[dict[str, object]] = []
        api._record_relay_probe_route_health = lambda **kwargs: route_health_calls.append(
            kwargs
        )
        stream = FakeStream()

        async def _connect_tcp(host: str, port: int):
            assert host == "10.0.0.2"
            assert port == 52435
            return stream

        with patch("cai.api.main.anyio.connect_tcp", side_effect=_connect_tcp):
            payload = await api.cai_relay_rpc_probe(
                target_host="10.0.0.2",
                target_port=52435,
                sink_node_id="node-sink",
                source_node_id="node-source",
                transit_node_id="node-relay",
                protocol="llama_cpp_rpc",
            )
        return payload, stream, route_health_calls

    payload, stream, route_health_calls = asyncio.run(scenario())

    assert payload["ready"] is True
    assert payload["protocolReady"] is True
    assert payload["protocolVersion"] == "3.6.0"
    assert stream.sent[0] == 14
    assert struct.unpack("<Q", stream.sent[1:9])[0] == 24
    assert route_health_calls[0]["ready"] is True
    assert route_health_calls[0]["mode"] == "direct"

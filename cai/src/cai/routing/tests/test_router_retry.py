# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import pytest

from cai.routing import router as router_module
from cai.routing import topics
from cai.routing.router import Router, TopicRouter
from cai.utils.channels import channel


class _QueueFullError(Exception):
    pass


class _NoPeersSubscribedError(Exception):
    pass


class _MessageTooLargeError(Exception):
    pass


class _FakeNetworkingHandle:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    async def gossipsub_publish(self, topic: str, data: bytes) -> None:
        self.calls += 1
        if not self._outcomes:
            return
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome


@pytest.mark.anyio
async def test_publish_network_message_retries_until_queue_drains(monkeypatch):
    networking = _FakeNetworkingHandle(
        [_QueueFullError(), _QueueFullError(), None]
    )
    router = object.__new__(Router)
    router._net = networking

    monkeypatch.setattr(router_module, "AllQueuesFullError", _QueueFullError)
    monkeypatch.setattr(router_module, "GOSSIPSUB_QUEUE_RETRY_BASE_SECONDS", 0.0)
    monkeypatch.setattr(router_module, "GOSSIPSUB_QUEUE_RETRY_MAX_SECONDS", 0.0)

    await router._publish_network_message("commands", b"payload")

    assert networking.calls == 3


@pytest.mark.anyio
async def test_publish_network_message_returns_when_no_peers(monkeypatch):
    networking = _FakeNetworkingHandle([_NoPeersSubscribedError()])
    router = object.__new__(Router)
    router._net = networking

    monkeypatch.setattr(
        router_module, "NoPeersSubscribedToTopicError", _NoPeersSubscribedError
    )

    await router._publish_network_message("commands", b"payload")

    assert networking.calls == 1


@pytest.mark.anyio
async def test_publish_network_message_drops_oversize_messages_without_retry(
    monkeypatch,
):
    networking = _FakeNetworkingHandle([_MessageTooLargeError()])
    router = object.__new__(Router)
    router._net = networking

    monkeypatch.setattr(router_module, "MessageTooLargeError", _MessageTooLargeError)

    await router._publish_network_message("commands", b"payload")

    assert networking.calls == 1


@pytest.mark.anyio
async def test_incoming_malformed_network_message_is_dropped():
    send, recv = channel[tuple[str, bytes]]()
    try:
        router = object.__new__(Router)
        router.topic_routers = {
            topics.GLOBAL_EVENTS.topic: TopicRouter(topics.GLOBAL_EVENTS, send)
        }

        accepted = await router._publish_incoming_network_message(
            origin="peer-a",
            topic=topics.GLOBAL_EVENTS.topic,
            data=b'{"event":{"NodeGatheredInfo":{"unexpected":true}}}',
        )

        assert accepted is False
    finally:
        send.close()
        recv.close()

# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from copy import copy
from itertools import count
from math import inf
from os import PathLike
from pathlib import Path
from typing import cast

from anyio import (
    BrokenResourceError,
    ClosedResourceError,
    sleep,
    move_on_after,
    sleep_forever,
)
from cai.routing.native_bindings import (
    AllQueuesFullError,
    Keypair,
    MessageTooLargeError,
    NetworkingHandle,
    NoPeersSubscribedToTopicError,
    PyFromSwarm,
)
from filelock import FileLock
from loguru import logger

from cai.shared.constants import CAI_NODE_ID_KEYPAIR
from cai.utils.channels import Receiver, Sender, channel
from cai.utils.pydantic_ext import CamelCaseModel
from cai.utils.task_group import TaskGroup

from .connection_message import ConnectionMessage
from .topics import CONNECTION_MESSAGES, PublishPolicy, TypedTopic

GOSSIPSUB_QUEUE_RETRY_BASE_SECONDS = 0.05
GOSSIPSUB_QUEUE_RETRY_MAX_SECONDS = 1.0
GOSSIPSUB_QUEUE_RETRY_LOG_EVERY = 10


# A significant current limitation of the TopicRouter is that it is not capable
# of preventing feedback, as it does not ask for a system id so cannot tell
# which message is coming/going to which system.
# This is currently only relevant for Election
class TopicRouter[T: CamelCaseModel]:
    def __init__(
        self,
        topic: TypedTopic[T],
        networking_sender: Sender[tuple[str, bytes]],
        max_buffer_size: float = inf,
    ):
        self.topic: TypedTopic[T] = topic
        self.senders: set[Sender[T]] = set()
        send, recv = channel[T]()
        self.receiver: Receiver[T] = recv
        self._sender: Sender[T] = send
        self.networking_sender: Sender[tuple[str, bytes]] = networking_sender

    async def run(self):
        logger.debug(f"Topic Router {self.topic} ready to send")
        with self.receiver as items:
            async for item in items:
                # Check if we should send to network
                if (
                    len(self.senders) == 0
                    and self.topic.publish_policy is PublishPolicy.Minimal
                ):
                    await self._send_out(item)
                    continue
                if self.topic.publish_policy is PublishPolicy.Always:
                    await self._send_out(item)
                # Then publish to all senders
                await self.publish(item)

    async def shutdown(self):
        logger.debug(f"Shutting down Topic Router {self.topic}")
        # Close all the things!
        for sender in self.senders:
            sender.close()
        self._sender.close()
        self.receiver.close()

    async def publish(self, item: T):
        """
        Publish item T on this topic to all senders.
        NB: this sends to ALL receivers, potentially including receivers held by the object doing the sending.
        You should handle your own output if you hold a sender + receiver pair.
        """
        to_clear: set[Sender[T]] = set()
        for sender in copy(self.senders):
            try:
                await sender.send(item)
            except (ClosedResourceError, BrokenResourceError):
                to_clear.add(sender)
        self.senders -= to_clear

    async def publish_bytes(self, data: bytes):
        await self.publish(self.topic.deserialize(data))

    def new_sender(self) -> Sender[T]:
        return self._sender.clone()

    async def _send_out(self, item: T):
        logger.trace(f"TopicRouter {self.topic.topic} sending {item}")
        await self.networking_sender.send(
            (str(self.topic.topic), self.topic.serialize(item))
        )


class Router:
    @classmethod
    def create(
        cls,
        identity: Keypair,
        bootstrap_peers: Sequence[str] = (),
        listen_port: int = 0,
    ) -> "Router":
        return cls(
            handle=NetworkingHandle(identity, list(bootstrap_peers), listen_port)
        )

    def __init__(self, handle: NetworkingHandle):
        self.topic_routers: dict[str, TopicRouter[CamelCaseModel]] = {}
        send, recv = channel[tuple[str, bytes]]()
        self.networking_receiver: Receiver[tuple[str, bytes]] = recv
        self._net: NetworkingHandle = handle
        self._tmp_networking_sender: Sender[tuple[str, bytes]] | None = send
        self._id_count = count()
        self._tg: TaskGroup = TaskGroup()

    async def register_topic[T: CamelCaseModel](self, topic: TypedTopic[T]):
        send = self._tmp_networking_sender
        if send:
            self._tmp_networking_sender = None
        else:
            send = self.networking_receiver.clone_sender()
        router = TopicRouter[T](topic, send)
        self.topic_routers[topic.topic] = cast(TopicRouter[CamelCaseModel], router)
        if self._tg.is_running():
            await self._networking_subscribe(topic.topic)

    def sender[T: CamelCaseModel](self, topic: TypedTopic[T]) -> Sender[T]:
        router = self.topic_routers.get(topic.topic, None)
        # There's gotta be a way to do this without THIS many asserts
        assert router is not None
        assert router.topic == topic
        sender = cast(TopicRouter[T], router).new_sender()
        return sender

    def receiver[T: CamelCaseModel](self, topic: TypedTopic[T]) -> Receiver[T]:
        router = self.topic_routers.get(topic.topic, None)
        # There's gotta be a way to do this without THIS many asserts

        assert router is not None
        assert router.topic == topic
        assert router.topic.model_type == topic.model_type

        send, recv = channel[T]()
        router.senders.add(cast(Sender[CamelCaseModel], send))

        return recv

    async def run(self):
        logger.debug("Starting Router")
        try:
            async with self._tg as tg:
                for topic in self.topic_routers:
                    router = self.topic_routers[topic]
                    tg.start_soon(router.run)
                tg.start_soon(self._networking_recv)
                tg.start_soon(self._networking_publish)
                # subscribe to pending topics
                for topic in self.topic_routers:
                    await self._networking_subscribe(topic)
                # Router only shuts down if you cancel it.
                await sleep_forever()
        finally:
            with move_on_after(1, shield=True):
                for topic in self.topic_routers:
                    await self._networking_unsubscribe(str(topic))

    async def shutdown(self):
        logger.debug("Shutting down Router")
        self._tg.cancel_tasks()

    async def _networking_subscribe(self, topic: str):
        await self._net.gossipsub_subscribe(topic)
        logger.info(f"Subscribed to {topic}")

    async def _networking_unsubscribe(self, topic: str):
        await self._net.gossipsub_unsubscribe(topic)
        logger.info(f"Unsubscribed from {topic}")

    async def _networking_recv(self):
        try:
            while True:
                from_swarm = await self._net.recv()
                logger.debug(from_swarm)
                match from_swarm:
                    case PyFromSwarm.Message(origin, topic, data):
                        logger.trace(
                            f"Received message on {topic} from {origin} with payload {data}"
                        )
                        if topic not in self.topic_routers:
                            logger.warning(
                                f"Received message on unknown or inactive topic {topic}"
                            )
                            continue
                        await self._publish_incoming_network_message(
                            origin=origin,
                            topic=topic,
                            data=data,
                        )
                    case PyFromSwarm.Connection():
                        message = ConnectionMessage.from_update(from_swarm)
                        logger.trace(
                            f"Received message on connection_messages with payload {message}"
                        )
                        if CONNECTION_MESSAGES.topic in self.topic_routers:
                            router = self.topic_routers[CONNECTION_MESSAGES.topic]
                            assert router.topic.model_type == ConnectionMessage
                            router = cast(TopicRouter[ConnectionMessage], router)
                            await router.publish(message)
                    case _:
                        logger.critical(
                            "failed to exhaustively check FromSwarm messages - logic error"
                        )
        except Exception as exception:
            logger.opt(exception=exception).error(
                "Gossipsub receive loop terminated unexpectedly"
            )
            raise

    async def _publish_incoming_network_message(
        self, *, origin: object, topic: str, data: bytes
    ) -> bool:
        router = self.topic_routers[topic]
        try:
            await router.publish_bytes(data)
        except Exception as exception:
            logger.opt(exception=exception).warning(
                "Dropping malformed network message on topic {} from {} ({} bytes)",
                topic,
                origin,
                len(data),
            )
            return False
        return True

    async def _networking_publish(self):
        with self.networking_receiver as networked_items:
            async for topic, data in networked_items:
                await self._publish_network_message(topic, data)

    async def _publish_network_message(self, topic: str, data: bytes) -> None:
        logger.trace(f"Sending message on {topic} with payload {data}")
        if len(data) > 1024 * 1024:
            logger.warning(
                "Sending overlarge payload, network performance may be temporarily degraded"
            )

        attempt = 0
        while True:
            try:
                await self._net.gossipsub_publish(topic, data)
                return
            except NoPeersSubscribedToTopicError:
                return
            except AllQueuesFullError:
                attempt += 1
                delay = min(
                    GOSSIPSUB_QUEUE_RETRY_MAX_SECONDS,
                    GOSSIPSUB_QUEUE_RETRY_BASE_SECONDS
                    * (2 ** min(attempt - 1, 6)),
                )
                if attempt == 1 or attempt % GOSSIPSUB_QUEUE_RETRY_LOG_EVERY == 0:
                    logger.warning(
                        "All peer queues full for topic {}. Retrying publish attempt {} in {:.2f}s",
                        topic,
                        attempt,
                        delay,
                    )
                await sleep(delay)
            except MessageTooLargeError:
                logger.warning(
                    f"Message too large for gossipsub on {topic} ({len(data)} bytes), dropping"
                )
                return


def get_node_id_keypair(
    path: str | bytes | PathLike[str] | PathLike[bytes] = CAI_NODE_ID_KEYPAIR,
) -> Keypair:
    """
    Obtains the :class:`Keypair` associated with this node-ID.
    Obtain the :class:`PeerId` by from it.
    """
    def lock_path(path: str | bytes | PathLike[str] | PathLike[bytes]) -> Path:
        return Path(str(path) + ".lock")

    # operate with cross-process lock to avoid race conditions
    with FileLock(lock_path(path)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a+b") as f:  # opens in append-mode => starts at EOF
            # if non-zero EOF, then file exists => use to get node-ID
            if f.tell() != 0:
                f.seek(0)  # go to start & read protobuf-encoded bytes
                protobuf_encoded = f.read()

                try:  # if decoded successfully, save & return
                    return Keypair.from_bytes(protobuf_encoded)
                except ValueError as e:  # on runtime error, assume corrupt file
                    logger.warning(f"Encountered error when trying to get keypair: {e}")

        # if no valid credentials, create new ones and persist
        with open(path, "w+b") as f:
            keypair = Keypair.generate()
            f.write(keypair.to_bytes())
            return keypair


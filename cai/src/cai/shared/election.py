# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Self

import anyio
from anyio import (
    CancelScope,
    Event,
    get_cancelled_exc_class,
)
from loguru import logger

from cai.routing.connection_message import ConnectionMessage
from cai.shared.types.commands import ForwarderCommand
from cai.shared.types.common import NodeId, SessionId
from cai.utils.channels import Receiver, Sender
from cai.utils.pydantic_ext import CamelCaseModel
from cai.utils.task_group import TaskGroup

DEFAULT_ELECTION_TIMEOUT = 3.0
DEFAULT_CONNECTION_DISCONNECT_GRACE = float(
    os.getenv("CAI_CONNECTION_DISCONNECT_GRACE_SECONDS")
    or "30.0"
)
DEFAULT_WORKER_MASTER_SENIORITY_BONUS = int(
    os.getenv("CAI_WORKER_MASTER_SENIORITY_BONUS")
    or "1000"
)
DEFAULT_WORKER_MASTER_RECLAIM_ENABLED = (
    (
        os.getenv("CAI_WORKER_CAN_RECLAIM_MASTER")
        or "0"
    ).strip().lower()
    in {"1", "true", "yes", "on"}
)
DEFAULT_WORKER_MODE_POLL_SECONDS = float(
    os.getenv("CAI_WORKER_MODE_POLL_SECONDS")
    or "1.0"
)


def _load_local_worker_mode_enabled() -> bool:
    try:
        from cai_compute_chain.model import WalletPolicy
        from cai_compute_chain.node_config import load_or_create_node_config

        config = load_or_create_node_config(WalletPolicy())
    except Exception:
        return False
    return bool(getattr(config, "worker_enabled", False))


class ElectionMessage(CamelCaseModel):
    clock: int
    seniority: int
    proposed_session: SessionId
    commands_seen: int

    # Could eventually include a list of neighbour nodes for centrality
    def __lt__(self, other: Self) -> bool:
        if self.clock != other.clock:
            return self.clock < other.clock
        if self.seniority != other.seniority:
            return self.seniority < other.seniority
        elif self.commands_seen != other.commands_seen:
            return self.commands_seen < other.commands_seen
        else:
            return (
                self.proposed_session.master_node_id
                < other.proposed_session.master_node_id
            )


class ElectionResult(CamelCaseModel):
    session_id: SessionId
    won_clock: int
    is_new_master: bool


class Election:
    def __init__(
        self,
        node_id: NodeId,
        *,
        election_message_receiver: Receiver[ElectionMessage],
        election_message_sender: Sender[ElectionMessage],
        election_result_sender: Sender[ElectionResult],
        connection_message_receiver: Receiver[ConnectionMessage],
        command_receiver: Receiver[ForwarderCommand],
        is_candidate: bool = True,
        seniority: int = 0,
        initial_session_id: SessionId | None = None,
    ):
        # If we aren't a candidate, simply don't increment seniority.
        # For reference: This node can be elected master if all nodes are not master candidates
        # Any master candidate will automatically win out over this node.
        self.seniority = seniority if is_candidate else -1
        self.clock = 0
        self.node_id = node_id
        self.commands_seen = 0
        self._connected_peers: set[NodeId] = set()
        self._pending_disconnects: dict[NodeId, CancelScope] = {}
        self._worker_execution_enabled = _load_local_worker_mode_enabled()
        # Every node spawns as master
        self.current_session: SessionId = initial_session_id or SessionId(
            master_node_id=node_id,
            election_clock=0,
        )

        # Senders/Receivers
        self._em_sender = election_message_sender
        self._em_receiver = election_message_receiver
        self._er_sender = election_result_sender
        self._cm_receiver = connection_message_receiver
        self._co_receiver = command_receiver

        # Campaign state
        self._candidates: list[ElectionMessage] = []
        self._campaign_cancel_scope: CancelScope | None = None
        self._campaign_done: Event | None = None
        self._tg = TaskGroup()

    async def run(self):
        logger.info("Starting Election")
        try:
            async with self._tg as tg:
                tg.start_soon(self._election_receiver)
                tg.start_soon(self._connection_receiver)
                tg.start_soon(self._command_counter)
                tg.start_soon(self._watch_local_worker_mode)

                # And start an election immediately, that instantly resolves
                candidates: list[ElectionMessage] = []
                logger.debug("Starting initial campaign")
                self._candidates = candidates
                await self._campaign(candidates, campaign_timeout=0.0)
                logger.debug("Initial campaign finished")
        finally:
            # Cancel and wait for the last election to end
            if self._campaign_cancel_scope is not None:
                logger.debug("Cancelling campaign")
                self._campaign_cancel_scope.cancel()
            if self._campaign_done is not None:
                logger.debug("Waiting for campaign to finish")
                await self._campaign_done.wait()
            logger.debug("Campaign cancelled and finished")
            logger.info("Election shutdown")

    async def elect(self, em: ElectionMessage) -> None:
        logger.debug(f"Electing: {em}")
        is_new_master = em.proposed_session != self.current_session
        self.current_session = em.proposed_session
        logger.debug(f"Current session: {self.current_session}")
        await self._er_sender.send(
            ElectionResult(
                won_clock=em.clock,
                session_id=em.proposed_session,
                is_new_master=is_new_master,
            )
        )

    async def shutdown(self) -> None:
        self._tg.cancel_tasks()

    async def _election_receiver(self) -> None:
        with self._em_receiver as election_messages:
            async for message in election_messages:
                logger.debug(f"Election message received: {message}")
                # If a new round is starting, we participate
                if message.clock > self.clock:
                    self.clock = message.clock
                    logger.debug(f"New clock: {self.clock}")
                    logger.debug("Starting new campaign")
                    candidates: list[ElectionMessage] = [message]
                    logger.debug(f"Candidates: {candidates}")
                    logger.debug(f"Current candidates: {self._candidates}")
                    self._candidates = candidates
                    logger.debug(f"New candidates: {self._candidates}")
                    logger.debug("Starting new campaign")
                    self._tg.start_soon(
                        self._campaign, candidates, DEFAULT_ELECTION_TIMEOUT
                    )
                    logger.debug("Campaign started")
                    continue
                # Dismiss old messages
                if message.clock < self.clock:
                    logger.debug(f"Dropping old message: {message}")
                    continue
                if not self._campaign_active():
                    if message.proposed_session == self.current_session:
                        logger.debug(
                            "Ignoring settled same-clock message that matches current session: {}",
                            message,
                        )
                        continue
                    logger.info(
                        "Observed late peer election state after round {} settled; starting follow-up campaign",
                        self.clock,
                    )
                    await self._start_new_campaign(
                        "Late peer election state observed",
                        initial_candidates=[message],
                    )
                    continue
                logger.debug(f"Election added candidate {message}")
                # Now we are processing this rounds messages - including the message that triggered this round.
                self._candidates.append(message)

    async def _connection_receiver(self) -> None:
        with self._cm_receiver as connection_messages:
            async for first in connection_messages:
                # Delay after connection message for time to symmetrically setup
                await anyio.sleep(0.2)
                rest = connection_messages.collect()
                batch = [first, *rest]

                logger.debug(
                    f"Connection messages received: {first} followed by {rest}"
                )
                state_changed = False
                for message in batch:
                    if message.connected:
                        reconnected_during_grace = False
                        if pending := self._pending_disconnects.pop(
                            message.node_id, None
                        ):
                            pending.cancel()
                            reconnected_during_grace = True
                        if message.node_id in self._connected_peers:
                            if reconnected_during_grace:
                                logger.debug(
                                    "Peer {} reconnected during disconnect grace; replaying election status",
                                    message.node_id,
                                )
                                await self._em_sender.send(self._election_status())
                            continue
                        self._connected_peers.add(message.node_id)
                        state_changed = True
                        continue

                    if message.node_id not in self._connected_peers:
                        continue

                    if message.node_id in self._pending_disconnects:
                        continue

                    scope = CancelScope()
                    self._pending_disconnects[message.node_id] = scope
                    logger.debug(
                        f"Scheduling disconnect confirmation for {message.node_id}"
                    )
                    self._tg.start_soon(
                        self._confirm_disconnect, message.node_id, scope
                    )

                if not state_changed:
                    logger.debug(
                        "Ignoring duplicate connection updates with no state transition"
                    )
                    continue

                await self._start_new_campaign("Connection message added")

    async def _confirm_disconnect(self, node_id: NodeId, scope: CancelScope) -> None:
        try:
            with scope:
                await anyio.sleep(DEFAULT_CONNECTION_DISCONNECT_GRACE)
                if self._pending_disconnects.get(node_id) is not scope:
                    return
                if node_id not in self._connected_peers:
                    return
                self._connected_peers.remove(node_id)
                logger.debug(f"Disconnect confirmed for {node_id}")
                await self._start_new_campaign(
                    f"Disconnect confirmed for {node_id}"
                )
        except get_cancelled_exc_class():
            logger.debug(f"Disconnect confirmation cancelled for {node_id}")
        finally:
            if self._pending_disconnects.get(node_id) is scope:
                del self._pending_disconnects[node_id]

    def _campaign_active(self) -> bool:
        return self._campaign_cancel_scope is not None

    async def _start_new_campaign(
        self,
        reason: str,
        *,
        initial_candidates: list[ElectionMessage] | None = None,
    ) -> None:
        logger.debug(f"Current clock: {self.clock}")
        self.clock += 1
        logger.debug(f"New clock: {self.clock}")
        candidates: list[ElectionMessage] = [
            candidate.model_copy(update={"clock": self.clock})
            for candidate in (initial_candidates or [])
        ]
        self._candidates = candidates
        logger.debug("Starting new campaign")
        self._tg.start_soon(self._campaign, candidates, DEFAULT_ELECTION_TIMEOUT)
        logger.debug("Campaign started")
        logger.debug(reason)

    async def _command_counter(self) -> None:
        with self._co_receiver as commands:
            async for _command in commands:
                self.commands_seen += 1

    def set_worker_execution_enabled(self, enabled: bool) -> bool:
        normalized = bool(enabled)
        if self._worker_execution_enabled == normalized:
            return False
        self._worker_execution_enabled = normalized
        return True

    def _input_receivers_closed(self) -> bool:
        return all(
            bool(getattr(receiver, "_closed", False))
            for receiver in (
                self._em_receiver,
                self._cm_receiver,
                self._co_receiver,
            )
        )

    async def _watch_local_worker_mode(self) -> None:
        if DEFAULT_WORKER_MODE_POLL_SECONDS <= 0:
            return

        while True:
            await anyio.sleep(DEFAULT_WORKER_MODE_POLL_SECONDS)
            if self._input_receivers_closed():
                return
            enabled = _load_local_worker_mode_enabled()
            if not self.set_worker_execution_enabled(enabled):
                continue
            logger.info(
                "Local worker mode changed to {}. Triggering a new election round.",
                "enabled" if enabled else "disabled",
            )
            await self._start_new_campaign("Worker mode changed")

    async def _campaign(
        self, candidates: list[ElectionMessage], campaign_timeout: float
    ) -> None:
        clock = self.clock

        # Kill the old campaign
        if self._campaign_cancel_scope:
            logger.info("Cancelling other campaign")
            self._campaign_cancel_scope.cancel()
        if self._campaign_done:
            logger.info("Waiting for other campaign to finish")
            await self._campaign_done.wait()

        done = Event()
        self._campaign_done = done
        scope = CancelScope()
        self._campaign_cancel_scope = scope

        try:
            with scope:
                logger.debug(f"Election {clock} started")

                status = self._election_status(clock)
                candidates.append(status)
                await self._em_sender.send(status)

                logger.debug(f"Sleeping for {campaign_timeout} seconds")
                await anyio.sleep(campaign_timeout)
                # minor hack - rebroadcast status in case anyone has missed it.
                await self._em_sender.send(status)
                logger.debug("Woke up from sleep")
                # add an anyio checkpoint - anyio.lowlevel.chekpoint() or checkpoint_if_cancelled() is preferred, but wasn't typechecking last I checked
                await anyio.sleep(0)

                # Election finished!
                elected = max(candidates)
                logger.debug(f"Election queue {candidates}")
                logger.debug(f"Elected: {elected}")
                if (
                    self.node_id == elected.proposed_session.master_node_id
                    and self.seniority >= 0
                ):
                    logger.debug(
                        f"Node is a candidate and seniority is {self.seniority}"
                    )
                    self.seniority = max(self.seniority, len(candidates))
                    logger.debug(f"New seniority: {self.seniority}")
                else:
                    logger.debug(
                        f"Node is not a candidate or seniority is not {self.seniority}"
                    )
                logger.debug(
                    f"Election finished, new SessionId({elected.proposed_session}) with queue {candidates}"
                )
                logger.debug("Sending election result")
                await self.elect(elected)
                logger.debug("Election result sent")
        except get_cancelled_exc_class():
            logger.debug(f"Election {clock} cancelled")
        finally:
            logger.debug(f"Election {clock} finally")
            if self._campaign_cancel_scope is scope:
                self._campaign_cancel_scope = None
            logger.debug("Setting done event")
            done.set()
            logger.debug("Done event set")

    def _election_status(self, clock: int | None = None) -> ElectionMessage:
        c = self.clock if clock is None else clock
        current_master_id = self.current_session.master_node_id
        fallback_master_id = self.node_id
        if self.seniority < 0 and self._connected_peers:
            fallback_master_id = sorted(self._connected_peers, key=str)[0]
        prefer_local_execution_master = (
            DEFAULT_WORKER_MASTER_RECLAIM_ENABLED
            and self._worker_execution_enabled
            and current_master_id != self.node_id
        )
        non_candidate_self_master_should_yield = (
            self.seniority < 0
            and current_master_id == self.node_id
            and bool(self._connected_peers)
        )
        keep_current_master = (
            not prefer_local_execution_master
            and not non_candidate_self_master_should_yield
            and (
                current_master_id == self.node_id
                or current_master_id in self._connected_peers
                or current_master_id in self._pending_disconnects
            )
        )
        effective_seniority = self.seniority
        if self.seniority >= 0 and self._worker_execution_enabled:
            effective_seniority += DEFAULT_WORKER_MASTER_SENIORITY_BONUS
        return ElectionMessage(
            proposed_session=(
                self.current_session
                if keep_current_master
                else SessionId(master_node_id=fallback_master_id, election_clock=c)
            ),
            clock=c,
            seniority=effective_seniority,
            commands_seen=self.commands_seen,
        )


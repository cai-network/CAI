# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import anyio
import pytest

from cai.routing.event_router import EventRouter
from cai.shared.types.commands import ForwarderCommand
from cai.shared.types.common import NodeId, SessionId
from cai.shared.types.events import GlobalForwarderEvent, TestEvent
from cai.utils.channels import channel


@pytest.mark.anyio
async def test_event_router_accepts_same_master_with_different_session_clock() -> None:
    command_send, command_recv = channel[ForwarderCommand]()
    inbound_send, inbound_recv = channel[GlobalForwarderEvent]()
    outbound_send, outbound_recv = channel()
    router = EventRouter(
        SessionId(master_node_id=NodeId("master-a"), election_clock=5),
        command_send,
        inbound_recv,
        outbound_send,
    )
    receiver = router.receiver()
    event = TestEvent()

    async with anyio.create_task_group() as tg:
        tg.start_soon(router.run)
        await inbound_send.send(
            GlobalForwarderEvent(
                origin_idx=0,
                origin=NodeId("master-a"),
                session=SessionId(
                    master_node_id=NodeId("master-a"),
                    election_clock=3,
                ),
                event=event,
            )
        )
        with anyio.fail_after(1):
            indexed = await receiver.receive()
        router.shutdown()

    assert indexed.idx == 0
    assert indexed.event == event
    command_send.close()
    command_recv.close()
    inbound_send.close()
    outbound_recv.close()

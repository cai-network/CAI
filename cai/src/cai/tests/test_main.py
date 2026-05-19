# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import anyio
import importlib
import pytest
from types import SimpleNamespace

from cai.main import (
    Args,
    Node,
    NodeInfoPublisher,
    _resolve_advertised_bootstrap_peers,
    _resolve_master_candidate_enabled,
    _new_boot_session_clock,
    _stabilize_election_result,
)
from cai.shared.election import ElectionResult
from cai.shared.types.common import NodeId, SessionId
from cai.shared.types.events import NodeGatheredInfo, OverlayBootstrapPeersAdvertised
from cai.shared.types.multiaddr import Multiaddr
from cai.utils.channels import channel
from cai.utils.info_gatherer.info_gatherer import ApiEndpointInfo


def _result(
    master_node_id: str,
    *,
    election_clock: int,
    won_clock: int,
    is_new_master: bool,
) -> ElectionResult:
    return ElectionResult(
        session_id=SessionId(
            master_node_id=NodeId(master_node_id),
            election_clock=election_clock,
        ),
        won_clock=won_clock,
        is_new_master=is_new_master,
    )


def test_new_boot_session_clock_uses_explicit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAI_BOOT_SESSION_CLOCK", "123456")

    assert _new_boot_session_clock() == 123456


def test_new_boot_session_clock_is_process_incarnation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CAI_BOOT_SESSION_CLOCK", raising=False)

    first = _new_boot_session_clock()
    second = _new_boot_session_clock()

    assert first > 0
    assert second >= first


@pytest.mark.anyio
async def test_stabilize_election_result_coalesces_back_to_current_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("cai.main.DEFAULT_SESSION_TRANSITION_DEBOUNCE", 0.15)

    current_session = SessionId(
        master_node_id=NodeId("REMOTE"),
        election_clock=7,
    )
    first = _result(
        "LOCAL",
        election_clock=8,
        won_clock=8,
        is_new_master=True,
    )
    recovered = _result(
        "REMOTE",
        election_clock=7,
        won_clock=9,
        is_new_master=True,
    )

    send, recv = channel[ElectionResult]()

    async def _feed() -> None:
        await anyio.sleep(0.05)
        await send.send(recovered)
        send.close()

    async with anyio.create_task_group() as tg:
        tg.start_soon(_feed)
        stabilized = await _stabilize_election_result(
            recv,
            first,
            current_session_id=current_session,
        )

    assert stabilized.session_id == current_session
    assert stabilized.won_clock == 9


@pytest.mark.anyio
async def test_stabilize_election_result_keeps_latest_duplicate_session_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("cai.main.DEFAULT_SESSION_TRANSITION_DEBOUNCE", 0.15)

    current_session = SessionId(
        master_node_id=NodeId("REMOTE"),
        election_clock=7,
    )
    first = _result(
        "LOCAL",
        election_clock=8,
        won_clock=8,
        is_new_master=True,
    )
    duplicate = _result(
        "LOCAL",
        election_clock=8,
        won_clock=9,
        is_new_master=False,
    )

    send, recv = channel[ElectionResult]()

    async def _feed() -> None:
        await anyio.sleep(0.05)
        await send.send(duplicate)
        send.close()

    async with anyio.create_task_group() as tg:
        tg.start_soon(_feed)
        stabilized = await _stabilize_election_result(
            recv,
            first,
            current_session_id=current_session,
        )

    assert stabilized.session_id == first.session_id
    assert stabilized.won_clock == 9


@pytest.mark.anyio
async def test_node_info_publisher_forwards_local_gathered_info() -> None:
    event_send, event_recv = channel[NodeGatheredInfo]()
    info_send, info_recv = channel[ApiEndpointInfo]()
    publisher = NodeInfoPublisher(
        node_id=NodeId("node-local"),
        api_port=52415,
        event_sender=event_send,
    )

    async def _feed() -> None:
        await info_send.send(ApiEndpointInfo(host="85.137.164.250", port=52415))
        info_send.close()

    async with anyio.create_task_group() as tg:
        tg.start_soon(_feed)
        tg.start_soon(publisher._forward_info, info_recv)  # pyright: ignore[reportPrivateUsage]
        forwarded = await event_recv.receive()

    assert isinstance(forwarded, NodeGatheredInfo)
    assert forwarded.node_id == NodeId("node-local")
    assert isinstance(forwarded.info, ApiEndpointInfo)
    assert forwarded.info.host == "85.137.164.250"
    assert forwarded.info.port == 52415


@pytest.mark.anyio
async def test_node_retries_overlay_bootstrap_peer_advertisement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("cai.main._overlay_advertise_retry_delays", lambda: (0.0, 0.0))

    event_send, event_recv = channel()
    node = object.__new__(Node)
    node.node_id = NodeId("node-local")
    node._peer_event_sender = event_send
    node._advertised_bootstrap_peers = (
        Multiaddr(address="/ip4/26.97.29.153/tcp/52426"),
    )

    async with anyio.create_task_group() as tg:
        tg.start_soon(node._advertise_overlay_bootstrap_peers)  # pyright: ignore[reportPrivateUsage]
        first = await event_recv.receive()
        second = await event_recv.receive()

    assert isinstance(first, OverlayBootstrapPeersAdvertised)
    assert isinstance(second, OverlayBootstrapPeersAdvertised)
    assert first.node_id == NodeId("node-local")
    assert [peer.address for peer in first.peers] == ["/ip4/26.97.29.153/tcp/52426"]
    assert second.node_id == first.node_id


def test_resolve_master_candidate_enabled_disables_bootstrapped_worker_node() -> None:
    enabled, reason = _resolve_master_candidate_enabled(
        bootstrap_peers=["/ip4/85.137.164.250/tcp/52416/p2p/bootstrap"],
        worker_runtime_enabled=True,
        force_master=False,
        config=SimpleNamespace(
            worker_enabled=True,
            validator_enabled=False,
            validator_state="unbonded",
        ),
    )

    assert enabled is False
    assert reason == "disabled for bootstrap worker node"


def test_resolve_master_candidate_enabled_keeps_no_worker_bootstrap_node_candidate() -> None:
    enabled, reason = _resolve_master_candidate_enabled(
        bootstrap_peers=["/ip4/85.137.164.250/tcp/52416/p2p/bootstrap"],
        worker_runtime_enabled=False,
        force_master=False,
        config=SimpleNamespace(
            worker_enabled=True,
            validator_enabled=False,
            validator_state="unbonded",
        ),
    )

    assert enabled is True
    assert reason == "enabled by default"


def test_resolve_master_candidate_enabled_keeps_validator_candidate() -> None:
    enabled, reason = _resolve_master_candidate_enabled(
        bootstrap_peers=["/ip4/85.137.164.250/tcp/52416/p2p/bootstrap"],
        worker_runtime_enabled=True,
        force_master=False,
        config=SimpleNamespace(
            worker_enabled=True,
            validator_enabled=True,
            validator_state="bonded",
        ),
    )

    assert enabled is True
    assert reason == "enabled for validator-capable node"


def test_resolve_master_candidate_enabled_honors_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAI_MASTER_CANDIDATE", "0")

    enabled, reason = _resolve_master_candidate_enabled(
        bootstrap_peers=[],
        worker_runtime_enabled=False,
        force_master=False,
        config=SimpleNamespace(
            worker_enabled=False,
            validator_enabled=False,
            validator_state="unbonded",
        ),
    )

    assert enabled is False
    assert reason == "disabled via environment override"


def test_cai_alias_import_resolves_cai_modules() -> None:
    cai_api_main = importlib.import_module("cai.api.main")
    cai_api_main = importlib.import_module("cai.api.main")
    cai_common = importlib.import_module("cai.shared.types.common")
    cai_common = importlib.import_module("cai.shared.types.common")

    assert cai_api_main is cai_api_main
    assert cai_common is cai_common


def test_args_parse_prefers_cai_bootstrap_peers_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CAI_BOOTSTRAP_PEERS",
        "/ip4/85.137.164.250/tcp/52416,/ip4/198.51.100.20/tcp/52416",
    )
    monkeypatch.delenv("CAI_BOOTSTRAP_PEERS", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        ["cai", "--no-worker"],
    )

    args = Args.parse()

    assert args.bootstrap_peers == [
        "/ip4/85.137.164.250/tcp/52416",
        "/ip4/198.51.100.20/tcp/52416",
    ]


@pytest.mark.anyio
async def test_resolve_advertised_bootstrap_peers_uses_explicit_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CAI_ADVERTISE_PEERS",
        "/ip4/85.137.164.250/tcp/52416",
    )

    peers = await _resolve_advertised_bootstrap_peers(
        NodeId("node-a"),
        listen_port=52416,
    )

    assert [peer.address for peer in peers] == [
        "/ip4/85.137.164.250/tcp/52416"
    ]


@pytest.mark.anyio
async def test_resolve_advertised_bootstrap_peers_prefers_cai_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CAI_ADVERTISE_PEERS",
        "/ip4/85.137.164.250/tcp/52416",
    )
    monkeypatch.delenv("CAI_ADVERTISE_PEERS", raising=False)

    peers = await _resolve_advertised_bootstrap_peers(
        NodeId("node-a"),
        listen_port=52416,
    )

    assert [peer.address for peer in peers] == [
        "/ip4/85.137.164.250/tcp/52416"
    ]


@pytest.mark.anyio
async def test_resolve_advertised_bootstrap_peers_auto_builds_self_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CAI_ADVERTISE_PEERS", raising=False)
    monkeypatch.setenv("CAI_AUTO_ADVERTISE_OVERLAY_PEER", "1")

    async def _fake_resolve_advertised_host(explicit_host: str | None = None) -> str | None:
        assert explicit_host is None
        return "26.97.29.153"

    monkeypatch.setattr("cai.main.resolve_advertised_host", _fake_resolve_advertised_host)

    peers = await _resolve_advertised_bootstrap_peers(
        NodeId("node-auto"),
        listen_port=52426,
    )

    assert [peer.address for peer in peers] == [
        "/ip4/26.97.29.153/tcp/52426"
    ]

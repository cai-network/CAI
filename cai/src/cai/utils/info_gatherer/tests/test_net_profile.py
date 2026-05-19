# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import pytest

from cai.shared.topology import Topology
from cai.shared.types.common import NodeId
from cai.shared.types.profiling import NodeIdentity, NodeNetworkInfo, NetworkInterfaceInfo
from cai.utils.info_gatherer.net_profile import candidate_targets, check_reachable


def test_candidate_targets_prefers_advertised_endpoint() -> None:
    node_id = NodeId("node-a")
    node_network = {
        node_id: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(name="eth0", ip_address="10.0.0.5"),
                NetworkInterfaceInfo(name="eth1", ip_address="192.168.1.20"),
            ]
        )
    }
    node_identities = {
        node_id: NodeIdentity(api_host="203.0.113.10", api_port=18080)
    }

    assert list(
        candidate_targets(
            node_id,
            node_network,
            node_identities,
            default_api_port=52415,
        )
    ) == [
        ("203.0.113.10", 18080),
        ("10.0.0.5", 18080),
        ("192.168.1.20", 18080),
    ]


def test_candidate_targets_falls_back_to_default_port() -> None:
    node_id = NodeId("node-a")
    node_network = {
        node_id: NodeNetworkInfo(
            interfaces=[NetworkInterfaceInfo(name="eth0", ip_address="10.0.0.5")]
        )
    }

    assert list(
        candidate_targets(
            node_id,
            node_network,
            {},
            default_api_port=52415,
        )
    ) == [("10.0.0.5", 52415)]


def test_candidate_targets_skips_loopback_and_unspecified_remote_addresses() -> None:
    node_id = NodeId("node-a")
    node_network = {
        node_id: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(name="loopback4", ip_address="127.0.0.1"),
                NetworkInterfaceInfo(name="loopback6", ip_address="::1"),
                NetworkInterfaceInfo(name="unspecified4", ip_address="0.0.0.0"),
                NetworkInterfaceInfo(name="vpn", ip_address="26.242.160.75"),
                NetworkInterfaceInfo(name="lan", ip_address="192.168.0.103"),
            ]
        )
    }
    node_identities = {
        node_id: NodeIdentity(api_host="127.0.0.1", api_port=18080)
    }

    assert list(
        candidate_targets(
            node_id,
            node_network,
            node_identities,
            default_api_port=52415,
        )
    ) == [
        ("26.242.160.75", 18080),
        ("192.168.0.103", 18080),
    ]


@pytest.mark.anyio
async def test_check_reachable_uses_overlay_trusted_advertised_endpoint_when_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fail_probe(*args, **kwargs) -> None:  # noqa: ARG001
        return None

    monkeypatch.setattr(
        "cai.utils.info_gatherer.net_profile.check_reachability",
        _fail_probe,
    )

    self_node_id = NodeId("node-self")
    remote_node_id = NodeId("node-remote")
    topology = Topology()
    topology.add_node(self_node_id)
    topology.add_node(remote_node_id)

    node_network = {
        remote_node_id: NodeNetworkInfo(
            interfaces=[NetworkInterfaceInfo(name="eth0", ip_address="10.0.0.5")]
        )
    }
    node_identities = {
        remote_node_id: NodeIdentity(api_host="203.0.113.10", api_port=18080)
    }
    overlay_peers = {self_node_id: [remote_node_id]}

    discovered = [
        item
        async for item in check_reachable(
            topology,
            self_node_id,
            node_network,
            node_identities,
            api_port=52415,
            overlay_peers=overlay_peers,
            include_overlay_fallback=True,
        )
    ]

    assert discovered == [("203.0.113.10", 18080, remote_node_id)]


@pytest.mark.anyio
async def test_check_reachable_does_not_trust_non_overlay_peer_when_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fail_probe(*args, **kwargs) -> None:  # noqa: ARG001
        return None

    monkeypatch.setattr(
        "cai.utils.info_gatherer.net_profile.check_reachability",
        _fail_probe,
    )

    self_node_id = NodeId("node-self")
    remote_node_id = NodeId("node-remote")
    topology = Topology()
    topology.add_node(self_node_id)
    topology.add_node(remote_node_id)

    node_network = {
        remote_node_id: NodeNetworkInfo(
            interfaces=[NetworkInterfaceInfo(name="eth0", ip_address="10.0.0.5")]
        )
    }
    node_identities = {
        remote_node_id: NodeIdentity(api_host="203.0.113.10", api_port=18080)
    }

    discovered = [
        item
        async for item in check_reachable(
            topology,
            self_node_id,
            node_network,
            node_identities,
            api_port=52415,
            overlay_peers={},
            include_overlay_fallback=True,
        )
    ]

    assert discovered == []


@pytest.mark.anyio
async def test_check_reachable_is_strict_by_default_when_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fail_probe(*args, **kwargs) -> None:  # noqa: ARG001
        return None

    monkeypatch.setattr(
        "cai.utils.info_gatherer.net_profile.check_reachability",
        _fail_probe,
    )

    self_node_id = NodeId("node-self")
    remote_node_id = NodeId("node-remote")
    topology = Topology()
    topology.add_node(self_node_id)
    topology.add_node(remote_node_id)

    node_network = {
        remote_node_id: NodeNetworkInfo(
            interfaces=[NetworkInterfaceInfo(name="eth0", ip_address="10.0.0.5")]
        )
    }
    node_identities = {
        remote_node_id: NodeIdentity(api_host="203.0.113.10", api_port=18080)
    }
    overlay_peers = {self_node_id: [remote_node_id]}

    discovered = [
        item
        async for item in check_reachable(
            topology,
            self_node_id,
            node_network,
            node_identities,
            api_port=52415,
            overlay_peers=overlay_peers,
        )
    ]

    assert discovered == []


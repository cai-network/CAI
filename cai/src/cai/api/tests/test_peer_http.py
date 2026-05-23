# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

from cai.api.peer_http import (
    api_base_url_from_multiaddr,
    bootstrap_api_base_url_for_node,
    cai_summary_urls_by_node_id,
    candidate_http_urls_from_identity,
)
from cai.shared.types.common import NodeId
from cai.shared.types.profiling import AdvertisedTransportEndpoint, NodeIdentity


def test_api_base_url_from_multiaddr_supports_ip_and_dns() -> None:
    assert api_base_url_from_multiaddr("/ip4/192.145.29.212/tcp/4001", 52415) == (
        "http://192.145.29.212:52415"
    )
    assert api_base_url_from_multiaddr("/ip6/::1/tcp/4001", 52415) == (
        "http://[::1]:52415"
    )
    assert api_base_url_from_multiaddr("/dns4/validator.cai/tcp/4001", 52415) == (
        "http://validator.cai:52415"
    )


def test_bootstrap_api_base_url_prefers_matching_node_id() -> None:
    config = SimpleNamespace(
        default_api_port=52415,
        bootstrap_peers=[
            "/ip4/10.0.0.1/tcp/4001",
            "/ip4/192.145.29.212/tcp/4001/p2p/node-target",
        ],
    )

    assert bootstrap_api_base_url_for_node("node-target", config=config) == (
        "http://192.145.29.212:52415"
    )
    assert bootstrap_api_base_url_for_node("missing-node", config=config) == (
        "http://10.0.0.1:52415"
    )


def test_candidate_http_urls_from_mapping_prioritizes_direct_explicit_api_endpoint() -> None:
    identity = {
        "apiHost": "fallback.local",
        "apiPort": 52415,
        "transportEndpoints": [
            {
                "purpose": "api",
                "routeType": "relay",
                "host": "relay.local",
                "port": 52415,
                "source": "auto",
            },
            {
                "purpose": "api",
                "routeType": "direct",
                "host": "direct.local",
                "port": 52415,
                "source": "explicit",
            },
        ],
    }

    assert candidate_http_urls_from_identity(identity, endpoint_path="/v1/cai/summary") == [
        "http://direct.local:52415/v1/cai/summary",
        "http://relay.local:52415/v1/cai/summary",
        "http://fallback.local:52415/v1/cai/summary",
    ]


def test_cai_summary_urls_by_node_id_uses_local_loopback_and_remote_candidates() -> None:
    identities = {
        NodeId("local-node"): {"apiHost": "127.0.0.1", "apiPort": 52415},
        NodeId("remote-node"): NodeIdentity(
            api_host="remote.local",
            api_port=52416,
            transport_endpoints=[
                AdvertisedTransportEndpoint(
                    purpose="api",
                    route_type="direct",
                    host="remote-direct.local",
                    port=52416,
                    source="explicit",
                )
            ],
        ),
    }

    assert cai_summary_urls_by_node_id(
        node_identities=identities,
        local_port=52415,
    ) == {
        "local-node": "http://127.0.0.1:52415/v1/cai/summary",
        "remote-node": "http://remote-direct.local:52416/v1/cai/summary",
    }

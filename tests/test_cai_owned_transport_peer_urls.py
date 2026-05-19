# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import unittest

from cai_compute_chain.cai_owned_transport_peer_urls import (
    cai_owned_transport_peer_url_priority,
    cai_owned_transport_peer_url_route_class,
    clean_peer_cai_urls,
    prioritized_cai_owned_transport_peer_urls,
)


class CaiOwnedTransportPeerUrlTests(unittest.TestCase):
    def test_clean_peer_cai_urls_dedupes_and_normalizes_trailing_slashes(self) -> None:
        self.assertEqual(
            clean_peer_cai_urls(
                [
                    " http://node-a:52415/ ",
                    "http://node-a:52415",
                    "",
                    "cai-overlay:http://node-b:52415?relayRole=worker/",
                ]
            ),
            [
                "http://node-a:52415",
                "cai-overlay:http://node-b:52415?relayRole=worker",
            ],
        )

    def test_prioritized_peer_urls_prefer_direct_then_ordinary_then_bootstrap(self) -> None:
        direct = "http://worker:52415"
        ordinary = "cai-overlay:http://relay:52415?relayRole=worker"
        bootstrap = "cai-overlay:http://validator:52415?relayRole=bootstrap"
        generic = "cai-overlay:http://relay:52415"

        self.assertEqual(
            prioritized_cai_owned_transport_peer_urls(
                [bootstrap, generic, ordinary, direct]
            ),
            [direct, ordinary, generic, bootstrap],
        )
        self.assertLess(
            cai_owned_transport_peer_url_priority(direct),
            cai_owned_transport_peer_url_priority(ordinary),
        )
        self.assertLess(
            cai_owned_transport_peer_url_priority(ordinary),
            cai_owned_transport_peer_url_priority(bootstrap),
        )

    def test_peer_url_route_class_distinguishes_overlay_roles(self) -> None:
        self.assertEqual(
            cai_owned_transport_peer_url_route_class("http://worker:52415"),
            "direct",
        )
        self.assertEqual(
            cai_owned_transport_peer_url_route_class(
                "cai-overlay:http://relay:52415?relayRole=worker"
            ),
            "overlay_ordinary",
        )
        self.assertEqual(
            cai_owned_transport_peer_url_route_class(
                "cai-overlay:http://validator:52415?relayRole=bootstrap"
            ),
            "overlay_bootstrap",
        )
        self.assertEqual(
            cai_owned_transport_peer_url_route_class("cai-overlay:not a url"),
            "overlay_generic",
        )
        self.assertEqual(
            cai_owned_transport_peer_url_route_class("cai-overlay:http://["),
            "overlay_invalid",
        )


if __name__ == "__main__":
    unittest.main()

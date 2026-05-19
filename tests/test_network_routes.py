# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.network_routes import relay_route_candidates


class NetworkRouteTests(unittest.TestCase):
    def test_relay_routes_are_spread_across_equivalent_transit_peers(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-a": {},
                "node-b": {},
                "relay-a": {"relayEnabled": True},
                "node-relay-a": {"relayEnabled": True},
            },
            "overlayPeers": {
                "node-a": ["relay-a", "node-relay-a"],
                "node-b": ["relay-a", "node-relay-a"],
                "relay-a": ["node-a", "node-b"],
                "node-relay-a": ["node-a", "node-b"],
            },
            "topology": {
                "nodes": ["node-a", "node-b", "relay-a", "node-relay-a"],
                "connections": {},
            },
        }

        routes = relay_route_candidates(state_payload, ["node-a", "node-b"])
        transit_by_pair = {
            (route["sourceNodeId"], route["sinkNodeId"]): route["transitNodeId"]
            for route in routes
        }

        self.assertEqual(transit_by_pair[("node-a", "node-b")], "relay-a")
        self.assertEqual(transit_by_pair[("node-b", "node-a")], "node-relay-a")

    def test_relay_routes_balance_equivalent_transit_peers(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-a": {},
                "node-b": {},
                "node-c": {},
                "relay-a": {"relayEnabled": True},
                "relay-b": {"relayEnabled": True},
            },
            "overlayPeers": {
                "node-a": ["relay-a", "relay-b"],
                "node-b": ["relay-a", "relay-b"],
                "node-c": ["relay-a", "relay-b"],
                "relay-a": ["node-a", "node-b", "node-c"],
                "relay-b": ["node-a", "node-b", "node-c"],
            },
            "topology": {
                "nodes": ["node-a", "node-b", "node-c", "relay-a", "relay-b"],
                "connections": {},
            },
        }

        routes = relay_route_candidates(state_payload, ["node-a", "node-b", "node-c"])
        transit_counts = Counter(route["transitNodeId"] for route in routes)

        self.assertEqual(len(routes), 6)
        self.assertEqual(set(transit_counts), {"relay-a", "relay-b"})
        self.assertLessEqual(
            max(transit_counts.values()) - min(transit_counts.values()),
            1,
        )

    def test_relay_routes_can_return_all_alternatives_for_scoring(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-a": {},
                "node-b": {},
                "relay-a": {"relayEnabled": True},
                "relay-b": {"relayEnabled": True},
            },
            "overlayPeers": {
                "node-a": ["relay-a", "relay-b"],
                "node-b": ["relay-a", "relay-b"],
                "relay-a": ["node-a", "node-b"],
                "relay-b": ["node-a", "node-b"],
            },
            "topology": {
                "nodes": ["node-a", "node-b", "relay-a", "relay-b"],
                "connections": {},
            },
        }

        selected_routes = relay_route_candidates(state_payload, ["node-a", "node-b"])
        all_routes = relay_route_candidates(
            state_payload,
            ["node-a", "node-b"],
            include_alternatives=True,
        )

        self.assertEqual(len(selected_routes), 2)
        self.assertEqual(len(all_routes), 4)
        self.assertEqual(
            Counter(route["transitNodeId"] for route in all_routes),
            {"relay-a": 2, "relay-b": 2},
        )

    def test_relay_routes_can_use_third_participant_as_transit(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-a": {"workerEnabled": True},
                "node-b": {"workerEnabled": True},
                "node-c": {"workerEnabled": True, "relayEnabled": True},
            },
            "overlayPeers": {
                "node-a": ["node-c"],
                "node-c": ["node-b"],
            },
            "topology": {
                "nodes": ["node-a", "node-b", "node-c"],
                "connections": {},
            },
        }

        routes = relay_route_candidates(
            state_payload,
            ["node-a", "node-b", "node-c"],
        )

        route = next(
            item
            for item in routes
            if item["sourceNodeId"] == "node-a" and item["sinkNodeId"] == "node-b"
        )
        self.assertEqual(route["transitNodeId"], "node-c")
        self.assertTrue(route["transitParticipates"])


if __name__ == "__main__":
    unittest.main()

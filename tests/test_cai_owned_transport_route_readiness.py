# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import unittest

from cai_compute_chain.cai_owned_transport_route_readiness import (
    cai_owned_transport_route_health_readiness,
    preflight_cai_owned_transport_data_plane_routes,
)
from cai_compute_chain.decentralized_compute import (
    preflight_cai_owned_transport_data_plane_routes as compatibility_preflight,
)


class CaiOwnedTransportRouteReadinessTests(unittest.TestCase):
    def test_route_preflight_prefers_direct_peer_url_over_bootstrap_overlay(
        self,
    ) -> None:
        preflight = preflight_cai_owned_transport_data_plane_routes(
            requester_node_id=" requester ",
            executor_node_ids=["worker", "worker", ""],
            peer_cai_urls_by_node={
                "requester": ["http://requester:52415/"],
                "worker": [
                    "cai-overlay:http://validator:52415?relayRole=bootstrap",
                    "http://worker:52415",
                ],
            },
        )

        worker_audit = next(
            item for item in preflight["nodeAudits"] if item["nodeId"] == "worker"
        )
        self.assertEqual(preflight["status"], "ready")
        self.assertEqual(preflight["executorNodeIds"], ["worker"])
        self.assertEqual(worker_audit["selectedPeerCaiUrl"], "http://worker:52415")
        self.assertEqual(worker_audit["selectedRouteClass"], "direct")
        self.assertEqual(
            worker_audit["peerCaiUrls"],
            [
                "http://worker:52415",
                "cai-overlay:http://validator:52415?relayRole=bootstrap",
            ],
        )

    def test_route_health_readiness_accepts_camel_case_records(self) -> None:
        readiness = cai_owned_transport_route_health_readiness(
            source_node_id="requester",
            sink_node_ids=["worker"],
            route_health_records=[
                {
                    "sourceNodeId": "requester",
                    "sinkNodeId": "worker",
                    "routeType": "direct_data",
                    "endpointUrl": "tcp://worker:52435",
                    "reachable": True,
                    "latencyMs": 8.5,
                    "checkedAt": "2026-05-20T00:00:00+00:00",
                },
                {
                    "sourceNodeId": "worker",
                    "sinkNodeId": "requester",
                    "routeType": "direct_data",
                    "endpointUrl": "tcp://requester:52435",
                    "reachable": True,
                    "latencyMs": 8.5,
                    "checkedAt": "2026-05-20T00:00:00+00:00",
                }
            ],
        )

        self.assertTrue(readiness["ready"])
        self.assertEqual(readiness["status"], "ready")
        self.assertEqual(readiness["routeHealthAudits"][0]["routeHealthScore"], 4)
        self.assertEqual(readiness["routeHealthAudits"][0]["routeType"], "direct_data")

    def test_route_health_readiness_prefers_direct_data_over_relay(self) -> None:
        readiness = cai_owned_transport_route_health_readiness(
            source_node_id="requester",
            sink_node_ids=["worker"],
            route_health_records=[
                {
                    "sourceNodeId": "requester",
                    "sinkNodeId": "worker",
                    "routeType": "relay_active",
                    "reachable": True,
                    "transitNodeId": "validator",
                    "endpointUrl": "relay://validator/worker",
                    "checkedAt": "2026-05-20T00:00:10+00:00",
                },
                {
                    "sourceNodeId": "requester",
                    "sinkNodeId": "worker",
                    "routeType": "direct_data",
                    "endpointUrl": "tcp://worker:52435",
                    "reachable": True,
                    "checkedAt": "2026-05-20T00:00:00+00:00",
                },
                {
                    "sourceNodeId": "worker",
                    "sinkNodeId": "requester",
                    "routeType": "direct_data",
                    "endpointUrl": "tcp://requester:52435",
                    "reachable": True,
                    "checkedAt": "2026-05-20T00:00:00+00:00",
                },
            ],
        )

        hop = readiness["routeHealthAudits"][0]
        self.assertTrue(readiness["ready"])
        self.assertEqual(hop["routeType"], "direct_data")
        self.assertEqual(hop["routeHealthScore"], 4)
        self.assertEqual(hop["endpointUrl"], "tcp://worker:52435")
        self.assertIsNone(hop["transitNodeId"])

    def test_relay_quorum_requires_independent_transit_routes(self) -> None:
        route_policy = {
            "minimumRelayQuorum": 2,
            "executionDag": {
                "stages": [
                    {"sourceNodeId": "requester", "sinkNodeId": "worker-a"},
                    {"sourceNodeId": "requester", "sinkNodeId": "worker-b"},
                ]
            },
        }
        records = [
            _relay_record("requester", "worker-a", "relay-1"),
            _relay_record("requester", "worker-a", "relay-2"),
            _relay_record("requester", "worker-b", "relay-1"),
            _relay_record("requester", "worker-b", "relay-2"),
        ]

        preflight = preflight_cai_owned_transport_data_plane_routes(
            requester_node_id="requester",
            executor_node_ids=["worker-a", "worker-b"],
            peer_cai_urls_by_node={
                "requester": ["http://requester:52415"],
                "worker-a": ["http://worker-a:52415"],
                "worker-b": ["http://worker-b:52415"],
            },
            route_policy=route_policy,
            route_health_records=records,
            require_route_health=True,
        )
        failed_preflight = preflight_cai_owned_transport_data_plane_routes(
            requester_node_id="requester",
            executor_node_ids=["worker-a", "worker-b"],
            peer_cai_urls_by_node={
                "requester": ["http://requester:52415"],
                "worker-a": ["http://worker-a:52415"],
                "worker-b": ["http://worker-b:52415"],
            },
            route_policy=route_policy,
            route_health_records=records[:-1],
            require_route_health=True,
        )

        self.assertEqual(preflight["status"], "ready")
        self.assertTrue(all(item["ready"] for item in preflight["relayQuorumAudits"]))
        self.assertEqual(failed_preflight["status"], "failed")
        self.assertIn(
            "Need at least 2 independent transit route(s)",
            " ".join(failed_preflight["fatalReasons"]),
        )

    def test_decentralized_compute_keeps_compatibility_import(self) -> None:
        self.assertIs(
            compatibility_preflight,
            preflight_cai_owned_transport_data_plane_routes,
        )


def _relay_record(source: str, sink: str, transit: str) -> dict[str, object]:
    return {
        "sourceNodeId": source,
        "sinkNodeId": sink,
        "routeType": "relay_active",
        "reachable": True,
        "transitNodeId": transit,
        "endpointUrl": f"relay://{transit}/{sink}",
        "checkedAt": "2026-05-20T00:00:00+00:00",
    }


if __name__ == "__main__":
    unittest.main()

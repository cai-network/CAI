# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
import json
import struct
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.model import WalletPolicy
from cai_compute_chain.route_health import (
    RouteHealthRecord,
    list_route_health_records,
    llama_cpp_compute_cell_profile_for_path,
    probe_direct_api_routes,
    probe_direct_data_routes,
    probe_llama_cpp_rpc_routes,
    prune_stale_route_health_records,
    record_llama_cpp_rpc_result,
    record_overlay_routes_from_state,
    record_relay_probe_result,
    record_route_health_from_network_audit,
    route_health_score_for_path,
    save_route_health_records,
    score_relay_route_candidates,
)

DEFAULT_BACKOFF_TEST_SECONDS = 3600


class _FakeSocket:
    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None


class _FakeRpcSocket:
    def __init__(self, *, major: int = 3, minor: int = 6, patch: int = 0) -> None:
        response = bytes([major, minor, patch, 0]) + bytes(24)
        self._response = struct.pack("<Q", len(response)) + response
        self.sent = bytearray()

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def sendall(self, payload: bytes) -> None:
        self.sent.extend(payload)

    def recv(self, size: int) -> bytes:
        chunk = self._response[:size]
        self._response = self._response[size:]
        return chunk


class _FakeHttpResponse:
    def __init__(self, payload: object, *, status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def read(self) -> bytes:
        if isinstance(self.payload, bytes):
            return self.payload
        return json.dumps(self.payload).encode("utf-8")


class RouteHealthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo_patch = patch(
            "cai_compute_chain.wallet.repo_root",
            return_value=Path(self.tempdir.name),
        )
        self.repo_patch.start()
        self.policy = WalletPolicy(wallet_data_dirname=".tmp-route-health")

    def tearDown(self) -> None:
        self.repo_patch.stop()
        self.tempdir.cleanup()

    def test_list_route_health_records_heals_corrupt_file(self) -> None:
        path = Path(self.tempdir.name) / ".tmp-route-health" / "route-health.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00" * 128)

        self.assertEqual(list_route_health_records(self.policy), [])
        self.assertEqual(path.read_text(encoding="utf-8"), "[]")
        backups = list(path.parent.glob("route-health.corrupt-*.json"))
        self.assertEqual(len(backups), 1)

    def test_record_route_health_from_network_audit(self) -> None:
        records = record_route_health_from_network_audit(
            {
                "checkedDirectSocketLinks": [
                    {
                        "sourceNodeId": "node-a",
                        "sinkNodeId": "node-b",
                        "bidirectional": True,
                    }
                ],
                "checkedRelayRoutes": [
                    {
                        "candidateOnly": False,
                        "sourceNodeId": "node-a",
                        "sinkNodeId": "node-c",
                        "transitNodeId": "node-relay",
                    },
                    {
                        "candidateOnly": True,
                        "sourceNodeId": "node-b",
                        "sinkNodeId": "node-c",
                        "transitNodeId": "node-relay",
                    },
                ],
            },
            policy=self.policy,
        )

        self.assertEqual(len(records), 3)
        stored = list_route_health_records(self.policy)
        self.assertEqual(len(stored), 3)
        direct = next(item for item in stored if item.route_type == "direct_data")
        self.assertTrue(direct.reachable)
        active_relay = next(item for item in stored if item.route_type == "relay_active")
        self.assertTrue(active_relay.reachable)
        candidate_relay = next(
            item for item in stored if item.route_type == "relay_candidate"
        )
        self.assertFalse(candidate_relay.reachable)

    def test_probe_direct_api_routes_records_success_and_failure(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-local": {"apiHost": "127.0.0.1", "apiPort": 52415},
                "node-ok": {"apiHost": "198.51.100.10", "apiPort": 52415},
                "node-missing": {},
            }
        }

        with patch(
            "cai_compute_chain.route_health.urlopen",
            return_value=_FakeHttpResponse("node-ok"),
        ) as urlopen:
            records = probe_direct_api_routes(
                state_payload=state_payload,
                local_node_id="node-local",
                policy=self.policy,
            )

        self.assertEqual(len(records), 2)
        ok = next(item for item in records if item.sink_node_id == "node-ok")
        missing = next(item for item in records if item.sink_node_id == "node-missing")
        self.assertTrue(ok.reachable)
        self.assertEqual(ok.endpoint_url, "http://198.51.100.10:52415")
        self.assertFalse(missing.reachable)
        self.assertIn("no candidate", missing.error or "")
        urlopen.assert_called_once_with(
            "http://198.51.100.10:52415/node_id",
            timeout=1.0,
        )

    def test_probe_direct_api_routes_rejects_wrong_http_node(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-local": {"apiHost": "127.0.0.1", "apiPort": 52415},
                "node-expected": {"apiHost": "198.51.100.10", "apiPort": 52415},
            }
        }

        with patch(
            "cai_compute_chain.route_health.urlopen",
            return_value=_FakeHttpResponse("node-other"),
        ):
            records = probe_direct_api_routes(
                state_payload=state_payload,
                local_node_id="node-local",
                policy=self.policy,
            )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertFalse(record.reachable)
        self.assertEqual(record.endpoint_url, "http://198.51.100.10:52415")
        self.assertIn("node_id mismatch", record.error or "")

    def test_probe_direct_data_routes_records_data_endpoint_reachability(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-local": {"apiHost": "127.0.0.1", "apiPort": 52415},
                "node-endpoint": {
                    "transportEndpoints": [
                        {
                            "purpose": "data",
                            "routeType": "overlay",
                            "host": "26.97.29.153",
                            "port": 62002,
                        }
                    ]
                },
                "node-fallback": {
                    "dataHost": "198.51.100.20",
                    "dataPort": 52435,
                },
                "node-missing": {},
            }
        }

        with patch(
            "cai_compute_chain.route_health.socket.create_connection",
            return_value=_FakeSocket(),
        ) as create_connection:
            records = probe_direct_data_routes(
                state_payload=state_payload,
                local_node_id="node-local",
                policy=self.policy,
            )

        self.assertEqual(len(records), 3)
        endpoint = next(item for item in records if item.sink_node_id == "node-endpoint")
        fallback = next(item for item in records if item.sink_node_id == "node-fallback")
        missing = next(item for item in records if item.sink_node_id == "node-missing")
        self.assertTrue(endpoint.reachable)
        self.assertEqual(endpoint.route_type, "direct_data")
        self.assertEqual(endpoint.endpoint_url, "tcp://26.97.29.153:62002")
        self.assertTrue(fallback.reachable)
        self.assertEqual(fallback.endpoint_url, "tcp://198.51.100.20:52435")
        self.assertFalse(missing.reachable)
        self.assertIn("no candidate", missing.error or "")
        create_connection.assert_any_call(("26.97.29.153", 62002), timeout=1.0)
        create_connection.assert_any_call(("198.51.100.20", 52435), timeout=1.0)

    def test_probe_llama_cpp_rpc_routes_records_protocol_success(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-local": {"workerEnabled": False},
                "node-worker": {
                    "workerEnabled": True,
                    "dataHost": "198.51.100.20",
                    "dataPort": 50052,
                },
            }
        }
        fake_socket = _FakeRpcSocket()

        with patch(
            "cai_compute_chain.route_health.socket.create_connection",
            return_value=fake_socket,
        ) as create_connection:
            records = probe_llama_cpp_rpc_routes(
                state_payload=state_payload,
                local_node_id="node-local",
                policy=self.policy,
            )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertTrue(record.reachable)
        self.assertEqual(record.route_type, "llama_cpp_rpc_direct")
        self.assertEqual(record.endpoint_url, "llama-cpp-rpc://198.51.100.20:50052")
        self.assertIsNone(record.error)
        self.assertEqual(fake_socket.sent[0], 14)
        self.assertEqual(struct.unpack("<Q", fake_socket.sent[1:9])[0], 24)
        create_connection.assert_called_once_with(
            ("198.51.100.20", 50052),
            timeout=1.0,
        )
        self.assertEqual(
            route_health_score_for_path("node-local", ["node-worker"], records),
            (5, 5, 0),
        )

    def test_llama_cpp_rpc_route_health_prefers_direct_over_newer_relay(self) -> None:
        records = [
            RouteHealthRecord(
                route_id="direct-old",
                source_node_id="node-a",
                sink_node_id="node-b",
                route_type="llama_cpp_rpc_direct",
                endpoint_url="llama-cpp-rpc://node-b:50052",
                reachable=True,
                checked_at="2026-05-02T00:00:00+00:00",
                latency_ms=7.0,
            ),
            RouteHealthRecord(
                route_id="relay-new",
                source_node_id="node-a",
                sink_node_id="node-b",
                route_type="llama_cpp_rpc_relay",
                endpoint_url="relay://node-vps/node-b:50052",
                reachable=True,
                checked_at="2026-05-02T00:05:00+00:00",
                latency_ms=12.0,
                transit_node_id="node-vps",
            ),
        ]

        self.assertEqual(
            route_health_score_for_path("node-a", ["node-b"], records),
            (5, 5, 0),
        )
        profile = llama_cpp_compute_cell_profile_for_path("node-a", ["node-b"], records)
        self.assertEqual(
            profile["pairProfiles"][0]["routeType"],
            "llama_cpp_rpc_direct",
        )
        self.assertEqual(profile["maxLatencyMs"], 7.0)

    def test_probe_llama_cpp_rpc_failure_does_not_poison_placement(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-local": {"workerEnabled": False},
                "node-worker": {
                    "workerEnabled": True,
                    "dataHost": "198.51.100.20",
                    "dataPort": 50052,
                },
            }
        }

        with patch(
            "cai_compute_chain.route_health.socket.create_connection",
            side_effect=OSError("refused"),
        ):
            records = probe_llama_cpp_rpc_routes(
                state_payload=state_payload,
                local_node_id="node-local",
                policy=self.policy,
            )

        self.assertEqual(len(records), 1)
        self.assertFalse(records[0].reachable)
        self.assertEqual(records[0].route_type, "llama_cpp_rpc_probe")
        self.assertIn("refused", records[0].error or "")
        self.assertEqual(
            route_health_score_for_path("node-local", ["node-worker"], records),
            (1, 1, 0),
        )

    def test_probe_llama_cpp_rpc_protocol_failure_blocks_placement(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-local": {"workerEnabled": False},
                "node-worker": {
                    "workerEnabled": True,
                    "dataHost": "198.51.100.20",
                    "dataPort": 50052,
                },
            }
        }

        with patch(
            "cai_compute_chain.route_health.socket.create_connection",
            return_value=_FakeRpcSocket(major=0),
        ):
            records = probe_llama_cpp_rpc_routes(
                state_payload=state_payload,
                local_node_id="node-local",
                policy=self.policy,
            )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertFalse(record.reachable)
        self.assertEqual(record.route_type, "llama_cpp_rpc_direct")
        self.assertIn("rpc_protocol_failed", record.error or "")
        self.assertEqual(
            route_health_score_for_path("node-local", ["node-worker"], records),
            (0, 0, -1),
        )

    def test_probe_llama_cpp_rpc_routes_records_relay_protocol_success(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-local": {"workerEnabled": True},
                "node-worker": {
                    "workerEnabled": True,
                    "transportEndpoints": [
                        {
                            "purpose": "data",
                            "routeType": "overlay",
                            "host": "203.0.113.42",
                            "port": 50052,
                        }
                    ],
                },
                "node-relay": {
                    "relayEnabled": True,
                    "apiHost": "198.51.100.10",
                    "apiPort": 52415,
                },
            },
            "overlayPeers": {
                "node-local": ["node-relay"],
                "node-relay": ["node-worker"],
            },
        }
        probe_urls: list[str] = []

        def _urlopen(url: str, timeout: float):
            probe_urls.append(url)
            parsed = urlparse(url)
            query = parse_qs(parsed.query)
            self.assertEqual(query["protocol"], ["llama_cpp_rpc"])
            self.assertEqual(query["source_node_id"], ["node-local"])
            self.assertEqual(query["transit_node_id"], ["node-relay"])
            self.assertEqual(query["sink_node_id"], ["node-worker"])
            self.assertEqual(query["target_host"], ["203.0.113.42"])
            self.assertEqual(query["target_port"], ["50052"])
            self.assertEqual(timeout, 1.0)
            return _FakeHttpResponse(
                {
                    "ready": True,
                    "mode": "direct",
                    "protocolReady": True,
                    "protocolVersion": "3.6.0",
                }
            )

        with (
            patch(
                "cai_compute_chain.route_health.socket.create_connection",
                side_effect=OSError("not directly reachable"),
            ),
            patch("cai_compute_chain.route_health.urlopen", side_effect=_urlopen),
        ):
            records = probe_llama_cpp_rpc_routes(
                state_payload=state_payload,
                local_node_id="node-local",
                policy=self.policy,
            )

        relay_record = next(
            item for item in records if item.route_type == "llama_cpp_rpc_relay"
        )
        self.assertTrue(probe_urls)
        self.assertTrue(relay_record.reachable)
        self.assertEqual(relay_record.transit_node_id, "node-relay")
        self.assertEqual(relay_record.endpoint_url, "relay://node-relay/203.0.113.42:50052")
        self.assertEqual(
            route_health_score_for_path("node-local", ["node-worker"], records),
            (3, 3, 0),
        )

    def test_probe_llama_cpp_rpc_relay_protocol_failure_blocks_placement(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-local": {"workerEnabled": True},
                "node-worker": {
                    "workerEnabled": True,
                    "transportEndpoints": [
                        {
                            "purpose": "data",
                            "routeType": "overlay",
                            "host": "203.0.113.42",
                            "port": 50052,
                        }
                    ],
                },
                "node-relay": {
                    "relayEnabled": True,
                    "apiHost": "198.51.100.10",
                    "apiPort": 52415,
                },
            },
            "overlayPeers": {
                "node-local": ["node-relay"],
                "node-relay": ["node-worker"],
            },
        }

        def _urlopen(url: str, timeout: float):
            return _FakeHttpResponse(
                {
                    "ready": True,
                    "mode": "direct",
                    "protocolReady": False,
                    "error": "malformed HELLO",
                }
            )

        with (
            patch(
                "cai_compute_chain.route_health.socket.create_connection",
                side_effect=OSError("not directly reachable"),
            ),
            patch("cai_compute_chain.route_health.urlopen", side_effect=_urlopen),
        ):
            records = probe_llama_cpp_rpc_routes(
                state_payload=state_payload,
                local_node_id="node-local",
                policy=self.policy,
            )

        relay_record = next(
            item for item in records if item.route_type == "llama_cpp_rpc_relay"
        )
        self.assertFalse(relay_record.reachable)
        self.assertIn("rpc_protocol_failed", relay_record.error or "")
        self.assertEqual(
            route_health_score_for_path("node-local", ["node-worker"], records),
            (0, 0, -1),
        )

    def test_failed_probe_increments_consecutive_failures(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-local": {"apiHost": "127.0.0.1", "apiPort": 52415},
                "node-peer": {"apiHost": "198.51.100.20", "apiPort": 52415},
            }
        }

        with patch(
            "cai_compute_chain.route_health.socket.create_connection",
            side_effect=OSError("refused"),
        ):
            probe_direct_api_routes(
                state_payload=state_payload,
                local_node_id="node-local",
                policy=self.policy,
            )
            probe_direct_api_routes(
                state_payload=state_payload,
                local_node_id="node-local",
                policy=self.policy,
            )

        record = list_route_health_records(self.policy)[0]
        self.assertFalse(record.reachable)
        self.assertEqual(record.consecutive_failures, 2)
        self.assertIn("refused", record.error or "")

    def test_route_health_score_prefers_healthy_path_over_failed_path(self) -> None:
        records = [
            RouteHealthRecord(
                route_id="bad",
                source_node_id="node-a",
                sink_node_id="node-b",
                route_type="direct_data",
                endpoint_url=None,
                reachable=False,
                checked_at="2026-05-02T00:00:00+00:00",
            ),
            RouteHealthRecord(
                route_id="good",
                source_node_id="node-c",
                sink_node_id="node-b",
                route_type="direct_data",
                endpoint_url=None,
                reachable=True,
                checked_at="2026-05-02T00:00:00+00:00",
            ),
        ]

        self.assertEqual(route_health_score_for_path("node-a", ["node-b"], records), (0, 0, -1))
        self.assertEqual(route_health_score_for_path("node-c", ["node-b"], records), (4, 4, 0))

    def test_direct_data_route_health_is_directional_for_nat(self) -> None:
        records = [
            RouteHealthRecord(
                route_id="reverse-ok",
                source_node_id="node-b",
                sink_node_id="node-a",
                route_type="direct_data",
                endpoint_url="tcp://node-a:50052",
                reachable=True,
                checked_at="2026-05-02T00:00:00+00:00",
            ),
            RouteHealthRecord(
                route_id="reverse-bad",
                source_node_id="node-b",
                sink_node_id="node-c",
                route_type="direct_data",
                endpoint_url="tcp://node-c:50052",
                reachable=False,
                checked_at="2026-05-02T00:00:00+00:00",
            ),
        ]

        self.assertEqual(
            route_health_score_for_path("node-a", ["node-b"], records),
            (1, 1, 0),
        )
        self.assertEqual(
            route_health_score_for_path("node-c", ["node-b"], records),
            (1, 1, 0),
        )
        self.assertEqual(
            route_health_score_for_path("node-b", ["node-a"], records),
            (4, 4, 0),
        )

    def test_llama_cpp_rpc_failure_overrides_direct_socket_success(self) -> None:
        checked_at = datetime.now(tz=UTC).isoformat()
        records = [
            RouteHealthRecord(
                route_id="tcp-ok",
                source_node_id="node-a",
                sink_node_id="node-b",
                route_type="direct_data",
                endpoint_url="tcp://node-b:50052",
                reachable=True,
                checked_at="2026-05-02T00:00:00+00:00",
            ),
            RouteHealthRecord(
                route_id="rpc-bad",
                source_node_id="node-a",
                sink_node_id="node-b",
                route_type="llama_cpp_rpc_direct",
                endpoint_url="llama-cpp-rpc://node-b:50052",
                reachable=False,
                checked_at=checked_at,
                error="Remote RPC server crashed or returned malformed response",
            ),
        ]

        self.assertEqual(
            route_health_score_for_path("node-a", ["node-b"], records),
            (0, 0, -1),
        )

    def test_llama_cpp_rpc_failure_backoff_expires_old_failure(self) -> None:
        checked_at = (
            datetime.now(tz=UTC)
            - timedelta(seconds=DEFAULT_BACKOFF_TEST_SECONDS)
        ).isoformat()
        records = [
            RouteHealthRecord(
                route_id="rpc-old",
                source_node_id="node-a",
                sink_node_id="node-b",
                route_type="llama_cpp_rpc_direct",
                endpoint_url="llama-cpp-rpc://node-b:50052",
                reachable=False,
                checked_at=checked_at,
                error="old malformed response",
            )
        ]

        self.assertEqual(
            route_health_score_for_path("node-a", ["node-b"], records),
            (1, 1, 0),
        )

    def test_record_llama_cpp_rpc_result_can_replace_failure_with_success(self) -> None:
        record_llama_cpp_rpc_result(
            source_node_id="node-a",
            sink_node_id="node-b",
            endpoint_url="llama-cpp-rpc://node-b:50052",
            reachable=False,
            error="malformed response",
            policy=self.policy,
        )
        record_llama_cpp_rpc_result(
            source_node_id="node-a",
            sink_node_id="node-b",
            endpoint_url="llama-cpp-rpc://node-b:50052",
            reachable=True,
            policy=self.policy,
        )

        records = list_route_health_records(self.policy)
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0].reachable)
        self.assertEqual(
            route_health_score_for_path("node-a", ["node-b"], records),
            (5, 5, 0),
        )

    def test_compute_cell_profile_accepts_low_latency_rpc_path(self) -> None:
        records = [
            RouteHealthRecord(
                route_id="rpc-ok",
                source_node_id="node-a",
                sink_node_id="node-b",
                route_type="llama_cpp_rpc_direct",
                endpoint_url="llama-cpp-rpc://node-b:50052",
                reachable=True,
                checked_at="2026-05-02T00:00:00+00:00",
                latency_ms=8.5,
            )
        ]

        profile = llama_cpp_compute_cell_profile_for_path(
            "node-a",
            ["node-b"],
            records,
            low_latency_max_ms=20,
            wan_risky_max_ms=60,
        )

        self.assertEqual(profile["profile"], "low_latency_sharded_cell")
        self.assertTrue(profile["readyForLlamaCppRpc"])
        self.assertEqual(profile["maxLatencyMs"], 8.5)

    def test_compute_cell_profile_flags_wan_risky_rpc_path(self) -> None:
        records = [
            RouteHealthRecord(
                route_id="rpc-ok",
                source_node_id="node-a",
                sink_node_id="node-b",
                route_type="llama_cpp_rpc_direct",
                endpoint_url="llama-cpp-rpc://node-b:50052",
                reachable=True,
                checked_at="2026-05-02T00:00:00+00:00",
                latency_ms=48.0,
            )
        ]

        profile = llama_cpp_compute_cell_profile_for_path(
            "node-a",
            ["node-b"],
            records,
            low_latency_max_ms=20,
            wan_risky_max_ms=60,
        )

        self.assertEqual(profile["profile"], "wan_risky_sharded_cell")
        self.assertFalse(profile["readyForLlamaCppRpc"])
        self.assertEqual(profile["pairProfiles"][0]["status"], "wan_risky")

    def test_compute_cell_profile_requires_runtime_probe_when_unproven(self) -> None:
        records = [
            RouteHealthRecord(
                route_id="tcp-ok",
                source_node_id="node-a",
                sink_node_id="node-b",
                route_type="direct_data",
                endpoint_url="tcp://node-b:50052",
                reachable=True,
                checked_at="2026-05-02T00:00:00+00:00",
                latency_ms=4.0,
            )
        ]

        profile = llama_cpp_compute_cell_profile_for_path(
            "node-a",
            ["node-b"],
            records,
        )

        self.assertEqual(profile["profile"], "unproven_sharded_cell")
        self.assertFalse(profile["readyForLlamaCppRpc"])
        self.assertEqual(profile["pairProfiles"][0]["routeScore"], 4)

    def test_record_overlay_routes_from_state(self) -> None:
        records = record_overlay_routes_from_state(
            state_payload={
                "overlayPeers": {
                    "node-a": ["node-b", "node-c"],
                    "node-b": ["node-a"],
                }
            },
            visible_node_ids={"node-a", "node-b"},
            policy=self.policy,
        )

        self.assertEqual(len(records), 2)
        stored = list_route_health_records(self.policy)
        self.assertEqual({item.route_type for item in stored}, {"overlay_peer"})
        self.assertEqual(
            {(item.source_node_id, item.sink_node_id) for item in stored},
            {("node-a", "node-b"), ("node-b", "node-a")},
        )
        self.assertTrue(all(item.reachable for item in stored))

    def test_score_relay_route_candidates_flags_single_transit_bottleneck(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-a": {"workerEnabled": True},
                "node-b": {"workerEnabled": True},
                "node-relay": {"relayEnabled": True},
            },
            "overlayPeers": {
                "node-a": ["node-relay"],
                "node-relay": ["node-b"],
            },
        }
        overlay_records = record_overlay_routes_from_state(
            state_payload=state_payload,
            policy=self.policy,
        )

        score = score_relay_route_candidates(
            state_payload=state_payload,
            route_health_records=overlay_records,
        )

        self.assertEqual(score["participantNodeIds"], ["node-a", "node-b"])
        self.assertEqual(score["candidateCount"], 1)
        self.assertTrue(score["bottleneckRisk"])
        self.assertEqual(score["bottleneckTransitNodeIds"], ["node-relay"])
        candidate = score["candidates"][0]
        self.assertEqual(candidate["transitNodeId"], "node-relay")
        self.assertEqual(candidate["sourceSegmentHealthScore"], 2)
        self.assertEqual(candidate["sinkSegmentHealthScore"], 2)

    def test_score_relay_route_candidates_uses_alternatives_for_bottleneck(self) -> None:
        state_payload = {
            "nodeIdentities": {
                "node-a": {"workerEnabled": True},
                "node-b": {"workerEnabled": True},
                "relay-a": {"relayEnabled": True},
                "relay-b": {"relayEnabled": True},
            },
            "overlayPeers": {
                "node-a": ["relay-a", "relay-b"],
                "node-b": ["relay-a", "relay-b"],
                "relay-a": ["node-a", "node-b"],
                "relay-b": ["node-a", "node-b"],
            },
        }
        overlay_records = record_overlay_routes_from_state(
            state_payload=state_payload,
            policy=self.policy,
        )

        score = score_relay_route_candidates(
            state_payload=state_payload,
            route_health_records=overlay_records,
        )

        self.assertEqual(score["candidateCount"], 4)
        self.assertFalse(score["bottleneckRisk"])
        self.assertEqual(score["transitNodeCounts"], {"relay-a": 2, "relay-b": 2})
        self.assertEqual(score["bottleneckTransitNodeIds"], [])

    def test_record_relay_probe_result_tracks_reverse_availability(self) -> None:
        record = record_relay_probe_result(
            source_node_id="node-a",
            sink_node_id="node-b",
            transit_node_id="node-relay",
            target_host="127.0.0.1",
            target_port=59657,
            ready=True,
            mode="reverse",
            reverse_channels=1,
            policy=self.policy,
        )

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.route_type, "reverse_relay_available")
        self.assertEqual(record.transit_node_id, "node-relay")
        self.assertEqual(record.endpoint_url, "relay://node-relay/127.0.0.1:59657")
        self.assertEqual(
            route_health_score_for_path("node-a", ["node-b"], [record]),
            (3, 3, 0),
        )

    def test_prune_stale_route_health_records_removes_old_records(self) -> None:
        old = RouteHealthRecord(
            route_id="old",
            source_node_id="node-a",
            sink_node_id="node-b",
            route_type="direct_api",
            endpoint_url=None,
            reachable=False,
            checked_at="2020-01-01T00:00:00+00:00",
        )
        fresh = RouteHealthRecord(
            route_id="fresh",
            source_node_id="node-a",
            sink_node_id="node-c",
            route_type="direct_api",
            endpoint_url=None,
            reachable=True,
            checked_at="2999-01-01T00:00:00+00:00",
        )
        save_route_health_records([old, fresh], self.policy)

        pruned = prune_stale_route_health_records(
            max_age_seconds=60,
            policy=self.policy,
        )

        self.assertEqual(pruned, 1)
        stored = list_route_health_records(self.policy)
        self.assertEqual([item.route_id for item in stored], ["fresh"])


if __name__ == "__main__":
    unittest.main()

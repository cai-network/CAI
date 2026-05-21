# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import unittest

from cai_compute_chain import cai_owned_transport_peer_urls as peer_urls
from cai_compute_chain import cai_owned_transport_common as common
from cai_compute_chain import cai_owned_transport_execution_plan as execution_plan
from cai_compute_chain import cai_owned_transport_ids as transport_ids
from cai_compute_chain import cai_owned_transport_payload_codec as payload_codec
from cai_compute_chain import cai_owned_transport_protocol as protocol
from cai_compute_chain import cai_owned_transport_storage as storage
from cai_compute_chain import cai_owned_transport_auth as transport_auth
from cai_compute_chain import decentralized_compute
from cai_compute_chain.model import ChainNetwork, WalletPolicy


class CaiOwnedTransportProtocolTests(unittest.TestCase):
    def test_protocol_constants_are_shared_by_compatibility_modules(self) -> None:
        self.assertEqual(
            decentralized_compute.CAI_OWNED_TRANSPORT_PROTOCOL,
            protocol.CAI_OWNED_TRANSPORT_PROTOCOL,
        )
        self.assertEqual(
            storage.CAI_OWNED_TRANSPORT_PROTOCOL,
            protocol.CAI_OWNED_TRANSPORT_PROTOCOL,
        )
        self.assertEqual(
            peer_urls.CAI_OWNED_TRANSPORT_OVERLAY_URL_PREFIX,
            protocol.CAI_OWNED_TRANSPORT_OVERLAY_URL_PREFIX,
        )
        self.assertIs(
            decentralized_compute._parse_cai_owned_transport_overlay_url,
            peer_urls.parse_cai_owned_transport_overlay_url,
        )
        self.assertEqual(
            decentralized_compute.EXECUTION_MODE_CAI_OWNED_TRANSPORT_REQUIRED,
            protocol.EXECUTION_MODE_CAI_OWNED_TRANSPORT_REQUIRED,
        )
        self.assertIs(
            decentralized_compute.sign_cai_owned_transport_session_offer,
            transport_auth.sign_cai_owned_transport_session_offer,
        )
        self.assertIs(
            decentralized_compute.validate_cai_owned_transport_payload_signature,
            transport_auth.validate_cai_owned_transport_payload_signature,
        )
        self.assertIs(
            decentralized_compute.cai_owned_transport_auth_headers,
            transport_auth.cai_owned_transport_auth_headers,
        )
        self.assertIs(
            decentralized_compute.validate_cai_owned_transport_request_auth,
            transport_auth.validate_cai_owned_transport_request_auth,
        )
        self.assertIs(
            decentralized_compute.validate_cai_owned_transport_local_runtime_auth,
            transport_auth.validate_cai_owned_transport_local_runtime_auth,
        )
        self.assertIs(
            decentralized_compute._decode_cai_owned_transport_batch_payload,
            payload_codec.decode_cai_owned_transport_batch_payload,
        )
        self.assertIs(
            decentralized_compute._cai_owned_transport_batch_id,
            transport_ids.cai_owned_transport_batch_id,
        )
        self.assertIs(
            decentralized_compute._cai_owned_transport_stage_id,
            transport_ids.cai_owned_transport_stage_id,
        )
        self.assertIs(
            decentralized_compute._normalize_cai_owned_transport_shard_ranges,
            execution_plan.normalize_cai_owned_transport_shard_ranges,
        )
        self.assertIs(
            decentralized_compute._cai_owned_transport_output_route_plan_from_dag,
            execution_plan.cai_owned_transport_output_route_plan_from_dag,
        )
        self.assertIs(
            decentralized_compute._execution_mode_for_compute_cell,
            execution_plan.execution_mode_for_compute_cell,
        )
        self.assertIs(
            decentralized_compute._clean_sink_node_ids,
            execution_plan.clean_sink_node_ids,
        )

    def test_common_helpers_match_legacy_transport_expectations(self) -> None:
        self.assertEqual(
            common.clean_node_ids([" node-a ", "", "node-a", "node-b"]),
            ["node-a", "node-b"],
        )
        self.assertEqual(
            common.cai_owned_transport_chain_id(chain_id=" MAINNET "),
            "mainnet",
        )
        self.assertEqual(
            common.cai_owned_transport_chain_id(
                WalletPolicy(chain_network=ChainNetwork.TESTNET)
            ),
            "testnet",
        )

        parsed = common.parse_cai_owned_transport_datetime("2026-05-20T12:00:00")
        self.assertIsNotNone(parsed)
        self.assertIsNotNone(parsed.tzinfo)

        self.assertTrue(
            common.is_safe_transport_file_id("caibatch_abc-123", prefix="caibatch_")
        )
        self.assertFalse(
            common.is_safe_transport_file_id("../caibatch_abc", prefix="caibatch_")
        )
        self.assertEqual(
            common.require_safe_transport_file_id(
                " caistage_stage-1 ",
                prefix="caistage_",
            ),
            "caistage_stage-1",
        )

    def test_peer_url_helpers_parse_overlay_targets(self) -> None:
        self.assertEqual(
            peer_urls.parse_cai_owned_transport_overlay_url(
                "cai-overlay:https://relay.example:52415/path?targetNodeId=node-b"
            ),
            ("https://relay.example:52415/path", "node-b"),
        )
        self.assertEqual(
            peer_urls.parse_cai_owned_transport_overlay_url(
                "http://direct.example:52415"
            ),
            None,
        )
        with self.assertRaisesRegex(ValueError, "targetNodeId"):
            peer_urls.parse_cai_owned_transport_overlay_url(
                "cai-overlay:https://relay.example:52415"
            )

    def test_execution_plan_helpers_preserve_shard_ranges(self) -> None:
        self.assertEqual(
            execution_plan.clean_sink_node_ids(
                "node-a",
                ["node-a", "node-b", "node-b", "", "node-c"],
            ),
            ["node-b", "node-c"],
        )
        self.assertEqual(
            execution_plan.execution_mode_for_compute_cell(
                {"profile": "single_node", "readyForLlamaCppRpc": False}
            ),
            protocol.EXECUTION_MODE_SINGLE_NODE,
        )
        self.assertEqual(
            execution_plan.execution_mode_for_compute_cell(
                {"profile": "low_latency_sharded_cell", "readyForLlamaCppRpc": True}
            ),
            protocol.EXECUTION_MODE_LLAMA_CPP_RPC_LOW_LATENCY,
        )
        self.assertEqual(
            execution_plan.execution_mode_for_compute_cell(
                {
                    "profile": "proven_unknown_latency_sharded_cell",
                    "readyForLlamaCppRpc": True,
                }
            ),
            protocol.EXECUTION_MODE_LLAMA_CPP_RPC_PROVEN_UNKNOWN_LATENCY,
        )
        self.assertEqual(
            execution_plan.execution_mode_for_compute_cell(
                {"profile": "wan_risky_sharded_cell", "reason": "WAN path."}
            ),
            protocol.EXECUTION_MODE_CAI_OWNED_TRANSPORT_REQUIRED,
        )
        self.assertIn(
            "WAN path.",
            execution_plan.execution_reason(
                protocol.EXECUTION_MODE_CAI_OWNED_TRANSPORT_REQUIRED,
                {"reason": "WAN path."},
            ),
        )
        self.assertEqual(
            execution_plan.normalize_cai_owned_transport_shard_ranges(
                ["node-a", "node-b", "node-c"],
                10,
            ),
            [
                {
                    "nodeId": "node-a",
                    "layerStart": 0,
                    "layerEnd": 4,
                    "layerCount": 4,
                },
                {
                    "nodeId": "node-b",
                    "layerStart": 4,
                    "layerEnd": 7,
                    "layerCount": 3,
                },
                {
                    "nodeId": "node-c",
                    "layerStart": 7,
                    "layerEnd": 10,
                    "layerCount": 3,
                },
            ],
        )
        with self.assertRaisesRegex(ValueError, "must be contiguous"):
            execution_plan.normalize_cai_owned_transport_shard_ranges(
                ["node-a", "node-b"],
                8,
                shard_ranges=[
                    {"nodeId": "node-a", "layerStart": 0, "layerEnd": 4},
                    {"nodeId": "node-b", "layerStart": 5, "layerEnd": 8},
                ],
            )
        self.assertEqual(
            execution_plan.cai_owned_transport_output_route_plan_from_dag(
                {
                    "stages": [
                        {
                            "stageId": "stage-1",
                            "sinkNodeId": "node-a",
                            "phase": "prefill_activation_batches",
                            "sequence": 0,
                            "executorNodeId": "node-a",
                            "layerStart": 0,
                            "layerEnd": 4,
                        },
                        {
                            "stageId": "stage-2",
                            "sinkNodeId": "node-b",
                            "phase": "decode_activation_batches",
                            "sequence": 1,
                            "executorNodeId": "node-b",
                            "layerStart": 4,
                            "layerEnd": 8,
                        },
                    ]
                },
                requester_node_id="requester",
            ),
            [
                {
                    "sinkNodeId": "node-b",
                    "phase": "decode_activation_batches",
                    "sequence": 1,
                    "stageId": "stage-2",
                    "executorNodeId": "node-b",
                    "layerStart": 4,
                    "layerEnd": 8,
                },
                {
                    "sinkNodeId": "requester",
                    "phase": "decode_activation_batches",
                    "sequence": 2,
                    "stageId": "final_result",
                    "finalOutput": True,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()

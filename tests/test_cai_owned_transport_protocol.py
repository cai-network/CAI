# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from datetime import UTC, datetime, timedelta
import unittest

from cai_compute_chain import cai_owned_transport_peer_urls as peer_urls
from cai_compute_chain import cai_owned_transport_common as common
from cai_compute_chain import cai_owned_llm_runtime_metadata as llm_runtime_metadata
from cai_compute_chain import cai_owned_transport_batch_lifecycle as batch_lifecycle
from cai_compute_chain import cai_owned_transport_execution_plan as execution_plan
from cai_compute_chain import cai_owned_transport_ids as transport_ids
from cai_compute_chain import cai_owned_transport_payload_codec as payload_codec
from cai_compute_chain import cai_owned_transport_protocol as protocol
from cai_compute_chain import cai_owned_transport_receipts as transport_receipts
from cai_compute_chain import cai_owned_transport_storage as storage
from cai_compute_chain import cai_owned_transport_auth as transport_auth
from cai_compute_chain import cai_owned_transport_versioning as versioning
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
            decentralized_compute._validate_cai_owned_transport_chain_id,
            common.validate_cai_owned_transport_chain_id,
        )
        self.assertIs(
            decentralized_compute._validate_cai_owned_transport_created_at,
            common.validate_cai_owned_transport_created_at,
        )
        self.assertIs(
            decentralized_compute._jsonable_dict,
            common.jsonable_dict,
        )
        self.assertIs(
            decentralized_compute._optional_int,
            common.optional_int,
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
        self.assertIs(
            decentralized_compute._cai_owned_transport_frame_kind_for_phase,
            execution_plan.cai_owned_transport_frame_kind_for_phase,
        )
        self.assertIs(
            decentralized_compute._runtime_metadata_text,
            llm_runtime_metadata.runtime_metadata_text,
        )
        self.assertIs(
            decentralized_compute._cai_owned_transport_llm_runtime_metadata,
            llm_runtime_metadata.cai_owned_transport_llm_runtime_metadata,
        )
        self.assertIs(
            decentralized_compute._require_runtime_metadata_layer_range_supported,
            llm_runtime_metadata.require_runtime_metadata_layer_range_supported,
        )
        self.assertIs(
            decentralized_compute._clean_cai_owned_transport_receipt_batch_ids,
            transport_receipts.clean_cai_owned_transport_receipt_batch_ids,
        )
        self.assertIs(
            decentralized_compute._cai_owned_transport_proof_batch_ids,
            transport_receipts.cai_owned_transport_proof_batch_ids,
        )
        self.assertIs(
            decentralized_compute._append_unique_metric,
            transport_receipts.append_unique_metric,
        )
        self.assertIs(
            decentralized_compute._apply_cai_owned_transport_batch_lease,
            batch_lifecycle.apply_cai_owned_transport_batch_lease,
        )
        self.assertIs(
            decentralized_compute._clear_cai_owned_transport_batch_runtime_claim,
            batch_lifecycle.clear_cai_owned_transport_batch_runtime_claim,
        )
        self.assertIs(
            decentralized_compute.cai_owned_transport_version_compatibility,
            versioning.cai_owned_transport_version_compatibility,
        )
        self.assertIs(
            decentralized_compute._cai_owned_transport_version_label,
            versioning.cai_owned_transport_version_label,
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
        self.assertEqual(
            common.cai_owned_transport_payload_chain_id(
                {"network": " MAINNET "}
            ),
            "mainnet",
        )
        self.assertEqual(
            common.validate_cai_owned_transport_chain_id(
                {"chainId": "mainnet", "network": "mainnet"},
                expected_chain_id="mainnet",
                payload_name="test payload",
            ),
            (True, None, "mainnet"),
        )
        self.assertEqual(
            common.normalize_sha256_hex("A" * 64, field_name="hash"),
            "a" * 64,
        )
        self.assertIsNone(common.optional_sha256_hex(None, field_name="hash"))
        self.assertEqual(common.optional_int("42"), 42)
        self.assertEqual(
            common.jsonable_dict({"value": 7}, field_name="payload"),
            {"value": 7},
        )
        now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
        self.assertEqual(
            common.validate_cai_owned_transport_created_at(
                {"createdAt": now.isoformat()},
                payload_name="test payload",
                max_age_seconds=60,
                now=now,
            ),
            (True, None),
        )
        self.assertEqual(
            common.validate_cai_owned_transport_created_at(
                {"createdAt": (now - timedelta(seconds=61)).isoformat()},
                payload_name="test payload",
                max_age_seconds=60,
                now=now,
            ),
            (False, "CAI-owned transport test payload has expired."),
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
        self.assertEqual(
            execution_plan.cai_owned_transport_frame_kind_for_phase(
                "decode_activation_batches"
            ),
            "decode",
        )
        self.assertEqual(
            execution_plan.cai_owned_transport_frame_kind_for_phase(
                "prefill_activation_batches"
            ),
            "activation",
        )
        self.assertEqual(
            execution_plan.cai_owned_transport_template_token_start(
                "decode_activation_batches",
                5,
            ),
            5,
        )
        self.assertEqual(
            execution_plan.cai_owned_transport_template_token_end(
                "decode_activation_batches",
                5,
            ),
            6,
        )
        self.assertEqual(
            execution_plan.cai_owned_transport_template_token_end(
                "prefill_activation_batches",
                5,
            ),
            5,
        )

    def test_llm_runtime_metadata_helpers_preserve_descriptor_fields(self) -> None:
        metadata = {
            "model_id": "model-a",
            "total_layers": "12",
            "activation_shape": ["1", "3", "768"],
            "layerRangeSupported": "true",
            "shardCompatibility": "layer_range_supported",
            "preferredFilename": "model.gguf",
            "context_length": "2048",
            "kvCache": {"format": "caikv-v1"},
        }

        self.assertEqual(
            llm_runtime_metadata.runtime_metadata_text(
                metadata,
                "modelId",
                "model_id",
            ),
            "model-a",
        )
        self.assertEqual(
            llm_runtime_metadata.runtime_metadata_int(
                metadata,
                "totalLayerCount",
                "total_layers",
            ),
            12,
        )
        self.assertIs(
            llm_runtime_metadata.runtime_metadata_bool(
                metadata,
                "layerRangeSupported",
            ),
            True,
        )
        self.assertEqual(
            llm_runtime_metadata.runtime_metadata_shape(metadata),
            [1, 3, 768],
        )
        self.assertEqual(
            llm_runtime_metadata.runtime_metadata_mapping(metadata, "kvCache"),
            {"format": "caikv-v1"},
        )
        self.assertEqual(
            llm_runtime_metadata.runtime_metadata_external_shard_descriptor(metadata),
            {
                "preferredFilename": "model.gguf",
                "shardCompatibility": "layer_range_supported",
                "layerRangeSupported": True,
                "contextLength": 2048,
            },
        )

        llm_runtime_metadata.require_runtime_metadata_layer_range_supported(
            metadata,
            model_id="model-a",
        )
        with self.assertRaisesRegex(ValueError, "layerRangeSupported is false"):
            llm_runtime_metadata.require_runtime_metadata_layer_range_supported(
                {"layerRangeSupported": False},
                model_id="model-a",
            )
        self.assertEqual(
            llm_runtime_metadata.cai_owned_transport_llm_runtime_metadata(
                {"backend": "llama.cpp"},
                model_id="model-a",
                total_layer_count=12,
                tokenizer_config_hash="b" * 64,
            ),
            {
                "backend": "llama.cpp",
                "modelId": "model-a",
                "totalLayerCount": 12,
                "tokenizerConfigHash": "b" * 64,
            },
        )

    def test_receipt_helpers_clean_and_validate_proof_batch_ids(self) -> None:
        proof = {
            "shardReceipts": [
                {"batchIds": ["caibatch_alpha", "caibatch_alpha"]},
                {"batchIds": ["caibatch_beta"]},
            ]
        }
        batch_ids, errors = transport_receipts.cai_owned_transport_proof_batch_ids(
            proof,
        )

        self.assertEqual(batch_ids, {"caibatch_alpha", "caibatch_beta"})
        self.assertEqual(
            errors,
            ["CAI-owned transport proof duplicates batch id 'caibatch_alpha'."],
        )
        self.assertEqual(
            transport_receipts.clean_cai_owned_transport_receipt_stage_ids(
                ["caistage_1", "caistage_1", "", "caistage_2"],
            ),
            ["caistage_1", "caistage_2"],
        )
        self.assertEqual(
            transport_receipts.clean_cai_owned_transport_receipt_sequences(
                [0, "1", "-2", "bad", 1],
            ),
            [0, 1],
        )
        self.assertEqual(
            transport_receipts.clean_cai_owned_transport_receipt_hashes(
                ["A" * 64, "a" * 64],
                field_name="hashes",
            ),
            ["a" * 64],
        )
        self.assertEqual(
            transport_receipts.clean_cai_owned_transport_receipt_audits(
                [{"route": "direct"}],
            ),
            [{"route": "direct"}],
        )
        self.assertEqual(
            transport_receipts.max_receipt_count(
                [
                    {"activationBatchCount": "2"},
                    {"activationBatchCount": 5},
                    {"activationBatchCount": "bad"},
                ],
                "activationBatchCount",
            ),
            5,
        )
        values: list[str] = []
        transport_receipts.append_unique(values, " node-a ")
        transport_receipts.append_unique(values, "node-a")
        self.assertEqual(values, ["node-a"])

        metrics: dict[str, object] = {}
        transport_receipts.append_unique_metric(metrics, "adapterIds", "adapter-a")
        transport_receipts.append_unique_metric(metrics, "adapterIds", "adapter-a")
        self.assertEqual(metrics, {"adapterIds": ["adapter-a"]})
        self.assertEqual(
            transport_receipts.first_metric_value(
                {"inputTokens": 7, "inputTokenCount": 9},
                ("missing", "inputTokens", "inputTokenCount"),
            ),
            7,
        )

    def test_batch_lifecycle_helpers_preserve_claim_and_timeout_state(self) -> None:
        now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
        batch = {
            "runtimeId": "runtime-a",
            "heartbeatAt": "old",
            "leaseExpiresAt": "old",
            "leaseSeconds": 1,
            "claimedByNodeId": "node-a",
        }

        batch_lifecycle.apply_cai_owned_transport_batch_lease(
            batch,
            now,
            lease_seconds=30,
        )
        self.assertEqual(batch["heartbeatAt"], now.isoformat())
        self.assertEqual(batch["leaseSeconds"], 30.0)
        self.assertFalse(
            batch_lifecycle.cai_owned_transport_batch_lease_expired(
                batch,
                now + timedelta(seconds=29),
            )
        )
        self.assertTrue(
            batch_lifecycle.cai_owned_transport_batch_lease_expired(
                batch,
                now + timedelta(seconds=30),
            )
        )
        self.assertEqual(
            batch_lifecycle.cai_owned_transport_batch_attempt_count(
                {"attemptCount": "3"}
            ),
            3,
        )
        self.assertTrue(
            batch_lifecycle.cai_owned_transport_batch_claim_expired(
                {"updatedAt": (now - timedelta(seconds=31)).isoformat()},
                now,
                timeout_seconds=30,
            )
        )

        batch_lifecycle.mark_cai_owned_transport_batch_timed_out(
            batch,
            now,
            error="lease expired",
            reason="lease_timeout",
        )
        self.assertEqual(batch["status"], "timed_out")
        self.assertEqual(batch["timeoutReason"], "lease_timeout")
        self.assertFalse(batch["retryable"])

        batch_lifecycle.clear_cai_owned_transport_batch_runtime_claim(batch)
        self.assertEqual(batch["previousRuntimeId"], "runtime-a")
        for key in (
            "runtimeId",
            "heartbeatAt",
            "leaseExpiresAt",
            "leaseSeconds",
            "claimedByNodeId",
        ):
            self.assertNotIn(key, batch)

    def test_versioning_helpers_validate_protocol_and_runtime_fields(self) -> None:
        compatible = versioning.cai_owned_transport_version_compatibility(
            {
                "protocol": protocol.CAI_OWNED_TRANSPORT_PROTOCOL,
                "protocolVersion": protocol.CAI_OWNED_TRANSPORT_PROTOCOL_VERSION,
                "compatibleProtocolVersions": [
                    protocol.CAI_OWNED_TRANSPORT_PROTOCOL_VERSION
                ],
                "runtimeVersion": "cai-owned-runtime/0.1",
                "adapterId": "llama.cpp",
                "adapterVersion": "0.1.0",
            },
            require_runtime_versions=True,
        )
        self.assertTrue(compatible["compatible"])
        self.assertEqual(compatible["errors"], [])
        self.assertEqual(
            versioning.cai_owned_transport_int_list([1, "1", "bad", 2]),
            [1, 2],
        )
        self.assertEqual(
            versioning.cai_owned_transport_version_label(" runtime/0.1+gpu "),
            "runtime/0.1+gpu",
        )

        incompatible = versioning.cai_owned_transport_version_compatibility(
            {
                "protocol": "other",
                "protocolVersion": 999,
                "runtimeVersion": "bad version",
                "adapterId": "adapter",
                "adapterVersion": "",
            },
            require_runtime_versions=True,
        )
        self.assertFalse(incompatible["compatible"])
        self.assertIn(
            "CAI-owned transport protocol is incompatible.",
            incompatible["errors"],
        )
        self.assertIn(
            "CAI-owned transport protocol version is unsupported.",
            incompatible["errors"],
        )
        self.assertIn(
            "CAI-owned transport runtime version is invalid.",
            incompatible["errors"],
        )
        self.assertIn(
            "CAI-owned transport adapter version is missing.",
            incompatible["errors"],
        )


if __name__ == "__main__":
    unittest.main()

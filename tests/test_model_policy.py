# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import unittest

from cai_compute_chain.model import (
    ChainNetwork,
    DEFAULT_PRIVATE_EXECUTION_MODEL_ID,
    DEFAULT_PRIVATE_NETWORK_MODEL_ID,
    LEGACY_PRIVATE_NETWORK_MODEL_ID,
    NetworkModelPolicy,
    WalletPolicy,
    curated_model_for_id,
    curated_model_registry,
    curated_network_model_ids,
    curated_private_model_ids,
    effective_private_worker_shard_minimum,
    is_private_curated_model_id,
    is_registered_curated_model_id,
    normalize_network_model_id,
    resolve_execution_model_id,
)


class NetworkModelPolicyTests(unittest.TestCase):
    def test_curated_registry_contains_cai_network_private_model(self) -> None:
        registry = curated_model_registry()

        model = next(
            item for item in registry if item.model_id == DEFAULT_PRIVATE_NETWORK_MODEL_ID
        )
        self.assertEqual(model.model_id, DEFAULT_PRIVATE_NETWORK_MODEL_ID)
        self.assertEqual(model.execution_model_id, DEFAULT_PRIVATE_EXECUTION_MODEL_ID)
        self.assertEqual(model.source_repo_id, DEFAULT_PRIVATE_NETWORK_MODEL_ID)
        self.assertFalse(model.allow_single_node_fallback)
        self.assertEqual(model.minimum_worker_shards, 2)
        self.assertEqual(model.model_format, "gguf")
        self.assertEqual(model.gguf_architecture, "qwen3")
        self.assertEqual(model.shard_compatibility, "layer_range_supported")
        self.assertTrue(model.layer_range_supported)
        self.assertEqual(model.layer_range_probe_abi, "cai-layer-range-v1")
        self.assertIn("qwen3-production-binary-conformance", model.layer_range_probe_report or "")
        self.assertIn(
            "qwen3-layer-range-equivalence-probe",
            model.layer_range_equivalence_probe_report or "",
        )
        self.assertEqual(
            model.activation_state_format,
            "ggml-tensor-v1/layer-range-activation-v1",
        )
        self.assertEqual(
            model.decode_state_format,
            "ggml-kv-cache-v1/token-step-kv-cache-v1",
        )
        self.assertNotIn("qwen3-0.6b-4bit", model.model_id.lower())

    def test_default_policy_is_derived_from_curated_registry(self) -> None:
        policy = NetworkModelPolicy()

        self.assertEqual(policy.private_curated_model_ids, curated_private_model_ids())
        self.assertFalse(policy.private_model_allows_full_single_worker_copy)
        self.assertTrue(policy.private_model_requires_sharded_distribution)
        self.assertEqual(policy.minimum_worker_shards, 2)
        self.assertTrue(policy.allow_single_node_private_inference)
        self.assertIn(DEFAULT_PRIVATE_NETWORK_MODEL_ID, curated_network_model_ids())
        self.assertEqual(curated_private_model_ids(), (DEFAULT_PRIVATE_NETWORK_MODEL_ID,))
        self.assertIn(DEFAULT_PRIVATE_NETWORK_MODEL_ID, policy.private_runtime_model_ids)
        self.assertEqual(policy.network_default_execution_model_id, DEFAULT_PRIVATE_EXECUTION_MODEL_ID)
        self.assertNotIn("Qwen/Qwen3-0.6B-GGUF", policy.private_runtime_model_ids)

    def test_effective_private_worker_minimum_allows_single_available_worker(self) -> None:
        self.assertEqual(
            effective_private_worker_shard_minimum(available_worker_count=1),
            1,
        )
        self.assertEqual(
            effective_private_worker_shard_minimum(available_worker_count=3),
            1,
        )

    def test_curated_model_lookup_accepts_network_and_execution_ids(self) -> None:
        network_model = curated_model_for_id(DEFAULT_PRIVATE_NETWORK_MODEL_ID)
        execution_model = curated_model_for_id(DEFAULT_PRIVATE_EXECUTION_MODEL_ID)

        self.assertIsNotNone(network_model)
        self.assertIsNotNone(execution_model)
        self.assertEqual(network_model.execution_model_id, DEFAULT_PRIVATE_EXECUTION_MODEL_ID)
        self.assertEqual(execution_model.model_id, DEFAULT_PRIVATE_EXECUTION_MODEL_ID)
        self.assertTrue(is_registered_curated_model_id(DEFAULT_PRIVATE_NETWORK_MODEL_ID))
        self.assertTrue(is_registered_curated_model_id(DEFAULT_PRIVATE_EXECUTION_MODEL_ID))

    def test_legacy_private_model_id_normalizes_but_is_not_registered(self) -> None:
        self.assertEqual(
            normalize_network_model_id(LEGACY_PRIVATE_NETWORK_MODEL_ID),
            DEFAULT_PRIVATE_NETWORK_MODEL_ID,
        )
        self.assertFalse(is_registered_curated_model_id(LEGACY_PRIVATE_NETWORK_MODEL_ID))

    def test_execution_resolution_uses_registry_for_network_id(self) -> None:
        self.assertEqual(
            resolve_execution_model_id(DEFAULT_PRIVATE_NETWORK_MODEL_ID),
            DEFAULT_PRIVATE_EXECUTION_MODEL_ID,
        )
        self.assertEqual(
            resolve_execution_model_id(DEFAULT_PRIVATE_EXECUTION_MODEL_ID),
            DEFAULT_PRIVATE_EXECUTION_MODEL_ID,
        )
        self.assertTrue(is_private_curated_model_id(DEFAULT_PRIVATE_NETWORK_MODEL_ID))

    def test_wallet_policy_pq_strictness_follows_explicit_chain_network(self) -> None:
        mainnet_policy = WalletPolicy(chain_network=ChainNetwork.MAINNET)
        testnet_policy = WalletPolicy(chain_network=ChainNetwork.TESTNET)

        self.assertTrue(mainnet_policy.require_post_quantum_wallet_signatures)
        self.assertFalse(testnet_policy.require_post_quantum_wallet_signatures)


if __name__ == "__main__":
    unittest.main()

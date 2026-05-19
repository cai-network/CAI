# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from cai_compute_chain.gguf_shard_policy import (
    GGUF_GEMMA_2B_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT,
    GGUF_GEMMA_2B_LAYER_RANGE_PROBE_REPORT,
    GGUF_GEMMA2_2B_IT_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT,
    GGUF_GEMMA2_2B_IT_LAYER_RANGE_PROBE_REPORT,
    GGUF_GEMMA3_1B_IT_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT,
    GGUF_GEMMA3_1B_IT_LAYER_RANGE_PROBE_REPORT,
    GGUF_GPTNEOX_PYTHIA_14M_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT,
    GGUF_GPTNEOX_PYTHIA_14M_LAYER_RANGE_PROBE_REPORT,
    GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS,
    GGUF_LLAMA_TINYLLAMA_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT,
    GGUF_LLAMA_TINYLLAMA_LAYER_RANGE_PROBE_REPORT,
    GGUF_MISTRAL3_MINISTRAL_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT,
    GGUF_MISTRAL3_MINISTRAL_LAYER_RANGE_PROBE_REPORT,
    GGUF_PHI2_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT,
    GGUF_PHI2_LAYER_RANGE_PROBE_REPORT,
    GGUF_PHI3_MINI_4K_INSTRUCT_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT,
    GGUF_PHI3_MINI_4K_INSTRUCT_LAYER_RANGE_PROBE_REPORT,
    GGUF_QWEN2_5_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT,
    GGUF_QWEN2_5_LAYER_RANGE_PROBE_REPORT,
    GGUF_QWEN_1_8B_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT,
    GGUF_QWEN_1_8B_LAYER_RANGE_PROBE_REPORT,
    GGUF_QWEN3_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT,
    GGUF_QWEN3_LAYER_RANGE_PROBE_REPORT,
    GGUF_SHARD_MODE_FULL_MODEL_LOCAL,
    GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED,
    GGUF_SHARD_MODE_UNSUPPORTED_FOR_SHARDING,
    gguf_layer_range_architecture_proof,
    gguf_shard_compatibility,
    load_gguf_layer_range_conformance_report,
    load_gguf_layer_range_equivalence_probe_report,
    normalize_gguf_architecture,
    validate_gguf_layer_range_architecture_proof,
    validate_gguf_layer_range_conformance_report,
    validate_gguf_layer_range_equivalence_probe_report,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class GgufShardPolicyTests(unittest.TestCase):
    def test_qwen3_is_a_checked_layer_range_architecture(self) -> None:
        compatibility = gguf_shard_compatibility(
            model_id="cai-network/Qwen3-0.6B-GGUF",
            gguf_architecture="qwen",
            filename="Qwen3-0.6B-Q8_0.gguf",
        )

        self.assertEqual(compatibility.gguf_architecture, "qwen3")
        self.assertEqual(
            compatibility.shard_compatibility,
            GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED,
        )
        self.assertTrue(compatibility.layer_range_supported)
        self.assertEqual(compatibility.layer_range_probe_abi, "cai-layer-range-v1")
        self.assertIn("qwen3-production-binary-conformance", compatibility.layer_range_probe_report or "")
        self.assertIn(
            "qwen3-layer-range-equivalence-probe",
            compatibility.layer_range_equivalence_probe_report or "",
        )
        self.assertIn(
            "checked CAI equivalence and production probes",
            compatibility.reason,
        )

    def test_qwen3_layer_range_support_is_registered_as_a_proof_bundle(self) -> None:
        proof = gguf_layer_range_architecture_proof("qwen3")

        self.assertIsNotNone(proof)
        assert proof is not None
        self.assertEqual(proof.reference_model_id, "cai-network/Qwen3-0.6B-GGUF")
        self.assertIn("qwen3", GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS)
        self.assertIn(
            "qwen3-layer-range-equivalence-probe",
            proof.equivalence_probe_report,
        )
        self.assertIn(
            "qwen3-production-binary-conformance",
            proof.conformance_probe_report,
        )

    def test_qwen3_coder_uses_explicit_qwen3_gguf_architecture(self) -> None:
        compatibility = gguf_shard_compatibility(
            model_id="Qwen/Qwen3-Coder-0.6B-GGUF",
            gguf_architecture="qwen3",
            filename="qwen3-coder-0.6b-q8_0.gguf",
        )

        self.assertEqual(compatibility.gguf_architecture, "qwen3")
        self.assertEqual(
            compatibility.shard_compatibility,
            GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED,
        )
        self.assertTrue(compatibility.layer_range_supported)

    def test_qwen_family_variants_are_classified_without_false_whitelisting(self) -> None:
        self.assertEqual(
            normalize_gguf_architecture(model_id="Qwen/Qwen3-Coder-7B-GGUF"),
            "qwen3",
        )
        self.assertEqual(
            normalize_gguf_architecture(model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF"),
            "qwen2",
        )
        self.assertEqual(
            normalize_gguf_architecture(model_id="Qwen/Qwen1.5-0.5B-Chat-GGUF"),
            "qwen2",
        )
        self.assertEqual(
            normalize_gguf_architecture(model_id="Qwen/Qwen3.5-2B-GGUF"),
            "qwen35",
        )
        self.assertEqual(
            normalize_gguf_architecture(model_id="Qwen/Qwen3.5-35B-A3B-GGUF"),
            "moe",
        )
        self.assertEqual(
            normalize_gguf_architecture(architecture="qwen3_5_moe"),
            "moe",
        )
        self.assertEqual(
            normalize_gguf_architecture(model_id="Qwen/Qwen3-Next-80B-A3B-GGUF"),
            "qwen3next",
        )
        self.assertEqual(
            normalize_gguf_architecture(model_id="mradermacher/Qwen-1_8B-Chat-GGUF"),
            "qwen",
        )

    def test_mistral_family_variants_are_classified_without_false_whitelisting(self) -> None:
        self.assertEqual(
            normalize_gguf_architecture(model_id="TheBloke/Mistral-7B-GGUF"),
            "mistral",
        )
        self.assertEqual(
            normalize_gguf_architecture(architecture="mistral3"),
            "mistral3",
        )
        self.assertEqual(
            normalize_gguf_architecture(architecture="mistral4"),
            "mistral4",
        )

        for architecture in ("mistral", "mistral4"):
            with self.subTest(architecture=architecture):
                compatibility = gguf_shard_compatibility(
                    model_id=f"mistralai/{architecture}-candidate-GGUF",
                    gguf_architecture=architecture,
                    filename=f"{architecture}-candidate-q4_k_m.gguf",
                )

                self.assertEqual(compatibility.gguf_architecture, architecture)
                self.assertEqual(
                    compatibility.shard_compatibility,
                    GGUF_SHARD_MODE_UNSUPPORTED_FOR_SHARDING,
                )
                self.assertFalse(compatibility.layer_range_supported)

    def test_gemma_family_variants_are_classified_without_false_whitelisting(self) -> None:
        self.assertEqual(
            normalize_gguf_architecture(model_id="tensorblock/gemma-2b-GGUF"),
            "gemma",
        )
        self.assertEqual(
            normalize_gguf_architecture(model_id="google/gemma-2-2b-it-GGUF"),
            "gemma2",
        )
        self.assertEqual(
            normalize_gguf_architecture(model_id="google/gemma-3-1b-it-GGUF"),
            "gemma3",
        )
        self.assertEqual(
            normalize_gguf_architecture(model_id="google/gemma-3n-E2B-it-GGUF"),
            "gemma3n",
        )
        self.assertEqual(
            normalize_gguf_architecture(model_id="google/gemma-4-2b-it-GGUF"),
            "gemma4",
        )

        for architecture in ("gemma3n", "gemma4"):
            with self.subTest(architecture=architecture):
                compatibility = gguf_shard_compatibility(
                    model_id=f"google/{architecture}-candidate-GGUF",
                    gguf_architecture=architecture,
                    filename=f"{architecture}-candidate-q4_k_m.gguf",
                )

                self.assertEqual(compatibility.gguf_architecture, architecture)
                self.assertEqual(
                    compatibility.shard_compatibility,
                    GGUF_SHARD_MODE_UNSUPPORTED_FOR_SHARDING,
                )
                self.assertFalse(compatibility.layer_range_supported)

    def test_phi3_is_registered_after_phi3_mini_conformance(self) -> None:
        proof = gguf_layer_range_architecture_proof("phi3")
        valid, error = validate_gguf_layer_range_architecture_proof(
            architecture="phi3",
            repo_root=REPO_ROOT,
        )
        compatibility = gguf_shard_compatibility(
            model_id="tensorblock/Phi-3-mini-4k-instruct-GGUF",
            gguf_architecture="phi3",
            filename="Phi-3-mini-4k-instruct-Q2_K.gguf",
        )

        self.assertIsNotNone(proof)
        assert proof is not None
        self.assertEqual(
            proof.reference_model_id,
            "tensorblock/Phi-3-mini-4k-instruct-GGUF",
        )
        self.assertIn("phi3", GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS)
        self.assertTrue(valid, error)
        self.assertIsNone(error)
        self.assertEqual(compatibility.gguf_architecture, "phi3")
        self.assertEqual(
            compatibility.shard_compatibility,
            GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED,
        )
        self.assertTrue(compatibility.layer_range_supported)
        self.assertIn(
            "phi3-mini-4k-instruct-layer-range-equivalence-probe",
            compatibility.layer_range_equivalence_probe_report or "",
        )
        self.assertIn(
            "phi3-mini-4k-instruct-production-binary-conformance",
            compatibility.layer_range_probe_report or "",
        )

    def test_phi2_is_registered_after_phi2_conformance(self) -> None:
        proof = gguf_layer_range_architecture_proof("phi2")
        valid, error = validate_gguf_layer_range_architecture_proof(
            architecture="phi2",
            repo_root=REPO_ROOT,
        )
        compatibility = gguf_shard_compatibility(
            model_id="TheBloke/phi-2-GGUF",
            gguf_architecture="phi2",
            filename="phi-2.Q2_K.gguf",
        )

        self.assertIsNotNone(proof)
        assert proof is not None
        self.assertEqual(proof.reference_model_id, "TheBloke/phi-2-GGUF")
        self.assertIn("phi2", GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS)
        self.assertTrue(valid, error)
        self.assertIsNone(error)
        self.assertEqual(compatibility.gguf_architecture, "phi2")
        self.assertEqual(
            compatibility.shard_compatibility,
            GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED,
        )
        self.assertTrue(compatibility.layer_range_supported)
        self.assertIn(
            "phi2-layer-range-equivalence-probe",
            compatibility.layer_range_equivalence_probe_report or "",
        )
        self.assertIn(
            "phi2-production-binary-conformance",
            compatibility.layer_range_probe_report or "",
        )

    def test_gptneox_is_registered_after_pythia14m_conformance(self) -> None:
        proof = gguf_layer_range_architecture_proof("gptneox")
        valid, error = validate_gguf_layer_range_architecture_proof(
            architecture="gptneox",
            repo_root=REPO_ROOT,
        )
        compatibility = gguf_shard_compatibility(
            model_id="DevQuasar-3/EleutherAI.pythia-14m-GGUF",
            gguf_architecture="gptneox",
            filename="pythia-14m.Q4_K_M.gguf",
        )

        self.assertIsNotNone(proof)
        assert proof is not None
        self.assertEqual(
            proof.reference_model_id,
            "DevQuasar-3/EleutherAI.pythia-14m-GGUF",
        )
        self.assertIn("gptneox", GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS)
        self.assertTrue(valid, error)
        self.assertIsNone(error)
        self.assertEqual(compatibility.gguf_architecture, "gptneox")
        self.assertEqual(
            compatibility.shard_compatibility,
            GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED,
        )
        self.assertTrue(compatibility.layer_range_supported)
        self.assertIn(
            "gptneox-pythia-14m-layer-range-equivalence-probe",
            compatibility.layer_range_equivalence_probe_report or "",
        )
        self.assertIn(
            "gptneox-pythia-14m-production-binary-conformance",
            compatibility.layer_range_probe_report or "",
        )

    def test_gemma3_is_registered_after_gemma3_it_conformance(self) -> None:
        proof = gguf_layer_range_architecture_proof("gemma3")
        valid, error = validate_gguf_layer_range_architecture_proof(
            architecture="gemma3",
            repo_root=REPO_ROOT,
        )
        compatibility = gguf_shard_compatibility(
            model_id="second-state/gemma-3-1b-it-GGUF",
            gguf_architecture="gemma3",
            filename="gemma-3-1b-it-Q2_K.gguf",
        )

        self.assertIsNotNone(proof)
        assert proof is not None
        self.assertEqual(
            proof.reference_model_id,
            "second-state/gemma-3-1b-it-GGUF",
        )
        self.assertIn("gemma3", GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS)
        self.assertTrue(valid, error)
        self.assertIsNone(error)
        self.assertEqual(compatibility.gguf_architecture, "gemma3")
        self.assertEqual(
            compatibility.shard_compatibility,
            GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED,
        )
        self.assertTrue(compatibility.layer_range_supported)
        self.assertIn(
            "gemma3-1b-it-layer-range-equivalence-probe",
            compatibility.layer_range_equivalence_probe_report or "",
        )
        self.assertIn(
            "gemma3-1b-it-production-binary-conformance",
            compatibility.layer_range_probe_report or "",
        )

    def test_gemma2_is_registered_after_gemma2_it_conformance(self) -> None:
        proof = gguf_layer_range_architecture_proof("gemma2")
        valid, error = validate_gguf_layer_range_architecture_proof(
            architecture="gemma2",
            repo_root=REPO_ROOT,
        )
        compatibility = gguf_shard_compatibility(
            model_id="MaziyarPanahi/gemma-2-2b-it-GGUF",
            gguf_architecture="gemma2",
            filename="gemma-2-2b-it.Q2_K.gguf",
        )

        self.assertIsNotNone(proof)
        assert proof is not None
        self.assertEqual(
            proof.reference_model_id,
            "MaziyarPanahi/gemma-2-2b-it-GGUF",
        )
        self.assertIn("gemma2", GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS)
        self.assertTrue(valid, error)
        self.assertIsNone(error)
        self.assertEqual(compatibility.gguf_architecture, "gemma2")
        self.assertEqual(
            compatibility.shard_compatibility,
            GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED,
        )
        self.assertTrue(compatibility.layer_range_supported)
        self.assertIn(
            "gemma2-2b-it-layer-range-equivalence-probe",
            compatibility.layer_range_equivalence_probe_report or "",
        )
        self.assertIn(
            "gemma2-2b-it-production-binary-conformance",
            compatibility.layer_range_probe_report or "",
        )

    def test_gemma_is_registered_after_gemma2b_conformance(self) -> None:
        proof = gguf_layer_range_architecture_proof("gemma")
        valid, error = validate_gguf_layer_range_architecture_proof(
            architecture="gemma",
            repo_root=REPO_ROOT,
        )
        compatibility = gguf_shard_compatibility(
            model_id="tensorblock/gemma-2b-GGUF",
            gguf_architecture="gemma",
            filename="gemma-2b-Q2_K.gguf",
        )

        self.assertIsNotNone(proof)
        assert proof is not None
        self.assertEqual(proof.reference_model_id, "tensorblock/gemma-2b-GGUF")
        self.assertIn("gemma", GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS)
        self.assertTrue(valid, error)
        self.assertIsNone(error)
        self.assertEqual(compatibility.gguf_architecture, "gemma")
        self.assertEqual(
            compatibility.shard_compatibility,
            GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED,
        )
        self.assertTrue(compatibility.layer_range_supported)
        self.assertIn(
            "gemma-2b-layer-range-equivalence-probe",
            compatibility.layer_range_equivalence_probe_report or "",
        )
        self.assertIn(
            "gemma-2b-production-binary-conformance",
            compatibility.layer_range_probe_report or "",
        )

    def test_mistral3_is_registered_after_ministral_conformance(self) -> None:
        proof = gguf_layer_range_architecture_proof("mistral3")
        valid, error = validate_gguf_layer_range_architecture_proof(
            architecture="mistral3",
            repo_root=REPO_ROOT,
        )
        compatibility = gguf_shard_compatibility(
            model_id="mistralai/Ministral-3-3B-Instruct-2512-GGUF",
            gguf_architecture="mistral3",
            filename="Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
        )

        self.assertIsNotNone(proof)
        assert proof is not None
        self.assertEqual(
            proof.reference_model_id,
            "mistralai/Ministral-3-3B-Instruct-2512-GGUF",
        )
        self.assertIn("mistral3", GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS)
        self.assertTrue(valid, error)
        self.assertIsNone(error)
        self.assertEqual(compatibility.gguf_architecture, "mistral3")
        self.assertEqual(
            compatibility.shard_compatibility,
            GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED,
        )
        self.assertTrue(compatibility.layer_range_supported)
        self.assertIn(
            "mistral3-ministral-3-3b-layer-range-equivalence-probe",
            compatibility.layer_range_equivalence_probe_report or "",
        )
        self.assertIn(
            "mistral3-ministral-3-3b-production-binary-conformance",
            compatibility.layer_range_probe_report or "",
        )

    def test_qwen2_is_registered_after_qwen25_conformance(self) -> None:
        proof = gguf_layer_range_architecture_proof("qwen2")
        valid, error = validate_gguf_layer_range_architecture_proof(
            architecture="qwen2",
            repo_root=REPO_ROOT,
        )
        compatibility = gguf_shard_compatibility(
            model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            gguf_architecture="qwen2",
            filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        )

        self.assertIsNotNone(proof)
        assert proof is not None
        self.assertEqual(proof.reference_model_id, "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
        self.assertIn("qwen2", GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS)
        self.assertTrue(valid, error)
        self.assertIsNone(error)
        self.assertEqual(
            compatibility.shard_compatibility,
            GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED,
        )
        self.assertTrue(compatibility.layer_range_supported)
        self.assertIn(
            "qwen2.5-layer-range-equivalence-probe",
            compatibility.layer_range_equivalence_probe_report or "",
        )
        self.assertIn(
            "qwen2.5-production-binary-conformance",
            compatibility.layer_range_probe_report or "",
        )

    def test_qwen_is_registered_after_qwen18_conformance(self) -> None:
        proof = gguf_layer_range_architecture_proof("qwen")
        valid, error = validate_gguf_layer_range_architecture_proof(
            architecture="qwen",
            repo_root=REPO_ROOT,
        )
        compatibility = gguf_shard_compatibility(
            model_id="mradermacher/Qwen-1_8B-Chat-GGUF",
            gguf_architecture="qwen",
            filename="Qwen-1_8B-Chat.Q2_K.gguf",
        )

        self.assertIsNotNone(proof)
        assert proof is not None
        self.assertEqual(proof.reference_model_id, "mradermacher/Qwen-1_8B-Chat-GGUF")
        self.assertIn("qwen", GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS)
        self.assertTrue(valid, error)
        self.assertIsNone(error)
        self.assertEqual(compatibility.gguf_architecture, "qwen")
        self.assertEqual(
            compatibility.shard_compatibility,
            GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED,
        )
        self.assertTrue(compatibility.layer_range_supported)
        self.assertIn(
            "qwen-1.8b-chat-layer-range-equivalence-probe",
            compatibility.layer_range_equivalence_probe_report or "",
        )
        self.assertIn(
            "qwen-1.8b-chat-production-binary-conformance",
            compatibility.layer_range_probe_report or "",
        )

    def test_qwen15_model_ids_are_covered_by_qwen2_architecture(self) -> None:
        compatibility = gguf_shard_compatibility(
            model_id="Qwen/Qwen1.5-0.5B-Chat-GGUF",
            gguf_architecture="qwen",
            filename="qwen1_5-0_5b-chat-q4_k_m.gguf",
        )

        self.assertEqual(compatibility.gguf_architecture, "qwen2")
        self.assertEqual(
            compatibility.shard_compatibility,
            GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED,
        )
        self.assertTrue(compatibility.layer_range_supported)

    def test_layer_range_whitelist_is_exactly_the_registered_proof_set(self) -> None:
        cases = (
            ("cai-network/Qwen3-0.6B-GGUF", "qwen", "Qwen3-0.6B-Q8_0.gguf"),
            (
                "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                "qwen",
                "qwen2.5-0.5b-instruct-q4_k_m.gguf",
            ),
            (
                "Qwen/Qwen1.5-0.5B-Chat-GGUF",
                "qwen",
                "qwen1_5-0_5b-chat-q4_k_m.gguf",
            ),
            (
                "mradermacher/Qwen-1_8B-Chat-GGUF",
                "qwen",
                "Qwen-1_8B-Chat.Q2_K.gguf",
            ),
            ("Qwen/Qwen3.5-2B-GGUF", "qwen", "qwen3.5-2b-q4_k_m.gguf"),
            (
                "Qwen/Qwen3-Next-80B-A3B-GGUF",
                "qwen",
                "qwen3-next-80b-a3b-q4_k_m.gguf",
            ),
            (
                "cai-network/TinyLlama-1.1B-Chat-v1.0-GGUF",
                "llama",
                "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            ),
            ("TheBloke/Mistral-7B-GGUF", "mistral", "mistral-7b-q4_k_m.gguf"),
            ("mistralai/Mistral-Small-3.1-GGUF", "mistral3", "mistral3-q4_k_m.gguf"),
            ("mistralai/Mistral-Large-4-GGUF", "mistral4", "mistral4-q4_k_m.gguf"),
            ("tensorblock/gemma-2b-GGUF", "gemma", "gemma-2b-Q2_K.gguf"),
            ("google/gemma-2-2b-it-GGUF", "gemma2", "gemma-2-2b-it-q4_k_m.gguf"),
            ("google/gemma-3-1b-it-GGUF", "gemma3", "gemma-3-1b-it-q4_k_m.gguf"),
            (
                "tensorblock/Phi-3-mini-4k-instruct-GGUF",
                "phi3",
                "Phi-3-mini-4k-instruct-Q2_K.gguf",
            ),
            ("TheBloke/phi-2-GGUF", "phi2", "phi-2.Q2_K.gguf"),
            (
                "DevQuasar-3/EleutherAI.pythia-14m-GGUF",
                "gptneox",
                "pythia-14m.Q4_K_M.gguf",
            ),
        )

        for model_id, architecture, filename in cases:
            with self.subTest(model_id=model_id):
                compatibility = gguf_shard_compatibility(
                    model_id=model_id,
                    gguf_architecture=architecture,
                    filename=filename,
                )
                has_registered_proof = (
                    compatibility.gguf_architecture
                    in GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS
                )

                self.assertEqual(
                    compatibility.layer_range_supported,
                    has_registered_proof,
                )
                if has_registered_proof:
                    self.assertEqual(
                        compatibility.shard_compatibility,
                        GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED,
                    )
                else:
                    self.assertEqual(
                        compatibility.shard_compatibility,
                        GGUF_SHARD_MODE_UNSUPPORTED_FOR_SHARDING,
                    )

    def test_explicit_unproven_header_wins_over_proven_model_name(self) -> None:
        compatibility = gguf_shard_compatibility(
            model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            gguf_architecture="mistral",
            filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        )

        self.assertEqual(compatibility.gguf_architecture, "mistral")
        self.assertEqual(
            compatibility.shard_compatibility,
            GGUF_SHARD_MODE_UNSUPPORTED_FOR_SHARDING,
        )
        self.assertFalse(compatibility.layer_range_supported)

    def test_classic_mistral_with_llama_header_uses_llama_proof_gate(self) -> None:
        compatibility = gguf_shard_compatibility(
            model_id="TheBloke/Mistral-7B-GGUF",
            gguf_architecture="llama",
            filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        )

        self.assertEqual(compatibility.gguf_architecture, "llama")
        self.assertEqual(
            compatibility.shard_compatibility,
            GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED,
        )
        self.assertTrue(compatibility.layer_range_supported)

    def test_registered_qwen3_proof_reports_pass(self) -> None:
        valid, error = validate_gguf_layer_range_architecture_proof(
            architecture="qwen3",
            repo_root=REPO_ROOT,
        )

        self.assertTrue(valid, error)
        self.assertIsNone(error)

    def test_llama_layer_range_support_is_registered_after_production_conformance(
        self,
    ) -> None:
        proof = gguf_layer_range_architecture_proof("llama")
        valid, error = validate_gguf_layer_range_architecture_proof(
            architecture="llama",
            repo_root=REPO_ROOT,
        )

        self.assertIsNotNone(proof)
        assert proof is not None
        self.assertEqual(
            proof.reference_model_id,
            "cai-network/TinyLlama-1.1B-Chat-v1.0-GGUF",
        )
        self.assertTrue(valid, error)
        self.assertIsNone(error)

    def test_unproven_architectures_are_explicitly_unsupported_for_sharding(self) -> None:
        for model_id, family, expected_architecture in (
            ("Qwen/Qwen3.5-2B-GGUF", "qwen", "qwen35"),
            ("Qwen/Qwen3.5-35B-A3B-GGUF", "qwen", "moe"),
            ("TheBloke/Mistral-7B-GGUF", "mistral", "mistral"),
            ("mistralai/Mistral-Large-4-GGUF", "mistral4", "mistral4"),
            ("google/gemma-4-2b-it-GGUF", "gemma4", "gemma4"),
            ("tiiuae/falcon-7b-instruct-GGUF", "falcon", "falcon"),
        ):
            with self.subTest(model_id=model_id):
                compatibility = gguf_shard_compatibility(
                    model_id=model_id,
                    gguf_architecture=family,
                )

                self.assertEqual(
                    compatibility.gguf_architecture,
                    expected_architecture,
                )
                self.assertEqual(
                    compatibility.shard_compatibility,
                    GGUF_SHARD_MODE_UNSUPPORTED_FOR_SHARDING,
                )
                self.assertFalse(compatibility.layer_range_supported)

    def test_unproven_public_gguf_can_be_marked_full_model_local_only(self) -> None:
        compatibility = gguf_shard_compatibility(
            model_id="TheBloke/Mistral-7B-GGUF",
            gguf_architecture="mistral",
            allow_full_model_local=True,
        )

        self.assertEqual(compatibility.gguf_architecture, "mistral")
        self.assertEqual(
            compatibility.shard_compatibility,
            GGUF_SHARD_MODE_FULL_MODEL_LOCAL,
        )
        self.assertFalse(compatibility.layer_range_supported)
        self.assertIn("single-node full-model local inference", compatibility.reason)

    def test_multimodal_qwen_variants_are_not_treated_as_dense_qwen(self) -> None:
        for kwargs in (
            {"model_id": "Qwen/Qwen3-VL-4B-Instruct-GGUF"},
            {"model_id": "Qwen/Qwen2.5-Omni-7B-GGUF"},
            {"model_id": "Qwen/Qwen2-Audio-7B-Instruct-GGUF"},
            {"architecture": "qwen3vl"},
            {"architecture": "qwen2audio"},
            {"architecture": "qwen2.5omni"},
        ):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(
                    normalize_gguf_architecture(**kwargs),
                    "multimodal",
                )

        compatibility = gguf_shard_compatibility(
            model_id="Qwen/Qwen2.5-Omni-7B-GGUF",
            gguf_architecture="qwen",
        )

        self.assertEqual(compatibility.gguf_architecture, "multimodal")
        self.assertEqual(
            compatibility.shard_compatibility,
            GGUF_SHARD_MODE_UNSUPPORTED_FOR_SHARDING,
        )
        self.assertFalse(compatibility.layer_range_supported)

    def test_qwen3_policy_report_passes_layer_range_conformance_gate(self) -> None:
        report = load_gguf_layer_range_conformance_report(
            REPO_ROOT / GGUF_QWEN3_LAYER_RANGE_PROBE_REPORT,
            expected_model_id="cai-network/Qwen3-0.6B-GGUF",
            expected_architecture="qwen3",
        )

        valid, error = validate_gguf_layer_range_conformance_report(
            report,
            expected_model_id="cai-network/Qwen3-0.6B-GGUF",
            expected_architecture="qwen3",
        )

        self.assertTrue(valid, error)
        self.assertIsNone(error)

    def test_qwen3_policy_report_passes_layer_range_equivalence_gate(self) -> None:
        report = load_gguf_layer_range_equivalence_probe_report(
            REPO_ROOT / GGUF_QWEN3_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT,
            expected_model_id="cai-network/Qwen3-0.6B-GGUF",
            expected_architecture="qwen3",
        )

        valid, error = validate_gguf_layer_range_equivalence_probe_report(
            report,
            expected_model_id="cai-network/Qwen3-0.6B-GGUF",
            expected_architecture="qwen3",
        )

        self.assertTrue(valid, error)
        self.assertIsNone(error)

    def test_layer_range_equivalence_gate_accepts_enriched_legacy_probe_abi(self) -> None:
        report = json.loads(
            (
                REPO_ROOT / GGUF_QWEN3_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT
            ).read_text(encoding="utf-8")
        )
        report["probeAbi"] = "cai-qwen3-layer-range-v1"
        del report["layerRange"]["activationShape"]

        valid, error = validate_gguf_layer_range_equivalence_probe_report(
            report,
            expected_model_id="cai-network/Qwen3-0.6B-GGUF",
            expected_architecture="qwen3",
        )

        self.assertTrue(valid, error)

    def test_layer_range_equivalence_gate_requires_hash_shape_and_diff_budget(self) -> None:
        report = json.loads(
            (
                REPO_ROOT / GGUF_QWEN3_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT
            ).read_text(encoding="utf-8")
        )
        tampered = copy.deepcopy(report)
        tampered["model"]["ggufSha256Hex"] = "not-a-hash"

        valid, error = validate_gguf_layer_range_equivalence_probe_report(
            tampered,
            expected_model_id="cai-network/Qwen3-0.6B-GGUF",
            expected_architecture="qwen3",
        )

        self.assertFalse(valid)
        self.assertEqual(
            error,
            "GGUF layer-range equivalence probe GGUF hash is invalid.",
        )

        tampered = copy.deepcopy(report)
        tampered["layerRange"]["activationFloatCount"] = 1

        valid, error = validate_gguf_layer_range_equivalence_probe_report(
            tampered,
            expected_model_id="cai-network/Qwen3-0.6B-GGUF",
            expected_architecture="qwen3",
        )

        self.assertFalse(valid)
        self.assertEqual(
            error,
            "GGUF layer-range equivalence probe activation shape does not match "
            "activationFloatCount.",
        )

        tampered = copy.deepcopy(report)
        tampered["result"]["maxAbsDiff"] = 0.5

        valid, error = validate_gguf_layer_range_equivalence_probe_report(
            tampered,
            expected_model_id="cai-network/Qwen3-0.6B-GGUF",
            expected_architecture="qwen3",
        )

        self.assertFalse(valid)
        self.assertEqual(
            error,
            "GGUF layer-range equivalence probe maxAbsDiff is too high.",
        )

    def test_llama_reports_pass_and_whitelist_llama(self) -> None:
        report = load_gguf_layer_range_equivalence_probe_report(
            REPO_ROOT / GGUF_LLAMA_TINYLLAMA_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT,
            expected_model_id="cai-network/TinyLlama-1.1B-Chat-v1.0-GGUF",
            expected_architecture="llama",
        )
        conformance_report = load_gguf_layer_range_conformance_report(
            REPO_ROOT / GGUF_LLAMA_TINYLLAMA_LAYER_RANGE_PROBE_REPORT,
            expected_model_id="cai-network/TinyLlama-1.1B-Chat-v1.0-GGUF",
            expected_architecture="llama",
        )
        compatibility = gguf_shard_compatibility(
            model_id="cai-network/TinyLlama-1.1B-Chat-v1.0-GGUF",
            gguf_architecture="llama",
            filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        )

        valid, error = validate_gguf_layer_range_equivalence_probe_report(
            report,
            expected_model_id="cai-network/TinyLlama-1.1B-Chat-v1.0-GGUF",
            expected_architecture="llama",
        )
        conformance_valid, conformance_error = (
            validate_gguf_layer_range_conformance_report(
                conformance_report,
                expected_model_id="cai-network/TinyLlama-1.1B-Chat-v1.0-GGUF",
                expected_architecture="llama",
            )
        )

        self.assertTrue(valid, error)
        self.assertIsNone(error)
        self.assertTrue(conformance_valid, conformance_error)
        self.assertIsNone(conformance_error)
        self.assertEqual(report["layerRange"]["totalLayers"], 22)
        self.assertEqual(report["layerRange"]["splitLayer"], 11)
        self.assertEqual(report["result"]["maxAbsDiff"], 0)
        self.assertEqual(
            compatibility.shard_compatibility,
            GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED,
        )
        self.assertTrue(compatibility.layer_range_supported)
        self.assertIn(
            "llama-tinyllama-production-binary-conformance",
            compatibility.layer_range_probe_report or "",
        )

    def test_qwen25_reports_pass_and_whitelist_qwen2(self) -> None:
        report = load_gguf_layer_range_equivalence_probe_report(
            REPO_ROOT / GGUF_QWEN2_5_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT,
            expected_model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            expected_architecture="qwen2",
        )
        conformance_report = load_gguf_layer_range_conformance_report(
            REPO_ROOT / GGUF_QWEN2_5_LAYER_RANGE_PROBE_REPORT,
            expected_model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            expected_architecture="qwen2",
        )
        compatibility = gguf_shard_compatibility(
            model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            gguf_architecture="qwen2",
            filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        )

        valid, error = validate_gguf_layer_range_equivalence_probe_report(
            report,
            expected_model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            expected_architecture="qwen2",
        )
        conformance_valid, conformance_error = (
            validate_gguf_layer_range_conformance_report(
                conformance_report,
                expected_model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                expected_architecture="qwen2",
            )
        )

        self.assertTrue(valid, error)
        self.assertIsNone(error)
        self.assertTrue(conformance_valid, conformance_error)
        self.assertIsNone(conformance_error)
        self.assertEqual(report["layerRange"]["totalLayers"], 24)
        self.assertEqual(report["layerRange"]["splitLayer"], 12)
        self.assertEqual(report["result"]["maxAbsDiff"], 0)
        self.assertEqual(
            compatibility.shard_compatibility,
            GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED,
        )
        self.assertTrue(compatibility.layer_range_supported)

    def test_qwen18_reports_pass_and_whitelist_qwen(self) -> None:
        report = load_gguf_layer_range_equivalence_probe_report(
            REPO_ROOT / GGUF_QWEN_1_8B_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT,
            expected_model_id="mradermacher/Qwen-1_8B-Chat-GGUF",
            expected_architecture="qwen",
        )
        conformance_report = load_gguf_layer_range_conformance_report(
            REPO_ROOT / GGUF_QWEN_1_8B_LAYER_RANGE_PROBE_REPORT,
            expected_model_id="mradermacher/Qwen-1_8B-Chat-GGUF",
            expected_architecture="qwen",
        )
        compatibility = gguf_shard_compatibility(
            model_id="mradermacher/Qwen-1_8B-Chat-GGUF",
            gguf_architecture="qwen",
            filename="Qwen-1_8B-Chat.Q2_K.gguf",
        )

        valid, error = validate_gguf_layer_range_equivalence_probe_report(
            report,
            expected_model_id="mradermacher/Qwen-1_8B-Chat-GGUF",
            expected_architecture="qwen",
        )
        conformance_valid, conformance_error = (
            validate_gguf_layer_range_conformance_report(
                conformance_report,
                expected_model_id="mradermacher/Qwen-1_8B-Chat-GGUF",
                expected_architecture="qwen",
            )
        )

        self.assertTrue(valid, error)
        self.assertIsNone(error)
        self.assertTrue(conformance_valid, conformance_error)
        self.assertIsNone(conformance_error)
        self.assertEqual(report["layerRange"]["totalLayers"], 24)
        self.assertEqual(report["layerRange"]["splitLayer"], 12)
        self.assertEqual(report["layerRange"]["nEmbedding"], 2048)
        self.assertEqual(report["result"]["maxAbsDiff"], 0)
        self.assertEqual(report["result"]["meanAbsDiff"], 0)
        self.assertTrue(report["result"]["topTokenMatch"])
        self.assertEqual(
            compatibility.shard_compatibility,
            GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED,
        )
        self.assertTrue(compatibility.layer_range_supported)

    def test_mistral3_reports_pass_and_whitelist_mistral3(self) -> None:
        report = load_gguf_layer_range_equivalence_probe_report(
            REPO_ROOT / GGUF_MISTRAL3_MINISTRAL_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT,
            expected_model_id="mistralai/Ministral-3-3B-Instruct-2512-GGUF",
            expected_architecture="mistral3",
        )
        conformance_report = load_gguf_layer_range_conformance_report(
            REPO_ROOT / GGUF_MISTRAL3_MINISTRAL_LAYER_RANGE_PROBE_REPORT,
            expected_model_id="mistralai/Ministral-3-3B-Instruct-2512-GGUF",
            expected_architecture="mistral3",
        )
        compatibility = gguf_shard_compatibility(
            model_id="mistralai/Ministral-3-3B-Instruct-2512-GGUF",
            gguf_architecture="mistral3",
            filename="Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
        )

        valid, error = validate_gguf_layer_range_equivalence_probe_report(
            report,
            expected_model_id="mistralai/Ministral-3-3B-Instruct-2512-GGUF",
            expected_architecture="mistral3",
        )
        conformance_valid, conformance_error = (
            validate_gguf_layer_range_conformance_report(
                conformance_report,
                expected_model_id="mistralai/Ministral-3-3B-Instruct-2512-GGUF",
                expected_architecture="mistral3",
            )
        )

        self.assertTrue(valid, error)
        self.assertIsNone(error)
        self.assertTrue(conformance_valid, conformance_error)
        self.assertIsNone(conformance_error)
        self.assertEqual(report["layerRange"]["totalLayers"], 26)
        self.assertEqual(report["layerRange"]["splitLayer"], 13)
        self.assertEqual(report["layerRange"]["nEmbedding"], 3072)
        self.assertEqual(report["result"]["maxAbsDiff"], 0)
        self.assertEqual(report["result"]["meanAbsDiff"], 0)
        self.assertTrue(report["result"]["topTokenMatch"])
        self.assertEqual(
            compatibility.shard_compatibility,
            GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED,
        )
        self.assertTrue(compatibility.layer_range_supported)

    def test_gemma_reports_pass_and_whitelist_gemma(self) -> None:
        report = load_gguf_layer_range_equivalence_probe_report(
            REPO_ROOT / GGUF_GEMMA_2B_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT,
            expected_model_id="tensorblock/gemma-2b-GGUF",
            expected_architecture="gemma",
        )
        conformance_report = load_gguf_layer_range_conformance_report(
            REPO_ROOT / GGUF_GEMMA_2B_LAYER_RANGE_PROBE_REPORT,
            expected_model_id="tensorblock/gemma-2b-GGUF",
            expected_architecture="gemma",
        )
        compatibility = gguf_shard_compatibility(
            model_id="tensorblock/gemma-2b-GGUF",
            gguf_architecture="gemma",
            filename="gemma-2b-Q2_K.gguf",
        )

        valid, error = validate_gguf_layer_range_equivalence_probe_report(
            report,
            expected_model_id="tensorblock/gemma-2b-GGUF",
            expected_architecture="gemma",
        )
        conformance_valid, conformance_error = (
            validate_gguf_layer_range_conformance_report(
                conformance_report,
                expected_model_id="tensorblock/gemma-2b-GGUF",
                expected_architecture="gemma",
            )
        )

        self.assertTrue(valid, error)
        self.assertIsNone(error)
        self.assertTrue(conformance_valid, conformance_error)
        self.assertIsNone(conformance_error)
        self.assertEqual(report["layerRange"]["totalLayers"], 18)
        self.assertEqual(report["layerRange"]["splitLayer"], 9)
        self.assertEqual(report["layerRange"]["nEmbedding"], 2048)
        self.assertEqual(report["result"]["maxAbsDiff"], 0)
        self.assertEqual(report["result"]["meanAbsDiff"], 0)
        self.assertTrue(report["result"]["topTokenMatch"])
        self.assertEqual(
            compatibility.shard_compatibility,
            GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED,
        )
        self.assertTrue(compatibility.layer_range_supported)

    def test_gemma2_reports_pass_and_whitelist_gemma2(self) -> None:
        report = load_gguf_layer_range_equivalence_probe_report(
            REPO_ROOT / GGUF_GEMMA2_2B_IT_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT,
            expected_model_id="MaziyarPanahi/gemma-2-2b-it-GGUF",
            expected_architecture="gemma2",
        )
        conformance_report = load_gguf_layer_range_conformance_report(
            REPO_ROOT / GGUF_GEMMA2_2B_IT_LAYER_RANGE_PROBE_REPORT,
            expected_model_id="MaziyarPanahi/gemma-2-2b-it-GGUF",
            expected_architecture="gemma2",
        )
        compatibility = gguf_shard_compatibility(
            model_id="MaziyarPanahi/gemma-2-2b-it-GGUF",
            gguf_architecture="gemma2",
            filename="gemma-2-2b-it.Q2_K.gguf",
        )

        valid, error = validate_gguf_layer_range_equivalence_probe_report(
            report,
            expected_model_id="MaziyarPanahi/gemma-2-2b-it-GGUF",
            expected_architecture="gemma2",
        )
        conformance_valid, conformance_error = (
            validate_gguf_layer_range_conformance_report(
                conformance_report,
                expected_model_id="MaziyarPanahi/gemma-2-2b-it-GGUF",
                expected_architecture="gemma2",
            )
        )

        self.assertTrue(valid, error)
        self.assertIsNone(error)
        self.assertTrue(conformance_valid, conformance_error)
        self.assertIsNone(conformance_error)
        self.assertEqual(report["layerRange"]["totalLayers"], 26)
        self.assertEqual(report["layerRange"]["splitLayer"], 13)
        self.assertEqual(report["layerRange"]["nEmbedding"], 2304)
        self.assertEqual(report["result"]["maxAbsDiff"], 0)
        self.assertEqual(report["result"]["meanAbsDiff"], 0)
        self.assertTrue(report["result"]["topTokenMatch"])
        self.assertEqual(
            compatibility.shard_compatibility,
            GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED,
        )
        self.assertTrue(compatibility.layer_range_supported)

    def test_gemma3_reports_pass_and_whitelist_gemma3(self) -> None:
        report = load_gguf_layer_range_equivalence_probe_report(
            REPO_ROOT / GGUF_GEMMA3_1B_IT_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT,
            expected_model_id="second-state/gemma-3-1b-it-GGUF",
            expected_architecture="gemma3",
        )
        conformance_report = load_gguf_layer_range_conformance_report(
            REPO_ROOT / GGUF_GEMMA3_1B_IT_LAYER_RANGE_PROBE_REPORT,
            expected_model_id="second-state/gemma-3-1b-it-GGUF",
            expected_architecture="gemma3",
        )
        compatibility = gguf_shard_compatibility(
            model_id="second-state/gemma-3-1b-it-GGUF",
            gguf_architecture="gemma3",
            filename="gemma-3-1b-it-Q2_K.gguf",
        )

        valid, error = validate_gguf_layer_range_equivalence_probe_report(
            report,
            expected_model_id="second-state/gemma-3-1b-it-GGUF",
            expected_architecture="gemma3",
        )
        conformance_valid, conformance_error = (
            validate_gguf_layer_range_conformance_report(
                conformance_report,
                expected_model_id="second-state/gemma-3-1b-it-GGUF",
                expected_architecture="gemma3",
            )
        )

        self.assertTrue(valid, error)
        self.assertIsNone(error)
        self.assertTrue(conformance_valid, conformance_error)
        self.assertIsNone(conformance_error)
        self.assertEqual(report["layerRange"]["totalLayers"], 26)
        self.assertEqual(report["layerRange"]["splitLayer"], 13)
        self.assertEqual(report["layerRange"]["nEmbedding"], 1152)
        self.assertEqual(report["result"]["maxAbsDiff"], 0)
        self.assertEqual(report["result"]["meanAbsDiff"], 0)
        self.assertTrue(report["result"]["topTokenMatch"])
        self.assertEqual(
            compatibility.shard_compatibility,
            GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED,
        )
        self.assertTrue(compatibility.layer_range_supported)

    def test_phi3_reports_pass_and_whitelist_phi3(self) -> None:
        report = load_gguf_layer_range_equivalence_probe_report(
            REPO_ROOT / GGUF_PHI3_MINI_4K_INSTRUCT_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT,
            expected_model_id="tensorblock/Phi-3-mini-4k-instruct-GGUF",
            expected_architecture="phi3",
        )
        conformance_report = load_gguf_layer_range_conformance_report(
            REPO_ROOT / GGUF_PHI3_MINI_4K_INSTRUCT_LAYER_RANGE_PROBE_REPORT,
            expected_model_id="tensorblock/Phi-3-mini-4k-instruct-GGUF",
            expected_architecture="phi3",
        )
        compatibility = gguf_shard_compatibility(
            model_id="tensorblock/Phi-3-mini-4k-instruct-GGUF",
            gguf_architecture="phi3",
            filename="Phi-3-mini-4k-instruct-Q2_K.gguf",
        )

        valid, error = validate_gguf_layer_range_equivalence_probe_report(
            report,
            expected_model_id="tensorblock/Phi-3-mini-4k-instruct-GGUF",
            expected_architecture="phi3",
        )
        conformance_valid, conformance_error = (
            validate_gguf_layer_range_conformance_report(
                conformance_report,
                expected_model_id="tensorblock/Phi-3-mini-4k-instruct-GGUF",
                expected_architecture="phi3",
            )
        )

        self.assertTrue(valid, error)
        self.assertIsNone(error)
        self.assertTrue(conformance_valid, conformance_error)
        self.assertIsNone(conformance_error)
        self.assertEqual(report["layerRange"]["totalLayers"], 32)
        self.assertEqual(report["layerRange"]["splitLayer"], 16)
        self.assertEqual(report["layerRange"]["nEmbedding"], 3072)
        self.assertEqual(report["result"]["maxAbsDiff"], 0)
        self.assertEqual(report["result"]["meanAbsDiff"], 0)
        self.assertTrue(report["result"]["topTokenMatch"])
        self.assertEqual(
            compatibility.shard_compatibility,
            GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED,
        )
        self.assertTrue(compatibility.layer_range_supported)

    def test_phi2_reports_pass_and_whitelist_phi2(self) -> None:
        report = load_gguf_layer_range_equivalence_probe_report(
            REPO_ROOT / GGUF_PHI2_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT,
            expected_model_id="TheBloke/phi-2-GGUF",
            expected_architecture="phi2",
        )
        conformance_report = load_gguf_layer_range_conformance_report(
            REPO_ROOT / GGUF_PHI2_LAYER_RANGE_PROBE_REPORT,
            expected_model_id="TheBloke/phi-2-GGUF",
            expected_architecture="phi2",
        )
        compatibility = gguf_shard_compatibility(
            model_id="TheBloke/phi-2-GGUF",
            gguf_architecture="phi2",
            filename="phi-2.Q2_K.gguf",
        )

        valid, error = validate_gguf_layer_range_equivalence_probe_report(
            report,
            expected_model_id="TheBloke/phi-2-GGUF",
            expected_architecture="phi2",
        )
        conformance_valid, conformance_error = (
            validate_gguf_layer_range_conformance_report(
                conformance_report,
                expected_model_id="TheBloke/phi-2-GGUF",
                expected_architecture="phi2",
            )
        )

        self.assertTrue(valid, error)
        self.assertIsNone(error)
        self.assertTrue(conformance_valid, conformance_error)
        self.assertIsNone(conformance_error)
        self.assertEqual(report["layerRange"]["totalLayers"], 32)
        self.assertEqual(report["layerRange"]["splitLayer"], 16)
        self.assertEqual(report["layerRange"]["nEmbedding"], 2560)
        self.assertEqual(report["result"]["maxAbsDiff"], 0)
        self.assertEqual(report["result"]["meanAbsDiff"], 0)
        self.assertTrue(report["result"]["topTokenMatch"])
        self.assertEqual(
            compatibility.shard_compatibility,
            GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED,
        )
        self.assertTrue(compatibility.layer_range_supported)

    def test_gptneox_reports_pass_and_whitelist_gptneox(self) -> None:
        report = load_gguf_layer_range_equivalence_probe_report(
            REPO_ROOT / GGUF_GPTNEOX_PYTHIA_14M_LAYER_RANGE_EQUIVALENCE_PROBE_REPORT,
            expected_model_id="DevQuasar-3/EleutherAI.pythia-14m-GGUF",
            expected_architecture="gptneox",
        )
        conformance_report = load_gguf_layer_range_conformance_report(
            REPO_ROOT / GGUF_GPTNEOX_PYTHIA_14M_LAYER_RANGE_PROBE_REPORT,
            expected_model_id="DevQuasar-3/EleutherAI.pythia-14m-GGUF",
            expected_architecture="gptneox",
        )
        compatibility = gguf_shard_compatibility(
            model_id="DevQuasar-3/EleutherAI.pythia-14m-GGUF",
            gguf_architecture="gptneox",
            filename="pythia-14m.Q4_K_M.gguf",
        )

        valid, error = validate_gguf_layer_range_equivalence_probe_report(
            report,
            expected_model_id="DevQuasar-3/EleutherAI.pythia-14m-GGUF",
            expected_architecture="gptneox",
        )
        conformance_valid, conformance_error = (
            validate_gguf_layer_range_conformance_report(
                conformance_report,
                expected_model_id="DevQuasar-3/EleutherAI.pythia-14m-GGUF",
                expected_architecture="gptneox",
            )
        )

        self.assertTrue(valid, error)
        self.assertIsNone(error)
        self.assertTrue(conformance_valid, conformance_error)
        self.assertIsNone(conformance_error)
        self.assertEqual(report["layerRange"]["totalLayers"], 6)
        self.assertEqual(report["layerRange"]["splitLayer"], 3)
        self.assertEqual(report["layerRange"]["nEmbedding"], 128)
        self.assertEqual(report["result"]["maxAbsDiff"], 0)
        self.assertEqual(report["result"]["meanAbsDiff"], 0)
        self.assertTrue(report["result"]["topTokenMatch"])
        self.assertEqual(
            compatibility.shard_compatibility,
            GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED,
        )
        self.assertTrue(compatibility.layer_range_supported)

    def test_layer_range_conformance_gate_requires_real_layer_execution(self) -> None:
        report = json.loads(
            (REPO_ROOT / GGUF_QWEN3_LAYER_RANGE_PROBE_REPORT).read_text(
                encoding="utf-8",
            )
        )
        tampered = copy.deepcopy(report)
        tampered["checks"]["generationProbe"]["realLayerExecution"] = False
        tampered["checks"]["productionReadinessChecks"]["generationProbe"][
            "realLayerExecution"
        ] = False

        valid, error = validate_gguf_layer_range_conformance_report(
            tampered,
            expected_model_id="cai-network/Qwen3-0.6B-GGUF",
            expected_architecture="qwen3",
        )

        self.assertFalse(valid)
        self.assertEqual(
            error,
            "GGUF layer-range conformance requires realLayerExecution=true.",
        )


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from cai_compute_chain.model import (
    DEFAULT_PRIVATE_NETWORK_MODEL_ID,
    NetworkModelPolicy,
    curated_model_for_id,
    resolve_execution_model_id,
)
from cai_compute_chain.node_config import _canonical_worker_allowed_model_id


def test_private_default_model_routes_to_public_execution_default() -> None:
    policy = NetworkModelPolicy()

    assert policy.network_default_model_id == DEFAULT_PRIVATE_NETWORK_MODEL_ID
    assert policy.network_default_execution_model_id == "Qwen/Qwen3-0.6B-GGUF"
    assert (
        resolve_execution_model_id(policy.network_default_model_id, policy)
        == policy.network_default_execution_model_id
    )


def test_network_execution_model_ids_track_execution_targets_only() -> None:
    policy = NetworkModelPolicy()

    assert policy.network_default_execution_model_id in policy.network_execution_model_ids
    assert "Qwen/Qwen2.5-0.5B-Instruct-GGUF" in policy.network_execution_model_ids
    assert DEFAULT_PRIVATE_NETWORK_MODEL_ID not in policy.network_execution_model_ids


def test_public_qwen3_curated_id_keeps_public_metadata_and_allowlist_id() -> None:
    curated_model = curated_model_for_id("Qwen/Qwen3-0.6B-GGUF")

    assert curated_model is not None
    assert curated_model.model_id == "Qwen/Qwen3-0.6B-GGUF"
    assert _canonical_worker_allowed_model_id("Qwen/Qwen3-0.6B-GGUF") == (
        "Qwen/Qwen3-0.6B-GGUF"
    )

# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
import json

from cai_compute_chain.node_config import load_or_create_node_config
from cai_compute_chain.wallet import WalletPolicy


def test_load_or_create_node_config_restores_public_qwen3_from_legacy_collapsed_defaults(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "cai_compute_chain.node_config.data_root",
        lambda _policy: tmp_path,
    )
    policy = WalletPolicy()
    config_path = tmp_path / policy.node_config_file_name
    config_path.write_text(
        json.dumps(
            {
                "worker_allowed_model_ids": [
                    "cai-network/Qwen3-0.6B-GGUF",
                    "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                ]
            }
        ),
        encoding="utf-8",
    )

    config = load_or_create_node_config(policy)

    assert config.worker_allowed_model_ids == [
        "cai-network/Qwen3-0.6B-GGUF",
        "Qwen/Qwen3-0.6B-GGUF",
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    ]

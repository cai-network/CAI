# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import unittest
import os
from pathlib import Path
from unittest.mock import patch

from cai_compute_chain.cai_runtime_launcher import (
    build_cai_runtime_env,
    discover_peer_book_peers,
    load_peer_book,
)
from cai_compute_chain.model import CaiNetworkConfig


class CaiRuntimeLauncherTests(unittest.TestCase):
    def test_runtime_env_sets_legacy_libp2p_namespace_for_native_bindings(self) -> None:
        env = build_cai_runtime_env("C:/CAI_HOME", CaiNetworkConfig())

        self.assertEqual(env["CAI_LIBP2P_NAMESPACE"], "cai-ai-net")
        self.assertEqual(env["EXO_LIBP2P_NAMESPACE"], "cai-ai-net")
        self.assertEqual(env["CAI_ENABLE_TASK_LEVEL_TRANSPORT_JOBS"], "1")
        self.assertEqual(env["CAI_REQUIRE_TASK_LEVEL_TRANSPORT_JOBS"], "1")
        self.assertEqual(env["CAI_ALLOW_TASK_LEVEL_TRANSPORT_PRIVATE_MODELS"], "1")
        self.assertEqual(env["CAI_TASK_LEVEL_TRANSPORT_REQUIRE_RUNTIME_READY"], "1")
        self.assertEqual(env["CAI_TASK_LEVEL_TRANSPORT_REQUIRE_SHARD_READINESS"], "1")
        self.assertEqual(env["CAI_TASK_LEVEL_TRANSPORT_REQUIRE_DATA_PLANE_ROUTE"], "1")
        self.assertEqual(
            env["CAI_TASK_LEVEL_TRANSPORT_REQUIRE_PROVEN_DATA_PLANE_ROUTE"],
            "1",
        )
        self.assertEqual(env["CAI_TASK_LEVEL_TRANSPORT_EXECUTOR_COUNT"], "1")
        self.assertEqual(env["CAI_OWNED_TRANSPORT_GENERATION_ENABLED"], "1")
        self.assertEqual(env["CAI_OWNED_TRANSPORT_GENERATION_REQUIRED"], "1")
        self.assertEqual(env["CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_RUNTIME_READY"], "1")
        self.assertEqual(env["CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_SHARD_READINESS"], "1")
        self.assertEqual(env["CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_DATA_PLANE_ROUTE"], "1")
        self.assertEqual(env["CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_PROVEN_DATA_PLANE_ROUTE"], "1")
        self.assertEqual(env["CAI_OWNED_TRANSPORT_STARTUP_SELF_TEST"], "1")
        self.assertEqual(env["CAI_OWNED_TRANSPORT_REQUIRE_LIVE_PROOF"], "0")
        self.assertEqual(env["CAI_JOB_EXECUTION_FIRST_RESPONSE_TIMEOUT_SECONDS"], "120")
        self.assertEqual(env["CAI_PRIVATE_NETWORK_MODEL_MIN_NODES"], "2")
        self.assertNotIn("CAI_EXECUTION_CAI_URL", env)

    def test_runtime_env_preserves_explicit_cai_owned_task_level_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CAI_EXECUTION_CAI_URL": "http://validator.example:52415",
                "CAI_TASK_LEVEL_TRANSPORT_EXECUTOR_COUNT": "4",
                "CAI_TASK_LEVEL_TRANSPORT_REQUIRE_RUNTIME_READY": "1",
                "CAI_JOB_EXECUTION_FIRST_RESPONSE_TIMEOUT_SECONDS": "45",
            },
            clear=True,
        ):
            env = build_cai_runtime_env("C:/CAI_HOME", CaiNetworkConfig())

        self.assertEqual(env["CAI_EXECUTION_CAI_URL"], "http://validator.example:52415")
        self.assertEqual(env["CAI_TASK_LEVEL_TRANSPORT_EXECUTOR_COUNT"], "4")
        self.assertEqual(env["CAI_TASK_LEVEL_TRANSPORT_REQUIRE_RUNTIME_READY"], "1")
        self.assertEqual(env["CAI_JOB_EXECUTION_FIRST_RESPONSE_TIMEOUT_SECONDS"], "45")

    def test_runtime_env_prepends_runtime_src_to_pythonpath(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            existing_pythonpath = os.environ.get("PYTHONPATH")
            with patch(
                "cai_compute_chain.cai_runtime_launcher.repo_root",
                return_value=repo,
            ), patch.dict(os.environ, {"PYTHONPATH": "C:/existing"}, clear=False):
                env = build_cai_runtime_env("C:/CAI_HOME", CaiNetworkConfig())
            if existing_pythonpath is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = existing_pythonpath

        parts = env["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(parts[0], str(repo / "src"))
        self.assertIn("C:/existing", parts)
        self.assertEqual(env["CAI_RUNTIME_SRC"], str(repo / "src"))

    def test_runtime_env_configures_production_shard_adapter_when_engine_exists(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            engine = (
                repo
                / "_internal"
                / "llama.cpp"
                / "llama-cai-shard-engine.exe"
            )
            engine.parent.mkdir(parents=True)
            engine.write_text("", encoding="utf-8")
            with patch(
                "cai_compute_chain.cai_runtime_launcher.repo_root",
                return_value=repo,
            ), patch.dict(os.environ, {}, clear=True):
                env = build_cai_runtime_env("C:/CAI_HOME", CaiNetworkConfig())

        self.assertEqual(env["CAI_LLM_SHARD_ADAPTER"], "native_bridge")
        self.assertIn(
            "cai_llama_cpp_assignment_artifact_engine",
            env["CAI_LLM_SHARD_NATIVE_COMMAND"],
        )
        self.assertIn(
            "cai_llama_cpp_patched_executor_host",
            env["CAI_LLM_SHARD_ASSIGNMENT_EXECUTOR_COMMAND"],
        )
        self.assertIn(
            "cai_llama_cpp_patched_binary_executor",
            env["CAI_LLM_SHARD_PATCHED_ENGINE_COMMAND"],
        )
        self.assertIn(
            "llama-cai-shard-engine.exe",
            env["CAI_LLM_PATCHED_BINARY_COMMAND"],
        )
        self.assertEqual(env["CAI_LLM_PATCHED_BINARY_REQUIRE_REAL_LAYER_EXECUTION"], "1")
        self.assertEqual(env["CAI_LLM_PATCHED_BINARY_REQUIRE_SHARD_ONLY_LOADING"], "1")

    def test_load_peer_book_reads_bom_encoded_json(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            peer = "/ip4/85.137.164.250/tcp/52416"
            peer_book = Path(tmp) / ".cai-peer-book.json"
            peer_book.write_text("\ufeff" + f'["{peer}"]', encoding="utf-8")
            with patch(
                "cai_compute_chain.cai_runtime_launcher.peer_book_path",
                return_value=peer_book,
            ):
                self.assertEqual(load_peer_book(), [peer])

    def test_discover_peer_book_peers_crawls_imported_peers(self) -> None:
        bootstrap_peer = "/ip4/198.51.100.10/tcp/52416/p2p/bootstrap"
        peer_a = "/ip4/198.51.100.11/tcp/52416/p2p/node-a"
        peer_b = "/dns4/node-b.example/tcp/52416/p2p/node-b"
        payload_by_url = {
            "http://198.51.100.10:52415/state": {
                "overlayAdvertisedPeers": {
                    "bootstrap": [{"address": peer_a}],
                },
            },
            "http://198.51.100.11:52415/state": {
                "overlayAdvertisedPeers": {
                    "node-a": [peer_b],
                },
            },
            "http://node-b.example:52415/state": {
                "overlayAdvertisedPeers": {},
            },
        }

        imported, tried = discover_peer_book_peers(
            [bootstrap_peer],
            52415,
            max_state_urls=8,
            read_state_payload=lambda state_url: payload_by_url[state_url],
        )

        self.assertEqual(imported, [peer_a, peer_b])
        self.assertEqual(
            tried,
            [
                "http://198.51.100.10:52415/state",
                "http://198.51.100.11:52415/state",
                "http://node-b.example:52415/state",
            ],
        )

    def test_discover_peer_book_peers_respects_state_url_limit(self) -> None:
        bootstrap_peer = "/ip4/198.51.100.10/tcp/52416/p2p/bootstrap"
        peer_a = "/ip4/198.51.100.11/tcp/52416/p2p/node-a"

        imported, tried = discover_peer_book_peers(
            [bootstrap_peer],
            52415,
            max_state_urls=1,
            read_state_payload=lambda _state_url: {
                "overlayAdvertisedPeers": {
                    "bootstrap": [{"address": peer_a}],
                },
            },
        )

        self.assertEqual(imported, [peer_a])
        self.assertEqual(tried, ["http://198.51.100.10:52415/state"])


if __name__ == "__main__":
    unittest.main()

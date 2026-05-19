# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.cli import (  # noqa: E402
    handle_chunk_download_queue,
    handle_model_package_ensure_ready,
)
from cai_compute_chain.model_distribution import (  # noqa: E402
    ChunkSizePolicy,
    build_gguf_model_package_manifest,
    save_local_artifact_binding,
    save_model_package_manifest,
)


class CliModelDistributionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo_patch = patch(
            "cai_compute_chain.wallet.repo_root",
            return_value=Path(self.tempdir.name),
        )
        self.repo_patch.start()

    def tearDown(self) -> None:
        self.repo_patch.stop()
        self.tempdir.cleanup()

    def _write_temp_file(self, name: str, content: bytes) -> Path:
        path = Path(self.tempdir.name) / name
        path.write_bytes(content)
        return path

    def test_ensure_ready_output_reports_processed_source_breakdown(self) -> None:
        gguf_path = self._write_temp_file("cli-origin.gguf", b"abcdefghij")
        manifest = build_gguf_model_package_manifest(
            catalog_id="cli-origin",
            model_id="Qwen/Qwen3-0.6B-GGUF",
            version="v1",
            gguf_path=gguf_path,
            total_layers=5,
            chunk_size_policy=ChunkSizePolicy.ADAPTIVE,
            min_chunk_bytes=4,
            max_chunk_bytes=4,
            source_repo_id="Qwen/Qwen3-0.6B-GGUF",
        )
        save_model_package_manifest(manifest)
        save_local_artifact_binding(
            manifest.catalog_id,
            manifest.version,
            artifact_id="gguf-main",
            local_path=gguf_path,
        )

        output = handle_model_package_ensure_ready(
            manifest.catalog_id,
            manifest.version,
            start_layer=0,
            end_layer=5,
            device_rank=0,
            world_size=1,
            node_id="node-a",
            peer_inventory_json=None,
            seed_inventory_json=None,
            use_imported_peer_inventory=False,
            use_imported_seed_inventory=False,
            max_tasks=None,
        )

        self.assertIn("- initial_ready=false", output)
        self.assertIn("- final_ready=true", output)
        self.assertIn("- processed_completed=3", output)
        self.assertIn("- processed_failed=0", output)
        self.assertIn("- processed_source_peer_cache=0", output)
        self.assertIn("- processed_source_storage_seed=0", output)
        self.assertIn("- processed_source_origin=3", output)

        queue_output = handle_chunk_download_queue()
        self.assertIn("- completed=3", queue_output)
        self.assertIn("- queue_source_origin=3", queue_output)


if __name__ == "__main__":
    unittest.main()

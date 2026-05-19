# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import os
import secrets
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "generate-release-metadata.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location("release_metadata_tool", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ReleaseMetadataToolTests(unittest.TestCase):
    def test_generate_release_metadata_contains_hash_and_git_info(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cai-release-metadata-") as temp_dir:
            repo_root = Path(temp_dir)
            _init_repo(repo_root)
            artifact = repo_root / ".dist" / "CAI-portable.zip"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(b"portable artifact")
            output = repo_root / ".dist" / "release-metadata.json"

            subprocess.run(
                [
                    sys.executable,
                    str(TOOL_PATH),
                    "--repo-root",
                    str(repo_root),
                    "--artifact",
                    str(artifact),
                    "--output",
                    str(output),
                    "--version",
                    "0.1.0",
                    "--build-id",
                    "test-build",
                    "--no-sign",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            metadata = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(metadata["schemaVersion"], 1)
        self.assertEqual(metadata["kind"], "cai-release-artifact-metadata")
        self.assertEqual(metadata["version"], "0.1.0")
        self.assertEqual(metadata["buildId"], "test-build")
        self.assertFalse(metadata["gitDirty"])
        self.assertEqual(metadata["artifacts"][0]["path"], ".dist/CAI-portable.zip")
        self.assertEqual(metadata["artifacts"][0]["sizeBytes"], len(b"portable artifact"))
        self.assertEqual(
            metadata["artifacts"][0]["sha256"],
            "bac9f7463dd781b28883c91aae0c1ce24da27d42670304b08bc77e370bb857ba",
        )
        self.assertNotIn("signature", metadata)

    def test_signed_release_metadata_can_be_verified(self) -> None:
        module = _load_tool_module()
        seed = secrets.token_bytes(32)
        seed_b64 = module.encode_bytes(seed)
        public_key_b64 = module.public_key_b64_from_seed(seed)

        with tempfile.TemporaryDirectory(prefix="cai-release-metadata-signed-") as temp_dir:
            repo_root = Path(temp_dir)
            _init_repo(repo_root)
            artifact = repo_root / ".dist" / "CAI-portable.zip"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(b"signed portable artifact")
            output = repo_root / ".dist" / "release-metadata.json"
            env = os.environ.copy()
            env["CAI_RELEASE_SIGNING_SEED_B64"] = seed_b64

            subprocess.run(
                [
                    sys.executable,
                    str(TOOL_PATH),
                    "--repo-root",
                    str(repo_root),
                    "--artifact",
                    str(artifact),
                    "--output",
                    str(output),
                    "--version",
                    "0.1.0",
                ],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )

            metadata = json.loads(output.read_text(encoding="utf-8"))
            verify_env = os.environ.copy()
            verify_env["CAI_REQUIRE_SIGNED_RELEASES"] = "1"
            verify_env["CAI_RELEASE_TRUSTED_PUBLIC_KEYS_B64"] = public_key_b64
            completed = subprocess.run(
                [
                    sys.executable,
                    str(TOOL_PATH),
                    "--repo-root",
                    str(repo_root),
                    "--verify",
                    str(output),
                    "--require-signature",
                ],
                env=verify_env,
                capture_output=True,
                text=True,
                check=True,
            )

        self.assertEqual(
            metadata["signature"]["scheme"],
            "cai-release-metadata-hybrid-ed25519-ml-dsa-65-v1",
        )
        self.assertEqual(metadata["signature"]["public_key_b64"], public_key_b64)
        self.assertEqual(metadata["signature"]["pq_scheme"], "ml-dsa-65-v1")
        self.assertTrue(metadata["signature"]["pq_public_key_b64"])
        self.assertTrue(metadata["signature"]["pq_signature_b64"])
        self.assertIn("Release metadata is valid", completed.stdout)


def _init_repo(repo_root: Path) -> None:
    _run_git(repo_root, "init")
    _run_git(repo_root, "config", "user.email", "cai@example.invalid")
    _run_git(repo_root, "config", "user.name", "CAI Test")
    (repo_root / "README.md").write_text("initial\n", encoding="utf-8")
    _run_git(repo_root, "add", ".")
    _run_git(repo_root, "commit", "-m", "Initial commit")


def _run_git(repo_root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )


if __name__ == "__main__":
    unittest.main()

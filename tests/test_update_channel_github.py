# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cai_compute_chain.update_channel import (
    GitHubPortableUpdateSource,
    build_local_update_summary,
    check_for_updates,
    fetch_github_portable_update_manifest,
    resolve_github_update_repository,
    update_status_path,
)


def _run_git(repo_root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _create_source_repo(repo_root: Path, *, branch: str = "main") -> None:
    (repo_root / "src" / "cai_compute_chain").mkdir(parents=True, exist_ok=True)
    (repo_root / "cai" / "src" / "cai").mkdir(parents=True, exist_ok=True)
    (repo_root / "tools").mkdir(parents=True, exist_ok=True)

    (repo_root / "src" / "cai_compute_chain" / "__init__.py").write_text(
        "__version__ = '0.1.0'\n",
        encoding="utf-8",
    )
    (repo_root / "cai" / "src" / "cai" / "main.py").write_text(
        "def main():\n    return 0\n",
        encoding="utf-8",
    )
    (repo_root / "tools" / "run-cai-main.py").write_text(
        "print('run-cai-main')\n",
        encoding="utf-8",
    )

    _run_git(repo_root, "init")
    _run_git(repo_root, "checkout", "-b", branch)
    _run_git(repo_root, "config", "user.email", "cai@example.invalid")
    _run_git(repo_root, "config", "user.name", "CAI Test")
    _run_git(repo_root, "add", ".")
    _run_git(repo_root, "commit", "-m", "initial")


class GitHubUpdateChannelBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory(prefix="cai-update-github-")
        self.temp_dir = Path(self._temp_dir.name)

    def tearDown(self) -> None:
        self._temp_dir.cleanup()

    def test_check_for_updates_uses_github_repository_and_persists_status(self) -> None:
        repo_root = self.temp_dir / "repo"
        _create_source_repo(repo_root, branch="main")

        with patch.dict(
            "os.environ",
            {
                "CAI_UPDATE_CHANNEL": "github",
                "CAI_UPDATE_GITHUB_REPOSITORY": "octo/example",
                "CAI_UPDATE_GITHUB_BRANCH": "main",
            },
            clear=False,
        ), patch(
            "cai_compute_chain.update_channel.fetch_github_update_manifest",
            return_value={
                "gitCommit": "remote-commit-123",
                "gitBranch": "main",
                "version": None,
            },
        ):
            result = check_for_updates(repo_root, timeout_sec=1)

        self.assertEqual(result["channel"], "github")
        self.assertEqual(result["repository"], "octo/example")
        self.assertEqual(result["targetBranch"], "main")
        self.assertTrue(result["updateAvailable"])
        self.assertTrue(result["canApply"])
        self.assertIn("github.com/octo/example/tree/main", str(result["sourceUrl"]))

        status_payload = json.loads(update_status_path(repo_root).read_text(encoding="utf-8"))
        self.assertEqual(status_payload["channel"], "github")
        self.assertEqual(status_payload["repository"], "octo/example")
        self.assertEqual(status_payload["status"], "update_available")

        summary = build_local_update_summary(repo_root)
        self.assertEqual(summary["runtime"]["version"], "0.1.0")
        self.assertEqual(summary["updates"]["repository"], "octo/example")
        self.assertEqual(summary["updates"]["status"], "update_available")

    def test_check_for_updates_skips_when_github_branch_differs(self) -> None:
        repo_root = self.temp_dir / "repo"
        _create_source_repo(repo_root, branch="dev")

        with patch.dict(
            "os.environ",
            {
                "CAI_UPDATE_CHANNEL": "github",
                "CAI_UPDATE_GITHUB_REPOSITORY": "octo/example",
                "CAI_UPDATE_GITHUB_BRANCH": "main",
            },
            clear=False,
        ), patch(
            "cai_compute_chain.update_channel.fetch_github_update_manifest",
            return_value={
                "gitCommit": "remote-commit-123",
                "gitBranch": "main",
                "version": None,
            },
        ):
            result = check_for_updates(repo_root, timeout_sec=1)

        self.assertTrue(result["updateAvailable"])
        self.assertFalse(result["canApply"])
        self.assertIn("branch does not match", str(result["applyReason"]))
        self.assertEqual(result["status"], "skipped")

    def test_resolve_github_update_repository_uses_origin_remote(self) -> None:
        repo_root = self.temp_dir / "repo"
        _create_source_repo(repo_root, branch="main")
        _run_git(repo_root, "remote", "add", "origin", "https://github.com/octo/example.git")

        with patch.dict(
            "os.environ",
            {
                "CAI_UPDATE_GITHUB_REPOSITORY": "",
            },
            clear=False,
        ):
            repository = resolve_github_update_repository(repo_root)

        self.assertEqual(repository, "octo/example")

    def test_fetch_github_portable_update_manifest_uses_release_metadata(self) -> None:
        source = GitHubPortableUpdateSource(
            repository="octo/example",
            release_tag="latest",
            api_base_url="https://api.github.test",
            repo_url="https://github.com/octo/example.git",
            portable_asset_name="CAI-portable.zip",
            manifest_asset_names=("portable-update-manifest.json",),
            metadata_asset_names=("release-metadata.json",),
        )
        release_payload = {
            "tag_name": "v0.1.0",
            "target_commitish": "main",
            "published_at": "2026-05-19T00:00:00Z",
            "html_url": "https://github.com/octo/example/releases/tag/v0.1.0",
            "assets": [
                {
                    "name": "CAI-portable.zip",
                    "browser_download_url": "https://download.test/CAI-portable.zip",
                    "size": 123,
                },
                {
                    "name": "release-metadata.json",
                    "browser_download_url": "https://download.test/release-metadata.json",
                    "size": 456,
                },
            ],
        }
        metadata_payload = {
            "version": "0.1.0",
            "gitCommit": "abc123",
            "gitBranch": "main",
            "buildId": "0.1.0-0007-gabc123-20260519T000000Z",
            "artifacts": [
                {
                    "name": "CAI-portable.zip",
                    "path": ".dist/CAI-portable.zip",
                    "sizeBytes": 123,
                    "sha256": "a" * 64,
                }
            ],
        }

        def fake_fetch(url: str, *, timeout_sec: int) -> dict[str, object]:
            if url.endswith("/releases/latest"):
                return release_payload
            if url == "https://download.test/release-metadata.json":
                return metadata_payload
            raise AssertionError(f"unexpected URL: {url}")

        with patch("cai_compute_chain.update_channel._fetch_json_payload", side_effect=fake_fetch):
            manifest = fetch_github_portable_update_manifest(source, timeout_sec=1)

        self.assertEqual(manifest["channel"], "github")
        self.assertEqual(manifest["provider"], "github")
        self.assertEqual(manifest["repository"], "octo/example")
        self.assertEqual(manifest["installKind"], "portable")
        self.assertEqual(manifest["archiveUrl"], "https://download.test/CAI-portable.zip")
        self.assertEqual(manifest["archiveSha256"], "a" * 64)
        self.assertEqual(manifest["archiveSizeBytes"], 123)
        self.assertEqual(manifest["buildId"], "0.1.0-0007-gabc123-20260519T000000Z")

    def test_build_local_update_summary_reports_source_resolution_error(self) -> None:
        repo_root = self.temp_dir / "repo"
        _create_source_repo(repo_root, branch="main")

        with patch.dict(
            "os.environ",
            {
                "CAI_AUTO_UPDATE_ENABLED": "1",
                "CAI_UPDATE_CHANNEL": "bogus",
            },
            clear=False,
        ):
            summary = build_local_update_summary(repo_root)

        self.assertEqual(summary["updates"]["status"], "error")
        self.assertIn(
            "source could not be resolved",
            str(summary["updates"]["message"]),
        )
        self.assertEqual(
            summary["updates"]["sourceResolutionError"]["errorType"],
            "UpdateError",
        )
        self.assertIn(
            "Unsupported CAI update channel",
            summary["updates"]["sourceResolutionError"]["message"],
        )


if __name__ == "__main__":
    unittest.main()

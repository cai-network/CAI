# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "generate-release-notes.py"


class ReleaseNotesToolTests(unittest.TestCase):
    def test_generate_release_notes_from_git_range(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cai-release-notes-") as temp_dir:
            repo_root = Path(temp_dir)
            _run_git(repo_root, "init")
            _run_git(repo_root, "config", "user.email", "cai@example.invalid")
            _run_git(repo_root, "config", "user.name", "CAI Test")
            (repo_root / "README.md").write_text("initial\n", encoding="utf-8")
            _run_git(repo_root, "add", ".")
            _run_git(repo_root, "commit", "-m", "Initial commit")
            previous_commit = _run_git_text(repo_root, "rev-parse", "HEAD")
            (repo_root / "feature.txt").write_text("feature\n", encoding="utf-8")
            _run_git(repo_root, "add", ".")
            _run_git(repo_root, "commit", "-m", "Add release feature")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(TOOL_PATH),
                    "--repo-root",
                    str(repo_root),
                    "--from",
                    previous_commit,
                    "--to",
                    "HEAD",
                    "--version",
                    "v0.1.0",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

        self.assertIn("# CAI v0.1.0 Release Notes", completed.stdout)
        self.assertIn("Add release feature", completed.stdout)
        self.assertNotIn("Initial commit", completed.stdout)


def _run_git(repo_root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _run_git_text(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


if __name__ == "__main__":
    unittest.main()

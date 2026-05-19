#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def generate_release_notes(
    repo_root: Path,
    *,
    from_ref: str | None,
    to_ref: str,
    version: str,
    max_commits: int,
) -> str:
    target_commit = _run_git_text(repo_root, "rev-parse", to_ref)
    range_ref = f"{from_ref}..{to_ref}" if from_ref else to_ref
    log_args = [
        "log",
        f"--max-count={max(1, max_commits)}",
        "--pretty=format:%h%x09%s",
        range_ref,
    ]
    raw_log = _run_git_text(repo_root, *log_args)
    commits = [
        line.split("\t", 1)
        for line in raw_log.splitlines()
        if "\t" in line
    ]

    lines = [
        f"# CAI {version} Release Notes",
        "",
        f"- Version: {version}",
        f"- Target commit: {target_commit}",
        f"- Generated at: {datetime.now(tz=UTC).isoformat()}",
    ]
    if from_ref:
        lines.append(f"- Range: `{from_ref}..{to_ref}`")
    else:
        lines.append(f"- Range: `{to_ref}`")

    lines.extend(["", "## Changes"])
    if commits:
        lines.extend(f"- `{short}` {subject}" for short, subject in commits)
    else:
        lines.append("- No commits found in the selected range.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CAI release notes from git history.")
    parser.add_argument("--repo-root", default=".", help="CAI repository root.")
    parser.add_argument("--from", dest="from_ref", default=None, help="Previous tag/commit.")
    parser.add_argument("--to", dest="to_ref", default="HEAD", help="Release tag/commit.")
    parser.add_argument("--version", default="unreleased", help="Release version label.")
    parser.add_argument("--max-commits", type=int, default=100, help="Maximum commits to include.")
    parser.add_argument("--output", default=None, help="Optional Markdown output path.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    notes = generate_release_notes(
        repo_root,
        from_ref=args.from_ref,
        to_ref=args.to_ref,
        version=args.version,
        max_commits=args.max_commits,
    )
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = repo_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(notes, encoding="utf-8")
    else:
        print(notes, end="")
    return 0


def _run_git_text(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())

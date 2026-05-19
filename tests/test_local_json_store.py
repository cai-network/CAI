# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import errno
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.local_json_store import (  # noqa: E402
    atomic_write_json_array_file,
    atomic_write_json_object_file,
    atomic_write_text_file,
    read_json_array_file,
    read_json_object_file,
)


class LocalJsonStoreTests(unittest.TestCase):
    def test_atomic_write_text_file_retries_transient_replace_permission_error(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "settlements.json"
            path.write_text("old", encoding="utf-8")
            real_replace = os.replace
            attempts = {"count": 0}

            def flaky_replace(src, dst):
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise PermissionError(errno.EACCES, "Access denied")
                return real_replace(src, dst)

            with (
                patch(
                    "cai_compute_chain.local_json_store.os.replace",
                    side_effect=flaky_replace,
                ),
                patch("cai_compute_chain.local_json_store.time.sleep", return_value=None),
            ):
                atomic_write_text_file(path, "new")

            self.assertEqual(path.read_text(encoding="utf-8"), "new")
            self.assertEqual(attempts["count"], 2)

    def test_atomic_write_falls_back_to_overwrite_when_replace_is_locked(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            target = Path(tempdir) / "state.json"
            target.write_text("[]", encoding="utf-8")

            def locked_replace(_src, _dst):
                raise PermissionError(errno.EACCES, "locked")

            with patch(
                "cai_compute_chain.local_json_store.os.replace",
                side_effect=locked_replace,
            ):
                atomic_write_json_array_file(target, [{"ok": True}])

            self.assertEqual(read_json_array_file(target), [{"ok": True}])

    def test_read_json_object_file_recovers_empty_or_partial_object_payload(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "chain.json"
            path.write_text('{"blocks": []}\n\n', encoding="utf-8")
            self.assertEqual(
                read_json_object_file(path, heal_corrupt=True),
                {"blocks": []},
            )

    def test_atomic_write_json_object_file_writes_complete_object_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "chain.json"
            atomic_write_json_object_file(path, {"blocks": [{"height": 1}]})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"blocks": [{"height": 1}]},
            )


if __name__ == "__main__":
    unittest.main()

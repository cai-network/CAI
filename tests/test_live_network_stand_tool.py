# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "live-network-stand.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location("live_network_stand_tool", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class LiveNetworkStandToolTests(unittest.TestCase):
    def test_inventory_path_must_be_under_local_secrets_dir(self) -> None:
        module = _load_tool_module()
        with tempfile.TemporaryDirectory(prefix="cai-live-inventory-") as temp_dir:
            old_secrets_dir = module.LIVE_SECRETS_DIR
            try:
                module.LIVE_SECRETS_DIR = Path(temp_dir) / ".cai-local" / "secrets"
                allowed = module.LIVE_SECRETS_DIR / "live-stand-access.json"
                outside = Path(temp_dir) / "live-stand-access.json"

                module._assert_inventory_path_allowed(allowed)
                with self.assertRaisesRegex(ValueError, "must be stored under"):
                    module._assert_inventory_path_allowed(outside)
            finally:
                module.LIVE_SECRETS_DIR = old_secrets_dir

    def test_inventory_template_is_written_only_to_local_secrets_dir(self) -> None:
        module = _load_tool_module()
        with tempfile.TemporaryDirectory(prefix="cai-live-template-") as temp_dir:
            old_secrets_dir = module.LIVE_SECRETS_DIR
            try:
                module.LIVE_SECRETS_DIR = Path(temp_dir) / ".cai-local" / "secrets"
                inventory_path = module.LIVE_SECRETS_DIR / "live-stand-access.json"

                module.write_inventory_template(
                    SimpleNamespace(inventory=str(inventory_path), force=False)
                )

                payload = json.loads(inventory_path.read_text(encoding="utf-8"))
                self.assertEqual(payload["version"], 1)
                self.assertIn("nodes", payload)
                self.assertTrue(inventory_path.exists())
            finally:
                module.LIVE_SECRETS_DIR = old_secrets_dir

    def test_report_sanitizer_redacts_inventory_secrets(self) -> None:
        module = _load_tool_module()
        inventory = {
            "nodes": {
                "vps": {
                    "kind": "ssh",
                    "password": "secret-password",
                    "seed_phrase": "alpha beta gamma delta",
                    "api_token": "token-12345",
                }
            }
        }
        report = {
            "nodes": [
                {
                    "name": "vps",
                    "commands": [
                        {
                            "command": "echo secret-password",
                            "stdout": "token-12345 alpha beta gamma delta",
                            "password": "secret-password",
                        }
                    ],
                }
            ]
        }

        sanitized = module._sanitize_for_report(report, inventory)
        serialized = json.dumps(sanitized, ensure_ascii=False)

        self.assertNotIn("secret-password", serialized)
        self.assertNotIn("alpha beta gamma delta", serialized)
        self.assertNotIn("token-12345", serialized)
        self.assertIn("[redacted]", serialized)


if __name__ == "__main__":
    unittest.main()

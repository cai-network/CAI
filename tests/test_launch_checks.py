# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.launch_checks import render_alpha_launch_report, run_alpha_launch_checks


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode('utf-8')


class LaunchCheckTests(unittest.TestCase):
    def test_alpha_launch_report_passes_for_healthy_single_validator_alpha(self) -> None:
        local_state = {
            'nodeIdentities': {
                'local-worker': {'workerEnabled': True},
                'peer-worker': {'workerEnabled': True},
            },
            'overlayPeers': {'peer-vps': {}},
            'topology': {
                'nodes': [{'id': 'local'}, {'id': 'vps'}],
                'connections': {
                    'local-worker': {
                        'peer-worker': [{'sinkMultiaddr': {'ip_address': '10.0.0.11', 'port': 6001}}],
                    },
                    'peer-worker': {
                        'local-worker': [{'sinkMultiaddr': {'ip_address': '10.0.0.10', 'port': 6000}}],
                    },
                },
            },
        }
        local_summary = {
            'validator': {'validator_set_size': 0},
            'compute': {'execution_receipts': 2},
        }
        remote_state = local_state
        remote_summary = {
            'validator': {'validator_set_size': 1, 'validator_state': 'bonded'},
        }

        payload_by_url = {
            'http://127.0.0.1:52425/state': local_state,
            'http://127.0.0.1:52425/v1/cai/summary': local_summary,
            'http://192.145.29.212:52415/state': remote_state,
            'http://192.145.29.212:52415/v1/cai/summary': remote_summary,
        }

        def fake_urlopen(url: str, timeout: int = 0):
            return FakeResponse(payload_by_url[url])

        with patch('cai_compute_chain.launch_checks.urlopen', side_effect=fake_urlopen):
            report = run_alpha_launch_checks()

        text = render_alpha_launch_report(report)
        self.assertTrue(report.ready)
        self.assertIn('Ready: yes', text)
        self.assertIn('[PASS] settlement_validator_path: remote validator path ready (1 validator(s))', text)

    def test_alpha_launch_report_fails_when_cluster_is_not_connected(self) -> None:
        local_state = {
            'nodeIdentities': {
                'local': {'workerEnabled': False},
            },
            'overlayPeers': {},
            'topology': {
                'nodes': [{'id': 'local'}],
                'connections': {},
            },
        }
        local_summary = {
            'validator': {'validator_set_size': 0},
            'compute': {'execution_receipts': 0},
        }

        payload_by_url = {
            'http://127.0.0.1:52425/state': local_state,
            'http://127.0.0.1:52425/v1/cai/summary': local_summary,
        }

        def fake_urlopen(url: str, timeout: int = 0):
            if url not in payload_by_url:
                raise OSError('connection refused')
            return FakeResponse(payload_by_url[url])

        with patch('cai_compute_chain.launch_checks.urlopen', side_effect=fake_urlopen):
            report = run_alpha_launch_checks(
                remote_state_url=None,
                remote_summary_url=None,
            )

        text = render_alpha_launch_report(report)
        self.assertFalse(report.ready)
        self.assertIn('[FAIL] local_cluster_nodes', text)
        self.assertIn('[FAIL] local_overlay_peers', text)
        self.assertIn('[FAIL] settlement_validator_path', text)
        self.assertIn('[FAIL] distributed_worker_pool', text)
        self.assertIn('Ready: no', text)


if __name__ == '__main__':
    unittest.main()

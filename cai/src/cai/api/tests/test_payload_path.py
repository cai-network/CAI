# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from cai.api.payload_path import resolve_payload_path


def test_resolve_payload_path_reads_nested_dict_and_list_items() -> None:
    payload = {"nodeIdentities": [{"workerEnabled": True}]}

    assert resolve_payload_path(payload, "nodeIdentities/0/workerEnabled") is True


def test_resolve_payload_path_ignores_empty_segments() -> None:
    payload = {"state": {"height": 1}}

    assert resolve_payload_path(payload, "/state//height/") == 1


def test_resolve_payload_path_returns_payload_for_empty_path() -> None:
    payload = {"ok": True}

    assert resolve_payload_path(payload, "") == payload

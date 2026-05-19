# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
from pathlib import Path

from cai.api.endpoint_policy import (
    CAI_ENDPOINT_POLICIES,
    EndpointAccess,
    endpoint_policy_index,
    lookup_endpoint_policy,
)


def test_cai_endpoint_policy_covers_registered_cai_routes() -> None:
    main_py = Path(__file__).resolve().parents[1] / "main.py"
    source = main_py.read_text(encoding="utf-8")
    registered_routes = {
        (method.upper(), path)
        for method, path in re.findall(
            r"self\.app\.(get|post|put|delete)\(\"([^\"]+)\"",
            source,
        )
        if path == "/cai/summary" or path.startswith("/v1/cai/")
    }
    missing = sorted(registered_routes - set(endpoint_policy_index()))

    assert missing == []


def test_sensitive_cai_endpoints_are_classified_local_or_admin() -> None:
    index = endpoint_policy_index()
    sensitive_paths = {
        "/v1/cai/history",
        "/v1/cai/desktop/preferences",
        "/v1/cai/wallet/create",
        "/v1/cai/wallet/restore",
        "/v1/cai/wallet/unlock",
        "/v1/cai/wallet/send",
        "/v1/cai/node/validator",
        "/v1/cai/node/worker",
        "/v1/cai/node/relay",
        "/v1/cai/update-package",
        "/v1/cai/update-package.zip",
    }

    for policy in CAI_ENDPOINT_POLICIES:
        if policy.path in sensitive_paths:
            assert policy.access in {EndpointAccess.LOCAL_ONLY, EndpointAccess.ADMIN_ONLY}


def test_lookup_endpoint_policy_normalizes_method() -> None:
    policy = lookup_endpoint_policy("post", "/v1/cai/wallet/send")

    assert policy is not None
    assert policy.access == EndpointAccess.LOCAL_ONLY

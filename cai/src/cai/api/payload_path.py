# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from typing import Any


def resolve_payload_path(payload: Any, path: str) -> Any:
    current = payload
    for attr in path.split("/"):
        if attr == "":
            continue
        if isinstance(current, dict):
            current = current[attr]
        elif isinstance(current, list):
            current = current[int(attr)]
    return current

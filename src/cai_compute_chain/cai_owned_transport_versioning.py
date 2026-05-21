# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .cai_owned_transport_common import optional_int as _optional_int
from .cai_owned_transport_protocol import (
    CAI_OWNED_TRANSPORT_PROTOCOL,
    CAI_OWNED_TRANSPORT_PROTOCOL_VERSION,
)


def cai_owned_transport_version_compatibility(
    payload: Mapping[str, Any] | None,
    *,
    require_runtime_versions: bool = False,
    require_protocol_version: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return {
            "compatible": False,
            "errors": ["CAI-owned transport version payload is missing."],
        }

    protocol = str(payload.get("protocol") or "").strip()
    if protocol and protocol != CAI_OWNED_TRANSPORT_PROTOCOL:
        errors.append("CAI-owned transport protocol is incompatible.")

    raw_protocol_version = payload.get("protocolVersion")
    protocol_version = _optional_int(raw_protocol_version)
    if protocol_version is None:
        if require_protocol_version:
            errors.append("CAI-owned transport protocol version is missing.")
    elif protocol_version != CAI_OWNED_TRANSPORT_PROTOCOL_VERSION:
        errors.append("CAI-owned transport protocol version is unsupported.")

    compatible_protocol_versions = cai_owned_transport_int_list(
        payload.get("compatibleProtocolVersions")
        if payload.get("compatibleProtocolVersions") is not None
        else payload.get("compatible_protocol_versions")
    )
    if (
        compatible_protocol_versions
        and CAI_OWNED_TRANSPORT_PROTOCOL_VERSION not in compatible_protocol_versions
    ):
        errors.append("CAI-owned transport compatible protocol set excludes CAI.")

    runtime_version = cai_owned_transport_version_label(
        payload.get("runtimeVersion")
        if payload.get("runtimeVersion") is not None
        else payload.get("runtime_version")
    )
    adapter_id = cai_owned_transport_version_label(
        payload.get("adapterId")
        if payload.get("adapterId") is not None
        else payload.get("adapter_id")
    )
    adapter_version = cai_owned_transport_version_label(
        payload.get("adapterVersion")
        if payload.get("adapterVersion") is not None
        else payload.get("adapter_version")
    )

    if require_runtime_versions and not runtime_version:
        errors.append("CAI-owned transport runtime version is missing.")
    if require_runtime_versions and not adapter_version:
        errors.append("CAI-owned transport adapter version is missing.")
    if runtime_version is None:
        errors.append("CAI-owned transport runtime version is invalid.")
    if adapter_id is None:
        errors.append("CAI-owned transport adapter id is invalid.")
    if adapter_version is None:
        errors.append("CAI-owned transport adapter version is invalid.")

    return {
        "compatible": not errors,
        "errors": errors,
        "protocolVersion": protocol_version,
        "expectedProtocolVersion": CAI_OWNED_TRANSPORT_PROTOCOL_VERSION,
        "compatibleProtocolVersions": compatible_protocol_versions,
        "runtimeVersion": runtime_version or None,
        "adapterId": adapter_id or None,
        "adapterVersion": adapter_version or None,
    }


def cai_owned_transport_version_label(value: object) -> str | None:
    if value is None or str(value or "").strip() == "":
        return ""
    clean = str(value or "").strip()
    allowed_chars = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789._:/+-"
    )
    if (
        len(clean) > 128
        or not clean.isascii()
        or any(ch.isspace() for ch in clean)
        or any(ch not in allowed_chars for ch in clean)
    ):
        return None
    return clean


def cai_owned_transport_int_list(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    cleaned: list[int] = []
    for item in value:
        integer = _optional_int(item)
        if integer is None:
            continue
        if integer not in cleaned:
            cleaned.append(integer)
    return cleaned

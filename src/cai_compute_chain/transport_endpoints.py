# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_ROUTE_PRIORITY = {"direct": 0, "overlay": 1, "relay": 2}
_SOURCE_PRIORITY = {"explicit": 0, "auto": 1, "interface_scan": 2}


def format_host_for_url(host: str) -> str:
    normalized = str(host or "").strip()
    if ":" in normalized and not normalized.startswith("["):
        return f"[{normalized}]"
    return normalized


def build_http_url(host: str, port: int, path: str = "") -> str:
    normalized_path = str(path or "")
    if normalized_path and not normalized_path.startswith("/"):
        normalized_path = f"/{normalized_path}"
    return f"http://{format_host_for_url(host)}:{int(port)}{normalized_path}"


def identity_transport_endpoints(
    identity: Mapping[str, Any] | None,
    *,
    purpose: str | None = None,
    route_types: Sequence[str] | None = None,
    require_port: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(identity, Mapping):
        return []

    raw_endpoints = identity.get("transportEndpoints")
    if raw_endpoints is None:
        raw_endpoints = identity.get("transport_endpoints")
    if not isinstance(raw_endpoints, Sequence) or isinstance(
        raw_endpoints, (str, bytes, bytearray)
    ):
        return []

    allowed_route_types = {str(item).strip().lower() for item in route_types or []}
    normalized: list[dict[str, Any]] = []
    for raw in raw_endpoints:
        if not isinstance(raw, Mapping):
            continue
        endpoint_purpose = str(raw.get("purpose") or "").strip().lower()
        route_type = str(raw.get("routeType") or raw.get("route_type") or "").strip().lower()
        host = str(raw.get("host") or "").strip()
        port = raw.get("port")
        try:
            normalized_port = int(port) if port is not None else None
        except (TypeError, ValueError):
            normalized_port = None
        if not host or not endpoint_purpose or not route_type:
            continue
        if purpose is not None and endpoint_purpose != str(purpose).strip().lower():
            continue
        if route_types is not None and route_type not in allowed_route_types:
            continue
        if require_port and normalized_port is None:
            continue
        normalized.append(
            {
                "purpose": endpoint_purpose,
                "routeType": route_type,
                "host": host,
                "port": normalized_port,
                "source": str(raw.get("source") or "").strip().lower() or None,
                "interfaceName": str(
                    raw.get("interfaceName") or raw.get("interface_name") or ""
                ).strip()
                or None,
            }
        )

    normalized.sort(
        key=lambda item: (
            _ROUTE_PRIORITY.get(str(item.get("routeType")), 99),
            _SOURCE_PRIORITY.get(str(item.get("source") or "interface_scan"), 99),
            str(item.get("host") or ""),
            -1 if item.get("port") is None else int(item["port"]),
        )
    )
    return normalized


def candidate_identity_http_urls(
    identity: Mapping[str, Any] | None,
    *,
    endpoint_path: str = "",
) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()

    for endpoint in identity_transport_endpoints(
        identity,
        purpose="api",
        require_port=True,
    ):
        host = str(endpoint.get("host") or "").strip()
        port = endpoint.get("port")
        if not host or host in {"0.0.0.0", "::"} or port is None or int(port) <= 0:
            continue
        url = build_http_url(host, int(port), endpoint_path)
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)

    if not isinstance(identity, Mapping):
        return urls

    host = str(identity.get("apiHost") or identity.get("api_host") or "").strip()
    port = identity.get("apiPort")
    try:
        normalized_port = int(port)
    except (TypeError, ValueError):
        normalized_port = -1
    if host and host not in {"0.0.0.0", "::"} and normalized_port > 0:
        fallback_url = build_http_url(host, normalized_port, endpoint_path)
        if fallback_url not in seen:
            urls.append(fallback_url)
    return urls


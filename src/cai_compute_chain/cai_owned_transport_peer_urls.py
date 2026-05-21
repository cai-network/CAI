# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Sequence
from urllib.parse import parse_qs, urlsplit

from .cai_owned_transport_protocol import CAI_OWNED_TRANSPORT_OVERLAY_URL_PREFIX


def clean_peer_cai_urls(peer_cai_urls: Sequence[str]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for raw_url in peer_cai_urls:
        url = str(raw_url or "").strip().rstrip("/")
        if not url or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def prioritized_cai_owned_transport_peer_urls(
    peer_cai_urls: Sequence[str],
) -> list[str]:
    urls = clean_peer_cai_urls(peer_cai_urls)
    indexed_urls = list(enumerate(urls))
    indexed_urls.sort(
        key=lambda item: (
            cai_owned_transport_peer_url_priority(item[1]),
            item[0],
        )
    )
    return [url for _index, url in indexed_urls]


def cai_owned_transport_peer_url_priority(peer_cai_url: str) -> int:
    raw = str(peer_cai_url or "").strip()
    if not raw.startswith(CAI_OWNED_TRANSPORT_OVERLAY_URL_PREFIX):
        return 0
    rest = raw[len(CAI_OWNED_TRANSPORT_OVERLAY_URL_PREFIX) :].strip()
    try:
        parsed = urlsplit(rest)
    except Exception:
        return 50
    relay_role = _overlay_relay_role(parsed.query)
    if relay_role in {"ordinary", "participant", "peer", "worker", "transit"}:
        return 10
    if relay_role in {"bootstrap", "validator", "vps", "primary"}:
        return 30
    return 20


def cai_owned_transport_peer_url_route_class(peer_cai_url: str) -> str:
    raw = str(peer_cai_url or "").strip()
    if not raw.startswith(CAI_OWNED_TRANSPORT_OVERLAY_URL_PREFIX):
        return "direct"
    rest = raw[len(CAI_OWNED_TRANSPORT_OVERLAY_URL_PREFIX) :].strip()
    try:
        parsed = urlsplit(rest)
    except Exception:
        return "overlay_invalid"
    relay_role = _overlay_relay_role(parsed.query)
    if relay_role in {"ordinary", "participant", "peer", "worker", "transit"}:
        return "overlay_ordinary"
    if relay_role in {"bootstrap", "validator", "vps", "primary"}:
        return "overlay_bootstrap"
    return "overlay_generic"


def parse_cai_owned_transport_overlay_url(value: str) -> tuple[str, str] | None:
    raw = str(value or "").strip()
    if not raw.startswith(CAI_OWNED_TRANSPORT_OVERLAY_URL_PREFIX):
        return None
    rest = raw[len(CAI_OWNED_TRANSPORT_OVERLAY_URL_PREFIX) :].strip()
    parsed = urlsplit(rest)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "CAI-owned transport overlay URL must be "
            "cai-overlay:http(s)://host:port?targetNodeId=<node_id>."
        )
    query = parse_qs(parsed.query)
    target_node_id = (
        (query.get("targetNodeId") or query.get("target_node_id") or [""])[0]
    ).strip()
    if not target_node_id:
        raise ValueError("CAI-owned transport overlay URL targetNodeId is required.")
    relay_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
    if not relay_url:
        raise ValueError("CAI-owned transport overlay relay URL is required.")
    return relay_url, target_node_id


def _overlay_relay_role(query: str) -> str:
    parsed_query = parse_qs(query)
    return str(
        (
            parsed_query.get("relayRole")
            or parsed_query.get("relay_role")
            or parsed_query.get("transitRole")
            or parsed_query.get("transit_role")
            or [""]
        )[0]
    ).strip().lower()

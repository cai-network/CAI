# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Sequence
from urllib.parse import parse_qs, urlsplit


CAI_OWNED_TRANSPORT_OVERLAY_URL_PREFIX = "cai-overlay:"


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

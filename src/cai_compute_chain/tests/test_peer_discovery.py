# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT

from cai_compute_chain import validators


def test_discover_peer_cai_urls_includes_bootstrap_seed(monkeypatch) -> None:
    monkeypatch.setattr(
        validators,
        "default_bootstrap_peers",
        lambda: ("/ip4/192.145.29.212/tcp/52416",),
    )
    monkeypatch.setattr(validators, "default_api_port", lambda: 52415)

    urls = validators.discover_peer_cai_urls(
        state_payload={"nodeIdentities": {}},
        cai_url="http://127.0.0.1:52415",
        endpoint_path="/v1/cai/validators",
    )

    assert urls == ["http://192.145.29.212:52415/v1/cai/validators"]


def test_discover_peer_cai_urls_deduplicates_local_bootstrap_identity(monkeypatch) -> None:
    monkeypatch.setattr(
        validators,
        "default_bootstrap_peers",
        lambda: ("/ip4/192.145.29.212/tcp/52416",),
    )
    monkeypatch.setattr(validators, "default_api_port", lambda: 52415)

    urls = validators.discover_peer_cai_urls(
        state_payload={
            "nodeIdentities": {
                "local-node": {
                    "apiHost": "192.145.29.212",
                    "apiPort": 52415,
                }
            }
        },
        cai_url="http://192.145.29.212:52415",
        local_node_id="local-node",
        endpoint_path="/v1/cai/chain",
    )

    assert urls == []

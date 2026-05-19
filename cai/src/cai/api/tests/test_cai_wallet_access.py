# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cai.api.main import API


def _make_api(service: object | None = None) -> API:
    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api.port = 52415
    api._get_cai_service = lambda: service or SimpleNamespace()  # pyright: ignore[method-assign]
    app.post("/v1/cai/wallet/create")(api.create_cai_wallet)
    app.post("/v1/cai/wallet/restore")(api.restore_cai_wallet)
    app.post("/v1/cai/wallet/select")(api.select_cai_wallet)
    app.post("/v1/cai/wallet/unlock")(api.unlock_cai_wallet)
    app.post("/v1/cai/wallet/lock")(api.lock_cai_wallet)
    app.post("/v1/cai/wallet/logout")(api.logout_cai_wallet)
    app.post("/v1/cai/wallet/send")(api.send_cai_wallet_transfer)
    return api


def test_wallet_endpoints_block_remote_public_clients() -> None:
    service = MagicMock()
    api = _make_api(service)
    client = TestClient(api.app, client=("198.51.100.20", 40000))
    requests = [
        ("/v1/cai/wallet/create", {"name": "main", "password": "pass"}),
        (
            "/v1/cai/wallet/restore",
            {"name": "main", "password": "pass", "seed_phrase": "seed words"},
        ),
        ("/v1/cai/wallet/select", {"selector": "main"}),
        ("/v1/cai/wallet/unlock", {"password": "pass"}),
        ("/v1/cai/wallet/lock", {}),
        ("/v1/cai/wallet/logout", {}),
        ("/v1/cai/wallet/send", {"to": "abcd", "amount": "1.00000000"}),
    ]

    for path, payload in requests:
        response = client.post(path, json=payload)
        assert response.status_code == 404, path

    service.assert_not_called()
    assert service.method_calls == []


def test_wallet_create_allows_local_client() -> None:
    service = SimpleNamespace(
        create_wallet=MagicMock(
            return_value={"wallet": {"wallet_id": "wallet-1", "address": "abcd"}}
        )
    )
    api = _make_api(service)
    client = TestClient(api.app, client=("127.0.0.1", 40000))

    response = client.post(
        "/v1/cai/wallet/create",
        json={"name": "main", "password": "pass"},
    )

    assert response.status_code == 200
    assert response.json()["wallet"]["wallet_id"] == "wallet-1"
    service.create_wallet.assert_called_once_with(name="main", password="pass")

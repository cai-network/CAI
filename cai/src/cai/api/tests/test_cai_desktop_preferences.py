# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cai.api.main import API


def _make_api() -> API:
    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api.port = 52415
    app.get("/v1/cai/desktop/preferences")(api.get_cai_desktop_preferences)
    app.put("/v1/cai/desktop/preferences")(api.update_cai_desktop_preferences)
    return api


def test_desktop_preferences_are_local_only() -> None:
    api = _make_api()
    client = TestClient(api.app, client=("198.51.100.20", 40000))

    response = client.get("/v1/cai/desktop/preferences")

    assert response.status_code == 404


def test_desktop_preferences_return_resolved_language() -> None:
    api = _make_api()
    client = TestClient(api.app, client=("127.0.0.1", 40000))

    with patch("cai.api.main.resolve_language", return_value="ru") as resolve_mock:
        response = client.get("/v1/cai/desktop/preferences")

    assert response.status_code == 200
    assert response.json() == {"language": "ru"}
    resolve_mock.assert_called_once_with("auto")


def test_desktop_preferences_update_persists_language() -> None:
    api = _make_api()
    client = TestClient(api.app, client=("127.0.0.1", 40000))

    with (
        patch("cai.api.main.save_desktop_language") as save_mock,
        patch("cai.api.main.resolve_language", return_value="de") as resolve_mock,
    ):
        response = client.put(
            "/v1/cai/desktop/preferences",
            json={"language": "de"},
        )

    assert response.status_code == 200
    assert response.json() == {"language": "de"}
    save_mock.assert_called_once_with("de")
    resolve_mock.assert_called_once_with("auto")


def test_desktop_preferences_update_rejects_unknown_language() -> None:
    api = _make_api()
    client = TestClient(api.app, client=("127.0.0.1", 40000))

    with patch("cai.api.main.save_desktop_language") as save_mock:
        response = client.put(
            "/v1/cai/desktop/preferences",
            json={"language": "it"},
        )

    assert response.status_code == 400
    save_mock.assert_not_called()


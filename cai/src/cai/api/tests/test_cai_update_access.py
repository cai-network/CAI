# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cai.api.main import API


def _make_api(update_server_enabled: bool) -> API:
    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api.port = 52415
    api.update_server_enabled = update_server_enabled
    app.get("/v1/cai/update-manifest")(api.get_cai_update_manifest)
    app.get("/v1/cai/update-package")(api.get_cai_update_package)
    app.get("/v1/cai/update-package.zip")(api.get_cai_update_package)
    app.get("/v1/cai/update/status")(api.get_cai_update_status)
    app.post("/v1/cai/update/check")(api.check_cai_update)
    app.post("/v1/cai/update/apply")(api.apply_cai_update)
    return api


def test_update_manifest_blocks_when_server_disabled() -> None:
    api = _make_api(update_server_enabled=False)
    client = TestClient(api.app, client=("198.51.100.20", 40000))

    with patch("cai.api.main.build_update_manifest") as manifest_mock:
        response = client.get("/v1/cai/update-manifest")

    assert response.status_code == 404
    manifest_mock.assert_not_called()


def test_update_manifest_returns_payload_when_enabled() -> None:
    api = _make_api(update_server_enabled=True)
    client = TestClient(api.app, client=("198.51.100.20", 40000))

    with patch("cai.api.main._resolve_cai_repo_root", return_value=Path("/tmp/cai")), patch(
        "cai.api.main.build_update_manifest",
        return_value={"gitCommit": "abc123", "archiveUrl": "http://example.invalid/update.zip"},
    ) as manifest_mock:
        response = client.get("/v1/cai/update-manifest")

    assert response.status_code == 200
    assert response.json()["gitCommit"] == "abc123"
    manifest_mock.assert_called_once()


def test_update_manifest_forwards_portable_install_kind() -> None:
    api = _make_api(update_server_enabled=True)
    client = TestClient(api.app, client=("198.51.100.20", 40000))

    with patch("cai.api.main._resolve_cai_repo_root", return_value=Path("/tmp/cai")), patch(
        "cai.api.main.build_update_manifest",
        return_value={"gitCommit": "abc123", "archiveUrl": "http://example.invalid/update.zip"},
    ) as manifest_mock:
        response = client.get("/v1/cai/update-manifest?install_kind=portable")

    assert response.status_code == 200
    manifest_mock.assert_called_once_with(
        Path("/tmp/cai"),
        base_url="http://testserver",
        install_kind="portable",
    )


def test_update_package_returns_archive_when_enabled(tmp_path: Path) -> None:
    archive_path = tmp_path / "cai-update.zip"
    archive_path.write_bytes(b"zip-bytes")
    api = _make_api(update_server_enabled=True)
    client = TestClient(api.app, client=("198.51.100.20", 40000))

    with patch("cai.api.main._resolve_cai_repo_root", return_value=tmp_path), patch(
        "cai.api.main.build_update_package",
        return_value=archive_path,
    ) as archive_mock:
        response = client.get("/v1/cai/update-package")

    assert response.status_code == 200
    assert response.content == b"zip-bytes"
    assert response.headers["content-type"] == "application/zip"
    archive_mock.assert_called_once_with(tmp_path, install_kind=None)


def test_update_package_forwards_portable_install_kind(tmp_path: Path) -> None:
    archive_path = tmp_path / "cai-portable.zip"
    archive_path.write_bytes(b"portable-zip-bytes")
    api = _make_api(update_server_enabled=True)
    client = TestClient(api.app, client=("198.51.100.20", 40000))

    with patch("cai.api.main._resolve_cai_repo_root", return_value=tmp_path), patch(
        "cai.api.main.build_update_package",
        return_value=archive_path,
    ) as archive_mock:
        response = client.get("/v1/cai/update-package?install_kind=portable")

    assert response.status_code == 200
    assert response.content == b"portable-zip-bytes"
    archive_mock.assert_called_once_with(tmp_path, install_kind="portable")


def test_update_package_supports_bounded_range(tmp_path: Path) -> None:
    archive_path = tmp_path / "cai-portable.zip"
    archive_path.write_bytes(b"portable-zip-bytes")
    api = _make_api(update_server_enabled=True)
    client = TestClient(api.app, client=("198.51.100.20", 40000))

    with patch("cai.api.main._resolve_cai_repo_root", return_value=tmp_path), patch(
        "cai.api.main.build_update_package",
        return_value=archive_path,
    ):
        response = client.get(
            "/v1/cai/update-package.zip?install_kind=portable",
            headers={"Range": "bytes=9-16"},
        )

    assert response.status_code == 206
    assert response.content == b"zip-byte"
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["content-range"] == "bytes 9-16/18"
    assert response.headers["content-length"] == "8"


def test_update_package_supports_open_ended_range(tmp_path: Path) -> None:
    archive_path = tmp_path / "cai-portable.zip"
    archive_path.write_bytes(b"portable-zip-bytes")
    api = _make_api(update_server_enabled=True)
    client = TestClient(api.app, client=("198.51.100.20", 40000))

    with patch("cai.api.main._resolve_cai_repo_root", return_value=tmp_path), patch(
        "cai.api.main.build_update_package",
        return_value=archive_path,
    ):
        response = client.get(
            "/v1/cai/update-package.zip?install_kind=portable",
            headers={"Range": "bytes=9-"},
        )

    assert response.status_code == 206
    assert response.content == b"zip-bytes"
    assert response.headers["content-range"] == "bytes 9-17/18"
    assert response.headers["content-length"] == "9"


def test_update_package_rejects_unsatisfied_range(tmp_path: Path) -> None:
    archive_path = tmp_path / "cai-portable.zip"
    archive_path.write_bytes(b"portable-zip-bytes")
    api = _make_api(update_server_enabled=True)
    client = TestClient(api.app, client=("198.51.100.20", 40000))

    with patch("cai.api.main._resolve_cai_repo_root", return_value=tmp_path), patch(
        "cai.api.main.build_update_package",
        return_value=archive_path,
    ):
        response = client.get(
            "/v1/cai/update-package.zip?install_kind=portable",
            headers={"Range": "bytes=99-"},
        )

    assert response.status_code == 416
    assert response.headers["content-range"] == "bytes */18"


def test_update_package_zip_alias_returns_archive(tmp_path: Path) -> None:
    archive_path = tmp_path / "cai-portable.zip"
    archive_path.write_bytes(b"portable-zip-bytes")
    api = _make_api(update_server_enabled=True)
    client = TestClient(api.app, client=("198.51.100.20", 40000))

    with patch("cai.api.main._resolve_cai_repo_root", return_value=tmp_path), patch(
        "cai.api.main.build_update_package",
        return_value=archive_path,
    ) as archive_mock:
        response = client.get("/v1/cai/update-package.zip?install_kind=portable")

    assert response.status_code == 200
    assert response.content == b"portable-zip-bytes"
    archive_mock.assert_called_once_with(tmp_path, install_kind="portable")


def test_update_status_check_and_apply_are_local_only() -> None:
    api = _make_api(update_server_enabled=False)
    remote_client = TestClient(api.app, client=("198.51.100.20", 40000))

    assert remote_client.get("/v1/cai/update/status").status_code == 404
    assert remote_client.post("/v1/cai/update/check").status_code == 404
    assert remote_client.post("/v1/cai/update/apply").status_code == 404


def test_update_status_check_and_apply_call_local_service() -> None:
    api = _make_api(update_server_enabled=False)
    local_client = TestClient(api.app, client=("127.0.0.1", 40000))

    class Service:
        def update_status(self) -> dict[str, object]:
            return {"status": "ready"}

        def check_update(self) -> dict[str, object]:
            return {"status": "up_to_date"}

        def apply_update(self) -> dict[str, object]:
            return {"status": "updated"}

    with patch.object(api, "_get_cai_service", return_value=Service()):
        status_response = local_client.get("/v1/cai/update/status")
        check_response = local_client.post("/v1/cai/update/check")
        apply_response = local_client.post("/v1/cai/update/apply")

    assert status_response.status_code == 200
    assert status_response.json()["status"] == "ready"
    assert check_response.status_code == 200
    assert check_response.json()["status"] == "up_to_date"
    assert apply_response.status_code == 200
    assert apply_response.json()["status"] == "updated"


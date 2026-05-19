# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import hashlib
import subprocess
import zipfile
from pathlib import Path
from unittest.mock import patch

from cai_compute_chain import update_channel as update_channel_module
from cai_compute_chain.update_channel import (
    PACKAGE_METADATA_PATH,
    UpdateError,
    apply_remote_update,
    apply_portable_update_archive,
    apply_update_archive,
    build_local_update_summary,
    build_runtime_version_label,
    build_update_archive,
    build_update_manifest,
    build_update_package,
    cancel_pending_portable_update,
    maybe_stage_portable_auto_update_on_launch,
    portable_update_batch_path,
    portable_update_cancel_path,
    portable_update_plan_path,
    portable_update_script_path,
    portable_update_activity_snapshot,
    record_portable_update_activity,
    resume_pending_portable_update_on_launch,
    schedule_portable_update_after_exit,
    check_for_updates,
    sha256_file,
    sign_update_manifest,
    update_rollback_backup_dir,
    update_rollback_marker_path,
    update_stage_dir,
    validate_update_manifest,
)
from cai_compute_chain.wallet_signing import (
    address_from_public_key_b64,
    encode_bytes,
    generate_signing_seed,
    public_key_b64_from_seed,
    sign_payload_b64,
)


def _run_git(repo_root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _create_source_repo(
    repo_root: Path,
    *,
    app_value: str,
    build_value: str,
    include_obsolete: bool,
) -> None:
    (repo_root / "src" / "cai_compute_chain").mkdir(parents=True, exist_ok=True)
    (repo_root / "cai" / "src" / "cai").mkdir(parents=True, exist_ok=True)
    (repo_root / "cai" / "dashboard" / "build").mkdir(parents=True, exist_ok=True)
    (repo_root / "tools").mkdir(parents=True, exist_ok=True)

    (repo_root / "src" / "cai_compute_chain" / "__init__.py").write_text(
        "__version__ = '0.1.0'\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "cai_compute_chain" / "app.py").write_text(
        f"value = {app_value!r}\n",
        encoding="utf-8",
    )
    (repo_root / "cai" / "src" / "cai" / "main.py").write_text(
        "def main():\n    return 0\n",
        encoding="utf-8",
    )
    (repo_root / "tools" / "run-cai-main.py").write_text(
        "print('run-cai-main')\n",
        encoding="utf-8",
    )
    (repo_root / ".gitignore").write_text(
        ".cai-update/\n.cai-update-cache/\n",
        encoding="utf-8",
    )
    (repo_root / "cai" / "dashboard" / "build" / "index.html").write_text(
        build_value,
        encoding="utf-8",
    )
    if include_obsolete:
        (repo_root / "obsolete.txt").write_text("obsolete\n", encoding="utf-8")

    _run_git(repo_root, "init")
    _run_git(repo_root, "config", "user.email", "cai@example.invalid")
    _run_git(repo_root, "config", "user.name", "CAI Test")
    _run_git(repo_root, "add", ".")
    _run_git(repo_root, "commit", "-m", "initial")


def _create_portable_root(portable_root: Path, *, app_value: str, include_data: bool) -> None:
    portable_root.mkdir(parents=True, exist_ok=True)
    (portable_root / "CAI.exe").write_text(f"exe={app_value}\n", encoding="utf-8")
    (portable_root / "_internal").mkdir(parents=True, exist_ok=True)
    (portable_root / "_internal" / "app.txt").write_text(
        f"app={app_value}\n",
        encoding="utf-8",
    )
    if include_data:
        (portable_root / "data" / ".cai-local").mkdir(parents=True, exist_ok=True)
        (portable_root / "data" / ".cai-local" / "wallets.json").write_text(
            "[]\n",
            encoding="utf-8",
        )


def _create_portable_archive(archive_path: Path, *, app_value: str, wrapped: bool = False) -> None:
    prefix = "CAI-portable/" if wrapped else ""
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(f"{prefix}CAI.exe", f"exe={app_value}\n")
        bundle.writestr(f"{prefix}_internal/app.txt", f"app={app_value}\n")


def test_runtime_version_label_prefers_build_id() -> None:
    label = build_runtime_version_label(
        "0.1.0",
        git_commit="abcdef1234567890",
        build_id="0.1.0+gabcdef123456.20260513T120000Z",
    )

    assert label == "0.1.0+gabcdef123456.20260513T120000Z"


def test_runtime_version_label_prefers_build_number_label() -> None:
    label = build_runtime_version_label(
        "0.1.0+gabcdef123456.20260513T120000Z",
        git_commit="abcdef1234567890",
        build_id="0.1.0-0001-gabcdef123456-20260513T120000Z",
        build_number_label="0001",
    )

    assert label == "0.1.0 0001"


def test_runtime_version_label_uses_commit_when_build_id_is_missing() -> None:
    label = build_runtime_version_label(
        "0.1.0",
        git_commit="abcdef1234567890",
    )

    assert label == "0.1.0+gabcdef123456"


def test_build_local_update_summary_exposes_portable_version_label(
    tmp_path: Path,
) -> None:
    portable_root = tmp_path / "portable"
    _create_portable_root(portable_root, app_value="local", include_data=False)
    (portable_root / ".dist").mkdir(parents=True, exist_ok=True)
    (portable_root / ".dist" / "release-metadata.json").write_text(
        json.dumps(
            {
                "version": "0.1.0+gabcdef123456.20260513T120000Z",
                "gitCommit": "abcdef1234567890",
                "gitBranch": "main",
                "buildId": "0.1.0+gabcdef123456.20260513T120000Z",
            }
        ),
        encoding="utf-8",
    )

    with patch.dict("os.environ", {"CAI_AUTO_UPDATE": "0"}, clear=False):
        summary = build_local_update_summary(portable_root)

    assert summary["runtime"]["version"] == "0.1.0+gabcdef123456.20260513T120000Z"
    assert summary["runtime"]["versionLabel"] == "0.1.0+gabcdef123456.20260513T120000Z"
    assert summary["runtime"]["buildId"] == "0.1.0+gabcdef123456.20260513T120000Z"


def test_build_local_update_summary_exposes_portable_build_number_label(
    tmp_path: Path,
) -> None:
    portable_root = tmp_path / "portable"
    _create_portable_root(portable_root, app_value="local", include_data=False)
    (portable_root / "release-metadata.json").write_text(
        json.dumps(
            {
                "version": "0.1.0",
                "gitCommit": "abcdef1234567890",
                "gitBranch": "main",
                "buildId": "0.1.0-0001-gabcdef123456-20260513T120000Z",
                "buildNumber": 1,
                "buildNumberLabel": "0001",
            }
        ),
        encoding="utf-8",
    )

    with patch.dict("os.environ", {"CAI_AUTO_UPDATE": "0"}, clear=False):
        summary = build_local_update_summary(portable_root)

    assert summary["runtime"]["version"] == "0.1.0"
    assert summary["runtime"]["versionLabel"] == "0.1.0 0001"
    assert summary["runtime"]["buildId"] == "0.1.0-0001-gabcdef123456-20260513T120000Z"
    assert summary["runtime"]["buildNumber"] == 1
    assert summary["runtime"]["buildNumberLabel"] == "0001"


def test_schedule_portable_update_after_exit_writes_plan_and_launcher(
    tmp_path: Path,
) -> None:
    portable_root = tmp_path / "portable"
    archive_path = tmp_path / "portable-update.zip"
    _create_portable_root(portable_root, app_value="local", include_data=False)
    _create_portable_archive(archive_path, app_value="remote")

    with patch("cai_compute_chain.update_channel.subprocess.Popen") as popen_mock:
        result = schedule_portable_update_after_exit(
            portable_root,
            archive_path,
            relaunch_command=["CAI.exe", "--no-tray"],
            parent_pid=1234,
            remote_manifest={
                "version": "0.1.0+gabc.20260513T120000Z",
                "gitCommit": "abcdef",
                "buildId": "build-1",
            },
        )

    plan = json.loads(portable_update_plan_path(portable_root).read_text(encoding="utf-8"))
    batch = portable_update_batch_path(portable_root).read_text(encoding="utf-8")

    assert result["status"] == "restart_pending"
    assert result["restartScheduled"] is True
    assert result["restartRequired"] is True
    assert result["canCancel"] is True
    assert result["portableUpdateScriptPath"] == str(portable_update_batch_path(portable_root))
    assert result["portableUpdatePowerShellPath"] is None
    assert plan["archivePath"] == str(archive_path.resolve())
    assert plan["cancelPath"] == str(portable_update_cancel_path(portable_root))
    assert plan["parentPid"] == 1234
    assert plan["waitTimeoutSeconds"] == 3600
    assert plan["autoTerminateParent"] is True
    assert plan["relaunchCommand"] == ["CAI.exe", "--no-tray"]
    assert not portable_update_script_path(portable_root).exists()
    assert "powershell.exe" in batch
    assert "CAI_PORTABLE_UPDATE_POWERSHELL_PAYLOAD" in batch
    assert "Stop-CaiUpdateIfCancelled" in batch
    assert "Start-CaiUpdateWindow" in batch
    assert "Assert-CaiPortableUpdateDiskSpace" in batch
    assert "Wait-CaiPortableRuntimeProcesses" in batch
    assert "Closing CAI to apply update" in batch
    assert "Invoke-CaiFileOperationWithRetry" in batch
    assert "Get-CaiPortableRuntimeProcesses" in batch
    assert "ExtractToDirectory" in batch
    assert "-WindowStyle Hidden" in batch
    assert "WindowStyle = \"Hidden\"" not in batch
    popen_mock.assert_called_once()


def test_cancel_pending_portable_update_writes_cancel_marker(tmp_path: Path) -> None:
    portable_root = tmp_path / "portable"
    archive_path = tmp_path / "portable-update.zip"
    _create_portable_root(portable_root, app_value="local", include_data=False)
    _create_portable_archive(archive_path, app_value="remote")

    with patch("cai_compute_chain.update_channel.subprocess.Popen"):
        schedule_portable_update_after_exit(
            portable_root,
            archive_path,
            relaunch_command=["CAI.exe"],
            parent_pid=1234,
        )

    result = cancel_pending_portable_update(portable_root)

    assert result["cancelled"] is True
    assert result["status"] == "cancelled"
    assert result["cancelRequested"] is True
    assert result["canCancel"] is False
    assert portable_update_cancel_path(portable_root).is_file()


def test_resume_pending_portable_update_reuses_downloaded_archive(tmp_path: Path) -> None:
    portable_root = tmp_path / "portable"
    archive_path = tmp_path / "portable-update.zip"
    _create_portable_root(portable_root, app_value="local", include_data=False)
    _create_portable_archive(archive_path, app_value="remote")

    with patch("cai_compute_chain.update_channel.subprocess.Popen"):
        schedule_portable_update_after_exit(
            portable_root,
            archive_path,
            relaunch_command=["CAI.exe"],
            parent_pid=1234,
            start_process=False,
        )

    with patch("cai_compute_chain.update_channel.subprocess.Popen") as popen_mock:
        popen_mock.return_value.pid = 4321
        result = resume_pending_portable_update_on_launch(
            portable_root,
            relaunch_command=["CAI.exe", "--language", "ru"],
            parent_pid=5678,
        )

    assert result is not None
    assert result["status"] == "restart_pending"
    assert result["restartScheduled"] is True
    assert result["portableUpdateApplyPid"] == 4321
    assert result["archivePath"] == str(archive_path.resolve())
    assert result["portableUpdateScriptPath"] == str(portable_update_batch_path(portable_root))
    assert portable_update_batch_path(portable_root).is_file()
    assert "already downloaded" in result["message"]
    plan = json.loads(portable_update_plan_path(portable_root).read_text(encoding="utf-8"))
    assert plan["archivePath"] == str(archive_path.resolve())
    assert plan["parentPid"] == 5678
    assert plan["relaunchCommand"] == ["CAI.exe", "--language", "ru"]
    popen_mock.assert_called_once()


def test_portable_auto_update_resumes_pending_archive_before_download(
    tmp_path: Path,
) -> None:
    portable_root = tmp_path / "portable"
    archive_path = tmp_path / "portable-update.zip"
    _create_portable_root(portable_root, app_value="local", include_data=False)
    _create_portable_archive(archive_path, app_value="remote")

    schedule_portable_update_after_exit(
        portable_root,
        archive_path,
        relaunch_command=["CAI.exe"],
        parent_pid=1234,
        start_process=False,
    )

    with patch(
        "cai_compute_chain.update_channel.fetch_remote_update_manifest",
        side_effect=AssertionError("should not fetch a manifest for a pending archive"),
    ), patch(
        "cai_compute_chain.update_channel._download_update_archive",
        side_effect=AssertionError("should not re-download a pending archive"),
    ), patch(
        "cai_compute_chain.update_channel.subprocess.Popen",
    ) as popen_mock:
        popen_mock.return_value.pid = 9876
        result = maybe_stage_portable_auto_update_on_launch(
            portable_root,
            relaunch_command=["CAI.exe"],
            base_url="http://validator:52415",
            parent_pid=4321,
        )

    assert result["status"] == "restart_pending"
    assert result["portableUpdateApplyPid"] == 9876
    assert result["archivePath"] == str(archive_path.resolve())
    popen_mock.assert_called_once()


def test_portable_auto_update_reports_download_progress(tmp_path: Path) -> None:
    portable_root = tmp_path / "portable"
    archive_path = tmp_path / "portable-update.zip"
    _create_portable_root(portable_root, app_value="local", include_data=False)
    _create_portable_archive(archive_path, app_value="remote")
    archive_size = archive_path.stat().st_size
    manifest = {
        "gitCommit": "portable-commit-123",
        "gitBranch": "release",
        "version": "9.9.9",
        "buildId": "portable-build-123",
        "installKind": "portable",
        "repoKind": "portable",
        "archiveUrl": archive_path.as_uri(),
        "archiveSha256": sha256_file(archive_path),
        "archiveSizeBytes": archive_size,
    }
    captured_statuses: list[dict[str, object]] = []
    original_write_status = update_channel_module._write_update_status

    def capture_status(repo_root: Path, payload: dict[str, object]) -> None:
        captured_statuses.append(dict(payload))
        original_write_status(repo_root, payload)

    with patch.dict("os.environ", {"CAI_UPDATE_CHANNEL": "validator"}, clear=False), patch(
        "cai_compute_chain.update_channel.fetch_remote_update_manifest",
        return_value=manifest,
    ), patch(
        "cai_compute_chain.update_channel._write_update_status",
        side_effect=capture_status,
    ):
        result = maybe_stage_portable_auto_update_on_launch(
            portable_root,
            relaunch_command=["CAI.exe"],
            base_url="http://validator:52415",
            start_process=False,
        )

    download_statuses = [
        status for status in captured_statuses if status.get("status") == "downloading"
    ]
    assert result["status"] == "restart_pending"
    assert download_statuses
    assert any(status.get("totalBytes") == archive_size for status in download_statuses)
    assert any(status.get("downloadedBytes") == archive_size for status in download_statuses)
    assert any(status.get("progress") == 20 for status in download_statuses)
    assert any(status.get("progress") == 65 for status in download_statuses)
    assert not all(status.get("progress") == 45 for status in download_statuses)


def test_download_update_archive_retries_short_bounded_range(tmp_path: Path) -> None:
    payload = b"abcdef"
    manifest = {
        "archiveUrl": "http://validator:52415/v1/cai/update-package.zip?install_kind=portable",
        "archiveSha256": hashlib.sha256(payload).hexdigest(),
        "archiveSizeBytes": len(payload),
    }
    ranges: list[str | None] = []

    class FakeHeaders(dict):
        def get(self, key: str, default: object = None) -> object:
            return super().get(key, default)

    class FakeResponse:
        def __init__(self, data: bytes, *, status: int) -> None:
            self._data = data
            self._offset = 0
            self.status = status
            self.headers = FakeHeaders({"Content-Length": str(len(data))})

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def getcode(self) -> int:
            return self.status

        def getheader(self, key: str) -> str | None:
            return self.headers.get(key)  # type: ignore[return-value]

        def read(self, size: int = -1) -> bytes:
            if self._offset >= len(self._data):
                return b""
            if size is None or size < 0:
                size = len(self._data) - self._offset
            end = min(len(self._data), self._offset + size)
            chunk = self._data[self._offset:end]
            self._offset = end
            return chunk

    def fake_urlopen(request: object, *, timeout: int) -> FakeResponse:
        ranges.append(request.get_header("Range"))  # type: ignore[attr-defined]
        if len(ranges) == 1:
            return FakeResponse(payload[:3], status=206)
        return FakeResponse(payload, status=206)

    with patch("cai_compute_chain.update_channel.urlopen", side_effect=fake_urlopen):
        archive_path = update_channel_module._download_update_archive(
            manifest,
            tmp_path,
            timeout_sec=5,
        )

    assert archive_path.read_bytes() == payload
    assert ranges == ["bytes=0-5", "bytes=0-5"]


def test_download_update_archive_resumes_existing_partial_with_bounded_range(
    tmp_path: Path,
) -> None:
    payload = b"abcdef"
    partial_path = tmp_path / "update-package.zip.tmp"
    partial_path.write_bytes(payload[:3])
    manifest = {
        "archiveUrl": "http://validator:52415/v1/cai/update-package.zip?install_kind=portable",
        "archiveSha256": hashlib.sha256(payload).hexdigest(),
        "archiveSizeBytes": len(payload),
    }
    ranges: list[str | None] = []

    class FakeHeaders(dict):
        def get(self, key: str, default: object = None) -> object:
            return super().get(key, default)

    class FakeResponse:
        def __init__(self, data: bytes) -> None:
            self._data = data
            self._offset = 0
            self.status = 206
            self.headers = FakeHeaders({"Content-Length": str(len(data))})

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def getcode(self) -> int:
            return self.status

        def getheader(self, key: str) -> str | None:
            return self.headers.get(key)  # type: ignore[return-value]

        def read(self, size: int = -1) -> bytes:
            if self._offset >= len(self._data):
                return b""
            if size is None or size < 0:
                size = len(self._data) - self._offset
            end = min(len(self._data), self._offset + size)
            chunk = self._data[self._offset:end]
            self._offset = end
            return chunk

    def fake_urlopen(request: object, *, timeout: int) -> FakeResponse:
        ranges.append(request.get_header("Range"))  # type: ignore[attr-defined]
        return FakeResponse(payload[3:])

    with patch("cai_compute_chain.update_channel.urlopen", side_effect=fake_urlopen):
        archive_path = update_channel_module._download_update_archive(
            manifest,
            tmp_path,
            timeout_sec=5,
        )

    assert archive_path.read_bytes() == payload
    assert ranges == ["bytes=3-5"]


def test_portable_auto_update_defers_until_interface_idle(tmp_path: Path) -> None:
    portable_root = tmp_path / "portable"
    archive_path = tmp_path / "portable-update.zip"
    _create_portable_root(portable_root, app_value="local", include_data=False)
    _create_portable_archive(archive_path, app_value="remote")
    manifest = {
        "gitCommit": "portable-commit-123",
        "gitBranch": "release",
        "version": "9.9.9",
        "buildId": "portable-build-123",
        "installKind": "portable",
        "repoKind": "portable",
        "archiveUrl": archive_path.as_uri(),
        "archiveSha256": sha256_file(archive_path),
        "archiveSizeBytes": archive_path.stat().st_size,
    }
    record_portable_update_activity(
        portable_root,
        source="test",
        active_request_count=1,
        user_active=True,
    )

    with patch.dict("os.environ", {"CAI_UPDATE_CHANNEL": "validator"}, clear=False), patch(
        "cai_compute_chain.update_channel.fetch_remote_update_manifest",
        return_value=manifest,
    ), patch(
        "cai_compute_chain.update_channel._download_update_archive",
        side_effect=AssertionError("should not download while interface is busy"),
    ):
        result = maybe_stage_portable_auto_update_on_launch(
            portable_root,
            relaunch_command=["CAI.exe"],
            base_url="http://validator:52415",
            start_process=False,
            idle_seconds=60,
            idle_timeout_sec=1,
        )

    snapshot = portable_update_activity_snapshot(portable_root, idle_seconds=60)
    assert result["status"] == "deferred"
    assert result["phase"] == "waiting_for_idle"
    assert result["activity"]["activeRequestCount"] == 1
    assert snapshot["idle"] is False


def test_apply_update_archive_replaces_tracked_files_and_generated_build(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote"
    local_root = tmp_path / "local"
    _create_source_repo(
        remote_root,
        app_value="remote",
        build_value="<html>remote</html>\n",
        include_obsolete=False,
    )
    _create_source_repo(
        local_root,
        app_value="local",
        build_value="<html>local</html>\n",
        include_obsolete=True,
    )

    archive_path = build_update_archive(remote_root)
    result = apply_update_archive(local_root, archive_path)

    assert result["message"] == "CAI source checkout updated from validator package."
    assert (local_root / "src" / "cai_compute_chain" / "app.py").read_text(encoding="utf-8") == (
        "value = 'remote'\n"
    )
    assert (local_root / "cai" / "dashboard" / "build" / "index.html").read_text(
        encoding="utf-8"
    ) == "<html>remote</html>\n"
    assert not (local_root / "obsolete.txt").exists()


def test_apply_update_archive_writes_applied_rollback_marker(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote"
    local_root = tmp_path / "local"
    _create_source_repo(
        remote_root,
        app_value="remote",
        build_value="<html>remote</html>\n",
        include_obsolete=False,
    )
    _create_source_repo(
        local_root,
        app_value="local",
        build_value="<html>local</html>\n",
        include_obsolete=False,
    )

    archive_path = build_update_archive(remote_root)
    result = apply_update_archive(local_root, archive_path)
    marker = json.loads(update_rollback_marker_path(local_root).read_text(encoding="utf-8"))

    assert marker["status"] == "applied"
    assert marker["previousGitCommit"]
    assert marker["packageGitCommit"] == result["localGitCommit"]
    assert marker["writtenFileCount"] == result["writtenFileCount"]
    assert not update_stage_dir(local_root).exists()


def test_apply_update_archive_keeps_failed_rollback_marker(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _create_source_repo(
        repo_root,
        app_value="local",
        build_value="<html>local</html>\n",
        include_obsolete=False,
    )
    archive_path = tmp_path / "broken-update.zip"
    metadata = {
        "packageFormatVersion": 1,
        "protocolVersion": 1,
        "minCompatibleProtocolVersion": 1,
        "maxCompatibleProtocolVersion": 1,
        "gitCommit": "broken-package",
        "trackedFiles": ["src/cai_compute_chain/app.py"],
        "generatedRoots": [],
        "files": [{"path": "src/cai_compute_chain/app.py", "mode": 0o644}],
    }
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(PACKAGE_METADATA_PATH, json.dumps(metadata))

    try:
        apply_update_archive(repo_root, archive_path)
    except UpdateError as exc:
        assert "Archive payload is missing" in str(exc)
    else:
        raise AssertionError("Expected broken update package to fail during apply.")

    marker = json.loads(update_rollback_marker_path(repo_root).read_text(encoding="utf-8"))
    assert marker["status"] == "rolled_back"
    assert marker["packageGitCommit"] == "broken-package"
    assert "Archive payload is missing" in marker["error"]
    assert update_stage_dir(repo_root).exists()


def test_apply_update_archive_rolls_back_partial_source_update_failure(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _create_source_repo(
        repo_root,
        app_value="local",
        build_value="<html>local</html>\n",
        include_obsolete=False,
    )
    archive_path = tmp_path / "partial-update.zip"
    metadata = {
        "packageFormatVersion": 1,
        "protocolVersion": 1,
        "minCompatibleProtocolVersion": 1,
        "maxCompatibleProtocolVersion": 1,
        "gitCommit": "partial-package",
        "trackedFiles": [
            "src/cai_compute_chain/app.py",
            "src/cai_compute_chain/missing.py",
        ],
        "generatedRoots": [],
        "files": [
            {"path": "src/cai_compute_chain/app.py", "mode": 0o644},
            {"path": "src/cai_compute_chain/missing.py", "mode": 0o644},
        ],
    }
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(PACKAGE_METADATA_PATH, json.dumps(metadata))
        bundle.writestr("src/cai_compute_chain/app.py", "value = 'remote-before-fail'\n")

    try:
        apply_update_archive(repo_root, archive_path)
    except UpdateError as exc:
        assert "Archive payload is missing" in str(exc)
    else:
        raise AssertionError("Expected partial update package to fail during apply.")

    assert (repo_root / "src" / "cai_compute_chain" / "app.py").read_text(
        encoding="utf-8"
    ) == "value = 'local'\n"
    marker = json.loads(update_rollback_marker_path(repo_root).read_text(encoding="utf-8"))
    assert marker["status"] == "rolled_back"
    assert "src/cai_compute_chain/app.py" in marker["backedUpFiles"]
    assert update_rollback_backup_dir(repo_root).exists()


def test_apply_portable_update_archive_preserves_data_home(tmp_path: Path) -> None:
    portable_root = tmp_path / "portable"
    archive_path = tmp_path / "portable-update.zip"
    _create_portable_root(portable_root, app_value="local", include_data=True)
    (portable_root / "obsolete.txt").write_text("obsolete\n", encoding="utf-8")
    _create_portable_archive(archive_path, app_value="remote")

    result = apply_portable_update_archive(portable_root, archive_path)

    assert result["restartRequired"] is True
    assert (portable_root / "CAI.exe").read_text(encoding="utf-8") == "exe=remote\n"
    assert (portable_root / "_internal" / "app.txt").read_text(encoding="utf-8") == "app=remote\n"
    assert not (portable_root / "obsolete.txt").exists()
    assert (
        portable_root / "data" / ".cai-local" / "wallets.json"
    ).read_text(encoding="utf-8") == "[]\n"
    marker = json.loads(update_rollback_marker_path(portable_root).read_text(encoding="utf-8"))
    assert marker["status"] == "applied"
    assert marker["updateKind"] == "portable"
    assert not update_stage_dir(portable_root).exists()


def test_apply_portable_update_archive_accepts_wrapped_payload_root(tmp_path: Path) -> None:
    portable_root = tmp_path / "portable"
    archive_path = tmp_path / "portable-update.zip"
    _create_portable_root(portable_root, app_value="local", include_data=False)
    _create_portable_archive(archive_path, app_value="remote", wrapped=True)

    apply_portable_update_archive(portable_root, archive_path)

    assert (portable_root / "CAI.exe").read_text(encoding="utf-8") == "exe=remote\n"


def test_build_update_manifest_includes_hash_and_protocol_range(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _create_source_repo(
        repo_root,
        app_value="local",
        build_value="<html>local</html>\n",
        include_obsolete=False,
    )

    manifest = build_update_manifest(repo_root, base_url="http://validator:52415")

    assert manifest["protocolVersion"] == 1
    assert manifest["minCompatibleProtocolVersion"] == 1
    assert manifest["maxCompatibleProtocolVersion"] == 1
    assert len(manifest["archiveSha256"]) == 64
    assert manifest["archiveSizeBytes"] > 0


def test_build_update_manifest_can_use_configured_source_update_root(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime-snapshot"
    (runtime_root / "src" / "cai_compute_chain").mkdir(parents=True, exist_ok=True)
    (runtime_root / "cai" / "src" / "cai").mkdir(parents=True, exist_ok=True)
    (runtime_root / "cai" / "src" / "cai" / "main.py").write_text(
        "def main():\n    return 0\n",
        encoding="utf-8",
    )

    source_root = tmp_path / "source-checkout"
    _create_source_repo(
        source_root,
        app_value="source",
        build_value="<html>source</html>\n",
        include_obsolete=False,
    )

    with patch.dict("os.environ", {"CAI_UPDATE_SOURCE_ROOT": str(source_root)}, clear=False):
        manifest = build_update_manifest(runtime_root, base_url="http://validator:52415")
        manifest_archive_path = next((source_root / ".cai-update-cache").glob("cai-update-*.zip"))
        manifest_archive_sha = sha256_file(manifest_archive_path)
        archive_path = build_update_package(runtime_root)

    package_metadata = json.loads(
        zipfile.ZipFile(archive_path).read(PACKAGE_METADATA_PATH).decode("utf-8")
    )
    assert manifest["installKind"] == "source"
    assert manifest["gitCommit"] == package_metadata["gitCommit"]
    assert manifest["trackedFileCount"] == len(package_metadata["trackedFiles"])
    assert manifest["archiveSha256"] == manifest_archive_sha
    assert archive_path.parent == source_root / ".cai-update-cache"


def test_build_update_manifest_can_use_configured_source_artifact(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime-snapshot"
    (runtime_root / "src" / "cai_compute_chain").mkdir(parents=True, exist_ok=True)
    (runtime_root / "cai" / "src" / "cai").mkdir(parents=True, exist_ok=True)
    (runtime_root / "cai" / "src" / "cai" / "main.py").write_text(
        "def main():\n    return 0\n",
        encoding="utf-8",
    )
    source_root = tmp_path / "source-checkout"
    _create_source_repo(
        source_root,
        app_value="source",
        build_value="<html>source</html>\n",
        include_obsolete=False,
    )
    archive_path = build_update_archive(source_root)

    with patch.dict(
        "os.environ",
        {"CAI_UPDATE_SOURCE_ARTIFACT": str(archive_path)},
        clear=False,
    ):
        manifest = build_update_manifest(runtime_root, base_url="http://validator:52415")
        package_path = build_update_package(runtime_root)

    package_metadata = json.loads(
        zipfile.ZipFile(archive_path).read(PACKAGE_METADATA_PATH).decode("utf-8")
    )
    assert package_path == archive_path
    assert manifest["installKind"] == "source"
    assert manifest["gitCommit"] == package_metadata["gitCommit"]
    assert manifest["archiveSha256"] == sha256_file(archive_path)
    assert manifest["archiveSizeBytes"] == archive_path.stat().st_size


def test_build_portable_update_manifest_uses_portable_artifact(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _create_source_repo(
        repo_root,
        app_value="local",
        build_value="<html>local</html>\n",
        include_obsolete=False,
    )
    dist_dir = repo_root / ".dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dist_dir / "CAI-portable.zip"
    _create_portable_archive(archive_path, app_value="portable-remote")
    (dist_dir / "release-metadata.json").write_text(
        json.dumps(
            {
                "version": "9.9.9",
                "gitCommit": "portable-commit",
                "gitBranch": "release",
                "buildId": "portable-build",
            }
        ),
        encoding="utf-8",
    )

    manifest = build_update_manifest(
        repo_root,
        base_url="http://validator:52415",
        install_kind="portable",
    )

    assert manifest["installKind"] == "portable"
    assert manifest["repoKind"] == "portable"
    assert manifest["archiveUrl"] == "http://validator:52415/v1/cai/update-package.zip?install_kind=portable"
    assert manifest["archiveSha256"] == sha256_file(archive_path)
    assert manifest["archiveSizeBytes"] == archive_path.stat().st_size
    assert manifest["version"] == "9.9.9"
    assert manifest["gitCommit"] == "portable-commit"
    assert manifest["buildId"] == "portable-build"


def test_build_portable_update_manifest_uses_cached_manifest_without_rehashing(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    _create_source_repo(
        repo_root,
        app_value="local",
        build_value="<html>local</html>\n",
        include_obsolete=False,
    )
    dist_dir = repo_root / ".dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dist_dir / "CAI-portable.zip"
    _create_portable_archive(archive_path, app_value="portable-remote")
    cached_sha256 = "1" * 64
    (dist_dir / "portable-update-manifest.json").write_text(
        json.dumps(
            {
                "manifestVersion": 1,
                "protocolVersion": 1,
                "minCompatibleProtocolVersion": 1,
                "maxCompatibleProtocolVersion": 1,
                "channel": "validator",
                "version": "9.9.9",
                "gitCommit": "portable-commit",
                "gitBranch": "release",
                "gitDirty": False,
                "buildId": "portable-build",
                "archiveUrl": "http://old-validator/v1/cai/update-package.zip?install_kind=portable",
                "archiveSha256": cached_sha256,
                "archiveSizeBytes": archive_path.stat().st_size,
                "repoKind": "portable",
                "installKind": "portable",
            }
        ),
        encoding="utf-8",
    )

    with patch("cai_compute_chain.update_channel.sha256_file") as sha256_mock:
        manifest = build_update_manifest(
            repo_root,
            base_url="http://validator:52415",
            install_kind="portable",
        )

    sha256_mock.assert_not_called()
    assert manifest["archiveSha256"] == cached_sha256
    assert manifest["archiveSizeBytes"] == archive_path.stat().st_size
    assert (
        manifest["archiveUrl"]
        == "http://validator:52415/v1/cai/update-package.zip?install_kind=portable"
    )
    assert manifest["installKind"] == "portable"
    assert manifest["buildId"] == "portable-build"


def test_download_archive_name_adds_zip_suffix_for_extensionless_endpoint() -> None:
    name = update_channel_module._download_archive_name(
        {
            "buildId": "0.1.0-0007-gabcdef-20260514T050000Z",
            "archiveSha256": "a" * 64,
        },
        "/v1/cai/update-package",
    )

    assert name == "0.1.0-0007-gabcdef-20260514T050000Z.zip"


def test_build_portable_update_manifest_accepts_bom_release_metadata(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _create_source_repo(
        repo_root,
        app_value="local",
        build_value="<html>local</html>\n",
        include_obsolete=False,
    )
    dist_dir = repo_root / ".dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    _create_portable_archive(dist_dir / "CAI-portable.zip", app_value="portable-remote")
    metadata_bytes = json.dumps(
        {
            "version": "9.9.9",
            "gitCommit": "portable-commit",
            "gitBranch": "release",
            "buildId": "portable-build",
        }
    ).encode("utf-8")
    (dist_dir / "release-metadata.json").write_bytes(b"\xef\xbb\xbf" + metadata_bytes)

    manifest = build_update_manifest(
        repo_root,
        base_url="http://validator:52415",
        install_kind="portable",
    )

    assert manifest["version"] == "9.9.9"
    assert manifest["gitCommit"] == "portable-commit"
    assert manifest["buildId"] == "portable-build"


def test_build_update_manifest_signs_when_release_seed_is_configured(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _create_source_repo(
        repo_root,
        app_value="local",
        build_value="<html>local</html>\n",
        include_obsolete=False,
    )
    signing_seed = generate_signing_seed()
    public_key_b64 = public_key_b64_from_seed(signing_seed)

    with patch.dict(
        "os.environ",
        {
            "CAI_UPDATE_SIGNING_SEED_B64": encode_bytes(signing_seed),
            "CAI_UPDATE_TRUSTED_PUBLIC_KEYS_B64": public_key_b64,
        },
        clear=False,
    ):
        manifest = build_update_manifest(repo_root, base_url="http://validator:52415")
        validate_update_manifest(manifest, require_archive_hash=True)

    assert (
        manifest["signature"]["scheme"]
        == "cai-update-manifest-hybrid-ed25519-ml-dsa-65-v1"
    )
    assert manifest["signature"]["public_key_b64"] == public_key_b64
    assert manifest["signature"]["pq_scheme"] == "ml-dsa-65-v1"
    assert manifest["signature"]["pq_public_key_b64"]
    assert manifest["signature"]["pq_signature_b64"]


def test_validate_update_manifest_rejects_unsigned_when_signature_required() -> None:
    try:
        validate_update_manifest(
            {
                "archiveUrl": "http://validator:52415/v1/cai/update-package",
                "archiveSha256": "0" * 64,
            },
            require_archive_hash=True,
            require_signature=True,
        )
    except UpdateError as exc:
        assert "signature is missing" in str(exc)
    else:
        raise AssertionError("Expected unsigned update manifest to be rejected.")


def test_validate_update_manifest_accepts_legacy_ed25519_signature() -> None:
    signing_seed = generate_signing_seed()
    public_key_b64 = public_key_b64_from_seed(signing_seed)
    manifest = {
        "archiveUrl": "http://validator:52415/v1/cai/update-package",
        "archiveSha256": "0" * 64,
    }
    manifest["signature"] = {
        "scheme": "cai-update-manifest-ed25519-v1",
        "public_key_b64": public_key_b64,
        "public_key_address": address_from_public_key_b64(public_key_b64),
        "signature_b64": sign_payload_b64(
            signing_seed,
            update_channel_module.update_manifest_signing_body(manifest),
        ),
        "signed_at": "2026-05-02T00:00:00+00:00",
    }

    validate_update_manifest(
        manifest,
        require_archive_hash=True,
        require_signature=True,
    )


def test_validate_update_manifest_rejects_tampered_signature() -> None:
    signing_seed = generate_signing_seed()
    signed_manifest = sign_update_manifest(
        {
            "archiveUrl": "http://validator:52415/v1/cai/update-package",
            "archiveSha256": "0" * 64,
        },
        signing_seed_b64=encode_bytes(signing_seed),
        signed_at="2026-05-02T00:00:00+00:00",
    )
    signed_manifest["archiveSha256"] = "1" * 64

    try:
        validate_update_manifest(
            signed_manifest,
            require_archive_hash=True,
            require_signature=True,
        )
    except UpdateError as exc:
        assert "signature is invalid" in str(exc)
    else:
        raise AssertionError("Expected tampered update manifest signature to be rejected.")


def test_validate_update_manifest_rejects_untrusted_signature_key() -> None:
    signing_seed = generate_signing_seed()
    trusted_seed = generate_signing_seed()
    signed_manifest = sign_update_manifest(
        {
            "archiveUrl": "http://validator:52415/v1/cai/update-package",
            "archiveSha256": "0" * 64,
        },
        signing_seed_b64=encode_bytes(signing_seed),
        signed_at="2026-05-02T00:00:00+00:00",
    )

    with patch.dict(
        "os.environ",
        {"CAI_UPDATE_TRUSTED_PUBLIC_KEYS_B64": public_key_b64_from_seed(trusted_seed)},
        clear=False,
    ):
        try:
            validate_update_manifest(signed_manifest, require_archive_hash=True)
        except UpdateError as exc:
            assert "key is not trusted" in str(exc)
        else:
            raise AssertionError("Expected update manifest signed by an untrusted key to be rejected.")


def test_apply_remote_update_rejects_validator_manifest_without_hash(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _create_source_repo(
        repo_root,
        app_value="local",
        build_value="<html>local</html>\n",
        include_obsolete=False,
    )

    with patch.dict("os.environ", {"CAI_UPDATE_CHANNEL": "validator"}, clear=False), patch(
        "cai_compute_chain.update_channel.fetch_remote_update_manifest",
        return_value={
            "gitCommit": "remote-commit",
            "gitBranch": "main",
            "version": "0.1.0",
            "archiveUrl": "http://validator:52415/v1/cai/update-package",
        },
    ), patch("cai_compute_chain.update_channel._download_update_archive") as download_mock:
        try:
            apply_remote_update(
                repo_root,
                base_url="http://validator:52415",
                timeout_sec=1,
            )
        except UpdateError as exc:
            assert "archiveSha256" in str(exc)
        else:
            raise AssertionError("Expected validator update manifest without hash to be rejected.")

    download_mock.assert_not_called()


def test_apply_remote_update_e2e_clean_node_from_validator_package(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote"
    local_root = tmp_path / "local"
    _create_source_repo(
        remote_root,
        app_value="remote",
        build_value="<html>remote</html>\n",
        include_obsolete=False,
    )
    _create_source_repo(
        local_root,
        app_value="local",
        build_value="<html>local</html>\n",
        include_obsolete=False,
    )
    manifest = build_update_manifest(remote_root, base_url="http://validator:52415")
    archive_path = next((remote_root / ".cai-update-cache").glob("cai-update-*.zip"))
    manifest["archiveUrl"] = archive_path.as_uri()

    with patch.dict("os.environ", {"CAI_UPDATE_CHANNEL": "validator"}, clear=False), patch(
        "cai_compute_chain.update_channel.fetch_remote_update_manifest",
        return_value=manifest,
    ):
        result = apply_remote_update(
            local_root,
            base_url="http://validator:52415",
            timeout_sec=5,
        )

    assert result["updated"] is True
    assert result["status"] == "updated"
    assert (local_root / "src" / "cai_compute_chain" / "app.py").read_text(
        encoding="utf-8"
    ) == "value = 'remote'\n"


def test_apply_remote_update_e2e_clean_portable_from_validator_package(tmp_path: Path) -> None:
    portable_root = tmp_path / "portable"
    archive_path = tmp_path / "portable-update.zip"
    _create_portable_root(portable_root, app_value="local", include_data=True)
    _create_portable_archive(archive_path, app_value="remote")
    manifest = {
        "gitCommit": "portable-commit-123",
        "gitBranch": "release",
        "version": "9.9.9",
        "installKind": "portable",
        "repoKind": "portable",
        "archiveUrl": archive_path.as_uri(),
        "archiveSha256": sha256_file(archive_path),
        "archiveSizeBytes": archive_path.stat().st_size,
    }

    with patch.dict("os.environ", {"CAI_UPDATE_CHANNEL": "validator"}, clear=False), patch(
        "cai_compute_chain.update_channel.fetch_remote_update_manifest",
        return_value=manifest,
    ):
        result = apply_remote_update(
            portable_root,
            base_url="http://validator:52415",
            timeout_sec=5,
        )

    assert result["updated"] is True
    assert result["status"] == "updated"
    assert result["installKind"] == "portable"
    assert result["restartRequired"] is True
    assert (portable_root / "CAI.exe").read_text(encoding="utf-8") == "exe=remote\n"


def test_validate_update_manifest_rejects_incompatible_protocol() -> None:
    try:
        validate_update_manifest(
            {
                "archiveUrl": "http://validator:52415/v1/cai/update-package",
                "archiveSha256": "0" * 64,
                "protocolVersion": 2,
                "minCompatibleProtocolVersion": 2,
                "maxCompatibleProtocolVersion": 3,
            },
            require_archive_hash=True,
        )
    except UpdateError as exc:
        assert "requires update protocol" in str(exc)
    else:
        raise AssertionError("Expected incompatible update protocol range to be rejected.")


def test_apply_update_archive_keeps_local_runtime_state_even_if_tracked(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote"
    local_root = tmp_path / "local"
    _create_source_repo(
        remote_root,
        app_value="remote",
        build_value="<html>remote</html>\n",
        include_obsolete=False,
    )
    _create_source_repo(
        local_root,
        app_value="local",
        build_value="<html>local</html>\n",
        include_obsolete=False,
    )
    (local_root / "data").mkdir(parents=True, exist_ok=True)
    (local_root / "data" / "wallets.json").write_text("[]\n", encoding="utf-8")
    _run_git(local_root, "add", "data/wallets.json")
    _run_git(local_root, "commit", "-m", "add legacy runtime state")

    archive_path = build_update_archive(remote_root)
    apply_update_archive(local_root, archive_path)

    assert (local_root / "data" / "wallets.json").read_text(encoding="utf-8") == "[]\n"


def test_apply_update_archive_rejects_runtime_state_file_in_package(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _create_source_repo(
        repo_root,
        app_value="local",
        build_value="<html>local</html>\n",
        include_obsolete=False,
    )
    archive_path = tmp_path / "malicious-update.zip"
    metadata = {
        "packageFormatVersion": 1,
        "protocolVersion": 1,
        "minCompatibleProtocolVersion": 1,
        "maxCompatibleProtocolVersion": 1,
        "trackedFiles": [],
        "generatedRoots": [],
        "files": [{"path": "data/wallets.json", "mode": 0o644}],
    }
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(PACKAGE_METADATA_PATH, json.dumps(metadata))
        bundle.writestr("data/wallets.json", "[]\n")

    try:
        apply_update_archive(repo_root, archive_path)
    except UpdateError as exc:
        assert "sensitive runtime path" in str(exc)
    else:
        raise AssertionError("Expected update package with runtime state to be rejected.")


def test_check_for_updates_reports_dirty_checkout(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _create_source_repo(
        repo_root,
        app_value="local",
        build_value="<html>local</html>\n",
        include_obsolete=False,
    )
    (repo_root / "src" / "cai_compute_chain" / "app.py").write_text(
        "value = 'dirty'\n",
        encoding="utf-8",
    )

    with patch(
        "cai_compute_chain.update_channel.fetch_remote_update_manifest",
        return_value={
            "gitCommit": "remote-commit",
            "gitBranch": "main",
            "version": "0.1.0",
            "archiveSha256": "0" * 64,
        },
    ), patch(
        "cai_compute_chain.update_channel.resolve_update_base_url",
        return_value="http://validator:52415",
    ):
        result = check_for_updates(
            repo_root,
            base_url="http://validator:52415",
            timeout_sec=1,
        )

    assert result["updateAvailable"] is True
    assert result["canApply"] is False
    assert "uncommitted changes" in result["applyReason"]


def test_check_for_updates_supports_portable_install(tmp_path: Path) -> None:
    portable_root = tmp_path / "portable"
    _create_portable_root(portable_root, app_value="local", include_data=False)

    with patch(
        "cai_compute_chain.update_channel.fetch_remote_update_manifest",
        return_value={
            "gitCommit": "portable-remote-commit",
            "gitBranch": "release",
            "version": "9.9.9",
            "installKind": "portable",
            "repoKind": "portable",
            "archiveSha256": "0" * 64,
        },
    ), patch(
        "cai_compute_chain.update_channel.resolve_update_base_url",
        return_value="http://validator:52415",
    ):
        result = check_for_updates(
            portable_root,
            base_url="http://validator:52415",
            timeout_sec=1,
        )

    assert result["installKind"] == "portable"
    assert result["remoteInstallKind"] == "portable"
    assert result["updateAvailable"] is True
    assert result["canApply"] is True


def test_check_for_updates_marks_unversioned_portable_outdated_by_remote_build_id(
    tmp_path: Path,
) -> None:
    portable_root = tmp_path / "portable"
    _create_portable_root(portable_root, app_value="local", include_data=False)

    with patch(
        "cai_compute_chain.update_channel.fetch_remote_update_manifest",
        return_value={
            "gitCommit": "portable-remote-commit",
            "gitBranch": "release",
            "version": "0.1.0",
            "buildId": "portable-build-remote",
            "installKind": "portable",
            "repoKind": "portable",
            "archiveSha256": "0" * 64,
        },
    ), patch(
        "cai_compute_chain.update_channel.resolve_update_base_url",
        return_value="http://validator:52415",
    ):
        result = check_for_updates(
            portable_root,
            base_url="http://validator:52415",
            timeout_sec=1,
        )

    assert result["localBuildId"] is None
    assert result["remoteBuildId"] == "portable-build-remote"
    assert result["updateAvailable"] is True
    assert result["canApply"] is True


def test_check_for_updates_portable_prefers_validator_over_github_auto(tmp_path: Path) -> None:
    portable_root = tmp_path / "portable"
    _create_portable_root(portable_root, app_value="local", include_data=False)

    with patch.dict(
        "os.environ",
        {
            "CAI_UPDATE_CHANNEL": "auto",
            "CAI_UPDATE_GITHUB_REPOSITORY": "octo/example",
        },
        clear=False,
    ), patch(
        "cai_compute_chain.update_channel.fetch_remote_update_manifest",
        return_value={
            "gitCommit": "portable-remote-commit",
            "gitBranch": "release",
            "version": "9.9.9",
            "installKind": "portable",
            "repoKind": "portable",
            "archiveSha256": "0" * 64,
        },
    ) as remote_manifest_mock, patch(
        "cai_compute_chain.update_channel.fetch_github_update_manifest"
    ) as github_manifest_mock, patch(
        "cai_compute_chain.update_channel.resolve_update_base_url",
        return_value="http://validator:52415",
    ):
        result = check_for_updates(
            portable_root,
            base_url="http://validator:52415",
            timeout_sec=1,
        )

    assert result["channel"] == "validator"
    remote_manifest_mock.assert_called_once()
    github_manifest_mock.assert_not_called()


def test_update_archive_rejects_tracked_api_token(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _create_source_repo(
        repo_root,
        app_value="local",
        build_value="<html>local</html>\n",
        include_obsolete=False,
    )
    (repo_root / ".cai-api-token").write_text("secret-token\n", encoding="utf-8")
    _run_git(repo_root, "add", ".cai-api-token")
    _run_git(repo_root, "commit", "-m", "add token")

    try:
        build_update_archive(repo_root)
    except UpdateError as exc:
        assert ".cai-api-token" in str(exc)
    else:
        raise AssertionError("Expected update archive to reject tracked API token.")


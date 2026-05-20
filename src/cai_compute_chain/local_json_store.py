# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
import secrets
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_ATOMIC_REPLACE_RETRYABLE_WINERRORS = {5, 32}
_ATOMIC_REPLACE_RETRYABLE_ERRNOS = {13}
_ATOMIC_REPLACE_MAX_ATTEMPTS = 80
_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[str, threading.RLock] = {}


def _path_lock(path: Path) -> threading.RLock:
    try:
        key = str(path.expanduser().resolve(strict=False))
    except OSError:
        key = str(path)
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[key] = lock
        return lock


def _is_retryable_atomic_replace_error(exc: OSError) -> bool:
    winerror = getattr(exc, "winerror", None)
    if isinstance(winerror, int) and winerror in _ATOMIC_REPLACE_RETRYABLE_WINERRORS:
        return True
    errno_value = getattr(exc, "errno", None)
    return isinstance(errno_value, int) and errno_value in _ATOMIC_REPLACE_RETRYABLE_ERRNOS


def _replace_file_with_retry(temp_path: Path, path: Path) -> None:
    last_error: OSError | None = None
    for attempt in range(_ATOMIC_REPLACE_MAX_ATTEMPTS):
        try:
            os.replace(temp_path, path)
            return
        except OSError as exc:
            last_error = exc
            if not _is_retryable_atomic_replace_error(exc):
                raise
            if attempt >= _ATOMIC_REPLACE_MAX_ATTEMPTS - 1:
                raise
            time.sleep(min(0.02 * (attempt + 1), 0.25))
    if last_error is not None:
        raise last_error


def _overwrite_file_with_retry(temp_path: Path, path: Path) -> None:
    payload = temp_path.read_bytes()
    last_error: OSError | None = None
    for attempt in range(_ATOMIC_REPLACE_MAX_ATTEMPTS):
        try:
            path.write_bytes(payload)
            return
        except OSError as exc:
            last_error = exc
            if not _is_retryable_atomic_replace_error(exc):
                raise
            if attempt >= _ATOMIC_REPLACE_MAX_ATTEMPTS - 1:
                raise
            time.sleep(min(0.02 * (attempt + 1), 0.25))
    if last_error is not None:
        raise last_error


def atomic_write_text_file(path: Path, text: str) -> None:
    with _path_lock(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(
            f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
        )
        try:
            temp_path.write_text(text, encoding="utf-8")
            try:
                _replace_file_with_retry(temp_path, path)
            except OSError as exc:
                if not _is_retryable_atomic_replace_error(exc):
                    raise
                # Windows can keep read handles open without delete sharing.  In
                # that case os.replace keeps failing even though a normal write is
                # allowed; falling back avoids surfacing transient state-file
                # races as user-visible API failures.
                _overwrite_file_with_retry(temp_path, path)
        finally:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass


def atomic_write_json_array_file(path: Path, payload: list[Any]) -> None:
    atomic_write_text_file(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2),
    )


def atomic_write_json_object_file(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text_file(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2),
    )


def _corrupt_json_backup_path(path: Path) -> Path:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return path.with_name(
        f"{path.stem}.corrupt-{stamp}-{secrets.token_hex(4)}{path.suffix}"
    )


def _heal_file_if_unchanged(
    path: Path,
    *,
    expected_bytes: bytes,
    recovered_text: str,
) -> None:
    try:
        with _path_lock(path):
            if path.read_bytes() != expected_bytes:
                return
            backup_path = _corrupt_json_backup_path(path)
            backup_path.write_bytes(expected_bytes)
            atomic_write_text_file(path, recovered_text)
    except OSError:
        return


def _read_file_bytes(path: Path) -> bytes:
    with _path_lock(path):
        return path.read_bytes()


def read_json_array_file(path: Path, *, heal_corrupt: bool = False) -> list[Any]:
    if not path.exists():
        return []
    try:
        raw_bytes = _read_file_bytes(path)
    except OSError:
        return []
    if not raw_bytes:
        if heal_corrupt:
            _heal_file_if_unchanged(
                path,
                expected_bytes=raw_bytes,
                recovered_text="[]",
            )
        return []
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        if heal_corrupt:
            _heal_file_if_unchanged(
                path,
                expected_bytes=raw_bytes,
                recovered_text="[]",
            )
        return []
    if not text.strip():
        if heal_corrupt:
            _heal_file_if_unchanged(
                path,
                expected_bytes=raw_bytes,
                recovered_text="[]",
            )
        return []
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        stripped = text.lstrip("\ufeff\r\n\t \x00")
        recovered: list[Any] | None = None
        try:
            decoded, _index = json.JSONDecoder().raw_decode(stripped)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, list):
            recovered = decoded
        if recovered is None:
            recovered = []
        if heal_corrupt:
            _heal_file_if_unchanged(
                path,
                expected_bytes=raw_bytes,
                recovered_text=json.dumps(recovered, ensure_ascii=False, indent=2),
            )
        return recovered
    if not isinstance(raw, list):
        if heal_corrupt:
            _heal_file_if_unchanged(
                path,
                expected_bytes=raw_bytes,
                recovered_text="[]",
        )
        return []
    return raw


def read_json_object_file(path: Path, *, heal_corrupt: bool = False) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw_bytes = _read_file_bytes(path)
    except OSError:
        return {}
    if not raw_bytes:
        if heal_corrupt:
            _heal_file_if_unchanged(
                path,
                expected_bytes=raw_bytes,
                recovered_text="{}",
            )
        return {}
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        if heal_corrupt:
            _heal_file_if_unchanged(
                path,
                expected_bytes=raw_bytes,
                recovered_text="{}",
            )
        return {}
    if not text.strip():
        if heal_corrupt:
            _heal_file_if_unchanged(
                path,
                expected_bytes=raw_bytes,
                recovered_text="{}",
            )
        return {}
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        stripped = text.lstrip("\ufeff\r\n\t \x00")
        recovered: dict[str, Any] | None = None
        try:
            decoded, _index = json.JSONDecoder().raw_decode(stripped)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, dict):
            recovered = decoded
        if recovered is None:
            recovered = {}
        if heal_corrupt:
            _heal_file_if_unchanged(
                path,
                expected_bytes=raw_bytes,
                recovered_text=json.dumps(recovered, ensure_ascii=False, indent=2),
            )
        return recovered
    if not isinstance(raw, dict):
        if heal_corrupt:
            _heal_file_if_unchanged(
                path,
                expected_bytes=raw_bytes,
                recovered_text="{}",
            )
        return {}
    return raw


def read_jsonl_object_file(
    path: Path,
    *,
    heal_corrupt: bool = False,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        raw_bytes = _read_file_bytes(path)
    except OSError:
        return []
    if not raw_bytes:
        return []
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        if heal_corrupt:
            _heal_file_if_unchanged(
                path,
                expected_bytes=raw_bytes,
                recovered_text="",
            )
        return []

    items: list[dict[str, Any]] = []
    cleaned_lines: list[str] = []
    corrupted = False

    for line in text.splitlines():
        if not line.strip():
            continue
        sanitized = line.replace("\x00", "").strip()
        if not sanitized:
            corrupted = True
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            try:
                payload, _index = json.JSONDecoder().raw_decode(sanitized)
            except json.JSONDecodeError:
                corrupted = True
                continue
            corrupted = True
        if not isinstance(payload, dict):
            corrupted = True
            continue
        items.append(payload)
        cleaned_lines.append(json.dumps(payload, ensure_ascii=False))

    if heal_corrupt and corrupted:
        _heal_file_if_unchanged(
            path,
            expected_bytes=raw_bytes,
            recovered_text="\n".join(cleaned_lines) + ("\n" if cleaned_lines else ""),
        )
    return items

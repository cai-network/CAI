# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from http import HTTPStatus

from fastapi.responses import StreamingResponse

from cai.api.update_archive import (
    iter_cai_update_archive_range,
    parse_cai_update_archive_range,
    stream_cai_update_archive_response,
)


def test_parse_cai_update_archive_range_supports_open_and_suffix_ranges() -> None:
    assert parse_cai_update_archive_range(None, size=10) is None
    assert parse_cai_update_archive_range("bytes=2-5", size=10) == (2, 5)
    assert parse_cai_update_archive_range("bytes=7-", size=10) == (7, 9)
    assert parse_cai_update_archive_range("bytes=-4", size=10) == (6, 9)
    assert parse_cai_update_archive_range("bytes=0-99", size=10) == (0, 9)


def test_parse_cai_update_archive_range_rejects_invalid_ranges() -> None:
    invalid_headers = [
        "items=0-1",
        "bytes=",
        "bytes=0-1,2-3",
        "bytes=10-11",
        "bytes=5-2",
        "bytes=-0",
    ]

    for header in invalid_headers:
        try:
            parse_cai_update_archive_range(header, size=10)
        except ValueError:
            continue
        raise AssertionError(f"Expected invalid range to fail: {header}")


def test_iter_cai_update_archive_range_reads_only_requested_bytes(tmp_path) -> None:
    archive = tmp_path / "artifact.zip"
    archive.write_bytes(b"0123456789")

    assert b"".join(iter_cai_update_archive_range(archive, start=3, end=6)) == b"3456"


def test_stream_cai_update_archive_response_sets_range_headers(tmp_path) -> None:
    archive = tmp_path / "CAI-portable.zip"
    archive.write_bytes(b"0123456789")

    response = stream_cai_update_archive_response(
        archive,
        range_header="bytes=3-6",
    )

    assert isinstance(response, StreamingResponse)
    assert response.status_code == HTTPStatus.PARTIAL_CONTENT
    assert response.headers["content-length"] == "4"
    assert response.headers["content-range"] == "bytes 3-6/10"
    assert response.headers["accept-ranges"] == "bytes"
    assert 'filename="CAI-portable.zip"' in response.headers["content-disposition"]


def test_stream_cai_update_archive_response_returns_416_for_bad_range(tmp_path) -> None:
    archive = tmp_path / "artifact.zip"
    archive.write_bytes(b"0123456789")

    response = stream_cai_update_archive_response(
        archive,
        range_header="bytes=20-30",
    )

    assert response.status_code == HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE
    assert response.headers["content-range"] == "bytes */10"

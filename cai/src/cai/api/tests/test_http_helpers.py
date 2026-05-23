# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from io import BytesIO
from urllib.error import HTTPError
from unittest.mock import patch

from fastapi import HTTPException

from cai.api.http_helpers import (
    api_command_send_timeout_seconds,
    execution_cai_base_url,
    http_error_detail,
    load_json_url,
    raise_cai_transport_http_error,
)


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def _http_error(payload: bytes) -> HTTPError:
    return HTTPError(
        "http://node.local/test",
        400,
        "Bad Request",
        hdrs=None,
        fp=BytesIO(payload),
    )


def test_load_json_url_reads_json_payload() -> None:
    with patch(
        "cai.api.http_helpers.urlopen",
        return_value=_Response(b'{"worker": {"worker_enabled": true}}'),
    ) as urlopen_mock:
        payload = load_json_url("http://node.local/v1/cai/summary", timeout=7)

    assert payload == {"worker": {"worker_enabled": True}}
    urlopen_mock.assert_called_once_with(
        "http://node.local/v1/cai/summary",
        timeout=7,
    )


def test_http_error_detail_prefers_openai_error_message() -> None:
    detail = http_error_detail(
        _http_error(b'{"error": {"message": "model failed"}}')
    )

    assert detail == "model failed"


def test_http_error_detail_prefers_detail_field() -> None:
    detail = http_error_detail(_http_error(b'{"detail": "range failed"}'))

    assert detail == "range failed"


def test_api_command_send_timeout_seconds_parses_env_and_clamps() -> None:
    with patch.dict("os.environ", {"CAI_API_COMMAND_SEND_TIMEOUT_SECONDS": "0"}):
        assert api_command_send_timeout_seconds() == 0.1

    with patch.dict("os.environ", {"CAI_API_COMMAND_SEND_TIMEOUT_SECONDS": "2.5"}):
        assert api_command_send_timeout_seconds() == 2.5

    with patch.dict("os.environ", {"CAI_API_COMMAND_SEND_TIMEOUT_SECONDS": "bad"}):
        assert api_command_send_timeout_seconds() == 30.0


def test_execution_cai_base_url_uses_env_or_loopback_default() -> None:
    with patch.dict("os.environ", {}, clear=True):
        assert execution_cai_base_url(52415) == "http://127.0.0.1:52415"

    with patch.dict("os.environ", {"CAI_EXECUTION_CAI_URL": "http://node:1/"}):
        assert execution_cai_base_url(52415) == "http://node:1"


def test_raise_cai_transport_http_error_formats_http_exception() -> None:
    try:
        raise_cai_transport_http_error(
            RuntimeError("transport missing"),
            status_code=503,
            operation="dispatch",
        )
    except HTTPException as exc:
        assert exc.status_code == 503
        assert "dispatch" in str(exc.detail)
        assert "transport missing" in str(exc.detail)
        return
    raise AssertionError("Expected HTTPException")

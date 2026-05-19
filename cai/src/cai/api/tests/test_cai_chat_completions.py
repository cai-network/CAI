# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from io import BytesIO
from typing import Any
from urllib.error import HTTPError, URLError

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cai.api.main import API


def _make_api() -> Any:
    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api.cai_api_bearer_token = ""
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]
    app.post("/v1/cai/chat/completions", response_model=None)(api.cai_chat_completions)
    return api


def _fake_cai_service() -> Any:
    class _Service:
        def chat_completion(
            self, payload: dict[str, Any], **kwargs: Any
        ) -> dict[str, Any]:
            assert payload["model"] == "Qwen/Qwen3-0.6B-GGUF"
            return {
                "response": {
                    "id": "chatcmpl-cai-test",
                    "created": 1_777_777_777,
                    "model": payload["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "4",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
            }

    return _Service()


def test_cai_chat_completions_returns_json_response() -> None:
    api = _make_api()
    api._get_cai_service = _fake_cai_service  # pyright: ignore[reportPrivateUsage]
    client = TestClient(api.app)

    response = client.post(
        "/v1/cai/chat/completions",
        json={
            "model": "Qwen/Qwen3-0.6B-GGUF",
            "messages": [{"role": "user", "content": "2+2=?"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "chatcmpl-cai-test"
    assert payload["choices"][0]["message"]["content"] == "4"


def test_cai_chat_completions_streams_sse_response() -> None:
    api = _make_api()
    api._get_cai_service = _fake_cai_service  # pyright: ignore[reportPrivateUsage]
    client = TestClient(api.app)

    with client.stream(
        "POST",
        "/v1/cai/chat/completions",
        json={
            "model": "Qwen/Qwen3-0.6B-GGUF",
            "messages": [{"role": "user", "content": "2+2=?"}],
            "stream": True,
        },
    ) as response:
        chunks = "".join(response.iter_text())

    assert response.status_code == 200
    assert "data:" in chunks
    assert "\"content\":\"4\"" in chunks
    assert "[DONE]" in chunks


def test_cai_chat_completions_streams_cai_execution_event() -> None:
    api = _make_api()

    class _Service:
        def chat_completion(
            self, payload: dict[str, Any], **kwargs: Any
        ) -> dict[str, Any]:
            return {
                "response": {
                    "id": "chatcmpl-cai-test",
                    "created": 1_777_777_777,
                    "model": payload["model"],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "total_tokens": 4,
                    },
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "4"},
                            "finish_reason": "stop",
                        }
                    ],
                },
                "job": {
                    "jobId": "job-1",
                    "receiptId": "receipt-1",
                    "settlementId": "settlement-1",
                    "executionAttempts": [
                        {
                            "attempt": 1,
                            "status": "retrying",
                            "message": "timed out",
                        },
                        {
                            "attempt": 2,
                            "status": "completed",
                        },
                    ],
                    "executionAttemptStatus": {
                        "attempt": 2,
                        "maxAttempts": 3,
                        "status": "completed",
                    },
                },
                "receipt": {
                    "receiptId": "receipt-1",
                    "jobId": "job-1",
                    "decentralizedChainAudit": {
                        "executorNodeIds": ["node-a", "node-b"],
                        "participantNodeIds": ["node-a", "node-b"],
                        "proof": {
                            "executed": True,
                            "verified": True,
                            "sessionId": "session-1",
                            "finalOutputBatchCount": 1,
                        },
                        "reward": {
                            "payoutCount": 2,
                            "payoutNodes": ["node-a", "node-b"],
                            "workerPayoutTotalAtomic": 100,
                        },
                    },
                    "networkAudit": {
                        "rewardPayoutSource": "cai_owned_transport_shard_receipts",
                        "rewardPayoutNodeIds": ["node-a", "node-b"],
                    },
                },
                "settlement": {
                    "settlementId": "settlement-1",
                    "status": "applied",
                    "chainRecorded": True,
                    "chainTransactionCount": 6,
                },
                "payouts": [
                    {"nodeId": "node-a", "status": "credited_local_wallet"},
                    {"nodeId": "node-b", "status": "credited_local_wallet"},
                ],
            }

    api._get_cai_service = lambda: _Service()  # pyright: ignore[reportPrivateUsage]
    client = TestClient(api.app)

    with client.stream(
        "POST",
        "/v1/cai/chat/completions",
        json={
            "model": "Qwen/Qwen3-0.6B-GGUF",
            "messages": [{"role": "user", "content": "2+2=?"}],
            "stream": True,
        },
    ) as response:
        chunks = "".join(response.iter_text())

    assert response.status_code == 200
    assert ": cai_execution " in chunks
    assert '"prompt_tokens_details"' in chunks
    assert '"proofVerified":true' in chunks
    assert '"settlementStatus":"applied"' in chunks
    assert '"executionAttemptCount":2' in chunks
    assert '"executionAttemptStatus":{"attempt":2' in chunks
    assert '"rewardPayoutSource":"cai_owned_transport_shard_receipts"' in chunks


def test_cai_chat_completions_rejects_missing_bearer_token() -> None:
    api = _make_api()
    api.cai_api_bearer_token = "super-secret-token"
    api._get_cai_service = _fake_cai_service  # pyright: ignore[reportPrivateUsage]
    client = TestClient(api.app)

    response = client.post(
        "/v1/cai/chat/completions",
        json={
            "model": "Qwen/Qwen3-0.6B-GGUF",
            "messages": [{"role": "user", "content": "2+2=?"}],
            "stream": False,
        },
    )

    assert response.status_code == 401


def test_cai_chat_completions_accepts_valid_bearer_token() -> None:
    api = _make_api()
    api.cai_api_bearer_token = "super-secret-token"
    api._get_cai_service = _fake_cai_service  # pyright: ignore[reportPrivateUsage]
    client = TestClient(api.app)

    response = client.post(
        "/v1/cai/chat/completions",
        headers={"Authorization": "Bearer super-secret-token"},
        json={
            "model": "Qwen/Qwen3-0.6B-GGUF",
            "messages": [{"role": "user", "content": "2+2=?"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "4"


def test_cai_chat_completions_maps_timeout_to_503() -> None:
    api = _make_api()

    class _Service:
        def chat_completion(
            self, payload: dict[str, Any], **kwargs: Any
        ) -> dict[str, Any]:
            raise TimeoutError(
                "Timed out waiting for ready CAI instance for model Qwen/Qwen3-0.6B-GGUF."
            )

    api._get_cai_service = lambda: _Service()  # pyright: ignore[reportPrivateUsage]
    client = TestClient(api.app)

    response = client.post(
        "/v1/cai/chat/completions",
        json={
            "model": "Qwen/Qwen3-0.6B-GGUF",
            "messages": [{"role": "user", "content": "2+2=?"}],
            "stream": False,
        },
    )

    assert response.status_code == 503
    payload = response.json()
    assert "Timed out waiting for ready CAI instance" in payload["error"]["message"]


def test_cai_chat_completions_maps_urlopen_error_to_503() -> None:
    api = _make_api()

    class _Service:
        def chat_completion(
            self, payload: dict[str, Any], **kwargs: Any
        ) -> dict[str, Any]:
            raise URLError("[WinError 10060] connection attempt failed")

    api._get_cai_service = lambda: _Service()  # pyright: ignore[reportPrivateUsage]
    client = TestClient(api.app)

    response = client.post(
        "/v1/cai/chat/completions",
        json={
            "model": "Qwen/Qwen3-0.6B-GGUF",
            "messages": [{"role": "user", "content": "2+2=?"}],
            "stream": False,
        },
    )

    assert response.status_code == 503
    assert "WinError 10060" in response.json()["error"]["message"]


def test_cai_chat_completions_preserves_http_error_detail() -> None:
    api = _make_api()

    class _Service:
        def chat_completion(
            self, payload: dict[str, Any], **kwargs: Any
        ) -> dict[str, Any]:
            raise HTTPError(
                url="http://127.0.0.1:52415/instance/placement",
                code=400,
                msg="Bad Request",
                hdrs=None,
                fp=BytesIO(
                    b'{"error":{"message":"No usable placement found for private model."}}'
                ),
            )

    api._get_cai_service = lambda: _Service()  # pyright: ignore[reportPrivateUsage]
    client = TestClient(api.app)

    response = client.post(
        "/v1/cai/chat/completions",
        json={
            "model": "Qwen/Qwen3-0.6B-GGUF",
            "messages": [{"role": "user", "content": "2+2=?"}],
            "stream": False,
        },
    )

    assert response.status_code == 400
    assert "No usable placement" in response.json()["error"]["message"]


def test_cai_chat_completions_maps_runtime_error_to_503() -> None:
    api = _make_api()

    class _Service:
        def chat_completion(
            self, payload: dict[str, Any], **kwargs: Any
        ) -> dict[str, Any]:
            raise RuntimeError(
                "CAI-owned task-level transport was required, but no usable "
                "requester/executor route was available."
            )

    api._get_cai_service = lambda: _Service()  # pyright: ignore[reportPrivateUsage]
    client = TestClient(api.app)

    response = client.post(
        "/v1/cai/chat/completions",
        json={
            "model": "Qwen/Qwen3-0.6B-GGUF",
            "messages": [{"role": "user", "content": "2+2=?"}],
            "stream": False,
        },
    )

    assert response.status_code == 503
    assert "task-level transport" in response.json()["error"]["message"]


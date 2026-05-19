# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cai.api.main import API
from cai.shared.types.chunks import TokenChunk


def _make_api() -> API:
    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]
    app.post("/v1/chat/completions", response_model=None)(api.chat_completions)
    return api


def test_local_chat_completions_nonstream_returns_json_response() -> None:
    api = _make_api()

    async def _resolve_model(model_id):
        return model_id

    async def _send_command(_task_params):
        return SimpleNamespace(command_id="cmd-local-ok")

    async def _token_stream(_command_id):
        yield TokenChunk(
            model="Qwen/Qwen3-0.6B-GGUF",
            text="28",
            token_id=1,
            usage=None,
            finish_reason="stop",
        )

    api._resolve_and_validate_text_model = _resolve_model  # pyright: ignore[reportPrivateUsage]
    api._send_text_generation_with_images = _send_command  # pyright: ignore[reportPrivateUsage]
    api._token_chunk_stream = _token_stream  # pyright: ignore[reportPrivateUsage]
    client = TestClient(api.app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3-0.6B-GGUF",
            "messages": [{"role": "user", "content": "5+23="}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "28"


def test_local_chat_completions_nonstream_returns_503_when_runner_emits_no_chunks() -> None:
    api = _make_api()

    async def _resolve_model(model_id):
        return model_id

    async def _send_command(_task_params):
        return SimpleNamespace(command_id="cmd-local-empty")

    async def _token_stream(_command_id):
        if False:
            yield None

    api._resolve_and_validate_text_model = _resolve_model  # pyright: ignore[reportPrivateUsage]
    api._send_text_generation_with_images = _send_command  # pyright: ignore[reportPrivateUsage]
    api._token_chunk_stream = _token_stream  # pyright: ignore[reportPrivateUsage]
    client = TestClient(api.app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3-0.6B-GGUF",
            "messages": [{"role": "user", "content": "5+23="}],
            "stream": False,
        },
    )

    assert response.status_code == 503
    payload = response.json()
    assert "No output chunks were received from the runner" in payload["error"]["message"]

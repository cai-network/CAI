# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from cai.api.rate_limit import InMemoryFixedWindowRateLimiter
from cai.api.request_guards import (
    maybe_rate_limit_public_request,
    rate_limit_client_key,
    request_is_local,
    require_cai_api_bearer_token,
    require_feature_enabled,
    require_local_request,
)


def test_request_is_local_accepts_loopback_clients() -> None:
    app = FastAPI()

    @app.get("/check")
    def check(request: Request) -> dict[str, object]:
        return {
            "is_local": request_is_local(request),
            "client": rate_limit_client_key(request),
        }

    response = TestClient(app, client=("127.0.0.1", 40000)).get("/check")

    assert response.status_code == 200
    assert response.json() == {"is_local": True, "client": "127.0.0.1"}


def test_require_local_request_blocks_remote_clients() -> None:
    app = FastAPI()

    @app.get("/local")
    def local_only(request: Request) -> dict[str, bool]:
        require_local_request(request)
        return {"ok": True}

    response = TestClient(app, client=("198.51.100.20", 40000)).get("/local")

    assert response.status_code == 404


def test_public_rate_limit_applies_only_to_public_remote_endpoints() -> None:
    app = FastAPI()
    limiter = InMemoryFixedWindowRateLimiter(limit=1, window_seconds=60)

    @app.get("/v1/cai/summary")
    def public_summary(request: Request):
        rate_limit_response = maybe_rate_limit_public_request(
            request,
            public_rate_limiter=limiter,
        )
        if rate_limit_response is not None:
            return rate_limit_response
        return {"ok": True}

    remote_client = TestClient(app, client=("198.51.100.20", 40000))

    assert remote_client.get("/v1/cai/summary").status_code == 200
    limited = remote_client.get("/v1/cai/summary")
    assert limited.status_code == 429
    assert int(limited.headers["retry-after"]) > 0

    local_client = TestClient(app, client=("127.0.0.1", 40000))
    assert local_client.get("/v1/cai/summary").status_code == 200


def test_require_feature_enabled_hides_disabled_feature() -> None:
    app = FastAPI()

    @app.get("/feature")
    def feature() -> dict[str, bool]:
        require_feature_enabled(False)
        return {"ok": True}

    response = TestClient(app).get("/feature")

    assert response.status_code == 404


def test_require_cai_api_bearer_token_accepts_matching_token() -> None:
    app = FastAPI()

    @app.get("/secure")
    def secure(request: Request) -> dict[str, bool]:
        require_cai_api_bearer_token(request, "super-secret-token")
        return {"ok": True}

    client = TestClient(app)

    missing = client.get("/secure")
    invalid = client.get("/secure", headers={"Authorization": "Bearer wrong"})
    valid = client.get(
        "/secure",
        headers={"Authorization": "Bearer super-secret-token"},
    )

    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert valid.status_code == 200
    assert valid.json() == {"ok": True}

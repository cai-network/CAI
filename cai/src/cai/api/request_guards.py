# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import hmac

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger

from cai.api.audit import safe_audit_event
from cai.api.endpoint_policy import EndpointAccess, lookup_endpoint_policy
from cai.api.rate_limit import InMemoryFixedWindowRateLimiter


def rate_limit_client_key(request: Request) -> str:
    client = request.client
    return str(client.host).strip().lower() if client and client.host else "unknown"


def request_is_local(request: Request) -> bool:
    client = request.client
    host = str(client.host).strip().lower() if client and client.host else ""
    return host in {"127.0.0.1", "::1", "localhost"}


def require_local_request(request: Request) -> None:
    if request_is_local(request):
        return
    logger.warning(
        "CAI audit: {}",
        safe_audit_event(
            "local_only_denied",
            method=request.method,
            path=request.url.path,
            client_host=rate_limit_client_key(request),
            status="denied",
        ),
    )
    raise HTTPException(status_code=404, detail="Not found.")


def maybe_rate_limit_public_request(
    request: Request,
    *,
    public_rate_limiter: InMemoryFixedWindowRateLimiter,
) -> JSONResponse | None:
    if request_is_local(request):
        return None
    policy = lookup_endpoint_policy(request.method, request.url.path)
    if policy is None or policy.access != EndpointAccess.PUBLIC:
        return None

    decision = public_rate_limiter.check(rate_limit_client_key(request))
    if decision.allowed:
        return None
    logger.warning(
        "CAI audit: {}",
        safe_audit_event(
            "public_rate_limit_exceeded",
            method=request.method,
            path=request.url.path,
            client_host=rate_limit_client_key(request),
            status="denied",
        ),
    )
    return JSONResponse(
        {"detail": "Rate limit exceeded."},
        status_code=429,
        headers={"Retry-After": str(decision.retry_after_seconds)},
    )


def require_feature_enabled(enabled: bool) -> None:
    if not bool(enabled):
        raise HTTPException(status_code=404, detail="Not found.")


def require_cai_api_bearer_token(request: Request, expected_token: str) -> None:
    expected_token = str(expected_token or "").strip()
    if not expected_token:
        return

    authorization = str(request.headers.get("authorization") or "").strip()
    prefix = "bearer "
    if not authorization.lower().startswith(prefix):
        logger.warning(
            "CAI audit: {}",
            safe_audit_event(
                "bearer_auth_failed",
                method=request.method,
                path=request.url.path,
                client_host=rate_limit_client_key(request),
                status="missing",
            ),
        )
        raise HTTPException(status_code=401, detail="Unauthorized.")

    provided_token = authorization[len(prefix) :].strip()
    if provided_token and hmac.compare_digest(provided_token, expected_token):
        return

    logger.warning(
        "CAI audit: {}",
        safe_audit_event(
            "bearer_auth_failed",
            method=request.method,
            path=request.url.path,
            client_host=rate_limit_client_key(request),
            status="invalid",
        ),
    )
    raise HTTPException(status_code=401, detail="Unauthorized.")

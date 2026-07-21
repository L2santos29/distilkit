"""Middleware for the DistilKit web application.

Extracted from ``webapp.py`` to keep files under the 400-line limit.
"""

import time
import uuid
from collections import defaultdict
from typing import Any

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from src.alert_manager import record_response
from src.log_config import logger, set_request_id
from src.settings import settings
from src.tracing import tracer

# ---------------------------------------------------------------------------
# Shared mutable state (used by middleware and route handlers)
# ---------------------------------------------------------------------------

# Server start time for uptime metrics
_start_time: float = time.time()

# Request-level metrics (reset on restart)
_metrics: dict = {
    "requests_total": 0,
    "requests_by_path": {},
    "errors_total": 0,
    "duration_total_sec": 0.0,
}

# Per-route rate limit overrides: (prefix, requests_per_minute)
# More sensitive/heavy endpoints get stricter limits.
_RATE_LIMITS: list[tuple[str, int]] = [
    ("/health", 60),
    ("/ready", 60),
    ("/live", 60),
    ("/api/train/", 10),
    ("/api/export/", 10),
]


# ---------------------------------------------------------------------------
# Authentication — API key via X-API-Key header
# ---------------------------------------------------------------------------


def require_api_key(request: Request) -> None:
    """Verify the ``X-API-Key`` header matches the configured API key.

    If ``settings.api_key`` is empty, authentication is disabled (local/dev
    mode).  If a key is configured, every protected endpoint **must** include
    the ``X-API-Key`` header with the matching value.
    """
    if not settings.api_key:
        return

    key = request.headers.get("X-API-Key")
    if not key:
        raise HTTPException(
            status_code=401,
            detail="Missing X-API-Key header. Set API_KEY env var on the server to enable authentication.",
            headers={"WWW-Authenticate": "APIKey"},
        )
    if key != settings.api_key:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )


# ---------------------------------------------------------------------------
# RateLimitMiddleware
# ---------------------------------------------------------------------------


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter keyed by client IP.

    Respects ``X-Forwarded-For`` when behind a reverse proxy.  The default
    limit (``settings.rate_limit_per_minute``) is overridden for specific
    route prefixes in ``_RATE_LIMITS``.  Set the env var to 0 to disable.
    """

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self._windows: dict[str, list[float]] = defaultdict(list)

    def _client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For", "")
        return forwarded.split(",")[0].strip() or request.client.host or "127.0.0.1"

    def _limit_for(self, path: str) -> int:
        default = settings.rate_limit_per_minute
        if default <= 0:
            return 0
        for prefix, limit in _RATE_LIMITS:
            if path.startswith(prefix):
                return max(0, limit)
        return default

    async def dispatch(self, request: Request, call_next):
        max_reqs = self._limit_for(request.url.path)
        if max_reqs <= 0:
            return await call_next(request)

        client = self._client_ip(request)
        now = time.time()
        window = self._windows[client]

        cutoff = now - 60.0
        while window and window[0] < cutoff:
            window.pop(0)

        if len(window) >= max_reqs:
            retry_after = int(60.0 - (now - window[0]))
            return Response(
                status_code=429,
                content='{"detail":"Rate limit exceeded. Try again later."}',
                media_type="application/json",
                headers={"Retry-After": str(max(retry_after, 1))},
            )

        window.append(now)
        return await call_next(request)


# ---------------------------------------------------------------------------
# SecurityHeadersMiddleware
# ---------------------------------------------------------------------------


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security-related HTTP headers to every response.

    Headers added:
    - ``X-Content-Type-Options: nosniff`` — prevent MIME sniffing.
    - ``X-Frame-Options: DENY`` — prevent clickjacking.
    - ``Strict-Transport-Security`` — only if ``settings.hsts_max_age > 0``.
    - ``Content-Security-Policy`` — restricts script/style sources to
      trusted CDNs (Tailwind, Chart.js) and self.
    """

    _CSP = (
        "default-src 'self';"
        " script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://cdn.jsdelivr.net;"
        " style-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com;"
        " img-src 'self' data:;"
        " font-src 'self';"
        " connect-src 'self';"
        " form-action 'self'"
    )

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        if settings.hsts_max_age > 0:
            response.headers["Strict-Transport-Security"] = (
                f"max-age={settings.hsts_max_age}; includeSubDomains"
            )
        response.headers["Content-Security-Policy"] = self._CSP
        return response


# ---------------------------------------------------------------------------
# RequestIDMiddleware
# ---------------------------------------------------------------------------


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assign a unique request ID, create a tracing span, and track metrics.

    Propagates W3C ``traceparent`` headers for distributed tracing.
    Incoming ``traceparent`` headers are honoured so upstream services
    can link their traces to DistilKit spans.
    """

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        set_request_id(rid)
        path = request.url.path
        method = request.method

        traceparent = request.headers.get("traceparent")
        span_name = f"{method} {path}"
        if traceparent:
            span = tracer.span_from_traceparent(span_name, traceparent)
        else:
            span = tracer.start_span(span_name)
        span.set_attribute("http.method", method)
        span.set_attribute("http.path", path)
        span.set_attribute("request_id", rid)

        logger.bind(request_id=rid).info(f"\u2192 {method} {path}")

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            _metrics["errors_total"] += 1
            _metrics["requests_total"] += 1
            _metrics["requests_by_path"].setdefault(path, 0)
            _metrics["requests_by_path"][path] += 1
            span.set_attribute("error", "true")
            span.end()
            set_request_id("")
            raise

        duration = time.perf_counter() - start
        _metrics["requests_total"] += 1
        _metrics["requests_by_path"].setdefault(path, 0)
        _metrics["requests_by_path"][path] += 1
        _metrics["duration_total_sec"] += duration

        if response.status_code >= 500:
            _metrics["errors_total"] += 1
            span.set_attribute("error", "true")

        record_response(response.status_code, duration)

        span.set_attribute("http.status_code", response.status_code)
        span.end()

        response.headers["X-Request-ID"] = rid
        response.headers["traceparent"] = span.to_traceparent()
        set_request_id("")
        return response

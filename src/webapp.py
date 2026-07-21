"""DistilKit Web GUI — FastAPI + Tailwind CSS.

Run with:
    python -m src.webapp
    # or: uvicorn src.webapp:app --reload
"""

import asyncio
import http.server
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware

from src.alert_manager import evaluate_alerts
from src.log_config import logger
from src.settings import settings
from src.task_manager import _tasks
from src.webapp_middleware import (
    RateLimitMiddleware,
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
)
from src.webapp_routes import api_router, legacy_router


# ---------------------------------------------------------------------------
# Lifespan — clean up background tasks on shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: cleanup tasks on shutdown."""
    task = asyncio.create_task(evaluate_alerts())
    try:
        yield
    finally:
        task.cancel()
        for t in _tasks.values():
            if t.status in ("pending", "running"):
                t.cancel()


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DistilKit",
    description="Knowledge Distillation \u2014 Web GUI",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable gzip compression for all responses (including SSE streams)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS — allow cross-origin requests from configured origins
origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)

# Security headers (set before any other middleware can modify them)
app.add_middleware(SecurityHeadersMiddleware)

# Rate limiter (before auth so unauthenticated floods are also blocked)
app.add_middleware(RateLimitMiddleware)

# Request ID middleware (must be after GZip to log original request)
app.add_middleware(RequestIDMiddleware)

# Register versioned and legacy routers
app.include_router(api_router)
app.include_router(legacy_router)


# ---------------------------------------------------------------------------
# Validation error handler — convert Pydantic 422 into 400
# ---------------------------------------------------------------------------


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    """Return Pydantic validation errors as HTTP 400 instead of the default 422."""
    messages: list[str] = []
    for err in exc.errors():
        loc = " \u2192 ".join(str(l) for l in err.get("loc", []))
        messages.append(f"{loc}: {err['msg']}" if loc else err["msg"])
    raise HTTPException(status_code=400, detail="; ".join(messages))


# ---------------------------------------------------------------------------
# Redirect server (HTTP \u2192 HTTPS)
# ---------------------------------------------------------------------------

from typing import Any  # noqa: E402


def _start_redirect_server(redirect_port: int, target_port: int) -> None:
    """Start a minimal HTTP server that redirects all traffic to HTTPS."""
    class RedirectHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(301)
            self.send_header(
                "Location",
                f"https://{self.headers.get('Host', 'localhost').split(':')[0]}:{target_port}{self.path}",
            )
            self.end_headers()

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug(f"HTTP\u2192HTTPS redirect: {fmt % args}")

    server = http.server.HTTPServer(("0.0.0.0", redirect_port), RedirectHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"   \u2192 http://0.0.0.0:{redirect_port} redirecting to https://...:{target_port}")


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------


def launch(port: int | None = None, host: str | None = None, api_only: bool = False) -> None:
    """Launch the web server.

    Args:
        port: Server port (default: ``settings.port``).
        host: Bind address (default: ``settings.host``).
        api_only: If True, only expose the REST API (no frontend).

    If ``settings.ssl_certfile`` and ``settings.ssl_keyfile`` are set, the
    server serves HTTPS.  An automatic HTTP\u2192HTTPS redirect server is started
    on port 80 (or ``settings.port + 1`` if port 80 is unavailable).
    """
    host = host or settings.host
    port = port or settings.port
    import uvicorn  # noqa: E402

    if api_only:
        settings.api_only = True

    is_https = bool(settings.ssl_certfile and settings.ssl_keyfile)
    scheme = "https" if is_https else "http"
    mode = "API-only" if settings.api_only else "Web GUI"
    logger.info(f"\u26a1 DistilKit {mode}")
    logger.info(f"   \u2192 {scheme}://{host}:{port}")

    if is_https:
        if settings.hsts_max_age == 0:
            settings.hsts_max_age = 31536000
        try:
            _start_redirect_server(80, port)
        except OSError:
            _start_redirect_server(port + 1, port)

    logger.info("   → Press Ctrl+C to stop\n")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="warning",
        ssl_certfile=settings.ssl_certfile or None,
        ssl_keyfile=settings.ssl_keyfile or None,
    )


# ---------------------------------------------------------------------------
# Entry point — allows ``python -m src.webapp`` and ``run_gui.sh``
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    launch()

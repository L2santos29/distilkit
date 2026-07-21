"""DistilKit Web GUI — FastAPI + Tailwind CSS.

Run with:
    python -m src.webapp
    # or: uvicorn src.webapp:app --reload
"""

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from collections import defaultdict
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware

from src import datasets as ds
from src.alert_manager import evaluate_alerts, record_response, record_task_failure
from src.log_config import logger, set_request_id
from src.onnx_export import export_to_onnx, export_to_torchscript
from src.settings import settings
from src.task_manager import TrainingTask, _history, _save_run, _tasks, get_history_store, get_tasks
from src.settings import settings
from src.tracing import tracer, current_span


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
        return  # Auth disabled — allow everything

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
# Pydantic models for request validation
# ---------------------------------------------------------------------------


class TrainRequest(BaseModel):
    """Validated request body for POST /api/train."""

    dataset: str = Field(default="CIFAR-10", description="Dataset name")
    teacher: str = Field(default="resnet18", description="Teacher model architecture")
    student: str = Field(default="MiniCNN", description="Student model architecture")
    compression_ratio: float = Field(
        default=0.05, ge=0.01, le=1.0, description="Target student/teacher parameter ratio"
    )
    epochs: int = Field(
        default=10, ge=1, le=1000, description="Number of training epochs"
    )
    temperature: float = Field(
        default=4.0, ge=0.1, le=100.0, description="Distillation temperature — higher = softer targets"
    )
    alpha: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Distillation loss weight (0-1). Higher = more teacher influence"
    )
    patience: int = Field(
        default=0, ge=0, le=100, description="Early stopping patience (0 to disable)"
    )
    batch_size: int = Field(
        default=64, ge=1, le=4096, description="Training batch size"
    )

    @field_validator("dataset")
    @classmethod
    def _check_dataset(cls, v: str) -> str:
        if v not in ds.DATASETS:
            raise ValueError(f"Invalid dataset. Choose: {ds.DATASET_CHOICES}")
        return v

    @field_validator("teacher")
    @classmethod
    def _check_teacher(cls, v: str) -> str:
        if v not in ds.TEACHER_CHOICES:
            raise ValueError(f"Invalid teacher. Choose: {ds.TEACHER_CHOICES}")
        return v

    @field_validator("student")
    @classmethod
    def _check_student(cls, v: str) -> str:
        if v not in ds.STUDENT_CHOICES:
            raise ValueError(f"Invalid student. Choose: {ds.STUDENT_CHOICES}")
        return v


class ExportRequest(BaseModel):
    """Validated request body for POST /api/export/{task_id}."""

    format: str = Field(default="onnx", description="Export format")

    @field_validator("format")
    @classmethod
    def _check_format(cls, v: str) -> str:
        if v not in ("onnx", "torchscript"):
            raise ValueError("Invalid format. Use 'onnx' or 'torchscript'")
        return v

HERE = Path(__file__).parent
TEMPLATE_FILE = HERE / "templates" / "index.html"

# Server start time for uptime metric
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
    ("/api/train/", 10),  # task detail, SSE stream
    ("/api/export/", 10),
]


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
            return 0  # globally disabled
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

        # Purge entries older than 60 s
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

        # ── Distributed tracing ─────────────────────────────────────
        traceparent = request.headers.get("traceparent")
        span_name = f"{method} {path}"
        if traceparent:
            span = tracer.span_from_traceparent(span_name, traceparent)
        else:
            span = tracer.start_span(span_name)
        span.set_attribute("http.method", method)
        span.set_attribute("http.path", path)
        span.set_attribute("request_id", rid)

        logger.bind(request_id=rid).info(f"→ {method} {path}")

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

        # Propagate trace context to the response
        response.headers["X-Request-ID"] = rid
        response.headers["traceparent"] = span.to_traceparent()
        set_request_id("")
        return response


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


app = FastAPI(
    title="DistilKit",
    description="Knowledge Distillation — Web GUI",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable gzip compression for all responses (including SSE streams)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS — allow cross-origin requests from configured origins
origins = [
    o.strip()
    for o in settings.cors_origins.split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-API-Key",
        "X-Request-ID",
    ],
    expose_headers=["X-Request-ID"],
)

# Security headers (set before any other middleware can modify them)
app.add_middleware(SecurityHeadersMiddleware)

# Rate limiter (before auth so unauthenticated floods are also blocked)
app.add_middleware(RateLimitMiddleware)

# Request ID middleware (must be after GZip to log original request)
app.add_middleware(RequestIDMiddleware)

# Versioned API router — all future breaking changes increment this prefix.
api_router = APIRouter(prefix="/api/v1")


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
        loc = " → ".join(str(l) for l in err.get("loc", []))
        messages.append(f"{loc}: {err['msg']}" if loc else err["msg"])
    raise HTTPException(status_code=400, detail="; ".join(messages))


@app.get("/health")
@app.get("/ready")
@app.get("/live")
async def health():
    """Health, readiness, and liveness endpoints for orchestration platforms."""
    return {
        "status": "ok",
        "service": "distilkit",
        "version": "0.1.0",
    }


@app.get("/metrics")
async def metrics():
    """Prometheus-format metrics for application monitoring."""
    uptime_sec = time.time() - _start_time
    task_statuses = [t.status for t in _tasks.values()]
    req_total = _metrics["requests_total"]
    dur_total = _metrics["duration_total_sec"]

    lines = [
        "# HELP distilkit_uptime_seconds Application uptime in seconds",
        "# TYPE distilkit_uptime_seconds gauge",
        f"distilkit_uptime_seconds {uptime_sec:.0f}",
        "",
        "# HELP distilkit_requests_total Total HTTP requests processed",
        "# TYPE distilkit_requests_total counter",
        f"distilkit_requests_total {req_total}",
        "",
        "# HELP distilkit_request_duration_seconds Cumulative request duration",
        "# TYPE distilkit_request_duration_seconds counter",
        f"distilkit_request_duration_seconds {dur_total:.3f}",
        "",
        "# HELP distilkit_errors_total Total 5xx responses",
        "# TYPE distilkit_errors_total counter",
        f"distilkit_errors_total {_metrics['errors_total']}",
        "",
        "# HELP distilkit_tasks_total Total tasks by status",
        "# TYPE distilkit_tasks_total gauge",
    ]
    for status in ("running", "completed", "failed", "cancelled", "pending"):
        count = task_statuses.count(status)
        lines.append(f'distilkit_tasks_total{{status="{status}"}} {count}')
    lines.append("")

    # Per-path request counts
    for path, count in sorted(_metrics["requests_by_path"].items()):
        lines.append('distilkit_requests_per_path{path="' + path + '"} ' + str(count))
    lines.append("")
    return PlainTextResponse("\n".join(lines))


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main page (HTML) or API info (JSON) in API-only mode."""
    if settings.api_only:
        return JSONResponse(
            {
                "service": "DistilKit API",
                "version": "0.1.0",
                "docs": "/docs",
                "openapi": "/openapi.json",
                "endpoints": [
                    {
                        "path": "/api/v1/config",
                        "method": "GET",
                        "description": "Available datasets, teachers, students",
                    },
                    {
                        "path": "/api/v1/train",
                        "method": "POST",
                        "description": "Start a new distillation task",
                    },
                    {
                        "path": "/api/v1/train/{task_id}",
                        "method": "GET",
                        "description": "Get task state",
                    },
                    {
                        "path": "/api/v1/train/{task_id}/stream",
                        "method": "GET",
                        "description": "SSE progress stream",
                    },
                    {
                        "path": "/api/v1/train/{task_id}/cancel",
                        "method": "POST",
                        "description": "Cancel a running task",
                    },
                    {
                        "path": "/api/v1/export/{task_id}",
                        "method": "POST",
                        "description": "Export trained model",
                    },
                    {
                        "path": "/api/v1/download/{filename}",
                        "method": "GET",
                        "description": "Download exported file",
                    },
                    {"path": "/api/v1/tasks", "method": "GET", "description": "List all tasks"},
                    {
                        "path": "/api/v1/history",
                        "method": "GET",
                        "description": "Completed training runs",
                    },
                ],
            }
        )
    html = TEMPLATE_FILE.read_text(encoding="utf-8")
    return HTMLResponse(html)


@api_router.post("/train")
@app.post("/api/train")
async def start_training(
    body: TrainRequest,
    _auth: None = Depends(require_api_key),
    tasks: dict = Depends(get_tasks),
):
    """Start a new distillation task."""
    config = body.model_dump()
    task = TrainingTask(config)
    tasks[task.id] = task
    task.start()

    return {"task_id": task.id}


@api_router.get("/train/{task_id}/stream")
@app.get("/api/train/{task_id}/stream")
async def stream_progress(
    task_id: str,
    request: Request,
    tasks: dict = Depends(get_tasks),
):
    """SSE endpoint for real-time training progress."""
    # Authenticate via header (preferred) or query-parameter fallback.
    # ``EventSource`` in browsers cannot set custom headers, so the frontend
    # passes the API key as ``?api_key=...`` when auth is enabled.
    if settings.api_key:
        key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if not key:
            raise HTTPException(401, detail="Missing X-API-Key header or ?api_key query parameter.")
        if key != settings.api_key:
            raise HTTPException(403, detail="Invalid API key.")

    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        """Yield SSE events with progress, logs, and results."""
        last_logs_len = 0

        while task.status in ("pending", "running"):
            # Only send data when something changed (dirty flag)
            if not task._dirty:
                await asyncio.sleep(0.5)
                continue

            task._dirty = False
            new_logs = task.logs[last_logs_len:]
            last_logs_len = len(task.logs)

            data = {
                "status": task.status,
                "progress": round(task.progress, 3),
                "current_epoch": task.current_epoch,
                "total_epochs": task.total_epochs,
                "current_loss": task.current_loss,
                "current_acc": task.current_acc,
                "eta_seconds": round(task.eta_seconds, 1),
                "losses": task.losses,
                "accuracies": task.accuracies,
                "logs": new_logs,
            }
            yield f"data: {json.dumps(data)}\n\n"

            if task.status == "completed":
                data["result"] = task.result
                yield f"event: complete\ndata: {json.dumps(data)}\n\n"
                return
            elif task.status == "cancelled":
                yield f"event: cancel\ndata: {json.dumps(data)}\n\n"
                return
            elif task.status == "failed":
                data["error"] = task.error
                yield f"event: error\ndata: {json.dumps(data)}\n\n"
                return

            await asyncio.sleep(0.1)

        # Send final status one more time
        if task.status == "completed" and task.result:
            data = {
                "status": "completed",
                "progress": 1.0,
                "result": task.result,
            }
            yield f"event: complete\ndata: {json.dumps(data)}\n\n"
        elif task.status == "failed":
            yield f"event: error\ndata: {json.dumps({'error': task.error})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@api_router.get("/train/{task_id}")
@app.get("/api/train/{task_id}")
async def get_task(
    task_id: str,
    _auth: None = Depends(require_api_key),
    tasks: dict = Depends(get_tasks),
):
    """Get the current state of a task."""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return {
        "id": task.id,
        "status": task.status,
        "progress": task.progress,
        "current_epoch": task.current_epoch,
        "total_epochs": task.total_epochs,
        "logs": task.logs[-2000:],
        "result": task.result,
        "error": task.error,
    }


@api_router.post("/export/{task_id}")
@app.post("/api/export/{task_id}")
async def export_model(
    task_id: str,
    body: ExportRequest,
    _auth: None = Depends(require_api_key),
    tasks: dict = Depends(get_tasks),
):
    """Export the trained student model."""
    task = tasks.get(task_id)
    if not task or task.status != "completed":
        raise HTTPException(400, "No completed model to export")
    if task.student is None:
        raise HTTPException(400, "Student model not available")

    fmt = body.format
    os.makedirs("checkpoints", exist_ok=True)

    # Unique filename with dataset + model info
    cfg = task.config
    tag = f"{cfg.get('dataset', 'dataset')}_{cfg.get('student', 'student')}"
    ext = "onnx" if fmt == "onnx" else "pt"
    filename = f"distilkit_{tag}.{ext}"
    filepath = os.path.join("checkpoints", filename)

    try:
        if fmt == "onnx":
            try:
                export_to_onnx(task.student, filepath)
            except (OSError, RuntimeError) as e:
                # ONNX export failed — degrade gracefully to TorchScript
                self_emit = getattr(task, "_emit", None)
                if self_emit:
                    self_emit(f"\u26a0\ufe0f ONNX export failed ({e}), falling back to TorchScript.")
                fallback = filepath.replace(".onnx", ".pt")
                export_to_torchscript(task.student, fallback)
                filename = filename.replace(".onnx", ".pt")
                filepath = fallback
                fmt = "torchscript"
        elif fmt == "torchscript":
            export_to_torchscript(task.student, filepath)
        else:
            raise HTTPException(400, "Invalid format. Use 'onnx' or 'torchscript'")
        return {"filename": filename, "path": filepath, "format": fmt}
    except (OSError, RuntimeError) as e:
        raise HTTPException(500, f"Export failed: {e}")


@api_router.get("/config")
@app.get("/api/config")
async def get_config():
    """Return app configuration (teachers list, device, cache status).

    Responses include a ``Cache-Control`` header so that reverse proxies
    and browsers can cache the result for up to 10 seconds, reducing
    repeated filesystem scans of the torch hub cache directory.
    """
    # Check torch hub cache for each teacher model
    cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
    cached_models: dict[str, bool] = {}
    if os.path.isdir(cache_dir):
        cached_files = set(os.listdir(cache_dir))
        for model in ds.TEACHER_CHOICES:
            # Check if any cached file starts with the model name
            cached_models[model] = any(f.startswith(model) for f in cached_files)
    else:
        for model in ds.TEACHER_CHOICES:
            cached_models[model] = False

    body = {
        "api_version": "1.0",
        "datasets": ds.DATASET_CHOICES,
        "teachers": ds.TEACHER_CHOICES,
        "students": ds.STUDENT_CHOICES,
        "device": settings.device,
        "cached_teachers": cached_models,
        "auth_required": bool(settings.api_key),
    }
    return JSONResponse(body, headers={"Cache-Control": "max-age=10"})


@api_router.post("/train/{task_id}/cancel")
@app.post("/api/train/{task_id}/cancel")
async def cancel_training(
    task_id: str,
    _auth: None = Depends(require_api_key),
    tasks: dict = Depends(get_tasks),
):
    """Cancel a running training task."""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task.status not in ("pending", "running"):
        raise HTTPException(400, f"Task is already {task.status}")
    task.cancel()
    return {"status": "cancelled"}


@api_router.get("/history")
@app.get("/api/history")
async def get_history(
    _auth: None = Depends(require_api_key),
    history_store: list = Depends(get_history_store),
    limit: int = Query(default=50, ge=1, le=1000, description="Max items to return"),
    offset: int = Query(default=0, ge=0, description="Number of items to skip"),
):
    """Return completed training runs with pagination."""
    return JSONResponse(history_store[offset:][:limit], headers={"Cache-Control": "max-age=5"})


@api_router.get("/download/{filename}")
@app.get("/api/download/{filename}")
async def download_file(
    filename: str,
    _auth: None = Depends(require_api_key),
):
    """Download an exported model file."""
    from fastapi.responses import FileResponse

    safe_path = os.path.join("checkpoints", os.path.basename(filename))
    if not os.path.exists(safe_path):
        raise HTTPException(404, f"File not found: {filename}")
    return FileResponse(
        safe_path,
        media_type="application/octet-stream",
        filename=filename,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "private, max-age=3600",
        },
    )


@api_router.get("/tasks")
@app.get("/api/tasks")
async def list_tasks(
    _auth: None = Depends(require_api_key),
    tasks: dict = Depends(get_tasks),
    limit: int = Query(default=50, ge=1, le=1000, description="Max items to return"),
    offset: int = Query(default=0, ge=0, description="Number of items to skip"),
):
    """List training tasks with pagination."""
    all_tasks = [
        {
            "id": t.id,
            "status": t.status,
            "progress": t.progress,
            "config": {
                "teacher": t.config["teacher"],
                "epochs": t.config["epochs"],
            },
            "created_at": t.created_at.isoformat(),
        }
        for t in tasks.values()
    ]
    return JSONResponse(all_tasks[offset:][:limit], headers={"Cache-Control": "max-age=3"})


# Register the versioned router (under /api/v1).
app.include_router(api_router)


def _start_redirect_server(redirect_port: int, target_port: int) -> None:
    """Start a minimal HTTP server that redirects all traffic to HTTPS."""
    import http.server
    import threading

    class RedirectHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(301)
            self.send_header("Location", f"https://{self.headers.get('Host', 'localhost').split(':')[0]}:{target_port}{self.path}")
            self.end_headers()

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug(f"HTTP→HTTPS redirect: {fmt % args}")

    server = http.server.HTTPServer(("0.0.0.0", redirect_port), RedirectHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"   → http://0.0.0.0:{redirect_port} redirecting to https://...:{target_port}")


def launch(port: int | None = None, host: str | None = None, api_only: bool = False) -> None:
    """Launch the web server.

    Args:
        port: Server port (default: ``settings.port``).
        host: Bind address (default: ``settings.host``).
        api_only: If True, only expose the REST API (no frontend).

    If ``settings.ssl_certfile`` and ``settings.ssl_keyfile`` are set, the
    server serves HTTPS.  An automatic HTTP→HTTPS redirect server is started
    on port 80 (or ``settings.port + 1`` if port 80 is unavailable).
    """
    host = host or settings.host
    port = port or settings.port
    import uvicorn

    if api_only:
        settings.api_only = True

    is_https = bool(settings.ssl_certfile and settings.ssl_keyfile)
    scheme = "https" if is_https else "http"
    mode = "API-only" if settings.api_only else "Web GUI"
    logger.info(f"⚡ DistilKit {mode}")
    logger.info(f"   → {scheme}://{host}:{port}")

    if is_https:
        # Auto-enable HSTS when TLS is active.
        if settings.hsts_max_age == 0:
            settings.hsts_max_age = 31536000
        # Start HTTP→HTTPS redirect on port 80 (or port+1 if 80 conflicts).
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


if __name__ == "__main__":
    launch()

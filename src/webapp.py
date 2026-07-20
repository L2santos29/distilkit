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

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware

from src import datasets as ds
from src.log_config import logger, set_request_id
from src.onnx_export import export_to_onnx, export_to_torchscript
from src.settings import settings
from src.task_manager import TrainingTask, _history, _save_run, _tasks, get_history_store, get_tasks

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


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assign a unique request ID and track request metrics."""

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        set_request_id(rid)
        path = request.url.path
        logger.bind(request_id=rid).info(f"→ {request.method} {path}")

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            _metrics["errors_total"] += 1
            _metrics["requests_total"] += 1
            _metrics["requests_by_path"].setdefault(path, 0)
            _metrics["requests_by_path"][path] += 1
            set_request_id("")
            raise

        duration = time.perf_counter() - start
        _metrics["requests_total"] += 1
        _metrics["requests_by_path"].setdefault(path, 0)
        _metrics["requests_by_path"][path] += 1
        _metrics["duration_total_sec"] += duration

        if response.status_code >= 500:
            _metrics["errors_total"] += 1

        response.headers["X-Request-ID"] = rid
        set_request_id("")
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: cleanup tasks on shutdown."""
    try:
        yield
    finally:
        for task in _tasks.values():
            if task.status in ("pending", "running"):
                task.cancel()


app = FastAPI(
    title="DistilKit",
    description="Knowledge Distillation — Web GUI",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable gzip compression for all responses (including SSE streams)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add request ID middleware (must be after GZip to log original request)
app.add_middleware(RequestIDMiddleware)


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
                        "path": "/api/config",
                        "method": "GET",
                        "description": "Available datasets, teachers, students",
                    },
                    {
                        "path": "/api/train",
                        "method": "POST",
                        "description": "Start a new distillation task",
                    },
                    {
                        "path": "/api/train/{task_id}",
                        "method": "GET",
                        "description": "Get task state",
                    },
                    {
                        "path": "/api/train/{task_id}/stream",
                        "method": "GET",
                        "description": "SSE progress stream",
                    },
                    {
                        "path": "/api/train/{task_id}/cancel",
                        "method": "POST",
                        "description": "Cancel a running task",
                    },
                    {
                        "path": "/api/export/{task_id}",
                        "method": "POST",
                        "description": "Export trained model",
                    },
                    {
                        "path": "/api/download/{filename}",
                        "method": "GET",
                        "description": "Download exported file",
                    },
                    {"path": "/api/tasks", "method": "GET", "description": "List all tasks"},
                    {
                        "path": "/api/history",
                        "method": "GET",
                        "description": "Completed training runs",
                    },
                ],
            }
        )
    html = TEMPLATE_FILE.read_text(encoding="utf-8")
    return HTMLResponse(html)


def _clamp_and_validate(value: float, name: str, lo: float, hi: float) -> float:
    """Clamp a numeric value to [lo, hi] or raise if it's not a valid number."""
    if lo <= value <= hi:
        return value
    raise HTTPException(400, f"{name} must be between {lo} and {hi}, got {value}")


@app.post("/api/train")
async def start_training(body: dict, tasks: dict = Depends(get_tasks)):
    """Start a new distillation task."""
    try:
        raw_compression = float(body.get("compression_ratio", 0.05))
        raw_epochs = int(body.get("epochs", 10))
        raw_temperature = float(body.get("temperature", 4.0))
        raw_alpha = float(body.get("alpha", 0.7))
        raw_patience = int(body.get("patience", 0))
        raw_batch_size = int(body.get("batch_size", 64))
    except (ValueError, TypeError) as e:
        raise HTTPException(400, f"Invalid numeric parameter: {e}")

    config = {
        "dataset": body.get("dataset", "CIFAR-10"),
        "teacher": body.get("teacher", "resnet18"),
        "student": body.get("student", "MiniCNN"),
        "compression_ratio": _clamp_and_validate(raw_compression, "compression_ratio", 0.01, 1.0),
        "epochs": _clamp_and_validate(raw_epochs, "epochs", 1, 1000),
        "temperature": _clamp_and_validate(raw_temperature, "temperature", 0.1, 100.0),
        "alpha": _clamp_and_validate(raw_alpha, "alpha", 0.0, 1.0),
        "patience": _clamp_and_validate(raw_patience, "patience", 0, 100),
        "batch_size": _clamp_and_validate(raw_batch_size, "batch_size", 1, 4096),
    }

    if config["dataset"] not in ds.DATASETS:
        raise HTTPException(400, f"Invalid dataset. Choose: {ds.DATASET_CHOICES}")
    if config["teacher"] not in ds.TEACHER_CHOICES:
        raise HTTPException(400, f"Invalid teacher. Choose: {ds.TEACHER_CHOICES}")
    if config["student"] not in ds.STUDENT_CHOICES:
        raise HTTPException(400, f"Invalid student. Choose: {ds.STUDENT_CHOICES}")

    task = TrainingTask(config)
    tasks[task.id] = task
    task.start()

    return {"task_id": task.id}


@app.get("/api/train/{task_id}/stream")
async def stream_progress(task_id: str, tasks: dict = Depends(get_tasks)):
    """SSE endpoint for real-time training progress."""
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


@app.get("/api/train/{task_id}")
async def get_task(task_id: str, tasks: dict = Depends(get_tasks)):
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


@app.post("/api/export/{task_id}")
async def export_model(task_id: str, body: dict, tasks: dict = Depends(get_tasks)):
    """Export the trained student model."""
    task = tasks.get(task_id)
    if not task or task.status != "completed":
        raise HTTPException(400, "No completed model to export")
    if task.student is None:
        raise HTTPException(400, "Student model not available")

    fmt = body.get("format", "onnx")
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
                    self_emit(f"⚠️ ONNX export failed ({e}), falling back to TorchScript.")
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


@app.get("/api/config")
async def get_config():
    """Return app configuration (teachers list, device, cache status)."""
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

    return {
        "datasets": ds.DATASET_CHOICES,
        "teachers": ds.TEACHER_CHOICES,
        "students": ds.STUDENT_CHOICES,
        "device": settings.device,
        "cached_teachers": cached_models,
    }


@app.post("/api/train/{task_id}/cancel")
async def cancel_training(task_id: str, tasks: dict = Depends(get_tasks)):
    """Cancel a running training task."""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task.status not in ("pending", "running"):
        raise HTTPException(400, f"Task is already {task.status}")
    task.cancel()
    return {"status": "cancelled"}


@app.get("/api/history")
async def get_history(history_store: list = Depends(get_history_store)):
    """Return all completed training runs."""
    return history_store


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Download an exported model file."""
    from fastapi.responses import FileResponse

    safe_path = os.path.join("checkpoints", os.path.basename(filename))
    if not os.path.exists(safe_path):
        raise HTTPException(404, f"File not found: {filename}")
    return FileResponse(
        safe_path,
        media_type="application/octet-stream",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/tasks")
async def list_tasks(tasks: dict = Depends(get_tasks)):
    """List all training tasks."""
    return [
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


def launch(port: int | None = None, host: str | None = None, api_only: bool = False) -> None:
    """Launch the web server.

    Args:
        port: Server port (default: ``settings.port``).
        host: Bind address (default: ``settings.host``).
        api_only: If True, only expose the REST API (no frontend).
    """
    host = host or settings.host
    port = port or settings.port
    import uvicorn

    if api_only:
        settings.api_only = True

    mode = "API-only" if settings.api_only else "Web GUI"
    logger.info(f"⚡ DistilKit {mode}")
    logger.info(f"   → http://{host}:{port}")
    logger.info("   → Press Ctrl+C to stop\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    launch()

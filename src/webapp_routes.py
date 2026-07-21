"""Route handlers for the DistilKit web GUI.

Extracted from ``webapp.py`` to keep files under the 400-line limit.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

from src import datasets as ds
from src.onnx_export import export_to_onnx, export_to_torchscript
from src.settings import settings
from src.task_manager import TrainingTask, _tasks, get_history_store, get_tasks
from src.webapp_middleware import _metrics, _start_time, require_api_key
from src.webapp_models import ExportRequest, TrainRequest

HERE = Path(__file__).parent
TEMPLATE_FILE = HERE / "templates" / "index.html"

# Versioned API router — all future breaking changes increment this prefix.
api_router = APIRouter(prefix="/api/v1")
# Legacy bare router for backward compatibility (no version prefix).
legacy_router = APIRouter()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@api_router.get("/health")
@legacy_router.get("/health")
@api_router.get("/ready")
@legacy_router.get("/ready")
@api_router.get("/live")
@legacy_router.get("/live")
async def health():
    """Health, readiness, and liveness endpoints for orchestration platforms."""
    return {
        "status": "ok",
        "service": "distilkit",
        "version": "0.1.0",
    }


@api_router.get("/metrics")
@legacy_router.get("/metrics")
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

    for path, count in sorted(_metrics["requests_by_path"].items()):
        lines.append('distilkit_requests_per_path{path="' + path + '"} ' + str(count))
    lines.append("")
    return PlainTextResponse("\n".join(lines))


@api_router.get("/")
@legacy_router.get("/")
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
                    {"path": "/api/v1/config", "method": "GET", "description": "Available datasets, teachers, students"},
                    {"path": "/api/v1/train", "method": "POST", "description": "Start a new distillation task"},
                    {"path": "/api/v1/train/{task_id}", "method": "GET", "description": "Get task state"},
                    {"path": "/api/v1/train/{task_id}/stream", "method": "GET", "description": "SSE progress stream"},
                    {"path": "/api/v1/train/{task_id}/cancel", "method": "POST", "description": "Cancel a running task"},
                    {"path": "/api/v1/export/{task_id}", "method": "POST", "description": "Export trained model"},
                    {"path": "/api/v1/download/{filename}", "method": "GET", "description": "Download exported file"},
                    {"path": "/api/v1/tasks", "method": "GET", "description": "List all tasks"},
                    {"path": "/api/v1/history", "method": "GET", "description": "Completed training runs"},
                ],
            }
        )
    try:
        html = TEMPLATE_FILE.read_text(encoding="utf-8")
    except (OSError, IOError) as e:
        logger.error("Failed to read template: %s", e)
        return HTMLResponse(
            "<html><body><h1>DistilKit</h1><p>Web GUI template unavailable.</p></body></html>",
            status_code=503,
        )
    return HTMLResponse(html)


@api_router.post("/train")
@legacy_router.post("/api/train")
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
@legacy_router.get("/api/train/{task_id}/stream")
async def stream_progress(
    task_id: str,
    request: Request,
    tasks: dict = Depends(get_tasks),
):
    """SSE endpoint for real-time training progress."""
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
        last_logs_len = 0

        while task.status in ("pending", "running"):
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
@legacy_router.get("/api/train/{task_id}")
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
@legacy_router.post("/api/export/{task_id}")
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
@legacy_router.get("/api/config")
async def get_config():
    """Return app configuration (teachers list, device, cache status)."""
    cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
    cached_models: dict[str, bool] = {}
    if os.path.isdir(cache_dir):
        cached_files = set(os.listdir(cache_dir))
        for model in ds.TEACHER_CHOICES:
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
@legacy_router.post("/api/train/{task_id}/cancel")
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
@legacy_router.get("/api/history")
async def get_history(
    _auth: None = Depends(require_api_key),
    history_store: list = Depends(get_history_store),
    limit: int = Query(default=50, ge=1, le=1000, description="Max items to return"),
    offset: int = Query(default=0, ge=0, description="Number of items to skip"),
):
    """Return completed training runs with pagination."""
    return JSONResponse(history_store[offset:][:limit], headers={"Cache-Control": "max-age=5"})


@api_router.get("/download/{filename}")
@legacy_router.get("/api/download/{filename}")
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
@legacy_router.get("/api/tasks")
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

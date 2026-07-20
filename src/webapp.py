"""DistilKit Web GUI — FastAPI + Tailwind CSS.

Run with:
    python -m src.webapp
    # or: uvicorn src.webapp:app --reload
"""

import asyncio
import io
import json
import os
import subprocess
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.middleware.gzip import GZipMiddleware

from src import datasets as ds
from src.benchmarks import compare_teacher_student
from src.distiller import Distiller
from src.log_config import logger
from src.onnx_export import export_to_onnx, export_to_torchscript
from src.student import build_student, STUDENT_REGISTRY
from src.teacher import load_teacher

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEVICE = "cpu"

# Maximum size of the training log buffer (in characters).
# Older entries are trimmed to prevent UI slowdown on long runs.
MAX_LOG_SIZE = 100_000

# ---------------------------------------------------------------------------
# API-only mode (no frontend)
# ---------------------------------------------------------------------------

API_ONLY = os.environ.get("API_ONLY", "").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# HTML template (served as static file)
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent
TEMPLATE_FILE = HERE / "templates" / "index.html"

# ---------------------------------------------------------------------------
# Run history (persisted to runs/ directory)
# ---------------------------------------------------------------------------

RUNS_DIR = "runs"

# ---------------------------------------------------------------------------
# Student model cache
# ---------------------------------------------------------------------------

_student_cache: dict[str, nn.Module] = {}


def _get_cached_student(
    teacher: nn.Module,
    student_type: str,
    compression_ratio: float,
    num_classes: int,
    in_channels: int,
) -> nn.Module:
    """Build and cache a student, or deep-copy a cached one."""
    import copy
    teacher_params = sum(p.numel() for p in teacher.parameters())
    key = f"{student_type}:{compression_ratio}:{num_classes}:{in_channels}:{teacher_params}"
    if key not in _student_cache:
        _student_cache[key] = build_student(
            teacher=teacher,
            student_type=student_type,
            compression_ratio=compression_ratio,
            num_classes=num_classes,
            in_channels=in_channels,
        )
    return copy.deepcopy(_student_cache[key])

# ---------------------------------------------------------------------------
# Student model cache: avoids rebuilding the same architecture repeatedly.
# Keyed by (student_type, compression_ratio, num_classes, in_channels).
# ---------------------------------------------------------------------------

_student_cache: dict[tuple, nn.Module] = {}


def _get_student(
    student_type: str,
    compression_ratio: float,
    num_classes: int,
    in_channels: int,
) -> nn.Module:
    """Return a cloned student model from cache, or build + cache a new one."""
    key = (student_type, compression_ratio, num_classes, in_channels)
    if key not in _student_cache:
        # Build a temporary teacher to compute compression ratio
        _student_cache[key] = STUDENT_REGISTRY[student_type](
            in_channels=in_channels, num_classes=num_classes, width=1.0
        )
    # Always return a fresh copy (deep copy the state dict into a new instance)
    base = STUDENT_REGISTRY[student_type](
        in_channels=in_channels, num_classes=num_classes,
        width=_compute_width(_student_cache[key], compression_ratio),
    )
    return base


def _compute_width(base_model: nn.Module, compression_ratio: float) -> float:
    """Estimate the width multiplier for a target compression ratio."""
    base_params = sum(p.numel() for p in base_model.parameters())
    if base_params == 0 or compression_ratio <= 0:
        return 1.0
    # Params scale roughly with width^2 → width ≈ sqrt(ratio * teacher_params / base_params)
    # Since we don't have teacher_params here, use compression_ratio directly
    return max(0.125, min(4.0, compression_ratio ** 0.5))


def _load_history() -> list[dict]:
    """Load completed runs from disk."""
    history = []
    if not os.path.isdir(RUNS_DIR):
        return history
    for fname in sorted(os.listdir(RUNS_DIR), reverse=True):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(RUNS_DIR, fname)) as f:
                    history.append(json.load(f))
            except Exception:
                pass
    return history


def _save_run(run_data: dict) -> None:
    """Persist a completed run to disk."""
    os.makedirs(RUNS_DIR, exist_ok=True)
    fname = f"{run_data['id']}.json"
    with open(os.path.join(RUNS_DIR, fname), "w") as f:
        json.dump(run_data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Training task manager
# ---------------------------------------------------------------------------

_tasks: dict[str, "TrainingTask"] = {}
_history: list[dict] = _load_history()


class TrainingTask:
    """Background training task with progress tracking."""

    def __init__(self, config: dict) -> None:
        """Initialize a training task with the given configuration.

        Args:
            config: Training parameters (dataset, teacher, epochs, etc.).
        """
        self.id = uuid.uuid4().hex[:12]
        self.config = config
        self.status = "pending"  # pending → running → completed | failed
        self.progress = 0.0  # 0.0 – 1.0
        self.current_epoch = 0
        self.total_epochs = config["epochs"]
        self.current_loss: float | None = None
        self.current_acc: float | None = None
        self.losses: list[float] = []
        self.accuracies: list[float] = []
        self.logs = ""
        self.result: dict | None = None
        self.error: str | None = None
        self.student: nn.Module | None = None
        self._thread: threading.Thread | None = None
        self._log_buffer = io.StringIO()
        self._cancel_requested = False
        self._subprocess: subprocess.Popen | None = None
        self.eta_seconds: float = 0.0
        self._epoch_times: list[float] = []
        self.created_at = datetime.now()
        self._dirty = False

    def cancel(self) -> None:
        """Cancel a running training task."""
        self._cancel_requested = True
        # Kill subprocess (wget/curl) if running
        if self._subprocess and self._subprocess.poll() is None:
            self._subprocess.terminate()
            try:
                self._subprocess.wait(timeout=5)
            except Exception:
                self._subprocess.kill()
        self.status = "cancelled"
        self._emit("\n⛔ Training cancelled.")
        self._flush_logs()

    def start(self) -> None:
        """Start the training task in a background thread."""
        self.status = "running"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _emit(self, msg: str) -> None:
        """Write to both logging and the task log buffer."""
        logger.info(msg)
        self._log_buffer.write(msg + "\n")
        self._dirty = True

    def _flush_logs(self) -> None:
        """Transfer accumulated buffer to the logs string, capping total size."""
        self.logs += self._log_buffer.getvalue()
        self._log_buffer.truncate(0)
        self._log_buffer.seek(0)
        # Trim oldest logs if over the limit
        if len(self.logs) > MAX_LOG_SIZE:
            self.logs = self.logs[-MAX_LOG_SIZE:]

    def _prepare_dataset(self, dataset_name: str, data_root: str) -> tuple | None:
        """Delegate to ``datasets.get_dataset_loaders`` with cancel support."""
        subprocess_tracker: list = []
        self._subprocess = subprocess_tracker

        def cancel_flag() -> bool:
            return self._cancel_requested

        result = ds.get_dataset_loaders(
            dataset_name,
            self.config["batch_size"],
            data_root,
            cancel_flag=cancel_flag,
            subprocess_tracker=subprocess_tracker,
        )
        self._subprocess = None
        return result

    def _run(self) -> None:
        """Execute the full distillation pipeline."""
        try:
            # --- Data ---
            dataset_name = self.config.get("dataset", "CIFAR-10")
            self.progress = 0.02
            self._emit(f"📦 Preparing {dataset_name}...")

            result = self._prepare_dataset(dataset_name, "./data")
            if result is None:
                if self._cancel_requested:
                    self._emit("⛔ Cancelled.")
                else:
                    self._emit("❌ Dataset preparation failed.")
                if self.status not in ("cancelled", "failed"):
                    self.status = "cancelled" if self._cancel_requested else "failed"
                    self._flush_logs()
                    _save_run(
                        {
                            "id": self.id,
                            "timestamp": datetime.now().isoformat(),
                            "config": self.config,
                            "status": self.status,
                            "error": self.error,
                        }
                    )
                return

            train_loader, val_loader, num_classes, in_channels = result
            self._flush_logs()

            # --- Teacher ---
            self.progress = 0.10
            self._emit(f"🧠 Loading teacher ({self.config['teacher']})...")
            teacher = load_teacher(self.config["teacher"], num_classes=num_classes)
            teacher.to(DEVICE).eval()
            teacher_params = sum(p.numel() for p in teacher.parameters())
            self._emit(f"   Teacher parameters: {teacher_params:,}")
            self._flush_logs()

            # --- Student ---
            self.progress = 0.18
            student_name = self.config.get("student", "MiniCNN")
            self._emit(f"🔧 Building student ({student_name})...")
            student = _get_cached_student(
                teacher=teacher,
                student_type=student_name,
                compression_ratio=self.config.get("compression_ratio", 0.05),
                num_classes=num_classes,
                in_channels=in_channels,
            )
            student.to(DEVICE)
            student_params = sum(p.numel() for p in student.parameters())
            self._emit(f"   Student parameters: {student_params:,}")
            self._emit(f"   Compression ratio: {student_params / teacher_params:.2%}")
            self._flush_logs()

            # --- Distillation ---
            self.progress = 0.25
            self._emit(
                f"🔄 Distilling (T={self.config['temperature']}, α={self.config['alpha']})...\n"
            )
            self._flush_logs()

            distiller = Distiller(
                teacher,
                student,
                temperature=self.config["temperature"],
                alpha=self.config["alpha"],
                device=DEVICE,
            )

            # ── Checkpoint directory ──
            ckpt_dir = "checkpoints"
            ckpt_every = self.config.get("ckpt_every", 5)
            os.makedirs(ckpt_dir, exist_ok=True)

            optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config["epochs"])

            start_epoch = 0
            # Resume from checkpoint if provided
            resume_from = self.config.get("resume")
            if resume_from and os.path.exists(resume_from):
                self._emit(f"📂 Resuming from checkpoint: {resume_from}")
                self._flush_logs()
                ckpt = torch.load(resume_from, map_location=DEVICE, weights_only=False)
                student.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
                start_epoch = ckpt["epoch"]
                self.losses = ckpt.get("losses", [])
                self.accuracies = ckpt.get("accuracies", [])
                self._emit(f"   Resumed at epoch {start_epoch}/{self.config['epochs']}")
                self._flush_logs()

            for epoch in range(start_epoch, self.config["epochs"]):
                if self._cancel_requested:
                    self._emit("\n⛔ Training cancelled during epoch.")
                    self._flush_logs()
                    return

                import time as _time

                _epoch_start = _time.time()

                # --- Train ---
                student.train()
                epoch_loss = 0.0
                num_batches = len(train_loader)

                for batch_idx, (images, labels) in enumerate(train_loader):
                    images, labels = images.to(DEVICE), labels.to(DEVICE)

                    with torch.no_grad():
                        teacher_logits = teacher(images)

                    student_logits = student(images)
                    loss = distiller.criterion(student_logits, teacher_logits, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                    # Sub-epoch progress (within the 25-75% range)
                    sub_progress = (batch_idx + 1) / num_batches
                    self.progress = 0.25 + 0.50 * (epoch + sub_progress) / self.config["epochs"]

                avg_loss = epoch_loss / num_batches
                self.losses.append(avg_loss)
                self.current_loss = avg_loss

                self._dirty = True

                # --- Validate ---
                student.eval()
                correct = total = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(DEVICE), labels.to(DEVICE)
                        outputs = student(images)
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
                acc = correct / total
                self.accuracies.append(acc)
                self.current_acc = acc

                self._dirty = True

                # --- Early stopping ---
                patience = self.config.get("patience", 0)
                if patience > 0:
                    if not hasattr(self, "_best_acc"):
                        self._best_acc = 0.0
                        self._patience_counter = 0
                    if acc > self._best_acc + 0.001:
                        self._best_acc = acc
                        self._patience_counter = 0
                    else:
                        self._patience_counter += 1
                        if self._patience_counter >= patience:
                            self._emit(
                                f"   ⏹️ Early stopping (no improvement for {patience} epochs, "
                                f"best: {self._best_acc:.2%})"
                            )
                            self._flush_logs()
                            break

                scheduler.step()

                # --- Update state ---
                self.current_epoch = epoch + 1
                self._emit(
                    f"Epoch {epoch + 1}/{self.config['epochs']} — "
                    f"Loss: {avg_loss:.4f} — Val Acc: {acc:.2%}"
                )

                # --- ETA ---
                _elapsed = _time.time() - _epoch_start
                self._epoch_times.append(_elapsed)
                avg_epoch = sum(self._epoch_times) / len(self._epoch_times)
                remaining = self.config["epochs"] - (epoch + 1)
                self.eta_seconds = avg_epoch * remaining

                # --- Checkpoint ---
                if ckpt_every > 0 and (epoch + 1) % ckpt_every == 0:
                    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model": student.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "losses": self.losses,
                            "accuracies": self.accuracies,
                            "config": self.config,
                        },
                        ckpt_path,
                    )
                    self._emit(f"   💾 Checkpoint saved: {ckpt_path}")

                self._flush_logs()

            # --- Benchmark ---
            self.progress = 0.90
            self._emit("\n📊 Benchmarking teacher vs. student (CPU)...")
            self._flush_logs()
            comparison = compare_teacher_student(teacher, student, target="cpu")
            self._emit(
                f"   Teacher: {comparison['teacher']['mean_ms']:.2f} ms  "
                f"({comparison['teacher']['throughput_imgs_per_sec']:.0f} img/s)"
            )
            self._emit(
                f"   Student: {comparison['student']['mean_ms']:.2f} ms  "
                f"({comparison['student']['throughput_imgs_per_sec']:.0f} img/s)"
            )
            self._emit(f"   ⚡ Speedup: {comparison['speedup']}x")
            self._emit(f"   📦 Size: {comparison['compression']:.2%} of teacher")
            self._flush_logs()

            # --- Build result ---
            self.progress = 0.95
            self.result = {
                "teacher_params": teacher_params,
                "student_params": student_params,
                "compression_pct": round((1 - student_params / teacher_params) * 100, 1),
                "speedup": comparison["speedup"],
                "teacher_latency_ms": comparison["teacher"]["mean_ms"],
                "student_latency_ms": comparison["student"]["mean_ms"],
                "teacher_throughput": comparison["teacher"]["throughput_imgs_per_sec"],
                "student_throughput": comparison["student"]["throughput_imgs_per_sec"],
                "final_loss": round(self.losses[-1], 4),
                "final_accuracy": round(self.accuracies[-1], 4),
                "losses": [round(loss_val, 4) for loss_val in self.losses],
                "accuracies": [round(a, 4) for a in self.accuracies],
                "dataset": self.config.get("dataset", "CIFAR-10"),
                "teacher_name": self.config["teacher"],
                "student_name": self.config.get("student", "MiniCNN"),
            }

            self.student = student
            self.status = "completed"
            self.progress = 1.0
            self._emit("\n✅ Training complete!")
            self._flush_logs()

            # Persist to history
            run = {
                "id": self.id,
                "timestamp": datetime.now().isoformat(),
                "config": self.config,
                "result": self.result,
            }
            _save_run(run)
            _history.insert(0, run)

        except Exception as e:
            self.status = "failed"
            self.error = str(e)
            import traceback

            tb = traceback.format_exc()
            self._emit(f"\n❌ Error: {e}")
            self._emit(tb)
            _save_run(
                {
                    "id": self.id,
                    "timestamp": datetime.now().isoformat(),
                    "config": self.config,
                    "status": "failed",
                    "error": str(e),
                }
            )
        finally:
            self._flush_logs()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


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


# ---- Routes ----


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main page (HTML) or API info (JSON) in API-only mode."""
    if API_ONLY:
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


@app.post("/api/train")
async def start_training(body: dict):
    """Start a new distillation task."""
    config = {
        "dataset": body.get("dataset", "CIFAR-10"),
        "teacher": body.get("teacher", "resnet18"),
        "student": body.get("student", "MiniCNN"),
        "compression_ratio": float(body.get("compression_ratio", 0.05)),
        "epochs": int(body.get("epochs", 10)),
        "temperature": float(body.get("temperature", 4.0)),
        "alpha": float(body.get("alpha", 0.7)),
        "patience": int(body.get("patience", 0)),
        "batch_size": int(body.get("batch_size", 64)),
    }

    if config["dataset"] not in ds.DATASETS:
        raise HTTPException(400, f"Invalid dataset. Choose: {ds.DATASET_CHOICES}")
    if config["teacher"] not in ds.TEACHER_CHOICES:
        raise HTTPException(400, f"Invalid teacher. Choose: {ds.TEACHER_CHOICES}")
    if config["student"] not in ds.STUDENT_CHOICES:
        raise HTTPException(400, f"Invalid student. Choose: {ds.STUDENT_CHOICES}")

    task = TrainingTask(config)
    _tasks[task.id] = task
    task.start()

    return {"task_id": task.id}


@app.get("/api/train/{task_id}/stream")
async def stream_progress(task_id: str):
    """SSE endpoint for real-time training progress."""
    task = _tasks.get(task_id)
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
async def get_task(task_id: str):
    """Get the current state of a task."""
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    return {
        "id": task.id,
        "status": task.status,
        "progress": task.progress,
        "current_epoch": task.current_epoch,
        "total_epochs": task.total_epochs,
        "logs": task.logs[-2000:],  # Last 2KB
        "result": task.result,
        "error": task.error,
    }


@app.post("/api/export/{task_id}")
async def export_model(task_id: str, body: dict):
    """Export the trained student model."""
    task = _tasks.get(task_id)
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
            export_to_onnx(task.student, filepath)
        elif fmt == "torchscript":
            export_to_torchscript(task.student, filepath)
        else:
            raise HTTPException(400, "Invalid format. Use 'onnx' or 'torchscript'")
        return {"filename": filename, "path": filepath, "format": fmt}
    except Exception as e:
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
        "device": DEVICE,
        "cached_teachers": cached_models,
    }


@app.post("/api/train/{task_id}/cancel")
async def cancel_training(task_id: str):
    """Cancel a running training task."""
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task.status not in ("pending", "running"):
        raise HTTPException(400, f"Task is already {task.status}")
    task.cancel()
    return {"status": "cancelled"}


@app.get("/api/history")
async def get_history():
    """Return all completed training runs."""
    return _history


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
async def list_tasks():
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
        for t in _tasks.values()
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def launch(port: int = 7860, host: str = "127.0.0.1", api_only: bool = False) -> None:
    """Launch the web server.

    Args:
        port: Server port.
        host: Bind address.
        api_only: If True, only expose the REST API (no frontend).
    """
    import uvicorn

    global API_ONLY
    if api_only:
        API_ONLY = True

    mode = "API-only" if API_ONLY else "Web GUI"
    logger.info(f"⚡ DistilKit {mode}")
    logger.info(f"   → http://{host}:{port}")
    logger.info("   → Press Ctrl+C to stop\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    launch()

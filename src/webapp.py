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
import sys
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.teacher import load_teacher
from src.student import build_student
from src.distiller import Distiller
from src.benchmarks import compare_teacher_student
from src.onnx_export import export_to_onnx, export_to_torchscript

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.247, 0.243, 0.261)
TEACHER_CHOICES = [
    "resnet18", "resnet34", "resnet50", "resnet101",
    "mobilenet_v2", "mobilenet_v3_large",
    "efficientnet_b0", "efficientnet_b1",
]
DEVICE = "cpu"  # CPU-only mode (see: src/distiller.py)
STUDENT_CHOICES = ["MiniCNN", "MiniResNet"]

# ---------------------------------------------------------------------------
# HTML template (served as static file)
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent
TEMPLATE_FILE = HERE / "templates" / "index.html"

# ---------------------------------------------------------------------------
# Training task manager
# ---------------------------------------------------------------------------

_tasks: dict[str, "TrainingTask"] = {}


class TrainingTask:
    """Background training task with progress tracking."""

    def __init__(self, config: dict):
        self.id = uuid.uuid4().hex[:12]
        self.config = config
        self.status = "pending"       # pending → running → completed | failed
        self.progress = 0.0           # 0.0 – 1.0
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
        self.created_at = datetime.now()

    def cancel(self):
        """Cancel a running training task."""
        self._cancel_requested = True
        # Kill subprocess (wget/curl) if running
        if self._subprocess and self._subprocess.poll() is None:
            self._subprocess.terminate()
            try:
                self._subprocess.wait(timeout=5)
            except:
                self._subprocess.kill()
        self.status = "cancelled"
        self._emit("\n⛔ Training cancelled.")
        self._flush_logs()

    def start(self):
        self.status = "running"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _emit(self, msg: str):
        """Write to both real stdout and the log buffer."""
        print(msg)
        self._log_buffer.write(msg + "\n")

    def _flush_logs(self):
        """Transfer accumulated buffer to the logs string."""
        self.logs += self._log_buffer.getvalue()
        self._log_buffer.truncate(0)
        self._log_buffer.seek(0)

    def _download_and_extract_cifar10(self, data_root: str) -> bool:
        """Download + extract CIFAR-10. Tries aria2c, mirrors, wget, curl, Python."""
        import hashlib
        import tarfile

        cifar_urls = [
            # Primary — University of Toronto
            "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
            # Mirror — Oxford Robotics
            "https://thor.robots.ox.ac.uk/pascal/data/cifar10/cifar-10-python.tar.gz",
        ]
        cifar_tgz = os.path.join(data_root, "cifar-10-python.tar.gz")
        extracted_dir = os.path.join(data_root, "cifar-10-batches-py")
        expected_md5 = "c58f30108f718f92721af3b95e74349a"
        expected_size = 170_498_071
        os.makedirs(data_root, exist_ok=True)

        # Already extracted
        if os.path.isdir(extracted_dir):
            return True

        # Valid tar.gz from a previous attempt — just extract
        if os.path.exists(cifar_tgz) and os.path.getsize(cifar_tgz) == expected_size:
            self._emit("   Extracting previously downloaded file...")
            self._flush_logs()
            with tarfile.open(cifar_tgz, "r:gz") as tar:
                tar.extractall(path=data_root)
            self._emit("   ✅ CIFAR-10 ready!")
            self._flush_logs()
            return True

        # Delete partial file
        if os.path.exists(cifar_tgz):
            old_size = os.path.getsize(cifar_tgz)
            self._emit(f"   Removing partial download ({old_size/1e6:.0f} MB)...")
            os.remove(cifar_tgz)

        self._emit("⬇️ Downloading CIFAR-10 (170 MB)...")
        self._flush_logs()

        # --- Detect available download tools (ordered by speed) ---
        has_aria2c = (
            subprocess.run(["which", "aria2c"], capture_output=True).returncode == 0
        )
        has_wget = (
            subprocess.run(["which", "wget"], capture_output=True).returncode == 0
        )
        has_curl = (
            subprocess.run(["which", "curl"], capture_output=True).returncode == 0
        )

        # --- Try each URL with the fastest available tool ---
        downloaded_ok = False
        for url in cifar_urls:
            if self._cancel_requested:
                return False
            if downloaded_ok:
                break

            if has_aria2c:
                self._emit(f"   Trying aria2c (4 connections)...")
                self._flush_logs()
                self._subprocess = subprocess.Popen([
                    "aria2c", "-x", "4", "-s", "4",
                    "-d", data_root, "-o", "cifar-10-python.tar.gz",
                    url,
                ])
                self._subprocess.wait()
                self._subprocess = None
                if os.path.exists(cifar_tgz) and os.path.getsize(cifar_tgz) == expected_size:
                    downloaded_ok = True
                    break
                if self._cancel_requested:
                    return False

            if has_wget:
                self._emit(f"   Trying wget...")
                self._flush_logs()
                self._subprocess = subprocess.Popen([
                    "wget", "-O", cifar_tgz, "--show-progress", url,
                ])
                self._subprocess.wait()
                self._subprocess = None
                if os.path.exists(cifar_tgz) and os.path.getsize(cifar_tgz) == expected_size:
                    downloaded_ok = True
                    break
                if self._cancel_requested:
                    return False

            if has_curl and not downloaded_ok:
                self._emit(f"   Trying curl...")
                self._flush_logs()
                self._subprocess = subprocess.Popen([
                    "curl", "-#", "-Lo", cifar_tgz, url,
                ])
                self._subprocess.wait()
                self._subprocess = None
                if os.path.exists(cifar_tgz) and os.path.getsize(cifar_tgz) == expected_size:
                    downloaded_ok = True
                    break
                if self._cancel_requested:
                    return False

        # --- Pure Python fallback ---
        if not downloaded_ok:
            import urllib.request
            self._emit("   Trying Python (fallback)...")
            self._flush_logs()
            for url in cifar_urls:
                if self._cancel_requested:
                    return False
                try:
                    req = urllib.request.Request(url, headers={
                        "User-Agent": "Mozilla/5.0"
                    })
                    response = urllib.request.urlopen(req)
                    total = int(response.headers.get("Content-Length", expected_size))
                    downloaded = 0
                    chunk_size = 8192
                    with open(cifar_tgz, "wb") as f:
                        while True:
                            if self._cancel_requested:
                                return False
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            if downloaded % (chunk_size * 200) == 0:
                                pct = min(downloaded / total * 100, 100)
                                self._emit(f"   {pct:.0f}% ({downloaded/1e6:.0f}/{total/1e6:.0f} MB)")
                    if os.path.getsize(cifar_tgz) == expected_size:
                        downloaded_ok = True
                        break
                except Exception as e:
                    self._emit(f"   Error: {e}")
                    continue

        if not downloaded_ok:
            self._emit("❌ Could not download from any server.")
            return False

        if self._cancel_requested:
            return False

        # --- Verify MD5 ---
        self._emit("   Verifying integrity...")
        self._flush_logs()
        md5_actual = hashlib.md5(open(cifar_tgz, "rb").read()).hexdigest()
        if md5_actual != expected_md5:
            self._emit(f"   ⚠️  MD5 mismatch — file may be corrupted, will retry next time.")
            os.remove(cifar_tgz)
            return False

        # --- Extract ---
        self._emit("   Extracting...")
        self._flush_logs()
        with tarfile.open(cifar_tgz, "r:gz") as tar:
            tar.extractall(path=data_root)

        self._emit("   ✅ CIFAR-10 ready!")
        self._flush_logs()
        return True

    def _run(self):
        """Execute the full distillation pipeline."""
        try:
            # --- Data ---
            self.progress = 0.02
            self._emit("📦 Preparing CIFAR-10...")

            if not self._download_and_extract_cifar10("./data"):
                if self._cancel_requested:
                    self._emit("⛔ Cancelled during download.")
                else:
                    self._emit("❌ Download failed.")
                self.status = "cancelled" if self._cancel_requested else "failed"
                self._flush_logs()
                return

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])
            transform_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])
            train_set = datasets.CIFAR10(
                root="./data", train=True, download=False, transform=transform_train
            )
            val_set = datasets.CIFAR10(
                root="./data", train=False, download=False, transform=transform_val
            )
            train_loader = DataLoader(
                train_set, batch_size=self.config["batch_size"], shuffle=True, num_workers=2
            )
            val_loader = DataLoader(
                val_set, batch_size=self.config["batch_size"], shuffle=False, num_workers=2
            )
            self._flush_logs()

            # --- Teacher ---
            self.progress = 0.10
            self._emit(f"🧠 Loading teacher ({self.config['teacher']})...")
            teacher = load_teacher(self.config["teacher"], num_classes=10)
            teacher.to(DEVICE).eval()
            teacher_params = sum(p.numel() for p in teacher.parameters())
            self._emit(f"   Teacher parameters: {teacher_params:,}")
            self._flush_logs()

            # --- Student ---
            self.progress = 0.18
            student_name = self.config.get("student", "MiniCNN")
            self._emit(f"🔧 Building student ({student_name})...")
            student = build_student(
                student_type=student_name, num_classes=10
            )
            student.to(DEVICE)
            student_params = sum(p.numel() for p in student.parameters())
            self._emit(f"   Student parameters: {student_params:,}")
            self._emit(f"   Compression ratio: {student_params/teacher_params:.2%}")
            self._flush_logs()

            # --- Distillation ---
            self.progress = 0.25
            self._emit(f"🔄 Distilling (T={self.config['temperature']}, α={self.config['alpha']})...\n")
            self._flush_logs()

            distiller = Distiller(
                teacher, student,
                temperature=self.config["temperature"],
                alpha=self.config["alpha"],
                device=DEVICE,
            )

            optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.config["epochs"]
            )

            for epoch in range(self.config["epochs"]):
                if self._cancel_requested:
                    self._emit("\n⛔ Training cancelled during epoch.")
                    self._flush_logs()
                    return

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

                scheduler.step()

                # --- Update state ---
                self.current_epoch = epoch + 1
                self._emit(
                    f"Epoch {epoch+1}/{self.config['epochs']} — "
                    f"Loss: {avg_loss:.4f} — Val Acc: {acc:.2%}"
                )
                self._flush_logs()

            # --- Benchmark ---
            self.progress = 0.90
            self._emit("\n📊 Benchmarking teacher vs. student (CPU)...")
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
                "losses": [round(l, 4) for l in self.losses],
                "accuracies": [round(a, 4) for a in self.accuracies],
                "teacher_name": self.config["teacher"],
                "student_name": self.config.get("student", "MiniCNN"),
            }

            self.student = student
            self.status = "completed"
            self.progress = 1.0
            self._emit("\n✅ Training complete!")

        except Exception as e:
            self.status = "failed"
            self.error = str(e)
            import traceback
            tb = traceback.format_exc()
            self._emit(f"\n❌ Error: {e}")
            self._emit(tb)
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


# ---- Routes ----


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main page."""
    html = TEMPLATE_FILE.read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.post("/api/train")
async def start_training(body: dict):
    """Start a new distillation task."""
    config = {
        "teacher": body.get("teacher", "resnet18"),
        "student": body.get("student", "MiniCNN"),
        "epochs": int(body.get("epochs", 10)),
        "temperature": float(body.get("temperature", 4.0)),
        "alpha": float(body.get("alpha", 0.7)),
        "batch_size": int(body.get("batch_size", 64)),
    }

    if config["teacher"] not in TEACHER_CHOICES:
        raise HTTPException(400, f"Invalid teacher. Choose: {TEACHER_CHOICES}")
    if config["student"] not in STUDENT_CHOICES:
        raise HTTPException(400, f"Invalid student. Choose: {STUDENT_CHOICES}")

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
        last_logs_len = 0

        while task.status in ("pending", "running"):
            new_logs = task.logs[last_logs_len:]
            last_logs_len = len(task.logs)

            data = {
                "status": task.status,
                "progress": round(task.progress, 3),
                "current_epoch": task.current_epoch,
                "total_epochs": task.total_epochs,
                "current_loss": task.current_loss,
                "current_acc": task.current_acc,
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

            await asyncio.sleep(0.3)

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

    try:
        if fmt == "onnx":
            path = export_to_onnx(task.student, "checkpoints/student.onnx")
        elif fmt == "torchscript":
            path = export_to_torchscript(task.student, "checkpoints/student.pt")
        else:
            raise HTTPException(400, "Invalid format. Use 'onnx' or 'torchscript'")
        return {"path": str(path), "format": fmt}
    except Exception as e:
        raise HTTPException(500, f"Export failed: {e}")


@app.get("/api/config")
async def get_config():
    """Return app configuration (teachers list, device)."""
    return {
        "teachers": TEACHER_CHOICES,
        "students": STUDENT_CHOICES,
        "device": DEVICE,
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

def launch(port: int = 7860, host: str = "127.0.0.1"):
    """Launch the web GUI server."""
    import uvicorn
    print(f"⚡ DistilKit Web GUI")
    print(f"   → http://{host}:{port}")
    print(f"   → Press Ctrl+C to stop\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    launch()

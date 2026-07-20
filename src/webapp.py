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

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = {
    "CIFAR-10": {
        "num_classes": 10,
        "in_channels": 3,
        "input_size": 32,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.247, 0.243, 0.261),
        "module": "torchvision.datasets",
        "class_name": "CIFAR10",
        "url": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "md5": "c58f30108f718f92721af3b95e74349a",
        "filename": "cifar-10-python.tar.gz",
        "extracted_dir": "cifar-10-batches-py",
    },
    "MNIST": {
        "num_classes": 10,
        "in_channels": 1,
        "input_size": 28,
        "mean": (0.1307,),
        "std": (0.3081,),
        "module": "torchvision.datasets",
        "class_name": "MNIST",
    },
    "FashionMNIST": {
        "num_classes": 10,
        "in_channels": 1,
        "input_size": 28,
        "mean": (0.2860,),
        "std": (0.3530,),
        "module": "torchvision.datasets",
        "class_name": "FashionMNIST",
    },
    "SVHN": {
        "num_classes": 10,
        "in_channels": 3,
        "input_size": 32,
        "mean": (0.4377, 0.4438, 0.4728),
        "std": (0.1980, 0.2010, 0.1970),
        "module": "torchvision.datasets",
        "class_name": "SVHN",
        "extra_train": True,  # SVHN has an extra training set
    },
}

DATASET_CHOICES = list(DATASETS.keys())
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
# Run history (persisted to runs/ directory)
# ---------------------------------------------------------------------------

RUNS_DIR = "runs"


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
        self.eta_seconds: float = 0.0
        self._epoch_times: list[float] = []
        self.created_at = datetime.now()

    def cancel(self) -> None:
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

    def start(self) -> None:
        self.status = "running"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _emit(self, msg: str) -> None:
        """Write to both real stdout and the log buffer."""
        print(msg)
        self._log_buffer.write(msg + "\n")

    def _flush_logs(self) -> None:
        """Transfer accumulated buffer to the logs string."""
        self.logs += self._log_buffer.getvalue()
        self._log_buffer.truncate(0)
        self._log_buffer.seek(0)

    def _prepare_dataset(self, dataset_name: str, data_root: str) -> tuple | None:
        """Get (train_loader, val_loader, num_classes, in_channels) for any dataset.

        Returns None if preparation fails (download error, missing files, etc.).
        """
        info = DATASETS[dataset_name]
        ds_class = getattr(datasets, info["class_name"])
        num_classes = info["num_classes"]
        in_channels = info["in_channels"]
        input_size = info["input_size"]
        mean, std = info["mean"], info["std"]
        ds_root = os.path.join(data_root, dataset_name)
        os.makedirs(ds_root, exist_ok=True)

        # ── Verify / Download ──
        try:
            if dataset_name == "CIFAR-10":
                self._download_cifar10(ds_root, info)
                # After download attempt, verify extraction exists
                extracted = os.path.join(ds_root, info["extracted_dir"])
                if not os.path.isdir(extracted):
                    self._emit("❌ CIFAR-10 data not found after download.")
                    self._flush_logs()
                    return None
            else:
                # For MNIST/FashionMNIST/SVHN, check if already downloaded first
                has_files = self._check_torchvision_dataset(dataset_name, ds_root, info)
                if not has_files:
                    self._emit(f"⬇️ Downloading {dataset_name}...")
                    self._flush_logs()
        except (OSError, IOError, RuntimeError) as e:
            self._emit(f"❌ Dataset I/O error: {e}")
            self._flush_logs()
            return None

        # ── Transforms ──
        if input_size <= 32:
            train_transform = transforms.Compose([
                transforms.RandomCrop(input_size, padding=4 if input_size >= 28 else 2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # ── Load datasets ──
        try:
            if dataset_name == "SVHN":
                train_set = ds_class(
                    root=ds_root, split="train", download=False, transform=train_transform
                )
                val_set = ds_class(
                    root=ds_root, split="test", download=False, transform=val_transform
                )
            else:
                train_set = ds_class(
                    root=ds_root, train=True, download=False, transform=train_transform
                )
                val_set = ds_class(
                    root=ds_root, train=False, download=False, transform=val_transform
                )
        except (OSError, RuntimeError) as e:
            self._emit(f"❌ Failed to load dataset: {e}")
            self._flush_logs()
            return None

        train_loader = DataLoader(
            train_set, batch_size=self.config["batch_size"],
            shuffle=True, num_workers=0  # 0 to avoid file-lock issues
        )
        val_loader = DataLoader(
            val_set, batch_size=self.config["batch_size"],
            shuffle=False, num_workers=0
        )

        return train_loader, val_loader, num_classes, in_channels

    def _check_torchvision_dataset(self, name: str, root: str, info: dict) -> bool:
        """Check if a torchvision dataset's raw files exist on disk."""
        raw_dir = os.path.join(root, "raw")
        if os.path.isdir(raw_dir) and len(os.listdir(raw_dir)) > 0:
            return True
        # Also check processed/
        processed_dir = os.path.join(root, info["class_name"] if name != "SVHN" else "".join(c for c in name if c.isalnum()), "processed")
        processed_dir = os.path.join(root, "processed")
        if os.path.isdir(processed_dir) and len(os.listdir(processed_dir)) > 0:
            return True
        return False

    def _download_cifar10(self, ds_root: str, info: dict) -> None:
        """Optimised download for CIFAR-10 using aria2c/wget with fallback."""
        import hashlib
        import tarfile

        cifar_urls = [
            "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
            "https://thor.robots.ox.ac.uk/pascal/data/cifar10/cifar-10-python.tar.gz",
        ]
        cifar_tgz = os.path.join(ds_root, info["filename"])
        extracted_dir = os.path.join(ds_root, info["extracted_dir"])
        expected_size = 170_498_071
        os.makedirs(ds_root, exist_ok=True)

        if os.path.isdir(extracted_dir):
            return
        if os.path.exists(cifar_tgz) and os.path.getsize(cifar_tgz) == expected_size:
            self._emit("   Extracting previously downloaded file...")
            self._flush_logs()
            with tarfile.open(cifar_tgz, "r:gz") as tar:
                tar.extractall(path=ds_root)
            self._emit("   ✅ CIFAR-10 ready!")
            self._flush_logs()
            return
        if os.path.exists(cifar_tgz):
            self._emit(f"   Removing partial download ({os.path.getsize(cifar_tgz)/1e6:.0f} MB)...")
            os.remove(cifar_tgz)

        self._emit("⬇️ Downloading CIFAR-10 (170 MB)...")
        self._flush_logs()

        has_aria2c = subprocess.run(["which", "aria2c"], capture_output=True).returncode == 0
        has_wget = subprocess.run(["which", "wget"], capture_output=True).returncode == 0
        has_curl = subprocess.run(["which", "curl"], capture_output=True).returncode == 0

        max_retries = 3
        base_delay = 2  # seconds

        for attempt in range(1, max_retries + 1):
            if self._cancel_requested:
                return

            if attempt > 1:
                delay = base_delay * (2 ** (attempt - 2))  # 2, 4, 8
                self._emit(f"   Retry {attempt}/{max_retries} in {delay}s...")
                self._flush_logs()
                import time as _time
                _time.sleep(delay)
                if self._cancel_requested:
                    return
                # Remove partial file from previous attempt
                if os.path.exists(cifar_tgz):
                    os.remove(cifar_tgz)

            downloaded_ok = False
            for url in cifar_urls:
                if self._cancel_requested:
                    return
                if downloaded_ok:
                    break

                if has_aria2c:
                    self._emit("   Trying aria2c (4 connections)...")
                    self._flush_logs()
                    self._subprocess = subprocess.Popen([
                        "aria2c", "-x", "4", "-s", "4",
                        "-d", ds_root, "-o", info["filename"], url,
                    ])
                    self._subprocess.wait()
                    self._subprocess = None
                    if os.path.exists(cifar_tgz) and os.path.getsize(cifar_tgz) == expected_size:
                        downloaded_ok = True
                        break
                    if self._cancel_requested:
                        return

                if has_wget:
                    self._emit("   Trying wget...")
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
                        return

                if has_curl:
                    self._emit("   Trying curl...")
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
                        return

            if not downloaded_ok:
                import urllib.request
                self._emit("   Trying Python (fallback)...")
                self._flush_logs()
                for url in cifar_urls:
                    if self._cancel_requested:
                        return
                    try:
                        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                        resp = urllib.request.urlopen(req)
                        total = int(resp.headers.get("Content-Length", expected_size))
                        downloaded, chunk = 0, 8192
                        with open(cifar_tgz, "wb") as f:
                            while True:
                                if self._cancel_requested:
                                    return
                                data = resp.read(chunk)
                                if not data:
                                    break
                                f.write(data)
                                downloaded += len(data)
                                if downloaded % (chunk * 200) == 0:
                                    self._emit(f"   {min(downloaded/total*100,100):.0f}%")
                        if os.path.getsize(cifar_tgz) == expected_size:
                            downloaded_ok = True
                            break
                    except Exception as e:
                        self._emit(f"   Error: {e}")
                        continue

            if downloaded_ok:
                break

        if not downloaded_ok:
            self._emit("❌ All retries exhausted. Could not download from any server.")
            self.status = "failed"
            self._flush_logs()
            return

        md5_actual = hashlib.md5(open(cifar_tgz, "rb").read()).hexdigest()
        if md5_actual != info["md5"]:
            self._emit("⚠️  MD5 mismatch — file corrupted, will retry next time.")
            os.remove(cifar_tgz)
            self.status = "failed"
            self._flush_logs()
            return

        self._emit("   Extracting...")
        self._flush_logs()
        with tarfile.open(cifar_tgz, "r:gz") as tar:
            tar.extractall(path=ds_root)
        self._emit("✅ CIFAR-10 ready!")
        self._flush_logs()

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
                    _save_run({
                        "id": self.id,
                        "timestamp": datetime.now().isoformat(),
                        "config": self.config,
                        "status": self.status,
                        "error": self.error,
                    })
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
            student = build_student(
                teacher=teacher,
                student_type=student_name,
                compression_ratio=self.config.get("compression_ratio", 0.05),
                num_classes=num_classes,
                in_channels=in_channels,
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

            # ── Checkpoint directory ──
            ckpt_dir = "checkpoints"
            ckpt_every = self.config.get("ckpt_every", 5)
            os.makedirs(ckpt_dir, exist_ok=True)

            optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.config["epochs"]
            )

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
                    f"Epoch {epoch+1}/{self.config['epochs']} — "
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
                    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch+1}.pt")
                    torch.save({
                        "epoch": epoch + 1,
                        "model": student.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "losses": self.losses,
                        "accuracies": self.accuracies,
                        "config": self.config,
                    }, ckpt_path)
                    self._emit(f"   💾 Checkpoint saved: {ckpt_path}")

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
            _save_run({
                "id": self.id,
                "timestamp": datetime.now().isoformat(),
                "config": self.config,
                "status": "failed",
                "error": str(e),
            })
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

    if config["dataset"] not in DATASETS:
        raise HTTPException(400, f"Invalid dataset. Choose: {DATASET_CHOICES}")
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
    """Return app configuration (teachers list, device)."""
    return {
        "datasets": DATASET_CHOICES,
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
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
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

def launch(port: int = 7860, host: str = "127.0.0.1") -> None:
    """Launch the web GUI server."""
    import uvicorn
    print(f"⚡ DistilKit Web GUI")
    print(f"   → http://{host}:{port}")
    print(f"   → Press Ctrl+C to stop\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    launch()

"""Background training task management for the DistilKit web GUI.

Provides the ``TrainingTask`` class, history persistence helpers, and
FastAPI-compatible dependency providers.  Extracted from ``webapp.py``
so that file stays under the 400-line limit.
"""

import io
import json
import os
import subprocess
import threading
import uuid
from datetime import datetime

import torch.nn as nn

from src.alert_manager import record_task_failure
from src.log_config import logger
from src.pipeline import DatasetError, PipelineError, run_distillation_pipeline
from src.settings import settings

# ---------------------------------------------------------------------------
# History persistence
# ---------------------------------------------------------------------------


def _load_history() -> list[dict]:
    """Load completed runs from disk."""
    history: list[dict] = []
    if not os.path.isdir(settings.runs_dir):
        return history
    for fname in sorted(os.listdir(settings.runs_dir), reverse=True):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(settings.runs_dir, fname)) as f:
                    history.append(json.load(f))
            except (OSError, ValueError) as e:
                logger.warning(f"Skipping corrupted history file {fname}: {e}")
    return history


def _save_run(run_data: dict) -> None:
    """Persist a completed run to disk."""
    os.makedirs(settings.runs_dir, exist_ok=True)
    fname = f"{run_data['id']}.json"
    with open(os.path.join(settings.runs_dir, fname), "w") as f:
        json.dump(run_data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_tasks: dict[str, "TrainingTask"] = {}
_history: list[dict] = _load_history()


# ── Dependency providers (used via FastAPI ``Depends()``) ────────


def get_tasks() -> dict[str, "TrainingTask"]:
    """Provide the task store — can be overridden in tests."""
    return _tasks


def get_history_store() -> list[dict]:
    """Provide the history store — can be overridden in tests."""
    return _history


# ---------------------------------------------------------------------------
# Training task
# ---------------------------------------------------------------------------


class TrainingTask:
    """Background training task with progress tracking."""

    def __init__(self, config: dict) -> None:
        """Initialize a training task with the given configuration."""
        self.id = uuid.uuid4().hex[:12]
        self.config = config
        self.status = "pending"  # pending → running → completed | failed
        self.progress = 0.0  # 0.0 - 1.0
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
            except subprocess.TimeoutExpired:
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
        """Write to both logging and the task log buffer with task ID context."""
        logger.bind(task_id=self.id[:8]).info(msg)
        self._log_buffer.write(msg + "\n")
        self._dirty = True

    def _flush_logs(self) -> None:
        """Transfer accumulated buffer to the logs string, capping total size."""
        self.logs += self._log_buffer.getvalue()
        self._log_buffer.truncate(0)
        self._log_buffer.seek(0)
        # Trim oldest logs if over the limit
        if len(self.logs) > settings.max_log_size:
            self.logs = self.logs[-settings.max_log_size :]

    # ── Result helpers ────────────────────────────────────────────

    @staticmethod
    def _build_result_dict(
        dataset_name: str,
        student_name: str,
        teacher_name: str,
        teacher_params: int,
        student_params: int,
        comparison: dict,
        losses: list[float],
        accuracies: list[float],
    ) -> dict:
        """Build the training result dict from pipeline outputs."""
        return {
            "teacher_params": teacher_params,
            "student_params": student_params,
            "compression_pct": round((1 - student_params / teacher_params) * 100, 1),
            "speedup": comparison["speedup"],
            "teacher_latency_ms": comparison["teacher"]["mean_ms"],
            "student_latency_ms": comparison["student"]["mean_ms"],
            "teacher_throughput": comparison["teacher"]["throughput_imgs_per_sec"],
            "student_throughput": comparison["student"]["throughput_imgs_per_sec"],
            "final_loss": round(losses[-1], 4),
            "final_accuracy": round(accuracies[-1], 4),
            "losses": [round(loss_val, 4) for loss_val in losses],
            "accuracies": [round(a, 4) for a in accuracies],
            "dataset": dataset_name,
            "teacher_name": teacher_name,
            "student_name": student_name,
        }

    def _save_error_run(self, status: str) -> None:
        """Persist a failed/cancelled run to disk."""
        _save_run(
            {
                "id": self.id,
                "timestamp": datetime.now().isoformat(),
                "config": self.config,
                "status": status,
                "error": self.error,
            }
        )

    def _run(self) -> None:
        """Execute the full distillation pipeline."""
        import time as _time

        # Set up dataset subprocess tracker for cancel support during download
        _subprocess_tracker: list = []
        self._subprocess = _subprocess_tracker

        dataset_name = self.config.get("dataset", "CIFAR-10")
        student_name = self.config.get("student", "MiniCNN")
        epochs = self.config["epochs"]
        _last_epoch_time = [_time.time()]

        try:
            self.progress = 0.02
            self._emit(f"📦 Preparing {dataset_name}...")

            def _on_batch_end(
                epoch: int, total_epochs: int, batch_idx: int, total_batches: int, _loss: float
            ) -> None:
                sub_progress = (batch_idx + 1) / total_batches
                self.progress = 0.25 + 0.50 * (epoch + sub_progress) / total_epochs
                self._dirty = True

            def _on_epoch_end(
                epoch: int, total_epochs: int, avg_loss: float, acc: float | None
            ) -> None:
                self.current_epoch = epoch + 1
                self.current_loss = avg_loss
                self.current_acc = acc
                self.losses.append(avg_loss)
                if acc is not None:
                    self.accuracies.append(acc)
                self._dirty = True
                self._emit(
                    f"Epoch {epoch + 1}/{total_epochs} \u2014 "
                    f"Loss: {avg_loss:.4f}"
                    + (f" \u2014 Val Acc: {acc:.2%}" if acc is not None else "")
                )

                # ETA tracked per-epoch via closure
                now = _time.time()
                _elapsed = now - _last_epoch_time[0]
                _last_epoch_time[0] = now
                self._epoch_times.append(_elapsed)
                avg_epoch_t = sum(self._epoch_times) / len(self._epoch_times)
                remaining = total_epochs - (epoch + 1)
                self.eta_seconds = avg_epoch_t * remaining
                self._flush_logs()

            pipeline_result = run_distillation_pipeline(
                dataset_name=dataset_name,
                teacher_name=self.config["teacher"],
                student_type=student_name,
                compression_ratio=self.config.get("compression_ratio", 0.05),
                batch_size=self.config["batch_size"],
                data_root="./data",
                epochs=epochs,
                temperature=self.config["temperature"],
                alpha=self.config["alpha"],
                device=settings.device,
                patience=self.config.get("patience", 0),
                ckpt_dir="checkpoints",
                ckpt_every=self.config.get("ckpt_every", 5),
                resume=self.config.get("resume"),
                benchmark_target="cpu",
                export_format=None,
                teacher_fallback_random=True,
                dataset_subprocess_tracker=_subprocess_tracker,
                on_message=self._emit,
                on_epoch_end=_on_epoch_end,
                on_batch_end=_on_batch_end,
                cancel_flag=lambda: self._cancel_requested,
            )

            # If cancelled mid-training, stop here
            if self._cancel_requested:
                self._emit("\n⛔ Training cancelled.")
                self._flush_logs()
                return

            # --- Build result ---
            self.progress = 0.95
            self.result = self._build_result_dict(
                dataset_name=dataset_name,
                student_name=student_name,
                teacher_name=self.config["teacher"],
                teacher_params=pipeline_result["teacher_params"],
                student_params=pipeline_result["student_params"],
                comparison=pipeline_result["comparison"],
                losses=self.losses,
                accuracies=self.accuracies,
            )

            self.student = pipeline_result["student"]
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

        except DatasetError:
            if self._cancel_requested:
                self._emit("⛔ Cancelled.")
                self.status = "cancelled"
                self._save_error_run("cancelled")
            else:
                self._emit(
                    "❌ Dataset preparation failed. "
                    "Check your internet connection and try again, "
                    "or choose a different dataset (MNIST / FashionMNIST "
                    "are smaller and faster to download)."
                )
                self.status = "failed"
                self.error = "Dataset preparation failed"
                record_task_failure(self.id)
                self._save_error_run("failed")
            self._flush_logs()

        except PipelineError as e:
            self.status = "failed"
            self.error = str(e)
            record_task_failure(self.id)
            self._emit("\n❌ Pipeline error: " + str(e))
            self._save_error_run("failed")

        except Exception as e:
            self.status = "failed"
            self.error = str(e)
            record_task_failure(self.id)
            import traceback

            tb = traceback.format_exc()
            self._emit("\n❌ Unexpected error: " + str(e))
            self._emit(tb)
            self._save_error_run("failed")
        finally:
            self._subprocess = None
            self._flush_logs()

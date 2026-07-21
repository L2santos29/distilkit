"""Shared distillation pipeline orchestration.

Extracts the common dataset→teacher→student→distill→benchmark flow
used by both ``cli.py`` and ``webapp.py`` so it lives in one place.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src import datasets as ds
from src.benchmarks import DeviceTarget, compare_teacher_student
from src.circuit_breaker import CircuitBreaker, CircuitOpenError
from src.distiller import Distiller
from src.log_config import logger
from src.onnx_export import export_to_onnx, export_to_torchscript
from src.student import build_student
from src.teacher import load_teacher
from src.tracing import tracer

# Circuit breakers for network-dependent pipeline stages.
# Preventing cascading failures from repeated download errors.
_dataset_cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60, name="dataset")
_teacher_cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60, name="teacher")


class PipelineError(Exception):
    """Raised when a non-recoverable error occurs in the pipeline."""


class DatasetError(PipelineError):
    """Dataset preparation failed (e.g. download error, I/O error)."""


class TeacherError(PipelineError):
    """Teacher model could not be loaded."""


def run_distillation_pipeline(
    *,
    # Data
    dataset_name: str,
    teacher_name: str,
    student_type: str = "MiniCNN",
    compression_ratio: float = 0.05,
    batch_size: int = 64,
    data_root: str = "./data",
    # Training
    epochs: int = 10,
    temperature: float = 4.0,
    alpha: float = 0.7,
    device: str = "cpu",
    patience: int = 0,
    # Checkpointing
    ckpt_dir: str = "checkpoints",
    ckpt_every: int = 5,
    resume: str | None = None,
    # Post-training
    benchmark_target: DeviceTarget | None = None,
    export_format: str | None = None,
    export_output_dir: str = "checkpoints",
    # Behaviour
    teacher_fallback_random: bool = False,
    student_cache: dict[str, nn.Module] | None = None,
    dataset_subprocess_tracker: list | None = None,
    # Optimizer
    learning_rate: float = 1e-3,
    on_message: Callable[[str], None] = logger.info,
    on_epoch_end: "Callable[[int, int, float, float | None], None] | None" = None,
    on_batch_end: "Callable[[int, int, int, int, float], None] | None" = None,
    cancel_flag: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run the full distillation pipeline end-to-end.

    Returns a dict with keys ``teacher``, ``student``, ``distiller``,
    ``history``, ``comparison``, ``teacher_params``, ``student_params``,
    ``exported_path``, and ``start_epoch``.

    Raises
    ------
    DatasetError
        If dataset preparation fails (download error, I/O error).
    TeacherError
        If the teacher model cannot be loaded and fallback is disabled.
    """
    _msg = on_message
    _cancel = cancel_flag or (lambda: False)

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    with tracer.start_span("pipeline.dataset") as span:
        span.set_attribute("dataset", dataset_name)
        try:
            result = _dataset_cb.call(
                ds.get_dataset_loaders,
                dataset_name,
                batch_size,
                data_root,
                cancel_flag=_cancel,
                subprocess_tracker=dataset_subprocess_tracker,
            )
        except CircuitOpenError:
            raise DatasetError(
                "Dataset downloads are temporarily suspended due to repeated "
                "failures.  Wait a moment and try again."
            )
        if result is None:
            raise DatasetError(
                f"Failed to prepare dataset '{dataset_name}'. "
                f"Check your internet connection and try again, "
                f"or choose a different dataset (MNIST / FashionMNIST "
                f"are smaller and faster to download)."
            )
        train_loader, val_loader, num_classes, in_channels = result
        span.set_attribute("num_classes", num_classes)

    # ------------------------------------------------------------------
    # 2. Teacher
    # ------------------------------------------------------------------
    with tracer.start_span("pipeline.teacher") as span:
        span.set_attribute("teacher", teacher_name)
        try:
            teacher = _teacher_cb.call(load_teacher, teacher_name, num_classes=num_classes)
        except CircuitOpenError:
            raise TeacherError(
                "Teacher model downloads are temporarily suspended due to "
                "repeated failures.  Wait a moment and try again."
            )
        except (OSError, RuntimeError) as e:
            if teacher_fallback_random:
                _msg(
                    f"   ⚠️ Could not load pretrained weights: {e}. "
                    f"Falling back to random initialization."
                )
                teacher = load_teacher(teacher_name, num_classes=num_classes, pretrained=False)
            else:
                raise TeacherError(f"Failed to load teacher '{teacher_name}': {e}") from e

        teacher.to(device).eval()
        teacher_params = sum(p.numel() for p in teacher.parameters())
        span.set_attribute("teacher_params", teacher_params)
        _msg(f"   Teacher parameters: {teacher_params:,}")

    # ------------------------------------------------------------------
    # 3. Student
    # ------------------------------------------------------------------
    with tracer.start_span("pipeline.student") as span:
        student = build_student(
            teacher=teacher,
            student_type=student_type,
            compression_ratio=compression_ratio,
            num_classes=num_classes,
            in_channels=in_channels,
        )
        student.to(device)
        student_params = sum(p.numel() for p in student.parameters())
        span.set_attribute("student_type", student_type)
        span.set_attribute("student_params", student_params)
        span.set_attribute("compression_ratio", compression_ratio)
        _msg(f"   Student parameters: {student_params:,}")
        _msg(f"   Compression ratio: {student_params / teacher_params:.2%}")

        # Cache the student if a cache dict was provided (webapp uses this)
        if student_cache is not None:
            student_cache[teacher_name] = student

    # ------------------------------------------------------------------
    # 4. Distiller
    # ------------------------------------------------------------------
    with tracer.start_span("pipeline.distiller_init") as span:
        distiller = Distiller(
            teacher,
            student,
            temperature=temperature,
            alpha=alpha,
            device=device,
        )
        span.set_attribute("temperature", temperature)
        span.set_attribute("alpha", alpha)

    # ------------------------------------------------------------------
    # 5. Optimizer & scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # ------------------------------------------------------------------
    # 6. Resume from checkpoint
    # ------------------------------------------------------------------
    start_epoch = 0
    initial_history = None
    if resume and Path(resume).exists():
        _msg(f"📂 Resuming from {resume}...")
        try:
            ckpt = torch.load(resume, map_location=device, weights_only=False)
            student.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt.get("epoch", 0)
            initial_history = {
                "train_loss": ckpt.get("losses", []),
                "val_acc": ckpt.get("accuracies", []),
            }
            _msg(f"   Resumed at epoch {start_epoch}/{epochs}")
        except (OSError, RuntimeError, ValueError, KeyError) as e:
            _msg(f"   ⚠️ Could not load checkpoint: {e}. Starting from scratch.")
            start_epoch = 0
            initial_history = None

    # ------------------------------------------------------------------
    # 7. Train
    # ------------------------------------------------------------------
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    def _save_ckpt(epoch: int, ckpt_data: dict) -> None:
        if ckpt_every > 0 and epoch % ckpt_every == 0:
            ckpt_path = Path(ckpt_dir) / f"checkpoint_epoch_{epoch}.pt"
            torch.save(ckpt_data, ckpt_path)
            _msg(f"   💾 Checkpoint saved: {ckpt_path}")

    with tracer.start_span("pipeline.train") as span:
        span.set_attribute("epochs", epochs)
        span.set_attribute("batch_size", batch_size)
        span.set_attribute("patience", patience)
        history = distiller.train(
            train_loader,
            val_loader,
            epochs=epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            patience=patience,
            start_epoch=start_epoch,
            initial_history=initial_history,
            ckpt_callback=_save_ckpt,
            on_epoch_end=on_epoch_end,
            on_batch_end=on_batch_end,
            cancel_flag=cancel_flag,
        )
        span.set_attribute("epochs_trained", len(history.get("train_loss", [])))

    # ------------------------------------------------------------------
    # 8. Benchmark
    # ------------------------------------------------------------------
    comparison = None
    if benchmark_target and benchmark_target != "none":
        with tracer.start_span("pipeline.benchmark") as span:
            span.set_attribute("target", benchmark_target)
            comparison = compare_teacher_student(teacher, student, target=benchmark_target)
            if comparison:
                span.set_attribute("speedup", comparison.get("speedup", 0))
                span.set_attribute("compression", comparison.get("compression", 0))
            _msg(
                f"   Teacher : {comparison['teacher']['mean_ms']:.2f} ms  "
                f"({comparison['teacher']['parameters']:,} params)"
            )
            _msg(
                f"   Student : {comparison['student']['mean_ms']:.2f} ms  "
                f"({comparison['student']['parameters']:,} params)"
            )
            _msg(f"   Speedup : {comparison['speedup']}x")
            _msg(f"   Size    : {comparison['compression']:.2%} of teacher")

    # ------------------------------------------------------------------
    # 9. Export
    # ------------------------------------------------------------------
    exported_path = None
    if export_format and export_format != "none":
        with tracer.start_span("pipeline.export") as span:
            Path(export_output_dir).mkdir(parents=True, exist_ok=True)
            span.set_attribute("format", export_format)
            if export_format == "onnx":
                exported_path = export_to_onnx(student, f"{export_output_dir}/student.onnx")
            else:
                exported_path = export_to_torchscript(student, f"{export_output_dir}/student.pt")
            span.set_attribute("path", str(exported_path))
            _msg(f"   Exported to: {exported_path}")

    return {
        "teacher": teacher,
        "student": student,
        "distiller": distiller,
        "history": history,
        "comparison": comparison,
        "teacher_params": teacher_params,
        "student_params": student_params,
        "exported_path": str(exported_path) if exported_path else None,
        "start_epoch": start_epoch,
    }

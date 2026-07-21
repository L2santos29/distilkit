"""Integration tests for core DistilKit components.

Verifies that components work together correctly:
- model construction + benchmarking
- distillation training loop with synthetic data
- teacher/student comparison
- ONNX and TorchScript export
- checkpoint callbacks
"""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.benchmarks import benchmark, compare_teacher_student
from src.distiller import Distiller
from src.onnx_export import export_to_onnx, export_to_torchscript
from src.student import MiniCNN, build_student

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TinyTeacher(nn.Module):
    """Minimal CNN teacher for fast integration tests."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(8, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _synthetic_loaders(
    batch_size: int = 8,
    num_samples: int = 32,
    num_classes: int = 10,
    input_shape: tuple = (3, 32, 32),
) -> tuple[DataLoader, DataLoader]:
    """Create synthetic train/val data loaders for fast testing."""
    x = torch.randn(num_samples, *input_shape)
    y = torch.randint(0, num_classes, (num_samples,))
    ds = TensorDataset(x, y)
    train = DataLoader(ds, batch_size=batch_size, shuffle=True)
    val = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return train, val


# ---------------------------------------------------------------------------
# Benchmark integration
# ---------------------------------------------------------------------------


def test_benchmark_returns_all_keys():
    """benchmark() with a real model returns all expected metrics."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(32 * 32 * 3, 10))
    results = benchmark(model, input_shape=(1, 3, 32, 32), warmup_runs=2, benchmark_runs=5)

    assert results["target"] == "cpu"
    assert results["batch_size"] == 1
    assert results["runs"] == 5
    assert results["mean_ms"] > 0
    assert results["median_ms"] > 0
    assert results["p95_ms"] > 0
    assert results["throughput_imgs_per_sec"] > 0


def test_benchmark_with_batched_input():
    """benchmark() handles batched input correctly."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(32 * 32 * 3, 10))
    results = benchmark(model, input_shape=(8, 3, 32, 32), warmup_runs=2, benchmark_runs=5)

    assert results["batch_size"] == 8
    assert results["mean_ms"] > 0
    # Throughput should be higher with batch=8 vs batch=1
    assert isinstance(results["throughput_imgs_per_sec"], float)


# ---------------------------------------------------------------------------
# Teacher / Student comparison
# ---------------------------------------------------------------------------


def test_compare_teacher_student_structure():
    """compare_teacher_student() returns expected nested structure."""
    teacher = TinyTeacher(num_classes=10)
    student = MiniCNN(in_channels=3, num_classes=10, width=0.5)

    comparison = compare_teacher_student(teacher, student, input_shape=(1, 3, 32, 32), target="cpu")

    # Top-level keys
    assert "teacher" in comparison
    assert "student" in comparison
    assert "speedup" in comparison
    assert "compression" in comparison

    # Both models should have benchmark results
    for key in ("teacher", "student"):
        assert "parameters" in comparison[key]
        assert comparison[key]["parameters"] > 0
        assert "mean_ms" in comparison[key]
        assert comparison[key]["mean_ms"] > 0

    # Both models should have positive parameter counts
    assert comparison["teacher"]["parameters"] > 0
    assert comparison["student"]["parameters"] > 0
    assert comparison["compression"] > 0
    assert comparison["speedup"] > 0


# ---------------------------------------------------------------------------
# Distillation training with custom callbacks
# ---------------------------------------------------------------------------


def test_distiller_trains_with_synthetic_data():
    """Distiller.train() completes 2 epochs with synthetic data and returns history."""
    teacher = TinyTeacher(num_classes=10)
    student = MiniCNN(in_channels=3, num_classes=10, width=0.5)
    train_loader, val_loader = _synthetic_loaders()

    distiller = Distiller(teacher, student, temperature=4.0, alpha=0.7)
    history = distiller.train(train_loader, val_loader, epochs=2)

    assert "train_loss" in history
    assert "val_acc" in history
    assert len(history["train_loss"]) == 2
    assert len(history["val_acc"]) == 2
    assert all(loss > 0 for loss in history["train_loss"])
    assert all(0 <= acc <= 1 for acc in history["val_acc"])


def test_distiller_custom_optimizer_and_scheduler():
    """Distiller.train() accepts pre-created optimizer and scheduler."""
    teacher = TinyTeacher(num_classes=10)
    student = MiniCNN(in_channels=3, num_classes=10, width=0.5)
    train_loader, val_loader = _synthetic_loaders()

    distiller = Distiller(teacher, student, temperature=4.0, alpha=0.7)
    optimizer = torch.optim.SGD(student.parameters(), lr=1e-2, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    history = distiller.train(
        train_loader, val_loader, epochs=2, optimizer=optimizer, scheduler=scheduler
    )

    assert len(history["train_loss"]) == 2
    # Scheduler step should have been called
    assert optimizer.param_groups[0]["lr"] < 1e-2  # StepLR decayed


def test_distiller_on_epoch_end_callback():
    """Distiller.train() fires on_epoch_end after each epoch."""
    teacher = TinyTeacher(num_classes=10)
    student = MiniCNN(in_channels=3, num_classes=10, width=0.5)
    train_loader, val_loader = _synthetic_loaders()

    distiller = Distiller(teacher, student, temperature=4.0, alpha=0.7)
    epochs_run = []

    def track_epoch(epoch: int, total: int, _loss: float, acc: float | None) -> None:
        epochs_run.append((epoch, total, acc))

    history = distiller.train(train_loader, val_loader, epochs=2, on_epoch_end=track_epoch)

    assert len(epochs_run) == 2
    assert epochs_run[0] == (0, 2, history["val_acc"][0])
    assert epochs_run[1] == (1, 2, history["val_acc"][1])


def test_distiller_on_batch_end_callback():
    """Distiller.train() fires on_batch_end after each batch."""
    teacher = TinyTeacher(num_classes=10)
    student = MiniCNN(in_channels=3, num_classes=10, width=0.5)
    train_loader, val_loader = _synthetic_loaders(batch_size=8, num_samples=32)
    expected_batches = len(train_loader)  # 32 / 8 = 4

    distiller = Distiller(teacher, student, temperature=4.0, alpha=0.7)
    batch_count = 0

    def track_batch(_epoch: int, _total: int, _batch_idx: int, _total_b: int, _loss: float) -> None:
        nonlocal batch_count
        batch_count += 1

    distiller.train(train_loader, val_loader, epochs=2, on_batch_end=track_batch)

    assert batch_count == expected_batches * 2  # 2 epochs × 4 batches


def test_distiller_ckpt_callback():
    """Distiller.train() fires ckpt_callback every epoch."""
    teacher = TinyTeacher(num_classes=10)
    student = MiniCNN(in_channels=3, num_classes=10, width=0.5)
    train_loader, val_loader = _synthetic_loaders()

    distiller = Distiller(teacher, student, temperature=4.0, alpha=0.7)
    checkpoints = []

    def track_ckpt(epoch: int, ckpt: dict) -> None:
        checkpoints.append((epoch, set(ckpt.keys())))

    distiller.train(train_loader, val_loader, epochs=2, ckpt_callback=track_ckpt)

    assert len(checkpoints) == 2
    for epoch, keys in checkpoints:
        assert "model" in keys
        assert "optimizer" in keys
        assert "losses" in keys
        assert "accuracies" in keys


def test_distiller_cancel_flag():
    """Distiller.train() stops early when cancel_flag returns True."""
    teacher = TinyTeacher(num_classes=10)
    student = MiniCNN(in_channels=3, num_classes=10, width=0.5)
    train_loader, val_loader = _synthetic_loaders()

    distiller = Distiller(teacher, student, temperature=4.0, alpha=0.7)
    call_count = 0

    def cancel_after_epoch() -> bool:
        nonlocal call_count
        call_count += 1
        return call_count >= 1  # Cancel after first epoch

    history = distiller.train(train_loader, val_loader, epochs=5, cancel_flag=cancel_after_epoch)

    # Should have stopped after 1 epoch (or at least fewer than 5)
    assert len(history["train_loss"]) < 5


def test_distiller_patience_early_stopping():
    """Distiller.train() stops early when validation accuracy stops improving."""
    teacher = TinyTeacher(num_classes=10)
    student = MiniCNN(in_channels=3, num_classes=10, width=0.5)
    train_loader, val_loader = _synthetic_loaders()

    distiller = Distiller(teacher, student, temperature=4.0, alpha=0.7)

    # patience=1 means stop after 1 epoch without improvement.
    # With random synthetic data, accuracy won't improve significantly.
    history = distiller.train(train_loader, val_loader, epochs=10, patience=1)

    # Should stop well before 10 epochs with patience=1 on random data
    assert len(history["train_loss"]) < 10


def test_distiller_resume_from_history():
    """Distiller.train() resumes from initial_history correctly."""
    teacher = TinyTeacher(num_classes=10)
    student = MiniCNN(in_channels=3, num_classes=10, width=0.5)
    train_loader, val_loader = _synthetic_loaders()

    distiller = Distiller(teacher, student, temperature=4.0, alpha=0.7)

    # Run 2 epochs
    history_phase1 = distiller.train(train_loader, val_loader, epochs=2)

    # Resume for 2 more epochs with prior history
    history_phase2 = distiller.train(
        train_loader,
        val_loader,
        epochs=4,
        start_epoch=2,
        initial_history=history_phase1,
    )

    assert len(history_phase2["train_loss"]) == 4  # 2 original + 2 new
    assert history_phase2["train_loss"][:2] == history_phase1["train_loss"]


# ---------------------------------------------------------------------------
# Export integration
# ---------------------------------------------------------------------------


def test_export_to_onnx_creates_file():
    """export_to_onnx() creates a valid .onnx file."""
    model = MiniCNN(in_channels=3, num_classes=10, width=0.5)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = export_to_onnx(model, Path(tmpdir) / "test.onnx", opset_version=17)

        assert path.exists()
        assert path.suffix == ".onnx"
        assert path.stat().st_size > 0

        # Verify it's a valid ONNX file
        import onnx

        onnx_model = onnx.load(str(path))
        onnx.checker.check_model(onnx_model)


def test_export_to_torchscript_creates_file():
    """export_to_torchscript() creates a valid .pt file."""
    model = MiniCNN(in_channels=3, num_classes=10, width=0.5)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = export_to_torchscript(model, Path(tmpdir) / "test.pt")

        assert path.exists()
        assert path.suffix == ".pt"
        assert path.stat().st_size > 0

        # Verify it loads back
        loaded = torch.jit.load(str(path))
        dummy = torch.randn(1, 3, 32, 32)
        out = loaded(dummy)
        assert out.shape == (1, 10)


def test_export_to_onnx_with_dynamic_batch():
    """export_to_onnx() with dynamic_batch=True allows variable batch sizes."""
    model = MiniCNN(in_channels=3, num_classes=10, width=0.5)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = export_to_onnx(
            model, Path(tmpdir) / "dynamic.onnx", dynamic_batch=True, opset_version=17
        )

        assert path.exists()

        # Verify with ONNX Runtime using different batch sizes
        import onnxruntime as ort

        session = ort.InferenceSession(str(path))
        for batch_size in (1, 4, 8):
            dummy = torch.randn(batch_size, 3, 32, 32).numpy()
            output = session.run(None, {"input": dummy})
            assert output[0].shape == (batch_size, 10)


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------


def test_full_pipeline_synthetic():
    """End-to-end: build → distill → benchmark → export with synthetic data."""
    teacher = TinyTeacher(num_classes=10)
    student = build_student(
        teacher=teacher,
        student_type="MiniCNN",
        compression_ratio=0.5,
        num_classes=10,
        in_channels=3,
    )
    train_loader, val_loader = _synthetic_loaders(batch_size=8, num_samples=32)

    distiller = Distiller(teacher, student, temperature=4.0, alpha=0.7)
    history = distiller.train(train_loader, val_loader, epochs=2)

    assert len(history["train_loss"]) == 2
    assert history["val_acc"][-1] >= 0  # validation ran

    # Benchmark
    comparison = compare_teacher_student(teacher, student, target="cpu")
    assert comparison["speedup"] > 0
    assert comparison["student"]["mean_ms"] > 0

    # Export to ONNX
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = export_to_onnx(student, Path(tmpdir) / "student.onnx")
        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Edge cases — boundary conditions, empty inputs, extreme values
# ---------------------------------------------------------------------------


class TinyEdgeTeacher(nn.Module):
    """Minimal teacher with 2 classes for edge-case tests."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Distillation edge cases ─────────────────────────────────────


def test_distiller_alpha_zero_is_ce_only():
    """alpha=0 means purely hard-label (cross-entropy) loss."""
    teacher = TinyEdgeTeacher(num_classes=2)
    student = MiniCNN(in_channels=3, num_classes=2, width=0.25)
    train_loader, val_loader = _synthetic_loaders(batch_size=4, num_samples=8, num_classes=2)

    distiller = Distiller(teacher, student, temperature=4.0, alpha=0.0)
    history = distiller.train(train_loader, val_loader, epochs=1)

    assert len(history["train_loss"]) == 1
    assert history["train_loss"][0] > 0


def test_distiller_alpha_one_is_pure_distillation():
    """alpha=1.0 means purely distillation (ignores hard labels)."""
    teacher = TinyEdgeTeacher(num_classes=2)
    student = MiniCNN(in_channels=3, num_classes=2, width=0.25)
    train_loader, val_loader = _synthetic_loaders(batch_size=4, num_samples=8, num_classes=2)

    distiller = Distiller(teacher, student, temperature=4.0, alpha=1.0)
    history = distiller.train(train_loader, val_loader, epochs=1)

    assert len(history["train_loss"]) == 1


def test_distiller_temperature_one():
    """temperature=1.0 is a valid edge (no softening)."""
    teacher = TinyEdgeTeacher(num_classes=2)
    student = MiniCNN(in_channels=3, num_classes=2, width=0.25)
    train_loader, val_loader = _synthetic_loaders(batch_size=4, num_samples=8, num_classes=2)

    distiller = Distiller(teacher, student, temperature=1.0, alpha=0.7)
    history = distiller.train(train_loader, val_loader, epochs=1)

    assert len(history["train_loss"]) == 1


def test_distiller_single_batch():
    """Training with a single batch per epoch does not crash."""
    teacher = TinyEdgeTeacher(num_classes=2)
    student = MiniCNN(in_channels=3, num_classes=2, width=0.25)
    # 4 samples with batch_size=4 → exactly 1 batch
    train_loader, val_loader = _synthetic_loaders(batch_size=4, num_samples=4, num_classes=2)

    distiller = Distiller(teacher, student, temperature=4.0, alpha=0.7)
    history = distiller.train(train_loader, val_loader, epochs=2)

    assert len(history["train_loss"]) == 2


def test_distiller_minimum_batch_size():
    """batch_size=1 is the minimum and should not crash."""
    teacher = TinyEdgeTeacher(num_classes=2)
    student = MiniCNN(in_channels=3, num_classes=2, width=0.25)
    train_loader, val_loader = _synthetic_loaders(batch_size=1, num_samples=4, num_classes=2)

    distiller = Distiller(teacher, student, temperature=4.0, alpha=0.7)
    history = distiller.train(train_loader, val_loader, epochs=1)

    assert len(history["train_loss"]) == 1


def test_distiller_temperature_high():
    """High temperature (100.0) should not cause numerical issues."""
    teacher = TinyEdgeTeacher(num_classes=2)
    student = MiniCNN(in_channels=3, num_classes=2, width=0.25)
    train_loader, val_loader = _synthetic_loaders(batch_size=4, num_samples=8, num_classes=2)

    distiller = Distiller(teacher, student, temperature=100.0, alpha=0.7)
    history = distiller.train(train_loader, val_loader, epochs=1)

    assert len(history["train_loss"]) == 1
    assert torch.isfinite(torch.tensor(history["train_loss"][0]))


# ── Benchmark edge cases ────────────────────────────────────────


def test_benchmark_single_run():
    """benchmark with benchmark_runs=1 works (minimum valid)."""
    model = nn.Linear(10, 2)
    results = benchmark(model, input_shape=(1, 10), warmup_runs=0, benchmark_runs=1)

    assert results["runs"] == 1
    assert results["mean_ms"] > 0


def test_benchmark_no_warmup():
    """benchmark with warmup_runs=0 does not crash."""
    model = nn.Linear(10, 2)
    results = benchmark(model, input_shape=(1, 10), warmup_runs=0, benchmark_runs=5)

    assert results["runs"] == 5


def test_benchmark_tiny_input():
    """benchmark handles smallest reasonable input (1x1)."""
    model = nn.Linear(1, 1)
    results = benchmark(model, input_shape=(1, 1), warmup_runs=2, benchmark_runs=5)

    assert results["mean_ms"] > 0
    assert results["throughput_imgs_per_sec"] > 0


def test_benchmark_large_batch():
    """benchmark with batch=64 does not crash."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(32 * 32 * 3, 10))
    results = benchmark(model, input_shape=(64, 3, 32, 32), warmup_runs=2, benchmark_runs=5)

    assert results["batch_size"] == 64


# ── Export edge cases ───────────────────────────────────────────


def test_export_to_onnx_single_class():
    """export_to_onnx with a 1-class output does not crash."""
    model = MiniCNN(in_channels=3, num_classes=1, width=0.25)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = export_to_onnx(model, Path(tmpdir) / "single.onnx", opset_version=17)
        assert path.exists()
        assert path.stat().st_size > 0


def test_export_to_onnx_batched_64():
    """export_to_onnx with batch=64 produces correct output shape."""
    model = MiniCNN(in_channels=3, num_classes=10, width=0.25)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = export_to_onnx(
            model,
            Path(tmpdir) / "batch64.onnx",
            input_shape=(64, 3, 32, 32),
            opset_version=17,
        )
        assert path.exists()

        # Verify with ONNX Runtime
        import onnxruntime as ort

        session = ort.InferenceSession(str(path))
        dummy = torch.randn(64, 3, 32, 32).numpy()
        output = session.run(None, {"input": dummy})
        assert output[0].shape == (64, 10)


def test_export_to_torchscript_single_channel():
    """export_to_torchscript with a 1-channel model does not crash."""
    model = MiniCNN(in_channels=1, num_classes=1, width=0.25)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = export_to_torchscript(model, Path(tmpdir) / "single.pt", input_shape=(1, 1, 32, 32))
        assert path.exists()

        loaded = torch.jit.load(str(path))
        dummy = torch.randn(1, 1, 32, 32)
        out = loaded(dummy)
        assert out.shape == (1, 1)


# ── Student building edge cases ─────────────────────────────────


def test_build_student_tiny_compression():
    """build_student with compression_ratio near 0 (minimum) does not crash."""
    teacher = TinyEdgeTeacher(num_classes=10)
    student = build_student(
        teacher=teacher,
        student_type="MiniCNN",
        compression_ratio=0.001,
        num_classes=10,
        in_channels=3,
    )
    assert student is not None
    # Width should have been clamped to 0.125 minimum
    assert sum(p.numel() for p in student.parameters()) > 0


def test_build_student_max_compression():
    """build_student with compression_ratio >= 1.0 is clamped correctly."""
    teacher = TinyEdgeTeacher(num_classes=10)
    student = build_student(
        teacher=teacher,
        student_type="MiniCNN",
        compression_ratio=2.0,
        num_classes=10,
        in_channels=3,
    )
    assert student is not None
    assert sum(p.numel() for p in student.parameters()) > 0


def test_build_student_no_teacher():
    """build_student without a teacher falls back to width=1.0."""
    student = build_student(
        teacher=None,
        student_type="MiniCNN",
        compression_ratio=0.0,
        num_classes=10,
        in_channels=3,
    )
    assert student is not None
    base = MiniCNN(in_channels=3, num_classes=10, width=1.0)
    assert sum(p.numel() for p in student.parameters()) == sum(p.numel() for p in base.parameters())


def test_build_student_single_channel():
    """build_student with in_channels=1 (grayscale) works."""
    teacher = TinyEdgeTeacher(num_classes=10)
    student = build_student(
        teacher=teacher,
        student_type="MiniCNN",
        compression_ratio=0.5,
        num_classes=10,
        in_channels=1,
    )
    assert student is not None
    dummy = torch.randn(1, 1, 32, 32)
    out = student(dummy)
    assert out.shape == (1, 10)


def test_build_student_miniresnet():
    """build_student with MiniResNet architecture works."""
    teacher = TinyEdgeTeacher(num_classes=10)
    student = build_student(
        teacher=teacher,
        student_type="MiniResNet",
        compression_ratio=0.5,
        num_classes=10,
        in_channels=3,
    )
    assert student is not None
    dummy = torch.randn(1, 3, 32, 32)
    out = student(dummy)
    assert out.shape == (1, 10)


# ── Distillation error-handling edge cases ──────────────────────


def test_distiller_cancel_at_start():
    """Cancelling before any epoch completes gracefully."""
    teacher = TinyEdgeTeacher(num_classes=2)
    student = MiniCNN(in_channels=3, num_classes=2, width=0.25)
    train_loader, val_loader = _synthetic_loaders(batch_size=4, num_samples=8, num_classes=2)

    distiller = Distiller(teacher, student, temperature=4.0, alpha=0.7)
    history = distiller.train(train_loader, val_loader, epochs=5, cancel_flag=lambda: True)

    # Should have 0 epochs — cancelled immediately
    assert len(history["train_loss"]) == 0


def test_distiller_patience_zero():
    """patience=0 runs all epochs without early stopping."""
    teacher = TinyEdgeTeacher(num_classes=2)
    student = MiniCNN(in_channels=3, num_classes=2, width=0.25)
    train_loader, val_loader = _synthetic_loaders(batch_size=4, num_samples=8, num_classes=2)

    distiller = Distiller(teacher, student, temperature=4.0, alpha=0.7)
    history = distiller.train(train_loader, val_loader, epochs=3, patience=0)

    assert len(history["train_loss"]) == 3


def test_distiller_no_val_loader():
    """Distiller.train() without val_loader still returns history with no acc."""
    teacher = TinyEdgeTeacher(num_classes=2)
    student = MiniCNN(in_channels=3, num_classes=2, width=0.25)
    train_loader, _ = _synthetic_loaders(batch_size=4, num_samples=8, num_classes=2)

    distiller = Distiller(teacher, student, temperature=4.0, alpha=0.7)
    history = distiller.train(train_loader, val_loader=None, epochs=2)

    assert len(history["train_loss"]) == 2
    assert len(history["val_acc"]) == 0  # no validation ran


# ── Unknown model name edge cases ───────────────────────────────


def test_teacher_unsupported_name():
    """load_teacher with an unsupported name raises ValueError."""
    import pytest

    from src.teacher import load_teacher

    with pytest.raises(ValueError, match="Unknown model"):
        load_teacher("nonexistent_model")


def test_student_unsupported_name():
    """build_student with an unsupported student_type raises ValueError."""
    import pytest

    with pytest.raises(ValueError, match="Unknown student"):
        build_student(student_type="NonExistent", compression_ratio=0)

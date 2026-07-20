#!/usr/bin/env python3
"""Profile the critical paths in the DistilKit pipeline.

Measures wall-clock time for:
  1. Model build (teacher, student)
  2. Distillation training (per-epoch breakdown)
  3. Model export (ONNX, TorchScript)
  4. Model inference (teacher vs student)

Usage:
    python scripts/measure_perf.py              # CPU only (default)
    python scripts/measure_perf.py --device cuda  # GPU if available

Results are printed to stdout and written to ``PROFILING.md``.
"""

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = REPO_ROOT / "PROFILING.md"

# Ensure the project root is on sys.path so ``from src.…`` imports work
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NUM_CLASSES = 10
IN_CHANNELS = 3
INPUT_SIZE = 32
BATCH_SIZE = 64
EPOCHS = 2
SYNTHETIC_BATCHES = 10


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def _synthetic_loaders(batch_size: int, num_batches: int = SYNTHETIC_BATCHES):
    """Return tiny (train, val) loaders with random images."""
    from torch.utils.data import DataLoader, TensorDataset

    n = num_batches * batch_size
    x = torch.randn(n, IN_CHANNELS, INPUT_SIZE, INPUT_SIZE)
    y = torch.randint(0, NUM_CLASSES, (n,))
    ds = TensorDataset(x, y)
    return (
        DataLoader(ds, batch_size=batch_size, shuffle=False),
        DataLoader(ds, batch_size=batch_size, shuffle=False),
    )


# ---------------------------------------------------------------------------
# Simple profiler
# ---------------------------------------------------------------------------


@dataclass
class ProfileResult:
    """A single measured operation."""

    label: str
    elapsed_sec: float
    note: str = ""


class Profiler:
    """Collects timing measurements for pipeline phases."""

    def __init__(self) -> None:
        self.results: list[ProfileResult] = []

    def measure(self, label: str, note: str = "") -> "_MeasureContext":
        return self._MeasureContext(self, label, note)

    class _MeasureContext:
        def __init__(self, profiler: "Profiler", label: str, note: str):
            self.profiler = profiler
            self.label = label
            self.note = note
            self.start: float = 0.0

        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args: Any) -> None:
            elapsed = time.perf_counter() - self.start
            self.profiler.results.append(ProfileResult(self.label, elapsed, self.note))

    def table(self, heading: str) -> str:
        """Render collected results as a markdown table and clear the list."""
        lines = [
            f"\n## {heading}",
            "",
            "| Operation | Time (s) | Note |",
            "|----------|----------|------|",
        ]
        for r in self.results:
            note = r.note or "—"
            lines.append(f"| {r.label} | {r.elapsed_sec:.3f} | {note} |")
        lines.append("")
        out = "\n".join(lines)
        self.results.clear()
        return out


# ---------------------------------------------------------------------------
# Profiled pipeline
# ---------------------------------------------------------------------------


def profile_pipeline(device: str = "cpu") -> str:
    """Run profiling and return the markdown report section."""
    p = Profiler()
    sections: list[str] = []

    # ---- 1. Model build ----
    from src.teacher import load_teacher
    from src.student import build_student

    with p.measure("Build teacher (resnet18)"):
        teacher = load_teacher("resnet18", num_classes=NUM_CLASSES)
    teacher.to(device).eval()

    with p.measure("Build student (MiniCNN, ratio=0.05)"):
        student = build_student(
            teacher=teacher,
            student_type="MiniCNN",
            compression_ratio=0.05,
            num_classes=NUM_CLASSES,
            in_channels=IN_CHANNELS,
        )
    student.to(device)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    sections.append(p.table("1. Model Build"))

    # ---- 2. Distillation training ----
    from src.distiller import Distiller

    train_loader, val_loader = _synthetic_loaders(BATCH_SIZE)

    distiller = Distiller(teacher, student, temperature=4.0, alpha=0.7, device=device)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    with p.measure(f"Training ({EPOCHS} epochs, batch={BATCH_SIZE})", note="synthetic data"):
        history = distiller.train(
            train_loader, val_loader, epochs=EPOCHS, optimizer=optimizer, scheduler=scheduler
        )

    final_loss = history["train_loss"][-1] if history["train_loss"] else 0.0
    final_acc = history["val_acc"][-1] if history["val_acc"] else None

    sections.append(p.table("2. Distillation Training"))

    # Per-epoch detail table
    sections.append(
        "| Detail | Value |\n"
        "|--------|-------|\n"
        f"| Epochs | {EPOCHS} |\n"
        f"| Batch size | {BATCH_SIZE} |\n"
        f"| Batches/epoch | {len(train_loader)} |\n"
        f"| Final loss | {final_loss:.4f} |\n"
        f"| Final accuracy | {final_acc:.2%} |\n"
        f"| Teacher params | {teacher_params:,} |\n"
        f"| Student params | {student_params:,} |\n"
        f"| Compression ratio | {student_params/teacher_params:.2%} |\n\n"
    )

    # ---- 3. Inference benchmark (uses existing benchmark module) ----
    from src.benchmarks import benchmark

    teacher.eval()
    student.eval()

    with p.measure("Inference — teacher (100 runs)"):
        teacher_bench = benchmark(teacher, target=device, benchmark_runs=100)

    with p.measure("Inference — student (100 runs)"):
        student_bench = benchmark(student, target=device, benchmark_runs=100)

    sections.append(p.table("3. Inference Benchmark"))
    sections.append(
        "| Metric | Teacher | Student | Speedup |\n"
        "|--------|---------|---------|--------|\n"
        f"| Mean (ms) | {teacher_bench['mean_ms']:.3f} | {student_bench['mean_ms']:.3f} | "
        f"{teacher_bench['mean_ms']/student_bench['mean_ms']:.2f}x |\n"
        f"| Throughput (img/s) | {teacher_bench['throughput_imgs_per_sec']:.1f} | "
        f"{student_bench['throughput_imgs_per_sec']:.1f} | "
        f"{student_bench['throughput_imgs_per_sec']/teacher_bench['throughput_imgs_per_sec']:.2f}x |\n\n"
    )

    # ---- 4. Model export ----
    from src.onnx_export import export_to_onnx, export_to_torchscript

    with TemporaryDirectory() as tmp:
        onnx_path = str(Path(tmp) / "model.onnx")
        ts_path = str(Path(tmp) / "model.pt")

        with p.measure("Export to ONNX"):
            export_to_onnx(student, onnx_path)

        with p.measure("Export to TorchScript"):
            export_to_torchscript(student, ts_path)

        onnx_size = Path(onnx_path).stat().st_size
        ts_size = Path(ts_path).stat().st_size

    sections.append(p.table("4. Model Export"))
    sections.append(
        "| Format | Size (MB) |\n"
        "|--------|----------|\n"
        f"| ONNX | {onnx_size / 1e6:.2f} |\n"
        f"| TorchScript | {ts_size / 1e6:.2f} |\n\n"
    )

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile DistilKit critical paths")
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda)")
    args = parser.parse_args()

    print("=" * 60)
    print("  ⚡ DistilKit Profiler")
    print(f"  Device : {args.device}")
    print("=" * 60)

    report = profile_pipeline(device=args.device)

    heading = (
        "# Profiling Report\n\n"
        f"Auto-generated by ``scripts/measure_perf.py`` on {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Device: ``{args.device}``\n\n"
        "> **Note:** Numbers are hardware-dependent. Run ``python scripts/measure_perf.py``\n"
        "> on your target hardware to get accurate baselines for your setup.\n"
    )

    full = heading + report
    RESULTS_FILE.write_text(full)

    print(report)
    print("=" * 60)
    print(f"  Report written to {RESULTS_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()

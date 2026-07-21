"""Model benchmarking for latency and throughput.

Measures inference time and throughput across CPU, GPU, and (optionally) NPU
targets. Useful for comparing teacher vs. student and quantifying compression gains.
"""

import time
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

DeviceTarget = Literal["cpu", "cuda", "npu"]

# Seconds → milliseconds conversion factor
MS_PER_SEC: int = 1000


def benchmark(
    model: nn.Module,
    input_shape: tuple[int, int, int, int] = (1, 3, 32, 32),
    target: DeviceTarget = "cpu",
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
) -> dict:
    """Benchmark model inference latency and throughput.

    Args:
        model: PyTorch model (already on the correct device).
        input_shape: Shape of input tensor (batch, channels, height, width).
        target: Device to benchmark on.
        warmup_runs: Initial inference calls to warm up GPU/CPU cache.
        benchmark_runs: Inference calls for timing measurement.

    Returns:
        Dict with mean_ms, median_ms, p95_ms, std_ms, throughput_imgs_per_sec.
    """
    device = _resolve_device(target)
    model = model.to(device).eval()

    dummy_input = torch.randn(*input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    # Benchmark
    timings = []
    with torch.no_grad():
        for _ in range(benchmark_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            if target == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append((end - start) * MS_PER_SEC)

    timings = np.array(timings)
    batch_size = input_shape[0]

    return {
        "target": target,
        "batch_size": batch_size,
        "input_shape": input_shape,
        "runs": benchmark_runs,
        "mean_ms": round(float(np.mean(timings)), 3),
        "median_ms": round(float(np.median(timings)), 3),
        "p95_ms": round(float(np.percentile(timings, 95)), 3),
        "std_ms": round(float(np.std(timings)), 3),
        "throughput_imgs_per_sec": round(batch_size / (np.mean(timings) / MS_PER_SEC), 1),
    }


def compare_teacher_student(
    teacher: nn.Module,
    student: nn.Module,
    input_shape: tuple[int, int, int, int] = (1, 3, 32, 32),
    target: DeviceTarget = "cpu",
) -> dict:
    """Benchmark teacher and student models side-by-side.

    Args:
        teacher: Teacher model.
        student: Student model.
        input_shape: Input tensor shape.
        target: Device target.

    Returns:
        Dict with teacher and student benchmark results plus comparison.
    """
    teacher_results = benchmark(teacher, input_shape, target)
    student_results = benchmark(student, input_shape, target)

    # Parameter counts
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())

    return {
        "teacher": {
            "parameters": teacher_params,
            **teacher_results,
        },
        "student": {
            "parameters": student_params,
            **student_results,
        },
        "speedup": round(teacher_results["mean_ms"] / student_results["mean_ms"], 2),
        "compression": round(student_params / teacher_params, 4),
    }


def _resolve_device(target: DeviceTarget) -> torch.device:
    # Give a clear error if CUDA was requested but isn't available,
    # rather than letting torch.device("cuda") fail with a cryptic message.
    if target == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(target)

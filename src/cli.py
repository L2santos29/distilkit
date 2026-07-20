#!/usr/bin/env python3
"""DistilKit CLI — Run knowledge distillation from the command line."""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.teacher import load_teacher
from src.student import build_student
from src.distiller import Distiller
from src.benchmarks import compare_teacher_student, benchmark
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

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _download_cifar10(data_root: str = "./data"):
    """Download CIFAR-10 using wget/curl if available (faster than urllib)."""
    import subprocess

    cifar_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    cifar_tgz = os.path.join(data_root, "cifar-10-python.tar.gz")
    os.makedirs(data_root, exist_ok=True)

    if os.path.exists(cifar_tgz):
        return

    wget_ok = subprocess.run(
        ["which", "wget"], capture_output=True
    ).returncode == 0
    curl_ok = subprocess.run(
        ["which", "curl"], capture_output=True
    ).returncode == 0

    if wget_ok:
        print("⬇️ Downloading CIFAR-10 with wget (fast)...")
        subprocess.run(["wget", "-O", cifar_tgz, cifar_url], check=True)
    elif curl_ok:
        print("⬇️ Downloading CIFAR-10 with curl...")
        subprocess.run(["curl", "-Lo", cifar_tgz, cifar_url], check=True)
    else:
        print("⬇️ Downloading CIFAR-10 (urllib — speed may vary)...")


def _get_cifar10_loaders(batch_size: int, data_root: str = "./data"):
    """Return (train_loader, val_loader) for CIFAR-10."""
    _download_cifar10(data_root)

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
        root=data_root, train=True, download=True, transform=transform_train
    )
    val_set = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_val
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="distilkit",
        description="⚡ DistilKit — Knowledge Distillation Framework\n"
                    "Compress large teacher models into fast student models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  distilkit train --teacher resnet50 --epochs 10\n"
            "  distilkit train --teacher mobilenet_v2 --export onnx --benchmark cpu\n"
            "  distilkit gui\n"
            "  distilkit benchmark --model student.onnx --target cpu\n"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- train ----
    train_parser = subparsers.add_parser(
        "train", help="Run knowledge distillation (teacher → student)"
    )
    train_parser.add_argument(
        "--teacher", default="resnet18", choices=TEACHER_CHOICES,
        help="Teacher model architecture (default: resnet18)",
    )
    train_parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs (default: 10)",
    )
    train_parser.add_argument(
        "--temperature", type=float, default=4.0,
        help="Distillation temperature — higher = softer targets (default: 4.0)",
    )
    train_parser.add_argument(
        "--alpha", type=float, default=0.7,
        help="Distillation loss weight (0-1). Higher = more teacher influence (default: 0.7)",
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Training batch size (default: 64)",
    )
    train_parser.add_argument(
        "--export", choices=["onnx", "torchscript", "none"], default="none",
        help="Export trained student model (default: none)",
    )
    train_parser.add_argument(
        "--benchmark", choices=["cpu", "cuda", "none"], default="cpu",
        help="Benchmark teacher vs student on target device (default: cpu)",
    )
    train_parser.add_argument(
        "--output-dir", default="checkpoints",
        help="Directory for exported models (default: checkpoints)",
    )
    train_parser.add_argument(
        "--data-dir", default="./data",
        help="Dataset cache directory (default: ./data)",
    )

    # ---- benchmark ----
    bench_parser = subparsers.add_parser(
        "benchmark", help="Benchmark a trained model"
    )
    bench_parser.add_argument(
        "--model", required=True,
        help="Path to model file (.onnx, .pt, or .pth)",
    )
    bench_parser.add_argument(
        "--target", choices=["cpu", "cuda", "npu"], default="cpu",
        help="Target device (default: cpu)",
    )
    bench_parser.add_argument(
        "--runs", type=int, default=100,
        help="Number of inference runs (default: 100)",
    )

    # ---- export ----
    export_parser = subparsers.add_parser(
        "export", help="Export a trained model to ONNX or TorchScript"
    )
    export_parser.add_argument(
        "--model", required=True,
        help="Path to PyTorch model (.pth)",
    )
    export_parser.add_argument(
        "--format", choices=["onnx", "torchscript"], default="onnx",
        help="Export format (default: onnx)",
    )
    export_parser.add_argument(
        "--output", default=None,
        help="Output path (default: checkpoints/student.<format>)",
    )

    # ---- gui ----
    subparsers.add_parser(
        "gui", help="Launch the web-based GUI (FastAPI + Tailwind CSS)"
    )

    return parser


def cmd_train(args: argparse.Namespace):
    """Run the full distillation pipeline."""
    print("=" * 60)
    print("  ⚡ DistilKit — Knowledge Distillation")
    print("=" * 60)
    print(f"  Teacher     : {args.teacher}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Temperature : {args.temperature}")
    print(f"  Alpha       : {args.alpha}")
    print(f"  Batch size  : {args.batch_size}")
    print("=" * 60)
    print()

    # 1. Data
    print("📦 Loading CIFAR-10...")
    train_loader, val_loader = _get_cifar10_loaders(args.batch_size, args.data_dir)

    # 2. Teacher
    print(f"🧠 Loading teacher ({args.teacher})...")
    teacher = load_teacher(args.teacher, num_classes=10)
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"   Parameters: {teacher_params:,}")

    # 3. Student
    print("🔧 Building student...")
    student = build_student(teacher, compression_ratio=0.25, num_classes=10)
    student_params = sum(p.numel() for p in student.parameters())
    print(f"   Parameters: {student_params:,}")
    print(f"   Compression: {student_params / teacher_params:.2%} of teacher size")
    print()

    # 4. Distill
    print(f"🔄 Training ({args.epochs} epochs)...")
    distiller = Distiller(
        teacher, student,
        temperature=args.temperature,
        alpha=args.alpha,
    )
    history = distiller.train(train_loader, val_loader, epochs=args.epochs)
    print()

    # 5. Benchmark
    if args.benchmark != "none":
        print(f"📊 Benchmarking on {args.benchmark}...")
        comparison = compare_teacher_student(teacher, student, target=args.benchmark)
        print(f"   Teacher : {comparison['teacher']['mean_ms']:.2f} ms  "
              f"({comparison['teacher']['parameters']:,} params)")
        print(f"   Student : {comparison['student']['mean_ms']:.2f} ms  "
              f"({comparison['student']['parameters']:,} params)")
        print(f"   Speedup : {comparison['speedup']}x")
        print(f"   Size    : {comparison['compression']:.2%} of teacher")
        print()

    # 6. Export
    if args.export != "none":
        os.makedirs(args.output_dir, exist_ok=True)
        if args.export == "onnx":
            path = export_to_onnx(student, f"{args.output_dir}/student.onnx")
        else:
            path = export_to_torchscript(student, f"{args.output_dir}/student.pt")
        print(f"   Exported to: {path}")

    print()
    print("✅ Training complete!")
    return student


def cmd_benchmark(args: argparse.Namespace):
    """Benchmark an existing model."""
    print(f"📊 Benchmarking {args.model} on {args.target}...")

    ext = os.path.splitext(args.model)[1].lower()
    if ext == ".onnx":
        # ONNX model — benchmark via ONNX Runtime
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(args.model)
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape

        dummy = np.random.randn(*input_shape).astype(np.float32)

        # Warmup
        for _ in range(10):
            session.run(None, {input_name: dummy})

        # Benchmark
        import time
        timings = []
        for _ in range(args.runs):
            start = time.perf_counter()
            session.run(None, {input_name: dummy})
            end = time.perf_counter()
            timings.append((end - start) * 1000)

        mean_ms = sum(timings) / len(timings)
        print(f"   Mean inference: {mean_ms:.3f} ms")
        print(f"   Throughput: {1000 / mean_ms:.1f} img/s")
    else:
        # PyTorch model
        model = torch.load(args.model, map_location=args.target)
        if isinstance(model, dict) and "state_dict" in model:
            # Try to reconstruct student and load weights
            print("   Loading checkpoint...")
            from src.student import MiniCNN
            model = MiniCNN(num_classes=10)
            model.load_state_dict(torch.load(args.model, map_location=args.target))

        results = benchmark(model, target=args.target, benchmark_runs=args.runs)
        print(f"   Mean    : {results['mean_ms']:.3f} ms")
        print(f"   Median  : {results['median_ms']:.3f} ms")
        print(f"   P95     : {results['p95_ms']:.3f} ms")
        print(f"   Through : {results['throughput_imgs_per_sec']:.1f} img/s")


def cmd_export(args: argparse.Namespace):
    """Export a trained model."""
    os.makedirs("checkpoints", exist_ok=True)

    # Load the PyTorch model
    from src.student import MiniCNN
    model = MiniCNN(num_classes=10)
    state = torch.load(args.model, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)

    output = args.output or f"checkpoints/student.{'onnx' if args.format == 'onnx' else 'pt'}"

    if args.format == "onnx":
        export_to_onnx(model, output)
    else:
        export_to_torchscript(model, output)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None or args.command == "gui":
        # Default to GUI if no command, or explicit gui command
        if args.command is None and not hasattr(args, "gui"):
            parser.print_help()
            sys.exit(1)
        try:
            from src.webapp import launch
            launch()
        except ImportError as e:
            print("❌ Web GUI dependencies not installed.")
            print("   Run: pip install -r requirements.txt")
            print(f"   Error: {e}")
            sys.exit(1)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "export":
        cmd_export(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

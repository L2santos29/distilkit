#!/usr/bin/env python3
"""DistilKit CLI — Run knowledge distillation from the command line."""

import argparse
import os
import sys

import torch
import torch.nn as nn

from src import datasets as ds
from src.benchmarks import benchmark, compare_teacher_student
from src.distiller import Distiller
from src.log_config import logger
from src.onnx_export import export_to_onnx, export_to_torchscript
from src.student import build_student
from src.teacher import load_teacher

DATASET_CHOICES = ds.DATASET_CHOICES
TEACHER_CHOICES = ds.TEACHER_CHOICES

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _get_dataset_loaders(dataset_name: str, batch_size: int, data_root: str = "./data"):
    """Return (train_loader, val_loader, num_classes, in_channels) via shared module."""
    return ds.get_dataset_loaders(dataset_name, batch_size, data_root)


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
        "--dataset",
        default="CIFAR-10",
        choices=DATASET_CHOICES,
        help="Dataset to use (default: CIFAR-10)",
    )
    train_parser.add_argument(
        "--teacher",
        default="resnet18",
        choices=TEACHER_CHOICES,
        help="Teacher model architecture (default: resnet18)",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    train_parser.add_argument(
        "--temperature",
        type=float,
        default=4.0,
        help="Distillation temperature — higher = softer targets (default: 4.0)",
    )
    train_parser.add_argument(
        "--compression-ratio",
        type=float,
        default=0.05,
        help="Target student/teacher parameter ratio (default: 0.05 = 5%)",
    )
    train_parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Distillation loss weight (0-1). Higher = more teacher influence (default: 0.7)",
    )
    train_parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stopping patience (0 to disable, default: 0)",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64)",
    )
    train_parser.add_argument(
        "--export",
        choices=["onnx", "torchscript", "none"],
        default="none",
        help="Export trained student model (default: none)",
    )
    train_parser.add_argument(
        "--benchmark",
        choices=["cpu", "cuda", "none"],
        default="cpu",
        help="Benchmark teacher vs student on target device (default: cpu)",
    )
    train_parser.add_argument(
        "--output-dir",
        default="checkpoints",
        help="Directory for exported models (default: checkpoints)",
    )
    train_parser.add_argument(
        "--ckpt-every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs (0 to disable, default: 5)",
    )
    train_parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint to resume from (.pt file)",
    )
    train_parser.add_argument(
        "--data-dir",
        default="./data",
        help="Dataset cache directory (default: ./data)",
    )

    # ---- benchmark ----
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark a trained model")
    bench_parser.add_argument(
        "--model",
        required=True,
        help="Path to model file (.onnx, .pt, or .pth)",
    )
    bench_parser.add_argument(
        "--target",
        choices=["cpu", "cuda", "npu"],
        default="cpu",
        help="Target device (default: cpu)",
    )
    bench_parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of inference runs (default: 100)",
    )

    # ---- export ----
    export_parser = subparsers.add_parser(
        "export", help="Export a trained model to ONNX or TorchScript"
    )
    export_parser.add_argument(
        "--model",
        required=True,
        help="Path to PyTorch model (.pth)",
    )
    export_parser.add_argument(
        "--format",
        choices=["onnx", "torchscript"],
        default="onnx",
        help="Export format (default: onnx)",
    )
    export_parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: checkpoints/student.<format>)",
    )

    # ---- gui ----
    gui_parser = subparsers.add_parser(
        "gui", help="Launch the web-based GUI (FastAPI + Tailwind CSS)"
    )
    gui_parser.add_argument(
        "--api-only",
        action="store_true",
        help="Run in API-only mode (no frontend)",
    )

    # ---- api ----
    subparsers.add_parser("api", help="Launch API-only server (no frontend)")

    return parser


def cmd_train(args: argparse.Namespace) -> nn.Module | None:
    """Run the full distillation pipeline."""
    logger.info("=" * 60)
    logger.info("  ⚡ DistilKit — Knowledge Distillation")
    logger.info("=" * 60)
    logger.info(f"  Dataset     : {args.dataset}")
    logger.info(f"  Teacher     : {args.teacher}")
    logger.info(f"  Epochs      : {args.epochs}")
    logger.info(f"  Temperature : {args.temperature}")
    logger.info(f"  Alpha       : {args.alpha}")
    logger.info(f"  Batch size  : {args.batch_size}")
    logger.info("=" * 60)
    print()

    # 1. Data
    logger.info(f"📦 Loading {args.dataset}...")
    train_loader, val_loader, num_classes, in_channels = _get_dataset_loaders(
        args.dataset, args.batch_size, args.data_dir
    )

    # 2. Teacher
    logger.info(f"🧠 Loading teacher ({args.teacher})...")
    teacher = load_teacher(args.teacher, num_classes=num_classes)
    teacher_params = sum(p.numel() for p in teacher.parameters())
    logger.info(f"   Parameters: {teacher_params:,}")

    # 3. Student
    logger.info("🔧 Building student...")
    student = build_student(
        teacher=teacher,
        student_type="MiniCNN",
        compression_ratio=args.compression_ratio,
        num_classes=num_classes,
        in_channels=in_channels,
    )
    student_params = sum(p.numel() for p in student.parameters())
    logger.info(f"   Parameters: {student_params:,}")
    logger.info(f"   Compression: {student_params / teacher_params:.2%} of teacher size")
    print()

    # 4. Distill
    logger.info(f"🔄 Training ({args.epochs} epochs)...")
    os.makedirs("checkpoints", exist_ok=True)

    if args.resume and os.path.exists(args.resume):
        logger.info(f"📂 Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        student.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0)
        logger.info(f"   Resumed at epoch {start_epoch}")
    else:
        start_epoch = 0

    distiller = Distiller(
        teacher,
        student,
        temperature=args.temperature,
        alpha=args.alpha,
    )

    # For checkpointing, we implement the training loop here
    # instead of calling distiller.train() so we can save mid-training

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(start_epoch, args.epochs):
        # Train
        student.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to("cpu"), labels.to("cpu")
            with torch.no_grad():
                teacher_logits = teacher(images)
            student_logits = student(images)
            loss = distiller.criterion(student_logits, teacher_logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        history["train_loss"].append(avg_loss)

        # Validate
        student.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to("cpu"), labels.to("cpu")
                outputs = student(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        acc = correct / total
        history["val_acc"].append(acc)

        # Early stopping
        if args.patience > 0:
            if not hasattr(cmd_train, "_best_acc"):
                cmd_train._best_acc = 0.0
                cmd_train._patience_counter = 0
            if acc > cmd_train._best_acc + 0.001:
                cmd_train._best_acc = acc
                cmd_train._patience_counter = 0
            else:
                cmd_train._patience_counter += 1
                if cmd_train._patience_counter >= args.patience:
                    logger.info(f"   ⏹️ Early stopping (best: {cmd_train._best_acc:.2%})")
                    break

        scheduler.step()

        logger.info(f"Epoch {epoch + 1}/{args.epochs} — Loss: {avg_loss:.4f} — Val Acc: {acc:.2%}")

        # Checkpoint
        if args.ckpt_every > 0 and (epoch + 1) % args.ckpt_every == 0:
            ckpt_path = f"checkpoints/checkpoint_epoch_{epoch + 1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "losses": history["train_loss"],
                    "accuracies": history["val_acc"],
                    "config": vars(args),
                },
                ckpt_path,
            )
            logger.info(f"   💾 Checkpoint saved: {ckpt_path}")

    print()

    # 5. Benchmark
    if args.benchmark != "none":
        logger.info(f"📊 Benchmarking on {args.benchmark}...")
        comparison = compare_teacher_student(teacher, student, target=args.benchmark)
        logger.info(
            f"   Teacher : {comparison['teacher']['mean_ms']:.2f} ms  "
            f"({comparison['teacher']['parameters']:,} params)"
        )
        logger.info(
            f"   Student : {comparison['student']['mean_ms']:.2f} ms  "
            f"({comparison['student']['parameters']:,} params)"
        )
        logger.info(f"   Speedup : {comparison['speedup']}x")
        logger.info(f"   Size    : {comparison['compression']:.2%} of teacher")
        print()

    # 6. Export
    if args.export != "none":
        os.makedirs(args.output_dir, exist_ok=True)
        if args.export == "onnx":
            path = export_to_onnx(student, f"{args.output_dir}/student.onnx")
        else:
            path = export_to_torchscript(student, f"{args.output_dir}/student.pt")
        logger.info(f"   Exported to: {path}")

    print()
    logger.info("✅ Training complete!")
    return student


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Benchmark an existing model."""
    logger.info(f"📊 Benchmarking {args.model} on {args.target}...")

    ext = os.path.splitext(args.model)[1].lower()
    if ext == ".onnx":
        # ONNX model — benchmark via ONNX Runtime
        import numpy as np
        import onnxruntime as ort

        nproc = os.cpu_count() or 4

        # ONNX Runtime session with optimized thread settings for CPU
        so = ort.SessionOptions()
        so.intra_op_num_threads = nproc
        so.inter_op_num_threads = max(1, nproc // 2)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_cpu_mem_arena = True

        session = ort.InferenceSession(args.model, so)
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape

        logger.info(f"   ONNX Runtime threads: intra={so.intra_op_num_threads}, inter={so.inter_op_num_threads}")

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
        logger.info(f"   Mean inference: {mean_ms:.3f} ms")
        logger.info(f"   Throughput: {1000 / mean_ms:.1f} img/s")
    else:
        # PyTorch model
        model = torch.load(args.model, map_location=args.target)
        if isinstance(model, dict) and "state_dict" in model:
            # Try to reconstruct student and load weights
            logger.info("   Loading checkpoint...")
            from src.student import MiniCNN

            model = MiniCNN(num_classes=10)
            model.load_state_dict(torch.load(args.model, map_location=args.target))

        results = benchmark(model, target=args.target, benchmark_runs=args.runs)
        logger.info(f"   Mean    : {results['mean_ms']:.3f} ms")
        logger.info(f"   Median  : {results['median_ms']:.3f} ms")
        logger.info(f"   P95     : {results['p95_ms']:.3f} ms")
        logger.info(f"   Through : {results['throughput_imgs_per_sec']:.1f} img/s")


def cmd_export(args: argparse.Namespace) -> None:
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


def main() -> None:
    """Entry point: parse arguments and dispatch to the right command."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None or args.command == "gui":
        # Default to GUI if no command, or explicit gui command
        if args.command is None and not hasattr(args, "gui"):
            parser.print_help()
            sys.exit(1)
        try:
            from src.webapp import launch

            api_only = getattr(args, "api_only", False) or args.command == "api"
            launch(api_only=api_only)
        except ImportError as e:
            logger.info("❌ Web GUI dependencies not installed.")
            logger.info("   Run: pip install -r requirements.txt")
            logger.info(f"   Error: {e}")
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

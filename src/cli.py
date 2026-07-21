#!/usr/bin/env python3
"""DistilKit CLI — Run knowledge distillation from the command line."""

import argparse
import os
import sys

import torch
import torch.nn as nn

from src import datasets as ds
from src.benchmarks import MS_PER_SEC, benchmark
from src.log_config import logger
from src.pipeline import PipelineError, run_distillation_pipeline
from src.settings import settings

DATASET_CHOICES = ds.DATASET_CHOICES
TEACHER_CHOICES = ds.TEACHER_CHOICES


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
        default=settings.device,
        help=f"Benchmark target device (default: {settings.device})",
    )
    train_parser.add_argument(
        "--output-dir",
        default=settings.checkpoints_dir,
        help=f"Directory for exported models (default: {settings.checkpoints_dir})",
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
        default=settings.data_dir,
        help=f"Dataset cache directory (default: {settings.data_dir})",
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
        default=settings.device,
        help=f"Target device (default: {settings.device})",
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

    try:
        result = run_distillation_pipeline(
            dataset_name=args.dataset,
            teacher_name=args.teacher,
            student_type="MiniCNN",
            compression_ratio=args.compression_ratio,
            batch_size=args.batch_size,
            data_root=args.data_dir,
            epochs=args.epochs,
            temperature=args.temperature,
            alpha=args.alpha,
            device=settings.device,
            patience=args.patience,
            ckpt_dir="checkpoints",
            ckpt_every=args.ckpt_every,
            resume=args.resume,
            benchmark_target=args.benchmark,
            export_format=args.export,
            export_output_dir=args.output_dir,
            on_message=logger.info,
        )
    except PipelineError as e:
        logger.error(f"❌ {e}")
        sys.exit(1)

    print()
    logger.info("✅ Training complete!")
    return result["student"]


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Benchmark an existing model."""
    logger.info(f"📊 Benchmarking {args.model} on {args.target}...")

    ext = os.path.splitext(args.model)[1].lower()
    if ext == ".onnx":
        # ONNX model — benchmark via ONNX Runtime
        import numpy as np
        import onnxruntime as ort

        nproc = os.cpu_count() or 4

        try:
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
                timings.append((end - start) * MS_PER_SEC)

            mean_ms = sum(timings) / len(timings)
            logger.info(f"   Mean inference: {mean_ms:.3f} ms")
            logger.info(f"   Throughput: {MS_PER_SEC / mean_ms:.1f} img/s")
        except (OSError, RuntimeError, ValueError) as e:
            logger.error(f"❌ ONNX benchmark failed for {args.model}: {e}")
            return
        except Exception as e:
            # onnxruntime exceptions (NoSuchFile, RuntimeException, etc.)
            # inherit directly from Exception with no public base class.
            logger.error(f"❌ ONNX benchmark failed for {args.model}: {e}")
            return
    else:
        # PyTorch model
        try:
            model = torch.load(args.model, map_location=args.target, weights_only=False)
        except (OSError, RuntimeError, ValueError) as e:
            logger.error(f"❌ Failed to load model {args.model}: {e}")
            return
        if isinstance(model, dict) and "state_dict" in model:
            # Try to reconstruct student and load weights
            logger.info("   Loading checkpoint...")
            from src.student import MiniCNN

            try:
                student = MiniCNN(num_classes=10)
                student.load_state_dict(model["state_dict"])
                model = student
            except (KeyError, OSError, RuntimeError, ValueError) as e:
                logger.error(f"❌ Failed to load checkpoint {args.model}: {e}")
                return

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

    try:
        model = MiniCNN(num_classes=10)
        state = torch.load(args.model, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)
    except Exception as e:
        logger.error(f"❌ Failed to load model {args.model}: {e}")
        return

    output = args.output or f"checkpoints/student.{'onnx' if args.format == 'onnx' else 'pt'}"

    from src.onnx_export import export_to_onnx, export_to_torchscript

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

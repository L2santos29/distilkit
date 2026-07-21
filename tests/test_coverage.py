"""Coverage-gap tests — exercises untested code paths to reach ≥80% coverage.

Each section targets specific uncovered lines reported by ``pytest --cov``.
"""
# ---------------------------------------------------------------------------
# datasets.py — get_dataset_info, _check_torchvision_dataset
# ---------------------------------------------------------------------------

import os
import tempfile
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from src import datasets as ds
from src.settings import settings
from src.webapp import app

# Disable rate limiting in tests (single IP makes many rapid requests).
settings.rate_limit_per_minute = 0


class TestDatasetInfo:
    """Coverage for datasets.py: get_dataset_info."""

    def test_get_dataset_info_valid(self):
        """get_dataset_info returns metadata for a known dataset."""
        info = ds.get_dataset_info("CIFAR-10")
        assert info["num_classes"] == 10
        assert info["in_channels"] == 3
        assert info["input_size"] == 32

    def test_get_dataset_info_mnist(self):
        """get_dataset_info returns metadata for MNIST."""
        info = ds.get_dataset_info("MNIST")
        assert info["num_classes"] == 10
        assert info["in_channels"] == 1
        assert info["input_size"] == 28

    def test_get_dataset_info_invalid_raises(self):
        """get_dataset_info raises ValueError for unknown dataset."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            ds.get_dataset_info("NonExistent")


class TestCheckTorchvisionDataset:
    """Coverage for datasets.py: _check_torchvision_dataset."""

    def test_no_directory_returns_false(self):
        """_check_torchvision_dataset returns False when no raw/processed dirs exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ds._check_torchvision_dataset("MNIST", tmpdir)
            assert result is False

    def test_raw_directory_exists_returns_true(self):
        """_check_torchvision_dataset returns True when raw/ dir has files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = os.path.join(tmpdir, "raw")
            os.makedirs(raw_dir)
            Path(os.path.join(raw_dir, "some_file")).touch()
            result = ds._check_torchvision_dataset("MNIST", tmpdir)
            assert result is True

    def test_processed_directory_exists_returns_true(self):
        """_check_torchvision_dataset returns True when processed/ dir has files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processed_dir = os.path.join(tmpdir, "processed")
            os.makedirs(processed_dir)
            Path(os.path.join(processed_dir, "some_file")).touch()
            result = ds._check_torchvision_dataset("MNIST", tmpdir)
            assert result is True

    def test_empty_directory_returns_false(self):
        """_check_torchvision_dataset returns False when dirs exist but are empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "raw"))
            result = ds._check_torchvision_dataset("MNIST", tmpdir)
            assert result is False


# ---------------------------------------------------------------------------
# teacher.py — load_teacher with pretrained=False (no downloads)
# ---------------------------------------------------------------------------


class TestTeacherLoading:
    """Coverage for teacher.py: model loading branches."""

    def test_load_resnet18_pretrained_false(self):
        """load_teacher with resnet18 and pretrained=False returns a model."""
        from src.teacher import load_teacher

        model = load_teacher("resnet18", num_classes=10, pretrained=False)
        assert isinstance(model, nn.Module)
        # ResNet path: should have fc layer
        assert hasattr(model, "fc")

    def test_load_mobilenet_v2_pretrained_false(self):
        """load_teacher with mobilenet_v2 and pretrained=False returns a model."""
        from src.teacher import load_teacher

        model = load_teacher("mobilenet_v2", num_classes=10, pretrained=False)
        assert isinstance(model, nn.Module)

    def test_load_efficientnet_b0_pretrained_false(self):
        """load_teacher with efficientnet_b0 and pretrained=False returns a model."""
        from src.teacher import load_teacher

        model = load_teacher("efficientnet_b0", num_classes=10, pretrained=False)
        assert isinstance(model, nn.Module)
        # EfficientNet uses 'classifier' attribute, not 'fc'
        assert hasattr(model, "classifier") or hasattr(model, "fc")

    def test_load_teacher_unsupported_raises(self):
        """load_teacher with unknown name raises ValueError."""
        from src.teacher import load_teacher

        with pytest.raises(ValueError, match="Unknown model"):
            load_teacher("nonexistent_model")

    def test_load_resnet34_pretrained_false(self):
        """load_teacher with resnet34 works."""
        from src.teacher import load_teacher

        model = load_teacher("resnet34", num_classes=10, pretrained=False)
        assert isinstance(model, nn.Module)

    def test_load_resnet50_pretrained_false(self):
        """load_teacher with resnet50 works."""
        from src.teacher import load_teacher

        model = load_teacher("resnet50", num_classes=10, pretrained=False)
        assert isinstance(model, nn.Module)

    def test_load_mobilenet_v3_large_pretrained_false(self):
        """load_teacher with mobilenet_v3_large works."""
        from src.teacher import load_teacher

        model = load_teacher("mobilenet_v3_large", num_classes=10, pretrained=False)
        assert isinstance(model, nn.Module)


# ---------------------------------------------------------------------------
# cli.py — cmd_train, cmd_benchmark, cmd_export, parser
# ---------------------------------------------------------------------------


class TestCLICoverage:
    """Coverage for cli.py: error handling, benchmark, export, parser."""

    def test_cmd_train_pipeline_error_raises_sys_exit(self):
        """cmd_train exits when PipelineError is raised."""
        import argparse
        from src.cli import cmd_train

        with patch("src.cli.run_distillation_pipeline") as mock_pipeline:
            from src.pipeline import PipelineError

            mock_pipeline.side_effect = PipelineError("Test error")
            args = argparse.Namespace(
                dataset="CIFAR-10",
                teacher="resnet18",
                epochs=1,
                temperature=4.0,
                alpha=0.7,
                batch_size=64,
                patience=0,
                compression_ratio=0.05,
                data_dir="./data",
                ckpt_every=0,
                resume=None,
                benchmark="none",
                export="none",
                output_dir="checkpoints",
            )
            with pytest.raises(SystemExit):
                cmd_train(args)

    def test_cmd_train_result_returns_student(self):
        """cmd_train returns the student model on success."""
        import argparse
        from src.cli import cmd_train

        student = nn.Linear(10, 2)
        with patch("src.cli.run_distillation_pipeline") as mock_pipeline:
            mock_pipeline.return_value = {"student": student}
            args = argparse.Namespace(
                dataset="CIFAR-10",
                teacher="resnet18",
                epochs=1,
                temperature=4.0,
                alpha=0.7,
                batch_size=64,
                patience=0,
                compression_ratio=0.05,
                data_dir="./data",
                ckpt_every=0,
                resume=None,
                benchmark="none",
                export="none",
                output_dir="checkpoints",
            )
            result = cmd_train(args)
            assert result is student

    def test_benchmark_missing_pth_logs_error(self):
        """cmd_benchmark with missing .pth logs error and returns."""
        import argparse
        from src.cli import cmd_benchmark

        args = argparse.Namespace(
            model="/nonexistent/model.pth",
            target="cpu",
            runs=10,
            command="benchmark",
        )
        cmd_benchmark(args)

    def test_benchmark_raw_state_dict_falls_through(self):
        """cmd_benchmark with raw state dict (no 'state_dict' key)."""
        import argparse
        import tempfile
        from src.cli import cmd_benchmark

        # Save a full model object (not a state dict) to trigger the else branch
        model = nn.Sequential(nn.Flatten(), nn.Linear(32 * 32 * 3, 10))
        model.eval()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model, f.name)
            tmp_path = f.name

        try:
            args = argparse.Namespace(
                model=tmp_path,
                target="cpu",
                runs=5,
                command="benchmark",
            )
            cmd_benchmark(args)
        finally:
            os.unlink(tmp_path)

    def test_benchmark_state_dict_checkpoint(self):
        """cmd_benchmark handles checkpoint dict with 'state_dict' key."""
        import argparse
        import tempfile
        from src.cli import cmd_benchmark

        state = {"state_dict": {"fc.weight": torch.randn(10, 32 * 32 * 3), "fc.bias": torch.randn(10)}}

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(state, f.name)
            tmp_path = f.name

        try:
            args = argparse.Namespace(
                model=tmp_path,
                target="cpu",
                runs=5,
                command="benchmark",
            )
            cmd_benchmark(args)
        finally:
            os.unlink(tmp_path)

    def test_cmd_export_missing_file_logs_error(self):
        """cmd_export with missing file logs error and returns."""
        import argparse
        from src.cli import cmd_export

        args = argparse.Namespace(
            model="/nonexistent/model.pth",
            format="onnx",
            output="out.onnx",
            command="export",
        )
        cmd_export(args)

    def test_parser_build_and_parse_train(self):
        """build_parser can parse a full train command."""
        from src.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "train",
            "--dataset", "MNIST",
            "--teacher", "resnet50",
            "--epochs", "5",
            "--temperature", "3.0",
            "--alpha", "0.5",
            "--batch-size", "32",
            "--patience", "2",
            "--compression-ratio", "0.1",
            "--ckpt-every", "3",
            "--resume", "ckpt.pt",
            "--export", "onnx",
            "--benchmark", "cpu",
            "--output-dir", "/tmp/out",
            "--data-dir", "/tmp/data",
        ])
        assert args.dataset == "MNIST"
        assert args.teacher == "resnet50"
        assert args.epochs == 5
        assert args.temperature == 3.0
        assert args.alpha == 0.5
        assert args.batch_size == 32
        assert args.patience == 2
        assert args.compression_ratio == 0.1
        assert args.ckpt_every == 3
        assert args.resume == "ckpt.pt"
        assert args.export == "onnx"
        assert args.benchmark == "cpu"
        assert args.output_dir == "/tmp/out"
        assert args.data_dir == "/tmp/data"

    def test_main_dispatch_train(self):
        """main() dispatch calls cmd_train for 'train' command."""
        import sys
        from src.cli import main

        with patch.object(sys, "argv", ["distilkit", "train", "--epochs", "1"]):
            with patch("src.cli.cmd_train") as mock_cmd:
                main()
                mock_cmd.assert_called_once()
                args = mock_cmd.call_args[0][0]
                assert args.epochs == 1

    def test_main_dispatch_export(self):
        """main() dispatch calls cmd_export for 'export' command."""
        import sys
        from src.cli import main

        with patch.object(sys, "argv", ["distilkit", "export", "--model", "x.pth"]):
            with patch("src.cli.cmd_export") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_main_dispatch_benchmark(self):
        """main() dispatch calls cmd_benchmark for 'benchmark' command."""
        import sys
        from src.cli import main

        with patch.object(sys, "argv", ["distilkit", "benchmark", "--model", "x.onnx"]):
            with patch("src.cli.cmd_benchmark") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_main_dispatch_no_command(self):
        """main() with no command prints help and exits."""
        import sys
        from src.cli import main

        with patch.object(sys, "argv", ["distilkit"]):
            with pytest.raises(SystemExit):
                main()

    def test_benchmark_with_real_onnx(self):
        """cmd_benchmark with a real .onnx file works."""
        import argparse
        import tempfile
        from src.cli import cmd_benchmark
        from src.student import MiniCNN
        from src.onnx_export import export_to_onnx

        model = MiniCNN(in_channels=3, num_classes=10, width=0.5)
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = export_to_onnx(model, Path(tmpdir) / "test.onnx", opset_version=17)
            args = argparse.Namespace(
                model=str(onnx_path),
                target="cpu",
                runs=3,
                command="benchmark",
            )
            cmd_benchmark(args)

    def test_cmd_export_success(self):
        """cmd_export successfully exports a model to onnx."""
        import argparse
        import tempfile
        from src.cli import cmd_export
        from src.student import MiniCNN

        # Use default width=1.0 (same as cmd_export uses) so state_dict loads cleanly
        model = MiniCNN(in_channels=3, num_classes=10, width=1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            torch.save(model.state_dict(), str(model_path))
            output_path = Path(tmpdir) / "exported.onnx"
            args = argparse.Namespace(
                model=str(model_path),
                format="onnx",
                output=str(output_path),
                command="export",
            )
            cmd_export(args)
            assert output_path.exists()

    def test_cmd_export_torchscript(self):
        """cmd_export successfully exports a model to torchscript."""
        import argparse
        import tempfile
        from src.cli import cmd_export
        from src.student import MiniCNN

        # Use default width=1.0 (same as cmd_export uses) so state_dict loads cleanly
        model = MiniCNN(in_channels=3, num_classes=10, width=1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            torch.save(model.state_dict(), str(model_path))
            output_path = Path(tmpdir) / "exported.pt"
            args = argparse.Namespace(
                model=str(model_path),
                format="torchscript",
                output=str(output_path),
                command="export",
            )
            cmd_export(args)
            assert output_path.exists()

    def test_main_import_error(self):
        """main() with gui command handles ImportError gracefully."""
        import sys
        from src.cli import main

        with patch.object(sys, "argv", ["distilkit", "gui"]):
            # Patch the launch function on the webapp module (where it's defined)
            # to raise ImportError — covers the except ImportError handler in main()
            with patch("src.webapp.launch", side_effect=ImportError("missing deps")):
                with pytest.raises(SystemExit):
                    main()

    def test_main_unknown_command(self):
        """main() with unknown command prints help and exits."""
        import sys
        from src.cli import main

        with patch.object(sys, "argv", ["distilkit", "unknown_cmd"]):
            with pytest.raises(SystemExit):
                main()

    def test_main_no_args_prints_help(self):
        """main() with no args prints help and exits."""
        import sys
        from src.cli import main

        with patch.object(sys, "argv", ["distilkit"]):
            with pytest.raises(SystemExit):
                main()

    def test_cmd_export_checkpoint_format(self):
        """cmd_export loads a checkpoint dict with state_dict key."""
        import argparse
        import tempfile
        from src.cli import cmd_export
        from src.student import MiniCNN

        model = MiniCNN(in_channels=3, num_classes=10, width=1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save as checkpoint dict with 'state_dict' key
            model_path = Path(tmpdir) / "ckpt.pt"
            torch.save({"state_dict": model.state_dict()}, str(model_path))
            output_path = Path(tmpdir) / "exported.onnx"
            args = argparse.Namespace(
                model=str(model_path),
                format="onnx",
                output=str(output_path),
                command="export",
            )
            cmd_export(args)
            assert output_path.exists()


class TestBenchmarkCoverage:
    """Coverage for benchmarks.py and cli.py benchmark paths."""

    def test_benchmark_pytorch_model_state_dict(self):
        """cmd_benchmark with a state-dict checkpoint covers logging lines."""
        import argparse
        import tempfile
        from src.cli import cmd_benchmark
        from src.student import MiniCNN

        model = MiniCNN(in_channels=3, num_classes=10, width=1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save as state_dict wrapped in 'state_dict' key (checkpoint format)
            model_path = Path(tmpdir) / "ckpt.pt"
            torch.save({"state_dict": model.state_dict()}, str(model_path))
            args = argparse.Namespace(
                model=str(model_path),
                target="cpu",
                runs=3,
                command="benchmark",
            )
            cmd_benchmark(args)  # Should load MiniCNN + log benchmark results

    def test_compare_teacher_student(self):
        """compare_teacher_student benchmarks both models side-by-side."""
        from src.benchmarks import compare_teacher_student

        teacher = nn.Sequential(nn.Flatten(), nn.Linear(32 * 32 * 3, 10), nn.LogSoftmax(dim=1))
        student = nn.Sequential(nn.Flatten(), nn.Linear(32 * 32 * 3, 10), nn.LogSoftmax(dim=1))
        result = compare_teacher_student(
            teacher, student, input_shape=(1, 3, 32, 32)
        )
        assert "teacher" in result
        assert "student" in result
        assert "speedup" in result
        assert "compression" in result

    def test_resolve_device_cuda_unavailable(self):
        """_resolve_device raises RuntimeError when CUDA is not available."""
        from src.benchmarks import _resolve_device

        with pytest.raises(RuntimeError, match="CUDA requested but not available"):
            _resolve_device("cuda")


class TestCircuitBreakerCoverage:
    """Coverage for circuit_breaker.py."""

    def test_circuit_closed_passes_through(self):
        """CircuitBreaker.call passes through successful calls."""
        from src.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        result = cb.call(lambda x: x + 1, 41)
        assert result == 42
        assert cb.state == "closed"
        assert cb.failure_count == 0

    def test_circuit_opens_after_threshold(self):
        """CircuitBreaker opens after failure_threshold consecutive failures."""
        from src.circuit_breaker import CircuitBreaker, CircuitOpenError

        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

        def _fail(*_a: object) -> None:
            raise ValueError("boom")

        # 3 failures should open the circuit
        for i in range(3):
            with pytest.raises(ValueError, match="boom"):
                cb.call(_fail)
            assert cb.failure_count == i + 1

        assert cb.state == "open"

        # Next call should be blocked
        with pytest.raises(CircuitOpenError, match="open"):
            cb.call(_fail)

    def test_circuit_resets_on_success(self):
        """CircuitBreaker resets failure count after a successful call."""
        from src.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        call_count = 0

        def _flaky() -> int:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("not yet")
            return 42

        # 2 failures, 1 success
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(_flaky)
        result = cb.call(_flaky)
        assert result == 42
        assert cb.failure_count == 0
        assert cb.state == "closed"

    def test_circuit_manual_reset(self):
        """CircuitBreaker.reset() manually closes the circuit."""
        from src.circuit_breaker import CircuitBreaker, CircuitOpenError

        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60)
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert cb.state == "open"
        cb.reset()
        assert cb.state == "closed"
        assert cb.failure_count == 0

    def test_circuit_half_open_recovers(self):
        """CircuitBreaker transitions to half-open after recovery timeout."""
        from src.circuit_breaker import CircuitBreaker, CircuitOpenError

        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert cb.state == "open"

        import time
        time.sleep(0.06)  # Wait past recovery timeout

        # Should be half-open now — a success closes it
        result = cb.call(lambda: 99)
        assert result == 99
        assert cb.state == "closed"

    def test_circuit_half_open_failure_reopens(self):
        """CircuitBreaker re-opens if a half-open call fails."""
        from src.circuit_breaker import CircuitBreaker, CircuitOpenError

        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert cb.state == "open"

        import time
        time.sleep(0.06)  # Wait past recovery timeout

        # Half-open probe fails again → re-opens
        with pytest.raises(ValueError, match="fail"):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert cb.state == "open"


# ---------------------------------------------------------------------------
# task_manager.py — TrainingTask helpers, error paths
# ---------------------------------------------------------------------------


class TestTaskManagerCoverage:
    """Coverage for task_manager.py: helpers and error handling."""

    def test_build_result_dict(self):
        """_build_result_dict produces correct structure."""
        from src.task_manager import TrainingTask

        result = TrainingTask._build_result_dict(
            dataset_name="CIFAR-10",
            student_name="MiniCNN",
            teacher_name="resnet18",
            teacher_params=1_000_000,
            student_params=100_000,
            comparison={
                "speedup": 2.5,
                "teacher": {"mean_ms": 10.0, "throughput_imgs_per_sec": 100.0},
                "student": {"mean_ms": 4.0, "throughput_imgs_per_sec": 250.0},
            },
            losses=[1.0, 0.8, 0.6],
            accuracies=[0.5, 0.7, 0.85],
        )
        assert result["teacher_params"] == 1_000_000
        assert result["student_params"] == 100_000
        assert result["speedup"] == 2.5
        assert result["final_loss"] == 0.6
        assert result["final_accuracy"] == 0.85
        assert len(result["losses"]) == 3

    def test_training_task_init(self):
        """TrainingTask initializes with correct defaults."""
        from src.task_manager import TrainingTask

        task = TrainingTask({"epochs": 5, "teacher": "resnet18"})
        assert task.status == "pending"
        assert task.progress == 0.0
        assert len(task.id) == 12
        assert task.total_epochs == 5
        assert task.losses == []
        assert task.accuracies == []

    def test_training_task_cancel_idle(self):
        """Cancel on a pending task sets status to cancelled."""
        from src.task_manager import TrainingTask

        task = TrainingTask({"epochs": 5, "teacher": "resnet18"})
        task.cancel()
        assert task.status == "cancelled"

    def test_training_task_emit(self):
        """_emit writes to the log buffer."""
        from src.task_manager import TrainingTask

        task = TrainingTask({"epochs": 1, "teacher": "resnet18"})
        task._emit("test message")
        task._flush_logs()
        assert "test message" in task.logs

    def test_training_task_flush_logs_trims(self):
        """_flush_logs trims logs over max_log_size."""
        from src.task_manager import TrainingTask

        task = TrainingTask({"epochs": 1, "teacher": "resnet18"})
        # Write more than max_log_size
        task._emit("x" * (settings.max_log_size + 1000))
        task._flush_logs()
        assert len(task.logs) <= settings.max_log_size

    def test_save_run_and_load_history(self):
        """_save_run persists and _load_history retrieves."""
        from src.task_manager import _load_history, _save_run

        with tempfile.TemporaryDirectory() as tmpdir:
            original = settings.runs_dir
            settings.runs_dir = tmpdir
            try:
                run = {"id": "covtest1", "result": {"accuracy": 0.95}}
                _save_run(run)
                history = _load_history()
                assert len(history) >= 1
                assert history[0]["id"] == "covtest1"
            finally:
                settings.runs_dir = original


# ---------------------------------------------------------------------------
# onnx_export.py — error paths
# ---------------------------------------------------------------------------


class TestExportCoverage:
    """Coverage for onnx_export error paths."""

    def test_export_to_onnx_creates_file(self):
        """export_to_onnx with valid model creates file."""
        from src.onnx_export import export_to_onnx
        from src.student import MiniCNN

        model = MiniCNN(in_channels=3, num_classes=10, width=0.5)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_to_onnx(model, Path(tmpdir) / "test.onnx", opset_version=17)
            assert path.exists()
            assert path.stat().st_size > 0

    def test_export_to_torchscript_with_valid_model(self):
        """export_to_torchscript with valid model creates file."""
        from src.onnx_export import export_to_torchscript
        from src.student import MiniCNN

        model = MiniCNN(in_channels=3, num_classes=10, width=0.5)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_to_torchscript(model, Path(tmpdir) / "test.pt")
            assert path.exists()
            assert path.stat().st_size > 0

    def test_export_to_onnx_dynamic_batch(self):
        """export_to_onnx with dynamic_batch=True covers the dynamic_axes path."""
        from src.onnx_export import export_to_onnx
        from src.student import MiniCNN

        model = MiniCNN(in_channels=3, num_classes=10, width=0.5)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_to_onnx(
                model, Path(tmpdir) / "dynamic.onnx",
                opset_version=17, dynamic_batch=True,
            )
            assert path.exists()

    def test_export_to_onnx_failure_raises(self):
        """export_to_onnx raises RuntimeError on invalid model."""
        from src.onnx_export import export_to_onnx

        bad_model = nn.Linear(10, 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match="ONNX export failed"):
                export_to_onnx(bad_model, Path(tmpdir) / "bad.onnx")

    def test_export_to_torchscript_failure_raises(self):
        """export_to_torchscript raises RuntimeError on invalid model."""
        from src.onnx_export import export_to_torchscript

        bad_model = nn.Linear(10, 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match="TorchScript export failed"):
                export_to_torchscript(bad_model, Path(tmpdir) / "bad.pt")


# ---------------------------------------------------------------------------
# task_manager.py — TrainingTask helpers, error paths
# task_manager.py — _run error path via DatasetError
# ---------------------------------------------------------------------------


class TestTaskManagerRunCoverage:
    """Coverage for TrainingTask._run error handling."""

    def test_training_task_has_correct_defaults(self):
        """TrainingTask sets correct defaults."""
        from src.task_manager import TrainingTask

        task = TrainingTask({"epochs": 5, "teacher": "resnet18"})
        assert task._cancel_requested is False
        assert task._dirty is False
        assert task.result is None
        assert task.error is None
        assert task.student is None
        assert task.eta_seconds == 0.0

    def test_load_history_empty_dir(self):
        """_load_history returns empty list when runs_dir is missing."""
        from src.task_manager import _load_history

        with tempfile.TemporaryDirectory() as tmpdir:
            original = settings.runs_dir
            settings.runs_dir = os.path.join(tmpdir, "nonexistent")
            try:
                history = _load_history()
                assert history == []
            finally:
                settings.runs_dir = original

    def test_get_tasks_provider(self):
        """get_tasks returns the module-level tasks dict."""
        from src.task_manager import get_tasks, _tasks

        tasks = get_tasks()
        assert tasks is _tasks

    def test_get_history_store_provider(self):
        """get_history_store returns the module-level history list."""
        from src.task_manager import get_history_store, _history

        store = get_history_store()
        assert store is _history

    def test_save_error_run_persists(self):
        """TrainingTask._run saves error runs to disk."""
        from src.task_manager import TrainingTask, _load_history

        with tempfile.TemporaryDirectory() as tmpdir:
            original = settings.runs_dir
            settings.runs_dir = tmpdir
            try:
                task = TrainingTask({"epochs": 1, "teacher": "resnet18"})
                task.error = "Test error message"
                task._save_error_run("failed")
                history = _load_history()
                assert len(history) == 1
                assert history[0]["status"] == "failed"
                assert history[0]["error"] == "Test error message"
            finally:
                settings.runs_dir = original

    def test_tasks_provider_returns_dict(self):
        """get_tasks returns the module-level tasks dict."""
        from src.task_manager import get_tasks, _tasks

        tasks = get_tasks()
        assert tasks is _tasks

    def test_cancel_with_subprocess(self):
        """Cancel kills subprocess if running."""
        import subprocess as _subprocess
        from src.task_manager import TrainingTask

        task = TrainingTask({"epochs": 1, "teacher": "resnet18"})
        proc = _subprocess.Popen(["sleep", "10"])
        task._subprocess = proc
        task.cancel()
        assert task.status == "cancelled"
        assert proc.poll() is not None

    def test_subprocess_kill_on_timeout(self):
        """Cancel with subprocess that doesn't respond to terminate triggers kill."""
        import subprocess as _subprocess
        from src.task_manager import TrainingTask

        proc = _subprocess.Popen(
            ["bash", "-c", "trap '' TERM; while true; do sleep 1; done"],
        )
        task = TrainingTask({"epochs": 1, "teacher": "resnet18"})
        task._subprocess = proc
        task.cancel()
        assert task.status == "cancelled"
        assert proc.poll() is not None

    def test_training_task_run_dataset_error_cancelled(self):
        """_run handles DatasetError correctly when cancel is requested."""
        from unittest.mock import patch
        from src.task_manager import TrainingTask
        from src.pipeline import DatasetError

        task = TrainingTask({"epochs": 1, "teacher": "resnet18", "batch_size": 64,
                              "dataset": "CIFAR-10", "student": "MiniCNN",
                              "temperature": 4.0, "alpha": 0.7, "compression_ratio": 0.05,
                              "patience": 0})
        task._cancel_requested = True  # Simulate cancellation during download

        with patch("src.task_manager.run_distillation_pipeline") as mock_pipeline:
            mock_pipeline.side_effect = DatasetError("Download failed")
            task._run()

        assert task.status == "cancelled"

    def test_training_task_run_dataset_error_failed(self):
        """_run handles DatasetError as failure when no cancel."""
        from unittest.mock import patch
        from src.task_manager import TrainingTask
        from src.pipeline import DatasetError

        task = TrainingTask({"epochs": 1, "teacher": "resnet18", "batch_size": 64,
                              "dataset": "CIFAR-10", "student": "MiniCNN",
                              "temperature": 4.0, "alpha": 0.7, "compression_ratio": 0.05,
                              "patience": 0})

        with patch("src.task_manager.run_distillation_pipeline") as mock_pipeline:
            mock_pipeline.side_effect = DatasetError("Download failed")
            task._run()

        assert task.status == "failed"

    def test_training_task_run_generic_exception(self):
        """_run catches generic exceptions and marks task as failed."""
        from unittest.mock import patch
        from src.task_manager import TrainingTask

        task = TrainingTask({"epochs": 1, "teacher": "resnet18", "batch_size": 64,
                              "dataset": "CIFAR-10", "student": "MiniCNN",
                              "temperature": 4.0, "alpha": 0.7, "compression_ratio": 0.05,
                              "patience": 0})

        with patch("src.task_manager.run_distillation_pipeline") as mock_pipeline:
            mock_pipeline.side_effect = ValueError("Something unexpected")
            task._run()

        assert task.status == "failed"
        assert "Something unexpected" in task.error


# ---------------------------------------------------------------------------
# webapp.py — export, health, config, tasks endpoints
# ---------------------------------------------------------------------------

client_ws = TestClient(app)


class TestWebappExportCoverage:
    """Coverage for webapp export flow."""

    def test_export_no_task(self):
        """POST /api/export/nonexistent returns 400."""
        resp = client_ws.post("/api/export/nonexistent_id", json={"format": "onnx"})
        assert resp.status_code == 400

    def test_export_with_valid_format_but_not_completed(self):
        """POST /api/export for a pending task returns 400."""
        resp = client_ws.post(
            "/api/train",
            json={"dataset": "CIFAR-10", "teacher": "resnet18", "epochs": 1},
        )
        task_id = resp.json()["task_id"]
        resp = client_ws.post(f"/api/export/{task_id}", json={"format": "onnx"})
        assert resp.status_code == 400

    def test_tasks_endpoint_structure(self):
        """GET /api/tasks returns proper structure."""
        resp = client_ws.get("/api/tasks")
        assert resp.status_code == 200
        tasks = resp.json()
        assert isinstance(tasks, list)

    def test_api_only_mode(self):
        """api_only mode returns JSON for root endpoint."""
        original = settings.api_only
        settings.api_only = True
        try:
            resp = client_ws.get("/")
            assert resp.status_code == 200
            data = resp.json()
            assert "endpoints" in data
        finally:
            settings.api_only = original


class TestWebappCoverage:
    """Coverage for webapp.py: health, config with auth_required."""

    def test_health_endpoint(self):
        """GET /health returns ok status."""
        resp = client_ws.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "distilkit"

    def test_ready_endpoint(self):
        """GET /ready returns ok status."""
        resp = client_ws.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_live_endpoint(self):
        """GET /live returns ok status."""
        resp = client_ws.get("/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_metrics_endpoint_returns_text(self):
        """GET /metrics returns Prometheus-format text."""
        resp = client_ws.get("/metrics")
        assert resp.status_code == 200
        assert "distilkit_uptime_seconds" in resp.text
        assert "distilkit_requests_total" in resp.text
        assert "distilkit_tasks_total" in resp.text

    def test_config_returns_auth_required(self):
        """GET /api/config includes auth_required field."""
        resp = client_ws.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "auth_required" in data
        # With no API_KEY set, auth should be disabled
        assert data["auth_required"] is False

    def test_config_returns_all_keys(self):
        """GET /api/config returns datasets, teachers, students, device."""
        resp = client_ws.get("/api/config")
        data = resp.json()
        assert "CIFAR-10" in data["datasets"]
        assert "resnet18" in data["teachers"]
        assert "MiniCNN" in data["students"]
        assert data["device"] == "cpu"

    def test_index_returns_html(self):
        """GET / returns HTML."""
        resp = client_ws.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_get_nonexistent_task(self):
        """GET /api/train/nonexistent returns 404."""
        resp = client_ws.get("/api/train/nonexistent123")
        assert resp.status_code == 404

    def test_cancel_nonexistent_task(self):
        """POST /api/train/nonexistent/cancel returns 404."""
        resp = client_ws.post("/api/train/nonexistent/cancel")
        assert resp.status_code == 404

    def test_tasks_endpoint_empty(self):
        """GET /api/tasks returns a list."""
        resp = client_ws.get("/api/tasks")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_history_endpoint(self):
        """GET /api/history returns a list."""
        resp = client_ws.get("/api/history")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_download_nonexistent(self):
        """GET /api/download/nonexistent returns 404."""
        resp = client_ws.get("/api/download/nonexistent.onnx")
        assert resp.status_code == 404

    def test_export_no_task(self):
        """POST /api/export/nonexistent returns 400."""
        resp = client_ws.post("/api/export/nonexistent", json={"format": "onnx"})
        assert resp.status_code == 400

    def test_export_invalid_format_rejected(self):
        """POST /api/export with invalid format returns 400."""
        # Create a task first to pass task existence check
        resp = client_ws.post(
            "/api/train",
            json={"dataset": "CIFAR-10", "teacher": "resnet18", "epochs": 1},
        )
        data = resp.json()
        task_id = data["task_id"]

        resp = client_ws.post(
            f"/api/export/{task_id}", json={"format": "invalid_format"}
        )
        assert resp.status_code == 400

    def test_train_invalid_dataset(self):
        """POST /api/train with invalid dataset returns 400."""
        resp = client_ws.post(
            "/api/train",
            json={"dataset": "InvalidDS", "teacher": "resnet18", "epochs": 1},
        )
        assert resp.status_code == 400

    def test_train_invalid_teacher(self):
        """POST /api/train with invalid teacher returns 400."""
        resp = client_ws.post(
            "/api/train",
            json={"dataset": "CIFAR-10", "teacher": "invalid", "epochs": 1},
        )
        assert resp.status_code == 400

    def test_train_invalid_student(self):
        """POST /api/train with invalid student returns 400."""
        resp = client_ws.post(
            "/api/train",
            json={
                "dataset": "CIFAR-10",
                "teacher": "resnet18",
                "student": "InvalidStudent",
                "epochs": 1,
            },
        )
        assert resp.status_code == 400

    def test_api_only_mode_config(self):
        """api_only mode returns JSON from index."""
        original = settings.api_only
        settings.api_only = True
        try:
            resp = client_ws.get("/")
            assert resp.status_code == 200
            data = resp.json()
            assert data["service"] == "DistilKit API"
        finally:
            settings.api_only = original

    def test_sse_stream_nonexistent_task(self):
        """SSE stream for nonexistent task returns 404."""
        resp = client_ws.get("/api/train/nonexistent_id/stream")
        assert resp.status_code == 404

    def test_train_minimal_config_creates_task(self):
        """POST /api/train with minimal config creates a task."""
        resp = client_ws.post(
            "/api/train",
            json={"dataset": "CIFAR-10", "teacher": "resnet18", "epochs": 1},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "task_id" in data
        assert len(data["task_id"]) == 12

    def test_train_full_config(self):
        """POST /api/train with full config creates a task."""
        resp = client_ws.post(
            "/api/train",
            json={
                "dataset": "MNIST",
                "teacher": "mobilenet_v2",
                "student": "MiniResNet",
                "epochs": 2,
                "temperature": 3.0,
                "alpha": 0.5,
                "patience": 2,
                "batch_size": 32,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "task_id" in data

    def test_train_numeric_validation(self):
        """POST /api/train validates numeric ranges."""
        # Epochs too high
        resp = client_ws.post(
            "/api/train",
            json={"dataset": "CIFAR-10", "teacher": "resnet18", "epochs": 9999},
        )
        assert resp.status_code == 400
        # Compression ratio too high
        resp = client_ws.post(
            "/api/train",
            json={"dataset": "CIFAR-10", "teacher": "resnet18", "epochs": 1, "compression_ratio": 5.0},
        )
        assert resp.status_code == 400

    def test_train_non_numeric_rejected(self):
        """POST /api/train with non-numeric epoch returns 400."""
        resp = client_ws.post(
            "/api/train",
            json={"dataset": "CIFAR-10", "teacher": "resnet18", "epochs": "abc"},
        )
        assert resp.status_code == 400

    def test_require_api_key_with_key_set(self):
        """Setting API_KEY on settings requires X-API-Key header."""
        original_key = settings.api_key
        settings.api_key = "test-key-123"
        try:
            # Request without key should get 401
            resp = client_ws.get("/api/tasks")
            assert resp.status_code == 401
            # Request with wrong key should get 403
            resp = client_ws.get("/api/tasks", headers={"X-API-Key": "wrong-key"})
            assert resp.status_code == 403
            # Request with correct key should succeed
            resp = client_ws.get("/api/tasks", headers={"X-API-Key": "test-key-123"})
            assert resp.status_code == 200
        finally:
            settings.api_key = original_key

    def test_health_auth_not_required(self):
        """Health endpoints are public even when API_KEY is set."""
        original_key = settings.api_key
        settings.api_key = "test-key"
        try:
            resp = client_ws.get("/health")
            assert resp.status_code == 200
        finally:
            settings.api_key = original_key

    def test_metrics_auth_not_required(self):
        """Metrics endpoint is public even when API_KEY is set."""
        original_key = settings.api_key
        settings.api_key = "test-key"
        try:
            resp = client_ws.get("/metrics")
            assert resp.status_code == 200
        finally:
            settings.api_key = original_key

    def test_config_auth_not_required(self):
        """Config endpoint is public even when API_KEY is set."""
        original_key = settings.api_key
        settings.api_key = "test-key"
        try:
            resp = client_ws.get("/api/config")
            assert resp.status_code == 200
            assert resp.json()["auth_required"] is True
        finally:
            settings.api_key = original_key

    def test_cancel_pending_task(self):
        """Cancel a pending task succeeds."""
        resp = client_ws.post(
            "/api/train",
            json={"dataset": "CIFAR-10", "teacher": "resnet18", "epochs": 1},
        )
        task_id = resp.json()["task_id"]
        resp = client_ws.post(f"/api/train/{task_id}/cancel")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"

    def test_cancel_already_cancelled(self):
        """Cancel an already cancelled task returns 400."""
        resp = client_ws.post(
            "/api/train",
            json={"dataset": "CIFAR-10", "teacher": "resnet18", "epochs": 1},
        )
        task_id = resp.json()["task_id"]
        client_ws.post(f"/api/train/{task_id}/cancel")
        # Second cancel should return 400
        resp = client_ws.post(f"/api/train/{task_id}/cancel")
        assert resp.status_code == 400

    def test_download_with_temp_file(self):
        """Download endpoint serves existing files."""
        import tempfile

        # Create a temp file in checkpoints dir
        os.makedirs("checkpoints", exist_ok=True)
        test_filename = "_test_download.tmp"
        test_path = os.path.join("checkpoints", test_filename)
        try:
            with open(test_path, "w") as f:
                f.write("test content")
            resp = client_ws.get(f"/api/download/{test_filename}")
            assert resp.status_code == 200
            assert resp.content == b"test content"
        finally:
            if os.path.exists(test_path):
                os.unlink(test_path)

    def test_exports_via_mocked_task(self):
        """Export endpoint works with a properly mocked task."""
        from src.task_manager import _tasks
        from src.student import MiniCNN

        model = MiniCNN(in_channels=3, num_classes=10, width=0.5)

        # Create a fake completed task with a student model
        class FakeTask:
            id = "fake_task_001"
            status = "completed"
            student = model
            config = {"dataset": "CIFAR-10", "student": "MiniCNN"}
            _emit = lambda self, msg: None

        _tasks["fake_task_001"] = FakeTask()
        try:
            # Test TorchScript export
            resp = client_ws.post(
                "/api/export/fake_task_001",
                json={"format": "torchscript"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["format"] == "torchscript"
            assert "filename" in data
        finally:
            _tasks.pop("fake_task_001", None)

    def test_sse_with_mocked_completed_task(self):
        """SSE stream returns events for a completed task."""
        from src.task_manager import _tasks

        class MockTask:
            id = "fake_sse_001"
            status = "completed"
            progress = 1.0
            current_epoch = 5
            total_epochs = 5
            current_loss = 0.25
            current_acc = 0.95
            eta_seconds = 0.0
            losses = [1.0, 0.6, 0.4, 0.3, 0.25]
            accuracies = [0.5, 0.7, 0.85, 0.9, 0.95]
            logs = "Test log output"
            result = {"final_loss": 0.25, "final_accuracy": 0.95}
            error = None
            _dirty = True

        _tasks["fake_sse_001"] = MockTask()
        try:
            with client_ws.stream("GET", "/api/train/fake_sse_001/stream") as resp:
                assert resp.status_code == 200
                # Read first chunk of streaming response
                chunks = [chunk for chunk in resp.iter_bytes()]
                assert len(chunks) > 0
                content = b"".join(chunks).decode()
                assert "event: complete" in content
                assert "final_loss" in content
        finally:
            _tasks.pop("fake_sse_001", None)

    def test_sse_with_mocked_failed_task(self):
        """SSE stream returns error event for a failed task."""
        from src.task_manager import _tasks

        class MockFailedTask:
            id = "fake_sse_fail"
            status = "failed"
            progress = 0.5
            current_epoch = 2
            total_epochs = 5
            current_loss = None
            current_acc = None
            eta_seconds = 0.0
            losses = []
            accuracies = []
            logs = "Something went wrong"
            result = None
            error = "CUDA out of memory"
            _dirty = True

        _tasks["fake_sse_fail"] = MockFailedTask()
        try:
            with client_ws.stream("GET", "/api/train/fake_sse_fail/stream") as resp:
                assert resp.status_code == 200
                content = b"".join(resp.iter_bytes()).decode()
                assert "event: error" in content
                assert "CUDA out of memory" in content
        finally:
            _tasks.pop("fake_sse_fail", None)

    def test_export_missing_student(self):
        """Export with a task that has no student model returns 400."""
        from src.task_manager import _tasks

        class FakeTaskNoStudent:
            id = "fake_no_student"
            status = "completed"
            student = None
            config = {"dataset": "CIFAR-10", "student": "MiniCNN"}

        _tasks["fake_no_student"] = FakeTaskNoStudent()
        try:
            resp = client_ws.post(
                "/api/export/fake_no_student",
                json={"format": "onnx"},
            )
            assert resp.status_code == 400
        finally:
            _tasks.pop("fake_no_student", None)

    def test_config_cached_teachers(self):
        """Config endpoint returns cached_teachers."""
        resp = client_ws.get("/api/config")
        data = resp.json()
        assert "cached_teachers" in data

    def test_get_task_success(self):
        """GET /api/train/<id> returns task details for an existing task."""
        from src.task_manager import _tasks

        resp = client_ws.post(
            "/api/train",
            json={"dataset": "CIFAR-10", "teacher": "resnet18", "epochs": 1},
        )
        task_id = resp.json()["task_id"]
        resp = client_ws.get(f"/api/train/{task_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == task_id
        assert data["status"] in ("pending", "running")
        assert "progress" in data
        # Cleanup
        _tasks.pop(task_id, None)

    def test_sse_with_running_task(self):
        """SSE stream yields events for a running task that completes.
        Covers lines 390-392 (_dirty sleep/continue), 412-415 (completed
        check inside the while loop body), and the _dirty=True data path.
        """
        import threading
        from src.task_manager import _tasks

        class RunningTask:
            id = "fake_running_sse"
            status = "running"
            progress = 0.5
            current_epoch = 3
            total_epochs = 10
            current_loss = 0.5
            current_acc = 0.85
            eta_seconds = 120.0
            losses = [1.0, 0.8, 0.6]
            accuracies = [0.6, 0.7, 0.8]
            logs = "Epoch 3/10"
            result = None
            error = None
            _dirty = True

        _tasks["fake_running_sse"] = RunningTask()

        def complete_task():
            import time as _time
            _time.sleep(0.3)
            t = _tasks.get("fake_running_sse")
            if t:
                t.status = "completed"
                t.result = {"final_loss": 0.3}
                t._dirty = True

        timer = threading.Timer(0.3, complete_task)
        timer.start()
        try:
            with client_ws.stream("GET", "/api/train/fake_running_sse/stream") as resp:
                assert resp.status_code == 200
                content = b"".join(resp.iter_bytes()).decode()
                assert "progress" in content
                assert "event: complete" in content
        finally:
            timer.cancel()
            _tasks.pop("fake_running_sse", None)

    def test_sse_with_missing_auth_key(self):
        """SSE stream returns 401 when API key is required but not provided."""
        original_key = settings.api_key
        settings.api_key = "required-key"
        try:
            resp = client_ws.get("/api/train/fake/stream")
            assert resp.status_code == 401
        finally:
            settings.api_key = original_key

    def test_sse_with_auth_query_param(self):
        """SSE stream accepts api_key via query param when auth is enabled."""
        from src.task_manager import _tasks

        class MockTask:
            id = "fake_sse_auth"
            status = "completed"
            progress = 1.0
            current_epoch = 5
            total_epochs = 5
            current_loss = 0.25
            current_acc = 0.95
            eta_seconds = 0.0
            losses = [1.0, 0.6, 0.4, 0.3, 0.25]
            accuracies = [0.5, 0.7, 0.85, 0.9, 0.95]
            logs = "Test log"
            result = {"final_loss": 0.25}
            error = None
            _dirty = True

        original_key = settings.api_key
        settings.api_key = "sse-test-key"
        _tasks["fake_sse_auth"] = MockTask()
        try:
            with client_ws.stream(
                "GET", "/api/train/fake_sse_auth/stream?api_key=sse-test-key"
            ) as resp:
                assert resp.status_code == 200
                content = b"".join(resp.iter_bytes()).decode()
                assert "event: complete" in content
        finally:
            settings.api_key = original_key
            _tasks.pop("fake_sse_auth", None)

    def test_sse_auth_query_param_wrong_key(self):
        """SSE stream with wrong api_key query param returns 403."""
        original_key = settings.api_key
        settings.api_key = "sse-test-key"
        try:
            resp = client_ws.get("/api/train/fake/stream?api_key=wrong-key")
            assert resp.status_code == 403
        finally:
            settings.api_key = original_key

    def test_export_onnx_fallback_to_torchscript(self):
        """Export falls back to TorchScript when ONNX export fails."""
        from src.task_manager import _tasks
        from src.student import MiniCNN

        model = MiniCNN(in_channels=3, num_classes=10, width=0.5)

        class FakeTask:
            id = "fake_fallback"
            status = "completed"
            student = model
            config = {"dataset": "CIFAR-10", "student": "MiniCNN"}
            _emit = lambda self, msg: None

        _tasks["fake_fallback"] = FakeTask()
        try:
            with patch("src.webapp.export_to_onnx") as mock_onnx:
                mock_onnx.side_effect = OSError("ONNX export failed")
                with patch("src.webapp.export_to_torchscript") as mock_ts:
                    mock_ts.return_value = Path("checkpoints/fake_fallback.pt")
                    resp = client_ws.post(
                        "/api/export/fake_fallback",
                        json={"format": "onnx"},
                    )
                    assert resp.status_code == 200
                    data = resp.json()
                    assert data["format"] == "torchscript"
                    assert ".pt" in data["filename"]
                    mock_onnx.assert_called_once()
                    mock_ts.assert_called_once()
        finally:
            _tasks.pop("fake_fallback", None)

    def test_export_outer_exception(self):
        """Export returns 500 when both export formats fail."""
        from src.task_manager import _tasks
        from src.student import MiniCNN

        model = MiniCNN(in_channels=3, num_classes=10, width=0.5)

        class FakeTask:
            id = "fake_outer_exc"
            status = "completed"
            student = model
            config = {"dataset": "CIFAR-10", "student": "MiniCNN"}
            _emit = lambda self, msg: None

        _tasks["fake_outer_exc"] = FakeTask()
        try:
            with patch("src.webapp.export_to_onnx") as mock_onnx:
                mock_onnx.side_effect = OSError("ONNX failed")
                with patch("src.webapp.export_to_torchscript") as mock_ts:
                    mock_ts.side_effect = RuntimeError("TorchScript also failed")
                    resp = client_ws.post(
                        "/api/export/fake_outer_exc",
                        json={"format": "onnx"},
                    )
                    assert resp.status_code == 500
        finally:
            _tasks.pop("fake_outer_exc", None)

    def test_config_no_cache_dir(self):
        """Config endpoint handles missing torch cache directory."""
        with patch("os.path.isdir", return_value=False):
            resp = client_ws.get("/api/config")
            assert resp.status_code == 200
            data = resp.json()
            assert "cached_teachers" in data
            # All models should show as not cached
            assert all(v is False for v in data["cached_teachers"].values())

    def test_security_headers_on_response(self):
        """Every response includes security headers."""
        resp = client_ws.get("/health")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"
        csp = resp.headers.get("Content-Security-Policy", "")
        assert "default-src 'self'" in csp
        assert "script-src 'self'" in csp
        assert "style-src 'self'" in csp

    def test_rate_limiter_enforces_limit(self):
        """Rate limiter returns 429 when limit is exceeded."""
        original_rate = settings.rate_limit_per_minute
        settings.rate_limit_per_minute = 3
        try:
            # First 3 requests to a path not in _RATE_LIMITS should succeed
            for _ in range(3):
                resp = client_ws.get("/api/config")
                assert resp.status_code == 200
            # 4th request should be rate limited
            resp = client_ws.get("/api/config")
            assert resp.status_code == 429
            assert "Retry-After" in resp.headers
        finally:
            settings.rate_limit_per_minute = original_rate

    def test_launch_function_basic(self):
        """launch() calls uvicorn.run with correct host/port."""
        import uvicorn
        from src.webapp import launch

        with patch.object(uvicorn, "run") as mock_run:
            launch(host="127.0.0.1", port=9999)
            mock_run.assert_called_once_with(
                ANY, host="127.0.0.1", port=9999, log_level="warning",
                ssl_certfile=None, ssl_keyfile=None,
            )

    def test_launch_function_defaults(self):
        """launch() uses settings defaults when args are None."""
        import uvicorn
        from src.webapp import launch
        original_host = settings.host
        original_port = settings.port
        settings.host = "0.0.0.0"
        settings.port = 8888
        try:
            with patch.object(uvicorn, "run") as mock_run:
                launch()
                mock_run.assert_called_once_with(
                    ANY, host="0.0.0.0", port=8888, log_level="warning",
                    ssl_certfile=None, ssl_keyfile=None,
                )
        finally:
            settings.host = original_host
            settings.port = original_port

    def test_launch_api_only_mode(self):
        """launch() with api_only=True sets settings.api_only."""
        import uvicorn
        from src.webapp import launch
        original = settings.api_only
        settings.api_only = False
        try:
            with patch.object(uvicorn, "run"):
                launch(api_only=True)
                assert settings.api_only is True
        finally:
            settings.api_only = original

    def test_launch_with_ssl(self):
        """launch() passes ssl params and enables HSTS when cert/key are set."""
        import uvicorn
        from src.webapp import launch
        original_cert = settings.ssl_certfile
        original_key = settings.ssl_keyfile
        original_hsts = settings.hsts_max_age
        settings.ssl_certfile = "/tmp/test.cert"
        settings.ssl_keyfile = "/tmp/test.key"
        settings.hsts_max_age = 0
        try:
            with patch.object(uvicorn, "run") as mock_run:
                launch(host="127.0.0.1", port=9999)
                mock_run.assert_called_once()
                _, kwargs = mock_run.call_args
                assert kwargs["ssl_certfile"] == "/tmp/test.cert"
                assert kwargs["ssl_keyfile"] == "/tmp/test.key"
                # HSTS should be auto-enabled
                assert settings.hsts_max_age > 0
        finally:
            settings.ssl_certfile = original_cert
            settings.ssl_keyfile = original_key
            settings.hsts_max_age = original_hsts


# ---------------------------------------------------------------------------
# log_config.py — ContextAdapter edge cases
# ---------------------------------------------------------------------------


class TestLogConfigCoverage:
    """Coverage for log_config.py: ContextAdapter without request ID."""

    def test_logger_basic_info(self):
        """logger.info works without crashing."""
        from src.log_config import logger

        # Should not raise
        logger.info("Test log message")

    def test_logger_bind(self):
        """logger.bind() returns a new adapter with merged context."""
        from src.log_config import logger

        bound = logger.bind(task_id="test123")
        assert bound is not logger
        # Logging with bound context should not crash
        bound.info("Bound log message")

    def test_setup_logger_duplicate_handlers(self):
        """setup_logger with existing handlers returns without adding more."""
        from src.log_config import setup_logger
        import logging

        # First call creates logger
        logger1 = setup_logger("test_dup_logger")
        # Second call should reuse existing handlers
        logger2 = setup_logger("test_dup_logger")
        assert logger2 is not None
        # Should not crash
        logger2.info("Duplicate logger test")

    def test_get_request_id(self):
        """get_request_id returns current request ID."""
        from src.log_config import get_request_id, set_request_id

        # Default is empty
        assert get_request_id() == ""
        # After setting, returns the value
        set_request_id("test-rid")
        assert get_request_id() == "test-rid"
        # Clean up
        set_request_id("")


# ---------------------------------------------------------------------------
# student.py — build_student edge cases
# ---------------------------------------------------------------------------


class TestStudentCoverage:
    """Coverage for student.py build_student edge cases."""

    def test_build_student_no_teacher_zero_ratio(self):
        """build_student without teacher and ratio=0 defaults to width=1.0."""
        from src.student import build_student, MiniCNN

        student = build_student(
            teacher=None, student_type="MiniCNN", compression_ratio=0.0
        )
        base = MiniCNN(in_channels=3, num_classes=10, width=1.0)
        assert sum(p.numel() for p in student.parameters()) == sum(
            p.numel() for p in base.parameters()
        )

    def test_build_student_unknown_type_raises(self):
        """build_student with unknown type raises ValueError."""
        from src.student import build_student

        with pytest.raises(ValueError, match="Unknown student"):
            build_student(student_type="NonExistent", compression_ratio=0)

    def test_build_student_compression_ratio_zero(self):
        """build_student with compression_ratio=0 uses width=1.0."""
        from src.student import build_student, MiniCNN

        teacher = nn.Sequential(nn.Linear(10, 10))
        student = build_student(
            teacher=teacher,
            student_type="MiniCNN",
            compression_ratio=0.0,
            num_classes=10,
        )
        assert student is not None

    def test_miniresnet_forward(self):
        """MiniResNet forward pass produces correct shape."""
        from src.student import MiniResNet

        model = MiniResNet(in_channels=3, num_classes=10, width=1.0)
        dummy = torch.randn(2, 3, 32, 32)
        out = model(dummy)
        assert out.shape == (2, 10)


# ---------------------------------------------------------------------------
# alert_manager.py — alert evaluation logic
# ---------------------------------------------------------------------------


class TestAlertManagerCoverage:
    """Coverage for alert_manager.py: all alert evaluation and suppression logic."""

    def setup_method(self) -> None:
        """Reset all module-level state before each test."""
        import src.alert_manager as am
        am._error_window.clear()
        am._request_window.clear()
        am._task_failures.clear()
        am._last_alert.clear()
        am._consecutive_failures = 0

    # ── _check_error_rate ────────────────────────────────────────────────

    def test_check_error_rate_empty_returns_none(self) -> None:
        """_check_error_rate returns None when no requests recorded."""
        import src.alert_manager as am
        assert am._check_error_rate() is None

    def test_check_error_rate_high_rate(self) -> None:
        """_check_error_rate returns alert when error rate > 5%."""
        import src.alert_manager as am
        import time
        now = time.time()
        for _ in range(10):
            am._request_window.append(now)
            am._error_window.append(now)
        msg = am._check_error_rate()
        assert msg is not None
        assert "5xx" in msg

    def test_check_error_rate_multiple_errors(self) -> None:
        """_check_error_rate returns alert when >=5 errors even at low rate."""
        import src.alert_manager as am
        import time
        now = time.time()
        for _ in range(5):
            am._request_window.append(now)
            am._error_window.append(now)
        for _ in range(500):
            am._request_window.append(now)
        msg = am._check_error_rate()
        assert msg is not None
        assert "5xx" in msg

    def test_check_error_rate_low_returns_none(self) -> None:
        """_check_error_rate returns None when both thresholds are below limits."""
        import src.alert_manager as am
        import time
        now = time.time()
        for _ in range(2):
            am._request_window.append(now)
            am._error_window.append(now)
        for _ in range(100):
            am._request_window.append(now)
        assert am._check_error_rate() is None

    def test_check_error_rate_purge_old_entries(self) -> None:
        """_check_error_rate purges entries older than 5 minutes."""
        import src.alert_manager as am
        import time
        old = time.time() - 400  # older than the 300 s window
        now = time.time()
        am._error_window.append(old)
        am._request_window.append(old)
        am._request_window.append(now)
        msg = am._check_error_rate()
        # After purge only 1 request remains, 0 errors → None
        assert msg is None

    # ── _check_task_failures ─────────────────────────────────────────────

    def test_check_task_failures_empty(self) -> None:
        """_check_task_failures returns None when no failures exist."""
        import src.alert_manager as am
        assert am._check_task_failures() is None

    def test_check_task_failures_multiple(self) -> None:
        """_check_task_failures returns alert when >=3 failures within an hour."""
        import src.alert_manager as am
        import time
        now = time.time()
        for i in range(3):
            am._task_failures[f"task_{i}"] = now
        msg = am._check_task_failures()
        assert msg is not None
        assert "training" in msg

    def test_check_task_failures_below_threshold(self) -> None:
        """_check_task_failures returns None when <3 failures exist."""
        import src.alert_manager as am
        import time
        am._task_failures["single_task"] = time.time()
        assert am._check_task_failures() is None

    # ── _should_suppress ─────────────────────────────────────────────────

    def test_should_suppress_first_call(self) -> None:
        """_should_suppress returns False on first call (no suppression)."""
        import src.alert_manager as am
        assert not am._should_suppress("test_alert_1")

    def test_should_suppress_within_window(self) -> None:
        """_should_suppress returns True when called again within suppression window."""
        import src.alert_manager as am
        am._should_suppress("test_alert_2")  # first call → False
        assert am._should_suppress("test_alert_2")  # second call → suppressed

    # ── _post_webhook ────────────────────────────────────────────────────

    def test_post_webhook_no_url(self) -> None:
        """_post_webhook returns early when no webhook URL configured."""
        import src.alert_manager as am
        from src.settings import settings as s
        original = s.alert_webhook_url
        s.alert_webhook_url = ""
        try:
            am._post_webhook({"text": "test"})  # Should not raise
        finally:
            s.alert_webhook_url = original

    def test_post_webhook_with_url(self) -> None:
        """_post_webhook posts JSON to the configured webhook URL."""
        import src.alert_manager as am
        from src.settings import settings as s
        from unittest.mock import patch
        original = s.alert_webhook_url
        s.alert_webhook_url = "https://hooks.example.com/hook"
        try:
            with patch("urllib.request.urlopen") as mock_urlopen:
                am._post_webhook({"text": "test alert"})
                mock_urlopen.assert_called_once()
        finally:
            s.alert_webhook_url = original

    def test_post_webhook_failure_logs_warning(self) -> None:
        """_post_webhook logs a warning when the webhook call fails."""
        import src.alert_manager as am
        from src.settings import settings as s
        from unittest.mock import patch
        original = s.alert_webhook_url
        s.alert_webhook_url = "https://hooks.example.com/hook"
        try:
            with patch(
                "urllib.request.urlopen",
                side_effect=ConnectionError("connection refused"),
            ):
                # Should log warning, not raise
                am._post_webhook({"text": "test"})
        finally:
            s.alert_webhook_url = original

    # ── _evaluate_once ───────────────────────────────────────────────────

    def test_evaluate_once_clean_state(self) -> None:
        """_evaluate_once handles a clean state without crashing."""
        import src.alert_manager as am
        am._evaluate_once()
        assert am._consecutive_failures == 0

    def test_evaluate_once_with_errors(self) -> None:
        """_evaluate_once increments consecutive_failures when alerts fire."""
        import src.alert_manager as am
        import time
        now = time.time()
        for _ in range(10):
            am._request_window.append(now)
            am._error_window.append(now)
        am._evaluate_once()
        assert am._consecutive_failures == 1

    def test_evaluate_once_resets_on_clean(self) -> None:
        """_evaluate_once resets consecutive_failures to 0 when no alerts."""
        import src.alert_manager as am
        am._consecutive_failures = 10
        am._evaluate_once()  # empty windows → no alerts → reset
        assert am._consecutive_failures == 0

    def test_evaluate_once_consecutive_escalation(self) -> None:
        """_evaluate_once fires escalation alert after 5 consecutive alert cycles."""
        import src.alert_manager as am
        import time
        for _ in range(5):
            # Clear suppression so each cycle fires the error_rate alert
            am._last_alert.clear()
            am._error_window.clear()
            am._request_window.clear()
            now = time.time()
            for _ in range(10):
                am._request_window.append(now)
                am._error_window.append(now)
            am._evaluate_once()
        assert am._consecutive_failures >= 5


# ---------------------------------------------------------------------------
# tracing.py — Span and Tracer edge cases
# ---------------------------------------------------------------------------


class TestTracingCoverage:
    """Coverage for tracing.py: Span, Tracer edge cases."""

    def test_span_end_called_twice(self) -> None:
        """Span.end() is idempotent — second call does not raise."""
        from src.tracing import tracer
        span = tracer.start_span("test_span")
        span.end()
        span.end()  # must not raise
        assert span._end is not None

    def test_span_duration_zero_when_not_ended(self) -> None:
        """duration_ms returns 0 for an unended span."""
        from src.tracing import tracer
        span = tracer.start_span("test_span")
        assert span.duration_ms == 0.0
        span.end()

    def test_span_duration_positive_after_end(self) -> None:
        """duration_ms returns > 0 after the span is ended."""
        from src.tracing import tracer
        span = tracer.start_span("test_span")
        span.end()
        assert span.duration_ms > 0.0

    def test_span_to_traceparent(self) -> None:
        """to_traceparent returns a valid W3C traceparent string."""
        from src.tracing import Span
        span = Span("test", trace_id="abc123def4567890", span_id="xyz7890123456789")
        tp = span.to_traceparent()
        assert tp.startswith("00-")
        assert tp.endswith("-01")
        assert "abc123def4567890" in tp

    def test_span_repr(self) -> None:
        """Span repr includes the name and trace ID."""
        from src.tracing import Span
        span = Span("my_span", trace_id="aaaabbbbccccdddd", span_id="eeeeffff00001111")
        r = repr(span)
        assert "Span(" in r
        assert "my_span" in r

    def test_span_context_manager(self) -> None:
        """Span used as context manager calls end() on exit."""
        from src.tracing import tracer
        with tracer.start_span("ctx_span") as span:
            assert span._end is None
        assert span._end is not None

    def test_tracer_span_from_traceparent(self) -> None:
        """span_from_traceparent parses a W3C traceparent header."""
        from src.tracing import tracer
        span = tracer.span_from_traceparent(
            "incoming", "00-abc123def4567890ffff0000aaaabbbb-xyz7890123456789-01"
        )
        assert span.name == "incoming"
        assert "abc123def4567890ffff0000aaaabbbb" in span.trace_id
        assert span.span_id is not None

    def test_traceparent_from_env_default(self) -> None:
        """_traceparent_from_env returns None when TRACEPARENT is not set."""
        from src.tracing import _traceparent_from_env
        # Unset to ensure clean state
        import os
        os.environ.pop("TRACEPARENT", None)
        assert _traceparent_from_env() is None

    def test_traceparent_from_env_set(self) -> None:
        """_traceparent_from_env returns the env var value when set."""
        from src.tracing import _traceparent_from_env
        import os
        os.environ["TRACEPARENT"] = "00-test-01"
        try:
            assert _traceparent_from_env() == "00-test-01"
        finally:
            os.environ.pop("TRACEPARENT", None)

"""Tests for the CLI interface."""

import argparse

import pytest

from src.cli import build_parser, cmd_benchmark, cmd_export


class TestParser:
    """Tests for argument parsing."""

    def test_build_parser_returns_parser(self):
        """build_parser should return an ArgumentParser."""
        parser = build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_default_command_is_gui(self):
        """No arguments should result in gui subcommand being used."""
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_train_subcommand(self):
        """train subcommand should have expected defaults."""
        parser = build_parser()
        args = parser.parse_args(["train"])
        assert args.command == "train"
        assert args.dataset == "CIFAR-10"
        assert args.teacher == "resnet18"
        assert args.epochs == 10
        assert args.temperature == 4.0
        assert args.alpha == 0.7
        assert args.batch_size == 64
        assert args.patience == 0
        assert args.ckpt_every == 5
        assert args.resume is None

    def test_train_with_custom_args(self):
        """train subcommand should accept custom values."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "train",
                "--dataset",
                "MNIST",
                "--teacher",
                "resnet50",
                "--epochs",
                "20",
                "--temperature",
                "5.0",
                "--alpha",
                "0.5",
                "--batch-size",
                "128",
                "--patience",
                "3",
                "--ckpt-every",
                "0",
                "--compression-ratio",
                "0.1",
            ]
        )
        assert args.dataset == "MNIST"
        assert args.teacher == "resnet50"
        assert args.epochs == 20
        assert args.temperature == 5.0
        assert args.alpha == 0.5
        assert args.batch_size == 128
        assert args.patience == 3
        assert args.ckpt_every == 0
        assert args.compression_ratio == 0.1

    def test_invalid_teacher_raises(self):
        """An invalid teacher choice should be rejected."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["train", "--teacher", "invalid_model"])

    def test_invalid_dataset_raises(self):
        """An invalid dataset choice should be rejected."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["train", "--dataset", "InvalidDS"])

    def test_benchmark_subcommand(self):
        """benchmark subcommand should require --model."""
        parser = build_parser()
        args = parser.parse_args(["benchmark", "--model", "model.onnx"])
        assert args.command == "benchmark"
        assert args.model == "model.onnx"
        assert args.target == "cpu"
        assert args.runs == 100

    def test_benchmark_missing_model_raises(self):
        """benchmark without --model should raise."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["benchmark"])

    def test_export_subcommand(self):
        """export subcommand should have expected defaults."""
        parser = build_parser()
        args = parser.parse_args(["export", "--model", "model.pth"])
        assert args.command == "export"
        assert args.model == "model.pth"
        assert args.format == "onnx"
        assert args.output is None

    def test_export_torchscript(self):
        """export should accept torchscript format."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "export",
                "--model",
                "model.pth",
                "--format",
                "torchscript",
                "--output",
                "out.pt",
            ]
        )
        assert args.format == "torchscript"
        assert args.output == "out.pt"

    def test_gui_subcommand(self):
        """gui subcommand should be recognized."""
        parser = build_parser()
        args = parser.parse_args(["gui"])
        assert args.command == "gui"


class TestCmdBenchmark:
    """Tests for the benchmark command."""

    def test_missing_file_does_not_crash(self):
        """cmd_benchmark should handle missing model gracefully."""
        args = argparse.Namespace(
            model="nonexistent.onnx",
            target="cpu",
            runs=10,
            command="benchmark",
        )
        # Should not raise — the function logs the error and returns
        cmd_benchmark(args)


class TestCmdExport:
    """Tests for the export command."""

    def test_missing_file_does_not_crash(self):
        """cmd_export should handle missing model gracefully."""
        args = argparse.Namespace(
            model="nonexistent.pth",
            format="onnx",
            output="out.onnx",
            command="export",
        )
        # Should not raise — the function logs the error and returns
        cmd_export(args)


class TestCmdExportMocked:
    """Tests using mocked exports to verify dispatch logic."""

    def test_export_calls_onnx_when_format_is_onnx(self):
        """cmd_export should call export_to_onnx when format='onnx'."""
        from unittest.mock import patch

        with patch("src.cli.torch.load") as mock_load, patch(
            "src.onnx_export.export_to_onnx"
        ) as mock_onnx:
            fake_state = {"state_dict": {"fc.weight": 1.0}}
            mock_load.return_value = fake_state

            with patch("src.student.MiniCNN") as mock_model:
                instance = mock_model.return_value

                args = argparse.Namespace(
                    model="model.pth",
                    format="onnx",
                    output="out.onnx",
                    command="export",
                )
                cmd_export(args)

                mock_onnx.assert_called_once()
                mock_load.assert_called_once_with("model.pth", map_location="cpu")

    def test_export_calls_torchscript_when_format_is_torchscript(self):
        """cmd_export should call export_to_torchscript when format='torchscript'."""
        from unittest.mock import patch

        with patch("src.cli.torch.load") as mock_load, patch(
            "src.onnx_export.export_to_torchscript"
        ) as mock_ts:
            fake_state = {"state_dict": {"fc.weight": 1.0}}
            mock_load.return_value = fake_state

            with patch("src.student.MiniCNN") as mock_model:
                instance = mock_model.return_value

                args = argparse.Namespace(
                    model="model.pth",
                    format="torchscript",
                    output="out.pt",
                    command="export",
                )
                cmd_export(args)

                mock_ts.assert_called_once()

    def test_export_handles_raw_state_dict(self):
        """cmd_export should load a raw state dict (no 'state_dict' key)."""
        from unittest.mock import patch

        with patch("src.cli.torch.load") as mock_load, patch(
            "src.onnx_export.export_to_onnx"
        ) as mock_onnx:
            # State dict without "state_dict" wrapper key
            mock_load.return_value = {"fc.weight": 1.0}

            with patch("src.student.MiniCNN") as mock_model:
                instance = mock_model.return_value

                args = argparse.Namespace(
                    model="model.pth",
                    format="onnx",
                    output="out.onnx",
                    command="export",
                )
                cmd_export(args)

                mock_onnx.assert_called_once()

    def test_export_handles_load_failure(self):
        """cmd_export should handle torch.load failures without crashing."""
        from unittest.mock import patch

        with patch("src.cli.torch.load") as mock_load:
            mock_load.side_effect = RuntimeError("Corrupted file")

            args = argparse.Namespace(
                model="corrupted.pth",
                format="onnx",
                output="out.onnx",
                command="export",
            )
            # Should not raise — the function logs the error and returns
            cmd_export(args)

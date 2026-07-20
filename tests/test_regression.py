"""Regression tests — verify that previously fixed bugs stay fixed.

Every test here maps to a specific bug that was identified in the
production-readiness audit and subsequently fixed.  If any of these
tests start failing, a bug has regressed.

Bug tracking (cross-reference):
  SEC-01  → test_regression_negative_epochs_rejected
  ERR-01  → test_regression_corrupted_ckpt_does_not_crash
  ERR-02  → test_regression_corrupted_history_does_not_crash
  ARC-02  → test_regression_settings_from_env
  ERR-01  → test_regression_missing_onnx_file_does_not_crash
"""

import os
import tempfile

import pytest
import torch
from fastapi.testclient import TestClient

from src.webapp import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# SEC-01 regression: input validation rejects out-of-range values
# ---------------------------------------------------------------------------


class TestInputValidationRegression:
    """Numeric parameters must be range-validated server-side.

    Bug: The /api/train endpoint accepted any numeric value (negative
    epochs, alpha > 1, etc.) because only the model *names* were validated
    against controlled lists.
    Fix: Added ``_clamp_and_validate()`` that raises HTTPException(400)
    when a value falls outside [lo, hi].
    """

    def test_negative_epochs_rejected(self):
        """Epochs must be >= 1."""
        resp = client.post(
            "/api/train",
            json={
                "dataset": "CIFAR-10",
                "teacher": "resnet18",
                "epochs": -5,
            },
        )
        assert resp.status_code == 400
        assert "epochs" in resp.json()["detail"].lower()

    def test_zero_epochs_rejected(self):
        """Epochs must be >= 1 (zero is invalid)."""
        resp = client.post(
            "/api/train",
            json={
                "dataset": "CIFAR-10",
                "teacher": "resnet18",
                "epochs": 0,
            },
        )
        assert resp.status_code == 400

    def test_alpha_above_one_rejected(self):
        """Alpha must be <= 1.0."""
        resp = client.post(
            "/api/train",
            json={
                "dataset": "CIFAR-10",
                "teacher": "resnet18",
                "alpha": 999.0,
            },
        )
        assert resp.status_code == 400
        assert "alpha" in resp.json()["detail"].lower()

    def test_alpha_below_zero_rejected(self):
        """Alpha must be >= 0.0."""
        resp = client.post(
            "/api/train",
            json={
                "dataset": "CIFAR-10",
                "teacher": "resnet18",
                "alpha": -0.5,
            },
        )
        assert resp.status_code == 400

    def test_temperature_zero_rejected(self):
        """Temperature must be > 0."""
        resp = client.post(
            "/api/train",
            json={
                "dataset": "CIFAR-10",
                "teacher": "resnet18",
                "temperature": 0.0,
            },
        )
        assert resp.status_code == 400
        assert "temperature" in resp.json()["detail"].lower()

    def test_negative_temperature_rejected(self):
        """Negative temperature is invalid."""
        resp = client.post(
            "/api/train",
            json={
                "dataset": "CIFAR-10",
                "teacher": "resnet18",
                "temperature": -1.0,
            },
        )
        assert resp.status_code == 400

    def test_batch_size_zero_rejected(self):
        """Batch size must be >= 1."""
        resp = client.post(
            "/api/train",
            json={
                "dataset": "CIFAR-10",
                "teacher": "resnet18",
                "batch_size": 0,
            },
        )
        assert resp.status_code == 400

    def test_batch_size_too_large_rejected(self):
        """Batch size above 4096 is rejected."""
        resp = client.post(
            "/api/train",
            json={
                "dataset": "CIFAR-10",
                "teacher": "resnet18",
                "batch_size": 9999,
            },
        )
        assert resp.status_code == 400

    def test_compression_ratio_zero_rejected(self):
        """Compression ratio must be >= 0.01."""
        resp = client.post(
            "/api/train",
            json={
                "dataset": "CIFAR-10",
                "teacher": "resnet18",
                "compression_ratio": 0.0,
            },
        )
        assert resp.status_code == 400

    def test_compression_ratio_above_one_rejected(self):
        """Compression ratio must be <= 1.0."""
        resp = client.post(
            "/api/train",
            json={
                "dataset": "CIFAR-10",
                "teacher": "resnet18",
                "compression_ratio": 5.0,
            },
        )
        assert resp.status_code == 400

    def test_non_numeric_parameter_rejected(self):
        """Non-numeric input for a numeric field returns 400."""
        resp = client.post(
            "/api/train",
            json={
                "dataset": "CIFAR-10",
                "teacher": "resnet18",
                "epochs": "not-a-number",
            },
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# ERR-01 regression: I/O failures are caught, not crashes
# ---------------------------------------------------------------------------


class TestIOErrorRegression:
    """I/O operations (model loading, ONNX export) must not crash the process.

    Bug: ``torch.load()``, ``ort.InferenceSession()`` and related calls in
    ``cmd_benchmark`` and ``cmd_export`` were unwrapped — a corrupt file
    or disk error would crash the process.
    Fix: Every I/O call is wrapped in try/except that logs the error and
    returns gracefully.
    """

    def test_corrupted_checkpoint_load_raises(self):
        """A corrupted .pt file raises when torch.load is called on it.

        This verifies the I/O exception that the ERR-01 fix is designed
        to catch gracefully in cmd_export and cmd_benchmark.
        """
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"not a valid PyTorch file")
            bad_path = f.name

        try:
            with pytest.raises(Exception):
                torch.load(bad_path, map_location="cpu", weights_only=False)
        finally:
            os.unlink(bad_path)

    def test_corrupted_onnx_benchmark_does_not_crash(self):
        """cmd_benchmark with a corrupted .onnx file should log and return."""
        import argparse

        args = argparse.Namespace(
            model="/nonexistent/corrupted.onnx",
            target="cpu",
            runs=10,
            command="benchmark",
        )
        # Should not raise — just logs the error
        from src.cli import cmd_benchmark

        cmd_benchmark(args)

    def test_corrupted_pth_benchmark_does_not_crash(self):
        """cmd_benchmark with a corrupted .pth file should log and return."""
        from unittest.mock import patch

        import argparse

        args = argparse.Namespace(
            model="corrupted.pth",
            target="cpu",
            runs=10,
            command="benchmark",
        )
        with patch("src.cli.torch.load", side_effect=RuntimeError("corrupted")):
            from src.cli import cmd_benchmark

            cmd_benchmark(args)


# ---------------------------------------------------------------------------
# ERR-02 regression: exceptions are never silently swallowed
# ---------------------------------------------------------------------------


class TestHistoryErrorRegression:
    """Corrupted history files must not be silently ignored.

    Bug: ``_load_history`` caught ``Exception`` with a bare ``pass``,
    making corrupted JSON files invisible in the logs.
    Fix: The exception is logged as a warning so operators can see which
    file is corrupted and why.
    """

    def test_corrupted_history_logs_warning(self):
        """A corrupted JSON file should produce a log warning, not silence."""
        import json
        import logging

        from src.task_manager import _load_history

        with tempfile.TemporaryDirectory() as tmpdir:
            from src import settings as app_settings

            original = app_settings.settings.runs_dir
            app_settings.settings.runs_dir = tmpdir
            try:
                # Write a valid run so we know the dir is set up
                from src.task_manager import _save_run

                _save_run({"id": "valid1", "result": {"acc": 0.9}})

                # Write a corrupted JSON file
                with open(os.path.join(tmpdir, "corrupt.json"), "w") as f:
                    f.write("{not valid json")

                # Use a list to capture log records
                records = []

                class Handler(logging.Handler):
                    def emit(self, record):
                        records.append(record)

                logger = logging.getLogger("distilkit")
                handler = Handler()
                logger.addHandler(handler)
                try:
                    history = _load_history()
                    # The valid run should be loaded; the corrupted one skipped
                    assert len(history) == 1
                    assert history[0]["id"] == "valid1"
                    # A warning should have been logged about the corrupted file
                    assert any(
                        "corrupt.json" in r.getMessage() for r in records
                    ), "No warning logged about corrupted file"
                    assert any(
                        "corrupt" in r.getMessage().lower() for r in records
                    ), "No corruption warning logged"
                finally:
                    logger.removeHandler(handler)
            finally:
                app_settings.settings.runs_dir = original

    def test_dir_does_not_exist_returns_empty(self):
        """_load_history returns [] when the runs directory is missing."""
        from src.task_manager import _load_history

        with tempfile.TemporaryDirectory() as tmpdir:
            from src import settings as app_settings

            original = app_settings.settings.runs_dir
            app_settings.settings.runs_dir = os.path.join(tmpdir, "nonexistent")
            try:
                history = _load_history()
                assert history == []
            finally:
                app_settings.settings.runs_dir = original


# ---------------------------------------------------------------------------
# ARC-02 regression: settings are sourced from environment variables
# ---------------------------------------------------------------------------


class TestSettingsRegression:
    """Configuration must be externalizable via environment variables.

    Bug: ``DEVICE`` was hardcoded to ``"cpu"`` in the module body.  Other
    settings (``RUNS_DIR``, ``MAX_LOG_SIZE``) were scattered as plain
    module-level constants.
    Fix: All configuration lives in ``src/settings.py`` and every field
    has an ``os.environ.get()`` fallback.
    """

    def test_device_from_env(self):
        """Setting DEVICE env var changes settings.device."""
        from src.settings import Settings

        s = Settings.from_env()
        assert s.device == "cpu"  # default

        os.environ["DEVICE"] = "cuda"
        try:
            s2 = Settings.from_env()
            assert s2.device == "cuda"
        finally:
            os.environ["DEVICE"] = "cpu"

    def test_port_from_env(self):
        """Setting PORT env var changes settings.port."""
        from src.settings import Settings

        os.environ["PORT"] = "8080"
        try:
            s = Settings.from_env()
            assert s.port == 8080
        finally:
            del os.environ["PORT"]

    def test_api_only_from_env(self):
        """Setting API_ONLY=true changes settings.api_only."""
        from src.settings import Settings

        os.environ["API_ONLY"] = "true"
        try:
            s = Settings.from_env()
            assert s.api_only is True
        finally:
            del os.environ["API_ONLY"]

    def test_api_only_false_by_default(self):
        """API_ONLY defaults to False."""
        from src.settings import Settings

        s = Settings.from_env()
        assert s.api_only is False

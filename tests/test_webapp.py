"""Tests for the web GUI API."""

import os
import tempfile

from fastapi.testclient import TestClient

# We need to set up the environment before importing the app
os.environ["DEVICE"] = "cpu"

from src import datasets as ds
from src.webapp import _load_history, _save_run, app

DATASET_CHOICES = ds.DATASET_CHOICES
TEACHER_CHOICES = ds.TEACHER_CHOICES
STUDENT_CHOICES = ds.STUDENT_CHOICES

client = TestClient(app)


class TestHistory:
    """Tests for the run history persistence."""

    def test_load_history_empty(self):
        """_load_history should return empty list when no runs directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original = "runs"
            import src.webapp

            src.webapp.RUNS_DIR = tmpdir
            try:
                history = _load_history()
                assert history == []
            finally:
                src.webapp.RUNS_DIR = original

    def test_save_and_load(self):
        """_save_run should persist and _load_history should retrieve."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import src.webapp

            original = src.webapp.RUNS_DIR
            src.webapp.RUNS_DIR = tmpdir
            try:
                run = {"id": "test123", "result": {"accuracy": 0.95}}
                _save_run(run)
                history = _load_history()
                assert len(history) == 1
                assert history[0]["id"] == "test123"
                assert history[0]["result"]["accuracy"] == 0.95
            finally:
                src.webapp.RUNS_DIR = original

    def test_save_corrupted_file_skipped(self):
        """Corrupted JSON files should be skipped during load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import src.webapp

            original = src.webapp.RUNS_DIR
            src.webapp.RUNS_DIR = tmpdir
            try:
                # Write a valid run
                _save_run({"id": "valid1", "result": {"acc": 0.9}})
                # Write a corrupted file
                with open(os.path.join(tmpdir, "corrupt.json"), "w") as f:
                    f.write("{not valid json")
                history = _load_history()
                assert len(history) == 1
                assert history[0]["id"] == "valid1"
            finally:
                src.webapp.RUNS_DIR = original


class TestAPI:
    """Tests for the FastAPI endpoints."""

    def test_index_returns_html(self):
        """GET / should return HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_config_endpoint(self):
        """GET /api/config should return valid config."""
        response = client.get("/api/config")
        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data
        assert "teachers" in data
        assert "students" in data
        assert "device" in data
        assert data["device"] == "cpu"
        assert all(d in data["datasets"] for d in ["CIFAR-10", "MNIST", "FashionMNIST", "SVHN"])
        assert "resnet18" in data["teachers"]
        assert "MiniCNN" in data["students"]

    def test_train_invalid_dataset(self):
        """POST /api/train with invalid dataset should return 400."""
        response = client.post(
            "/api/train",
            json={
                "dataset": "InvalidDataset",
                "teacher": "resnet18",
                "epochs": 1,
            },
        )
        assert response.status_code == 400
        assert "Invalid dataset" in response.json()["detail"]

    def test_train_invalid_teacher(self):
        """POST /api/train with invalid teacher should return 400."""
        response = client.post(
            "/api/train",
            json={
                "dataset": "CIFAR-10",
                "teacher": "invalid_model",
                "epochs": 1,
            },
        )
        assert response.status_code == 400
        assert "Invalid teacher" in response.json()["detail"]

    def test_train_invalid_student(self):
        """POST /api/train with invalid student should return 400."""
        response = client.post(
            "/api/train",
            json={
                "dataset": "CIFAR-10",
                "teacher": "resnet18",
                "student": "InvalidStudent",
                "epochs": 1,
            },
        )
        assert response.status_code == 400
        assert "Invalid student" in response.json()["detail"]

    def test_train_minimal_config(self):
        """POST /api/train with minimal config should create a task."""
        response = client.post(
            "/api/train",
            json={
                "dataset": "CIFAR-10",
                "teacher": "resnet18",
                "epochs": 1,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert len(data["task_id"]) == 12

    def test_train_full_config(self):
        """POST /api/train with full config should create a task."""
        response = client.post(
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
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data

    def test_tasks_endpoint(self):
        """GET /api/tasks should return a list."""
        # First create a task
        client.post(
            "/api/train",
            json={
                "dataset": "CIFAR-10",
                "teacher": "resnet18",
                "epochs": 1,
            },
        )
        response = client.get("/api/tasks")
        assert response.status_code == 200
        tasks = response.json()
        assert isinstance(tasks, list)
        assert len(tasks) > 0
        assert "id" in tasks[0]
        assert "status" in tasks[0]
        assert "config" in tasks[0]

    def test_get_nonexistent_task(self):
        """GET /api/train/nonexistent should return 404."""
        response = client.get("/api/train/nonexistent123")
        assert response.status_code == 404

    def test_cancel_nonexistent_task(self):
        """POST /api/train/nonexistent/cancel should return 404."""
        response = client.post("/api/train/nonexistent/cancel")
        assert response.status_code == 404

    def test_export_no_task(self):
        """POST /api/export/nonexistent should return 400."""
        response = client.post("/api/export/nonexistent", json={"format": "onnx"})
        assert response.status_code == 400

    def test_download_nonexistent(self):
        """GET /api/download/nonexistent should return 404."""
        response = client.get("/api/download/nonexistent.onnx")
        assert response.status_code == 404

    def test_history_endpoint(self):
        """GET /api/history should return a list."""
        response = client.get("/api/history")
        assert response.status_code == 200
        history = response.json()
        assert isinstance(history, list)

    def test_config_datasets_match_constant(self):
        """API config datasets should match DATASET_CHOICES."""
        response = client.get("/api/config")
        data = response.json()
        assert set(data["datasets"]) == set(DATASET_CHOICES)

    def test_config_teachers_match_constant(self):
        """API config teachers should match TEACHER_CHOICES."""
        response = client.get("/api/config")
        data = response.json()
        assert set(data["teachers"]) == set(TEACHER_CHOICES)

    def test_config_students_match_constant(self):
        """API config students should match STUDENT_CHOICES."""
        response = client.get("/api/config")
        data = response.json()
        assert set(data["students"]) == set(STUDENT_CHOICES)

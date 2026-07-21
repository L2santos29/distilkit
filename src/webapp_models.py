"""Pydantic request/response models for the DistilKit web API.

Extracted from ``webapp_routes.py`` to keep files under the 400-line limit.
"""

from pydantic import BaseModel, Field, field_validator

from src import datasets as ds


class TrainRequest(BaseModel):
    """Validated request body for POST /api/train."""

    dataset: str = Field(default="CIFAR-10", description="Dataset name")
    teacher: str = Field(default="resnet18", description="Teacher model architecture")
    student: str = Field(default="MiniCNN", description="Student model architecture")
    compression_ratio: float = Field(
        default=0.05, ge=0.01, le=1.0, description="Target student/teacher parameter ratio"
    )
    epochs: int = Field(default=10, ge=1, le=1000, description="Number of training epochs")
    temperature: float = Field(
        default=4.0, ge=0.1, le=100.0, description="Distillation temperature"
    )
    alpha: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Distillation loss weight"
    )
    patience: int = Field(default=0, ge=0, le=100, description="Early stopping patience")
    batch_size: int = Field(default=64, ge=1, le=4096, description="Training batch size")

    @field_validator("dataset")
    @classmethod
    def _check_dataset(cls, v: str) -> str:
        if v not in ds.DATASETS:
            raise ValueError(f"Invalid dataset. Choose: {ds.DATASET_CHOICES}")
        return v

    @field_validator("teacher")
    @classmethod
    def _check_teacher(cls, v: str) -> str:
        if v not in ds.TEACHER_CHOICES:
            raise ValueError(f"Invalid teacher. Choose: {ds.TEACHER_CHOICES}")
        return v

    @field_validator("student")
    @classmethod
    def _check_student(cls, v: str) -> str:
        if v not in ds.STUDENT_CHOICES:
            raise ValueError(f"Invalid student. Choose: {ds.STUDENT_CHOICES}")
        return v


class ExportRequest(BaseModel):
    """Validated request body for POST /api/export/{task_id}."""

    format: str = Field(default="onnx", description="Export format")

    @field_validator("format")
    @classmethod
    def _check_format(cls, v: str) -> str:
        if v not in ("onnx", "torchscript"):
            raise ValueError("Invalid format. Use 'onnx' or 'torchscript'")
        return v

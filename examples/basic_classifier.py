"""Full knowledge distillation example.

Demonstrates end-to-end workflow:
  teacher → distiller.train() → student → benchmark → export
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.teacher import load_teacher
from src.student import build_student
from src.distiller import Distiller
from src.benchmarks import compare_teacher_student
from src.onnx_export import export_to_onnx


def get_cifar10(batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
    """Load CIFAR-10 train/val dataloaders."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    val_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_val)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader


def main():
    print("⚡ DistilKit — Knowledge Distillation Example\n")

    # Configuration
    TEACHER_MODEL = "resnet18"
    EPOCHS = 5
    TEMPERATURE = 4.0
    ALPHA = 0.7
    BATCH_SIZE = 64

    # 1. Load data
    print("📦 Loading CIFAR-10...")
    train_loader, val_loader = get_cifar10(BATCH_SIZE)

    # 2. Load teacher
    print(f"🧠 Loading teacher ({TEACHER_MODEL})...")
    teacher = load_teacher(TEACHER_MODEL, num_classes=10)
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"   Teacher parameters: {teacher_params:,}")

    # 3. Build student
    print("🔧 Building student...")
    student = build_student(teacher, compression_ratio=0.25, num_classes=10)
    student_params = sum(p.numel() for p in student.parameters())
    print(f"   Student parameters: {student_params:,}")
    print(f"   Compression ratio: {student_params / teacher_params:.2%}")

    # 4. Distill
    print(f"\n🔄 Distilling ({EPOCHS} epochs, T={TEMPERATURE}, α={ALPHA})...")
    distiller = Distiller(teacher, student, temperature=TEMPERATURE, alpha=ALPHA)
    history = distiller.train(train_loader, val_loader, epochs=EPOCHS)

    # 5. Compare
    print("\n📊 Benchmarking teacher vs. student...")
    comparison = compare_teacher_student(teacher, student, target="cpu")
    print(f"   Teacher: {comparison['teacher']['mean_ms']} ms")
    print(f"   Student: {comparison['student']['mean_ms']} ms")
    print(f"   Speedup: {comparison['speedup']}x")
    print(f"   Size reduction: {(1 - comparison['compression']) * 100:.1f}%")

    # 6. Export
    print("\n💾 Exporting student model...")
    onnx_path = export_to_onnx(student, "checkpoints/student.onnx")
    print(f"   ONNX model: {onnx_path}")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()

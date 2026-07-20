"""Tests for the distillation training loop."""

import torch

from src.distiller import DistillationLoss


def test_distillation_loss_shape():
    """Distillation loss returns a scalar."""
    loss_fn = DistillationLoss(temperature=4.0, alpha=0.7)

    student_logits = torch.randn(8, 10)
    teacher_logits = torch.randn(8, 10)
    labels = torch.randint(0, 10, (8,))

    loss = loss_fn(student_logits, teacher_logits, labels)
    assert loss.ndim == 0  # Scalar
    assert loss.item() > 0


def test_full_alpha_is_pure_distillation():
    """With alpha=1.0, loss ignores hard labels."""
    loss_fn = DistillationLoss(temperature=1.0, alpha=1.0)

    student_logits = torch.randn(8, 10)
    labels = torch.randint(0, 10, (8,))

    # Same inputs should give zero loss when alpha=1.0
    loss = loss_fn(student_logits, student_logits, labels)
    assert loss.item() < 0.01

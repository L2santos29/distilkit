"""Core knowledge distillation training loop.

Implements the standard distillation loss:
    L = α * KL_div(student_logits/T, teacher_logits/T) * T²
        + (1 - α) * CrossEntropy(student_logits, labels)

where:
    α (alpha) = weight between distillation and hard-label loss
    T (temperature) = softness of probability distributions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class DistillationLoss(nn.Module):
    """Combined distillation loss: KL divergence + cross-entropy."""

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        """Initialize distillation loss.

        Args:
            temperature: Softening factor. Higher = softer distributions.
            alpha: Weight for distillation loss. (1-alpha) for hard-label loss.
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined distillation loss.

        Args:
            student_logits: Raw logits from student model.
            teacher_logits: Raw logits from teacher model.
            labels: Ground truth class labels.

        Returns:
            Scalar loss value.
        """
        # Distillation loss (KL divergence between softened distributions)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        distillation_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean")
        distillation_loss *= self.temperature ** 2

        # Hard-label loss (standard cross-entropy)
        hard_loss = self.ce_loss(student_logits, labels)

        return self.alpha * distillation_loss + (1 - self.alpha) * hard_loss


class Distiller:
    """Knowledge distiller: trains a student to mimic a teacher."""

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.7,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize distiller.

        Args:
            teacher: Pre-trained teacher model (frozen).
            student: Student model to be trained.
            temperature: Softening factor for logits.
            alpha: Distillation loss weight.
            device: Training device.
        """
        self.device = device
        self.teacher = teacher.to(device).eval()
        self.student = student.to(device)
        self.criterion = DistillationLoss(temperature, alpha)

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 10,
        lr: float = 1e-3,
    ) -> dict:
        """Run knowledge distillation training.

        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
            epochs: Number of training epochs.
            lr: Learning rate.

        Returns:
            Training history dict with 'train_loss' and 'val_acc' lists.
        """
        optimizer = torch.optim.Adam(self.student.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        history = {"train_loss": [], "val_acc": []}

        for epoch in range(epochs):
            # Training
            self.student.train()
            epoch_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                with torch.no_grad():
                    teacher_logits = self.teacher(images)

                student_logits = self.student(images)
                loss = self.criterion(student_logits, teacher_logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            history["train_loss"].append(avg_loss)

            # Validation
            if val_loader:
                acc = self._evaluate(val_loader)
                history["val_acc"].append(acc)

            scheduler.step()

            print(
                f"Epoch {epoch + 1}/{epochs} — "
                f"Loss: {avg_loss:.4f}"
                + (f" — Val Acc: {acc:.2%}" if val_loader else "")
            )

        return history

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> float:
        """Compute top-1 accuracy on a validation loader."""
        self.student.eval()
        correct, total = 0, 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.student(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        return correct / total

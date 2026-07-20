"""Core knowledge distillation training loop.

Implements the standard distillation loss:
    L = α * KL_div(student_logits/T, teacher_logits/T) * T²
        + (1 - α) * CrossEntropy(student_logits, labels)

where:
    α (alpha) = weight between distillation and hard-label loss
    T (temperature) = softness of probability distributions
"""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.log_config import logger


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
        distillation_loss *= self.temperature**2

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
        device: str = "cpu",
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
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        patience: int = 0,
        start_epoch: int = 0,
        initial_history: dict[str, list[float]] | None = None,
        *,
        on_epoch_end: "Callable[[int, int, float, float | None], None] | None" = None,
        on_batch_end: "Callable[[int, int, int, int, float], None] | None" = None,
        ckpt_callback: "Callable[[int, dict], None] | None" = None,
        cancel_flag: "Callable[[], bool] | None" = None,
    ) -> dict[str, list[float]]:
        """Run knowledge distillation training.

        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
            epochs: Number of training epochs.
            lr: Learning rate (used only when *optimizer* is not provided).
            optimizer: Optional pre-created optimizer. If ``None``, creates Adam.
            scheduler: Optional pre-created scheduler. If ``None``, creates
                ``CosineAnnealingLR``.
            patience: Early-stopping patience (0 = disabled).
            start_epoch: Starting epoch index (for resuming).
            initial_history: Previous training history to extend (for resuming).
            on_epoch_end: Called after each epoch with
                ``(epoch, total, avg_loss, acc)``.
            on_batch_end: Called after each batch with
                ``(epoch, total_epochs, batch_idx, total_batches, loss)``.
            ckpt_callback: Called every epoch with ``(epoch, checkpoint_dict)``.
                The dict contains ``"model"``, ``"optimizer"``, ``"losses"``,
                ``"accuracies"`` and ``"epoch"`` keys.
            cancel_flag: Callable that returns ``True`` to stop training.

        Returns:
            Training history dict with ``'train_loss'`` and ``'val_acc'`` lists.
        """
        opt = optimizer or torch.optim.Adam(self.student.parameters(), lr=lr)
        sched = scheduler or torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

        history: dict[str, list[float]] = {
            "train_loss": list(initial_history.get("train_loss", []) if initial_history else []),
            "val_acc": list(initial_history.get("val_acc", []) if initial_history else []),
        }

        best_acc = 0.0
        patience_counter = 0

        for epoch in range(start_epoch, epochs):
            if cancel_flag and cancel_flag():
                logger.info("Training cancelled.")
                break

            # --- Training ---
            self.student.train()
            epoch_loss = 0.0
            num_batches = len(train_loader)

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                with torch.no_grad():
                    teacher_logits = self.teacher(images)

                student_logits = self.student(images)
                loss = self.criterion(student_logits, teacher_logits, labels)

                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += loss.item()

                if on_batch_end:
                    on_batch_end(epoch, epochs, batch_idx, num_batches, loss.item())

            avg_loss = epoch_loss / num_batches
            history["train_loss"].append(avg_loss)

            # --- Validation ---
            acc = None
            if val_loader:
                acc = self._evaluate(val_loader)
                history["val_acc"].append(acc)

            sched.step()

            # --- Early stopping ---
            if patience > 0 and acc is not None:
                if acc > best_acc + 0.001:
                    best_acc = acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"   ⏹️ Early stopping (best: {best_acc:.2%})")
                        break

            logger.info(
                f"Epoch {epoch + 1}/{epochs} — "
                f"Loss: {avg_loss:.4f}" + (f" — Val Acc: {acc:.2%}" if acc is not None else "")
            )

            if on_epoch_end:
                on_epoch_end(epoch, epochs, avg_loss, acc)

            # --- Checkpoint ---
            if ckpt_callback:
                ckpt_callback(
                    epoch + 1,
                    {
                        "epoch": epoch + 1,
                        "model": self.student.state_dict(),
                        "optimizer": opt.state_dict(),
                        "losses": history["train_loss"],
                        "accuracies": history["val_acc"],
                    },
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

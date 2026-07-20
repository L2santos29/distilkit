"""Student model builder.

Constructs compact student models designed to be smaller, faster versions
of their teacher counterparts for knowledge distillation.
"""

import torch
import torch.nn as nn


class MiniCNN(nn.Module):
    """Lightweight CNN with configurable width for compression control.

    Width multiplier scales all channel counts. width=1.0 is the default.
    width=0.5 halves the parameters (~4x fewer), width=2.0 doubles them.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10, width: float = 1.0):
        """Build MiniCNN with width-scalable convolutional layers.

        Args:
            in_channels: Number of input image channels.
            num_classes: Number of output classes.
            width: Channel multiplier for compression control.
        """
        super().__init__()
        w = width
        c1, c2, c3, c4 = [int(32 * w), int(64 * w), int(128 * w), int(256 * w)]
        # Ensure at least 1 channel per layer
        c1, c2, c3, c4 = max(c1, 1), max(c2, 1), max(c3, 1), max(c4, 1)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c3, c4, 3, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class MiniResNet(nn.Module):
    """Tiny ResNet-style student with configurable width."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10, width: float = 1.0):
        """Build MiniResNet with residual blocks and scalable width.

        Args:
            in_channels: Number of input image channels.
            num_classes: Number of output classes.
            width: Channel multiplier for compression control.
        """
        super().__init__()
        w = width
        c1, c2 = int(16 * w), int(32 * w)
        c1, c2 = max(c1, 1), max(c2, 1)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            ResidualBlock(c1, c1),
            ResidualBlock(c1, c1),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            ResidualBlock(c2, c2),
            ResidualBlock(c2, c2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(c2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """Basic residual block with two conv layers."""

    def __init__(self, in_channels: int, out_channels: int):
        """Build a residual block.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


STUDENT_REGISTRY = {
    "MiniCNN": MiniCNN,
    "MiniResNet": MiniResNet,
}


def build_student(
    teacher: nn.Module | None = None,
    student_type: str = "MiniCNN",
    compression_ratio: float = 0.25,
    num_classes: int = 10,
    in_channels: int = 3,
) -> nn.Module:
    """Build a student model sized to approximate a target compression ratio.

    The width multiplier is computed so that the student has roughly
    ``compression_ratio * teacher_params`` parameters.

    Args:
        teacher: Teacher model (used to count params when compression_ratio is set).
        student_type: Name of the student architecture.
        compression_ratio: Target ratio of student/teacher parameters.
        num_classes: Number of output classes.
        in_channels: Number of input channels (1 for MNIST, 3 for CIFAR-10).

    Returns:
        Student model instance.
    """
    if student_type not in STUDENT_REGISTRY:
        raise ValueError(
            f"Unknown student: {student_type}. Available: {list(STUDENT_REGISTRY.keys())}"
        )

    # Estimate width from compression ratio
    if teacher is not None and compression_ratio > 0:
        teacher_params = sum(p.numel() for p in teacher.parameters())
        target_params = int(teacher_params * compression_ratio)
        # Base model at width=1.0
        base = STUDENT_REGISTRY[student_type](
            in_channels=in_channels, num_classes=num_classes, width=1.0
        )
        base_params = sum(p.numel() for p in base.parameters())
        # Params scale roughly with width² → width ≈ sqrt(target / base)
        if base_params > 0:
            w = max(0.125, min(4.0, (target_params / base_params) ** 0.5))
        else:
            w = 1.0
    else:
        w = 1.0

    return STUDENT_REGISTRY[student_type](in_channels=in_channels, num_classes=num_classes, width=w)

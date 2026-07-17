"""Student model builder.

Constructs compact student models designed to be smaller, faster versions
of their teacher counterparts for knowledge distillation.
"""

import torch.nn as nn


class MiniCNN(nn.Module):
    """Lightweight CNN suitable as a student for ResNet teachers on small datasets.

    Designed for datasets like CIFAR-10, MNIST. Small enough to show
    meaningful compression ratios but deep enough to achieve reasonable accuracy.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 4x4

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # 1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class MiniResNet(nn.Module):
    """Tiny ResNet-style student: ~2-5% the size of ResNet50."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Initial conv
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Block 1
            ResidualBlock(16, 16),
            ResidualBlock(16, 16),

            # Downsample
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 2
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """Basic residual block with two conv layers."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


def build_student(
    teacher: nn.Module, compression_ratio: float = 0.25, num_classes: int = 10
) -> nn.Module:
    """Build a student model based on the teacher and desired compression.

    Args:
        teacher: The teacher model (used to infer architecture type).
        compression_ratio: Rough parameter count ratio (student/teacher).
        num_classes: Number of output classes.

    Returns:
        Student model instance.
    """
    # Currently supports two student architectures
    if compression_ratio > 0.1:
        return MiniCNN(num_classes)
    return MiniResNet(num_classes)

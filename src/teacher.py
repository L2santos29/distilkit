"""Teacher model loader and wrapper.

Provides utilities to load pre-trained models for use as teacher models
in the distillation pipeline.
"""

import torch.nn as nn
from torchvision import models


def load_teacher(model_name: str, num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """Load a pre-trained model to serve as a teacher.

    Args:
        model_name: Model architecture (e.g., 'resnet50', 'resnet18',
                    'mobilenet_v3_large', 'efficientnet_b0').
        num_classes: Number of output classes.
        pretrained: Whether to load pre-trained weights.

    Returns:
        Loaded teacher model.

    Raises:
        ValueError: If model_name is not supported.
    """
    model_registry = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "mobilenet_v2": models.mobilenet_v2,
        "mobilenet_v3_large": models.mobilenet_v3_large,
        "efficientnet_b0": models.efficientnet_b0,
        "efficientnet_b1": models.efficientnet_b1,
    }

    if model_name not in model_registry:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_registry.keys())}")

    model_fn = model_registry[model_name]

    # Handle models that need explicit num_classes (ResNet) vs. those that don't (MobileNet)
    if model_name.startswith("resnet") or model_name.startswith("efficientnet"):
        model = model_fn(weights="DEFAULT" if pretrained else None)
        # Replace classifier head if num_classes differs from default
        if hasattr(model, "fc") and model.fc.out_features != num_classes:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        weights = "DEFAULT" if pretrained else None
        model = model_fn(weights=weights, num_classes=num_classes)

    return model

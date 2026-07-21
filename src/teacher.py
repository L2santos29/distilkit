"""Teacher model loader and wrapper.

Provides utilities to load pre-trained models for use as teacher models
in the distillation pipeline.
"""

import torch.nn as nn
from torchvision import models

# In-memory cache: loaded teacher models are reused across training runs.
# The key is ``{model_name}__{num_classes}__{pretrained}``.
_teacher_cache: dict[str, nn.Module] = {}


def _cache_key(model_name: str, num_classes: int, pretrained: bool) -> str:
    return f"{model_name}__{num_classes}__{pretrained}"


def load_teacher(model_name: str, num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """Load a torchvision model with a ``num_classes`` classifier head.

    Teachers are cached in memory once loaded so that repeated requests
    for the same architecture do not re-download weights or re-instantiate
    the model object.

    ResNet/EfficientNet models get their ``fc`` layer replaced when
    ``num_classes`` differs from the default; MobileNets accept the
    parameter directly.

    Raises:
        ValueError: If *model_name* is not in the supported registry.
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

    key = _cache_key(model_name, num_classes, pretrained)
    cached = _teacher_cache.get(key)
    if cached is not None:
        return cached

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

    _teacher_cache[key] = model
    return model

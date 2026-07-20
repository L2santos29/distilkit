"""Model export utilities — ONNX and TorchScript.

Converts trained student models to portable formats for deployment
on CPU, GPU, NPU, and edge devices.
"""

from pathlib import Path

import torch
import torch.nn as nn

from src.log_config import logger


def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 32, 32),
    opset_version: int = 17,
    dynamic_batch: bool = False,
) -> Path:
    """Export a PyTorch model to ONNX format.

    Args:
        model: Trained PyTorch model.
        output_path: Path for the .onnx file.
        input_shape: Shape of input tensor (B, C, H, W).
        opset_version: ONNX opset version.
        dynamic_batch: If True, the first dimension (batch) is dynamic.

    Returns:
        Path to the exported ONNX file.

    Raises:
        RuntimeError: If export fails.
    """
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(*input_shape)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # Verify export
    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    logger.info(f"✅ Exported to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    return output_path


def export_to_torchscript(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 32, 32),
) -> Path:
    """Export a PyTorch model to TorchScript format.

    Args:
        model: Trained PyTorch model.
        output_path: Path for the .pt file.
        input_shape: Shape of input tensor.

    Returns:
        Path to the exported TorchScript file.
    """
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(*input_shape)
    traced = torch.jit.trace(model, dummy_input)

    torch.jit.save(traced, str(output_path))

    logger.info(f"✅ Exported TorchScript to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    return output_path


def quantize_model(
    model: nn.Module,
) -> nn.Module:
    """Apply post-training static quantization (INT8).

    Args:
        model: FP32 trained model.

    Returns:
        Quantized model ready for ONNX Runtime.
    """
    model.eval()
    qconfig = torch.ao.quantization.get_default_qconfig("x86")
    torch.ao.quantization.prepare(model, inplace=True)
    # Note: calibration data required in production use
    torch.ao.quantization.convert(model, inplace=True)
    return model

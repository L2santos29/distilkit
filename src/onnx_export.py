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

    try:
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
    except (OSError, RuntimeError, ValueError) as e:
        logger.error(f"❌ ONNX export failed: {e}")
        raise RuntimeError(f"ONNX export failed: {e}") from e

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

    try:
        dummy_input = torch.randn(*input_shape)
        traced = torch.jit.trace(model, dummy_input)

        torch.jit.save(traced, str(output_path))
    except Exception as e:
        logger.error(f"❌ TorchScript export failed: {e}")
        raise RuntimeError(f"TorchScript export failed: {e}") from e

    logger.info(
        f"✅ Exported TorchScript to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)"
    )
    return output_path

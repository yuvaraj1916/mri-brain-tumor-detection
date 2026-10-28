"""Single-image tumor-region inference using saved experiment settings."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mri_tumor.preprocessing import PreprocessingConfig, load_image, image_to_tensor
from mri_tumor.segmentation.checkpoint import load_segmentation_checkpoint
from mri_tumor.segmentation.data import require_torch
from mri_tumor.segmentation.model import create_unet
from mri_tumor.utils.device import get_device


def predict_mask(
    model: Any,
    image_path: Path,
    preprocessing: PreprocessingConfig,
    *,
    device: Any = "cpu",
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Return per-pixel probabilities, a thresholded mask, and input metadata."""
    if not 0 <= threshold <= 1:
        raise ValueError("Prediction threshold must be in [0, 1].")
    torch = require_torch()
    image_path = image_path.expanduser().resolve()
    try:
        with Image.open(image_path) as source:
            original_size = source.size
            source.load()
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not decode MRI image {image_path}: {error}") from error
    image = load_image(image_path, preprocessing, resize=True)
    # The loader's output shape is the exact dimensions seen by the model.
    image_tensor = image_to_tensor(image).unsqueeze(0)
    target_device = torch.device(device) if isinstance(device, str) else device
    model = model.to(target_device)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            logits = model(image_tensor.to(target_device))
            if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 1:
                raise ValueError(
                    "Binary segmentation model must return logits with shape (1, 1, H, W)."
                )
            probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()
    finally:
        model.train(was_training)
    return {
        "probabilities": probabilities,
        "mask": probabilities >= threshold,
        "threshold": threshold,
        "preprocessed_shape": tuple(int(value) for value in image.shape[:2]),
        "preprocessed_channels": int(image.shape[2]),
        "original_size": tuple(int(value) for value in original_size),
        "preprocessing_config": asdict(preprocessing),
        "image_path": str(image_path),
    }


def predict_from_checkpoint(
    checkpoint_path: Path,
    image_path: Path,
    *,
    device_preference: str = "auto",
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Load a saved U-Net and run inference with its stored preprocessing."""
    torch = require_torch()
    device = get_device(device_preference)
    checkpoint = load_segmentation_checkpoint(checkpoint_path, map_location=device)
    required = {
        "model_state_dict",
        "model_config",
        "preprocessing_config",
        "training_config",
    }
    if not required.issubset(checkpoint):
        raise ValueError("Segmentation checkpoint is missing required model metadata.")
    preprocessing = PreprocessingConfig(**checkpoint["preprocessing_config"])
    model = create_unet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    result = predict_mask(
        model,
        image_path,
        preprocessing,
        device=device,
        threshold=threshold,
    )
    result["device"] = str(device)
    result["checkpoint_path"] = str(checkpoint_path.expanduser().resolve())
    return result

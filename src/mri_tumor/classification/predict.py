"""Single-image classification prediction using a saved run checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mri_tumor.classification.data import CLASS_NAMES, require_torch
from mri_tumor.classification.checkpoint import load_checkpoint
from mri_tumor.classification.model import create_model
from mri_tumor.preprocessing import PreprocessingConfig, image_to_tensor, load_image
from mri_tumor.utils.device import get_device


def preprocess_for_prediction(
    image_path: Path,
    preprocessing: PreprocessingConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the configured training preprocessing and return image metadata."""
    image = load_image(image_path, preprocessing)
    info = {
        "source_path": str(Path(image_path).expanduser()),
        "original_size": None,
        "processed_size": [int(image.shape[1]), int(image.shape[0])],
        "channels": int(image.shape[2]),
        "dtype": str(image.dtype),
        "normalization": preprocessing.image_normalization,
        "color_mode": preprocessing.color_mode,
        "image_resampling": preprocessing.image_resampling,
        "configured_size": list(preprocessing.image_size),
    }
    try:
        from PIL import Image

        with Image.open(image_path) as source:
            info["original_size"] = [int(source.width), int(source.height)]
    except OSError as error:
        raise ValueError(f"Could not read source image metadata {image_path}: {error}") from error
    return image, info


def predict_image(
    image_path: Path,
    checkpoint_path: Path,
    *,
    device: str = "auto",
) -> dict[str, Any]:
    """Load a selected checkpoint and predict one MRI using its saved mapping."""
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Classifier checkpoint does not exist: {checkpoint_path}. "
            "Train the model before requesting predictions."
        )
    torch = require_torch()
    target_device = get_device(device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=target_device)
    required = {
        "model_state_dict",
        "model_config",
        "class_to_index",
        "preprocessing_config",
    }
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError(
            f"Checkpoint is missing required classifier metadata: "
            f"{', '.join(sorted(required))}."
        )
    class_to_index = checkpoint["class_to_index"]
    if class_to_index != {name: index for index, name in enumerate(CLASS_NAMES)}:
        raise ValueError("Checkpoint class mapping is incompatible with this project.")
    model_config = checkpoint["model_config"]
    preprocessing = PreprocessingConfig(**checkpoint["preprocessing_config"])
    model = create_model(**model_config).to(target_device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    image, image_info = preprocess_for_prediction(image_path, preprocessing)
    tensor = image_to_tensor(image).unsqueeze(0).to(target_device)
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()
    predicted_index = int(np.argmax(probabilities))
    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "probabilities": {
            name: float(probabilities[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "preprocessed_image": image_info,
    }

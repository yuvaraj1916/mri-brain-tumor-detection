"""Combined single-image classification and segmentation inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from mri_tumor.classification.data import require_torch
from mri_tumor.classification.predict import predict_image
from mri_tumor.segmentation.predict import predict_from_checkpoint
from mri_tumor.utils.device import get_device

TUMOR_MASK_THRESHOLD = 0.5
_CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")


class InferenceError(RuntimeError):
    """Safe, user-facing error raised by the combined inference workflow."""


def _select_device(torch: Any) -> Any:
    """Prefer Apple MPS when available and otherwise use CPU."""
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    is_available = getattr(mps, "is_available", None)
    return get_device("mps" if callable(is_available) and is_available() else "cpu")


def _read_original_image(image_path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Decode the original image as RGB without applying model preprocessing."""
    try:
        with Image.open(image_path) as source:
            source.load()
            original_size = (int(source.width), int(source.height))
            pixels = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
    except (
        OSError,
        ValueError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as error:
        raise InferenceError("The input MRI image could not be read.") from None
    if pixels.ndim != 3 or pixels.shape[2] != 3 or min(original_size) <= 0:
        raise InferenceError("The input MRI image has an unsupported shape.")
    return pixels, original_size


def predict_mri(
    image_path: Path,
    classification_checkpoint: Path,
    segmentation_checkpoint: Path,
) -> dict[str, Any]:
    """Predict the class and tumor mask for one MRI using existing checkpoints.

    Each model applies preprocessing stored in its own checkpoint. The returned
    probability map and mask are in segmentation-model dimensions; the original
    image, aligned mask, and overlay are in the source image dimensions.
    """
    image_path = Path(image_path).expanduser()
    classification_checkpoint = Path(classification_checkpoint).expanduser()
    segmentation_checkpoint = Path(segmentation_checkpoint).expanduser()

    if not image_path.is_file():
        raise InferenceError("The input MRI image is unavailable.")
    if not classification_checkpoint.is_file() or not segmentation_checkpoint.is_file():
        raise InferenceError("A required trained model checkpoint is unavailable.")

    original_image, original_size = _read_original_image(image_path)
    torch = require_torch()
    device = _select_device(torch)

    try:
        classification = predict_image(
            image_path, classification_checkpoint, device=str(device)
        )
        segmentation = predict_from_checkpoint(
            segmentation_checkpoint,
            image_path,
            device_preference=str(device),
            threshold=TUMOR_MASK_THRESHOLD,
        )
    except FileNotFoundError:
        raise InferenceError("A required trained model checkpoint is unavailable.") from None
    except (OSError, ValueError):
        raise InferenceError(
            "The MRI could not be processed with the supplied model artifacts."
        ) from None
    except RuntimeError:
        raise InferenceError(
            "Model inference failed; verify the trained checkpoints and PyTorch installation."
        ) from None

    class_probabilities = classification.get("probabilities")
    if (
        not isinstance(class_probabilities, dict)
        or tuple(class_probabilities) != _CLASS_NAMES
    ):
        raise InferenceError("The classifier returned an incompatible class mapping.")
    class_values = np.asarray(
        [class_probabilities[name] for name in _CLASS_NAMES], dtype=np.float64
    )
    if (
        not np.isfinite(class_values).all()
        or np.any(class_values < 0)
        or np.any(class_values > 1)
        or not np.isclose(float(class_values.sum()), 1.0, atol=1e-5)
    ):
        raise InferenceError("The classifier returned invalid class probabilities.")
    predicted_class = classification.get("predicted_class")
    if predicted_class != _CLASS_NAMES[int(np.argmax(class_values))]:
        raise InferenceError("The classifier returned an inconsistent class label.")

    probabilities = np.asarray(segmentation.get("probabilities"))
    tumor_mask = np.asarray(segmentation.get("mask"))
    expected_shape = tuple(int(value) for value in segmentation["preprocessed_shape"])
    if (
        probabilities.ndim != 2
        or probabilities.shape != expected_shape
        or tumor_mask.shape != expected_shape
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.array_equal(tumor_mask, probabilities >= TUMOR_MASK_THRESHOLD)
    ):
        raise InferenceError("The segmenter returned an invalid probability map or mask.")

    width, height = original_size
    aligned_tumor_mask = np.asarray(
        Image.fromarray(tumor_mask.astype(np.uint8) * 255).resize(
            original_size, resample=Image.Resampling.NEAREST
        )
    ) >= 128
    overlay = original_image.astype(np.float32)
    red = np.asarray((255.0, 0.0, 0.0), dtype=np.float32)
    overlay[aligned_tumor_mask] = (
        0.65 * overlay[aligned_tumor_mask] + 0.35 * red
    )

    return {
        "predicted_class": predicted_class,
        "class_probabilities": {
            name: float(class_probabilities[name]) for name in _CLASS_NAMES
        },
        "segmentation_probability_map": probabilities.astype(np.float32, copy=False),
        "tumor_mask": tumor_mask.astype(bool, copy=False),
        "original_image": original_image,
        "aligned_tumor_mask": aligned_tumor_mask,
        "overlay": np.clip(overlay, 0, 255).astype(np.uint8),
        "threshold": TUMOR_MASK_THRESHOLD,
        "image_info": {
            "original_size": [width, height],
            "classification_processed_size": list(
                classification["preprocessed_image"]["processed_size"][::-1]
            ),
            "segmentation_processed_size": list(expected_shape),
            "device": str(device),
        },
    }

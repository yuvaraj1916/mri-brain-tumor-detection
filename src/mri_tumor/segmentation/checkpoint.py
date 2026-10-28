"""Checkpoint serialization for validation-selected U-Net weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mri_tumor.segmentation.data import require_torch


def save_segmentation_checkpoint(path: Path, checkpoint: dict[str, Any]) -> Path:
    """Save U-Net weights and complete experiment metadata."""
    torch = require_torch()
    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    return destination


def load_segmentation_checkpoint(
    path: Path, *, map_location: Any = "cpu"
) -> dict[str, Any]:
    """Load a U-Net checkpoint with clear missing and malformed-file errors."""
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Segmentation checkpoint does not exist: {source}")
    torch = require_torch()
    try:
        checkpoint = torch.load(source, map_location=map_location, weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(f"Could not load checkpoint {source}: {error}") from error
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Segmentation checkpoint must contain a mapping: {source}")
    return checkpoint

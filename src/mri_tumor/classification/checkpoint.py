"""Explicit classifier checkpoint serialization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mri_tumor.classification.data import require_torch


def save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> Path:
    """Save a complete checkpoint after creating its parent directory."""
    torch = require_torch()
    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    return destination


def load_checkpoint(path: Path, *, map_location: Any = "cpu") -> dict[str, Any]:
    """Load a checkpoint or raise a clear missing/corrupt-checkpoint error."""
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"Classifier checkpoint does not exist: {source}. "
            "Train the model before loading weights."
        )
    torch = require_torch()
    try:
        checkpoint = torch.load(source, map_location=map_location, weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(f"Could not load classifier checkpoint {source}: {error}") from error
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Classifier checkpoint must contain a mapping: {source}")
    return checkpoint

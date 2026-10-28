"""Reproducible preprocessing settings shared by loaders and scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mri_tumor.data.config import PROJECT_ROOT

ColorMode = Literal["grayscale", "rgb", "preserve"]
ImageNormalization = Literal["zero_one", "minus_one_one"]
ImageResampling = Literal["bilinear", "bicubic", "lanczos"]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "preprocessing.json"


@dataclass(frozen=True)
class PreprocessingConfig:
    """Image transforms and categorical-mask validation parameters.

    ``image_size`` is ``(width, height)`` to match Pillow's resize API.
    Intensity scaling is fixed to the uint8 domain and never uses statistics
    estimated from validation or test images.
    """

    image_size: tuple[int, int] = (256, 256)
    color_mode: ColorMode = "grayscale"
    image_normalization: ImageNormalization = "zero_one"
    image_resampling: ImageResampling = "bilinear"
    mask_resampling: Literal["nearest"] = "nearest"
    mask_background_value: int = 0
    mask_min_foreground_pixels: int = 1
    random_seed: int = 20261012

    def __post_init__(self) -> None:
        if len(self.image_size) != 2 or any(
            not isinstance(dimension, int) or dimension <= 0
            for dimension in self.image_size
        ):
            raise ValueError("image_size must contain two positive integers.")
        if self.color_mode not in {"grayscale", "rgb", "preserve"}:
            raise ValueError(f"Unsupported color_mode: {self.color_mode!r}")
        if self.image_normalization not in {"zero_one", "minus_one_one"}:
            raise ValueError(
                f"Unsupported image_normalization: {self.image_normalization!r}"
            )
        if self.image_resampling not in {"bilinear", "bicubic", "lanczos"}:
            raise ValueError(
                f"Unsupported image_resampling: {self.image_resampling!r}"
            )
        if self.mask_resampling != "nearest":
            raise ValueError("Segmentation masks must use nearest-neighbor resizing.")
        if self.mask_min_foreground_pixels < 1:
            raise ValueError("mask_min_foreground_pixels must be positive.")

    @classmethod
    def from_json(cls, path: Path = DEFAULT_CONFIG_PATH) -> PreprocessingConfig:
        """Load preprocessing settings without silently accepting bad values."""
        try:
            with path.open("r", encoding="utf-8") as source:
                settings = json.load(source)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not load preprocessing settings from {path}: {error}") from error
        if not isinstance(settings, dict):
            raise ValueError(f"Preprocessing settings must be a JSON object: {path}")
        if "image_size" in settings:
            settings["image_size"] = tuple(settings["image_size"])
        try:
            return cls(**settings)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid preprocessing settings in {path}: {error}") from error

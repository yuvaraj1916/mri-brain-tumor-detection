"""Safe MRI and categorical-mask loading and preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, UnidentifiedImageError

from mri_tumor.preprocessing.config import PreprocessingConfig

ColorMode = Literal["grayscale", "rgb", "preserve"]
_GRAYSCALE_MODES = {"1", "L", "I", "I;16", "I;16B", "I;16L", "F"}


class PreprocessingError(ValueError):
    """Raised when an image or mask cannot be safely preprocessed."""


def _read_image(path: Path, description: str) -> Image.Image:
    source_path = Path(path).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f"{description} file does not exist: {source_path}")
    try:
        with Image.open(source_path) as image:
            image.load()
            return image.copy()
    except (
        OSError,
        ValueError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as error:
        raise PreprocessingError(
            f"Could not decode {description} file {source_path}: {error}"
        ) from error


def _image_resampling(name: str) -> Image.Resampling:
    return {
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }[name]


def _normalize_pixels(
    pixels: np.ndarray, config: PreprocessingConfig
) -> np.ndarray:
    normalized = pixels.astype(np.float32) / np.float32(255.0)
    if config.image_normalization == "minus_one_one":
        normalized = normalized * np.float32(2.0) - np.float32(1.0)
    return normalized


def _load_image_array(
    path: Path,
    config: PreprocessingConfig,
    *,
    resize: bool,
    color_mode: ColorMode | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    image = _read_image(path, "MRI image")
    original_size = image.size
    selected_mode = config.color_mode if color_mode is None else color_mode
    if selected_mode == "preserve":
        selected_mode = (
            "grayscale" if image.mode in _GRAYSCALE_MODES else "rgb"
        )
    if selected_mode not in {"grayscale", "rgb"}:
        raise ValueError(f"Unsupported color mode: {selected_mode!r}")

    converted = image.convert("L" if selected_mode == "grayscale" else "RGB")
    if resize:
        converted = converted.resize(
            config.image_size, resample=_image_resampling(config.image_resampling)
        )
    pixels = np.asarray(converted, dtype=np.uint8)
    if pixels.ndim == 2:
        pixels = pixels[..., np.newaxis]
    return _normalize_pixels(pixels, config), original_size


def load_image(
    path: Path,
    config: PreprocessingConfig = PreprocessingConfig(),
    *,
    resize: bool = True,
    color_mode: ColorMode | None = None,
) -> np.ndarray:
    """Read one MRI as float32 HWC data while leaving its file untouched.

    Grayscale images have one trailing channel; RGB images have three.
    Intensity values are deterministically scaled to ``[0, 1]`` or ``[-1, 1]``.
    Set ``resize=False`` to retain the source dimensions.
    """
    image, _ = _load_image_array(
        path, config, resize=resize, color_mode=color_mode
    )
    return image


def _mask_pixels(image: Image.Image, path: Path) -> np.ndarray:
    if image.mode == "P":
        pixels = np.asarray(image)
    elif image.mode in _GRAYSCALE_MODES:
        pixels = np.asarray(image.convert("L") if image.mode == "1" else image)
    else:
        rgb = np.asarray(image.convert("RGB"))
        if not (
            np.array_equal(rgb[..., 0], rgb[..., 1])
            and np.array_equal(rgb[..., 0], rgb[..., 2])
        ):
            raise PreprocessingError(
                f"Mask must encode categorical labels in one grayscale channel: {path}"
            )
        pixels = rgb[..., 0]

    if pixels.ndim != 2 or pixels.size == 0:
        raise PreprocessingError(f"Mask must be a non-empty 2D label image: {path}")
    if not np.issubdtype(pixels.dtype, np.number) and pixels.dtype != np.bool_:
        raise PreprocessingError(f"Mask values must be numeric labels: {path}")
    if not np.isfinite(pixels).all():
        raise PreprocessingError(f"Mask contains non-finite values: {path}")
    if np.issubdtype(pixels.dtype, np.floating):
        if not np.equal(pixels, np.floor(pixels)).all():
            raise PreprocessingError(f"Mask contains non-integer label values: {path}")
        pixels = pixels.astype(np.int64)
    return np.array(pixels, copy=True)


def _validate_mask(
    pixels: np.ndarray, path: Path, config: PreprocessingConfig
) -> None:
    has_background = np.any(pixels == config.mask_background_value)
    foreground_pixels = int(np.count_nonzero(pixels != config.mask_background_value))
    if not has_background or foreground_pixels < config.mask_min_foreground_pixels:
        raise PreprocessingError(
            f"Mask must contain background label {config.mask_background_value} "
            f"and at least {config.mask_min_foreground_pixels} foreground pixels: "
            f"{path}"
        )


def _load_mask_array(
    path: Path,
    config: PreprocessingConfig,
    *,
    resize: bool,
) -> tuple[np.ndarray, tuple[int, int]]:
    mask_path = Path(path).expanduser()
    image = _read_image(mask_path, "segmentation mask")
    original_size = image.size
    pixels = _mask_pixels(image, mask_path)
    _validate_mask(pixels, mask_path, config)
    if resize and image.size != config.image_size:
        resized = Image.fromarray(pixels).resize(
            config.image_size, resample=Image.Resampling.NEAREST
        )
        pixels = np.asarray(resized).copy()
        _validate_mask(pixels, mask_path, config)
    return pixels, original_size


def load_mask(
    path: Path,
    config: PreprocessingConfig = PreprocessingConfig(),
    *,
    resize: bool = True,
) -> np.ndarray:
    """Read a 2D categorical mask without image normalization or label changes."""
    mask, _ = _load_mask_array(path, config, resize=resize)
    return mask


def load_segmentation_pair(
    image_path: Path,
    mask_path: Path,
    config: PreprocessingConfig = PreprocessingConfig(),
    *,
    resize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one verified image-mask pair and check source and output dimensions."""
    image, original_image_size = _load_image_array(
        image_path, config, resize=resize
    )
    mask, original_mask_size = _load_mask_array(mask_path, config, resize=resize)
    if original_image_size != original_mask_size:
        raise PreprocessingError(
            f"Source MRI and mask dimensions differ: {image_path} "
            f"{original_image_size} vs {mask_path} {original_mask_size}"
        )
    if image.shape[:2] != mask.shape:
        raise PreprocessingError(
            f"Preprocessed MRI and mask dimensions differ: {image_path} "
            f"{image.shape[:2]} vs {mask_path} {mask.shape}"
        )
    return image, mask


def image_to_tensor(image: np.ndarray) -> Any:
    """Convert normalized HWC image data to a float32 PyTorch CHW tensor."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for tensor conversion; install the build "
            "appropriate for this system from pytorch.org."
        ) from error
    if image.ndim != 3 or image.shape[2] not in {1, 3}:
        raise ValueError("Image arrays must have shape (height, width, 1 or 3).")
    chw = np.ascontiguousarray(np.transpose(image, (2, 0, 1)), dtype=np.float32)
    return torch.from_numpy(chw)


def mask_to_tensor(mask: np.ndarray) -> Any:
    """Convert a 2D categorical mask to an integer PyTorch label tensor."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for tensor conversion; install the build "
            "appropriate for this system from pytorch.org."
        ) from error
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("Mask arrays must be 2D integer categorical labels.")
    labels = np.ascontiguousarray(mask, dtype=np.int64)
    return torch.from_numpy(labels)


def load_image_tensor(
    path: Path,
    config: PreprocessingConfig = PreprocessingConfig(),
    *,
    resize: bool = True,
    color_mode: ColorMode | None = None,
) -> Any:
    """Load and preprocess an MRI directly as a PyTorch CHW tensor."""
    return image_to_tensor(
        load_image(path, config, resize=resize, color_mode=color_mode)
    )


def load_mask_tensor(
    path: Path,
    config: PreprocessingConfig = PreprocessingConfig(),
    *,
    resize: bool = True,
) -> Any:
    """Load a categorical mask directly as a PyTorch integer tensor."""
    return mask_to_tensor(load_mask(path, config, resize=resize))

"""Select an available PyTorch device without requiring a specific accelerator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch

DevicePreference = Literal["auto", "cpu", "cuda", "mps"]


def get_device(preference: DevicePreference = "auto") -> torch.device:
    """Return the requested PyTorch device or select the best available one.

    Automatic selection prefers CUDA, then Apple MPS, and finally CPU. PyTorch
    is imported only when this function is called so other project utilities
    remain usable before the deep-learning dependencies are installed.
    """
    if preference not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device preference: {preference!r}")

    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "PyTorch is required for device selection. Install a build "
            "appropriate for this computer from https://pytorch.org/get-started/locally/."
        ) from error

    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but it is not available.")
        return torch.device("cuda")
    if preference == "mps":
        if not _mps_is_available(torch):
            raise RuntimeError("Apple MPS was requested, but it is not available.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_is_available(torch):
        return torch.device("mps")
    return torch.device("cpu")


def _mps_is_available(torch_module: object) -> bool:
    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None)
    return mps is not None and mps.is_available()

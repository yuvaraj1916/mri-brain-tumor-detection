"""Project-relative dataset locations with optional environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

DatasetName = Literal["classification", "segmentation"]

_ENVIRONMENT_VARIABLES: dict[DatasetName, str] = {
    "classification": "MRI_CLASSIFICATION_DATASET_DIR",
    "segmentation": "MRI_SEGMENTATION_DATASET_DIR",
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DatasetPaths:
    """Roots for local classification and segmentation datasets.

    Defaults point to the checked-in project locations for the Brain Tumor MRI
    and BRISC 2025 datasets. Relative environment-variable values are resolved
    from the project root, so callers may run the project from any directory.
    """

    classification: Path = (
        PROJECT_ROOT / "datasets" / "classification" / "Brain Tumor MRI"
    )
    segmentation: Path = (
        PROJECT_ROOT / "datasets" / "segmentation" / "brisc2025"
    )

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> DatasetPaths:
        """Read path overrides without requiring datasets to exist."""
        environment = os.environ if environ is None else environ
        return cls(
            classification=_read_path(
                environment.get(_ENVIRONMENT_VARIABLES["classification"]),
                cls().classification,
            ),
            segmentation=_read_path(
                environment.get(_ENVIRONMENT_VARIABLES["segmentation"]),
                cls().segmentation,
            ),
        )

    def require(self, name: DatasetName) -> Path:
        """Return an existing dataset directory or raise an actionable error."""
        if name not in _ENVIRONMENT_VARIABLES:
            raise ValueError(f"Unsupported dataset name: {name!r}")

        path = getattr(self, name)
        environment_variable = _ENVIRONMENT_VARIABLES[name]
        resolved = path.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(
                f"The {name} dataset directory does not exist: {resolved}. "
                f"Place the dataset there or set {environment_variable}."
            )
        return resolved


def _read_path(value: str | None, default: Path) -> Path:
    if value is None or not value.strip():
        return default
    path = Path(value.strip()).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path

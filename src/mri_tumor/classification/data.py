"""Manifest-backed classification datasets and deterministic data loaders."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np

from mri_tumor.data.config import DatasetPaths, PROJECT_ROOT
from mri_tumor.preprocessing import PreprocessingConfig, image_to_tensor, load_image

SplitName = Literal["training", "validation", "testing"]
CLASS_NAMES: tuple[str, ...] = ("glioma", "meningioma", "notumor", "pituitary")
CLASS_TO_INDEX: dict[str, int] = {
    name: index for index, name in enumerate(CLASS_NAMES)
}
_SPLITS: tuple[SplitName, ...] = ("training", "validation", "testing")


@dataclass(frozen=True)
class ClassificationRecord:
    """One safe classification sample referenced by a derived manifest."""

    image_path: Path
    relative_image_path: str
    class_label: str
    label_index: int
    split: SplitName
    sha256: str


def require_torch() -> Any:
    """Import PyTorch only when tensor, model, or loader functionality is used."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for the classification data pipeline. "
            "Install a system-appropriate build using the commands in README.md."
        ) from error
    return torch


def latest_classification_manifest(
    manifest_dir: Path = PROJECT_ROOT / "results" / "data_manifests",
) -> Path:
    """Return the newest derived classification manifest; never use official splits."""
    directory = manifest_dir.expanduser().resolve()
    candidates = sorted(directory.glob("leakage_audit_*_classification.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No derived classification manifest exists in {directory}. "
            "Run scripts/audit_dataset_leakage.py first; official dataset "
            "splits are not used as a fallback."
        )
    return candidates[-1]


def _safe_image_path(root: Path, relative_path: str) -> Path:
    normalized = relative_path.strip().replace("\\", "/")
    relative = PurePosixPath(normalized)
    if not normalized or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe manifest image path: {relative_path!r}")
    resolved_root = root.expanduser().resolve()
    path = resolved_root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(resolved_root):
        raise ValueError(f"Manifest image path escapes dataset root: {relative_path!r}")
    return path


def read_classification_manifest(
    manifest_path: Path,
    dataset_root: Path | None = None,
) -> dict[SplitName, list[ClassificationRecord]]:
    """Read validated derived-manifest rows and verify no hash crosses splits."""
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Derived classification manifest does not exist: {manifest_path}. "
            "Run scripts/audit_dataset_leakage.py; no official-split fallback "
            "is performed."
        )
    if dataset_root is None:
        dataset_root = DatasetPaths.from_environment().require("classification")
    dataset_root = dataset_root.expanduser().resolve()

    required = {
        "relative_image_path",
        "class_label",
        "sha256",
        "proposed_derived_split",
        "review_flags",
    }
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"Derived manifest is missing required columns: "
                    f"{', '.join(sorted(missing))}"
                )
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"Could not read classification manifest {manifest_path}: {error}") from error

    partitions: dict[SplitName, list[ClassificationRecord]] = {
        split: [] for split in _SPLITS
    }
    split_hashes: dict[SplitName, set[str]] = {split: set() for split in _SPLITS}
    for line_number, row in enumerate(rows, start=2):
        proposed_split = row["proposed_derived_split"].strip()
        review_flags = row["review_flags"].strip()
        if proposed_split == "flagged" or review_flags:
            continue
        if proposed_split not in _SPLITS:
            raise ValueError(
                f"Unexpected derived split {proposed_split!r} on manifest "
                f"line {line_number}."
            )
        label = row["class_label"].strip().casefold()
        if label not in CLASS_TO_INDEX:
            raise ValueError(
                f"Unsupported class label {label!r} on manifest line "
                f"{line_number}; expected {', '.join(CLASS_NAMES)}."
            )
        relative_path = row["relative_image_path"].strip()
        image_path = _safe_image_path(dataset_root, relative_path)
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Manifest image is missing on line {line_number}: {image_path}"
            )
        digest = row["sha256"].strip().casefold()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(
                f"Invalid SHA-256 image hash on manifest line {line_number}: {digest!r}"
            )
        split = proposed_split
        split_hashes[split].add(digest)
        partitions[split].append(
            ClassificationRecord(
                image_path=image_path,
                relative_image_path=relative_path.replace("\\", "/"),
                class_label=label,
                label_index=CLASS_TO_INDEX[label],
                split=split,
                sha256=digest,
            )
        )

    verify_split_hashes(split_hashes)
    for split, records in partitions.items():
        if not records:
            raise ValueError(
                f"The derived classification manifest has no eligible "
                f"{split} samples."
            )
        partitions[split] = sorted(
            records, key=lambda record: record.relative_image_path.casefold()
        )
    return partitions


def verify_split_hashes(
    split_hashes: dict[str, set[str]],
) -> dict[str, int]:
    """Reject exact-image overlap among training, validation, and testing."""
    split_names = list(_SPLITS)
    overlap_counts: dict[str, int] = {}
    for index, first in enumerate(split_names):
        for second in split_names[index + 1 :]:
            overlap = split_hashes.get(first, set()) & split_hashes.get(second, set())
            overlap_counts[f"{first}:{second}"] = len(overlap)
    overlaps = {key: count for key, count in overlap_counts.items() if count}
    if overlaps:
        details = ", ".join(f"{pair}={count}" for pair, count in overlaps.items())
        raise ValueError(
            "Exact image hashes occur in multiple derived classification "
            f"partitions: {details}."
        )
    return overlap_counts


def class_distribution_report(
    partitions: dict[SplitName, list[ClassificationRecord]],
) -> dict[str, dict[str, int]]:
    """Count each fixed class in each derived partition, including zero counts."""
    return {
        split: {
            label: Counter(record.class_label for record in partitions[split]).get(
                label, 0
            )
            for label in CLASS_NAMES
        }
        for split in _SPLITS
    }


def save_class_mapping(path: Path) -> None:
    """Write the stable project class-to-index mapping as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(
            {"class_to_index": CLASS_TO_INDEX, "index_to_class": list(CLASS_NAMES)},
            output,
            indent=2,
        )
        output.write("\n")


class ClassificationDataset:
    """Map-style PyTorch dataset; augments training records only."""

    def __init__(
        self,
        records: list[ClassificationRecord],
        preprocessing: PreprocessingConfig,
        *,
        training: bool = False,
        rotation_degrees: float = 10.0,
        horizontal_flip_probability: float = 0.5,
    ) -> None:
        if rotation_degrees < 0:
            raise ValueError("rotation_degrees must be non-negative.")
        if not 0 <= horizontal_flip_probability <= 1:
            raise ValueError("horizontal_flip_probability must be between 0 and 1.")
        if training and rotation_degrees and preprocessing.color_mode == "preserve":
            raise ValueError(
                "Training augmentation requires a fixed grayscale or RGB "
                "preprocessing color mode, not 'preserve'."
            )
        self.records = tuple(records)
        self.preprocessing = preprocessing
        self.training = training
        self.rotation_degrees = rotation_degrees
        self.horizontal_flip_probability = horizontal_flip_probability

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        record = self.records[index]
        try:
            image = load_image(record.image_path, self.preprocessing)
            image_tensor = image_to_tensor(image)
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"Could not load classification image "
                f"{record.relative_image_path}: {error}"
            ) from error
        if self.training:
            image_tensor = self._augment(image_tensor)
        return image_tensor, record.label_index

    def _augment(self, image_tensor: Any) -> Any:
        torch = require_torch()
        functional = torch.nn.functional
        if random.random() < self.horizontal_flip_probability:
            image_tensor = torch.flip(image_tensor, dims=(-1,))
        if self.rotation_degrees:
            angle = math_radians(
                random.uniform(-self.rotation_degrees, self.rotation_degrees)
            )
            cosine = float(np.cos(angle))
            sine = float(np.sin(angle))
            transform = image_tensor.new_tensor(
                [[cosine, -sine, 0.0], [sine, cosine, 0.0]]
            ).unsqueeze(0)
            grid = functional.affine_grid(
                transform,
                (1, *image_tensor.shape),
                align_corners=False,
            )
            image_tensor = functional.grid_sample(
                image_tensor.unsqueeze(0),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).squeeze(0)
        return image_tensor.contiguous()


def math_radians(degrees: float) -> float:
    """Convert degrees to radians without importing an additional dependency."""
    return degrees * (np.pi / 180.0)


def seed_data_loader_worker(worker_id: int) -> None:
    """Seed Python and NumPy RNGs from each worker's PyTorch-assigned seed."""
    torch = require_torch()
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_dataloaders(
    manifest_path: Path,
    dataset_root: Path | None,
    preprocessing: PreprocessingConfig,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    rotation_degrees: float = 10.0,
    horizontal_flip_probability: float = 0.5,
) -> tuple[dict[str, Any], dict[str, list[ClassificationRecord]]]:
    """Build train/validation/test loaders after validating derived manifests."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    partitions = read_classification_manifest(manifest_path, dataset_root)
    torch = require_torch()
    generator = torch.Generator()
    generator.manual_seed(seed)
    loaders: dict[str, Any] = {}
    for split in _SPLITS:
        is_training = split == "training"
        dataset = ClassificationDataset(
            partitions[split],
            preprocessing,
            training=is_training,
            rotation_degrees=rotation_degrees,
            horizontal_flip_probability=horizontal_flip_probability,
        )
        loaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_training,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_data_loader_worker,
            generator=generator if is_training else None,
            persistent_workers=num_workers > 0,
        )
    return loaders, partitions

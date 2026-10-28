"""Leakage-safe BRISC image-mask datasets and synchronized augmentation."""

from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import numpy as np
from PIL import Image

from mri_tumor.data.config import DatasetPaths, PROJECT_ROOT
from mri_tumor.preprocessing import (
    PreprocessingConfig,
    image_to_tensor,
    load_segmentation_pair,
)

SplitName = Literal["training", "validation", "testing"]
_SPLITS: tuple[SplitName, ...] = ("training", "validation", "testing")
_ALLOWED_FLAGS = {"conflicting_masks_for_identical_image"}


def require_torch() -> Any:
    """Load PyTorch lazily and give an actionable dependency error."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for segmentation. Install a system-appropriate "
            "build using the commands in README.md."
        ) from error
    return torch


def latest_segmentation_manifest(
    manifest_dir: Path = PROJECT_ROOT / "results" / "data_manifests",
) -> Path:
    """Find the latest derived segmentation manifest; never use official splits."""
    directory = manifest_dir.expanduser().resolve()
    candidates = sorted(directory.glob("leakage_audit_*_segmentation.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No derived segmentation manifest exists in {directory}. Run "
            "scripts/audit_dataset_leakage.py first; there is no official-split fallback."
        )
    return candidates[-1]


@dataclass(frozen=True)
class SegmentationRecord:
    """One manifest-verified MRI and its explicitly linked categorical mask."""

    image_path: Path
    mask_path: Path
    relative_image_path: str
    relative_mask_path: str
    class_label: str
    image_sha256: str
    duplicate_group_id: str
    split: SplitName


def _safe_path(root: Path, relative_path: str, *, field: str) -> Path:
    normalized = relative_path.strip().replace("\\", "/")
    relative = PurePosixPath(normalized)
    if not normalized or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe manifest {field}: {relative_path!r}")
    resolved_root = root.expanduser().resolve()
    path = resolved_root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(resolved_root):
        raise ValueError(f"Manifest {field} escapes dataset root: {relative_path!r}")
    if not path.is_file():
        raise FileNotFoundError(f"Manifest {field} does not exist: {path}")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _mask_content_hash(path: Path) -> str:
    try:
        with Image.open(path) as source:
            source.load()
            pixels = np.asarray(source.convert("RGBA"))
            digest = hashlib.sha256()
            digest.update(f"{pixels.shape[1]}x{pixels.shape[0]}:RGBA:".encode("ascii"))
            digest.update(pixels.tobytes())
            return digest.hexdigest()
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise ValueError(f"Could not decode segmentation mask {path}: {error}") from error


def binary_target(mask: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Map binary-mask intensities and antialiased edges to 0/1 labels."""
    if threshold < 1 or threshold > 255:
        raise ValueError("mask_threshold must be in [1, 255].")
    if mask.ndim != 2 or mask.size == 0 or not np.issubdtype(mask.dtype, np.number):
        raise ValueError("Segmentation masks must be non-empty 2D numeric arrays.")
    if not np.isfinite(mask).all() or mask.min() < 0 or mask.max() > 255:
        raise ValueError("BRISC binary-mask intensities must be finite values in [0, 255].")
    return np.ascontiguousarray(mask >= threshold, dtype=np.uint8)


def read_segmentation_manifest(
    manifest_path: Path,
    dataset_root: Path | None = None,
    *,
    preprocessing: PreprocessingConfig = PreprocessingConfig(),
    mask_threshold: int = 128,
) -> tuple[dict[SplitName, list[SegmentationRecord]], dict[str, Any]]:
    """Validate all eligible linked pairs and exclude only explicitly flagged groups."""
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Derived segmentation manifest does not exist: {manifest_path}. "
            "Run scripts/audit_dataset_leakage.py; no official-split fallback is used."
        )
    root = (
        DatasetPaths.from_environment().require("segmentation")
        if dataset_root is None
        else dataset_root.expanduser().resolve()
    )
    required = {
        "relative_mri_path",
        "relative_mask_path",
        "class_label",
        "image_sha256",
        "duplicate_group_id",
        "proposed_derived_split",
        "review_flags",
    }
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            missing_columns = required - set(reader.fieldnames or ())
            if missing_columns:
                raise ValueError(
                    "Derived segmentation manifest is missing required columns: "
                    + ", ".join(sorted(missing_columns))
                )
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"Could not read segmentation manifest {manifest_path}: {error}") from error

    partitions: dict[SplitName, list[SegmentationRecord]] = {
        split: [] for split in _SPLITS
    }
    excluded_by_flag: dict[str, int] = defaultdict(int)
    eligible_rows: list[tuple[int, dict[str, str], SplitName]] = []
    group_rows: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)

    for line, row in enumerate(rows, start=2):
        split = row["proposed_derived_split"].strip().casefold()
        flags = {
            flag.strip()
            for flag in row["review_flags"].split(";")
            if flag.strip()
        }
        if split == "flagged" or flags:
            if not flags:
                raise ValueError(
                    f"Flagged manifest row {line} has no review reason; review the audit."
                )
            for flag in flags:
                excluded_by_flag[flag] += 1
            group_id = row["duplicate_group_id"].strip()
            if group_id:
                group_rows[group_id].append((line, row))
            continue
        if split not in _SPLITS:
            raise ValueError(
                f"Unsupported derived split {split!r} on manifest line {line}; "
                "only training, validation, testing, or explicitly flagged rows are valid."
            )
        eligible_rows.append((line, row, cast(SplitName, split)))
        group_id = row["duplicate_group_id"].strip()
        if group_id:
            group_rows[group_id].append((line, row))

    for group_id, members in group_rows.items():
        member_splits = {
            row["proposed_derived_split"].strip().casefold()
            for _, row in members
        }
        conflict_rows = [
            row
            for _, row in members
            if _ALLOWED_FLAGS.intersection(
                flag.strip() for flag in row["review_flags"].split(";") if flag.strip()
            )
        ]
        if conflict_rows and (
            member_splits != {"flagged"} or len(conflict_rows) != len(members)
        ):
            raise ValueError(
                f"Conflicting-mask group {group_id} is not fully marked flagged."
            )
        assigned = member_splits & set(_SPLITS)
        if len(assigned) > 1:
            raise ValueError(
                f"Duplicate MRI group {group_id} crosses derived partitions: "
                f"{sorted(assigned)}."
            )

    hash_to_splits: dict[str, set[str]] = defaultdict(set)
    group_mask_hashes: dict[str, set[str]] = defaultdict(set)
    group_mask_records: dict[str, list[tuple[int, dict[str, str], str]]] = defaultdict(list)
    for line, row, split in eligible_rows:
        relative_image = row["relative_mri_path"].strip()
        relative_mask = row["relative_mask_path"].strip()
        image_path = _safe_path(root, relative_image, field="relative_mri_path")
        mask_path = _safe_path(root, relative_mask, field="relative_mask_path")
        expected_hash = row["image_sha256"].strip().casefold()
        class_label = row["class_label"].strip().casefold()
        if not class_label:
            raise ValueError(f"Missing BRISC tumor class on manifest line {line}.")
        if len(expected_hash) != 64 or any(
            character not in "0123456789abcdef" for character in expected_hash
        ):
            raise ValueError(
                f"Invalid image SHA-256 on manifest line {line}: {expected_hash!r}"
            )
        actual_hash = _file_sha256(image_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"MRI hash does not match manifest line {line}: {relative_image}"
            )
        try:
            _, mask = load_segmentation_pair(
                image_path, mask_path, preprocessing, resize=False
            )
            target = binary_target(mask, mask_threshold)
        except (OSError, ValueError) as error:
            raise ValueError(
                f"Invalid MRI-mask pair on manifest line {line} "
                f"({relative_image}, {relative_mask}): {error}"
            ) from error
        mask_hash = _mask_content_hash(mask_path)
        group_id = row["duplicate_group_id"].strip() or f"sha256-{actual_hash[:16]}"
        group_mask_hashes[group_id].add(mask_hash)
        group_mask_records[group_id].append((line, row, mask_hash))
        hash_to_splits[actual_hash].add(split)
        partitions[split].append(
            SegmentationRecord(
                image_path=image_path,
                mask_path=mask_path,
                relative_image_path=relative_image.replace("\\", "/"),
                relative_mask_path=relative_mask.replace("\\", "/"),
                class_label=class_label,
                image_sha256=actual_hash,
                duplicate_group_id=group_id,
                split=split,
            )
        )
        # Decode the thresholded target here so invalid or empty targets are visible.
        if target.size == 0:
            raise ValueError(f"Empty decoded mask on manifest line {line}.")

    unresolved = {
        group_id: hashes
        for group_id, hashes in group_mask_hashes.items()
        if len(hashes) > 1
    }
    if unresolved:
        raise ValueError(
            "Unresolved conflicting-mask duplicate groups remain unflagged: "
            + ", ".join(sorted(unresolved)[:10])
        )

    split_hashes = {split: set() for split in _SPLITS}
    for split, records in partitions.items():
        if not records:
            raise ValueError(f"The derived segmentation manifest has no eligible {split} pairs.")
        for record in records:
            split_hashes[split].add(record.image_sha256)
        partitions[split].sort(key=lambda record: record.relative_image_path.casefold())
    overlaps = {
        f"{first}:{second}": len(split_hashes[first] & split_hashes[second])
        for index, first in enumerate(_SPLITS)
        for second in _SPLITS[index + 1 :]
    }
    if any(overlaps.values()):
        raise ValueError(f"Exact MRI hashes cross derived partitions: {overlaps}")

    summary = {
        "manifest_path": str(manifest_path),
        "dataset_root": str(root),
        "manifest_row_count": len(rows),
        "eligible_pair_counts": {split: len(partitions[split]) for split in _SPLITS},
        "excluded_flagged_pair_count": len(rows) - len(eligible_rows),
        "excluded_by_review_flag": dict(sorted(excluded_by_flag.items())),
        "exact_hash_overlap_counts": overlaps,
        "mask_formulation": "binary; grayscale threshold >= configured mask_threshold is tumor",
        "mask_threshold": mask_threshold,
    }
    return partitions, summary


class SegmentationDataset:
    """Load linked pairs; spatial augmentation is enabled only for training."""

    def __init__(
        self,
        records: list[SegmentationRecord],
        preprocessing: PreprocessingConfig,
        *,
        training: bool = False,
        mask_threshold: int = 128,
        rotation_degrees: float = 10.0,
        horizontal_flip_probability: float = 0.5,
    ) -> None:
        if rotation_degrees < 0:
            raise ValueError("rotation_degrees must be non-negative.")
        if not 0 <= horizontal_flip_probability <= 1:
            raise ValueError("horizontal_flip_probability must be in [0, 1].")
        self.records = tuple(records)
        self.preprocessing = preprocessing
        self.training = training
        self.mask_threshold = mask_threshold
        self.rotation_degrees = rotation_degrees
        self.horizontal_flip_probability = horizontal_flip_probability

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        torch = require_torch()
        record = self.records[index]
        try:
            image, mask = load_segmentation_pair(
                record.image_path, record.mask_path, self.preprocessing
            )
            image_tensor = image_to_tensor(image)
            target = torch.from_numpy(
                np.ascontiguousarray(
                    binary_target(mask, self.mask_threshold)[None, ...],
                    dtype=np.float32,
                )
            )
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"Could not load BRISC pair {record.relative_image_path} and "
                f"{record.relative_mask_path}: {error}"
            ) from error
        if self.training:
            image_tensor, target = self._augment(image_tensor, target)
        return image_tensor, target

    def _augment(self, image: Any, target: Any) -> tuple[Any, Any]:
        torch = require_torch()
        if random.random() < self.horizontal_flip_probability:
            image = torch.flip(image, dims=(-1,))
            target = torch.flip(target, dims=(-1,))
        if self.rotation_degrees:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            radians = angle * np.pi / 180.0
            cosine, sine = float(np.cos(radians)), float(np.sin(radians))
            transform = image.new_tensor(
                [[cosine, -sine, 0.0], [sine, cosine, 0.0]]
            ).unsqueeze(0)
            grid = torch.nn.functional.affine_grid(
                transform, (1, *image.shape), align_corners=False
            )
            image = torch.nn.functional.grid_sample(
                image.unsqueeze(0),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).squeeze(0)
            target = torch.nn.functional.grid_sample(
                target.unsqueeze(0),
                grid,
                mode="nearest",
                padding_mode="zeros",
                align_corners=False,
            ).squeeze(0)
        return image.contiguous(), target.contiguous()


def seed_data_loader_worker(worker_id: int) -> None:
    """Seed Python and NumPy augmentation RNGs from the PyTorch worker seed."""
    del worker_id
    torch = require_torch()
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def create_dataloaders(
    partitions: dict[SplitName, list[SegmentationRecord]],
    preprocessing: PreprocessingConfig,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    mask_threshold: int = 128,
    rotation_degrees: float = 10.0,
    horizontal_flip_probability: float = 0.5,
) -> dict[str, Any]:
    """Create train/validation/test loaders; only training shuffles/augments."""
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative.")
    torch = require_torch()
    generator = torch.Generator().manual_seed(seed)
    loaders: dict[str, Any] = {}
    for split in _SPLITS:
        is_training = split == "training"
        dataset = SegmentationDataset(
            partitions[split],
            preprocessing,
            training=is_training,
            mask_threshold=mask_threshold,
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
    return loaders

"""Create small, manifest-driven MRI dataset and preprocessing visualizations."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from mri_tumor.data.config import DatasetPaths, PROJECT_ROOT
from mri_tumor.preprocessing import (
    PreprocessingConfig,
    PreprocessingError,
    load_image,
    load_mask,
    load_segmentation_pair,
)

_DERIVED_SPLITS = {"training", "validation", "testing"}


def find_latest_manifest_pair(manifest_dir: Path) -> tuple[Path, Path]:
    """Find the latest complete timestamped classification/segmentation pair."""
    directory = manifest_dir.expanduser().resolve()
    candidates = sorted(directory.glob("leakage_audit_*_segmentation.csv"))
    for segmentation_manifest in reversed(candidates):
        prefix = segmentation_manifest.name.removesuffix("_segmentation.csv")
        classification_manifest = directory / f"{prefix}_classification.csv"
        if classification_manifest.is_file():
            return classification_manifest, segmentation_manifest
    raise FileNotFoundError(
        f"No matching derived classification and segmentation manifests found "
        f"in {directory}. Run scripts/audit_dataset_leakage.py first; the "
        "visualizer will not fall back to official dataset splits."
    )


def read_manifest(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    """Read and validate a derived CSV manifest."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            missing_columns = required_columns - set(reader.fieldnames or ())
            if missing_columns:
                raise ValueError(
                    f"Manifest {path} is missing required columns: "
                    f"{', '.join(sorted(missing_columns))}"
                )
            return list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"Could not read manifest {path}: {error}") from error


def eligible_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep only unflagged rows assigned to a derived partition."""
    return [
        row
        for row in rows
        if row.get("proposed_derived_split", "").strip() in _DERIVED_SPLITS
        and not row.get("review_flags", "").strip()
    ]


def _safe_manifest_path(root: Path, relative_path: str) -> Path:
    normalized = relative_path.strip().replace("\\", "/")
    relative = PurePosixPath(normalized)
    if not normalized or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Manifest path is not a safe relative path: {relative_path!r}")
    resolved_root = root.expanduser().resolve()
    resolved_path = resolved_root.joinpath(*relative.parts).resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"Manifest path escapes its dataset root: {relative_path!r}")
    return resolved_path


def _next_figure_path(figures_dir: Path, filename: str) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    destination = figures_dir / filename
    if not destination.exists():
        return destination
    stem, suffix = destination.stem, destination.suffix
    index = 1
    while True:
        candidate = figures_dir / f"{stem}_{index:03d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _display_pixels(image: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    pixels = image[..., 0] if image.shape[-1] == 1 else image
    if config.image_normalization == "minus_one_one":
        pixels = (pixels + 1.0) / 2.0
    return pixels


def _save_figure(
    figure: plt.Figure, figures_dir: Path, filename: str, outputs: list[Path]
) -> None:
    path = _next_figure_path(figures_dir, filename)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    outputs.append(path)


def _classification_samples(
    rows: list[dict[str, str]],
    dataset_root: Path,
    config: PreprocessingConfig,
    figures_dir: Path,
    samples_per_class: int,
    seed: int,
    outputs: list[Path],
    problems: list[str],
) -> None:
    training_rows = [
        row for row in eligible_rows(rows)
        if row["proposed_derived_split"] == "training"
    ]
    labels = sorted({row["class_label"] for row in rows if row["class_label"]})
    if not labels:
        raise ValueError("The classification manifest contains no class labels.")

    chosen: dict[str, list[dict[str, str]]] = {}
    for label_index, label in enumerate(labels):
        candidates = sorted(
            (row for row in training_rows if row["class_label"] == label),
            key=lambda row: row["relative_image_path"].casefold(),
        )
        if not candidates:
            problems.append(
                f"No eligible derived-training classification images for class {label}."
            )
            chosen[label] = []
            continue
        rng = random.Random(seed + label_index)
        chosen[label] = sorted(
            rng.sample(candidates, min(samples_per_class, len(candidates))),
            key=lambda row: row["relative_image_path"].casefold(),
        )

    figure, axes = plt.subplots(
        len(labels),
        samples_per_class,
        squeeze=False,
        figsize=(3.2 * samples_per_class, 2.9 * len(labels)),
    )
    for row_index, label in enumerate(labels):
        for column_index in range(samples_per_class):
            axis = axes[row_index][column_index]
            if column_index >= len(chosen[label]):
                axis.axis("off")
                continue
            record = chosen[label][column_index]
            try:
                image = load_image(
                    _safe_manifest_path(
                        dataset_root, record["relative_image_path"]
                    ),
                    config,
                )
                axis.imshow(_display_pixels(image, config), cmap="gray", vmin=0, vmax=1)
                axis.set_title(
                    f"{label} — {record['proposed_derived_split']}"
                    if column_index == 0
                    else record["proposed_derived_split"]
                )
            except (OSError, ValueError, PreprocessingError) as error:
                problems.append(
                    f"Classification image {record['relative_image_path']}: {error}"
                )
                axis.set_title("Unreadable image")
            axis.axis("off")
    figure.suptitle("Classification MRI samples (derived training split)")
    _save_figure(
        figure, figures_dir, "classification_samples_by_class.png", outputs
    )


def _classification_before_after(
    rows: list[dict[str, str]],
    dataset_root: Path,
    config: PreprocessingConfig,
    figures_dir: Path,
    outputs: list[Path],
    problems: list[str],
) -> None:
    labels = sorted({row["class_label"] for row in rows if row["class_label"]})
    figure, axes = plt.subplots(
        len(labels), 2, squeeze=False, figsize=(7, 3.0 * len(labels))
    )
    training_rows = [
        row for row in eligible_rows(rows)
        if row["proposed_derived_split"] == "training"
    ]
    for row_index, label in enumerate(labels):
        candidates = sorted(
            (row for row in training_rows if row["class_label"] == label),
            key=lambda row: row["relative_image_path"].casefold(),
        )
        for column_index, axis in enumerate(axes[row_index]):
            axis.axis("off")
        if not candidates:
            problems.append(
                f"No eligible derived-training classification image for class {label}."
            )
            continue
        record = candidates[0]
        try:
            image_path = _safe_manifest_path(
                dataset_root, record["relative_image_path"]
            )
            original = load_image(image_path, config, resize=False)
            processed = load_image(image_path, config, resize=True)
            for axis, pixels, title in (
                (axes[row_index][0], original, "Original"),
                (
                    axes[row_index][1],
                    processed,
                    f"Preprocessed {config.image_size[0]}×{config.image_size[1]}",
                ),
            ):
                axis.imshow(_display_pixels(pixels, config), cmap="gray", vmin=0, vmax=1)
                axis.set_title(f"{label} — {title}")
                axis.axis("off")
        except (OSError, ValueError, PreprocessingError) as error:
            problems.append(
                f"Classification image {record['relative_image_path']}: {error}"
            )
    figure.suptitle("MRI before and after configured preprocessing")
    _save_figure(
        figure,
        figures_dir,
        "classification_preprocessing_before_after.png",
        outputs,
    )


def _class_distribution(
    rows: list[dict[str, str]],
    *,
    path: str,
    title: str,
    flagged_count: int,
    figures_dir: Path,
    outputs: list[Path],
) -> None:
    labels = sorted({row["class_label"] for row in rows if row["class_label"]})
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in eligible_rows(rows):
        counts[row["class_label"]][row["proposed_derived_split"]] += 1
    splits = [name for name in ("training", "validation", "testing") if any(
        counts[label][name] for label in labels
    )]
    figure, axis = plt.subplots(figsize=(max(7, 1.7 * len(labels)), 4.5))
    x_positions = np.arange(len(labels))
    bar_width = 0.8 / max(1, len(splits))
    for split_index, split in enumerate(splits):
        offsets = (
            x_positions
            - 0.4
            + bar_width / 2
            + split_index * bar_width
        )
        axis.bar(
            offsets,
            [counts[label][split] for label in labels],
            bar_width,
            label=split,
        )
    axis.set_xticks(x_positions, labels, rotation=20, ha="right")
    axis.set_ylabel("Images" if "classification" in path else "MRI/mask pairs")
    axis.set_title(title)
    axis.legend()
    if flagged_count:
        axis.set_xlabel(f"Flagged rows excluded: {flagged_count}")
    figure.tight_layout()
    _save_figure(figure, figures_dir, path, outputs)


def _coverage_examples(
    rows: list[dict[str, str]],
    dataset_root: Path,
    config: PreprocessingConfig,
    figures_dir: Path,
    seed: int,
    example_count: int,
    outputs: list[Path],
    problems: list[str],
) -> None:
    training_rows = [
        row for row in eligible_rows(rows)
        if row["proposed_derived_split"] == "training"
    ]
    if not training_rows:
        raise ValueError(
            "The segmentation manifest contains no eligible derived-training pairs."
        )
    rng = random.Random(seed)
    candidates = sorted(
        training_rows, key=lambda row: row["relative_mri_path"].casefold()
    )
    candidates = rng.sample(candidates, min(96, len(candidates)))
    measured: list[tuple[float, dict[str, str]]] = []
    for record in candidates:
        try:
            mask = load_mask(
                _safe_manifest_path(dataset_root, record["relative_mask_path"]),
                config,
                resize=False,
            )
        except (OSError, ValueError, PreprocessingError) as error:
            problems.append(
                f"Segmentation mask {record['relative_mask_path']}: {error}"
            )
            continue
        coverage = float(
            np.count_nonzero(mask != config.mask_background_value) / mask.size
        )
        measured.append((coverage, record))
    if not measured:
        raise ValueError("No readable, valid, unflagged segmentation masks were found.")

    measured.sort(
        key=lambda item: (item[0], item[1]["relative_mri_path"].casefold())
    )
    selected_indexes = sorted(
        {
            round(index * (len(measured) - 1) / max(1, example_count - 1))
            for index in range(example_count)
        }
    )
    selected = [measured[index] for index in selected_indexes]
    figure, axes = plt.subplots(
        len(selected), 3, squeeze=False, figsize=(10.5, 3.5 * len(selected))
    )
    for row_index, (coverage, record) in enumerate(selected):
        try:
            image, mask = load_segmentation_pair(
                _safe_manifest_path(dataset_root, record["relative_mri_path"]),
                _safe_manifest_path(dataset_root, record["relative_mask_path"]),
                config,
                resize=False,
            )
        except (OSError, ValueError, PreprocessingError) as error:
            problems.append(
                f"Segmentation pair {record['relative_mri_path']} / "
                f"{record['relative_mask_path']}: {error}"
            )
            for axis in axes[row_index]:
                axis.axis("off")
            continue

        foreground = mask != config.mask_background_value
        axes[row_index][0].imshow(
            _display_pixels(image, config), cmap="gray", vmin=0, vmax=1
        )
        axes[row_index][0].set_title(f"Original MRI — {record['class_label']}")
        axes[row_index][1].imshow(mask, cmap="gray")
        axes[row_index][1].set_title("Ground-truth mask")
        axes[row_index][2].imshow(
            _display_pixels(image, config), cmap="gray", vmin=0, vmax=1
        )
        axes[row_index][2].imshow(
            np.ma.masked_where(~foreground, foreground),
            cmap=matplotlib.colors.ListedColormap(["red"]),
            alpha=0.48,
            vmin=0,
            vmax=1,
        )
        axes[row_index][2].set_title(
            f"Overlay — {coverage:.2%} foreground"
        )
        for axis in axes[row_index]:
            axis.axis("off")
    figure.suptitle(
        "BRISC derived-training examples at different foreground coverage"
    )
    _save_figure(
        figure,
        figures_dir,
        "brisc_segmentation_coverage_examples.png",
        outputs,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-manifest", type=Path)
    parser.add_argument("--segmentation-manifest", type=Path)
    parser.add_argument("--classification-root", type=Path)
    parser.add_argument("--segmentation-root", type=Path)
    parser.add_argument(
        "--manifest-dir", type=Path, default=Path("results/data_manifests")
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=Path("results/figures")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/preprocessing.json")
    )
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--segmentation-examples", type=int, default=3)
    parser.add_argument("--seed", type=int)
    arguments = parser.parse_args(argv)
    if arguments.samples_per_class < 1 or arguments.segmentation_examples < 1:
        parser.error("--samples-per-class and --segmentation-examples must be positive.")

    manifest_dir = (
        arguments.manifest_dir
        if arguments.manifest_dir.is_absolute()
        else PROJECT_ROOT / arguments.manifest_dir
    )
    if arguments.classification_manifest and arguments.segmentation_manifest:
        classification_manifest = arguments.classification_manifest
        segmentation_manifest = arguments.segmentation_manifest
    elif arguments.classification_manifest or arguments.segmentation_manifest:
        parser.error(
            "Supply both --classification-manifest and --segmentation-manifest, "
            "or neither to select the latest matching derived pair."
        )
    else:
        classification_manifest, segmentation_manifest = find_latest_manifest_pair(
            manifest_dir
        )

    dataset_paths = DatasetPaths.from_environment()
    classification_root = (
        arguments.classification_root
        if arguments.classification_root is not None
        else dataset_paths.classification
    )
    segmentation_root = (
        arguments.segmentation_root
        if arguments.segmentation_root is not None
        else dataset_paths.segmentation
    )
    for root in (classification_root, segmentation_root):
        if not root.expanduser().is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    config_path = arguments.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = PreprocessingConfig.from_json(config_path)
    seed = config.random_seed if arguments.seed is None else arguments.seed

    classification_rows = read_manifest(
        classification_manifest,
        {
            "relative_image_path",
            "class_label",
            "proposed_derived_split",
            "review_flags",
        },
    )
    segmentation_rows = read_manifest(
        segmentation_manifest,
        {
            "relative_mri_path",
            "relative_mask_path",
            "class_label",
            "proposed_derived_split",
            "review_flags",
        },
    )
    eligible_segmentation = eligible_rows(segmentation_rows)
    flagged_segmentation_count = sum(
        row.get("proposed_derived_split", "").strip() == "flagged"
        or bool(row.get("review_flags", "").strip())
        for row in segmentation_rows
    )
    figures_dir = arguments.figures_dir
    if not figures_dir.is_absolute():
        figures_dir = PROJECT_ROOT / figures_dir

    outputs: list[Path] = []
    problems: list[str] = []
    _classification_samples(
        classification_rows,
        classification_root,
        config,
        figures_dir,
        arguments.samples_per_class,
        seed,
        outputs,
        problems,
    )
    _classification_before_after(
        classification_rows,
        classification_root,
        config,
        figures_dir,
        outputs,
        problems,
    )
    _class_distribution(
        classification_rows,
        path="classification_class_distribution.png",
        title="Classification class distribution by derived split",
        flagged_count=sum(
            row.get("proposed_derived_split", "").strip() == "flagged"
            or bool(row.get("review_flags", "").strip())
            for row in classification_rows
        ),
        figures_dir=figures_dir,
        outputs=outputs,
    )
    _class_distribution(
        segmentation_rows,
        path="segmentation_class_distribution.png",
        title="BRISC class distribution by derived split (unflagged pairs only)",
        flagged_count=flagged_segmentation_count,
        figures_dir=figures_dir,
        outputs=outputs,
    )
    _coverage_examples(
        segmentation_rows,
        segmentation_root,
        config,
        figures_dir,
        seed,
        arguments.segmentation_examples,
        outputs,
        problems,
    )

    print(f"Classification manifest: {classification_manifest}")
    print(f"Segmentation manifest: {segmentation_manifest}")
    print(
        f"Segmentation visualization: {len(eligible_segmentation)} eligible pairs; "
        f"{flagged_segmentation_count} flagged pairs excluded."
    )
    for path in outputs:
        print(f"Figure: {path}")
    if problems:
        print(f"Visualization issues ({len(problems)}):")
        for problem in problems:
            print(f"  REVIEW: {problem}")
    else:
        print("Visualization issues: none in the selected sample.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

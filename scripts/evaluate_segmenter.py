"""Evaluate a validation-selected U-Net on the derived testing partition."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from mri_tumor.preprocessing import PreprocessingConfig
from mri_tumor.segmentation.checkpoint import load_segmentation_checkpoint
from mri_tumor.segmentation.data import (
    SegmentationDataset,
    read_segmentation_manifest,
    require_torch,
)
from mri_tumor.segmentation.metrics import (
    aggregate_segmentation_metrics,
    binary_segmentation_metrics,
)
from mri_tumor.segmentation.model import create_unet
from mri_tumor.utils.device import get_device


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, ensure_ascii=False)
        output.write("\n")


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    manifest_path: Path | None = None,
    dataset_root: Path | None = None,
    output_dir: Path | None = None,
    device_preference: str = "auto",
    visualization_count: int = 12,
) -> dict[str, Any]:
    """Evaluate all eligible derived-test pairs and save per-image/aggregate outputs."""
    if visualization_count < 0:
        raise ValueError("visualization_count must be non-negative.")
    checkpoint_path = checkpoint_path.expanduser().resolve()
    torch = require_torch()
    device = get_device(device_preference)
    checkpoint = load_segmentation_checkpoint(checkpoint_path, map_location=device)
    required = {
        "model_state_dict",
        "model_config",
        "preprocessing_config",
        "training_config",
        "manifest_path",
        "dataset_root",
        "best_epoch",
    }
    if not required.issubset(checkpoint):
        raise ValueError("Checkpoint is incomplete; it cannot be evaluated safely.")
    manifest_path = (
        Path(checkpoint["manifest_path"]) if manifest_path is None else manifest_path
    )
    dataset_root = (
        Path(checkpoint["dataset_root"]) if dataset_root is None else dataset_root
    )
    preprocessing = PreprocessingConfig(**checkpoint["preprocessing_config"])
    mask_threshold = int(checkpoint["training_config"]["mask_threshold"])
    partitions, manifest_summary = read_segmentation_manifest(
        manifest_path,
        dataset_root,
        preprocessing=preprocessing,
        mask_threshold=mask_threshold,
    )
    testing = partitions["testing"]
    dataset = SegmentationDataset(
        testing,
        preprocessing,
        training=False,
        mask_threshold=mask_threshold,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(checkpoint["training_config"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    model = create_unet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    per_image: list[dict[str, float | int]] = []
    visual_samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
    with torch.inference_mode():
        for batch_images, batch_targets in loader:
            logits = model(batch_images.to(device))
            predictions = torch.sigmoid(logits) >= 0.5
            targets = batch_targets >= 0.5
            for image, target, prediction in zip(
                batch_images, targets, predictions.cpu(), strict=True
            ):
                ground_truth = target[0].numpy().astype(bool)
                predicted_mask = prediction[0].numpy().astype(bool)
                per_image.append(
                    binary_segmentation_metrics(ground_truth, predicted_mask)
                )
                if len(visual_samples) < visualization_count:
                    array = image.numpy()
                    visual_image = np.transpose(array, (1, 2, 0))
                    if visual_image.shape[-1] == 1:
                        visual_image = visual_image[..., 0]
                    visual_samples.append(
                        (
                            visual_image,
                            ground_truth,
                            predicted_mask,
                            testing[len(per_image) - 1].class_label,
                        )
                    )
    if len(per_image) != len(testing):
        raise RuntimeError(
            f"Test loader returned {len(per_image)} images for "
            f"{len(testing)} derived testing pairs."
        )
    aggregate = aggregate_segmentation_metrics(per_image)
    metrics: dict[str, Any] = {
        **aggregate,
        "per_image_metrics_csv": "per_image_metrics.csv",
        "evaluated_split": "testing",
        "checkpoint_path": str(checkpoint_path),
        "manifest_path": str(Path(manifest_path).expanduser().resolve()),
        "best_epoch_selected_on_validation": int(checkpoint["best_epoch"]),
        "best_validation_mean_dice": float(checkpoint["best_validation_mean_dice"]),
        "device": str(device),
        "mask_formulation": manifest_summary["mask_formulation"],
        "mask_threshold": mask_threshold,
        "prototype_notice": "Academic research prototype; not clinically validated.",
    }
    destination = (
        checkpoint_path.parent
        if output_dir is None
        else output_dir.expanduser().resolve()
    )
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "test_metrics.json", metrics)
    with (destination / "per_image_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "test_sample_index",
                "tumor_class",
                "dice",
                "iou",
                "precision",
                "recall",
                "true_positive_pixels",
                "ground_truth_pixels",
                "predicted_pixels",
            ],
        )
        writer.writeheader()
        for index, row in enumerate(per_image):
            writer.writerow(
                {
                    "test_sample_index": index,
                    "tumor_class": testing[index].class_label,
                    **row,
                }
            )
    if visual_samples:
        columns = 4
        rows = len(visual_samples)
        figure, axes = plt.subplots(rows, columns, figsize=(12, max(3 * rows, 4)))
        axes = np.asarray(axes).reshape(rows, columns)
        for index, (image, truth, prediction, tumor_class) in enumerate(visual_samples):
            axis = axes[index]
            axis[0].imshow(image, cmap="gray" if image.ndim == 2 else None)
            axis[0].set_title(f"MRI ({tumor_class})")
            axis[1].imshow(truth, cmap="gray", vmin=0, vmax=1)
            axis[1].set_title("Ground truth")
            axis[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
            axis[2].set_title("Prediction")
            axis[3].imshow(image, cmap="gray" if image.ndim == 2 else None)
            axis[3].imshow(np.ma.masked_where(~truth, truth), cmap="Greens", alpha=0.65)
            axis[3].imshow(
                np.ma.masked_where(~prediction, prediction),
                cmap="Reds",
                alpha=0.5,
            )
            axis[3].set_title("Overlay (GT green, prediction red)")
            for current in axis:
                current.axis("off")
        figure.tight_layout()
        figure.savefig(destination / "test_visualizations.png", dpi=140)
        plt.close(figure)
    print(
        f"Derived testing: images={len(testing)} mean Dice={aggregate['mean_dice']:.5f} "
        f"mean IoU={aggregate['mean_iou']:.5f} device={device}"
    )
    print(f"Metrics and figures saved under {destination}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--visualization-count", type=int, default=12)
    arguments = parser.parse_args(argv)
    if arguments.manifest is None and not arguments.checkpoint.expanduser().is_file():
        parser.error(f"Checkpoint not found: {arguments.checkpoint}")
    evaluate_checkpoint(
        arguments.checkpoint,
        manifest_path=arguments.manifest,
        dataset_root=arguments.dataset_root,
        output_dir=arguments.output_dir,
        device_preference=arguments.device,
        visualization_count=arguments.visualization_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

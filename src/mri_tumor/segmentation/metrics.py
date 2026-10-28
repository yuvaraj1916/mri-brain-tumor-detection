"""Binary tumor-region metrics with explicit empty-mask handling."""

from __future__ import annotations

from typing import Any

import numpy as np


def _binary_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite array.")
    return array.astype(bool)


def binary_segmentation_metrics(
    ground_truth: Any, prediction: Any
) -> dict[str, float | int]:
    """Calculate one mask's pixel metrics; jointly empty masks score 1.0."""
    truth = _binary_array(ground_truth, "ground_truth")
    predicted = _binary_array(prediction, "prediction")
    if truth.shape != predicted.shape:
        raise ValueError(
            f"Ground-truth and prediction shapes differ: {truth.shape} vs {predicted.shape}."
        )
    true_positive = int(np.count_nonzero(truth & predicted))
    ground_truth_positive = int(np.count_nonzero(truth))
    predicted_positive = int(np.count_nonzero(predicted))
    union = ground_truth_positive + predicted_positive - true_positive
    if union == 0:
        dice = iou = precision = recall = 1.0
    else:
        dice = 2.0 * true_positive / (ground_truth_positive + predicted_positive)
        iou = true_positive / union
        precision = (
            true_positive / predicted_positive if predicted_positive else 0.0
        )
        recall = (
            true_positive / ground_truth_positive
            if ground_truth_positive
            else 0.0
        )
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "true_positive_pixels": true_positive,
        "ground_truth_pixels": ground_truth_positive,
        "predicted_pixels": predicted_positive,
    }


def aggregate_segmentation_metrics(
    per_image: list[dict[str, float | int]],
) -> dict[str, float | int]:
    """Report macro per-image means and pixel-micro aggregate scores."""
    if not per_image:
        raise ValueError("At least one image is required for metric aggregation.")
    metric_names = ("dice", "iou", "precision", "recall")
    result: dict[str, float | int] = {
        f"mean_{name}": float(np.mean([float(row[name]) for row in per_image]))
        for name in metric_names
    }
    tp = sum(int(row["true_positive_pixels"]) for row in per_image)
    gt = sum(int(row["ground_truth_pixels"]) for row in per_image)
    pred = sum(int(row["predicted_pixels"]) for row in per_image)
    union = gt + pred - tp
    if union == 0:
        micro_dice = micro_iou = micro_precision = micro_recall = 1.0
    else:
        micro_dice = 2 * tp / (gt + pred) if gt + pred else 1.0
        micro_iou = tp / union
        micro_precision = tp / pred if pred else 0.0
        micro_recall = tp / gt if gt else 0.0
    result.update(
        {
            "micro_dice": float(micro_dice),
            "micro_iou": float(micro_iou),
            "micro_precision": float(micro_precision),
            "micro_recall": float(micro_recall),
            "image_count": len(per_image),
            "empty_ground_truth_images": sum(
                int(row["ground_truth_pixels"]) == 0 for row in per_image
            ),
            "true_positive_pixels": tp,
            "ground_truth_pixels": gt,
            "predicted_pixels": pred,
            "empty_mask_convention": (
                "For an empty ground truth: Dice/IoU are 1 only when prediction is "
                "also empty, otherwise 0; precision/recall are 1 only when both "
                "are empty, otherwise 0."
            ),
        }
    )
    return result

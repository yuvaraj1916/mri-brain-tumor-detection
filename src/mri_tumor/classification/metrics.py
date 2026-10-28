"""NumPy classification metrics independent of optional sklearn."""

from __future__ import annotations

from typing import Any

import numpy as np


def classification_metrics(
    targets: list[int] | np.ndarray,
    predictions: list[int] | np.ndarray,
    *,
    class_names: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Calculate accuracy, per-class metrics, averages, and confusion matrix."""
    target_array = np.asarray(targets, dtype=np.int64).reshape(-1)
    prediction_array = np.asarray(predictions, dtype=np.int64).reshape(-1)
    names = tuple(class_names)
    if target_array.size == 0 or prediction_array.size == 0:
        raise ValueError("Metric inputs must contain at least one sample.")
    if target_array.shape != prediction_array.shape:
        raise ValueError("Targets and predictions must have the same number of items.")
    if not names:
        raise ValueError("At least one class name is required.")
    if (
        target_array.min() < 0
        or prediction_array.min() < 0
        or target_array.max() >= len(names)
        or prediction_array.max() >= len(names)
    ):
        raise ValueError("Targets and predictions must be valid class indices.")

    matrix = np.zeros((len(names), len(names)), dtype=np.int64)
    np.add.at(matrix, (target_array, prediction_array), 1)
    support = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros(len(names), dtype=np.float64),
        where=predicted_count != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(len(names), dtype=np.float64),
        where=support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(names), dtype=np.float64),
        where=(precision + recall) != 0,
    )
    total = int(target_array.size)
    return {
        "accuracy": float(true_positive.sum() / total),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(names)
        },
        "confusion_matrix": matrix.tolist(),
        "class_names": list(names),
        "support": int(total),
    }

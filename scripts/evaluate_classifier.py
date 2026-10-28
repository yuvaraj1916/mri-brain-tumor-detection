"""Evaluate a selected CNN checkpoint on the derived testing partition."""

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

from mri_tumor.classification.data import (
    CLASS_NAMES,
    ClassificationDataset,
    read_classification_manifest,
    require_torch,
)
from mri_tumor.classification.checkpoint import load_checkpoint
from mri_tumor.classification.metrics import classification_metrics
from mri_tumor.classification.model import create_model
from mri_tumor.data.config import PROJECT_ROOT
from mri_tumor.preprocessing import PreprocessingConfig
from mri_tumor.utils.device import get_device


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as output:
        json.dump(data, output, indent=2, ensure_ascii=False)
        output.write("\n")


def evaluate_checkpoint(
    checkpoint_path: Path,
    manifest_path: Path | None,
    dataset_root: Path | None,
    output_dir: Path,
    *,
    device_preference: str = "auto",
) -> dict[str, Any]:
    """Load the best validation-selected model and evaluate derived testing rows."""
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Classifier checkpoint does not exist: {checkpoint_path}")
    torch = require_torch()
    target_device = get_device(device_preference)
    checkpoint = load_checkpoint(checkpoint_path, map_location=target_device)
    required = {
        "model_state_dict",
        "model_config",
        "class_to_index",
        "preprocessing_config",
        "history",
        "best_epoch",
        "manifest_path",
        "dataset_root",
    }
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError(
            "Checkpoint is incomplete; it must contain model weights, class "
            "mapping, preprocessing settings, and validation-selected history."
        )
    expected_mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    if checkpoint["class_to_index"] != expected_mapping:
        raise ValueError("Checkpoint class mapping is incompatible with this project.")
    if manifest_path is None:
        saved_manifest = checkpoint.get("manifest_path")
        if not saved_manifest:
            raise ValueError(
                "Checkpoint does not record the derived training manifest; "
                "supply --manifest explicitly."
            )
        manifest_path = Path(saved_manifest)
    if dataset_root is None:
        saved_root = checkpoint.get("dataset_root")
        if not saved_root:
            raise ValueError(
                "Checkpoint does not record its classification dataset root; "
                "supply --dataset-root explicitly."
            )
        dataset_root = Path(saved_root)
    preprocessing = PreprocessingConfig(**checkpoint["preprocessing_config"])
    partitions = read_classification_manifest(manifest_path, dataset_root)
    test_dataset = ClassificationDataset(
        partitions["testing"], preprocessing, training=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )
    model = create_model(**checkpoint["model_config"]).to(target_device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = 0.0
    sample_count = 0
    targets: list[int] = []
    predictions: list[int] = []
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(target_device)
            labels = labels.to(target_device, dtype=torch.long)
            logits = model(images)
            loss = criterion(logits, labels)
            count = int(labels.shape[0])
            test_loss += float(loss.item()) * count
            sample_count += count
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    if sample_count != len(test_dataset):
        raise RuntimeError(
            f"Test loader returned {sample_count} items for a "
            f"{len(test_dataset)}-item derived testing partition."
        )
    metrics = classification_metrics(
        targets, predictions, class_names=CLASS_NAMES
    )
    metrics.update(
        {
            "test_loss": test_loss / sample_count,
            "evaluated_split": "testing",
            "checkpoint_path": str(checkpoint_path),
            "manifest_path": str(manifest_path.expanduser().resolve()),
            "best_epoch_selected_on_validation": int(checkpoint["best_epoch"]),
            "device": str(target_device),
            "prototype_notice": (
                "Academic research prototype; not clinically validated."
            ),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "test_metrics.json", metrics)
    with (output_dir / "per_class_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["class_label", "precision", "recall", "f1", "support"],
        )
        writer.writeheader()
        for name in CLASS_NAMES:
            writer.writerow(
                {"class_label": name, **metrics["per_class"][name]}
            )
    with (output_dir / "confusion_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.writer(output)
        writer.writerow(["true_label / predicted_label", *CLASS_NAMES])
        for name, row in zip(CLASS_NAMES, metrics["confusion_matrix"], strict=True):
            writer.writerow([name, *row])

    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted class",
        ylabel="True class",
        title="Derived testing partition confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")
    threshold = max(1, int(matrix.max())) / 2
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
                color="white" if matrix[row_index, column_index] > threshold else "black",
            )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    history = checkpoint["history"]
    epochs = [int(row["epoch"]) for row in history]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(epochs, [float(row["training_loss"]) for row in history], label="Training")
    axis.plot(
        epochs,
        [float(row["validation_loss"]) for row in history],
        label="Validation",
    )
    axis.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Training and validation loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    f1_scores = [metrics["per_class"][name]["f1"] for name in CLASS_NAMES]
    axis.bar(CLASS_NAMES, f1_scores)
    axis.set(
        xlabel="Class",
        ylabel="F1-score",
        ylim=(0, 1),
        title="Per-class F1 on derived testing partition",
    )
    figure.tight_layout()
    figure.savefig(output_dir / "per_class_f1.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    arguments = parser.parse_args(argv)
    manifest = arguments.manifest
    dataset_root = arguments.dataset_root
    if dataset_root is not None and not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root
    output_dir = arguments.output_dir
    if output_dir is None:
        output_dir = arguments.checkpoint.expanduser().resolve().parent
    elif not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    metrics = evaluate_checkpoint(
        arguments.checkpoint,
        manifest,
        dataset_root,
        output_dir,
        device_preference=arguments.device,
    )
    print(
        f"Derived testing: accuracy={metrics['accuracy']:.4f}, "
        f"macro-F1={metrics['macro_f1']:.4f}, "
        f"weighted-F1={metrics['weighted_f1']:.4f}"
    )
    print(f"Metrics and figures saved under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

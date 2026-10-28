"""Seeded U-Net training with validation Dice checkpoint selection."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from mri_tumor.data.config import DatasetPaths
from mri_tumor.preprocessing import PreprocessingConfig
from mri_tumor.segmentation.checkpoint import (
    load_segmentation_checkpoint,
    save_segmentation_checkpoint,
)
from mri_tumor.segmentation.data import (
    SplitName,
    create_dataloaders,
    latest_segmentation_manifest,
    read_segmentation_manifest,
    require_torch,
)
from mri_tumor.segmentation.metrics import aggregate_segmentation_metrics, binary_segmentation_metrics
from mri_tumor.segmentation.model import create_unet
from mri_tumor.utils.device import get_device


@dataclass(frozen=True)
class SegmentationTrainingConfig:
    batch_size: int = 4
    epochs: int = 30
    learning_rate: float = 1e-3
    num_workers: int = 0
    seed: int = 20261012
    patience: int = 7
    base_channels: int = 16
    rotation_degrees: float = 10.0
    horizontal_flip_probability: float = 0.5
    dice_loss_weight: float = 0.5
    mask_threshold: int = 128
    device: str = "auto"

    def __post_init__(self) -> None:
        if min(self.batch_size, self.epochs, self.patience, self.base_channels) <= 0:
            raise ValueError("batch_size, epochs, patience, and base_channels must be positive.")
        if self.learning_rate <= 0 or self.num_workers < 0:
            raise ValueError("learning_rate must be positive and num_workers non-negative.")
        if self.rotation_degrees < 0:
            raise ValueError("rotation_degrees must be non-negative.")
        if not 0 <= self.horizontal_flip_probability <= 1:
            raise ValueError("horizontal_flip_probability must be in [0, 1].")
        if not 0 <= self.dice_loss_weight <= 1:
            raise ValueError("dice_loss_weight must be in [0, 1].")
        if not 1 <= self.mask_threshold <= 255:
            raise ValueError("mask_threshold must be in [1, 255].")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError(f"Unsupported device preference: {self.device!r}")


def load_segmentation_config(path: Path) -> SegmentationTrainingConfig:
    """Load a JSON training config and reject unknown or invalid options."""
    try:
        with path.open("r", encoding="utf-8") as source:
            settings = json.load(source)
        return SegmentationTrainingConfig(**settings)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"Could not load segmentation config from {path}: {error}") from error


def seed_everything(seed: int) -> Any:
    """Seed CPU and accelerator generators and request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch = require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return torch


def create_segmentation_run_directory(root: Path) -> Path:
    """Create a unique segmentation experiment directory."""
    root.mkdir(parents=True, exist_ok=True)
    while True:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = root / f"unet_baseline_{timestamp}_{uuid4().hex[:8]}"
        try:
            destination.mkdir()
            return destination
        except FileExistsError:
            continue


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as output:
        json.dump(data, output, indent=2, ensure_ascii=False)
        output.write("\n")


def _loss_function(torch: Any, dice_loss_weight: float) -> Any:
    bce = torch.nn.BCEWithLogitsLoss()

    def combined(logits: Any, targets: Any) -> Any:
        binary_cross_entropy = bce(logits, targets)
        probabilities = torch.sigmoid(logits)
        dimensions = (1, 2, 3)
        intersection = (probabilities * targets).sum(dim=dimensions)
        denominator = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
        dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        return (
            (1.0 - dice_loss_weight) * binary_cross_entropy
            + dice_loss_weight * dice_loss
        )

    return combined


def _run_epoch(
    model: Any,
    loader: Any,
    *,
    loss_function: Any,
    device: Any,
    optimizer: Any | None,
    max_batches: int | None = None,
) -> tuple[float, dict[str, float | int], int, int]:
    torch = require_torch()
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    image_count = 0
    batch_count = 0
    pixel_count = 0
    probability_sum = 0.0
    probability_squared_sum = 0.0
    per_image: list[dict[str, float | int]] = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, (images, targets) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device)
            targets = targets.to(device, dtype=torch.float32)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_function(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite {'training' if training else 'validation'} loss "
                    f"at batch {batch_index + 1}."
                )
            if training:
                loss.backward()
                for parameter_name, parameter in model.named_parameters():
                    if parameter.grad is None:
                        raise RuntimeError(
                            f"Parameter {parameter_name} has no gradient at "
                            f"training batch {batch_index + 1}."
                        )
                    if not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(
                            f"Non-finite gradient for {parameter_name} at "
                            f"training batch {batch_index + 1}."
                        )
                optimizer.step()
            batch_size = int(targets.shape[0])
            total_loss += float(loss.detach().item()) * batch_size
            image_count += batch_size
            batch_count += 1
            probabilities = torch.sigmoid(logits)
            predictions = probabilities >= 0.5
            probability_values = probabilities.detach().to(device="cpu").to(
                dtype=torch.float64
            )
            pixel_count += probability_values.numel()
            probability_sum += float(probability_values.sum().item())
            probability_squared_sum += float(
                probability_values.square().sum().item()
            )
            truth_cpu = targets.detach().cpu().numpy() >= 0.5
            prediction_cpu = predictions.detach().cpu().numpy()
            for truth, prediction in zip(truth_cpu, prediction_cpu, strict=True):
                per_image.append(binary_segmentation_metrics(truth[0], prediction[0]))
    if image_count == 0:
        raise RuntimeError("A segmentation loader produced no samples.")
    aggregate = aggregate_segmentation_metrics(per_image)
    mean_probability = probability_sum / pixel_count
    aggregate.update(
        {
            "predicted_foreground_fraction": (
                sum(int(item["predicted_pixels"]) for item in per_image)
                / pixel_count
            ),
            "mean_probability": mean_probability,
            "probability_std": max(
                0.0,
                probability_squared_sum / pixel_count - mean_probability**2,
            )
            ** 0.5,
        }
    )
    return (
        total_loss / image_count,
        aggregate,
        image_count,
        batch_count,
    )


def fit_segmentation(
    *,
    manifest_path: Path,
    dataset_root: Path | None,
    preprocessing: PreprocessingConfig,
    config: SegmentationTrainingConfig,
    run_directory: Path,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
) -> Path:
    """Train on derived training pairs and select using validation mean Dice."""
    if max_train_batches is not None and max_train_batches <= 0:
        raise ValueError("max_train_batches must be positive when specified.")
    if max_validation_batches is not None and max_validation_batches <= 0:
        raise ValueError("max_validation_batches must be positive when specified.")
    torch = seed_everything(config.seed)
    partitions, manifest_summary = read_segmentation_manifest(
        manifest_path,
        dataset_root,
        preprocessing=preprocessing,
        mask_threshold=config.mask_threshold,
    )
    dataset_root_resolved = (
        DatasetPaths.from_environment().require("segmentation")
        if dataset_root is None
        else dataset_root.expanduser().resolve()
    )
    print(
        "Validated derived BRISC pairs: "
        + ", ".join(
            f"{split}={len(partitions[split])}"
            for split in ("training", "validation", "testing")
        )
        + f"; excluded flagged pairs={manifest_summary['excluded_flagged_pair_count']}"
    )
    loaders = create_dataloaders(
        partitions,
        preprocessing,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        mask_threshold=config.mask_threshold,
        rotation_degrees=config.rotation_degrees,
        horizontal_flip_probability=config.horizontal_flip_probability,
    )
    if preprocessing.color_mode == "preserve":
        raise ValueError(
            "Segmentation requires configured grayscale or RGB input, not preserve."
        )
    device = get_device(config.device)
    model_config = {
        "input_channels": 1 if preprocessing.color_mode == "grayscale" else 3,
        "base_channels": config.base_channels,
    }
    model = create_unet(**model_config).to(device)
    criterion = _loss_function(torch, config.dice_loss_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(1, config.patience // 3)
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_directory / "best_model.pt"
    settings = {
        "training": asdict(config),
        "preprocessing": asdict(preprocessing),
        "model": model_config,
        "manifest": manifest_summary,
        "selection_metric": "validation_mean_dice",
        "smoke_batch_limits": {
            "training": max_train_batches,
            "validation": max_validation_batches,
        },
        "test_partition_used_for_selection": False,
    }
    _write_json(run_directory / "training_config.json", settings)
    _write_json(run_directory / "manifest_validation.json", manifest_summary)

    history: list[dict[str, float | int]] = []
    best_dice = -1.0
    stale_epochs = 0
    for epoch in range(1, config.epochs + 1):
        training_loss, training_metrics, train_count, train_batches = _run_epoch(
            model,
            loaders["training"],
            loss_function=criterion,
            device=device,
            optimizer=optimizer,
            max_batches=max_train_batches,
        )
        validation_loss, validation_metrics, val_count, val_batches = _run_epoch(
            model,
            loaders["validation"],
            loss_function=criterion,
            device=device,
            optimizer=None,
            max_batches=max_validation_batches,
        )
        validation_dice = float(validation_metrics["mean_dice"])
        row: dict[str, float | int] = {
            "epoch": epoch,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "validation_mean_dice": validation_dice,
            "validation_mean_iou": float(validation_metrics["mean_iou"]),
            "training_predicted_foreground_fraction": float(
                training_metrics["predicted_foreground_fraction"]
            ),
            "validation_predicted_foreground_fraction": float(
                validation_metrics["predicted_foreground_fraction"]
            ),
            "validation_mean_probability": float(
                validation_metrics["mean_probability"]
            ),
            "validation_probability_std": float(
                validation_metrics["probability_std"]
            ),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "training_images": train_count,
            "training_batches": train_batches,
            "validation_images": val_count,
            "validation_batches": val_batches,
        }

        print(
            f"epoch {epoch:03d}/{config.epochs}: train_loss={training_loss:.5f} "
            f"val_loss={validation_loss:.5f} validation_mean_dice={validation_dice:.5f} "
            f"validation_mean_iou={validation_metrics['mean_iou']:.5f} "
            f"val_fg_fraction="
            f"{validation_metrics['predicted_foreground_fraction']:.5f} "
            f"images={train_count}/{val_count} device={device}"
        )
        history.append(row)
        _write_history(run_directory / "epoch_metrics.csv", history)
        scheduler.step(validation_dice)
        if validation_dice > best_dice:
            best_dice = validation_dice
            stale_epochs = 0
            save_segmentation_checkpoint(
                checkpoint_path,
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "preprocessing_config": asdict(preprocessing),
                    "training_config": asdict(config),
                    "manifest_path": str(manifest_path.expanduser().resolve()),
                    "dataset_root": str(dataset_root_resolved),
                    "best_epoch": epoch,
                    "best_validation_mean_dice": best_dice,
                    "validation_metrics": validation_metrics,
                    "history": history.copy(),
                },
            )
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            print(f"Early stopping; best validation mean Dice={best_dice:.5f}.")
            break
    _write_history(run_directory / "training_history.csv", history)
    checkpoint = load_segmentation_checkpoint(checkpoint_path, map_location="cpu")
    checkpoint["history"] = history
    save_segmentation_checkpoint(checkpoint_path, checkpoint)
    _write_json(
        run_directory / "training_summary.json",
        {
            "status": "completed",
            "epochs_completed": len(history),
            "best_epoch": checkpoint["best_epoch"],
            "best_validation_mean_dice": best_dice,
            "checkpoint_path": str(checkpoint_path),
            "device": str(device),
            "smoke_batch_limits": settings["smoke_batch_limits"],
            "test_partition_evaluated": False,
        },
    )
    _plot_learning_curves(history, run_directory / "learning_curves.png")
    return checkpoint_path


def _write_history(path: Path, history: list[dict[str, float | int]]) -> None:
    """Persist completed epochs immediately if a later epoch fails."""
    if not history:
        return
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _plot_learning_curves(
    history: list[dict[str, float | int]], destination: Path
) -> None:
    """Save loss, validation overlap, and predicted-foreground curves."""
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(
        epochs, [float(row["training_loss"]) for row in history], label="Training"
    )
    axes[0].plot(
        epochs, [float(row["validation_loss"]) for row in history], label="Validation"
    )
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(
        epochs, [float(row["validation_mean_dice"]) for row in history], label="Dice"
    )
    axes[1].plot(
        epochs, [float(row["validation_mean_iou"]) for row in history], label="IoU"
    )
    axes[1].set(title="Validation overlap", xlabel="Epoch", ylabel="Score")
    axes[1].legend()
    axes[2].plot(
        epochs,
        [
            float(row["validation_predicted_foreground_fraction"])
            for row in history
        ],
        label="Predicted foreground",
    )
    axes[2].plot(
        epochs,
        [float(row["validation_mean_probability"]) for row in history],
        label="Mean probability",
    )
    axes[2].set(
        title="Validation output", xlabel="Epoch", ylabel="Fraction / probability"
    )
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def default_manifest_path() -> Path:
    """Convenience for command-line entry points."""
    return latest_segmentation_manifest()

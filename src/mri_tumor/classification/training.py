"""Training utilities for the compact, manifest-backed CNN baseline."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from mri_tumor.classification.data import (
    CLASS_NAMES,
    class_distribution_report,
    create_dataloaders,
    save_class_mapping,
)
from mri_tumor.classification.checkpoint import load_checkpoint, save_checkpoint
from mri_tumor.classification.metrics import classification_metrics
from mri_tumor.classification.model import create_model
from mri_tumor.data.config import DatasetPaths, PROJECT_ROOT
from mri_tumor.preprocessing import PreprocessingConfig
from mri_tumor.utils.device import get_device


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-3
    num_workers: int = 0
    seed: int = 20261012
    patience: int = 7
    rotation_degrees: float = 10.0
    horizontal_flip_probability: float = 0.5
    dropout: float = 0.25
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.num_workers < 0 or self.patience < 1:
            raise ValueError("num_workers must be non-negative and patience positive.")
        if self.rotation_degrees < 0:
            raise ValueError("rotation_degrees must be non-negative.")
        if not 0 <= self.horizontal_flip_probability <= 1:
            raise ValueError("horizontal_flip_probability must be between 0 and 1.")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError(f"Unsupported device preference: {self.device!r}")


def load_training_config(path: Path) -> TrainingConfig:
    """Load a JSON experiment config and reject unknown or invalid options."""
    try:
        with path.open("r", encoding="utf-8") as source:
            settings = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load training config from {path}: {error}") from error
    if not isinstance(settings, dict):
        raise ValueError(f"Training config must be a JSON object: {path}")
    try:
        return TrainingConfig(**settings)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid training config in {path}: {error}") from error


def seed_everything(seed: int) -> Any:
    """Seed Python, NumPy, and PyTorch and request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for model training. Install the appropriate "
            "build as documented in README.md; no dependency was installed."
        ) from error
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return torch


def create_run_directory(root: Path) -> Path:
    """Create a unique run directory without overwriting previous results."""
    root.mkdir(parents=True, exist_ok=True)
    while True:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = root / f"cnn_baseline_{stamp}_{uuid4().hex[:8]}"
        try:
            destination.mkdir()
        except FileExistsError:
            continue
        return destination


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as output:
        json.dump(data, output, indent=2, ensure_ascii=False)
        output.write("\n")


def _run_epoch(
    model: Any,
    loader: Any,
    *,
    loss_function: Any,
    device: Any,
    optimizer: Any | None,
) -> tuple[float, list[int], list[int]]:
    torch = seed_torch_module()
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    sample_count = 0
    targets: list[int] = []
    predictions: list[int] = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_function(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = int(labels.shape[0])
            total_loss += float(loss.detach().item()) * batch_size
            sample_count += batch_size
            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
    if sample_count == 0:
        raise RuntimeError("A classification data loader produced no samples.")
    return total_loss / sample_count, targets, predictions


def seed_torch_module() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for classifier training and evaluation."
        ) from error
    return torch


def fit_classifier(
    *,
    manifest_path: Path,
    dataset_root: Path | None,
    preprocessing: PreprocessingConfig,
    config: TrainingConfig,
    run_directory: Path,
) -> Path:
    """Train on derived training rows and select the checkpoint on validation F1."""
    torch = seed_everything(config.seed)
    run_directory.mkdir(parents=True, exist_ok=True)
    loaders, partitions = create_dataloaders(
        manifest_path,
        dataset_root,
        preprocessing,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        rotation_degrees=config.rotation_degrees,
        horizontal_flip_probability=config.horizontal_flip_probability,
    )
    root = (
        dataset_root.expanduser().resolve()
        if dataset_root is not None
        else DatasetPaths.from_environment().require("classification")
    )
    device = get_device(config.device)
    input_channels = 1 if preprocessing.color_mode == "grayscale" else 3
    if preprocessing.color_mode == "preserve":
        raise ValueError(
            "Classification training requires grayscale or RGB preprocessing; "
            "'preserve' may produce mixed channel counts."
        )
    model_config = {
        "input_channels": input_channels,
        "num_classes": len(CLASS_NAMES),
        "dropout": config.dropout,
    }
    model = create_model(**model_config).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(1, config.patience // 3)
    )

    resolved_manifest = manifest_path.expanduser().resolve()
    run_settings = {
        "training": asdict(config),
        "preprocessing": asdict(preprocessing),
        "model": model_config,
        "manifest_path": str(resolved_manifest),
        "dataset_root": str(root),
        "class_to_index": {
            label: index for index, label in enumerate(CLASS_NAMES)
        },
        "split_counts": {
            split: len(records) for split, records in partitions.items()
        },
        "class_distribution": class_distribution_report(partitions),
        "selection_metric": "validation_macro_f1",
    }
    _write_json(run_directory / "training_config.json", run_settings)
    _write_json(
        run_directory / "class_distribution.json",
        run_settings["class_distribution"],
    )
    save_class_mapping(run_directory / "class_mapping.json")

    history: list[dict[str, float | int]] = []
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    checkpoint_path = run_directory / "best_model.pt"
    for epoch in range(1, config.epochs + 1):
        train_loss, _, _ = _run_epoch(
            model,
            loaders["training"],
            loss_function=loss_function,
            device=device,
            optimizer=optimizer,
        )
        validation_loss, validation_targets, validation_predictions = _run_epoch(
            model,
            loaders["validation"],
            loss_function=loss_function,
            device=device,
            optimizer=None,
        )
        validation = classification_metrics(
            validation_targets, validation_predictions, class_names=CLASS_NAMES
        )
        validation_macro_f1 = validation["macro_f1"]
        scheduler.step(validation_macro_f1)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "training_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_macro_f1": validation_macro_f1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{config.epochs}: "
            f"train_loss={train_loss:.5f} val_loss={validation_loss:.5f} "
            f"val_macro_f1={validation_macro_f1:.5f}"
        )
        if validation_macro_f1 > best_macro_f1:
            best_macro_f1 = validation_macro_f1
            epochs_without_improvement = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
                "class_to_index": {
                    label: index for index, label in enumerate(CLASS_NAMES)
                },
                "preprocessing_config": asdict(preprocessing),
                "training_config": asdict(config),
                "manifest_path": str(resolved_manifest),
                "dataset_root": str(root),
                "best_epoch": epoch,
                "best_validation_macro_f1": best_macro_f1,
                "run_config_path": str(run_directory / "training_config.json"),
                "history": history.copy(),
            }
            save_checkpoint(checkpoint_path, checkpoint)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            print(
                f"Early stopping after {epoch} epochs; best validation "
                f"macro-F1={best_macro_f1:.5f}."
            )
            break

    with (run_directory / "training_history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "epoch",
                "training_loss",
                "validation_loss",
                "validation_macro_f1",
                "learning_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(history)
    best_checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    best_checkpoint["history"] = history
    save_checkpoint(checkpoint_path, best_checkpoint)
    _write_json(
        run_directory / "training_summary.json",
        {
            "status": "completed",
            "epochs_completed": len(history),
            "best_epoch": int(
                min(
                    history,
                    key=lambda row: (
                        -float(row["validation_macro_f1"]),
                        int(row["epoch"]),
                    ),
                )["epoch"]
            ),
            "best_validation_macro_f1": best_macro_f1,
            "checkpoint_path": str(checkpoint_path),
            "device": str(device),
            "test_partition_evaluated": False,
        },
    )
    return checkpoint_path

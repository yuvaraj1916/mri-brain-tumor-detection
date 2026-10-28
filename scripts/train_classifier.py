"""Train the academic CNN baseline using the derived classification manifest."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from mri_tumor.classification.data import latest_classification_manifest
from mri_tumor.classification.data import require_torch
from mri_tumor.classification.training import (
    TrainingConfig,
    create_run_directory,
    fit_classifier,
    load_training_config,
)
from mri_tumor.data.config import DatasetPaths, PROJECT_ROOT
from mri_tumor.preprocessing import PreprocessingConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("results/data_manifests"),
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/classification.json"),
    )
    parser.add_argument(
        "--preprocessing-config",
        type=Path,
        default=Path("configs/preprocessing.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/classification"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    arguments = parser.parse_args(argv)

    config_path = arguments.training_config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    base_config = asdict(load_training_config(config_path))
    for option in (
        "batch_size",
        "epochs",
        "learning_rate",
        "num_workers",
        "seed",
        "patience",
        "device",
    ):
        value = getattr(arguments, option)
        if value is not None:
            base_config[option] = value
    config = TrainingConfig(**base_config)

    preprocessing_path = arguments.preprocessing_config
    if not preprocessing_path.is_absolute():
        preprocessing_path = PROJECT_ROOT / preprocessing_path
    preprocessing = PreprocessingConfig.from_json(preprocessing_path)

    manifest_path = arguments.manifest
    if manifest_path is None:
        manifest_dir = arguments.manifest_dir
        if not manifest_dir.is_absolute():
            manifest_dir = PROJECT_ROOT / manifest_dir
        manifest_path = latest_classification_manifest(manifest_dir)
    dataset_root = arguments.dataset_root
    if dataset_root is None:
        dataset_root = DatasetPaths.from_environment().require("classification")
    require_torch()
    output_root = arguments.output_dir
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    run_directory = create_run_directory(output_root)
    checkpoint_path = fit_classifier(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        preprocessing=preprocessing,
        config=config,
        run_directory=run_directory,
    )
    print(f"Training completed. Best checkpoint: {checkpoint_path}")
    print(
        "The derived test partition was not used during training or model selection."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

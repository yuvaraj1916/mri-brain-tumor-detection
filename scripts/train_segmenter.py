"""Train a binary BRISC U-Net using the leakage-audited derived manifest."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from mri_tumor.data.config import DatasetPaths, PROJECT_ROOT
from mri_tumor.preprocessing import PreprocessingConfig
from mri_tumor.segmentation.data import latest_segmentation_manifest, require_torch
from mri_tumor.segmentation.training import (
    SegmentationTrainingConfig,
    create_segmentation_run_directory,
    fit_segmentation,
    load_segmentation_config,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-dir", type=Path, default=Path("results/data_manifests"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--training-config", type=Path, default=Path("configs/segmentation.json"))
    parser.add_argument(
        "--preprocessing-config", type=Path, default=Path("configs/preprocessing.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/segmentation"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    arguments = parser.parse_args(argv)

    config_path = arguments.training_config
    config_path = config_path if config_path.is_absolute() else PROJECT_ROOT / config_path
    values = asdict(load_segmentation_config(config_path))
    for key in (
        "batch_size",
        "epochs",
        "learning_rate",
        "num_workers",
        "seed",
        "patience",
        "device",
    ):
        value = getattr(arguments, key)
        if value is not None:
            values[key] = value
    config = SegmentationTrainingConfig(**values)

    preprocessing_path = arguments.preprocessing_config
    preprocessing_path = (
        preprocessing_path
        if preprocessing_path.is_absolute()
        else PROJECT_ROOT / preprocessing_path
    )
    preprocessing = PreprocessingConfig.from_json(preprocessing_path)

    manifest_path = arguments.manifest
    if manifest_path is None:
        manifest_dir = arguments.manifest_dir
        manifest_dir = (
            manifest_dir if manifest_dir.is_absolute() else PROJECT_ROOT / manifest_dir
        )
        manifest_path = latest_segmentation_manifest(manifest_dir)
    dataset_root = (
        arguments.dataset_root
        if arguments.dataset_root is not None
        else DatasetPaths.from_environment().require("segmentation")
    )
    output_root = arguments.output_dir
    output_root = output_root if output_root.is_absolute() else PROJECT_ROOT / output_root

    require_torch()
    run_directory = create_segmentation_run_directory(output_root)
    checkpoint = fit_segmentation(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        preprocessing=preprocessing,
        config=config,
        run_directory=run_directory,
        max_train_batches=arguments.max_train_batches,
        max_validation_batches=arguments.max_validation_batches,
    )
    print(f"Training completed. Best checkpoint: {checkpoint}")
    print("The derived test partition was not used during training or checkpoint selection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

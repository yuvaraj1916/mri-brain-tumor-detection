"""Audit exact-image leakage and write safe, group-level derived manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from mri_tumor.data.config import DatasetPaths, PROJECT_ROOT
from mri_tumor.data.leakage import audit_datasets, write_audit_artifacts


def _resolve_path(value: Path | None, default: Path) -> Path:
    if value is None:
        return default
    path = value.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-root", type=Path)
    parser.add_argument("--segmentation-root", type=Path)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("results/data_manifests"),
        help="Output directory for separate classification and segmentation CSVs.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("results/dataset_inspection"),
        help="Output directory for the Markdown and JSON audit.",
    )
    parser.add_argument("--seed", type=int, default=20261012)
    arguments = parser.parse_args()

    configured = DatasetPaths.from_environment()
    classification_root = _resolve_path(
        arguments.classification_root, configured.classification
    )
    segmentation_root = _resolve_path(
        arguments.segmentation_root, configured.segmentation
    )
    audit, manifests = audit_datasets(
        classification_root, segmentation_root, seed=arguments.seed
    )
    paths = write_audit_artifacts(
        audit,
        manifests,
        manifest_dir=arguments.manifest_dir,
        report_dir=arguments.report_dir,
    )
    for dataset in ("classification", "segmentation"):
        summary = audit[dataset]
        print(
            f"{dataset}: {summary['image_count']} images, "
            f"{summary['duplicate_group_count']} duplicate groups, "
            f"{summary['official_split_leakage_group_count']} official split leakage groups"
        )
        if dataset == "segmentation":
            print(
                f"  {summary['mask_file_count']} manifest-linked masks, "
                f"{summary['verified_image_mask_pair_count']} verified image-mask pairs"
            )
        for error in summary["errors"]:
            print(f"  REVIEW: {error}")
    overlap = audit["cross_dataset_image_hash_overlap"]
    print(
        "Cross-dataset shared image hashes: "
        f"{overlap['distinct_shared_sha256_count']}"
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

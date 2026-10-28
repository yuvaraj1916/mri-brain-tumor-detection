"""Inspect the configured MRI datasets and write Markdown/JSON reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from mri_tumor.data.config import DatasetPaths, PROJECT_ROOT
from mri_tumor.data.inspection import inspect_dataset, write_reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classification-root",
        type=Path,
        help="Override the configured classification dataset directory.",
    )
    parser.add_argument(
        "--segmentation-root",
        type=Path,
        help="Override the configured segmentation dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dataset_inspection"),
        help="Report directory, relative to the project root unless absolute.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=12,
        help="Maximum number of readable image and mask samples per dataset.",
    )
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="Skip SHA-256 hashing and duplicate detection.",
    )
    arguments = parser.parse_args()
    paths = DatasetPaths.from_environment()
    roots = {
        "classification": _resolve_override(
            arguments.classification_root, paths.classification
        ),
        "segmentation": _resolve_override(
            arguments.segmentation_root, paths.segmentation
        ),
    }
    hash_files = not arguments.no_hash
    reports = [
        inspect_dataset(
            name,
            root,
            sample_size=arguments.sample_size,
            hash_files=hash_files,
        )
        for name, root in roots.items()
    ]
    markdown_path, json_path = write_reports(
        reports, arguments.output_dir, hash_files=hash_files
    )
    for report in reports:
        print(
            f"{report['name']}: {report['total_file_count']} files, "
            f"{report['image_candidate_count']} image candidates, "
            f"{report['readable_image_count']} readable images, "
            f"{report['mask_file_count']} masks"
        )
        if report["errors"]:
            for error in report["errors"]:
                print(f"  ERROR: {error}")
    print(f"Markdown report: {markdown_path}")
    print(f"JSON report: {json_path}")
    return 0


def _resolve_override(value: Path | None, default: Path) -> Path:
    if value is None:
        return default
    path = value.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())

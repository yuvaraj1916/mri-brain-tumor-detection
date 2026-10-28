"""Incremental inspection of image datasets and image-mask relationships."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
ANNOTATION_EXTENSIONS = {
    ".csv",
    ".dcm",
    ".gz",
    ".json",
    ".mat",
    ".mha",
    ".mhd",
    ".nrrd",
    ".nii",
    ".npz",
    ".npy",
    ".sha256",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
_SPLIT_NAMES = {"train", "training", "test", "testing", "validation", "valid", "val"}
_MASK_PARTS = {
    "annotation",
    "annotations",
    "ground_truth",
    "ground-truth",
    "gt",
    "label",
    "labels",
    "mask",
    "masks",
}
_MASK_NAME = re.compile(r"(?:^|[_\-.])(mask|masks|label|labels|seg|segmentation)(?:$|[_\-.])", re.I)
_CHUNK_SIZE = 1024 * 1024


def inspect_dataset(
    name: str,
    root: Path,
    *,
    sample_size: int = 12,
    hash_files: bool = True,
) -> dict[str, Any]:
    """Inspect one dataset without changing its files or loading it into memory."""
    resolved_root = root.expanduser().resolve()
    report: dict[str, Any] = {
        "name": name,
        "root": str(resolved_root),
        "exists": resolved_root.is_dir(),
        "total_file_count": 0,
        "total_bytes": 0,
        "file_extensions": {},
        "directory_structure": {},
        "class_labels": {},
        "image_candidate_count": 0,
        "source_image_count": 0,
        "readable_image_count": 0,
        "unreadable_images": [],
        "sample_image_properties": [],
        "annotation_files": [],
        "mask_file_count": 0,
        "mask_samples": [],
        "image_mask_pairs": {},
        "duplicate_file_groups": [],
        "manifest_summary": {},
        "errors": [],
        "_hash_entries": [],
    }
    if not resolved_root.is_dir():
        report["errors"].append(f"Dataset directory is missing: {resolved_root}")
        return report

    files = sorted(
        (path for path in resolved_root.rglob("*") if path.is_file()),
        key=lambda path: path.as_posix().casefold(),
    )
    report["total_file_count"] = len(files)
    sizes_by_extension: dict[str, int] = defaultdict(int)
    counts_by_extension: dict[str, int] = defaultdict(int)
    structure: dict[str, int] = defaultdict(int)
    labels: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    mask_files: list[Path] = []
    image_files: list[Path] = []
    hash_paths: dict[str, list[str]] = defaultdict(list)
    manifest_masks: set[str] = set()
    manifest_links: dict[str, str] = {}
    manifest_segmentation_images: set[str] = set()
    manifest_rows = 0
    manifest_mask_rows = 0
    manifest_link_rows = 0
    image_sample_size = max(0, sample_size)
    image_dimensions: dict[str, tuple[int, int] | None] = {}

    for path in files:
        relative = path.relative_to(resolved_root)
        extension = path.suffix.lower() or "[no extension]"
        try:
            file_size = path.stat().st_size
        except OSError as error:
            report["errors"].append(f"Could not stat {relative}: {error}")
            file_size = 0
        report["total_bytes"] += file_size
        counts_by_extension[extension] += 1
        sizes_by_extension[extension] += file_size
        structure[relative.parent.as_posix() or "."] += 1

        if extension in ANNOTATION_EXTENSIONS:
            report["annotation_files"].append(
                {
                    "path": relative.as_posix(),
                    "extension": extension,
                    "size_bytes": file_size,
                }
            )

        if extension == ".csv" and path.name.casefold().startswith("manifest"):
            try:
                with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
                    for row in csv.DictReader(csv_file):
                        manifest_rows += 1
                        if row.get("is_mask", "").strip().casefold() in {
                            "true",
                            "1",
                            "yes",
                        }:
                            manifest_mask_rows += 1
                            relative_mask = _normalize_manifest_path(
                                row.get("relative_path", "")
                            )
                            if relative_mask:
                                manifest_masks.add(relative_mask)
                            linked_image = _normalize_manifest_path(
                                row.get("linked_image", "")
                            )
                            if relative_mask and linked_image:
                                manifest_links[relative_mask] = linked_image
                                manifest_link_rows += 1
                        elif row.get("task", "").strip().casefold() == "segmentation":
                            relative_image = _normalize_manifest_path(
                                row.get("relative_path", "")
                            )
                            if relative_image:
                                manifest_segmentation_images.add(relative_image)
            except (OSError, UnicodeError, csv.Error) as error:
                report["errors"].append(
                    f"Could not read manifest CSV {relative}: {error}"
                )

        if hash_files:
            try:
                digest = _sha256(path)
                hash_paths[digest].append(relative.as_posix())
                report["_hash_entries"].append(
                    {"dataset": name, "path": relative.as_posix(), "sha256": digest}
                )
            except OSError as error:
                report["errors"].append(f"Could not hash {relative}: {error}")

        if extension in IMAGE_EXTENSIONS:
            image_parts = {part.casefold() for part in relative.parts[:-1]}
            filename_is_mask = bool(_MASK_NAME.search(path.stem))
            explicitly_marked_mask = (
                relative.as_posix() in manifest_masks
                or any(part in _MASK_PARTS for part in image_parts)
                or filename_is_mask
            )
            if explicitly_marked_mask:
                mask_files.append(path)
            else:
                image_files.append(path)
                report["source_image_count"] += 1
                split = next(
                    (
                        part.casefold()
                        for part in relative.parts[:-1]
                        if part.casefold() in _SPLIT_NAMES
                    ),
                    None,
                )
                parent = relative.parent.name.casefold()
                if split and parent not in _MASK_PARTS | {"images", "image"}:
                    labels[parent][split] += 1

            report["image_candidate_count"] += 1
            try:
                with Image.open(path) as image:
                    width, height = image.size
                    mode = image.mode
                    bands = image.getbands()
                    image.load()
                report["readable_image_count"] += 1
                image_dimensions[relative.as_posix().casefold()] = (width, height)
                if len(report["sample_image_properties"]) < image_sample_size:
                    report["sample_image_properties"].append(
                        {
                            "path": relative.as_posix(),
                            "width": width,
                            "height": height,
                            "mode": mode,
                            "channels": len(bands),
                            "bands": list(bands),
                        }
                    )
                if explicitly_marked_mask and len(report["mask_samples"]) < image_sample_size:
                    report["mask_samples"].append(
                        {
                            "path": relative.as_posix(),
                            "width": width,
                            "height": height,
                            "mode": mode,
                            "channels": len(bands),
                        }
                    )
            except (
                OSError,
                ValueError,
                UnidentifiedImageError,
                Image.DecompressionBombError,
            ) as error:
                image_dimensions[relative.as_posix().casefold()] = None
                report["unreadable_images"].append(
                    {"path": relative.as_posix(), "error": str(error)}
                )

    report["file_extensions"] = {
        extension: {"count": counts_by_extension[extension], "bytes": sizes_by_extension[extension]}
        for extension in sorted(counts_by_extension)
    }
    report["directory_structure"] = dict(sorted(structure.items()))
    report["class_labels"] = {
        label: dict(sorted(split_counts.items()))
        for label, split_counts in sorted(labels.items())
    }
    report["mask_file_count"] = len(mask_files)
    report["manifest_summary"] = {
        "csv_rows": manifest_rows,
        "declared_mask_rows": manifest_mask_rows,
        "declared_mask_rows_with_linked_image": manifest_link_rows,
        "declared_mask_paths_found": sum(
            1
            for path in mask_files
            if path.relative_to(resolved_root).as_posix().casefold() in manifest_masks
        ),
    }
    report["image_mask_pairs"] = _pair_images_and_masks(
        resolved_root,
        image_files,
        mask_files,
        manifest_links,
        manifest_segmentation_images,
        image_dimensions,
    )
    report["duplicate_file_groups"] = [
        {"sha256": digest, "paths": sorted(paths), "count": len(paths)}
        for digest, paths in sorted(hash_paths.items())
        if len(paths) > 1
    ]
    cross_split_groups = []
    for group in report["duplicate_file_groups"]:
        splits = {
            split
            for path in group["paths"]
            if (split := _normalized_split(path)) is not None
        }
        if len(splits) > 1:
            cross_split_groups.append({**group, "splits": sorted(splits)})
    report["duplicate_summary"] = {
        "hash_group_count": len(report["duplicate_file_groups"]),
        "duplicate_file_count": sum(
            group["count"] for group in report["duplicate_file_groups"]
        ),
        "cross_split_group_count": len(cross_split_groups),
        "cross_split_groups": cross_split_groups,
    }
    return report


def _normalized_split(path: str) -> str | None:
    for part in path.casefold().split("/"):
        if part in {"train", "training"}:
            return "train"
        if part in {"test", "testing"}:
            return "test"
        if part in {"validation", "valid", "val"}:
            return "validation"
    return None


def _normalize_manifest_path(value: str) -> str:
    return PurePosixPath(value.strip().replace("\\", "/")).as_posix().casefold()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _pair_images_and_masks(
    root: Path,
    images: list[Path],
    masks: list[Path],
    manifest_links: dict[str, str],
    manifest_segmentation_images: set[str],
    image_dimensions: dict[str, tuple[int, int] | None],
) -> dict[str, Any]:
    all_images_by_relative_path = {
        path.relative_to(root).as_posix().casefold(): path for path in images
    }
    explicit_source_image_paths: set[str] = set()
    for image in images:
        relative = image.relative_to(root)
        if any(part.casefold() in {"image", "images"} for part in relative.parts[:-1]):
            explicit_source_image_paths.add(relative.as_posix().casefold())
    linked_image_paths = {path for path in manifest_links.values()}
    pairable_images = {
        path
        for path in images
        if path.relative_to(root).as_posix().casefold() in explicit_source_image_paths
        or path.relative_to(root).as_posix().casefold() in manifest_segmentation_images
        or path.relative_to(root).as_posix().casefold() in linked_image_paths
    }
    images_by_stem: dict[str, list[Path]] = defaultdict(list)
    for image in pairable_images:
        images_by_stem[image.stem.casefold()].append(image)

    pairs: list[dict[str, str]] = []
    unmatched_masks: list[str] = []
    ambiguous_masks: list[dict[str, Any]] = []
    matched_images: set[Path] = set()
    dimension_mismatches: list[dict[str, str]] = []
    unreadable_pairs: list[dict[str, str]] = []
    for mask in masks:
        relative_mask = mask.relative_to(root).as_posix()
        manifest_link = manifest_links.get(relative_mask.casefold())
        candidates: list[Path] = []
        if manifest_link:
            linked_path = all_images_by_relative_path.get(manifest_link)
            if linked_path is not None:
                candidates = [linked_path]
        else:
            candidates = images_by_stem.get(mask.stem.casefold(), [])
        if len(candidates) == 1:
            image = candidates[0]
            matched_images.add(image)
            relative_image = image.relative_to(root).as_posix()
            image_size = image_dimensions.get(relative_image.casefold())
            mask_size = image_dimensions.get(relative_mask.casefold())
            if image_size is None or mask_size is None:
                unreadable_pairs.append({"image": relative_image, "mask": relative_mask})
            elif image_size != mask_size:
                dimension_mismatches.append(
                    {
                        "image": relative_image,
                        "mask": relative_mask,
                        "image_dimensions": f"{image_size[0]}x{image_size[1]}",
                        "mask_dimensions": f"{mask_size[0]}x{mask_size[1]}",
                    }
                )
            pairs.append(
                {
                    "image": relative_image,
                    "mask": relative_mask,
                    "match_method": "manifest_link" if manifest_link else "unique_basename",
                    "dimensions_match": (
                        image_size is not None
                        and mask_size is not None
                        and image_size == mask_size
                    ),
                }
            )
        elif len(candidates) > 1:
            ambiguous_masks.append(
                {
                    "mask": relative_mask,
                    "candidate_images": [
                        path.relative_to(root).as_posix() for path in candidates
                    ],
                }
            )
        else:
            unmatched_masks.append(relative_mask)

    return {
        "matched_pair_count": len(pairs),
        "pairing_applicable": bool(masks),
        "candidate_image_count": len(pairable_images) if masks else 0,
        "candidate_mask_count": len(masks),
        "unmatched_image_count": (
            len(pairable_images - matched_images) if masks else 0
        ),
        "unmatched_mask_count": len(unmatched_masks),
        "ambiguous_mask_count": len(ambiguous_masks),
        "dimension_mismatch_count": len(dimension_mismatches),
        "unreadable_pair_count": len(unreadable_pairs),
        "verified_pair_count": sum(pair["dimensions_match"] for pair in pairs),
        "pairs": pairs,
        "unmatched_masks": unmatched_masks,
        "ambiguous_masks": ambiguous_masks,
        "dimension_mismatches": dimension_mismatches,
        "unreadable_pairs": unreadable_pairs,
    }


def build_combined_duplicate_groups(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find identical files across all inspected datasets from their hashes."""
    by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for report in reports:
        for entry in report["_hash_entries"]:
            by_hash[entry["sha256"]].append(
                {"dataset": entry["dataset"], "path": entry["path"]}
            )
    return [
        {"sha256": digest, "files": files, "count": len(files)}
        for digest, files in sorted(by_hash.items())
        if len(files) > 1 and len({entry["dataset"] for entry in files}) > 1
    ]


def write_reports(
    reports: list[dict[str, Any]],
    output_dir: Path,
    *,
    hash_files: bool,
) -> tuple[Path, Path]:
    """Write unique Markdown and JSON reports, never replacing existing files."""
    from datetime import datetime, timezone
    from uuid import uuid4

    destination = output_dir.expanduser()
    if not destination.is_absolute():
        destination = Path(__file__).resolve().parents[3] / destination
    destination.mkdir(parents=True, exist_ok=True)

    combined_duplicates = build_combined_duplicate_groups(reports) if hash_files else []
    public_reports = [
        {key: value for key, value in report.items() if not key.startswith("_")}
        for report in reports
    ]
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hashing_enabled": hash_files,
        "datasets": public_reports,
        "cross_dataset_duplicate_file_groups": combined_duplicates,
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    while True:
        basename = f"dataset_inspection_{timestamp}_{uuid4().hex[:8]}"
        json_path = destination / f"{basename}.json"
        markdown_path = destination / f"{basename}.md"
        try:
            with json_path.open("x", encoding="utf-8") as json_file:
                json.dump(payload, json_file, indent=2, ensure_ascii=False)
                json_file.write("\n")
            with markdown_path.open("x", encoding="utf-8") as markdown_file:
                markdown_file.write(render_markdown(payload))
            return markdown_path, json_path
        except FileExistsError:
            continue


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Dataset inspection report",
        "",
        f"Generated (UTC): {payload['generated_at_utc']}",
        f"Content hashing enabled: {payload['hashing_enabled']}",
        "",
    ]
    for report in payload["datasets"]:
        lines.extend(
            [
                f"## {report['name']}",
                "",
                f"- Root: `{report['root']}`",
                f"- Exists: {report['exists']}",
                f"- Total files: {report['total_file_count']:,}",
                f"- Approximate logical size: {_format_bytes(report['total_bytes'])} "
                f"({report['total_bytes']:,} bytes)",
                f"- Image files (raster extensions, including masks): "
                f"{report['image_candidate_count']:,}",
                f"- Non-mask source images: {report['source_image_count']:,}",
                f"- Readable images: {report['readable_image_count']:,}",
                f"- Unreadable images: {len(report['unreadable_images']):,}",
                f"- Discovered masks: {report['mask_file_count']:,}",
                "",
                "### Extensions",
                "",
                "| Extension | Files | Approx. size |",
                "|---|---:|---:|",
            ]
        )
        for extension, details in report["file_extensions"].items():
            lines.append(
                f"| `{extension}` | {details['count']:,} | {_format_bytes(details['bytes'])} |"
            )
        lines.extend(["", "### Directory structure", ""])
        lines.extend(
            f"- `{directory}/`: {count:,} files"
            for directory, count in report["directory_structure"].items()
        )
        if not report["directory_structure"]:
            lines.append("- (no files)")
        lines.extend(["", "### Observed class labels", ""])
        if report["class_labels"]:
            lines.extend(
                f"- `{label}`: "
                + ", ".join(f"{split}={count:,}" for split, count in split_counts.items())
                for label, split_counts in report["class_labels"].items()
            )
        else:
            lines.append("- No class-label directory structure detected.")
        lines.extend(["", "### Image samples", ""])
        if report["sample_image_properties"]:
            lines.extend(
                f"- `{sample['path']}`: {sample['width']}×{sample['height']}, "
                f"mode {sample['mode']}, {sample['channels']} channel(s)"
                for sample in report["sample_image_properties"]
            )
        else:
            lines.append("- No readable image samples.")
        lines.extend(["", "### Masks and pairing", ""])
        pair_summary = report["image_mask_pairs"]
        lines.extend(
            [
                f"- Mask-based pairing applicable: {pair_summary['pairing_applicable']}",
                f"- Matched image-mask pairs: {pair_summary['matched_pair_count']:,}",
                f"- Images without a paired mask: {pair_summary['unmatched_image_count']:,}",
                f"- Masks without a paired image: {pair_summary['unmatched_mask_count']:,}",
                f"- Ambiguous mask matches: {pair_summary['ambiguous_mask_count']:,}",
                f"- Pairs with matching readable dimensions: "
                f"{pair_summary['verified_pair_count']:,}",
                f"- Paired dimension mismatches / unreadable pairs: "
                f"{pair_summary['dimension_mismatch_count']:,} / "
                f"{pair_summary['unreadable_pair_count']:,}",
                f"- Manifest rows / declared masks / declared links: "
                f"{report['manifest_summary'].get('csv_rows', 0):,} / "
                f"{report['manifest_summary'].get('declared_mask_rows', 0):,} / "
                f"{report['manifest_summary'].get('declared_mask_rows_with_linked_image', 0):,}",
            ]
        )
        if report["mask_samples"]:
            lines.extend(
                [
                    "",
                    "Mask samples:",
                    *(
                        f"- `{sample['path']}`: {sample['width']}×{sample['height']}, "
                        f"mode {sample['mode']}, {sample['channels']} channel(s)"
                        for sample in report["mask_samples"]
                    ),
                ]
            )
        lines.extend(["", "### Annotation files", ""])
        if report["annotation_files"]:
            lines.extend(
                f"- `{annotation['path']}` ({annotation['extension']}, "
                f"{_format_bytes(annotation['size_bytes'])})"
                for annotation in report["annotation_files"]
            )
        else:
            lines.append("- No common annotation/metadata file extensions found.")
        lines.extend(["", "### Duplicate files", ""])
        if report["duplicate_file_groups"]:
            duplicate_groups = report["duplicate_file_groups"]
            lines.append(
                f"- {len(duplicate_groups):,} identical-content groups "
                f"({sum(group['count'] for group in duplicate_groups):,} files total)."
            )
            duplicate_summary = report["duplicate_summary"]
            lines.append(
                f"- Exact duplicate groups spanning named train/test/validation splits: "
                f"{duplicate_summary['cross_split_group_count']:,}."
            )
            for group in duplicate_summary["cross_split_groups"][:10]:
                lines.append(
                    f"- Cross-split SHA-256 `{group['sha256']}` "
                    f"({', '.join(group['splits'])}): "
                    + ", ".join(f"`{path}`" for path in group["paths"])
                )
            for group in duplicate_groups[:20]:
                lines.append(
                    f"- SHA-256 `{group['sha256']}` ({group['count']} files): "
                    + ", ".join(f"`{path}`" for path in group["paths"])
                )
            if len(duplicate_groups) > 20:
                lines.append(
                    f"- {len(duplicate_groups) - 20:,} additional groups are detailed in JSON."
                )
        else:
            lines.append(
                "- No identical-file groups found."
                if payload["hashing_enabled"]
                else "- Not checked (hashing disabled)."
            )
        if report["unreadable_images"]:
            lines.extend(["", "### Unreadable images", ""])
            lines.extend(
                f"- `{failure['path']}`: {failure['error']}"
                for failure in report["unreadable_images"]
            )
        if report["errors"]:
            lines.extend(["", "### Inspection errors", ""])
            lines.extend(f"- {error}" for error in report["errors"])
        lines.append("")
    lines.extend(["## Cross-dataset duplicates", ""])
    duplicates = payload["cross_dataset_duplicate_file_groups"]
    if duplicates:
        lines.append(f"- {len(duplicates):,} groups span multiple datasets.")
        for group in duplicates[:20]:
            lines.append(
                f"- SHA-256 `{group['sha256']}`: "
                + ", ".join(f"`{item['dataset']}/{item['path']}`" for item in group["files"])
            )
        if len(duplicates) > 20:
            lines.append(
                f"- {len(duplicates) - 20:,} additional groups are detailed in JSON."
            )
    else:
        lines.append(
            "- None found."
            if payload["hashing_enabled"]
            else "- Not checked (hashing disabled)."
        )
    return "\n".join(lines) + "\n"


def _format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TB"

"""Exact-hash leakage audits and deterministic, non-destructive manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

from mri_tumor.data.config import PROJECT_ROOT
from mri_tumor.data.inspection import IMAGE_EXTENSIONS

SPLIT_PROPORTIONS: dict[str, float] = {
    "training": 0.70,
    "validation": 0.15,
    "testing": 0.15,
}
_MASK_TRUE_VALUES = {"true", "1", "yes"}
_CHUNK_SIZE = 1024 * 1024


@dataclass
class ImageRecord:
    """One source image, retaining its verified source annotations and split."""

    image_path: str
    label: str
    original_split: str
    sha256: str
    mask_path: str = ""
    mask_sha256: str = ""
    mask_pixel_sha256: str = ""
    proposed_split: str = ""
    duplicate_group_id: str = ""
    flags: tuple[str, ...] = ()


def sha256_file(path: Path) -> str:
    """Hash a file in bounded-memory chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_relative(value: str) -> str:
    return PurePosixPath(value.strip().replace("\\", "/")).as_posix()


def _normalized_split(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized in {"train", "training"}:
        return "training"
    if normalized in {"val", "valid", "validation"}:
        return "validation"
    if normalized in {"test", "testing"}:
        return "testing"
    return normalized or "unknown"


def _split_from_path(relative_path: Path) -> tuple[str, str]:
    parts = relative_path.parts
    for index, part in enumerate(parts[:-1]):
        split = _normalized_split(part)
        if split in SPLIT_PROPORTIONS and index + 1 < len(parts) - 1:
            return split, parts[index + 1]
    return "unknown", relative_path.parent.name or "unknown"


def _readable_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            dimensions = image.size
            image.load()
            return dimensions
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
        return None


def _decoded_pixel_sha256(path: Path) -> str | None:
    try:
        with Image.open(path) as image:
            rgba = image.convert("RGBA")
            digest = hashlib.sha256()
            digest.update(f"{rgba.width}x{rgba.height}:RGBA:".encode("ascii"))
            digest.update(rgba.tobytes())
            return digest.hexdigest()
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
        return None


def _assign_group_splits(
    groups: dict[str, list[ImageRecord]],
    seed: int,
) -> None:
    """Assign groups within label strata to near-target proportions, repeatably."""
    groups_by_label: dict[str, list[str]] = defaultdict(list)
    for group_key, records in groups.items():
        labels = {record.label for record in records}
        if len(labels) != 1:
            for record in records:
                record.flags = tuple(sorted(set(record.flags) | {"conflicting_labels"}))
                record.proposed_split = "flagged"
            continue
        if any(
            _normalized_split(record.original_split) == "unknown"
            for record in records
        ):
            for record in records:
                record.flags = tuple(
                    sorted(set(record.flags) | {"unknown_official_split"})
                )
                record.proposed_split = "flagged"
            continue
        if any(
            "unreadable_image" in record.flags
            or "missing_or_invalid_mask" in record.flags
            for record in records
        ):
            for record in records:
                record.proposed_split = "flagged"
            continue
        if len({record.mask_pixel_sha256 for record in records if record.mask_path}) > 1:
            for record in records:
                record.flags = tuple(
                    sorted(
                        set(record.flags)
                        | {"conflicting_masks_for_identical_image"}
                    )
                )
                record.proposed_split = "flagged"
            continue
        groups_by_label[next(iter(labels))].append(group_key)

    for label_index, label in enumerate(sorted(groups_by_label)):
        keys = groups_by_label[label]
        rng = random.Random(seed + label_index * 1_000_003)
        rng.shuffle(keys)
        total = len(keys)
        targets = {
            split: total * proportion
            for split, proportion in SPLIT_PROPORTIONS.items()
        }
        assigned = {split: 0 for split in SPLIT_PROPORTIONS}
        for group_key in keys:
            split = min(
                SPLIT_PROPORTIONS,
                key=lambda candidate: (
                    (assigned[candidate] + 1 - targets[candidate]) ** 2
                    - (assigned[candidate] - targets[candidate]) ** 2,
                    list(SPLIT_PROPORTIONS).index(candidate),
                ),
            )
            assigned[split] += 1
            group = groups[group_key]
            mask_hashes = {
                record.mask_pixel_sha256
                for record in group
                if record.mask_path and record.mask_pixel_sha256
            }
            group_flags: set[str] = set()
            if any("missing_or_invalid_mask" in record.flags for record in group):
                group_flags.add("missing_or_invalid_mask")
            if len(mask_hashes) > 1:
                group_flags.add("conflicting_masks_for_identical_image")
            original_splits = {
                _normalized_split(record.original_split)
                for record in group
                if record.original_split
            }
            if len(original_splits) > 1:
                group_flags.add("official_split_conflict_reassigned")
            for record in group:
                record.proposed_split = (
                    "flagged"
                    if group_flags
                    & {
                        "conflicting_masks_for_identical_image",
                        "missing_or_invalid_mask",
                    }
                    else split
                )
                record.flags = tuple(sorted(set(record.flags) | group_flags))


def _duplicate_groups(records: list[ImageRecord]) -> dict[str, list[ImageRecord]]:
    by_hash: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_hash[record.sha256].append(record)
    groups = {
        digest: sorted(items, key=lambda record: record.image_path.casefold())
        for digest, items in by_hash.items()
    }
    for digest, group in groups.items():
        if len(group) > 1:
            group_id = f"sha256-{digest[:16]}"
            for record in group:
                record.duplicate_group_id = group_id
    return groups


def _group_findings(
    records: list[ImageRecord], groups: dict[str, list[ImageRecord]]
) -> dict[str, Any]:
    duplicate_groups = []
    cross_split_groups = []
    label_conflicts = []
    mask_conflicts = []
    for digest, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        labels = sorted({record.label for record in group})
        splits = sorted(
            {
                _normalized_split(record.original_split)
                for record in group
                if record.original_split
            }
        )
        paths = [record.image_path for record in group]
        item = {
            "duplicate_group_id": f"sha256-{digest[:16]}",
            "sha256": digest,
            "count": len(group),
            "image_paths": paths,
            "labels": labels,
            "original_splits": splits,
        }
        duplicate_groups.append(item)
        if len(splits) > 1:
            cross_split_groups.append(item)
        if len(labels) > 1:
            label_conflicts.append(
                {
                    **item,
                    "reason": "Byte-identical image files have different class labels.",
                }
            )
        mask_hashes = {
            record.mask_pixel_sha256
            for record in group
            if record.mask_path and record.mask_pixel_sha256
        }
        if len(mask_hashes) > 1:
            mask_conflicts.append(
                {
                    **item,
                    "masks": [
                        {
                            "image_path": record.image_path,
                            "mask_path": record.mask_path,
                            "mask_sha256": record.mask_sha256,
                            "mask_pixel_sha256": record.mask_pixel_sha256,
                        }
                        for record in group
                    ],
                    "reason": (
                        "Byte-identical MRI image content is associated with "
                        "different decoded mask pixel contents."
                    ),
                }
            )
    return {
        "image_count": len(records),
        "unique_image_hash_count": len(groups),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_file_count": sum(group["count"] for group in duplicate_groups),
        "cross_official_split_group_count": len(cross_split_groups),
        "cross_official_split_groups": cross_split_groups,
        "duplicate_groups": duplicate_groups,
        "duplicate_class_label_conflicts": label_conflicts,
        "conflicting_masks_for_identical_images": mask_conflicts,
    }


def _split_summaries(records: list[ImageRecord]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in (*SPLIT_PROPORTIONS, "flagged"):
        subset = [record for record in records if record.proposed_split == split]
        if not subset:
            continue
        result[split] = {
            "image_or_pair_count": len(subset),
            "group_count": len({record.sha256 for record in subset}),
            "class_counts": dict(sorted(Counter(record.label for record in subset).items())),
        }
    return result


def _build_classification_records(root: Path) -> tuple[list[ImageRecord], list[str]]:
    records: list[ImageRecord] = []
    errors: list[str] = []
    if not root.is_dir():
        return records, [f"Dataset directory is missing: {root}"]
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(root)
        split, label = _split_from_path(relative)
        if split == "unknown":
            errors.append(f"Could not determine official split from path: {relative}")
        dimensions = _readable_dimensions(path)
        flags: tuple[str, ...] = ()
        if dimensions is None:
            errors.append(f"Image is unreadable: {relative}")
            flags = ("unreadable_image",)
        try:
            digest = sha256_file(path)
        except OSError as error:
            errors.append(f"Could not hash image {relative}: {error}")
            continue
        records.append(
            ImageRecord(
                image_path=relative.as_posix(),
                label=label,
                original_split=split,
                sha256=digest,
                flags=flags,
            )
        )
    return records, errors


def _build_segmentation_records(root: Path) -> tuple[list[ImageRecord], list[str]]:
    records: list[ImageRecord] = []
    errors: list[str] = []
    if not root.is_dir():
        return records, [f"Dataset directory is missing: {root}"]
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        return records, [f"Required BRISC manifest is missing: {manifest}"]
    rows: list[dict[str, str]] = []
    try:
        with manifest.open("r", encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
    except (OSError, UnicodeError, csv.Error) as error:
        return records, [f"Could not read BRISC manifest {manifest}: {error}"]

    images: dict[str, dict[str, str]] = {}
    masks_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("task", "").strip().casefold() != "segmentation":
            continue
        relative = _portable_relative(row.get("relative_path", ""))
        if row.get("is_mask", "").strip().casefold() in _MASK_TRUE_VALUES:
            linked = _portable_relative(row.get("linked_image", ""))
            if linked and relative:
                masks_by_image[linked.casefold()].append(row)
        elif relative:
            images[relative.casefold()] = row

    for image_key, row in sorted(images.items()):
        image_relative = _portable_relative(row.get("relative_path", ""))
        image_path = root.joinpath(*PurePosixPath(image_relative).parts)
        if not image_path.is_file():
            errors.append(f"Manifest MRI image is missing: {image_relative}")
            continue
        image_dimensions = _readable_dimensions(image_path)
        if image_dimensions is None:
            errors.append(f"Manifest MRI image is unreadable: {image_relative}")
        try:
            image_digest = sha256_file(image_path)
        except OSError as error:
            errors.append(f"Could not hash MRI image {image_relative}: {error}")
            continue

        linked_masks = masks_by_image.get(image_key, [])
        selected_mask = linked_masks[0] if len(linked_masks) == 1 else None
        mask_relative = ""
        mask_digest = ""
        mask_pixel_digest = ""
        pair_valid = selected_mask is not None
        if len(linked_masks) > 1:
            pair_valid = False
            errors.append(
                f"Multiple manifest masks link to MRI image {image_relative}: "
                + ", ".join(
                    _portable_relative(mask.get("relative_path", ""))
                    for mask in linked_masks
                )
            )
        elif selected_mask is None:
            pair_valid = False
            errors.append(f"No manifest-linked mask for MRI image {image_relative}")
        else:
            mask_relative = _portable_relative(selected_mask.get("relative_path", ""))
            mask_path = root.joinpath(*PurePosixPath(mask_relative).parts)
            if not mask_path.is_file():
                pair_valid = False
                errors.append(f"Manifest mask is missing: {mask_relative}")
                mask_relative = ""
            else:
                mask_dimensions = _readable_dimensions(mask_path)
                if mask_dimensions is None:
                    pair_valid = False
                    errors.append(f"Manifest mask is unreadable: {mask_relative}")
                elif image_dimensions is not None and image_dimensions != mask_dimensions:
                    pair_valid = False
                    errors.append(
                        f"Image-mask dimensions differ: {image_relative} "
                        f"{image_dimensions} vs {mask_relative} {mask_dimensions}"
                    )
                try:
                    mask_digest = sha256_file(mask_path)
                    mask_pixel_digest = _decoded_pixel_sha256(mask_path) or ""
                    if not mask_pixel_digest:
                        pair_valid = False
                        errors.append(
                            f"Could not decode mask pixels for {mask_relative}"
                        )
                except OSError as error:
                    pair_valid = False
                    errors.append(f"Could not hash mask {mask_relative}: {error}")
                    mask_relative = ""
                    mask_digest = ""
        records.append(
            ImageRecord(
                image_path=image_relative,
                label=row.get("tumor_label", "").strip() or "unknown",
                original_split=_normalized_split(row.get("split", "")),
                sha256=image_digest,
                mask_path=mask_relative,
                mask_sha256=mask_digest,
                mask_pixel_sha256=mask_pixel_digest,
                flags=tuple(
                    flag
                    for flag, failed in (
                        ("unreadable_image", image_dimensions is None),
                        ("missing_or_invalid_mask", not pair_valid),
                    )
                    if failed
                ),
            )
        )
    return records, errors


def _build_brisc_image_hash_index(root: Path) -> tuple[dict[str, list[str]], list[str]]:
    """Hash every non-mask image listed by either BRISC task for overlap checks."""
    hashes: dict[str, list[str]] = defaultdict(list)
    errors: list[str] = []
    manifest = root / "manifest.csv"
    if not root.is_dir() or not manifest.is_file():
        return {}, []
    try:
        with manifest.open("r", encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                if row.get("is_mask", "").strip().casefold() in _MASK_TRUE_VALUES:
                    continue
                relative = _portable_relative(row.get("relative_path", ""))
                path = root.joinpath(*PurePosixPath(relative).parts)
                if path.suffix.casefold() not in IMAGE_EXTENSIONS or not path.is_file():
                    continue
                try:
                    hashes[sha256_file(path)].append(relative)
                except OSError as error:
                    errors.append(f"Could not hash BRISC image {relative}: {error}")
    except (OSError, UnicodeError, csv.Error) as error:
        errors.append(f"Could not read BRISC manifest {manifest}: {error}")
    return hashes, errors


def audit_datasets(
    classification_root: Path,
    segmentation_root: Path,
    *,
    seed: int = 20261012,
) -> tuple[dict[str, Any], dict[str, list[dict[str, str]]]]:
    """Hash, audit, and propose safe group-level splits for both datasets."""
    classification_records, classification_errors = _build_classification_records(
        classification_root.expanduser().resolve()
    )
    segmentation_records, segmentation_errors = _build_segmentation_records(
        segmentation_root.expanduser().resolve()
    )
    brisc_image_paths_by_hash, overlap_errors = _build_brisc_image_hash_index(
        segmentation_root.expanduser().resolve()
    )
    classification_groups = _duplicate_groups(classification_records)
    segmentation_groups = _duplicate_groups(segmentation_records)
    classification_findings = _group_findings(
        classification_records, classification_groups
    )
    segmentation_findings = _group_findings(segmentation_records, segmentation_groups)
    _assign_group_splits(classification_groups, seed)
    _assign_group_splits(segmentation_groups, seed)

    classification_hashes = {record.sha256 for record in classification_records}
    segmentation_hashes = set(brisc_image_paths_by_hash)
    shared_hashes = classification_hashes & segmentation_hashes
    cross_dataset_overlap = [
        {
            "sha256": digest,
            "classification_paths": sorted(
                record.image_path
                for record in classification_records
                if record.sha256 == digest
            ),
            "brisc_paths": sorted(brisc_image_paths_by_hash[digest]),
        }
        for digest in sorted(shared_hashes)
    ]
    classification_findings.update(
        {
            "root": str(classification_root.expanduser().resolve()),
            "official_split_leakage_group_count": classification_findings[
                "cross_official_split_group_count"
            ],
            "derived_split_counts": _split_summaries(classification_records),
            "errors": classification_errors,
        }
    )
    segmentation_findings.update(
        {
            "root": str(segmentation_root.expanduser().resolve()),
            "official_split_leakage_group_count": segmentation_findings[
                "cross_official_split_group_count"
            ],
            "derived_split_counts": _split_summaries(segmentation_records),
            "errors": segmentation_errors,
            "manifest_image_record_count": len(segmentation_records),
            "mask_file_count": sum(
                bool(record.mask_path) for record in segmentation_records
            ),
            "verified_image_mask_pair_count": sum(
                bool(
                    record.mask_path
                    and record.mask_sha256
                    and record.mask_pixel_sha256
                    and not {
                        "unreadable_image",
                        "missing_or_invalid_mask",
                    }.intersection(record.flags)
                )
                for record in segmentation_records
            ),
        }
    )
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "proposed_split_proportions": SPLIT_PROPORTIONS,
        "split_method": (
            "Within each class-label stratum, unique image-SHA256 groups are "
            "shuffled by a seeded PRNG and greedily assigned to approximate "
            "70/15/15 group-count targets. Duplicate groups are atomic."
        ),
        "classification": classification_findings,
        "segmentation": segmentation_findings,
        "cross_dataset_overlap_errors": overlap_errors,
        "cross_dataset_image_hash_overlap": {
            "distinct_shared_sha256_count": len(shared_hashes),
            "image_path_count_in_classification": sum(
                record.sha256 in shared_hashes for record in classification_records
            ),
            "image_path_count_in_brisc_all_tasks": sum(
                len(brisc_image_paths_by_hash[digest]) for digest in shared_hashes
            ),
            "groups": cross_dataset_overlap,
        },
    }
    manifests = {
        "classification": [
            {
                "relative_image_path": record.image_path,
                "class_label": record.label,
                "original_split": record.original_split,
                "sha256": record.sha256,
                "duplicate_group_id": record.duplicate_group_id,
                "proposed_derived_split": record.proposed_split,
                "review_flags": ";".join(record.flags),
            }
            for record in sorted(
                classification_records, key=lambda record: record.image_path.casefold()
            )
        ],
        "segmentation": [
            {
                "relative_mri_path": record.image_path,
                "relative_mask_path": record.mask_path,
                "class_label": record.label,
                "original_split": record.original_split,
                "image_sha256": record.sha256,
                "duplicate_group_id": record.duplicate_group_id,
                "proposed_derived_split": record.proposed_split,
                "review_flags": ";".join(record.flags),
            }
            for record in sorted(
                segmentation_records, key=lambda record: record.image_path.casefold()
            )
        ],
    }
    return audit, manifests


def write_audit_artifacts(
    audit: dict[str, Any],
    manifests: dict[str, list[dict[str, str]]],
    *,
    manifest_dir: Path,
    report_dir: Path,
) -> dict[str, Path]:
    """Write timestamped audit/manifests without overwriting existing files."""
    destination = _resolve_output_dir(manifest_dir)
    reports_destination = _resolve_output_dir(report_dir)
    destination.mkdir(parents=True, exist_ok=True)
    reports_destination.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    while True:
        basename = f"leakage_audit_{timestamp}_{uuid4().hex[:8]}"
        paths = {
            "classification_manifest": destination / f"{basename}_classification.csv",
            "segmentation_manifest": destination / f"{basename}_segmentation.csv",
            "json_report": reports_destination / f"{basename}.json",
            "markdown_report": reports_destination / f"{basename}.md",
        }
        if not any(path.exists() for path in paths.values()):
            break

    _write_csv(
        paths["classification_manifest"],
        manifests["classification"],
        [
            "relative_image_path",
            "class_label",
            "original_split",
            "sha256",
            "duplicate_group_id",
            "proposed_derived_split",
            "review_flags",
        ],
    )
    _write_csv(
        paths["segmentation_manifest"],
        manifests["segmentation"],
        [
            "relative_mri_path",
            "relative_mask_path",
            "class_label",
            "original_split",
            "image_sha256",
            "duplicate_group_id",
            "proposed_derived_split",
            "review_flags",
        ],
    )
    with paths["json_report"].open("x", encoding="utf-8") as output:
        json.dump(audit, output, indent=2, ensure_ascii=False)
        output.write("\n")
    with paths["markdown_report"].open("x", encoding="utf-8") as output:
        output.write(render_audit_markdown(audit, paths))
    return paths


def _resolve_output_dir(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_csv(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    with path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_audit_markdown(audit: dict[str, Any], paths: dict[str, Path]) -> str:
    lines = [
        "# Dataset leakage audit and derived split report",
        "",
        f"Generated (UTC): {audit['generated_at_utc']}",
        f"Seed: `{audit['seed']}`",
        f"Derived proportions: `{audit['proposed_split_proportions']}`",
        "",
        "This audit uses exact SHA-256 hashes of image file bytes, not image "
        "dimensions or visual similarity. Source files were read only. "
        "Classification and segmentation manifests remain separate.",
        "",
    ]
    for key, title in (
        ("classification", "Brain Tumor MRI classification"),
        ("segmentation", "BRISC 2025 segmentation"),
    ):
        dataset = audit[key]
        lines.extend(
            [
                f"## {title}",
                "",
                f"- Dataset root: `{dataset['root']}`",
                f"- Source images: {dataset['image_count']:,}",
                f"- Distinct image hashes: {dataset['unique_image_hash_count']:,}",
                f"- Exact duplicate groups / files in those groups: "
                f"{dataset['duplicate_group_count']:,} / "
                f"{dataset['duplicate_file_count']:,}",
                f"- Exact duplicate groups crossing official splits: "
                f"{dataset['official_split_leakage_group_count']:,}",
                f"- Conflicting class-label groups: "
                f"{len(dataset['duplicate_class_label_conflicts']):,}",
                f"- Different masks for identical image content: "
                f"{len(dataset['conflicting_masks_for_identical_images']):,}",
            ]
        )
        if key == "segmentation":
            lines.append(
                f"- Manifest-linked masks retained: {dataset['mask_file_count']:,}"
            )
            lines.append(
                f"- Verified manifest-linked MRI/mask pairs: "
                f"{dataset['verified_image_mask_pair_count']:,}"
            )
        lines.extend(["", "### Derived split counts and class distributions", ""])
        lines.extend(
            [
                "| Derived split | Images / pairs | Duplicate groups | Class distribution |",
                "|---|---:|---:|---|",
            ]
        )
        for split, summary in dataset["derived_split_counts"].items():
            classes = ", ".join(
                f"{label}: {count:,}" for label, count in summary["class_counts"].items()
            )
            lines.append(
                f"| {split} | {summary['image_or_pair_count']:,} | "
                f"{summary['group_count']:,} | {classes} |"
            )
        lines.extend(["", "### Groups requiring review", ""])
        if dataset["cross_official_split_groups"]:
            lines.append(
                "Official split crossings; members remain together in derived "
                "partitions or as a flagged group:"
            )
            for group in dataset["cross_official_split_groups"]:
                lines.append(
                    f"- `{group['duplicate_group_id']}` ({group['count']} image paths, "
                    f"splits {', '.join(group['original_splits'])}): "
                    + ", ".join(f"`{path}`" for path in group["image_paths"])
                )
        if dataset["duplicate_class_label_conflicts"]:
            lines.append("Duplicate images assigned conflicting class labels:")
            for group in dataset["duplicate_class_label_conflicts"]:
                lines.append(
                    f"- `{group['duplicate_group_id']}` labels "
                    f"{', '.join(group['labels'])}: "
                    + ", ".join(f"`{path}`" for path in group["image_paths"])
                )
        if dataset["conflicting_masks_for_identical_images"]:
            lines.append("Identical MRI content associated with different masks:")
            for group in dataset["conflicting_masks_for_identical_images"]:
                lines.append(f"- `{group['duplicate_group_id']}`:")
                lines.extend(
                    f"  - `{pair['image_path']}` → `{pair['mask_path']}` "
                    f"(mask SHA-256 `{pair['mask_sha256']}`)"
                    for pair in group["masks"]
                )
        if not (
            dataset["cross_official_split_groups"]
            or dataset["duplicate_class_label_conflicts"]
            or dataset["conflicting_masks_for_identical_images"]
        ):
            lines.append("- No exact-hash groups flagged for review.")
        if dataset["errors"]:
            lines.extend(["", "Inspection errors:"])
            lines.extend(f"- {error}" for error in dataset["errors"])
        lines.append("")

    overlap = audit["cross_dataset_image_hash_overlap"]
    lines.extend(
        [
            "## Cross-dataset exact-image overlap",
            "",
            f"- Distinct shared image hashes: {overlap['distinct_shared_sha256_count']:,}",
            f"- Brain Tumor MRI paths whose hash is shared: "
            f"{overlap['image_path_count_in_classification']:,}",
            f"- BRISC MRI image paths across both tasks whose hash is shared: "
            f"{overlap['image_path_count_in_brisc_all_tasks']:,}",
            "- This overlap is reported only; the two task datasets are not merged.",
            "",
            "## Generated artifacts",
            "",
            f"- Classification CSV: `{paths['classification_manifest']}`",
            f"- Segmentation CSV: `{paths['segmentation_manifest']}`",
            "- JSON audit contains duplicate-group members and all shared-hash paths.",
            "",
            "## Limitations and recommendations",
            "",
            "- The proposed partitions are derived research splits, not official test sets.",
            (
                "- BRISC has exact duplicate image content across official training and "
                "test assignments. Duplicate-group members are assigned together in "
                "derived partitions or retained together as flagged groups if their "
                "masks conflict. Do not report the original test score as leakage-free."
                if audit["segmentation"]["official_split_leakage_group_count"]
                else "- No exact-byte duplicate group crosses BRISC's official split."
            ),
            "- Any `flagged` rows are retained in CSV for review and should not be used "
            "for training or evaluation until resolved.",
            "- SHA-256 detects byte-identical files only; it does not rule out "
            "visually similar, re-encoded, transformed, or same-patient images.",
            "- Confirm dataset licensing, patient-level independence, and annotation "
            "provenance before clinical or external-validity claims.",
            "",
        ]
    )
    return "\n".join(lines)

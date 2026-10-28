import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from mri_tumor.data.leakage import audit_datasets


_BRISC_FIELDS = [
    "relative_path",
    "task",
    "split",
    "tumor_label",
    "is_mask",
    "linked_image",
]


def _write_jpeg(path: Path, color: tuple[int, int, int] = (30, 80, 120)) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color=color).save(path, format="JPEG")
    return path.read_bytes()


def _write_manifest(root: Path, rows: list[dict[str, str]]) -> None:
    with (root / "manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as manifest:
        writer = csv.DictWriter(manifest, fieldnames=_BRISC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class DatasetLeakageTests(unittest.TestCase):
    def test_duplicate_group_crossing_official_splits_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            classification = root / "classification"
            segmentation = root / "segmentation"
            content = _write_jpeg(classification / "Training" / "glioma" / "a.jpg")
            (classification / "Testing" / "glioma").mkdir(parents=True)
            (classification / "Testing" / "glioma" / "b.jpg").write_bytes(content)

            audit, manifests = audit_datasets(classification, segmentation, seed=9)

        findings = audit["classification"]
        self.assertEqual(findings["duplicate_group_count"], 1)
        self.assertEqual(findings["official_split_leakage_group_count"], 1)
        records = manifests["classification"]
        self.assertEqual(len({row["duplicate_group_id"] for row in records}), 1)
        self.assertEqual(
            len({row["proposed_derived_split"] for row in records}), 1
        )
        self.assertNotEqual(records[0]["proposed_derived_split"], "flagged")
        self.assertIn("official_split_conflict_reassigned", records[0]["review_flags"])

    def test_conflicting_duplicate_labels_are_reported_and_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            classification = root / "classification"
            segmentation = root / "segmentation"
            content = _write_jpeg(classification / "Training" / "glioma" / "a.jpg")
            second = classification / "Training" / "meningioma" / "b.jpg"
            second.parent.mkdir(parents=True)
            second.write_bytes(content)

            audit, manifests = audit_datasets(classification, segmentation)

        self.assertEqual(
            len(audit["classification"]["duplicate_class_label_conflicts"]), 1
        )
        self.assertTrue(
            all(
                row["proposed_derived_split"] == "flagged"
                and "conflicting_labels" in row["review_flags"]
                for row in manifests["classification"]
            )
        )

    def test_segmentation_pairs_remain_associated_and_duplicate_groups_are_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            classification = root / "classification"
            segmentation = root / "segmentation"
            image_content = _write_jpeg(
                segmentation / "segmentation_task" / "train" / "images" / "a.jpg"
            )
            mask_paths = []
            rows = []
            for split, name in (("train", "a"), ("test", "b")):
                image_path = (
                    segmentation
                    / "segmentation_task"
                    / split
                    / "images"
                    / f"{name}.jpg"
                )
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(image_content)
                mask_path = (
                    segmentation
                    / "segmentation_task"
                    / split
                    / "masks"
                    / f"{name}.png"
                )
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("L", (12, 12), color=1).save(mask_path)
                mask_paths.append(mask_path.relative_to(segmentation).as_posix())
                image_relative = image_path.relative_to(segmentation).as_posix()
                rows.extend(
                    [
                        {
                            "relative_path": image_relative,
                            "task": "segmentation",
                            "split": split,
                            "tumor_label": "glioma",
                            "is_mask": "False",
                            "linked_image": "",
                        },
                        {
                            "relative_path": mask_path.relative_to(
                                segmentation
                            ).as_posix(),
                            "task": "segmentation",
                            "split": split,
                            "tumor_label": "glioma",
                            "is_mask": "True",
                            "linked_image": image_relative,
                        },
                    ]
                )
            _write_manifest(segmentation, rows)

            audit, manifests = audit_datasets(classification, segmentation, seed=7)

        pairs = manifests["segmentation"]
        self.assertEqual(audit["segmentation"]["mask_file_count"], 2)
        self.assertEqual(audit["segmentation"]["verified_image_mask_pair_count"], 2)
        self.assertEqual([row["relative_mask_path"] for row in pairs], sorted(mask_paths))
        self.assertEqual(len({row["duplicate_group_id"] for row in pairs}), 1)
        self.assertEqual(len({row["proposed_derived_split"] for row in pairs}), 1)
        self.assertEqual(
            audit["segmentation"]["official_split_leakage_group_count"], 1
        )

    def test_different_masks_for_identical_image_content_are_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            classification = root / "classification"
            segmentation = root / "segmentation"
            contents = _write_jpeg(
                segmentation / "segmentation_task" / "train" / "images" / "one.jpg"
            )
            rows = []
            for split, name, mask_value in (
                ("train", "one", 0),
                ("train", "two", 255),
            ):
                image_path = (
                    segmentation
                    / "segmentation_task"
                    / split
                    / "images"
                    / f"{name}.jpg"
                )
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(contents)
                mask_path = (
                    segmentation
                    / "segmentation_task"
                    / split
                    / "masks"
                    / f"{name}.png"
                )
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("L", (12, 12), color=mask_value).save(mask_path)
                image_relative = image_path.relative_to(segmentation).as_posix()
                rows.extend(
                    [
                        {
                            "relative_path": image_relative,
                            "task": "segmentation",
                            "split": split,
                            "tumor_label": "glioma",
                            "is_mask": "False",
                            "linked_image": "",
                        },
                        {
                            "relative_path": mask_path.relative_to(
                                segmentation
                            ).as_posix(),
                            "task": "segmentation",
                            "split": split,
                            "tumor_label": "glioma",
                            "is_mask": "True",
                            "linked_image": image_relative,
                        },
                    ]
                )
            _write_manifest(segmentation, rows)

            audit, manifests = audit_datasets(classification, segmentation)

        conflicts = audit["segmentation"]["conflicting_masks_for_identical_images"]
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(len(conflicts[0]["masks"]), 2)
        self.assertTrue(
            all(row["proposed_derived_split"] == "flagged" for row in manifests["segmentation"])
        )
        self.assertTrue(
            all(row["relative_mask_path"] for row in manifests["segmentation"])
        )

    def test_assignments_are_deterministic_and_paths_are_portable(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mri audit ") as temporary:
            root = Path(temporary)
            classification = root / "class data"
            segmentation = root / "seg data"
            for index in range(12):
                _write_jpeg(
                    classification
                    / "Training"
                    / "glioma"
                    / f"sample_{index}.jpg",
                    color=(index, 80, 120),
                )

            first_audit, first = audit_datasets(classification, segmentation, seed=123)
            second_audit, second = audit_datasets(classification, segmentation, seed=123)

        self.assertEqual(first["classification"], second["classification"])
        self.assertEqual(
            first_audit["classification"]["derived_split_counts"],
            second_audit["classification"]["derived_split_counts"],
        )
        for row in first["classification"]:
            self.assertNotIn("\\", row["relative_image_path"])
            self.assertFalse(Path(row["relative_image_path"]).is_absolute())
            self.assertNotIn("..", Path(row["relative_image_path"]).parts)

    def test_cross_dataset_overlap_is_reported_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            classification = root / "classification"
            segmentation = root / "segmentation"
            content = _write_jpeg(classification / "Training" / "glioma" / "one.jpg")
            brisc_image = segmentation / "classification_task" / "train" / "glioma" / "brisc.jpg"
            brisc_image.parent.mkdir(parents=True)
            brisc_image.write_bytes(content)
            _write_manifest(
                segmentation,
                [
                    {
                        "relative_path": brisc_image.relative_to(segmentation).as_posix(),
                        "task": "classification",
                        "split": "train",
                        "tumor_label": "glioma",
                        "is_mask": "False",
                        "linked_image": "",
                    }
                ],
            )

            audit, _ = audit_datasets(classification, segmentation)

        overlap = audit["cross_dataset_image_hash_overlap"]
        self.assertEqual(overlap["distinct_shared_sha256_count"], 1)
        self.assertEqual(overlap["image_path_count_in_classification"], 1)
        self.assertEqual(overlap["image_path_count_in_brisc_all_tasks"], 1)

    def test_missing_dataset_roots_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit, manifests = audit_datasets(
                root / "missing classification", root / "missing segmentation"
            )

        self.assertTrue(audit["classification"]["errors"])
        self.assertTrue(audit["segmentation"]["errors"])
        self.assertEqual(manifests["classification"], [])
        self.assertEqual(manifests["segmentation"], [])


if __name__ == "__main__":
    unittest.main()

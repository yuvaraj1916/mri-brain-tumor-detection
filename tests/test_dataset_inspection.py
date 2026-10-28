import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from mri_tumor.data.config import DatasetPaths, PROJECT_ROOT
from mri_tumor.data.inspection import (
    build_combined_duplicate_groups,
    inspect_dataset,
    write_reports,
)


class DatasetInspectionTests(unittest.TestCase):
    def test_counts_extensions_labels_and_sample_properties(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            class_dir = root / "Training" / "glioma"
            class_dir.mkdir(parents=True)
            Image.new("RGB", (13, 7), color="red").save(class_dir / "one.JPG")
            Image.new("L", (8, 9), color=1).save(class_dir / "two.png")
            (class_dir / "notes.txt").write_text("metadata", encoding="utf-8")

            report = inspect_dataset("fixture", root, sample_size=3)

        self.assertEqual(report["total_file_count"], 3)
        self.assertEqual(report["image_candidate_count"], 2)
        self.assertEqual(report["readable_image_count"], 2)
        self.assertEqual(report["file_extensions"][".jpg"]["count"], 1)
        self.assertEqual(report["file_extensions"][".png"]["count"], 1)
        self.assertEqual(report["file_extensions"][".txt"]["count"], 1)
        self.assertEqual(report["class_labels"], {"glioma": {"training": 2}})
        self.assertEqual(report["sample_image_properties"][0]["width"], 13)
        self.assertEqual(report["sample_image_properties"][0]["channels"], 3)

    def test_corrupt_image_is_counted_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corrupt = root / "broken.jpg"
            corrupt.write_bytes(b"not an image")

            report = inspect_dataset("fixture", root, hash_files=False)

        self.assertEqual(report["image_candidate_count"], 1)
        self.assertEqual(report["readable_image_count"], 0)
        self.assertEqual(report["unreadable_images"][0]["path"], "broken.jpg")

    def test_mask_discovery_and_manifest_linkage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_dir = root / "train" / "images"
            mask_dir = root / "train" / "masks"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            Image.new("RGB", (11, 11)).save(image_dir / "sample.jpg")
            Image.new("L", (11, 11)).save(mask_dir / "sample.png")
            Image.new("RGB", (11, 11)).save(image_dir / "mismatch.jpg")
            Image.new("L", (10, 11)).save(mask_dir / "mismatch.png")
            with (root / "manifest.csv").open(
                "w", encoding="utf-8", newline=""
            ) as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=["relative_path", "is_mask", "linked_image"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "relative_path": r"train\masks\sample.png",
                        "is_mask": "True",
                        "linked_image": r"train\images\sample.jpg",
                    }
                )

            report = inspect_dataset("fixture", root, sample_size=2)

        self.assertEqual(report["mask_file_count"], 2)
        self.assertEqual(report["manifest_summary"]["declared_mask_rows"], 1)
        self.assertEqual(report["manifest_summary"]["declared_mask_paths_found"], 1)
        self.assertEqual(report["image_mask_pairs"]["matched_pair_count"], 2)
        self.assertEqual(report["image_mask_pairs"]["candidate_image_count"], 2)
        manifest_pair = next(
            pair
            for pair in report["image_mask_pairs"]["pairs"]
            if pair["mask"].endswith("sample.png")
        )
        self.assertEqual(
            manifest_pair["match_method"], "manifest_link"
        )
        self.assertEqual(report["image_mask_pairs"]["verified_pair_count"], 1)
        self.assertEqual(report["image_mask_pairs"]["dimension_mismatch_count"], 1)
        self.assertEqual(report["image_mask_pairs"]["unmatched_image_count"], 0)

    def test_missing_directory_is_reported_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = inspect_dataset("missing", Path(temporary) / "absent")

        self.assertFalse(report["exists"])
        self.assertEqual(report["total_file_count"], 0)
        self.assertTrue(report["errors"])

    def test_hashes_report_identical_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.bin").write_bytes(b"identical")
            (root / "b.bin").write_bytes(b"identical")

            report = inspect_dataset("fixture", root)

        self.assertEqual(report["duplicate_file_groups"][0]["count"], 2)

    def test_hash_summary_flags_exact_duplicates_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("Training", "Testing"):
                directory = root / split / "glioma"
                directory.mkdir(parents=True)
                (directory / f"{split}.jpg").write_bytes(b"same image content")

            report = inspect_dataset("fixture", root)

        self.assertEqual(report["duplicate_summary"]["cross_split_group_count"], 1)
        self.assertEqual(
            report["duplicate_summary"]["cross_split_groups"][0]["splits"],
            ["test", "train"],
        )

    def test_cross_dataset_duplicates_excludes_within_dataset_copies(self) -> None:
        reports = [
            {
                "_hash_entries": [
                    {"dataset": "first", "path": "a", "sha256": "same"},
                    {"dataset": "first", "path": "b", "sha256": "same"},
                    {"dataset": "first", "path": "c", "sha256": "cross"},
                ]
            },
            {
                "_hash_entries": [
                    {"dataset": "second", "path": "d", "sha256": "cross"}
                ]
            },
        ]

        duplicates = build_combined_duplicate_groups(reports)

        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0]["sha256"], "cross")

    def test_relative_environment_paths_are_project_relative(self) -> None:
        paths = DatasetPaths.from_environment(
            {
                "MRI_CLASSIFICATION_DATASET_DIR": "fixture data/classification",
                "MRI_SEGMENTATION_DATASET_DIR": "fixture data/segmentation",
            }
        )

        self.assertEqual(
            paths.classification,
            PROJECT_ROOT / "fixture data" / "classification",
        )
        self.assertEqual(
            paths.segmentation,
            PROJECT_ROOT / "fixture data" / "segmentation",
        )

    def test_report_files_are_unique_and_machine_readable(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            root.mkdir()
            report = inspect_dataset("fixture", root, hash_files=False)
            first = write_reports([report], Path(temporary) / "reports", hash_files=False)
            second = write_reports([report], Path(temporary) / "reports", hash_files=False)

            self.assertNotEqual(first, second)
            self.assertTrue(first[0].is_file())
            self.assertTrue(first[1].is_file())
            self.assertEqual(json.loads(first[1].read_text())["datasets"][0]["name"], "fixture")


if __name__ == "__main__":
    unittest.main()

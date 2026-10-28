import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from mri_tumor.preprocessing import (
    PreprocessingConfig,
    PreprocessingError,
    image_to_tensor,
    load_image,
    load_mask,
    load_segmentation_pair,
    mask_to_tensor,
)
from scripts.visualize_datasets import (
    eligible_rows,
    find_latest_manifest_pair,
    read_manifest,
)


class PreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PreprocessingConfig(image_size=(8, 6))
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_rgb_and_grayscale_images_have_consistent_channel_shapes(self) -> None:
        rgb_path = self.root / "rgb.png"
        gray_path = self.root / "gray.png"
        Image.new("RGB", (11, 7), color=(255, 128, 0)).save(rgb_path)
        Image.new("L", (5, 9), color=64).save(gray_path)

        gray_default = load_image(gray_path, self.config)
        rgb_default = load_image(rgb_path, self.config)
        rgb_explicit = load_image(rgb_path, self.config, color_mode="rgb")
        preserved_rgb = load_image(
            rgb_path, self.config, color_mode="preserve"
        )

        self.assertEqual(gray_default.shape, (6, 8, 1))
        self.assertEqual(rgb_default.shape, (6, 8, 1))
        self.assertEqual(rgb_explicit.shape, (6, 8, 3))
        self.assertEqual(preserved_rgb.shape, (6, 8, 3))
        self.assertEqual(
            load_image(gray_path, self.config, color_mode="preserve").shape,
            (6, 8, 1),
        )
        self.assertAlmostEqual(float(rgb_explicit[0, 0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(rgb_explicit[0, 0, 1]), 128 / 255, places=5)

    def test_resize_can_be_disabled_and_image_sizes_need_not_match(self) -> None:
        first = self.root / "first.png"
        second = self.root / "second.png"
        Image.new("L", (11, 7), color=10).save(first)
        Image.new("L", (3, 13), color=20).save(second)

        resized = load_image(first, self.config)
        original = load_image(first, self.config, resize=False)
        second_original = load_image(second, self.config, resize=False)

        self.assertEqual(resized.shape, (6, 8, 1))
        self.assertEqual(original.shape, (7, 11, 1))
        self.assertEqual(second_original.shape, (13, 3, 1))

    def test_image_normalization_uses_configured_deterministic_range(self) -> None:
        path = self.root / "intensities.png"
        Image.fromarray(np.array([[0, 127, 255]], dtype=np.uint8)).save(path)
        config = PreprocessingConfig(
            image_size=(3, 1), image_normalization="minus_one_one"
        )

        normalized = load_image(path, config)
        repeated = load_image(path, config)

        self.assertEqual(normalized.dtype, np.float32)
        self.assertEqual(float(normalized.min()), -1.0)
        self.assertEqual(float(normalized.max()), 1.0)
        np.testing.assert_array_equal(normalized, repeated)

    def test_tensor_conversion_is_typed_and_preserves_label_integers(self) -> None:
        image = np.zeros((6, 8, 1), dtype=np.float32)
        mask = np.array([[0, 2], [3, 0]], dtype=np.uint8)

        if importlib.util.find_spec("torch") is None:
            with self.assertRaisesRegex(RuntimeError, "PyTorch is required"):
                image_to_tensor(image)
            with self.assertRaisesRegex(RuntimeError, "PyTorch is required"):
                mask_to_tensor(mask)
        else:
            image_tensor = image_to_tensor(image)
            mask_tensor = mask_to_tensor(mask)
            self.assertEqual(tuple(image_tensor.shape), (1, 6, 8))
            self.assertEqual(tuple(mask_tensor.shape), (2, 2))
            self.assertEqual(mask_tensor.tolist(), [[0, 2], [3, 0]])

    def test_masks_use_nearest_neighbor_and_preserve_categorical_values(self) -> None:
        path = self.root / "labels.png"
        labels = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        Image.fromarray(labels).save(path)

        resized = load_mask(path, self.config)

        self.assertEqual(resized.shape, (6, 8))
        self.assertEqual(set(np.unique(resized)), {0, 1, 2, 3})
        self.assertTrue(set(np.unique(resized)).issubset(set(np.unique(labels))))

    def test_mask_and_image_pair_dimensions_are_validated_before_resize(self) -> None:
        image_path = self.root / "mri.png"
        mask_path = self.root / "mask.png"
        Image.new("L", (7, 7), color=128).save(image_path)
        Image.fromarray(
            np.array([[0, 1], [0, 0]], dtype=np.uint8)
        ).save(mask_path)

        with self.assertRaisesRegex(PreprocessingError, "dimensions differ"):
            load_segmentation_pair(image_path, mask_path, self.config)

    def test_matching_pair_uses_equal_preprocessed_dimensions(self) -> None:
        image_path = self.root / "mri.png"
        mask_path = self.root / "mask.png"
        Image.new("RGB", (9, 7), color=(100, 100, 100)).save(image_path)
        labels = np.zeros((7, 9), dtype=np.uint8)
        labels[2:4, 3:5] = 1
        Image.fromarray(labels).save(mask_path)

        image, mask = load_segmentation_pair(image_path, mask_path, self.config)

        self.assertEqual(image.shape[:2], mask.shape)
        self.assertEqual(mask.shape, (6, 8))
        self.assertEqual(set(np.unique(mask)), {0, 1})

    def test_empty_or_all_foreground_masks_are_rejected(self) -> None:
        empty = self.root / "empty.png"
        foreground_only = self.root / "foreground.png"
        Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(empty)
        Image.fromarray(np.ones((4, 4), dtype=np.uint8)).save(foreground_only)

        for path in (empty, foreground_only):
            with self.subTest(path=path.name), self.assertRaisesRegex(
                PreprocessingError, "must contain background"
            ):
                load_mask(path, self.config)

    def test_missing_and_corrupted_image_and_mask_files_are_reported(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_image(self.root / "missing.png", self.config)
        with self.assertRaises(FileNotFoundError):
            load_mask(self.root / "missing-mask.png", self.config)

        for name in ("broken-image.png", "broken-mask.png"):
            path = self.root / name
            path.write_bytes(b"not a decodable image")
            with self.subTest(name=name), self.assertRaisesRegex(
                PreprocessingError, "Could not decode"
            ):
                (load_mask if "mask" in name else load_image)(path, self.config)


class ManifestVisualizationTests(unittest.TestCase):
    def test_flagged_segmentation_rows_are_excluded(self) -> None:
        rows = [
            {"proposed_derived_split": "training", "review_flags": ""},
            {
                "proposed_derived_split": "flagged",
                "review_flags": "conflicting_masks_for_identical_image",
            },
            {"proposed_derived_split": "training", "review_flags": "review_needed"},
        ]

        self.assertEqual(eligible_rows(rows), rows[:1])

    def test_latest_matching_manifest_pair_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for suffix in ("classification", "segmentation"):
                (directory / f"leakage_audit_20261001_a_{suffix}.csv").touch()
            (directory / "leakage_audit_20261002_b_segmentation.csv").touch()

            classification, segmentation = find_latest_manifest_pair(directory)

        self.assertEqual(classification.name, "leakage_audit_20261001_a_classification.csv")
        self.assertEqual(segmentation.name, "leakage_audit_20261001_a_segmentation.csv")

    def test_manifest_reader_requires_derived_split_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=["relative_image_path"])
                writer.writeheader()
                writer.writerow({"relative_image_path": "a.png"})

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                read_manifest(manifest, {"relative_image_path", "proposed_derived_split"})


if __name__ == "__main__":
    unittest.main()

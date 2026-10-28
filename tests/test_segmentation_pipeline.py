from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
from PIL import Image

from mri_tumor.preprocessing import PreprocessingConfig
from mri_tumor.segmentation.checkpoint import (
    load_segmentation_checkpoint,
    save_segmentation_checkpoint,
)
from mri_tumor.segmentation.data import (
    SegmentationDataset,
    binary_target,
    create_dataloaders,
    read_segmentation_manifest,
)
from mri_tumor.segmentation.metrics import (
    aggregate_segmentation_metrics,
    binary_segmentation_metrics,
)
from mri_tumor.segmentation.model import create_unet
from mri_tumor.segmentation.predict import predict_mask
from mri_tumor.segmentation.training import _loss_function, _run_epoch

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_FIELDS = [
    "relative_mri_path",
    "relative_mask_path",
    "class_label",
    "original_split",
    "image_sha256",
    "duplicate_group_id",
    "proposed_derived_split",
    "review_flags",
]


class SegmentationManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset_root = self.root / "brisc"
        self.manifest = self.root / "derived_segmentation.csv"
        self.config = PreprocessingConfig(image_size=(32, 32))
        self.rows: list[dict[str, str]] = []
        for index, split in enumerate(("training", "validation", "testing")):
            relative_image = f"segmentation_task/{split}/images/image_{index}.jpg"
            relative_mask = f"segmentation_task/{split}/masks/mask_{index}.png"
            image_path = self.dataset_root / relative_image
            mask_path = self.dataset_root / relative_mask
            image_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (32, 32), color=25 + index).save(image_path)
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[8:16, 8:16] = 255
            mask_path_image = Image.fromarray(mask)
            mask_path_image.save(mask_path)
            import hashlib

            self.rows.append(
                {
                    "relative_mri_path": relative_image,
                    "relative_mask_path": relative_mask,
                    "class_label": ("glioma", "meningioma", "pituitary")[index],
                    "original_split": split,
                    "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    "duplicate_group_id": "",
                    "proposed_derived_split": split,
                    "review_flags": "",
                }
            )
        self._write_manifest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_manifest(self) -> None:
        with self.manifest.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)

    def test_manifest_uses_explicit_pairs_and_separates_splits(self) -> None:
        partitions, summary = read_segmentation_manifest(
            self.manifest, self.dataset_root, preprocessing=self.config
        )

        self.assertEqual(
            {split: len(records) for split, records in partitions.items()},
            {"training": 1, "validation": 1, "testing": 1},
        )
        self.assertEqual(partitions["training"][0].relative_mask_path,
                         "segmentation_task/training/masks/mask_0.png")
        self.assertEqual(summary["exact_hash_overlap_counts"], {
            "training:validation": 0,
            "training:testing": 0,
            "validation:testing": 0,
        })

    def test_conflicting_flagged_pairs_are_excluded(self) -> None:
        import hashlib

        relative_image = "segmentation_task/flagged/images/conflict.jpg"
        image_path = self.dataset_root / relative_image
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (32, 32), color=99).save(image_path)
        image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        for index in range(2):
            relative_mask = f"segmentation_task/flagged/masks/conflict_{index}.png"
            mask_path = self.dataset_root / relative_mask
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[4 + index : 12 + index, 4:12] = 255
            Image.fromarray(mask).save(mask_path)
            self.rows.append(
                {
                    "relative_mri_path": relative_image,
                    "relative_mask_path": relative_mask,
                    "class_label": "glioma",
                    "original_split": "testing",
                    "image_sha256": image_hash,
                    "duplicate_group_id": "sha256-conflict",
                    "proposed_derived_split": "flagged",
                    "review_flags": "conflicting_masks_for_identical_image",
                }
            )
        self._write_manifest()

        partitions, summary = read_segmentation_manifest(
            self.manifest, self.dataset_root, preprocessing=self.config
        )

        self.assertEqual(sum(len(records) for records in partitions.values()), 3)
        self.assertEqual(summary["excluded_flagged_pair_count"], 2)
        self.assertEqual(
            summary["excluded_by_review_flag"]
            ["conflicting_masks_for_identical_image"],
            2,
        )

    def test_duplicate_group_crossing_partitions_is_rejected(self) -> None:
        self.rows[0]["duplicate_group_id"] = "sha256-shared"
        self.rows[1]["duplicate_group_id"] = "sha256-shared"
        self.rows[1]["image_sha256"] = self.rows[0]["image_sha256"]
        self._write_manifest()

        with self.assertRaisesRegex(ValueError, "crosses derived partitions"):
            read_segmentation_manifest(
                self.manifest, self.dataset_root, preprocessing=self.config
            )

    def test_unflagged_conflicting_masks_for_identical_mri_are_rejected(self) -> None:
        train_row = self.rows[0]
        train_row["duplicate_group_id"] = "sha256-unresolved"
        extra = dict(train_row)
        extra["relative_mask_path"] = "segmentation_task/training/masks/different.png"
        extra_mask = self.dataset_root / extra["relative_mask_path"]
        extra_mask.parent.mkdir(parents=True, exist_ok=True)
        other = np.zeros((32, 32), dtype=np.uint8)
        other[20:28, 20:28] = 255
        Image.fromarray(other).save(extra_mask)
        self.rows.append(extra)
        self._write_manifest()

        with self.assertRaisesRegex(ValueError, "Unresolved conflicting-mask"):
            read_segmentation_manifest(
                self.manifest, self.dataset_root, preprocessing=self.config
            )

    def test_missing_path_or_manifest_does_not_fall_back(self) -> None:
        self.rows[0]["relative_mask_path"] = "missing.png"
        self._write_manifest()
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            read_segmentation_manifest(
                self.manifest, self.dataset_root, preprocessing=self.config
            )
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaisesRegex(FileNotFoundError, "no official-split fallback"):
                from mri_tumor.segmentation.data import latest_segmentation_manifest

                latest_segmentation_manifest(Path(empty))

    def test_dataset_augmentation_is_training_only(self) -> None:
        partitions, _ = read_segmentation_manifest(
            self.manifest, self.dataset_root, preprocessing=self.config
        )
        training = SegmentationDataset(
            partitions["training"], self.config, training=True
        )
        validation = SegmentationDataset(
            partitions["validation"], self.config, training=False
        )
        self.assertTrue(training.training)
        self.assertFalse(validation.training)


class BinarySegmentationTests(unittest.TestCase):
    def test_grayscale_mask_boundary_maps_to_binary_semantics(self) -> None:
        mask = np.asarray([[0, 127, 128, 255]], dtype=np.uint8)

        target = binary_target(mask, threshold=128)

        self.assertEqual(target.tolist(), [[0, 0, 1, 1]])
        self.assertEqual(set(np.unique(target)), {0, 1})

    def test_dice_iou_and_empty_ground_truth_convention(self) -> None:
        metrics = binary_segmentation_metrics(
            np.asarray([[1, 1, 0]]), np.asarray([[1, 0, 1]])
        )
        self.assertEqual(metrics["dice"], 0.5)
        self.assertAlmostEqual(metrics["iou"], 1 / 3)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        empty = binary_segmentation_metrics(
            np.zeros((3, 3), dtype=np.uint8), np.zeros((3, 3), dtype=np.uint8)
        )
        self.assertEqual(empty["dice"], 1.0)
        self.assertEqual(empty["iou"], 1.0)
        aggregate = aggregate_segmentation_metrics([metrics, empty])
        self.assertEqual(aggregate["empty_ground_truth_images"], 1)

    def test_all_foreground_prediction_is_counted_as_false_positives(self) -> None:
        truth = np.asarray([[1, 0], [0, 0]], dtype=np.uint8)
        prediction = np.ones((2, 2), dtype=np.uint8)

        metrics = binary_segmentation_metrics(truth, prediction)

        self.assertEqual(metrics["dice"], 0.4)
        self.assertEqual(metrics["iou"], 0.25)
        self.assertEqual(metrics["precision"], 0.25)
        self.assertEqual(metrics["recall"], 1.0)

    def test_empty_ground_truth_with_foreground_prediction_scores_zero(self) -> None:
        metrics = binary_segmentation_metrics(
            np.zeros((2, 2), dtype=np.uint8),
            np.ones((2, 2), dtype=np.uint8),
        )

        self.assertEqual(metrics["dice"], 0.0)
        self.assertEqual(metrics["iou"], 0.0)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)

    def test_metric_shape_mismatch_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            binary_segmentation_metrics(np.zeros((2, 2)), np.zeros((3, 3)))

    @unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch is required for model tests")
    def test_unet_returns_single_channel_logits_at_input_size(self) -> None:
        import torch

        for channels in (1, 3):
            model = create_unet(input_channels=channels, base_channels=4)
            inputs = torch.rand(1, channels, 65, 71)
            logits = model(inputs)
            self.assertEqual(tuple(logits.shape), (1, 1, 65, 71))
            self.assertTrue(torch.isfinite(logits).all())

    @unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch is required for model tests")
    def test_combined_bce_dice_loss_is_finite(self) -> None:
        import torch

        logits = torch.zeros(2, 1, 16, 16, requires_grad=True)
        targets = torch.zeros(2, 1, 16, 16)
        targets[:, :, 4:10, 5:11] = 1
        loss = _loss_function(torch, 0.5)(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)

    @unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch is required for model tests")
    def test_bce_component_consumes_logits_without_prior_sigmoid(self) -> None:
        import torch

        logits = torch.tensor([[[[2.0, -2.0]]]])
        targets = torch.tensor([[[[1.0, 0.0]]]])

        actual = _loss_function(torch, 0.0)(logits, targets)
        expected = torch.nn.BCEWithLogitsLoss()(logits, targets)

        torch.testing.assert_close(actual, expected)
        self.assertAlmostEqual(float(actual), 0.126928, places=5)

    @unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch is required for training tests")
    def test_epoch_checks_gradients_and_records_output_statistics(self) -> None:
        import torch

        model = torch.nn.Conv2d(1, 1, kernel_size=1)
        images = torch.ones(1, 1, 2, 2)
        targets = torch.zeros(1, 1, 2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        loss, metrics, image_count, batch_count = _run_epoch(
            model,
            [(images, targets)],
            loss_function=_loss_function(torch, 0.5),
            device=torch.device("cpu"),
            optimizer=optimizer,
        )

        self.assertTrue(np.isfinite(loss))
        self.assertEqual((image_count, batch_count), (1, 1))
        self.assertIn("mean_iou", metrics)
        self.assertIn("predicted_foreground_fraction", metrics)
        self.assertIn("mean_probability", metrics)
        self.assertIn("probability_std", metrics)
        self.assertTrue(np.isfinite(metrics["mean_probability"]))
        self.assertTrue(np.isfinite(metrics["probability_std"]))
        self.assertGreaterEqual(metrics["mean_probability"], 0.0)
        self.assertLessEqual(metrics["mean_probability"], 1.0)
        self.assertGreaterEqual(metrics["probability_std"], 0.0)

    @unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch is required for model tests")
    def test_checkpoint_round_trip(self) -> None:
        import torch

        model = create_unet(input_channels=1, base_channels=4)
        with tempfile.TemporaryDirectory() as directory:
            path = save_segmentation_checkpoint(
                Path(directory) / "model.pt",
                {"model_state_dict": model.state_dict(), "best_epoch": 1},
            )
            checkpoint = load_segmentation_checkpoint(path)
            restored = create_unet(input_channels=1, base_channels=4)
            restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.assertEqual(checkpoint["best_epoch"], 1)
        self.assertEqual(
            set(checkpoint["model_state_dict"]), set(model.state_dict())
        )
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key])

    @unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch is required for model tests")
    def test_initialization_is_reproducible_when_seed_is_reset(self) -> None:
        import torch

        torch.manual_seed(20261012)
        first = create_unet(input_channels=1, base_channels=4).state_dict()
        torch.manual_seed(20261012)
        second = create_unet(input_channels=1, base_channels=4).state_dict()
        self.assertEqual(set(first), set(second))
        for key in first:
            torch.testing.assert_close(first[key], second[key])

    @unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch is required for model tests")
    def test_prediction_shape_and_threshold(self) -> None:
        import torch

        class FixedLogits(torch.nn.Module):
            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                logits = torch.full(
                    (inputs.shape[0], 1, inputs.shape[-2], inputs.shape[-1]),
                    -2.0,
                    device=inputs.device,
                )
                logits[:, :, :4, :4] = 2.0
                return logits

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mri.png"
            Image.new("L", (24, 20), color=40).save(path)
            result = predict_mask(
                FixedLogits(),
                path,
                PreprocessingConfig(image_size=(32, 32)),
                threshold=0.5,
            )
        self.assertEqual(result["mask"].shape, (32, 32))
        self.assertTrue(result["mask"][:4, :4].all())
        self.assertFalse(result["mask"][8:, 8:].any())
        self.assertAlmostEqual(
            float(result["probabilities"][0, 0]),
            float(torch.sigmoid(torch.tensor(2.0))),
            places=6,
        )
        self.assertEqual(result["original_size"], (24, 20))

    @unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch is required for augmentation tests")
    def test_horizontal_augmentation_keeps_image_and_mask_aligned(self) -> None:
        import torch

        mask = torch.zeros(1, 16, 16)
        mask[:, 3:8, 2:6] = 1
        image = mask.clone()
        dataset = SegmentationDataset(
            [],
            PreprocessingConfig(image_size=(16, 16)),
            training=True,
            rotation_degrees=0,
            horizontal_flip_probability=1,
        )
        with patch("mri_tumor.segmentation.data.random.random", return_value=0.0):
            augmented_image, augmented_mask = dataset._augment(image, mask)

        torch.testing.assert_close(augmented_image, augmented_mask)
        self.assertTrue(torch.equal(augmented_mask, torch.flip(mask, dims=(-1,))))

    @unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch is required for augmentation tests")
    def test_rotation_augmentation_uses_the_same_grid_for_image_and_mask(self) -> None:
        import torch

        mask = torch.zeros(1, 16, 16)
        mask[:, 3:8, 2:6] = 1
        dataset = SegmentationDataset(
            [],
            PreprocessingConfig(image_size=(16, 16)),
            training=True,
            rotation_degrees=90,
            horizontal_flip_probability=0,
        )
        with patch(
            "mri_tumor.segmentation.data.random.random", return_value=0.9
        ), patch("mri_tumor.segmentation.data.random.uniform", return_value=90):
            augmented_image, augmented_mask = dataset._augment(mask.clone(), mask.clone())

        torch.testing.assert_close(augmented_image, augmented_mask, atol=1e-6, rtol=0)

    @unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch is required for data-loader tests")
    def test_loaders_keep_validation_and_test_unaugmented(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            rows = []
            import hashlib

            for index, split in enumerate(("training", "validation", "testing")):
                image_path = root / f"{split}.jpg"
                mask_path = root / f"{split}.png"
                Image.new("L", (32, 32), color=20 + index).save(image_path)
                mask = np.zeros((32, 32), dtype=np.uint8)
                mask[3:8, 3:8] = 255
                Image.fromarray(mask).save(mask_path)
                rows.append(
                    {
                        "relative_mri_path": image_path.name,
                        "relative_mask_path": mask_path.name,
                        "class_label": "glioma",
                        "original_split": split,
                        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                        "duplicate_group_id": "",
                        "proposed_derived_split": split,
                        "review_flags": "",
                    }
                )
            with manifest.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            config = PreprocessingConfig(image_size=(32, 32))
            partitions, _ = read_segmentation_manifest(
                manifest, root, preprocessing=config
            )
            loaders = create_dataloaders(
                partitions,
                config,
                batch_size=1,
                num_workers=0,
                seed=123,
            )
        self.assertTrue(loaders["training"].dataset.training)
        self.assertFalse(loaders["validation"].dataset.training)
        self.assertFalse(loaders["testing"].dataset.training)

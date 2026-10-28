import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from mri_tumor.classification.checkpoint import load_checkpoint, save_checkpoint
from mri_tumor.classification.data import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    ClassificationDataset,
    class_distribution_report,
    latest_classification_manifest,
    read_classification_manifest,
    verify_split_hashes,
)
from mri_tumor.classification.metrics import classification_metrics
from mri_tumor.classification.model import create_model
from mri_tumor.classification.predict import preprocess_for_prediction
from mri_tumor.classification.training import TrainingConfig
from mri_tumor.preprocessing import PreprocessingConfig

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_MANIFEST_FIELDS = [
    "relative_image_path",
    "class_label",
    "original_split",
    "sha256",
    "duplicate_group_id",
    "proposed_derived_split",
    "review_flags",
]


class ClassificationManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset_root = self.root / "classification"
        self.manifest = self.root / "derived.csv"
        rows = []
        for split_index, split in enumerate(("training", "validation", "testing")):
            relative = f"{split}/glioma/sample.png"
            image_path = self.dataset_root / relative
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (8, 8), color=split_index * 10 + 1).save(image_path)
            rows.append(
                {
                    "relative_image_path": relative,
                    "class_label": "glioma",
                    "original_split": "training",
                    "sha256": f"{split_index + 1:064x}",
                    "duplicate_group_id": "",
                    "proposed_derived_split": split,
                    "review_flags": "",
                }
            )
        with self.manifest.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=_MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_derived_manifest_loads_paths_and_fixed_labels(self) -> None:
        partitions = read_classification_manifest(self.manifest, self.dataset_root)

        self.assertEqual({name: len(rows) for name, rows in partitions.items()}, {
            "training": 1,
            "validation": 1,
            "testing": 1,
        })
        record = partitions["training"][0]
        self.assertEqual(record.relative_image_path, "training/glioma/sample.png")
        self.assertEqual(record.class_label, "glioma")
        self.assertEqual(record.label_index, CLASS_TO_INDEX["glioma"])
        self.assertEqual(CLASS_NAMES, ("glioma", "meningioma", "notumor", "pituitary"))

    def test_review_flagged_rows_are_not_included(self) -> None:
        text = self.manifest.read_text(encoding="utf-8")
        text += (
            "flagged/glioma/sample.png,glioma,testing,"
            f"{99:064x},,flagged,conflicting_labels\n"
        )
        self.manifest.write_text(text, encoding="utf-8")

        partitions = read_classification_manifest(self.manifest, self.dataset_root)

        self.assertEqual(sum(map(len, partitions.values())), 3)

    def test_no_official_manifest_fallback_and_missing_images_fail_clearly(self) -> None:
        empty_manifest_dir = self.root / "no-manifest"
        empty_manifest_dir.mkdir()
        with self.assertRaisesRegex(FileNotFoundError, "not used as a fallback"):
            latest_classification_manifest(empty_manifest_dir)

        missing_image_manifest = self.root / "missing-image.csv"
        with missing_image_manifest.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=_MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerow(
                {
                    "relative_image_path": "not-here.png",
                    "class_label": "glioma",
                    "original_split": "training",
                    "sha256": f"{1:064x}",
                    "duplicate_group_id": "",
                    "proposed_derived_split": "training",
                    "review_flags": "",
                }
            )
        with self.assertRaisesRegex(FileNotFoundError, "Manifest image is missing"):
            read_classification_manifest(missing_image_manifest, self.dataset_root)

    def test_cross_partition_duplicate_hash_is_rejected(self) -> None:
        shared_hashes = {
            "training": {"a" * 64},
            "validation": {"a" * 64},
            "testing": {"b" * 64},
        }
        with self.assertRaisesRegex(ValueError, "Exact image hashes occur"):
            verify_split_hashes(shared_hashes)
        self.assertEqual(
            verify_split_hashes(
                {
                    "training": {"a" * 64},
                    "validation": {"b" * 64},
                    "testing": {"c" * 64},
                }
            ),
            {
                "training:validation": 0,
                "training:testing": 0,
                "validation:testing": 0,
            },
        )

    def test_split_class_distribution_uses_fixed_four_class_mapping(self) -> None:
        partitions = read_classification_manifest(self.manifest, self.dataset_root)

        counts = class_distribution_report(partitions)

        self.assertEqual(counts["training"]["glioma"], 1)
        self.assertEqual(counts["training"]["pituitary"], 0)
        self.assertEqual(counts["testing"]["glioma"], 1)

    def test_dataset_marks_augmentation_for_training_only(self) -> None:
        partitions = read_classification_manifest(self.manifest, self.dataset_root)
        preprocessing = PreprocessingConfig(image_size=(8, 8))
        training = ClassificationDataset(
            partitions["training"], preprocessing, training=True
        )
        validation = ClassificationDataset(
            partitions["validation"], preprocessing, training=False
        )
        self.assertTrue(training.training)
        self.assertFalse(validation.training)
        self.assertEqual(training.records[0].label_index, CLASS_TO_INDEX["glioma"])

    def test_malformed_class_labels_are_rejected(self) -> None:
        text = self.manifest.read_text(encoding="utf-8")
        self.manifest.write_text(text.replace(",glioma,", ",unknown-class,", 1))
        with self.assertRaisesRegex(ValueError, "Unsupported class label"):
            read_classification_manifest(self.manifest, self.dataset_root)


class ClassificationMetricsTests(unittest.TestCase):
    def test_metrics_report_accuracy_per_class_and_confusion_matrix(self) -> None:
        metrics = classification_metrics(
            [0, 1, 2, 3, 0],
            [0, 2, 2, 3, 1],
            class_names=CLASS_NAMES,
        )

        self.assertAlmostEqual(metrics["accuracy"], 0.6)
        self.assertAlmostEqual(metrics["macro_f1"], 7 / 12)
        self.assertAlmostEqual(metrics["weighted_f1"], 0.6)
        self.assertEqual(metrics["per_class"]["glioma"]["support"], 2)
        self.assertAlmostEqual(metrics["per_class"]["glioma"]["f1"], 2 / 3)
        self.assertEqual(
            metrics["confusion_matrix"],
            [
                [1, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        )

    def test_metrics_reject_empty_or_mismatched_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            classification_metrics([], [], class_names=CLASS_NAMES)
        with self.assertRaisesRegex(ValueError, "same number"):
            classification_metrics([0], [0, 1], class_names=CLASS_NAMES)


class OptionalTorchTests(unittest.TestCase):
    @unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch is not installed.")
    def test_model_output_shape_and_reproducible_initialization(self) -> None:
        import torch

        torch.manual_seed(41)
        first = create_model(input_channels=1, num_classes=4)
        torch.manual_seed(41)
        second = create_model(input_channels=1, num_classes=4)
        for key, tensor in first.state_dict().items():
            torch.testing.assert_close(tensor, second.state_dict()[key])
        first.eval()
        with torch.inference_mode():
            logits = first(torch.zeros(2, 1, 64, 80))
        self.assertEqual(tuple(logits.shape), (2, 4))

    @unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch is not installed.")
    def test_dataset_returns_tensor_label_and_augments_training_only(self) -> None:
        from mri_tumor.classification.data import ClassificationRecord

        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "scan.png"
            Image.fromarray(np.tile(np.arange(8, dtype=np.uint8), (8, 1))).save(
                image_path
            )
            record = ClassificationRecord(
                image_path=image_path,
                relative_image_path="scan.png",
                class_label="glioma",
                label_index=CLASS_TO_INDEX["glioma"],
                split="training",
                sha256="a" * 64,
            )
            preprocessing = PreprocessingConfig(image_size=(8, 8))
            training = ClassificationDataset(
                [record],
                preprocessing,
                training=True,
                rotation_degrees=0,
                horizontal_flip_probability=1,
            )
            validation = ClassificationDataset(
                [record],
                preprocessing,
                training=False,
                rotation_degrees=10,
                horizontal_flip_probability=1,
            )
            training_image, training_label = training[0]
            validation_image, validation_label = validation[0]

        self.assertEqual(tuple(training_image.shape), (1, 8, 8))
        self.assertEqual(training_label, CLASS_TO_INDEX["glioma"])
        self.assertEqual(validation_label, training_label)
        self.assertTrue(
            np.array_equal(
                training_image.numpy(),
                np.flip(validation_image.numpy(), axis=2),
            )
        )

    @unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch is not installed.")
    def test_checkpoint_save_and_load_round_trip(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "checkpoint.pt"
            expected = {"weights": torch.tensor([1.5, 2.5]), "epoch": 4}
            save_checkpoint(path, expected)
            actual = load_checkpoint(path)
        torch.testing.assert_close(actual["weights"], expected["weights"])
        self.assertEqual(actual["epoch"], 4)

    @unittest.skipUnless(not _TORCH_AVAILABLE, "This test covers missing PyTorch.")
    def test_missing_pytorch_is_explicit(self) -> None:
        from mri_tumor.classification.data import (
            ClassificationDataset,
            ClassificationRecord,
            require_torch,
        )

        with self.assertRaisesRegex(RuntimeError, "PyTorch is required"):
            require_torch()
        with self.assertRaisesRegex(RuntimeError, "PyTorch is required"):
            create_model()
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "scan.png"
            Image.new("L", (4, 4), color=12).save(image_path)
            record = ClassificationRecord(
                image_path=image_path,
                relative_image_path="scan.png",
                class_label="glioma",
                label_index=CLASS_TO_INDEX["glioma"],
                split="training",
                sha256="a" * 64,
            )
            dataset = ClassificationDataset(
                [record], PreprocessingConfig(image_size=(4, 4))
            )
            with self.assertRaisesRegex(RuntimeError, "PyTorch is required"):
                dataset[0]

    @unittest.skipUnless(not _TORCH_AVAILABLE, "Prediction requires PyTorch.")
    def test_missing_checkpoint_never_returns_random_prediction(self) -> None:
        from mri_tumor.classification.predict import predict_image

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scan.png"
            Image.new("L", (5, 5), color=1).save(path)
            with self.assertRaisesRegex(FileNotFoundError, "checkpoint does not exist"):
                predict_image(path, Path(temporary) / "missing.pt")


class PredictionPreprocessingTests(unittest.TestCase):
    def test_prediction_preprocessing_reports_dimensions_and_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "sample.png"
            Image.new("RGB", (17, 9), color=(20, 80, 150)).save(image_path)
            preprocessing = PreprocessingConfig(
                image_size=(12, 10), color_mode="grayscale"
            )

            image, info = preprocess_for_prediction(image_path, preprocessing)

        self.assertEqual(image.shape, (10, 12, 1))
        self.assertEqual(info["original_size"], [17, 9])
        self.assertEqual(info["processed_size"], [12, 10])
        self.assertEqual(info["normalization"], "zero_one")


class TrainingConfigurationTests(unittest.TestCase):
    def test_training_configuration_rejects_invalid_runtime_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size and epochs"):
            TrainingConfig(batch_size=0)
        with self.assertRaisesRegex(ValueError, "Unsupported device"):
            TrainingConfig(device="tpu")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from mri_tumor.classification.checkpoint import save_checkpoint
from mri_tumor.classification.model import create_model
from mri_tumor.inference import InferenceError, predict_mri
from mri_tumor.preprocessing import PreprocessingConfig
from mri_tumor.segmentation.checkpoint import save_segmentation_checkpoint
from mri_tumor.segmentation.model import create_unet
from mri_tumor.utils.device import get_device

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch is required for inference tests.")
class CombinedInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch

        self.torch = torch
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.classification_checkpoint = self.root / "classifier.pt"
        self.segmentation_checkpoint = self.root / "segmenter.pt"
        self.image_path = self.root / "input.png"

        torch.manual_seed(17)
        class_preprocessing = PreprocessingConfig(
            image_size=(48, 32), color_mode="grayscale"
        )
        segmentation_preprocessing = PreprocessingConfig(
            image_size=(64, 48), color_mode="rgb"
        )
        class_model_config = {
            "input_channels": 1,
            "num_classes": 4,
            "dropout": 0.25,
        }
        segmentation_model_config = {"input_channels": 3, "base_channels": 4}
        save_checkpoint(
            self.classification_checkpoint,
            {
                "model_state_dict": create_model(**class_model_config).state_dict(),
                "model_config": class_model_config,
                "class_to_index": {
                    "glioma": 0,
                    "meningioma": 1,
                    "notumor": 2,
                    "pituitary": 3,
                },
                "preprocessing_config": class_preprocessing.__dict__,
            },
        )
        save_segmentation_checkpoint(
            self.segmentation_checkpoint,
            {
                "model_state_dict": create_unet(**segmentation_model_config).state_dict(),
                "model_config": segmentation_model_config,
                "preprocessing_config": segmentation_preprocessing.__dict__,
                "training_config": {"mask_threshold": 128},
            },
        )
        pixels = np.zeros((70, 90, 3), dtype=np.uint8)
        pixels[15:55, 20:75] = (80, 120, 160)
        Image.fromarray(pixels).save(self.image_path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_predict_mri_returns_finite_outputs_in_aligned_dimensions(self) -> None:
        result = predict_mri(
            self.image_path,
            self.classification_checkpoint,
            self.segmentation_checkpoint,
        )

        self.assertIn(
            result["predicted_class"],
            ("glioma", "meningioma", "notumor", "pituitary"),
        )
        self.assertEqual(
            tuple(result["class_probabilities"]),
            ("glioma", "meningioma", "notumor", "pituitary"),
        )
        self.assertAlmostEqual(sum(result["class_probabilities"].values()), 1.0)
        probability_map = result["segmentation_probability_map"]
        self.assertEqual(probability_map.shape, (48, 64))
        self.assertTrue(np.isfinite(probability_map).all())
        self.assertTrue(np.all((probability_map >= 0) & (probability_map <= 1)))
        self.assertEqual(result["threshold"], 0.5)
        self.assertTrue(
            np.array_equal(result["tumor_mask"], probability_map >= 0.5)
        )
        self.assertEqual(result["original_image"].shape, (70, 90, 3))
        self.assertEqual(result["aligned_tumor_mask"].shape, (70, 90))
        self.assertEqual(result["overlay"].shape, (70, 90, 3))
        expected_aligned_mask = np.asarray(
            Image.fromarray(result["tumor_mask"].astype(np.uint8) * 255).resize(
                (90, 70), resample=Image.Resampling.NEAREST
            )
        ) >= 128
        self.assertTrue(
            np.array_equal(result["aligned_tumor_mask"], expected_aligned_mask)
        )
        self.assertEqual(result["image_info"]["classification_processed_size"], [32, 48])
        self.assertEqual(result["image_info"]["segmentation_processed_size"], [48, 64])
        self.assertIn(result["image_info"]["device"], ("mps", "cpu"))
        self.assertNotIn("checkpoint_path", result)
        self.assertNotIn("image_path", result)

    def test_missing_checkpoint_has_a_safe_error(self) -> None:
        with self.assertRaisesRegex(
            InferenceError, "required trained model checkpoint is unavailable"
        ) as raised:
            predict_mri(
                self.image_path,
                self.root / "missing-classifier.pt",
                self.segmentation_checkpoint,
            )
        self.assertNotIn(str(self.root), str(raised.exception))

    def test_missing_input_image_has_a_safe_error(self) -> None:
        with self.assertRaisesRegex(InferenceError, "input MRI image is unavailable"):
            predict_mri(
                self.root / "missing.png",
                self.classification_checkpoint,
                self.segmentation_checkpoint,
            )

    def test_corrupted_image_has_a_safe_error(self) -> None:
        invalid_image = self.root / "invalid.png"
        invalid_image.write_bytes(b"not an image")
        with self.assertRaisesRegex(InferenceError, "input MRI image could not be read"):
            predict_mri(
                invalid_image,
                self.classification_checkpoint,
                self.segmentation_checkpoint,
            )

    def test_inference_uses_cpu_when_mps_is_unavailable(self) -> None:
        with (
            patch.object(
                self.torch.backends.mps, "is_available", return_value=False
            ),
            patch("mri_tumor.inference.get_device", wraps=get_device) as select_device,
        ):
            result = predict_mri(
                self.image_path,
                self.classification_checkpoint,
                self.segmentation_checkpoint,
            )

        select_device.assert_called_once_with("cpu")
        self.assertEqual(result["image_info"]["device"], "cpu")
        self.assertTrue(
            np.isfinite(result["segmentation_probability_map"]).all()
        )

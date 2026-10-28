import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mri_tumor.data.config import DatasetPaths, PROJECT_ROOT
from mri_tumor.utils.device import get_device


class DatasetPathsTests(unittest.TestCase):
    def test_defaults_are_project_relative(self) -> None:
        paths = DatasetPaths.from_environment({})

        self.assertEqual(
            paths.classification,
            PROJECT_ROOT / "datasets" / "classification" / "Brain Tumor MRI",
        )
        self.assertEqual(
            paths.segmentation,
            PROJECT_ROOT / "datasets" / "segmentation" / "brisc2025",
        )

    def test_environment_overrides_support_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mri data ") as temporary:
            root = Path(temporary)
            classification = root / "classification scans"
            segmentation = root / "segmentation scans"
            paths = DatasetPaths.from_environment(
                {
                    "MRI_CLASSIFICATION_DATASET_DIR": str(classification),
                    "MRI_SEGMENTATION_DATASET_DIR": str(segmentation),
                }
            )

        self.assertEqual(paths.classification, classification)
        self.assertEqual(paths.segmentation, segmentation)

    def test_relative_overrides_are_project_relative(self) -> None:
        paths = DatasetPaths.from_environment(
            {
                "MRI_CLASSIFICATION_DATASET_DIR": str(
                    Path("external datasets") / "classification"
                )
            }
        )

        self.assertEqual(
            paths.classification,
            PROJECT_ROOT / "external datasets" / "classification",
        )

    def test_blank_override_uses_default(self) -> None:
        paths = DatasetPaths.from_environment(
            {"MRI_SEGMENTATION_DATASET_DIR": "  "}
        )

        self.assertEqual(
            paths.segmentation,
            PROJECT_ROOT / "datasets" / "segmentation" / "brisc2025",
        )

    def test_require_returns_existing_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mri dataset ") as directory:
            paths = DatasetPaths(classification=Path(directory))

            self.assertEqual(
                paths.require("classification"), Path(directory).resolve()
            )

    def test_require_reports_missing_directory_and_override_name(self) -> None:
        paths = DatasetPaths(
            segmentation=PROJECT_ROOT / "missing test directory"
        )

        with self.assertRaisesRegex(
            FileNotFoundError, "MRI_SEGMENTATION_DATASET_DIR"
        ):
            paths.require("segmentation")

    def test_unknown_dataset_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported dataset"):
            DatasetPaths().require("other")  # type: ignore[arg-type]

    def test_auto_device_prefers_cuda_then_mps_then_cpu(self) -> None:
        for cuda_available, mps_available, expected in (
            (True, True, "cuda"),
            (False, True, "mps"),
            (False, False, "cpu"),
        ):
            torch_mock = SimpleNamespace(
                device=lambda name: name,
                cuda=SimpleNamespace(is_available=lambda: cuda_available),
                backends=SimpleNamespace(
                    mps=SimpleNamespace(is_available=lambda: mps_available)
                ),
            )
            with self.subTest(expected=expected), patch.dict(
                "sys.modules", {"torch": torch_mock}
            ):
                self.assertEqual(get_device(), expected)

    def test_explicit_unavailable_device_raises(self) -> None:
        torch_mock = SimpleNamespace(
            device=lambda name: name,
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: False)
            ),
        )
        with patch.dict("sys.modules", {"torch": torch_mock}):
            with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
                get_device("cuda")

    def test_unsupported_device_preference_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported device"):
            get_device("tpu")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()

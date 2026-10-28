from __future__ import annotations

import base64
import http.client
from io import BytesIO
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.serve_app import MAX_UPLOAD_BYTES, _existing_checkpoint, make_request_handler


class WebAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.web_root = self.root / "web"
        self.web_root.mkdir()
        (self.web_root / "index.html").write_text("<h1>MRI</h1>", encoding="utf-8")
        self.classifier = self.root / "classifier.pt"
        self.segmenter = self.root / "segmenter.pt"
        self.classifier.touch()
        self.segmenter.touch()
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_request_handler(self.classifier, self.segmenter, self.web_root),
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=5
        )

    def tearDown(self) -> None:
        self.connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.temporary.cleanup()

    @staticmethod
    def _multipart(filename: str, content: bytes, boundary: str = "test-boundary") -> tuple[str, bytes]:
        body = (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"image\"; filename=\"{filename}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
        return f"multipart/form-data; boundary={boundary}", body

    def _post(self, filename: str, content: bytes, boundary: str = "test-boundary") -> tuple[int, dict]:
        content_type, body = self._multipart(filename, content, boundary)
        self.connection.request(
            "POST",
            "/api/analyze",
            body,
            {"Content-Type": content_type, "Content-Length": str(len(body))},
        )
        response = self.connection.getresponse()
        return response.status, json.loads(response.read())

    @staticmethod
    def _png_bytes() -> bytes:
        buffer = BytesIO()
        Image.new("RGB", (18, 12), color=(80, 100, 120)).save(buffer, format="PNG")
        buffer.seek(0)
        return buffer.read()

    @staticmethod
    def _inference_result() -> dict:
        original = np.full((12, 18, 3), 80, dtype=np.uint8)
        mask = np.zeros((12, 18), dtype=bool)
        mask[3:8, 5:12] = True
        overlay = original.copy()
        overlay[mask] = (150, 20, 20)
        return {
            "predicted_class": "glioma",
            "class_probabilities": {
                "glioma": 0.7,
                "meningioma": 0.1,
                "notumor": 0.1,
                "pituitary": 0.1,
            },
            "threshold": 0.5,
            "image_info": {
                "device": "cpu",
                "original_size": [18, 12],
                "segmentation_processed_size": [256, 256],
            },
            "original_image": original,
            "aligned_tumor_mask": mask,
            "overlay": overlay,
        }

    def test_home_and_health_routes(self) -> None:
        self.connection.request("GET", "/")
        home = self.connection.getresponse()
        self.assertEqual(home.status, 200)
        self.assertIn(b"MRI", home.read())

        self.connection.request("GET", "/api/health")
        health = self.connection.getresponse()
        self.assertEqual(json.loads(health.read()), {"status": "ready"})

    def test_valid_image_calls_real_inference_contract_and_returns_downloadable_images(self) -> None:
        with patch(
            "scripts.serve_app.predict_mri",
            return_value=self._inference_result(),
        ) as predict:
            status, payload = self._post("scan.png", self._png_bytes())

        self.assertEqual(status, 200)
        predict.assert_called_once()
        self.assertEqual(payload["predicted_class"], "glioma")
        self.assertEqual(payload["class_probabilities"]["glioma"], 0.7)
        self.assertEqual(payload["threshold"], 0.5)
        self.assertEqual(set(payload["images"]), {"original", "mask", "overlay"})
        for data_url in payload["images"].values():
            self.assertTrue(data_url.startswith("data:image/png;base64,"))
            encoded_image = data_url.split(",", maxsplit=1)[1]
            with Image.open(BytesIO(base64.b64decode(encoded_image))) as image:
                self.assertEqual(image.size, (18, 12))
        self.assertNotIn(str(self.root), json.dumps(payload))

    def test_invalid_image_bytes_and_unsupported_extension_are_rejected(self) -> None:
        status, payload = self._post("bad.png", b"not-image")
        self.assertEqual(status, 400)
        self.assertIn("valid", payload["error"])

        status, payload = self._post("scan.dcm", b"anything")
        self.assertEqual(status, 400)
        self.assertIn("PNG", payload["error"])

    def test_empty_uploads_are_rejected(self) -> None:
        status, payload = self._post("empty.png", b"")
        self.assertEqual(status, 400)
        self.assertIn("empty", payload["error"])

        self.connection.request(
            "POST",
            "/api/analyze",
            b"",
            {"Content-Type": "multipart/form-data; boundary=empty", "Content-Length": "0"},
        )
        response = self.connection.getresponse()
        self.assertEqual(response.status, 400)
        self.assertIn("Choose an MRI image", json.loads(response.read())["error"])

    def test_oversized_upload_is_rejected_before_inference(self) -> None:
        status, payload = self._post("large.png", b"x" * (MAX_UPLOAD_BYTES + 1))
        self.assertEqual(status, 413)
        self.assertIn("10 MB", payload["error"])

    def test_inference_errors_are_sanitized(self) -> None:
        from mri_tumor.inference import InferenceError

        with patch(
            "scripts.serve_app.predict_mri",
            side_effect=InferenceError("internal location should not appear"),
        ):
            status, payload = self._post("scan.png", self._png_bytes())
        self.assertEqual(status, 422)
        self.assertNotIn("internal", payload["error"])
        self.assertNotIn(str(self.root), json.dumps(payload))

    def test_invalid_probability_mapping_and_unexpected_inference_failure_are_safe(self) -> None:
        invalid = self._inference_result()
        invalid["class_probabilities"] = {
            "glioma": 0.1,
            "meningioma": 0.1,
            "notumor": 0.7,
            "pituitary": 0.1,
        }
        with patch("scripts.serve_app.predict_mri", return_value=invalid):
            status, payload = self._post("scan.png", self._png_bytes())
        self.assertEqual(status, 500)
        self.assertEqual(payload["error"], "Analysis failed unexpectedly. Please try again.")
        self.assertNotIn(str(self.root), json.dumps(payload))

        with patch(
            "scripts.serve_app.predict_mri",
            side_effect=AssertionError(f"private path: {self.root}"),
        ):
            status, payload = self._post("scan.png", self._png_bytes())
        self.assertEqual(status, 500)
        self.assertEqual(payload["error"], "Analysis failed unexpectedly. Please try again.")
        self.assertNotIn(str(self.root), json.dumps(payload))

    def test_missing_checkpoint_is_reported_instead_of_synthesized(self) -> None:
        missing = self.root / "missing.pt"
        with self.assertRaisesRegex(FileNotFoundError, str(missing)):
            _existing_checkpoint(missing, "classification")

    def test_missing_checkpoint_returns_safe_api_error(self) -> None:
        self.classifier.unlink()
        status, payload = self._post("scan.png", self._png_bytes())
        self.assertEqual(status, 422)
        self.assertIn("configured model files", payload["error"])
        self.assertNotIn(str(self.root), json.dumps(payload))


if __name__ == "__main__":
    unittest.main()

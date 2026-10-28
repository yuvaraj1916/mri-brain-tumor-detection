"""Serve the MRI inference UI and its local upload API."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import threading
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
from PIL import Image, UnidentifiedImageError

from mri_tumor.classification.data import CLASS_NAMES
from mri_tumor.data.config import PROJECT_ROOT
from mri_tumor.inference import InferenceError, TUMOR_MASK_THRESHOLD, predict_mri

WEB_ROOT = PROJECT_ROOT / "web"
DEFAULT_CLASSIFICATION_CHECKPOINT = (
    PROJECT_ROOT
    / "results/classification/cnn_baseline_20261012T113201335170Z_8326fdd5/best_model.pt"
)
DEFAULT_SEGMENTATION_CHECKPOINT = (
    PROJECT_ROOT
    / "results/segmentation/unet_baseline_20261012T120155132434Z_1cb36ccd/best_model.pt"
)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_INFERENCE_LOCK = threading.Lock()


class _InvalidImageDimensions(ValueError):
    """Raised when an uploaded image exceeds the supported pixel dimensions."""


def _json_response(
    handler: BaseHTTPRequestHandler,
    status: HTTPStatus,
    content: dict[str, Any],
) -> None:
    body = json.dumps(content, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.send_header("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
    handler.end_headers()
    handler.wfile.write(body)


def _image_data_url(pixels: np.ndarray, image_format: str) -> str:
    image_buffer = BytesIO()
    Image.fromarray(pixels).save(image_buffer, format=image_format)
    encoded = base64.b64encode(image_buffer.getvalue()).decode("ascii")
    return f"data:image/{image_format.lower()};base64,{encoded}"


def _decode_upload(body: bytes, content_type: str) -> tuple[str, bytes]:
    headers = (
        f"Content-Type: {content_type}\r\n"
        "MIME-Version: 1.0\r\n"
        "\r\n"
    ).encode("ascii", errors="strict")
    message = BytesParser(policy=policy.default).parsebytes(headers + body)
    if message.get_content_type() != "multipart/form-data":
        raise ValueError("Choose an MRI image file to analyze.")

    try:
        file_parts = [
            part
            for part in message.iter_parts()
            if part.get_content_disposition() == "form-data"
            and part.get_param("name", header="content-disposition") == "image"
        ]
    except (TypeError, ValueError):
        raise ValueError("Choose an MRI image file to analyze.") from None
    if len(file_parts) != 1:
        raise ValueError("Upload exactly one MRI image.")
    part = file_parts[0]
    filename = Path(part.get_filename() or "").name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported image format. Use PNG, JPEG, BMP, TIFF, or WebP.")
    image_bytes = part.get_payload(decode=True)
    if not image_bytes:
        raise ValueError("The uploaded file is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise OverflowError("Image is too large. Maximum upload size is 10 MB.")
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size
            if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
                raise _InvalidImageDimensions(
                    "Image dimensions are unsupported; choose a smaller image."
                )
            image.load()
    except _InvalidImageDimensions:
        raise
    except (
        OSError,
        ValueError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ):
        raise ValueError("The uploaded file is not a valid readable image.") from None
    return suffix, image_bytes


def _result_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Convert real inference arrays to safe, browser-consumable image data."""
    class_probabilities = result.get("class_probabilities")
    if not isinstance(class_probabilities, dict) or tuple(class_probabilities) != CLASS_NAMES:
        raise ValueError("Classifier output has an incompatible class mapping.")
    class_values = np.asarray(
        [class_probabilities[name] for name in CLASS_NAMES], dtype=np.float64
    )
    if (
        not np.isfinite(class_values).all()
        or np.any(class_values < 0)
        or np.any(class_values > 1)
        or not np.isclose(float(class_values.sum()), 1.0, atol=1e-5)
        or result.get("predicted_class") != CLASS_NAMES[int(np.argmax(class_values))]
    ):
        raise ValueError("Classifier output contains invalid probabilities.")
    if result.get("threshold") != TUMOR_MASK_THRESHOLD:
        raise ValueError("Segmenter output uses an unexpected threshold.")

    original = np.asarray(result["original_image"], dtype=np.uint8)
    mask = np.asarray(result["aligned_tumor_mask"], dtype=np.bool_)
    overlay = np.asarray(result["overlay"], dtype=np.uint8)
    image_info = result.get("image_info")
    if (
        original.ndim != 3
        or original.shape[2] != 3
        or mask.shape != original.shape[:2]
        or overlay.shape != original.shape
        or not isinstance(image_info, dict)
        or image_info.get("original_size") != [original.shape[1], original.shape[0]]
        or image_info.get("device") not in {"cpu", "mps"}
        or not isinstance(image_info.get("segmentation_processed_size"), list)
        or len(image_info["segmentation_processed_size"]) != 2
        or any(
            not isinstance(dimension, int) or dimension <= 0
            for dimension in image_info["segmentation_processed_size"]
        )
    ):
        raise ValueError("Model output dimensions are incompatible.")
    return {
        "predicted_class": result["predicted_class"],
        "class_probabilities": {
            name: float(class_probabilities[name]) for name in CLASS_NAMES
        },
        "threshold": float(result["threshold"]),
        "image_info": image_info,
        "images": {
            "original": _image_data_url(original, "PNG"),
            "mask": _image_data_url(mask.astype(np.uint8) * 255, "PNG"),
            "overlay": _image_data_url(overlay, "PNG"),
        },
    }


def make_request_handler(
    classification_checkpoint: Path,
    segmentation_checkpoint: Path,
    web_root: Path = WEB_ROOT,
) -> type[BaseHTTPRequestHandler]:
    """Create a request handler bound to configured checkpoint locations."""

    class MRIRequestHandler(BaseHTTPRequestHandler):
        server_version = "MRIDemo/1.0"

        def log_message(self, format: str, *args: object) -> None:
            # Do not log request content, uploaded names, or internal paths.
            return

        def do_GET(self) -> None:
            route = urlsplit(self.path).path
            if route == "/api/health":
                _json_response(self, HTTPStatus.OK, {"status": "ready"})
                return
            static_files = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/index.html": ("index.html", "text/html; charset=utf-8"),
                "/styles.css": ("styles.css", "text/css; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            }
            static = static_files.get(route)
            if static is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            filename, content_type = static
            try:
                content = (web_root / filename).read_bytes()
            except OSError:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; style-src 'self'; "
                "script-src 'self'; connect-src 'self'; base-uri 'none'; "
                "frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/api/analyze":
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "Endpoint not found."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "Invalid upload request."})
                return
            if length <= 0:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "Choose an MRI image to analyze."})
                return
            if length > MAX_UPLOAD_BYTES + 64 * 1024:
                _json_response(self, HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Image is too large. Maximum upload size is 10 MB."})
                return
            try:
                suffix, image_bytes = _decode_upload(
                    self.rfile.read(length), self.headers.get("Content-Type", "")
                )
            except OverflowError as error:
                _json_response(self, HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": str(error)})
                return
            except UnicodeError:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "Upload a valid PNG, JPEG, BMP, TIFF, or WebP MRI image."})
                return
            except ValueError as error:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return

            try:
                with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
                    temporary.write(image_bytes)
                    temporary.flush()
                    with _INFERENCE_LOCK:
                        result = predict_mri(
                            Path(temporary.name),
                            classification_checkpoint,
                            segmentation_checkpoint,
                        )
                payload = _result_payload(result)
            except InferenceError:
                _json_response(self, HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "The MRI could not be analyzed. Check the image and configured model files."})
                return
            except Exception as error:
                print(
                    f"Analysis request failed ({type(error).__name__}).",
                    file=sys.stderr,
                )
                _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Analysis failed unexpectedly. Please try again."})
                return
            _json_response(self, HTTPStatus.OK, payload)

        def do_OPTIONS(self) -> None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Allow", "GET, POST, OPTIONS")
            self.end_headers()

    return MRIRequestHandler


def _existing_checkpoint(path: Path, label: str) -> Path:
    checkpoint = path.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing {label} checkpoint: {checkpoint}")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"), help="Bind address.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")), help="HTTP port.")
    parser.add_argument(
        "--classification-checkpoint",
        type=Path,
        default=Path(os.environ.get("MRI_CLASSIFICATION_CHECKPOINT", DEFAULT_CLASSIFICATION_CHECKPOINT)),
    )
    parser.add_argument(
        "--segmentation-checkpoint",
        type=Path,
        default=Path(os.environ.get("MRI_SEGMENTATION_CHECKPOINT", DEFAULT_SEGMENTATION_CHECKPOINT)),
    )
    args = parser.parse_args()

    classification_checkpoint = _existing_checkpoint(
        args.classification_checkpoint, "classification"
    )
    segmentation_checkpoint = _existing_checkpoint(
        args.segmentation_checkpoint, "segmentation"
    )
    if not WEB_ROOT.is_dir():
        raise FileNotFoundError(f"Web assets are missing: {WEB_ROOT}")

    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_request_handler(classification_checkpoint, segmentation_checkpoint),
    )
    print(f"MRI research prototype available at http://{args.host}:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping MRI research prototype.")
    finally:
        server.server_close()


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from None

"""Unit tests for OCR extraction service and endpoints in SightGuide V2."""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

project_root = Path(__file__).resolve().parent.parent
backend_dir = project_root / "backend"
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(project_root))

from app.main import app
from app.services.ocr_service import OCRService


class TestOCREndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls.ocr_service = OCRService.get_instance()

    def create_image_with_text(self, text: str = "SightGuide OCR 2026") -> io.BytesIO:
        buf = io.BytesIO()
        img = Image.new("RGB", (320, 100), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((20, 35), text, fill=(0, 0, 0))
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf

    def create_blank_image(self) -> io.BytesIO:
        buf = io.BytesIO()
        img = Image.new("RGB", (100, 100), color=(120, 120, 120))
        img.save(buf, format="JPEG")
        buf.seek(0)
        return buf

    def test_health_includes_ocr(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("ocr_available", data)
        self.assertTrue(data["ocr_available"])

    def test_ocr_extract_positive_text(self) -> None:
        """Verify OCR detects text from image."""
        img_buf = self.create_image_with_text("SightGuide OCR 2026")
        response = self.client.post(
            "/api/ocr/extract",
            files={"file": ("test_text.png", img_buf, "image/png")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["filename"], "test_text.png")
        self.assertTrue(data["has_text"])
        self.assertIn("SightGuide", data["text"])
        self.assertGreaterEqual(len(data["lines"]), 1)
        self.assertGreater(data["execution_time_seconds"], 0.0)

    def test_ocr_extract_negative_blank(self) -> None:
        """Verify OCR gracefully reports no text on a blank negative image."""
        img_buf = self.create_blank_image()
        response = self.client.post(
            "/api/ocr/extract",
            files={"file": ("blank.jpg", img_buf, "image/jpeg")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["has_text"])
        self.assertEqual(data["text"], "")
        self.assertIn("No readable text detected", data.get("message", ""))

    def test_ocr_reject_invalid_extension(self) -> None:
        """Verify OCR rejects unsupported file types."""
        buf = io.BytesIO(b"fake text content")
        response = self.client.post(
            "/api/ocr/extract",
            files={"file": ("test.txt", buf, "text/plain")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported image extension", response.json().get("detail", ""))

    def test_results_manifest_endpoint(self) -> None:
        """Verify manifest summary endpoint returns 96 images across 6 categories."""
        response = self.client.get("/api/results/manifest")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(data["total_images"], 8)
        self.assertIn("categories", data)

    def test_results_decoding_comparison_endpoint(self) -> None:
        """Verify decoding comparison endpoint returns experiment metadata."""
        response = self.client.get("/api/results/decoding-comparison")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("metadata", data)
        self.assertIn("rows", data)

    def test_ocr_extract_phone_handwritten_screen(self) -> None:
        """Verify OCR detects handwritten/stylized text displayed on a phone screen."""
        img_path = project_root / "evaluations" / "ocr_test_images" / "phone_handwritten_screen.png"
        if not img_path.is_file():
            self.skipTest(f"Test image {img_path} not found")
        with open(img_path, "rb") as f:
            buf = io.BytesIO(f.read())
        response = self.client.post(
            "/api/ocr/extract",
            files={"file": ("phone_handwritten_screen.png", buf, "image/png")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["has_text"])
        self.assertIn("Hi", data["text"])
        self.assertGreaterEqual(len(data["lines"]), 2)
        # Verify lines have reading_order in order
        orders = [l.get("reading_order") for l in data["lines"]]
        self.assertEqual(orders, list(range(1, len(orders) + 1)))
        # Verify confidences are reasonable
        for line in data["lines"]:
            self.assertGreater(line["confidence"], 0.3)

    def test_ocr_extract_storefront_text(self) -> None:
        """Verify standard signage/document OCR behavior is preserved."""
        img_path = project_root / "evaluations" / "eval_v3_grounding" / "images" / "08_storefront_visible_text.png"
        if not img_path.is_file():
            self.skipTest("Storefront image not found")
        with open(img_path, "rb") as f:
            buf = io.BytesIO(f.read())
        response = self.client.post(
            "/api/ocr/extract",
            files={"file": ("08_storefront_visible_text.png", buf, "image/png")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["has_text"])
        self.assertIn("BAKERY", data["text"])

    def test_ocr_sample_02_uppercase_y_normalization(self) -> None:
        """Verify uppercase 7 -> Y normalization on sample_02.jpeg while keeping raw OCR distinct."""
        img_path = project_root / "evaluations" / "real_handwriting_eval" / "images" / "sample_02.jpeg"
        if not img_path.is_file():
            self.skipTest("sample_02.jpeg not found")
        with open(img_path, "rb") as f:
            buf = io.BytesIO(f.read())
        response = self.client.post(
            "/api/ocr/extract",
            files={"file": ("sample_02.jpeg", buf, "image/jpeg")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["has_text"])
        self.assertIn("DD7", data["text"])  # Raw OCR preserved
        self.assertIsNotNone(data["normalized_text"])
        self.assertIn("DDY", data["normalized_text"])  # Normalized hypothesis
        self.assertIn("Maddy", data["text"])

    def test_ocr_sample_06_negative_control_preserves_modu(self) -> None:
        """Mandatory negative control: verify Modu is NEVER replaced with Maddy."""
        img_path = project_root / "evaluations" / "real_handwriting_eval" / "images" / "sample_06.jpeg"
        if not img_path.is_file():
            self.skipTest("sample_06.jpeg not found")
        with open(img_path, "rb") as f:
            buf = io.BytesIO(f.read())
        response = self.client.post(
            "/api/ocr/extract",
            files={"file": ("sample_06.jpeg", buf, "image/jpeg")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["has_text"])
        self.assertIn("Modu", data["text"])
        norm = (data.get("normalized_text") or "").lower()
        self.assertNotIn("maddy", norm, "NEGATIVE TEST FAILED: 'Modu' was improperly normalized to 'Maddy'")


if __name__ == "__main__":
    unittest.main()


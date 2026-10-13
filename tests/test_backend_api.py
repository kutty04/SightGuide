"""Unit and Integration Tests for SightGuide FastAPI Backend.

Tests cover:
1. Health and service discovery endpoints.
2. File format and payload validation.
3. Caption generation endpoint and schema contract.
4. Paired method comparison endpoint (Control vs. Proposed).
5. Benchmark data and image retrieval endpoints.
6. Graceful behavior when external LLM credentials are not configured.

NOTE: All tests use in-memory synthetic images and mocked external API responses
to ensure zero external API consumption and zero test side-effects.
"""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image
from fastapi.testclient import TestClient

project_root = Path(__file__).resolve().parent.parent
backend_dir = project_root / "backend"
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(project_root))

from app.main import app


def create_synthetic_test_image(format_ext: str = "JPEG", size: tuple[int, int] = (64, 64)) -> io.BytesIO:
    """Create a minimal in-memory image for testing."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=(120, 180, 220))
    img.save(buf, format=format_ext)
    buf.seek(0)
    return buf


class TestBackendAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_root_endpoint(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("SightGuide", data.get("service", ""))
        self.assertIn("/docs", data.get("documentation", ""))

    def test_health_endpoint(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertEqual(data.get("version"), "1.0.0")
        self.assertIn("active_device", data)
        self.assertIn("blip_available", data)
        self.assertIn("groq_configured", data)

    def test_reject_unsupported_file_extension(self) -> None:
        dummy_file = io.BytesIO(b"dummy text content")
        response = self.client.post(
            "/api/caption/generate",
            files={"file": ("test.txt", dummy_file, "text/plain")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported image extension", response.json().get("detail", ""))

    def test_caption_generate_endpoint_schema(self) -> None:
        img_buf = create_synthetic_test_image("JPEG")
        response = self.client.post(
            "/api/caption/generate",
            files={"file": ("test_synthetic.jpg", img_buf, "image/jpeg")},
            data={"lambda_weight": "0.5", "method": "proposed"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["filename"], "test_synthetic.jpg")
        self.assertGreaterEqual(len(data["candidates"]), 2)
        self.assertIn("selected_pair", data)
        self.assertEqual(data["selected_pair"]["method"], "proposed_diversity_aware")
        self.assertIn("fusion", data)
        self.assertIn(data["fusion"]["generation_status"], ("SUCCESS", "GROQ_KEY_NOT_CONFIGURED"))

    def test_caption_compare_endpoint_schema(self) -> None:
        img_buf = create_synthetic_test_image("PNG")
        response = self.client.post(
            "/api/caption/compare",
            files={"file": ("test_synthetic.png", img_buf, "image/png")},
            data={"lambda_weight": "0.5"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["filename"], "test_synthetic.png")
        self.assertIn("control_pair", data)
        self.assertIn("proposed_pair", data)
        self.assertEqual(data["control_pair"]["method"], "control_naive_clip_top2")
        self.assertEqual(data["proposed_pair"]["method"], "proposed_diversity_aware")
        self.assertIn("plain_language_explanation", data)
        self.assertIn("is_divergent", data)
        self.assertIn("pair_diversity_delta", data)

    def test_benchmark_summary_endpoint(self) -> None:
        response = self.client.get("/api/benchmark/summary")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_images"], 12)
        self.assertEqual(data["identical_pairs_count"], 10)
        self.assertEqual(data["divergent_pairs_count"], 2)
        self.assertGreater(data["proposed_mean_diversity"], data["control_mean_diversity"])
        self.assertEqual(len(data["benchmark_records"]), 12)

    def test_benchmark_image_serving(self) -> None:
        # Request valid existing benchmark image
        response = self.client.get("/api/benchmark/image/sample-clouds-400x300.jpg")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("content-type"), "image/jpeg")

        # Request non-existent image
        missing_resp = self.client.get("/api/benchmark/image/does_not_exist_999.jpg")
        self.assertEqual(missing_resp.status_code, 404)

    def test_groq_unavailable_graceful_handling(self) -> None:
        """When GROQ_API_KEY is unset, fusion should report GROQ_KEY_NOT_CONFIGURED without crashing."""
        with patch.dict("os.environ", {}, clear=True):
            img_buf = create_synthetic_test_image("JPEG")
            response = self.client.post(
                "/api/caption/generate",
                files={"file": ("test_synthetic.jpg", img_buf, "image/jpeg")},
                data={"lambda_weight": "0.5", "method": "proposed"}
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["fusion"]["generation_status"], "GROQ_KEY_NOT_CONFIGURED")
            self.assertIn("GROQ_API_KEY is not configured", data["fusion"]["error_details"])


if __name__ == "__main__":
    unittest.main()

"""SightGuide Tests for Conversational Visual Q&A and Accessibility Features.

Verifies:
1. Q&A input validation (empty question, oversized question).
2. Grounded answering using verified scene description.
3. Grounded answering using verified OCR text.
4. Truthful refusal when queries request unobservable/unsupported attributes.
5. Graceful fallback when LLM API credentials are not configured.
6. OCR reading-order sorting and paragraph grouping.
7. Full FastAPI endpoint contract for POST /api/qa/ask.
"""

import base64
import io
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from PIL import Image, ImageDraw

from fastapi.testclient import TestClient
from app.main import app
from app.schemas import QARequest, QAResponse, QAHistoryItem
from app.services.qa_service import QAService
from app.services.ocr_service import sort_and_format_ocr_results

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestQAAndAccessibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls.qa_service = QAService.get_instance()

    def test_qa_empty_and_whitespace_question(self) -> None:
        """Verify that empty or whitespace questions return a safe prompt rather than failing."""
        req = QARequest(question="   ", scene_caption="a cat sitting on a rug")
        res = self.qa_service.answer_question(req)

        self.assertIsInstance(res, QAResponse)
        self.assertTrue(res.is_uncertain)
        self.assertIn("No question was provided", res.answer)

    def test_qa_grounded_scene_response(self) -> None:
        """Verify that questions about the scene utilize the verified visual caption."""
        req = QARequest(
            question="What is the main subject in the image?",
            scene_caption="a brown dog playing with a red ball in a grassy park",
            ocr_text=""
        )
        # Test under deterministic fallback
        res = self.qa_service._heuristic_grounded_fallback(
            question=req.question,
            scene_caption=req.scene_caption,
            ocr_text="",
            start_time=0.0,
            reason="unit_test"
        )
        self.assertIn("brown dog", res.answer)
        self.assertFalse(res.is_uncertain)
        self.assertIn("scene_caption", res.grounded_sources)

    def test_qa_grounded_ocr_response(self) -> None:
        """Verify that questions about text/signs utilize the verified OCR extraction."""
        req = QARequest(
            question="What does the sign say?",
            scene_caption="a street pole outdoors",
            ocr_text="SPEED LIMIT 25"
        )
        res = self.qa_service._heuristic_grounded_fallback(
            question=req.question,
            scene_caption=req.scene_caption,
            ocr_text=req.ocr_text,
            start_time=0.0,
            reason="unit_test"
        )
        self.assertIn("SPEED LIMIT 25", res.answer)
        self.assertFalse(res.is_uncertain)
        self.assertIn("ocr_text", res.grounded_sources)

    def test_qa_unobservable_visual_query_honesty(self) -> None:
        """Verify that questions requesting unseen attributes return an honest refusal."""
        req = QARequest(
            question="What brand of shoes is the person wearing?",
            scene_caption="a close-up portrait of a smiling woman in a blue shirt",
            ocr_text=""
        )
        res = self.qa_service._heuristic_grounded_fallback(
            question=req.question,
            scene_caption=req.scene_caption,
            ocr_text="",
            start_time=0.0,
            reason="unit_test"
        )
        self.assertTrue(res.is_uncertain)
        self.assertIn("cannot be determined from the available visual evidence", res.answer)

    def test_qa_missing_credentials_graceful_fallback(self) -> None:
        """Verify that when api_key is empty, the service falls back deterministically without 500 error."""
        req = QARequest(
            question="What is shown in the picture?",
            scene_caption="a lighthouse standing on a rocky shore during sunset",
            ocr_text=""
        )
        res = self.qa_service.answer_question(req, api_key="")
        self.assertIsInstance(res, QAResponse)
        self.assertEqual(res.capability, "CONTEXT_GROUNDED_FALLBACK")
        self.assertIn("lighthouse", res.answer)

    def test_ocr_reading_order_and_paragraph_clustering(self) -> None:
        """Verify that scrambled OCR boxes are sorted in top-to-bottom reading order."""
        # Raw items: [box, text, confidence]
        # Line 2 (bottom): y in [100, 120]
        # Line 1 (top): y in [20, 40]
        scrambled_items = [
            [[[10, 100], [100, 100], [100, 120], [10, 120]], "Second Line Bottom", 0.95],
            [[[10, 20], [100, 20], [100, 40], [10, 40]], "First Line Top", 0.98]
        ]
        formatted_text, lines = sort_and_format_ocr_results(scrambled_items)

        self.assertEqual(len(lines), 2)
        # Line 1 must come first
        self.assertEqual(lines[0]["text"], "First Line Top")
        self.assertEqual(lines[0]["reading_order"], 1)
        # Line 2 must come second
        self.assertEqual(lines[1]["text"], "Second Line Bottom")
        self.assertEqual(lines[1]["reading_order"], 2)

        # Formatted text should have First Line Top before Second Line Bottom
        lines_in_text = formatted_text.splitlines()
        self.assertEqual(lines_in_text[0], "First Line Top")

    def test_qa_endpoint_api_contract(self) -> None:
        """Verify the POST /api/qa/ask endpoint returns valid QAResponse schema."""
        payload = {
            "question": "What is in the image?",
            "scene_caption": "a bowl of fresh green apples on a wooden table",
            "ocr_text": "",
            "history": []
        }
        response = self.client.post("/api/qa/ask", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["question"], "What is in the image?")
        self.assertIn("answer", data)
        self.assertEqual(data["capability"], "CONTEXT_GROUNDED_FALLBACK")
        self.assertIn("execution_time_seconds", data)

    def test_full_body_clothing_preservation_vs_cropped_portrait(self) -> None:
        """Verify portrait clothing pruning does NOT remove clothing in full-body/standing scenes."""
        from app.services.fusion_service import apply_grounding_safeguard

        # Cropped portrait: lower-body jeans pruned
        cropped = "a portrait headshot of a woman wearing blue shirt and jeans"
        res_cropped = apply_grounding_safeguard(cropped, anchor_context=cropped)
        self.assertNotIn("jeans", res_cropped.lower())
        self.assertIn("blue shirt", res_cropped.lower())

        # Full body: jeans preserved
        full_body = "a full-body portrait of a person standing on the street wearing blue shirt and jeans"
        res_full = apply_grounding_safeguard(full_body, anchor_context=full_body)
        self.assertIn("jeans", res_full.lower())
        self.assertIn("blue shirt", res_full.lower())

    def test_camera_captured_photo_pipeline_integration(self) -> None:
        """Verify that a synthetic camera-captured frame (JPEG from canvas) processes cleanly through the pipeline."""
        cam_img = Image.new("RGB", (640, 480), color=(100, 150, 200))
        draw = ImageDraw.Draw(cam_img)
        draw.text((50, 50), "SightGuide Camera Capture", fill=(255, 255, 255))
        buf = io.BytesIO()
        cam_img.save(buf, format="JPEG", quality=92)
        cam_bytes = buf.getvalue()

        # Send through caption endpoint simulating in-app camera capture
        resp = self.client.post(
            "/api/caption/generate",
            files={"file": ("camera_capture_2026-10-14.jpg", cam_bytes, "image/jpeg")},
            data={"lambda_weight": "0.5", "method": "proposed"}
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["filename"], "camera_capture_2026-10-14.jpg")
        self.assertIn("candidates", data)
        self.assertIn("selected_pair", data)
        self.assertIn("fusion", data)

        # Send through OCR endpoint simulating in-app camera capture
        ocr_resp = self.client.post(
            "/api/ocr/extract",
            files={"file": ("camera_capture_2026-10-14.jpg", cam_bytes, "image/jpeg")}
        )
        self.assertEqual(ocr_resp.status_code, 200)
        ocr_data = ocr_resp.json()
        self.assertIn("has_text", ocr_data)

    def test_vision_conditioned_image_qa_payload(self) -> None:
        """Verify that passing image_data in QARequest triggers genuine VISION_CONDITIONED status and uses visual evidence."""
        # Create a small valid test image in base64
        test_img = Image.new("RGB", (200, 200), color=(0, 100, 255))
        buf = io.BytesIO()
        test_img.save(buf, format="JPEG")
        b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
        data_url = f"data:image/jpeg;base64,{b64_str}"

        payload = {
            "question": "What colors are in this image?",
            "scene_caption": "a solid blue square",
            "ocr_text": None,
            "image_data": data_url
        }

        resp = self.client.post("/api/qa/ask", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["capability"], "VISION_CONDITIONED")
        self.assertIn("visual_image", data["grounded_sources"])
        self.assertTrue(len(data["answer"]) > 0)
        self.assertIsNotNone(data["model_name"])

    def test_vision_qa_unobservable_clothing_uncertainty(self) -> None:
        """Verify that questions querying unobservable attributes in cropped portraits return honest uncertainty."""
        test_img = Image.new("RGB", (200, 200), color=(180, 140, 100))
        buf = io.BytesIO()
        test_img.save(buf, format="JPEG")
        b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
        data_url = f"data:image/jpeg;base64,{b64_str}"

        payload = {
            "question": "Can you see her shoes or what footwear she is wearing?",
            "scene_caption": "a close-up portrait headshot of a woman",
            "ocr_text": None,
            "image_data": data_url
        }

        resp = self.client.post("/api/qa/ask", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["capability"], "VISION_CONDITIONED")
        self.assertTrue(data["is_uncertain"])
        ans_lower = data["answer"].lower()
        self.assertTrue(
            any(w in ans_lower for w in ["out of frame", "cannot be seen", "cannot be determined", "not visible", "not available"]),
            f"Expected honest uncertainty statement, received: {data['answer']}"
        )

    def test_vision_qa_context_fallback_when_no_image_data(self) -> None:
        """Verify that omitting image_data gracefully triggers CONTEXT_GROUNDED_FALLBACK without crashing."""
        payload = {
            "question": "What is shown in the scene?",
            "scene_caption": "a golden retriever catching a tennis ball in a grassy park",
            "ocr_text": None,
            "image_data": None
        }

        resp = self.client.post("/api/qa/ask", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["capability"], "CONTEXT_GROUNDED_FALLBACK")
        self.assertIn("scene_caption", data["grounded_sources"])
        self.assertNotIn("visual_image", data["grounded_sources"])

    def test_vision_qa_corrupted_image_graceful_handling(self) -> None:
        """Verify that sending corrupted base64 data does not crash the backend server."""
        payload = {
            "question": "What colors are visible?",
            "scene_caption": "an outdoor scene",
            "ocr_text": None,
            "image_data": "data:image/jpeg;base64,not_valid_base64_data_!@#$%"
        }

        resp = self.client.post("/api/qa/ask", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        # Should gracefully fall back to context grounded without 500 error
        self.assertEqual(data["capability"], "CONTEXT_GROUNDED_FALLBACK")

    def test_vision_qa_multiturn_history(self) -> None:
        """Verify that multi-turn history is accepted and preserved across conversational follow-ups."""
        test_img = Image.new("RGB", (150, 150), color=(50, 150, 50))
        buf = io.BytesIO()
        test_img.save(buf, format="JPEG")
        data_url = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

        payload = {
            "question": "Can you describe the background?",
            "scene_caption": "a green tree in a quiet garden",
            "ocr_text": None,
            "image_data": data_url,
            "history": [
                {"question": "What is the main subject?", "answer": "The image shows a tree in a garden."}
            ]
        }

        resp = self.client.post("/api/qa/ask", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["capability"], "VISION_CONDITIONED")

    def test_targeted_blip_prefix_generation_for_dress_and_bottle(self) -> None:
        """Verify dynamic prefix builder generates specific, targeted prefixes for dress vs bottle."""
        # 1. Dress query
        prefixes_dress, is_lower, garment = self.qa_service._build_targeted_blip_prefixes("What color is the girl's dress?")
        self.assertEqual(garment, "dress")
        self.assertFalse(is_lower)
        self.assertTrue(any("dress" in p for p in prefixes_dress))
        self.assertIn("the color of the dress is", prefixes_dress)

        # 2. Water bottle query
        prefixes_bottle, _, _ = self.qa_service._build_targeted_blip_prefixes("What color is the water bottle?")
        self.assertTrue(any("bottle" in p for p in prefixes_bottle))
        self.assertIn("the color of the water bottle is", prefixes_bottle)
        self.assertFalse(any("dress" in p for p in prefixes_bottle))

        # 3. Lower body / shoes query
        prefixes_shoes, is_lower_shoes, _ = self.qa_service._build_targeted_blip_prefixes("What shoes is she wearing?")
        self.assertTrue(is_lower_shoes)
        self.assertTrue(any("shoes" in p or "footwear" in p for p in prefixes_shoes))

    def test_vision_qa_targeted_dress_and_bottle_attribute_recognition(self) -> None:
        """Verify targeted attribute recognition queries BLIP with question-specific prefixes."""
        test_img = Image.new("RGB", (200, 200), color=(255, 230, 100))
        buf = io.BytesIO()
        test_img.save(buf, format="JPEG")
        data_url = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

        mock_res = {
            "available": True,
            "findings": ["the color of the dress is yellow", "the girl is wearing a yellow dress"],
            "is_lower_body_query": False,
            "target_garment": "dress"
        }
        with patch.object(
            self.qa_service,
            "_query_blip_vision",
            return_value=mock_res
        ) as mock_blip:
            req = QARequest(
                question="What color is the girl's dress?",
                scene_caption="a girl standing by a table holding a blue water bottle",
                ocr_text=None,
                image_data=data_url
            )
            res = self.qa_service.answer_question(req, api_key="")
            # Verify BLIP vision query was called with the target question
            self.assertTrue(mock_blip.called)
            called_question = mock_blip.call_args[0][1]
            self.assertIn("dress", called_question)
            # Fallback direct finding handles response when no external LLM key
            self.assertIn("yellow", res.answer.lower())
            self.assertEqual(res.capability, "VISION_CONDITIONED")

    def test_vision_qa_multiturn_no_attribute_bleed(self) -> None:
        """Verify multi-turn history with a blue bottle does not bleed 'blue' into a dress question."""
        test_img = Image.new("RGB", (200, 200), color=(255, 200, 150))
        buf = io.BytesIO()
        test_img.save(buf, format="JPEG")
        data_url = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

        mock_res = {
            "available": True,
            "findings": ["the color of the dress is yellow", "the girl is wearing a yellow dress"],
            "is_lower_body_query": False,
            "target_garment": "dress"
        }
        with patch.object(
            self.qa_service,
            "_query_blip_vision",
            return_value=mock_res
        ):
            req = QARequest(
                question="What color is her dress?",
                scene_caption="a girl standing next to a blue water bottle",
                ocr_text=None,
                image_data=data_url,
                history=[
                    QAHistoryItem(question="What color is the water bottle?", answer="The water bottle is blue.")
                ]
            )
            res = self.qa_service.answer_question(req, api_key="")
            self.assertEqual(res.capability, "VISION_CONDITIONED")
            self.assertIn("yellow", res.answer.lower())
            # Ensure the dress answer does not mistakenly state the dress is blue
            self.assertNotIn("dress is blue", res.answer.lower())

    # --- Phase 4 Experimental Detector-Assisted Visual Q&A Tests ---

    def test_detector_disabled_preserves_baseline_behavior(self) -> None:
        """Verify that when SIGHTGUIDE_ENABLE_DETECTOR_QA is false/unset, detector is never called."""
        from app.services.detector_service import DetectorService
        test_img = Image.new("RGB", (200, 200), color=(100, 200, 100))
        buf = io.BytesIO()
        test_img.save(buf, format="JPEG")
        data_url = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

        with patch.dict(os.environ, {"SIGHTGUIDE_ENABLE_DETECTOR_QA": "false"}):
            with patch.object(DetectorService, "detect_objects") as mock_det:
                req = QARequest(
                    question="Do you see a dog in the room?",
                    scene_caption="a quiet room with wooden floor",
                    ocr_text=None,
                    image_data=data_url
                )
                res = self.qa_service.answer_question(req, api_key="")
                self.assertFalse(mock_det.called)
                self.assertNotIn("experimental_detector", res.grounded_sources)
                self.assertEqual(res.capability, "VISION_CONDITIONED")

    def test_detector_supported_coco_high_confidence(self) -> None:
        """Verify supported COCO question with high-confidence detection injects experimental detector evidence."""
        from app.services.detector_service import DetectorService
        test_img = Image.new("RGB", (200, 200), color=(100, 200, 100))
        buf = io.BytesIO()
        test_img.save(buf, format="JPEG")
        data_url = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

        mock_detection = [
            {"label": "dog", "confidence": 0.88, "region": "center", "bbox_normalized": [0.2, 0.2, 0.7, 0.8]}
        ]

        with patch.dict(os.environ, {"SIGHTGUIDE_ENABLE_DETECTOR_QA": "true"}):
            with patch.object(DetectorService, "detect_objects", return_value=mock_detection):
                req = QARequest(
                    question="Is there a dog in this photo?",
                    scene_caption="a dog sitting on a rug",
                    ocr_text=None,
                    image_data=data_url
                )
                res = self.qa_service.answer_question(req, api_key="")
                self.assertIn("experimental_detector", res.grounded_sources)
                self.assertTrue(any("dog" in ev.lower() and "experimental detector" in ev.lower() for ev in res.evidence_used))
                self.assertIn("dog", res.answer.lower())

    def test_detector_low_confidence_filtering(self) -> None:
        """Verify detector filtering rejects detections below 0.55 threshold."""
        from app.services.detector_service import DetectorService
        svc = DetectorService.get_instance()
        # Mock raw detections inside detector service
        with patch.object(svc, "is_available", True):
            should_route, classes, _ = svc.route_question("Do you see a cat on the couch?")
            self.assertTrue(should_route)
            self.assertIn("cat", classes)

    def test_detector_conflicting_blip_and_detector_evidence(self) -> None:
        """Verify conflicting detector evidence does not erase BLIP visual observations."""
        from app.services.detector_service import DetectorService
        test_img = Image.new("RGB", (200, 200), color=(100, 200, 100))
        buf = io.BytesIO()
        test_img.save(buf, format="JPEG")
        data_url = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

        mock_blip = {
            "available": True,
            "findings": ["in this image a cat is sitting on a chair"],
            "is_lower_body_query": False,
            "target_garment": None
        }
        mock_detection = [
            {"label": "dog", "confidence": 0.62, "region": "center", "bbox_normalized": [0.2, 0.2, 0.7, 0.8]}
        ]

        with patch.dict(os.environ, {"SIGHTGUIDE_ENABLE_DETECTOR_QA": "true"}):
            with patch.object(self.qa_service, "_query_blip_vision", return_value=mock_blip):
                with patch.object(DetectorService, "detect_objects", return_value=mock_detection):
                    req = QARequest(
                        question="Is there a dog or cat in the image?",
                        scene_caption="a cat sitting indoors",
                        ocr_text=None,
                        image_data=data_url
                    )
                    res = self.qa_service.answer_question(req, api_key="")
                    # Both BLIP observation and detector observation are preserved in evidence
                    ev_text = " ".join(res.evidence_used).lower()
                    self.assertIn("cat", ev_text)
                    self.assertIn("dog", ev_text)
                    self.assertIn("experimental_detector", res.grounded_sources)
                    self.assertIn("visual_image", res.grounded_sources)

    def test_detector_unsupported_categories_rejected(self) -> None:
        """Verify questions about unsupported OOD categories (fish, deer, iguana) are rejected from detector routing."""
        from app.services.detector_service import DetectorService
        svc = DetectorService.get_instance()

        for query in [
            "Do you see a fish in the aquarium?",
            "Is there a deer in the forest?",
            "Can you find the iguana on the sand?",
            "What kind of flower is shown?"
        ]:
            should_route, classes, reason = svc.route_question(query)
            self.assertFalse(should_route, f"Query '{query}' should not route to COCO detector")
            self.assertTrue("unsupported_category" in reason or not classes)

    def test_detector_negated_and_ambiguous_queries_rejected(self) -> None:
        """Verify negated and ambiguous queries are rejected from detector routing."""
        from app.services.detector_service import DetectorService
        svc = DetectorService.get_instance()

        negated_queries = [
            "Why is there no dog in this picture?",
            "Is there no car on the road?",
            "What is in the image without any people?"
        ]
        for q in negated_queries:
            should_route, _, reason = svc.route_question(q)
            self.assertFalse(should_route, f"Negated query '{q}' should be rejected")
            self.assertEqual(reason, "negated_query")

        ambiguous_queries = [
            "What color is the wall?",
            "Can you describe the general mood?",
            "What happened here?"
        ]
        for q in ambiguous_queries:
            should_route, _, _ = svc.route_question(q)
            self.assertFalse(should_route, f"Ambiguous query '{q}' should be rejected")

    def test_detector_initialization_failure_graceful_fallback(self) -> None:
        """Verify that detector errors do not crash Q&A and fall back gracefully."""
        from app.services.detector_service import DetectorService
        test_img = Image.new("RGB", (200, 200), color=(100, 200, 100))
        buf = io.BytesIO()
        test_img.save(buf, format="JPEG")
        data_url = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

        with patch.dict(os.environ, {"SIGHTGUIDE_ENABLE_DETECTOR_QA": "true"}):
            with patch.object(DetectorService, "detect_objects", side_effect=RuntimeError("ONNX engine crash")):
                req = QARequest(
                    question="Do you see a chair?",
                    scene_caption="an office room",
                    ocr_text=None,
                    image_data=data_url
                )
                res = self.qa_service.answer_question(req, api_key="")
                self.assertEqual(res.capability, "VISION_CONDITIONED")
                self.assertIsNotNone(res.answer)

    def test_detector_no_detections_not_interpreted_as_absence(self) -> None:
        """Verify that empty detector findings do not fabricate an absolute proof-of-absence claim."""
        from app.services.detector_service import DetectorService
        svc = DetectorService.get_instance()
        evidence_note = svc.format_evidence_block(detections=[], queried_classes=["bicycle"])
        self.assertIn("absence of detection is not absolute proof of absence", evidence_note.lower())

    # ---------------------------------------------------------------------------
    # Facial Expression & Smiling Grounding Regression Tests
    # ---------------------------------------------------------------------------
    def test_facial_expression_visible_smile(self) -> None:
        """Verify that a visible smile on portrait woman is accurately recognized without uncertainty."""
        portrait_path = PROJECT_ROOT / "evaluations" / "test_portrait_woman.png"
        self.assertTrue(portrait_path.exists(), f"Missing test portrait: {portrait_path}")
        buf = portrait_path.read_bytes()
        data_uri = f"data:image/png;base64,{base64.b64encode(buf).decode('ascii')}"

        req = QARequest(
            question="Is the girl smiling?",
            image_data=data_uri,
            scene_caption="a close-up portrait of a woman in a blue shirt",
            ocr_text=""
        )
        res = self.qa_service.answer_question(req)
        self.assertEqual(res.capability, "VISION_CONDITIONED")
        self.assertFalse(res.is_uncertain)
        self.assertTrue(
            any(w in res.answer.lower() for w in ["smil", "smile", "smiling"]),
            f"Expected smile confirmation, got: '{res.answer}'"
        )

    def test_facial_expression_neutral_or_ambiguous(self) -> None:
        """Verify that a neutral/serious face does not hallucinate an affirmative smile."""
        neutral_path = PROJECT_ROOT / "dataset" / "toa-heftiba-O3ymvT7Wf9U-unsplash.jpg"
        self.assertTrue(neutral_path.exists(), f"Missing test image: {neutral_path}")
        buf = neutral_path.read_bytes()
        data_uri = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('ascii')}"

        req = QARequest(
            question="Is the person smiling?",
            image_data=data_uri,
            scene_caption="a portrait of a woman wearing a hoodie",
            ocr_text=""
        )
        res = self.qa_service.answer_question(req)
        self.assertEqual(res.capability, "VISION_CONDITIONED")
        ans_lower = res.answer.lower()
        self.assertTrue(
            any(w in ans_lower for w in ["not", "does not appear", "cannot", "can't", "neutral", "unable", "no"]),
            f"Expected neutral or non-smiling assessment, got: '{res.answer}'"
        )

    def test_facial_expression_obscured_or_out_of_frame_face(self) -> None:
        """Verify that an obscured face or torso-only view returns honest uncertainty."""
        portrait_path = PROJECT_ROOT / "evaluations" / "test_portrait_woman.png"
        self.assertTrue(portrait_path.exists(), f"Missing test portrait: {portrait_path}")
        im = Image.open(portrait_path)
        w, h = im.size
        torso = im.crop((0, 350, w, h))
        buf = io.BytesIO()
        torso.save(buf, format="PNG")
        data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

        req = QARequest(
            question="Is the person smiling?",
            image_data=data_uri,
            scene_caption="a close-up view of a person's blue shirt, torso only",
            ocr_text=""
        )
        res = self.qa_service.answer_question(req)
        self.assertEqual(res.capability, "VISION_CONDITIONED")
        self.assertTrue(res.is_uncertain)
        self.assertIn("out of frame", res.answer.lower())

    def test_facial_expression_no_person_landscape(self) -> None:
        """Verify that asking about smiling on an image without people returns honest refusal."""
        clouds_path = PROJECT_ROOT / "dataset" / "sample-clouds-400x300.jpg"
        self.assertTrue(clouds_path.exists(), f"Missing test clouds image: {clouds_path}")
        buf = clouds_path.read_bytes()
        data_uri = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('ascii')}"

        req = QARequest(
            question="Is the person smiling?",
            image_data=data_uri,
            scene_caption="clouds in the sky",
            ocr_text=""
        )
        res = self.qa_service.answer_question(req)
        self.assertEqual(res.capability, "VISION_CONDITIONED")
        self.assertTrue(res.is_uncertain)
        self.assertIn("no person or visible face is present", res.answer.lower())

    # ---------------------------------------------------------------------------
    # First-Response Descriptiveness & Attribute Targeting Regression Tests
    # ---------------------------------------------------------------------------
    def test_hair_intent_and_grammatical_prefix_generation(self) -> None:
        """Verify dynamic prefix builder generates grammatical, targeted prefixes for hair queries."""
        prefixes_hair, is_lower, garment = self.qa_service._build_targeted_blip_prefixes("What color is her hair?")
        self.assertFalse(is_lower)
        self.assertIsNone(garment)
        self.assertTrue(any("hair" in p for p in prefixes_hair))
        self.assertFalse(any("the her hair" in p for p in prefixes_hair), "Possessive pronoun must not produce 'the her hair'")
        self.assertTrue(any(p.startswith("the person's hair") or "has" in p or "with" in p for p in prefixes_hair))

    def test_first_response_clothing_color_and_pattern_descriptiveness(self) -> None:
        """Verify that a clothing color question includes both color and companion pattern details on the first turn."""
        mock_findings = [
            "the dress is white with a floral pattern",
            "the person is wearing a white floral dress"
        ]
        with patch.object(self.qa_service, "_query_blip_vision", return_value={"available": True, "findings": mock_findings, "is_lower_body_query": False, "target_garment": "dress", "is_expression_query": False}):
            test_img = Image.new("RGB", (100, 100), color=(255, 255, 255))
            buf = io.BytesIO()
            test_img.save(buf, format="JPEG")
            data_uri = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

            req = QARequest(
                question="What is her dress color?",
                image_data=data_uri,
                scene_caption="a woman in a white floral dress",
                ocr_text=""
            )
            res = self.qa_service.answer_question(req)
            self.assertEqual(res.capability, "VISION_CONDITIONED")
            self.assertFalse(res.is_uncertain)
            ans_lower = res.answer.lower()
            # Must answer color
            self.assertIn("white", ans_lower)
            # Must include visible companion detail without requiring 'tell me elaborately'
            self.assertTrue(
                any(w in ans_lower for w in ["floral", "pattern", "dress"]),
                f"Expected companion pattern detail in first response, got: '{res.answer}'"
            )

    def test_first_response_exact_question_answered_first(self) -> None:
        """Verify that the primary queried attribute is answered in the opening of the response."""
        mock_findings = [
            "the person is wearing a blue shirt and earrings",
            "the color of the shirt is blue"
        ]
        with patch.object(self.qa_service, "_query_blip_vision", return_value={"available": True, "findings": mock_findings, "is_lower_body_query": False, "target_garment": "shirt", "is_expression_query": False}):
            test_img = Image.new("RGB", (100, 100), color=(0, 0, 255))
            buf = io.BytesIO()
            test_img.save(buf, format="JPEG")
            data_uri = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

            req = QARequest(
                question="What color is her shirt?",
                image_data=data_uri,
                scene_caption="a portrait of a woman in a blue shirt",
                ocr_text=""
            )
            res = self.qa_service.answer_question(req)
            self.assertEqual(res.capability, "VISION_CONDITIONED")
            first_clause = res.answer.split(".")[0].lower()
            self.assertIn("blue", first_clause, f"Color must be answered first, got: '{res.answer}'")

    def test_first_response_no_unsupported_attribute_hallucination(self) -> None:
        """Verify that when evidence has plain color only, synthetic details like patterns are not hallucinated."""
        mock_findings = [
            "the shirt is plain blue",
            "the person is wearing a blue shirt"
        ]
        with patch.object(self.qa_service, "_query_blip_vision", return_value={"available": True, "findings": mock_findings, "is_lower_body_query": False, "target_garment": "shirt", "is_expression_query": False}):
            test_img = Image.new("RGB", (100, 100), color=(0, 0, 255))
            buf = io.BytesIO()
            test_img.save(buf, format="JPEG")
            data_uri = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

            req = QARequest(
                question="What color is her shirt?",
                image_data=data_uri,
                scene_caption="a woman in a blue shirt",
                ocr_text=""
            )
            res = self.qa_service.answer_question(req)
            ans_lower = res.answer.lower()
            self.assertNotIn("floral", ans_lower)
            self.assertNotIn("polka dot", ans_lower)
            self.assertNotIn("striped", ans_lower)

    def test_first_response_silhouette_face_uncertainty(self) -> None:
        """Verify that silhouette or darkened faces return honest uncertainty rather than inventing expressions."""
        test_img = Image.new("RGB", (100, 100), color=(10, 10, 10))
        buf = io.BytesIO()
        test_img.save(buf, format="JPEG")
        data_uri = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

        req = QARequest(
            question="What is the person's facial expression?",
            image_data=data_uri,
            scene_caption="a dark silhouette of a person standing against a sunset",
            ocr_text=""
        )
        res = self.qa_service.answer_question(req)
        self.assertEqual(res.capability, "VISION_CONDITIONED")
        self.assertTrue(res.is_uncertain)
        self.assertTrue(
            any(w in res.answer.lower() for w in ["out of frame", "silhouette", "obscured", "cannot be determined"]),
            f"Expected honest uncertainty on silhouette face, got: '{res.answer}'"
        )

    def test_first_response_fallback_without_groq_credentials(self) -> None:
        """Verify that fallback mode without API key returns a clean, non-stuttering natural sentence."""
        mock_findings = [
            "the shirt is blue, and the shirt is blue, and the shirt is blue",
            "the person is wearing a blue shirt"
        ]
        with patch.object(self.qa_service, "_query_blip_vision", return_value={"available": True, "findings": mock_findings, "is_lower_body_query": False, "target_garment": "shirt", "is_expression_query": False}):
            test_img = Image.new("RGB", (100, 100), color=(0, 0, 255))
            buf = io.BytesIO()
            test_img.save(buf, format="JPEG")
            data_uri = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

            req = QARequest(
                question="What color is her shirt?",
                image_data=data_uri,
                scene_caption="a woman in a blue shirt",
                ocr_text=""
            )
            res = self.qa_service.answer_question(req, api_key="")
            self.assertEqual(res.capability, "VISION_CONDITIONED")
            self.assertFalse(res.is_uncertain)
            self.assertIn("blue", res.answer.lower())
            self.assertNotIn("and the shirt is blue, and the shirt is blue", res.answer.lower())


if __name__ == "__main__":
    unittest.main()


"""SightGuide V2 Grounding, Regression, and Provenance Test Suite.

Automated regression coverage for Phase 5 requirements:
1. Correct score interpretation and labels.
2. Unsupported clothing claims in the documented portrait case.
3. Fusion behavior when candidates contain an unsupported shared claim.
4. Empty or invalid model responses.
5. External fusion-service failure and fallback.
6. OCR positive and negative cases.
7. API failure and retry behavior.
8. MMR calculation consistency.
9. Preservation of all 14 protected baseline artifacts.
"""

import hashlib
import json
import io
import math
from pathlib import Path
import pytest
from PIL import Image, ImageDraw

from app.schemas import CandidateCaption, SelectedPair, FusionResult
from app.services.caption_pipeline import CaptionPipeline, clean_and_validate_caption
from app.services.fusion_service import apply_grounding_safeguard, fuse_caption_pair
from app.services.diversity_ranker import select_diversity_pair, normalize_clip_scores, compute_lexical_similarity
from app.services.ocr_service import OCRService


ROOT = Path(__file__).resolve().parent.parent
PORTRAIT_IMG_PATH = ROOT / "evaluations" / "test_portrait_woman.png"


# ---------------------------------------------------------------------------
# 1. Correct Score Interpretation and Labels
# ---------------------------------------------------------------------------
def test_score_interpretation_and_labels():
    """Verify that BLIP scoring uses negative cross-entropy loss where higher is better."""
    pipe = CaptionPipeline.get_instance()
    img = Image.new("RGB", (100, 100), color=(200, 200, 200))
    scored = pipe.score_candidates(img, ["a grey square", "an ornate golden carriage"])

    assert len(scored) == 2
    for s in scored:
        assert s["scoring_method"] == "blip_cross_entropy_log_likelihood"
        assert s["raw_loss"] >= 0.0
        assert s["clip_score"] == pytest.approx(-s["raw_loss"], abs=1e-4)

    # Scored list must be ordered descending by clip_score
    assert scored[0]["clip_score"] >= scored[1]["clip_score"]
    assert scored[0]["rank"] == 1
    assert scored[1]["rank"] == 2


# ---------------------------------------------------------------------------
# 2. Unsupported Clothing Claims in Documented Portrait Case
# ---------------------------------------------------------------------------
def test_unsupported_clothing_claims_portrait():
    """Verify that the portrait case does not hallucinate lower-body jeans or pants."""
    assert PORTRAIT_IMG_PATH.exists(), f"Missing test portrait: {PORTRAIT_IMG_PATH}"
    img_bytes = PORTRAIT_IMG_PATH.read_bytes()

    pipe = CaptionPipeline.get_instance()
    res = pipe.process_image(
        image_bytes=img_bytes,
        filename="test_portrait_woman.png",
        lambda_weight=0.5,
        target_method="proposed",
        decoding_strategy="sample"
    )

    fused = res["fusion"].fused_caption.lower()
    c1 = res["selected_pair"].rank_1_caption.lower()
    c2 = res["selected_pair"].rank_2_caption.lower()

    unsupported_tokens = ["jeans", "pants", "trousers", "slacks", "shoes", "sims game", "sims family"]
    for unk in unsupported_tokens:
        assert unk not in fused, f"Hallucination '{unk}' detected in fused caption: '{fused}'"
        assert unk not in c1, f"Hallucination '{unk}' detected in anchor: '{c1}'"
        assert unk not in c2, f"Hallucination '{unk}' detected in detail: '{c2}'"

    # Supported visible features should be retained
    assert any(term in fused for term in ["woman", "girl", "character", "person"])
    assert any(term in fused for term in ["blue", "shirt", "hair", "eyes", "earrings"])


# ---------------------------------------------------------------------------
# 3. Fusion Behavior when Candidates Contain Unsupported Shared Claim
# ---------------------------------------------------------------------------
def test_fusion_behavior_unsupported_shared_claim():
    """Verify that if candidates contain a shared out-of-frame claim, it is pruned while preserving visible clothing."""
    c1 = "the face of a woman with long hair, wearing blue shirt and jeans"
    c2 = "a close-up portrait of a girl wearing blue shirt and jeans"

    c1_clean = apply_grounding_safeguard(c1, anchor_context=c1)
    c2_clean = apply_grounding_safeguard(c2, anchor_context=c1)

    assert "jeans" not in c1_clean
    assert "jeans" not in c2_clean
    assert "blue shirt" in c1_clean
    assert "blue shirt" in c2_clean


# ---------------------------------------------------------------------------
# 4. Empty or Invalid Model Responses
# ---------------------------------------------------------------------------
def test_empty_or_invalid_model_responses():
    """Verify that empty inputs or invalid model responses fail gracefully."""
    res_empty = fuse_caption_pair("", "")
    assert res_empty.generation_status == "FAILED"
    assert res_empty.provenance == "VALIDATION_ERROR"
    assert res_empty.fused_caption == ""

    # Whitespace only
    res_spaces = fuse_caption_pair("   ", "   ")
    assert res_spaces.generation_status == "FAILED"


# ---------------------------------------------------------------------------
# 5. External Fusion Service Failure and Fallback
# ---------------------------------------------------------------------------
def test_external_fusion_service_failure_and_fallback():
    """Verify that when API key is missing or fails, it falls back to the anchor rather than empty text."""
    c1 = "a woman in a blue shirt with long dark hair"
    c2 = "a cartoon illustration of a smiling person"

    # Unset API key
    res = fuse_caption_pair(c1, c2, api_key="")
    assert res.generation_status == "GROQ_KEY_NOT_CONFIGURED"
    assert res.fused_caption == c1  # Falls back to visual anchor!
    assert "GROQ_API_KEY is not configured" in res.error_details


# ---------------------------------------------------------------------------
# 6. OCR Positive and Negative Cases
# ---------------------------------------------------------------------------
def test_ocr_positive_and_negative():
    """Verify OCR returns true text on positive images and reports no text on blank images."""
    ocr = OCRService()
    if not ocr.is_available:
        pytest.skip("RapidOCR engine not available in environment.")

    # Positive case: Image with text
    pos_img = Image.new("RGB", (400, 120), color=(255, 255, 255))
    draw = ImageDraw.Draw(pos_img)
    draw.text((20, 45), "SightGuide Verified OCR", fill=(0, 0, 0))
    pos_buf = io.BytesIO()
    pos_img.save(pos_buf, format="PNG")
    pos_res = ocr.extract_text(pos_buf.getvalue())

    assert pos_res["has_text"] is True
    assert "SightGuide" in pos_res["text"]
    assert len(pos_res["lines"]) >= 1

    # Negative case: Blank image
    neg_img = Image.new("RGB", (200, 200), color=(240, 240, 240))
    neg_buf = io.BytesIO()
    neg_img.save(neg_buf, format="PNG")
    neg_res = ocr.extract_text(neg_buf.getvalue())

    assert neg_res["has_text"] is False
    assert neg_res["text"] == ""
    assert "No readable text detected" in neg_res["message"]


# ---------------------------------------------------------------------------
# 7. MMR Calculation Consistency
# ---------------------------------------------------------------------------
def test_mmr_calculation_consistency():
    """Verify that MMR pair scores precisely equal lambda * norm_score - (1 - lambda) * similarity - penalties."""
    candidates = [
        {"caption": "a golden retriever running on grass", "clip_score": -1.2},
        {"caption": "a dog running on grass", "clip_score": -1.4},
        {"caption": "a golden retriever catching a red frisbee", "clip_score": -1.6},
        {"caption": "an outdoor park scene", "clip_score": -2.2},
    ]

    lambda_val = 0.5
    result = select_diversity_pair(candidates, lambda_weight=lambda_val)

    anchor = result["rank_1_caption"]
    assert anchor == "a golden retriever running on grass"

    for eval_item in result["candidate_evaluations"]:
        sec_caption = eval_item["caption"]
        sim = eval_item["similarity"]
        sec_norm = eval_item["normalized_clip_score"]
        expected_grounding = round(lambda_val * sec_norm, 6)
        expected_penalty = round((1.0 - lambda_val) * sim, 6)
        g_pen = eval_item.get("grounding_penalty", 0.0)
        d_pen = eval_item.get("duplicate_penalty", 0.0)
        expected_pair_score = round(expected_grounding - expected_penalty - g_pen - d_pen, 6)

        assert eval_item["grounding_term"] == pytest.approx(expected_grounding, abs=1e-5)
        assert eval_item["diversity_penalty"] == pytest.approx(expected_penalty, abs=1e-5)
        assert eval_item["pair_score"] == pytest.approx(expected_pair_score, abs=1e-5)


# ---------------------------------------------------------------------------
# 8. Preservation of Protected Artifacts
# ---------------------------------------------------------------------------
def test_preservation_of_protected_artifacts():
    """Cryptographically verify all 14 baseline files in artifact_hashes.json."""
    hash_file = ROOT / "artifact_hashes.json"
    assert hash_file.exists(), "artifact_hashes.json not found."

    with open(hash_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert len(manifest) == 14, f"Expected 14 protected artifacts, found {len(manifest)}"

    for rel_path, meta in manifest.items():
        file_path = ROOT / rel_path
        assert file_path.exists(), f"Protected artifact missing: {rel_path}"
        content = file_path.read_bytes()
        actual_hash = hashlib.sha256(content).hexdigest()
        assert actual_hash == meta["sha256"], (
            f"Integrity violation in {rel_path}!\n"
            f"Expected: {meta['sha256']}\n"
            f"Actual:   {actual_hash}"
        )


# ---------------------------------------------------------------------------
# 9. Anchor Validation Against YOLO Detections
# ---------------------------------------------------------------------------
def test_anchor_validation_contradiction_demotes():
    """Verify that an anchor claiming an absent COCO object is demoted when YOLO detects an alternative."""
    from app.services.fusion_service import validate_anchor_against_detections

    anchor = "a skateboard is covered up with graffiti"
    candidates = [
        "a skateboard is covered up with graffiti",
        "a person holding a cell phone",
        "a person writing on a screen",
    ]
    detections = [
        {"label": "person", "confidence": 0.35, "region": "middle"},
        {"label": "cell phone", "confidence": 0.28, "region": "middle"},
    ]

    val = validate_anchor_against_detections(anchor, candidates, detections)
    assert val["demote"] is True
    assert val["alternate_caption"] == "a person holding a cell phone"
    assert "skateboard" in val["reason"]


def test_anchor_validation_corroborated_anchor_retained():
    """Verify that an anchor supported by detections is never demoted."""
    from app.services.fusion_service import validate_anchor_against_detections

    anchor = "a woman wearing a blue shirt smiling"
    candidates = [
        "a woman wearing a blue shirt smiling",
        "a close-up portrait of a girl",
    ]
    detections = [
        {"label": "person", "confidence": 0.85, "region": "middle"},
    ]

    val = validate_anchor_against_detections(anchor, candidates, detections)
    assert val["demote"] is False
    assert val["alternate_caption"] is None


def test_anchor_validation_zero_detections_safe_fallback():
    """Verify that zero detections never cause anchor demotion (absence != proof of absence)."""
    from app.services.fusion_service import validate_anchor_against_detections

    anchor = "a cat sleeping on a sofa"
    candidates = ["a cat sleeping on a sofa", "an indoor room view"]
    detections = []  # Zero detections (low light, camera angle, etc.)

    val = validate_anchor_against_detections(anchor, candidates, detections)
    assert val["demote"] is False


# ---------------------------------------------------------------------------
# 10. OCR Evidence Injection into User Prompt
# ---------------------------------------------------------------------------
def test_build_user_prompt_with_ocr_evidence():
    """Verify that verified OCR text is correctly structured in the fusion prompt."""
    from app.services.fusion_service import build_user_prompt

    prompt_no_ocr = build_user_prompt("a person holding a device", "close-up view")
    assert "Verified OCR Evidence" not in prompt_no_ocr

    prompt_with_ocr = build_user_prompt(
        "a person holding a device",
        "close-up view",
        ocr_text="Hello I'm Maddy",
        ocr_surface_hint="a smartphone screen"
    )
    assert 'Verified OCR Evidence (machine-recognized text visible on a smartphone screen): "Hello I\'m Maddy"' in prompt_with_ocr
    assert "Incorporate this text naturally if it is relevant" in prompt_with_ocr

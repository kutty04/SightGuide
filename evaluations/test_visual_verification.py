"""Test visual grounding verification and enhanced fusion prompt."""

import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.services.caption_pipeline import CaptionPipeline

pipe = CaptionPipeline.get_instance()
img = Image.open("evaluations/test_portrait_woman.png").convert("RGB")

def test_visual_verification():
    # Test candidates from reproduction
    c1 = "the face of a woman with long black hair, wearing blue shirt"
    c_jeans = "the face of a woman with long hair, wearing blue shirt and jeans"
    c_sims = "a cartoon character in the sims game"
    c_clean = "a cartoon character with long dark hair and brown eyes"

    print("Testing visual verification on claims:")
    claims = [
        "woman with long black hair",
        "wearing blue shirt",
        "wearing jeans",
        "in the sims game",
        "with brown eyes",
        "wearing earrings"
    ]

    for claim in claims:
        # Measure BLIP conditional loss
        score_res = pipe.score_candidates(img, [claim])[0]
        loss = score_res["raw_loss"]
        # Grounded if loss is low
        is_grounded = loss < 5.5
        print(f"  Claim: '{claim:28}' -> Loss: {loss:.3f} | Visually Grounded: {is_grounded}")

if __name__ == "__main__":
    test_visual_verification()

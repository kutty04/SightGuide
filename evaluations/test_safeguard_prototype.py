"""Prototype grounding safeguard test."""

import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.services.caption_pipeline import CaptionPipeline

pipe = CaptionPipeline.get_instance()
img = Image.open("evaluations/test_portrait_woman.png").convert("RGB")

anchor = "the face of a woman with long black hair, wearing blue shirt"
candidate_with_jeans = "the face of a woman with long hair, wearing blue shirt and jeans"
candidate_with_sims = "a cartoon character in the sims game"
candidate_with_eyes = "a cartoon character with long hair and brown eyes"

print("ANCHOR SCORE:")
scored_anchor = pipe.score_candidates(img, [anchor])[0]
print(f"  Anchor score: {scored_anchor['clip_score']} | loss: {scored_anchor['raw_loss']}")

print("\nCANDIDATES SCORES:")
scored_cands = pipe.score_candidates(img, [candidate_with_jeans, candidate_with_sims, candidate_with_eyes])
for c in scored_cands:
    print(f"  Caption: {c['caption']} | score: {c['clip_score']} | loss: {c['raw_loss']}")

print("\nNOVELTY ITEMS SCORES:")
items = ['with brown eyes', 'brown eyes', 'wearing jeans', 'jeans', 'in the sims', 'the sims']
for it in items:
    s = pipe.score_candidates(img, [it])[0]
    print(f"  Item: '{it}' | score: {s['clip_score']} | loss: {s['raw_loss']}")

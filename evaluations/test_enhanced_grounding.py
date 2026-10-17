"""Test enhanced grounding and fusion on the portrait image."""

import sys
import re
import os
from pathlib import Path
from PIL import Image

_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ[_k.strip()] = _v.strip().strip('"').strip("'")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.services.caption_pipeline import CaptionPipeline, clean_and_validate_caption
from app.services.fusion_service import fuse_caption_pair

pipe = CaptionPipeline.get_instance()
img_path = Path("evaluations/test_portrait_woman.png")
img = Image.open(img_path).convert("RGB")

print("1. Testing BLIP candidates on portrait image:")
cands = pipe.generate_candidates(img, num_candidates=5, decoding_strategy="sample", seed=100)
print("Raw generated candidates (seed 100):")
for idx, c in enumerate(cands, 1):
    print(f"  {idx}. {c}")

scored = pipe.score_candidates(img, cands)
print("\nScored candidates:")
for s in scored:
    print(f"  Rank {s['rank']}: {s['caption']} (loss={s['raw_loss']}, score={s['clip_score']})")

# Test fusion on top 2
c1 = scored[0]["caption"]
c2 = scored[1]["caption"]
print(f"\nAnchor c1*: {c1}")
print(f"Detail c2*: {c2}")

fusion = fuse_caption_pair(c1, c2)
print(f"\nFused Caption ({fusion.generation_status}):")
print(f"  {fusion.fused_caption}")
if fusion.error_details:
    print(f"  Details: {fusion.error_details}")

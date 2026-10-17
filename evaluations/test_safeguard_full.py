"""Test full end-to-end safeguard on the portrait image."""

import sys
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
from app.services.caption_pipeline import CaptionPipeline

img_path = Path("evaluations/test_portrait_woman.png")
image_bytes = img_path.read_bytes()

pipe = CaptionPipeline.get_instance()
res = pipe.process_image(
    image_bytes=image_bytes,
    filename="test_portrait_woman.png",
    lambda_weight=0.5,
    target_method="proposed",
    decoding_strategy="sample"
)

print("ANCHOR (c1*):", res["selected_pair"].rank_1_caption)
print("DETAIL (c2*):", res["selected_pair"].rank_2_caption)
print("FUSION STATUS:", res["fusion"].generation_status)
print("FUSED CAPTION:", res["fusion"].fused_caption)
print("ALL CANDIDATES:")
for c in res["candidates"]:
    print(f"  - {c.caption} (score={c.clip_score})")

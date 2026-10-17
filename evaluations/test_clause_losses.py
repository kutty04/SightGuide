import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.services.caption_pipeline import CaptionPipeline

pipe = CaptionPipeline.get_instance()
img = Image.open("evaluations/test_portrait_woman.png").convert("RGB")

test_phrases = [
    "a woman with long black hair",
    "a woman wearing a blue shirt",
    "wearing a blue shirt",
    "wearing blue jeans",
    "wearing jeans",
    "wearing pants",
    "in the sims game",
    "a cartoon character",
    "wearing earrings",
    "wearing a necklace"
]

scored = pipe.score_candidates(img, test_phrases)
print("CLAUSE CROSS-ENTROPY LOSSES:")
for s in scored:
    print(f"  Loss: {s['raw_loss']:.3f} | Score: {s['clip_score']:.3f} | Phrase: {s['caption']}")

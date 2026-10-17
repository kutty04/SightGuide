import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.app.services.caption_pipeline import CaptionPipeline

pipe = CaptionPipeline.get_instance()

portrait_smiling = Image.open('evaluations/test_portrait_woman.png')
neutral_woman = Image.open('dataset/toa-heftiba-O3ymvT7Wf9U-unsplash.jpg')
man_hat = Image.open('dataset/ian-dooley-d1UPkiFd04A-unsplash.jpg')
landscape = Image.open('dataset/sample-clouds-400x300.jpg')

# Create a cropped image of woman where face is cut off (lower torso only)
w, h = portrait_smiling.size
obscured_face = portrait_smiling.crop((0, int(h * 0.6), w, h))

test_set = [
    ("Visible Smile (test_portrait_woman)", portrait_smiling),
    ("Neutral / Serious (toa_heftiba_woman)", neutral_woman),
    ("Smiling / Grinning Man (ian_dooley_man)", man_hat),
    ("Face Out of Frame (portrait lower crop)", obscured_face),
    ("No Person (clouds landscape)", landscape),
]

candidates = [
    "the person is smiling",
    "the person is not smiling",
    "the face is obscured or not visible"
]

for name, im in test_set:
    print(f"\n--- {name} ---")
    scored = pipe.score_candidates(im, candidates)
    for s in scored:
        print(f"  loss: {s['raw_loss']:.4f} (score: {s['clip_score']:+.4f}) -> {s['caption']}")

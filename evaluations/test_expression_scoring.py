import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.app.services.caption_pipeline import CaptionPipeline

pipe = CaptionPipeline.get_instance()

images = [
    ("test_portrait_woman", Image.open('evaluations/test_portrait_woman.png')),
    ("ian_dooley_man", Image.open('dataset/ian-dooley-d1UPkiFd04A-unsplash.jpg')),
    ("toa_heftiba_woman", Image.open('dataset/toa-heftiba-O3ymvT7Wf9U-unsplash.jpg')),
]

statements = [
    "a portrait of a smiling person",
    "the person is smiling",
    "a photo of a smiling woman",
    "a photo of a smiling man",
    "a photo of a person with a neutral expression",
    "the person is not smiling",
    "a photo of a person with a serious expression",
    "the face is not visible"
]

for name, im in images:
    print(f"\n=================== IMAGE: {name} ===================")
    scored = pipe.score_candidates(im, statements)
    for s in scored:
        print(f"loss: {s['raw_loss']:.4f} (score: {s['clip_score']:+.4f}) -> {s['caption']}")

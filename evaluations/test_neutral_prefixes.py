import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.app.services.caption_pipeline import CaptionPipeline

pipe = CaptionPipeline.get_instance()

portrait_smiling = Image.open('evaluations/test_portrait_woman.png')
neutral_woman = Image.open('dataset/toa-heftiba-O3ymvT7Wf9U-unsplash.jpg')
landscape = Image.open('dataset/sample-clouds-400x300.jpg')
w, h = portrait_smiling.size
crop_lower = portrait_smiling.crop((0, int(h * 0.55), w, h))

test_cases = [
    ("Smiling portrait (test_portrait_woman)", portrait_smiling),
    ("Neutral woman (toa_heftiba_woman)", neutral_woman),
    ("Crop no head (torso only)", crop_lower),
    ("No person (clouds)", landscape)
]

candidate_neutral_prefixes = [
    "a close up of the face",
    "the expression on the face is",
    "the person's expression is",
    "the face is",
    "a portrait of",
    "the person looks"
]

for name, im in test_cases:
    print(f"\n==================== {name} ====================")
    for p in candidate_neutral_prefixes:
        inputs = pipe.blip_processor(im, p, return_tensors='pt').to(pipe.device)
        out = pipe.blip_model.generate(**inputs, max_new_tokens=25, num_beams=3)
        dec = pipe.blip_processor.decode(out[0], skip_special_tokens=True).strip()
        print(f"  [{p}] -> [{dec}]")

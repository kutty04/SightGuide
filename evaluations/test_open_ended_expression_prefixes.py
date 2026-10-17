import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.app.services.caption_pipeline import CaptionPipeline

pipe = CaptionPipeline.get_instance()

portrait_smiling = Image.open('evaluations/test_portrait_woman.png')
neutral_woman = Image.open('dataset/toa-heftiba-O3ymvT7Wf9U-unsplash.jpg')
man_smiling = Image.open('dataset/ian-dooley-d1UPkiFd04A-unsplash.jpg')
landscape = Image.open('dataset/sample-clouds-400x300.jpg')
w, h = portrait_smiling.size
crop_torso = portrait_smiling.crop((0, 320, w, h))

test_cases = [
    ("Smiling girl (test_portrait_woman)", portrait_smiling, "the girl is", "girl"),
    ("Smiling man (ian_dooley_man)", man_smiling, "the man is", "man"),
    ("Neutral woman (toa_heftiba_woman)", neutral_woman, "the woman is", "woman"),
    ("No head torso crop", crop_torso, "the woman is", "woman"),
    ("Clouds landscape", landscape, "the person is", "person")
]

open_prefixes = [
    "{person_phrase} looking at the camera with a",
    "{person_phrase} with a",
    "a portrait of a {person_noun} with a",
    "a photo of a {person_noun} who is",
    "the expression on the {person_noun}'s face is",
    "the {person_noun}'s face shows a"
]

for name, im, p_phrase, p_noun in test_cases:
    print(f"\n==================== {name} ====================")
    for tpl in open_prefixes:
        p = tpl.format(person_phrase=p_phrase, person_noun=p_noun)
        inputs = pipe.blip_processor(im, p, return_tensors='pt').to(pipe.device)
        out = pipe.blip_model.generate(**inputs, max_new_tokens=25, num_beams=3)
        dec = pipe.blip_processor.decode(out[0], skip_special_tokens=True).strip()
        # Extract only the completion
        completion = dec[len(p):].strip()
        print(f"  [{p}] -> completion: [{completion}] | full: [{dec}]")

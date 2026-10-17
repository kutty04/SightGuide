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
crop_no_head = portrait_smiling.crop((0, int(h * 0.5), w, h))

test_cases = [
    ("Smiling portrait (test_portrait_woman)", portrait_smiling, "the woman is", "woman"),
    ("Neutral woman (toa_heftiba_woman)", neutral_woman, "the woman is", "woman"),
    ("No person (clouds)", landscape, "the person is", "person"),
    ("Crop no head (torso only)", crop_no_head, "the woman is", "woman")
]

prefixes_template = [
    "{person_phrase} smiling",
    "the facial expression of {person_phrase}",
    "the face of {person_phrase}",
    "a portrait of a {person_noun}"
]

for name, im, p_phrase, p_noun in test_cases:
    print(f"\n=== {name} ===")
    for tpl in prefixes_template:
        p = tpl.format(person_phrase=p_phrase, person_noun=p_noun)
        inputs = pipe.blip_processor(im, p, return_tensors='pt').to(pipe.device)
        out = pipe.blip_model.generate(**inputs, max_new_tokens=30, num_beams=3)
        dec = pipe.blip_processor.decode(out[0], skip_special_tokens=True).strip()
        print(f"  [{p}] -> [{dec}]")

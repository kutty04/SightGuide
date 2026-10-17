import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.app.services.caption_pipeline import CaptionPipeline

pipe = CaptionPipeline.get_instance()

portrait = Image.open('evaluations/test_portrait_woman.png')
man_hat = Image.open('dataset/ian-dooley-d1UPkiFd04A-unsplash.jpg')
woman_hoodie = Image.open('dataset/toa-heftiba-O3ymvT7Wf9U-unsplash.jpg')
no_person = Image.open('dataset/sample-clouds-400x300.jpg')

images = [
    ("test_portrait_woman (smiling)", portrait),
    ("man_hat (smiling/posed)", man_hat),
    ("woman_hoodie (serious/moody)", woman_hoodie),
    ("no_person (clouds)", no_person)
]

# Let's test targeted prefixes
def get_expression_prefixes(q: str, person_phrase: str, person_noun: str):
    prefixes = []
    # If question specifically mentions smile/smiling
    if any(w in q.lower() for w in ['smile', 'smiling']):
        prefixes.append(f"{person_phrase} smiling")
        prefixes.append(f"a portrait of a smiling {person_noun}")
        prefixes.append(f"{person_phrase} not smiling")
    # General expression queries
    prefixes.append(f"the expression on the {person_noun}'s face is")
    prefixes.append(f"the facial expression of {person_phrase}")
    prefixes.append(f"{person_phrase} in the photo is")
    return prefixes

for name, img in images:
    print(f"\n==================== {name} ====================")
    prefs = get_expression_prefixes("Is the girl smiling?", "the girl is", "girl")
    for p in prefs[:3]:
        inputs = pipe.blip_processor(img, p, return_tensors='pt').to(pipe.device)
        out = pipe.blip_model.generate(**inputs, max_new_tokens=30, num_beams=3)
        dec = pipe.blip_processor.decode(out[0], skip_special_tokens=True).strip()
        print(f"  [{p}] -> [{dec}]")

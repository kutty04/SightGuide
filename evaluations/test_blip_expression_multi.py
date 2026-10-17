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

candidate_prefixes = [
    "the person is smiling",
    "the person is not smiling",
    "the expression on the person's face is",
    "the person's facial expression",
    "the person appears to be",
    "the face is",
]

for name, im in images:
    print(f"\n=================== IMAGE: {name} ===================")
    for p in candidate_prefixes:
        inputs = pipe.blip_processor(im, p, return_tensors='pt').to(pipe.device)
        out = pipe.blip_model.generate(**inputs, max_new_tokens=30, num_beams=3)
        dec = pipe.blip_processor.decode(out[0], skip_special_tokens=True).strip()
        print(f"[{p}] -> [{dec}]")

import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.app.services.caption_pipeline import CaptionPipeline

pipe = CaptionPipeline.get_instance()
img = Image.open('evaluations/test_portrait_woman.png')

test_prefixes = [
    "the girl is smiling",
    "the girl is",
    "the expression on the girl's face is",
    "the girl's facial expression is",
    "the girl looks",
    "the girl appears to be",
    "the face of the girl is",
    "the person is smiling",
    "the person's expression is",
    "the facial expression is",
    "a portrait of a smiling",
    "the person is looking"
]

print("Testing prefixes with BLIP generate on test_portrait_woman.png:")
for p in test_prefixes:
    inputs = pipe.blip_processor(img, p, return_tensors='pt').to(pipe.device)
    out = pipe.blip_model.generate(**inputs, max_new_tokens=35, num_beams=3)
    dec = pipe.blip_processor.decode(out[0], skip_special_tokens=True).strip()
    print(f"Prefix: [{p}] -> Decoded: [{dec}]")

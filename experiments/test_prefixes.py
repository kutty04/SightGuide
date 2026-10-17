import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app.services.caption_pipeline import CaptionPipeline

pipeline = CaptionPipeline.get_instance()
im = Image.open(ROOT / "evaluations" / "test_portrait_woman.png")

prefixes = [
    "the expression on the person's face is",
    "the color of the shirt is",
    "the woman with",
    "the person's hair is",
]

print("--- WITHOUT repetition penalty ---")
for p in prefixes:
    inputs = pipeline.blip_processor(im, p, return_tensors="pt").to(pipeline.device)
    out = pipeline.blip_model.generate(**inputs, max_new_tokens=35, num_beams=3)
    dec = pipeline.blip_processor.decode(out[0], skip_special_tokens=True).strip()
    print(f'P: "{p}" -> "{dec}"')

print("\n--- WITH repetition penalty 1.25 ---")
for p in prefixes:
    inputs = pipeline.blip_processor(im, p, return_tensors="pt").to(pipeline.device)
    out = pipeline.blip_model.generate(**inputs, max_new_tokens=35, num_beams=3, repetition_penalty=1.25)
    dec = pipeline.blip_processor.decode(out[0], skip_special_tokens=True).strip()
    print(f'P: "{p}" -> "{dec}"')


from pathlib import Path
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL_DIR = Path(
    r"C:\Users\Admin\.cache\huggingface\hub\models--Salesforce--blip-image-captioning-base\snapshots\82a37760796d32b1411fe092ab5d4e227313294b"
)

print("SightGuide: Loading BLIP from local cache...")

if not MODEL_DIR.is_dir():
    raise FileNotFoundError(f"Model folder not found: {MODEL_DIR}")

processor = BlipProcessor.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True
)

model = BlipForConditionalGeneration.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True,
    use_safetensors=False
)

model.eval()

print("BLIP model loaded successfully!")

image_path = input(
    "Enter the full path to an image: "
).strip().strip('"')

if not Path(image_path).is_file():
    raise FileNotFoundError(f"Image not found: {image_path}")

image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")

print("\nGenerating 3 candidate captions...")

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        do_sample=True,
        num_return_sequences=3,
        num_beams=1,
        max_new_tokens=60,
        top_p=0.9,
        temperature=0.8
    )

captions = processor.batch_decode(
    outputs,
    skip_special_tokens=True
)

# Remove duplicate captions while preserving their order.
unique_captions = list(dict.fromkeys(
    caption.strip() for caption in captions if caption.strip()
))

print("\n" + "=" * 45)
print("SIGHTGUIDE: CANDIDATE CAPTIONS")
print("=" * 45)

for index, caption in enumerate(unique_captions, start=1):
    print(f"\nCandidate {index}:")
    print(caption)

if len(unique_captions) < 3:
    print(
        "\nNote: BLIP generated fewer than 3 unique captions "
        "for this image."
    )

print("\nCandidate generation completed!")

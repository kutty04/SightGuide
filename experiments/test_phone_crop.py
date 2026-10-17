import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR()
img_path = ROOT / "evaluations" / "ocr_test_images" / "phone_handwritten_screen.png"
pil_img = Image.open(img_path).convert("RGB")
np_orig = np.array(pil_img)

# Let's inspect the phone region
# Phone screen in original image (1024x576):
# The phone is from y ~ 0 to 576, x ~ 320 to 800
# Let's crop just the phone:
phone_crop = pil_img.crop((320, 0, 800, 576))
phone_crop.save(ROOT / "experiments" / "phone_screen_cropped.png")

# Run OCR on phone crop directly:
res_phone, _ = ocr(np.array(phone_crop))
print("OCR on phone crop alone:", res_phone)

# What if we invert the phone crop (black background -> white background)?
phone_inv = Image.fromarray(255 - np.array(phone_crop))
phone_inv.save(ROOT / "experiments" / "phone_screen_inv.png")
res_phone_inv, _ = ocr(np.array(phone_inv))
print("OCR on inverted phone crop:", res_phone_inv)

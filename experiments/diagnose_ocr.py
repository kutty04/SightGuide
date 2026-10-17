import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from rapidocr_onnxruntime import RapidOCR

engine = RapidOCR()

img_path = ROOT / "evaluations" / "ocr_test_images" / "phone_handwritten_screen.png"
pil_img = Image.open(img_path).convert("RGB")
np_orig = np.array(pil_img)

print("--- 1. Raw Original ---")
res, _ = engine(np_orig)
print("Raw engine result:", res)

# Test Inversion (Dark background with white text inverted to light background with dark text)
print("\n--- 2. Color Inversion (Negative) ---")
np_inv = 255 - np_orig
res_inv, _ = engine(np_inv)
print("Inverted engine result:", res_inv)

# Test Grayscale
print("\n--- 3. Grayscale Inverted ---")
gray = cv2.cvtColor(np_orig, cv2.COLOR_RGB2GRAY)
gray_inv = 255 - gray
res_gray_inv, _ = engine(gray_inv)
print("Gray inverted engine result:", res_gray_inv)

# Test Contrast Enhancement on Inverted
print("\n--- 4. Contrast Enhanced Inverted ---")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl = clahe.apply(gray_inv)
res_cl, _ = engine(cl)
print("CLAHE inverted engine result:", res_cl)

# Test Cropped Phone Screen
# The phone screen is approximately in the center
# Let's find out what the phone screen crop yields
w, h = pil_img.size
# Crop center phone screen: x from 0.3*w to 0.8*w, y from 0.05*h to 0.95*h
phone_crop = pil_img.crop((int(0.3 * w), int(0.02 * h), int(0.82 * w), int(0.98 * h)))
np_crop = np.array(phone_crop)
res_crop, _ = engine(np_crop)
print("\n--- 5. Phone Screen Crop (Original) ---")
print("Crop original result:", res_crop)

np_crop_inv = 255 - np_crop
res_crop_inv, _ = engine(np_crop_inv)
print("\n--- 6. Phone Screen Crop (Inverted) ---")
print("Crop inverted result:", res_crop_inv)

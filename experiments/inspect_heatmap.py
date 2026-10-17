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
img = Image.open(ROOT / "experiments" / "phone_screen_cropped.png")
np_img = np.array(img)

# Let's inspect det
det = ocr.text_det

# Preprocess image for det
# Let's inspect how det prepares image:
src_h, src_w = np_img.shape[:2]
# det resize:
limit_side_len = det.limit_side_len
ratio = float(limit_side_len) / min(src_h, src_w)
resize_h = int(round(src_h * ratio / 32) * 32)
resize_w = int(round(src_w * ratio / 32) * 32)

resized_img = cv2.resize(np_img, (resize_w, resize_h))
norm_img = (resized_img.astype(np.float32) / 255.0 - np.array(det.mean)) / np.array(det.std)
tensor_img = np.transpose(norm_img, (2, 0, 1))[np.newaxis, :].astype(np.float32)

preds = det.infer(tensor_img)
prob_map = preds[0][0, 0, :, :]
print("Prob map shape:", prob_map.shape)
print("Min:", prob_map.min(), "Max:", prob_map.max())

# Save probability map as an image
prob_img = (prob_map * 255).astype(np.uint8)
Image.fromarray(prob_img).save(ROOT / "experiments" / "dbnet_heatmap.png")
print("Saved dbnet_heatmap.png")

# Now let's test contours on thresholded probability map
for th in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    binary = (prob_map > th).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 20]
    print(f"Thresh {th:.1f}: {len(valid_contours)} contours found with areas {[int(cv2.contourArea(c)) for c in valid_contours]}")

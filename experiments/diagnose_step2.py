import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from rapidocr_onnxruntime import RapidOCR

engine = RapidOCR()
img_path = ROOT / "evaluations" / "ocr_test_images" / "phone_handwritten_screen.png"
pil_img = Image.open(img_path).convert("RGB")
np_orig = np.array(pil_img)

dt_boxes, _ = engine.text_det(np_orig)
print("Detected boxes count:", len(dt_boxes) if dt_boxes is not None else 0)
print("Box 0:\n", dt_boxes[0])

# Get crop image list for recognition
crop_img_list = engine.get_crop_img_list(np_orig, dt_boxes)
print("Crop image count:", len(crop_img_list))
crop0 = crop_img_list[0]
print("Crop 0 shape:", crop0.shape)

# Save crop 0 to disk to see what DBNet cropped!
Image.fromarray(crop0).save(ROOT / "experiments" / "crop0_debug.png")
print("Saved crop0_debug.png")

# Now let's see what text_rec does on crop0:
rec_res, elapse = engine.text_rec([crop0])
print("Recognition result on crop0:", rec_res)

# Check what filter_result or text_score threshold is used:
print("engine.text_score threshold:", engine.text_score)
print("engine.min_height:", engine.min_height)

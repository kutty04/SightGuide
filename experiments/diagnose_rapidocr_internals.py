import sys
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from rapidocr_onnxruntime import RapidOCR

engine = RapidOCR()
print("RapidOCR attributes:", dir(engine))

# Check parameters
if hasattr(engine, "text_det"):
    print("text_det:", dir(engine.text_det))
    if hasattr(engine.text_det, "args"):
        print("text_det args:", engine.text_det.args)

img_path = ROOT / "evaluations" / "ocr_test_images" / "phone_handwritten_screen.png"
pil_img = Image.open(img_path).convert("RGB")
np_orig = np.array(pil_img)

# Let's inspect text detection directly
dt_boxes, elapse = engine.text_det(np_orig)
print("text_det on raw image dt_boxes:", dt_boxes)

# Let's try text_det on inverted
dt_boxes_inv, _ = engine.text_det(255 - np_orig)
print("text_det on inverted dt_boxes:", dt_boxes_inv)

# What about with lower box_thresh / unclip_ratio?
# Let's see what parameters RapidOCR can be initialized with!
import rapidocr_onnxruntime
print("rapidocr_onnxruntime version:", rapidocr_onnxruntime.__version__)
print("rapidocr file:", rapidocr_onnxruntime.__file__)

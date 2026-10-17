import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from rapidocr_onnxruntime import RapidOCR

img_path = ROOT / "evaluations" / "ocr_test_images" / "phone_handwritten_screen.png"
pil_img = Image.open(img_path).convert("RGB")
np_orig = np.array(pil_img)

print("--- Testing RapidOCR parameter variations on phone image ---")

# Let's test passing det parameters to RapidOCR:
# RapidOCR allows kwargs or modifying postprocess_op:
for unclip in [1.5, 1.3, 1.2, 1.1, 1.0]:
    for box_thresh in [0.5, 0.4, 0.3]:
        ocr = RapidOCR()
        ocr.text_det.postprocess_op.unclip_ratio = unclip
        ocr.text_det.postprocess_op.box_thresh = box_thresh
        res, _ = ocr(np_orig)
        if res:
            print(f"unclip={unclip}, box_thresh={box_thresh} -> {len(res)} lines:")
            for item in res:
                print(f"    {item[1]} (conf: {item[2]:.3f})")

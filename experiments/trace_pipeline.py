import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR()
img_path = ROOT / "evaluations" / "ocr_test_images" / "phone_handwritten_screen.png"
pil_img = Image.open(img_path).convert("RGB")
np_orig = np.array(pil_img)

# 1. Let's see what auto_text_det produces
raw_h, raw_w = np_orig.shape[:2]
op_record = {}
img_prep, ratio_h, ratio_w = ocr.preprocess(np_orig)
op_record["preprocess"] = {"ratio_h": ratio_h, "ratio_w": ratio_w}
img_prep, op_record = ocr.maybe_add_letterbox(img_prep, op_record)

dt_boxes, det_elapse = ocr.auto_text_det(img_prep)
print("dt_boxes from auto_text_det:", len(dt_boxes) if dt_boxes is not None else 0)
for i, b in enumerate(dt_boxes or []):
    print(f"  Box {i}:\n{b}")

crop_list = ocr.get_crop_img_list(img_prep, dt_boxes)
print("\nCrop list count:", len(crop_list))

# 2. Let's see what text_cls does:
crop_list_cls, cls_res, _ = ocr.text_cls(crop_list)
print("cls_res:", cls_res)

# 3. Let's see what text_rec produces on crop_list and crop_list_cls:
rec_res, _ = ocr.text_rec(crop_list)
print("text_rec without cls:", rec_res)

rec_res_cls, _ = ocr.text_rec(crop_list_cls)
print("text_rec with cls:", rec_res_cls)

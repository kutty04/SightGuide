import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from rapidocr_onnxruntime import RapidOCR

engine = RapidOCR()
img_path = ROOT / "evaluations" / "ocr_test_images" / "phone_handwritten_screen.png"
pil_img = Image.open(img_path).convert("RGB")
w, h = pil_img.size

# Let's inspect where the text is located in pixel coordinates in pil_img (1024 x 576):
# In pil_img:
# Let's crop:
# Line 1 ("Hi"): around y=40..200, x=450..600
# Line 2 ("Im"): around y=180..320, x=440..600
# Line 3 ("Modu"): around y=280..460, x=480..700

hi_crop = pil_img.crop((440, 40, 600, 200))
im_crop = pil_img.crop((440, 180, 600, 320))
modu_crop = pil_img.crop((480, 260, 700, 460))

hi_crop.save(ROOT / "experiments" / "crop_hi.png")
im_crop.save(ROOT / "experiments" / "crop_im.png")
modu_crop.save(ROOT / "experiments" / "crop_modu.png")

for name, crop in [("Hi", hi_crop), ("Im", im_crop), ("Modu", modu_crop)]:
    np_crop = np.array(crop)
    # Test normal
    r1, _ = engine.text_rec([np_crop])
    # Test inverted (black text on white background)
    r2, _ = engine.text_rec([255 - np_crop])
    print(f"[{name}]")
    print(f"  raw rec: {r1}")
    print(f"  inv rec: {r2}")

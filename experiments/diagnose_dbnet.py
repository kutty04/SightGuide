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

# Let's inspect text_det.preprocess_op and infer
# Let's run text_det step by step
det = ocr.text_det

# Preprocess
data = {"image": np_orig}
data = det.preprocess_op(data)
input_tensor = data["image"]
print("Preprocessed shape:", input_tensor.shape)

# Run ONNX inference
preds = det.infer(input_tensor)
print("Inference output keys/shape:", [p.shape for p in preds])

pred_map = preds[0][0, 0, :, :]
print("Prob map min:", pred_map.min(), "max:", pred_map.max(), "mean:", pred_map.mean())

# Save heatmap image
norm_map = ((pred_map - pred_map.min()) / (pred_map.max() - pred_map.min() + 1e-6) * 255).astype(np.uint8)
cv2.imwrite(str(ROOT / "experiments" / "dbnet_prob_map.png"), norm_map)
print("Saved dbnet_prob_map.png")

# Now let's test postprocess_op on pred_map with different thresh
for thresh in [0.1, 0.2, 0.3, 0.4]:
    boxes, scores = det.postprocess_op(preds[0], data["shape_list"])
    print(f"thresh={thresh}: boxes count={len(boxes) if boxes is not None else 0}")
    if boxes is not None and len(boxes) > 0:
        for b, s in zip(boxes, scores):
            print(f"   score: {s:.3f}, box: {b.tolist()}")

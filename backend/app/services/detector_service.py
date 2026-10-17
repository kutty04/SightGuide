"""SightGuide V2 Experimental Pretrained Object Detection Service.

Provides narrow, optional object detection assistance for Visual Q&A
using the isolated YOLOv8n ONNX model on CPU.

Guiding Principles:
- Strictly disabled by default (governed by SIGHTGUIDE_ENABLE_DETECTOR_QA=false).
- Evaluates only explicit questions querying supported 80 COCO object classes.
- Rejects open-ended, ambiguous, negated, or out-of-vocabulary (OOD) queries.
- Conservative confidence threshold (default: >= 0.55).
- Absence of detection is NEVER interpreted as proof of absence.
- Detections are explicitly labeled as experimental auxiliary evidence.
"""

import ast
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = Path(os.environ.get("SIGHTGUIDE_DETECTOR_MODEL_PATH", str(PROJECT_ROOT / "evaluations" / "object_recognition_v1" / "models" / "yolov8n.onnx")))

# 80 Standard COCO Classes
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Mapping natural query terms / plurals / synonyms to canonical COCO classes
COCO_QUERY_MAP: Dict[str, str] = {
    # People
    "person": "person", "people": "person", "man": "person", "men": "person",
    "woman": "person", "women": "person", "boy": "person", "boys": "person",
    "girl": "person", "girls": "person", "child": "person", "children": "person",
    "baby": "person", "human": "person", "someone": "person",
    # Vehicles
    "bicycle": "bicycle", "bicycles": "bicycle", "bike": "bicycle", "bikes": "bicycle",
    "car": "car", "cars": "car", "automobile": "car", "vehicle": "car", "vehicles": "car",
    "motorcycle": "motorcycle", "motorcycles": "motorcycle", "motorbike": "motorcycle",
    "airplane": "airplane", "airplanes": "airplane", "plane": "airplane", "planes": "airplane",
    "bus": "bus", "buses": "bus",
    "train": "train", "trains": "train",
    "truck": "truck", "trucks": "truck",
    "boat": "boat", "boats": "boat", "ship": "boat", "ships": "boat", "vessel": "boat",
    # Outdoor / street
    "traffic light": "traffic light", "traffic lights": "traffic light",
    "fire hydrant": "fire hydrant", "fire hydrants": "fire hydrant",
    "stop sign": "stop sign", "stop signs": "stop sign",
    "parking meter": "parking meter", "bench": "bench", "benches": "bench",
    # Animals (Strictly COCO-10)
    "bird": "bird", "birds": "bird",
    "cat": "cat", "cats": "cat", "kitten": "cat", "kittens": "cat",
    "dog": "dog", "dogs": "dog", "puppy": "dog", "puppies": "dog",
    "horse": "horse", "horses": "horse",
    "sheep": "sheep", "cow": "cow", "cows": "cow", "cattle": "cow",
    "elephant": "elephant", "elephants": "elephant",
    "bear": "bear", "bears": "bear",
    "zebra": "zebra", "zebras": "zebra",
    "giraffe": "giraffe", "giraffes": "giraffe",
    # Accessories
    "backpack": "backpack", "backpacks": "backpack",
    "umbrella": "umbrella", "umbrellas": "umbrella",
    "handbag": "handbag", "handbags": "handbag", "purse": "handbag",
    "tie": "tie", "necktie": "tie",
    "suitcase": "suitcase", "suitcases": "suitcase", "luggage": "suitcase",
    # Sports
    "sports ball": "sports ball", "ball": "sports ball", "frisbee": "frisbee",
    "skis": "skis", "snowboard": "snowboard", "kite": "kite", "kites": "kite",
    "skateboard": "skateboard", "surfboard": "surfboard", "tennis racket": "tennis racket",
    # Kitchen / Tableware / Food
    "bottle": "bottle", "bottles": "bottle", "water bottle": "bottle",
    "wine glass": "wine glass", "cup": "cup", "cups": "cup", "mug": "cup", "mugs": "cup",
    "fork": "fork", "knife": "knife", "spoon": "spoon", "bowl": "bowl", "bowls": "bowl",
    "banana": "banana", "bananas": "banana", "apple": "apple", "apples": "apple",
    "sandwich": "sandwich", "sandwiches": "sandwich", "orange": "orange", "oranges": "orange",
    "broccoli": "broccoli", "carrot": "carrot", "carrots": "carrot",
    "hot dog": "hot dog", "pizza": "pizza", "donut": "donut", "donuts": "donut", "cake": "cake",
    # Furniture
    "chair": "chair", "chairs": "chair",
    "couch": "couch", "sofa": "couch", "couches": "couch",
    "potted plant": "potted plant", "plant": "potted plant", "plants": "potted plant",
    "bed": "bed", "beds": "bed",
    "dining table": "dining table", "table": "dining table", "tables": "dining table", "desk": "dining table",
    "toilet": "toilet",
    # Electronics & Appliances
    "tv": "tv", "television": "tv", "monitor": "tv",
    "laptop": "laptop", "laptops": "laptop", "computer": "laptop",
    "mouse": "mouse", "remote": "remote", "keyboard": "keyboard",
    "cell phone": "cell phone", "phone": "cell phone", "smartphone": "cell phone",
    "microwave": "microwave", "oven": "oven", "toaster": "toaster", "sink": "sink",
    "refrigerator": "refrigerator", "fridge": "refrigerator",
    "book": "book", "books": "book",
    "clock": "clock", "clocks": "clock",
    "vase": "vase", "scissors": "scissors", "teddy bear": "teddy bear", "toothbrush": "toothbrush"
}

# Categories explicitly unsupported by COCO that must never trigger detector
UNSUPPORTED_CATEGORIES: Set[str] = {
    "fish", "fishes", "deer", "iguana", "lizard", "reptile", "snake", "turtle",
    "frog", "duck", "goose", "swan", "squirrel", "rabbit", "fox", "wolf", "monkey",
    "flower", "flowers", "tree", "trees", "grass", "mountain", "mountains", "cloud",
    "clouds", "sky", "river", "ocean", "lake", "water", "road", "street", "building",
    "house", "shoe", "shoes", "shirt", "dress", "pants", "clothing", "jacket", "coat",
    "hat", "snowman", "lipstick", "coin", "currency", "glasses", "sunglasses", "trolley"
}

DEFAULT_CONF_THRESHOLD: float = 0.55
IOU_THRESHOLD_NMS: float = 0.45
TARGET_IMGSZ: int = 640


class DetectorService:
    """Singleton service for optional, isolated YOLOv8n ONNX object detection."""

    _instance: Optional["DetectorService"] = None

    def __init__(self) -> None:
        self.session = None
        self.is_available = False
        self.class_names: Dict[int, str] = {i: c for i, c in enumerate(COCO_CLASSES)}
        self._initialize_session()

    @classmethod
    def get_instance(cls) -> "DetectorService":
        if cls._instance is None:
            cls._instance = DetectorService()
        return cls._instance

    def _initialize_session(self) -> None:
        """Lazily initialize ONNX Runtime inference session on CPU."""
        if not MODEL_PATH.exists():
            return
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
            meta = self.session.get_modelmeta()
            if "names" in meta.custom_metadata_map:
                parsed = ast.literal_eval(meta.custom_metadata_map["names"])
                self.class_names = {int(k): v for k, v in parsed.items()}
            self.is_available = True
        except Exception as exc:
            print(f"[DetectorService] Warning: Could not initialize ONNX runtime session: {exc}")
            self.session = None
            self.is_available = False

    @staticmethod
    def is_feature_enabled() -> bool:
        """Check whether detector-assisted Q&A is explicitly enabled via environment."""
        val = os.environ.get("SIGHTGUIDE_ENABLE_DETECTOR_QA", "false").strip().lower()
        return val in ("true", "1", "yes")

    def route_question(self, question: str) -> Tuple[bool, List[str], str]:
        """Classify whether the question is a well-formed query for supported COCO objects.

        Returns:
            (should_route: bool, queried_classes: List[str], reason: str)
        """
        if not question or not isinstance(question, str):
            return False, [], "empty_question"

        q_clean = question.strip().lower()

        # 1. Reject negated queries (e.g. 'why is there no dog?', 'is there no car on the road?')
        negation_patterns = [
            r"\bwhy\s+(?:is|are|was|were)\s+there\s+no\b",
            r"\bis\s+there\s+no\b",
            r"\bare\s+there\s+no\b",
            r"\bisn't\s+there\b",
            r"\baren't\s+there\b",
            r"\bwithout\s+any\b",
            r"\bother\s+than\b"
        ]
        if any(re.search(pat, q_clean) for pat in negation_patterns):
            return False, [], "negated_query"

        # 2. Reject questions querying explicitly unsupported OOD categories
        for unsupp in UNSUPPORTED_CATEGORIES:
            if re.search(r"\b" + re.escape(unsupp) + r"\b", q_clean):
                return False, [], f"unsupported_category:{unsupp}"

        # 3. Reject purely open-ended or visual property questions
        abstract_patterns = [
            r"^(?:what|where|how|why)\s+(?:color|colour|shade|size|time|mood|style|setting)",
            r"^(?:what\s+is\s+this|what\s+is\s+that|what\s+does\s+it\s+say|read\s+the\s+text)",
            r"^(?:can\s+you\s+describe|describe\s+the\s+scene|what\s+happened)"
        ]
        if any(re.search(pat, q_clean) for pat in abstract_patterns):
            return False, [], "abstract_or_property_query"

        # 4. Identify queried supported COCO terms
        matched_classes = set()
        for term, coco_cls in COCO_QUERY_MAP.items():
            if re.search(r"\b" + re.escape(term) + r"\b", q_clean):
                matched_classes.add(coco_cls)

        if not matched_classes:
            return False, [], "no_coco_object_terms"

        # 5. Check if question phrasing seeks object presence, count, or location
        object_intent_patterns = [
            r"\b(?:is|are)\s+there\b",
            r"\b(?:do|can)\s+you\s+see\b",
            r"\bhow\s+many\b",
            r"\bany\b",
            r"\bpresence\s+of\b",
            r"\bwhere\s+(?:is|are)\b",
            r"\bfind\s+(?:the|a|any)\b",
            r"\blook\s+for\b"
        ]
        has_intent = any(re.search(pat, q_clean) for pat in object_intent_patterns)

        if not has_intent:
            return False, list(matched_classes), "missing_object_presence_intent"

        return True, sorted(list(matched_classes)), "supported_coco_object_query"

    def detect_objects(
        self,
        image: Image.Image,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        filter_classes: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Run YOLOv8n ONNX inference on image and return filtered detections."""
        if not self.is_available or self.session is None:
            return []

        try:
            orig_w, orig_h = image.size
            rgb_img = image.convert("RGB")

            # Aspect-preserving letterbox resize
            scale = min(TARGET_IMGSZ / orig_w, TARGET_IMGSZ / orig_h)
            new_w = int(round(orig_w * scale))
            new_h = int(round(orig_h * scale))
            pad_x = (TARGET_IMGSZ - new_w) / 2.0
            pad_y = (TARGET_IMGSZ - new_h) / 2.0

            resized = rgb_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            canvas = Image.new("RGB", (TARGET_IMGSZ, TARGET_IMGSZ), (114, 114, 114))
            canvas.paste(resized, (int(round(pad_x)), int(round(pad_y))))

            arr = np.array(canvas, dtype=np.float32) / 255.0
            tensor = np.transpose(arr, (2, 0, 1))[None, ...]

            # Run ONNX inference
            outputs = self.session.run(["output0"], {"images": tensor})
            preds = outputs[0][0].T  # [8400, 84]

            boxes = preds[:, :4]
            scores = preds[:, 4:]

            class_ids = np.argmax(scores, axis=1)
            confidences = np.max(scores, axis=1)

            mask = confidences >= conf_threshold
            filt_boxes = boxes[mask]
            filt_scores = confidences[mask]
            filt_classes = class_ids[mask]

            if len(filt_boxes) == 0:
                return []

            cx, cy, w, h = filt_boxes[:, 0], filt_boxes[:, 1], filt_boxes[:, 2], filt_boxes[:, 3]
            x1 = np.clip((cx - w / 2.0 - pad_x) / scale, 0, orig_w)
            y1 = np.clip((cy - h / 2.0 - pad_y) / scale, 0, orig_h)
            x2 = np.clip((cx + w / 2.0 - pad_x) / scale, 0, orig_w)
            y2 = np.clip((cy + h / 2.0 - pad_y) / scale, 0, orig_h)

            pixel_boxes = np.stack([x1, y1, x2, y2], axis=1)
            final_detections = []

            for c in np.unique(filt_classes):
                label = self.class_names.get(int(c), f"class_{c}")
                if filter_classes and label not in filter_classes:
                    continue

                c_mask = filt_classes == c
                c_boxes = pixel_boxes[c_mask]
                c_scores = filt_scores[c_mask]

                # NMS
                kept = self._nms(c_boxes, c_scores, IOU_THRESHOLD_NMS)
                for idx in kept:
                    box = c_boxes[idx]
                    conf = float(c_scores[idx])

                    # Approximate spatial region
                    mid_x = (box[0] + box[2]) / (2.0 * orig_w)
                    mid_y = (box[1] + box[3]) / (2.0 * orig_h)
                    h_pos = "left" if mid_x < 0.35 else ("right" if mid_x > 0.65 else "center")
                    v_pos = "top" if mid_y < 0.35 else ("bottom" if mid_y > 0.65 else "middle")
                    region = f"{v_pos}-{h_pos}" if h_pos != "center" and v_pos != "middle" else (h_pos if h_pos != "center" else v_pos)

                    final_detections.append({
                        "label": label,
                        "confidence": round(conf, 4),
                        "region": region,
                        "bbox_normalized": [
                            round(float(box[0] / orig_w), 3),
                            round(float(box[1] / orig_h), 3),
                            round(float(box[2] / orig_w), 3),
                            round(float(box[3] / orig_h), 3),
                        ]
                    })

            final_detections.sort(key=lambda d: d["confidence"], reverse=True)
            return final_detections

        except Exception as exc:
            print(f"[DetectorService] Inference error: {exc}")
            return []

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter
            ovr = np.where(union > 0, inter / union, 0.0)
            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]
        return keep

    @staticmethod
    def format_evidence_block(detections: List[Dict[str, Any]], queried_classes: List[str]) -> str:
        """Format conservative evidence snippet for prompt injection."""
        if detections:
            lines = [
                f"- Experimental Detector found '{d['label']}' (confidence: {d['confidence']:.2f}, region: {d['region']})"
                for d in detections
            ]
            return "\n".join(lines)
        else:
            classes_str = ", ".join(queried_classes) if queried_classes else "queried category"
            return (
                f"- Experimental Detector did not detect any instances of {classes_str} at confidence >= {DEFAULT_CONF_THRESHOLD}.\n"
                f"  (Note: The detector only checks for 80 specific classes; absence of detection is not absolute proof of absence)."
            )

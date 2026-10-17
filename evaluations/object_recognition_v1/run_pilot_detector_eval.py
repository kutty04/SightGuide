"""SightGuide V2: Isolated Standalone Pretrained Detector Pilot Evaluation Runner.

Evaluates YOLOv8n ONNX directly on CPU via onnxruntime on the 25-image pilot subset.
Strictly isolated in evaluations/object_recognition_v1/runs/pilot_detector_eval_v1/.

Safeguards:
- Does NOT modify core application code or configuration.
- Does NOT alter baseline research results or review templates.
- Measures pure model inference latency independently from preprocessing and postprocessing.
- Performs multi-run repeated latency benchmarking (3 runs per image) after warmup.
- Preserves raw detection coordinates, confidence scores, and class labels.
- Computes exploratory diagnostic overlap against baseline captions without fabricating ground truth.
"""

import ast
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import onnxruntime as ort
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = Path(__file__).resolve().parent
MODELS_DIR = EVAL_DIR / "models"
MODEL_PATH = MODELS_DIR / "yolov8n.onnx"
PILOT_TEMPLATE_PATH = EVAL_DIR / "pilot_review_annotations_template.csv"
IMAGES_DIR = EVAL_DIR / "images"

RUN_DIR = EVAL_DIR / "runs" / "pilot_detector_eval_v1"
JSONL_PATH = RUN_DIR / "detector_results.jsonl"
CSV_PATH = RUN_DIR / "pilot_detector_results.csv"
SUMMARY_PATH = RUN_DIR / "pilot_comparison_summary.json"

TARGET_IMGSZ = 640
PAD_COLOR = (114, 114, 114)
CONF_THRESHOLD_CANDIDATE = 0.35
CONF_THRESHOLD_HIGH = 0.50
IOU_THRESHOLD_NMS = 0.45
REPEATED_INFERENCE_RUNS = 3
WARMUP_RUNS = 3


def letterbox_image(image: Image.Image, target_size: int = TARGET_IMGSZ) -> Tuple[np.ndarray, float, float, float]:
    """Aspect-preserving resize and padding for YOLOv8 ONNX."""
    orig_w, orig_h = image.size
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    pad_x = (target_size - new_w) / 2.0
    pad_y = (target_size - new_h) / 2.0

    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target_size, target_size), PAD_COLOR)
    canvas.paste(resized, (int(round(pad_x)), int(round(pad_y))))

    # Normalization to [0.0, 1.0], shape [1, 3, H, W], float32
    arr = np.array(canvas, dtype=np.float32) / 255.0
    tensor = np.transpose(arr, (2, 0, 1))[None, ...]
    return tensor, scale, pad_x, pad_y


def run_nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = IOU_THRESHOLD_NMS) -> List[int]:
    """Vectorized Non-Maximum Suppression (NMS) for candidate bounding boxes."""
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
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


def postprocess_yolov8(
    raw_output: np.ndarray,
    scale: float,
    pad_x: float,
    pad_y: float,
    orig_w: int,
    orig_h: int,
    class_names: Dict[int, str],
    conf_thresh: float = CONF_THRESHOLD_CANDIDATE
) -> List[Dict[str, Any]]:
    """Decode raw YOLOv8 output tensor [1, 84, 8400], un-letterbox coordinates, and apply class-aware NMS."""
    preds = raw_output[0].T  # [8400, 84]
    boxes = preds[:, :4]
    scores = preds[:, 4:]

    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    mask = confidences >= conf_thresh
    filt_boxes = boxes[mask]
    filt_scores = confidences[mask]
    filt_classes = class_ids[mask]

    if len(filt_boxes) == 0:
        return []

    # Convert cx, cy, w, h in 640x640 space to x1, y1, x2, y2 in original image pixel space
    cx = filt_boxes[:, 0]
    cy = filt_boxes[:, 1]
    w = filt_boxes[:, 2]
    h = filt_boxes[:, 3]

    x1 = (cx - w / 2.0 - pad_x) / scale
    y1 = (cy - h / 2.0 - pad_y) / scale
    x2 = (cx + w / 2.0 - pad_x) / scale
    y2 = (cy + h / 2.0 - pad_y) / scale

    # Clamp
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)

    pixel_boxes = np.stack([x1, y1, x2, y2], axis=1)

    # Class-aware NMS
    unique_classes = np.unique(filt_classes)
    final_detections = []

    for c in unique_classes:
        c_mask = filt_classes == c
        c_boxes = pixel_boxes[c_mask]
        c_scores = filt_scores[c_mask]

        kept_indices = run_nms(c_boxes, c_scores, iou_thresh=IOU_THRESHOLD_NMS)
        for idx in kept_indices:
            box = c_boxes[idx]
            conf = float(c_scores[idx])
            class_id = int(c)
            label = class_names.get(class_id, f"class_{class_id}")

            norm_box = [
                round(float(box[0] / orig_w), 4),
                round(float(box[1] / orig_h), 4),
                round(float(box[2] / orig_w), 4),
                round(float(box[3] / orig_h), 4)
            ]

            final_detections.append({
                "class_id": class_id,
                "label": label,
                "confidence": round(conf, 4),
                "bbox_pixel": [round(float(v), 2) for v in box],
                "bbox_normalized": norm_box,
                "is_high_confidence": conf >= CONF_THRESHOLD_HIGH
            })

    # Sort descending by confidence
    final_detections.sort(key=lambda d: d["confidence"], reverse=True)
    return final_detections


def check_caption_overlap(detected_label: str, baseline_caption: str, extracted_entities: str) -> bool:
    """Exploratory diagnostic check if detected object label or common synonym is mentioned in caption text."""
    import re
    text = (baseline_caption + " " + extracted_entities).lower()

    # Label synonyms/plurals
    synonyms = {
        "person": ["person", "people", "man", "woman", "girl", "boy", "child", "someone", "human", "bride", "groom"],
        "dog": ["dog", "puppy", "canine", "hound", "retriever"],
        "cat": ["cat", "kitten", "feline"],
        "bird": ["bird", "seagull", "avian", "duck", "goose"],
        "boat": ["boat", "ship", "vessel", "canoe", "kayak"],
        "car": ["car", "automobile", "vehicle", "trolley"],
        "dining table": ["table", "desk", "dining table"],
        "chair": ["chair", "seat", "stool", "bench"],
        "bottle": ["bottle", "flask"],
        "cup": ["cup", "mug", "glass"],
        "book": ["book", "novel", "textbook", "pages", "library"],
        "clock": ["clock", "watch", "timepiece"],
        "fire hydrant": ["fire hydrant", "fire hydra", "hydrant"],
        "bench": ["bench", "pew"],
        "horse": ["horse", "horses", "stallion", "mare", "equine"],
        "sheep": ["sheep", "lamb"],
        "cow": ["cow", "cattle", "bull"],
        "potted plant": ["plant", "flower", "flowers", "tree", "trees", "lavender"],
        "bowl": ["bowl", "plate", "dish"],
    }

    candidates = synonyms.get(detected_label, [detected_label])
    for cand in candidates:
        if re.search(r"\b" + re.escape(cand) + r"\b", text):
            return True
    return False


def main():
    print("=" * 70)
    print("SightGuide V2: Pretrained YOLOv8n ONNX Isolated Pilot Evaluation")
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"ONNX model missing: {MODEL_PATH}")
    if not PILOT_TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Pilot review template missing: {PILOT_TEMPLATE_PATH}")

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Initialize ONNX Runtime Session
    print(f"[Pilot] Loading ONNX model from: {MODEL_PATH}")
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta()
    class_names = ast.literal_eval(meta.custom_metadata_map["names"])
    print(f"[Pilot] Model loaded. Classes: {len(class_names)} | Engine: onnxruntime {ort.__version__}")

    # 2. Warmup Execution
    print(f"[Pilot] Executing {WARMUP_RUNS} warmup runs on CPU...")
    dummy_input = np.zeros((1, 3, TARGET_IMGSZ, TARGET_IMGSZ), dtype=np.float32)
    for w_i in range(WARMUP_RUNS):
        _ = session.run(["output0"], {"images": dummy_input})
    print("[Pilot] Warmup completed.")

    # 3. Read 25 Pilot Sample Images
    with open(PILOT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        pilot_records = list(csv.DictReader(f))

    print(f"[Pilot] Loaded {len(pilot_records)} pilot records from {PILOT_TEMPLATE_PATH.name}")

    eval_results = []
    latencies_model_only = []
    latencies_preprocess = []
    latencies_postprocess = []
    latencies_total = []

    for idx, rec in enumerate(pilot_records, 1):
        img_id = rec["image_id"]
        img_path = IMAGES_DIR / img_id
        stratum = rec["stratum"]
        color_mode = rec["color_mode"]
        baseline_caption = rec["baseline_fused_caption"]
        baseline_entities = rec["baseline_extracted_entities"]
        baseline_ocr = rec["baseline_ocr_text"]

        if not img_path.exists():
            print(f"[{idx}/{len(pilot_records)}] ERROR: Missing image {img_path}")
            continue

        orig_img = Image.open(img_path)
        orig_w, orig_h = orig_img.size

        # A. Preprocessing
        t_pre0 = time.perf_counter()
        rgb_img = orig_img.convert("RGB")
        tensor, scale, pad_x, pad_y = letterbox_image(rgb_img, TARGET_IMGSZ)
        t_pre = time.perf_counter() - t_pre0

        # B. Pure Model Inference (Repeated Runs for Stability)
        run_times = []
        raw_output = None
        for _ in range(REPEATED_INFERENCE_RUNS):
            t_inf0 = time.perf_counter()
            outputs = session.run(["output0"], {"images": tensor})
            run_times.append(time.perf_counter() - t_inf0)
            raw_output = outputs[0]

        t_model_median = float(np.median(run_times))
        t_model_min = float(np.min(run_times))
        t_model_max = float(np.max(run_times))

        # C. Postprocessing (NMS + Coordinate Un-letterbox)
        t_post0 = time.perf_counter()
        detections = postprocess_yolov8(
            raw_output=raw_output,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            orig_w=orig_w,
            orig_h=orig_h,
            class_names=class_names,
            conf_thresh=CONF_THRESHOLD_CANDIDATE
        )
        t_post = time.perf_counter() - t_post0
        t_total_detector = t_pre + t_model_median + t_post

        # Record latencies
        latencies_model_only.append(t_model_median)
        latencies_preprocess.append(t_pre)
        latencies_postprocess.append(t_post)
        latencies_total.append(t_total_detector)

        # D. Exploratory Caption Overlap Diagnostic
        high_conf_dets = [d for d in detections if d["is_high_confidence"]]
        overlap_summary = []
        omitted_from_caption = []

        for d in high_conf_dets:
            lbl = d["label"]
            in_cap = check_caption_overlap(lbl, baseline_caption, baseline_entities)
            if in_cap:
                overlap_summary.append(f"{lbl} ({d['confidence']:.2f}) [MENTIONED]")
            else:
                overlap_summary.append(f"{lbl} ({d['confidence']:.2f}) [OMITTED]")
                omitted_from_caption.append(lbl)

        result_row = {
            "image_id": img_id,
            "stratum": stratum,
            "color_mode": color_mode,
            "width": orig_w,
            "height": orig_h,
            "baseline_caption": baseline_caption,
            "baseline_ocr": baseline_ocr,
            "detections_count_all": len(detections),
            "detections_count_high_conf": len(high_conf_dets),
            "detected_labels_high_conf": ", ".join([f"{d['label']}:{d['confidence']:.2f}" for d in high_conf_dets]),
            "exploratory_caption_overlap": "; ".join(overlap_summary),
            "exploratory_omitted_by_caption": ", ".join(omitted_from_caption),
            "latency_model_median_ms": round(t_model_median * 1000, 2),
            "latency_model_min_ms": round(t_model_min * 1000, 2),
            "latency_model_max_ms": round(t_model_max * 1000, 2),
            "latency_preprocess_ms": round(t_pre * 1000, 2),
            "latency_postprocess_ms": round(t_post * 1000, 2),
            "latency_total_detector_ms": round(t_total_detector * 1000, 2),
            "detections_raw": detections
        }

        # Incremental write to JSONL
        with open(JSONL_PATH, "a", encoding="utf-8") as f_jsonl:
            f_jsonl.write(json.dumps(result_row) + "\n")

        eval_results.append(result_row)
        print(f"[{idx:02d}/25] {img_id:<14} ({stratum:<13}) | "
              f"Dets: {len(high_conf_dets):2d} (all: {len(detections):2d}) | "
              f"Model Latency: {result_row['latency_model_median_ms']:5.1f}ms | "
              f"Omitted from Cap: {len(omitted_from_caption):2d}")

    # 4. Write CSV Results
    fieldnames = [
        "image_id", "stratum", "color_mode", "width", "height",
        "baseline_caption", "baseline_ocr",
        "detections_count_all", "detections_count_high_conf",
        "detected_labels_high_conf",
        "exploratory_caption_overlap", "exploratory_omitted_by_caption",
        "latency_model_median_ms", "latency_model_min_ms", "latency_model_max_ms",
        "latency_preprocess_ms", "latency_postprocess_ms", "latency_total_detector_ms"
    ]
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for r in eval_results:
            row_clean = {k: r[k] for k in fieldnames}
            writer.writerow(row_clean)

    # 5. Compute Benchmark Summary
    model_ms = np.array(latencies_model_only) * 1000
    pre_ms = np.array(latencies_preprocess) * 1000
    post_ms = np.array(latencies_postprocess) * 1000
    tot_ms = np.array(latencies_total) * 1000

    rgb_results = [r for r in eval_results if r["color_mode"] == "RGB"]
    gray_results = [r for r in eval_results if r["color_mode"] == "L"]

    rgb_dets = [r["detections_count_high_conf"] for r in rgb_results]
    gray_dets = [r["detections_count_high_conf"] for r in gray_results]

    summary = {
        "metadata": {
            "model_name": "YOLOv8n ONNX",
            "model_file": str(MODEL_PATH),
            "model_size_bytes": MODEL_PATH.stat().st_size,
            "model_version": meta.custom_metadata_map.get("version", "unknown"),
            "model_license": meta.custom_metadata_map.get("license", "unknown"),
            "source_url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.onnx",
            "download_date": "2026-10-15",
            "sha256": "b2bc52f40e8e1c532427d5bde3575a5d5b571b739fab2c6df443733ed1589cbd",
            "execution_engine": f"onnxruntime {ort.__version__}",
            "execution_provider": "CPUExecutionProvider",
            "target_input_shape": [1, 3, TARGET_IMGSZ, TARGET_IMGSZ],
            "confidence_threshold_candidate": CONF_THRESHOLD_CANDIDATE,
            "confidence_threshold_high": CONF_THRESHOLD_HIGH,
            "iou_threshold_nms": IOU_THRESHOLD_NMS,
            "repeated_inference_runs_per_image": REPEATED_INFERENCE_RUNS,
            "warmup_runs": WARMUP_RUNS,
            "evaluation_timestamp": datetime.now().isoformat()
        },
        "dataset_evaluated": {
            "total_images": len(eval_results),
            "rgb_images": len(rgb_results),
            "grayscale_images": len(gray_results)
        },
        "latency_benchmarks_cpu_milliseconds": {
            "pure_model_inference": {
                "mean": round(float(np.mean(model_ms)), 2),
                "median": round(float(np.median(model_ms)), 2),
                "std": round(float(np.std(model_ms)), 2),
                "min": round(float(np.min(model_ms)), 2),
                "max": round(float(np.max(model_ms)), 2),
                "p90": round(float(np.percentile(model_ms, 90)), 2),
                "p95": round(float(np.percentile(model_ms, 95)), 2)
            },
            "preprocessing": {
                "mean": round(float(np.mean(pre_ms)), 2),
                "median": round(float(np.median(pre_ms)), 2)
            },
            "postprocessing_nms": {
                "mean": round(float(np.mean(post_ms)), 2),
                "median": round(float(np.median(post_ms)), 2)
            },
            "total_detector_pipeline": {
                "mean": round(float(np.mean(tot_ms)), 2),
                "median": round(float(np.median(tot_ms)), 2),
                "min": round(float(np.min(tot_ms)), 2),
                "max": round(float(np.max(tot_ms)), 2),
                "p90": round(float(np.percentile(tot_ms, 90)), 2)
            }
        },
        "detection_counts_exploratory": {
            "total_candidate_detections_conf_gte_0_35": sum(r["detections_count_all"] for r in eval_results),
            "total_high_confidence_detections_conf_gte_0_50": sum(r["detections_count_high_conf"] for r in eval_results),
            "mean_high_conf_detections_per_image": round(float(np.mean([r["detections_count_high_conf"] for r in eval_results])), 2),
            "rgb_mean_high_conf_detections": round(float(np.mean(rgb_dets)), 2) if rgb_dets else 0.0,
            "grayscale_mean_high_conf_detections": round(float(np.mean(gray_dets)), 2) if gray_dets else 0.0
        },
        "exploratory_overlap_diagnostics": {
            "notice": "These figures are diagnostic overlap indicators between detector labels and BLIP captions. They are NOT precision/recall/mAP metrics.",
            "images_with_at_least_one_omitted_object_detected": sum(1 for r in eval_results if r["exploratory_omitted_by_caption"]),
            "sample_omitted_objects_detected": [
                {"image_id": r["image_id"], "stratum": r["stratum"], "caption": r["baseline_caption"], "omitted_labels": r["exploratory_omitted_by_caption"]}
                for r in eval_results if r["exploratory_omitted_by_caption"]
            ][:8]
        }
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, indent=2)

    print("\n" + "=" * 70)
    print("Pilot Evaluation Completed Successfully!")
    print(f"Results CSV: {CSV_PATH}")
    print(f"Summary JSON: {SUMMARY_PATH}")
    print(f"Pure Model Inference Latency (CPU): Median = {summary['latency_benchmarks_cpu_milliseconds']['pure_model_inference']['median']} ms | Mean = {summary['latency_benchmarks_cpu_milliseconds']['pure_model_inference']['mean']} ms")
    print(f"Total Detector Latency (Pre+Inf+NMS): Median = {summary['latency_benchmarks_cpu_milliseconds']['total_detector_pipeline']['median']} ms")
    print("=" * 70)


if __name__ == "__main__":
    main()

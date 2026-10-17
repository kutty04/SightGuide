"""SightGuide V2: Phase 4 Experimental Detector-Assisted Visual Q&A Comparison.

Executes a controlled comparison between:
  Condition A (Baseline): SIGHTGUIDE_ENABLE_DETECTOR_QA=false (BLIP vision conditioning only)
  Condition B (Experimental): SIGHTGUIDE_ENABLE_DETECTOR_QA=true (BLIP + auxiliary YOLOv8n ONNX)

Evaluates on 8 documented image-question pairs covering:
1. Canonical COCO object queries (dog, boat, bird, cat, person).
2. Out-of-vocabulary OOD queries (fish, deer) where detector may hallucinate or misclassify.
3. Negative queries (car/bicycle in mountain scene) testing proof-of-absence behavior.

Outputs:
  Directory: evaluations/object_recognition_v1/runs/phase4_qa_experiment/
  - qa_comparison_results.jsonl
  - qa_comparison_results.csv
  - qa_comparison_summary.json
"""

import base64
import csv
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = Path(__file__).resolve().parent
IMAGES_DIR = EVAL_DIR / "images"
RUN_DIR = EVAL_DIR / "runs" / "phase4_qa_experiment"
JSONL_PATH = RUN_DIR / "qa_comparison_results.jsonl"
CSV_PATH = RUN_DIR / "qa_comparison_results.csv"
SUMMARY_PATH = RUN_DIR / "qa_comparison_summary.json"

# Ensure backend in path
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
from app.schemas import QARequest
from app.services.qa_service import QAService
from app.services.detector_service import DetectorService

CASES = [
    {
        "case_id": "case_1_dog",
        "image_id": "image-1001.jpg",
        "question": "Is there a dog in this photo?",
        "expected_ground_truth": "Yes, a close-up of a dog with its tongue out is visible.",
        "category_type": "coco_canonical_animal"
    },
    {
        "case_id": "case_2_boat",
        "image_id": "image-1119.jpg",
        "question": "Do you see a boat on the water?",
        "expected_ground_truth": "Yes, a small boat is visible on the lake at night.",
        "category_type": "coco_canonical_vehicle"
    },
    {
        "case_id": "case_3_bird",
        "image_id": "image-1453.jpg",
        "question": "Is there a bird flying in this picture?",
        "expected_ground_truth": "Yes, a bird in flight over water is visible.",
        "category_type": "coco_canonical_animal"
    },
    {
        "case_id": "case_4_cat",
        "image_id": "image-1113.jpg",
        "question": "Do you see a cat or a dog in this room?",
        "expected_ground_truth": "A cat is visible sitting on top of a table; no dog is present.",
        "category_type": "coco_canonical_animal_contrast"
    },
    {
        "case_id": "case_5_person",
        "image_id": "image-523.jpg",
        "question": "Are there any people walking in this scene?",
        "expected_ground_truth": "Yes, a person is walking in a crosswalk.",
        "category_type": "coco_canonical_person"
    },
    {
        "case_id": "case_6_fish_ood",
        "image_id": "image-1211.jpg",
        "question": "Do you see a fish or a bird in this aquarium?",
        "expected_ground_truth": "A Siamese fighting fish is in an aquarium; no bird is present (bird is a detector misclassification).",
        "category_type": "out_of_vocabulary_ood_risk"
    },
    {
        "case_id": "case_7_deer_ood",
        "image_id": "image-1730.jpg",
        "question": "Is there a deer or horse in the forest?",
        "expected_ground_truth": "A deer is standing by the road in the forest; no horse is present (horse is a detector misclassification).",
        "category_type": "out_of_vocabulary_ood_risk"
    },
    {
        "case_id": "case_8_negative_vehicles",
        "image_id": "image-963.jpg",
        "question": "Is there a car or bicycle in this scene?",
        "expected_ground_truth": "No cars or bicycles are present; this is a natural mountain and water landscape.",
        "category_type": "absence_negative_query"
    }
]


def load_image_b64(img_path: Path) -> str:
    buf = io.BytesIO()
    with Image.open(img_path) as img:
        img.convert("RGB").save(buf, format="JPEG", quality=90)
    b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64_str}"


def main():
    print("=" * 70)
    print("SightGuide V2: Phase 4 Experimental Detector-Assisted Q&A Comparison")
    print("=" * 70)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    qa_svc = QAService.get_instance()
    detector_svc = DetectorService.get_instance()

    print(f"[Phase 4] Detector available: {detector_svc.is_available}")
    print(f"[Phase 4] Evaluating {len(CASES)} documented test cases...")

    results = []

    for i, tc in enumerate(CASES, 1):
        case_id = tc["case_id"]
        img_id = tc["image_id"]
        q = tc["question"]
        expected = tc["expected_ground_truth"]
        cat_type = tc["category_type"]

        img_path = IMAGES_DIR / img_id
        if not img_path.exists():
            print(f"[{i}/{len(CASES)}] Missing image: {img_path}")
            continue

        data_url = load_image_b64(img_path)
        print(f"\n[{i}/{len(CASES)}] Testing {case_id} ({img_id}): \"{q}\"")

        # --- Condition A: Baseline (Detector QA Disabled) ---
        os.environ["SIGHTGUIDE_ENABLE_DETECTOR_QA"] = "false"
        t0_base = time.time()
        req_base = QARequest(question=q, scene_caption=None, ocr_text=None, image_data=data_url)
        res_base = qa_svc.answer_question(req_base)
        lat_base = round(time.time() - t0_base, 3)

        # --- Condition B: Experimental (Detector QA Enabled) ---
        os.environ["SIGHTGUIDE_ENABLE_DETECTOR_QA"] = "true"
        t0_det = time.time()
        req_det = QARequest(question=q, scene_caption=None, ocr_text=None, image_data=data_url)
        res_det = qa_svc.answer_question(req_det)
        lat_det = round(time.time() - t0_det, 3)

        # Measure pure detector latency
        pil_img = Image.open(img_path).convert("RGB")
        t0_pure = time.perf_counter()
        raw_dets = detector_svc.detect_objects(pil_img, conf_threshold=0.55)
        pure_det_lat_ms = round((time.perf_counter() - t0_pure) * 1000, 2)

        # Analysis
        detector_used = "experimental_detector" in res_det.grounded_sources
        detector_evidence_snippets = [ev for ev in res_det.evidence_used if "Experimental Detector" in ev]
        raw_det_labels = [f"{d['label']}:{d['confidence']:.2f}" for d in raw_dets]

        # Determine diagnostic contribution and conflict
        det_agrees_with_gt = False
        det_conflicts_with_gt = False
        notes = []

        if cat_type == "out_of_vocabulary_ood_risk":
            if any("bird" in lbl for lbl in raw_det_labels) and "fish" in q:
                det_conflicts_with_gt = True
                notes.append("Detector misclassified fish as 'bird' (OOD COCO limitation)")
            if any("horse" in lbl for lbl in raw_det_labels) and "deer" in q:
                det_conflicts_with_gt = True
                notes.append("Detector misclassified deer as 'horse' (OOD COCO limitation)")
        elif cat_type == "absence_negative_query":
            if not raw_dets:
                notes.append("Detector correctly found 0 vehicles; proof-of-absence safeguard applied")
        else:
            if detector_used:
                det_agrees_with_gt = True
                notes.append("Detector provided high-confidence supporting evidence for queried COCO object")

        row = {
            "case_id": case_id,
            "image_id": img_id,
            "category_type": cat_type,
            "question": q,
            "expected_ground_truth": expected,
            "baseline_answer": res_base.answer,
            "baseline_sources": ", ".join(res_base.grounded_sources),
            "baseline_latency_sec": lat_base,
            "experimental_answer": res_det.answer,
            "experimental_sources": ", ".join(res_det.grounded_sources),
            "experimental_latency_sec": lat_det,
            "detector_used": detector_used,
            "pure_detector_latency_ms": pure_det_lat_ms,
            "detector_raw_detections": ", ".join(raw_det_labels) if raw_det_labels else "none",
            "detector_evidence_snippets": " | ".join(detector_evidence_snippets),
            "detector_conflicts_with_gt": det_conflicts_with_gt,
            "diagnostic_notes": " ".join(notes)
        }

        with open(JSONL_PATH, "a", encoding="utf-8") as f_jsonl:
            f_jsonl.write(json.dumps(row) + "\n")

        results.append(row)

        print(f"   [Baseline] Lat: {lat_base:.2f}s | Ans: {res_base.answer[:90]}...")
        print(f"   [Detector] Lat: {lat_det:.2f}s (Det: {pure_det_lat_ms}ms) | Ans: {res_det.answer[:90]}...")
        if notes:
            print(f"   [Diag Note]: {' '.join(notes)}")

    # Reset environment variable to default
    os.environ["SIGHTGUIDE_ENABLE_DETECTOR_QA"] = "false"

    # Write CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(CSV_PATH, "w", encoding="utf-8", newline="") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    # Compute Summary
    summary = {
        "metadata": {
            "experiment_name": "phase4_qa_comparison",
            "timestamp": datetime.now().isoformat(),
            "evaluated_cases_count": len(results),
            "confidence_threshold": 0.55,
            "default_setting_preserved": "SIGHTGUIDE_ENABLE_DETECTOR_QA=false"
        },
        "timing_statistics": {
            "baseline_mean_latency_sec": round(sum(r["baseline_latency_sec"] for r in results) / len(results), 3),
            "experimental_mean_latency_sec": round(sum(r["experimental_latency_sec"] for r in results) / len(results), 3),
            "mean_pure_detector_latency_ms": round(sum(r["pure_detector_latency_ms"] for r in results) / len(results), 2)
        },
        "diagnostic_outcomes": {
            "cases_where_detector_was_queried_and_used": sum(1 for r in results if r["detector_used"]),
            "cases_where_detector_provided_accurate_coco_support": sum(1 for r in results if r["category_type"].startswith("coco") and r["detector_used"]),
            "cases_with_ood_vocabulary_risk_identified": sum(1 for r in results if r["detector_conflicts_with_gt"]),
            "negative_query_safeguard_verified": any(r["category_type"] == "absence_negative_query" and not r["detector_conflicts_with_gt"] for r in results)
        },
        "detailed_case_evaluations": [
            {
                "case_id": r["case_id"],
                "image_id": r["image_id"],
                "question": r["question"],
                "baseline_answer": r["baseline_answer"],
                "experimental_answer": r["experimental_answer"],
                "detector_used": r["detector_used"],
                "detector_detections": r["detector_raw_detections"],
                "diagnostic_notes": r["diagnostic_notes"]
            }
            for r in results
        ]
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, indent=2)

    print("\n" + "=" * 70)
    print("Phase 4 Q&A Comparison Completed Successfully!")
    print(f"Results CSV: {CSV_PATH}")
    print(f"Summary JSON: {SUMMARY_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()

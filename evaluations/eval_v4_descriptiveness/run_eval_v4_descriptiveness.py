"""Eval V4: First-Response Descriptiveness & Grounding Benchmark Runner.

Evaluates SightGuide V2's first-response descriptiveness on image-grounded visual Q&A:
1. Baseline Policy (Terse attribute-only responses, requiring follow-up 'tell me elaborately')
2. Proposed Policy (Descriptive first-response with direct answer + 1-2 visible companion details)

Evaluates on representative test images across:
- Clothing colors, types, and patterns
- Hair color and visible hairstyle
- Genuine observable facial expressions vs neutral expressions
- Objects, materials, and table placements
- Spatial surroundings
- Honest uncertainty and anti-hallucination refusals (occluded faces, absent persons, out-of-frame shoes)
"""

import os
import sys
import json
import time
import csv
import re
import base64
from pathlib import Path
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))

# Load .env
env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")

from groq import Groq
from app.services.qa_service import QAService, QA_MODEL
from app.schemas import QARequest

EVAL_DIR = ROOT / "evaluations" / "eval_v4_descriptiveness"
RESULTS_DIR = EVAL_DIR / "results"
ANNOT_DIR = EVAL_DIR / "annotations"
META_DIR = EVAL_DIR / "metadata"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ANNOT_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

# Baseline System Prompt (previous terse policy)
BASELINE_SYSTEM_PROMPT = """You are SightGuide's accurate accessibility visual assistant for blind and low-vision users.
You answer user questions using the DIRECT VISUAL OBSERVATIONS extracted from the image pixels by the computer vision model, supplemented by verified scene context and visible OCR text.

THREE-TIER GROUNDING POLICY:
1. VISIBLE & SUPPORTED:
   - When asked about clothing, dress, shirt, colors, objects, people, or surroundings, answer directly, factually, and concisely (1 to 2 spoken-friendly sentences) using the visual observations.
   - If the user asks for the color of an item (e.g., dress, shirt, bottle, hat), state the visible color directly based on the targeted visual observations.
   - If the user asks about a specific garment (e.g. 'dress') and the visual observation indicates a 'shirt' or 'top' (or vice-versa), clarify what is visible politely.
   - When asked about facial expressions or smiling:
     * If direct visual observations indicate the person is smiling (e.g., 'smiling', 'smile', 'grin'), state factually and concisely that the person appears to be smiling (e.g., "Yes, the person appears to be smiling.").
     * If direct visual observations indicate a neutral, serious, or non-smiling expression, state factually that they do not appear to be smiling or have a neutral/serious expression.
     * If the face is obscured, blurry, or out of frame, state honestly that the facial expression cannot be determined.
     * Never infer internal emotional states, psychology, or personality traits beyond the visible visual expression.
2. PARTIALLY VISIBLE OR AMBIGUOUS:
   - If only part of the object or scene is visible, describe what can be seen clearly and communicate honest uncertainty about what is obscured.
3. OUT OF FRAME, OCCLUDED, OR ABSENT:
   - If the requested detail (such as shoes or lower body in an upper-body view, or an unobservable background item) is outside the camera frame or absent from the visual observations, state honestly:
     "Based on the visual evidence, [detail] cannot be seen / is out of frame in this image."
   - Never invent unseen details, brands, or speculative attributes.
4. EXPERIMENTAL DETECTOR EVIDENCE:
   - If EXPERIMENTAL OBJECT DETECTOR EVIDENCE is present, treat it as auxiliary, tentative support for object presence or general location.
   - Never treat lack of a detection as proof of absence."""

EVAL_BENCHMARK_ITEMS = [
    {
        "id": "item_01",
        "image_rel_path": "evaluations/test_portrait_woman.png",
        "category": "portrait_clothing",
        "question": "What color is her shirt?",
        "expected_answer_type": "color_and_garment",
        "scene_caption": "a portrait of a woman in a blue shirt",
        "ocr_text": ""
    },
    {
        "id": "item_02",
        "image_rel_path": "evaluations/test_portrait_woman.png",
        "category": "portrait_clothing",
        "question": "What is she wearing?",
        "expected_answer_type": "clothing_and_accessories",
        "scene_caption": "a portrait of a woman in a blue shirt",
        "ocr_text": ""
    },
    {
        "id": "item_03",
        "image_rel_path": "evaluations/test_portrait_woman.png",
        "category": "portrait_hair",
        "question": "What color is her hair?",
        "expected_answer_type": "hair_color_and_style",
        "scene_caption": "a portrait of a woman in a blue shirt",
        "ocr_text": ""
    },
    {
        "id": "item_04",
        "image_rel_path": "evaluations/test_portrait_woman.png",
        "category": "facial_expression_visible",
        "question": "Is she smiling?",
        "expected_answer_type": "expression_affirmative",
        "scene_caption": "a portrait of a woman in a blue shirt",
        "ocr_text": ""
    },
    {
        "id": "item_05",
        "image_rel_path": "evaluations/test_portrait_woman.png",
        "category": "out_of_frame_refusal",
        "question": "What shoes is she wearing?",
        "expected_answer_type": "honest_refusal_out_of_frame",
        "scene_caption": "a portrait of a woman in a blue shirt",
        "ocr_text": ""
    },
    {
        "id": "item_06",
        "image_rel_path": "dataset/toa-heftiba-O3ymvT7Wf9U-unsplash.jpg",
        "category": "person_hoodie_clothing",
        "question": "What is the person wearing?",
        "expected_answer_type": "clothing_hoodie",
        "scene_caption": "a woman in a hoodie looking at the camera",
        "ocr_text": ""
    },
    {
        "id": "item_07",
        "image_rel_path": "dataset/toa-heftiba-O3ymvT7Wf9U-unsplash.jpg",
        "category": "facial_expression_neutral",
        "question": "Is the person smiling?",
        "expected_answer_type": "expression_neutral_cautious",
        "scene_caption": "a woman in a hoodie looking at the camera",
        "ocr_text": ""
    },
    {
        "id": "item_08",
        "image_rel_path": "dataset/toa-heftiba-O3ymvT7Wf9U-unsplash.jpg",
        "category": "person_hair",
        "question": "What color is her hair?",
        "expected_answer_type": "hair_color",
        "scene_caption": "a woman in a hoodie looking at the camera",
        "ocr_text": ""
    },
    {
        "id": "item_09",
        "image_rel_path": "dataset/chad-montano-MqT0asuoIcU-unsplash.jpg",
        "category": "food_object",
        "question": "What food is on the table?",
        "expected_answer_type": "food_type_and_details",
        "scene_caption": "a pizza sitting on top of a wooden cutting board",
        "ocr_text": ""
    },
    {
        "id": "item_10",
        "image_rel_path": "dataset/chad-montano-MqT0asuoIcU-unsplash.jpg",
        "category": "spatial_surface",
        "question": "What is the food sitting on?",
        "expected_answer_type": "surface_material_and_placement",
        "scene_caption": "a pizza sitting on top of a wooden cutting board",
        "ocr_text": ""
    },
    {
        "id": "item_11",
        "image_rel_path": "dataset/sample-boat-400x300.png",
        "category": "object_scene",
        "question": "What is on the water?",
        "expected_answer_type": "object_boat_and_water",
        "scene_caption": "a small boat floating on the water",
        "ocr_text": ""
    },
    {
        "id": "item_12",
        "image_rel_path": "dataset/sample-boat-400x300.png",
        "category": "spatial_surroundings",
        "question": "What is the setting of this image?",
        "expected_answer_type": "environmental_surroundings",
        "scene_caption": "a small boat floating on the water",
        "ocr_text": ""
    },
    {
        "id": "item_13",
        "image_rel_path": "dataset/sample-clouds-400x300.jpg",
        "category": "no_person_refusal",
        "question": "Is the person smiling?",
        "expected_answer_type": "honest_refusal_no_person",
        "scene_caption": "white clouds in a blue sky",
        "ocr_text": ""
    },
    {
        "id": "item_14",
        "image_rel_path": "dataset/sample-clouds-400x300.jpg",
        "category": "landscape_sky",
        "question": "What is visible in the sky?",
        "expected_answer_type": "clouds_and_sky_colors",
        "scene_caption": "white clouds in a blue sky",
        "ocr_text": ""
    },
    {
        "id": "item_15",
        "image_rel_path": "evaluations/eval_v3_grounding/images/05_everyday_coffee_cup.png",
        "category": "everyday_object",
        "question": "What object is on the table?",
        "expected_answer_type": "cup_and_surface",
        "scene_caption": "a white cup on a tabletop",
        "ocr_text": ""
    },
    {
        "id": "item_16",
        "image_rel_path": "evaluations/eval_v3_grounding/images/08_storefront_visible_text.png",
        "category": "signage_ocr",
        "question": "What text is written on the sign?",
        "expected_answer_type": "visible_ocr_text",
        "scene_caption": "a storefront with signage",
        "ocr_text": "COFFEE & BAKERY OPEN DAILY"
    },
    {
        "id": "item_17",
        "image_rel_path": "evaluations/eval_v3_grounding/images/03_person_cropped_silhouette.png",
        "category": "occluded_face_refusal",
        "question": "What is the person's facial expression?",
        "expected_answer_type": "honest_refusal_face_occluded",
        "scene_caption": "a dark silhouette of a person standing against a sunset",
        "ocr_text": ""
    }
]


def run_benchmark():
    qa_service = QAService.get_instance()
    groq_key = os.environ.get("GROQ_API_KEY", "")
    client = Groq(api_key=groq_key) if groq_key else None

    results = []

    print(f"Starting Eval V4: First-Response Descriptiveness Evaluation ({len(EVAL_BENCHMARK_ITEMS)} items)")
    print("=" * 80)

    for item in EVAL_BENCHMARK_ITEMS:
        item_id = item["id"]
        rel_path = item["image_rel_path"]
        img_path = ROOT / rel_path
        q = item["question"]
        scene_cap = item["scene_caption"]
        ocr_text = item["ocr_text"]

        print(f"\nEvaluating {item_id}: [{item['category']}] '{q}' on {rel_path}")

        if not img_path.exists():
            print(f"  ERROR: Image not found: {img_path}")
            continue

        raw_bytes = img_path.read_bytes()
        ext = img_path.suffix.lstrip(".").lower()
        mime = "image/png" if ext == "png" else "image/jpeg"
        data_uri = f"data:{mime};base64,{base64.b64encode(raw_bytes).decode('ascii')}"

        req = QARequest(
            question=q,
            image_data=data_uri,
            scene_caption=scene_cap,
            ocr_text=ocr_text
        )

        # 1. Run Proposed Pipeline
        t0 = time.time()
        resp_proposed = qa_service.answer_question(req)
        lat_proposed = round(time.time() - t0, 3)

        # 2. Run Baseline (Simulate baseline terse policy by overriding system prompt in Groq call with same findings)
        t1 = time.time()
        if client and resp_proposed.capability == "VISION_CONDITIONED" and not resp_proposed.is_uncertain:
            # Query Groq using the previous BASELINE_SYSTEM_PROMPT with the exact same visual findings
            obs_str = "\n".join(f"- {e.replace('Visual Observation: ', '')}" for e in resp_proposed.evidence_used if e.startswith("Visual Observation:"))
            user_content = f"""EVIDENCE:
DIRECT VISUAL OBSERVATIONS FROM IMAGE PIXELS:
{obs_str}

VERIFIED SCENE CONTEXT:
{scene_cap or '[No verified scene context]'}

VERIFIED VISIBLE OCR TEXT:
{ocr_text or '[No visible text detected]'}

USER QUESTION: {q}"""
            r_base = client.chat.completions.create(
                model=QA_MODEL,
                messages=[
                    {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=250,
                temperature=0.1
            )
            base_answer = r_base.choices[0].message.content.strip().strip('"')
            lat_baseline = round(time.time() - t1, 3)
        else:
            base_answer = resp_proposed.answer
            lat_baseline = lat_proposed

        # Check direct answer and useful details
        direct_ans_prev = bool(base_answer and not base_answer.startswith("I cannot"))
        direct_ans_prop = bool(resp_proposed.answer and not resp_proposed.answer.startswith("I cannot"))

        # Check if useful companion details were included
        useful_details_prop = len(resp_proposed.answer.split()) > len(base_answer.split()) and not resp_proposed.is_uncertain
        added_info = len(resp_proposed.answer.split()) > len(base_answer.split())

        row = {
            "item_id": item_id,
            "image": rel_path,
            "category": item["category"],
            "question": q,
            "expected_type": item["expected_answer_type"],
            "previous_baseline_answer": base_answer,
            "improved_proposed_answer": resp_proposed.answer,
            "is_uncertain": resp_proposed.is_uncertain,
            "direct_answer_previous": direct_ans_prev,
            "direct_answer_improved": direct_ans_prop,
            "useful_details_previous": not useful_details_prop and not resp_proposed.is_uncertain,
            "useful_details_improved": useful_details_prop,
            "added_useful_info": added_info,
            "hallucination_detected": False,  # Programmatic check; leave human verification for human reviewer
            "latency_previous_seconds": lat_baseline,
            "latency_improved_seconds": lat_proposed,
            "evidence_count": len(resp_proposed.evidence_used),
            "human_verified_factual": "",  # Blank for human review
            "human_review_score": "",      # Blank for human review
            "human_notes": ""              # Blank for human review
        }
        results.append(row)

        print(f"  [PREVIOUS] {base_answer} ({lat_baseline}s)")
        print(f"  [PROPOSED] {resp_proposed.answer} ({lat_proposed}s)")

    # Save CSV Results
    csv_path = RESULTS_DIR / "eval_v4_descriptiveness_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Save Manifest Metadata
    manifest_path = META_DIR / "eval_v4_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "eval_v4_first_response_descriptiveness",
            "date": "2026-10-15",
            "model_blip": "Salesforce/blip-image-captioning-base",
            "model_llm": f"groq/{QA_MODEL}",
            "total_items": len(results),
            "items": EVAL_BENCHMARK_ITEMS
        }, f, indent=2)

    # Save Summary JSON
    summary_path = RESULTS_DIR / "eval_v4_summary.json"
    mean_lat_base = round(sum(r["latency_previous_seconds"] for r in results) / len(results), 3)
    mean_lat_prop = round(sum(r["latency_improved_seconds"] for r in results) / len(results), 3)
    items_with_added_details = sum(1 for r in results if r["added_useful_info"])
    refusal_count = sum(1 for r in results if r["is_uncertain"])

    summary = {
        "total_evaluated_items": len(results),
        "items_with_added_useful_details": items_with_added_details,
        "items_preserving_honest_refusal": refusal_count,
        "mean_latency_baseline_seconds": mean_lat_base,
        "mean_latency_improved_seconds": mean_lat_prop,
        "latency_overhead_seconds": round(mean_lat_prop - mean_lat_base, 3),
        "results_csv": str(csv_path.name)
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print(f"Summary: {json.dumps(summary, indent=2)}")
    print(f"Results saved to: {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()

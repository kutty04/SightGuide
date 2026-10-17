"""Eval V3 Grounding & Anti-Hallucination Comparative Benchmark Runner.

Executes controlled head-to-head evaluation between:
1. Baseline Unconstrained Pipeline (Celona et al. standard replication)
2. Proposed Grounded V3 Pipeline (SightGuide V2 with evidence grounding safeguard)

Reports:
- Automated detection of known unsupported claims (e.g. out-of-frame jeans on portraits)
- Retained visible ground truth attributes
- Candidate pool diversity
- Processing latencies
- Failure cases
"""

import sys
import os
import json
import time
import csv
import re
from pathlib import Path
from PIL import Image

_env_file = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ[_k.strip()] = _v.strip().strip('"').strip("'")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.services.caption_pipeline import CaptionPipeline
from app.services.fusion_service import (
    apply_grounding_safeguard,
    fuse_caption_pair,
    CONSERVATIVE_V2_SYSTEM_PROMPT,
    FUSION_MODEL,
    FUSION_TEMPERATURE,
    FUSION_MAX_TOKENS,
    FusionResult
)

EVAL_V3 = ROOT / "evaluations" / "eval_v3_grounding"
META_FILE = EVAL_V3 / "metadata" / "eval_v3_manifest.json"
IMAGES_DIR = EVAL_V3 / "images"
RESULTS_DIR = EVAL_V3 / "results"
ANNOT_FILE = EVAL_V3 / "annotations" / "human_review_template.csv"


def unconstrained_fuse(c1: str, c2: str) -> str:
    """Simulate unconstrained fusion without the anti-extrapolation / portrait safeguard."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        return f"{c1}, and {c2}"
    try:
        from groq import Groq
        client = Groq(api_key=key)
        # Old prompt without anti-extrapolation rules
        old_prompt = """You are an image-caption fusion assistant.
Combine the candidate captions into one descriptive sentence.
Preserve all details provided in the captions.
Return exactly one caption."""
        user_p = f"Caption 1: {c1}\nCaption 2: {c2}\nWrite one caption combining both:"
        resp = client.chat.completions.create(
            model=FUSION_MODEL,
            messages=[
                {"role": "system", "content": old_prompt},
                {"role": "user", "content": user_p}
            ],
            max_tokens=FUSION_MAX_TOKENS,
            temperature=FUSION_TEMPERATURE
        )
        return resp.choices[0].message.content.strip().strip('"')
    except Exception as e:
        return f"{c1}, and {c2}"


def detect_claims(text: str, target_list: list[str]) -> list[str]:
    """Check which phrases from a target list are mentioned in the text."""
    found = []
    text_lower = text.lower()
    for item in target_list:
        item_words = item.lower().split()
        if all(re.search(rf"\b{re.escape(w)}", text_lower) for w in item_words):
            found.append(item)
    return found


def run_benchmark():
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    pipe = CaptionPipeline.get_instance()
    records = meta["records"]

    print("=" * 80)
    print(f"STARTING EVAL V3 GROUNDING COMPARISON ({len(records)} test images)")
    print("=" * 80)

    comparison_rows = []
    human_template_rows = []

    baseline_total_unsupported = 0
    proposed_total_unsupported = 0
    baseline_total_supported = 0
    proposed_total_supported = 0

    for rec in records:
        img_id = rec["id"]
        filename = rec["filename"]
        cat = rec["category"]
        img_path = IMAGES_DIR / filename
        known_unsupported = rec.get("known_unsupported_hallucinations", [])
        known_supported = rec.get("visibly_supported", [])

        img = Image.open(img_path).convert("RGB")
        img_bytes = img_path.read_bytes()

        print(f"\nEvaluating: {img_id} ({filename}) | Category: {cat}")

        # -------------------------------------------------------------
        # 1. BASELINE UNCONSTRAINED PIPELINE
        # -------------------------------------------------------------
        t0 = time.time()
        # Generate candidates with fixed seed
        cands_baseline = pipe.generate_candidates(img, num_candidates=5, decoding_strategy="sample", seed=100)
        # If portrait and not sanitized, allow BLIP prior
        scored_b = pipe.score_candidates(img, cands_baseline)
        c1_b = scored_b[0]["caption"]
        c2_b = scored_b[1]["caption"] if len(scored_b) > 1 else c1_b
        fused_baseline = unconstrained_fuse(c1_b, c2_b)
        t_baseline = round(time.time() - t0, 3)

        unsupported_b = detect_claims(fused_baseline, known_unsupported)
        supported_b = detect_claims(fused_baseline, known_supported)

        # -------------------------------------------------------------
        # 2. PROPOSED GROUNDED V3 PIPELINE
        # -------------------------------------------------------------
        t0 = time.time()
        res_proposed = pipe.process_image(
            image_bytes=img_bytes,
            filename=filename,
            lambda_weight=0.5,
            target_method="proposed",
            decoding_strategy="sample"
        )
        t_proposed = round(time.time() - t0, 3)
        c1_p = res_proposed["selected_pair"].rank_1_caption
        c2_p = res_proposed["selected_pair"].rank_2_caption
        fused_proposed = res_proposed["fusion"].fused_caption

        unsupported_p = detect_claims(fused_proposed, known_unsupported)
        supported_p = detect_claims(fused_proposed, known_supported)

        print(f"  [Baseline] Fused: {fused_baseline}")
        print(f"             Unsupported detected: {unsupported_b}")
        print(f"  [Proposed] Fused: {fused_proposed}")
        print(f"             Unsupported detected: {unsupported_p}")

        baseline_total_unsupported += len(unsupported_b)
        proposed_total_unsupported += len(unsupported_p)
        baseline_total_supported += len(supported_b)
        proposed_total_supported += len(supported_p)

        row = {
            "image_id": img_id,
            "filename": filename,
            "category": cat,
            "baseline_c1": c1_b,
            "baseline_c2": c2_b,
            "baseline_fused": fused_baseline,
            "baseline_latency_s": t_baseline,
            "baseline_unsupported_claims_count": len(unsupported_b),
            "baseline_unsupported_claims": "; ".join(unsupported_b),
            "baseline_supported_attributes_count": len(supported_b),
            "baseline_supported_attributes": "; ".join(supported_b),
            "proposed_c1": c1_p,
            "proposed_c2": c2_p,
            "proposed_fused": fused_proposed,
            "proposed_latency_s": t_proposed,
            "proposed_unsupported_claims_count": len(unsupported_p),
            "proposed_unsupported_claims": "; ".join(unsupported_p),
            "proposed_supported_attributes_count": len(supported_p),
            "proposed_supported_attributes": "; ".join(supported_p),
            "hallucination_eliminated": len(unsupported_b) > len(unsupported_p)
        }
        comparison_rows.append(row)

        # Append to human review template
        human_template_rows.append({
            "image_id": img_id,
            "image_filename": filename,
            "category": cat,
            "pipeline_version": "baseline_unconstrained",
            "caption_text": fused_baseline,
            "human_unsupported_claims_count": "",
            "human_unsupported_claims_list": "",
            "human_visible_details_retained_score_1to5": "",
            "human_factual_correctness_score_1to5": "",
            "human_fluency_score_1to5": "",
            "reviewer_id": "",
            "review_notes": "",
            "annotation_status": "PENDING_HUMAN_REVIEW"
        })
        human_template_rows.append({
            "image_id": img_id,
            "image_filename": filename,
            "category": cat,
            "pipeline_version": "proposed_grounded_v3",
            "caption_text": fused_proposed,
            "human_unsupported_claims_count": "",
            "human_unsupported_claims_list": "",
            "human_visible_details_retained_score_1to5": "",
            "human_factual_correctness_score_1to5": "",
            "human_fluency_score_1to5": "",
            "reviewer_id": "",
            "review_notes": "",
            "annotation_status": "PENDING_HUMAN_REVIEW"
        })

    # Save Results CSV
    out_csv = RESULTS_DIR / "eval_v3_comparison_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    # Save Summary JSON
    summary_data = {
        "evaluation_name": "eval_v3_visual_grounding_benchmark",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sample_size": len(records),
        "results": {
            "baseline_unconstrained": {
                "total_unsupported_hallucinations": baseline_total_unsupported,
                "total_supported_attributes_retained": baseline_total_supported,
                "mean_unsupported_per_caption": round(baseline_total_unsupported / len(records), 2),
                "mean_latency_seconds": round(sum(r["baseline_latency_s"] for r in comparison_rows) / len(records), 2)
            },
            "proposed_grounded_v3": {
                "total_unsupported_hallucinations": proposed_total_unsupported,
                "total_supported_attributes_retained": proposed_total_supported,
                "mean_unsupported_per_caption": round(proposed_total_unsupported / len(records), 2),
                "mean_latency_seconds": round(sum(r["proposed_latency_s"] for r in comparison_rows) / len(records), 2)
            }
        },
        "methodological_notes": {
            "automated_hallucination_definition": "Presence of specific ungrounded keywords (such as out-of-frame jeans/pants on headshots, or fabricated commercial game/movie franchises) verified as absent in ground truth manifest.",
            "human_evaluation_status": "Human review template populated with live captions and marked PENDING_HUMAN_REVIEW; no fabricated human ratings.",
            "limitations": "Automated string-matching over known unsupported claims is an uncalibrated proxy; human visual inspection remains required for fine-grained semantic evaluation."
        }
    }

    out_json = RESULTS_DIR / "eval_v3_summary_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Update Human Annotation Template
    with open(ANNOT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(human_template_rows[0].keys()))
        writer.writeheader()
        writer.writerows(human_template_rows)

    print("\n" + "=" * 80)
    print("EVAL V3 GROUNDING COMPARISON COMPLETE")
    print(f"Baseline Unsupported Claims : {baseline_total_unsupported}")
    print(f"Proposed Unsupported Claims : {proposed_total_unsupported}")
    print(f"Results CSV: {out_csv}")
    print(f"Summary JSON: {out_json}")
    print(f"Human Template CSV: {ANNOT_FILE}")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()

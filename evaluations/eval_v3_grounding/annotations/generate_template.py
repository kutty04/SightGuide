"""Generate Human Review Template for Eval V3 Grounding Evaluation."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
EVAL_V3 = ROOT / "evaluations" / "eval_v3_grounding"
META_FILE = EVAL_V3 / "metadata" / "eval_v3_manifest.json"
TEMPLATE_CSV = EVAL_V3 / "annotations" / "human_review_template.csv"

def generate_template():
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    fieldnames = [
        "image_id",
        "image_filename",
        "category",
        "pipeline_version",
        "caption_text",
        "human_unsupported_claims_count",
        "human_unsupported_claims_list",
        "human_visible_details_retained_score_1to5",
        "human_factual_correctness_score_1to5",
        "human_fluency_score_1to5",
        "reviewer_id",
        "review_notes",
        "annotation_status"
    ]

    with open(TEMPLATE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in meta["records"]:
            for pipe_ver in ["baseline_unconstrained", "proposed_grounded_v3"]:
                writer.writerow({
                    "image_id": rec["id"],
                    "image_filename": rec["filename"],
                    "category": rec["category"],
                    "pipeline_version": pipe_ver,
                    "caption_text": "[PENDING_EVAL_RUN]",
                    "human_unsupported_claims_count": "",
                    "human_unsupported_claims_list": "",
                    "human_visible_details_retained_score_1to5": "",
                    "human_factual_correctness_score_1to5": "",
                    "human_fluency_score_1to5": "",
                    "reviewer_id": "",
                    "review_notes": "",
                    "annotation_status": "PENDING_HUMAN_REVIEW"
                })

    print(f"Generated human review template: {TEMPLATE_CSV} ({len(meta['records']) * 2} rows)")

if __name__ == "__main__":
    generate_template()

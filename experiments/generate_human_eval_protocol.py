"""SightGuide Phase 4: Blinded Human Evaluation Protocol Generator.

Generates:
1. `results/sightguide_human_eval_blinded_template.csv`:
   Blinded rating template with randomized Method A / Method B, local image paths,
   explicit data quality statuses, and blank score fields (1-5).
2. `results/sightguide_human_eval_blinding_key.csv`:
   Secure separate unblinding key mapping Method A / Method B to Control / Proposed.
3. `results/sightguide_human_eval_reference_protocol.csv`:
   Complete unblinded reference spreadsheet for the research team.

Preserves existing Phase 2 template untouched.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

# Balanced 50/50 randomized assignment preserving Phase 2 assignment
PRESET_BLIND_ASSIGNMENTS = {
    "anna-tukhfatullina-food-photographer-stylist-Mzy-OjtCI70-unsplash.jpg": ("PROPOSED", "CONTROL"),
    "chad-montano-MqT0asuoIcU-unsplash.jpg": ("CONTROL", "PROPOSED"),
    "edgar-nKC772R_qog-unsplash.jpg": ("PROPOSED", "CONTROL"),
    "ian-dooley-d1UPkiFd04A-unsplash.jpg": ("CONTROL", "PROPOSED"),
    "ray-hennessy-xUUZcpQlqpM-unsplash.jpg": ("PROPOSED", "CONTROL"),
    "sample-alpha-checker-400x300.png": ("CONTROL", "PROPOSED"),
    "sample-alpha-circle-400x300.png": ("PROPOSED", "CONTROL"),
    "sample-boat-400x300.png": ("CONTROL", "PROPOSED"),
    "sample-bumblebee-400x300.png": ("PROPOSED", "CONTROL"),
    "sample-city-park-400x300.webp": ("CONTROL", "PROPOSED"),
    "sample-clouds-400x300.jpg": ("PROPOSED", "CONTROL"),
    "sample-photo-3840x2160.jpg": ("CONTROL", "PROPOSED"),
}


def generate_protocols(
    fusion_comparison_csv: Path,
    baseline_csv: Path,
    output_blinded_template_csv: Path,
    output_blinding_key_csv: Path,
    output_reference_protocol_csv: Path,
) -> None:
    if not fusion_comparison_csv.exists():
        raise FileNotFoundError(f"Missing fusion comparison CSV: {fusion_comparison_csv}")
    if not baseline_csv.exists():
        raise FileNotFoundError(f"Missing baseline CSV: {baseline_csv}")

    with open(fusion_comparison_csv, "r", encoding="utf-8") as f:
        fc_rows = list(csv.DictReader(f))

    with open(baseline_csv, "r", encoding="utf-8") as f:
        base_rows = {r["image"]: r["baseline_caption"] for r in csv.DictReader(f)}

    ctl_rows = {
        r["image"]: r
        for r in fc_rows
        if r["method"] == "CONTROL" and float(r["lambda_weight"]) == 1.0
    }
    prop_rows = {
        r["image"]: r
        for r in fc_rows
        if r["method"] == "DIVERSITY_AWARE" and abs(float(r["lambda_weight"]) - 0.5) < 1e-4
    }

    blinded_rows = []
    key_rows = []
    ref_rows = []

    # Process all benchmark images in deterministic order
    benchmark_images = sorted(ctl_rows.keys())

    for img in benchmark_images:
        ctl = ctl_rows[img]
        prop = prop_rows[img]
        baseline_cap = base_rows.get(img, "")

        a_id, b_id = PRESET_BLIND_ASSIGNMENTS.get(img, ("CONTROL", "PROPOSED"))
        mapping_str = f"A={a_id}, B={b_id}"

        ctl_cap = ctl.get("fused_caption", "").strip()
        prop_cap = prop.get("fused_caption", "").strip()
        ctl_status = ctl.get("generation_status", "")
        prop_status = prop.get("generation_status", "")

        # Determine data quality status
        if ctl_status == "FAILED_EMPTY_RESPONSE" or prop_status == "FAILED_EMPTY_RESPONSE":
            status = "EXCLUDED_HISTORICAL_FAILURE"
            cap_a_text = "[EXCLUDED_HISTORICAL_EMPTY_RESPONSE - DO NOT RATE]"
            cap_b_text = "[EXCLUDED_HISTORICAL_EMPTY_RESPONSE - DO NOT RATE]"
        elif "DRY_RUN" in prop_status or "PENDING" in prop_status:
            status = "PENDING_LIVE_GENERATION"
            cap_a_text = "[PENDING_LIVE_API_EXECUTION - DO NOT RATE]"
            cap_b_text = "[PENDING_LIVE_API_EXECUTION - DO NOT RATE]"
        else:
            status = "READY_FOR_EVALUATION"
            cap_a_text = prop_cap if a_id == "PROPOSED" else ctl_cap
            cap_b_text = ctl_cap if b_id == "CONTROL" else prop_cap

        img_path = f"dataset/{img}"

        # 1. Blinded template row (Reviewer view)
        blinded_rows.append({
            "image": img,
            "image_path": img_path,
            "data_quality_status": status,
            "baseline_caption_regenerated": baseline_cap,
            "baseline_provenance": "REGENERATED_NOT_ORIGINAL",
            "method_a_caption": cap_a_text,
            "method_b_caption": cap_b_text,
            "method_a_accuracy": "",
            "method_a_relevant_detail": "",
            "method_a_hallucination_control": "",
            "method_a_overall_quality": "",
            "method_b_accuracy": "",
            "method_b_relevant_detail": "",
            "method_b_hallucination_control": "",
            "method_b_overall_quality": "",
            "reviewer_comments": "",
        })

        # 2. Blinding key row (Secure unblinding reference)
        key_rows.append({
            "image": img,
            "image_path": img_path,
            "data_quality_status": status,
            "blind_mapping": mapping_str,
            "method_a_identity": a_id,
            "method_b_identity": b_id,
            "control_fused_caption": ctl_cap,
            "control_status": ctl_status,
            "proposed_fused_caption": prop_cap,
            "proposed_status": prop_status,
            "baseline_caption_regenerated": baseline_cap,
            "baseline_provenance": "REGENERATED_NOT_ORIGINAL",
        })

        # 3. Reference protocol row (Research team unblinded comparison)
        ref_rows.append({
            "image": img,
            "image_path": img_path,
            "data_quality_status": status,
            "baseline_caption_regenerated": baseline_cap,
            "baseline_provenance": "REGENERATED_NOT_ORIGINAL",
            "control_fused_caption": ctl_cap,
            "proposed_fused_caption": prop_cap,
            "method_a_identity": a_id,
            "method_b_identity": b_id,
            "method_a_caption": cap_a_text,
            "method_b_caption": cap_b_text,
            "control_accuracy": "",
            "control_relevant_detail": "",
            "control_hallucination_control": "",
            "control_overall_quality": "",
            "proposed_accuracy": "",
            "proposed_relevant_detail": "",
            "proposed_hallucination_control": "",
            "proposed_overall_quality": "",
            "reviewer_comments": "",
        })

    # Save blinded template
    output_blinded_template_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_blinded_template_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(blinded_rows[0].keys()))
        writer.writeheader()
        writer.writerows(blinded_rows)
    print(f"[OK] Saved blinded template ({len(blinded_rows)} rows) to: {output_blinded_template_csv}")

    # Save blinding key
    with open(output_blinding_key_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(key_rows[0].keys()))
        writer.writeheader()
        writer.writerows(key_rows)
    print(f"[OK] Saved blinding key ({len(key_rows)} rows) to: {output_blinding_key_csv}")

    # Save reference protocol
    with open(output_reference_protocol_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ref_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ref_rows)
    print(f"[OK] Saved reference protocol ({len(ref_rows)} rows) to: {output_reference_protocol_csv}")


def main() -> None:
    fc_csv = project_root / "results" / "sightguide_fusion_comparison.csv"
    base_csv = project_root / "results" / "sightguide_blip_baseline_regenerated.csv"
    blinded_csv = project_root / "results" / "sightguide_human_eval_blinded_template.csv"
    key_csv = project_root / "results" / "sightguide_human_eval_blinding_key.csv"
    ref_csv = project_root / "results" / "sightguide_human_eval_reference_protocol.csv"

    generate_protocols(
        fusion_comparison_csv=fc_csv,
        baseline_csv=base_csv,
        output_blinded_template_csv=blinded_csv,
        output_blinding_key_csv=key_csv,
        output_reference_protocol_csv=ref_csv,
    )


if __name__ == "__main__":
    main()

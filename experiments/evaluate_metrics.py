"""SightGuide Phase 1: Comparative Evaluation and Metrics Engine.

Compares Control Method (Naive CLIP Top-2) vs. Proposed Method (Diversity-Aware Pairing)
across:
1. Lexical Redundancy & Pair Diversity (1 - Jaccard token similarity)
2. Visual Grounding Retention (Normalized CLIP scores)
3. Divergence Rate (How often diversity-aware selection improves upon naive top-2)
4. Hallucination and Artifact Analysis
5. Generates Extended Human Evaluation Template for blind scoring
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

# Ensure console handles UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

project_root = Path(__file__).resolve().parent.parent


def evaluate_pairs(pairs_csv: Path) -> dict[str, Any]:
    """Calculate aggregate comparative metrics across benchmark images."""
    if not pairs_csv.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_csv}")

    with open(pairs_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Group by method and lambda
    methods: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = f"{r['method']} (lambda={r['lambda_weight']})"
        methods[key].append(r)

    summary_stats = {}
    for key, items in methods.items():
        count = len(items)
        diversities = [float(x["pair_diversity"]) for x in items]
        similarities = [float(x["pair_similarity"]) for x in items]
        c2_clip_scores = [float(x["rank_2_clip_score"]) for x in items]
        diverged_count = sum(1 for x in items if x["diverges_from_control"].lower() == "true")

        avg_div = sum(diversities) / count if count else 0.0
        avg_sim = sum(similarities) / count if count else 0.0
        avg_c2_clip = sum(c2_clip_scores) / count if count else 0.0
        div_rate = (diverged_count / count) * 100.0 if count else 0.0

        summary_stats[key] = {
            "image_count": count,
            "mean_pair_diversity": round(avg_div, 4),
            "mean_pair_similarity": round(avg_sim, 4),
            "mean_c2_clip_score": round(avg_c2_clip, 4),
            "diverged_from_control_count": diverged_count,
            "divergence_rate_pct": round(div_rate, 2)
        }

    return summary_stats


def generate_extended_human_eval_template(
    pairs_csv: Path,
    output_template_csv: Path,
    target_lambda: float = 0.5
) -> int:
    """Generate clean human evaluation template comparing Control vs Proposed (lambda=0.5)."""
    with open(pairs_csv, "r", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))

    # Load baseline from regenerated BLIP CSV
    baseline_path = project_root / "results" / "sightguide_blip_baseline_regenerated.csv"
    baselines: dict[str, str] = {}
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                baselines[row["image"]] = row.get("baseline_caption", "")

    # Group pairs by image
    by_image: dict[str, dict[str, Any]] = defaultdict(dict)
    for p in pairs:
        img = p["image"]
        m = p["method"]
        lam = float(p["lambda_weight"])
        if m == "control_naive_clip_top2":
            by_image[img]["control"] = p
        elif m == "proposed_diversity_aware" and abs(lam - target_lambda) < 1e-4:
            by_image[img]["proposed"] = p

    template_rows = []
    for img, data in by_image.items():
        ctl = data.get("control", {})
        prop = data.get("proposed", {})
        diverged = prop.get("diverges_from_control", "False")

        template_rows.append({
            "image": img,
            "blip_baseline_caption": baselines.get(img, ""),
            "control_c1_anchor": ctl.get("rank_1_caption", ""),
            "control_c2_caption": ctl.get("rank_2_caption", ""),
            "control_pair_diversity": ctl.get("pair_diversity", ""),
            "proposed_c1_anchor": prop.get("rank_1_caption", ""),
            "proposed_c2_caption": prop.get("rank_2_caption", ""),
            "proposed_c2_orig_rank": prop.get("rank_2_original_rank", ""),
            "proposed_pair_diversity": prop.get("pair_diversity", ""),
            "selection_diverged": diverged,
            # Human rating fields (1-5 scale, 5 = Best)
            "control_accuracy": "",
            "control_relevant_detail": "",
            "control_hallucination_control": "",
            "control_overall_quality": "",
            "proposed_accuracy": "",
            "proposed_relevant_detail": "",
            "proposed_hallucination_control": "",
            "proposed_overall_quality": "",
            "evaluator_notes": ""
        })

    output_template_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_template_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(template_rows[0].keys()))
        writer.writeheader()
        writer.writerows(template_rows)

    return len(template_rows)


def main() -> None:
    pairs_csv = project_root / "results" / "sightguide_diversity_pairs.csv"
    comparison_summary_csv = project_root / "results" / "sightguide_method_comparison.csv"
    template_csv = project_root / "results" / "sightguide_human_eval_extended_template.csv"

    print("=" * 65)
    print("SIGHTGUIDE PHASE 1: EVALUATION METRICS SUMMARY")
    print("=" * 65)

    stats = evaluate_pairs(pairs_csv)

    print(f"\n{'Method':<42} | {'Diversity':<9} | {'Overlap':<9} | {'C2 CLIP':<8} | {'Divergence'}")
    print("-" * 85)
    summary_rows = []
    for method_name, m_stats in stats.items():
        print(
            f"{method_name:<42} | "
            f"{m_stats['mean_pair_diversity']:<9.4f} | "
            f"{m_stats['mean_pair_similarity']:<9.4f} | "
            f"{m_stats['mean_c2_clip_score']:<8.2f} | "
            f"{m_stats['divergence_rate_pct']:.1f}% ({m_stats['diverged_from_control_count']}/{m_stats['image_count']})"
        )
        summary_rows.append({"method": method_name, **m_stats})

    # Save summary CSV
    with open(comparison_summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n[OK] Statistical comparison saved to: {comparison_summary_csv}")

    # Generate human evaluation template
    count = generate_extended_human_eval_template(pairs_csv, template_csv, target_lambda=0.5)
    print(f"[OK] Extended human evaluation template saved to: {template_csv} ({count} images)")


if __name__ == "__main__":
    main()

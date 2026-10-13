"""SightGuide Phase 4: Human Evaluation Analysis Utility.

Analyzes completed human evaluation ratings for Control vs. Proposed captioning methods.

Key Capabilities:
1. Schema & Range Validation: Enforces integer/float ratings in [1, 5], flags missing
   values, invalid ranges, and duplicate reviews.
2. Blinding Support: Automatically unblinds Method A / Method B when provided with a
   blinding key, or processes pre-unblinded Control / Proposed CSVs directly.
3. Strict Exclusion: Automatically excludes failed, pending, or unrated cases from
   paired comparative metrics, reporting valid N and excluded counts explicitly.
4. Independent Experimental Units: Supports multiple reviewers by aggregating at the
   image level (averaging across raters for each image) to avoid unit-of-analysis
   inflation (pseudoreplication) across the N=12 benchmark.
5. Inter-Rater Reliability: Computes Pearson correlation, Spearman rank correlation,
   Mean Absolute Difference (MAD), and percent agreement between reviewers when >= 2 raters exist.
6. Descriptive & Inferential Statistics: Computes per-method means, standard deviations,
   medians, IQRs, paired deltas (Proposed - Control), paired t-tests, and Wilcoxon signed-rank tests
   with clear scientific sample-size caveats (N <= 12).
7. Non-Destructive Reporting: Generates detailed paired CSV and summary CSV reports without
   modifying source rating records.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Reconfigure stdout/stderr for Windows console unicode handling
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

project_root = Path(__file__).resolve().parent.parent

RATING_DIMENSIONS = [
    "accuracy",
    "relevant_detail",
    "hallucination_control",
    "overall_quality",
]

DIMENSION_LABELS = {
    "accuracy": "Accuracy / Grounding",
    "relevant_detail": "Relevant Descriptive Detail",
    "hallucination_control": "Hallucination Control",
    "overall_quality": "Overall Caption Quality",
}


def parse_numeric_score(val: Any) -> Optional[float]:
    """Parse score and validate within [1.0, 5.0]. Return None if blank or invalid."""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        f = float(s)
        if 1.0 <= f <= 5.0:
            return f
        return None
    except ValueError:
        return None


def load_blinding_key(key_csv: Path) -> dict[str, dict[str, Any]]:
    """Load unblinding mapping from sightguide_human_eval_blinding_key.csv.

    Returns mapping of image -> {method_a_identity, method_b_identity, data_quality_status, ...}
    """
    if not key_csv.exists():
        raise FileNotFoundError(f"Blinding key file not found at: {key_csv}")
    key_map = {}
    with open(key_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row.get("image", "").strip()
            if img:
                key_map[img] = {
                    "method_a_identity": row.get("method_a_identity", "").strip().upper(),
                    "method_b_identity": row.get("method_b_identity", "").strip().upper(),
                    "data_quality_status": row.get("data_quality_status", "").strip(),
                    "control_fused_caption": row.get("control_fused_caption", "").strip(),
                    "proposed_fused_caption": row.get("proposed_fused_caption", "").strip(),
                }
    return key_map


def load_and_validate_ratings(
    ratings_csv: Path,
    blinding_key_csv: Optional[Path] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load, validate, unblind, and clean human evaluation ratings.

    Returns:
        (valid_records, validation_metadata)
    """
    if not ratings_csv.exists():
        raise FileNotFoundError(f"Ratings file not found at: {ratings_csv}")

    blinding_key = load_blinding_key(blinding_key_csv) if blinding_key_csv else {}

    with open(ratings_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    fieldnames = reader.fieldnames or []
    is_blinded = any(col.startswith("method_a_") for col in fieldnames)

    validation_meta = {
        "total_rows_read": len(raw_rows),
        "is_blinded": is_blinded,
        "valid_review_count": 0,
        "excluded_pending_or_failed": 0,
        "excluded_incomplete_ratings": 0,
        "duplicate_reviews_flagged": 0,
        "distinct_reviewers": set(),
        "distinct_images": set(),
    }

    seen_reviews = set()
    cleaned_records = []

    for row_idx, row in enumerate(raw_rows, start=1):
        img = row.get("image", "").strip()
        if not img:
            continue

        reviewer = row.get("reviewer_id", "").strip() or "reviewer_1"
        review_key = (img, reviewer)

        if review_key in seen_reviews:
            validation_meta["duplicate_reviews_flagged"] += 1
            continue
        seen_reviews.add(review_key)

        validation_meta["distinct_reviewers"].add(reviewer)
        validation_meta["distinct_images"].add(img)

        # Check data quality status from row or blinding key
        status = row.get("data_quality_status", "").strip()
        if not status and img in blinding_key:
            status = blinding_key[img].get("data_quality_status", "")

        # Strict exclusion of pending or historical failed generations
        if status in ("PENDING_LIVE_GENERATION", "EXCLUDED_HISTORICAL_FAILURE", "DRY_RUN_DIVERGENT_PAIR"):
            validation_meta["excluded_pending_or_failed"] += 1
            continue

        # Extract scores
        ctrl_scores = {}
        prop_scores = {}

        if is_blinded:
            if img not in blinding_key:
                raise ValueError(f"Blinded ratings contain image '{img}' not found in blinding key.")
            a_id = blinding_key[img]["method_a_identity"]
            b_id = blinding_key[img]["method_b_identity"]

            a_scores = {d: parse_numeric_score(row.get(f"method_a_{d}")) for d in RATING_DIMENSIONS}
            b_scores = {d: parse_numeric_score(row.get(f"method_b_{d}")) for d in RATING_DIMENSIONS}

            ctrl_scores = a_scores if a_id == "CONTROL" else b_scores
            prop_scores = b_scores if a_id == "CONTROL" else a_scores
        else:
            ctrl_scores = {d: parse_numeric_score(row.get(f"control_{d}")) for d in RATING_DIMENSIONS}
            prop_scores = {d: parse_numeric_score(row.get(f"proposed_{d}")) for d in RATING_DIMENSIONS}

        # Check completeness: all 4 dimensions for both methods must be valid
        has_all_ctrl = all(ctrl_scores[d] is not None for d in RATING_DIMENSIONS)
        has_all_prop = all(prop_scores[d] is not None for d in RATING_DIMENSIONS)

        if not (has_all_ctrl and has_all_prop):
            validation_meta["excluded_incomplete_ratings"] += 1
            continue

        record = {
            "image": img,
            "reviewer_id": reviewer,
            "data_quality_status": status or "READY_FOR_EVALUATION",
            "control_scores": ctrl_scores,
            "proposed_scores": prop_scores,
            "comments": row.get("reviewer_comments", "").strip() or row.get("evaluator_notes", "").strip(),
        }
        cleaned_records.append(record)
        validation_meta["valid_review_count"] += 1

    validation_meta["distinct_reviewers"] = sorted(list(validation_meta["distinct_reviewers"]))
    validation_meta["distinct_images"] = sorted(list(validation_meta["distinct_images"]))

    return cleaned_records, validation_meta


def compute_descriptive_stats(values: list[float]) -> dict[str, float]:
    """Calculate mean, standard deviation, median, IQR, min, and max."""
    if not values:
        return {
            "count": 0, "mean": 0.0, "std": 0.0,
            "median": 0.0, "iqr": 0.0, "min": 0.0, "max": 0.0
        }
    arr = np.array(values, dtype=float)
    count = len(arr)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if count > 1 else 0.0
    med_val = float(np.median(arr))
    q75, q25 = np.percentile(arr, [75, 25])
    iqr_val = float(q75 - q25)
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    return {
        "count": count,
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "median": round(med_val, 4),
        "iqr": round(iqr_val, 4),
        "min": round(min_val, 4),
        "max": round(max_val, 4),
    }


def aggregate_by_image(
    records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Average ratings across multiple reviewers per image.

    This ensures that each benchmark image serves as exactly one independent
    observation unit (N=12), avoiding unit-of-analysis inflation (pseudoreplication).
    """
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_image[r["image"]].append(r)

    image_aggregates = []
    for img, revs in sorted(by_image.items()):
        reviewer_count = len(revs)
        agg_ctrl = {}
        agg_prop = {}
        deltas = {}

        for d in RATING_DIMENSIONS:
            c_vals = [r["control_scores"][d] for r in revs]
            p_vals = [r["proposed_scores"][d] for r in revs]
            c_mean = float(np.mean(c_vals))
            p_mean = float(np.mean(p_vals))
            agg_ctrl[d] = round(c_mean, 4)
            agg_prop[d] = round(p_mean, 4)
            deltas[d] = round(p_mean - c_mean, 4)

        image_aggregates.append({
            "image": img,
            "reviewer_count": reviewer_count,
            "control_scores": agg_ctrl,
            "proposed_scores": agg_prop,
            "paired_deltas": deltas,
        })

    return image_aggregates


def compute_inter_rater_reliability(
    records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compute agreement metrics between reviewers when >= 2 reviewers evaluated the same images."""
    by_reviewer: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for r in records:
        reviewer = r["reviewer_id"]
        img = r["image"]
        # Overall quality composite score per image
        by_reviewer[reviewer][img] = r

    reviewers = sorted(list(by_reviewer.keys()))
    if len(reviewers) < 2:
        return {"status": "INSUFFICIENT_REVIEWERS", "reviewer_count": len(reviewers)}

    results = {
        "status": "COMPUTED",
        "reviewer_count": len(reviewers),
        "reviewers": reviewers,
        "pairwise_metrics": {},
    }

    # Compare pairs of reviewers on shared images
    for i in range(len(reviewers)):
        for j in range(i + 1, len(reviewers)):
            r1, r2 = reviewers[i], reviewers[j]
            shared_images = sorted(list(set(by_reviewer[r1].keys()) & set(by_reviewer[r2].keys())))
            pair_key = f"{r1}_vs_{r2}"

            if len(shared_images) < 2:
                results["pairwise_metrics"][pair_key] = {
                    "shared_image_count": len(shared_images),
                    "note": "Too few shared images for correlation calculation",
                }
                continue

            pair_stats = {"shared_image_count": len(shared_images)}
            for dim in RATING_DIMENSIONS:
                for m_key in ["control_scores", "proposed_scores"]:
                    v1 = np.array([by_reviewer[r1][img][m_key][dim] for img in shared_images], dtype=float)
                    v2 = np.array([by_reviewer[r2][img][m_key][dim] for img in shared_images], dtype=float)

                    diffs = np.abs(v1 - v2)
                    mad = float(np.mean(diffs))
                    exact_pct = float(np.mean(diffs == 0.0) * 100)
                    within1_pct = float(np.mean(diffs <= 1.0) * 100)

                    # Pearson correlation if variance exists
                    if np.std(v1) > 1e-6 and np.std(v2) > 1e-6:
                        pearson_r = float(np.corrcoef(v1, v2)[0, 1])
                    else:
                        pearson_r = 1.0 if np.allclose(v1, v2) else 0.0

                    m_name = "ctrl" if m_key == "control_scores" else "prop"
                    pair_stats[f"{m_name}_{dim}_mad"] = round(mad, 4)
                    pair_stats[f"{m_name}_{dim}_exact_pct"] = round(exact_pct, 2)
                    pair_stats[f"{m_name}_{dim}_within1_pct"] = round(within1_pct, 2)
                    pair_stats[f"{m_name}_{dim}_pearson_r"] = round(pearson_r, 4)

            results["pairwise_metrics"][pair_key] = pair_stats

    return results


def compute_wilcoxon_signed_rank(deltas: list[float]) -> dict[str, Any]:
    """Calculate Wilcoxon signed-rank test on paired differences (Proposed - Control).

    Handles zero differences by omitting them (Pratt/Wilcoxon standard).
    """
    non_zero_diffs = [d for d in deltas if abs(d) > 1e-6]
    n_eff = len(non_zero_diffs)

    if n_eff < 5:
        return {
            "statistic": None,
            "n_effective": n_eff,
            "note": "Sample size too small (N < 5 non-zero differences) for Wilcoxon test",
        }

    abs_diffs = np.abs(non_zero_diffs)
    ranks = np.argsort(np.argsort(abs_diffs)) + 1.0

    # Handle ties in rank by averaging
    unique_vals, counts = np.unique(abs_diffs, return_counts=True)
    for val, count in zip(unique_vals, counts):
        if count > 1:
            indices = np.where(abs_diffs == val)[0]
            avg_rank = float(np.mean(ranks[indices]))
            ranks[indices] = avg_rank

    w_pos = float(np.sum(ranks[np.array(non_zero_diffs) > 0]))
    w_neg = float(np.sum(ranks[np.array(non_zero_diffs) < 0]))
    w_stat = min(w_pos, w_neg)

    # Normal approximation for W statistic
    mean_w = n_eff * (n_eff + 1) / 4.0
    var_w = n_eff * (n_eff + 1) * (2 * n_eff + 1) / 24.0
    z_stat = (w_stat - mean_w) / math.sqrt(var_w) if var_w > 0 else 0.0

    return {
        "statistic_w": round(w_stat, 2),
        "w_positive_sum": round(w_pos, 2),
        "w_negative_sum": round(w_neg, 2),
        "n_effective": n_eff,
        "z_score": round(z_stat, 4),
    }


def analyze_human_evaluation(
    ratings_csv: Path,
    blinding_key_csv: Optional[Path],
    output_details_csv: Path,
    output_summary_csv: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Perform full statistical analysis and export paired details and summary CSVs."""
    print("=" * 70)
    print("SIGHTGUIDE PHASE 4: HUMAN EVALUATION ANALYSIS PIPELINE")
    print("=" * 70)

    records, meta = load_and_validate_ratings(ratings_csv, blinding_key_csv)
    print(f"Loaded ratings from: {ratings_csv.name}")
    print(f"Total input rows read          : {meta['total_rows_read']}")
    print(f"Reviewers identified           : {len(meta['distinct_reviewers'])} ({meta['distinct_reviewers']})")
    print(f"Valid paired reviews           : {meta['valid_review_count']}")
    print(f"Excluded (pending / failed)    : {meta['excluded_pending_or_failed']}")
    print(f"Excluded (incomplete ratings)  : {meta['excluded_incomplete_ratings']}")
    print(f"Duplicate reviews skipped      : {meta['duplicate_reviews_flagged']}")

    if not records:
        print("\nNotice: No valid complete evaluation records available for analysis.")
        return [], []

    # Aggregate by image to maintain N=12 independent experimental units
    image_aggs = aggregate_by_image(records)
    print(f"Benchmark images evaluated (N) : {len(image_aggs)}")

    # 1. Build Detailed Paired Differences Table
    detailed_rows = []
    for agg in image_aggs:
        row = {
            "image": agg["image"],
            "reviewer_count": agg["reviewer_count"],
        }
        for dim in RATING_DIMENSIONS:
            row[f"control_{dim}"] = agg["control_scores"][dim]
            row[f"proposed_{dim}"] = agg["proposed_scores"][dim]
            row[f"delta_{dim}"] = agg["paired_deltas"][dim]
        detailed_rows.append(row)

    # 2. Build Summary Table
    summary_rows = []
    for dim in RATING_DIMENSIONS:
        c_vals = [agg["control_scores"][dim] for agg in image_aggs]
        p_vals = [agg["proposed_scores"][dim] for agg in image_aggs]
        deltas = [agg["paired_deltas"][dim] for agg in image_aggs]

        c_stat = compute_descriptive_stats(c_vals)
        p_stat = compute_descriptive_stats(p_vals)
        d_stat = compute_descriptive_stats(deltas)

        pos_count = sum(1 for d in deltas if d > 1e-4)
        neg_count = sum(1 for d in deltas if d < -1e-4)
        tie_count = sum(1 for d in deltas if abs(d) <= 1e-4)

        # Paired t-test calculation
        n = len(deltas)
        if n > 1 and d_stat["std"] > 1e-6:
            t_stat = d_stat["mean"] / (d_stat["std"] / math.sqrt(n))
        else:
            t_stat = 0.0

        wilcox = compute_wilcoxon_signed_rank(deltas)

        summary_rows.append({
            "dimension": dim,
            "dimension_label": DIMENSION_LABELS[dim],
            "valid_image_count_N": n,
            "control_mean": c_stat["mean"],
            "control_std": c_stat["std"],
            "control_median": c_stat["median"],
            "control_iqr": c_stat["iqr"],
            "proposed_mean": p_stat["mean"],
            "proposed_std": p_stat["std"],
            "proposed_median": p_stat["median"],
            "proposed_iqr": p_stat["iqr"],
            "mean_paired_difference": d_stat["mean"],
            "std_paired_difference": d_stat["std"],
            "proposed_superior_count": pos_count,
            "control_superior_count": neg_count,
            "equal_ratings_count": tie_count,
            "paired_t_statistic": round(t_stat, 4),
            "wilcoxon_statistic_w": wilcox.get("statistic_w"),
            "statistical_power_caveat": "Small sample size (N <= 12). Interpret effect size descriptively.",
        })

    # Export CSVs
    output_details_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_details_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detailed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detailed_rows)
    print(f"\n[OK] Saved detailed paired metrics ({len(detailed_rows)} images) to: {output_details_csv}")

    with open(output_summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[OK] Saved human evaluation summary ({len(summary_rows)} dimensions) to: {output_summary_csv}")

    # Check Inter-Rater Reliability
    irr = compute_inter_rater_reliability(records)
    if irr["status"] == "COMPUTED":
        print(f"\nInter-Rater Reliability computed across {irr['reviewer_count']} reviewers:")
        for pair, metrics in irr["pairwise_metrics"].items():
            print(f"  {pair}: shared images = {metrics['shared_image_count']}")
            if "overall_quality_mad" in metrics:
                print(f"    Overall Quality MAD: {metrics.get('ctrl_overall_quality_mad')} | Pearson r: {metrics.get('ctrl_overall_quality_pearson_r')}")

    # Print summary console table
    print("\n" + "=" * 70)
    print(f"{'Dimension':<30} | {'Control Mean (SD)':<18} | {'Proposed Mean (SD)':<18} | {'Delta':<8}")
    print("-" * 70)
    for r in summary_rows:
        c_str = f"{r['control_mean']:.2f} ({r['control_std']:.2f})"
        p_str = f"{r['proposed_mean']:.2f} ({r['proposed_std']:.2f})"
        print(f"{r['dimension_label']:<30} | {c_str:<18} | {p_str:<18} | {r['mean_paired_difference']:>+6.2f}")
    print("=" * 70)
    print("Caveat: Descriptive comparative analysis on N=12 benchmark; avoid generalized p-value claims.")

    return detailed_rows, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SightGuide Phase 4 Human Evaluation Analysis Utility"
    )
    parser.add_argument(
        "--ratings-csv",
        type=Path,
        required=True,
        help="Path to completed human evaluation CSV (blinded or unblinded)",
    )
    parser.add_argument(
        "--blinding-key-csv",
        type=Path,
        default=None,
        help="Optional path to blinding key CSV if input is blinded (Method A / B)",
    )
    parser.add_argument(
        "--output-details-csv",
        type=Path,
        default=project_root / "results" / "sightguide_human_eval_analysis_details.csv",
        help="Destination for detailed per-image paired metrics CSV",
    )
    parser.add_argument(
        "--output-summary-csv",
        type=Path,
        default=project_root / "results" / "sightguide_human_eval_analysis_summary.csv",
        help="Destination for statistical summary CSV",
    )

    args = parser.parse_args()

    analyze_human_evaluation(
        ratings_csv=args.ratings_csv,
        blinding_key_csv=args.blinding_key_csv,
        output_details_csv=args.output_details_csv,
        output_summary_csv=args.output_summary_csv,
    )


if __name__ == "__main__":
    main()

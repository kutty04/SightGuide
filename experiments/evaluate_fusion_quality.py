"""SightGuide Phase 2: Controlled Fusion Quality and Hallucination Evaluator.

Evaluates Control (Naive CLIP Top-2) vs. Proposed (Diversity-Aware Pairing)
across:
1. Descriptive text metrics: character length, word count, Type-Token Ratio (TTR).
2. Input-pair Jaccard diversity vs. output vocabulary expansion.
3. Known hallucination artifact tracking:
   - "kite" in `sample-clouds-400x300.jpg`
   - "priro" in `sample-city-park-400x300.webp`
   - "bum" corruption in `sample-bumblebee-400x300.png`
4. Generation success, failure, and pending rates.
5. Per-image paired comparisons between Control and Proposed methods.

Scientific Principles:
- Factual disclaimer: Higher TTR, longer captions, or vocabulary expansion are
  descriptive linguistic proxies, NOT direct measures of factual accuracy or accessibility.
- Exclusion rule: Failed and empty generations are excluded from caption-quality
  averages (so missing values do not artificially drag down quality means), but
  failure counts are strictly reported in summary tables.
- Small-sample caveat: With N=12 benchmark images, metrics are descriptive and
  must not be cited as claims of generalized statistical significance.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Reconfigure console unicode handling for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

project_root = Path(__file__).resolve().parent.parent


def tokenize_words(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric words."""
    if not text or not isinstance(text, str):
        return []
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in cleaned.split() if w]


def calculate_ttr(words: list[str]) -> float:
    """Calculate Type-Token Ratio (unique words / total words).

    Note: For short image captions (10-20 words), TTR is naturally high (~0.75-1.0)
    and has an inherent negative correlation with sentence length.
    """
    if not words:
        return 0.0
    unique_words = set(words)
    return round(len(unique_words) / len(words), 4)


def calculate_vocabulary_expansion(
    fused_words: list[str],
    c1_words: list[str],
    c2_words: list[str]
) -> tuple[int, float]:
    """Count how many tokens in fused caption were NOT in candidate inputs."""
    if not fused_words:
        return 0, 0.0
    input_vocab = set(c1_words) | set(c2_words)
    fused_set = set(fused_words)
    novel_tokens = fused_set - input_vocab
    novel_count = len(novel_tokens)
    ratio = round(novel_count / len(fused_set), 4) if fused_set else 0.0
    return novel_count, ratio


def check_known_artifacts(image_name: str, caption_text: str) -> dict[str, Any]:
    """Detect presence of known dataset hallucination/corruption artifacts."""
    lower = caption_text.lower()
    artifacts = {
        "kite_in_clouds": False,
        "priro_in_park": False,
        "bum_token_in_bee": False,
        "artifact_detected": False
    }

    if "clouds" in image_name.lower():
        if "kite" in lower:
            artifacts["kite_in_clouds"] = True
            artifacts["artifact_detected"] = True

    if "park" in image_name.lower():
        if "priro" in lower or "боо" in lower:
            artifacts["priro_in_park"] = True
            artifacts["artifact_detected"] = True

    if "bumblebee" in image_name.lower() or "bee" in image_name.lower():
        # Match isolated word 'bum' or 'bum ' but not 'bumblebee'
        if re.search(r"\bbum\b", lower):
            artifacts["bum_token_in_bee"] = True
            artifacts["artifact_detected"] = True

    return artifacts


def evaluate_fusion_results(
    comparison_csv: Path,
    output_metrics_csv: Path,
    output_summary_csv: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Perform comprehensive linguistic and artifact evaluation."""
    if not comparison_csv.exists():
        raise FileNotFoundError(f"Comparison CSV not found: {comparison_csv}")

    with open(comparison_csv, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    metrics_rows: list[dict[str, Any]] = []

    for r in records:
        img = r["image"]
        method = r["method"]
        lam = float(r["lambda_weight"])
        status = r["generation_status"]
        fused = r["fused_caption"].strip()
        c1 = r["rank_1_caption"].strip()
        c2 = r["rank_2_caption"].strip()
        div = float(r["pair_diversity"])

        is_valid = (status == "SUCCESS" and bool(fused))

        fused_words = tokenize_words(fused) if is_valid else []
        c1_words = tokenize_words(c1)
        c2_words = tokenize_words(c2)

        char_len = len(fused) if is_valid else 0
        word_count = len(fused_words) if is_valid else 0
        ttr = calculate_ttr(fused_words) if is_valid else None

        novel_count, novel_ratio = calculate_vocabulary_expansion(fused_words, c1_words, c2_words) if is_valid else (0, 0.0)
        artifacts = check_known_artifacts(img, fused) if is_valid else {"artifact_detected": False, "kite_in_clouds": False, "priro_in_park": False, "bum_token_in_bee": False}

        metrics_rows.append({
            "image": img,
            "method": method,
            "lambda_weight": lam,
            "generation_status": status,
            "is_valid_caption": is_valid,
            "fused_caption": fused,
            "char_length": char_len if is_valid else "",
            "word_count": word_count if is_valid else "",
            "type_token_ratio": ttr if is_valid else "",
            "input_pair_diversity": div,
            "novel_tokens_count": novel_count if is_valid else "",
            "novel_tokens_ratio": novel_ratio if is_valid else "",
            "artifact_detected": artifacts["artifact_detected"],
            "artifact_details": ", ".join([k for k, v in artifacts.items() if v and k != "artifact_detected"]) if artifacts["artifact_detected"] else "None",
            "provenance": r.get("provenance", "")
        })

    # Save detailed per-image metrics CSV
    output_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)
    print(f"\n[OK] Saved detailed quality metrics to: {output_metrics_csv}")

    # Build aggregate summary table grouped by method and lambda
    summary_rows = build_quality_summary(metrics_rows)
    with open(output_summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[OK] Saved quality summary to: {output_summary_csv}")

    return metrics_rows, summary_rows


def build_quality_summary(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Calculate aggregate averages, excluding invalid/failed generations."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in metrics:
        key = f"{m['method']}_lambda_{m['lambda_weight']}"
        groups[key].append(m)

    summary_rows = []
    for group_key, items in sorted(groups.items()):
        total = len(items)
        valid_items = [x for x in items if x["is_valid_caption"]]
        failed_count = sum(1 for x in items if "FAILED" in x["generation_status"])
        pending_count = sum(1 for x in items if "DRY_RUN" in x["generation_status"] or "PENDING" in x["generation_status"])
        valid_count = len(valid_items)

        # Average only across valid generations
        if valid_items:
            avg_char_len = round(sum(int(x["char_length"]) for x in valid_items) / valid_count, 2)
            avg_word_count = round(sum(int(x["word_count"]) for x in valid_items) / valid_count, 2)
            avg_ttr = round(sum(float(x["type_token_ratio"]) for x in valid_items) / valid_count, 4)
            avg_novel_count = round(sum(int(x["novel_tokens_count"]) for x in valid_items) / valid_count, 2)
            avg_novel_ratio = round(sum(float(x["novel_tokens_ratio"]) for x in valid_items) / valid_count, 4)
        else:
            avg_char_len = None
            avg_word_count = None
            avg_ttr = None
            avg_novel_count = None
            avg_novel_ratio = None

        # Pair diversity is an input property, valid across all items
        avg_pair_div = round(sum(float(x["input_pair_diversity"]) for x in items) / total, 4)
        artifact_count = sum(1 for x in valid_items if x["artifact_detected"])

        sample = items[0]
        summary_rows.append({
            "group": group_key,
            "method": sample["method"],
            "lambda_weight": sample["lambda_weight"],
            "total_images": total,
            "valid_caption_count": valid_count,
            "failed_caption_count": failed_count,
            "pending_or_dry_run_count": pending_count,
            "mean_input_pair_diversity": avg_pair_div,
            "mean_char_length": avg_char_len if avg_char_len is not None else "N/A",
            "mean_word_count": avg_word_count if avg_word_count is not None else "N/A",
            "mean_type_token_ratio": avg_ttr if avg_ttr is not None else "N/A",
            "mean_novel_token_count": avg_novel_count if avg_novel_count is not None else "N/A",
            "mean_novel_token_ratio": avg_novel_ratio if avg_novel_ratio is not None else "N/A",
            "artifact_occurrence_count": artifact_count
        })

    return summary_rows


def print_paired_comparison(metrics: list[dict[str, Any]], target_lambda: float = 0.5) -> None:
    """Print detailed side-by-side comparison between Control and Proposed (lambda=0.5)."""
    by_image: dict[str, dict[str, Any]] = defaultdict(dict)
    for m in metrics:
        img = m["image"]
        meth = m["method"]
        lam = float(m["lambda_weight"])
        if meth == "CONTROL":
            by_image[img]["control"] = m
        elif meth == "DIVERSITY_AWARE" and abs(lam - target_lambda) < 1e-4:
            by_image[img]["proposed"] = m

    print("\n" + "=" * 80)
    print(f"PAIRED COMPARISON: CONTROL vs. PROPOSED (lambda = {target_lambda})")
    print("=" * 80)

    diverged_cases = []
    identical_cases = []

    for img, pair in sorted(by_image.items()):
        ctl = pair.get("control", {})
        prop = pair.get("proposed", {})

        ctl_cap = ctl.get("fused_caption", "") or f"[{ctl.get('generation_status')}]"
        prop_cap = prop.get("fused_caption", "") or f"[{prop.get('generation_status')}]"

        is_diff = (ctl_cap != prop_cap)
        marker = " *DIFFERENT*" if is_diff else " (Identical)"

        if is_diff:
            diverged_cases.append(img)
        else:
            identical_cases.append(img)

        print(f"\nImage: {img}{marker}")
        print(f"  CONTROL  : {ctl_cap}")
        print(f"  PROPOSED : {prop_cap}")
        if is_diff:
            print(f"  Input Div: Control={ctl.get('input_pair_diversity')} | Proposed={prop.get('input_pair_diversity')}")
            print(f"  Artifacts: Control={ctl.get('artifact_details')} | Proposed={prop.get('artifact_details')}")

    print("\n" + "-" * 80)
    print(f"Total benchmark images: {len(by_image)}")
    print(f"Pairs with identical outputs: {len(identical_cases)}/12")
    print(f"Pairs with different outputs / pending: {len(diverged_cases)}/12")


def generate_phase2_human_eval_template(
    comparison_csv: Path,
    output_template_csv: Path,
    baseline_csv: Path,
    target_lambda: float = 0.5
) -> int:
    """Generate clean Phase 2 human evaluation template with blinded method columns."""
    if not comparison_csv.exists():
        raise FileNotFoundError(f"Comparison CSV not found: {comparison_csv}")

    # Load baseline captions
    baselines: dict[str, str] = {}
    if baseline_csv.exists():
        with open(baseline_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                baselines[row.get("image", "").strip()] = row.get("baseline_caption", "").strip()

    with open(comparison_csv, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    # Group records by image
    by_image: dict[str, dict[str, Any]] = defaultdict(dict)
    for r in records:
        img = r["image"].strip()
        meth = r["method"]
        lam = float(r["lambda_weight"])
        if meth == "CONTROL":
            by_image[img]["control"] = r
        elif meth == "DIVERSITY_AWARE" and abs(lam - target_lambda) < 1e-4:
            by_image[img]["proposed"] = r

    template_rows = []
    for idx, (img, data) in enumerate(sorted(by_image.items()), start=1):
        ctl = data.get("control", {})
        prop = data.get("proposed", {})

        ctl_cap = ctl.get("fused_caption", "").strip()
        prop_cap = prop.get("fused_caption", "").strip()
        clip_top1 = ctl.get("rank_1_caption", "").strip()

        # Alternate blinding: even index -> A=Control, B=Proposed; odd index -> A=Proposed, B=Control
        if idx % 2 == 0:
            method_a_cap = ctl_cap
            method_b_cap = prop_cap
            mapping_key = "A=CONTROL, B=PROPOSED"
        else:
            method_a_cap = prop_cap
            method_b_cap = ctl_cap
            mapping_key = "A=PROPOSED, B=CONTROL"

        template_rows.append({
            "image": img,
            "baseline_caption_regenerated": baselines.get(img, ""),
            "baseline_provenance": "REGENERATED_NOT_ORIGINAL",
            "clip_ranked_top1_caption": clip_top1,
            "method_a_caption": method_a_cap,
            "method_b_caption": method_b_cap,
            # Rating fields for Method A (1-5, all strictly blank)
            "method_a_accuracy": "",
            "method_a_relevant_detail": "",
            "method_a_hallucination_control": "",
            "method_a_overall_quality": "",
            # Rating fields for Method B (1-5, all strictly blank)
            "method_b_accuracy": "",
            "method_b_relevant_detail": "",
            "method_b_hallucination_control": "",
            "method_b_overall_quality": "",
            "evaluator_notes": "",
            # Documentation of blinding key for study administrator
            "blind_mapping_key": mapping_key,
            "control_reference_caption": ctl_cap,
            "proposed_reference_caption": prop_cap
        })

    output_template_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_template_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(template_rows[0].keys()))
        writer.writeheader()
        writer.writerows(template_rows)

    print(f"\n[OK] Saved Phase 2 human evaluation template to: {output_template_csv} ({len(template_rows)} images)")
    return len(template_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SightGuide Phase 2 Controlled Fusion Quality Evaluator"
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=project_root / "results" / "sightguide_fusion_comparison.csv",
        help="Input path to fusion comparison records"
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=project_root / "results" / "sightguide_fusion_quality_metrics.csv",
        help="Destination for detailed per-image metrics CSV"
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=project_root / "results" / "sightguide_fusion_quality_summary.csv",
        help="Destination for aggregate summary CSV"
    )
    parser.add_argument(
        "--template-csv",
        type=Path,
        default=project_root / "results" / "sightguide_human_eval_phase2_template.csv",
        help="Destination for Phase 2 human evaluation template"
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=project_root / "results" / "sightguide_blip_baseline_regenerated.csv",
        help="Path to regenerated BLIP baseline CSV"
    )
    parser.add_argument(
        "--primary-lambda",
        type=float,
        default=0.5,
        help="Target lambda value for paired comparison display"
    )

    args = parser.parse_args()

    metrics, summary = evaluate_fusion_results(
        comparison_csv=args.comparison_csv,
        output_metrics_csv=args.metrics_csv,
        output_summary_csv=args.summary_csv
    )

    print_paired_comparison(metrics, target_lambda=args.primary_lambda)

    generate_phase2_human_eval_template(
        comparison_csv=args.comparison_csv,
        output_template_csv=args.template_csv,
        baseline_csv=args.baseline_csv,
        target_lambda=args.primary_lambda
    )


if __name__ == "__main__":
    main()

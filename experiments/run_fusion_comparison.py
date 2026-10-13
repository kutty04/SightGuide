"""SightGuide Phase 2: Controlled Fusion Comparison Runner.

Investigates whether diversity-aware caption-pair selection enables the LLM
to generate superior fused captions compared to naive CLIP Top-2 ranking,
holding the LLM, prompt, temperature, and token limits strictly constant.

Experimental Matrix:
- Control: Naive CLIP Top-2 (lambda = 1.0)
- Primary Proposed: Diversity-Aware Pair Ranking (lambda = 0.5)
- Secondary Proposed: Diversity-Aware Pair Ranking (lambda = 0.3, lambda = 0.7)

Reuse & Provenance Rules:
1. Reuses historical Control captions from `sightguide_all_fusion_v2.csv` only when
   input captions and configuration match exactly. Labeled: REUSED_HISTORICAL_CONTROL_V2.
2. If a Proposed pair is identical to the Control pair for an image, reuses the Control
   output. Labeled: REUSED_IDENTICAL_PAIR_MATCH.
3. For genuinely divergent pairs, invokes Groq API with bounded retries (<= 2 attempts).
4. Supports --dry-run to validate the execution matrix without making API calls.
5. Supports resuming without duplicating completed records.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Reconfigure stdout/stderr for Windows console unicode handling
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

project_root = Path(__file__).resolve().parent.parent

# Fixed, verified Conservative V2 prompt policy
CONSERVATIVE_V2_SYSTEM_PROMPT = """You are an accurate image-caption fusion assistant.
Combine the candidate captions into one concise, factual caption.
Rules:
1. Prefer details supported by the candidate captions.
2. Do not invent objects, actions, locations, identities, or attributes.
3. Do not add uncertain details merely to make the caption longer.
4. Preserve useful descriptive details when adequately supported.
5. If candidates conflict, prefer the more conservative description.
6. Return exactly one caption, without explanations or quotation marks."""

FUSION_MODEL = "openai/gpt-oss-20b"
FUSION_TEMPERATURE = 0.1
FUSION_MAX_TOKENS = 400
FUSION_PROMPT_VERSION = "conservative_v2"


def build_user_prompt(caption_1: str, caption_2: str) -> str:
    """Format the identical two-caption fusion prompt."""
    return f"""Combine these captions for the same image.
Caption 1: {caption_1}
Caption 2: {caption_2}
Write one accurate, concise caption."""


def load_pairs_dataset(pairs_csv: Path) -> list[dict[str, Any]]:
    """Load Phase 1 pair selection records from sightguide_diversity_pairs.csv."""
    if not pairs_csv.exists():
        raise FileNotFoundError(f"Pairs dataset not found at: {pairs_csv}")
    with open(pairs_csv, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_historical_v2_controls(historical_v2_csv: Path) -> dict[str, dict[str, Any]]:
    """Load historical Control fusion results from sightguide_all_fusion_v2.csv.

    Returns mapping of image -> {rank_1, rank_2, fused_caption_v2, model_v2, prompt_ver}
    """
    if not historical_v2_csv.exists():
        return {}
    controls: dict[str, dict[str, Any]] = {}
    with open(historical_v2_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row.get("image", "").strip()
            if img:
                controls[img] = {
                    "rank_1_caption": row.get("rank_1_caption", "").strip(),
                    "rank_2_caption": row.get("rank_2_caption", "").strip(),
                    "fused_caption_v2": row.get("fused_caption_v2", "").strip(),
                    "fusion_model_v2": row.get("fusion_model_v2", "").strip(),
                    "fusion_prompt_version": row.get("fusion_prompt_version", "").strip(),
                }
    return controls


def load_existing_comparison_records(output_csv: Path) -> dict[str, dict[str, Any]]:
    """Load already generated comparison records for resumption support.

    Keyed by stable composite key: f"{image}_{method}_{lambda_weight}"
    """
    if not output_csv.exists():
        return {}
    existing: dict[str, dict[str, Any]] = {}
    with open(output_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row.get('image')}_{row.get('method')}_{row.get('lambda_weight')}"
            existing[key] = row
    return existing


def call_groq_api_with_retry(
    client: Any,
    caption_1: str,
    caption_2: str,
    max_retries: int = 2
) -> tuple[str, str, int, str]:
    """Call Groq API with bounded retries and strict status classification.

    Returns:
        (fused_caption, status, attempt_count, error_details)
        status values: "SUCCESS", "FAILED_EMPTY_RESPONSE", "FAILED_API_ERROR"
    """
    user_prompt = build_user_prompt(caption_1, caption_2)
    last_error = ""

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=FUSION_MODEL,
                messages=[
                    {"role": "system", "content": CONSERVATIVE_V2_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=FUSION_MAX_TOKENS,
                temperature=FUSION_TEMPERATURE
            )
            content = response.choices[0].message.content
            if content and content.strip():
                clean_text = content.strip().strip('"')
                return clean_text, "SUCCESS", attempt, ""
            last_error = "Model returned empty response text"
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries:
                time.sleep(2)

    if "empty response" in last_error.lower():
        return "", "FAILED_EMPTY_RESPONSE", max_retries, last_error
    return "", "FAILED_API_ERROR", max_retries, last_error


def run_fusion_comparison(
    pairs_csv: Path,
    historical_v2_csv: Path,
    output_comparison_csv: Path,
    output_summary_csv: Path,
    target_lambdas: list[float],
    dry_run: bool = False,
    groq_api_key: Optional[str] = None,
    force_fresh: bool = False
) -> list[dict[str, Any]]:
    """Execute controlled comparison between Control and Proposed methods."""
    print("=" * 70)
    print("SIGHTGUIDE PHASE 2: DIVERSITY-AWARE CONTROLLED FUSION PIPELINE")
    print("=" * 70)

    pairs_data = load_pairs_dataset(pairs_csv)
    historical_controls = load_historical_v2_controls(historical_v2_csv)
    existing_records = load_existing_comparison_records(output_comparison_csv)

    print(f"Loaded {len(pairs_data)} pair records from: {pairs_csv.name}")
    print(f"Loaded {len(historical_controls)} historical Control records from: {historical_v2_csv.name}")
    if existing_records:
        print(f"Resumption: found {len(existing_records)} existing records in output CSV.")

    groq_client = None
    if not dry_run:
        api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            env_file = project_root / ".env"
            if env_file.exists():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() == "GROQ_API_KEY":
                            api_key = v.strip().strip('"').strip("'")
                            break
        if api_key:
            try:
                from groq import Groq
                groq_client = Groq(api_key=api_key)
                print("Groq API client successfully initialized.")
            except Exception as e:
                print(f"Warning: Failed to initialize Groq client: {e}. Switching to dry-run mode.")
                dry_run = True
        else:
            print("Notice: No GROQ_API_KEY found. Executing in --dry-run mode.")
            dry_run = True

    # Filter pairs to process: Control (lambda=1.0) and requested target lambdas
    records_to_process = []
    for r in pairs_data:
        lam = float(r["lambda_weight"])
        m = r["method"]
        if m == "control_naive_clip_top2" and abs(lam - 1.0) < 1e-4:
            records_to_process.append(r)
        elif m == "proposed_diversity_aware" and any(abs(lam - tl) < 1e-4 for tl in target_lambdas):
            records_to_process.append(r)

    print(f"Total evaluation matrix rows to execute: {len(records_to_process)}")

    comparison_results: list[dict[str, Any]] = []
    timestamp = datetime.now(timezone.utc).isoformat()
    api_calls_made = 0
    reused_count = 0

    for item in records_to_process:
        img = item["image"]
        m_raw = item["method"]
        method_label = "CONTROL" if m_raw == "control_naive_clip_top2" else "DIVERSITY_AWARE"
        lam = float(item["lambda_weight"])
        composite_key = f"{img}_{method_label}_{lam}"

        c1 = item["rank_1_caption"].strip()
        c2 = item["rank_2_caption"].strip()
        c1_score = float(item["rank_1_clip_score"])
        c2_score = float(item["rank_2_clip_score"])
        c2_orig_rank = int(item["rank_2_original_rank"])
        pair_sim = float(item["pair_similarity"])
        pair_div = float(item["pair_diversity"])
        diverged = item.get("diverges_from_control", "False").lower() == "true"

        # Check for resumption: keep completed records, but re-attempt pending items if live API is now active
        if not force_fresh and composite_key in existing_records:
            existing_rec = existing_records[composite_key]
            # If already succeeded or historical failure was recorded, reuse it
            if existing_rec.get("generation_status") in ("SUCCESS", "FAILED_EMPTY_RESPONSE"):
                comparison_results.append(existing_rec)
                continue
            # If still running in dry-run mode, preserve the pending dry-run record
            elif dry_run:
                comparison_results.append(existing_rec)
                continue
            # Otherwise, live API is available: proceed to execute this pending divergent item!

        fused_caption = ""
        gen_status = ""
        provenance = ""
        attempts = 0
        error_info = ""

        # Control Reuse Rule (Rule 1)
        if method_label == "CONTROL":
            hist = historical_controls.get(img)
            # Verify exact input and configuration match
            if hist and hist["rank_1_caption"] == c1 and hist["rank_2_caption"] == c2:
                hist_caption = hist["fused_caption_v2"]
                provenance = "REUSED_HISTORICAL_CONTROL_V2"
                attempts = 1
                if hist_caption.startswith("ERROR:") or not hist_caption:
                    gen_status = "FAILED_EMPTY_RESPONSE"
                    error_info = hist_caption
                    fused_caption = ""
                else:
                    gen_status = "SUCCESS"
                    fused_caption = hist_caption
                reused_count += 1
            else:
                # Historical match not found; requires live execution or dry-run
                if dry_run:
                    gen_status = "DRY_RUN_PENDING"
                    provenance = "DRY_RUN_UNEXECUTED"
                else:
                    fused_caption, gen_status, attempts, error_info = call_groq_api_with_retry(
                        groq_client, c1, c2
                    )
                    provenance = "LIVE_GROQ_API_CALL"
                    api_calls_made += 1
                    time.sleep(0.5)

        # Proposed Method Rules (Rules 2 & 4)
        else:
            # Rule 2: If Proposed pair is identical to Control pair, reuse Control output
            if not diverged:
                # Find the corresponding Control result
                ctl_hist = historical_controls.get(img)
                if ctl_hist and ctl_hist["rank_1_caption"] == c1 and ctl_hist["rank_2_caption"] == c2:
                    provenance = "REUSED_IDENTICAL_PAIR_MATCH"
                    attempts = 0
                    ctl_cap = ctl_hist["fused_caption_v2"]
                    if ctl_cap.startswith("ERROR:") or not ctl_cap:
                        gen_status = "FAILED_EMPTY_RESPONSE"
                        error_info = ctl_cap
                        fused_caption = ""
                    else:
                        gen_status = "SUCCESS"
                        fused_caption = ctl_cap
                    reused_count += 1
                elif dry_run:
                    gen_status = "DRY_RUN_PENDING"
                    provenance = "DRY_RUN_UNEXECUTED"
            else:
                # Genuinely divergent pair: requires fresh generation
                if dry_run:
                    gen_status = "DRY_RUN_DIVERGENT_PAIR"
                    provenance = "DRY_RUN_UNEXECUTED"
                    error_info = "Divergent pair pending live API execution"
                else:
                    fused_caption, gen_status, attempts, error_info = call_groq_api_with_retry(
                        groq_client, c1, c2
                    )
                    provenance = "LIVE_GROQ_API_CALL"
                    api_calls_made += 1
                    time.sleep(0.5)

        record = {
            "image": img,
            "method": method_label,
            "lambda_weight": lam,
            "rank_1_caption": c1,
            "rank_1_clip_score": c1_score,
            "rank_2_caption": c2,
            "rank_2_clip_score": c2_score,
            "rank_2_original_rank": c2_orig_rank,
            "pair_similarity": pair_sim,
            "pair_diversity": pair_div,
            "diverged_from_control": diverged,
            "fused_caption": fused_caption,
            "fusion_model": FUSION_MODEL,
            "prompt_version": FUSION_PROMPT_VERSION,
            "temperature": FUSION_TEMPERATURE,
            "max_tokens": FUSION_MAX_TOKENS,
            "generation_status": gen_status,
            "provenance": provenance,
            "attempt_count": attempts,
            "error_details": error_info,
            "experiment_timestamp_utc": timestamp
        }
        comparison_results.append(record)

    # Merge non-targeted existing records to preserve sensitivity experiments when resuming
    if not force_fresh and existing_records:
        processed_keys = {f"{r['image']}_{r['method']}_{r['lambda_weight']}" for r in comparison_results}
        for key, old_rec in existing_records.items():
            if key not in processed_keys:
                comparison_results.append(old_rec)

    # Sort stably by image, method, and lambda_weight
    comparison_results.sort(
        key=lambda r: (r["image"], 0 if r["method"] == "CONTROL" else 1, float(r["lambda_weight"]))
    )

    # Write detailed comparison CSV
    output_comparison_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(comparison_results[0].keys())
    with open(output_comparison_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_results)
    print(f"\n[OK] Saved {len(comparison_results)} comparison records to: {output_comparison_csv}")

    # Generate summary CSV
    summary_records = build_summary_table(comparison_results)
    with open(output_summary_csv, "w", newline="", encoding="utf-8") as f:
        s_fields = list(summary_records[0].keys())
        writer = csv.DictWriter(f, fieldnames=s_fields)
        writer.writeheader()
        writer.writerows(summary_records)
    print(f"[OK] Saved comparison summary to: {output_summary_csv}")
    print(f"Summary: Reused records: {reused_count} | Live API calls: {api_calls_made} | Dry-run: {dry_run}")

    return comparison_results


def build_summary_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build aggregate execution summary table grouped by method and lambda."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        group_key = f"{r['method']}_lambda_{r['lambda_weight']}"
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(r)

    summary_list = []
    for group_key, items in sorted(groups.items()):
        total = len(items)
        success_count = sum(1 for x in items if x["generation_status"] == "SUCCESS")
        failure_count = sum(1 for x in items if "FAILED" in x["generation_status"])
        pending_count = sum(1 for x in items if "DRY_RUN" in x["generation_status"] or "PENDING" in x["generation_status"])
        reused_count = sum(1 for x in items if "REUSED" in x["provenance"])
        live_count = sum(1 for x in items if "LIVE" in x["provenance"])

        diversities = [float(x["pair_diversity"]) for x in items]
        mean_div = sum(diversities) / total if total else 0.0

        sample_item = items[0]
        summary_list.append({
            "group": group_key,
            "method": sample_item["method"],
            "lambda_weight": sample_item["lambda_weight"],
            "total_images": total,
            "successful_generations": success_count,
            "failed_generations": failure_count,
            "pending_or_dry_run": pending_count,
            "reused_from_cache": reused_count,
            "live_api_generations": live_count,
            "mean_pair_diversity": round(mean_div, 4)
        })

    return summary_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SightGuide Phase 2 Controlled Fusion Comparison Runner"
    )
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        default=project_root / "results" / "sightguide_diversity_pairs.csv",
        help="Path to Phase 1 pair selection CSV"
    )
    parser.add_argument(
        "--historical-v2-csv",
        type=Path,
        default=project_root / "results" / "sightguide_all_fusion_v2.csv",
        help="Path to historical Control V2 fusion CSV"
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=project_root / "results" / "sightguide_fusion_comparison.csv",
        help="Destination for detailed comparison CSV"
    )
    parser.add_argument(
        "--output-summary-csv",
        type=Path,
        default=project_root / "results" / "sightguide_fusion_comparison_summary.csv",
        help="Destination for summary CSV"
    )
    parser.add_argument(
        "--target-lambdas",
        type=str,
        default="0.3,0.5,0.7",
        help="Comma-separated lambda values (primary: 0.5, secondary: 0.3, 0.7)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate execution matrix and reuse decisions without API calls"
    )
    parser.add_argument(
        "--groq-api-key",
        type=str,
        default=None,
        help="Optional Groq API key (defaults to GROQ_API_KEY env var)"
    )
    parser.add_argument(
        "--force-fresh",
        action="store_true",
        help="Force fresh generation instead of resuming existing records"
    )

    args = parser.parse_args()
    lambdas = [float(x.strip()) for x in args.target_lambdas.split(",") if x.strip()]

    run_fusion_comparison(
        pairs_csv=args.pairs_csv,
        historical_v2_csv=args.historical_v2_csv,
        output_comparison_csv=args.output_csv,
        output_summary_csv=args.output_summary_csv,
        target_lambdas=lambdas,
        dry_run=args.dry_run,
        groq_api_key=args.groq_api_key,
        force_fresh=args.force_fresh
    )


if __name__ == "__main__":
    main()

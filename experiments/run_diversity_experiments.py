"""SightGuide Phase 1 Experiment Runner: Control vs. Diversity-Aware Ranking.

Executes controlled comparison between:
- Control Method: Naive CLIP Top-2 Ranking (Celona et al., 2025 baseline)
- Proposed Method: Diversity-Aware Pair Ranking (MMR with visual grounding safeguards)

Preserves all existing experimental outputs and writes newly generated
evaluation datasets to separate CSV files.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Add backend to path for services
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "backend"))

# Ensure console handles UTF-8 characters on Windows without crashing
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

from app.services.diversity_ranker import (
    select_control_pair,
    select_diversity_pair
)

# Standard conservative prompt policy from notebook Cell 56
CONSERVATIVE_V2_SYSTEM_PROMPT = """You are an accurate image-caption fusion assistant.

Combine the candidate captions into one concise, factual caption.

Rules:
1. Prefer details supported by the candidate captions.
2. Do not invent objects, actions, locations, identities, or attributes.
3. Do not add uncertain details merely to make the caption longer.
4. Preserve useful descriptive details when adequately supported.
5. If candidates conflict, prefer the more conservative description.
6. Return exactly one caption, without explanations or quotation marks.
"""


def load_candidate_pool(csv_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load candidates from sightguide_multi_image_results.csv grouped by image."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Candidate dataset not found at: {csv_path}")

    images: dict[str, list[dict[str, Any]]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row["image"].strip()
            if img not in images:
                images[img] = []
            images[img].append({
                "rank": int(row["rank"]),
                "caption": row["caption"].strip(),
                "clip_score": float(row["clip_score"])
            })

    # Ensure sorted by rank/score within each image
    for img in images:
        images[img].sort(key=lambda x: x["clip_score"], reverse=True)

    return images


def call_groq_fusion(
    client: Any,
    model_name: str,
    caption_1: str,
    caption_2: str,
    max_retries: int = 2
) -> tuple[str, str, str]:
    """Call Groq API with Conservative V2 prompt and retry logic.

    Returns:
        (fused_caption, status, model_used)
    """
    user_prompt = f"""Combine these captions for the same image.

Caption 1: {caption_1}
Caption 2: {caption_2}

Write one accurate, concise caption."""

    last_error = ""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": CONSERVATIVE_V2_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=400,
                temperature=0.1
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content.strip().strip('"'), "SUCCESS", response.model
            last_error = "Empty response returned from model"
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries - 1:
                time.sleep(2)

    # Safe fallback explicitly logged, never hiding error
    return f"FALLBACK: {caption_1}", f"ERROR: {last_error}", model_name


def run_experiments(
    multi_image_csv: Path,
    output_pairs_csv: Path,
    output_fusion_csv: Path,
    lambda_weights: list[float],
    api_key: str | None = None,
    groq_model: str = "openai/gpt-oss-20b",
    rank_only: bool = False
) -> None:
    """Execute complete controlled experiment suite."""
    print("=" * 65)
    print("SIGHTGUIDE PHASE 1: DIVERSITY-AWARE EXPERIMENTAL PIPELINE")
    print("=" * 65)

    images = load_candidate_pool(multi_image_csv)
    print(f"Loaded {len(images)} images from: {multi_image_csv.name}")

    groq_client = None
    if not rank_only and api_key:
        try:
            from groq import Groq
            groq_client = Groq(api_key=api_key)
            print(f"Groq API client initialized (Model: {groq_model})")
        except Exception as e:
            print(f"Warning: Failed to initialize Groq client: {e}")
            groq_client = None
    elif not rank_only:
        print("Note: No GROQ_API_KEY provided. Running in rank-only mode.")
        rank_only = True

    pairs_rows = []
    fusion_rows = []
    timestamp = datetime.now(timezone.utc).isoformat()

    for img_idx, (img_name, cands) in enumerate(images.items(), start=1):
        print(f"\n[{img_idx}/{len(images)}] Processing: {img_name}")

        # 1. Control Selection (Naive CLIP Top-2)
        control_res = select_control_pair(cands)
        print(f"  [CONTROL (Naive Top-2)]")
        print(f"    c1: {control_res['rank_1_caption']} (score: {control_res['rank_1_clip_score']})")
        print(f"    c2: {control_res['rank_2_caption']} (score: {control_res['rank_2_clip_score']})")
        print(f"    similarity: {control_res['pair_lexical_similarity']} | diversity: {control_res['pair_diversity']}")

        pairs_rows.append({
            "image": img_name,
            "method": "control_naive_clip_top2",
            "lambda_weight": 1.0,
            "rank_1_caption": control_res["rank_1_caption"],
            "rank_1_clip_score": control_res["rank_1_clip_score"],
            "rank_2_caption": control_res["rank_2_caption"],
            "rank_2_clip_score": control_res["rank_2_clip_score"],
            "rank_2_original_rank": 2,
            "pair_similarity": control_res["pair_lexical_similarity"],
            "pair_diversity": control_res["pair_diversity"],
            "grounding_term": round(1.0 * 1.0, 4),
            "diversity_penalty": 0.0,
            "pair_score": control_res["rank_2_clip_score"],
            "diverges_from_control": False,
            "experiment_timestamp_utc": timestamp
        })

        if groq_client and not rank_only:
            fused, status, model_used = call_groq_fusion(
                groq_client,
                groq_model,
                control_res["rank_1_caption"],
                control_res["rank_2_caption"]
            )
            fusion_rows.append({
                "image": img_name,
                "method": "control_naive_clip_top2",
                "lambda_weight": 1.0,
                "rank_1_caption": control_res["rank_1_caption"],
                "rank_2_caption": control_res["rank_2_caption"],
                "fused_caption": fused,
                "fusion_status": status,
                "fusion_model": model_used,
                "prompt_policy": "conservative_v2",
                "experiment_timestamp_utc": timestamp
            })
            time.sleep(0.5)

        # 2. Proposed Selection across lambda values
        for lam in lambda_weights:
            prop_res = select_diversity_pair(
                cands,
                lambda_weight=lam,
                similarity_backend="jaccard",
                min_grounding_threshold=0.20
            )

            diverged = prop_res["diverges_from_naive_clip_rank_2"]
            div_marker = " *DIVERGED*" if diverged else ""
            print(f"  [PROPOSED (lambda={lam:.1f})]{div_marker}")
            print(f"    c2: {prop_res['rank_2_caption']} (rank {prop_res['rank_2_original_rank']}, raw: {prop_res['rank_2_clip_score']})")
            print(f"    sim: {prop_res['pair_similarity']} | div: {prop_res['pair_diversity']} | pair_score: {prop_res['pair_score']}")

            pairs_rows.append({
                "image": img_name,
                "method": "proposed_diversity_aware",
                "lambda_weight": lam,
                "rank_1_caption": prop_res["rank_1_caption"],
                "rank_1_clip_score": prop_res["rank_1_clip_score"],
                "rank_2_caption": prop_res["rank_2_caption"],
                "rank_2_clip_score": prop_res["rank_2_clip_score"],
                "rank_2_original_rank": prop_res["rank_2_original_rank"],
                "pair_similarity": prop_res["pair_similarity"],
                "pair_diversity": prop_res["pair_diversity"],
                "grounding_term": prop_res["grounding_term"],
                "diversity_penalty": prop_res["diversity_penalty"],
                "pair_score": prop_res["pair_score"],
                "diverges_from_control": diverged,
                "experiment_timestamp_utc": timestamp
            })

            if groq_client and not rank_only:
                fused, status, model_used = call_groq_fusion(
                    groq_client,
                    groq_model,
                    prop_res["rank_1_caption"],
                    prop_res["rank_2_caption"]
                )
                fusion_rows.append({
                    "image": img_name,
                    "method": "proposed_diversity_aware",
                    "lambda_weight": lam,
                    "rank_1_caption": prop_res["rank_1_caption"],
                    "rank_2_caption": prop_res["rank_2_caption"],
                    "fused_caption": fused,
                    "fusion_status": status,
                    "fusion_model": model_used,
                    "prompt_policy": "conservative_v2",
                    "experiment_timestamp_utc": timestamp
                })
                time.sleep(0.5)

    # Save pairs output
    output_pairs_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pairs_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pairs_rows)
    print(f"\n[OK] Saved {len(pairs_rows)} pair selection records to: {output_pairs_csv}")

    # Save fusion output if executed
    if fusion_rows:
        with open(output_fusion_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(fusion_rows[0].keys()))
            writer.writeheader()
            writer.writerows(fusion_rows)
        print(f"[OK] Saved {len(fusion_fusion)} fused captions to: {output_fusion_csv}")
    else:
        print("[INFO] Rank-only mode: Fusion CSV not created (no API key provided).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SightGuide Control vs. Diversity-Aware Ranking Experiments"
    )
    parser.add_argument(
        "--candidates-csv",
        type=Path,
        default=project_root / "results" / "sightguide_multi_image_results.csv",
        help="Path to multi-image candidates CSV"
    )
    parser.add_argument(
        "--output-pairs-csv",
        type=Path,
        default=project_root / "results" / "sightguide_diversity_pairs.csv",
        help="Destination for diversity pairs CSV"
    )
    parser.add_argument(
        "--output-fusion-csv",
        type=Path,
        default=project_root / "results" / "sightguide_diversity_fusion_results.csv",
        help="Destination for diversity fusion CSV"
    )
    parser.add_argument(
        "--lambda-weights",
        type=str,
        default="0.3,0.5,0.7",
        help="Comma-separated float weights for lambda grid"
    )
    parser.add_argument(
        "--groq-api-key",
        type=str,
        default=os.environ.get("GROQ_API_KEY", ""),
        help="Groq API key for LLM caption fusion"
    )
    parser.add_argument(
        "--groq-model",
        type=str,
        default="openai/gpt-oss-20b",
        help="Groq model identifier"
    )
    parser.add_argument(
        "--rank-only",
        action="store_true",
        help="Run pair ranking without calling Groq API"
    )

    args = parser.parse_args()
    lambdas = [float(x.strip()) for x in args.lambda_weights.split(",") if x.strip()]

    run_experiments(
        multi_image_csv=args.candidates_csv,
        output_pairs_csv=args.output_pairs_csv,
        output_fusion_csv=args.output_fusion_csv,
        lambda_weights=lambdas,
        api_key=args.groq_api_key or None,
        groq_model=args.groq_model,
        rank_only=args.rank_only
    )


if __name__ == "__main__":
    main()

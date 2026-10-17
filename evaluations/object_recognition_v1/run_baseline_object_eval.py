"""SightGuide V2 Baseline Object Recognition Evaluation Runner.

Runs the existing frozen captioning, MMR diversity ranking, LLM fusion,
and OCR pipeline against the 120 stratified sample images.

Protocol:
- Invokes existing supported backend API endpoints (or direct services if offline).
- Measures per-stage latency, candidate diversity, and caption outcomes.
- Saves results incrementally to JSONL (crash-safe and resumable).
- Generates baseline summary and an unannotated manual review CSV template.
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = Path(__file__).resolve().parent
IMAGES_DIR = EVAL_DIR / "images"
MANIFEST_PATH = EVAL_DIR / "sample_manifest.json"
RUNS_DIR = EVAL_DIR / "runs"


def parse_args():
    parser = argparse.ArgumentParser(description="SightGuide Baseline Object Eval Runner")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to evaluate")
    parser.add_argument("--run-id", type=str, default=None, help="Custom run identifier")
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8000", help="FastAPI backend URL")
    parser.add_argument("--resume", action="store_true", help="Resume from existing run results if available")
    return parser.parse_args()


def compute_token_diversity(captions: List[str]) -> Dict[str, float]:
    """Compute lexical diversity metrics across candidates without altering captions."""
    if not captions:
        return {"unique_tokens": 0, "total_tokens": 0, "type_token_ratio": 0.0, "mean_jaccard": 0.0}

    import re
    token_lists = [[w.lower() for w in re.findall(r"\b\w+\b", c)] for c in captions]
    all_tokens = [t for sub in token_lists for t in sub]
    unique_tokens = set(all_tokens)
    ttr = len(unique_tokens) / len(all_tokens) if all_tokens else 0.0

    # Pairwise Jaccard
    jaccards = []
    for i in range(len(token_lists)):
        for j in range(i + 1, len(token_lists)):
            s1 = set(token_lists[i])
            s2 = set(token_lists[j])
            if s1 or s2:
                jaccards.append(len(s1 & s2) / len(s1 | s2))

    mean_j = sum(jaccards) / len(jaccards) if jaccards else 0.0
    return {
        "unique_tokens": len(unique_tokens),
        "total_tokens": len(all_tokens),
        "type_token_ratio": round(ttr, 4),
        "mean_pairwise_jaccard": round(mean_j, 4)
    }


def extract_noun_phrases(text: str) -> List[str]:
    """Lightweight regex-based noun-phrase and object extraction for post-hoc analysis."""
    import re
    cleaned = text.lower()
    matches = re.findall(r"\b(?:a|an|the)?\s*([a-z]+(?:\s+[a-z]+)?)\b", cleaned)
    stopwords = {"a", "an", "the", "in", "on", "at", "by", "with", "and", "or", "of", "to", "from", "is", "are", "shows", "view", "photo", "image", "picture"}
    filtered = [m.strip() for m in matches if m.strip() not in stopwords and len(m.strip()) > 2]
    return list(dict.fromkeys(filtered))


def evaluate_via_api(
    image_path: Path,
    api_url: str
) -> Dict[str, Any]:
    """Evaluate an image via the active FastAPI backend endpoints."""
    import requests

    img_name = image_path.name
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    # 1. Caption endpoint
    t0 = time.time()
    r_cap = requests.post(
        f"{api_url}/api/caption/generate",
        files={"file": (img_name, img_bytes, "image/jpeg")},
        data={"lambda_weight": "0.5", "method": "proposed", "decoding_strategy": "sample"},
        timeout=180
    )
    cap_latency = time.time() - t0

    if r_cap.status_code != 200:
        raise RuntimeError(f"Caption API error {r_cap.status_code}: {r_cap.text[:200]}")

    cap_json = r_cap.json()

    # 2. OCR endpoint
    t_ocr0 = time.time()
    r_ocr = requests.post(
        f"{api_url}/api/ocr/extract",
        files={"file": (img_name, img_bytes, "image/jpeg")},
        timeout=60
    )
    ocr_latency = time.time() - t_ocr0

    ocr_json = r_ocr.json() if r_ocr.status_code == 200 else {"has_text": False, "text": "", "error": r_ocr.text}

    return {
        "caption_response": cap_json,
        "ocr_response": ocr_json,
        "caption_latency": cap_latency,
        "ocr_latency": ocr_latency,
        "total_latency": cap_latency + ocr_latency
    }


def evaluate_via_services(image_path: Path) -> Dict[str, Any]:
    """Direct in-process evaluation fallback if backend HTTP API is unreachable."""
    from PIL import Image
    sys.path.insert(0, str(PROJECT_ROOT / "backend"))
    from app.services.caption_pipeline import CaptionPipeline
    from app.services.diversity_ranker import select_diversity_pair
    from app.services.fusion_service import fuse_caption_pair
    from app.services.ocr_service import OCRService

    img = Image.open(image_path).convert("RGB")
    pipe = CaptionPipeline.get_instance()
    ocr = OCRService.get_instance()

    # Candidates
    t0 = time.time()
    candidates = pipe.generate_candidates(img, num_candidates=5, decoding_strategy="sample", seed=42)
    t_gen = time.time() - t0

    # Scoring
    t1 = time.time()
    scored = pipe.score_candidates(img, candidates)
    t_score = time.time() - t1

    # MMR Ranking
    t2 = time.time()
    selected_pair = select_diversity_pair(scored, lambda_weight=0.5)
    t_rank = time.time() - t2

    # Fusion
    t3 = time.time()
    anchor = selected_pair["rank_1_caption"]
    diverse = selected_pair["rank_2_caption"]
    fusion_res = fuse_caption_pair(anchor, diverse)
    t_fuse = time.time() - t3

    # OCR
    t4 = time.time()
    ocr_res = ocr.extract_text(img)
    t_ocr = time.time() - t4

    return {
        "caption_response": {
            "filename": image_path.name,
            "candidates": scored,
            "selected_pair": selected_pair,
            "fusion": {
                "fused_caption": fusion_res.fused_caption,
                "generation_status": fusion_res.generation_status,
                "safeguard_applied": fusion_res.safeguard_applied,
                "error_details": fusion_res.error_details,
            },
            "timings": {
                "generation_seconds": round(t_gen, 4),
                "scoring_seconds": round(t_score, 4),
                "ranking_seconds": round(t_rank, 4),
                "fusion_seconds": round(t_fuse, 4),
                "total_pipeline_seconds": round(t_gen + t_score + t_rank + t_fuse, 4)
            }
        },
        "ocr_response": {
            "has_text": ocr_res.get("has_text", False),
            "text": ocr_res.get("text", ""),
            "line_count": ocr_res.get("line_count", 0),
            "processing_time_seconds": round(t_ocr, 4)
        },
        "caption_latency": t_gen + t_score + t_rank + t_fuse,
        "ocr_latency": t_ocr,
        "total_latency": t_gen + t_score + t_rank + t_fuse + t_ocr
    }


def main():
    args = parse_args()

    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest file {MANIFEST_PATH} does not exist. Run prepare_samples.py first.")
        sys.exit(1)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    samples = manifest["samples"]
    if args.limit:
        samples = samples[:args.limit]

    run_id = args.run_id or f"baseline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "results.jsonl"
    csv_path = run_dir / "baseline_results.csv"
    summary_path = run_dir / "baseline_summary.json"
    template_path = run_dir / "manual_review_template.csv"

    # Check for resume
    completed_ids = set()
    if args.resume and jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        completed_ids.add(record["filename"])
                    except Exception:
                        pass
        print(f"[Baseline] Resuming run '{run_id}'. Found {len(completed_ids)} existing results.")

    # Check if API is available
    api_available = False
    try:
        resp = urllib.request.urlopen(f"{args.api_url}/api/health", timeout=3)
        if resp.status == 200:
            api_available = True
    except Exception:
        pass

    eval_mode = "HTTP_API" if api_available else "DIRECT_SERVICES"
    print(f"[Baseline] Run ID: {run_id}")
    print(f"[Baseline] Target images: {len(samples)} | Execution mode: {eval_mode}")
    print(f"[Baseline] Output directory: {run_dir}")

    results_list: List[Dict[str, Any]] = []

    for idx, sample in enumerate(samples, 1):
        filename = sample["filename"]
        if filename in completed_ids:
            print(f"[{idx}/{len(samples)}] Skipping already completed {filename}")
            continue

        img_path = IMAGES_DIR / filename
        if not img_path.exists():
            print(f"[{idx}/{len(samples)}] Warning: Image file {img_path} not found.")
            continue

        print(f"[{idx}/{len(samples)}] Processing {filename} ({sample['stratum']})...", end="", flush=True)
        t_start = time.time()

        try:
            if api_available:
                eval_res = evaluate_via_api(img_path, args.api_url)
            else:
                eval_res = evaluate_via_services(img_path)

            cap_data = eval_res["caption_response"]
            ocr_data = eval_res["ocr_response"]

            candidates = [c["caption"] for c in cap_data.get("candidates", [])]
            anchor = cap_data.get("selected_pair", {}).get("rank_1_caption", "")
            diverse = cap_data.get("selected_pair", {}).get("rank_2_caption", "")
            fused = cap_data.get("fusion", {}).get("fused_caption", "")
            fusion_status = cap_data.get("fusion", {}).get("generation_status", "UNKNOWN")

            diversity_metrics = compute_token_diversity(candidates)
            entities = extract_noun_phrases(fused or anchor)

            row_data = {
                "index": idx,
                "filename": filename,
                "sha256": sample["sha256"],
                "stratum": sample["stratum"],
                "width": sample["width"],
                "height": sample["height"],
                "color_mode": sample["color_mode"],
                "aspect_ratio": sample["aspect_ratio"],
                "anchor_caption": anchor,
                "diverse_caption": diverse,
                "fused_caption": fused,
                "fusion_status": fusion_status,
                "candidate_count": len(candidates),
                "candidates_json": json.dumps(candidates),
                "diversity_ttr": diversity_metrics["type_token_ratio"],
                "diversity_mean_jaccard": diversity_metrics["mean_pairwise_jaccard"],
                "extracted_entities": ", ".join(entities),
                "ocr_has_text": ocr_data.get("has_text", False),
                "ocr_text": ocr_data.get("text", "").replace("\n", " ")[:150],
                "caption_latency_sec": round(eval_res["caption_latency"], 3),
                "ocr_latency_sec": round(eval_res["ocr_latency"], 3),
                "total_latency_sec": round(eval_res["total_latency"], 3),
                "error": None
            }
            print(f" Done ({row_data['total_latency_sec']}s)")

        except Exception as exc:
            print(f" FAILED: {exc}")
            row_data = {
                "index": idx,
                "filename": filename,
                "sha256": sample["sha256"],
                "stratum": sample["stratum"],
                "width": sample["width"],
                "height": sample["height"],
                "color_mode": sample["color_mode"],
                "aspect_ratio": sample["aspect_ratio"],
                "anchor_caption": "",
                "diverse_caption": "",
                "fused_caption": "",
                "fusion_status": "FAILED",
                "candidate_count": 0,
                "candidates_json": "[]",
                "diversity_ttr": 0.0,
                "diversity_mean_jaccard": 0.0,
                "extracted_entities": "",
                "ocr_has_text": False,
                "ocr_text": "",
                "caption_latency_sec": 0.0,
                "ocr_latency_sec": 0.0,
                "total_latency_sec": round(time.time() - t_start, 3),
                "error": str(exc)
            }

        # Incremental write to JSONL
        with open(jsonl_path, "a", encoding="utf-8") as f_jsonl:
            f_jsonl.write(json.dumps(row_data) + "\n")

        results_list.append(row_data)

    # Read complete JSONL for writing final CSV and summary
    all_records = []
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f_jsonl:
            for line in f_jsonl:
                if line.strip():
                    try:
                        all_records.append(json.loads(line))
                    except Exception:
                        pass

    # Write CSV
    if all_records:
        fieldnames = list(all_records[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)

    # Compute Summary
    successful = [r for r in all_records if r.get("error") is None and r.get("anchor_caption")]
    latencies = [r["total_latency_sec"] for r in successful]
    ttrs = [r["diversity_ttr"] for r in successful]
    jaccards = [r["diversity_mean_jaccard"] for r in successful]

    def median_val(lst):
        if not lst:
            return 0.0
        s = sorted(lst)
        n = len(s)
        return (s[n // 2] if n % 2 != 0 else (s[n // 2 - 1] + s[n // 2]) / 2)

    summary_data = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "execution_mode": eval_mode,
        "sample_manifest": str(MANIFEST_PATH),
        "total_requested": len(samples),
        "total_processed": len(all_records),
        "successful_evaluations": len(successful),
        "failed_evaluations": len(all_records) - len(successful),
        "latencies_seconds": {
            "mean": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
            "min": round(min(latencies), 3) if latencies else 0.0,
            "max": round(max(latencies), 3) if latencies else 0.0,
            "median": round(median_val(latencies), 3)
        },
        "candidate_diversity": {
            "mean_type_token_ratio": round(sum(ttrs) / len(ttrs), 4) if ttrs else 0.0,
            "mean_pairwise_jaccard": round(sum(jaccards) / len(jaccards), 4) if jaccards else 0.0
        },
        "stratum_breakdown": {}
    }

    # Strata breakdown
    for r in all_records:
        st = r["stratum"]
        if st not in summary_data["stratum_breakdown"]:
            summary_data["stratum_breakdown"][st] = {"count": 0, "success": 0, "failed": 0}
        summary_data["stratum_breakdown"][st]["count"] += 1
        if r.get("error") is None:
            summary_data["stratum_breakdown"][st]["success"] += 1
        else:
            summary_data["stratum_breakdown"][st]["failed"] += 1

    with open(summary_path, "w", encoding="utf-8") as f_sum:
        json.dump(summary_data, f_sum, indent=2)

    # 10. Manual Review CSV Template (stratified subset of 25 images)
    # Strictly leave human annotation fields blank!
    review_subset = []
    if successful:
        import random
        rng_rev = random.Random(42)
        # Stratified selection for review (25 items)
        by_stratum = {}
        for r in successful:
            by_stratum.setdefault(r["stratum"], []).append(r)
        
        for st, items in by_stratum.items():
            k = max(1, round(len(items) / len(successful) * 25))
            review_subset.extend(rng_rev.sample(items, min(k, len(items))))
        review_subset = review_subset[:25]

    if review_subset:
        review_fieldnames = [
            "filename", "stratum", "anchor_caption", "diverse_caption", "fused_caption",
            "extracted_entities", "ocr_text",
            "human_verified_visible_objects",
            "human_missing_salient_objects",
            "human_hallucinated_false_claims",
            "human_grounding_rating_1_to_5",
            "human_reviewer_notes"
        ]
        with open(template_path, "w", encoding="utf-8", newline="") as f_temp:
            writer = csv.DictWriter(f_temp, fieldnames=review_fieldnames)
            writer.writeheader()
            for item in review_subset:
                writer.writerow({
                    "filename": item["filename"],
                    "stratum": item["stratum"],
                    "anchor_caption": item["anchor_caption"],
                    "diverse_caption": item["diverse_caption"],
                    "fused_caption": item["fused_caption"],
                    "extracted_entities": item["extracted_entities"],
                    "ocr_text": item["ocr_text"],
                    "human_verified_visible_objects": "",
                    "human_missing_salient_objects": "",
                    "human_hallucinated_false_claims": "",
                    "human_grounding_rating_1_to_5": "",
                    "human_reviewer_notes": ""
                })
        print(f"[Baseline] Manual review template generated: {template_path}")

    print(f"\n[Baseline] Evaluation completed successfully.")
    print(f"[Baseline] Results saved to: {csv_path}")
    print(f"[Baseline] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

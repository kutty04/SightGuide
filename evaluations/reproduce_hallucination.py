"""Diagnostic reproduction script for SightGuide V2 Visual Grounding.

Tests the animated portrait example (woman in blue shirt, upper torso only).
Checks whether unsupported claims (jeans, pants, Sims family, lower body)
appear in:
1. BLIP candidate generation (sampling and beam search)
2. Score assignment
3. MMR candidate selection
4. LLM Fusion prompt and output
"""

import sys
import json
import hashlib
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.services.caption_pipeline import CaptionPipeline

def run_diagnosis(image_path: Path):
    print("=" * 80)
    print(f"DIAGNOSIS ON: {image_path.name}")
    print("=" * 80)

    image_bytes = image_path.read_bytes()
    img_sha256 = hashlib.sha256(image_bytes).hexdigest()
    img = Image.open(image_path).convert("RGB")
    print(f"Dimensions: {img.size[0]}x{img.size[1]} | SHA-256: {img_sha256}")

    pipe = CaptionPipeline.get_instance()
    print(f"BLIP Model loaded: {pipe.blip_loaded} on device: {pipe.device}")

    # 1. Test Multiple Seeds / Strategies
    strategies = [
        ("sampling_seed42", "sample", 42),
        ("sampling_seed100", "sample", 100),
        ("sampling_seed2026", "sample", 2026),
        ("beam_search", "beam", None),
    ]

    all_runs = {}

    for strat_name, strategy, seed in strategies:
        print(f"\n--- Strategy: {strat_name} ({strategy}, seed={seed}) ---")
        candidates = pipe.generate_candidates(img, num_candidates=5, decoding_strategy=strategy, seed=seed)
        scored = pipe.score_candidates(img, candidates)
        
        print(f"Generated {len(candidates)} candidates:")
        for idx, s in enumerate(scored, 1):
            cap = s["caption"]
            score = s["clip_score"]
            raw_loss = s.get("raw_loss")
            norm_s = s.get("normalized_clip_score")
            has_jeans = any(w in cap.lower() for w in ["jean", "jeans", "pants", "trousers", "sims", "standing"])
            flag = " [HALLUCINATION DETECTED]" if has_jeans else ""
            print(f"  {idx}. [Score: {score:.3f} | Loss: {raw_loss} | Norm: {norm_s:.3f}] {cap}{flag}")

        # Run full process
        res = pipe.process_image(
            image_bytes=image_bytes,
            filename=image_path.name,
            lambda_weight=0.5,
            target_method="proposed",
            decoding_strategy=strategy
        )

        c1 = res["selected_pair"].rank_1_caption
        c2 = res["selected_pair"].rank_2_caption
        fusion = res["fusion"]

        print(f"Selected Pair (Proposed MMR lambda=0.5):")
        print(f"  Anchor c1*: {c1}")
        print(f"  Detail c2*: {c2}")
        print(f"Fused Caption ({fusion.generation_status}):")
        print(f"  {fusion.fused_caption}")
        if fusion.error_details:
            print(f"  Error: {fusion.error_details}")

        all_runs[strat_name] = {
            "candidates": scored,
            "anchor": c1,
            "detail": c2,
            "fused": fusion.fused_caption,
            "fusion_status": fusion.generation_status
        }

    # Save output for analysis
    out_file = image_path.parent / "reproduce_hallucination_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2)
    print(f"\nSaved diagnostic record to {out_file}")

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "test_portrait_woman.png"
    if target.exists():
        run_diagnosis(target)
    else:
        print(f"Error: {target} not found")

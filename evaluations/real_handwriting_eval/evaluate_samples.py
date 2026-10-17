"""Handwriting OCR Evaluation Script for SightGuide V2.

Computes character-level error rate (CER) and word-level error rate (WER)
against explicit ground truth transcriptions across real user handwriting samples.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "backend"))

from app.services.ocr_service import OCRService


def levenshtein(s1: list | str, s2: list | str) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev[j + 1] + 1
            dels = curr[j] + 1
            subs = prev[j] + (c1 != c2)
            curr.append(min(ins, dels, subs))
        prev = curr
    return prev[-1]


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER)."""
    ref_clean = "".join(reference.split()).lower()
    hyp_clean = "".join(hypothesis.split()).lower()
    if not ref_clean:
        return 0.0 if not hyp_clean else 1.0
    dist = levenshtein(ref_clean, hyp_clean)
    return round(dist / len(ref_clean), 4)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER)."""
    ref_words = [w.lower() for w in reference.split() if w]
    hyp_words = [w.lower() for w in hypothesis.split() if w]
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = levenshtein(ref_words, hyp_words)
    return round(dist / len(ref_words), 4)


GROUND_TRUTH: dict[str, dict[str, Any]] = {
    "sample_01.jpeg": {
        "description": "Handwritten alphabet on phone note (dark theme)",
        "ground_truth_handwriting": "abcde fghijkl mnopq rstuvw xyz",
        "expected_words": ["abcde", "fghijkl", "mnopq", "rstuvw", "xyz"],
        "is_negative_case": False,
    },
    "sample_02.jpeg": {
        "description": "Handwritten uppercase & lowercase Maddy",
        "ground_truth_handwriting": "MA DDY maddy",
        "expected_words": ["ma", "ddy", "maddy"],
        "is_negative_case": False,
    },
    "sample_03.jpeg": {
        "description": "Handwritten diagonal text 'Cat is Cute'",
        "ground_truth_handwriting": "Cat is Cute",
        "expected_words": ["cat", "is", "cute"],
        "is_negative_case": False,
    },
    "sample_04.jpeg": {
        "description": "Handwritten text near doodle 'HELLO HI ♡'",
        "ground_truth_handwriting": "HELLO HI",
        "expected_words": ["hello", "hi"],
        "is_negative_case": False,
    },
    "sample_05.jpeg": {
        "description": "Handwritten greeting with colored text and heart doodle",
        "ground_truth_handwriting": "Hello I'm Maddy",
        "expected_words": ["hello", "i'm", "maddy"],
        "is_negative_case": False,
    },
    "sample_06.jpeg": {
        "description": "Negative case: intended name is Modu, NOT Maddy",
        "ground_truth_handwriting": "Hi Im Modu",
        "expected_words": ["hi", "im", "modu"],
        "is_negative_case": True,
        "prohibited_substitutions": ["maddy"],  # Must NOT be normalized to Maddy!
    },
}


def run_evaluation() -> dict[str, Any]:
    ocr = OCRService.get_instance()
    img_dir = Path(__file__).resolve().parent / "images"

    results = []
    total_raw_cer = 0.0
    total_norm_cer = 0.0
    total_raw_wer = 0.0
    total_norm_wer = 0.0

    print("=" * 80)
    print("SIGHTGUIDE V2: REAL HANDWRITING OCR EVALUATION BENCHMARK")
    print("=" * 80)

    for filename, gt in GROUND_TRUTH.items():
        img_path = img_dir / filename
        if not img_path.is_file():
            print(f"Skipping {filename} (not found)")
            continue

        with open(img_path, "rb") as f:
            data = f.read()

        ocr_res = ocr.extract_text(data)
        raw_text = ocr_res.get("text", "")
        norm_text = ocr_res.get("normalized_text") or raw_text

        # Extract only the handwriting portion (ignore top status bar if present)
        ref = gt["ground_truth_handwriting"]
        raw_cer = compute_cer(ref, raw_text)
        norm_cer = compute_cer(ref, norm_text)
        raw_wer = compute_wer(ref, raw_text)
        norm_wer = compute_wer(ref, norm_text)

        total_raw_cer += raw_cer
        total_norm_cer += norm_cer
        total_raw_wer += raw_wer
        total_norm_wer += norm_wer

        # Check negative case violations
        prohibited = gt.get("prohibited_substitutions", [])
        violations = [p for p in prohibited if p in norm_text.lower() and p not in ref.lower()]

        item_res = {
            "filename": filename,
            "description": gt["description"],
            "ground_truth": ref,
            "raw_text": raw_text.replace("\n", " "),
            "normalized_text": norm_text.replace("\n", " "),
            "raw_cer": raw_cer,
            "norm_cer": norm_cer,
            "raw_wer": raw_wer,
            "norm_wer": norm_wer,
            "has_violations": len(violations) > 0,
            "violations": violations,
        }
        results.append(item_res)

        print(f"\n[{filename}] - {gt['description']}")
        print(f"  Ground Truth:    {ref}")
        print(f"  Raw OCR:         {raw_text.replace(chr(10), ' ')}")
        print(f"  Normalized OCR:  {norm_text.replace(chr(10), ' ')}")
        print(f"  Raw CER: {raw_cer:.2%} | Norm CER: {norm_cer:.2%}")
        print(f"  Raw WER: {raw_wer:.2%} | Norm WER: {norm_wer:.2%}")
        if violations:
            print(f"  [ALERT] Negative test violation: prohibited word {violations} found!")

    n = max(1, len(results))
    print("\n" + "=" * 80)
    print("SUMMARY METRICS ACROSS EVALUATION SET:")
    print(f"  Average Raw CER:        {total_raw_cer / n:.2%}")
    print(f"  Average Normalized CER: {total_norm_cer / n:.2%}")
    print(f"  Average Raw WER:        {total_raw_wer / n:.2%}")
    print(f"  Average Normalized WER: {total_norm_wer / n:.2%}")
    print("=" * 80)

    return {
        "results": results,
        "mean_raw_cer": round(total_raw_cer / n, 4),
        "mean_norm_cer": round(total_norm_cer / n, 4),
        "mean_raw_wer": round(total_raw_wer / n, 4),
        "mean_norm_wer": round(total_norm_wer / n, 4),
    }


if __name__ == "__main__":
    run_evaluation()

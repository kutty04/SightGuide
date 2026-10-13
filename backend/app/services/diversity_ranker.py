"""SightGuide Diversity-Aware Caption-Pair Ranking Service.

Research Extension to Celona et al. (Springer, 2025):
"Improving image captioning descriptiveness by ranking and LLM-based fusion".

This module implements:
1. Control Method: Naive CLIP Top-2 selection (Rank 1 and Rank 2 by visual alignment).
2. Proposed Method: Diversity-Aware Pair Selection using Maximal Marginal Relevance (MMR)
   with visual grounding safeguards to penalize semantic redundancy while preventing hallucinations.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union


def clean_and_tokenize(text: str) -> set[str]:
    """Extract lowercased alphanumeric tokens from text."""
    if not text or not isinstance(text, str):
        return set()
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = {t for t in cleaned.split() if t}
    return tokens


def normalize_clip_scores(
    scores: list[float],
    method: str = "minmax",
    epsilon: float = 1e-6
) -> list[float]:
    """Normalize raw CLIP image-text similarity scores into [0.0, 1.0].

    Args:
        scores: List of raw float similarity scores.
        method: Normalization method ("minmax" or "softmax").
        epsilon: Small constant to avoid division by zero.

    Returns:
        List of normalized float scores in [0.0, 1.0].
    """
    if not scores:
        return []
    if len(scores) == 1:
        return [1.0]

    valid_scores = [float(s) if not math.isnan(s) else 0.0 for s in scores]
    min_s = min(valid_scores)
    max_s = max(valid_scores)

    if math.isclose(max_s, min_s, abs_tol=epsilon):
        # All scores are effectively identical
        return [1.0] * len(scores)

    if method == "minmax":
        spread = max_s - min_s
        return [round((s - min_s) / spread, 6) for s in valid_scores]
    elif method == "softmax":
        exp_scores = [math.exp(s - max_s) for s in valid_scores]
        total_exp = sum(exp_scores)
        return [round(e / total_exp, 6) for e in exp_scores]
    else:
        raise ValueError(f"Unsupported normalization method: '{method}'")


def compute_lexical_similarity(c1: str, c2: str) -> float:
    """Compute token-level Jaccard similarity between two captions.

    Returns:
        Float value in [0.0, 1.0], where 1.0 indicates identical token sets
        and 0.0 indicates completely disjoint token vocabularies.
    """
    tokens1 = clean_and_tokenize(c1)
    tokens2 = clean_and_tokenize(c2)

    if not tokens1 and not tokens2:
        return 1.0  # Both empty -> completely identical empty content
    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return round(intersection / union, 6) if union > 0 else 0.0


def compute_clip_text_cosine_similarity(
    c1: str,
    c2: str,
    clip_model: Any = None,
    clip_processor: Any = None,
    device: str = "cpu"
) -> float:
    """Compute neural semantic similarity between two captions using CLIP text embeddings.

    Falls back to lexical Jaccard similarity if CLIP model/processor is not provided.

    Returns:
        Float value in [0.0, 1.0].
    """
    if clip_model is None or clip_processor is None:
        return compute_lexical_similarity(c1, c2)

    try:
        import torch
        import torch.nn.functional as F

        inputs = clip_processor(
            text=[c1, c2],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.inference_mode():
            if hasattr(clip_model, "get_text_features"):
                embeddings = clip_model.get_text_features(**inputs)
            else:
                text_outputs = clip_model.text_model(**inputs)
                embeddings = text_outputs[1]

            norm_emb = F.normalize(embeddings, p=2, dim=-1)
            cosine_sim = torch.sum(norm_emb[0] * norm_emb[1]).item()

        # Cosine similarity for natural English captions is typically in [0.0, 1.0]
        clamped_sim = max(0.0, min(1.0, cosine_sim))
        return round(float(clamped_sim), 6)
    except Exception:
        # Fall back gracefully to lexical similarity if torch/CLIP computation fails
        return compute_lexical_similarity(c1, c2)


def select_control_pair(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Select candidate pair using the Control Method: Naive CLIP Top-2.

    Identical to the baseline selection in Celona et al. (2025) and Colab Notebook.

    Args:
        candidates: List of dicts with 'caption' (str) and 'clip_score' (float).

    Returns:
        Dict detailing the selected control pair, component scores, and metadata.
    """
    if not candidates:
        raise ValueError("Candidate pool is empty.")
    if len(candidates) < 2:
        raise ValueError(
            f"Control selection requires at least 2 candidates; received {len(candidates)}."
        )

    # Sort strictly by descending raw CLIP score
    sorted_candidates = sorted(
        candidates,
        key=lambda item: float(item.get("clip_score", 0.0)),
        reverse=True
    )

    c1 = sorted_candidates[0]
    c2 = sorted_candidates[1]

    caption_1 = str(c1.get("caption", "")).strip()
    caption_2 = str(c2.get("caption", "")).strip()
    score_1 = float(c1.get("clip_score", 0.0))
    score_2 = float(c2.get("clip_score", 0.0))

    sim = compute_lexical_similarity(caption_1, caption_2)

    return {
        "method": "control_naive_clip_top2",
        "rank_1_caption": caption_1,
        "rank_1_clip_score": round(score_1, 4),
        "rank_2_caption": caption_2,
        "rank_2_clip_score": round(score_2, 4),
        "rank_2_original_rank": 2,
        "pair_lexical_similarity": sim,
        "pair_diversity": round(1.0 - sim, 6),
        "selection_notes": "Selected strictly by highest individual CLIP alignment."
    }


def select_diversity_pair(
    candidates: list[dict[str, Any]],
    lambda_weight: float = 0.5,
    similarity_backend: str = "jaccard",
    min_grounding_threshold: float = 0.20,
    clip_model: Any = None,
    clip_processor: Any = None,
    device: str = "cpu"
) -> dict[str, Any]:
    """Select candidate pair using the Proposed Method: Diversity-Aware Ranking.

    Formulation:
    1. Primary visual anchor c1* is fixed as the highest-grounded candidate (Rank 1):
       c1* = argmax_{c_i} S_CLIP(c_i, I)
    2. Normalize CLIP grounding scores across candidate pool into [0.0, 1.0].
    3. For all secondary candidates c_j in C \\ {c1*}:
       Score(c1*, c_j; lambda) = lambda * S_norm_CLIP(c_j) - (1 - lambda) * Sim(c1*, c_j) - Penalty(grounding)
    4. Best complementary candidate c2* maximizes the score.

    Safeguards:
    - lambda in [0.0, 1.0] controls grounding vs. diversity trade-off.
      When lambda = 1.0, this mathematically reproduces the exact Control Method.
    - min_grounding_threshold prevents selecting hallucinated or irrelevant captions
      merely because they differ from the anchor.

    Args:
        candidates: List of candidate dicts with 'caption' and 'clip_score'.
        lambda_weight: Trade-off parameter in [0.0, 1.0] (default 0.5).
        similarity_backend: "jaccard" (lexical) or "clip_text" (neural semantic).
        min_grounding_threshold: Minimum normalized CLIP score required to be eligible
                                 without grounding penalty.
        clip_model: Optional pre-loaded CLIP model for neural text similarity.
        clip_processor: Optional pre-loaded CLIP processor.
        device: Device for CLIP model inference.

    Returns:
        Structured dict containing selected pair, component scores, penalties,
        and per-candidate evaluations.
    """
    if not candidates:
        raise ValueError("Candidate pool is empty.")
    if len(candidates) < 2:
        raise ValueError(
            f"Diversity selection requires at least 2 candidates; received {len(candidates)}."
        )
    if not (0.0 <= lambda_weight <= 1.0):
        raise ValueError(
            f"lambda_weight must be in range [0.0, 1.0]; got {lambda_weight}."
        )

    # Clean and filter valid candidates
    cleaned_pool = []
    for idx, c in enumerate(candidates):
        cap = str(c.get("caption", "")).strip()
        score = float(c.get("clip_score", 0.0))
        if cap:
            cleaned_pool.append({
                "caption": cap,
                "clip_score": score,
                "input_order": idx
            })

    if len(cleaned_pool) < 2:
        raise ValueError("Fewer than 2 non-empty candidates in pool.")

    # Sort pool to identify natural CLIP ranking
    cleaned_pool.sort(key=lambda x: x["clip_score"], reverse=True)
    for natural_rank, item in enumerate(cleaned_pool, start=1):
        item["natural_clip_rank"] = natural_rank

    # Anchor candidate c1* is strictly the top CLIP candidate
    anchor = cleaned_pool[0]
    raw_scores = [c["clip_score"] for c in cleaned_pool]
    norm_scores = normalize_clip_scores(raw_scores, method="minmax")
    for item, n_score in zip(cleaned_pool, norm_scores):
        item["normalized_clip_score"] = n_score

    anchor_caption = anchor["caption"]
    anchor_raw_clip = anchor["clip_score"]
    anchor_norm_clip = anchor["normalized_clip_score"]

    evaluated_secondaries = []

    for secondary in cleaned_pool[1:]:
        sec_caption = secondary["caption"]
        sec_raw_clip = secondary["clip_score"]
        sec_norm_clip = secondary["normalized_clip_score"]

        # Compute pairwise similarity
        if similarity_backend == "clip_text" and clip_model is not None:
            sim = compute_clip_text_cosine_similarity(
                anchor_caption,
                sec_caption,
                clip_model=clip_model,
                clip_processor=clip_processor,
                device=device
            )
        else:
            sim = compute_lexical_similarity(anchor_caption, sec_caption)

        diversity = round(1.0 - sim, 6)
        grounding_term = round(lambda_weight * sec_norm_clip, 6)
        diversity_penalty = round((1.0 - lambda_weight) * sim, 6)

        # Grounding safeguard: penalize candidates below threshold to prevent hallucination promotion
        grounding_penalty = 0.0
        if sec_norm_clip < min_grounding_threshold:
            gap = min_grounding_threshold - sec_norm_clip
            grounding_penalty = round(gap * 2.0, 6)

        # Duplicate safeguard: penalize exact duplicate captions to prevent pairing anchor with itself
        duplicate_penalty = 0.0
        if math.isclose(sim, 1.0, abs_tol=1e-5) or sec_caption.lower() == anchor_caption.lower():
            duplicate_penalty = 1.0

        pair_score = round(
            grounding_term - diversity_penalty - grounding_penalty - duplicate_penalty,
            6
        )

        evaluated_secondaries.append({
            "caption": sec_caption,
            "natural_clip_rank": secondary["natural_clip_rank"],
            "raw_clip_score": round(sec_raw_clip, 4),
            "normalized_clip_score": sec_norm_clip,
            "similarity": sim,
            "diversity": diversity,
            "grounding_term": grounding_term,
            "diversity_penalty": diversity_penalty,
            "grounding_penalty": grounding_penalty,
            "pair_score": pair_score
        })

    # Select secondary candidate with highest pair_score
    # In case of tie, tie-break by higher raw_clip_score
    evaluated_secondaries.sort(
        key=lambda item: (item["pair_score"], item["raw_clip_score"]),
        reverse=True
    )
    best_secondary = evaluated_secondaries[0]

    # Check whether the selected secondary differs from Naive Rank 2
    naive_rank_2 = cleaned_pool[1]
    diverges_from_naive = (
        best_secondary["caption"] != naive_rank_2["caption"]
    )

    return {
        "method": "proposed_diversity_aware",
        "similarity_backend": similarity_backend,
        "lambda_weight": lambda_weight,
        "min_grounding_threshold": min_grounding_threshold,
        "rank_1_caption": anchor_caption,
        "rank_1_clip_score": round(anchor_raw_clip, 4),
        "rank_1_clip_norm": anchor_norm_clip,
        "rank_2_caption": best_secondary["caption"],
        "rank_2_clip_score": best_secondary["raw_clip_score"],
        "rank_2_clip_norm": best_secondary["normalized_clip_score"],
        "rank_2_original_rank": best_secondary["natural_clip_rank"],
        "pair_similarity": best_secondary["similarity"],
        "pair_diversity": best_secondary["diversity"],
        "grounding_term": best_secondary["grounding_term"],
        "diversity_penalty": best_secondary["diversity_penalty"],
        "grounding_penalty": best_secondary["grounding_penalty"],
        "pair_score": best_secondary["pair_score"],
        "diverges_from_naive_clip_rank_2": diverges_from_naive,
        "candidate_evaluations": evaluated_secondaries
    }

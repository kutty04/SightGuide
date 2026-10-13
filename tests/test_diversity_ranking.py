"""Unit and Regression Tests for SightGuide Diversity-Aware Ranking.

Tests verify:
1. Normalization properties (minmax scaling, identical scores, single score, empty pool).
2. Pairwise similarity metrics (Jaccard token calculation, case/punctuation invariance, duplicate penalty).
3. Control method baseline fidelity (naive CLIP Top-2 selection).
4. Proposed diversity-aware ranking behavior:
   - Redundancy penalty against near-duplicate captions.
   - Grounding safeguard preventing hallucination promotion.
   - Equivalence to Control method when lambda = 1.0.
   - Deterministic reproducibility across repeated invocations.
   - Robust error handling for empty/malformed inputs.
"""

import math
import sys
import unittest
from pathlib import Path

# Add backend to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "backend"))

from app.services.diversity_ranker import (
    clean_and_tokenize,
    normalize_clip_scores,
    compute_lexical_similarity,
    select_control_pair,
    select_diversity_pair
)


class TestNormalization(unittest.TestCase):
    def test_minmax_normalization(self):
        scores = [20.0, 25.0, 30.0]
        norm = normalize_clip_scores(scores, method="minmax")
        self.assertEqual(len(norm), 3)
        self.assertAlmostEqual(norm[0], 0.0, places=5)
        self.assertAlmostEqual(norm[1], 0.5, places=5)
        self.assertAlmostEqual(norm[2], 1.0, places=5)

    def test_identical_scores(self):
        scores = [30.0, 30.0, 30.0]
        norm = normalize_clip_scores(scores, method="minmax")
        self.assertEqual(norm, [1.0, 1.0, 1.0])

    def test_single_score(self):
        scores = [27.5]
        norm = normalize_clip_scores(scores, method="minmax")
        self.assertEqual(norm, [1.0])

    def test_empty_scores(self):
        self.assertEqual(normalize_clip_scores([]), [])


class TestSimilarityMetrics(unittest.TestCase):
    def test_clean_and_tokenize(self):
        text = "A dog, sitting happily... on the porch!"
        tokens = clean_and_tokenize(text)
        expected = {"a", "dog", "sitting", "happily", "on", "the", "porch"}
        self.assertEqual(tokens, expected)

    def test_identical_captions_similarity(self):
        c1 = "A small red fox on a rock."
        c2 = "a small red fox on a rock"
        self.assertAlmostEqual(compute_lexical_similarity(c1, c2), 1.0, places=5)

    def test_completely_disjoint_captions(self):
        c1 = "white clouds in blue sky"
        c2 = "pizza on wooden board"
        self.assertAlmostEqual(compute_lexical_similarity(c1, c2), 0.0, places=5)

    def test_partial_token_overlap(self):
        c1 = "apple orange banana"
        c2 = "banana grape melon"
        # Intersection = {banana} (1), Union = {apple, orange, banana, grape, melon} (5) -> 1/5 = 0.2
        self.assertAlmostEqual(compute_lexical_similarity(c1, c2), 0.2, places=5)


class TestControlMethod(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            {"caption": "A cat sitting on a rug", "clip_score": 28.5},
            {"caption": "A cute cat on a fluffy rug", "clip_score": 32.1},  # Rank 1
            {"caption": "Cat indoors near furniture", "clip_score": 29.8},   # Rank 2
            {"caption": "A furry animal sleeping", "clip_score": 24.0},
        ]

    def test_select_control_pair_picks_top_two(self):
        result = select_control_pair(self.candidates)
        self.assertEqual(result["method"], "control_naive_clip_top2")
        self.assertEqual(result["rank_1_caption"], "A cute cat on a fluffy rug")
        self.assertEqual(result["rank_2_caption"], "Cat indoors near furniture")
        self.assertAlmostEqual(result["rank_1_clip_score"], 32.1, places=2)
        self.assertAlmostEqual(result["rank_2_clip_score"], 29.8, places=2)

    def test_insufficient_candidates_raises_error(self):
        with self.assertRaises(ValueError):
            select_control_pair([{"caption": "only one", "clip_score": 30.0}])


class TestDiversityAwareMethod(unittest.TestCase):
    def setUp(self):
        # Simulates real kitten case from SightGuide benchmark
        self.candidates = [
            {"caption": "small kitten on the ground", "clip_score": 29.89},               # Rank 1 anchor
            {"caption": "a small kitten sitting on the ground", "clip_score": 29.46},     # Rank 2 (near duplicate)
            {"caption": "a small kitten sits on the floor near a bench", "clip_score": 29.36}, # Rank 3 (informative)
            {"caption": "small kitten on the floor", "clip_score": 29.09},
            {"caption": "a kitten is sitting on a floor", "clip_score": 27.72},
        ]

    def test_redundancy_penalty_promotes_diverse_candidate(self):
        # At lambda = 0.5, the high redundancy of Rank 2 against Rank 1 should allow Rank 3 to be selected
        result = select_diversity_pair(self.candidates, lambda_weight=0.5)
        self.assertEqual(result["method"], "proposed_diversity_aware")
        self.assertEqual(result["rank_1_caption"], "small kitten on the ground")
        self.assertEqual(result["rank_2_caption"], "a small kitten sits on the floor near a bench")
        self.assertEqual(result["rank_2_original_rank"], 3)
        self.assertTrue(result["diverges_from_naive_clip_rank_2"])

    def test_lambda_one_matches_control_selection(self):
        # When lambda = 1.0, diversity penalty is zero, exactly reproducing Naive CLIP Top-2
        result_div = select_diversity_pair(self.candidates, lambda_weight=1.0)
        result_ctl = select_control_pair(self.candidates)

        self.assertEqual(result_div["rank_1_caption"], result_ctl["rank_1_caption"])
        self.assertEqual(result_div["rank_2_caption"], result_ctl["rank_2_caption"])
        self.assertFalse(result_div["diverges_from_naive_clip_rank_2"])

    def test_grounding_safeguard_prevents_ungrounded_selection(self):
        candidates_with_hallucination = [
            {"caption": "a large white cloud", "clip_score": 30.0},                  # Anchor
            {"caption": "a large puffy white cloud in sky", "clip_score": 29.0},     # Grounded candidate
            {"caption": "a flying superhero in outer space", "clip_score": 10.0},    # Spurious, low score
        ]
        # Even at a low lambda (e.g. 0.3), the superhero caption should be penalized
        # by the grounding safeguard threshold and NOT selected
        result = select_diversity_pair(
            candidates_with_hallucination,
            lambda_weight=0.3,
            min_grounding_threshold=0.25
        )
        self.assertEqual(result["rank_2_caption"], "a large puffy white cloud in sky")

    def test_exact_duplicate_penalty(self):
        candidates = [
            {"caption": "red bicycle parked outside", "clip_score": 30.0},
            {"caption": "red bicycle parked outside", "clip_score": 29.9},  # Exact duplicate of anchor
            {"caption": "a bicycle near a brick wall", "clip_score": 29.0}, # Grounded, distinct
        ]
        result = select_diversity_pair(candidates, lambda_weight=0.5)
        # Exact duplicate has similarity 1.0, incurring maximum diversity penalty
        self.assertEqual(result["rank_2_caption"], "a bicycle near a brick wall")

    def test_deterministic_reproducibility(self):
        res1 = select_diversity_pair(self.candidates, lambda_weight=0.5)
        res2 = select_diversity_pair(self.candidates, lambda_weight=0.5)
        self.assertEqual(res1["pair_score"], res2["pair_score"])
        self.assertEqual(res1["rank_2_caption"], res2["rank_2_caption"])

    def test_invalid_lambda_raises_value_error(self):
        with self.assertRaises(ValueError):
            select_diversity_pair(self.candidates, lambda_weight=1.5)
        with self.assertRaises(ValueError):
            select_diversity_pair(self.candidates, lambda_weight=-0.2)


if __name__ == "__main__":
    unittest.main()

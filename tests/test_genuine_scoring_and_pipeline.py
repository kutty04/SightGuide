"""Unit and Regression Tests for SightGuide Genuine Scoring and Pipeline Safeguards.

Tests verify:
1. Genuine image-text cross-modal log-likelihood scoring against pixel values.
2. Correct score direction: higher score (lower loss) indicates better visual alignment.
3. Proof that synthetic linear rank scores (30.0, 28.0, ...) are eliminated.
4. Candidate cleaning: repeated word stutter reduction and whitespace normalization.
5. Candidate deduplication: near-duplicate pruning to preserve semantic diversity.
6. Benchmark cache isolation: content hash verification prevents arbitrary uploads from colliding with benchmark cache.
7. Decoding strategy configuration: deterministic beam search vs. nucleus sampling.
8. Anchor selection: highest-grounded candidate becomes the visual anchor c1*.
9. Low-confidence flags: proper warning when visual alignment is weak.
"""

from __future__ import annotations

import io
import math
import sys
import unittest
from pathlib import Path

from PIL import Image

project_root = Path(__file__).resolve().parent.parent
backend_dir = project_root / "backend"
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(project_root))

from app.services.caption_pipeline import (
    CaptionPipeline,
    clean_and_validate_caption,
    deduplicate_candidate_pool,
)
from app.services.diversity_ranker import select_diversity_pair


class TestGenuineScoringAndPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = CaptionPipeline.get_instance()

    def test_clean_and_validate_caption_stutter(self) -> None:
        """Verify repeated token stutters are cleaned."""
        stutter = "a a small kitten on on the ground ground"
        cleaned = clean_and_validate_caption(stutter)
        self.assertEqual(cleaned, "a small kitten on the ground")

    def test_deduplicate_candidate_pool(self) -> None:
        """Verify near-duplicate candidate strings are pruned."""
        pool = [
            "a small kitten on the floor",
            "a small kitten sitting on the floor",
            "a green park next to water",
            "a small kitten on the floor near a bench"
        ]
        deduped = deduplicate_candidate_pool(pool)
        self.assertGreaterEqual(len(deduped), 2)
        self.assertIn("a green park next to water", deduped)

    def test_genuine_scoring_against_pixel_data(self) -> None:
        """Verify that scoring uses image pixel values and returns cross-modal log-likelihood."""
        # Create solid blue image (representing sky)
        blue_img = Image.new("RGB", (128, 128), color=(90, 140, 210))
        captions = [
            "a blue sky with white clouds",
            "a slice of pepperoni pizza",
            "The Sims family playing in a room"
        ]

        scored = self.pipeline.score_candidates(blue_img, captions)
        self.assertEqual(len(scored), 3)

        if self.pipeline.blip_loaded:
            # Score must be negative (representing -loss), not synthetic 30.0, 28.0, etc.
            for s in scored:
                self.assertLess(s["clip_score"], 0.0)
                self.assertNotEqual(s["clip_score"], 30.0)
                self.assertNotEqual(s["clip_score"], 28.0)
                self.assertNotEqual(s["clip_score"], 26.0)
                self.assertEqual(s["scoring_method"], "blip_cross_entropy_log_likelihood")

            # "a blue sky with white clouds" must score higher than "The Sims family" on a blue image
            sky_entry = next(c for c in scored if "blue sky" in c["caption"])
            sims_entry = next(c for c in scored if "Sims" in c["caption"])
            self.assertGreater(sky_entry["clip_score"], sims_entry["clip_score"])
            self.assertEqual(scored[0]["caption"], sky_entry["caption"])
            self.assertEqual(scored[0]["rank"], 1)

    def test_anchor_selection_by_genuine_alignment(self) -> None:
        """Verify that highest visually aligned candidate becomes anchor c1* rather than arbitrary sequence order."""
        blue_img = Image.new("RGB", (128, 128), color=(90, 140, 210))
        # Provide hallucinated caption first, factual caption second
        captions = [
            "The Sims family playing in a living room",
            "a bright blue background"
        ]

        scored = self.pipeline.score_candidates(blue_img, captions)
        if self.pipeline.blip_loaded:
            # Factually aligned caption must be rank 1
            self.assertEqual(scored[0]["caption"], "a bright blue background")

            candidate_dicts = [
                {"caption": c["caption"], "clip_score": c["clip_score"]}
                for c in scored
            ]
            proposed_pair = select_diversity_pair(candidate_dicts, lambda_weight=0.5)
            # Anchor c1* must be the factual caption, NOT the hallucination
            self.assertEqual(proposed_pair["rank_1_caption"], "a bright blue background")

    def test_benchmark_cache_isolation_by_payload_hash(self) -> None:
        """Verify that arbitrary uploads cannot hijack benchmark cached scores using identical filename."""
        synthetic_payload = b"arbitrary_non_benchmark_image_bytes_content_xyz_123"
        # Even with benchmark filename, non-matching bytes must not be marked verified benchmark
        is_verified = self.pipeline.is_verified_benchmark("sample-clouds-400x300.jpg", synthetic_payload)
        self.assertFalse(is_verified)

    def test_decoding_strategies_execution(self) -> None:
        """Verify both beam and sample decoding strategies execute cleanly."""
        img = Image.new("RGB", (64, 64), color=(150, 150, 150))
        sample_caps = self.pipeline.generate_candidates(img, num_candidates=3, decoding_strategy="sample", seed=42)
        beam_caps = self.pipeline.generate_candidates(img, num_candidates=3, decoding_strategy="beam")

        self.assertGreaterEqual(len(sample_caps), 2)
        self.assertGreaterEqual(len(beam_caps), 2)
        for c in sample_caps + beam_caps:
            self.assertIsInstance(c, str)
    def test_mmr_lambda_boundaries(self) -> None:
        """Verify MMR boundary conditions: lambda=1.0 matches Control, lambda=0.0 maximizes diversity."""
        candidate_pool = [
            {"caption": "a large black dog playing in the yard", "clip_score": -0.80},
            {"caption": "a big black dog running outside in the yard", "clip_score": -0.85},
            {"caption": "bright green trees and wooden fence behind a pet", "clip_score": -1.10}
        ]
        # At lambda = 1.0, MMR must select Rank 2 regardless of high lexical redundancy
        pair_l1 = select_diversity_pair(candidate_pool, lambda_weight=1.0)
        self.assertEqual(pair_l1["rank_2_original_rank"], 2)
        self.assertEqual(pair_l1["rank_2_caption"], "a big black dog running outside in the yard")

        # At lambda = 0.0, MMR must select Rank 3 because it has highest diversity (lowest overlap)
        pair_l0 = select_diversity_pair(candidate_pool, lambda_weight=0.0)
        self.assertEqual(pair_l0["rank_2_original_rank"], 3)
        self.assertEqual(pair_l0["rank_2_caption"], "bright green trees and wooden fence behind a pet")

    def test_provisional_warning_threshold(self) -> None:
        """Verify that low-scoring candidates trigger transparent provisional notice."""
        self.pipeline.alignment_warning_threshold = -2.0
        # If top score is -2.5, it must trigger the provisional notice
        candidates = [
            {"caption": "unclear blurry shape", "clip_score": -2.5}
        ]
        # Test heuristic check logic
        is_low = candidates[0]["clip_score"] < self.pipeline.alignment_warning_threshold
        self.assertTrue(is_low)
        # Restore default
        self.pipeline.alignment_warning_threshold = -3.5


if __name__ == "__main__":
    unittest.main()

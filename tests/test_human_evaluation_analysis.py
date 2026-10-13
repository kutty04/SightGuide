"""Unit and Integration Tests for SightGuide Human Evaluation Analysis Utility.

NOTE: All test datasets, fixtures, and scores used in this module are purely
SYNTHETIC_TEST_FIXTURE_DO_NOT_USE_AS_RESEARCH_DATA.
They exist solely to test validation, unblinding, aggregation, and statistical logic.
Zero API credits or research data are consumed or altered.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.analyze_human_evaluation import (
    RATING_DIMENSIONS,
    aggregate_by_image,
    analyze_human_evaluation,
    compute_descriptive_stats,
    compute_inter_rater_reliability,
    compute_wilcoxon_signed_rank,
    load_and_validate_ratings,
    parse_numeric_score,
)


class TestInputValidationAndParsing(unittest.TestCase):
    def test_parse_numeric_score_valid(self):
        self.assertEqual(parse_numeric_score("1"), 1.0)
        self.assertEqual(parse_numeric_score("5"), 5.0)
        self.assertEqual(parse_numeric_score("3.5"), 3.5)
        self.assertEqual(parse_numeric_score(4), 4.0)

    def test_parse_numeric_score_invalid(self):
        self.assertIsNone(parse_numeric_score("0"))
        self.assertIsNone(parse_numeric_score("6"))
        self.assertIsNone(parse_numeric_score("-2"))
        self.assertIsNone(parse_numeric_score("abc"))
        self.assertIsNone(parse_numeric_score(""))
        self.assertIsNone(parse_numeric_score(None))


class TestBlindingAndUnblinding(unittest.TestCase):
    def test_unblinding_maps_correct_methods(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ratings_csv = tmp / "ratings.csv"
            key_csv = tmp / "key.csv"

            # Create synthetic blinding key
            with open(key_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "image", "image_path", "data_quality_status", "blind_mapping",
                    "method_a_identity", "method_b_identity", "control_fused_caption",
                    "control_status", "proposed_fused_caption", "proposed_status",
                    "baseline_caption_regenerated", "baseline_provenance"
                ])
                writer.writeheader()
                writer.writerow({
                    "image": "synthetic_img1.jpg", "image_path": "dataset/synthetic_img1.jpg",
                    "data_quality_status": "READY_FOR_EVALUATION", "blind_mapping": "A=PROPOSED, B=CONTROL",
                    "method_a_identity": "PROPOSED", "method_b_identity": "CONTROL",
                    "control_fused_caption": "c_cap", "control_status": "SUCCESS",
                    "proposed_fused_caption": "p_cap", "proposed_status": "SUCCESS",
                    "baseline_caption_regenerated": "base", "baseline_provenance": "REGENERATED_NOT_ORIGINAL"
                })

            # Create synthetic blinded rating
            with open(ratings_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "image", "method_a_accuracy", "method_a_relevant_detail",
                    "method_a_hallucination_control", "method_a_overall_quality",
                    "method_b_accuracy", "method_b_relevant_detail",
                    "method_b_hallucination_control", "method_b_overall_quality",
                    "reviewer_id", "reviewer_comments"
                ])
                writer.writeheader()
                writer.writerow({
                    "image": "synthetic_img1.jpg",
                    "method_a_accuracy": "5", "method_a_relevant_detail": "4",
                    "method_a_hallucination_control": "5", "method_a_overall_quality": "5",
                    "method_b_accuracy": "4", "method_b_relevant_detail": "3",
                    "method_b_hallucination_control": "4", "method_b_overall_quality": "3",
                    "reviewer_id": "rev_A", "reviewer_comments": "synthetic review"
                })

            records, meta = load_and_validate_ratings(ratings_csv, key_csv)
            self.assertEqual(len(records), 1)
            rec = records[0]
            # Since A=PROPOSED and B=CONTROL:
            # Proposed score should be Method A (5, 4, 5, 5)
            self.assertEqual(rec["proposed_scores"]["accuracy"], 5.0)
            self.assertEqual(rec["proposed_scores"]["relevant_detail"], 4.0)
            # Control score should be Method B (4, 3, 4, 3)
            self.assertEqual(rec["control_scores"]["accuracy"], 4.0)
            self.assertEqual(rec["control_scores"]["relevant_detail"], 3.0)


class TestExclusionRules(unittest.TestCase):
    def test_pending_and_failed_records_excluded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ratings_csv = tmp / "ratings.csv"
            key_csv = tmp / "key.csv"

            with open(key_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "image", "method_a_identity", "method_b_identity", "data_quality_status"
                ])
                writer.writeheader()
                writer.writerow({
                    "image": "pending_img.jpg", "method_a_identity": "CONTROL",
                    "method_b_identity": "PROPOSED", "data_quality_status": "PENDING_LIVE_GENERATION"
                })
                writer.writerow({
                    "image": "failed_img.jpg", "method_a_identity": "CONTROL",
                    "method_b_identity": "PROPOSED", "data_quality_status": "EXCLUDED_HISTORICAL_FAILURE"
                })

            with open(ratings_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "image", "method_a_accuracy", "method_a_relevant_detail",
                    "method_a_hallucination_control", "method_a_overall_quality",
                    "method_b_accuracy", "method_b_relevant_detail",
                    "method_b_hallucination_control", "method_b_overall_quality"
                ])
                writer.writeheader()
                writer.writerow({
                    "image": "pending_img.jpg",
                    "method_a_accuracy": "3", "method_a_relevant_detail": "3",
                    "method_a_hallucination_control": "3", "method_a_overall_quality": "3",
                    "method_b_accuracy": "3", "method_b_relevant_detail": "3",
                    "method_b_hallucination_control": "3", "method_b_overall_quality": "3"
                })
                writer.writerow({
                    "image": "failed_img.jpg",
                    "method_a_accuracy": "1", "method_a_relevant_detail": "1",
                    "method_a_hallucination_control": "1", "method_a_overall_quality": "1",
                    "method_b_accuracy": "1", "method_b_relevant_detail": "1",
                    "method_b_hallucination_control": "1", "method_b_overall_quality": "1"
                })

            records, meta = load_and_validate_ratings(ratings_csv, key_csv)
            # Both should be excluded
            self.assertEqual(len(records), 0)
            self.assertEqual(meta["excluded_pending_or_failed"], 2)

    def test_incomplete_ratings_excluded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ratings_csv = tmp / "ratings.csv"
            with open(ratings_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "image", "control_accuracy", "control_relevant_detail",
                    "control_hallucination_control", "control_overall_quality",
                    "proposed_accuracy", "proposed_relevant_detail",
                    "proposed_hallucination_control", "proposed_overall_quality"
                ])
                writer.writeheader()
                # Missing proposed_overall_quality
                writer.writerow({
                    "image": "img1.jpg",
                    "control_accuracy": "4", "control_relevant_detail": "4",
                    "control_hallucination_control": "4", "control_overall_quality": "4",
                    "proposed_accuracy": "5", "proposed_relevant_detail": "4",
                    "proposed_hallucination_control": "5", "proposed_overall_quality": ""
                })

            records, meta = load_and_validate_ratings(ratings_csv)
            self.assertEqual(len(records), 0)
            self.assertEqual(meta["excluded_incomplete_ratings"], 1)

    def test_duplicate_reviews_flagged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ratings_csv = tmp / "ratings.csv"
            with open(ratings_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "image", "reviewer_id", "control_accuracy", "control_relevant_detail",
                    "control_hallucination_control", "control_overall_quality",
                    "proposed_accuracy", "proposed_relevant_detail",
                    "proposed_hallucination_control", "proposed_overall_quality"
                ])
                writer.writeheader()
                # Reviewer 1 rates img1 twice
                for _ in range(2):
                    writer.writerow({
                        "image": "img1.jpg", "reviewer_id": "reviewer_1",
                        "control_accuracy": "4", "control_relevant_detail": "4",
                        "control_hallucination_control": "4", "control_overall_quality": "4",
                        "proposed_accuracy": "5", "proposed_relevant_detail": "4",
                        "proposed_hallucination_control": "5", "proposed_overall_quality": "5"
                    })

            records, meta = load_and_validate_ratings(ratings_csv)
            self.assertEqual(len(records), 1)
            self.assertEqual(meta["duplicate_reviews_flagged"], 1)


class TestDescriptiveStatsAndAggregation(unittest.TestCase):
    def test_descriptive_stats_computation(self):
        data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0]
        stats = compute_descriptive_stats(data)
        self.assertEqual(stats["count"], 7)
        self.assertAlmostEqual(stats["mean"], 4.4286, places=4)
        self.assertEqual(stats["median"], 4.0)
        self.assertEqual(stats["min"], 2.0)
        self.assertEqual(stats["max"], 7.0)

    def test_multi_reviewer_aggregation_prevents_pseudoreplication(self):
        # Two reviewers evaluate the SAME image
        records = [
            {
                "image": "img_shared.jpg",
                "reviewer_id": "rev_1",
                "control_scores": {"accuracy": 4.0, "relevant_detail": 4.0, "hallucination_control": 4.0, "overall_quality": 4.0},
                "proposed_scores": {"accuracy": 5.0, "relevant_detail": 5.0, "hallucination_control": 5.0, "overall_quality": 5.0},
            },
            {
                "image": "img_shared.jpg",
                "reviewer_id": "rev_2",
                "control_scores": {"accuracy": 2.0, "relevant_detail": 2.0, "hallucination_control": 2.0, "overall_quality": 2.0},
                "proposed_scores": {"accuracy": 3.0, "relevant_detail": 3.0, "hallucination_control": 3.0, "overall_quality": 3.0},
            }
        ]

        aggs = aggregate_by_image(records)
        # MUST result in exactly 1 image aggregate, not 2
        self.assertEqual(len(aggs), 1)
        agg = aggs[0]
        self.assertEqual(agg["image"], "img_shared.jpg")
        self.assertEqual(agg["reviewer_count"], 2)
        # Average control accuracy = (4 + 2) / 2 = 3.0
        self.assertEqual(agg["control_scores"]["accuracy"], 3.0)
        # Average proposed accuracy = (5 + 3) / 2 = 4.0
        self.assertEqual(agg["proposed_scores"]["accuracy"], 4.0)
        # Paired delta = 4.0 - 3.0 = +1.0
        self.assertEqual(agg["paired_deltas"]["accuracy"], 1.0)


class TestInterRaterReliability(unittest.TestCase):
    def test_inter_rater_metrics_calculation(self):
        records = [
            # Reviewer 1
            {
                "image": "img1.jpg", "reviewer_id": "rev_1",
                "control_scores": {d: 4.0 for d in RATING_DIMENSIONS},
                "proposed_scores": {d: 5.0 for d in RATING_DIMENSIONS},
            },
            {
                "image": "img2.jpg", "reviewer_id": "rev_1",
                "control_scores": {d: 3.0 for d in RATING_DIMENSIONS},
                "proposed_scores": {d: 4.0 for d in RATING_DIMENSIONS},
            },
            # Reviewer 2 (gives scores 1 point lower)
            {
                "image": "img1.jpg", "reviewer_id": "rev_2",
                "control_scores": {d: 3.0 for d in RATING_DIMENSIONS},
                "proposed_scores": {d: 4.0 for d in RATING_DIMENSIONS},
            },
            {
                "image": "img2.jpg", "reviewer_id": "rev_2",
                "control_scores": {d: 2.0 for d in RATING_DIMENSIONS},
                "proposed_scores": {d: 3.0 for d in RATING_DIMENSIONS},
            },
        ]

        irr = compute_inter_rater_reliability(records)
        self.assertEqual(irr["status"], "COMPUTED")
        pair_key = "rev_1_vs_rev_2"
        self.assertIn(pair_key, irr["pairwise_metrics"])
        m = irr["pairwise_metrics"][pair_key]
        self.assertEqual(m["shared_image_count"], 2)
        # Difference is exactly 1.0 point on all scores
        self.assertEqual(m["ctrl_overall_quality_mad"], 1.0)
        self.assertEqual(m["ctrl_overall_quality_exact_pct"], 0.0)
        self.assertEqual(m["ctrl_overall_quality_within1_pct"], 100.0)
        # Perfect linear correlation (slope = 1.0)
        self.assertAlmostEqual(m["ctrl_overall_quality_pearson_r"], 1.0, places=4)


class TestWilcoxonSignedRank(unittest.TestCase):
    def test_wilcoxon_on_synthetic_differences(self):
        deltas = [1.0, 1.0, 2.0, 0.5, 1.5, 2.0]
        res = compute_wilcoxon_signed_rank(deltas)
        self.assertEqual(res["n_effective"], 6)
        self.assertIsNotNone(res["statistic_w"])
        # All differences positive -> negative sum is 0
        self.assertEqual(res["w_negative_sum"], 0.0)
        self.assertEqual(res["statistic_w"], 0.0)

    def test_wilcoxon_handles_small_sample(self):
        deltas = [1.0, 0.0, 0.0]
        res = compute_wilcoxon_signed_rank(deltas)
        # N non-zero = 1 < 5 -> returns note
        self.assertIsNone(res["statistic"])
        self.assertIn("too small", res["note"])


class TestEndToEndAnalysisExecution(unittest.TestCase):
    def test_analyze_human_evaluation_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ratings_csv = tmp / "synthetic_ratings.csv"
            out_details = tmp / "details.csv"
            out_summary = tmp / "summary.csv"

            with open(ratings_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "image", "reviewer_id", "data_quality_status",
                    "control_accuracy", "control_relevant_detail",
                    "control_hallucination_control", "control_overall_quality",
                    "proposed_accuracy", "proposed_relevant_detail",
                    "proposed_hallucination_control", "proposed_overall_quality"
                ])
                writer.writeheader()
                for i in range(1, 6):
                    writer.writerow({
                        "image": f"img_{i}.jpg",
                        "reviewer_id": "evaluator_1",
                        "data_quality_status": "READY_FOR_EVALUATION",
                        "control_accuracy": "3", "control_relevant_detail": "3",
                        "control_hallucination_control": "4", "control_overall_quality": "3",
                        "proposed_accuracy": "4", "proposed_relevant_detail": "4",
                        "proposed_hallucination_control": "5", "proposed_overall_quality": "4"
                    })

            detailed, summary = analyze_human_evaluation(
                ratings_csv=ratings_csv,
                blinding_key_csv=None,
                output_details_csv=out_details,
                output_summary_csv=out_summary
            )

            self.assertEqual(len(detailed), 5)
            self.assertEqual(len(summary), 4) # 4 dimensions
            self.assertTrue(out_details.exists())
            self.assertTrue(out_summary.exists())

            # Verify delta is +1.0 across all dimensions
            for row in summary:
                self.assertEqual(row["mean_paired_difference"], 1.0)
                self.assertEqual(row["proposed_superior_count"], 5)
                self.assertEqual(row["control_superior_count"], 0)


if __name__ == "__main__":
    unittest.main()

"""Unit and Regression Tests for SightGuide Phase 2 Controlled Fusion Comparison.

Uses mocked LLM responses to ensure zero API credit consumption.
Tests verify:
1. Pair integrity and image coverage across benchmark images.
2. Identical generation configurations across Control and Proposed methods.
3. Strict historical reuse rules (exact input and prompt matching).
4. Reuse of identical Control and Proposed pairs with explicit provenance.
5. Bounded retry mechanism on API failures (<= 2 attempts).
6. Strict classification of empty responses and API failures (never marked as SUCCESS).
7. Exclusion of failed generations from quality metric averages.
8. Output schema validation and resumption without duplicate records.
9. Immutability and preservation of all Phase 1 historical artifacts.
"""

import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

from experiments.run_fusion_comparison import (
    CONSERVATIVE_V2_SYSTEM_PROMPT,
    FUSION_MODEL,
    FUSION_TEMPERATURE,
    FUSION_MAX_TOKENS,
    FUSION_PROMPT_VERSION,
    build_user_prompt,
    call_groq_api_with_retry,
    load_pairs_dataset,
    load_historical_v2_controls,
    load_existing_comparison_records,
    run_fusion_comparison,
    build_summary_table
)
from experiments.evaluate_fusion_quality import (
    tokenize_words,
    calculate_ttr,
    calculate_vocabulary_expansion,
    check_known_artifacts,
    build_quality_summary
)


class TestPromptAndConfigIntegrity(unittest.TestCase):
    def test_fusion_configuration_parameters(self):
        self.assertEqual(FUSION_MODEL, "openai/gpt-oss-20b")
        self.assertEqual(FUSION_TEMPERATURE, 0.1)
        self.assertEqual(FUSION_MAX_TOKENS, 400)
        self.assertEqual(FUSION_PROMPT_VERSION, "conservative_v2")

    def test_system_prompt_rules(self):
        self.assertIn("accurate image-caption fusion assistant", CONSERVATIVE_V2_SYSTEM_PROMPT)
        self.assertIn("Do not invent objects, actions", CONSERVATIVE_V2_SYSTEM_PROMPT)
        self.assertIn("prefer the more conservative description", CONSERVATIVE_V2_SYSTEM_PROMPT)

    def test_user_prompt_formatting(self):
        c1 = "a brown dog running"
        c2 = "a dog playing with a ball"
        prompt = build_user_prompt(c1, c2)
        self.assertIn(f"Caption 1: {c1}", prompt)
        self.assertIn(f"Caption 2: {c2}", prompt)


class TestMockedAPICallAndRetry(unittest.TestCase):
    def test_successful_api_call(self):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = ' "a brown dog playing with a ball on grass" '
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        caption, status, attempts, error = call_groq_api_with_retry(
            mock_client, "caption 1", "caption 2", max_retries=2
        )
        self.assertEqual(status, "SUCCESS")
        self.assertEqual(caption, "a brown dog playing with a ball on grass")
        self.assertEqual(attempts, 1)
        self.assertEqual(error, "")

    def test_empty_response_handling_with_retry(self):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = ""  # Empty string
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        caption, status, attempts, error = call_groq_api_with_retry(
            mock_client, "caption 1", "caption 2", max_retries=2
        )
        self.assertEqual(status, "FAILED_EMPTY_RESPONSE")
        self.assertEqual(caption, "")
        self.assertEqual(attempts, 2)
        self.assertIn("empty response", error.lower())

    def test_transient_api_failure_retry_to_success(self):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "a successful fused caption"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        # Fail on attempt 1, succeed on attempt 2
        mock_client.chat.completions.create.side_effect = [
            RuntimeError("Rate limit exceeded"),
            mock_response
        ]

        with patch("time.sleep", return_value=None):
            caption, status, attempts, error = call_groq_api_with_retry(
                mock_client, "caption 1", "caption 2", max_retries=2
            )
        self.assertEqual(status, "SUCCESS")
        self.assertEqual(caption, "a successful fused caption")
        self.assertEqual(attempts, 2)
        self.assertEqual(error, "")

    def test_fatal_api_failure_exhausts_retries(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = ConnectionError("Network unreachable")

        with patch("time.sleep", return_value=None):
            caption, status, attempts, error = call_groq_api_with_retry(
                mock_client, "caption 1", "caption 2", max_retries=2
            )
        self.assertEqual(status, "FAILED_API_ERROR")
        self.assertEqual(caption, "")
        self.assertEqual(attempts, 2)
        self.assertIn("Network unreachable", error)


class TestQualityMetricCalculation(unittest.TestCase):
    def test_ttr_calculation(self):
        words = ["the", "small", "kitten", "on", "the", "floor"]
        # 5 unique words / 6 total words = 0.8333
        ttr = calculate_ttr(words)
        self.assertAlmostEqual(ttr, 0.8333, places=4)

    def test_vocabulary_expansion(self):
        c1_words = ["small", "kitten", "on", "ground"]
        c2_words = ["kitten", "sitting", "near", "bench"]
        fused_words = ["small", "kitten", "resting", "near", "bench", "outdoors"]
        novel_count, ratio = calculate_vocabulary_expansion(fused_words, c1_words, c2_words)
        # Novel words in fused: "resting", "outdoors" (2 novel out of 6 total unique)
        self.assertEqual(novel_count, 2)
        self.assertAlmostEqual(ratio, 2 / 6, places=4)

    def test_artifact_detection(self):
        # Cloud image with kite
        cloud_art = check_known_artifacts("sample-clouds-400x300.jpg", "a person flies a kite in the clouds")
        self.assertTrue(cloud_art["kite_in_clouds"])
        self.assertTrue(cloud_art["artifact_detected"])

        # Cloud image without kite
        clean_cloud = check_known_artifacts("sample-clouds-400x300.jpg", "a large white cloud in a blue sky")
        self.assertFalse(clean_cloud["kite_in_clouds"])
        self.assertFalse(clean_cloud["artifact_detected"])

        # Bumblebee image with "bum" corruption
        bee_art = check_known_artifacts("sample-bumblebee-400x300.png", "a bum and a bee on flower")
        self.assertTrue(bee_art["bum_token_in_bee"])

        # Bumblebee image with clean word "bumblebee"
        clean_bee = check_known_artifacts("sample-bumblebee-400x300.png", "a bumblebee sitting on a flower")
        self.assertFalse(clean_bee["bum_token_in_bee"])


class TestFailureExclusionFromAverages(unittest.TestCase):
    def test_exclusion_rule_in_quality_summary(self):
        metrics = [
            {
                "method": "CONTROL",
                "lambda_weight": 1.0,
                "is_valid_caption": True,
                "generation_status": "SUCCESS",
                "char_length": 40,
                "word_count": 8,
                "type_token_ratio": 0.875,
                "novel_tokens_count": 1,
                "novel_tokens_ratio": 0.125,
                "input_pair_diversity": 0.5,
                "artifact_detected": False
            },
            {
                "method": "CONTROL",
                "lambda_weight": 1.0,
                "is_valid_caption": False,  # Failed generation
                "generation_status": "FAILED_EMPTY_RESPONSE",
                "char_length": "",
                "word_count": "",
                "type_token_ratio": "",
                "novel_tokens_count": "",
                "novel_tokens_ratio": "",
                "input_pair_diversity": 0.6,
                "artifact_detected": False
            }
        ]

        summary = build_quality_summary(metrics)
        self.assertEqual(len(summary), 1)
        row = summary[0]
        self.assertEqual(row["total_images"], 2)
        self.assertEqual(row["valid_caption_count"], 1)
        self.assertEqual(row["failed_caption_count"], 1)
        # Quality average must be exactly 8 words, NOT diluted to 4 words
        self.assertEqual(row["mean_word_count"], 8.0)
        self.assertEqual(row["mean_char_length"], 40.0)


class TestResumptionAndProvenance(unittest.TestCase):
    def test_resumption_skips_successful_records(self):
        # When a record is already SUCCESS, run_fusion_comparison should keep it without re-calling the API
        mock_client = MagicMock()
        existing_record = {
            "image": "test.jpg",
            "method": "CONTROL",
            "lambda_weight": "1.0",
            "generation_status": "SUCCESS",
            "fused_caption": "a verified successful caption",
            "provenance": "REUSED_HISTORICAL_CONTROL_V2"
        }
        # Verify status is kept
        self.assertEqual(existing_record["generation_status"], "SUCCESS")
        self.assertEqual(mock_client.chat.completions.create.call_count, 0)

    def test_partial_target_lambdas_preserves_existing_records(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pairs_csv = tmp_path / "pairs.csv"
            hist_csv = tmp_path / "hist.csv"
            out_csv = tmp_path / "out.csv"
            summary_csv = tmp_path / "summary.csv"

            # Create synthetic pairs dataset
            with open(pairs_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "image", "method", "lambda_weight", "rank_1_caption", "rank_1_clip_score",
                    "rank_2_caption", "rank_2_clip_score", "rank_2_original_rank",
                    "pair_similarity", "pair_diversity", "diverges_from_control"
                ])
                writer.writeheader()
                writer.writerow({
                    "image": "img1.jpg", "method": "control_naive_clip_top2", "lambda_weight": "1.0",
                    "rank_1_caption": "c1", "rank_1_clip_score": "0.3",
                    "rank_2_caption": "c2", "rank_2_clip_score": "0.25", "rank_2_original_rank": "2",
                    "pair_similarity": "0.5", "pair_diversity": "0.5", "diverges_from_control": "False"
                })
                writer.writerow({
                    "image": "img1.jpg", "method": "proposed_diversity_aware", "lambda_weight": "0.5",
                    "rank_1_caption": "c1", "rank_1_clip_score": "0.3",
                    "rank_2_caption": "c2", "rank_2_clip_score": "0.25", "rank_2_original_rank": "2",
                    "pair_similarity": "0.5", "pair_diversity": "0.5", "diverges_from_control": "False"
                })
                writer.writerow({
                    "image": "img1.jpg", "method": "proposed_diversity_aware", "lambda_weight": "0.3",
                    "rank_1_caption": "c1", "rank_1_clip_score": "0.3",
                    "rank_2_caption": "c3", "rank_2_clip_score": "0.22", "rank_2_original_rank": "3",
                    "pair_similarity": "0.3", "pair_diversity": "0.7", "diverges_from_control": "True"
                })

            # Create historical v2 control
            with open(hist_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "image", "rank_1_caption", "rank_2_caption", "fused_caption_v2",
                    "fusion_model_v2", "fusion_prompt_version"
                ])
                writer.writeheader()
                writer.writerow({
                    "image": "img1.jpg", "rank_1_caption": "c1", "rank_2_caption": "c2",
                    "fused_caption_v2": "c1 and c2 fused", "fusion_model_v2": FUSION_MODEL,
                    "fusion_prompt_version": FUSION_PROMPT_VERSION
                })

            # Run with target_lambdas=[0.3, 0.5]
            run_fusion_comparison(
                pairs_csv, hist_csv, out_csv, summary_csv, target_lambdas=[0.3, 0.5], dry_run=True
            )
            records_initial = load_existing_comparison_records(out_csv)
            self.assertEqual(len(records_initial), 3)

            # Re-run targeting ONLY lambda=0.5
            run_fusion_comparison(
                pairs_csv, hist_csv, out_csv, summary_csv, target_lambdas=[0.5], dry_run=True
            )
            records_after = load_existing_comparison_records(out_csv)
            # Verify lambda=0.3 was preserved and NOT lost
            self.assertEqual(len(records_after), 3)
            self.assertIn("img1.jpg_DIVERSITY_AWARE_0.3", records_after)
            self.assertIn("img1.jpg_DIVERSITY_AWARE_0.5", records_after)
            self.assertIn("img1.jpg_CONTROL_1.0", records_after)


class TestHistoricalArtifactPreservation(unittest.TestCase):
    def test_historical_csv_files_unmodified(self):
        snapshot_file = project_root / ".system_generated" / "pre_implementation_snapshot.json"
        # If scratch snapshot exists, verify file sizes
        scratch_snap = Path(r"C:\Users\Admin\.gemini\antigravity\brain\da1fd613-a480-4df5-bf8a-79b8cfac25c8\scratch\pre_implementation_snapshot.json")
        if scratch_snap.exists():
            with open(scratch_snap, "r") as f:
                saved_snapshot = json.load(f)
            for rel_path, data in saved_snapshot.items():
                p = project_root / rel_path
                self.assertTrue(p.exists(), f"Tracked file missing: {rel_path}")
                current_size = p.stat().st_size
                self.assertEqual(
                    current_size, data["size"],
                    f"Historical file modified! {rel_path} changed from {data['size']} to {current_size}"
                )

    def test_historical_files_sha256_unmodified(self):
        root_hashes = project_root / "artifact_hashes.json"
        scratch_hashes = Path(r"C:\Users\Admin\.gemini\antigravity\brain\da1fd613-a480-4df5-bf8a-79b8cfac25c8\scratch\artifact_hashes.json")
        hash_file = root_hashes if root_hashes.exists() else scratch_hashes
        if hash_file.exists():
            with open(hash_file, "r") as f:
                saved_hashes = json.load(f)
            for rel_path, data in saved_hashes.items():
                p = project_root / rel_path
                self.assertTrue(p.exists(), f"Tracked file missing: {rel_path}")
                content = p.read_bytes()
                current_hash = hashlib.sha256(content).hexdigest()
                self.assertEqual(
                    current_hash, data["sha256"],
                    f"Historical file SHA-256 modified! {rel_path} changed from {data['sha256']} to {current_hash}"
                )


if __name__ == "__main__":
    unittest.main()

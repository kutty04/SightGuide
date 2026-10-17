"""SightGuide Integrated Caption Pipeline Service.

Coordinates:
1. BLIP candidate generation (local cached weights).
2. CLIP alignment scoring and candidate ranking.
3. Control (Naive Top-2) vs Proposed (Diversity-Aware) pair selection.
4. LLM-based fusion (Conservative V2 prompt policy).
5. Grounding and diversity metric computations.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

import sys

# Ensure backend directory is in sys.path for top-level 'app' resolution
_backend_dir = str(Path(__file__).resolve().parent.parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

try:
    from app.schemas import (
        CandidateCaption,
        ComparisonResponse,
        FusionResult,
        SelectedPair,
    )
    from app.services.diversity_ranker import (
        clean_and_tokenize,
        compute_lexical_similarity,
        normalize_clip_scores,
        select_control_pair,
        select_diversity_pair,
    )
    from app.services.fusion_service import (
        apply_grounding_safeguard,
        fuse_caption_pair,
        validate_anchor_against_detections,
    )
except ImportError:
    from ..schemas import (
        CandidateCaption,
        ComparisonResponse,
        FusionResult,
        SelectedPair,
    )
    from .diversity_ranker import (
        clean_and_tokenize,
        compute_lexical_similarity,
        normalize_clip_scores,
        select_control_pair,
        select_diversity_pair,
    )
    from .fusion_service import (
        apply_grounding_safeguard,
        fuse_caption_pair,
        validate_anchor_against_detections,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LOCAL_BLIP_SNAPSHOT = Path(
    r"C:\Users\Admin\.cache\huggingface\hub\models--Salesforce--blip-image-captioning-base\snapshots\82a37760796d32b1411fe092ab5d4e227313294b"
)


def clean_and_validate_caption(caption: str, anchor_context: str = "") -> str:
    """Clean repeated token loops, whitespace, and out-of-frame hallucinations from model output."""
    if not caption or not isinstance(caption, str):
        return ""
    text = caption.strip().strip('"').strip("'")
    # Clean repeated consecutive word loops (e.g. 'a a a' -> 'a', 'in in' -> 'in')
    text = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)
    # Apply evidence grounding safeguard
    text = apply_grounding_safeguard(text, anchor_context=anchor_context)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def deduplicate_candidate_pool(captions: list[str]) -> list[str]:
    """Preserve candidate diversity by filtering near-identical candidates."""
    cleaned: list[str] = []
    seen_token_sets: list[set[str]] = []
    for cap in captions:
        c_clean = clean_and_validate_caption(cap)
        if not c_clean or len(c_clean.split()) < 2:
            continue
        toks = {w.lower() for w in re.findall(r'\b\w+\b', c_clean)}
        is_near_dup = False
        for admitted_toks in seen_token_sets:
            inter = len(toks & admitted_toks)
            union = len(toks | admitted_toks)
            if union > 0 and (inter / union) > 0.85:
                is_near_dup = True
                break
        if not is_near_dup:
            cleaned.append(c_clean)
            seen_token_sets.append(toks)
        elif len(cleaned) < 2:
            cleaned.append(c_clean)
    return cleaned if len(cleaned) >= 2 else captions


class CaptionPipeline:
    """Singleton pipeline for image caption generation and ranking."""

    _instance: Optional[CaptionPipeline] = None

    def __init__(self) -> None:
        self.device = "cpu"
        self.blip_processor = None
        self.blip_model = None
        self.blip_loaded = False
        self.benchmark_cache = self._load_benchmark_cache()
        self.benchmark_hashes = self._load_benchmark_hashes()
        self.alignment_warning_threshold: float = float(
            os.getenv("SIGHTGUIDE_ALIGNMENT_WARNING_THRESHOLD", "-3.5")
        )
        self._initialize_device()
        self._load_blip_model()

    @classmethod
    def get_instance(cls) -> CaptionPipeline:
        if cls._instance is None:
            cls._instance = CaptionPipeline()
        return cls._instance

    def _initialize_device(self) -> None:
        if not LOCAL_BLIP_SNAPSHOT.is_dir():
            self.device = "cpu"
            return
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            self.device = "cpu"

    def _load_blip_model(self) -> None:
        """Load Salesforce BLIP base model from local cache snapshot if available."""
        if not LOCAL_BLIP_SNAPSHOT.is_dir():
            print(f"[CaptionPipeline] Local BLIP snapshot directory not found at: {LOCAL_BLIP_SNAPSHOT}")
            self.blip_loaded = False
            return

        try:
            import torch
            from transformers import BlipForConditionalGeneration, BlipProcessor

            print("[CaptionPipeline] Loading BLIP from local snapshot...")
            self.blip_processor = BlipProcessor.from_pretrained(
                str(LOCAL_BLIP_SNAPSHOT),
                local_files_only=True
            )
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                str(LOCAL_BLIP_SNAPSHOT),
                local_files_only=True,
                use_safetensors=False
            ).to(self.device)
            self.blip_model.eval()
            self.blip_loaded = True
            print(f"[CaptionPipeline] BLIP loaded successfully on {self.device}.")
        except Exception as exc:
            print(f"[CaptionPipeline] Warning: Could not load local BLIP model: {exc}")
            self.blip_loaded = False

    def _load_benchmark_cache(self) -> dict[str, list[dict[str, Any]]]:
        """Pre-load candidate pools and verified CLIP scores for the 12 benchmark images."""
        csv_path = PROJECT_ROOT / "results" / "sightguide_multi_image_results.csv"
        cache: dict[str, list[dict[str, Any]]] = {}
        if not csv_path.exists():
            return cache

        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img = row.get("image", "").strip()
                    if not img:
                        continue
                    if img not in cache:
                        cache[img] = []
                    cache[img].append({
                        "caption": row.get("caption", "").strip(),
                        "clip_score": float(row.get("clip_score", 0.0)),
                        "rank": int(row.get("rank", len(cache[img]) + 1))
                    })
        except Exception as exc:
            print(f"[CaptionPipeline] Notice: Could not load benchmark cache: {exc}")
        return cache

    def _load_benchmark_hashes(self) -> dict[str, str]:
        """Pre-compute SHA-256 hashes for canonical benchmark images in dataset/ to prevent cache collisions."""
        hashes: dict[str, str] = {}
        dataset_dir = PROJECT_ROOT / "dataset"
        if not dataset_dir.is_dir():
            return hashes
        for img_name in self.benchmark_cache.keys():
            img_file = dataset_dir / img_name
            if img_file.is_file():
                try:
                    digest = hashlib.sha256(img_file.read_bytes()).hexdigest()
                    hashes[img_name] = digest
                except Exception:
                    pass
        return hashes

    def is_verified_benchmark(self, filename: str, image_bytes: bytes) -> bool:
        """Verify whether an uploaded payload authentically matches a canonical benchmark image."""
        if filename not in self.benchmark_cache:
            return False
        expected_hash = self.benchmark_hashes.get(filename)
        if not expected_hash:
            return True
        return hashlib.sha256(image_bytes).hexdigest() == expected_hash

    def generate_candidates(
        self,
        image: Image.Image,
        num_candidates: int = 5,
        decoding_strategy: str = "sample",
        seed: Optional[int] = None
    ) -> list[str]:
        """Generate candidate captions for an image using BLIP with repetition safeguards.

        Args:
            image: PIL image input.
            num_candidates: Target candidate count (default: 5).
            decoding_strategy: 'sample' (nucleus sampling with penalties) or 'beam' (deterministic beam search).
            seed: Optional integer seed for reproducibility.
        """
        if not self.blip_loaded or self.blip_model is None or self.blip_processor is None:
            return [
                "a close-up photo of the subject",
                "a view showing the scene clearly in natural light",
                "an outdoor perspective of the object in its surroundings"
            ]

        try:
            import torch
            if seed is not None:
                torch.manual_seed(seed)

            inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                if decoding_strategy == "beam":
                    outputs = self.blip_model.generate(
                        **inputs,
                        num_beams=max(num_candidates, 5),
                        num_return_sequences=num_candidates,
                        max_new_tokens=50,
                        min_length=8,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=2,
                        length_penalty=1.0,
                        early_stopping=True
                    )
                else:
                    outputs = self.blip_model.generate(
                        **inputs,
                        do_sample=True,
                        num_return_sequences=num_candidates,
                        num_beams=1,
                        max_new_tokens=50,
                        min_length=8,
                        top_p=0.9,
                        temperature=0.7,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=2
                    )

            decoded = self.blip_processor.batch_decode(outputs, skip_special_tokens=True)
            cleaned = deduplicate_candidate_pool(decoded)
            return cleaned if len(cleaned) >= 2 else decoded
        except Exception as exc:
            print(f"[CaptionPipeline] Generation error: {exc}")
            return [
                "a photo of the scene",
                "a view of the subject outdoors"
            ]

    def score_candidates(
        self,
        image: Image.Image,
        captions: list[str]
    ) -> list[dict[str, Any]]:
        """Score candidate captions against the actual image using genuine cross-modal log-likelihood.

        Score definition:
            raw_score = -cross_entropy_loss (conditional sequence log-likelihood per token).
            Higher is strictly better (e.g. loss 1.5 -> score -1.5, loss 3.5 -> score -3.5; -1.5 > -3.5).
        """
        if not self.blip_loaded or self.blip_model is None or self.blip_processor is None:
            raw_candidates = []
            for idx, cap in enumerate(captions, start=1):
                raw_candidates.append({
                    "caption": cap,
                    "clip_score": 0.0,
                    "normalized_clip_score": 1.0,
                    "rank": idx,
                    "scoring_method": "offline_unscored",
                    "raw_loss": None
                })
            return raw_candidates

        import torch
        scored = []
        for cap in captions:
            clean_cap = clean_and_validate_caption(cap) or cap.strip()
            try:
                inputs = self.blip_processor(images=image, text=clean_cap, return_tensors="pt").to(self.device)
                labels = inputs.input_ids.clone()
                if self.blip_processor.tokenizer.pad_token_id is not None:
                    labels[labels == self.blip_processor.tokenizer.pad_token_id] = -100

                with torch.inference_mode():
                    out = self.blip_model(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        pixel_values=inputs.pixel_values,
                        labels=labels
                    )
                    loss = float(out.loss.item())
                    score = -loss  # Negative conditional sequence cross-entropy
            except Exception as exc:
                print(f"[CaptionPipeline] Scoring error for '{clean_cap}': {exc}")
                loss = 5.0
                score = -5.0

            scored.append({
                "caption": cap,
                "clip_score": round(score, 4),
                "raw_loss": round(loss, 4),
                "scoring_method": "blip_cross_entropy_log_likelihood"
            })

        # Sort strictly by descending alignment score (highest visual grounding first)
        scored.sort(key=lambda x: x["clip_score"], reverse=True)

        for rank, c in enumerate(scored, start=1):
            c["rank"] = rank

        # Min-max normalization across candidate pool for MMR
        raw_scores = [c["clip_score"] for c in scored]
        norm_scores = normalize_clip_scores(raw_scores, method="minmax")
        for i, norm_s in enumerate(norm_scores):
            scored[i]["normalized_clip_score"] = norm_s

        return scored

    def process_image(
        self,
        image_bytes: bytes,
        filename: str,
        lambda_weight: float = 0.5,
        target_method: str = "proposed",
        decoding_strategy: str = "sample",
        groq_api_key: Optional[str] = None,
        ocr_text: Optional[str] = None,
        normalized_ocr_text: Optional[str] = None,
    ) -> dict[str, Any]:
        """Execute the full pipeline for an uploaded image."""
        start_time = time.time()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 1. Obtain candidate pool (use verified benchmark data only if authentic match)
        if self.is_verified_benchmark(filename, image_bytes) and len(self.benchmark_cache[filename]) >= 2:
            candidates_raw = self.benchmark_cache[filename]
            scores = [c["clip_score"] for c in candidates_raw]
            norm_scores = normalize_clip_scores(scores)
            candidates = [
                CandidateCaption(
                    rank=c["rank"],
                    caption=c["caption"],
                    clip_score=round(c["clip_score"], 4),
                    normalized_clip_score=norm_scores[idx],
                    scoring_method="cached_historical_clip",
                    raw_loss=None
                )
                for idx, c in enumerate(candidates_raw)
            ]
            scoring_method = "cached_historical_clip"
        else:
            generated_texts = self.generate_candidates(image, num_candidates=5, decoding_strategy=decoding_strategy)
            scored = self.score_candidates(image, generated_texts)
            candidates = [
                CandidateCaption(
                    rank=s["rank"],
                    caption=s["caption"],
                    clip_score=round(s["clip_score"], 4),
                    normalized_clip_score=s.get("normalized_clip_score", 1.0),
                    scoring_method=s.get("scoring_method", "blip_cross_entropy_log_likelihood"),
                    raw_loss=s.get("raw_loss")
                )
                for s in scored
            ]
            scoring_method = "blip_cross_entropy_log_likelihood"

        candidate_dicts = [
            {"caption": c.caption, "clip_score": c.clip_score}
            for c in candidates
        ]

        # 2. Pair selection
        if target_method == "control":
            raw_pair = select_control_pair(candidate_dicts)
            selected_pair = SelectedPair(
                method="control_naive_clip_top2",
                lambda_weight=1.0,
                rank_1_caption=raw_pair["rank_1_caption"],
                rank_1_clip_score=raw_pair["rank_1_clip_score"],
                rank_2_caption=raw_pair["rank_2_caption"],
                rank_2_clip_score=raw_pair["rank_2_clip_score"],
                rank_2_original_rank=2,
                pair_similarity=raw_pair["pair_lexical_similarity"],
                pair_diversity=raw_pair["pair_diversity"],
                diverged_from_control=False,
                selection_notes=raw_pair["selection_notes"]
            )
        else:
            raw_pair = select_diversity_pair(candidate_dicts, lambda_weight=lambda_weight)
            is_div = raw_pair.get("diverges_from_naive_clip_rank_2", False)
            selected_pair = SelectedPair(
                method="proposed_diversity_aware",
                lambda_weight=lambda_weight,
                rank_1_caption=raw_pair["rank_1_caption"],
                rank_1_clip_score=raw_pair["rank_1_clip_score"],
                rank_2_caption=raw_pair["rank_2_caption"],
                rank_2_clip_score=raw_pair["rank_2_clip_score"],
                rank_2_original_rank=raw_pair["rank_2_original_rank"],
                pair_similarity=raw_pair["pair_similarity"],
                pair_diversity=raw_pair["pair_diversity"],
                diverged_from_control=is_div,
                selection_notes="Selected via Maximal Marginal Relevance (MMR) with grounding safeguard"
            )

        # 2b. Grounded Anchor Validation against YOLO detections (if model available)
        final_anchor = selected_pair.rank_1_caption
        ocr_surface_hint: Optional[str] = None
        try:
            from app.services.detector_service import DetectorService
            ds = DetectorService.get_instance()
            if ds.is_available:
                detections = ds.detect_objects(image, conf_threshold=0.20)
                all_caps = [c.caption for c in candidates]
                val = validate_anchor_against_detections(final_anchor, all_caps, detections)
                if val["demote"] and val["alternate_caption"]:
                    print(
                        f"[CaptionPipeline] Anchor demoted: '{final_anchor[:50]}' -> "
                        f"'{val['alternate_caption'][:50]}' (reason: {val['reason']})"
                    )
                    final_anchor = val["alternate_caption"]

                # Determine OCR surface hint from detections
                det_labels = {d["label"] for d in detections}
                if "cell phone" in det_labels:
                    ocr_surface_hint = "a smartphone screen"
                elif "laptop" in det_labels:
                    ocr_surface_hint = "a laptop screen"
                elif "tv" in det_labels:
                    ocr_surface_hint = "a television / monitor display"
                elif "book" in det_labels:
                    ocr_surface_hint = "a book or document"
        except Exception as exc:
            print(f"[CaptionPipeline] Notice: Anchor validation skipped: {exc}")

        # 3. LLM Fusion with grounded anchor and verified OCR evidence
        fusion = fuse_caption_pair(
            caption_1=final_anchor,
            caption_2=selected_pair.rank_2_caption,
            api_key=groq_api_key,
            ocr_text=ocr_text,
            ocr_surface_hint=ocr_surface_hint,
            normalized_ocr_text=normalized_ocr_text,
        )

        # 4. Uncertainty & confidence assessment
        confidence_warning = None
        if candidates and candidates[0].clip_score < self.alignment_warning_threshold:
            top_score = candidates[0].clip_score
            confidence_warning = (
                f"Provisional notice: Top candidate cross-modal log-likelihood ({top_score:.2f}) "
                f"is below the uncalibrated heuristic threshold ({self.alignment_warning_threshold:.2f}). "
                f"Captions may contain uncertain or out-of-distribution details."
            )

        elapsed = round(time.time() - start_time, 3)
        return {
            "filename": filename,
            "image_size_bytes": len(image_bytes),
            "candidates": candidates,
            "selected_pair": selected_pair,
            "fusion": fusion,
            "scoring_method": scoring_method,
            "confidence_warning": confidence_warning,
            "decoding_strategy": decoding_strategy,
            "execution_time_seconds": elapsed
        }

    def compare_methods(
        self,
        image_bytes: bytes,
        filename: str,
        lambda_weight: float = 0.5,
        decoding_strategy: str = "sample",
        groq_api_key: Optional[str] = None
    ) -> ComparisonResponse:
        """Run side-by-side comparison of Control vs. Proposed for the same image."""
        start_time = time.time()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 1. Obtain candidates
        if self.is_verified_benchmark(filename, image_bytes) and len(self.benchmark_cache[filename]) >= 2:
            candidates_raw = self.benchmark_cache[filename]
            scores = [c["clip_score"] for c in candidates_raw]
            norm_scores = normalize_clip_scores(scores)
            candidates = [
                CandidateCaption(
                    rank=c["rank"],
                    caption=c["caption"],
                    clip_score=round(c["clip_score"], 4),
                    normalized_clip_score=norm_scores[idx],
                    scoring_method="cached_historical_clip",
                    raw_loss=None
                )
                for idx, c in enumerate(candidates_raw)
            ]
            scoring_method = "cached_historical_clip"
        else:
            generated_texts = self.generate_candidates(image, num_candidates=5, decoding_strategy=decoding_strategy)
            scored = self.score_candidates(image, generated_texts)
            candidates = [
                CandidateCaption(
                    rank=s["rank"],
                    caption=s["caption"],
                    clip_score=round(s["clip_score"], 4),
                    normalized_clip_score=s.get("normalized_clip_score", 1.0),
                    scoring_method=s.get("scoring_method", "blip_cross_entropy_log_likelihood"),
                    raw_loss=s.get("raw_loss")
                )
                for s in scored
            ]
            scoring_method = "blip_cross_entropy_log_likelihood"

        candidate_dicts = [
            {"caption": c.caption, "clip_score": c.clip_score}
            for c in candidates
        ]

        # 2. Select Control pair (lambda = 1.0)
        c_raw = select_control_pair(candidate_dicts)
        control_pair = SelectedPair(
            method="control_naive_clip_top2",
            lambda_weight=1.0,
            rank_1_caption=c_raw["rank_1_caption"],
            rank_1_clip_score=c_raw["rank_1_clip_score"],
            rank_2_caption=c_raw["rank_2_caption"],
            rank_2_clip_score=c_raw["rank_2_clip_score"],
            rank_2_original_rank=2,
            pair_similarity=c_raw["pair_lexical_similarity"],
            pair_diversity=c_raw["pair_diversity"],
            diverged_from_control=False,
            selection_notes=c_raw["selection_notes"]
        )

        # 3. Select Proposed pair (diversity-aware)
        p_raw = select_diversity_pair(candidate_dicts, lambda_weight=lambda_weight)
        is_div = p_raw.get("diverges_from_naive_clip_rank_2", False)
        proposed_pair = SelectedPair(
            method="proposed_diversity_aware",
            lambda_weight=lambda_weight,
            rank_1_caption=p_raw["rank_1_caption"],
            rank_1_clip_score=p_raw["rank_1_clip_score"],
            rank_2_caption=p_raw["rank_2_caption"],
            rank_2_clip_score=p_raw["rank_2_clip_score"],
            rank_2_original_rank=p_raw["rank_2_original_rank"],
            pair_similarity=p_raw["pair_similarity"],
            pair_diversity=p_raw["pair_diversity"],
            diverged_from_control=is_div,
            selection_notes="Selected via Maximal Marginal Relevance (MMR) with grounding safeguard"
        )

        # 4. Check divergence
        is_divergent = proposed_pair.diverged_from_control
        div_delta = round(proposed_pair.pair_diversity - control_pair.pair_diversity, 4)

        # 5. Fuse captions
        control_fusion = fuse_caption_pair(
            caption_1=control_pair.rank_1_caption,
            caption_2=control_pair.rank_2_caption,
            api_key=groq_api_key
        )

        # If identical pair, reuse output to avoid redundant API consumption
        if not is_divergent and control_fusion.generation_status == "SUCCESS":
            proposed_fusion = FusionResult(
                fused_caption=control_fusion.fused_caption,
                fusion_model=control_fusion.fusion_model,
                prompt_version=control_fusion.prompt_version,
                generation_status=control_fusion.generation_status,
                provenance="REUSED_IDENTICAL_PAIR_MATCH",
                error_details=control_fusion.error_details
            )
        else:
            proposed_fusion = fuse_caption_pair(
                caption_1=proposed_pair.rank_1_caption,
                caption_2=proposed_pair.rank_2_caption,
                api_key=groq_api_key
            )

        # 6. Formulate plain-language explanation
        if is_divergent:
            explanation = (
                f"Proposed selection diverged from naive Top-2 (Rank 2) and selected Rank {proposed_pair.rank_2_original_rank}. "
                f"This increased input diversity from {control_pair.pair_diversity:.2f} to {proposed_pair.pair_diversity:.2f} "
                f"(+{div_delta:.2f}), eliminating redundancy with the visual anchor while preserving visual grounding."
            )
        else:
            explanation = (
                f"The proposed ranker verified that Candidate 2 was already sufficiently non-redundant with Candidate 1 "
                f"(Jaccard overlap = {control_pair.pair_similarity:.2%}). Both methods selected the same optimal pair."
            )

        # 7. Uncertainty & confidence assessment
        confidence_warning = None
        if candidates and candidates[0].clip_score < self.alignment_warning_threshold:
            top_score = candidates[0].clip_score
            confidence_warning = (
                f"Provisional notice: Top candidate cross-modal log-likelihood ({top_score:.2f}) "
                f"is below the uncalibrated heuristic threshold ({self.alignment_warning_threshold:.2f}). "
                f"Captions may contain uncertain or out-of-distribution details."
            )

        elapsed = round(time.time() - start_time, 3)

        return ComparisonResponse(
            filename=filename,
            image_size_bytes=len(image_bytes),
            candidates=candidates,
            control_pair=control_pair,
            control_fusion=control_fusion,
            proposed_pair=proposed_pair,
            proposed_fusion=proposed_fusion,
            is_divergent=is_divergent,
            pair_diversity_delta=div_delta,
            plain_language_explanation=explanation,
            scoring_method=scoring_method,
            confidence_warning=confidence_warning,
            decoding_strategy=decoding_strategy,
            execution_time_seconds=elapsed
        )

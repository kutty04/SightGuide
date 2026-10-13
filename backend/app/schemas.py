"""SightGuide Backend Pydantic Schemas.

Defines request and response data contracts for caption generation,
diversity-aware ranking, fusion, and benchmark inspection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status ('healthy')")
    version: str = Field("1.0.0", description="API version")
    blip_available: bool = Field(..., description="Whether BLIP captioning model weights are loaded")
    groq_configured: bool = Field(..., description="Whether GROQ_API_KEY is configured on the backend")
    active_device: str = Field(..., description="Active compute device ('cpu' or 'cuda')")


class CandidateCaption(BaseModel):
    rank: int = Field(..., description="Candidate rank (1-indexed)")
    caption: str = Field(..., description="Generated caption text")
    clip_score: float = Field(..., description="Raw CLIP image-text alignment score")
    normalized_clip_score: Optional[float] = Field(None, description="Normalized score in [0, 1]")


class SelectedPair(BaseModel):
    method: str = Field(..., description="Method name ('control_naive_clip_top2' or 'proposed_diversity_aware')")
    lambda_weight: float = Field(..., description="Grounding vs. diversity trade-off parameter lambda")
    rank_1_caption: str = Field(..., description="Primary visual anchor caption (c1*)")
    rank_1_clip_score: float = Field(..., description="Anchor raw CLIP score")
    rank_2_caption: str = Field(..., description="Selected secondary candidate caption (c2*)")
    rank_2_clip_score: float = Field(..., description="Secondary candidate raw CLIP score")
    rank_2_original_rank: int = Field(..., description="Original rank of secondary candidate in candidate pool")
    pair_similarity: float = Field(..., description="Lexical Jaccard similarity between c1 and c2")
    pair_diversity: float = Field(..., description="Complementary diversity (1 - similarity)")
    diverged_from_control: bool = Field(..., description="Whether selection diverged from naive Top-2")
    selection_notes: str = Field(..., description="Rationale for selection")


class FusionResult(BaseModel):
    fused_caption: str = Field(..., description="Final fused caption text")
    fusion_model: str = Field("openai/gpt-oss-20b", description="LLM model identifier")
    prompt_version: str = Field("conservative_v2", description="Prompt policy version")
    generation_status: str = Field(..., description="Generation status ('SUCCESS', 'GROQ_KEY_NOT_CONFIGURED', 'FAILED')")
    provenance: str = Field(..., description="Data provenance ('LIVE_GROQ_API_CALL', 'REUSED_HISTORICAL_CONTROL_V2', etc.)")
    error_details: Optional[str] = Field(None, description="Error explanation if unsuccessful")


class CaptionResponse(BaseModel):
    filename: str = Field(..., description="Name of uploaded image")
    image_size_bytes: int = Field(..., description="Size of uploaded image in bytes")
    candidates: List[CandidateCaption] = Field(..., description="Pool of generated candidate captions")
    selected_pair: SelectedPair = Field(..., description="Selected caption pair based on requested method")
    fusion: FusionResult = Field(..., description="LLM-fused caption output")
    execution_time_seconds: float = Field(..., description="Total pipeline execution time in seconds")


class ComparisonResponse(BaseModel):
    filename: str = Field(..., description="Name of uploaded image")
    image_size_bytes: int = Field(..., description="Size of uploaded image in bytes")
    candidates: List[CandidateCaption] = Field(..., description="Shared pool of candidate captions")
    control_pair: SelectedPair = Field(..., description="Naive CLIP Top-2 selected pair (lambda=1.0)")
    control_fusion: FusionResult = Field(..., description="Control fused caption")
    proposed_pair: SelectedPair = Field(..., description="Diversity-Aware selected pair (lambda=0.5)")
    proposed_fusion: FusionResult = Field(..., description="Proposed fused caption")
    is_divergent: bool = Field(..., description="Whether Proposed pair differs from Control pair")
    pair_diversity_delta: float = Field(..., description="Input diversity difference (Proposed - Control)")
    plain_language_explanation: str = Field(..., description="Plain language explanation of the selection dynamics")
    execution_time_seconds: float = Field(..., description="Total processing time in seconds")


class BenchmarkImageRecord(BaseModel):
    image: str
    image_path: str
    control_c1: str
    control_c2: str
    control_fused: str
    proposed_c1: str
    proposed_c2: str
    proposed_fused: str
    is_divergent: bool
    control_pair_diversity: float
    proposed_pair_diversity: float
    status: str


class BenchmarkSummaryResponse(BaseModel):
    total_images: int
    identical_pairs_count: int
    divergent_pairs_count: int
    control_mean_diversity: float
    proposed_mean_diversity: float
    mean_diversity_delta: float
    historical_failure_count: int
    pending_live_count: int
    benchmark_records: List[BenchmarkImageRecord]

/**
 * SightGuide API TypeScript Interfaces.
 * Matches backend Pydantic models in app.schemas.
 */

export interface HealthResponse {
  status: string;
  version: string;
  blip_available: boolean;
  groq_configured: boolean;
  active_device: string;
}

export interface CandidateCaption {
  rank: number;
  caption: string;
  clip_score: number;
  normalized_clip_score?: number;
}

export interface SelectedPair {
  method: string;
  lambda_weight: number;
  rank_1_caption: string;
  rank_1_clip_score: number;
  rank_2_caption: string;
  rank_2_clip_score: number;
  rank_2_original_rank: number;
  pair_similarity: number;
  pair_diversity: number;
  diverged_from_control: boolean;
  selection_notes: string;
}

export interface FusionResult {
  fused_caption: string;
  fusion_model: string;
  prompt_version: string;
  generation_status: 'SUCCESS' | 'GROQ_KEY_NOT_CONFIGURED' | 'FAILED' | 'DRY_RUN_DIVERGENT_PAIR' | 'FAILED_EMPTY_RESPONSE';
  provenance: string;
  error_details?: string;
}

export interface CaptionResponse {
  filename: string;
  image_size_bytes: number;
  candidates: CandidateCaption[];
  selected_pair: SelectedPair;
  fusion: FusionResult;
  execution_time_seconds: number;
}

export interface ComparisonResponse {
  filename: string;
  image_size_bytes: number;
  candidates: CandidateCaption[];
  control_pair: SelectedPair;
  control_fusion: FusionResult;
  proposed_pair: SelectedPair;
  proposed_fusion: FusionResult;
  is_divergent: boolean;
  pair_diversity_delta: number;
  plain_language_explanation: string;
  execution_time_seconds: number;
}

export interface BenchmarkImageRecord {
  image: string;
  image_path: string;
  control_c1: string;
  control_c2: string;
  control_fused: string;
  proposed_c1: string;
  proposed_c2: string;
  proposed_fused: string;
  is_divergent: boolean;
  control_pair_diversity: number;
  proposed_pair_diversity: number;
  status: string;
}

export interface BenchmarkSummaryResponse {
  total_images: number;
  identical_pairs_count: number;
  divergent_pairs_count: number;
  control_mean_diversity: number;
  proposed_mean_diversity: number;
  mean_diversity_delta: number;
  historical_failure_count: number;
  pending_live_count: number;
  benchmark_records: BenchmarkImageRecord[];
}

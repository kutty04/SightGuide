/**
 * SightGuide API TypeScript Interfaces.
 * Matches backend Pydantic models in app.schemas.
 */

export interface HealthResponse {
  status: string;
  version: string;
  blip_available: boolean;
  ocr_available?: boolean;
  groq_configured: boolean;
  active_device: string;
}

export interface CandidateCaption {
  rank: number;
  caption: string;
  clip_score: number;
  normalized_clip_score?: number;
  raw_loss?: number;
  scoring_method?: string;
  confidence_warning?: boolean;
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
  scoring_method?: string;
  decoding_strategy?: string;
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
  scoring_method?: string;
  decoding_strategy?: string;
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
  scoring_method?: string;
  decoding_strategy?: string;
  confidence_warning?: string | null;
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

export interface OCRLine {
  text: string;
  confidence: number;
  box?: number[][];
  reading_order?: number;
}

export interface OCRResponse {
  filename: string;
  text: string;
  normalized_text?: string | null;
  has_text: boolean;
  lines: OCRLine[];
  message?: string;
  engine: string;
  execution_time_seconds: number;
}

export interface DecodingComparisonRow {
  image_id: string;
  category: string;
  split: string;
  decoding_strategy: string;
  total_time_seconds: string | number;
  generation_time_seconds: string | number;
  scoring_time_seconds: string | number;
  candidate_pool_size: string | number;
  pool_diversity: string | number;
  pool_distinct_1_ratio?: string | number;
  pool_distinct_2_ratio?: string | number;
  pool_avg_caption_len?: string | number;
  pool_vocab_size?: string | number;
  pool_repetition_rate: string | number;
  anchor_caption: string;
  anchor_score: string | number;
  control_c2: string;
  control_score: string | number;
  proposed_c2: string;
  proposed_score: string | number;
  proposed_rank2_original_rank: string | number;
  is_divergent: string | boolean;
  pair_diversity_delta: string | number;
  scoring_method: string;
}

export interface DecodingComparisonMetadata {
  experiment_name: string;
  timestamp_utc: string;
  total_images_evaluated: number;
  model_checkpoint: string;
  scoring_method: string;
  hardware_environment: string;
  methodological_notes?: {
    repetition_rate: string;
    unconstrained_diversity_metrics: string;
  };
  results_by_strategy: {
    sample: {
      mean_latency_seconds: number;
      mean_pool_diversity: number;
      mean_distinct_1_ratio?: number;
      mean_distinct_2_ratio?: number;
      mean_caption_length?: number;
      mean_repetition_rate: number;
      divergent_pairs_count: number;
      divergence_percentage: number;
    };
    beam: {
      mean_latency_seconds: number;
      mean_pool_diversity: number;
      mean_distinct_1_ratio?: number;
      mean_distinct_2_ratio?: number;
      mean_caption_length?: number;
      mean_repetition_rate: number;
      divergent_pairs_count: number;
      divergence_percentage: number;
    };
  };
}

export interface ResultsComparisonResponse {
  metadata: DecodingComparisonMetadata;
  rows: DecodingComparisonRow[];
  total_records: number;
  source_file: string;
}

export interface DatasetManifestRecord {
  image_id: string;
  filename: string;
  category: string;
  split: string;
  dimensions: string;
  sha256: string;
  source: string;
  license: string;
  annotation_status: string;
}

export interface DatasetManifestResponse {
  total_images: number;
  categories: Record<string, number>;
  splits: Record<string, number>;
  records: DatasetManifestRecord[];
}

export interface QAHistoryItem {
  question: string;
  answer: string;
}

export interface QARequest {
  question: string;
  scene_caption?: string | null;
  ocr_text?: string | null;
  history?: QAHistoryItem[];
  image_data?: string | null;
}

export interface QAResponse {
  question: string;
  answer: string;
  capability: string;
  is_uncertain: boolean;
  grounded_sources: string[];
  evidence_used: string[];
  limitation_notice?: string;
  model_name?: string | null;
  execution_time_seconds: number;
}

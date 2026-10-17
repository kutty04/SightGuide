import type {
  BenchmarkSummaryResponse,
  CaptionResponse,
  ComparisonResponse,
  HealthResponse,
  OCRResponse,
  ResultsComparisonResponse,
  DatasetManifestResponse,
  QARequest,
  QAResponse,
} from '../types/api';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export async function fetchHealth(): Promise<HealthResponse> {
  const resp = await fetch(`${API_BASE}/api/health`);
  if (!resp.ok) {
    throw new Error(`Health check failed: HTTP ${resp.status}`);
  }
  return resp.json();
}

export async function generateCaption(
  file: File,
  lambdaWeight: number = 0.5,
  method: 'proposed' | 'control' = 'proposed',
  decodingStrategy: 'sample' | 'beam' = 'sample',
  ocrText?: string | null,
  normalizedOcrText?: string | null
): Promise<CaptionResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('lambda_weight', lambdaWeight.toString());
  formData.append('method', method);
  formData.append('decoding_strategy', decodingStrategy);
  if (ocrText && ocrText.trim()) {
    formData.append('ocr_text', ocrText.trim());
  }
  if (normalizedOcrText && normalizedOcrText.trim()) {
    formData.append('normalized_ocr_text', normalizedOcrText.trim());
  }

  const resp = await fetch(`${API_BASE}/api/caption/generate`, {
    method: 'POST',
    body: formData,
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }));
    throw new Error(err.detail || 'Caption generation failed.');
  }

  return resp.json();
}

export async function compareCaptions(
  file: File,
  lambdaWeight: number = 0.5,
  decodingStrategy: 'sample' | 'beam' = 'sample'
): Promise<ComparisonResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('lambda_weight', lambdaWeight.toString());
  formData.append('decoding_strategy', decodingStrategy);

  const resp = await fetch(`${API_BASE}/api/caption/compare`, {
    method: 'POST',
    body: formData,
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }));
    throw new Error(err.detail || 'Caption comparison failed.');
  }

  return resp.json();
}

export async function fetchBenchmarkSummary(): Promise<BenchmarkSummaryResponse> {
  const resp = await fetch(`${API_BASE}/api/benchmark/summary`);
  if (!resp.ok) {
    throw new Error(`Failed to load benchmark summary: HTTP ${resp.status}`);
  }
  return resp.json();
}

export function getBenchmarkImageUrl(imageName: string): string {
  return `${API_BASE}/api/benchmark/image/${encodeURIComponent(imageName)}`;
}

export async function extractOCR(file: File): Promise<OCRResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const resp = await fetch(`${API_BASE}/api/ocr/extract`, {
    method: 'POST',
    body: formData,
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }));
    throw new Error(err.detail || 'OCR extraction failed.');
  }

  return resp.json();
}

export async function fetchDecodingComparison(): Promise<ResultsComparisonResponse> {
  const resp = await fetch(`${API_BASE}/api/results/decoding-comparison`);
  if (!resp.ok) {
    throw new Error(`Failed to load decoding comparison: HTTP ${resp.status}`);
  }
  return resp.json();
}

export async function fetchDatasetManifest(): Promise<DatasetManifestResponse> {
  const resp = await fetch(`${API_BASE}/api/results/manifest`);
  if (!resp.ok) {
    throw new Error(`Failed to load dataset manifest: HTTP ${resp.status}`);
  }
  return resp.json();
}

export async function askQuestion(request: QARequest): Promise<QAResponse> {
  const resp = await fetch(`${API_BASE}/api/qa/ask`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }));
    throw new Error(err.detail || 'Visual Q&A request failed.');
  }

  return resp.json();
}

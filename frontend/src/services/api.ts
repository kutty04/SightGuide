import type {
  BenchmarkSummaryResponse,
  CaptionResponse,
  ComparisonResponse,
  HealthResponse,
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
  method: 'proposed' | 'control' = 'proposed'
): Promise<CaptionResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('lambda_weight', lambdaWeight.toString());
  formData.append('method', method);

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
  lambdaWeight: number = 0.5
): Promise<ComparisonResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('lambda_weight', lambdaWeight.toString());

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

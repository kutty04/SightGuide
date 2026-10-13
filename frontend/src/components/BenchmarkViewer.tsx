import React, { useEffect, useState } from 'react';
import type { BenchmarkImageRecord, BenchmarkSummaryResponse } from '../types/api';
import { fetchBenchmarkSummary, getBenchmarkImageUrl } from '../services/api';
import { CaptionPlayer } from './CaptionPlayer';
import {
  AlertCircle,
  Database,
  ExternalLink,
  Sparkles,
} from 'lucide-react';

export const BenchmarkViewer: React.FC = () => {
  const [summary, setSummary] = useState<BenchmarkSummaryResponse | null>(null);
  const [selectedRecord, setSelectedRecord] = useState<BenchmarkImageRecord | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    fetchBenchmarkSummary()
      .then((data) => {
        if (mounted) {
          setSummary(data);
          if (data.benchmark_records.length > 0) {
            // Default select the first divergent record to showcase the research contribution
            const divRec = data.benchmark_records.find((r) => r.is_divergent) || data.benchmark_records[0];
            setSelectedRecord(divRec);
          }
          setLoading(false);
        }
      })
      .catch((err) => {
        if (mounted) {
          setError(err.message || 'Failed to load benchmark data.');
          setLoading(false);
        }
      });

    return () => {
      mounted = false;
    };
  }, []);

  if (loading) {
    return (
      <div className="p-12 text-center text-slate-400 space-y-3">
        <div className="w-6 h-6 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin mx-auto" />
        <p className="text-sm">Loading 12-image Springer Replication Benchmark...</p>
      </div>
    );
  }

  if (error || !summary) {
    return (
      <div className="p-6 bg-red-950/40 border border-red-800 text-red-200 rounded-2xl flex items-center gap-3">
        <AlertCircle className="w-5 h-5 flex-shrink-0" />
        <p className="text-sm">{error || 'Benchmark records could not be retrieved.'}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Benchmark Summary Stats Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">
            Benchmark Images
          </span>
          <span className="text-2xl font-bold font-mono text-slate-100 mt-1 block">
            {summary.total_images}
          </span>
          <span className="text-[11px] text-slate-400 mt-0.5 block">100% Curated Dataset</span>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">
            Identical Pairs
          </span>
          <span className="text-2xl font-bold font-mono text-slate-300 mt-1 block">
            {summary.identical_pairs_count} / {summary.total_images}
          </span>
          <span className="text-[11px] text-slate-400 mt-0.5 block">
            {((summary.identical_pairs_count / summary.total_images) * 100).toFixed(0)}% non-redundant
          </span>
        </div>

        <div className="bg-slate-900/80 border border-emerald-900/60 rounded-xl p-4">
          <span className="text-xs font-semibold text-emerald-400 uppercase tracking-wider block">
            Divergent Pairs
          </span>
          <span className="text-2xl font-bold font-mono text-emerald-300 mt-1 block">
            {summary.divergent_pairs_count} / {summary.total_images}
          </span>
          <span className="text-[11px] text-emerald-400/80 mt-0.5 block">
            Kitten &amp; Bumblebee cases
          </span>
        </div>

        <div className="bg-slate-900/80 border border-indigo-900/60 rounded-xl p-4">
          <span className="text-xs font-semibold text-indigo-400 uppercase tracking-wider block">
            Diversity Delta
          </span>
          <span className="text-2xl font-bold font-mono text-indigo-300 mt-1 block">
            +{(summary.mean_diversity_delta * 100).toFixed(1)}%
          </span>
          <span className="text-[11px] text-indigo-400/80 mt-0.5 block">
            Input complementarity gain
          </span>
        </div>
      </div>

      {/* Main Interactive Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Image Selector List */}
        <div className="lg:col-span-5 bg-slate-900/80 border border-slate-800 rounded-2xl p-4 space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5 text-indigo-400" />
              Benchmark Images (N=12)
            </h3>
            <span className="text-[11px] font-mono text-slate-400">Click to inspect</span>
          </div>

          <div className="space-y-2 max-h-[32rem] overflow-y-auto pr-1">
            {summary.benchmark_records.map((rec) => {
              const isSelected = selectedRecord?.image === rec.image;
              return (
                <button
                  type="button"
                  key={rec.image}
                  onClick={() => setSelectedRecord(rec)}
                  className={`w-full text-left p-2.5 rounded-xl border flex items-center gap-3 transition-all ${
                    isSelected
                      ? 'bg-indigo-950/50 border-indigo-500/80 ring-1 ring-indigo-500/40'
                      : 'bg-slate-950/40 border-slate-800/80 hover:border-slate-700'
                  }`}
                >
                  <img
                    src={getBenchmarkImageUrl(rec.image)}
                    alt={rec.image}
                    className="w-12 h-12 rounded-lg object-cover bg-slate-900 flex-shrink-0 border border-slate-800"
                    loading="lazy"
                  />
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-medium text-slate-200 truncate">{rec.image}</p>
                    <div className="flex items-center gap-2 mt-1">
                      {rec.is_divergent ? (
                        <span className="text-[10px] bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 px-1.5 py-0.5 rounded font-mono font-semibold">
                          DIVERGENT
                        </span>
                      ) : (
                        <span className="text-[10px] bg-slate-800 text-slate-400 border border-slate-700 px-1.5 py-0.5 rounded font-mono">
                          IDENTICAL
                        </span>
                      )}
                      <span className="text-[10px] font-mono text-slate-400">
                        Div: {(rec.proposed_pair_diversity * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Right Column: Detailed Selected Benchmark Record */}
        {selectedRecord && (
          <div className="lg:col-span-7 bg-slate-900/90 border border-slate-800 rounded-2xl p-5 space-y-4 shadow-xl">
            <div className="flex items-start justify-between gap-3 border-b border-slate-800 pb-3">
              <div>
                <h4 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                  <span>{selectedRecord.image}</span>
                  {selectedRecord.is_divergent ? (
                    <span className="text-xs bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 px-2 py-0.5 rounded-full font-mono">
                      MMR Divergent Pair
                    </span>
                  ) : (
                    <span className="text-xs bg-slate-800 text-slate-400 border border-slate-700 px-2 py-0.5 rounded-full font-mono">
                      Exact Match Reused
                    </span>
                  )}
                </h4>
                <p className="text-xs text-slate-400 mt-0.5">
                  Controlled empirical test at &lambda; = 0.5 vs. Naive CLIP Top-2
                </p>
              </div>

              <a
                href={getBenchmarkImageUrl(selectedRecord.image)}
                target="_blank"
                rel="noreferrer"
                className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1 flex-shrink-0"
              >
                <span>Full Image</span>
                <ExternalLink className="w-3.5 h-3.5" />
              </a>
            </div>

            {/* Image Preview & Captions Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="sm:col-span-1">
                <img
                  src={getBenchmarkImageUrl(selectedRecord.image)}
                  alt={selectedRecord.image}
                  className="w-full h-44 object-cover rounded-xl border border-slate-800 bg-black"
                />
              </div>

              <div className="sm:col-span-2 space-y-2 text-xs">
                <div className="p-2.5 bg-slate-950/60 rounded-lg border border-slate-800/80 space-y-1">
                  <span className="text-[10px] font-bold text-indigo-400 uppercase tracking-wider block">
                    Control Candidate Pair (&lambda;=1.0)
                  </span>
                  <p className="text-slate-300"><strong className="text-slate-400">c1:</strong> "{selectedRecord.control_c1}"</p>
                  <p className="text-slate-300"><strong className="text-slate-400">c2:</strong> "{selectedRecord.control_c2}"</p>
                  <p className="text-[11px] font-mono text-slate-400 pt-1 border-t border-slate-800">
                    Pair Diversity: {(selectedRecord.control_pair_diversity * 100).toFixed(1)}%
                  </p>
                </div>

                <div className="p-2.5 bg-emerald-950/20 rounded-lg border border-emerald-900/50 space-y-1">
                  <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider block">
                    Proposed Candidate Pair (&lambda;=0.5)
                  </span>
                  <p className="text-slate-300"><strong className="text-slate-400">c1*:</strong> "{selectedRecord.proposed_c1}"</p>
                  <p className="text-slate-300"><strong className="text-slate-400">c2*:</strong> "{selectedRecord.proposed_c2}"</p>
                  <p className="text-[11px] font-mono text-emerald-400 pt-1 border-t border-emerald-900/40">
                    Pair Diversity: {(selectedRecord.proposed_pair_diversity * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
            </div>

            {/* Fused Output Comparison */}
            <div className="pt-2 border-t border-slate-800 space-y-3">
              <h5 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
                Fused Outputs Comparison
              </h5>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 space-y-1.5">
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
                    Control Output
                  </span>
                  <p className="text-slate-200 font-medium italic min-h-[2.5rem]">
                    {selectedRecord.control_fused ? `"${selectedRecord.control_fused}"` : '[FAILED_EMPTY_RESPONSE]'}
                  </p>
                  {selectedRecord.control_fused && (
                    <CaptionPlayer captionText={selectedRecord.control_fused} label="Control Audio" />
                  )}
                </div>

                <div className="p-3 bg-slate-950 rounded-xl border border-emerald-950 space-y-1.5">
                  <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider block">
                    Proposed Output
                  </span>
                  <p className="text-slate-200 font-medium italic min-h-[2.5rem]">
                    {selectedRecord.proposed_fused
                      ? `"${selectedRecord.proposed_fused}"`
                      : selectedRecord.status === 'DRY_RUN_DIVERGENT_PAIR'
                      ? '[Pending Live Groq API Execution]'
                      : '[FAILED_EMPTY_RESPONSE]'}
                  </p>
                  {selectedRecord.proposed_fused && (
                    <CaptionPlayer captionText={selectedRecord.proposed_fused} label="Proposed Audio" />
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

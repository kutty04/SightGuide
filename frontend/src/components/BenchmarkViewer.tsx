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
      <div className="p-12 text-center text-zinc-400 font-mono text-xs space-y-3">
        <div className="w-6 h-6 border-2 border-[#D4FF00] border-t-transparent rounded-full animate-spin mx-auto" />
        <p className="tracking-widest uppercase">Loading 12-image Springer Replication Benchmark...</p>
      </div>
    );
  }

  if (error || !summary) {
    return (
      <div className="p-6 bg-red-950/70 border border-red-700 text-red-200 text-xs font-mono flex items-center gap-3">
        <AlertCircle className="w-5 h-5 flex-shrink-0 text-red-400" />
        <p>{error || 'Benchmark records could not be retrieved.'}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Benchmark Summary Stats Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="bg-[#0c0c0c] border border-white/10 p-4">
          <span className="text-[10px] font-mono font-bold text-zinc-400 uppercase tracking-widest block">
            BENCHMARK IMAGES
          </span>
          <span className="text-3xl font-display font-bold text-white mt-1 block">
            {summary.total_images}
          </span>
          <span className="text-[10px] font-mono text-zinc-500 mt-0.5 block uppercase">100% Curated Dataset</span>
        </div>

        <div className="bg-[#0c0c0c] border border-white/10 p-4">
          <span className="text-[10px] font-mono font-bold text-zinc-400 uppercase tracking-widest block">
            IDENTICAL PAIRS
          </span>
          <span className="text-3xl font-display font-bold text-zinc-300 mt-1 block">
            {summary.identical_pairs_count} / {summary.total_images}
          </span>
          <span className="text-[10px] font-mono text-zinc-500 mt-0.5 block uppercase">
            {((summary.identical_pairs_count / summary.total_images) * 100).toFixed(0)}% Non-Redundant
          </span>
        </div>

        <div className="bg-[#0c0c0c] border border-[#D4FF00] p-4 shadow-[0_0_15px_rgba(212,255,0,0.12)]">
          <span className="text-[10px] font-mono font-bold text-[#D4FF00] uppercase tracking-widest block">
            DIVERGENT PAIRS
          </span>
          <span className="text-3xl font-display font-bold text-[#D4FF00] mt-1 block">
            {summary.divergent_pairs_count} / {summary.total_images}
          </span>
          <span className="text-[10px] font-mono text-zinc-400 mt-0.5 block uppercase">
            Kitten &amp; Bumblebee Cases
          </span>
        </div>

        <div className="bg-[#0c0c0c] border border-white/10 p-4">
          <span className="text-[10px] font-mono font-bold text-zinc-400 uppercase tracking-widest block">
            DIVERSITY DELTA
          </span>
          <span className="text-3xl font-display font-bold text-white mt-1 block">
            +{(summary.mean_diversity_delta * 100).toFixed(1)}%
          </span>
          <span className="text-[10px] font-mono text-zinc-500 mt-0.5 block uppercase">
            Complementarity Gain
          </span>
        </div>
      </div>

      {/* Main Interactive Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Image Selector List */}
        <div className="lg:col-span-5 bg-[#0c0c0c] border border-white/10 p-4 space-y-3 reticle-bracket">
          <div className="flex items-center justify-between pb-2 border-b border-white/10">
            <h3 className="text-xs font-mono font-bold uppercase tracking-widest text-[#D4FF00] flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5 text-[#D4FF00]" />
              <span>BENCHMARK IMAGES (N=12)</span>
            </h3>
            <span className="text-[10px] font-mono text-zinc-500 uppercase">CLICK TO INSPECT</span>
          </div>

          <div className="space-y-2 max-h-[32rem] overflow-y-auto pr-1">
            {summary.benchmark_records.map((rec) => {
              const isSelected = selectedRecord?.image === rec.image;
              return (
                <button
                  type="button"
                  key={rec.image}
                  onClick={() => setSelectedRecord(rec)}
                  className={`w-full text-left p-2.5 border flex items-center gap-3 transition-all ${
                    isSelected
                      ? 'bg-black border-[#D4FF00] shadow-[0_0_10px_rgba(212,255,0,0.2)]'
                      : 'bg-black border-white/15 hover:border-[#D4FF00]/50'
                  }`}
                >
                  <img
                    src={getBenchmarkImageUrl(rec.image)}
                    alt={rec.image}
                    className="w-12 h-12 object-cover bg-black flex-shrink-0 border border-white/10"
                    loading="lazy"
                  />
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-mono text-white truncate">{rec.image}</p>
                    <div className="flex items-center gap-2 mt-1">
                      {rec.is_divergent ? (
                        <span className="text-[9px] bg-[#D4FF00] text-black px-1.5 py-0.2 font-mono font-bold uppercase">
                          DIVERGENT
                        </span>
                      ) : (
                        <span className="text-[9px] bg-black text-zinc-400 border border-white/20 px-1.5 py-0.2 font-mono uppercase">
                          IDENTICAL
                        </span>
                      )}
                      <span className="text-[10px] font-mono text-zinc-400">
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
          <div className="lg:col-span-7 bg-[#0c0c0c] border border-white/10 p-5 space-y-4 shadow-xl reticle-bracket">
            <div className="flex items-start justify-between gap-3 border-b border-white/10 pb-3">
              <div>
                <h4 className="text-sm font-mono font-bold text-white flex items-center gap-2">
                  <span>{selectedRecord.image}</span>
                  {selectedRecord.is_divergent ? (
                    <span className="text-[10px] bg-black text-[#D4FF00] border border-[#D4FF00] px-2 py-0.5 font-mono uppercase font-bold">
                      MMR Divergent Pair
                    </span>
                  ) : (
                    <span className="text-[10px] bg-black text-zinc-400 border border-white/20 px-2 py-0.5 font-mono uppercase">
                      Exact Match Reused
                    </span>
                  )}
                </h4>
                <p className="text-xs text-zinc-400 mt-0.5 font-sans">
                  Controlled empirical test at &lambda; = 0.5 vs. Naive CLIP Top-2
                </p>
              </div>

              <a
                href={getBenchmarkImageUrl(selectedRecord.image)}
                target="_blank"
                rel="noreferrer"
                className="text-xs font-mono text-[#D4FF00] hover:underline flex items-center gap-1 flex-shrink-0"
              >
                <span>FULL IMAGE</span>
                <ExternalLink className="w-3.5 h-3.5" />
              </a>
            </div>

            {/* Image Preview & Captions Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="sm:col-span-1">
                <img
                  src={getBenchmarkImageUrl(selectedRecord.image)}
                  alt={selectedRecord.image}
                  className="w-full h-44 object-cover border border-white/10 bg-black"
                />
              </div>

              <div className="sm:col-span-2 space-y-2 text-xs">
                <div className="p-3 bg-black border border-white/10 space-y-1 font-sans">
                  <span className="text-[10px] font-mono font-bold text-zinc-400 uppercase tracking-wider block">
                    Control Candidate Pair (&lambda;=1.0)
                  </span>
                  <p className="text-zinc-300"><strong className="text-zinc-500 font-mono">c1:</strong> &ldquo;{selectedRecord.control_c1}&rdquo;</p>
                  <p className="text-zinc-300"><strong className="text-zinc-500 font-mono">c2:</strong> &ldquo;{selectedRecord.control_c2}&rdquo;</p>
                  <p className="text-[11px] font-mono text-zinc-400 pt-1 border-t border-white/10">
                    Pair Diversity: {(selectedRecord.control_pair_diversity * 100).toFixed(1)}%
                  </p>
                </div>

                <div className="p-3 bg-black border border-[#D4FF00] space-y-1 font-sans shadow-[0_0_10px_rgba(212,255,0,0.1)]">
                  <span className="text-[10px] font-mono font-bold text-[#D4FF00] uppercase tracking-wider block">
                    Proposed Candidate Pair (&lambda;=0.5)
                  </span>
                  <p className="text-zinc-200"><strong className="text-[#D4FF00] font-mono">c1*:</strong> &ldquo;{selectedRecord.proposed_c1}&rdquo;</p>
                  <p className="text-zinc-200"><strong className="text-[#D4FF00] font-mono">c2*:</strong> &ldquo;{selectedRecord.proposed_c2}&rdquo;</p>
                  <p className="text-[11px] font-mono text-[#D4FF00] pt-1 border-t border-white/10">
                    Pair Diversity: {(selectedRecord.proposed_pair_diversity * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
            </div>

            {/* Fused Output Comparison */}
            <div className="pt-2 border-t border-white/10 space-y-3">
              <h5 className="text-xs font-mono font-bold uppercase tracking-wider text-white flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-[#D4FF00]" />
                <span>FUSED OUTPUTS COMPARISON</span>
              </h5>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                <div className="p-3 bg-black border border-white/10 space-y-1.5">
                  <span className="text-[10px] font-mono font-bold text-zinc-400 uppercase tracking-wider block">
                    Control Output
                  </span>
                  <p className="text-zinc-200 font-medium italic min-h-[2.5rem] font-sans">
                    {selectedRecord.control_fused ? `"${selectedRecord.control_fused}"` : '[FAILED_EMPTY_RESPONSE]'}
                  </p>
                  {selectedRecord.control_fused && (
                    <CaptionPlayer captionText={selectedRecord.control_fused} label="Control Audio" />
                  )}
                </div>

                <div className="p-3 bg-black border border-[#D4FF00] space-y-1.5 shadow-[0_0_10px_rgba(212,255,0,0.1)]">
                  <span className="text-[10px] font-mono font-bold text-[#D4FF00] uppercase tracking-wider block">
                    Proposed Output
                  </span>
                  <p className="text-white font-medium italic min-h-[2.5rem] font-sans">
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

export default BenchmarkViewer;

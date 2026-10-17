import React, { useState, useEffect } from 'react';
import { compareCaptions, fetchBenchmarkSummary, getBenchmarkImageUrl } from '../services/api';
import type { ComparisonResponse, BenchmarkSummaryResponse } from '../types/api';
import { AppNavbar } from '../components/layout/AppNavbar';
import { AppFooter } from '../components/layout/AppFooter';
import { ImageUploader } from '../components/ImageUploader';
import { CaptionPlayer } from '../components/CaptionPlayer';
import {
  FlaskConical,
  Calculator,
  GitCompare,
  Database,
  Sliders,
  AlertCircle,
  TrendingUp,
} from 'lucide-react';

export const ExperimentLab: React.FC = () => {
  const [lambdaWeight, setLambdaWeight] = useState<number>(0.5);
  const [decodingStrategy, setDecodingStrategy] = useState<'sample' | 'beam'>('sample');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Benchmark images state
  const [benchmarkSummary, setBenchmarkSummary] = useState<BenchmarkSummaryResponse | null>(null);
  const [selectedBenchmarkImage, setSelectedBenchmarkImage] = useState<string | null>(null);

  useEffect(() => {
    fetchBenchmarkSummary()
      .then(setBenchmarkSummary)
      .catch((err) => console.warn('Could not load benchmark summary:', err));
  }, []);

  const handleImageSelected = async (file: File, lambda: number) => {
    setIsLoading(true);
    setErrorMsg(null);
    setSelectedBenchmarkImage(null);
    try {
      const result = await compareCaptions(file, lambda, decodingStrategy);
      setComparison(result);
    } catch (err: any) {
      setErrorMsg(err.message || 'Comparison failed.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectBenchmark = async (imageName: string) => {
    setIsLoading(true);
    setErrorMsg(null);
    setSelectedBenchmarkImage(imageName);
    try {
      const url = getBenchmarkImageUrl(imageName);
      const res = await fetch(url);
      const blob = await res.blob();
      const file = new File([blob], imageName, { type: 'image/jpeg' });
      const result = await compareCaptions(file, lambdaWeight, decodingStrategy);
      setComparison(result);
    } catch (err: any) {
      setErrorMsg(err.message || `Failed to evaluate benchmark image ${imageName}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Helper to compute token Jaccard similarity
  const computeJaccard = (s1: string, s2: string): number => {
    const t1 = new Set(s1.toLowerCase().replace(/[^a-z0-9 ]/g, '').split(/\s+/).filter(Boolean));
    const t2 = new Set(s2.toLowerCase().replace(/[^a-z0-9 ]/g, '').split(/\s+/).filter(Boolean));
    if (t1.size === 0 && t2.size === 0) return 1.0;
    const intersection = new Set([...t1].filter((x) => t2.has(x)));
    const union = new Set([...t1, ...t2]);
    return union.size === 0 ? 0 : intersection.size / union.size;
  };

  return (
    <div className="min-h-screen bg-[#050505] text-[#F7F6F2] flex flex-col font-sans selection:bg-[#D4FF00] selection:text-black">
      <AppNavbar />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Header */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono uppercase tracking-widest text-[#D4FF00] font-bold bg-[#D4FF00]/10 border border-[#D4FF00]/40 px-2.5 py-0.5">
              03 // INTERACTIVE EXPERIMENT WORKBENCH
            </span>
            <span className="text-[10px] font-mono text-[#D4FF00] bg-black border border-white/20 px-2 py-0.5">
              LIVE MMR CALCULATOR
            </span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-display font-bold text-white tracking-tight mt-1.5">
            MMR DIVERSITY WORKBENCH &amp; STEP-BY-STEP INSPECTOR
          </h1>
          <p className="text-sm text-zinc-400 max-w-3xl leading-relaxed font-sans">
            Test candidate pool generation, view exact substituted MMR equations, compare Control (greedy top-2) vs. Proposed (diversity MMR) pairs, and inspect LLM fusion in real time.
          </p>
        </div>

        {errorMsg && (
          <div className="p-4 bg-red-950/70 border border-red-700 text-red-200 text-xs font-mono flex items-center gap-3">
            <AlertCircle className="w-5 h-5 flex-shrink-0 text-red-400" />
            <span>{errorMsg}</span>
          </div>
        )}

        {/* Experiment Controls & Upload Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: Upload or Benchmark Presets */}
          <div className="lg:col-span-5 space-y-6">
            <div className="bg-[#0c0c0c] border border-white/10 p-6 space-y-5 shadow-xl reticle-bracket">
              <h2 className="text-xs font-mono font-bold uppercase tracking-widest text-[#D4FF00] flex items-center gap-2">
                <Sliders className="w-3.5 h-3.5 text-[#D4FF00]" />
                <span>INPUT &amp; HYPERPARAMETERS</span>
              </h2>

              {/* Decoding strategy selector */}
              <div>
                <label className="text-[11px] font-mono uppercase tracking-wider text-zinc-400 block mb-1.5">
                  BLIP Decoding Strategy:
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => setDecodingStrategy('sample')}
                    className={`py-2 px-3 text-xs font-mono uppercase tracking-wider border text-center transition ${
                      decodingStrategy === 'sample'
                        ? 'bg-[#D4FF00] border-[#D4FF00] text-black font-bold'
                        : 'bg-black border-white/20 text-zinc-400 hover:text-white'
                    }`}
                  >
                    Sampling (Stochastic)
                  </button>
                  <button
                    type="button"
                    onClick={() => setDecodingStrategy('beam')}
                    className={`py-2 px-3 text-xs font-mono uppercase tracking-wider border text-center transition ${
                      decodingStrategy === 'beam'
                        ? 'bg-[#D4FF00] border-[#D4FF00] text-black font-bold'
                        : 'bg-black border-white/20 text-zinc-400 hover:text-white'
                    }`}
                  >
                    Beam Search (Deterministic)
                  </button>
                </div>
              </div>

              {/* Live Image Uploader */}
              <div className="border-t border-white/10 pt-4">
                <span className="text-[11px] font-mono uppercase tracking-wider text-zinc-300 block mb-2 font-bold">
                  OPTION A // UPLOAD CUSTOM LIVE IMAGE
                </span>
                <ImageUploader
                  onImageSelected={handleImageSelected}
                  isLoading={isLoading}
                  lambdaWeight={lambdaWeight}
                  onLambdaChange={setLambdaWeight}
                />
              </div>

              {/* Benchmark Preset Selector */}
              {benchmarkSummary && (
                <div className="border-t border-white/10 pt-4 space-y-2.5">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-mono font-bold text-zinc-300 uppercase tracking-wider flex items-center gap-1.5">
                      <Database className="w-3.5 h-3.5 text-[#D4FF00]" />
                      <span>OPTION B // QUICK BENCHMARK PRESETS (N=12)</span>
                    </span>
                    <span className="text-[10px] font-mono text-[#D4FF00]">
                      SPRINGER 2025
                    </span>
                  </div>
                  <div className="grid grid-cols-4 gap-2 pt-1">
                    {benchmarkSummary.benchmark_records.map((rec) => {
                      const isSelected = selectedBenchmarkImage === rec.image;
                      return (
                        <button
                          key={rec.image}
                          type="button"
                          onClick={() => handleSelectBenchmark(rec.image)}
                          disabled={isLoading}
                          className={`p-1.5 border text-center transition ${
                            isSelected
                              ? 'bg-black border-[#D4FF00] text-[#D4FF00] shadow-[0_0_10px_rgba(212,255,0,0.2)]'
                              : 'bg-black border-white/15 text-zinc-400 hover:text-white hover:border-[#D4FF00]/50'
                          }`}
                        >
                          <img
                            src={getBenchmarkImageUrl(rec.image)}
                            alt={rec.image}
                            className="w-full h-10 object-cover mb-1"
                          />
                          <span className="text-[9px] font-mono block truncate">
                            {rec.image.replace('.jpg', '')}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Right Column: Mathematical MMR Inspection and Side-by-Side Comparison */}
          <div className="lg:col-span-7 space-y-6">
            {comparison ? (
              <>
                {/* Side-by-Side Comparison Card */}
                <div className="bg-[#0c0c0c] border border-white/10 p-6 space-y-5 shadow-xl reticle-bracket">
                  <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 pb-4">
                    <div className="flex items-center gap-2">
                      <GitCompare className="w-4 h-4 text-[#D4FF00]" />
                      <h3 className="text-xs font-mono font-bold uppercase tracking-widest text-white">
                        SIDE-BY-SIDE METHOD EVALUATION
                      </h3>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className={`text-[10px] font-mono uppercase px-2 py-0.5 border ${
                        comparison.is_divergent
                          ? 'bg-black text-[#D4FF00] border-[#D4FF00] font-bold'
                          : 'bg-black text-zinc-400 border-white/20'
                      }`}>
                        {comparison.is_divergent ? '[ DIVERGENT SELECTION ]' : '[ IDENTICAL PAIR ]'}
                      </span>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    {/* Control Column */}
                    <div className="bg-black border border-white/10 p-4 space-y-3">
                      <div className="flex items-center justify-between border-b border-white/10 pb-2">
                        <span className="text-xs font-mono font-bold text-zinc-300 uppercase">
                          Control (Greedy Top-2)
                        </span>
                        <span className="text-[10px] font-mono text-zinc-500">
                          Rank 1 + Rank 2
                        </span>
                      </div>
                      <div className="space-y-1 text-xs">
                        <span className="text-[10px] text-zinc-500 block font-mono uppercase">Anchor (c₁):</span>
                        <p className="text-zinc-300 italic font-sans">&ldquo;{comparison.control_pair.rank_1_caption}&rdquo;</p>
                      </div>
                      <div className="space-y-1 text-xs">
                        <span className="text-[10px] text-zinc-500 block font-mono uppercase">Detail (c₂):</span>
                        <p className="text-zinc-300 italic font-sans">&ldquo;{comparison.control_pair.rank_2_caption}&rdquo;</p>
                      </div>
                      <div className="border-t border-white/10 pt-2 space-y-1">
                        <span className="text-[10px] text-zinc-400 font-mono uppercase block">Control Fused Caption:</span>
                        <p className="text-xs text-zinc-200 font-medium font-sans">&ldquo;{comparison.control_fusion.fused_caption}&rdquo;</p>
                        <div className="pt-1 text-[11px] font-mono text-zinc-400">
                          Pair Diversity: <strong className="text-white">{(comparison.control_pair.pair_diversity * 100).toFixed(1)}%</strong>
                        </div>
                      </div>
                    </div>

                    {/* Proposed Column */}
                    <div className="bg-black border border-[#D4FF00] p-4 space-y-3 shadow-[0_0_15px_rgba(212,255,0,0.15)]">
                      <div className="flex items-center justify-between border-b border-white/10 pb-2">
                        <span className="text-xs font-mono font-bold text-[#D4FF00] uppercase">
                          Proposed (MMR λ={lambdaWeight})
                        </span>
                        <span className="text-[10px] font-mono text-[#D4FF00]">
                          Rank 1 + MMR c₂*
                        </span>
                      </div>
                      <div className="space-y-1 text-xs">
                        <span className="text-[10px] text-zinc-500 block font-mono uppercase">Anchor (c₁*):</span>
                        <p className="text-zinc-200 italic font-sans">&ldquo;{comparison.proposed_pair.rank_1_caption}&rdquo;</p>
                      </div>
                      <div className="space-y-1 text-xs">
                        <span className="text-[10px] text-[#D4FF00] block font-mono uppercase">Selected (c₂*):</span>
                        <p className="text-white italic font-semibold font-sans">&ldquo;{comparison.proposed_pair.rank_2_caption}&rdquo;</p>
                      </div>
                      <div className="border-t border-white/10 pt-2 space-y-1">
                        <span className="text-[10px] text-[#D4FF00] font-mono uppercase block">Proposed Fused Caption:</span>
                        <p className="text-xs text-white font-medium font-sans">&ldquo;{comparison.proposed_fusion.fused_caption}&rdquo;</p>
                        <div className="pt-1 text-[11px] font-mono text-zinc-400">
                          Pair Diversity: <strong className="text-[#D4FF00]">{(comparison.proposed_pair.pair_diversity * 100).toFixed(1)}%</strong>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Audio for proposed */}
                  <div className="pt-2">
                    <CaptionPlayer
                      captionText={comparison.proposed_fusion.fused_caption}
                      label="Proposed Fused Caption Audio"
                    />
                  </div>
                </div>

                {/* Step-by-Step MMR Calculation Breakdown */}
                <div className="bg-[#0c0c0c] border border-white/10 p-6 space-y-5 shadow-xl reticle-bracket">
                  <div className="flex items-center gap-2 border-b border-white/10 pb-3">
                    <Calculator className="w-4 h-4 text-[#D4FF00]" />
                    <h3 className="text-xs font-mono font-bold uppercase tracking-widest text-white">
                      STEP-BY-STEP MMR MATHEMATICAL INSPECTION
                    </h3>
                  </div>

                  {/* Mathematical Formula Display */}
                  <div className="bg-black border border-white/20 p-4 text-xs font-mono text-[#D4FF00] space-y-1">
                    <div className="text-zinc-400 text-[10px] uppercase tracking-wider">Substituted Objective Equation:</div>
                    <div className="text-sm sm:text-base font-bold">
                      MMR(c_j) = {lambdaWeight.toFixed(2)} &times; s_norm(c_j) - {(1 - lambdaWeight).toFixed(2)} &times; Sim(c₁*, c_j)
                    </div>
                    <div className="text-[11px] text-zinc-400 pt-1">
                      Visual Anchor c₁* = &ldquo;{comparison.proposed_pair.rank_1_caption}&rdquo;
                    </div>
                  </div>

                  {/* Candidate Pool Table with Substituted Values */}
                  <div className="overflow-x-auto border border-white/10">
                    <table className="w-full text-left text-xs">
                      <thead className="bg-black border-b border-white/10 text-zinc-400 font-mono text-[10px] uppercase tracking-wider">
                        <tr>
                          <th className="py-2.5 px-3">Cand</th>
                          <th className="py-2.5 px-3">Caption Text</th>
                          <th className="py-2.5 px-3">Raw Score</th>
                          <th className="py-2.5 px-3">Sim(c₁*, c_j)</th>
                          <th className="py-2.5 px-3 text-[#D4FF00]">MMR Calculation</th>
                          <th className="py-2.5 px-3 text-right">Status</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-white/10 text-zinc-300 font-sans">
                        {(() => {
                          const anchorText = comparison.proposed_pair.rank_1_caption || '';
                          const selectedC2Text = comparison.proposed_pair.rank_2_caption || '';
                          const scores = comparison.candidates.map((c) => c.clip_score);
                          const minS = Math.min(...scores);
                          const maxS = Math.max(...scores);
                          const span = maxS - minS > 1e-6 ? maxS - minS : 1.0;

                          return comparison.candidates.map((c, idx) => {
                            const isAnchor = c.caption === anchorText;
                            const isSelected = c.caption === selectedC2Text && !isAnchor;
                            const sNorm = (c.clip_score - minS) / span;
                            const sim = computeJaccard(anchorText, c.caption);
                            const mmrVal = lambdaWeight * sNorm - (1 - lambdaWeight) * sim;

                            return (
                              <tr
                                key={idx}
                                className={
                                  isAnchor
                                    ? 'bg-black'
                                    : isSelected
                                    ? 'bg-black font-medium'
                                    : 'bg-[#0c0c0c]'
                                }
                              >
                                <td className="py-2.5 px-3 font-mono text-[11px] text-[#D4FF00]">
                                  c_{idx + 1}
                                </td>
                                <td className="py-2.5 px-3 max-w-xs truncate" title={c.caption}>
                                  {c.caption}
                                </td>
                                <td className="py-2.5 px-3 font-mono text-[11px]">
                                  {c.clip_score.toFixed(3)}
                                </td>
                                <td className="py-2.5 px-3 font-mono text-[11px]">
                                  {isAnchor ? '1.000 (Self)' : sim.toFixed(3)}
                                </td>
                                <td className="py-2.5 px-3 font-mono text-[11px] text-[#D4FF00]">
                                  {isAnchor ? (
                                    <span className="text-zinc-500">Anchor</span>
                                  ) : (
                                    <span>
                                      {lambdaWeight.toFixed(2)}({sNorm.toFixed(2)}) - {(1 - lambdaWeight).toFixed(2)}({sim.toFixed(2)}) ={' '}
                                      <strong className="text-white">{mmrVal.toFixed(3)}</strong>
                                    </span>
                                  )}
                                </td>
                                <td className="py-2.5 px-3 text-right">
                                  {isAnchor ? (
                                    <span className="text-[10px] font-mono bg-[#D4FF00] text-black px-2 py-0.5 font-bold">
                                      Anchor (c₁*)
                                    </span>
                                  ) : isSelected ? (
                                    <span className="text-[10px] font-mono bg-black text-[#D4FF00] px-2 py-0.5 border border-[#D4FF00] font-bold">
                                      Selected (c₂*)
                                    </span>
                                  ) : (
                                    <span className="text-[10px] font-mono text-zinc-500">
                                      Alternative
                                    </span>
                                  )}
                                </td>
                              </tr>
                            );
                          });
                        })()}
                      </tbody>
                    </table>
                  </div>

                  {/* Plain Language Interpretation */}
                  <div className="bg-black border border-white/10 p-4 text-xs text-zinc-300 space-y-1.5 font-sans">
                    <span className="text-xs font-mono font-bold uppercase tracking-wider text-[#D4FF00] block flex items-center gap-1.5">
                      <TrendingUp className="w-3.5 h-3.5" />
                      <span>Mathematical Selection Rationale</span>
                    </span>
                    <p className="leading-relaxed text-zinc-400">
                      {comparison.plain_language_explanation}
                    </p>
                  </div>
                </div>
              </>
            ) : (
              <div className="bg-[#0c0c0c] border border-white/15 border-dashed p-12 text-center text-zinc-500 space-y-3 reticle-bracket">
                <FlaskConical className="w-10 h-10 mx-auto text-[#D4FF00]/50" />
                <h3 className="text-xs font-mono font-bold uppercase tracking-widest text-zinc-300">
                  NO IMAGE EVALUATED IN LAB
                </h3>
                <p className="text-xs max-w-md mx-auto text-zinc-400 font-sans leading-relaxed">
                  Select a benchmark image from the presets on the left or upload an image to inspect live MMR numerical derivations, candidate penalties, and comparison metrics.
                </p>
              </div>
            )}
          </div>
        </div>
      </main>

      <AppFooter />
    </div>
  );
};

export default ExperimentLab;

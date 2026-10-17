import React from 'react';
import type { ComparisonResponse } from '../types/api';
import { CaptionPlayer } from './CaptionPlayer';
import {
  CheckCircle2,
  GitCompare,
  Info,
  Sparkles,
  Zap,
} from 'lucide-react';

interface ComparisonViewProps {
  comparison: ComparisonResponse;
}

export const ComparisonView: React.FC<ComparisonViewProps> = ({ comparison }) => {
  const {
    control_pair,
    control_fusion,
    proposed_pair,
    proposed_fusion,
    is_divergent,
    pair_diversity_delta,
    plain_language_explanation,
  } = comparison;

  return (
    <div className="space-y-6">
      {/* Divergence Header Badge */}
      <div
        className={`p-4 border flex flex-wrap items-center justify-between gap-3 shadow-md ${
          is_divergent
            ? 'bg-[#D4FF00]/10 border-[#D4FF00] text-white'
            : 'bg-[#0c0c0c] border-white/20 text-white'
        }`}
      >
        <div className="flex items-center gap-3">
          <div
            className={`w-9 h-9 border flex items-center justify-center font-bold ${
              is_divergent ? 'bg-[#D4FF00] text-black border-[#D4FF00]' : 'bg-black text-[#D4FF00] border-white/20'
            }`}
          >
            <GitCompare className="w-5 h-5" />
          </div>
          <div>
            <h4 className="text-sm font-mono font-bold flex items-center gap-2">
              {is_divergent ? (
                <>
                  <span className="text-[#D4FF00]">SELECTION DIVERGED FROM NAIVE TOP-2</span>
                  <span className="text-xs bg-black text-[#D4FF00] border border-[#D4FF00] px-2 py-0.5 font-mono">
                    +{(pair_diversity_delta * 100).toFixed(1)}% DIVERSITY DELTA
                  </span>
                </>
              ) : (
                <>
                  <span>IDENTICAL CANDIDATE PAIR SELECTED</span>
                  <span className="text-xs bg-black text-zinc-400 border border-white/20 px-2 py-0.5 font-mono">
                    OPTIMAL NON-REDUNDANT
                  </span>
                </>
              )}
            </h4>
            <p className="text-xs text-zinc-400 mt-1 font-sans">
              {is_divergent
                ? 'MMR penalty eliminated candidate redundancy, selecting a complementary secondary caption.'
                : 'Candidate 2 was already non-redundant with visual anchor; both methods chose the same pair.'}
            </p>
          </div>
        </div>

        <div className="text-right">
          <span className="text-[10px] font-mono uppercase text-zinc-400 block tracking-wider">Proposed MMR Lambda</span>
          <span className="text-sm font-mono font-bold text-[#D4FF00]">
            &lambda; = {proposed_pair.lambda_weight.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Confidence Warning Banner */}
      {comparison.confidence_warning && (
        <div className="p-3.5 bg-amber-950/40 border border-amber-500/50 text-amber-200 text-xs font-mono flex items-center gap-2.5">
          <Info className="w-4 h-4 text-amber-400 flex-shrink-0" />
          <span>
            <strong>LOW VISUAL ALIGNMENT WARNING:</strong> {comparison.confidence_warning}
          </span>
        </div>
      )}

      {/* Side-by-Side Comparison Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* CONTROL METHOD CARD */}
        <div className="bg-[#0c0c0c] border border-white/10 p-5 space-y-4 flex flex-col justify-between">
          <div className="space-y-3">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <div>
                <span className="text-[10px] font-mono font-bold uppercase tracking-widest text-zinc-400 block">
                  Control Method (Baseline)
                </span>
                <h4 className="text-sm font-bold text-white flex items-center gap-1.5 mt-0.5">
                  <CheckCircle2 className="w-4 h-4 text-zinc-400" />
                  Naive Top-2 Selection
                </h4>
              </div>
              <span className="text-xs font-mono bg-black text-zinc-300 border border-white/20 px-2 py-1">
                &lambda; = 1.00
              </span>
            </div>

            {/* Input Pair Details */}
            <div className="bg-black p-3.5 border border-white/10 space-y-2.5">
              <span className="text-[10px] font-mono font-bold text-zinc-400 uppercase tracking-widest block">
                Selected Input Captions
              </span>
              <div className="text-xs text-zinc-300 space-y-1.5 font-sans">
                <p>
                  <strong className="text-[#D4FF00] font-mono">c1:</strong> &ldquo;{control_pair.rank_1_caption}&rdquo;
                </p>
                <p>
                  <strong className="text-zinc-400 font-mono">c2 (Rank {control_pair.rank_2_original_rank}):</strong> &ldquo;
                  {control_pair.rank_2_caption}&rdquo;
                </p>
              </div>

              <div className="pt-2 border-t border-white/10 flex justify-between text-[11px] font-mono text-zinc-400">
                <span>
                  Input Diversity: <strong className="text-white">{(control_pair.pair_diversity * 100).toFixed(1)}%</strong>
                </span>
                <span>
                  Jaccard Overlap: <strong className="text-white">{(control_pair.pair_similarity * 100).toFixed(1)}%</strong>
                </span>
              </div>
            </div>

            {/* Final Fused Output */}
            <div className="space-y-2 pt-2">
              <span className="text-xs font-mono font-bold uppercase tracking-wider text-zinc-300 flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-zinc-400" />
                Control LLM-Fused Narrative
              </span>
              <div className="p-4 bg-black border border-white/10 min-h-[4.5rem] flex items-center">
                {control_fusion.generation_status === 'SUCCESS' ? (
                  <p className="text-sm text-zinc-200 font-sans italic leading-relaxed">
                    &ldquo;{control_fusion.fused_caption}&rdquo;
                  </p>
                ) : control_fusion.generation_status === 'GROQ_KEY_NOT_CONFIGURED' ? (
                  <p className="text-xs font-mono text-amber-300/80 italic">
                    [Pending Live LLM Fusion: Configure GROQ_API_KEY]
                  </p>
                ) : (
                  <p className="text-xs font-mono text-red-300/80 italic">
                    [{control_fusion.generation_status}: {control_fusion.error_details || 'Failed'}]
                  </p>
                )}
              </div>
            </div>
          </div>

          {/* Text-to-Speech Player */}
          {control_fusion.generation_status === 'SUCCESS' && (
            <div className="pt-2">
              <CaptionPlayer captionText={control_fusion.fused_caption} label="Control Audio" />
            </div>
          )}
        </div>

        {/* PROPOSED METHOD CARD */}
        <div className="bg-[#0c0c0c] border border-[#D4FF00] p-5 space-y-4 flex flex-col justify-between shadow-[0_0_20px_rgba(212,255,0,0.15)] reticle-bracket">
          <div className="space-y-3">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <div>
                <span className="text-[10px] font-mono font-bold uppercase tracking-widest text-[#D4FF00] block">
                  Proposed Research Contribution
                </span>
                <h4 className="text-sm font-bold text-white flex items-center gap-1.5 mt-0.5">
                  <Zap className="w-4 h-4 text-[#D4FF00]" />
                  Diversity-Aware Ranking (MMR)
                </h4>
              </div>
              <span className="text-xs font-mono bg-black text-[#D4FF00] border border-[#D4FF00] px-2 py-1 font-bold">
                &lambda; = {proposed_pair.lambda_weight.toFixed(2)}
              </span>
            </div>

            {/* Input Pair Details */}
            <div className="bg-black p-3.5 border border-white/10 space-y-2.5">
              <span className="text-[10px] font-mono font-bold text-[#D4FF00] uppercase tracking-widest block">
                Selected Input Captions
              </span>
              <div className="text-xs text-zinc-200 space-y-1.5 font-sans">
                <p>
                  <strong className="text-[#D4FF00] font-mono">c1*:</strong> &ldquo;{proposed_pair.rank_1_caption}&rdquo;
                </p>
                <p>
                  <strong className="text-[#D4FF00] font-mono">c2* (Rank {proposed_pair.rank_2_original_rank}):</strong> &ldquo;
                  {proposed_pair.rank_2_caption}&rdquo;
                </p>
              </div>

              <div className="pt-2 border-t border-white/10 flex justify-between text-[11px] font-mono text-zinc-400">
                <span>
                  Input Diversity: <strong className="text-[#D4FF00] font-bold">{(proposed_pair.pair_diversity * 100).toFixed(1)}%</strong>
                </span>
                <span>
                  Jaccard Overlap: <strong className="text-white">{(proposed_pair.pair_similarity * 100).toFixed(1)}%</strong>
                </span>
              </div>
            </div>

            {/* Final Fused Output */}
            <div className="space-y-2 pt-2">
              <span className="text-xs font-mono font-bold uppercase tracking-wider text-[#D4FF00] flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-[#D4FF00]" />
                Proposed LLM-Fused Narrative
              </span>
              <div className="p-4 bg-black border border-[#D4FF00]/50 min-h-[4.5rem] flex items-center">
                {proposed_fusion.generation_status === 'SUCCESS' ? (
                  <p className="text-sm text-white font-sans font-medium italic leading-relaxed">
                    &ldquo;{proposed_fusion.fused_caption}&rdquo;
                  </p>
                ) : proposed_fusion.generation_status === 'GROQ_KEY_NOT_CONFIGURED' ? (
                  <p className="text-xs font-mono text-amber-300/80 italic">
                    [Pending Live LLM Fusion: Configure GROQ_API_KEY]
                  </p>
                ) : (
                  <p className="text-xs font-mono text-red-300/80 italic">
                    [{proposed_fusion.generation_status}: {proposed_fusion.error_details || 'Failed'}]
                  </p>
                )}
              </div>
            </div>
          </div>

          {/* Text-to-Speech Player */}
          {proposed_fusion.generation_status === 'SUCCESS' && (
            <div className="pt-2">
              <CaptionPlayer captionText={proposed_fusion.fused_caption} label="Proposed Audio" />
            </div>
          )}
        </div>
      </div>

      {/* Plain Language Explanation Callout */}
      <div className="p-4 bg-[#0c0c0c] border border-white/10 flex items-start gap-3 text-xs text-zinc-300">
        <Info className="w-4 h-4 text-[#D4FF00] flex-shrink-0 mt-0.5" />
        <div className="space-y-1">
          <span className="font-mono font-bold text-white uppercase tracking-wider block">
            Ranker Rationale &amp; Analysis
          </span>
          <p className="leading-relaxed text-zinc-400 font-sans">{plain_language_explanation}</p>
        </div>
      </div>
    </div>
  );
};

export default ComparisonView;

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
        className={`p-4 rounded-2xl border flex flex-wrap items-center justify-between gap-3 shadow-md ${
          is_divergent
            ? 'bg-emerald-950/30 border-emerald-500/50 text-emerald-200'
            : 'bg-slate-900/80 border-slate-700/80 text-slate-200'
        }`}
      >
        <div className="flex items-center gap-3">
          <div
            className={`w-9 h-9 rounded-xl flex items-center justify-center font-bold ${
              is_divergent ? 'bg-emerald-500 text-slate-950' : 'bg-slate-800 text-slate-300'
            }`}
          >
            <GitCompare className="w-5 h-5" />
          </div>
          <div>
            <h4 className="text-sm font-bold flex items-center gap-2">
              {is_divergent ? (
                <>
                  <span>Selection Diverged from Naive Top-2</span>
                  <span className="text-xs bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 px-2 py-0.5 rounded-full font-mono">
                    +{(pair_diversity_delta * 100).toFixed(1)}% Diversity Delta
                  </span>
                </>
              ) : (
                <>
                  <span>Identical Candidate Pair Selected</span>
                  <span className="text-xs bg-slate-800 text-slate-400 border border-slate-700 px-2 py-0.5 rounded-full font-mono">
                    Optimal Non-Redundant
                  </span>
                </>
              )}
            </h4>
            <p className="text-xs text-slate-400 mt-0.5">
              {is_divergent
                ? 'MMR penalty eliminated candidate redundancy, selecting a complementary secondary caption.'
                : 'Candidate 2 was already non-redundant with visual anchor; both methods chose the same pair.'}
            </p>
          </div>
        </div>

        <div className="text-right">
          <span className="text-xs font-mono text-slate-400 block">Proposed MMR Lambda</span>
          <span className="text-sm font-mono font-bold text-indigo-400">
            &lambda; = {proposed_pair.lambda_weight.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Side-by-Side Comparison Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* CONTROL METHOD CARD */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 space-y-4 flex flex-col justify-between">
          <div className="space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400 block">
                  Control Method (Baseline)
                </span>
                <h4 className="text-sm font-semibold text-slate-100 flex items-center gap-1.5">
                  <CheckCircle2 className="w-4 h-4 text-sky-400" />
                  Naive CLIP Top-2 Selection
                </h4>
              </div>
              <span className="text-xs font-mono bg-sky-950/60 text-sky-300 border border-sky-800/60 px-2 py-1 rounded-md">
                &lambda; = 1.00
              </span>
            </div>

            {/* Input Pair Details */}
            <div className="bg-slate-950/50 rounded-xl p-3 border border-slate-800/80 space-y-2">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">
                Selected Input Captions
              </span>
              <div className="text-xs text-slate-300 space-y-1">
                <p>
                  <strong className="text-indigo-400">c1:</strong> "{control_pair.rank_1_caption}"
                </p>
                <p>
                  <strong className="text-sky-400">c2 (Rank {control_pair.rank_2_original_rank}):</strong> "
                  {control_pair.rank_2_caption}"
                </p>
              </div>

              <div className="pt-2 border-t border-slate-800/60 flex justify-between text-[11px] font-mono text-slate-400">
                <span>
                  Input Diversity: <strong className="text-slate-200">{(control_pair.pair_diversity * 100).toFixed(1)}%</strong>
                </span>
                <span>
                  Jaccard Overlap: <strong className="text-slate-200">{(control_pair.pair_similarity * 100).toFixed(1)}%</strong>
                </span>
              </div>
            </div>

            {/* Final Fused Output */}
            <div className="space-y-2 pt-2">
              <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-sky-400" />
                Control LLM-Fused Caption
              </span>
              <div className="p-3.5 bg-slate-950 rounded-xl border border-slate-800 min-h-[4rem] flex items-center">
                {control_fusion.generation_status === 'SUCCESS' ? (
                  <p className="text-sm text-slate-100 font-medium italic">
                    "{control_fusion.fused_caption}"
                  </p>
                ) : control_fusion.generation_status === 'GROQ_KEY_NOT_CONFIGURED' ? (
                  <p className="text-xs text-amber-300/80 italic">
                    [Pending Live LLM Fusion: Configure GROQ_API_KEY]
                  </p>
                ) : (
                  <p className="text-xs text-red-300/80 italic">
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
        <div className="bg-slate-900/90 border border-emerald-900/60 rounded-2xl p-5 space-y-4 flex flex-col justify-between shadow-lg shadow-emerald-950/20">
          <div className="space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <span className="text-[11px] font-bold uppercase tracking-wider text-emerald-400 block">
                  Proposed Research Contribution
                </span>
                <h4 className="text-sm font-semibold text-slate-100 flex items-center gap-1.5">
                  <Zap className="w-4 h-4 text-emerald-400" />
                  Diversity-Aware Ranking (MMR)
                </h4>
              </div>
              <span className="text-xs font-mono bg-emerald-950/60 text-emerald-300 border border-emerald-800/60 px-2 py-1 rounded-md">
                &lambda; = {proposed_pair.lambda_weight.toFixed(2)}
              </span>
            </div>

            {/* Input Pair Details */}
            <div className="bg-slate-950/50 rounded-xl p-3 border border-slate-800/80 space-y-2">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">
                Selected Input Captions
              </span>
              <div className="text-xs text-slate-300 space-y-1">
                <p>
                  <strong className="text-indigo-400">c1*:</strong> "{proposed_pair.rank_1_caption}"
                </p>
                <p>
                  <strong className="text-emerald-400">c2* (Rank {proposed_pair.rank_2_original_rank}):</strong> "
                  {proposed_pair.rank_2_caption}"
                </p>
              </div>

              <div className="pt-2 border-t border-slate-800/60 flex justify-between text-[11px] font-mono text-slate-400">
                <span>
                  Input Diversity: <strong className="text-emerald-300 font-bold">{(proposed_pair.pair_diversity * 100).toFixed(1)}%</strong>
                </span>
                <span>
                  Jaccard Overlap: <strong className="text-slate-200">{(proposed_pair.pair_similarity * 100).toFixed(1)}%</strong>
                </span>
              </div>
            </div>

            {/* Final Fused Output */}
            <div className="space-y-2 pt-2">
              <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
                Proposed LLM-Fused Caption
              </span>
              <div className="p-3.5 bg-slate-950 rounded-xl border border-emerald-900/60 min-h-[4rem] flex items-center">
                {proposed_fusion.generation_status === 'SUCCESS' ? (
                  <p className="text-sm text-slate-100 font-medium italic">
                    "{proposed_fusion.fused_caption}"
                  </p>
                ) : proposed_fusion.generation_status === 'GROQ_KEY_NOT_CONFIGURED' ? (
                  <p className="text-xs text-amber-300/80 italic">
                    [Pending Live LLM Fusion: Configure GROQ_API_KEY]
                  </p>
                ) : (
                  <p className="text-xs text-red-300/80 italic">
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
      <div className="p-4 bg-slate-900/70 border border-slate-800 rounded-2xl flex items-start gap-3 text-xs text-slate-300">
        <Info className="w-4 h-4 text-indigo-400 flex-shrink-0 mt-0.5" />
        <div className="space-y-1">
          <span className="font-semibold text-slate-200 uppercase tracking-wider block">
            Ranker Rationale & Analysis
          </span>
          <p className="leading-relaxed text-slate-400">{plain_language_explanation}</p>
        </div>
      </div>
    </div>
  );
};

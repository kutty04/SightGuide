import React from 'react';
import type { CandidateCaption, SelectedPair } from '../types/api';
import { Award, Compass, Eye, ShieldCheck } from 'lucide-react';

interface CandidateListProps {
  candidates: CandidateCaption[];
  selectedPair: SelectedPair;
  controlPair?: SelectedPair;
}

export const CandidateList: React.FC<CandidateListProps> = ({
  candidates,
  selectedPair,
  controlPair,
}) => {
  const scores = candidates.map((c) => c.clip_score);
  const minScore = Math.min(...scores);
  const maxScore = Math.max(...scores);
  const scoreRange = maxScore - minScore;

  return (
    <div className="bg-[#0c0c0c] border border-white/10 p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-xs font-mono font-bold uppercase tracking-widest text-[#D4FF00] flex items-center gap-2">
            <Eye className="w-3.5 h-3.5 text-[#D4FF00]" />
            <span>Generated Candidate Captions ({candidates.length})</span>
          </h3>
          <p className="text-xs text-zinc-400 mt-1 font-sans">
            Initial BLIP candidates ranked by genuine cross-modal image-text alignment score
          </p>
        </div>
      </div>

      <div className="space-y-2.5">
        {candidates.map((cand) => {
          const isAnchor = cand.caption === selectedPair.rank_1_caption;
          const isProposedSecondary = cand.caption === selectedPair.rank_2_caption;
          const isControlSecondary = controlPair && cand.caption === controlPair.rank_2_caption;

          // Normalized relative bar height (35% to 100%)
          const pct = scoreRange > 0.0001
            ? Math.round(35 + 65 * ((cand.clip_score - minScore) / scoreRange))
            : 100;

          const isLogLikelihood = cand.clip_score <= 0;
          const scoreLabel = isLogLikelihood
            ? `LL Score: ${cand.clip_score.toFixed(3)}`
            : `CLIP: ${cand.clip_score.toFixed(2)}`;

          return (
            <div
              key={cand.rank}
              className={`p-3.5 border transition-all ${
                isAnchor
                  ? 'bg-black border-[#D4FF00] shadow-[0_0_15px_rgba(212,255,0,0.15)]'
                  : isProposedSecondary
                  ? 'bg-black border-[#D4FF00]/60'
                  : isControlSecondary
                  ? 'bg-black border-white/30'
                  : 'bg-black/60 border-white/10'
              }`}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-start gap-2.5 flex-1 min-w-0">
                  <span className={`flex-shrink-0 w-6 h-6 border font-mono text-xs flex items-center justify-center font-bold ${
                    isAnchor ? 'bg-[#D4FF00] text-black border-[#D4FF00]' : 'bg-black text-[#D4FF00] border-white/20'
                  }`}>
                    {cand.rank}
                  </span>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-zinc-200 font-sans leading-relaxed">
                      &ldquo;{cand.caption}&rdquo;
                    </p>
                  </div>
                </div>

                <div className="flex-shrink-0 flex items-center gap-1.5 flex-wrap justify-end">
                  {cand.confidence_warning && (
                    <span className="inline-flex items-center gap-1 text-[10px] font-mono uppercase bg-amber-500/20 text-amber-300 border border-amber-500/40 px-2 py-0.5">
                      Low Confidence
                    </span>
                  )}
                  {isAnchor && (
                    <span className="inline-flex items-center gap-1 text-[10px] font-mono uppercase bg-[#D4FF00] text-black font-bold px-2 py-0.5">
                      <Award className="w-3 h-3 text-black" />
                      Visual Anchor (c1*)
                    </span>
                  )}
                  {isProposedSecondary && (
                    <span className="inline-flex items-center gap-1 text-[10px] font-mono uppercase bg-black text-[#D4FF00] border border-[#D4FF00] font-bold px-2 py-0.5">
                      <Compass className="w-3 h-3 text-[#D4FF00]" />
                      MMR Selected (c2*)
                    </span>
                  )}
                  {isControlSecondary && !isProposedSecondary && (
                    <span className="inline-flex items-center gap-1 text-[10px] font-mono uppercase bg-zinc-900 text-zinc-300 border border-zinc-700 px-2 py-0.5">
                      <ShieldCheck className="w-3 h-3 text-zinc-400" />
                      Naive Top-2 (c2)
                    </span>
                  )}
                </div>
              </div>

              {/* Alignment Bar */}
              <div className="mt-2.5 pt-2 border-t border-white/10 flex items-center gap-3">
                <div className="flex-1 bg-white/5 h-1.5 overflow-hidden">
                  <div
                    className={`h-full transition-all duration-500 ${
                      isAnchor || isProposedSecondary
                        ? 'bg-[#D4FF00]'
                        : 'bg-zinc-600'
                    }`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <span className="text-[11px] font-mono text-zinc-400 flex-shrink-0">
                  <strong className="text-white font-mono">{scoreLabel}</strong>
                </span>
              </div>
            </div>
          );
        })}
      </div>
      <p className="text-[11px] text-zinc-500 font-mono pt-1 border-t border-white/10">
        * Note: Negative sequence loss (LL Score) represents conditional cross-modal log-likelihood. It serves as an uncalibrated relative ranking metric, not an absolute confidence probability or percentage accuracy.
      </p>
    </div>
  );
};

export default CandidateList;

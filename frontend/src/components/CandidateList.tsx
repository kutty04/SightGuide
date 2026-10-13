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
  const maxScore = Math.max(...candidates.map((c) => c.clip_score), 1.0);

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
            <Eye className="w-4 h-4 text-indigo-400" />
            Generated Candidate Captions ({candidates.length})
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Initial BLIP candidates ranked by raw CLIP image-text alignment score
          </p>
        </div>
      </div>

      <div className="space-y-2.5">
        {candidates.map((cand) => {
          const isAnchor = cand.caption === selectedPair.rank_1_caption;
          const isProposedSecondary = cand.caption === selectedPair.rank_2_caption;
          const isControlSecondary = controlPair && cand.caption === controlPair.rank_2_caption;

          const pct = Math.max(10, Math.min(100, (cand.clip_score / maxScore) * 100));

          return (
            <div
              key={cand.rank}
              className={`p-3.5 rounded-xl border transition-all ${
                isAnchor
                  ? 'bg-indigo-950/40 border-indigo-500/80 ring-1 ring-indigo-500/50'
                  : isProposedSecondary
                  ? 'bg-emerald-950/30 border-emerald-500/70'
                  : isControlSecondary
                  ? 'bg-sky-950/30 border-sky-500/60'
                  : 'bg-slate-950/40 border-slate-800/80'
              }`}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-start gap-2.5 flex-1 min-w-0">
                  <span className="flex-shrink-0 w-5 h-5 rounded-full bg-slate-800 text-slate-300 font-mono text-xs flex items-center justify-center font-bold">
                    {cand.rank}
                  </span>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-slate-200 font-medium leading-relaxed">
                      "{cand.caption}"
                    </p>
                  </div>
                </div>

                <div className="flex-shrink-0 flex items-center gap-1.5 flex-wrap justify-end">
                  {isAnchor && (
                    <span className="inline-flex items-center gap-1 text-[11px] font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 px-2 py-0.5 rounded-md">
                      <Award className="w-3 h-3" />
                      Visual Anchor (c1*)
                    </span>
                  )}
                  {isProposedSecondary && (
                    <span className="inline-flex items-center gap-1 text-[11px] font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 px-2 py-0.5 rounded-md">
                      <Compass className="w-3 h-3" />
                      MMR Selected (c2*)
                    </span>
                  )}
                  {isControlSecondary && !isProposedSecondary && (
                    <span className="inline-flex items-center gap-1 text-[11px] font-semibold bg-sky-500/20 text-sky-300 border border-sky-500/40 px-2 py-0.5 rounded-md">
                      <ShieldCheck className="w-3 h-3" />
                      Naive Top-2 (c2)
                    </span>
                  )}
                </div>
              </div>

              {/* CLIP Alignment Bar */}
              <div className="mt-2.5 pt-2 border-t border-slate-800/60 flex items-center gap-3">
                <div className="flex-1 bg-slate-800/80 h-1.5 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      isAnchor
                        ? 'bg-indigo-400'
                        : isProposedSecondary
                        ? 'bg-emerald-400'
                        : 'bg-slate-500'
                    }`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <span className="text-[11px] font-mono text-slate-400 flex-shrink-0">
                  CLIP: <strong className="text-slate-200 font-semibold">{cand.clip_score.toFixed(2)}</strong>
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

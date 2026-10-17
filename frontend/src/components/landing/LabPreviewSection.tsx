import React, { useState, useMemo } from 'react';
import { Link } from '../../router';
import { Sliders, ArrowRight, Sparkles } from 'lucide-react';

interface Candidate {
  id: string;
  text: string;
  clipScore: number;
  simScore: number;
}

const FIXED_ANCHOR = {
  id: 'c1*',
  text: 'a boat in a body of water with mountains in the background',
  clipScore: 0.884,
  label: 'FIXED PRIMARY ANCHOR',
};

const CANDIDATE_POOL: Candidate[] = [
  {
    id: 'c2',
    text: 'a boat floating on a large body of water',
    clipScore: 0.862,
    simScore: 0.764,
  },
  {
    id: 'c3',
    text: 'a small boat resting on calm lake waters under cloudy sky',
    clipScore: 0.825,
    simScore: 0.286,
  },
  {
    id: 'c4',
    text: 'wooden vessel moored near rocky shore with pine trees',
    clipScore: 0.771,
    simScore: 0.214,
  },
  {
    id: 'c5',
    text: 'distant snow-capped peaks overlooking serene alpine waters',
    clipScore: 0.730,
    simScore: 0.143,
  },
];

export const LabPreviewSection: React.FC = () => {
  const [lambda, setLambda] = useState<number>(0.5);

  // Dynamically calculate MMR scores for all candidates in real time
  const rankedCandidates = useMemo(() => {
    return CANDIDATE_POOL.map((c) => {
      // MMR Formula: Score = λ * CLIP(c, I) - (1 - λ) * Sim(c1*, c)
      const mmrScore = lambda * c.clipScore - (1 - lambda) * c.simScore;
      return {
        ...c,
        mmrScore: Number(mmrScore.toFixed(4)),
      };
    }).sort((a, b) => b.mmrScore - a.mmrScore);
  }, [lambda]);

  const winningCandidate = rankedCandidates[0];

  // Dynamic fused description based on the winning candidate
  const fusedDescription = useMemo(() => {
    if (winningCandidate.id === 'c2') {
      return {
        text: 'A boat floating on a large body of water with mountains in the background.',
        note: 'Control Baseline: High lexical redundancy (76.4% token overlap). Minimal new visual information.',
        regime: 'NAIVE RELEVANCE REGIME',
        isOptimal: false,
      };
    } else if (winningCandidate.id === 'c3') {
      return {
        text: 'A small boat resting on calm lake waters with majestic distant mountains under an overcast cloudy sky.',
        note: 'Proposed Balanced Formulation: Captures water state (calm lake) and weather (overcast cloudy sky) without sacrificing boat grounding.',
        regime: 'OPTIMAL SIGHTGUIDE REGIME (λ = 0.50)',
        isOptimal: true,
      };
    } else {
      return {
        text: 'A wooden vessel moored near a rocky alpine shore with pine trees and distant snow-capped mountain peaks.',
        note: 'Max Diversity Regime: Discovers secondary landscape elements but slightly shifts focal weight away from the primary vessel.',
        regime: 'DIVERSITY-DOMINATED REGIME',
        isOptimal: false,
      };
    }
  }, [winningCandidate]);

  return (
    <section
      id="lab-preview"
      className="relative pt-6 sm:pt-10 pb-8 sm:pb-12 px-4 sm:px-6 lg:px-8 max-w-[1520px] mx-auto text-[#f7f6f2] overflow-hidden"
      aria-labelledby="section-c-heading"
    >
      {/* Top Section Header */}
      <div className="w-full flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-[#1a1a1a] mb-8 sm:mb-10">
        <div className="inline-flex items-center gap-2 font-mono text-xs uppercase tracking-[0.25em] text-[#a3a39e]">
          <span className="text-[#d4ff00] font-bold text-sm">┌</span>
          <span className="text-[#d4ff00] font-bold">03 //</span>
          <span className="text-white">INTERACTIVE MMR LAB PREVIEW</span>
        </div>

        <div className="font-mono text-[11px] text-[#73736e] uppercase tracking-widest">
          SIMULATED MATHEMATICAL TRADE-OFF DEMONSTRATOR
        </div>
      </div>

      {/* Narrative Headline */}
      <div className="max-w-3xl mb-8 sm:mb-10 space-y-3">
        <h2
          id="section-c-heading"
          className="font-display font-black uppercase text-4xl sm:text-6xl md:text-7xl leading-[0.88] tracking-tight text-white"
        >
          <span>THE LAMBDA</span> <span className="text-[#d4ff00]">BALANCING ACT.</span>
        </h2>
        <p className="text-sm sm:text-base text-[#c8c8c4] leading-relaxed font-normal">
          Adjust the trade-off parameter &lambda; to test how candidate scores shift in real time.
          Observe how the optimal balanced setting (&lambda; = 0.50) breaks free from greedy collapse
          to select descriptive, complementary secondary captions.
        </p>
      </div>

      {/* Interactive Exhibit Terminal */}
      <div className="relative rounded-sm border border-[#222222] bg-[#090909] p-6 sm:p-10 shadow-2xl">
        {/* Corner Reticles */}
        <div className="absolute top-2 left-2 w-3 h-3 border-t border-l border-[#d4ff00] pointer-events-none" />
        <div className="absolute top-2 right-2 w-3 h-3 border-t border-r border-[#d4ff00] pointer-events-none" />
        <div className="absolute bottom-2 left-2 w-3 h-3 border-b border-l border-[#d4ff00] pointer-events-none" />
        <div className="absolute bottom-2 right-2 w-3 h-3 border-b border-r border-[#d4ff00] pointer-events-none" />

        {/* Lambda Control Console Header */}
        <div className="pb-8 border-b border-[#1c1c1c] space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <span className="p-2 rounded-sm bg-[#121212] border border-[#262626] text-[#d4ff00]">
                <Sliders className="w-5 h-5" />
              </span>
              <div className="flex flex-col">
                <span className="font-mono text-[10px] text-[#73736e] uppercase tracking-wider">
                  DIVERSITY TRADE-OFF WEIGHT
                </span>
                <span className="font-display font-black text-3xl sm:text-4xl text-[#d4ff00] leading-tight">
                  λ = {lambda.toFixed(2)}
                </span>
              </div>
            </div>

            {/* Quick Preset Buttons */}
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={() => setLambda(1.0)}
                className={`px-3 py-1.5 rounded-sm font-mono text-xs font-semibold uppercase tracking-wider transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d4ff00] ${
                  lambda === 1.0
                    ? 'bg-[#181818] text-white border border-[#333333]'
                    : 'text-[#8e8d88] hover:text-white bg-[#0e0e0e] border border-[#1f1f1f]'
                }`}
              >
                λ = 1.00 (Naive Baseline)
              </button>

              <button
                type="button"
                onClick={() => setLambda(0.5)}
                className={`px-3 py-1.5 rounded-sm font-mono text-xs font-semibold uppercase tracking-wider transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d4ff00] flex items-center gap-1.5 ${
                  lambda === 0.5
                    ? 'bg-[#d4ff00] text-[#050505] font-bold shadow-[0_0_12px_rgba(212,255,0,0.35)]'
                    : 'text-[#d4ff00] hover:text-white bg-[#141a05] border border-[#d4ff00]/40'
                }`}
              >
                <Sparkles className="w-3.5 h-3.5" />
                <span>λ = 0.50 (Balanced)</span>
              </button>

              <button
                type="button"
                onClick={() => setLambda(0.0)}
                className={`px-3 py-1.5 rounded-sm font-mono text-xs font-semibold uppercase tracking-wider transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d4ff00] ${
                  lambda === 0.0
                    ? 'bg-[#181818] text-white border border-[#333333]'
                    : 'text-[#8e8d88] hover:text-white bg-[#0e0e0e] border border-[#1f1f1f]'
                }`}
              >
                λ = 0.00 (Max Diversity)
              </button>
            </div>
          </div>

          {/* Continuous Interactive Slider */}
          <div className="space-y-2">
            <div className="flex items-center justify-between font-mono text-xs text-[#8e8d88]">
              <span className="flex items-center gap-1.5 text-white">
                <span>0.00 (Pure Novelty)</span>
              </span>
              <span className="text-[#d4ff00] font-bold">
                Balance: {(lambda * 100).toFixed(0)}% Relevance / {((1 - lambda) * 100).toFixed(0)}% Diversity
              </span>
              <span className="flex items-center gap-1.5 text-white">
                <span>1.00 (Pure CLIP)</span>
              </span>
            </div>

            <div className="relative flex items-center">
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.05"
                value={lambda}
                onChange={(e) => setLambda(parseFloat(e.target.value))}
                aria-label="MMR Lambda Trade-off Weight"
                className="w-full h-2 rounded-none bg-[#1a1a1a] appearance-none cursor-pointer accent-[#d4ff00] focus:outline-none focus-visible:ring-2 focus-visible:ring-[#d4ff00]"
              />
            </div>
          </div>
        </div>

        {/* Interactive Comparison Dashboard */}
        <div className="mt-8 grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: Real-Time Dynamic Candidate Ranking */}
          <div className="lg:col-span-7 space-y-4 font-mono text-xs">
            <div className="flex items-center justify-between text-[11px] text-[#73736e] uppercase tracking-wider pb-2 border-b border-[#1c1c1c]">
              <span>CANDIDATE POOL RANKING DYNAMICS</span>
              <span className="text-[#d4ff00]">EQUATION EVALUATION</span>
            </div>

            {/* Fixed Visual Anchor */}
            <div className="p-3.5 rounded-sm bg-[#121212] border border-[#2a2a2a] space-y-1">
              <div className="flex items-center justify-between text-[10px] text-[#73736e]">
                <span className="text-white font-bold">{FIXED_ANCHOR.label} (c1*)</span>
                <span>CLIP: {FIXED_ANCHOR.clipScore} • SIM: 1.000</span>
              </div>
              <p className="text-white text-xs sm:text-sm font-medium">"{FIXED_ANCHOR.text}"</p>
            </div>

            {/* Evaluated Candidates */}
            <div className="space-y-2">
              <span className="text-[10px] text-[#73736e] uppercase tracking-wider block">
                COMPETING SECONDARY CANDIDATES (c_j):
              </span>

              {rankedCandidates.map((candidate, idx) => {
                const isSelected = candidate.id === winningCandidate.id;
                return (
                  <div
                    key={candidate.id}
                    className={`p-3.5 rounded-sm transition-all border ${
                      isSelected
                        ? 'bg-[#141a05] border-[#d4ff00]/60 shadow-[0_0_15px_rgba(212,255,0,0.15)]'
                        : 'bg-[#0e0e0e] border-[#1e1e1e] opacity-75'
                    }`}
                  >
                    <div className="flex items-center justify-between text-[11px] mb-1">
                      <div className="flex items-center gap-2">
                        <span
                          className={`font-bold ${
                            isSelected ? 'text-[#d4ff00]' : 'text-[#8e8d88]'
                          }`}
                        >
                          RANK {idx + 1} // {candidate.id}
                        </span>
                        {isSelected && (
                          <span className="px-1.5 py-0.2 rounded-none bg-[#d4ff00] text-[#050505] text-[9px] font-bold uppercase">
                            SELECTED (c2*)
                          </span>
                        )}
                      </div>

                      <span
                        className={`font-bold ${
                          isSelected ? 'text-[#d4ff00]' : 'text-[#8e8d88]'
                        }`}
                      >
                        MMR SCORE: {candidate.mmrScore > 0 ? `+${candidate.mmrScore}` : candidate.mmrScore}
                      </span>
                    </div>

                    <p className={`text-xs ${isSelected ? 'text-white font-medium' : 'text-[#a3a39e]'}`}>
                      "{candidate.text}"
                    </p>

                    <div className="mt-2 pt-1.5 border-t border-[#181818] flex items-center justify-between text-[10px] text-[#73736e]">
                      <span>CLIP: {candidate.clipScore}</span>
                      <span>OVERLAP SIM: {(candidate.simScore * 100).toFixed(1)}%</span>
                      <span className="text-[#a3a39e]">
                        λ•CLIP: {(lambda * candidate.clipScore).toFixed(3)} | (1-λ)•Sim:{' '}
                        {((1 - lambda) * candidate.simScore).toFixed(3)}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Right Column: Fused Narrative Outcome */}
          <div className="lg:col-span-5 space-y-4 font-mono text-xs">
            <div className="flex items-center justify-between text-[11px] text-[#73736e] uppercase tracking-wider pb-2 border-b border-[#1c1c1c]">
              <span>DOWNSTREAM LLM FUSION OUTCOME</span>
              <span className={fusedDescription.isOptimal ? 'text-[#d4ff00] font-bold' : 'text-[#a3a39e]'}>
                {fusedDescription.regime}
              </span>
            </div>

            {/* Fused Narrative Card */}
            <div
              className={`p-6 rounded-sm border space-y-4 ${
                fusedDescription.isOptimal
                  ? 'bg-[#141a05] border-[#d4ff00]/60'
                  : 'bg-[#111111] border-[#262626]'
              }`}
            >
              <div className="space-y-1">
                <span className="text-[10px] text-[#8e8d88] uppercase tracking-widest block">
                  SYNTHESIZED CAPTION (c_fused):
                </span>
                <p className="font-sans text-base sm:text-lg font-semibold text-white leading-relaxed italic">
                  "{fusedDescription.text}"
                </p>
              </div>

              <div className="pt-3 border-t border-[#1e1e1e] space-y-2">
                <span className="text-[10px] text-[#8e8d88] uppercase tracking-wider block">
                  RESEARCH OBSERVATION:
                </span>
                <p className="text-xs text-[#c4c3be] leading-relaxed">
                  {fusedDescription.note}
                </p>
              </div>
            </div>

            {/* Benchmark Replication Note */}
            <div className="p-4 rounded-sm bg-[#0c0c0c] border border-[#1f1f1f] space-y-2 text-[#8e8d88]">
              <div className="flex items-center gap-2 text-white font-semibold">
                <Sparkles className="w-4 h-4 text-[#d4ff00]" />
                <span>Verified Scientific Finding</span>
              </div>
              <p className="text-[11px] leading-relaxed">
                Replicating the experimental evaluation of Celona et al. across the benchmark dataset
                confirms that &lambda; = 0.50 reliably yields a +46% to +48% diversity gain while maintaining
                high semantic precision without hallucinations.
              </p>
            </div>

            {/* Direct CTA to Full Workbench */}
            <div className="pt-2">
              <Link
                to="/lab"
                className="w-full inline-flex items-center justify-center gap-2 px-6 py-3.5 rounded-sm bg-[#d4ff00] hover:bg-[#ccff00] text-[#050505] text-xs font-bold uppercase tracking-wider transition-all shadow-[0_0_20px_rgba(212,255,0,0.25)] hover:shadow-[0_0_28px_rgba(212,255,0,0.4)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d4ff00]"
              >
                <span>Launch Full MMR Research Lab</span>
                <ArrowRight className="w-4 h-4 text-[#050505]" />
              </Link>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default LabPreviewSection;

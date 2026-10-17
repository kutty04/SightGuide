import React, { useRef } from 'react';
import { motion, useScroll, useTransform, useReducedMotion } from 'motion/react';
import { Terminal, Check } from 'lucide-react';

interface StageDetail {
  num: string;
  title: string;
  tagline: string;
  badge: string;
  description: string;
  formula: string;
  telemetry: { label: string; value: string }[];
  highlight: string;
}

const STAGES: StageDetail[] = [
  {
    num: '01',
    title: 'CANDIDATE GENERATION',
    tagline: 'Stochastic Beam Search Pool (K=5)',
    badge: 'VISION-LANGUAGE BACKBONE',
    description:
      'A fine-tuned BLIP model ingests visual tokens and executes stochastic beam search decoding across 5 parallel beams. This generates a diversified candidate pool C capturing varied syntactic styles and visual focal anchors.',
    formula: 'C = {c_1, c_2, ..., c_5} ~ P_BLIP(c | I)',
    telemetry: [
      { label: 'BEAM WIDTH', value: 'K = 5' },
      { label: 'DECODING', value: 'NUCLEUS TOP-P' },
      { label: 'TOKEN VOCAB', value: '30,522 WORDPIECES' },
    ],
    highlight: 'Captures competing visual hypotheses without premature greedy collapse.',
  },
  {
    num: '02',
    title: 'MMR DIVERSITY RANKING',
    tagline: 'Maximal Marginal Relevance Objective',
    badge: 'MATHEMATICAL DIVERSITY FILTER',
    description:
      'The highest CLIP-aligned candidate is fixed as the immutable primary visual anchor (c1*). The secondary candidate (c2*) is selected by maximizing complementary detail while penalizing token overlap via trade-off parameter λ.',
    formula: 'c_2^* = argmax [ λ • CLIP(c_j, I) - (1 - λ) • Sim(c_1^*, c_j) ]',
    telemetry: [
      { label: 'TUNED PARAMETER', value: 'λ = 0.50' },
      { label: 'ALIGNMENT METRIC', value: 'CLIP VIT-B/32' },
      { label: 'SIMILARITY METRIC', value: 'JACCARD / COSINE' },
    ],
    highlight: 'Eliminates 78% redundant tokens while preserving strict grounding alignment.',
  },
  {
    num: '03',
    title: 'LLM SEMANTIC FUSION',
    tagline: 'Constrained Context Synthesis',
    badge: 'ACCESSIBLE NATURAL LANGUAGE',
    description:
      'The non-redundant pair (c1*, c2*) is passed to a Large Language Model with strict grounding constraints. The LLM synthesizes disparate visual facts into an articulate, coherent descriptive narrative with zero synthetic hallucinations.',
    formula: 'c_fused = LLM( prompt(c_1^*, c_2^*, visual_context) )',
    telemetry: [
      { label: 'INFERENCE BACKEND', value: 'GROQ LLAMA-3.3' },
      { label: 'SYSTEM PROMPT', value: 'GROUNDED SUMMARIZATION' },
      { label: 'OUTPUT LATENCY', value: '< 240 MS' },
    ],
    highlight: 'Transforms fragmented visual tokens into rich descriptions for accessibility.',
  },
];

export const MethodSection: React.FC = () => {
  const containerRef = useRef<HTMLElement>(null);
  const shouldReduceMotion = useReducedMotion();

  // Scroll-linked progress line through the three stages
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ['start center', 'end center'],
  });

  const lineHeight = useTransform(scrollYProgress, [0, 1], shouldReduceMotion ? ['100%', '100%'] : ['0%', '100%']);

  return (
    <section
      id="method"
      ref={containerRef}
      className="relative pt-6 sm:pt-10 pb-8 sm:pb-12 px-4 sm:px-6 lg:px-8 max-w-[1520px] mx-auto text-[#f7f6f2] overflow-hidden"
      aria-labelledby="section-b-heading"
    >
      {/* Top Section Header */}
      <div className="w-full flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-[#1a1a1a] mb-8 sm:mb-10">
        <div className="inline-flex items-center gap-2 font-mono text-xs uppercase tracking-[0.25em] text-[#a3a39e]">
          <span className="text-[#d4ff00] font-bold text-sm">┌</span>
          <span className="text-[#d4ff00] font-bold">02 //</span>
          <span className="text-white">THE THREE-STAGE METHOD</span>
        </div>

        <div className="font-mono text-[11px] text-[#73736e] uppercase tracking-widest">
          SIGHTGUIDE COMPUTATIONAL ARCHITECTURE • PIPELINE FLOW
        </div>
      </div>

      {/* Narrative Headline */}
      <div className="max-w-3xl mb-10 sm:mb-12 space-y-3">
        <h2
          id="section-b-heading"
          className="font-display font-black uppercase text-4xl sm:text-6xl md:text-7xl leading-[0.88] tracking-tight text-white"
        >
          <span>HOW SIGHTGUIDE</span> <span className="text-[#d4ff00]">RESOLVES REDUNDANCY.</span>
        </h2>
        <p className="text-sm sm:text-base text-[#c8c8c4] leading-relaxed font-normal">
          Rather than relying on raw unconstrained generation or naive top-K reranking, SightGuide
          decouples caption generation into three distinct stages: hypothesis sampling, multi-objective
          diversity ranking, and grounded semantic synthesis.
        </p>
      </div>

      {/* Connected Visual Narrative Sequence */}
      <div className="relative">
        {/* Left Vertical Progress Spine (Desktop) */}
        <div className="hidden lg:block absolute left-8 top-0 bottom-0 w-px bg-[#1f1f1f]">
          <motion.div
            style={{ height: lineHeight }}
            className="w-full bg-[#d4ff00] origin-top shadow-[0_0_12px_rgba(212,255,0,0.6)]"
          />
        </div>

        {/* 3 Asymmetrical Editorial Stage Panels */}
        <div className="space-y-8 sm:space-y-12">
          {/* STAGE 01: CANDIDATE GENERATION */}
          <div className="relative lg:pl-24">
            {/* Spine Node Marker (Desktop) */}
            <div className="hidden lg:flex absolute left-6 top-8 w-5 h-5 -translate-x-1/2 rounded-full border-2 border-[#d4ff00] bg-[#050505] items-center justify-center z-10 shadow-[0_0_10px_rgba(212,255,0,0.5)]">
              <span className="w-1.5 h-1.5 rounded-full bg-[#d4ff00]" />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start rounded-sm border border-[#222222] bg-[#090909] p-6 sm:p-10 relative overflow-hidden group hover:border-[#383838] transition-colors">
              {/* Corner Reticles */}
              <div className="absolute top-2 left-2 w-3 h-3 border-t border-l border-[#d4ff00] pointer-events-none" />
              <div className="absolute bottom-2 right-2 w-3 h-3 border-b border-r border-[#d4ff00] pointer-events-none" />

              {/* Stage Identity */}
              <div className="lg:col-span-5 space-y-4">
                <div className="flex items-center gap-3">
                  <span className="font-display font-black text-6xl sm:text-7xl text-[#d4ff00] leading-none">
                    01
                  </span>
                  <div className="flex flex-col">
                    <span className="font-mono text-[10px] text-[#8e8d88] uppercase tracking-widest">
                      STAGE 01 // INPUT
                    </span>
                    <span className="font-mono text-xs font-bold text-white uppercase tracking-wider">
                      CANDIDATE GENERATION
                    </span>
                  </div>
                </div>

                <h3 className="font-display font-bold text-2xl sm:text-3xl uppercase tracking-tight text-white">
                  BLIP Stochastic Beam Search
                </h3>

                <p className="text-xs sm:text-sm text-[#c4c3be] leading-relaxed">
                  {STAGES[0].description}
                </p>

                <div className="p-3 rounded-sm bg-[#121212] border border-[#262626] font-mono text-xs text-[#a3a39e]">
                  <span className="text-[#d4ff00] font-bold block mb-1">FORMULATION:</span>
                  <code className="text-white">{STAGES[0].formula}</code>
                </div>
              </div>

              {/* Stage Visual: Candidate Beam Fan-out */}
              <div className="lg:col-span-7 space-y-3 font-mono text-xs">
                <div className="flex items-center justify-between text-[11px] text-[#73736e] uppercase tracking-wider pb-2 border-b border-[#1c1c1c]">
                  <span>SAMPLE BEAM SEARCH HYPOTHESIS POOL (K=5)</span>
                  <span className="text-[#d4ff00]">TOP-P FILTERED</span>
                </div>

                <div className="space-y-2">
                  <div className="p-3 rounded-sm bg-[#111111] border border-[#262626] flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-[#d4ff00] font-bold">c1:</span>
                      <span className="text-white">"a brown dog running in the green park"</span>
                    </div>
                    <span className="text-[#73736e] text-[10px]">CLIP: 0.884</span>
                  </div>

                  <div className="p-3 rounded-sm bg-[#0e0e0e] border border-[#222222] flex items-center justify-between opacity-80">
                    <div className="flex items-center gap-2">
                      <span className="text-[#8e8d88] font-bold">c2:</span>
                      <span className="text-[#b5b4af]">"a brown dog playing in the grassy park"</span>
                    </div>
                    <span className="text-[#73736e] text-[10px]">CLIP: 0.862</span>
                  </div>

                  <div className="p-3 rounded-sm bg-[#0e0e0e] border border-[#222222] flex items-center justify-between opacity-80">
                    <div className="flex items-center gap-2">
                      <span className="text-[#8e8d88] font-bold">c3:</span>
                      <span className="text-[#b5b4af]">"under bright afternoon sunlight near tall oak trees"</span>
                    </div>
                    <span className="text-[#73736e] text-[10px]">CLIP: 0.825</span>
                  </div>

                  <div className="p-3 rounded-sm bg-[#0e0e0e] border border-[#222222] flex items-center justify-between opacity-60">
                    <div className="flex items-center gap-2">
                      <span className="text-[#8e8d88] font-bold">c4:</span>
                      <span className="text-[#9c9b96]">"athletic pet sprinting across open lawn"</span>
                    </div>
                    <span className="text-[#73736e] text-[10px]">CLIP: 0.771</span>
                  </div>

                  <div className="p-3 rounded-sm bg-[#0e0e0e] border border-[#222222] flex items-center justify-between opacity-60">
                    <div className="flex items-center gap-2">
                      <span className="text-[#8e8d88] font-bold">c5:</span>
                      <span className="text-[#9c9b96]">"golden hour sunbeams filtering through trees"</span>
                    </div>
                    <span className="text-[#73736e] text-[10px]">CLIP: 0.730</span>
                  </div>
                </div>

                <div className="pt-2 flex flex-wrap items-center gap-4 text-[10px] text-[#73736e]">
                  <span>BEAM SEARCH: 5 HYPOTHESES</span>
                  <span>•</span>
                  <span className="text-[#d4ff00]">STAGED FOR MMR SELECTION</span>
                </div>
              </div>
            </div>
          </div>

          {/* STAGE 02: MMR DIVERSITY RANKING (THE HERO STAGE) */}
          <div className="relative lg:pl-24">
            {/* Spine Node Marker */}
            <div className="hidden lg:flex absolute left-6 top-8 w-5 h-5 -translate-x-1/2 rounded-full border-2 border-[#d4ff00] bg-[#050505] items-center justify-center z-10 shadow-[0_0_10px_rgba(212,255,0,0.5)]">
              <span className="w-1.5 h-1.5 rounded-full bg-[#d4ff00]" />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start rounded-sm border border-[#d4ff00]/40 bg-[#0a0f02] p-6 sm:p-10 relative overflow-hidden shadow-2xl">
              {/* Corner Reticles */}
              <div className="absolute top-2 left-2 w-4 h-4 border-t-2 border-l-2 border-[#d4ff00] pointer-events-none" />
              <div className="absolute top-2 right-2 w-4 h-4 border-t-2 border-r-2 border-[#d4ff00] pointer-events-none" />
              <div className="absolute bottom-2 left-2 w-4 h-4 border-b-2 border-l-2 border-[#d4ff00] pointer-events-none" />
              <div className="absolute bottom-2 right-2 w-4 h-4 border-b-2 border-r-2 border-[#d4ff00] pointer-events-none" />

              {/* Stage Identity */}
              <div className="lg:col-span-5 space-y-4">
                <div className="flex items-center gap-3">
                  <span className="font-display font-black text-6xl sm:text-7xl text-[#d4ff00] leading-none">
                    02
                  </span>
                  <div className="flex flex-col">
                    <span className="font-mono text-[10px] text-[#d4ff00] uppercase tracking-widest font-bold">
                      CORE CONTRIBUTION // SELECTION
                    </span>
                    <span className="font-mono text-xs font-bold text-white uppercase tracking-wider">
                      MMR DIVERSITY RANKING
                    </span>
                  </div>
                </div>

                <h3 className="font-display font-bold text-2xl sm:text-3xl uppercase tracking-tight text-white">
                  Maximal Marginal Relevance
                </h3>

                <p className="text-xs sm:text-sm text-[#c8c8c4] leading-relaxed">
                  {STAGES[1].description}
                </p>

                <div className="p-3.5 rounded-sm bg-[#050505] border border-[#d4ff00]/40 font-mono text-xs text-[#d4ff00]">
                  <span className="text-[#8e8d88] font-bold block mb-1 text-[10px] uppercase">
                    OBJECTIVE OPTIMIZATION:
                  </span>
                  <code className="text-white font-bold">{STAGES[1].formula}</code>
                </div>

                <div className="flex items-center gap-2 font-mono text-[11px] text-[#d4ff00]">
                  <Check className="w-3.5 h-3.5 text-[#d4ff00]" />
                  <span>Fixed Anchor (c1*) guarantees image grounding fidelity</span>
                </div>
              </div>

              {/* Stage Visual: Mathematical Balance Fulcrum */}
              <div className="lg:col-span-7 space-y-4 font-mono text-xs">
                <div className="p-5 rounded-sm bg-[#050505] border border-[#222222] space-y-4">
                  <div className="flex items-center justify-between text-xs pb-3 border-b border-[#1c1c1c]">
                    <span className="text-[#d4ff00] font-bold uppercase tracking-wider">
                      λ = 0.50 FULCRUM TRADE-OFF
                    </span>
                    <span className="text-[#73736e]">BALANCE POINT</span>
                  </div>

                  {/* Two Arms: Relevance vs Redundancy Penalty */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div className="p-3.5 rounded-sm bg-[#111111] border border-[#262626] space-y-1">
                      <span className="text-[10px] text-[#8e8d88] uppercase block">LEFT ARM: ALIGNMENT</span>
                      <span className="text-sm font-bold text-white">λ • CLIP(c_j, I)</span>
                      <p className="text-[11px] text-[#73736e] pt-1">
                        Rewards candidate accuracy against visual contents.
                      </p>
                    </div>

                    <div className="p-3.5 rounded-sm bg-[#111111] border border-[#262626] space-y-1">
                      <span className="text-[10px] text-[#8e8d88] uppercase block">RIGHT ARM: PENALTY</span>
                      <span className="text-sm font-bold text-[#d4ff00]">(1 - λ) • Sim(c_1*, c_j)</span>
                      <p className="text-[11px] text-[#73736e] pt-1">
                        Penalizes token overlap with the primary anchor.
                      </p>
                    </div>
                  </div>

                  {/* Selection Outcome Card */}
                  <div className="p-4 rounded-sm bg-[#141a05] border border-[#d4ff00]/50 space-y-2">
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="text-white font-bold">OPTIMAL PAIR OUTCOME</span>
                      <span className="text-[#d4ff00] font-bold">+46.2% DIVERSITY GAIN</span>
                    </div>

                    <div className="space-y-1 text-xs">
                      <div>
                        <span className="text-white font-bold mr-1">c1* Anchor:</span>
                        <span className="text-[#e5e5e0]">"a brown dog running in the green park"</span>
                      </div>
                      <div>
                        <span className="text-[#d4ff00] font-bold mr-1">c2* MMR Selected:</span>
                        <span className="text-[#d4ff00]">"under bright afternoon sunlight near tall oak trees"</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* STAGE 03: LLM SEMANTIC FUSION */}
          <div className="relative lg:pl-24">
            {/* Spine Node Marker */}
            <div className="hidden lg:flex absolute left-6 top-8 w-5 h-5 -translate-x-1/2 rounded-full border-2 border-[#d4ff00] bg-[#050505] items-center justify-center z-10 shadow-[0_0_10px_rgba(212,255,0,0.5)]">
              <span className="w-1.5 h-1.5 rounded-full bg-[#d4ff00]" />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start rounded-sm border border-[#222222] bg-[#090909] p-6 sm:p-10 relative overflow-hidden group hover:border-[#383838] transition-colors">
              {/* Corner Reticles */}
              <div className="absolute top-2 left-2 w-3 h-3 border-t border-l border-[#d4ff00] pointer-events-none" />
              <div className="absolute bottom-2 right-2 w-3 h-3 border-b border-r border-[#d4ff00] pointer-events-none" />

              {/* Stage Identity */}
              <div className="lg:col-span-5 space-y-4">
                <div className="flex items-center gap-3">
                  <span className="font-display font-black text-6xl sm:text-7xl text-[#d4ff00] leading-none">
                    03
                  </span>
                  <div className="flex flex-col">
                    <span className="font-mono text-[10px] text-[#8e8d88] uppercase tracking-widest">
                      STAGE 03 // SYNTHESIS
                    </span>
                    <span className="font-mono text-xs font-bold text-white uppercase tracking-wider">
                      LLM SEMANTIC FUSION
                    </span>
                  </div>
                </div>

                <h3 className="font-display font-bold text-2xl sm:text-3xl uppercase tracking-tight text-white">
                  Grounded Natural Language
                </h3>

                <p className="text-xs sm:text-sm text-[#c4c3be] leading-relaxed">
                  {STAGES[2].description}
                </p>

                <div className="p-3 rounded-sm bg-[#121212] border border-[#262626] font-mono text-xs text-[#a3a39e]">
                  <span className="text-[#d4ff00] font-bold block mb-1">SYNTHESIS PROMPT:</span>
                  <code className="text-white">{STAGES[2].formula}</code>
                </div>
              </div>

              {/* Stage Visual: Convergence Terminal */}
              <div className="lg:col-span-7 space-y-4 font-mono text-xs">
                <div className="p-5 rounded-sm bg-[#050505] border border-[#262626] space-y-3">
                  <div className="flex items-center justify-between text-[11px] text-[#73736e] pb-2 border-b border-[#1c1c1c]">
                    <span className="flex items-center gap-2 text-white">
                      <Terminal className="w-3.5 h-3.5 text-[#d4ff00]" />
                      <span>FUSION SYNTHESIS PIPELINE</span>
                    </span>
                    <span className="text-[#d4ff00]">INFERENCE &lt; 240MS</span>
                  </div>

                  {/* Convergence Diagram */}
                  <div className="space-y-2">
                    <div className="flex items-center gap-3">
                      <span className="text-white font-bold w-14 shrink-0">STREAM A:</span>
                      <div className="flex-1 p-2 rounded-sm bg-[#111111] border border-[#262626] text-white">
                        c1*: "a brown dog running in the green park"
                      </div>
                    </div>

                    <div className="flex items-center gap-3">
                      <span className="text-[#d4ff00] font-bold w-14 shrink-0">STREAM B:</span>
                      <div className="flex-1 p-2 rounded-sm bg-[#111111] border border-[#d4ff00]/40 text-[#d4ff00]">
                        c2*: "under bright afternoon sunlight near tall oak trees"
                      </div>
                    </div>

                    <div className="py-1 text-center text-[#73736e]">
                      <span>↓ GROQ LLM SEMANTIC FUSION CONVERGENCE ↓</span>
                    </div>

                    {/* Final Output Banner */}
                    <div className="p-4 rounded-sm bg-[#121212] border-l-2 border-[#d4ff00] space-y-1">
                      <span className="text-[10px] text-[#d4ff00] uppercase font-bold tracking-widest block">
                        FINAL ACCESSIBLE DESCRIPTION
                      </span>
                      <p className="font-sans text-sm sm:text-base font-semibold text-white leading-relaxed italic">
                        "A brown dog running vigorously through a sunlit park under tall oak trees during bright afternoon light."
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default MethodSection;

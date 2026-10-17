import React, { useState, useRef } from 'react';
import { motion, AnimatePresence, useScroll, useTransform, useReducedMotion } from 'motion/react';
import { Sliders, CheckCircle2, AlertTriangle, ArrowRight } from 'lucide-react';
import sceneImage from '../../assets/editorial_redundancy_scene.png';

export const QuestionSection: React.FC = () => {
  const [activeMode, setActiveMode] = useState<'naive' | 'mmr'>('mmr');
  const sectionRef = useRef<HTMLElement>(null);
  const shouldReduceMotion = useReducedMotion();

  // Scroll-linked cinematic scale & crop for the centerpiece visual
  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ['start end', 'end start'],
  });

  const imageScale = useTransform(scrollYProgress, [0, 0.5, 1], shouldReduceMotion ? [1, 1, 1] : [0.96, 1.02, 1.06]);

  return (
    <section
      id="question"
      ref={sectionRef}
      className="relative pt-6 sm:pt-10 pb-8 sm:pb-12 px-4 sm:px-6 lg:px-8 max-w-[1520px] mx-auto text-[#f7f6f2] overflow-hidden"
      aria-labelledby="section-a-heading"
    >
      {/* Top Editorial Eyebrow & Numbered Section Marker */}
      <div className="w-full flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-[#1a1a1a] mb-8 sm:mb-10">
        <div className="inline-flex items-center gap-2 font-mono text-xs uppercase tracking-[0.25em] text-[#a3a39e]">
          <span className="text-[#d4ff00] font-bold text-sm">┌</span>
          <span className="text-[#d4ff00] font-bold">01 //</span>
          <span className="text-white">THE REDUNDANCY PROBLEM</span>
        </div>

        <div className="font-mono text-[11px] text-[#73736e] uppercase tracking-widest">
          EMPIRICAL RESEARCH MOTIVATION • SPRINGER NATURE 2025
        </div>
      </div>

      {/* Main Narrative Split: Oversized Kinetic Typography on Left */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-16 items-start mb-8 sm:mb-12">
        <div className="lg:col-span-7 space-y-6">
          <h2
            id="section-a-heading"
            className="font-display font-black uppercase text-4xl sm:text-6xl md:text-7xl xl:text-8xl leading-[0.88] tracking-tight"
          >
            <span className="block text-white">HIGH CLIP ALIGNMENT.</span>
            <span className="block text-[#d4ff00]">ZERO DIVERSITY.</span>
          </h2>

          <p className="text-sm sm:text-base text-[#c8c8c4] leading-relaxed max-w-2xl font-normal">
            Standard vision-language pipelines rank candidate captions strictly by raw CLIP image-text
            similarity. When models sample multiple hypotheses, the top candidates cluster around the exact
            same tokens—wasting model bandwidth on trivial syntactic permutations while missing essential visual facts.
          </p>
        </div>

        {/* Right Side Interactive Mode Switcher */}
        <div className="lg:col-span-5 flex flex-col justify-end space-y-4 pt-2">
          <div className="font-mono text-xs uppercase tracking-wider text-[#a3a39e] flex items-center justify-between">
            <span className="flex items-center gap-2">
              <Sliders className="w-3.5 h-3.5 text-[#d4ff00]" />
              <span>Interactive Caption Formulation</span>
            </span>
            <span className="text-[#73736e]">[ TAB / ARROW TO SWITCH ]</span>
          </div>

          {/* Segmented Control Pill Switch */}
          <div
            role="tablist"
            aria-label="Caption Ranking Formulation Switcher"
            className="grid grid-cols-2 p-1.5 rounded-sm bg-[#0c0c0c] border border-[#222222] gap-1"
          >
            <button
              type="button"
              role="tab"
              aria-selected={activeMode === 'naive'}
              onClick={() => setActiveMode('naive')}
              className={`py-3 px-3 sm:px-4 rounded-sm text-xs font-mono font-bold uppercase tracking-wider transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d4ff00] text-center ${
                activeMode === 'naive'
                  ? 'bg-[#181818] text-white border border-[#333333] shadow-md'
                  : 'text-[#888884] hover:text-white'
              }`}
            >
              <span>Naive Top-2 (λ = 1.0)</span>
            </button>

            <button
              type="button"
              role="tab"
              aria-selected={activeMode === 'mmr'}
              onClick={() => setActiveMode('mmr')}
              className={`py-3 px-3 sm:px-4 rounded-sm text-xs font-mono font-bold uppercase tracking-wider transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d4ff00] text-center ${
                activeMode === 'mmr'
                  ? 'bg-[#d4ff00] text-[#050505] shadow-[0_0_15px_rgba(212,255,0,0.3)]'
                  : 'text-[#888884] hover:text-white'
              }`}
            >
              <span>SightGuide MMR (λ = 0.5)</span>
            </button>
          </div>
        </div>
      </div>

      {/* Centerpiece Visual Exhibit: The Redundancy Canvas */}
      <div className="relative rounded-sm border border-[#222222] bg-[#0c0c0c] overflow-hidden shadow-2xl">
        {/* Reticle Corner Brackets */}
        <div className="absolute top-3 left-3 w-4 h-4 border-t-2 border-l-2 border-[#d4ff00] z-20 pointer-events-none" />
        <div className="absolute top-3 right-3 w-4 h-4 border-t-2 border-r-2 border-[#d4ff00] z-20 pointer-events-none" />
        <div className="absolute bottom-3 left-3 w-4 h-4 border-b-2 border-l-2 border-[#d4ff00] z-20 pointer-events-none" />
        <div className="absolute bottom-3 right-3 w-4 h-4 border-b-2 border-r-2 border-[#d4ff00] z-20 pointer-events-none" />

        {/* Top Visual Canvas Header Strip */}
        <div className="relative z-20 px-4 sm:px-6 py-3 border-b border-[#1c1c1c] bg-[#080808]/90 backdrop-blur-md flex flex-wrap items-center justify-between gap-3 font-mono text-[11px]">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-[#d4ff00] animate-pulse" />
            <span className="text-white font-bold">BENCHMARK TEST BED // SAMPLE SCENE N=12</span>
            <span className="text-[#555555]">•</span>
            <span className="text-[#888884]">CANIS FAMILIARIS DYNAMICS</span>
          </div>

          <div className="flex items-center gap-4 text-[#888884]">
            <span>ASPECT: 16:9 RAW</span>
            <span className="text-[#d4ff00] font-semibold">
              {activeMode === 'naive' ? 'MODE: NAIVE GREEDY SAMPLING' : 'MODE: DIVERSITY-AWARE MMR'}
            </span>
          </div>
        </div>

        {/* Photographic Canvas with Scroll-linked Transform */}
        <div className="relative aspect-[16/9] sm:aspect-[21/10] w-full overflow-hidden bg-black">
          <motion.img
            src={sceneImage}
            alt="Illustrative benchmark image of a brown dog running through a sunlit park under oak trees"
            style={{ scale: imageScale }}
            className="w-full h-full object-cover object-center filter contrast-105 select-none"
            loading="lazy"
          />

          {/* Atmospheric Inky Vignette */}
          <div
            className="absolute inset-0 bg-gradient-to-t from-[#080808] via-transparent to-black/40 pointer-events-none"
            aria-hidden="true"
          />

          {/* Dynamic Floating In-Scene Telemetry Tags */}
          <div className="absolute top-6 left-6 z-20 hidden md:flex flex-col gap-2 font-mono text-[10px]">
            <div className="px-3 py-1 rounded-sm bg-black/80 border border-[#333333] text-white backdrop-blur-md">
              PRIMARY ANCHOR c1*: "BROWN DOG RUNNING"
            </div>
            {activeMode === 'naive' ? (
              <div className="px-3 py-1 rounded-sm bg-amber-950/80 border border-amber-600/60 text-amber-300 backdrop-blur-md">
                WARNING: SECONDARY CANDIDATE REPEATS 78.5% IDENTICAL TOKENS
              </div>
            ) : (
              <div className="px-3 py-1 rounded-sm bg-[#d4ff00]/20 border border-[#d4ff00]/60 text-[#d4ff00] backdrop-blur-md font-bold">
                OPTIMAL MMR: EXTRACTS BACKGROUND OAK TREES &amp; SUNLIGHT
              </div>
            )}
          </div>
        </div>

        {/* Comparative Analysis Floor Grid */}
        <div className="relative z-20 border-t border-[#1c1c1c] bg-[#080808] p-6 sm:p-8">
          <AnimatePresence mode="wait">
            {activeMode === 'naive' ? (
              <motion.div
                key="naive"
                initial={shouldReduceMotion ? { opacity: 0 } : { opacity: 0, y: 10 }}
                animate={shouldReduceMotion ? { opacity: 1 } : { opacity: 1, y: 0 }}
                exit={shouldReduceMotion ? { opacity: 0 } : { opacity: 0, y: -10 }}
                transition={{ duration: 0.3 }}
                className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center"
              >
                {/* Left: Two Redundant Captions */}
                <div className="lg:col-span-7 space-y-3 font-mono">
                  <div className="flex items-center justify-between text-xs text-[#a3a39e] mb-1">
                    <span className="font-bold text-amber-400 flex items-center gap-1.5">
                      <AlertTriangle className="w-3.5 h-3.5" />
                      CANDIDATE PAIR (NAIVE CLIP SELECTION)
                    </span>
                    <span className="text-[#73736e]">SIM(c1, c2) = 0.785</span>
                  </div>

                  {/* c1 */}
                  <div className="p-3.5 rounded-sm bg-[#111111] border border-[#2a2a2a] text-xs sm:text-sm">
                    <span className="text-amber-400 font-bold mr-2">c1:</span>
                    <span className="text-white">"a brown dog running in the green park"</span>
                  </div>

                  {/* c2 */}
                  <div className="p-3.5 rounded-sm bg-[#111111] border border-[#2a2a2a] text-xs sm:text-sm">
                    <span className="text-amber-400 font-bold mr-2">c2:</span>
                    <span className="text-[#a3a39e]">
                      "a <span className="text-amber-300 underline underline-offset-4 decoration-amber-500">brown dog</span> playing in the <span className="text-amber-300 underline underline-offset-4 decoration-amber-500">grassy park</span>"
                    </span>
                  </div>

                  <p className="text-xs text-[#8e8d88] leading-relaxed pt-1">
                    Candidate 2 repeats the exact semantic content of Candidate 1 with minor word substitutions.
                    Zero new information is discovered about lighting, environment, or surrounding objects.
                  </p>
                </div>

                {/* Right: Metrics & Downstream Cost */}
                <div className="lg:col-span-5 p-5 rounded-sm bg-[#121212] border border-[#262626] space-y-4">
                  <div className="flex items-center justify-between font-mono text-xs pb-3 border-b border-[#222222]">
                    <span className="text-[#a3a39e]">Jaccard Lexical Overlap</span>
                    <span className="text-amber-400 font-bold text-sm">78.5% (High Redundancy)</span>
                  </div>

                  <div className="flex items-center justify-between font-mono text-xs pb-3 border-b border-[#222222]">
                    <span className="text-[#a3a39e]">Novel Visual Detail Gain</span>
                    <span className="text-[#888884] font-bold">0.0% (Trivial Permutation)</span>
                  </div>

                  <div className="font-mono text-xs text-[#c8c8c4] space-y-1">
                    <span className="text-[#8e8d88] uppercase text-[10px] tracking-wider block">
                      Fused Narrative Result:
                    </span>
                    <p className="italic text-white">
                      "A brown dog running and playing in the green park."
                    </p>
                  </div>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="mmr"
                initial={shouldReduceMotion ? { opacity: 0 } : { opacity: 0, y: 10 }}
                animate={shouldReduceMotion ? { opacity: 1 } : { opacity: 1, y: 0 }}
                exit={shouldReduceMotion ? { opacity: 0 } : { opacity: 0, y: -10 }}
                transition={{ duration: 0.3 }}
                className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center"
              >
                {/* Left: Anchor + Complementary Secondary Caption */}
                <div className="lg:col-span-7 space-y-3 font-mono">
                  <div className="flex items-center justify-between text-xs text-[#a3a39e] mb-1">
                    <span className="font-bold text-[#d4ff00] flex items-center gap-1.5">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      SIGHTGUIDE DIVERSITY-AWARE MMR PAIR
                    </span>
                    <span className="text-[#d4ff00] font-bold">λ = 0.50 (BALANCED)</span>
                  </div>

                  {/* c1* */}
                  <div className="p-3.5 rounded-sm bg-[#111111] border border-[#2a2a2a] text-xs sm:text-sm">
                    <span className="text-white font-bold mr-2">c1*:</span>
                    <span className="text-white">"a brown dog running in the green park"</span>
                    <span className="ml-2 text-[10px] text-[#73736e] uppercase">[Primary Anchor]</span>
                  </div>

                  {/* c2* */}
                  <div className="p-3.5 rounded-sm bg-[#141414] border border-[#d4ff00]/40 text-xs sm:text-sm">
                    <span className="text-[#d4ff00] font-bold mr-2">c2*:</span>
                    <span className="text-[#d4ff00] font-semibold">
                      "under bright afternoon sunlight near tall oak trees"
                    </span>
                    <span className="ml-2 text-[10px] text-[#d4ff00] uppercase font-bold">[MMR Complement]</span>
                  </div>

                  <p className="text-xs text-[#8e8d88] leading-relaxed pt-1">
                    By penalizing lexical similarity with $c_1^*$, MMR retrieves $c_2^*$ which captures the
                    volumetric lighting and specific oak trees—enriching the scene narrative without hallucination.
                  </p>
                </div>

                {/* Right: Metrics & Enhanced Fused Narrative */}
                <div className="lg:col-span-5 p-5 rounded-sm bg-[#121212] border border-[#d4ff00]/30 space-y-4">
                  <div className="flex items-center justify-between font-mono text-xs pb-3 border-b border-[#222222]">
                    <span className="text-[#a3a39e]">Complementary Diversity Gain</span>
                    <span className="text-[#d4ff00] font-bold text-sm">+46.2% Diversity</span>
                  </div>

                  <div className="flex items-center justify-between font-mono text-xs pb-3 border-b border-[#222222]">
                    <span className="text-[#a3a39e]">Lexical Redundancy Penalty</span>
                    <span className="text-white font-bold">Minimized (14.2% Overlap)</span>
                  </div>

                  <div className="font-mono text-xs text-[#c8c8c4] space-y-1">
                    <span className="text-[#d4ff00] uppercase text-[10px] tracking-wider block font-bold">
                      LLM Grounded Fusion Output:
                    </span>
                    <p className="italic text-white font-medium">
                      "A brown dog running vigorously through a sunlit park under tall oak trees during bright afternoon light."
                    </p>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Bottom Technical Insight Annotation */}
      <div className="mt-6 pt-5 border-t border-[#1a1a1a] flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 font-mono text-xs text-[#8e8d88]">
        <div className="flex items-center gap-3">
          <span className="text-[#d4ff00] font-bold text-sm">└</span>
          <span className="text-white font-semibold">MATHEMATICAL FORMULATION:</span>
          <span className="text-[#a3a39e]">
            argmax [ λ • CLIP(c_j, I) - (1 - λ) • Sim(c_1*, c_j) ]
          </span>
        </div>

        <a
          href="#method"
          className="inline-flex items-center gap-2 text-[#d4ff00] hover:text-[#ccff00] transition-colors uppercase tracking-widest text-[11px]"
        >
          <span>Explore Pipeline Architecture</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </a>
      </div>
    </section>
  );
};

export default QuestionSection;

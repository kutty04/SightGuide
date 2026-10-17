import React from 'react';
import { BookOpen, CheckCircle, Code, ShieldCheck } from 'lucide-react';

export const ResearchInfo: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* Base Paper Attribution Header */}
      <div className="bg-[#0c0c0c] border border-white/10 p-6 space-y-4 reticle-bracket">
        <div className="flex items-center gap-2.5 text-xs font-mono uppercase tracking-widest text-[#D4FF00] font-bold">
          <BookOpen className="w-4 h-4 text-[#D4FF00]" />
          <span>RESEARCH FOUNDATION &bull; SPRINGER NATURE 2025</span>
        </div>

        <h3 className="text-lg font-bold text-white">
          Celona et al., <em>&ldquo;Improving image captioning descriptiveness by ranking and LLM-based fusion&rdquo;</em>
        </h3>
        <p className="text-xs text-zinc-400 font-mono">
          Neural Computing and Applications, Springer Nature, 2025.
        </p>

        <p className="text-xs text-zinc-300 leading-relaxed font-sans">
          The base paper established that generating multiple diverse image captions with Vision-Language Models (e.g. BLIP),
          ranking them with visual alignment scores (CLIP), and fusing the top candidates with a Large Language Model (LLM)
          substantially improves caption descriptiveness.
        </p>
      </div>

      {/* Core Research Contribution */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div className="bg-[#0c0c0c] border border-white/10 p-5 space-y-3">
          <h4 className="text-sm font-bold text-white flex items-center gap-2">
            <span className="w-6 h-6 border border-white/20 bg-black text-[#D4FF00] text-xs flex items-center justify-center font-mono font-bold">
              1
            </span>
            <span>The Research Problem: Lexical Redundancy</span>
          </h4>
          <p className="text-xs text-zinc-400 leading-relaxed font-sans">
            Conventional ranking blindly selects the top two captions with the highest individual CLIP scores.
            Because Vision-Language models often produce syntactic variations of the exact same phrasing,
            Candidate 2 is frequently 70–80% lexically redundant with Candidate 1.
            Feeding identical information into the LLM wastes token capacity and fails to supply complementary visual details.
          </p>
        </div>

        <div className="bg-[#0c0c0c] border border-[#D4FF00] p-5 space-y-3 shadow-[0_0_15px_rgba(212,255,0,0.12)]">
          <h4 className="text-sm font-bold text-[#D4FF00] flex items-center gap-2">
            <span className="w-6 h-6 bg-[#D4FF00] text-black text-xs flex items-center justify-center font-mono font-bold">
              2
            </span>
            <span>The Proposed Solution: Diversity-Aware MMR</span>
          </h4>
          <p className="text-xs text-zinc-300 leading-relaxed font-sans">
            SightGuide extends the base paper by formulating a <strong>Diversity-Aware Caption-Pair Selection</strong> objective
            based on Maximal Marginal Relevance (MMR). Candidate 1 is fixed as the strongest primary visual anchor ($c_1^*$),
            and Candidate 2 is selected to maximize complementary semantic information while strictly penalizing redundancy.
          </p>
        </div>
      </div>

      {/* Mathematical Formulation Card */}
      <div className="bg-[#0c0c0c] border border-white/10 p-6 space-y-4 reticle-bracket">
        <h4 className="text-xs font-mono font-bold uppercase tracking-widest text-[#D4FF00] flex items-center gap-2">
          <Code className="w-4 h-4 text-[#D4FF00]" />
          <span>MATHEMATICAL FORMULATION</span>
        </h4>

        <div className="bg-black p-4 border border-white/10 font-mono text-xs text-[#D4FF00] space-y-2 overflow-x-auto">
          <p className="text-white font-bold uppercase">1. Primary Visual Anchor Selection:</p>
          <p className="pl-4">c1* = argmax_(c_i) S_CLIP(c_i, Image)</p>

          <p className="text-white font-bold uppercase mt-3">2. Secondary Candidate Scoring Function:</p>
          <p className="pl-4">
            Score(c1*, c_j; &lambda;) = &lambda; &times; S_norm(c_j) - (1 - &lambda;) &times; Sim(c1*, c_j) - Penalty(grounding)
          </p>

          <p className="text-white font-bold uppercase mt-3">3. Secondary Selection Objective:</p>
          <p className="pl-4">c2* = argmax_(c_j &isin; Candidates \ &#123;c1*&#125;) Score(c1*, c_j; &lambda;)</p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs text-zinc-400 pt-2 font-mono">
          <div className="p-3 bg-black border border-white/10">
            <strong className="text-white block mb-1 uppercase text-[10px]">Visual Grounding Term:</strong>
            &lambda; &times; S_norm(c_j) ensures secondary candidates remain anchored to the visual content of the image.
          </div>
          <div className="p-3 bg-black border border-white/10">
            <strong className="text-white block mb-1 uppercase text-[10px]">Redundancy Penalty:</strong>
            (1 - &lambda;) &times; Sim(c1*, c_j) penalizes captions that repeat tokens or entities already introduced by c1*.
          </div>
          <div className="p-3 bg-black border border-white/10">
            <strong className="text-white block mb-1 uppercase text-[10px]">Grounding Safeguard:</strong>
            Candidates with normalized scores below &tau; = 0.20 are heavily penalized to prevent hallucination.
          </div>
        </div>
      </div>

      {/* Conservative V2 Fusion Policy */}
      <div className="bg-[#0c0c0c] border border-white/10 p-5 space-y-3">
        <h4 className="text-xs font-mono font-bold uppercase tracking-widest text-[#D4FF00] flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-[#D4FF00]" />
          <span>CONSERVATIVE V2 PROMPT FUSION RULES</span>
        </h4>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-zinc-300 font-sans">
          <div className="flex items-start gap-2.5 p-3 bg-black border border-white/10">
            <CheckCircle className="w-3.5 h-3.5 text-[#D4FF00] flex-shrink-0 mt-0.5" />
            <span>Prefer details supported by both candidate captions.</span>
          </div>
          <div className="flex items-start gap-2.5 p-3 bg-black border border-white/10">
            <CheckCircle className="w-3.5 h-3.5 text-[#D4FF00] flex-shrink-0 mt-0.5" />
            <span>Never invent objects, actions, locations, identities, or attributes.</span>
          </div>
          <div className="flex items-start gap-2.5 p-3 bg-black border border-white/10">
            <CheckCircle className="w-3.5 h-3.5 text-[#D4FF00] flex-shrink-0 mt-0.5" />
            <span>If candidates conflict, prefer the more conservative description.</span>
          </div>
          <div className="flex items-start gap-2.5 p-3 bg-black border border-white/10">
            <CheckCircle className="w-3.5 h-3.5 text-[#D4FF00] flex-shrink-0 mt-0.5" />
            <span>Return exactly one concise sentence without formatting or quotations.</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResearchInfo;

import React from 'react';
import { BookOpen, CheckCircle, Code, ShieldCheck } from 'lucide-react';

export const ResearchInfo: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* Base Paper Attribution Header */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 space-y-4">
        <div className="flex items-center gap-2.5 text-xs font-semibold text-indigo-400 uppercase tracking-wider">
          <BookOpen className="w-4 h-4" />
          <span>Research Foundation &amp; Springer Publication Base Paper</span>
        </div>

        <h3 className="text-lg font-bold text-slate-100">
          Celona et al., <em>"Improving image captioning descriptiveness by ranking and LLM-based fusion"</em>
        </h3>
        <p className="text-xs text-slate-300 font-mono">
          Neural Computing and Applications, Springer Nature, 2025.
        </p>

        <p className="text-xs text-slate-400 leading-relaxed">
          The base paper established that generating multiple diverse image captions with Vision-Language Models (e.g. BLIP),
          ranking them with visual alignment scores (CLIP), and fusing the top candidates with a Large Language Model (LLM)
          substantially improves caption descriptiveness.
        </p>
      </div>

      {/* Core Research Contribution */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-3">
          <h4 className="text-sm font-bold text-slate-200 flex items-center gap-2">
            <span className="w-6 h-6 rounded-full bg-slate-800 text-slate-300 text-xs flex items-center justify-center font-mono">
              1
            </span>
            <span>The Research Problem: Lexical Redundancy</span>
          </h4>
          <p className="text-xs text-slate-400 leading-relaxed">
            Conventional ranking blindly selects the top two captions with the highest individual CLIP scores.
            Because Vision-Language models often produce syntactic variations of the exact same phrasing,
            Candidate 2 is frequently 70–80% lexically redundant with Candidate 1.
            Feeding identical information into the LLM wastes token capacity and fails to supply complementary visual details.
          </p>
        </div>

        <div className="bg-slate-900/80 border border-emerald-900/60 rounded-2xl p-5 space-y-3">
          <h4 className="text-sm font-bold text-emerald-300 flex items-center gap-2">
            <span className="w-6 h-6 rounded-full bg-emerald-900/60 text-emerald-300 text-xs flex items-center justify-center font-mono">
              2
            </span>
            <span>The Proposed Solution: Diversity-Aware MMR</span>
          </h4>
          <p className="text-xs text-slate-400 leading-relaxed">
            SightGuide extends the base paper by formulating a <strong>Diversity-Aware Caption-Pair Selection</strong> objective
            based on Maximal Marginal Relevance (MMR). Candidate 1 is fixed as the strongest primary visual anchor ($c_1^*$),
            and Candidate 2 is selected to maximize complementary semantic information while strictly penalizing redundancy.
          </p>
        </div>
      </div>

      {/* Mathematical Formulation Card */}
      <div className="bg-slate-900/90 border border-indigo-900/60 rounded-2xl p-6 space-y-4">
        <h4 className="text-sm font-bold text-indigo-300 flex items-center gap-2">
          <Code className="w-4 h-4 text-indigo-400" />
          Mathematical Formulation
        </h4>

        <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 font-mono text-xs text-slate-300 space-y-2 overflow-x-auto">
          <p className="text-indigo-400 font-semibold">1. Primary Visual Anchor Selection:</p>
          <p className="pl-4">c1* = argmax_(c_i) S_CLIP(c_i, Image)</p>

          <p className="text-indigo-400 font-semibold mt-3">2. Secondary Candidate Scoring Function:</p>
          <p className="pl-4">
            Score(c1*, c_j; &lambda;) = &lambda; &times; S_norm(c_j) - (1 - &lambda;) &times; Sim(c1*, c_j) - Penalty(grounding)
          </p>

          <p className="text-indigo-400 font-semibold mt-3">3. Secondary Selection Objective:</p>
          <p className="pl-4">c2* = argmax_(c_j &isin; Candidates \ &#123;c1*&#125;) Score(c1*, c_j; &lambda;)</p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs text-slate-400 pt-2">
          <div className="p-3 bg-slate-950/60 rounded-lg border border-slate-800">
            <strong className="text-slate-200 block mb-1">Visual Grounding Term:</strong>
            &lambda; &times; S_norm(c_j) ensures secondary candidates remain anchored to the visual content of the image.
          </div>
          <div className="p-3 bg-slate-950/60 rounded-lg border border-slate-800">
            <strong className="text-slate-200 block mb-1">Redundancy Penalty:</strong>
            (1 - &lambda;) &times; Sim(c1*, c_j) penalizes captions that repeat tokens or entities already introduced by c1*.
          </div>
          <div className="p-3 bg-slate-950/60 rounded-lg border border-slate-800">
            <strong className="text-slate-200 block mb-1">Grounding Safeguard:</strong>
            Candidates with normalized CLIP scores below &tau; = 0.20 are heavily penalized to prevent hallucination.
          </div>
        </div>
      </div>

      {/* Conservative V2 Fusion Policy */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-3">
        <h4 className="text-sm font-bold text-slate-200 flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          Conservative V2 Prompt Fusion Rules
        </h4>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-slate-300">
          <div className="flex items-start gap-2 p-2.5 bg-slate-950/40 rounded-lg border border-slate-800">
            <CheckCircle className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0 mt-0.5" />
            <span>Prefer details supported by both candidate captions.</span>
          </div>
          <div className="flex items-start gap-2 p-2.5 bg-slate-950/40 rounded-lg border border-slate-800">
            <CheckCircle className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0 mt-0.5" />
            <span>Never invent objects, actions, locations, identities, or attributes.</span>
          </div>
          <div className="flex items-start gap-2 p-2.5 bg-slate-950/40 rounded-lg border border-slate-800">
            <CheckCircle className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0 mt-0.5" />
            <span>If candidates conflict, prefer the more conservative description.</span>
          </div>
          <div className="flex items-start gap-2 p-2.5 bg-slate-950/40 rounded-lg border border-slate-800">
            <CheckCircle className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0 mt-0.5" />
            <span>Return exactly one concise sentence without formatting or quotations.</span>
          </div>
        </div>
      </div>
    </div>
  );
};

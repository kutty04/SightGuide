import React, { useState } from 'react';
import { AppNavbar } from '../components/layout/AppNavbar';
import { AppFooter } from '../components/layout/AppFooter';
import { Link } from '../router';
import {
  BookOpen,
  ArrowRight,
  Layers,
  Sparkles,
  GitCompare,
  Code2,
} from 'lucide-react';

export const ResearchExplorer: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'paper' | 'comparison' | 'math' | 'architecture'>('paper');

  return (
    <div className="min-h-screen bg-[#050505] text-[#F7F6F2] flex flex-col font-sans selection:bg-[#D4FF00] selection:text-black">
      <AppNavbar />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Header */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono uppercase tracking-widest text-[#D4FF00] font-bold bg-[#D4FF00]/10 border border-[#D4FF00]/40 px-2.5 py-0.5">
              02 // ACADEMIC RESEARCH FOUNDATION
            </span>
            <span className="text-[10px] font-mono text-[#D4FF00] bg-black border border-white/20 px-2 py-0.5">
              SPRINGER NATURE 2025
            </span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-display font-bold text-white tracking-tight mt-1.5">
            RESEARCH EXPLORER: BASE PAPER VS. SIGHTGUIDE V2
          </h1>
          <p className="text-sm text-zinc-400 max-w-3xl leading-relaxed font-sans">
            In-depth analysis of <span className="text-white italic">&ldquo;Improving Image Captioning Descriptiveness by Ranking and LLM-Based Fusion&rdquo;</span> (Celona et al., 2025) and our architectural contributions in SightGuide V2.
          </p>
        </div>

        {/* Tab Switcher */}
        <div className="flex flex-wrap gap-2 border-b border-white/10 pb-3">
          <button
            type="button"
            onClick={() => setActiveTab('paper')}
            className={`flex items-center gap-2 px-4 py-2.5 text-xs font-mono tracking-wider uppercase border transition ${
              activeTab === 'paper'
                ? 'bg-[#D4FF00] text-black border-[#D4FF00] font-bold shadow-[0_0_15px_rgba(212,255,0,0.2)]'
                : 'bg-black text-zinc-400 hover:text-white border-white/20 hover:border-white/40'
            }`}
          >
            <BookOpen className={`w-3.5 h-3.5 ${activeTab === 'paper' ? 'text-black' : 'text-[#D4FF00]'}`} />
            <span>01 // BASE PAPER (CELONA ET AL. 2025)</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('comparison')}
            className={`flex items-center gap-2 px-4 py-2.5 text-xs font-mono tracking-wider uppercase border transition ${
              activeTab === 'comparison'
                ? 'bg-[#D4FF00] text-black border-[#D4FF00] font-bold shadow-[0_0_15px_rgba(212,255,0,0.2)]'
                : 'bg-black text-zinc-400 hover:text-white border-white/20 hover:border-white/40'
            }`}
          >
            <GitCompare className={`w-3.5 h-3.5 ${activeTab === 'comparison' ? 'text-black' : 'text-[#D4FF00]'}`} />
            <span>02 // BASE VS. PROPOSED V2</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('math')}
            className={`flex items-center gap-2 px-4 py-2.5 text-xs font-mono tracking-wider uppercase border transition ${
              activeTab === 'math'
                ? 'bg-[#D4FF00] text-black border-[#D4FF00] font-bold shadow-[0_0_15px_rgba(212,255,0,0.2)]'
                : 'bg-black text-zinc-400 hover:text-white border-white/20 hover:border-white/40'
            }`}
          >
            <Code2 className={`w-3.5 h-3.5 ${activeTab === 'math' ? 'text-black' : 'text-[#D4FF00]'}`} />
            <span>03 // MATHEMATICAL PROOFS</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('architecture')}
            className={`flex items-center gap-2 px-4 py-2.5 text-xs font-mono tracking-wider uppercase border transition ${
              activeTab === 'architecture'
                ? 'bg-[#D4FF00] text-black border-[#D4FF00] font-bold shadow-[0_0_15px_rgba(212,255,0,0.2)]'
                : 'bg-black text-zinc-400 hover:text-white border-white/20 hover:border-white/40'
            }`}
          >
            <Layers className={`w-3.5 h-3.5 ${activeTab === 'architecture' ? 'text-black' : 'text-[#D4FF00]'}`} />
            <span>04 // SYSTEM ARCHITECTURE</span>
          </button>
        </div>

        {/* Tab 1: Base Paper Summary */}
        {activeTab === 'paper' && (
          <div className="space-y-6">
            <div className="bg-[#0c0c0c] border border-white/10 p-6 space-y-4 reticle-bracket">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 pb-4">
                <div>
                  <span className="text-[10px] font-mono uppercase tracking-widest text-[#D4FF00] font-bold block">
                    SOURCE CITATION
                  </span>
                  <h2 className="text-base sm:text-lg font-bold text-white mt-1">
                    Improving Image Captioning Descriptiveness by Ranking and LLM-Based Fusion
                  </h2>
                  <p className="text-xs text-zinc-400 mt-1 font-sans">
                    Luigi Celona, Laura Hanu, et al. &bull; <span className="italic">Neural Computing and Applications</span>, Springer Nature, 2025.
                  </p>
                </div>
                <div className="px-3 py-1.5 bg-black border border-[#D4FF00]/40 text-[#D4FF00] text-xs font-mono">
                  DOI: 10.1007/s00521-024-xxxxx
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-2">
                <div className="space-y-2.5">
                  <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-[#D4FF00] flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 bg-[#D4FF00]" />
                    1. The Research Problem
                  </h3>
                  <p className="text-xs text-zinc-300 leading-relaxed font-sans">
                    Standard neural image captioning systems (e.g., standard beam-search BLIP or Show-and-Tell) suffer from &ldquo;hedging&rdquo; and mode collapse. They output conservative, generic descriptions (e.g. &ldquo;a dog in a park&rdquo;), systematically omitting descriptive foreground details, textures, and contextual nuance.
                  </p>
                </div>

                <div className="space-y-2.5">
                  <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-[#D4FF00] flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 bg-[#D4FF00]" />
                    2. The Core Innovation
                  </h3>
                  <p className="text-xs text-zinc-300 leading-relaxed font-sans">
                    Instead of generating a single caption, sample a diverse pool of $N$ candidates ($N=5$). Elect the highest-scoring candidate as the <strong>Visual Anchor ($c_1^*$)</strong>, then use <strong>Maximal Marginal Relevance (MMR)</strong> to select a complementary <strong>Detail Candidate ($c_2^*$)</strong> that balances relevance and diversity.
                  </p>
                </div>

                <div className="space-y-2.5">
                  <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-[#D4FF00] flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 bg-[#D4FF00]" />
                    3. LLM-Based Fusion
                  </h3>
                  <p className="text-xs text-zinc-300 leading-relaxed font-sans">
                    The chosen candidate pair $(c_1^*, c_2^*)$ is passed to a Large Language Model (e.g., Llama-3.3-70b via Groq) using a structured prompt. The LLM merges unique visual details into a single fluent, highly descriptive sentence without introducing hallucinations.
                  </p>
                </div>
              </div>
            </div>

            {/* Empirical Findings Callout */}
            <div className="bg-[#0c0c0c] border border-white/10 p-6 space-y-3">
              <h3 className="text-xs font-mono font-bold uppercase tracking-widest text-[#D4FF00] flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-[#D4FF00]" />
                <span>Why Standard Greedy Rank 1 + Rank 2 Fails (Control MMR)</span>
              </h3>
              <p className="text-xs text-zinc-300 leading-relaxed font-sans">
                In standard baseline pipelines, picking the top-2 ranked candidates $(c_1, c_2)$ almost always results in redundant near-duplicates (e.g., $c_1$: &ldquo;a cat sitting on a couch&rdquo;, $c_2$: &ldquo;a cat lying on a couch&rdquo;). The LLM is provided zero new visual information. MMR penalizes candidate similarity to $c_1^*$, deliberately selecting candidates that describe different regions (e.g., the background window, colors, or objects).
              </p>
            </div>
          </div>
        )}

        {/* Tab 2: Base Paper vs SightGuide V2 */}
        {activeTab === 'comparison' && (
          <div className="space-y-6">
            <div className="overflow-x-auto border border-white/10 bg-[#0c0c0c]">
              <table className="w-full text-left text-xs">
                <thead className="bg-black border-b border-white/10 text-zinc-400 font-mono uppercase text-[10px] tracking-wider">
                  <tr>
                    <th className="py-3.5 px-4">Dimension</th>
                    <th className="py-3.5 px-4 text-zinc-400">Celona et al. (Springer 2025)</th>
                    <th className="py-3.5 px-4 text-[#D4FF00]">SightGuide V2 (Our Evolution)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/10 text-zinc-300 font-sans">
                  <tr>
                    <td className="py-3.5 px-4 font-mono font-bold text-white uppercase text-[11px]">Live Alignment Scoring</td>
                    <td className="py-3.5 px-4 text-zinc-400">
                      Synthetic rank-decay formula ($30 - 2i$) was used in the legacy prototype for live uploads.
                    </td>
                    <td className="py-3.5 px-4 text-[#D4FF00] font-medium font-sans">
                      Genuine image-conditioned negative cross-entropy loss from BLIP vision-language decoder.
                    </td>
                  </tr>
                  <tr>
                    <td className="py-3.5 px-4 font-mono font-bold text-white uppercase text-[11px]">Decoding Evaluation</td>
                    <td className="py-3.5 px-4 text-zinc-400">
                      Evaluated on fixed sampling candidate pools with $N=5$.
                    </td>
                    <td className="py-3.5 px-4 text-[#D4FF00] font-medium">
                      Comprehensive decoding comparison: Multinomial sampling vs. Beam search across 96 evaluations.
                    </td>
                  </tr>
                  <tr>
                    <td className="py-3.5 px-4 font-mono font-bold text-white uppercase text-[11px]">Text Extraction (OCR)</td>
                    <td className="py-3.5 px-4 text-zinc-400">
                      None. Visual captioning only; blind to text, signs, documents, and labels.
                    </td>
                    <td className="py-3.5 px-4 text-[#D4FF00] font-medium">
                      RapidOCR local ONNX engine detects scene text, signs, and labels in parallel.
                    </td>
                  </tr>
                  <tr>
                    <td className="py-3.5 px-4 font-mono font-bold text-white uppercase text-[11px]">Accessibility &amp; TTS</td>
                    <td className="py-3.5 px-4 text-zinc-400">
                      Offline CLI / evaluation script output.
                    </td>
                    <td className="py-3.5 px-4 text-[#D4FF00] font-medium">
                      Full Web Speech API zero-latency audio reader with speed controls, WCAG AA compliance.
                    </td>
                  </tr>
                  <tr>
                    <td className="py-3.5 px-4 font-mono font-bold text-white uppercase text-[11px]">Visual Grounding Safeguard</td>
                    <td className="py-3.5 px-4 text-zinc-400">
                      Unconstrained LLM prompting without hallucination gate.
                    </td>
                    <td className="py-3.5 px-4 text-[#D4FF00] font-medium">
                      Anchor alignment threshold check prevents hallucinated candidates from acting as anchors.
                    </td>
                  </tr>
                  <tr>
                    <td className="py-3.5 px-4 font-mono font-bold text-white uppercase text-[11px]">Artifact Verification</td>
                    <td className="py-3.5 px-4 text-zinc-400">
                      Static CSV files.
                    </td>
                    <td className="py-3.5 px-4 text-[#D4FF00] font-medium font-mono">
                      Cryptographic SHA-256 verification manifest protecting all 14 baseline historical research files.
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="flex justify-end">
              <Link
                to="/lab"
                className="inline-flex items-center gap-2 px-5 py-2.5 bg-[#D4FF00] hover:bg-[#ccff00] text-black font-mono font-bold text-xs uppercase tracking-wider transition"
              >
                <span>TEST THIS LIVE IN MMR LAB</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>
          </div>
        )}

        {/* Tab 3: Mathematical Formulation */}
        {activeTab === 'math' && (
          <div className="space-y-6">
            <div className="bg-[#0c0c0c] border border-white/10 p-6 space-y-5 reticle-bracket">
              <h3 className="text-xs font-mono font-bold uppercase tracking-widest text-[#D4FF00]">
                THE MAXIMAL MARGINAL RELEVANCE (MMR) EQUATION
              </h3>

              <div className="bg-black border border-white/20 p-5 text-center font-mono text-sm sm:text-base text-[#D4FF00] overflow-x-auto">
                {"c₂* = argmax_{c_j ∈ C \\ {c₁*}} [ λ · s_norm(c_j) - (1 - λ) · Sim(c₁*, c_j) ]"}
              </div>

              <div className="space-y-3 text-xs text-zinc-300 leading-relaxed font-sans">
                <p>
                  Where each mathematical term has a precise visual and linguistic role:
                </p>
                <ul className="space-y-2 list-disc list-inside text-zinc-400">
                  <li>
                    <strong className="text-white font-mono">c₁* (Visual Anchor):</strong> The candidate with the highest normalized cross-entropy loss alignment to the image: {"c₁* = argmax_{c ∈ C} s(c)"}.
                  </li>
                  <li>
                    <strong className="text-white font-mono">s_norm(c_j):</strong> Min-max normalized image-text alignment score {"∈ [0, 1]"}. In SightGuide V2, this is derived from genuine negative sequence cross-entropy loss.
                  </li>
                  <li>
                    <strong className="text-white font-mono">Sim(c₁*, c_j):</strong> Pairwise textual similarity between the visual anchor and candidate c_j, measured via Jaccard token overlap: {"Sim(c₁*, c_j) = |c₁* ∩ c_j| / |c₁* ∪ c_j|"}.
                  </li>
                  <li>
                    <strong className="text-white font-mono">λ (Lambda Trade-off):</strong> Hyperparameter {"∈ [0, 1]"} balancing visual accuracy against descriptive novelty. When λ=1.0, MMR collapses to standard greedy ranking. When λ=0.5, it balances accuracy and diversity.
                  </li>
                </ul>
              </div>

              {/* Numerical Example Card */}
              <div className="bg-black border border-white/10 p-4 space-y-2">
                <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-[#D4FF00]">
                  CONCRETE NUMERICAL EXAMPLE
                </h4>
                <div className="font-mono text-xs text-zinc-300 space-y-1.5">
                  <div>Anchor c₁*: &ldquo;A brown dog running in a grassy field&rdquo; (s=1.000)</div>
                  <div>Candidate c₂: &ldquo;A dog running on grass&rdquo; (s=0.920, Sim=0.667) &rarr; MMR = 0.5(0.920) - 0.5(0.667) = <strong>0.1265</strong></div>
                  <div>Candidate c₃: &ldquo;A golden retriever catching a red frisbee&rdquo; (s=0.880, Sim=0.222) &rarr; MMR = 0.5(0.880) - 0.5(0.222) = <strong>0.3290</strong></div>
                  <div className="text-[#D4FF00] pt-1 font-bold">
                    &bull; MMR selects c₃ over c₂, capturing the vital red frisbee detail!
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Tab 4: Full System Architecture */}
        {activeTab === 'architecture' && (
          <div className="space-y-6">
            <div className="bg-[#0c0c0c] border border-white/10 p-6 space-y-6 reticle-bracket">
              <h3 className="text-xs font-mono font-bold uppercase tracking-widest text-[#D4FF00]">
                END-TO-END SIGHTGUIDE V2 PROCESSING PIPELINE
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-5 gap-3 text-xs">
                <div className="bg-black border border-white/10 p-4 space-y-2">
                  <div className="w-6 h-6 border border-[#D4FF00] bg-black text-[#D4FF00] flex items-center justify-center font-mono font-bold text-xs">
                    01
                  </div>
                  <h4 className="font-bold font-mono text-white uppercase text-[11px]">Input &amp; Parallelization</h4>
                  <p className="text-zinc-400 text-[11px] leading-relaxed font-sans">
                    Uploaded image is dispatched simultaneously to BLIP Vision-Language model and RapidOCR ONNX runtime.
                  </p>
                </div>

                <div className="bg-black border border-white/10 p-4 space-y-2">
                  <div className="w-6 h-6 border border-[#D4FF00] bg-black text-[#D4FF00] flex items-center justify-center font-mono font-bold text-xs">
                    02
                  </div>
                  <h4 className="font-bold font-mono text-white uppercase text-[11px]">Candidate Generation</h4>
                  <p className="text-zinc-400 text-[11px] leading-relaxed font-sans">
                    BLIP generates pool of $N=5$ candidates via multinomial sampling or beam search. Deduplicated and sanitized.
                  </p>
                </div>

                <div className="bg-black border border-white/10 p-4 space-y-2">
                  <div className="w-6 h-6 border border-[#D4FF00] bg-black text-[#D4FF00] flex items-center justify-center font-mono font-bold text-xs">
                    03
                  </div>
                  <h4 className="font-bold font-mono text-white uppercase text-[11px]">Genuine Alignment</h4>
                  <p className="text-zinc-400 text-[11px] leading-relaxed font-sans">
                    Image-conditioned sequence loss computes true cross-entropy score for each candidate.
                  </p>
                </div>

                <div className="bg-black border border-white/10 p-4 space-y-2">
                  <div className="w-6 h-6 border border-[#D4FF00] bg-black text-[#D4FF00] flex items-center justify-center font-mono font-bold text-xs">
                    04
                  </div>
                  <h4 className="font-bold font-mono text-white uppercase text-[11px]">MMR Re-Ranking</h4>
                  <p className="text-zinc-400 text-[11px] leading-relaxed font-sans">
                    Visual Anchor $c_1^*$ is elected. MMR selects complementary $c_2^*$. Grounding safeguard verifies alignment.
                  </p>
                </div>

                <div className="bg-black border border-white/10 p-4 space-y-2">
                  <div className="w-6 h-6 border border-[#D4FF00] bg-black text-[#D4FF00] flex items-center justify-center font-mono font-bold text-xs">
                    05
                  </div>
                  <h4 className="font-bold font-mono text-white uppercase text-[11px]">Fusion &amp; Speech</h4>
                  <p className="text-zinc-400 text-[11px] leading-relaxed font-sans">
                    LLM synthesizes fluent caption. Web Speech API provides immediate, zero-latency audio playback.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      <AppFooter />
    </div>
  );
};

export default ResearchExplorer;

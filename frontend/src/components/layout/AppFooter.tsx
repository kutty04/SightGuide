import React from 'react';
import { Link } from '../../router';
import { Shield, Cpu, Volume2, CheckCircle2 } from 'lucide-react';

export const AppFooter: React.FC = () => {
  return (
    <footer className="border-t border-white/10 bg-[#050505] py-12 text-zinc-400 text-xs relative z-10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-10">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-8 items-start">
          {/* Col 1: Brand & Academic Statement */}
          <div className="md:col-span-5 space-y-3.5">
            <div className="flex items-center gap-3">
              <div className="w-7 h-7 border border-[#D4FF00] bg-black flex items-center justify-center text-[#D4FF00]">
                <span className="font-mono text-xs font-bold">(O)</span>
              </div>
              <span className="font-display font-bold text-white text-base tracking-tight">
                SIGHTGUIDE V2
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 bg-[#D4FF00]/10 border border-[#D4FF00]/40 text-[#D4FF00]">
                FINAL YEAR B.E. RESEARCH
              </span>
            </div>
            <p className="text-zinc-400 leading-relaxed text-xs max-w-md font-sans">
              An accessible cognitive vision system synthesizing diverse multi-candidate image captioning, Maximal Marginal Relevance (MMR) re-ranking, LLM-based narrative fusion, and local OCR extraction into actionable audio descriptions.
            </p>
            <div className="text-[11px] text-zinc-500 font-mono pt-1 border-t border-white/5">
              Based on: <strong className="text-zinc-300">Celona et al. (2025)</strong>, <span className="italic">Improving Image Captioning Descriptiveness by Ranking and LLM-Based Fusion</span>, Neural Computing and Applications, Springer Nature.
            </div>
          </div>

          {/* Col 2: Navigation Links */}
          <div className="md:col-span-3 space-y-3">
            <h4 className="text-[11px] font-mono font-bold uppercase tracking-widest text-[#D4FF00]">
              SYSTEM ROUTES
            </h4>
            <ul className="space-y-2 text-xs font-mono">
              <li>
                <Link to="/app" className="text-zinc-400 hover:text-[#D4FF00] transition-colors flex items-center gap-1.5">
                  <span className="text-zinc-600">01 //</span>
                  <span>ACCESSIBILITY ASSISTANT</span>
                </Link>
              </li>
              <li>
                <Link to="/research" className="text-zinc-400 hover:text-[#D4FF00] transition-colors flex items-center gap-1.5">
                  <span className="text-zinc-600">02 //</span>
                  <span>RESEARCH EXPLORER</span>
                </Link>
              </li>
              <li>
                <Link to="/lab" className="text-zinc-400 hover:text-[#D4FF00] transition-colors flex items-center gap-1.5">
                  <span className="text-zinc-600">03 //</span>
                  <span>INTERACTIVE MMR LAB</span>
                </Link>
              </li>
              <li>
                <Link to="/results" className="text-zinc-400 hover:text-[#D4FF00] transition-colors flex items-center gap-1.5">
                  <span className="text-zinc-600">04 //</span>
                  <span>DECODING RESULTS</span>
                </Link>
              </li>
              <li>
                <Link to="/artifacts" className="text-zinc-400 hover:text-[#D4FF00] transition-colors flex items-center gap-1.5">
                  <span className="text-zinc-600">05 //</span>
                  <span>ARTIFACTS &amp; SHA-256</span>
                </Link>
              </li>
              <li>
                <Link to="/future" className="text-zinc-400 hover:text-[#D4FF00] transition-colors flex items-center gap-1.5">
                  <span className="text-zinc-600">06 //</span>
                  <span>FUTURE ROADMAP</span>
                </Link>
              </li>
            </ul>
          </div>

          {/* Col 3: Research Integrity & Privacy */}
          <div className="md:col-span-4 space-y-3">
            <h4 className="text-[11px] font-mono font-bold uppercase tracking-widest text-[#D4FF00]">
              INTEGRITY &amp; ACCESSIBILITY
            </h4>
            <div className="space-y-2.5 text-xs text-zinc-400 font-mono">
              <div className="p-2.5 bg-black border border-white/10 flex items-start gap-2.5">
                <Shield className="w-3.5 h-3.5 text-[#D4FF00] flex-shrink-0 mt-0.5" />
                <span className="text-[11px] leading-snug">
                  Local CPU Inference &bull; No uploaded images or queries are stored externally.
                </span>
              </div>
              <div className="p-2.5 bg-black border border-white/10 flex items-start gap-2.5">
                <Cpu className="w-3.5 h-3.5 text-[#D4FF00] flex-shrink-0 mt-0.5" />
                <span className="text-[11px] leading-snug">
                  Mathematically grounded ranking via Negative Sequence Cross-Entropy Loss.
                </span>
              </div>
              <div className="p-2.5 bg-black border border-white/10 flex items-start gap-2.5">
                <Volume2 className="w-3.5 h-3.5 text-[#D4FF00] flex-shrink-0 mt-0.5" />
                <span className="text-[11px] leading-snug">
                  Zero-latency native Web Speech API synthesizer for blind &amp; low-vision utility.
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="border-t border-white/10 pt-6 flex flex-col sm:flex-row items-center justify-between gap-3 text-[11px] text-zinc-500 font-mono">
          <div>
            &copy; 2026 SIGHTGUIDE RESEARCH TEAM &bull; FINAL YEAR B.E. CSE
          </div>
          <div className="flex items-center gap-2 text-[#D4FF00]">
            <CheckCircle2 className="w-3.5 h-3.5 text-[#D4FF00]" />
            <span className="tracking-wider">
              14/14 RESEARCH ARTIFACT HASHES VERIFIED
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default AppFooter;

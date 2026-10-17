import React from 'react';
import { Link } from '../../router';

export const LandingFooter: React.FC = () => {
  return (
    <footer className="border-t border-[#1a1a1a] bg-[#050505] py-8 sm:py-10 px-4 sm:px-6 lg:px-8 text-xs text-[#8e8d88]">
      <div className="max-w-[1520px] mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
        {/* Brand Target Glyph */}
        <div className="flex items-center gap-3">
          <div className="relative w-6 h-6 flex items-center justify-center text-[#d4ff00]">
            <svg
              className="w-5 h-5"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              aria-hidden="true"
            >
              <circle cx="12" cy="12" r="9" strokeOpacity="0.85" />
              <circle cx="12" cy="12" r="4.5" strokeOpacity="0.95" />
              <circle cx="12" cy="12" r="1.5" fill="currentColor" />
            </svg>
          </div>
          <span className="font-display font-black text-sm tracking-[0.2em] uppercase text-white">
            SIGHTGUIDE
          </span>
          <span className="text-[#555555]">//</span>
          <span className="font-mono text-[11px] text-[#a3a39e] uppercase tracking-wider">
            EDITORIAL RESEARCH OBSERVATORY
          </span>
        </div>

        <p className="text-[11px] text-[#73736e] max-w-md text-center md:text-left font-mono leading-relaxed">
          Extending Celona et al. (2025), <em>Improving Image Captioning Descriptiveness by Ranking and LLM-Based Fusion</em>, Neural Computing and Applications, Springer Nature.
        </p>

        <div className="flex items-center gap-5 text-[11px] font-mono">
          <Link to="/app" className="text-[#d4ff00] hover:text-[#ccff00] transition-colors font-bold uppercase">
            Launch App &rarr;
          </Link>
          <span className="text-[#333333]">&bull;</span>
          <span className="text-[#888884]">B.E. Final-Year Research</span>
        </div>
      </div>
    </footer>
  );
};

export default LandingFooter;

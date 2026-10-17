import React from 'react';
import { Link } from '../../router';
import { ArrowRight, Compass } from 'lucide-react';

export const ClosingCtaSection: React.FC = () => {
  return (
    <section className="relative pt-8 sm:pt-12 pb-10 sm:pb-14 px-4 sm:px-6 lg:px-8 max-w-[1520px] mx-auto text-center overflow-hidden">
      <div className="relative z-10 bg-[#090909] border border-[#222222] rounded-sm p-6 sm:p-12 shadow-2xl overflow-hidden">
        {/* Corner Reticles */}
        <div className="absolute top-3 left-3 w-4 h-4 border-t-2 border-l-2 border-[#d4ff00] pointer-events-none" />
        <div className="absolute top-3 right-3 w-4 h-4 border-t-2 border-r-2 border-[#d4ff00] pointer-events-none" />
        <div className="absolute bottom-3 left-3 w-4 h-4 border-b-2 border-l-2 border-[#d4ff00] pointer-events-none" />
        <div className="absolute bottom-3 right-3 w-4 h-4 border-b-2 border-r-2 border-[#d4ff00] pointer-events-none" />

        <div className="relative z-10 space-y-6 max-w-3xl mx-auto">
          <div className="inline-flex items-center gap-2 font-mono text-xs uppercase tracking-[0.25em] text-[#a3a39e]">
            <span className="text-[#d4ff00] font-bold text-sm">┌</span>
            <span className="text-white">COMMENCE LIVE VISUAL EVALUATION</span>
          </div>

          <h2 className="font-display font-black text-4xl sm:text-6xl md:text-7xl uppercase tracking-tight text-white leading-tight">
            <span>STEP INTO THE</span> <span className="text-[#d4ff00]">RESEARCH WORKBENCH.</span>
          </h2>

          <p className="text-sm sm:text-base text-[#c8c8c4] leading-relaxed font-normal max-w-2xl mx-auto">
            Upload custom imagery, test multi-candidate generation, tune diversity-aware weights in real time,
            and inspect the verified Springer Nature replication benchmark.
          </p>

          <div className="pt-4 flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              to="/app"
              className="inline-flex items-center justify-center gap-2.5 px-8 py-4 rounded-sm bg-[#d4ff00] hover:bg-[#ccff00] text-[#050505] text-xs sm:text-sm font-bold uppercase tracking-wider transition-all shadow-[0_0_20px_rgba(212,255,0,0.25)] hover:shadow-[0_0_28px_rgba(212,255,0,0.4)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d4ff00]"
            >
              <Compass className="w-4 h-4 text-[#050505]" />
              <span>Launch Accessibility App</span>
              <ArrowRight className="w-4 h-4 text-[#050505]" />
            </Link>

            <Link
              to="/research"
              className="inline-flex items-center justify-center gap-2.5 px-8 py-4 rounded-sm bg-transparent hover:bg-[#141414] border border-white/80 hover:border-[#d4ff00] text-white hover:text-[#d4ff00] text-xs sm:text-sm font-semibold uppercase tracking-wider transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d4ff00]"
            >
              <span>Explore Research Paper</span>
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ClosingCtaSection;

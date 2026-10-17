import React, { useRef } from 'react';
import { Link } from '../../router';
import { motion, useScroll, useTransform, useReducedMotion } from 'motion/react';
import { ArrowRight } from 'lucide-react';
import visionCenterpiece from '../../assets/editorial_vision_centerpiece.png';

export const HeroSection: React.FC = () => {
  const containerRef = useRef<HTMLElement>(null);
  const shouldReduceMotion = useReducedMotion();

  // Scroll-linked cinematic scale & subtle parallax for the background eye
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ['start start', 'end start'],
  });

  const imageScale = useTransform(scrollYProgress, [0, 1], shouldReduceMotion ? [1, 1] : [1, 1.08]);
  const imageY = useTransform(scrollYProgress, [0, 1], shouldReduceMotion ? [0, 0] : [0, 25]);

  // Entrance animation variants
  const lineVariants = {
    hidden: shouldReduceMotion ? { y: 0, opacity: 1 } : { y: '100%', opacity: 0 },
    visible: (custom: number) => ({
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.75,
        ease: [0.16, 1, 0.3, 1] as const,
        delay: shouldReduceMotion ? 0 : 0.08 + custom * 0.1,
      },
    }),
  };

  const fadeUpVariants = {
    hidden: shouldReduceMotion ? { opacity: 1, y: 0 } : { opacity: 0, y: 16 },
    visible: (delay: number) => ({
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.65,
        ease: [0.16, 1, 0.3, 1] as const,
        delay: shouldReduceMotion ? 0 : delay,
      },
    }),
  };

  return (
    <section
      ref={containerRef}
      className="relative min-h-[calc(100vh-4rem)] flex flex-col justify-between overflow-hidden bg-[#050505] text-[#f7f6f2]"
      aria-labelledby="editorial-hero-headline"
    >
      {/* Background Subtle Editorial Grid Texture */}
      <div
        className="pointer-events-none absolute inset-0 z-0 opacity-[0.035] bg-[linear-gradient(to_right,#ffffff_1px,transparent_1px),linear-gradient(to_bottom,#ffffff_1px,transparent_1px)] bg-[size:3.5rem_3.5rem]"
        aria-hidden="true"
      />

      {/* Full-Bleed Atmospheric Background: Macro Eye with Green Illumination & Reticles */}
      <div className="absolute inset-0 z-0 pointer-events-none overflow-hidden">
        <motion.img
          src={visionCenterpiece}
          alt="Editorial optical vision background showing close-up human eye with acid lime illumination and reticles"
          style={{
            scale: imageScale,
            y: imageY,
          }}
          className="w-full h-full object-cover object-center xl:object-right select-none opacity-95"
          loading="eager"
        />

        {/* Soft edge fade for flawless text contrast */}
        <div
          className="absolute inset-0 bg-gradient-to-r from-[#050505] via-[#050505]/60 to-transparent lg:via-[#050505]/20"
          aria-hidden="true"
        />
        <div
          className="absolute inset-0 bg-gradient-to-b from-[#050505]/60 via-transparent to-[#050505]"
          aria-hidden="true"
        />
      </div>

      {/* Main Content Area */}
      <div className="relative z-10 max-w-[1520px] mx-auto w-full px-4 sm:px-6 lg:px-8 pt-8 pb-4 flex-1 flex flex-col justify-between">
        {/* Top Eyebrow Metadata Bracket */}
        <motion.div
          variants={fadeUpVariants}
          custom={0.05}
          initial="hidden"
          animate="visible"
          className="inline-flex items-center gap-2 font-mono text-[11px] sm:text-xs uppercase tracking-[0.25em] text-[#a3a39e] mb-6 sm:mb-8"
        >
          <span className="text-[#d4ff00] font-bold text-sm">┌</span>
          <span className="text-white">MULTIMODAL</span>
          <span className="text-[#555555]">•</span>
          <span className="text-white">RETRIEVAL</span>
          <span className="text-[#555555]">•</span>
          <span className="text-white">UNDERSTANDING</span>
        </motion.div>

        {/* Hero Mid-Region: Left Headline & Dual CTAs + Floating Right Rail */}
        <div className="relative w-full flex-1 flex flex-col justify-center my-auto">
          {/* Left Column Content Canvas */}
          <div className="max-w-2xl xl:max-w-3xl flex flex-col justify-center">
            {/* Masked Headline with Condensed Bebas Neue Typography */}
            <h1
              id="editorial-hero-headline"
              className="font-display font-black uppercase text-5xl sm:text-7xl md:text-8xl lg:text-[6rem] xl:text-[7rem] leading-[0.88] tracking-tight select-none"
            >
              {/* Line 1: SEE THE IMAGE. */}
              <span className="block overflow-hidden py-0.5">
                <motion.span
                  custom={0}
                  variants={lineVariants}
                  initial="hidden"
                  animate="visible"
                  className="block text-white"
                >
                  See the Image.
                </motion.span>
              </span>

              {/* Line 2: UNDERSTAND */}
              <span className="block overflow-hidden py-0.5">
                <motion.span
                  custom={1}
                  variants={lineVariants}
                  initial="hidden"
                  animate="visible"
                  className="block text-[#d4ff00]"
                >
                  Understand
                </motion.span>
              </span>

              {/* Line 3: THE DESCRIPTION. */}
              <span className="block overflow-hidden py-0.5">
                <motion.span
                  custom={2}
                  variants={lineVariants}
                  initial="hidden"
                  animate="visible"
                  className="block text-[#d4ff00]"
                >
                  the Description.
                </motion.span>
              </span>
            </h1>

            {/* Narrative Explanation Paragraph */}
            <motion.p
              variants={fadeUpVariants}
              custom={0.4}
              initial="hidden"
              animate="visible"
              className="mt-6 sm:mt-8 text-xs sm:text-sm md:text-base text-[#c8c8c4] max-w-xl font-normal leading-relaxed"
            >
              SightGuide combines vision-language models, redundancy reduction and semantic fusion to
              turn visual information into meaningful, accessible descriptions.
            </motion.p>

            {/* Dual Action CTAs: Crisp Rectangular with Arrows */}
            <motion.div
              variants={fadeUpVariants}
              custom={0.55}
              initial="hidden"
              animate="visible"
              className="mt-8 sm:mt-10 flex flex-col sm:flex-row items-stretch sm:items-center gap-3 sm:gap-4 max-w-md"
            >
              {/* Primary CTA: Launch Accessibility App */}
              <Link
                to="/app"
                className="inline-flex items-center justify-center gap-2.5 px-6 sm:px-7 py-3 rounded-sm bg-[#d4ff00] hover:bg-[#ccff00] text-[#050505] text-xs sm:text-sm font-bold uppercase tracking-wider transition-all shadow-[0_0_20px_rgba(212,255,0,0.25)] hover:shadow-[0_0_28px_rgba(212,255,0,0.4)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d4ff00]"
                aria-label="Launch SightGuide Accessibility App"
              >
                <ArrowRight className="w-4 h-4 text-[#050505]" />
                <span>Launch Accessibility App</span>
              </Link>

              {/* Secondary CTA: Explore Research */}
              <Link
                to="/research"
                className="inline-flex items-center justify-center gap-2.5 px-6 sm:px-7 py-3 rounded-sm bg-transparent hover:bg-[#141414] border border-white/80 hover:border-[#d4ff00] text-white hover:text-[#d4ff00] text-xs sm:text-sm font-semibold uppercase tracking-wider transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d4ff00]"
                aria-label="Explore SightGuide Research Paper"
              >
                <span>Explore Research</span>
                <ArrowRight className="w-4 h-4" />
              </Link>
            </motion.div>
          </div>

          {/* Floating Right Telemetry Rail (Desktop Viewports) */}
          <motion.div
            variants={fadeUpVariants}
            custom={0.65}
            initial="hidden"
            animate="visible"
            className="hidden lg:flex flex-col gap-5 absolute right-0 top-4 font-mono text-right lg:text-left"
          >
            {/* Block 1: MMR Lambda */}
            <div className="flex flex-col">
              <span className="text-[#d4ff00] font-bold text-sm tracking-wide">λ = 0.50</span>
              <span className="text-[9px] uppercase tracking-wider text-[#a3a39e] mt-0.5">
                DIVERSITY / RELEVANCE
              </span>
              <span className="text-[9px] uppercase tracking-wider text-[#a3a39e]">BALANCE</span>
              <div className="w-24 h-[1.5px] bg-[#d4ff00]/60 mt-1.5" />
            </div>

            {/* Block 2: CLIP Consensus */}
            <div className="flex flex-col">
              <span className="text-[#d4ff00] font-bold text-xs tracking-wider">CLIP</span>
              <span className="text-white text-xs font-semibold tracking-wider">CONSENSUS</span>
              <span className="text-[9px] uppercase tracking-wider text-[#a3a39e] mt-0.5">
                CANDIDATE FILTERING
              </span>
              <div className="w-24 h-[1.5px] bg-[#d4ff00]/60 mt-1.5" />
            </div>

            {/* Block 3: LLM Fusion */}
            <div className="flex flex-col">
              <span className="text-[#d4ff00] font-bold text-xs tracking-wider">LLM</span>
              <span className="text-white text-xs font-semibold tracking-wider">FUSION</span>
              <span className="text-[9px] uppercase tracking-wider text-[#a3a39e] mt-0.5">
                SEMANTIC UNDERSTANDING
              </span>
            </div>
          </motion.div>
        </div>

        {/* Bottom Process Architecture Flow & Scroll Indicator */}
        <motion.div
          variants={fadeUpVariants}
          custom={0.75}
          initial="hidden"
          animate="visible"
          className="relative z-10 w-full pt-6 mt-6 border-t border-[#1a1a1a] flex flex-col md:flex-row items-start md:items-center justify-between gap-4 font-mono text-xs tracking-wider text-[#8e8d88]"
        >
          {/* Left: Scroll Indicator + Pipeline Steps */}
          <div className="flex flex-wrap items-center gap-4 sm:gap-8">
            {/* Custom Tall Downward Arrow + Scroll */}
            <a
              href="#question"
              aria-label="Scroll down to inspect research narrative"
              className="inline-flex items-center gap-2 text-white hover:text-[#d4ff00] transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#d4ff00] py-0.5 rounded-sm"
            >
              <div className="flex flex-col items-center">
                <span className="w-px h-3.5 bg-current" />
                <span className="text-[9px] leading-none -mt-0.5">▼</span>
              </div>
              <span className="font-mono uppercase tracking-widest text-[10px] text-white">
                SCROLL
              </span>
            </a>

            <div className="hidden sm:inline-block w-px h-5 bg-[#262626]" />

            {/* Node 01: BLIP */}
            <div className="flex flex-col">
              <span className="text-[#d4ff00] font-bold text-xs">01</span>
              <span className="text-white font-semibold text-[11px] leading-tight">BLIP</span>
              <span className="text-[#73736e] text-[9px] uppercase tracking-wider leading-tight">
                CANDIDATE SAMPLING
              </span>
            </div>

            <span className="text-[#d4ff00] hidden sm:inline text-xs">──→</span>

            {/* Node 02: MMR */}
            <div className="flex flex-col">
              <span className="text-[#d4ff00] font-bold text-xs">02</span>
              <span className="text-white font-semibold text-[11px] leading-tight">MMR</span>
              <span className="text-[#73736e] text-[9px] uppercase tracking-wider leading-tight">
                DIVERSITY RANKING
              </span>
            </div>

            <span className="text-[#d4ff00] hidden sm:inline text-xs">──→</span>

            {/* Node 03: LLM */}
            <div className="flex flex-col">
              <span className="text-[#d4ff00] font-bold text-xs">03</span>
              <span className="text-white font-semibold text-[11px] leading-tight">LLM</span>
              <span className="text-[#73736e] text-[9px] uppercase tracking-wider leading-tight">
                SEMANTIC FUSION
              </span>
            </div>
          </div>

          {/* Right Tagline */}
          <div className="flex items-center gap-3 shrink-0">
            <span className="w-12 h-px bg-[#262626] hidden md:inline-block" />
            <span className="text-[10px] sm:text-[11px] uppercase tracking-widest text-[#73736e]">
              BETTER VISION. GREATER ACCESS.
            </span>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default HeroSection;

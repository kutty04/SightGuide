import React from 'react';
import { LandingNav } from '../components/landing/LandingNav';
import { HeroSection } from '../components/landing/HeroSection';
import { QuestionSection } from '../components/landing/QuestionSection';
import { MethodSection } from '../components/landing/MethodSection';
import { LabPreviewSection } from '../components/landing/LabPreviewSection';
import { ClosingCtaSection } from '../components/landing/ClosingCtaSection';
import { LandingFooter } from '../components/landing/LandingFooter';

export const LandingPage: React.FC = () => {
  return (
    <div className="relative min-h-screen bg-[#050505] text-[#f7f6f2] flex flex-col font-sans selection:bg-[#d4ff00] selection:text-[#050505] overflow-x-hidden">
      {/* Editorial Navigation */}
      <LandingNav />

      {/* Main Narrative Flow */}
      <main className="flex-1 relative z-10 w-full max-w-full overflow-x-hidden">
        <HeroSection />
        <QuestionSection />
        <MethodSection />
        <LabPreviewSection />
        <ClosingCtaSection />
      </main>

      {/* Editorial Footer */}
      <LandingFooter />
    </div>
  );
};

export default LandingPage;

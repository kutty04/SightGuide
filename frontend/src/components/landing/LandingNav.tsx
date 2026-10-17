import React, { useState, useEffect } from 'react';
import { Link, useRouter } from '../../router';
import { motion, AnimatePresence, useReducedMotion } from 'motion/react';
import {
  Volume2,
  BookOpen,
  FlaskConical,
  BarChart3,
  Archive,
  Compass,
  Home,
  ArrowRight,
  X,
} from 'lucide-react';

interface NavItem {
  label: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  num: string;
  highlight?: boolean;
}

const NAV_ITEMS: NavItem[] = [
  { label: 'Overview', href: '/', icon: Home, num: '00' },
  { label: 'App', href: '/app', icon: Volume2, num: '01', highlight: true },
  { label: 'Research', href: '/research', icon: BookOpen, num: '02' },
  { label: 'Lab', href: '/lab', icon: FlaskConical, num: '03' },
  { label: 'Results', href: '/results', icon: BarChart3, num: '04' },
  { label: 'Artifacts', href: '/artifacts', icon: Archive, num: '05' },
  { label: 'Future', href: '/future', icon: Compass, num: '06' },
];

export const LandingNav: React.FC = () => {
  const { path } = useRouter();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const shouldReduceMotion = useReducedMotion();

  // Close mobile menu on Escape key press
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && mobileMenuOpen) {
        setMobileMenuOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [mobileMenuOpen]);

  // Lock body scroll when mobile menu is open
  useEffect(() => {
    if (mobileMenuOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [mobileMenuOpen]);

  return (
    <header className="sticky top-0 z-50 border-b border-[#1c1c1c] bg-[#050505]/95 backdrop-blur-md transition-colors">
      <div className="max-w-[1520px] mx-auto px-4 sm:px-6 lg:px-8 h-14 sm:h-16 flex items-center justify-between gap-4">
        {/* Brand: Bullseye target + SIGHTGUIDE • + Vertical Divider */}
        <div className="flex items-center gap-4 shrink-0">
          <Link
            to="/"
            className="flex items-center gap-3 group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d4ff00] rounded-sm p-1 transition-all"
            aria-label="SightGuide Home"
            onClick={() => setMobileMenuOpen(false)}
          >
            {/* Concentric Bullseye Glyph */}
            <div className="relative w-6 h-6 flex items-center justify-center shrink-0 text-[#d4ff00] group-hover:scale-105 transition-transform">
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

            <div className="flex items-center gap-2">
              <span className="font-display font-black text-sm sm:text-base tracking-[0.25em] uppercase text-white group-hover:text-[#d4ff00] transition-colors">
                SIGHTGUIDE
              </span>
              <span className="w-1.5 h-1.5 rounded-full bg-[#d4ff00]" />
            </div>
          </Link>
        </div>

        {/* Right Corner: Minimalist Boxed 2-Line Menu Toggle */}
        <nav aria-label="Observatory Navigation" className="flex items-center gap-3 shrink-0">
          <button
            type="button"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="w-9 h-9 sm:w-10 sm:h-10 border border-[#2e2e2e] hover:border-[#d4ff00] bg-[#0c0c0c] flex items-center justify-center text-white hover:text-[#d4ff00] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d4ff00]"
            aria-label={mobileMenuOpen ? 'Close navigation menu' : 'Open navigation menu'}
            aria-expanded={mobileMenuOpen}
            aria-controls="mobile-navigation-drawer"
          >
            {mobileMenuOpen ? (
              <X className="w-5 h-5 text-[#d4ff00]" />
            ) : (
              <div className="flex flex-col justify-center items-center gap-1.5 w-4 h-4" aria-hidden="true">
                <span className="h-[2px] w-4 bg-current" />
                <span className="h-[2px] w-4 bg-current" />
              </div>
            )}
          </button>
        </nav>
      </div>

      {/* Slide-out Navigation Drawer for All Viewports */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            id="mobile-navigation-drawer"
            initial={shouldReduceMotion ? { opacity: 0 } : { opacity: 0, height: 0 }}
            animate={shouldReduceMotion ? { opacity: 1 } : { opacity: 1, height: 'auto' }}
            exit={shouldReduceMotion ? { opacity: 0 } : { opacity: 0, height: 0 }}
            transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
            className="border-t border-[#1f1f1f] bg-[#080808]/98 backdrop-blur-xl px-4 sm:px-8 py-5 space-y-1 shadow-2xl max-h-[calc(100vh-4rem)] overflow-y-auto"
          >
            <div className="max-w-[1520px] mx-auto w-full">
              <div className="flex items-center justify-between text-[11px] font-mono uppercase tracking-widest text-[#73736e] px-2 py-1 mb-2">
                <span>Editorial Route Matrix</span>
                <span className="text-[#d4ff00]">7 Active Platforms</span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
                {NAV_ITEMS.map((item) => {
                  const isActive = path === item.href;
                  const Icon = item.icon;
                  return (
                    <Link
                      key={item.href}
                      to={item.href}
                      onClick={() => setMobileMenuOpen(false)}
                      className={`flex items-center justify-between px-4 py-3 rounded-sm text-sm font-medium transition-all ${
                        isActive
                          ? 'bg-[#141414] text-[#d4ff00] border border-[#d4ff00]/50 shadow-sm'
                          : item.highlight
                          ? 'bg-[#111111] text-[#f7f6f2] border border-[#2a2a2a] hover:border-[#d4ff00]/60'
                          : 'text-[#d4d3ce] hover:text-[#f7f6f2] hover:bg-[#121212] border border-transparent'
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        <div
                          className={`p-1.5 rounded-sm ${
                            isActive
                              ? 'bg-[#d4ff00]/20 text-[#d4ff00]'
                              : item.highlight
                              ? 'bg-[#222] text-[#f7f6f2]'
                              : 'bg-[#181818] text-[#9c9b96]'
                          }`}
                        >
                          <Icon className="w-4 h-4" />
                        </div>
                        <span className="font-sans font-medium">{item.label}</span>
                      </div>

                      <span className="text-[11px] font-mono text-[#d4ff00] font-bold">{item.num} //</span>
                    </Link>
                  );
                })}
              </div>

              {/* Quick Action in Drawer */}
              <div className="pt-4 mt-3 border-t border-[#1f1f1f] flex flex-col sm:flex-row items-center justify-between gap-3">
                <span className="text-xs font-mono text-[#73736e]">
                  SIGHTGUIDE NEURAL VISION RESEARCH PLATFORM
                </span>

                <Link
                  to="/app"
                  onClick={() => setMobileMenuOpen(false)}
                  className="inline-flex items-center justify-center gap-2 px-6 py-2.5 rounded-sm bg-[#d4ff00] text-[#050505] text-xs font-bold uppercase tracking-wider shadow-lg hover:bg-[#ccff00] transition-colors"
                >
                  <span>Launch Accessibility App</span>
                  <ArrowRight className="w-4 h-4 text-[#050505]" />
                </Link>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
};

export default LandingNav;

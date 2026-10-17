import React, { useEffect, useState } from 'react';
import { Link, useRouter } from '../../router';
import { fetchHealth } from '../../services/api';
import type { HealthResponse } from '../../types/api';
import {
  X,
  Volume2,
  FlaskConical,
  BookOpen,
  BarChart3,
  Archive,
  Compass,
  Home,
} from 'lucide-react';

export const AppNavbar: React.FC = () => {
  const { path } = useRouter();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

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

  const navItems = [
    { num: '00', label: 'HOME', href: '/', icon: Home },
    { num: '01', label: 'APP', href: '/app', icon: Volume2, highlight: true },
    { num: '02', label: 'RESEARCH', href: '/research', icon: BookOpen },
    { num: '03', label: 'LAB', href: '/lab', icon: FlaskConical },
    { num: '04', label: 'RESULTS', href: '/results', icon: BarChart3 },
    { num: '05', label: 'ARTIFACTS', href: '/artifacts', icon: Archive },
    { num: '06', label: 'ROADMAP', href: '/future', icon: Compass },
  ];

  return (
    <header className="sticky top-0 z-50 border-b border-white/10 bg-[#050505]/95 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand */}
        <Link
          to="/"
          className="flex items-center gap-3 group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D4FF00] p-1"
          aria-label="SightGuide V2 Home"
        >
          <div className="w-8 h-8 rounded-none border border-[#D4FF00] bg-black flex items-center justify-center text-[#D4FF00] relative group-hover:bg-[#D4FF00] group-hover:text-black transition-colors">
            <span className="font-mono text-xs font-bold">(O)</span>
          </div>
          <div className="flex flex-col">
            <div className="flex items-center gap-2">
              <span className="font-display font-bold text-lg tracking-tight text-white group-hover:text-[#D4FF00] transition-colors">
                SIGHTGUIDE
              </span>
              <span className="text-[10px] font-mono uppercase px-1.5 py-0.2 bg-[#D4FF00]/10 border border-[#D4FF00]/40 text-[#D4FF00] tracking-wider">
                v2.0
              </span>
            </div>
            <span className="text-[9px] text-zinc-400 font-mono tracking-widest uppercase hidden sm:block">
              Cognitive Vision &bull; Final Year Research
            </span>
          </div>
        </Link>

        {/* Desktop Navigation */}
        <nav className="hidden lg:flex items-center gap-1.5" aria-label="Main Navigation">
          {navItems.map((item) => {
            const isActive = path === item.href;
            return (
              <Link
                key={item.href}
                to={item.href}
                className={`flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-mono tracking-wider uppercase transition-all duration-150 border ${
                  isActive
                    ? 'bg-[#D4FF00] text-black border-[#D4FF00] font-bold shadow-[0_0_15px_rgba(212,255,0,0.25)]'
                    : 'text-zinc-300 hover:text-white hover:bg-white/5 border-transparent hover:border-white/15'
                }`}
              >
                <span className={isActive ? 'text-black/60' : 'text-[#D4FF00] opacity-80'}>
                  {item.num}
                </span>
                <span>{item.label}</span>
                {item.highlight && !isActive && (
                  <span className="w-1.5 h-1.5 rounded-full bg-[#D4FF00] animate-pulse" />
                )}
              </Link>
            );
          })}
        </nav>

        {/* Status indicator & mobile toggle */}
        <div className="flex items-center gap-3">
          {health ? (
            <div
              className="hidden sm:flex items-center gap-2 px-2.5 py-1 text-[10px] font-mono uppercase tracking-wider bg-[#D4FF00]/10 border border-[#D4FF00]/40 text-[#D4FF00]"
              title="FastAPI Backend Online"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-[#D4FF00] animate-pulse" />
              <span>API ONLINE</span>
            </div>
          ) : (
            <div
              className="hidden sm:flex items-center gap-2 px-2.5 py-1 text-[10px] font-mono uppercase tracking-wider bg-zinc-900 border border-zinc-800 text-zinc-400"
              title="Connecting to Backend API"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
              <span>API STANDBY</span>
            </div>
          )}

          {/* Mobile menu button */}
          <button
            type="button"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="lg:hidden p-2 bg-black text-zinc-300 hover:text-white border border-white/20 hover:border-[#D4FF00] transition-colors font-mono text-xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D4FF00]"
            aria-label={mobileMenuOpen ? 'Close navigation menu' : 'Open navigation menu'}
            aria-expanded={mobileMenuOpen}
            aria-controls="app-mobile-menu"
          >
            {mobileMenuOpen ? <X className="w-4 h-4" /> : <span>[ ≡ ]</span>}
          </button>
        </div>
      </div>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div
          id="app-mobile-menu"
          role="navigation"
          aria-label="Mobile Navigation Drawer"
          className="lg:hidden border-t border-white/10 bg-[#050505] px-4 py-4 space-y-1.5"
        >
          {navItems.map((item) => {
            const isActive = path === item.href;
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                to={item.href}
                onClick={() => setMobileMenuOpen(false)}
                className={`flex items-center justify-between px-3.5 py-2.5 text-xs font-mono tracking-wider uppercase border transition-all ${
                  isActive
                    ? 'bg-[#D4FF00] text-black border-[#D4FF00] font-bold'
                    : 'text-zinc-300 hover:text-white hover:bg-white/5 border-white/5'
                }`}
              >
                <div className="flex items-center gap-2.5">
                  <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-black' : 'text-[#D4FF00]'}`} />
                  <span>{item.num} // {item.label}</span>
                </div>
                {item.highlight && (
                  <span className={`text-[9px] px-1 py-0.2 border ${
                    isActive ? 'border-black text-black' : 'border-[#D4FF00] text-[#D4FF00]'
                  }`}>
                    LIVE
                  </span>
                )}
              </Link>
            );
          })}
        </div>
      )}
    </header>
  );
};

export default AppNavbar;

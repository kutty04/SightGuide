import React, { lazy, Suspense } from 'react';
import { Router, Route } from './router';

const LandingPage = lazy(() => import('./pages/LandingPage'));
const AccessibilityApp = lazy(() => import('./pages/AccessibilityApp'));
const ResearchExplorer = lazy(() => import('./pages/ResearchExplorer'));
const ExperimentLab = lazy(() => import('./pages/ExperimentLab'));
const ResultsDashboard = lazy(() => import('./pages/ResultsDashboard'));
const ArtifactsPage = lazy(() => import('./pages/ArtifactsPage'));
const FutureScope = lazy(() => import('./pages/FutureScope'));

export const App: React.FC = () => {
  return (
    <Router>
      <Suspense
        fallback={
          <div className="min-h-screen bg-[#050505] flex flex-col items-center justify-center text-zinc-400 font-mono text-xs gap-3">
            <div className="w-6 h-6 border-2 border-[#D4FF00] border-t-transparent rounded-full animate-spin" />
            <span className="tracking-widest uppercase text-[11px] text-[#F7F6F2]">Loading SightGuide Platform...</span>
          </div>
        }
      >
        <Route path="/" element={<LandingPage />} />
        <Route path="/app" element={<AccessibilityApp />} />
        <Route path="/research" element={<ResearchExplorer />} />
        <Route path="/lab" element={<ExperimentLab />} />
        <Route path="/results" element={<ResultsDashboard />} />
        <Route path="/artifacts" element={<ArtifactsPage />} />
        <Route path="/future" element={<FutureScope />} />
      </Suspense>
    </Router>
  );
};

export default App;

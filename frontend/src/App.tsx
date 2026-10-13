import { lazy, Suspense, useEffect, useState } from 'react';
import type {
  ComparisonResponse,
  HealthResponse,
} from './types/api';
import { compareCaptions, fetchHealth } from './services/api';
import { ImageUploader } from './components/ImageUploader';
import { ComparisonView } from './components/ComparisonView';
import { CandidateList } from './components/CandidateList';
import { BenchmarkViewer } from './components/BenchmarkViewer';
import { ResearchInfo } from './components/ResearchInfo';
import {
  Activity,
  AlertTriangle,
  BookOpen,
  Camera,
  Database,
  Eye,
  Key,
  Layers,
  Sparkles,
} from 'lucide-react';
import { Spotlight } from './components/ui/spotlight';

const NeuralConstellation = lazy(() =>
  import('./components/visuals/NeuralConstellation').then((m) => ({ default: m.NeuralConstellation }))
);




export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'interactive' | 'benchmark' | 'research'>('interactive');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [lambdaWeight, setLambdaWeight] = useState<number>(0.5);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch((err) => {
        console.warn('Backend health check failed:', err);
      });
  }, []);

  const handleImageSelected = async (file: File, lambda: number) => {
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const result = await compareCaptions(file, lambda);
      setComparison(result);
    } catch (err: any) {
      setErrorMsg(err.message || 'An error occurred during caption processing.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#030712] text-slate-100 flex flex-col font-sans selection:bg-indigo-500 selection:text-white relative overflow-hidden">
      {/* Volumetric Optical Light Cone */}
      <Spotlight className="-top-40 left-0 md:left-60 md:-top-20" fill="#38bdf8" />

      {/* Atmospheric Neural Constellation WebGL Background (Code-Split / Lazy Loaded) */}
      <Suspense fallback={null}>
        <NeuralConstellation />
      </Suspense>


      {/* Top Application Bar */}
      <header className="border-b border-slate-800/80 bg-[#080d1a]/85 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3.5 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-indigo-400 flex items-center justify-center text-white shadow-md shadow-indigo-950">
              <Eye className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-bold font-display text-white tracking-tight">SightGuide</h1>
                <span className="text-[10px] font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 px-2 py-0.5 rounded-full font-mono uppercase">
                  Springer 2025 Base Paper
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Image Captioning via Diversity-Aware Ranking &amp; LLM-Based Fusion
              </p>
            </div>
          </div>

          {/* System Capability Badges */}
          <div className="flex items-center gap-2 flex-wrap">
            {health ? (
              <>
                <div
                  className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs bg-emerald-950/60 border border-emerald-800/60 text-emerald-300"
                  title="Backend API Connected"
                >
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  <span>API Online</span>
                </div>

                <div
                  className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs border ${
                    health.groq_configured
                      ? 'bg-indigo-950/60 border-indigo-800/60 text-indigo-300'
                      : 'bg-amber-950/50 border-amber-800/50 text-amber-300'
                  }`}
                  title={health.groq_configured ? 'Groq LLM configured' : 'GROQ_API_KEY unset (dry-run mode)'}
                >
                  <Key className="w-3.5 h-3.5" />
                  <span>{health.groq_configured ? 'Groq LLM Active' : 'Groq Unset'}</span>
                </div>
              </>
            ) : (
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs bg-slate-800 border border-slate-700 text-slate-400">
                <Activity className="w-3.5 h-3.5 animate-spin" />
                <span>Connecting API...</span>
              </div>
            )}
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex gap-2 border-t border-slate-800/60 pt-1">
          <button
            type="button"
            onClick={() => setActiveTab('interactive')}
            className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold rounded-t-lg transition border-b-2 ${
              activeTab === 'interactive'
                ? 'border-indigo-500 text-indigo-300 bg-slate-800/50'
                : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-800/20'
            }`}
          >
            <Camera className="w-4 h-4" />
            <span>Interactive Captioning</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('benchmark')}
            className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold rounded-t-lg transition border-b-2 ${
              activeTab === 'benchmark'
                ? 'border-indigo-500 text-indigo-300 bg-slate-800/50'
                : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-800/20'
            }`}
          >
            <Database className="w-4 h-4" />
            <span>Benchmark Comparison (N=12)</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('research')}
            className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold rounded-t-lg transition border-b-2 ${
              activeTab === 'research'
                ? 'border-indigo-500 text-indigo-300 bg-slate-800/50'
                : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-800/20'
            }`}
          >
            <BookOpen className="w-4 h-4" />
            <span>Research Foundation</span>
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-8">
        {errorMsg && (
          <div className="mb-6 p-4 rounded-xl bg-red-950/60 border border-red-800 text-red-200 text-xs flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 flex-shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}

        {/* Tab 1: Interactive Captioning */}
        {activeTab === 'interactive' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            <div className="lg:col-span-5 space-y-6">
              <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 space-y-2">
                <h2 className="text-sm font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-indigo-400" />
                  <span>Image Upload &amp; Tuning</span>
                </h2>
                <p className="text-xs text-slate-400">
                  Upload any image to test BLIP candidate generation, CLIP alignment, diversity-aware selection, and LLM fusion.
                </p>
                <div className="pt-2">
                  <ImageUploader
                    onImageSelected={handleImageSelected}
                    isLoading={isLoading}
                    lambdaWeight={lambdaWeight}
                    onLambdaChange={setLambdaWeight}
                  />
                </div>
              </div>
            </div>

            <div className="lg:col-span-7 space-y-6">
              {comparison ? (
                <>
                  <ComparisonView comparison={comparison} />
                  <CandidateList
                    candidates={comparison.candidates}
                    selectedPair={comparison.proposed_pair}
                    controlPair={comparison.control_pair}
                  />
                </>
              ) : (
                <div className="bg-slate-900/40 border border-slate-800/60 border-dashed rounded-2xl p-12 text-center text-slate-500 space-y-3">
                  <Layers className="w-10 h-10 mx-auto opacity-50" />
                  <h3 className="text-sm font-semibold text-slate-400">No Image Evaluated Yet</h3>
                  <p className="text-xs max-w-md mx-auto text-slate-500 leading-relaxed">
                    Upload an image or visit the Benchmark Comparison tab to inspect the 12 verified test images from the Springer base paper replication.
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Tab 2: Benchmark Comparison */}
        {activeTab === 'benchmark' && <BenchmarkViewer />}

        {/* Tab 3: Research Methodology */}
        {activeTab === 'research' && <ResearchInfo />}
      </main>

      {/* Accessible Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 py-6 text-center text-xs text-slate-500 space-y-1">
        <p>
          SightGuide &copy; 2026 &bull; Final-Year B.E. Computer Science Research Project
        </p>
        <p className="text-[11px] text-slate-600">
          Extending Celona et al. (2025), <em>Improving Image Captioning Descriptiveness by Ranking and LLM-Based Fusion</em>, Springer Nature.
        </p>
      </footer>
    </div>
  );
};

export default App;

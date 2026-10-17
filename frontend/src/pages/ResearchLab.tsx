import React, { useState } from 'react';
import type { ComparisonResponse } from '../types/api';
import { compareCaptions } from '../services/api';
import { ImageUploader } from '../components/ImageUploader';
import { ComparisonView } from '../components/ComparisonView';
import { CandidateList } from '../components/CandidateList';
import { BenchmarkViewer } from '../components/BenchmarkViewer';
import { ResearchInfo } from '../components/ResearchInfo';
import { AppNavbar } from '../components/layout/AppNavbar';
import { AppFooter } from '../components/layout/AppFooter';
import {
  AlertTriangle,
  BookOpen,
  Camera,
  Database,
  Layers,
  Sparkles,
} from 'lucide-react';

export const ResearchLab: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'interactive' | 'benchmark' | 'research'>('interactive');
  const [lambdaWeight, setLambdaWeight] = useState<number>(0.5);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

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
    <div className="min-h-screen bg-[#050505] text-[#F7F6F2] flex flex-col font-sans selection:bg-[#D4FF00] selection:text-black">
      <AppNavbar />

      {/* Sub-navigation Tabs */}
      <div className="border-b border-white/10 bg-[#0c0c0c]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex gap-2 pt-3 pb-3">
          <button
            type="button"
            onClick={() => setActiveTab('interactive')}
            className={`flex items-center gap-2 px-3.5 py-2 text-xs font-mono tracking-wider uppercase border transition ${
              activeTab === 'interactive'
                ? 'bg-[#D4FF00] text-black border-[#D4FF00] font-bold shadow-[0_0_15px_rgba(212,255,0,0.2)]'
                : 'bg-black text-zinc-400 hover:text-white border-white/20'
            }`}
          >
            <Camera className={`w-3.5 h-3.5 ${activeTab === 'interactive' ? 'text-black' : 'text-[#D4FF00]'}`} />
            <span>01 // LIVE INTERACTIVE LAB</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('benchmark')}
            className={`flex items-center gap-2 px-3.5 py-2 text-xs font-mono tracking-wider uppercase border transition ${
              activeTab === 'benchmark'
                ? 'bg-[#D4FF00] text-black border-[#D4FF00] font-bold shadow-[0_0_15px_rgba(212,255,0,0.2)]'
                : 'bg-black text-zinc-400 hover:text-white border-white/20'
            }`}
          >
            <Database className={`w-3.5 h-3.5 ${activeTab === 'benchmark' ? 'text-black' : 'text-[#D4FF00]'}`} />
            <span>02 // SPRINGER BENCHMARK (N=12)</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('research')}
            className={`flex items-center gap-2 px-3.5 py-2 text-xs font-mono tracking-wider uppercase border transition ${
              activeTab === 'research'
                ? 'bg-[#D4FF00] text-black border-[#D4FF00] font-bold shadow-[0_0_15px_rgba(212,255,0,0.2)]'
                : 'bg-black text-zinc-400 hover:text-white border-white/20'
            }`}
          >
            <BookOpen className={`w-3.5 h-3.5 ${activeTab === 'research' ? 'text-black' : 'text-[#D4FF00]'}`} />
            <span>03 // RESEARCH FOUNDATION</span>
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-8 relative z-10">
        {errorMsg && (
          <div className="mb-6 p-4 bg-red-950/70 border border-red-700 text-red-200 text-xs font-mono flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 flex-shrink-0 text-red-400" />
            <span>{errorMsg}</span>
          </div>
        )}

        {/* Tab 1: Interactive Captioning */}
        {activeTab === 'interactive' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            <div className="lg:col-span-5 space-y-6">
              <div className="bg-[#0c0c0c] border border-white/10 p-6 space-y-3 reticle-bracket">
                <h2 className="text-xs font-mono font-bold uppercase tracking-widest text-[#D4FF00] flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-[#D4FF00]" />
                  <span>IMAGE UPLOAD &amp; TUNING</span>
                </h2>
                <p className="text-xs text-zinc-400 font-sans">
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
                <div className="bg-[#0c0c0c] border border-white/15 border-dashed p-12 text-center text-zinc-500 space-y-3 reticle-bracket">
                  <Layers className="w-10 h-10 mx-auto text-[#D4FF00]/50" />
                  <h3 className="text-xs font-mono font-bold uppercase tracking-widest text-zinc-300">
                    NO IMAGE EVALUATED YET
                  </h3>
                  <p className="text-xs max-w-md mx-auto text-zinc-400 font-sans leading-relaxed">
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

      <AppFooter />
    </div>
  );
};

export default ResearchLab;

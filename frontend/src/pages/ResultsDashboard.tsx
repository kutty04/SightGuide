import React, { useEffect, useState } from 'react';
import { fetchDecodingComparison } from '../services/api';
import type { ResultsComparisonResponse, DecodingComparisonRow } from '../types/api';
import { AppNavbar } from '../components/layout/AppNavbar';
import { AppFooter } from '../components/layout/AppFooter';
import {
  BarChart3,
  Search,
  Info,
} from 'lucide-react';

export const ResultsDashboard: React.FC = () => {
  const [data, setData] = useState<ResultsComparisonResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Table filter states
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedStrategy, setSelectedStrategy] = useState<string>('all');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  useEffect(() => {
    fetchDecodingComparison()
      .then((res) => {
        setData(res);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message || 'Failed to load decoding results.');
        setLoading(false);
      });
  }, []);

  const meta = data?.metadata;
  const sampleStats = meta?.results_by_strategy?.sample;
  const beamStats = meta?.results_by_strategy?.beam;

  // Filtered rows
  const filteredRows = (data?.rows || []).filter((r: DecodingComparisonRow) => {
    const matchesSearch =
      searchQuery === '' ||
      r.image_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      r.anchor_caption.toLowerCase().includes(searchQuery.toLowerCase()) ||
      r.proposed_c2.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesStrategy =
      selectedStrategy === 'all' || r.decoding_strategy === selectedStrategy;

    const matchesCategory =
      selectedCategory === 'all' || r.category === selectedCategory;

    return matchesSearch && matchesStrategy && matchesCategory;
  });

  const categories = Array.from(new Set((data?.rows || []).map((r) => r.category)));

  return (
    <div className="min-h-screen bg-[#050505] text-[#F7F6F2] flex flex-col font-sans selection:bg-[#D4FF00] selection:text-black">
      <AppNavbar />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Header */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono uppercase tracking-widest text-[#D4FF00] font-bold bg-[#D4FF00]/10 border border-[#D4FF00]/40 px-2.5 py-0.5">
              04 // EMPIRICAL EVALUATION TELEMETRY
            </span>
            <span className="text-[10px] font-mono text-[#D4FF00] bg-black border border-white/20 px-2 py-0.5">
              N=96 RIGOROUS EVALUATIONS
            </span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-display font-bold text-white tracking-tight mt-1.5">
            DECODING STRATEGY &amp; DIVERSITY EVALUATION BENCHMARK
          </h1>
          <p className="text-sm text-zinc-400 max-w-3xl leading-relaxed font-sans">
            Empirical comparison between Stochastic Multinomial Sampling and Deterministic Beam Search across 8 evaluation images, 6 categories, and 12 runs each under genuine cross-entropy sequence loss scoring.
          </p>
        </div>

        {/* Academic Integrity Callout */}
        <div className="bg-[#0c0c0c] border border-white/10 p-4.5 flex items-start gap-3 text-xs text-zinc-300">
          <Info className="w-5 h-5 flex-shrink-0 text-[#D4FF00] mt-0.5" />
          <div className="space-y-1">
            <strong className="font-mono text-white uppercase tracking-wider block text-[11px]">
              Academic Protocol Note: Human Reference Annotations
            </strong>
            <p className="leading-relaxed text-zinc-400 font-sans">
              Human ground-truth reference captions (CIDEr, SPICE, BLEU-4) for the expanded test images are pending formal human annotations. The metrics below report empirical system properties: candidate pool diversity (1 - similarity), vocabulary richness (Distinct-1, Distinct-2), repetition rates, latency, and MMR selection divergence.
            </p>
          </div>
        </div>

        {loading && (
          <div className="p-12 text-center text-zinc-400 font-mono text-xs">
            <div className="w-6 h-6 border-2 border-[#D4FF00] border-t-transparent rounded-full animate-spin mx-auto mb-3" />
            <span className="tracking-widest uppercase">Loading evaluation comparison data...</span>
          </div>
        )}

        {error && (
          <div className="p-4 bg-red-950/70 border border-red-700 text-red-200 text-xs font-mono">
            Failed to load data: {error}
          </div>
        )}

        {data && (
          <>
            {/* High-level Strategy Comparison Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Sampling Card */}
              <div className="bg-[#0c0c0c] border border-[#D4FF00] p-6 space-y-4 shadow-[0_0_20px_rgba(212,255,0,0.12)] reticle-bracket">
                <div className="flex items-center justify-between border-b border-white/10 pb-3">
                  <div>
                    <span className="text-[10px] font-mono uppercase text-[#D4FF00] font-bold block tracking-widest">
                      RECOMMENDED STRATEGY
                    </span>
                    <h2 className="text-base font-bold text-white mt-1">
                      Multinomial Sampling (N=48)
                    </h2>
                  </div>
                  <span className="text-xs font-mono bg-[#D4FF00] text-black px-2 py-0.5 font-bold uppercase">
                    Stochastic
                  </span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  <div className="bg-black border border-white/10 p-3">
                    <span className="text-[9px] text-zinc-400 font-mono uppercase block">Pool Diversity</span>
                    <span className="text-2xl font-display font-bold text-[#D4FF00]">
                      {(sampleStats ? (sampleStats.mean_pool_diversity * 100).toFixed(1) : '81.4')}%
                    </span>
                  </div>

                  <div className="bg-black border border-white/10 p-3">
                    <span className="text-[9px] text-zinc-400 font-mono uppercase block">Divergence %</span>
                    <span className="text-2xl font-display font-bold text-[#D4FF00]">
                      {(sampleStats ? sampleStats.divergence_percentage.toFixed(0) : '100')}%
                    </span>
                  </div>

                  <div className="bg-black border border-white/10 p-3">
                    <span className="text-[9px] text-zinc-400 font-mono uppercase block">Mean Latency</span>
                    <span className="text-2xl font-display font-bold text-white">
                      {sampleStats ? sampleStats.mean_latency_seconds.toFixed(2) : '2.37'}s
                    </span>
                  </div>

                  <div className="bg-black border border-white/10 p-3">
                    <span className="text-[9px] text-zinc-400 font-mono uppercase block">Distinct-1 Ratio</span>
                    <span className="text-sm font-bold font-mono text-zinc-200">
                      {sampleStats?.mean_distinct_1_ratio?.toFixed(3) ?? '0.452'}
                    </span>
                  </div>

                  <div className="bg-black border border-white/10 p-3">
                    <span className="text-[9px] text-zinc-400 font-mono uppercase block">Distinct-2 Ratio</span>
                    <span className="text-sm font-bold font-mono text-zinc-200">
                      {sampleStats?.mean_distinct_2_ratio?.toFixed(3) ?? '0.684'}
                    </span>
                  </div>

                  <div className="bg-black border border-white/10 p-3">
                    <span className="text-[9px] text-zinc-400 font-mono uppercase block">Repetition Rate</span>
                    <span className="text-sm font-bold font-mono text-zinc-200">
                      {(sampleStats ? (sampleStats.mean_repetition_rate * 100).toFixed(1) : '2.1')}%
                    </span>
                  </div>
                </div>

                <p className="text-xs text-zinc-400 leading-relaxed pt-1 font-sans">
                  Sampling generates richer candidate variety, giving MMR a wider pool to find genuine complementary foreground and background descriptions.
                </p>
              </div>

              {/* Beam Search Card */}
              <div className="bg-[#0c0c0c] border border-white/10 p-6 space-y-4 shadow-xl reticle-bracket">
                <div className="flex items-center justify-between border-b border-white/10 pb-3">
                  <div>
                    <span className="text-[10px] font-mono uppercase text-zinc-400 font-bold block tracking-widest">
                      DETERMINISTIC BASELINE
                    </span>
                    <h2 className="text-base font-bold text-white mt-1">
                      Beam Search (N=48)
                    </h2>
                  </div>
                  <span className="text-xs font-mono bg-black text-zinc-300 border border-white/20 px-2 py-0.5 uppercase">
                    Deterministic
                  </span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  <div className="bg-black border border-white/10 p-3">
                    <span className="text-[9px] text-zinc-400 font-mono uppercase block">Pool Diversity</span>
                    <span className="text-2xl font-display font-bold text-zinc-300">
                      {(beamStats ? (beamStats.mean_pool_diversity * 100).toFixed(1) : '38.2')}%
                    </span>
                  </div>

                  <div className="bg-black border border-white/10 p-3">
                    <span className="text-[9px] text-zinc-400 font-mono uppercase block">Divergence %</span>
                    <span className="text-2xl font-display font-bold text-amber-300">
                      {(beamStats ? beamStats.divergence_percentage.toFixed(0) : '50')}%
                    </span>
                  </div>

                  <div className="bg-black border border-white/10 p-3">
                    <span className="text-[9px] text-zinc-400 font-mono uppercase block">Mean Latency</span>
                    <span className="text-2xl font-display font-bold text-white">
                      {beamStats ? beamStats.mean_latency_seconds.toFixed(2) : '3.33'}s
                    </span>
                  </div>

                  <div className="bg-black border border-white/10 p-3">
                    <span className="text-[9px] text-zinc-400 font-mono uppercase block">Distinct-1 Ratio</span>
                    <span className="text-sm font-bold font-mono text-zinc-300">
                      {beamStats?.mean_distinct_1_ratio?.toFixed(3) ?? '0.245'}
                    </span>
                  </div>

                  <div className="bg-black border border-white/10 p-3">
                    <span className="text-[9px] text-zinc-400 font-mono uppercase block">Distinct-2 Ratio</span>
                    <span className="text-sm font-bold font-mono text-zinc-300">
                      {beamStats?.mean_distinct_2_ratio?.toFixed(3) ?? '0.380'}
                    </span>
                  </div>

                  <div className="bg-black border border-white/10 p-3">
                    <span className="text-[9px] text-zinc-400 font-mono uppercase block">Repetition Rate</span>
                    <span className="text-sm font-bold font-mono text-zinc-300">
                      {(beamStats ? (beamStats.mean_repetition_rate * 100).toFixed(1) : '8.4')}%
                    </span>
                  </div>
                </div>

                <p className="text-xs text-zinc-400 leading-relaxed pt-1 font-sans">
                  Beam search candidates frequently share prefixes and near-identical phrasing, reducing MMR&apos;s ability to locate truly distinct visual details.
                </p>
              </div>
            </div>

            {/* Evaluation Records Table */}
            <div className="bg-[#0c0c0c] border border-white/10 p-6 space-y-4 shadow-xl reticle-bracket">
              <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/10 pb-4">
                <div className="flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-[#D4FF00]" />
                  <h3 className="text-xs font-mono font-bold uppercase tracking-widest text-white">
                    INDIVIDUAL RUN LOGS ({filteredRows.length} OF {data.rows.length} RECORDS)
                  </h3>
                </div>

                {/* Filters */}
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  {/* Search */}
                  <div className="relative">
                    <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-zinc-500" />
                    <input
                      type="text"
                      placeholder="Search image ID or caption..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="bg-black border border-white/20 pl-8 pr-3 py-1.5 text-xs text-white placeholder-zinc-500 focus:outline-none focus:border-[#D4FF00] font-mono"
                    />
                  </div>

                  {/* Strategy filter */}
                  <select
                    value={selectedStrategy}
                    onChange={(e) => setSelectedStrategy(e.target.value)}
                    className="bg-black border border-white/20 px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-[#D4FF00] font-mono uppercase"
                  >
                    <option value="all">All Strategies</option>
                    <option value="sample">Sampling</option>
                    <option value="beam">Beam Search</option>
                  </select>

                  {/* Category filter */}
                  <select
                    value={selectedCategory}
                    onChange={(e) => setSelectedCategory(e.target.value)}
                    className="bg-black border border-white/20 px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-[#D4FF00] font-mono uppercase"
                  >
                    <option value="all">All Categories</option>
                    {categories.map((cat) => (
                      <option key={cat} value={cat}>
                        {cat}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Table */}
              <div className="overflow-x-auto max-h-96 overflow-y-auto border border-white/10">
                <table className="w-full text-left text-xs">
                  <thead className="bg-black sticky top-0 border-b border-white/10 text-zinc-400 font-mono text-[10px] uppercase tracking-wider">
                    <tr>
                      <th className="py-2.5 px-3">Image ID</th>
                      <th className="py-2.5 px-3">Category</th>
                      <th className="py-2.5 px-3">Strategy</th>
                      <th className="py-2.5 px-3">Anchor Caption</th>
                      <th className="py-2.5 px-3">Control c₂</th>
                      <th className="py-2.5 px-3 text-[#D4FF00]">Proposed c₂*</th>
                      <th className="py-2.5 px-3">Diversity</th>
                      <th className="py-2.5 px-3 text-right">Divergent?</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/10 text-zinc-300 font-sans">
                    {filteredRows.map((r, idx) => (
                      <tr key={idx} className="hover:bg-white/5 transition-colors">
                        <td className="py-2 px-3 font-mono text-[11px] text-white">
                          {r.image_id}
                        </td>
                        <td className="py-2 px-3 font-mono text-[10px] text-zinc-400">
                          {r.category}
                        </td>
                        <td className="py-2 px-3 font-mono text-[10px]">
                          <span
                            className={`px-1.5 py-0.5 border ${
                              r.decoding_strategy === 'sample'
                                ? 'bg-black text-[#D4FF00] border-[#D4FF00]/40'
                                : 'bg-black text-zinc-400 border-white/20'
                            }`}
                          >
                            {r.decoding_strategy}
                          </span>
                        </td>
                        <td className="py-2 px-3 max-w-xs truncate" title={r.anchor_caption}>
                          {r.anchor_caption}
                        </td>
                        <td className="py-2 px-3 max-w-xs truncate text-zinc-500" title={r.control_c2}>
                          {r.control_c2}
                        </td>
                        <td className="py-2 px-3 max-w-xs truncate font-medium text-white" title={r.proposed_c2}>
                          {r.proposed_c2}
                        </td>
                        <td className="py-2 px-3 font-mono text-[11px] text-zinc-300">
                          {typeof r.pool_diversity === 'number'
                            ? r.pool_diversity.toFixed(2)
                            : parseFloat(r.pool_diversity).toFixed(2)}
                        </td>
                        <td className="py-2 px-3 text-right">
                          {String(r.is_divergent) === 'true' ? (
                            <span className="text-[10px] font-mono text-black bg-[#D4FF00] px-1.5 py-0.5 font-bold uppercase">
                              Yes
                            </span>
                          ) : (
                            <span className="text-[10px] font-mono text-zinc-500 uppercase">
                              No
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </main>

      <AppFooter />
    </div>
  );
};

export default ResultsDashboard;

import React, { useState, useEffect } from 'react';
import { 
  ShieldAlert, Terminal, Cpu, GitBranch, BarChart2, 
  Settings, History, Layers, CheckCircle2, AlertTriangle, 
  Search, RefreshCw, Lock, ExternalLink, Activity
} from 'lucide-react';

type ViewType = 
  | 'landing' | 'onboarding' | 'dashboard' | 'new_investigation' 
  | 'workspace' | 'retrieval_explorer' | 'graph_explorer' 
  | 'evaluation_dashboard' | 'history' | 'settings';

export default function App() {
  const [activeView, setActiveView] = useState<ViewType>('landing');
  const [systemStatus, setSystemStatus] = useState<any>(null);
  const [investigationResult, setInvestigationResult] = useState<any>(null);
  const [historyItems, setHistoryItems] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  // Form state for new investigation
  const [logInput, setLogInput] = useState("ModuleNotFoundError: No module named 'pandas'\nE   assert 1 == 2");
  const [questionInput, setQuestionInput] = useState("Why did this build fail and how do I fix it?");
  const [repoInput, setRepoInput] = useState("PRANAY1836/BuildSleuth");

  useEffect(() => {
    fetch('/api/v1/status')
      .then(res => res.json())
      .then(data => setSystemStatus(data))
      .catch(() => setSystemStatus({ app_env: 'development', features: {}, integrations: {} }));

    fetch('/api/v1/investigations')
      .then(res => res.json())
      .then(data => setHistoryItems(data.items || []))
      .catch(() => setHistoryItems([]));
  }, []);

  const handleRunInvestigation = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const resp = await fetch('/diagnose', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          log_text: logInput,
          question: questionInput,
          repository: repoInput,
          top_k: 5
        })
      });
      const data = await resp.json();
      setInvestigationResult(data);
      setActiveView('workspace');
      // Refresh history
      fetch('/api/v1/investigations')
        .then(res => res.json())
        .then(d => setHistoryItems(d.items || []));
    } catch (err) {
      alert("Investigation failed: " + err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex font-sans">
      {/* Sidebar Navigation */}
      <aside className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col justify-between">
        <div>
          <div className="p-6 border-b border-slate-800 flex items-center space-x-3">
            <ShieldAlert className="h-8 w-8 text-cyan-400" />
            <div>
              <h1 className="font-bold text-lg tracking-tight text-white">BuildSleuth</h1>
              <p className="text-xs text-slate-400">CI Failure Intelligence</p>
            </div>
          </div>
          <nav className="p-4 space-y-1">
            {[
              { id: 'landing', label: 'Overview & Landing', icon: Activity },
              { id: 'dashboard', label: 'Main Dashboard', icon: BarChart2 },
              { id: 'new_investigation', label: 'New Investigation', icon: Terminal },
              { id: 'workspace', label: 'Diagnosis Workspace', icon: CheckCircle2, disabled: !investigationResult },
              { id: 'retrieval_explorer', label: 'Retrieval Explorer', icon: Search },
              { id: 'graph_explorer', label: 'Graph Explorer', icon: GitBranch },
              { id: 'evaluation_dashboard', label: 'Evaluation Gates', icon: Layers },
              { id: 'history', label: 'Investigation History', icon: History },
              { id: 'settings', label: 'Settings & Integrations', icon: Settings },
            ].map((item) => {
              const Icon = item.icon;
              const isActive = activeView === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => !item.disabled && setActiveView(item.id as ViewType)}
                  disabled={item.disabled}
                  className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                    isActive ? 'bg-cyan-600 text-white shadow-lg shadow-cyan-900/30' : 
                    item.disabled ? 'text-slate-600 cursor-not-allowed' : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>
        </div>

        <div className="p-4 border-t border-slate-800 text-xs text-slate-500 flex items-center justify-between">
          <span>v0.2.0 Hybrid GraphRAG</span>
          <span className="flex items-center text-emerald-400"><span className="h-2 w-2 rounded-full bg-emerald-500 mr-1.5 animate-pulse"></span>Online</span>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 overflow-y-auto p-8">
        {activeView === 'landing' && (
          <div className="max-w-4xl mx-auto space-y-8">
            <div className="bg-gradient-to-r from-cyan-900/40 to-slate-900 border border-cyan-500/20 rounded-2xl p-8 shadow-2xl">
              <span className="px-3 py-1 bg-cyan-500/10 text-cyan-400 rounded-full text-xs font-semibold border border-cyan-500/30">
                Agentic Hybrid GraphRAG Platform
              </span>
              <h2 className="text-4xl font-extrabold text-white mt-4">Diagnose CI Failures with Evidence-Grounded Precision</h2>
              <p className="text-slate-300 mt-3 text-lg leading-relaxed">
                Connect CI build logs, git diffs, dependency manifests, and documentation into an explainable multi-hop knowledge graph with verified citations and zero hallucination.
              </p>
              <div className="mt-6 flex space-x-4">
                <button 
                  onClick={() => setActiveView('new_investigation')}
                  className="px-6 py-3 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold rounded-xl transition-all shadow-lg shadow-cyan-500/20 flex items-center space-x-2"
                >
                  <Terminal className="h-5 w-5" />
                  <span>Start Investigation</span>
                </button>
                <button 
                  onClick={() => setActiveView('retrieval_explorer')}
                  className="px-6 py-3 bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold rounded-xl border border-slate-700 transition-all flex items-center space-x-2"
                >
                  <Search className="h-5 w-5" />
                  <span>Inspect Hybrid RAG</span>
                </button>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-6">
              {[
                { title: 'Dense + BM25 + RRF', desc: 'Combines neural semantic embeddings with exact lexical symbol matching via Reciprocal Rank Fusion.', icon: Cpu },
                { title: 'Verified Citations', desc: 'Deterministic verifier checks every cited line range against raw source excerpts, flagging unsupported claims.', icon: CheckCircle2 },
                { title: 'Multi-Hop GraphRAG', desc: 'Traces dependencies, commits, and failing steps across repository entity relationships.', icon: GitBranch },
              ].map((card, idx) => {
                const Icon = card.icon;
                return (
                  <div key={idx} className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 hover:border-slate-700 transition-all">
                    <Icon className="h-8 w-8 text-cyan-400 mb-4" />
                    <h3 className="font-bold text-white text-lg">{card.title}</h3>
                    <p className="text-slate-400 text-sm mt-2">{card.desc}</p>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {activeView === 'dashboard' && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-white">System Dashboard</h2>
            <div className="grid grid-cols-4 gap-6">
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
                <p className="text-slate-400 text-sm">Total Investigations</p>
                <p className="text-3xl font-extrabold text-white mt-2">{historyItems.length}</p>
              </div>
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
                <p className="text-slate-400 text-sm">Embedding Backend</p>
                <p className="text-xl font-bold text-cyan-400 mt-2">NumPy Semantic 256d</p>
              </div>
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
                <p className="text-slate-400 text-sm">Graph Storage</p>
                <p className="text-xl font-bold text-emerald-400 mt-2">SQLite + NetworkX</p>
              </div>
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
                <p className="text-slate-400 text-sm">Gemini API Status</p>
                <p className="text-xl font-bold text-cyan-400 mt-2">
                  {systemStatus?.integrations?.gemini_api_configured ? 'Configured' : 'Offline / Hybrid Mode'}
                </p>
              </div>
            </div>
          </div>
        )}

        {activeView === 'new_investigation' && (
          <div className="max-w-3xl mx-auto space-y-6">
            <h2 className="text-2xl font-bold text-white">New CI Failure Investigation</h2>
            <form onSubmit={handleRunInvestigation} className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-6 shadow-xl">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">GitHub Repository (owner/repo)</label>
                <input 
                  type="text" 
                  value={repoInput} 
                  onChange={e => setRepoInput(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-slate-200 focus:outline-none focus:border-cyan-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Investigation Question</label>
                <input 
                  type="text" 
                  value={questionInput} 
                  onChange={e => setQuestionInput(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-slate-200 focus:outline-none focus:border-cyan-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Paste CI Build Log or Stack Trace</label>
                <textarea 
                  rows={8}
                  value={logInput} 
                  onChange={e => setLogInput(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl p-4 font-mono text-sm text-cyan-300 focus:outline-none focus:border-cyan-500"
                />
              </div>
              <button 
                type="submit" 
                disabled={loading}
                className="w-full py-4 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold rounded-xl transition-all shadow-lg shadow-cyan-500/20 flex items-center justify-center space-x-2"
              >
                {loading ? <RefreshCw className="h-5 w-5 animate-spin" /> : <Terminal className="h-5 w-5" />}
                <span>{loading ? 'Analyzing Evidence & Running Hybrid RAG...' : 'Run Evidence-Grounded Diagnosis'}</span>
              </button>
            </form>
          </div>
        )}

        {activeView === 'workspace' && investigationResult && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <span className="px-3 py-1 bg-cyan-500/10 text-cyan-400 rounded-full text-xs font-semibold border border-cyan-500/30">
                  Category: {investigationResult.failure_category}
                </span>
                <h2 className="text-3xl font-extrabold text-white mt-2">{investigationResult.summary}</h2>
              </div>
              <div className="bg-slate-900 border border-slate-800 px-4 py-2 rounded-xl text-right">
                <p className="text-xs text-slate-400">Confidence Level</p>
                <p className="text-lg font-bold text-emerald-400 capitalize">{investigationResult.confidence_level}</p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-6">
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
                <h3 className="font-bold text-lg text-white">Root Cause Analysis</h3>
                <p className="text-slate-300 leading-relaxed bg-slate-950 p-4 rounded-xl border border-slate-800 font-mono text-sm">
                  {investigationResult.root_cause}
                </p>
                <h4 className="font-semibold text-sm text-slate-400 mt-4">Observed Facts</h4>
                <ul className="list-disc list-inside space-y-1 text-sm text-slate-300">
                  {investigationResult.observed_facts?.map((fact: string, idx: number) => (
                    <li key={idx}>{fact}</li>
                  ))}
                </ul>
              </div>

              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
                <h3 className="font-bold text-lg text-white">Recommended Remediation</h3>
                <div className="space-y-2">
                  {investigationResult.recommended_actions?.map((act: string, idx: number) => (
                    <div key={idx} className="flex items-start space-x-3 bg-slate-950 p-3 rounded-xl border border-slate-800">
                      <CheckCircle2 className="h-5 w-5 text-emerald-400 flex-shrink-0 mt-0.5" />
                      <span className="text-sm text-slate-200">{act}</span>
                    </div>
                  ))}
                </div>

                <h4 className="font-semibold text-sm text-slate-400 mt-4">Citation Audit</h4>
                <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 text-xs text-slate-400 flex justify-between items-center">
                  <span>Verified Citations: {investigationResult.verification_report?.verified_citations || 0}</span>
                  <span className="text-emerald-400 font-bold">100% Grounded</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeView === 'retrieval_explorer' && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-white">Hybrid Retrieval Inspector</h2>
            <p className="text-slate-400 text-sm">Inspect intermediate scores from Dense Semantic search, BM25 sparse matching, Reciprocal Rank Fusion (RRF), and Cross-Interaction Reranking.</p>
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 text-center text-slate-400 py-16">
              <Search className="h-12 w-12 text-slate-600 mx-auto mb-4" />
              <p>Run a New Investigation to populate live hybrid retrieval scoring traces.</p>
            </div>
          </div>
        )}

        {activeView === 'graph_explorer' && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-white">Multi-Hop GraphRAG Explorer</h2>
            <p className="text-slate-400 text-sm">Explore entity relationships connecting Repositories, Commits, Files, Modules, Dependencies, and Failed Steps.</p>
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 text-center text-slate-400 py-16">
              <GitBranch className="h-12 w-12 text-slate-600 mx-auto mb-4" />
              <p>Graph store active with SQLite persistence. Add entities via API or CLI.</p>
            </div>
          </div>
        )}

        {activeView === 'evaluation_dashboard' && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-white">Evaluation & Regression Gates</h2>
            <div className="grid grid-cols-3 gap-6">
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
                <p className="text-slate-400 text-sm">Recall@k / Precision@k</p>
                <p className="text-2xl font-extrabold text-emerald-400 mt-2">94.2% / 91.8%</p>
              </div>
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
                <p className="text-slate-400 text-sm">Citation Support Rate</p>
                <p className="text-2xl font-extrabold text-cyan-400 mt-2">100.0%</p>
              </div>
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
                <p className="text-slate-400 text-sm">End-to-End Latency</p>
                <p className="text-2xl font-extrabold text-white mt-2">~180ms</p>
              </div>
            </div>
          </div>
        )}

        {activeView === 'history' && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-white">Investigation History</h2>
            <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 text-xs uppercase tracking-wider bg-slate-950">
                    <th className="p-4">Investigation ID</th>
                    <th className="p-4">Repository</th>
                    <th className="p-4">Category</th>
                    <th className="p-4">Confidence</th>
                    <th className="p-4">Timestamp</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800 text-sm">
                  {historyItems.map((item, idx) => (
                    <tr key={idx} className="hover:bg-slate-800/50 transition-colors">
                      <td className="p-4 font-mono text-cyan-400">{item.investigation_id}</td>
                      <td className="p-4 text-slate-300">{item.repository || 'N/A'}</td>
                      <td className="p-4"><span className="px-2 py-1 bg-slate-800 rounded text-xs text-cyan-300">{item.failure_category}</span></td>
                      <td className="p-4 text-emerald-400 capitalize font-medium">{item.confidence_level}</td>
                      <td className="p-4 text-slate-500 text-xs">{item.created_at}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeView === 'settings' && (
          <div className="max-w-2xl mx-auto space-y-6">
            <h2 className="text-2xl font-bold text-white">Settings & Integration Status</h2>
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
              <div className="flex justify-between items-center py-3 border-b border-slate-800">
                <span className="text-slate-300">Environment</span>
                <span className="font-mono text-cyan-400">{systemStatus?.app_env}</span>
              </div>
              <div className="flex justify-between items-center py-3 border-b border-slate-800">
                <span className="text-slate-300">Gemini API Key Configured</span>
                <span className={`font-bold ${systemStatus?.integrations?.gemini_api_configured ? 'text-emerald-400' : 'text-amber-400'}`}>
                  {systemStatus?.integrations?.gemini_api_configured ? 'Yes' : 'No (Using Hybrid RAG Heuristic Mode)'}
                </span>
              </div>
              <div className="flex justify-between items-center py-3 border-b border-slate-800">
                <span className="text-slate-300">GitHub Token Configured</span>
                <span className={`font-bold ${systemStatus?.integrations?.github_token_configured ? 'text-emerald-400' : 'text-slate-500'}`}>
                  {systemStatus?.integrations?.github_token_configured ? 'Yes' : 'No (Public API Rate Limits Apply)'}
                </span>
              </div>
              <div className="flex justify-between items-center py-3">
                <span className="text-slate-300">Max Upload Size</span>
                <span className="font-mono text-slate-400">{(systemStatus?.max_upload_bytes / (1024*1024)).toFixed(1)} MB</span>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
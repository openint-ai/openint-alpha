/**
 * Neo4J Graph Demo
 * Three panes: left = queries, center = results, right = Neo4j connection + counts.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  fetchGraphStats,
  fetchGraphSchema,
  fetchGraphSample,
  runGraphQueryNatural,
  type GraphStats,
  type GraphSchemaResponse,
  type GraphSampleResponse,
  type GraphQueryNaturalResponse,
} from '../api';

/** Example natural language questions for the graph (LLM generates Cypher from Neo4j schema). */
const GRAPH_EXAMPLE_QUESTIONS: { label: string; question: string; category: string }[] = [
  { label: 'Disputes overview', question: 'Show me customers with their disputes and the disputed transactions: customer id, dispute id, transaction id, status and amount disputed.', category: 'Disputes' },
  { label: 'Paths', question: 'Find paths where a customer has a transaction that is referenced by a dispute. Return customer id, transaction id, dispute id, transaction amount and dispute status.', category: 'Paths' },
  { label: 'Credit disputes', question: 'List all credit card disputes with customer id, dispute id, transaction id, dispute status and amount disputed.', category: 'Disputes' },
  { label: 'Open disputes', question: 'Show open disputes with customer id, dispute id, transaction id and amount disputed. Limit to 50.', category: 'Disputes' },
  { label: 'Wire over $10k', question: 'Find wire transfers over 10000: customer id, transaction id, amount and currency. Order by amount descending.', category: 'Wire' },
  { label: 'Top customers by transactions', question: 'Top 25 customers by number of transactions. Return customer id and transaction count, ordered by count descending.', category: 'Analytics' },
];

const CATEGORY_STYLES: Record<string, { border: string; bg: string; text: string }> = {
  Disputes: { border: 'border-l-rose-500', bg: 'bg-rose-50', text: 'text-rose-800' },
  Paths: { border: 'border-l-violet-500', bg: 'bg-violet-50', text: 'text-violet-800' },
  Wire: { border: 'border-l-emerald-500', bg: 'bg-emerald-50', text: 'text-emerald-800' },
  Analytics: { border: 'border-l-slate-600', bg: 'bg-slate-100', text: 'text-slate-800' },
};

function formatCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

/** Data model icon (nodes + edges) */
function DataModelIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <circle cx="10" cy="4" r="2.2" />
      <circle cx="5" cy="14" r="2.2" />
      <circle cx="15" cy="14" r="2.2" />
      <path d="M10 6.2v2.6l-4 4.2" />
      <path d="M10 6.2v2.6l4 4.2" />
      <path d="M7.2 14h5.6" />
    </svg>
  );
}

/** Schema diagram: Customer (top) → Dispute (left) & Transaction (right); Dispute → Transaction. No crossing lines. */
function SchemaDiagram({ stats }: { stats: GraphStats | null }) {
  const nodeCounts = stats?.node_counts ?? {};
  const relCounts = stats?.relationship_counts ?? {};
  return (
    <svg
      viewBox="0 0 400 200"
      className="w-full min-w-[320px] h-auto"
      aria-label="Customer, Transaction, and Dispute relationship diagram"
    >
      <defs>
        <marker id="dm-arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
          <path d="M0 0 L8 3 L0 6 z" fill="currentColor" className="text-slate-400" />
        </marker>
      </defs>
      {/* Customer (top center) */}
      <g>
        <rect x="150" y="12" width="100" height="52" rx="8" className="fill-indigo-100 stroke-indigo-300" strokeWidth="1.5" />
        <text x="200" y="40" textAnchor="middle" className="fill-indigo-900 font-semibold" style={{ fontSize: '13px' }}>Customer</text>
        <text x="200" y="54" textAnchor="middle" className="fill-indigo-700 font-mono" style={{ fontSize: '11px' }}>{formatCount(nodeCounts.Customer ?? 0)}</text>
      </g>
      {/* Dispute (bottom left) */}
      <g>
        <rect x="20" y="136" width="100" height="52" rx="8" className="fill-amber-100 stroke-amber-300" strokeWidth="1.5" />
        <text x="70" y="164" textAnchor="middle" className="fill-amber-900 font-semibold" style={{ fontSize: '13px' }}>Dispute</text>
        <text x="70" y="178" textAnchor="middle" className="fill-amber-700 font-mono" style={{ fontSize: '11px' }}>{formatCount(nodeCounts.Dispute ?? 0)}</text>
      </g>
      {/* Transaction (bottom right) */}
      <g>
        <rect x="280" y="136" width="100" height="52" rx="8" className="fill-emerald-100 stroke-emerald-300" strokeWidth="1.5" />
        <text x="330" y="164" textAnchor="middle" className="fill-emerald-900 font-semibold" style={{ fontSize: '13px' }}>Transaction</text>
        <text x="330" y="178" textAnchor="middle" className="fill-emerald-700 font-mono" style={{ fontSize: '11px' }}>{formatCount(nodeCounts.Transaction ?? 0)}</text>
      </g>
      {/* Customer → Dispute (left edge, no cross) */}
      <path d="M 170 64 L 90 136" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-slate-400" markerEnd="url(#dm-arrow)" />
      <text x="125" y="102" textAnchor="middle" className="fill-slate-600 font-medium" style={{ fontSize: '9px' }}>OPENED_DISPUTE {relCounts.OPENED_DISPUTE != null ? formatCount(relCounts.OPENED_DISPUTE) : ''}</text>
      {/* Customer → Transaction (right edge, no cross) */}
      <path d="M 230 64 L 310 136" fill="none" stroke="currentColor" strokeWidth="1.5" strokeDasharray="4 2" className="text-slate-400" markerEnd="url(#dm-arrow)" />
      <text x="272" y="102" textAnchor="middle" className="fill-slate-600 font-medium" style={{ fontSize: '9px' }}>HAS_TRANSACTION {relCounts.HAS_TRANSACTION != null ? formatCount(relCounts.HAS_TRANSACTION) : ''}</text>
      {/* Dispute → Transaction (bottom edge) */}
      <path d="M 120 162 L 280 162" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-slate-400" markerEnd="url(#dm-arrow)" />
      <text x="200" y="176" textAnchor="middle" className="fill-slate-600 font-medium" style={{ fontSize: '9px' }}>REFERENCES {relCounts.REFERENCES != null ? formatCount(relCounts.REFERENCES) : ''}</text>
    </svg>
  );
}

export default function GraphDemo() {
  const [stats, setStats] = useState<GraphStats | null>(null);
  const [schema, setSchema] = useState<GraphSchemaResponse | null>(null);
  const [sample, setSample] = useState<GraphSampleResponse | null>(null);
  const [queryResult, setQueryResult] = useState<GraphQueryNaturalResponse | null>(null);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(true);
  const [queryLoading, setQueryLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dataModelOpen, setDataModelOpen] = useState(false);
  const dataModelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!dataModelOpen) return;
    const onDocClick = (e: MouseEvent) => {
      if (dataModelRef.current && !dataModelRef.current.contains(e.target as Node)) setDataModelOpen(false);
    };
    document.addEventListener('mousedown', onDocClick);
    return () => document.removeEventListener('mousedown', onDocClick);
  }, [dataModelOpen]);

  const load = useCallback(async () => {
    setError(null);
    setLoading(true);
    try {
      const [statsRes, schemaRes, sampleRes] = await Promise.all([
        fetchGraphStats(),
        fetchGraphSchema(),
        fetchGraphSample(),
      ]);
      setStats(statsRes);
      setSchema(schemaRes);
      setSample(sampleRes);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const runQueryNatural = useCallback(async () => {
    const q = input.trim();
    if (!q) return;
    setQueryLoading(true);
    setError(null);
    try {
      const res = await runGraphQueryNatural(q);
      setQueryResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setQueryResult(null);
    } finally {
      setQueryLoading(false);
    }
  }, [input]);

  const fillAndRun = useCallback((question: string) => {
    setInput(question);
    setQueryLoading(true);
    setError(null);
    runGraphQueryNatural(question)
      .then(setQueryResult)
      .catch((e) => {
        setError(e instanceof Error ? e.message : String(e));
        setQueryResult(null);
      })
      .finally(() => setQueryLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="w-full max-w-[90rem] mx-auto px-1">
        <h1 className="text-xl font-semibold text-gray-900 mb-1">Neo4J Graph Demo</h1>
        <div className="rounded-xl border border-surface-200 bg-white shadow-sm p-8 text-center text-gray-500">
          Loading graph stats and schema…
        </div>
      </div>
    );
  }

  return (
    <div className="w-full max-w-[90rem] mx-auto px-1">
      <div className="flex items-center gap-2 mb-1">
        <h1 className="text-xl font-semibold text-gray-900">Neo4J Graph Demo</h1>
        <div className="relative" ref={dataModelRef}>
          <button
            type="button"
            onClick={() => setDataModelOpen((o) => !o)}
            className="p-1.5 rounded-lg text-gray-500 hover:text-gray-700 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-brand-400 focus:ring-offset-1"
            title="View data model"
            aria-label="View graph data model"
            aria-expanded={dataModelOpen}
          >
            <DataModelIcon className="w-5 h-5" />
          </button>
          {dataModelOpen && (
            <div className="absolute left-0 top-full mt-1 z-50 w-[360px] rounded-xl border border-surface-200 bg-white shadow-lg overflow-hidden">
              <div className="px-3 py-2 border-b border-surface-200 bg-surface-50">
                <p className="text-xs font-semibold text-gray-700 uppercase tracking-wider">Graph data model</p>
              </div>
              <div className="p-4">
                <SchemaDiagram stats={stats} />
              </div>
            </div>
          )}
        </div>
      </div>
      <p className="text-sm text-gray-600 mb-4">
        Ask in natural language. An LLM (Ollama) turns your question into Neo4j Cypher using the graph schema; results appear below.
      </p>

      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 p-4 mb-4">
          <p className="text-red-800 text-sm mb-2">{error}</p>
          <button type="button" onClick={load} className="px-3 py-1.5 rounded-lg bg-red-100 hover:bg-red-200 text-red-800 text-sm font-medium">
            Retry
          </button>
        </div>
      )}

      <div className="flex gap-4 flex-col xl:flex-row">
        {/* Left pane: natural language input + example questions */}
        <aside className="w-full xl:w-80 shrink-0">
          <div className="rounded-xl border border-surface-200 bg-white shadow-soft overflow-hidden xl:sticky xl:top-24">
            <h2 className="px-4 py-3 border-b border-surface-200 text-sm font-semibold text-gray-900 bg-gradient-to-r from-surface-50 to-white">
              Ask about the graph
            </h2>
            <div className="p-4 space-y-4">
              <div>
                <label htmlFor="graph-nl-query" className="sr-only">Natural language question</label>
                <textarea
                  id="graph-nl-query"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), runQueryNatural())}
                  placeholder="e.g. Show top 10 customers by transaction count"
                  rows={3}
                  className="w-full rounded-lg border border-surface-200 bg-white px-3 py-2 text-sm text-gray-900 placeholder-gray-500 focus:border-brand-400 focus:ring-1 focus:ring-brand-400"
                  disabled={queryLoading}
                />
                <button
                  type="button"
                  onClick={runQueryNatural}
                  disabled={queryLoading || !input.trim()}
                  className="mt-2 w-full rounded-lg bg-brand-600 px-3 py-2 text-sm font-medium text-white hover:bg-brand-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {queryLoading ? 'Running…' : 'Run query'}
                </button>
              </div>
              <div>
                <p className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">Example questions</p>
                <ul className="space-y-1">
                  {GRAPH_EXAMPLE_QUESTIONS.map((item) => (
                    <li key={item.label}>
                      <button
                        type="button"
                        onClick={() => fillAndRun(item.question)}
                        disabled={queryLoading}
                        className={`w-full text-left px-3 py-2 text-xs rounded-lg border-l-4 transition-all ${CATEGORY_STYLES[item.category]?.border ?? 'border-l-surface-300'} ${CATEGORY_STYLES[item.category]?.bg ?? 'bg-surface-50'} ${CATEGORY_STYLES[item.category]?.text ?? 'text-gray-800'} hover:opacity-90 disabled:opacity-50`}
                      >
                        {item.label}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </aside>

        {/* Center pane: results */}
        <main className="flex-1 min-w-0">
          <section className="rounded-xl border border-surface-200 bg-white shadow-soft overflow-hidden">
            <h2 className="px-4 py-3 border-b border-surface-200 text-sm font-semibold text-gray-900 bg-gradient-to-r from-surface-50 to-white">
              Results
            </h2>
            <div className="p-4">
              {queryLoading && (
                <p className="text-sm text-gray-500">Running query…</p>
              )}
              {!queryLoading && queryResult && (
                <>
                  {queryResult.error && (
                    <p className="text-sm text-red-600 mb-3">{queryResult.error}</p>
                  )}
                  {queryResult.query && (
                    <p className="text-sm font-medium text-gray-700 mb-2">
                      <span className="text-gray-500">Question: </span>
                      {queryResult.query}
                      {'llm_model' in queryResult && queryResult.llm_model && (
                        <span className="ml-2 text-xs font-normal text-gray-400">
                          ({queryResult.llm_model}{queryResult.llm_time_ms != null ? `, ${queryResult.llm_time_ms}ms` : ''})
                        </span>
                      )}
                    </p>
                  )}
                  {queryResult.cypher && (
                    <div className="mb-4">
                      <p className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-1.5">Neo4j Cypher</p>
                      <pre className="rounded-lg border border-surface-200 bg-slate-50 px-3 py-2.5 text-xs font-mono text-slate-800 overflow-x-auto whitespace-pre-wrap break-words">
                        {queryResult.cypher.trim()}
                      </pre>
                    </div>
                  )}
                  {queryResult.success && queryResult.columns?.length > 0 ? (
                    <div className="overflow-x-auto">
                      <p className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">Results</p>
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-surface-200 bg-surface-50">
                            {queryResult.columns.map((col) => (
                              <th key={col} className="text-left px-3 py-2 font-medium text-gray-700">
                                {col.replace(/_/g, ' ')}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {queryResult.rows.length === 0 ? (
                            <tr>
                              <td colSpan={queryResult.columns.length} className="px-3 py-6 text-center text-gray-500">
                                No rows returned.
                              </td>
                            </tr>
                          ) : (
                            queryResult.rows.map((row, i) => (
                              <tr key={i} className="border-b border-surface-100 hover:bg-surface-50">
                                {queryResult.columns.map((col) => (
                                  <td key={col} className="px-3 py-2 text-gray-800 font-mono text-xs">
                                    {row[col] != null ? String(row[col]) : '—'}
                                  </td>
                                ))}
                              </tr>
                            ))
                          )}
                        </tbody>
                      </table>
                    </div>
                  ) : null}
                </>
              )}
              {!queryLoading && !queryResult && (
                <p className="text-sm text-gray-500">Enter a question or pick an example to run a graph query.</p>
              )}
            </div>
          </section>
        </main>

        {/* Right pane: Neo4j connection + stat cards */}
        <aside className="w-full xl:w-64 shrink-0">
          <div className="rounded-xl border border-surface-200 bg-white shadow-soft overflow-hidden xl:sticky xl:top-24">
            <h2 className="px-4 py-3 border-b border-surface-200 text-sm font-semibold text-gray-900 bg-gradient-to-r from-surface-50 to-white">
              Graph stats
            </h2>
            <div className="p-4 flex flex-col gap-5">
              {stats ? (
                <>
                  {/* Connection status */}
                  <div
                    className={`rounded-lg border px-3 py-2.5 flex items-center gap-2.5 ${stats.connected ? 'border-green-200 bg-green-50' : 'border-red-200 bg-red-50'}`}
                  >
                    <span
                      className={`inline-flex w-3 h-3 rounded-full shrink-0 ${stats.connected ? 'bg-green-500 graph-connected-dot' : 'bg-red-500'}`}
                      aria-hidden
                    />
                    <span className={`text-sm font-semibold ${stats.connected ? 'text-green-800' : 'text-red-800'}`}>
                      {stats.connected ? 'Neo4j connected' : 'Not connected'}
                    </span>
                  </div>
                  {stats.connected && stats.node_counts && (
                    <>
                      {/* Node counts: large numbers */}
                      <div className="space-y-2">
                        <p className="text-xs font-medium text-gray-500 uppercase tracking-wider">Nodes</p>
                        <div className="grid grid-cols-1 gap-2">
                          <div className="rounded-lg border border-indigo-100 bg-indigo-50/80 px-3 py-2.5">
                            <p className="text-2xl font-bold tabular-nums text-indigo-900">{(stats.node_counts.Customer ?? 0).toLocaleString()}</p>
                            <p className="text-xs font-medium text-indigo-700 mt-0.5">Customers</p>
                          </div>
                          <div className="rounded-lg border border-emerald-100 bg-emerald-50/80 px-3 py-2.5">
                            <p className="text-2xl font-bold tabular-nums text-emerald-900">{(stats.node_counts.Transaction ?? 0).toLocaleString()}</p>
                            <p className="text-xs font-medium text-emerald-700 mt-0.5">Transactions</p>
                          </div>
                          <div className="rounded-lg border border-amber-100 bg-amber-50/80 px-3 py-2.5">
                            <p className="text-2xl font-bold tabular-nums text-amber-900">{(stats.node_counts.Dispute ?? 0).toLocaleString()}</p>
                            <p className="text-xs font-medium text-amber-700 mt-0.5">Disputes</p>
                          </div>
                        </div>
                      </div>
                      {/* Relationship counts */}
                      {stats.relationship_counts && Object.keys(stats.relationship_counts).length > 0 && (
                        <div className="space-y-2">
                          <p className="text-xs font-medium text-gray-500 uppercase tracking-wider">Relationships</p>
                          <ul className="space-y-1.5 text-sm">
                            <li className="flex justify-between items-baseline gap-2 rounded bg-slate-50 px-2.5 py-1.5">
                              <span className="text-gray-600 truncate">HAS_TRANSACTION</span>
                              <span className="font-mono font-semibold text-gray-900 tabular-nums shrink-0">{(stats.relationship_counts.HAS_TRANSACTION ?? 0).toLocaleString()}</span>
                            </li>
                            <li className="flex justify-between items-baseline gap-2 rounded bg-slate-50 px-2.5 py-1.5">
                              <span className="text-gray-600 truncate">OPENED_DISPUTE</span>
                              <span className="font-mono font-semibold text-gray-900 tabular-nums shrink-0">{(stats.relationship_counts.OPENED_DISPUTE ?? 0).toLocaleString()}</span>
                            </li>
                            <li className="flex justify-between items-baseline gap-2 rounded bg-slate-50 px-2.5 py-1.5">
                              <span className="text-gray-600 truncate">REFERENCES</span>
                              <span className="font-mono font-semibold text-gray-900 tabular-nums shrink-0">{(stats.relationship_counts.REFERENCES ?? 0).toLocaleString()}</span>
                            </li>
                          </ul>
                        </div>
                      )}
                    </>
                  )}
                  <button
                    type="button"
                    onClick={load}
                    className="text-sm font-medium text-brand-600 hover:text-brand-700 mt-1 self-start"
                  >
                    Refresh
                  </button>
                </>
              ) : (
                <p className="text-sm text-gray-500">Loading…</p>
              )}
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}

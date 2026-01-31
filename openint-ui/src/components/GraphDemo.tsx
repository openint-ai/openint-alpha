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
  fetchGraphRecentQueries,
  fetchGraphNodeDetails,
  fetchGraphSentiment,
  type GraphStats,
  type GraphSchemaResponse,
  type GraphSampleResponse,
  type GraphQueryNaturalResponse,
  type GraphNodeDetailsResponse,
} from '../api';
import { getModelDisplayName, getModelUrl } from '../utils/modelMeta';

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

/** Map column name (from Cypher RETURN) to link type. LLM often returns c.id, d.id, t.id. */
function getIdLinkType(col: string): 'customer_id' | 'transaction_id' | 'dispute_id' | null {
  const k = col.trim().toLowerCase();
  if (k === 'customer_id' || k === 'c.id') return 'customer_id';
  if (k === 'transaction_id' || k === 't.id') return 'transaction_id';
  if (k === 'dispute_id' || k === 'd.id') return 'dispute_id';
  return null;
}

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
  const [, setSchema] = useState<GraphSchemaResponse | null>(null);
  const [, setSample] = useState<GraphSampleResponse | null>(null);
  const [queryResult, setQueryResult] = useState<GraphQueryNaturalResponse | null>(null);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(true);
  const [queryLoading, setQueryLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dataModelOpen, setDataModelOpen] = useState(false);
  const dataModelRef = useRef<HTMLDivElement>(null);
  /** When viewing detail (after clicking an ID link): previous result + input for "Back to results". */
  const [previousResult, setPreviousResult] = useState<GraphQueryNaturalResponse | null>(null);
  const [previousInput, setPreviousInput] = useState('');
  /** Breadcrumb for detail view, e.g. { label: "Customer CUST001" }. */
  const [detailBreadcrumb, setDetailBreadcrumb] = useState<{ label: string } | null>(null);
  /** Full node details when user clicks an ID (one row table). */
  const [nodeDetailResult, setNodeDetailResult] = useState<GraphNodeDetailsResponse | null>(null);
  /** Recent questions from Redis (cached). */
  const [recentQueries, setRecentQueries] = useState<string[]>([]);
  /** Tooltip for History: full question text + position. */
  const [historyTooltip, setHistoryTooltip] = useState<{ text: string; x: number; y: number } | null>(null);
  /** Sentiment for current dispute (LLM-generated: free-form text + optional emoji + confidence). */
  const [disputeSentiment, setDisputeSentiment] = useState<{ sentiment: string; confidence: number; emoji?: string } | null>(null);
  const [sentimentLoading, setSentimentLoading] = useState(false);
  const [sentimentError, setSentimentError] = useState<string | null>(null);

  useEffect(() => {
    if (!dataModelOpen) return;
    const onDocClick = (e: MouseEvent) => {
      if (dataModelRef.current && !dataModelRef.current.contains(e.target as Node)) setDataModelOpen(false);
    };
    document.addEventListener('mousedown', onDocClick);
    return () => document.removeEventListener('mousedown', onDocClick);
  }, [dataModelOpen]);

  const loadRecentQueries = useCallback(async () => {
    try {
      const res = await fetchGraphRecentQueries();
      setRecentQueries(res.queries || []);
    } catch {
      setRecentQueries([]);
    }
  }, []);

  // When showing a Dispute node, run LLM sentiment on description (or status + amount). Future: replace with sentiment-agent.
  useEffect(() => {
    if (!nodeDetailResult?.success || nodeDetailResult.label !== 'Dispute' || !nodeDetailResult.rows?.[0]) {
      setDisputeSentiment(null);
      setSentimentError(null);
      return;
    }
    const row = nodeDetailResult.rows[0] as Record<string, unknown>;
    const description = row.description != null ? String(row.description) : '';
    const status = row.dispute_status != null ? String(row.dispute_status) : '';
    const amount = row.amount_disputed != null ? String(row.amount_disputed) : '';
    const text = description.trim() || [status, amount, nodeDetailResult.id].filter(Boolean).join(' ').trim();
    if (!text) {
      setDisputeSentiment(null);
      setSentimentError(null);
      return;
    }
    let cancelled = false;
    setSentimentLoading(true);
    setSentimentError(null);
    setDisputeSentiment(null);
    fetchGraphSentiment(text)
      .then((res) => {
        if (cancelled) return;
        setSentimentLoading(false);
        if (res.success && res.sentiment != null && res.confidence != null) {
          setDisputeSentiment({
            sentiment: res.sentiment,
            confidence: res.confidence,
            emoji: res.emoji ?? undefined,
          });
          setSentimentError(null);
        } else {
          setSentimentError(res.error || 'Sentiment analysis failed');
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setSentimentLoading(false);
          setSentimentError(e instanceof Error ? e.message : 'Sentiment failed');
        }
      });
    return () => {
      cancelled = true;
    };
  }, [nodeDetailResult?.success, nodeDetailResult?.label, nodeDetailResult?.id, nodeDetailResult?.rows]);

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
      await loadRecentQueries();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [loadRecentQueries]);

  useEffect(() => {
    load();
  }, [load]);

  const runQueryNatural = useCallback(async () => {
    const q = input.trim();
    if (!q) return;
    setQueryLoading(true);
    setError(null);
    setDetailBreadcrumb(null);
    setPreviousResult(null);
    setPreviousInput('');
    setNodeDetailResult(null);
    try {
      const res = await runGraphQueryNatural(q);
      setQueryResult(res);
      await loadRecentQueries();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setQueryResult(null);
    } finally {
      setQueryLoading(false);
    }
  }, [input, loadRecentQueries]);

  const fillAndRun = useCallback((question: string) => {
    setInput(question);
    setQueryLoading(true);
    setError(null);
    setDetailBreadcrumb(null);
    setPreviousResult(null);
    setPreviousInput('');
    setNodeDetailResult(null);
    runGraphQueryNatural(question)
      .then(setQueryResult)
      .then(() => loadRecentQueries())
      .catch((e) => {
        setError(e instanceof Error ? e.message : String(e));
        setQueryResult(null);
      })
      .finally(() => setQueryLoading(false));
  }, [loadRecentQueries]);

  const goBackToResults = useCallback(() => {
    setQueryResult(previousResult);
    setInput(previousInput);
    setDetailBreadcrumb(null);
    setNodeDetailResult(null);
    setPreviousResult(null);
    setPreviousInput('');
  }, [previousResult, previousInput]);

  const linkTypeToLabel: Record<string, string> = { customer_id: 'Customer', transaction_id: 'Transaction', dispute_id: 'Dispute' };

  const runQueryForId = useCallback((col: string, value: string) => {
    const linkType = getIdLinkType(col);
    if (!linkType) return;
    const label = linkTypeToLabel[linkType];
    if (!label) return;
    setPreviousResult(queryResult);
    setPreviousInput(input);
    const entityLabel = linkType.replace(/_id$/, '').replace(/^./, (s) => s.toUpperCase());
    setDetailBreadcrumb({ label: `${entityLabel} ${value}` });
    setNodeDetailResult(null);
    setQueryLoading(true);
    setError(null);
    fetchGraphNodeDetails(label, String(value).trim())
      .then((res) => {
        setNodeDetailResult(res);
        if (res.success && res.rows?.length) setInput(`Details for ${label} ${value}`);
      })
      .catch((e) => {
        setError(e instanceof Error ? e.message : String(e));
        setNodeDetailResult(null);
      })
      .finally(() => setQueryLoading(false));
  }, [queryResult, input]);

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
              {recentQueries.length > 0 && (
                <div>
                  <p className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">History</p>
                  <ul className="space-y-1 max-h-48 overflow-y-auto">
                    {recentQueries.map((q) => (
                      <li key={q}>
                        <button
                          type="button"
                          onClick={() => fillAndRun(q)}
                          disabled={queryLoading}
                          onMouseEnter={(e) => setHistoryTooltip({ text: q, x: e.clientX, y: e.clientY })}
                          onMouseLeave={() => setHistoryTooltip(null)}
                          className="w-full text-left px-3 py-2 text-xs rounded-lg bg-slate-50 border border-slate-200 text-blue-600 underline underline-offset-2 hover:bg-slate-100 hover:border-slate-300 truncate disabled:opacity-50"
                        >
                          {q.length > 60 ? `${q.slice(0, 60)}…` : q}
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        </aside>

        {/* History tooltip: full question on hover (matches center-pane Question block) */}
        {historyTooltip && (
          <div
            className="fixed z-[100] pointer-events-none max-w-sm rounded-xl border border-indigo-100 bg-gradient-to-br from-indigo-50/80 to-white p-4 shadow-sm"
            style={{
              left: historyTooltip.x,
              top: historyTooltip.y - 8,
              transform: 'translate(-50%, -100%)',
            }}
          >
            <p className="text-xs font-semibold text-indigo-600 uppercase tracking-wider mb-2">Question</p>
            <p className="text-sm text-gray-800 leading-relaxed whitespace-pre-wrap break-words">{historyTooltip.text}</p>
          </div>
        )}

        {/* Center pane: results */}
        <main className="flex-1 min-w-0">
          <section className="rounded-xl border border-surface-200 bg-white shadow-soft overflow-hidden">
            <h2 className="px-4 py-3 border-b border-surface-200 text-sm font-semibold text-gray-900 bg-gradient-to-r from-surface-50 to-white">
              Results
            </h2>
            <div className="p-4">
              {detailBreadcrumb && (
                <nav className="mb-4 flex items-center gap-2 text-sm" aria-label="Breadcrumb">
                  <button
                    type="button"
                    onClick={goBackToResults}
                    className="text-blue-600 hover:text-blue-800 hover:underline underline-offset-2 font-medium"
                  >
                    Results
                  </button>
                  <span className="text-gray-400" aria-hidden>›</span>
                  <span className="text-gray-700 font-medium">{detailBreadcrumb.label}</span>
                </nav>
              )}
              {queryLoading && (
                <p className="text-sm text-gray-500">Running query…</p>
              )}
              {!queryLoading && nodeDetailResult && (
                <>
                  {nodeDetailResult.error && (
                    <p className="text-sm text-red-600 mb-3">{nodeDetailResult.error}</p>
                  )}
                  {nodeDetailResult.cypher && (
                    <div className="mb-4 rounded-xl border border-slate-200 bg-slate-800 shadow-sm overflow-hidden">
                      <div className="px-4 py-2 border-b border-slate-600 bg-slate-700/50">
                        <p className="text-xs font-semibold text-slate-300 uppercase tracking-wider">Neo4j Cypher (node lookup)</p>
                      </div>
                      <pre className="p-4 text-xs font-mono text-slate-200 leading-relaxed overflow-x-auto whitespace-pre-wrap break-words">
                        {nodeDetailResult.cypher.trim()}
                      </pre>
                      {nodeDetailResult.params && Object.keys(nodeDetailResult.params).length > 0 && (
                        <div className="px-4 pb-4 pt-0">
                          <p className="text-xs text-slate-400 mb-1">Parameters:</p>
                          <pre className="text-xs font-mono text-slate-300 whitespace-pre-wrap break-words">
                            {JSON.stringify(nodeDetailResult.params, null, 2)}
                          </pre>
                        </div>
                      )}
                    </div>
                  )}
                  {nodeDetailResult.success && nodeDetailResult.rows?.[0] && (
                    <div className="overflow-x-auto">
                      <p className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">
                        {nodeDetailResult.label} details · {nodeDetailResult.id}
                      </p>
                      <p className="text-sm text-gray-600 mb-3">
                        All properties for this {nodeDetailResult.label.toLowerCase()} in table format.
                      </p>
                      {nodeDetailResult.label === 'Customer' && (() => {
                        const cols = nodeDetailResult.columns ?? Object.keys(nodeDetailResult.rows[0]);
                        const onlyId = cols.length === 1 && (cols[0] === 'id' || cols[0] === 'customer_id');
                        return onlyId ? (
                          <div className="mb-4 rounded-xl border border-amber-200 bg-amber-50/80 p-4 text-sm text-amber-800">
                            <p className="font-medium mb-1">Only ID is stored for this customer in the graph.</p>
                            <p className="text-amber-700">
                              To see full details (name, email, address, etc.), run the Neo4j loader with customer enrichment and ensure <code className="bg-amber-100 px-1 rounded">dimensions/customers.csv</code> is loaded (step 3: enrich Customer nodes).
                            </p>
                          </div>
                        ) : null;
                      })()}
                      {nodeDetailResult.label === 'Dispute' && (
                        <div className="mb-4 rounded-xl border border-amber-200 bg-gradient-to-br from-amber-50/80 to-white p-4 shadow-sm">
                          <p className="text-xs font-semibold text-amber-700 uppercase tracking-wider mb-1">Sentiment (LLM-generated, from description)</p>
                          <p className="text-xs text-amber-600/90 mb-2">Powered by LLM · future: sentiment-agent</p>
                          {sentimentLoading && (
                            <p className="text-sm text-gray-500">Analyzing sentiment…</p>
                          )}
                          {sentimentError && !sentimentLoading && (
                            <p className="text-sm text-amber-700">{sentimentError}</p>
                          )}
                          {disputeSentiment && !sentimentLoading && (
                            <div className="flex flex-wrap items-center gap-3">
                              <span className="text-2xl" title={disputeSentiment.sentiment} aria-hidden>
                                {disputeSentiment.emoji ?? '💬'}
                              </span>
                              <span className="text-sm font-medium text-gray-800">
                                {disputeSentiment.sentiment}
                              </span>
                              <span className="text-xs text-gray-500 font-mono">
                                Confidence: {disputeSentiment.confidence.toFixed(2)} (0–1)
                              </span>
                            </div>
                          )}
                        </div>
                      )}
                      <table className="w-full text-sm border-collapse border border-surface-200">
                        <thead>
                          <tr className="border-b border-surface-200 bg-surface-50">
                            <th className="text-left px-3 py-2 font-medium text-gray-700 border-b border-surface-200">Property</th>
                            <th className="text-left px-3 py-2 font-medium text-gray-700 border-b border-surface-200">Value</th>
                          </tr>
                        </thead>
                        <tbody>
                          {(nodeDetailResult.columns ?? Object.keys(nodeDetailResult.rows[0])).map((col) => {
                            const raw = nodeDetailResult.rows[0][col];
                            const display =
                              raw == null
                                ? '—'
                                : typeof raw === 'object'
                                  ? JSON.stringify(raw, null, 2)
                                  : String(raw);
                            return (
                              <tr key={col} className="border-b border-surface-100 hover:bg-surface-50">
                                <td className="px-3 py-2 font-medium text-gray-700 align-top border-r border-surface-100">{col.replace(/_/g, ' ')}</td>
                                <td className="px-3 py-2 text-gray-800 font-mono text-xs whitespace-pre-wrap break-words align-top">
                                  {display}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  )}
                </>
              )}
              {!queryLoading && !nodeDetailResult && queryResult && (
                <>
                  {queryResult.error && (
                    <p className="text-sm text-red-600 mb-3">{queryResult.error}</p>
                  )}
                  {queryResult.query && (
                    <div className="mb-4 rounded-xl border border-indigo-100 bg-gradient-to-br from-indigo-50/80 to-white p-4 shadow-sm">
                      <p className="text-xs font-semibold text-indigo-600 uppercase tracking-wider mb-2">Question</p>
                      <p className="text-sm text-gray-800 leading-relaxed">
                        {queryResult.query}
                      </p>
                      {'llm_model' in queryResult && (queryResult.llm_model || queryResult.llm_time_ms != null) && (
                        <p className="mt-2 text-xs text-gray-500">
                          {getModelUrl(queryResult.llm_model as string) ? (
                            <>
                              <a
                                href={getModelUrl(queryResult.llm_model as string)!}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-brand-600 hover:underline"
                              >
                                {getModelDisplayName(queryResult.llm_model as string)}
                              </a>
                              {queryResult.llm_time_ms != null ? ` · ${queryResult.llm_time_ms}ms` : ''}
                            </>
                          ) : (
                            <>{getModelDisplayName(queryResult.llm_model as string)}{queryResult.llm_time_ms != null ? ` · ${queryResult.llm_time_ms}ms` : ''}</>
                          )}
                        </p>
                      )}
                    </div>
                  )}
                  {queryResult.cypher && (
                    <div className="mb-4 rounded-xl border border-slate-200 bg-slate-800 shadow-sm overflow-hidden">
                      <div className="px-4 py-2 border-b border-slate-600 bg-slate-700/50">
                        <p className="text-xs font-semibold text-slate-300 uppercase tracking-wider">Neo4j Cypher</p>
                      </div>
                      <pre className="p-4 text-xs font-mono text-slate-200 leading-relaxed overflow-x-auto whitespace-pre-wrap break-words">
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
                                {queryResult.columns.map((col) => {
                                  const val = row[col];
                                  const str = val != null ? String(val) : '—';
                                  const linkType = getIdLinkType(col);
                                  const isIdLink = linkType !== null && val != null && String(val).trim() !== '';
                                  return (
                                    <td key={col} className="px-3 py-2 text-gray-800 font-mono text-xs">
                                      {isIdLink ? (
                                        <button
                                          type="button"
                                          onClick={() => runQueryForId(col, String(val))}
                                          disabled={queryLoading}
                                          className="text-blue-600 underline underline-offset-2 hover:text-blue-800 cursor-pointer disabled:opacity-50 text-left font-mono"
                                          title={`Fetch graph details for this ${linkType.replace(/_id$/, '')}`}
                                        >
                                          {str}
                                        </button>
                                      ) : (
                                        str
                                      )}
                                    </td>
                                  );
                                })}
                              </tr>
                            ))
                          )}
                        </tbody>
                      </table>
                    </div>
                  ) : null}
                </>
              )}
              {!queryLoading && !queryResult && !nodeDetailResult && (
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

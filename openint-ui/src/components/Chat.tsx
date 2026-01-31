import { useState, useRef, useEffect, useLayoutEffect, useMemo, useCallback } from 'react';
import { chat, previewSemanticAnalysis, previewSemanticAnalysisMulti, type ChatSource, type ChatDebug, type DebugSemantic, type SemanticPreviewModelResult } from '../api';
import { getLogger } from '../observability';
import { AnswerRenderer } from './AnswerRenderer';
import {
  parseStructuredContent,
  getColumnOrder,
  getSectionTitle,
  ENTITY_TABLE_ORDER,
  FIELD_LABELS,
} from '../utils/structuredData';

type SuggestionItem = { query: string; category: string };

/** Full pool: Customer, Transactions (ACH, Wire, Credit Card, Check), Analytics & Insights. */
const EXAMPLE_POOL: SuggestionItem[] = [
  // Customer
  { query: 'Show me transactions for customer 1001', category: 'Customer' },
  { query: 'Customers in California with active accounts', category: 'Customer' },
  { query: 'Transactions for customer 1000000001', category: 'Customer' },
  { query: 'Customers in New York or Texas', category: 'Customer' },
  { query: 'Customers in ZIP code 90210', category: 'Customer' },
  { query: 'Customers with both ACH and wire activity', category: 'Customer' },
  { query: 'Where are our high-value wire transfer customers?', category: 'Customer' },
  { query: 'Top 10 most active customers this month', category: 'Customer' },
  { query: 'Top 5 customer IDs by total transaction amount', category: 'Customer' },
  { query: 'Breakdown of active vs closed accounts by region', category: 'Customer' },
  // ACH
  { query: 'Pending ACH payments', category: 'ACH' },
  { query: 'Top 15 highest ACH debits', category: 'ACH' },
  { query: 'ACH transactions over $5,000', category: 'ACH' },
  { query: 'ACH debits by customer', category: 'ACH' },
  { query: 'Failed or reversed ACH transactions', category: 'ACH' },
  // Wire
  { query: 'Wire transfers over $10,000', category: 'Wire' },
  { query: 'Top 10 largest wire transfers', category: 'Wire' },
  { query: 'Top 10 international wire transfers by amount', category: 'Wire' },
  { query: 'States with highest share of international wires', category: 'Wire' },
  { query: 'Pending wire transactions', category: 'Wire' },
  // Credit Card
  { query: 'Credit card disputes', category: 'Credit Card' },
  { query: 'Top 10 credit card charges by amount', category: 'Credit Card' },
  { query: 'Where do we have the most disputed credit card transactions?', category: 'Credit Card' },
  { query: 'Credit card transactions over $1,000', category: 'Credit Card' },
  { query: 'Disputed credit card transactions by state', category: 'Credit Card' },
  // Check
  { query: 'Check payments over 2000', category: 'Check' },
  { query: 'Top 20 largest check payments', category: 'Check' },
  { query: 'Which regions have the most check usage?', category: 'Check' },
  { query: 'Pending check payments', category: 'Check' },
  { query: 'Check transactions between 1000 and 5000 USD', category: 'Check' },
  // Analytics & Insights
  { query: 'Top 10 customers by number of transactions', category: 'Analytics & Insights' },
  { query: 'Top 5 states by customer count', category: 'Analytics & Insights' },
  { query: 'Top 5 ZIP codes by transaction volume', category: 'Analytics & Insights' },
  { query: 'Top 5 states with most pending transactions', category: 'Analytics & Insights' },
  { query: 'Top 8 cities by customer count', category: 'Analytics & Insights' },
  { query: 'Which states have the most closed accounts?', category: 'Analytics & Insights' },
  { query: 'Which transaction types have the most pending status?', category: 'Analytics & Insights' },
  { query: 'Compare ACH vs wire transaction volumes', category: 'Analytics & Insights' },
  { query: 'Summary of transaction mix by state', category: 'Analytics & Insights' },
  { query: 'Which ZIP codes have the most failed or reversed transactions?', category: 'Analytics & Insights' },
  { query: 'Insight: customers with large debits and small credits', category: 'Analytics & Insights' },
  { query: 'Transactions between 1000 and 5000 USD', category: 'Analytics & Insights' },
];

const CATEGORY_ORDER = ['Analytics & Insights', 'Customer', 'ACH', 'Wire', 'Credit Card', 'Check'] as const;

/** Category accent for clear visual distinction. */
const CATEGORY_STYLES: Record<string, { label: string; border: string; bg: string; text: string }> = {
  Customer: { label: 'Customer', border: 'border-l-blue-500', bg: 'bg-blue-50', text: 'text-blue-800' },
  ACH: { label: 'ACH', border: 'border-l-amber-500', bg: 'bg-amber-50', text: 'text-amber-800' },
  Wire: { label: 'Wire', border: 'border-l-emerald-500', bg: 'bg-emerald-50', text: 'text-emerald-800' },
  'Credit Card': { label: 'Credit Card', border: 'border-l-violet-500', bg: 'bg-violet-50', text: 'text-violet-800' },
  Check: { label: 'Check', border: 'border-l-teal-500', bg: 'bg-teal-50', text: 'text-teal-800' },
  'Analytics & Insights': { label: 'Analytics & Insights', border: 'border-l-slate-600', bg: 'bg-slate-100', text: 'text-slate-800' },
};

function shuffle<T>(arr: T[], seed: number): T[] {
  const out = [...arr];
  let s = seed;
  for (let i = out.length - 1; i > 0; i--) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    const j = s % (i + 1);
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

function groupByCategory(pool: SuggestionItem[], seed: number): Map<string, string[]> {
  const shuffled = shuffle(pool, seed);
  const map = new Map<string, string[]>();
  for (const { query, category } of shuffled) {
    if (!map.has(category)) map.set(category, []);
    map.get(category)!.push(query);
  }
  return map;
}

const SEMANTIC_COLORS: Record<string, string> = {
  amount_min: 'bg-gradient-to-r from-amber-100 to-amber-50 text-amber-900 border-amber-300 shadow-sm',
  amount_max: 'bg-gradient-to-r from-amber-100 to-amber-50 text-amber-900 border-amber-300 shadow-sm',
  amount_between: 'bg-gradient-to-r from-amber-100 to-amber-50 text-amber-900 border-amber-300 shadow-sm',
  state: 'bg-gradient-to-r from-blue-100 to-blue-50 text-blue-900 border-blue-300 shadow-sm',
  status: 'bg-gradient-to-r from-slate-100 to-slate-50 text-slate-800 border-slate-300 shadow-sm',
  customer_id: 'bg-gradient-to-r from-indigo-100 to-indigo-50 text-indigo-900 border-indigo-300 shadow-sm',
  intent: 'bg-gradient-to-r from-emerald-100 to-emerald-50 text-emerald-900 border-emerald-300 shadow-sm',
  top_n: 'bg-gradient-to-r from-violet-100 to-violet-50 text-violet-900 border-violet-300 shadow-sm',
  analytical: 'bg-gradient-to-r from-purple-100 to-purple-50 text-purple-900 border-purple-300 shadow-sm',
  result_shape: 'bg-gradient-to-r from-sky-100 to-sky-50 text-sky-900 border-sky-300 shadow-sm',
  dispute_status: 'bg-gradient-to-r from-rose-100 to-rose-50 text-rose-900 border-rose-300 shadow-sm',
  date_range: 'bg-gradient-to-r from-teal-100 to-teal-50 text-teal-900 border-teal-300 shadow-sm',
  want_count: 'bg-gradient-to-r from-cyan-100 to-cyan-50 text-cyan-900 border-cyan-300 shadow-sm',
};
function getSemanticColor(type: string): string {
  return SEMANTIC_COLORS[type] ?? 'bg-gradient-to-r from-gray-100 to-gray-50 text-gray-800 border-gray-300 shadow-sm';
}

/** Model metadata for the info popup: author, description, details, and Hugging Face URL. */
const MODEL_META: Record<
  string,
  { author: string; description: string; details: string; url: string }
> = {
  'mukaj/fin-mpnet-base': {
    author: 'mukaj',
    description:
      'State-of-the-art for financial documents (79.91 FiQA). Trained on 150k+ financial document QA examples. Best for banking/finance semantic search.',
    details: '768 dimensions · Fast · Use for banking/finance applications.',
    url: 'https://huggingface.co/mukaj/fin-mpnet-base',
  },
  'ProsusAI/finbert': {
    author: 'Prosus AI',
    description:
      'Finance-oriented model suited for financial text understanding and semantic tasks.',
    details: 'Finance domain · Use for financial sentiment and understanding.',
    url: 'https://huggingface.co/ProsusAI/finbert',
  },
  'sentence-transformers/all-mpnet-base-v2': {
    author: 'sentence-transformers',
    description:
      'Popular, powerful open source model. Strong general-purpose embeddings (768d). Good balance of quality and speed.',
    details: '768 dimensions · Popular · Use for general semantic tasks.',
    url: 'https://huggingface.co/sentence-transformers/all-mpnet-base-v2',
  },
};

/** Clickable model name that opens a modal with author, description, details, and link to Hugging Face. */
function ModelLabelWithInfo({ modelId, modelLabel }: { modelId: string; modelLabel: string }) {
  const [popupOpen, setPopupOpen] = useState(false);
  const meta = modelId && modelId !== 'all' ? MODEL_META[modelId] : null;

  useEffect(() => {
    if (!popupOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setPopupOpen(false);
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [popupOpen]);

  if (!meta) {
    return (
      <span className="rounded-md bg-brand-100/80 px-1.5 py-0.5 text-[10px] font-semibold text-brand-800 border border-brand-200/80">
        {modelLabel}
      </span>
    );
  }

  return (
    <>
      <button
        type="button"
        onClick={() => setPopupOpen(true)}
        className="rounded-md bg-brand-100/80 px-1.5 py-0.5 text-[10px] font-semibold text-brand-800 border border-brand-200/80 hover:bg-brand-200/80 hover:border-brand-300 transition-colors cursor-pointer text-left underline decoration-brand-400/60 hover:decoration-brand-600"
        title="Click for model details"
      >
        {modelLabel}
      </button>
      {popupOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40"
          onClick={() => setPopupOpen(false)}
          onKeyDown={(e) => e.key === 'Escape' && setPopupOpen(false)}
          role="dialog"
          aria-modal="true"
          aria-labelledby="model-popup-title"
        >
          <div
            className="bg-white rounded-xl border border-surface-200 shadow-xl max-w-md w-full p-5 space-y-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-2">
              <h3 id="model-popup-title" className="text-sm font-bold text-gray-900">
                {modelLabel}
              </h3>
              <button
                type="button"
                onClick={() => setPopupOpen(false)}
                className="shrink-0 rounded-lg p-1 text-gray-500 hover:bg-surface-100 hover:text-gray-700"
                aria-label="Close"
              >
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <p className="text-[11px] text-gray-600">
              <span className="font-semibold text-gray-700">By {meta.author}</span>
            </p>
            <p className="text-[12px] text-gray-700 leading-snug">{meta.description}</p>
            <p className="text-[11px] text-gray-500">{meta.details}</p>
            <a
              href={meta.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 rounded-lg bg-brand-500 px-3 py-2 text-[12px] font-medium text-white hover:bg-brand-600 transition-colors"
            >
              View on Hugging Face
              <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
            </a>
          </div>
        </div>
      )}
    </>
  );
}

/** Reusable block: semantic annotation (highlights + tags) for a query. Used for live preview and per-turn in left pane. */
function SemanticPreviewBlock({
  query,
  models,
  loading,
  title,
  modelLabel,
  modelId,
  semanticAnnotationTimeMs,
}: {
  query: string;
  models: Record<string, SemanticPreviewModelResult>;
  loading: boolean;
  title?: string;
  /** Model name from dropdown to show after the title (e.g. "Finance MPNet"). */
  modelLabel?: string;
  /** Model id for info bubble (e.g. "mukaj/fin-mpnet-base"). */
  modelId?: string;
  /** Time in ms for semantic annotation using the dropdown model (displayed when set). */
  semanticAnnotationTimeMs?: number;
}) {
  const label = title ?? 'Semantic annotation';
  return (
    <div className="rounded-lg bg-white border border-surface-200 p-3 shadow-sm space-y-4">
      <div className="flex items-center justify-between mb-2 gap-2 flex-wrap">
        <span className="text-[10px] font-semibold text-gray-600 uppercase tracking-wide flex items-center gap-1.5 flex-wrap">
          {label}
          {modelLabel && (
            <>
              <span className="text-gray-400">—</span>
              <ModelLabelWithInfo modelId={modelId ?? ''} modelLabel={modelLabel} />
            </>
          )}
        </span>
        <span className="flex items-center gap-2">
          {semanticAnnotationTimeMs != null && !loading && (
            <span className="text-[10px] font-mono text-gray-500 tabular-nums" title="Time for semantic annotation with selected model">
              {semanticAnnotationTimeMs < 1 && semanticAnnotationTimeMs > 0
                ? `${(semanticAnnotationTimeMs * 1000).toFixed(0)} μs`
                : `${semanticAnnotationTimeMs} ms`}
            </span>
          )}
          {loading && <span className="text-[9px] text-gray-400 animate-pulse">Analyzing…</span>}
        </span>
      </div>
      {loading ? (
        <div className="text-xs text-gray-400 italic">Processing sentence tokenization and semantic tags…</div>
      ) : Object.keys(models).length > 0 ? (
        <div className="space-y-4">
          {Object.entries(models).map(([modelId, modelResult]) => {
            const segments = modelResult.highlighted_segments ?? [];
            const tags = modelResult.tags ?? [];
            return (
              <div key={modelId} className="rounded-lg border border-surface-200/80 bg-surface-50/50 p-2.5 space-y-2">
                {modelResult.error && (
                  <div className="flex justify-end">
                    <span className="text-[9px] text-rose-600" title={modelResult.error}>Error</span>
                  </div>
                )}
                <div className="text-sm text-gray-700 leading-relaxed">
                  {segments.length > 0 ? (
                    segments.map((seg, i) =>
                      seg.type === 'highlight' && seg.tag ? (
                        <span
                          key={i}
                          className={`inline-block rounded-md px-1.5 py-0.5 mx-0.5 border font-medium transition-all hover:scale-105 ${getSemanticColor(seg.tag_type ?? seg.tag.type)}`}
                          title={`${seg.label ?? seg.tag.label}: ${seg.tag.value} (confidence: ${(seg.confidence ?? seg.tag.confidence ?? 0).toFixed(2)})`}
                        >
                          {seg.text}
                        </span>
                      ) : (
                        <span key={i} className="text-gray-800">{seg.text}</span>
                      )
                    )
                  ) : (
                    <span className="text-gray-600">{query}</span>
                  )}
                </div>
                {tags.length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-1.5">
                    {tags.map((tag, i) => (
                      <span
                        key={i}
                        className={`inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-medium border ${getSemanticColor(tag.type)}`}
                        title={`${tag.label}: ${tag.value ?? tag.snippet} (${(tag.confidence ?? 0).toFixed(2)})`}
                      >
                        {tag.type}: {typeof tag.value !== 'undefined' ? String(tag.value) : tag.snippet}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ) : (
        <div className="text-xs text-gray-400 italic">No semantic tags detected</div>
      )}
    </div>
  );
}

/** Format ISO timestamp for display: "3:45 PM" same day, "Jan 28, 3:45 PM" otherwise. */
function formatMessageTime(iso?: string): string {
  if (!iso) return '';
  const d = new Date(iso);
  const now = new Date();
  const sameDay = d.getDate() === now.getDate() && d.getMonth() === now.getMonth() && d.getFullYear() === now.getFullYear();
  return sameDay
    ? d.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })
    : d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
}

/** Full date and time for clarity: "Jan 28, 2025, 3:45 PM". */
function formatMessageTimeFull(iso?: string): string {
  if (!iso) return '';
  return new Date(iso).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

/** Relative time for recent messages: "just now", "2 min ago", "1 hr ago", or short date. */
function formatRelativeTime(iso?: string): string {
  if (!iso) return '';
  const d = new Date(iso);
  const now = new Date();
  const sec = Math.floor((now.getTime() - d.getTime()) / 1000);
  if (sec < 10) return 'just now';
  if (sec < 60) return `${sec} sec ago`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min} min ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24 && d.getDate() === now.getDate()) return `${hr} hr ago`;
  return formatMessageTime(iso);
}

/** Renders query text with semantic snippets highlighted with modern gradient backgrounds. */
function QueryWithHighlights({ query, semantics }: { query: string; semantics: DebugSemantic[] }) {
  const safeQuery = query ?? '';
  const safeSemantics = Array.isArray(semantics) ? semantics : [];
  type Seg = { start: number; end: number; type?: string; label?: string };
  const segments: Seg[] = [];
  const used: boolean[] = [];
  for (let i = 0; i < safeQuery.length; i++) used[i] = false;
  
  // Sort semantics by position in query, handling overlaps
  const sorted = safeSemantics
    .filter((s) => s.snippet && safeQuery.toLowerCase().includes(String(s.snippet).toLowerCase()))
    .map((s) => {
      // Find all occurrences of snippet (case-insensitive)
      const snippetLower = String(s.snippet).toLowerCase();
      const queryLower = safeQuery.toLowerCase();
      const idx = queryLower.indexOf(snippetLower);
      return { start: idx, end: idx + String(s.snippet).length, type: s.type, label: s.label, snippet: s.snippet };
    })
    .filter((s) => s.start >= 0)
    .sort((a, b) => a.start - b.start);
  
  // Build segments, handling overlaps by taking the first match
  let last = 0;
  for (const s of sorted) {
    // Skip if this segment overlaps with a previous one
    if (s.start < last) continue;
    
    // Add text before this segment
    if (s.start > last) {
      segments.push({ start: last, end: s.start });
    }
    
    // Add the highlighted segment
    segments.push({ start: s.start, end: s.end, type: s.type, label: s.label });
    last = s.end;
  }
  
  // Add remaining text
  if (last < safeQuery.length) {
    segments.push({ start: last, end: safeQuery.length });
  }
  
  return (
    <div className="bg-white rounded-lg p-3 border border-surface-200 shadow-sm">
      <p className="text-sm text-gray-700 leading-relaxed font-medium">
        {segments.map((seg, i) =>
          seg.type ? (
            <span
              key={i}
              className={`inline-block rounded-md px-1.5 py-0.5 mx-0.5 border font-medium transition-all hover:scale-105 ${getSemanticColor(seg.type)}`}
              title={`${seg.label}: ${safeQuery.slice(seg.start, seg.end)}`}
            >
              {safeQuery.slice(seg.start, seg.end)}
            </span>
          ) : (
            <span key={i} className="text-gray-800">{safeQuery.slice(seg.start, seg.end)}</span>
          )
        )}
      </p>
    </div>
  );
}

/** Debug panel: semantic annotation (from dropdown model) + query vector visualization. Merged with Semantic annotation. */
function DebugPanel({
  debug,
  queriedAt,
  selectedModel,
  semanticPreview,
}: {
  debug: ChatDebug;
  queriedAt?: string;
  /** Model from dropdown – used for label and, when semanticPreview is provided, for semantic section */
  selectedModel: string;
  /** Optional per-turn semantic annotation (from dropdown model); when present, used for Sentence → Semantic Parts */
  semanticPreview?: { query: string; models: Record<string, SemanticPreviewModelResult>; loading: boolean };
}) {
  const { query, semantics, token_semantics } = debug ?? {};
  const safeQuery = query ?? '';
  const safeSemantics = Array.isArray(semantics) ? semantics : [];

  const isAllModels = selectedModel === 'all';
  const modelResult = semanticPreview?.models[selectedModel];
  const useSemanticPreviewSingle = semanticPreview && !semanticPreview.loading && modelResult;
  const useSemanticPreviewAll = isAllModels && semanticPreview && !semanticPreview.loading && Object.keys(semanticPreview.models ?? {}).length > 0;
  const allModelEntries = useSemanticPreviewAll ? Object.entries(semanticPreview!.models!) : [];
  const segments = useSemanticPreviewSingle ? (modelResult!.highlighted_segments ?? []) : [];
  const previewTags = useSemanticPreviewSingle ? (modelResult!.tags ?? []) : [];
  const selectedModelLabel = selectedModel === 'all' ? 'All models' : (MODEL_DISPLAY_NAMES[selectedModel] ?? selectedModel);
  const semanticsModelLabel = useSemanticPreviewSingle || useSemanticPreviewAll
    ? selectedModelLabel
    : (debug?.embedding_model ? (MODEL_DISPLAY_NAMES[debug.embedding_model] ?? debug.embedding_model) : '—');

  return (
    <div className="rounded-xl bg-gradient-to-br from-surface-50 to-white border-2 border-surface-200 p-5 space-y-5 mb-4 shadow-lg">
      <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-surface-200">
        <span className="text-xs font-bold text-gray-700 uppercase tracking-wider">🔍 Debug: Query Understanding</span>
        {queriedAt && (
          <span className="text-[10px] text-gray-400" title={formatMessageTimeFull(queriedAt)}>
            {formatMessageTime(queriedAt)}
          </span>
        )}
      </div>
      <p className="text-[11px] font-medium text-gray-600 -mt-1 flex items-center gap-1.5 flex-wrap">
        Semantics extracted with:{' '}
        <ModelLabelWithInfo
          modelId={useSemanticPreviewSingle || useSemanticPreviewAll ? selectedModel : (debug?.embedding_model ?? '')}
          modelLabel={semanticsModelLabel}
        />
      </p>

      {/* When "All models": list each model's interpretation */}
      {useSemanticPreviewAll && (
        <div className="space-y-4">
          {allModelEntries.map(([modelId, mr]) => {
            const modelSegments = mr.highlighted_segments ?? [];
            const modelTags = mr.tags ?? [];
            const modelLabel = MODEL_DISPLAY_NAMES[modelId] ?? modelId;
            return (
              <div key={modelId} className="rounded-lg border border-surface-200 bg-surface-50/50 p-4 space-y-3">
                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-semibold text-gray-700">{modelLabel}</span>
                  <ModelLabelWithInfo modelId={modelId} modelLabel={modelLabel} />
                </div>
                <div className="space-y-1.5">
                  <span className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide">Sentence → Semantic Parts</span>
                  {modelSegments.length > 0 ? (
                    <div className="text-sm text-gray-700 leading-relaxed">
                      {modelSegments.map((seg, i) =>
                        seg.type === 'highlight' && seg.tag ? (
                          <span
                            key={i}
                            className={`inline-block rounded-md px-1.5 py-0.5 mx-0.5 border font-medium ${getSemanticColor(seg.tag_type ?? seg.tag.type)}`}
                            title={`${seg.label ?? seg.tag.label}: ${seg.tag.value}`}
                          >
                            {seg.text}
                          </span>
                        ) : (
                          <span key={i} className="text-gray-800">{seg.text}</span>
                        )
                      )}
                    </div>
                  ) : (
                    <span className="text-xs text-gray-500">No segments</span>
                  )}
                </div>
                {modelTags.length > 0 && (
                  <div className="space-y-1.5">
                    <span className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide">Extracted Semantics</span>
                    <div className="flex flex-wrap gap-2">
                      {modelTags.map((s, i) => (
                        <div
                          key={i}
                          className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-[10px] font-semibold ${getSemanticColor(s.type)}`}
                        >
                          <span className="font-bold opacity-80">{(s as { label?: string }).label ?? (s as { type: string }).type}:</span>
                          <span className="font-mono">{String((s as { value?: string | number }).value ?? (s as { snippet?: string }).snippet ?? s)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {mr.error && (
                  <p className="text-[10px] text-rose-600">{mr.error}</p>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Single model: Semantic annotation (from dropdown model): Sentence → Semantic Parts */}
      {!useSemanticPreviewAll && (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-semibold text-gray-600 uppercase tracking-wide">Sentence → Semantic Parts</span>
            <span className="text-[9px] text-gray-400">(highlighted in query)</span>
          </div>
          {useSemanticPreviewSingle && segments.length > 0 ? (
            <div className="text-sm text-gray-700 leading-relaxed">
              {segments.map((seg, i) =>
                seg.type === 'highlight' && seg.tag ? (
                  <span
                    key={i}
                    className={`inline-block rounded-md px-1.5 py-0.5 mx-0.5 border font-medium ${getSemanticColor(seg.tag_type ?? seg.tag.type)}`}
                    title={`${seg.label ?? seg.tag.label}: ${seg.tag.value}`}
                  >
                    {seg.text}
                  </span>
                ) : (
                  <span key={i} className="text-gray-800">{seg.text}</span>
                )
              )}
            </div>
          ) : (
            <QueryWithHighlights query={safeQuery} semantics={safeSemantics} />
          )}
        </div>
      )}

      {/* Extracted Semantics Tags (single model or debug) */}
      {!useSemanticPreviewAll && (useSemanticPreviewSingle ? previewTags : safeSemantics).length > 0 && (
        <div className="space-y-2">
          <div className="text-[11px] font-semibold text-gray-600 uppercase tracking-wide">Extracted Semantics</div>
          <div className="flex flex-wrap gap-2">
            {(useSemanticPreviewSingle ? previewTags : safeSemantics).map((s, i) => (
              <div
                key={i}
                className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-[10px] font-semibold transition-transform hover:scale-105 ${getSemanticColor(s.type)}`}
              >
                <span className="font-bold opacity-80">{(s as { label?: string }).label ?? (s as { type: string }).type}:</span>
                <span className="font-mono">{String((s as { value?: string | number }).value ?? (s as { snippet?: string }).snippet ?? s)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Token-level semantic mapping */}
      {token_semantics && token_semantics.length > 0 && (
        <div className="space-y-2 pt-2 border-t border-surface-200">
          <div className="text-[11px] font-semibold text-gray-600 uppercase tracking-wide">
            Token → Semantic Mapping
          </div>
          <div className="bg-white rounded-lg p-3 border border-surface-200">
            <div className="flex flex-wrap gap-2">
              {token_semantics.map((ts, i) => (
                <div
                  key={i}
                  className="inline-flex flex-col gap-1 rounded-md border border-surface-200 bg-surface-50 px-2 py-1.5"
                >
                  <span className="text-[10px] font-bold text-gray-700">{ts.token}</span>
                  <div className="flex flex-wrap gap-1">
                    {ts.meanings.map((meaning, j) => (
                      <span
                        key={j}
                        className="text-[9px] text-gray-600 bg-white px-1.5 py-0.5 rounded border border-surface-200"
                      >
                        {meaning}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
          <p className="text-[9px] text-gray-500 italic">
            {useSemanticPreviewSingle
              ? `Tokens are mapped to semantic meanings from ${selectedModelLabel}.`
              : 'Each token in your query is mapped to potential semantic meanings for vector DB queries.'}
          </p>
        </div>
      )}
    </div>
  );
}

/** Right panel: suggestions grouped by category with clear visual hierarchy. */
function SuggestionsPanel({
  byCategory,
  usedInSession,
  onSelect,
  onMore,
  loading,
}: {
  byCategory: Map<string, string[]>;
  usedInSession: Set<string>;
  onSelect: (q: string) => void;
  onMore: () => void;
  loading: boolean;
}) {
  return (
    <aside className="w-72 lg:w-80 shrink-0 hidden md:flex flex-col h-full min-h-0 order-last">
      <div className="flex-1 min-h-0 flex flex-col rounded-xl border border-surface-200 bg-white shadow-soft overflow-hidden">
        <div className="flex-shrink-0 px-4 py-3.5 border-b border-surface-200 bg-surface-50 flex items-center justify-between">
          <span className="text-sm font-bold text-gray-800">Suggestions</span>
          <button
            type="button"
            onClick={onMore}
            className="text-xs font-semibold text-brand-600 hover:text-brand-700 hover:bg-brand-50 px-2.5 py-1.5 rounded-lg transition-colors"
          >
            Shuffle
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-3 space-y-5">
          {CATEGORY_ORDER.map((cat) => {
            const queries = byCategory.get(cat);
            if (!queries?.length) return null;
            const style = CATEGORY_STYLES[cat] ?? { label: cat, border: 'border-l-gray-400', bg: 'bg-gray-50', text: 'text-gray-800' };
            return (
              <section key={cat} className="space-y-2">
                <div className={`pl-3 py-1.5 border-l-4 rounded-r ${style.border} ${style.bg} ${style.text}`}>
                  <span className="text-xs font-bold uppercase tracking-wider">{style.label}</span>
                </div>
                <div className="flex flex-col gap-1.5 pl-0.5">
                  {queries.map((q, idx) => {
                    const used = usedInSession.has(q);
                    return (
                      <button
                        key={`${cat}-${idx}-${q.slice(0, 15)}`}
                        type="button"
                        onClick={() => onSelect(q)}
                        disabled={loading}
                        className={`w-full px-3 py-2 rounded-lg text-xs font-medium transition-colors border text-left text-gray-700 ${
                          used
                            ? 'bg-gray-50 text-gray-400 border-gray-200 cursor-default'
                            : 'bg-white hover:bg-brand-50 hover:border-brand-200 hover:text-brand-800 border-surface-200'
                        } ${loading ? 'opacity-70' : ''}`}
                        title={q}
                      >
                        <span className="line-clamp-2">{q}</span>
                      </button>
                    );
                  })}
                </div>
              </section>
            );
          })}
        </div>
      </div>
    </aside>
  );
}

/** Finance-specific embedding model */
const FINANCE_MODEL = 'mukaj/fin-mpnet-base';

/** Display names for multi-model semantic preview */
const MODEL_DISPLAY_NAMES: Record<string, string> = {
  'mukaj/fin-mpnet-base': 'Finance MPNet',
  'ProsusAI/finbert': 'FinBERT',
  'sentence-transformers/all-mpnet-base-v2': 'General MPNet',
};

/** Options for sentence tokenization / semantic tagging model dropdown */
const SEMANTIC_MODEL_OPTIONS: { value: string; label: string }[] = [
  { value: 'all', label: 'All models' },
  ...Object.entries(MODEL_DISPLAY_NAMES).map(([id, label]) => ({ value: id, label })),
];

const SEMANTIC_MODEL_STORAGE_KEY = 'openint_semantic_model';

/** Accent colors to visually link each query (left) with its result (center). Same index = same color. */
const PAIR_ACCENT_COLORS = [
  { border: 'border-l-4 border-l-blue-500', bg: 'bg-blue-50/50', bar: 'bg-blue-500', stroke: '#3b82f6' },
  { border: 'border-l-4 border-l-amber-500', bg: 'bg-amber-50/50', bar: 'bg-amber-500', stroke: '#f59e0b' },
  { border: 'border-l-4 border-l-emerald-500', bg: 'bg-emerald-50/50', bar: 'bg-emerald-500', stroke: '#10b981' },
  { border: 'border-l-4 border-l-violet-500', bg: 'bg-violet-50/50', bar: 'bg-violet-500', stroke: '#8b5cf6' },
  { border: 'border-l-4 border-l-rose-500', bg: 'bg-rose-50/50', bar: 'bg-rose-500', stroke: '#f43f5e' },
  { border: 'border-l-4 border-l-teal-500', bg: 'bg-teal-50/50', bar: 'bg-teal-500', stroke: '#14b8a6' },
] as const;

function getPairAccent(pairIndex: number) {
  return PAIR_ACCENT_COLORS[pairIndex % PAIR_ACCENT_COLORS.length];
}

/** Stable creation index: oldest pair = 0, newest = length-1. Colors stay consistent when new pairs are added. */
function getCreationIndex(idx: number, totalPairs: number) {
  return totalPairs - 1 - idx;
}

/** Apple-style switch: rounded pill, green when on. */
function DebugToggle({ on, onChange }: { on: boolean; onChange: (on: boolean) => void }) {
  return (
    <label className="inline-flex items-center gap-2 cursor-pointer select-none">
      <span className="text-xs text-gray-500">Debug</span>
      <button
        type="button"
        role="switch"
        aria-checked={on}
        aria-label={on ? 'Debug mode on' : 'Debug mode off'}
        onClick={() => onChange(!on)}
        className={`relative inline-flex h-6 w-11 shrink-0 rounded-full border-2 border-transparent transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-brand-400 focus:ring-offset-2 ${
          on ? 'bg-brand-500' : 'bg-gray-200'
        }`}
      >
        <span
          className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ${
            on ? 'translate-x-5' : 'translate-x-0.5'
          }`}
          style={{ marginTop: 1 }}
        />
      </button>
    </label>
  );
}


/** Subtle visualization: sentence → embedding vector → Vector DB (shown under each answer) */
function EmbeddingVisualization({ dims, model }: { dims: number; model?: string }) {
  const safeDims = typeof dims === 'number' && Number.isFinite(dims) ? dims : 0;
  const bars = 20;
  const tooltip = `Your question is turned into a ${safeDims}-dimensional vector (embedding) and used to search the vector DB.`;
  return (
    <div
      className="mt-2.5 pt-2.5 border-t border-surface-50/80 flex flex-wrap items-center gap-1.5 text-[9px] text-gray-400/90"
      title={tooltip}
    >
      <span className="shrink-0 text-gray-400/80">Sentence</span>
      <span className="text-gray-300/70">→</span>
      <div className="flex items-center gap-[1px]" aria-hidden>
        {Array.from({ length: bars }, (_, i) => (
          <div
            key={i}
            className="w-1 h-2 rounded-[1px] bg-gradient-to-t from-brand-400/30 to-brand-500/50"
            style={{ opacity: 0.35 + (0.5 * (i + 1)) / bars }}
          />
        ))}
      </div>
      <span className="shrink-0 font-mono text-gray-500/90">{safeDims}d</span>
      <span className="text-gray-300/70">→</span>
      <span className="shrink-0 text-gray-400/80">Vector DB</span>
      {(model != null && String(model).trim()) ? (
        <span className="shrink-0 text-gray-400/70 truncate max-w-[100px]" title={String(model)}>
          · {model}
        </span>
      ) : null}
    </div>
  );
}

/** Group sources by entity type and build one table per type with fields as columns. */
type GroupedRecords = { fileType: string; records: Record<string, unknown>[] };
function groupSourcesByEntityType(sources: ChatSource[]): GroupedRecords[] {
  const byType = new Map<string, Record<string, unknown>[]>();
  for (const src of sources) {
    const parsed = parseStructuredContent(src.content);
    if (!parsed) continue;
    const fileType = ((src.metadata?.file_type ?? src.metadata?.fileType) as string) ?? 'record';
    const key = String(fileType).toLowerCase().trim();
    if (!byType.has(key)) byType.set(key, []);
    byType.get(key)!.push(parsed);
  }
  const result: GroupedRecords[] = [];
  for (const key of ENTITY_TABLE_ORDER) {
    const records = byType.get(key);
    if (records?.length) result.push({ fileType: key, records });
  }
  for (const [key, records] of byType) {
    if (!ENTITY_TABLE_ORDER.includes(key as (typeof ENTITY_TABLE_ORDER)[number]))
      result.push({ fileType: key, records });
  }
  return result;
}

/** Styling for table cell values (amounts, IDs, dates). */
function getTableValueClass(value: unknown): string {
  const s = value == null ? '' : String(value);
  const isAmount = /^-?\$?[\d,]+(\.\d{2})?/.test(s) || /^-?[\d,]+\.\d{2}$/.test(s);
  const isId = /^\d{10}$/.test(s) || /^CUST|^ACH|^WIRE|^CHK|^CRD|^DBT/i.test(s) || /^\(?CUST[\d)]+$/.test(s);
  const isDate = /^\d{4}-\d{2}-\d{2}/.test(s);
  if (isAmount) return 'text-amber-800 font-semibold bg-amber-50/80';
  if (isId) return 'text-indigo-800 font-mono text-xs bg-indigo-50/60';
  if (isDate) return 'text-emerald-800 bg-emerald-50/60';
  return 'text-gray-800';
}

/** Format Redis query time for display: sub-ms as μs, else ms. */
function formatRedisQueryTime(ms: number): string {
  if (ms < 1 && ms > 0) return `${(ms * 1000).toFixed(2)} μs`;
  return `${ms.toFixed(2)} ms`;
}

/** Results as proper tables: one per entity type, fields as columns, sticky header, highlighted cells. */
function ResultsTableFromSources({
  sources,
  queryTimeMs,
  embeddingTimeMs,
  vectorDbQueryTimeMs,
  fromCache,
  redisQueryTimeMs,
}: {
  sources: ChatSource[];
  queryTimeMs?: number;
  /** When set with vectorDbQueryTimeMs, shows breakdown so total latency is clear */
  embeddingTimeMs?: number;
  vectorDbQueryTimeMs?: number;
  /** True when response was served from Redis cache */
  fromCache?: boolean;
  /** Time to fetch from Redis in ms */
  redisQueryTimeMs?: number;
}) {
  const groups = useMemo(() => groupSourcesByEntityType(sources), [sources]);
  const showBreakdown = embeddingTimeMs != null && vectorDbQueryTimeMs != null;
  const displayMs = vectorDbQueryTimeMs ?? queryTimeMs;

  return (
    <div className="space-y-6">
      {/* Source + timing: Redis cache vs Vector DB */}
      {fromCache && redisQueryTimeMs != null ? (
        <div className="flex items-center gap-3 px-3 py-2.5 rounded-lg bg-gradient-to-r from-amber-50 to-orange-50 border border-amber-200/80 shadow-sm">
          <svg className="w-5 h-5 text-amber-600 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
          </svg>
          <div className="flex-1 min-w-0 flex items-center gap-2 flex-wrap">
            <span className="text-xs font-semibold text-amber-800 uppercase tracking-wider">Source:</span>
            <span className="inline-flex items-center px-2 py-0.5 rounded-md bg-amber-200/80 text-amber-900 font-medium text-xs">Redis cache</span>
            <span className="text-sm font-mono text-amber-700 tabular-nums">· query time {formatRedisQueryTime(redisQueryTimeMs)}</span>
          </div>
        </div>
      ) : (displayMs != null || showBreakdown) ? (
        <div className="flex items-center gap-3 px-3 py-2.5 rounded-lg bg-gradient-to-r from-brand-50 to-brand-100/80 border border-brand-200/70 shadow-sm">
          <svg className="w-5 h-5 text-brand-600 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap mb-0.5">
              <span className="text-xs font-semibold text-brand-800 uppercase tracking-wider">Source:</span>
              <span className="inline-flex items-center px-2 py-0.5 rounded-md bg-brand-200/80 text-brand-900 font-medium text-xs">Vector DB</span>
            </div>
            {showBreakdown ? (
              <p className="text-sm font-mono text-brand-700 tabular-nums">
                Embedding: {embeddingTimeMs} ms · Vector search: {vectorDbQueryTimeMs} ms
                {queryTimeMs != null && (
                  <span className="text-brand-600 font-normal ml-1">(total: {queryTimeMs} ms)</span>
                )}
              </p>
            ) : (
              <p className="text-xl font-bold text-brand-700 font-mono tabular-nums">{displayMs} ms</p>
            )}
          </div>
          {!showBreakdown && <span className="text-xs text-brand-600 ml-auto">Time to fetch results</span>}
        </div>
      ) : null}

      {groups.length === 0 && (
        <p className="text-sm text-surface-500 py-2">No structured records to display in table form.</p>
      )}

      {groups.map(({ fileType, records }) => {
        const order = getColumnOrder(fileType);
        const allKeys = new Set<string>();
        for (const r of records) Object.keys(r).forEach((k) => allKeys.add(k));
        const columns = [...order.filter((k) => allKeys.has(k)), ...[...allKeys].filter((k) => !order.includes(k))];
        const sectionTitle = getSectionTitle(fileType);

        return (
          <div key={fileType} className="rounded-xl border border-surface-200 bg-white shadow-sm overflow-hidden">
            <div className="px-4 py-2.5 bg-surface-100 border-b border-surface-200">
              <h3 className="text-sm font-semibold text-surface-800">{sectionTitle}</h3>
              <p className="text-xs text-surface-500 mt-0.5">{records.length} record{records.length !== 1 ? 's' : ''}</p>
            </div>
            <div className="overflow-x-auto overflow-y-auto max-h-[420px]">
              <table className="w-full border-collapse text-sm min-w-[600px]">
                <thead className="sticky top-0 z-10 bg-surface-100 border-b-2 border-surface-200 shadow-sm">
                  <tr>
                    {columns.map((col) => (
                      <th
                        key={col}
                        className="px-3 py-2.5 text-left text-xs font-semibold text-surface-600 uppercase tracking-wider whitespace-nowrap"
                      >
                        {FIELD_LABELS[col] ?? col.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {records.map((record, ri) => (
                    <tr
                      key={ri}
                      className={ri % 2 === 0 ? 'bg-surface-50/70 hover:bg-surface-100/80' : 'bg-white hover:bg-surface-50/80'}
                    >
                      {columns.map((col) => {
                        const value = record[col];
                        const display = value == null || value === '' ? '—' : String(value);
                        return (
                          <td
                            key={col}
                            className={`px-3 py-2 align-top border-b border-surface-100 ${getTableValueClass(value)}`}
                          >
                            {display}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Array<{
    role: 'user' | 'assistant';
    content: string;
    timestamp?: string;
    sources?: ChatSource[];
    queryTimeMs?: number;
    /** Actual vector DB (Milvus) search time in ms when provided by backend */
    vectorDbQueryTimeMs?: number;
    /** Time to embed the query (ms). Explains total latency when vector DB is fast. */
    embeddingTimeMs?: number;
    embeddingDims?: number;
    embeddingModel?: string;
    debug?: ChatDebug;
    /** True when response was served from Redis cache */
    fromCache?: boolean;
    /** Time to fetch from Redis in ms (when fromCache is true) */
    redisQueryTimeMs?: number;
  }>>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [_error, setError] = useState<string | null>(null);
  const [debugOn, setDebugOn] = useState(true);
  const [selectedSemanticModel, setSelectedSemanticModel] = useState<string>(() => {
    try {
      const s = localStorage.getItem(SEMANTIC_MODEL_STORAGE_KEY);
      if (s && MODEL_DISPLAY_NAMES[s]) return s;
    } catch {
      /* ignore */
    }
    return FINANCE_MODEL;
  });
  const [exampleSeed, setExampleSeed] = useState(() => Math.floor(Math.random() * 1e6));
  const [semanticPreview, setSemanticPreview] = useState<{
    query: string;
    models: Record<string, SemanticPreviewModelResult>;
    loading: boolean;
    semanticAnnotationTimeMs?: number;
  } | null>(null);
  /** Per-turn semantic annotation for sent queries (shown first in left pane, then vector DB results). */
  const [semanticPreviewByKey, setSemanticPreviewByKey] = useState<
    Record<string, { query: string; models: Record<string, SemanticPreviewModelResult>; loading: boolean; semanticAnnotationTimeMs?: number }>
  >({});
  const previewTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const leftScrollRef = useRef<HTMLDivElement>(null);
  const middleScrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const sendingRef = useRef(false); // Prevent duplicate sends
  const pipesContainerRef = useRef<HTMLDivElement>(null);
  const leftPairRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const rightPairRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const [pipePaths, setPipePaths] = useState<Array<{ key: string; d: string; stroke: string }>>([]);
  const [pipesSvgSize, setPipesSvgSize] = useState<{ w: number; h: number } | null>(null);

  const usedInSession = useMemo(() => {
    const set = new Set<string>();
    messages.forEach((m) => {
      if (m.role === 'user' && typeof m.content === 'string' && m.content.trim()) set.add(m.content.trim());
    });
    return set;
  }, [messages]);

  const suggestionsByCategory = useMemo(
    () => groupByCategory(EXAMPLE_POOL, exampleSeed),
    [exampleSeed]
  );

  // Persist selected semantic model to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(SEMANTIC_MODEL_STORAGE_KEY, selectedSemanticModel);
    } catch {
      /* ignore */
    }
  }, [selectedSemanticModel]);

  // Newest-first layout: scroll to top when messages or loading change so latest is in view
  useEffect(() => {
    leftScrollRef.current && (leftScrollRef.current.scrollTop = 0);
    middleScrollRef.current && (middleScrollRef.current.scrollTop = 0);
  }, [messages, loading]);

  // Real-time semantic preview (sentence tokenization + tagging) when debug on and input changes (debounced)
  useEffect(() => {
    if (previewTimeoutRef.current) {
      clearTimeout(previewTimeoutRef.current);
    }

    if (!debugOn || !input.trim()) {
      setSemanticPreview(null);
      return;
    }

    const query = input.trim();
    setSemanticPreview({ query, models: {}, loading: true });

    previewTimeoutRef.current = setTimeout(async () => {
      try {
        if (selectedSemanticModel === 'all') {
          const data = await previewSemanticAnalysisMulti(query);
          if (data.success && data.models) {
            setSemanticPreview({
              query: data.query,
              models: data.models,
              loading: false,
              semanticAnnotationTimeMs: data.semantic_annotation_time_ms,
            });
          } else {
            setSemanticPreview(null);
          }
        } else {
          const data = await previewSemanticAnalysis(query, selectedSemanticModel);
          if (data.success) {
            setSemanticPreview({
              query: data.query,
              models: {
                [data.model]: {
                  tags: data.tags ?? [],
                  highlighted_segments: data.highlighted_segments ?? [],
                  error: data.error,
                },
              },
              loading: false,
              semanticAnnotationTimeMs: data.semantic_annotation_time_ms,
            });
          } else {
            setSemanticPreview(null);
          }
        }
      } catch (err) {
        getLogger('Chat').warn('Semantic preview error', { error: err instanceof Error ? err.message : String(err) });
        setSemanticPreview(null);
      }
    }, 500);

    return () => {
      if (previewTimeoutRef.current) {
        clearTimeout(previewTimeoutRef.current);
      }
    };
  }, [input, debugOn, selectedSemanticModel]);

  const send = useCallback(async () => {
    const msg = input.trim();
    
    // Prevent duplicate calls using ref guard
    if (!msg || loading || sendingRef.current) return;
    
    // Set sending flag immediately to prevent race conditions
    sendingRef.current = true;
    setInput('');
    setError(null);
    setLoading(true);
    
    const userTimestamp = new Date().toISOString();
    const pairKey = `${msg}-${userTimestamp}`;
    setMessages((m) => [...m, { role: 'user', content: msg, timestamp: userTimestamp }]);
    setSemanticPreviewByKey((prev) => ({ ...prev, [pairKey]: { query: msg, models: {}, loading: true } }));

    // Fire semantic annotation immediately (don't await) so it shows as soon as it returns; chat runs in parallel
    if (debugOn) {
      if (selectedSemanticModel === 'all') {
        previewSemanticAnalysisMulti(msg)
          .then((data) => {
            if (data.success && data.models) {
              setSemanticPreviewByKey((prev) => ({
                ...prev,
                [pairKey]: {
                  query: data.query,
                  models: data.models,
                  loading: false,
                  semanticAnnotationTimeMs: data.semantic_annotation_time_ms,
                },
              }));
            } else {
              setSemanticPreviewByKey((prev) => ({ ...prev, [pairKey]: { query: msg, models: {}, loading: false } }));
            }
          })
          .catch(() => setSemanticPreviewByKey((prev) => ({ ...prev, [pairKey]: { query: msg, models: {}, loading: false } })));
      } else {
        previewSemanticAnalysis(msg, selectedSemanticModel)
          .then((data) => {
            if (data.success) {
              setSemanticPreviewByKey((prev) => ({
                ...prev,
                [pairKey]: {
                  query: data.query,
                  models: {
                    [data.model]: {
                      tags: data.tags ?? [],
                      highlighted_segments: data.highlighted_segments ?? [],
                      error: data.error,
                    },
                  },
                  loading: false,
                  semanticAnnotationTimeMs: data.semantic_annotation_time_ms,
                },
              }));
            } else {
              setSemanticPreviewByKey((prev) => ({ ...prev, [pairKey]: { query: msg, models: {}, loading: false } }));
            }
          })
          .catch(() => setSemanticPreviewByKey((prev) => ({ ...prev, [pairKey]: { query: msg, models: {}, loading: false } })));
      }
    } else {
      setSemanticPreviewByKey((prev) => ({ ...prev, [pairKey]: { query: msg, models: {}, loading: false } }));
    }

    try {
      const res = await chat(msg, 15, debugOn, FINANCE_MODEL);
      const assistantTimestamp = new Date().toISOString();
      const embeddingModel = res.embedding_model ?? FINANCE_MODEL;
      const debugWithModel = res.debug ? { ...res.debug, embedding_model: res.debug.embedding_model ?? embeddingModel } : undefined;
      setMessages((m) => [
        ...m,
        {
          role: 'assistant',
          content: res.answer,
          timestamp: assistantTimestamp,
          sources: res.sources ?? [],
          queryTimeMs: res.query_time_ms,
          vectorDbQueryTimeMs: res.vector_db_query_time_ms,
          embeddingTimeMs: res.embedding_time_ms,
          embeddingDims: res.embedding_dims,
          embeddingModel,
          debug: debugWithModel,
          fromCache: res.from_cache,
          redisQueryTimeMs: res.redis_query_time_ms,
        },
      ]);
    } catch (e) {
      const err = e instanceof Error ? e.message : 'Request failed';
      // Don't set error state - show error as assistant message in center pane only
      const errorTimestamp = new Date().toISOString();
      setMessages((m) => [...m, { 
        role: 'assistant', 
        content: `Error: ${err}`, 
        timestamp: errorTimestamp 
      }]);
    } finally {
      setLoading(false);
      sendingRef.current = false; // Reset flag
    }
  }, [input, loading, debugOn, selectedSemanticModel]);

  const fillExample = useCallback((q: string) => {
    // Auto-send when suggestion is selected for better UX
    if (q.trim() && !loading && !sendingRef.current) {
      setInput(''); // Clear input immediately to prevent duplicate display
      setError(null);
      setLoading(true);
      sendingRef.current = true;
      
      const userTimestamp = new Date().toISOString();
      const pairKey = `${q.trim()}-${userTimestamp}`;
      setMessages((m) => [...m, { role: 'user', content: q.trim(), timestamp: userTimestamp }]);
      setSemanticPreviewByKey((prev) => ({ ...prev, [pairKey]: { query: q.trim(), models: {}, loading: true } }));

      (async () => {
        if (debugOn) {
          if (selectedSemanticModel === 'all') {
            previewSemanticAnalysisMulti(q.trim())
              .then((data) => {
                if (data.success && data.models) {
                  setSemanticPreviewByKey((prev) => ({
                    ...prev,
                    [pairKey]: {
                      query: data.query,
                      models: data.models,
                      loading: false,
                      semanticAnnotationTimeMs: data.semantic_annotation_time_ms,
                    },
                  }));
                } else {
                  setSemanticPreviewByKey((prev) => ({ ...prev, [pairKey]: { query: q.trim(), models: {}, loading: false } }));
                }
              })
              .catch(() => setSemanticPreviewByKey((prev) => ({ ...prev, [pairKey]: { query: q.trim(), models: {}, loading: false } })));
          } else {
            previewSemanticAnalysis(q.trim(), selectedSemanticModel)
              .then((data) => {
                if (data.success) {
                  setSemanticPreviewByKey((prev) => ({
                    ...prev,
                    [pairKey]: {
                      query: data.query,
                      models: {
                        [data.model]: {
                          tags: data.tags ?? [],
                          highlighted_segments: data.highlighted_segments ?? [],
                          error: data.error,
                        },
                      },
                      loading: false,
                      semanticAnnotationTimeMs: data.semantic_annotation_time_ms,
                    },
                  }));
                } else {
                  setSemanticPreviewByKey((prev) => ({ ...prev, [pairKey]: { query: q.trim(), models: {}, loading: false } }));
                }
              })
              .catch(() => setSemanticPreviewByKey((prev) => ({ ...prev, [pairKey]: { query: q.trim(), models: {}, loading: false } })));
          }
        } else {
          setSemanticPreviewByKey((prev) => ({ ...prev, [pairKey]: { query: q.trim(), models: {}, loading: false } }));
        }
        const res = await chat(q.trim(), 15, debugOn, FINANCE_MODEL);
        const assistantTimestamp = new Date().toISOString();
        const embeddingModel = res.embedding_model ?? FINANCE_MODEL;
        const debugWithModel = res.debug ? { ...res.debug, embedding_model: res.debug.embedding_model ?? embeddingModel } : undefined;
        setMessages((m) => [
          ...m,
          {
            role: 'assistant',
            content: res.answer ?? '',
            timestamp: assistantTimestamp,
            sources: res.sources ?? [],
            queryTimeMs: res.query_time_ms,
            vectorDbQueryTimeMs: res.vector_db_query_time_ms,
            embeddingTimeMs: res.embedding_time_ms,
            embeddingDims: res.embedding_dims,
            embeddingModel,
            debug: debugWithModel,
            fromCache: res.from_cache,
            redisQueryTimeMs: res.redis_query_time_ms,
          },
        ]);
      })()
        .catch((e) => {
          const err = e instanceof Error ? e.message : 'Request failed';
          const errorTimestamp = new Date().toISOString();
          setMessages((m) => [...m, { role: 'assistant', content: `Error: ${err}`, timestamp: errorTimestamp }]);
        })
        .finally(() => {
          setLoading(false);
          sendingRef.current = false;
        });
    }
  }, [loading, debugOn, selectedSemanticModel]);

  /** Edit a past query: put it in the input and focus so user can modify and send. */
  const editQuery = (content: string) => {
    setInput(content);
    setError(null);
    inputRef.current?.focus();
  };

  // Pair user + assistant for aligned left (query + debug) / middle (results) layout
  // Use a Map to track unique user messages by content+timestamp to prevent duplicates
  const pairs: { user: typeof messages[0]; assistant?: typeof messages[0]; key: string }[] = [];
  const seenUserKeys = new Set<string>();
  
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    if (msg.role === 'user') {
      // Create a unique key for this user message
      const userKey = `${msg.content}-${msg.timestamp}`;
      
      // Skip if we've already seen this exact user message
      if (seenUserKeys.has(userKey)) continue;
      seenUserKeys.add(userKey);
      
      // Find the corresponding assistant message (next message if it's an assistant)
      const assistant = i + 1 < messages.length && messages[i + 1]?.role === 'assistant' 
        ? messages[i + 1] 
        : undefined;
      
      pairs.push({ user: msg, assistant, key: userKey });
    }
  }
  
  // Add loading state for the last user message if loading
  if (loading && messages.length > 0 && messages[messages.length - 1]?.role === 'user') {
    const lastUser = messages[messages.length - 1];
    const lastKey = `${lastUser.content}-${lastUser.timestamp}`;
    if (!seenUserKeys.has(lastKey)) {
      pairs.push({ user: lastUser, assistant: undefined, key: lastKey });
    }
  }
  
  // Newest first: reverse so latest chat and results appear at top
  const pairsNewestFirst = [...pairs].reverse();

  // Measure and draw pipe lines from left pane blocks to right pane blocks (same color = same pair)
  const updatePipePaths = useCallback(() => {
    const container = pipesContainerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const paths: Array<{ key: string; d: string; stroke: string }> = [];
    pairsNewestFirst.forEach(({ key }, idx) => {
      const leftEl = leftPairRefs.current[key];
      const rightEl = rightPairRefs.current[key];
      if (!leftEl || !rightEl) return;
      const leftR = leftEl.getBoundingClientRect();
      const rightR = rightEl.getBoundingClientRect();
      const x1 = leftR.right - rect.left;
      const y1 = leftR.top + leftR.height / 2 - rect.top;
      const x2 = rightR.left - rect.left;
      const y2 = rightR.top + rightR.height / 2 - rect.top;
      const creationIndex = getCreationIndex(idx, pairsNewestFirst.length);
      const accent = getPairAccent(creationIndex);
      const stroke = accent.stroke ?? '#94a3b8';
      const midX = (x1 + x2) / 2;
      const d = `M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2} ${y2}`;
      paths.push({ key, d, stroke });
    });
    setPipePaths(paths);
    setPipesSvgSize({ w: rect.width, h: rect.height });
  }, [pairsNewestFirst]);

  // Depend only on stable values so we don't re-run every render (pairsNewestFirst is a new array each time).
  const pairCount = pairsNewestFirst.length;
  useLayoutEffect(() => {
    updatePipePaths();
    const container = pipesContainerRef.current;
    const leftScroll = leftScrollRef.current;
    const middleScroll = middleScrollRef.current;
    const onScrollOrResize = () => {
      requestAnimationFrame(updatePipePaths);
    };
    leftScroll?.addEventListener('scroll', onScrollOrResize, { passive: true });
    middleScroll?.addEventListener('scroll', onScrollOrResize, { passive: true });
    const ro = container ? new ResizeObserver(onScrollOrResize) : null;
    if (container && ro) ro.observe(container);
    return () => {
      leftScroll?.removeEventListener('scroll', onScrollOrResize);
      middleScroll?.removeEventListener('scroll', onScrollOrResize);
      ro?.disconnect();
    };
  }, [pairCount, loading]); // updatePipePaths reads refs and pairsNewestFirst from closure when effect runs

  return (
    <div ref={pipesContainerRef} className="relative flex h-[calc(100vh-6rem)] min-h-[320px] gap-2 sm:gap-4 w-full max-w-[100vw] min-w-0">
      {/* Pipes overlay: same-color lines linking left query blocks to right result blocks */}
      {pipesSvgSize && pipePaths.length > 0 && (
        <div className="absolute inset-0 pointer-events-none z-10" aria-hidden>
          <svg
            width={pipesSvgSize.w}
            height={pipesSvgSize.h}
            className="absolute left-0 top-0 overflow-visible"
            style={{ minWidth: '100%', minHeight: '100%' }}
          >
            {pipePaths.map(({ key, d, stroke }) => (
              <path
                key={key}
                d={d}
                fill="none"
                stroke={stroke}
                strokeWidth="2"
                strokeLinecap="round"
                strokeOpacity="0.6"
              />
            ))}
          </svg>
        </div>
      )}
      {/* Left column: chat input at top, then Debug + user messages + semantic understanding */}
      <div className="relative z-20 w-80 lg:w-96 shrink-0 flex flex-col rounded-xl bg-white border border-surface-200 shadow-soft overflow-hidden min-h-0">
        <div className="flex-shrink-0 flex items-center justify-between gap-2 px-3 py-2 border-b border-surface-200 bg-surface-50">
          <div className="flex items-center gap-3 flex-wrap">
            <DebugToggle on={debugOn} onChange={setDebugOn} />
            {debugOn && (
              <label className="flex items-center gap-1.5">
                <span className="text-xs text-gray-500 whitespace-nowrap">Model:</span>
                <select
                  value={selectedSemanticModel}
                  onChange={(e) => setSelectedSemanticModel(e.target.value)}
                  className="text-xs font-medium text-gray-700 bg-white border border-surface-200 rounded-lg px-2 py-1.5 focus:outline-none focus:ring-2 focus:ring-brand-400/50 focus:border-brand-400 min-w-0 max-w-[180px]"
                  title="Model for sentence tokenization and semantic tagging"
                  aria-label="Model for sentence tokenization and semantic tagging"
                >
                  {SEMANTIC_MODEL_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </label>
            )}
          </div>
        </div>
        <div className="flex-shrink-0 border-b border-surface-200 bg-gradient-to-b from-surface-50 to-white px-3 py-3 space-y-2">
          <div className="relative flex gap-2 rounded-xl bg-white/90 backdrop-blur-sm border border-surface-200/80 shadow-soft p-2 focus-within:border-brand-400/80 focus-within:shadow-glow focus-within:ring-2 focus-within:ring-brand-400/20 transition-all duration-200">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  e.stopPropagation();
                  if (!loading && input.trim()) {
                    send();
                  }
                }
              }}
              placeholder="Ask about customers, transactions, states, or amounts…"
              rows={2}
              maxLength={2000}
              disabled={loading}
              className="flex-1 min-h-[56px] max-h-36 resize-y bg-transparent rounded-lg px-3 py-2 text-surface-900 placeholder-surface-400 border-0 focus:outline-none focus:ring-0 disabled:opacity-50 text-sm leading-relaxed"
              aria-label="Search query"
            />
            <button
              type="button"
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                if (!loading && input.trim()) {
                  send();
                }
              }}
              disabled={loading || !input.trim()}
              className="flex-shrink-0 h-12 w-12 rounded-xl bg-brand-500 hover:bg-brand-600 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center text-white shadow-md hover:shadow-lg hover:scale-[1.02] active:scale-[0.98] transition-all duration-200"
              title="Send"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            </button>
          </div>
          
          {/* Real-time semantic preview (sentence tokenization + tagging) while typing */}
          {debugOn && input.trim() && semanticPreview && (
            <SemanticPreviewBlock
              query={semanticPreview.query}
              models={semanticPreview.models}
              loading={semanticPreview.loading}
              title="Semantic Preview"
              modelLabel={selectedSemanticModel === 'all' ? 'All models' : (MODEL_DISPLAY_NAMES[selectedSemanticModel] ?? selectedSemanticModel)}
              modelId={selectedSemanticModel}
              semanticAnnotationTimeMs={semanticPreview.semanticAnnotationTimeMs}
            />
          )}
        </div>
        <div ref={leftScrollRef} className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden px-3 py-3">
          {pairsNewestFirst.map(({ user, assistant, key }, idx) => {
            if (!user || user.role !== 'user') return null;
            const creationIndex = getCreationIndex(idx, pairsNewestFirst.length);
            const accent = getPairAccent(creationIndex);
            const pairNumber = creationIndex + 1;
            return (
              <div
                key={key || `user-${idx}-${user.timestamp}`}
                className={`space-y-3 mb-4 pl-2 -ml-2 rounded-r-lg border-l-4 ${accent.border} ${accent.bg} py-2 pr-2`}
                title={`Query ${pairNumber} — same number and color link to the result in the center`}
              >
                {/* Pair link: number badge + bar so left query is clearly linked to center result */}
                <div className="flex items-center gap-2 flex-wrap">
                  <span
                    className={`inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[10px] font-bold text-white ${accent.bar}`}
                    aria-label={`Query ${pairNumber}`}
                    title={`Query ${pairNumber} → Result ${pairNumber} (center)`}
                  >
                    {pairNumber}
                  </span>
                  <span className={`inline-flex h-5 w-1 rounded-full ${accent.bar} shrink-0`} aria-hidden />
                  {user.timestamp && (
                    <p className="text-[10px] font-medium text-surface-500 uppercase tracking-wider" title={user.timestamp}>
                      {formatMessageTimeFull(user.timestamp)}
                    </p>
                  )}
                </div>
                <div className="group rounded-xl px-3 py-2.5 bg-white/90 text-brand-900 border border-brand-200/80 text-sm shadow-sm">
                  <p className="whitespace-pre-wrap">{user.content ?? ''}</p>
                  <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
                    {user.timestamp && (
                      <p className="text-[10px] text-brand-600/80" title={formatMessageTimeFull(user.timestamp)}>
                        Sent {formatRelativeTime(user.timestamp)}
                      </p>
                    )}
                    <button
                      type="button"
                      onClick={() => editQuery(user.content ?? '')}
                      disabled={loading}
                      className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1 text-[10px] font-medium text-brand-600 hover:text-brand-700 hover:bg-brand-100/80 border border-transparent hover:border-brand-200/60 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                      title="Edit and send a new query"
                    >
                      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                      </svg>
                      Edit & run again
                    </button>
                  </div>
                </div>
                {/* Merged: semantic annotation (dropdown model) + Debug Query Understanding; or just semantic annotation while waiting */}
                {debugOn && assistant?.debug ? (
                  <DebugPanel
                    debug={assistant.debug}
                    queriedAt={assistant?.timestamp}
                    selectedModel={selectedSemanticModel}
                    semanticPreview={semanticPreviewByKey[key]}
                  />
                ) : semanticPreviewByKey[key] ? (
                  <SemanticPreviewBlock
                    query={semanticPreviewByKey[key].query}
                    models={semanticPreviewByKey[key].models}
                    loading={semanticPreviewByKey[key].loading}
                    title="Semantic annotation"
                    modelLabel={selectedSemanticModel === 'all' ? 'All models' : (MODEL_DISPLAY_NAMES[selectedSemanticModel] ?? selectedSemanticModel)}
                    modelId={selectedSemanticModel}
                    semanticAnnotationTimeMs={semanticPreviewByKey[key].semanticAnnotationTimeMs}
                  />
                ) : null}
              </div>
            );
          })}
          {/* Don't show error state here - errors are shown as assistant messages in center pane */}
        </div>
      </div>

      {/* Middle column: results (assistant answers) – aligned with left */}
      <div className="relative z-20 flex-1 min-w-0 flex flex-col rounded-xl bg-white border border-surface-200 shadow-soft overflow-hidden min-h-0">
        <div ref={middleScrollRef} className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden px-3 sm:px-4 py-3">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center text-center py-8 px-4 min-h-full">
              <h1 className="text-xl font-semibold text-gray-900 mb-2">Semantic Search for your Data</h1>
              <p className="text-gray-600 text-sm max-w-md mb-2">
                Ask about customers, transactions, locations (state, ZIP), and amounts. Answers are built from your
                Milvus vector DB (Customer, Transaction, Address, State, Zip).
              </p>
              <p className="text-gray-500 text-xs max-w-md mb-6">
                Use the chat on the left to ask a question. Debug shows how your query is understood.
              </p>
              <p className="text-[10px] text-surface-400 text-center">
                Powered by openInt.ai · © {new Date().getFullYear()} All rights reserved
              </p>
            </div>
          )}
          {pairsNewestFirst.map(({ user, assistant, key }, idx) => {
            const creationIndex = getCreationIndex(idx, pairsNewestFirst.length);
            const accent = getPairAccent(creationIndex);
            const pairNumber = creationIndex + 1;
            if (!assistant || assistant.role !== 'assistant') {
              // Show loading for the newest pair (idx 0) so color and number match the left pane
              if (user && loading && idx === 0) {
                return (
                  <div ref={(el) => { rightPairRefs.current[key] = el; }} key={`loading-${key || idx}`} className={`mb-6 pl-2 -ml-2 rounded-r-lg border-l-4 ${accent.border} ${accent.bg} py-2 pr-2`} title={`Result ${pairNumber} — same number and color as the query on the left`}>
                    <div className="flex items-center gap-2 mb-2">
                      <span className={`inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[10px] font-bold text-white ${accent.bar}`} aria-label={`Result ${pairNumber}`} title={`Result ${pairNumber} ← Query ${pairNumber} (left)`}>
                        {pairNumber}
                      </span>
                      <span className={`inline-flex h-5 w-1 rounded-full ${accent.bar} shrink-0`} aria-hidden />
                    </div>
                    <div className="rounded-xl border border-surface-200 bg-surface-50 p-4 text-gray-500 text-sm">
                      Thinking…
                    </div>
                  </div>
                );
              }
              return null;
            }

            return (
              <div
                ref={(el) => { rightPairRefs.current[key] = el; }}
                key={`assistant-${key || idx}-${assistant.timestamp}`}
                className={`mb-6 pl-2 -ml-2 rounded-r-lg border-l-4 ${accent.border} ${accent.bg} py-2 pr-2`}
                title={`Result ${pairNumber} — same number and color as the query on the left`}
              >
                {/* Pair link: number badge + bar so center result is clearly linked to left query */}
                <div className="flex items-center gap-2 mb-2">
                  <span
                    className={`inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[10px] font-bold text-white ${accent.bar}`}
                    aria-label={`Result ${pairNumber}`}
                    title={`Result ${pairNumber} ← Query ${pairNumber} (left)`}
                  >
                    {pairNumber}
                  </span>
                  <span className={`inline-flex h-5 w-1 rounded-full ${accent.bar} shrink-0`} aria-hidden />
                  {assistant.timestamp && (
                    <p className="text-[10px] font-medium text-surface-500 uppercase tracking-wider" title={assistant.timestamp}>
                      {formatMessageTimeFull(assistant.timestamp)}
                    </p>
                  )}
                </div>
                <div className="rounded-xl border border-surface-200 bg-white p-4 shadow-sm">
                  {/* Results: tables from sources (fields as columns) + query time, or fallback to answer text */}
                  {assistant.sources && assistant.sources.length > 0 ? (
                    <>
                      <ResultsTableFromSources
                        sources={assistant.sources}
                        queryTimeMs={assistant.queryTimeMs}
                        embeddingTimeMs={assistant.embeddingTimeMs}
                        vectorDbQueryTimeMs={assistant.vectorDbQueryTimeMs}
                        fromCache={assistant.fromCache}
                        redisQueryTimeMs={assistant.redisQueryTimeMs}
                      />
                      {(assistant.timestamp && (
                        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 mt-3 pt-3 border-t border-surface-100 text-[10px] text-surface-500">
                          <span title={formatMessageTimeFull(assistant.timestamp)}>
                            Generated {formatMessageTimeFull(assistant.timestamp)}
                          </span>
                        </div>
                      ))}
                      {assistant.embeddingDims != null && (
                        <EmbeddingVisualization dims={assistant.embeddingDims} model={assistant.embeddingModel} />
                      )}
                    </>
                  ) : (
                    <>
                      {(assistant.fromCache && assistant.redisQueryTimeMs != null) ? (
                        <div className="mb-4 pb-3 border-b border-surface-200">
                          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gradient-to-r from-amber-50 to-orange-50 border border-amber-200/60">
                            <svg className="w-4 h-4 text-amber-600 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                            </svg>
                            <span className="text-xs font-semibold text-amber-800 uppercase tracking-wider">Source:</span>
                            <span className="inline-flex items-center px-2 py-0.5 rounded-md bg-amber-200/80 text-amber-900 font-medium text-xs">Redis cache</span>
                            <span className="text-sm font-mono text-amber-600">· query time {formatRedisQueryTime(assistant.redisQueryTimeMs)}</span>
                          </div>
                        </div>
                      ) : (assistant.vectorDbQueryTimeMs != null || assistant.embeddingTimeMs != null || assistant.queryTimeMs != null) ? (
                        <div className="mb-4 pb-3 border-b border-surface-200">
                          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gradient-to-r from-brand-50 to-brand-100/50 border border-brand-200/60">
                            <svg className="w-4 h-4 text-brand-600 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                            </svg>
                            <span className="text-xs font-semibold text-brand-700 uppercase tracking-wider">Source:</span>
                            <span className="inline-flex items-center px-2 py-0.5 rounded-md bg-brand-200/80 text-brand-900 font-medium text-xs">Vector DB</span>
                            {assistant.embeddingTimeMs != null && assistant.vectorDbQueryTimeMs != null ? (
                              <span className="text-sm font-mono text-brand-600">
                                · Embedding: {assistant.embeddingTimeMs} ms · Vector search: {assistant.vectorDbQueryTimeMs} ms
                                {assistant.queryTimeMs != null && ` (total: ${assistant.queryTimeMs} ms)`}
                              </span>
                            ) : (
                              <span className="text-lg font-bold text-brand-600 font-mono">{assistant.vectorDbQueryTimeMs ?? assistant.queryTimeMs} ms</span>
                            )}
                          </div>
                        </div>
                      ) : null}
                      {assistant.timestamp && (
                        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 mb-3 text-[10px] text-surface-500">
                          <span title={formatMessageTimeFull(assistant.timestamp)}>
                            Generated {formatMessageTimeFull(assistant.timestamp)}
                          </span>
                        </div>
                      )}
                      <div className="text-gray-800 leading-relaxed">
                        <AnswerRenderer text={assistant.content ?? ''} />
                      </div>
                      {assistant.embeddingDims != null && (
                        <EmbeddingVisualization dims={assistant.embeddingDims} model={assistant.embeddingModel} />
                      )}
                    </>
                  )}
                </div>
              </div>
            );
          })}
          {messages.length > 0 && (
            <div className="mt-8 pt-4 border-t border-surface-200">
              <p className="text-[10px] text-surface-400 text-center">
                Powered by openInt.ai · © {new Date().getFullYear()} All rights reserved
              </p>
            </div>
          )}
        </div>
      </div>

      <SuggestionsPanel
        byCategory={suggestionsByCategory}
        usedInSession={usedInSession}
        onSelect={fillExample}
        onMore={() => setExampleSeed((s) => s + 1)}
        loading={loading}
      />
    </div>
  );
}

/**
 * Semantic Annotation Comparison
 * Single textbox at top; all 3 models output side-by-side with consistency vs difference highlighting.
 * Fourth panel: suggestions (same as Chat).
 */

import { useState, useCallback, useMemo } from 'react';
import {
  previewSemanticAnalysisMulti,
  getLuckySuggestion,
  type SemanticPreviewMulti,
  type SemanticPreviewModelResult,
  type SemanticPreviewSegment,
} from '../api';
import { getModelDisplayName, getModelUrl } from '../utils/modelMeta';

type SuggestionItem = { query: string; category: string };

/** Full pool: same as Chat (Customer, Transactions, Analytics & Insights). */
const EXAMPLE_POOL: SuggestionItem[] = [
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
  { query: 'Pending ACH payments', category: 'ACH' },
  { query: 'Top 15 highest ACH debits', category: 'ACH' },
  { query: 'ACH transactions over $5,000', category: 'ACH' },
  { query: 'ACH debits by customer', category: 'ACH' },
  { query: 'Failed or reversed ACH transactions', category: 'ACH' },
  { query: 'Wire transfers over $10,000', category: 'Wire' },
  { query: 'Top 10 largest wire transfers', category: 'Wire' },
  { query: 'Top 10 international wire transfers by amount', category: 'Wire' },
  { query: 'States with highest share of international wires', category: 'Wire' },
  { query: 'Pending wire transactions', category: 'Wire' },
  { query: 'Credit card disputes', category: 'Credit Card' },
  { query: 'Top 10 credit card charges by amount', category: 'Credit Card' },
  { query: 'Where do we have the most disputed credit card transactions?', category: 'Credit Card' },
  { query: 'Credit card transactions over $1,000', category: 'Credit Card' },
  { query: 'Disputed credit card transactions by state', category: 'Credit Card' },
  { query: 'Check payments over 2000', category: 'Check' },
  { query: 'Top 20 largest check payments', category: 'Check' },
  { query: 'Which regions have the most check usage?', category: 'Check' },
  { query: 'Pending check payments', category: 'Check' },
  { query: 'Check transactions between 1000 and 5000 USD', category: 'Check' },
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

/** Fourth panel: suggestions grouped by category (same as Chat). */
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
    <aside className="w-72 lg:w-80 shrink-0 hidden md:flex flex-col min-h-0">
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

const MODEL_ORDER = [
  'mukaj/fin-mpnet-base',
  'ProsusAI/finbert',
  'sentence-transformers/all-mpnet-base-v2',
] as const;

const MODEL_DISPLAY_NAMES: Record<string, string> = {
  'mukaj/fin-mpnet-base': 'Finance MPNet',
  'ProsusAI/finbert': 'FinBERT',
  'sentence-transformers/all-mpnet-base-v2': 'General MPNet',
};

type ModelSourceKind = 'huggingface' | 'microsoft' | 'github';
const MODEL_SOURCES: Record<string, { name: string; kind: ModelSourceKind }> = {
  'mukaj/fin-mpnet-base': { name: 'Hugging Face', kind: 'huggingface' },
  'ProsusAI/finbert': { name: 'Hugging Face', kind: 'huggingface' },
  'sentence-transformers/all-mpnet-base-v2': { name: 'Hugging Face', kind: 'huggingface' },
};

function getModelSource(modelId: string): { name: string; kind: ModelSourceKind } | null {
  if (MODEL_SOURCES[modelId]) return MODEL_SOURCES[modelId];
  if (modelId.startsWith('mukaj/') || modelId.startsWith('ProsusAI/') || modelId.startsWith('sentence-transformers/')) {
    return { name: 'Hugging Face', kind: 'huggingface' };
  }
  if (modelId.startsWith('microsoft/') || modelId.toLowerCase().includes('microsoft')) {
    return { name: 'Microsoft', kind: 'microsoft' };
  }
  if (modelId.includes('github.com') || modelId.toLowerCase().includes('github')) {
    return { name: 'GitHub', kind: 'github' };
  }
  return { name: 'Hugging Face', kind: 'huggingface' };
}

function SourceIcon({ kind, className }: { kind: ModelSourceKind; className?: string }) {
  const c = className ?? 'w-3.5 h-3.5';
  if (kind === 'huggingface') {
    return (
      <svg className={c} viewBox="0 0 24 24" fill="currentColor" aria-hidden>
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-5-9c.83 0 1.5-.67 1.5-1.5S7.83 8 7 8s-1.5.67-1.5 1.5S6.17 11 7 11zm10 0c.83 0 1.5-.67 1.5-1.5S17.83 8 17 8s-1.5.67-1.5 1.5.67 1.5 1.5 1.5zm-5 6c2.33 0 4.31-1.46 5.11-3.5H6.89c.8 2.04 2.78 3.5 5.11 3.5z" />
      </svg>
    );
  }
  if (kind === 'microsoft') {
    return (
      <svg className={c} viewBox="0 0 24 24" fill="currentColor" aria-hidden>
        <path d="M11.4 24H0V12.6h11.4V24zM24 24H12.6V12.6H24V24zM11.4 11.4H0V0h11.4v11.4zm12.6 0H12.6V0H24v11.4z" />
      </svg>
    );
  }
  if (kind === 'github') {
    return (
      <svg className={c} viewBox="0 0 24 24" fill="currentColor" aria-hidden>
        <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
      </svg>
    );
  }
  return null;
}

/** Per-model performance over the session (ranked by speed: 1st = fastest per query). */
interface ModelPerformance {
  first: number;
  second: number;
  third: number;
}

/** Update leaderboard from a compare result: rank models by semantic_annotation_time_ms (lower = better). */
function updateLeaderboard(
  prev: { queryCount: number; modelStats: Record<string, ModelPerformance> },
  res: SemanticPreviewMulti
): { queryCount: number; modelStats: Record<string, ModelPerformance> } {
  const models = res.models ?? {};
  const entries = Object.entries(models)
    .map(([id, m]) => ({ id, timeMs: m?.semantic_annotation_time_ms ?? Infinity }))
    .filter((e) => e.timeMs !== Infinity);
  if (entries.length < 2) return prev;
  entries.sort((a, b) => a.timeMs - b.timeMs);
  const modelStats = { ...prev.modelStats };
  for (let i = 0; i < entries.length; i++) {
    const id = entries[i].id;
    if (!modelStats[id]) modelStats[id] = { first: 0, second: 0, third: 0 };
    if (i === 0) modelStats[id].first += 1;
    else if (i === 1) modelStats[id].second += 1;
    else if (i === 2) modelStats[id].third += 1;
  }
  return { queryCount: prev.queryCount + 1, modelStats };
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
  model: 'bg-gradient-to-r from-slate-100 to-slate-50 text-slate-700 border-slate-300 shadow-sm',
  embedding_norm: 'bg-gradient-to-r from-sky-100 to-sky-50 text-sky-900 border-sky-300 shadow-sm',
  embedding_dim: 'bg-gradient-to-r from-cyan-100 to-cyan-50 text-cyan-900 border-cyan-300 shadow-sm',
  embedding_peak: 'bg-gradient-to-r from-indigo-100 to-indigo-50 text-indigo-900 border-indigo-300 shadow-sm',
  schema_field: 'bg-gradient-to-r from-violet-100 to-violet-50 text-violet-900 border-violet-400 shadow-sm',
  schema_dataset: 'bg-gradient-to-r from-violet-100 to-violet-50 text-violet-900 border-violet-400 shadow-sm',
};

/** Tag type is DataHub schema-related (shown in separate section). */
function isSchemaTag(type: string): boolean {
  const t = (type ?? '').toLowerCase();
  return t.includes('schema') || t.includes('dataset') || t === 'schema_field' || t === 'schema_dataset';
}

/** Tag represents a dataset/asset (label starts with "Dataset:"); value is asset name. */
function isAssetTag(tag: { type?: string; label?: string }): boolean {
  const label = (tag.label ?? '').trim();
  return label.startsWith('Dataset:') || (tag.type ?? '').toLowerCase() === 'schema_dataset';
}

function getSemanticColor(type: string): string {
  return SEMANTIC_COLORS[type] ?? 'bg-gradient-to-r from-gray-100 to-gray-50 text-gray-800 border-gray-300 shadow-sm';
}

/** Segment with start/end in query for alignment */
interface AlignedSegment {
  start: number;
  end: number;
  type: string;
  label: string;
  text: string;
}

function segmentsToAligned(query: string, segments: SemanticPreviewSegment[]): AlignedSegment[] {
  const out: AlignedSegment[] = [];
  let pos = 0;
  const q = query.toLowerCase();
  for (const seg of segments) {
    const text = seg.text ?? '';
    if (!text) continue;
    const idx = q.indexOf(text.toLowerCase(), pos);
    if (idx < 0) {
      pos = query.length;
      continue;
    }
    const start = idx;
    const end = idx + text.length;
    const type = (seg.tag_type ?? seg.tag?.type ?? 'text').toString();
    const label = (seg.label ?? seg.tag?.label ?? type).toString();
    out.push({ start, end, type, label, text });
    pos = end;
  }
  return out;
}

/** Partition query into non-overlapping ranges with per-model annotation (type). */
function buildRanges(
  query: string,
  modelSegments: Record<string, AlignedSegment[]>
): Array<{ start: number; end: number; types: Record<string, string> }> {
  const breaks = new Set<number>([0, query.length]);
  for (const segs of Object.values(modelSegments)) {
    for (const s of segs) {
      breaks.add(s.start);
      breaks.add(s.end);
    }
  }
  const sorted = Array.from(breaks).sort((a, b) => a - b);
  const ranges: Array<{ start: number; end: number; types: Record<string, string> }> = [];
  for (let i = 0; i < sorted.length - 1; i++) {
    const start = sorted[i];
    const end = sorted[i + 1];
    if (start >= end) continue;
    const types: Record<string, string> = {};
    for (const [modelId, segs] of Object.entries(modelSegments)) {
      const seg = segs.find((s) => s.start < end && s.end > start);
      if (seg) types[modelId] = `${seg.type}:${seg.label}`;
    }
    ranges.push({ start, end, types });
  }
  return ranges;
}

function ModelColumn({
  modelId,
  query,
  result,
  loading,
  consensusMap,
  schemaAssets,
}: {
  modelId: string;
  query: string;
  result: SemanticPreviewModelResult | undefined;
  loading: boolean;
  consensusMap: Map<string, 'consistent' | 'different'>;
  /** DataHub asset names (same for all models); when set, used for Asset line instead of per-model tags */
  schemaAssets?: string[];
}) {
  const segments = result?.highlighted_segments ?? [];
  const tags = result?.tags ?? [];
  const displayName = getModelDisplayName(modelId);
  const source = getModelSource(modelId);
  const modelHref = getModelUrl(modelId) ?? (modelId.includes('/') ? `https://huggingface.co/${modelId}` : null);

  const timeMs = result?.semantic_annotation_time_ms;

  return (
    <div className="rounded-xl border border-surface-200 bg-white shadow-sm overflow-hidden flex flex-col">
      <div className="px-3 py-2 border-b border-surface-200 bg-surface-50/80 flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <h3 className="text-xs font-semibold text-gray-800">{displayName}</h3>
          <p className="text-[10px] text-gray-500 font-mono truncate mt-0.5" title={modelId}>
            {modelId}
          </p>
          {source && (
            <div className="flex items-center gap-1.5 mt-1 text-[10px] text-gray-500" title={`Model source: ${source.name}`}>
              <span className="flex-shrink-0 text-gray-400">
                <SourceIcon kind={source.kind} />
              </span>
              {source.kind === 'huggingface' && modelHref ? (
                <a
                  href={modelHref}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="truncate text-brand-600 hover:text-brand-700 hover:underline"
                  title={`View on Hugging Face: ${displayName}`}
                >
                  {source.name}
                </a>
              ) : source ? (
                <span className="truncate">{source.name}</span>
              ) : null}
            </div>
          )}
        </div>
        {timeMs != null && !loading && (
          <span className="shrink-0 text-[10px] font-semibold text-brand-700 tabular-nums whitespace-nowrap" title="Semantic annotation time">
            {timeMs < 1 && timeMs > 0
              ? `${(timeMs * 1000).toFixed(0)} μs`
              : `${timeMs} ms`}
          </span>
        )}
      </div>
      <div className="p-3 flex-1 min-h-[80px]">
        {loading ? (
          <div className="text-xs text-gray-400 italic animate-pulse">Analyzing…</div>
        ) : result?.error ? (
          <p className="text-xs text-rose-600">{result.error}</p>
        ) : segments.length > 0 ? (
          (() => {
            // Merge consecutive duplicate highlight segments (same text) — backend also dedupes by (start,end)
            const merged: typeof segments = [];
            for (let i = 0; i < segments.length; i++) {
              const seg = segments[i];
              const prev = merged[merged.length - 1];
              const isHighlight = seg.type === 'highlight' && seg.tag;
              if (
                isHighlight &&
                prev?.type === 'highlight' &&
                prev?.tag &&
                seg.text === (prev as { text?: string }).text
              ) {
                continue;
              }
              merged.push(seg);
            }
            return (
              <div className="text-xs text-gray-800 leading-5 tracking-normal break-words">
                {merged.map((seg, i) => {
                  const key = `${modelId}-${i}-${seg.text}`;
                  const type = seg.tag_type ?? seg.tag?.type ?? 'text';
                  const label = seg.label ?? seg.tag?.label ?? type;
                  const isHighlight = seg.type === 'highlight' && seg.tag;
                  const rangeKey = `${seg.text}-${String(type).toLowerCase()}:${String(label).toLowerCase()}`;
                  const consensus = consensusMap.get(rangeKey);
                  // In compare view, color by agreement only (legend: green = all agree, amber = disagree)
                  const agreementFill =
                    consensus === 'consistent'
                      ? 'bg-emerald-100 text-emerald-900 border-emerald-300'
                      : consensus === 'different'
                        ? 'bg-amber-100 text-amber-900 border-amber-300'
                        : 'bg-gray-100 text-gray-800 border-gray-300';
                  const baseClass = isHighlight
                    ? `inline rounded px-0.5 py-0 mx-0.5 border text-[11px] font-medium ${agreementFill}`
                    : 'text-gray-800';
                  return isHighlight ? (
                    <span
                      key={key}
                      className={baseClass}
                      title={`${seg.label ?? seg.tag?.label}: ${seg.tag?.value ?? seg.text} (tag: ${type})${consensus ? ` · ${consensus}` : ' · one model'}`}
                    >
                      {seg.text}
                    </span>
                  ) : (
                    <span key={key} className="text-gray-800">
                      {seg.text}
                    </span>
                  );
                })}
              </div>
            );
          })()
        ) : (
          <span className="text-gray-500 text-xs">{query || '—'}</span>
        )}
        {!loading && (() => {
          const typeLower = (t: { type?: string }) => (t.type ?? '').toLowerCase();
          const allFiltered = tags.filter((t) => {
            const type = typeLower(t);
            if (type === 'model') return false;
            if (type.startsWith('embedding')) return false;
            return true;
          });
          const schemaTagsRaw = allFiltered.filter((t) => isSchemaTag(t.type ?? ''));
          const schemaTagsDeduped = schemaTagsRaw.filter((tag, idx, arr) => {
            const key = `${(tag.type ?? '').toLowerCase()}:${String(tag.value ?? tag.snippet ?? '').trim()}`;
            const firstIdx = arr.findIndex((t) => `${(t.type ?? '').toLowerCase()}:${String(t.value ?? t.snippet ?? '').trim()}` === key);
            return firstIdx === idx;
          });
          const schemaTagsDedupedSorted = [...schemaTagsDeduped].sort((a, b) => {
            const typeA = (a.type ?? '').toLowerCase();
            const typeB = (b.type ?? '').toLowerCase();
            if (typeA !== typeB) return typeA.localeCompare(typeB);
            const valA = String(a.value ?? a.snippet ?? '').trim();
            const valB = String(b.value ?? b.snippet ?? '').trim();
            return valA.localeCompare(valB);
          });
          const assetNames =
            schemaAssets != null && schemaAssets.length > 0
              ? schemaAssets
              : Array.from(
                  new Set(
                    schemaTagsDedupedSorted
                      .filter(isAssetTag)
                      .map((t) => String(t.value ?? (t.label ?? '').replace(/^Dataset:\s*/i, '')).trim())
                      .filter(Boolean)
                  )
                ).sort((a, b) => a.localeCompare(b));
          const schemaFieldsOnly = schemaTagsDedupedSorted.filter((t) => !isAssetTag(t));
          const entityTags = allFiltered.filter((t) => !isSchemaTag(t.type ?? ''));
          return (
            <>
              {entityTags.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mt-2 pt-2 border-t border-surface-200">
                  <span className="w-full text-[10px] text-gray-600 font-bold mb-0.5">Entities identified</span>
                  {entityTags.map((tag, i) => (
                    <span
                      key={`tag-${i}`}
                      className={`inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-medium border ${getSemanticColor(tag.type)}`}
                      title={`${tag.label}: ${tag.value ?? tag.snippet}`}
                    >
                      {tag.type}: {typeof tag.value !== 'undefined' ? String(tag.value) : tag.snippet}
                    </span>
                  ))}
                </div>
              )}
              {(assetNames.length > 0 || schemaFieldsOnly.length > 0) && (
                <div className="mt-2 pt-2 border-t border-gray-200 bg-gray-50/80 rounded px-2 py-1.5 -mx-0.5">
                  <p className="text-[10px] text-gray-600 font-bold mb-1">
                    DataHub schema (assets & schema used for semantic tagging)
                  </p>
                  {assetNames.length > 0 && (
                    <div className="mb-1.5">
                      <span className="text-[10px] text-gray-600 font-bold mr-1">Asset:</span>
                      <span className="text-[10px] text-gray-700">
                        {assetNames.join(', ')}
                      </span>
                    </div>
                  )}
                  {schemaFieldsOnly.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {schemaFieldsOnly.map((tag, i) => (
                        <span
                          key={`schema-${i}`}
                          className="inline-flex items-center rounded px-1.5 py-0.5 text-[10px] text-gray-600 bg-white border border-gray-200"
                          title={`${tag.label}: ${tag.value ?? tag.snippet}`}
                        >
                          {tag.type}: {typeof tag.value !== 'undefined' ? String(tag.value) : tag.snippet}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </>
          );
        })()}
      </div>
    </div>
  );
}

export default function SemanticCompare() {
  const [input, setInput] = useState('');
  const [query, setQuery] = useState('');
  const [data, setData] = useState<SemanticPreviewMulti | null>(null);
  const [loading, setLoading] = useState(false);
  const [luckyLoading, setLuckyLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [exampleSeed, setExampleSeed] = useState(() => Date.now());
  const [usedInSession, setUsedInSession] = useState<Set<string>>(() => new Set());
  const [_leaderboard, setLeaderboard] = useState<{
    queryCount: number;
    modelStats: Record<string, ModelPerformance>;
  }>(() => ({ queryCount: 0, modelStats: {} }));
  /** After "I'm feeling lucky!", which LLM (or template) generated the sentence */
  const [luckySource, setLuckySource] = useState<{ source: string; llm_model?: string; sg_agent_time_ms?: number } | null>(null);

  const suggestionsByCategory = useMemo(
    () => groupByCategory(EXAMPLE_POOL, exampleSeed),
    [exampleSeed]
  );

  const runCompare = useCallback(async () => {
    const q = input.trim();
    if (!q) return;
    setQuery(q);
    setLoading(true);
    setError(null);
    try {
      const res = await previewSemanticAnalysisMulti(q);
      setData(res);
      if (res?.models && Object.keys(res.models).length > 0) {
        setLeaderboard((prev) => updateLeaderboard(prev, res));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to run comparison');
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [input]);

  const fillAndCompare = useCallback(
    (q: string) => {
      setInput(q);
      setUsedInSession((prev) => new Set(prev).add(q));
      if (q.trim() && !loading) {
        setQuery(q.trim());
        setLoading(true);
        setError(null);
        previewSemanticAnalysisMulti(q.trim())
          .then((res) => {
            setData(res);
            if (res?.models && Object.keys(res.models).length > 0) {
              setLeaderboard((prev) => updateLeaderboard(prev, res));
            }
          })
          .catch((e) => {
            setError(e instanceof Error ? e.message : 'Failed to run comparison');
            setData(null);
          })
          .finally(() => setLoading(false));
      }
    },
    [loading]
  );

  const shuffleSuggestions = useCallback(() => {
    setExampleSeed(Date.now());
  }, []);

  const handleLucky = useCallback(async () => {
    setLuckyLoading(true);
    setError(null);
    setLuckySource(null);
    try {
      const data = await getLuckySuggestion();
      const sentence = data.sentence?.trim();
      if (!sentence) {
        setLuckyLoading(false);
        return;
      }
      setLuckySource({
        source: data.source ?? 'template',
        llm_model: data.llm_model,
        sg_agent_time_ms: data.sg_agent_time_ms,
      });
      setInput(sentence);
      setQuery(sentence);
      setUsedInSession((prev) => new Set(prev).add(sentence));
      setLoading(true);
      setError(null);
      const res = await previewSemanticAnalysisMulti(sentence);
      setData(res);
      if (res?.models && Object.keys(res.models).length > 0) {
        setLeaderboard((prev) => updateLeaderboard(prev, res));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to get suggestion');
      setData(null);
    } finally {
      setLoading(false);
      setLuckyLoading(false);
    }
  }, []);

  // Use actual keys from API response (same as Chat); dedupe so we never show duplicate model columns
  const modelIds =
    data?.models && Object.keys(data.models).length > 0
      ? [...new Set(Object.keys(data.models))].sort(
          (a, b) =>
            (MODEL_ORDER as readonly string[]).indexOf(a) -
            (MODEL_ORDER as readonly string[]).indexOf(b)
        )
      : [...new Set(MODEL_ORDER)];

  const modelSegments: Record<string, AlignedSegment[]> = {};
  if (data?.models && query) {
    for (const id of modelIds) {
      const m = data.models[id];
      if (m?.highlighted_segments) modelSegments[id] = segmentsToAligned(query, m.highlighted_segments);
      else modelSegments[id] = [];
    }
  }
  const ranges = query && Object.keys(modelSegments).length > 0 ? buildRanges(query, modelSegments) : [];
  const consensusMap = new Map<string, 'consistent' | 'different'>();
  for (const r of ranges) {
    const types = Object.values(r.types).filter(Boolean);
    if (types.length === 0) continue;
    const unique = new Set(types.map((t) => t.toLowerCase()));
    const text = query.slice(r.start, r.end);
    const consistent = types.length >= 2 && unique.size === 1;
    for (const t of types) {
      const key = `${text}-${t.toLowerCase()}`;
      consensusMap.set(key, consistent ? 'consistent' : 'different');
    }
  }

  return (
    <div className="flex gap-4 w-full min-h-0">
      {/* Main: input + 3-column model view */}
      <div className="flex-1 min-w-0 flex flex-col gap-4">
        <div>
          <h1 className="text-xl font-semibold text-gray-900 mb-1">Semantic annotation comparison</h1>
          <p className="text-sm text-gray-600">
            Enter a sentence and compare how all three models annotate it. Consistent annotations are highlighted in green; differences in amber.
          </p>
        </div>

        <div className="rounded-xl border border-surface-200 bg-white shadow-sm p-4">
          <div className="flex flex-col gap-3">
            <textarea
              id="compare-input"
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                setLuckySource(null);
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  runCompare();
                }
              }}
              placeholder="e.g. Show me transactions in California over $1000"
              rows={5}
              className="w-full min-h-[120px] rounded-lg border border-surface-200 px-4 py-3 text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent resize-y"
              aria-label="Sentence or query to compare"
            />
            <div className="flex flex-wrap gap-2 items-center">
              <button
                type="button"
                onClick={runCompare}
                disabled={loading || !input.trim()}
                className="px-5 py-3 rounded-lg bg-brand-500 text-white text-sm font-medium hover:bg-brand-600 disabled:opacity-50 disabled:pointer-events-none transition-colors"
              >
                {loading ? 'Comparing…' : 'Compare'}
              </button>
              <button
                type="button"
                onClick={handleLucky}
                disabled={loading || luckyLoading}
                className="px-4 py-3 rounded-lg bg-amber-500 text-white text-sm font-medium hover:bg-amber-600 disabled:opacity-50 disabled:pointer-events-none transition-colors whitespace-nowrap"
                title="Get a random sentence from sg-agent (business analytics, customer support, or regulatory)"
              >
                {luckyLoading ? '…' : "I'm feeling lucky!"}
              </button>
            </div>
            {luckySource?.llm_model && (
              <p className="text-xs text-gray-500 flex items-center gap-1.5 mt-1" role="status">
                <span className="inline-flex items-center justify-center w-4 h-4 rounded bg-gray-100 text-gray-400" aria-hidden>◇</span>
                Generated with{' '}
                {getModelUrl(luckySource.llm_model) ? (
                  <a href={getModelUrl(luckySource.llm_model)!} target="_blank" rel="noopener noreferrer" className="font-medium text-brand-600 hover:underline">
                    {getModelDisplayName(luckySource.llm_model)}
                  </a>
                ) : (
                  <span className="font-medium text-gray-600">{getModelDisplayName(luckySource.llm_model)}</span>
                )}
                {luckySource.sg_agent_time_ms != null && (
                  <span
                    className="text-gray-400 font-mono tabular-nums text-[11px]"
                    title={`sg-agent: ${luckySource.sg_agent_time_ms} ms`}
                  >
                    · {luckySource.sg_agent_time_ms >= 1000
                      ? `${(luckySource.sg_agent_time_ms / 1000).toFixed(1)}s`
                      : `${luckySource.sg_agent_time_ms}ms`}
                  </span>
                )}
              </p>
            )}
          </div>
          {error && (
            <p className="mt-2 text-sm text-rose-600" role="alert">
              {error}
            </p>
          )}
        </div>

        {/* Legend: highlight background = agreement across 3 models */}
        <div className="flex flex-wrap items-center gap-4 text-xs text-gray-600">
          <span className="inline-flex items-center gap-1.5">
            <span className="w-4 h-4 rounded border border-emerald-300 bg-emerald-100" aria-hidden />
            Green — all 3 models agree on this span
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-4 h-4 rounded border border-amber-300 bg-amber-100" aria-hidden />
            Amber — models disagree on this span
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-4 h-4 rounded border border-gray-300 bg-gray-100" aria-hidden />
            Gray — only one model annotated this span
          </span>
        </div>

        {/* 3-column pane view: model results */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {modelIds.map((modelId) => (
            <ModelColumn
              key={modelId}
              modelId={modelId}
              query={query}
              result={data?.models?.[modelId]}
              loading={loading}
              consensusMap={consensusMap}
              schemaAssets={data?.schema_assets}
            />
          ))}
        </div>

        {data?.semantic_annotation_time_ms != null && !loading && (
          <p className="text-xs text-gray-500 text-right">
            Annotated in {data.semantic_annotation_time_ms < 1 ? `${(data.semantic_annotation_time_ms * 1000).toFixed(0)} μs` : `${data.semantic_annotation_time_ms} ms`}
          </p>
        )}
      </div>

      <SuggestionsPanel
        byCategory={suggestionsByCategory}
        usedInSession={usedInSession}
        onSelect={fillAndCompare}
        onMore={shuffleSuggestions}
        loading={loading}
      />
    </div>
  );
}

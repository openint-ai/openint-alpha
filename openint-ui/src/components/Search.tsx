import { useState, useEffect, useRef, useCallback } from 'react';
import { search, stats, type SearchResult } from '../api';
import {
  parseStructuredContent,
  formatStructuredForDisplay,
  formatOneLineSummary,
} from '../utils/structuredData';

const ENTITY_LABELS: Record<string, string> = {
  customers: 'Customer',
  ach_transactions: 'Transaction (ACH)',
  wire_transactions: 'Transaction (Wire)',
  credit_transactions: 'Transaction (Credit)',
  debit_transactions: 'Transaction (Debit)',
  check_transactions: 'Transaction (Check)',
  country_codes: 'Country',
  state_codes: 'State',
  zip_codes: 'Address / ZIP',
};

/** Type filter options for "display only" scope */
const TYPE_FILTER_OPTIONS: { value: string; label: string }[] = [
  { value: '', label: 'All' },
  { value: 'customers', label: 'Customer' },
  { value: 'ach_transactions', label: 'ACH' },
  { value: 'wire_transactions', label: 'Wire' },
  { value: 'credit_transactions', label: 'Credit' },
  { value: 'debit_transactions', label: 'Debit' },
  { value: 'check_transactions', label: 'Check' },
  { value: 'zip_codes', label: 'ZIP' },
  { value: 'state_codes', label: 'State' },
  { value: 'country_codes', label: 'Country' },
];

const ENTITY_BADGE_CLASS: Record<string, string> = {
  customers: 'bg-semantic-entity text-teal-800 border-teal-200',
  ach_transactions: 'bg-semantic-type text-sky-800 border-sky-200',
  wire_transactions: 'bg-semantic-type text-sky-800 border-sky-200',
  credit_transactions: 'bg-semantic-amount text-amber-800 border-amber-200',
  debit_transactions: 'bg-semantic-amount text-amber-800 border-amber-200',
  check_transactions: 'bg-semantic-amount text-amber-800 border-amber-200',
  country_codes: 'bg-semantic-location text-pink-800 border-pink-200',
  state_codes: 'bg-semantic-location text-pink-800 border-pink-200',
  zip_codes: 'bg-semantic-location text-pink-800 border-pink-200',
};

function textPart(content: string): string {
  return content.split('Structured Data:')[0].trim();
}

function isAmountLike(val: unknown): boolean {
  if (typeof val !== 'string' && typeof val !== 'number') return false;
  const s = String(val);
  return /^-?\$?[\d,]+(\.\d{2})?(\s*(USD|usd))?$/.test(s) || /^-?[\d,]+\.\d{2}$/.test(s);
}

function isIdLike(val: unknown): boolean {
  if (typeof val !== 'string') return false;
  return /^(CUST|ACH|WIRE|CHK|CRD|DBT)\d+/i.test(val) || /^[A-Z0-9]{10,}$/i.test(String(val).slice(0, 20));
}

function isDateLike(val: unknown): boolean {
  if (typeof val !== 'string') return false;
  return /^\d{4}-\d{2}-\d{2}/.test(String(val)) || /\d{1,2}\/\d{1,2}\/\d{2,4}/.test(String(val));
}

function StructuredValue({ value }: { value: unknown }) {
  const s = value === null || value === undefined ? '' : String(value);
  if (isAmountLike(value)) return <span className="px-1.5 py-0.5 rounded bg-semantic-amount text-amber-900 text-xs font-medium">{s}</span>;
  if (isIdLike(value)) return <span className="px-1.5 py-0.5 rounded bg-semantic-id text-indigo-900 text-xs font-mono">{s}</span>;
  if (isDateLike(value)) return <span className="px-1.5 py-0.5 rounded bg-semantic-date text-emerald-800 text-xs">{s}</span>;
  return <span className="text-gray-700 text-xs">{s}</span>;
}

function ResultItem({ r }: { r: SearchResult }) {
  const [expanded, setExpanded] = useState(false);
  const content = r.content ?? '';
  const meta = r.metadata ?? {};
  const fileType = ((meta.file_type as string) ?? '').trim().toLowerCase();
  const entityLabel = ENTITY_LABELS[fileType] ?? (fileType.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()) || 'Record');
  const badgeClass = ENTITY_BADGE_CLASS[fileType] ?? 'bg-surface-200 text-gray-700 border-surface-300';
  const structured = parseStructuredContent(content);
  const snippet = textPart(content);
  const isLong = snippet.length > 320;
  const showSnippet = expanded ? snippet : snippet.slice(0, 320) + (isLong ? '…' : '');
  const sections = structured ? formatStructuredForDisplay(structured, fileType) : [];
  const oneLine = structured ? formatOneLineSummary(structured, fileType) : '';

  return (
    <div className="rounded-xl bg-white border border-surface-200 p-4 shadow-sm hover:shadow-md hover:border-brand-200 transition-all">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <span className={`inline-flex items-center px-2.5 py-1 rounded-lg text-xs font-semibold border ${badgeClass}`}>
          {entityLabel}
        </span>
        <div className="flex items-center gap-2">
          <span className="font-mono text-xs text-gray-500 truncate max-w-[140px]" title={r.id}>{r.id}</span>
          <span className="text-xs text-brand-600 font-medium shrink-0">{(1 - (r.score ?? 0)).toFixed(3)}</span>
        </div>
      </div>
      {oneLine && (
        <p className="text-sm text-gray-800 font-medium mb-2">{oneLine}</p>
      )}
      {!structured && showSnippet && (
        <p className="text-sm text-gray-600 whitespace-pre-wrap mb-3">{showSnippet}</p>
      )}
      {structured && showSnippet && (
        <p className="text-sm text-gray-600 whitespace-pre-wrap mb-2 line-clamp-2">{showSnippet}</p>
      )}
      {structured && isLong && (
        <button
          type="button"
          onClick={() => setExpanded((e) => !e)}
          className="mb-2 text-sm text-brand-600 hover:text-brand-700 font-medium"
        >
          {expanded ? 'Show less' : 'Show more'}
        </button>
      )}
      {sections.length > 0 && (
        <div className="mt-3 pt-3 border-t border-surface-200">
          <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Indexed data</div>
          {sections.map((sec, si) => (
            <div key={si} className="mb-3 last:mb-0">
              {sec.title && (
                <div className="text-xs font-medium text-gray-600 mb-1.5">{sec.title}</div>
              )}
              <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-2 text-sm">
                {sec.entries.map(({ label, key, value }) => (
                  <div key={key} className="flex gap-2 min-w-0">
                    <dt className="text-gray-500 shrink-0">{label}:</dt>
                    <dd className="min-w-0">
                      <StructuredValue value={value} />
                    </dd>
                  </div>
                ))}
              </dl>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const SUGGEST_DEBOUNCE_MS = 280;
const SUGGEST_MIN_LEN = 2;
const SUGGEST_TOP_K = 8;

export default function SearchPage() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[] | null>(null);
  const [queryTime, setQueryTime] = useState<number | null>(null);
  const [resultsTimestamp, setResultsTimestamp] = useState<string | null>(null);
  const [totalRecords, setTotalRecords] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searched, setSearched] = useState(false);
  const [selectedType, setSelectedType] = useState('');
  const [suggestions, setSuggestions] = useState<SearchResult[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [suggestLoading, setSuggestLoading] = useState(false);
  const suggestTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const inputContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    stats()
      .then((s) => setTotalRecords(s.total_records))
      .catch(() => setTotalRecords(null));
  }, []);

  // Debounced autocomplete suggestions
  useEffect(() => {
    const q = query.trim();
    if (q.length < SUGGEST_MIN_LEN) {
      setSuggestions([]);
      setShowSuggestions(false);
      if (suggestTimeoutRef.current) {
        clearTimeout(suggestTimeoutRef.current);
        suggestTimeoutRef.current = null;
      }
      return;
    }
    if (suggestTimeoutRef.current) clearTimeout(suggestTimeoutRef.current);
    suggestTimeoutRef.current = setTimeout(async () => {
      suggestTimeoutRef.current = null;
      setSuggestLoading(true);
      try {
        const res = await search(q, SUGGEST_TOP_K, selectedType || undefined);
        setSuggestions(res.results ?? []);
        setShowSuggestions(true);
      } catch {
        setSuggestions([]);
      } finally {
        setSuggestLoading(false);
      }
    }, SUGGEST_DEBOUNCE_MS);
    return () => {
      if (suggestTimeoutRef.current) clearTimeout(suggestTimeoutRef.current);
    };
  }, [query, selectedType]);

  // Close suggestions on click outside
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (inputContainerRef.current && !inputContainerRef.current.contains(e.target as Node)) {
        setShowSuggestions(false);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  const run = useCallback(async () => {
    const q = query.trim();
    if (!q || loading) return;
    setError(null);
    setShowSuggestions(false);
    setLoading(true);
    setSearched(true);
    try {
      const res = await search(q, 10, selectedType || undefined);
      setResults(res.results ?? []);
      setQueryTime(res.query_time ?? null);
      setResultsTimestamp(new Date().toISOString());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Search failed');
      setResults([]);
      setQueryTime(null);
      setResultsTimestamp(null);
    } finally {
      setLoading(false);
    }
  }, [query, selectedType, loading]);

  const handleSuggestionSelect = useCallback(
    (r: SearchResult) => {
      const meta = r.metadata ?? {};
      const fileType = ((meta.file_type as string) ?? '').trim().toLowerCase();
      const structured = parseStructuredContent(r.content ?? '');
      const oneLine = structured ? formatOneLineSummary(structured, fileType) : r.id;
      setQuery(oneLine);
      setSelectedType(fileType);
      setShowSuggestions(false);
      setSuggestions([]);
      setSearched(true);
      setError(null);
      setLoading(true);
      search(oneLine, 10, fileType)
        .then((res) => {
          setResults(res.results ?? []);
          setQueryTime(res.query_time ?? null);
          setResultsTimestamp(new Date().toISOString());
        })
        .catch((e) => {
          setError(e instanceof Error ? e.message : 'Search failed');
          setResults([]);
          setQueryTime(null);
          setResultsTimestamp(null);
        })
        .finally(() => setLoading(false));
    },
    []
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-gray-900 mb-1">Semantic Search</h1>
        <p className="text-sm text-gray-500">
          {totalRecords != null ? `${totalRecords.toLocaleString()} documents indexed in Milvus` : 'Loading…'}
        </p>
      </div>

      <div className="flex flex-wrap gap-2 items-center">
        <span className="text-xs font-medium text-gray-500">Show only:</span>
        {TYPE_FILTER_OPTIONS.map((opt) => (
          <button
            key={opt.value || 'all'}
            type="button"
            onClick={() => setSelectedType(opt.value)}
            className={`px-2.5 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
              selectedType === opt.value
                ? 'bg-brand-100 text-brand-700 border-brand-300'
                : 'bg-white text-gray-600 border-surface-200 hover:border-brand-200 hover:text-brand-600'
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>

      <div ref={inputContainerRef} className="relative flex gap-2">
        <div className="flex-1 relative">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                setShowSuggestions(false);
                run();
              }
              if (e.key === 'Escape') setShowSuggestions(false);
            }}
            onFocus={() => query.trim().length >= SUGGEST_MIN_LEN && suggestions.length > 0 && setShowSuggestions(true)}
            placeholder="Search by customer ID, transaction, wire number, ZIP, state…"
            className="w-full rounded-xl bg-white border border-surface-200 px-4 py-3 text-gray-800 placeholder-gray-400 focus:outline-none focus:border-brand-400 focus:ring-2 focus:ring-brand-100 transition-all"
            autoComplete="off"
          />
          {showSuggestions && (suggestions.length > 0 || suggestLoading) && (
            <div className="absolute top-full left-0 right-0 z-20 mt-1 rounded-xl border border-surface-200 bg-white shadow-lg overflow-hidden">
              {suggestLoading ? (
                <div className="px-4 py-3 text-sm text-gray-500">Looking up…</div>
              ) : (
                <ul className="max-h-72 overflow-y-auto py-1">
                  {suggestions.map((r) => {
                    const meta = r.metadata ?? {};
                    const fileType = ((meta.file_type as string) ?? '').trim().toLowerCase();
                    const entityLabel = ENTITY_LABELS[fileType] ?? fileType;
                    const badgeClass = ENTITY_BADGE_CLASS[fileType] ?? 'bg-surface-200 text-gray-700';
                    const structured = parseStructuredContent(r.content ?? '');
                    const oneLine = structured ? formatOneLineSummary(structured, fileType) : r.id;
                    return (
                      <li key={r.id}>
                        <button
                          type="button"
                          onClick={() => handleSuggestionSelect(r)}
                          className="w-full text-left px-4 py-2.5 hover:bg-surface-50 flex items-center gap-2 border-b border-surface-100 last:border-b-0"
                        >
                          <span className={`shrink-0 px-2 py-0.5 rounded text-xs font-medium ${badgeClass}`}>
                            {entityLabel}
                          </span>
                          <span className="min-w-0 truncate text-sm text-gray-800">{oneLine}</span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          )}
        </div>
        <button
          type="button"
          onClick={run}
          disabled={loading || !query.trim()}
          className="px-5 py-3 rounded-xl bg-brand-500 hover:bg-brand-600 disabled:opacity-40 font-medium text-sm text-white transition-colors shadow-sm shrink-0"
        >
          {loading ? 'Searching…' : 'Search'}
        </button>
      </div>

      {(queryTime != null || resultsTimestamp) && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-brand-600">
          {queryTime != null && <span className="font-mono">⚡ {queryTime}ms</span>}
          {resultsTimestamp && (
            <span className="text-gray-500 font-normal" title={new Date(resultsTimestamp).toLocaleString()}>
              Results as of {new Date(resultsTimestamp).toLocaleString(undefined, { month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit' })}
            </span>
          )}
        </div>
      )}

      {error && (
        <div className="rounded-xl bg-red-50 border border-red-200 px-4 py-3 text-red-700 text-sm">
          {error}
        </div>
      )}

      {searched && results && (
        <div>
          <h2 className="text-sm font-semibold text-gray-600 mb-3">
            Results {results.length > 0 ? `(${results.length})` : ''}
          </h2>
          {results.length === 0 ? (
            <p className="text-gray-500 text-sm">No results found.</p>
          ) : (
            <div className="space-y-4">
              {results.map((r) => (
                <ResultItem key={r.id} r={r} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

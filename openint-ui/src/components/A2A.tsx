/**
 * A2A — Agent-to-Agent (Google A2A protocol)
 * sg-agent generates sentences → modelmgmt-agent annotates them.
 * Shows A2A flow animation and results.
 */

import { useState, useCallback } from 'react';
import { runA2A, type A2ARunResponse, type A2AAnnotationItem } from '../api';

type Phase = 'idle' | 'sg-agent' | 'modelmgmt-agent' | 'done' | 'error';

export default function A2A() {
  const [sentenceCount, setSentenceCount] = useState(3);
  const [phase, setPhase] = useState<Phase>('idle');
  const [data, setData] = useState<A2ARunResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    setError(null);
    setData(null);
    setPhase('sg-agent');
    try {
      // Simulate step visibility: sg-agent runs first
      const t1 = setTimeout(() => setPhase('modelmgmt-agent'), 800);
      const result = await runA2A(sentenceCount);
      clearTimeout(t1);
      setData(result);
      setPhase(result.success ? 'done' : 'error');
      if (!result.success) setError(result.error || 'A2A run failed');
    } catch (e) {
      setPhase('error');
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [sentenceCount]);

  const isRunning = phase === 'sg-agent' || phase === 'modelmgmt-agent';

  return (
    <div className="w-full max-w-[90rem] mx-auto px-1">
      <h1 className="text-xl font-semibold text-gray-900 mb-1">A2A — Agent-to-Agent</h1>
      <p className="text-sm text-gray-600 mb-6">
        Google&apos;s A2A protocol: <strong>sg-agent</strong> generates example sentences from DataHub schema; <strong>modelmgmt-agent</strong> annotates them with semantic tags. Watch the flow below.
      </p>

      <div className="rounded-xl border border-surface-200 bg-white shadow-sm p-6 mb-6">
        <div className="flex flex-wrap items-center gap-4 mb-6">
          <label className="flex items-center gap-2 text-sm text-gray-700">
            <span>Sentences to generate:</span>
            <select
              value={sentenceCount}
              onChange={(e) => setSentenceCount(Number(e.target.value))}
              disabled={isRunning}
              className="rounded-lg border border-surface-200 px-3 py-1.5 text-sm bg-white"
            >
              {[1, 2, 3, 4, 5].map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </label>
          <button
            type="button"
            onClick={run}
            disabled={isRunning}
            className="px-5 py-2.5 rounded-lg bg-brand-500 text-white text-sm font-medium hover:bg-brand-600 disabled:opacity-50 disabled:pointer-events-none transition-colors"
          >
            {isRunning ? 'Running A2A…' : 'Run A2A'}
          </button>
        </div>

        {/* A2A flow animation: distinct colors + time per model */}
        <div className="flex items-center justify-center gap-4 py-8 px-4 bg-gray-50/80 rounded-xl border border-surface-200">
          <AgentCard
            agent="sg-agent"
            name="sg-agent"
            description="Sentence generation"
            active={phase === 'sg-agent'}
            done={phase !== 'idle' && phase !== 'sg-agent'}
            timeMs={data?.sg_agent_time_ms ?? null}
          />
          <Arrow active={phase === 'modelmgmt-agent' || phase === 'done'} />
          <AgentCard
            agent="modelmgmt-agent"
            name="modelmgmt-agent"
            description="Semantic annotation"
            active={phase === 'modelmgmt-agent'}
            done={phase === 'done' || phase === 'error'}
            timeMs={data?.modelmgmt_agent_time_ms ?? null}
          />
        </div>

        {phase === 'error' && error && (
          <p className="mt-4 text-sm text-rose-600" role="alert">{error}</p>
        )}
      </div>

      {/* Results: sentence | arrow | all 3 model outputs — wide layout */}
      {data?.success && data.sentences?.length > 0 && (
        <div className="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
          <h2 className="text-sm font-semibold text-slate-600 px-4 py-3 border-b border-slate-200 bg-slate-50/80">
            Generated sentence → modelmgmt-agent → semantic annotation (all 3 models)
          </h2>
          <ul className="divide-y divide-slate-200">
            {data.sentences.map((sent, i) => (
              <li key={i} className="p-4">
                <div className="flex flex-col gap-4 lg:flex-row lg:items-stretch lg:gap-0">
                  {/* Sentence + wedge grouped so wedge apex aligns with sentence center */}
                  <div className="hidden lg:flex shrink-0 items-stretch w-full lg:w-auto">
                    {/* Generated sentence (sg-agent) */}
                    <div className="shrink-0 w-80 xl:w-96 rounded-l-lg border border-slate-200 bg-slate-50/80 border-l-4 border-l-blue-400 px-4 py-3 border-r-0 rounded-r-none">
                      <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-1">Generated sentence (sg-agent)</p>
                      <p className="text-sm text-slate-800 leading-snug">{sent.text}</p>
                      {sent.category && (
                        <span className="inline-block mt-2 text-[10px] text-slate-600 bg-slate-200/80 px-1.5 py-0.5 rounded">
                          {sent.category}
                        </span>
                      )}
                    </div>
                    {/* Wedge: apex at center of sentence box — visible flow to 3 model sections */}
                    <div className="shrink-0 w-10 xl:w-14 flex items-stretch min-w-0" aria-hidden>
                      <WedgeSvg />
                    </div>
                  </div>
                  {/* Mobile: sentence only (no wedge) */}
                  <div className="lg:hidden w-full rounded-lg border border-slate-200 bg-slate-50/80 border-l-4 border-l-blue-400 px-4 py-3">
                    <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-1">Generated sentence (sg-agent)</p>
                    <p className="text-sm text-slate-800 leading-snug">{sent.text}</p>
                    {sent.category && (
                      <span className="inline-block mt-2 text-[10px] text-slate-600 bg-slate-200/80 px-1.5 py-0.5 rounded">
                        {sent.category}
                      </span>
                    )}
                  </div>
                  {/* All 3 model outputs — grid */}
                  <div className="flex-1 min-w-0 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4 lg:pl-1">
                    {data.annotations?.[i] ? (
                      <SemanticAnnotationOutput item={data.annotations[i]} />
                    ) : (
                      <p className="text-sm text-slate-500 col-span-full">No annotation</p>
                    )}
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

/** Format ms for display in animation (e.g. "1.2s" or "234 ms") */
function formatTimeMs(ms: number | null | undefined): string | null {
  if (ms == null || ms < 0) return null;
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.round(ms)} ms`;
}

const SG_AGENT_STYLES = {
  idle: 'border-violet-200 bg-white',
  active: 'border-violet-500 bg-violet-50 shadow-md shadow-violet-200/50 scale-105',
  done: 'border-violet-400 bg-violet-50/90',
  badge: 'bg-violet-500 text-white',
  badgeDone: 'bg-violet-600 text-white',
  time: 'text-violet-600',
};

const MODELMGMT_AGENT_STYLES = {
  idle: 'border-teal-200 bg-white',
  active: 'border-teal-500 bg-teal-50 shadow-md shadow-teal-200/50 scale-105',
  done: 'border-teal-400 bg-teal-50/90',
  badge: 'bg-teal-500 text-white',
  badgeDone: 'bg-teal-600 text-white',
  time: 'text-teal-600',
};

function AgentCard({
  agent,
  name,
  description,
  active,
  done,
  timeMs,
}: {
  agent: 'sg-agent' | 'modelmgmt-agent';
  name: string;
  description: string;
  active: boolean;
  done: boolean;
  timeMs: number | null;
}) {
  const styles = agent === 'sg-agent' ? SG_AGENT_STYLES : MODELMGMT_AGENT_STYLES;
  const boxClass = active ? styles.active : done ? styles.done : styles.idle;
  const badgeClass = active ? styles.badge + ' animate-pulse' : done ? styles.badgeDone : 'bg-gray-200 text-gray-600';
  const timeStr = formatTimeMs(timeMs);

  return (
    <div
      className={`flex flex-col items-center justify-center rounded-xl border-2 px-6 py-5 min-w-[180px] transition-all duration-300 ${boxClass}`}
    >
      <div className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold mb-2 ${badgeClass}`}>
        {agent === 'sg-agent' ? 'SG' : 'MM'}
      </div>
      <span className="text-sm font-semibold text-gray-800">{name}</span>
      <span className="text-[10px] text-gray-500 mt-0.5">{description}</span>
      {timeStr != null && (
        <span className={`mt-2 text-[11px] font-mono tabular-nums ${styles.time}`} title={`${timeMs} ms`}>
          {timeStr}
        </span>
      )}
    </div>
  );
}

/** Wedge SVG: one point (sentence) fans out to three points (3 models). Visible fill + lines. */
function WedgeSvg() {
  return (
    <svg
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
      className="w-full h-full min-h-[120px]"
      aria-hidden
    >
      {/* Filled wedge: left apex (sentence) → right edge (3 model sections) — visible emerald tint */}
      <path
        d="M 0 50 L 100 5 L 100 95 Z"
        fill="#6ee7b7"
        fillOpacity="0.35"
      />
      {/* Three lines: sentence center → each model section — clear stroke */}
      <path d="M 0 50 L 100 5" stroke="#10b981" strokeWidth="2" strokeOpacity="0.65" fill="none" strokeLinecap="round" />
      <path d="M 0 50 L 100 50" stroke="#10b981" strokeWidth="2" strokeOpacity="0.65" fill="none" strokeLinecap="round" />
      <path d="M 0 50 L 100 95" stroke="#10b981" strokeWidth="2" strokeOpacity="0.65" fill="none" strokeLinecap="round" />
    </svg>
  );
}

function Arrow({ active }: { active: boolean }) {
  return (
    <div className={`flex items-center transition-opacity duration-300 ${active ? 'opacity-100' : 'opacity-50'}`}>
      <svg
        className={`w-10 h-10 transition-all duration-300 ${active ? 'text-brand-500 animate-[pulse_1s_ease-in-out_infinite]' : 'text-gray-400'}`}
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
        aria-hidden
      >
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
      </svg>
    </div>
  );
}

/** External-link icon */
function ExternalLinkIcon({ className }: { className?: string }) {
  return (
    <svg className={className ?? 'w-3.5 h-3.5'} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
    </svg>
  );
}

const HUGGINGFACE_BASE = 'https://huggingface.co/';

/** All 3 model outputs: one card per model with clear name (HF link) and tags. */
function SemanticAnnotationOutput({ item }: { item: A2AAnnotationItem }) {
  const ann = item.annotation;
  if (!ann) return <p className="text-sm text-slate-500 col-span-full">Annotation failed</p>;
  if (ann.error) return <p className="text-sm text-rose-600 col-span-full">{ann.error}</p>;
  const models = ann.models || {};
  const modelIds = Object.keys(models);
  if (modelIds.length === 0) return <p className="text-sm text-slate-500 col-span-full">No model results</p>;

  /** Dedupe tags by (type, label, value) — keep first occurrence */
  const dedupeTags = (arr: Array<{ type?: string; label?: string; value?: string }>) => {
    const seen = new Set<string>();
    return arr.filter((t) => {
      const key = `${(t.type ?? '').toLowerCase()}:${(t.label ?? '').toLowerCase()}:${String(t.value ?? '').trim()}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  };

  /** Dedupe highlighted segments by (text, type) — keep first occurrence */
  const dedupeSegments = (arr: Array<{ text?: string; tag?: { type?: string; label?: string } }>) => {
    const seen = new Set<string>();
    return arr.filter((s) => {
      const key = `${String(s.text ?? '').trim()}:${(s.tag?.type ?? s.tag?.label ?? '').toLowerCase()}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  };

  return (
    <>
      {modelIds.map((id) => {
        const m = models[id];
        const rawTags = (m?.tags ?? []) as Array<{ type?: string; label?: string; value?: string }>;
        const rawSegments = (m?.highlighted_segments ?? []) as Array<{ text?: string; tag?: { type?: string; label?: string } }>;
        const tags = dedupeTags(rawTags).sort((a, b) => {
          const ka = `${(a.label ?? a.type ?? '').toLowerCase()}:${String(a.value ?? '').toLowerCase()}`;
          const kb = `${(b.label ?? b.type ?? '').toLowerCase()}:${String(b.value ?? '').toLowerCase()}`;
          return ka.localeCompare(kb);
        });
        const segments = dedupeSegments(rawSegments);
        const href = `${HUGGINGFACE_BASE}${encodeURIComponent(id)}`;
        const isBest = ann.best_model === id;

        return (
          <div
            key={id}
            className={`rounded-lg border border-slate-200 bg-slate-50/80 border-l-4 px-4 py-3 min-h-[120px] flex flex-col ${
              isBest ? 'border-l-emerald-500 ring-1 ring-emerald-200/60' : 'border-l-emerald-400'
            }`}
          >
            {/* Model name — prominent, Hugging Face link */}
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 text-sm font-semibold text-slate-800 hover:text-emerald-700 hover:underline mb-2 group"
              title={`View on Hugging Face: ${id}`}
            >
              <span className="truncate">{id}</span>
              <ExternalLinkIcon className="w-4 h-4 shrink-0 text-slate-400 group-hover:text-emerald-500" />
              {isBest && (
                <span className="shrink-0 text-[10px] font-medium text-emerald-600 bg-emerald-100 px-1.5 py-0.5 rounded">
                  Best
                </span>
              )}
            </a>
            {/* Tags — full list (deduped) */}
            <div className="flex-1 min-w-0">
              <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-1.5">Tags</p>
              {tags.length > 0 ? (
                <div className="flex flex-wrap gap-1.5">
                  {tags.map((t, j) => (
                    <span
                      key={j}
                      className="px-2 py-0.5 rounded-md bg-white border border-slate-200 text-xs text-slate-700"
                      title={`${t.label ?? t.type}: ${t.value ?? ''}`}
                    >
                      {t.label ?? t.type}: {String(t.value ?? '').slice(0, 36)}
                    </span>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-slate-400">—</p>
              )}
              {/* Highlighted spans — compact */}
              {segments.length > 0 && (
                <>
                  <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mt-2 mb-1">Highlighted</p>
                  <p className="text-xs text-slate-600 leading-relaxed">
                    {segments.slice(0, 8).map((s, j) => (
                      <span key={j}>
                        <mark className="bg-emerald-100/80 text-emerald-900 rounded px-0.5">{s.text}</mark>
                        <span className="text-slate-400 ml-0.5">({s.tag?.type ?? s.tag?.label ?? '—'})</span>
                        {j < Math.min(8, segments.length) - 1 ? ' ' : ''}
                      </span>
                    ))}
                    {segments.length > 8 && <span className="text-slate-400"> +{segments.length - 8}</span>}
                  </p>
                </>
              )}
            </div>
          </div>
        );
      })}
    </>
  );
}

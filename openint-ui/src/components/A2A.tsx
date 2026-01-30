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
    <div className="max-w-4xl mx-auto">
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

        {/* A2A flow animation */}
        <div className="flex items-center justify-center gap-4 py-8 px-4 bg-gray-50/80 rounded-xl border border-surface-200">
          <AgentCard
            name="sg-agent"
            description="Sentence generation"
            active={phase === 'sg-agent'}
            done={phase !== 'idle' && phase !== 'sg-agent'}
          />
          <Arrow active={phase === 'modelmgmt-agent' || phase === 'done'} />
          <AgentCard
            name="modelmgmt-agent"
            description="Semantic annotation"
            active={phase === 'modelmgmt-agent'}
            done={phase === 'done' || phase === 'error'}
          />
        </div>

        {phase === 'error' && error && (
          <p className="mt-4 text-sm text-rose-600" role="alert">{error}</p>
        )}
      </div>

      {/* Results: sentence (sg-agent) → modelmgmt-agent → semantic annotation */}
      {data?.success && data.sentences?.length > 0 && (
        <div className="rounded-xl border border-surface-200 bg-white shadow-sm overflow-hidden">
          <h2 className="text-sm font-semibold text-gray-800 px-4 py-3 border-b border-surface-200 bg-surface-50">
            Generated sentence → modelmgmt-agent → semantic annotation
          </h2>
          <ul className="divide-y divide-surface-200">
            {data.sentences.map((sent, i) => (
              <li key={i} className="p-4">
                <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:gap-4">
                  {/* 1. Generated sentence (from sg-agent) */}
                  <div className="flex-1 min-w-0 rounded-lg border border-emerald-200 bg-emerald-50/80 px-3 py-2.5">
                    <p className="text-[10px] font-semibold text-emerald-800 uppercase tracking-wider mb-1">Generated sentence (sg-agent)</p>
                    <p className="text-sm text-gray-800">{sent.text}</p>
                    {sent.category && (
                      <span className="inline-block mt-1.5 text-[10px] text-emerald-700 bg-emerald-100 px-1.5 py-0.5 rounded">
                        {sent.category}
                      </span>
                    )}
                  </div>
                  {/* 2. Arrow into model */}
                  <div className="flex items-center justify-center shrink-0 text-gray-400">
                    <svg className="w-6 h-6 sm:w-8 sm:h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                    </svg>
                  </div>
                  {/* 3. Semantic annotation (from modelmgmt-agent) */}
                  <div className="flex-1 min-w-0 rounded-lg border border-amber-200 bg-amber-50/80 px-3 py-2.5">
                    <p className="text-[10px] font-semibold text-amber-800 uppercase tracking-wider mb-1.5">Semantic annotation (modelmgmt-agent)</p>
                    {data.annotations?.[i] ? (
                      <SemanticAnnotationOutput item={data.annotations[i]} />
                    ) : (
                      <p className="text-xs text-amber-700">No annotation</p>
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

function AgentCard({
  name,
  description,
  active,
  done,
}: {
  name: string;
  description: string;
  active: boolean;
  done: boolean;
}) {
  return (
    <div
      className={`
        flex flex-col items-center justify-center rounded-xl border-2 px-6 py-5 min-w-[180px] transition-all duration-300
        ${active ? 'border-brand-500 bg-brand-50 shadow-md scale-105' : ''}
        ${done && !active ? 'border-emerald-300 bg-emerald-50/80' : ''}
        ${!active && !done ? 'border-surface-200 bg-white' : ''}
      `}
    >
      <div
        className={`
          w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold mb-2
          ${active ? 'bg-brand-500 text-white animate-pulse' : done ? 'bg-emerald-500 text-white' : 'bg-gray-200 text-gray-600'}
        `}
      >
        {name === 'sg-agent' ? 'SG' : 'MM'}
      </div>
      <span className="text-sm font-semibold text-gray-800">{name}</span>
      <span className="text-[10px] text-gray-500 mt-0.5">{description}</span>
    </div>
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

/** Renders the semantic annotation output for one sentence (tags, highlighted segments summary). */
function SemanticAnnotationOutput({ item }: { item: A2AAnnotationItem }) {
  const ann = item.annotation;
  if (!ann) return <p className="text-xs text-amber-700">Annotation failed</p>;
  if (ann.error) return <p className="text-xs text-rose-600">{ann.error}</p>;
  const models = ann.models || {};
  const modelIds = Object.keys(models);
  if (modelIds.length === 0) return <p className="text-xs text-gray-500">No model results</p>;
  return (
    <div className="text-xs text-gray-700 space-y-2">
      {modelIds.map((id) => {
        const m = models[id];
        const tags = (m?.tags ?? []) as Array<{ type?: string; label?: string; value?: string }>;
        const segments = (m?.highlighted_segments ?? []) as Array<{ text?: string; tag?: { type?: string; label?: string } }>;
        const shortId = id.split('/').pop() ?? id;
        return (
          <div key={id} className="border-b border-amber-200/60 pb-2 last:border-0 last:pb-0">
            <p className="font-medium text-amber-900 mb-1">{shortId}</p>
            {tags.length > 0 && (
              <p className="mb-1">
                <span className="text-amber-800">Tags: </span>
                {tags.slice(0, 8).map((t, j) => (
                  <span key={j} className="inline-block mr-1 mt-0.5 px-1.5 py-0.5 rounded bg-amber-100 text-amber-900">
                    {t.label ?? t.type}: {String(t.value ?? '').slice(0, 30)}
                  </span>
                ))}
                {tags.length > 8 && <span className="text-gray-500"> +{tags.length - 8} more</span>}
              </p>
            )}
            {segments.length > 0 && (
              <p className="text-gray-600">
                <span className="text-amber-800">Highlighted: </span>
                {segments.slice(0, 5).map((s, j) => (
                  <span key={j} className="mr-1">&quot;{s.text}&quot; ({s.tag?.type ?? s.tag?.label ?? '—'})</span>
                ))}
                {segments.length > 5 && <span> +{segments.length - 5} more</span>}
              </p>
            )}
          </div>
        );
      })}
      {ann.best_model && (
        <p className="text-[10px] text-gray-500 pt-1">Best model: {ann.best_model.split('/').pop()}</p>
      )}
    </div>
  );
}

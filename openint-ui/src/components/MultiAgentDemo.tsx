/**
 * Multi-Agent Demo
 * Flow: sg-agent (sentence) → sentiment-agent →
 * parallel (vectordb-agent + graph-agent) → aggregator-agent (LLM answer).
 * Debug mode: A2A-style animation + LangGraph coordination. Agents can refine if answer not satisfactory.
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { runMultiAgentDemo, getLuckySuggestion, fetchMultiAgentRecentQueries, type MultiAgentDemoResponse, type MultiAgentDemoStep, type MultiAgentHistoryEntry } from '../api';
import { AnswerRenderer } from './AnswerRenderer';

const STAGGER_MS = 380;

type Phase = 'idle' | 'sg-agent' | 'sentiment-agent' | 'vectordb-agent' | 'graph-agent' | 'enrich-agent' | 'aggregator_agent' | 'done' | 'error';

const AGENT_ORDER: Phase[] = ['sg-agent', 'sentiment-agent', 'vectordb-agent', 'graph-agent', 'enrich-agent', 'aggregator_agent'];

/** Format ISO datetime for History pane: "Jan 30, 2025, 3:45 PM" or "Today, 3:45 PM". */
function formatHistoryDateTime(iso: string): string {
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    const now = new Date();
    const sameDay = d.getDate() === now.getDate() && d.getMonth() === now.getMonth() && d.getFullYear() === now.getFullYear();
    const datePart = sameDay ? 'Today' : d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
    const timePart = d.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit', hour12: true });
    return `${datePart}, ${timePart}`;
  } catch {
    return iso;
  }
}

/** Map sentiment keywords to color theme and icon. Used when API emoji is missing or for styling. */
function getSentimentTheme(sentiment: string = '', emoji?: string): { bg: string; border: string; text: string; icon: string; bar: string } {
  const s = (sentiment || '').toLowerCase();
  if (s.includes('positive') || s.includes('hopeful') || s.includes('optimistic') || emoji === '😊' || emoji === '🙂') {
    return { bg: 'from-emerald-50 to-emerald-100/80', border: 'border-emerald-300', text: 'text-emerald-800', icon: '😊', bar: 'bg-emerald-400' };
  }
  if (s.includes('negative') || s.includes('frustrated') || (s.includes('urgent') && s.includes('concern')) || emoji === '😟' || emoji === '😕') {
    return { bg: 'from-rose-50 to-rose-100/80', border: 'border-rose-300', text: 'text-rose-800', icon: '😟', bar: 'bg-rose-400' };
  }
  if (s.includes('curious') || s.includes('exploratory') || emoji === '🔍' || emoji === '🤔') {
    return { bg: 'from-amber-50 to-amber-100/80', border: 'border-amber-300', text: 'text-amber-800', icon: '🔍', bar: 'bg-amber-400' };
  }
  if (s.includes('urgent') || s.includes('critical') || emoji === '⚡' || emoji === '🔥') {
    return { bg: 'from-orange-50 to-orange-100/80', border: 'border-orange-300', text: 'text-orange-800', icon: '⚡', bar: 'bg-orange-400' };
  }
  // Default: neutral, analytical
  return { bg: 'from-sky-50 to-sky-100/80', border: 'border-sky-300', text: 'text-sky-800', icon: emoji || '💭', bar: 'bg-sky-400' };
}

/** Dedicated Sentiment card: icon based on sentiment, labeled "Sentiment", with confidence and reasoning. */
function SentimentCard({
  sentiment,
  emoji,
  confidence,
  reasoning,
  error,
}: {
  sentiment?: string;
  emoji?: string;
  confidence?: number;
  /** Brief explanation of why this sentiment was detected */
  reasoning?: string;
  /** Error from sentiment-agent when analysis failed */
  error?: string;
}) {
  const hasData = sentiment || emoji;
  const theme = getSentimentTheme(sentiment ?? '', emoji);
  const displayIcon = emoji || theme.icon;
  const conf = typeof confidence === 'number' ? Math.round(confidence * 100) : null;
  const errorMsg = typeof error === 'string' && error.trim() ? error.trim() : null;
  return (
    <div
      className={`rounded-xl border-2 ${theme.border} bg-gradient-to-br ${theme.bg} p-4 shadow-sm`}
    >
      <div className="flex items-center gap-4">
        <div
          className={`flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-white/90 shadow-sm text-2xl ring-2 ${theme.border} ring-opacity-30`}
          title={sentiment || 'Sentiment'}
        >
          {displayIcon}
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500 mb-0.5">Sentiment</p>
          <p className={`text-sm font-medium ${theme.text} leading-snug`}>
            {hasData ? (sentiment || 'Neutral') : errorMsg ? `— ${errorMsg}` : '— Run a query above to analyze your question’s tone (Ollama required)'}
          </p>
          {conf != null && (
            <div className="mt-2 flex items-center gap-2">
              <div className="h-1.5 flex-1 max-w-24 rounded-full bg-white/70 overflow-hidden">
                <div
                  className={`h-full rounded-full ${theme.bar}`}
                  style={{ width: `${conf}%` }}
                />
              </div>
              <span className="text-[10px] font-medium text-slate-600 tabular-nums">{conf}%</span>
            </div>
          )}
        </div>
      </div>
      {reasoning && reasoning.trim() && (
        <div className="mt-3 pt-3 border-t border-white/60">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500 mb-1">Why</p>
          <p className="text-sm text-slate-600 leading-relaxed">{reasoning}</p>
        </div>
      )}
    </div>
  );
}

function getPhaseFromSteps(steps: MultiAgentDemoStep[]): Phase {
  const running = steps.find((s) => s.status === 'running');
  if (running) {
    const k = running.agent.replace(/-/g, '_') as Phase;
    if (AGENT_ORDER.includes(k)) return k;
    if (running.agent === 'aggregator-agent') return 'aggregator_agent';
    if (running.agent === 'sentiment-agent') return 'sentiment-agent';
    return 'aggregator_agent';
  }
  const failed = steps.find((s) => s.status === 'failed');
  if (failed) return 'error';
  if (steps.length > 0) return 'done';
  return 'idle';
}

export default function MultiAgentDemo() {
  const [message, setMessage] = useState('');
  const [debug, setDebug] = useState(true);
  const [phase, setPhase] = useState<Phase>('idle');
  const [data, setData] = useState<MultiAgentDemoResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [steps, setSteps] = useState<MultiAgentDemoStep[]>([]);
  const [revealedAgentTimes, setRevealedAgentTimes] = useState<Set<string>>(() => new Set());
  const staggerRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const lastRunWasLuckyRef = useRef(false);
  /** Message we sent with the last run; used as fallback when backend doesn't return original_query */
  const lastRunMessageRef = useRef<string>('');
  const [luckyLoading, setLuckyLoading] = useState(false);
  const [luckySource, setLuckySource] = useState<{ source: string; llm_model?: string; sg_agent_time_ms?: number } | null>(null);
  /** Recent questions from Redis (History pane), with optional issued_at. */
  const [historyQueries, setHistoryQueries] = useState<MultiAgentHistoryEntry[]>([]);

  const loadHistory = useCallback(async () => {
    try {
      const { queries } = await fetchMultiAgentRecentQueries();
      setHistoryQueries((prev) => (Array.isArray(queries) && queries.length > 0 ? queries : prev));
    } catch {
      // Keep previous list on fetch error (e.g. optimistic update stays)
    }
  }, []);

  const run = useCallback(async (overrideMessage?: string) => {
    lastRunWasLuckyRef.current = false;
    staggerRef.current.forEach((t) => clearTimeout(t));
    staggerRef.current = [];
    setError(null);
    setData(null);
    setSteps([]);
    setRevealedAgentTimes(new Set());
    setPhase('sg-agent');
    const q = String(overrideMessage != null ? overrideMessage : message ?? '').trim();
    lastRunMessageRef.current = q;
    try {
      const result = await runMultiAgentDemo(q, debug, { from_lucky: false });
      setData(result);
      setSteps(result.steps || []);
      setPhase(result.success ? 'done' : 'error');
      if (!result.success) setError(result.error || 'Run failed');
      if (result.success) {
        const questionRun = String(result.original_query ?? result.sentence ?? q ?? '').trim();
        if (questionRun) {
          const issuedAt = new Date().toISOString();
          setHistoryQueries((prev) =>
            prev[0]?.query === questionRun ? prev : [{ query: questionRun, issued_at: issuedAt }, ...prev.filter((x) => x.query !== questionRun)]
          );
        }
        loadHistory();
      }
    } catch (e) {
      setPhase('error');
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [message, debug, loadHistory]);

  const handleLucky = useCallback(async () => {
    setLuckyLoading(true);
    setError(null);
    setLuckySource(null);
    try {
      const luckyData = await getLuckySuggestion();
      const sentence = luckyData.sentence?.trim();
      const sgTimeMs = luckyData.sg_agent_time_ms ?? undefined;
      if (!sentence) {
        setLuckyLoading(false);
        return;
      }
      setLuckySource({
        source: luckyData.source ?? 'template',
        llm_model: luckyData.llm_model,
        sg_agent_time_ms: sgTimeMs,
      });
      setMessage(sentence);
      lastRunWasLuckyRef.current = true;
      staggerRef.current.forEach((t) => clearTimeout(t));
      staggerRef.current = [];
      setData(null);
      setError(null);
      // Activate animation immediately: sg-agent done with same time as "Generated with …"
      setSteps([
        {
          agent: 'sg-agent',
          action: 'use_lucky_sentence',
          status: 'completed',
          duration_ms: sgTimeMs,
          sentence: sentence.slice(0, 200),
        },
      ]);
      setRevealedAgentTimes(new Set(['sg-agent']));
      setPhase('sentiment-agent');
      lastRunMessageRef.current = sentence;
      const result = await runMultiAgentDemo(sentence, debug, {
        from_lucky: true,
        sg_agent_time_ms: sgTimeMs,
      });
      setData(result);
      setSteps(result.steps || []);
      setPhase(result.success ? 'done' : 'error');
      if (!result.success) setError(result.error || 'Run failed');
      if (result.success) {
        const questionRun = String(result.original_query ?? result.sentence ?? sentence ?? '').trim();
        if (questionRun) {
          const issuedAt = new Date().toISOString();
          setHistoryQueries((prev) =>
            prev[0]?.query === questionRun ? prev : [{ query: questionRun, issued_at: issuedAt }, ...prev.filter((x) => x.query !== questionRun)]
          );
        }
        loadHistory();
      }
    } catch (e) {
      setPhase('error');
      setError(e instanceof Error ? e.message : 'Failed to get suggestion or run demo');
    } finally {
      setLuckyLoading(false);
    }
  }, [debug, loadHistory]);

  const isRunning = phase !== 'idle' && phase !== 'done' && phase !== 'error';
  const currentPhase = data ? getPhaseFromSteps(data.steps || []) : phase;
  const rawOriginal = data?.original_query;
  const originalDisplay: string =
    data?.success && data?.answer
      ? (typeof rawOriginal === 'string' && rawOriginal.trim() !== ''
          ? rawOriginal.trim()
          : (lastRunMessageRef.current ?? ''))
      : '';
  const generatedDisplay: string =
    typeof data?.sentence === 'string' ? data.sentence.trim() : '';

  // Staggered reveal in exact execution order: reveal each step's time one by one (backend steps order)
  useEffect(() => {
    staggerRef.current.forEach((t) => clearTimeout(t));
    staggerRef.current = [];
    if (phase !== 'done' || steps.length === 0) {
      if (phase === 'idle' || phase === 'error') setRevealedAgentTimes(new Set());
      return;
    }
    steps.forEach((step, i) => {
      const agent = step.agent;
      const t = setTimeout(() => {
        setRevealedAgentTimes((prev) => new Set([...prev, agent]));
      }, i * STAGGER_MS);
      staggerRef.current.push(t);
    });
    return () => {
      staggerRef.current.forEach((t) => clearTimeout(t));
      staggerRef.current = [];
    };
  }, [phase, steps]);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  const fillAndRun = useCallback((q: string) => {
    setMessage(q);
    run(q);
  }, [run]);

  return (
    <div className="w-full min-h-0 flex flex-col">
      <div className="flex items-start justify-between gap-4 mb-3">
        <div className="min-w-0">
          <h1 className="text-xl font-semibold text-gray-900">Multi-Agent Demo</h1>
          <p className="text-xs text-gray-600 mt-0.5">
            sg-agent → sentiment-agent → parallel (vectordb + graph) → aggregator-agent. LangGraph orchestrates.
          </p>
        </div>
      </div>

      <div className="flex flex-col md:flex-row gap-4 flex-1 min-h-0">
        {/* Main: input/flow at top, then results — on small screens below History (order-2); on md+ left (order-1) */}
        <div className="flex-1 min-w-0 flex flex-col gap-4 order-2 md:order-1">
      {/* Input, Debug, Retrieval query at top (order-1) */}
      <div className="order-1 rounded-xl border border-surface-200 bg-white shadow-sm p-4 sm:p-5 flex-1 min-h-0 flex flex-col">
        {/* Debug toggle at top right */}
        <div className="flex justify-end mb-3">
          <label className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer select-none">
            <span className="relative inline-flex h-6 w-11 shrink-0 rounded-full bg-gray-200 transition-colors focus-within:ring-2 focus-within:ring-brand-500 focus-within:ring-offset-2 aria-disabled:opacity-50" aria-disabled={isRunning}>
              <input
                type="checkbox"
                checked={debug}
                onChange={(e) => setDebug(e.target.checked)}
                disabled={isRunning}
                className="sr-only peer"
                aria-label="Debug (show animation)"
              />
              <span className="pointer-events-none absolute inset-0 rounded-full transition-colors peer-checked:bg-brand-500 peer-disabled:pointer-events-none" />
              <span className="pointer-events-none absolute left-1 top-1 h-4 w-4 rounded-full bg-white shadow ring-0 transition peer-checked:translate-x-5 peer-disabled:opacity-50" />
            </span>
            <span>Debug</span>
          </label>
        </div>
        {/* Textbox, Run, I'm feeling lucky in one row */}
        <div className="flex items-stretch gap-3 flex-wrap">
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                run();
              }
            }}
            placeholder="Ask a question (e.g. show me disputes or wire transfers)"
            rows={2}
            className="flex-1 min-w-[200px] rounded-lg border border-surface-200 px-4 py-2.5 text-sm bg-white focus:ring-2 focus:ring-brand-400 focus:border-brand-400 resize-y"
            disabled={isRunning}
            aria-label="Question for multi-agent demo"
          />
          <button
            type="button"
            onClick={() => run()}
            disabled={isRunning}
            className="shrink-0 px-5 py-2.5 rounded-lg bg-brand-500 text-white text-sm font-medium hover:bg-brand-600 disabled:opacity-50 disabled:pointer-events-none transition-colors"
          >
            {isRunning ? 'Running…' : 'Run'}
          </button>
          <button
            type="button"
            onClick={handleLucky}
            disabled={isRunning || luckyLoading}
            className="shrink-0 px-4 py-2.5 rounded-lg bg-amber-500 text-white text-sm font-medium hover:bg-amber-600 disabled:opacity-50 disabled:pointer-events-none transition-colors whitespace-nowrap"
            title="Get a random sentence from sg-agent and run the multi-agent demo"
          >
            {luckyLoading ? '…' : "I'm feeling lucky!"}
          </button>
        </div>
        {luckySource?.llm_model && (
          <p className="text-xs text-gray-500 flex items-center gap-1.5 mt-1" role="status">
            <span className="inline-flex items-center justify-center w-4 h-4 rounded bg-gray-100 text-gray-400" aria-hidden>◇</span>
            Generated with <span className="font-medium text-gray-600">{luckySource.llm_model}</span>
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

        {/* Retrieval query (sg-agent: LLM-improved question for vector + graph) — manual run only */}
        {data?.success && data?.sentence && !lastRunWasLuckyRef.current && (
          <div className="mt-3 rounded-xl border border-slate-200 bg-slate-800 shadow-sm overflow-hidden">
            <div className="px-4 py-2 border-b border-slate-600 bg-slate-700/50">
              <p className="text-xs font-semibold text-slate-300 uppercase tracking-wider">Retrieval query (sg-agent)</p>
            </div>
            <pre className="p-4 text-xs font-mono text-slate-200 leading-relaxed overflow-x-auto whitespace-pre-wrap break-words">
              {String(data.sentence ?? '').trim()}
            </pre>
          </div>
        )}

        {/* LangGraph + agent flow: compact, fits viewport when Debug is on */}
        {debug && (
          <>
            <div className="mt-4 pt-4 border-t border-surface-200 flex-shrink-0">
              <div
                className={`relative flex flex-col items-center gap-2 mb-4 rounded-lg border px-4 py-2.5 transition-all duration-300 ${
                  currentPhase !== 'idle' && currentPhase !== 'error'
                    ? 'border-amber-200 bg-amber-50/50'
                    : 'border-amber-100 bg-amber-50/30'
                }`}
                role="region"
                aria-label="LangGraph orchestration"
              >
                <div className="absolute right-2 top-2 opacity-[0.12] pointer-events-none" aria-hidden>
                  <AgentHeroIcon className="w-8 h-8 text-brand-500" />
                </div>
                <div className="flex items-center gap-1.5">
                  <LangGraphBadgeIcon />
                  <span className="text-[10px] font-medium uppercase tracking-wider text-amber-700/90">LangGraph</span>
                </div>
                <div className="flex items-center gap-2 flex-wrap justify-center">
                  <LangGraphNode label="select_agents" active={currentPhase === 'sg-agent'} done={steps.length > 0} />
                  <LangGraphArrow active={isRunning} />
                  <LangGraphNode label="run_agents" active={isRunning} done={phase === 'done' || phase === 'error'} />
                  <LangGraphArrow active={isRunning} />
                  <LangGraphNode label="aggregate" active={false} done={phase === 'done' || phase === 'error'} />
                </div>
              </div>

              <div className="relative flex flex-wrap items-center justify-center gap-2 sm:gap-3 py-4 px-3 sm:py-5 sm:px-4 bg-gray-50/80 rounded-lg border border-surface-200">
              <div className="absolute right-2 top-1/2 -translate-y-1/2 opacity-[0.1] pointer-events-none" aria-hidden>
                <AgentHeroIcon className="w-10 h-10 text-brand-500" />
              </div>
              <AgentCard
                agent="sg-agent"
                name="sg-agent"
                desc="Sentence"
                step={(() => {
                  const sgStep = steps.find((s) => s.agent === 'sg-agent');
                  if (lastRunWasLuckyRef.current && luckySource?.sg_agent_time_ms != null && sgStep)
                    return { ...sgStep, duration_ms: luckySource.sg_agent_time_ms };
                  return sgStep;
                })()}
                showTime={revealedAgentTimes.has('sg-agent')}
              />
              <AnimatedConnector active={isRunning} color="violet" />
              <AgentCard agent="sentiment-agent" name="sentiment-agent" desc="Sentiment" step={steps.find((s) => s.agent === 'sentiment-agent')} showTime={revealedAgentTimes.has('sentiment-agent')} />
              <AnimatedConnector active={isRunning} color="pink" />
              {/* Parallel: vectordb + graph */}
              <ForkTwoLines active={isRunning} />
              <div className="flex flex-col rounded-lg border-2 border-dashed border-slate-300 bg-slate-50/60 px-2 py-2">
                <span className="text-[9px] font-semibold text-slate-500 uppercase tracking-wider mb-1 text-center">parallel (vectordb + graph)</span>
                <div className="flex items-center gap-2">
                  <div className="rounded border border-sky-200/80 bg-white/80 px-1.5 py-1 min-w-0">
                    <AgentCard agent="vectordb-agent" name="vectordb-agent" desc="Vector search" step={steps.find((s) => s.agent === 'vectordb-agent')} showTime={revealedAgentTimes.has('vectordb-agent')} />
                  </div>
                  <div className="rounded border border-amber-200/80 bg-white/80 px-1.5 py-1 min-w-0">
                    <AgentCard agent="graph-agent" name="graph-agent" desc="Neo4j Cypher" step={steps.find((s) => s.agent === 'graph-agent')} showTime={revealedAgentTimes.has('graph-agent')} />
                  </div>
                </div>
              </div>
              <EnrichCollaborationConnector active={isRunning} enrichStep={steps.find((s) => s.agent === 'enrich-agent')} />
              <AgentCard agent="enrich-agent" name="enrich-agent" desc="ID lookup" step={steps.find((s) => s.agent === 'enrich-agent')} showTime={revealedAgentTimes.has('enrich-agent')} />
              <AnimatedConnector active={isRunning} color="emerald" />
              <Arrow active={isRunning} />
              <AgentCard agent="aggregator-agent" name="aggregator-agent" desc="Aggregate & LLM" step={steps.find((s) => s.agent === 'aggregator-agent')} showTime={revealedAgentTimes.has('aggregator-agent')} />
              </div>
            </div>
          </>
        )}

        {phase === 'error' && error && <p className="mt-4 text-sm text-rose-600" role="alert">{error}</p>}
      </div>

      {/* Results (order-2) */}
      {data?.success && (data?.answer != null || data?.sentence != null) && (
        <div className="order-2 result-card-enter rounded-2xl border border-slate-200/90 bg-white shadow-lg shadow-slate-200/50 overflow-hidden ring-1 ring-slate-100">
          <div className="px-5 py-3 border-b border-slate-200/80 bg-gradient-to-r from-slate-50 to-slate-100/80 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-slate-700 text-white text-xs font-bold shadow-sm">✓</span>
              <span className="text-sm font-semibold text-slate-700 tracking-tight">Result</span>
              {(data.steps?.length ?? 0) > 0 && (
                <span className="text-xs font-medium text-slate-500 bg-slate-200/60 px-2 py-0.5 rounded-md">{data.steps?.length ?? 0} steps</span>
              )}
            </div>
          </div>
          <div className="p-5 sm:p-6 space-y-6">
            {(originalDisplay || generatedDisplay) && (
              <section className="space-y-3" aria-label="Query comparison">
                <div className="flex items-center gap-2">
                  <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-slate-600 text-white text-xs font-bold shadow-sm">1</span>
                  <h3 className="text-sm font-semibold text-slate-700 tracking-tight">Query comparison</h3>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="rounded-xl border border-amber-200/60 bg-gradient-to-br from-amber-50 to-amber-50/50 p-4 shadow-sm">
                    <div className="flex items-baseline justify-between gap-2 mb-2">
                      <span className="text-[10px] font-semibold uppercase tracking-wider text-amber-700">Original</span>
                      {(originalDisplay?.length ?? 0) > 0 && (
                        <span className="text-[10px] font-mono text-amber-600/80 tabular-nums">{originalDisplay?.length ?? 0} ch</span>
                      )}
                    </div>
                    <p className="text-sm text-slate-800 whitespace-pre-wrap break-words leading-relaxed">{originalDisplay || '—'}</p>
                  </div>
                  <div className="rounded-xl border border-emerald-200/60 bg-gradient-to-br from-emerald-50 to-emerald-50/50 p-4 shadow-sm">
                    <div className="flex items-baseline justify-between gap-2 mb-2">
                      <span className="text-[10px] font-semibold uppercase tracking-wider text-emerald-700">Generated (sg-agent)</span>
                      {generatedDisplay.length > 0 && (
                        <span className="text-[10px] font-mono text-emerald-600/80 tabular-nums">{generatedDisplay.length} ch</span>
                      )}
                    </div>
                    <p className="text-sm text-slate-800 whitespace-pre-wrap break-words leading-relaxed">{generatedDisplay || '—'}</p>
                  </div>
                </div>
              </section>
            )}
            <section className="space-y-3" aria-label="Sentiment">
              <div className="flex items-center gap-2">
                <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-pink-500 text-white text-xs font-bold shadow-sm">2</span>
                <div>
                  <h3 className="text-sm font-semibold text-slate-700 tracking-tight">Sentiment</h3>
                  <p className="text-xs text-slate-500 mt-0.5">Tone of your question (e.g. curious, urgent, neutral)</p>
                </div>
              </div>
              <SentimentCard
                sentiment={data.sentiment?.sentiment ?? steps.find((s) => s.agent === 'sentiment-agent')?.sentiment}
                emoji={data.sentiment?.emoji ?? steps.find((s) => s.agent === 'sentiment-agent')?.emoji}
                confidence={data.sentiment?.confidence ?? steps.find((s) => s.agent === 'sentiment-agent')?.confidence}
                reasoning={data.sentiment?.reasoning ?? steps.find((s) => s.agent === 'sentiment-agent')?.reasoning}
                error={steps.find((s) => s.agent === 'sentiment-agent')?.error}
              />
            </section>
            {(data?.enriched_entities?.length ?? 0) > 0 && (
              <section className="space-y-3" aria-label="Enriched entities">
                <div className="flex items-center gap-2">
                  <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-cyan-500 text-white shadow-sm">
                    <EnrichAgentIcon className="w-4 h-4" />
                  </span>
                  <div>
                    <h3 className="text-sm font-semibold text-slate-700 tracking-tight">Enriched by enrich-agent</h3>
                    <p className="text-xs text-slate-500 mt-0.5">IDs looked up in Neo4j (highlighted in answer below)</p>
                  </div>
                </div>
                <div className="rounded-xl border border-cyan-200/60 bg-gradient-to-br from-cyan-50 to-cyan-50/50 p-4 shadow-sm">
                  <div className="flex flex-wrap gap-2">
                    {(data.enriched_entities ?? []).map((eid) => {
                      const detail = data.enriched_details?.[eid];
                      const label = (detail?.label ?? 'Entity').toLowerCase();
                      const isCustomer = label === 'customer';
                      const isTransaction = label === 'transaction';
                      const isDispute = label === 'dispute';
                      const iconBg = isCustomer ? 'bg-emerald-500' : isTransaction ? 'bg-amber-500' : isDispute ? 'bg-rose-500' : 'bg-cyan-500';
                      const displayText = detail?.display_name || `${detail?.label ?? 'Entity'} ${eid}`;
                      return (
                        <span
                          key={eid}
                          className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-white/90 border border-cyan-200 text-sm text-cyan-800"
                        >
                          <span className={`flex h-5 w-5 shrink-0 items-center justify-center rounded ${iconBg} text-white`}>
                            {isCustomer && <EnrichEntityCustomerIcon className="w-3 h-3" />}
                            {isTransaction && <EnrichEntityTransactionIcon className="w-3 h-3" />}
                            {isDispute && <EnrichEntityDisputeIcon className="w-3 h-3" />}
                            {!isCustomer && !isTransaction && !isDispute && <EnrichAgentIcon className="w-3 h-3" />}
                          </span>
                          <span>{displayText}</span>
                        </span>
                      );
                    })}
                  </div>
                </div>
              </section>
            )}
            <section className="space-y-3" aria-label="Answer">
              <div className="flex items-center gap-2">
                <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-brand-500 text-white text-xs font-bold shadow-sm">{data?.enriched_entities?.length ? '4' : '3'}</span>
                <h3 className="text-sm font-semibold text-slate-700 tracking-tight">Answer</h3>
                <span className="text-xs font-medium text-slate-400 bg-slate-100 px-2 py-0.5 rounded-md tabular-nums">{String(data?.answer ?? '').length} ch</span>
              </div>
              <div className="rounded-xl border border-slate-200/80 bg-gradient-to-br from-slate-50 via-white to-slate-50/50 p-5 sm:p-6 shadow-sm min-h-[4rem]">
                <div className="text-[15px] sm:text-base leading-[1.7] max-w-[75ch]">
                  <AnswerRenderer
                    text={data?.answer ?? ''}
                    enrichedIds={data?.enriched_entities ? new Set(data.enriched_entities) : undefined}
                    enrichedDetails={data?.enriched_details}
                  />
                </div>
              </div>
            </section>
          </div>
        </div>
      )}

      {debug && data && (
        <div className="order-3 rounded-xl border border-surface-200 bg-white shadow-sm overflow-hidden">
          <h2 className="text-sm font-semibold text-slate-600 px-4 py-3 border-b border-surface-200 bg-surface-50/80">Debug: Vector & Graph results</h2>
          <div className="p-4 space-y-4">
            {/* Retrieval query used for BOTH vector and graph lookups */}
            {data.sentence && (
              <div>
                <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">Retrieval query (used for both lookups)</p>
                <p className="text-sm text-slate-700 bg-slate-50 px-3 py-2 rounded border border-slate-200 font-medium">&ldquo;{data.sentence}&rdquo;</p>
              </div>
            )}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Vector search results ({data.vector_results?.length ?? 0})</p>
                {(data.vector_results?.length ?? 0) > 0 ? (
                  <ul className="text-xs text-slate-700 space-y-1 max-h-40 overflow-y-auto">
                    {(data.vector_results || []).slice(0, 5).map((r, i) => (
                      <li key={i} className="break-words" title={r.content}>
                        {r.content?.slice(0, 150)}{(r.content?.length ?? 0) > 150 ? '…' : ''}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-xs text-slate-500 italic">No vector results.</p>
                )}
              </div>
              <div>
                <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Graph (Neo4j Cypher) results</p>
                {data.graph_results?.cypher && (
                  <pre className="text-[11px] font-mono bg-slate-100 p-2 rounded overflow-x-auto mb-2">{data.graph_results.cypher}</pre>
                )}
                {data.graph_results?.error && (
                  <p className="text-xs text-rose-600 bg-rose-50 px-2 py-1 rounded mb-2">{data.graph_results.error}</p>
                )}
                <p className="text-xs text-slate-600">{data.graph_results?.rows?.length ?? 0} rows</p>
                {(data.graph_results?.rows?.length ?? 0) > 0 && data.graph_results?.rows?.[0] && typeof data.graph_results.rows[0] === 'object' ? (
                  <pre className="text-[10px] font-mono bg-slate-50 p-2 rounded mt-1 overflow-x-auto max-h-24">
                    {JSON.stringify(data.graph_results.rows[0] as Record<string, unknown>, null, 2)}
                  </pre>
                ) : null}
              </div>
            </div>
          </div>
        </div>
      )}
        </div>

        {/* Right pane: History (~1/3, Redis-cached) — numbered list with date/time */}
        <aside className="w-full md:w-1/3 md:min-w-[280px] shrink-0 order-1 md:order-2 md:sticky md:top-24 self-start max-h-[calc(100vh-6rem)] overflow-y-auto">
          <div className="rounded-xl border border-surface-200 bg-white shadow-soft overflow-visible">
            <h2 className="px-4 py-3 border-b border-surface-200 text-sm font-semibold text-gray-900 bg-gradient-to-r from-surface-50 to-white sticky top-0 bg-white z-10 flex items-center gap-2">
              <HistoryIcon className="w-4 h-4 text-slate-500" />
              History
            </h2>
            <div className="p-4">
              {(historyQueries ?? []).length > 0 ? (
                <ol className="space-y-3 list-none pl-0">
                  {(historyQueries ?? []).map((entry, index) => (
                    <li key={`${index}-${entry.query.slice(0, 40)}`} className="flex gap-3 items-start">
                      <span
                        className="shrink-0 flex h-6 w-6 items-center justify-center rounded-md bg-slate-200 text-slate-700 text-xs font-semibold tabular-nums"
                        aria-hidden
                      >
                        {index + 1}
                      </span>
                      <div className="min-w-0 flex-1">
                        {entry.issued_at != null && (
                          <time
                            dateTime={entry.issued_at}
                            className="block text-[10px] font-medium text-slate-500 uppercase tracking-wider mb-1"
                          >
                            {formatHistoryDateTime(entry.issued_at)}
                          </time>
                        )}
                        <button
                          type="button"
                          onClick={() => fillAndRun(entry.query)}
                          disabled={isRunning}
                          className="w-full text-left px-3 py-2 text-xs rounded-lg bg-slate-50 border border-slate-200 text-brand-600 hover:bg-slate-100 hover:border-slate-300 disabled:opacity-50 whitespace-pre-wrap break-words"
                        >
                          {entry.query}
                        </button>
                      </div>
                    </li>
                  ))}
                </ol>
              ) : (
                <p className="text-sm text-gray-500">Questions you run will appear here (cached in Redis).</p>
              )}
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}

/** Clock/history icon for History pane. */
function HistoryIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <circle cx="12" cy="12" r="10" />
      <path d="M12 6v6l4 2" />
    </svg>
  );
}

/** Subtle agent icon inspired by hero logo (rounded rect + circle + bar). */
function AgentHeroIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 96 96" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
      <rect x="8" y="8" width="80" height="80" rx="20" fill="currentColor" />
      <circle cx="36" cy="48" r="14" fill="white" fillOpacity="0.9" />
      <rect x="58" y="28" width="8" height="40" rx="4" fill="white" fillOpacity="0.9" />
    </svg>
  );
}

/** Entity icons for enriched section (Customer, Transaction, Dispute) */
function EnrichEntityCustomerIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <circle cx="12" cy="8" r="3.5" />
      <path d="M5 20c0-3.5 3.1-6 7-6s7 2.5 7 6" />
    </svg>
  );
}
function EnrichEntityTransactionIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="2" y="6" width="20" height="12" rx="2" />
      <path d="M12 12h.01" strokeWidth="2" />
      <path d="M7 12h.01" strokeWidth="2" />
      <path d="M17 12h.01" strokeWidth="2" />
    </svg>
  );
}
function EnrichEntityDisputeIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z" />
      <path d="M4 22v-7" />
    </svg>
  );
}

/** Sparkles icon for enrich-agent: suggests adding context and value from graph lookups. */
function EnrichAgentIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M12 3l1.5 4.5L18 9l-4.5 1.5L12 15l-1.5-4.5L6 9l4.5-1.5L12 3z" />
      <path d="M5 16l.75 2.25L8 19l-2.25.75L5 22l-.75-2.25L2 19l2.25-.75L5 16z" />
      <path d="M19 6l.5 1.5L21 8l-1.5.5L19 10l-.5-1.5L17 8l1.5-.5L19 6z" />
    </svg>
  );
}

function LangGraphBadgeIcon() {
  return (
    <svg className="w-4 h-4 text-amber-600/90" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <circle cx="12" cy="12" r="3" />
      <path d="M12 2v2M12 20v2M2 12h2M20 12h2" />
      <path d="M5.64 5.64l1.42 1.42M16.94 16.94l1.42 1.42M5.64 18.36l1.42-1.42M16.94 7.06l1.42-1.42" />
    </svg>
  );
}

function LangGraphArrow({ active }: { active: boolean }) {
  return <span className={`inline-flex items-center text-amber-500/90 transition-opacity ${active ? 'opacity-100' : 'opacity-50'}`} aria-hidden>→</span>;
}

function LangGraphNode({ label, active, done }: { label: string; active: boolean; done: boolean }) {
  const base = 'text-[10px] font-medium px-2 py-1 rounded border transition-all duration-200';
  const style = active ? 'border-amber-400 bg-amber-50 text-amber-800' : done ? 'border-amber-200 bg-amber-50/60 text-amber-700' : 'border-amber-100 bg-white/80 text-amber-700/80';
  return <span className={`${base} ${style} ${active ? 'ring-1 ring-amber-300/50' : ''}`}>{label}</span>;
}

/** Animated connector between agents (compact). */
function AnimatedConnector({ active, color }: { active: boolean; color: 'violet' | 'teal' | 'emerald' | 'pink' }) {
  const dotClass =
    color === 'violet' ? 'bg-violet-400' : color === 'teal' ? 'bg-teal-400' : color === 'pink' ? 'bg-pink-400' : 'bg-emerald-400';
  const dotAnim = active ? 'a2a-flow-dot' : '';
  return (
    <div className="flex-1 min-w-[24px] max-w-[36px] h-12 flex items-center relative overflow-hidden">
      <div className="absolute inset-0 flex items-center border-t border-dashed border-gray-300" aria-hidden />
      <span className={`absolute top-1/2 -translate-y-1/2 left-0 w-1 h-1 rounded-full ${dotClass} opacity-90 ${dotAnim}`} />
      <span className={`absolute top-1/2 -translate-y-1/2 left-0 w-1 h-1 rounded-full ${dotClass} opacity-90 ${dotAnim} a2a-flow-dot-2`} />
      <span className={`absolute top-1/2 -translate-y-1/2 left-0 w-1 h-1 rounded-full ${dotClass} opacity-90 ${dotAnim} a2a-flow-dot-3`} />
    </div>
  );
}

/** Enrich-agent collaboration: shows data flow from vectordb + graph into enrich */
function EnrichCollaborationConnector({
  active: _active,
  enrichStep,
}: {
  active: boolean;
  enrichStep?: MultiAgentDemoStep;
}) {
  const isEnrichActive = enrichStep?.status === 'running';
  const isEnrichDone = enrichStep?.status === 'completed';
  const showCollab = isEnrichActive || isEnrichDone;
  const animateDots = showCollab; /* pulse when collaborating or just collaborated */
  const dotAnim = animateDots ? 'enrich-collab-dot' : '';
  const dotAnim2 = animateDots ? 'enrich-collab-dot enrich-collab-dot-2' : '';
  const dotAnim3 = animateDots ? 'enrich-collab-dot enrich-collab-dot-3' : '';
  return (
    <div className="flex flex-col items-center justify-center shrink-0 min-w-[52px] px-1" role="group" aria-label="enrich-agent collaborating with vectordb and graph">
      <div className="flex items-center gap-2 mb-1">
        <div className="flex items-center gap-0.5" title="from vectordb-agent">
          <span className={`w-1.5 h-1.5 rounded-full bg-sky-400 shrink-0 ${dotAnim} ${isEnrichDone ? 'opacity-80' : 'opacity-50'}`} />
          <span className={`w-1.5 h-1.5 rounded-full bg-sky-400 shrink-0 ${dotAnim2} ${isEnrichDone ? 'opacity-80' : 'opacity-50'}`} />
          <span className={`w-1.5 h-1.5 rounded-full bg-sky-400 shrink-0 ${dotAnim3} ${isEnrichDone ? 'opacity-80' : 'opacity-50'}`} />
          <span className="text-[8px] font-medium text-sky-600 ml-0.5">vec</span>
        </div>
        <span className="text-slate-400 text-xs font-bold">+</span>
        <div className="flex items-center gap-0.5" title="from graph-agent">
          <span className={`w-1.5 h-1.5 rounded-full bg-amber-400 shrink-0 ${dotAnim} ${isEnrichDone ? 'opacity-80' : 'opacity-50'}`} />
          <span className={`w-1.5 h-1.5 rounded-full bg-amber-400 shrink-0 ${dotAnim2} ${isEnrichDone ? 'opacity-80' : 'opacity-50'}`} />
          <span className={`w-1.5 h-1.5 rounded-full bg-amber-400 shrink-0 ${dotAnim3} ${isEnrichDone ? 'opacity-80' : 'opacity-50'}`} />
          <span className="text-[8px] font-medium text-amber-600 ml-0.5">graph</span>
        </div>
      </div>
      <div className="flex flex-col items-center w-full">
        <div className="h-3 w-px bg-gradient-to-b from-cyan-300/70 to-cyan-500/90" aria-hidden />
        <span className={`text-[9px] font-semibold mt-0.5 ${isEnrichActive ? 'text-cyan-600 animate-pulse' : isEnrichDone ? 'text-cyan-700' : 'text-slate-500'}`}>
          {isEnrichActive ? 'collaborating…' : isEnrichDone ? '✓ collaborated' : '← IDs'}
        </span>
      </div>
    </div>
  );
}

/** Two lines from sentiment to vectordb and graph (compact). */
function ForkTwoLines({ active }: { active: boolean }) {
  const dotAnim = active ? 'a2a-flow-dot' : '';
  return (
    <div className="flex flex-col justify-center gap-2 shrink-0 min-w-[28px] w-8">
      <div className="flex-1 min-h-[16px] flex items-center relative overflow-hidden">
        <div className="absolute inset-0 flex items-center border-t border-dashed border-teal-300" aria-hidden />
        <span className={`absolute top-1/2 -translate-y-1/2 left-0 w-1 h-1 rounded-full bg-sky-400 opacity-90 ${dotAnim}`} />
        <span className={`absolute top-1/2 -translate-y-1/2 left-0 w-1 h-1 rounded-full bg-sky-400 opacity-90 ${dotAnim} a2a-flow-dot-2`} />
        <span className={`absolute top-1/2 -translate-y-1/2 left-0 w-1 h-1 rounded-full bg-sky-400 opacity-90 ${dotAnim} a2a-flow-dot-3`} />
      </div>
      <div className="flex-1 min-h-[16px] flex items-center relative overflow-hidden">
        <div className="absolute inset-0 flex items-center border-t border-dashed border-teal-300" aria-hidden />
        <span className={`absolute top-1/2 -translate-y-1/2 left-0 w-1 h-1 rounded-full bg-amber-400 opacity-90 ${dotAnim}`} />
        <span className={`absolute top-1/2 -translate-y-1/2 left-0 w-1 h-1 rounded-full bg-amber-400 opacity-90 ${dotAnim} a2a-flow-dot-2`} />
        <span className={`absolute top-1/2 -translate-y-1/2 left-0 w-1 h-1 rounded-full bg-amber-400 opacity-90 ${dotAnim} a2a-flow-dot-3`} />
      </div>
    </div>
  );
}

/** Arrow between agents (compact). */
function Arrow({ active }: { active: boolean }) {
  return (
    <div className={`flex flex-col items-center justify-center shrink-0 transition-opacity duration-300 ${active ? 'opacity-100' : 'opacity-50'}`}>
      <svg
        className={`w-6 h-6 transition-all duration-300 ${active ? 'text-brand-500 animate-[pulse_1s_ease-in-out_infinite]' : 'text-gray-400'}`}
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

const AGENT_STYLES: Record<
  string,
  { idle: string; active: string; done: string; failed: string; badge: string; badgeDone: string; time: string }
> = {
  'sg-agent': {
    idle: 'border-violet-200 bg-white',
    active: 'border-violet-500 bg-violet-50 shadow shadow-violet-200/40',
    done: 'border-violet-400 bg-violet-50/90',
    failed: 'border-rose-400 bg-rose-50/80',
    badge: 'bg-violet-500 text-white',
    badgeDone: 'bg-violet-600 text-white',
    time: 'text-violet-600',
  },
  'vectordb-agent': {
    idle: 'border-sky-200 bg-white',
    active: 'border-sky-500 bg-sky-50 shadow shadow-sky-200/40',
    done: 'border-sky-400 bg-sky-50/90',
    failed: 'border-rose-400 bg-rose-50/80',
    badge: 'bg-sky-500 text-white',
    badgeDone: 'bg-sky-600 text-white',
    time: 'text-sky-600',
  },
  'graph-agent': {
    idle: 'border-amber-200 bg-white',
    active: 'border-amber-500 bg-amber-50 shadow shadow-amber-200/40',
    done: 'border-amber-400 bg-amber-50/90',
    failed: 'border-rose-400 bg-rose-50/80',
    badge: 'bg-amber-500 text-white',
    badgeDone: 'bg-amber-600 text-white',
    time: 'text-amber-600',
  },
  'aggregator-agent': {
    idle: 'border-emerald-200 bg-white',
    active: 'border-emerald-500 bg-emerald-50 shadow shadow-emerald-200/40',
    done: 'border-emerald-400 bg-emerald-50/90',
    failed: 'border-rose-400 bg-rose-50/80',
    badge: 'bg-emerald-500 text-white',
    badgeDone: 'bg-emerald-600 text-white',
    time: 'text-emerald-600',
  },
  'sentiment-agent': {
    idle: 'border-pink-200 bg-white',
    active: 'border-pink-500 bg-pink-50 shadow shadow-pink-200/40',
    done: 'border-pink-400 bg-pink-50/90',
    failed: 'border-rose-400 bg-rose-50/80',
    badge: 'bg-pink-500 text-white',
    badgeDone: 'bg-pink-600 text-white',
    time: 'text-pink-600',
  },
  'enrich-agent': {
    idle: 'border-cyan-200 bg-white',
    active: 'border-cyan-500 bg-cyan-50 shadow shadow-cyan-200/40',
    done: 'border-cyan-400 bg-cyan-50/90',
    failed: 'border-rose-400 bg-rose-50/80',
    badge: 'bg-cyan-500 text-white',
    badgeDone: 'bg-cyan-600 text-white',
    time: 'text-cyan-600',
  },
};

function AgentCard({
  agent,
  name,
  desc,
  step,
  showTime = false,
}: {
  agent: string;
  name: string;
  desc: string;
  step?: MultiAgentDemoStep;
  showTime?: boolean;
}) {
  const status = step?.status ?? 'idle';
  const active = status === 'running';
  const done = status === 'completed';
  const failed = status === 'failed';
  const styles = AGENT_STYLES[agent] ?? AGENT_STYLES['sg-agent'];
  const boxClass = active ? styles.active : failed ? styles.failed : done ? styles.done : styles.idle;
  const badgeClass = active ? `${styles.badge} animate-pulse` : done ? styles.badgeDone : failed ? 'bg-rose-500 text-white' : 'bg-gray-200 text-gray-600';
  const shortName = name.replace(/-agent$/, '') || name;
  const showDuration = showTime && step?.duration_ms != null;
  const badgeIcon = agent === 'sentiment-agent'
    ? (step?.emoji || '💬')
    : agent === 'vectordb-agent'
      ? '🔎'
      : agent === 'graph-agent'
        ? '🕸️'
        : agent === 'aggregator-agent'
          ? '📋'
          : agent === 'enrich-agent'
            ? <EnrichAgentIcon className="w-3.5 h-3.5" />
            : agent === 'sg-agent'
              ? '✏️'
              : null;
  return (
    <div
      className={`flex flex-col items-center justify-center rounded-lg border-2 px-2.5 py-2 min-w-[72px] max-w-[100px] transition-all duration-300 ${boxClass}`}
    >
      <div className={`w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold mb-1 ${badgeClass}`} title={agent === 'sentiment-agent' && step?.sentiment ? step.sentiment : undefined}>
        {badgeIcon || shortName.slice(0, 2).toUpperCase()}
      </div>
      <span className="text-[10px] font-semibold text-gray-800 leading-tight">{shortName}</span>
      <span className="text-[9px] text-gray-500 mt-0.5 leading-tight">{desc}</span>
      {showDuration && step?.duration_ms != null && (
        <span className={`mt-1 text-[10px] font-semibold font-mono tabular-nums ${styles.time}`} title={`${step.duration_ms} ms`}>
          {step.duration_ms >= 1000 ? `${(step.duration_ms / 1000).toFixed(1)}s` : `${step.duration_ms}ms`}
        </span>
      )}
    </div>
  );
}

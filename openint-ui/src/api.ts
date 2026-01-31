import { getLogger } from './observability';

const API_BASE = '';
const log = getLogger('api');

export interface ChatSource {
  id: string;
  content: string;
  score: number;
  metadata: Record<string, unknown>;
}

export interface DebugSemantic {
  type: string;
  label: string;
  value: string | number;
  snippet: string;
}

export interface TokenSemantic {
  token: string;
  meanings: string[];
}

export interface ChatDebug {
  query: string;
  semantics: DebugSemantic[];
  query_vector: number[];
  embedding_dims: number;
  embedding_model: string;
  token_semantics?: TokenSemantic[];
}

export interface ChatResponse {
  success: boolean;
  answer: string;
  sources: ChatSource[];
  query_time_ms: number;
  /** Actual Milvus/vector DB search time in ms (when provided by backend). */
  vector_db_query_time_ms?: number;
  /** Time to embed the query (ms). Explains total latency when vector DB is fast. */
  embedding_time_ms?: number;
  embedding_dims?: number;
  embedding_model?: string;
  debug?: ChatDebug;
  /** True when response was served from Redis cache. */
  from_cache?: boolean;
  /** Time to fetch from Redis in ms (when from_cache is true). */
  redis_query_time_ms?: number;
}

export interface SearchResult {
  id: string;
  content: string;
  score: number;
  metadata: Record<string, unknown>;
  is_chunk?: boolean;
  chunk_info?: { chunk_index: number; total_chunks: number; original_id: string };
}

export interface SearchResponse {
  success: boolean;
  query: string;
  results: SearchResult[];
  count: number;
  query_time?: number;
}

export interface StatsResponse {
  success: boolean;
  total_records: number;
}

export async function chat(message: string, topK = 15, debug = false, embeddingModel?: string): Promise<ChatResponse> {
  const body: { message: string; top_k: number; debug: boolean; embedding_model?: string } = {
    message,
    top_k: topK,
    debug,
  };
  if (embeddingModel) {
    body.embedding_model = embeddingModel;
  }
  log.info('Chat request', { url: '/api/chat', message_preview: message.slice(0, 80) });
  const t0 = performance.now();
  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const durationMs = Math.round(performance.now() - t0);
    if (!res.ok) {
      let errorMessage = 'Chat request failed';
      try {
        const text = await res.text();
        if (text && text.trim().length > 0) {
          const errorData = JSON.parse(text) as { error?: string; answer?: string; message?: string };
          errorMessage = errorData.error ?? errorData.message ?? errorData.answer?.replace(/^Error:\s*/i, '') ?? errorMessage;
        }
      } catch {
        errorMessage = res.status === 500
          ? 'Server error. Check backend logs or try again.'
          : res.status === 503
            ? 'Backend not ready yet. The server can take 30–90 seconds to start. Try again in a minute.'
            : (res.statusText || `HTTP ${res.status}`);
      }
      if (res.status === 503 && errorMessage.toLowerCase().includes('agent system')) {
        errorMessage = 'Backend not ready yet. The server can take 30–90 seconds to start. Try again in a minute.';
      }
      log.error('Chat request failed', { status: res.status, error: errorMessage, duration_ms: durationMs });
      throw new Error(errorMessage);
    }
    let data: ChatResponse;
    try {
      const text = await res.text();
      if (!text || text.trim().length === 0) {
        throw new Error('Empty response from server');
      }
      data = JSON.parse(text);
    } catch (e) {
      if (e instanceof SyntaxError) {
        throw new Error(`Invalid JSON response from server: ${e.message}`);
      }
      throw new Error(`Failed to parse response: ${e instanceof Error ? e.message : 'Unknown error'}`);
    }
    log.info('Chat request success', { duration_ms: durationMs, query_time_ms: data.query_time_ms });
    return data;
  } catch (e) {
    const durationMs = Math.round(performance.now() - t0);
    const msg = e instanceof Error ? e.message : String(e);
    log.error('Chat request error', { error: msg, duration_ms: durationMs });
    // "Failed to fetch" = network error: backend unreachable, connection refused, or timeout
    if (msg === 'Failed to fetch' || (e instanceof TypeError && msg.includes('fetch'))) {
      throw new Error(
        'Cannot reach the backend. Start it with ./start_backend.sh and ensure it listens on port 3001. If it is running, the request may have timed out (try again).'
      );
    }
    throw e;
  }
}

export async function search(
  query: string,
  topK = 10,
  fileType?: string
): Promise<SearchResponse> {
  const body: { query: string; top_k: number; file_type?: string } = {
    query,
    top_k: topK,
  };
  if (fileType) body.file_type = fileType;
  log.info('Search request', { url: '/api/search', query_preview: query.slice(0, 80) });
  const start = performance.now();
  try {
    const res = await fetch(`${API_BASE}/api/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    const durationMs = Math.round(performance.now() - start);
    if (!res.ok) {
      log.error('Search request failed', { status: res.status, error: (data as { error?: string }).error, duration_ms: durationMs });
      throw new Error((data as { error?: string }).error || 'Search failed');
    }
    log.info('Search request success', { duration_ms: durationMs, count: (data as SearchResponse).count });
    return data as SearchResponse;
  } catch (e) {
    log.error('Search request error', { error: e instanceof Error ? e.message : String(e) });
    throw e;
  }
}

export async function stats(): Promise<StatsResponse> {
  log.debug('Stats request', { url: '/api/stats' });
  const res = await fetch(`${API_BASE}/api/stats`);
  const data = await res.json();
  if (!res.ok) {
    log.warn('Stats request failed', { status: res.status });
    throw new Error((data as { error?: string }).error || 'Stats failed');
  }
  return data as StatsResponse;
}

export interface SemanticPreviewSegment {
  text: string;
  type: 'text' | 'highlight';
  tag?: {
    type: string;
    label: string;
    value: string | number;
    snippet: string;
    confidence: number;
  };
  tag_type?: string;
  label?: string;
  confidence?: number;
}

export interface SemanticPreview {
  success: boolean;
  query: string;
  model: string;
  tags: Array<{
    type: string;
    label: string;
    value: string | number;
    snippet: string;
    confidence: number;
  }>;
  highlighted_segments: SemanticPreviewSegment[];
  token_semantics?: Array<{
    token: string;
    meanings: string[];
  }>;
  embedding_stats?: {
    dimension: number;
    norm: number;
    mean: number;
    std: number;
  };
  error?: string;
  /** Time in ms for semantic annotation using the selected model */
  semantic_annotation_time_ms?: number;
}

export async function previewSemanticAnalysis(
  query: string,
  model: string
): Promise<SemanticPreview> {
  log.debug('Semantic preview request', { url: '/api/semantic/preview', model });
  try {
    const res = await fetch(`${API_BASE}/api/semantic/preview`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, model }),
    });
    if (!res.ok) {
      let errorMessage = 'Semantic preview failed';
      try {
        const errorData = await res.json();
        errorMessage = (errorData as { error?: string }).error || errorMessage;
      } catch {
        errorMessage = res.statusText || `HTTP ${res.status}`;
      }
      log.warn('Semantic preview failed', { status: res.status, error: errorMessage });
      throw new Error(errorMessage);
    }
    const data = await res.json();
    return data as SemanticPreview;
  } catch (e) {
    log.error('Semantic preview error', { error: e instanceof Error ? e.message : String(e) });
    throw e;
  }
}

/** Per-model result from multi-model semantic preview */
export interface SemanticPreviewModelResult {
  tags: Array<{
    type: string;
    label: string;
    value?: string | number;
    snippet: string;
    confidence?: number;
  }>;
  highlighted_segments: SemanticPreviewSegment[];
  error?: string;
  /** Time in ms for this model's semantic annotation */
  semantic_annotation_time_ms?: number;
}

export interface SemanticPreviewMulti {
  success: boolean;
  query: string;
  models: Record<string, SemanticPreviewModelResult>;
  best_model?: string;
  error?: string;
  /** Total time in ms for semantic annotation across all models */
  semantic_annotation_time_ms?: number;
  /** DataHub asset (dataset) names used for semantic tagging; same for all models */
  schema_assets?: string[];
}

/** Timeout for multi-model semantic preview (first run can load 3 models; 3 min). */
const PREVIEW_MULTI_TIMEOUT_MS = 180_000;

export async function previewSemanticAnalysisMulti(
  query: string
): Promise<SemanticPreviewMulti> {
  log.debug('Semantic preview multi request', { url: '/api/semantic/preview-multi' });
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), PREVIEW_MULTI_TIMEOUT_MS);
  try {
    const res = await fetch(`${API_BASE}/api/semantic/preview-multi`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    if (!res.ok) {
      let errorMessage = 'Multi-model semantic preview failed';
      try {
        const errorData = await res.json();
        errorMessage = (errorData as { error?: string }).error || errorMessage;
      } catch {
        errorMessage = res.statusText || `HTTP ${res.status}`;
      }
      log.warn('Semantic preview multi failed', { status: res.status, error: errorMessage });
      throw new Error(errorMessage);
    }
    const data = await res.json();
    return data as SemanticPreviewMulti;
  } catch (e) {
    clearTimeout(timeoutId);
    if (e instanceof Error) {
      if (e.name === 'AbortError') {
        log.warn('Semantic preview multi timed out', { timeoutMs: PREVIEW_MULTI_TIMEOUT_MS });
        throw new Error(
          'Request timed out. The first comparison can take several minutes while models load. Try again in a moment.'
        );
      }
      // Network/socket errors (e.g. "socket hang up", "Failed to fetch")
      const msg = e.message || String(e);
      if (/timeout|hang up|network|failed to fetch/i.test(msg)) {
        throw new Error(
          `Connection failed: ${msg}. The first run loads 3 models and can take a few minutes—check backend logs and try again.`
        );
      }
    }
    log.error('Semantic preview multi error', { error: e instanceof Error ? e.message : String(e) });
    throw e;
  }
}

/** Response from GET /api/suggestions/lucky (sa-agent "I'm feeling lucky!") */
export interface LuckySuggestion {
  success: boolean;
  sentence?: string;
  category?: string;
  /** "ollama" or "template" */
  source?: string;
  /** When source is "ollama", the model name (e.g. llama3.2) */
  llm_model?: string;
  /** Time in ms for sa-agent to generate the sentence (schema + LLM/template) */
  sa_agent_time_ms?: number;
  error?: string;
}

export async function getLuckySuggestion(): Promise<LuckySuggestion> {
  log.debug('Lucky suggestion request', { url: '/api/suggestions/lucky' });
  const res = await fetch(`${API_BASE}/api/suggestions/lucky`, { method: 'GET' });
  const text = await res.text();
  let data: LuckySuggestion;
  try {
    data = (text ? JSON.parse(text) : {}) as LuckySuggestion;
  } catch {
    throw new Error(res.ok ? 'Invalid response from server' : (res.statusText || `Request failed (${res.status})`));
  }
  if (!res.ok) {
    throw new Error(data.error || res.statusText || 'Failed to get suggestion');
  }
  return data;
}

/** A2A run: sa-agent → modelmgmt-agent flow */
export interface A2AStep {
  agent: string;
  action: string;
  status: 'running' | 'completed' | 'failed';
  count?: number;
}

export interface A2ASentence {
  text: string;
  category?: string;
}

export interface A2AAnnotationItem {
  sentence: string;
  annotation: {
    success?: boolean;
    query?: string;
    models?: Record<string, { tags?: unknown[]; highlighted_segments?: unknown[]; semantic_annotation_time_ms?: number }>;
    best_model?: string;
    schema_assets?: string[];
    error?: string;
  } | null;
  success: boolean;
}

export interface A2ARunResponse {
  success: boolean;
  steps: A2AStep[];
  sentences: A2ASentence[];
  annotations: A2AAnnotationItem[];
  /** Time in ms for sa-agent to generate sentences */
  sa_agent_time_ms?: number | null;
  /** Time in ms for modelmgmt-agent to annotate all sentences */
  modelmgmt_agent_time_ms?: number | null;
  error?: string;
}

export async function runA2A(sentenceCount = 3): Promise<A2ARunResponse> {
  const res = await fetch(`${API_BASE}/api/a2a/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sentence_count: sentenceCount }),
  });
  const text = await res.text();
  let data: A2ARunResponse;
  try {
    data = (text ? JSON.parse(text) : {}) as A2ARunResponse;
  } catch {
    throw new Error(res.ok ? 'Invalid response from server' : (res.statusText || `Request failed (${res.status})`));
  }
  if (!res.ok) {
    throw new Error(data.error || res.statusText || 'A2A run failed');
  }
  return data;
}

// --- Neo4j Graph Demo ---
export interface GraphStats {
  success: boolean;
  connected: boolean;
  node_counts?: Record<string, number>;
  relationship_counts?: Record<string, number>;
  error?: string;
}

export interface GraphSchemaNode {
  label: string;
  id_property: string;
  type_property?: string;
  description: string;
}

export interface GraphSchemaRel {
  type: string;
  from: string;
  to: string;
  description: string;
}

export interface GraphSchemaResponse {
  success: boolean;
  schema: {
    nodes: GraphSchemaNode[];
    relationships: GraphSchemaRel[];
    source: string;
  };
}

export interface GraphSampleResponse {
  success: boolean;
  disputes_overview: Array<{
    customer_id?: string;
    dispute_id?: string;
    transaction_id?: string;
    status?: string;
    amount_disputed?: number;
    currency?: string;
  }>;
  paths: Array<{
    customer_id?: string;
    transaction_id?: string;
    dispute_id?: string;
    tx_amount?: number;
    amount_disputed?: number;
    dispute_status?: string;
  }>;
  error?: string;
}

async function parseGraphJson<T>(res: Response, fallbackError: string): Promise<{ data?: T; error: string }> {
  const statusText = res.status === 404 ? 'Graph API not found. Is the backend running? Start it with ./start_backend.sh and ensure it serves /api/graph/*.' : (res.statusText || fallbackError);
  try {
    const text = await res.text();
    if (!text || !text.trim()) return { error: statusText };
    const data = JSON.parse(text) as T;
    if (!res.ok) {
      const err = (data as { error?: string }).error || statusText;
      return { error: err };
    }
    return { data, error: '' };
  } catch {
    return { error: statusText };
  }
}

export async function fetchGraphStats(): Promise<GraphStats> {
  const res = await fetch(`${API_BASE}/api/graph/stats`);
  const { data, error } = await parseGraphJson<GraphStats>(res, 'Graph stats failed');
  if (error) throw new Error(error);
  return data!;
}

export async function fetchGraphSchema(): Promise<GraphSchemaResponse> {
  const res = await fetch(`${API_BASE}/api/graph/schema`);
  const { data, error } = await parseGraphJson<GraphSchemaResponse>(res, 'Graph schema failed');
  if (error) throw new Error(error);
  return data!;
}

export async function fetchGraphSample(): Promise<GraphSampleResponse> {
  const res = await fetch(`${API_BASE}/api/graph/sample`);
  const { data, error } = await parseGraphJson<GraphSampleResponse>(res, 'Graph sample failed');
  if (error) throw new Error(error);
  return data!;
}

/** Predefined graph query: run by query_id, returns cypher + columns + rows */
export interface GraphQueryResponse {
  success: boolean;
  query_id: string | null;
  label: string | null;
  /** Neo4j Cypher query that was executed */
  cypher?: string | null;
  columns: string[];
  rows: Record<string, unknown>[];
  error?: string;
}

export async function runGraphQuery(queryId: string): Promise<GraphQueryResponse> {
  const params = new URLSearchParams({ query_id: queryId });
  const res = await fetch(`${API_BASE}/api/graph/query?${params.toString()}`, {
    method: 'GET',
  });
  const text = await res.text();
  let data: GraphQueryResponse;
  try {
    data = (text ? JSON.parse(text) : {}) as GraphQueryResponse;
  } catch {
    throw new Error(res.ok ? 'Invalid response' : (res.statusText || 'Graph query failed'));
  }
  return data;
}

/** Natural language graph query: LLM (Ollama) generates Cypher from question + Neo4j schema */
export interface GraphQueryNaturalResponse {
  success: boolean;
  query: string | null;
  cypher?: string | null;
  columns: string[];
  rows: Record<string, unknown>[];
  error?: string;
  llm_model?: string;
  llm_time_ms?: number;
}

export async function runGraphQueryNatural(query: string): Promise<GraphQueryNaturalResponse> {
  const res = await fetch(`${API_BASE}/api/graph/query-natural`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: query.trim() }),
  });
  const text = await res.text();
  let data: GraphQueryNaturalResponse;
  try {
    data = (text ? JSON.parse(text) : {}) as GraphQueryNaturalResponse;
  } catch {
    throw new Error(res.ok ? 'Invalid response' : (res.statusText || 'Graph query failed'));
  }
  return data;
}

/** Enrich: lookup ID without label (tries Customer, Transaction, Dispute). For UI popup. */
export interface GraphEnrichResponse {
  success: boolean;
  label: string | null;
  id: string;
  /** Human-readable: e.g. "John Smith (account / customer id: 1000000001)" */
  display_name?: string;
  properties: Record<string, unknown>;
  error?: string;
}

export async function fetchGraphEnrich(id: string, label?: string): Promise<GraphEnrichResponse> {
  const params = new URLSearchParams({ id });
  if (label && ['Customer', 'Transaction', 'Dispute'].includes(label)) {
    params.set('label', label);
  }
  const res = await fetch(`${API_BASE}/api/graph/enrich?${params.toString()}`);
  const text = await res.text();
  let data: GraphEnrichResponse;
  try {
    data = (text ? JSON.parse(text) : { success: false }) as GraphEnrichResponse;
  } catch {
    throw new Error(res.ok ? 'Invalid response from server' : (res.statusText || 'Enrich lookup failed'));
  }
  return data;
}

/** Full node details for one graph node (table: one row, all properties). */
export interface GraphNodeDetailsResponse {
  success: boolean;
  label: string;
  id: string;
  columns: string[];
  rows: Record<string, unknown>[];
  /** Cypher used for the node lookup (e.g. MATCH (n:Customer {id: $id}) RETURN n). */
  cypher?: string;
  /** Parameters used (e.g. { id: "1000000001" }). */
  params?: Record<string, unknown>;
  error?: string;
}

export async function fetchGraphNodeDetails(label: string, id: string): Promise<GraphNodeDetailsResponse> {
  const params = new URLSearchParams({ label, id });
  const res = await fetch(`${API_BASE}/api/graph/node-details?${params.toString()}`);
  const text = await res.text();
  let data: GraphNodeDetailsResponse;
  try {
    data = (text ? JSON.parse(text) : {}) as GraphNodeDetailsResponse;
  } catch {
    throw new Error(res.ok ? 'Invalid response' : (res.statusText || 'Node details failed'));
  }
  return data;
}

/** Sentiment analysis for dispute text (LLM-generated, free-form). */
export interface GraphSentimentResponse {
  success: boolean;
  sentiment?: string | null;
  confidence?: number | null;
  emoji?: string | null;
  error?: string;
}

export async function fetchGraphSentiment(text: string): Promise<GraphSentimentResponse> {
  const res = await fetch(`${API_BASE}/api/graph/sentiment`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  const data = (await res.json()) as GraphSentimentResponse;
  return data;
}

/** Recent graph questions from Redis (for History list). */
export interface GraphRecentQueriesResponse {
  queries: string[];
}

export async function fetchGraphRecentQueries(): Promise<GraphRecentQueriesResponse> {
  const res = await fetch(`${API_BASE}/api/graph/queries/recent`);
  const text = await res.text();
  let data: GraphRecentQueriesResponse;
  try {
    data = (text ? JSON.parse(text) : { queries: [] }) as GraphRecentQueriesResponse;
  } catch {
    return { queries: [] };
  }
  return data;
}

// --- Multi-Agent Demo ---
export interface MultiAgentDemoStep {
  agent: string;
  action: string;
  status: 'running' | 'completed' | 'failed';
  duration_ms?: number;
  sentence?: string;
  result_count?: number;
  /** sentiment-agent output */
  sentiment?: string;
  emoji?: string;
  confidence?: number;
  /** Brief explanation of why this sentiment was detected */
  reasoning?: string;
  error?: string;
}

/** Sentiment analysis from sentiment-agent (LLM). */
export interface MultiAgentDemoSentiment {
  sentiment?: string;
  confidence?: number;
  emoji?: string;
  /** Brief explanation of why this sentiment was detected */
  reasoning?: string;
}

export interface MultiAgentDemoResponse {
  success: boolean;
  answer: string;
  steps: MultiAgentDemoStep[];
  langgraph_steps: string[];
  /** User's original input (message). */
  original_query?: string;
  /** sa-agent generated/corrected sentence. */
  sentence: string;
  /** Sentiment analysis from sentiment-agent (after sa-agent). */
  sentiment?: MultiAgentDemoSentiment;
  chunking_strategies: Record<string, unknown>;
  vector_results: Array<{ id: string; content: string; score: number }>;
  graph_results: { query?: string; cypher?: string; columns?: string[]; rows?: unknown[]; error?: string };
  /** IDs that enrich-agent successfully looked up in Neo4j */
  enriched_entities?: string[];
  /** Per-ID enrich details: { label, display_name, properties } */
  enriched_details?: Record<string, { label?: string; display_name?: string; properties?: Record<string, unknown> }>;
  error?: string;
}

export interface RunMultiAgentDemoOptions {
  debug?: boolean;
  from_lucky?: boolean;
  sa_agent_time_ms?: number;
}

export async function runMultiAgentDemo(
  message: string,
  debug = false,
  options?: RunMultiAgentDemoOptions
): Promise<MultiAgentDemoResponse> {
  const url = `${API_BASE}/api/multi-agent-demo/run`;
  const body: { message: string; debug: boolean; from_lucky?: boolean; sa_agent_time_ms?: number } = {
    message,
    debug,
  };
  if (options?.from_lucky === true) {
    body.from_lucky = true;
    if (options.sa_agent_time_ms != null) body.sa_agent_time_ms = options.sa_agent_time_ms;
  }
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    cache: 'no-store',
  });
  if (res.status === 405) {
    const text = await res.text();
    let msg = 'Backend returned 405 (Method Not Allowed).';
    try {
      const body = text ? JSON.parse(text) : {};
      if (body.method_received === 'GET') {
        msg = 'Request was sent as GET instead of POST. Ensure the backend is the target of the proxy (e.g. Vite proxy to port 3001) and restart the backend.';
      } else if (body.message) {
        msg = body.message;
      }
    } catch {
      // ignore
    }
    throw new Error(msg);
  }
  const text = await res.text();
  let data: MultiAgentDemoResponse;
  try {
    data = (text ? JSON.parse(text) : {}) as MultiAgentDemoResponse;
  } catch {
    if (!res.ok) {
      throw new Error(res.statusText || `Request failed (${res.status})`);
    }
    throw new Error('Invalid response from server');
  }
  if (!res.ok) {
    throw new Error(data.error || res.statusText || 'Multi-agent demo failed');
  }
  return data;
}

/** Single history entry: query text and optional issued-at (ISO datetime). */
export interface MultiAgentHistoryEntry {
  query: string;
  issued_at: string | null;
}

/** Recent multi-agent demo questions from Redis (for History pane). */
export interface MultiAgentRecentQueriesResponse {
  queries: MultiAgentHistoryEntry[];
}

function normalizeHistoryEntries(raw: unknown): MultiAgentHistoryEntry[] {
  if (!Array.isArray(raw)) return [];
  return raw.map((item): MultiAgentHistoryEntry => {
    if (item && typeof item === 'object' && 'query' in item && typeof (item as { query: unknown }).query === 'string') {
      const o = item as { query: string; issued_at?: string | null };
      return { query: o.query, issued_at: o.issued_at ?? null };
    }
    if (typeof item === 'string' && item.trim()) {
      return { query: item.trim(), issued_at: null };
    }
    return { query: '', issued_at: null };
  }).filter((e) => e.query.length > 0);
}

export async function fetchMultiAgentRecentQueries(): Promise<MultiAgentRecentQueriesResponse> {
  try {
    const res = await fetch(`${API_BASE}/api/multi-agent-demo/queries/recent`);
    const text = await res.text();
    let data: { queries?: unknown };
    try {
      data = (text ? JSON.parse(text) : { queries: [] }) as { queries?: unknown };
    } catch {
      return { queries: [] };
    }
    return { queries: normalizeHistoryEntries(data.queries) };
  } catch {
    return { queries: [] };
  }
}

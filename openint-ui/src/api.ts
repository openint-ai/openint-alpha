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

/** Response from GET /api/suggestions/lucky (sg-agent "I'm feeling lucky!") */
export interface LuckySuggestion {
  success: boolean;
  sentence?: string;
  category?: string;
  /** "openai" or "template" */
  source?: string;
  /** When source is "openai", the model name (e.g. gpt-4o-mini) */
  llm_model?: string;
  error?: string;
}

export async function getLuckySuggestion(): Promise<LuckySuggestion> {
  log.debug('Lucky suggestion request', { url: '/api/suggestions/lucky' });
  const res = await fetch(`${API_BASE}/api/suggestions/lucky`, { method: 'GET' });
  const data = (await res.json()) as LuckySuggestion;
  if (!res.ok) {
    throw new Error(data.error || res.statusText || 'Failed to get suggestion');
  }
  return data;
}

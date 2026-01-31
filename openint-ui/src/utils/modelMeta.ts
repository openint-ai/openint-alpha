/**
 * Shared model metadata: display names and Hugging Face URLs.
 * Used across Chat, SemanticCompare, A2A, MultiAgentDemo, GraphDemo to show which model was used and link to HF.
 */

export interface ModelMeta {
  label: string;
  url?: string;
  author?: string;
  description?: string;
  details?: string;
}

/** Embedding / semantic models (Vector DB, Compare, A2A). Keys match backend model IDs. */
const EMBEDDING_MODELS: Record<string, ModelMeta> = {
  'mukaj/fin-mpnet-base': {
    label: 'Finance MPNet',
    author: 'mukaj',
    description: 'State-of-the-art for financial documents (79.91 FiQA). Best for banking/finance semantic search.',
    details: '768 dimensions · Fast · Use for banking/finance applications.',
    url: 'https://huggingface.co/mukaj/fin-mpnet-base',
  },
  'ProsusAI/finbert': {
    label: 'FinBERT',
    author: 'Prosus AI',
    description: 'Finance-oriented model for financial text understanding and semantic tasks.',
    details: 'Finance domain · Use for financial sentiment and understanding.',
    url: 'https://huggingface.co/ProsusAI/finbert',
  },
  'sentence-transformers/all-mpnet-base-v2': {
    label: 'General MPNet',
    author: 'sentence-transformers',
    description: 'Popular general-purpose embeddings (768d). Good balance of quality and speed.',
    details: '768 dimensions · Popular · Use for general semantic tasks.',
    url: 'https://huggingface.co/sentence-transformers/all-mpnet-base-v2',
  },
  'all-MiniLM-L6-v2': {
    label: 'MiniLM L6 v2',
    author: 'sentence-transformers',
    description: 'Lightweight, fast sentence embeddings (384d).',
    details: '384 dimensions · Lightweight · Fast.',
    url: 'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2',
  },
  'sentence-transformers/all-MiniLM-L6-v2': {
    label: 'MiniLM L6 v2',
    author: 'sentence-transformers',
    description: 'Lightweight, fast sentence embeddings (384d).',
    details: '384 dimensions · Lightweight · Fast.',
    url: 'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2',
  },
};

/** Common LLM names (Ollama / backend) -> Hugging Face model page when available. */
const LLM_MODELS: Record<string, ModelMeta> = {
  'qwen2.5:7b': { label: 'Qwen2.5 7B', url: 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct' },
  'qwen2.5:3b': { label: 'Qwen2.5 3B', url: 'https://huggingface.co/Qwen/Qwen2.5-3B-Instruct' },
  'llama3.2': { label: 'Llama 3.2', url: 'https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct' },
  'llama3.2:3b': { label: 'Llama 3.2 3B', url: 'https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct' },
  'llama3.1': { label: 'Llama 3.1', url: 'https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct' },
  'mistral': { label: 'Mistral', url: 'https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2' },
  'phi3': { label: 'Phi-3', url: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct' },
};

const ALL_MODELS: Record<string, ModelMeta> = { ...EMBEDDING_MODELS, ...LLM_MODELS };

/** Normalize backend model id for lookup (e.g. "fin-mpnet-base" -> try "mukaj/fin-mpnet-base"). */
function normalizeId(id: string): string {
  const s = (id || '').trim();
  if (!s) return s;
  if (ALL_MODELS[s]) return s;
  const lower = s.toLowerCase();
  for (const key of Object.keys(EMBEDDING_MODELS)) {
    if (key.endsWith('/' + lower) || key === lower) return key;
  }
  return s;
}

export function getModelMeta(id: string | null | undefined): ModelMeta | null {
  if (id == null || String(id).trim() === '') return null;
  const normalized = normalizeId(String(id).trim());
  return ALL_MODELS[normalized] ?? null;
}

export function getModelDisplayName(id: string | null | undefined): string {
  const meta = getModelMeta(id);
  if (meta) return meta.label;
  return (id ?? '').trim() || '—';
}

export function getModelUrl(id: string | null | undefined): string | null {
  const meta = getModelMeta(id);
  return meta?.url ?? null;
}

/**
 * Build Hugging Face URL for a model id that looks like "org/repo".
 * Use when id is not in our map but follows HF convention.
 */
export function huggingFaceUrlForId(id: string | null | undefined): string | null {
  const s = (id ?? '').trim();
  if (!s) return null;
  if (s.includes('/')) return `https://huggingface.co/${s}`;
  return null;
}

export { EMBEDDING_MODELS, LLM_MODELS, ALL_MODELS };

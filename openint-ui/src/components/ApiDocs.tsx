import { useState, useCallback } from 'react';
import { Link } from 'react-router-dom';

const DEFAULT_BASE = typeof window !== 'undefined' ? window.location.origin : 'http://localhost:3001';
const SAMPLE_SENTENCE = 'Show me transactions for customer 1001 in California';
const SAMPLE_MODEL = 'mukaj/fin-mpnet-base';

const GET_CURL = (base: string) =>
  `curl -X GET "${base}/api/semantic/interpret?sentence=${encodeURIComponent(SAMPLE_SENTENCE)}&model=${encodeURIComponent(SAMPLE_MODEL)}"`;

const POST_CURL = (base: string) =>
  `curl -X POST "${base}/api/semantic/preview" \\
  -H "Content-Type: application/json" \\
  -d '{"query": "${SAMPLE_SENTENCE.replace(/"/g, '\\"')}", "model": "${SAMPLE_MODEL}"}'`;

const FETCH_GET = (base: string) =>
  `const sentence = encodeURIComponent("${SAMPLE_SENTENCE}");
const model = "${SAMPLE_MODEL}";
const res = await fetch(\`${base}/api/semantic/interpret?sentence=\${sentence}&model=\${model}\`);
const data = await res.json();`;

const FETCH_POST = (base: string) =>
  `const res = await fetch("${base}/api/semantic/preview", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    query: "${SAMPLE_SENTENCE}",
    model: "${SAMPLE_MODEL}"
  })
});
const data = await res.json();`;

/** Supported models (same as UI dropdown and backend). Pass `id` as the `model` parameter to interpret/preview APIs. */
const MODELS_REFERENCE = [
  {
    id: 'mukaj/fin-mpnet-base',
    display_name: 'Finance MPNet',
    author: 'mukaj',
    description: 'State-of-the-art for financial documents (79.91 FiQA). Trained on 150k+ financial QA examples. Best for banking/finance semantic search.',
    details: '768 dimensions · Fast · Use for banking/finance applications.',
    url: 'https://huggingface.co/mukaj/fin-mpnet-base',
  },
  {
    id: 'ProsusAI/finbert',
    display_name: 'FinBERT',
    author: 'Prosus AI',
    description: 'Finance-oriented model suited for financial text understanding and semantic tasks.',
    details: 'Finance domain · Use for financial sentiment and understanding.',
    url: 'https://huggingface.co/ProsusAI/finbert',
  },
  {
    id: 'sentence-transformers/all-mpnet-base-v2',
    display_name: 'General MPNet',
    author: 'sentence-transformers',
    description: 'Popular, powerful open source model. Strong general-purpose embeddings (768d). Good balance of quality and speed.',
    details: '768 dimensions · Popular · Use for general semantic tasks.',
    url: 'https://huggingface.co/sentence-transformers/all-mpnet-base-v2',
  },
];

const SAMPLE_RESPONSE = {
  success: true,
  query: SAMPLE_SENTENCE,
  model: SAMPLE_MODEL,
  tags: [
    { type: 'customer_id', label: 'Customer ID', value: '1001', snippet: '1001', confidence: 0.95 },
    { type: 'state', label: 'State', value: 'California', snippet: 'California', confidence: 0.9 },
    { type: 'intent', label: 'Intent', value: 'transactions', snippet: 'transactions', confidence: 0.85 },
  ],
  highlighted_segments: [
    { type: 'plain', text: 'Show me ' },
    { type: 'highlight', text: 'transactions', tag: { type: 'intent', label: 'Intent', value: 'transactions' } },
    { type: 'plain', text: ' for customer ' },
    { type: 'highlight', text: '1001', tag: { type: 'customer_id', label: 'Customer ID', value: '1001' } },
    { type: 'plain', text: ' in ' },
    { type: 'highlight', text: 'California', tag: { type: 'state', label: 'State', value: 'California' } },
  ],
  token_semantics: [
    { token: 'transactions', meanings: ['Intent:intent'] },
    { token: '1001', meanings: ['Customer ID:customer_id'] },
    { token: 'california', meanings: ['State:state'] },
  ],
  embedding_stats: { dimension: 768, norm: 1.0 },
};

function CopyButton({ text, label = 'Copy' }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  const copy = useCallback(async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [text]);
  return (
    <button
      type="button"
      onClick={copy}
      className="absolute top-2 right-2 rounded-lg px-2 py-1 text-xs font-medium bg-surface-200 text-gray-700 hover:bg-surface-300 transition-colors"
    >
      {copied ? 'Copied!' : label}
    </button>
  );
}

export default function ApiDocs() {
  const [baseUrl, setBaseUrl] = useState(DEFAULT_BASE);

  return (
    <article className="doc-page max-w-4xl space-y-8">
      <nav className="text-xs text-gray-500 mb-2" aria-label="Breadcrumb">
        <span className="text-gray-400">Documentation</span>
        <span className="mx-1.5">/</span>
        <span className="text-gray-700 font-medium">API</span>
      </nav>
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 tracking-tight">Semantic layer API</h1>
        <p className="mt-3 text-lg text-gray-600 leading-relaxed">
          Integrate with the semantic interpretation API: send a sentence and model, get tags, highlighted segments, and token semantics.
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          <span className="inline-flex items-center rounded-full bg-brand-100 px-3 py-1 text-xs font-medium text-brand-800">
            REST
          </span>
          <span className="inline-flex items-center rounded-full bg-surface-200 px-3 py-1 text-xs font-medium text-gray-700">
            GET / POST
          </span>
        </div>
      </header>

      <div className="rounded-xl border border-surface-200 bg-white p-4 shadow-sm">
        <label className="block text-sm font-medium text-gray-700 mb-2">API base URL (for examples)</label>
        <input
          type="url"
          value={baseUrl}
          onChange={(e) => setBaseUrl(e.target.value.trim() || DEFAULT_BASE)}
          className="w-full max-w-xl rounded-lg border border-surface-200 px-3 py-2 text-sm text-gray-900 focus:border-brand-500 focus:ring-1 focus:ring-brand-500"
          placeholder="http://localhost:3001"
        />
        <p className="mt-1 text-xs text-gray-500">
          Use your backend base URL. When the UI is served with the same host as the API, use <code className="rounded bg-surface-100 px-1">window.location.origin</code>.
        </p>
      </div>

      {/* Models reference */}
      <section id="models-reference" className="scroll-mt-6">
        <h2 className="text-xl font-semibold text-gray-900 border-b border-surface-200 pb-2 mb-4">
          Models reference
        </h2>
        <p className="text-sm text-gray-600 mb-4">
          Pass the <strong>Model ID</strong> as the <code className="rounded bg-surface-100 px-1">model</code> parameter to interpret and preview APIs. Same list is available via <code className="rounded bg-surface-100 px-1">GET /api/semantic/models-with-meta</code>.
        </p>
        <div className="space-y-4">
          {MODELS_REFERENCE.map((m) => (
            <div
              key={m.id}
              className="rounded-xl border border-surface-200 bg-white p-5 shadow-sm"
            >
              <div className="flex flex-wrap items-center gap-2 mb-2">
                <code className="rounded bg-brand-100 px-2 py-0.5 text-sm font-semibold text-brand-800">
                  {m.id}
                </code>
                <span className="text-sm font-medium text-gray-700">{m.display_name}</span>
              </div>
              <p className="text-sm text-gray-600 mb-1">
                <strong className="text-gray-700">Author:</strong> {m.author}
              </p>
              <p className="text-sm text-gray-600 mb-1">
                <strong className="text-gray-700">Description:</strong> {m.description}
              </p>
              <p className="text-sm text-gray-500 mb-2">
                <strong className="text-gray-600">Details:</strong> {m.details}
              </p>
              <a
                href={m.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs font-medium text-brand-600 hover:text-brand-700 underline"
              >
                View on Hugging Face →
              </a>
            </div>
          ))}
        </div>
      </section>

      {/* GET interpret */}
      <section id="get-interpret" className="space-y-3 scroll-mt-6">
        <h2 className="text-xl font-semibold text-gray-900 border-b border-surface-200 pb-2">GET — Interpret sentence (query params)</h2>
        <p className="text-sm text-gray-600">
          <code className="rounded bg-surface-100 px-1.5 py-0.5">GET /api/semantic/interpret</code> — Pass the sentence and model as query parameters. Easy for curl and browser links.
        </p>
        <div className="rounded-lg border border-surface-200 bg-surface-50 p-3 font-mono text-xs text-gray-800 overflow-x-auto">
          <span className="text-brand-600">GET</span> {baseUrl}/api/semantic/interpret?sentence=&lt;sentence&gt;&amp;model=&lt;model&gt;
        </div>
        <div className="rounded-lg border border-surface-200 bg-surface-50 p-4 relative">
          <CopyButton text={GET_CURL(baseUrl)} />
          <pre className="pr-20 overflow-x-auto text-xs text-gray-800 whitespace-pre-wrap break-all">{GET_CURL(baseUrl)}</pre>
        </div>
      </section>

      {/* POST preview */}
      <section id="post-preview" className="space-y-3 scroll-mt-6">
        <h2 className="text-xl font-semibold text-gray-900 border-b border-surface-200 pb-2">POST — Interpret sentence (JSON body)</h2>
        <p className="text-sm text-gray-600">
          <code className="rounded bg-surface-100 px-1.5 py-0.5">POST /api/semantic/preview</code> — Same semantic interpretation; sentence and model in the request body.
        </p>
        <div className="rounded-lg border border-surface-200 bg-surface-50 p-3 font-mono text-xs text-gray-800 overflow-x-auto">
          <span className="text-brand-600">POST</span> {baseUrl}/api/semantic/preview
        </div>
        <p className="text-xs text-gray-600">Request body:</p>
        <div className="rounded-lg border border-surface-200 bg-surface-50 p-4 relative">
          <CopyButton text={JSON.stringify({ query: SAMPLE_SENTENCE, model: SAMPLE_MODEL }, null, 2)} />
          <pre className="pr-16 overflow-x-auto text-xs text-gray-800 whitespace-pre-wrap">
            {JSON.stringify({ query: SAMPLE_SENTENCE, model: SAMPLE_MODEL }, null, 2)}
          </pre>
        </div>
        <div className="rounded-lg border border-surface-200 bg-surface-50 p-4 relative">
          <CopyButton text={POST_CURL(baseUrl)} />
          <pre className="pr-20 overflow-x-auto text-xs text-gray-800 whitespace-pre-wrap">{POST_CURL(baseUrl)}</pre>
        </div>
      </section>

      {/* Interpret-all (all models) */}
      <section id="interpret-all" className="space-y-3 scroll-mt-6">
        <h2 className="text-xl font-semibold text-gray-900 border-b border-surface-200 pb-2">Interpret sentence with all supported models</h2>
        <p className="text-sm text-gray-600">
          <code className="rounded bg-surface-100 px-1.5 py-0.5">GET /api/semantic/interpret-all?sentence=...</code> or <code className="rounded bg-surface-100 px-1.5 py-0.5">POST /api/semantic/interpret-all</code> with body <code className="rounded bg-surface-100 px-1">{`{ "query": "sentence" }`}</code>. Returns interpretation from every model listed in the Models reference above (same as the UI dropdown).
        </p>
        <div className="rounded-lg border border-surface-200 bg-surface-50 p-3 font-mono text-xs text-gray-800 overflow-x-auto">
          <span className="text-brand-600">GET</span> {baseUrl}/api/semantic/interpret-all?sentence=&lt;sentence&gt;
        </div>
        <div className="rounded-lg border border-surface-200 bg-surface-50 p-4 relative">
          <CopyButton text={`curl -X GET "${baseUrl}/api/semantic/interpret-all?sentence=${encodeURIComponent(SAMPLE_SENTENCE)}"`} />
          <pre className="pr-20 overflow-x-auto text-xs text-gray-800 whitespace-pre-wrap break-all">{`curl -X GET "${baseUrl}/api/semantic/interpret-all?sentence=${encodeURIComponent(SAMPLE_SENTENCE)}"`}</pre>
        </div>
        <p className="text-xs text-gray-600">POST body:</p>
        <div className="rounded-lg border border-surface-200 bg-surface-50 p-4 relative">
          <CopyButton text={JSON.stringify({ query: SAMPLE_SENTENCE }, null, 2)} />
          <pre className="pr-16 overflow-x-auto text-xs text-gray-800 whitespace-pre-wrap">{JSON.stringify({ query: SAMPLE_SENTENCE }, null, 2)}</pre>
        </div>
        <p className="text-xs text-gray-600">Response: <code className="rounded bg-surface-100 px-1">success</code>, <code className="rounded bg-surface-100 px-1">query</code>, <code className="rounded bg-surface-100 px-1">models</code> (object keyed by model id; each value has the same shape as single-model interpret).</p>
      </section>

      {/* Response */}
      <section id="response" className="space-y-3 scroll-mt-6">
        <h2 className="text-xl font-semibold text-gray-900 border-b border-surface-200 pb-2">Response (200 OK)</h2>
        <p className="text-sm text-gray-600">
          Both endpoints return the same shape: <code className="rounded bg-surface-100 px-1">success</code>, <code className="rounded bg-surface-100 px-1">query</code>, <code className="rounded bg-surface-100 px-1">model</code>, <code className="rounded bg-surface-100 px-1">tags</code>, <code className="rounded bg-surface-100 px-1">highlighted_segments</code>, <code className="rounded bg-surface-100 px-1">token_semantics</code>, and optional <code className="rounded bg-surface-100 px-1">embedding_stats</code>.
        </p>
        <div className="rounded-lg border border-surface-200 bg-surface-50 p-4 relative max-h-96 overflow-auto">
          <CopyButton text={JSON.stringify(SAMPLE_RESPONSE, null, 2)} />
          <pre className="pr-16 overflow-x-auto text-xs text-gray-800 whitespace-pre-wrap">
            {JSON.stringify(SAMPLE_RESPONSE, null, 2)}
          </pre>
        </div>
      </section>

      {/* Integrate: fetch */}
      <section id="integrate" className="space-y-3 scroll-mt-6">
        <h2 className="text-xl font-semibold text-gray-900 border-b border-surface-200 pb-2">Integrate (JavaScript / fetch)</h2>
        <p className="text-sm text-gray-600">GET — interpret by sentence and model:</p>
        <div className="rounded-lg border border-surface-200 bg-surface-50 p-4 relative">
          <CopyButton text={FETCH_GET(baseUrl)} />
          <pre className="pr-16 overflow-x-auto text-xs text-gray-800 whitespace-pre-wrap">{FETCH_GET(baseUrl)}</pre>
        </div>
        <p className="text-sm text-gray-600 mt-4">POST — same with JSON body:</p>
        <div className="rounded-lg border border-surface-200 bg-surface-50 p-4 relative">
          <CopyButton text={FETCH_POST(baseUrl)} />
          <pre className="pr-16 overflow-x-auto text-xs text-gray-800 whitespace-pre-wrap">{FETCH_POST(baseUrl)}</pre>
        </div>
      </section>

      {/* List models */}
      <section id="list-models" className="space-y-3 scroll-mt-6">
        <h2 className="text-xl font-semibold text-gray-900 border-b border-surface-200 pb-2">List available models</h2>
        <p className="text-sm text-gray-600">
          <code className="rounded bg-surface-100 px-1.5 py-0.5">GET /api/semantic/models</code> — Returns model IDs and dimensions. <code className="rounded bg-surface-100 px-1.5 py-0.5">GET /api/semantic/models-with-meta</code> — Returns full metadata (id, display_name, author, description, details, url) for each supported model.
        </p>
        <div className="rounded-lg border border-surface-200 bg-surface-50 p-3 font-mono text-xs text-gray-800">
          <span className="text-brand-600">GET</span> {baseUrl}/api/semantic/models
        </div>
        <div className="rounded-lg border border-surface-200 bg-surface-50 p-4 relative">
          <CopyButton text={`curl -X GET "${baseUrl}/api/semantic/models"`} />
          <pre className="pr-16 overflow-x-auto text-xs text-gray-800">{`curl -X GET "${baseUrl}/api/semantic/models"`}</pre>
        </div>
        <div className="rounded-lg border border-surface-200 bg-surface-50 p-3 font-mono text-xs text-gray-800">
          <span className="text-brand-600">GET</span> {baseUrl}/api/semantic/models-with-meta
        </div>
        <div className="rounded-lg border border-surface-200 bg-surface-50 p-4 relative">
          <CopyButton text={`curl -X GET "${baseUrl}/api/semantic/models-with-meta"`} />
          <pre className="pr-16 overflow-x-auto text-xs text-gray-800">{`curl -X GET "${baseUrl}/api/semantic/models-with-meta"`}</pre>
        </div>
      </section>

      <section className="rounded-xl border border-brand-200 bg-brand-50 p-4 text-sm text-brand-900">
        <p className="font-medium">Semantic layer</p>
        <p className="mt-1 text-brand-800/90">
          Use these endpoints to integrate your apps with the semantic layer: send natural language sentences and get structured tags (customer IDs, states, intents, etc.) and token-level semantics for search, analytics, or downstream services.
        </p>
      </section>

      <section className="flex flex-wrap gap-3 pt-4 border-t border-surface-200">
        <span className="text-sm text-gray-500">Related:</span>
        <Link
          to="/docs/mcp"
          className="inline-flex items-center gap-2 rounded-lg border border-surface-200 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-surface-50 hover:border-brand-300 transition-colors"
        >
          MCP Server docs
        </Link>
      </section>
    </article>
  );
}

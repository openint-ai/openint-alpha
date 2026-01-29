import { useState, useCallback } from 'react';
import { Link } from 'react-router-dom';

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
      className="absolute top-2 right-2 rounded-lg px-2 py-1 text-xs font-medium bg-surface-200 text-gray-700 hover:bg-surface-300 transition-colors z-10"
    >
      {copied ? 'Copied!' : label}
    </button>
  );
}

const CURSOR_MCP_JSON = `{
  "mcpServers": {
    "openint-semantic": {
      "command": "python",
      "args": ["/path/to/openint-mcp/server.py"],
      "env": {
        "OPENINT_BACKEND_URL": "http://localhost:3001"
      }
    }
  }
}`;

const TOOL_INTERPRET_INPUT = `{
  "sentence": "Show me transactions for customer 1001 in California",
  "model": "mukaj/fin-mpnet-base"
}`;

const TOOL_INTERPRET_OUTPUT = `{
  "success": true,
  "query": "Show me transactions for customer 1001 in California",
  "model": "mukaj/fin-mpnet-base",
  "tags": [
    { "type": "customer_id", "label": "Customer ID", "value": "1001", "snippet": "1001" },
    { "type": "state", "label": "State", "value": "California", "snippet": "California" },
    { "type": "intent", "label": "Intent", "value": "transactions", "snippet": "transactions" }
  ],
  "highlighted_segments": [...],
  "token_semantics": [
    { "token": "1001", "meanings": ["Customer ID:customer_id"] },
    { "token": "california", "meanings": ["State:state"] }
  ],
  "embedding_stats": { "dimension": 768 }
}`;

export default function McpDocs() {
  return (
    <article className="doc-page max-w-4xl">
      {/* Breadcrumb */}
      <nav className="text-xs text-gray-500 mb-6" aria-label="Breadcrumb">
        <span className="text-gray-400">Documentation</span>
        <span className="mx-1.5">/</span>
        <span className="text-gray-700 font-medium">MCP Server</span>
      </nav>

      {/* Hero */}
      <header className="mb-10">
        <h1 className="text-3xl font-bold text-gray-900 tracking-tight">
          MCP Server — Semantic Layer for Agents
        </h1>
        <p className="mt-3 text-lg text-gray-600 leading-relaxed">
          Expose the OpenInt semantic layer to any MCP-capable agent (Cursor, Claude Desktop, custom assistants).
          Agents call tools to interpret natural language sentences and list embedding models—no direct HTTP required.
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          <span className="inline-flex items-center rounded-full bg-brand-100 px-3 py-1 text-xs font-medium text-brand-800">
            Model Context Protocol
          </span>
          <span className="inline-flex items-center rounded-full bg-surface-200 px-3 py-1 text-xs font-medium text-gray-700">
            Python 3.10+
          </span>
          <span className="inline-flex items-center rounded-full bg-surface-200 px-3 py-1 text-xs font-medium text-gray-700">
            stdio / HTTP
          </span>
        </div>
      </header>

      {/* Overview */}
      <section id="overview" className="mb-10 scroll-mt-6">
        <h2 className="text-xl font-semibold text-gray-900 border-b border-surface-200 pb-2 mb-4">
          Overview
        </h2>
        <p className="text-gray-700 leading-relaxed mb-4">
          The OpenInt MCP server is a thin bridge: it exposes two tools and forwards calls to the OpenInt backend HTTP API.
          The backend runs the embedding models and semantic logic; the MCP server stays lightweight and stateless.
        </p>
        <div className="rounded-xl border border-surface-200 bg-surface-50 p-4 font-mono text-sm text-gray-800">
          <div className="text-gray-500 text-xs uppercase tracking-wider mb-2">Flow</div>
          Agent → MCP (stdio/HTTP) → OpenInt backend (GET /api/semantic/interpret, /api/semantic/models)
        </div>
      </section>

      {/* Quick start */}
      <section id="quick-start" className="mb-10 scroll-mt-6">
        <h2 className="text-xl font-semibold text-gray-900 border-b border-surface-200 pb-2 mb-4">
          Quick start
        </h2>
        <p className="text-gray-700 leading-relaxed mb-4">
          Prerequisites: <strong>Python 3.10+</strong> and the <strong>OpenInt backend</strong> running (e.g. <code className="rounded bg-surface-200 px-1.5 py-0.5 text-sm">http://localhost:3001</code>).
        </p>
        <div className="space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-gray-800 mb-2">1. Install</h3>
            <div className="rounded-lg border border-surface-200 bg-gray-900 p-4 relative">
              <CopyButton text="cd openint-mcp && pip install -r requirements.txt" />
              <pre className="pr-20 text-sm text-gray-100 font-mono overflow-x-auto">
                cd openint-mcp &amp;&amp; pip install -r requirements.txt
              </pre>
            </div>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-gray-800 mb-2">2. Configure</h3>
            <div className="rounded-lg border border-surface-200 bg-gray-900 p-4 relative">
              <CopyButton text="export OPENINT_BACKEND_URL=http://localhost:3001" />
              <pre className="pr-20 text-sm text-gray-100 font-mono overflow-x-auto">
                export OPENINT_BACKEND_URL=http://localhost:3001
              </pre>
            </div>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-gray-800 mb-2">3. Run (stdio)</h3>
            <div className="rounded-lg border border-surface-200 bg-gray-900 p-4 relative">
              <CopyButton text="python server.py" />
              <pre className="pr-20 text-sm text-gray-100 font-mono overflow-x-auto">python server.py</pre>
            </div>
            <p className="mt-2 text-sm text-gray-600">
              The server reads/writes JSON-RPC on stdin/stdout. Use this for Cursor and Claude Desktop.
            </p>
          </div>
        </div>
      </section>

      {/* Tools reference */}
      <section id="tools" className="mb-10 scroll-mt-6">
        <h2 className="text-xl font-semibold text-gray-900 border-b border-surface-200 pb-2 mb-4">
          Tools reference
        </h2>

        <div className="space-y-8">
          <div className="rounded-xl border border-surface-200 bg-white p-5 shadow-sm">
            <div className="flex items-center gap-2 mb-2">
              <span className="inline-flex items-center rounded bg-brand-100 px-2 py-0.5 text-xs font-semibold text-brand-800">
                Tool
              </span>
              <code className="text-base font-mono font-semibold text-gray-900">semantic_interpret</code>
            </div>
            <p className="text-gray-700 text-sm mb-4">
              Interpret a natural language sentence with the semantic layer. Returns tags (e.g. customer_id, state, intent), highlighted segments, token semantics, and optional embedding stats.
            </p>
            <div className="grid gap-3 sm:grid-cols-2">
              <div>
                <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">Input</div>
                <div className="rounded-lg border border-surface-200 bg-surface-50 p-3 relative">
                  <CopyButton text={TOOL_INTERPRET_INPUT} />
                  <pre className="pr-14 text-xs text-gray-800 font-mono whitespace-pre-wrap overflow-x-auto">
                    {TOOL_INTERPRET_INPUT}
                  </pre>
                </div>
              </div>
              <div>
                <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">Output (excerpt)</div>
                <div className="rounded-lg border border-surface-200 bg-surface-50 p-3 relative max-h-48 overflow-auto">
                  <CopyButton text={TOOL_INTERPRET_OUTPUT} />
                  <pre className="pr-14 text-xs text-gray-800 font-mono whitespace-pre-wrap overflow-x-auto">
                    {TOOL_INTERPRET_OUTPUT}
                  </pre>
                </div>
              </div>
            </div>
            <p className="mt-3 text-xs text-gray-500">
              <strong>Arguments:</strong> <code>sentence</code> (string, required), <code>model</code> (string, optional, default <code>mukaj/fin-mpnet-base</code>).
            </p>
          </div>

          <div className="rounded-xl border border-surface-200 bg-white p-5 shadow-sm">
            <div className="flex items-center gap-2 mb-2">
              <span className="inline-flex items-center rounded bg-brand-100 px-2 py-0.5 text-xs font-semibold text-brand-800">
                Tool
              </span>
              <code className="text-base font-mono font-semibold text-gray-900">semantic_interpret_all</code>
            </div>
            <p className="text-gray-700 text-sm mb-2">
              Interpret a sentence with <strong>all</strong> supported models (same as the UI dropdown). Returns one result per model: <code>models: {`{ model_id: result }`}</code>.
            </p>
            <p className="text-xs text-gray-500">
              <strong>Arguments:</strong> <code>sentence</code> (string, required). Returns <code>success</code>, <code>query</code>, <code>models</code>.
            </p>
          </div>

          <div className="rounded-xl border border-surface-200 bg-white p-5 shadow-sm">
            <div className="flex items-center gap-2 mb-2">
              <span className="inline-flex items-center rounded bg-brand-100 px-2 py-0.5 text-xs font-semibold text-brand-800">
                Tool
              </span>
              <code className="text-base font-mono font-semibold text-gray-900">semantic_list_models</code>
            </div>
            <p className="text-gray-700 text-sm mb-2">
              List available embedding model IDs. Use these as the <code>model</code> argument for <code>semantic_interpret</code>.
            </p>
            <p className="text-xs text-gray-500">
              <strong>Arguments:</strong> None. Returns <code>success</code>, <code>models</code>, <code>count</code>.
            </p>
          </div>

          <div className="rounded-xl border border-surface-200 bg-white p-5 shadow-sm">
            <div className="flex items-center gap-2 mb-2">
              <span className="inline-flex items-center rounded bg-brand-100 px-2 py-0.5 text-xs font-semibold text-brand-800">
                Tool
              </span>
              <code className="text-base font-mono font-semibold text-gray-900">semantic_list_models_with_meta</code>
            </div>
            <p className="text-gray-700 text-sm mb-2">
              List supported models with metadata: id, display_name, author, description, details, url (Hugging Face).
            </p>
            <p className="text-xs text-gray-500">
              <strong>Arguments:</strong> None. Returns <code>success</code>, <code>models</code>, <code>count</code>.
            </p>
          </div>
        </div>
      </section>

      {/* Cursor / Claude */}
      <section id="cursor-claude" className="mb-10 scroll-mt-6">
        <h2 className="text-xl font-semibold text-gray-900 border-b border-surface-200 pb-2 mb-4">
          Cursor & Claude Desktop
        </h2>
        <p className="text-gray-700 leading-relaxed mb-4">
          Add the MCP server in your client config. Use the <strong>full path</strong> to <code className="rounded bg-surface-200 px-1.5 py-0.5">server.py</code> in <code className="rounded bg-surface-200 px-1.5 py-0.5">openint-mcp</code>.
        </p>
        <div className="rounded-xl border border-surface-200 bg-gray-900 p-5 relative">
          <CopyButton text={CURSOR_MCP_JSON} />
          <pre className="pr-14 text-sm text-gray-100 font-mono whitespace-pre-wrap overflow-x-auto">
            {CURSOR_MCP_JSON}
          </pre>
        </div>
        <ul className="mt-4 text-sm text-gray-700 list-disc list-inside space-y-1">
          <li>
            <strong>Cursor:</strong> Settings → MCP → Add server, or edit <code className="rounded bg-surface-100 px-1">.cursor/mcp.json</code> (or your project MCP config).
          </li>
          <li>
            <strong>Claude Desktop:</strong> Add the server entry in <code className="rounded bg-surface-100 px-1">claude_desktop_config.json</code>; see{' '}
            <a href="https://modelcontextprotocol.io" target="_blank" rel="noopener noreferrer" className="text-brand-600 hover:text-brand-700 underline">
              modelcontextprotocol.io
            </a>.
          </li>
        </ul>
      </section>

      {/* Related */}
      <section id="related" className="mb-6 scroll-mt-6">
        <h2 className="text-xl font-semibold text-gray-900 border-b border-surface-200 pb-2 mb-4">
          Related
        </h2>
        <div className="flex flex-wrap gap-3">
          <Link
            to="/docs/api"
            className="inline-flex items-center gap-2 rounded-lg border border-surface-200 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-surface-50 hover:border-brand-300 transition-colors"
          >
            REST API docs
          </Link>
          <a
            href="https://modelcontextprotocol.io"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 rounded-lg border border-surface-200 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-surface-50 transition-colors"
          >
            MCP specification
          </a>
        </div>
      </section>
    </article>
  );
}

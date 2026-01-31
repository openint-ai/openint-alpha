/**
 * Marketing / Landing page — hero with image and product list.
 * First page of the app; Products dropdown links to each product.
 */

import { Link } from 'react-router-dom';

const PRODUCTS = [
  {
    path: '/chat',
    label: 'Vector DB Chat Demo',
    description: 'Natural language search over your data with vector embeddings. Ask questions in plain English and get answers with sources from the vector store.',
    icon: '🔍',
  },
  {
    path: '/compare',
    label: 'Sentence Annotation Compare',
    description: 'Compare how multiple semantic models interpret the same sentence. See tags, highlights, and consistency across finance and general-purpose models.',
    icon: '📊',
  },
  {
    path: '/a2a',
    label: 'A2A Demo',
    description: 'Watch the Agent-to-Agent (A2A) flow: sa-agent generates sentences from your schema, modelmgmt-agent annotates them with semantic tags in real time.',
    icon: '🤝',
  },
  {
    path: '/graph',
    label: 'Neo4J Graph Demo',
    description: 'Explore the graph database built from DataHub schema. Customers, transactions, disputes and relationships — browse counts and sample paths.',
    icon: '🕸️',
  },
] as const;

export default function Marketing() {
  return (
    <div className="min-h-screen">
      {/* Hero */}
      <section className="relative overflow-hidden bg-gradient-to-br from-surface-950 via-surface-900 to-brand-900/30">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_60%_at_50%_-20%,rgba(16,185,159,0.15),transparent)]" />
        <div className="relative max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 pt-16 sm:pt-20 pb-12 sm:pb-16">
          <div className="grid lg:grid-cols-2 gap-10 lg:gap-14 items-center">
            <div>
              <p className="text-brand-300 text-sm font-medium tracking-wider uppercase mb-3">
                Intelligent data & agents
              </p>
              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-white tracking-tight leading-[1.1]">
                Search, understand, and act on your data
              </h1>
              <p className="mt-5 text-lg text-slate-300 max-w-xl">
                openInt brings vector search, semantic annotation, agent-to-agent workflows, and graph exploration into one platform. Try the products below.
              </p>
              <div className="mt-8 flex flex-wrap gap-3">
                <Link
                  to="/chat"
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-brand-500 text-white font-medium text-sm hover:bg-brand-400 focus:outline-none focus:ring-2 focus:ring-brand-400 focus:ring-offset-2 focus:ring-offset-surface-900 transition-colors"
                >
                  Try Vector Chat
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                  </svg>
                </Link>
                <a
                  href="#products"
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl border border-slate-500/60 text-slate-200 font-medium text-sm hover:bg-white/5 hover:border-slate-400 transition-colors"
                >
                  View all products
                </a>
              </div>
            </div>
            <div className="relative flex justify-center lg:justify-end">
              <div className="relative w-full max-w-lg aspect-[4/3] rounded-2xl overflow-hidden shadow-2xl shadow-black/40 ring-1 ring-white/10">
                <img
                  src="/ai-agent-hero.png"
                  alt="AI agent and data intelligence"
                  className="w-full h-full object-cover object-center"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-surface-950/60 via-transparent to-transparent pointer-events-none" />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Products */}
      <section id="products" className="bg-surface-50 py-16 sm:py-20">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl sm:text-4xl font-bold text-surface-900">
              Products
            </h2>
            <p className="mt-3 text-lg text-surface-600 max-w-2xl mx-auto">
              Explore demos for vector search, semantic comparison, A2A flows, and the Neo4j graph.
            </p>
          </div>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {PRODUCTS.map((product) => (
              <Link
                key={product.path}
                to={product.path}
                className="group relative flex flex-col rounded-2xl border border-surface-200 bg-white p-6 shadow-soft hover:shadow-glow hover:border-brand-200 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-brand-400 focus:ring-offset-2 focus:ring-offset-surface-50"
              >
                <div className="text-2xl mb-4" aria-hidden>
                  {product.icon}
                </div>
                <h3 className="text-lg font-semibold text-surface-900 group-hover:text-brand-700 transition-colors">
                  {product.label}
                </h3>
                <p className="mt-2 text-sm text-surface-600 flex-1">
                  {product.description}
                </p>
                <span className="mt-4 inline-flex items-center gap-1.5 text-sm font-medium text-brand-600 group-hover:gap-2 transition-all">
                  Open demo
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                  </svg>
                </span>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* Footer strip */}
      <footer className="border-t border-surface-200 bg-white py-6">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-wrap items-center justify-between gap-4">
          <span className="text-sm text-surface-500">
            openInt — vector search, semantic annotation, A2A, graph
          </span>
          <Link
            to="/help"
            className="text-sm font-medium text-brand-600 hover:text-brand-700"
          >
            Help & docs
          </Link>
        </div>
      </footer>
    </div>
  );
}

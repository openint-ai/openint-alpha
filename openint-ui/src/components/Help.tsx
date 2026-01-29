export default function Help() {
  const links = [
    {
      title: 'Milvus documentation',
      description: 'Vector database docs, concepts, and API reference',
      href: 'https://milvus.io/docs',
    },
    {
      title: 'PyMilvus API reference',
      description: 'Python client for Milvus',
      href: 'https://milvus.io/api-reference/pymilvus/v2.4.x/About.md',
    },
    {
      title: 'Sentence Transformers',
      description: 'Embeddings models used for semantic search',
      href: 'https://www.sbert.net/',
    },
    {
      title: 'API health check',
      description: 'Check server and Milvus connection status',
      href: '/health',
    },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-gray-900 mb-1">Help & documentation</h1>
        <p className="text-sm text-gray-500">
          Links to documentation and APIs used by openInt.ai
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-1">
        {links.map((item) => (
          <a
            key={item.href}
            href={item.href}
            target={item.href.startsWith('http') ? '_blank' : undefined}
            rel={item.href.startsWith('http') ? 'noopener noreferrer' : undefined}
            className="block rounded-xl border border-surface-200 bg-white p-4 shadow-sm hover:border-brand-300 hover:shadow-md transition-all group"
          >
            <div className="font-medium text-gray-900 group-hover:text-brand-600 transition-colors">
              {item.title}
            </div>
            <div className="mt-1 text-sm text-gray-500">{item.description}</div>
            <div className="mt-2 text-xs text-brand-600 font-medium">
              {item.href.startsWith('http') ? item.href : `${window.location.origin}${item.href}`}
            </div>
          </a>
        ))}
      </div>

      <div className="rounded-xl border border-surface-200 bg-surface-50 p-4 text-sm text-gray-600">
        <p className="font-medium text-gray-800 mb-1">About this app</p>
        <p>
          openInt.ai uses Milvus for vector search over customers, transactions, addresses, states, and ZIP codes.
          Chat answers and search results are built from your indexed data. Load data with{' '}
          <code className="rounded bg-surface-200 px-1.5 py-0.5 text-xs">load_bank_data_to_milvus.py</code>.
        </p>
      </div>
    </div>
  );
}

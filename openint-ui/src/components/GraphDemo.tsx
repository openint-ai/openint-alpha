/**
 * Neo4J Graph Demo
 * Shows graph schema (from DataHub + loader), Neo4j connectivity, node/relationship counts,
 * and sample data: disputes overview and customer–transaction–dispute paths.
 */

import { useState, useEffect, useCallback } from 'react';
import {
  fetchGraphStats,
  fetchGraphSchema,
  fetchGraphSample,
  type GraphStats,
  type GraphSchemaResponse,
  type GraphSampleResponse,
} from '../api';

export default function GraphDemo() {
  const [stats, setStats] = useState<GraphStats | null>(null);
  const [schema, setSchema] = useState<GraphSchemaResponse | null>(null);
  const [sample, setSample] = useState<GraphSampleResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    setLoading(true);
    try {
      const [statsRes, schemaRes, sampleRes] = await Promise.all([
        fetchGraphStats(),
        fetchGraphSchema(),
        fetchGraphSample(),
      ]);
      setStats(statsRes);
      setSchema(schemaRes);
      setSample(sampleRes);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  if (loading) {
    return (
      <div className="w-full max-w-[90rem] mx-auto px-1">
        <h1 className="text-xl font-semibold text-gray-900 mb-1">Neo4J Graph Demo</h1>
        <p className="text-sm text-gray-600 mb-6">
          Graph schema from DataHub and openInt loader: Customer, Transaction, Dispute nodes and relationships.
        </p>
        <div className="rounded-xl border border-surface-200 bg-white shadow-sm p-8 text-center text-gray-500">
          Loading graph stats, schema, and sample data…
        </div>
      </div>
    );
  }

  return (
    <div className="w-full max-w-[90rem] mx-auto px-1">
      <h1 className="text-xl font-semibold text-gray-900 mb-1">Neo4J Graph Demo</h1>
      <p className="text-sm text-gray-600 mb-6">
        Graph schema from <strong>DataHub</strong> (openint-datahub/schemas.py) and the Neo4j loader (openint-testdata/loaders/load_openint_data_to_neo4j.py):
        Customer, Transaction, Dispute nodes with HAS_TRANSACTION, OPENED_DISPUTE, REFERENCES relationships.
      </p>

      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 p-4 mb-6 text-red-800 text-sm">
          {error}
          <button
            type="button"
            onClick={load}
            className="ml-3 px-3 py-1 rounded-lg bg-red-100 hover:bg-red-200 text-red-800 text-sm font-medium"
          >
            Retry
          </button>
        </div>
      )}

      <div className="grid gap-6">
        {/* Schema */}
        {schema?.schema && (
          <section className="rounded-xl border border-surface-200 bg-white shadow-sm overflow-hidden">
            <h2 className="px-6 py-4 border-b border-surface-200 text-base font-semibold text-gray-900 bg-surface-50">
              Graph Schema
            </h2>
            <div className="p-6">
              <p className="text-sm text-gray-600 mb-4">{schema.schema.source}</p>
              <div className="grid sm:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-sm font-medium text-gray-700 mb-2">Node labels</h3>
                  <ul className="space-y-2">
                    {schema.schema.nodes.map((n) => (
                      <li key={n.label} className="flex items-start gap-2 text-sm">
                        <span className="inline-flex px-2 py-0.5 rounded bg-brand-100 text-brand-800 font-medium">
                          {n.label}
                        </span>
                        <span className="text-gray-600">{n.description}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h3 className="text-sm font-medium text-gray-700 mb-2">Relationships</h3>
                  <ul className="space-y-2">
                    {schema.schema.relationships.map((r) => (
                      <li key={r.type} className="text-sm">
                        <span className="font-medium text-gray-800">{r.from}</span>
                        <span className="text-gray-400 mx-1">—[{r.type}]→</span>
                        <span className="font-medium text-gray-800">{r.to}</span>
                        <span className="text-gray-600 ml-1">· {r.description}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* Stats */}
        {stats && (
          <section className="rounded-xl border border-surface-200 bg-white shadow-sm overflow-hidden">
            <h2 className="px-6 py-4 border-b border-surface-200 text-base font-semibold text-gray-900 bg-surface-50">
              Neo4j connection &amp; counts
            </h2>
            <div className="p-6">
              <div className="flex items-center gap-2 mb-4">
                <span
                  className={`inline-flex w-3 h-3 rounded-full ${stats.connected ? 'bg-green-500' : 'bg-red-500'}`}
                  aria-hidden
                />
                <span className="text-sm font-medium">{stats.connected ? 'Connected' : 'Not connected'}</span>
                {stats.error && <span className="text-sm text-red-600">{stats.error}</span>}
              </div>
              {stats.connected && stats.node_counts && (
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                  <div className="rounded-lg border border-surface-200 p-3">
                    <div className="text-xs text-gray-500 uppercase tracking-wider">Customers</div>
                    <div className="text-lg font-semibold text-gray-900">{stats.node_counts.Customer ?? 0}</div>
                  </div>
                  <div className="rounded-lg border border-surface-200 p-3">
                    <div className="text-xs text-gray-500 uppercase tracking-wider">Transactions</div>
                    <div className="text-lg font-semibold text-gray-900">{stats.node_counts.Transaction ?? 0}</div>
                  </div>
                  <div className="rounded-lg border border-surface-200 p-3">
                    <div className="text-xs text-gray-500 uppercase tracking-wider">Disputes</div>
                    <div className="text-lg font-semibold text-gray-900">{stats.node_counts.Dispute ?? 0}</div>
                  </div>
                </div>
              )}
              {stats.connected && stats.relationship_counts && (
                <div className="mt-4 pt-4 border-t border-surface-200">
                  <div className="text-xs text-gray-500 uppercase tracking-wider mb-2">Relationships</div>
                  <div className="flex flex-wrap gap-3 text-sm">
                    <span>HAS_TRANSACTION: <strong>{stats.relationship_counts.HAS_TRANSACTION ?? 0}</strong></span>
                    <span>OPENED_DISPUTE: <strong>{stats.relationship_counts.OPENED_DISPUTE ?? 0}</strong></span>
                    <span>REFERENCES: <strong>{stats.relationship_counts.REFERENCES ?? 0}</strong></span>
                  </div>
                </div>
              )}
              <button
                type="button"
                onClick={load}
                className="mt-4 px-4 py-2 rounded-lg bg-brand-500 text-white text-sm font-medium hover:bg-brand-600"
              >
                Refresh
              </button>
            </div>
          </section>
        )}

        {/* Sample: disputes overview */}
        {sample?.success && sample.disputes_overview?.length !== undefined && (
          <section className="rounded-xl border border-surface-200 bg-white shadow-sm overflow-hidden">
            <h2 className="px-6 py-4 border-b border-surface-200 text-base font-semibold text-gray-900 bg-surface-50">
              Sample: Customer → Dispute → Transaction
            </h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-surface-200 bg-surface-50">
                    <th className="text-left px-4 py-3 font-medium text-gray-700">Customer ID</th>
                    <th className="text-left px-4 py-3 font-medium text-gray-700">Dispute ID</th>
                    <th className="text-left px-4 py-3 font-medium text-gray-700">Transaction ID</th>
                    <th className="text-left px-4 py-3 font-medium text-gray-700">Status</th>
                    <th className="text-right px-4 py-3 font-medium text-gray-700">Amount</th>
                    <th className="text-left px-4 py-3 font-medium text-gray-700">Currency</th>
                  </tr>
                </thead>
                <tbody>
                  {sample.disputes_overview.length === 0 ? (
                    <tr>
                      <td colSpan={6} className="px-4 py-6 text-center text-gray-500">
                        No disputes in graph. Load data with openint-testdata/loaders/load_openint_data_to_neo4j.py
                      </td>
                    </tr>
                  ) : (
                    sample.disputes_overview.map((row, i) => (
                      <tr key={i} className="border-b border-surface-100 hover:bg-surface-50">
                        <td className="px-4 py-2 font-mono text-gray-800">{row.customer_id ?? '—'}</td>
                        <td className="px-4 py-2 font-mono text-gray-800">{row.dispute_id ?? '—'}</td>
                        <td className="px-4 py-2 font-mono text-gray-800">{row.transaction_id ?? '—'}</td>
                        <td className="px-4 py-2 text-gray-700">{row.status ?? '—'}</td>
                        <td className="px-4 py-2 text-right text-gray-700">{row.amount_disputed ?? '—'}</td>
                        <td className="px-4 py-2 text-gray-600">{row.currency ?? '—'}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* Sample: paths (Customer - HAS_TRANSACTION -> Transaction <- REFERENCES - Dispute) */}
        {sample?.success && sample.paths?.length !== undefined && (
          <section className="rounded-xl border border-surface-200 bg-white shadow-sm overflow-hidden">
            <h2 className="px-6 py-4 border-b border-surface-200 text-base font-semibold text-gray-900 bg-surface-50">
              Sample: Customer → Transaction ← Dispute (paths)
            </h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-surface-200 bg-surface-50">
                    <th className="text-left px-4 py-3 font-medium text-gray-700">Customer ID</th>
                    <th className="text-left px-4 py-3 font-medium text-gray-700">Transaction ID</th>
                    <th className="text-left px-4 py-3 font-medium text-gray-700">Dispute ID</th>
                    <th className="text-right px-4 py-3 font-medium text-gray-700">Tx amount</th>
                    <th className="text-right px-4 py-3 font-medium text-gray-700">Disputed</th>
                    <th className="text-left px-4 py-3 font-medium text-gray-700">Dispute status</th>
                  </tr>
                </thead>
                <tbody>
                  {sample.paths.length === 0 ? (
                    <tr>
                      <td colSpan={6} className="px-4 py-6 text-center text-gray-500">
                        No paths in graph.
                      </td>
                    </tr>
                  ) : (
                    sample.paths.map((row, i) => (
                      <tr key={i} className="border-b border-surface-100 hover:bg-surface-50">
                        <td className="px-4 py-2 font-mono text-gray-800">{row.customer_id ?? '—'}</td>
                        <td className="px-4 py-2 font-mono text-gray-800">{row.transaction_id ?? '—'}</td>
                        <td className="px-4 py-2 font-mono text-gray-800">{row.dispute_id ?? '—'}</td>
                        <td className="px-4 py-2 text-right text-gray-700">{row.tx_amount ?? '—'}</td>
                        <td className="px-4 py-2 text-right text-gray-700">{row.amount_disputed ?? '—'}</td>
                        <td className="px-4 py-2 text-gray-600">{row.dispute_status ?? '—'}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}

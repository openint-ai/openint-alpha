# Enrich Agent

Extracts customer_id, transaction_id, dispute_id from **trusted sources only** (graph-agent rows, vectordb-agent results) and enriches them via Neo4j. Does not use the user query or sg-agent sentence to avoid hallucinated IDs.

## Capabilities

- **enrich**: Extracts IDs from graph rows (explicit columns) and vector results (indexed content), validates each in Neo4j, returns only verified entity details.

## Trusted Sources (no hallucination)

1. **Graph rows** (primary): Explicit `customer_id`, `transaction_id`, `dispute_id` columns from graph-agent's Neo4j query results. Source of truth.
2. **Vector results** (secondary): Content from VectorDB agent's Milvus search. Real indexed documents.
3. **Excluded**: User query and sg-agent sentence—these can contain example or incorrect IDs.

## Flow

```
sg-agent (sentence) → sentiment-agent → [vectordb-agent | graph-agent] (parallel)
                                                    ↓
                                            enrich-agent
                                    (extracts IDs from graph_rows + vector_results only,
                                     validates in Neo4j, returns verified details)
                                                    ↓
                                            aggregator-agent
```

## Usage

1. **Multi-Agent Demo**: Backend passes `graph_rows` and `vector_results` with `trusted_only: true`. Enrich-agent only enriches IDs from those sources.
2. **LangGraph**: Selected when the query mentions "details", "lookup", "information about", etc.
3. **UI**: Entity-type icons and popup for verified IDs.

## ID Formats Supported

- 10+ digit numbers (e.g. `1000000001`)
- `CUST` + digits
- `TX`/`TRANSACTION` + digits
- `DBT`/`DSP`/`DISPUTE` + digits
- `ACH`/`WIRE`/`CREDIT`/`DEBIT`/`CHECK` + digits

## Dependencies

- Neo4j (openint-graph) with Customer, Transaction, Dispute nodes loaded (see `openint-testdata/loaders/load_openint_data_to_neo4j.py`).

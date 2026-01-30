# OpenInt Graph (Neo4j)

Neo4j client and GraphAgent support for relationship and path queries (customers, transactions, disputes).

## Connecting to Neo4j (Docker Desktop + DataHub)

When Neo4j runs with DataHub via Docker Desktop, it is exposed on **bolt://localhost:7687**. The client defaults are set for that setup: URI `bolt://localhost:7687`, user/password `neo4j`/`datahub`, database `graph.db`. No env vars are required; ensure the DataHub stack is up (e.g. `docker-compose -f docker-compose.datahub.yml up -d`). To verify from the host:

```bash
python -c "from openint_graph.neo4j_client import Neo4jClient; c = Neo4jClient(); print('Connected:', c.verify_connectivity())"
```

## Environment variables

Override defaults via environment (e.g. in `.env` or your process):

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Bolt URI |
| `NEO4J_USER` | `neo4j` | Username |
| `NEO4J_PASSWORD` | `datahub` | Password |
| `NEO4J_DATABASE` | `graph.db` | Database name (DataHub Neo4j default; override if DataHub’s Neo4j in docker-compose.datahub.yml) |

## Deployment

- **Development (Docker Desktop + DataHub):** Neo4j from `docker-compose.datahub.yml` is on ports 7474 (HTTP) and 7687 (Bolt). Defaults connect to bolt://localhost:7687 with database `graph.db`; no config needed.
- **Production:** Run Neo4j (or use a managed service), then set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, and optionally `NEO4J_DATABASE` where the backend and loader run.

Ensure the agent/backend process can reach Neo4j (localhost:7687 when using Docker Desktop).

## Loading data

From the repo root (or `openint-testdata/loaders`), after generating test data:

```bash
# Install neo4j driver if needed: pip install neo4j
# Defaults connect to Neo4j with DataHub (bolt://localhost:7687, graph.db)
python openint-testdata/loaders/load_openint_data_to_neo4j.py
```

Options: `--max-customers`, `--max-transactions`, `--max-disputes`, `--only-customers`, `--only-transactions`, `--only-disputes`, `--skip-disputes`, `--batch-size`, `--data-dir`.

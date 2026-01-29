# OpenInt Graph (Neo4j)

Neo4j client and GraphAgent support for relationship and path queries (customers, transactions, disputes).

## Environment variables

Configure the Neo4j connection via environment (e.g. in `.env` or your process):

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Bolt URI |
| `NEO4J_USER` | `neo4j` | Username |
| `NEO4J_PASSWORD` | `datahub` | Password |
| `NEO4J_DATABASE` | `neo4j` | Database name (use `graph.db` when using DataHub’s Neo4j in docker-compose.datahub.yml) |

## Deployment

- **Development**: Use the existing Neo4j service in `docker-compose.datahub.yml` (ports 7474, 7687; auth `neo4j/datahub`; default database `graph.db`). Set `NEO4J_URI=bolt://localhost:7687`, `NEO4J_USER=neo4j`, `NEO4J_PASSWORD=datahub`, and `NEO4J_DATABASE=graph.db` so the backend and loader use the same DB.
- **Production**: Run Neo4j (or use a managed service), then set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, and optionally `NEO4J_DATABASE` in the environment where the backend and loader run.

Ensure the agent/backend process can reach Neo4j (same network or host/port).

## Loading data

From the repo root (or `openint-testdata/loaders`), after generating test data:

```bash
# Install neo4j driver if needed: pip install neo4j
# With DataHub Neo4j: NEO4J_DATABASE=graph.db
python openint-testdata/loaders/load_openint_data_to_neo4j.py
```

Options: `--max-customers`, `--max-transactions`, `--max-disputes`, `--only-customers`, `--only-transactions`, `--only-disputes`, `--skip-disputes`, `--batch-size`, `--data-dir`.

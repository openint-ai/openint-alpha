"""
Graph Agent
Performs relationship and path queries in Neo4j graph database
"""

import sys
import os
import threading
from typing import Dict, List, Any, Optional

# Add repo root and openint-graph to path
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_repo_root = os.path.abspath(os.path.join(_parent, ".."))
for p in (_parent, _repo_root):
    if p not in sys.path:
        sys.path.insert(0, p)
_openint_graph = os.path.join(_repo_root, "openint-graph")
if os.path.isdir(_openint_graph) and _openint_graph not in sys.path:
    sys.path.insert(0, _openint_graph)

try:
    from neo4j_client import Neo4jClient
except ImportError:
    try:
        from openint_graph.neo4j_client import Neo4jClient
    except ImportError:
        Neo4jClient = None

import logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent, AgentResponse
from communication.agent_registry import AgentCapability

logger = logging.getLogger(__name__)


# Cypher templates for graph queries (openint schema: Customer, Transaction, Dispute)
CYPHER_DISPUTES_OVERVIEW = """
MATCH (c:Customer)-[:OPENED_DISPUTE]->(d:Dispute)-[:REFERENCES]->(t:Transaction)
RETURN c.id AS customer_id, d.id AS dispute_id, t.id AS transaction_id,
       d.dispute_status AS status, d.amount_disputed AS amount_disputed, d.currency AS currency
LIMIT 50
"""

CYPHER_CUSTOMER_DISPUTES = """
MATCH (c:Customer)-[:OPENED_DISPUTE]->(d:Dispute)-[:REFERENCES]->(t:Transaction)
WHERE c.id = $customer_id
RETURN d.id AS dispute_id, t.id AS transaction_id, d.dispute_status, d.amount_disputed, d.currency
LIMIT 50
"""

CYPHER_PATH_CUSTOMER_TRANSACTION_DISPUTE = """
MATCH path = (c:Customer)-[:HAS_TRANSACTION]->(t:Transaction)<-[:REFERENCES]-(d:Dispute)
RETURN c.id AS customer_id, t.id AS transaction_id, d.id AS dispute_id,
       t.amount AS tx_amount, d.amount_disputed, d.dispute_status
LIMIT 30
"""

CYPHER_CUSTOMER_TRANSACTIONS = """
MATCH (c:Customer)-[:HAS_TRANSACTION]->(t:Transaction)
WHERE c.id = $customer_id
RETURN t.id AS transaction_id, t.type AS transaction_type, t.amount, t.currency
LIMIT 50
"""

CYPHER_LINKED_ACCOUNTS = """
MATCH (c:Customer)-[:HAS_TRANSACTION]->(t:Transaction)
RETURN c.id AS customer_id, count(t) AS transaction_count
ORDER BY transaction_count DESC
LIMIT 20
"""


def _path_summary(record: Dict[str, Any]) -> str:
    """Build a short text summary from a graph record."""
    parts = []
    for k, v in record.items():
        if v is not None and v != "":
            parts.append(f"{k}: {v}")
    return " | ".join(parts)


class GraphAgent(BaseAgent):
    """
    Agent for graph/relationship queries in Neo4j.
    Registers capability "graph" for routing by the orchestrator.
    """

    def __init__(self, neo4j_client: Optional[Any] = None):
        capabilities = [
            AgentCapability(
                name="graph",
                description="Relationship and path queries (customers, transactions, disputes)",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language or structured query"},
                    },
                },
                output_schema={
                    "type": "array",
                    "items": {"type": "object", "properties": {"content": {}, "metadata": {}}},
                },
            )
        ]
        super().__init__(
            name="graph_agent",
            description="Performs graph and relationship queries in Neo4j (customers, transactions, disputes)",
            capabilities=capabilities,
        )
        if neo4j_client is not None:
            self._client = neo4j_client
        elif Neo4jClient is not None:
            try:
                self._client = Neo4jClient()
            except Exception as e:
                logger.warning("Could not initialize Neo4j client: %s", e)
                self._client = None
        else:
            self._client = None

    # Timeout for graph queries so a slow/unreachable Neo4j does not block the whole request (seconds)
    GRAPH_QUERY_TIMEOUT = 5

    def process_query(self, query: str, context: Dict[str, Any] = None) -> AgentResponse:
        """
        Interpret query and run one or more Cypher queries; return results as AgentResponse
        with results shaped like SearchAgent (content, metadata including file_type: "graph_path").
        """
        if self._client is None:
            return AgentResponse(
                success=False,
                results=[],
                message="Neo4j client not available",
                metadata={"error": "Neo4j not initialized"},
            )
        try:
            self.update_status("BUSY")
            query_lower = (query or "").strip().lower()
            records: List[Dict[str, Any]] = []
            error_holder: List[str] = []

            def _run_queries() -> None:
                nonlocal records, error_holder
                try:
                    if "dispute" in query_lower and ("customer" in query_lower or "path" in query_lower or "link" in query_lower):
                        try:
                            rows = self._client.run(CYPHER_PATH_CUSTOMER_TRANSACTION_DISPUTE)
                            records.extend(rows)
                        except Exception as e:
                            error_holder.append(str(e))
                    if "dispute" in query_lower and not records:
                        try:
                            customer_id = None
                            for w in query_lower.replace(",", " ").split():
                                if w.startswith("cust") and len(w) >= 10:
                                    customer_id = w.upper()
                                    break
                            if customer_id:
                                rows = self._client.run(CYPHER_CUSTOMER_DISPUTES, {"customer_id": customer_id})
                            else:
                                rows = self._client.run(CYPHER_DISPUTES_OVERVIEW)
                            records.extend(rows)
                        except Exception as e:
                            error_holder.append(str(e))
                    if ("transaction" in query_lower or "account" in query_lower) and "customer" in query_lower:
                        customer_id = None
                        for w in query_lower.replace(",", " ").split():
                            if w.startswith("cust") and len(w) >= 10:
                                customer_id = w.upper()
                                break
                        if customer_id:
                            try:
                                rows = self._client.run(CYPHER_CUSTOMER_TRANSACTIONS, {"customer_id": customer_id})
                                records.extend(rows)
                            except Exception as e:
                                error_holder.append(str(e))
                    if ("link" in query_lower or "connected" in query_lower or "relationship" in query_lower) and not records:
                        try:
                            rows = self._client.run(CYPHER_LINKED_ACCOUNTS)
                            records.extend(rows)
                        except Exception as e:
                            error_holder.append(str(e))
                    if not records and ("graph" in query_lower or "related" in query_lower or "dispute" in query_lower):
                        try:
                            rows = self._client.run(CYPHER_DISPUTES_OVERVIEW)
                            records.extend(rows)
                        except Exception as e:
                            error_holder.append(str(e))
                except Exception as e:
                    error_holder.append(str(e))

            th = threading.Thread(target=_run_queries)
            th.daemon = True
            th.start()
            th.join(timeout=self.GRAPH_QUERY_TIMEOUT)
            if th.is_alive():
                logger.warning("GraphAgent query timed out after %ss (Neo4j slow or unreachable)", self.GRAPH_QUERY_TIMEOUT)
                records = []
                error_holder.append("Graph query timed out (Neo4j slow or unreachable)")

            # Format as list of { content, metadata } for backend aggregation
            results = []
            for rec in records[:50]:
                content = _path_summary(rec)
                results.append({
                    "id": rec.get("dispute_id") or rec.get("transaction_id") or rec.get("customer_id") or "",
                    "content": content,
                    "metadata": {
                        "file_type": "graph_path",
                        **{k: v for k, v in rec.items() if v is not None},
                    },
                })

            self.update_status("IDLE")
            return AgentResponse(
                success=True,
                results=results,
                message=f"Found {len(results)} graph result(s)",
                metadata={"query": query, "results_count": len(results)},
            )
        except Exception as e:
            self.update_status("ERROR")
            return AgentResponse(
                success=False,
                results=[],
                message=f"Graph query error: {str(e)}",
                metadata={"error": str(e)},
            )

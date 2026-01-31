"""
GraphDB Agent
Performs relationship and path queries in Neo4j graph database.
Uses natural language + Neo4j schema with an LLM (Ollama) to generate Cypher when available;
falls back to keyword-based template matching otherwise.
"""

import sys
import os
import json
import threading
import urllib.request
import urllib.error
import ssl
from typing import Dict, List, Any, Optional, Tuple

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

# Neo4j schema summary for LLM (matches openint-backend GRAPH_SCHEMA / load_openint_data_to_neo4j)
GRAPH_SCHEMA_SUMMARY = """
Neo4j schema (node labels and relationship types):
- Node label: Customer. id property: id. Customer dimension (from DataHub customers schema).
- Node label: Transaction. id property: id. type property: type (e.g. ach, wire, credit, debit, check). Transaction facts from DataHub fact tables.
- Node label: Dispute. id property: id. Dispute fact (from DataHub disputes schema).
Relationship types (from -> to):
- HAS_TRANSACTION: (Customer)-[:HAS_TRANSACTION]->(Transaction). Customer has transactions.
- OPENED_DISPUTE: (Customer)-[:OPENED_DISPUTE]->(Dispute). Customer opened a dispute.
- REFERENCES: (Dispute)-[:REFERENCES]->(Transaction). Dispute references a transaction.
"""


def _extract_cypher_from_llm(text: str) -> str:
    """Extract Cypher from LLM response; strip markdown code blocks if present.
    Rejects invalid Cypher (e.g. RETURN-only without MATCH — variable `d` not defined).
    """
    text = (text or "").strip()
    if not text:
        return ""

    def _validate(cypher: str) -> str:
        cypher = cypher.strip()
        if not cypher or "MATCH" not in cypher.upper():
            return ""
        return cypher

    if "```" in text:
        parts = text.split("```", 2)
        if len(parts) >= 2:
            block = parts[1].strip()
            if "\n" in block:
                first, rest = block.split("\n", 1)
                if first.strip().lower() in ("cypher", "neo4j"):
                    block = rest.strip()
            return _validate(block)
    return _validate(text)


def _generate_cypher_with_ollama(question: str) -> Tuple[Optional[str], Optional[str]]:
    """Use Ollama to generate Cypher from natural language. Returns (cypher, error)."""
    host = (os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL") or "qwen2.5:7b"
    prompt = f"""You are a Neo4j Cypher expert. Given the schema below and the user question, return ONLY a valid Cypher query. No explanation, no markdown, no code block wrapper.

{GRAPH_SCHEMA_SUMMARY}

Rules:
- In MATCH, use exactly these variable names: c for Customer, t for Transaction, d for Dispute. Example: MATCH (c:Customer)-[:OPENED_DISPUTE]->(d:Dispute)-[:REFERENCES]->(t:Transaction)
- In RETURN, only use the node variables c, d, t and their properties: c.id, t.id, d.id, t.amount, t.type, d.amount_disputed, d.dispute_status, t.currency, etc. Never use a relationship variable in RETURN (e.g. no r.target; use t.id for transaction_id).
- Relationship types: HAS_TRANSACTION (Customer->Transaction), OPENED_DISPUTE (Customer->Dispute), REFERENCES (Dispute->Transaction).
- Every variable in RETURN must appear in MATCH. Add LIMIT 50 for exploration queries.
- CRITICAL: All customer, transaction, and dispute IDs are exactly 10-digit numbers (e.g. 1000000001, 1000001234). Use them in Cypher as strings: c.id = '1000000001'. Never use prefixes like CUST or TX. Never treat IDs as amounts.
- Return ONLY the Cypher statement, nothing else.

User question: {question}"""
    try:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 1024},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{host}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=45, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        message = data.get("message") or {}
        text = (message.get("content") or "").strip()
        cypher = _extract_cypher_from_llm(text)
        if not cypher:
            return (None, "Ollama returned empty or invalid Cypher.")
        return (cypher, None)
    except urllib.error.URLError as e:
        msg = str(e.reason) if getattr(e, "reason", None) else str(e)
        if "Connection refused" in msg or "111" in msg:
            msg = "Ollama not running. Start ollama serve and pull a model (e.g. ollama pull qwen2.5:7b)."
        return (None, msg)
    except Exception as e:
        return (None, str(e))


def _path_summary(record: Dict[str, Any]) -> str:
    """Build a short text summary from a graph record."""
    parts = []
    for k, v in record.items():
        if v is not None and v != "":
            parts.append(f"{k}: {v}")
    return " | ".join(parts)


class GraphdbAgent(BaseAgent):
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
            name="graphdb-agent",
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
        Interpret query (natural language or structured) and run Cypher; return results as AgentResponse.
        Uses Ollama + Neo4j schema to generate Cypher from natural language when available;
        falls back to keyword-based template matching otherwise.
        """
        if self._client is None:
            return AgentResponse(
                success=False,
                results=[],
                message="Neo4j client not available",
                metadata={"error": "Neo4j not initialized"},
            )
        query = (query or "").strip()
        if not query:
            return AgentResponse(
                success=False,
                results=[],
                message="Empty query",
                metadata={"error": "Empty query"},
            )
        try:
            self.update_status("BUSY")
            records: List[Dict[str, Any]] = []
            cypher_used: Optional[str] = None
            # 1) Try LLM-generated Cypher (natural language -> Ollama + schema -> Cypher)
            cypher, llm_error = _generate_cypher_with_ollama(query)
            if cypher:
                try:
                    records = self._client.run(cypher) or []
                    cypher_used = cypher
                    logger.info("GraphdbAgent: ran LLM-generated Cypher (%d rows)", len(records))
                except Exception as e:
                    logger.warning("GraphdbAgent: LLM Cypher execution failed, falling back to templates", extra={"error": str(e)})
                    records = []
            # 2) Fallback: keyword-based template matching
            if not records:
                query_lower = query.lower()
                error_holder: List[str] = []

                def _run_templates() -> None:
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

                th = threading.Thread(target=_run_templates)
                th.daemon = True
                th.start()
                th.join(timeout=self.GRAPH_QUERY_TIMEOUT)
                if th.is_alive():
                    logger.warning("GraphdbAgent query timed out after %ss", self.GRAPH_QUERY_TIMEOUT)
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

            meta = {"query": query, "results_count": len(results)}
            if cypher_used:
                meta["cypher"] = cypher_used
            self.update_status("IDLE")
            return AgentResponse(
                success=True,
                results=results,
                message=f"Found {len(results)} graph result(s)",
                metadata=meta,
            )
        except Exception as e:
            self.update_status("ERROR")
            return AgentResponse(
                success=False,
                results=[],
                message=f"Graph query error: {str(e)}",
                metadata={"error": str(e)},
            )

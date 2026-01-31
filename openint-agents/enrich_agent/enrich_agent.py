"""
Enrich Agent
Extracts customer_id, transaction_id, dispute_id from trusted sources (graph rows,
vector results, user query) and enriches via Neo4j.

Uses:
- LLM (Ollama): Infer entity type (Customer/Transaction/Dispute) from text context
  when ID has no label hint (e.g. raw 10-digit ID in vector content).
- Neo4j (Graph): Direct lookup by ID + augmentation (transactions, disputes, related
  entities). Tries graph-agent via A2A first; falls back to direct Cypher when A2A
  fails or returns empty.
- Vector (search_agent via A2A): Only when trusted_only=False — fetches context for
  label inference. With trusted_only=True, vector content is already in context.
"""

import json
import logging
import os
import re
import ssl
import sys
import urllib.error
import urllib.request
from typing import Dict, List, Any, Optional, Set, Tuple

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent, AgentResponse
from communication.agent_registry import AgentCapability

logger = logging.getLogger(__name__)

# ID patterns: 10-digit numbers, CUST/TX/DBT/DSP prefixes
ID_PATTERNS = [
    (re.compile(r"\b(\d{10,})\b"), None),  # 10+ digit numeric
    (re.compile(r"\bCUST\s*(\d+)\b", re.I), "Customer"),
    (re.compile(r"\b(?:TX|TRANSACTION)\s*(\d+)\b", re.I), "Transaction"),
    (re.compile(r"\b(?:DBT|DSP|DISPUTE)\s*(\d+)\b", re.I), "Dispute"),
]

def _normalize_id_for_lookup(val: str) -> str:
    """Normalize ID for Neo4j lookup: handle 1000005586.0 -> 1000005586, CUST001 -> 1000000001."""
    s = str(val).strip()
    # Handle float string (e.g. "1000005586.0")
    if "." in s and s.replace(".", "", 1).replace("-", "", 1).isdigit():
        try:
            return str(int(float(s)))
        except (ValueError, TypeError):
            pass
    # Handle CUST/TX/DBT prefix -> 10-digit
    if any(s.upper().startswith(p) for p in ("CUST", "TX", "DBT", "DSP")):
        digits = "".join(c for c in s if c.isdigit())
        if digits:
            try:
                n = int(digits)
                return str(1000000000 + (n % 1000000000))
            except (TypeError, ValueError):
                pass
    return s


# Normalize prefixed IDs to 10-digit form (matches load_openint_data_to_neo4j)
def _normalize_id(raw: str, hint: Optional[str] = None) -> str:
    digits = "".join(c for c in raw if c.isdigit())
    if not digits:
        return raw
    try:
        n = int(digits)
        return str(1000000000 + (n % 1000000000))
    except (TypeError, ValueError):
        return raw


def _extract_ids_from_text(text: str) -> List[Tuple[str, Optional[str]]]:
    """Extract (normalized_id, label_hint) from text."""
    if not text or not isinstance(text, str):
        return []
    seen: Set[str] = set()
    result: List[Tuple[str, Optional[str]]] = []
    for pattern, label_hint in ID_PATTERNS:
        for m in pattern.finditer(text):
            raw = m.group(1)
            nid = _normalize_id(raw)
            if nid and nid not in seen:
                seen.add(nid)
                result.append((nid, label_hint))
    return result


# Patterns for explicit user query: "customer 1000003621", "transaction 1000032631", "dispute 1000001234"
_USER_QUERY_ID_PATTERNS = [
    (re.compile(r"\bcustomer\s+(\d{10,})\b", re.I), "Customer"),
    (re.compile(r"\btransaction\s+(?:id\s+)?(\d{10,})\b", re.I), "Transaction"),
    (re.compile(r"\bdispute\s+(?:id\s+)?(\d{10,})\b", re.I), "Dispute"),
]


def _extract_ids_from_user_query(text: str) -> List[Tuple[str, Optional[str]]]:
    """
    Extract IDs from user's raw query when they explicitly mention customer/transaction/dispute.
    Used as supplement so we never miss the ID the user asked about.
    """
    if not text or not isinstance(text, str):
        return []
    seen: Set[str] = set()
    out: List[Tuple[str, Optional[str]]] = []
    for pattern, label in _USER_QUERY_ID_PATTERNS:
        for m in pattern.finditer(text):
            raw = m.group(1)
            nid = _normalize_id(raw)
            if nid and nid not in seen:
                seen.add(nid)
                out.append((nid, label))
    return out


def _get_context_around_id(text: str, id_str: str, window: int = 100) -> str:
    """Get surrounding text around an ID for context inference."""
    if not text or not id_str:
        return ""
    pos = text.find(id_str)
    if pos < 0:
        # Try normalized form
        digits = "".join(c for c in id_str if c.isdigit())
        if digits:
            pos = text.find(digits)
    if pos < 0:
        return text[: window * 2] if len(text) > window * 2 else text
    start = max(0, pos - window)
    end = min(len(text), pos + len(id_str) + window)
    return text[start:end]


def _infer_label_from_context_llm(context: str) -> Optional[str]:
    """
    Use LLM to infer entity type (Customer, Transaction, Dispute) from text context.
    Returns label or None on failure.
    """
    if not context or len(context.strip()) < 5:
        return None
    host = (os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL") or "qwen2.5:7b"
    prompt = f"""In this text snippet, a numeric ID (10+ digits) appears. What entity type does it refer to?
Options: Customer, Transaction, Dispute.
Reply with ONLY one word: Customer, Transaction, or Dispute.

Text: {context[:400]}"""
    try:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 20},
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
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = (data.get("message") or {}).get("content") or ""
        content = content.strip().lower()
        if "customer" in content:
            return "Customer"
        if "transaction" in content:
            return "Transaction"
        if "dispute" in content:
            return "Dispute"
    except Exception as e:
        logger.debug("LLM context inference failed: %s", e)
    return None


def _infer_label_from_context_keywords(context: str) -> Optional[str]:
    """Rule-based fallback: infer label from keywords near the ID."""
    t = (context or "").lower()
    # Order matters: more specific first
    if any(w in t for w in ("dispute", "dsp", "disputed", "dispute_id")):
        return "Dispute"
    if any(w in t for w in ("transaction", "tx ", "tx_id", "wire", "ach", "merchant", "amount", "payment")):
        return "Transaction"
    if any(w in t for w in ("customer", "cust", "account", "customer_id", "first_name", "last_name")):
        return "Customer"
    return None


def _infer_label_from_context(context: str) -> Optional[str]:
    """Infer Customer/Transaction/Dispute from text context. Uses LLM with keyword fallback."""
    label = _infer_label_from_context_llm(context)
    if label:
        return label
    return _infer_label_from_context_keywords(context)


# Explicit graph row columns -> Neo4j label (source of truth, no hallucination)
# Supports both "customer_id" and LLM Cypher aliases like "c.id"
_GRAPH_ID_COLUMNS = [
    ("customer_id", "Customer"),
    ("transaction_id", "Transaction"),
    ("dispute_id", "Dispute"),
    ("c.id", "Customer"),
    ("t.id", "Transaction"),
    ("d.id", "Dispute"),
]


def _extract_ids_from_graph_rows(rows: List[Dict[str, Any]]) -> List[Tuple[str, Optional[str]]]:
    """
    Extract IDs ONLY from graph agent rows (explicit columns). No free-text parsing.
    Graph rows are the source of truth - IDs here were returned by Neo4j.
    """
    seen: Set[str] = set()
    out: List[Tuple[str, Optional[str]]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        for col, label in _GRAPH_ID_COLUMNS:
            val = row.get(col)
            if val is None:
                continue
            nid = _normalize_id(str(val).strip())
            if nid and nid not in seen:
                seen.add(nid)
                out.append((nid, label))
    return out


def _extract_ids_from_vector_results(results: List[Any], use_llm_context: bool = True) -> List[Tuple[str, Optional[str]]]:
    """
    Extract IDs from vector search result content. When hint is None (raw 10-digit ID),
    uses LLM + keyword rules to infer Customer/Transaction/Dispute from context.
    """
    seen: Set[str] = set()
    out: List[Tuple[str, Optional[str]]] = []
    for item in results or []:
        content = ""
        if isinstance(item, dict):
            content = item.get("content") or ""
            meta = item.get("metadata") or {}
            texts = [str(content)]
            for k, v in meta.items():
                if v is not None and k in ("customer_id", "transaction_id", "dispute_id"):
                    texts.append(str(v))
        elif isinstance(item, str):
            content = item
            texts = [content]
        else:
            continue
        for t in texts:
            for nid, hint in _extract_ids_from_text(t):
                if nid and nid not in seen:
                    seen.add(nid)
                    # Infer label from context when hint is None
                    if hint is None and use_llm_context and t:
                        ctx = _get_context_around_id(t, nid)
                        inferred = _infer_label_from_context(ctx)
                        if inferred:
                            hint = inferred
                    out.append((nid, hint))
    return out


def _extract_ids_from_results(results: List[Any], use_llm_context: bool = True) -> List[Tuple[str, Optional[str]]]:
    """Extract IDs from generic agent results. Uses LLM context inference when hint is None."""
    seen: Set[str] = set()
    out: List[Tuple[str, Optional[str]]] = []
    for item in results or []:
        content = ""
        if isinstance(item, str):
            content = item
            for nid, hint in _extract_ids_from_text(item):
                if nid not in seen:
                    seen.add(nid)
                    if hint is None and use_llm_context and content:
                        ctx = _get_context_around_id(content, nid)
                        inferred = _infer_label_from_context(ctx)
                        if inferred:
                            hint = inferred
                    out.append((nid, hint))
        elif isinstance(item, dict):
            content = item.get("content") or ""
            meta = item.get("metadata") or {}
            texts = [str(content)]
            for k, v in meta.items():
                if v is not None and isinstance(v, (str, int, float)):
                    texts.append(str(v))
            for t in texts:
                for nid, hint in _extract_ids_from_text(t):
                    if nid not in seen:
                        seen.add(nid)
                        if hint is None and use_llm_context and t:
                            ctx = _get_context_around_id(t, nid)
                            inferred = _infer_label_from_context(ctx)
                            if inferred:
                                hint = inferred
                        out.append((nid, hint))
        else:
            content = str(item)
            for nid, hint in _extract_ids_from_text(content):
                if nid not in seen:
                    seen.add(nid)
                    if hint is None and use_llm_context and content:
                        ctx = _get_context_around_id(content, nid)
                        inferred = _infer_label_from_context(ctx)
                        if inferred:
                            hint = inferred
                    out.append((nid, hint))
    return out


def _node_props_to_dict(node: Any) -> Dict[str, Any]:
    """Convert Neo4j Node to flat dict of properties."""
    if node is None:
        return {}
    if hasattr(node, "items"):
        return dict(node)
    if hasattr(node, "__iter__") and not isinstance(node, (str, bytes)):
        try:
            return dict(node)
        except (TypeError, ValueError):
            pass
    return {}


def _format_transaction_display(id_val: str, props: Dict[str, Any]) -> str:
    """
    Build rich transaction display: type, amount, date, merchant/payee (id).
    Includes all key transaction details: type (ach/wire/credit/debit/check),
    amount, transaction_date, merchant_name/payee_name/beneficiary_name/description.
    """
    parts: List[str] = []
    tx_channel = str(props.get("type") or "").strip().lower() or "transaction"
    tx_dir = str(props.get("transaction_type") or "").strip()
    if tx_dir:
        parts.append(f"{tx_channel} {tx_dir}")
    else:
        parts.append(tx_channel)

    amount = props.get("amount")
    if amount is not None:
        try:
            amt = float(amount)
            parts.append(f"${amt:,.2f}")
        except (TypeError, ValueError):
            pass

    merchant = (
        str(props.get("merchant_name") or props.get("payee_name") or props.get("beneficiary_name") or "")
        .strip()
        or str(props.get("description") or "").strip()
    )
    if merchant:
        parts.append(f"@{merchant[:35]}{'…' if len(merchant) > 35 else ''}")

    tx_date = props.get("transaction_date") or props.get("transaction_datetime")
    if tx_date is not None:
        try:
            d = str(tx_date)[:10] if len(str(tx_date)) >= 10 else str(tx_date)
            if d:
                parts.append(d)
        except (TypeError, ValueError):
            pass

    status = str(props.get("status") or "").strip()
    if status:
        parts.append(status)

    prefix = " — ".join(parts) if parts else "Transaction"
    return f"{prefix} ({id_val})"


def _format_dispute_display(id_val: str, props: Dict[str, Any]) -> str:
    """
    Build rich dispute display: amount, reason, status, date raised (id).
    Includes all key dispute details: amount_disputed, dispute_reason,
    dispute_status, dispute_date, and related transaction type.
    """
    parts: List[str] = []
    amt = props.get("amount_disputed") or props.get("amount")
    if amt is not None:
        try:
            a = float(amt)
            parts.append(f"${a:,.2f}")
        except (TypeError, ValueError):
            pass

    reason = str(props.get("dispute_reason") or "").strip()
    if reason:
        parts.append(reason[:40] + ("…" if len(reason) > 40 else ""))

    status = str(props.get("dispute_status") or "").strip()
    if status:
        parts.append(status)

    disp_date = props.get("dispute_date") or props.get("dispute_datetime") or props.get("created_at")
    if disp_date is not None:
        try:
            d = str(disp_date)[:10] if len(str(disp_date)) >= 10 else str(disp_date)
            if d:
                parts.append(d)
        except (TypeError, ValueError):
            pass

    tx_type = str(props.get("transaction_type") or "").strip()
    if tx_type:
        parts.append(tx_type)

    prefix = " — ".join(parts) if parts else "Dispute"
    return f"{prefix} ({id_val})"


def _format_display_name(label: str, node_id: str, props: Dict[str, Any]) -> str:
    """
    Build human-readable display name with meaningful enrichment.
    Customer: "First Last (ID: X, mobile: Y, email: Z)"
    Transaction/Dispute: type, amount, etc.
    """
    id_val = str(props.get("id", node_id))
    if label == "Customer":
        first = str(props.get("first_name") or "").strip()
        last = str(props.get("last_name") or "").strip()
        name = " ".join(x for x in (first, last) if x).strip() or "Customer"
        mobile = str(props.get("phone") or props.get("mobile") or "").strip()
        email = str(props.get("email") or "").strip()
        parts = [f"ID: {id_val}"]
        if mobile:
            parts.append(f"mobile: {mobile}")
        if email:
            parts.append(f"email: {email}")
        return f"{name} ({', '.join(parts)})"
    if label == "Transaction":
        return _format_transaction_display(id_val, props)
    if label == "Dispute":
        return _format_dispute_display(id_val, props)
    return f"{label} id: {id_val}"


def lookup_id_in_neo4j(
    client: "Neo4jClient",
    node_id: str,
    label_hint: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Look up a node by id in Neo4j. Tries label_hint first, then Customer, Transaction, Dispute.
    Uses toString(n.id) = $idStr for type-agnostic matching (handles string, int, float storage).
    Returns { "label": str, "id": str, "properties": dict } or None.
    """
    id_str = _normalize_id_for_lookup(node_id)
    if not id_str:
        return None
    id_candidates = [id_str]
    if id_str.isdigit():
        id_candidates.append(id_str + ".0")  # legacy: node stored as "1000005586.0"
    labels = []
    if label_hint and label_hint in ("Customer", "Transaction", "Dispute"):
        labels = [label_hint]
    if not labels:
        labels = ["Customer", "Transaction", "Dispute"]

    for label in labels:
        for cand in id_candidates:
            try:
                # Type-agnostic: matches n.id whether stored as string, int, or float
                cypher = f"MATCH (n:{label}) WHERE toString(n.id) = $idStr RETURN n"
                rows = client.run(cypher, {"idStr": cand}) or []
                if not rows:
                    continue
                rec = rows[0]
                node = rec.get("n")
                props = _node_props_to_dict(node) if node else {}
                display_name = _format_display_name(label, node_id, props)
                return {
                    "label": label,
                    "id": node_id,
                    "properties": props,
                    "display_name": display_name,
                }
            except Exception as e:
                logger.debug("Neo4j lookup %s as %s failed: %s", node_id, label, e)
                continue
    return None


def _get_agent_instance(agent_id: str) -> Optional[Any]:
    """Get A2A-registered agent instance (graph_agent, search_agent) if available."""
    try:
        _backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "openint-backend"))
        if _backend_dir not in sys.path:
            sys.path.insert(0, _backend_dir)
        from a2a import get_agent_instance as _a2a_get
        return _a2a_get(agent_id)
    except ImportError:
        return None


def _invoke_search_agent(query: str, top_k: int = 5) -> Optional[List[Dict[str, Any]]]:
    """Call vectordb-agent (VectorDB) via A2A to get vector search results for context."""
    agent = _get_agent_instance("search_agent")
    if not agent:
        return None
    try:
        resp = agent.process_query(query, {"top_k": top_k})
        if resp.success and resp.results:
            return resp.results
        return None
    except Exception as e:
        logger.debug("search_agent A2A call failed: %s", e)
        return None


def _invoke_graph_agent(query: str) -> Optional[List[Dict[str, Any]]]:
    """Call graph_agent via A2A to get Neo4j query results (joined data)."""
    agent = _get_agent_instance("graph_agent")
    if not agent:
        return None
    try:
        resp = agent.process_query(query)
        if resp.success and resp.results and isinstance(resp.results, list):
            return resp.results
        return None
    except Exception as e:
        logger.debug("graph_agent A2A call failed: %s", e)
        return None


def _augment_with_neo4j_direct(
    client: Any,
    detail: Dict[str, Any],
    user_query: str = "",
) -> Dict[str, Any]:
    """
    Fallback: augment via direct Cypher when graph-agent A2A fails.
    Uses known Cypher templates (no LLM) for reliability.
    """
    if not client or not detail:
        return detail
    label = detail.get("label")
    node_id = str(detail.get("id", "")).strip()
    id_str = _normalize_id_for_lookup(node_id)
    if not id_str or label not in ("Dispute", "Transaction", "Customer"):
        return detail
    props = detail.get("properties") or {}
    merged = dict(props)
    try:
        if label == "Customer":
            cypher = """
            MATCH (c:Customer)-[:HAS_TRANSACTION]->(t:Transaction)
            WHERE toString(c.id) = $idStr
            RETURN t.id AS transaction_id, t.amount AS amount, t.type AS type,
                   t.merchant_name AS merchant_name, t.description AS description, t.currency AS currency
            LIMIT 50
            """
            rows = client.run(cypher, {"idStr": id_str}) or []
            if rows:
                transactions = []
                for r in rows:
                    if not isinstance(r, dict):
                        continue
                    tx_id = str(r.get("transaction_id") or r.get("id") or "")
                    if not tx_id:
                        continue
                    tx = {k: v for k, v in {
                        "transaction_id": tx_id,
                        "amount": r.get("amount"),
                        "type": r.get("type"),
                        "merchant_name": r.get("merchant_name") or r.get("description"),
                        "currency": r.get("currency"),
                    }.items() if v is not None}
                    transactions.append(tx)
                merged["transactions"] = transactions
                merged["transaction_count"] = len(transactions)
            if "dispute" in (user_query or "").lower():
                cypher2 = """
                MATCH (c:Customer)-[:OPENED_DISPUTE]->(d:Dispute)-[:REFERENCES]->(t:Transaction)
                WHERE toString(c.id) = $idStr
                RETURN d.id AS dispute_id, d.amount_disputed AS amount_disputed,
                       d.dispute_status AS dispute_status, d.dispute_reason AS dispute_reason
                LIMIT 20
                """
                rows2 = client.run(cypher2, {"idStr": id_str}) or []
                if rows2:
                    disputes = [
                        {k: v for k, v in {
                            "dispute_id": str(r.get("dispute_id") or ""),
                            "amount_disputed": r.get("amount_disputed"),
                            "dispute_status": r.get("dispute_status"),
                            "dispute_reason": r.get("dispute_reason"),
                        }.items() if v is not None}
                    for r in rows2 if isinstance(r, dict)]
                    merged["disputes"] = disputes
        elif label == "Transaction":
            cypher = """
            MATCH (c:Customer)-[:HAS_TRANSACTION]->(t:Transaction)
            WHERE toString(t.id) = $idStr
            RETURN c.id AS customer_id, c.first_name AS first_name, c.last_name AS last_name,
                   c.email AS email, c.phone AS phone
            LIMIT 1
            """
            rows = client.run(cypher, {"idStr": id_str}) or []
            if rows and isinstance(rows[0], dict):
                r = rows[0]
                for k, v in r.items():
                    if v is not None and k not in merged:
                        merged[k] = v
        elif label == "Dispute":
            cypher = """
            MATCH (d:Dispute)-[:REFERENCES]->(t:Transaction)
            WHERE toString(d.id) = $idStr
            RETURN t.id AS transaction_id, t.amount AS amount, t.type AS type,
                   t.merchant_name AS merchant_name, t.description AS description
            LIMIT 1
            """
            rows = client.run(cypher, {"idStr": id_str}) or []
            if rows and isinstance(rows[0], dict):
                r = rows[0]
                for k, v in r.items():
                    if v is not None and k not in merged:
                        merged[k] = v
        if merged != props:
            detail["properties"] = merged
    except Exception as e:
        logger.debug("Direct Neo4j augmentation failed: %s", e)
    return detail


def _augment_with_a2a(
    detail: Dict[str, Any],
    use_graph: bool = True,
    user_query: str = "",
    neo4j_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Augment enriched detail with joined data from graphdb-agent via A2A.
    When A2A fails or returns empty, falls back to direct Neo4j Cypher.
    - Customer: fetches HAS_TRANSACTION list (transactions) and optionally OPENED_DISPUTE (disputes)
    - Transaction: fetches related Customer
    - Dispute: fetches referenced Transaction
    """
    if not detail or not use_graph:
        return detail
    label = detail.get("label")
    node_id = detail.get("id")
    if not node_id or label not in ("Dispute", "Transaction", "Customer"):
        return detail
    props = detail.get("properties") or {}
    merged = dict(props)
    if label == "Customer":
        # Try graph-agent via A2A first
        q = f"Return the Customer with id {node_id} and all their transactions via HAS_TRANSACTION. Return transaction_id, amount, type, merchant_name, currency for each transaction."
        rows = _invoke_graph_agent(q)
        if rows and isinstance(rows, list):
            transactions: List[Dict[str, Any]] = []
            for r in rows:
                if isinstance(r, dict):
                    meta = r.get("metadata") or r
                    tx_id = meta.get("transaction_id") or meta.get("t.id") or meta.get("id")
                    if tx_id is not None:
                        tx = {
                            "transaction_id": str(tx_id),
                            "amount": meta.get("amount") or meta.get("t.amount"),
                            "type": meta.get("type") or meta.get("transaction_type") or meta.get("t.type"),
                            "merchant_name": meta.get("merchant_name") or meta.get("description"),
                            "currency": meta.get("currency"),
                        }
                        transactions.append({k: v for k, v in tx.items() if v is not None})
            if transactions:
                merged["transactions"] = transactions
                merged["transaction_count"] = len(transactions)
        # Optionally fetch disputes if query suggests interest
        q_lower = (user_query or "").lower()
        if "dispute" in q_lower or "complaint" in q_lower:
            q2 = f"Return the Customer with id {node_id} and all their disputes via OPENED_DISPUTE. Return dispute_id, amount_disputed, dispute_status, dispute_reason."
            rows2 = _invoke_graph_agent(q2)
            if rows2 and isinstance(rows2, list):
                disputes = []
                for r in rows2:
                    if isinstance(r, dict):
                        meta = r.get("metadata") or r
                        d_id = meta.get("dispute_id") or meta.get("d.id") or meta.get("id")
                        if d_id is not None:
                            disputes.append({
                                "dispute_id": str(d_id),
                                "amount_disputed": meta.get("amount_disputed"),
                                "dispute_status": meta.get("dispute_status"),
                                "dispute_reason": meta.get("dispute_reason"),
                            })
                if disputes:
                    merged["disputes"] = disputes
    elif label == "Dispute":
        q = f"Return the Dispute node with id {node_id} and its referenced Transaction with all properties"
        rows = _invoke_graph_agent(q)
        if rows and isinstance(rows, list) and len(rows) > 0 and isinstance(rows[0], dict):
            row = rows[0].get("metadata") or rows[0]
            for k, v in row.items():
                if v is not None and k not in merged:
                    merged[k] = v
    elif label == "Transaction":
        q = f"Return the Transaction node with id {node_id} and the Customer who has it with all properties"
        rows = _invoke_graph_agent(q)
        if rows and isinstance(rows, list) and len(rows) > 0 and isinstance(rows[0], dict):
            row = rows[0].get("metadata") or rows[0]
            for k, v in row.items():
                if v is not None and k not in merged:
                    merged[k] = v
    # Fallback: when A2A returned no useful augmentation, use direct Neo4j Cypher (no LLM)
    if merged == props and neo4j_client:
        detail = _augment_with_neo4j_direct(neo4j_client, detail, user_query)
    elif merged != props:
        detail["properties"] = merged
    return detail


def _build_enrich_summary(enriched: Dict[str, Dict[str, Any]]) -> str:
    """
    Build a concise summary of enriched data for the aggregator LLM.
    Especially lists transactions for customers so the answer can include them.
    """
    if not enriched:
        return ""
    parts: List[str] = []
    for eid, detail in enriched.items():
        label = detail.get("label", "")
        props = detail.get("properties") or {}
        display_name = detail.get("display_name", f"{label} {eid}")
        if label == "Customer":
            tx_list = props.get("transactions") or []
            if tx_list:
                tx_lines = []
                for tx in tx_list[:10]:
                    tx_id = tx.get("transaction_id")
                    amt = tx.get("amount")
                    tx_type = tx.get("type", "")
                    merchant = tx.get("merchant_name", "")
                    line = f"  - transaction {tx_id}: {tx_type}"
                    if amt is not None:
                        try:
                            line += f" ${float(amt):,.2f}"
                        except (TypeError, ValueError):
                            pass
                    if merchant:
                        line += f" @ {merchant[:30]}"
                    tx_lines.append(line.strip())
                parts.append(f"{display_name} has {len(tx_list)} transaction(s):")
                parts.extend(tx_lines)
                if len(tx_list) > 10:
                    parts.append(f"  ... and {len(tx_list) - 10} more")
            disputes = props.get("disputes") or []
            if disputes:
                disp_lines = []
                for d in disputes[:5]:
                    amt = d.get("amount_disputed", 0)
                    amt_str = f"${float(amt):,.2f}" if amt is not None else ""
                    disp_lines.append(f"  - dispute {d.get('dispute_id', '')}: {d.get('dispute_status', '')} {amt_str}".strip())
                parts.append(f"{display_name} has {len(disputes)} dispute(s):")
                parts.extend(disp_lines)
    return "\n".join(parts) if parts else ""


def enrich_ids(
    client: "Neo4jClient",
    ids_with_hints: List[Tuple[str, Optional[str]]],
    max_per_query: int = 20,
    use_a2a: bool = True,
    user_query: str = "",
) -> Dict[str, Dict[str, Any]]:
    """
    Look up each ID in Neo4j. Optionally augment with graphdb-agent via A2A (transactions, disputes).
    Returns map: id -> { label, id, properties, display_name }.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for nid, hint in ids_with_hints[:max_per_query]:
        if nid in out:
            continue
        detail = lookup_id_in_neo4j(client, nid, hint)
        if detail:
            if use_a2a:
                detail = _augment_with_a2a(detail, use_graph=True, user_query=user_query, neo4j_client=client)
            out[nid] = detail
    return out


class EnrichAgent(BaseAgent):
    """
    Agent that extracts IDs (customer, transaction, dispute) from results
    and enriches them with graph DB lookups. Registers capability "enrich".
    """

    def __init__(self, neo4j_client: Optional[Any] = None):
        capabilities = [
            AgentCapability(
                name="enrich",
                description="Enrich results with graph DB lookups for customer_id, transaction_id, dispute_id",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Optional query context"},
                        "results": {"type": "array", "description": "Agent results to enrich"},
                        "text": {"type": "string", "description": "Raw text to extract IDs from"},
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "enriched": {"type": "object", "description": "id -> {label, id, properties}"},
                        "ids_found": {"type": "array", "description": "List of IDs extracted"},
                    },
                },
            )
        ]
        super().__init__(
            name="enrich_agent",
            description="Extracts IDs from results and enriches them with Neo4j graph lookups",
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

    def process_query(self, query: str, context: Dict[str, Any] = None) -> AgentResponse:
        """
        Extract IDs from trusted sources only (graph rows, vector results) and enrich via Neo4j.
        Does NOT extract from query or sentence - those can contain hallucinated/example IDs.
        Context: graph_rows (list of dicts from graph-agent), vector_results (from VectorDB agent),
        results (legacy), agent_responses (legacy). Use trusted_only=True to skip sentence/text.
        """
        context = context or {}
        trusted_only = context.get("trusted_only", True)
        ids_with_hints: List[Tuple[str, Optional[str]]] = []
        ids_seen: Set[str] = set()

        # 1. PRIMARY: Graph rows (explicit customer_id, transaction_id, dispute_id) - source of truth
        graph_rows = context.get("graph_rows") or []
        for nid, hint in _extract_ids_from_graph_rows(graph_rows):
            if nid and nid not in ids_seen:
                ids_seen.add(nid)
                ids_with_hints.append((nid, hint))

        # 2. SECONDARY: Vector results (content from indexed docs - real data)
        vector_results = context.get("vector_results") or []
        for nid, hint in _extract_ids_from_vector_results(vector_results):
            if nid and nid not in ids_seen:
                ids_seen.add(nid)
                ids_with_hints.append((nid, hint))

        # 2b. SUPPLEMENT: User query when they explicitly ask about customer/transaction/dispute
        # Ensures we never miss the ID the user asked about (e.g. "whats up with customer 1000003621")
        user_msg = context.get("query") or context.get("message") or ""
        if user_msg and isinstance(user_msg, str):
            for nid, hint in _extract_ids_from_user_query(user_msg):
                if nid and nid not in ids_seen:
                    ids_seen.add(nid)
                    ids_with_hints.append((nid, hint))

        # 3. Fallback: agent_responses with structured results (when graph_rows/vector_results not passed)
        if not ids_with_hints:
            agent_responses = context.get("agent_responses") or []
            for r in agent_responses:
                content = r.get("content") or {}
                agent_results = content.get("results") or []
                for item in agent_results:
                    if isinstance(item, dict) and item.get("metadata", {}).get("file_type") == "graph_path":
                        row = item.get("metadata") or {}
                        for nid, hint in _extract_ids_from_graph_rows([row]):
                            if nid and nid not in ids_seen:
                                ids_seen.add(nid)
                                ids_with_hints.append((nid, hint))
                for nid, hint in _extract_ids_from_results(agent_results):
                    if nid and nid not in ids_seen:
                        ids_seen.add(nid)
                        ids_with_hints.append((nid, hint))
            results = context.get("results") or []
            for nid, hint in _extract_ids_from_results(results):
                if nid and nid not in ids_seen:
                    ids_seen.add(nid)
                    ids_with_hints.append((nid, hint))

        # 4. Only if NOT trusted_only: allow sentence/query (e.g. for backward compat or explicit override)
        if not trusted_only:
            text = context.get("text") or query or ""
            if text:
                use_a2a = context.get("use_a2a", True)
                a2a_search_count = 0
                max_a2a_search = 3
                for nid, hint in _extract_ids_from_text(text):
                    if nid and nid not in ids_seen:
                        ids_seen.add(nid)
                        # Use search_agent via A2A for context when hint is None (limit calls)
                        if hint is None and use_a2a and a2a_search_count < max_a2a_search:
                            search_res = _invoke_search_agent(f"customer transaction dispute {nid}", top_k=3)
                            a2a_search_count += 1
                            if search_res:
                                ctx_parts = []
                                for r in search_res:
                                    if isinstance(r, dict):
                                        ctx_parts.append(str(r.get("content") or r.get("text") or ""))
                                    elif isinstance(r, str):
                                        ctx_parts.append(r)
                                ctx_str = " ".join(ctx_parts)[:400]
                                if ctx_str:
                                    inferred = _infer_label_from_context(ctx_str)
                                    if inferred:
                                        hint = inferred
                        ids_with_hints.append((nid, hint))

        unique_ids = ids_with_hints

        if not self._client:
            return AgentResponse(
                success=False,
                results=[],
                message="Neo4j client not available",
                metadata={"error": "Neo4j not initialized", "ids_found": [x[0] for x in unique_ids]},
            )

        if not unique_ids:
            return AgentResponse(
                success=True,
                results=[],
                message="No IDs found to enrich",
                metadata={"ids_found": []},
            )

        try:
            self.update_status("BUSY")
            use_a2a = context.get("use_a2a", True)
            user_query = context.get("query") or context.get("message") or query or ""
            enriched = enrich_ids(
                self._client, unique_ids, max_per_query=20, use_a2a=use_a2a, user_query=user_query
            )
            enrich_summary = _build_enrich_summary(enriched)
            self.update_status("IDLE")
            return AgentResponse(
                success=True,
                results=[{
                    "enriched": enriched,
                    "ids_found": [x[0] for x in unique_ids],
                    "enrich_summary": enrich_summary,
                }],
                message=f"Enriched {len(enriched)} of {len(unique_ids)} ID(s)",
                metadata={
                    "enriched_count": len(enriched),
                    "ids_found": [x[0] for x in unique_ids],
                    "enrich_summary": enrich_summary,
                },
            )
        except Exception as e:
            self.update_status("ERROR")
            logger.warning("EnrichAgent error: %s", e, exc_info=True)
            return AgentResponse(
                success=False,
                results=[],
                message=f"Enrichment failed: {e}",
                metadata={"error": str(e), "ids_found": [x[0] for x in unique_ids]},
            )

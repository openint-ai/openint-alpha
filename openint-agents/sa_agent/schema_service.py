"""
Schema service for sa-agent (standalone / programmatic use).
Gets schema from DataHub or openint-datahub (with optional Redis cache),
generates example sentences via Ollama, and filters by hint.
"""

import logging
from typing import Dict, List, Any, Optional

from sa_agent.datahub_client import get_schema_and_source
from sa_agent.sentence_generator import generate_sentences

# Use shared agent state store (Redis) for schema cache
from communication.agent_state_store import (
    get_state as get_agent_state,
    set_state as set_agent_state,
    SCHEMA_CACHE_TTL,
    is_available as state_store_available,
)

logger = logging.getLogger(__name__)
AGENT_NAME = "sa-agent"

_schema_cache: Optional[Dict[str, Dict[str, Any]]] = None


def get_schema() -> Dict[str, Dict[str, Any]]:
    """Get schema: from Redis if available, else from DataHub/openint-datahub; in-memory fallback when Redis down."""
    if state_store_available():
        cached = get_agent_state(AGENT_NAME, "schema")
        if cached is not None and isinstance(cached, dict):
            logger.debug("sa-agent: schema loaded from Redis (%s datasets)", len(cached))
            return cached
    global _schema_cache
    if _schema_cache is not None:
        return _schema_cache
    schema, _ = get_schema_and_source()
    if not schema:
        return {}
    if state_store_available():
        set_agent_state(AGENT_NAME, "schema", schema, ttl_seconds=SCHEMA_CACHE_TTL)
    _schema_cache = schema
    return schema


def generate_suggestions(
    query_hint: str = "",
    count: int = 25,
    prefer_llm: bool = True,
) -> tuple[List[Dict[str, Any]], str, Optional[str]]:
    """
    Generate example suggestions. Returns (results, message, error).
    results: list of {"query", "category", "source"}.
    query_hint: optional filter - "analyst", "customer"/"care", "business".
    """
    schema = get_schema()
    if not schema:
        return (
            [],
            "No schema available. Ensure DataHub is running or openint-datahub/schemas.py is present.",
            "no_schema",
        )
    _, schema_source = get_schema_and_source()
    sentences, gen_error = generate_sentences(schema, count=count, prefer_llm=prefer_llm, schema_source=schema_source)
    query_lower = (query_hint or "").strip().lower()
    if query_lower:
        if "analyst" in query_lower and "customer" not in query_lower and "business" not in query_lower:
            sentences = [s for s in sentences if s.get("category") == "Analyst"]
        elif "customer" in query_lower or "care" in query_lower:
            sentences = [s for s in sentences if s.get("category") == "Customer Care"]
        elif "business" in query_lower:
            sentences = [s for s in sentences if s.get("category") == "Business Analyst"]
    results = [
        {"query": s["sentence"], "category": s["category"], "source": s.get("source", "template")}
        for s in sentences
    ]
    message = f"Generated {len(results)} example questions from schema ({len(schema)} datasets)."
    return (results, message, gen_error if not results else None)

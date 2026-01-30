"""
Schema service for openint-sg-agent (standalone).
Gets schema from DataHub or openint-datahub (with optional Redis cache),
generates example sentences via Ollama, and filters by hint.
"""

import logging
from typing import Dict, List, Any, Optional

from datahub_client import get_schema_and_source
from sentence_generator import generate_sentences
from state_store import get_schema_cache, set_schema_cache, is_available as state_store_available

logger = logging.getLogger(__name__)


_schema_cache: Optional[Dict[str, Dict[str, Any]]] = None


def get_schema() -> Dict[str, Dict[str, Any]]:
    """Get schema: from Redis if available, else from DataHub/openint-datahub; in-memory fallback when Redis down."""
    if state_store_available():
        cached = get_schema_cache()
        if cached is not None and isinstance(cached, dict):
            logger.debug("openint-sg-agent: schema loaded from Redis (%s datasets)", len(cached))
            return cached
    global _schema_cache
    if _schema_cache is not None:
        return _schema_cache
    schema, _ = get_schema_and_source()
    if not schema:
        return {}
    if state_store_available():
        set_schema_cache(schema)
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

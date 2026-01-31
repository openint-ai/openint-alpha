"""
Sentence Generation Agent (sa-agent).
Connects to DataHub (or uses local schema), reads dataset schemas, and uses the best
available generative model to produce example sentences that analysts, customer care,
and business users would ask in a banking context.
"""

import sys
import os
import logging
from typing import Dict, List, Any, Optional

# Ensure openint-agents root is on path for communication and agents
_AGENTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENTS_ROOT not in sys.path:
    sys.path.insert(0, _AGENTS_ROOT)

from agents.base_agent import BaseAgent, AgentResponse
from communication.agent_registry import AgentCapability

from communication.agent_state_store import (
    get_state as get_agent_state,
    set_state as set_agent_state,
    SCHEMA_CACHE_TTL,
    is_available as state_store_available,
)
from sa_agent.datahub_client import get_schema_and_source
from sa_agent.sentence_generator import generate_sentences

logger = logging.getLogger(__name__)
AGENT_NAME = "sa-agent"


class SchemaGeneratorAgent(BaseAgent):
    """
    Agent that reads dataset schema (from DataHub or openint-datahub/schemas.py)
    and generates example sentences that a typical analyst, customer care support,
    or business analyst in a bank would ask.
    """

    def __init__(self):
        capabilities = [
            AgentCapability(
                name="suggestions",
                description="Generate example questions analysts and customer care would ask",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Optional hint (e.g. 'analyst', 'customer care')"},
                        "count": {"type": "integer", "default": 25},
                    },
                },
                output_schema={
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sentence": {"type": "string"},
                            "category": {"type": "string"},
                            "source": {"type": "string"},
                        },
                    },
                },
            )
        ]
        super().__init__(
            name="sa-agent",
            description="Sentence generation: connects to DataHub, reads schema, and generates example analyst/customer-care/business questions",
            capabilities=capabilities,
        )
        self._schema_cache: Optional[Dict[str, Dict[str, Any]]] = None

    def _get_schema(self) -> Dict[str, Dict[str, Any]]:
        """Get schema: from Redis if available, else from DataHub/openint-datahub; in-memory fallback when Redis down."""
        # Try Redis-backed cache first (survives restarts)
        if state_store_available():
            cached = get_agent_state(AGENT_NAME, "schema")
            if cached is not None and isinstance(cached, dict):
                logger.debug("sa-agent: schema loaded from Redis (%s datasets)", len(cached))
                return cached
        # In-memory fallback
        if self._schema_cache is not None:
            return self._schema_cache
        schema, _ = get_schema_and_source()
        if not schema:
            return {}
        if state_store_available():
            set_agent_state(AGENT_NAME, "schema", schema, ttl_seconds=SCHEMA_CACHE_TTL)
        self._schema_cache = schema
        return schema

    def process_query(self, query: str, context: Dict[str, Any] = None) -> AgentResponse:
        """
        Process a request for example sentences.
        Query can be empty (return all) or hint at role (analyst, customer care, business analyst).
        """
        context = context or {}
        try:
            self.update_status("BUSY")
            schema = self._get_schema()
            if not schema:
                return AgentResponse(
                    success=False,
                    results=[],
                    message="No schema available. Ensure DataHub is running or openint-datahub/schemas.py is present.",
                    metadata={"file_type": "suggestions"},
                )
            count = int(context.get("count", 25))
            prefer_llm = context.get("prefer_llm", True)
            sentences, _error = generate_sentences(schema, count=count, prefer_llm=prefer_llm)
            # Optional filter by query hint
            query_lower = (query or "").strip().lower()
            if query_lower:
                if "analyst" in query_lower and "customer" not in query_lower and "business" not in query_lower:
                    sentences = [s for s in sentences if s.get("category") == "Analyst"]
                elif "customer" in query_lower or "care" in query_lower:
                    sentences = [s for s in sentences if s.get("category") == "Customer Care"]
                elif "business" in query_lower:
                    sentences = [s for s in sentences if s.get("category") == "Business Analyst"]
            # Format for UI: list of { query, category } for suggestions panel
            results = [
                {"query": s["sentence"], "category": s["category"], "source": s.get("source", "template")}
                for s in sentences
            ]
            return AgentResponse(
                success=True,
                results=results,
                message=f"Generated {len(results)} example questions from schema ({len(schema)} datasets).",
                metadata={
                    "file_type": "suggestions",
                    "datasets": len(schema),
                    "count": len(results),
                },
            )
        except Exception as e:
            logger.warning("Schema generator agent error: %s", e, exc_info=True)
            return AgentResponse(
                success=False,
                results=[],
                message=f"Failed to generate sentences: {e}",
                metadata={"file_type": "suggestions", "error": str(e)},
            )
        finally:
            self.update_status("IDLE")

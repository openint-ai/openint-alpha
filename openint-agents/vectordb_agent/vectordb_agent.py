"""
VectorDB Agent
Performs semantic search in Milvus vector database.
"""

import sys
import os
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# Resolve repo root (openint-agents/vectordb_agent -> ../ = openint-agents -> ../ = repo root)
_agent_file = os.path.abspath(__file__)
_repo_root = os.path.abspath(os.path.join(os.path.dirname(_agent_file), '..', '..'))
_vectordb_milvus = os.path.join(_repo_root, 'openint-vectordb', 'milvus')
if os.path.isdir(_vectordb_milvus) and _vectordb_milvus not in sys.path:
    sys.path.insert(0, _vectordb_milvus)

try:
    from milvus_client import MilvusClient
except ImportError:
    MilvusClient = None
    logger.warning("milvus_client not found; VectorDB agent will not work")

# Add parent directory (openint-agents) to path for agents.base_agent and communication
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent, AgentResponse
from communication.agent_registry import AgentCapability


class VectorDBAgent(BaseAgent):
    """
    Agent for semantic search in Milvus vector database.
    """

    def __init__(self, milvus_client: Any = None):
        capabilities = [
            AgentCapability(
                name="search",
                description="Semantic search in openint data",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {"type": "integer", "default": 10}
                    }
                },
                output_schema={
                    "type": "array",
                    "items": {"type": "object"}
                }
            )
        ]
        super().__init__(
            name="vectordb-agent",
            description="Performs semantic search in openint data using VectorDB (Milvus)",
            capabilities=capabilities
        )
        if milvus_client:
            self.milvus_client = milvus_client
        elif MilvusClient:
            try:
                self.milvus_client = MilvusClient(embedding_model="all-MiniLM-L6-v2")
            except Exception as e:
                logger.warning("Could not initialize Milvus client: %s", e)
                self.milvus_client = None
        else:
            self.milvus_client = None

    def process_query(self, query: str, context: Dict[str, Any] = None) -> AgentResponse:
        """
        Process a search query.
        """
        if not MilvusClient:
            return AgentResponse(
                success=False,
                results=[],
                message="Milvus client not available",
                metadata={"error": "Milvus not initialized"}
            )
        try:
            self.update_status("BUSY")
            top_k = context.get("top_k", 10) if context else 10
            embedding_model = context.get("embedding_model") or context.get("best_model")
            client_to_use = self.milvus_client
            if embedding_model and MilvusClient:
                default_model = getattr(self.milvus_client, "embedding_model_name", None) if self.milvus_client else None
                if default_model != embedding_model:
                    try:
                        client_to_use = MilvusClient(embedding_model=embedding_model)
                    except Exception as e:
                        logger.warning("Could not create client with model %s, using default: %s", embedding_model, e)
                        client_to_use = self.milvus_client
            if not client_to_use:
                return AgentResponse(
                    success=False,
                    results=[],
                    message="Milvus client not available",
                    metadata={"error": "Milvus not initialized"}
                )
            results, total_time_ms, embedding_time_ms, vector_search_time_ms = client_to_use.search(query, top_k=top_k)
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.get("id"),
                    "content": result.get("content", "")[:500],
                    "score": result.get("score"),
                    "metadata": result.get("metadata", {})
                })
            self.update_status("IDLE")
            metadata = {
                "query": query,
                "top_k": top_k,
                "results_count": len(formatted_results),
                "vector_db_query_time_ms": vector_search_time_ms,
                "embedding_time_ms": embedding_time_ms,
                "total_search_time_ms": total_time_ms,
            }
            if embedding_model:
                metadata["embedding_model"] = embedding_model
                if client_to_use != self.milvus_client:
                    metadata["model_used"] = "custom"
            return AgentResponse(
                success=True,
                results=formatted_results,
                message=f"Found {len(formatted_results)} results",
                metadata=metadata
            )
        except Exception as e:
            self.update_status("ERROR")
            return AgentResponse(
                success=False,
                results=[],
                message=f"Search error: {str(e)}",
                metadata={"error": str(e)}
            )

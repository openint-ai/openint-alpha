"""
Search Agent
Performs semantic search in Milvus vector database
"""

import sys
import os
from typing import Dict, List, Any

# Add parent directory to path to import milvus_client
parent_dir = os.path.join(os.path.dirname(__file__), '../../')
sys.path.insert(0, parent_dir)

try:
    # Import from openint-vectordb package (handle hyphenated directory name)
    vectordb_path = os.path.join(parent_dir, 'openint-vectordb')
    if vectordb_path not in sys.path:
        sys.path.insert(0, vectordb_path)
    # Import using the milvus subdirectory
    sys.path.insert(0, os.path.join(vectordb_path, 'milvus'))
    from milvus_client import MilvusClient
except ImportError:
    # Try alternative path
    try:
        root_dir = os.path.join(os.path.dirname(__file__), '../../../')
        vectordb_path = os.path.join(root_dir, 'openint-vectordb', 'milvus')
        sys.path.insert(0, vectordb_path)
        from milvus_client import MilvusClient
    except ImportError:
        # Fallback if milvus_client is not available
        MilvusClient = None
        logger.warning("milvus_client not found; search agent will not work")

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from agents.base_agent import BaseAgent, AgentResponse
from communication.agent_registry import AgentCapability

logger = logging.getLogger(__name__)


class SearchAgent(BaseAgent):
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
            name="search_agent",
            description="Performs semantic search in openint data using Milvus",
            capabilities=capabilities
        )
        
        # Initialize Milvus client
        if milvus_client:
            self.milvus_client = milvus_client
        elif MilvusClient:
            try:
                self.milvus_client = MilvusClient()
            except Exception as e:
                logger.warning("Could not initialize Milvus client: %s", e)
                self.milvus_client = None
        else:
            self.milvus_client = None
    
    def process_query(self, query: str, context: Dict[str, Any] = None) -> AgentResponse:
        """
        Process a search query.
        
        Args:
            query: Search query text
            context: Optional context (may contain top_k, filters, embedding_model, etc.)
            
        Returns:
            AgentResponse with search results
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
            
            # Get parameters from context
            top_k = context.get("top_k", 10) if context else 10
            embedding_model = context.get("embedding_model") or context.get("best_model")  # Use best model if provided
            
            # Reuse default client when requested model matches (avoids loading model on every request)
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
            
            # Perform search (returns results and timing: total, embedding, vector search)
            results, total_time_ms, embedding_time_ms, vector_search_time_ms = client_to_use.search(query, top_k=top_k)

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.get("id"),
                    "content": result.get("content", "")[:500],  # Truncate for display
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

            # Add model info if custom model was used
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

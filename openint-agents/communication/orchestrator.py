"""
Agent Orchestrator
Coordinates multi-agent workflows and manages agent collaboration
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
import threading

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from communication.message_bus import MessageBus, MessageType, AgentMessage, get_message_bus
from communication.agent_registry import AgentRegistry, AgentCapability, get_registry

logger = logging.getLogger(__name__)


@dataclass
class QueryContext:
    """Context for a user query"""
    query_id: str
    user_query: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AgentOrchestrator:
    """
    Orchestrator for coordinating multiple agents.
    Routes queries to appropriate agents and aggregates responses.
    """
    
    def __init__(self, message_bus: Optional[MessageBus] = None, registry: Optional[AgentRegistry] = None):
        self.message_bus = message_bus or get_message_bus()
        self.registry = registry or get_registry()
        self._active_queries: Dict[str, QueryContext] = {}
        self._query_responses: Dict[str, List[Dict[str, Any]]] = {}
        self._response_lock = threading.Lock()  # thread-safe when agents run in parallel
        
        # Subscribe to agent responses
        self.message_bus.subscribe("response", self._handle_agent_response)
    
    def process_query(
        self,
        user_query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user query using multiple agents.
        
        Args:
            user_query: User's query text
            session_id: Optional session ID
            user_id: Optional user ID
            metadata: Optional metadata
            
        Returns:
            Aggregated response from agents
        """
        import uuid
        query_id = str(uuid.uuid4())
        
        context = QueryContext(
            query_id=query_id,
            user_query=user_query,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {}
        )
        
        self._active_queries[query_id] = context
        self._query_responses[query_id] = []
        
        # Determine which agents to involve (metadata can request semantic_annotate → modelmgmt-agent only)
        agents_to_query = self._select_agents(user_query, context.metadata)
        
        # Send query to all selected agents in parallel (avoids 2x latency when search + graph)
        logger.debug("Sending query to %s agent(s): %s", len(agents_to_query), [a.name for a in agents_to_query])
        def send_to_agent(agent_info):
            self.message_bus.send_message(
                from_agent="orchestrator",
                to_agent=agent_info.name,
                content={
                    "query": user_query,
                    "query_id": query_id,
                    "context": context.metadata
                },
                message_type=MessageType.QUERY,
                correlation_id=query_id
            )
            logger.debug("Sent message to %s", agent_info.name)
        threads = [threading.Thread(target=send_to_agent, args=(a,)) for a in agents_to_query]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        return {
            "query_id": query_id,
            "status": "processing",
            "agents_queried": [a.name for a in agents_to_query],
            "message": "Query submitted to agents"
        }
    
    def _select_agents(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> List:
        """
        Select appropriate agents for a query.
        
        Args:
            query: User query
            metadata: Optional metadata (e.g. intent=semantic_annotate for modelmgmt-agent only)
            
        Returns:
            List of agent info objects
        """
        metadata = metadata or {}
        query_lower = query.lower()
        selected_agents = []

        # Backend semantic API: route to modelmgmt-agent only for sentence annotation
        if metadata.get("intent") == "semantic_annotate":
            agents = self.registry.find_agents_by_capability("semantic_annotate")
            selected_agents.extend(agents)
            if selected_agents:
                return selected_agents

        # Keyword-based agent selection
        if any(word in query_lower for word in ["search", "find", "look", "show"]):
            agents = self.registry.find_agents_by_capability("search")
            selected_agents.extend(agents)
        
        if any(word in query_lower for word in ["analyze", "analysis", "insight", "trend"]):
            agents = self.registry.find_agents_by_capability("analysis")
            selected_agents.extend(agents)
        
        if any(word in query_lower for word in ["recommend", "suggest", "similar"]):
            agents = self.registry.find_agents_by_capability("recommendation")
            selected_agents.extend(agents)
        
        if any(word in query_lower for word in ["transaction", "payment", "transfer"]):
            agents = self.registry.find_agents_by_capability("transaction")
            selected_agents.extend(agents)
        
        if any(word in query_lower for word in ["customer", "client", "account"]):
            agents = self.registry.find_agents_by_capability("customer")
            selected_agents.extend(agents)
        
        if any(word in query_lower for word in ["related", "connected", "path", "link", "dispute", "account", "relationship", "graph"]):
            agents = self.registry.find_agents_by_capability("graph")
            selected_agents.extend(agents)
        
        if any(word in query_lower for word in ["suggest", "example question", "what can i ask", "generate sentence", "sample query", "example query", "analyst question", "business analyst"]):
            agents = self.registry.find_agents_by_capability("suggestions")
            selected_agents.extend(agents)
        
        # If no specific agents found, use search agent as default
        if not selected_agents:
            agents = self.registry.find_agents_by_capability("search")
            if agents:
                selected_agents.extend(agents)
        
        # Remove duplicates
        seen = set()
        unique_agents = []
        for agent in selected_agents:
            if agent.name not in seen:
                seen.add(agent.name)
                unique_agents.append(agent)
        
        return unique_agents
    
    def _handle_agent_response(self, message: AgentMessage):
        """Handle response from an agent (thread-safe when agents run in parallel)."""
        logger.debug("Orchestrator received response from %s (correlation_id: %s)", message.from_agent, message.correlation_id)
        if message.correlation_id and message.correlation_id in self._active_queries:
            query_id = message.correlation_id
            with self._response_lock:
                if query_id not in self._query_responses:
                    self._query_responses[query_id] = []
                self._query_responses[query_id].append({
                    "agent": message.from_agent,
                    "content": message.content,
                    "timestamp": message.timestamp.isoformat()
                })
                count = len(self._query_responses[query_id])
            logger.debug("Added response to query %s (total responses: %s)", query_id, count)
        else:
            logger.warning("No active query found for correlation_id: %s", message.correlation_id)
    
    def get_query_result(self, query_id: str) -> Optional[Dict[str, Any]]:
        """
        Get aggregated result for a query.
        
        Args:
            query_id: Query ID
            
        Returns:
            Aggregated response or None if not found
        """
        if query_id not in self._active_queries:
            return None
        
        context = self._active_queries[query_id]
        responses = self._query_responses.get(query_id, [])
        
        # Aggregate responses
        aggregated = self._aggregate_responses(context, responses)
        
        return aggregated
    
    def _aggregate_responses(
        self,
        context: QueryContext,
        responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate responses from multiple agents.
        
        Args:
            context: Query context
            responses: List of agent responses
            
        Returns:
            Aggregated response
        """
        if not responses:
            return {
                "query_id": context.query_id,
                "status": "no_responses",
                "message": "No agents responded",
                "results": []
            }
        
        # Simple aggregation - combine all results
        all_results = []
        for response in responses:
            agent_results = response.get("content", {}).get("results", [])
            if isinstance(agent_results, list):
                all_results.extend(agent_results)
            else:
                all_results.append(agent_results)
        
        return {
            "query_id": context.query_id,
            "status": "completed",
            "query": context.user_query,
            "agents_responded": len(responses),
            "results": all_results,
            "agent_responses": responses
        }

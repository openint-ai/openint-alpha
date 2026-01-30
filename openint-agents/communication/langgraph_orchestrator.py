"""
LangGraph-based Agent Orchestrator
Manages communication flow between agents via a StateGraph (select_agents -> run_agents -> aggregate).
"""

from __future__ import annotations

import logging
import sys
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from communication.agent_registry import AgentRegistry, get_registry

logger = logging.getLogger(__name__)

# LangGraph import (optional at module load for graceful fallback)
try:
    from langgraph.graph import START, END, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None  # type: ignore
    START = END = None  # type: ignore


class OrchestratorState(TypedDict, total=False):
    """State schema for the LangGraph orchestration flow."""
    query_id: str
    user_query: str
    session_id: str
    user_id: str
    metadata: Dict[str, Any]
    selected_agent_names: List[str]
    agent_responses: List[Dict[str, Any]]
    status: str
    query: str
    agents_responded: int
    results: List[Any]
    message: str


def _select_agent_names(
    query: str,
    metadata: Optional[Dict[str, Any]],
    registry: AgentRegistry,
) -> List[str]:
    """
    Port of orchestrator _select_agents: returns list of agent names.
    """
    metadata = metadata or {}
    query_lower = query.lower()
    selected_agents = []

    if metadata.get("intent") == "semantic_annotate":
        agents = registry.find_agents_by_capability("semantic_annotate")
        selected_agents.extend(agents)
        if selected_agents:
            return [a.name for a in selected_agents]

    if any(word in query_lower for word in ["search", "find", "look", "show"]):
        agents = registry.find_agents_by_capability("search")
        selected_agents.extend(agents)
    if any(word in query_lower for word in ["analyze", "analysis", "insight", "trend"]):
        agents = registry.find_agents_by_capability("analysis")
        selected_agents.extend(agents)
    if any(word in query_lower for word in ["recommend", "suggest", "similar"]):
        agents = registry.find_agents_by_capability("recommendation")
        selected_agents.extend(agents)
    if any(word in query_lower for word in ["transaction", "payment", "transfer"]):
        agents = registry.find_agents_by_capability("transaction")
        selected_agents.extend(agents)
    if any(word in query_lower for word in ["customer", "client", "account"]):
        agents = registry.find_agents_by_capability("customer")
        selected_agents.extend(agents)
    if any(word in query_lower for word in ["related", "connected", "path", "link", "dispute", "account", "relationship", "graph"]):
        agents = registry.find_agents_by_capability("graph")
        selected_agents.extend(agents)
    if any(word in query_lower for word in ["suggest", "example question", "what can i ask", "generate sentence", "sample query", "example query", "analyst question", "business analyst"]):
        agents = registry.find_agents_by_capability("suggestions")
        selected_agents.extend(agents)

    if not selected_agents:
        agents = registry.find_agents_by_capability("search")
        if agents:
            selected_agents.extend(agents)

    seen = set()
    unique = []
    for agent in selected_agents:
        if agent.name not in seen:
            seen.add(agent.name)
            unique.append(agent)
    return [a.name for a in unique]


def build_orchestrator_graph(
    registry: AgentRegistry,
    agent_instances: Dict[str, Any],
) -> Any:
    """
    Build and compile the LangGraph: select_agents -> run_agents -> aggregate.
    agent_instances: map agent name -> BaseAgent instance (must have process_query(query, context)).
    """
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError("langgraph is not installed; add langgraph to requirements and reinstall.")

    def select_agents_node(state: OrchestratorState) -> Dict[str, Any]:
        names = _select_agent_names(
            state["user_query"],
            state.get("metadata"),
            registry,
        )
        logger.debug("LangGraph select_agents: %s", names)
        return {"selected_agent_names": names}

    def run_agents_node(state: OrchestratorState) -> Dict[str, Any]:
        names = state.get("selected_agent_names") or []
        user_query = state["user_query"]
        context = state.get("metadata") or {}
        responses: List[Dict[str, Any]] = []

        def run_one(name: str) -> Optional[Dict[str, Any]]:
            agent = agent_instances.get(name)
            if not agent:
                logger.warning("No instance for agent %s", name)
                return None
            try:
                resp = agent.process_query(user_query, context)
                return {
                    "agent": name,
                    "content": {
                        "results": resp.results,
                        "message": resp.message,
                        "metadata": resp.metadata,
                    },
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                logger.warning("Agent %s error: %s", name, e, exc_info=True)
                return {
                    "agent": name,
                    "content": {"error": str(e), "message": f"Error: {e}"},
                    "timestamp": datetime.now().isoformat(),
                }

        if not names:
            return {"agent_responses": responses}

        with ThreadPoolExecutor(max_workers=len(names)) as executor:
            futures = {executor.submit(run_one, n): n for n in names}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    responses.append(result)

        logger.debug("LangGraph run_agents: %s responses", len(responses))
        return {"agent_responses": responses}

    def aggregate_node(state: OrchestratorState) -> Dict[str, Any]:
        responses = state.get("agent_responses") or []
        query_id = state["query_id"]
        user_query = state["user_query"]

        if not responses:
            return {
                "status": "no_responses",
                "message": "No agents responded",
                "results": [],
                "query": user_query,
                "agents_responded": 0,
            }

        all_results: List[Any] = []
        for r in responses:
            content = r.get("content") or {}
            agent_results = content.get("results", [])
            if isinstance(agent_results, list):
                all_results.extend(agent_results)
            else:
                all_results.append(agent_results)

        return {
            "status": "completed",
            "query": user_query,
            "agents_responded": len(responses),
            "results": all_results,
            "agent_responses": responses,
        }

    graph = StateGraph(OrchestratorState)
    graph.add_node("select_agents", select_agents_node)
    graph.add_node("run_agents", run_agents_node)
    graph.add_node("aggregate", aggregate_node)

    graph.add_edge(START, "select_agents")
    graph.add_edge("select_agents", "run_agents")
    graph.add_edge("run_agents", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()


class LangGraphOrchestrator:
    """
    Orchestrator that runs the LangGraph once per query and returns the aggregated result.
    Exposes run() for single-call execution; no polling.
    """

    def __init__(
        self,
        registry: Optional[AgentRegistry] = None,
        agent_instances: Optional[Dict[str, Any]] = None,
    ):
        self.registry = registry or get_registry()
        self._agent_instances = agent_instances or {}
        self._graph = None
        if self._agent_instances:
            self._graph = build_orchestrator_graph(self.registry, self._agent_instances)

    def run(
        self,
        user_query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run the graph synchronously and return the aggregated result
        (query_id, status, query, agents_responded, results, agent_responses).
        """
        if not self._graph:
            return {
                "query_id": str(uuid.uuid4()),
                "status": "no_responses",
                "message": "No agent instances configured",
                "results": [],
                "agents_responded": 0,
                "agent_responses": [],
            }
        query_id = str(uuid.uuid4())
        initial: OrchestratorState = {
            "query_id": query_id,
            "user_query": user_query,
            "session_id": session_id or "",
            "user_id": user_id or "",
            "metadata": metadata or {},
        }
        final = self._graph.invoke(initial)
        # Return the same shape as get_query_result for backend compatibility
        return {
            "query_id": final.get("query_id", query_id),
            "status": final.get("status", "no_responses"),
            "query": final.get("query", user_query),
            "agents_responded": final.get("agents_responded", 0),
            "results": final.get("results", []),
            "agent_responses": final.get("agent_responses", []),
            "message": final.get("message", ""),
        }

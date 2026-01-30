"""
Main entry point for OpenInt Agents System
"""

import logging
import os
import sys
from typing import Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Turn off debug logging project-wide (cap at INFO, default WARNING)
_log_level = (os.environ.get("LOG_LEVEL") or "WARNING").strip().upper()
if _log_level == "DEBUG":
    _log_level = "INFO"
logging.basicConfig(
    level=getattr(logging, _log_level, logging.WARNING),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

from communication.orchestrator import AgentOrchestrator
from communication.agent_registry import AgentCapability
from agents.search_agent import SearchAgent


def initialize_agents(orchestrator: Optional[AgentOrchestrator] = None):
    """Initialize and register all agents. Returns list of agent instances for LangGraph."""
    agents_list = []

    # Initialize Search Agent
    search_agent = SearchAgent()
    agents_list.append(search_agent)
    print(f"✅ Initialized {search_agent.name}")

    # Initialize Graph Agent
    try:
        from agents.graph_agent import GraphAgent
        graph_agent = GraphAgent()
        agents_list.append(graph_agent)
        print(f"✅ Initialized {graph_agent.name}")
    except Exception as e:
        print(f"⚠️  Graph agent not loaded: {e}")

    # Initialize Schema Generator Agent (sg-agent)
    try:
        from sg_agent.schema_generator_agent import SchemaGeneratorAgent
        sg_agent = SchemaGeneratorAgent()
        agents_list.append(sg_agent)
        print(f"✅ Initialized {sg_agent.name}")
    except Exception as e:
        print(f"⚠️  Schema generator agent not loaded: {e}")

    # Initialize Model Management Agent (modelmgmt-agent): Hugging Face + Redis, sentence annotation
    try:
        from modelmgmt_agent.modelmgmt_agent import ModelMgmtAgent
        modelmgmt_agent = ModelMgmtAgent()
        agents_list.append(modelmgmt_agent)
        print(f"✅ Initialized {modelmgmt_agent.name}")
    except Exception as e:
        print(f"⚠️  Model management agent not loaded: {e}")

    return agents_list


def main():
    """Main function"""
    print("=" * 80)
    print("🏦 OpenInt Agents System")
    print("=" * 80)

    # Initialize agents first (they register with the global registry)
    agents = initialize_agents()
    print(f"✅ Initialized {len(agents)} agent(s)")

    # Build orchestrator with agent instances for LangGraph (select_agents -> run_agents -> aggregate)
    agent_instances = {a.name: a for a in agents}
    orchestrator = AgentOrchestrator(agent_instances=agent_instances)
    print("✅ Orchestrator initialized (LangGraph when agent_instances provided)")

    # List registered agents
    registry = orchestrator.registry
    print("\n📋 Registered Agents:")
    for agent_info in registry.list_agents():
        print(f"   • {agent_info.name}: {agent_info.description}")
        print(f"     Capabilities: {[c.name for c in agent_info.capabilities]}")
    
    print("\n" + "=" * 80)
    print("✅ OpenInt Agents System Ready")
    print("=" * 80)
    print("\n💡 Usage:")
    print("   from openint_agents.main import orchestrator")
    print("   response = orchestrator.process_query('your query here')")
    print("=" * 80)
    
    return orchestrator


if __name__ == "__main__":
    orchestrator = main()
    
    # Keep running (in production, use proper server like FastAPI/Flask)
    print("\n💡 Agent system is running. Press Ctrl+C to stop.")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down agent system...")
        sys.exit(0)

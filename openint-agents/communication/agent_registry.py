"""
Agent Registry for Service Discovery
Manages available agents and their capabilities
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class AgentStatus(Enum):
    """Agent status"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class AgentCapability:
    """Agent capability description"""
    name: str
    description: str
    input_schema: Optional[Dict] = None
    output_schema: Optional[Dict] = None


@dataclass
class AgentInfo:
    """Information about an agent"""
    name: str
    description: str
    capabilities: List[AgentCapability] = field(default_factory=list)
    status: AgentStatus = AgentStatus.IDLE
    last_seen: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


class AgentRegistry:
    """
    Registry for managing agents and their capabilities.
    Provides service discovery for the agent system.
    """
    
    def __init__(self):
        self._agents: Dict[str, AgentInfo] = {}
        self._capability_index: Dict[str, Set[str]] = {}  # capability -> set of agent names
    
    def register_agent(
        self,
        name: str,
        description: str,
        capabilities: List[AgentCapability],
        metadata: Optional[Dict] = None
    ):
        """
        Register an agent.
        
        Args:
            name: Agent name (must be unique)
            description: Agent description
            capabilities: List of agent capabilities
            metadata: Optional metadata
        """
        if name in self._agents:
            raise ValueError(f"Agent {name} is already registered")
        
        agent_info = AgentInfo(
            name=name,
            description=description,
            capabilities=capabilities,
            metadata=metadata or {}
        )
        
        self._agents[name] = agent_info
        
        # Update capability index
        for capability in capabilities:
            if capability.name not in self._capability_index:
                self._capability_index[capability.name] = set()
            self._capability_index[capability.name].add(name)
    
    def unregister_agent(self, name: str):
        """Unregister an agent"""
        if name not in self._agents:
            return
        
        # Remove from capability index
        agent_info = self._agents[name]
        for capability in agent_info.capabilities:
            if capability.name in self._capability_index:
                self._capability_index[capability.name].discard(name)
                if not self._capability_index[capability.name]:
                    del self._capability_index[capability.name]
        
        del self._agents[name]
    
    def update_agent_status(self, name: str, status: AgentStatus):
        """Update agent status"""
        if name in self._agents:
            self._agents[name].status = status
            self._agents[name].last_seen = datetime.now()
    
    def get_agent(self, name: str) -> Optional[AgentInfo]:
        """Get agent information"""
        return self._agents.get(name)
    
    def list_agents(self, status: Optional[AgentStatus] = None) -> List[AgentInfo]:
        """List all agents, optionally filtered by status"""
        agents = list(self._agents.values())
        if status:
            agents = [a for a in agents if a.status == status]
        return agents
    
    def find_agents_by_capability(self, capability_name: str) -> List[AgentInfo]:
        """
        Find agents that have a specific capability.
        
        Args:
            capability_name: Name of the capability
            
        Returns:
            List of agents with the capability
        """
        agent_names = self._capability_index.get(capability_name, set())
        return [self._agents[name] for name in agent_names if name in self._agents]
    
    def find_agents_by_keyword(self, keyword: str) -> List[AgentInfo]:
        """
        Find agents by keyword search in name or description.
        
        Args:
            keyword: Search keyword
            
        Returns:
            List of matching agents
        """
        keyword_lower = keyword.lower()
        matches = []
        for agent in self._agents.values():
            if (keyword_lower in agent.name.lower() or 
                keyword_lower in agent.description.lower()):
                matches.append(agent)
        return matches


# Global registry instance
_registry: Optional[AgentRegistry] = None


def get_registry() -> AgentRegistry:
    """Get or create global agent registry"""
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry

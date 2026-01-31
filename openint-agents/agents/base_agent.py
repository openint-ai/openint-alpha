"""
Base Agent Class
All agents inherit from this base class
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from communication.message_bus import MessageBus, MessageType, AgentMessage, get_message_bus
from communication.agent_registry import AgentRegistry, AgentCapability, get_registry

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Standard agent response format"""
    success: bool
    results: List[Any]
    message: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseAgent(ABC):
    """
    Base class for all agents.
    Provides common functionality for agent communication and registration.
    """

    def __init__(
        self,
        name: str,
        description: str,
        capabilities: List[AgentCapability],
        message_bus: Optional[MessageBus] = None,
        registry: Optional[AgentRegistry] = None
    ):
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.message_bus = message_bus or get_message_bus()
        self.registry = registry or get_registry()

        # Register this agent
        self.registry.register_agent(
            name=self.name,
            description=self.description,
            capabilities=self.capabilities
        )

        # Subscribe to messages directed to this agent
        self.message_bus.subscribe(self.name, self._handle_message)
        self.message_bus.subscribe("all", self._handle_broadcast)

    @abstractmethod
    def process_query(self, query: str, context: Dict[str, Any] = None) -> AgentResponse:
        """
        Process a query. Must be implemented by subclasses.

        Args:
            query: User query
            context: Optional context dictionary

        Returns:
            AgentResponse with results
        """
        pass

    def _handle_message(self, message: AgentMessage):
        """Handle incoming message"""
        logger.debug("%s received message: to=%s, type=%s, from=%s", self.name, message.to_agent, message.type.value, message.from_agent)
        if message.to_agent == self.name and message.type == MessageType.QUERY:
            try:
                query = message.content.get("query", "")
                context = message.content.get("context", {})
                logger.debug("%s processing query: %.50s...", self.name, query)
                response = self.process_query(query, context)
                logger.debug("%s query processed: success=%s, results=%s", self.name, response.success, len(response.results))
                self.message_bus.send_message(
                    from_agent=self.name,
                    to_agent=message.from_agent,
                    content={
                        "results": response.results,
                        "message": response.message,
                        "metadata": response.metadata
                    },
                    message_type=MessageType.RESPONSE,
                    correlation_id=message.correlation_id
                )
            except Exception as e:
                logger.warning("%s error processing query: %s", self.name, e, exc_info=True)
                import traceback
                traceback.print_exc()
                # Send error response
                self.message_bus.send_message(
                    from_agent=self.name,
                    to_agent=message.from_agent,
                    content={
                        "error": str(e),
                        "message": f"Error processing query: {e}"
                    },
                    message_type=MessageType.ERROR,
                    correlation_id=message.correlation_id
                )
        else:
            logger.debug("Skipping message (to=%s, type=%s)", message.to_agent, message.type.value)

    def _handle_broadcast(self, message: AgentMessage):
        """Handle broadcast messages (optional)"""
        # Override in subclasses if needed
        pass

    def send_message(
        self,
        to_agent: str,
        content: Dict[str, Any],
        message_type: MessageType = MessageType.REQUEST
    ) -> AgentMessage:
        """Send a message to another agent"""
        return self.message_bus.send_message(
            from_agent=self.name,
            to_agent=to_agent,
            content=content,
            message_type=message_type
        )

    def broadcast(self, content: Dict[str, Any], message_type: MessageType = MessageType.NOTIFICATION):
        """Broadcast a message to all agents"""
        return self.message_bus.broadcast(
            from_agent=self.name,
            content=content,
            message_type=message_type
        )

    def update_status(self, status: str):
        """Update agent status"""
        from communication.agent_registry import AgentStatus
        status_enum = AgentStatus[status.upper()] if hasattr(AgentStatus, status.upper()) else AgentStatus.IDLE
        self.registry.update_agent_status(self.name, status_enum)

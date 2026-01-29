"""
Message Bus for Agent Communication
Provides pub/sub messaging between agents
"""

import logging
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import uuid
from enum import Enum

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Message types for agent communication"""
    QUERY = "query"
    RESPONSE = "response"
    REQUEST = "request"
    NOTIFICATION = "notification"
    ERROR = "error"


@dataclass
class AgentMessage:
    """Message structure for agent communication"""
    id: str
    type: MessageType
    from_agent: str
    to_agent: Optional[str]  # None for broadcast
    content: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None  # For request/response pairing
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            "id": self.id,
            "type": self.type.value,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "metadata": self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create message from dictionary"""
        return cls(
            id=data["id"],
            type=MessageType(data["type"]),
            from_agent=data["from_agent"],
            to_agent=data.get("to_agent"),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            correlation_id=data.get("correlation_id"),
            metadata=data.get("metadata")
        )


class MessageBus:
    """
    Message bus for agent communication.
    Supports pub/sub pattern with topic-based routing.
    """
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._message_history: List[AgentMessage] = []
        self._max_history = 1000
    
    def subscribe(self, topic: str, callback: Callable[[AgentMessage], None]):
        """
        Subscribe to a topic.
        
        Args:
            topic: Topic name (e.g., "search", "analysis", "all")
            callback: Function to call when message is received
        """
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)
    
    def unsubscribe(self, topic: str, callback: Callable[[AgentMessage], None]):
        """Unsubscribe from a topic"""
        if topic in self._subscribers:
            self._subscribers[topic].remove(callback)
    
    def publish(self, message: AgentMessage):
        """
        Publish a message to subscribers.
        
        Args:
            message: Message to publish
        """
        # Store in history
        self._message_history.append(message)
        if len(self._message_history) > self._max_history:
            self._message_history.pop(0)
        
        # Determine topics to publish to (directed QUERY only goes to the intended agent for performance)
        topics = []
        if message.to_agent:
            topics.append(message.to_agent)
        if message.type == MessageType.QUERY and message.to_agent:
            # Directed query: only deliver to that agent (avoids extra callback invocations)
            pass
        else:
            topics.append(message.type.value)
            topics.append("all")
        
        # Publish to all relevant subscribers
        logger.debug("Publishing message to topics: %s", topics)
        for topic in topics:
            if topic in self._subscribers:
                callbacks = self._subscribers[topic]
                logger.debug("Topic %r has %s subscriber(s)", topic, len(callbacks))
                for callback in callbacks:
                    try:
                        callback(message)
                    except Exception as e:
                        logger.warning("Error in subscriber callback for topic %s: %s", topic, e, exc_info=True)
            else:
                logger.debug("No subscribers for topic %r", topic)
    
    def send_message(
        self,
        from_agent: str,
        to_agent: str,
        content: Dict[str, Any],
        message_type: MessageType = MessageType.REQUEST,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """
        Send a message to a specific agent.
        
        Args:
            from_agent: Sender agent name
            to_agent: Recipient agent name
            content: Message content
            message_type: Type of message
            correlation_id: Optional correlation ID for request/response
            metadata: Optional metadata
            
        Returns:
            Created message
        """
        message = AgentMessage(
            id=str(uuid.uuid4()),
            type=message_type,
            from_agent=from_agent,
            to_agent=to_agent,
            content=content,
            timestamp=datetime.now(),
            correlation_id=correlation_id,
            metadata=metadata
        )
        self.publish(message)
        return message
    
    def broadcast(
        self,
        from_agent: str,
        content: Dict[str, Any],
        message_type: MessageType = MessageType.NOTIFICATION,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """
        Broadcast a message to all agents.
        
        Args:
            from_agent: Sender agent name
            content: Message content
            message_type: Type of message
            metadata: Optional metadata
            
        Returns:
            Created message
        """
        message = AgentMessage(
            id=str(uuid.uuid4()),
            type=message_type,
            from_agent=from_agent,
            to_agent=None,  # None = broadcast
            content=content,
            timestamp=datetime.now(),
            metadata=metadata
        )
        self.publish(message)
        return message
    
    def get_history(self, agent_name: Optional[str] = None, limit: int = 100) -> List[AgentMessage]:
        """
        Get message history.
        
        Args:
            agent_name: Filter by agent name (None for all)
            limit: Maximum number of messages to return
            
        Returns:
            List of messages
        """
        messages = self._message_history[-limit:]
        if agent_name:
            messages = [m for m in messages if m.from_agent == agent_name or m.to_agent == agent_name]
        return messages


# Global message bus instance
_message_bus: Optional[MessageBus] = None


def get_message_bus() -> MessageBus:
    """Get or create global message bus instance"""
    global _message_bus
    if _message_bus is None:
        _message_bus = MessageBus()
    return _message_bus

"""
Shared API Schemas
Common data structures for inter-service communication
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class MessageRole(Enum):
    """Message roles in chat"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    """Chat message structure"""
    role: MessageRole
    content: str
    timestamp: datetime
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class QueryRequest:
    """Query request structure"""
    query: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class QueryResponse:
    """Query response structure"""
    query_id: str
    status: str
    results: List[Dict[str, Any]]
    message: str
    agents_responded: int = 0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentInfo:
    """Agent information structure"""
    name: str
    description: str
    capabilities: List[str]
    status: str

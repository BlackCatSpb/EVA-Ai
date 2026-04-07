"""
Типы для Chat Module
Часть модуля chat_module.py (разделение на логические компоненты)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class MessageType(Enum):
    """Типы сообщений."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class MessageRole(Enum):
    """Роли в чате."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    """Сообщение в чате."""
    message_id: str
    role: MessageRole
    content: str
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    attachments: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "role": self.role.value if isinstance(self.role, MessageRole) else self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "attachments": self.attachments
        }


@dataclass
class ChatSession:
    """Сессия чата."""
    session_id: str
    user_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: float = 0.0
    last_activity: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "context": self.context,
            "is_active": self.is_active
        }


@dataclass
class ChatContext:
    """Контекст чата."""
    session_id: str
    user_id: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    current_topic: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "conversation_history": self.conversation_history,
            "current_topic": self.current_topic,
            "entities": self.entities,
            "preferences": self.preferences
        }

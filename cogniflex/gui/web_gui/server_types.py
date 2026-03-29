"""
Типы для Web GUI Server
Часть модуля server.py (разделение на логические компоненты)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class SessionStatus(Enum):
    """Статус сессии."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"


@dataclass
class WebSession:
    """Веб-сессия."""
    session_id: str
    user_id: str
    username: str
    created_at: str
    last_active: str
    context_nodes: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    status: SessionStatus = SessionStatus.ACTIVE
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "username": self.username,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "context_nodes": self.context_nodes,
            "entities": self.entities,
            "status": self.status.value if isinstance(self.status, SessionStatus) else self.status
        }


@dataclass
class UserCredentials:
    """Учётные данные пользователя."""
    username: str
    password_hash: str
    created_at: str
    role: str = "user"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "username": self.username,
            "password_hash": self.password_hash,
            "created_at": self.created_at,
            "role": self.role
        }

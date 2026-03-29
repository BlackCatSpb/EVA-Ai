"""
Типы памяти для MemoryManager
Часть модуля memory_manager.py (разделение на логические компоненты)
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class MemoryType(Enum):
    """Типы памяти в системе CogniFlex."""
    WORKING = "working"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    LONG_TERM = "long_term"
    SHORT_TERM = "short_term"
    USER_PROFILE = "user_profile"


@dataclass
class MemoryEntry:
    """Запись в памяти."""
    key: str
    value: Any
    memory_type: MemoryType
    timestamp: float
    access_count: int = 0
    last_access: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    decay_factor: float = 0.95
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "memory_type": self.memory_type.value,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "last_access": self.last_access,
            "metadata": self.metadata,
            "importance": self.importance,
            "decay_factor": self.decay_factor
        }


@dataclass  
class UserProfile:
    """Профиль пользователя."""
    user_id: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_level: str = "beginner"
    learning_style: str = "visual"
    cultural_profile: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    last_updated: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "preferences": self.preferences,
            "interaction_history": self.interaction_history,
            "knowledge_level": self.knowledge_level,
            "learning_style": self.learning_style,
            "cultural_profile": self.cultural_profile,
            "timestamp": self.timestamp,
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        return cls(
            user_id=data.get("user_id", ""),
            preferences=data.get("preferences", {}),
            interaction_history=data.get("interaction_history", []),
            knowledge_level=data.get("knowledge_level", "beginner"),
            learning_style=data.get("learning_style", "visual"),
            cultural_profile=data.get("cultural_profile", {}),
            timestamp=data.get("timestamp", 0.0),
            last_updated=data.get("last_updated", 0.0)
        )


@dataclass
class EpisodicMemory:
    """Эпизодическая память - хранит конкретные события/взаимодействия."""
    event_id: str
    user_id: str
    event_type: str
    content: str
    context: Dict[str, Any]
    emotional_valence: float = 0.0
    timestamp: float = 0.0
    duration: Optional[float] = None
    entities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "user_id": self.user_id,
            "event_type": self.event_type,
            "content": self.content,
            "context": self.context,
            "emotional_valence": self.emotional_valence,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "entities": self.entities
        }

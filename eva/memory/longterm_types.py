"""
Типы для Long Term Memory
Часть модуля memory_long_term.py (разделение на логические компоненты)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class MemoryCategory(Enum):
    """Категории памяти."""
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    AUTOBIOGRAPHICAL = "autobiographical"


class RetentionPolicy(Enum):
    """Политики хранения."""
    PERMANENT = "permanent"
    LONG_TERM = "long_term"
    SHORT_TERM = "short_term"
    EPHEMERAL = "ephemeral"


@dataclass
class SemanticEntry:
    """Семантическая запись."""
    entry_id: str
    concept: str
    representation: Dict[str, Any]
    strength: float = 1.0
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "concept": self.concept,
            "representation": self.representation,
            "strength": self.strength,
            "timestamp": self.timestamp
        }


@dataclass
class EpisodicEntry:
    """Эпизодическая запись."""
    episode_id: str
    event_type: str
    content: str
    context: Dict[str, Any]
    emotional_valence: float = 0.0
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "event_type": self.event_type,
            "content": self.content,
            "context": self.context,
            "emotional_valence": self.emotional_valence,
            "timestamp": self.timestamp
        }

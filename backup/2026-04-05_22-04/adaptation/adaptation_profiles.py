"""Модель профилей пользователей и фидбэка для системы адаптации ЕВА"""
import os
import logging
import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger("eva.adaptation.profiles")

@dataclass
class UserFeedback:
    """Представляет пользовательский фидбэк."""
    id: str
    user_id: str
    query: str
    response: str
    feedback_type: str  # positive, negative, neutral
    feedback_text: str
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует фидбэк в словарь."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "query": self.query,
            "response": self.response,
            "feedback_type": self.feedback_type,
            "feedback_text": self.feedback_text,
            "timestamp": self.timestamp,
            "context": self.context,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserFeedback':
        """Создает фидбэк из словаря."""
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            query=data["query"],
            response=data["response"],
            feedback_type=data["feedback_type"],
            feedback_text=data["feedback_text"],
            timestamp=data.get("timestamp", time.time()),
            context=data.get("context", {}),
            metadata=data.get("metadata", {})
        )


@dataclass
class UserProfile:
    """Представляет профиль пользователя для адаптации."""
    user_id: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    adaptation_level: float = 0.5
    last_updated: float = field(default_factory=time.time)
    learning_style: str = "balanced"
    knowledge_level: float = 0.5
    response_preferences: Dict[str, float] = field(default_factory=lambda: {"formal": 0.5, "casual": 0.5})
    cultural_profile: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует профиль в словарь."""
        return {
            "user_id": self.user_id,
            "preferences": self.preferences,
            "interaction_history": self.interaction_history,
            "adaptation_level": self.adaptation_level,
            "last_updated": self.last_updated,
            "learning_style": self.learning_style,
            "knowledge_level": self.knowledge_level,
            "response_preferences": self.response_preferences,
            "cultural_profile": self.cultural_profile
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """Создает профиль из словаря."""
        return cls(
            user_id=data["user_id"],
            preferences=data.get("preferences", {}),
            interaction_history=data.get("interaction_history", []),
            adaptation_level=data.get("adaptation_level", 0.5),
            last_updated=data.get("last_updated", time.time()),
            learning_style=data.get("learning_style", "balanced"),
            knowledge_level=data.get("knowledge_level", 0.5),
            response_preferences=data.get("response_preferences", {"formal": 0.5, "casual": 0.5}),
            cultural_profile=data.get("cultural_profile", {})
        )
"""
Типы для Adaptation
Часть модулей adaptation_manager.py (разделение на логические компоненты)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class AdaptationLevel(Enum):
    """Уровни адаптации."""
    USER = "user"
    SESSION = "session"
    CONTEXT = "context"
    SYSTEM = "system"


class LearningStyle(Enum):
    """Стили обучения."""
    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    READING = "reading"
    MIXED = "mixed"


@dataclass
class UserPreferences:
    """Предпочтения пользователя."""
    user_id: str
    preferred_domains: List[str] = field(default_factory=list)
    learning_style: LearningStyle = LearningStyle.MIXED
    language: str = "ru"
    timezone: str = "UTC"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "preferred_domains": self.preferred_domains,
            "learning_style": self.learning_style.value if isinstance(self.learning_style, LearningStyle) else self.learning_style,
            "language": self.language,
            "timezone": self.timezone
        }


@dataclass
class AdaptationProfile:
    """Профиль адаптации."""
    profile_id: str
    user_id: str
    level: AdaptationLevel
    settings: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "user_id": self.user_id,
            "level": self.level.value if isinstance(self.level, AdaptationLevel) else self.level,
            "settings": self.settings,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

"""
Типы для Response Generator
Часть модуля response_generator.py (разделение на логические компоненты)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class ResponseStyle(Enum):
    """Стили ответа."""
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    CREATIVE = "creative"


class ResponseType(Enum):
    """Типы ответа."""
    ANSWER = "answer"
    QUESTION = "question"
    EXPLANATION = "explanation"
    SUMMARY = "summary"


@dataclass
class ResponseContext:
    """Контекст ответа."""
    query: str
    user_id: str
    style: ResponseStyle
    max_length: int = 500
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "user_id": self.user_id,
            "style": self.style.value if isinstance(self.style, ResponseStyle) else self.style,
            "max_length": self.max_length
        }


@dataclass
class GeneratedResponse:
    """Сгенерированный ответ."""
    text: str
    confidence: float
    response_type: ResponseType
    sources: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "response_type": self.response_type.value if isinstance(self.response_type, ResponseType) else self.response_type,
            "sources": self.sources
        }

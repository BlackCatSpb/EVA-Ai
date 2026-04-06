"""Shared types for self-dialog learning: enums, dataclasses."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DialogRole(Enum):
    """Роли участников самодиалога."""
    ASSISTANT = "assistant"
    CRITIC = "critic"
    LEARNER = "learner"
    TEACHER = "teacher"
    OBSERVER = "observer"


class LearningType(Enum):
    """Типы обучения."""
    EXPANSION = "expansion"
    REFINEMENT = "refinement"
    UPDATING = "updating"
    INTEGRATION = "integration"


@dataclass
class DialogTurn:
    """Один ход в самодиалоге."""
    role: DialogRole
    content: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0


@dataclass
class SelfDialog:
    """Полный самодиалог системы с самим собой."""
    id: str
    topic: str
    turns: List[DialogTurn]
    start_time: float
    end_time: Optional[float] = None
    outcome: Optional[str] = None
    learning_type: Optional[LearningType] = None
    knowledge_gaps: List[str] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)

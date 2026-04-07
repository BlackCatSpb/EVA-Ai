"""
Типы для Core Reasoning Engine
Часть модуля reasoning_engine.py (разделение на логические компоненты)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class ReasoningType(Enum):
    """Типы рассуждений."""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"


class ReasoningStatus(Enum):
    """Статусы рассуждения."""
    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReasoningContext:
    """Контекст рассуждения."""
    query: str
    history: List[Dict[str, Any]] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    max_depth: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "history": self.history,
            "constraints": self.constraints,
            "max_depth": self.max_depth
        }


@dataclass
class ReasoningChain:
    """Цепочка рассуждений."""
    chain_id: str
    reasoning_type: ReasoningType
    steps: List[Dict[str, Any]] = field(default_factory=list)
    status: ReasoningStatus
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "reasoning_type": self.reasoning_type.value if isinstance(self.reasoning_type, ReasoningType) else self.reasoning_type,
            "steps": self.steps,
            "status": self.status.value if isinstance(self.status, ReasoningStatus) else self.status,
            "confidence": self.confidence
        }

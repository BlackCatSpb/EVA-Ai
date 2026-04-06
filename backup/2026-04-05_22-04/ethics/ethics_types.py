"""
Типы для Ethics Framework
Часть модуля ethics_framework.py (разделение на логические компоненты)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class EthicalPrincipleType(Enum):
    """Типы этических принципов."""
    HARM_PREVENTION = "harm_prevention"
    FAIRNESS = "fairness"
    TRANSPARENCY = "transparency"
    PRIVACY = "privacy"
    ACCOUNTABILITY = "accountability"
    BENEFICENCE = "beneficence"


class ViolationSeverity(Enum):
    """Серьёзность нарушения."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class EthicalPrinciple:
    """Этический принцип."""
    name: str
    description: str
    principle_type: EthicalPrincipleType
    weight: float = 1.0
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "principle_type": self.principle_type.value if isinstance(self.principle_type, EthicalPrincipleType) else self.principle_type,
            "weight": self.weight,
            "is_active": self.is_active
        }


@dataclass
class EthicalDecision:
    """Этическое решение."""
    decision_id: str
    timestamp: float
    context: Dict[str, Any]
    principle_applied: str
    outcome: str
    reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp,
            "context": self.context,
            "principle_applied": self.principle_applied,
            "outcome": self.outcome,
            "reasoning": self.reasoning
        }


@dataclass
class EthicalIssue:
    """Этическая проблема."""
    issue_id: str
    description: str
    severity: ViolationSeverity
    category: str
    detected_at: float
    resolved: bool = False
    resolution: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "description": self.description,
            "severity": self.severity.value if isinstance(self.severity, ViolationSeverity) else self.severity,
            "category": self.category,
            "detected_at": self.detected_at,
            "resolved": self.resolved,
            "resolution": self.resolution
        }


@dataclass
class EthicsAnalysisResult:
    """Результат этического анализа."""
    is_allowed: bool
    confidence: float
    issues: List[EthicalIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    principles_violated: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_allowed": self.is_allowed,
            "confidence": self.confidence,
            "issues": [i.to_dict() for i in self.issues],
            "recommendations": self.recommendations,
            "principles_violated": self.principles_violated,
            "metadata": self.metadata
        }

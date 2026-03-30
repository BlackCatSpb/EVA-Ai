"""
Типы для Contradiction Detection
Часть модуля contradiction_detection.py (разделение на логические компоненты)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class ContradictionType(Enum):
    """Типы противоречий."""
    LOGICAL = "logical"
    FACTUAL = "factual"
    TEMPORAL = "temporal"
    CONTEXTUAL = "contextual"
    SEMANTIC = "semantic"


class ContradictionSeverity(Enum):
    """Серьёзность противоречия."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Contradiction:
    """Противоречие."""
    contradiction_id: str
    type: ContradictionType
    severity: ContradictionSeverity
    statement_a: str
    statement_b: str
    evidence: List[str] = field(default_factory=list)
    resolution: Optional[str] = None
    detected_at: float = 0.0
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "contradiction_id": self.contradiction_id,
            "type": self.type.value if isinstance(self.type, ContradictionType) else self.type,
            "severity": self.severity.value if isinstance(self.severity, ContradictionSeverity) else self.severity,
            "statement_a": self.statement_a,
            "statement_b": self.statement_b,
            "evidence": self.evidence,
            "resolution": self.resolution,
            "detected_at": self.detected_at,
            "resolved": self.resolved
        }


@dataclass
class ContradictionReport:
    """Отчёт о противоречиях."""
    report_id: str
    total_contradictions: int
    unresolved_count: int
    contradictions: List[Contradiction] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    created_at: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "total_contradictions": self.total_contradictions,
            "unresolved_count": self.unresolved_count,
            "contradictions": [c.to_dict() for c in self.contradictions],
            "recommendations": self.recommendations,
            "created_at": self.created_at
        }

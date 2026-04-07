"""
Типы для Self Analyzer
Часть модуля self_analyzer.py (разделение на логические компоненты)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class AnalysisType(Enum):
    """Типы анализа."""
    PERFORMANCE = "performance"
    PATTERN = "pattern"
    ANOMALY = "anomaly"
    OPPORTUNITY = "opportunity"


class OpportunityPriority(Enum):
    """Приоритеты возможностей."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class AnalysisResult:
    """Результат анализа."""
    analysis_id: str
    analysis_type: AnalysisType
    findings: List[Dict[str, Any]] = field(default_factory=list)
    score: float = 0.0
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "analysis_type": self.analysis_type.value if isinstance(self.analysis_type, AnalysisType) else self.analysis_type,
            "findings": self.findings,
            "score": self.score,
            "timestamp": self.timestamp
        }


@dataclass
class LearningOpportunity:
    """Возможность для обучения."""
    opportunity_id: str
    description: str
    priority: OpportunityPriority
    estimated_impact: float
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "opportunity_id": self.opportunity_id,
            "description": self.description,
            "priority": self.priority.value if isinstance(self.priority, OpportunityPriority) else self.priority,
            "estimated_impact": self.estimated_impact,
            "timestamp": self.timestamp
        }

"""
Типы для Analytics
Часть модуля analytics_module.py (разделение на логические компоненты)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class MetricType(Enum):
    """Типы метрик."""
    PERFORMANCE = "performance"
    USAGE = "usage"
    QUALITY = "quality"
    SYSTEM = "system"


class TimeRange(Enum):
    """Временные диапазоны."""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


@dataclass
class AnalyticsMetric:
    """Аналитическая метрика."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "metric_type": self.metric_type.value if isinstance(self.metric_type, MetricType) else self.metric_type,
            "timestamp": self.timestamp,
            "unit": self.unit,
            "tags": self.tags
        }


@dataclass
class AnalyticsReport:
    """Аналитический отчёт."""
    report_id: str
    title: str
    time_range: TimeRange
    metrics: List[AnalyticsMetric] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "title": self.title,
            "time_range": self.time_range.value if isinstance(self.time_range, TimeRange) else self.time_range,
            "metrics": [m.to_dict() for m in self.metrics],
            "summary": self.summary,
            "created_at": self.created_at
        }

"""
Типы для System Monitoring
Часть модулей system/ (разделение на логические компоненты)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class HealthStatus(Enum):
    """Статусы здоровья."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AlertLevel(Enum):
    """Уровни алертов."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SystemHealth:
    """Здоровье системы."""
    status: HealthStatus
    uptime: float
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value if isinstance(self.status, HealthStatus) else self.status,
            "uptime": self.uptime,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "disk_percent": self.disk_percent
        }


@dataclass
class SystemAlert:
    """Системный алерт."""
    alert_id: str
    level: AlertLevel
    message: str
    source: str
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "level": self.level.value if isinstance(self.level, AlertLevel) else self.level,
            "message": self.message,
            "source": self.source,
            "timestamp": self.timestamp
        }

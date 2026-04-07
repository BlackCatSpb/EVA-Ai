"""
Типы для Background Coordinator
Часть модуля background_coordinator.py (разделение на логические компоненты)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class TaskPriority(Enum):
    """Приоритеты задач."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    IDLE = 5


class TaskStatus(Enum):
    """Статусы задач."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackgroundTask:
    """Фоновая задача."""
    task_id: str
    name: str
    task_type: str
    priority: TaskPriority
    status: TaskStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: float = 0.0
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "task_type": self.task_type,
            "priority": self.priority.value if isinstance(self.priority, TaskPriority) else self.priority,
            "status": self.status.value if isinstance(self.status, TaskStatus) else self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": self.progress,
            "error": self.error
        }


@dataclass
class TaskSchedule:
    """Расписание задачи."""
    task_id: str
    cron_expression: Optional[str] = None
    interval_seconds: Optional[float] = None
    enabled: bool = True
    next_run: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "cron_expression": self.cron_expression,
            "interval_seconds": self.interval_seconds,
            "enabled": self.enabled,
            "next_run": self.next_run
        }

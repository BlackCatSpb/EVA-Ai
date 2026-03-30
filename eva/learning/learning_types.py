"""
Типы для Learning Scheduler
Часть модуля learning_scheduler.py (разделение на логические компоненты)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class LearningTaskType(Enum):
    """Типы задач обучения."""
    CONCEPT_LEARNING = "concept_learning"
    KNOWLEDGE_UPDATE = "knowledge_update"
    KNOWLEDGE_ANALYSIS = "knowledge_analysis"
    MAP_CONNECTIONS = "map_connections"
    MAINTAIN_KNOWLEDGE = "maintain_knowledge"
    CONTRADICTION_RESOLUTION = "contradiction_resolution"
    TOPIC_EXPLORATION = "topic_exploration"


class LearningTaskStatus(Enum):
    """Статусы задач обучения."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class LearningTask:
    """Задача обучения."""
    task_id: str
    task_type: LearningTaskType
    concept: str
    priority: int = 5
    status: LearningTaskStatus = LearningTaskStatus.PENDING
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value if isinstance(self.task_type, LearningTaskType) else self.task_type,
            "concept": self.concept,
            "priority": self.priority,
            "status": self.status.value if isinstance(self.status, LearningTaskStatus) else self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata
        }


@dataclass
class LearningPlan:
    """План обучения."""
    plan_id: str
    user_id: str
    target_concepts: List[str]
    current_concept: Optional[str] = None
    completed_concepts: List[str] = field(default_factory=list)
    failed_concepts: List[str] = field(default_factory=list)
    progress: float = 0.0
    estimated_duration: float = 0.0
    actual_duration: float = 0.0
    created_at: float = 0.0
    updated_at: float = 0.0
    status: str = "active"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "user_id": self.user_id,
            "target_concepts": self.target_concepts,
            "current_concept": self.current_concept,
            "completed_concepts": self.completed_concepts,
            "failed_concepts": self.failed_concepts,
            "progress": self.progress,
            "estimated_duration": self.estimated_duration,
            "actual_duration": self.actual_duration,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status
        }


@dataclass
class ResourceAllocation:
    """Распределение ресурсов для обучения."""
    max_memory_mb: int = 512
    max_cpu_percent: int = 50
    max_gpu_memory_mb: int = 1024
    batch_size: int = 16
    learning_rate: float = 0.001
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_memory_mb": self.max_memory_mb,
            "max_cpu_percent": self.max_cpu_percent,
            "max_gpu_memory_mb": self.max_gpu_memory_mb,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate
        }

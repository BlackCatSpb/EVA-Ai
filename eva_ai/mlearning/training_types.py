"""
Типы для Training Orchestrator
Часть модуля training_orchestrator.py (разделение на логические компоненты)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class TrainingPhase(Enum):
    """Фазы обучения."""
    PREPROCESSING = "preprocessing"
    TOKENIZATION = "tokenization"
    EMBEDDING = "embedding"
    TRAINING = "training"
    VALIDATION = "validation"
    COMPLETED = "completed"


class CheckpointStatus(Enum):
    """Статус контрольной точки."""
    CREATED = "created"
    LOADED = "loaded"
    FAILED = "failed"


@dataclass
class TrainingCheckpoint:
    """Контрольная точка обучения."""
    checkpoint_id: str
    epoch: int
    step: int
    path: str
    status: CheckpointStatus
    timestamp: float
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "epoch": self.epoch,
            "step": self.step,
            "path": self.path,
            "status": self.status.value if isinstance(self.status, CheckpointStatus) else self.status,
            "timestamp": self.timestamp,
            "metrics": self.metrics
        }


@dataclass
class TrainingMetrics:
    """Метрики обучения."""
    epoch: int
    step: int
    loss: float
    accuracy: float
    learning_rate: float
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "loss": self.loss,
            "accuracy": self.accuracy,
            "learning_rate": self.learning_rate,
            "timestamp": self.timestamp
        }

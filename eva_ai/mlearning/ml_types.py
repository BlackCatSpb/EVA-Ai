"""
Типы для ML Unit
Часть модуля ml_unit.py (разделение на логические компоненты)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class ModelType(Enum):
    """Типы моделей."""
    QWEN = "qwen"
    FRACTAL = "fractal"
    BITNET = "bitnet"
    HYBRID = "hybrid"


class ProcessingMode(Enum):
    """Режимы обработки."""
    SYNC = "sync"
    ASYNC = "async"
    STREAM = "stream"


@dataclass
class MLModelConfig:
    """Конфигурация ML модели."""
    model_type: ModelType
    model_path: str
    device: str = "auto"
    max_memory_mb: int = 2048
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_quantization: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type.value if isinstance(self.model_type, ModelType) else self.model_type,
            "model_path": self.model_path,
            "device": self.device,
            "max_memory_mb": self.max_memory_mb,
            "load_in_8bit": self.load_in_8bit,
            "load_in_4bit": self.load_in_4bit,
            "use_quantization": self.use_quantization
        }


@dataclass
class TrainingConfig:
    """Конфигурация обучения."""
    batch_size: int = 16
    learning_rate: float = 0.001
    epochs: int = 10
    max_seq_length: int = 512
    warmup_steps: int = 100
    gradient_accumulation: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "max_seq_length": self.max_seq_length,
            "warmup_steps": self.warmup_steps,
            "gradient_accumulation": self.gradient_accumulation
        }


@dataclass
class ModelStats:
    """Статистика модели."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_processing_time: float = 0.0
    last_request_time: float = 0.0
    avg_inference_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_processing_time": self.total_processing_time,
            "last_request_time": self.last_request_time,
            "avg_inference_time": self.avg_inference_time
        }

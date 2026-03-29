"""
Типы для Storage
Часть модулей storage/ (разделение на логические компоненты)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class StorageType(Enum):
    """Типы хранилища."""
    MEMORY = "memory"
    DISK = "disk"
    DISTRIBUTED = "distributed"
    CACHE = "cache"


class AccessPattern(Enum):
    """Паттерны доступа."""
    RANDOM = "random"
    SEQUENTIAL = "sequential"
    STREAMING = "streaming"


@dataclass
class StorageMetrics:
    """Метрики хранилища."""
    total_capacity: int = 0
    used_space: int = 0
    available_space: int = 0
    access_count: int = 0
    hit_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_capacity": self.total_capacity,
            "used_space": self.used_space,
            "available_space": self.available_space,
            "access_count": self.access_count,
            "hit_rate": self.hit_rate
        }


@dataclass
class StorageEntry:
    """Запись в хранилище."""
    key: str
    value: Any
    storage_type: StorageType
    size_bytes: int
    created_at: float = 0.0
    accessed_at: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "storage_type": self.storage_type.value if isinstance(self.storage_type, StorageType) else self.storage_type,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at
        }

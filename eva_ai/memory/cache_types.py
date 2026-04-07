"""
Типы для Memory Cache
Часть модулей hybrid_token_cache.py, disk_cache.py (разделение на логические компоненты)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class CacheLevel(Enum):
    """Уровни кэша."""
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    DISK = "disk"


class CacheStrategy(Enum):
    """Стратегии кэширования."""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    ADAPTIVE = "adaptive"


@dataclass
class CacheEntry:
    """Запись в кэше."""
    key: str
    value: Any
    level: CacheLevel
    size_bytes: int
    created_at: float
    last_access: float
    access_count: int = 0
    ttl: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "level": self.level.value if isinstance(self.level, CacheLevel) else self.level,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "last_access": self.last_access,
            "access_count": self.access_count,
            "ttl": self.ttl
        }


@dataclass
class CacheStats:
    """Статистика кэша."""
    total_entries: int = 0
    total_size_bytes: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    hit_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_entries": self.total_entries,
            "total_size_bytes": self.total_size_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate
        }


@dataclass
class CacheConfig:
    """Конфигурация кэша."""
    max_size_mb: int = 512
    max_entries: int = 10000
    default_ttl: float = 3600.0
    strategy: CacheStrategy = CacheStrategy.LRU
    enable_disk_cache: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_size_mb": self.max_size_mb,
            "max_entries": self.max_entries,
            "default_ttl": self.default_ttl,
            "strategy": self.strategy.value if isinstance(self.strategy, CacheStrategy) else self.strategy,
            "enable_disk_cache": self.enable_disk_cache
        }

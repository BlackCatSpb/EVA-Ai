"""
EvictionPolicy - политика вытеснения записей из кэша.
Поддерживает LRU, LFU и гибридные стратегии.
"""
import time
import logging
from typing import Dict, Optional, List
from collections import OrderedDict

logger = logging.getLogger("eva.memory.fractal_cache.eviction")


class EvictionPolicy:
    """
    Управляет вытеснением записей из кэша.
    
    Стратегии:
    - LRU: Least Recently Used (по умолчанию)
    - LFU: Least Frequently Used
    - Hybrid: Комбинация LRU + LFU
    """
    
    def __init__(self, max_entries: int = 500000, strategy: str = "lru"):
        self.max_entries = max_entries
        self.strategy = strategy
        
        # Структуры для отслеживания
        self._access_order: OrderedDict[str, float] = OrderedDict()  # LRU
        self._access_count: Dict[str, int] = {}  # LFU
        self._timestamps: Dict[str, float] = {}
        
        logger.info(f"EvictionPolicy инициализирован: max={max_entries}, strategy={strategy}")
    
    @property
    def size(self) -> int:
        """Возвращает текущее количество записей."""
        return len(self._access_order)
    
    def should_evict(self) -> bool:
        """Проверяет нужно ли вытеснение."""
        return self.size >= self.max_entries
    
    def get_eviction_candidate(self) -> Optional[str]:
        """
        Возвращает ключ записи для вытеснения.
        
        Returns:
            Optional[str]: Ключ или None
        """
        if not self._access_order:
            return None
        
        if self.strategy == "lru":
            # Наименее недавно использованный
            return next(iter(self._access_order))
        
        elif self.strategy == "lfu":
            # Наименее часто используемый
            if not self._access_count:
                return next(iter(self._access_order))
            
            min_count = min(self._access_count.values())
            candidates = [
                k for k, v in self._access_count.items()
                if v == min_count
            ]
            
            # Среди наименее частых выбираем самый старый
            for key in self._access_order:
                if key in candidates:
                    return key
            
            return candidates[0]
        
        else:  # hybrid
            # Комбинация: частота × время
            if not self._access_count:
                return next(iter(self._access_order))
            
            scores = {}
            now = time.time()
            
            for key in self._access_order:
                freq = self._access_count.get(key, 1)
                age = now - self._timestamps.get(key, now)
                scores[key] = freq / max(1, age)
            
            return min(scores, key=scores.get)
    
    def record_access(self, key: str):
        """
        Записывает факт обращения к записи.
        
        Args:
            key: Ключ записи
        """
        # Обновляем порядок (LRU)
        if key in self._access_order:
            self._access_order.move_to_end(key)
        else:
            self._access_order[key] = time.time()
        
        # Обновляем счётчик (LFU)
        self._access_count[key] = self._access_count.get(key, 0) + 1
        
        # Обновляем timestamp
        self._timestamps[key] = time.time()
    
    def remove(self, key: str):
        """
        Удаляет запись из отслеживания.
        
        Args:
            key: Ключ записи
        """
        self._access_order.pop(key, None)
        self._access_count.pop(key, None)
        self._timestamps.pop(key, None)
    
    def clear(self):
        """Очищает все данные."""
        self._access_order.clear()
        self._access_count.clear()
        self._timestamps.clear()
    
    def get_stats(self) -> Dict:
        """Возвращает статистику."""
        if not self._access_count:
            return {"size": 0, "avg_access": 0, "max_access": 0}
        
        return {
            "size": self.size,
            "avg_access": sum(self._access_count.values()) / max(1, len(self._access_count)),
            "max_access": max(self._access_count.values()) if self._access_count else 0
        }

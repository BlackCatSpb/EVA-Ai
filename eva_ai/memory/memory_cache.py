"""
Модуль для работы с кэшем в оперативной памяти
"""
import logging
import time
import threading
from collections import OrderedDict
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class MemoryCache:
    """Класс для управления кэшем в оперативной памяти с LRU логикой"""

    def __init__(self, max_memory_tokens: int = 10000):
        self.max_memory_tokens = max_memory_tokens
        self.memory_cache = OrderedDict()
        self.memory_lock = threading.Lock()

        # Статистика для памяти
        self.memory_hits = 0
        self.total_requests = 0

    def get(self, token_hash: str) -> Optional[Dict[str, Any]]:
        """Получает токен из памяти"""
        with self.memory_lock:
            self.total_requests += 1

            if token_hash in self.memory_cache:
                token_data = self.memory_cache[token_hash]
                self.memory_hits += 1
                # Перемещаем в конец (LRU)
                self.memory_cache.move_to_end(token_hash)
                return token_data

        return None

    def put(self, token_hash: str, token_data: Dict[str, Any]) -> None:
        """Добавляет токен в память"""
        with self.memory_lock:
            # Если память полна, удаляем самый старый элемент
            if len(self.memory_cache) >= self.max_memory_tokens:
                self.memory_cache.popitem(last=False)

            # Добавляем новый токен
            self.memory_cache[token_hash] = token_data

    def clear(self) -> None:
        """Очищает кэш в памяти"""
        with self.memory_lock:
            self.memory_cache.clear()
            self.memory_hits = 0
            self.total_requests = 0

    def remove_expired(self, expired_tokens: List[str]) -> None:
        """Удаляет устаревшие токены из памяти"""
        with self.memory_lock:
            for token_hash in expired_tokens:
                if token_hash in self.memory_cache:
                    del self.memory_cache[token_hash]

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику памяти"""
        with self.memory_lock:
            stats = {
                'memory_size': len(self.memory_cache),
                'max_memory_tokens': self.max_memory_tokens,
                'memory_hits': self.memory_hits,
                'memory_requests': self.total_requests
            }

            if self.total_requests > 0:
                stats['memory_hit_rate'] = self.memory_hits / self.total_requests
            else:
                stats['memory_hit_rate'] = 0.0

            return stats

    def __len__(self) -> int:
        """Возвращает количество токенов в памяти"""
        return len(self.memory_cache)

    def __contains__(self, token_hash: str) -> bool:
        """Проверяет наличие токена в кэше"""
        return token_hash in self.memory_cache

    def __getitem__(self, token_hash: str) -> Dict[str, Any]:
        """Получает токен (как словарь)"""
        result = self.get(token_hash)
        if result is None:
            raise KeyError(f"Token {token_hash} not found in memory cache")
        return result

"""RAM cache management, LRU/LFU strategies, hot token storage."""
import threading
from typing import Any, Optional
from collections import OrderedDict


class LRUCache:
    """Простая реализация LRU кэша на основе OrderedDict."""

    def __init__(self, max_size: int):
        self.max_size = max(1, max_size)
        self.cache = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            elif len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[key] = value

    def remove(self, key: str) -> bool:
        with self._lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self.cache.clear()

    def __contains__(self, key: str) -> bool:
        return key in self.cache

    def __len__(self) -> int:
        return len(self.cache)

    def keys(self):
        return list(self.cache.keys())

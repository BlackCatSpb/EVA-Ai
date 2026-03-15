"""Менеджер кэша для веб-поиска CogniFlex"""
import os
import json
import time
import hashlib
import logging
from typing import List, Dict, Any, Optional
from .search_models import SearchResult

logger = logging.getLogger("cogniflex.web_search.cache")

class CacheManager:
    """Управляет кэшированием результатов поиска."""
    
    def __init__(self, cache_dir: str, cache_ttl: int = 86400):
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl
        self.search_cache = {}
        self._load_cache()
    
    def _load_cache(self):
        """Загружает кэш поисковых запросов."""
        try:
            cache_path = os.path.join(self.cache_dir, "search_cache.json")
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    self.search_cache = json.load(f)
                logger.debug(f"Загружено {len(self.search_cache)} кэшированных запросов")
        except Exception as e:
            logger.error(f"Ошибка загрузки кэша поиска: {e}")
    
    def _save_cache(self):
        """Сохраняет кэш поисковых запросов."""
        try:
            cache_path = os.path.join(self.cache_dir, "search_cache.json")
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.search_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения кэша поиска: {e}")
    
    def generate_cache_key(self, query: str) -> str:
        """Генерирует ключ для кэша на основе запроса."""
        return hashlib.md5(query.encode()).hexdigest()
    
    def get_cached_results(self, query: str) -> Optional[List[SearchResult]]:
        """Получает результаты из кэша."""
        cache_key = self.generate_cache_key(query)
        if cache_key in self.search_cache:
            cached_data = self.search_cache[cache_key]
            if time.time() - cached_data["timestamp"] < self.cache_ttl:
                return [SearchResult(**item) for item in cached_data["results"]]
        return None
    
    def save_to_cache(self, query: str, results: List[SearchResult]):
        """Сохраняет результаты в кэш."""
        cache_key = self.generate_cache_key(query)
        self.search_cache[cache_key] = {
            "timestamp": time.time(),
            "results": [result.__dict__ for result in results]
        }
        self._save_cache()
    
    def clear_cache(self):
        """Очищает кэш."""
        self.search_cache = {}
        self._save_cache()
    
    def get_cache_size(self) -> int:
        """Возвращает размер кэша."""
        return len(self.search_cache)
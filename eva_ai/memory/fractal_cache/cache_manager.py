"""
FractalCache - главный менеджер фрактального кэша.
Хранит сгенерированные ответы с семантическим индексом.
"""
import os
import json
import time
import hashlib
import logging
from typing import Dict, Any, Optional, List, Tuple

from .semantic_embedder import SemanticEmbedder
from .response_store import ResponseStore
from .similarity_engine import SimilarityEngine
from .eviction_policy import EvictionPolicy

logger = logging.getLogger("eva_ai.memory.fractal_cache")


class FractalCache:
    """
    Фрактальный кэш с семантическим поиском.
    
    Параметры:
    - max_entries: Максимум записей (500K для мощных Qwen)
    - context_window: Максимальный контекст в токенах (128K)
    - similarity_threshold: Порог схожести (0.95)
    - cache_dir: Директория для хранения на диске
    """
    
    def __init__(
        self,
        max_entries: int = 500000,
        context_window: int = 128000,
        similarity_threshold: float = 0.95,
        cache_dir: Optional[str] = None
    ):
        self.max_entries = max_entries
        self.context_window = context_window
        self.similarity_threshold = similarity_threshold
        
        # Директория кэша
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), "fractal_cache_data"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Инициализация компонентов
        self.embedder = SemanticEmbedder()
        self._response_store = ResponseStore(self.cache_dir, max_entries)
        self.similarity = SimilarityEngine()
        self.eviction = EvictionPolicy(max_entries)
        
        # Статистика
        self.stats = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
            "total_queries": 0
        }
        
        logger.info(
            f"FractalCache инициализирован: max_entries={max_entries}, "
            f"context_window={context_window}, threshold={similarity_threshold}"
        )
    
    def store(self, query: str, response: str, metadata: Optional[Dict] = None) -> str:
        """
        Сохраняет ответ в кэш.
        
        Args:
            query: Запрос пользователя
            response: Ответ системы
            metadata: Дополнительные метаданные
            
        Returns:
            str: ID записи
        """
        # Генерируем ключ
        key = self._generate_key(query)
        
        # Создаём эмбеддинг
        embedding = self.embedder.encode(query)
        
        # Подготавливаем данные
        entry = {
            "key": key,
            "query": query,
            "response": response,
            "embedding": embedding,
            "metadata": metadata or {},
            "timestamp": time.time(),
            "access_count": 0
        }
        
        # Проверяем лимит
        if self.eviction.should_evict():
            evicted_key = self.eviction.get_eviction_candidate()
            if evicted_key:
                self._response_store.delete(evicted_key)
                self.stats["evictions"] += 1
        
        # Сохраняем
        self._response_store.save(key, entry)
        
        self.stats["stores"] += 1
        logger.debug(f"Кэш: сохранена запись {key[:16]}...")
        
        return key
    
    def search(self, query: str, top_k: int = 1) -> Optional[Dict[str, Any]]:
        """
        Ищет похожий запрос в кэше.
        
        Args:
            query: Запрос для поиска
            top_k: Количество лучших результатов
            
        Returns:
            Optional[Dict]: Найденный ответ или None
        """
        self.stats["total_queries"] += 1
        
        # Проверяем точное совпадение
        exact_key = self._generate_key(query)
        exact_match = self._response_store.load(exact_key)
        if exact_match:
            exact_match["access_count"] += 1
            self.eviction.record_access(exact_key)
            self.stats["hits"] += 1
            logger.info(f"Кэш: точное совпадение для {query[:30]}...")
            return {
                "response": exact_match["response"],
                "similarity": 1.0,
                "source": "cache_exact",
                "metadata": exact_match.get("metadata", {})
            }
        
        # Ищем семантическое совпадение
        query_embedding = self.embedder.encode(query)
        
        # Получаем кандидатов из кэша
        candidates = self._response_store.get_recent(limit=min(10000, self.eviction.size))
        
        best_match = None
        best_similarity = 0.0
        
        for entry in candidates:
            if "embedding" not in entry:
                continue
            
            sim = self.similarity.compute(
                query_embedding,
                entry["embedding"]
            )
            
            if sim > best_similarity:
                best_similarity = sim
                best_match = entry
        
        # Проверяем порог
        if best_similarity >= self.similarity_threshold and best_match:
            best_match["access_count"] += 1
            self.eviction.record_access(best_match["key"])
            self.stats["hits"] += 1
            
            logger.info(
                f"Кэш: семантическое совпадение ({best_similarity:.3f}) "
                f"для {query[:30]}..."
            )
            
            return {
                "response": best_match["response"],
                "similarity": best_similarity,
                "source": "cache_semantic",
                "metadata": best_match.get("metadata", {})
            }
        
        self.stats["misses"] += 1
        logger.debug(f"Кэш: промах для {query[:30]}...")
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику кэша."""
        total = self.stats["hits"] + self.stats["misses"]
        return {
            **self.stats,
            "size": self.eviction.size,
            "hit_rate": self.stats["hits"] / max(1, total),
            "utilization": self.eviction.size / max(1, self.max_entries)
        }
    
    def _generate_key(self, query: str) -> str:
        """Генерирует ключ для запроса."""
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def clear(self):
        """Очищает весь кэш."""
        self._response_store.clear_all()
        self.eviction.clear()
        self.stats = {k: 0 for k in self.stats}
        logger.info("Кэш очищен")

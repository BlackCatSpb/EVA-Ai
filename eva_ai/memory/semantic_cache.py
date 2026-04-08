"""
Semantic Cache - семантическое кэширование с порогом схожести.
Если cosine_similarity(query, cached_query) > threshold -> возвращает кэшированный ответ.
"""
import logging
import time
import threading
from typing import Dict, Optional, Any, List
from dataclasses import dataclass

logger = logging.getLogger("eva_ai.semantic_cache")

@dataclass
class CachedEntry:
    """Кэшированная запись с эмбеддингом."""
    query: str
    response: Dict[str, Any]
    embedding: List[float]
    timestamp: float
    access_count: int

class SemanticCache:
    """
    Семантическое кэширование с косинусной схожестью.
    Если similarity > 0.87 -> возвращает кэшированный ответ с лёгкой адаптацией.
    """
    
    def __init__(self, similarity_threshold: float = 0.87, max_entries: int = 200):
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self._cache: Dict[str, CachedEntry] = {}
        self._lock = threading.RLock()
        
    def get(self, query: str, query_embedding: List[float]) -> Optional[Dict[str, Any]]:
        """
        Получить кэшированный ответ по семантической схожести.
        
        Args:
            query: Текст запроса
            query_embedding: Эмбеддинг запроса
            
        Returns:
            Кэшированный ответ или None
        """
        if not query_embedding:
            return None
            
        with self._lock:
            best_match = None
            best_similarity = 0.0
            
            for key, entry in self._cache.items():
                similarity = self._cosine_similarity(query_embedding, entry.embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = entry
            
            if best_match and best_similarity >= self.similarity_threshold:
                best_match.access_count += 1
                logger.info(f"Semantic cache hit: similarity={best_similarity:.2f}, access_count={best_match.access_count}")
                return best_match.response
            
            return None
    
    def put(self, query: str, query_embedding: List[float], response: Dict[str, Any]) -> None:
        """
        Сохранить запрос и ответ в кэш.
        
        Args:
            query: Текст запроса
            query_embedding: Эмбеддинг запроса
            response: Ответ системы
        """
        with self._lock:
            cache_key = query.strip().lower()
            
            # Удаляем oldest если переполнено
            if len(self._cache) >= self.max_entries:
                self._evict_oldest()
            
            self._cache[cache_key] = CachedEntry(
                query=query,
                response=response,
                embedding=query_embedding,
                timestamp=time.time(),
                access_count=1
            )
            logger.debug(f"Added to semantic cache: {cache_key[:50]}")
    
    def _cosine_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Вычислить косинусное сходство между эмбеддингами."""
        if not emb1 or not emb2 or len(emb1) != len(emb2):
            return 0.0
            
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = sum(a * a for a in emb1) ** 0.5
        norm2 = sum(b * b for b in emb2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def _evict_oldest(self) -> None:
        """Удалить самую старую запись."""
        if not self._cache:
            return
            
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
        del self._cache[oldest_key]
    
    def clear(self) -> None:
        """Очистить кэш."""
        with self._lock:
            self._cache.clear()
            logger.info("Semantic cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику кэша."""
        with self._lock:
            total_accesses = sum(e.access_count for e in self._cache.values())
            return {
                "entries": len(self._cache),
                "max_entries": self.max_entries,
                "total_accesses": total_accesses,
                "similarity_threshold": self.similarity_threshold
            }


def create_semantic_cache(threshold: float = 0.87, max_entries: int = 200) -> SemanticCache:
    """Создать инстанс семантического кэша."""
    return SemanticCache(similarity_threshold=threshold, max_entries=max_entries)
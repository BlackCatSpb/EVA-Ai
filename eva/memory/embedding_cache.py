"""
Кеш эмбеддингов для EVA
Хранит вычисленные эмбеддинги текстов, избегая повторных вычислений.
Использует SQLite для persistence + LRU eviction.
"""
import os
import hashlib
import json
import sqlite3
import logging
import threading
from typing import Optional, List, Dict
from datetime import datetime

logger = logging.getLogger("eva.embedding_cache")


class EmbeddingCache:
    """
    Persistent LRU cache для эмбеддингов.
    
    - Ключ: SHA256 хеш текста
    - Значение: embedding vector (JSON array)
    - Max size: configurable (по умолчанию 100,000 записей)
    - Backend: SQLite для persistence
    """
    
    def __init__(self, cache_dir: str = None, max_size: int = 100000):
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     'memory', 'embedding_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        self.db_path = os.path.join(cache_dir, 'embeddings.db')
        self.max_size = max_size
        self._lock = threading.RLock()
        self._init_db()
        
        stats = self.get_stats()
        logger.info(f"EmbeddingCache инициализирован: {stats['count']} записей, max={max_size}")
    
    def _init_db(self):
        """Инициализация SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    hash TEXT PRIMARY KEY,
                    embedding TEXT NOT NULL,
                    text_preview TEXT,
                    dimension INTEGER,
                    created_at TEXT,
                    accessed_at TEXT,
                    access_count INTEGER DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_accessed_at ON embeddings(accessed_at)
            """)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.commit()
    
    def _hash_text(self, text: str) -> str:
        """SHA256 хеш текста."""
        return hashlib.sha256(text.strip().encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[List[float]]:
        """Получает эмбеддинг из кеша."""
        text_hash = self._hash_text(text)
        
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT embedding FROM embeddings WHERE hash = ?",
                    (text_hash,)
                )
                row = cursor.fetchone()
                if row:
                    embedding = json.loads(row[0])
                    # Обновляем статистику доступа
                    conn.execute(
                        "UPDATE embeddings SET accessed_at = ?, access_count = access_count + 1 WHERE hash = ?",
                        (datetime.now().isoformat(), text_hash)
                    )
                    conn.commit()
                    return embedding
        return None
    
    def put(self, text: str, embedding: List[float]):
        """Сохраняет эмбеддинг в кеш."""
        text_hash = self._hash_text(text)
        dimension = len(embedding)
        text_preview = text[:100]
        now = datetime.now().isoformat()
        
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO embeddings 
                       (hash, embedding, text_preview, dimension, created_at, accessed_at, access_count)
                       VALUES (?, ?, ?, ?, ?, ?, 1)""",
                    (text_hash, json.dumps(embedding), text_preview, dimension, now, now)
                )
                conn.commit()
            
            # Eviction если превышен лимит
            self._evict_if_needed()
    
    def get_or_compute(self, text: str, compute_fn) -> List[float]:
        """
        Получает из кеша или вычисляет через функцию.
        
        Args:
            text: Текст для эмбеддинга
            compute_fn: Функция вычисления эмбеддинга (принимает text, возвращает list[float])
        
        Returns:
            Embedding vector
        """
        cached = self.get(text)
        if cached is not None:
            return cached
        
        embedding = compute_fn(text)
        if embedding:
            self.put(text, embedding)
        return embedding
    
    def _evict_if_needed(self):
        """Удаляет старые записи если превышен лимит."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
                count = cursor.fetchone()[0]
                
                if count > self.max_size:
                    to_delete = count - self.max_size
                    conn.execute(
                        """DELETE FROM embeddings 
                           WHERE hash IN (
                               SELECT hash FROM embeddings 
                               ORDER BY accessed_at ASC 
                               LIMIT ?
                           )""",
                        (to_delete,)
                    )
                    conn.commit()
                    logger.info(f"EmbeddingCache: удалено {to_delete} старых записей")
    
    def get_stats(self) -> Dict:
        """Статистика кеша."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
                count = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT SUM(LENGTH(embedding)) FROM embeddings")
                size_bytes = cursor.fetchone()[0] or 0
                
                cursor = conn.execute("SELECT dimension FROM embeddings LIMIT 1")
                row = cursor.fetchone()
                dimension = row[0] if row else 0
                
                return {
                    'count': count,
                    'size_mb': round(size_bytes / (1024 * 1024), 2),
                    'dimension': dimension,
                    'max_size': self.max_size,
                    'hit_rate': self._compute_hit_rate() if count > 0 else 0,
                }
    
    def _compute_hit_rate(self) -> float:
        """Вычисляет hit rate кеша."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT AVG(access_count) FROM embeddings")
                row = cursor.fetchone()
                avg = row[0] if row and row[0] else 1.0
                return round(avg / (avg + 1) * 100, 1)
    
    def clear(self):
        """Очищает кеш."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM embeddings")
                conn.commit()
            logger.info("EmbeddingCache очищен")
    
    def batch_get(self, texts: List[str]) -> Dict[str, Optional[List[float]]]:
        """Batch получение эмбеддингов."""
        results = {}
        hashes_to_texts = {}
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                for text in texts:
                    text_hash = self._hash_text(text)
                    hashes_to_texts[text_hash] = text
                    results[text] = None

                if hashes_to_texts:
                    placeholders = ",".join("?" for _ in hashes_to_texts)
                    cursor = conn.execute(
                        f"SELECT hash, embedding FROM embeddings WHERE hash IN ({placeholders})",
                        tuple(hashes_to_texts.keys())
                    )
                    now = datetime.now().isoformat()
                    for row in cursor:
                        h, emb = row
                        text = hashes_to_texts[h]
                        results[text] = json.loads(emb)
                        conn.execute(
                            "UPDATE embeddings SET accessed_at = ?, access_count = access_count + 1 WHERE hash = ?",
                            (now, h)
                        )
                conn.commit()
        return results


# Singleton
_embedding_cache = None
_cache_lock = threading.Lock()


def get_embedding_cache(cache_dir: str = None, max_size: int = 100000) -> EmbeddingCache:
    """Возвращает singleton кеш эмбеддингов."""
    global _embedding_cache
    with _cache_lock:
        if _embedding_cache is None:
            _embedding_cache = EmbeddingCache(cache_dir=cache_dir, max_size=max_size)
        return _embedding_cache


def clear_embedding_cache():
    """Очищает кеш эмбеддингов."""
    global _embedding_cache
    with _cache_lock:
        if _embedding_cache:
            _embedding_cache.clear()
            _embedding_cache = None

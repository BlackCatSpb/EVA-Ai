"""
Дисковый кэш для токенов с SQLite метаданными.
"""

import os
import pickle
import sqlite3
import time
import zlib
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

class DiskCache:
    """Управление кэшем на жестком диске с SQLite метаданными."""
    
    def __init__(self, cache_dir: str, max_size_gb: float = 10.0):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.db_path = os.path.join(cache_dir, "cache_metadata.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_tables()
        
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.current_size = self._calculate_current_size()
    
    def _create_tables(self):
        """Создает таблицы метаданных."""
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            last_accessed REAL NOT NULL,
            created_at REAL NOT NULL
        )
        """)
        self.conn.commit()
    
    def _calculate_current_size(self) -> int:
        """Вычисляет текущий размер кэша."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT SUM(size_bytes) FROM metadata")
        result = cursor.fetchone()[0]
        return result or 0
    
    def _evict_old_entries(self):
        """Удаляет старые записи при превышении лимита."""
        while self.current_size > self.max_size_bytes * 0.9:
            cursor = self.conn.cursor()
            cursor.execute("""
            SELECT key, file_path, size_bytes 
            FROM metadata 
            ORDER BY last_accessed ASC 
            LIMIT 1
            """)
            result = cursor.fetchone()
            if not result:
                break
                
            key, file_path, size = result
            if os.path.exists(file_path):
                os.remove(file_path)
            
            cursor.execute("DELETE FROM metadata WHERE key = ?", (key,))
            self.conn.commit()
            self.current_size -= size
    
    def put(self, key: str, data: Any):
        """Сохраняет данные в кэш с компрессией."""
        try:
            file_path = os.path.join(self.cache_dir, f"{key}.pkl")
            
            # Сериализация и сжатие
            serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            compressed = zlib.compress(serialized)
            
            with open(file_path, "wb") as f:
                f.write(compressed)
            
            size = os.path.getsize(file_path)
            
            cursor = self.conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO metadata 
            (key, file_path, size_bytes, last_accessed, created_at)
            VALUES (?, ?, ?, ?, ?)
            """, (key, file_path, size, time.time(), time.time()))
            
            self.conn.commit()
            self.current_size += size
            self._evict_old_entries()
            
        except Exception as e:
            logger.error(f"Ошибка сохранения в дисковый кэш: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Получает данные из кэша."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT file_path FROM metadata WHERE key = ?", (key,))
            result = cursor.fetchone()
            
            if not result:
                return None
                
            file_path = result[0]
            if not os.path.exists(file_path):
                cursor.execute("DELETE FROM metadata WHERE key = ?", (key,))
                self.conn.commit()
                return None
            
            # Обновляем время доступа
            cursor.execute(
                "UPDATE metadata SET last_accessed = ? WHERE key = ?",
                (time.time(), key)
            )
            self.conn.commit()
            
            # Загружаем и распаковываем данные
            with open(file_path, "rb") as f:
                compressed = f.read()
                decompressed = zlib.decompress(compressed)
                return pickle.loads(decompressed)
                
        except Exception as e:
            logger.error(f"Ошибка загрузки из дискового кэша: {e}")
            return None
    
    def get_stats(self) -> dict:
        """Возвращает статистику кэша."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM metadata")
        count = cursor.fetchone()[0]
        
        return {
            "entries": count,
            "size_bytes": self.current_size,
            "size_mb": self.current_size / (1024**2),
            "max_size_gb": self.max_size_bytes / (1024**3)
        }
    
    def close(self):
        """Закрывает соединение с базой."""
        self.conn.close()
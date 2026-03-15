"""Менеджер базы данных для веб-поиска CogniFlex"""
import os
import sqlite3
import threading
import logging
from typing import List, Dict, Any
from datetime import datetime
from .search_models import SearchResult

logger = logging.getLogger("cogniflex.web_search.db")

class DatabaseManager:
    """Управляет базой данных для хранения истории поиска."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.db_path = os.path.join(cache_dir, "search_history.db")
        self._local = threading.local()
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Возвращает соединение с базой данных для текущего потока."""
        if not hasattr(self._local, "connection"):
            self._local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False
            )
        return self._local.connection
    
    def _init_database(self):
        """Инициализирует базу данных."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Создаем таблицу запросов
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                response_time REAL,
                success BOOLEAN,
                result_count INTEGER,
                cached BOOLEAN DEFAULT 0
            )
            """)
            
            # Таблица результатов
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id INTEGER NOT NULL,
                title TEXT,
                url TEXT NOT NULL,
                snippet TEXT,
                source TEXT,
                relevance_score REAL DEFAULT 1.0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (query_id) REFERENCES search_queries(id)
            )
            """)
            
            # Таблица статистики
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_stats (
                id INTEGER PRIMARY KEY,
                total_queries INTEGER DEFAULT 0,
                successful_queries INTEGER DEFAULT 0,
                failed_queries INTEGER DEFAULT 0,
                avg_processing_time REAL DEFAULT 0.0,
                last_update DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Инициализируем статистику, если таблица пуста
            cursor.execute("SELECT COUNT(*) FROM search_stats")
            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                INSERT INTO search_stats (total_queries, successful_queries, failed_queries, avg_processing_time)
                VALUES (0, 0, 0, 0.0)
                """)
            
            conn.commit()
        except Exception as e:
            logger.error(f"Ошибка инициализации базы данных: {e}", exc_info=True)
            raise
    
    def save_query(self, query: str, status: str, results: List[SearchResult], 
                   message: str, processing_time: float):
        """Сохраняет запрос и результаты в базу данных."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Сохраняем запрос
            success = 1 if status == "completed" and results else 0
            cursor.execute("""
            INSERT INTO search_queries (query, response_time, success, result_count, cached)
            VALUES (?, ?, ?, ?, 0)
            """, (query, processing_time, success, len(results)))
            
            query_id = cursor.lastrowid
            
            # Сохраняем результаты
            for result in results:
                cursor.execute("""
                INSERT INTO search_results (query_id, title, url, snippet, source, relevance_score)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    query_id,
                    result.title,
                    result.url,
                    result.snippet,
                    result.source,
                    result.relevance_score
                ))
            
            conn.commit()
            logger.debug(f"Результаты поиска сохранены в БД: {query}")
        except Exception as e:
            logger.error(f"Ошибка сохранения результатов поиска в БД: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Получает статистику из базы данных."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM search_stats ORDER BY id DESC LIMIT 1")
            stats = cursor.fetchone()
            
            if stats:
                return {
                    "total_queries": stats[1],
                    "successful_queries": stats[2],
                    "failed_queries": stats[3],
                    "avg_processing_time": stats[4],
                    "last_update": stats[5]
                }
            return {}
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {}
    
    def get_last_query(self) -> str:
        """Получает последний запрос."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT query FROM search_queries ORDER BY timestamp DESC LIMIT 1")
            result = cursor.fetchone()
            return result[0] if result else ""
        except Exception as e:
            logger.error(f"Ошибка получения последнего запроса: {e}")
            return ""
    
    def update_stats(self, stats: Dict[str, Any]):
        """Обновляет статистику в базе данных."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
            UPDATE search_stats
            SET total_queries = ?,
                successful_queries = ?,
                failed_queries = ?,
                avg_processing_time = ?,
                last_update = ?
            WHERE id = 1
            """, (
                stats["total_queries"],
                stats["successful_queries"],
                stats["failed_queries"],
                stats["avg_processing_time"],
                datetime.now().isoformat()
            ))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Ошибка обновления статистики: {e}")
    
    def get_recent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Возвращает недавние поисковые запросы."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
            SELECT query, timestamp, response_time, success, result_count
            FROM search_queries
            ORDER BY timestamp DESC
            LIMIT ?
            """, (limit,))
            
            queries = []
            for row in cursor.fetchall():
                queries.append({
                    "query": row[0],
                    "timestamp": row[1],
                    "response_time": row[2],
                    "success": bool(row[3]),
                    "result_count": row[4]
                })
            
            return queries
        except Exception as e:
            logger.error(f"Ошибка получения недавних запросов: {e}")
            return []
    
    def close(self):
        """Закрывает соединение с базой данных."""
        if hasattr(self._local, "connection"):
            try:
                self._local.connection.close()
                del self._local.connection
            except Exception as e:
                logger.error(f"Ошибка закрытия соединения с БД: {e}")
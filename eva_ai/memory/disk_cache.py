
"""
Дисковый кэш для токенов с SQLite метаданными.
"""

import os
import pickle
import sqlite3
import time
import zlib
from typing import Any, Optional, Tuple
import logging
import threading

# Опционально используем psutil для динамической подстройки лимитов I/O
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

logger = logging.getLogger(__name__)

class DiskCache:
    """Управление кэшем на жестком диске с SQLite метаданными."""
    
    def __init__(self, cache_dir: str, max_size_gb: float = 10.0,
                 write_mb_s: float = 40.0,
                 read_mb_s: float = 200.0,
                 burst_factor: float = 2.0,
                 resource_queue: Optional[object] = None):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.db_path = os.path.join(cache_dir, "cache_metadata.db")
        # DB lock for thread-safety
        self._db_lock = threading.RLock()
        # Open connection with check_same_thread=False to allow usage across threads (guarded by our lock)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        # Set pragmas: WAL for concurrency, normal synchronous, busy timeout
        with self._db_lock:
            cur = self.conn.cursor()
            try:
                cur.execute("PRAGMA journal_mode=WAL;")
                cur.execute("PRAGMA synchronous=NORMAL;")
                cur.execute("PRAGMA temp_store=MEMORY;")
                cur.execute("PRAGMA mmap_size=134217728;")  # 128MB
                cur.execute("PRAGMA busy_timeout=5000;")
            finally:
                cur.close()
        self._create_tables()
        
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.current_size = self._calculate_current_size()
        
        # Параметры троттлинга I/O
        self._write_rate_bps = max(1.0, write_mb_s) * 1024 * 1024
        self._read_rate_bps = max(1.0, read_mb_s) * 1024 * 1024
        self._burst = max(1.0, burst_factor)
        self._last_write_time = time.time()
        self._last_read_time = time.time()
        self._write_tokens = self._write_rate_bps * self._burst
        self._read_tokens = self._read_rate_bps * self._burst
        # Глобальная очередь ресурсов (опционально): системный троттлинг IO
        self._rq = resource_queue

    def _refill_tokens(self):
        """Пополняет токены для чтения/записи в соответствии с прошедшим временем."""
        now = time.time()
        # write tokens
        elapsed_w = max(0.0, now - self._last_write_time)
        self._write_tokens = min(self._write_rate_bps * self._burst,
                                 self._write_tokens + elapsed_w * self._write_rate_bps)
        self._last_write_time = now
        # read tokens
        elapsed_r = max(0.0, now - self._last_read_time)
        self._read_tokens = min(self._read_rate_bps * self._burst,
                                self._read_tokens + elapsed_r * self._read_rate_bps)
        self._last_read_time = now

    def _dynamic_backoff_factor(self) -> float:
        """При наличии psutil понижает лимит, если система активно использует диск.
        Возвращает множитель (0..1], которым масштабируется лимит.
        """
        try:
            if psutil is None:
                return 1.0
            # Грубая эвристика: если использование диска близко к 100%, замедляемся
            disk = psutil.disk_usage('/')
            busy = float(disk.percent) / 100.0
            # Также можно считать скорость записи ОС и замедляться при больших значениях
            # но здесь ограничимся занятостью диска
            if busy >= 0.95:
                return 0.25
            if busy >= 0.90:
                return 0.5
            if busy >= 0.80:
                return 0.75
            return 1.0
        except Exception:
            return 1.0

    def _throttle_write(self, size_bytes: int):
        """Ограничивает скорость записи, чтобы не перегружать SSD."""
        if size_bytes <= 0:
            return
        self._refill_tokens()
        factor = self._dynamic_backoff_factor()
        need = float(size_bytes)
        while need > 0:
            # доступные токены с учетом backoff
            avail = self._write_tokens * factor
            if avail <= 0:
                time.sleep(0.01)
                self._refill_tokens()
                continue
            take = min(avail, need)
            # списываем из базовых токенов, без factor, чтобы refill был корректен
            self._write_tokens -= take
            need -= take
            if need > 0:
                # спим, чтобы пополнить токены
                time.sleep(max(0.0, 0.01))
                self._refill_tokens()

    def _throttle_read(self, size_bytes: int):
        """Ограничивает скорость чтения, чтобы не мешать ОС."""
        if size_bytes <= 0:
            return
        self._refill_tokens()
        factor = self._dynamic_backoff_factor()
        need = float(size_bytes)
        while need > 0:
            avail = self._read_tokens * factor
            if avail <= 0:
                time.sleep(0.01)
                self._refill_tokens()
                continue
            take = min(avail, need)
            self._read_tokens -= take
            need -= take
            if need > 0:
                time.sleep(max(0.0, 0.01))
                self._refill_tokens()
    
    def _execute(self, sql: str, params: Tuple = (), commit: bool = False):
        """Безопасное выполнение SQL с блокировкой и повтором при SQLITE_BUSY."""
        attempts = 0
        last_exc = None
        while attempts < 5:
            with self._db_lock:
                try:
                    cur = self.conn.cursor()
                    try:
                        cur.execute(sql, params)
                        if commit:
                            self.conn.commit()
                        return cur
                    finally:
                        # Don't close here if we need the cursor's data; callers should fetch before close
                        pass
                except sqlite3.OperationalError as e:
                    last_exc = e
                    # Retry if database is locked or busy
                    if "locked" in str(e).lower() or "busy" in str(e).lower():
                        time.sleep(0.05 * (attempts + 1))
                        attempts += 1
                        continue
                    raise
                finally:
                    try:
                        cur.close()
                    except Exception:
                        pass
        if last_exc:
            raise last_exc

    def _create_tables(self):
        """Создает таблицы метаданных."""
        with self._db_lock:
            cursor = self.conn.cursor()
            try:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS metadata (
                        key TEXT PRIMARY KEY,
                        file_path TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        last_accessed REAL NOT NULL,
                        created_at REAL NOT NULL
                    )
                    """
                )
                self.conn.commit()
            finally:
                cursor.close()
    
    def _calculate_current_size(self) -> int:
        """Вычисляет текущий размер кэша."""
        with self._db_lock:
            cursor = self.conn.cursor()
            try:
                cursor.execute("SELECT SUM(size_bytes) FROM metadata")
                result = cursor.fetchone()[0]
                return result or 0
            finally:
                cursor.close()
    
    def _evict_old_entries(self):
        """Удаляет старые записи при превышении лимита."""
        while self.current_size > self.max_size_bytes * 0.9:
            with self._db_lock:
                cursor = self.conn.cursor()
                try:
                    cursor.execute(
                        """
                        SELECT key, file_path, size_bytes 
                        FROM metadata 
                        ORDER BY last_accessed ASC 
                        LIMIT 1
                        """
                    )
                    result = cursor.fetchone()
                finally:
                    cursor.close()
            if not result:
                break

            key, file_path, size = result
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.debug(f"Не удалось удалить файл при вытеснении: {file_path}: {e}")

            with self._db_lock:
                cursor = self.conn.cursor()
                try:
                    cursor.execute("DELETE FROM metadata WHERE key = ?", (key,))
                    self.conn.commit()
                finally:
                    cursor.close()
            self.current_size -= size
    
    def put(self, key: str, data: Any):
        """Сохраняет данные в кэш с компрессией."""
        try:
            file_path = os.path.join(self.cache_dir, f"{key}.pkl")
            
            # Сериализация и сжатие
            serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            compressed = zlib.compress(serialized)
            # Глобальный системный троттлинг IO (если доступен)
            try:
                if self._rq is not None:
                    self._rq.acquire_io(len(compressed))
            except Exception as e:
                logger.debug(f"Resource queue acquire_io failed during put: {e}")
            # Локальный троттлинг записи по размеру (safety net)
            self._throttle_write(len(compressed))

            with open(file_path, "wb") as f:
                f.write(compressed)
            
            size = os.path.getsize(file_path)

            with self._db_lock:
                cursor = self.conn.cursor()
                try:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO metadata 
                        (key, file_path, size_bytes, last_accessed, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (key, file_path, size, time.time(), time.time()),
                    )
                    self.conn.commit()
                finally:
                    cursor.close()
            self.current_size += size
            self._evict_old_entries()
            
        except Exception as e:
            logger.error(f"Ошибка сохранения в дисковый кэш: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Получает данные из кэша."""
        try:
            with self._db_lock:
                cursor = self.conn.cursor()
                try:
                    cursor.execute("SELECT file_path FROM metadata WHERE key = ?", (key,))
                    result = cursor.fetchone()
                finally:
                    cursor.close()
            
            if not result:
                return None
                
            file_path = result[0]
            if not os.path.exists(file_path):
                with self._db_lock:
                    cursor = self.conn.cursor()
                    try:
                        cursor.execute("DELETE FROM metadata WHERE key = ?", (key,))
                        self.conn.commit()
                    finally:
                        cursor.close()
                return None
            
            # Обновляем время доступа
            with self._db_lock:
                cursor = self.conn.cursor()
                try:
                    cursor.execute(
                        "UPDATE metadata SET last_accessed = ? WHERE key = ?",
                        (time.time(), key),
                    )
                    self.conn.commit()
                finally:
                    cursor.close()
            
            # Загружаем и распаковываем данные с троттлингом
            size = os.path.getsize(file_path)
            # Глобальный системный троттлинг IO (если доступен)
            try:
                if self._rq is not None:
                    self._rq.acquire_io(size)
            except Exception as e:
                logger.debug(f"Resource queue acquire_io failed during get: {e}")
            # Локальный троттлинг чтения (safety net)
            self._throttle_read(size)
            with open(file_path, "rb") as f:
                compressed = f.read()
                if not compressed or len(compressed) < 2:
                    logger.error(f"Пустые или поврежденные данные для ключа {key}")
                    return None
                try:
                    decompressed = zlib.decompress(compressed)
                except zlib.error as e:
                    logger.error(f"Ошибка декомпрессии для ключа {key}: {e}")
                    return None
                try:
                    return pickle.loads(decompressed, fix_imports=True, encoding='bytes', errors='strict')
                except (pickle.UnpicklingError, AttributeError, ValueError) as e:
                    logger.error(f"Ошибка десериализации для ключа {key}: {e}")
                    return None
                
        except Exception as e:
            logger.error(f"Ошибка загрузки из дискового кэша: {e}")
            return None
    
    def get_stats(self) -> dict:
        """Возвращает статистику кэша."""
        with self._db_lock:
            cursor = self.conn.cursor()
            try:
                cursor.execute("SELECT COUNT(*) FROM metadata")
                count = cursor.fetchone()[0]
            finally:
                cursor.close()
        
        return {
            "entries": count,
            "size_bytes": self.current_size,
            "size_mb": self.current_size / (1024**2),
            "max_size_gb": self.max_size_bytes / (1024**3)
        }
    
    def close(self):
        """Закрывает соединение с базой."""
        with self._db_lock:
            self.conn.close()
            
    def clear(self) -> None:
        """Полностью очищает кэш, удаляя все данные и метаданные."""
        with self._db_lock:
            # Закрываем текущее соединение с базой
            self.conn.close()
            
            # Удаляем все файлы в директории кэша
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        import shutil
                        shutil.rmtree(file_path)
                except Exception as e:
                    logger.error(f'Не удалось удалить {file_path}. Причина: {e}')
            
            # Сбрасываем размер кэша
            self.current_size = 0
            
            # Создаем новое соединение с базой
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
            # Восстанавливаем настройки базы
            cur = self.conn.cursor()
            try:
                cur.execute("PRAGMA journal_mode=WAL;")
                cur.execute("PRAGMA synchronous=NORMAL;")
                cur.execute("PRAGMA temp_store=MEMORY;")
                cur.execute("PRAGMA mmap_size=134217728;")  # 128MB
                cur.execute("PRAGMA busy_timeout=5000;")
            finally:
                cur.close()
            
            # Пересоздаем таблицы
            self._create_tables()

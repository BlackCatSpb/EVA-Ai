"""
ChainCache - LRU+TTL cache for response chains with hybrid RAM/SSD storage.
Adapted from metod.txt proposal for low-RAM systems.
"""
import os
import json
import time
import hashlib
import threading
import logging
from collections import OrderedDict
from typing import Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ChainCache:
    """
    Гибридный кэш цепочек ответа (LRU + TTL) с поддержкой SSD.
    
    Особенности:
    - LRU вытеснение в RAM
    - TTL (Time-To-Live) для автоматической очистки устаревших записей
    - SSD fallback для систем с ограниченной ОЗУ
    - Структурированное хранение JSON-объектов
    
    Адаптировано из metod.txt для гибридной архитектуры EVA.
    """
    
    def __init__(
        self,
        cache_dir: str = "./cache/chains",
        max_memory_entries: int = 1000,
        ttl_seconds: int = 3600,
        max_memory_mb: int = 100,
        enable_disk_cache: bool = True
    ):
        """
        Args:
            cache_dir: Директория для дискового кэша
            max_memory_entries: Максимум записей в RAM (LRU)
            ttl_seconds: Время жизни записи в секундах
            max_memory_mb: Максимум RAM для кэша в мегабайтах
            enable_disk_cache: Включить SSD fallback
        """
        self.cache_dir = Path(cache_dir)
        self.max_memory_entries = max_memory_entries
        self.ttl_seconds = ttl_seconds
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.enable_disk_cache = enable_disk_cache
        
        self._ram_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._current_memory_size = 0
        self._lock = threading.RLock()
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._disk_index_file = self.cache_dir / "chain_cache_index.json"
        self._disk_cache: Dict[str, Dict[str, Any]] = {}
        
        self._load_disk_index()
        
        logger.info(
            f"ChainCache инициализирован: RAM={max_memory_entries} entries, "
            f"TTL={ttl_seconds}s, disk_cache={enable_disk_cache}"
        )
    
    def _load_disk_index(self):
        """Загрузка индекса дискового кэша при старте."""
        if not self.enable_disk_cache:
            return
            
        try:
            if self._disk_index_file.exists():
                with open(self._disk_index_file, 'r', encoding='utf-8') as f:
                    self._disk_cache = json.load(f)
                logger.debug(f"Загружен дисковый индекс: {len(self._disk_cache)} записей")
        except Exception as e:
            logger.debug(f"Ошибка загрузки дискового индекса: {e}")
            self._disk_cache = {}
    
    def _save_disk_index(self):
        """Сохранение индекса дискового кэша."""
        if not self.enable_disk_cache:
            return
            
        try:
            with open(self._disk_index_file, 'w', encoding='utf-8') as f:
                json.dump(self._disk_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug(f"Ошибка сохранения дискового индекса: {e}")
    
    def _get_chain_key(self, query: str) -> str:
        """Генерация ключа для цепочки на основе хеша запроса."""
        return f"chain_{hashlib.md5(query.encode('utf-8')).hexdigest()}"
    
    def _estimate_size(self, entry: Dict[str, Any]) -> int:
        """Оценка размера записи в байтах."""
        try:
            return len(json.dumps(entry).encode('utf-8'))
        except Exception:
            return 1024  # fallback
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Получить цепочку из кэша.
        
        Поиск сначала в RAM, затем на SSD.
        При успехе запись перемещается в конец (LRU update).
        """
        key = self._get_chain_key(query)
        
        with self._lock:
            # 1. Поиск в RAM
            if key in self._ram_cache:
                entry = self._ram_cache[key]
                
                # Проверка TTL
                if time.time() - entry.get('_cached_at', 0) > self.ttl_seconds:
                    del self._ram_cache[key]
                    self._current_memory_size -= self._estimate_size(entry)
                    logger.debug(f"TTL expired for key: {key[:20]}...")
                    return None
                
                # LRU: перемещаем в конец
                self._ram_cache.move_to_end(key)
                logger.debug(f"RAM cache hit: {key[:20]}...")
                return entry
            
            # 2. Поиск на SSD
            if self.enable_disk_cache and key in self._disk_cache:
                entry = self._disk_cache[key]
                
                # Проверка TTL
                if time.time() - entry.get('_cached_at', 0) > self.ttl_seconds:
                    del self._disk_cache[key]
                    self._save_disk_index()
                    logger.debug(f"TTL expired on disk: {key[:20]}...")
                    return None
                
                # Загружаем в RAM (с проверкой места)
                self.set(query, entry)
                
                # Удаляем из disk cache
                del self._disk_cache[key]
                self._save_disk_index()
                
                logger.debug(f"Disk cache hit: {key[:20]}...")
                return entry
            
            return None
    
    def set(self, query: str, chain_data: Dict[str, Any]) -> None:
        """
        Сохранить цепочку в кэш.
        
        LRU + TTL политика:
        - Если RAM заполнен - вытесняем oldest
        - При достижении лимита памяти - перемещаем в SSD
        """
        key = self._get_chain_key(query)
        
        # Добавляем метаданные
        chain_data['_cached_at'] = time.time()
        chain_data['_query_hash'] = key
        
        entry_size = self._estimate_size(chain_data)
        
        with self._lock:
            # Удаляем старую запись если есть
            if key in self._ram_cache:
                old_entry = self._ram_cache[key]
                self._current_memory_size -= self._estimate_size(old_entry)
                del self._ram_cache[key]
            
            # Проверяем лимит RAM
            while (
                len(self._ram_cache) >= self.max_memory_entries or
                self._current_memory_size + entry_size > self.max_memory_bytes
            ):
                if not self._ram_cache:
                    break
                    
                # LRU: вытесняем oldest (первый)
                oldest_key, oldest_entry = self._ram_cache.popitem(last=False)
                self._current_memory_size -= self._estimate_size(oldest_entry)
                
                # Пробуем переместить на SSD
                if self.enable_disk_cache:
                    oldest_entry['_cached_at'] = time.time()  # обновляем время
                    self._disk_cache[oldest_key] = oldest_entry
                    logger.debug(f"Evicted to disk: {oldest_key[:20]}...")
            
            # Добавляем в RAM
            self._ram_cache[key] = chain_data
            self._current_memory_size += entry_size
            
            # Сохраняем disk index
            if self.enable_disk_cache:
                self._save_disk_index()
    
    def _evict_expired(self) -> int:
        """Удалить все просроченные записи (TTL cleanup)."""
        count = 0
        current_time = time.time()
        
        with self._lock:
            # RAM
            expired_keys = [
                k for k, v in self._ram_cache.items()
                if current_time - v.get('_cached_at', 0) > self.ttl_seconds
            ]
            for key in expired_keys:
                entry = self._ram_cache.pop(key, None)
                if entry:
                    self._current_memory_size -= self._estimate_size(entry)
                    count += 1
            
            # Disk
            if self.enable_disk_cache:
                expired_disk_keys = [
                    k for k, v in self._disk_cache.items()
                    if current_time - v.get('_cached_at', 0) > self.ttl_seconds
                ]
                for key in expired_disk_keys:
                    self._disk_cache.pop(key, None)
                    count += 1
                
                if expired_disk_keys:
                    self._save_disk_index()
        
        if count > 0:
            logger.info(f"TTL cleanup: removed {count} expired entries")
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику кэша."""
        with self._lock:
            total_entries = len(self._ram_cache) + len(self._disk_cache)
            
            return {
                'ram_entries': len(self._ram_cache),
                'disk_entries': len(self._disk_cache) if self.enable_disk_cache else 0,
                'total_entries': total_entries,
                'ram_memory_mb': self._current_memory_size / (1024 * 1024),
                'max_ram_mb': self.max_memory_bytes / (1024 * 1024),
                'ttl_seconds': self.ttl_seconds,
                'hit_rate': getattr(self, '_hit_count', 0) / max(getattr(self, '_total_gets', 1), 1)
            }
    
    def clear(self):
        """Очистить весь кэш."""
        with self._lock:
            self._ram_cache.clear()
            self._disk_cache.clear()
            self._current_memory_size = 0
            
            if self.enable_disk_cache:
                self._save_disk_index()
        
        logger.info("ChainCache cleared")
    
    def save_chain(
        self,
        query: str,
        short_answer: str,
        conclusion: str,
        full_answer: str,
        concept_ids: list = None,
        contradiction_ids: list = None
    ) -> str:
        """
        Сохранить цепочку ответа.
        
        Args:
            query: Запрос пользователя
            short_answer: Краткий ответ
            conclusion: Вывод
            full_answer: Полный ответ
            concept_ids: IDs использованных концептов
            contradiction_ids: IDs использованных противоречий
            
        Returns:
            chain_id
        """
        chain_id = self._get_chain_key(query)
        
        chain_data = {
            'query': query,
            'short_answer': short_answer,
            'conclusion': conclusion,
            'full_answer': full_answer,
            'concept_ids': concept_ids or [],
            'contradiction_ids': contradiction_ids or [],
            'timestamp': time.time()
        }
        
        self.set(query, chain_data)
        logger.debug(f"Saved chain: {chain_id[:20]}...")
        
        return chain_id
    
    def get_chain(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Получить цепочку по запросу.
        
        Returns:
            chain_data or None if не найдено/просрочено
        """
        # TTL cleanup при каждом get (概率 10%)
        import random
        if random.random() < 0.1:
            self._evict_expired()
        
        return self.get(query)


def create_chain_cache(
    cache_dir: str = None,
    max_entries: int = 1000,
    ttl_seconds: int = 3600,
    max_memory_mb: int = 100
) -> ChainCache:
    """Factory function для создания ChainCache."""
    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), 'cache', 'chains')
    
    return ChainCache(
        cache_dir=cache_dir,
        max_memory_entries=max_entries,
        ttl_seconds=ttl_seconds,
        max_memory_mb=max_memory_mb
    )

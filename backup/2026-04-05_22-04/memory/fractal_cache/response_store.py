"""
ResponseStore - хранение ответов на диске с Redis-совместимым интерфейсом.
Использует локальные файлы с индексом для быстрого доступа.
"""
import os
import json
import time
import logging
import threading
from typing import Dict, Any, Optional, List
from collections import OrderedDict

logger = logging.getLogger("eva.memory.fractal_cache.store")


class ResponseStore:
    """
    Хранилище ответов на диске.
    
    Структура:
    - index.json: Индекс ключей и метаданных
    - data/: Директория с файлами ответов (по 1 на запись)
    """
    
    def __init__(self, cache_dir: str, max_entries: int = 500000):
        self.cache_dir = cache_dir
        self.max_entries = max_entries
        
        # Директории
        self.data_dir = os.path.join(cache_dir, "data")
        self.index_file = os.path.join(cache_dir, "store_index.json")
        
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Индекс в памяти
        self._index: OrderedDict[str, Dict] = OrderedDict()
        self._lock = threading.RLock()
        
        # Загружаем индекс
        self._load_index()
        
        logger.info(f"ResponseStore инициализирован: {len(self._index)} записей")
    
    def _load_index(self):
        """Загружает индекс из файла."""
        try:
            if os.path.exists(self.index_file):
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key, meta in data.items():
                        self._index[key] = meta
                logger.info(f"Загружен индекс: {len(self._index)} записей")
        except Exception as e:
            logger.warning(f"Ошибка загрузки индекса: {e}")
            self._index = OrderedDict()
    
    def _save_index(self):
        """Сохраняет индекс в файл."""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(dict(self._index), f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Ошибка сохранения индекса: {e}")
    
    def save(self, key: str, entry: Dict[str, Any]):
        """
        Сохраняет запись.
        
        Args:
            key: Ключ записи
            entry: Данные записи
        """
        with self._lock:
            # Сохраняем данные в файл
            data_file = os.path.join(self.data_dir, f"{key[:16]}.json")
            
            # Убираем эмбеддинг из файла (слишком большой)
            save_data = {k: v for k, v in entry.items() if k != "embedding"}
            save_data["_embedding_stored"] = True
            
            try:
                with open(data_file, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Ошибка сохранения данных {key}: {e}")
                return
            
            # Обновляем индекс
            self._index[key] = {
                "file": f"{key[:16]}.json",
                "timestamp": entry.get("timestamp", time.time()),
                "query_len": len(entry.get("query", "")),
                "response_len": len(entry.get("response", ""))
            }
            
            # Сохраняем индекс
            self._save_index()
    
    def load(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Загружает запись по ключу.
        
        Args:
            key: Ключ записи
            
        Returns:
            Optional[Dict]: Данные записи или None
        """
        with self._lock:
            if key not in self._index:
                return None
            
            meta = self._index[key]
            data_file = os.path.join(self.data_dir, meta["file"])
            
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Ошибка загрузки {key}: {e}")
                return None
    
    def delete(self, key: str) -> bool:
        """
        Удаляет запись.
        
        Args:
            key: Ключ записи
            
        Returns:
            bool: True если удалено
        """
        with self._lock:
            if key not in self._index:
                return False
            
            # Удаляем файл
            meta = self._index[key]
            data_file = os.path.join(self.data_dir, meta["file"])
            
            try:
                if os.path.exists(data_file):
                    os.remove(data_file)
            except Exception as e:
                logger.warning(f"Ошибка удаления файла {key}: {e}")
            
            # Удаляем из индекса
            del self._index[key]
            self._save_index()
            
            return True
    
    def get_recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Получает последние записи.
        
        Args:
            limit: Максимум записей
            
        Returns:
            List[Dict]: Список записей
        """
        with self._lock:
            results = []
            
            # Сортируем по времени
            sorted_keys = sorted(
                self._index.keys(),
                key=lambda k: self._index[k].get("timestamp", 0),
                reverse=True
            )
            
            for key in sorted_keys[:limit]:
                entry = self.load(key)
                if entry:
                    entry["key"] = key
                    results.append(entry)
            
            return results
    
    def clear_all(self):
        """Очищает все данные."""
        with self._lock:
            # Удаляем все файлы данных
            for filename in os.listdir(self.data_dir):
                filepath = os.path.join(self.data_dir, filename)
                try:
                    os.remove(filepath)
                except Exception:
                    pass
            
            # Очищаем индекс
            self._index.clear()
            self._save_index()
            
            logger.info("Все данные очищены")
    
    @property
    def size(self) -> int:
        """Возвращает количество записей."""
        return len(self._index)

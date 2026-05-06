# hybrid_token_cache.py
"""Гибридный кэш токенов для ЕВА."""

import os
import sys
import time
import json
import logging
import threading
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

logger = logging.getLogger("eva_ai.token_cache")

class HybridTokenCache:
    """Гибридный кэш токенов с поддержкой VRAM, RAM и диска."""

    def __init__(self, brain, max_memory_tokens: int, disk_cache_dir: str,
                 vram_ratio: float = 0.7, vram_threshold: float = 0.2,
                 ram_threshold: float = 0.15, eviction_policy: str = "hybrid"):
        self.brain = brain
        self.max_memory_tokens = max_memory_tokens
        self.disk_cache_dir = disk_cache_dir
        self.vram_ratio = vram_ratio
        self.vram_threshold = vram_threshold
        self.ram_threshold = ram_threshold
        self.eviction_policy = eviction_policy
        self.vram_cache = {}
        self.ram_cache = {}
        self.disk_cache = {}
        self.access_times = {}
        self.token_counts = {}
        self.lock = threading.RLock()
        self.memory_monitor_thread = None
        self.stop_monitoring = threading.Event()
        self.logger = logging.getLogger("eva_ai.token_cache")

        # Создаем директорию кэша, если не существует
        os.makedirs(disk_cache_dir, exist_ok=True)
        self.logger.info(f"Гибридный кэш токенов инициализирован: max_tokens={max_memory_tokens}")

    def get(self, key: str) -> Optional[Any]:
        """Получает значение из кэша."""
        with self.lock:
            # Сначала проверяем VRAM
            if key in self.vram_cache:
                self._update_access_time(key)
                self.logger.debug(f"Кэш hit в VRAM для ключа: {key}")
                return self.vram_cache[key]

            # Затем проверяем RAM
            if key in self.ram_cache:
                # Перемещаем в VRAM, если есть место
                if self._can_move_to_vram(key):
                    self._move_to_vram(key)
                    self._update_access_time(key)
                    self.logger.debug(f"Кэш hit в RAM, перемещено в VRAM для ключа: {key}")
                    return self.vram_cache[key]

                self._update_access_time(key)
                self.logger.debug(f"Кэш hit в RAM для ключа: {key}")
                return self.ram_cache[key]

            # Наконец, проверяем диск
            if key in self.disk_cache:
                # Загружаем в RAM или VRAM
                value = self._load_from_disk(key)
                if value is not None:
                    self.set(key, value)
                    self.logger.debug(f"Кэш hit на диске, загружено в кэш для ключа: {key}")
                    return value

            self.logger.debug(f"Кэш miss для ключа: {key}")
            return None

    def set(self, key: str, value: Any, token_count: Optional[int] = None):
        """Устанавливает значение в кэш."""
        with self.lock:
            # Определяем количество токенов, если не указано
            if token_count is None:
                token_count = self._estimate_token_count(value)

            # Сохраняем количество токенов
            self.token_counts[key] = token_count

            # Проверяем, можем ли добавить в VRAM
            if self._can_add_to_vram(token_count):
                self.vram_cache[key] = value
                self._update_access_time(key)
                self.logger.debug(f"Добавлено в VRAM кэш: {key}")
                return

            # Проверяем, можем ли добавить в RAM
            if self._can_add_to_ram(token_count):
                self.ram_cache[key] = value
                self._update_access_time(key)
                self.logger.debug(f"Добавлено в RAM кэш: {key}")
                return

            # Сохраняем на диск как последнее средство
            self._save_to_disk(key, value)
            self.logger.debug(f"Добавлено в дисковый кэш: {key}")

    def _can_add_to_vram(self, token_count: int) -> bool:
        """Проверяет, можно ли добавить элемент в VRAM кэш."""
        current_vram_tokens = sum(self.token_counts.get(k, 0) for k in self.vram_cache)
        return current_vram_tokens + token_count <= self.max_memory_tokens * self.vram_ratio

    def _can_add_to_ram(self, token_count: int) -> bool:
        """Проверяет, можно ли добавить элемент в RAM кэш."""
        current_ram_tokens = sum(self.token_counts.get(k, 0) for k in self.ram_cache)
        max_ram_tokens = self.max_memory_tokens * (1 - self.vram_ratio)
        return current_ram_tokens + token_count <= max_ram_tokens

    def _can_move_to_vram(self, key: str) -> bool:
        """Проверяет, можно ли переместить элемент из RAM в VRAM."""
        token_count = self.token_counts.get(key, 0)
        current_vram_tokens = sum(self.token_counts.get(k, 0) for k in self.vram_cache)
        return current_vram_tokens + token_count <= self.max_memory_tokens * self.vram_ratio

    def _move_to_vram(self, key: str):
        """Перемещает элемент из RAM в VRAM."""
        if key in self.ram_cache:
            value = self.ram_cache.pop(key)
            self.vram_cache[key] = value
            self._update_access_time(key)

    def _update_access_time(self, key: str):
        """Обновляет время доступа к элементу."""
        self.access_times[key] = time.time()

    def _estimate_token_count(self, value: Any) -> int:
        """Оценивает количество токенов в значении."""
        # Простая оценка - количество символов деленное на 4
        if isinstance(value, str):
            return max(1, len(value) // 4)
        elif isinstance(value, list):
            return len(value)
        elif isinstance(value, np.ndarray):
            return value.size
        return 1

    def _save_to_disk(self, key: str, value: Any):
        """Сохраняет значение на диск."""
        try:
            file_path = os.path.join(self.disk_cache_dir, f"{key}.cache")
            with open(file_path, 'wb') as f:
                # Здесь должна быть сериализация значения
                # Для упрощения используем простую строковую сериализацию
                if isinstance(value, str):
                    f.write(value.encode('utf-8'))
                else:
                    f.write(str(value).encode('utf-8'))
            self.disk_cache[key] = file_path
        except Exception as e:
            self.logger.error(f"Ошибка сохранения в дисковый кэш: {e}")

    def _load_from_disk(self, key: str) -> Optional[Any]:
        """Загружает значение с диска."""
        try:
            if key not in self.disk_cache:
                return None

            file_path = self.disk_cache[key]
            if not os.path.exists(file_path):
                del self.disk_cache[key]
                return None

            with open(file_path, 'rb') as f:
                data = f.read()
                # Здесь должна быть десериализация
                return data.decode('utf-8')
        except Exception as e:
            self.logger.error(f"Ошибка загрузки из дискового кэша: {e}")
            return None

    def clear(self):
        """Очищает весь кэш."""
        with self.lock:
            self.vram_cache.clear()
            self.ram_cache.clear()
            # Дисковый кэш не очищаем, чтобы сохранить данные между сессиями
            self.access_times.clear()
            self.token_counts.clear()
            self.logger.info("Гибридный кэш токенов очищен")

    def clear_inactive(self, inactive_threshold: float = 300.0):
        """Очищает неактивные элементы кэша."""
        current_time = time.time()
        with self.lock:
            # Сначала проверяем VRAM
            vram_keys = list(self.vram_cache.keys())
            for key in vram_keys:
                if current_time - self.access_times.get(key, 0) > inactive_threshold:
                    # Перемещаем в RAM, если возможно
                    if self._can_add_to_ram(self.token_counts.get(key, 0)):
                        self.ram_cache[key] = self.vram_cache.pop(key)
                    else:
                        # Иначе удаляем полностью
                        del self.vram_cache[key]
                        del self.access_times[key]
                        del self.token_counts[key]

            # Затем проверяем RAM
            ram_keys = list(self.ram_cache.keys())
            for key in ram_keys:
                if current_time - self.access_times.get(key, 0) > inactive_threshold:
                    # Сохраняем на диск
                    self._save_to_disk(key, self.ram_cache[key])
                    del self.ram_cache[key]
                    del self.access_times[key]
                    del self.token_counts[key]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Возвращает статистику кэша."""
        with self.lock:
            return {
                "vram_items": len(self.vram_cache),
                "ram_items": len(self.ram_cache),
                "disk_items": len(self.disk_cache),
                "vram_usage": sum(self.token_counts.get(k, 0) for k in self.vram_cache),
                "ram_usage": sum(self.token_counts.get(k, 0) for k in self.ram_cache),
                "total_usage": sum(self.token_counts.values()),
                "max_capacity": self.max_memory_tokens
            }

    def start_memory_monitor(self, interval: float = 10.0):
        """Запускает фоновый мониторинг памяти."""
        if self.memory_monitor_thread and self.memory_monitor_thread.is_alive():
            return  # Уже запущен

        self.stop_monitoring.clear()

        def monitor_loop():
            while not self.stop_monitoring.is_set():
                try:
                    self._monitor_memory_usage()
                    self.stop_monitoring.wait(interval)
                except Exception as e:
                    self.logger.error(f"Ошибка в мониторе памяти: {e}")
                    time.sleep(1)

        self.memory_monitor_thread = threading.Thread(
            target=monitor_loop,
            name="MemoryMonitor",
            daemon=True
        )
        self.memory_monitor_thread.start()
        self.logger.info("Запущен фоновый мониторинг памяти гибридного кэша")

    def stop_memory_monitoring(self):
        """Останавливает фоновый мониторинг памяти."""
        self.stop_monitoring.set()
        if self.memory_monitor_thread and self.memory_monitor_thread.is_alive():
            self.memory_monitor_thread.join(timeout=2.0)
        self.logger.info("Фоновый мониторинг памяти гибридного кэша остановлен")

    def _monitor_memory_usage(self):
        """Мониторит использование памяти и при необходимости очищает кэш."""
        with self.lock:
            stats = self.get_cache_stats()
            total_usage = stats["total_usage"]

            self.logger.debug(
                f"Использование кэша: {total_usage}/{self.max_memory_tokens} токенов "
                f"({(total_usage / self.max_memory_tokens * 100):.1f}%)"
            )

            # Если превышен лимит, запускаем очистку
            if total_usage > self.max_memory_tokens * 0.9:
                self.logger.info("Превышен лимит кэша, запуск очистки...")
                self._evict_items()

    def _evict_items(self):
        """Выполняет вытеснение элементов из кэша согласно политике."""
        if self.eviction_policy == "hybrid":
            self._hybrid_eviction()
        elif self.eviction_policy == "lru":
            self._lru_eviction()
        elif self.eviction_policy == "lfu":
            self._lfu_eviction()

    def _hybrid_eviction(self):
        """Гибридная политика вытеснения: LRU для VRAM, LFU для RAM."""
        self._lru_eviction_vram()
        self._lfu_eviction_ram()

    def _lru_eviction_vram(self, target_usage: Optional[float] = None):
        """LRU вытеснение из VRAM."""
        if target_usage is None:
            target_usage = self.max_memory_tokens * self.vram_ratio * self.vram_threshold

        current_usage = sum(self.token_counts.get(k, 0) for k in self.vram_cache)
        if current_usage <= target_usage:
            return

        # Сортируем по времени доступа (старые первыми)
        sorted_items = sorted(
            self.vram_cache.items(),
            key=lambda x: self.access_times.get(x[0], 0)
        )

        # Удаляем самые старые элементы
        for key, _ in sorted_items:
            if current_usage <= target_usage:
                break

            token_count = self.token_counts.get(key, 0)
            del self.vram_cache[key]
            del self.access_times[key]
            del self.token_counts[key]
            current_usage -= token_count

    def _lfu_eviction_ram(self, target_usage: Optional[float] = None):
        """LFU вытеснение из RAM."""
        # Для упрощения используем LRU вместо LFU
        self._lru_eviction_ram(target_usage)

    def _lru_eviction_ram(self, target_usage: Optional[float] = None):
        """LRU вытеснение из RAM."""
        if target_usage is None:
            max_ram_usage = self.max_memory_tokens * (1 - self.vram_ratio)
            target_usage = max_ram_usage * self.ram_threshold

        current_usage = sum(self.token_counts.get(k, 0) for k in self.ram_cache)
        if current_usage <= target_usage:
            return

        # Сортируем по времени доступа (старые первыми)
        sorted_items = sorted(
            self.ram_cache.items(),
            key=lambda x: self.access_times.get(x[0], 0)
        )

        # Удаляем самые старые элементы
        for key, _ in sorted_items:
            if current_usage <= target_usage:
                break

            token_count = self.token_counts.get(key, 0)
            # Сохраняем на диск перед удалением
            self._save_to_disk(key, self.ram_cache[key])
            del self.ram_cache[key]
            del self.access_times[key]
            del self.token_counts[key]
            current_usage -= token_count

# global_resource_queue.py
"""Глобальная очередь ресурсов для ЕВА."""

import os
import sys
import time
import json
import logging
import threading
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger("eva_ai.resource_queue")

class GlobalResourceQueue:
    """Система управления глобальными ресурсами (CPU, память, IO)."""

    def __init__(self, brain, max_memory_bytes: int, cpu_tokens: int,
                 io_rate_bps: float, io_burst_factor: float = 1.5):
        self.brain = brain
        self.max_memory_bytes = max_memory_bytes
        self.cpu_tokens = cpu_tokens
        self.io_rate_bps = io_rate_bps
        self.io_burst_factor = io_burst_factor
        self.current_memory_usage = 0
        self.current_cpu_usage = 0
        self.io_budget = io_rate_bps * io_burst_factor
        self.io_last_update = time.time()
        self.lock = threading.RLock()
        self.memory_condition = threading.Condition(self.lock)
        self.cpu_condition = threading.Condition(self.lock)
        self.io_condition = threading.Condition(self.lock)
        self.logger = logging.getLogger("eva_ai.resource_queue")

        # Запускаем фоновый монитор
        self.stop_event = threading.Event()
        self.monitor_thread = threading.Thread(
            target=self._monitor_resources,
            name="ResourceMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("ResourceQueue инициализирована")

    def acquire_memory(self, nbytes: int, timeout: Optional[float] = None) -> bool:
        """Запрашивает выделение памяти."""
        with self.memory_condition:
            start_time = time.time()

            while self.current_memory_usage + nbytes > self.max_memory_bytes:
                if timeout is not None and time.time() - start_time > timeout:
                    return False
                self.memory_condition.wait(timeout)

            self.current_memory_usage += nbytes
            return True

    def release_memory(self, nbytes: int) -> None:
        """Освобождает ранее выделенную память."""
        with self.memory_condition:
            self.current_memory_usage = max(0, self.current_memory_usage - nbytes)
            self.memory_condition.notify_all()

    def acquire_cpu(self, n: int = 1, timeout: Optional[float] = None) -> bool:
        """Запрашивает CPU токены."""
        with self.cpu_condition:
            start_time = time.time()

            while self.current_cpu_usage + n > self.cpu_tokens:
                if timeout is not None and time.time() - start_time > timeout:
                    return False
                self.cpu_condition.wait(timeout)

            self.current_cpu_usage += n
            return True

    def release_cpu(self, n: int = 1) -> None:
        """Освобождает CPU токены."""
        with self.cpu_condition:
            self.current_cpu_usage = max(0, self.current_cpu_usage - n)
            self.cpu_condition.notify_all()

    def acquire_io(self, nbytes: int, timeout: Optional[float] = None) -> bool:
        """Запрашивает IO ресурсы."""
        with self.io_condition:
            current_time = time.time()
            elapsed = current_time - self.io_last_update

            # Обновляем бюджет IO
            self.io_budget = min(
                self.io_rate_bps * self.io_burst_factor,
                self.io_budget + (self.io_rate_bps * elapsed)
            )
            self.io_last_update = current_time

            # Проверяем, достаточно ли бюджета
            if self.io_budget >= nbytes:
                self.io_budget -= nbytes
                return True

            # Если не хватает, ждем
            if timeout is None:
                timeout = 1.0  # По умолчанию ждем 1 секунду

            start_time = time.time()
            while self.io_budget < nbytes:
                if time.time() - start_time > timeout:
                    return False

                # Обновляем бюджет во время ожидания
                current_time = time.time()
                elapsed = current_time - self.io_last_update
                self.io_budget = min(
                    self.io_rate_bps * self.io_burst_factor,
                    self.io_budget + (self.io_rate_bps * elapsed)
                )
                self.io_last_update = current_time

                self.io_condition.wait(0.1)  # Проверяем каждые 100ms

            self.io_budget -= nbytes
            return True

    def release_io(self, nbytes: int) -> None:
        """Освобождает IO ресурсы."""
        with self.lock:
            self.io_budget = min(self.io_rate_bps * self.io_burst_factor, 
                                self.io_budget + nbytes)
            self.io_condition.notify_all()

    def get_resource_usage(self) -> Dict[str, float]:
        """Возвращает текущее использование ресурсов."""
        with self.lock:
            return {
                "cpu_percent": (self.current_cpu_usage / max(1, self.cpu_tokens)) * 100,
                "memory_percent": (self.current_memory_usage / max(1, self.max_memory_bytes)) * 100,
                "io_utilization": ((self.io_rate_bps * self.io_burst_factor - self.io_budget) /
                                 (self.io_rate_bps * self.io_burst_factor)) * 100
            }

    def update_memory_limit(self, new_limit_bytes: int):
        """Динамически обновляет лимит памяти."""
        with self.lock:
            self.logger.info(f"Обновление лимита памяти с {self.max_memory_bytes} до {new_limit_bytes}")
            self.max_memory_bytes = new_limit_bytes
            # Уведомляем потоки, ожидающие освобождения памяти
            self.memory_condition.notify_all()

    def _monitor_resources(self):
        """Фоновый мониторинг ресурсов."""
        while not self.stop_event.is_set():
            try:
                with self.lock:
                    # Логируем использование ресурсов каждые 5 секунд
                    usage = self.get_resource_usage()
                    self.logger.debug(
                        f"Ресурсы: CPU={usage['cpu_percent']:.1f}%, "
                        f"Memory={usage['memory_percent']:.1f}%, "
                        f"IO={usage['io_utilization']:.1f}%"
                    )

                # Ждем следующего цикла
                self.stop_event.wait(15.0) # Увеличим интервал, т.к. это только для логирования
            except Exception as e:
                self.logger.error(f"Ошибка в мониторе ресурсов: {e}")
                time.sleep(5)

    def stop(self):
        """Останавливает фоновый мониторинг."""
        self.stop_event.set()
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        self.logger.info("ResourceQueue остановлена")

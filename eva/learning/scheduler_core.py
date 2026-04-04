"""Core module for EVA learning scheduler - base classes, lifecycle, and worker loop."""

import sys
import os
import time
import logging
if os.environ.get("CFX_DEBUG_IMPORTS"):
    print("Проверка импорта LearningScheduler:")
    print(f"Текущая директория: {os.getcwd()}")
    print(f"Пути поиска модулей: {sys.path}")
logger = logging.getLogger(__name__)
import threading
import json
import heapq
from typing import Dict, List, Optional, Any, Set, Union, Callable, Deque, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("eva.learning_scheduler")


@dataclass
class ResourceAllocation:
    """Управление ресурсами для выполнения задач."""
    max_concurrent: int = 4
    current_concurrent: int = 0
    resource_lock: threading.Lock = field(default_factory=threading.Lock)
    task_slots: Dict[str, float] = field(default_factory=dict)

    def acquire_slot(self, task_id: str) -> bool:
        """Пытается занять слот для выполнения задачи."""
        with self.resource_lock:
            if self.current_concurrent < self.max_concurrent:
                self.current_concurrent += 1
                self.task_slots[task_id] = time.time()
                return True
            return False

    def release_slot(self, task_id: str) -> None:
        """Освобождает слот после выполнения задачи."""
        with self.resource_lock:
            if task_id in self.task_slots:
                del self.task_slots[task_id]
                self.current_concurrent = max(0, self.current_concurrent - 1)

    def get_slot_usage(self) -> float:
        """Возвращает текущую загрузку ресурсов (0.0-1.0)."""
        with self.resource_lock:
            return self.current_concurrent / max(1, self.max_concurrent)

    def get_active_tasks(self) -> List[str]:
        """Возвращает список активных задач."""
        with self.resource_lock:
            return list(self.task_slots.keys())


@dataclass
class LearningTask:
    """Представляет задачу обучения."""
    task_id: str
    task_type: str
    concept: str
    priority: float
    scheduled_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    dependents: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует задачу в словарь."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "concept": self.concept,
            "priority": self.priority,
            "scheduled_time": self.scheduled_time,
            "metadata": self.metadata,
            "dependencies": self.dependencies,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "result": self.result,
            "error": self.error,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "dependents": self.dependents
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningTask':
        """Создает задачу из словаря."""
        return cls(
            task_id=data.get("task_id", ""),
            task_type=data.get("task_type", "unknown"),
            concept=data.get("concept", ""),
            priority=data.get("priority", 0),
            scheduled_time=data.get("scheduled_time", 0),
            metadata=data.get("metadata", {}),
            dependencies=data.get("dependencies", []),
            status=data.get("status", "pending"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            result=data.get("result"),
            error=data.get("error"),
            retries=data.get("retries", 0),
            max_retries=data.get("max_retries", 3),
            dependents=data.get("dependents", [])
        )

    def __lt__(self, other: 'LearningTask') -> bool:
        """Определяет порядок приоритета в очереди."""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.scheduled_time < other.scheduled_time

    def get_duration(self) -> Optional[float]:
        """Возвращает продолжительность выполнения задачи."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None

    def is_overdue(self) -> bool:
        """Проверяет, просрочена ли задача."""
        return time.time() > self.scheduled_time


class LearningSchedulerCore:
    """Base class for LearningScheduler - initialization, lifecycle, persistence, worker loop."""

    def __init__(self, brain=None, cache_dir: Optional[str] = None):
        """Инициализирует планировщик задач обучения."""
        self.brain = brain
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "eva_learning_scheduler_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.start_time = time.time()

        self.task_registry: Dict[str, LearningTask] = {}
        self.task_queue: List[LearningTask] = []
        heapq.heapify(self.task_queue)

        self.lock = threading.Lock()

        self.worker_threads: List[threading.Thread] = []
        self.stop_event = threading.Event()
        self.running = False

        self.task_timeout = 300
        self.max_concurrent_tasks = 8

        self.resource_allocation = ResourceAllocation(max_concurrent=self.max_concurrent_tasks)

        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "pending_tasks": 0,
            "in_progress_tasks": 0,
            "last_update": time.time()
        }

        self.learning_rate = 0.1
        self.task_retry_delay = 60

        self._load_tasks()
        self.start()

        logger.info("Планировщик задач обучения инициализирован")

    def _load_tasks(self):
        """Загружает сохраненные задачи из кэша."""
        try:
            tasks_file = os.path.join(self.cache_dir, "tasks.json")
            if os.path.exists(tasks_file):
                with open(tasks_file, 'r', encoding='utf-8') as f:
                    tasks_data = json.load(f)

                for task_data in tasks_data:
                    task = LearningTask.from_dict(task_data)
                    self.task_registry[task.task_id] = task

                    if task.status == "pending":
                        heapq.heappush(self.task_queue, task)

                self._update_stats()

                logger.info(f"Загружено {len(self.task_registry)} задач обучения")
        except Exception as e:
            logger.error(f"Ошибка загрузки задач: {e}")

    def _save_tasks(self):
        """Сохраняет задачи в кэш."""
        try:
            tasks_file = os.path.join(self.cache_dir, "tasks.json")
            tasks_data = [task.to_dict() for task in self.task_registry.values()]
            with open(tasks_file, 'w', encoding='utf-8') as f:
                json.dump(tasks_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения задач: {e}")

    def _update_stats(self):
        """Обновляет статистику планировщика."""
        with self.lock:
            self.stats["pending_tasks"] = sum(1 for task in self.task_registry.values() if task.status == "pending")
            self.stats["in_progress_tasks"] = sum(1 for task in self.task_registry.values() if task.status == "in_progress")
            self.stats["completed_tasks"] = sum(1 for task in self.task_registry.values() if task.status == "completed")
            self.stats["failed_tasks"] = sum(1 for task in self.task_registry.values() if task.status == "failed")
            self.stats["total_tasks"] = len(self.task_registry)
            self.stats["last_update"] = time.time()

    def _check_system_health(self):
        """Проверяет здоровье системы планировщика."""
        stats = self.get_scheduler_statistics()

        if stats["pending_tasks"] > 50:
            logger.warning(f"Высокая загрузка планировщика: {stats['pending_tasks']} задач в очереди")

        if stats["failed_tasks"] > 10:
            logger.warning(f"Высокое количество неудачных задач: {stats['failed_tasks']}")

        if stats["avg_completion_time"] > 300:
            logger.warning(f"Высокое среднее время выполнения задач: {stats['avg_completion_time']:.2f} секунд")

    def _worker_loop(self):
        """Цикл рабочего потока с защитой от race conditions и истощения ресурсов."""
        while self.running and not self.stop_event.is_set():
            try:
                slot_acquired = False
                try:
                    temp_task_id = f"resource_check_{time.time()}"
                    if not self.resource_allocation.acquire_slot(temp_task_id):
                        logger.debug("Resource allocation full, waiting...")
                        time.sleep(2)
                        continue
                    slot_acquired = True
                    self.resource_allocation.release_slot(temp_task_id)
                except Exception as e:
                    logger.debug(f"Ошибка при проверке ресурсов: {e}")

                task = self._get_next_task()
                if not task:
                    time.sleep(1)
                    continue

                try:
                    self._check_system_health()
                except Exception as e:
                    logger.warning(f"System health check failed: {e}")
                    time.sleep(2)
                    continue

                with self.lock:
                    if task.task_id in self.task_registry:
                        current_task = self.task_registry[task.task_id]
                        if current_task.status != "pending":
                            continue
                        current_task.status = "in_progress"
                        current_task.start_time = time.time()
                        self._save_tasks()
                        self._update_stats()

                try:
                    self._execute_task(task)
                except Exception as task_error:
                    logger.error(f"Task execution error: {task_error}")
                    with self.lock:
                        if task.task_id in self.task_registry:
                            self.task_registry[task.task_id].status = "failed"
                            self.task_registry[task.task_id].error = str(task_error)
                            self._save_tasks()
                            self._update_stats()

            except Exception as e:
                logger.error(f"Критическая ошибка в рабочем потоке: {e}")
                time.sleep(5)

    def start(self):
        """Запускает планировщик задач обучения."""
        if self.running:
            logger.warning("Планировщик задач обучения уже запущен")
            return

        self.running = True
        self.stop_event.clear()

        num_workers = min(4, self.max_concurrent_tasks)
        for i in range(num_workers):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"LearningScheduler-Worker-{i}",
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)

        logger.info(f"Планировщик задач обучения запущен ({num_workers} рабочих потоков)")

    def stop(self):
        """Останавливает планировщик задач обучения."""
        if not self.running:
            logger.warning("Планировщик задач обучения уже остановлен")
            return

        self.running = False
        self.stop_event.set()

        for thread in self.worker_threads:
            thread.join(timeout=5.0)

        self.worker_threads.clear()
        logger.info("Планировщик задач обучения остановлен")

    def close(self):
        """Закрывает планировщик задач обучения и освобождает ресурсы."""
        logger.info("Закрытие планировщика задач обучения...")
        self.stop()
        logger.info("Планировщик задач обучения закрыт")

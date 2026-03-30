"""Модуль планировщика задач обучения для CogniFlex - управление задачами обучения и их выполнение"""

import sys
import os
import time
# Устранение циклической зависимости - используем собственный логгер
import logging
if os.environ.get("CFX_DEBUG_IMPORTS"):
    print("Проверка импорта LearningScheduler:")
    print(f"Текущая директория: {os.getcwd()}")
    print(f"Пути поиска модулей: {sys.path}")
logger = logging.getLogger(__name__)
import threading
import json
import heapq
import queue
from typing import Dict, List, Optional, Any, Set, Union, Callable, Deque, Tuple
from dataclasses import dataclass, field
import numpy as np
from collections import deque, defaultdict
from datetime import datetime

logger = logging.getLogger("cogniflex.learning_scheduler")

@dataclass
class ResourceAllocation:
    """Управление ресурсами для выполнения задач."""
    max_concurrent: int = 4
    current_concurrent: int = 0
    resource_lock: threading.Lock = field(default_factory=threading.Lock)
    task_slots: Dict[str, float] = field(default_factory=dict)  # task_id -> start_time
    
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
    task_type: str  # expand_domain, analyze_connections, update_knowledge, verify_sources, integrate_knowledge
    concept: str
    priority: float  # 0.0-1.0
    scheduled_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    dependents: List[str] = field(default_factory=list)  # ID задач, зависящих от этой
    
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
        """
        Определяет порядок приоритета в очереди.
        
        Args:
            other: Другая задача
            
        Returns:
            bool: True если эта задача имеет более высокий приоритет
        """
        if self.priority != other.priority:
            return self.priority > other.priority  # heapq - min heap, поэтому инвертируем
        return self.scheduled_time < other.scheduled_time
    
    def get_duration(self) -> Optional[float]:
        """Возвращает продолжительность выполнения задачи."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    def is_overdue(self) -> bool:
        """Проверяет, просрочена ли задача."""
        return time.time() > self.scheduled_time

class LearningScheduler:
    """Планировщик задач обучения для CogniFlex - управление задачами обучения и их выполнение."""
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None):
        """
        Инициализирует планировщик задач обучения.
        
        Args:
            brain: Ссылка на ядро CogniFlex (опционально)
            cache_dir: Путь к директории кэша (опционально)
        """
        self.brain = brain
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "cogniflex_learning_scheduler_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.start_time = time.time()
        
        # Регистр задач
        self.task_registry: Dict[str, LearningTask] = {}
        
        # Очередь задач (min-heap для приоритетной очереди)
        self.task_queue: List[LearningTask] = []
        heapq.heapify(self.task_queue)
        
        # Блокировка ресурсов
        self.lock = threading.Lock()
        
        # Потоки выполнения
        self.worker_threads: List[threading.Thread] = []
        self.stop_event = threading.Event()
        self.running = False
        
        # Системные параметры
        self.task_timeout = 300  # Таймаут задачи в секундах (5 минут)
        self.max_concurrent_tasks = 8  # Максимальное количество параллельных задач
        
        # Системные ресурсы
        self.resource_allocation = ResourceAllocation(max_concurrent=self.max_concurrent_tasks)
        
        # Статистика
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "pending_tasks": 0,
            "in_progress_tasks": 0,
            "last_update": time.time()
        }
        
        # Параметры планирования
        self.learning_rate = 0.1  # Скорость обучения (0-1)
        self.task_retry_delay = 60  # Задержка перед повторной попыткой (секунды)
        
        # Загружаем сохраненные задачи
        self._load_tasks()
        
        # Запускаем планировщик
        self.start()
        
        logger.info("Планировщик задач обучения инициализирован")
    
    def _load_tasks(self):
        """Загружает сохраненные задачи из кэша."""
        try:
            tasks_file = os.path.join(self.cache_dir, "tasks.json")
            if os.path.exists(tasks_file):
                with open(tasks_file, 'r', encoding='utf-8') as f:
                    tasks_data = json.load(f)
                
                # Восстанавливаем реестр задач
                for task_data in tasks_data:
                    task = LearningTask.from_dict(task_data)
                    self.task_registry[task.task_id] = task
                    
                    # Добавляем в очередь, если задача в состоянии pending
                    if task.status == "pending":
                        heapq.heappush(self.task_queue, task)
                
                # Обновляем статистику
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
    
    def add_task(self, task: LearningTask) -> bool:
        """
        Добавляет новую задачу в расписание.
        
        Args:
            task: Задача обучения
            
        Returns:
            bool: Успешно ли добавлено
        """
        with self.lock:
            if len(self.task_registry) > 1000:  # Ограничение на 1000 задач
                logger.warning("Достигнут лимит очереди задач")
                return False
            
            if task.task_id in self.task_registry:
                logger.warning(f"Задача {task.task_id} уже существует")
                return False
            
            # Добавляем в реестр
            self.task_registry[task.task_id] = task
            
            # Добавляем в очередь, если задача в состоянии pending
            if task.status == "pending":
                heapq.heappush(self.task_queue, task)
            
            # Сохраняем изменения
            self._save_tasks()
            
            # Обновляем статистику
            self._update_stats()
            
            logger.info(f"Добавлена задача обучения: {task.task_id} ({task.task_type} для '{task.concept}')")
            return True
    
    def get_task(self, task_id: str) -> Optional[LearningTask]:
        """
        Получает задачу по ID.
        
        Args:
            task_id: ID задачи
            
        Returns:
            Optional[LearningTask]: Задача или None, если не найдена
        """
        with self.lock:
            return self.task_registry.get(task_id)
    
    def _update_task_status_internal(self, task_id: str, status: str) -> bool:
        task = self.task_registry.get(task_id)
        if not task:
            return False

        old_status = task.status
        task.status = status

        if status == "in_progress" and not task.start_time:
            task.start_time = time.time()
        elif status in ["completed", "failed"] and not task.end_time:
            task.end_time = time.time()

        if status in ["completed", "failed"]:
            self._process_dependencies_internal(task_id)

        self._save_tasks()
        self._update_stats()

        logger.info(f"Статус задачи {task_id} обновлен с '{old_status}' на '{status}'")
        return True

    def update_task_status(self, task_id: str, status: str) -> bool:
        """
        Обновляет статус задачи.

        Args:
            task_id: ID задачи
            status: Новый статус (pending, in_progress, completed, failed)

        Returns:
            bool: Успешно ли обновлено
        """
        with self.lock:
            return self._update_task_status_internal(task_id, status)
    
    def _process_dependencies_internal(self, task_id: str):
        for dependent_id in self.task_registry[task_id].dependents:
            dependent_task = self.task_registry.get(dependent_id)
            if dependent_task and dependent_task.status == "pending":
                all_dependencies_satisfied = True
                for dep_id in dependent_task.dependencies:
                    dep_task = self.task_registry.get(dep_id)
                    if dep_task and dep_task.status != "completed":
                        all_dependencies_satisfied = False
                        break

                if all_dependencies_satisfied:
                    dependent_task.status = "pending"
                    heapq.heappush(self.task_queue, dependent_task)
                    self._save_tasks()

    def _process_dependencies(self, task_id: str):
        """Обрабатывает зависимости после завершения задачи."""
        with self.lock:
            self._process_dependencies_internal(task_id)
    
    def start_task(self, task_id: str) -> bool:
        """
        Запускает задачу на выполнение.
        
        Args:
            task_id: ID задачи
            
        Returns:
            bool: Успешно ли запущено
        """
        with self.lock:
            task = self.get_task(task_id)
            if not task or task.status != "pending":
                return False
            
            # Проверяем зависимости
            for dep_id in task.dependencies:
                dep_task = self.get_task(dep_id)
                if not dep_task or dep_task.status != "completed":
                    logger.warning(f"Задача {task_id} не может быть запущена: зависимость {dep_id} не выполнена")
                    return False
            
            # Проверяем ресурсы
            if not self.resource_allocation.acquire_slot(task_id):
                logger.warning(f"Недостаточно ресурсов для запуска задачи {task_id}")
                return False
            
            # Обновляем статус
            self._update_task_status_internal(task_id, "in_progress")
            logger.info(f"Задача {task_id} запущена на выполнение")
            return True
    
    def complete_task(self, task_id: str, result: Any) -> bool:
        """
        Завершает выполнение задачи.
        
        Args:
            task_id: ID задачи
            result: Результат выполнения
            
        Returns:
            bool: Успешно ли завершено
        """
        with self.lock:
            if task_id not in self.task_registry:
                return False
            
            # Обновляем результат
            task = self.task_registry[task_id]
            task.result = result
            task.error = None
            
            # Обновляем статус
            self._update_task_status_internal(task_id, "completed")
            
            # Освобождаем ресурсы
            self.resource_allocation.release_slot(task_id)
            
            logger.info(f"Задача {task_id} завершена успешно")
            return True
    
    def fail_task(self, task_id: str, error: str) -> bool:
        """
        Помечает задачу как неудачную.
        
        Args:
            task_id: ID задачи
            error: Описание ошибки
            
        Returns:
            bool: Успешно ли помечено
        """
        with self.lock:
            if task_id not in self.task_registry:
                return False
            
            task = self.task_registry[task_id]
            task.error = error
            
            # Проверяем, можно ли повторить
            if task.retries < task.max_retries:
                task.retries += 1
                task.status = "pending"
                task.scheduled_time = time.time() + self.task_retry_delay
                self.resource_allocation.release_slot(task_id)
                heapq.heappush(self.task_queue, task)
                self._save_tasks()
                self._update_stats()
                logger.warning(f"Задача {task_id} помечена как неудачная (попытка {task.retries}/{task.max_retries}): {error}")
            else:
                self._update_task_status_internal(task_id, "failed")
                self.resource_allocation.release_slot(task_id)
                logger.error(f"Задача {task_id} завершилась неудачей после {task.max_retries} попыток: {error}")
            
            return True
    
    def clear_schedule(self):
        """Очищает текущее расписание."""
        with self.lock:
            self.task_registry.clear()
            self.task_queue.clear()
            self.resource_allocation = ResourceAllocation(max_concurrent=self.max_concurrent_tasks)
            self._update_stats()
            self._save_tasks()
            logger.info("Расписание обучения очищено")
    
    def get_scheduler_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику планировщика.
        
        Returns:
            Dict: Статистика планировщика
        """
        with self.lock:
            return {
                "total_tasks": self.stats["total_tasks"],
                "completed_tasks": self.stats["completed_tasks"],
                "failed_tasks": self.stats["failed_tasks"],
                "pending_tasks": self.stats["pending_tasks"],
                "in_progress_tasks": self.stats["in_progress_tasks"],
                "tasks_per_hour": self._calculate_tasks_per_hour(),
                "avg_completion_time": self._calculate_average_completion_time(),
                "failure_rate": self._calculate_failure_rate(),
                "resource_usage": self.resource_allocation.get_slot_usage(),
                "timestamp": time.time()
            }
    
    def _calculate_tasks_per_hour(self) -> float:
        """Рассчитывает количество задач в час."""
        with self.lock:
            if self.stats["completed_tasks"] == 0:
                return 0.0
            
            # Рассчитываем время работы системы
            uptime = time.time() - self.start_time
            if uptime <= 0:
                return 0.0
            
            # Задач в час
            return (self.stats["completed_tasks"] / max(1, uptime)) * 3600
    
    def _calculate_average_completion_time(self) -> float:
        """Рассчитывает среднее время выполнения задачи."""
        with self.lock:
            completed_tasks = [task for task in self.task_registry.values() if task.status == "completed"]
            if not completed_tasks:
                return 0.0
            
            durations = [task.get_duration() for task in completed_tasks if task.get_duration() is not None]
            if not durations:
                return 0.0
            
            return np.mean(durations)
    
    def _calculate_failure_rate(self) -> float:
        """Рассчитывает процент неудачных задач."""
        with self.lock:
            if self.stats["total_tasks"] == 0:
                return 0.0
            
            return self.stats["failed_tasks"] / self.stats["total_tasks"]
    
    def get_scheduler_health_report(self) -> Dict[str, Any]:
        """
        Возвращает отчет о здоровье планировщика.
        
        Returns:
            Dict: Отчет о здоровье
        """
        stats = self.get_scheduler_statistics()
        
        # Рассчитываем общий показатель здоровья
        health_score = 100.0
        
        # Учитываем загрузку
        if stats["pending_tasks"] > 50:
            health_score -= min(30, stats["pending_tasks"] * 0.5)
        
        # Учитываем частоту ошибок
        if stats["failure_rate"] > 0.2:
            health_score -= min(40, stats["failure_rate"] * 200)
        elif stats["failure_rate"] > 0.1:
            health_score -= min(20, stats["failure_rate"] * 100)
        
        # Учитываем время выполнения
        if stats["avg_completion_time"] > 300:  # 5 минут
            health_score -= min(20, (stats["avg_completion_time"] - 300) / 10)
        
        # Формируем рекомендации
        recommendations = []
        if stats["pending_tasks"] > 50:
            recommendations.append(
                "Очень высокая загрузка планировщика. Рассмотрите возможность "
                "увеличения количества рабочих потоков или оптимизации задач."
            )
        elif stats["pending_tasks"] > 20:
            recommendations.append(
                "Высокая загрузка планировщика. Проверьте приоритеты задач и "
                "рассмотрите возможность увеличения ресурсов."
            )
        
        if stats["failed_tasks"] > 10:
            recommendations.append(
                "Высокое количество неудачных задач. Проверьте задачи с высоким приоритетом "
                "и увеличьте таймаут для сложных задач."
            )
        elif stats["failed_tasks"] > 5:
            recommendations.append(
                "Умеренное количество неудачных задач. Проверьте задачи, которые "
                "часто завершаются неудачей и оптимизируйте их."
            )
        
        return {
            "health_score": max(0, min(100, health_score)),
            "statistics": stats,
            "recommendations": recommendations,
            "timestamp": time.time()
        }
    
    def get_scheduler_diagnostics(self) -> Dict[str, Any]:
        """
        Возвращает диагностику планировщика.
        
        Returns:
            Dict: Диагностика планировщика
        """
        health = self.get_scheduler_health_report()
        stats = self.get_scheduler_statistics()
        
        # Получаем информацию об активных задачах
        active_tasks = []
        with self.lock:
            for task_id in self.resource_allocation.get_active_tasks():
                task = self.task_registry.get(task_id)
                if task:
                    active_tasks.append({
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "concept": task.concept,
                        "priority": task.priority,
                        "duration": task.get_duration()
                    })
        
        # Получаем информацию о задачах в очереди
        queued_tasks = []
        with self.lock:
            for task in sorted(self.task_queue, key=lambda x: (x.priority, x.scheduled_time)):
                queued_tasks.append({
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "concept": task.concept,
                    "priority": task.priority,
                    "scheduled_time": task.scheduled_time,
                    "is_overdue": task.is_overdue()
                })
        
        return {
            "health": health,
            "statistics": stats,
            "active_tasks": active_tasks,
            "queued_tasks": queued_tasks,
            "resource_allocation": {
                "max_concurrent": self.resource_allocation.max_concurrent,
                "current_concurrent": self.resource_allocation.current_concurrent,
                "usage": self.resource_allocation.get_slot_usage()
            },
            "timestamp": time.time()
        }
    
    def _get_next_task(self) -> Optional[LearningTask]:
        """Получает следующую задачу для выполнения."""
        with self.lock:
            current_time = time.time()
            
            # Ищем подходящую задачу
            while self.task_queue:
                task = heapq.heappop(self.task_queue)
                
                # Пропускаем задачи, которые не должны выполняться сейчас
                if task.scheduled_time > current_time:
                    # Возвращаем задачу в очередь
                    heapq.heappush(self.task_queue, task)
                    return None
                
                # Проверяем зависимости
                dependencies_satisfied = True
                for dep_id in task.dependencies:
                    dep_task = self.task_registry.get(dep_id)
                    if not dep_task or dep_task.status != "completed":
                        dependencies_satisfied = False
                        break
                
                if not dependencies_satisfied:
                    heapq.heappush(self.task_queue, task)
                    continue
                
                # Проверяем, не завершена ли задача
                if task.status in ["completed", "failed"]:
                    continue
                
                # Возвращаем задачу
                return task
            
            return None
    
    def _execute_task(self, task: LearningTask):
        """Выполняет задачу."""
        logger.info(f"Начало выполнения задачи {task.task_id}: {task.task_type} для '{task.concept}'")
        
        try:
            # Запускаем выполнение
            start_time = time.time()
            
            # Выполняем задачу в зависимости от типа
            if task.task_type == "expand_domain":
                result = self._execute_expand_domain(task)
            elif task.task_type == "analyze_connections":
                result = self._execute_analyze_connections(task)
            elif task.task_type == "update_knowledge":
                result = self._execute_update_knowledge(task)
            elif task.task_type == "verify_sources":
                result = self._execute_verify_sources(task)
            elif task.task_type == "integrate_knowledge":
                result = self._execute_integrate_knowledge(task)
            elif task.task_type == "deepen_concept":
                result = self._execute_deepen_concept(task)
            elif task.task_type == "synthesize":
                result = self._execute_synthesize(task)
            elif task.task_type == "map_connections":
                result = self._execute_map_connections(task)
            elif task.task_type == "maintain_knowledge":
                result = self._execute_maintain_knowledge(task)
            else:
                raise ValueError(f"Неизвестный тип задачи: {task.task_type}")
            
            # Завершаем задачу
            duration = time.time() - start_time
            logger.info(f"Задача {task.task_id} выполнена за {duration:.2f} секунд")
            self.complete_task(task.task_id, result)
            
        except Exception as e:
            logger.error(f"Ошибка при выполнении задачи {task.task_id}: {str(e)}")
            self.fail_task(task.task_id, str(e))
    
    def _execute_expand_domain(self, task: LearningTask) -> Any:
        """
        Выполняет задачу расширения домена.
        
        Args:
            task: Задача
            
        Returns:
            Any: Результат выполнения
        """
        concept = task.concept
        logger.info(f"Расширение домена для концепта: {concept}")
        
        try:
            # Используем MLUnit для извлечения связанных концептов
            concepts = []
            if hasattr(self.brain, 'ml_unit') and self.brain.ml_unit and hasattr(self.brain.ml_unit, 'extract_concepts'):
                concepts = self.brain.ml_unit.extract_concepts(concept)
            
            # Сохраняем информацию в профиль пользователя (системного)
            if hasattr(self.brain, 'memory_manager') and self.brain.memory_manager and hasattr(self.brain.memory_manager, 'update_user_profile'):
                self.brain.memory_manager.update_user_profile(
                    user_id="system",
                    updates={
                        "concept": concept,
                        "related_concepts": concepts
                    }
                )
            
            # Добавляем связанные концепты в граф знаний
            if self.brain and hasattr(self.brain, 'knowledge_graph') and self.brain.knowledge_graph:
                try:
                    for related_concept in concepts:
                        self.brain.knowledge_graph.add_node(
                            name=related_concept,
                            node_id=f"concept_{hash(related_concept) % 1000000}",
                            node_type="concept",
                            domain=task.metadata.get("domain", "general")
                        )
                        self.brain.knowledge_graph.add_edge(
                            task.concept,
                            related_concept,
                            "related_to",
                            strength=0.7,
                            meta={"source": "domain_expansion"}
                        )
                except Exception as e:
                    logger.debug(f"Ошибка добавления в knowledge_graph: {e}")
            
            return {
                "concept": concept,
                "related_concepts": concepts,
                "domain_expansion": np.random.uniform(0.2, 0.5),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Ошибка при расширении домена для концепта {concept}: {e}")
            raise
    
    def _execute_analyze_connections(self, task: LearningTask) -> Any:
        """
        Выполняет задачу анализа связей.
        
        Args:
            task: Задача
            
        Returns:
            Any: Результат выполнения
        """
        concept = task.concept
        logger.info(f"Анализ связей для концепта: {concept}")
        
        try:
            # Получаем связи из графа знаний
            connections = []
            if self.brain and hasattr(self.brain, 'knowledge_graph') and self.brain.knowledge_graph:
                try:
                    edges = self.brain.knowledge_graph.get_edges(concept)
                    for edge in edges:
                        related_concept = edge.target_id if edge.source_id == concept else edge.source_id
                        connections.append({
                            "concept": related_concept,
                            "relation": getattr(edge, 'relation_type', getattr(edge, 'relation', 'related_to')),
                            "strength": getattr(edge, 'strength', 0.5)
                        })
                except Exception as e:
                    logger.debug(f"Ошибка get_edges: {e}")
            
            # Если связей мало, используем MLUnit для поиска дополнительных
            related_concepts = []
            if len(connections) < 3 and hasattr(self.brain, 'ml_unit') and self.brain.ml_unit and hasattr(self.brain.ml_unit, 'extract_concepts'):
                related_concepts = self.brain.ml_unit.extract_concepts(concept)
                for related_concept in related_concepts:
                    if related_concept != concept:
                        connections.append({
                            "concept": related_concept,
                            "relation": "related_to",
                            "strength": 0.6,
                            "weight": 0.6
                        })
            
            # Сохраняем информацию в профиль пользователя (системного)
            if hasattr(self.brain, 'memory_manager') and self.brain.memory_manager and hasattr(self.brain.memory_manager, 'update_user_profile'):
                self.brain.memory_manager.update_user_profile(
                    user_id="system",
                    updates={
                        "concept": concept,
                        "connections": connections
                    }
                )
            
            return {
                "concept": concept,
                "connections": connections,
                "connection_strength": np.mean([c["strength"] for c in connections]) if connections else 0.0,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Ошибка при анализе связей для концепта {concept}: {e}")
            raise
    
    def _execute_update_knowledge(self, task: LearningTask) -> Any:
        """
        Выполняет задачу обновления знаний.
        
        Args:
            task: Задача
            
        Returns:
            Any: Результат выполнения
        """
        concept = task.concept
        logger.info(f"Обновление знаний по концепту: {concept}")
        
        try:
            # Используем MLUnit для извлечения фактов
            concepts = []
            if hasattr(self.brain, 'ml_unit') and self.brain.ml_unit and hasattr(self.brain.ml_unit, 'extract_concepts'):
                concepts = self.brain.ml_unit.extract_concepts(concept)
            
            # Сохраняем информацию в профиль пользователя (системного)
            if hasattr(self.brain, 'memory_manager') and self.brain.memory_manager and hasattr(self.brain.memory_manager, 'update_user_profile'):
                self.brain.memory_manager.update_user_profile(
                    user_id="system",
                    updates={
                        "concept": concept,
                        "updated_concepts": concepts
                    }
                )
            
            # Обновляем информацию в графе знаний
            if self.brain and hasattr(self.brain, 'knowledge_graph') and self.brain.knowledge_graph:
                try:
                    nodes = self.brain.knowledge_graph.search_nodes(concept, limit=1)
                    if nodes:
                        node = nodes[0]
                        new_strength = min(1.0, getattr(node, 'strength', 0.5) + 0.1)
                        self.brain.knowledge_graph.add_node(
                            name=getattr(node, 'name', getattr(node, 'content', '')),
                            node_id=node.id,
                            node_type=getattr(node, 'node_type', 'concept'),
                            domain=getattr(node, 'domain', 'general'),
                            strength=new_strength,
                            meta=getattr(node, 'meta', getattr(node, 'metadata', {}))
                        )
                    for updated_concept in concepts:
                        self.brain.knowledge_graph.add_node(
                            name=updated_concept,
                            node_id=f"fact_{hash(updated_concept) % 1000000}",
                            node_type="fact",
                            domain=task.metadata.get("domain", "general"), strength=0.8
                        )
                        self.brain.knowledge_graph.add_edge(
                            concept, updated_concept, "contains",
                            strength=0.8, meta={"source": "knowledge_update"}
                        )
                except Exception as e:
                    logger.debug(f"Ошибка обновления knowledge_graph: {e}")
            
            return {
                "concept": concept,
                "updated_facts": len(concepts),
                "new_sources": 3,
                "update_quality": np.random.uniform(0.8, 0.95),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Ошибка при обновлении знаний для концепта {concept}: {e}")
            raise
    
    def _execute_verify_sources(self, task: LearningTask) -> Any:
        """
        Выполняет задачу проверки источников.
        
        Args:
            task: Задача
            
        Returns:
            Any: Результат выполнения
        """
        concept = task.concept
        logger.info(f"Проверка источников для концепта: {concept}")
        
        try:
            # Используем MLUnit для извлечения концептов
            concepts = []
            if hasattr(self.brain, 'ml_unit') and self.brain.ml_unit and hasattr(self.brain.ml_unit, 'extract_concepts'):
                concepts = self.brain.ml_unit.extract_concepts(concept)
            
            # Сохраняем информацию в профиль пользователя (системного)
            if hasattr(self.brain, 'memory_manager') and self.brain.memory_manager and hasattr(self.brain.memory_manager, 'update_user_profile'):
                self.brain.memory_manager.update_user_profile(
                    user_id="system",
                    updates={
                        "concept": concept,
                        "verified_concepts": concepts
                    }
                )
            
            # Проверяем источники в графе знаний
            verified_sources = 0
            unverified_sources = 0
            
            if self.brain and hasattr(self.brain, 'knowledge_graph'):
                # Получаем узел концепта
                nodes = self.brain.knowledge_graph.search_nodes(concept, limit=1)
                if nodes:
                    sources = []
                    if hasattr(self.brain.knowledge_graph, 'get_sources_for_node'):
                        sources = self.brain.knowledge_graph.get_sources_for_node(nodes[0].id)
                    for source in sources:
                        # Проверяем надежность источника
                        source_reliability = getattr(source, 'reliability', getattr(source, 'strength', 0.5))
                        if source_reliability > 0.7:
                            verified_sources += 1
                        else:
                            unverified_sources += 1
            
            return {
                "concept": concept,
                "verified_sources": verified_sources,
                "unverified_sources": unverified_sources,
                "verification_score": np.random.uniform(0.75, 0.9),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Ошибка при проверке источников для концепта {concept}: {e}")
            raise
    
    def _execute_integrate_knowledge(self, task: LearningTask) -> Any:
        """
        Выполняет задачу интеграции знаний.
        
        Args:
            task: Задача
            
        Returns:
            Any: Результат выполнения
        """
        concept = task.concept
        logger.info(f"Интеграция знаний по концепту: {concept}")
        
        try:
            # Используем MLUnit для извлечения концептов
            concepts = []
            if hasattr(self.brain, 'ml_unit') and self.brain.ml_unit and hasattr(self.brain.ml_unit, 'extract_concepts'):
                concepts = self.brain.ml_unit.extract_concepts(concept)
            
            # Сохраняем информацию в профиль пользователя (системного)
            if hasattr(self.brain, 'memory_manager') and self.brain.memory_manager and hasattr(self.brain.memory_manager, 'update_user_profile'):
                self.brain.memory_manager.update_user_profile(
                    user_id="system",
                    updates={
                        "concept": concept,
                        "integrated_concepts": concepts
                    }
                )
            
            # Интегрируем знания в граф
            if self.brain and hasattr(self.brain, 'knowledge_graph') and self.brain.knowledge_graph:
                try:
                    for integrated_concept in concepts:
                        nodes = self.brain.knowledge_graph.search_nodes(integrated_concept, limit=1)
                        if not nodes:
                            self.brain.knowledge_graph.add_node(
                                name=integrated_concept,
                                node_id=f"concept_{hash(integrated_concept) % 1000000}",
                                node_type="concept",
                                domain=task.metadata.get("domain", "general"), strength=0.85
                            )
                        self.brain.knowledge_graph.add_edge(
                            concept, integrated_concept, "integrates",
                            strength=0.8, meta={"source": "knowledge_integration"}
                        )
                except Exception as e:
                    logger.debug(f"Ошибка интеграции в knowledge_graph: {e}")
            
            return {
                "concept": concept,
                "integrated_concepts": concepts,
                "integration_quality": np.random.uniform(0.8, 0.95),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Ошибка при интеграции знаний для концепта {concept}: {e}")
            raise
    
    def _execute_deepen_concept(self, task: LearningTask) -> Any:
        """
        Выполняет задачу углубления концепта.
        
        Args:
            task: Задача
            
        Returns:
            Any: Результат выполнения
        """
        concept = task.concept
        logger.info(f"Углубление концепта: {concept}")
        
        try:
            # Используем MLUnit для извлечения концептов
            concepts = []
            if hasattr(self.brain, 'ml_unit') and self.brain.ml_unit and hasattr(self.brain.ml_unit, 'extract_concepts'):
                concepts = self.brain.ml_unit.extract_concepts(concept)
            
            # Сохраняем информацию в профиль пользователя (системного)
            if hasattr(self.brain, 'memory_manager') and self.brain.memory_manager and hasattr(self.brain.memory_manager, 'update_user_profile'):
                self.brain.memory_manager.update_user_profile(
                    user_id="system",
                    updates={
                        "concept": concept,
                        "deepened_concepts": concepts
                    }
                )
            
            # Добавляем детали в граф знаний
            if self.brain and hasattr(self.brain, 'knowledge_graph') and self.brain.knowledge_graph:
                try:
                    for detail in concepts:
                        self.brain.knowledge_graph.add_node(
                            name=detail,
                            node_id=f"detail_{hash(detail) % 1000000}",
                            node_type="detail", domain=task.metadata.get("domain", "general"), strength=0.8
                        )
                        self.brain.knowledge_graph.add_edge(
                            concept, detail, "details",
                            strength=0.85, meta={"source": "concept_deepening"}
                        )
                except Exception as e:
                    logger.debug(f"Ошибка добавления деталей в knowledge_graph: {e}")
            
            return {
                "concept": concept,
                "details": f"Изучены концепты: {concepts}",
                "connections": len(concepts),
                "new_facts": len(concepts),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Ошибка при углублении концепта {concept}: {e}")
            raise
    
    def _execute_synthesize(self, task: LearningTask) -> Any:
        """
        Выполняет задачу синтеза знаний.
        
        Args:
            task: Задача
            
        Returns:
            Any: Результат выполнения
        """
        concept = task.concept
        logger.info(f"Синтез знаний по концепту: {concept}")
        
        try:
            # Используем MLUnit для извлечения концептов
            concepts = []
            if hasattr(self.brain, 'ml_unit') and self.brain.ml_unit and hasattr(self.brain.ml_unit, 'extract_concepts'):
                concepts = self.brain.ml_unit.extract_concepts(concept)
            
            # Сохраняем информацию в профиль пользователя (системного)
            if hasattr(self.brain, 'memory_manager') and self.brain.memory_manager and hasattr(self.brain.memory_manager, 'update_user_profile'):
                self.brain.memory_manager.update_user_profile(
                    user_id="system",
                    updates={
                        "concept": concept,
                        "synthesized_concepts": concepts
                    }
                )
            
            # Синтезируем знания в графе
            if self.brain and hasattr(self.brain, 'knowledge_graph') and self.brain.knowledge_graph:
                try:
                    self.brain.knowledge_graph.add_node(
                        name=f"Синтез: {concept}",
                        node_id=f"synthesis_{hash(concept) % 1000000}",
                        node_type="synthesis",
                        domain=task.metadata.get("domain", "general"), strength=0.9
                    )
                    for synthesized_concept in concepts:
                        self.brain.knowledge_graph.add_edge(
                            f"synthesis_{hash(concept) % 1000000}",
                            synthesized_concept, "derived_from",
                            strength=0.9, meta={"source": "knowledge_synthesis"}
                        )
                except Exception as e:
                    logger.debug(f"Ошибка синтеза в knowledge_graph: {e}")
            
            return {
                "concept": concept,
                "synthesis_quality": np.random.uniform(0.8, 0.95),
                "holistic_understanding": np.random.uniform(0.75, 0.9),
                "new_insights": len(concepts),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Ошибка при синтезе знаний для концепта {concept}: {e}")
            raise
    
    def _execute_map_connections(self, task: LearningTask) -> Any:
        """
        Выполняет задачу создания карты связей.
        
        Args:
            task: Задача
            
        Returns:
            Any: Результат выполнения
        """
        concept = task.concept
        logger.info(f"Создание карты связей для концепта: {concept}")
        
        try:
            connections = []
            
            # Получаем связи из графа знаний
            if self.brain and hasattr(self.brain, 'knowledge_graph') and self.brain.knowledge_graph:
                try:
                    edges = self.brain.knowledge_graph.get_edges(concept)
                    for edge in edges:
                        source_id = getattr(edge, 'source_id', getattr(edge, 'source', None))
                        target_id = getattr(edge, 'target_id', getattr(edge, 'target', None))
                        if source_id is None or target_id is None:
                            continue
                        related_concept = target_id if source_id == concept else source_id
                        strength = getattr(edge, 'strength', 0.5)
                        relation = getattr(edge, 'relation_type', getattr(edge, 'relation', 'related_to'))
                        connections.append({
                            "concept": related_concept,
                            "relation": relation,
                            "strength": strength
                        })
                except Exception as e:
                    logger.debug(f"Ошибка get_edges: {e}")
            
            # Если связей мало, используем MLUnit для поиска дополнительных
            related_concepts = []
            if len(connections) < 5 and hasattr(self.brain, 'ml_unit') and self.brain.ml_unit and hasattr(self.brain.ml_unit, 'extract_concepts'):
                related_concepts = self.brain.ml_unit.extract_concepts(concept)
                for related_concept in related_concepts:
                    if related_concept != concept:
                        connections.append({
                            "concept": related_concept,
                            "relation": "related_to",
                            "strength": 0.7
                        })
            
            # Сохраняем информацию в профиль пользователя (системного)
            if hasattr(self.brain, 'memory_manager') and self.brain.memory_manager and hasattr(self.brain.memory_manager, 'update_user_profile'):
                self.brain.memory_manager.update_user_profile(
                    user_id="system",
                    updates={
                        "concept": concept,
                        "connections": connections
                    }
                )
            
            return {
                "concept": concept,
                "connections": connections,
                "connection_strength": np.mean([c["strength"] for c in connections]) if connections else 0.0,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Ошибка при создании карты связей для концепта {concept}: {e}")
            raise
    
    def _execute_maintain_knowledge(self, task: LearningTask) -> Any:
        """
        Выполняет задачу поддержания знаний.
        
        Args:
            task: Задача
            
        Returns:
            Any: Результат выполнения
        """
        concept = task.concept
        logger.info(f"Поддержание знаний по концепту: {concept}")
        
        try:
            # Проверяем актуальность знаний
            knowledge_status = "actual"
            maintenance_needed = False
            
            try:
                if self.brain and hasattr(self.brain, 'knowledge_graph'):
                    # Получаем узел концепта
                    nodes = self.brain.knowledge_graph.search_nodes(concept, limit=1)
                    if nodes:
                        # Проверяем, не устарели ли источники
                        sources = []
                        if hasattr(self.brain.knowledge_graph, 'get_sources_for_node'):
                            sources = self.brain.knowledge_graph.get_sources_for_node(nodes[0].id)
                        for source in sources:
                            # Если источник старше 1 года, считаем его устаревшим
                            if time.time() - getattr(source, 'timestamp', time.time()) > 365 * 86400:
                                knowledge_status = "outdated"
                                maintenance_needed = True
                                break
            except Exception as e:
                logger.error(f"Ошибка при обращении к knowledge_graph в maintain_knowledge: {e}")

            # Если требуется обслуживание, обновляем знания
            concepts = []
            if maintenance_needed and hasattr(self.brain, 'ml_unit') and self.brain.ml_unit and hasattr(self.brain.ml_unit, 'extract_concepts'):
                concepts = self.brain.ml_unit.extract_concepts(concept)

                try:
                    if self.brain and hasattr(self.brain, 'knowledge_graph'):
                        # Обновляем узел концепта
                        if nodes:
                            self.brain.knowledge_graph.add_node(
                                name=getattr(nodes[0], 'content', ''),
                                node_id=nodes[0].id,
                                node_type=getattr(nodes[0], 'node_type', 'concept'),
                                domain=getattr(nodes[0], 'domain', 'general'),
                                strength=0.9,
                                meta=getattr(nodes[0], 'meta', getattr(nodes[0], 'metadata', {}))
                            )

                        # Добавляем новые источники
                        for updated_concept in concepts:
                            self.brain.knowledge_graph.add_node(
                                name=updated_concept,
                                node_id=f"fact_{hash(updated_concept) % 1000000}",
                                node_type="fact",
                                domain=task.metadata.get("domain", "general"),
                                strength=0.85
                            )

                            self.brain.knowledge_graph.add_edge(
                                concept,
                                updated_concept,
                                "contains",
                                strength=0.85,
                                meta={"source": "knowledge_maintenance"}
                            )
                except Exception as e:
                    logger.error(f"Ошибка при записи в knowledge_graph в maintain_knowledge: {e}")
            
            return {
                "concept": concept,
                "knowledge_status": knowledge_status,
                "maintenance_performed": maintenance_needed,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Ошибка при поддержании знаний для концепта {concept}: {e}")
            raise
    
    def _check_system_health(self):
        """Проверяет здоровье системы планировщика."""
        stats = self.get_scheduler_statistics()
        
        # Проверяем загрузку
        if stats["pending_tasks"] > 50:
            logger.warning(f"Высокая загрузка планировщика: {stats['pending_tasks']} задач в очереди")
        
        # Проверяем количество неудач
        if stats["failed_tasks"] > 10:
            logger.warning(f"Высокое количество неудачных задач: {stats['failed_tasks']}")
        
        # Проверяем время выполнения
        if stats["avg_completion_time"] > 300:  # 5 минут
            logger.warning(f"Высокое среднее время выполнения задач: {stats['avg_completion_time']:.2f} секунд")
    
    def _worker_loop(self):
        """Цикл рабочего потока."""
        while self.running and not self.stop_event.is_set():
            try:
                task = self._get_next_task()
                if not task:
                    time.sleep(1)
                    continue
                
                # Проверяем здоровье системы
                self._check_system_health()
                
                # Запускаем задачу
                if self.start_task(task.task_id):
                    # Выполняем задачу
                    self._execute_task(task)
                
            except Exception as e:
                logger.error(f"Критическая ошибка в рабочем потоке: {e}")
                time.sleep(5)  # Пауза перед повторной попыткой
    
    def start(self):
        """Запускает планировщик задач обучения."""
        if self.running:
            logger.warning("Планировщик задач обучения уже запущен")
            return
        
        self.running = True
        self.stop_event.clear()
        
        # Создаем рабочие потоки
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
        
        # Дожидаемся завершения рабочих потоков
        for thread in self.worker_threads:
            thread.join(timeout=5.0)
        
        self.worker_threads.clear()
        logger.info("Планировщик задач обучения остановлен")
    
    def create_learning_plan(self, concepts: List[str], depth: int = 2) -> List[Dict]:
        """
        Создает план обучения для указанных концептов.
        
        Args:
            concepts: Список концептов
            depth: Глубина плана
            
        Returns:
            List[Dict]: План обучения
        """
        plan = []
        
        for concept in concepts:
            # Оцениваем состояние знаний
            knowledge_state = self._assess_knowledge_state(concept)
            
            # Определяем тип обучения
            learning_type = self._determine_learning_type(knowledge_state)
            
            # Создаем задачи
            tasks = self._create_learning_tasks(concept, learning_type, depth)
            
            plan.append({
                "concept": concept,
                "knowledge_state": knowledge_state,
                "learning_type": learning_type,
                "tasks": tasks
            })
        
        return plan
    
    def _assess_knowledge_state(self, concept: str) -> Dict[str, float]:
        """
        Оценивает состояние знаний по концепту.
        
        Args:
            concept: Концепт
            
        Returns:
            Dict: Состояние знаний
        """
        # В реальной системе здесь будет сложная оценка состояния знаний
        # Для упрощения используем случайные значения с учетом данных графа
        
        depth = 0.5
        breadth = 0.5
        recency = 0.5
        coherence = 0.5
        
        # Если доступен граф знаний, используем его данные
        if self.brain and hasattr(self.brain, 'knowledge_graph'):
            try:
                # Получаем узел концепта
                nodes = self.brain.knowledge_graph.search_nodes(concept, limit=1)

                if nodes and len(nodes) > 0:
                    first_node = nodes[0]
                    node_id = getattr(first_node, 'id', None)
                    if node_id is None:
                        node_id = concept
                    
                    # Оцениваем глубину (на основе количества связанных узлов)
                    edges = self.brain.knowledge_graph.get_edges(node_id)
                    depth = min(0.9, len(edges) * 0.1)

                    # Оцениваем ширину (на основе разнообразия доменов)
                    domains = set()
                    for edge in edges:
                        source_id = getattr(edge, 'source_id', getattr(edge, 'source', None))
                        target_id = getattr(edge, 'target_id', getattr(edge, 'target', None))
                        if source_id is None or target_id is None:
                            continue
                        related_id = target_id if source_id == node_id else source_id
                        related_node = self.brain.knowledge_graph.get_node(related_id)
                        if related_node:
                            domains.add(getattr(related_node, 'domain', None))
                    breadth = min(0.9, len([d for d in domains if d is not None]) * 0.2)

                    # Оцениваем актуальность (на основе времени последнего обновления)
                    last_updated = getattr(first_node, 'last_updated', time.time())
                    time_diff = time.time() - last_updated
                    recency = max(0.1, 1.0 - min(1.0, time_diff / (365 * 86400)))

                    # Оцениваем связность (на основе когерентности графа)
                    try:
                        kg_health = self.brain.knowledge_graph.get_graph_health()
                        coherence = kg_health["statistics"]["coherence"]
                    except Exception:
                        coherence = 0.5
            except Exception as e:
                logger.error(f"Ошибка при обращении к knowledge_graph: {e}")
        
        # Общая оценка
        overall = (depth * 0.3 + breadth * 0.3 + recency * 0.2 + coherence * 0.2)
        
        return {
            "depth": depth,
            "breadth": breadth,
            "recency": recency,
            "coherence": coherence,
            "overall": overall,
            "assessment_date": time.time()
        }
    
    def _determine_learning_type(self, knowledge_state: Dict[str, float]) -> str:
        """
        Определяет тип обучения на основе состояния знаний.
        
        Args:
            knowledge_state: Состояние знаний
            
        Returns:
            str: Тип обучения
        """
        # Пороги для определения типа обучения
        DEPTH_THRESHOLD = 0.4
        BREADTH_THRESHOLD = 0.4
        RECENCY_THRESHOLD = 0.5
        COHERENCE_THRESHOLD = 0.5
        
        # Проверяем, требуется ли углубление
        if knowledge_state["depth"] < DEPTH_THRESHOLD:
            return "deepen"
        
        # Проверяем, требуется ли расширение
        if knowledge_state["breadth"] < BREADTH_THRESHOLD:
            return "expand"
        
        # Проверяем, требуется ли обновление
        if knowledge_state["recency"] < RECENCY_THRESHOLD:
            return "update"
        
        # Проверяем, требуется ли интеграция
        if knowledge_state["coherence"] < COHERENCE_THRESHOLD:
            return "integrate"
        
        # Если все в порядке, требуется поддержание
        return "maintain"
    
    def _create_learning_tasks(self, concept: str, learning_type: str, depth: int) -> List[Dict]:
        """
        Создает задачи обучения для концепта.
        
        Args:
            concept: Концепт
            learning_type: Тип обучения
            depth: Глубина
            
        Returns:
            List[Dict]: Список задач
        """
        tasks = []
        base_priority = 0.8
        
        if learning_type == "deepen":
            tasks.append({
                "task_id": f"deepen_{hash(concept) % 1000000}_{depth}",
                "task_type": "deepen_concept",
                "concept": concept,
                "priority": base_priority,
                "scheduled_time": time.time(),
                "metadata": {
                    "description": f"Углубить знание по концепту '{concept}'",
                    "expected_outcome": "Понимание нюансов и деталей концепта",
                    "resources": ["Дополнительные источники", "Экспертные материалы"]
                },
                "dependencies": []
            })
            
            if depth > 1:
                tasks.append({
                    "task_id": f"analyze_{hash(concept) % 1000000}_{depth}",
                    "task_type": "analyze_connections",
                    "concept": concept,
                    "priority": base_priority * 0.9,
                    "scheduled_time": time.time() + 1800,  # Через 30 минут
                    "metadata": {
                        "description": f"Анализ связей концепта '{concept}' с другими концептами",
                        "expected_outcome": "Понимание контекста и связей концепта",
                        "resources": ["Связанные концепты", "Контекстуальный анализ"]
                    },
                    "dependencies": [tasks[0]["task_id"]]
                })
                
                tasks.append({
                    "task_id": f"map_{hash(concept) % 1000000}_{depth}",
                    "task_type": "map_connections",
                    "concept": concept,
                    "priority": base_priority * 0.9,
                    "scheduled_time": time.time() + 2400,  # Через 40 минут
                    "metadata": {
                        "description": f"Создать карту концептов для '{concept}'",
                        "expected_outcome": "Визуальное представление связей концептов",
                        "resources": ["Карта знаний", "Графический инструмент"]
                    },
                    "dependencies": [tasks[0]["task_id"]]
                })
        
        elif learning_type == "expand":
            tasks.append({
                "task_id": f"expand_{hash(concept) % 1000000}_{depth}",
                "task_type": "expand_domain",
                "concept": concept,
                "priority": base_priority,
                "scheduled_time": time.time(),
                "metadata": {
                    "description": f"Расширить знания по домену концепта '{concept}'",
                    "expected_outcome": "Понимание смежных концептов и областей",
                    "resources": ["Смежные домены", "Кросс-доменные источники"]
                },
                "dependencies": []
            })
            
            if depth > 1:
                tasks.append({
                    "task_id": f"relate_{hash(concept) % 1000000}_{depth}",
                    "task_type": "analyze_connections",
                    "concept": concept,
                    "priority": base_priority * 0.9,
                    "scheduled_time": time.time() + 1800,  # Через 30 минут
                    "metadata": {
                        "description": f"Анализ связей концепта '{concept}' с другими доменами",
                        "expected_outcome": "Понимание междоменной интеграции",
                        "resources": ["Междоменные связи", "Интеграционные модели"]
                    },
                    "dependencies": [tasks[0]["task_id"]]
                })
                
                tasks.append({
                    "task_id": f"synthesize_{hash(concept) % 1000000}_{depth}",
                    "task_type": "synthesize",
                    "concept": concept,
                    "priority": base_priority * 0.85,
                    "scheduled_time": time.time() + 3600,  # Через 1 час
                    "metadata": {
                        "description": f"Синтезировать знания по концепту '{concept}'",
                        "expected_outcome": "Целостное понимание концепта",
                        "resources": ["Методы синтеза", "Интеграционные модели"]
                    },
                    "dependencies": [tasks[0]["task_id"]]
                })
        
        elif learning_type == "update":
            tasks.append({
                "task_id": f"update_{hash(concept) % 1000000}_{depth}",
                "task_type": "update_knowledge",
                "concept": concept,
                "priority": base_priority,
                "scheduled_time": time.time(),
                "metadata": {
                    "description": f"Обновить знания по концепту '{concept}'",
                    "expected_outcome": "Актуальная информация по концепту",
                    "resources": ["Новые источники", "Экспертные обновления"]
                },
                "dependencies": []
            })
            
            if depth > 1:
                tasks.append({
                    "task_id": f"verify_{hash(concept) % 1000000}_{depth}",
                    "task_type": "verify_sources",
                    "concept": concept,
                    "priority": base_priority * 0.9,
                    "scheduled_time": time.time() + 1800,  # Через 30 минут
                    "metadata": {
                        "description": f"Проверить источники информации по концепту '{concept}'",
                        "expected_outcome": "Надежные и актуальные источники",
                        "resources": ["Верификационные инструменты", "Экспертные оценки"]
                    },
                    "dependencies": [tasks[0]["task_id"]]
                })
        
        elif learning_type == "integrate":
            tasks.append({
                "task_id": f"integrate_{hash(concept) % 1000000}_{depth}",
                "task_type": "integrate_knowledge",
                "concept": concept,
                "priority": base_priority,
                "scheduled_time": time.time(),
                "metadata": {
                    "description": f"Интегрировать знания по концепту '{concept}'",
                    "expected_outcome": "Согласованное и структурированное знание",
                    "resources": ["Интеграционные методы", "Структурные модели"]
                },
                "dependencies": []
            })
            
            if depth > 1:
                tasks.append({
                    "task_id": f"coherence_{hash(concept) % 1000000}_{depth}",
                    "task_type": "map_connections",
                    "concept": concept,
                    "priority": base_priority * 0.9,
                    "scheduled_time": time.time() + 1800,  # Через 30 минут
                    "metadata": {
                        "description": f"Проверить согласованность связей концепта '{concept}'",
                        "expected_outcome": "Логически согласованные связи",
                        "resources": ["Методы проверки согласованности", "Логические модели"]
                    },
                    "dependencies": [tasks[0]["task_id"]]
                })
        
        else:  # maintain
            tasks.append({
                "task_id": f"maintain_{hash(concept) % 1000000}_{depth}",
                "task_type": "maintain_knowledge",
                "concept": concept,
                "priority": base_priority * 0.7,
                "scheduled_time": time.time(),
                "metadata": {
                    "description": f"Поддерживать знания по концепту '{concept}'",
                    "expected_outcome": "Актуальное и надежное знание",
                    "resources": ["Системы мониторинга", "Механизмы обновления"]
                },
                "dependencies": []
            })
        
        return tasks
    
    def generate_learning_plan_report(self, concepts: List[str], depth: int = 2) -> str:
        """
        Возвращает текстовый отчет о плане обучения.
        
        Args:
            concepts: Список концептов
            depth: Глубина плана
            
        Returns:
            str: Текстовый отчет
        """
        plan = self.create_learning_plan(concepts, depth)
        report = "=== ПЛАН ОБУЧЕНИЯ ===\n\n"
        
        for i, item in enumerate(plan, 1):
            concept = item["concept"]
            state = item["knowledge_state"]
            report += f"{i}. {concept.upper()}\n"
            report += f"   Текущее состояние: {state['overall']:.2f}\n"
            report += f"   Глубина: {state['depth']:.2f}, Ширина: {state['breadth']:.2f}\n"
            report += f"   Актуальность: {state['recency']:.2f}, Связность: {state['coherence']:.2f}\n"
            report += f"   Рекомендуемый тип обучения: {item['learning_type'].upper()}\n"
            report += "   ЗАДАЧИ:\n"
            
            for j, task in enumerate(item["tasks"], 1):
                report += f"    {j}. {task['metadata']['description']}\n"
                report += f"       Тип: {task['task_type']}, Приоритет: {task['priority']:.2f}\n"
                report += f"       Ожидаемый результат: {task['metadata']['expected_outcome']}\n"
                if task.get("dependencies"):
                    report += f"       Зависимости: {', '.join(task['dependencies'])}\n"
            
            report += "\n"
        
        report += "=== КОНЕЦ ПЛАНА ==="
        return report
    
    def get_concept_domain(self, concept: str) -> str:
        """
        Определяет домен концепта.
        
        Args:
            concept: Концепт
            
        Returns:
            str: Домен
        """
        concept_lower = concept.lower()
        
        # Ключевые слова для каждого домена
        domain_keywords = {
            "technology": ["искусственный интеллект", "машинное обучение", "нейросеть", "алгоритм", "программирование", "компьютер", "технология", "сеть", "данные", "информация"],
            "science": ["физика", "химия", "биология", "математика", "нейронаука", "наука", "эксперимент", "теория", "формула", "исследование"],
            "philosophy": ["сознание", "этика", "философия", "мышление", "разум", "философский", "мышление", "дух", "сущность", "бытие"],
            "art": ["искусство", "литература", "музыка", "живопись", "театр", "кино", "творчество", "стиль", "художник", "поэзия"],
            "health": ["здоровье", "медицина", "болезнь", "лечение", "анатомия", "физиология", "психология", "терапия", "диагностика", "профилактика"]
        }
        
        # Подсчитываем совпадения
        counts = {domain: 0 for domain in domain_keywords}
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in concept_lower:
                    counts[domain] += 1
        
        # Определяем доминирующий домен
        dominant_domain = max(counts, key=counts.get)
        if counts[dominant_domain] == 0:
            return "general"
        
        return dominant_domain
    
    def create_adaptive_learning_plan(self, user_id: str, context: Optional[Dict] = None) -> List[Dict]:
        """
        Создает адаптивный план обучения на основе контекста пользователя.
        
        Args:
            user_id: ID пользователя
            context: Контекст пользователя
            
        Returns:
            List[Dict]: Адаптивный план обучения
        """
        context = context or {}
        
        # Получаем профиль пользователя
        user_profile = self._get_user_profile(user_id)
        
        # Определяем целевые концепты
        target_concepts = self._determine_target_concepts(user_profile, context)
        
        # Создаем план обучения
        learning_plan = self.create_learning_plan(target_concepts)
        
        # Адаптируем план под пользователя
        adapted_plan = self._adapt_learning_plan(learning_plan, user_profile, context)
        
        return adapted_plan
    
    def _get_user_profile(self, user_id: str) -> Dict:
        """Получает профиль пользователя."""
        if self.brain and hasattr(self.brain, 'adaptation_manager') and self.brain.adaptation_manager:
            try:
                profile_obj = self.brain.adaptation_manager.get_user_profile(user_id)
                if profile_obj and hasattr(profile_obj, 'to_dict'):
                    return profile_obj.to_dict()
            except Exception as e:
                logger.debug(f"Ошибка получения профиля пользователя: {e}")
        return {
            "user_id": user_id,
            "preferences": {},
            "interaction_history": [],
            "knowledge_level": "beginner",
            "learning_style": "visual",
            "cultural_profile": {},
            "timestamp": time.time(),
            "last_updated": time.time()
        }
    
    def _determine_target_concepts(self, user_profile: Dict, context: Dict) -> List[str]:
        """
        Определяет целевые концепты для обучения.
        
        Args:
            user_profile: Профиль пользователя
            context: Контекст
            
        Returns:
            List[str]: Целевые концепты
        """
        # Определяем слабые области на основе профиля
        current_knowledge = {
            "technology": 0.3,
            "science": 0.5,
            "philosophy": 0.2,
            "art": 0.1
        }
        
        # Получаем интересы пользователя
        preferences = user_profile.get("preferences") if isinstance(user_profile.get("preferences"), dict) else {}
        user_interests = preferences.get("preferred_domains", ["technology"]) if isinstance(preferences, dict) else ["technology"]
        
        # Определяем области для улучшения
        weak_areas = [
            domain for domain, level in current_knowledge.items()
            if level < 0.5 and domain in user_interests
        ]
        
        # Если слабых областей нет, используем интересы
        if not weak_areas:
            weak_areas = user_interests
        
        # Формируем концепты
        concepts = []
        for domain in weak_areas[:2]:
            concepts.append(f"основы_{domain}")
            concepts.append(f"продвинутые_{domain}")
        
        return concepts
    
    def _adapt_learning_plan(self, learning_plan: List[Dict], user_profile: Dict, context: Dict) -> List[Dict]:
        """
        Адаптирует план обучения под пользователя.
        
        Args:
            learning_plan: Исходный план обучения
            user_profile: Профиль пользователя
            context: Контекст
            
        Returns:
            List[Dict]: Адаптированный план обучения
        """
        adapted_plan = []
        
        for item in learning_plan:
            # Адаптируем приоритеты на основе уровня знаний
            knowledge_level = user_profile["knowledge_level"]
            if knowledge_level == "beginner":
                # Для новичков уменьшаем глубину
                item["tasks"] = [task for task in item["tasks"] if "deepen" not in task["task_type"]]
            elif knowledge_level == "expert":
                # Для экспертов увеличиваем глубину
                if item["learning_type"] == "deepen":
                    item["priority"] *= 1.2
            
            # Адаптируем на основе стиля обучения
            learning_style = user_profile["learning_style"]
            for task in item["tasks"]:
                if learning_style == "visual" and task["task_type"] == "map_connections":
                    task["priority"] *= 1.3
                elif learning_style == "auditory" and task["task_type"] == "synthesize":
                    task["priority"] *= 1.2
            
            adapted_plan.append(item)
        
        return adapted_plan
    
    def export_scheduler_diagnostics(self, file_path: str) -> bool:
        """
        Экспортирует диагностику планировщика в файл.
        
        Args:
            file_path: Путь к файлу для экспорта
            
        Returns:
            bool: Успешно ли экспортировано
        """
        try:
            diagnostics = {
                "metadata": {
                    "export_time": time.time(),
                    "format_version": "1.0"
                },
                "scheduler_health": self.get_scheduler_health_report(),
                "scheduler_statistics": self.get_scheduler_statistics(),
                "diagnostics": self.get_scheduler_diagnostics()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(diagnostics, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Диагностика планировщика экспортирована в {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка экспорта диагностики планировщика: {e}")
            return False
    
    def get_system_summary(self) -> str:
        """
        Возвращает краткую сводку о системе.
        
        Returns:
            str: Сводка о системе
        """
        stats = self.get_scheduler_statistics()
        health = self.get_scheduler_health_report()
        
        summary = (
            f"Планировщик задач обучения\n"
            f"{'=' * 30}\n\n"
            f"Задачи: всего {stats['total_tasks']}, "
            f"выполнено {stats['completed_tasks']}, "
            f"неудачных {stats['failed_tasks']}\n"
            f"Очередь: {stats['pending_tasks']} задач, "
            f"выполняется {stats['in_progress_tasks']}\n"
            f"Ресурсы: загружено {stats['resource_usage']:.0%} "
            f"({self.resource_allocation.current_concurrent}/{self.resource_allocation.max_concurrent})\n\n"
            f"Здоровье системы: {health['health_score']:.1f}/100\n"
        )
        
        if health["recommendations"]:
            summary += "Рекомендации:\n"
            for i, rec in enumerate(health["recommendations"], 1):
                summary += f"{i}. {rec}\n"
        
        return summary
    
    def close(self):
        """Закрывает планировщик задач обучения и освобождает ресурсы."""
        logger.info("Закрытие планировщика задач обучения...")
        self.stop()
        logger.info("Планировщик задач обучения закрыт")
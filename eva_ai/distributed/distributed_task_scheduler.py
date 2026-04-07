"""Модуль планировщика задач для распределенной системы ЕВА"""
import logging
import threading
import time
import os
from typing import Dict, List, Optional, Any, Callable

logger = logging.getLogger("eva_ai.distributed.distributed_task_scheduler")

class Task:
    """Базовый класс для задач в распределенной системе"""
    def __init__(self, task_id: str, function: Callable, *args, **kwargs):
        self.task_id = task_id
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.status = "PENDING"
        self.result = None
        self.error = None
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None

class TaskScheduler:
    """Базовый класс планировщика задач для распределенной системы"""
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None, max_workers: int = 4):
        """
        Инициализирует планировщик задач.
        
        Args:
            brain: Ссылка на ядро ЕВА
            cache_dir: Путь к директории кэша
            max_workers: Максимальное количество рабочих потоков
        """
        self.brain = brain
        self.cache_dir = cache_dir
        self.max_workers = max_workers
        self.active_tasks: Dict[str, Task] = {}
        self.task_queue = []
        self.worker_threads = []
        self.running = False
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        
        # Создаем директорию кэша если нужно
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        logger.info(f"TaskScheduler инициализирован с {max_workers} рабочими потоками")
    
    def schedule_task(self, task_id: str, function: Callable, *args, **kwargs) -> str:
        """Запланировать выполнение задачи"""
        with self.lock:
            task = Task(task_id, function, *args, **kwargs)
            self.task_queue.append(task)
            self.active_tasks[task_id] = task
            logger.debug(f"Задача {task_id} добавлена в очередь")
            return task_id
    
    def start(self):
        """Запустить обработку задач"""
        if self.running:
            logger.warning("TaskScheduler уже запущен")
            return
        
        self.running = True
        self.stop_event.clear()
        
        # Создаем рабочие потоки
        for i in range(min(self.max_workers, len(self.task_queue) + 1)):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"TaskScheduler-Worker-{i}",
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)
        
        logger.info(f"TaskScheduler запущен с {len(self.worker_threads)} рабочими потоками")
    
    def stop(self):
        """Остановить обработку задач"""
        if not self.running:
            return
        
        self.running = False
        self.stop_event.set()
        
        # Дожидаемся завершения потоков
        for thread in self.worker_threads:
            thread.join(timeout=2.0)
        
        self.worker_threads = []
        logger.info("TaskScheduler остановлен")
    
    def _worker_loop(self):
        """Цикл рабочего потока"""
        while self.running and not self.stop_event.is_set():
            task = None
            with self.lock:
                if self.task_queue:
                    task = self.task_queue.pop(0)
                    task.status = "IN_PROGRESS"
                    task.started_at = time.time()
            
            if task:
                try:
                    logger.info(f"Выполнение задачи {task.task_id}")
                    task.result = task.function(*task.args, **task.kwargs)
                    task.status = "COMPLETED"
                    task.completed_at = time.time()
                    logger.info(f"Задача {task.task_id} завершена успешно")
                except Exception as e:
                    task.status = "FAILED"
                    task.error = str(e)
                    task.completed_at = time.time()
                    logger.error(f"Ошибка выполнения задачи {task.task_id}: {e}")
            
            # Небольшая пауза, если задач нет
            if not task:
                time.sleep(0.1)
    
    def get_task_status(self, task_id: str) -> str:
        """Получить статус задачи"""
        with self.lock:
            task = self.active_tasks.get(task_id)
            return task.status if task else "NOT_FOUND"
    
    def get_task_result(self, task_id: str):
        """Получить результат выполнения задачи"""
        with self.lock:
            task = self.active_tasks.get(task_id)
            if task and task.status == "COMPLETED":
                return task.result
            return None
    
    def get_scheduler_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику планировщика"""
        with self.lock:
            pending = sum(1 for task in self.active_tasks.values() if task.status == "PENDING")
            in_progress = sum(1 for task in self.active_tasks.values() if task.status == "IN_PROGRESS")
            completed = sum(1 for task in self.active_tasks.values() if task.status == "COMPLETED")
            failed = sum(1 for task in self.active_tasks.values() if task.status == "FAILED")
            
            return {
                "total_tasks": len(self.active_tasks),
                "pending_tasks": pending,
                "in_progress_tasks": in_progress,
                "completed_tasks": completed,
                "failed_tasks": failed,
                "worker_threads": len(self.worker_threads),
                "resource_usage": in_progress / max(1, self.max_workers),
                "timestamp": time.time()
            }
    
    def get_scheduler_health_report(self) -> Dict[str, Any]:
        """Возвращает отчет о здоровье планировщика"""
        stats = self.get_scheduler_statistics()
        
        # Рассчитываем общий показатель здоровья (0-100)
        health_score = 100.0
        
        # Учитываем количество зависших задач
        if stats["pending_tasks"] > 50:
            health_score -= min(30, (stats["pending_tasks"] - 50) * 0.5)
        
        # Учитываем частоту ошибок
        failure_rate = stats["failed_tasks"] / max(1, stats["total_tasks"])
        if failure_rate > 0.2:
            health_score -= min(40, failure_rate * 200)
        elif failure_rate > 0.1:
            health_score -= min(20, failure_rate * 100)
        
        # Учитываем использование ресурсов
        if stats["resource_usage"] > 0.9:
            health_score -= min(20, (stats["resource_usage"] - 0.9) * 200)
        
        # Формируем список проблем
        problem_areas = []
        if stats["pending_tasks"] > 50:
            problem_areas.append(f"Высокая очередь задач ({stats['pending_tasks']} задач)")
        
        if failure_rate > 0.2:
            problem_areas.append(f"Высокий уровень ошибок ({failure_rate:.1%})")
        
        if stats["resource_usage"] > 0.9:
            problem_areas.append("Высокая загрузка ресурсов")
        
        # Формируем рекомендации
        recommendations = []
        if problem_areas:
            recommendations.append("Требуется оптимизация распределения задач")
            recommendations.append("Проверьте логи на наличие повторяющихся ошибок")
        
        return {
            "health_score": max(0, min(100, health_score)),
            "statistics": stats,
            "problem_areas": problem_areas,
            "recommendations": recommendations,
            "timestamp": time.time()
        }

class SimpleTaskScheduler(TaskScheduler):
    """Простая реализация планировщика задач"""
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None, max_workers: int = 4):
        super().__init__(brain=brain, cache_dir=cache_dir, max_workers=max_workers)
        logger.info("Используется SimpleTaskScheduler как базовая реализация")
"""
ModelAccessManager - Управление доступом к модели с приоритизацией и очередью.

Решает проблемы:
1. Конфликты доступа к модели между запросами и самодиалогом
2. Приоритизация запросов (пользовательские важнее фоновых)
3. Координация через EventBus
"""

import time
import threading
import queue
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from eva_ai.core.event_bus import Event, EventPriority

logger = logging.getLogger("eva_ai.model_access")


class AccessPriority(Enum):
    """Приоритеты доступа к модели."""
    CRITICAL = 0  # Пользовательские запросы
    HIGH = 1       # Самодиалог концепты/противоречия
    NORMAL = 2     # Фоновые задачи
    LOW = 3        # Долгосрочное обучение


@dataclass
class ModelAccessRequest:
    """Запрос на доступ к модели."""
    id: str
    priority: AccessPriority
    task_type: str  # query, self_dialog, concept, contradiction, code
    callback: Callable  # Функция для выполнения с моделью
    args: Tuple
    kwargs: Dict[str, Any]
    created_at: float
    timeout: float = 60.0
    result: Any = None
    error: Optional[str] = None
    completed: bool = False


class ModelAccessManager:
    """
    Менеджер управления доступом к модели.
    
    Обеспечивает:
    - Приоритетную очередь запросов
    - Блокировку для предотвращения конфликтов
    - EventBus интеграцию для координации
    - Статистику использования
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, event_bus=None, max_workers: int = None):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        import os
        self._initialized = True
        self.event_bus = event_bus
        # OpenVINO использует CPU, оставляем 2 ядра для системы
        self.max_workers = max_workers or max(2, (os.cpu_count() or 4) - 2)
        
        self._access_lock = threading.RLock()
        self._model_busy = False
        self._current_request_id: Optional[str] = None
        
        self.request_queue = queue.PriorityQueue()
        self.active_requests: Dict[str, ModelAccessRequest] = {}
        self.completed_requests: List[ModelAccessRequest] = []
        self.max_completed_history = 100
        
        self.stats = {
            'total_requests': 0,
            'completed': 0,
            'failed': 0,
            'rejected': 0,
            'queue_size': 0,
            'avg_wait_time': 0.0
        }
        
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ModelAccess")
        self._processing_thread = None
        self._running = False
        
        self._subscribe_to_events()
        
        logger.info("ModelAccessManager initialized")
    
    def _subscribe_to_events(self):
        """Подписка на события EventBus."""
        if not self.event_bus:
            return
        
        try:
            self.event_bus.subscribe("model.request", self._on_model_request, priority=5)
            self.event_bus.subscribe("model.release", self._on_model_release, priority=5)
            self.event_bus.subscribe("model.status", self._on_model_status, priority=5)
            logger.debug("Subscribed to model events")
        except Exception as e:
            logger.debug(f"EventBus subscription error: {e}")
    
    def _on_model_request(self, event):
        """Обработка запроса на модель от EventBus."""
        data = event.data if hasattr(event, 'data') else event
        request_id = data.get('request_id')
        priority = data.get('priority', 'NORMAL')
        task_type = data.get('task_type', 'query')
        
        if request_id in self.active_requests:
            logger.debug(f"Request {request_id} already active")
            return
        
        self.stats['total_requests'] += 1
    
    def _on_model_release(self, event):
        """Обработка освобождения модели."""
        data = event.data if hasattr(event, 'data') else event
        request_id = data.get('request_id')
        if request_id == self._current_request_id:
            self._model_busy = False
            self._current_request_id = None
    
    def _on_model_status(self, event):
        """Обработка запроса статуса."""
        logger.debug(f"model_status event received")
    
    def start(self):
        """Запуск менеджера."""
        if self._running:
            return
        
        self._running = True
        self._processing_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._processing_thread.start()
        logger.info("ModelAccessManager started")
    
    def stop(self):
        """Остановка менеджера."""
        self._running = False
        if self._processing_thread:
            self._processing_thread.join(timeout=5)
        logger.info("ModelAccessManager stopped")
    
    def request_access(
        self,
        priority: AccessPriority,
        task_type: str,
        callback: Callable,
        *args,
        timeout: float = 60.0,
        **kwargs
    ) -> str:
        """
        Запросить доступ к модели.
        
        Args:
            priority: Приоритет запроса
            task_type: Тип задачи (query, self_dialog, concept, contradiction, code)
            callback: Функция для выполнения
            timeout: Таймаут ожидания (секунды)
            **kwargs: Аргументы для callback
            
        Returns:
            request_id для отслеживания
        """
        import uuid
        
        request_id = str(uuid.uuid4())[:8]
        
        request = ModelAccessRequest(
            id=request_id,
            priority=priority,
            task_type=task_type,
            callback=callback,
            args=args,
            kwargs=kwargs,
            created_at=time.time(),
            timeout=timeout
        )
        
        self.active_requests[request_id] = request
        self.stats['total_requests'] += 1
        
        self.request_queue.put((priority.value, request_id))
        
        logger.debug(f"Queued request {request_id}: priority={priority.name}, task={task_type}")
        
        if self.event_bus:
            self.event_bus.publish(Event(
                event_type="model.request",
                source="model_access_manager",
                data={
                    'request_id': request_id,
                    'priority': priority.name,
                    'task_type': task_type,
                    'queue_size': self.request_queue.qsize()
                },
                priority=EventPriority.HIGH
            ))
        
        return request_id
    
    def get_result(self, request_id: str, timeout: float = 30.0) -> Any:
        """
        Получить результат запроса.
        
        Args:
            request_id: ID запроса
            timeout: Максимальное время ожидания
            
        Returns:
            Результат выполнения или None
        """
        start = time.time()
        
        while time.time() - start < timeout:
            if request_id not in self.active_requests:
                request = self._find_completed_request(request_id)
                if request:
                    if request.error:
                        raise Exception(request.error)
                    return request.result
                return None
            
            request = self.active_requests.get(request_id)
            if request and request.completed:
                if request.error:
                    raise Exception(request.error)
                return request.result
            
            time.sleep(0.1)
        
        return None
    
    def _find_completed_request(self, request_id: str) -> Optional[ModelAccessRequest]:
        for req in self.completed_requests:
            if req.id == request_id:
                return req
        return None
    
    def _process_loop(self):
        """Основной цикл обработки запросов."""
        while self._running:
            try:
                if self.request_queue.empty():
                    time.sleep(0.1)
                    continue
                
                priority, request_id = self.request_queue.get(timeout=1.0)
                
                if request_id not in self.active_requests:
                    continue
                
                request = self.active_requests[request_id]
                
                with self._access_lock:
                    if self._model_busy:
                        self.request_queue.put((priority, request_id))
                        time.sleep(0.1)
                        continue
                    
                    self._model_busy = True
                    self._current_request_id = request_id
                
                try:
                    logger.debug(f"Processing request {request_id}: {request.task_type}")
                    
                    request.result = request.callback(*request.args, **request.kwargs)
                    request.completed = True
                    self.stats['completed'] += 1
                    
                    if self.event_bus:
                        self.event_bus.publish(Event(
                            event_type="model.completed",
                            source="model_access_manager",
                            data={
                                'request_id': request_id,
                                'task_type': request.task_type,
                                'elapsed': time.time() - request.created_at
                            },
                            priority=EventPriority.NORMAL
                        ))
                    
                except Exception as e:
                    logger.error(f"Request {request_id} failed: {e}")
                    request.error = str(e)
                    request.completed = True
                    self.stats['failed'] += 1
                    
                    if self.event_bus:
                        self.event_bus.publish(Event(
                            event_type="model.failed",
                            source="model_access_manager",
                            data={
                                'request_id': request_id,
                                'task_type': request.task_type,
                                'error': str(e)
                            },
                            priority=EventPriority.HIGH
                        ))
                
                finally:
                    with self._access_lock:
                        self._model_busy = False
                        self._current_request_id = None
                    
                    self._move_to_history(request)
                    self.stats['queue_size'] = self.request_queue.qsize()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                time.sleep(0.5)
    
    def _move_to_history(self, request: ModelAccessRequest):
        """Переместить завершённый запрос в историю."""
        self.completed_requests.append(request)
        if len(self.completed_requests) > self.max_completed_history:
            self.completed_requests.pop(0)
        
        if request.id in self.active_requests:
            del self.active_requests[request.id]
    
    def get_status(self) -> Dict[str, Any]:
        """Получить статус менеджера."""
        return {
            'running': self._running,
            'model_busy': self._model_busy,
            'current_request': self._current_request_id,
            'queue_size': self.request_queue.qsize(),
            'active_requests': len(self.active_requests),
            'stats': self.stats
        }
    
    def is_available(self) -> bool:
        """Проверить доступность модели."""
        return not self._model_busy


class ModelAccessContext:
    """
    Контекстный менеджер для безопасного доступа к модели.
    
    Usage:
        with ModelAccessContext(priority=AccessPriority.HIGH, task_type="concept") as token:
            if token.acquired:
                result = model.generate(...)
    """
    
    def __init__(
        self,
        manager: ModelAccessManager,
        priority: AccessPriority = AccessPriority.NORMAL,
        task_type: str = "query",
        timeout: float = 60.0
    ):
        self.manager = manager
        self.priority = priority
        self.task_type = task_type
        self.timeout = timeout
        self.acquired = False
        self.request_id = None
        self.event_bus = getattr(manager, 'event_bus', None)
    
    def __enter__(self) -> 'ModelAccessContext':
        self.request_id = self.manager.request_access(
            priority=self.priority,
            task_type=self.task_type,
            callback=self._dummy_callback,
            timeout=self.timeout
        )
        
        try:
            result = self.manager.get_result(self.request_id, timeout=self.timeout)
            self.acquired = result
        except Exception as e:
            self.acquired = False
            logger.debug(f"Failed to get access result: {e}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.event_bus and self.request_id:
            self.event_bus.publish(Event(
                event_type="model.release",
                source="model_access_manager",
                data={'request_id': self.request_id},
                priority=EventPriority.NORMAL
            ))
        return False
    
    def _dummy_callback(self):
        """Dummy callback для получения доступа."""
        return True

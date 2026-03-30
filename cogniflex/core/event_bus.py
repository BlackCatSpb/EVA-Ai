"""
Событийная система CogniFlex - EventBus

Центральная шина событий для координации компонентов системы.
"""

import logging
import time
import threading
import weakref
from typing import Dict, List, Callable, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict, deque
import queue

logger = logging.getLogger("cogniflex.events")

class EventPriority(Enum):
    """Приоритеты событий"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Event:
    """Базовый класс события"""
    event_type: str
    source: str
    data: Dict[str, Any]
    timestamp: float = 0
    priority: EventPriority = EventPriority.NORMAL
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()

class EventTypes:
    """Типы событий системы"""
    # Системные события
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    SYSTEM_ERROR = "system.error"
    SYSTEM_READY = "system.ready"
    
    # Компонентные события
    COMPONENT_INITIALIZED = "component.initialized"
    COMPONENT_ERROR = "component.error"
    COMPONENT_STOPPED = "component.stopped"
    COMPONENT_STARTED = "component.started"
    
    # Обучающие события
    LEARNING_STARTED = "learning.started"
    LEARNING_COMPLETED = "learning.completed"
    LEARNING_FAILED = "learning.failed"
    LEARNING_PROGRESS = "learning.progress"
    
    # Противоречия
    CONTRADICTION_DETECTED = "contradiction.detected"
    CONTRADICTION_RESOLVED = "contradiction.resolved"
    CONTRADICTION_FAILED = "contradiction.failed"
    
    # Этика
    ETHICS_VIOLATION = "ethics.violation"
    ETHICS_WARNING = "ethics.warning"
    ETHICS_ASSESSMENT = "ethics.assessment"
    
    # Аналитика
    ANALYTICS_INSIGHT = "analytics.insight"
    ANALYTICS_REPORT = "analytics.report"
    ANALYTICS_ALERT = "analytics.alert"
    
    # Знания
    KNOWLEDGE_UPDATED = "knowledge.updated"
    KNOWLEDGE_ADDED = "knowledge.added"
    KNOWLEDGE_DELETED = "knowledge.deleted"
    
    # Веб-поиск
    WEB_SEARCH_STARTED = "web.search.started"
    WEB_SEARCH_COMPLETED = "web.search.completed"
    WEB_SEARCH_FAILED = "web.search.failed"
    
    # Память
    MEMORY_CLEARED = "memory.cleared"
    MEMORY_OPTIMIZED = "memory.optimized"
    MEMORY_WARNING = "memory.warning"
    
    # Адаптация
    ADAPTATION_STARTED = "adaptation.started"
    ADAPTATION_COMPLETED = "adaptation.completed"
    ADAPTATION_FAILED = "adaptation.failed"

class EventBus:
    """Центральная шина событий CogniFlex"""
    
    def __init__(self, max_history: int = 10000):
        """
        Инициализация EventBus
        
        Args:
            max_history: Максимальный размер истории событий
        """
        self.max_history = max_history
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._event_history: deque = deque(maxlen=max_history)
        self._event_queue: queue.Queue = queue.Queue()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._stats = {
            'events_published': 0,
            'events_processed': 0,
            'events_failed': 0,
            'subscribers_count': 0,
            'start_time': time.time()
        }
        
        logger.info("EventBus инициализирован")
    
    def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> str:
        """
        Подписка на события
        
        Args:
            event_type: Тип события
            handler: Обработчик события
            
        Returns:
            str: ID подписки
        """
        with self._lock:
            # Используем weakref для автоматической очистки
            if hasattr(handler, '__self__'):
                # Для методов используем WeakMethod
                weak_handler = weakref.WeakMethod(handler)
            else:
                # Для функций используем weakref
                weak_handler = weakref.ref(handler)
            
            subscription_id = f"{event_type}_{id(handler)}"
            self._subscribers[event_type].append((subscription_id, weak_handler))
            
            self._stats['subscribers_count'] = sum(len(handlers) for handlers in self._subscribers.values())
            
            logger.debug(f"Подписка на {event_type}: {subscription_id}")
            return subscription_id
    
    def unsubscribe(self, event_type: str, handler_or_id: Callable | str) -> bool:
        """
        Отписка от событий
        
        Args:
            event_type: Тип события
            handler_or_id: Обработчик или ID подписки
            
        Returns:
            bool: Успешность отписки
        """
        with self._lock:
            if event_type not in self._subscribers:
                return False
            
            original_count = len(self._subscribers[event_type])
            
            if isinstance(handler_or_id, str):
                # Отписка по ID
                self._subscribers[event_type] = [
                    (sid, handler) for sid, handler in self._subscribers[event_type]
                    if sid != handler_or_id
                ]
            else:
                # Отписка по обработчику
                filtered = []
                for sid, handler in self._subscribers[event_type]:
                    resolved = handler()
                    if resolved is None:
                        continue
                    if resolved != handler_or_id:
                        filtered.append((sid, handler))
                self._subscribers[event_type] = filtered
            
            # Очищаем мертвые ссылки
            self._subscribers[event_type] = [
                (sid, handler) for sid, handler in self._subscribers[event_type]
                if handler() is not None
            ]
            
            self._stats['subscribers_count'] = sum(len(handlers) for handlers in self._subscribers.values())
            
            removed = original_count - len(self._subscribers[event_type])
            if removed > 0:
                logger.debug(f"Отписано от {event_type}: {removed} обработчиков")
                return True
            
            return False
    
    def publish(self, event: Event) -> bool:
        """
        Публикация события
        
        Args:
            event: Событие для публикации
            
        Returns:
            bool: Успешность публикации
        """
        try:
            # Добавляем в историю
            self._event_history.append(event)
            self._stats['events_published'] += 1
            
            # Добавляем в очередь обработки
            self._event_queue.put(event)
            
            logger.debug(f"Событие опубликовано: {event.event_type} от {event.source}")
            return True
            
        except Exception as e:
            self._stats['events_failed'] += 1
            logger.error(f"Ошибка публикации события {event.event_type}: {e}")
            return False
    
    def publish_sync(self, event: Event) -> int:
        """
        Синхронная публикация события с немедленной обработкой
        
        Args:
            event: Событие для публикации
            
        Returns:
            int: Количество обработанных подписчиков
        """
        try:
            # Добавляем в историю
            self._event_history.append(event)
            self._stats['events_published'] += 1
            
            # Обрабатываем немедленно
            processed = self._process_event(event)
            
            logger.debug(f"Синхронное событие обработано: {event.event_type}, обработчиков: {processed}")
            return processed
            
        except Exception as e:
            self._stats['events_failed'] += 1
            logger.error(f"Ошибка синхронной публикации события {event.event_type}: {e}")
            return 0
    
    def _process_event(self, event: Event) -> int:
        """
        Обработка события
        
        Args:
            event: Событие для обработки
            
        Returns:
            int: Количество обработанных подписчиков
        """
        processed = 0
        
        with self._lock:
            subscribers = self._subscribers.get(event.event_type, [])
            
            # Очищаем мертвые ссылки
            active_subscribers = []
            for subscription_id, weak_handler in subscribers:
                handler = weak_handler()
                if handler is not None:
                    active_subscribers.append((subscription_id, handler))
            
            # Обновляем список активных подписчиков
            self._subscribers[event.event_type] = active_subscribers
            
            # Вызываем обработчики
            for subscription_id, handler in active_subscribers:
                try:
                    handler(event)
                    processed += 1
                except Exception as e:
                    self._stats['events_failed'] += 1
                    logger.error(f"Ошибка в обработчике {subscription_id} для события {event.event_type}: {e}")
            
            self._stats['events_processed'] += 1
        
        return processed
    
    def _worker_loop(self):
        """Рабочий цикл обработки событий"""
        logger.info("EventBus worker loop запущен")
        
        while self._running:
            try:
                # Получаем событие с таймаутом
                try:
                    event = self._event_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Обрабатываем событие
                self._process_event(event)
                
                # Помечаем задачу как выполненную
                self._event_queue.task_done()
                
            except Exception as e:
                logger.error(f"Ошибка в worker loop EventBus: {e}")
        
        logger.info("EventBus worker loop остановлен")
    
    def start(self):
        """Запуск EventBus"""
        if self._running:
            logger.warning("EventBus уже запущен")
            return
        
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        
        logger.info("EventBus запущен")
    
    def stop(self):
        """Остановка EventBus"""
        if not self._running:
            logger.warning("EventBus уже остановлен")
            return
        
        self._running = False
        
        # Ждем завершения worker thread
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        
        # Очищаем очередь
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("EventBus остановлен")
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики EventBus"""
        uptime = time.time() - self._stats['start_time']
        queue_size = self._event_queue.qsize()
        
        return {
            **self._stats,
            'uptime': uptime,
            'queue_size': queue_size,
            'history_size': len(self._event_history),
            'events_per_second': self._stats['events_published'] / uptime if uptime > 0 else 0
        }
    
    def get_recent_events(self, count: int = 100) -> List[Event]:
        """Получение последних событий"""
        with self._lock:
            return list(self._event_history)[-count:]
    
    def get_events_by_type(self, event_type: str, count: int = 100) -> List[Event]:
        """Получение событий по типу"""
        with self._lock:
            events = [
                event for event in self._event_history
                if event.event_type == event_type
            ]
            return events[-count:]
    
    def clear_history(self):
        """Очистка истории событий"""
        with self._lock:
            self._event_history.clear()
        logger.info("История событий EventBus очищена")
    
    def get_subscribers_count(self, event_type: Optional[str] = None) -> int:
        """Получение количества подписчиков"""
        with self._lock:
            if event_type:
                return len([h for h in self._subscribers.get(event_type, []) if h[1]() is not None])
            else:
                return sum(
                    len([h for h in handlers if h[1]() is not None])
                    for handlers in self._subscribers.values()
                )

# Глобальный экземпляр EventBus
_global_event_bus: Optional[EventBus] = None

def get_event_bus() -> EventBus:
    """Получение глобального экземпляра EventBus"""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
        _global_event_bus.start()
    return _global_event_bus

def reset_event_bus():
    """Сброс глобального экземпляра EventBus"""
    global _global_event_bus
    if _global_event_bus is not None:
        _global_event_bus.stop()
    _global_event_bus = None

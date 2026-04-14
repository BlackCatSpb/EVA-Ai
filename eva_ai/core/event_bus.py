"""
Событийная система ЕВА - EventBus

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

logger = logging.getLogger("eva_ai.events")

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
    KNOWLEDGE_GRAPH_UPDATED = "knowledge.graph.updated"
    
    # Память
    MEMORY_CLEARED = "memory.cleared"
    MEMORY_OPTIMIZED = "memory.optimized"
    MEMORY_WARNING = "memory.warning"
    MEMORY_GRAPH_UPDATED = "memory.graph.updated"
    
    # Обучение
    LEARNING_STATS_UPDATED = "learning.stats.updated"
    LEARNING_OPPORTUNITIES_UPDATED = "learning.opportunities.updated"
    LEARNING_DIALOGS_UPDATED = "learning.dialogs.updated"
    
    # Адаптация
    ADAPTATION_STARTED = "adaptation.started"
    ADAPTATION_COMPLETED = "adaptation.completed"
    ADAPTATION_FAILED = "adaptation.failed"
    
    # Two-Model Pipeline (GGUF Model A + Model B)
    PIPELINE_START = "pipeline.start"
    PIPELINE_MODEL_A_START = "pipeline.model_a.start"
    PIPELINE_MODEL_A_COMPLETE = "pipeline.model_a.complete"
    PIPELINE_WEB_SEARCH_COMPLETE = "pipeline.web_search.complete"
    PIPELINE_CONTRADICTION_CHECK_COMPLETE = "pipeline.contradiction.check_complete"
    PIPELINE_ETHICS_CHECK_COMPLETE = "pipeline.ethics.check_complete"
    PIPELINE_MODEL_B_START = "pipeline.model_b.start"
    PIPELINE_MODEL_B_COMPLETE = "pipeline.model_b.complete"
    PIPELINE_RELEVANCE_CHECK_COMPLETE = "pipeline.relevance.check_complete"
    PIPELINE_REFINEMENT_NEEDED = "pipeline.refinement.needed"
    PIPELINE_REFINEMENT_ATTEMPT = "pipeline.refinement.attempt"
    PIPELINE_COMPLETE = "pipeline.complete"
    PIPELINE_FAILED = "pipeline.failed"
    
    COMMAND_COMPLETED = "command.completed"
    COMMAND_FAILED = "command.failed"
    
    # Concept Miner - концептуальный вывод
    CONCEPT_MINING_START = "concept.mining.start"
    CONCEPT_MINING_COMPLETE = "concept.mining.complete"
    CONCEPT_MINING_FAILED = "concept.mining.failed"
    CONCEPT_CANDIDATE_GENERATED = "concept.candidate.generated"
    CONCEPT_VALIDATION_COMPLETE = "concept.validation.complete"
    CONCEPT_WEB_VERIFY = "concept.web.verify"
    CONCEPT_LIFECYCLE_UPDATE = "concept.lifecycle.update"
    MEMORY_CLUSTERING_COMPLETE = "memory.clustering.complete"

class EventBus:
    """Центральная шина событий ЕВА"""
    
    def __init__(self, max_history: int = 10000):
        """
        Инициализация EventBus
        
        Args:
            max_history: Максимальный размер истории событий
        """
        self.max_history = max_history
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._event_history: deque = deque(maxlen=max_history)
        self._event_queue: queue.PriorityQueue = queue.PriorityQueue()
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
        
        logger.info("EventBus инициализирован с PriorityQueue")
    
    def subscribe(self, event_type: str, handler: Callable[[Event], None], priority: int = 5) -> str:
        """
        Подписка на события с приоритетом.
        
        Args:
            event_type: Тип события
            handler: Обработчик события
            priority: Приоритет обработчика (0=высший, 255=низший, по умолчанию 128)
            
        Returns:
            str: ID подписки
        """
        handler_name = getattr(handler, '__name__', str(handler))
        logger.info("=== EVENT BUS: Subscribe ===")
        logger.info("  Event type: {}".format(event_type))
        logger.info("  Handler: {}".format(handler_name))
        logger.info("  Priority: {} (0=highest, 255=lowest)".format(priority))
        
        with self._lock:
            # Используем weakref для автоматической очистки
            if hasattr(handler, '__self__'):
                # Для методов используем WeakMethod
                weak_handler = weakref.WeakMethod(handler)
                handler_type = "method"
            else:
                # Для функций используем weakref
                weak_handler = weakref.ref(handler)
                handler_type = "function"
            
            # priority хранится вместе с подпиской для сортировки
            subscription_id = "{}::{}::{}".format(event_type, handler_type, id(handler))
            self._subscribers[event_type].append((subscription_id, weak_handler, priority))
            
            # СОРТИРУЕМ подписчиков по приоритету (меньше = выше приоритет)
            self._subscribers[event_type].sort(key=lambda x: x[2])
            
            self._stats['subscribers_count'] = sum(len(handlers) for handlers in self._subscribers.values())
            
            logger.info("SUBSCRIBED: {} -> {} (id: {}, priority: {})".format(
                event_type, handler_name, subscription_id, priority))
            logger.debug("  Total subscribers for {}: {}".format(
                event_type, len(self._subscribers[event_type])))
            
            return subscription_id
    
    def unsubscribe(self, event_type: str, handler_or_id: Callable | str) -> bool:
        """
        Отписка от событий с логированием
        
        Args:
            event_type: Тип события
            handler_or_id: Обработчик или ID подписки
            
        Returns:
            bool: Успешность отписки
        """
        logger.info("=== EVENT BUS: Unsubscribe ===")
        logger.info("  Event type: {}".format(event_type))
        
        with self._lock:
            if event_type not in self._subscribers:
                logger.warning("  No subscribers for event type {}".format(event_type))
                return False
            
            original_count = len(self._subscribers[event_type])
            
            if isinstance(handler_or_id, str):
                # Отписка по ID
                self._subscribers[event_type] = [
                    (sid, handler, prio) for sid, handler, prio in self._subscribers[event_type]
                    if sid != handler_or_id
                ]
            else:
                # Отписка по обработчику
                filtered = []
                for sid, handler, prio in self._subscribers[event_type]:
                    resolved = handler()
                    if resolved is None:
                        continue
                    if resolved != handler_or_id:
                        filtered.append((sid, handler, prio))
                self._subscribers[event_type] = filtered
            
            self._stats['subscribers_count'] = sum(len(handlers) for handlers in self._subscribers.values())
            
            removed = original_count - len(self._subscribers[event_type])
            if removed > 0:
                logger.debug(f"Отписано от {event_type}: {removed} обработчиков")
                return True
            
            return False
    
    def _cleanup_dead_subscribers(self):
        """Периодическая очистка мертвых ссылок."""
        with self._lock:
            for event_type in list(self._subscribers.keys()):
                self._subscribers[event_type] = [
                    (sid, handler, prio) for sid, handler, prio in self._subscribers[event_type]
                    if handler() is not None
                ]
    
    def publish(self, event: Event) -> bool:
        """
        Публикация события с приоритетной обработкой.
        
        Args:
            event: Событие для публикации
            
        Returns:
            bool: Успешность публикации
        """
        logger.debug("=== EVENT BUS: Publishing event ===")
        logger.debug("  Event type: {}".format(event.event_type))
        logger.debug("  Source: {}".format(event.source))
        logger.debug("  Priority: {}".format(event.priority))
        
        try:
            with self._lock:
                self._event_history.append(event)
                self._stats['events_published'] += 1
            
            priority_value = 5 - event.priority.value
            self._event_queue.put((priority_value, event.timestamp, event))
            
            logger.info("EVENT published: {} from {}".format(event.event_type, event.source))
            logger.debug("  Queue size: {}".format(self._event_queue.qsize()))
            return True
            
        except Exception as e:
            with self._lock:
                self._stats['events_failed'] += 1
            logger.error("Ошибка публикации события {}: {}".format(event.event_type, e))
            import traceback
            logger.error("  Traceback: {}".format(traceback.format_exc()))
            return False
    
    def publish_sync(self, event: Event) -> int:
        """
        Синхронная публикация события с немедленной обработкой и логированием
        
        Args:
            event: Событие для публикации
            
        Returns:
            int: Количество обработанных подписчиков
        """
        logger.debug("=== EVENT BUS: Publishing SYNC event ===")
        logger.debug("  Event type: {}".format(event.event_type))
        logger.debug("  Source: {}".format(event.source))
        logger.debug("  Data: {}".format(event.data))
        
        try:
            with self._lock:
                self._event_history.append(event)
                self._stats['events_published'] += 1
            
            logger.info("EVENT sync: {} from {}".format(event.event_type, event.source))
            
            processed = self._process_event(event)
            
            logger.info("SYNC event {} processed, {} handlers".format(event.event_type, processed))
            return processed
            
        except Exception as e:
            with self._lock:
                self._stats['events_failed'] += 1
            logger.error("Ошибка синхронной публикации события {}: {}".format(event.event_type, e))
            import traceback
            logger.error("  Traceback: {}".format(traceback.format_exc()))
            return 0
    
    def _process_event(self, event: Event) -> int:
        """
        Обработка события с подробным логированием
        
        Args:
            event: Событие для обработки
            
        Returns:
            int: Количество обработанных подписчиков
        """
        processed = 0
        
        logger.debug("=== EVENT BUS: Processing event ===")
        logger.debug("  Event type: {}".format(event.event_type))
        logger.debug("  Source: {}".format(event.source))
        logger.debug("  Data: {}".format(event.data))
        logger.debug("  Priority: {}".format(event.priority))
        
        with self._lock:
            subscribers = self._subscribers.get(event.event_type, [])
            
            logger.debug("  Raw subscribers count: {}".format(len(subscribers)))
            
            # Очищаем мертвые ссылки
            active_subscribers = []
            dead_subscribers = 0
            for subscription_id, weak_handler, priority in subscribers:
                handler = weak_handler()
                if handler is not None:
                    active_subscribers.append((subscription_id, weak_handler, priority))
                else:
                    dead_subscribers += 1
            
            if dead_subscribers > 0:
                logger.debug("  Cleaned {} dead subscribers".format(dead_subscribers))
            
            # Обновляем список активных подписчиков
            self._subscribers[event.event_type] = active_subscribers
            
            logger.debug("  Active subscribers: {}".format(len(active_subscribers)))
            
            # Вызываем обработчики (уже отсортированы по priority)
            for subscription_id, weak_handler, priority in active_subscribers:
                try:
                    handler = weak_handler()
                    if handler is None:
                        logger.debug("  Handler {} is dead, skipping".format(subscription_id))
                        continue
                    
                    handler_name = getattr(handler, '__name__', str(handler))
                    handler_type = type(handler)
                    logger.info("  Calling handler: {} (type: {}) for event {}".format(handler_name, handler_type, event.event_type))
                    logger.info("  Handler is bound method: {}".format(hasattr(handler, '__self__')))
                    if hasattr(handler, '__self__'):
                        logger.info("  Handler self: {}".format(handler.__self__))
                    
                    # Проверяем тип handler для отладки
                    if callable(handler):
                        logger.info("  Handler {} is callable, calling with event={}".format(handler_name, event))
                        # ЯВНЫЙ вызов для отладки
                        try:
                            result = handler(event)
                            logger.info("  Handler {} returned: {}".format(handler_name, result))
                        except TypeError as te:
                            logger.error("  TypeError when calling handler: {}".format(te))
                            raise
                    else:
                        logger.warning("  Handler {} is not callable: {}".format(handler_name, type(handler)))
                        continue
                        
                    processed += 1
                    logger.info("  Handler {} processed successfully".format(handler_name))
                except TypeError as te:
                    self._stats['events_failed'] += 1
                    logger.error("Ошибка вызова обработчика {} для события {}: {}".format(
                        subscription_id, event.event_type, te))
                    logger.error("  Handler type: {}, Handler: {}".format(type(handler) if 'handler' in dir() else 'N/A', handler if 'handler' in dir() else 'N/A'))
                    import traceback
                    logger.error("  Traceback: {}".format(traceback.format_exc()))
                except Exception as e:
                    self._stats['events_failed'] += 1
                    logger.error("Ошибка в обработчике {} для события {}: {}".format(
                        subscription_id, event.event_type, e))
                    import traceback
                    logger.error("  Traceback: {}".format(traceback.format_exc()))
            
            self._stats['events_processed'] += 1
        
        logger.debug("=== EVENT BUS: Processed {} handlers for event {} ===".format(
            processed, event.event_type))
        
        return processed
    
    def _worker_loop(self):
        """Рабочий цикл обработки событий с приоритетами"""
        logger.info("EventBus worker loop запущен с PriorityQueue")
        
        while self._running:
            try:
                try:
                    item = self._event_queue.get(timeout=1.0)
                    if isinstance(item, tuple) and len(item) == 3:
                        _, _, event = item
                    else:
                        event = item
                except queue.Empty:
                    continue
                
                self._process_event(event)
                
                # Помечаем задачу как выполненную
                self._event_queue.task_done()
                
            except Exception as e:
                logger.error(f"Ошибка в worker loop EventBus: {e}")
                import traceback
                logger.error(f"  Traceback: {traceback.format_exc()}")
        
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
    if hasattr(_global_event_bus, 'start') and not getattr(_global_event_bus, '_running', False):
        _global_event_bus.start()
    return _global_event_bus

def reset_event_bus():
    """Сброс глобального экземпляра EventBus"""
    global _global_event_bus
    if _global_event_bus is not None:
        _global_event_bus.stop()
    _global_event_bus = None

"""
Событийная система для ЕВА - управление уведомлениями о готовности компонентов
и детальный таймлайн событий/обработчиков для отладки распределения времени/ресурсов.
"""
import logging
import time
import threading
import weakref
from typing import Dict, List, Callable, Any, Deque, Optional
from collections import defaultdict, deque

logger = logging.getLogger("eva.events")
timeline_logger = logging.getLogger("eva.events.timeline")


class _WrappedCallback:
    """Opaque wrapper storing callback metadata without polluting function attrs."""
    __slots__ = ('original_callback', '_event_priority', '_event_name')

    def __init__(self, callback: Callable, event_name: str, priority: int):
        self.original_callback = callback
        self._event_priority = priority
        self._event_name = event_name

    def __call__(self, data: Any) -> None:
        try:
            self.original_callback(data)
        except Exception as e:
            logger.error(f"Ошибка в обработчике события {self._event_name}: {e}")

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _WrappedCallback):
            return self.original_callback == other.original_callback
        return self.original_callback == other

    def __hash__(self) -> int:
        return hash(self.original_callback)


class EventBus:
    """
    Централизованная событийная шина для координации всех компонентов системы.

    Поддерживаемые ключевые события:
    - 'query_received' - получен новый запрос от пользователя
    - 'tokenize_request' - запрос на токенизацию
    - 'tokens_ready' - токенизация завершена
    - 'hot_window_ready' - горячее окно сформировано
    - 'response_generated' - ответ сгенерирован
    - 'contradiction_detected' - обнаружено противоречие в знаниях
    - 'system_health_check' - запрос на проверку состояния системы
    - 'self_dialog_request' - запрос на запуск самодиалога
    - 'learning_opportunity' - выявлена возможность обучения
    - 'ethical_check_request' - запрос на этическую проверку
    - 'memory_optimization' - запрос на оптимизацию памяти
    - 'component_initialization_requested' - запрошена инициализация компонента
    - 'component_initialized' - компонент успешно инициализирован
    - 'component_initialization_failed' - ошибка инициализации компонента
    """

    def __init__(self, timeline_maxlen: int = 1000, enable_timeline: bool = True):
        self.listeners: Dict[str, List[Callable]] = defaultdict(list)
        self._triggered_events: Dict[str, Any] = {}
        self._timeline: Deque[Dict[str, Any]] = deque(maxlen=max(100, int(timeline_maxlen)))
        self._enable_timeline: bool = bool(enable_timeline)
        self._seq: int = 0
        self._lock = threading.RLock()

        # Приоритеты событий для обработки в порядке важности
        self.event_priorities = {
            'system_health_check': 10,  # Максимальный приоритет
            'contradiction_detected': 9,
            'component_initialization_failed': 8,  # Высокий приоритет для ошибок
            'query_received': 7,
            'component_initialized': 6,  # Уведомление об успешной инициализации
            'ethical_check_request': 5,
            'component_initialization_requested': 4,  # Запрос инициализации
            'tokenize_request': 3,
            'tokens_ready': 2,
            'hot_window_ready': 1,
            'learning_opportunity': 0,
            'self_dialog_request': -1,
            'response_generated': -2,
            'memory_optimization': -3  # Минимальный приоритет
        }

    def subscribe(self, event_name: str, callback: Callable, priority: int = 5):
        """
        Подписывается на событие с указанием приоритета.

        Args:
            event_name: Имя события
            callback: Функция обратного вызова
            priority: Приоритет обработки (0-10, выше = важнее)
        """
        wrapped_callback = _WrappedCallback(callback, event_name, priority)

        self.listeners[event_name].append(wrapped_callback)

        if event_name in self._triggered_events:
            try:
                wrapped_callback(data=self._triggered_events[event_name])
            except Exception as e:
                logger.error(f"Ошибка в обработчике события {event_name}: {e}")

    def unsubscribe(self, event_name: str, callback: Callable):
        """Отписывается от события."""
        if event_name not in self.listeners:
            return
        for wcb in list(self.listeners[event_name]):
            orig = wcb.original_callback if isinstance(wcb, _WrappedCallback) else getattr(wcb, '_original_callback', wcb)
            if orig == callback or wcb == callback:
                self.listeners[event_name].remove(wcb)
                return

    def trigger(self, event_name: str, data: Any = None, priority_override: Optional[int] = None):
        """
        Вызывает событие и уведомляет всех подписчиков в порядке приоритета.

        Args:
            event_name: Имя события
            data: Данные для передачи подписчикам
            priority_override: Переопределение приоритета события
        """
        with self._lock:
            self._triggered_events[event_name] = data

        listeners = list(self.listeners.get(event_name, []))

        # Сортируем обработчики по приоритету
        event_priority = priority_override if priority_override is not None else self.event_priorities.get(event_name, 5)
        
        # Use priority_override as fallback, but prefer callback's own priority
        if priority_override is not None:
            listeners.sort(key=lambda cb: getattr(cb, '_event_priority', priority_override), reverse=True)
        else:
            listeners.sort(key=lambda cb: getattr(cb, '_event_priority', 5), reverse=True)

        ts = time.time()
        with self._lock:
            self._seq += 1
            seq = self._seq

        if self._enable_timeline:
            record = {
                "type": "event_triggered",
                "seq": seq,
                "ts": ts,
                "event": event_name,
                "listeners": len(listeners),
                "priority": event_priority,
            }
            self._timeline.append(record)
            try:
                timeline_logger.debug(record)
            except Exception:
                pass

        for cb in listeners:
            cb_name = getattr(cb.original_callback if isinstance(cb, _WrappedCallback) else getattr(cb, '_original_callback', cb), "__name__", str(cb))
            start = time.perf_counter()

            if self._enable_timeline:
                with self._lock:
                    self._seq += 1
                    hseq = self._seq
                hrec = {
                    "type": "handler_start",
                    "seq": hseq,
                    "ts": time.time(),
                    "event": event_name,
                    "handler": cb_name,
                    "priority": getattr(cb, '_event_priority', 5),
                }
                self._timeline.append(hrec)
                try:
                    timeline_logger.debug(hrec)
                except Exception:
                    pass

            try:
                cb(data)
            except Exception as e:
                logger.error(f"Ошибка в обработчике события {event_name}: {e}")
            finally:
                if self._enable_timeline:
                    dur = time.perf_counter() - start
                    with self._lock:
                        self._seq += 1
                        dseq = self._seq
                    done = {
                        "type": "handler_done",
                        "seq": dseq,
                        "ts": time.time(),
                        "event": event_name,
                        "handler": cb_name,
                        "duration_s": round(dur, 6),
                        "priority": getattr(cb, '_event_priority', 5),
                    }
                    self._timeline.append(done)
                    try:
                        timeline_logger.debug(done)
                    except Exception:
                        pass

    def is_triggered(self, event_name: str) -> bool:
        """Проверяет, было ли событие уже вызвано."""
        return event_name in self._triggered_events

    def get_event_data(self, event_name: str) -> Any:
        """Получает данные события, если оно было вызвано."""
        return self._triggered_events.get(event_name)

    def get_event_stats(self) -> Dict[str, Any]:
        """Возвращает статистику по событиям."""
        stats = {
            "total_events_triggered": len(self._triggered_events),
            "total_listeners": sum(len(listeners) for listeners in self.listeners.values()),
            "timeline_entries": len(self._timeline),
            "events_by_priority": {}
        }

        # Группируем события по приоритетам
        for event_name in self._triggered_events:
            priority = self.event_priorities.get(event_name, 5)
            if priority not in stats["events_by_priority"]:
                stats["events_by_priority"][priority] = []
            stats["events_by_priority"][priority].append(event_name)

        return stats

    def clear_triggered_events(self):
        """Очищает историю вызванных событий."""
        self._triggered_events.clear()

    # --- Таймлайн ---
    def get_timeline(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Возвращает копию последних записей таймлайна."""
        try:
            if limit is None or limit <= 0:
                return list(self._timeline)
            return list(self._timeline)[-int(limit):]
        except Exception:
            return []

    def clear_timeline(self) -> None:
        """Очищает таймлайн событий."""
        try:
            self._timeline.clear()
        except Exception:
            pass

    def enable_timeline(self, enable: bool = True) -> None:
        self._enable_timeline = bool(enable)

    def get_event_flow(self, event_name: str) -> List[Dict[str, Any]]:
        """Возвращает поток обработки конкретного события."""
        try:
            flow = []
            for record in self._timeline:
                if record.get("event") == event_name:
                    flow.append(record)
            return flow
        except Exception:
            return []

    def get_performance_stats(self) -> Dict[str, Any]:
        """Возвращает статистику производительности обработки событий."""
        try:
            stats = {
                "total_processing_time": 0.0,
                "average_handler_time": 0.0,
                "slowest_handler": None,
                "fastest_handler": None,
                "event_counts": {}
            }

            handler_times = []

            for record in self._timeline:
                if record.get("type") == "handler_done":
                    duration = record.get("duration_s", 0)
                    handler_times.append(duration)
                    stats["total_processing_time"] += duration

                    # Обновляем slowest/fastest
                    if stats["slowest_handler"] is None or duration > stats["slowest_handler"]["duration"]:
                        stats["slowest_handler"] = {
                            "handler": record.get("handler", "unknown"),
                            "duration": duration
                        }
                    if stats["fastest_handler"] is None or duration < stats["fastest_handler"]["duration"]:
                        stats["fastest_handler"] = {
                            "handler": record.get("handler", "unknown"),
                            "duration": duration
                        }

            if handler_times:
                stats["average_handler_time"] = stats["total_processing_time"] / len(handler_times)

            # Подсчитываем события по типам
            for record in self._timeline:
                event_type = record.get("type", "unknown")
                stats["event_counts"][event_type] = stats["event_counts"].get(event_type, 0) + 1

            return stats

        except Exception as e:
            logger.error(f"Ошибка при сборе статистики производительности: {e}")
            return {"error": str(e)}


class ComponentInitializationManager:
    """
    Менеджер инициализации компонентов с использованием событийной системы.

    Предотвращает повторную инициализацию уже инициализированных компонентов.
    Координирует инициализацию через EventBus.
    """

    def __init__(self, event_bus: EventBus):
        """
        Инициализирует менеджер инициализации компонентов.

        Args:
            event_bus: Экземпляр EventBus для координации
        """
        self.event_bus = event_bus
        self._initialized_components: Dict[str, Dict[str, Any]] = {}
        self._initialization_locks: Dict[str, threading.Lock] = {}
        self._logger = logging.getLogger("eva.component_init")

        # Подписываемся на события инициализации
        self.event_bus.subscribe("component_initialized", self._on_component_initialized, priority=10)
        self.event_bus.subscribe("component_initialization_failed", self._on_component_initialization_failed, priority=9)

    def initialize_component(self, component_name: str, component_instance: Any,
                           init_function: Callable, *args, **kwargs) -> bool:
        """
        Инициализирует компонент, если он еще не инициализирован.

        Args:
            component_name: Уникальное имя компонента
            component_instance: Экземпляр компонента
            init_function: Функция инициализации
            *args, **kwargs: Аргументы для функции инициализации

        Returns:
            bool: True если инициализация прошла успешно или компонент уже инициализирован
        """
        # Проверяем, уже ли инициализирован компонент
        if self.is_component_initialized(component_name):
            self._logger.debug(f"Компонент {component_name} уже инициализирован, пропускаем")
            return True

        # Создаем лок для этого компонента
        if component_name not in self._initialization_locks:
            self._initialization_locks[component_name] = threading.Lock()

        with self._initialization_locks[component_name]:
            # Повторная проверка внутри лока
            if self.is_component_initialized(component_name):
                return True

            # Триггерим событие запроса инициализации
            self.event_bus.trigger("component_initialization_requested", {
                "component_name": component_name,
                "component_instance": component_instance,
                "timestamp": time.time()
            })

            try:
                self._logger.info(f"Инициализируем компонент: {component_name}")

                # Вызываем функцию инициализации
                result = init_function(*args, **kwargs)

                if result is True or result is None:  # Успешная инициализация
                    # Сохраняем информацию об инициализированном компоненте
                    self._initialized_components[component_name] = {
                        "instance": component_instance,
                        "initialized_at": time.time(),
                        "init_result": result
                    }

                    # Триггерим событие успешной инициализации
                    self.event_bus.trigger("component_initialized", {
                        "component_name": component_name,
                        "component_instance": component_instance,
                        "timestamp": time.time()
                    })

                    self._logger.info(f"Компонент {component_name} готов")
                    return True
                else:
                    raise RuntimeError(f"Функция инициализации вернула {result}")

            except Exception as e:
                error_msg = f"Ошибка инициализации компонента {component_name}: {str(e)}"
                self._logger.error(error_msg, exc_info=True)

                # Триггерим событие ошибки инициализации
                self.event_bus.trigger("component_initialization_failed", {
                    "component_name": component_name,
                    "component_instance": component_instance,
                    "error": str(e),
                    "timestamp": time.time()
                })

                return False

    def is_component_initialized(self, component_name: str) -> bool:
        """
        Проверяет, инициализирован ли компонент.

        Args:
            component_name: Имя компонента

        Returns:
            bool: True если компонент инициализирован
        """
        return component_name in self._initialized_components

    def get_initialized_component(self, component_name: str) -> Optional[Any]:
        """
        Получает инициализированный компонент.

        Args:
            component_name: Имя компонента

        Returns:
            Компонент или None если не инициализирован
        """
        if self.is_component_initialized(component_name):
            return self._initialized_components[component_name]["instance"]
        return None

    def get_initialization_status(self) -> Dict[str, Any]:
        """
        Возвращает статус инициализации всех компонентов.

        Returns:
            Dict с информацией о статусе инициализации
        """
        status = {
            "total_components": len(self._initialized_components),
            "initialized_components": list(self._initialized_components.keys()),
            "component_details": {}
        }

        for name, info in self._initialized_components.items():
            status["component_details"][name] = {
                "initialized_at": info["initialized_at"],
                "init_result": info["init_result"]
            }

        return status

    def reset_component(self, component_name: str) -> bool:
        """
        Сбрасывает статус инициализации компонента.

        Args:
            component_name: Имя компонента

        Returns:
            bool: True если компонент был сброшен
        """
        if component_name in self._initialized_components:
            del self._initialized_components[component_name]
            self._logger.info(f"Сброшен статус инициализации компонента: {component_name}")
            return True
        return False

    def _on_component_initialized(self, event_data: Dict[str, Any]):
        """
        Обработчик события успешной инициализации компонента.
        """
        if event_data and "component_name" in event_data:
            component_name = event_data["component_name"]
            self._logger.debug(f"Получено уведомление об инициализации компонента: {component_name}")

    def _on_component_initialization_failed(self, event_data: Dict[str, Any]):
        """
        Обработчик события ошибки инициализации компонента.
        """
        if event_data and "component_name" in event_data:
            component_name = event_data["component_name"]
            error = event_data.get("error", "неизвестная ошибка")
            self._logger.warning(f"Получено уведомление об ошибке инициализации компонента {component_name}: {error}")


class EventSystem:
    """
    Обертка над EventBus для обратной совместимости.
    
    Предоставляет унифицированный интерфейс для работы с событиями в ЕВА.
    """
    
    def __init__(self, timeline_maxlen: int = 1000, enable_timeline: bool = True):
        """
        Инициализирует событийную систему.
        
        Args:
            timeline_maxlen: Максимальное количество записей в таймлайне
            enable_timeline: Включить/выключить таймлайн событий
        """
        self.event_bus = EventBus(timeline_maxlen=timeline_maxlen, enable_timeline=enable_timeline)
        self.component_manager = ComponentInitializationManager(self.event_bus)
        self._logger = logging.getLogger("eva.event_system")
    
    def subscribe(self, event_name: str, callback: Callable, priority: int = 5):
        """Подписывается на событие."""
        return self.event_bus.subscribe(event_name, callback, priority)
    
    def trigger(self, event_name: str, data: Any = None, priority_override: Optional[int] = None):
        """Вызывает событие."""
        return self.event_bus.trigger(event_name, data, priority_override)
    
    def unsubscribe(self, event_name: str, callback: Callable):
        """Отписывается от события."""
        return self.event_bus.unsubscribe(event_name, callback)
    
    def is_triggered(self, event_name: str) -> bool:
        """Проверяет, было ли событие вызвано."""
        return self.event_bus.is_triggered(event_name)
    
    def get_event_data(self, event_name: str) -> Any:
        """Получает данные события."""
        return self.event_bus.get_event_data(event_name)
    
    def get_event_stats(self) -> Dict[str, Any]:
        """Возвращает статистику по событиям."""
        return self.event_bus.get_event_stats()
    
    def get_timeline(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Возвращает таймлайн событий."""
        return self.event_bus.get_timeline(limit)
    
    def clear_timeline(self) -> None:
        """Очищает таймлайн."""
        return self.event_bus.clear_timeline()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Возвращает статистику производительности."""
        return self.event_bus.get_performance_stats()
    
    def initialize_component(self, component_name: str, component_instance: Any,
                           init_function: Callable, *args, **kwargs) -> bool:
        """Инициализирует компонент через менеджер."""
        return self.component_manager.initialize_component(
            component_name, component_instance, init_function, *args, **kwargs
        )
    
    def is_component_initialized(self, component_name: str) -> bool:
        """Проверяет, инициализирован ли компонент."""
        return self.component_manager.is_component_initialized(component_name)
    
    def get_initialized_component(self, component_name: str) -> Optional[Any]:
        """Получает инициализированный компонент."""
        return self.component_manager.get_initialized_component(component_name)


# Экспорт для совместимости
__all__ = ['EventBus', 'EventSystem', 'ComponentInitializationManager']

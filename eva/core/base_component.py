"""
Базовый класс для всех компонентов системы ЕВА с событийной интеграцией

Этот модуль предоставляет базовый класс для всех компонентов, которые взаимодействуют с CoreBrain
и поддерживают событийную архитектуру.
"""
import logging
import time
import threading
from typing import Optional, Any, Dict, List, Type, TypeVar, Generic, Callable, Set
from abc import ABC, abstractmethod
from enum import Enum

from .event_bus import EventBus, Event, EventTypes, get_event_bus
from ..security.security_framework import get_security_manager

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ComponentState(Enum):
    """Состояния компонента"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

class BaseComponent(ABC):
    """
    Базовый класс для всех компонентов системы ЕВА с событийной интеграцией.
    
    Атрибуты:
        brain: Ссылка на основной объект CoreBrain
        event_bus: Шина событий для компонента
        is_initialized (bool): Флаг инициализации компонента
        dependencies (List[str]): Список имен зависимостей компонента
        config (Dict[str, Any]): Конфигурация компонента
    """
    
    def __init__(self, 
                 brain: Optional[Any] = None, 
                 config: Optional[Dict[str, Any]] = None,
                 event_bus: Optional[EventBus] = None,
                 name: Optional[str] = None):
        """
        Инициализация базового компонента.
        
        Args:
            brain: Ссылка на основной объект CoreBrain
            config: Конфигурация компонента
            event_bus: Шина событий (опционально)
            name: Имя компонента (опционально)
            config: Конфигурация компонента
        """
        self.brain = brain
        self.is_initialized = False
        self.config = config or {}
        self.dependencies: List[str] = []
        self._required_components: List[str] = []
        self._optional_components: List[str] = []
        
        # Событийная система
        self.name = name or self.__class__.__name__
        self.event_bus = event_bus or get_event_bus()
        self.security_manager = get_security_manager()
        self._state = ComponentState.UNINITIALIZED
        self._lock = threading.RLock()
        self._start_time = 0.0
        self._last_activity = time.time()
        self._error_count = 0
        self._stats = {
            'operations_count': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'created_at': time.time()
        }
        
        # Зависимости компонента
        self._dependencies = set()
        
        # Подписки на события: Dict[event_type, subscription_id]
        self._subscriptions: Dict[str, str] = {}
        
        logger.debug(f"BaseComponent {self.name} создан")
        
    def initialize(self) -> bool:
        """
        Инициализирует компонент и его зависимости с событийной поддержкой.
        
        Returns:
            bool: True если инициализация прошла успешно
        """
        with self._lock:
            if self._state == ComponentState.READY:
                logger.debug(f"Компонент {self.name} уже инициализирован")
                return True
            
            if self._state == ComponentState.INITIALIZING:
                logger.debug(f"Компонент {self.name} уже инициализируется")
                return True
            
            if self._state != ComponentState.UNINITIALIZED:
                if self._state == ComponentState.RUNNING:
                    logger.warning(f"Компонент {self.name} уже запущен, инициализация не требуется")
                    return True
                logger.warning(f"Компонент {self.name} имеет состояние {self._state}, инициализация невозможна")
                return False
            
            try:
                self._state = ComponentState.INITIALIZING
                
                # Проверяем зависимости
                if not self._check_dependencies():
                    self._state = ComponentState.ERROR
                    return False
                
                # Инициализация базового функционала
                if not self._do_initialize():
                    self._state = ComponentState.ERROR
                    return False
                
                # Подписываемся на события
                self._setup_event_subscriptions()
                
                self._state = ComponentState.READY
                self.is_initialized = True
                self._last_activity = time.time()
                
                # Публикуем событие инициализации
                self._emit_event(EventTypes.COMPONENT_INITIALIZED, {
                    'component': self.name,
                    'dependencies': list(self._dependencies)
                })
                
                logger.info(f"Компонент {self.name} инициализирован")
                return True
                
            except Exception as e:
                self._state = ComponentState.ERROR
                self._error_count += 1
                logger.error(f"Ошибка инициализации компонента {self.name}: {e}")
                return False
    
    def start(self) -> bool:
        """Запуск компонента"""
        with self._lock:
            if self._state == ComponentState.RUNNING:
                logger.debug(f"Компонент {self.name} уже запущен")
                return True
            
            if self._state == ComponentState.STARTING:
                logger.debug(f"Компонент {self.name} уже запускается")
                return True
            
            if self._state != ComponentState.READY:
                logger.warning(f"Компонент {self.name} не готов к запуску (состояние: {self._state})")
                return False
            
            try:
                self._state = ComponentState.STARTING
                self._start_time = time.time()
                
                if not self._do_start():
                    self._state = ComponentState.ERROR
                    return False
                
                self._state = ComponentState.RUNNING
                self._last_activity = time.time()
                
                self._emit_event(EventTypes.COMPONENT_STARTED, {
                    'component': self.name,
                    'start_time': self._start_time
                })
                
                logger.info(f"Компонент {self.name} запущен")
                return True
                
            except Exception as e:
                self._state = ComponentState.ERROR
                self._error_count += 1
                logger.error(f"Ошибка запуска компонента {self.name}: {e}")
                return False
    
    def stop(self) -> bool:
        """Остановка компонента"""
        with self._lock:
            if self._state != ComponentState.RUNNING:
                logger.warning(f"Компонент {self.name} не запущен (состояние: {self._state})")
                return False
            
            try:
                self._state = ComponentState.STOPPING
                
                if not self._do_stop():
                    self._state = ComponentState.ERROR
                    return False
                
                self._state = ComponentState.STOPPED
                self._last_activity = time.time()
                
                self._emit_event(EventTypes.COMPONENT_STOPPED, {
                    'component': self.name,
                    'runtime': time.time() - self._start_time if self._start_time > 0 else 0
                })
                
                logger.info(f"Компонент {self.name} остановлен")
                return True
                
            except Exception as e:
                self._state = ComponentState.ERROR
                self._error_count += 1
                logger.error(f"Ошибка остановки компонента {self.name}: {e}")
                return False
    
    def get_state(self) -> ComponentState:
        """Получение текущего состояния"""
        return self._state
    
    def is_ready(self) -> bool:
        """Проверка готовности компонента"""
        return self._state == ComponentState.READY
    
    def is_running(self) -> bool:
        """Проверка запущенности компонента"""
        return self._state == ComponentState.RUNNING
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики компонента"""
        with self._lock:
            uptime = time.time() - self._stats['created_at']
            runtime = time.time() - self._start_time if self._start_time > 0 and self._state == ComponentState.RUNNING else 0
            
            return {
                **self._stats,
                'name': self.name,
                'state': self._state.value,
                'uptime': uptime,
                'runtime': runtime,
                'last_activity': self._last_activity,
                'error_count': self._error_count,
                'subscriptions_count': len(self._subscriptions)
            }
    
    def _check_dependencies(self) -> bool:
        """Проверка зависимостей компонента"""
        if not self.brain or not self._required_components:
            return True
        
        for dep_name in self._required_components:
            if hasattr(self.brain, 'components') and dep_name in self.brain.components:
                dep_component = self.brain.components[dep_name]
                if hasattr(dep_component, 'is_ready') and not dep_component.is_ready():
                    logger.warning(f"Зависимость {dep_name} не готова для компонента {self.name}")
                    return False
            else:
                logger.warning(f"Зависимость {dep_name} не найдена для компонента {self.name}")
                return False
        
        return True
    
    def _setup_event_subscriptions(self):
        """Настройка подписок на события"""
        self._subscribe(EventTypes.SYSTEM_START, self._handle_system_start)
        self._subscribe(EventTypes.SYSTEM_STOP, self._handle_system_stop)
    
    def _subscribe(self, event_type: str, handler):
        """Подписка на событие"""
        try:
            subscription_id = self.event_bus.subscribe(event_type, handler)
            self._subscriptions[event_type] = subscription_id
        except Exception as e:
            logger.error(f"Ошибка подписки на {event_type}: {e}")
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Публикация события"""
        try:
            from .event_bus import Event
            event = Event(
                event_type=event_type,
                source=self.name,
                data=data,
                timestamp=time.time()
            )
            self.event_bus.publish(event)
        except Exception as e:
            logger.error(f"Ошибка публикации события {event_type}: {e}")
    
    def _handle_system_start(self, event):
        """Обработчик запуска системы"""
        if self._state == ComponentState.READY:
            self.start()
    
    def _handle_system_stop(self, event):
        """Обработчик остановки системы"""
        if self._state == ComponentState.RUNNING:
            self.stop()
    
    def _do_initialize(self) -> bool:
        """Базовая инициализация (переопределяется в дочерних классах)"""
        return True
    
    def _do_start(self) -> bool:
        """Базовый запуск (переопределяется в дочерних классах)"""
        return True
    
    def _do_stop(self) -> bool:
        """Базовая остановка (переопределяется в дочерних классах)"""
        return True
    
    def _do_cleanup(self):
        """Базовая очистка ресурсов компонента."""
        try:
            # Отписываемся от событий
            for event_type, subscription_id in list(self._subscriptions.items()):
                try:
                    self.event_bus.unsubscribe(event_type, subscription_id)
                except Exception:
                    pass
            self._subscriptions.clear()
            
            # Очищаем зависимости
            self._dependencies.clear()
            
            logger.debug(f"Базовая очистка компонента {self.name} выполнена")
            return True
        except Exception as e:
            logger.error(f"Ошибка очистки компонента {self.name}: {e}")
            return False
    
    def _setup_custom_subscriptions(self):
        """Настройка кастомных подписок на события."""
        # Базовая реализация - можно переопределить в дочерних классах
        pass
    
    # Вспомогательные методы
    def _update_stats(self, processing_time: float):
        """Обновление статистики"""
        with self._lock:
            self._stats['operations_count'] += 1
            self._stats['total_processing_time'] += processing_time
            self._stats['average_processing_time'] = (
                self._stats['total_processing_time'] / self._stats['operations_count']
            )
            self._last_activity = time.time()
    
    def _handle_error(self, error: Exception, context: str = ""):
        """Обработка ошибки"""
        with self._lock:
            self._error_count += 1
            
            # Публикуем событие ошибки
            self._emit_event(EventTypes.COMPONENT_ERROR, {
                'component': self.name,
                'error': str(error),
                'context': context,
                'error_count': self._error_count
            })
        
        logger.error(f"Ошибка в компоненте {self.name}: {error}")
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', state={self._state.value})"
    
    def __repr__(self) -> str:
        return self.__str__()

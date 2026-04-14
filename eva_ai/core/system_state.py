"""
Менеджер состояния системы для ЕВА с событийной интеграцией
"""

import time
import threading
from enum import Enum
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass
import logging

try:
    from .event_bus import EventBus, Event, EventTypes, get_event_bus
except ImportError:
    EventBus = None
    Event = None
    EventTypes = None
    get_event_bus = None

try:
    from .base_component import ComponentState
except ImportError:
    class ComponentState:
        UNINITIALIZED = "uninitialized"
        INITIALIZING = "initializing"
        READY = "ready"
        STARTING = "starting"
        RUNNING = "running"
        STOPPING = "stopping"
        STOPPED = "stopped"
        ERROR = "error"

logger = logging.getLogger(__name__)

class SystemState(Enum):
    """Состояния системы ЕВА."""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PROCESSING = "processing"
    ERROR = "error"
    RECOVERY = "recovery"
    SHUTTING_DOWN = "shutting_down"
    OFFLINE = "offline"

@dataclass
class ComponentStateInfo:
    """Информация о состоянии компонента"""
    name: str
    state: ComponentState
    last_activity: float
    error_count: int
    uptime: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            'name': self.name,
            'state': self.state.value,
            'last_activity': self.last_activity,
            'error_count': self.error_count,
            'uptime': self.uptime
        }

class SystemStateManager:
    """Управляет состоянием системы ЕВА с событийной интеграцией."""
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """Инициализирует менеджер состояния."""
        self.current_state = SystemState.INITIALIZING
        self.state_history = []
        self.state_lock = threading.RLock()
        self.state_listeners = []
        
        # Событийная система
        self.event_bus = event_bus
        if self.event_bus is None and get_event_bus is not None:
            try:
                self.event_bus = get_event_bus()
            except Exception:
                pass
        self._subscriptions: Set[tuple] = set()
        
        # Состояния компонентов
        self._component_states: Dict[str, ComponentStateInfo] = {}
        
        # Статистика
        self._stats = {
            'state_changes': 0,
            'component_errors': 0,
            'recoveries': 0,
            'start_time': time.time()
        }
        
        # Настройка подписок
        self._setup_event_subscriptions()
        
        logger.info("SystemStateManager инициализирован с событийной поддержкой")
    
    def update_component_state(self, 
                             component_name: str, 
                             state: ComponentState,
                             stats: Optional[Dict[str, Any]] = None):
        """
        Обновление состояния компонента
        
        Args:
            component_name: Имя компонента
            state: Новое состояние
            stats: Статистика компонента
        """
        with self.state_lock:
            current_time = time.time()
            
            # Получаем текущее состояние
            current_info = self._component_states.get(component_name)
            error_count = current_info.error_count if current_info else 0
            
            # Увеличиваем счетчик ошибок
            if state == ComponentState.ERROR:
                error_count += 1
                self._stats['component_errors'] += 1
            
            # Создаем новую информацию
            new_info = ComponentStateInfo(
                name=component_name,
                state=state,
                last_activity=current_time,
                error_count=error_count,
                uptime=current_time - self._stats['start_time']
            )
            
            self._component_states[component_name] = new_info
            
            # Обновляем состояние системы
            self._update_system_state()
            
            logger.debug(f"Состояние компонента {component_name} обновлено: {state.value}")
    
    def get_component_state(self, component_name: str) -> Optional[ComponentStateInfo]:
        """Получение состояния компонента"""
        with self.state_lock:
            return self._component_states.get(component_name)
    
    def get_all_component_states(self) -> Dict[str, ComponentStateInfo]:
        """Получение состояний всех компонентов"""
        with self.state_lock:
            return dict(self._component_states)
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Получение сводки о состоянии системы"""
        with self.state_lock:
            # Анализ состояний компонентов
            state_counts = {}
            for comp_state in self._component_states.values():
                state_name = comp_state.state.value
                state_counts[state_name] = state_counts.get(state_name, 0) + 1
            
            return {
                'system_state': self.current_state.value,
                'uptime': time.time() - self._stats['start_time'],
                'total_components': len(self._component_states),
                'component_states': state_counts,
                'statistics': dict(self._stats),
                'last_update': time.time()
            }
    
    def get_state(self) -> SystemState:
        """Получает текущее состояние системы"""
        return self.current_state
    
    def set_state(self, new_state: SystemState, reason: str = ""):
        """Установка нового состояния системы"""
        with self.state_lock:
            old_state = self.current_state
            self.current_state = new_state
            self._stats['state_changes'] += 1
            
            # Публикуем событие изменения состояния
            self._emit_event("system.state_changed", {
                'old_state': old_state.value,
                'new_state': new_state.value,
                'reason': reason,
                'timestamp': time.time()
            })
            
            logger.info(f"Состояние системы изменено: {old_state.value} -> {new_state.value} ({reason})")
    
    def _update_system_state(self):
        """Обновление состояния системы на основе состояний компонентов"""
        if not self._component_states:
            return
        
        # Считаем состояния компонентов
        error_count = 0
        running_count = 0
        ready_count = 0
        
        for info in self._component_states.values():
            if info.state == ComponentState.ERROR:
                error_count += 1
            elif info.state == ComponentState.RUNNING:
                running_count += 1
            elif info.state == ComponentState.READY:
                ready_count += 1
        
        # Определяем состояние системы
        total_components = len(self._component_states)
        
        if error_count > 0:
            if error_count > total_components * 0.3:  # >30% ошибок
                new_state = SystemState.ERROR
            else:
                new_state = SystemState.RECOVERY
        elif running_count == total_components:
            new_state = SystemState.RUNNING
        elif running_count + ready_count == total_components:
            new_state = SystemState.READY
        else:
            new_state = SystemState.INITIALIZING
        
        if new_state != self.current_state:
            self.set_state(new_state)
    
    def _setup_event_subscriptions(self):
        """Настройка подписок на события"""
        if self.event_bus is None:
            logger.warning("event_bus is None, skipping event subscriptions")
            return
        self._subscribe(EventTypes.COMPONENT_INITIALIZED, self._handle_component_initialized)
        self._subscribe(EventTypes.COMPONENT_STARTED, self._handle_component_started)
        self._subscribe(EventTypes.COMPONENT_STOPPED, self._handle_component_stopped)
        self._subscribe(EventTypes.COMPONENT_ERROR, self._handle_component_error)
    
    def _subscribe(self, event_type: str, handler):
        """Подписка на событие"""
        try:
            subscription_id = self.event_bus.subscribe(event_type, handler)
            self._subscriptions.add((event_type, subscription_id))
        except Exception as e:
            logger.error(f"Ошибка подписки на {event_type}: {e}")
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Публикация события"""
        try:
            event = Event(
                event_type=event_type,
                source="system_state",
                data=data,
                timestamp=time.time()
            )
            self.event_bus.publish(event)
        except Exception as e:
            logger.error(f"Ошибка публикации события {event_type}: {e}")
    
    # Обработчики событий
    def _handle_component_initialized(self, event):
        """Обработчик инициализации компонента"""
        component_name = None
        
        if isinstance(event, str):
            component_name = event
        elif hasattr(event, 'data'):
            component_name = event.data.get('component')
        elif hasattr(event, 'get'):
            component_name = event.get('component')
        
        if component_name:
            self.update_component_state(component_name, ComponentState.READY)
    
    def _handle_component_started(self, event):
        """Обработчик запуска компонента"""
        component_name = None
        
        if isinstance(event, str):
            component_name = event
        elif hasattr(event, 'data'):
            component_name = event.data.get('component')
        elif hasattr(event, 'get'):
            component_name = event.get('component')
        
        if component_name:
            self.update_component_state(component_name, ComponentState.RUNNING)
    
    def _handle_component_stopped(self, event):
        """Обработчик остановки компонента"""
        component_name = None
        
        if isinstance(event, str):
            component_name = event
        elif hasattr(event, 'data'):
            component_name = event.data.get('component')
        elif hasattr(event, 'get'):
            component_name = event.get('component')
        
        if component_name:
            self.update_component_state(component_name, ComponentState.STOPPED)
    
    def _handle_component_error(self, event):
        """Обработчик ошибки компонента"""
        if isinstance(event, str):
            component_name = event
        else:
            component_name = event.data.get('component') if hasattr(event, 'data') else None
        
        if component_name:
            self.update_component_state(component_name, ComponentState.ERROR)
    
    def cleanup(self):
        """Очистка ресурсов"""
        with self.state_lock:
            for event_type, subscription_id in self._subscriptions:
                try:
                    self.event_bus.unsubscribe(event_type, subscription_id)
                except Exception as e:
                    logger.warning(f"Ошибка отписки {subscription_id}: {e}")
            
            self._subscriptions.clear()
            self._component_states.clear()
            self.state_history.clear()
            
            logger.info("SystemStateManager очищен")

# Глобальный экземпляр
_global_system_state: Optional[SystemStateManager] = None

def get_system_state() -> SystemStateManager:
    """Получение глобального экземпляра SystemState"""
    global _global_system_state
    if _global_system_state is None:
        _global_system_state = SystemStateManager()
    return _global_system_state

def reset_system_state():
    """Сброс глобального экземпляра"""
    global _global_system_state
    if _global_system_state is not None:
        _global_system_state.cleanup()
    _global_system_state = None

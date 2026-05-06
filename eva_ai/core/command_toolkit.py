"""
Command Toolkit для CoreBrain
Инструментарий команд управления всеми компонентами системы через EventBus и DeferredCommandSystem.
Обеспечивает singleton паттерн и централизованное управление.
"""

import logging
import threading
import time
from typing import Dict, Any, Optional, Callable, List
from enum import Enum

from .event_bus import EventBus, Event, EventTypes, EventPriority
from .deferred_command_system import (
    DeferredCommandSystem, 
    DeferredCommand, 
    CommandPriority, 
    CommandStatus,
    set_event_bus
)
from .brain_commands import (
    BrainCommandExecutor,
    InitializeComponentCommand,
    StartComponentCommand,
    StopComponentCommand,
    GetComponentStatusCommand,
    StartOnlineTrainingCommand,
    StopOnlineTrainingCommand,
    GetTrainingStatusCommand,
    GenerateTextCommand,
    GetSystemHealthCommand
)

logger = logging.getLogger("eva_ai.command_toolkit")


class CommandToolkitState(Enum):
    INITIALIZED = "initialized"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


class CommandResult:
    """Результат выполнения команды"""
    def __init__(self, success: bool, data: Any = None, error: str = None):
        self.success = success
        self.data = data
        self.error = error
        self.timestamp = time.time()
    
    def to_dict(self):
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "timestamp": self.timestamp
        }


class CoreBrainCommand:
    """Базовый класс команды для CoreBrain"""
    
    def __init__(self, name: str, brain, priority: CommandPriority = CommandPriority.NORMAL):
        self.name = name
        self.brain = brain
        self.priority = priority
        self.created_at = time.time()
    
    def execute(self) -> CommandResult:
        """Выполнить команду - должен быть переопределён"""
        raise NotImplementedError
    
    def validate(self) -> bool:
        """Валидация команды перед выполнением"""
        return True
    
    def get_description(self) -> str:
        return f"Command: {self.name}"


class ComponentCommand(CoreBrainCommand):
    """Команда управления компонентом"""
    
    def __init__(self, name: str, brain, component_name: str, action: str, **kwargs):
        super().__init__(name, brain, CommandPriority.NORMAL)
        self.component_name = component_name
        self.action = action
        self.params = kwargs
    
    def execute(self) -> CommandResult:
        try:
            component = getattr(self.brain, self.component_name, None)
            if component is None:
                return CommandResult(False, error=f"Component {self.component_name} not found")
            
            if not hasattr(component, self.action):
                return CommandResult(False, error=f"Action {self.action} not found on {self.component_name}")
            
            action_method = getattr(component, self.action)
            result = action_method(**self.params)
            
            return CommandResult(True, data=result)
        except Exception as e:
            logger.error(f"ComponentCommand error: {e}")
            return CommandResult(False, error=str(e))
    
    def get_description(self) -> str:
        return f"Component {self.component_name}.{self.action}"


class EventPublishCommand(CoreBrainCommand):
    """Команда публикации события"""
    
    def __init__(self, name: str, brain, event_type: str, event_data: Dict[str, Any], priority: CommandPriority = CommandPriority.NORMAL):
        super().__init__(name, brain, priority)
        self.event_type = event_type
        self.event_data = event_data
    
    def execute(self) -> CommandResult:
        try:
            event_bus = getattr(self.brain, 'events', None)
            if event_bus is None:
                return CommandResult(False, error="EventBus not available")
            
            event = Event(
                event_type=self.event_type,
                source=f"command.{self.name}",
                data=self.event_data,
                priority=EventPriority.NORMAL
            )
            event_bus.publish(event)
            return CommandResult(True, data={"event_type": self.event_type})
        except Exception as e:
            return CommandResult(False, error=str(e))


class BrainStateCommand(CoreBrainCommand):
    """Команда управления состоянием brain"""
    
    def __init__(self, name: str, brain, new_state: str):
        super().__init__(name, brain, CommandPriority.HIGH)
        self.new_state = new_state
    
    def execute(self) -> CommandResult:
        try:
            if hasattr(self.brain, 'set_state'):
                self.brain.set_state(self.new_state)
            elif hasattr(self.brain, 'state'):
                self.brain.state = self.new_state
            
            return CommandResult(True, data={"state": self.new_state})
        except Exception as e:
            return CommandResult(False, error=str(e))


class CommandToolkit:
    """
    Инструментарий команд для CoreBrain.
    Обеспечивает:
    - Единый интерфейс для всех команд
    - Интеграцию с EventBus
    - Использование DeferredCommandSystem для асинхронного выполнения
    - Singleton паттерн через EventBus
    """
    
    _instance = None
    _instance_lock = threading.Lock()
    
    def __new__(cls, brain=None):
        if cls._instance is not None:
            return cls._instance
        
        with cls._instance_lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                cls._instance = instance
                return instance
        return cls._instance
    
    def __init__(self, brain=None):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.brain = brain
        self._event_bus: Optional[EventBus] = None
        self._deferred_system: Optional[DeferredCommandSystem] = None
        self._state = CommandToolkitState.INITIALIZED
        self._command_registry: Dict[str, type] = {}
        self._active_commands: Dict[str, DeferredCommand] = {}
        self._lock = threading.RLock()
        
        self._initialized = True
        logger.info("[CommandToolkit] Initialized (singleton)")
    
    def initialize(self, event_bus: EventBus, deferred_system: DeferredCommandSystem = None):
        """
        Инициализировать toolkit с EventBus и DeferredCommandSystem.
        Вызывается один раз при старте системы.
        """
        with self._lock:
            self._event_bus = event_bus
            self._deferred_system = deferred_system
            
            # Подключить EventBus к DeferredCommandSystem
            if deferred_system:
                set_event_bus(event_bus)
            
            # Зарегистрировать все команды
            self._register_default_commands()
            
            # Подписаться на системные события
            self._subscribe_to_events()
            
            self._state = CommandToolkitState.RUNNING
            logger.info("[CommandToolkit] Initialized with EventBus and DeferredCommandSystem")
    
    def _register_default_commands(self):
        """Зарегистрировать стандартные команды"""
        self._command_registry = {
            "component_control": ComponentCommand,
            "event_publish": EventPublishCommand,
            "brain_state": BrainStateCommand,
            "init_component": InitializeComponentCommand,
            "start_component": StartComponentCommand,
            "stop_component": StopComponentCommand,
            "component_status": GetComponentStatusCommand,
            "start_training": StartOnlineTrainingCommand,
            "stop_training": StopOnlineTrainingCommand,
            "training_status": GetTrainingStatusCommand,
            "generate_text": GenerateTextCommand,
            "system_health": GetSystemHealthCommand,
        }
        
        # Создать исполнитель команд
        self._command_executor = BrainCommandExecutor(brain) if brain else None
    
    def _subscribe_to_events(self):
        """Подписаться на события системы"""
        if self._event_bus is None:
            return
        
        self._event_bus.subscribe(EventTypes.SYSTEM_START, self._on_system_start, priority=1)
        self._event_bus.subscribe(EventTypes.SYSTEM_STOP, self._on_system_stop, priority=1)
        self._event_bus.subscribe(EventTypes.COMPONENT_ERROR, self._on_component_error, priority=1)
        self._event_bus.subscribe(EventTypes.SYSTEM_ERROR, self._on_system_error, priority=1)
    
    def _on_system_start(self, event: Event):
        logger.info(f"[CommandToolkit] System started: {event.data}")
    
    def _on_system_stop(self, event: Event):
        logger.info(f"[CommandToolkit] System stopped: {event.data}")
        self._state = CommandToolkitState.STOPPED
    
    def _on_component_error(self, event: Event):
        logger.warning(f"[CommandToolkit] Component error: {event.data}")
        # Здесь можно добавить логику восстановления
    
    def _on_system_error(self, event: Event):
        logger.error(f"[CommandToolkit] System error: {event.data}")
        self._state = CommandToolkitState.ERROR
    
    def execute_command(self, command: CoreBrainCommand, async_mode: bool = False) -> CommandResult:
        """
        Выполнить команду синхронно или асинхронно.
        
        Args:
            command: Объект команды
            async_mode: Если True - выполнять через DeferredCommandSystem
        """
        with self._lock:
            if not command.validate():
                return CommandResult(False, error="Command validation failed")
            
            if async_mode and self._deferred_system:
                return self._execute_async(command)
            else:
                return command.execute()
    
    def _execute_async(self, command: CoreBrainCommand) -> CommandResult:
        """Асинхронное выполнение через DeferredCommandSystem"""
        def wrapped_execute():
            return command.execute()
        
        deferred_cmd = DeferredCommand(
            id=f"{command.name}_{int(time.time()*1000)}",
            command=wrapped_execute,
            args=(),
            kwargs={},
            priority=command.priority,
            max_retries=3,
            retry_delay=1.0,
            timeout=30.0,
            created_at=time.time()
        )
        
        self._deferred_system.submit(deferred_cmd)
        self._active_commands[deferred_cmd.id] = deferred_cmd
        
        return CommandResult(True, data={"command_id": deferred_cmd.id, "async": True})
    
    def execute_component_action(self, component_name: str, action: str, **kwargs) -> CommandResult:
        """Упрощённый интерфейс для выполнения действия над компонентом"""
        cmd = ComponentCommand(
            name=f"{component_name}_{action}",
            brain=self.brain,
            component_name=component_name,
            action=action,
            **kwargs
        )
        return self.execute_command(cmd, async_mode=False)
    
    def publish_event(self, event_type: str, event_data: Dict[str, Any], priority: CommandPriority = CommandPriority.NORMAL) -> CommandResult:
        """Опубликовать событие через команду"""
        cmd = EventPublishCommand(
            name=f"event_{event_type}",
            brain=self.brain,
            event_type=event_type,
            event_data=event_data,
            priority=priority
        )
        return self.execute_command(cmd, async_mode=False)
    
    def set_brain_state(self, state: str) -> CommandResult:
        """Установить состояние brain"""
        cmd = BrainStateCommand(
            name=f"state_{state}",
            brain=self.brain,
            new_state=state
        )
        return self.execute_command(cmd, async_mode=False)
    
    def get_status(self) -> Dict[str, Any]:
        """Получить статус toolkit"""
        return {
            "state": self._state.value,
            "brain_connected": self.brain is not None,
            "event_bus_connected": self._event_bus is not None,
            "deferred_system_connected": self._deferred_system is not None,
            "active_commands": len(self._active_commands),
            "registered_commands": len(self._command_registry)
        }
    
    def shutdown(self):
        """Остановить toolkit"""
        self._state = CommandToolkitState.STOPPED
        logger.info("[CommandToolkit] Shutdown")


def get_command_toolkit(brain=None) -> CommandToolkit:
    """Получить экземпляр CommandToolkit (singleton)"""
    return CommandToolkit(brain)
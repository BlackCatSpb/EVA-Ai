"""
Набор команд для управления CoreBrain.
Каждая команда - это класс, который может быть выполнен синхронно или через DeferredCommandSystem.
"""

import logging
import time
import abc
from typing import Dict, Any, Optional
from enum import Enum

from .deferred_command_system import CommandPriority
from .event_bus import EventTypes, Event, EventPriority

logger = logging.getLogger("eva_ai.brain_commands")


class BrainCommandCategory(Enum):
    """Категории команд brain"""
    SYSTEM = "system"
    COMPONENT = "component"
    LEARNING = "learning"
    MEMORY = "memory"
    GENERATION = "generation"
    MONITORING = "monitoring"


class BrainCommand(abc.ABC):
    """Базовый класс команды для brain"""
    
    category = BrainCommandCategory.SYSTEM
    
    def __init__(self, brain, command_id: str = None):
        self.brain = brain
        self.command_id = command_id or f"{self.__class__.__name__}_{int(time.time()*1000)}"
        self.created_at = time.time()
        self.result = None
    
    @abc.abstractmethod
    def execute(self) -> Dict[str, Any]:
        """Выполнить команду - должен быть переопределён в подклассах"""
        pass
    
    def validate(self) -> bool:
        """Валидация перед выполнением"""
        return True
    
    def publish_event(self, event_type: str, data: Dict[str, Any]):
        """Опубликовать событие после выполнения"""
        event_bus = getattr(self.brain, '_new_event_bus', None)
        if event_bus:
            event = Event(
                event_type=event_type,
                source=f"command.{self.command_id}",
                data=data,
                priority=EventPriority.NORMAL
            )
            event_bus.publish(event)


class InitializeComponentCommand(BrainCommand):
    """Команда инициализации компонента"""
    
    category = BrainCommandCategory.COMPONENT
    
    def __init__(self, brain, component_name: str, **init_params):
        super().__init__(brain)
        self.component_name = component_name
        self.init_params = init_params
    
    def execute(self) -> Dict[str, Any]:
        try:
            init_method = getattr(self.brain, f'_init_{self.component_name}', None)
            if init_method:
                init_method(self.brain)
                self.publish_event(EventTypes.COMPONENT_INITIALIZED, {
                    "component": self.component_name,
                    "status": "initialized"
                })
                return {"success": True, "component": self.component_name}
            else:
                return {"success": False, "error": f"Init method not found for {self.component_name}"}
        except Exception as e:
            self.publish_event(EventTypes.COMPONENT_ERROR, {
                "component": self.component_name,
                "error": str(e)
            })
            return {"success": False, "error": str(e)}


class StartComponentCommand(BrainCommand):
    """Команда запуска компонента"""
    
    category = BrainCommandCategory.COMPONENT
    
    def __init__(self, brain, component_name: str):
        super().__init__(brain)
        self.component_name = component_name
    
    def execute(self) -> Dict[str, Any]:
        component = getattr(self.brain, self.component_name, None)
        if component is None:
            return {"success": False, "error": f"Component {self.component_name} not found"}
        
        if hasattr(component, 'start'):
            component.start()
            self.publish_event(EventTypes.COMPONENT_STARTED, {
                "component": self.component_name
            })
            return {"success": True, "component": self.component_name}
        
        return {"success": False, "error": "Component has no start method"}


class StopComponentCommand(BrainCommand):
    """Команда остановки компонента"""
    
    category = BrainCommandCategory.COMPONENT
    
    def __init__(self, brain, component_name: str):
        super().__init__(brain)
        self.component_name = component_name
    
    def execute(self) -> Dict[str, Any]:
        component = getattr(self.brain, self.component_name, None)
        if component is None:
            return {"success": False, "error": f"Component {self.component_name} not found"}
        
        if hasattr(component, 'stop'):
            component.stop()
            self.publish_event(EventTypes.COMPONENT_STOPPED, {
                "component": self.component_name
            })
            return {"success": True, "component": self.component_name}
        
        return {"success": False, "error": "Component has no stop method"}


class GetComponentStatusCommand(BrainCommand):
    """Команда получения статуса компонента"""
    
    category = BrainCommandCategory.COMPONENT
    
    def __init__(self, brain, component_name: str):
        super().__init__(brain)
        self.component_name = component_name
    
    def execute(self) -> Dict[str, Any]:
        component = getattr(self.brain, self.component_name, None)
        if component is None:
            return {"success": False, "error": f"Component {self.component_name} not found"}
        
        status = {"component": self.component_name}
        
        if hasattr(component, 'get_status'):
            status.update(component.get_status())
        if hasattr(component, 'running'):
            status["running"] = component.running
        if hasattr(component, 'initialized'):
            status["initialized"] = component.initialized
        
        return {"success": True, "status": status}


class StartOnlineTrainingCommand(BrainCommand):
    """Команда запуска онлайн обучения"""
    
    category = BrainCommandCategory.LEARNING
    
    def __init__(self, brain, trainer_type: str = "all"):
        super().__init__(brain)
        self.trainer_type = trainer_type
    
    def execute(self) -> Dict[str, Any]:
        try:
            online_trainer = getattr(self.brain, 'online_trainer', None)
            if online_trainer is None:
                return {"success": False, "error": "OnlineTrainer not initialized"}
            
            online_trainer.start()
            
            self.publish_event(EventTypes.LEARNING_STARTED, {
                "trainer_type": self.trainer_type
            })
            
            return {"success": True, "trainer_type": self.trainer_type}
        except Exception as e:
            self.publish_event(EventTypes.LEARNING_FAILED, {"error": str(e)})
            return {"success": False, "error": str(e)}


class StopOnlineTrainingCommand(BrainCommand):
    """Команда остановки онлайн обучения"""
    
    category = BrainCommandCategory.LEARNING
    
    def __init__(self, brain):
        super().__init__(brain)
    
    def execute(self) -> Dict[str, Any]:
        try:
            online_trainer = getattr(self.brain, 'online_trainer', None)
            if online_trainer is None:
                return {"success": False, "error": "OnlineTrainer not initialized"}
            
            online_trainer.stop()
            
            self.publish_event(EventTypes.LEARNING_COMPLETED, {})
            
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}


class GetTrainingStatusCommand(BrainCommand):
    """Команда получения статуса обучения"""
    
    category = BrainCommandCategory.LEARNING
    
    def __init__(self, brain):
        super().__init__(brain)
    
    def execute(self) -> Dict[str, Any]:
        online_trainer = getattr(self.brain, 'online_trainer', None)
        if online_trainer is None:
            return {"success": False, "error": "OnlineTrainer not initialized"}
        
        return {"success": True, "status": online_trainer.get_status()}


class GenerateTextCommand(BrainCommand):
    """Команда генерации текста"""
    
    category = BrainCommandCategory.GENERATION
    
    def __init__(self, brain, prompt: str, **generation_params):
        super().__init__(brain)
        self.prompt = prompt
        self.generation_params = generation_params
    
    def execute(self) -> Dict[str, Any]:
        try:
            pipeline = getattr(self.brain, 'fcp_pipeline', None)
            if pipeline is None:
                return {"success": False, "error": "Pipeline not initialized"}
            
            result = pipeline.generate(self.prompt, **self.generation_params)
            
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}


class GetSystemHealthCommand(BrainCommand):
    """Команда получения здоровья системы"""
    
    category = BrainCommandCategory.MONITORING
    
    def __init__(self, brain):
        super().__init__(brain)
    
    def execute(self) -> Dict[str, Any]:
        health = {
            "timestamp": time.time(),
            "initialized": getattr(self.brain, 'initialized', False),
            "running": getattr(self.brain, 'running', False),
            "components": {}
        }
        
        key_components = [
            'fcp_pipeline', 'hybrid_dialog_manager', 'memory_manager',
            'knowledge_graph', 'online_trainer', 'events'
        ]
        
        for comp_name in key_components:
            comp = getattr(self.brain, comp_name, None)
            health["components"][comp_name] = comp is not None
        
        return {"success": True, "health": health}


class BrainCommandExecutor:
    """
    Исполнитель команд brain.
    Может выполнять команды синхронно или через DeferredCommandSystem.
    """
    
    def __init__(self, brain):
        self.brain = brain
        self.deferred_system = getattr(brain, 'deferred_system', None)
        self.event_bus = getattr(brain, '_new_event_bus', None)
        self.component_bridge = getattr(brain, 'component_bridge', None)
    
    def execute(self, command: BrainCommand, async_mode: bool = False) -> Dict[str, Any]:
        """Выполнить команду"""
        if not command.validate():
            return {"success": False, "error": "Validation failed"}
        
        if async_mode and self.deferred_system:
            from eva_ai.core.deferred_command_system import DeferredCommand
            
            def wrapped_execute():
                return command.execute()
            
            deferred_cmd = DeferredCommand(
                id=command.command_id,
                command=wrapped_execute,
                args=(),
                kwargs={},
                priority=CommandPriority.NORMAL,
                max_retries=3,
                retry_delay=1.0,
                timeout=30.0,
                created_at=time.time()
            )
            
            self.deferred_system.submit(deferred_cmd)
            return {"success": True, "async": True, "command_id": command.command_id}
        
        return command.execute()
    
    def execute_component_action(self, component: str, action: str, **params) -> Dict[str, Any]:
        """Упрощённый интерфейс для выполнения действия над компонентом"""
        # Использовать component_bridge если доступно
        if self.component_bridge:
            command_map = {
                "start": "start_component",
                "stop": "stop_component",
                "status": "get_component_status",
                "restart": "restart_component"
            }
            cmd = command_map.get(action)
            if cmd:
                return self.component_bridge.execute_command(cmd, {"component": component})
        
        # Fallback на старые команды
        if action == "start":
            cmd = StartComponentCommand(self.brain, component)
        elif action == "stop":
            cmd = StopComponentCommand(self.brain, component)
        elif action == "status":
            cmd = GetComponentStatusCommand(self.brain, component)
        else:
            return {"success": False, "error": f"Unknown action: {action}"}
        
        return self.execute(cmd)
    
    def get_all_components(self) -> Dict[str, Any]:
        """Получить все зарегистрированные компоненты"""
        if self.component_bridge:
            return self.component_bridge.get_status()
        return {"success": False, "error": "Component bridge not available"}
    
    def get_command_list(self) -> Dict[str, type]:
        """Получить список доступных команд"""
        return {
            "InitializeComponentCommand": InitializeComponentCommand,
            "StartComponentCommand": StartComponentCommand,
            "StopComponentCommand": StopComponentCommand,
            "GetComponentStatusCommand": GetComponentStatusCommand,
            "StartOnlineTrainingCommand": StartOnlineTrainingCommand,
            "StopOnlineTrainingCommand": StopOnlineTrainingCommand,
            "GetTrainingStatusCommand": GetTrainingStatusCommand,
            "GenerateTextCommand": GenerateTextCommand,
            "GetSystemHealthCommand": GetSystemHealthCommand,
        }
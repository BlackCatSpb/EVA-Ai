"""
Component Event Bridge - автоматическая подписка всех компонентов на EventBus.
Обеспечивает единую систему управления через события и команды.
"""

import logging
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field

logger = logging.getLogger("eva_ai.component_event_bridge")


@dataclass
class ComponentRegistration:
    """Регистрация компонента"""
    name: str
    instance: Any
    event_handlers: Dict[str, Callable] = field(default_factory=dict)
    is_running: bool = False
    is_initialized: bool = False


class ComponentEventBridge:
    """
    Мост для подписки всех компонентов на EventBus.
    Обеспечивает:
    - Централизованную регистрацию компонентов
    - Подписку на системные события
    - Управление через команды
    - Синхронизацию состояний
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._event_bus = None
        self._brain = None
        self._components: Dict[str, ComponentRegistration] = {}
        self._command_handlers: Dict[str, Callable] = {}
        self._lock = threading.RLock()
        
        self._initialized = True
        logger.info("[ComponentEventBridge] Initialized (singleton)")
    
    def initialize(self, brain, event_bus):
        """Инициализировать мост с brain и EventBus"""
        with self._lock:
            self._brain = brain
            self._event_bus = event_bus
            
            # Зарегистрировать все компоненты brain
            self._register_all_components()
            
            # Подписаться на системные события
            self._subscribe_to_system_events()
            
            # Зарегистрировать обработчики команд
            self._register_command_handlers()
            
            logger.info("[ComponentEventBridge] Fully initialized")
    
    def _register_all_components(self):
        """Зарегистрировать все компоненты brain"""
        if not self._brain:
            return
        
        component_mapping = {
            'memory_manager': self._register_memory_manager,
            'knowledge_graph': self._register_knowledge_graph,
            'learning_manager': self._register_learning_manager,
            'self_analyzer': self._register_self_analyzer,
            'fcp_pipeline': self._register_fcp_pipeline,
            'hybrid_dialog_manager': self._register_hybrid_dialog_manager,
            'online_trainer': self._register_online_trainer,
            'state_manager': self._register_state_manager,
            'resource_manager': self._register_resource_manager,
            'config_manager': self._register_config_manager,
            'metrics_manager': self._register_metrics_manager,
            'mode_controller': self._register_mode_controller,
            'events': self._register_events_system,
            'deferred_system': self._register_deferred_system,
            'reasoning_chain': self._register_reasoning_chain,
            'graph_curator': self._register_graph_curator,
            'learning_orchestrator': self._register_learning_orchestrator,
            'shadow_lora_manager': self._register_shadow_lora,
        }
        
        for comp_name, register_func in component_mapping.items():
            try:
                comp = getattr(self._brain, comp_name, None)
                if comp is not None:
                    register_func(comp)
            except Exception as e:
                logger.debug(f"Register {comp_name}: {e}")
    
    def _register_memory_manager(self, comp):
        self._components['memory_manager'] = ComponentRegistration(
            name='memory_manager',
            instance=comp,
            is_initialized=getattr(comp, 'initialized', True)
        )
        logger.info("[ComponentEventBridge] Registered: memory_manager")
    
    def _register_knowledge_graph(self, comp):
        self._components['knowledge_graph'] = ComponentRegistration(
            name='knowledge_graph',
            instance=comp,
            is_initialized=True
        )
        logger.info("[ComponentEventBridge] Registered: knowledge_graph")
    
    def _register_learning_manager(self, comp):
        self._components['learning_manager'] = ComponentRegistration(
            name='learning_manager',
            instance=comp,
            is_initialized=True
        )
        logger.info("[ComponentEventBridge] Registered: learning_manager")
    
    def _register_self_analyzer(self, comp):
        self._components['self_analyzer'] = ComponentRegistration(
            name='self_analyzer',
            instance=comp,
            is_initialized=getattr(comp, 'initialized', True)
        )
        logger.info("[ComponentEventBridge] Registered: self_analyzer")
    
    def _register_fcp_pipeline(self, comp):
        self._components['fcp_pipeline'] = ComponentRegistration(
            name='fcp_pipeline',
            instance=comp,
            is_initialized=getattr(comp, 'pipeline', None) is not None
        )
        logger.info("[ComponentEventBridge] Registered: fcp_pipeline")
        
        # Регистрация внутренних компонентов FCPipeline
        if hasattr(comp, 'reasoning_chain') and comp.reasoning_chain:
            self._components['fcp_reasoning_chain'] = ComponentRegistration(
                name='fcp_reasoning_chain',
                instance=comp.reasoning_chain,
                is_initialized=True
            )
            logger.info("[ComponentEventBridge] Registered: fcp_reasoning_chain")
        
        if hasattr(comp, 'reasoning_manager') and comp.reasoning_manager:
            self._components['fcp_reasoning_manager'] = ComponentRegistration(
                name='fcp_reasoning_manager',
                instance=comp.reasoning_manager,
                is_initialized=True
            )
            logger.info("[ComponentEventBridge] Registered: fcp_reasoning_manager")
        
        if hasattr(comp, 'lora_manager') and comp.lora_manager:
            self._components['fcp_lora_manager'] = ComponentRegistration(
                name='fcp_lora_manager',
                instance=comp.lora_manager,
                is_initialized=True
            )
            logger.info("[ComponentEventBridge] Registered: fcp_lora_manager")
    
    def _register_hybrid_dialog_manager(self, comp):
        self._components['hybrid_dialog_manager'] = ComponentRegistration(
            name='hybrid_dialog_manager',
            instance=comp,
            is_initialized=getattr(comp, 'initialized', False)
        )
        logger.info("[ComponentEventBridge] Registered: hybrid_dialog_manager")
    
    def _register_online_trainer(self, comp):
        self._components['online_trainer'] = ComponentRegistration(
            name='online_trainer',
            instance=comp,
            is_initialized=True,
            is_running=True
        )
        logger.info("[ComponentEventBridge] Registered: online_trainer")
    
    def _register_state_manager(self, comp):
        self._components['state_manager'] = ComponentRegistration(
            name='state_manager',
            instance=comp,
            is_initialized=True
        )
        logger.info("[ComponentEventBridge] Registered: state_manager")
    
    def _register_resource_manager(self, comp):
        self._components['resource_manager'] = ComponentRegistration(
            name='resource_manager',
            instance=comp,
            is_initialized=True
        )
        logger.info("[ComponentEventBridge] Registered: resource_manager")
    
    def _register_config_manager(self, comp):
        self._components['config_manager'] = ComponentRegistration(
            name='config_manager',
            instance=comp,
            is_initialized=True
        )
        logger.info("[ComponentEventBridge] Registered: config_manager")
    
    def _register_metrics_manager(self, comp):
        self._components['metrics_manager'] = ComponentRegistration(
            name='metrics_manager',
            instance=comp,
            is_initialized=True
        )
        logger.info("[ComponentEventBridge] Registered: metrics_manager")
    
    def _register_mode_controller(self, comp):
        self._components['mode_controller'] = ComponentRegistration(
            name='mode_controller',
            instance=comp,
            is_initialized=True
        )
        logger.info("[ComponentEventBridge] Registered: mode_controller")
    
    def _register_events_system(self, comp):
        self._components['events'] = ComponentRegistration(
            name='events',
            instance=comp,
            is_initialized=True
        )
        logger.info("[ComponentEventBridge] Registered: events")
    
    def _register_deferred_system(self, comp):
        self._components['deferred_system'] = ComponentRegistration(
            name='deferred_system',
            instance=comp,
            is_initialized=True
        )
        logger.info("[ComponentEventBridge] Registered: deferred_system")
    
    def _register_reasoning_chain(self, comp):
        self._components['reasoning_chain'] = ComponentRegistration(
            name='reasoning_chain',
            instance=comp,
            is_initialized=True
        )
        logger.info("[ComponentEventBridge] Registered: reasoning_chain")
    
    def _register_graph_curator(self, comp):
        self._components['graph_curator'] = ComponentRegistration(
            name='graph_curator',
            instance=comp,
            is_initialized=True
        )
        logger.info("[ComponentEventBridge] Registered: graph_curator")
    
    def _register_learning_orchestrator(self, comp):
        self._components['learning_orchestrator'] = ComponentRegistration(
            name='learning_orchestrator',
            instance=comp,
            is_initialized=True
        )
        logger.info("[ComponentEventBridge] Registered: learning_orchestrator")
    
    def _register_shadow_lora(self, comp):
        self._components['shadow_lora_manager'] = ComponentRegistration(
            name='shadow_lora_manager',
            instance=comp,
            is_initialized=True
        )
        logger.info("[ComponentEventBridge] Registered: shadow_lora_manager")
    
    def _subscribe_to_system_events(self):
        """Подписаться на системные события EventBus"""
        if not self._event_bus:
            return
        
        from eva_ai.core.event_bus import EventTypes, Event
        
        def on_component_init(event: Event):
            comp_name = event.data.get('component')
            if comp_name:
                with self._lock:
                    if comp_name in self._components:
                        self._components[comp_name].is_initialized = True
                logger.info(f"[ComponentEventBridge] Component initialized: {comp_name}")
        
        def on_learning_progress(event: Event):
            logger.debug(f"[ComponentEventBridge] Learning progress: {event.data}")
        
        def on_knowledge_update(event: Event):
            logger.debug(f"[ComponentEventBridge] Knowledge updated: {event.data}")
        
        self._event_bus.subscribe(EventTypes.COMPONENT_INITIALIZED, on_component_init, priority=3)
        self._event_bus.subscribe(EventTypes.LEARNING_PROGRESS, on_learning_progress, priority=5)
        self._event_bus.subscribe(EventTypes.KNOWLEDGE_UPDATED, on_knowledge_update, priority=5)
        
        logger.info("[ComponentEventBridge] Subscribed to system events")
    
    def _register_command_handlers(self):
        """Зарегистрировать обработчики команд"""
        self._command_handlers = {
            'get_components': self._handle_get_components,
            'get_component_status': self._handle_get_status,
            'start_component': self._handle_start_component,
            'stop_component': self._handle_stop_component,
            'restart_component': self._handle_restart_component,
        }
        
        # Зарегистрировать дополнительные компоненты из brain_init
        self._register_additional_components()
    
    def _register_additional_components(self):
        """Зарегистрировать дополнительные компоненты из brain_init и других мест"""
        if not self._brain:
            return
        
        additional_components = [
            'generation_coordinator',
            'wikipedia_kb',
            'wikipedia_loader',
            'reasoning_integration',
            'performance_monitor',
            'self_evaluation',
            'model_manager',
            'text_processor',
            'hybrid_cache',
            'knowledge_graph',
            'self_dialog_learning',
            'unified_generator'
        ]
        
        for comp_name in additional_components:
            try:
                comp = getattr(self._brain, comp_name, None)
                if comp is not None:
                    self._components[comp_name] = ComponentRegistration(
                        name=comp_name,
                        instance=comp,
                        is_initialized=True
                    )
                    logger.debug(f"[ComponentEventBridge] Registered: {comp_name}")
            except Exception as e:
                logger.debug(f"Register {comp_name}: {e}")
    
    def _handle_get_components(self, data: Dict) -> Dict:
        """Получить список всех компонентов"""
        return {
            "success": True,
            "components": [
                {"name": name, "initialized": reg.is_initialized, "running": reg.is_running}
                for name, reg in self._components.items()
            ]
        }
    
    def _handle_get_status(self, data: Dict) -> Dict:
        """Получить статус компонента"""
        comp_name = data.get('component')
        if not comp_name or comp_name not in self._components:
            return {"success": False, "error": "Component not found"}
        
        reg = self._components[comp_name]
        return {
            "success": True,
            "component": comp_name,
            "initialized": reg.is_initialized,
            "running": reg.is_running,
            "instance_type": type(reg.instance).__name__
        }
    
    def _handle_start_component(self, data: Dict) -> Dict:
        """Запустить компонент"""
        comp_name = data.get('component')
        if not comp_name or comp_name not in self._components:
            return {"success": False, "error": "Component not found"}
        
        reg = self._components[comp_name]
        if hasattr(reg.instance, 'start'):
            reg.instance.start()
            reg.is_running = True
            return {"success": True, "component": comp_name, "status": "started"}
        return {"success": False, "error": "Component has no start method"}
    
    def _handle_stop_component(self, data: Dict) -> Dict:
        """Остановить компонент"""
        comp_name = data.get('component')
        if not comp_name or comp_name not in self._components:
            return {"success": False, "error": "Component not found"}
        
        reg = self._components[comp_name]
        if hasattr(reg.instance, 'stop'):
            reg.instance.stop()
            reg.is_running = False
            return {"success": True, "component": comp_name, "status": "stopped"}
        return {"success": False, "error": "Component has no stop method"}
    
    def _handle_restart_component(self, data: Dict) -> Dict:
        """Перезапустить компонент"""
        comp_name = data.get('component')
        if not comp_name or comp_name not in self._components:
            return {"success": False, "error": "Component not found"}
        
        # Stop first
        stop_result = self._handle_stop_component({"component": comp_name})
        if not stop_result.get("success"):
            return stop_result
        
        # Then start
        return self._handle_start_component({"component": comp_name})
    
    def execute_command(self, command: str, data: Dict = None) -> Dict:
        """Выполнить команду управления компонентом"""
        data = data or {}
        handler = self._command_handlers.get(command)
        if handler:
            return handler(data)
        return {"success": False, "error": f"Unknown command: {command}"}
    
    def get_all_components(self) -> List[str]:
        """Получить список всех зарегистрированных компонентов"""
        return list(self._components.keys())
    
    def get_component(self, name: str) -> Optional[Any]:
        """Получить экземпляр компонента по имени"""
        reg = self._components.get(name)
        return reg.instance if reg else None
    
    def get_status(self) -> Dict[str, Any]:
        """Получить полный статус моста"""
        return {
            "registered_components": len(self._components),
            "command_handlers": len(self._command_handlers),
            "event_bus_connected": self._event_bus is not None,
            "brain_connected": self._brain is not None,
            "components": {
                name: {"initialized": reg.is_initialized, "running": reg.is_running}
                for name, reg in self._components.items()
            }
        }


def get_component_bridge() -> ComponentEventBridge:
    """Получить экземпляр моста (singleton)"""
    return ComponentEventBridge()
"""
Типы и состояния для CoreBrain
Часть модуля core_brain.py (разделение на логические компоненты)
"""
from enum import Enum
from typing import Dict, Any, Optional


class SystemState(Enum):
    """Состояния системы CogniFlex."""
    INITIALIZING = "initializing"
    INITIALIZING_COMPONENTS = "initializing_components"
    CONNECTING_SERVICES = "connecting_services"
    READY = "ready"
    RUNNING = "running"
    PROCESSING = "processing"
    IDLE = "idle"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    DEGRADED = "degraded"
    SHUTTING_DOWN = "shutting_down"
    MAINTENANCE = "maintenance"
    
    @classmethod
    def is_valid_state(cls, state: str) -> bool:
        """Проверяет, является ли состояние валидным."""
        return hasattr(cls, state) and isinstance(getattr(cls, state), str)
    
    @classmethod
    def get_all_states(cls) -> list:
        """Возвращает все доступные состояния."""
        return [attr for attr in dir(cls) if not attr.startswith('_') and isinstance(getattr(cls, attr), str)]
    
    @classmethod
    def is_operational_state(cls, state: str) -> bool:
        """Проверяет, является ли состояние рабочим."""
        operational_states = [cls.READY, cls.INITIALIZING_COMPONENTS, cls.CONNECTING_SERVICES]
        return state in operational_states
    
    @classmethod
    def is_error_state(cls, state: str) -> bool:
        """Проверяет, является ли состояние ошибочным."""
        error_states = [cls.ERROR, cls.DEGRADED, cls.SHUTTING_DOWN]
        return state in error_states


class ComponentStatus:
    """Статус компонента системы."""
    
    def __init__(self, name: str, state: str = "uninitialized", error: Optional[str] = None):
        self.name = name
        self.state = state
        self.error = error
        self.last_update = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state,
            "error": self.error,
            "last_update": self.last_update
        }

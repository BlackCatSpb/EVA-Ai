"""
Типы для Component Initializer
Часть модуля component_initializer.py (разделение на логические компоненты)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class ComponentType(Enum):
    """Типы компонентов."""
    ML_UNIT = "ml_unit"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    TEXT_PROCESSOR = "text_processor"
    RESPONSE_GENERATOR = "response_generator"
    ETHICS_FRAMEWORK = "ethics_framework"
    ADAPTATION_MANAGER = "adaptation_manager"
    CONTRADICTION_MANAGER = "contradiction_manager"
    WEB_SEARCH_ENGINE = "web_search_engine"
    REASONING_ENGINE = "reasoning_engine"


class ComponentState(Enum):
    """Состояния компонента."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class ComponentInfo:
    """Информация о компоненте."""
    name: str
    component_type: ComponentType
    state: ComponentState
    dependencies: List[str] = field(default_factory=list)
    load_order: int = 0
    is_lazy: bool = False
    initialized_at: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "component_type": self.component_type.value if isinstance(self.component_type, ComponentType) else self.component_type,
            "state": self.state.value if isinstance(self.state, ComponentState) else self.state,
            "dependencies": self.dependencies,
            "load_order": self.load_order,
            "is_lazy": self.is_lazy,
            "initialized_at": self.initialized_at
        }


@dataclass
class Dependency:
    """Зависимость компонента."""
    component_name: str
    required: bool = True
    optional: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_name": self.component_name,
            "required": self.required,
            "optional": self.optional
        }

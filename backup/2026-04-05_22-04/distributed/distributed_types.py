"""
Типы для Distributed System
Часть модулей distributed_system.py (разделение на логические компоненты)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class NodeRole(Enum):
    """Роли узлов."""
    MASTER = "master"
    WORKER = "worker"
    COORDINATOR = "coordinator"
    OBSERVER = "observer"


class TaskState(Enum):
    """Состояния задач."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class NodeInfo:
    """Информация об узле."""
    node_id: str
    role: NodeRole
    address: str
    status: str
    resources: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "role": self.role.value if isinstance(self.role, NodeRole) else self.role,
            "address": self.address,
            "status": self.status,
            "resources": self.resources
        }


@dataclass
class DistributedTask:
    """Распределённая задача."""
    task_id: str
    task_type: str
    state: TaskState
    assigned_node: Optional[str] = None
    progress: float = 0.0
    result: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "state": self.state.value if isinstance(self.state, TaskState) else self.state,
            "assigned_node": self.assigned_node,
            "progress": self.progress,
            "result": self.result
        }

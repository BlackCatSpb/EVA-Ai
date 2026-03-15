"""
BaseJob: интерфейс задач для Автопилота
"""
from __future__ import annotations
from typing import Any
try:
    from cogniflex.core.deferred_command_system import CommandPriority
except Exception:  # fallback when deferred system is unavailable
    class CommandPriority:  # type: ignore
        LOW = 10
        MEDIUM = 20
        HIGH = 30


class BaseJob:
    job_type: str = "base"
    resource_class: str = "CPU"  # CPU | GPU | IO
    default_priority = CommandPriority.LOW
    is_cancelable: bool = True

    def __init__(self, brain: Any, **kwargs) -> None:
        self.brain = brain
        self.kwargs = kwargs
        self._canceled = False

    def cancel(self) -> None:
        self._canceled = True

    def run(self, context: dict) -> None:  # override
        raise NotImplementedError

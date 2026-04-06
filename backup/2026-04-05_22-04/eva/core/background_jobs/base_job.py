from __future__ import annotations

"""
BaseJob: интерфейс задач для Автопилота
"""
from typing import Any
import logging
try:
    from eva.core.deferred_command_system import CommandPriority
except (ImportError, ModuleNotFoundError) as e:  # fallback when deferred system is unavailable
    logger = logging.getLogger(__name__)
    logger.warning(f"Deferred command system unavailable: {e}")
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
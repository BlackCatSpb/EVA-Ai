"""
State management for CoreBrain - SystemState enum and state helpers.
"""
from typing import Dict, Any, Optional
import logging

try:
    from .system_state import SystemState, SystemStateManager
except Exception:
    from enum import Enum

    class SystemState(Enum):
        INITIALIZING = "initializing"
        READY = "ready"
        RUNNING = "running"
        PROCESSING = "processing"
        ERROR = "error"
        RECOVERY = "recovery"
        SHUTTING_DOWN = "shutting_down"
        OFFLINE = "offline"

    class SystemStateManager:
        def __init__(self): pass
        def set_state(self, state, reason=""): pass
        def get_state(self): return SystemState.INITIALIZING
        def get_system_summary(self): return {}

query_logger = logging.getLogger("eva_ai.core_brain.query_processing")


class StateMixin:
    """Mixin providing state management helpers to CoreBrain."""

    def _update_state(self, state: SystemState, reason: str = "") -> None:
        """Updates system state via state_manager if available."""
        if hasattr(self, 'state_manager') and self.state_manager and hasattr(self.state_manager, 'set_state'):
            self.state_manager.set_state(state, reason)

    def _get_current_state_value(self) -> str:
        """Returns current state as string."""
        if hasattr(self, 'state_manager') and self.state_manager and hasattr(self.state_manager, 'get_state'):
            state = self.state_manager.get_state()
            return state.value if hasattr(state, 'value') else str(state)
        return "unknown"

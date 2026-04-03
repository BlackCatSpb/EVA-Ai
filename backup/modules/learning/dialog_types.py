"""
Типы для Self Dialog Learning
Часть модуля self_dialog_learning.py (разделение на логические компоненты)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class DialogState(Enum):
    """Состояния диалога."""
    IDLE = "idle"
    LEARNING = "learning"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"


class LearningMode(Enum):
    """Режимы обучения."""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    SELF_IMPROVEMENT = "self_improvement"


@dataclass
class DialogTurn:
    """Ответ в диалоге."""
    turn_id: str
    user_input: str
    system_response: str
    timestamp: float
    feedback: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "user_input": self.user_input,
            "system_response": self.system_response,
            "timestamp": self.timestamp,
            "feedback": self.feedback
        }


@dataclass
class LearningSession:
    """Сессия обучения."""
    session_id: str
    mode: LearningMode
    state: DialogState
    turns: List[DialogTurn] = field(default_factory=list)
    start_time: float = 0.0
    end_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "mode": self.mode.value if isinstance(self.mode, LearningMode) else self.mode,
            "state": self.state.value if isinstance(self.state, DialogState) else self.state,
            "turns": [t.to_dict() for t in self.turns],
            "start_time": self.start_time,
            "end_time": self.end_time
        }

"""
Chain-of-Thought (CoT) logging для прозрачности рассуждений.
Форматирует внутренние шаги в JSON для рендеринга в GUI.
"""
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
import time

logger = logging.getLogger("eva_ai.cot_logger")

@dataclass
class CoTStage:
    """Один этап рассуждения в цепочке."""
    stage: str  # A, B, C, hypothesis_test, validation, etc.
    hypotheses: List[str] = field(default_factory=list)
    selected: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертировать в словарь для JSON."""
        return {
            "stage": self.stage,
            "hypotheses": self.hypotheses,
            "selected": self.selected,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }

@dataclass
class CoTLog:
    """Полный лог цепочки рассуждений."""
    query: str
    stages: List[CoTStage] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_confidence: float = 0.0
    
    def add_stage(self, stage: CoTStage):
        """Добавить этап в лог."""
        self.stages.append(stage)
        
    def finalize(self, total_confidence: float):
        """Завершить лог."""
        self.end_time = time.time()
        self.total_confidence = total_confidence
        
    def to_json(self) -> str:
        """Конвертировать в JSON строку для SSE."""
        return json.dumps({
            "query": self.query,
            "stages": [s.to_dict() for s in self.stages],
            "total_confidence": round(self.total_confidence, 3),
            "processing_time": round(self.end_time - self.start_time, 3) if self.end_time else 0,
            "stage_count": len(self.stages)
        }, ensure_ascii=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертировать в словарь."""
        return {
            "query": self.query,
            "stages": [s.to_dict() for s in self.stages],
            "total_confidence": round(self.total_confidence, 3),
            "processing_time": round(self.end_time - self.start_time, 3) if self.end_time else 0,
            "stage_count": len(self.stages)
        }


class CoTLogger:
    """Логгер для Chain-of-Thought рассуждений."""
    
    def __init__(self, query: str):
        self.log = CoTLog(query=query)
        self._current_stage: Optional[CoTStage] = None
        
    def start_stage(self, stage_name: str, reasoning: str = ""):
        """Начать новый этап."""
        self._current_stage = CoTStage(stage=stage_name, reasoning=reasoning)
        
    def add_hypothesis(self, hypothesis: str):
        """Добавить гипотезу на текущий этап."""
        if self._current_stage:
            self._current_stage.hypotheses.append(hypothesis)
            
    def select_hypothesis(self, selected: str, confidence: float):
        """Выбрать гипотезу и установить уверенность."""
        if self._current_stage:
            self._current_stage.selected = selected
            self._current_stage.confidence = confidence
            
    def set_metadata(self, key: str, value: Any):
        """Установить метаданные для текущего этапа."""
        if self._current_stage:
            self._current_stage.metadata[key] = value
            
    def end_stage(self):
        """Завершить текущий этап."""
        if self._current_stage:
            self.log.add_stage(self._current_stage)
            self._current_stage = None
            
    def finalize(self, total_confidence: float) -> CoTLog:
        """Завершить лог и вернуть объект."""
        self.log.finalize(total_confidence)
        return self.log
    
    def get_current_stage(self) -> Optional[CoTStage]:
        """Получить текущий этап."""
        return self._current_stage


def create_cot_logger(query: str) -> CoTLogger:
    """Создать CoT логгер для запроса."""
    return CoTLogger(query)


# Утилиты для форматирования под SSE
def format_for_sse(cot_log: CoTLog) -> Dict[str, Any]:
    """Форматировать для Server-Sent Events."""
    return {
        "type": "cot_update",
        "data": cot_log.to_dict()
    }

def create_stage_json(stage: str, hypotheses: List[str], selected: str, 
                      confidence: float, reasoning: str = "") -> str:
    """Создать JSON для одного этапа (быстрое создание)."""
    stage_obj = CoTStage(
        stage=stage,
        hypotheses=hypotheses,
        selected=selected,
        confidence=confidence,
        reasoning=reasoning
    )
    return json.dumps(stage_obj.to_dict(), ensure_ascii=False)

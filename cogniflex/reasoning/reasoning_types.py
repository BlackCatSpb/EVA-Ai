"""
Типы данных для Self-Reasoning Engine
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
import time


class ReasoningPhase(Enum):
    """Фазы процесса рассуждения"""
    INITIAL = "initial"
    GENERATION = "generation"
    ETHICS_CHECK = "ethics_check"
    CONTRADICTION_CHECK = "contradiction_check"
    KNOWLEDGE_CHECK = "knowledge_check"
    CLARIFICATION = "clarification"
    FINAL_SYNTHESIS = "final_synthesis"


@dataclass
class ReasoningStep:
    """Шаг рассуждения"""
    phase: str
    thought: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "phase": self.phase,
            "thought": self.thought,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class ReasoningResult:
    """Результат рассуждения"""
    final_response: str
    confidence: float
    iterations: int
    steps: List[ReasoningStep] = field(default_factory=list)
    clarification_questions: List[str] = field(default_factory=list)
    reasoning_chain_ids: List[str] = field(default_factory=list)
    
    # Метаданные
    query: str = ""
    processing_time: float = 0.0
    model_used: str = "qwen3.5-0.8b"
    
    def to_dict(self) -> Dict:
        return {
            "final_response": self.final_response,
            "confidence": self.confidence,
            "iterations": self.iterations,
            "steps": [s.to_dict() for s in self.steps],
            "clarification_questions": self.clarification_questions,
            "reasoning_chain_ids": self.reasoning_chain_ids,
            "query": self.query,
            "processing_time": self.processing_time,
            "model_used": self.model_used
        }
    
    @property
    def reasoning_text(self) -> str:
        """Текстовое представление для GUI панели рассуждений"""
        lines = []
        lines.append("=" * 40)
        lines.append("ПРОЦЕСС РАССУЖДЕНИЯ")
        lines.append("=" * 40)
        lines.append(f"Запрос: {self.query}")
        lines.append(f"Итераций: {self.iterations}")
        lines.append(f"Уверенность: {self.confidence:.2f}")
        lines.append("-" * 40)
        
        for i, step in enumerate(self.steps, 1):
            lines.append(f"[{i}] {step.phase.upper()}")
            lines.append(f"    Мысль: {step.thought[:100]}...")
            lines.append(f"    Уверенность: {step.confidence:.2f}")
            lines.append("")
        
        if self.clarification_questions:
            lines.append("-" * 40)
            lines.append("Уточняющие вопросы:")
            for q in self.clarification_questions:
                lines.append(f"  • {q}")
        
        lines.append("-" * 40)
        lines.append(f"Ответ: {self.final_response[:200]}...")
        
        return "\n".join(lines)


@dataclass
class AnalysisResult:
    """Результат анализа ответа"""
    ethics_result: Optional[Dict[str, Any]] = None
    contradiction_result: Optional[Dict[str, Any]] = None
    knowledge_result: Optional[Dict[str, Any]] = None
    
    # Дополнительные данные
    gaps_found: List[Dict] = field(default_factory=list)
    missing_entities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "ethics_result": self.ethics_result,
            "contradiction_result": self.contradiction_result,
            "knowledge_result": self.knowledge_result,
            "gaps_found": self.gaps_found,
            "missing_entities": self.missing_entities
        }

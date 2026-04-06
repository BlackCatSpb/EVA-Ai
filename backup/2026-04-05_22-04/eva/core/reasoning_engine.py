"""
Модуль рассуждений ЕВА - внутренний диалог и многоуровневое мышление
Интегрирует все компоненты системы для глубокого анализа перед ответом пользователю

Этот модуль является фасадом для подмодулей:
- engine_core: ReasoningEngine, process_query(), жизненный цикл
- engine_steps: ReasoningPhase, ReasoningStep, InternalDialogue, шаги рассуждения
- engine_analysis: анализ, оценка факторов, скоринг
- engine_synthesis: синтез, объединение результатов, финальная генерация
"""

from .engine_core import ReasoningEngine, create_reasoning_engine
from .engine_steps import ReasoningPhase, ReasoningStep, InternalDialogue

__all__ = [
    'ReasoningEngine',
    'create_reasoning_engine',
    'ReasoningPhase',
    'ReasoningStep',
    'InternalDialogue',
]

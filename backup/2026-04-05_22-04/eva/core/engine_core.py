"""
Модуль ядра движка рассуждений ЕВА - основной класс ReasoningEngine, process_query(), жизненный цикл
"""

import time
import logging
from typing import Dict, Any, Optional

from .engine_steps import (
    ReasoningPhase,
    ReasoningStep,
    InternalDialogue,
    ReasoningStepsMixin,
)
from .engine_analysis import ReasoningAnalysisMixin
from .engine_synthesis import ReasoningSynthesisMixin

logger = logging.getLogger("eva.reasoning")


class ReasoningEngine(ReasoningStepsMixin, ReasoningAnalysisMixin, ReasoningSynthesisMixin):
    """
    Движок рассуждений ЕВА - реализует внутренний диалог системы

    Перед ответом пользователю система проводит многоуровневый анализ:
    1. Извлечение контекста из памяти (граф знаний, горячее окно)
    2. Проверка на противоречия
    3. Поиск актуальной информации (веб-поиск при необходимости)
    4. Этическая проверка
    5. Синтез ответа через внутренний диалог
    6. Рефлексия и самокоррекция
    """

    MAX_REASONING_STEPS = 3

    def __init__(self, brain=None, config: Optional[Dict] = None):
        """
        Инициализация движка рассуждений

        Args:
            brain: Ссылка на CoreBrain
            config: Конфигурация рассуждений
        """
        self.brain = brain
        self.config = config or {}
        self.active_dialogues: Dict[str, InternalDialogue] = {}
        self.reasoning_history: list = []
        self.max_history_size = self.config.get('max_history', 100)

        self.enabled_phases = {
            ReasoningPhase.INITIAL_ANALYSIS: True,
            ReasoningPhase.MEMORY_RETRIEVAL: True,
            ReasoningPhase.CONTRADICTION_CHECK: True,
            ReasoningPhase.WEB_SEARCH: self.config.get('enable_web_search', True),
            ReasoningPhase.KNOWLEDGE_GRAPH_QUERY: True,
            ReasoningPhase.ETHICS_CHECK: True,
            ReasoningPhase.ANALYTICS_CHECK: self.config.get('enable_analytics', True),
            ReasoningPhase.PERFORMANCE_ANALYSIS: self.config.get('enable_performance', True),
            ReasoningPhase.SYNTHESIS: True,
            ReasoningPhase.REFLECTION: True,
            ReasoningPhase.FINAL_ANSWER: True
        }

        self.thresholds = {
            'min_confidence_for_answer': 0.6,
            'web_search_threshold': 0.4,
            'reflection_threshold': 0.7,
            'max_reasoning_time': 30.0
        }

        logger.info("ReasoningEngine инициализирован")

    def reason(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Основной метод рассуждения - внутренний диалог перед ответом

        Args:
            query: Запрос пользователя
            context: Дополнительный контекст

        Returns:
            Dict с результатом рассуждения и финальным ответом
        """
        start_time = time.time()
        query_id = f"reason_{int(start_time * 1000)}"

        dialogue = InternalDialogue(
            query_id=query_id,
            original_query=query
        )
        self.active_dialogues[query_id] = dialogue

        try:
            if self.enabled_phases[ReasoningPhase.INITIAL_ANALYSIS]:
                self._initial_analysis(dialogue, query, context)

            if len(dialogue.steps) >= self.MAX_REASONING_STEPS:
                logger.warning("Max reasoning steps reached, stopping")
                return self._finalize_answer(dialogue, start_time)

            if self.enabled_phases[ReasoningPhase.MEMORY_RETRIEVAL]:
                self._memory_retrieval(dialogue)

            if self.enabled_phases[ReasoningPhase.CONTRADICTION_CHECK]:
                self._check_contradictions(dialogue)

            if self.enabled_phases[ReasoningPhase.WEB_SEARCH]:
                self._web_search_if_needed(dialogue)

            accumulated_context = {}
            for step in dialogue.steps:
                if step.phase == ReasoningPhase.INITIAL_ANALYSIS:
                    if isinstance(step.result, dict):
                        accumulated_context['query_type'] = step.result.get('query_type', '')
                        accumulated_context['entities'] = step.result.get('entities', [])
                elif step.phase == ReasoningPhase.MEMORY_RETRIEVAL:
                    if isinstance(step.result, dict):
                        accumulated_context['memory'] = step.result.get('relevant_memories', [])
                    elif isinstance(step.result, list):
                        accumulated_context['memory'] = step.result
                elif step.phase == ReasoningPhase.CONTRADICTION_CHECK:
                    if isinstance(step.result, dict):
                        accumulated_context['contradictions'] = step.result.get('contradictions', [])
                    elif isinstance(step.result, list):
                        accumulated_context['contradictions'] = step.result
                elif step.phase == ReasoningPhase.WEB_SEARCH:
                    if isinstance(step.result, dict):
                        accumulated_context['web_search'] = step.result.get('search_results', [])
                    elif isinstance(step.result, list):
                        accumulated_context['web_search'] = step.result

            if self.enabled_phases[ReasoningPhase.KNOWLEDGE_GRAPH_QUERY]:
                self._query_knowledge_graph(dialogue)

            if self.enabled_phases[ReasoningPhase.ETHICS_CHECK]:
                self._ethics_check(dialogue)

            if self.enabled_phases[ReasoningPhase.ANALYTICS_CHECK]:
                self._analytics_check(dialogue)

            if self.enabled_phases[ReasoningPhase.PERFORMANCE_ANALYSIS]:
                self._performance_analysis(dialogue)

            if self.enabled_phases[ReasoningPhase.SYNTHESIS]:
                self._synthesize_answer(dialogue)

            if self.enabled_phases[ReasoningPhase.REFLECTION]:
                self._reflection(dialogue)

            result = self._finalize_answer(dialogue, start_time)

            self._add_to_history(dialogue)

            self._update_memory_graph(dialogue)

            return result

        except Exception as e:
            logger.error(f"Ошибка в процессе рассуждения: {e}", exc_info=True)
            return {
                'query_id': query_id,
                'answer': f"Произошла ошибка при обработке запроса: {str(e)}",
                'confidence': 0.0,
                'processing_time': time.time() - start_time,
                'error': str(e)
            }
        finally:
            if query_id in self.active_dialogues:
                del self.active_dialogues[query_id]

    def process_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Обёртка над reason() для внешнего использования.

        Args:
            query: Запрос пользователя
            context: Дополнительный контекст

        Returns:
            Dict с результатом рассуждения
        """
        return self.reason(query, context)


def create_reasoning_engine(brain=None, config: Optional[Dict] = None) -> ReasoningEngine:
    """Factory function for creating a ReasoningEngine."""
    return ReasoningEngine(brain=brain, config=config)

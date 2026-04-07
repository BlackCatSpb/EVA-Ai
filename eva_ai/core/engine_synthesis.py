"""
Модуль синтеза ЕВА - синтез, объединение результатов, финальная генерация ответа
"""

import time
import logging
from typing import Dict, Any, Optional, List

from .engine_steps import (
    ReasoningPhase,
    ReasoningStep,
    InternalDialogue,
)

logger = logging.getLogger("eva_ai.reasoning")


class ReasoningSynthesisMixin:
    """Mixin with synthesis, result merging, and final response generation."""

    def _synthesize_answer(self, dialogue: InternalDialogue):
        logger.debug("Синтез ответа (skip internal generation)...")

        dialogue.steps.append(ReasoningStep(
            phase=ReasoningPhase.SYNTHESIS,
            query=dialogue.original_query,
            result={'synthesized': True, 'skipped': True},
            timestamp=time.time(),
            confidence=0.8
        ))

        logger.debug("Синтез завершен (пропущен)")

    def _direct_generate(self, query: str) -> str:
        if self.brain and hasattr(self.brain, 'generation_coordinator') and self.brain.generation_coordinator:
            try:
                response = self.brain.generation_coordinator.generate_response(
                    query, max_tokens=300
                )
                if response and response.strip():
                    return response
            except Exception as e:
                logger.debug(f"Ошибка прямой генерации через coordinator: {e}")

        if self.brain and hasattr(self.brain, 'ml_unit') and self.brain.ml_unit:
            try:
                ml_unit = self.brain.ml_unit
                if hasattr(ml_unit, 'generate_response'):
                    response = ml_unit.generate_response(query)
                    if response:
                        if isinstance(response, dict):
                            text = response.get('text') or response.get('response', '')
                            if text:
                                return text
                        elif isinstance(response, str):
                            return response
            except Exception as e:
                logger.debug(f"Ошибка прямой генерации через ml_unit: {e}")

        if self.brain and hasattr(self.brain, 'model_manager') and self.brain.model_manager:
            try:
                mm = self.brain.model_manager
                if hasattr(mm, 'generate'):
                    response = mm.generate(query)
                    if response:
                        if isinstance(response, dict):
                            return response.get('text', response.get('response', str(response)))
                        return str(response)
            except Exception as e:
                logger.debug(f"Ошибка прямой генерации через model_manager: {e}")

        return "Я получил ваш запрос, но не могу сейчас сгенерировать полный ответ."

    def _finalize_answer(self, dialogue: InternalDialogue, start_time: float) -> Dict:
        logger.debug("Финализация ответа...")

        final_answer = self._direct_generate(dialogue.original_query)

        if len(final_answer) > 500:
            final_answer = final_answer[:500]

        confidence = self._calculate_overall_confidence(dialogue)

        if len(final_answer) > 500:
            sentences = final_answer.rstrip().split('.')
            if len(sentences) > 2:
                final_answer = '.'.join(sentences[:3]) + '.'
            else:
                final_answer = final_answer[:500]

        processing_time = time.time() - start_time

        dialogue.final_answer = final_answer
        dialogue.confidence = confidence
        dialogue.processing_time = processing_time

        result = {
            'query_id': dialogue.query_id,
            'answer': final_answer,
            'confidence': confidence,
            'processing_time': processing_time,
            'reasoning_steps': len(dialogue.steps),
            'insights_count': len(dialogue.insights),
            'contradictions_found': len(dialogue.contradictions_found),
            'reasoning_phases': [s.phase.value for s in dialogue.steps]
        }

        logger.info(f"Рассуждение завершено: {len(dialogue.steps)} шагов, уверенность={confidence:.2f}")

        return result

    def _update_memory_graph(self, dialogue: InternalDialogue):
        logger.debug("Обновление графа памяти...")

        if not self.brain:
            return

        try:
            if hasattr(self.brain, 'knowledge_graph'):
                self.brain.knowledge_graph.add_concept(
                    id=f"query_{dialogue.query_id}",
                    name=dialogue.original_query,
                    description=dialogue.final_answer[:200],
                    strength=dialogue.confidence,
                    domain="user_queries"
                )

                initial_step = self._get_step_by_phase(dialogue, ReasoningPhase.INITIAL_ANALYSIS)
                if initial_step:
                    entities = initial_step.result.get('entities', [])
                    for entity in entities:
                        self.brain.knowledge_graph.add_relation(
                            from_concept=entity,
                            to_concept=f"query_{dialogue.query_id}",
                            relation_type="mentioned_in",
                            strength=0.7
                        )

                for i, insight in enumerate(dialogue.insights[:3]):
                    if insight and len(insight) > 10:
                        self.brain.knowledge_graph.add_concept(
                            id=f"insight_{dialogue.query_id}_{i}",
                            name=f"Инсайт {i+1}",
                            description=insight[:200],
                            strength=0.6,
                            domain="insights"
                        )

                if hasattr(self.brain, 'memory_graph_ml') and self.brain.memory_graph_ml:
                    try:
                        mgml = self.brain.memory_graph_ml
                        for insight in dialogue.insights:
                            if insight and len(insight) > 10:
                                mgml.add_insight(
                                    insight=insight,
                                    source_query=dialogue.original_query,
                                    metadata={
                                        'confidence': dialogue.confidence,
                                        'query_id': dialogue.query_id,
                                        'reasoning_steps': len(dialogue.steps)
                                    }
                                )
                        logger.debug(f"Инсайты добавлены в MemoryGraphML: {len(dialogue.insights)} шт")
                    except Exception as e:
                        logger.debug(f"Ошибка добавления в MemoryGraphML: {e}")

                logger.debug(f"Граф памяти обновлен для {dialogue.query_id}")
        except Exception as e:
            logger.debug(f"Ошибка обновления графа памяти: {e}")

    def _add_to_history(self, dialogue: InternalDialogue):
        self.reasoning_history.append(dialogue)

        if len(self.reasoning_history) > self.max_history_size:
            self.reasoning_history = self.reasoning_history[-self.max_history_size:]

    def _get_step_by_phase(self, dialogue: InternalDialogue, phase: ReasoningPhase) -> Optional[ReasoningStep]:
        for step in dialogue.steps:
            if step.phase == phase:
                return step
        return None

    def get_reasoning_stats(self) -> Dict:
        if not self.reasoning_history:
            return {'total_reasonings': 0}

        avg_confidence = sum(d.confidence for d in self.reasoning_history) / len(self.reasoning_history)
        avg_time = sum(d.processing_time for d in self.reasoning_history) / len(self.reasoning_history)

        return {
            'total_reasonings': len(self.reasoning_history),
            'average_confidence': avg_confidence,
            'average_processing_time': avg_time,
            'active_dialogues': len(self.active_dialogues)
        }

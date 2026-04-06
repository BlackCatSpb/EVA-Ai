"""
Модуль анализа ЕВА - методы анализа, оценка факторов, скоринг
"""

import time
import logging
from typing import Dict, Any, Optional, List

from .engine_steps import (
    ReasoningPhase,
    ReasoningStep,
    InternalDialogue,
)

logger = logging.getLogger("eva.reasoning")


class ReasoningAnalysisMixin:
    """Mixin with all analysis, evaluation, and scoring methods."""

    def _ethics_check(self, dialogue: InternalDialogue):
        logger.debug("Этическая проверка...")

        ethics_result = {'approved': True, 'issues': []}

        if self.brain and hasattr(self.brain, 'ethics_framework'):
            try:
                check = self.brain.ethics_framework.analyze_content(dialogue.original_query)
                ethics_result = {
                    'approved': check.approved if hasattr(check, 'approved') else True,
                    'issues': check.violations if hasattr(check, 'violations') else [],
                    'score': check.confidence if hasattr(check, 'confidence') else 1.0
                }
            except Exception as e:
                logger.debug(f"Ошибка этической проверки: {e}")

        step = ReasoningStep(
            phase=ReasoningPhase.ETHICS_CHECK,
            query=dialogue.original_query,
            result=ethics_result,
            timestamp=time.time(),
            confidence=1.0 if ethics_result['approved'] else 0.0,
            metadata={'ethics_approved': ethics_result['approved']}
        )
        dialogue.steps.append(step)

        if not ethics_result['approved']:
            logger.warning(f"Этические проблемы: {ethics_result['issues']}")

    def _analytics_check(self, dialogue: InternalDialogue):
        logger.debug("Сбор аналитических данных...")

        analytics_data = {
            'system_state': {},
            'component_health': {},
            'metrics': {},
            'insights': []
        }

        try:
            if self.brain and hasattr(self.brain, 'analytics_manager'):
                try:
                    analytics_manager = self.brain.analytics_manager
                    if hasattr(analytics_manager, 'get_system_analytics'):
                        analytics_data['system_state'] = analytics_manager.get_system_analytics()
                    if hasattr(analytics_manager, 'get_metrics'):
                        analytics_data['metrics'] = analytics_manager.get_metrics()
                except Exception as e:
                    logger.debug(f"Ошибка получения аналитики от analytics_manager: {e}")

            if self.brain and hasattr(self.brain, 'components'):
                for comp_name, comp in self.brain.components.items():
                    if comp is not None:
                        try:
                            if hasattr(comp, 'get_status'):
                                status = comp.get_status()
                                analytics_data['component_health'][comp_name] = status
                            elif hasattr(comp, 'health') or hasattr(comp, 'is_ready'):
                                analytics_data['component_health'][comp_name] = {
                                    'available': True,
                                    'health': getattr(comp, 'health', 'unknown'),
                                    'ready': getattr(comp, 'is_ready', None)
                                }
                        except Exception:
                            pass

            if analytics_data['component_health']:
                healthy_count = sum(1 for h in analytics_data['component_health'].values()
                                  if h.get('available', False))
                total_count = len(analytics_data['component_health'])
                health_ratio = healthy_count / max(total_count, 1)

                if health_ratio < 0.5:
                    analytics_data['insights'].append(f"Только {healthy_count}/{total_count} компонентов работают")

        except Exception as e:
            logger.debug(f"Ошибка сбора аналитики: {e}")

        step = ReasoningStep(
            phase=ReasoningPhase.ANALYTICS_CHECK,
            query=dialogue.original_query,
            result=analytics_data,
            timestamp=time.time(),
            confidence=0.8,
            metadata={'components_checked': len(analytics_data.get('component_health', {}))}
        )
        dialogue.steps.append(step)

    def _performance_analysis(self, dialogue: InternalDialogue):
        logger.debug("Анализ производительности...")

        perf_data = {
            'response_time': 0.0,
            'memory_usage': 0.0,
            'cpu_usage': 0.0,
            'recommendations': [],
            'bottlenecks': []
        }

        try:
            if self.brain and hasattr(self.brain, 'performance_analyzer'):
                try:
                    pa = self.brain.performance_analyzer
                    if hasattr(pa, 'get_performance_report'):
                        report = pa.get_performance_report()
                        perf_data.update(report)
                    elif hasattr(pa, 'analyze'):
                        perf_data = pa.analyze()
                except Exception as e:
                    logger.debug(f"Ошибка анализа производительности: {e}")

            if not perf_data.get('response_time') and self.brain:
                if hasattr(self.brain, 'metrics_manager'):
                    try:
                        mm = self.brain.metrics_manager
                        if hasattr(mm, 'get_average_response_time'):
                            perf_data['response_time'] = mm.get_average_response_time()
                    except Exception:
                        pass

            if perf_data.get('response_time', 0) > 5.0:
                perf_data['bottlenecks'].append('Медленный ответ системы')
                perf_data['recommendations'].append('Оптимизировать кэширование')

            if perf_data.get('memory_usage', 0) > 80:
                perf_data['bottlenecks'].append('Высокое использование памяти')
                perf_data['recommendations'].append('Очистить неиспользуемые данные')

        except Exception as e:
            logger.debug(f"Ошибка анализа производительности: {e}")

        step = ReasoningStep(
            phase=ReasoningPhase.PERFORMANCE_ANALYSIS,
            query=dialogue.original_query,
            result=perf_data,
            timestamp=time.time(),
            confidence=0.7,
            metadata={
                'bottlenecks_count': len(perf_data.get('bottlenecks', [])),
                'recommendations_count': len(perf_data.get('recommendations', []))
            }
        )
        dialogue.steps.append(step)

    def _generate_internal_questions(self, dialogue: InternalDialogue, context: Dict) -> List[str]:
        questions = []
        query_type = context.get('query_type', 'general')

        ml_entities = context.get('ml_entities', [])
        graph_context = context.get('graph_context', {})

        if ml_entities:
            for entity in ml_entities[:3]:
                if isinstance(entity, dict):
                    entity_name = entity.get('name', entity.get('id', ''))
                    if entity_name:
                        questions.append(f"Как связана сущность '{entity_name}' с запросом '{dialogue.original_query}'?")

        questions.extend([
            f"Какие ключевые аспекты запроса '{dialogue.original_query}' нужно учесть?",
            "Какая информация из памяти наиболее релевантна?",
            "Есть ли пробелы в знаниях, которые нужно заполнить?"
        ])

        if graph_context:
            patterns = graph_context.get('patterns', [])
            if patterns:
                questions.append("Какие паттерны из прошлых рассуждений применимы?")

        if query_type == 'explanatory':
            questions.extend([
                "Как объяснить это просто и понятно?",
                "Какие примеры помогут понять?",
                "На какие подвопросы стоит разбить объяснение?"
            ])
        elif query_type == 'factual':
            questions.extend([
                "Насколько точны имеющиеся факты?",
                "Нужно ли уточнить информацию?",
                "Какие источники подтверждают эти факты?"
            ])
        elif query_type == 'comparative':
            questions.extend([
                "Какие критерии сравнения важны?",
                "Что общего и в чем различия?",
                "Какой вывод из сравнения?"
            ])

        if context.get('contradictions'):
            questions.append("Как разрешить обнаруженные противоречия?")

        if not context.get('memory') and not context.get('web_search'):
            questions.append("Как ответить при недостатке информации?")

        return questions[:5]

    def _ask_internal(self, question: str, context: Dict, dialogue: InternalDialogue) -> str:
        if self.brain and hasattr(self.brain, 'generation_coordinator'):
            try:
                prompt = f"""Вопрос для самоанализа: {question}
Контекст: {context.get('original_query', '')}
Ответь кратко и по существу:"""

                response = self.brain.generation_coordinator.generate_response(
                    prompt, max_tokens=100
                )
                if response and response.strip():
                    return response
            except Exception as e:
                logger.debug(f"Ошибка внутреннего вопроса: {e}")

        if self.brain:
            try:
                if hasattr(self.brain, 'ml_unit') and self.brain.ml_unit:
                    ml_unit = self.brain.ml_unit
                    if hasattr(ml_unit, 'generate_response'):
                        response = ml_unit.generate_response(question)
                        if response and isinstance(response, dict):
                            return response.get('text', response.get('response', ''))
                        elif isinstance(response, str):
                            return response
                if hasattr(self.brain, 'model_manager') and self.brain.model_manager:
                    mm = self.brain.model_manager
                    if hasattr(mm, 'generate'):
                        response = mm.generate(question)
                        if response:
                            return response
            except Exception as e:
                logger.debug(f"Fallback generation failed: {e}")

        return ""

    def _calculate_synthesis_confidence(self, insights: List[Dict]) -> float:
        if not insights:
            return 0.3

        base_confidence = 0.5

        for insight in insights:
            answer = insight.get('answer', '')
            if len(answer) > 20:
                base_confidence += 0.1

        return min(0.95, base_confidence)

    def _reflection(self, dialogue: InternalDialogue):
        logger.debug("Фаза рефлексии...")

        synthesis_step = self._get_step_by_phase(dialogue, ReasoningPhase.SYNTHESIS)
        if not synthesis_step:
            return

        insights = synthesis_step.result.get('insights', [])

        reflection_notes = []

        for insight in insights:
            answer = insight.get('answer', '')
            if len(answer) < 10 or self._is_gibberish(answer):
                reflection_notes.append(f"Инсайт слабый: {insight.get('question', '')}")

        needs_regeneration = len(reflection_notes) > 2

        step = ReasoningStep(
            phase=ReasoningPhase.REFLECTION,
            query=dialogue.original_query,
            result={
                'reflection_notes': reflection_notes,
                'needs_regeneration': needs_regeneration,
                'quality_score': 1.0 - (len(reflection_notes) / max(len(insights), 1))
            },
            timestamp=time.time(),
            confidence=0.8 if not needs_regeneration else 0.4,
            metadata={'issues_found': len(reflection_notes)}
        )
        dialogue.steps.append(step)

        logger.debug(f"Рефлексия завершена: {len(reflection_notes)} замечаний")

    def _is_gibberish(self, text: str) -> bool:
        words = text.split()
        if not words:
            return True

        avg_len = sum(len(w) for w in words) / len(words)
        if avg_len > 20:
            return True

        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            return True

        return False

    def _calculate_overall_confidence(self, dialogue: InternalDialogue) -> float:
        confidences = [s.confidence for s in dialogue.steps]
        if not confidences:
            return 0.3

        weights = {
            ReasoningPhase.INITIAL_ANALYSIS: 0.1,
            ReasoningPhase.MEMORY_RETRIEVAL: 0.15,
            ReasoningPhase.CONTRADICTION_CHECK: 0.15,
            ReasoningPhase.WEB_SEARCH: 0.1,
            ReasoningPhase.KNOWLEDGE_GRAPH_QUERY: 0.15,
            ReasoningPhase.ETHICS_CHECK: 0.1,
            ReasoningPhase.SYNTHESIS: 0.2,
            ReasoningPhase.REFLECTION: 0.05
        }

        total_weight = 0
        weighted_sum = 0

        for step in dialogue.steps:
            weight = weights.get(step.phase, 0.1)
            weighted_sum += step.confidence * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.3

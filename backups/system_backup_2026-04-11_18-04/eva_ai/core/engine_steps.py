"""
Модуль шагов рассуждений ЕВА - фазы, шаги, генерация шагов
"""

import re
import time
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("eva_ai.reasoning")


class ReasoningPhase:
    INITIAL_ANALYSIS = "initial_analysis"
    MEMORY_RETRIEVAL = "memory_retrieval"
    CONTRADICTION_CHECK = "contradiction_check"
    WEB_SEARCH = "web_search"
    KNOWLEDGE_GRAPH_QUERY = "knowledge_graph_query"
    ETHICS_CHECK = "ethics_check"
    ANALYTICS_CHECK = "analytics_check"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    SYNTHESIS = "synthesis"
    REFLECTION = "reflection"
    FINAL_ANSWER = "final_answer"


from enum import Enum as _Enum


class _ReasoningPhaseEnum(_Enum):
    INITIAL_ANALYSIS = "initial_analysis"
    MEMORY_RETRIEVAL = "memory_retrieval"
    CONTRADICTION_CHECK = "contradiction_check"
    WEB_SEARCH = "web_search"
    KNOWLEDGE_GRAPH_QUERY = "knowledge_graph_query"
    ETHICS_CHECK = "ethics_check"
    ANALYTICS_CHECK = "analytics_check"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    SYNTHESIS = "synthesis"
    REFLECTION = "reflection"
    FINAL_ANSWER = "final_answer"


ReasoningPhase = _ReasoningPhaseEnum


from dataclasses import dataclass, field


@dataclass
class ReasoningStep:
    phase: ReasoningPhase
    query: str
    result: Any
    timestamp: float
    confidence: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class InternalDialogue:
    query_id: str
    original_query: str
    steps: List[ReasoningStep] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    contradictions_found: List[Dict] = field(default_factory=list)
    knowledge_gaps: List[str] = field(default_factory=list)
    final_answer: str = ""
    confidence: float = 0.0
    processing_time: float = 0.0


class ReasoningStepsMixin:
    """Mixin with all reasoning step and phase methods."""

    def _initial_analysis(self, dialogue: InternalDialogue, query: str, context: Optional[Dict]):
        logger.debug(f"Начальный анализ запроса: {query[:50]}...")

        analysis = {
            'query_type': self._classify_query(query),
            'complexity': self._estimate_complexity(query),
            'entities': self._extract_entities(query),
            'intent': self._extract_intent(query),
            'urgency': self._estimate_urgency(query),
            'context': context or {}
        }

        step = ReasoningStep(
            phase=ReasoningPhase.INITIAL_ANALYSIS,
            query=query,
            result=analysis,
            timestamp=time.time(),
            confidence=0.8,
            metadata={'query_length': len(query)}
        )
        dialogue.steps.append(step)

        logger.debug(f"Анализ завершен: тип={analysis['query_type']}, сложность={analysis['complexity']}")

    def _classify_query(self, query: str) -> str:
        query_lower = query.lower()

        if any(w in query_lower for w in ['почему', 'зачем', 'как', 'объясни']):
            return 'explanatory'
        elif any(w in query_lower for w in ['кто', 'что', 'где', 'когда', 'сколько']):
            return 'factual'
        elif any(w in query_lower for w in ['сравни', 'разница', 'отличие']):
            return 'comparative'
        elif any(w in query_lower for w in ['мнение', 'что думаешь', 'как считаешь']):
            return 'opinion'
        elif any(w in query_lower for w in ['помоги', 'помощь', 'подскажи']):
            return 'help'
        else:
            return 'general'

    def _estimate_complexity(self, query: str) -> str:
        words = query.split()
        length = len(words)

        if length <= 3:
            return 'simple'
        elif length <= 8:
            return 'medium'
        else:
            return 'complex'

    def _extract_entities(self, text: str) -> List[str]:
        entities = []

        proper_nouns = re.findall(r'\b[А-Я][а-я]+\b', text)
        entities.extend(proper_nouns)

        long_words = re.findall(r'\b[а-яА-Я]{7,}\b', text)
        entities.extend([w for w in long_words if w.lower() not in
                        ['сегодня', 'завтра', 'вчера', 'сейчас', 'потому', 'поэтому']])

        return list(set(entities))

    def _extract_intent(self, query: str) -> str:
        query_lower = query.lower()

        intents = {
            'learn': ['учить', 'обучи', 'расскажи', 'объясни', 'как'],
            'create': ['создай', 'напиши', 'сгенерируй', 'придумай'],
            'analyze': ['анализ', 'проанализируй', 'оцени', 'сравни'],
            'find': ['найди', 'поиск', 'где', 'когда', 'кто'],
            'validate': ['верно', 'правда', 'проверь', 'это так']
        }

        for intent, keywords in intents.items():
            if any(kw in query_lower for kw in keywords):
                return intent

        return 'general'

    def _estimate_urgency(self, query: str) -> str:
        urgent_words = ['срочно', 'быстро', 'немедленно', 'сейчас', 'asap']
        if any(w in query.lower() for w in urgent_words):
            return 'high'
        return 'normal'

    def _memory_retrieval(self, dialogue: InternalDialogue):
        logger.debug("Извлечение информации из памяти...")

        memories = []

        if self.brain:
            if hasattr(self.brain, 'memory_manager'):
                try:
                    hot_window = self.brain.memory_manager.get_hot_window_data()
                    if hot_window:
                        memories.append({
                            'source': 'hot_window',
                            'data': hot_window,
                            'relevance': 0.9
                        })
                except Exception as e:
                    logger.debug(f"Ошибка доступа к горячему окну: {e}")

            if hasattr(self.brain, 'knowledge_graph'):
                try:
                    entities = dialogue.steps[0].result.get('entities', [])
                    for entity in entities[:3]:
                        related = self.brain.knowledge_graph.get_related_concepts(entity)
                        if related:
                            memories.append({
                                'source': 'knowledge_graph',
                                'entity': entity,
                                'data': related,
                                'relevance': 0.8
                            })
                except Exception as e:
                    logger.debug(f"Ошибка доступа к графу знаний: {e}")

            if hasattr(self.brain, 'get_dialog_history'):
                try:
                    history = self.brain.get_dialog_history(limit=5)
                    if history:
                        memories.append({
                            'source': 'dialog_history',
                            'data': history,
                            'relevance': 0.7
                        })
                except Exception as e:
                    logger.debug(f"Ошибка доступа к истории: {e}")

        step = ReasoningStep(
            phase=ReasoningPhase.MEMORY_RETRIEVAL,
            query=dialogue.original_query,
            result=memories,
            timestamp=time.time(),
            confidence=0.7 if memories else 0.3,
            metadata={'memory_count': len(memories)}
        )
        dialogue.steps.append(step)

        logger.debug(f"Извлечено {len(memories)} источников из памяти")

    def _check_contradictions(self, dialogue: InternalDialogue):
        logger.debug("Проверка на противоречия...")

        contradictions = []

        if self.brain and hasattr(self.brain, 'contradiction_manager'):
            try:
                memory_step = self._get_step_by_phase(dialogue, ReasoningPhase.MEMORY_RETRIEVAL)
                if memory_step and memory_step.result:
                    for memory in memory_step.result:
                        if isinstance(memory.get('data'), dict):
                            contradictions = self.brain.contradiction_manager.detect_contradictions(
                                str(memory['data'])
                            )
            except Exception as e:
                logger.debug(f"Ошибка проверки противоречий: {e}")

        dialogue.contradictions_found = contradictions

        step = ReasoningStep(
            phase=ReasoningPhase.CONTRADICTION_CHECK,
            query=dialogue.original_query,
            result={'contradictions': contradictions, 'has_contradictions': len(contradictions) > 0},
            timestamp=time.time(),
            confidence=0.9 if not contradictions else 0.5,
            metadata={'contradiction_count': len(contradictions)}
        )
        dialogue.steps.append(step)

        if contradictions:
            logger.warning(f"Обнаружено {len(contradictions)} противоречий")

    def _web_search_if_needed(self, dialogue: InternalDialogue):
        logger.debug("Оценка необходимости веб-поиска...")

        memory_step = self._get_step_by_phase(dialogue, ReasoningPhase.MEMORY_RETRIEVAL)
        confidence = memory_step.confidence if memory_step else 0.0

        search_results = []

        if confidence < self.thresholds['web_search_threshold']:
            if self.brain and hasattr(self.brain, 'web_search_engine'):
                try:
                    query = dialogue.original_query
                    search_results = self.brain.web_search_engine.search(query, max_results=3)
                    logger.info(f"Выполнен веб-поиск: найдено {len(search_results)} результатов")
                except Exception as e:
                    logger.debug(f"Ошибка веб-поиска: {e}")

        step = ReasoningStep(
            phase=ReasoningPhase.WEB_SEARCH,
            query=dialogue.original_query,
            result=search_results,
            timestamp=time.time(),
            confidence=0.6 if search_results else 0.0,
            metadata={'search_performed': len(search_results) > 0, 'results_count': len(search_results)}
        )
        dialogue.steps.append(step)

    def _query_knowledge_graph(self, dialogue: InternalDialogue):
        logger.debug("Запрос к графу знаний через MemoryGraphML...")

        knowledge = []
        graph_context = {}

        if self.brain and hasattr(self.brain, 'memory_graph_ml') and self.brain.memory_graph_ml:
            try:
                for level in range(self.brain.memory_graph_ml.fractal_levels):
                    level_context = self.brain.memory_graph_ml.get_fractal_context(
                        dialogue.original_query,
                        level=level
                    )
                    if level_context:
                        graph_context[f'level_{level}'] = level_context
                        related = level_context.get('related_concepts', [])
                        patterns = level_context.get('patterns', [])
                        knowledge.extend([f"[L{level}] {c}" for c in related[:10]])
                        knowledge.extend([f"[Pattern] {p.get('pattern_id', 'unknown')}" for p in patterns[:5]])

                logger.debug(f"Получен фрактальный контекст из {len(graph_context)} уровней")
            except Exception as e:
                logger.debug(f"MemoryGraphML недоступен: {e}")

        if not knowledge and self.brain and hasattr(self.brain, 'knowledge_graph'):
            try:
                entities = dialogue.steps[0].result.get('entities', []) if dialogue.steps else []

                for entity in entities:
                    related = self.brain.knowledge_graph.find_path_between_concepts(
                        entity, dialogue.original_query
                    )
                    if related:
                        knowledge.append(related)

                    facts = self.brain.knowledge_graph.get_entity_facts(entity)
                    if facts:
                        knowledge.extend(facts)
            except Exception as e:
                logger.debug(f"Ошибка запроса к графу знаний: {e}")

        step = ReasoningStep(
            phase=ReasoningPhase.KNOWLEDGE_GRAPH_QUERY,
            query=dialogue.original_query,
            result={
                'knowledge_items': knowledge,
                'graph_context': graph_context
            },
            timestamp=time.time(),
            confidence=0.7 if knowledge else 0.3,
            metadata={
                'knowledge_items_count': len(knowledge),
                'fractal_levels': len(graph_context),
                'memory_graph_ml_used': bool(graph_context)
            }
        )
        dialogue.steps.append(step)

        logger.debug(f"Получено {len(knowledge)} элементов знаний из {len(graph_context)} фрактальных уровней")

    def _gather_context(self, dialogue: InternalDialogue) -> Dict[str, Any]:
        context = {
            'query_type': '',
            'entities': [],
            'memory': [],
            'contradictions': [],
            'web_search': [],
            'knowledge_graph': [],
            'graph_context': {},
            'ethics': {},
            'analytics': {},
            'performance': {},
            'patterns': []
        }

        for step in dialogue.steps:
            if step.phase == ReasoningPhase.INITIAL_ANALYSIS:
                if isinstance(step.result, dict):
                    context['query_type'] = step.result.get('query_type', '')
                    context['entities'] = step.result.get('entities', [])
            elif step.phase == ReasoningPhase.MEMORY_RETRIEVAL:
                if isinstance(step.result, dict):
                    context['memory'] = step.result.get('relevant_memories', [])
                elif isinstance(step.result, list):
                    context['memory'] = step.result
            elif step.phase == ReasoningPhase.CONTRADICTION_CHECK:
                if isinstance(step.result, dict):
                    context['contradictions'] = step.result.get('contradictions', [])
                elif isinstance(step.result, list):
                    context['contradictions'] = step.result
            elif step.phase == ReasoningPhase.WEB_SEARCH:
                if isinstance(step.result, dict):
                    context['web_search'] = step.result.get('search_results', [])
                elif isinstance(step.result, list):
                    context['web_search'] = step.result
            elif step.phase == ReasoningPhase.KNOWLEDGE_GRAPH_QUERY:
                if isinstance(step.result, dict):
                    context['knowledge_graph'] = step.result.get('knowledge_items', [])
                    context['graph_context'] = step.result.get('graph_context', {})
                elif isinstance(step.result, list):
                    context['knowledge_graph'] = step.result
            elif step.phase == ReasoningPhase.ETHICS_CHECK:
                if isinstance(step.result, dict):
                    context['ethics'] = step.result
            elif step.phase == ReasoningPhase.ANALYTICS_CHECK:
                if isinstance(step.result, dict):
                    context['analytics'] = step.result
            elif step.phase == ReasoningPhase.PERFORMANCE_ANALYSIS:
                if isinstance(step.result, dict):
                    context['performance'] = step.result

        if self.brain and hasattr(self.brain, 'memory_graph_ml') and self.brain.memory_graph_ml:
            try:
                ml_context = self.brain.memory_graph_ml.get_context_for_query(dialogue.original_query)
                if ml_context:
                    context['ml_entities'] = ml_context.get('entities', [])
                    context['ml_related'] = ml_context.get('related_concepts', [])
            except Exception as e:
                logger.debug(f"Ошибка получения ML контекста: {e}")

        return context
